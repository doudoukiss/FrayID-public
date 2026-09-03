from __future__ import annotations

import os
import random
import subprocess
import sys
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from platform import platform, python_version
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import trimesh
from torch import Tensor, nn

from frayid.camera import make_intrinsics
from frayid.config import ReconstructionConfig, TrainingStageConfig
from frayid.dataset import DATASET_MANIFEST_FILENAME, read_dataset_manifest, validate_dataset
from frayid.geometry import (
    CanonicalSDF,
    ResidualDeformer,
    canonical_face_orientation_report,
    canonical_topology_is_valid,
    canonical_topology_losses,
    deformation_jacobian,
    eikonal_loss,
    extract_sdf_mesh,
    jacobian_regularization,
    linear_blend_skinning,
    rigid_transform_from_axis_angle,
    safe_root_mean_square,
    signed_sdf_mesh_consistency,
    temporal_second_difference,
)
from frayid.initialization import (
    INITIALIZATION_EVALUATION_FILENAME,
    load_initialization,
    validate_initialization_contract,
)
from frayid.io import read_json, sha256_file, write_json
from frayid.renderer import (
    differentiable_boundary_loss,
    normal_cosine_loss,
    render_soft_mesh,
    scaled_splat_parameters,
    silhouette_loss,
)
from frayid.replay_state import (
    CHECKPOINT_SCHEMA_V2,
    CheckpointStateV2,
    SamplerState,
    capture_checkpoint_state,
    configure_deterministic_execution,
    restore_checkpoint_state,
)
from frayid.schemas import (
    EpochMetrics,
    FrameRecord,
    GeometryTrainingReport,
    InitializationEvaluation,
    RunProvenance,
    SequenceInitialization,
    SmokeRunReport,
)


@dataclass
class TrainingEvidence:
    masks: Tensor
    normals: Tensor
    transforms: Tensor
    frame_indices: Tensor
    intrinsics: Tensor
    source_image_size: tuple[int, int]


@dataclass(frozen=True)
class LoadedCheckpoint:
    epoch: int
    global_step: int
    stage: str
    sampler_state: SamplerState | None
    next_step_replay_capable: bool


class TransformArchive(Protocol):
    def __getitem__(self, key: str) -> np.ndarray: ...


class CanonicalGeometryModel(nn.Module):
    def __init__(
        self,
        vertices: Tensor,
        faces: Tensor,
        weights: Tensor,
        frame_count: int,
        config: ReconstructionConfig,
    ) -> None:
        super().__init__()
        self.base_vertices: Tensor
        self.faces: Tensor
        self.skinning_weights: Tensor
        self.register_buffer("base_vertices", vertices)
        self.register_buffer("faces", faces.long())
        self.register_buffer("skinning_weights", weights)
        face_edges = torch.cat(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
            dim=0,
        )
        topology_edges = torch.unique(torch.sort(face_edges.long(), dim=1).values, dim=0)
        self.topology_edges: Tensor
        self.register_buffer("topology_edges", topology_edges, persistent=False)
        self.canonical_offsets = nn.Parameter(torch.zeros_like(vertices))
        self.root_rotation_corrections_raw = nn.Parameter(
            torch.zeros((frame_count, 3), dtype=vertices.dtype, device=vertices.device)
        )
        self.root_translation_corrections_raw = nn.Parameter(
            torch.zeros((frame_count, 3), dtype=vertices.dtype, device=vertices.device)
        )
        self.maximum_root_rotation_correction_radians = float(
            np.deg2rad(config.model.maximum_root_rotation_correction_degrees)
        )
        self.maximum_root_translation_correction_m = (
            config.model.maximum_root_translation_correction_m
        )
        self.sdf = CanonicalSDF(
            hidden_dim=config.model.sdf_hidden_dim,
            layer_count=config.model.sdf_layer_count,
            frequency_count=config.model.positional_encoding_frequencies,
        )
        self.deformer = ResidualDeformer(
            frame_count=frame_count,
            code_dim=config.model.frame_code_dim,
            hidden_dim=config.model.residual_hidden_dim,
            layer_count=config.model.residual_layer_count,
        )

    @property
    def canonical_vertices(self) -> Tensor:
        return self.base_vertices + self.canonical_offsets

    @property
    def root_rotation_corrections(self) -> Tensor:
        return self._bounded_vectors(
            self.root_rotation_corrections_raw,
            self.maximum_root_rotation_correction_radians,
        )

    @property
    def root_translation_corrections(self) -> Tensor:
        return self._bounded_vectors(
            self.root_translation_corrections_raw,
            self.maximum_root_translation_correction_m,
        )

    @staticmethod
    def _bounded_vectors(raw: Tensor, maximum_norm: float) -> Tensor:
        safe_norm = torch.sqrt(raw.square().sum(dim=-1, keepdim=True) + 1e-12)
        scale = maximum_norm * torch.tanh(safe_norm) / safe_norm
        return raw * scale

    def corrected_transforms(self, frame_slot: int, transforms: Tensor) -> Tensor:
        correction = rigid_transform_from_axis_angle(
            self.root_rotation_corrections[frame_slot],
            self.root_translation_corrections[frame_slot],
        )
        return correction.unsqueeze(0) @ transforms

    def posed_vertices(
        self,
        frame_slot: int,
        transforms: Tensor,
        *,
        residual_enabled: bool = True,
    ) -> tuple[Tensor, Tensor]:
        canonical = self.canonical_vertices
        if residual_enabled:
            slot = torch.tensor(frame_slot, device=canonical.device)
            residual = self.deformer(canonical, slot)
        else:
            residual = torch.zeros_like(canonical)
        corrected_transforms = self.corrected_transforms(frame_slot, transforms)
        posed = linear_blend_skinning(
            canonical + residual, self.skinning_weights, corrected_transforms
        )
        return posed, residual

    def posed_vertices_with_code(self, code: Tensor, transforms: Tensor) -> tuple[Tensor, Tensor]:
        canonical = self.canonical_vertices
        residual = self.deformer.forward_with_code(canonical, code)
        posed = linear_blend_skinning(canonical + residual, self.skinning_weights, transforms)
        return posed, residual


def run_geometry_smoke(
    config: ReconstructionConfig,
    *,
    device_name: str | None = None,
    resume_path: Path | None = None,
    config_source_path: Path | None = None,
) -> SmokeRunReport:
    """Run the only authorized reconstruction job (24 frames, 2 epochs)."""
    configure_deterministic_execution()
    dataset_report = validate_dataset(config)
    blockers = list(dataset_report.blockers)
    evaluation_path = config.paths.dataset_root / INITIALIZATION_EVALUATION_FILENAME
    if not evaluation_path.is_file():
        blockers.append("initialization_gate_not_evaluated")
    else:
        initialization_evaluation = InitializationEvaluation.model_validate(
            read_json(evaluation_path)
        )
        if initialization_evaluation.status != "pass":
            blockers.append("initialization_gate_not_passed")
    if blockers:
        report = SmokeRunReport(
            run_id=config.run_id,
            status="blocked",
            frame_count=0,
            epoch_count=0,
            optimizer_steps=0,
            checkpoint_resume_verified=False,
            enabled_losses=_enabled_losses(config),
            gradient_parameter_groups={},
            epoch_metrics=[],
            blockers=sorted(set(blockers)),
        )
        output = config.paths.run_root / config.run_id / "smoke/smoke_report.json"
        write_json(output, report)
        return report

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    _seed_everything(config.seed)
    manifest = read_dataset_manifest(config.paths.dataset_root / DATASET_MANIFEST_FILENAME)
    initialization_path = config.paths.dataset_root / config.evidence.initialization_filename
    initialization = load_initialization(initialization_path)
    init_blockers = validate_initialization_contract(
        initialization, {frame.source_frame_index for frame in manifest.frames}
    )
    if init_blockers:
        raise ValueError("Initialization contract failed: " + ", ".join(init_blockers))
    if not initialization.canonical_mesh_path or not initialization.skinning_weights_path:
        raise ValueError("Initialization must provide canonical mesh and SMPL skinning weights")
    if not initialization.joint_transforms_path:
        raise ValueError("Initialization must provide real per-frame SMPL joint transforms")
    mesh = np.load(initialization.canonical_mesh_path)
    weight_data = np.load(initialization.skinning_weights_path)
    transform_data = np.load(initialization.joint_transforms_path)
    train_records = [frame for frame in manifest.frames if frame.split == "train"]
    selected = _distributed(train_records, config.smoke.frame_count)
    evidence = _load_evidence(
        config,
        selected,
        initialization,
        transform_data,
        device,
        resolution=config.training.stages[0].mask_resolution,
    )
    model = CanonicalGeometryModel(
        torch.tensor(mesh["vertices"], dtype=torch.float32, device=device),
        torch.tensor(mesh["faces"], dtype=torch.long, device=device),
        torch.tensor(weight_data["weights"], dtype=torch.float32, device=device),
        len(selected),
        config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.smoke.learning_rate)
    start_epoch = 0
    resume_sampler: SamplerState | None = None
    optimizer_steps = 0
    if resume_path is not None:
        loaded = _load_checkpoint_runtime(resume_path, model, optimizer, device)
        optimizer_steps = loaded.global_step
        if loaded.sampler_state is not None and not loaded.sampler_state.complete:
            start_epoch = loaded.epoch
            resume_sampler = loaded.sampler_state
        else:
            start_epoch = loaded.epoch + 1
    _assert_canonical_topology(model, config, "smoke_start")
    run_root = config.paths.run_root / config.run_id / "smoke"
    run_root.mkdir(parents=True, exist_ok=True)
    initial_parameters = _parameter_vector(model).detach().clone()
    initial_fixed_objective = _evaluate_fixed_objective(model, evidence, config)
    epoch_metrics: list[EpochMetrics] = []
    gradient_groups = {
        "sdf": False,
        "canonical_mesh": False,
        "residual_deformation": False,
        "root_correction": False,
    }
    for epoch in range(start_epoch, start_epoch + config.smoke.epoch_count):
        stage = training_stage_for_epoch(config, epoch)
        if stage.name != "coarse":
            raise RuntimeError("The authorized smoke may only execute the coarse stage")
        metrics, step_count, observed_gradients, sampler_state = _train_epoch_resumable(
            model,
            optimizer,
            evidence,
            config,
            epoch,
            sampler_state=resume_sampler if epoch == start_epoch else None,
        )
        optimizer_steps += step_count
        for name, observed in observed_gradients.items():
            gradient_groups[name] = gradient_groups[name] or observed
        parameter_delta = float(
            torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters)
        )
        epoch_metrics.append(
            EpochMetrics(
                epoch=epoch,
                stage=stage.name,
                total=metrics["total"],
                losses={key: value for key, value in metrics.items() if key != "total"},
                parameter_delta_norm=parameter_delta,
            )
        )
        checkpoint_path = run_root / f"checkpoint_epoch_{epoch:04d}.pt"
        _save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            config,
            global_step=optimizer_steps,
            stage=stage.name,
            sampler_state=sampler_state,
        )
        if (epoch + 1) % config.training.refresh_mesh_every_epochs == 0:
            _write_sdf_mesh_snapshot(model, stage, epoch, run_root, device)

    final_checkpoint = run_root / f"checkpoint_epoch_{epoch_metrics[-1].epoch:04d}.pt"
    checkpoint_resume_verified = _verify_checkpoint_resume(final_checkpoint, model, config, device)
    final_fixed_objective = _evaluate_fixed_objective(model, evidence, config)
    initial_loss = initial_fixed_objective
    final_loss = final_fixed_objective
    reduction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-8)
    enabled_losses = _enabled_losses(config)
    nonzero_losses = all(
        all(
            name in item.losses and np.isfinite(item.losses[name]) and item.losses[name] > 0
            for name in enabled_losses
        )
        for item in epoch_metrics
    )
    parameters_changed = epoch_metrics[-1].parameter_delta_norm > 0
    smoke_blockers: list[str] = []
    if not nonzero_losses:
        smoke_blockers.append("enabled_loss_nonfinite_or_zero")
    if not all(gradient_groups.values()):
        smoke_blockers.append("gradient_missing_from_parameter_group")
    if not parameters_changed:
        smoke_blockers.append("trainable_parameters_did_not_change")
    if not checkpoint_resume_verified:
        smoke_blockers.append("checkpoint_resume_verification_failed")
    if reduction < config.smoke.required_loss_reduction_fraction:
        smoke_blockers.append("geometry_loss_reduction_below_gate")

    topology_report_path = run_root / "canonical_topology_report.json"
    topology_report = _write_canonical_topology_report(model, config, topology_report_path)
    smoke_blockers.extend(topology_report["blockers"])

    provenance = _write_provenance(
        config,
        selected,
        optimizer_steps,
        final_checkpoint,
        initialization_path,
        run_root,
        config_source_path or Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
        device,
    )
    report = SmokeRunReport(
        run_id=config.run_id,
        status="pass" if not smoke_blockers else "fail",
        frame_count=len(selected),
        epoch_count=len(epoch_metrics),
        optimizer_steps=optimizer_steps,
        checkpoint_resume_verified=checkpoint_resume_verified,
        enabled_losses=enabled_losses,
        gradient_parameter_groups=gradient_groups,
        epoch_metrics=epoch_metrics,
        initial_fixed_objective=initial_fixed_objective,
        final_fixed_objective=final_fixed_objective,
        loss_reduction_fraction=reduction,
        checkpoint_path=str(final_checkpoint),
        provenance_path=str(provenance),
        topology_report_path=str(topology_report_path),
        blockers=smoke_blockers,
    )
    write_json(run_root / "smoke_report.json", report)
    return report


def run_geometry_training(
    config: ReconstructionConfig,
    *,
    device_name: str | None = None,
    resume_path: Path | None = None,
    config_source_path: Path | None = None,
) -> GeometryTrainingReport:
    """Run authorized coarse-to-fine training after every fixed gate passes."""
    configure_deterministic_execution()
    blockers: list[str] = []
    if not config.smoke.full_training_authorized:
        blockers.append("full_training_not_authorized")
    dataset_report = validate_dataset(config)
    blockers.extend(dataset_report.blockers)
    initialization_evaluation_path = config.paths.dataset_root / INITIALIZATION_EVALUATION_FILENAME
    if not initialization_evaluation_path.is_file():
        blockers.append("initialization_gate_not_evaluated")
    else:
        initialization_evaluation = InitializationEvaluation.model_validate(
            read_json(initialization_evaluation_path)
        )
        if initialization_evaluation.status != "pass":
            blockers.append("initialization_gate_not_passed")
    smoke_report_path = config.paths.run_root / config.run_id / "smoke/smoke_report.json"
    if not smoke_report_path.is_file():
        blockers.append("smoke_gate_not_evaluated")
    else:
        smoke_report = SmokeRunReport.model_validate(read_json(smoke_report_path))
        if smoke_report.status != "pass":
            blockers.append("smoke_gate_not_passed")
    run_root = config.paths.run_root / config.run_id / "full"
    if blockers:
        report = GeometryTrainingReport(
            run_id=config.run_id,
            status="blocked",
            frame_count=0,
            epoch_count=0,
            optimizer_steps=0,
            epoch_metrics=[],
            blockers=sorted(set(blockers)),
        )
        write_json(run_root / "training_report.json", report)
        return report

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    _seed_everything(config.seed)
    manifest = read_dataset_manifest(config.paths.dataset_root / DATASET_MANIFEST_FILENAME)
    train_records = [frame for frame in manifest.frames if frame.split == "train"]
    if len(train_records) < config.dataset.minimum_usable_frame_count:
        raise ValueError("Full training has fewer than the minimum usable training frames")
    initialization_path = config.paths.dataset_root / config.evidence.initialization_filename
    initialization = load_initialization(initialization_path)
    init_blockers = validate_initialization_contract(
        initialization, {frame.source_frame_index for frame in manifest.frames}
    )
    if init_blockers:
        raise ValueError("Initialization contract failed: " + ", ".join(init_blockers))
    if not initialization.canonical_mesh_path or not initialization.skinning_weights_path:
        raise ValueError("Initialization must provide canonical mesh and SMPL skinning weights")
    if not initialization.joint_transforms_path:
        raise ValueError("Initialization must provide real per-frame SMPL joint transforms")
    mesh = np.load(initialization.canonical_mesh_path)
    weight_data = np.load(initialization.skinning_weights_path)
    transform_data = np.load(initialization.joint_transforms_path)
    model = CanonicalGeometryModel(
        torch.tensor(mesh["vertices"], dtype=torch.float32, device=device),
        torch.tensor(mesh["faces"], dtype=torch.long, device=device),
        torch.tensor(weight_data["weights"], dtype=torch.float32, device=device),
        len(train_records),
        config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    start_epoch = 0
    resume_sampler: SamplerState | None = None
    optimizer_steps = 0
    if resume_path is not None:
        loaded = _load_checkpoint_runtime(resume_path, model, optimizer, device)
        optimizer_steps = loaded.global_step
        if loaded.sampler_state is not None and not loaded.sampler_state.complete:
            start_epoch = loaded.epoch
            resume_sampler = loaded.sampler_state
        else:
            start_epoch = loaded.epoch + 1
    _assert_canonical_topology(model, config, "full_training_start")
    if start_epoch >= config.training.full_epoch_count:
        raise ValueError("Resume checkpoint is already at or beyond the configured schedule")

    run_root.mkdir(parents=True, exist_ok=True)
    initial_parameters = _parameter_vector(model).detach().clone()
    epoch_metrics: list[EpochMetrics] = []
    evidence_by_stage: dict[str, TrainingEvidence] = {}
    previous_stage_name: str | None = None
    latest_sdf_mesh_path: Path | None = None
    for epoch in range(start_epoch, config.training.full_epoch_count):
        stage = training_stage_for_epoch(config, epoch)
        if stage.name not in evidence_by_stage:
            evidence_by_stage[stage.name] = _load_evidence(
                config,
                train_records,
                initialization,
                transform_data,
                device,
                resolution=stage.mask_resolution,
            )
        metrics, step_count, _, sampler_state = _train_epoch_resumable(
            model,
            optimizer,
            evidence_by_stage[stage.name],
            config,
            epoch,
            residual_enabled=stage.name != "coarse",
            sampler_state=resume_sampler if epoch == start_epoch else None,
        )
        optimizer_steps += step_count
        epoch_metrics.append(
            EpochMetrics(
                epoch=epoch,
                stage=stage.name,
                total=metrics["total"],
                losses={key: value for key, value in metrics.items() if key != "total"},
                parameter_delta_norm=float(
                    torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters)
                ),
            )
        )
        if (
            (epoch + 1) % config.training.checkpoint_every_epochs == 0
            or epoch + 1 == config.training.full_epoch_count
        ):
            _save_checkpoint(
                run_root / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                config,
                global_step=optimizer_steps,
                stage=stage.name,
                sampler_state=sampler_state,
            )
        stage_changed = previous_stage_name != stage.name
        periodic_refresh = (epoch + 1) % config.training.refresh_mesh_every_epochs == 0
        if stage_changed or periodic_refresh or epoch + 1 == config.training.full_epoch_count:
            latest_sdf_mesh_path = _write_sdf_mesh_snapshot(model, stage, epoch, run_root, device)
        previous_stage_name = stage.name

    final_epoch = epoch_metrics[-1].epoch
    final_checkpoint = run_root / f"checkpoint_epoch_{final_epoch:04d}.pt"
    if not final_checkpoint.is_file():
        _save_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            final_epoch,
            config,
            global_step=optimizer_steps,
            stage=training_stage_for_epoch(config, final_epoch).name,
            sampler_state=sampler_state,
        )
    if not _verify_checkpoint_resume(final_checkpoint, model, config, device):
        raise RuntimeError("Final checkpoint resume verification failed")
    explicit_mesh_path = run_root / "canonical_explicit_mesh.ply"
    trimesh.Trimesh(
        vertices=model.canonical_vertices.detach().cpu().numpy(),
        faces=model.faces.detach().cpu().numpy(),
        process=False,
    ).export(explicit_mesh_path)
    topology_report_path = run_root / "canonical_topology_report.json"
    topology_report = _write_canonical_topology_report(model, config, topology_report_path)
    provenance_path = _write_provenance(
        config,
        train_records,
        optimizer_steps,
        final_checkpoint,
        initialization_path,
        run_root,
        config_source_path or Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
        device,
    )
    report = GeometryTrainingReport(
        run_id=config.run_id,
        status="complete" if topology_report["status"] == "pass" else "fail",
        frame_count=len(train_records),
        epoch_count=len(epoch_metrics),
        optimizer_steps=optimizer_steps,
        epoch_metrics=epoch_metrics,
        checkpoint_path=str(final_checkpoint),
        provenance_path=str(provenance_path),
        explicit_mesh_path=str(explicit_mesh_path),
        sdf_mesh_path=str(latest_sdf_mesh_path) if latest_sdf_mesh_path else None,
        topology_report_path=str(topology_report_path),
        blockers=list(topology_report["blockers"]),
    )
    write_json(run_root / "training_report.json", report)
    return report


def _train_epoch_resumable(
    model: CanonicalGeometryModel,
    optimizer: torch.optim.Optimizer,
    evidence: TrainingEvidence,
    config: ReconstructionConfig,
    epoch: int,
    *,
    residual_enabled: bool = True,
    sampler_state: SamplerState | None = None,
    maximum_steps: int | None = None,
) -> tuple[dict[str, float], int, dict[str, bool], SamplerState]:
    del epoch
    accumulated = {name: 0.0 for name in [*_enabled_losses(config), "total"]}
    gradients = {
        "sdf": False,
        "canonical_mesh": False,
        "residual_deformation": False,
        "root_correction": False,
    }
    if sampler_state is None:
        order = torch.randperm(evidence.masks.shape[0], device=evidence.masks.device).cpu().tolist()
        sampler_state = SamplerState([int(value) for value in order])
    sampler_state.validate(item_count=int(evidence.masks.shape[0]))
    completed_steps = 0
    while not sampler_state.complete:
        if maximum_steps is not None and completed_steps >= maximum_steps:
            break
        slot = sampler_state.take()
        optimizer.zero_grad(set_to_none=True)
        loss_values = _loss_values_for_slot(
            model, evidence, config, slot, residual_enabled=residual_enabled
        )
        total = sum(getattr(config.losses, name) * value for name, value in loss_values.items())
        if not torch.isfinite(total):
            raise RuntimeError("Training produced a non-finite geometry loss")
        total.backward()
        gradients["sdf"] = gradients["sdf"] or _has_gradient(model.sdf.parameters())
        gradients["canonical_mesh"] = (
            gradients["canonical_mesh"] or model.canonical_offsets.grad is not None
        )
        gradients["residual_deformation"] = gradients["residual_deformation"] or _has_gradient(
            model.deformer.parameters()
        )
        gradients["root_correction"] = gradients["root_correction"] or _has_gradient(
            (model.root_rotation_corrections_raw, model.root_translation_corrections_raw)
        )
        previous_offsets = model.canonical_offsets.detach().clone()
        optimizer.step()
        _backtrack_canonical_update(model, previous_offsets, optimizer, config)
        for name, value in loss_values.items():
            accumulated[name] += float(value.detach().cpu())
        accumulated["total"] += float(total.detach().cpu())
        completed_steps += 1
    count = max(completed_steps, 1)
    return (
        {name: value / count for name, value in accumulated.items()},
        completed_steps,
        gradients,
        sampler_state,
    )


def _train_epoch(
    model: CanonicalGeometryModel,
    optimizer: torch.optim.Optimizer,
    evidence: TrainingEvidence,
    config: ReconstructionConfig,
    epoch: int,
    *,
    residual_enabled: bool = True,
) -> tuple[dict[str, float], int, dict[str, bool]]:
    """Compatibility wrapper for closed runners that only checkpoint at epoch boundaries."""
    metrics, steps, gradients, _ = _train_epoch_resumable(
        model,
        optimizer,
        evidence,
        config,
        epoch,
        residual_enabled=residual_enabled,
    )
    return metrics, steps, gradients


def _evaluate_fixed_objective(
    model: CanonicalGeometryModel,
    evidence: TrainingEvidence,
    config: ReconstructionConfig,
) -> float:
    """Evaluate identical slots and samples without perturbing the training RNG."""
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    totals: list[float] = []
    with torch.random.fork_rng(devices=cuda_devices):
        for slot in range(int(evidence.masks.shape[0])):
            torch.manual_seed(config.seed + 100_000 + slot)
            loss_values = _loss_values_for_slot(model, evidence, config, slot)
            total = sum(getattr(config.losses, name) * value for name, value in loss_values.items())
            if not torch.isfinite(total):
                raise RuntimeError("Fixed smoke objective produced a non-finite loss")
            totals.append(float(total.detach().cpu()))
    return float(np.mean(totals))


def _loss_values_for_slot(
    model: CanonicalGeometryModel,
    evidence: TrainingEvidence,
    config: ReconstructionConfig,
    slot: int,
    *,
    residual_enabled: bool = True,
) -> dict[str, Tensor]:
    posed, residual = model.posed_vertices(
        slot, evidence.transforms[slot], residual_enabled=residual_enabled
    )
    image_size = (int(evidence.masks.shape[-2]), int(evidence.masks.shape[-1]))
    silhouette, normals = render_soft_mesh(
        posed,
        model.faces,
        evidence.intrinsics,
        image_size,
        source_image_size=evidence.source_image_size,
        sigma_pixels=_renderer_sigma_pixels(config, image_size),
        sample_count=_renderer_sample_count(config, image_size),
        reference_sample_count=_renderer_reference_sample_count(config, image_size),
        depth_temperature_m=config.model.renderer_depth_temperature_m,
    )
    canonical = model.canonical_vertices
    sample_count = min(config.model.eikonal_sample_count, max(128, canonical.shape[0]))
    random_points = (
        torch.rand((sample_count, 3), dtype=canonical.dtype, device=canonical.device) * 2.4 - 1.2
    )
    jacobian_points = canonical[:: max(1, canonical.shape[0] // 64)].detach().requires_grad_(True)
    jacobian_displacement = model.deformer(
        jacobian_points, torch.tensor(slot, device=canonical.device)
    )
    topology_losses = canonical_topology_losses(
        model.base_vertices,
        canonical,
        model.faces,
        model.topology_edges,
        model.canonical_offsets,
        orientation_margin=config.model.canonical_orientation_margin,
        minimum_area_ratio=config.model.canonical_minimum_area_ratio,
    )
    root_rotation = model.root_rotation_corrections
    root_translation = model.root_translation_corrections
    return {
        "silhouette": silhouette_loss(silhouette, evidence.masks[slot]),
        "boundary": differentiable_boundary_loss(silhouette, evidence.masks[slot]),
        "normal": normal_cosine_loss(normals, evidence.normals[slot], evidence.masks[slot]),
        "eikonal": eikonal_loss(model.sdf, random_points),
        "sdf_mesh_consistency": signed_sdf_mesh_consistency(
            model.sdf,
            canonical,
            model.faces,
            maximum_sample_count=config.model.eikonal_sample_count,
        ),
        "deformation": safe_root_mean_square(residual),
        "jacobian": jacobian_regularization(
            deformation_jacobian(jacobian_displacement, jacobian_points)
        ).clamp_min(1e-12),
        "temporal": temporal_second_difference(model.deformer.frame_codes.weight).clamp_min(1e-12),
        "root_rotation_correction": (root_rotation[slot].square().mean() + 1e-12).sqrt(),
        "root_translation_correction": (root_translation[slot].square().mean() + 1e-12).sqrt(),
        "root_correction_temporal": (
            temporal_second_difference(root_rotation) + temporal_second_difference(root_translation)
        ).clamp_min(1e-12),
        **topology_losses,
    }


def _renderer_sample_count(config: ReconstructionConfig, image_size: tuple[int, int]) -> int:
    return scaled_splat_parameters(
        image_size,
        reference_resolution=config.model.renderer_resolution,
        reference_sigma_pixels=config.model.renderer_sigma_pixels,
        reference_sample_count=config.model.renderer_max_vertices,
    )[1]


def _renderer_sigma_pixels(config: ReconstructionConfig, image_size: tuple[int, int]) -> float:
    return scaled_splat_parameters(
        image_size,
        reference_resolution=config.model.renderer_resolution,
        reference_sigma_pixels=config.model.renderer_sigma_pixels,
        reference_sample_count=config.model.renderer_max_vertices,
    )[0]


def _renderer_reference_sample_count(
    config: ReconstructionConfig, image_size: tuple[int, int]
) -> int:
    return scaled_splat_parameters(
        image_size,
        reference_resolution=config.model.renderer_resolution,
        reference_sigma_pixels=config.model.renderer_sigma_pixels,
        reference_sample_count=config.model.renderer_reference_sample_count,
    )[1]


def training_stage_for_epoch(config: ReconstructionConfig, epoch: int) -> TrainingStageConfig:
    if epoch < 0 or epoch >= config.training.full_epoch_count:
        raise ValueError("epoch is outside the configured full training schedule")
    active = config.training.stages[0]
    for stage in config.training.stages:
        if stage.start_epoch <= epoch:
            active = stage
    return active


def _write_sdf_mesh_snapshot(
    model: CanonicalGeometryModel,
    stage: TrainingStageConfig,
    epoch: int,
    run_root: Path,
    device: torch.device,
) -> Path:
    """Periodically regenerate the explicit surface from the current SDF."""
    canonical = model.canonical_vertices.detach()
    margin = 0.15
    low = tuple((canonical.amin(dim=0) - margin).cpu().tolist())
    high = tuple((canonical.amax(dim=0) + margin).cpu().tolist())
    extracted = extract_sdf_mesh(
        model.sdf,
        resolution=stage.grid_resolution,
        bounds=(low, high),
        device=device,
    )
    path = run_root / f"sdf_mesh_epoch_{epoch:04d}_{stage.name}.ply"
    trimesh.Trimesh(
        vertices=extracted.vertices,
        faces=extracted.faces,
        vertex_normals=extracted.normals,
        process=False,
    ).export(path)
    return path


def _load_evidence(
    config: ReconstructionConfig,
    records: Sequence[FrameRecord],
    initialization: SequenceInitialization,
    transform_data: TransformArchive,
    device: torch.device,
    *,
    resolution: int | None = None,
) -> TrainingEvidence:
    dataset_root = config.paths.dataset_root
    masks: list[Tensor] = []
    normals: list[Tensor] = []
    source_indices: list[int] = []
    init_frames = {frame.source_frame_index: frame for frame in initialization.frames}
    transform_indices = transform_data["source_frame_indices"].tolist()
    transform_lookup = {int(value): index for index, value in enumerate(transform_indices)}
    transforms: list[Tensor] = []
    for record in records:
        name = Path(record.image_path).name
        mask = cv2.imread(
            str(dataset_root / config.evidence.masks_subdirectory / name), cv2.IMREAD_GRAYSCALE
        )
        normal = cv2.imread(
            str(dataset_root / config.evidence.normals_subdirectory / name), cv2.IMREAD_COLOR
        )
        if mask is None or normal is None:
            raise FileNotFoundError(f"Missing validated evidence for {name}")
        render_resolution = resolution or config.model.renderer_resolution
        size = (render_resolution, render_resolution)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
        normal = cv2.resize(normal, size, interpolation=cv2.INTER_LINEAR)
        masks.append(torch.tensor(mask / 255.0, dtype=torch.float32, device=device))
        decoded = torch.tensor(
            normal[..., ::-1].copy() / 127.5 - 1.0, dtype=torch.float32, device=device
        )
        normals.append(torch.nn.functional.normalize(decoded, dim=-1, eps=1e-8))
        source_index = int(record.source_frame_index)
        if source_index not in init_frames or source_index not in transform_lookup:
            raise ValueError(f"Missing initialization transform for source frame {source_index}")
        transforms.append(
            torch.tensor(
                transform_data["transforms"][transform_lookup[source_index]],
                dtype=torch.float32,
                device=device,
            )
        )
        source_indices.append(source_index)
    return TrainingEvidence(
        masks=torch.stack(masks),
        normals=torch.stack(normals),
        transforms=torch.stack(transforms),
        frame_indices=torch.tensor(source_indices, dtype=torch.long, device=device),
        intrinsics=make_intrinsics(
            initialization.shared_focal_length_px,
            initialization.shared_principal_point_px,
            device=device,
        ),
        source_image_size=(initialization.image_height, initialization.image_width),
    )


def _save_checkpoint(
    path: Path,
    model: CanonicalGeometryModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: ReconstructionConfig,
    *,
    global_step: int = 0,
    stage: str = "unknown",
    sampler_state: SamplerState | None = None,
) -> None:
    _assert_canonical_topology(model, config, "checkpoint_write")
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = capture_checkpoint_state(
        model,
        optimizer,
        epoch=epoch,
        global_step=global_step,
        stage=stage,
        sampler_state=sampler_state or SamplerState([], 0),
    )
    payload = runtime.state_dict()
    payload["config"] = config.model_dump(mode="json")
    torch.save(payload, path)


def _backtrack_canonical_update(
    model: CanonicalGeometryModel,
    previous_offsets: Tensor,
    optimizer: torch.optim.Optimizer,
    config: ReconstructionConfig,
) -> float:
    """Project an Adam step onto the topology-safe segment from the last valid mesh."""
    candidate_offsets = model.canonical_offsets.detach().clone()
    accepted_scale = 1.0
    for _ in range(config.model.canonical_backtracking_steps + 1):
        with torch.no_grad():
            model.canonical_offsets.copy_(
                previous_offsets + accepted_scale * (candidate_offsets - previous_offsets)
            )
        if canonical_topology_is_valid(
            model.base_vertices,
            model.canonical_vertices,
            model.faces,
            minimum_signed_area_ratio=config.model.canonical_minimum_signed_area_ratio,
            minimum_area_ratio=config.model.canonical_minimum_area_ratio,
        ):
            break
        accepted_scale *= 0.5
    else:
        accepted_scale = 0.0
        with torch.no_grad():
            model.canonical_offsets.copy_(previous_offsets)

    if accepted_scale < 1.0:
        state = optimizer.state.get(model.canonical_offsets, {})
        first_moment = state.get("exp_avg")
        second_moment = state.get("exp_avg_sq")
        if isinstance(first_moment, Tensor):
            first_moment.mul_(accepted_scale)
        if isinstance(second_moment, Tensor):
            second_moment.mul_(accepted_scale * accepted_scale)
    return accepted_scale


def _assert_canonical_topology(
    model: CanonicalGeometryModel,
    config: ReconstructionConfig,
    context: str,
) -> None:
    if not canonical_topology_is_valid(
        model.base_vertices,
        model.canonical_vertices,
        model.faces,
        minimum_signed_area_ratio=config.model.canonical_minimum_signed_area_ratio,
        minimum_area_ratio=config.model.canonical_minimum_area_ratio,
    ):
        raise RuntimeError(f"Canonical topology gate failed during {context}")


def _write_canonical_topology_report(
    model: CanonicalGeometryModel,
    config: ReconstructionConfig,
    path: Path,
) -> dict[str, Any]:
    report = canonical_face_orientation_report(
        model.base_vertices.detach().cpu().numpy(),
        model.canonical_vertices.detach().cpu().numpy(),
        model.faces.detach().cpu().numpy(),
        minimum_area_ratio=config.model.canonical_minimum_area_ratio,
    )
    write_json(path, report)
    return report


def _load_checkpoint(
    path: Path,
    model: CanonicalGeometryModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    return _load_checkpoint_runtime(path, model, optimizer, device).epoch


def _load_checkpoint_runtime(
    path: Path,
    model: CanonicalGeometryModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> LoadedCheckpoint:
    payload = torch.load(path, map_location=device, weights_only=False)
    schema = payload.get("schema_version")
    if schema == CHECKPOINT_SCHEMA_V2:
        state = CheckpointStateV2.from_state_dict(payload)
        sampler_state = restore_checkpoint_state(state, model, optimizer)
        result = LoadedCheckpoint(
            epoch=state.epoch,
            global_step=state.global_step,
            stage=state.stage,
            sampler_state=sampler_state,
            next_step_replay_capable=True,
        )
    elif schema == "canonical_checkpoint.v1":
        load_canonical_model_state(model, payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        warnings.warn(
            "canonical_checkpoint.v1 loaded read-only; it cannot prove exact next-step replay",
            RuntimeWarning,
            stacklevel=2,
        )
        result = LoadedCheckpoint(
            epoch=int(payload["epoch"]),
            global_step=0,
            stage="legacy_v1_unknown",
            sampler_state=None,
            next_step_replay_capable=False,
        )
    else:
        raise ValueError("Unsupported checkpoint schema")
    return result


def load_canonical_model_state(
    model: CanonicalGeometryModel,
    state: dict[str, Tensor],
) -> None:
    """Load checkpoints before or after bounded root corrections were added."""
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "root_rotation_corrections_raw",
        "root_translation_corrections_raw",
    }
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    if unexpected or missing - allowed_missing:
        raise ValueError(
            "Checkpoint state is incompatible: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _verify_checkpoint_resume(
    path: Path,
    source: CanonicalGeometryModel,
    config: ReconstructionConfig,
    device: torch.device,
) -> bool:
    clone = CanonicalGeometryModel(
        source.base_vertices.detach().clone(),
        source.faces.detach().clone(),
        source.skinning_weights.detach().clone(),
        source.deformer.frame_codes.num_embeddings,
        config,
    ).to(device)
    optimizer = torch.optim.Adam(clone.parameters(), lr=config.training.learning_rate)
    _load_checkpoint(path, clone, optimizer, device)
    return all(
        torch.equal(first, second)
        for first, second in zip(
            source.state_dict().values(), clone.state_dict().values(), strict=True
        )
    )


def _write_provenance(
    config: ReconstructionConfig,
    records: Sequence[FrameRecord],
    optimizer_steps: int,
    checkpoint_path: Path,
    initialization_path: Path,
    run_root: Path,
    config_path: Path,
    device: torch.device,
) -> Path:
    manifest_path = config.paths.dataset_root / DATASET_MANIFEST_FILENAME
    commit = os.environ.get("FRAYID_GIT_COMMIT", "").strip()
    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown"
    provenance = RunProvenance(
        run_id=config.run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        git_commit=commit,
        command=sys.argv,
        environment={
            "platform": platform(),
            "python": python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": str(torch.cuda.is_available()),
        },
        input_video_sha256=sha256_file(config.paths.input_video),
        asset_manifest_sha256=sha256_file(config.paths.asset_manifest),
        config_sha256=sha256_file(config_path),
        dataset_manifest_sha256=sha256_file(manifest_path),
        initialization_sha256=sha256_file(initialization_path),
        seed=config.seed,
        frame_indices=[int(record.source_frame_index) for record in records],
        enabled_losses={
            name: float(getattr(config.losses, name)) for name in _enabled_losses(config)
        },
        optimizer_steps=optimizer_steps,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
    )
    path = run_root / "provenance.json"
    write_json(path, provenance)
    return path


def _parameter_vector(model: nn.Module) -> Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def _has_gradient(parameters: Iterable[nn.Parameter]) -> bool:
    return any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def _enabled_losses(config: ReconstructionConfig) -> list[str]:
    return [name for name, value in config.losses.model_dump().items() if value > 0]


def _distributed(values: Sequence[FrameRecord], count: int) -> list[FrameRecord]:
    if len(values) < count:
        raise ValueError(f"Smoke requires {count} training frames, found {len(values)}")
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return [values[index] for index in positions.tolist()]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
