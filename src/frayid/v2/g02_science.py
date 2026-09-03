from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt  # type: ignore[import-untyped]
from torch import Tensor, nn

from frayid.io import read_json, sha256_file, write_json
from frayid.normal_integrable_sdf import render_neus_sdf, trilinear_grid_sample
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.evidence import EvidenceVolume
from frayid.v2.field import V2NeuralSDF
from frayid.v2.g02_shortcut_resistant import (
    G02_EXPERIMENT_ID,
    _output_layer,
    prepare_shortcut_resistant_field,
)
from frayid.v2.q03_interval_tracks import load_interval_material_track_graph
from frayid.v2.t05_fixed_camera import FixedCameraHumanSolution

G02_SCIENCE_EVIDENCE_SCHEMA = "frayid_v2_g02_science_training_evidence.v1"
G02_SCIENCE_ARM_SCHEMA = "frayid_v2_g02_science_arm_binding.v1"
G02_SCIENCE_CHECKPOINT_SCHEMA = "frayid_v2_g02_science_checkpoint.v1"
G02_SCIENCE_REPORT_SCHEMA = "frayid_v2_g02_science_training_report.v1"
G02_SCIENCE_EVALUATOR = "g02_independent_endpoint_evaluator.v1"
G02_SCIENCE_EXTRACTOR = "pinned_flexicubes_search_then_exact_cpu_commit_audit.v1"
SCIENCE_STAGE_SEQUENCE = (
    "mask_profile_free_space_eikonal",
    "confidence_normal_and_deformation",
    "q03_tracks_visibility_and_strain",
)
RAY_STRATA = ("foreground", "profile_contour", "free_space", "semantic_boundary")


@dataclass(frozen=True)
class G02ScienceSchedule:
    stage_steps: tuple[int, int, int] = (240, 240, 120)
    ray_batch_size: int = 128
    volume_batch_size: int = 128
    track_batch_size: int = 48
    coarse_samples: int = 24
    hierarchical_samples: int = 8
    learning_rate: float = 2.0e-4
    eikonal_weight: float = 0.04
    evidence_sdf_weight: float = 0.08
    silhouette_weight: float = 1.0
    profile_weight: float = 0.35
    free_space_weight: float = 0.20
    normal_weight: float = 0.18
    track_weight: float = 0.25
    maximum_gradient_norm: float = 2.0
    checkpoint_interval: int = 120
    endpoint_evaluation_rays: int = 512
    seed: int = 20260903

    def __post_init__(self) -> None:
        if len(self.stage_steps) != len(SCIENCE_STAGE_SEQUENCE) or any(
            step < 1 for step in self.stage_steps
        ):
            raise ValueError("G02 science requires one positive step count per stage")
        counts = (
            self.ray_batch_size,
            self.volume_batch_size,
            self.track_batch_size,
            self.coarse_samples,
            self.checkpoint_interval,
            self.endpoint_evaluation_rays,
        )
        if any(value < 1 for value in counts) or self.hierarchical_samples < 0:
            raise ValueError("G02 science schedule counts are invalid")
        weights = (
            self.learning_rate,
            self.eikonal_weight,
            self.evidence_sdf_weight,
            self.silhouette_weight,
            self.profile_weight,
            self.free_space_weight,
            self.normal_weight,
            self.track_weight,
            self.maximum_gradient_norm,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("G02 science schedule weights must be positive and finite")

    @property
    def optimizer_steps(self) -> int:
        return sum(self.stage_steps)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stage_steps"] = list(self.stage_steps)
        value["stage_names"] = list(SCIENCE_STAGE_SEQUENCE)
        value["optimizer_steps"] = self.optimizer_steps
        value["bounded_rgb_stage"] = "disabled_until_geometry_endpoint_passes"
        return value


def g02_science_preflight_schedule() -> G02ScienceSchedule:
    return G02ScienceSchedule(
        stage_steps=(1, 1, 1),
        ray_batch_size=8,
        volume_batch_size=16,
        track_batch_size=8,
        coarse_samples=8,
        hierarchical_samples=0,
        learning_rate=1.0e-6,
        checkpoint_interval=1,
        endpoint_evaluation_rays=32,
    )


def g02_science_target_preflight_schedule() -> G02ScienceSchedule:
    """Exercise production tensor shapes without consuming the full step budget."""

    production = G02ScienceSchedule()
    values = asdict(production)
    values.update(stage_steps=(1, 1, 1), checkpoint_interval=1)
    return G02ScienceSchedule(**values)


@dataclass(frozen=True)
class G02ScienceEvidence:
    ray_pixels: Tensor
    ray_targets: Tensor
    ray_normals: Tensor
    ray_confidence: Tensor
    ray_profile_distance: Tensor
    ray_stratum_codes: Tensor
    intrinsics: Tensor
    rotations: Tensor
    translations: Tensor
    source_frame_indices: Tensor
    motion_uncertainty: Tensor
    track_anchors: Tensor
    track_weights: Tensor
    track_covariance_trace: Tensor
    track_view_slots: Tensor
    source_hashes: dict[str, str]

    @property
    def ray_count(self) -> int:
        return int(self.ray_pixels.shape[0])

    @property
    def view_count(self) -> int:
        return int(self.rotations.shape[0])


def _sample_without_replacement(mask: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    candidates = np.argwhere(mask)
    if not len(candidates):
        raise ValueError("G02 science ray stratum is empty")
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(candidates), size=count, replace=len(candidates) < count)
    return candidates[selected]


def _normal_for_source(root: Path, source_index: int, shape: tuple[int, int]) -> np.ndarray:
    paths = sorted(root.glob(f"*source_{source_index:06d}.png"))
    if len(paths) != 1:
        raise ValueError(f"G02 science expected one normal map for source {source_index}")
    image = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"G02 science could not read normal map for source {source_index}")
    resized = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return resized[..., ::-1].copy().astype(np.float32) / 127.5 - 1.0


def prepare_g02_science_training_evidence(
    hull_binding_path: Path,
    normal_root: Path,
    t05_solution_path: Path,
    q03_binding_path: Path,
    output_path: Path,
    *,
    rays_per_view_stratum: int = 16,
    seed: int = 20260903,
) -> Path:
    """Freeze train-only rays/normals/tracks before a scientific attempt.

    The resulting archive intentionally contains no development frame, score,
    threshold outcome, RGB image, or sealed capability.
    """

    paths = [
        hull_binding_path,
        normal_root,
        t05_solution_path,
        q03_binding_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G02 science training evidence is immutable")
    if rays_per_view_stratum < 4:
        raise ValueError("G02 science requires at least four rays per view/stratum")
    solution = FixedCameraHumanSolution.model_validate(read_json(t05_solution_path))
    graph = load_interval_material_track_graph(q03_binding_path)
    with np.load(hull_binding_path, allow_pickle=False) as archive:
        silhouettes = archive["silhouettes"].astype(np.float32)
        intrinsics = archive["intrinsics"].astype(np.float32)
        rotations = archive["rotations"].astype(np.float32)
        translations = archive["translations"].astype(np.float32)
        source_indices = archive["source_frame_indices"].astype(np.int64)
        motion_uncertainty = archive["motion_uncertainty"].astype(np.float32)
        semantic_maps = [
            archive[name].astype(np.float32)
            for name in archive.files
            if name.startswith("semantic__")
        ]
    expected_sources = np.asarray(
        [frame.source_frame_index for frame in solution.frames], dtype=np.int64
    )
    expected_translations = np.asarray(
        [frame.root_translation_metres for frame in solution.frames], dtype=np.float32
    )
    if not np.array_equal(source_indices, expected_sources):
        raise ValueError("G02 science T05 and hull training sources differ")
    if not np.array_equal(translations, expected_translations):
        raise ValueError("G02 science hull translations do not exactly bind T05")
    if silhouettes.shape[0] != 144 or len(semantic_maps) < 1:
        raise ValueError("G02 science requires all 144 train silhouettes and semantics")
    height, width = silhouettes.shape[-2:]
    ray_pixels: list[np.ndarray] = []
    ray_targets: list[np.ndarray] = []
    ray_normals: list[np.ndarray] = []
    ray_confidence: list[np.ndarray] = []
    ray_profile: list[np.ndarray] = []
    ray_codes: list[np.ndarray] = []
    for slot, source_index in enumerate(source_indices.tolist()):
        silhouette = silhouettes[slot] >= 0.5
        eroded = binary_erosion(silhouette, iterations=2)
        boundary = silhouette & ~binary_erosion(silhouette, iterations=1)
        semantic_boundary = np.zeros_like(silhouette)
        for values in semantic_maps:
            semantic = values[slot] >= 0.25
            semantic_boundary |= semantic & ~binary_erosion(semantic, iterations=1)
        signed_profile = distance_transform_edt(silhouette) - distance_transform_edt(~silhouette)
        signed_profile = signed_profile / math.hypot(height, width)
        normal_map = _normal_for_source(normal_root, source_index, (height, width))
        normal_norm = np.linalg.norm(normal_map, axis=-1)
        strata = (eroded, boundary, ~silhouette, semantic_boundary & silhouette)
        for code, eligible in enumerate(strata):
            selected = _sample_without_replacement(
                eligible,
                rays_per_view_stratum,
                seed=seed + slot * 101 + code * 17,
            )
            y = selected[:, 0]
            x = selected[:, 1]
            ray_pixels.append(np.column_stack((x, y, np.full(len(selected), slot, dtype=np.int64))))
            ray_targets.append(silhouettes[slot, y, x])
            ray_normals.append(normal_map[y, x])
            confidence = np.clip(normal_norm[y, x], 0.0, 1.0) * (1.0 - motion_uncertainty[slot])
            ray_confidence.append(confidence.astype(np.float32))
            ray_profile.append(signed_profile[y, x].astype(np.float32))
            ray_codes.append(np.full(len(selected), code, dtype=np.uint8))
    accepted = graph.accepted.astype(bool)
    if int(accepted.sum()) != 249:
        raise ValueError("G02 science requires the frozen 249 accepted Q03 tracks")
    anchors = graph.material_anchor_mean_metres[accepted].astype(np.float32)
    weights = graph.track_weights[accepted].astype(np.float32)
    covariance_trace = np.trace(
        graph.material_anchor_covariance_metres2[accepted], axis1=1, axis2=2
    ).astype(np.float32)
    source_to_slot = {source: slot for slot, source in enumerate(source_indices.tolist())}
    accepted_indices = np.flatnonzero(accepted)
    track_view_slots = np.asarray(
        [
            source_to_slot[int(graph.source_frame_indices[int(graph.track_offsets[index])])]
            for index in accepted_indices
        ],
        dtype=np.int64,
    )
    source_hashes = {
        "hull_binding": sha256_file(hull_binding_path),
        "t05_solution": sha256_file(t05_solution_path),
        "q03_binding": sha256_file(q03_binding_path),
    }
    normal_hash_manifest = {
        str(source): sha256_file(sorted(normal_root.glob(f"*source_{source:06d}.png"))[0])
        for source in source_indices.tolist()
    }
    source_hashes["train_normal_manifest"] = hashlib.sha256(
        json.dumps(normal_hash_manifest, sort_keys=True).encode()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(G02_SCIENCE_EVIDENCE_SCHEMA),
        ray_pixels=np.concatenate(ray_pixels).astype(np.int64),
        ray_targets=np.concatenate(ray_targets).astype(np.float32),
        ray_normals=np.concatenate(ray_normals).astype(np.float32),
        ray_confidence=np.concatenate(ray_confidence).astype(np.float32),
        ray_profile_distance=np.concatenate(ray_profile).astype(np.float32),
        ray_stratum_codes=np.concatenate(ray_codes).astype(np.uint8),
        intrinsics=intrinsics,
        rotations=rotations,
        translations=translations,
        source_frame_indices=source_indices,
        motion_uncertainty=motion_uncertainty,
        track_anchors=anchors,
        track_weights=weights,
        track_covariance_trace=covariance_trace,
        track_view_slots=track_view_slots,
        stratum_codebook=np.asarray(json.dumps(dict(enumerate(RAY_STRATA)), sort_keys=True)),
        source_hashes=np.asarray(json.dumps(source_hashes, sort_keys=True)),
        training_view_count=np.asarray(len(source_indices), dtype=np.int64),
        development_records=np.asarray(0, dtype=np.int64),
        sealed_test_records=np.asarray(0, dtype=np.int64),
    )
    return output_path


def load_g02_science_evidence(path: Path, *, device: torch.device | str) -> G02ScienceEvidence:
    reject_sealed_capability([path])
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"]) != G02_SCIENCE_EVIDENCE_SCHEMA:
            raise ValueError("G02 science evidence schema is invalid")
        if int(archive["development_records"]) != 0 or int(archive["sealed_test_records"]) != 0:
            raise ValueError("G02 science training evidence crosses the evaluator boundary")
        target = torch.device(device)

        def tensor(name: str, *, dtype: torch.dtype | None = None) -> Tensor:
            return torch.as_tensor(archive[name], dtype=dtype, device=target)

        result = G02ScienceEvidence(
            ray_pixels=tensor("ray_pixels", dtype=torch.long),
            ray_targets=tensor("ray_targets", dtype=torch.float32),
            ray_normals=tensor("ray_normals", dtype=torch.float32),
            ray_confidence=tensor("ray_confidence", dtype=torch.float32),
            ray_profile_distance=tensor("ray_profile_distance", dtype=torch.float32),
            ray_stratum_codes=tensor("ray_stratum_codes", dtype=torch.long),
            intrinsics=tensor("intrinsics", dtype=torch.float32),
            rotations=tensor("rotations", dtype=torch.float32),
            translations=tensor("translations", dtype=torch.float32),
            source_frame_indices=tensor("source_frame_indices", dtype=torch.long),
            motion_uncertainty=tensor("motion_uncertainty", dtype=torch.float32),
            track_anchors=tensor("track_anchors", dtype=torch.float32),
            track_weights=tensor("track_weights", dtype=torch.float32),
            track_covariance_trace=tensor("track_covariance_trace", dtype=torch.float32),
            track_view_slots=tensor("track_view_slots", dtype=torch.long),
            source_hashes=json.loads(str(archive["source_hashes"])),
        )
    if result.view_count != 144 or result.track_anchors.shape != (249, 3):
        raise ValueError("G02 science evidence does not cover the frozen train/track sets")
    if result.ray_pixels.shape != (result.ray_count, 3):
        raise ValueError("G02 science ray pixels are invalid")
    if set(result.ray_stratum_codes.detach().cpu().tolist()) != set(range(len(RAY_STRATA))):
        raise ValueError("G02 science ray strata are incomplete")
    return result


def build_g02_science_arm_binding(
    output_path: Path,
    *,
    source_revision: str,
    qualification_lifecycle_path: Path,
    training_evidence_path: Path,
    evidence_volume_path: Path,
    q03_binding_path: Path,
    schedule: G02ScienceSchedule,
    attempt_id: str = "scientific-attempt-r01",
) -> Path:
    paths = [
        output_path,
        qualification_lifecycle_path,
        training_evidence_path,
        evidence_volume_path,
        q03_binding_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G02 science arm binding is immutable")
    if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise ValueError("G02 science source revision must be a full lowercase Git revision")
    if not attempt_id.startswith("scientific-attempt-r") or not attempt_id[-2:].isdigit():
        raise ValueError("G02 science attempt ID must be explicitly revisioned")
    lifecycle = read_json(qualification_lifecycle_path)
    if lifecycle.get("status") != "pass" or lifecycle.get("state") != "qualified":
        raise ValueError("G02 science requires a passing target-CUDA lifecycle")
    hashes = {
        "qualification_lifecycle": sha256_file(qualification_lifecycle_path),
        "training_evidence": sha256_file(training_evidence_path),
        "evidence_volume": sha256_file(evidence_volume_path),
        "q03_binding": sha256_file(q03_binding_path),
    }
    common = {
        "source_revision": source_revision,
        "source_hashes": hashes,
        "schedule": schedule.as_dict(),
        "ray_batches": "one_immutable_permutation_consumed_by_both_arms",
        "camera_root_pose": "exact_t05_bound_root_transform",
        "control_geometry_updates": 0,
        "control_scheduled_noop_steps": schedule.optimizer_steps,
        "extraction": G02_SCIENCE_EXTRACTOR,
        "evaluator": G02_SCIENCE_EVALUATOR,
        "development_available_to_trainer": False,
        "sealed_test_accesses": 0,
        "automatic_retries": 0,
    }
    common_hash = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
    return write_json(
        output_path,
        {
            "schema_version": G02_SCIENCE_ARM_SCHEMA,
            "experiment_id": G02_EXPERIMENT_ID,
            "attempt_id": attempt_id,
            "common": common,
            "common_binding_sha256": common_hash,
            "treatment": {
                "mechanism": "trainable_shortcut_resistant_direct_multiresolution_field",
                "output_root": f"{attempt_id}/treatment",
                "common_binding_sha256": common_hash,
            },
            "control": {
                "mechanism": "frozen_uncertainty_aware_visual_hull_carrier",
                "output_root": f"{attempt_id}/control",
                "common_binding_sha256": common_hash,
            },
            "separate_output_roots": True,
            "attempt_marker_created": False,
            "scientific_state": "registered",
        },
    )


class FrozenEvidenceSDF(nn.Module):
    values_grid: Tensor

    def __init__(self, evidence: EvidenceVolume) -> None:
        super().__init__()
        self.extent = float(evidence.metadata.extent)
        self.register_buffer("values_grid", evidence.signed_distance.detach().clone())

    def forward(self, points: Tensor) -> Tensor:
        return trilinear_grid_sample(self.values_grid, points, extent=self.extent)


@dataclass
class DeterministicPermutationSampler:
    size: int
    generator: torch.Generator
    permutation: Tensor
    cursor: int = 0
    epoch: int = 0

    @classmethod
    def create(
        cls, size: int, *, seed: int, device: torch.device
    ) -> DeterministicPermutationSampler:
        if size < 1:
            raise ValueError("G02 science sampler requires a nonempty population")
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed)
        permutation = torch.randperm(size, generator=generator, device=device)
        return cls(size=size, generator=generator, permutation=permutation)

    def take(self, count: int) -> Tensor:
        if count < 1:
            raise ValueError("G02 science sampler batch must be positive")
        pieces: list[Tensor] = []
        remaining = count
        while remaining:
            available = self.size - self.cursor
            take = min(available, remaining)
            pieces.append(self.permutation[self.cursor : self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor == self.size:
                self.epoch += 1
                self.permutation = torch.randperm(
                    self.size,
                    generator=self.generator,
                    device=self.permutation.device,
                )
                self.cursor = 0
        return torch.cat(pieces)

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "generator_state": self.generator.get_state().cpu(),
            "permutation": self.permutation.detach().cpu(),
            "cursor": self.cursor,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("G02 science sampler population changed")
        permutation = state["permutation"]
        if not isinstance(permutation, Tensor) or permutation.shape != (self.size,):
            raise ValueError("G02 science sampler permutation is invalid")
        self.generator.set_state(state["generator_state"].cpu())
        self.permutation = permutation.to(self.permutation.device)
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])


def _volume_training_pool(evidence: EvidenceVolume) -> tuple[Tensor, Tensor, Tensor]:
    resolution = evidence.metadata.resolution
    coordinates = torch.linspace(
        -evidence.metadata.extent,
        evidence.metadata.extent,
        resolution,
        dtype=evidence.signed_distance.dtype,
        device=evidence.signed_distance.device,
    )
    xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    target = evidence.signed_distance.reshape(-1)
    support = evidence.support_count.to(torch.float32).reshape(-1)
    uncertainty = evidence.mask_uncertainty.reshape(-1).clamp(0.0, 1.0)
    prior = evidence.prior_contribution.reshape(-1).clamp(0.0, 1.0)
    weights = (1.0 + torch.log1p(support)) * (1.0 - 0.75 * uncertainty) * (1.0 - prior)
    weights = weights / weights.mean().clamp_min(1.0e-6)
    return points, target, weights


def _surface_visibility_points(science: G02ScienceEvidence, track_indices: Tensor) -> Tensor:
    slots = science.track_view_slots[track_indices]
    rotations = science.rotations[slots]
    translations = science.translations[slots]
    origins = torch.einsum("bi,bij->bj", -translations, rotations)
    anchors = science.track_anchors[track_indices]
    return origins + 0.85 * (anchors - origins)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-8)


def _build_science_rays(
    pixels: Tensor,
    intrinsics: Tensor,
    rotations: Tensor,
    translations: Tensor,
) -> tuple[Tensor, Tensor]:
    slots = pixels[:, 2].to(torch.long)
    xy = pixels[:, :2].to(intrinsics.dtype)
    camera_directions = torch.stack(
        (
            (xy[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0],
            (xy[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(xy[:, 0]),
        ),
        dim=-1,
    )
    selected_rotations = rotations[slots]
    selected_translations = translations[slots]
    directions = F.normalize(
        torch.einsum("bi,bij->bj", camera_directions, selected_rotations), dim=-1
    )
    origins = torch.einsum("bi,bij->bj", -selected_translations, selected_rotations)
    return origins, directions


def _science_step(
    model: V2NeuralSDF,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    science: G02ScienceEvidence,
    volume_points: Tensor,
    volume_targets: Tensor,
    volume_weights: Tensor,
    ray_sampler: DeterministicPermutationSampler,
    volume_sampler: DeterministicPermutationSampler,
    track_sampler: DeterministicPermutationSampler,
    schedule: G02ScienceSchedule,
    *,
    stage_index: int,
) -> dict[str, float]:
    ray_indices = ray_sampler.take(schedule.ray_batch_size)
    grid_indices = volume_sampler.take(schedule.volume_batch_size)
    track_indices = track_sampler.take(schedule.track_batch_size)
    pixels = science.ray_pixels[ray_indices]
    origins, directions = _build_science_rays(
        pixels,
        science.intrinsics,
        science.rotations,
        science.translations,
    )
    slots = pixels[:, 2].to(torch.long)
    jacobians = science.rotations[slots, None]
    extent = float((volume_points.max() - volume_points.min()).detach()) * 0.5
    optimizer.zero_grad(set_to_none=True)
    rendered = render_neus_sdf(
        model,
        origins,
        directions,
        near=max(0.01, 0.01 * extent),
        far=4.0 * extent,
        sample_count=schedule.coarse_samples,
        hierarchical_sample_count=schedule.hierarchical_samples,
        inverse_sharpness=64.0,
        deformation_jacobian=jacobians,
        create_graph=True,
        ray_chunk_size=min(schedule.ray_batch_size, 128),
    )
    target = science.ray_targets[ray_indices]
    view_weight = 1.0 - 0.75 * science.motion_uncertainty[slots].clamp(0.0, 1.0)
    bce = F.binary_cross_entropy(
        rendered.silhouette.clamp(1.0e-5, 1.0 - 1.0e-5),
        target,
        reduction="none",
    )
    silhouette = _weighted_mean(bce, view_weight)
    profile_weight = torch.exp(-48.0 * science.ray_profile_distance[ray_indices].abs())
    profile = _weighted_mean(bce, view_weight * profile_weight)
    free = science.ray_stratum_codes[ray_indices] == RAY_STRATA.index("free_space")
    free_space = (
        _weighted_mean(rendered.silhouette[free], view_weight[free])
        if torch.any(free)
        else rendered.silhouette.sum() * 0.0
    )
    prediction = model(volume_points[grid_indices])
    evidence_sdf = _weighted_mean(
        F.smooth_l1_loss(prediction, volume_targets[grid_indices], reduction="none"),
        volume_weights[grid_indices],
    )
    differentiable_points = volume_points[grid_indices].detach().requires_grad_(True)
    gradients = torch.autograd.grad(
        model(differentiable_points).sum(),
        differentiable_points,
        create_graph=True,
    )[0]
    eikonal = (torch.linalg.vector_norm(gradients, dim=-1) - 1.0).square().mean()
    normal = rendered.silhouette.sum() * 0.0
    if stage_index >= 1:
        normal_valid = target > 0.5
        if torch.any(normal_valid):
            predicted_normal = F.normalize(rendered.normals[normal_valid], dim=-1, eps=1.0e-8)
            observed_normal = F.normalize(
                science.ray_normals[ray_indices][normal_valid], dim=-1, eps=1.0e-8
            )
            normal_residual = 1.0 - (predicted_normal * observed_normal).sum(dim=-1)
            normal = _weighted_mean(
                normal_residual,
                science.ray_confidence[ray_indices][normal_valid].clamp_min(1.0e-3),
            )
    track_surface = rendered.silhouette.sum() * 0.0
    track_visibility = rendered.silhouette.sum() * 0.0
    if stage_index >= 2:
        anchors = science.track_anchors[track_indices]
        covariance_weight = 1.0 / (
            1.0 + 400.0 * science.track_covariance_trace[track_indices].clamp_min(0.0)
        )
        weights = science.track_weights[track_indices].clamp_min(1.0e-4) * covariance_weight
        track_surface = _weighted_mean(model(anchors).abs(), weights)
        visibility_points = _surface_visibility_points(science, track_indices)
        track_visibility = _weighted_mean(
            F.softplus(-20.0 * model(visibility_points)) / 20.0,
            weights,
        )
    loss = (
        schedule.silhouette_weight * silhouette
        + schedule.profile_weight * profile
        + schedule.free_space_weight * free_space
        + schedule.evidence_sdf_weight * evidence_sdf
        + schedule.eikonal_weight * eikonal
        + schedule.normal_weight * normal
        + schedule.track_weight * (track_surface + 0.25 * track_visibility)
    )
    if not torch.isfinite(loss):
        raise RuntimeError("G02 science objective became non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        schedule.maximum_gradient_norm,
    )
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("G02 science gradient became non-finite")
    optimizer.step()
    scheduler.step()
    output = _output_layer(model)
    if output.bias.requires_grad or float(output.bias.detach()) != 0.0:
        raise RuntimeError("G02 science output bias escaped the frozen-zero contract")
    return {
        "loss": float(loss.detach()),
        "silhouette": float(silhouette.detach()),
        "profile": float(profile.detach()),
        "free_space": float(free_space.detach()),
        "evidence_sdf": float(evidence_sdf.detach()),
        "eikonal": float(eikonal.detach()),
        "normal": float(normal.detach()),
        "track_surface": float(track_surface.detach()),
        "track_visibility": float(track_visibility.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def _endpoint_metrics(
    field: nn.Module,
    science: G02ScienceEvidence,
    *,
    schedule: G02ScienceSchedule,
    maximum_rays: int = 512,
) -> dict[str, float]:
    count = min(science.ray_count, maximum_rays)
    indices = (
        torch.linspace(
            0,
            science.ray_count - 1,
            count,
            device=science.ray_pixels.device,
        )
        .round()
        .to(torch.long)
    )
    rendered_silhouettes: list[Tensor] = []
    rendered_normals: list[Tensor] = []
    for chunk_indices in indices.split(min(count, 128)):  # type: ignore[no-untyped-call]
        pixels = science.ray_pixels[chunk_indices]
        origins, directions = _build_science_rays(
            pixels,
            science.intrinsics,
            science.rotations,
            science.translations,
        )
        slots = pixels[:, 2].to(torch.long)
        with torch.enable_grad():
            rendered = render_neus_sdf(
                field,
                origins,
                directions,
                near=0.01,
                far=6.0,
                sample_count=schedule.coarse_samples,
                hierarchical_sample_count=schedule.hierarchical_samples,
                inverse_sharpness=64.0,
                deformation_jacobian=science.rotations[slots, None],
                create_graph=False,
                ray_chunk_size=len(chunk_indices),
            )
        rendered_silhouettes.append(rendered.silhouette.detach())
        rendered_normals.append(rendered.normals.detach())
    target = science.ray_targets[indices]
    predicted = torch.cat(rendered_silhouettes)
    predicted_normals = torch.cat(rendered_normals)
    binary = predicted >= 0.5
    truth = target >= 0.5
    intersection = torch.logical_and(binary, truth).sum()
    union = torch.logical_or(binary, truth).sum().clamp_min(1)
    bce = F.binary_cross_entropy(predicted.clamp(1.0e-5, 1.0 - 1.0e-5), target)
    valid = truth
    if torch.any(valid):
        cosine = (
            F.normalize(predicted_normals[valid], dim=-1, eps=1.0e-8)
            * F.normalize(science.ray_normals[indices][valid], dim=-1, eps=1.0e-8)
        ).sum(dim=-1)
        normal_degrees = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
        median_normal = float(torch.median(normal_degrees))
    else:
        median_normal = math.inf
    track_residual = field(science.track_anchors).detach().abs()
    return {
        "sampled_ray_iou": float(intersection / union),
        "sampled_ray_bce": float(bce),
        "sampled_median_normal_degrees": median_normal,
        "q03_anchor_median_abs_sdf_metres": float(torch.median(track_residual)),
    }


def _capture_science_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    samplers: Mapping[str, DeterministicPermutationSampler],
    *,
    completed_steps: int,
    stage_index: int,
    arm_binding: Mapping[str, Any],
    evidence_hashes: Mapping[str, str],
    topology_state: Mapping[str, Any],
) -> bytes:
    payload = {
        "schema_version": G02_SCIENCE_CHECKPOINT_SCHEMA,
        "completed_steps": completed_steps,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "grad_scaler_state": {"enabled": False},
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "named_sampler_states": {name: sampler.state_dict() for name, sampler in samplers.items()},
        "stage_controller_state": {
            "stage_index": stage_index,
            "stage_name": SCIENCE_STAGE_SEQUENCE[stage_index],
            "completed_steps": completed_steps,
        },
        "immutable_arm_binding": dict(arm_binding),
        "evidence_hashes": dict(evidence_hashes),
        "topology_state": dict(topology_state),
        "extractor_identifier": G02_SCIENCE_EXTRACTOR,
        "evaluator_identifier": G02_SCIENCE_EVALUATOR,
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _restore_science_checkpoint(
    data: bytes,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    samplers: Mapping[str, DeterministicPermutationSampler],
    *,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(io.BytesIO(data), map_location=device, weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != G02_SCIENCE_CHECKPOINT_SCHEMA
    ):
        raise ValueError("G02 science checkpoint schema is invalid")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    for name, sampler in samplers.items():
        sampler.load_state_dict(payload["named_sampler_states"][name])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_states = payload["cuda_rng_state_all"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("G02 science checkpoint requires unavailable CUDA RNG state")
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])
    return payload


def _parameter_states_equal(first: Mapping[str, Tensor], second: Mapping[str, Tensor]) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[name], second[name]) for name in first
    )


def _make_samplers(
    science: G02ScienceEvidence,
    volume_count: int,
    schedule: G02ScienceSchedule,
    device: torch.device,
) -> dict[str, DeterministicPermutationSampler]:
    return {
        "rays": DeterministicPermutationSampler.create(
            science.ray_count, seed=schedule.seed + 11, device=device
        ),
        "volume": DeterministicPermutationSampler.create(
            volume_count, seed=schedule.seed + 23, device=device
        ),
        "tracks": DeterministicPermutationSampler.create(
            len(science.track_anchors), seed=schedule.seed + 37, device=device
        ),
    }


def _create_attempt_marker(
    path: Path,
    *,
    source_revision: str,
    arm_binding_path: Path,
    schedule: G02ScienceSchedule,
    maximum_cost_usd: float,
    provider_rate_usd_per_hour: float,
    price_checked_at: str,
    attempt_id: str = "scientific-attempt-r01",
) -> Path:
    if maximum_cost_usd <= 0 or provider_rate_usd_per_hour <= 0 or not price_checked_at:
        raise ValueError("G02 scientific attempt must be priced before optimizer step one")
    payload = {
        "schema_version": "frayid_v2_scientific_attempt.v1",
        "event": "attempt_started",
        "experiment_id": G02_EXPERIMENT_ID,
        "run_id": "registered-20260903-r01",
        "attempt_id": attempt_id,
        "source_revision": source_revision,
        "optimizer_step": 1,
        "arm_binding_sha256": sha256_file(arm_binding_path),
        "schedule": schedule.as_dict(),
        "provider_rate_usd_per_hour": provider_rate_usd_per_hour,
        "price_checked_at": price_checked_at,
        "maximum_cost_usd": maximum_cost_usd,
        "automatic_retries": 0,
        "sealed_test_accesses": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _write_raw_field_grid(
    model: nn.Module,
    output_path: Path,
    *,
    extent: float,
    resolution: int,
    source_revision: str,
    arm_binding_sha256: str,
) -> Path:
    if resolution < 16:
        raise ValueError("G02 raw field grid resolution must be at least 16")
    coordinates = torch.linspace(
        -extent,
        extent,
        resolution,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    values: list[Tensor] = []
    with torch.no_grad():
        for chunk in points.split(65536):  # type: ignore[no-untyped-call]
            values.append(model(chunk).detach().cpu())
    field = torch.cat(values).reshape(resolution, resolution, resolution).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray("frayid_v2_g02_raw_canonical_field.v1"),
        signed_distance=field.astype(np.float32),
        extent=np.asarray(extent, dtype=np.float32),
        resolution=np.asarray(resolution, dtype=np.int64),
        source_revision=np.asarray(source_revision),
        arm_binding_sha256=np.asarray(arm_binding_sha256),
        representation=np.asarray("authoritative_raw_direct_field_candidate"),
        topology_state=np.asarray("search_not_committed"),
    )
    return output_path


def run_g02_science_training(
    evidence_volume_path: Path,
    training_evidence_path: Path,
    arm_binding_path: Path,
    output_root: Path,
    *,
    source_revision: str,
    device: torch.device | str,
    schedule: G02ScienceSchedule | None = None,
    raw_field_resolution: int = 48,
    scientific_attempt: bool = False,
    attempt_marker_path: Path | None = None,
    provider_rate_usd_per_hour: float | None = None,
    price_checked_at: str | None = None,
    maximum_cost_usd: float | None = None,
    attempt_marker_commit: Callable[[], None] | None = None,
) -> Path:
    """Train one frozen G02 treatment/control schedule without development data.

    A successful return freezes a raw field candidate but deliberately does not
    call it passing. Development metrics and exact COMMIT audits belong to
    separate post-endpoint evaluators.
    """

    started_at = time.perf_counter()
    paths = [evidence_volume_path, training_evidence_path, arm_binding_path, output_root]
    if attempt_marker_path is not None:
        paths.append(attempt_marker_path)
    reject_sealed_capability(paths)
    if output_root.exists():
        raise FileExistsError("G02 scientific output root is immutable")
    if scientific_attempt != (attempt_marker_path is not None):
        raise ValueError("G02 science marker capability must exactly match attempt mode")
    if attempt_marker_commit is not None and not scientific_attempt:
        raise ValueError("G02 science marker commit is only valid for an attempt")
    if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise ValueError("G02 science source revision must be a full lowercase Git revision")
    active_schedule = schedule or G02ScienceSchedule()
    arm = read_json(arm_binding_path)
    if arm.get("schema_version") != G02_SCIENCE_ARM_SCHEMA:
        raise ValueError("G02 science arm binding schema is invalid")
    common = arm.get("common", {})
    if common.get("source_revision") != source_revision:
        raise ValueError("G02 science source differs from its immutable arm binding")
    if common.get("schedule") != active_schedule.as_dict():
        raise ValueError("G02 science schedule differs from its immutable arm binding")
    attempt_id = arm.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.startswith("scientific-attempt-r"):
        raise ValueError("G02 science arm binding has no valid attempt ID")
    expected_hashes = common.get("source_hashes", {})
    if expected_hashes.get("training_evidence") != sha256_file(training_evidence_path):
        raise ValueError("G02 science training evidence hash changed")
    if expected_hashes.get("evidence_volume") != sha256_file(evidence_volume_path):
        raise ValueError("G02 science evidence volume hash changed")
    target_device = torch.device(device)
    torch.manual_seed(active_schedule.seed)
    np.random.seed(active_schedule.seed)
    random.seed(active_schedule.seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(active_schedule.seed)
        torch.cuda.reset_peak_memory_stats(target_device)
    evidence = EvidenceVolume.load(evidence_volume_path, device=target_device)
    science = load_g02_science_evidence(training_evidence_path, device=target_device)
    model = prepare_shortcut_resistant_field(evidence, seed=active_schedule.seed).to(target_device)
    model.train()
    control = FrozenEvidenceSDF(evidence).to(target_device)
    control.eval()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=active_schedule.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    volume_points, volume_targets, volume_weights = _volume_training_pool(evidence)
    samplers = _make_samplers(
        science,
        len(volume_points),
        active_schedule,
        target_device,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    treatment_root = output_root / "treatment"
    control_root = output_root / "control"
    checkpoint_root = treatment_root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    control_root.mkdir(parents=True)
    baseline_control = _endpoint_metrics(
        control,
        science,
        schedule=active_schedule,
        maximum_rays=active_schedule.endpoint_evaluation_rays,
    )
    baseline_treatment = _endpoint_metrics(
        model,
        science,
        schedule=active_schedule,
        maximum_rays=active_schedule.endpoint_evaluation_rays,
    )
    marker_created = False
    if scientific_attempt:
        assert attempt_marker_path is not None
        if (
            provider_rate_usd_per_hour is None
            or price_checked_at is None
            or maximum_cost_usd is None
        ):
            raise ValueError("G02 scientific attempt is missing its price/cost record")
        _create_attempt_marker(
            attempt_marker_path,
            source_revision=source_revision,
            arm_binding_path=arm_binding_path,
            schedule=active_schedule,
            maximum_cost_usd=float(maximum_cost_usd),
            provider_rate_usd_per_hour=float(provider_rate_usd_per_hour),
            price_checked_at=str(price_checked_at),
            attempt_id=attempt_id,
        )
        if attempt_marker_commit is not None:
            attempt_marker_commit()
        marker_created = True

    loss_trace: list[dict[str, float | int | str]] = []
    stage_reports: list[dict[str, Any]] = []
    completed_steps = 0
    for stage_index, (stage_name, stage_steps) in enumerate(
        zip(SCIENCE_STAGE_SEQUENCE, active_schedule.stage_steps, strict=True)
    ):
        first_losses: list[float] = []
        last_losses: list[float] = []
        for stage_step in range(stage_steps):
            metrics = _science_step(
                model,
                optimizer,
                scheduler,
                science,
                volume_points,
                volume_targets,
                volume_weights,
                samplers["rays"],
                samplers["volume"],
                samplers["tracks"],
                active_schedule,
                stage_index=stage_index,
            )
            completed_steps += 1
            record: dict[str, float | int | str] = {
                "global_step": completed_steps,
                "stage_index": stage_index,
                "stage_name": stage_name,
                "stage_step": stage_step + 1,
                **metrics,
            }
            loss_trace.append(record)
            if stage_step < min(16, stage_steps):
                first_losses.append(metrics["loss"])
            if stage_step >= max(0, stage_steps - 16):
                last_losses.append(metrics["loss"])
            if completed_steps % active_schedule.checkpoint_interval == 0:
                checkpoint = _capture_science_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    samplers,
                    completed_steps=completed_steps,
                    stage_index=stage_index,
                    arm_binding=arm,
                    evidence_hashes=expected_hashes,
                    topology_state={"stage": "search", "connectivity_sha256": None},
                )
                (checkpoint_root / f"step_{completed_steps:05d}.pt").write_bytes(checkpoint)
        endpoint = _endpoint_metrics(
            model,
            science,
            schedule=active_schedule,
            maximum_rays=active_schedule.endpoint_evaluation_rays,
        )
        blockers: list[str] = []
        if (
            not first_losses
            or not last_losses
            or not all(math.isfinite(value) for value in [*first_losses, *last_losses])
        ):
            blockers.append("nonfinite_or_empty_stage_loss")
        if endpoint["sampled_ray_bce"] > baseline_control["sampled_ray_bce"] + 0.02:
            blockers.append("sampled_train_silhouette_regressed_against_frozen_control")
        if stage_index >= 1 and (
            endpoint["sampled_median_normal_degrees"]
            > baseline_control["sampled_median_normal_degrees"] + 2.0
        ):
            blockers.append("sampled_train_normal_regressed_against_frozen_control")
        if stage_index >= 2 and (
            endpoint["q03_anchor_median_abs_sdf_metres"]
            > baseline_control["q03_anchor_median_abs_sdf_metres"] * 1.05 + 1.0e-5
        ):
            blockers.append("q03_anchor_surface_residual_regressed_against_frozen_control")
        stage_reports.append(
            {
                "stage_index": stage_index,
                "stage_name": stage_name,
                "registered_steps": stage_steps,
                "completed_steps": stage_steps,
                "initial_window_mean_loss": float(np.mean(first_losses)),
                "final_window_mean_loss": float(np.mean(last_losses)),
                "endpoint_metrics": endpoint,
                "status": "pass" if not blockers else "fail",
                "blockers": blockers,
            }
        )
        if blockers:
            break
    final_stage_index = len(stage_reports) - 1
    checkpoint = _capture_science_checkpoint(
        model,
        optimizer,
        scheduler,
        samplers,
        completed_steps=completed_steps,
        stage_index=final_stage_index,
        arm_binding=arm,
        evidence_hashes=expected_hashes,
        topology_state={"stage": "search", "connectivity_sha256": None},
    )
    final_checkpoint_path = treatment_root / "final_checkpoint.pt"
    final_checkpoint_path.write_bytes(checkpoint)

    replay_states: list[dict[str, Tensor]] = []
    replay_metrics: list[dict[str, float]] = []
    for _ in range(2):
        replay_model = prepare_shortcut_resistant_field(
            evidence, seed=active_schedule.seed + 999
        ).to(target_device)
        replay_optimizer = torch.optim.Adam(
            [parameter for parameter in replay_model.parameters() if parameter.requires_grad],
            lr=9.0,
        )
        replay_scheduler = torch.optim.lr_scheduler.LambdaLR(replay_optimizer, lambda _: 1.0)
        replay_samplers = _make_samplers(
            science,
            len(volume_points),
            active_schedule,
            target_device,
        )
        _restore_science_checkpoint(
            checkpoint,
            replay_model,
            replay_optimizer,
            replay_scheduler,
            replay_samplers,
            device=target_device,
        )
        replay_metrics.append(
            _science_step(
                replay_model,
                replay_optimizer,
                replay_scheduler,
                science,
                volume_points,
                volume_targets,
                volume_weights,
                replay_samplers["rays"],
                replay_samplers["volume"],
                replay_samplers["tracks"],
                active_schedule,
                stage_index=final_stage_index,
            )
        )
        replay_states.append(copy.deepcopy(replay_model.state_dict()))
    replay_exact = _parameter_states_equal(replay_states[0], replay_states[1]) and (
        replay_metrics[0] == replay_metrics[1]
    )
    final_metrics = _endpoint_metrics(
        model,
        science,
        schedule=active_schedule,
        maximum_rays=active_schedule.endpoint_evaluation_rays,
    )
    arm_hash = sha256_file(arm_binding_path)
    raw_field_path = _write_raw_field_grid(
        model,
        treatment_root / "raw_canonical_field.npz",
        extent=evidence.metadata.extent,
        resolution=raw_field_resolution,
        source_revision=source_revision,
        arm_binding_sha256=arm_hash,
    )
    np.savez_compressed(
        control_root / "frozen_canonical_field.npz",
        schema_version=np.asarray("frayid_v2_g02_frozen_control_field.v1"),
        signed_distance=evidence.signed_distance.detach().cpu().numpy().astype(np.float32),
        extent=np.asarray(evidence.metadata.extent, dtype=np.float32),
        resolution=np.asarray(evidence.metadata.resolution, dtype=np.int64),
        arm_binding_sha256=np.asarray(arm_hash),
    )
    write_json(treatment_root / "loss_trace.json", loss_trace)
    stage_blockers = [
        f"stage_{stage['stage_index']}:{blocker}"
        for stage in stage_reports
        for blocker in stage["blockers"]
    ]
    blockers = list(stage_blockers)
    if completed_steps != active_schedule.optimizer_steps:
        blockers.append("registered_optimizer_schedule_incomplete")
    if not replay_exact:
        blockers.append("same_device_next_step_replay_failed")
    if _output_layer(model).bias.requires_grad or float(_output_layer(model).bias.detach()) != 0.0:
        blockers.append("output_bias_not_frozen_zero")
    report = {
        "schema_version": G02_SCIENCE_REPORT_SCHEMA,
        "experiment_id": G02_EXPERIMENT_ID,
        "attempt_id": attempt_id if scientific_attempt else None,
        "status": "endpoint_frozen_unscored" if not blockers else "terminal_training_failure",
        "scientific_attempt": scientific_attempt,
        "scientific_attempt_marker_created": marker_created,
        "source_revision": source_revision,
        "device": str(target_device),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if target_device.type == "cuda" else None
        ),
        "schedule": active_schedule.as_dict(),
        "wall_time_seconds": time.perf_counter() - started_at,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else None
        ),
        "completed_optimizer_steps": completed_steps,
        "control_scheduled_noop_steps": active_schedule.optimizer_steps,
        "automatic_retries": 0,
        "development_records_available_to_trainer": 0,
        "development_outcomes_read": 0,
        "sealed_test_accesses": 0,
        "bounded_rgb_stage": "not_activated_geometry_only_endpoint",
        "arm_binding_sha256": arm_hash,
        "training_evidence_sha256": sha256_file(training_evidence_path),
        "evidence_volume_sha256": sha256_file(evidence_volume_path),
        "baseline_control_metrics": baseline_control,
        "baseline_treatment_metrics": baseline_treatment,
        "stage_reports": stage_reports,
        "final_training_metrics": final_metrics,
        "checkpoint": {
            "path": str(final_checkpoint_path),
            "sha256": sha256_file(final_checkpoint_path),
            "same_device_next_step_replay_exact": replay_exact,
            "complete_state": True,
        },
        "raw_field": {
            "path": str(raw_field_path),
            "sha256": sha256_file(raw_field_path),
            "resolution": raw_field_resolution,
            "topology_state": "search_not_committed",
        },
        "independent_evaluation_pending": True,
        "authoritative_result_claimed": False,
        "blockers": blockers,
    }
    return write_json(output_root / "training_report.json", report)
