from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from torch import Tensor, nn

from frayid.dataset import read_dataset_manifest
from frayid.initialization import load_initialization
from frayid.io import read_json, sha256_file, write_json
from frayid.normal_integrable_sdf import (
    hierarchical_depth_samples,
    neus_interval_weights,
    render_neus_sdf,
)
from frayid.renderer import normalized_boundary_error, soft_silhouette_iou
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.evaluation import inherited_real_gate
from frayid.v2.evidence import EvidenceVolume
from frayid.v2.g02_science import (
    G02_SCIENCE_CHECKPOINT_SCHEMA,
    FrozenEvidenceSDF,
)
from frayid.v2.g02_shortcut_resistant import prepare_shortcut_resistant_field
from frayid.v2.t05_fixed_camera import FixedCameraHumanSolution

G02_ENDPOINT_EVIDENCE_SCHEMA = "frayid_v2_g02_endpoint_evidence.v1"
G02_ENDPOINT_REPORT_SCHEMA = "frayid_v2_g02_independent_endpoint_evaluation.v1"


def _directory_file_hash(root: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        digest.update(name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def prepare_g02_endpoint_evidence(
    scientific_envelope_path: Path,
    checkpoint_path: Path,
    evidence_volume_path: Path,
    manifest_path: Path,
    initialization_path: Path,
    t05_solution_path: Path,
    mask_root: Path,
    normal_root: Path,
    output_path: Path,
    *,
    render_resolution: int = 128,
) -> Path:
    """Bind development evidence only after the scientific endpoint is frozen."""

    paths = [
        scientific_envelope_path,
        checkpoint_path,
        evidence_volume_path,
        manifest_path,
        initialization_path,
        t05_solution_path,
        mask_root,
        normal_root,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G02 endpoint evidence is immutable")
    if render_resolution < 16:
        raise ValueError("G02 endpoint evaluation resolution is too small")
    envelope = read_json(scientific_envelope_path)
    report = envelope.get("training_report", {})
    if (
        envelope.get("status") != "endpoint_frozen_unscored"
        or envelope.get("scientific_attempt") is not True
        or report.get("status") != "endpoint_frozen_unscored"
        or report.get("scientific_attempt_marker_created") is not True
        or report.get("independent_evaluation_pending") is not True
        or envelope.get("training_report_sha256")
        != sha256_file(scientific_envelope_path.with_name("training_report.json"))
        or report.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint_path)
    ):
        raise ValueError("G02 endpoint is not a hash-bound frozen scientific candidate")
    manifest = read_dataset_manifest(manifest_path)
    initialization = load_initialization(initialization_path)
    solution = FixedCameraHumanSolution.model_validate(read_json(t05_solution_path))
    if len(manifest.frames) != 180 or manifest.held_out_frame_count != 36:
        raise ValueError("G02 endpoint evaluator requires the frozen 144/36 split")
    if solution.development_records_used_for_fit != 0 or solution.development_images_read != 0:
        raise ValueError("G02 endpoint evaluator received a T05 solution exposed to development")
    initialization_by_source = {frame.source_frame_index: frame for frame in initialization.frames}
    masks: list[np.ndarray] = []
    development_normals: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    source_indices: list[int] = []
    split_codes: list[int] = []
    names: list[str] = []
    development_names: list[str] = []
    for record in manifest.frames:
        if not record.quality_accepted:
            raise ValueError("G02 endpoint evaluator cannot silently omit a frozen frame")
        frame = initialization_by_source[record.source_frame_index]
        name = Path(record.image_path).name
        mask = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"G02 endpoint mask is absent: {name}")
        masks.append(
            cv2.resize(
                mask,
                (render_resolution, render_resolution),
                interpolation=cv2.INTER_AREA,
            )
        )
        split_codes.append(1 if record.split == "held_out" else 0)
        names.append(name)
        if record.split == "held_out":
            normal = cv2.imread(str(normal_root / name), cv2.IMREAD_COLOR)
            if normal is None:
                raise FileNotFoundError(f"G02 endpoint normal is absent: {name}")
            development_normals.append(
                cv2.resize(
                    normal,
                    (render_resolution, render_resolution),
                    interpolation=cv2.INTER_LINEAR,
                )[..., ::-1].copy()
            )
            development_names.append(name)
        rotations.append(Rotation.from_rotvec(np.asarray(frame.global_orient)).as_matrix())
        translations.append(np.asarray(frame.translation, dtype=np.float32))
        source_indices.append(record.source_frame_index)
    split_array = np.asarray(split_codes, dtype=np.uint8)
    if int((split_array == 0).sum()) != 144 or int((split_array == 1).sum()) != 36:
        raise ValueError("G02 endpoint evaluator split counts changed")
    original_height, original_width = initialization.image_height, initialization.image_width
    original_intrinsics = np.asarray(solution.shared_intrinsics, dtype=np.float64)
    intrinsics = original_intrinsics.copy()
    intrinsics[0] *= render_resolution / original_width
    intrinsics[1] *= render_resolution / original_height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(G02_ENDPOINT_EVIDENCE_SCHEMA),
        masks=np.stack(masks).astype(np.uint8),
        development_normals_rgb=np.stack(development_normals).astype(np.uint8),
        rotations=np.stack(rotations).astype(np.float32),
        translations=np.stack(translations).astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        source_frame_indices=np.asarray(source_indices, dtype=np.int64),
        split_codes=split_array,
        source_revision=np.asarray(str(envelope["source_revision"])),
        attempt_id=np.asarray(str(envelope["attempt_id"])),
        source_hashes=np.asarray(
            json.dumps(
                {
                    "scientific_envelope": sha256_file(scientific_envelope_path),
                    "checkpoint": sha256_file(checkpoint_path),
                    "evidence_volume": sha256_file(evidence_volume_path),
                    "manifest": sha256_file(manifest_path),
                    "initialization": sha256_file(initialization_path),
                    "t05_solution": sha256_file(t05_solution_path),
                    "all_180_masks": _directory_file_hash(mask_root, names),
                    "all_36_development_normals": _directory_file_hash(
                        normal_root, development_names
                    ),
                },
                sort_keys=True,
            )
        ),
        train_records=np.asarray(144, dtype=np.int64),
        development_records=np.asarray(36, dtype=np.int64),
        sealed_test_records=np.asarray(0, dtype=np.int64),
        optimizer_steps=np.asarray(0, dtype=np.int64),
    )
    return output_path


def _rays_for_view(
    intrinsics: Tensor,
    rotation: Tensor,
    translation: Tensor,
    *,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=intrinsics.dtype, device=intrinsics.device),
        torch.arange(width, dtype=intrinsics.dtype, device=intrinsics.device),
        indexing="ij",
    )
    camera = torch.stack(
        (
            (xx - intrinsics[0, 2]) / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(xx),
        ),
        dim=-1,
    ).reshape(-1, 3)
    directions = F.normalize(camera @ rotation, dim=-1)
    origin = (-translation) @ rotation
    return origin.expand_as(directions), directions


def _render_silhouette_only(
    field: nn.Module,
    origins: Tensor,
    directions: Tensor,
    *,
    coarse_samples: int,
    hierarchical_samples: int,
    ray_batch_size: int,
) -> Tensor:
    depths = torch.linspace(
        0.01,
        6.0,
        coarse_samples + 1,
        dtype=origins.dtype,
        device=origins.device,
    )
    silhouettes: list[Tensor] = []
    with torch.no_grad():
        for chunk_origins, chunk_directions in zip(
            origins.split(ray_batch_size),  # type: ignore[no-untyped-call]
            directions.split(ray_batch_size),  # type: ignore[no-untyped-call]
            strict=True,
        ):
            render_depths = depths.expand(len(chunk_origins), -1)
            points = chunk_origins[:, None] + chunk_directions[:, None] * depths[None, :, None]
            weights = neus_interval_weights(field(points), 64.0)
            if hierarchical_samples:
                fine = hierarchical_depth_samples(
                    depths,
                    weights,
                    sample_count=hierarchical_samples,
                )
                render_depths = torch.sort(torch.cat((render_depths, fine), dim=-1), dim=-1)[0]
                points = (
                    chunk_origins[:, None] + chunk_directions[:, None] * render_depths[..., None]
                )
                weights = neus_interval_weights(field(points), 64.0)
            silhouettes.append(weights.sum(dim=-1).clamp(0.0, 1.0))
    return torch.cat(silhouettes)


def _render_view(
    field: nn.Module,
    intrinsics: Tensor,
    rotation: Tensor,
    translation: Tensor,
    *,
    height: int,
    width: int,
    need_normals: bool,
    coarse_samples: int,
    hierarchical_samples: int,
    ray_batch_size: int,
) -> tuple[Tensor, Tensor | None]:
    origins, directions = _rays_for_view(
        intrinsics,
        rotation,
        translation,
        height=height,
        width=width,
    )
    if not need_normals:
        silhouette = _render_silhouette_only(
            field,
            origins,
            directions,
            coarse_samples=coarse_samples,
            hierarchical_samples=hierarchical_samples,
            ray_batch_size=ray_batch_size,
        )
        return silhouette.reshape(height, width), None
    silhouettes: list[Tensor] = []
    normals: list[Tensor] = []
    for chunk_origins, chunk_directions in zip(
        origins.split(ray_batch_size),  # type: ignore[no-untyped-call]
        directions.split(ray_batch_size),  # type: ignore[no-untyped-call]
        strict=True,
    ):
        jacobian = rotation[None, None].expand(len(chunk_origins), 1, 3, 3)
        with torch.enable_grad():
            rendered = render_neus_sdf(
                field,
                chunk_origins,
                chunk_directions,
                near=0.01,
                far=6.0,
                sample_count=coarse_samples,
                hierarchical_sample_count=hierarchical_samples,
                inverse_sharpness=64.0,
                deformation_jacobian=jacobian,
                create_graph=False,
                ray_chunk_size=len(chunk_origins),
            )
        silhouettes.append(rendered.silhouette.detach())
        normals.append(rendered.normals.detach())
    return (
        torch.cat(silhouettes).reshape(height, width),
        torch.cat(normals).reshape(height, width, 3),
    )


def _aggregate_arm(
    field: nn.Module,
    masks: Tensor,
    development_normals: Tensor,
    rotations: Tensor,
    translations: Tensor,
    intrinsics: Tensor,
    split_codes: Tensor,
    *,
    coarse_samples: int,
    hierarchical_samples: int,
    ray_batch_size: int,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    height, width = int(masks.shape[1]), int(masks.shape[2])
    train_ious: list[float] = []
    development_ious: list[float] = []
    development_boundaries: list[float] = []
    pooled_normal_errors: list[np.ndarray] = []
    per_frame: list[dict[str, float | int | str]] = []
    development_slot = 0
    for frame_slot in range(len(masks)):
        is_development = bool(split_codes[frame_slot])
        silhouette, normals = _render_view(
            field,
            intrinsics,
            rotations[frame_slot],
            translations[frame_slot],
            height=height,
            width=width,
            need_normals=is_development,
            coarse_samples=coarse_samples,
            hierarchical_samples=hierarchical_samples,
            ray_batch_size=ray_batch_size,
        )
        target = masks[frame_slot]
        iou = float(soft_silhouette_iou(silhouette, target))
        frame_report: dict[str, float | int | str] = {
            "frame_slot": frame_slot,
            "split": "held_out" if is_development else "train",
            "silhouette_iou": iou,
        }
        if not is_development:
            train_ious.append(iou)
        else:
            development_ious.append(iou)
            boundary = normalized_boundary_error(silhouette, target)
            development_boundaries.append(boundary)
            assert normals is not None
            target_normal = development_normals[development_slot]
            valid = target > 0.5
            cosine = (
                F.normalize(normals[valid], dim=-1, eps=1.0e-8)
                * F.normalize(target_normal[valid], dim=-1, eps=1.0e-8)
            ).sum(dim=-1)
            errors = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).cpu().numpy()
            if not len(errors):
                raise ValueError("G02 endpoint evaluator found an empty held-out mask")
            pooled_normal_errors.append(errors)
            frame_report.update(
                normalized_boundary_error=boundary,
                median_normal_error_degrees=float(np.median(errors)),
            )
            development_slot += 1
        per_frame.append(frame_report)
    pooled = np.concatenate(pooled_normal_errors)
    metrics = {
        "train_iou": float(np.median(train_ious)),
        "held_out_iou": float(np.median(development_ious)),
        "normalized_boundary_error": float(np.median(development_boundaries)),
        "median_normal_error_degrees": float(np.median(pooled)),
        "train_held_out_iou_gap": float(np.median(train_ious) - np.median(development_ious)),
    }
    return metrics, per_frame


def evaluate_g02_frozen_endpoint(
    checkpoint_path: Path,
    evidence_volume_path: Path,
    endpoint_evidence_path: Path,
    output_path: Path,
    *,
    device: torch.device | str,
    coarse_samples: int = 24,
    hierarchical_samples: int = 8,
    ray_batch_size: int = 512,
) -> Path:
    """Score frozen treatment/control fields without optimizer or training access."""

    paths = [checkpoint_path, evidence_volume_path, endpoint_evidence_path, output_path]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G02 endpoint evaluation report is immutable")
    target_device = torch.device(device)
    with np.load(endpoint_evidence_path, allow_pickle=False) as archive:
        if str(archive["schema_version"]) != G02_ENDPOINT_EVIDENCE_SCHEMA:
            raise ValueError("G02 endpoint evidence schema is invalid")
        if int(archive["sealed_test_records"]) != 0 or int(archive["optimizer_steps"]) != 0:
            raise ValueError("G02 endpoint evidence crosses the evaluation boundary")
        masks = torch.as_tensor(archive["masks"].copy(), device=target_device).float() / 255.0
        development_normals = (
            torch.as_tensor(archive["development_normals_rgb"].copy(), device=target_device).float()
            / 127.5
            - 1.0
        )
        rotations = torch.as_tensor(archive["rotations"].copy(), device=target_device)
        translations = torch.as_tensor(archive["translations"].copy(), device=target_device)
        intrinsics = torch.as_tensor(archive["intrinsics"].copy(), device=target_device)
        split_codes = torch.as_tensor(archive["split_codes"].copy(), device=target_device)
        source_indices = archive["source_frame_indices"].astype(np.int64)
        source_revision = str(archive["source_revision"])
        attempt_id = str(archive["attempt_id"])
        source_hashes = json.loads(str(archive["source_hashes"]))
    if masks.shape[0] != 180 or int((split_codes == 1).sum()) != 36:
        raise ValueError("G02 endpoint evidence does not contain the frozen 180 frames")
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    if checkpoint.get("schema_version") != G02_SCIENCE_CHECKPOINT_SCHEMA:
        raise ValueError("G02 endpoint checkpoint schema is invalid")
    if checkpoint.get("completed_steps") != 600:
        raise ValueError("G02 endpoint checkpoint is not the completed treatment")
    # Science checkpoints persist the complete registered arm under the
    # immutable key used by the capture/restore path.  Keep the evaluator
    # bound to that original training revision; the evaluator's own revision
    # is recorded separately in the evaluation envelope.
    arm = checkpoint.get("immutable_arm_binding", {})
    if (
        arm.get("attempt_id") != attempt_id
        or arm.get("common", {}).get("source_revision") != source_revision
    ):
        raise ValueError("G02 endpoint checkpoint identity does not match evaluation evidence")
    evidence = EvidenceVolume.load(evidence_volume_path, device=target_device)
    treatment = prepare_shortcut_resistant_field(evidence, seed=20260903).to(target_device)
    treatment.load_state_dict(checkpoint["model_state"], strict=True)
    treatment.eval()
    control = FrozenEvidenceSDF(evidence).to(target_device)
    control.eval()
    for field in (treatment, control):
        for parameter in field.parameters():
            parameter.requires_grad_(False)
    treatment_metrics, treatment_frames = _aggregate_arm(
        treatment,
        masks,
        development_normals,
        rotations,
        translations,
        intrinsics,
        split_codes,
        coarse_samples=coarse_samples,
        hierarchical_samples=hierarchical_samples,
        ray_batch_size=ray_batch_size,
    )
    control_metrics, control_frames = _aggregate_arm(
        control,
        masks,
        development_normals,
        rotations,
        translations,
        intrinsics,
        split_codes,
        coarse_samples=coarse_samples,
        hierarchical_samples=hierarchical_samples,
        ray_batch_size=ray_batch_size,
    )
    blockers = inherited_real_gate(treatment_metrics)
    comparisons = {
        "held_out_iou_not_worse_than_control": treatment_metrics["held_out_iou"]
        >= control_metrics["held_out_iou"],
        "boundary_not_worse_than_control": treatment_metrics["normalized_boundary_error"]
        <= control_metrics["normalized_boundary_error"],
        "normal_not_worse_than_control": treatment_metrics["median_normal_error_degrees"]
        <= control_metrics["median_normal_error_degrees"],
    }
    blockers.extend(name for name, passed in comparisons.items() if not passed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return write_json(
        output_path,
        {
            "schema_version": G02_ENDPOINT_REPORT_SCHEMA,
            "status": "pass" if not blockers else "fail",
            "experiment_id": "postv2_g02_direct_multires_field_matched_science_r01",
            "attempt_id": attempt_id,
            "training_source_revision": source_revision,
            "device": str(target_device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if target_device.type == "cuda" else None
            ),
            "render_resolution": int(masks.shape[-1]),
            "coarse_samples": coarse_samples,
            "hierarchical_samples": hierarchical_samples,
            "treatment": treatment_metrics,
            "frozen_control": control_metrics,
            "treatment_minus_control": {
                key: treatment_metrics[key] - control_metrics[key] for key in treatment_metrics
            },
            "matched_comparison_gates": comparisons,
            "inherited_real_gate_blockers": inherited_real_gate(treatment_metrics),
            "per_frame": [
                {
                    "source_frame_index": int(source),
                    "treatment": treatment_frame,
                    "control": control_frame,
                }
                for source, treatment_frame, control_frame in zip(
                    source_indices.tolist(), treatment_frames, control_frames, strict=True
                )
            ],
            "source_hashes": {
                **source_hashes,
                "endpoint_evidence": sha256_file(endpoint_evidence_path),
                "checkpoint": sha256_file(checkpoint_path),
                "evidence_volume": sha256_file(evidence_volume_path),
            },
            "training_records_read": 144,
            "development_records_read": 36,
            "development_records_used_for_fit": 0,
            "optimizer_steps": 0,
            "automatic_retries": 0,
            "sealed_test_accesses": 0,
            "topology_state": "search_not_committed",
            "topology_audit_pending": True,
            "authoritative_result_claimed": False,
            "blockers": blockers,
        },
    )


def endpoint_gate_sensitivity() -> dict[str, bool]:
    passing = {
        "held_out_iou": 0.90,
        "normalized_boundary_error": 0.004,
        "median_normal_error_degrees": 20.0,
        "train_held_out_iou_gap": 0.01,
    }
    return {
        "passing_fixture": not inherited_real_gate(passing),
        "iou_failure": "held_out_iou_below_r03"
        in inherited_real_gate({**passing, "held_out_iou": 0.80}),
        "boundary_failure": "boundary_worse_than_r03"
        in inherited_real_gate({**passing, "normalized_boundary_error": 0.01}),
        "normal_failure": "normal_worse_than_r03"
        in inherited_real_gate({**passing, "median_normal_error_degrees": 30.0}),
        "gap_failure": "train_held_out_gap"
        in inherited_real_gate({**passing, "train_held_out_iou_gap": 0.10}),
        "finite": all(math.isfinite(value) for value in passing.values()),
    }
