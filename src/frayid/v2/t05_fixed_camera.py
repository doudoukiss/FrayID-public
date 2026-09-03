from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from frayid.camera import make_intrinsics
from frayid.dataset import read_dataset_manifest
from frayid.geometry import linear_blend_skinning
from frayid.io import read_json, sha256_file, write_json
from frayid.renderer import normalized_boundary_error, render_soft_mesh, soft_silhouette_iou
from frayid.schemas import SequenceInitialization
from frayid.v2.contracts import QualificationState, advance_qualification, reject_sealed_capability
from frayid.v2.turntable_ba import (
    make_synthetic_turntable_problem,
    turntable_identifiability_benchmark,
)
from frayid.v2.video_forensics import camera_verdict, estimate_background_transforms

T05_EXPERIMENT_ID = "postv2_t05_background_anchored_fixed_camera_human_ba_r01"
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class T05FrameState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_frame_index: int = Field(ge=0)
    yaw_radians: float
    observed_yaw_radians: float
    yaw_confidence: float = Field(ge=0.0, le=1.0)
    root_translation_metres: list[float] = Field(min_length=3, max_length=3)
    low_frequency_root_translation_metres: list[float] = Field(min_length=3, max_length=3)
    root_residual_translation_metres: list[float] = Field(min_length=3, max_length=3)
    observed_global_orient_rotvec: list[float] = Field(min_length=3, max_length=3)
    residual_rotation_rotvec: list[float] = Field(min_length=3, max_length=3)
    body_pose: list[float]
    body_pose_source: Literal["frozen_camerahmr_smpl_scaffold"] = "frozen_camerahmr_smpl_scaffold"


class FixedCameraHumanSolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["frayid_v2_t05_fixed_camera_human_solution.v1"] = (
        "frayid_v2_t05_fixed_camera_human_solution.v1"
    )
    experiment_id: Literal["postv2_t05_background_anchored_fixed_camera_human_ba_r01"] = (
        "postv2_t05_background_anchored_fixed_camera_human_ba_r01"
    )
    status: Literal["qualification_candidate"] = "qualification_candidate"
    source_revision: str
    shared_intrinsics: list[list[float]]
    distortion_coefficients: list[float]
    physical_camera_rotation: list[list[float]]
    physical_camera_translation: list[float]
    spin_axis_camera: list[float]
    root_center_camera_metres: list[float]
    base_orientation_rotvec: list[float]
    micromotion_basis: list[list[float]]
    micromotion_codes: list[list[float]]
    micromotion_retained_variance: float = Field(ge=0.0, le=1.0)
    frames: list[T05FrameState]
    gauge_policy: dict[str, str]
    uncertainty: dict[str, float]
    provenance: dict[str, str]
    source_hashes: dict[str, str]
    training_frame_count: int = Field(gt=0)
    development_records_used_for_fit: Literal[0] = 0
    development_images_read: Literal[0] = 0
    sealed_test_reads: Literal[0] = 0
    optimizer_steps: Literal[0] = 0
    paid_jobs: Literal[0] = 0


def weighted_isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators projection onto nondecreasing values."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or len(values) < 2:
        raise ValueError("weighted isotonic input must be aligned nontrivial vectors")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("weighted isotonic input must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("weighted isotonic weights must be positive")
    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([index, index + 1, float(weight), float(value)])
        while len(blocks) >= 2 and float(blocks[-2][3]) > float(blocks[-1][3]):
            right = blocks.pop()
            left = blocks.pop()
            total = float(left[2]) + float(right[2])
            mean = (float(left[2]) * float(left[3]) + float(right[2]) * float(right[3])) / total
            blocks.append([int(left[0]), int(right[1]), total, mean])
    result = np.empty_like(values)
    for start, end, _, value in blocks:
        result[int(start) : int(end)] = float(value)
    return result


def decompose_fixed_camera_human_motion(
    rotations: np.ndarray,
    translations: np.ndarray,
    source_indices: np.ndarray,
    weights: np.ndarray,
    *,
    low_frequency_degree: int = 7,
) -> dict[str, np.ndarray]:
    """Assign the shared camera no motion and decompose all SE(3) motion onto the human."""
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    source_indices = np.asarray(source_indices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    frame_count = len(source_indices)
    if rotations.shape != (frame_count, 3, 3) or translations.shape != (frame_count, 3):
        raise ValueError("T05 rotation and translation shapes are invalid")
    if weights.shape != (frame_count,) or frame_count < 8:
        raise ValueError("T05 decomposition requires aligned weights and at least eight frames")
    right = np.einsum("tij,j->ti", rotations, np.asarray([1.0, 0.0, 0.0]))
    observed_yaw = np.unwrap(np.arctan2(right[:, 2], right[:, 0]))
    direction = 1.0 if observed_yaw[-1] >= observed_yaw[0] else -1.0
    directed = direction * observed_yaw
    monotonic = weighted_isotonic(directed, weights)
    yaw = direction * monotonic
    relative = np.einsum("ij,tjk->tik", rotations[0].T, rotations)
    relative_vectors = Rotation.from_matrix(relative).as_rotvec()
    magnitudes = np.linalg.norm(relative_vectors, axis=1)
    useful = magnitudes > math.radians(10.0)
    axes = relative_vectors[useful] / magnitudes[useful, None]
    axis_weights = weights[useful] * np.minimum(magnitudes[useful], math.pi)
    axes[axes[:, 1] < 0.0] *= -1.0
    spin_axis = np.average(axes, axis=0, weights=axis_weights)
    spin_axis /= np.linalg.norm(spin_axis)
    base = Rotation.from_matrix(rotations[0])
    spin = Rotation.from_rotvec((yaw - yaw[0])[:, None] * spin_axis[None])
    structured = base * spin
    residual = structured.inv() * Rotation.from_matrix(rotations)
    residual_vectors = residual.as_rotvec()
    normalized_time = (
        2.0
        * (source_indices - source_indices.min())
        / (source_indices.max() - source_indices.min())
        - 1.0
    )
    degree = min(low_frequency_degree, frame_count - 2)
    basis = np.polynomial.legendre.legvander(normalized_time, degree)
    weighted_basis = basis * np.sqrt(weights)[:, None]
    low_frequency = np.empty_like(translations)
    for dimension in range(3):
        target = translations[:, dimension] * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(weighted_basis, target, rcond=None)
        low_frequency[:, dimension] = basis @ coefficients
    root_residual = translations - low_frequency
    residual_mean = np.average(root_residual, axis=0, weights=weights)
    low_frequency += residual_mean[None]
    root_residual -= residual_mean[None]
    return {
        "yaw": yaw,
        "observed_yaw": observed_yaw,
        "spin_axis": spin_axis,
        "base_rotation_vector": base.as_rotvec(),
        "residual_rotation_vectors": residual_vectors,
        "low_frequency_translation": low_frequency,
        "root_residual_translation": root_residual,
        "reconstructed_rotations": (
            structured * Rotation.from_rotvec(residual_vectors)
        ).as_matrix(),
        "reconstructed_translations": low_frequency + root_residual,
    }


def fit_t05_fixed_camera_solution(
    initialization_path: Path,
    manifest_path: Path,
    v00_lifecycle_path: Path,
    output_path: Path,
    *,
    source_revision: str,
    t04_solution_path: Path | None = None,
    micromotion_rank: int = 4,
) -> Path:
    paths = [initialization_path, manifest_path, v00_lifecycle_path, output_path]
    if t04_solution_path is not None:
        paths.append(t04_solution_path)
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("T05 solution is immutable")
    if _REVISION_PATTERN.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be a full lowercase Git commit")
    v00_lifecycle = read_json(v00_lifecycle_path)
    if v00_lifecycle.get("status") != "pass" or v00_lifecycle.get("state") != "qualified":
        raise ValueError("T05 requires a passing qualified V00 lifecycle")
    initialization = SequenceInitialization.model_validate(read_json(initialization_path))
    manifest = read_dataset_manifest(manifest_path)
    initialization_by_source = {frame.source_frame_index: frame for frame in initialization.frames}
    training = [
        frame for frame in manifest.frames if frame.split == "train" and frame.quality_accepted
    ]
    if len(training) != manifest.train_frame_count or len(training) != 144:
        raise ValueError("T05 requires the complete frozen 144-frame training split")
    frames = [initialization_by_source[record.source_frame_index] for record in training]
    source_indices = np.asarray([frame.source_frame_index for frame in frames], dtype=np.float64)
    rotations = Rotation.from_rotvec(
        np.asarray([frame.global_orient for frame in frames], dtype=np.float64)
    ).as_matrix()
    translations = np.asarray([frame.translation for frame in frames], dtype=np.float64)
    t04_confidence: dict[int, float] = {}
    if t04_solution_path is not None:
        t04_payload = read_json(t04_solution_path)
        t04_rows = t04_payload.get("frames")
        if not isinstance(t04_rows, list):
            raise ValueError("T04 solution has no frames")
        t04_confidence = {
            int(row["source_frame_index"]): float(row["confidence"])
            for row in t04_rows
            if isinstance(row, dict)
        }
    weights = np.asarray(
        [
            max(1.0e-3, frame.detection_score)
            * max(1.0e-3, t04_confidence.get(frame.source_frame_index, 1.0))
            for frame in frames
        ],
        dtype=np.float64,
    )
    decomposition = decompose_fixed_camera_human_motion(
        rotations,
        translations,
        source_indices,
        weights,
    )
    body_pose = np.asarray([frame.body_pose for frame in frames], dtype=np.float64)
    centered_pose = body_pose - np.average(body_pose, axis=0, weights=weights)
    _, singular, right = np.linalg.svd(centered_pose, full_matrices=False)
    rank = min(micromotion_rank, len(right))
    micromotion_basis = right[:rank]
    micromotion_codes = centered_pose @ micromotion_basis.T
    total = float(np.square(singular).sum())
    retained = float(np.square(singular[:rank]).sum() / total) if total > 0.0 else 1.0
    root_center = np.average(translations, axis=0, weights=weights)
    focal = initialization.shared_focal_length_px
    cx, cy = initialization.shared_principal_point_px
    solution = FixedCameraHumanSolution(
        source_revision=source_revision,
        shared_intrinsics=[[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        distortion_coefficients=[0.0] * 5,
        physical_camera_rotation=np.eye(3).tolist(),
        physical_camera_translation=[0.0, 0.0, 0.0],
        spin_axis_camera=decomposition["spin_axis"].tolist(),
        root_center_camera_metres=root_center.tolist(),
        base_orientation_rotvec=decomposition["base_rotation_vector"].tolist(),
        micromotion_basis=micromotion_basis.tolist(),
        micromotion_codes=micromotion_codes.tolist(),
        micromotion_retained_variance=retained,
        frames=[
            T05FrameState(
                source_frame_index=frame.source_frame_index,
                yaw_radians=float(decomposition["yaw"][slot]),
                observed_yaw_radians=float(decomposition["observed_yaw"][slot]),
                yaw_confidence=float(np.clip(weights[slot], 0.0, 1.0)),
                root_translation_metres=translations[slot].tolist(),
                low_frequency_root_translation_metres=decomposition["low_frequency_translation"][
                    slot
                ].tolist(),
                root_residual_translation_metres=decomposition["root_residual_translation"][
                    slot
                ].tolist(),
                observed_global_orient_rotvec=list(frame.global_orient),
                residual_rotation_rotvec=decomposition["residual_rotation_vectors"][slot].tolist(),
                body_pose=list(frame.body_pose),
            )
            for slot, frame in enumerate(frames)
        ],
        gauge_policy={
            "physical_camera": "identity_world_to_camera_fixed_by_static_background",
            "yaw_origin": "first_training_frame",
            "yaw_direction": "chosen_once_then_weighted_isotonic",
            "spin_axis_sign": "positive_camera_vertical_component",
            "root_residual_mean": "confidence_weighted_zero",
            "micromotion": "zero_mean_orthonormal_svd_basis",
            "scale_and_intrinsics": "frozen_camerahmr_scaffold_prior_for_qualification",
        },
        uncertainty={
            "median_t04_confidence": float(np.median(list(t04_confidence.values())))
            if t04_confidence
            else 1.0,
            "root_residual_rms_metres": float(
                np.sqrt(np.mean(np.square(decomposition["root_residual_translation"])))
            ),
            "rotation_residual_median_degrees": float(
                np.degrees(
                    np.median(np.linalg.norm(decomposition["residual_rotation_vectors"], axis=1))
                )
            ),
        },
        provenance={
            "camera": "v00_static_background_fixed_physical_camera",
            "root_pose": "frozen_camerahmr_reowned_as_human_initialization",
            "pose": "frozen_camerahmr_smpl_scaffold",
            "uncertainty": "t04_weak_prior_only" if t04_solution_path else "unit_confidence",
            "cloth_cycle": "not_required",
        },
        source_hashes={
            "initialization": sha256_file(initialization_path),
            "manifest": sha256_file(manifest_path),
            "v00_lifecycle": sha256_file(v00_lifecycle_path),
            **(
                {"t04_solution": sha256_file(t04_solution_path)}
                if t04_solution_path is not None
                else {}
            ),
        },
        training_frame_count=len(frames),
    )
    return write_json(output_path, solution)


def reconstruct_t05_rotations(solution: FixedCameraHumanSolution) -> np.ndarray:
    base = Rotation.from_rotvec(np.asarray(solution.base_orientation_rotvec, dtype=np.float64))
    axis = np.asarray(solution.spin_axis_camera, dtype=np.float64)
    yaw = np.asarray([frame.yaw_radians for frame in solution.frames], dtype=np.float64)
    spin = Rotation.from_rotvec((yaw - yaw[0])[:, None] * axis[None])
    residual = Rotation.from_rotvec(
        np.asarray([frame.residual_rotation_rotvec for frame in solution.frames])
    )
    return np.asarray((base * spin * residual).as_matrix(), dtype=np.float64)


def _synthetic_fixed_camera_recovery(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    frame_count = 36
    true_axis = np.asarray([0.014, 1.0, -0.011], dtype=np.float64)
    true_axis /= np.linalg.norm(true_axis)
    increments = np.linspace(0.155, 0.205, frame_count - 1)
    true_yaw = np.concatenate((np.zeros(1), np.cumsum(increments)))
    base = Rotation.from_euler("xyz", [math.pi - 0.04, 0.03, -0.02])
    phase = np.linspace(0.0, 2.0 * math.pi, frame_count, endpoint=False)
    residual_vectors = np.stack(
        (0.006 * np.sin(phase), 0.004 * np.cos(phase), 0.005 * np.sin(2.0 * phase)),
        axis=1,
    )
    rotations = (
        base
        * Rotation.from_rotvec(true_yaw[:, None] * true_axis[None])
        * Rotation.from_rotvec(residual_vectors)
    ).as_matrix()
    true_center = np.asarray([0.04, 0.16, 2.55], dtype=np.float64)
    translation_motion = np.stack(
        (0.018 * np.sin(phase), 0.012 * np.cos(phase), 0.010 * np.sin(2.0 * phase)),
        axis=1,
    )
    translations = true_center + translation_motion
    source_indices = np.arange(frame_count, dtype=np.float64) * 5.0
    weights = np.ones(frame_count, dtype=np.float64)
    decomposition = decompose_fixed_camera_human_motion(
        rotations,
        translations,
        source_indices,
        weights,
    )
    local_points = rng.normal(size=(48, 3)) * np.asarray([0.26, 0.75, 0.18])
    camera_points = np.einsum("tij,pj->tpi", rotations, local_points) + translations[:, None]
    true_focal = 910.0
    principal = np.asarray([360.0, 560.0])
    normalized = camera_points[..., :2] / camera_points[..., 2:3]
    pixels = normalized * true_focal + principal
    pixels += rng.normal(scale=0.12, size=pixels.shape)
    a = normalized.reshape(-1)
    b = (pixels - principal).reshape(-1)
    recovered_focal = float(a @ b / (a @ a))
    recovered_axis = decomposition["spin_axis"]
    axis_error = math.degrees(math.acos(float(np.clip(recovered_axis @ true_axis, -1.0, 1.0))))
    recovered_yaw = decomposition["yaw"] - decomposition["yaw"][0]
    yaw_error = np.degrees(np.abs(recovered_yaw - true_yaw))
    recovered_center = np.average(translations, axis=0, weights=weights)
    subject_diagonal = float(np.linalg.norm(np.ptp(local_points, axis=0)))
    center_error = float(np.linalg.norm(recovered_center - true_center) / subject_diagonal)
    rotation_replay_error = Rotation.from_matrix(
        np.einsum("tji,tjk->tik", decomposition["reconstructed_rotations"], rotations)
    ).magnitude()
    return {
        "true_axis": true_axis.tolist(),
        "recovered_axis": recovered_axis.tolist(),
        "axis_error_degrees": axis_error,
        "center_error_fraction_subject_diagonal": center_error,
        "median_yaw_error_degrees": float(np.median(yaw_error)),
        "maximum_yaw_error_degrees": float(np.max(yaw_error)),
        "focal_error_fraction": abs(recovered_focal - true_focal) / true_focal,
        "maximum_rotation_replay_error_degrees": float(np.degrees(rotation_replay_error.max())),
        "maximum_translation_replay_error_metres": float(
            np.max(np.abs(decomposition["reconstructed_translations"] - translations))
        ),
    }


def write_t05_public_benchmark(output_path: Path, *, seed: int = 20260903) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("T05 public benchmark is immutable")
    fit = _synthetic_fixed_camera_recovery(seed)
    identifiability = turntable_identifiability_benchmark(make_synthetic_turntable_problem(seed))
    gates = {
        "axis_error_degrees_at_most_2": float(fit["axis_error_degrees"]) <= 2.0,
        "center_error_fraction_at_most_0_02": float(fit["center_error_fraction_subject_diagonal"])
        <= 0.02,
        "median_yaw_error_degrees_at_most_2": float(fit["median_yaw_error_degrees"]) <= 2.0,
        "focal_error_fraction_at_most_0_02": float(fit["focal_error_fraction"]) <= 0.02,
        "rotation_replay_at_most_1e_8_degrees": float(fit["maximum_rotation_replay_error_degrees"])
        <= 1.0e-8,
        "translation_replay_at_most_1e_10_metres": float(
            fit["maximum_translation_replay_error_metres"]
        )
        <= 1.0e-10,
        "geometry_nuisance_correlation_drop_at_least_0_05": float(
            identifiability["geometry_nuisance_correlation_drop"]
        )
        >= 0.05,
        "smallest_informative_schur_eigenvalue_rise_at_least_0_25": float(
            identifiability["smallest_informative_schur_eigenvalue_rise_fraction"]
        )
        >= 0.25,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_t05_public_benchmark.v1",
            "experiment_id": T05_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "seed": seed,
            "fit": fit,
            "identifiability": identifiability,
            "gates": gates,
            "blockers": blockers,
            "private_reads": 0,
            "development_reads": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "attempt_marker_created": False,
        },
    )


def audit_t05_training_background(
    evidence_master_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    sample_count: int = 24,
) -> Path:
    reject_sealed_capability([evidence_master_path, manifest_path, output_path])
    if output_path.exists():
        raise FileExistsError("T05 background audit is immutable")
    evidence = read_json(evidence_master_path)
    manifest = read_dataset_manifest(manifest_path)
    training = [
        frame for frame in manifest.frames if frame.split == "train" and frame.quality_accepted
    ]
    if len(training) != 144:
        raise ValueError("T05 background audit requires all 144 frozen training frames")
    slots = np.linspace(0, len(training) - 1, min(sample_count, len(training)), dtype=np.int64)
    selected = [training[int(slot)].source_frame_index for slot in slots]
    rows = evidence.get("frames")
    if not isinstance(rows, list):
        raise ValueError("V00 evidence master has no frames")
    row_by_source = {int(row["source_frame_index"]): row for row in rows if isinstance(row, dict)}
    root = evidence_master_path.parent
    images: list[np.ndarray] = []
    for source in selected:
        row = row_by_source[source]
        path = root / str(row["lossless_frame_path"])
        if sha256_file(path) != row["lossless_frame_sha256"]:
            raise ValueError(f"T05 background frame hash mismatch: {source}")
        images.append(np.asarray(Image.open(path).convert("RGB")))
    first = estimate_background_transforms(images, source_indices=selected)
    second = estimate_background_transforms(images, source_indices=selected)
    repeatable = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    median_reprojection = float(first["summary"]["median_reprojection_error_pixels"])
    verdict = camera_verdict(first)
    gates = {
        "training_only_background_audit_repeatable": repeatable,
        "median_background_reprojection_at_most_0_3_pixels": median_reprojection <= 0.3,
        "fixed_physical_camera_verdict": verdict == "fixed_to_subpixel_precision",
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_t05_training_background_audit.v1",
            "experiment_id": T05_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "source_frame_indices": selected,
            "background_audit": first,
            "physical_camera_verdict": verdict,
            "gates": gates,
            "blockers": blockers,
            "input_hashes": {
                "evidence_master": sha256_file(evidence_master_path),
                "manifest": sha256_file(manifest_path),
            },
            "training_images_read": len(images),
            "development_images_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
        },
    )


def evaluate_t05_development_nonregression(
    solution_path: Path,
    initialization_path: Path,
    manifest_path: Path,
    mask_root: Path,
    canonical_mesh_path: Path,
    skinning_weights_path: Path,
    joint_transforms_path: Path,
    output_path: Path,
    *,
    source_image_size: tuple[int, int] = (1120, 720),
    render_resolution: int = 64,
    seed: int = 20260903,
) -> Path:
    """Verify that fixed-camera reownership preserves the frozen held-out scaffold."""
    paths = [
        solution_path,
        initialization_path,
        manifest_path,
        mask_root,
        canonical_mesh_path,
        skinning_weights_path,
        joint_transforms_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("T05 development evaluation is immutable")
    solution = FixedCameraHumanSolution.model_validate(read_json(solution_path))
    initialization = SequenceInitialization.model_validate(read_json(initialization_path))
    initialization_by_source = {frame.source_frame_index: frame for frame in initialization.frames}
    manifest = read_dataset_manifest(manifest_path)
    development = [
        frame for frame in manifest.frames if frame.split == "held_out" and frame.quality_accepted
    ]
    if len(development) != 36 or len(development) != manifest.held_out_frame_count:
        raise ValueError("T05 evaluator requires the complete frozen 36-frame development split")
    with np.load(canonical_mesh_path, allow_pickle=False) as archive:
        canonical_vertices = torch.as_tensor(archive["vertices"].copy(), dtype=torch.float32)
        faces = torch.as_tensor(archive["faces"].copy(), dtype=torch.long)
    with np.load(skinning_weights_path, allow_pickle=False) as archive:
        weights = torch.as_tensor(archive["weights"].copy(), dtype=torch.float32)
    with np.load(joint_transforms_path, allow_pickle=False) as archive:
        transform_sources = archive["source_frame_indices"].astype(np.int64)
        transforms = torch.as_tensor(archive["transforms"].copy(), dtype=torch.float32)
    transform_slot = {int(source): slot for slot, source in enumerate(transform_sources)}
    train_sources = np.asarray(
        [frame.source_frame_index for frame in solution.frames], dtype=np.float64
    )
    train_yaw = np.asarray([frame.yaw_radians for frame in solution.frames], dtype=np.float64)
    base = Rotation.from_rotvec(np.asarray(solution.base_orientation_rotvec, dtype=np.float64))
    axis = np.asarray(solution.spin_axis_camera, dtype=np.float64)
    intrinsics = make_intrinsics(
        float(solution.shared_intrinsics[0][0]),
        (float(solution.shared_intrinsics[0][2]), float(solution.shared_intrinsics[1][2])),
    )
    treatment_ious: list[float] = []
    treatment_boundaries: list[float] = []
    control_ious: list[float] = []
    control_boundaries: list[float] = []
    vertex_replay_errors: list[float] = []
    per_frame: list[dict[str, float | int]] = []
    for slot, record in enumerate(development):
        source = record.source_frame_index
        frame = initialization_by_source[source]
        observed = Rotation.from_rotvec(np.asarray(frame.global_orient, dtype=np.float64))
        observed_matrix = observed.as_matrix()
        right = observed_matrix @ np.asarray([1.0, 0.0, 0.0])
        wrapped_yaw = math.atan2(float(right[2]), float(right[0]))
        expected_yaw = float(np.interp(source, train_sources, train_yaw))
        query_yaw = wrapped_yaw + 2.0 * math.pi * round(
            (expected_yaw - wrapped_yaw) / (2.0 * math.pi)
        )
        structured = base * Rotation.from_rotvec((query_yaw - train_yaw[0]) * axis)
        residual = structured.inv() * observed
        reconstructed = structured * residual
        reconstructed_matrix = reconstructed.as_matrix()
        posed_camera = linear_blend_skinning(
            canonical_vertices,
            weights,
            transforms[transform_slot[source]],
        )
        observed_rotation = torch.as_tensor(observed_matrix, dtype=torch.float32)
        reconstructed_rotation = torch.as_tensor(reconstructed_matrix, dtype=torch.float32)
        translation = torch.as_tensor(frame.translation, dtype=torch.float32)
        local_pose = (posed_camera - translation) @ observed_rotation
        treatment_vertices = local_pose @ reconstructed_rotation.T + translation
        vertex_error = float(torch.max(torch.abs(treatment_vertices - posed_camera)))
        vertex_replay_errors.append(vertex_error)
        mask_path = mask_root / Path(record.image_path).name
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"T05 development mask is absent: {mask_path}")
        resized = cv2.resize(
            raw_mask,
            (render_resolution, render_resolution),
            interpolation=cv2.INTER_AREA,
        )
        target = torch.as_tensor(resized / 255.0, dtype=torch.float32)
        torch.manual_seed(seed + slot)
        treatment_silhouette, _ = render_soft_mesh(
            treatment_vertices,
            faces,
            intrinsics,
            (render_resolution, render_resolution),
            source_image_size=source_image_size,
            sigma_pixels=1.75 * render_resolution / 128.0,
            sample_count=2048,
            reference_sample_count=2048,
        )
        torch.manual_seed(seed + slot)
        control_silhouette, _ = render_soft_mesh(
            posed_camera,
            faces,
            intrinsics,
            (render_resolution, render_resolution),
            source_image_size=source_image_size,
            sigma_pixels=1.75 * render_resolution / 128.0,
            sample_count=2048,
            reference_sample_count=2048,
        )
        treatment_iou = float(soft_silhouette_iou(treatment_silhouette, target))
        control_iou = float(soft_silhouette_iou(control_silhouette, target))
        treatment_boundary = normalized_boundary_error(treatment_silhouette, target)
        control_boundary = normalized_boundary_error(control_silhouette, target)
        treatment_ious.append(treatment_iou)
        treatment_boundaries.append(treatment_boundary)
        control_ious.append(control_iou)
        control_boundaries.append(control_boundary)
        per_frame.append(
            {
                "source_frame_index": source,
                "query_yaw_radians": query_yaw,
                "vertex_replay_maximum_metres": vertex_error,
                "treatment_iou": treatment_iou,
                "control_iou": control_iou,
                "treatment_boundary": treatment_boundary,
                "control_boundary": control_boundary,
            }
        )
    treatment = {
        "mean_iou": float(np.mean(treatment_ious)),
        "mean_normalized_boundary_error": float(np.mean(treatment_boundaries)),
    }
    control = {
        "mean_iou": float(np.mean(control_ious)),
        "mean_normalized_boundary_error": float(np.mean(control_boundaries)),
    }
    iou_change = treatment["mean_iou"] - control["mean_iou"]
    boundary_change = (
        treatment["mean_normalized_boundary_error"] - control["mean_normalized_boundary_error"]
    )
    gates = {
        "held_out_iou_may_regress_at_most_0_002": iou_change >= -0.002,
        "held_out_boundary_may_not_regress": boundary_change <= 1.0e-12,
        "fixed_camera_reownership_vertex_replay_at_most_1e_6_metres": max(vertex_replay_errors)
        <= 1.0e-6,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_t05_development_nonregression.v1",
            "experiment_id": T05_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "device": "cpu",
            "render_resolution": render_resolution,
            "development_frame_count": len(development),
            "treatment": treatment,
            "matched_free_root_control": control,
            "treatment_minus_control_iou": iou_change,
            "treatment_minus_control_boundary": boundary_change,
            "maximum_vertex_replay_error_metres": max(vertex_replay_errors),
            "gates": gates,
            "blockers": blockers,
            "per_frame": per_frame,
            "input_hashes": {
                "solution": sha256_file(solution_path),
                "initialization": sha256_file(initialization_path),
                "manifest": sha256_file(manifest_path),
                "canonical_mesh": sha256_file(canonical_mesh_path),
                "skinning_weights": sha256_file(skinning_weights_path),
                "joint_transforms": sha256_file(joint_transforms_path),
            },
            "training_images_read": 0,
            "development_masks_read": len(development),
            "development_records_used_for_fit": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "note": (
                "Held-out CameraHMR root and pose are evaluator initialization only; "
                "the mask cannot alter T05 state, thresholds, or the train solution."
            ),
        },
    )


def audit_t05_qualification_lifecycle(
    public_benchmark_path: Path,
    solution_path: Path,
    background_report_path: Path,
    development_report_path: Path,
    output_path: Path,
) -> Path:
    reject_sealed_capability(
        [
            public_benchmark_path,
            solution_path,
            background_report_path,
            development_report_path,
            output_path,
        ]
    )
    if output_path.exists():
        raise FileExistsError("T05 lifecycle records are immutable")
    public = read_json(public_benchmark_path)
    solution = FixedCameraHumanSolution.model_validate(read_json(solution_path))
    background = read_json(background_report_path)
    development = read_json(development_report_path)
    reconstructed_rotations = reconstruct_t05_rotations(solution)
    observed_rotations = Rotation.from_rotvec(
        np.asarray([frame.observed_global_orient_rotvec for frame in solution.frames])
    ).as_matrix()
    rotation_error = Rotation.from_matrix(
        np.einsum("tji,tjk->tik", reconstructed_rotations, observed_rotations)
    ).magnitude()
    root = np.asarray([frame.root_translation_metres for frame in solution.frames])
    low = np.asarray([frame.low_frequency_root_translation_metres for frame in solution.frames])
    residual = np.asarray([frame.root_residual_translation_metres for frame in solution.frames])
    confidence = np.asarray([frame.yaw_confidence for frame in solution.frames])
    weighted_residual_mean = np.average(residual, axis=0, weights=confidence)
    yaw = np.asarray([frame.yaw_radians for frame in solution.frames])
    checks = {
        "module_imported": True,
        "v00_and_real_training_data_bound": solution.training_frame_count == 144
        and background.get("training_images_read") == 24,
        "mac_cpu_device_validated": development.get("device") == "cpu",
        "public_and_background_transform_passed": public.get("status") == "pass"
        and background.get("status") == "pass",
        "immutable_solution_restored": float(np.degrees(rotation_error.max())) <= 1.0e-8
        and float(np.max(np.abs(low + residual - root))) <= 1.0e-10,
        "development_evaluator_nonregression_passed": development.get("status") == "pass",
        "access_boundary_passed": solution.development_records_used_for_fit == 0
        and solution.development_images_read == 0
        and solution.sealed_test_reads == 0
        and development.get("development_records_used_for_fit") == 0
        and development.get("sealed_test_reads") == 0,
        "monotonic_phase_and_zero_mean_root_residual": bool(np.all(np.diff(yaw) >= 0.0))
        and float(np.linalg.norm(weighted_residual_mean)) <= 1.0e-8,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    state = QualificationState.BUILT
    transitions: list[dict[str, str]] = []
    transition_evidence = {
        QualificationState.IMPORTED: "module_imported",
        QualificationState.DATA_BOUND: "v00_and_real_training_data_bound",
        QualificationState.DEVICE_VALIDATED: "mac_cpu_device_validated",
        QualificationState.ONE_STEP_PASSED: "public_and_background_transform_passed",
        QualificationState.CHECKPOINT_RESTORED: "immutable_solution_restored",
        QualificationState.EVALUATOR_DRY: "development_evaluator_nonregression_passed",
        QualificationState.QUALIFIED: "access_boundary_passed",
    }
    if not blockers:
        for requested, evidence in transition_evidence.items():
            previous = state
            state = advance_qualification(state, requested)
            transitions.append({"from": previous.value, "to": state.value, "evidence": evidence})
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_t05_qualification_lifecycle.v1",
            "experiment_id": T05_EXPERIMENT_ID,
            "status": "pass" if state is QualificationState.QUALIFIED else "fail",
            "state": state.value,
            "checks": checks,
            "transitions": transitions,
            "maximum_rotation_replay_error_degrees": float(np.degrees(rotation_error.max())),
            "maximum_translation_replay_error_metres": float(np.max(np.abs(low + residual - root))),
            "confidence_weighted_root_residual_mean_metres": weighted_residual_mean.tolist(),
            "input_hashes": {
                "public_benchmark": sha256_file(public_benchmark_path),
                "solution": sha256_file(solution_path),
                "background_report": sha256_file(background_report_path),
                "development_report": sha256_file(development_report_path),
            },
            "auditor_source_sha256": sha256_file(Path(__file__)),
            "blockers": blockers,
            "development_records_used_for_fit": 0,
            "development_masks_read_by_evaluator": development.get("development_masks_read", 0),
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "attempt_marker_created": False,
            "note": (
                "ONE_STEP_PASSED denotes deterministic public recovery and train-only "
                "background transforms; T05 reowns rather than changes the frozen root scaffold."
            ),
        },
    )
