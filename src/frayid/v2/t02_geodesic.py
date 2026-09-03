from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp  # type: ignore[import-untyped]
from torch import Tensor

from frayid.camera import axis_angle_to_matrix
from frayid.dataset import read_dataset_manifest
from frayid.geometry import linear_blend_skinning
from frayid.io import read_json, sha256_file, write_json
from frayid.renderer import normalized_boundary_error, render_soft_mesh, soft_silhouette_iou
from frayid.schemas import SequenceInitialization
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.t01_evaluator import _load_joint


@dataclass(frozen=True)
class GeodesicTrajectoryPrediction:
    rotations: Tensor
    translations: Tensor
    residual_rotation_degrees: Tensor
    residual_translation_metres: Tensor


def internal_train_validation_slots(frame_count: int) -> tuple[Tensor, Tensor]:
    """Freeze every fifth accepted train slot as T02's internal validation set."""

    if frame_count < 10:
        raise ValueError("T02 internal split requires at least ten training frames")
    slots = torch.arange(frame_count, dtype=torch.long)
    validation = slots.remainder(5) == 0
    return slots[~validation], slots[validation]


def _slerp_with_extrapolation(
    fit_sources: np.ndarray,
    fit_rotations: Rotation,
    query_sources: np.ndarray,
) -> Rotation:
    interpolation = Slerp(fit_sources, fit_rotations)
    matrices: list[np.ndarray] = []
    first = float(fit_sources[0])
    last = float(fit_sources[-1])
    for query in query_sources:
        if query < first:
            fraction = (query - fit_sources[0]) / (fit_sources[1] - fit_sources[0])
            relative = fit_rotations[0].inv() * fit_rotations[1]
            predicted = fit_rotations[0] * Rotation.from_rotvec(fraction * relative.as_rotvec())
        elif query > last:
            fraction = (query - fit_sources[-2]) / (fit_sources[-1] - fit_sources[-2])
            relative = fit_rotations[-2].inv() * fit_rotations[-1]
            predicted = fit_rotations[-2] * Rotation.from_rotvec(fraction * relative.as_rotvec())
        else:
            predicted = interpolation(float(query))
        matrices.append(predicted.as_matrix())
    return Rotation.from_matrix(np.stack(matrices))


def _linear_with_extrapolation(
    fit_sources: np.ndarray,
    fit_values: np.ndarray,
    query_sources: np.ndarray,
) -> np.ndarray:
    result = np.column_stack(
        [np.interp(query_sources, fit_sources, fit_values[:, axis]) for axis in range(3)]
    )
    before = query_sources < fit_sources[0]
    after = query_sources > fit_sources[-1]
    first_slope = (fit_values[1] - fit_values[0]) / (fit_sources[1] - fit_sources[0])
    last_slope = (fit_values[-1] - fit_values[-2]) / (fit_sources[-1] - fit_sources[-2])
    result[before] = fit_values[0] + (query_sources[before] - fit_sources[0])[:, None] * first_slope
    result[after] = fit_values[-1] + (query_sources[after] - fit_sources[-1])[:, None] * last_slope
    return result


def interpolate_bounded_geodesic_residual(
    fit_source_indices: Tensor,
    fit_observed_rotations: Tensor,
    fit_observed_translations: Tensor,
    query_source_indices: Tensor,
    dominant_query_rotations: Tensor,
    base_translation: Tensor,
    *,
    maximum_residual_rotation_degrees: float = 30.0,
    maximum_residual_translation_metres: float = 0.08,
) -> GeodesicTrajectoryPrediction:
    """Interpolate observed roots, then project residuals into hard T02 bounds."""

    fit_count = fit_source_indices.numel()
    query_count = query_source_indices.numel()
    if fit_count < 2 or query_count < 1:
        raise ValueError("T02 geodesic interpolation requires fit and query frames")
    if fit_observed_rotations.shape != (fit_count, 3, 3):
        raise ValueError("fit rotations must have shape [fit, 3, 3]")
    if fit_observed_translations.shape != (fit_count, 3):
        raise ValueError("fit translations must have shape [fit, 3]")
    if dominant_query_rotations.shape != (query_count, 3, 3):
        raise ValueError("dominant query rotations must have shape [query, 3, 3]")
    if base_translation.shape != (3,):
        raise ValueError("T02 base translation must have shape [3]")
    if bool(torch.any(fit_source_indices[1:] <= fit_source_indices[:-1])):
        raise ValueError("T02 fit sources must increase strictly")
    if min(maximum_residual_rotation_degrees, maximum_residual_translation_metres) <= 0:
        raise ValueError("T02 residual bounds must be positive")
    fit_sources = fit_source_indices.detach().cpu().numpy().astype(np.float64)
    query_sources = query_source_indices.detach().cpu().numpy().astype(np.float64)
    observed = Rotation.from_matrix(fit_observed_rotations.detach().cpu().numpy())
    interpolated = _slerp_with_extrapolation(fit_sources, observed, query_sources)
    dominant = Rotation.from_matrix(dominant_query_rotations.detach().cpu().numpy())
    residual_vectors = (interpolated * dominant.inv()).as_rotvec()
    residual_norm = np.linalg.norm(residual_vectors, axis=1)
    rotation_limit = math.radians(maximum_residual_rotation_degrees)
    rotation_scale = np.minimum(1.0, rotation_limit / np.maximum(residual_norm, 1.0e-12))
    clipped_residual = residual_vectors * rotation_scale[:, None]
    rotations = Rotation.from_rotvec(clipped_residual) * dominant
    interpolated_translation = _linear_with_extrapolation(
        fit_sources,
        fit_observed_translations.detach().cpu().numpy(),
        query_sources,
    )
    base = base_translation.detach().cpu().numpy()
    translation_residual = interpolated_translation - base
    translation_norm = np.linalg.norm(translation_residual, axis=1)
    translation_scale = np.minimum(
        1.0,
        maximum_residual_translation_metres / np.maximum(translation_norm, 1.0e-12),
    )
    translations = base + translation_residual * translation_scale[:, None]
    return GeodesicTrajectoryPrediction(
        rotations=fit_observed_rotations.new_tensor(rotations.as_matrix()),
        translations=fit_observed_translations.new_tensor(translations),
        residual_rotation_degrees=fit_observed_translations.new_tensor(
            np.degrees(np.minimum(residual_norm, rotation_limit))
        ),
        residual_translation_metres=fit_observed_translations.new_tensor(
            np.minimum(translation_norm, maximum_residual_translation_metres)
        ),
    )


def qualify_real_t02_internal_validation(
    solution_path: Path,
    factor_binding_path: Path,
    phase_checkpoint_path: Path,
    center_checkpoint_path: Path,
    joint_checkpoint_path: Path,
    initialization_path: Path,
    manifest_path: Path,
    mask_root: Path,
    canonical_mesh_path: Path,
    skinning_weights_path: Path,
    joint_transforms_path: Path,
    split_output_path: Path,
    report_output_path: Path,
    *,
    source_image_size: tuple[int, int] = (1120, 720),
    render_resolution: int = 64,
    device: str = "cpu",
    seed: int = 20260902,
) -> Path:
    """Evaluate T02's bounded trajectory on frozen train-internal validation."""

    paths = [
        solution_path,
        factor_binding_path,
        phase_checkpoint_path,
        center_checkpoint_path,
        joint_checkpoint_path,
        initialization_path,
        manifest_path,
        mask_root,
        canonical_mesh_path,
        skinning_weights_path,
        joint_transforms_path,
        split_output_path,
        report_output_path,
    ]
    reject_sealed_capability(paths)
    if device != "cpu":
        raise ValueError("T02 internal validation is registered for deterministic Mac CPU")
    if report_output_path.exists():
        raise FileExistsError("T02 internal-validation report is immutable")
    solution, joint = _load_joint(
        solution_path,
        factor_binding_path,
        phase_checkpoint_path,
        center_checkpoint_path,
        joint_checkpoint_path,
        device=device,
    )
    initialization = SequenceInitialization.model_validate(read_json(initialization_path))
    initialization_by_source = {frame.source_frame_index: frame for frame in initialization.frames}
    manifest = read_dataset_manifest(manifest_path)
    training = [
        frame for frame in manifest.frames if frame.split == "train" and frame.quality_accepted
    ]
    if len(training) != manifest.train_frame_count or len(training) != len(
        solution.source_frame_indices
    ):
        raise ValueError("T02 internal split must cover the complete accepted training set")
    fit_slots, validation_slots = internal_train_validation_slots(len(training))
    split = {
        "schema_version": "frayid_v2_t02_internal_train_split.v1",
        "split_rule": "accepted_train_slot_modulo_5_equals_0_is_validation",
        "fit_source_frame_indices": [training[int(slot)].source_frame_index for slot in fit_slots],
        "validation_source_frame_indices": [
            training[int(slot)].source_frame_index for slot in validation_slots
        ],
        "fit_frame_count": int(fit_slots.numel()),
        "validation_frame_count": int(validation_slots.numel()),
        "legacy_development_frames_bound": 0,
        "sealed_test_accesses": 0,
    }
    if split_output_path.exists():
        if read_json(split_output_path) != split:
            raise ValueError("existing T02 internal split does not match the frozen rule")
    else:
        write_json(split_output_path, split)
    source_indices = torch.tensor(
        [frame.source_frame_index for frame in training], dtype=torch.float32
    )
    observed_rotation_vectors = torch.tensor(
        [initialization_by_source[frame.source_frame_index].global_orient for frame in training],
        dtype=torch.float32,
    )
    observed_rotations = axis_angle_to_matrix(observed_rotation_vectors)
    observed_translations = torch.tensor(
        [initialization_by_source[frame.source_frame_index].translation for frame in training],
        dtype=torch.float32,
    )
    dominant_rotations = joint.rotations().detach() @ observed_rotations[0]
    prediction = interpolate_bounded_geodesic_residual(
        source_indices[fit_slots],
        observed_rotations[fit_slots],
        observed_translations[fit_slots],
        source_indices[validation_slots],
        dominant_rotations[validation_slots],
        joint.center.detach(),
    )
    with np.load(canonical_mesh_path, allow_pickle=False) as archive:
        canonical_vertices = torch.as_tensor(archive["vertices"].copy(), dtype=torch.float32)
        faces = torch.as_tensor(archive["faces"].copy(), dtype=torch.long)
    with np.load(skinning_weights_path, allow_pickle=False) as archive:
        weights = torch.as_tensor(archive["weights"].copy(), dtype=torch.float32)
    with np.load(joint_transforms_path, allow_pickle=False) as archive:
        transform_sources = archive["source_frame_indices"].astype(np.int64)
        transforms = torch.as_tensor(archive["transforms"].copy(), dtype=torch.float32)
    transform_slot = {int(source): slot for slot, source in enumerate(transform_sources)}
    treatment_ious: list[float] = []
    treatment_boundaries: list[float] = []
    control_ious: list[float] = []
    control_boundaries: list[float] = []
    per_frame: list[dict[str, float | int]] = []
    treatment_intrinsics = joint.intrinsics().detach()
    for query_slot, validation_slot in enumerate(validation_slots.tolist()):
        record = training[validation_slot]
        source = record.source_frame_index
        frame = initialization_by_source[source]
        posed_camera = linear_blend_skinning(
            canonical_vertices, weights, transforms[transform_slot[source]]
        )
        observed_rotation = observed_rotations[validation_slot]
        observed_translation = observed_translations[validation_slot]
        local_pose = (posed_camera - observed_translation) @ observed_rotation
        treatment_vertices = local_pose @ prediction.rotations[query_slot].T
        treatment_vertices = treatment_vertices + prediction.translations[query_slot]
        control_intrinsics = torch.tensor(
            [
                [frame.focal_length_px, 0.0, frame.principal_point_px[0]],
                [0.0, frame.focal_length_px, frame.principal_point_px[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        mask_path = mask_root / Path(record.image_path).name
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"T02 validation mask is absent: {mask_path}")
        resized = cv2.resize(
            raw_mask,
            (render_resolution, render_resolution),
            interpolation=cv2.INTER_AREA,
        )
        target = torch.tensor(resized / 255.0, dtype=torch.float32)
        torch.manual_seed(seed + query_slot)
        treatment_silhouette, _ = render_soft_mesh(
            treatment_vertices,
            faces,
            treatment_intrinsics,
            (render_resolution, render_resolution),
            source_image_size=source_image_size,
            sigma_pixels=1.75 * render_resolution / 128.0,
            sample_count=2048,
            reference_sample_count=2048,
        )
        torch.manual_seed(seed + query_slot)
        control_silhouette, _ = render_soft_mesh(
            posed_camera,
            faces,
            control_intrinsics,
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
                "residual_rotation_degrees": float(
                    prediction.residual_rotation_degrees[query_slot]
                ),
                "residual_translation_metres": float(
                    prediction.residual_translation_metres[query_slot]
                ),
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
    blockers: list[str] = []
    if iou_change < -0.002:
        blockers.append("internal_validation_iou_regressed_beyond_0_002")
    if boundary_change > 0:
        blockers.append("internal_validation_boundary_regressed")
    if float(prediction.residual_rotation_degrees.max()) > 30.0001:
        blockers.append("T02_residual_rotation_bound_exceeded")
    if float(prediction.residual_translation_metres.max()) > 0.080001:
        blockers.append("T02_residual_translation_bound_exceeded")
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_t02_internal_validation.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t02_geodesic_internal_validation_r02",
        "device": device,
        "dtype": "float32",
        "split_path": str(split_output_path),
        "split_sha256": sha256_file(split_output_path),
        "fit_frame_count": int(fit_slots.numel()),
        "validation_frame_count": int(validation_slots.numel()),
        "treatment": treatment,
        "matched_free_camera_control": control,
        "treatment_minus_control_iou": iou_change,
        "treatment_minus_control_boundary": boundary_change,
        "maximum_residual_rotation_degrees": float(prediction.residual_rotation_degrees.max()),
        "maximum_residual_translation_metres": float(prediction.residual_translation_metres.max()),
        "per_frame": per_frame,
        "input_hashes": {
            "solution": sha256_file(solution_path),
            "factors": sha256_file(factor_binding_path),
            "phase_checkpoint": sha256_file(phase_checkpoint_path),
            "center_checkpoint": sha256_file(center_checkpoint_path),
            "joint_checkpoint": sha256_file(joint_checkpoint_path),
            "initialization": sha256_file(initialization_path),
            "manifest": sha256_file(manifest_path),
            "canonical_mesh": sha256_file(canonical_mesh_path),
            "skinning_weights": sha256_file(skinning_weights_path),
            "joint_transforms": sha256_file(joint_transforms_path),
        },
        "blockers": blockers,
        "optimizer_steps": 0,
        "training_masks_read": int(validation_slots.numel()),
        "legacy_development_images_read": 0,
        "sealed_test_accesses": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "Only the frozen train-internal validation subset is scored.",
            "T01 shared parameters are a fixed predecessor initialization, not refit on validation.",
            "The geodesic residual is continuous and hard-clipped; no free validation camera is fitted.",
        ],
    }
    return write_json(report_output_path, report)
