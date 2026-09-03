from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from torch import Tensor

from frayid.camera import axis_angle_to_matrix, make_intrinsics
from frayid.dataset import read_dataset_manifest
from frayid.geometry import linear_blend_skinning
from frayid.io import read_json, sha256_file, write_json
from frayid.renderer import normalized_boundary_error, render_soft_mesh, soft_silhouette_iou
from frayid.schemas import SequenceInitialization
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.t02_geodesic import _linear_with_extrapolation, _slerp_with_extrapolation


@dataclass(frozen=True)
class SequenceRegularizedPrediction:
    rotations: Tensor
    translations: Tensor
    rotation_correction_degrees: Tensor
    translation_correction_metres: Tensor


def t03_internal_slots(frame_count: int) -> tuple[Tensor, Tensor, Tensor]:
    """Exclude T02's fold, reserve a new fold, and fit on the remainder."""

    if frame_count < 15:
        raise ValueError("T03 internal split requires at least fifteen training frames")
    slots = torch.arange(frame_count, dtype=torch.long)
    residue = slots.remainder(5)
    excluded = residue == 0
    validation = residue == 1
    fit = ~(excluded | validation)
    return slots[fit], slots[validation], slots[excluded]


def sequence_regularized_dynamic_prediction(
    fit_source_indices: Tensor,
    fit_rotations: Tensor,
    fit_translations: Tensor,
    query_source_indices: Tensor,
    query_initial_rotations: Tensor,
    query_initial_translations: Tensor,
    *,
    initialization_retention: float = 0.90,
    maximum_rotation_correction_degrees: float = 2.0,
    maximum_translation_correction_metres: float = 0.02,
) -> SequenceRegularizedPrediction:
    """Shrink frozen per-frame initialization toward a continuous SE(3) path."""

    fit_count = fit_source_indices.numel()
    query_count = query_source_indices.numel()
    if fit_count < 2 or query_count < 1:
        raise ValueError("T03 sequence regularization requires fit and query frames")
    if fit_rotations.shape != (fit_count, 3, 3) or fit_translations.shape != (
        fit_count,
        3,
    ):
        raise ValueError("T03 fit camera arrays have invalid shape")
    if query_initial_rotations.shape != (
        query_count,
        3,
        3,
    ) or query_initial_translations.shape != (query_count, 3):
        raise ValueError("T03 query camera arrays have invalid shape")
    if not 0.0 < initialization_retention < 1.0:
        raise ValueError("T03 initialization retention must lie strictly inside (0, 1)")
    fit_sources = fit_source_indices.detach().cpu().numpy().astype(np.float64)
    query_sources = query_source_indices.detach().cpu().numpy().astype(np.float64)
    spline_rotation = _slerp_with_extrapolation(
        fit_sources,
        Rotation.from_matrix(fit_rotations.detach().cpu().numpy()),
        query_sources,
    )
    initial_rotation = Rotation.from_matrix(query_initial_rotations.detach().cpu().numpy())
    toward_initial = spline_rotation.inv() * initial_rotation
    desired_rotation = spline_rotation * Rotation.from_rotvec(
        initialization_retention * toward_initial.as_rotvec()
    )
    correction = desired_rotation * initial_rotation.inv()
    correction_vectors = correction.as_rotvec()
    correction_norm = np.linalg.norm(correction_vectors, axis=1)
    rotation_limit = math.radians(maximum_rotation_correction_degrees)
    rotation_scale = np.minimum(1.0, rotation_limit / np.maximum(correction_norm, 1.0e-12))
    clipped_correction = correction_vectors * rotation_scale[:, None]
    rotations = Rotation.from_rotvec(clipped_correction) * initial_rotation
    spline_translation = _linear_with_extrapolation(
        fit_sources,
        fit_translations.detach().cpu().numpy(),
        query_sources,
    )
    initial_translation = query_initial_translations.detach().cpu().numpy()
    desired_translation = spline_translation + initialization_retention * (
        initial_translation - spline_translation
    )
    translation_correction = desired_translation - initial_translation
    translation_norm = np.linalg.norm(translation_correction, axis=1)
    translation_scale = np.minimum(
        1.0,
        maximum_translation_correction_metres / np.maximum(translation_norm, 1.0e-12),
    )
    clipped_translation_correction = translation_correction * translation_scale[:, None]
    translations = initial_translation + clipped_translation_correction
    return SequenceRegularizedPrediction(
        rotations=query_initial_rotations.new_tensor(rotations.as_matrix()),
        translations=query_initial_translations.new_tensor(translations),
        rotation_correction_degrees=query_initial_translations.new_tensor(
            np.degrees(np.minimum(correction_norm, rotation_limit))
        ),
        translation_correction_metres=query_initial_translations.new_tensor(
            np.minimum(translation_norm, maximum_translation_correction_metres)
        ),
    )


def qualify_real_t03_internal_validation(
    factor_binding_path: Path,
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
    """Cross-validate fixed T03 shrinkage on a previously unread train fold."""

    paths = [
        factor_binding_path,
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
        raise ValueError("T03 internal validation is registered for deterministic Mac CPU")
    if split_output_path.exists() or report_output_path.exists():
        raise FileExistsError("T03 internal-validation outputs are immutable")
    initialization = SequenceInitialization.model_validate(read_json(initialization_path))
    initialization_by_source = {frame.source_frame_index: frame for frame in initialization.frames}
    manifest = read_dataset_manifest(manifest_path)
    training = [
        frame for frame in manifest.frames if frame.split == "train" and frame.quality_accepted
    ]
    if len(training) != manifest.train_frame_count:
        raise ValueError("T03 internal split must cover the accepted training set")
    fit_slots, validation_slots, excluded_slots = t03_internal_slots(len(training))
    split = {
        "schema_version": "frayid_v2_t03_internal_train_split.v1",
        "split_rule": "exclude_slot_modulo_5_equals_0_validate_equals_1_fit_equals_2_3_4",
        "fit_source_frame_indices": [training[int(slot)].source_frame_index for slot in fit_slots],
        "validation_source_frame_indices": [
            training[int(slot)].source_frame_index for slot in validation_slots
        ],
        "excluded_prior_validation_source_frame_indices": [
            training[int(slot)].source_frame_index for slot in excluded_slots
        ],
        "fit_frame_count": int(fit_slots.numel()),
        "validation_frame_count": int(validation_slots.numel()),
        "excluded_frame_count": int(excluded_slots.numel()),
        "legacy_development_frames_bound": 0,
        "sealed_test_accesses": 0,
    }
    write_json(split_output_path, split)
    sources = torch.tensor([frame.source_frame_index for frame in training], dtype=torch.float32)
    rotation_vectors = torch.tensor(
        [initialization_by_source[frame.source_frame_index].global_orient for frame in training],
        dtype=torch.float32,
    )
    rotations = axis_angle_to_matrix(rotation_vectors)
    translations = torch.tensor(
        [initialization_by_source[frame.source_frame_index].translation for frame in training],
        dtype=torch.float32,
    )
    prediction = sequence_regularized_dynamic_prediction(
        sources[fit_slots],
        rotations[fit_slots],
        translations[fit_slots],
        sources[validation_slots],
        rotations[validation_slots],
        translations[validation_slots],
    )
    replay = sequence_regularized_dynamic_prediction(
        sources[fit_slots],
        rotations[fit_slots],
        translations[fit_slots],
        sources[validation_slots],
        rotations[validation_slots],
        translations[validation_slots],
    )
    replay_exact = all(
        torch.equal(first, second)
        for first, second in zip(
            (
                prediction.rotations,
                prediction.translations,
                prediction.rotation_correction_degrees,
                prediction.translation_correction_metres,
            ),
            (
                replay.rotations,
                replay.translations,
                replay.rotation_correction_degrees,
                replay.translation_correction_metres,
            ),
            strict=True,
        )
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
    shared_intrinsics = make_intrinsics(
        initialization.shared_focal_length_px,
        initialization.shared_principal_point_px,
    )
    for query_slot, validation_slot in enumerate(validation_slots.tolist()):
        record = training[validation_slot]
        source = record.source_frame_index
        posed_camera = linear_blend_skinning(
            canonical_vertices, weights, transforms[transform_slot[source]]
        )
        observed_rotation = rotations[validation_slot]
        observed_translation = translations[validation_slot]
        local_pose = (posed_camera - observed_translation) @ observed_rotation
        treatment_vertices = local_pose @ prediction.rotations[query_slot].T
        treatment_vertices = treatment_vertices + prediction.translations[query_slot]
        mask_path = mask_root / Path(record.image_path).name
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"T03 validation mask is absent: {mask_path}")
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
            shared_intrinsics,
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
            shared_intrinsics,
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
                "rotation_correction_degrees": float(
                    prediction.rotation_correction_degrees[query_slot]
                ),
                "translation_correction_metres": float(
                    prediction.translation_correction_metres[query_slot]
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
        blockers.append("T03_internal_validation_iou_regressed_beyond_0_002")
    if boundary_change > 0:
        blockers.append("T03_internal_validation_boundary_regressed")
    if float(prediction.rotation_correction_degrees.max()) > 2.0001:
        blockers.append("T03_rotation_correction_bound_exceeded")
    if float(prediction.translation_correction_metres.max()) > 0.020001:
        blockers.append("T03_translation_correction_bound_exceeded")
    if not replay_exact:
        blockers.append("T03_prediction_replay_mismatch")
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_t03_internal_validation.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t03_sequence_regularized_internal_validation_r01",
        "device": device,
        "dtype": "float32",
        "initialization_retention_fraction": 0.90,
        "split_path": str(split_output_path),
        "split_sha256": sha256_file(split_output_path),
        "fit_frame_count": int(fit_slots.numel()),
        "validation_frame_count": int(validation_slots.numel()),
        "excluded_prior_validation_frame_count": int(excluded_slots.numel()),
        "treatment": treatment,
        "matched_free_camera_control": control,
        "treatment_minus_control_iou": iou_change,
        "treatment_minus_control_boundary": boundary_change,
        "maximum_rotation_correction_degrees": float(prediction.rotation_correction_degrees.max()),
        "maximum_translation_correction_metres": float(
            prediction.translation_correction_metres.max()
        ),
        "same_device_replay_exact": replay_exact,
        "per_frame": per_frame,
        "input_hashes": {
            "factors": sha256_file(factor_binding_path),
            "initialization": sha256_file(initialization_path),
            "manifest": sha256_file(manifest_path),
            "canonical_mesh": sha256_file(canonical_mesh_path),
            "skinning_weights": sha256_file(skinning_weights_path),
            "joint_transforms": sha256_file(joint_transforms_path),
        },
        "blockers": blockers,
        "optimizer_steps": 0,
        "training_masks_read": int(validation_slots.numel()),
        "prior_validation_masks_read": 0,
        "legacy_development_images_read": 0,
        "sealed_test_accesses": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "CameraHMR is frozen per-frame initialization, not optimizer output or observed truth.",
            "The query mask is evaluator-only; prediction uses its frozen initialization and the fit spline.",
            "The previously scored T02 fold and legacy development split are not read.",
        ],
    }
    return write_json(report_output_path, report)
