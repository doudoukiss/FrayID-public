from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from frayid.camera import axis_angle_to_matrix, make_intrinsics
from frayid.dataset import read_dataset_manifest
from frayid.geometry import linear_blend_skinning
from frayid.io import read_json, sha256_file, write_json
from frayid.renderer import normalized_boundary_error, render_soft_mesh, soft_silhouette_iou
from frayid.schemas import SequenceInitialization
from frayid.v2.checkpoint import restore_checkpoint
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import TurntableSolution
from frayid.v2.t01_joint import T01JointSchurBlock, _checkpoint_model_state
from frayid.v2.t01_phase import BoundedPhaseAxisBlock
from frayid.v2.t01_silhouette import T01CenterFocalBlock
from frayid.v2.track_factors import load_pairwise_tracklet_factors
from frayid.v2.turntable import axis_angle_rotation


def interpolate_turntable_angles(
    training_source_indices: Tensor,
    training_angles: Tensor,
    query_source_indices: Tensor,
) -> Tensor:
    """Piecewise-linear phase interpolation with endpoint extrapolation."""

    if training_source_indices.ndim != 1 or training_angles.shape != training_source_indices.shape:
        raise ValueError("training phase coordinates must be aligned vectors")
    if query_source_indices.ndim != 1 or training_source_indices.numel() < 2:
        raise ValueError("phase interpolation requires vector queries and two training points")
    if bool(torch.any(training_source_indices[1:] <= training_source_indices[:-1])):
        raise ValueError("training source indices must increase strictly")
    if bool(torch.any(training_angles[1:] <= training_angles[:-1])):
        raise ValueError("training angles must increase strictly")
    source = training_source_indices.detach().cpu().numpy().astype(np.float64)
    angles = training_angles.detach().cpu().numpy().astype(np.float64)
    query = query_source_indices.detach().cpu().numpy().astype(np.float64)
    result = np.interp(query, source, angles)
    first_slope = (angles[1] - angles[0]) / (source[1] - source[0])
    last_slope = (angles[-1] - angles[-2]) / (source[-1] - source[-2])
    before = query < source[0]
    after = query > source[-1]
    result[before] = angles[0] + (query[before] - source[0]) * first_slope
    result[after] = angles[-1] + (query[after] - source[-1]) * last_slope
    return training_angles.new_tensor(result)


def _load_joint(
    solution_path: Path,
    factor_binding_path: Path,
    phase_checkpoint_path: Path,
    center_checkpoint_path: Path,
    joint_checkpoint_path: Path,
    *,
    device: str,
) -> tuple[TurntableSolution, T01JointSchurBlock]:
    solution = TurntableSolution.model_validate(read_json(solution_path))
    factors = load_pairwise_tracklet_factors(factor_binding_path, device=device)
    phase = BoundedPhaseAxisBlock(solution, factors).to(device)
    phase_optimizer = torch.optim.SGD(phase.parameters(), lr=1.0)
    restore_checkpoint(phase_checkpoint_path.read_bytes(), phase, phase_optimizer, device=device)
    center_state = _checkpoint_model_state(center_checkpoint_path, device=device)
    silhouette = T01CenterFocalBlock(
        center_state["posed_body_vertices"],
        center_state["rotations"],
        center_state["target_bboxes"],
        base_center=center_state["base_center"],
        base_focal=float(solution.shared_intrinsics[0][0]),
        principal_point=center_state["principal_point"],
        center_bounds=(
            float(center_state["center_bounds"][0]),
            float(center_state["center_bounds"][1]),
            float(center_state["center_bounds"][2]),
        ),
    ).to(device)
    silhouette_optimizer = torch.optim.SGD(silhouette.parameters(), lr=1.0)
    restore_checkpoint(
        center_checkpoint_path.read_bytes(),
        silhouette,
        silhouette_optimizer,
        device=device,
    )
    joint = T01JointSchurBlock(phase, silhouette).to(device)
    joint_optimizer = torch.optim.SGD(joint.parameters(), lr=1.0)
    restore_checkpoint(joint_checkpoint_path.read_bytes(), joint, joint_optimizer, device=device)
    return solution, joint


def evaluate_real_turntable_development(
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
    output_path: Path,
    *,
    source_image_size: tuple[int, int] = (1120, 720),
    render_resolution: int = 64,
    device: str = "cpu",
    seed: int = 20260902,
) -> Path:
    """Score accepted T01 cameras on development masks without fitting."""

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
        output_path,
    ]
    reject_sealed_capability(paths)
    if device != "cpu":
        raise ValueError("T01 development evaluator is registered for deterministic Mac CPU")
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
    development = [
        frame for frame in manifest.frames if frame.split == "held_out" and frame.quality_accepted
    ]
    if len(development) != manifest.held_out_frame_count or not development:
        raise ValueError("T01 evaluator requires the complete frozen development split")
    with np.load(canonical_mesh_path, allow_pickle=False) as archive:
        canonical_vertices = torch.as_tensor(archive["vertices"].copy(), dtype=torch.float32)
        faces = torch.as_tensor(archive["faces"].copy(), dtype=torch.long)
    with np.load(skinning_weights_path, allow_pickle=False) as archive:
        weights = torch.as_tensor(archive["weights"].copy(), dtype=torch.float32)
    with np.load(joint_transforms_path, allow_pickle=False) as archive:
        transform_sources = archive["source_frame_indices"].astype(np.int64)
        transforms = torch.as_tensor(archive["transforms"].copy(), dtype=torch.float32)
    transform_slot = {int(source): slot for slot, source in enumerate(transform_sources)}
    first_train_source = solution.source_frame_indices[0]
    base_rotation = axis_angle_to_matrix(
        torch.tensor(
            initialization_by_source[first_train_source].global_orient,
            dtype=torch.float32,
        )
    )
    development_sources = torch.tensor(
        [frame.source_frame_index for frame in development], dtype=torch.float32
    )
    development_angles = interpolate_turntable_angles(
        torch.tensor(solution.source_frame_indices, dtype=torch.float32),
        joint.angles.detach(),
        development_sources,
    )
    treatment_rotations = axis_angle_rotation(joint.axis.detach(), development_angles)
    treatment_intrinsics = joint.intrinsics().detach()
    treatment_ious: list[float] = []
    treatment_boundaries: list[float] = []
    control_ious: list[float] = []
    control_boundaries: list[float] = []
    per_frame: list[dict[str, float | int]] = []
    for slot, (record, turntable_rotation) in enumerate(
        zip(development, treatment_rotations, strict=True)
    ):
        source = record.source_frame_index
        frame = initialization_by_source[source]
        posed_camera = linear_blend_skinning(
            canonical_vertices, weights, transforms[transform_slot[source]]
        )
        root_rotation = axis_angle_to_matrix(torch.tensor(frame.global_orient, dtype=torch.float32))
        translation = torch.tensor(frame.translation, dtype=torch.float32)
        local_pose = (posed_camera - translation) @ root_rotation
        treatment_vertices = turntable_rotation @ (local_pose @ base_rotation.T).T
        treatment_vertices = treatment_vertices.T + joint.center.detach()[None]
        control_intrinsics = make_intrinsics(frame.focal_length_px, frame.principal_point_px)
        mask_path = mask_root / Path(record.image_path).name
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"development mask is absent: {mask_path}")
        resized = cv2.resize(
            raw_mask,
            (render_resolution, render_resolution),
            interpolation=cv2.INTER_AREA,
        )
        target = torch.tensor(resized / 255.0, dtype=torch.float32)
        torch.manual_seed(seed + slot)
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
        torch.manual_seed(seed + slot)
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
                "interpolated_angle_radians": float(development_angles[slot]),
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
        blockers.append("development_iou_regressed_beyond_0_002_against_free_camera_control")
    if boundary_change > 0:
        blockers.append("development_boundary_regressed_against_free_camera_control")
    finite = all(
        math.isfinite(value)
        for value in (
            *treatment_ious,
            *treatment_boundaries,
            *control_ious,
            *control_boundaries,
        )
    )
    if not finite:
        blockers.append("development_camera_evaluation_nonfinite")
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_t01_development_camera_evaluation.v1",
        "status": "pass" if not blockers else "fail",
        "evaluation_id": "postv2_t01_development_camera_control_r01",
        "device": device,
        "dtype": "float32",
        "render_resolution": render_resolution,
        "development_frame_count": len(development),
        "treatment": treatment,
        "matched_free_camera_control": control,
        "treatment_minus_control_iou": iou_change,
        "treatment_minus_control_boundary": boundary_change,
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
        "training_images_read": 0,
        "development_masks_read": len(development),
        "sealed_test_accesses": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "Development masks are evaluator-only and cannot alter T01 parameters or thresholds.",
            "Treatment and control use the identical fixed clothing scaffold and identical samples.",
            "The matched control is each frame's original free CameraHMR root/camera initialization.",
        ],
    }
    return write_json(output_path, report)
