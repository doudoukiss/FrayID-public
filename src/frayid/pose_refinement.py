from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from torch import Tensor

from frayid.camera import axis_angle_to_matrix, make_intrinsics
from frayid.config import ReconstructionConfig
from frayid.dataset import DATASET_MANIFEST_FILENAME, read_dataset_manifest
from frayid.geometry import linear_blend_skinning
from frayid.initialization import (
    SMPL_BODY_JOINT_INDICES,
    SMPLRuntime,
    load_initialization,
    load_observed_pose,
    validate_initialization_contract,
)
from frayid.io import sha256_file, write_json
from frayid.renderer import differentiable_boundary_loss, render_soft_mesh, silhouette_loss
from frayid.schemas import SequenceInitialization


def bounded_vector(raw: Tensor, maximum_norm: float) -> Tensor:
    """Map arbitrary vectors smoothly into an L2 ball."""
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    scale = torch.tanh(norm) * maximum_norm / norm.clamp_min(1e-8)
    return torch.where(norm > 1e-8, raw * scale, raw * maximum_norm)


def geman_mcclure(residual: Tensor, scale: float | Tensor) -> Tensor:
    """Bounded robust penalty with a quadratic basin around zero."""
    scale_tensor = torch.as_tensor(scale, dtype=residual.dtype, device=residual.device)
    squared = residual.square()
    return scale_tensor.square() * squared / (scale_tensor.square() + squared)


def rotation_geodesic_radians(first: Tensor, second: Tensor) -> Tensor:
    """Stable SO(3) geodesic angle for matrices shaped ``[..., 3, 3]``."""
    relative = first.transpose(-1, -2) @ second
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    return torch.atan2(sine, cosine.clamp(-1.0, 1.0))


def rotation_geodesic_acceleration(rotations: Tensor) -> Tensor:
    """Mean change between consecutive SO(3) angular increments."""
    if rotations.shape[0] < 3:
        return rotations.sum() * 0.0
    velocity = rotations[:-1].transpose(-1, -2) @ rotations[1:]
    return rotation_geodesic_radians(velocity[:-1], velocity[1:]).square().mean()


def translation_acceleration(values: Tensor) -> Tensor:
    if values.shape[0] < 3:
        return values.sum() * 0.0
    return (values[2:] - 2.0 * values[1:-1] + values[:-2]).square().sum(-1).mean()


def robust_weighted_reprojection(
    predicted: Tensor,
    target: Tensor,
    confidence: Tensor,
    *,
    scale_pixels: float,
    image_diagonal: float,
) -> Tensor:
    residual = torch.linalg.vector_norm(predicted - target, dim=-1)
    weights = confidence.clamp_min(0.0)
    per_frame = (geman_mcclure(residual, scale_pixels) * weights).sum(-1) / weights.sum(
        -1
    ).clamp_min(1.0)
    return per_frame.mean() / image_diagonal


def _project(points: Tensor, intrinsics: Tensor) -> Tensor:
    z = points[..., 2].clamp_min(1e-5)
    return torch.stack(
        (
            points[..., 0] / z * intrinsics[0, 0] + intrinsics[0, 2],
            points[..., 1] / z * intrinsics[1, 1] + intrinsics[1, 2],
        ),
        dim=-1,
    )


def _pose(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size > 69:
        raise ValueError("SMPL body pose cannot exceed 69 values")
    return np.pad(array, (0, 69 - array.size)).reshape(23, 3)


def _silhouette_slots(step: int, frame_count: int, batch_size: int) -> list[int]:
    """Cover every frame with one split-independent deterministic schedule."""
    stride = max(frame_count // batch_size, 1)
    offset = step % stride
    return [min(offset + index * stride, frame_count - 1) for index in range(batch_size)]


def _resolve_downloaded_artifact(value: str) -> Path:
    """Resolve Modal's one-level repeated directory download without changing provenance."""
    path = Path(value)
    if path.is_file():
        return path
    repeated = path.parent / path.parent.name / path.name
    if repeated.is_file():
        return repeated
    raise FileNotFoundError(f"Required initialization artifact is missing: {path}")


def refit_pose_sequence(
    config: ReconstructionConfig,
    *,
    input_path: Path,
    output_directory: Path,
    model_root: Path,
    steps: int | None = None,
    device_name: str = "cpu",
) -> tuple[SequenceInitialization, dict[str, Any]]:
    """Refit every SMPL joint from real sequence evidence while freezing shape/camera.

    The operation is immutable: ``output_directory`` must not already contain files.
    Held-out labels and held-out metrics are never read; every selected frame follows
    the same confidence-weighted objective and deterministic schedule.
    """
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite pose-refit output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    settings = config.pose_refit
    step_count = steps or settings.steps
    if step_count <= 0:
        raise ValueError("Pose-refit steps must be positive")
    device = torch.device(device_name)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    initialization = load_initialization(input_path)
    manifest = read_dataset_manifest(config.paths.dataset_root / DATASET_MANIFEST_FILENAME)
    expected_indices = {frame.source_frame_index for frame in manifest.frames}
    blockers = validate_initialization_contract(initialization, expected_indices)
    if blockers:
        raise ValueError("Pose-refit input failed: " + ", ".join(blockers))
    if not initialization.canonical_mesh_path or not initialization.skinning_weights_path:
        raise ValueError("Pose refit requires the validated shared clothing envelope")

    frames = sorted(initialization.frames, key=lambda item: item.source_frame_index)
    observed = load_observed_pose(
        config.paths.dataset_root / config.evidence.observed_pose_filename
    )
    observed_by_index = {frame.source_frame_index: frame for frame in observed.frames}
    if expected_indices.difference(observed_by_index):
        raise ValueError("Pose refit requires Sapiens2 joint evidence for every selected frame")
    runtime = SMPLRuntime(model_root, device)
    frame_count = len(frames)
    anchor_body = torch.tensor(
        np.stack([_pose(frame.body_pose) for frame in frames]),
        dtype=torch.float32,
        device=device,
    )
    anchor_root = torch.tensor(
        [frame.global_orient for frame in frames], dtype=torch.float32, device=device
    )
    anchor_translation = torch.tensor(
        [frame.translation for frame in frames], dtype=torch.float32, device=device
    )
    body_raw = torch.nn.Parameter(torch.zeros_like(anchor_body))
    root_raw = torch.nn.Parameter(torch.zeros_like(anchor_root))
    translation_raw = torch.nn.Parameter(torch.zeros_like(anchor_translation))
    optimizer = torch.optim.Adam((body_raw, root_raw, translation_raw), lr=settings.learning_rate)
    betas = torch.tensor(initialization.shared_betas[:10], dtype=torch.float32, device=device)[
        None
    ].expand(frame_count, -1)
    intrinsics = make_intrinsics(
        initialization.shared_focal_length_px,
        initialization.shared_principal_point_px,
        device=device,
    )
    camera_keypoints = torch.stack(
        [torch.tensor(frame.keypoints_2d, dtype=torch.float32, device=device) for frame in frames]
    )
    raw_joints = torch.stack(
        [torch.tensor(frame.joints_3d, dtype=torch.float32, device=device) for frame in frames]
    )
    observed_keypoints = torch.stack(
        [
            torch.tensor(
                observed_by_index[frame.source_frame_index].keypoints_body12,
                dtype=torch.float32,
                device=device,
            )
            for frame in frames
        ]
    )
    minimum_confidence = config.initialization_gate.observed_pose_minimum_confidence
    observed_confidence = observed_keypoints[..., 2] * (
        observed_keypoints[..., 2] >= minimum_confidence
    )
    if torch.any((observed_confidence > 0).sum(-1) < 6):
        raise ValueError("Pose refit requires at least six confident Sapiens2 joints per frame")
    camera_confidence = (
        camera_keypoints[..., 2]
        if camera_keypoints.shape[-1] >= 3
        else torch.ones_like(camera_keypoints[..., 0])
    )
    mesh = np.load(_resolve_downloaded_artifact(initialization.canonical_mesh_path))
    canonical_vertices = torch.tensor(mesh["vertices"], dtype=torch.float32, device=device)
    faces = torch.tensor(mesh["faces"], dtype=torch.long, device=device)
    weights = torch.tensor(
        np.load(_resolve_downloaded_artifact(initialization.skinning_weights_path))["weights"],
        dtype=torch.float32,
        device=device,
    )
    records_by_index = {record.source_frame_index: record for record in manifest.frames}
    masks: list[Tensor] = []
    for frame in frames:
        record = records_by_index[frame.source_frame_index]
        path = (
            config.paths.dataset_root
            / config.evidence.masks_subdirectory
            / Path(record.image_path).name
        )
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Real Sapiens2 mask is required: {path}")
        target = torch.tensor(image / 255.0, dtype=torch.float32, device=device)
        masks.append(
            torch.nn.functional.interpolate(
                target[None, None],
                size=(settings.render_resolution, settings.render_resolution),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        )
    image_diagonal = math.hypot(initialization.image_width, initialization.image_height)
    anchor_body_rotations = axis_angle_to_matrix(anchor_body)
    anchor_root_rotations = axis_angle_to_matrix(anchor_root)
    initial_loss: float | None = None
    final_terms: dict[str, float] = {}

    for step in range(step_count):
        optimizer.zero_grad(set_to_none=True)
        body_delta = bounded_vector(
            body_raw, math.radians(settings.maximum_joint_correction_degrees)
        )
        root_delta = bounded_vector(
            root_raw, math.radians(settings.maximum_root_correction_degrees)
        )
        translation_delta = bounded_vector(
            translation_raw, settings.maximum_translation_correction_m
        )
        body = anchor_body + body_delta
        root = anchor_root + root_delta
        translation = anchor_translation + translation_delta
        output = runtime.forward(betas, body.reshape(frame_count, 69), root, translation)
        camera_count = min(camera_keypoints.shape[1], output.joints.shape[1])
        camera_projection = _project(output.joints[:, :camera_count], intrinsics)
        camera_reprojection = robust_weighted_reprojection(
            camera_projection,
            camera_keypoints[:, :camera_count, :2],
            camera_confidence[:, :camera_count],
            scale_pixels=settings.robust_scale_pixels,
            image_diagonal=image_diagonal,
        )
        observed_projection = _project(output.joints[:, list(SMPL_BODY_JOINT_INDICES)], intrinsics)
        observed_reprojection = robust_weighted_reprojection(
            observed_projection,
            observed_keypoints[..., :2],
            observed_confidence,
            scale_pixels=settings.robust_scale_pixels,
            image_diagonal=image_diagonal,
        )
        raw_count = min(raw_joints.shape[1], output.joints.shape[1])
        joint_3d_anchor = geman_mcclure(
            torch.linalg.vector_norm(
                output.joints[:, :raw_count] - raw_joints[:, :raw_count], dim=-1
            ),
            0.05,
        ).mean()
        body_rotations = axis_angle_to_matrix(body)
        root_rotations = axis_angle_to_matrix(root)
        pose_anchor = (
            rotation_geodesic_radians(anchor_body_rotations, body_rotations).square().mean()
        )
        root_anchor = (
            rotation_geodesic_radians(anchor_root_rotations, root_rotations).square().mean()
        )
        translation_anchor = translation_delta.square().sum(-1).mean()
        rotation_acceleration = rotation_geodesic_acceleration(
            torch.cat((root_rotations[:, None], body_rotations), dim=1)
        )
        translation_accel = translation_acceleration(translation)
        transforms = runtime.joint_transforms(
            betas, body.reshape(frame_count, 69), root, translation
        )
        silhouette_terms: list[Tensor] = []
        boundary_terms: list[Tensor] = []
        for slot in _silhouette_slots(
            step, frame_count, min(settings.silhouette_batch_size, frame_count)
        ):
            posed = linear_blend_skinning(canonical_vertices, weights, transforms[slot])
            torch.manual_seed(config.seed + slot)
            predicted, _ = render_soft_mesh(
                posed,
                faces,
                intrinsics,
                (settings.render_resolution, settings.render_resolution),
                source_image_size=(initialization.image_height, initialization.image_width),
                sigma_pixels=config.model.renderer_sigma_pixels,
                sample_count=config.model.renderer_max_vertices,
                reference_sample_count=config.model.renderer_reference_sample_count,
                depth_temperature_m=config.model.renderer_depth_temperature_m,
            )
            silhouette_terms.append(silhouette_loss(predicted, masks[slot]))
            boundary_terms.append(differentiable_boundary_loss(predicted, masks[slot]))
        silhouette = torch.stack(silhouette_terms).mean()
        boundary = torch.stack(boundary_terms).mean()
        terms = {
            "observed_reprojection": observed_reprojection,
            "camerahmr_reprojection": camera_reprojection,
            "joint_3d_anchor": joint_3d_anchor,
            "silhouette": silhouette,
            "boundary": boundary,
            "pose_anchor": pose_anchor,
            "root_anchor": root_anchor,
            "translation_anchor": translation_anchor,
            "rotation_acceleration": rotation_acceleration,
            "translation_acceleration": translation_accel,
        }
        loss = (
            settings.observed_joint_weight * observed_reprojection
            + settings.camerahmr_joint_weight * camera_reprojection
            + settings.joint_3d_anchor_weight * joint_3d_anchor
            + settings.silhouette_weight * silhouette
            + settings.boundary_weight * boundary
            + settings.pose_anchor_weight * pose_anchor
            + settings.root_anchor_weight * root_anchor
            + settings.translation_anchor_weight * translation_anchor
            + settings.rotation_acceleration_weight * rotation_acceleration
            + settings.translation_acceleration_weight * translation_accel
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Pose-evidence refit produced a non-finite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_((body_raw, root_raw, translation_raw), 5.0)
        optimizer.step()
        final_terms = {name: float(value.detach().cpu()) for name, value in terms.items()}
        final_terms["total"] = float(loss.detach().cpu())

    with torch.no_grad():
        body_delta = bounded_vector(
            body_raw, math.radians(settings.maximum_joint_correction_degrees)
        )
        root_delta = bounded_vector(
            root_raw, math.radians(settings.maximum_root_correction_degrees)
        )
        translation_delta = bounded_vector(
            translation_raw, settings.maximum_translation_correction_m
        )
        body = anchor_body + body_delta
        root = anchor_root + root_delta
        translation = anchor_translation + translation_delta
        transforms = runtime.joint_transforms(
            betas, body.reshape(frame_count, 69), root, translation
        )

    artifact_root = output_directory / "initialization_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=False)
    transforms_path = artifact_root / "smpl_joint_transforms.npz"
    np.savez_compressed(
        transforms_path,
        source_frame_indices=np.asarray(
            [frame.source_frame_index for frame in frames], dtype=np.int64
        ),
        transforms=transforms.detach().cpu().numpy(),
    )
    refined_frames = [
        frame.model_copy(
            update={
                "body_pose": body[index].detach().cpu().reshape(-1).tolist(),
                "global_orient": root[index].detach().cpu().tolist(),
                "translation": translation[index].detach().cpu().tolist(),
            }
        )
        for index, frame in enumerate(frames)
    ]
    output_path = output_directory / "sequence_initialization.json"
    refined = initialization.model_copy(
        update={
            "frames": refined_frames,
            "joint_transforms_path": str(transforms_path),
            "blockers": [],
        }
    )
    write_json(output_path, refined)
    correction_degrees = torch.rad2deg(torch.linalg.vector_norm(body_delta, dim=-1))
    root_degrees = torch.rad2deg(torch.linalg.vector_norm(root_delta, dim=-1))
    translation_m = torch.linalg.vector_norm(translation_delta, dim=-1)
    report: dict[str, Any] = {
        "schema_version": "pose_evidence_refit.v1",
        "method": "bounded_full_smpl_pose_geman_mcclure_so3_acceleration_v1",
        "source_revision": os.environ.get("FRAYID_GIT_COMMIT", "unknown"),
        "input_initialization_path": str(input_path),
        "input_initialization_sha256": sha256_file(input_path),
        "output_initialization_path": str(output_path),
        "output_initialization_sha256": sha256_file(output_path),
        "joint_transforms_path": str(transforms_path),
        "joint_transforms_sha256": sha256_file(transforms_path),
        "frame_count": frame_count,
        "steps": step_count,
        "learning_rate": settings.learning_rate,
        "initial_total_loss": cast(float, initial_loss),
        "final_losses": final_terms,
        "maximum_joint_correction_degrees": float(correction_degrees.max().cpu()),
        "median_joint_correction_degrees": float(correction_degrees.median().cpu()),
        "maximum_root_correction_degrees": float(root_degrees.max().cpu()),
        "median_root_correction_degrees": float(root_degrees.median().cpu()),
        "maximum_translation_correction_m": float(translation_m.max().cpu()),
        "median_translation_correction_m": float(translation_m.median().cpu()),
        "shared_shape_frozen": True,
        "shared_camera_frozen": True,
        "held_out_labels_read": False,
        "held_out_metrics_read": False,
    }
    write_json(output_directory / "pose_evidence_refit_report.json", report)
    return refined, report
