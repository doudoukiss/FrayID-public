from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from torch import Tensor

from frayid.camera import axis_angle_to_matrix, make_intrinsics, rotation_angle_degrees
from frayid.config import ReconstructionConfig
from frayid.dataset import DATASET_MANIFEST_FILENAME, read_dataset_manifest
from frayid.geometry import linear_blend_skinning, vertex_normals
from frayid.io import read_json, write_json
from frayid.renderer import (
    differentiable_boundary_loss,
    normalized_boundary_error,
    render_soft_mesh,
    silhouette_loss,
    soft_silhouette_iou,
)
from frayid.schemas import (
    InitializationEvaluation,
    ObservedPoseSequence,
    SequenceInitialization,
)

INITIALIZATION_EVALUATION_FILENAME = "initialization_evaluation.json"
SMPL_BODY_JOINT_INDICES = (16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8)


class SMPLRuntime:
    """Thin, lazy wrapper around licensed local SMPL assets."""

    def __init__(self, model_root: Path, device: torch.device) -> None:
        try:
            import smplx
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("SMPL fitting requires the 'smpl' optional dependency") from exc
        if not model_root.is_dir():
            raise FileNotFoundError(f"Local SMPL model directory is missing: {model_root}")
        direct_model = next(
            (
                path
                for path in (model_root / "SMPL_NEUTRAL.pkl", model_root / "smpl_neutral.pkl")
                if path.is_file()
            ),
            None,
        )
        if direct_model is not None:
            try:
                self.model = smplx.SMPL(str(direct_model), gender="neutral").to(device)
            except ModuleNotFoundError as exc:
                if exc.name != "chumpy":
                    raise
                from smplx.utils import Struct

                data = _load_legacy_smpl_pickle(direct_model)
                self.model = smplx.SMPL("", gender="neutral", data_struct=Struct(**data)).to(device)
        else:
            self.model = smplx.create(
                str(model_root), model_type="smpl", gender="neutral", ext="pkl", use_pca=False
            ).to(device)
        self.faces = torch.as_tensor(self.model.faces.astype(np.int64), device=device)
        self.weights = self.model.lbs_weights.detach().to(device)

    def forward(
        self,
        betas: Tensor,
        body_pose: Tensor,
        global_orient: Tensor,
        translation: Tensor,
    ) -> Any:
        return self.model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=translation,
            return_verts=True,
        )

    def joint_transforms(
        self,
        betas: Tensor,
        body_pose: Tensor,
        global_orient: Tensor,
        translation: Tensor,
    ) -> Tensor:
        from smplx.lbs import (
            batch_rigid_transform,
            batch_rodrigues,
            blend_shapes,
            vertices2joints,
        )

        shaped = self.model.v_template + blend_shapes(betas, self.model.shapedirs)
        joints = vertices2joints(self.model.J_regressor, shaped)
        full_pose = torch.cat((global_orient, body_pose), dim=-1)
        rotations = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(full_pose.shape[0], -1, 3, 3)
        _, transforms = batch_rigid_transform(
            rotations, joints, self.model.parents, dtype=betas.dtype
        )
        transforms = transforms.clone()
        transforms[:, :, :3, 3] += translation[:, None, :]
        return cast(Tensor, transforms)


def load_initialization(path: Path) -> SequenceInitialization:
    return SequenceInitialization.model_validate(read_json(path))


def load_observed_pose(path: Path) -> ObservedPoseSequence:
    return ObservedPoseSequence.model_validate(read_json(path))


def validate_initialization_contract(
    initialization: SequenceInitialization,
    expected_source_indices: set[int] | None = None,
    *,
    require_shared: bool = True,
) -> list[str]:
    blockers: list[str] = []
    if initialization.proxy_camera:
        blockers.append("proxy_camera_forbidden")
    if initialization.zero_pose:
        blockers.append("zero_pose_packet_forbidden")
    if initialization.status == "blocked":
        blockers.append("initialization_declares_blocked")
    if not initialization.frames:
        blockers.append("initialization_has_no_frames")
        return blockers
    indices = [frame.source_frame_index for frame in initialization.frames]
    if len(indices) != len(set(indices)):
        blockers.append("duplicate_source_frame_indices")
    if expected_source_indices is not None:
        missing = expected_source_indices.difference(indices)
        if missing:
            blockers.append(f"initialization_missing_frames:{len(missing)}")
    shared_betas = np.asarray(initialization.shared_betas, dtype=np.float64)
    if not np.isfinite(shared_betas).all():
        blockers.append("nonfinite_shared_shape")
    if not math.isfinite(initialization.shared_focal_length_px):
        blockers.append("nonfinite_shared_focal")
    if initialization.shared_focal_length_px <= 0:
        blockers.append("nonpositive_shared_focal")
    for frame in initialization.frames:
        values = np.asarray(
            [*frame.betas, *frame.body_pose, *frame.global_orient, *frame.translation],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            blockers.append(f"nonfinite_frame_parameters:{frame.source_frame_index}")
        if np.linalg.norm(frame.body_pose) + np.linalg.norm(frame.global_orient) < 1e-6:
            blockers.append(f"zero_pose_forbidden:{frame.source_frame_index}")
        if require_shared and not np.allclose(
            frame.betas[: len(shared_betas)], shared_betas, atol=1e-4
        ):
            blockers.append(f"framewise_shape_drift:{frame.source_frame_index}")
        if require_shared and not math.isclose(
            frame.focal_length_px, initialization.shared_focal_length_px, rel_tol=1e-5
        ):
            blockers.append(f"framewise_focal_drift:{frame.source_frame_index}")
        if len(frame.keypoints_2d) == 0:
            blockers.append(f"missing_camerahmr_keypoints:{frame.source_frame_index}")
    return sorted(set(blockers))


def fit_initialization(
    config: ReconstructionConfig,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    model_root: Path,
    steps: int = 100,
    device_name: str = "cpu",
) -> SequenceInitialization:
    """Jointly refine shared shape/intrinsics and framewise SMPL state.

    The optimization consumes real CameraHMR keypoints and licensed local SMPL
    assets. It never fabricates a pose, camera, or body packet.
    """
    dataset_root = config.paths.dataset_root
    source_path = input_path or dataset_root / config.evidence.initialization_filename
    destination = output_path or source_path
    initialization = load_initialization(source_path)
    manifest = read_dataset_manifest(dataset_root / DATASET_MANIFEST_FILENAME)
    expected_indices = {frame.source_frame_index for frame in manifest.frames}
    blockers = validate_initialization_contract(
        initialization, expected_indices, require_shared=False
    )
    keypoint_blockers = [item for item in blockers if "keypoints" in item]
    other_blockers = [item for item in blockers if "keypoints" not in item]
    if other_blockers:
        raise ValueError("Initialization contract failed: " + ", ".join(other_blockers))
    if keypoint_blockers:
        raise ValueError("Real CameraHMR keypoints are required for fitting")
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = torch.device(device_name)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    runtime = SMPLRuntime(model_root, device)
    frames = sorted(initialization.frames, key=lambda item: item.source_frame_index)
    observed_pose_path = dataset_root / config.evidence.observed_pose_filename
    if not observed_pose_path.is_file():
        raise FileNotFoundError(f"Real Sapiens2 observed pose is required: {observed_pose_path}")
    observed_pose = load_observed_pose(observed_pose_path)
    observed_by_index = {frame.source_frame_index: frame for frame in observed_pose.frames}
    missing_observed = expected_indices.difference(observed_by_index)
    if missing_observed:
        raise ValueError(f"Sapiens2 observed pose is missing {len(missing_observed)} frames")
    frame_count = len(frames)
    pose_dimension = 69
    body_pose_data = np.stack([_pad_pose(frame.body_pose, pose_dimension) for frame in frames])
    median_betas = np.median(
        np.asarray([frame.betas[:10] for frame in frames], dtype=np.float32), axis=0
    )
    median_focal = float(np.median([frame.focal_length_px for frame in frames]))
    median_principal = np.median(
        np.asarray([frame.principal_point_px for frame in frames], dtype=np.float32), axis=0
    )
    shared_betas = torch.nn.Parameter(
        torch.tensor(median_betas, dtype=torch.float32, device=device)[None]
    )
    body_pose = torch.nn.Parameter(torch.tensor(body_pose_data, dtype=torch.float32, device=device))
    root = torch.nn.Parameter(
        torch.tensor([frame.global_orient for frame in frames], dtype=torch.float32, device=device)
    )
    translation = torch.nn.Parameter(
        torch.tensor([frame.translation for frame in frames], dtype=torch.float32, device=device)
    )
    log_focal = torch.nn.Parameter(torch.tensor(math.log(median_focal), device=device))
    principal = torch.nn.Parameter(
        torch.tensor(median_principal, dtype=torch.float32, device=device)
    )
    envelope_logits = torch.nn.Parameter(
        torch.zeros(runtime.weights.shape[0], dtype=torch.float32, device=device)
    )
    parameters = [
        shared_betas,
        body_pose,
        root,
        translation,
        log_focal,
        principal,
        envelope_logits,
    ]
    optimizer = torch.optim.Adam(parameters, lr=2e-3)
    initial_betas = shared_betas.detach().clone()
    initial_body_pose = body_pose.detach().clone()
    initial_root = root.detach().clone()
    initial_translation = translation.detach().clone()
    initial_focal = log_focal.detach().clone()
    initial_principal = principal.detach().clone()
    mesh_edges = _unique_mesh_edges(runtime.faces)
    records_by_index = {record.source_frame_index: record for record in manifest.frames}
    silhouette_slots = np.linspace(
        0,
        frame_count - 1,
        min(config.initialization_gate.overlay_frame_count, frame_count),
        dtype=np.int64,
    ).tolist()
    fit_resolution = min(96, config.model.renderer_resolution)
    silhouette_targets: dict[int, Tensor] = {}
    for slot in silhouette_slots:
        frame = frames[slot]
        record = records_by_index[frame.source_frame_index]
        mask_path = dataset_root / config.evidence.masks_subdirectory / Path(record.image_path).name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(
                f"Real Sapiens2 mask is required for initialization fit: {mask_path}"
            )
        mask_tensor = torch.tensor(mask / 255.0, dtype=torch.float32, device=device)
        silhouette_targets[slot] = torch.nn.functional.interpolate(
            mask_tensor[None, None],
            size=(fit_resolution, fit_resolution),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    image_diagonal = math.hypot(initialization.image_width, initialization.image_height)
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
    if camera_keypoints.ndim != 3 or camera_keypoints.shape[2] < 2:
        raise ValueError("CameraHMR keypoints must have shape (frames, joints, 2+)")
    if raw_joints.ndim != 3 or raw_joints.shape[2] != 3:
        raise ValueError("CameraHMR 3D joints must have shape (frames, joints, 3)")
    observed_confidence = observed_keypoints[:, :, 2]
    observed_valid = (
        observed_confidence >= config.initialization_gate.observed_pose_minimum_confidence
    )
    insufficient_observed = observed_valid.sum(dim=1) < 6
    if torch.any(insufficient_observed):
        bad_index = int(torch.nonzero(insufficient_observed, as_tuple=False)[0, 0].item())
        raise ValueError(
            "Sapiens2 observed pose has fewer than six confident body joints for frame "
            f"{frames[bad_index].source_frame_index}"
        )
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        repeated_betas = shared_betas.expand(frame_count, -1)
        output = runtime.forward(repeated_betas, body_pose, root, translation)
        canonical = runtime.forward(
            shared_betas,
            torch.zeros((1, pose_dimension), device=device),
            torch.zeros((1, 3), device=device),
            torch.zeros((1, 3), device=device),
        )
        canonical_envelope, envelope_offsets = shared_normal_envelope(
            canonical.vertices[0],
            runtime.faces,
            envelope_logits,
            config.initialization_gate.envelope_maximum_offset_m,
        )
        focal = log_focal.exp()
        intrinsics = make_intrinsics(focal, principal, device=device)
        camera_joint_count = min(camera_keypoints.shape[1], output.joints.shape[1])
        camera_projected = _project(output.joints[:, :camera_joint_count], intrinsics)
        camera_confidence = (
            camera_keypoints[:, :camera_joint_count, 2]
            if camera_keypoints.shape[2] >= 3
            else torch.ones_like(camera_projected[:, :, 0])
        )
        reprojection = (
            (
                torch.nn.functional.smooth_l1_loss(
                    camera_projected,
                    camera_keypoints[:, :camera_joint_count, :2],
                    reduction="none",
                ).sum(-1)
                * camera_confidence
            ).sum(dim=1)
            / camera_confidence.sum(dim=1).clamp_min(1.0)
        ).mean() / image_diagonal
        predicted_observed = _project(output.joints[:, list(SMPL_BODY_JOINT_INDICES)], intrinsics)
        observed_per_joint = torch.nn.functional.smooth_l1_loss(
            predicted_observed, observed_keypoints[:, :, :2], reduction="none"
        ).sum(-1)
        observed_weights = observed_confidence * observed_valid
        observed_reprojection = (
            (observed_per_joint * observed_weights).sum(dim=1)
            / observed_weights.sum(dim=1).clamp_min(1.0)
        ).mean() / image_diagonal
        raw_joint_count = min(raw_joints.shape[1], output.joints.shape[1])
        joint_3d = torch.linalg.vector_norm(
            output.joints[:, :raw_joint_count] - raw_joints[:, :raw_joint_count], dim=-1
        ).mean()
        active_silhouette_slots = [
            silhouette_slots[(step + offset) % len(silhouette_slots)] for offset in range(4)
        ]
        silhouette_losses: list[Tensor] = []
        boundary_losses: list[Tensor] = []
        for silhouette_slot in active_silhouette_slots:
            transforms = runtime.joint_transforms(
                shared_betas,
                body_pose[silhouette_slot : silhouette_slot + 1],
                root[silhouette_slot : silhouette_slot + 1],
                translation[silhouette_slot : silhouette_slot + 1],
            )
            posed_envelope = linear_blend_skinning(
                canonical_envelope, runtime.weights, transforms[0]
            )
            torch.manual_seed(config.seed + silhouette_slot)
            predicted_silhouette, _ = render_soft_mesh(
                posed_envelope,
                runtime.faces,
                intrinsics,
                (fit_resolution, fit_resolution),
                source_image_size=(initialization.image_height, initialization.image_width),
                sigma_pixels=config.model.renderer_sigma_pixels,
                sample_count=config.model.renderer_max_vertices,
                reference_sample_count=config.model.renderer_reference_sample_count,
            )
            target_silhouette = silhouette_targets[silhouette_slot]
            silhouette_losses.append(silhouette_loss(predicted_silhouette, target_silhouette))
            boundary_losses.append(
                differentiable_boundary_loss(predicted_silhouette, target_silhouette)
            )
        silhouette = torch.stack(silhouette_losses).mean()
        boundary = torch.stack(boundary_losses).mean()
        temporal = (
            _second_difference(root)
            + _second_difference(translation)
            + 0.1 * _second_difference(body_pose)
        )
        shape_prior = (shared_betas - initial_betas).square().mean()
        camera_prior = (log_focal - initial_focal).square()
        principal_prior = ((principal - initial_principal) / image_diagonal).square().mean()
        pose_prior = (body_pose - initial_body_pose).square().mean()
        root_prior = (root - initial_root).square().mean()
        translation_prior = (translation - initial_translation).square().mean()
        envelope_magnitude = envelope_offsets.square().mean()
        envelope_smoothness = (
            (envelope_offsets[mesh_edges[:, 0]] - envelope_offsets[mesh_edges[:, 1]])
            .square()
            .mean()
        )
        envelope_inward = torch.relu(-envelope_offsets).square().mean()
        loss = (
            15.0 * observed_reprojection
            + 2.0 * reprojection
            + 0.5 * joint_3d
            + 4.0 * silhouette
            + 0.5 * boundary
            + 0.2 * temporal
            + 0.05 * shape_prior
            + camera_prior
            + principal_prior
            + 0.5 * pose_prior
            + 0.5 * root_prior
            + translation_prior
            + config.initialization_gate.envelope_magnitude_weight * envelope_magnitude
            + config.initialization_gate.envelope_smoothness_weight * envelope_smoothness
            + 2.0 * envelope_inward
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Initialization fitting produced a non-finite loss")
        loss.backward()
        optimizer.step()

    refined_betas = shared_betas.detach().cpu().reshape(-1).tolist()
    refined_focal = float(log_focal.detach().exp().cpu())
    refined_principal = principal.detach().cpu().tolist()
    refined_frames = []
    for index, frame in enumerate(frames):
        refined_frames.append(
            frame.model_copy(
                update={
                    "betas": refined_betas,
                    "body_pose": body_pose[index].detach().cpu().tolist(),
                    "global_orient": root[index].detach().cpu().tolist(),
                    "translation": translation[index].detach().cpu().tolist(),
                    "focal_length_px": refined_focal,
                    "principal_point_px": refined_principal,
                }
            )
        )
    canonical = runtime.forward(
        shared_betas.detach(),
        torch.zeros((1, pose_dimension), device=device),
        torch.zeros((1, 3), device=device),
        torch.zeros((1, 3), device=device),
    )
    canonical_envelope, envelope_offsets = shared_normal_envelope(
        canonical.vertices[0],
        runtime.faces,
        envelope_logits.detach(),
        config.initialization_gate.envelope_maximum_offset_m,
    )
    artifact_root = destination.parent / "initialization_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    mesh_path = artifact_root / "shared_clothing_envelope.npz"
    weights_path = artifact_root / "smpl_skinning_weights.npz"
    transforms_path = artifact_root / "smpl_joint_transforms.npz"
    np.savez_compressed(
        mesh_path,
        vertices=canonical_envelope.detach().cpu().numpy(),
        faces=runtime.faces.detach().cpu().numpy(),
    )
    np.savez_compressed(weights_path, weights=runtime.weights.detach().cpu().numpy())
    transforms = runtime.joint_transforms(
        shared_betas.detach().expand(frame_count, -1),
        body_pose.detach(),
        root.detach(),
        translation.detach(),
    )
    np.savez_compressed(
        transforms_path,
        source_frame_indices=np.asarray(
            [frame.source_frame_index for frame in frames], dtype=np.int64
        ),
        transforms=transforms.detach().cpu().numpy(),
    )
    refined = initialization.model_copy(
        update={
            "status": "refined",
            "shared_betas": refined_betas,
            "shared_focal_length_px": refined_focal,
            "shared_principal_point_px": refined_principal,
            "frames": refined_frames,
            "canonical_mesh_path": str(mesh_path),
            "canonical_mesh_role": "shared_clothing_envelope",
            "skinning_weights_path": str(weights_path),
            "joint_transforms_path": str(transforms_path),
            "envelope_maximum_offset_m": float(envelope_offsets.abs().max().detach().cpu()),
            "envelope_rms_offset_m": float(
                torch.sqrt(envelope_offsets.square().mean()).detach().cpu()
            ),
            "envelope_laplacian_rms_m": float(
                torch.sqrt(
                    (envelope_offsets[mesh_edges[:, 0]] - envelope_offsets[mesh_edges[:, 1]])
                    .square()
                    .mean()
                )
                .detach()
                .cpu()
            ),
            "blockers": [],
        }
    )
    write_json(destination, refined)
    return refined


def evaluate_initialization(
    config: ReconstructionConfig,
    *,
    initialization_path: Path | None = None,
    model_root: Path,
    device_name: str = "cpu",
    output_path: Path | None = None,
) -> InitializationEvaluation:
    dataset_root = config.paths.dataset_root
    path = initialization_path or dataset_root / config.evidence.initialization_filename
    initialization = load_initialization(path)
    manifest = read_dataset_manifest(dataset_root / DATASET_MANIFEST_FILENAME)
    expected_indices = {frame.source_frame_index for frame in manifest.frames}
    blockers = validate_initialization_contract(initialization, expected_indices)
    if blockers:
        report = InitializationEvaluation(
            status="blocked",
            shared_focal=not any("focal_drift" in item for item in blockers),
            shared_shape=not any("shape_drift" in item for item in blockers),
            evaluated_frame_count=0,
            blockers=blockers,
        )
        write_json(output_path or dataset_root / INITIALIZATION_EVALUATION_FILENAME, report)
        return report

    device = torch.device(device_name)
    runtime = SMPLRuntime(model_root, device)
    frames_by_index = {frame.source_frame_index: frame for frame in initialization.frames}
    observed_pose_path = dataset_root / config.evidence.observed_pose_filename
    if not observed_pose_path.is_file():
        blockers.append("missing_real_sapiens2_observed_pose")
        observed_by_index = {}
    else:
        observed_pose = load_observed_pose(observed_pose_path)
        observed_by_index = {frame.source_frame_index: frame for frame in observed_pose.frames}
    artifact_data = _load_initialization_artifacts(initialization, device)
    distributed = _distributed(manifest.frames, config.initialization_gate.overlay_frame_count)
    ious: list[float] = []
    boundaries: list[float] = []
    keypoint_errors: list[float] = []
    for record in distributed:
        frame = frames_by_index[record.source_frame_index]
        output = runtime.forward(
            torch.tensor(frame.betas[:10], dtype=torch.float32, device=device)[None],
            torch.tensor(_pad_pose(frame.body_pose, 69), dtype=torch.float32, device=device)[None],
            torch.tensor(frame.global_orient, dtype=torch.float32, device=device)[None],
            torch.tensor(frame.translation, dtype=torch.float32, device=device)[None],
        )
        vertices = output.vertices[0]
        if artifact_data is not None:
            canonical_vertices, weights, transforms_by_index = artifact_data
            transform = transforms_by_index.get(record.source_frame_index)
            if transform is None:
                blockers.append(f"missing_joint_transform:{record.source_frame_index}")
                continue
            vertices = linear_blend_skinning(canonical_vertices, weights, transform)
        mask_path = dataset_root / config.evidence.masks_subdirectory / Path(record.image_path).name
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            blockers.append(f"missing_real_sapiens2_mask:{record.source_frame_index}")
            continue
        target = torch.from_numpy(mask_image).to(device=device, dtype=torch.float32) / 255.0
        render_size = (config.model.renderer_resolution, config.model.renderer_resolution)
        target = torch.nn.functional.interpolate(
            target[None, None], size=render_size, mode="bilinear", align_corners=False
        )[0, 0]
        intrinsics = make_intrinsics(
            initialization.shared_focal_length_px,
            initialization.shared_principal_point_px,
            device=device,
        )
        torch.manual_seed(config.seed + record.ordinal)
        silhouette, _ = render_soft_mesh(
            vertices,
            runtime.faces,
            intrinsics,
            render_size,
            source_image_size=(initialization.image_height, initialization.image_width),
            sigma_pixels=config.model.renderer_sigma_pixels,
            sample_count=config.model.renderer_max_vertices,
            reference_sample_count=config.model.renderer_reference_sample_count,
            depth_temperature_m=config.model.renderer_depth_temperature_m,
        )
        ious.append(float(soft_silhouette_iou(silhouette, target).detach().cpu()))
        boundaries.append(normalized_boundary_error(silhouette, target))
        observed_frame = observed_by_index.get(record.source_frame_index)
        if observed_frame is None:
            blockers.append(f"missing_observed_pose:{record.source_frame_index}")
        else:
            keypoints = torch.tensor(
                observed_frame.keypoints_body12, dtype=torch.float32, device=device
            )
            projected = _project(output.joints[0, list(SMPL_BODY_JOINT_INDICES)], intrinsics)
            valid = keypoints[:, 2] >= config.initialization_gate.observed_pose_minimum_confidence
            errors = torch.linalg.vector_norm(projected[valid] - keypoints[valid, :2], dim=-1)
            if errors.numel():
                keypoint_errors.extend(errors.detach().cpu().tolist())

    rotations = axis_angle_to_matrix(
        torch.tensor(
            [
                frame.global_orient
                for frame in sorted(initialization.frames, key=lambda item: item.source_frame_index)
            ],
            dtype=torch.float32,
        )
    )
    jumps = rotation_angle_degrees(rotations[:-1], rotations[1:])
    maximum_jump = float(jumps.max()) if jumps.numel() else 0.0
    median_iou = float(np.median(ious)) if ious else None
    if median_iou is None:
        blockers.append("no_initialization_silhouette_metrics")
    elif median_iou < config.initialization_gate.minimum_median_silhouette_iou:
        blockers.append("median_initialization_silhouette_iou_below_gate")
    if maximum_jump > config.initialization_gate.maximum_root_rotation_jump_degrees:
        blockers.append("root_rotation_discontinuity")
    report = InitializationEvaluation(
        status="pass" if not blockers else ("blocked" if not ious else "fail"),
        median_silhouette_iou=median_iou,
        median_normalized_boundary_error=(float(np.median(boundaries)) if boundaries else None),
        median_keypoint_reprojection_error_px=(
            float(np.median(keypoint_errors)) if keypoint_errors else None
        ),
        keypoint_source="sapiens2_pose" if keypoint_errors else None,
        maximum_root_rotation_jump_degrees=maximum_jump,
        shared_focal=True,
        shared_shape=True,
        evaluated_frame_count=len(ious),
        blockers=sorted(set(blockers)),
    )
    write_json(output_path or dataset_root / INITIALIZATION_EVALUATION_FILENAME, report)
    return report


def _project(points: Tensor, intrinsics: Tensor) -> Tensor:
    z = points[..., 2].clamp_min(1e-5)
    return torch.stack(
        (
            points[..., 0] / z * intrinsics[0, 0] + intrinsics[0, 2],
            points[..., 1] / z * intrinsics[1, 1] + intrinsics[1, 2],
        ),
        dim=-1,
    )


def _pad_pose(values: list[float], size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size > size:
        raise ValueError(f"Body pose has {array.size} values; expected at most {size}")
    return np.pad(array, (0, size - array.size))


def _second_difference(values: Tensor) -> Tensor:
    if values.shape[0] < 3:
        return values.sum() * 0.0
    return (values[2:] - 2.0 * values[1:-1] + values[:-2]).square().mean()


def shared_normal_envelope(
    vertices: Tensor,
    faces: Tensor,
    offset_logits: Tensor,
    maximum_offset_m: float,
) -> tuple[Tensor, Tensor]:
    """Apply one bounded canonical normal-offset field shared by every frame."""
    if offset_logits.shape != (vertices.shape[0],):
        raise ValueError("offset_logits must provide one scalar per vertex")
    offsets = torch.tanh(offset_logits) * maximum_offset_m
    normals = vertex_normals(vertices, faces).detach()
    return vertices + normals * offsets[:, None], offsets


def pose_shared_displacements(
    canonical_displacements: Tensor,
    weights: Tensor,
    joint_transforms: Tensor,
) -> Tensor:
    """Rotate shared canonical displacements with the same SMPL/LBS transform."""
    if joint_transforms.ndim != 3:
        raise ValueError("joint_transforms must have shape (J, 4, 4)")
    blended_rotations = torch.einsum("vj,jxy->vxy", weights, joint_transforms[:, :3, :3])
    return torch.einsum("vxy,vy->vx", blended_rotations, canonical_displacements)


def _unique_mesh_edges(faces: Tensor) -> Tensor:
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0)
    return cast(Tensor, torch.unique(torch.sort(edges, dim=-1).values, dim=0))


def _load_initialization_artifacts(
    initialization: SequenceInitialization,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[int, Tensor]] | None:
    if initialization.canonical_mesh_role != "shared_clothing_envelope":
        return None
    paths = (
        initialization.canonical_mesh_path,
        initialization.skinning_weights_path,
        initialization.joint_transforms_path,
    )
    if not all(paths):
        raise ValueError("Shared clothing envelope is missing required SMPL artifacts")
    mesh = np.load(cast(str, initialization.canonical_mesh_path))
    weight_data = np.load(cast(str, initialization.skinning_weights_path))
    transform_data = np.load(cast(str, initialization.joint_transforms_path))
    canonical_vertices = torch.tensor(mesh["vertices"], dtype=torch.float32, device=device)
    weights = torch.tensor(weight_data["weights"], dtype=torch.float32, device=device)
    source_indices = transform_data["source_frame_indices"].tolist()
    transforms = torch.tensor(transform_data["transforms"], dtype=torch.float32, device=device)
    return (
        canonical_vertices,
        weights,
        {int(source_index): transforms[index] for index, source_index in enumerate(source_indices)},
    )


def _distributed(values: list[Any], count: int) -> list[Any]:
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return [values[index] for index in positions.tolist()]


class _LegacyChumpyArray:
    """Enough of chumpy's pickle state to convert old licensed SMPL assets."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.state = state

    def as_array(self) -> np.ndarray:
        value = self.state.get("x")
        if value is None:
            raise ValueError("Unsupported legacy chumpy state in SMPL asset")
        return np.asarray(value)


class _SMPLUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "chumpy.ch" and name == "Ch":
            return _LegacyChumpyArray
        return super().find_class(module, name)


def _load_legacy_smpl_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = _SMPLUnpickler(stream, encoding="latin1").load()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an SMPL parameter dictionary: {path}")
    return {
        key: value.as_array() if isinstance(value, _LegacyChumpyArray) else value
        for key, value in payload.items()
    }
