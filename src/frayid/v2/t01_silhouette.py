from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from frayid.camera import axis_angle_to_matrix, make_intrinsics
from frayid.dataset import read_dataset_manifest
from frayid.geometry import linear_blend_skinning
from frayid.io import read_json, sha256_file, write_json
from frayid.renderer import normalized_boundary_error, render_soft_mesh, soft_silhouette_iou
from frayid.schemas import SequenceInitialization
from frayid.v2.checkpoint import capture_checkpoint, restore_checkpoint
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import TurntableSolution
from frayid.v2.t01_phase import BoundedPhaseAxisBlock
from frayid.v2.track_factors import load_pairwise_tracklet_factors
from frayid.v2.turntable import axis_angle_rotation

CENTER_TRUST_RADII = (2.0, 1.0, 0.5, 0.25, 0.1, 0.0)
FOCAL_TRUST_RADII = (2.0, 1.0, 0.5, 0.25, 0.1, 0.0)


class T01CenterFocalBlock(nn.Module):
    """Fit bounded center/focal from mask boundary moments with phase frozen."""

    posed_body_vertices: Tensor
    rotations: Tensor
    target_bboxes: Tensor
    base_center: Tensor
    center_bounds: Tensor
    principal_point: Tensor

    def __init__(
        self,
        posed_body_vertices: Tensor,
        rotations: Tensor,
        target_bboxes: Tensor,
        *,
        base_center: Tensor,
        base_focal: float,
        principal_point: Tensor,
        center_bounds: tuple[float, float, float] = (0.08, 0.08, 0.15),
        maximum_focal_relative_change: float = 0.10,
        soft_extrema_temperature_pixels: float = 3.0,
    ) -> None:
        super().__init__()
        if posed_body_vertices.ndim != 3 or posed_body_vertices.shape[-1] != 3:
            raise ValueError("posed body vertices must have shape [sample, vertex, 3]")
        if rotations.shape != (posed_body_vertices.shape[0], 3, 3):
            raise ValueError("one frozen turntable rotation is required per silhouette sample")
        if target_bboxes.shape != (posed_body_vertices.shape[0], 4):
            raise ValueError("target mask boxes must have shape [sample, 4]")
        if base_center.shape != (3,) or principal_point.shape != (2,):
            raise ValueError("center and principal point have invalid shape")
        if base_focal <= 0 or maximum_focal_relative_change <= 0:
            raise ValueError("focal values and bounds must be positive")
        if soft_extrema_temperature_pixels <= 0:
            raise ValueError("soft extrema temperature must be positive")
        self.center_raw = nn.Parameter(torch.zeros(3))
        self.focal_raw = nn.Parameter(torch.zeros(()))
        self.register_buffer("posed_body_vertices", posed_body_vertices)
        self.register_buffer("rotations", rotations)
        self.register_buffer("target_bboxes", target_bboxes)
        self.register_buffer("base_center", base_center)
        self.register_buffer("center_bounds", torch.tensor(center_bounds, dtype=torch.float32))
        self.register_buffer("principal_point", principal_point)
        self.base_focal = base_focal
        self.maximum_log_focal_change = math.log1p(maximum_focal_relative_change)
        self.soft_extrema_temperature_pixels = soft_extrema_temperature_pixels

    @property
    def center(self) -> Tensor:
        return self.base_center + self.center_bounds * torch.tanh(self.center_raw)

    @property
    def focal(self) -> Tensor:
        return self.focal_raw.new_tensor(self.base_focal) * torch.exp(
            self.maximum_log_focal_change * torch.tanh(self.focal_raw)
        )

    def camera_vertices(self) -> Tensor:
        return (
            torch.einsum("sij,svj->svi", self.rotations, self.posed_body_vertices)
            + self.center[None, None, :]
        )

    def projected_vertices(self) -> Tensor:
        camera = self.camera_vertices()
        depth = camera[..., 2].clamp_min(0.1)
        return torch.stack(
            (
                self.focal * camera[..., 0] / depth + self.principal_point[0],
                self.focal * camera[..., 1] / depth + self.principal_point[1],
            ),
            dim=-1,
        )

    def soft_bboxes(self) -> Tensor:
        pixels = self.projected_vertices()
        temperature = self.soft_extrema_temperature_pixels
        x = pixels[..., 0]
        y = pixels[..., 1]
        return torch.stack(
            (
                -temperature * torch.logsumexp(-x / temperature, dim=1),
                -temperature * torch.logsumexp(-y / temperature, dim=1),
                temperature * torch.logsumexp(x / temperature, dim=1),
                temperature * torch.logsumexp(y / temperature, dim=1),
            ),
            dim=-1,
        )

    def moment_loss(self, *, image_size: tuple[int, int]) -> Tensor:
        height, width = image_size
        if height <= 0 or width <= 0:
            raise ValueError("source image size must be positive")
        diagonal = float((height * height + width * width) ** 0.5)
        predicted = self.soft_bboxes()
        target = self.target_bboxes
        edge = F.smooth_l1_loss(predicted / diagonal, target / diagonal)
        predicted_center = 0.5 * (predicted[:, :2] + predicted[:, 2:])
        target_center = 0.5 * (target[:, :2] + target[:, 2:])
        center = F.smooth_l1_loss(predicted_center / diagonal, target_center / diagonal)
        predicted_size = torch.log((predicted[:, 2:] - predicted[:, :2]).clamp_min(1.0))
        target_size = torch.log((target[:, 2:] - target[:, :2]).clamp_min(1.0))
        scale = F.smooth_l1_loss(predicted_size, target_size)
        return edge + 0.5 * center + 0.25 * scale

    def center_offset_norm(self) -> float:
        return float(torch.linalg.vector_norm(self.center - self.base_center).detach())

    def focal_relative_change(self) -> float:
        return abs(float(self.focal.detach()) / self.base_focal - 1.0)


@dataclass(frozen=True)
class CenterFocalStep:
    initial_loss: float
    accepted_loss: float
    center_radius: float
    focal_radius: float
    candidate_evaluations: int
    evidence_improvement_fraction: float
    center_offset_norm: float
    focal_relative_change: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "initial_loss": self.initial_loss,
            "accepted_loss": self.accepted_loss,
            "center_radius": self.center_radius,
            "focal_radius": self.focal_radius,
            "candidate_evaluations": self.candidate_evaluations,
            "evidence_improvement_fraction": self.evidence_improvement_fraction,
            "center_offset_norm": self.center_offset_norm,
            "focal_relative_change": self.focal_relative_change,
        }


def _descent(value: Tensor) -> Tensor:
    gradient = value.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise ValueError("center/focal gradient is absent or nonfinite")
    norm = torch.linalg.vector_norm(gradient)
    if float(norm) <= 0:
        return torch.zeros_like(gradient)
    result: Tensor = -gradient / norm
    return result


def take_center_focal_trust_region_step(
    model: T01CenterFocalBlock,
    *,
    image_size: tuple[int, int],
) -> CenterFocalStep:
    model.zero_grad(set_to_none=True)
    initial_tensor = model.moment_loss(image_size=image_size)
    initial_tensor.backward()  # type: ignore[no-untyped-call]
    center_direction = _descent(model.center_raw).detach()
    focal_direction = _descent(model.focal_raw).detach()
    initial_loss = float(initial_tensor.detach())
    best_loss = initial_loss
    best_center = model.center_raw.detach().clone()
    best_focal = model.focal_raw.detach().clone()
    best_radii = (0.0, 0.0)
    evaluations = 0
    with torch.no_grad():
        for center_radius in CENTER_TRUST_RADII:
            for focal_radius in FOCAL_TRUST_RADII:
                model.center_raw.copy_(center_direction * center_radius)
                model.focal_raw.copy_(focal_direction * focal_radius)
                candidate = float(model.moment_loss(image_size=image_size))
                evaluations += 1
                if math.isfinite(candidate) and candidate < best_loss:
                    best_loss = candidate
                    best_center = model.center_raw.detach().clone()
                    best_focal = model.focal_raw.detach().clone()
                    best_radii = (center_radius, focal_radius)
        model.center_raw.copy_(best_center)
        model.focal_raw.copy_(best_focal)
    if best_loss >= initial_loss:
        raise ValueError("bounded center/focal trust region found no improving step")
    return CenterFocalStep(
        initial_loss=initial_loss,
        accepted_loss=best_loss,
        center_radius=best_radii[0],
        focal_radius=best_radii[1],
        candidate_evaluations=evaluations,
        evidence_improvement_fraction=(initial_loss - best_loss) / initial_loss,
        center_offset_norm=model.center_offset_norm(),
        focal_relative_change=model.focal_relative_change(),
    )


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _mask_bbox(mask: np.ndarray) -> list[float]:
    foreground = mask >= 128
    y, x = np.nonzero(foreground)
    if len(x) == 0:
        raise ValueError("training mask has no foreground")
    return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]


def _independent_silhouette_metrics(
    model: T01CenterFocalBlock,
    faces: Tensor,
    masks: list[Tensor],
    *,
    source_image_size: tuple[int, int],
    render_resolution: int,
    seed: int,
) -> dict[str, float]:
    ious: list[float] = []
    boundaries: list[float] = []
    intrinsics = make_intrinsics(model.focal, model.principal_point)
    for slot, (vertices, mask) in enumerate(zip(model.camera_vertices(), masks, strict=True)):
        torch.manual_seed(seed + slot)
        silhouette, _ = render_soft_mesh(
            vertices,
            faces,
            intrinsics,
            (render_resolution, render_resolution),
            source_image_size=source_image_size,
            sigma_pixels=1.75 * render_resolution / 128.0,
            sample_count=2048,
            reference_sample_count=2048,
        )
        ious.append(float(soft_silhouette_iou(silhouette, mask).detach()))
        boundaries.append(normalized_boundary_error(silhouette, mask))
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_normalized_boundary_error": float(np.mean(boundaries)),
    }


def qualify_real_center_focal_step(
    solution_path: Path,
    factor_binding_path: Path,
    phase_checkpoint_path: Path,
    initialization_path: Path,
    manifest_path: Path,
    mask_root: Path,
    canonical_mesh_path: Path,
    skinning_weights_path: Path,
    joint_transforms_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    source_image_size: tuple[int, int] = (1120, 720),
    sample_frame_count: int = 8,
    render_resolution: int = 64,
    device: str = "cpu",
    seed: int = 20260902,
) -> Path:
    """Run one bounded center/focal engineering step from train masks only."""

    paths = [
        solution_path,
        factor_binding_path,
        phase_checkpoint_path,
        initialization_path,
        manifest_path,
        mask_root,
        canonical_mesh_path,
        skinning_weights_path,
        joint_transforms_path,
        output_path,
        checkpoint_path,
    ]
    reject_sealed_capability(paths)
    if device != "cpu":
        raise ValueError("T01 center/focal qualification is registered for Mac CPU")
    if checkpoint_path.exists():
        raise FileExistsError(f"T01 center/focal checkpoint already exists: {checkpoint_path}")
    solution = TurntableSolution.model_validate(read_json(solution_path))
    factors = load_pairwise_tracklet_factors(factor_binding_path, device=device)
    phase_model = BoundedPhaseAxisBlock(solution, factors).to(device)
    phase_optimizer = torch.optim.SGD(phase_model.parameters(), lr=1.0)
    restore_checkpoint(
        phase_checkpoint_path.read_bytes(), phase_model, phase_optimizer, device=device
    )
    initialization = SequenceInitialization.model_validate(read_json(initialization_path))
    initialization_by_source = {frame.source_frame_index: frame for frame in initialization.frames}
    manifest = read_dataset_manifest(manifest_path)
    record_by_source = {frame.source_frame_index: frame for frame in manifest.frames}
    if sample_frame_count < 3 or sample_frame_count > len(solution.source_frame_indices):
        raise ValueError("center/focal silhouette sample count is invalid")
    sample_slots = np.linspace(
        0,
        len(solution.source_frame_indices) - 1,
        sample_frame_count,
        dtype=np.int64,
    ).tolist()
    with np.load(canonical_mesh_path, allow_pickle=False) as archive:
        canonical_vertices = torch.as_tensor(archive["vertices"], dtype=torch.float32)
        faces = torch.as_tensor(archive["faces"], dtype=torch.long)
    with np.load(skinning_weights_path, allow_pickle=False) as archive:
        weights = torch.as_tensor(archive["weights"], dtype=torch.float32)
    with np.load(joint_transforms_path, allow_pickle=False) as archive:
        transform_sources = archive["source_frame_indices"].astype(np.int64)
        transforms = torch.as_tensor(archive["transforms"], dtype=torch.float32)
    transform_slot = {int(source): slot for slot, source in enumerate(transform_sources)}
    first_source = solution.source_frame_indices[0]
    base_rotation = axis_angle_to_matrix(
        torch.tensor(initialization_by_source[first_source].global_orient, dtype=torch.float32)
    )
    posed_body: list[Tensor] = []
    target_boxes: list[list[float]] = []
    masks: list[Tensor] = []
    selected_sources: list[int] = []
    for sample_index, phase_slot in enumerate(sample_slots):
        source = solution.source_frame_indices[phase_slot]
        record = record_by_source[source]
        if record.split != "train" or not record.quality_accepted:
            raise ValueError("center/focal qualification selected nontraining evidence")
        frame = initialization_by_source[source]
        posed_camera = linear_blend_skinning(
            canonical_vertices,
            weights,
            transforms[transform_slot[source]],
        )
        root_rotation = axis_angle_to_matrix(torch.tensor(frame.global_orient, dtype=torch.float32))
        translation = torch.tensor(frame.translation, dtype=torch.float32)
        local_pose = (posed_camera - translation) @ root_rotation
        posed_body.append(local_pose @ base_rotation.T)
        mask_path = mask_root / Path(record.image_path).name
        reject_sealed_capability([mask_path])
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"training mask is absent: {mask_path}")
        target_boxes.append(_mask_bbox(raw_mask))
        resized = cv2.resize(
            raw_mask,
            (render_resolution, render_resolution),
            interpolation=cv2.INTER_AREA,
        )
        masks.append(torch.tensor(resized / 255.0, dtype=torch.float32))
        selected_sources.append(source)
        torch.manual_seed(seed + sample_index)
    selected_slot_tensor = torch.tensor(sample_slots, dtype=torch.long)
    block = T01CenterFocalBlock(
        torch.stack(posed_body),
        axis_angle_rotation(phase_model.axis.detach(), phase_model.angles.detach())[
            selected_slot_tensor
        ],
        torch.tensor(target_boxes, dtype=torch.float32),
        base_center=torch.tensor(solution.center, dtype=torch.float32),
        base_focal=float(solution.shared_intrinsics[0][0]),
        principal_point=torch.tensor(
            [solution.shared_intrinsics[0][2], solution.shared_intrinsics[1][2]],
            dtype=torch.float32,
        ),
    ).to(device)
    optimizer = torch.optim.SGD(block.parameters(), lr=1.0)
    initial_checkpoint = capture_checkpoint(
        block, optimizer, step=0, topology_connectivity_sha256=None
    )
    independent_before = _independent_silhouette_metrics(
        block,
        faces,
        masks,
        source_image_size=source_image_size,
        render_resolution=render_resolution,
        seed=seed,
    )
    step = take_center_focal_trust_region_step(block, image_size=source_image_size)
    independent_after = _independent_silhouette_metrics(
        block,
        faces,
        masks,
        source_image_size=source_image_size,
        render_resolution=render_resolution,
        seed=seed,
    )
    accepted_center = block.center.detach().cpu()
    accepted_focal = float(block.focal.detach())
    accepted_state_hash = _state_sha256(block)
    accepted_checkpoint = capture_checkpoint(
        block, optimizer, step=1, topology_connectivity_sha256=None
    )
    restore_checkpoint(initial_checkpoint, block, optimizer, device=device)
    replay = take_center_focal_trust_region_step(block, image_size=source_image_size)
    replay_exact = replay == step and _state_sha256(block) == accepted_state_hash
    restore_checkpoint(accepted_checkpoint, block, optimizer, device=device)
    restore_exact = _state_sha256(block) == accepted_state_hash
    blockers: list[str] = []
    if step.evidence_improvement_fraction < 0.01:
        blockers.append("center_focal_moment_improvement_below_one_percent")
    if step.center_offset_norm > math.sqrt(0.08**2 + 0.08**2 + 0.15**2) + 1.0e-6:
        blockers.append("center_trust_bound_exceeded")
    if step.focal_relative_change > 0.100001:
        blockers.append("focal_trust_bound_exceeded")
    iou_change = independent_after["mean_iou"] - independent_before["mean_iou"]
    boundary_change = (
        independent_after["mean_normalized_boundary_error"]
        - independent_before["mean_normalized_boundary_error"]
    )
    if iou_change < -0.002:
        blockers.append("independent_train_iou_regressed_beyond_0_002")
    if boundary_change > 0:
        blockers.append("independent_train_boundary_regressed")
    if not replay_exact:
        blockers.append("center_focal_same_device_replay_mismatch")
    if not restore_exact:
        blockers.append("center_focal_checkpoint_restore_mismatch")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(accepted_checkpoint)
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_t01_center_focal_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t01_center_focal_mac_cpu_one_step_r01",
        "device": device,
        "dtype": "float32",
        "selected_training_source_frame_indices": selected_sources,
        "training_masks_read": len(selected_sources),
        "source_image_size": list(source_image_size),
        "render_resolution": render_resolution,
        "step": step.as_dict(),
        "accepted_center": accepted_center.tolist(),
        "accepted_focal": accepted_focal,
        "independent_before": independent_before,
        "independent_after": independent_after,
        "independent_iou_change": iou_change,
        "independent_boundary_change": boundary_change,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(accepted_checkpoint).hexdigest(),
        "checkpoint_restore_exact": restore_exact,
        "same_device_replay_exact": replay_exact,
        "input_hashes": {
            "solution": sha256_file(solution_path),
            "phase_checkpoint": sha256_file(phase_checkpoint_path),
            "canonical_mesh": sha256_file(canonical_mesh_path),
            "skinning_weights": sha256_file(skinning_weights_path),
            "joint_transforms": sha256_file(joint_transforms_path),
            "manifest": sha256_file(manifest_path),
        },
        "blockers": blockers,
        "optimizer_steps": 1,
        "replay_steps": 1,
        "scientific_attempt_marker_created": False,
        "development_metrics_read": 0,
        "held_out_images_read": 0,
        "sealed_test_accesses": 0,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "Boundary moments drive the step; a separate soft renderer supplies the train-only gate.",
            "Phase, axis, body shape, pose, residual twists, and micromotion are frozen.",
            "The clothing envelope is a fixed scaffold and cannot declare final geometry.",
        ],
    }
    return write_json(output_path, report)
