from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from frayid.camera import make_intrinsics
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.checkpoint import V2_CHECKPOINT_SCHEMA, capture_checkpoint, restore_checkpoint
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import TurntableSolution
from frayid.v2.t01_phase import BoundedPhaseAxisBlock
from frayid.v2.t01_silhouette import T01CenterFocalBlock
from frayid.v2.track_factors import (
    PairwiseTrackletFactors,
    load_pairwise_tracklet_factors,
    pairwise_sampson_loss,
)
from frayid.v2.turntable import axis_angle_rotation, turntable_fundamental_matrices

JOINT_TRUST_RADII = (1.0, 0.5, 0.25, 0.1, 0.05)


class T01JointSchurBlock(nn.Module):
    """Small joint correction around accepted phase and silhouette blocks."""

    base_axis: Tensor
    base_center: Tensor
    base_intrinsics: Tensor
    base_angles: Tensor
    base_increments: Tensor
    phase_basis: Tensor
    first_slots: Tensor
    second_slots: Tensor
    accepted_phase_coefficients: Tensor
    accepted_axis_tilt: Tensor
    accepted_center_raw: Tensor
    accepted_focal_raw: Tensor
    center_bounds: Tensor
    principal_point: Tensor
    posed_body_vertices: Tensor
    target_bboxes: Tensor
    silhouette_phase_slots: Tensor

    def __init__(
        self,
        phase: BoundedPhaseAxisBlock,
        silhouette: T01CenterFocalBlock,
        *,
        phase_raw_radius: float = 0.25,
        axis_raw_radius: float = 0.10,
        center_raw_radius: float = 0.15,
        focal_raw_radius: float = 0.15,
    ) -> None:
        super().__init__()
        if min(phase_raw_radius, axis_raw_radius, center_raw_radius, focal_raw_radius) <= 0:
            raise ValueError("joint raw-parameter radii must be positive")
        self.motion_delta = nn.Parameter(torch.zeros(phase.phase_coefficients.numel() + 2))
        self.camera_delta = nn.Parameter(torch.zeros(4))
        self.register_buffer("base_axis", phase.base_axis.detach().clone())
        self.register_buffer("base_center", silhouette.base_center.detach().clone())
        self.register_buffer("base_intrinsics", phase.base_intrinsics.detach().clone())
        self.register_buffer("base_angles", phase.base_angles.detach().clone())
        self.register_buffer("base_increments", phase.base_increments.detach().clone())
        self.register_buffer("phase_basis", phase.phase_basis.detach().clone())
        self.register_buffer("first_slots", phase.first_slots.detach().clone())
        self.register_buffer("second_slots", phase.second_slots.detach().clone())
        self.register_buffer(
            "accepted_phase_coefficients", phase.phase_coefficients.detach().clone()
        )
        self.register_buffer("accepted_axis_tilt", phase.axis_tilt.detach().clone())
        self.register_buffer("accepted_center_raw", silhouette.center_raw.detach().clone())
        self.register_buffer("accepted_focal_raw", silhouette.focal_raw.detach().clone())
        self.register_buffer("center_bounds", silhouette.center_bounds.detach().clone())
        self.register_buffer("principal_point", silhouette.principal_point.detach().clone())
        self.register_buffer("posed_body_vertices", silhouette.posed_body_vertices.detach().clone())
        self.register_buffer("target_bboxes", silhouette.target_bboxes.detach().clone())
        with torch.no_grad():
            accepted_rotations = axis_angle_rotation(phase.axis, phase.angles)
            rotation_distance = (
                (silhouette.rotations[:, None] - accepted_rotations[None, :])
                .square()
                .sum(dim=(-2, -1))
            )
            silhouette_phase_slots = torch.argmin(rotation_distance, dim=1)
            if bool(torch.any(silhouette_phase_slots[1:] <= silhouette_phase_slots[:-1])):
                raise ValueError("silhouette checkpoint rotations do not map to unique phase slots")
        self.register_buffer("silhouette_phase_slots", silhouette_phase_slots)
        self.maximum_log_increment_adjustment = phase.maximum_log_increment_adjustment
        self.maximum_axis_tilt_radians = phase.maximum_axis_tilt_radians
        self.base_focal = silhouette.base_focal
        self.maximum_log_focal_change = silhouette.maximum_log_focal_change
        self.soft_extrema_temperature_pixels = silhouette.soft_extrema_temperature_pixels
        self.phase_raw_radius = phase_raw_radius
        self.axis_raw_radius = axis_raw_radius
        self.center_raw_radius = center_raw_radius
        self.focal_raw_radius = focal_raw_radius

    @property
    def phase_coefficients(self) -> Tensor:
        count = self.accepted_phase_coefficients.numel()
        return self.accepted_phase_coefficients + self.phase_raw_radius * torch.tanh(
            self.motion_delta[:count]
        )

    @property
    def axis_tilt(self) -> Tensor:
        count = self.accepted_phase_coefficients.numel()
        return self.accepted_axis_tilt + self.axis_raw_radius * torch.tanh(
            self.motion_delta[count:]
        )

    @property
    def center_raw(self) -> Tensor:
        return self.accepted_center_raw + self.center_raw_radius * torch.tanh(self.camera_delta[:3])

    @property
    def focal_raw(self) -> Tensor:
        return self.accepted_focal_raw + self.focal_raw_radius * torch.tanh(self.camera_delta[3])

    @property
    def angles(self) -> Tensor:
        log_adjustment = self.maximum_log_increment_adjustment * torch.tanh(
            self.phase_basis @ self.phase_coefficients
        )
        raw = self.base_increments * torch.exp(log_adjustment)
        increments = raw * (self.base_increments.sum() / raw.sum())
        return torch.cat((increments.new_zeros(1), torch.cumsum(increments, dim=0)))

    @property
    def axis(self) -> Tensor:
        tangent_scale = math.tan(self.maximum_axis_tilt_radians)
        addition = torch.stack(
            (
                tangent_scale * torch.tanh(self.axis_tilt[0]),
                self.axis_tilt.new_zeros(()),
                tangent_scale * torch.tanh(self.axis_tilt[1]),
            )
        )
        return F.normalize(self.base_axis + addition, dim=0, eps=1.0e-12)

    @property
    def center(self) -> Tensor:
        return self.base_center + self.center_bounds * torch.tanh(self.center_raw)

    @property
    def focal(self) -> Tensor:
        return self.focal_raw.new_tensor(self.base_focal) * torch.exp(
            self.maximum_log_focal_change * torch.tanh(self.focal_raw)
        )

    def rotations(self) -> Tensor:
        return axis_angle_rotation(self.axis, self.angles)

    def intrinsics(self) -> Tensor:
        return make_intrinsics(self.focal, self.principal_point)

    def fundamental_matrices(self) -> Tensor:
        return turntable_fundamental_matrices(
            self.rotations(),
            self.center,
            self.intrinsics(),
            self.first_slots,
            self.second_slots,
        )

    def soft_bboxes(self) -> Tensor:
        rotations = self.rotations()[self.silhouette_phase_slots]
        camera = (
            torch.einsum("sij,svj->svi", rotations, self.posed_body_vertices)
            + self.center[None, None, :]
        )
        depth = camera[..., 2].clamp_min(0.1)
        x = self.focal * camera[..., 0] / depth + self.principal_point[0]
        y = self.focal * camera[..., 1] / depth + self.principal_point[1]
        temperature = self.soft_extrema_temperature_pixels
        return torch.stack(
            (
                -temperature * torch.logsumexp(-x / temperature, dim=1),
                -temperature * torch.logsumexp(-y / temperature, dim=1),
                temperature * torch.logsumexp(x / temperature, dim=1),
                temperature * torch.logsumexp(y / temperature, dim=1),
            ),
            dim=-1,
        )

    def track_loss(
        self, factors: PairwiseTrackletFactors, *, image_size: tuple[int, int]
    ) -> Tensor:
        return pairwise_sampson_loss(self.fundamental_matrices(), factors, image_size=image_size)

    def moment_residuals(self, *, image_size: tuple[int, int]) -> Tensor:
        height, width = image_size
        diagonal = float((height * height + width * width) ** 0.5)
        predicted = self.soft_bboxes()
        target = self.target_bboxes
        edges = (predicted - target) / diagonal
        centers = (
            0.5 * (predicted[:, :2] + predicted[:, 2:] - target[:, :2] - target[:, 2:]) / diagonal
        )
        sizes = torch.log((predicted[:, 2:] - predicted[:, :2]).clamp_min(1.0)) - torch.log(
            (target[:, 2:] - target[:, :2]).clamp_min(1.0)
        )
        return torch.cat(
            (edges.reshape(-1), math.sqrt(0.5) * centers.reshape(-1), 0.5 * sizes.reshape(-1))
        )

    def moment_loss(self, *, image_size: tuple[int, int]) -> Tensor:
        height, width = image_size
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

    def track_residuals(
        self,
        factors: PairwiseTrackletFactors,
        *,
        image_size: tuple[int, int],
        robust_delta_fraction_of_diagonal: float = 0.0025,
    ) -> Tensor:
        matrices = self.fundamental_matrices()[factors.factor_edge_indices()]
        ones = torch.ones(
            (factors.factor_count, 1),
            dtype=factors.first_pixels.dtype,
            device=factors.first_pixels.device,
        )
        first = torch.cat((factors.first_pixels, ones), dim=-1)
        second = torch.cat((factors.second_pixels, ones), dim=-1)
        first_lines = torch.einsum("nij,nj->ni", matrices, first)
        second_lines = torch.einsum("nji,nj->ni", matrices, second)
        numerator = torch.einsum("ni,ni->n", second, first_lines)
        denominator = (
            first_lines[:, :2].square().sum(dim=-1) + second_lines[:, :2].square().sum(dim=-1)
        ).clamp_min(1.0e-12)
        diagonal = float((image_size[0] ** 2 + image_size[1] ** 2) ** 0.5)
        signed = numerator / torch.sqrt(denominator) / diagonal
        delta = signed.new_tensor(robust_delta_fraction_of_diagonal)
        robust_scale = torch.pow(1.0 + (signed / delta).square(), -0.25)
        weights = factors.observation_weights.to(dtype=signed.dtype, device=signed.device)
        normalized_weights = torch.sqrt(weights / weights.sum())
        return normalized_weights * robust_scale * signed

    def forward(
        self,
        factors: PairwiseTrackletFactors,
        image_size: tuple[int, int],
        track_scale: Tensor,
        moment_scale: Tensor,
    ) -> Tensor:
        return torch.cat(
            (
                self.track_residuals(factors, image_size=image_size) / track_scale,
                self.moment_residuals(image_size=image_size) / moment_scale,
            )
        )

    def maximum_increment_relative_change(self) -> float:
        relative = self.angles.diff() / self.base_increments - 1.0
        return float(torch.max(torch.abs(relative)).detach())

    def axis_tilt_degrees(self) -> float:
        cosine = torch.clamp(torch.abs(self.axis @ self.base_axis), 0.0, 1.0)
        return math.degrees(math.acos(float(cosine.detach())))


@dataclass(frozen=True)
class JointEvidence:
    track_loss: float
    moment_loss: float

    def as_dict(self) -> dict[str, float]:
        return {"track_loss": self.track_loss, "moment_loss": self.moment_loss}


@dataclass(frozen=True)
class JointSchurStep:
    initial: JointEvidence
    accepted: JointEvidence
    trust_radius: float
    candidate_evaluations: int
    combined_improvement_fraction: float
    track_relative_change: float
    moment_relative_change: float
    damped_schur_minimum_eigenvalue: float
    damped_schur_condition_number: float
    raw_direction_maximum_absolute_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial": self.initial.as_dict(),
            "accepted": self.accepted.as_dict(),
            "trust_radius": self.trust_radius,
            "candidate_evaluations": self.candidate_evaluations,
            "combined_improvement_fraction": self.combined_improvement_fraction,
            "track_relative_change": self.track_relative_change,
            "moment_relative_change": self.moment_relative_change,
            "damped_schur_minimum_eigenvalue": self.damped_schur_minimum_eigenvalue,
            "damped_schur_condition_number": self.damped_schur_condition_number,
            "raw_direction_maximum_absolute_value": self.raw_direction_maximum_absolute_value,
        }


def _evidence(
    model: T01JointSchurBlock,
    factors: PairwiseTrackletFactors,
    *,
    image_size: tuple[int, int],
) -> JointEvidence:
    return JointEvidence(
        track_loss=float(model.track_loss(factors, image_size=image_size).detach()),
        moment_loss=float(model.moment_loss(image_size=image_size).detach()),
    )


def take_joint_schur_step(
    model: T01JointSchurBlock,
    factors: PairwiseTrackletFactors,
    *,
    image_size: tuple[int, int],
    maximum_single_evidence_regression: float = 0.002,
) -> JointSchurStep:
    """Take one damped block-Schur step with separate evidence preservation."""

    factors.validate()
    initial = _evidence(model, factors, image_size=image_size)
    if min(initial.track_loss, initial.moment_loss) <= 0:
        raise ValueError("joint evidence losses must be positive")
    with torch.no_grad():
        track_scale = torch.linalg.vector_norm(
            model.track_residuals(factors, image_size=image_size)
        ).clamp_min(1.0e-12)
        moment_scale = torch.linalg.vector_norm(
            model.moment_residuals(image_size=image_size)
        ).clamp_min(1.0e-12)
    with torch.no_grad():
        model.motion_delta.zero_()
        model.camera_delta.zero_()
        residual = model(factors, image_size, track_scale, moment_scale).detach().double()
        epsilon = 1.0e-3
        motion_columns: list[Tensor] = []
        for column in range(model.motion_delta.numel()):
            model.motion_delta.zero_()
            model.motion_delta[column] = epsilon
            positive = model(factors, image_size, track_scale, moment_scale).detach().double()
            model.motion_delta[column] = -epsilon
            negative = model(factors, image_size, track_scale, moment_scale).detach().double()
            motion_columns.append((positive - negative) / (2.0 * epsilon))
        model.motion_delta.zero_()
        camera_columns: list[Tensor] = []
        for column in range(model.camera_delta.numel()):
            model.camera_delta.zero_()
            model.camera_delta[column] = epsilon
            positive = model(factors, image_size, track_scale, moment_scale).detach().double()
            model.camera_delta[column] = -epsilon
            negative = model(factors, image_size, track_scale, moment_scale).detach().double()
            camera_columns.append((positive - negative) / (2.0 * epsilon))
        model.camera_delta.zero_()
    jm = torch.stack(motion_columns, dim=1)
    jc = torch.stack(camera_columns, dim=1)
    h_mm = jm.T @ jm
    h_mc = jm.T @ jc
    h_cc = jc.T @ jc
    g_m = jm.T @ residual
    g_c = jc.T @ residual
    motion_damping = 1.0e-4 * (torch.trace(h_mm) / h_mm.shape[0] + 1.0e-8)
    camera_damping = 1.0e-4 * (torch.trace(h_cc) / h_cc.shape[0] + 1.0e-8)
    h_mm = h_mm + motion_damping * torch.eye(h_mm.shape[0], dtype=h_mm.dtype)
    h_cc = h_cc + camera_damping * torch.eye(h_cc.shape[0], dtype=h_cc.dtype)
    camera_inverse_h_cm = torch.linalg.solve(h_cc, h_mc.T)
    schur = h_mm - h_mc @ camera_inverse_h_cm
    reduced_gradient = g_m - h_mc @ torch.linalg.solve(h_cc, g_c)
    motion_direction = -torch.linalg.solve(schur, reduced_gradient)
    camera_direction = -torch.linalg.solve(h_cc, g_c + h_mc.T @ motion_direction)
    direction_maximum = max(
        float(torch.max(torch.abs(motion_direction))),
        float(torch.max(torch.abs(camera_direction))),
    )
    if not math.isfinite(direction_maximum) or direction_maximum <= 0:
        raise ValueError("joint Schur direction is absent or nonfinite")
    motion_direction = motion_direction / direction_maximum
    camera_direction = camera_direction / direction_maximum
    eigenvalues = torch.linalg.eigvalsh(0.5 * (schur + schur.T))
    minimum_eigenvalue = float(eigenvalues[0])
    condition = float(eigenvalues[-1] / eigenvalues[0].clamp_min(1.0e-18))
    best: JointEvidence | None = None
    best_radius = 0.0
    best_combined = 2.0
    evaluations = 0
    with torch.no_grad():
        for radius in JOINT_TRUST_RADII:
            model.motion_delta.copy_(motion_direction.float() * radius)
            model.camera_delta.copy_(camera_direction.float() * radius)
            candidate = _evidence(model, factors, image_size=image_size)
            evaluations += 1
            track_ratio = candidate.track_loss / initial.track_loss
            moment_ratio = candidate.moment_loss / initial.moment_loss
            combined = track_ratio + moment_ratio
            preserves = (
                track_ratio <= 1.0 + maximum_single_evidence_regression
                and moment_ratio <= 1.0 + maximum_single_evidence_regression
            )
            if preserves and combined < best_combined:
                best = candidate
                best_radius = radius
                best_combined = combined
        if best is None or best_combined >= 2.0:
            model.motion_delta.zero_()
            model.camera_delta.zero_()
            raise ValueError("joint Schur step found no Pareto-safe improving radius")
        model.motion_delta.copy_(motion_direction.float() * best_radius)
        model.camera_delta.copy_(camera_direction.float() * best_radius)
    return JointSchurStep(
        initial=initial,
        accepted=best,
        trust_radius=best_radius,
        candidate_evaluations=evaluations,
        combined_improvement_fraction=(2.0 - best_combined) / 2.0,
        track_relative_change=best.track_loss / initial.track_loss - 1.0,
        moment_relative_change=best.moment_loss / initial.moment_loss - 1.0,
        damped_schur_minimum_eigenvalue=minimum_eigenvalue,
        damped_schur_condition_number=condition,
        raw_direction_maximum_absolute_value=direction_maximum,
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


def _checkpoint_model_state(path: Path, *, device: str) -> dict[str, Tensor]:
    try:
        payload = torch.load(
            io.BytesIO(path.read_bytes()),
            map_location=torch.device(device),
            weights_only=False,
        )
    except Exception as error:
        raise ValueError("V2 checkpoint schema is invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != V2_CHECKPOINT_SCHEMA:
        raise ValueError("V2 checkpoint schema is invalid")
    state = payload.get("model_state")
    if not isinstance(state, dict) or not all(
        isinstance(value, Tensor) for value in state.values()
    ):
        raise ValueError("V2 checkpoint model state is invalid")
    return state


def qualify_real_joint_schur_step(
    solution_path: Path,
    factor_binding_path: Path,
    phase_checkpoint_path: Path,
    center_checkpoint_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    image_size: tuple[int, int] = (1120, 720),
    device: str = "cpu",
) -> Path:
    """Run one joint Mac-CPU compatibility step without reopening images."""

    paths = [
        solution_path,
        factor_binding_path,
        phase_checkpoint_path,
        center_checkpoint_path,
        output_path,
        checkpoint_path,
    ]
    reject_sealed_capability(paths)
    if device != "cpu":
        raise ValueError("T01 joint qualification is registered for deterministic Mac CPU")
    if checkpoint_path.exists():
        raise FileExistsError(f"T01 joint checkpoint already exists: {checkpoint_path}")
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
    model = T01JointSchurBlock(phase, silhouette).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    initial_checkpoint = capture_checkpoint(
        model, optimizer, step=0, topology_connectivity_sha256=None
    )
    step = take_joint_schur_step(model, factors, image_size=image_size)
    accepted_hash = _state_sha256(model)
    accepted_checkpoint = capture_checkpoint(
        model, optimizer, step=1, topology_connectivity_sha256=None
    )
    restore_checkpoint(initial_checkpoint, model, optimizer, device=device)
    replay = take_joint_schur_step(model, factors, image_size=image_size)
    replay_exact = replay == step and _state_sha256(model) == accepted_hash
    restore_checkpoint(accepted_checkpoint, model, optimizer, device=device)
    restore_exact = _state_sha256(model) == accepted_hash
    blockers: list[str] = []
    if step.combined_improvement_fraction < 0.0001:
        blockers.append("joint_combined_improvement_below_0_01_percent")
    if step.track_relative_change > 0.002:
        blockers.append("joint_track_evidence_regressed_beyond_0_2_percent")
    if step.moment_relative_change > 0.002:
        blockers.append("joint_silhouette_evidence_regressed_beyond_0_2_percent")
    if model.maximum_increment_relative_change() > 0.20:
        blockers.append("joint_phase_increment_bound_exceeded")
    if model.axis_tilt_degrees() > 5.0001:
        blockers.append("joint_axis_tilt_bound_exceeded")
    if not replay_exact:
        blockers.append("joint_same_device_replay_mismatch")
    if not restore_exact:
        blockers.append("joint_checkpoint_restore_mismatch")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(accepted_checkpoint)
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_t01_joint_schur_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t01_joint_schur_mac_cpu_one_step_r01",
        "device": device,
        "dtype": "float32_parameters_float64_normal_equations",
        "step": step.as_dict(),
        "accepted_axis": model.axis.detach().cpu().tolist(),
        "accepted_center": model.center.detach().cpu().tolist(),
        "accepted_focal": float(model.focal.detach()),
        "accepted_angle_span_radians": float((model.angles[-1] - model.angles[0]).detach()),
        "maximum_increment_relative_change": model.maximum_increment_relative_change(),
        "axis_tilt_degrees": model.axis_tilt_degrees(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(accepted_checkpoint).hexdigest(),
        "checkpoint_restore_exact": restore_exact,
        "same_device_replay_exact": replay_exact,
        "input_hashes": {
            "solution": sha256_file(solution_path),
            "factors": sha256_file(factor_binding_path),
            "phase_checkpoint": sha256_file(phase_checkpoint_path),
            "center_checkpoint": sha256_file(center_checkpoint_path),
        },
        "blockers": blockers,
        "optimizer_steps": 1,
        "replay_steps": 1,
        "candidate_evaluations_per_step": len(JOINT_TRUST_RADII),
        "training_images_read": 0,
        "development_metrics_read": 0,
        "held_out_images_read": 0,
        "sealed_test_accesses": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "The block-Schur update jointly linearizes ten motion and four camera variables.",
            "Track and silhouette evidence have separate Pareto preservation guards.",
            "Checkpoint buffers reuse prior train-only summaries; no image is reopened.",
            "Residual twists, micromotion, geometry, RGB, and appearance remain frozen.",
        ],
    }
    return write_json(output_path, report)
