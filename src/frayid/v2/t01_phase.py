from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.checkpoint import capture_checkpoint, restore_checkpoint
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import TurntableSolution
from frayid.v2.track_factors import (
    PairwiseTrackletFactors,
    load_pairwise_tracklet_factors,
    pairwise_sampson_loss,
)
from frayid.v2.turntable import (
    axis_angle_rotation,
    turntable_edge_slots,
    turntable_fundamental_matrices,
)

PHASE_TRUST_RADII = (2.0, 1.0, 0.5, 0.25, 0.1)
AXIS_TRUST_RADII = (1.0, 0.5, 0.25, 0.1, 0.0)


class BoundedPhaseAxisBlock(nn.Module):
    """Low-rank monotonic phase and bounded axis tilt with center/focal frozen."""

    base_axis: Tensor
    base_center: Tensor
    base_intrinsics: Tensor
    base_angles: Tensor
    base_increments: Tensor
    phase_basis: Tensor
    first_slots: Tensor
    second_slots: Tensor

    def __init__(
        self,
        solution: TurntableSolution,
        factors: PairwiseTrackletFactors,
        *,
        mode_count: int = 8,
        maximum_log_increment_adjustment: float = 0.15,
        maximum_axis_tilt_degrees: float = 5.0,
    ) -> None:
        super().__init__()
        if mode_count < 1 or mode_count >= len(solution.angles_radians):
            raise ValueError("phase mode count must be smaller than the frame count")
        if maximum_log_increment_adjustment <= 0 or maximum_axis_tilt_degrees <= 0:
            raise ValueError("phase and axis bounds must be positive")
        base_angles = torch.tensor(solution.angles_radians, dtype=torch.float32)
        increments = base_angles.diff()
        if bool(torch.any(increments <= 0)):
            raise ValueError("phase-axis block requires strictly increasing base angles")
        interval_count = len(increments)
        time = (torch.arange(interval_count, dtype=torch.float32) + 0.5) / interval_count
        basis = torch.stack(
            [torch.cos(math.pi * (mode + 1) * time) for mode in range(mode_count)],
            dim=1,
        )
        basis = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True).clamp_min(1.0e-12)
        first_slots, second_slots = turntable_edge_slots(
            solution.source_frame_indices,
            factors.first_source_frame_indices,
            factors.second_source_frame_indices,
        )
        self.phase_coefficients = nn.Parameter(torch.zeros(mode_count))
        self.axis_tilt = nn.Parameter(torch.zeros(2))
        self.register_buffer("base_axis", torch.tensor(solution.axis, dtype=torch.float32))
        self.register_buffer("base_center", torch.tensor(solution.center, dtype=torch.float32))
        self.register_buffer(
            "base_intrinsics", torch.tensor(solution.shared_intrinsics, dtype=torch.float32)
        )
        self.register_buffer("base_angles", base_angles)
        self.register_buffer("base_increments", increments)
        self.register_buffer("phase_basis", basis)
        self.register_buffer("first_slots", first_slots)
        self.register_buffer("second_slots", second_slots)
        self.maximum_log_increment_adjustment = maximum_log_increment_adjustment
        self.maximum_axis_tilt_radians = math.radians(maximum_axis_tilt_degrees)

    @property
    def angles(self) -> Tensor:
        log_adjustment = self.maximum_log_increment_adjustment * torch.tanh(
            self.phase_basis @ self.phase_coefficients
        )
        raw_increments = self.base_increments * torch.exp(log_adjustment)
        increments = raw_increments * (self.base_increments.sum() / raw_increments.sum())
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

    def fundamental_matrices(self) -> Tensor:
        return turntable_fundamental_matrices(
            axis_angle_rotation(self.axis, self.angles),
            self.base_center,
            self.base_intrinsics,
            self.first_slots,
            self.second_slots,
        )

    def evidence_loss(
        self,
        factors: PairwiseTrackletFactors,
        *,
        image_size: tuple[int, int],
    ) -> Tensor:
        return pairwise_sampson_loss(
            self.fundamental_matrices(),
            factors,
            image_size=image_size,
        )

    def maximum_increment_relative_change(self) -> float:
        relative = self.angles.diff() / self.base_increments - 1.0
        return float(torch.max(torch.abs(relative)).detach())

    def axis_tilt_degrees(self) -> float:
        cosine = torch.clamp(torch.abs(self.axis @ self.base_axis), 0.0, 1.0)
        return math.degrees(math.acos(float(cosine.detach())))


@dataclass(frozen=True)
class TrustRegionStep:
    initial_loss: float
    accepted_loss: float
    phase_radius: float
    axis_radius: float
    candidate_evaluations: int
    evidence_improvement_fraction: float
    maximum_increment_relative_change: float
    axis_tilt_degrees: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "initial_loss": self.initial_loss,
            "accepted_loss": self.accepted_loss,
            "phase_radius": self.phase_radius,
            "axis_radius": self.axis_radius,
            "candidate_evaluations": self.candidate_evaluations,
            "evidence_improvement_fraction": self.evidence_improvement_fraction,
            "maximum_increment_relative_change": self.maximum_increment_relative_change,
            "axis_tilt_degrees": self.axis_tilt_degrees,
        }


def _normalized_descent(value: Tensor) -> Tensor:
    gradient = value.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise ValueError("phase trust-region gradient is absent or nonfinite")
    norm = torch.linalg.vector_norm(gradient)
    if float(norm) <= 0:
        return torch.zeros_like(gradient)
    result: Tensor = -gradient / norm
    return result


def take_phase_axis_trust_region_step(
    model: BoundedPhaseAxisBlock,
    factors: PairwiseTrackletFactors,
    *,
    image_size: tuple[int, int],
) -> TrustRegionStep:
    """Take one deterministic bounded step; candidate evaluations are not optimizer retries."""

    model.zero_grad(set_to_none=True)
    initial_tensor = model.evidence_loss(factors, image_size=image_size)
    initial_tensor.backward()  # type: ignore[no-untyped-call]
    phase_direction = _normalized_descent(model.phase_coefficients).detach()
    axis_direction = _normalized_descent(model.axis_tilt).detach()
    initial_loss = float(initial_tensor.detach())
    best_loss = initial_loss
    best_phase = model.phase_coefficients.detach().clone()
    best_axis = model.axis_tilt.detach().clone()
    best_radii = (0.0, 0.0)
    evaluations = 0
    with torch.no_grad():
        for phase_radius in PHASE_TRUST_RADII:
            for axis_radius in AXIS_TRUST_RADII:
                model.phase_coefficients.copy_(phase_direction * phase_radius)
                model.axis_tilt.copy_(axis_direction * axis_radius)
                candidate = float(model.evidence_loss(factors, image_size=image_size))
                evaluations += 1
                if math.isfinite(candidate) and candidate < best_loss:
                    best_loss = candidate
                    best_phase = model.phase_coefficients.detach().clone()
                    best_axis = model.axis_tilt.detach().clone()
                    best_radii = (phase_radius, axis_radius)
        model.phase_coefficients.copy_(best_phase)
        model.axis_tilt.copy_(best_axis)
    if best_loss >= initial_loss:
        raise ValueError("bounded phase-axis trust region found no improving step")
    return TrustRegionStep(
        initial_loss=initial_loss,
        accepted_loss=best_loss,
        phase_radius=best_radii[0],
        axis_radius=best_radii[1],
        candidate_evaluations=evaluations,
        evidence_improvement_fraction=(initial_loss - best_loss) / initial_loss,
        maximum_increment_relative_change=model.maximum_increment_relative_change(),
        axis_tilt_degrees=model.axis_tilt_degrees(),
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


def qualify_real_phase_axis_step(
    solution_path: Path,
    factor_binding_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    image_size: tuple[int, int],
    device: str = "cpu",
) -> Path:
    """Run one train-factor engineering step with exact same-device replay."""

    reject_sealed_capability([solution_path, factor_binding_path, output_path, checkpoint_path])
    if checkpoint_path.exists():
        raise FileExistsError(f"T01 phase checkpoint already exists: {checkpoint_path}")
    if device != "cpu":
        raise ValueError("T01 phase qualification is registered for deterministic Mac CPU")
    solution = TurntableSolution.model_validate(read_json(solution_path))
    factors = load_pairwise_tracklet_factors(factor_binding_path, device=device)
    model = BoundedPhaseAxisBlock(solution, factors).to(device)
    checkpoint_optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    initial_checkpoint = capture_checkpoint(
        model,
        checkpoint_optimizer,
        step=0,
        topology_connectivity_sha256=None,
    )
    step = take_phase_axis_trust_region_step(model, factors, image_size=image_size)
    accepted_state_hash = _state_sha256(model)
    accepted_angles = model.angles.detach().cpu()
    accepted_axis = model.axis.detach().cpu()
    accepted_checkpoint = capture_checkpoint(
        model,
        checkpoint_optimizer,
        step=1,
        topology_connectivity_sha256=None,
    )
    restore_checkpoint(
        initial_checkpoint,
        model,
        checkpoint_optimizer,
        device=device,
    )
    replay = take_phase_axis_trust_region_step(model, factors, image_size=image_size)
    replay_exact = (
        replay == step
        and _state_sha256(model) == accepted_state_hash
        and torch.equal(model.angles.detach().cpu(), accepted_angles)
        and torch.equal(model.axis.detach().cpu(), accepted_axis)
    )
    restore_checkpoint(
        accepted_checkpoint,
        model,
        checkpoint_optimizer,
        device=device,
    )
    restore_exact = (
        _state_sha256(model) == accepted_state_hash
        and torch.equal(model.angles.detach().cpu(), accepted_angles)
        and torch.equal(model.axis.detach().cpu(), accepted_axis)
    )
    blockers: list[str] = []
    if step.evidence_improvement_fraction < 0.01:
        blockers.append("phase_axis_evidence_improvement_below_one_percent")
    if step.maximum_increment_relative_change > 0.20:
        blockers.append("phase_increment_trust_bound_exceeded")
    if step.axis_tilt_degrees > 5.0001:
        blockers.append("axis_tilt_trust_bound_exceeded")
    if not replay_exact:
        blockers.append("phase_axis_same_device_replay_mismatch")
    if not restore_exact:
        blockers.append("phase_axis_checkpoint_restore_mismatch")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(accepted_checkpoint)
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_t01_phase_axis_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t01_phase_axis_mac_cpu_one_step_r01",
        "device": device,
        "dtype": "float32",
        "solution_path": str(solution_path),
        "solution_sha256": sha256_file(solution_path),
        "factor_binding_path": str(factor_binding_path),
        "factor_binding_sha256": sha256_file(factor_binding_path),
        "factor_count": factors.factor_count,
        "edge_count": factors.edge_count,
        "step": step.as_dict(),
        "accepted_axis": accepted_axis.tolist(),
        "accepted_angle_span_radians": float(accepted_angles[-1] - accepted_angles[0]),
        "base_angle_span_radians": float(model.base_angles[-1] - model.base_angles[0]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(accepted_checkpoint).hexdigest(),
        "checkpoint_restore_exact": restore_exact,
        "same_device_replay_exact": replay_exact,
        "blockers": blockers,
        "optimizer_steps": 1,
        "replay_steps": 1,
        "scientific_attempt_marker_created": False,
        "training_images_read": 0,
        "development_metrics_read": 0,
        "held_out_images_read": 0,
        "sealed_test_accesses": 0,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "The accepted step is engineering qualification, not the T01 scientific attempt.",
            "Center, focal, residual twists, micromotion, RGB, and photometric capacity are frozen.",
            "All candidate evaluations belong to one deterministic bounded trust-region step.",
        ],
    }
    return write_json(output_path, report)
