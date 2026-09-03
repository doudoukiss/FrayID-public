from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from frayid.camera import axis_angle_to_matrix
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.checkpoint import restore_checkpoint
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import TurntableSolution
from frayid.v2.t01_joint import (
    T01JointSchurBlock,
    _checkpoint_model_state,
)
from frayid.v2.t01_phase import BoundedPhaseAxisBlock
from frayid.v2.t01_silhouette import T01CenterFocalBlock
from frayid.v2.track_factors import PairwiseTrackletFactors, load_pairwise_tracklet_factors

CAPACITY_TRUST_RADII = (2.0, 1.0, 0.5, 0.25, 0.1, 0.05)


def _bounded_vectors(values: Tensor, maximum_norm: float) -> Tensor:
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    scale = maximum_norm * torch.tanh(norm) / norm.clamp_min(1.0e-12)
    result: Tensor = values * scale
    return result


class BoundedResidualCapacity(nn.Module):
    """Zero-mean low-rank residual camera and image-motion stress proxy."""

    temporal_basis: Tensor
    base_rotations: Tensor
    base_translations: Tensor
    intrinsics: Tensor
    first_slots: Tensor
    second_slots: Tensor

    def __init__(
        self,
        joint: T01JointSchurBlock,
        *,
        rank: int = 4,
        maximum_rotation_degrees: float = 0.25,
        maximum_translation_metres: float = 0.005,
        maximum_image_motion_pixels: float = 0.5,
    ) -> None:
        super().__init__()
        frame_count = joint.angles.numel()
        if rank < 1 or rank >= frame_count:
            raise ValueError("residual-capacity rank must be smaller than frame count")
        if (
            min(
                maximum_rotation_degrees,
                maximum_translation_metres,
                maximum_image_motion_pixels,
            )
            <= 0
        ):
            raise ValueError("residual-capacity bounds must be positive")
        time = (torch.arange(frame_count, dtype=torch.float32) + 0.5) / frame_count
        basis = torch.stack([torch.cos(math.pi * (mode + 1) * time) for mode in range(rank)], dim=1)
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis = basis / torch.max(torch.abs(basis), dim=0, keepdim=True).values.clamp_min(1.0e-12)
        base_rotations = joint.rotations().detach()
        center = joint.center.detach()
        base_translations = center[None] - torch.einsum("tij,j->ti", base_rotations, center)
        self.twist_coefficients = nn.Parameter(torch.zeros(rank, 6))
        self.image_motion_coefficients = nn.Parameter(torch.zeros(rank, 2))
        self.register_buffer("temporal_basis", basis)
        self.register_buffer("base_rotations", base_rotations)
        self.register_buffer("base_translations", base_translations)
        self.register_buffer("intrinsics", joint.intrinsics().detach())
        self.register_buffer("first_slots", joint.first_slots.detach().clone())
        self.register_buffer("second_slots", joint.second_slots.detach().clone())
        self.maximum_rotation_radians = math.radians(maximum_rotation_degrees)
        self.maximum_translation_metres = maximum_translation_metres
        self.maximum_image_motion_pixels = maximum_image_motion_pixels

    def residual_rotation_vectors(self) -> Tensor:
        raw = self.temporal_basis @ self.twist_coefficients[:, :3]
        return _bounded_vectors(raw, self.maximum_rotation_radians)

    def residual_translations(self) -> Tensor:
        raw = self.temporal_basis @ self.twist_coefficients[:, 3:]
        return _bounded_vectors(raw, self.maximum_translation_metres)

    def image_motion(self) -> Tensor:
        raw = self.temporal_basis @ self.image_motion_coefficients
        return _bounded_vectors(raw, self.maximum_image_motion_pixels)

    def extrinsics(self) -> tuple[Tensor, Tensor]:
        residual_rotations = axis_angle_to_matrix(self.residual_rotation_vectors())
        rotations = residual_rotations @ self.base_rotations
        translations = (
            torch.einsum("tij,tj->ti", residual_rotations, self.base_translations)
            + self.residual_translations()
        )
        return rotations, translations

    def fundamental_matrices(self) -> Tensor:
        rotations, translations = self.extrinsics()
        first_rotation = rotations[self.first_slots]
        second_rotation = rotations[self.second_slots]
        first_translation = translations[self.first_slots]
        second_translation = translations[self.second_slots]
        relative = second_rotation @ first_rotation.transpose(-1, -2)
        translation = second_translation - torch.einsum("eij,ej->ei", relative, first_translation)
        x, y, z = translation.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(-1, 3, 3)
        essential = skew @ relative
        inverse = torch.linalg.inv(self.intrinsics)
        fundamental = inverse.T[None] @ essential @ inverse[None]
        normalized: Tensor = fundamental / torch.linalg.vector_norm(
            fundamental, dim=(-2, -1), keepdim=True
        ).clamp_min(1.0e-12)
        return normalized

    def normalized_signed_errors(
        self,
        factors: PairwiseTrackletFactors,
        *,
        image_size: tuple[int, int],
    ) -> Tensor:
        matrices = self.fundamental_matrices()[factors.factor_edge_indices()]
        edge_indices = factors.factor_edge_indices()
        flow = self.image_motion()
        first_pixels = factors.first_pixels - flow[self.first_slots[edge_indices]]
        second_pixels = factors.second_pixels - flow[self.second_slots[edge_indices]]
        ones = torch.ones(
            (factors.factor_count, 1),
            dtype=first_pixels.dtype,
            device=first_pixels.device,
        )
        first = torch.cat((first_pixels, ones), dim=-1)
        second = torch.cat((second_pixels, ones), dim=-1)
        first_lines = torch.einsum("nij,nj->ni", matrices, first)
        second_lines = torch.einsum("nji,nj->ni", matrices, second)
        numerator = torch.einsum("ni,ni->n", second, first_lines)
        denominator = (
            first_lines[:, :2].square().sum(dim=-1) + second_lines[:, :2].square().sum(dim=-1)
        ).clamp_min(1.0e-12)
        diagonal = float((image_size[0] ** 2 + image_size[1] ** 2) ** 0.5)
        return numerator / torch.sqrt(denominator) / diagonal

    def signed_residuals(
        self,
        factors: PairwiseTrackletFactors,
        *,
        image_size: tuple[int, int],
        robust_delta_fraction_of_diagonal: float = 0.0025,
    ) -> Tensor:
        signed = self.normalized_signed_errors(factors, image_size=image_size)
        delta = signed.new_tensor(robust_delta_fraction_of_diagonal)
        robust_scale = torch.pow(1.0 + (signed / delta).square(), -0.25)
        weights = factors.observation_weights.to(dtype=signed.dtype, device=signed.device)
        return torch.sqrt(weights / weights.sum()) * robust_scale * signed

    def track_loss(
        self,
        factors: PairwiseTrackletFactors,
        *,
        image_size: tuple[int, int],
        robust_delta_fraction_of_diagonal: float = 0.0025,
    ) -> Tensor:
        signed = self.normalized_signed_errors(factors, image_size=image_size)
        delta = signed.new_tensor(robust_delta_fraction_of_diagonal)
        penalty = delta * (torch.sqrt(1.0 + (signed / delta).square()) - 1.0)
        weights = factors.observation_weights.to(dtype=penalty.dtype, device=penalty.device)
        return (weights * penalty).sum() / weights.sum()

    def regularization_residuals(self) -> Tensor:
        scale = math.sqrt(1.0e-3 / (self.twist_coefficients.numel() + 8))
        return scale * torch.cat(
            (
                torch.tanh(self.twist_coefficients).reshape(-1),
                torch.tanh(self.image_motion_coefficients).reshape(-1),
            )
        )

    def maximum_residuals(self) -> dict[str, float]:
        return {
            "rotation_degrees": math.degrees(
                float(
                    torch.linalg.vector_norm(self.residual_rotation_vectors(), dim=-1)
                    .max()
                    .detach()
                )
            ),
            "translation_metres": float(
                torch.linalg.vector_norm(self.residual_translations(), dim=-1).max().detach()
            ),
            "image_motion_pixels": float(
                torch.linalg.vector_norm(self.image_motion(), dim=-1).max().detach()
            ),
        }


@dataclass(frozen=True)
class CapacityFit:
    initial_loss: float
    accepted_loss: float
    trust_radius: float
    candidate_evaluations: int
    maximum_residuals: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_loss": self.initial_loss,
            "accepted_loss": self.accepted_loss,
            "trust_radius": self.trust_radius,
            "candidate_evaluations": self.candidate_evaluations,
            "maximum_residuals": self.maximum_residuals,
        }


def fit_residual_capacity_once(
    model: BoundedResidualCapacity,
    factors: PairwiseTrackletFactors,
    *,
    image_size: tuple[int, int],
) -> CapacityFit:
    """Fit one bounded Gauss-Newton capacity probe; candidates are not retries."""

    factors.validate()
    with torch.no_grad():
        model.twist_coefficients.zero_()
        model.image_motion_coefficients.zero_()
        initial_loss = float(model.track_loss(factors, image_size=image_size))
        base_residuals = model.signed_residuals(factors, image_size=image_size)
        scale = torch.linalg.vector_norm(base_residuals).clamp_min(1.0e-12)

        def residual_vector() -> Tensor:
            return torch.cat(
                (
                    model.signed_residuals(factors, image_size=image_size) / scale,
                    model.regularization_residuals(),
                )
            ).double()

        baseline = residual_vector()
        epsilon = 1.0e-3
        columns: list[Tensor] = []
        flat_parameters = (model.twist_coefficients, model.image_motion_coefficients)
        for parameter in flat_parameters:
            for index in range(parameter.numel()):
                model.twist_coefficients.zero_()
                model.image_motion_coefficients.zero_()
                parameter.view(-1)[index] = epsilon
                positive = residual_vector()
                parameter.view(-1)[index] = -epsilon
                negative = residual_vector()
                columns.append((positive - negative) / (2.0 * epsilon))
        model.twist_coefficients.zero_()
        model.image_motion_coefficients.zero_()
        jacobian = torch.stack(columns, dim=1)
        hessian = jacobian.T @ jacobian
        damping = 1.0e-4 * (torch.trace(hessian) / hessian.shape[0] + 1.0e-8)
        hessian = hessian + damping * torch.eye(hessian.shape[0], dtype=hessian.dtype)
        direction = -torch.linalg.solve(hessian, jacobian.T @ baseline)
        direction_maximum = float(torch.max(torch.abs(direction)))
        if not math.isfinite(direction_maximum) or direction_maximum <= 0:
            raise ValueError("residual-capacity direction is absent or nonfinite")
        direction = direction / direction_maximum
        twist_count = model.twist_coefficients.numel()
        best_loss = initial_loss
        best_radius = 0.0
        best_twist = model.twist_coefficients.detach().clone()
        best_motion = model.image_motion_coefficients.detach().clone()
        evaluations = 0
        for radius in CAPACITY_TRUST_RADII:
            model.twist_coefficients.copy_(
                direction[:twist_count].reshape_as(model.twist_coefficients).float() * radius
            )
            model.image_motion_coefficients.copy_(
                direction[twist_count:].reshape_as(model.image_motion_coefficients).float() * radius
            )
            candidate = float(model.track_loss(factors, image_size=image_size))
            evaluations += 1
            if math.isfinite(candidate) and candidate < best_loss:
                best_loss = candidate
                best_radius = radius
                best_twist = model.twist_coefficients.detach().clone()
                best_motion = model.image_motion_coefficients.detach().clone()
        model.twist_coefficients.copy_(best_twist)
        model.image_motion_coefficients.copy_(best_motion)
    if best_loss >= initial_loss:
        raise ValueError("residual-capacity probe found no improving bounded step")
    return CapacityFit(
        initial_loss=initial_loss,
        accepted_loss=best_loss,
        trust_radius=best_radius,
        candidate_evaluations=evaluations,
        maximum_residuals=model.maximum_residuals(),
    )


def inject_capacity_stress_factors(
    factors: PairwiseTrackletFactors,
    temporal_basis: Tensor,
    first_slots: Tensor,
    second_slots: Tensor,
    *,
    mode: str,
    sign: float = 1.0,
) -> PairwiseTrackletFactors:
    """Create deterministic valid-low-rank or invalid-factor train-only controls."""

    edge_indices = factors.factor_edge_indices()
    if first_slots.shape != (factors.edge_count,) or second_slots.shape != (factors.edge_count,):
        raise ValueError("capacity injection requires one compact slot pair per edge")
    if mode == "valid_low_rank_motion":
        if sign not in (-1.0, 1.0):
            raise ValueError("valid low-rank stress sign must be -1 or +1")
        flow = sign * temporal_basis[:, -1, None] * temporal_basis.new_tensor([0.50, -0.35])
        first = factors.first_pixels + flow[first_slots[edge_indices]]
        second = factors.second_pixels + flow[second_slots[edge_indices]]
    elif mode == "invalid_factor_corruption":
        index = torch.arange(factors.factor_count, device=factors.first_pixels.device)
        selected = index.remainder(5) == 0
        x_sign = torch.where(index.remainder(2) == 0, 1.0, -1.0)
        y_sign = torch.where(index.mul(7).remainder(4) < 2, 1.0, -1.0)
        offset = torch.stack((6.0 * x_sign, 6.0 * y_sign), dim=-1)
        first = factors.first_pixels
        second = factors.second_pixels + selected[:, None] * offset
    else:
        raise ValueError("unknown residual-capacity stress mode")
    result = PairwiseTrackletFactors(
        first_ordinals=factors.first_ordinals,
        second_ordinals=factors.second_ordinals,
        first_source_frame_indices=factors.first_source_frame_indices,
        second_source_frame_indices=factors.second_source_frame_indices,
        edge_offsets=factors.edge_offsets,
        first_pixels=first,
        second_pixels=second,
        observation_weights=factors.observation_weights,
        geometric_model_codes=factors.geometric_model_codes,
    )
    result.validate()
    return result


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def qualify_real_residual_capacity(
    solution_path: Path,
    factor_binding_path: Path,
    phase_checkpoint_path: Path,
    center_checkpoint_path: Path,
    joint_checkpoint_path: Path,
    output_path: Path,
    *,
    image_size: tuple[int, int] = (1120, 720),
    device: str = "cpu",
) -> Path:
    """Stress bounded nuisance capacity without accepting its fitted state."""

    paths = [
        solution_path,
        factor_binding_path,
        phase_checkpoint_path,
        center_checkpoint_path,
        joint_checkpoint_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if device != "cpu":
        raise ValueError("T01 capacity qualification is registered for deterministic Mac CPU")
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
    clean_model = BoundedResidualCapacity(joint).to(device)
    clean_loss = float(clean_model.track_loss(factors, image_size=image_size).detach())
    valid_candidates = [
        inject_capacity_stress_factors(
            factors,
            clean_model.temporal_basis,
            clean_model.first_slots,
            clean_model.second_slots,
            mode="valid_low_rank_motion",
            sign=sign,
        )
        for sign in (-1.0, 1.0)
    ]
    valid_candidate_losses = [
        float(BoundedResidualCapacity(joint).track_loss(candidate, image_size=image_size).detach())
        for candidate in valid_candidates
    ]
    valid_candidate_index = int(valid_candidate_losses[1] > valid_candidate_losses[0])
    valid_factors = valid_candidates[valid_candidate_index]
    valid_stress_sign = (-1.0, 1.0)[valid_candidate_index]
    invalid_factors = inject_capacity_stress_factors(
        factors,
        clean_model.temporal_basis,
        clean_model.first_slots,
        clean_model.second_slots,
        mode="invalid_factor_corruption",
    )
    valid_model = BoundedResidualCapacity(joint).to(device)
    invalid_model = BoundedResidualCapacity(joint).to(device)
    valid_fit = fit_residual_capacity_once(valid_model, valid_factors, image_size=image_size)
    invalid_fit = fit_residual_capacity_once(invalid_model, invalid_factors, image_size=image_size)

    def absorption(fit: CapacityFit) -> float:
        excess = fit.initial_loss - clean_loss
        if excess <= 0:
            return math.nan
        return (fit.initial_loss - fit.accepted_loss) / excess

    valid_absorption = absorption(valid_fit)
    invalid_absorption = absorption(invalid_fit)
    valid_replay_model = BoundedResidualCapacity(joint).to(device)
    invalid_replay_model = BoundedResidualCapacity(joint).to(device)
    valid_replay = fit_residual_capacity_once(
        valid_replay_model, valid_factors, image_size=image_size
    )
    invalid_replay = fit_residual_capacity_once(
        invalid_replay_model, invalid_factors, image_size=image_size
    )
    replay_exact = (
        valid_replay == valid_fit
        and invalid_replay == invalid_fit
        and _state_sha256(valid_replay_model) == _state_sha256(valid_model)
        and _state_sha256(invalid_replay_model) == _state_sha256(invalid_model)
    )
    blockers: list[str] = []
    if not math.isfinite(valid_absorption):
        blockers.append("valid_low_rank_stress_did_not_induce_positive_excess")
    elif valid_absorption < 0.50:
        blockers.append("bounded_capacity_did_not_recover_valid_low_rank_motion")
    if not math.isfinite(invalid_absorption):
        blockers.append("invalid_factor_stress_did_not_induce_positive_excess")
    elif invalid_absorption > 0.25:
        blockers.append("bounded_capacity_absorbed_too_much_invalid_correspondence_error")
    if not replay_exact:
        blockers.append("residual_capacity_same_device_replay_mismatch")
    for label, fit in (("valid", valid_fit), ("invalid", invalid_fit)):
        maxima = fit.maximum_residuals
        if maxima["rotation_degrees"] > 0.25001:
            blockers.append(f"{label}_residual_rotation_bound_exceeded")
        if maxima["translation_metres"] > 0.005001:
            blockers.append(f"{label}_residual_translation_bound_exceeded")
        if maxima["image_motion_pixels"] > 0.50001:
            blockers.append(f"{label}_image_motion_bound_exceeded")
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_t01_residual_capacity_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t01_residual_capacity_mac_cpu_r02",
        "device": device,
        "dtype": "float32_parameters_float64_normal_equations",
        "clean_track_loss": clean_loss,
        "valid_low_rank_motion": {
            "fit": valid_fit.as_dict(),
            "induced_excess_absorbed_fraction": valid_absorption,
            "symmetric_stress_candidate_losses": valid_candidate_losses,
            "selected_stress_sign": valid_stress_sign,
        },
        "invalid_factor_corruption": {
            "fit": invalid_fit.as_dict(),
            "induced_excess_absorbed_fraction": invalid_absorption,
            "corrupted_factor_fraction": 0.20,
            "corruption_pixels_per_axis": 6.0,
        },
        "same_device_replay_exact": replay_exact,
        "input_hashes": {
            "solution": sha256_file(solution_path),
            "factors": sha256_file(factor_binding_path),
            "phase_checkpoint": sha256_file(phase_checkpoint_path),
            "center_checkpoint": sha256_file(center_checkpoint_path),
            "joint_checkpoint": sha256_file(joint_checkpoint_path),
        },
        "blockers": blockers,
        "accepted_capacity_state": False,
        "diagnostic_fits": 2,
        "replay_fits": 2,
        "training_images_read": 0,
        "development_metrics_read": 0,
        "held_out_images_read": 0,
        "sealed_test_accesses": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "Residual capacity is a rejected-after-measurement diagnostic, not an accepted camera update.",
            "The image-motion proxy upper-bounds low-rank micromotion that is observable without material-track association.",
            "Physical micromotion remains frozen until scaffold or Q02 material association exists.",
        ],
    }
    return write_json(output_path, report)
