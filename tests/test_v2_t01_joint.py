from __future__ import annotations

import math

import torch

from frayid.v2.schemas import TurntableSolution
from frayid.v2.t01_capacity import (
    BoundedResidualCapacity,
    fit_residual_capacity_once,
    inject_capacity_stress_factors,
)
from frayid.v2.t01_joint import T01JointSchurBlock, take_joint_schur_step
from frayid.v2.t01_phase import BoundedPhaseAxisBlock
from frayid.v2.t01_silhouette import T01CenterFocalBlock
from frayid.v2.track_factors import PairwiseTrackletFactors
from frayid.v2.turntable import axis_angle_rotation


def _fixture() -> tuple[T01JointSchurBlock, PairwiseTrackletFactors]:
    torch.manual_seed(23)
    frame_count = 12
    point_count = 32
    vertex_count = 96
    truth_angles = torch.linspace(0.0, 2.0 * math.pi, frame_count)
    base_angles = truth_angles + 0.05 * torch.sin(2.0 * truth_angles)
    axis = torch.tensor([0.0, 1.0, 0.0])
    base_center = torch.tensor([0.08, -0.03, 4.0])
    truth_center = torch.tensor([0.10, -0.01, 4.05])
    principal_point = torch.tensor([360.0, 560.0])
    intrinsics = torch.tensor([[700.0, 0.0, 360.0], [0.0, 700.0, 560.0], [0.0, 0.0, 1.0]])
    points = truth_center + (torch.rand(point_count, 3) - 0.5) * torch.tensor([0.8, 1.6, 0.5])
    truth_rotations = axis_angle_rotation(axis, truth_angles)
    trajectories = (
        torch.einsum("tij,pj->tpi", truth_rotations, points - truth_center) + truth_center
    )
    pixels = torch.stack(
        (
            720.0 * trajectories[..., 0] / trajectories[..., 2] + principal_point[0],
            720.0 * trajectories[..., 1] / trajectories[..., 2] + principal_point[1],
        ),
        dim=-1,
    )
    edge_count = frame_count - 1
    factors = PairwiseTrackletFactors(
        first_ordinals=torch.arange(edge_count, dtype=torch.long),
        second_ordinals=torch.arange(1, frame_count, dtype=torch.long),
        first_source_frame_indices=torch.arange(edge_count, dtype=torch.long) * 5,
        second_source_frame_indices=torch.arange(1, frame_count, dtype=torch.long) * 5,
        edge_offsets=torch.arange(0, frame_count * point_count, point_count, dtype=torch.long),
        first_pixels=pixels[:-1].reshape(-1, 2).detach(),
        second_pixels=pixels[1:].reshape(-1, 2).detach(),
        observation_weights=torch.ones(edge_count * point_count),
        geometric_model_codes=torch.zeros(edge_count, dtype=torch.long),
    )
    solution = TurntableSolution(
        status="qualification_candidate",
        shared_intrinsics=intrinsics.tolist(),
        axis=axis.tolist(),
        center=base_center.tolist(),
        angles_radians=base_angles.tolist(),
        residual_twists=[[0.0] * 6 for _ in range(frame_count)],
        micromotion_basis=[],
        micromotion_codes=[[] for _ in range(frame_count)],
        source_frame_indices=(torch.arange(frame_count) * 5).tolist(),
        gauge_policy={"fixture": "public"},
        uncertainty={},
        source_provenance={"role": "public_fixture"},
    )
    phase = BoundedPhaseAxisBlock(solution, factors)
    posed = (torch.rand(frame_count, vertex_count, 3) - 0.5) * torch.tensor([0.8, 1.8, 0.5])
    truth_silhouette = T01CenterFocalBlock(
        posed,
        truth_rotations,
        torch.zeros(frame_count, 4),
        base_center=truth_center,
        base_focal=720.0,
        principal_point=principal_point,
    )
    silhouette = T01CenterFocalBlock(
        posed,
        axis_angle_rotation(axis, base_angles),
        truth_silhouette.soft_bboxes().detach(),
        base_center=base_center,
        base_focal=700.0,
        principal_point=principal_point,
    )
    return T01JointSchurBlock(phase, silhouette), factors


def test_joint_schur_step_improves_combined_evidence_without_block_regression() -> None:
    first, factors = _fixture()
    second, second_factors = _fixture()
    first_step = take_joint_schur_step(first, factors, image_size=(1120, 720))
    second_step = take_joint_schur_step(second, second_factors, image_size=(1120, 720))
    assert first_step == second_step
    assert first_step.combined_improvement_fraction >= 0.0001
    assert first_step.track_relative_change <= 0.002
    assert first_step.moment_relative_change <= 0.002
    assert first_step.damped_schur_minimum_eigenvalue > 0
    assert first.maximum_increment_relative_change() <= 0.20
    assert first.axis_tilt_degrees() <= 5.0
    for first_value, second_value in zip(
        first.state_dict().values(), second.state_dict().values(), strict=True
    ):
        assert torch.equal(first_value, second_value)


def test_bounded_capacity_recovers_valid_motion_but_not_factor_corruption() -> None:
    joint, factors = _fixture()
    take_joint_schur_step(joint, factors, image_size=(1120, 720))
    clean = BoundedResidualCapacity(joint)
    clean_loss = float(clean.track_loss(factors, image_size=(1120, 720)).detach())
    valid_candidates = [
        inject_capacity_stress_factors(
            factors,
            clean.temporal_basis,
            clean.first_slots,
            clean.second_slots,
            mode="valid_low_rank_motion",
            sign=sign,
        )
        for sign in (-1.0, 1.0)
    ]
    valid_factors = max(
        valid_candidates,
        key=lambda candidate: float(
            BoundedResidualCapacity(joint).track_loss(candidate, image_size=(1120, 720)).detach()
        ),
    )
    invalid_factors = inject_capacity_stress_factors(
        factors,
        clean.temporal_basis,
        clean.first_slots,
        clean.second_slots,
        mode="invalid_factor_corruption",
    )
    valid = fit_residual_capacity_once(
        BoundedResidualCapacity(joint), valid_factors, image_size=(1120, 720)
    )
    invalid = fit_residual_capacity_once(
        BoundedResidualCapacity(joint), invalid_factors, image_size=(1120, 720)
    )
    valid_absorption = (valid.initial_loss - valid.accepted_loss) / (
        valid.initial_loss - clean_loss
    )
    invalid_absorption = (invalid.initial_loss - invalid.accepted_loss) / (
        invalid.initial_loss - clean_loss
    )
    assert valid_absorption >= 0.50
    assert invalid_absorption <= 0.25
    assert valid.maximum_residuals["rotation_degrees"] <= 0.25
    assert valid.maximum_residuals["translation_metres"] <= 0.005
    assert valid.maximum_residuals["image_motion_pixels"] <= 0.5
