from __future__ import annotations

import math

import torch

from frayid.v2.t01_silhouette import (
    T01CenterFocalBlock,
    take_center_focal_trust_region_step,
)
from frayid.v2.turntable import axis_angle_rotation


def _fixture() -> tuple[T01CenterFocalBlock, T01CenterFocalBlock]:
    torch.manual_seed(19)
    sample_count = 8
    vertex_count = 96
    angles = torch.linspace(0.0, 2.0 * math.pi, sample_count)
    rotations = axis_angle_rotation(torch.tensor([0.0, 1.0, 0.0]), angles)
    vertices = (torch.rand(sample_count, vertex_count, 3) - 0.5) * torch.tensor([0.8, 1.8, 0.5])
    base_center = torch.tensor([0.04, -0.02, 3.90])
    principal_point = torch.tensor([360.0, 560.0])
    truth = T01CenterFocalBlock(
        vertices,
        rotations,
        torch.zeros(sample_count, 4),
        base_center=torch.tensor([0.07, -0.04, 3.98]),
        base_focal=735.0,
        principal_point=principal_point,
    )
    target_bboxes = truth.soft_bboxes().detach()
    first = T01CenterFocalBlock(
        vertices,
        rotations,
        target_bboxes,
        base_center=base_center,
        base_focal=700.0,
        principal_point=principal_point,
    )
    second = T01CenterFocalBlock(
        vertices,
        rotations,
        target_bboxes,
        base_center=base_center,
        base_focal=700.0,
        principal_point=principal_point,
    )
    return first, second


def test_bounded_center_focal_step_is_deterministic_and_improves_moments() -> None:
    first, second = _fixture()
    first_step = take_center_focal_trust_region_step(first, image_size=(1120, 720))
    second_step = take_center_focal_trust_region_step(second, image_size=(1120, 720))
    assert first_step == second_step
    assert first_step.candidate_evaluations == 36
    assert first_step.evidence_improvement_fraction >= 0.01
    assert first_step.center_offset_norm <= math.sqrt(0.08**2 + 0.08**2 + 0.15**2)
    assert first_step.focal_relative_change <= 0.10
    for first_value, second_value in zip(
        first.state_dict().values(), second.state_dict().values(), strict=True
    ):
        assert torch.equal(first_value, second_value)


def test_center_focal_parameterization_enforces_componentwise_hard_bounds() -> None:
    first, _ = _fixture()
    with torch.no_grad():
        first.center_raw.fill_(100.0)
        first.focal_raw.fill_(-100.0)
    center_offset = first.center - first.base_center
    torch.testing.assert_close(
        center_offset,
        torch.tensor([0.08, 0.08, 0.15]),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert math.isclose(float(first.focal), 700.0 / 1.1, rel_tol=1.0e-6)
