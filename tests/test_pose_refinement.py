from __future__ import annotations

import math

import torch

from frayid.camera import axis_angle_to_matrix
from frayid.pose_refinement import (
    bounded_vector,
    geman_mcclure,
    rotation_geodesic_acceleration,
    rotation_geodesic_radians,
    translation_acceleration,
)


def test_bounded_vector_uses_vector_norm() -> None:
    raw = torch.tensor([[100.0, 100.0, 100.0], [0.0, 0.0, 0.0]])
    bounded = bounded_vector(raw, 0.2)
    assert float(torch.linalg.vector_norm(bounded[0])) <= 0.2 + 1e-6
    torch.testing.assert_close(bounded[1], torch.zeros(3))


def test_geodesic_acceleration_is_zero_for_constant_angular_velocity() -> None:
    angles = torch.linspace(0.0, 0.4, 5)
    axis_angles = torch.stack((torch.zeros_like(angles), angles, torch.zeros_like(angles)), -1)
    rotations = axis_angle_to_matrix(axis_angles)[:, None]
    assert float(rotation_geodesic_acceleration(rotations)) < 1e-10
    translations = torch.stack((angles, 2.0 * angles, -angles), -1)
    assert float(translation_acceleration(translations)) < 1e-12


def test_robust_rotation_translation_recovery_with_outlier() -> None:
    torch.manual_seed(7)
    points = torch.tensor(
        [
            [-0.4, -0.3, 2.0],
            [0.3, -0.2, 2.2],
            [-0.2, 0.5, 2.4],
            [0.5, 0.4, 2.1],
            [0.0, 0.0, 2.7],
            [0.2, -0.5, 2.5],
            [-0.5, 0.2, 2.3],
            [0.4, 0.1, 2.6],
        ],
        dtype=torch.float64,
    )
    target_axis_angle = torch.tensor([0.08, -0.12, 0.05], dtype=torch.float64)
    target_translation = torch.tensor([0.06, -0.04, 0.03], dtype=torch.float64)
    target = points @ axis_angle_to_matrix(target_axis_angle).T + target_translation
    target = target.clone()
    target[0] += torch.tensor([2.0, -1.5, 1.0], dtype=torch.float64)
    estimate_rotation = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    estimate_translation = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = torch.optim.Adam((estimate_rotation, estimate_translation), lr=0.03)
    confidence = torch.ones(len(points), dtype=torch.float64)
    confidence[0] = 0.05
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        prediction = points @ axis_angle_to_matrix(estimate_rotation).T + estimate_translation
        residual = torch.linalg.vector_norm(prediction - target, dim=-1)
        loss = (geman_mcclure(residual, 0.08) * confidence).sum() / confidence.sum()
        loss.backward()
        optimizer.step()
    angular_error = rotation_geodesic_radians(
        axis_angle_to_matrix(estimate_rotation), axis_angle_to_matrix(target_axis_angle)
    )
    assert math.degrees(float(angular_error)) < 0.25
    assert float(torch.linalg.vector_norm(estimate_translation - target_translation)) < 0.005
