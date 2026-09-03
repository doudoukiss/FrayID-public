from __future__ import annotations

import math

import torch

from frayid.camera import axis_angle_to_matrix
from frayid.v2.t02_geodesic import (
    internal_train_validation_slots,
    interpolate_bounded_geodesic_residual,
)


def test_internal_train_validation_split_is_frozen_at_twenty_percent() -> None:
    fit, validation = internal_train_validation_slots(144)
    assert fit.numel() == 115
    assert validation.numel() == 29
    assert validation.tolist()[:4] == [0, 5, 10, 15]
    assert set(fit.tolist()).isdisjoint(validation.tolist())


def test_bounded_geodesic_residual_recovers_smooth_public_trajectory() -> None:
    sources = torch.arange(20, dtype=torch.float32) * 5.0
    fit, validation = internal_train_validation_slots(20)
    dominant_angles = sources * (2.0 * math.pi / sources[-1])
    dominant_vectors = torch.stack(
        (torch.zeros_like(dominant_angles), -dominant_angles, torch.zeros_like(dominant_angles)),
        dim=-1,
    )
    dominant = axis_angle_to_matrix(dominant_vectors)
    residual_vectors = torch.stack(
        (
            0.12 * torch.sin(dominant_angles),
            torch.zeros_like(dominant_angles),
            0.08 * torch.cos(dominant_angles),
        ),
        dim=-1,
    )
    observed = axis_angle_to_matrix(residual_vectors) @ dominant
    translations = torch.stack(
        (
            0.02 * torch.sin(dominant_angles),
            0.01 * torch.cos(dominant_angles),
            torch.full_like(dominant_angles, 2.2),
        ),
        dim=-1,
    )
    prediction = interpolate_bounded_geodesic_residual(
        sources[fit],
        observed[fit],
        translations[fit],
        sources[validation],
        dominant[validation],
        torch.tensor([0.0, 0.0, 2.2]),
    )
    relative = prediction.rotations @ observed[validation].transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    error = torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))
    assert float(torch.median(error)) <= 2.0
    assert float(prediction.residual_rotation_degrees.max()) <= 30.0
    assert float(prediction.residual_translation_metres.max()) <= 0.08
