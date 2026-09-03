from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


def cubic_bspline_basis(times: Tensor, control_count: int) -> Tensor:
    """Evaluate a clamped open-uniform cubic B-spline basis on normalized time."""
    if times.ndim != 1 or not torch.isfinite(times).all():
        raise ValueError("Spline times must be one finite dimension")
    if control_count < 4:
        raise ValueError("A cubic spline requires at least four controls")
    if bool((times < 0).any()) or bool((times > 1).any()):
        raise ValueError("Spline times must lie in [0, 1]")

    degree = 3
    interior_count = control_count - degree - 1
    interior = (
        torch.arange(1, interior_count + 1, device=times.device, dtype=times.dtype)
        / (interior_count + 1)
        if interior_count
        else times.new_empty((0,))
    )
    knots = torch.cat(
        (
            times.new_zeros(degree + 1),
            interior,
            times.new_ones(degree + 1),
        )
    )
    basis = ((times[:, None] >= knots[:-1]) & (times[:, None] < knots[1:])).to(times.dtype)
    basis[times == 1, control_count - 1] = 1
    for order in range(1, degree + 1):
        next_basis = times.new_zeros((len(times), len(knots) - order - 1))
        for index in range(next_basis.shape[1]):
            left_width = knots[index + order] - knots[index]
            right_width = knots[index + order + 1] - knots[index + 1]
            if float(left_width) > 0:
                next_basis[:, index] += (times - knots[index]) / left_width * basis[:, index]
            if float(right_width) > 0:
                next_basis[:, index] += (
                    (knots[index + order + 1] - times) / right_width * basis[:, index + 1]
                )
        basis = next_basis
    return basis[:, :control_count]


class CubicControlTrajectory(nn.Module):
    """A differentiable fixed-knot trajectory whose only state is its controls."""

    def __init__(self, controls: Tensor) -> None:
        super().__init__()
        if controls.ndim != 2 or controls.shape[0] < 4:
            raise ValueError("Spline controls must have shape [K>=4, D]")
        self.controls = nn.Parameter(controls.clone())

    def forward(self, normalized_times: Tensor) -> Tensor:
        return cubic_bspline_basis(normalized_times, self.controls.shape[0]) @ self.controls


def initialize_cubic_controls(
    normalized_times: Tensor,
    slot_values: Tensor,
    *,
    control_count: int,
) -> Tensor:
    """Least-squares initialize fixed spline controls from training-slot values."""
    if slot_values.ndim != 2 or slot_values.shape[0] != len(normalized_times):
        raise ValueError("Slot values must have shape [T, D] aligned to spline times")
    basis = cubic_bspline_basis(normalized_times, control_count)
    return cast(Tensor, torch.linalg.lstsq(basis, slot_values).solution)


def interpolate_slots(
    query_times: Tensor,
    slot_times: Tensor,
    slot_values: Tensor,
) -> Tensor:
    """Piecewise-linear incumbent interpolation over irregular local source time."""
    if query_times.ndim != 1 or slot_times.ndim != 1:
        raise ValueError("Slot interpolation times must be one-dimensional")
    if slot_values.ndim != 2 or slot_values.shape[0] != len(slot_times):
        raise ValueError("Slot values must align with slot times")
    if len(slot_times) < 2 or not bool(torch.all(slot_times[1:] > slot_times[:-1])):
        raise ValueError("Slot times must be strictly increasing")
    upper = torch.searchsorted(slot_times, query_times, right=True).clamp(1, len(slot_times) - 1)
    lower = upper - 1
    denominator = (slot_times[upper] - slot_times[lower]).clamp_min(1e-12)
    fraction = ((query_times - slot_times[lower]) / denominator).clamp(0, 1)
    return slot_values[lower] + fraction[:, None] * (slot_values[upper] - slot_values[lower])
