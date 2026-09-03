from __future__ import annotations

import torch

from frayid.continuous_time import (
    CubicControlTrajectory,
    cubic_bspline_basis,
    initialize_cubic_controls,
    interpolate_slots,
)


def test_cubic_basis_is_clamped_partition_of_unity() -> None:
    times = torch.linspace(0, 1, 101, dtype=torch.float64)
    basis = cubic_bspline_basis(times, 32)
    torch.testing.assert_close(basis.sum(dim=1), torch.ones_like(times))
    torch.testing.assert_close(
        basis[0], torch.nn.functional.one_hot(torch.tensor(0), 32).to(torch.float64)
    )
    torch.testing.assert_close(
        basis[-1], torch.nn.functional.one_hot(torch.tensor(31), 32).to(torch.float64)
    )


def test_cubic_controls_initialize_and_receive_finite_gradient() -> None:
    times = torch.linspace(0, 1, 40, dtype=torch.float64)
    values = torch.column_stack((torch.sin(2 * torch.pi * times), times.square()))
    controls = initialize_cubic_controls(times, values, control_count=12)
    trajectory = CubicControlTrajectory(controls)
    loss = (trajectory(times) - values).square().mean()
    loss.backward()
    assert trajectory.controls.grad is not None
    assert bool(torch.isfinite(trajectory.controls.grad).all())
    assert float(loss) < 1e-6


def test_irregular_slot_interpolation_uses_local_source_time() -> None:
    slots = torch.tensor([0.0, 0.2, 0.9, 1.0])
    values = torch.column_stack((slots, 2 * slots))
    query = torch.tensor([0.1, 0.55, 0.95])
    expected = torch.column_stack((query, 2 * query))
    torch.testing.assert_close(interpolate_slots(query, slots, values), expected)
