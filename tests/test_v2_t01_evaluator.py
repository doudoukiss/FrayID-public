from __future__ import annotations

import pytest
import torch

from frayid.v2.t01_evaluator import interpolate_turntable_angles


def test_turntable_phase_interpolation_is_exact_and_extrapolates_endpoints() -> None:
    sources = torch.tensor([5.0, 10.0, 20.0, 30.0])
    angles = torch.tensor([0.1, 0.2, 0.4, 0.6])
    query = torch.tensor([0.0, 5.0, 15.0, 30.0, 35.0])
    result = interpolate_turntable_angles(sources, angles, query)
    torch.testing.assert_close(
        result,
        torch.tensor([0.0, 0.1, 0.3, 0.6, 0.7]),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert bool(torch.all(result[1:].diff() > 0))


def test_turntable_phase_interpolation_rejects_nonmonotonic_training_input() -> None:
    with pytest.raises(ValueError):
        interpolate_turntable_angles(
            torch.tensor([0.0, 2.0, 1.0]),
            torch.tensor([0.0, 0.1, 0.2]),
            torch.tensor([0.5]),
        )
