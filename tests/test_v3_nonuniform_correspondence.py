from __future__ import annotations

from itertools import pairwise

import numpy as np

from frayid.v3.nonuniform_correspondence import ordered_dtw, qualify_ordered_correspondence


def _cycle(phases: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.sin(phases),
            np.cos(phases),
            np.sin(2.0 * phases),
            np.cos(2.0 * phases),
        ),
        axis=1,
    )


def test_nonuniform_repeated_cycle_passes_frozen_controls() -> None:
    first_phase = np.linspace(0.0, 2.0 * np.pi, 12)
    second_phase = np.asarray(
        [0.0, 0.25, 0.65, 1.1, 1.6, 2.1, 2.7, 3.3, 3.9, 4.5, 5.0, 5.5, 5.9, 2.0 * np.pi]
    )
    result = qualify_ordered_correspondence(_cycle(first_phase), _cycle(second_phase))
    assert result["blockers"] == []
    assert result["matched_views"] >= 10
    assert result["margin"] >= 0.30


def test_reversed_cycle_fails_control_separation() -> None:
    phases = np.linspace(0.0, 2.0 * np.pi, 12)
    first = _cycle(phases)
    result = qualify_ordered_correspondence(first, first[::-1])
    assert "control_cost_margin_below_0_30" in result["blockers"]


def test_ordered_dtw_path_never_moves_backward() -> None:
    first = _cycle(np.linspace(0.0, 2.0 * np.pi, 12))
    second = _cycle(np.linspace(0.0, 2.0 * np.pi, 15))
    cost, path, _ = ordered_dtw(first, second)
    assert cost >= 0.0
    assert path[0] == (0, 0)
    assert path[-1] == (11, 14)
    assert all(right[0] >= left[0] and right[1] >= left[1] for left, right in pairwise(path))
