from __future__ import annotations

import numpy as np
import pytest

from frayid.normalized_ti_oracle import (
    normalize_linear_trajectory,
    normalized_ti_path_oracle,
)

ipctk = pytest.importorskip("ipctk")


def _two_triangles(scale: float) -> tuple[object, np.ndarray, np.ndarray]:
    start = np.asfortranarray(
        [
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 0.2 * scale],
            [scale, 0.0, 0.2 * scale],
            [0.0, scale, 0.2 * scale],
        ]
    )
    faces = np.asfortranarray([[0, 1, 2], [3, 5, 4]], dtype=np.int32)
    edges = np.asfortranarray([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    end = start.copy(order="F")
    end[3:, 2] -= 0.18 * scale
    return ipctk.CollisionMesh(start, edges, faces), start, end


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_normalized_oracle_accepts_same_safe_gap_at_all_scales(scale: float) -> None:
    mesh, start, end = _two_triangles(scale)
    result = normalized_ti_path_oracle(mesh, start, end)
    assert result.status == "pass"
    assert result.collision_free


def test_normalized_oracle_rejects_crossing() -> None:
    mesh, start, end = _two_triangles(1e-6)
    end[3:, 2] -= 0.04e-6
    result = normalized_ti_path_oracle(mesh, start, end)
    assert result.status == "pass"
    assert not result.collision_free


def test_normalization_repeats_bitwise_and_rejects_zero_extent() -> None:
    _, start, end = _two_triangles(1.0)
    first = normalize_linear_trajectory(start, end)
    second = normalize_linear_trajectory(start, end)
    assert first.scale == second.scale
    assert np.array_equal(first.center, second.center)
    assert np.array_equal(first.start, second.start)
    assert np.array_equal(first.end, second.end)
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        normalize_linear_trajectory(np.zeros((2, 3)), np.zeros((2, 3)))
