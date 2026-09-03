from __future__ import annotations

import numpy as np
import pytest

from frayid.full_isolated_upstream_filter import full_isolated_upstream_filter

ipctk = pytest.importorskip("ipctk")


def _two_triangles() -> tuple[object, np.ndarray, np.ndarray]:
    start = np.asfortranarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.2],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.2],
        ],
        dtype=np.float64,
    )
    faces = np.asfortranarray([[0, 1, 2], [3, 5, 4]], dtype=np.int32)
    edges = np.asfortranarray([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    proposal = start.copy(order="F")
    proposal[3:, 2] -= 0.4
    return ipctk.CollisionMesh(start, edges, faces), start, proposal


def test_full_isolated_filter_is_bitwise_deterministic() -> None:
    mesh, start, proposal = _two_triangles()
    first = full_isolated_upstream_filter(mesh, start, proposal, dhat=0.05)
    second = full_isolated_upstream_filter(mesh, start, proposal, dhat=0.05)
    assert first.status == "pass"
    assert first.singleton_failures == 0
    assert np.array_equal(first.candidate_ids, second.candidate_ids)
    assert np.array_equal(first.isolated_contributions, second.isolated_contributions)
    assert np.array_equal(first.accepted_vertices, second.accepted_vertices)


def test_zero_motion_and_validation() -> None:
    mesh, start, _ = _two_triangles()
    result = full_isolated_upstream_filter(mesh, start, start, dhat=0.05)
    assert result.status == "pass"
    assert result.retained_displacement_ratio == 1.0
    with pytest.raises(ValueError, match="dmin == 0"):
        full_isolated_upstream_filter(mesh, start, start, dhat=0.05, dmin=1e-6)
