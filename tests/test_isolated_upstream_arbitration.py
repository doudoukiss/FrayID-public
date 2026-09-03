from __future__ import annotations

import numpy as np
import pytest

from frayid.isolated_upstream_arbitration import arbitrate_isolated_upstream

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


def test_isolated_arbitration_matches_and_repeats() -> None:
    mesh, start, proposal = _two_triangles()
    first = arbitrate_isolated_upstream(mesh, start, proposal, dhat=0.05)
    second = arbitrate_isolated_upstream(mesh, start, proposal, dhat=0.05)
    assert first.status == "pass"
    assert first.singleton_failures == 0
    assert first.consensus_ratio_mismatches == 0
    assert np.array_equal(first.candidate_ids, second.candidate_ids)
    assert np.array_equal(first.selected_contributions, second.selected_contributions)
    assert np.array_equal(first.accepted_vertices, second.accepted_vertices)


def test_zero_motion_and_validation() -> None:
    mesh, start, _ = _two_triangles()
    result = arbitrate_isolated_upstream(mesh, start, start, dhat=0.05)
    assert result.status == "pass"
    assert result.retained_displacement_ratio == 1.0
    with pytest.raises(ValueError, match="dmin == 0"):
        arbitrate_isolated_upstream(mesh, start, start, dhat=0.05, dmin=1e-6)
