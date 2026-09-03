from __future__ import annotations

import numpy as np
import pytest

from frayid.candidate_contribution_certificate import certify_candidate_contributions

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


def test_candidate_contributions_match_and_repeat() -> None:
    mesh, start, proposal = _two_triangles()
    result = certify_candidate_contributions(mesh, start, proposal, dhat=0.05)
    assert result.status == "pass"
    assert result.candidate_count == 15
    assert result.maximum_absolute_contribution_difference <= 1e-12
    assert result.maximum_absolute_vertex_minimum_difference <= 1e-12
    assert result.skipped_decision_mismatches == 0
    assert result.constrained_decision_mismatches == 0
    assert result.bitwise_batched_repeat


def test_zero_motion_and_validation() -> None:
    mesh, start, _ = _two_triangles()
    result = certify_candidate_contributions(mesh, start, start, dhat=0.05)
    assert result.status == "pass"
    assert result.retained_displacement_ratio == 1.0
    with pytest.raises(ValueError, match="dmin == 0"):
        certify_candidate_contributions(mesh, start, start, dhat=0.05, dmin=1e-6)
