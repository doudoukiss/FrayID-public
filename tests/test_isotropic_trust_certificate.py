from __future__ import annotations

import numpy as np
import pytest

from frayid.isotropic_trust_certificate import isotropic_trust_path_certificate
from frayid.normalized_ti_oracle import normalized_ti_path_oracle

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
        ]
    )
    faces = np.asfortranarray([[0, 1, 2], [3, 5, 4]], dtype=np.int32)
    edges = np.asfortranarray([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    proposal = start.copy(order="F")
    proposal[3:, 2] -= 0.4
    return ipctk.CollisionMesh(start, edges, faces), start, proposal


def test_isotropic_filter_is_safe_and_deterministic() -> None:
    mesh, start, proposal = _two_triangles()
    first = isotropic_trust_path_certificate(mesh, start, proposal, dhat=0.05)
    second = isotropic_trust_path_certificate(mesh, start, proposal, dhat=0.05)
    assert first.status == second.status == "pass"
    assert 0.0 < first.retained_displacement_ratio < 1.0
    assert np.array_equal(first.accepted_vertices, second.accepted_vertices)
    assert np.array_equal(first.filtered_displacements, second.filtered_displacements)
    assert np.array_equal(first.trust_region_radii, second.trust_region_radii)
    oracle = normalized_ti_path_oracle(mesh, start, first.accepted_vertices)
    assert oracle.collision_free


def test_isotropic_filter_rejects_nonzero_dmin() -> None:
    mesh, start, _ = _two_triangles()
    with pytest.raises(ValueError, match="dmin == 0"):
        isotropic_trust_path_certificate(mesh, start, start, dhat=0.05, dmin=1e-6)
