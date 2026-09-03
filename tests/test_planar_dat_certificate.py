from __future__ import annotations

import numpy as np
import pytest

from frayid.planar_dat_certificate import planar_dat_path_certificate

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


def test_planar_dat_truncates_crossing_and_passes_full_oracle() -> None:
    mesh, start, proposal = _two_triangles()
    result = planar_dat_path_certificate(mesh, start, proposal, dhat=0.05)
    assert result.status == "pass"
    assert result.full_oracle_safe
    assert result.candidate_count > 0
    assert result.restricted_vertex_count == 3
    assert 0.0 < result.retained_displacement_ratio < 1.0
    assert np.all(result.accepted_vertices[3:, 2] > result.accepted_vertices[:3, 2])


def test_planar_dat_certificate_is_bitwise_deterministic() -> None:
    mesh, start, proposal = _two_triangles()
    first = planar_dat_path_certificate(mesh, start, proposal, dhat=0.05)
    second = planar_dat_path_certificate(mesh, start, proposal, dhat=0.05)
    assert first.candidate_keys == second.candidate_keys
    assert np.array_equal(first.trust_region_centers, second.trust_region_centers)
    assert np.array_equal(first.trust_region_radii, second.trust_region_radii)
    assert np.array_equal(first.filtered_displacements, second.filtered_displacements)
    assert np.array_equal(first.accepted_vertices, second.accepted_vertices)
    assert first.report()["accepted_vertices_sha256"] == second.report()["accepted_vertices_sha256"]


def test_zero_motion_and_input_validation() -> None:
    mesh, start, _ = _two_triangles()
    result = planar_dat_path_certificate(mesh, start, start, dhat=0.05)
    assert result.status == "pass"
    assert result.retained_displacement_ratio == 1.0
    assert result.restricted_vertex_count == 0
    with pytest.raises(ValueError, match="dmin == 0"):
        planar_dat_path_certificate(mesh, start, start, dhat=0.05, dmin=1e-6)
