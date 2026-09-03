from __future__ import annotations

import numpy as np
import pytest

from frayid.batched_planar_dat import batched_planar_dat_path
from frayid.planar_dat_certificate import planar_dat_path_certificate

ipctk = pytest.importorskip("ipctk")


def _two_triangles(scale: float = 1.0) -> tuple[object, np.ndarray, np.ndarray]:
    start = np.asfortranarray(
        np.array(
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
        * scale
    )
    faces = np.asfortranarray([[0, 1, 2], [3, 5, 4]], dtype=np.int32)
    edges = np.asfortranarray([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    proposal = start.copy(order="F")
    proposal[3:, 2] -= 0.4 * scale
    return ipctk.CollisionMesh(start, edges, faces), start, proposal


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_batched_filter_matches_upstream(scale: float) -> None:
    mesh, start, proposal = _two_triangles(scale)
    upstream = planar_dat_path_certificate(
        mesh, start, proposal, dhat=0.05 * scale, verify_full_path=False
    )
    batched = batched_planar_dat_path(mesh, start, proposal, dhat=0.05 * scale)
    assert batched.status == "pass"
    assert batched.edge_edge_count == 9
    assert batched.face_vertex_count == 6
    assert np.allclose(
        batched.filtered_displacements,
        upstream.filtered_displacements,
        rtol=0.0,
        atol=1e-12,
    )
    assert batched.restricted_vertex_count == upstream.restricted_vertex_count


def test_batched_filter_is_bitwise_deterministic() -> None:
    mesh, start, proposal = _two_triangles()
    first = batched_planar_dat_path(mesh, start, proposal, dhat=0.05)
    second = batched_planar_dat_path(mesh, start, proposal, dhat=0.05)
    assert np.array_equal(first.candidate_ids, second.candidate_ids)
    assert np.array_equal(first.candidate_kinds, second.candidate_kinds)
    assert np.array_equal(first.truncation_ratios, second.truncation_ratios)
    assert np.array_equal(first.filtered_displacements, second.filtered_displacements)
    assert np.array_equal(first.accepted_vertices, second.accepted_vertices)


def test_zero_motion_and_input_validation() -> None:
    mesh, start, _ = _two_triangles()
    result = batched_planar_dat_path(mesh, start, start, dhat=0.05)
    assert result.status == "pass"
    assert result.retained_displacement_ratio == 1.0
    assert result.restricted_vertex_count == 0
    with pytest.raises(ValueError, match="dmin == 0"):
        batched_planar_dat_path(mesh, start, start, dhat=0.05, dmin=1e-6)
