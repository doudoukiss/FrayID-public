from __future__ import annotations

import numpy as np
import pytest

from frayid.collision_partition import (
    collision_candidate_summary,
    conservative_collision_partition,
)

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
    edges = np.asfortranarray(
        [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]],
        dtype=np.int32,
    )
    proposal = start.copy(order="F")
    proposal[3:, 2] -= 0.4
    return ipctk.CollisionMesh(start, edges, faces), start, proposal


def test_empty_near_set_is_bounded_instead_of_accepting_crossing() -> None:
    mesh, start, proposal = _two_triangles()
    result = conservative_collision_partition(
        mesh,
        start,
        proposal,
        dhat=0.05,
    )
    assert result.status == "pass"
    assert result.near_candidate_count == 0
    assert result.far_fraction == pytest.approx(0.0625)
    assert result.certified_fraction == pytest.approx(0.05)
    assert result.full_oracle_safe

    candidates = ipctk.Candidates()
    candidates.build(mesh, start, 0.025, ipctk.SweepAndPrune())
    unsafe_library_fraction = candidates.compute_cfl_stepsize(
        mesh,
        start,
        proposal,
        0.05,
        0.0,
        ipctk.SweepAndPrune(),
        ipctk.TightInclusionCCD(),
    )
    assert unsafe_library_fraction == 1.0
    assert not ipctk.is_step_collision_free(
        mesh,
        start,
        proposal,
        0.0,
        ipctk.SweepAndPrune(),
        ipctk.TightInclusionCCD(),
    )


def test_near_collision_is_tight_inclusion_bounded_and_deterministic() -> None:
    mesh, start, proposal = _two_triangles()
    first = conservative_collision_partition(mesh, start, proposal, dhat=0.5)
    second = conservative_collision_partition(mesh, start, proposal, dhat=0.5)
    assert first.status == second.status == "pass"
    assert first.near_candidate_count > 0
    assert first.certified_fraction == second.certified_fraction
    assert first.near_fraction == second.near_fraction
    assert first.far_fraction == second.far_fraction
    assert first.near_candidate_keys == second.near_candidate_keys
    assert first.full_oracle_safe and second.full_oracle_safe
    first_summary = collision_candidate_summary(mesh, start, proposal, dhat=0.5)
    second_summary = collision_candidate_summary(mesh, start, proposal, dhat=0.5)
    assert first_summary.near_candidate_keys == second_summary.near_candidate_keys
    assert first_summary.near_candidate_count == second_summary.near_candidate_count
    assert first_summary.full_swept_candidate_count == second_summary.full_swept_candidate_count
    assert (
        first_summary.near_to_full_swept_candidate_ratio
        == second_summary.near_to_full_swept_candidate_ratio
    )


def test_zero_motion_is_finite_and_rejects_nonzero_dmin() -> None:
    mesh, start, _ = _two_triangles()
    result = conservative_collision_partition(mesh, start, start, dhat=0.05)
    assert result.status == "pass"
    assert result.maximum_vertex_displacement == 0.0
    assert result.far_fraction == 1.0
    assert result.certified_fraction == pytest.approx(0.8)
    with pytest.raises(ValueError, match="dmin == 0"):
        conservative_collision_partition(mesh, start, start, dhat=0.05, dmin=1e-6)
