from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from frayid.barrier_sliding_carrier import (
    MAXIMUM_LINEAR_RESIDUAL,
    MINIMUM_ACCEPTED_STEP,
    barrier_sliding_carrier,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    source = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
    grid = 2.0**-30
    parent = np.rint((np.asarray(source.vertices) * 1.05) / grid) * grid
    return (
        parent,
        np.asarray(source.faces, dtype=np.int64),
        np.asarray(source.vertices, dtype=np.float64),
        np.asarray(source.faces, dtype=np.int64),
        grid,
    )


def test_barrier_sliding_is_deterministic_and_certified() -> None:
    pytest.importorskip("ipctk")
    parent, parent_faces, source, source_faces, grid = _fixture()
    first = barrier_sliding_carrier(
        parent,
        parent_faces,
        source,
        source_faces,
        pitch=0.25,
        parent_grid=grid,
    )
    second = barrier_sliding_carrier(
        parent,
        parent_faces,
        source,
        source_faces,
        pitch=0.25,
        parent_grid=grid,
    )
    assert first.status == "pass"
    assert first.certificate.status == "pass"
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)
    assert first.report()["steps"] == second.report()["steps"]
    assert all(step.accepted_step >= MINIMUM_ACCEPTED_STEP for step in first.steps)
    assert all(step.normalized_linear_residual <= MAXIMUM_LINEAR_RESIDUAL for step in first.steps)


def test_barrier_sliding_rejects_uncertified_parent_grid() -> None:
    pytest.importorskip("ipctk")
    parent, parent_faces, source, source_faces, grid = _fixture()
    parent[0, 0] += grid / 3.0
    result = barrier_sliding_carrier(
        parent,
        parent_faces,
        source,
        source_faces,
        pitch=0.25,
        parent_grid=grid,
    )
    assert result.status == "fail"
    assert result.blockers == ("p2_exact_refinement_certificate",)


def test_e15_runner_is_public_only_and_has_no_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_post_v1_e15_public_gate.py"
    ).read_text()
    for counter in (
        '"private_input_reads": 0',
        '"development_evidence_reads": 0',
        '"image_loads": 0',
        '"optimizer_steps": 0',
        '"modal_invocations": 0',
        '"sealed_test_accesses": 0',
    ):
        assert counter in source
    assert "barrier_sliding_carrier(" in source
    assert "source_exclusion_shrinkwrap(" not in source
    assert "worker.join(MAX_SECONDS)" in source
    assert "worker.terminate()" in source
    assert '"wall_time_ceiling_exceeded"' in source
