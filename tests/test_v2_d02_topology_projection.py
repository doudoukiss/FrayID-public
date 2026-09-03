from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from frayid.cli import app
from frayid.io import read_json, write_json
from frayid.v2.d02_topology_projection import (
    D02_EXPERIMENT_ID,
    D02_TRAIN_GATES,
    D02_TRAIN_PROJECTION_SCHEDULE,
    exact_face_constraint_audit,
    ipctk_has_self_intersections,
    run_d02_public_benchmark,
    topology_constrained_local_trust_projection,
    write_d02_public_benchmark,
    write_d02_train_projection_plan,
)


def _square() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def test_d02_local_trust_rejects_an_unsafe_face_and_returns_safe_connectivity() -> None:
    reference, faces = _square()
    raw = reference.copy()
    raw[2] = np.asarray([-0.6, 0.2, 0.0])
    assert exact_face_constraint_audit(reference, raw, faces)["status"] == "fail"
    candidate, scales, report = topology_constrained_local_trust_projection(reference, raw, faces)
    assert report["status"] == "pass"
    assert report["rejected_proposal_count"] > 0
    assert report["every_accepted_step_safe"] is True
    assert exact_face_constraint_audit(reference, candidate, faces)["status"] == "pass"
    assert np.any(scales < 1.0)


def test_d02_public_benchmark_passes_quality_and_constraint_gates() -> None:
    report = run_d02_public_benchmark()
    assert report["status"] == "pass"
    assert report["experiment_id"] == D02_EXPERIMENT_ID
    assert report["gates"]["injected_flip_and_collapse_detected"] is True
    assert report["gates"]["every_accepted_step_safe"] is True
    assert report["position_rmse_relative_improvement"] >= 0.10
    assert report["median_normal_improvement_degrees"] >= 2.0
    assert report["provenance"]["development_records_read"] == 0


def test_d02_public_benchmark_write_and_cli(tmp_path: Path) -> None:
    direct = tmp_path / "direct.json"
    write_d02_public_benchmark(direct)
    assert read_json(direct)["status"] == "pass"
    with pytest.raises(FileExistsError, match="immutable"):
        write_d02_public_benchmark(direct)
    cli_output = tmp_path / "cli.json"
    result = CliRunner().invoke(
        app,
        ["v2", "benchmark-d02-topology-projection", "--output", str(cli_output)],
    )
    assert result.exit_code == 0, result.stdout
    assert read_json(cli_output)["status"] == "pass"


def test_d02_real_projection_plan_binds_terminal_d01_without_reopening_it(
    tmp_path: Path,
) -> None:
    public = tmp_path / "public.json"
    terminal = tmp_path / "terminal.json"
    evidence = tmp_path / "evidence.json"
    write_json(public, {"status": "pass", "experiment_id": D02_EXPERIMENT_ID})
    write_json(terminal, {"decision": "terminal_failed_train_topology_precheck"})
    write_json(
        evidence,
        {
            "status": "train_only_evidence_bound",
            "training_records_read": 144,
            "development_records_read": 0,
        },
    )
    output = tmp_path / "plan.json"
    write_d02_train_projection_plan(
        public,
        terminal,
        evidence,
        output,
        source_revision="a" * 40,
    )
    plan = read_json(output)
    assert plan["projection_schedule"] == D02_TRAIN_PROJECTION_SCHEDULE
    assert plan["training_gates"] == D02_TRAIN_GATES
    assert plan["raw_proposal_role"] == "immutable_failed_d01_candidate_not_a_passing_dependency"
    assert plan["cleanup_operations_authorized"] == 0


def test_d02_ipctk_exact_predicate_detects_crossing_disconnected_triangles() -> None:
    pytest.importorskip("ipctk")
    vertices = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, -1.0],
            [0.0, 0.5, 1.0],
            [0.0, -0.5, 0.0],
        ]
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    assert ipctk_has_self_intersections(vertices, faces) is True
    separated = vertices.copy()
    separated[3:, 0] += 3.0
    assert ipctk_has_self_intersections(separated, faces) is False
