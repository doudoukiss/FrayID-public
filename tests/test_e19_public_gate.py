from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_post_v1_e19_public_gate.py"
STATUS = PROJECT_ROOT / "configs/evaluation/post_v1_e19_coarse_orientation_preserving_r01.yaml"


def test_e19_status_freezes_distinct_exact_determinant_mechanism() -> None:
    if not STATUS.exists():
        pytest.skip("private status record is intentionally absent from public snapshot")
    status = yaml.safe_load(STATUS.read_text())
    assert status["experiment_id"] == "postv1_e19_coarse_orientation_preserving_r01"
    assert status["representation"]["lattice_vertex_count"] == 512
    assert status["representation"]["lattice_tetrahedron_count"] == 2058
    assert status["representation"]["carrier_resolution"] == ("exact_p2_round_two_10592_faces")
    assert status["certificate"]["every_coarse_tetrahedron_required"]
    assert not status["certificate"]["lipschitz_or_frobenius_bound_required"]
    assert status["public_gate"]["minimum_tangential_full_vector_retention"] == 0.25
    assert status["public_gate"]["repetitions"] == 2
    assert status["failure_disposition"] == (
        "close_e19_without_grid_control_tolerance_or_time_variant"
    )


def test_e19_runner_has_public_only_negative_guards() -> None:
    source = RUNNER.read_text()
    assert '"private_input_reads": 0' in source
    assert '"image_loads": 0' in source
    assert '"development_evidence_reads": 0' in source
    assert '"sealed_test_accesses": 0' in source
    assert '"gpu_hours": 0' in source
    assert '"cloud_invocations": 0' in source
    assert '"spend_usd": 0' in source
    assert "data/private" not in source
    assert "docs/assets" not in source
    assert "modal.run" not in source


def test_e19_runner_enforces_immutable_outputs_and_no_retry() -> None:
    source = RUNNER.read_text()
    assert "immutable E19 report exists" in source
    assert "immutable E19 artifact directory exists" in source
    assert "worker.join(MAXIMUM_TOTAL_SECONDS)" in source
    assert '"automatic_paid_retries": 0' in source
