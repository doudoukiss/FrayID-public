from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_post_v1_e21_public_gate.py"
STATUS = PROJECT_ROOT / "configs/evaluation/post_v1_e21_active_determinant_tangent_r01.yaml"


def test_e21_status_freezes_active_tangent_direction_only() -> None:
    if not STATUS.exists():
        pytest.skip("private status record is intentionally absent from public snapshot")
    status = yaml.safe_load(STATUS.read_text())
    assert status["experiment_id"] == "postv1_e21_active_determinant_tangent_r01"
    assert status["representation"]["block_count"] == 4
    assert status["direction"]["active_ratio"] == 0.25
    assert status["direction"]["tangent_condition"] == (
        "first_order_determinant_change_equals_zero"
    )
    assert status["direction"]["normalized_kkt_residual_maximum"] == 1.0e-10
    assert status["direction"]["maximum_tangent_residual"] == 1.0e-10
    assert status["certificate"]["unchanged_from_e20"]
    assert status["public_gate"]["minimum_tangential_full_vector_retention"] == 0.25
    assert status["failure_disposition"] == (
        "close_e21_without_active_ratio_block_grid_residual_tolerance_or_time_variant"
    )


def test_e21_runner_has_public_only_negative_guards() -> None:
    source = RUNNER.read_text()
    for counter in (
        "private_input_reads",
        "image_loads",
        "development_evidence_reads",
        "sealed_test_accesses",
        "gpu_hours",
        "cloud_invocations",
        "spend_usd",
    ):
        assert f'"{counter}": 0' in source
    assert "data/private" not in source
    assert "docs/assets" not in source
    assert "modal.run" not in source


def test_e21_runner_enforces_profile_outputs_and_no_retry() -> None:
    source = RUNNER.read_text()
    assert "BLOCK_COUNT = 4" in source
    assert "ACTIVE_DETERMINANT_RATIO" in source
    assert "immutable E21 report exists" in source
    assert "immutable E21 artifact directory exists" in source
    assert "worker.join(MAXIMUM_TOTAL_SECONDS)" in source
    assert '"automatic_paid_retries": 0' in source
