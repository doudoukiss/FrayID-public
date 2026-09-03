from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_post_v1_e20_public_gate.py"
STATUS = PROJECT_ROOT / "configs/evaluation/post_v1_e20_composed_orientation_preserving_r01.yaml"


def test_e20_status_freezes_four_block_material_composition() -> None:
    if not STATUS.exists():
        pytest.skip("private status record is intentionally absent from public snapshot")
    status = yaml.safe_load(STATUS.read_text())
    assert status["experiment_id"] == "postv1_e20_composed_orientation_preserving_r01"
    assert status["status"] == "closed_public_tangential_motion_retention"
    assert status["representation"]["block_count"] == 4
    assert status["representation"]["lattice_vertex_count"] == 512
    assert status["representation"]["lattice_tetrahedron_count"] == 2058
    assert status["representation"][
        "material_tetrahedron_and_barycentric_identity_preserved_between_blocks"
    ]
    assert status["certificate"]["every_block_complete_path_required"]
    assert status["public_gate"]["minimum_tangential_full_vector_retention"] == 0.25
    assert status["public_gate"]["repetitions"] == 2
    assert status["failure_disposition"] == (
        "close_e20_without_block_grid_control_tolerance_or_time_variant"
    )
    assert status["execution"]["tangential_retained_displacement_ratio"] == (0.09314151713375927)
    assert status["execution"]["tangential_exact_endpoint_status"] == (
        "not_run_after_retention_failure"
    )


def test_e20_runner_has_public_only_negative_guards() -> None:
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


def test_e20_runner_enforces_profile_outputs_and_no_retry() -> None:
    source = RUNNER.read_text()
    assert "BLOCK_COUNT = 4" in source
    assert "immutable E20 report exists" in source
    assert "immutable E20 artifact directory exists" in source
    assert "worker.join(MAXIMUM_TOTAL_SECONDS)" in source
    assert '"automatic_paid_retries": 0' in source
