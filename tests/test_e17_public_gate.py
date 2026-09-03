from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_e17_runner_has_no_protected_or_paid_dependency() -> None:
    source = (PROJECT_ROOT / "scripts/run_post_v1_e17_public_gate.py").read_text()
    for forbidden in (
        "docs/assets/",
        "models/private",
        "models/checkpoints",
        "data/private",
        "modal.App",
        "modal.Function",
    ):
        assert forbidden not in source
    for counter in (
        '"private_input_reads": 0',
        '"image_loads": 0',
        '"optimizer_steps": 0',
        '"development_evidence_reads": 0',
        '"modal_invocations": 0',
        '"sealed_test_accesses": 0',
        '"automatic_paid_retries": 0',
    ):
        assert counter in source


def test_e17_machine_status_records_fixed_profile_closure() -> None:
    path = PROJECT_ROOT / "configs/evaluation/post_v1_e17_coarse_bilipschitz_r01.yaml"
    if not path.exists():
        pytest.skip("private status record is intentionally absent from public snapshot")
    status = yaml.safe_load(path.read_text())
    assert status["status"] == "closed_public_tangential_motion_retention"
    assert status["prerequisite"]["e16_status"] == "closed_public_g16_total_wall_time"
    representation = status["representation"]
    assert representation["block_count"] == 1
    assert representation["lattice_nodes_per_axis"] == 8
    assert representation["three_dimensional_control_count"] == 512
    assert representation["free_interior_control_count"] == 216
    assert representation["boundary_controls_fixed_zero"] is True
    assert representation["chordal_unrefined_surface_substitution_allowed"] is False
    assert status["certificate"]["maximum_lipschitz_kappa"] == 0.5
    assert status["public_gate"]["state"] == "failed_tangential_motion_retention"
    execution = status["execution"]
    assert execution["public_runs"] == 1
    assert execution["attempted_repetitions"] == 1
    assert execution["passing_repetitions"] == 0
    assert execution["controls_status"] == "pass"
    assert execution["tangential_retained_displacement_ratio"] < 0.25
    assert execution["native_exact_endpoint_intersection_pairs"] == 0
    assert execution["partial_results_promoted"] is False
    for counter in (
        "private_input_reads",
        "image_loads",
        "optimizer_steps",
        "development_evidence_reads",
        "modal_invocations",
        "sealed_test_accesses",
        "automatic_paid_retries",
    ):
        assert execution[counter] == 0
