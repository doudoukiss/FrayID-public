from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_g16_runner_has_no_protected_input_or_cloud_dependency() -> None:
    source = (PROJECT_ROOT / "scripts/run_post_v1_g16_public_gate.py").read_text()
    for forbidden in (
        "docs/assets/",
        "models/private",
        "models/checkpoints",
        "data/private",
        "modal.App",
        "modal.Function",
    ):
        assert forbidden not in source
    assert '"private_input_reads": 0' in source
    assert '"development_evidence_reads": 0' in source
    assert '"sealed_test_accesses": 0' in source
    assert '"modal_invocations": 0' in source


def test_e16_machine_status_records_g16_closure_and_blocks_downstream() -> None:
    path = PROJECT_ROOT / "configs/evaluation/post_v1_e16_ambient_inverse_reconstruction_r01.yaml"
    if not path.exists():
        pytest.skip("private status record is intentionally absent from public snapshot")
    status = yaml.safe_load(path.read_text())
    assert status["status"] == "closed_public_g16_total_wall_time"
    assert status["gates"]["g16_public_global_path"]["state"] == "failed_total_wall_time"
    for gate in (
        "public_image_reconstruction",
        "certified_private_initialization",
        "paired_private_smoke",
        "bounded_private_comparison",
        "sdf_bridge",
        "one_shot_development",
    ):
        assert status["gates"][gate]["state"].startswith("closed_unreachable")
    execution = status["execution"]
    assert execution["g16_runs"] == 1
    assert execution["g16_failure_class"] == "total_wall_time"
    assert execution["g16_completed_repetitions"] == 0
    assert execution["g16_partial_results_promoted"] is False
    for counter in (
        "public_image_reconstruction_runs",
        "private_input_reads",
        "optimizer_steps",
        "development_evaluations",
        "modal_invocations",
        "sealed_test_accesses",
        "automatic_paid_retries",
    ):
        assert execution[counter] == 0
    assert status["conditional_followups"]["e17_coarse_bilipschitz"]["state"].startswith("eligible")
    assert status["conditional_followups"]["e18_profiled_shape_motion"]["state"].startswith(
        "blocked"
    )
