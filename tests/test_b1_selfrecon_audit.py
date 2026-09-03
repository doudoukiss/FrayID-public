from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/audit_post_v1_b1_selfrecon_public.py"
STATUS_PATH = PROJECT_ROOT / "configs/evaluation/post_v1_b1_selfrecon_public_baseline_r01.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("b1_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b1_registration_binds_full_official_public_path() -> None:
    if not STATUS_PATH.exists():
        pytest.skip("private status record is intentionally absent from public snapshot")
    status = yaml.safe_load(STATUS_PATH.read_text())
    assert status["status"] == "closed_public_prerequisites_unbound"
    assert status["upstream"]["revision"] == "344b86fc3e7617b94b5c9da3741c764ae93cacaa"
    assert status["assets"]["public_sequence"] == "male-3-casual"
    assert status["official_contract"]["epochs"] == 200
    assert status["official_contract"]["rgb_weights"] == [0.5, 1.0, 1.0]
    assert status["official_contract"]["normal_weight"] == 0.1
    assert not status["official_contract"]["nepoch_zero_is_pass"]
    assert not status["official_contract"]["rendering_only_is_pass"]
    assert not status["official_contract"]["file_existence_is_pass"]
    assert not status["budget"]["ceiling_is_spend_authorization"]
    assert status["execution"]["public_reconstruction_attempts"] == 0
    assert status["execution"]["epochs_completed"] == 0
    assert status["execution"]["spend_usd"] == 0
    report_sha256 = status["execution"]["report_sha256"]
    assert len(report_sha256) == 64
    int(report_sha256, 16)
    assert status["private_adapter"]["state"] == "closed_public_prerequisite_failure"


def test_b1_audit_rejects_protected_and_nonisolated_inputs() -> None:
    module = _module()
    with pytest.raises(ValueError, match="protected path"):
        module._assert_isolated_input(PROJECT_ROOT / "data/private", "fixture")
    with pytest.raises(ValueError, match="ignored external"):
        module._assert_isolated_input(PROJECT_ROOT / "docs", "fixture")


def test_b1_audit_has_no_download_or_training_side_effect() -> None:
    module = _module()
    source = SCRIPT_PATH.read_text()
    assert "urlopen" not in source
    assert "requests." not in source
    assert "git clone" not in source
    assert module._allowed_subprocesses() == ("git", "nvidia-smi")
    assert '"public_reconstruction_attempts": 0' in source
    assert '"private_input_reads": 0' in source
    assert '"sealed_test_accesses": 0' in source
