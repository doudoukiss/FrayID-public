from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import QualificationState, ScientificAttemptState, load_contract
from frayid.v2.g02_science import G02_SCIENCE_ARM_SCHEMA
from frayid.v2.g02_shortcut_resistant import G02_EXPERIMENT_ID

G02_SCIENCE_GPU_TYPE = "L40S"
G02_SCIENCE_RUNNER = "scripts/modal_v2_g02_science_r01.py"
G02_TARGET_PREFLIGHT_TIMEOUT_SECONDS = 900
G02_SCIENCE_TIMEOUT_SECONDS = 7200
G02_ENDPOINT_EVALUATION_TIMEOUT_SECONDS = 1800
G02_ENDPOINT_EVALUATION_RUNNER = "scripts/modal_v2_g02_endpoint_evaluation_r01.py"


def _git_output(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _is_ancestor(project_root: Path, revision: str, head: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, head],
            cwd=project_root,
            check=False,
            timeout=30,
        ).returncode
        == 0
    )


def _common_dispatch_checks(
    *,
    project_root: Path,
    contract_path: Path,
    qualification_lifecycle_path: Path,
    arm_binding_path: Path,
    expected_optimizer_steps: int,
) -> tuple[str, list[str], dict[str, Any]]:
    blockers: list[str] = []
    head = _git_output(project_root, "rev-parse", "HEAD")
    if _git_output(project_root, "status", "--porcelain", "--untracked-files=no"):
        blockers.append("source_worktree_dirty")
    contract = load_contract(contract_path)
    if contract.experiment_id != G02_EXPERIMENT_ID:
        blockers.append("wrong_experiment_contract")
    if contract.qualification_state is not QualificationState.QUALIFIED:
        blockers.append("g02_not_target_cuda_qualified")
    if contract.scientific_state is not ScientificAttemptState.REGISTERED:
        blockers.append("scientific_attempt_no_longer_registered")
    if not _is_ancestor(project_root, contract.source_commit, head):
        blockers.append("contract_source_not_in_head_history")
    lifecycle = read_json(qualification_lifecycle_path)
    if lifecycle.get("status") != "pass" or lifecycle.get("state") != "qualified":
        blockers.append("target_cuda_lifecycle_not_qualified")
    arm = read_json(arm_binding_path)
    common = arm.get("common", {})
    if arm.get("schema_version") != G02_SCIENCE_ARM_SCHEMA:
        blockers.append("science_arm_schema_invalid")
    if common.get("source_revision") != head:
        blockers.append("science_arm_source_not_exact_head")
    if common.get("schedule", {}).get("optimizer_steps") != expected_optimizer_steps:
        blockers.append("science_arm_schedule_mismatch")
    if common.get("automatic_retries") != 0:
        blockers.append("science_arm_retries_not_zero")
    if common.get("development_available_to_trainer") is not False:
        blockers.append("development_exposed_to_trainer")
    if common.get("sealed_test_accesses") != 0:
        blockers.append("sealed_test_capability_present")
    return head, blockers, arm


def _price_checks(
    *,
    blockers: list[str],
    rate: float | None,
    checked_at: str | None,
    cap: float | None,
    timeout_seconds: int,
) -> float | None:
    if rate is None or rate <= 0.0:
        blockers.append("provider_rate_not_recorded")
        return None
    if not checked_at:
        blockers.append("provider_rate_timestamp_not_recorded")
    minimum = rate * timeout_seconds / 3600.0 * 1.2
    if cap is None or cap <= 0.0:
        blockers.append("maximum_cost_not_recorded")
    elif cap + 1.0e-9 < minimum:
        blockers.append("maximum_cost_lacks_20_percent_contingency")
    return minimum


def build_g02_target_science_preflight_plan(
    *,
    project_root: Path,
    contract_path: Path,
    qualification_lifecycle_path: Path,
    local_preflight_report_path: Path,
    preflight_arm_binding_path: Path,
    output_path: Path,
    provider_rate_usd_per_hour: float | None,
    price_checked_at: str | None,
    maximum_cost_usd: float | None,
    dispatch_authorized: bool,
) -> Path:
    if output_path.exists():
        raise FileExistsError("G02 target science preflight plan is immutable")
    head, blockers, _ = _common_dispatch_checks(
        project_root=project_root,
        contract_path=contract_path,
        qualification_lifecycle_path=qualification_lifecycle_path,
        arm_binding_path=preflight_arm_binding_path,
        expected_optimizer_steps=3,
    )
    report = read_json(local_preflight_report_path)
    if (
        report.get("status") != "endpoint_frozen_unscored"
        or report.get("source_revision") != head
        or report.get("scientific_attempt") is not False
        or report.get("scientific_attempt_marker_created") is not False
        or report.get("development_outcomes_read") != 0
        or report.get("sealed_test_accesses") != 0
        or report.get("checkpoint", {}).get("same_device_next_step_replay_exact") is not True
        or len(report.get("stage_reports", [])) != 3
        or any(stage.get("status") != "pass" for stage in report.get("stage_reports", []))
    ):
        blockers.append("local_science_preflight_not_passing")
    minimum = _price_checks(
        blockers=blockers,
        rate=provider_rate_usd_per_hour,
        checked_at=price_checked_at,
        cap=maximum_cost_usd,
        timeout_seconds=G02_TARGET_PREFLIGHT_TIMEOUT_SECONDS,
    )
    if not dispatch_authorized:
        blockers.append("manual_paid_preflight_authorization_required")
    command = [
        "modal",
        "run",
        "--detach",
        G02_SCIENCE_RUNNER,
        "--source-revision",
        head,
        "--provider-rate-usd-per-hour",
        str(provider_rate_usd_per_hour or "<required>"),
        "--price-checked-at",
        price_checked_at or "<required>",
        "--maximum-cost-usd",
        str(maximum_cost_usd or "<required>"),
        "--mode",
        "preflight",
        "--paid-preflight-authorized",
    ]
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_g02_target_science_preflight_plan.v1",
            "status": "ready" if not blockers else "blocked",
            "experiment_id": G02_EXPERIMENT_ID,
            "run_id": "target-science-preflight-r02",
            "source_revision": head,
            "gpu": G02_SCIENCE_GPU_TYPE,
            "timeout_seconds": G02_TARGET_PREFLIGHT_TIMEOUT_SECONDS,
            "automatic_retries": 0,
            "scientific_attempt": False,
            "provider_rate_usd_per_hour": provider_rate_usd_per_hour,
            "price_checked_at": price_checked_at,
            "maximum_cost_usd": maximum_cost_usd,
            "minimum_cost_cap_with_contingency_usd": minimum,
            "input_hashes": {
                "qualification_lifecycle": sha256_file(qualification_lifecycle_path),
                "local_preflight_report": sha256_file(local_preflight_report_path),
                "preflight_arm_binding": sha256_file(preflight_arm_binding_path),
            },
            "client_disconnect_policy": "detached_remote_function_survives",
            "command": command,
            "sealed_test_accesses": 0,
            "blockers": blockers,
        },
    )


def audit_g02_target_science_preflight(
    envelope_path: Path,
    plan_path: Path,
    output_path: Path,
) -> Path:
    if output_path.exists():
        raise FileExistsError("G02 target science preflight audit is immutable")
    envelope = read_json(envelope_path)
    plan = read_json(plan_path)
    report = envelope.get("training_report", {})
    blockers: list[str] = []
    if plan.get("status") != "ready" or plan.get("scientific_attempt") is not False:
        blockers.append("target_preflight_plan_not_ready")
    if (
        envelope.get("status") != "pass"
        or envelope.get("scientific_attempt") is not False
        or envelope.get("automatic_retries") != 0
    ):
        blockers.append("target_preflight_envelope_not_passing")
    if not (
        envelope.get("source_revision")
        == plan.get("source_revision")
        == report.get("source_revision")
    ):
        blockers.append("target_preflight_source_mismatch")
    if (
        report.get("status") != "endpoint_frozen_unscored"
        or report.get("scientific_attempt_marker_created") is not False
        or report.get("development_outcomes_read") != 0
        or report.get("sealed_test_accesses") != 0
        or report.get("completed_optimizer_steps") != 3
        or report.get("checkpoint", {}).get("same_device_next_step_replay_exact") is not True
        or len(report.get("stage_reports", [])) != 3
        or any(stage.get("status") != "pass" for stage in report.get("stage_reports", []))
    ):
        blockers.append("target_preflight_training_gates_failed")
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_g02_target_science_preflight_audit.v1",
            "status": "pass" if not blockers else "fail",
            "state": "target_science_preflight_passed" if not blockers else "stopped",
            "source_revision": envelope.get("source_revision"),
            "wall_time_seconds": report.get("wall_time_seconds"),
            "cuda_peak_memory_bytes": report.get("cuda_peak_memory_bytes"),
            "input_hashes": {
                "envelope": sha256_file(envelope_path),
                "plan": sha256_file(plan_path),
            },
            "scientific_attempt_marker_created": False,
            "automatic_retries": 0,
            "sealed_test_accesses": 0,
            "blockers": blockers,
        },
    )


def build_g02_scientific_attempt_plan(
    *,
    project_root: Path,
    contract_path: Path,
    qualification_lifecycle_path: Path,
    target_preflight_audit_path: Path,
    science_arm_binding_path: Path,
    output_path: Path,
    provider_rate_usd_per_hour: float | None,
    price_checked_at: str | None,
    maximum_cost_usd: float | None,
    dispatch_authorized: bool,
) -> Path:
    if output_path.exists():
        raise FileExistsError("G02 scientific attempt plan is immutable")
    head, blockers, _ = _common_dispatch_checks(
        project_root=project_root,
        contract_path=contract_path,
        qualification_lifecycle_path=qualification_lifecycle_path,
        arm_binding_path=science_arm_binding_path,
        expected_optimizer_steps=600,
    )
    preflight = read_json(target_preflight_audit_path)
    if (
        preflight.get("status") != "pass"
        or preflight.get("state") != "target_science_preflight_passed"
        or preflight.get("source_revision") != head
        or preflight.get("scientific_attempt_marker_created") is not False
    ):
        blockers.append("target_science_preflight_not_passing_for_exact_source")
    minimum = _price_checks(
        blockers=blockers,
        rate=provider_rate_usd_per_hour,
        checked_at=price_checked_at,
        cap=maximum_cost_usd,
        timeout_seconds=G02_SCIENCE_TIMEOUT_SECONDS,
    )
    if not dispatch_authorized:
        blockers.append("manual_scientific_dispatch_authorization_required")
    command = [
        "modal",
        "run",
        "--detach",
        G02_SCIENCE_RUNNER,
        "--source-revision",
        head,
        "--provider-rate-usd-per-hour",
        str(provider_rate_usd_per_hour or "<required>"),
        "--price-checked-at",
        price_checked_at or "<required>",
        "--maximum-cost-usd",
        str(maximum_cost_usd or "<required>"),
        "--mode",
        "science",
        "--scientific-dispatch-authorized",
    ]
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_g02_scientific_attempt_plan.v1",
            "status": "ready" if not blockers else "blocked",
            "experiment_id": G02_EXPERIMENT_ID,
            "attempt_id": "scientific-attempt-r02",
            "source_revision": head,
            "gpu": G02_SCIENCE_GPU_TYPE,
            "timeout_seconds": G02_SCIENCE_TIMEOUT_SECONDS,
            "optimizer_steps": 600,
            "automatic_retries": 0,
            "scientific_attempt": True,
            "provider_rate_usd_per_hour": provider_rate_usd_per_hour,
            "price_checked_at": price_checked_at,
            "maximum_cost_usd": maximum_cost_usd,
            "minimum_cost_cap_with_contingency_usd": minimum,
            "input_hashes": {
                "qualification_lifecycle": sha256_file(qualification_lifecycle_path),
                "target_preflight_audit": sha256_file(target_preflight_audit_path),
                "science_arm_binding": sha256_file(science_arm_binding_path),
            },
            "attempt_marker_persistence": "fsync_then_modal_volume_commit_before_step_1",
            "client_disconnect_policy": "detached_remote_function_survives",
            "command": command,
            "sealed_test_accesses": 0,
            "blockers": blockers,
        },
    )


def build_g02_endpoint_evaluation_plan(
    *,
    project_root: Path,
    scientific_envelope_path: Path,
    checkpoint_path: Path,
    endpoint_evidence_path: Path,
    output_path: Path,
    provider_rate_usd_per_hour: float | None,
    price_checked_at: str | None,
    maximum_cost_usd: float | None,
    dispatch_authorized: bool,
) -> Path:
    if output_path.exists():
        raise FileExistsError("G02 endpoint evaluation plan is immutable")
    blockers: list[str] = []
    head = _git_output(project_root, "rev-parse", "HEAD")
    if _git_output(project_root, "status", "--porcelain", "--untracked-files=no"):
        blockers.append("evaluator_source_worktree_dirty")
    scientific = read_json(scientific_envelope_path)
    report = scientific.get("training_report", {})
    if (
        scientific.get("status") != "endpoint_frozen_unscored"
        or scientific.get("attempt_id") != "scientific-attempt-r02"
        or scientific.get("scientific_attempt") is not True
        or report.get("scientific_attempt_marker_created") is not True
        or report.get("completed_optimizer_steps") != 600
        or report.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint_path)
    ):
        blockers.append("scientific_endpoint_not_frozen_and_hash_bound")
    try:
        import numpy as np

        with np.load(endpoint_evidence_path, allow_pickle=False) as archive:
            if (
                int(archive["train_records"]) != 144
                or int(archive["development_records"]) != 36
                or int(archive["sealed_test_records"]) != 0
                or str(archive["attempt_id"]) != "scientific-attempt-r02"
            ):
                blockers.append("endpoint_evidence_boundary_invalid")
    except (KeyError, OSError, ValueError):
        blockers.append("endpoint_evidence_unreadable")
    minimum = _price_checks(
        blockers=blockers,
        rate=provider_rate_usd_per_hour,
        checked_at=price_checked_at,
        cap=maximum_cost_usd,
        timeout_seconds=G02_ENDPOINT_EVALUATION_TIMEOUT_SECONDS,
    )
    if not dispatch_authorized:
        blockers.append("manual_endpoint_evaluation_dispatch_authorization_required")
    command = [
        "modal",
        "run",
        "--detach",
        G02_ENDPOINT_EVALUATION_RUNNER,
        "--evaluator-source-revision",
        head,
        "--provider-rate-usd-per-hour",
        str(provider_rate_usd_per_hour or "<required>"),
        "--price-checked-at",
        price_checked_at or "<required>",
        "--maximum-cost-usd",
        str(maximum_cost_usd or "<required>"),
        "--dispatch-authorized",
    ]
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_g02_endpoint_evaluation_plan.v1",
            "status": "ready" if not blockers else "blocked",
            "experiment_id": G02_EXPERIMENT_ID,
            "attempt_id": "scientific-attempt-r02",
            "evaluation_id": "independent-endpoint-evaluation-r02",
            "evaluator_source_revision": head,
            "training_source_revision": scientific.get("source_revision"),
            "gpu": G02_SCIENCE_GPU_TYPE,
            "timeout_seconds": G02_ENDPOINT_EVALUATION_TIMEOUT_SECONDS,
            "automatic_retries": 0,
            "optimizer_steps": 0,
            "development_records_read": 36,
            "provider_rate_usd_per_hour": provider_rate_usd_per_hour,
            "price_checked_at": price_checked_at,
            "maximum_cost_usd": maximum_cost_usd,
            "minimum_cost_cap_with_contingency_usd": minimum,
            "input_hashes": {
                "scientific_envelope": sha256_file(scientific_envelope_path),
                "checkpoint": sha256_file(checkpoint_path),
                "endpoint_evidence": sha256_file(endpoint_evidence_path),
            },
            "command": command,
            "sealed_test_accesses": 0,
            "blockers": blockers,
        },
    )


def audit_g02_endpoint_evaluation(
    envelope_path: Path,
    plan_path: Path,
    output_path: Path,
) -> Path:
    if output_path.exists():
        raise FileExistsError("G02 endpoint evaluation audit is immutable")
    envelope = read_json(envelope_path)
    plan = read_json(plan_path)
    report = envelope.get("evaluation_report", {})
    blockers: list[str] = []
    if plan.get("status") != "ready" or plan.get("optimizer_steps") != 0:
        blockers.append("endpoint_evaluation_plan_not_ready")
    if envelope.get("status") not in {"pass", "fail"}:
        blockers.append("endpoint_evaluation_has_no_terminal_scientific_verdict")
    if not (
        envelope.get("evaluator_source_revision") == plan.get("evaluator_source_revision")
        and envelope.get("training_source_revision") == plan.get("training_source_revision")
        and envelope.get("attempt_id") == plan.get("attempt_id")
    ):
        blockers.append("endpoint_evaluation_identity_mismatch")
    if (
        envelope.get("optimizer_steps") != 0
        or envelope.get("development_records_read") != 36
        or envelope.get("sealed_test_accesses") != 0
        or envelope.get("automatic_retries") != 0
        or report.get("optimizer_steps") != 0
        or report.get("development_records_used_for_fit") != 0
        or report.get("sealed_test_accesses") != 0
        or report.get("authoritative_result_claimed") is not False
    ):
        blockers.append("endpoint_evaluation_scope_invalid")
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_g02_endpoint_evaluation_audit.v1",
            "status": "pass" if not blockers else "fail",
            "scientific_gate_status": report.get("status"),
            "experiment_id": G02_EXPERIMENT_ID,
            "attempt_id": envelope.get("attempt_id"),
            "evaluator_source_revision": envelope.get("evaluator_source_revision"),
            "training_source_revision": envelope.get("training_source_revision"),
            "input_hashes": {
                "envelope": sha256_file(envelope_path),
                "plan": sha256_file(plan_path),
            },
            "optimizer_steps": 0,
            "sealed_test_accesses": 0,
            "blockers": blockers,
        },
    )
