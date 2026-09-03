from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import QualificationState, load_contract, reject_sealed_capability
from frayid.v2.l03_open_layers import L03_EXPERIMENT_ID
from frayid.v2.l03_training import L03_TRAINING_QUALIFICATION_PLAN_SCHEMA

L03_GPU_TYPE = "L40S"
L03_QUALIFICATION_TIMEOUT_SECONDS = 900
L03_QUALIFICATION_RUN_ID = "target-cuda-r01"
L03_QUALIFICATION_RUNNER = "scripts/modal_v2_l03_cuda_qualification_r01.py"


def _git_output(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def build_l03_cuda_qualification_plan(
    *,
    project_root: Path,
    contract_path: Path,
    local_qualification_path: Path,
    training_qualification_plan_path: Path,
    output_path: Path,
    provider_rate_usd_per_hour: float | None,
    price_checked_at: str | None,
    maximum_cost_usd: float | None,
    dispatch_authorized: bool,
) -> Path:
    """Build a fail-closed, qualification-only L03 CUDA dispatch plan."""

    paths = [
        contract_path,
        local_qualification_path,
        training_qualification_plan_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("L03 CUDA qualification plan is immutable")
    blockers: list[str] = []
    for path in paths[:-1]:
        if not path.is_file():
            blockers.append(f"missing_input:{path.name}")
    contract = load_contract(contract_path) if contract_path.is_file() else None
    if contract is not None:
        if contract.experiment_id != L03_EXPERIMENT_ID:
            blockers.append("wrong_experiment_contract")
        if contract.qualification_state is not QualificationState.CHECKPOINT_RESTORED:
            blockers.append("l03_not_checkpoint_restored")
        if contract.scientific_state.value != "registered":
            blockers.append("scientific_state_not_registered")
        if contract.compute_cap.automatic_retries != 0:
            blockers.append("automatic_retries_not_zero")
    local = read_json(local_qualification_path) if local_qualification_path.is_file() else {}
    if (
        local.get("status") != "pass"
        or local.get("decision") != "local_training_qualification_passed_target_gpu_pending"
        or local.get("gates", {}).get("same_device_checkpoint_restore_exact") is not True
        or local.get("provenance", {}).get("scientific_attempt_marker_created") is not False
        or local.get("provenance", {}).get("development_records_read") != 0
        or local.get("provenance", {}).get("sealed_test_reads") != 0
    ):
        blockers.append("local_l03_qualification_not_passing")
    training_plan = (
        read_json(training_qualification_plan_path)
        if training_qualification_plan_path.is_file()
        else {}
    )
    if (
        training_plan.get("schema_version") != L03_TRAINING_QUALIFICATION_PLAN_SCHEMA
        or training_plan.get("status") != "local_training_qualification_planned"
        or local.get("plan_sha256")
        != (
            sha256_file(training_qualification_plan_path)
            if training_qualification_plan_path.is_file()
            else None
        )
    ):
        blockers.append("training_qualification_plan_not_bound")
    head = _git_output(project_root, "rev-parse", "HEAD")
    dirty = bool(_git_output(project_root, "status", "--porcelain", "--untracked-files=no"))
    if dirty:
        blockers.append("source_worktree_dirty")
    local_revision = local.get("source_revision")
    if (
        not isinstance(local_revision, str)
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", local_revision or "missing", head],
            cwd=project_root,
            check=False,
            timeout=30,
        ).returncode
        != 0
    ):
        blockers.append("local_qualification_source_not_in_head_history")
    if (
        contract is not None
        and subprocess.run(
            ["git", "merge-base", "--is-ancestor", contract.source_commit, head],
            cwd=project_root,
            check=False,
            timeout=30,
        ).returncode
        != 0
    ):
        blockers.append("contract_source_not_in_head_history")
    if provider_rate_usd_per_hour is None or provider_rate_usd_per_hour <= 0.0:
        blockers.append("provider_rate_not_recorded")
    if not price_checked_at:
        blockers.append("provider_rate_timestamp_not_recorded")
    if maximum_cost_usd is None or maximum_cost_usd <= 0.0:
        blockers.append("maximum_cost_not_recorded")
    minimum_cap = None
    if provider_rate_usd_per_hour is not None and provider_rate_usd_per_hour > 0.0:
        minimum_cap = provider_rate_usd_per_hour * L03_QUALIFICATION_TIMEOUT_SECONDS / 3600.0 * 1.2
        if maximum_cost_usd is not None and maximum_cost_usd + 1.0e-9 < minimum_cap:
            blockers.append("maximum_cost_lacks_20_percent_contingency")
    if not dispatch_authorized:
        blockers.append("manual_paid_dispatch_authorization_required")
    command = [
        "modal",
        "run",
        L03_QUALIFICATION_RUNNER,
        "--source-revision",
        head,
        "--provider-rate-usd-per-hour",
        str(provider_rate_usd_per_hour or "<required>"),
        "--price-checked-at",
        price_checked_at or "<required>",
        "--maximum-cost-usd",
        str(maximum_cost_usd or "<required>"),
        "--dispatch-authorized",
    ]
    payload: dict[str, Any] = {
        "schema_version": "frayid_v2_l03_cuda_qualification_plan.v1",
        "status": "ready" if not blockers else "blocked",
        "qualification_only": True,
        "scientific_attempt": False,
        "experiment_id": L03_EXPERIMENT_ID,
        "qualification_run_id": L03_QUALIFICATION_RUN_ID,
        "source_commit": head,
        "contract_source_commit": contract.source_commit if contract is not None else None,
        "gpu": L03_GPU_TYPE,
        "timeout_seconds": L03_QUALIFICATION_TIMEOUT_SECONDS,
        "automatic_retries": 0,
        "provider_rate_usd_per_hour": provider_rate_usd_per_hour,
        "price_checked_at": price_checked_at,
        "maximum_cost_usd": maximum_cost_usd,
        "minimum_cost_cap_with_contingency_usd": minimum_cap,
        "contingency_fraction": 0.2,
        "dispatch_authorized": dispatch_authorized,
        "input_hashes": {
            "local_qualification": sha256_file(local_qualification_path)
            if local_qualification_path.is_file()
            else None,
            "training_qualification_plan": sha256_file(training_qualification_plan_path)
            if training_qualification_plan_path.is_file()
            else None,
        },
        "command": command,
        "blockers": blockers,
        "sealed_test_accesses": 0,
    }
    return write_json(output_path, payload)


def audit_l03_target_cuda_qualification(
    *, envelope_path: Path, plan_path: Path, output_path: Path
) -> Path:
    reject_sealed_capability([envelope_path, plan_path, output_path])
    if output_path.exists():
        raise FileExistsError("L03 target-CUDA audit output is immutable")
    envelope = read_json(envelope_path)
    plan = read_json(plan_path)
    report = envelope.get("qualification_report", {})
    blockers: list[str] = []
    if plan.get("status") != "ready" or plan.get("automatic_retries") != 0:
        blockers.append("cuda_plan_not_ready")
    if (
        envelope.get("status") != "pass"
        or envelope.get("scientific_attempt") is not False
        or envelope.get("automatic_retries") != 0
    ):
        blockers.append("cuda_envelope_not_passing")
    if (
        report.get("status") != "pass"
        or report.get("device") != "cuda"
        or report.get("gates", {}).get("both_layer_gradients_active") is not True
        or report.get("gates", {}).get("both_layer_parameters_change") is not True
        or report.get("gates", {}).get("same_device_checkpoint_restore_exact") is not True
    ):
        blockers.append("cuda_qualification_gates_failed")
    if not (
        envelope.get("source_revision")
        == plan.get("source_commit")
        == report.get("packaged_source_revision")
    ):
        blockers.append("source_revision_binding_mismatch")
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_l03_target_cuda_qualification_audit.v1",
            "experiment_id": L03_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "state": QualificationState.QUALIFIED.value if not blockers else "checkpoint_restored",
            "input_hashes": {
                "envelope": sha256_file(envelope_path),
                "plan": sha256_file(plan_path),
            },
            "target_cuda_exercised": not blockers,
            "scientific_attempt_marker_created": False,
            "automatic_retries": 0,
            "sealed_test_accesses": 0,
            "blockers": blockers,
        },
    )
