from __future__ import annotations

import subprocess
from itertools import pairwise
from pathlib import Path

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import (
    QualificationState,
    advance_qualification,
    load_contract,
    reject_sealed_capability,
)
from frayid.v2.g02_shortcut_resistant import G02_EXPERIMENT_ID

G02_GPU_TYPE = "L40S"
G02_QUALIFICATION_TIMEOUT_SECONDS = 900
G02_QUALIFICATION_RUN_ID = "target-cuda-r01"
G02_QUALIFICATION_RUNNER = "scripts/modal_v2_g02_cuda_qualification_r01.py"


def _git_output(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def build_g02_cuda_qualification_plan(
    *,
    project_root: Path,
    contract_path: Path,
    local_qualification_path: Path,
    local_lifecycle_path: Path,
    matched_arm_binding_path: Path,
    input_paths: dict[str, Path],
    output_path: Path,
    provider_rate_usd_per_hour: float | None,
    price_checked_at: str | None,
    maximum_cost_usd: float | None,
    dispatch_authorized: bool,
) -> Path:
    paths = [
        contract_path,
        local_qualification_path,
        local_lifecycle_path,
        matched_arm_binding_path,
        *input_paths.values(),
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G02 CUDA qualification plan is immutable")
    blockers: list[str] = []
    for path in paths[:-1]:
        if not path.exists():
            blockers.append(f"missing_input:{path.name}")
    contract = load_contract(contract_path) if contract_path.is_file() else None
    if contract is not None:
        if contract.experiment_id != G02_EXPERIMENT_ID:
            blockers.append("wrong_experiment_contract")
        if contract.qualification_state is not QualificationState.CHECKPOINT_RESTORED:
            blockers.append("g02_not_checkpoint_restored")
        if contract.scientific_state.value != "registered":
            blockers.append("scientific_state_not_registered")
        if contract.compute_cap.automatic_retries != 0:
            blockers.append("automatic_retries_not_zero")
    local = read_json(local_qualification_path) if local_qualification_path.is_file() else {}
    if (
        local.get("status") != "pass"
        or local.get("checkpoint", {}).get("required_state_complete") is not True
        or local.get("checkpoint", {}).get("same_device_next_step_replay_exact") is not True
        or local.get("scientific_attempt_marker_created") is not False
    ):
        blockers.append("local_g02_qualification_not_passing")
    lifecycle = read_json(local_lifecycle_path) if local_lifecycle_path.is_file() else {}
    if lifecycle.get("status") != "pass" or lifecycle.get("state") != "checkpoint_restored":
        blockers.append("local_g02_lifecycle_not_checkpoint_restored")
    arm = read_json(matched_arm_binding_path) if matched_arm_binding_path.is_file() else {}
    if (
        local.get("matched_arm_binding", {}).get("sha256")
        != (sha256_file(matched_arm_binding_path) if matched_arm_binding_path.is_file() else None)
        or arm.get("arms_use_separate_output_roots") is not True
    ):
        blockers.append("matched_arm_binding_not_exact")
    head = _git_output(project_root, "rev-parse", "HEAD")
    dirty = bool(_git_output(project_root, "status", "--porcelain", "--untracked-files=no"))
    if dirty:
        blockers.append("source_worktree_dirty")
    local_revision = local.get("source_revision")
    if isinstance(local_revision, str):
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", local_revision, head],
            cwd=project_root,
            check=False,
            timeout=30,
        ).returncode
        if ancestor != 0:
            blockers.append("local_qualification_source_not_in_head_history")
    else:
        blockers.append("local_qualification_source_missing")
    if contract is not None:
        contract_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", contract.source_commit, head],
            cwd=project_root,
            check=False,
            timeout=30,
        ).returncode
        if contract_ancestor != 0:
            blockers.append("contract_source_not_in_head_history")
    if provider_rate_usd_per_hour is None or provider_rate_usd_per_hour <= 0.0:
        blockers.append("provider_rate_not_recorded")
    if not price_checked_at:
        blockers.append("provider_rate_timestamp_not_recorded")
    if maximum_cost_usd is None or maximum_cost_usd <= 0.0:
        blockers.append("maximum_cost_not_recorded")
    minimum_cap = None
    if provider_rate_usd_per_hour is not None and provider_rate_usd_per_hour > 0.0:
        minimum_cap = provider_rate_usd_per_hour * G02_QUALIFICATION_TIMEOUT_SECONDS / 3600.0 * 1.2
        if maximum_cost_usd is not None and maximum_cost_usd + 1.0e-9 < minimum_cap:
            blockers.append("maximum_cost_lacks_20_percent_contingency")
    if not dispatch_authorized:
        blockers.append("manual_paid_dispatch_authorization_required")
    input_hashes = {
        name: sha256_file(path) for name, path in sorted(input_paths.items()) if path.is_file()
    }
    input_hashes.update(
        {
            name: sha256_file(path)
            for name, path in (
                ("local_qualification", local_qualification_path),
                ("local_lifecycle", local_lifecycle_path),
                ("matched_arm_binding", matched_arm_binding_path),
            )
            if path.is_file()
        }
    )
    command = [
        "modal",
        "run",
        "--detach",
        G02_QUALIFICATION_RUNNER,
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
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_g02_cuda_qualification_plan.v1",
            "status": "ready" if not blockers else "blocked",
            "qualification_only": True,
            "scientific_attempt": False,
            "experiment_id": G02_EXPERIMENT_ID,
            "qualification_run_id": G02_QUALIFICATION_RUN_ID,
            "source_commit": head,
            "contract_source_commit": contract.source_commit if contract is not None else None,
            "gpu": G02_GPU_TYPE,
            "timeout_seconds": G02_QUALIFICATION_TIMEOUT_SECONDS,
            "automatic_retries": 0,
            "client_disconnect_policy": "detached_remote_function_survives",
            "result_recovery": (
                "immutable_modal_volume_then_explicit_local_hash_verified_download"
            ),
            "provider_rate_usd_per_hour": provider_rate_usd_per_hour,
            "price_checked_at": price_checked_at,
            "maximum_cost_usd": maximum_cost_usd,
            "minimum_cost_cap_with_contingency_usd": minimum_cap,
            "contingency_fraction": 0.2,
            "dispatch_authorized": dispatch_authorized,
            "input_hashes": input_hashes,
            "command": command,
            "blockers": blockers,
            "sealed_test_accesses": 0,
        },
    )


def audit_g02_target_cuda_qualification(
    envelope_path: Path,
    claim_path: Path,
    plan_path: Path,
    local_lifecycle_path: Path,
    output_path: Path,
) -> Path:
    paths = [envelope_path, claim_path, plan_path, local_lifecycle_path, output_path]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G02 target-CUDA lifecycle output is immutable")
    envelope = read_json(envelope_path)
    claim = read_json(claim_path)
    plan = read_json(plan_path)
    local = read_json(local_lifecycle_path)
    report = envelope.get("qualification_report", {})
    blockers: list[str] = []
    if plan.get("status") != "ready" or plan.get("automatic_retries") != 0:
        blockers.append("cuda_plan_not_ready")
    if claim.get("qualification_only") is not True or claim.get("scientific_attempt") is not False:
        blockers.append("cuda_claim_scope_invalid")
    if envelope.get("status") != "pass" or envelope.get("scientific_attempt") is not False:
        blockers.append("cuda_envelope_not_passing")
    if (
        report.get("status") != "pass"
        or report.get("target_cuda_exercised") is not True
        or report.get("promotion_eligible") is not True
        or report.get("checkpoint", {}).get("same_device_next_step_replay_exact") is not True
        or report.get("extraction", {}).get("status") != "pass"
        or report.get("extraction", {}).get("device") != "cpu"
    ):
        blockers.append("target_cuda_qualification_gates_failed")
    if local.get("status") != "pass" or local.get("state") != "checkpoint_restored":
        blockers.append("local_lifecycle_not_checkpoint_restored")
    if not (
        envelope.get("source_revision")
        == claim.get("source_revision")
        == plan.get("source_commit")
        == report.get("source_revision")
    ):
        blockers.append("source_revision_binding_mismatch")
    if not (
        envelope.get("provider_rate_usd_per_hour")
        == claim.get("provider_rate_usd_per_hour")
        == plan.get("provider_rate_usd_per_hour")
    ):
        blockers.append("provider_rate_binding_mismatch")
    states = [
        QualificationState.CHECKPOINT_RESTORED,
        QualificationState.EVALUATOR_DRY,
        QualificationState.QUALIFIED,
    ]
    for previous, following in pairwise(states):
        advance_qualification(previous, following)
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_g02_target_cuda_qualification_lifecycle.v1",
            "experiment_id": G02_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "state": QualificationState.QUALIFIED.value,
            "transitions": [state.value for state in states],
            "input_hashes": {
                "envelope": sha256_file(envelope_path),
                "claim": sha256_file(claim_path),
                "plan": sha256_file(plan_path),
                "local_lifecycle": sha256_file(local_lifecycle_path),
            },
            "target_cuda_exercised": True,
            "scientific_attempt_marker_created": False,
            "automatic_retries": 0,
            "sealed_test_accesses": 0,
            "blockers": blockers,
        },
    )
