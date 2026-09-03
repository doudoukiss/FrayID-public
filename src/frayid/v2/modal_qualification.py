from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import QualificationState, load_contract, reject_sealed_capability

G01_EXPERIMENT_ID = "postv2_g01_direct_multires_field_outer_r01"
G01_GPU_TYPE = "L40S"
G01_QUALIFICATION_TIMEOUT_SECONDS = 900
G01_QUALIFICATION_RUN_ID = "target-cuda-r04"
G01_QUALIFICATION_RUNNER = "scripts/modal_v2_g01_cuda_qualification_r04.py"


def _git_output(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def build_g01_cuda_qualification_plan(
    *,
    project_root: Path,
    contract_path: Path,
    evidence_volume_path: Path,
    evidence_binding_path: Path,
    hull_qualification_path: Path,
    local_qualification_path: Path,
    lifecycle_path: Path,
    output_path: Path,
    provider_rate_usd_per_hour: float | None = None,
    price_checked_at: str | None = None,
    maximum_cost_usd: float | None = None,
    dispatch_authorized: bool = False,
) -> Path:
    """Build a fail-closed CUDA qualification plan without invoking Modal."""

    paths = [
        contract_path,
        evidence_volume_path,
        evidence_binding_path,
        hull_qualification_path,
        local_qualification_path,
        lifecycle_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G01 CUDA qualification plans are immutable")
    blockers: list[str] = []
    for path in paths[:-1]:
        if not path.is_file():
            blockers.append(f"missing_input:{path.name}")
    contract = load_contract(contract_path) if contract_path.is_file() else None
    if contract is not None:
        if contract.experiment_id != G01_EXPERIMENT_ID:
            blockers.append("wrong_experiment_contract")
        if contract.qualification_state is not QualificationState.EVALUATOR_DRY:
            blockers.append("g01_not_evaluator_dry")
        if contract.scientific_state.value != "registered":
            blockers.append("scientific_state_not_registered")
        if contract.compute_cap.automatic_retries != 0:
            blockers.append("automatic_retries_not_zero")
    local = read_json(local_qualification_path) if local_qualification_path.is_file() else {}
    if (
        local.get("status") != "pass"
        or local.get("apple_gpu_exercised") is not True
        or local.get("same_device_replay_exact") is not True
        or local.get("extraction", {}).get("device") != "cpu"
    ):
        blockers.append("local_mps_cpu_extract_qualification_not_passing")
    lifecycle = read_json(lifecycle_path) if lifecycle_path.is_file() else {}
    if lifecycle.get("status") != "pass" or lifecycle.get("state") != "evaluator_dry":
        blockers.append("g01_lifecycle_not_evaluator_dry")
    head = _git_output(project_root, "rev-parse", "HEAD")
    dirty = bool(_git_output(project_root, "status", "--porcelain", "--untracked-files=no"))
    if dirty:
        blockers.append("source_worktree_dirty")
    if contract is not None:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", contract.source_commit, head],
            cwd=project_root,
            check=False,
            timeout=30,
        ).returncode
        if ancestor != 0:
            blockers.append("contract_source_not_in_head_history")
    if provider_rate_usd_per_hour is None or provider_rate_usd_per_hour <= 0:
        blockers.append("provider_rate_not_recorded")
    if not price_checked_at:
        blockers.append("provider_rate_timestamp_not_recorded")
    if maximum_cost_usd is None or maximum_cost_usd <= 0:
        blockers.append("maximum_cost_not_recorded")
    if provider_rate_usd_per_hour and maximum_cost_usd:
        minimum_cap = provider_rate_usd_per_hour * G01_QUALIFICATION_TIMEOUT_SECONDS / 3600.0 * 1.2
        if maximum_cost_usd + 1.0e-9 < minimum_cap:
            blockers.append("maximum_cost_lacks_20_percent_contingency")
    if not dispatch_authorized:
        blockers.append("manual_paid_dispatch_authorization_required")
    inputs: dict[str, str] = {}
    for name, path in (
        ("evidence_volume", evidence_volume_path),
        ("evidence_binding", evidence_binding_path),
        ("hull_qualification", hull_qualification_path),
        ("local_qualification", local_qualification_path),
        ("lifecycle", lifecycle_path),
    ):
        if path.is_file():
            inputs[name] = sha256_file(path)
    command = [
        "modal",
        "run",
        G01_QUALIFICATION_RUNNER,
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
    plan: dict[str, Any] = {
        "schema_version": "frayid_v2_g01_cuda_qualification_plan.v1",
        "status": "ready" if not blockers else "blocked",
        "qualification_only": True,
        "scientific_attempt": False,
        "experiment_id": G01_EXPERIMENT_ID,
        "qualification_run_id": G01_QUALIFICATION_RUN_ID,
        "source_commit": head,
        "contract_source_commit": contract.source_commit if contract is not None else None,
        "gpu": G01_GPU_TYPE,
        "timeout_seconds": G01_QUALIFICATION_TIMEOUT_SECONDS,
        "automatic_retries": 0,
        "provider_rate_usd_per_hour": provider_rate_usd_per_hour,
        "price_checked_at": price_checked_at,
        "maximum_cost_usd": maximum_cost_usd,
        "contingency_fraction": 0.2,
        "dispatch_authorized": dispatch_authorized,
        "input_hashes": inputs,
        "command": command,
        "blockers": blockers,
        "sealed_test_accesses": 0,
    }
    return write_json(output_path, plan)
