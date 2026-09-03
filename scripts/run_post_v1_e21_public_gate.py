"""Run the registered public-only E21 active-determinant tangent gate."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import run_post_v1_e17_public_gate as e17
import run_post_v1_e20_public_gate as e20
from frayid.active_tangent_orientation_map import (
    ACTIVE_DETERMINANT_RATIO,
    fit_and_certify_active_tangent_step,
    run_active_tangent_controls,
)
from frayid.genus_carrier import PUBLIC_FIDELITY_INPUT_SHA256
from frayid.io import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e21_active_determinant_tangent_r01"
REPORT_SCHEMA = "post_v1_e21_public_active_determinant_tangent_gate.v1"
SEED = 20260831
REPETITIONS = 2
BLOCK_COUNT = 4
NODES_PER_AXIS = 8
CONTROL_COUNT = 512
FREE_CONTROL_COUNT = 216
MINIMUM_TANGENTIAL_RETENTION = 0.25
MAXIMUM_TOTAL_SECONDS = 7_200.0
MAXIMUM_CERTIFICATE_SECONDS_PER_BLOCK = 60.0
MAXIMUM_ENDPOINT_AUDIT_SECONDS = 120.0
MAXIMUM_MEMORY_GIB = 16.0
CPU_CORE_LIMIT = 8


def _run_repetition(
    repetition: int,
    *,
    constructor: Path,
    auditor: Path,
    root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    namespace = vars(e20)
    original: Any = namespace["fit_and_certify_composed_orientation_step"]
    namespace["fit_and_certify_composed_orientation_step"] = fit_and_certify_active_tangent_step
    try:
        result = e20._run_repetition(
            repetition,
            constructor=constructor,
            auditor=auditor,
            root=root,
            artifact_root=artifact_root,
        )
    finally:
        namespace["fit_and_certify_composed_orientation_step"] = original
    result["direction_mechanism"] = "active_determinant_tangent_kkt"
    result["active_determinant_ratio"] = ACTIVE_DETERMINANT_RATIO
    return result


def run_public_gate(artifact_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_CORE_LIMIT))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_CORE_LIMIT))
    git = e17._git_binding()
    controls = run_active_tangent_controls()
    blockers: list[str] = []
    if not git["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")
    if controls["status"] != "pass":
        blockers.append("active_tangent_controls")
    repetitions: list[dict[str, Any]] = []
    if not blockers:
        constructor, auditor = e17._build_tools()
        with tempfile.TemporaryDirectory(prefix="frayid-e21-public-") as directory:
            root = Path(directory)
            for repetition in range(REPETITIONS):
                result = _run_repetition(
                    repetition,
                    constructor=constructor,
                    auditor=auditor,
                    root=root,
                    artifact_root=artifact_root,
                )
                repetitions.append(result)
                if result["status"] != "pass":
                    blockers.append(f"repetition_{repetition}")
                    break
    if len(repetitions) == REPETITIONS and all(
        repetition["status"] == "pass" for repetition in repetitions
    ):
        first, second = repetitions
        for key in ("lattice_sha256", "carrier_refinement", "source_refinement"):
            if first[key] != second[key]:
                blockers.append(f"repetition_mismatch:{key}")
        first_trajectories = first["trajectories"]
        second_trajectories = second["trajectories"]
        if [value["name"] for value in first_trajectories] != [
            value["name"] for value in second_trajectories
        ]:
            blockers.append("trajectory_identity_mismatch")
        elif any(
            left["step"]["decision_sha256"] != right["step"]["decision_sha256"]
            or left["artifact_sha256"] != right["artifact_sha256"]
            for left, right in zip(first_trajectories, second_trajectories, strict=True)
        ):
            blockers.append("trajectory_repetition_mismatch")
    elapsed = time.monotonic() - started
    peak_memory = e17._peak_memory_gib()
    if elapsed > MAXIMUM_TOTAL_SECONDS:
        blockers.append("total_wall_time")
    if peak_memory > MAXIMUM_MEMORY_GIB:
        blockers.append("resident_memory")
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_geometry_only",
        "git": git,
        "seed": SEED,
        "controls": controls,
        "repetitions": repetitions,
        "elapsed_seconds": elapsed,
        "peak_resident_memory_gib": peak_memory,
        "limits": {
            "cpu_cores": CPU_CORE_LIMIT,
            "resident_memory_gib": MAXIMUM_MEMORY_GIB,
            "total_wall_seconds": MAXIMUM_TOTAL_SECONDS,
            "certificate_seconds_per_block": MAXIMUM_CERTIFICATE_SECONDS_PER_BLOCK,
            "endpoint_audit_seconds_per_proposal": MAXIMUM_ENDPOINT_AUDIT_SECONDS,
            "nodes_per_axis": NODES_PER_AXIS,
            "control_count": CONTROL_COUNT,
            "free_control_count": FREE_CONTROL_COUNT,
            "block_count": BLOCK_COUNT,
            "active_determinant_ratio": ACTIVE_DETERMINANT_RATIO,
            "minimum_tangential_retention": MINIMUM_TANGENTIAL_RETENTION,
        },
        "execution_counters": {
            "private_input_reads": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "development_evidence_reads": 0,
            "modal_invocations": 0,
            "sealed_test_accesses": 0,
            "gpu_hours": 0,
            "cloud_invocations": 0,
            "spend_usd": 0,
            "automatic_paid_retries": 0,
        },
        "bindings": {
            "fixture_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
            "mechanism_source_sha256": e17._sha256(
                PROJECT_ROOT / "src/frayid/active_tangent_orientation_map.py"
            ),
            "certificate_source_sha256": e17._sha256(
                PROJECT_ROOT / "src/frayid/certified_tet_path.py"
            ),
            "runner_source_sha256": e17._sha256(Path(__file__)),
        },
        "blockers": blockers,
    }


def _worker(report_path: str, artifact_root: str) -> None:
    write_json(Path(report_path), run_public_gate(Path(artifact_root)))


def _failure_report(failure: str, started: float, exitcode: int | None) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "fail",
        "failure_class": failure,
        "worker_exitcode": exitcode,
        "elapsed_seconds": time.monotonic() - started,
        "automatic_retry_count": 0,
        "partial_results_promoted": False,
        "blockers": [failure],
        "execution_counters": {
            "private_input_reads": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "development_evidence_reads": 0,
            "modal_invocations": 0,
            "sealed_test_accesses": 0,
            "gpu_hours": 0,
            "cloud_invocations": 0,
            "spend_usd": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    artifact_root = arguments.output.parent / f"{arguments.output.stem}_artifacts"
    if arguments.output.exists():
        raise FileExistsError(f"immutable E21 report exists: {arguments.output}")
    if artifact_root.exists():
        raise FileExistsError(f"immutable E21 artifact directory exists: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-e21-supervisor-") as directory:
        worker_report = Path(directory) / "worker_report.json"
        worker = multiprocessing.get_context("spawn").Process(
            target=_worker, args=(str(worker_report), str(artifact_root))
        )
        worker.start()
        worker.join(MAXIMUM_TOTAL_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(30)
            report = _failure_report("total_wall_time", started, worker.exitcode)
        elif worker.exitcode != 0 or not worker_report.is_file():
            report = _failure_report("worker_failure", started, worker.exitcode)
        else:
            report = json.loads(worker_report.read_text())
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
