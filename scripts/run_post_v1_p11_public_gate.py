"""Run the preregistered public P11 full isolated-upstream filter gate."""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_post_v1_e14_public_gate as e14
import run_post_v1_p9_public_gate as p9
from frayid.full_isolated_upstream_filter import (
    FullIsolatedUpstreamFilter,
    full_isolated_upstream_filter,
)
from frayid.genus_carrier import PUBLIC_FIDELITY_INPUT_SHA256
from frayid.io import write_json

CORRECTNESS_ID = "postv1_p11_full_isolated_upstream_filter_r01"
REPORT_SCHEMA = "post_v1_p11_public_full_isolated_upstream_filter_gate.v1"
REGISTERED_REVISION = "7b32aba"
MAXIMUM_FILTER_SECONDS = 60.0
MAXIMUM_ORACLE_SECONDS = 120.0
MAX_SECONDS = 30 * 60


def _repeat_filter(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dhat: float,
    dmin: float = 0.0,
) -> FullIsolatedUpstreamFilter:
    first = full_isolated_upstream_filter(
        collision_mesh, vertices_t0, vertices_t1, dhat=dhat, dmin=dmin
    )
    second = full_isolated_upstream_filter(
        collision_mesh, vertices_t0, vertices_t1, dhat=dhat, dmin=dmin
    )
    deterministic = bool(
        first.candidate_keys == second.candidate_keys
        and np.array_equal(first.candidate_ids, second.candidate_ids)
        and np.array_equal(first.candidate_kinds, second.candidate_kinds)
        and np.array_equal(first.isolated_contributions, second.isolated_contributions)
        and np.array_equal(first.truncation_ratios, second.truncation_ratios)
        and np.array_equal(first.filtered_displacements, second.filtered_displacements)
        and np.array_equal(first.accepted_vertices, second.accepted_vertices)
    )
    blockers = list(first.blockers)
    blockers.extend(value for value in second.blockers if value not in blockers)
    if not deterministic:
        blockers.append("full_filter_nondeterminism")
    return dataclasses.replace(
        first,
        status="pass" if not blockers else "fail",
        coefficient_seconds=max(first.coefficient_seconds, second.coefficient_seconds),
        batched_seconds=first.batched_seconds + second.batched_seconds,
        full_isolated_filter_seconds=(
            first.full_isolated_filter_seconds + second.full_isolated_filter_seconds
        ),
        blockers=tuple(blockers),
    )


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    constructor, auditor = e14.build_tools()
    p9_module: Any = p9
    original_certificate = p9_module.certify_candidate_contributions
    original_batched_ceiling = p9.MAXIMUM_BATCHED_SECONDS
    original_oracle_ceiling = p9.MAXIMUM_ORACLE_SECONDS
    try:
        p9_module.certify_candidate_contributions = _repeat_filter
        p9.MAXIMUM_BATCHED_SECONDS = MAXIMUM_FILTER_SECONDS
        p9.MAXIMUM_ORACLE_SECONDS = MAXIMUM_ORACLE_SECONDS
        with tempfile.TemporaryDirectory(prefix="frayid-p11-public-") as directory:
            hairpin = p9._run_hairpin(constructor, auditor, Path(directory))
    finally:
        p9_module.certify_candidate_contributions = original_certificate
        p9.MAXIMUM_BATCHED_SECONDS = original_batched_ceiling
        p9.MAXIMUM_ORACLE_SECONDS = original_oracle_ceiling
    blockers = [] if hairpin["status"] == "pass" else ["hairpin"]
    elapsed = time.monotonic() - started
    if elapsed > MAX_SECONDS:
        blockers.append("wall_time_ceiling")
    return {
        "schema_version": REPORT_SCHEMA,
        "correctness_id": CORRECTNESS_ID,
        "gate": "public_full_isolated_upstream_mesh_certificate",
        "status": "pass" if not blockers else "fail",
        "registered_revision": REGISTERED_REVISION,
        "scope": "public_procedural_geometry_only",
        "inherited_e11_fixture_definition_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
        "hairpin": hairpin,
        "blockers": blockers,
        "elapsed_seconds": elapsed,
        "execution_counters": {
            "private_input_reads": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "development_evidence_reads": 0,
            "modal_invocations": 0,
            "sealed_test_accesses": 0,
            "automatic_retries": 0,
        },
    }


def _run_worker(report_path: str) -> None:
    write_json(Path(report_path), run_public_gate())


def _failure_report(failure_class: str, started: float, exitcode: int | None) -> dict[str, Any]:
    return {
        "schema_version": "post_v1_p11_public_failure_report.v1",
        "correctness_id": CORRECTNESS_ID,
        "status": "fail",
        "failure_class": failure_class,
        "worker_exitcode": exitcode,
        "wall_time_ceiling_seconds": MAX_SECONDS,
        "elapsed_seconds": time.monotonic() - started,
        "automatic_retry_count": 0,
        "partial_results_promoted": False,
        "blockers": [failure_class],
        "execution_counters": {
            "private_input_reads": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "development_evidence_reads": 0,
            "modal_invocations": 0,
            "sealed_test_accesses": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"immutable P11 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-p11-supervisor-") as directory:
        worker_report = Path(directory) / "worker_report.json"
        worker = multiprocessing.get_context("spawn").Process(
            target=_run_worker,
            args=(str(worker_report),),
        )
        worker.start()
        worker.join(MAX_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(30)
            report = _failure_report("wall_time_ceiling_exceeded", started, worker.exitcode)
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
