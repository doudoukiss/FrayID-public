"""Run the preregistered public P5 normalized Tight-Inclusion oracle gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from frayid.io import write_json
from frayid.normalized_ti_oracle import CORRECTNESS_ID, normalized_ti_path_oracle
from frayid.planar_dat_certificate import planar_dat_path_certificate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "post_v1_p5_public_normalized_ti_gate.v1"
SEED = 20260831
ANALYTIC_QUERY_COUNT = 1024
EXPECTED_ABSOLUTE_FILTERED_REJECTIONS = 82
MAX_SECONDS = 60


def _load_p4_runner() -> Any:
    scripts_path = str((PROJECT_ROOT / "scripts").resolve())
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    path = PROJECT_ROOT / "scripts/run_post_v1_p4_public_gate.py"
    specification = importlib.util.spec_from_file_location("frayid_p4_runner", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load frozen P4 analytic fixture implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _same_normalization(first: dict[str, Any], second: dict[str, Any]) -> bool:
    a = first["normalized_trajectory"]
    b = second["normalized_trajectory"]
    return bool(
        a["scale"] == b["scale"]
        and a["center"] == b["center"]
        and a["start_sha256"] == b["start_sha256"]
        and a["end_sha256"] == b["end_sha256"]
    )


def _query(index: int, rng: np.random.Generator, p4: Any) -> dict[str, Any]:
    primitive, case, scale, _, mesh, start, proposal, dhat = p4._analytic_geometry(index, rng)
    filtered = planar_dat_path_certificate(
        mesh,
        start,
        proposal,
        dhat=dhat,
        verify_full_path=False,
    )
    raw = [normalized_ti_path_oracle(mesh, start, proposal) for _ in range(2)]
    accepted = filtered.accepted_vertices
    judged = [normalized_ti_path_oracle(mesh, start, accepted) for _ in range(2)]
    raw_reports = [value.report() for value in raw]
    judged_reports = [value.report() for value in judged]
    expected_raw_safe = case in ("separating", "zero_motion")
    raw_correct = all(value.collision_free == expected_raw_safe for value in raw)
    filtered_correct = all(value.collision_free for value in judged)
    deterministic = bool(
        _same_normalization(raw_reports[0], raw_reports[1])
        and _same_normalization(judged_reports[0], judged_reports[1])
        and raw[0].collision_free == raw[1].collision_free
        and judged[0].collision_free == judged[1].collision_free
    )
    absolute_safe, _, _ = p4._full_path_oracle(mesh, start, accepted)
    blockers = [
        *("raw_analytic_label_disagreement" for _ in range(not raw_correct)),
        *("filtered_path_oracle_rejection" for _ in range(not filtered_correct)),
        *("nondeterministic_normalization_or_decision" for _ in range(not deterministic)),
        *("planar_dat_mechanism_failure" for _ in range(filtered.status != "pass")),
    ]
    return {
        "index": index,
        "primitive": primitive,
        "case": case,
        "scale": scale,
        "expected_raw_collision_free": expected_raw_safe,
        "normalized_raw_collision_free": raw[0].collision_free,
        "normalized_filtered_collision_free": judged[0].collision_free,
        "absolute_filtered_collision_free": absolute_safe,
        "deterministic": deterministic,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "maximum_normalized_oracle_seconds": max(
            *(value.elapsed_seconds for value in raw),
            *(value.elapsed_seconds for value in judged),
        ),
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    p4 = _load_p4_runner()
    rng = np.random.default_rng(SEED)
    queries = [_query(index, rng, p4) for index in range(ANALYTIC_QUERY_COUNT)]
    failed = [value["index"] for value in queries if value["status"] != "pass"]
    absolute_rejections = [
        value["index"] for value in queries if not value["absolute_filtered_collision_free"]
    ]
    blockers = [
        *("normalized_oracle_disagreement" for _ in failed[:1]),
        *(
            "absolute_p4_diagnostic_not_reproduced"
            for _ in range(len(absolute_rejections) != EXPECTED_ABSOLUTE_FILTERED_REJECTIONS)
        ),
    ]
    elapsed = time.monotonic() - started
    if elapsed > MAX_SECONDS:
        blockers.append("wall_time_ceiling")
    return {
        "schema_version": REPORT_SCHEMA,
        "correctness_id": CORRECTNESS_ID,
        "gate": "public_scale_normalized_tight_inclusion_oracle",
        "status": "pass" if not blockers else "fail",
        "registered_revision": "6a7c06b",
        "scope": "public_analytic_primitives_only",
        "seed": SEED,
        "query_count": len(queries),
        "passed_query_count": len(queries) - len(failed),
        "failed_query_indices": failed,
        "primitive_counts": {
            name: sum(value["primitive"] == name for value in queries)
            for name in ("point_triangle", "edge_edge")
        },
        "case_counts": {
            name: sum(value["case"] == name for value in queries)
            for name in ("crossing", "tangent", "separating", "zero_motion")
        },
        "normalized_raw_label_disagreement_count": sum(
            "raw_analytic_label_disagreement" in value["blockers"] for value in queries
        ),
        "normalized_filtered_rejection_count": sum(
            "filtered_path_oracle_rejection" in value["blockers"] for value in queries
        ),
        "nondeterministic_query_count": sum(not value["deterministic"] for value in queries),
        "absolute_p4_filtered_rejection_count": len(absolute_rejections),
        "absolute_p4_filtered_rejection_indices": absolute_rejections,
        "maximum_normalized_oracle_seconds": max(
            value["maximum_normalized_oracle_seconds"] for value in queries
        ),
        "blockers": blockers,
        "elapsed_seconds": elapsed,
        "execution_counters": {
            "mesh_fixture_reads": 0,
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
        "schema_version": "post_v1_p5_public_failure_report.v1",
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
            "mesh_fixture_reads": 0,
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
        raise FileExistsError(f"immutable P5 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-p5-supervisor-") as directory:
        worker_report = Path(directory) / "worker_report.json"
        worker = multiprocessing.get_context("spawn").Process(
            target=_run_worker,
            args=(str(worker_report),),
        )
        worker.start()
        worker.join(MAX_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
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
