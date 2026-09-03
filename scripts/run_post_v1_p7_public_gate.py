"""Run the preregistered public P7 isotropic trust-region mesh gate."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import tempfile
import time
from pathlib import Path
from typing import Any

import ipctk  # type: ignore[import-not-found]
import numpy as np

import run_post_v1_p6_public_gate as p6
from frayid.io import write_json
from frayid.isotropic_trust_certificate import isotropic_trust_path_certificate
from frayid.shrinkwrap_carrier import _unique_edges

CORRECTNESS_ID = "postv1_p7_isotropic_trust_mesh_certificate_r01"
REPORT_SCHEMA = "post_v1_p7_public_isotropic_trust_mesh_gate.v1"
MAXIMUM_HAIRPIN_MECHANISM_SECONDS = 5.0
MAXIMUM_HAIRPIN_ORACLE_SECONDS = 120.0
MAX_SECONDS = 30 * 60


def _mechanism_worker(
    sender: Any,
    combined: np.ndarray,
    combined_faces: np.ndarray,
    wrap_count: int,
    proposal: np.ndarray,
    dhat: float,
) -> None:
    mesh = ipctk.CollisionMesh(
        np.asfortranarray(combined, dtype=np.float64),
        np.asfortranarray(_unique_edges(combined_faces), dtype=np.int32),
        np.asfortranarray(combined_faces, dtype=np.int32),
    )
    mesh.can_collide = ipctk.make_static_obstacle_filter(wrap_count)
    result = isotropic_trust_path_certificate(mesh, combined, proposal, dhat=dhat)
    sender.send(
        {
            "certificate": result.report(),
            "accepted_vertices": result.accepted_vertices,
        }
    )
    sender.close()


def _timed_mechanism(
    combined: np.ndarray,
    combined_faces: np.ndarray,
    wrap_count: int,
    proposal: np.ndarray,
    dhat: float,
) -> tuple[dict[str, Any] | None, float]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    started = time.monotonic()
    worker = context.Process(
        target=_mechanism_worker,
        args=(sender, combined, combined_faces, wrap_count, proposal, dhat),
    )
    worker.start()
    sender.close()
    worker.join(MAXIMUM_HAIRPIN_MECHANISM_SECONDS)
    elapsed = time.monotonic() - started
    if worker.is_alive():
        worker.terminate()
        worker.join(10)
        receiver.close()
        return None, elapsed
    payload = receiver.recv() if worker.exitcode == 0 and receiver.poll() else None
    receiver.close()
    return payload, elapsed


def _mechanism(
    fixture_name: str,
    mesh: Any,
    combined: np.ndarray,
    combined_faces: np.ndarray,
    wrap_count: int,
    proposal: np.ndarray,
    dhat: float,
) -> tuple[dict[str, Any] | None, np.ndarray | None, float | None]:
    if fixture_name == "near_contact_hairpin":
        payload, supervised = _timed_mechanism(
            combined,
            combined_faces,
            wrap_count,
            proposal,
            dhat,
        )
        if payload is None:
            return None, None, supervised
        return (
            payload["certificate"],
            np.asfortranarray(payload["accepted_vertices"], dtype=np.float64),
            supervised,
        )
    result = isotropic_trust_path_certificate(mesh, combined, proposal, dhat=dhat)
    return result.report(), result.accepted_vertices, None


def run_public_gate() -> dict[str, Any]:
    original_mechanism = p6._mechanism
    original_mechanism_ceiling = p6.MAXIMUM_HAIRPIN_MECHANISM_SECONDS
    original_oracle_ceiling = p6.MAXIMUM_HAIRPIN_ORACLE_SECONDS
    try:
        p6._mechanism = _mechanism
        p6.MAXIMUM_HAIRPIN_MECHANISM_SECONDS = MAXIMUM_HAIRPIN_MECHANISM_SECONDS
        p6.MAXIMUM_HAIRPIN_ORACLE_SECONDS = MAXIMUM_HAIRPIN_ORACLE_SECONDS
        report = p6.run_public_gate()
    finally:
        p6._mechanism = original_mechanism
        p6.MAXIMUM_HAIRPIN_MECHANISM_SECONDS = original_mechanism_ceiling
        p6.MAXIMUM_HAIRPIN_ORACLE_SECONDS = original_oracle_ceiling
    report.update(
        {
            "schema_version": REPORT_SCHEMA,
            "correctness_id": CORRECTNESS_ID,
            "gate": "public_isotropic_trust_normalized_mesh_path_certificate",
            "registered_revision": "aa8d179",
        }
    )
    return report


def _run_worker(report_path: str) -> None:
    write_json(Path(report_path), run_public_gate())


def _failure_report(failure_class: str, started: float, exitcode: int | None) -> dict[str, Any]:
    return {
        "schema_version": "post_v1_p7_public_failure_report.v1",
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
        raise FileExistsError(f"immutable P7 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-p7-supervisor-") as directory:
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
