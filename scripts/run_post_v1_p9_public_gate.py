"""Run the preregistered public P9 candidate-contribution certificate gate."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_post_v1_e12_public_gate as e12
import run_post_v1_e14_public_gate as e14
import run_post_v1_p3_public_gate as p3
import run_post_v1_p6_public_gate as p6
from frayid.barrier_sliding_carrier import DHAT_BBOX_FRACTION
from frayid.candidate_contribution_certificate import certify_candidate_contributions
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import PUBLIC_FIDELITY_INPUT_SHA256, public_genus_fidelity_fixtures
from frayid.io import write_json
from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS_ID = "postv1_p9_candidate_contribution_certificate_r01"
REPORT_SCHEMA = "post_v1_p9_public_candidate_contribution_gate.v1"
REGISTERED_REVISION = "96b1e2c"
MINIMUM_TANGENTIAL_RETENTION = 0.25
MAXIMUM_SCALAR_SECONDS = 300.0
MAXIMUM_BATCHED_SECONDS = 5.0
MAXIMUM_ORACLE_SECONDS = 120.0
MAX_SECONDS = 30 * 60


def _run_hairpin(constructor: Path, auditor: Path, root: Path) -> dict[str, Any]:
    fixture = next(
        value for value in public_genus_fidelity_fixtures() if value.name == "near_contact_hairpin"
    )
    fixture_root = root / fixture.name
    fixture_root.mkdir(parents=True, exist_ok=False)
    source_path = fixture_root / "source.e6mesh"
    e12._write_fixture(source_path, fixture)
    parent_path = fixture_root / "parent.e10mesh"
    completed = subprocess.run(
        [str(constructor), str(source_path), str(parent_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAX_SECONDS,
    )
    if completed.returncode != 0 or not parent_path.is_file():
        return {
            **fixture.as_public_record(),
            "status": "fail",
            "blockers": ["constructor_failure"],
            "diagnostic": (completed.stdout + completed.stderr).strip(),
        }

    parent_vertices, parent_faces = read_e10_mesh(parent_path)
    refinement = subdivide_with_exact_provenance(parent_vertices, parent_faces, rounds=2)
    p2_certificate = certify_exact_dyadic_refinement(
        parent_vertices,
        parent_faces,
        refinement,
        parent_grid=e14._grid_for_fixture(fixture),
        rounds=2,
    )
    initial_path = fixture_root / "initial.e10mesh"
    e12._write_mesh(initial_path, refinement.vertices, refinement.faces)
    initial_audit, initial_diagnostic = e14._audit(
        auditor,
        source_path,
        initial_path,
        fixture_root / "initial_audit.json",
    )
    blockers: list[str] = []
    if p2_certificate.status != "pass":
        blockers.append("p2_exact_refinement_certificate")
    if initial_audit.get("status") != "pass":
        blockers.append("initial_exact_audit")
    if blockers:
        return {
            **fixture.as_public_record(),
            "status": "fail",
            "p2_certificate": p2_certificate.report(),
            "initial_exact_audit": initial_audit,
            "initial_exact_diagnostic": initial_diagnostic,
            "blockers": blockers,
        }

    mesh, combined, _combined_faces, source_vertices = p3._combined_problem(
        refinement.vertices,
        refinement.faces,
        fixture,
    )
    diagonal = float(np.linalg.norm(np.ptp(combined, axis=0)))
    dhat = DHAT_BBOX_FRACTION * diagonal
    trajectories: list[dict[str, Any]] = []
    for name, wrap_proposal in p3._trajectory_proposals(
        refinement.vertices,
        refinement.faces,
        fixture,
    ):
        proposal = np.asfortranarray(np.vstack((wrap_proposal, source_vertices)))
        certificate = certify_candidate_contributions(
            mesh,
            combined,
            proposal,
            dhat=dhat,
        )
        trajectory_blockers = list(certificate.blockers)
        if certificate.scalar_seconds > MAXIMUM_SCALAR_SECONDS:
            trajectory_blockers.append("scalar_reference_time")
        if any(value > MAXIMUM_BATCHED_SECONDS for value in certificate.batched_seconds):
            trajectory_blockers.append("batched_reduction_time")
        if name == "tangential_sliding" and (
            certificate.retained_displacement_ratio < MINIMUM_TANGENTIAL_RETENTION
        ):
            trajectory_blockers.append("tangential_motion_retention")
        if name == "native_pressure" and certificate.filtered_displacement_norm <= 0.0:
            trajectory_blockers.append("native_motion_retention")

        oracle_report: dict[str, Any] | None = None
        oracle_seconds: float | None = None
        endpoint_audit: dict[str, Any] | None = None
        endpoint_diagnostic = ""
        if not trajectory_blockers:
            oracle_report, oracle_seconds = p6._timed_oracle(
                combined,
                _combined_faces,
                len(refinement.vertices),
                certificate.accepted_vertices,
            )
            if oracle_report is None or oracle_report["status"] != "pass":
                trajectory_blockers.append("normalized_full_path_oracle")
            elif oracle_report["elapsed_seconds"] > MAXIMUM_ORACLE_SECONDS:
                trajectory_blockers.append("normalized_oracle_time")
            endpoint_path = fixture_root / f"{name}_endpoint.e10mesh"
            e12._write_mesh(
                endpoint_path,
                certificate.accepted_vertices[: len(refinement.vertices)],
                refinement.faces,
            )
            endpoint_audit, endpoint_diagnostic = e14._audit(
                auditor,
                source_path,
                endpoint_path,
                fixture_root / f"{name}_endpoint_audit.json",
            )
            if endpoint_audit.get("status") != "pass":
                trajectory_blockers.append("independent_endpoint_exact_audit")

        blockers.extend(f"{name}:{value}" for value in trajectory_blockers)
        trajectories.append(
            {
                "name": name,
                "status": "pass" if not trajectory_blockers else "fail",
                "candidate_contribution_certificate": certificate.report(),
                "normalized_oracle": oracle_report,
                "supervised_oracle_seconds": oracle_seconds,
                "independent_endpoint_exact_audit": endpoint_audit,
                "exact_diagnostic": endpoint_diagnostic,
                "blockers": trajectory_blockers,
            }
        )
        if trajectory_blockers:
            break

    return {
        **fixture.as_public_record(),
        "status": "pass" if not blockers else "fail",
        "p2_certificate": p2_certificate.report(),
        "initial_exact_audit": initial_audit,
        "initial_exact_diagnostic": initial_diagnostic,
        "dhat": dhat,
        "refined_vertex_count": len(refinement.vertices),
        "refined_face_count": len(refinement.faces),
        "trajectories": trajectories,
        "blockers": blockers,
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    constructor, auditor = e14.build_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-p9-public-") as directory:
        hairpin = _run_hairpin(constructor, auditor, Path(directory))
    blockers = [] if hairpin["status"] == "pass" else ["hairpin"]
    elapsed = time.monotonic() - started
    if elapsed > MAX_SECONDS:
        blockers.append("wall_time_ceiling")
    return {
        "schema_version": REPORT_SCHEMA,
        "correctness_id": CORRECTNESS_ID,
        "gate": "public_complete_candidate_contribution_certificate",
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
        "schema_version": "post_v1_p9_public_failure_report.v1",
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
        raise FileExistsError(f"immutable P9 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-p9-supervisor-") as directory:
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
