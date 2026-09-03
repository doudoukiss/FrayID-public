"""Run the preregistered public P6 Planar-DAT mesh-certificate gate."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import ipctk  # type: ignore[import-not-found]
import numpy as np

import run_post_v1_e12_public_gate as e12
import run_post_v1_e14_public_gate as e14
import run_post_v1_p3_public_gate as p3
import run_post_v1_p4_public_gate as p4
from frayid.barrier_sliding_carrier import DHAT_BBOX_FRACTION
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import (
    PUBLIC_FIDELITY_INPUT_SHA256,
    GenusCarrierFidelityFixture,
    public_genus_fidelity_fixtures,
)
from frayid.io import write_json
from frayid.normalized_ti_oracle import normalized_ti_path_oracle
from frayid.planar_dat_certificate import planar_dat_path_certificate
from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)
from frayid.shrinkwrap_carrier import _unique_edges

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS_ID = "postv1_p6_planar_dat_mesh_certificate_r01"
REPORT_SCHEMA = "post_v1_p6_public_planar_dat_mesh_gate.v1"
REPETITIONS = 2
MINIMUM_TANGENTIAL_RETENTION = 0.25
MAXIMUM_HAIRPIN_MECHANISM_SECONDS = 30.0
MAXIMUM_HAIRPIN_ORACLE_SECONDS = 120.0
MAX_SECONDS = 30 * 60


def _oracle_worker(
    sender: Any,
    combined: np.ndarray,
    combined_faces: np.ndarray,
    wrap_count: int,
    accepted: np.ndarray,
) -> None:
    mesh = ipctk.CollisionMesh(
        np.asfortranarray(combined, dtype=np.float64),
        np.asfortranarray(_unique_edges(combined_faces), dtype=np.int32),
        np.asfortranarray(combined_faces, dtype=np.int32),
    )
    mesh.can_collide = ipctk.make_static_obstacle_filter(wrap_count)
    result = normalized_ti_path_oracle(mesh, combined, accepted)
    sender.send(result.report())
    sender.close()


def _timed_oracle(
    combined: np.ndarray,
    combined_faces: np.ndarray,
    wrap_count: int,
    accepted: np.ndarray,
) -> tuple[dict[str, Any] | None, float]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    started = time.monotonic()
    worker = context.Process(
        target=_oracle_worker,
        args=(sender, combined, combined_faces, wrap_count, accepted),
    )
    worker.start()
    sender.close()
    worker.join(MAXIMUM_HAIRPIN_ORACLE_SECONDS)
    elapsed = time.monotonic() - started
    if worker.is_alive():
        worker.terminate()
        worker.join(10)
        receiver.close()
        return None, elapsed
    report = receiver.recv() if worker.exitcode == 0 and receiver.poll() else None
    receiver.close()
    return report, elapsed


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
        payload, supervised = p4._timed_mechanism(
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
    result = planar_dat_path_certificate(
        mesh,
        combined,
        proposal,
        dhat=dhat,
        verify_full_path=False,
    )
    return result.report(), result.accepted_vertices, None


def _oracle(
    fixture_name: str,
    mesh: Any,
    combined: np.ndarray,
    combined_faces: np.ndarray,
    wrap_count: int,
    accepted: np.ndarray,
) -> tuple[dict[str, Any] | None, float | None]:
    if fixture_name == "near_contact_hairpin":
        return _timed_oracle(combined, combined_faces, wrap_count, accepted)
    return normalized_ti_path_oracle(mesh, combined, accepted).report(), None


def _run_mesh_fixture(
    fixture: GenusCarrierFidelityFixture,
    *,
    constructor: Path,
    auditor: Path,
    root: Path,
) -> dict[str, Any]:
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
            "diagnostic": initial_diagnostic,
            "blockers": blockers,
        }

    mesh, combined, combined_faces, source_vertices = p3._combined_problem(
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
        repetitions: list[dict[str, Any]] = []
        mechanism_reports: list[dict[str, Any]] = []
        oracle_reports: list[dict[str, Any]] = []
        accepted_arrays: list[np.ndarray] = []
        for repetition in range(REPETITIONS):
            mechanism_report, accepted, supervised_mechanism = _mechanism(
                fixture.name,
                mesh,
                combined,
                combined_faces,
                len(refinement.vertices),
                proposal,
                dhat,
            )
            if mechanism_report is None or accepted is None:
                repetitions.append(
                    {
                        "repetition": repetition,
                        "status": "fail",
                        "supervised_mechanism_seconds": supervised_mechanism,
                        "blockers": ["hairpin_mechanism_time"],
                    }
                )
                break
            oracle_report, supervised_oracle = _oracle(
                fixture.name,
                mesh,
                combined,
                combined_faces,
                len(refinement.vertices),
                accepted,
            )
            if oracle_report is None:
                repetitions.append(
                    {
                        "repetition": repetition,
                        "mechanism": mechanism_report,
                        "status": "fail",
                        "supervised_mechanism_seconds": supervised_mechanism,
                        "supervised_oracle_seconds": supervised_oracle,
                        "blockers": ["hairpin_normalized_oracle_time"],
                    }
                )
                break
            endpoint_path = fixture_root / f"{name}_{repetition}.e10mesh"
            e12._write_mesh(endpoint_path, accepted[: len(refinement.vertices)], refinement.faces)
            endpoint_audit, endpoint_diagnostic = e14._audit(
                auditor,
                source_path,
                endpoint_path,
                fixture_root / f"{name}_{repetition}_audit.json",
            )
            repetition_blockers = list(mechanism_report["blockers"])
            if oracle_report["status"] != "pass" or not oracle_report["collision_free"]:
                repetition_blockers.append("normalized_full_path_oracle")
            if endpoint_audit.get("status") != "pass":
                repetition_blockers.append("independent_endpoint_exact_audit")
            if name == "tangential_sliding" and (
                mechanism_report["retained_displacement_ratio"] < MINIMUM_TANGENTIAL_RETENTION
            ):
                repetition_blockers.append("tangential_motion_retention")
            if name == "native_pressure" and mechanism_report["filtered_displacement_norm"] <= 0.0:
                repetition_blockers.append("native_motion_retention")
            if (
                fixture.name == "near_contact_hairpin"
                and mechanism_report["mechanism_elapsed_seconds"]
                > MAXIMUM_HAIRPIN_MECHANISM_SECONDS
            ):
                repetition_blockers.append("hairpin_mechanism_time")
            if (
                fixture.name == "near_contact_hairpin"
                and oracle_report["elapsed_seconds"] > MAXIMUM_HAIRPIN_ORACLE_SECONDS
            ):
                repetition_blockers.append("hairpin_normalized_oracle_time")
            mechanism_reports.append(mechanism_report)
            oracle_reports.append(oracle_report)
            accepted_arrays.append(accepted)
            repetitions.append(
                {
                    "repetition": repetition,
                    "mechanism": mechanism_report,
                    "normalized_oracle": oracle_report,
                    "supervised_mechanism_seconds": supervised_mechanism,
                    "supervised_oracle_seconds": supervised_oracle,
                    "independent_endpoint_exact_audit": endpoint_audit,
                    "exact_diagnostic": endpoint_diagnostic,
                    "status": "pass" if not repetition_blockers else "fail",
                    "blockers": repetition_blockers,
                }
            )
        deterministic = (
            bool(
                mechanism_reports[0]["candidate_keys_sha256"]
                == mechanism_reports[1]["candidate_keys_sha256"]
                and mechanism_reports[0]["trust_region_centers_sha256"]
                == mechanism_reports[1]["trust_region_centers_sha256"]
                and mechanism_reports[0]["trust_region_radii_sha256"]
                == mechanism_reports[1]["trust_region_radii_sha256"]
                and mechanism_reports[0]["filtered_displacements_sha256"]
                == mechanism_reports[1]["filtered_displacements_sha256"]
                and oracle_reports[0]["collision_free"] == oracle_reports[1]["collision_free"]
                and oracle_reports[0]["normalized_trajectory"]
                == oracle_reports[1]["normalized_trajectory"]
                and np.array_equal(accepted_arrays[0], accepted_arrays[1])
            )
            if len(mechanism_reports) == 2 and len(oracle_reports) == 2
            else None
        )
        trajectory_blockers = [
            *(
                f"repetition_{value['repetition']}"
                for value in repetitions
                if value["status"] != "pass"
            ),
            *("nondeterministic_certificate_or_endpoint" for _ in range(deterministic is False)),
        ]
        blockers.extend(f"{name}:{value}" for value in trajectory_blockers)
        trajectories.append(
            {
                "name": name,
                "repetitions": repetitions,
                "deterministic_certificate_and_endpoint": deterministic,
                "status": "pass" if not trajectory_blockers else "fail",
                "blockers": trajectory_blockers,
            }
        )
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
    fixtures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="frayid-p6-public-") as directory:
        ordered = sorted(
            public_genus_fidelity_fixtures(),
            key=lambda value: value.name != "near_contact_hairpin",
        )
        for fixture in ordered:
            result = _run_mesh_fixture(
                fixture,
                constructor=constructor,
                auditor=auditor,
                root=Path(directory),
            )
            fixtures.append(result)
            if result["status"] != "pass":
                break
    blockers = [f"fixture:{value['name']}" for value in fixtures if value["status"] != "pass"]
    if len(fixtures) != 8:
        blockers.append("incomplete_mesh_fixture_set")
    elapsed = time.monotonic() - started
    if elapsed > MAX_SECONDS:
        blockers.append("wall_time_ceiling")
    return {
        "schema_version": REPORT_SCHEMA,
        "correctness_id": CORRECTNESS_ID,
        "gate": "public_planar_dat_normalized_mesh_path_certificate",
        "status": "pass" if not blockers else "fail",
        "registered_revision": "3d74eba",
        "scope": "public_procedural_geometry_only",
        "inherited_e11_fixture_definition_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
        "fixtures": fixtures,
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
        "schema_version": "post_v1_p6_public_failure_report.v1",
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
        raise FileExistsError(f"immutable P6 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-p6-supervisor-") as directory:
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
