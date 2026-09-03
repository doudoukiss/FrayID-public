"""Run the preregistered public P8 batched Planar-DAT equivalence gate."""

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
import run_post_v1_p6_public_gate as p6
from frayid.barrier_sliding_carrier import DHAT_BBOX_FRACTION
from frayid.batched_planar_dat import batched_planar_dat_path
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import PUBLIC_FIDELITY_INPUT_SHA256, public_genus_fidelity_fixtures
from frayid.io import write_json
from frayid.planar_dat_certificate import planar_dat_path_certificate
from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)
from frayid.shrinkwrap_carrier import _unique_edges

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS_ID = "postv1_p8_deterministic_batched_planar_dat_r01"
REPORT_SCHEMA = "post_v1_p8_public_batched_planar_dat_gate.v1"
REGISTERED_REVISION = "b9b5534"
MAXIMUM_ABSOLUTE_DIFFERENCE = 1e-12
MINIMUM_TANGENTIAL_RETENTION = 0.25
MAXIMUM_UPSTREAM_SECONDS = 300.0
MAXIMUM_BATCHED_REDUCTION_SECONDS = 5.0
MAXIMUM_ORACLE_SECONDS = 120.0
MAX_SECONDS = 30 * 60


def _two_triangles(scale: float, reverse_faces: bool) -> tuple[Any, np.ndarray]:
    vertices = np.asfortranarray(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.2],
                [1.0, 0.0, 0.2],
                [0.0, 1.0, 0.2],
            ],
            dtype=np.float64,
        )
        * scale
    )
    faces_values = [[0, 1, 2], [3, 5, 4]]
    if reverse_faces:
        faces_values.reverse()
    faces = np.asfortranarray(faces_values, dtype=np.int32)
    edges = np.asfortranarray([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    return ipctk.CollisionMesh(vertices, edges, faces), vertices


def _mixed_stencil_corpus() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    blockers: list[str] = []
    motions = {
        "crossing": -0.4,
        "tangent": -0.2,
        "separating": 0.1,
        "zero": 0.0,
    }
    for scale in (1e-6, 1.0, 1e6):
        for reverse_faces in (False, True):
            mesh, start = _two_triangles(scale, reverse_faces)
            for motion_name, motion in motions.items():
                proposal = start.copy(order="F")
                proposal[3:, 2] += motion * scale
                upstream = planar_dat_path_certificate(
                    mesh,
                    start,
                    proposal,
                    dhat=0.05 * scale,
                    verify_full_path=False,
                )
                batched = batched_planar_dat_path(
                    mesh,
                    start,
                    proposal,
                    dhat=0.05 * scale,
                )
                difference = float(
                    np.max(
                        np.abs(upstream.filtered_displacements - batched.filtered_displacements),
                        initial=0.0,
                    )
                )
                case_blockers: list[str] = []
                if batched.status != "pass":
                    case_blockers.append("batched_filter")
                if upstream.candidate_keys != batched.candidate_keys:
                    case_blockers.append("candidate_identity")
                if difference > MAXIMUM_ABSOLUTE_DIFFERENCE:
                    case_blockers.append("upstream_equivalence")
                if upstream.restricted_vertex_count != batched.restricted_vertex_count:
                    case_blockers.append("restricted_vertex_decision")
                label = f"{scale:g}:{int(reverse_faces)}:{motion_name}"
                blockers.extend(f"{label}:{value}" for value in case_blockers)
                cases.append(
                    {
                        "label": label,
                        "scale": scale,
                        "face_order_reversed": reverse_faces,
                        "motion": motion_name,
                        "candidate_count": batched.candidate_count,
                        "edge_edge_count": batched.edge_edge_count,
                        "face_vertex_count": batched.face_vertex_count,
                        "maximum_absolute_difference": difference,
                        "status": "pass" if not case_blockers else "fail",
                        "blockers": case_blockers,
                    }
                )
    return {
        "case_count": len(cases),
        "cases": cases,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


def _upstream_worker(
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
    result = planar_dat_path_certificate(
        mesh,
        combined,
        proposal,
        dhat=dhat,
        verify_full_path=False,
    )
    sender.send(
        {
            "report": result.report(),
            "accepted_vertices": result.accepted_vertices,
            "filtered_displacements": result.filtered_displacements,
        }
    )
    sender.close()


def _timed_upstream(
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
        target=_upstream_worker,
        args=(sender, combined, combined_faces, wrap_count, proposal, dhat),
    )
    worker.start()
    sender.close()
    worker.join(MAXIMUM_UPSTREAM_SECONDS)
    elapsed = time.monotonic() - started
    if worker.is_alive():
        worker.terminate()
        worker.join(30)
        receiver.close()
        return None, elapsed
    payload = receiver.recv() if worker.exitcode == 0 and receiver.poll() else None
    receiver.close()
    return payload, elapsed


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
        trajectory_blockers: list[str] = []
        upstream_payload, upstream_seconds = _timed_upstream(
            combined,
            combined_faces,
            len(refinement.vertices),
            proposal,
            dhat,
        )
        if upstream_payload is None:
            trajectory_blockers.append("upstream_reference_time_or_worker")
            trajectories.append(
                {
                    "name": name,
                    "status": "fail",
                    "upstream_reference_seconds": upstream_seconds,
                    "blockers": trajectory_blockers,
                }
            )
            blockers.extend(f"{name}:{value}" for value in trajectory_blockers)
            break

        batched_results = [
            batched_planar_dat_path(mesh, combined, proposal, dhat=dhat) for _ in range(2)
        ]
        first, second = batched_results
        bitwise = bool(
            first.candidate_keys == second.candidate_keys
            and np.array_equal(first.candidate_ids, second.candidate_ids)
            and np.array_equal(first.candidate_kinds, second.candidate_kinds)
            and np.array_equal(first.truncation_ratios, second.truncation_ratios)
            and np.array_equal(first.filtered_displacements, second.filtered_displacements)
            and np.array_equal(first.accepted_vertices, second.accepted_vertices)
        )
        upstream_filtered = np.asarray(upstream_payload["filtered_displacements"], dtype=np.float64)
        maximum_difference = float(
            np.max(np.abs(first.filtered_displacements - upstream_filtered), initial=0.0)
        )
        upstream_report = upstream_payload["report"]
        if first.status != "pass" or second.status != "pass":
            trajectory_blockers.append("batched_filter")
        if not bitwise:
            trajectory_blockers.append("batched_nondeterminism")
        if first.report()["candidate_keys_sha256"] != upstream_report["candidate_keys_sha256"]:
            trajectory_blockers.append("candidate_identity")
        if maximum_difference > MAXIMUM_ABSOLUTE_DIFFERENCE:
            trajectory_blockers.append("upstream_equivalence")
        if any(
            value.reduction_seconds > MAXIMUM_BATCHED_REDUCTION_SECONDS for value in batched_results
        ):
            trajectory_blockers.append("batched_reduction_time")
        if name == "tangential_sliding" and (
            first.retained_displacement_ratio < MINIMUM_TANGENTIAL_RETENTION
        ):
            trajectory_blockers.append("tangential_motion_retention")
        if name == "native_pressure" and first.filtered_displacement_norm <= 0.0:
            trajectory_blockers.append("native_motion_retention")

        oracle_report: dict[str, Any] | None = None
        oracle_seconds: float | None = None
        endpoint_audit: dict[str, Any] | None = None
        endpoint_diagnostic = ""
        if not trajectory_blockers:
            oracle_report, oracle_seconds = p6._timed_oracle(
                combined,
                combined_faces,
                len(refinement.vertices),
                first.accepted_vertices,
            )
            if oracle_report is None or oracle_report["status"] != "pass":
                trajectory_blockers.append("normalized_full_path_oracle")
            elif oracle_report["elapsed_seconds"] > MAXIMUM_ORACLE_SECONDS:
                trajectory_blockers.append("normalized_oracle_time")
            endpoint_path = fixture_root / f"{name}_endpoint.e10mesh"
            e12._write_mesh(
                endpoint_path,
                first.accepted_vertices[: len(refinement.vertices)],
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
                "upstream_reference": upstream_report,
                "upstream_reference_seconds": upstream_seconds,
                "batched_repetitions": [value.report() for value in batched_results],
                "batched_bitwise_deterministic": bitwise,
                "maximum_absolute_upstream_difference": maximum_difference,
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
    mixed = _mixed_stencil_corpus()
    hairpin: dict[str, Any] | None = None
    blockers = [f"mixed:{value}" for value in mixed["blockers"]]
    if not blockers:
        constructor, auditor = e14.build_tools()
        with tempfile.TemporaryDirectory(prefix="frayid-p8-public-") as directory:
            hairpin = _run_hairpin(constructor, auditor, Path(directory))
        if hairpin["status"] != "pass":
            blockers.append("hairpin")
    elapsed = time.monotonic() - started
    if elapsed > MAX_SECONDS:
        blockers.append("wall_time_ceiling")
    return {
        "schema_version": REPORT_SCHEMA,
        "correctness_id": CORRECTNESS_ID,
        "gate": "public_deterministic_batched_planar_dat_equivalence",
        "status": "pass" if not blockers else "fail",
        "registered_revision": REGISTERED_REVISION,
        "scope": "public_procedural_geometry_only",
        "inherited_e11_fixture_definition_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
        "mixed_stencil_corpus": mixed,
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
        "schema_version": "post_v1_p8_public_failure_report.v1",
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
        raise FileExistsError(f"immutable P8 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-p8-supervisor-") as directory:
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
