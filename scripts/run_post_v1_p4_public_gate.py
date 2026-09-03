"""Run the preregistered public P4 Planar-DAT path-certificate gate."""

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
from frayid.barrier_sliding_carrier import DHAT_BBOX_FRACTION
from frayid.collision_partition import NEAR_INFLATION_FRACTION
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import (
    PUBLIC_FIDELITY_INPUT_SHA256,
    GenusCarrierFidelityFixture,
    public_genus_fidelity_fixtures,
)
from frayid.io import write_json
from frayid.planar_dat_certificate import (
    CORRECTNESS_ID,
    TIGHT_INCLUSION_CONSERVATIVE_RESCALING,
    TIGHT_INCLUSION_MAX_ITERATIONS,
    TIGHT_INCLUSION_TOLERANCE,
    planar_dat_path_certificate,
)
from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)
from frayid.shrinkwrap_carrier import _unique_edges

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "post_v1_p4_public_planar_dat_gate.v1"
SEED = 20260831
ANALYTIC_QUERY_COUNT = 1024
REPETITIONS = 2
MINIMUM_TANGENTIAL_RETENTION = 0.25
MAXIMUM_HAIRPIN_MECHANISM_SECONDS = 30.0
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
    result = planar_dat_path_certificate(
        mesh,
        combined,
        proposal,
        dhat=dhat,
        verify_full_path=False,
    )
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


def _full_path_oracle(
    mesh: Any,
    start: np.ndarray,
    accepted: np.ndarray,
) -> tuple[bool, int, float]:
    started = time.monotonic()
    candidates = ipctk.Candidates()
    candidates.build(mesh, start, accepted, 0.0, ipctk.SweepAndPrune())
    safe = bool(
        candidates.is_step_collision_free(
            mesh,
            start,
            accepted,
            0.0,
            ipctk.TightInclusionCCD(
                TIGHT_INCLUSION_TOLERANCE,
                TIGHT_INCLUSION_MAX_ITERATIONS,
                TIGHT_INCLUSION_CONSERVATIVE_RESCALING,
            ),
        )
    )
    return safe, len(candidates), time.monotonic() - started


def _analytic_geometry(
    index: int,
    rng: np.random.Generator,
) -> tuple[str, str, float, float | None, Any, np.ndarray, np.ndarray, float]:
    primitive = "point_triangle" if index % 2 == 0 else "edge_edge"
    case_index = (index // 2) % 4
    case = ("crossing", "tangent", "separating", "zero_motion")[case_index]
    exponent = -6.0 + 12.0 * ((index // 8) % 64) / 63.0
    scale = 10.0**exponent
    gap = 0.2 * scale
    if primitive == "point_triangle":
        if (index // 512) % 2:
            triangle = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1e-8, 1e-4, 0.0]])
            point = np.mean(triangle, axis=0)
        else:
            triangle = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            point = np.asarray([0.25, 0.25, 0.0])
        point[2] = gap / scale
        start = np.vstack((point, triangle)) * scale
        proposal = start.copy()
        if case == "crossing":
            proposal[0, 2] = -gap
            analytic_toi: float | None = 0.5
        elif case == "tangent":
            proposal[0, 2] = 0.0
            analytic_toi = 1.0
        elif case == "separating":
            proposal[0, 2] = 2.0 * gap
            analytic_toi = None
        else:
            analytic_toi = None
        edges = np.asarray([[1, 2], [2, 3], [1, 3]], dtype=np.int32)
        faces = np.asarray([[1, 2, 3]], dtype=np.int32)
    else:
        start = np.asarray(
            [
                [-0.5 * scale, 0.0, 0.0],
                [0.5 * scale, 0.0, 0.0],
                [0.0, -0.5 * scale, gap],
                [0.0, 0.5 * scale, gap],
            ]
        )
        proposal = start.copy()
        if case == "crossing":
            proposal[2:, 2] = -gap
            analytic_toi = 0.5
        elif case == "tangent":
            proposal[2:, 2] = 0.0
            analytic_toi = 1.0
        elif case == "separating":
            proposal[2:, 2] = 2.0 * gap
            analytic_toi = None
        else:
            analytic_toi = None
        edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        faces = np.empty((0, 3), dtype=np.int32)
    rotation = p3._rotation(rng)
    translation = rng.uniform(-2.0, 2.0, size=3) * scale
    start = p3._transform(start, rotation, translation)
    proposal = p3._transform(proposal, rotation, translation)
    mesh = ipctk.CollisionMesh(
        start,
        np.asfortranarray(edges, dtype=np.int32),
        np.asfortranarray(faces, dtype=np.int32),
    )
    dhat = (0.05 if case == "crossing" else 0.5) * scale
    return primitive, case, scale, analytic_toi, mesh, start, proposal, dhat


def _analytic_query(index: int, rng: np.random.Generator) -> dict[str, Any]:
    primitive, case, scale, analytic_toi, mesh, start, proposal, dhat = _analytic_geometry(
        index, rng
    )
    p3_near = ipctk.Candidates()
    p3_near.build(
        mesh,
        start,
        NEAR_INFLATION_FRACTION * dhat,
        ipctk.SweepAndPrune(),
    )
    result = planar_dat_path_certificate(mesh, start, proposal, dhat=dhat)
    blockers = list(result.blockers)
    if result.full_oracle_safe is not True:
        blockers.append("analytic_false_safe")
    return {
        "index": index,
        "primitive": primitive,
        "case": case,
        "scale": scale,
        "analytic_time_of_impact": analytic_toi,
        "p3_static_near_candidate_count": len(p3_near),
        "certificate": result.report(),
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


def _analytic_corpus() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    started = time.monotonic()
    queries = [_analytic_query(index, rng) for index in range(ANALYTIC_QUERY_COUNT)]
    failed = [value["index"] for value in queries if value["status"] != "pass"]
    empty_crossings = [
        value
        for value in queries
        if value["case"] == "crossing" and value["p3_static_near_candidate_count"] == 0
    ]
    return {
        "seed": SEED,
        "query_count": len(queries),
        "primitive_counts": {
            name: sum(value["primitive"] == name for value in queries)
            for name in ("point_triangle", "edge_edge")
        },
        "case_counts": {
            name: sum(value["case"] == name for value in queries)
            for name in ("crossing", "tangent", "separating", "zero_motion")
        },
        "minimum_scale": min(value["scale"] for value in queries),
        "maximum_scale": max(value["scale"] for value in queries),
        "empty_p3_near_set_crossing_count": len(empty_crossings),
        "false_safe_count": len(failed),
        "failed_query_indices": failed,
        "maximum_mechanism_seconds": max(
            value["certificate"]["mechanism_elapsed_seconds"] for value in queries
        ),
        "elapsed_seconds": time.monotonic() - started,
        "status": "pass" if not failed and empty_crossings else "fail",
        "blockers": [
            *("analytic_false_safe" for _ in failed[:1]),
            *("missing_empty_p3_near_set_crossing" for _ in range(not bool(empty_crossings))),
        ],
    }


def _complete_hairpin_certificate(
    mesh: Any,
    combined: np.ndarray,
    combined_faces: np.ndarray,
    wrap_count: int,
    proposal: np.ndarray,
    dhat: float,
) -> tuple[dict[str, Any] | None, np.ndarray | None, float]:
    payload, supervised_elapsed = _timed_mechanism(
        combined,
        combined_faces,
        wrap_count,
        proposal,
        dhat,
    )
    if payload is None:
        return None, None, supervised_elapsed
    accepted = np.asfortranarray(payload["accepted_vertices"], dtype=np.float64)
    safe, candidate_count, oracle_elapsed = _full_path_oracle(mesh, combined, accepted)
    report = payload["certificate"]
    report["full_swept_candidate_count"] = candidate_count
    report["full_oracle_safe"] = safe
    report["oracle_elapsed_seconds"] = oracle_elapsed
    report["elapsed_seconds"] += oracle_elapsed
    if not safe:
        report["blockers"].append("complete_dynamic_tight_inclusion_oracle")
        report["status"] = "fail"
    return report, accepted, supervised_elapsed


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
        reports: list[dict[str, Any]] = []
        accepted_arrays: list[np.ndarray] = []
        for repetition in range(REPETITIONS):
            supervised_elapsed: float | None = None
            if fixture.name == "near_contact_hairpin":
                report, accepted, supervised_elapsed = _complete_hairpin_certificate(
                    mesh,
                    combined,
                    combined_faces,
                    len(refinement.vertices),
                    proposal,
                    dhat,
                )
                if report is None or accepted is None:
                    repetitions.append(
                        {
                            "repetition": repetition,
                            "status": "fail",
                            "supervised_mechanism_seconds": supervised_elapsed,
                            "blockers": ["hairpin_mechanism_time"],
                        }
                    )
                    break
            else:
                result = planar_dat_path_certificate(mesh, combined, proposal, dhat=dhat)
                report = result.report()
                accepted = result.accepted_vertices
            endpoint_path = fixture_root / f"{name}_{repetition}.e10mesh"
            e12._write_mesh(endpoint_path, accepted[: len(refinement.vertices)], refinement.faces)
            endpoint_audit, endpoint_diagnostic = e14._audit(
                auditor,
                source_path,
                endpoint_path,
                fixture_root / f"{name}_{repetition}_audit.json",
            )
            repetition_blockers = list(report["blockers"])
            if endpoint_audit.get("status") != "pass":
                repetition_blockers.append("independent_endpoint_exact_audit")
            if name == "tangential_sliding" and (
                report["retained_displacement_ratio"] < MINIMUM_TANGENTIAL_RETENTION
            ):
                repetition_blockers.append("tangential_motion_retention")
            if name == "native_pressure" and report["filtered_displacement_norm"] <= 0.0:
                repetition_blockers.append("native_motion_retention")
            if (
                fixture.name == "near_contact_hairpin"
                and report["mechanism_elapsed_seconds"] > MAXIMUM_HAIRPIN_MECHANISM_SECONDS
            ):
                repetition_blockers.append("hairpin_mechanism_time")
            reports.append(report)
            accepted_arrays.append(accepted)
            repetitions.append(
                {
                    "repetition": repetition,
                    "certificate": report,
                    "supervised_mechanism_seconds": supervised_elapsed,
                    "independent_endpoint_exact_audit": endpoint_audit,
                    "exact_diagnostic": endpoint_diagnostic,
                    "status": "pass" if not repetition_blockers else "fail",
                    "blockers": repetition_blockers,
                }
            )
        deterministic = (
            bool(
                reports[0]["candidate_keys_sha256"] == reports[1]["candidate_keys_sha256"]
                and reports[0]["accepted_vertices_sha256"] == reports[1]["accepted_vertices_sha256"]
                and reports[0]["filtered_displacements_sha256"]
                == reports[1]["filtered_displacements_sha256"]
                and reports[0]["trust_region_centers_sha256"]
                == reports[1]["trust_region_centers_sha256"]
                and reports[0]["trust_region_radii_sha256"]
                == reports[1]["trust_region_radii_sha256"]
                and reports[0]["full_oracle_safe"] == reports[1]["full_oracle_safe"]
                and np.array_equal(accepted_arrays[0], accepted_arrays[1])
            )
            if len(reports) == 2
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
    analytic = _analytic_corpus()
    fixtures: list[dict[str, Any]] = []
    if analytic["status"] == "pass":
        constructor, auditor = e14.build_tools()
        with tempfile.TemporaryDirectory(prefix="frayid-p4-public-") as directory:
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
    blockers = [
        *("analytic_primitive_corpus" for _ in range(analytic["status"] != "pass")),
        *(f"fixture:{value['name']}" for value in fixtures if value["status"] != "pass"),
    ]
    if analytic["status"] == "pass" and len(fixtures) != 8:
        blockers.append("incomplete_mesh_fixture_set")
    elapsed = time.monotonic() - started
    if elapsed > MAX_SECONDS:
        blockers.append("wall_time_ceiling")
    return {
        "schema_version": REPORT_SCHEMA,
        "correctness_id": CORRECTNESS_ID,
        "gate": "public_planar_dat_path_correctness_motion_and_complexity",
        "status": "pass" if not blockers else "fail",
        "registered_revision": "34bc820",
        "scope": "public_procedural_geometry_only",
        "seed": SEED,
        "inherited_e11_fixture_definition_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
        "analytic_corpus": analytic,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"immutable P4 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-p4-supervisor-") as directory:
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


def _failure_report(failure_class: str, started: float, exitcode: int | None) -> dict[str, Any]:
    return {
        "schema_version": "post_v1_p4_public_failure_report.v1",
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


if __name__ == "__main__":
    main()
