"""Run the preregistered public P3 conservative collision-partition gate."""

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
import trimesh

import run_post_v1_e12_public_gate as e12
import run_post_v1_e14_public_gate as e14
from frayid.barrier_sliding_carrier import DHAT_BBOX_FRACTION, _native_target
from frayid.collision_partition import (
    CORRECTNESS_ID,
    collision_candidate_summary,
    conservative_collision_partition,
)
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import (
    PUBLIC_FIDELITY_INPUT_SHA256,
    GenusCarrierFidelityFixture,
    public_genus_fidelity_fixtures,
)
from frayid.io import write_json
from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)
from frayid.shrinkwrap_carrier import (
    MAXIMUM_MOTION_PITCH,
    _deduplicate_source_faces,
    _fixed_neighbors,
    _fixed_vertex_normals,
    _unique_edges,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "post_v1_p3_public_collision_partition_gate.v1"
SEED = 20260831
ANALYTIC_QUERY_COUNT = 1024
TRAJECTORIES_PER_FIXTURE = 2
REPETITIONS = 2
MAXIMUM_HAIRPIN_CANDIDATE_RATIO = 0.25
MAXIMUM_HAIRPIN_PARTITION_SECONDS = 30.0
MAX_SECONDS = 30 * 60


def _timed_certificate_worker(
    sender: Any,
    combined: np.ndarray,
    combined_faces: np.ndarray,
    wrap_count: int,
    proposal: np.ndarray,
    dhat: float,
) -> None:
    import ipctk  # type: ignore[import-not-found]

    mesh = ipctk.CollisionMesh(
        np.asfortranarray(combined, dtype=np.float64),
        np.asfortranarray(_unique_edges(combined_faces), dtype=np.int32),
        np.asfortranarray(combined_faces, dtype=np.int32),
    )
    mesh.can_collide = ipctk.make_static_obstacle_filter(wrap_count)
    result = conservative_collision_partition(
        mesh,
        combined,
        proposal,
        dhat=dhat,
        verify_full_path=True,
    )
    accepted = combined + result.certified_fraction * (proposal - combined)
    sender.send(
        {
            "certificate": result.report(),
            "accepted_wrap": np.asarray(accepted[:wrap_count]),
        }
    )
    sender.close()


def _timed_certificate(
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
        target=_timed_certificate_worker,
        args=(sender, combined, combined_faces, wrap_count, proposal, dhat),
    )
    worker.start()
    sender.close()
    worker.join(MAXIMUM_HAIRPIN_PARTITION_SECONDS)
    elapsed = time.monotonic() - started
    if worker.is_alive():
        worker.terminate()
        worker.join(10)
        receiver.close()
        return None, elapsed
    payload = receiver.recv() if worker.exitcode == 0 and receiver.poll() else None
    receiver.close()
    return payload, elapsed


def _rotation(rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def _transform(
    values: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return np.asfortranarray(values @ rotation.T + translation, dtype=np.float64)


def _analytic_query(index: int, rng: np.random.Generator) -> dict[str, Any]:
    import ipctk

    primitive = "point_triangle" if index % 2 == 0 else "edge_edge"
    case_index = (index // 2) % 4
    case = ("crossing", "tangent", "separating", "zero_motion")[case_index]
    exponent = -6.0 + 12.0 * ((index // 8) % 64) / 63.0
    scale = 10.0**exponent
    gap = 0.2 * scale
    if primitive == "point_triangle":
        if (index // 512) % 2:
            triangle = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1e-8, 1e-4, 0.0]],
                dtype=np.float64,
            )
            point_xy = np.mean(triangle, axis=0)
        else:
            triangle = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            )
            point_xy = np.asarray([0.25, 0.25, 0.0])
        point = point_xy.copy()
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
            ],
            dtype=np.float64,
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

    rotation = _rotation(rng)
    translation = rng.uniform(-2.0, 2.0, size=3) * scale
    start = _transform(start, rotation, translation)
    proposal = _transform(proposal, rotation, translation)
    edges = np.asfortranarray(edges, dtype=np.int32)
    faces = np.asfortranarray(faces, dtype=np.int32)
    mesh = ipctk.CollisionMesh(start, edges, faces)
    dhat = (0.05 if case == "crossing" else 0.5) * scale
    certificate = conservative_collision_partition(
        mesh,
        start,
        proposal,
        dhat=dhat,
        verify_full_path=True,
    )
    analytic_safe = bool(analytic_toi is None or certificate.certified_fraction < analytic_toi)
    blockers = list(certificate.blockers)
    if not analytic_safe:
        blockers.append("analytic_time_of_impact")
    return {
        "index": index,
        "primitive": primitive,
        "case": case,
        "scale": scale,
        "analytic_time_of_impact": analytic_toi,
        "analytic_safe": analytic_safe,
        "certificate": certificate.report(),
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
        if value["case"] == "crossing" and value["certificate"]["near_candidate_count"] == 0
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
        "empty_near_set_crossing_count": len(empty_crossings),
        "false_safe_count": len(failed),
        "failed_query_indices": failed,
        "maximum_partition_seconds": max(
            value["certificate"]["partition_elapsed_seconds"] for value in queries
        ),
        "elapsed_seconds": time.monotonic() - started,
        "status": "pass" if not failed and empty_crossings else "fail",
        "blockers": [
            *("analytic_false_safe" for _ in failed[:1]),
            *("missing_empty_near_set_crossing" for _ in range(not bool(empty_crossings))),
        ],
    }


def _combined_problem(
    vertices: np.ndarray,
    faces: np.ndarray,
    fixture: GenusCarrierFidelityFixture,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    import ipctk

    wrap_count = len(vertices)
    combined = np.asfortranarray(np.vstack((vertices, fixture.source_vertices)), dtype=np.float64)
    combined_faces = np.vstack((faces, fixture.source_faces + wrap_count))
    mesh = ipctk.CollisionMesh(
        combined,
        np.asfortranarray(_unique_edges(combined_faces), dtype=np.int32),
        np.asfortranarray(combined_faces, dtype=np.int32),
    )
    mesh.can_collide = ipctk.make_static_obstacle_filter(wrap_count)
    return mesh, combined, np.asarray(combined_faces), np.asarray(fixture.source_vertices)


def _trajectory_proposals(
    vertices: np.ndarray,
    faces: np.ndarray,
    fixture: GenusCarrierFidelityFixture,
) -> tuple[tuple[str, np.ndarray], ...]:
    attraction = trimesh.Trimesh(
        vertices=fixture.source_vertices,
        faces=_deduplicate_source_faces(fixture.source_faces),
        process=False,
    )
    target, _, _ = _native_target(
        vertices,
        faces,
        attraction,
        _fixed_neighbors(len(vertices), faces),
        pitch=fixture.pitch,
    )
    normals = _fixed_vertex_normals(vertices, faces)
    axis = np.asarray((0.3713906763541037, 0.5570860145311556, 0.7427813527082074))
    tangents = np.cross(normals, axis)
    lengths = np.linalg.norm(tangents, axis=1)
    fallback = lengths <= 1e-12
    tangents[fallback] = np.cross(normals[fallback], np.asarray((0.0, 1.0, 0.0)))
    lengths = np.linalg.norm(tangents, axis=1)
    tangents /= np.maximum(lengths, 1e-300)[:, None]
    tangential = vertices + MAXIMUM_MOTION_PITCH * fixture.pitch * tangents
    return (("native_pressure", target), ("tangential_sliding", tangential))


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
    certificate = certify_exact_dyadic_refinement(
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
    if certificate.status != "pass":
        blockers.append("p2_exact_refinement_certificate")
    if initial_audit.get("status") != "pass":
        blockers.append("initial_exact_audit")
    if blockers:
        return {
            **fixture.as_public_record(),
            "status": "fail",
            "p2_certificate": certificate.report(),
            "initial_exact_audit": initial_audit,
            "diagnostic": initial_diagnostic,
            "blockers": blockers,
        }

    mesh, combined, combined_faces, source_vertices = _combined_problem(
        refinement.vertices,
        refinement.faces,
        fixture,
    )
    diagonal = float(np.linalg.norm(np.ptp(combined, axis=0)))
    dhat = DHAT_BBOX_FRACTION * diagonal
    trajectory_reports: list[dict[str, Any]] = []
    for trajectory_name, wrap_proposal in _trajectory_proposals(
        refinement.vertices,
        refinement.faces,
        fixture,
    ):
        proposal = np.asfortranarray(np.vstack((wrap_proposal, source_vertices)), dtype=np.float64)
        preflight_summaries = [
            collision_candidate_summary(mesh, combined, proposal, dhat=dhat)
            for _ in range(REPETITIONS)
        ]
        preflight_deterministic = bool(
            preflight_summaries[0].near_candidate_keys == preflight_summaries[1].near_candidate_keys
            and preflight_summaries[0].near_candidate_count
            == preflight_summaries[1].near_candidate_count
            and preflight_summaries[0].full_swept_candidate_count
            == preflight_summaries[1].full_swept_candidate_count
            and preflight_summaries[0].near_to_full_swept_candidate_ratio
            == preflight_summaries[1].near_to_full_swept_candidate_ratio
        )
        preflight_blockers: list[str] = []
        if not preflight_deterministic:
            preflight_blockers.append("nondeterministic_candidate_partition")
        if fixture.name == "near_contact_hairpin":
            for repetition, summary in enumerate(preflight_summaries):
                if summary.near_to_full_swept_candidate_ratio > MAXIMUM_HAIRPIN_CANDIDATE_RATIO:
                    preflight_blockers.append(f"repetition_{repetition}:hairpin_candidate_ratio")
        if preflight_blockers:
            blockers.extend(f"{trajectory_name}:{value}" for value in preflight_blockers)
            trajectory_reports.append(
                {
                    "name": trajectory_name,
                    "candidate_preflight": [value.report() for value in preflight_summaries],
                    "deterministic_candidate_partition": preflight_deterministic,
                    "repetitions": [],
                    "status": "fail",
                    "blockers": preflight_blockers,
                }
            )
            continue
        repetitions: list[dict[str, Any]] = []
        accepted_arrays: list[np.ndarray] = []
        certificate_reports: list[dict[str, Any]] = []
        for repetition in range(REPETITIONS):
            if fixture.name == "near_contact_hairpin":
                payload, supervised_elapsed = _timed_certificate(
                    combined,
                    combined_faces,
                    len(refinement.vertices),
                    proposal,
                    dhat,
                )
                if payload is None:
                    repetitions.append(
                        {
                            "repetition": repetition,
                            "status": "fail",
                            "supervised_elapsed_seconds": supervised_elapsed,
                            "blockers": ["hairpin_partition_time"],
                        }
                    )
                    break
                certificate_report = payload["certificate"]
                accepted_wrap = np.asarray(payload["accepted_wrap"])
            else:
                result = conservative_collision_partition(
                    mesh,
                    combined,
                    proposal,
                    dhat=dhat,
                    verify_full_path=True,
                )
                certificate_report = result.report()
                accepted_combined = combined + result.certified_fraction * (proposal - combined)
                accepted_wrap = np.asarray(accepted_combined[: len(refinement.vertices)])
            certificate_reports.append(certificate_report)
            accepted_arrays.append(accepted_wrap)
            endpoint_path = fixture_root / f"{trajectory_name}_{repetition}.e10mesh"
            e12._write_mesh(endpoint_path, accepted_wrap, refinement.faces)
            endpoint_audit, endpoint_diagnostic = e14._audit(
                auditor,
                source_path,
                endpoint_path,
                fixture_root / f"{trajectory_name}_{repetition}_audit.json",
            )
            repetition_blockers = list(certificate_report["blockers"])
            if endpoint_audit.get("status") != "pass":
                repetition_blockers.append("independent_endpoint_exact_audit")
            repetitions.append(
                {
                    "repetition": repetition,
                    "certificate": certificate_report,
                    "independent_endpoint_exact_audit": endpoint_audit,
                    "exact_diagnostic": endpoint_diagnostic,
                    "status": "pass" if not repetition_blockers else "fail",
                    "blockers": repetition_blockers,
                }
            )
        deterministic = (
            bool(
                certificate_reports[0]["near_candidate_keys_sha256"]
                == certificate_reports[1]["near_candidate_keys_sha256"]
                and certificate_reports[0]["certified_fraction"]
                == certificate_reports[1]["certified_fraction"]
                and certificate_reports[0]["near_fraction"]
                == certificate_reports[1]["near_fraction"]
                and certificate_reports[0]["far_fraction"] == certificate_reports[1]["far_fraction"]
                and np.array_equal(accepted_arrays[0], accepted_arrays[1])
            )
            if len(certificate_reports) == 2
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
        if fixture.name == "near_contact_hairpin":
            for repetition, report in enumerate(certificate_reports):
                if report["partition_elapsed_seconds"] > MAXIMUM_HAIRPIN_PARTITION_SECONDS:
                    trajectory_blockers.append(f"repetition_{repetition}:hairpin_partition_time")
        blockers.extend(f"{trajectory_name}:{value}" for value in trajectory_blockers)
        trajectory_reports.append(
            {
                "name": trajectory_name,
                "candidate_preflight": [value.report() for value in preflight_summaries],
                "deterministic_candidate_partition": preflight_deterministic,
                "repetitions": repetitions,
                "deterministic_certificate_and_endpoint": deterministic,
                "status": "pass" if not trajectory_blockers else "fail",
                "blockers": trajectory_blockers,
            }
        )
    return {
        **fixture.as_public_record(),
        "status": "pass" if not blockers else "fail",
        "p2_certificate": certificate.report(),
        "initial_exact_audit": initial_audit,
        "initial_exact_diagnostic": initial_diagnostic,
        "dhat": dhat,
        "refined_vertex_count": len(refinement.vertices),
        "refined_face_count": len(refinement.faces),
        "trajectories": trajectory_reports,
        "blockers": blockers,
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    analytic = _analytic_corpus()
    fixtures: list[dict[str, Any]] = []
    if analytic["status"] == "pass":
        constructor, auditor = e14.build_tools()
        with tempfile.TemporaryDirectory(prefix="frayid-p3-public-") as directory:
            root = Path(directory)
            ordered_fixtures = sorted(
                public_genus_fidelity_fixtures(),
                key=lambda value: value.name != "near_contact_hairpin",
            )
            for fixture in ordered_fixtures:
                fixture_result = _run_mesh_fixture(
                    fixture,
                    constructor=constructor,
                    auditor=auditor,
                    root=root,
                )
                fixtures.append(fixture_result)
                if fixture_result["status"] != "pass":
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
        "gate": "public_conservative_collision_partition_correctness_and_complexity",
        "status": "pass" if not blockers else "fail",
        "registered_revision": "61b1b0c",
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
        raise FileExistsError(f"immutable P3 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-p3-supervisor-") as directory:
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
            report = {
                "schema_version": "post_v1_p3_public_failure_report.v1",
                "correctness_id": CORRECTNESS_ID,
                "status": "fail",
                "failure_class": "wall_time_ceiling_exceeded",
                "wall_time_ceiling_seconds": MAX_SECONDS,
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry_count": 0,
                "partial_results_promoted": False,
                "blockers": ["wall_time_ceiling_exceeded"],
                "execution_counters": {
                    "private_input_reads": 0,
                    "image_loads": 0,
                    "optimizer_steps": 0,
                    "development_evidence_reads": 0,
                    "modal_invocations": 0,
                    "sealed_test_accesses": 0,
                },
            }
        elif worker.exitcode != 0 or not worker_report.is_file():
            report = {
                "schema_version": "post_v1_p3_public_failure_report.v1",
                "correctness_id": CORRECTNESS_ID,
                "status": "fail",
                "failure_class": "worker_failure",
                "worker_exitcode": worker.exitcode,
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry_count": 0,
                "partial_results_promoted": False,
                "blockers": ["worker_failure"],
                "execution_counters": {
                    "private_input_reads": 0,
                    "image_loads": 0,
                    "optimizer_steps": 0,
                    "development_evidence_reads": 0,
                    "modal_invocations": 0,
                    "sealed_test_accesses": 0,
                },
            }
        else:
            report = json.loads(worker_report.read_text())
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
