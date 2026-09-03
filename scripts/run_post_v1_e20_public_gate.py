"""Run the registered public-only E20 composed orientation-preserving map gate."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_post_v1_e17_public_gate as e17
import run_post_v1_p3_public_gate as p3
from frayid.coarse_bilipschitz import FreudenthalLatticeV1, refine_surface_to_lattice
from frayid.composed_orientation_map import (
    apply_material_control_blocks,
    fit_and_certify_composed_orientation_step,
    run_composed_orientation_controls,
)
from frayid.genus_carrier import PUBLIC_FIDELITY_INPUT_SHA256
from frayid.io import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e20_composed_orientation_preserving_r01"
REPORT_SCHEMA = "post_v1_e20_public_composed_orientation_gate.v1"
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
BOUND_PADDING_DIAGONAL_FRACTION = 0.5


def _run_repetition(
    repetition: int,
    *,
    constructor: Path,
    auditor: Path,
    root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    repetition_root = root / f"repetition_{repetition}"
    repetition_root.mkdir(parents=True, exist_ok=False)
    output_root = artifact_root / f"repetition_{repetition}"
    output_root.mkdir(parents=True, exist_ok=False)
    fixture, carrier_vertices, carrier_faces, p2_report = e17._fixture_refinement(
        constructor, repetition_root
    )
    combined = np.vstack((carrier_vertices, fixture.source_vertices))
    lower = np.min(combined, axis=0)
    upper = np.max(combined, axis=0)
    padding = BOUND_PADDING_DIAGONAL_FRACTION * float(np.linalg.norm(upper - lower))
    lattice = FreudenthalLatticeV1.create(
        lower - padding, upper + padding, nodes_per_axis=NODES_PER_AXIS
    )
    if (
        lattice.vertices.shape[0] != CONTROL_COUNT
        or np.count_nonzero(~lattice.boundary_mask) != FREE_CONTROL_COUNT
    ):
        raise AssertionError("registered E20 control profile changed")
    refinement_started = time.monotonic()
    carrier_surface = refine_surface_to_lattice(
        lattice, carrier_vertices, carrier_faces, timeout_seconds=MAXIMUM_TOTAL_SECONDS
    )
    source_surface = refine_surface_to_lattice(
        lattice,
        fixture.source_vertices,
        fixture.source_faces,
        timeout_seconds=MAXIMUM_TOTAL_SECONDS,
    )
    refinement_elapsed = time.monotonic() - refinement_started
    reference_topology = e17._surface_topology(
        carrier_surface.reference_vertices, carrier_surface.faces
    )
    blockers: list[str] = []
    if reference_topology != {
        "watertight": True,
        "winding_consistent": True,
        "euler_number": 2,
        "component_count": 1,
    }:
        blockers.append("reference_refinement_topology")
    trajectories: list[dict[str, Any]] = []
    for name, target in p3._trajectory_proposals(carrier_vertices, carrier_faces, fixture):
        proposal = np.asarray(target - carrier_vertices, dtype=np.float64)
        step_started = time.monotonic()
        try:
            step = fit_and_certify_composed_orientation_step(
                lattice,
                carrier_vertices,
                carrier_faces,
                carrier_surface,
                proposal,
                block_count=BLOCK_COUNT,
                minimum_retained_displacement_ratio=(
                    MINIMUM_TANGENTIAL_RETENTION if name == "tangential_sliding" else 0.0
                ),
                timeout_seconds_per_block=MAXIMUM_CERTIFICATE_SECONDS_PER_BLOCK,
            )
        except TimeoutError as error:
            blockers.append(f"{name}:certificate_timeout")
            trajectories.append(
                {
                    "name": name,
                    "status": "fail",
                    "blockers": ["certificate_timeout"],
                    "diagnostic": str(error),
                }
            )
            break
        carrier_endpoint = step.final_refined_surface_vertices
        source_endpoint = apply_material_control_blocks(
            lattice, source_surface.reference_vertices, step.accepted_control_blocks
        )
        replay_endpoint = apply_material_control_blocks(
            lattice, carrier_surface.reference_vertices, step.accepted_control_blocks.copy()
        )
        topology = e17._surface_topology(carrier_endpoint, carrier_surface.faces)
        audit: dict[str, Any] = {}
        diagnostic = ""
        audit_elapsed = 0.0
        if step.status == "pass":
            try:
                audit, diagnostic, audit_elapsed = e17._exact_endpoint_audit(
                    auditor,
                    source_endpoint,
                    source_surface.faces,
                    carrier_endpoint,
                    carrier_surface.faces,
                    repetition_root,
                    name,
                )
            except subprocess.TimeoutExpired:
                audit = {"status": "fail", "blockers": ["endpoint_audit_timeout"]}
                audit_elapsed = MAXIMUM_ENDPOINT_AUDIT_SECONDS
        trajectory_blockers = list(step.blockers)
        if topology != {
            "watertight": True,
            "winding_consistent": True,
            "euler_number": 2,
            "component_count": 1,
        }:
            trajectory_blockers.append("endpoint_topology")
        if not np.array_equal(carrier_endpoint, replay_endpoint):
            trajectory_blockers.append("next_step_replay")
        if audit.get("status") != "pass":
            trajectory_blockers.append("independent_exact_endpoint_or_nesting")
        if audit_elapsed > MAXIMUM_ENDPOINT_AUDIT_SECONDS:
            trajectory_blockers.append("endpoint_audit_time")
        if name == "native_pressure" and step.retained_displacement_ratio <= 0.0:
            trajectory_blockers.append("native_motion_not_positive")
        blockers.extend(f"{name}:{value}" for value in trajectory_blockers)
        artifact_path = output_root / f"{name}_certified_endpoint.npz"
        e17._write_npz_immutable(
            artifact_path,
            lattice_reference_vertices=lattice.vertices,
            lattice_tetrahedra=lattice.tetrahedra,
            accepted_control_blocks=step.accepted_control_blocks,
            accepted_alphas=np.asarray(
                [block.accepted_alpha for block in step.blocks], dtype=np.float64
            ),
            final_lattice_vertices=step.final_lattice_vertices,
            carrier_reference_vertices=carrier_surface.reference_vertices,
            carrier_endpoint_vertices=carrier_endpoint,
            carrier_faces=carrier_surface.faces,
            carrier_parent_face_indices=carrier_surface.parent_face_indices,
            carrier_corner_barycentrics=carrier_surface.corner_barycentric_text,
            source_reference_vertices=source_surface.reference_vertices,
            source_endpoint_vertices=source_endpoint,
            source_faces=source_surface.faces,
            block_decision_sha256=np.asarray([block.decision_sha256 for block in step.blocks]),
            decision_sha256=np.asarray(step.decision_sha256),
        )
        trajectories.append(
            {
                "name": name,
                "status": "pass" if not trajectory_blockers else "fail",
                "step": step.report(),
                "topology": topology,
                "exact_endpoint_audit": audit,
                "exact_endpoint_diagnostic": diagnostic,
                "endpoint_audit_elapsed_seconds": audit_elapsed,
                "elapsed_seconds": time.monotonic() - step_started,
                "artifact": e17._report_path(artifact_path),
                "artifact_sha256": e17._sha256(artifact_path),
                "blockers": trajectory_blockers,
            }
        )
        if trajectory_blockers:
            break
    return {
        "repetition": repetition,
        "status": "pass" if not blockers else "fail",
        "p2_certificate": p2_report,
        "carrier_original_face_count": int(carrier_faces.shape[0]),
        "source_original_face_count": int(fixture.source_faces.shape[0]),
        "lattice_sha256": lattice.content_sha256(),
        "lattice_vertex_count": int(lattice.vertices.shape[0]),
        "lattice_tetrahedron_count": int(lattice.tetrahedra.shape[0]),
        "free_control_count": int(np.count_nonzero(~lattice.boundary_mask)),
        "block_count": BLOCK_COUNT,
        "carrier_refinement": carrier_surface.report(),
        "source_refinement": source_surface.report(),
        "reference_topology": reference_topology,
        "refinement_elapsed_seconds": refinement_elapsed,
        "trajectories": trajectories,
        "elapsed_seconds": time.monotonic() - started,
        "blockers": blockers,
    }


def run_public_gate(artifact_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_CORE_LIMIT))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_CORE_LIMIT))
    git = e17._git_binding()
    controls = run_composed_orientation_controls()
    blockers: list[str] = []
    if not git["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")
    if controls["status"] != "pass":
        blockers.append("composed_orientation_controls")
    repetitions: list[dict[str, Any]] = []
    if not blockers:
        constructor, auditor = e17._build_tools()
        with tempfile.TemporaryDirectory(prefix="frayid-e20-public-") as directory:
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
                PROJECT_ROOT / "src/frayid/composed_orientation_map.py"
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
        raise FileExistsError(f"immutable E20 report exists: {arguments.output}")
    if artifact_root.exists():
        raise FileExistsError(f"immutable E20 artifact directory exists: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-e20-supervisor-") as directory:
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
