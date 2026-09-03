"""Run the registered public-only E17 coarse bi-Lipschitz gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import run_post_v1_e12_public_gate as e12
import run_post_v1_e14_public_gate as e14
import run_post_v1_p3_public_gate as p3
from frayid.coarse_bilipschitz import (
    FreudenthalLatticeV1,
    fit_and_certify_bilipschitz_step,
    parent_area_path_report,
    refine_surface_to_lattice,
    run_bilipschitz_controls,
)
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import PUBLIC_FIDELITY_INPUT_SHA256, public_genus_fidelity_fixtures
from frayid.interface_field import write_interface_mesh
from frayid.io import write_json
from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e17_coarse_bilipschitz_r01"
REPORT_SCHEMA = "post_v1_e17_public_coarse_bilipschitz_gate.v1"
SEED = 20260831
REPETITIONS = 2
NODES_PER_AXIS = 8
CONTROL_COUNT = 512
FREE_CONTROL_COUNT = 216
KAPPA = 0.5
MINIMUM_TANGENTIAL_RETENTION = 0.25
MAXIMUM_TOTAL_SECONDS = 7_200.0
MAXIMUM_ENDPOINT_AUDIT_SECONDS = 120.0
MAXIMUM_MEMORY_GIB = 16.0
CPU_CORE_LIMIT = 8
BOUND_PADDING_DIAGONAL_FRACTION = 0.5
ALLOWED_UNTRACKED_PREFIX = "docs/0901/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _git_binding() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    records = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    disallowed = [
        record
        for record in records
        if not (record.startswith("?? ") and record[3:].startswith(ALLOWED_UNTRACKED_PREFIX))
    ]
    return {
        "revision": revision,
        "implementation_tree_clean": not disallowed,
        "allowed_untracked_advisory_prefix": ALLOWED_UNTRACKED_PREFIX,
        "disallowed_status_records": disallowed,
    }


def _build_tools() -> tuple[Path, Path]:
    return e14.build_tools()


def _write_npz_immutable(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"immutable E17 artifact exists: {path}")
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def _compact_surface(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    retained = np.unique(faces)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[retained] = np.arange(retained.size, dtype=np.int64)
    return vertices[retained], remap[faces]


def _exact_endpoint_audit(
    auditor: Path,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    carrier_vertices: np.ndarray,
    carrier_faces: np.ndarray,
    root: Path,
    name: str,
) -> tuple[dict[str, Any], str, float]:
    compact_source_vertices, compact_source_faces = _compact_surface(source_vertices, source_faces)
    compact_carrier_vertices, compact_carrier_faces = _compact_surface(
        carrier_vertices, carrier_faces
    )
    combined = np.vstack((compact_source_vertices, compact_carrier_vertices))
    lower = np.min(combined, axis=0)
    upper = np.max(combined, axis=0)
    padding = max(float(np.linalg.norm(upper - lower)) * 0.1, 1.0e-6)
    source_path = root / f"{name}_mapped_source.e6mesh"
    carrier_path = root / f"{name}_mapped_carrier.e10mesh"
    report_path = root / f"{name}_exact_audit.json"
    write_interface_mesh(
        source_path,
        compact_source_vertices,
        compact_source_faces,
        (lower - padding, upper + padding),
    )
    e12._write_mesh(carrier_path, compact_carrier_vertices, compact_carrier_faces)
    started = time.monotonic()
    completed = subprocess.run(
        [str(auditor), str(source_path), str(carrier_path), str(report_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAXIMUM_ENDPOINT_AUDIT_SECONDS,
    )
    elapsed = time.monotonic() - started
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    report["elapsed_seconds"] = elapsed
    report["returncode"] = completed.returncode
    return report, (completed.stdout + completed.stderr).strip(), elapsed


def _surface_topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return {
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "component_count": len(mesh.split(only_watertight=False)),
    }


def _fixture_refinement(
    constructor: Path, root: Path
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    fixture = next(
        value for value in public_genus_fidelity_fixtures() if value.name == "near_contact_hairpin"
    )
    source_path = root / "source.e6mesh"
    parent_path = root / "parent.e10mesh"
    e12._write_fixture(source_path, fixture)
    completed = subprocess.run(
        [str(constructor), str(source_path), str(parent_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAXIMUM_TOTAL_SECONDS,
    )
    if completed.returncode != 0 or not parent_path.is_file():
        raise RuntimeError(f"E14 parent construction failed: {completed.stdout}{completed.stderr}")
    parent_vertices, parent_faces = read_e10_mesh(parent_path)
    refinement = subdivide_with_exact_provenance(parent_vertices, parent_faces, rounds=2)
    certificate = certify_exact_dyadic_refinement(
        parent_vertices,
        parent_faces,
        refinement,
        parent_grid=e14._grid_for_fixture(fixture),
        rounds=2,
    )
    if certificate.status != "pass" or refinement.faces.shape[0] != 10_592:
        raise RuntimeError("P2 round-two binding failed")
    return fixture, refinement.vertices, refinement.faces, certificate.report()


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
    fixture, carrier_vertices, carrier_faces, p2_report = _fixture_refinement(
        constructor, repetition_root
    )
    combined = np.vstack((carrier_vertices, fixture.source_vertices))
    lower = np.min(combined, axis=0)
    upper = np.max(combined, axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    padding = BOUND_PADDING_DIAGONAL_FRACTION * diagonal
    lattice = FreudenthalLatticeV1.create(
        lower - padding, upper + padding, nodes_per_axis=NODES_PER_AXIS
    )
    if (
        lattice.vertices.shape[0] != CONTROL_COUNT
        or np.count_nonzero(~lattice.boundary_mask) != FREE_CONTROL_COUNT
    ):
        raise AssertionError("registered E17 control profile changed")
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
    reference_topology = _surface_topology(
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
            step = fit_and_certify_bilipschitz_step(
                lattice,
                carrier_vertices,
                proposal,
                minimum_retained_displacement_ratio=(
                    MINIMUM_TANGENTIAL_RETENTION if name == "tangential_sliding" else 0.0
                ),
                timeout_seconds=MAXIMUM_TOTAL_SECONDS,
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
        carrier_endpoint = carrier_surface.mapped_vertices(lattice, step.accepted_controls)
        source_endpoint = source_surface.mapped_vertices(lattice, step.accepted_controls)
        replay_endpoint = carrier_surface.mapped_vertices(lattice, step.accepted_controls.copy())
        area = parent_area_path_report(
            carrier_vertices, carrier_faces, carrier_surface, carrier_endpoint
        )
        topology = _surface_topology(carrier_endpoint, carrier_surface.faces)
        audit: dict[str, Any] = {}
        diagnostic = ""
        audit_elapsed = 0.0
        if step.status == "pass" and area["status"] == "pass":
            try:
                audit, diagnostic, audit_elapsed = _exact_endpoint_audit(
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
        trajectory_blockers.extend(area["blockers"])
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
        _write_npz_immutable(
            artifact_path,
            lattice_vertices=lattice.vertices,
            lattice_tetrahedra=lattice.tetrahedra,
            accepted_controls=step.accepted_controls,
            dyadic_scale_numerator=np.asarray(step.dyadic_scale_numerator, dtype=np.int64),
            carrier_reference_vertices=carrier_surface.reference_vertices,
            carrier_endpoint_vertices=carrier_endpoint,
            carrier_faces=carrier_surface.faces,
            carrier_parent_face_indices=carrier_surface.parent_face_indices,
            carrier_corner_barycentrics=carrier_surface.corner_barycentric_text,
            source_reference_vertices=source_surface.reference_vertices,
            source_endpoint_vertices=source_endpoint,
            source_faces=source_surface.faces,
            decision_sha256=np.asarray(step.decision_sha256),
        )
        trajectories.append(
            {
                "name": name,
                "status": "pass" if not trajectory_blockers else "fail",
                "step": step.report(),
                "area_path": area,
                "topology": topology,
                "exact_endpoint_audit": audit,
                "exact_endpoint_diagnostic": diagnostic,
                "endpoint_audit_elapsed_seconds": audit_elapsed,
                "elapsed_seconds": time.monotonic() - step_started,
                "artifact": _report_path(artifact_path),
                "artifact_sha256": _sha256(artifact_path),
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
        "carrier_refinement": carrier_surface.report(),
        "source_refinement": source_surface.report(),
        "reference_topology": reference_topology,
        "refinement_elapsed_seconds": refinement_elapsed,
        "trajectories": trajectories,
        "elapsed_seconds": time.monotonic() - started,
        "blockers": blockers,
    }


def _peak_memory_gib() -> float:
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return maximum / (1024.0**3)
    return maximum * 1024.0 / (1024.0**3)


def run_public_gate(artifact_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_CORE_LIMIT))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_CORE_LIMIT))
    git = _git_binding()
    controls = run_bilipschitz_controls()
    blockers: list[str] = []
    if not git["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")
    if controls["status"] != "pass":
        blockers.append("bilipschitz_controls")
    repetitions: list[dict[str, Any]] = []
    if not blockers:
        constructor, auditor = _build_tools()
        with tempfile.TemporaryDirectory(prefix="frayid-e17-public-") as directory:
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
        first = repetitions[0]
        second = repetitions[1]
        for key in (
            "lattice_sha256",
            "carrier_refinement",
            "source_refinement",
        ):
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
    peak_memory = _peak_memory_gib()
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
            "endpoint_audit_seconds_per_proposal": MAXIMUM_ENDPOINT_AUDIT_SECONDS,
            "nodes_per_axis": NODES_PER_AXIS,
            "control_count": CONTROL_COUNT,
            "free_control_count": FREE_CONTROL_COUNT,
            "block_count": 1,
            "kappa": KAPPA,
            "minimum_tangential_retention": MINIMUM_TANGENTIAL_RETENTION,
        },
        "execution_counters": {
            "private_input_reads": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "development_evidence_reads": 0,
            "modal_invocations": 0,
            "sealed_test_accesses": 0,
            "automatic_paid_retries": 0,
        },
        "bindings": {
            "fixture_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
            "mechanism_source_sha256": _sha256(PROJECT_ROOT / "src/frayid/coarse_bilipschitz.py"),
            "runner_source_sha256": _sha256(Path(__file__)),
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
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    artifact_root = arguments.output.parent / f"{arguments.output.stem}_artifacts"
    if arguments.output.exists():
        raise FileExistsError(f"immutable E17 report exists: {arguments.output}")
    if artifact_root.exists():
        raise FileExistsError(f"immutable E17 artifact directory exists: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-e17-supervisor-") as directory:
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
