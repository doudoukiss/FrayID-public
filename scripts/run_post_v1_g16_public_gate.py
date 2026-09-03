"""Run the registered public-only G16 ambient-scaffold global-path gate."""

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

import run_post_v1_e12_public_gate as e12
import run_post_v1_e14_public_gate as e14
import run_post_v1_p3_public_gate as p3
from frayid.ambient_scaffold import (
    ambient_scaffold_from_constrained_complex,
    read_constrained_ambient_complex,
    solve_harmonic_direction,
)
from frayid.certified_tet_path import certify_tet_step
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import PUBLIC_FIDELITY_INPUT_SHA256, public_genus_fidelity_fixtures
from frayid.global_path_controls import run_global_path_controls
from frayid.interface_field import write_interface_mesh
from frayid.io import write_json
from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e16_ambient_inverse_reconstruction_r01"
CORRECTNESS_ID = "postv1_g16_ambient_scaffold_global_path_r01"
REPORT_SCHEMA = "post_v1_g16_public_ambient_scaffold_gate.v1"
SEED = 20260831
REPETITIONS = 2
MINIMUM_TANGENTIAL_RETENTION = 0.25
MAXIMUM_TOTAL_SECONDS = 7_200.0
MAXIMUM_SOLVE_CERTIFICATE_SECONDS = 60.0
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


def _build_tools() -> tuple[Path, Path, Path]:
    constructor, auditor = e14.build_tools()
    source = PROJECT_ROOT / "tools/ambient_scaffold"
    build = PROJECT_ROOT / "build/e16_ambient_scaffold"
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", str(CPU_CORE_LIMIT)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return constructor, auditor, build / "frayid_e16_ambient_scaffold_builder"


def _compact_surface(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    retained = np.unique(faces)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[retained] = np.arange(retained.size, dtype=np.int64)
    return vertices[retained], remap[faces]


def _exact_endpoint_audit(
    auditor: Path,
    source_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    root: Path,
    name: str,
) -> tuple[dict[str, Any], str, float]:
    compact_vertices, compact_faces = _compact_surface(vertices, faces)
    mesh_path = root / f"{name}_endpoint.e10mesh"
    report_path = root / f"{name}_endpoint_exact_audit.json"
    e12._write_mesh(mesh_path, compact_vertices, compact_faces)
    started = time.monotonic()
    completed = subprocess.run(
        [str(auditor), str(source_path), str(mesh_path), str(report_path)],
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


def _write_npz_immutable(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"immutable G16 artifact exists: {path}")
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def _run_repetition(
    repetition: int,
    *,
    constructor: Path,
    auditor: Path,
    scaffold_builder: Path,
    root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    repetition_started = time.monotonic()
    fixture = next(
        value for value in public_genus_fidelity_fixtures() if value.name == "near_contact_hairpin"
    )
    repetition_root = root / f"repetition_{repetition}"
    repetition_root.mkdir(parents=True, exist_ok=False)
    output_root = artifact_root / f"repetition_{repetition}"
    output_root.mkdir(parents=True, exist_ok=False)
    source_path = repetition_root / "source.e6mesh"
    e12._write_fixture(source_path, fixture)
    parent_path = repetition_root / "parent.e10mesh"
    constructed = subprocess.run(
        [str(constructor), str(source_path), str(parent_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAXIMUM_TOTAL_SECONDS,
    )
    if constructed.returncode != 0 or not parent_path.is_file():
        return {
            "repetition": repetition,
            "status": "fail",
            "blockers": ["e14_parent_constructor"],
            "diagnostic": (constructed.stdout + constructed.stderr).strip(),
        }
    parent_vertices, parent_faces = read_e10_mesh(parent_path)
    refinement = subdivide_with_exact_provenance(parent_vertices, parent_faces, rounds=2)
    p2 = certify_exact_dyadic_refinement(
        parent_vertices,
        parent_faces,
        refinement,
        parent_grid=e14._grid_for_fixture(fixture),
        rounds=2,
    )
    if p2.status != "pass" or refinement.faces.shape[0] != 10_592:
        return {
            "repetition": repetition,
            "status": "fail",
            "p2_certificate": p2.report(),
            "blockers": ["p2_round_two_binding"],
        }

    combined_vertices = np.vstack((refinement.vertices, fixture.source_vertices))
    combined_faces = np.vstack(
        (refinement.faces, fixture.source_faces + refinement.vertices.shape[0])
    )
    lower = np.min(combined_vertices, axis=0)
    upper = np.max(combined_vertices, axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    padding = BOUND_PADDING_DIAGONAL_FRACTION * diagonal
    bounds = (lower - padding, upper + padding)
    combined_path = repetition_root / "combined_nested.e6mesh"
    field_path = repetition_root / "ambient.e16scaffold"
    write_interface_mesh(combined_path, combined_vertices, combined_faces, bounds)
    meshing_started = time.monotonic()
    meshed = subprocess.run(
        [str(scaffold_builder), str(combined_path), str(field_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAXIMUM_TOTAL_SECONDS,
    )
    meshing_elapsed = time.monotonic() - meshing_started
    if meshed.returncode != 0 or not field_path.is_file():
        return {
            "repetition": repetition,
            "status": "fail",
            "p2_certificate": p2.report(),
            "meshing_elapsed_seconds": meshing_elapsed,
            "blockers": ["ambient_scaffold_construction"],
            "diagnostic": (meshed.stdout + meshed.stderr).strip(),
        }
    complex_ = read_constrained_ambient_complex(field_path)
    scaffold = ambient_scaffold_from_constrained_complex(
        complex_,
        source_carrier_vertices=refinement.vertices,
        source_carrier_faces=refinement.faces,
        source_carrier_face_count=refinement.faces.shape[0],
        constructor_bindings={
            "builder": "frayid_e16_ambient_scaffold_builder",
            "cgal": "6.2",
            "builder_source_sha256": _sha256(
                PROJECT_ROOT / "tools/ambient_scaffold/ambient_scaffold_builder.cpp"
            ),
            "fixture_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
            "padding_diagonal_fraction": str(BOUND_PADDING_DIAGONAL_FRACTION),
        },
    )
    scaffold_path = output_root / "ambient_scaffold.npz"
    scaffold.save(scaffold_path)

    trajectories: list[dict[str, Any]] = []
    blockers: list[str] = []
    for name, target in p3._trajectory_proposals(refinement.vertices, refinement.faces, fixture):
        proposal = np.asarray(target - refinement.vertices, dtype=np.float64)
        solve_started = time.monotonic()
        harmonic = solve_harmonic_direction(scaffold, refinement.faces, proposal)
        step = certify_tet_step(
            scaffold,
            refinement.faces,
            proposal,
            harmonic,
            minimum_retained_displacement_ratio=MINIMUM_TANGENTIAL_RETENTION,
            timeout_seconds=MAXIMUM_SOLVE_CERTIFICATE_SECONDS,
        )
        solve_certificate_elapsed = time.monotonic() - solve_started
        audit: dict[str, Any] = {}
        diagnostic = ""
        endpoint_elapsed = 0.0
        if step.status == "pass":
            audit, diagnostic, endpoint_elapsed = _exact_endpoint_audit(
                auditor,
                source_path,
                step.accepted_vertices,
                scaffold.carrier_faces,
                repetition_root,
                name,
            )
            step = step.with_endpoint_audit(audit)
        trajectory_blockers = list(step.blockers)
        if solve_certificate_elapsed > MAXIMUM_SOLVE_CERTIFICATE_SECONDS:
            trajectory_blockers.append("solve_certificate_time")
        if endpoint_elapsed > MAXIMUM_ENDPOINT_AUDIT_SECONDS:
            trajectory_blockers.append("endpoint_audit_time")
        if name == "native_pressure" and step.retained_displacement_ratio <= 0.0:
            trajectory_blockers.append("native_motion_not_positive")
        if name == "tangential_sliding" and (
            step.retained_displacement_ratio < MINIMUM_TANGENTIAL_RETENTION
        ):
            trajectory_blockers.append("tangential_motion_retention")
        blockers.extend(f"{name}:{value}" for value in trajectory_blockers)
        endpoint_path = output_root / f"{name}_endpoint.npz"
        _write_npz_immutable(
            endpoint_path,
            vertices=step.accepted_vertices,
            carrier_faces=scaffold.carrier_faces,
            accepted_alpha=np.asarray(step.accepted_alpha),
            decision_sha256=np.asarray(step.decision_sha256),
        )
        trajectories.append(
            {
                "name": name,
                "status": "pass" if not trajectory_blockers else "fail",
                "solve_certificate_elapsed_seconds": solve_certificate_elapsed,
                "certificate": step.report(),
                "endpoint_exact_diagnostic": diagnostic,
                "artifact": _report_path(endpoint_path),
                "blockers": trajectory_blockers,
            }
        )
        if trajectory_blockers:
            break

    return {
        "repetition": repetition,
        "status": "pass" if not blockers else "fail",
        "p2_certificate": p2.report(),
        "source_face_count": int(fixture.source_faces.shape[0]),
        "carrier_face_count": int(refinement.faces.shape[0]),
        "meshing_elapsed_seconds": meshing_elapsed,
        "meshing_diagnostic": (meshed.stdout + meshed.stderr).strip(),
        "scaffold": scaffold.report(),
        "scaffold_artifact": _report_path(scaffold_path),
        "trajectories": trajectories,
        "elapsed_seconds": time.monotonic() - repetition_started,
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
    controls = run_global_path_controls()
    blockers: list[str] = []
    if not git["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")
    if controls["status"] != "pass":
        blockers.append("global_path_controls")
    repetitions: list[dict[str, Any]] = []
    if not blockers:
        constructor, auditor, scaffold_builder = _build_tools()
        with tempfile.TemporaryDirectory(prefix="frayid-g16-public-") as directory:
            root = Path(directory)
            for repetition in range(REPETITIONS):
                result = _run_repetition(
                    repetition,
                    constructor=constructor,
                    auditor=auditor,
                    scaffold_builder=scaffold_builder,
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
        if (
            repetitions[0]["scaffold"]["scaffold_sha256"]
            != repetitions[1]["scaffold"]["scaffold_sha256"]
        ):
            blockers.append("scaffold_repetition_mismatch")
        first_trajectories = repetitions[0]["trajectories"]
        second_trajectories = repetitions[1]["trajectories"]
        if [value["name"] for value in first_trajectories] != [
            value["name"] for value in second_trajectories
        ]:
            blockers.append("trajectory_identity_mismatch")
        elif any(
            first["certificate"]["decision_sha256"] != second["certificate"]["decision_sha256"]
            for first, second in zip(first_trajectories, second_trajectories, strict=True)
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
        "correctness_id": CORRECTNESS_ID,
        "gate": "public_ambient_scaffold_global_path",
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
            "solve_and_certificate_seconds_per_proposal": MAXIMUM_SOLVE_CERTIFICATE_SECONDS,
            "endpoint_audit_seconds_per_proposal": MAXIMUM_ENDPOINT_AUDIT_SECONDS,
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
        "blockers": blockers,
    }


def _worker(report_path: str, artifact_root: str) -> None:
    write_json(Path(report_path), run_public_gate(Path(artifact_root)))


def _failure_report(failure: str, started: float, exitcode: int | None) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "correctness_id": CORRECTNESS_ID,
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
        raise FileExistsError(f"immutable G16 report exists: {arguments.output}")
    if artifact_root.exists():
        raise FileExistsError(f"immutable G16 artifact directory exists: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-g16-supervisor-") as directory:
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
