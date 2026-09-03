"""Run the preregistered public-only E15 IPC barrier sliding gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import run_post_v1_e6_preflight as e6
import run_post_v1_e12_public_gate as e12
import run_post_v1_e14_public_gate as e14
from frayid.barrier_sliding_carrier import EXPERIMENT_ID, barrier_sliding_carrier
from frayid.embedded_carrier import embedded_surface_fidelity, read_e10_mesh
from frayid.genus_carrier import (
    FIDELITY_MAXIMUM_RELATIVE_VOLUME_ERROR,
    FIDELITY_SAMPLE_COUNT,
    FIDELITY_SEED,
    PUBLIC_FIDELITY_INPUT_SHA256,
    GenusCarrierFidelityFixture,
    public_genus_fidelity_fixtures,
)
from frayid.interface_field import certify_interface_surface, certify_zero_subcomplex
from frayid.io import write_json
from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "post_v1_e15_public_ipc_barrier_sliding_gate.v1"
MAX_SECONDS = 2 * 60 * 60


def _e6_consumer(
    builder: Path,
    mesh: trimesh.Trimesh,
    root: Path,
    name: str,
) -> dict[str, Any]:
    bounds = e6._bounds(mesh)
    try:
        field, elapsed, diagnostic = e6._run_builder(
            builder, mesh, bounds, root, f"{name}_e6_consumer"
        )
        zero = certify_zero_subcomplex(field)
        surface = certify_interface_surface(
            field, np.asarray(mesh.vertices), np.asarray(mesh.faces), bounds
        )
        blockers = []
        if zero["status"] != "pass":
            blockers.append("zero_subcomplex_certificate")
        if surface["status"] != "pass":
            blockers.append("interface_surface_certificate")
        return {
            "status": "pass" if not blockers else "fail",
            "field_vertex_count": len(field.vertices),
            "tetrahedron_count": len(field.tetrahedra),
            "elapsed_seconds": elapsed,
            "zero_subcomplex": zero,
            "surface": surface,
            "diagnostic": diagnostic,
            "blockers": blockers,
        }
    except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
        return {"status": "fail", "blockers": ["builder_failure"], "diagnostic": str(error)}


def _run_fixture(
    fixture: GenusCarrierFidelityFixture,
    *,
    constructor: Path,
    auditor: Path,
    e6_builder: Path,
    root: Path,
) -> dict[str, Any]:
    fixture_root = root / fixture.name
    fixture_root.mkdir(parents=True, exist_ok=False)
    source_path = fixture_root / "source.e6mesh"
    e12._write_fixture(source_path, fixture)
    grid = e14._grid_for_fixture(fixture)
    runs: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    initial_audits: list[dict[str, Any]] = []
    blockers: list[str] = []

    for repetition in range(2):
        parent_path = fixture_root / f"parent_{repetition}.e10mesh"
        constructed = subprocess.run(
            [str(constructor), str(source_path), str(parent_path)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_SECONDS,
        )
        if constructed.returncode != 0 or not parent_path.is_file():
            blockers.append(f"repetition_{repetition}:constructor_failure")
            runs.append(
                {
                    "repetition": repetition,
                    "status": "fail",
                    "diagnostic": (constructed.stdout + constructed.stderr).strip(),
                }
            )
            continue
        parent_vertices, parent_faces = read_e10_mesh(parent_path)
        refinement = subdivide_with_exact_provenance(parent_vertices, parent_faces, rounds=2)
        certificate = certify_exact_dyadic_refinement(
            parent_vertices,
            parent_faces,
            refinement,
            parent_grid=grid,
            rounds=2,
        )
        initial_path = fixture_root / f"initial_{repetition}.e10mesh"
        e12._write_mesh(initial_path, refinement.vertices, refinement.faces)
        audit_path = fixture_root / f"initial_audit_{repetition}.json"
        initial_audit, diagnostic = e14._audit(auditor, source_path, initial_path, audit_path)
        initial_audits.append(initial_audit)
        if certificate.status != "pass" or initial_audit.get("status") != "pass":
            blockers.append(f"repetition_{repetition}:initial_certificate_or_audit")
            runs.append(
                {
                    "repetition": repetition,
                    "status": "fail",
                    "certificate": certificate.report(),
                    "initial_exact_audit": initial_audit,
                    "diagnostic": diagnostic,
                }
            )
            continue
        result = barrier_sliding_carrier(
            parent_vertices,
            parent_faces,
            fixture.source_vertices,
            fixture.source_faces,
            pitch=fixture.pitch,
            parent_grid=grid,
        )
        output_path = fixture_root / f"barrier_sliding_{repetition}.e10mesh"
        e12._write_mesh(output_path, result.vertices, result.faces)
        output_paths.append(output_path)
        runs.append(
            {
                "repetition": repetition,
                "initial_exact_audit": initial_audit,
                **result.report(),
            }
        )
        if result.status != "pass":
            blockers.extend(f"repetition_{repetition}:{value}" for value in result.blockers)

    deterministic_bytes = bool(
        len(output_paths) == 2 and output_paths[0].read_bytes() == output_paths[1].read_bytes()
    )
    deterministic_steps = bool(len(runs) == 2 and runs[0].get("steps") == runs[1].get("steps"))
    deterministic_initial = bool(
        len(initial_audits) == 2 and initial_audits[0] == initial_audits[1]
    )
    if output_paths and not deterministic_bytes:
        blockers.append("nondeterministic_output_serialization")
    if not deterministic_steps:
        blockers.append("nondeterministic_iteration_decisions")
    if not deterministic_initial:
        blockers.append("nondeterministic_initial_exact_certificate")
    if not output_paths:
        return {
            **fixture.as_public_record(),
            "status": "fail",
            "runs": runs,
            "blockers": blockers,
        }

    final_audit_path = fixture_root / "final_exact_audit.json"
    final_audit, final_diagnostic = e14._audit(
        auditor, source_path, output_paths[0], final_audit_path
    )
    if final_audit.get("status") != "pass":
        blockers.append("independent_final_exact_audit")
    vertices, faces = read_e10_mesh(output_paths[0])
    target = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    reference = trimesh.Trimesh(
        vertices=fixture.reference_vertices,
        faces=fixture.reference_faces,
        process=False,
    )
    fidelity = embedded_surface_fidelity(
        reference,
        target,
        pitch=fixture.pitch,
        sample_count=FIDELITY_SAMPLE_COUNT,
        seed=FIDELITY_SEED,
        maximum_relative_volume_error=FIDELITY_MAXIMUM_RELATIVE_VOLUME_ERROR,
    )
    blockers.extend(f"fidelity:{value}" for value in fidelity["blockers"])
    feature = e12._feature_stratum_report(fixture, target)
    if feature["status"] == "fail":
        blockers.append("feature_stratum_p95_distance")
    probe_outside = (
        np.logical_not(target.contains(fixture.exterior_probes))
        if len(fixture.exterior_probes)
        else np.empty(0, dtype=np.bool_)
    )
    if len(probe_outside) and not bool(np.all(probe_outside)):
        blockers.append("registered_exterior_probe_closed")
    consumer = _e6_consumer(e6_builder, target, fixture_root, fixture.name)
    if consumer["status"] != "pass":
        blockers.append("e6_interface_field_consumer")
    return {
        **fixture.as_public_record(),
        "status": "pass" if not blockers else "fail",
        "runs": runs,
        "deterministic_byte_repeat": deterministic_bytes,
        "deterministic_iteration_decisions": deterministic_steps,
        "deterministic_initial_exact_certificate": deterministic_initial,
        "output_sha256": hashlib.sha256(output_paths[0].read_bytes()).hexdigest(),
        "independent_final_exact_audit": final_audit,
        "exact_diagnostic": final_diagnostic,
        "fidelity": fidelity,
        "feature_stratum": feature,
        "registered_exterior_probes_outside": int(np.count_nonzero(probe_outside)),
        "invariance_signature": e12._invariance_signature(fixture, target),
        "e6_interface_field_consumer": consumer,
        "blockers": blockers,
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    constructor, auditor = e14.build_tools()
    e6_builder = e6._builder_path()
    with tempfile.TemporaryDirectory(prefix="frayid-e15-public-") as temporary_name:
        root = Path(temporary_name)
        fixtures = [
            _run_fixture(
                fixture,
                constructor=constructor,
                auditor=auditor,
                e6_builder=e6_builder,
                root=root,
            )
            for fixture in public_genus_fidelity_fixtures()
        ]
    invariance = e12._invariance_reports(fixtures)
    blockers = [
        *(f"fixture:{value['name']}" for value in fixtures if value["status"] != "pass"),
        *(f"invariance:{value['name']}" for value in invariance if value["status"] != "pass"),
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "gate": "public_p2_certified_ipc_barrier_sliding_fidelity",
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_geometry_only",
        "inherited_e11_fixture_definition_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
        "fixtures": fixtures,
        "invariance": invariance,
        "blockers": blockers,
        "elapsed_seconds": time.monotonic() - started,
        "execution_counters": {
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "modal_invocations": 0,
            "sealed_test_accesses": 0,
        },
    }


def _run_worker(report_path: str) -> None:
    write_json(Path(report_path), run_public_gate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"immutable E15 report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-e15-supervisor-") as directory:
        worker_report = Path(directory) / "worker_report.json"
        worker = multiprocessing.get_context("spawn").Process(
            target=_run_worker, args=(str(worker_report),)
        )
        worker.start()
        worker.join(MAX_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(30)
            report = {
                "schema_version": "post_v1_e15_public_failure_report.v1",
                "experiment_id": EXPERIMENT_ID,
                "gate": "public_p2_certified_ipc_barrier_sliding_fidelity",
                "status": "fail",
                "failure_class": "wall_time_ceiling_exceeded",
                "wall_time_ceiling_seconds": MAX_SECONDS,
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry_count": 0,
                "partial_results_promoted": False,
                "blockers": ["wall_time_ceiling_exceeded"],
                "execution_counters": {
                    "private_input_reads": 0,
                    "development_evidence_reads": 0,
                    "image_loads": 0,
                    "optimizer_steps": 0,
                    "modal_invocations": 0,
                    "sealed_test_accesses": 0,
                },
            }
        elif worker.exitcode != 0 or not worker_report.is_file():
            report = {
                "schema_version": "post_v1_e15_public_failure_report.v1",
                "experiment_id": EXPERIMENT_ID,
                "gate": "public_p2_certified_ipc_barrier_sliding_fidelity",
                "status": "fail",
                "failure_class": "worker_failure",
                "worker_exitcode": worker.exitcode,
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry_count": 0,
                "partial_results_promoted": False,
                "blockers": ["worker_failure"],
                "execution_counters": {
                    "private_input_reads": 0,
                    "development_evidence_reads": 0,
                    "image_loads": 0,
                    "optimizer_steps": 0,
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
