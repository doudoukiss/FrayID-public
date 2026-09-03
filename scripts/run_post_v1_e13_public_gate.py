"""Run the preregistered public-only E13 source-exclusion shrinkwrap gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import run_post_v1_e12_public_gate as e12
from frayid.embedded_carrier import embedded_surface_fidelity, read_e10_mesh
from frayid.genus_carrier import (
    FIDELITY_MAXIMUM_RELATIVE_VOLUME_ERROR,
    FIDELITY_SAMPLE_COUNT,
    FIDELITY_SEED,
    PUBLIC_FIDELITY_INPUT_SHA256,
    GenusCarrierFidelityFixture,
    public_genus_fidelity_fixtures,
)
from frayid.io import write_json
from frayid.source_exclusion_carrier import (
    EXPERIMENT_ID,
    source_exclusion_shrinkwrap,
    uniform_conforming_subdivide,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "post_v1_e13_public_source_exclusion_gate.v1"
MAX_SECONDS = 2 * 60 * 60


def _exact_audit(
    auditor: Path, source_path: Path, wrap_path: Path, report_path: Path
) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(
        [str(auditor), str(source_path), str(wrap_path), str(report_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAX_SECONDS,
    )
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    diagnostic = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 and not diagnostic:
        diagnostic = f"exact auditor returned {completed.returncode}"
    return report, diagnostic


def _run_fixture(
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
    run_records: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    blockers: list[str] = []
    initial_audits: list[dict[str, Any]] = []
    for repetition in range(2):
        initial_path = fixture_root / f"initial_{repetition}.e10mesh"
        completed = subprocess.run(
            [str(constructor), str(source_path), str(initial_path)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_SECONDS,
        )
        if completed.returncode != 0 or not initial_path.is_file():
            blockers.append(f"repetition_{repetition}:initial_constructor_failure")
            run_records.append(
                {
                    "repetition": repetition,
                    "status": "fail",
                    "constructor_returncode": completed.returncode,
                    "diagnostic": (completed.stdout + completed.stderr).strip(),
                }
            )
            continue
        initial_vertices, initial_faces = read_e10_mesh(initial_path)
        try:
            refined_vertices, refined_faces = uniform_conforming_subdivide(
                initial_vertices, initial_faces
            )
        except ValueError as error:
            blockers.append(f"repetition_{repetition}:{error}")
            run_records.append(
                {"repetition": repetition, "status": "fail", "blockers": [str(error)]}
            )
            continue
        refined_path = fixture_root / f"refined_initial_{repetition}.e10mesh"
        e12._write_mesh(refined_path, refined_vertices, refined_faces)
        initial_audit_path = fixture_root / f"initial_exact_audit_{repetition}.json"
        initial_audit, diagnostic = _exact_audit(
            auditor, source_path, refined_path, initial_audit_path
        )
        initial_audits.append(initial_audit)
        if initial_audit.get("status") != "pass":
            blockers.append(f"repetition_{repetition}:initial_exact_audit_failure")
            run_records.append(
                {
                    "repetition": repetition,
                    "status": "fail",
                    "initial_exact_audit": initial_audit,
                    "diagnostic": diagnostic,
                }
            )
            continue
        result = source_exclusion_shrinkwrap(
            initial_vertices,
            initial_faces,
            fixture.source_vertices,
            fixture.source_faces,
            pitch=fixture.pitch,
        )
        output_path = fixture_root / f"source_exclusion_{repetition}.e10mesh"
        e12._write_mesh(output_path, result.vertices, result.faces)
        output_paths.append(output_path)
        run_records.append(
            {"repetition": repetition, "initial_exact_audit": initial_audit, **result.report()}
        )
        if result.status != "pass":
            blockers.extend(f"repetition_{repetition}:{value}" for value in result.blockers)
    deterministic = (
        len(output_paths) == 2 and output_paths[0].read_bytes() == output_paths[1].read_bytes()
    )
    decision_deterministic = len(run_records) == 2 and run_records[0].get("steps") == run_records[
        1
    ].get("steps")
    initial_deterministic = len(initial_audits) == 2 and initial_audits[0] == initial_audits[1]
    if output_paths and not deterministic:
        blockers.append("nondeterministic_output_serialization")
    if not decision_deterministic:
        blockers.append("nondeterministic_iteration_decisions")
    if not initial_deterministic:
        blockers.append("nondeterministic_initial_exact_certificate")
    if not output_paths:
        return {
            **fixture.as_public_record(),
            "status": "fail",
            "runs": run_records,
            "initial_exact_audits": initial_audits,
            "blockers": blockers,
        }

    final_audit_path = fixture_root / "final_exact_audit.json"
    exact_audit, exact_diagnostic = _exact_audit(
        auditor, source_path, output_paths[0], final_audit_path
    )
    if exact_audit.get("status") != "pass":
        blockers.append("independent_final_exact_audit_failure")
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
    feature_stratum = e12._feature_stratum_report(fixture, target)
    if feature_stratum["status"] == "fail":
        blockers.append("feature_stratum_p95_distance")
    probe_outside = (
        np.logical_not(target.contains(fixture.exterior_probes))
        if len(fixture.exterior_probes)
        else np.empty(0, dtype=np.bool_)
    )
    if len(probe_outside) and not bool(np.all(probe_outside)):
        blockers.append("registered_exterior_probe_closed")
    return {
        **fixture.as_public_record(),
        "status": "pass" if not blockers else "fail",
        "runs": run_records,
        "initial_exact_audits": initial_audits,
        "deterministic_byte_repeat": deterministic,
        "deterministic_iteration_decisions": decision_deterministic,
        "deterministic_initial_exact_certificate": initial_deterministic,
        "output_sha256": hashlib.sha256(output_paths[0].read_bytes()).hexdigest(),
        "independent_final_exact_audit": exact_audit,
        "exact_diagnostic": exact_diagnostic,
        "fidelity": fidelity,
        "feature_stratum": feature_stratum,
        "registered_exterior_probes_outside": int(np.count_nonzero(probe_outside)),
        "invariance_signature": e12._invariance_signature(fixture, target),
        "blockers": blockers,
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    constructor, auditor = e12.build_exact_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-e13-public-") as temporary_name:
        root = Path(temporary_name)
        fixture_reports = [
            _run_fixture(fixture, constructor=constructor, auditor=auditor, root=root)
            for fixture in public_genus_fidelity_fixtures()
        ]
    invariance = e12._invariance_reports(fixture_reports)
    blockers = [
        *(f"fixture:{value['name']}" for value in fixture_reports if value["status"] != "pass"),
        *(f"invariance:{value['name']}" for value in invariance if value["status"] != "pass"),
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "gate": "public_conforming_source_exclusion_fidelity",
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_geometry_only",
        "inherited_e11_fixture_definition_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
        "fixtures": fixture_reports,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"immutable E13 report exists: {arguments.output}")
    report = run_public_gate()
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
