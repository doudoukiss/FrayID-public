"""Run the preregistered public-only E12 CCD shrinkwrap fidelity gate."""

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

from frayid.embedded_carrier import embedded_surface_fidelity, read_e10_mesh
from frayid.genus_carrier import (
    FIDELITY_MAXIMUM_FEATURE_P95_DISTANCE_PITCH,
    FIDELITY_MAXIMUM_INVARIANCE_DELTA,
    FIDELITY_MAXIMUM_RELATIVE_VOLUME_ERROR,
    FIDELITY_SAMPLE_COUNT,
    FIDELITY_SEED,
    PUBLIC_FIDELITY_INPUT_SHA256,
    GenusCarrierFidelityFixture,
    public_genus_fidelity_fixtures,
)
from frayid.interface_field import write_interface_mesh
from frayid.io import write_json
from frayid.shrinkwrap_carrier import EXPERIMENT_ID, pressure_shrinkwrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "post_v1_e12_public_ccd_shrinkwrap_gate.v1"
MAX_SECONDS = 2 * 60 * 60


def build_exact_tools() -> tuple[Path, Path]:
    e11_build = PROJECT_ROOT / "build/e11_cgal"
    e10_build = PROJECT_ROOT / "build/e10_cgal"
    for source, build in (
        (PROJECT_ROOT / "tools/e11_cgal", e11_build),
        (PROJECT_ROOT / "tools/e10_cgal", e10_build),
    ):
        subprocess.run(
            ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--parallel", "8"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    return e11_build / "frayid_e11_convex_envelope", e10_build / "frayid_e10_exact_audit"


def _write_fixture(path: Path, fixture: GenusCarrierFidelityFixture) -> None:
    lower = np.min(fixture.source_vertices, axis=0)
    upper = np.max(fixture.source_vertices, axis=0)
    padding = max(float(np.max(upper - lower)) * 0.1, fixture.pitch)
    write_interface_mesh(
        path,
        fixture.source_vertices,
        fixture.source_faces,
        (lower - padding, upper + padding),
    )


def _write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    lines = ["FRAYID_E10_MESH 1", f"{len(points)} {len(triangles)}"]
    lines.extend(" ".join(format(float(value), ".17g") for value in row) for row in points)
    lines.extend(" ".join(str(int(value)) for value in row) for row in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _feature_stratum_report(
    fixture: GenusCarrierFidelityFixture, target: trimesh.Trimesh
) -> dict[str, float | int | str]:
    if not len(fixture.feature_face_indices):
        return {
            "status": "not_applicable",
            "face_count": 0,
            "p95_distance_pitch": 0.0,
            "median_normal_error_degrees": 0.0,
        }
    reference = trimesh.Trimesh(
        vertices=fixture.reference_vertices,
        faces=fixture.reference_faces,
        process=False,
    )
    indices = fixture.feature_face_indices
    points = np.asarray(reference.triangles_center)[indices]
    _, distances, target_faces = trimesh.proximity.closest_point(target, points)  # type: ignore[no-untyped-call]
    cosine = np.abs(
        np.einsum(
            "ij,ij->i",
            np.asarray(reference.face_normals)[indices],
            np.asarray(target.face_normals)[target_faces],
        )
    )
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    p95_distance = float(np.quantile(distances, 0.95) / fixture.pitch)
    return {
        "status": (
            "pass" if p95_distance <= FIDELITY_MAXIMUM_FEATURE_P95_DISTANCE_PITCH else "fail"
        ),
        "face_count": len(indices),
        "p95_distance_pitch": p95_distance,
        "median_normal_error_degrees": float(np.median(angles)),
    }


def _invariance_signature(
    fixture: GenusCarrierFidelityFixture, target: trimesh.Trimesh
) -> dict[str, float]:
    reference = trimesh.Trimesh(
        vertices=fixture.reference_vertices,
        faces=fixture.reference_faces,
        process=False,
    )
    points = np.vstack((np.asarray(reference.vertices), np.asarray(reference.triangles_center)))
    _, distances, _ = trimesh.proximity.closest_point(target, points)  # type: ignore[no-untyped-call]
    source_volume = abs(float(reference.volume))
    target_volume = abs(float(target.volume))
    return {
        "mean_distance_pitch": float(np.mean(distances) / fixture.pitch),
        "p95_distance_pitch": float(np.quantile(distances, 0.95) / fixture.pitch),
        "relative_volume_error": abs(target_volume - source_volume) / max(source_volume, 1e-12),
    }


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
    _write_fixture(source_path, fixture)
    run_records: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    blockers: list[str] = []
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
        result = pressure_shrinkwrap(
            initial_vertices,
            initial_faces,
            fixture.source_vertices,
            fixture.source_faces,
            pitch=fixture.pitch,
        )
        output_path = fixture_root / f"shrinkwrap_{repetition}.e10mesh"
        _write_mesh(output_path, result.vertices, result.faces)
        output_paths.append(output_path)
        run_records.append({"repetition": repetition, **result.report()})
        if result.status != "pass":
            blockers.extend(f"repetition_{repetition}:{value}" for value in result.blockers)
    deterministic = (
        len(output_paths) == 2 and output_paths[0].read_bytes() == output_paths[1].read_bytes()
    )
    decision_deterministic = len(run_records) == 2 and run_records[0].get("steps") == run_records[
        1
    ].get("steps")
    if not deterministic:
        blockers.append("nondeterministic_output_serialization")
    if not decision_deterministic:
        blockers.append("nondeterministic_iteration_decisions")
    if not output_paths:
        return {
            **fixture.as_public_record(),
            "status": "fail",
            "runs": run_records,
            "blockers": blockers,
        }

    audit_path = fixture_root / "exact_audit.json"
    audited = subprocess.run(
        [str(auditor), str(source_path), str(output_paths[0]), str(audit_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAX_SECONDS,
    )
    exact_audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    if audited.returncode != 0 or exact_audit.get("status") != "pass":
        blockers.append("independent_exact_audit_failure")
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
    feature_stratum = _feature_stratum_report(fixture, target)
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
        "deterministic_byte_repeat": deterministic,
        "deterministic_iteration_decisions": decision_deterministic,
        "output_sha256": hashlib.sha256(output_paths[0].read_bytes()).hexdigest(),
        "independent_exact_audit": exact_audit,
        "fidelity": fidelity,
        "feature_stratum": feature_stratum,
        "registered_exterior_probes_outside": int(np.count_nonzero(probe_outside)),
        "invariance_signature": _invariance_signature(fixture, target),
        "blockers": blockers,
    }


def _invariance_reports(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        group = fixture.get("invariance_group")
        if group and "invariance_signature" in fixture:
            groups.setdefault(str(group), []).append(fixture)
    reports: list[dict[str, Any]] = []
    for name, members in sorted(groups.items()):
        keys = tuple(members[0]["invariance_signature"])
        maximum_delta = max(
            abs(
                float(member["invariance_signature"][key])
                - float(members[0]["invariance_signature"][key])
            )
            for member in members[1:]
            for key in keys
        )
        blockers: list[str] = []
        if maximum_delta > FIDELITY_MAXIMUM_INVARIANCE_DELTA:
            blockers.append("normalized_metric_delta")
        reports.append(
            {
                "name": name,
                "members": [member["name"] for member in members],
                "maximum_normalized_metric_delta": maximum_delta,
                "status": "pass" if not blockers else "fail",
                "blockers": blockers,
            }
        )
    return reports


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    constructor, auditor = build_exact_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-e12-public-") as temporary_name:
        root = Path(temporary_name)
        fixture_reports = [
            _run_fixture(fixture, constructor=constructor, auditor=auditor, root=root)
            for fixture in public_genus_fidelity_fixtures()
        ]
    invariance = _invariance_reports(fixture_reports)
    blockers = [
        *(f"fixture:{value['name']}" for value in fixture_reports if value["status"] != "pass"),
        *(f"invariance:{value['name']}" for value in invariance if value["status"] != "pass"),
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "gate": "public_ccd_shrinkwrap_fidelity",
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
        raise FileExistsError(f"immutable E12 report exists: {arguments.output}")
    report = run_public_gate()
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
