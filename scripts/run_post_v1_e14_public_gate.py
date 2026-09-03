"""Run the preregistered public E14 dyadic-refinement constructor gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import run_post_v1_e12_public_gate as e12
from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import (
    PUBLIC_FIDELITY_INPUT_SHA256,
    GenusCarrierFidelityFixture,
    public_genus_fidelity_fixtures,
)
from frayid.io import write_json
from frayid.source_exclusion_carrier import uniform_conforming_subdivide

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e14_dyadic_refinement_constructor_r01"
REPORT_SCHEMA = "post_v1_e14_public_dyadic_refinement_gate.v1"
MAX_SECONDS = 30 * 60
MAXIMUM_FINAL_FACES = 20_000
MAXIMUM_IDENTITY_BBOX_FRACTION = 1e-12


def build_tools() -> tuple[Path, Path]:
    e14_build = PROJECT_ROOT / "build/e14_cgal"
    e10_build = PROJECT_ROOT / "build/e10_cgal"
    for source, build in (
        (PROJECT_ROOT / "tools/e14_cgal", e14_build),
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
    return e14_build / "frayid_e14_dyadic_envelope", e10_build / "frayid_e10_exact_audit"


def _grid_for_fixture(fixture: GenusCarrierFidelityFixture) -> float:
    lower = np.min(fixture.source_vertices, axis=0)
    upper = np.max(fixture.source_vertices, axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    magnitude = max(diagonal, float(np.max(np.abs(fixture.source_vertices))))
    return math.ldexp(1.0, math.floor(math.log2(magnitude)) - 40)


def _audit(
    auditor: Path, source_path: Path, mesh_path: Path, report_path: Path
) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(
        [str(auditor), str(source_path), str(mesh_path), str(report_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAX_SECONDS,
    )
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    return report, (completed.stdout + completed.stderr).strip()


def _identity_report(
    parent_vertices: np.ndarray,
    parent_faces: np.ndarray,
    refined_vertices: np.ndarray,
) -> dict[str, float | bool]:
    parent = trimesh.Trimesh(vertices=parent_vertices, faces=parent_faces, process=False)
    _, distances, _ = trimesh.proximity.closest_point(parent, refined_vertices)  # type: ignore[no-untyped-call]
    diagonal = float(np.linalg.norm(np.ptp(parent_vertices, axis=0)))
    maximum_fraction = float(np.max(distances, initial=0.0) / max(diagonal, 1e-300))
    return {
        "parent_vertices_retained_bitwise": bool(
            np.array_equal(refined_vertices[: len(parent_vertices)], parent_vertices)
        ),
        "maximum_vertex_to_parent_surface_bbox_fraction": maximum_fraction,
        "status": bool(maximum_fraction <= MAXIMUM_IDENTITY_BBOX_FRACTION),
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
    e12._write_fixture(source_path, fixture)
    repetitions: list[dict[str, Any]] = []
    level_bytes: list[list[bytes]] = []
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
            repetitions.append(
                {
                    "repetition": repetition,
                    "status": "fail",
                    "diagnostic": (constructed.stdout + constructed.stderr).strip(),
                }
            )
            continue
        parent_vertices, parent_faces = read_e10_mesh(parent_path)
        grid = _grid_for_fixture(fixture)
        grid_integers = np.rint(parent_vertices / grid)
        dyadic_round_trip = bool(
            np.max(np.abs(grid_integers), initial=0.0) <= 2**45
            and np.array_equal(parent_vertices, grid_integers * grid)
        )
        if not dyadic_round_trip:
            blockers.append(f"repetition_{repetition}:dyadic_round_trip_failure")
        level_reports: list[dict[str, Any]] = []
        serialized: list[bytes] = []
        for rounds in range(3):
            vertices, faces = uniform_conforming_subdivide(
                parent_vertices, parent_faces, rounds=rounds
            )
            mesh_path = fixture_root / f"round_{rounds}_{repetition}.e10mesh"
            e12._write_mesh(mesh_path, vertices, faces)
            serialized.append(mesh_path.read_bytes())
            audit_path = fixture_root / f"round_{rounds}_{repetition}_audit.json"
            audit, diagnostic = _audit(auditor, source_path, mesh_path, audit_path)
            identity = _identity_report(parent_vertices, parent_faces, vertices)
            expected_multiplier = 4**rounds
            level_blockers: list[str] = []
            if audit.get("status") != "pass":
                level_blockers.append("exact_audit")
            if len(faces) != len(parent_faces) * expected_multiplier:
                level_blockers.append("face_multiplier")
            if len(faces) > MAXIMUM_FINAL_FACES:
                level_blockers.append("face_cap")
            if not identity["parent_vertices_retained_bitwise"]:
                level_blockers.append("parent_vertex_retention")
            if not identity["status"]:
                level_blockers.append("geometric_identity")
            blockers.extend(
                f"repetition_{repetition}:round_{rounds}:{value}" for value in level_blockers
            )
            level_reports.append(
                {
                    "rounds": rounds,
                    "vertex_count": len(vertices),
                    "face_count": len(faces),
                    "expected_face_multiplier": expected_multiplier,
                    "exact_audit": audit,
                    "exact_diagnostic": diagnostic,
                    "identity": identity,
                    "status": "pass" if not level_blockers else "fail",
                    "blockers": level_blockers,
                }
            )
        level_bytes.append(serialized)
        repetitions.append(
            {
                "repetition": repetition,
                "status": (
                    "pass"
                    if dyadic_round_trip
                    and all(value["status"] == "pass" for value in level_reports)
                    else "fail"
                ),
                "dyadic_grid": grid,
                "dyadic_round_trip": dyadic_round_trip,
                "levels": level_reports,
            }
        )
    deterministic = (
        len(level_bytes) == 2
        and len(level_bytes[0]) == 3
        and all(first == second for first, second in zip(*level_bytes, strict=True))
    )
    if level_bytes and not deterministic:
        blockers.append("nondeterministic_level_serialization")
    return {
        **fixture.as_public_record(),
        "status": "pass" if not blockers else "fail",
        "repetitions": repetitions,
        "deterministic_all_level_bytes": deterministic,
        "round_2_sha256": (hashlib.sha256(level_bytes[0][2]).hexdigest() if level_bytes else None),
        "blockers": blockers,
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    constructor, auditor = build_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-e14-public-") as temporary_name:
        root = Path(temporary_name)
        fixture_reports = [
            _run_fixture(fixture, constructor=constructor, auditor=auditor, root=root)
            for fixture in public_genus_fidelity_fixtures()
        ]
    blockers = [
        f"fixture:{fixture['name']}" for fixture in fixture_reports if fixture["status"] != "pass"
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "gate": "public_dyadic_refinement_constructor",
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_geometry_only",
        "inherited_e11_fixture_definition_sha256": PUBLIC_FIDELITY_INPUT_SHA256,
        "constructor": "CGAL_6.2_EPECK_convex_hull_dyadic_129_over_128",
        "fixtures": fixture_reports,
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
        raise FileExistsError(f"immutable E14 report exists: {arguments.output}")
    report = run_public_gate()
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
