"""Run the public-only E11 genus-by-construction source gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from frayid.embedded_carrier import read_e10_mesh
from frayid.genus_carrier import (
    EXPANSION_DENOMINATOR,
    EXPANSION_NUMERATOR,
    EXPERIMENT_ID,
    PUBLIC_FIXTURE_INPUT_SHA256,
    GenusCarrierFixture,
    public_genus_carrier_fixtures,
)
from frayid.interface_field import write_interface_mesh
from frayid.io import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "post_v1_e11_public_genus_gate.v1"
MAX_SECONDS = 60 * 60


def build_exact_tools() -> tuple[Path, Path]:
    e11_build = PROJECT_ROOT / "build/e11_cgal"
    e10_build = PROJECT_ROOT / "build/e10_cgal"
    for source, build in (
        (PROJECT_ROOT / "tools/e11_cgal", e11_build),
        (PROJECT_ROOT / "tools/e10_cgal", e10_build),
    ):
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--parallel", "16"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    return e11_build / "frayid_e11_convex_envelope", e10_build / "frayid_e10_exact_audit"


def _write_fixture(path: Path, fixture: GenusCarrierFixture) -> None:
    lower = np.min(fixture.vertices, axis=0)
    upper = np.max(fixture.vertices, axis=0)
    padding = max(float(np.max(upper - lower)) * 0.1, 1e-3)
    write_interface_mesh(path, fixture.vertices, fixture.faces, (lower - padding, upper + padding))


def _run_fixture(
    fixture: GenusCarrierFixture,
    *,
    constructor: Path,
    auditor: Path,
    root: Path,
) -> dict[str, Any]:
    fixture_root = root / fixture.name
    fixture_root.mkdir(parents=True, exist_ok=False)
    source = fixture_root / "source.e6mesh"
    outputs = (fixture_root / "envelope_a.e10mesh", fixture_root / "envelope_b.e10mesh")
    _write_fixture(source, fixture)
    constructor_runs: list[dict[str, Any]] = []
    for output in outputs:
        completed = subprocess.run(
            [str(constructor), str(source), str(output)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_SECONDS,
        )
        constructor_runs.append(
            {
                "returncode": completed.returncode,
                "diagnostic": (completed.stdout + completed.stderr).strip(),
            }
        )
    deterministic = all(output.is_file() for output in outputs) and (
        outputs[0].read_bytes() == outputs[1].read_bytes()
    )
    blockers: list[str] = []
    if any(value["returncode"] != 0 for value in constructor_runs):
        blockers.append("constructor_failure")
    if not deterministic:
        blockers.append("nondeterministic_serialization")
    exact_audit: dict[str, Any] = {}
    output_summary: dict[str, Any] = {}
    if outputs[0].is_file():
        audit_path = fixture_root / "exact_audit.json"
        audited = subprocess.run(
            [str(auditor), str(source), str(outputs[0]), str(audit_path)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_SECONDS,
        )
        if audit_path.is_file():
            exact_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audited.returncode != 0 or exact_audit.get("status") != "pass":
            blockers.append("independent_exact_audit_failure")
        vertices, faces = read_e10_mesh(outputs[0])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        output_summary = {
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "diagnostic_euler_number": int(mesh.euler_number),
            "diagnostic_component_count": len(mesh.split(only_watertight=False)),
            "diagnostic_positive_volume": bool(mesh.volume > 0),
        }
    return {
        **fixture.as_public_record(),
        "status": "pass" if not blockers else "fail",
        "constructor_runs": constructor_runs,
        "deterministic_byte_repeat": deterministic,
        "output": output_summary,
        "independent_exact_audit": exact_audit,
        "blockers": blockers,
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    constructor, auditor = build_exact_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-e11-public-") as temporary_name:
        temporary = Path(temporary_name)
        fixtures = [
            _run_fixture(
                fixture,
                constructor=constructor,
                auditor=auditor,
                root=temporary,
            )
            for fixture in public_genus_carrier_fixtures()
        ]
    blockers = [f"fixture:{fixture['name']}" for fixture in fixtures if fixture["status"] != "pass"]
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_geometry_only",
        "public_fixture_definition_sha256": PUBLIC_FIXTURE_INPUT_SHA256,
        "mechanism": "CGAL_6.2_EPECK_convex_hull_with_fixed_general_position_then_exact_rational_expansion",
        "genus_control": "closed_convex_polytope_is_sphere_topology_by_construction",
        "expansion_ratio": {
            "numerator": EXPANSION_NUMERATOR,
            "denominator": EXPANSION_DENOMINATOR,
        },
        "independent_auditor": "existing_E10_CGAL_6.2_EPECK_serialized_mesh_audit",
        "required_exact_output": {
            "self_intersection_pair_count": 0,
            "watertight": True,
            "component_count": 1,
            "outward_oriented": True,
            "euler_number": 2,
        },
        "fixtures": fixtures,
        "blockers": blockers,
        "elapsed_seconds": time.monotonic() - started,
        "execution_counters": {
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "sealed_test_accesses": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_public_gate()
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
