"""Run the public-only E10 Alpha Wrapping parameter-freeze gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh

from frayid.embedded_carrier import embedded_surface_fidelity, read_e10_mesh
from frayid.interface_field import write_interface_mesh
from frayid.io import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e10_embedded_carrier_transfer_r01"
ALPHA_OVER_PITCH = (0.5, 1.0, 2.0, 4.0)
OFFSET_OVER_ALPHA = (0.05, 0.1)
SEED = 20260831


@dataclass(frozen=True)
class Fixture:
    name: str
    source_vertices: np.ndarray
    source_faces: np.ndarray
    target: trimesh.Trimesh
    pitch: float
    exterior_probes: np.ndarray


def _star_surface() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    radius = np.linalg.norm(vertices, axis=1)
    azimuth = np.arctan2(vertices[:, 1], vertices[:, 0])
    elevation_weight = 1.0 - np.square(vertices[:, 2] / radius)
    scale = 1.0 + 0.12 * np.cos(5.0 * azimuth) * elevation_weight
    mesh.vertices = vertices * scale[:, None]
    return cast(trimesh.Trimesh, mesh)


def _fixture(name: str, target: trimesh.Trimesh, *, defective: bool = False) -> Fixture:
    target = target.copy()
    target.fix_normals()
    diagonal = float(np.linalg.norm(target.bounds[1] - target.bounds[0]))
    pitch = diagonal / 40.0
    source_vertices = np.asarray(target.vertices, dtype=np.float64)
    source_faces = np.asarray(target.faces, dtype=np.int64)
    if defective:
        center = source_vertices.mean(axis=0)
        radius = float(np.max(np.linalg.norm(source_vertices - center, axis=1)))
        internal = np.asarray(
            [
                center + np.asarray((0.15 * radius, 0.0, 0.0)),
                center + np.asarray((-0.15 * radius, 0.0, 0.0)),
                center + np.asarray((0.0, 0.15 * radius, 0.1 * radius)),
            ]
        )
        base = len(source_vertices)
        source_vertices = np.vstack((source_vertices, internal))
        source_faces = np.vstack(
            (source_faces, source_faces[:2, ::-1], np.asarray([[base, base + 1, base + 2]]))
        )
    directions = np.eye(3, dtype=np.float64)
    directions = np.vstack((directions, -directions))
    extents = np.max(np.abs(np.asarray(target.vertices)), axis=0)
    probes = directions * (extents + 2.0 * pitch)
    return Fixture(name, source_vertices, source_faces, target, pitch, probes)


def public_fixtures() -> tuple[Fixture, ...]:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    ellipsoid = sphere.copy()
    ellipsoid.apply_scale((1.0, 0.72, 1.35))
    rotation = trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
        np.deg2rad(31.0), (0.3, 0.8, 0.5)
    )
    ellipsoid.apply_transform(rotation)
    star = _star_surface()
    rigid_star = star.copy()
    rigid_star.apply_transform(
        trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
            np.deg2rad(47.0), (0.7, 0.2, 0.6)
        )
    )
    rigid_star.apply_translation((0.4, -0.2, 0.3))
    return (
        _fixture("sphere", sphere),
        _fixture("rotated_ellipsoid", ellipsoid),
        _fixture("concave_high_curvature", star),
        _fixture("defective_rigid_surface", rigid_star, defective=True),
    )


def _build_tools() -> tuple[Path, Path]:
    build = PROJECT_ROOT / "build/e10_cgal"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT / "tools/e10_cgal"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "8"], cwd=PROJECT_ROOT, check=True
    )
    return build / "frayid_e10_alpha_wrap", build / "frayid_e10_exact_audit"


def _run_candidate(
    fixture: Fixture,
    *,
    alpha_over_pitch: float,
    offset_over_alpha: float,
    wrapper: Path,
    auditor: Path,
    root: Path,
) -> dict[str, Any]:
    fixture_root = root / fixture.name
    fixture_root.mkdir(parents=True)
    source_path = fixture_root / "source.e6mesh"
    lower, upper = fixture.source_vertices.min(axis=0), fixture.source_vertices.max(axis=0)
    padding = max(float(np.max(upper - lower)) * 0.25, fixture.pitch)
    write_interface_mesh(
        source_path,
        fixture.source_vertices,
        fixture.source_faces,
        (lower - padding, upper + padding),
    )
    alpha = alpha_over_pitch * fixture.pitch
    offset = offset_over_alpha * alpha
    output_paths = [fixture_root / f"wrap_{repeat}.e10mesh" for repeat in range(2)]
    elapsed = 0.0
    for output in output_paths:
        started = time.monotonic()
        completed = subprocess.run(
            [str(wrapper), str(source_path), repr(alpha), repr(offset), str(output)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        elapsed += time.monotonic() - started
        if completed.returncode:
            return {
                "status": "fail",
                "blockers": ["alpha_wrap_failure"],
                "diagnostic": (completed.stdout + completed.stderr).strip(),
                "elapsed_seconds": elapsed,
            }
    deterministic = output_paths[0].read_bytes() == output_paths[1].read_bytes()
    audit_path = fixture_root / "exact_audit.json"
    audited = subprocess.run(
        [str(auditor), str(source_path), str(output_paths[0]), str(audit_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    vertices, faces = read_e10_mesh(output_paths[0])
    wrapped = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    fidelity = embedded_surface_fidelity(
        fixture.target,
        wrapped,
        pitch=fixture.pitch,
        sample_count=5_000,
        seed=SEED,
    )
    outside = np.logical_not(wrapped.contains(fixture.exterior_probes))
    blockers: list[str] = []
    if not deterministic:
        blockers.append("nondeterministic_output")
    if audited.returncode or audit.get("status") != "pass":
        blockers.append("exact_audit")
    blockers.extend(str(value) for value in fidelity["blockers"])
    if not bool(np.all(outside)):
        blockers.append("registered_exterior_gap_probe_closed")
    worst = max(
        fidelity["source_to_target"]["mean_distance_pitch"] / 0.5,
        fidelity["target_to_source"]["mean_distance_pitch"] / 0.5,
        fidelity["source_to_target"]["p95_distance_pitch"],
        fidelity["target_to_source"]["p95_distance_pitch"],
        fidelity["source_to_target"]["median_normal_error_degrees"] / 5.0,
        fidelity["target_to_source"]["median_normal_error_degrees"] / 5.0,
        fidelity["legacy_relative_volume_error"] / 0.03,
    )
    return {
        "status": "pass" if not blockers else "fail",
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "deterministic_repeat": deterministic,
        "exact_audit": audit,
        "fidelity": fidelity,
        "registered_exterior_probe_count": len(outside),
        "registered_exterior_probes_outside": int(np.count_nonzero(outside)),
        "worst_normalized_fidelity_score": float(worst),
        "elapsed_seconds": elapsed,
        "blockers": blockers,
    }


def run_public_gate() -> dict[str, Any]:
    started = time.monotonic()
    wrapper, auditor = _build_tools()
    candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="frayid-e10-public-") as temporary_name:
        temporary = Path(temporary_name)
        for alpha_over_pitch in ALPHA_OVER_PITCH:
            for offset_over_alpha in OFFSET_OVER_ALPHA:
                fixture_reports = {
                    fixture.name: _run_candidate(
                        fixture,
                        alpha_over_pitch=alpha_over_pitch,
                        offset_over_alpha=offset_over_alpha,
                        wrapper=wrapper,
                        auditor=auditor,
                        root=temporary / f"alpha_{alpha_over_pitch:g}_offset_{offset_over_alpha:g}",
                    )
                    for fixture in public_fixtures()
                }
                passed = all(value["status"] == "pass" for value in fixture_reports.values())
                candidates.append(
                    {
                        "alpha_over_pitch": alpha_over_pitch,
                        "offset_over_alpha": offset_over_alpha,
                        "status": "pass" if passed else "fail",
                        "worst_normalized_fidelity_score": max(
                            float(value.get("worst_normalized_fidelity_score", float("inf")))
                            for value in fixture_reports.values()
                        ),
                        "maximum_face_count": max(
                            int(value.get("face_count", 0)) for value in fixture_reports.values()
                        ),
                        "fixtures": fixture_reports,
                    }
                )
    eligible = [value for value in candidates if value["status"] == "pass"]
    eligible.sort(
        key=lambda value: (
            value["worst_normalized_fidelity_score"],
            value["maximum_face_count"],
            value["alpha_over_pitch"],
            value["offset_over_alpha"],
        )
    )
    selected = (
        {
            "alpha_over_pitch": eligible[0]["alpha_over_pitch"],
            "offset_over_alpha": eligible[0]["offset_over_alpha"],
        }
        if eligible
        else None
    )
    return {
        "schema_version": "post_v1_e10_public_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if selected is not None else "fail",
        "seed": SEED,
        "parameter_grid": {
            "alpha_over_pitch": list(ALPHA_OVER_PITCH),
            "offset_over_alpha": list(OFFSET_OVER_ALPHA),
        },
        "selection_order": [
            "worst_normalized_fidelity_score",
            "maximum_face_count",
            "alpha_over_pitch",
            "offset_over_alpha",
        ],
        "selected": selected,
        "candidates": candidates,
        "blockers": [] if selected is not None else ["no_public_parameter_pair_passed"],
        "elapsed_seconds": time.monotonic() - started,
        "privacy": "public_synthetic_only",
        "development_evaluations": 0,
        "sealed_test_accesses": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_public_gate()
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"immutable E10 report exists: {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
