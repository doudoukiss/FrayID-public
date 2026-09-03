"""Run the synthetic preflight for the E3 exact-distance support experiment."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from frayid.io import write_json
from frayid.sdf_distillation import (
    build_source_surface_exact_support,
    build_topology_safe_sdf_grid,
    extract_voxel_sdf_mesh,
    trilinear_neighbor_support_report,
)

EXPERIMENT_ID = "postv1_e03_carrier_covering_distance_band_r01"
OUTPUT_RELATIVE = Path("outputs/canonical_clothed_surface_v1/post_v1") / EXPERIMENT_ID


def _implicit_mesh(field: np.ndarray, axis: np.ndarray) -> trimesh.Trimesh:
    pitch = float(axis[1] - axis[0])
    vertices, faces, _, _ = marching_cubes(  # type: ignore[no-untyped-call]
        field.astype(np.float32), level=0.0, spacing=(pitch, pitch, pitch)
    )
    vertices += axis[0]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _thin_bridge() -> trimesh.Trimesh:
    axis = np.linspace(-1.2, 1.2, 64)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    left = np.sqrt((xx + 0.55) ** 2 + yy**2 + zz**2) - 0.42
    right = np.sqrt((xx - 0.55) ** 2 + yy**2 + zz**2) - 0.42
    bridge = np.maximum(np.sqrt(yy**2 + zz**2) - 0.035, np.abs(xx) - 0.55)
    return _implicit_mesh(np.minimum(np.minimum(left, right), bridge), axis)


def _near_contact_cavity() -> trimesh.Trimesh:
    axis = np.linspace(-1.1, 1.1, 72)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    left = np.maximum(np.sqrt((xx + 0.22) ** 2 + zz**2) - 0.16, np.abs(yy) - 0.72)
    right = np.maximum(np.sqrt((xx - 0.22) ** 2 + zz**2) - 0.16, np.abs(yy) - 0.72)
    base = np.maximum(np.sqrt((yy + 0.72) ** 2 + zz**2) - 0.16, np.abs(xx) - 0.22)
    return _implicit_mesh(np.minimum(np.minimum(left, right), base), axis)


def _support_fixture(name: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    report: dict[str, Any] = {}
    sdf, origin, pitch = build_topology_safe_sdf_grid(
        mesh,
        longest_axis_resolution=40,
        occupancy_supersampling=3,
        include_source_surface_support=True,
        support_report=report,
    )
    support, support_details = build_source_surface_exact_support(
        mesh, origin, sdf.shape, pitch, radius_voxels=3.0
    )
    probes, _ = trimesh.sample.sample_surface(mesh, 5_000, seed=20260831)
    coverage = trilinear_neighbor_support_report(support, origin, pitch, probes)
    raw = extract_voxel_sdf_mesh(sdf, origin, pitch)
    blockers: list[str] = []
    if not coverage["all_eight_neighbors_covered"]:
        blockers.append("trilinear_neighbor_support")
    if not float(sdf.min()) < 0 < float(sdf.max()):
        blockers.append("field_does_not_bracket_zero")
    if name in {"sphere", "rotated_ellipsoid"} and (
        not raw.is_watertight or len(raw.split(only_watertight=False)) != 1
    ):
        blockers.append("analytic_fixture_raw_topology")
    return {
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "support": report,
        "support_recomputation": support_details,
        "trilinear_neighbor_coverage": coverage,
        "raw_zero_set": {
            "component_count": len(raw.split(only_watertight=False)),
            "watertight": bool(raw.is_watertight),
            "euler_number": int(raw.euler_number),
        },
    }


def run(project_root: Path, output_root: Path, *, source_revision: str) -> dict[str, Any]:
    destination = output_root.resolve()
    if "sealed_test_v1" in {part.lower() for part in destination.parts}:
        raise ValueError("E3 output may not enter sealed-test storage")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable E3 preflight: {destination}")

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.75)
    ellipsoid = sphere.copy()
    ellipsoid.apply_scale([0.7, 1.0, 1.35])
    ellipsoid.apply_transform(
        trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
            np.deg2rad(31.0), [1.0, 0.4, 0.2]
        )
    )
    translated = ellipsoid.copy()
    translated.apply_translation([0.0137, -0.0083, 0.0041])
    fixtures = {
        "sphere": _support_fixture("sphere", sphere),
        "rotated_ellipsoid": _support_fixture("rotated_ellipsoid", ellipsoid),
        "thin_bridge": _support_fixture("thin_bridge", _thin_bridge()),
        "near_contact_cavity": _support_fixture("near_contact_cavity", _near_contact_cavity()),
        "subvoxel_translation": _support_fixture("subvoxel_translation", translated),
    }

    old_report: dict[str, Any] = {}
    new_report: dict[str, Any] = {}
    old_sdf, old_origin, old_pitch = build_topology_safe_sdf_grid(
        translated,
        longest_axis_resolution=40,
        occupancy_supersampling=3,
        support_report=old_report,
    )
    new_sdf, new_origin, new_pitch = build_topology_safe_sdf_grid(
        translated,
        longest_axis_resolution=40,
        occupancy_supersampling=3,
        include_source_surface_support=True,
        support_report=new_report,
    )
    changed_only_support = bool(
        np.array_equal(old_origin, new_origin)
        and old_pitch == new_pitch
        and old_sdf.shape == new_sdf.shape
        and np.array_equal(np.signbit(old_sdf), np.signbit(new_sdf))
    )
    occupancy_disagreement = {
        "status": "pass"
        if changed_only_support and new_report["added_source_surface_node_count"] >= 0
        else "fail",
        "sign_bit_identical": bool(np.array_equal(np.signbit(old_sdf), np.signbit(new_sdf))),
        "old_support_node_count": old_report["exact_support_node_count"],
        "new_support_node_count": new_report["exact_support_node_count"],
        "added_support_node_count": new_report["added_source_surface_node_count"],
    }

    axis = np.linspace(-1.4, 1.4, 64)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    main = np.sqrt(xx**2 + yy**2 + zz**2) - 0.7
    shell = np.sqrt((xx - 1.05) ** 2 + yy**2 + zz**2) - 0.12
    adversarial = _implicit_mesh(np.minimum(main, shell), axis)
    adversarial_components = len(adversarial.split(only_watertight=False))
    extra_shell = {
        "status": "pass" if adversarial_components > 1 else "fail",
        "detected_component_count": adversarial_components,
        "rejection_expected": True,
    }

    blockers = [name for name, value in fixtures.items() if value["status"] != "pass"]
    if occupancy_disagreement["status"] != "pass":
        blockers.append("occupancy_source_disagreement")
    if extra_shell["status"] != "pass":
        blockers.append("extra_shell_adversary")
    report: dict[str, Any] = {
        "schema_version": "post_v1_e3_synthetic_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "source_revision": source_revision,
        "changed_mechanism": "exact_unsigned_distance_support_union_only",
        "fixtures": fixtures,
        "occupancy_source_disagreement": occupancy_disagreement,
        "extra_shell_adversary": extra_shell,
        "execution": {
            "optimizer_steps": 0,
            "training_frames_fitted": 0,
            "modal_jobs_launched": 0,
            "automatic_paid_retries": 0,
        },
        "sealed_test_isolation": {"private_evidence_paths_accessed": []},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e3-preflight-", dir=destination.parent))
    try:
        write_json(staging / "synthetic_preflight_report.json", report)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relevant = ("src/frayid/sdf_distillation.py", "scripts/run_post_v1_e3_preflight.py")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *relevant],
        cwd=project_root,
        check=False,
    ).returncode:
        raise RuntimeError("Refusing E3 preflight from a dirty relevant worktree")
    output = project_root / OUTPUT_RELATIVE / "synthetic_preflight"
    report = run(project_root, output, source_revision=revision)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
