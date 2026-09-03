"""Run the public-only E6 interface-conforming-field preflight."""

from __future__ import annotations

import copy
import json
import math
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from frayid.interface_field import (
    InterfaceField,
    certify_independent_signs,
    certify_interface_surface,
    certify_zero_subcomplex,
    read_interface_field,
    tetrahedron_volumes,
    write_interface_mesh,
)
from frayid.io import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e06_interface_conforming_field_r01"
SEED = 20260831
SIGN_SAMPLE_COUNT = 20_000
MAXIMUM_FIXTURE_SECONDS = 600
MAXIMUM_PUBLIC_VERTICES = 500_000
MAXIMUM_PUBLIC_TETRAHEDRA = 2_000_000
CGAL_VERSION = "6.2"


def _orient_outward(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = mesh.copy()
    if result.volume < 0:
        result.faces = result.faces[:, [0, 2, 1]]
    if not result.is_watertight or not result.is_winding_consistent or result.euler_number != 2:
        raise ValueError("generated public fixture is not a watertight oriented Euler-2 surface")
    return result


def _tube_mesh(
    centers: np.ndarray,
    tangents: np.ndarray,
    radii: np.ndarray,
    *,
    ring_size: int = 24,
) -> trimesh.Trimesh:
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.stack((-tangents[:, 1], tangents[:, 0], np.zeros(centers.shape[0])), axis=1)
    degenerate = np.linalg.norm(normals, axis=1) < 1e-8
    normals[degenerate] = np.array([0.0, 1.0, 0.0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    binormals = np.cross(tangents, normals)
    angles = np.linspace(0.0, 2.0 * math.pi, ring_size, endpoint=False)
    rings = centers[:, None, :] + radii[:, None, None] * (
        np.cos(angles)[None, :, None] * normals[:, None, :]
        + np.sin(angles)[None, :, None] * binormals[:, None, :]
    )
    vertices = rings.reshape(-1, 3).tolist()
    faces: list[list[int]] = []
    for ring in range(centers.shape[0] - 1):
        for slot in range(ring_size):
            following = (slot + 1) % ring_size
            current = ring * ring_size + slot
            current_next = ring * ring_size + following
            next_current = (ring + 1) * ring_size + slot
            next_following = (ring + 1) * ring_size + following
            faces.extend(
                ([current, next_following, next_current], [current, current_next, next_following])
            )
    start_center = len(vertices)
    vertices.append(centers[0].tolist())
    end_center = len(vertices)
    vertices.append(centers[-1].tolist())
    last_ring = (centers.shape[0] - 1) * ring_size
    for slot in range(ring_size):
        following = (slot + 1) % ring_size
        faces.append([start_center, following, slot])
        faces.append([end_center, last_ring + slot, last_ring + following])
    return _orient_outward(
        trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
    )


def _hairpin(gap: float) -> trimesh.Trimesh:
    radius = 0.16
    bend_radius = radius + gap / 2.0
    left = np.stack(
        (
            np.full(8, -bend_radius),
            np.linspace(1.0, 0.0, 8),
            np.zeros(8),
        ),
        axis=1,
    )
    theta = np.linspace(math.pi, 2.0 * math.pi, 33)[1:]
    bend = np.stack(
        (
            bend_radius * np.cos(theta),
            bend_radius * np.sin(theta),
            np.zeros(theta.size),
        ),
        axis=1,
    )
    right = np.stack(
        (
            np.full(7, bend_radius),
            np.linspace(1.0 / 7.0, 1.0, 7),
            np.zeros(7),
        ),
        axis=1,
    )
    centers = np.concatenate((left, bend, right), axis=0)
    tangents = np.gradient(centers, axis=0)
    return _tube_mesh(centers, tangents, np.full(centers.shape[0], radius), ring_size=20)


def _thin_bridge() -> trimesh.Trimesh:
    x = np.linspace(-1.0, 1.0, 35)
    centers = np.stack((x, np.zeros_like(x), np.zeros_like(x)), axis=1)
    tangents = np.repeat(np.array([[1.0, 0.0, 0.0]]), x.size, axis=0)
    radii = 0.055 + 0.30 * np.power(np.abs(x), 1.5)
    return _tube_mesh(centers, tangents, radii, ring_size=24)


def _concave_pocket() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    radial = np.exp(-((vertices[:, 0] / 0.32) ** 2 + (vertices[:, 2] / 0.32) ** 2))
    upper = np.clip((vertices[:, 1] - 0.2) / 0.8, 0.0, 1.0)
    vertices *= (1.0 - 0.42 * radial * upper)[:, None]
    return _orient_outward(
        trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)
    )


def _axis_angle_transform(angle: float, axis: list[float]) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1.0 - cosine
    rotation = np.array(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _valid_fixtures() -> dict[str, trimesh.Trimesh]:
    sphere = _orient_outward(trimesh.creation.icosphere(subdivisions=2, radius=0.8))
    ellipsoid = sphere.copy()
    ellipsoid.vertices *= np.array([1.0, 0.65, 0.42])
    ellipsoid.apply_transform(_axis_angle_transform(0.71, [0.4, 0.8, -0.2]))
    rigid = ellipsoid.copy()
    rigid.apply_transform(_axis_angle_transform(-0.47, [0.2, -0.5, 0.8]))
    rigid.apply_translation([0.31, -0.22, 0.17])
    fixtures: dict[str, trimesh.Trimesh] = {
        "sphere": sphere,
        "rotated_ellipsoid": ellipsoid,
        "thin_bridge": _thin_bridge(),
        "concave_pocket": _concave_pocket(),
        "rigid_rotation_translation": rigid,
    }
    reference_h = 0.05
    for ratio in (2.0, 1.0, 0.5, 0.25, 0.1):
        fixtures[f"near_contact_gap_{str(ratio).replace('.', '_')}h"] = _hairpin(
            ratio * reference_h
        )
    return fixtures


def _bounds(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.asarray(mesh.bounds, dtype=np.float64)
    padding = max(float(np.max(upper - lower)) * 0.35, 0.25)
    return lower - padding, upper + padding


def _builder_path() -> Path:
    build_root = PROJECT_ROOT / "build/e6_cgal"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT / "tools/e6_cgal"),
            "-B",
            str(build_root),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    subprocess.run(
        ["cmake", "--build", str(build_root), "--parallel", "8"],
        check=True,
        cwd=PROJECT_ROOT,
    )
    executable = build_root / "frayid_e6_field_builder"
    if not executable.is_file():
        raise FileNotFoundError("CGAL E6 builder was not produced")
    return executable


def _run_builder(
    executable: Path,
    mesh: trimesh.Trimesh,
    bounds: tuple[np.ndarray, np.ndarray],
    root: Path,
    name: str,
) -> tuple[InterfaceField, float, str]:
    input_path = root / f"{name}.e6mesh"
    output_path = root / f"{name}.e6field"
    write_interface_mesh(
        input_path,
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        bounds,
    )
    started = time.monotonic()
    completed = subprocess.run(
        [str(executable), str(input_path), str(output_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAXIMUM_FIXTURE_SECONDS,
    )
    elapsed = time.monotonic() - started
    diagnostic = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode:
        raise RuntimeError(f"CGAL builder rejected valid fixture {name}: {diagnostic}")
    field = read_interface_field(output_path)
    if field.vertices.shape[0] > MAXIMUM_PUBLIC_VERTICES:
        raise RuntimeError(f"{name} exceeded the public vertex cap")
    if field.tetrahedra.shape[0] > MAXIMUM_PUBLIC_TETRAHEDRA:
        raise RuntimeError(f"{name} exceeded the public tetrahedron cap")
    return field, elapsed, diagnostic


def _fixture_report(
    executable: Path,
    mesh: trimesh.Trimesh,
    root: Path,
    name: str,
    fixture_index: int,
) -> dict[str, Any]:
    bounds = _bounds(mesh)
    first, first_seconds, first_diagnostic = _run_builder(
        executable, mesh, bounds, root, f"{name}_first"
    )
    second, second_seconds, second_diagnostic = _run_builder(
        executable, mesh, bounds, root, f"{name}_second"
    )
    first_zero = certify_zero_subcomplex(first)
    second_zero = certify_zero_subcomplex(second)
    first_surface = certify_interface_surface(
        first, np.asarray(mesh.vertices), np.asarray(mesh.faces), bounds
    )
    second_surface = certify_interface_surface(
        second, np.asarray(mesh.vertices), np.asarray(mesh.faces), bounds
    )
    signs = certify_independent_signs(
        first,
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        sample_count=SIGN_SAMPLE_COUNT,
        seed=SEED + fixture_index,
    )
    volumes = tetrahedron_volumes(first)
    blockers: list[str] = []
    if first_zero["status"] != "pass" or second_zero["status"] != "pass":
        blockers.append("zero_subcomplex_certificate")
    if first_surface["status"] != "pass" or second_surface["status"] != "pass":
        blockers.append("interface_surface_certificate")
    if signs["status"] != "pass":
        blockers.append("independent_sign_certificate")
    if (
        first_surface["canonical_extraction_sha256"]
        != second_surface["canonical_extraction_sha256"]
    ):
        blockers.append("nondeterministic_canonical_extraction")
    return {
        "name": name,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "source_vertex_count": int(mesh.vertices.shape[0]),
        "source_face_count": int(mesh.faces.shape[0]),
        "field_vertex_count": int(first.vertices.shape[0]),
        "tetrahedron_count": int(first.tetrahedra.shape[0]),
        "inside_cell_count": first.inside_cell_count,
        "outside_cell_count": first.outside_cell_count,
        "minimum_tetrahedron_volume": float(volumes.min()),
        "float64_nonpositive_tetrahedron_count": int(np.count_nonzero(volumes <= 0.0)),
        "exact_tetrahedron_nondegeneracy": "certified_by_cgal_epeck_builder",
        "first_elapsed_seconds": first_seconds,
        "second_elapsed_seconds": second_seconds,
        "first_builder_diagnostic": first_diagnostic,
        "second_builder_diagnostic": second_diagnostic,
        "zero_subcomplex": first_zero,
        "surface": first_surface,
        "signs": signs,
    }


def _invalid_meshes() -> dict[str, tuple[trimesh.Trimesh, str]]:
    first = trimesh.creation.icosphere(subdivisions=1, radius=0.65)
    second = first.copy()
    first.apply_translation([-0.25, 0.0, 0.0])
    second.apply_translation([0.25, 0.0, 0.0])
    vertices = np.concatenate((first.vertices, second.vertices), axis=0)
    faces = np.concatenate((first.faces, second.faces + first.vertices.shape[0]), axis=0)
    intersecting = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    non_manifold = trimesh.creation.icosphere(subdivisions=1, radius=0.7)
    non_manifold.faces = np.concatenate((non_manifold.faces, non_manifold.faces[:1]), axis=0)
    return {
        "global_self_intersection": (intersecting, "self-intersection"),
        "non_manifold_edge": (non_manifold, "non-manifold"),
    }


def _invalid_mesh_report(executable: Path, root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, (mesh, expected) in _invalid_meshes().items():
        input_path = root / f"negative_{name}.e6mesh"
        output_path = root / f"negative_{name}.e6field"
        write_interface_mesh(input_path, mesh.vertices, mesh.faces, _bounds(mesh))
        completed = subprocess.run(
            [str(executable), str(input_path), str(output_path)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=MAXIMUM_FIXTURE_SECONDS,
        )
        diagnostic = (completed.stdout + "\n" + completed.stderr).strip()
        passed = completed.returncode != 0 and expected in diagnostic
        results.append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "expected_diagnostic": expected,
                "return_code": completed.returncode,
                "diagnostic": diagnostic,
            }
        )
    return results


def _field_negative_report(reference: InterfaceField) -> list[dict[str, Any]]:
    outside = np.flatnonzero(
        (reference.cell_regions == 1)
        & ~reference.interface_vertices[reference.tetrahedra].any(axis=1)
    )
    if not outside.size:
        raise RuntimeError("reference fixture has no exterior tetrahedron for negative tests")
    tetrahedron = reference.tetrahedra[int(outside[0])]
    cases = {
        "non_interface_all_zero_edge": tetrahedron[:2],
        "non_interface_all_zero_face": tetrahedron[:3],
        "all_zero_tetrahedron": tetrahedron,
        "extra_zero_shell": tetrahedron,
    }
    results: list[dict[str, Any]] = []
    for name, selected in cases.items():
        mutated = copy.deepcopy(reference)
        values = mutated.values.copy()
        values[selected] = 0.0
        interface_mask = mutated.interface_vertices.copy()
        interface_mask[selected] = True
        mutated = InterfaceField(
            **{
                **mutated.__dict__,
                "values": values,
                "interface_vertices": interface_mask,
            }
        )
        certificate = certify_zero_subcomplex(mutated)
        results.append(
            {
                "name": name,
                "status": "pass" if certificate["status"] == "fail" else "fail",
                "certificate": certificate,
            }
        )
    return results


def run_preflight(source_revision: str) -> dict[str, Any]:
    started = time.monotonic()
    executable = _builder_path()
    fixtures = _valid_fixtures()
    with tempfile.TemporaryDirectory(prefix="frayid-e6-public-") as temporary:
        root = Path(temporary)
        fixture_results: list[dict[str, Any]] = []
        first_field: InterfaceField | None = None
        for index, (name, mesh) in enumerate(fixtures.items()):
            result = _fixture_report(executable, mesh, root, name, index)
            fixture_results.append(result)
            if first_field is None:
                first_field, _, _ = _run_builder(
                    executable, mesh, _bounds(mesh), root, "negative_base"
                )
        if first_field is None:
            raise AssertionError("E6 public fixture registry is empty")
        invalid_mesh_results = _invalid_mesh_report(executable, root)
        field_negative_results = _field_negative_report(first_field)
    blockers = [result["name"] for result in fixture_results if result["status"] != "pass"]
    blockers.extend(
        result["name"]
        for result in (*invalid_mesh_results, *field_negative_results)
        if result["status"] != "pass"
    )
    return {
        "schema_version": "post_v1_e6_public_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "source_revision": source_revision,
        "seed": SEED,
        "toolchain": {
            "library": "CGAL",
            "version": CGAL_VERSION,
            "kernel": "Exact_predicates_exact_constructions_kernel",
            "builder": "tools/e6_cgal/interface_field_builder.cpp",
            "platform": platform.platform(),
        },
        "valid_fixtures": fixture_results,
        "invalid_mesh_fixtures": invalid_mesh_results,
        "invalid_field_fixtures": field_negative_results,
        "elapsed_seconds": time.monotonic() - started,
        "execution": {
            "public_synthetic_only": True,
            "accepted_source_reads": 0,
            "source_images_loaded": 0,
            "optimizer_steps": 0,
            "modal_jobs_launched": 0,
            "development_evaluations": 0,
            "sealed_test_accesses": 0,
            "automatic_paid_retries": 0,
        },
    }


def main() -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            "configs",
            "tools/e6_cgal",
            __file__,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Refusing official E6 public preflight from a dirty relevant worktree")
    result = run_preflight(revision)
    destination = (
        PROJECT_ROOT
        / "outputs/canonical_clothed_surface_v1/post_v1"
        / EXPERIMENT_ID
        / f"public_preflight_{revision[:12]}"
    )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable E6 output: {destination}")
    destination.mkdir(parents=True)
    write_json(destination / "public_preflight_report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
