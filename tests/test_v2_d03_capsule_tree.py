from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from frayid.cli import app
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.d03_capsule_tree import (
    D03_EXPERIMENT_ID,
    Capsule,
    build_d03_real_initialization,
    build_d03_real_initialization_plan,
    capsule_signed_distance,
    capsule_tree_signed_distance,
    offset_field_from_surface_displacement,
    run_d03_public_benchmark,
    write_d03_public_benchmark,
    write_d03_real_initialization_plan,
)


def test_d03_capsule_distance_identity_and_hard_union() -> None:
    capsule = Capsule((0.0, -0.2, 0.0), (0.0, 0.2, 0.0), 0.3, "body")
    points = np.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert np.allclose(capsule_signed_distance(points, capsule), [-0.3, 0.0, 0.2])
    second = Capsule((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.2, "second")
    union = capsule_tree_signed_distance(points, (capsule, second))
    assert np.all(union <= capsule_signed_distance(points, capsule))


def test_d03_field_offset_is_bounded_and_replays_exactly() -> None:
    coordinates = np.linspace(-0.5, 0.5, 24)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    field = np.sqrt(xx**2 + yy**2 + zz**2).astype(np.float32) - 0.25

    surface = np.asarray([[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]], dtype=np.float64)
    displacement = np.asarray([0.01, -0.01], dtype=np.float64)
    first = offset_field_from_surface_displacement(
        field,
        coordinates,
        surface,
        displacement,
        neighbours=2,
        epsilon_metres=0.005,
        update_band_metres=0.12,
    )
    second = offset_field_from_surface_displacement(
        field,
        coordinates,
        surface,
        displacement,
        neighbours=2,
        epsilon_metres=0.005,
        update_band_metres=0.12,
    )
    assert np.array_equal(first, second)
    assert float(np.max(np.abs(first - field))) <= 0.0100001
    assert np.array_equal(first[np.abs(field) >= 0.12], field[np.abs(field) >= 0.12])


def test_d03_public_capsule_tree_is_embedded_and_beats_ellipsoid() -> None:
    pytest.importorskip("ipctk")
    report = run_d03_public_benchmark(resolution=64)
    assert report["status"] == "pass"
    assert report["experiment_id"] == D03_EXPERIMENT_ID
    assert report["treatment_topology"]["exact_self_intersections"] is False
    assert report["treatment_topology"]["euler_number"] == 2
    assert report["truth_error_relative_improvement"] >= 0.20
    assert report["gates"]["topology_changing_proposal_rejected"] is True


def test_d03_public_write_and_cli(tmp_path: Path) -> None:
    pytest.importorskip("ipctk")
    direct = tmp_path / "direct.json"
    write_d03_public_benchmark(direct)
    assert read_json(direct)["status"] == "pass"
    with pytest.raises(FileExistsError, match="immutable"):
        write_d03_public_benchmark(direct)
    cli_output = tmp_path / "cli.json"
    result = CliRunner().invoke(
        app,
        ["v2", "benchmark-d03-capsule-tree", "--output", str(cli_output)],
    )
    assert result.exit_code == 0, result.stdout
    assert read_json(cli_output)["status"] == "pass"


def _write_synthetic_smpl_scaffold(tmp_path: Path) -> tuple[Path, Path]:
    joint_centers = np.stack(
        (
            np.linspace(-0.8, 0.8, 24),
            np.linspace(-1.1, 0.6, 24),
            0.04 * np.sin(np.arange(24)),
        ),
        axis=-1,
    )
    vertices = np.repeat(joint_centers, 2, axis=0)
    vertices[::2, 2] -= 0.04
    vertices[1::2, 2] += 0.04
    weights = np.zeros((48, 24), dtype=np.float32)
    weights[np.arange(48), np.repeat(np.arange(24), 2)] = 1.0
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    scaffold_path = tmp_path / "scaffold.npz"
    weights_path = tmp_path / "weights.npz"
    np.savez_compressed(scaffold_path, vertices=vertices, faces=faces)
    np.savez_compressed(weights_path, weights=weights)
    return scaffold_path, weights_path


def test_d03_real_initialization_plan_is_deterministic_and_immutable(tmp_path: Path) -> None:
    scaffold_path, weights_path = _write_synthetic_smpl_scaffold(tmp_path)
    first = build_d03_real_initialization_plan(
        scaffold_mesh_path=scaffold_path,
        skinning_weights_path=weights_path,
        source_revision="a" * 40,
    )
    second = build_d03_real_initialization_plan(
        scaffold_mesh_path=scaffold_path,
        skinning_weights_path=weights_path,
        source_revision="a" * 40,
    )
    assert first == second
    assert first["source"]["role"] == "geometry_and_rig_prior_only_never_topology_reference"
    assert len(first["capsule_fit"]["capsules"]) == 18
    output = tmp_path / "plan.json"
    write_d03_real_initialization_plan(
        scaffold_mesh_path=scaffold_path,
        skinning_weights_path=weights_path,
        source_revision="a" * 40,
        output_path=output,
    )
    with pytest.raises(FileExistsError, match="immutable"):
        write_d03_real_initialization_plan(
            scaffold_mesh_path=scaffold_path,
            skinning_weights_path=weights_path,
            source_revision="a" * 40,
            output_path=output,
        )


def test_d03_real_initialization_consumes_frozen_plan(tmp_path: Path) -> None:
    pytest.importorskip("ipctk")
    scaffold_path, weights_path = _write_synthetic_smpl_scaffold(tmp_path)
    public_capsules = (
        Capsule((0.0, -0.4, 0.0), (0.0, 0.4, 0.0), 0.25, "torso"),
        Capsule((0.0, 0.3, 0.0), (0.0, 0.6, 0.0), 0.14, "head"),
    )
    capsules = list(public_capsules)
    while len(capsules) < 18:
        index = len(capsules)
        capsules.append(Capsule((0.0, -0.2, 0.0), (0.02 * index, -0.2, 0.0), 0.08, f"part_{index}"))
    plan = {
        "schema_version": "frayid_v2_d03_real_initialization_plan.v1",
        "experiment_id": D03_EXPERIMENT_ID,
        "status": "real_initialization_planned",
        "source_revision": "b" * 40,
        "source": {
            "scaffold_mesh_path": str(scaffold_path),
            "scaffold_mesh_sha256": sha256_file(scaffold_path),
            "skinning_weights_path": str(weights_path),
            "skinning_weights_sha256": sha256_file(weights_path),
            "role": "geometry_and_rig_prior_only_never_topology_reference",
            "vertex_count": 48,
            "face_count": 1,
        },
        "field": {
            "representation": "hard_union_of_closed_capsules",
            "resolution": 64,
            "symmetric_extent_metres": 1.4,
            "surface_level": 0.0,
            "extraction": "skimage_marching_cubes_allow_degenerate_false",
            "cleanup_operations_allowed": 0,
        },
        "capsule_fit": {
            "capsules": [
                {
                    "label": spec[0],
                    "start": list(capsule.start),
                    "end": list(capsule.end),
                    "radius_metres": capsule.radius,
                }
                for spec, capsule in zip(
                    (
                        ("torso",),
                        ("head",),
                        ("left_upper_arm",),
                        ("left_arm",),
                        ("left_forearm",),
                        ("left_hand",),
                        ("right_upper_arm",),
                        ("right_arm",),
                        ("right_forearm",),
                        ("right_hand",),
                        ("left_thigh",),
                        ("left_lower_thigh",),
                        ("left_shin",),
                        ("left_foot",),
                        ("right_thigh",),
                        ("right_lower_thigh",),
                        ("right_shin",),
                        ("right_foot",),
                    ),
                    capsules,
                    strict=True,
                )
            ]
        },
        "rig_transfer": {
            "method": "nearest_scaffold_vertex_skinning_weights",
            "topology_inherited": False,
        },
        "gates": {
            "component_count": 1,
            "euler_number": 2,
            "exact_self_intersections": 0,
            "watertight": True,
            "winding_consistent": True,
            "outward": True,
            "exact_replay": True,
        },
        "provenance": {
            "training_records_read": 0,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
        },
    }
    plan_path = tmp_path / "frozen-plan.json"
    write_json(plan_path, plan)
    output_root = tmp_path / "real-init"
    report_path = build_d03_real_initialization(plan_path=plan_path, output_root=output_root)
    report = read_json(report_path)
    assert report["status"] == "initial_field_qualified"
    assert report["topology"]["exact_self_intersections"] is False
    assert report["surface"]["topology_inherited_from_scaffold"] is False
    with pytest.raises(FileExistsError, match="immutable"):
        build_d03_real_initialization(plan_path=plan_path, output_root=output_root)
