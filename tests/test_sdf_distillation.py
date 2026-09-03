from __future__ import annotations

import numpy as np
import pytest
import trimesh
from skimage.measure import marching_cubes

from frayid.sdf_distillation import (
    build_signed_distance_samples,
    build_source_surface_exact_support,
    build_topology_safe_sdf_grid,
    evaluate_topology_safe_sdf_fidelity,
    extract_topology_constrained_sdf_mesh,
    extract_voxel_sdf_mesh,
    mesh_signed_distances,
    trilinear_neighbor_support_report,
)


def test_mesh_signed_distance_uses_negative_inside_convention() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    values = mesh_signed_distances(
        mesh,
        np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
    )
    assert values[0] < 0
    assert values[1] > 0


def test_signed_distance_samples_cover_both_sides_of_surface() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    points, targets = build_signed_distance_samples(
        mesh,
        global_count=64,
        surface_count=64,
        seed=7,
    )
    assert points.shape == (256, 3)
    assert targets.shape == (256,)
    assert float(targets.min()) < -0.01
    assert float(targets.max()) > 0.01
    assert np.isfinite(targets).all()
    assert np.median(np.linalg.norm(points, axis=1)) == pytest.approx(0.75, abs=0.4)


def _assert_topology_safe_fidelity(source: trimesh.Trimesh) -> None:
    sdf, origin, pitch = build_topology_safe_sdf_grid(
        source,
        longest_axis_resolution=48,
        occupancy_supersampling=4,
    )
    extracted, projection = extract_topology_constrained_sdf_mesh(source, sdf, origin, pitch)
    repeated, repeated_projection = extract_topology_constrained_sdf_mesh(
        source, sdf, origin, pitch
    )
    report = evaluate_topology_safe_sdf_fidelity(
        source,
        extracted,
        sdf,
        pitch,
        sample_count=12_000,
        seed=17,
    )
    assert sdf.ndim == 3
    assert float(sdf.min()) < 0 < float(sdf.max())
    assert report["status"] == "pass", report["blockers"]
    assert report["source_to_extracted"]["mean_distance_voxels"] < 0.5
    assert report["extracted_to_source"]["mean_distance_voxels"] < 0.5
    assert report["maximum_directional_median_normal_error_degrees"] <= 5.0
    assert report["relative_volume_error"] <= 0.03
    assert extracted.is_watertight
    assert len(extracted.split(only_watertight=False)) == 1
    assert extracted.euler_number == source.euler_number
    np.testing.assert_array_equal(extracted.faces, source.faces)
    assert projection["topology"]["status"] == "pass"
    assert projection == repeated_projection
    np.testing.assert_array_equal(extracted.vertices, repeated.vertices)
    np.testing.assert_array_equal(extracted.faces, repeated.faces)


def test_topology_safe_sdf_preserves_sphere_surface() -> None:
    _assert_topology_safe_fidelity(trimesh.creation.icosphere(subdivisions=3, radius=0.75))


def test_topology_safe_sdf_preserves_rotated_ellipsoid_surface() -> None:
    source = trimesh.creation.icosphere(subdivisions=3, radius=0.75)
    source.apply_scale([0.7, 1.0, 1.4])
    source.apply_transform(
        trimesh.transformations.rotation_matrix(
            np.deg2rad(31.0),
            [1.0, 0.4, 0.2],
        )
    )
    _assert_topology_safe_fidelity(source)


def test_topology_safe_projection_enforces_external_original_reference() -> None:
    original = trimesh.creation.icosphere(subdivisions=2, radius=0.75)
    source = original.copy()
    source.apply_scale([0.96, 1.02, 1.01])
    sdf, origin, pitch = build_topology_safe_sdf_grid(
        source,
        longest_axis_resolution=40,
        occupancy_supersampling=3,
    )
    extracted, projection = extract_topology_constrained_sdf_mesh(
        source,
        sdf,
        origin,
        pitch,
        topology_reference_vertices=np.asarray(original.vertices),
        minimum_signed_area_ratio=0.01,
        minimum_area_ratio=0.1,
    )
    assert projection["schema_version"] == "topology_constrained_sdf_projection.v2"
    assert projection["topology_reference"] == "external_original"
    assert projection["topology"]["status"] == "pass"
    assert projection["topology"]["minimum_signed_area_ratio"] >= 0.01
    assert projection["topology"]["minimum_unsigned_area_ratio"] >= 0.1
    np.testing.assert_array_equal(extracted.faces, source.faces)


def test_topology_safe_projection_rejects_mismatched_external_reference() -> None:
    source = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    sdf, origin, pitch = build_topology_safe_sdf_grid(
        source,
        longest_axis_resolution=32,
        occupancy_supersampling=3,
    )
    with pytest.raises(ValueError, match="reference must match"):
        extract_topology_constrained_sdf_mesh(
            source,
            sdf,
            origin,
            pitch,
            topology_reference_vertices=np.asarray(source.vertices[:-1]),
        )


def test_topology_safe_projection_records_stricter_internal_signed_guard() -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.75)
    sdf, origin, pitch = build_topology_safe_sdf_grid(
        source,
        longest_axis_resolution=40,
        occupancy_supersampling=3,
    )
    _, projection = extract_topology_constrained_sdf_mesh(
        source,
        sdf,
        origin,
        pitch,
        topology_reference_vertices=np.asarray(source.vertices),
        minimum_signed_area_ratio=0.01001,
        minimum_area_ratio=0.1,
    )
    assert projection["minimum_signed_area_ratio"] == 0.01001
    assert projection["topology"]["minimum_signed_area_ratio"] >= 0.01001
    assert projection["topology"]["signed_area_floor_violation_count"] == 0


@pytest.mark.parametrize("translation", [0.0, 0.37])
def test_source_surface_support_covers_all_trilinear_neighbors(translation: float) -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.75)
    source.apply_scale([0.7, 1.0, 1.35])
    source.apply_transform(
        trimesh.transformations.rotation_matrix(np.deg2rad(29.0), [0.4, 1.0, 0.2])
    )
    source.apply_translation([translation * 0.03, -translation * 0.02, translation * 0.01])
    extent = float(np.max(source.extents))
    pitch = extent / 39
    origin = np.floor(source.bounds[0] / pitch) * pitch - 4 * pitch
    upper = np.ceil(source.bounds[1] / pitch) * pitch + 4 * pitch
    shape = tuple((np.ceil((upper - origin) / pitch).astype(np.int64) + 1).tolist())
    support, diagnostics = build_source_surface_exact_support(
        source, origin, shape, pitch, radius_voxels=3.0
    )
    probes, _ = trimesh.sample.sample_surface(source, 4_000, seed=20260831)
    coverage = trilinear_neighbor_support_report(support, origin, pitch, probes)
    assert diagnostics["selected_node_count"] > 0
    assert coverage["all_eight_neighbors_covered"]
    assert coverage["uncovered_probe_count"] == 0


def test_source_surface_union_is_opt_in_and_reports_added_support() -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.75)
    old_report: dict[str, object] = {}
    new_report: dict[str, object] = {}
    old_sdf, old_origin, old_pitch = build_topology_safe_sdf_grid(
        source,
        longest_axis_resolution=40,
        occupancy_supersampling=3,
        support_report=old_report,
    )
    new_sdf, new_origin, new_pitch = build_topology_safe_sdf_grid(
        source,
        longest_axis_resolution=40,
        occupancy_supersampling=3,
        include_source_surface_support=True,
        support_report=new_report,
    )
    np.testing.assert_array_equal(old_origin, new_origin)
    assert old_pitch == new_pitch
    assert old_sdf.shape == new_sdf.shape
    assert old_report["source_surface_covering_enabled"] is False
    assert new_report["source_surface_covering_enabled"] is True
    assert int(new_report["exact_support_node_count"]) >= int(
        old_report["exact_support_node_count"]
    )


def test_support_masks_expose_the_exact_frozen_membership() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.7)
    report: dict[str, object] = {}
    masks: dict[str, np.ndarray] = {}
    build_topology_safe_sdf_grid(
        mesh,
        longest_axis_resolution=24,
        include_source_surface_support=True,
        support_report=report,
        support_masks=masks,
    )
    assert masks["occupancy_band"].dtype == np.bool_
    assert masks["exact_support"].shape == masks["occupancy_band"].shape
    assert np.all(masks["exact_support"] | ~masks["occupancy_band"])
    assert int(np.count_nonzero(masks["exact_support"])) == report["exact_support_node_count"]


def test_topology_safe_sdf_rejects_negative_conservative_radius() -> None:
    with pytest.raises(ValueError, match="Conservative occupancy radius"):
        build_topology_safe_sdf_grid(
            trimesh.creation.icosphere(subdivisions=1),
            longest_axis_resolution=32,
            conservative_occupancy_radius=-1,
        )


def test_conservative_occupancy_preserves_a_subvoxel_connection() -> None:
    axis = np.linspace(-1.2, 1.2, 64)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    left = np.sqrt((xx + 0.55) ** 2 + yy**2 + zz**2) - 0.42
    right = np.sqrt((xx - 0.55) ** 2 + yy**2 + zz**2) - 0.42
    thin_bridge = np.maximum(np.sqrt(yy**2 + zz**2) - 0.03, np.abs(xx) - 0.55)
    implicit = np.minimum(np.minimum(left, right), thin_bridge)
    vertices, faces, _, _ = marching_cubes(
        implicit,
        level=0.0,
        spacing=(axis[1] - axis[0],) * 3,
    )
    vertices += axis[0]
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    source.apply_transform(
        trimesh.transformations.rotation_matrix(
            np.deg2rad(41.0),
            [0.3, 1.0, 0.2],
        )
    )
    fragmented_sdf, origin, pitch = build_topology_safe_sdf_grid(
        source,
        longest_axis_resolution=24,
        occupancy_supersampling=3,
        conservative_occupancy_radius=0,
    )
    fragmented = extract_voxel_sdf_mesh(fragmented_sdf, origin, pitch)
    repaired_sdf, origin, pitch = build_topology_safe_sdf_grid(
        source,
        longest_axis_resolution=24,
        occupancy_supersampling=3,
        conservative_occupancy_radius=1,
    )
    repaired = extract_voxel_sdf_mesh(repaired_sdf, origin, pitch)
    assert len(fragmented.split(only_watertight=False)) > 1
    assert repaired.is_watertight
    assert len(repaired.split(only_watertight=False)) == 1
    assert int(repaired.euler_number) == 2
