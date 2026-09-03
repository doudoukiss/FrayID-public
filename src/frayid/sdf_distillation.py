from __future__ import annotations

from typing import Any

import numpy as np
import torch
import trimesh
from scipy.ndimage import (  # type: ignore[import-untyped]
    binary_dilation,
    distance_transform_edt,
    map_coordinates,
    maximum_filter,
)
from skimage.measure import marching_cubes
from torch import Tensor, nn

from frayid.geometry import canonical_face_orientation_report, eikonal_loss


def build_topology_safe_sdf_grid(
    mesh: trimesh.Trimesh,
    *,
    longest_axis_resolution: int = 256,
    padding_voxels: int = 4,
    narrow_band_voxels: float = 3.0,
    occupancy_supersampling: int = 3,
    conservative_occupancy_radius: int = 0,
    include_source_surface_support: bool = False,
    query_chunk_size: int = 131_072,
    support_report: dict[str, Any] | None = None,
    support_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a topology-safe SDF using occupancy sign and exact unsigned distance.

    A supersampled, conservatively voxelized and filled mesh supplies only the
    inside/outside sign. A positive ``conservative_occupancy_radius`` applies a
    small high-resolution maximum filter before target-grid sampling, so
    sub-voxel connections are not lost to nearest-sample downsampling. Target-
    grid samples near that occupancy boundary use unsigned closest-triangle
    distance; samples away from the zero set retain a signed EDT value. This
    deliberately avoids near-surface ray or winding signed-distance queries,
    whose sign instability fragmented earlier fields.
    """
    if not mesh.is_watertight:
        raise ValueError("Topology-safe SDF construction requires a watertight source mesh")
    if longest_axis_resolution < 16:
        raise ValueError("Topology-safe SDF resolution must be at least 16")
    if padding_voxels < 3:
        raise ValueError("Topology-safe SDF padding must be at least three voxels")
    if narrow_band_voxels < 1.0:
        raise ValueError("Topology-safe SDF narrow band must be at least one voxel")
    if occupancy_supersampling < 2:
        raise ValueError("Occupancy supersampling must be at least two")
    if conservative_occupancy_radius < 0:
        raise ValueError("Conservative occupancy radius cannot be negative")
    if query_chunk_size <= 0:
        raise ValueError("Closest-distance query chunk size must be positive")
    extent = float(np.max(mesh.extents))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Source mesh has invalid extent")

    pitch = extent / (longest_axis_resolution - 1)
    occupancy_pitch = pitch / occupancy_supersampling
    voxelized = mesh.voxelized(occupancy_pitch).fill()
    linear = np.asarray(voxelized.transform[:3, :3], dtype=np.float64)
    if not np.allclose(linear, np.eye(3) * occupancy_pitch, atol=1e-8):
        raise ValueError("Supersampled occupancy transform is not axis-aligned")
    high_resolution_occupancy = np.asarray(voxelized.matrix, dtype=bool)
    high_resolution_origin = np.asarray(voxelized.transform[:3, 3], dtype=np.float64)
    if conservative_occupancy_radius:
        diameter = 2 * conservative_occupancy_radius + 1
        high_resolution_occupancy = maximum_filter(
            high_resolution_occupancy,
            size=diameter,
            mode="constant",
            cval=False,
        )

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    origin = np.floor(bounds[0] / pitch) * pitch - padding_voxels * pitch
    upper = np.ceil(bounds[1] / pitch) * pitch + padding_voxels * pitch
    shape = np.ceil((upper - origin) / pitch).astype(np.int64) + 1
    target_axes = [origin[axis] + np.arange(shape[axis]) * pitch for axis in range(3)]
    occupancy_indices = [
        np.rint((target_axes[axis] - high_resolution_origin[axis]) / occupancy_pitch).astype(
            np.int64
        )
        for axis in range(3)
    ]
    valid = [
        (occupancy_indices[axis] >= 0)
        & (occupancy_indices[axis] < high_resolution_occupancy.shape[axis])
        for axis in range(3)
    ]
    valid_target_indices = [np.flatnonzero(axis_valid) for axis_valid in valid]
    occupancy = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    occupancy[np.ix_(*valid_target_indices)] = high_resolution_occupancy[
        np.ix_(*[occupancy_indices[axis][valid_target_indices[axis]] for axis in range(3)])
    ]
    if not occupancy.any() or occupancy.all():
        raise ValueError("Supersampled occupancy did not produce both inside and outside samples")

    outside_distance = distance_transform_edt(~occupancy)
    inside_distance = distance_transform_edt(occupancy)
    sdf = ((outside_distance - inside_distance) * pitch).astype(np.float32)
    occupancy_band = np.abs(sdf) <= narrow_band_voxels * pitch
    exact_support = occupancy_band
    source_support_report: dict[str, Any] | None = None
    if include_source_surface_support:
        source_support, source_support_report = build_source_surface_exact_support(
            mesh,
            origin,
            (int(shape[0]), int(shape[1]), int(shape[2])),
            pitch,
            radius_voxels=narrow_band_voxels,
            query_chunk_size=query_chunk_size,
        )
        exact_support = occupancy_band | source_support
    band_indices = np.argwhere(exact_support)
    band_distances = np.empty(band_indices.shape[0], dtype=np.float64)
    for start in range(0, band_indices.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, band_indices.shape[0])
        points = origin + band_indices[start:end] * pitch
        _, distances, _ = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
            mesh, points
        )
        band_distances[start:end] = np.asarray(distances, dtype=np.float64)
    if not np.isfinite(band_distances).all():
        raise ValueError("Unsigned closest-distance query returned non-finite values")
    band_tuple = tuple(band_indices[:, axis] for axis in range(3))
    sdf[band_tuple] = np.where(occupancy[band_tuple], -band_distances, band_distances)
    if not float(sdf.min()) < 0.0 < float(sdf.max()):
        raise ValueError("Topology-safe SDF does not bracket the zero level")
    if support_report is not None:
        support_report.update(
            {
                "schema_version": "topology_safe_exact_support.v1",
                "occupancy_band_node_count": int(np.count_nonzero(occupancy_band)),
                "exact_support_node_count": int(np.count_nonzero(exact_support)),
                "added_source_surface_node_count": int(
                    np.count_nonzero(exact_support & ~occupancy_band)
                ),
                "source_surface_covering_enabled": include_source_surface_support,
                "source_surface": source_support_report,
            }
        )
    if support_masks is not None:
        support_masks.update(
            {
                "occupancy_band": occupancy_band,
                "exact_support": exact_support,
            }
        )
    return sdf, origin.astype(np.float32), pitch


def build_source_surface_exact_support(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    pitch: float,
    *,
    radius_voxels: float = 3.0,
    query_chunk_size: int = 131_072,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return grid nodes whose exact distance to the source is within the radius."""
    if np.asarray(origin).shape != (3,) or len(shape) != 3 or min(shape) < 1:
        raise ValueError("Source-support grid contract is invalid")
    if pitch <= 0 or radius_voxels < 1 or query_chunk_size <= 0:
        raise ValueError("Source-support radius, pitch, and chunk size must be positive")
    grid_origin = np.asarray(origin, dtype=np.float64)
    surface_voxels = mesh.voxelized(pitch)
    surface_points = np.asarray(surface_voxels.points, dtype=np.float64)
    seeds = np.rint((surface_points - grid_origin) / pitch).astype(np.int64)
    valid = np.all((seeds >= 0) & (seeds < np.asarray(shape, dtype=np.int64)), axis=1)
    seeds = np.unique(seeds[valid], axis=0)
    if not len(seeds):
        raise ValueError("Source surface did not intersect the target grid")
    seed_mask = np.zeros(shape, dtype=bool)
    seed_mask[tuple(seeds[:, axis] for axis in range(3))] = True
    dilation_iterations = int(np.ceil(radius_voxels)) + 2
    candidates = binary_dilation(
        seed_mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=dilation_iterations,
    )
    candidate_indices = np.argwhere(candidates)
    distances = np.empty(len(candidate_indices), dtype=np.float64)
    for start in range(0, len(candidate_indices), query_chunk_size):
        end = min(start + query_chunk_size, len(candidate_indices))
        points = grid_origin + candidate_indices[start:end] * pitch
        _, chunk_distances, _ = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
            mesh, points
        )
        distances[start:end] = np.asarray(chunk_distances, dtype=np.float64)
    if not np.isfinite(distances).all():
        raise ValueError("Source-support closest-distance query returned non-finite values")
    selected = distances <= radius_voxels * pitch + 1e-12
    support = np.zeros(shape, dtype=bool)
    selected_indices = candidate_indices[selected]
    support[tuple(selected_indices[:, axis] for axis in range(3))] = True
    return support, {
        "radius_voxels": radius_voxels,
        "surface_voxel_seed_count": len(seeds),
        "candidate_query_count": len(candidate_indices),
        "selected_node_count": int(np.count_nonzero(selected)),
        "dilation_iterations": dilation_iterations,
    }


def trilinear_neighbor_support_report(
    support: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    points: np.ndarray,
) -> dict[str, Any]:
    """Check whether all eight grid neighbors of each point are in exact support."""
    probes = np.asarray(points, dtype=np.float64)
    if support.ndim != 3 or np.asarray(origin).shape != (3,) or pitch <= 0:
        raise ValueError("Trilinear support grid contract is invalid")
    if probes.ndim != 2 or probes.shape[1] != 3:
        raise ValueError("Trilinear support probes must have shape [N, 3]")
    lower = np.floor((probes - np.asarray(origin, dtype=np.float64)) / pitch).astype(np.int64)
    offsets = np.asarray(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64
    )
    neighbors = lower[:, None, :] + offsets[None, :, :]
    in_bounds = np.all(
        (neighbors >= 0) & (neighbors < np.asarray(support.shape, dtype=np.int64)), axis=2
    )
    covered = np.zeros(in_bounds.shape, dtype=bool)
    flat_neighbors = neighbors.reshape(-1, 3)
    flat_valid = in_bounds.reshape(-1)
    valid_neighbors = flat_neighbors[flat_valid]
    covered.reshape(-1)[flat_valid] = support[tuple(valid_neighbors[:, axis] for axis in range(3))]
    probe_covered = np.all(covered & in_bounds, axis=1)
    return {
        "probe_count": len(probes),
        "covered_probe_count": int(np.count_nonzero(probe_covered)),
        "uncovered_probe_count": int(np.count_nonzero(~probe_covered)),
        "all_eight_neighbors_covered": bool(np.all(probe_covered)),
    }


def extract_voxel_sdf_mesh(
    sdf: np.ndarray,
    origin: np.ndarray,
    pitch: float,
) -> trimesh.Trimesh:
    """Extract a canonical mesh from a dense signed-distance grid."""
    if sdf.ndim != 3 or min(sdf.shape) < 2:
        raise ValueError("SDF grid must be a non-empty 3D array")
    if np.asarray(origin).shape != (3,) or pitch <= 0:
        raise ValueError("SDF grid origin and pitch are invalid")
    vertices, faces, _, _ = marching_cubes(  # type: ignore[no-untyped-call]
        np.asarray(sdf, dtype=np.float32),
        level=0.0,
        spacing=(pitch, pitch, pitch),
        allow_degenerate=False,
    )
    vertices += np.asarray(origin, dtype=np.float64)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def extract_topology_constrained_sdf_mesh(
    source_mesh: trimesh.Trimesh,
    sdf: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    *,
    topology_reference_vertices: np.ndarray | None = None,
    iteration_count: int = 12,
    maximum_step_voxels: float = 0.5,
    minimum_signed_area_ratio: float = 0.01,
    minimum_area_ratio: float = 0.1,
    backtracking_steps: int = 16,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Project an accepted source topology onto a grid SDF zero level.

    Marching Cubes is useful for unconstrained diagnostics, but it can create
    handles and isolated shells when close surface sheets occupy neighboring
    voxels. This extractor retains the accepted source connectivity and moves
    only its vertices along trilinearly sampled finite-difference gradients.
    Every global Newton step is backtracked against one immutable topology
    reference until all signed- and unsigned-area floors pass. Callers that
    begin from a previously deformed source must pass its original reference;
    rebasing to ``source_mesh.vertices`` cannot certify cumulative deformation.
    """
    if not source_mesh.is_watertight:
        raise ValueError("Topology-constrained extraction requires a watertight source mesh")
    if sdf.ndim != 3 or min(sdf.shape) < 3:
        raise ValueError("Topology-constrained extraction requires a non-empty 3D SDF")
    if np.asarray(origin).shape != (3,) or pitch <= 0:
        raise ValueError("SDF grid origin and pitch are invalid")
    if iteration_count <= 0 or maximum_step_voxels <= 0 or backtracking_steps <= 0:
        raise ValueError("Projection iteration, step, and backtracking values must be positive")
    if minimum_signed_area_ratio <= 0 or minimum_area_ratio <= 0:
        raise ValueError("Projection signed and unsigned area ratios must be positive")

    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    faces = np.asarray(source_mesh.faces, dtype=np.int64)
    reference_vertices = (
        source_vertices
        if topology_reference_vertices is None
        else np.asarray(topology_reference_vertices, dtype=np.float64)
    )
    if reference_vertices.shape != source_vertices.shape:
        raise ValueError("Projection topology reference must match source vertices [V, 3]")
    reference_triangles = reference_vertices[faces]
    reference_area_vectors = np.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
    )
    reference_area_squared = np.sum(reference_area_vectors**2, axis=1)
    if np.any(reference_area_squared <= 1e-24):
        raise ValueError("Projection topology reference contains a degenerate face")
    reference_area = np.sqrt(reference_area_squared)

    def topology_floors(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidate_triangles = candidate[faces]
        candidate_area_vectors = np.cross(
            candidate_triangles[:, 1] - candidate_triangles[:, 0],
            candidate_triangles[:, 2] - candidate_triangles[:, 0],
        )
        signed = np.sum(reference_area_vectors * candidate_area_vectors, axis=1) / (
            reference_area_squared
        )
        unsigned = np.linalg.norm(candidate_area_vectors, axis=1) / reference_area
        return signed, unsigned

    source_signed, source_unsigned = topology_floors(source_vertices)
    if np.any(source_signed < minimum_signed_area_ratio) or np.any(
        source_unsigned < minimum_area_ratio
    ):
        raise ValueError("Projection source already violates the immutable topology reference")

    vertices = source_vertices.copy()
    grid_origin = np.asarray(origin, dtype=np.float64)
    field = np.asarray(sdf, dtype=np.float32)
    gradients = [
        np.asarray(component, dtype=np.float32)
        for component in np.gradient(field, pitch, edge_order=2)
    ]

    def sample(values: np.ndarray, points: np.ndarray) -> np.ndarray:
        coordinates = ((points - grid_origin) / pitch).T
        return np.asarray(
            map_coordinates(values, coordinates, order=1, mode="nearest"),
            dtype=np.float64,
        )

    history: list[dict[str, float | int]] = []
    for iteration in range(iteration_count):
        values = sample(field, vertices)
        sampled_gradients = np.column_stack(
            [sample(component, vertices) for component in gradients]
        )
        squared_norms = np.maximum(np.sum(sampled_gradients**2, axis=1), 1e-8)
        update = -(values / squared_norms)[:, None] * sampled_gradients
        update_lengths = np.linalg.norm(update, axis=1)
        maximum_step = maximum_step_voxels * pitch
        update *= np.minimum(1.0, maximum_step / np.maximum(update_lengths, 1e-12))[:, None]

        accepted_scale = 1.0
        accepted_vertices: np.ndarray | None = None
        for _ in range(backtracking_steps + 1):
            candidate = vertices + accepted_scale * update
            signed_ratios, unsigned_ratios = topology_floors(candidate)
            if np.all(signed_ratios >= minimum_signed_area_ratio) and np.all(
                unsigned_ratios >= minimum_area_ratio
            ):
                accepted_vertices = candidate
                break
            accepted_scale *= 0.5
        if accepted_vertices is None:
            accepted_scale = 0.0
        else:
            vertices = accepted_vertices
        history.append(
            {
                "iteration": iteration,
                "median_absolute_sdf_m": float(np.median(np.abs(values))),
                "maximum_absolute_sdf_m": float(np.max(np.abs(values))),
                "accepted_step_scale": accepted_scale,
            }
        )
        if accepted_scale == 0.0:
            break

    final_values = sample(field, vertices)
    topology_report = canonical_face_orientation_report(
        reference_vertices,
        vertices,
        faces,
        minimum_area_ratio=minimum_area_ratio,
    )
    final_signed, final_unsigned = topology_floors(vertices)
    signed_floor_violations = int(np.count_nonzero(final_signed < minimum_signed_area_ratio))
    unsigned_floor_violations = int(np.count_nonzero(final_unsigned < minimum_area_ratio))
    topology_blockers = [str(value) for value in topology_report["blockers"]]
    if signed_floor_violations:
        topology_blockers.append("canonical_signed_area_floor")
    if unsigned_floor_violations and "canonical_face_area_collapse" not in topology_blockers:
        topology_blockers.append("canonical_unsigned_area_floor")
    topology_report.update(
        {
            "status": "pass" if not topology_blockers else "fail",
            "minimum_signed_area_ratio": float(np.min(final_signed)),
            "minimum_unsigned_area_ratio": float(np.min(final_unsigned)),
            "signed_area_floor_violation_count": signed_floor_violations,
            "unsigned_area_floor_violation_count": unsigned_floor_violations,
            "blockers": topology_blockers,
        }
    )
    source_relative_topology = canonical_face_orientation_report(
        source_vertices,
        vertices,
        faces,
        minimum_area_ratio=minimum_area_ratio,
    )
    extracted = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    diagnostics: dict[str, Any] = {
        "schema_version": "topology_constrained_sdf_projection.v2",
        "iteration_count": len(history),
        "maximum_step_voxels": maximum_step_voxels,
        "minimum_signed_area_ratio": minimum_signed_area_ratio,
        "minimum_area_ratio": minimum_area_ratio,
        "backtracking_steps": backtracking_steps,
        "median_absolute_sdf_m": float(np.median(np.abs(final_values))),
        "maximum_absolute_sdf_m": float(np.max(np.abs(final_values))),
        "topology": topology_report,
        "source_relative_topology": source_relative_topology,
        "topology_reference": (
            "source_vertices" if topology_reference_vertices is None else "external_original"
        ),
        "history": history,
    }
    return extracted, diagnostics


def _interpolated_vertex_normals(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_indices: np.ndarray,
) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)[face_indices]
    barycentric = np.asarray(
        trimesh.triangles.points_to_barycentric(  # type: ignore[no-untyped-call]
            triangles, points
        ),
        dtype=np.float64,
    )
    corner_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)[
        np.asarray(mesh.faces, dtype=np.int64)[face_indices]
    ]
    normals = np.sum(corner_normals * barycentric[:, :, None], axis=1)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("Surface-normal interpolation produced a zero vector")
    return np.asarray(normals / lengths, dtype=np.float64)


def _directional_surface_fidelity(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, float]:
    points, source_faces = trimesh.sample.sample_surface(
        source,
        sample_count,
        seed=seed,
    )
    closest, distances, target_faces = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
        target, points
    )
    distances = np.asarray(distances, dtype=np.float64)
    target_faces = np.asarray(target_faces, dtype=np.int64)
    if not np.isfinite(distances).all() or np.any(target_faces < 0):
        raise ValueError("Surface fidelity query returned invalid closest points")
    source_normals = _interpolated_vertex_normals(
        source,
        np.asarray(points, dtype=np.float64),
        np.asarray(source_faces, dtype=np.int64),
    )
    target_normals = _interpolated_vertex_normals(
        target,
        np.asarray(closest, dtype=np.float64),
        target_faces,
    )
    cosine = np.clip(np.sum(source_normals * target_normals, axis=1), -1.0, 1.0)
    angles = np.rad2deg(np.arccos(cosine))
    return {
        "mean_distance_m": float(np.mean(distances)),
        "median_distance_m": float(np.median(distances)),
        "p95_distance_m": float(np.quantile(distances, 0.95)),
        "median_normal_error_degrees": float(np.median(angles)),
        "p95_normal_error_degrees": float(np.quantile(angles, 0.95)),
    }


def evaluate_topology_safe_sdf_fidelity(
    source: trimesh.Trimesh,
    extracted: trimesh.Trimesh,
    sdf: np.ndarray,
    pitch: float,
    *,
    sample_count: int = 50_000,
    seed: int = 42,
    maximum_directional_chamfer_voxels: float = 0.5,
    maximum_median_normal_error_degrees: float = 5.0,
    maximum_relative_volume_error: float = 0.03,
) -> dict[str, Any]:
    """Measure and apply the fixed local topology-safe SDF fidelity gate."""
    if sample_count <= 0 or pitch <= 0:
        raise ValueError("Fidelity sample count and grid pitch must be positive")
    if sdf.ndim != 3 or min(sdf.shape) < 3:
        raise ValueError("Fidelity evaluation requires a non-empty 3D SDF")
    source_to_extracted = _directional_surface_fidelity(
        source,
        extracted,
        sample_count=sample_count,
        seed=seed,
    )
    extracted_to_source = _directional_surface_fidelity(
        extracted,
        source,
        sample_count=sample_count,
        seed=seed + 1,
    )
    source_to_extracted["mean_distance_voxels"] = source_to_extracted["mean_distance_m"] / pitch
    extracted_to_source["mean_distance_voxels"] = extracted_to_source["mean_distance_m"] / pitch
    source_volume = abs(float(source.volume))
    if source_volume <= 0 or not np.isfinite(source_volume):
        raise ValueError("Source mesh has invalid enclosed volume")
    relative_volume_error = abs(abs(float(extracted.volume)) - source_volume) / source_volume
    components = extracted.split(only_watertight=False)
    boundary_values = np.concatenate(
        (
            sdf[0, :, :].ravel(),
            sdf[-1, :, :].ravel(),
            sdf[:, 0, :].ravel(),
            sdf[:, -1, :].ravel(),
            sdf[:, :, 0].ravel(),
            sdf[:, :, -1].ravel(),
        )
    )
    blockers: list[str] = []
    if source_to_extracted["mean_distance_voxels"] > maximum_directional_chamfer_voxels:
        blockers.append("source_to_extracted_chamfer_above_gate")
    if extracted_to_source["mean_distance_voxels"] > maximum_directional_chamfer_voxels:
        blockers.append("extracted_to_source_chamfer_above_gate")
    maximum_directional_normal = max(
        source_to_extracted["median_normal_error_degrees"],
        extracted_to_source["median_normal_error_degrees"],
    )
    if maximum_directional_normal > maximum_median_normal_error_degrees:
        blockers.append("median_normal_error_above_gate")
    if relative_volume_error > maximum_relative_volume_error:
        blockers.append("relative_volume_error_above_gate")
    if not extracted.is_watertight:
        blockers.append("extracted_mesh_not_watertight")
    if len(components) != 1:
        blockers.append("extracted_mesh_component_count_not_one")
    if np.any(boundary_values <= 0.0):
        blockers.append("zero_level_touches_grid_boundary")
    return {
        "schema_version": "topology_safe_sdf_fidelity.v1",
        "status": "pass" if not blockers else "fail",
        "sample_count_per_direction": sample_count,
        "pitch_m": pitch,
        "source_to_extracted": source_to_extracted,
        "extracted_to_source": extracted_to_source,
        "maximum_directional_median_normal_error_degrees": maximum_directional_normal,
        "relative_volume_error": relative_volume_error,
        "component_count": len(components),
        "watertight": bool(extracted.is_watertight),
        "minimum_boundary_sdf_m": float(np.min(boundary_values)),
        "thresholds": {
            "maximum_directional_chamfer_voxels": maximum_directional_chamfer_voxels,
            "maximum_median_normal_error_degrees": maximum_median_normal_error_degrees,
            "maximum_relative_volume_error": maximum_relative_volume_error,
        },
        "blockers": blockers,
    }


def mesh_signed_distances(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Return the project SDF convention: negative inside, positive outside."""
    values = np.asarray(
        -trimesh.proximity.signed_distance(  # type: ignore[no-untyped-call]
            mesh, np.asarray(points, dtype=np.float64)
        ),
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("Mesh signed-distance query returned non-finite values")
    return values


def build_signed_distance_samples(
    mesh: trimesh.Trimesh,
    *,
    global_count: int,
    surface_count: int,
    seed: int,
    margin: float = 0.15,
    maximum_surface_offset: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Create balanced global and near-surface supervision for SDF distillation."""
    if global_count <= 0 or surface_count <= 0:
        raise ValueError("SDF sample counts must be positive")
    if margin <= 0 or maximum_surface_offset <= 0:
        raise ValueError("SDF sampling distances must be positive")
    if not mesh.is_watertight:
        raise ValueError("SDF distillation requires a watertight source mesh")
    generator = np.random.default_rng(seed)
    low = mesh.bounds[0] - margin
    high = mesh.bounds[1] + margin
    global_points = generator.uniform(low, high, size=(global_count, 3))
    surface_points, face_indices = trimesh.sample.sample_surface(mesh, surface_count, seed=seed)
    surface_normals = mesh.face_normals[face_indices]
    offsets = generator.uniform(0.002, maximum_surface_offset, size=(surface_count, 1))
    near_surface_points = np.concatenate(
        (
            surface_points,
            surface_points + surface_normals * offsets,
            surface_points - surface_normals * offsets,
        ),
        axis=0,
    )
    points = np.concatenate((global_points, near_surface_points), axis=0)
    targets = mesh_signed_distances(mesh, points)
    order = generator.permutation(points.shape[0])
    return points[order].astype(np.float32), targets[order]


def distill_sdf(
    sdf: nn.Module,
    points: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    steps: int = 2000,
    batch_size: int = 8192,
    learning_rate: float = 5e-4,
    eikonal_weight: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit an implicit field to exact mesh distances without changing deformation."""
    if points.shape != (targets.shape[0], 3):
        raise ValueError("SDF points and targets have incompatible shapes")
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("SDF optimizer parameters must be positive")
    torch.manual_seed(seed)
    point_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    validation_count = max(1024, point_tensor.shape[0] // 10)
    validation_points = point_tensor[-validation_count:]
    validation_targets = target_tensor[-validation_count:]
    train_points = point_tensor[:-validation_count]
    train_targets = target_tensor[:-validation_count]
    optimizer = torch.optim.Adam(sdf.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    sdf.train()
    for step in range(steps):
        indices = torch.randint(0, train_points.shape[0], (batch_size,), device=device)
        batch_points = train_points[indices]
        batch_targets = train_targets[indices]
        optimizer.zero_grad(set_to_none=True)
        prediction = sdf(batch_points).reshape(-1)
        regression: Tensor = torch.nn.functional.smooth_l1_loss(
            prediction, batch_targets, beta=0.01
        )
        eikonal = eikonal_loss(sdf, batch_points[:512])
        loss: Tensor = regression + eikonal_weight * eikonal
        if not torch.isfinite(loss):
            raise RuntimeError("SDF distillation produced a non-finite loss")
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % 250 == 0:
            history.append(
                {
                    "step": float(step),
                    "total": float(loss.detach().cpu()),
                    "regression": float(regression.detach().cpu()),
                    "eikonal": float(eikonal.detach().cpu()),
                }
            )
    sdf.eval()
    predictions: list[Tensor] = []
    with torch.no_grad():
        for chunk in validation_points.split(65_536):  # type: ignore[no-untyped-call]
            predictions.append(sdf(chunk).reshape(-1))
    predicted = torch.cat(predictions)
    absolute_error = (predicted - validation_targets).abs()
    sign_mask = validation_targets.abs() >= 0.002
    sign_accuracy = (
        ((predicted[sign_mask] < 0) == (validation_targets[sign_mask] < 0)).float().mean()
    )
    return {
        "sample_count": int(point_tensor.shape[0]),
        "train_sample_count": int(train_points.shape[0]),
        "validation_sample_count": int(validation_count),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "eikonal_weight": eikonal_weight,
        "validation_mae": float(absolute_error.mean().cpu()),
        "validation_p95_absolute_error": float(torch.quantile(absolute_error, 0.95).cpu()),
        "validation_sign_accuracy": float(sign_accuracy.cpu()),
        "history": history,
    }
