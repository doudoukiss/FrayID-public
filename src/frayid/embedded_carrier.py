from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

EMBEDDED_CARRIER_SCHEMA = "embedded_carrier.v1"


def read_e10_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the strict canonicalized mesh interchange emitted by the GPL tool."""
    with path.open("r", encoding="utf-8") as stream:
        if stream.readline().split() != ["FRAYID_E10_MESH", "1"]:
            raise ValueError("unsupported E10 mesh format")
        counts = [int(value) for value in stream.readline().split()]
        if len(counts) != 2 or min(counts) <= 0:
            raise ValueError("invalid E10 mesh counts")
        vertex_count, face_count = counts
        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        faces = np.empty((face_count, 3), dtype=np.int64)
        for index in range(vertex_count):
            record = stream.readline().split()
            if len(record) != 3:
                raise ValueError("truncated E10 vertex record")
            vertices[index] = [float(value) for value in record]
        for index in range(face_count):
            record = stream.readline().split()
            if len(record) != 3:
                raise ValueError("truncated E10 face record")
            faces[index] = [int(value) for value in record]
        if stream.read().strip():
            raise ValueError("E10 mesh contains trailing records")
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= vertex_count):
        raise ValueError("E10 mesh contains invalid arrays")
    return vertices, faces


@dataclass(frozen=True)
class CarrierTransfer:
    """Deterministic source-triangle map for a new-connectivity carrier."""

    source_face_indices: np.ndarray
    source_barycentrics: np.ndarray
    weights: np.ndarray
    distances: np.ndarray
    ambiguous: np.ndarray

    def validate(self, *, vertex_count: int, source_face_count: int) -> None:
        if self.source_face_indices.shape != (vertex_count,):
            raise ValueError("source face indices must have shape [V]")
        if self.source_barycentrics.shape != (vertex_count, 3):
            raise ValueError("source barycentrics must have shape [V, 3]")
        if self.weights.ndim != 2 or self.weights.shape[0] != vertex_count:
            raise ValueError("transferred weights must have shape [V, J]")
        if self.distances.shape != (vertex_count,) or self.ambiguous.shape != (vertex_count,):
            raise ValueError("transfer diagnostics must have shape [V]")
        if np.any(self.source_face_indices < 0) or np.any(
            self.source_face_indices >= source_face_count
        ):
            raise ValueError("source face index is outside the source mesh")
        if not all(
            np.isfinite(value).all()
            for value in (self.source_barycentrics, self.weights, self.distances)
        ):
            raise ValueError("carrier transfer contains non-finite values")
        if float(np.min(self.source_barycentrics, initial=0.0)) < -1e-8:
            raise ValueError("source barycentrics must be nonnegative")
        if not np.allclose(self.source_barycentrics.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("source barycentrics must sum to one")
        if float(np.min(self.weights, initial=0.0)) < -1e-7:
            raise ValueError("transferred skinning weights must be nonnegative")
        if not np.allclose(self.weights.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("transferred skinning weights must sum to one")


@dataclass(frozen=True)
class EmbeddedCarrierResult:
    """Structured E10 construction/transfer result with no implicit fallback."""

    status: str
    alpha_over_pitch: float
    offset_over_alpha: float
    pitch: float
    input_vertex_count: int
    input_face_count: int
    output_vertex_count: int = 0
    output_face_count: int = 0
    constructor: str = "CGAL_6.2_alpha_wrap_3_EPICK"
    exact_auditor: str = "CGAL_6.2_EPECK"
    elapsed_seconds: float = 0.0
    deterministic_repeat: bool = False
    exact_audit: dict[str, Any] = field(default_factory=dict)
    containment: dict[str, Any] = field(default_factory=dict)
    fidelity: dict[str, Any] = field(default_factory=dict)
    transfer: dict[str, Any] = field(default_factory=dict)
    posed_fidelity: dict[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "embedded_carrier_result.v1", **asdict(self)}


def _closest_points_for_candidates(
    point: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    repeated = np.repeat(point[None, :], len(triangles), axis=0)
    closest = trimesh.triangles.closest_point(triangles, repeated)  # type: ignore[no-untyped-call]
    distances = np.linalg.norm(closest - point[None, :], axis=1)
    return closest, distances


def build_barycentric_transfer(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_weights: np.ndarray,
    target_vertices: np.ndarray,
    *,
    target_normals: np.ndarray | None = None,
    source_residual_trajectories: np.ndarray | None = None,
    pitch: float,
    ambiguity_distance_fraction: float = 0.05,
    ambiguity_normal_tolerance: float = 0.05,
    ambiguity_weight_l1: float = 0.25,
    ambiguity_residual_rms_pitch: float = 0.25,
) -> CarrierTransfer:
    """Map target vertices to closest same-side source triangles with stable ties."""
    if pitch <= 0:
        raise ValueError("pitch must be positive")
    source_points = np.asarray(source_vertices, dtype=np.float64)
    source_triangles = np.asarray(source_faces, dtype=np.int64)
    weights = np.asarray(source_weights, dtype=np.float64)
    targets = np.asarray(target_vertices, dtype=np.float64)
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source vertices must have shape [V, 3]")
    if source_triangles.ndim != 2 or source_triangles.shape[1] != 3:
        raise ValueError("source faces must have shape [F, 3]")
    if weights.ndim != 2 or weights.shape[0] != source_points.shape[0]:
        raise ValueError("source weights must have shape [V, J]")
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target vertices must have shape [V, 3]")
    normals = None if target_normals is None else np.asarray(target_normals, dtype=np.float64)
    if normals is not None and normals.shape != targets.shape:
        raise ValueError("target normals must match target vertices")
    residuals = (
        None
        if source_residual_trajectories is None
        else np.asarray(source_residual_trajectories, dtype=np.float64)
    )
    if residuals is not None and (
        residuals.ndim != 3
        or residuals.shape[1:] != (len(source_points), 3)
        or not np.isfinite(residuals).all()
    ):
        raise ValueError("source residual trajectories must have shape [T, V, 3]")
    source_mesh = trimesh.Trimesh(vertices=source_points, faces=source_triangles, process=False)
    candidate_sets = trimesh.proximity.nearby_faces(source_mesh, targets)
    selected_faces = np.empty(len(targets), dtype=np.int64)
    barycentrics = np.empty((len(targets), 3), dtype=np.float64)
    distances = np.empty(len(targets), dtype=np.float64)
    ambiguous = np.zeros(len(targets), dtype=np.bool_)
    triangle_vertices = source_points[source_triangles]
    face_normals = np.asarray(source_mesh.face_normals, dtype=np.float64)
    for index, (point, candidate_values) in enumerate(zip(targets, candidate_sets, strict=True)):
        candidates = np.unique(np.asarray(candidate_values, dtype=np.int64))
        if candidates.size == 0:
            raise ValueError("source proximity query returned no candidate face")
        closest, candidate_distances = _closest_points_for_candidates(
            point, triangle_vertices[candidates]
        )
        alignments = (
            np.ones(len(candidates), dtype=np.float64)
            if normals is None
            else face_normals[candidates] @ normals[index]
        )
        same_side = alignments >= 0.0
        eligible = np.flatnonzero(same_side)
        if not eligible.size:
            raise ValueError("target vertex has no same-side source triangle")
        order = np.lexsort((candidates[eligible], candidate_distances[eligible]))
        chosen_local = int(eligible[order[0]])
        chosen_face = int(candidates[chosen_local])
        chosen_point = closest[chosen_local]
        chosen_distance = float(candidate_distances[chosen_local])
        chosen_barycentric = trimesh.triangles.points_to_barycentric(  # type: ignore[no-untyped-call]
            triangle_vertices[[chosen_face]], chosen_point[None, :]
        )[0]
        chosen_barycentric = np.maximum(chosen_barycentric, 0.0)
        chosen_barycentric /= chosen_barycentric.sum()
        selected_faces[index] = chosen_face
        barycentrics[index] = chosen_barycentric
        distances[index] = chosen_distance
        near = np.flatnonzero(
            same_side
            & (candidate_distances <= chosen_distance + ambiguity_distance_fraction * pitch)
            & (alignments >= alignments[chosen_local] - ambiguity_normal_tolerance)
        )
        chosen_weights = (weights[source_triangles[chosen_face]] * chosen_barycentric[:, None]).sum(
            axis=0
        )
        chosen_residuals = (
            None
            if residuals is None
            else (
                residuals[:, source_triangles[chosen_face]] * chosen_barycentric[None, :, None]
            ).sum(axis=1)
        )
        for other_local in near:
            other_face = int(candidates[other_local])
            if other_face == chosen_face:
                continue
            other_barycentric = trimesh.triangles.points_to_barycentric(  # type: ignore[no-untyped-call]
                triangle_vertices[[other_face]], closest[other_local][None, :]
            )[0]
            other_weights = (
                weights[source_triangles[other_face]] * other_barycentric[:, None]
            ).sum(axis=0)
            if float(np.abs(chosen_weights - other_weights).sum()) > ambiguity_weight_l1:
                ambiguous[index] = True
                break
            if residuals is not None and chosen_residuals is not None:
                other_residuals = (
                    residuals[:, source_triangles[other_face]] * other_barycentric[None, :, None]
                ).sum(axis=1)
                residual_rms = float(
                    np.sqrt(np.mean(np.square(chosen_residuals - other_residuals)) + 1e-12)
                )
                if residual_rms > ambiguity_residual_rms_pitch * pitch:
                    ambiguous[index] = True
                    break
    transferred = interpolate_vertex_field(
        weights, source_triangles, selected_faces, barycentrics
    ).astype(np.float32)
    transferred /= transferred.sum(axis=1, keepdims=True)
    result = CarrierTransfer(
        source_face_indices=selected_faces,
        source_barycentrics=barycentrics,
        weights=transferred,
        distances=distances,
        ambiguous=ambiguous,
    )
    result.validate(vertex_count=len(targets), source_face_count=len(source_triangles))
    return result


def interpolate_vertex_field(
    source_field: np.ndarray,
    source_faces: np.ndarray,
    source_face_indices: np.ndarray,
    source_barycentrics: np.ndarray,
) -> np.ndarray:
    """Interpolate any source-vertex field through a frozen triangle map."""
    field = np.asarray(source_field)
    faces = np.asarray(source_faces, dtype=np.int64)
    indices = np.asarray(source_face_indices, dtype=np.int64)
    barycentric = np.asarray(source_barycentrics, dtype=np.float64)
    if field.ndim < 2 or field.shape[0] <= int(indices.max(initial=-1)):
        raise ValueError("source field does not cover the transfer map")
    corners = field[faces[indices]]
    shape = (len(barycentric), 3) + (1,) * (corners.ndim - 2)
    return np.asarray((corners * barycentric.reshape(shape)).sum(axis=1))


def write_embedded_carrier(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    transfer: CarrierTransfer,
    *,
    source_face_count: int,
    pitch: float,
    alpha_over_pitch: float,
    offset_over_alpha: float,
) -> None:
    """Write the private immutable archive consumed by evaluation and future training."""
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("embedded carrier vertices must be finite with shape [V, 3]")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("embedded carrier faces must have shape [F, 3]")
    if triangles.size and (
        int(triangles.min()) < 0
        or int(triangles.max()) >= len(points)
        or bool(np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0))
    ):
        raise ValueError("embedded carrier faces contain invalid or repeated vertices")
    transfer.validate(vertex_count=len(points), source_face_count=source_face_count)
    np.savez_compressed(
        path,
        schema_version=np.asarray(EMBEDDED_CARRIER_SCHEMA),
        vertices=points,
        faces=triangles,
        weights=transfer.weights,
        source_face_indices=transfer.source_face_indices,
        source_barycentrics=transfer.source_barycentrics,
        source_face_count=np.asarray(source_face_count, dtype=np.int64),
        transfer_distances=transfer.distances,
        transfer_ambiguous=transfer.ambiguous,
        reference_pitch=np.asarray(pitch, dtype=np.float64),
        alpha_over_pitch=np.asarray(alpha_over_pitch, dtype=np.float64),
        offset_over_alpha=np.asarray(offset_over_alpha, dtype=np.float64),
    )


def read_embedded_carrier(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, CarrierTransfer, dict[str, float]]:
    with np.load(path, allow_pickle=False) as archive:
        if str(np.asarray(archive["schema_version"]).item()) != EMBEDDED_CARRIER_SCHEMA:
            raise ValueError("unsupported embedded carrier schema")
        vertices = np.asarray(archive["vertices"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        source_face_count = int(archive["source_face_count"])
        transfer = CarrierTransfer(
            source_face_indices=np.asarray(archive["source_face_indices"], dtype=np.int64),
            source_barycentrics=np.asarray(archive["source_barycentrics"], dtype=np.float64),
            weights=np.asarray(archive["weights"], dtype=np.float32),
            distances=np.asarray(archive["transfer_distances"], dtype=np.float64),
            ambiguous=np.asarray(archive["transfer_ambiguous"], dtype=np.bool_),
        )
        metadata = {
            "pitch": float(archive["reference_pitch"]),
            "alpha_over_pitch": float(archive["alpha_over_pitch"]),
            "offset_over_alpha": float(archive["offset_over_alpha"]),
        }
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("embedded carrier vertices must be finite with shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("embedded carrier faces must have shape [F, 3]")
    if faces.size and (
        int(faces.min()) < 0
        or int(faces.max()) >= len(vertices)
        or bool(np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0))
    ):
        raise ValueError("embedded carrier faces contain invalid or repeated vertices")
    transfer.validate(vertex_count=len(vertices), source_face_count=source_face_count)
    return vertices, faces, transfer, metadata


def directional_surface_fidelity(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    pitch: float,
    sample_count: int,
    seed: int,
) -> dict[str, float]:
    points, source_face_indices = trimesh.sample.sample_surface(source, sample_count, seed=seed)
    closest, distances, target_face_indices = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
        target, points
    )
    if not np.isfinite(closest).all() or not np.isfinite(distances).all():
        raise ValueError("surface fidelity query returned non-finite values")
    cosine = np.abs(
        np.einsum(
            "ij,ij->i",
            np.asarray(source.face_normals)[source_face_indices],
            np.asarray(target.face_normals)[target_face_indices],
        )
    )
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "mean_distance_pitch": float(np.mean(distances) / pitch),
        "p95_distance_pitch": float(np.quantile(distances, 0.95) / pitch),
        "median_normal_error_degrees": float(np.median(angles)),
        "p95_normal_error_degrees": float(np.quantile(angles, 0.95)),
    }


def embedded_surface_fidelity(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    pitch: float,
    sample_count: int = 100_000,
    seed: int = 20260831,
    maximum_relative_volume_error: float = 0.03,
) -> dict[str, Any]:
    """Apply the frozen cross-representation distance/normal/volume gates."""
    if pitch <= 0 or sample_count <= 0 or maximum_relative_volume_error < 0:
        raise ValueError("pitch and sample count must be positive and volume tolerance nonnegative")
    source_to_target = directional_surface_fidelity(
        source, target, pitch=pitch, sample_count=sample_count, seed=seed
    )
    target_to_source = directional_surface_fidelity(
        target, source, pitch=pitch, sample_count=sample_count, seed=seed + 1
    )
    source_volume = abs(float(source.volume))
    target_volume = abs(float(target.volume))
    relative_volume_error = abs(target_volume - source_volume) / max(source_volume, 1e-12)
    blockers: list[str] = []
    for name, report in (
        ("source_to_target", source_to_target),
        ("target_to_source", target_to_source),
    ):
        if report["mean_distance_pitch"] > 0.5:
            blockers.append(f"{name}_mean_distance")
        if report["median_normal_error_degrees"] > 5.0:
            blockers.append(f"{name}_median_normal")
        if report["p95_distance_pitch"] > 1.0:
            blockers.append(f"{name}_p95_distance")
    if relative_volume_error > maximum_relative_volume_error:
        blockers.append("legacy_relative_volume")
    return {
        "schema_version": "embedded_surface_fidelity.v1",
        "status": "pass" if not blockers else "fail",
        "sample_count_per_direction": sample_count,
        "pitch": pitch,
        "source_to_target": source_to_target,
        "target_to_source": target_to_source,
        "legacy_relative_volume_error": relative_volume_error,
        "maximum_relative_volume_error": maximum_relative_volume_error,
        "legacy_volume_is_physical_ground_truth": False,
        "blockers": blockers,
    }
