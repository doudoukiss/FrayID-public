from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import trimesh

EXPERIMENT_ID = "postv1_e12_ccd_shrinkwrap_carrier_r01"
MAXIMUM_EDGE_PITCH = 1.5
MAXIMUM_SUBDIVISION_ROUNDS = 6
MAXIMUM_FACE_COUNT = 50_000
PRESSURE_ITERATIONS = 96
CONVERGENCE_TOLERANCE_PITCH = 1e-10
TARGET_OFFSET_PITCH = 0.0025
MAXIMUM_MOTION_PITCH = 0.25
TARGET_DISTANCE_CAP = 0.45
INWARD_NORMAL_WEIGHT = 0.85
CLOSEST_DIRECTION_WEIGHT = 0.15
SMOOTHING_WEIGHT = 0.15
MAXIMUM_BACKTRACKS = 12
CCD_ACCEPTED_FRACTION = 0.8


@dataclass(frozen=True)
class ShrinkwrapStep:
    iteration: int
    objective_before: float
    objective_after: float
    ccd_maximum_step: float
    accepted_step: float
    backtracks: int
    minimum_face_area_pitch2: float


@dataclass(frozen=True)
class ShrinkwrapResult:
    status: str
    vertices: np.ndarray
    faces: np.ndarray
    initial_vertex_count: int
    initial_face_count: int
    output_vertex_count: int
    output_face_count: int
    steps: tuple[ShrinkwrapStep, ...]
    converged: bool
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "post_v1_e12_shrinkwrap_result.v1",
            "status": self.status,
            "initial_vertex_count": self.initial_vertex_count,
            "initial_face_count": self.initial_face_count,
            "output_vertex_count": self.output_vertex_count,
            "output_face_count": self.output_face_count,
            "step_count": len(self.steps),
            "converged": self.converged,
            "steps": [asdict(step) for step in self.steps],
            "blockers": list(self.blockers),
        }


def _deduplicate_source_faces(faces: np.ndarray) -> np.ndarray:
    kept: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for face in np.asarray(faces, dtype=np.int64):
        ordered = sorted(int(value) for value in face)
        key = (ordered[0], ordered[1], ordered[2])
        if key not in seen:
            seen.add(key)
            kept.append(face.copy())
    if not kept:
        raise ValueError("source contains no attraction triangles")
    return np.asarray(kept, dtype=np.int64)


def _adaptive_subdivide(
    vertices: np.ndarray, faces: np.ndarray, *, pitch: float
) -> tuple[np.ndarray, np.ndarray]:
    refined_vertices, refined_faces = trimesh.remesh.subdivide_to_size(  # type: ignore[no-untyped-call]
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        max_edge=MAXIMUM_EDGE_PITCH * pitch,
        max_iter=MAXIMUM_SUBDIVISION_ROUNDS,
    )
    refined_vertices = np.asarray(refined_vertices, dtype=np.float64)
    refined_faces = np.asarray(refined_faces, dtype=np.int64)
    if len(refined_faces) > MAXIMUM_FACE_COUNT:
        raise ValueError("subdivision_face_cap")
    mesh = trimesh.Trimesh(vertices=refined_vertices, faces=refined_faces, process=False)
    if not mesh.is_watertight or int(mesh.euler_number) != 2:
        raise ValueError("subdivision_changed_topology")
    return refined_vertices, refined_faces


def _fixed_neighbors(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, ...]:
    values: list[set[int]] = [set() for _ in range(vertex_count)]
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = (int(value) for value in face)
        values[a].update((b, c))
        values[b].update((a, c))
        values[c].update((a, b))
    return tuple(np.asarray(sorted(row), dtype=np.int64) for row in values)


def _fixed_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    contributions = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    incident: list[list[int]] = [[] for _ in range(len(vertices))]
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for vertex_index in face:
            incident[int(vertex_index)].append(face_index)
    for vertex_index, indices in enumerate(incident):
        for face_index in indices:
            normals[vertex_index] += contributions[face_index]
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.0) or not np.isfinite(lengths).all():
        raise ValueError("nonfinite_or_zero_vertex_normal")
    return np.asarray(normals / lengths[:, None], dtype=np.float64)


def _closest(mesh: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    closest, distances, _ = trimesh.proximity.closest_point(mesh, points)  # type: ignore[no-untyped-call]
    if not np.isfinite(closest).all() or not np.isfinite(distances).all():
        raise ValueError("nonfinite_source_proximity")
    return np.asarray(closest, dtype=np.float64), np.asarray(distances, dtype=np.float64)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.asarray(faces, dtype=np.int64)[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    return np.asarray(np.unique(np.sort(edges, axis=1), axis=0), dtype=np.int64)


def _minimum_face_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return float(np.min(areas, initial=np.inf))


def pressure_shrinkwrap(
    initial_vertices: np.ndarray,
    initial_faces: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    *,
    pitch: float,
) -> ShrinkwrapResult:
    """Run the frozen E12 fixed-connectivity pressure policy on CPU float64."""
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("E12 requires the pinned collision extra: ipctk==1.6.0") from error
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pitch must be positive")
    start_vertices = np.asarray(initial_vertices, dtype=np.float64)
    start_faces = np.asarray(initial_faces, dtype=np.int64)
    source_points = np.asarray(source_vertices, dtype=np.float64)
    source_triangles = np.asarray(source_faces, dtype=np.int64)
    if start_vertices.ndim != 2 or start_vertices.shape[1] != 3:
        raise ValueError("initial vertices must have shape [V,3]")
    if start_faces.ndim != 2 or start_faces.shape[1] != 3:
        raise ValueError("initial faces must have shape [F,3]")
    attraction_faces = _deduplicate_source_faces(source_triangles)
    attraction = trimesh.Trimesh(
        vertices=source_points,
        faces=attraction_faces,
        process=False,
    )
    try:
        vertices, faces = _adaptive_subdivide(start_vertices, start_faces, pitch=pitch)
    except ValueError as error:
        return ShrinkwrapResult(
            status="fail",
            vertices=start_vertices.copy(),
            faces=start_faces.copy(),
            initial_vertex_count=len(start_vertices),
            initial_face_count=len(start_faces),
            output_vertex_count=len(start_vertices),
            output_face_count=len(start_faces),
            steps=(),
            converged=False,
            blockers=(str(error),),
        )
    neighbors = _fixed_neighbors(len(vertices), faces)
    edges = _unique_edges(faces)
    collision_mesh = ipctk.CollisionMesh(
        np.asfortranarray(vertices, dtype=np.float64),
        np.asfortranarray(edges, dtype=np.int32),
        np.asfortranarray(faces, dtype=np.int32),
    )
    if ipctk.has_intersections(collision_mesh, np.asfortranarray(vertices)):
        return ShrinkwrapResult(
            status="fail",
            vertices=vertices,
            faces=faces,
            initial_vertex_count=len(start_vertices),
            initial_face_count=len(start_faces),
            output_vertex_count=len(vertices),
            output_face_count=len(faces),
            steps=(),
            converged=False,
            blockers=("initial_ipc_intersection",),
        )

    steps: list[ShrinkwrapStep] = []
    blockers: list[str] = []
    converged = False
    offset = TARGET_OFFSET_PITCH * pitch
    maximum_motion = MAXIMUM_MOTION_PITCH * pitch
    minimum_area = 1e-12 * pitch * pitch
    for iteration in range(PRESSURE_ITERATIONS):
        closest, distances = _closest(attraction, vertices)
        objective_before = float(np.mean(distances))
        normals = _fixed_vertex_normals(vertices, faces)
        closest_vectors = closest - vertices
        closest_lengths = np.linalg.norm(closest_vectors, axis=1)
        closest_directions = np.zeros_like(closest_vectors)
        nonzero = closest_lengths > 0.0
        closest_directions[nonzero] = closest_vectors[nonzero] / closest_lengths[nonzero, None]
        directions = -INWARD_NORMAL_WEIGHT * normals + CLOSEST_DIRECTION_WEIGHT * closest_directions
        direction_lengths = np.linalg.norm(directions, axis=1)
        valid_directions = direction_lengths > 0.0
        directions[valid_directions] /= direction_lengths[valid_directions, None]
        travel = np.minimum(
            maximum_motion,
            np.maximum(distances - offset, 0.0) * TARGET_DISTANCE_CAP,
        )
        if float(np.max(travel, initial=0.0)) <= CONVERGENCE_TOLERANCE_PITCH * pitch:
            converged = True
            break
        displacement = directions * travel[:, None]
        smoothed = np.empty_like(displacement)
        for vertex_index, adjacent in enumerate(neighbors):
            smoothed[vertex_index] = (
                displacement[vertex_index]
                if not len(adjacent)
                else np.mean(displacement[adjacent], axis=0)
            )
        displacement = (1.0 - SMOOTHING_WEIGHT) * displacement + SMOOTHING_WEIGHT * smoothed
        displacement_lengths = np.linalg.norm(displacement, axis=1)
        cap = np.minimum(maximum_motion, distances * TARGET_DISTANCE_CAP)
        rescale = np.minimum(1.0, cap / np.maximum(displacement_lengths, 1e-300))
        displacement *= rescale[:, None]
        if not np.isfinite(displacement).all() or float(np.max(displacement_lengths)) == 0.0:
            blockers.append(f"iteration_{iteration}:zero_or_nonfinite_pressure_step")
            break
        candidate = vertices + displacement
        ccd_maximum = float(
            ipctk.compute_collision_free_stepsize(
                collision_mesh,
                np.asfortranarray(vertices),
                np.asfortranarray(candidate),
                0.0,
            )
        )
        if not np.isfinite(ccd_maximum) or ccd_maximum <= 0.0:
            blockers.append(f"iteration_{iteration}:ccd_no_positive_step")
            break
        accepted_step = min(1.0, CCD_ACCEPTED_FRACTION * ccd_maximum)
        accepted: np.ndarray | None = None
        objective_after = np.inf
        accepted_area = 0.0
        accepted_backtracks = 0
        for backtrack_index in range(MAXIMUM_BACKTRACKS + 1):
            accepted_backtracks = backtrack_index
            attempted = vertices + accepted_step * displacement
            accepted_area = _minimum_face_area(attempted, faces)
            _, attempted_distances = _closest(attraction, attempted)
            objective_after = float(np.mean(attempted_distances))
            if accepted_area > minimum_area and objective_after < objective_before:
                accepted = attempted
                break
            accepted_step *= 0.5
        if accepted is None or accepted_step <= 0.0:
            blockers.append(f"iteration_{iteration}:objective_backtracking_exhausted")
            break
        if not ipctk.is_step_collision_free(
            collision_mesh,
            np.asfortranarray(vertices),
            np.asfortranarray(accepted),
            0.0,
        ):
            blockers.append(f"iteration_{iteration}:independent_ipc_step_check_failed")
            break
        vertices = np.asarray(accepted, dtype=np.float64)
        steps.append(
            ShrinkwrapStep(
                iteration=iteration,
                objective_before=objective_before,
                objective_after=objective_after,
                ccd_maximum_step=ccd_maximum,
                accepted_step=accepted_step,
                backtracks=accepted_backtracks,
                minimum_face_area_pitch2=accepted_area / (pitch * pitch),
            )
        )
    if len(steps) != PRESSURE_ITERATIONS and not converged:
        blockers.append("incomplete_pressure_schedule")
    return ShrinkwrapResult(
        status="pass" if not blockers else "fail",
        vertices=vertices,
        faces=faces,
        initial_vertex_count=len(start_vertices),
        initial_face_count=len(start_faces),
        output_vertex_count=len(vertices),
        output_face_count=len(faces),
        steps=tuple(steps),
        converged=converged,
        blockers=tuple(blockers),
    )
