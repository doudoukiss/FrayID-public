from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import trimesh

from frayid.shrinkwrap_carrier import (
    CCD_ACCEPTED_FRACTION,
    CLOSEST_DIRECTION_WEIGHT,
    CONVERGENCE_TOLERANCE_PITCH,
    INWARD_NORMAL_WEIGHT,
    MAXIMUM_BACKTRACKS,
    MAXIMUM_MOTION_PITCH,
    PRESSURE_ITERATIONS,
    SMOOTHING_WEIGHT,
    TARGET_DISTANCE_CAP,
    TARGET_OFFSET_PITCH,
    ShrinkwrapStep,
    _closest,
    _deduplicate_source_faces,
    _fixed_neighbors,
    _fixed_vertex_normals,
    _minimum_face_area,
    _unique_edges,
)

EXPERIMENT_ID = "postv1_e13_source_exclusion_shrinkwrap_r01"
SUBDIVISION_ROUNDS = 2
MAXIMUM_FACE_COUNT = 20_000


@dataclass(frozen=True)
class SourceExclusionResult:
    status: str
    vertices: np.ndarray
    faces: np.ndarray
    initial_vertex_count: int
    initial_face_count: int
    output_vertex_count: int
    output_face_count: int
    combined_collision_vertex_count: int
    static_source_vertex_count: int
    steps: tuple[ShrinkwrapStep, ...]
    converged: bool
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "post_v1_e13_source_exclusion_result.v1",
            "status": self.status,
            "initial_vertex_count": self.initial_vertex_count,
            "initial_face_count": self.initial_face_count,
            "output_vertex_count": self.output_vertex_count,
            "output_face_count": self.output_face_count,
            "combined_collision_vertex_count": self.combined_collision_vertex_count,
            "static_source_vertex_count": self.static_source_vertex_count,
            "source_source_pairs_filtered": True,
            "wrap_wrap_pairs_active": True,
            "wrap_source_pairs_active": True,
            "step_count": len(self.steps),
            "converged": self.converged,
            "steps": [asdict(step) for step in self.steps],
            "blockers": list(self.blockers),
        }


def _shared_midpoint(
    points: list[np.ndarray], midpoints: dict[tuple[int, int], int], first: int, second: int
) -> int:
    key = (min(first, second), max(first, second))
    if key not in midpoints:
        midpoints[key] = len(points)
        points.append(0.5 * (points[key[0]] + points[key[1]]))
    return midpoints[key]


def uniform_conforming_subdivide(
    vertices: np.ndarray, faces: np.ndarray, *, rounds: int = SUBDIVISION_ROUNDS
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide every oriented triangle using one midpoint per sorted edge."""
    if rounds < 0:
        raise ValueError("subdivision rounds cannot be negative")
    points = [row.copy() for row in np.asarray(vertices, dtype=np.float64)]
    triangles = np.asarray(faces, dtype=np.int64)
    for _ in range(rounds):
        midpoints: dict[tuple[int, int], int] = {}
        children: list[tuple[int, int, int]] = []

        for face in triangles:
            a, b, c = (int(value) for value in face)
            ab = _shared_midpoint(points, midpoints, a, b)
            bc = _shared_midpoint(points, midpoints, b, c)
            ca = _shared_midpoint(points, midpoints, c, a)
            children.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        triangles = np.asarray(children, dtype=np.int64)
        if len(triangles) > MAXIMUM_FACE_COUNT:
            raise ValueError("subdivision_face_cap")
    result_vertices = np.asarray(points, dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=result_vertices, faces=triangles, process=False)
    if not mesh.is_watertight or int(mesh.euler_number) != 2:
        raise ValueError("conforming_subdivision_changed_topology")
    return result_vertices, triangles


def source_exclusion_shrinkwrap(
    initial_vertices: np.ndarray,
    initial_faces: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    *,
    pitch: float,
) -> SourceExclusionResult:
    """Run E13 pressure with moving-wrap/self and moving-wrap/source CCD."""
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("E13 requires the pinned collision extra: ipctk==1.6.0") from error
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pitch must be positive")
    start_vertices = np.asarray(initial_vertices, dtype=np.float64)
    start_faces = np.asarray(initial_faces, dtype=np.int64)
    source_points = np.asarray(source_vertices, dtype=np.float64)
    source_triangles = np.asarray(source_faces, dtype=np.int64)
    try:
        vertices, faces = uniform_conforming_subdivide(start_vertices, start_faces)
    except ValueError as error:
        return SourceExclusionResult(
            status="fail",
            vertices=start_vertices.copy(),
            faces=start_faces.copy(),
            initial_vertex_count=len(start_vertices),
            initial_face_count=len(start_faces),
            output_vertex_count=len(start_vertices),
            output_face_count=len(start_faces),
            combined_collision_vertex_count=len(start_vertices) + len(source_points),
            static_source_vertex_count=len(source_points),
            steps=(),
            converged=False,
            blockers=(str(error),),
        )
    attraction = trimesh.Trimesh(
        vertices=source_points,
        faces=_deduplicate_source_faces(source_triangles),
        process=False,
    )
    wrap_vertex_count = len(vertices)
    combined_vertices = np.vstack((vertices, source_points))
    combined_faces = np.vstack((faces, source_triangles + wrap_vertex_count))
    combined_edges = _unique_edges(combined_faces)
    collision_mesh = ipctk.CollisionMesh(
        np.asfortranarray(combined_vertices, dtype=np.float64),
        np.asfortranarray(combined_edges, dtype=np.int32),
        np.asfortranarray(combined_faces, dtype=np.int32),
    )
    collision_mesh.can_collide = ipctk.make_static_obstacle_filter(wrap_vertex_count)
    if ipctk.has_intersections(collision_mesh, np.asfortranarray(combined_vertices)):
        return SourceExclusionResult(
            status="fail",
            vertices=vertices,
            faces=faces,
            initial_vertex_count=len(start_vertices),
            initial_face_count=len(start_faces),
            output_vertex_count=len(vertices),
            output_face_count=len(faces),
            combined_collision_vertex_count=len(combined_vertices),
            static_source_vertex_count=len(source_points),
            steps=(),
            converged=False,
            blockers=("initial_combined_ipc_intersection",),
        )

    neighbors = _fixed_neighbors(len(vertices), faces)
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
        candidate_wrap = vertices + displacement
        combined_start = np.vstack((vertices, source_points))
        combined_candidate = np.vstack((candidate_wrap, source_points))
        ccd_maximum = float(
            ipctk.compute_collision_free_stepsize(
                collision_mesh,
                np.asfortranarray(combined_start),
                np.asfortranarray(combined_candidate),
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
        combined_accepted = np.vstack((accepted, source_points))
        if not ipctk.is_step_collision_free(
            collision_mesh,
            np.asfortranarray(combined_start),
            np.asfortranarray(combined_accepted),
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
    return SourceExclusionResult(
        status="pass" if not blockers else "fail",
        vertices=vertices,
        faces=faces,
        initial_vertex_count=len(start_vertices),
        initial_face_count=len(start_faces),
        output_vertex_count=len(vertices),
        output_face_count=len(faces),
        combined_collision_vertex_count=len(vertices) + len(source_points),
        static_source_vertex_count=len(source_points),
        steps=tuple(steps),
        converged=converged,
        blockers=tuple(blockers),
    )
