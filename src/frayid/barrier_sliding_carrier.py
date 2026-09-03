from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as sparse  # type: ignore[import-untyped]
import scipy.sparse.linalg as sparse_linalg  # type: ignore[import-untyped]
import trimesh

from frayid.refinement_certificate import (
    ExactRefinementCertificate,
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)
from frayid.shrinkwrap_carrier import (
    CLOSEST_DIRECTION_WEIGHT,
    CONVERGENCE_TOLERANCE_PITCH,
    INWARD_NORMAL_WEIGHT,
    MAXIMUM_BACKTRACKS,
    MAXIMUM_MOTION_PITCH,
    PRESSURE_ITERATIONS,
    SMOOTHING_WEIGHT,
    TARGET_DISTANCE_CAP,
    TARGET_OFFSET_PITCH,
    _closest,
    _deduplicate_source_faces,
    _fixed_neighbors,
    _fixed_vertex_normals,
    _minimum_face_area,
    _unique_edges,
)

EXPERIMENT_ID = "postv1_e15_ipc_barrier_sliding_carrier_r01"
DHAT_BBOX_FRACTION = 1e-3
DMIN = 0.0
CCD_ACCEPTED_FRACTION = 0.8
MINIMUM_ACCEPTED_STEP = 1e-8
MAXIMUM_LINEAR_RESIDUAL = 1e-8
REPORT_SIGNIFICANT_DIGITS = 9
REPORT_ZERO_FLOOR = 1e-14


def _report_float(value: float) -> float:
    if abs(value) < REPORT_ZERO_FLOOR:
        return 0.0
    return float(format(float(value), f".{REPORT_SIGNIFICANT_DIGITS}g"))


@dataclass(frozen=True)
class BarrierSlidingStep:
    iteration: int
    objective_before: float
    objective_after: float
    native_energy_before: float
    native_energy_after: float
    barrier_energy_before: float
    barrier_energy_after: float
    barrier_stiffness: float
    active_collision_count: int
    minimum_distance: float
    ccd_maximum_step: float
    accepted_step: float
    backtracks: int
    normalized_linear_residual: float
    minimum_face_area_pitch2: float


@dataclass(frozen=True)
class BarrierSlidingResult:
    status: str
    vertices: np.ndarray
    faces: np.ndarray
    certificate: ExactRefinementCertificate
    steps: tuple[BarrierSlidingStep, ...]
    converged: bool
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "post_v1_e15_barrier_sliding_result.v1",
            "status": self.status,
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "certificate": self.certificate.report(),
            "step_count": len(self.steps),
            "converged": self.converged,
            "steps": [asdict(step) for step in self.steps],
            "blockers": list(self.blockers),
        }


def _native_target(
    vertices: np.ndarray,
    faces: np.ndarray,
    attraction: trimesh.Trimesh,
    neighbors: tuple[np.ndarray, ...],
    *,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    closest, distances = _closest(attraction, vertices)
    normals = _fixed_vertex_normals(vertices, faces)
    vectors = closest - vertices
    lengths = np.linalg.norm(vectors, axis=1)
    directions_to_source = np.zeros_like(vectors)
    nonzero = lengths > 0.0
    directions_to_source[nonzero] = vectors[nonzero] / lengths[nonzero, None]
    directions = -INWARD_NORMAL_WEIGHT * normals + CLOSEST_DIRECTION_WEIGHT * directions_to_source
    direction_lengths = np.linalg.norm(directions, axis=1)
    valid = direction_lengths > 0.0
    directions[valid] /= direction_lengths[valid, None]
    travel = np.minimum(
        MAXIMUM_MOTION_PITCH * pitch,
        np.maximum(distances - TARGET_OFFSET_PITCH * pitch, 0.0) * TARGET_DISTANCE_CAP,
    )
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
    cap = np.minimum(MAXIMUM_MOTION_PITCH * pitch, distances * TARGET_DISTANCE_CAP)
    displacement *= np.minimum(1.0, cap / np.maximum(displacement_lengths, 1e-300))[:, None]
    return vertices + displacement, distances, float(np.max(travel, initial=0.0))


def _collision_count(collisions: Any) -> int:
    return sum(
        len(getattr(collisions, name))
        for name in ("vv_collisions", "ev_collisions", "ee_collisions", "fv_collisions")
    )


def barrier_sliding_carrier(
    parent_vertices: np.ndarray,
    parent_faces: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    *,
    pitch: float,
    parent_grid: float,
) -> BarrierSlidingResult:
    """Run the frozen E15 IPC barrier-Newton public mechanism."""
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("E15 requires the pinned collision extra: ipctk==1.6.0") from error
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pitch must be positive")

    parent_points = np.asarray(parent_vertices, dtype=np.float64)
    parent_triangles = np.asarray(parent_faces, dtype=np.int64)
    source_points = np.asarray(source_vertices, dtype=np.float64)
    source_triangles = np.asarray(source_faces, dtype=np.int64)
    refinement = subdivide_with_exact_provenance(parent_points, parent_triangles, rounds=2)
    certificate = certify_exact_dyadic_refinement(
        parent_points,
        parent_triangles,
        refinement,
        parent_grid=parent_grid,
        rounds=2,
    )
    vertices = refinement.vertices.copy()
    faces = refinement.faces.copy()
    if certificate.status != "pass":
        return BarrierSlidingResult(
            status="fail",
            vertices=vertices,
            faces=faces,
            certificate=certificate,
            steps=(),
            converged=False,
            blockers=("p2_exact_refinement_certificate",),
        )

    attraction = trimesh.Trimesh(
        vertices=source_points,
        faces=_deduplicate_source_faces(source_triangles),
        process=False,
    )
    neighbors = _fixed_neighbors(len(vertices), faces)
    wrap_count = len(vertices)
    combined_vertices = np.vstack((vertices, source_points))
    combined_faces = np.vstack((faces, source_triangles + wrap_count))
    collision_mesh = ipctk.CollisionMesh(
        np.asfortranarray(combined_vertices, dtype=np.float64),
        np.asfortranarray(_unique_edges(combined_faces), dtype=np.int32),
        np.asfortranarray(combined_faces, dtype=np.int32),
    )
    collision_mesh.can_collide = ipctk.make_static_obstacle_filter(wrap_count)
    if ipctk.has_intersections(
        collision_mesh, np.asfortranarray(combined_vertices, dtype=np.float64)
    ):
        return BarrierSlidingResult(
            status="fail",
            vertices=vertices,
            faces=faces,
            certificate=certificate,
            steps=(),
            converged=False,
            blockers=("initial_combined_ipc_intersection",),
        )

    diagonal = float(np.linalg.norm(np.ptp(combined_vertices, axis=0)))
    initial_lower = np.min(combined_vertices, axis=0)
    initial_upper = np.max(combined_vertices, axis=0)
    state_origin = 0.5 * (initial_lower + initial_upper)
    state_grid = math.ldexp(1.0, math.floor(math.log2(diagonal)) - 40)
    dhat = DHAT_BBOX_FRACTION * diagonal
    broad_phase = ipctk.SweepAndPrune()
    barrier = ipctk.ClampedLogBarrier()
    unit_potential = ipctk.BarrierPotential(barrier, dhat, 1.0)
    wrap_dofs = 3 * wrap_count
    native_hessian = sparse.identity(wrap_dofs, dtype=np.float64, format="csc") / wrap_count
    minimum_area = 1e-12 * pitch * pitch
    stiffness = 0.0
    maximum_stiffness = 0.0
    previous_minimum_distance = np.inf
    steps: list[BarrierSlidingStep] = []
    blockers: list[str] = []
    converged = False

    for iteration in range(PRESSURE_ITERATIONS):
        target, distances, maximum_travel = _native_target(
            vertices, faces, attraction, neighbors, pitch=pitch
        )
        objective_before = float(np.mean(distances))
        if maximum_travel <= CONVERGENCE_TOLERANCE_PITCH * pitch:
            converged = True
            break
        combined = np.asfortranarray(np.vstack((vertices, source_points)))
        collisions = ipctk.NormalCollisions()
        collisions.build(collision_mesh, combined, dhat, DMIN, broad_phase)
        native_gradient_wrap = (vertices - target).reshape(-1) / wrap_count
        native_gradient = np.zeros(combined.size, dtype=np.float64)
        native_gradient[:wrap_dofs] = native_gradient_wrap
        barrier_gradient = np.asarray(
            unit_potential.gradient(collisions, collision_mesh, combined),
            dtype=np.float64,
        ).reshape(-1)
        active_count = _collision_count(collisions)
        if active_count and stiffness == 0.0:
            stiffness, maximum_stiffness = ipctk.initial_barrier_stiffness(
                diagonal,
                barrier,
                dhat,
                1.0,
                native_gradient.reshape(-1, 1),
                barrier_gradient.reshape(-1, 1),
                dmin=DMIN,
            )
        barrier_hessian = unit_potential.hessian(
            collisions,
            collision_mesh,
            combined,
            ipctk.PSDProjectionMethod.CLAMP,
        )
        system = native_hessian + stiffness * barrier_hessian[:wrap_dofs, :wrap_dofs]
        rhs = -(native_gradient_wrap + stiffness * barrier_gradient[:wrap_dofs])
        direction_flat = np.asarray(sparse_linalg.spsolve(system, rhs), dtype=np.float64)
        residual = np.asarray(system @ direction_flat - rhs, dtype=np.float64)
        normalized_residual = float(
            np.linalg.norm(residual, ord=np.inf) / max(1.0, float(np.linalg.norm(rhs, ord=np.inf)))
        )
        if not np.isfinite(direction_flat).all() or normalized_residual > MAXIMUM_LINEAR_RESIDUAL:
            blockers.append(f"iteration_{iteration}:linear_solve_certificate")
            break
        direction = direction_flat.reshape(wrap_count, 3)
        if float(np.max(np.linalg.norm(direction, axis=1), initial=0.0)) == 0.0:
            blockers.append(f"iteration_{iteration}:zero_newton_direction")
            break

        candidate = np.asfortranarray(np.vstack((vertices + direction, source_points)))
        ccd_maximum = float(
            ipctk.compute_collision_free_stepsize(
                collision_mesh, combined, candidate, DMIN, broad_phase
            )
        )
        accepted_step = min(1.0, CCD_ACCEPTED_FRACTION * ccd_maximum)
        native_before = float(0.5 * np.sum((vertices - target) ** 2) / wrap_count)
        barrier_before = float(unit_potential(collisions, collision_mesh, combined))
        total_before = native_before + stiffness * barrier_before
        accepted: np.ndarray | None = None
        accepted_native = np.inf
        accepted_barrier = np.inf
        accepted_area = 0.0
        accepted_minimum_distance = np.inf
        accepted_backtracks = 0
        for backtrack in range(MAXIMUM_BACKTRACKS + 1):
            accepted_backtracks = backtrack
            if accepted_step < MINIMUM_ACCEPTED_STEP:
                break
            attempted_raw = vertices + accepted_step * direction
            attempted = (
                state_origin + np.rint((attempted_raw - state_origin) / state_grid) * state_grid
            )
            accepted_area = _minimum_face_area(attempted, faces)
            attempted_combined = np.asfortranarray(
                np.vstack((attempted, source_points)), dtype=np.float64
            )
            attempted_collisions = ipctk.NormalCollisions()
            attempted_collisions.build(collision_mesh, attempted_combined, dhat, DMIN, broad_phase)
            accepted_native = float(0.5 * np.sum((attempted - target) ** 2) / wrap_count)
            accepted_barrier = float(
                unit_potential(attempted_collisions, collision_mesh, attempted_combined)
            )
            if (
                accepted_area > minimum_area
                and accepted_native + stiffness * accepted_barrier < total_before
            ):
                accepted = attempted
                accepted_minimum_distance = float(
                    attempted_collisions.compute_minimum_distance(
                        collision_mesh, attempted_combined
                    )
                )
                break
            accepted_step *= 0.5
        if accepted is None:
            blockers.append(f"iteration_{iteration}:barrier_backtracking_or_step_floor")
            break
        accepted_combined = np.asfortranarray(
            np.vstack((accepted, source_points)), dtype=np.float64
        )
        if not ipctk.is_step_collision_free(
            collision_mesh, combined, accepted_combined, DMIN, broad_phase
        ):
            blockers.append(f"iteration_{iteration}:independent_ipc_step_check")
            break
        _, accepted_distances = _closest(attraction, accepted)
        objective_after = float(np.mean(accepted_distances))
        step_stiffness = stiffness
        if stiffness > 0.0 and np.isfinite(previous_minimum_distance):
            stiffness = float(
                ipctk.update_barrier_stiffness(
                    previous_minimum_distance,
                    accepted_minimum_distance,
                    maximum_stiffness,
                    stiffness,
                    diagonal,
                    dmin=DMIN,
                )
            )
        previous_minimum_distance = accepted_minimum_distance
        vertices = np.asarray(accepted, dtype=np.float64)
        steps.append(
            BarrierSlidingStep(
                iteration=iteration,
                objective_before=_report_float(objective_before),
                objective_after=_report_float(objective_after),
                native_energy_before=_report_float(native_before),
                native_energy_after=_report_float(accepted_native),
                barrier_energy_before=_report_float(barrier_before),
                barrier_energy_after=_report_float(accepted_barrier),
                barrier_stiffness=_report_float(step_stiffness),
                active_collision_count=active_count,
                minimum_distance=_report_float(accepted_minimum_distance),
                ccd_maximum_step=_report_float(ccd_maximum),
                accepted_step=_report_float(accepted_step),
                backtracks=accepted_backtracks,
                normalized_linear_residual=_report_float(normalized_residual),
                minimum_face_area_pitch2=_report_float(accepted_area / (pitch * pitch)),
            )
        )

    if len(steps) != PRESSURE_ITERATIONS and not converged:
        blockers.append("incomplete_barrier_schedule")
    return BarrierSlidingResult(
        status="pass" if not blockers else "fail",
        vertices=vertices,
        faces=faces,
        certificate=certificate,
        steps=tuple(steps),
        converged=converged,
        blockers=tuple(blockers),
    )
