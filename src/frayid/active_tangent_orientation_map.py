from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from frayid.certified_tet_path import CertifiedPLPathV1, certify_piecewise_affine_path
from frayid.coarse_bilipschitz import (
    ConformingSurfaceV1,
    FreudenthalLatticeV1,
    parent_area_path_report,
)
from frayid.coarse_orientation_map import fit_coarse_controls
from frayid.composed_orientation_map import MaterialEmbeddingV1

ACTIVE_DETERMINANT_RATIO = 0.25
MAXIMUM_NORMALIZED_KKT_RESIDUAL = 1.0e-10
MAXIMUM_TANGENT_RESIDUAL = 1.0e-10


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def tetrahedron_determinants(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)[np.asarray(tetrahedra, dtype=np.int64)]
    return np.asarray(
        np.einsum(
            "ij,ij->i",
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            points[:, 3] - points[:, 0],
        ),
        dtype=np.float64,
    )


def tetrahedron_determinant_vertex_gradients(points: np.ndarray) -> np.ndarray:
    local = np.asarray(points, dtype=np.float64)
    if local.shape != (4, 3) or not np.isfinite(local).all():
        raise ValueError("tetrahedron points must have finite shape [4,3]")
    first = local[1] - local[0]
    second = local[2] - local[0]
    third = local[3] - local[0]
    gradients = np.empty((4, 3), dtype=np.float64)
    gradients[1] = np.cross(second, third)
    gradients[2] = np.cross(third, first)
    gradients[3] = np.cross(first, second)
    gradients[0] = -(gradients[1] + gradients[2] + gradients[3])
    return gradients


def _design_matrix(
    lattice: FreudenthalLatticeV1, reference_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights, _ = lattice.locate(np.asarray(reference_points, dtype=np.float64))
    free = np.flatnonzero(~lattice.boundary_mask)
    lookup = np.full(lattice.vertices.shape[0], -1, dtype=np.int64)
    lookup[free] = np.arange(free.size, dtype=np.int64)
    design = np.zeros((nodes.shape[0], free.size), dtype=np.float64)
    for column in range(4):
        free_columns = lookup[nodes[:, column]]
        rows = np.flatnonzero(free_columns >= 0)
        design[rows, free_columns[rows]] += weights[rows, column]
    return design, free, lookup


def _active_tangent_rows(
    lattice: FreudenthalLatticeV1,
    current_lattice_vertices: np.ndarray,
    free_lookup: np.ndarray,
    *,
    active_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = tetrahedron_determinants(lattice.vertices, lattice.tetrahedra)
    current = tetrahedron_determinants(current_lattice_vertices, lattice.tetrahedra)
    if np.any(reference <= 0.0) or np.any(current <= 0.0):
        raise ValueError("active tangent solve requires positive current/reference tetrahedra")
    ratios = current / reference
    active = np.flatnonzero(ratios <= active_ratio)
    free_count = int(np.count_nonzero(~lattice.boundary_mask))
    rows = np.zeros((active.size, free_count * 3), dtype=np.float64)
    for output_row, tetrahedron_index in enumerate(active):
        tetrahedron = lattice.tetrahedra[tetrahedron_index]
        gradients = tetrahedron_determinant_vertex_gradients(
            np.asarray(current_lattice_vertices, dtype=np.float64)[tetrahedron]
        )
        for local, node in enumerate(tetrahedron):
            free_index = free_lookup[int(node)]
            if free_index >= 0:
                rows[output_row, 3 * free_index : 3 * free_index + 3] += gradients[local]
        norm = float(np.linalg.norm(rows[output_row]))
        if norm > 0.0:
            rows[output_row] /= norm
    nonzero = np.linalg.norm(rows, axis=1) > 0.0
    return rows[nonzero], active[nonzero], ratios


@dataclass(frozen=True)
class ActiveTangentSolveV1:
    controls: np.ndarray
    active_tetrahedron_indices: np.ndarray
    active_ratio_minimum: float
    active_ratio_maximum: float
    constraint_rank: int
    normalized_kkt_residual: float
    maximum_tangent_residual: float
    status: str
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "active_determinant_tangent_solve.v1",
            "status": self.status,
            "active_tetrahedron_count": int(self.active_tetrahedron_indices.size),
            "active_tetrahedron_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.active_tetrahedron_indices, dtype="<i8").tobytes()
            ).hexdigest(),
            "active_ratio_minimum": self.active_ratio_minimum,
            "active_ratio_maximum": self.active_ratio_maximum,
            "constraint_rank": self.constraint_rank,
            "normalized_kkt_residual": self.normalized_kkt_residual,
            "maximum_tangent_residual": self.maximum_tangent_residual,
            "controls_sha256": hashlib.sha256(
                np.ascontiguousarray(self.controls, dtype="<f8").tobytes()
            ).hexdigest(),
            "blockers": list(self.blockers),
        }


def fit_active_determinant_tangent_controls(
    lattice: FreudenthalLatticeV1,
    reference_carrier_vertices: np.ndarray,
    residual: np.ndarray,
    current_lattice_vertices: np.ndarray,
    *,
    active_ratio: float = ACTIVE_DETERMINANT_RATIO,
    tikhonov: float = 1.0e-10,
    rcond: float = 1.0e-12,
    deadline: float | None = None,
) -> ActiveTangentSolveV1:
    if not 0.0 < active_ratio < 1.0:
        raise ValueError("active determinant ratio must lie in (0,1)")
    target = np.asarray(residual, dtype=np.float64)
    design, free, lookup = _design_matrix(lattice, reference_carrier_vertices)
    tangent_rows, active_indices, ratios = _active_tangent_rows(
        lattice,
        current_lattice_vertices,
        lookup,
        active_ratio=active_ratio,
    )
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("active determinant tangent construction timed out")
    if active_indices.size == 0:
        controls, _ = fit_coarse_controls(
            lattice,
            reference_carrier_vertices,
            target,
            tikhonov=tikhonov,
            rcond=rcond,
            deadline=deadline,
        )
        return ActiveTangentSolveV1(
            controls=controls,
            active_tetrahedron_indices=active_indices,
            active_ratio_minimum=1.0,
            active_ratio_maximum=1.0,
            constraint_rank=0,
            normalized_kkt_residual=0.0,
            maximum_tangent_residual=0.0,
            status="pass",
            blockers=(),
        )
    base = design.T @ design + tikhonov * np.eye(free.size, dtype=np.float64)
    hessian = np.kron(base, np.eye(3, dtype=np.float64))
    right = (design.T @ target).reshape(-1)
    zero = np.zeros((tangent_rows.shape[0], tangent_rows.shape[0]), dtype=np.float64)
    kkt = np.block([[hessian, tangent_rows.T], [tangent_rows, zero]])
    target_vector = np.concatenate((right, np.zeros(tangent_rows.shape[0], dtype=np.float64)))
    solution, _, _, _ = np.linalg.lstsq(kkt, target_vector, rcond=rcond)
    normalized_residual = float(np.linalg.norm(kkt @ solution - target_vector)) / max(
        float(np.linalg.norm(target_vector)), 1.0
    )
    free_controls = solution[: free.size * 3].reshape(free.size, 3)
    maximum_tangent = float(np.max(np.abs(tangent_rows @ free_controls.reshape(-1)), initial=0.0))
    controls = np.zeros_like(lattice.vertices)
    controls[free] = free_controls
    blockers: list[str] = []
    if normalized_residual > MAXIMUM_NORMALIZED_KKT_RESIDUAL:
        blockers.append("normalized_kkt_residual")
    if maximum_tangent > MAXIMUM_TANGENT_RESIDUAL:
        blockers.append("tangent_constraint_residual")
    if np.any(controls[lattice.boundary_mask] != 0.0):
        blockers.append("fixed_boundary")
    if not np.isfinite(controls).all():
        blockers.append("nonfinite_controls")
    return ActiveTangentSolveV1(
        controls=controls,
        active_tetrahedron_indices=active_indices,
        active_ratio_minimum=float(np.min(ratios[active_indices])),
        active_ratio_maximum=float(np.max(ratios[active_indices])),
        constraint_rank=int(np.linalg.matrix_rank(tangent_rows, tol=rcond)),
        normalized_kkt_residual=normalized_residual,
        maximum_tangent_residual=maximum_tangent,
        status="pass" if not blockers else "fail",
        blockers=tuple(blockers),
    )


@dataclass(frozen=True)
class CertifiedActiveTangentBlockV1:
    index: int
    tangent_solve: ActiveTangentSolveV1
    accepted_controls: np.ndarray
    accepted_alpha: float
    proposal_path: CertifiedPLPathV1
    accepted_path: CertifiedPLPathV1
    residual_norm_before: float
    residual_norm_after: float
    status: str
    blockers: tuple[str, ...]
    elapsed_seconds: float
    decision_sha256: str

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "certified_active_tangent_block.v1",
            "index": self.index,
            "status": self.status,
            "accepted_alpha": self.accepted_alpha,
            "accepted_alpha_hex": self.accepted_alpha.hex(),
            "tangent_solve": self.tangent_solve.report(),
            "accepted_controls_sha256": hashlib.sha256(
                np.ascontiguousarray(self.accepted_controls, dtype="<f8").tobytes()
            ).hexdigest(),
            "proposal_path": self.proposal_path.report(),
            "accepted_serialized_path": self.accepted_path.report(),
            "residual_norm_before": self.residual_norm_before,
            "residual_norm_after": self.residual_norm_after,
            "decision_sha256": self.decision_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class CertifiedActiveTangentStepV1:
    blocks: tuple[CertifiedActiveTangentBlockV1, ...]
    accepted_control_blocks: np.ndarray
    final_lattice_vertices: np.ndarray
    final_carrier_vertices: np.ndarray
    final_refined_surface_vertices: np.ndarray
    retained_displacement_ratio: float
    relative_endpoint_error: float
    proposal_cosine: float
    parent_area_report: dict[str, Any]
    status: str
    blockers: tuple[str, ...]
    elapsed_seconds: float
    decision_sha256: str

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "certified_active_tangent_step.v1",
            "status": self.status,
            "block_count": len(self.blocks),
            "blocks": [block.report() for block in self.blocks],
            "retained_displacement_ratio": self.retained_displacement_ratio,
            "relative_endpoint_error": self.relative_endpoint_error,
            "proposal_cosine": self.proposal_cosine,
            "parent_area": self.parent_area_report,
            "final_lattice_vertices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.final_lattice_vertices, dtype="<f8").tobytes()
            ).hexdigest(),
            "final_carrier_vertices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.final_carrier_vertices, dtype="<f8").tobytes()
            ).hexdigest(),
            "final_refined_surface_vertices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.final_refined_surface_vertices, dtype="<f8").tobytes()
            ).hexdigest(),
            "decision_sha256": self.decision_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "blockers": list(self.blockers),
        }


def fit_and_certify_active_tangent_step(
    lattice: FreudenthalLatticeV1,
    carrier_vertices: np.ndarray,
    carrier_faces: np.ndarray,
    refined_surface: ConformingSurfaceV1,
    proposed_displacements: np.ndarray,
    *,
    block_count: int = 4,
    minimum_retained_displacement_ratio: float = 0.25,
    active_ratio: float = ACTIVE_DETERMINANT_RATIO,
    tikhonov: float = 1.0e-10,
    rcond: float = 1.0e-12,
    timeout_seconds_per_block: float | None = 60.0,
) -> CertifiedActiveTangentStepV1:
    started = time.monotonic()
    if block_count < 2:
        raise ValueError("active tangent path requires at least two blocks")
    reference_carrier = np.asarray(carrier_vertices, dtype=np.float64)
    faces = np.asarray(carrier_faces, dtype=np.int64)
    proposal = np.asarray(proposed_displacements, dtype=np.float64)
    target = np.asarray(reference_carrier + proposal, dtype=np.float64)
    carrier_embedding = MaterialEmbeddingV1.create(lattice, reference_carrier)
    surface_embedding = MaterialEmbeddingV1.create(lattice, refined_surface.reference_vertices)
    current_lattice = lattice.vertices.copy()
    current_carrier = reference_carrier.copy()
    current_surface = refined_surface.reference_vertices.copy()
    blocks: list[CertifiedActiveTangentBlockV1] = []
    accepted_controls: list[np.ndarray] = []
    blockers: list[str] = []
    for index in range(block_count):
        block_started = time.monotonic()
        deadline = (
            None if timeout_seconds_per_block is None else block_started + timeout_seconds_per_block
        )
        residual = np.asarray(target - current_carrier, dtype=np.float64)
        tangent = fit_active_determinant_tangent_controls(
            lattice,
            reference_carrier,
            residual,
            current_lattice,
            active_ratio=active_ratio,
            tikhonov=tikhonov,
            rcond=rcond,
            deadline=deadline,
        )
        raw_controls = tangent.controls
        raw_surface_direction = surface_embedding.evaluate(raw_controls)
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        proposal_path = certify_piecewise_affine_path(
            current_lattice,
            lattice.tetrahedra,
            raw_controls,
            current_surface,
            refined_surface.faces,
            raw_surface_direction,
            timeout_seconds=remaining,
        )
        alpha = proposal_path.accepted_alpha
        block_controls = np.asarray(raw_controls * alpha, dtype=np.float64)
        carrier_direction = carrier_embedding.evaluate(block_controls)
        surface_direction = surface_embedding.evaluate(block_controls)
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        accepted_path = certify_piecewise_affine_path(
            current_lattice,
            lattice.tetrahedra,
            block_controls,
            current_surface,
            refined_surface.faces,
            surface_direction,
            timeout_seconds=remaining,
        )
        block_blockers = list(tangent.blockers)
        if proposal_path.status != "pass":
            block_blockers.extend(f"proposal_path:{value}" for value in proposal_path.blockers)
        if accepted_path.status != "pass" or accepted_path.accepted_alpha != 1.0:
            block_blockers.append("accepted_serialized_path_not_fully_certified")
        if not np.any(block_controls != 0.0):
            block_blockers.append("zero_block_motion")
        next_lattice = np.asarray(current_lattice + block_controls, dtype=np.float64)
        next_carrier = np.asarray(current_carrier + carrier_direction, dtype=np.float64)
        next_surface = np.asarray(current_surface + surface_direction, dtype=np.float64)
        decision = _sha256_arrays(
            current_lattice.astype("<f8"),
            current_carrier.astype("<f8"),
            raw_controls.astype("<f8"),
            block_controls.astype("<f8"),
            next_lattice.astype("<f8"),
            next_carrier.astype("<f8"),
            tangent.active_tetrahedron_indices.astype("<i8"),
        )
        block = CertifiedActiveTangentBlockV1(
            index=index,
            tangent_solve=tangent,
            accepted_controls=block_controls,
            accepted_alpha=alpha,
            proposal_path=proposal_path,
            accepted_path=accepted_path,
            residual_norm_before=float(np.linalg.norm(residual)),
            residual_norm_after=float(np.linalg.norm(target - next_carrier)),
            status="pass" if not block_blockers else "fail",
            blockers=tuple(block_blockers),
            elapsed_seconds=time.monotonic() - block_started,
            decision_sha256=decision,
        )
        blocks.append(block)
        accepted_controls.append(block_controls)
        if block_blockers:
            blockers.extend(f"block_{index}:{value}" for value in block_blockers)
            break
        current_lattice = next_lattice
        current_carrier = next_carrier
        current_surface = next_surface
    if len(blocks) != block_count:
        blockers.append("incomplete_block_sequence")
    control_array = np.asarray(accepted_controls, dtype=np.float64)
    displacement = np.asarray(current_carrier - reference_carrier, dtype=np.float64)
    proposal_norm = float(np.linalg.norm(proposal))
    displacement_norm = float(np.linalg.norm(displacement))
    retention = displacement_norm / proposal_norm if proposal_norm > 0.0 else 0.0
    endpoint_error = (
        float(np.linalg.norm(current_carrier - target)) / proposal_norm
        if proposal_norm > 0.0
        else math.inf
    )
    denominator = displacement_norm * proposal_norm
    cosine = float(np.sum(displacement * proposal)) / denominator if denominator > 0.0 else 0.0
    parent_area = parent_area_path_report(
        reference_carrier, faces, refined_surface, current_surface
    )
    if proposal_norm <= 0.0:
        blockers.append("nonpositive_proposed_motion")
    if retention < minimum_retained_displacement_ratio:
        blockers.append("motion_retention")
    if displacement_norm <= 0.0:
        blockers.append("zero_composed_motion")
    blockers.extend(parent_area["blockers"])
    decision = _sha256_arrays(
        lattice.vertices.astype("<f8"),
        lattice.tetrahedra.astype("<i8"),
        control_array.astype("<f8"),
        current_lattice.astype("<f8"),
        current_carrier.astype("<f8"),
        current_surface.astype("<f8"),
    )
    return CertifiedActiveTangentStepV1(
        blocks=tuple(blocks),
        accepted_control_blocks=control_array,
        final_lattice_vertices=current_lattice,
        final_carrier_vertices=current_carrier,
        final_refined_surface_vertices=current_surface,
        retained_displacement_ratio=retention,
        relative_endpoint_error=endpoint_error,
        proposal_cosine=cosine,
        parent_area_report=parent_area,
        status="pass" if not blockers else "fail",
        blockers=tuple(blockers),
        elapsed_seconds=time.monotonic() - started,
        decision_sha256=decision,
    )


def run_active_tangent_controls() -> dict[str, Any]:
    unit = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    direction = np.asarray(
        [[0.1, -0.2, 0.05], [-0.1, 0.3, 0.2], [0.2, 0.1, -0.2], [-0.2, -0.1, 0.1]],
        dtype=np.float64,
    )
    gradients = tetrahedron_determinant_vertex_gradients(unit)
    analytic = float(np.sum(gradients * direction))
    epsilon = 1.0e-6
    plus = tetrahedron_determinants(
        unit + epsilon * direction, np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    )[0]
    minus = tetrahedron_determinants(
        unit - epsilon * direction, np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    )[0]
    finite_difference = float((plus - minus) / (2.0 * epsilon))
    lattice = FreudenthalLatticeV1.create(
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        nodes_per_axis=4,
    )
    current = lattice.vertices.copy()
    interior = np.flatnonzero(~lattice.boundary_mask)
    current[interior[0], 0] += 0.4
    reference_det = tetrahedron_determinants(lattice.vertices, lattice.tetrahedra)
    current_det = tetrahedron_determinants(current, lattice.tetrahedra)
    minimum_ratio = float(np.min(current_det / reference_det))
    points = np.asarray(
        [[x, y, z] for x in (-0.4, 0.0, 0.4) for y in (-0.4, 0.0, 0.4) for z in (-0.4, 0.0, 0.4)],
        dtype=np.float64,
    )
    residual = np.zeros_like(points)
    residual[:, 0] = 0.1
    solve = fit_active_determinant_tangent_controls(
        lattice,
        points,
        residual,
        current,
        active_ratio=max(0.5, minimum_ratio + 1.0e-6),
        deadline=None,
    )
    checks = {
        "determinant_gradient_matches_finite_difference": abs(analytic - finite_difference)
        <= 1.0e-8,
        "active_constraints_detected": solve.active_tetrahedron_indices.size > 0,
        "kkt_residual_passes": solve.normalized_kkt_residual <= MAXIMUM_NORMALIZED_KKT_RESIDUAL,
        "tangent_residual_passes": solve.maximum_tangent_residual <= MAXIMUM_TANGENT_RESIDUAL,
        "fixed_boundary_preserved": np.all(solve.controls[lattice.boundary_mask] == 0.0),
        "nonzero_tangent_direction": np.any(solve.controls != 0.0),
    }
    native_checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema_version": "post_v1_e21_public_controls.v1",
        "status": "pass" if all(native_checks.values()) else "fail",
        "checks": native_checks,
        "analytic_derivative": analytic,
        "finite_difference_derivative": finite_difference,
        "solve": solve.report(),
    }
