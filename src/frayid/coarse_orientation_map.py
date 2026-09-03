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


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CertifiedCoarseOrientationStepV1:
    raw_controls: np.ndarray
    accepted_controls: np.ndarray
    accepted_displacements: np.ndarray
    accepted_alpha: float
    retained_displacement_ratio: float
    relative_fit_error: float
    proposal_cosine: float
    proposal_path: CertifiedPLPathV1
    accepted_path: CertifiedPLPathV1
    parent_area_report: dict[str, Any]
    status: str
    blockers: tuple[str, ...]
    elapsed_seconds: float
    decision_sha256: str

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "certified_coarse_orientation_step.v1",
            "status": self.status,
            "accepted_alpha": self.accepted_alpha,
            "accepted_alpha_hex": self.accepted_alpha.hex(),
            "retained_displacement_ratio": self.retained_displacement_ratio,
            "relative_fit_error": self.relative_fit_error,
            "proposal_cosine": self.proposal_cosine,
            "proposal_path": self.proposal_path.report(),
            "accepted_serialized_path": self.accepted_path.report(),
            "parent_area": self.parent_area_report,
            "decision_sha256": self.decision_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "blockers": list(self.blockers),
        }


def fit_coarse_controls(
    lattice: FreudenthalLatticeV1,
    carrier_vertices: np.ndarray,
    proposed_displacements: np.ndarray,
    *,
    tikhonov: float,
    rcond: float,
    deadline: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(carrier_vertices, dtype=np.float64)
    proposal = np.asarray(proposed_displacements, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or proposal.shape != vertices.shape:
        raise ValueError("carrier vertices and proposal must share finite shape [V,3]")
    if not np.isfinite(vertices).all() or not np.isfinite(proposal).all():
        raise ValueError("carrier vertices and proposal must be finite")
    if tikhonov < 0.0 or rcond < 0.0:
        raise ValueError("least-squares controls must be nonnegative")
    nodes, weights, _ = lattice.locate(vertices)
    free = np.flatnonzero(~lattice.boundary_mask)
    free_lookup = np.full(lattice.vertices.shape[0], -1, dtype=np.int64)
    free_lookup[free] = np.arange(free.size, dtype=np.int64)
    design = np.zeros((vertices.shape[0], free.size), dtype=np.float64)
    for column in range(4):
        free_columns = free_lookup[nodes[:, column]]
        rows = np.flatnonzero(free_columns >= 0)
        design[rows, free_columns[rows]] += weights[rows, column]
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("coarse orientation lift construction timed out")
    augmented_design = np.vstack((design, math.sqrt(tikhonov) * np.eye(free.size)))
    augmented_target = np.vstack((proposal, np.zeros((free.size, 3), dtype=np.float64)))
    fitted, _, _, _ = np.linalg.lstsq(augmented_design, augmented_target, rcond=rcond)
    controls = np.zeros_like(lattice.vertices)
    controls[free] = fitted
    if np.any(controls[lattice.boundary_mask] != 0.0):
        raise AssertionError("fixed outer boundary changed during coarse fitting")
    return controls, lattice.evaluate(controls, vertices)


def fit_and_certify_coarse_orientation_step(
    lattice: FreudenthalLatticeV1,
    carrier_vertices: np.ndarray,
    carrier_faces: np.ndarray,
    refined_surface: ConformingSurfaceV1,
    proposed_displacements: np.ndarray,
    *,
    minimum_retained_displacement_ratio: float = 0.25,
    tikhonov: float = 1.0e-10,
    rcond: float = 1.0e-12,
    timeout_seconds: float | None = 60.0,
) -> CertifiedCoarseOrientationStepV1:
    """Fit a coarse full-box direction and certify its exact determinant path."""

    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    proposal = np.asarray(proposed_displacements, dtype=np.float64)
    raw_controls, fitted_displacements = fit_coarse_controls(
        lattice,
        carrier_vertices,
        proposal,
        tikhonov=tikhonov,
        rcond=rcond,
        deadline=deadline,
    )
    remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
    raw_surface_direction = lattice.evaluate(raw_controls, refined_surface.reference_vertices)
    proposal_path = certify_piecewise_affine_path(
        lattice.vertices,
        lattice.tetrahedra,
        raw_controls,
        refined_surface.reference_vertices,
        refined_surface.faces,
        raw_surface_direction,
        timeout_seconds=remaining,
    )
    alpha = proposal_path.accepted_alpha
    accepted_controls = np.asarray(raw_controls * alpha, dtype=np.float64)
    if np.any(accepted_controls[lattice.boundary_mask] != 0.0):
        raise AssertionError("accepted coarse map changed the fixed outer boundary")
    accepted_surface = refined_surface.mapped_vertices(lattice, accepted_controls)
    accepted_surface_direction = np.asarray(
        accepted_surface - refined_surface.reference_vertices, dtype=np.float64
    )
    remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
    accepted_path = certify_piecewise_affine_path(
        lattice.vertices,
        lattice.tetrahedra,
        accepted_controls,
        refined_surface.reference_vertices,
        refined_surface.faces,
        accepted_surface_direction,
        timeout_seconds=remaining,
    )
    accepted_displacements = lattice.evaluate(accepted_controls, carrier_vertices)
    proposal_norm = float(np.linalg.norm(proposal))
    fitted_norm = float(np.linalg.norm(fitted_displacements - proposal))
    accepted_norm = float(np.linalg.norm(accepted_displacements))
    retention = accepted_norm / proposal_norm if proposal_norm > 0.0 else 0.0
    relative_fit_error = fitted_norm / proposal_norm if proposal_norm > 0.0 else math.inf
    dot = float(np.sum(accepted_displacements * proposal))
    cosine_denominator = accepted_norm * proposal_norm
    cosine = dot / cosine_denominator if cosine_denominator > 0.0 else 0.0
    parent_area = parent_area_path_report(
        carrier_vertices, carrier_faces, refined_surface, accepted_surface
    )
    blockers: list[str] = []
    if proposal_path.status != "pass":
        blockers.extend(f"proposal_path:{value}" for value in proposal_path.blockers)
    if accepted_path.status != "pass" or accepted_path.accepted_alpha != 1.0:
        blockers.append("accepted_serialized_path_not_fully_certified")
    if proposal_norm <= 0.0:
        blockers.append("nonpositive_proposed_motion")
    if retention < minimum_retained_displacement_ratio:
        blockers.append("motion_retention")
    if accepted_norm <= 0.0:
        blockers.append("zero_motion")
    blockers.extend(parent_area["blockers"])
    decision = _sha256_arrays(
        lattice.vertices.astype("<f8"),
        lattice.tetrahedra.astype("<i8"),
        raw_controls.astype("<f8"),
        accepted_controls.astype("<f8"),
        accepted_displacements.astype("<f8"),
        np.asarray([alpha], dtype="<f8"),
    )
    return CertifiedCoarseOrientationStepV1(
        raw_controls=raw_controls,
        accepted_controls=accepted_controls,
        accepted_displacements=accepted_displacements,
        accepted_alpha=alpha,
        retained_displacement_ratio=retention,
        relative_fit_error=relative_fit_error,
        proposal_cosine=cosine,
        proposal_path=proposal_path,
        accepted_path=accepted_path,
        parent_area_report=parent_area,
        status="pass" if not blockers else "fail",
        blockers=tuple(blockers),
        elapsed_seconds=time.monotonic() - started,
        decision_sha256=decision,
    )


def run_coarse_orientation_controls() -> dict[str, Any]:
    unit_tetrahedron = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tetrahedron = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    face = np.asarray([[0, 1, 2]], dtype=np.int64)
    shear = np.zeros_like(unit_tetrahedron)
    shear[2, 0] = 1.5
    shear_path = certify_piecewise_affine_path(
        unit_tetrahedron,
        tetrahedron,
        shear,
        unit_tetrahedron,
        face,
        shear,
        timeout_seconds=None,
    )
    fold = np.zeros_like(unit_tetrahedron)
    fold[1, 0] = -2.0
    fold_path = certify_piecewise_affine_path(
        unit_tetrahedron,
        tetrahedron,
        fold,
        unit_tetrahedron,
        face,
        fold,
        timeout_seconds=None,
    )
    interior_inversion = np.zeros_like(unit_tetrahedron)
    interior_inversion[1, 0] = -3.0
    interior_inversion[2, 1] = -2.0
    inversion_path = certify_piecewise_affine_path(
        unit_tetrahedron,
        tetrahedron,
        interior_inversion,
        unit_tetrahedron,
        face,
        interior_inversion,
        timeout_seconds=None,
    )
    lattice = FreudenthalLatticeV1.create(
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        nodes_per_axis=4,
    )
    positive = np.zeros_like(lattice.vertices)
    positive[np.flatnonzero(~lattice.boundary_mask), 0] = 0.02
    triangle = np.asarray([[-0.4, -0.2, 0.0], [0.4, -0.2, 0.0], [0.0, 0.4, 0.0]], dtype=np.float64)
    positive_path = certify_piecewise_affine_path(
        lattice.vertices,
        lattice.tetrahedra,
        positive,
        triangle,
        np.asarray([[0, 1, 2]], dtype=np.int64),
        lattice.evaluate(positive, triangle),
        timeout_seconds=None,
    )
    checks = {
        "large_shear_accepted_without_lipschitz_bound": (
            shear_path.status == "pass" and shear_path.accepted_alpha == 1.0
        ),
        "fold_truncated_before_first_singularity": (
            fold_path.status == "pass" and 0.0 < fold_path.accepted_alpha < 1.0
        ),
        "positive_endpoints_interior_inversion_truncated": (
            inversion_path.status == "pass" and 0.0 < inversion_path.accepted_alpha < 1.0
        ),
        "fixed_boundary_positive_interior_map": (
            positive_path.status == "pass"
            and positive_path.accepted_alpha == 1.0
            and np.all(positive[lattice.boundary_mask] == 0.0)
        ),
    }
    serializable_checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema_version": "post_v1_e19_public_controls.v1",
        "status": "pass" if all(serializable_checks.values()) else "fail",
        "checks": serializable_checks,
        "shear": shear_path.report(),
        "fold": fold_path.report(),
        "interior_inversion": inversion_path.report(),
        "fixed_boundary": positive_path.report(),
    }
