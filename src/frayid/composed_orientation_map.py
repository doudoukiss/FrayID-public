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


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MaterialEmbeddingV1:
    nodes: np.ndarray
    weights: np.ndarray
    tetrahedron_indices: np.ndarray
    content_sha256: str

    @classmethod
    def create(
        cls, lattice: FreudenthalLatticeV1, reference_points: np.ndarray
    ) -> MaterialEmbeddingV1:
        points = np.asarray(reference_points, dtype=np.float64)
        nodes, weights, tetrahedra = lattice.locate(points)
        digest = _sha256_arrays(
            points.astype("<f8"),
            nodes.astype("<i8"),
            weights.astype("<f8"),
            tetrahedra.astype("<i8"),
        )
        return cls(nodes, weights, tetrahedra, digest)

    def evaluate(self, controls: np.ndarray) -> np.ndarray:
        values = np.asarray(controls, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("material controls must have shape [N,3]")
        if np.any(self.nodes < 0) or np.any(self.nodes >= values.shape[0]):
            raise ValueError("material embedding node is out of range")
        return np.asarray(
            np.einsum("ni,nij->nj", self.weights, values[self.nodes]), dtype=np.float64
        )


@dataclass(frozen=True)
class CertifiedOrientationBlockV1:
    index: int
    raw_controls: np.ndarray
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
            "schema_version": "certified_orientation_block.v1",
            "index": self.index,
            "status": self.status,
            "accepted_alpha": self.accepted_alpha,
            "accepted_alpha_hex": self.accepted_alpha.hex(),
            "residual_norm_before": self.residual_norm_before,
            "residual_norm_after": self.residual_norm_after,
            "raw_controls_sha256": hashlib.sha256(
                np.ascontiguousarray(self.raw_controls, dtype="<f8").tobytes()
            ).hexdigest(),
            "accepted_controls_sha256": hashlib.sha256(
                np.ascontiguousarray(self.accepted_controls, dtype="<f8").tobytes()
            ).hexdigest(),
            "proposal_path": self.proposal_path.report(),
            "accepted_serialized_path": self.accepted_path.report(),
            "decision_sha256": self.decision_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class CertifiedComposedOrientationStepV1:
    blocks: tuple[CertifiedOrientationBlockV1, ...]
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
            "schema_version": "certified_composed_orientation_step.v1",
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


def apply_material_control_blocks(
    lattice: FreudenthalLatticeV1,
    reference_points: np.ndarray,
    control_blocks: np.ndarray,
) -> np.ndarray:
    embedding = MaterialEmbeddingV1.create(lattice, reference_points)
    current = np.asarray(reference_points, dtype=np.float64).copy()
    blocks = np.asarray(control_blocks, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[1:] != lattice.vertices.shape:
        raise ValueError("control blocks must have shape [B,N,3]")
    for controls in blocks:
        current = np.asarray(current + embedding.evaluate(controls), dtype=np.float64)
    return current


def fit_and_certify_composed_orientation_step(
    lattice: FreudenthalLatticeV1,
    carrier_vertices: np.ndarray,
    carrier_faces: np.ndarray,
    refined_surface: ConformingSurfaceV1,
    proposed_displacements: np.ndarray,
    *,
    block_count: int = 4,
    minimum_retained_displacement_ratio: float = 0.25,
    tikhonov: float = 1.0e-10,
    rcond: float = 1.0e-12,
    timeout_seconds_per_block: float | None = 60.0,
) -> CertifiedComposedOrientationStepV1:
    """Fit and certify a fixed sequence of material-coordinate PL homeomorphisms."""

    started = time.monotonic()
    if block_count < 1:
        raise ValueError("composed orientation map requires at least one block")
    reference_carrier = np.asarray(carrier_vertices, dtype=np.float64)
    faces = np.asarray(carrier_faces, dtype=np.int64)
    proposal = np.asarray(proposed_displacements, dtype=np.float64)
    if proposal.shape != reference_carrier.shape or not np.isfinite(proposal).all():
        raise ValueError("composed proposal must match finite carrier vertices")
    target = np.asarray(reference_carrier + proposal, dtype=np.float64)
    carrier_embedding = MaterialEmbeddingV1.create(lattice, reference_carrier)
    surface_embedding = MaterialEmbeddingV1.create(lattice, refined_surface.reference_vertices)
    current_lattice = lattice.vertices.copy()
    current_carrier = reference_carrier.copy()
    current_surface = refined_surface.reference_vertices.copy()
    blocks: list[CertifiedOrientationBlockV1] = []
    accepted_controls: list[np.ndarray] = []
    blockers: list[str] = []
    for index in range(block_count):
        block_started = time.monotonic()
        deadline = (
            None if timeout_seconds_per_block is None else block_started + timeout_seconds_per_block
        )
        residual = np.asarray(target - current_carrier, dtype=np.float64)
        raw_controls, _ = fit_coarse_controls(
            lattice,
            reference_carrier,
            residual,
            tikhonov=tikhonov,
            rcond=rcond,
            deadline=deadline,
        )
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
        if np.any(block_controls[lattice.boundary_mask] != 0.0):
            raise AssertionError("composed block changed the fixed outer boundary")
        volume_direction = block_controls
        carrier_direction = carrier_embedding.evaluate(block_controls)
        surface_direction = surface_embedding.evaluate(block_controls)
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        accepted_path = certify_piecewise_affine_path(
            current_lattice,
            lattice.tetrahedra,
            volume_direction,
            current_surface,
            refined_surface.faces,
            surface_direction,
            timeout_seconds=remaining,
        )
        block_blockers: list[str] = []
        if proposal_path.status != "pass":
            block_blockers.extend(f"proposal_path:{value}" for value in proposal_path.blockers)
        if accepted_path.status != "pass" or accepted_path.accepted_alpha != 1.0:
            block_blockers.append("accepted_serialized_path_not_fully_certified")
        if not np.any(block_controls != 0.0):
            block_blockers.append("zero_block_motion")
        next_lattice = np.asarray(current_lattice + volume_direction, dtype=np.float64)
        next_carrier = np.asarray(current_carrier + carrier_direction, dtype=np.float64)
        next_surface = np.asarray(current_surface + surface_direction, dtype=np.float64)
        residual_after = float(np.linalg.norm(target - next_carrier))
        decision = _sha256_arrays(
            current_lattice.astype("<f8"),
            current_carrier.astype("<f8"),
            raw_controls.astype("<f8"),
            block_controls.astype("<f8"),
            next_lattice.astype("<f8"),
            next_carrier.astype("<f8"),
        )
        block = CertifiedOrientationBlockV1(
            index=index,
            raw_controls=raw_controls,
            accepted_controls=block_controls,
            accepted_alpha=alpha,
            proposal_path=proposal_path,
            accepted_path=accepted_path,
            residual_norm_before=float(np.linalg.norm(residual)),
            residual_norm_after=residual_after,
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
    if not accepted_controls:
        control_array = np.empty((0, *lattice.vertices.shape), dtype=np.float64)
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
    return CertifiedComposedOrientationStepV1(
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


def run_composed_orientation_controls() -> dict[str, Any]:
    unit = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tetrahedron = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    face = np.asarray([[0, 1, 2]], dtype=np.int64)
    rotation_90 = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rotation_180 = rotation_90 @ rotation_90
    first_endpoint = np.asarray(unit @ rotation_90.T, dtype=np.float64)
    second_endpoint = np.asarray(unit @ rotation_180.T, dtype=np.float64)
    one_shot = certify_piecewise_affine_path(
        unit,
        tetrahedron,
        second_endpoint - unit,
        unit,
        face,
        second_endpoint - unit,
        timeout_seconds=None,
    )
    first = certify_piecewise_affine_path(
        unit,
        tetrahedron,
        first_endpoint - unit,
        unit,
        face,
        first_endpoint - unit,
        timeout_seconds=None,
    )
    second = certify_piecewise_affine_path(
        first_endpoint,
        tetrahedron,
        second_endpoint - first_endpoint,
        first_endpoint,
        face,
        second_endpoint - first_endpoint,
        timeout_seconds=None,
    )
    lattice = FreudenthalLatticeV1.create(
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        nodes_per_axis=4,
    )
    triangle = np.asarray([[-0.4, -0.2, 0.0], [0.4, -0.2, 0.0], [0.0, 0.4, 0.0]], dtype=np.float64)
    embedding = MaterialEmbeddingV1.create(lattice, triangle)
    current_lattice = lattice.vertices.copy()
    current_triangle = triangle.copy()
    fixed_blocks_pass = True
    for axis in (0, 1):
        controls = np.zeros_like(lattice.vertices)
        controls[np.flatnonzero(~lattice.boundary_mask), axis] = 0.02
        direction = embedding.evaluate(controls)
        path = certify_piecewise_affine_path(
            current_lattice,
            lattice.tetrahedra,
            controls,
            current_triangle,
            face,
            direction,
            timeout_seconds=None,
        )
        fixed_blocks_pass = (
            fixed_blocks_pass and path.status == "pass" and path.accepted_alpha == 1.0
        )
        current_lattice = np.asarray(current_lattice + controls, dtype=np.float64)
        current_triangle = np.asarray(current_triangle + direction, dtype=np.float64)
    checks = {
        "one_shot_half_turn_is_truncated": one_shot.status == "pass"
        and one_shot.accepted_alpha < 1.0,
        "two_orientation_preserving_quarter_turns_pass": first.status == "pass"
        and first.accepted_alpha == 1.0
        and second.status == "pass"
        and second.accepted_alpha == 1.0,
        "material_coordinates_replay": np.array_equal(
            current_triangle,
            apply_material_control_blocks(
                lattice,
                triangle,
                np.asarray(
                    [
                        np.where(
                            (~lattice.boundary_mask)[:, None],
                            np.asarray([0.02, 0.0, 0.0]),
                            0.0,
                        ),
                        np.where(
                            (~lattice.boundary_mask)[:, None],
                            np.asarray([0.0, 0.02, 0.0]),
                            0.0,
                        ),
                    ],
                    dtype=np.float64,
                ),
            ),
        ),
        "fixed_boundary_blocks_pass": fixed_blocks_pass,
    }
    native_checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema_version": "post_v1_e20_public_controls.v1",
        "status": "pass" if all(native_checks.values()) else "fail",
        "checks": native_checks,
        "one_shot": one_shot.report(),
        "first_quarter_turn": first.report(),
        "second_quarter_turn": second.report(),
    }
