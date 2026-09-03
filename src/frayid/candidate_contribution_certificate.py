from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from frayid.batched_planar_dat import (
    APPROACH_TOLERANCE,
    CROSSING_TIME_EPSILON,
    DENOMINATOR_EPSILON,
    DISTANCE_EPSILON,
    EE_PARALLEL_TOLERANCE_SQUARED,
    CandidateContributions,
    _batched_candidate_contributions,
    _deterministic_segmented_minimum,
    _extract_candidate_batch,
    _isotropic_betas,
)
from frayid.planar_dat_certificate import (
    RELAXED_RADIUS_SCALING,
    UPDATE_THRESHOLD,
    _array_sha256,
    _candidate_keys,
)

CORRECTNESS_ID = "postv1_p9_candidate_contribution_certificate_r01"


@dataclass(frozen=True)
class CandidateContributionCertificate:
    status: str
    accepted_vertices: np.ndarray
    filtered_displacements: np.ndarray
    truncation_ratios: np.ndarray
    candidate_ids: np.ndarray
    candidate_kinds: np.ndarray
    candidate_keys: tuple[str, ...]
    scalar_contributions: np.ndarray
    batched_contributions: np.ndarray
    scalar_skipped: np.ndarray
    batched_skipped: np.ndarray
    scalar_constrained: np.ndarray
    batched_constrained: np.ndarray
    candidate_count: int
    edge_edge_count: int
    face_vertex_count: int
    restricted_vertex_count: int
    retained_displacement_ratio: float
    filtered_displacement_norm: float
    maximum_absolute_contribution_difference: float
    maximum_absolute_vertex_minimum_difference: float
    skipped_decision_mismatches: int
    constrained_decision_mismatches: int
    scalar_seconds: float
    batched_seconds: tuple[float, float]
    coefficient_seconds: float
    bitwise_batched_repeat: bool
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        result = asdict(self)
        for field in (
            "accepted_vertices",
            "filtered_displacements",
            "truncation_ratios",
            "candidate_ids",
            "candidate_kinds",
            "candidate_keys",
            "scalar_contributions",
            "batched_contributions",
            "scalar_skipped",
            "batched_skipped",
            "scalar_constrained",
            "batched_constrained",
        ):
            result.pop(field)
        result.update(
            {
                "schema_version": "post_v1_p9_candidate_contribution_certificate.v1",
                "correctness_id": CORRECTNESS_ID,
                "accepted_vertices_sha256": _array_sha256(self.accepted_vertices),
                "filtered_displacements_sha256": _array_sha256(self.filtered_displacements),
                "truncation_ratios_sha256": _array_sha256(self.truncation_ratios),
                "candidate_ids_sha256": _array_sha256(self.candidate_ids),
                "candidate_kinds_sha256": _array_sha256(self.candidate_kinds),
                "candidate_keys_sha256": hashlib.sha256(
                    json.dumps(self.candidate_keys, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "scalar_contributions_sha256": _array_sha256(self.scalar_contributions),
                "batched_contributions_sha256": _array_sha256(self.batched_contributions),
            }
        )
        return result


def _scalar_candidate_contributions(
    vertices: np.ndarray,
    displacements: np.ndarray,
    ids: np.ndarray,
    kinds: np.ndarray,
    coefficients: np.ndarray,
    distance_vectors: np.ndarray,
) -> CandidateContributions:
    started = time.monotonic()
    values = np.ones((len(ids), 4), dtype=np.float64)
    skipped = np.zeros(len(ids), dtype=np.bool_)
    constrained = np.zeros((len(ids), 4), dtype=np.bool_)
    for candidate_index in range(len(ids)):
        local_ids = ids[candidate_index]
        positions = vertices[local_ids]
        local_dx = displacements[local_ids]
        coeffs = coefficients[candidate_index]
        distance_vector = distance_vectors[candidate_index]
        distance = float(np.linalg.norm(distance_vector))
        if distance < DISTANCE_EPSILON:
            skipped[candidate_index] = True
            continue
        normal = distance_vector / distance
        closest_second = np.zeros(3, dtype=np.float64)
        delta_first = 0.0
        delta_second = 0.0
        for local_index in range(4):
            coefficient = float(coeffs[local_index])
            dot = float(np.dot(local_dx[local_index], normal))
            if coefficient > 0.0:
                delta_first = max(delta_first, -dot)
            elif coefficient < 0.0:
                closest_second -= coefficient * positions[local_index]
                delta_second = max(delta_second, dot)
        delta_first = max(delta_first, 0.0)
        delta_second = max(delta_second, 0.0)

        if kinds[candidate_index] == 0 and (
            delta_first + delta_second <= APPROACH_TOLERANCE * distance
        ):
            edge_a = positions[1] - positions[0]
            edge_b = positions[3] - positions[2]
            denominator = float(np.dot(edge_a, edge_a) * np.dot(edge_b, edge_b))
            with np.errstate(divide="ignore", invalid="ignore"):
                sin_angle_squared = (
                    float(np.dot(np.cross(edge_a, edge_b), np.cross(edge_a, edge_b))) / denominator
                )
            if sin_angle_squared < EE_PARALLEL_TOLERANCE_SQUARED:
                skipped[candidate_index] = True
                continue

        total_delta = delta_first + delta_second
        fraction = 0.5 if total_delta == 0.0 else delta_second / total_delta
        plane_point = closest_second + fraction * distance_vector
        for local_index in range(4):
            denominator = float(np.dot(local_dx[local_index], normal))
            if abs(denominator) < DENOMINATOR_EPSILON:
                continue
            crossing_time = (
                -float(np.dot(positions[local_index] - plane_point, normal)) / denominator
            )
            if (
                crossing_time < CROSSING_TIME_EPSILON
                or crossing_time >= 1.0 / RELAXED_RADIUS_SCALING
            ):
                continue
            constrained[candidate_index, local_index] = True
            values[candidate_index, local_index] = RELAXED_RADIUS_SCALING * crossing_time
    return CandidateContributions(
        values=values,
        skipped=skipped,
        constrained=constrained,
        elapsed_seconds=time.monotonic() - started,
    )


def certify_candidate_contributions(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dhat: float,
    dmin: float = 0.0,
) -> CandidateContributionCertificate:
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("P9 requires the pinned collision extra: ipctk==1.6.0") from error

    start = np.asfortranarray(vertices_t0, dtype=np.float64)
    proposal = np.asfortranarray(vertices_t1, dtype=np.float64)
    if start.shape != proposal.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("vertices_t0 and vertices_t1 must have equal shape (n, 3)")
    if start.shape[0] != int(collision_mesh.num_vertices):
        raise ValueError("P9 does not support hidden collision-mesh DOFs")
    if not np.isfinite(start).all() or not np.isfinite(proposal).all():
        raise ValueError("trajectory vertices must be finite")
    if not math.isfinite(dhat) or dhat <= 0.0:
        raise ValueError("dhat must be finite and positive")
    if dmin != 0.0:
        raise ValueError("P9 is registered only for dmin == 0")

    trust_region = ipctk.ogc.TrustRegion(dhat)
    trust_region.relaxed_radius_scaling = RELAXED_RADIUS_SCALING
    trust_region.update_threshold = UPDATE_THRESHOLD
    collisions = ipctk.NormalCollisions()
    trust_region.update(collision_mesh, start, collisions, dmin, ipctk.SweepAndPrune())
    edges = np.asarray(collision_mesh.edges, dtype=np.int64)
    faces = np.asarray(collision_mesh.faces, dtype=np.int64)
    (
        ids,
        kinds,
        coefficients,
        distance_vectors,
        coefficient_seconds,
        extraction_blockers,
    ) = _extract_candidate_batch(trust_region.candidates, start, edges, faces)
    displacement = np.asfortranarray(proposal - start, dtype=np.float64)
    scalar = _scalar_candidate_contributions(
        start, displacement, ids, kinds, coefficients, distance_vectors
    )
    first = _batched_candidate_contributions(
        start, displacement, ids, kinds, coefficients, distance_vectors
    )
    second = _batched_candidate_contributions(
        start, displacement, ids, kinds, coefficients, distance_vectors
    )
    bitwise = bool(
        np.array_equal(first.values, second.values)
        and np.array_equal(first.skipped, second.skipped)
        and np.array_equal(first.constrained, second.constrained)
    )
    maximum_contribution_difference = float(
        np.max(np.abs(first.values - scalar.values), initial=0.0)
    )
    skipped_mismatches = int(np.count_nonzero(first.skipped != scalar.skipped))
    constrained_mismatches = int(np.count_nonzero(first.constrained != scalar.constrained))
    scalar_ratios = _deterministic_segmented_minimum(ids, scalar.values, len(start))
    batched_ratios = _deterministic_segmented_minimum(ids, first.values, len(start))
    centers = np.asarray(trust_region.trust_region_centers, dtype=np.float64)
    isotropic = _isotropic_betas(
        start,
        displacement,
        centers,
        float(trust_region.trust_region_inflation_radius),
    )
    scalar_ratios = np.minimum(scalar_ratios, isotropic)
    batched_ratios = np.minimum(batched_ratios, isotropic)
    maximum_vertex_difference = float(np.max(np.abs(batched_ratios - scalar_ratios), initial=0.0))
    filtered = np.asfortranarray(displacement * batched_ratios[:, None])
    accepted = np.asfortranarray(start + filtered)
    proposed_norm = float(np.linalg.norm(displacement))
    filtered_norm = float(np.linalg.norm(filtered))
    retention = 1.0 if proposed_norm == 0.0 else filtered_norm / proposed_norm
    blockers = list(extraction_blockers)
    if maximum_contribution_difference > 1e-12:
        blockers.append("candidate_contribution_difference")
    if skipped_mismatches:
        blockers.append("candidate_skip_decision")
    if constrained_mismatches:
        blockers.append("candidate_constrain_decision")
    if maximum_vertex_difference > 1e-12:
        blockers.append("vertex_minimum_difference")
    if not bitwise:
        blockers.append("batched_nondeterminism")
    if not np.isfinite(filtered).all() or not np.isfinite(accepted).all():
        blockers.append("nonfinite_filtered_path")

    return CandidateContributionCertificate(
        status="pass" if not blockers else "fail",
        accepted_vertices=accepted,
        filtered_displacements=filtered,
        truncation_ratios=batched_ratios,
        candidate_ids=ids,
        candidate_kinds=kinds,
        candidate_keys=_candidate_keys(trust_region.candidates),
        scalar_contributions=scalar.values,
        batched_contributions=first.values,
        scalar_skipped=scalar.skipped,
        batched_skipped=first.skipped,
        scalar_constrained=scalar.constrained,
        batched_constrained=first.constrained,
        candidate_count=len(ids),
        edge_edge_count=int(np.count_nonzero(kinds == 0)),
        face_vertex_count=int(np.count_nonzero(kinds == 1)),
        restricted_vertex_count=int(np.count_nonzero(batched_ratios < 1.0)),
        retained_displacement_ratio=retention,
        filtered_displacement_norm=filtered_norm,
        maximum_absolute_contribution_difference=maximum_contribution_difference,
        maximum_absolute_vertex_minimum_difference=maximum_vertex_difference,
        skipped_decision_mismatches=skipped_mismatches,
        constrained_decision_mismatches=constrained_mismatches,
        scalar_seconds=scalar.elapsed_seconds,
        batched_seconds=(first.elapsed_seconds, second.elapsed_seconds),
        coefficient_seconds=coefficient_seconds,
        bitwise_batched_repeat=bitwise,
        blockers=tuple(blockers),
    )
