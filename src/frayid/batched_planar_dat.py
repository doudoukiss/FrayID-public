from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from frayid.planar_dat_certificate import (
    RELAXED_RADIUS_SCALING,
    UPDATE_THRESHOLD,
    _array_sha256,
    _candidate_keys,
)

CORRECTNESS_ID = "postv1_p8_deterministic_batched_planar_dat_r01"
DISTANCE_EPSILON = 1e-10
APPROACH_TOLERANCE = 1e-3
EE_PARALLEL_TOLERANCE_SQUARED = 1e-6
DENOMINATOR_EPSILON = 1e-15
CROSSING_TIME_EPSILON = 1e-10


@dataclass(frozen=True)
class BatchedPlanarDATResult:
    status: str
    accepted_vertices: np.ndarray
    filtered_displacements: np.ndarray
    truncation_ratios: np.ndarray
    trust_region_centers: np.ndarray
    trust_region_radii: np.ndarray
    candidate_ids: np.ndarray
    candidate_kinds: np.ndarray
    candidate_keys: tuple[str, ...]
    candidate_count: int
    edge_edge_count: int
    face_vertex_count: int
    restricted_vertex_count: int
    proposed_displacement_norm: float
    filtered_displacement_norm: float
    retained_displacement_ratio: float
    coefficient_seconds: float
    reduction_seconds: float
    mechanism_seconds: float
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        result = asdict(self)
        for field in (
            "accepted_vertices",
            "filtered_displacements",
            "truncation_ratios",
            "trust_region_centers",
            "trust_region_radii",
            "candidate_ids",
            "candidate_kinds",
            "candidate_keys",
        ):
            result.pop(field)
        result.update(
            {
                "schema_version": "post_v1_p8_batched_planar_dat.v1",
                "correctness_id": CORRECTNESS_ID,
                "accepted_vertices_sha256": _array_sha256(self.accepted_vertices),
                "filtered_displacements_sha256": _array_sha256(self.filtered_displacements),
                "truncation_ratios_sha256": _array_sha256(self.truncation_ratios),
                "trust_region_centers_sha256": _array_sha256(self.trust_region_centers),
                "trust_region_radii_sha256": _array_sha256(self.trust_region_radii),
                "candidate_ids_sha256": _array_sha256(self.candidate_ids),
                "candidate_kinds_sha256": _array_sha256(self.candidate_kinds),
                "candidate_keys_sha256": hashlib.sha256(
                    json.dumps(self.candidate_keys, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
        return result


@dataclass(frozen=True)
class CandidateContributions:
    values: np.ndarray
    skipped: np.ndarray
    constrained: np.ndarray
    elapsed_seconds: float


def _extract_candidate_batch(
    candidates: Any,
    vertices: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, tuple[str, ...]]:
    unsupported = (
        len(candidates.vv_candidates)
        + len(candidates.ev_candidates)
        + len(candidates.pv_candidates)
    )
    blockers: list[str] = []
    if unsupported:
        blockers.append("unsupported_candidate_stencil")

    edge_edge = sorted(
        candidates.ee_candidates,
        key=lambda value: (
            min(int(value.edge0_id), int(value.edge1_id)),
            max(int(value.edge0_id), int(value.edge1_id)),
        ),
    )
    face_vertex = sorted(
        candidates.fv_candidates,
        key=lambda value: (int(value.face_id), int(value.vertex_id)),
    )
    count = len(edge_edge) + len(face_vertex)
    ids = np.empty((count, 4), dtype=np.int64)
    kinds = np.empty(count, dtype=np.uint8)
    coefficients = np.empty((count, 4), dtype=np.float64)
    distance_vectors = np.empty((count, 3), dtype=np.float64)

    started = time.monotonic()
    cursor = 0
    for candidate in edge_edge:
        edge0 = edges[int(candidate.edge0_id)]
        edge1 = edges[int(candidate.edge1_id)]
        candidate_ids = np.array([edge0[0], edge0[1], edge1[0], edge1[1]], dtype=np.int64)
        positions = np.ascontiguousarray(vertices[candidate_ids]).reshape(-1)
        distance_vector, coeffs = candidate.compute_distance_vector_with_coefficients(positions)
        ids[cursor] = candidate_ids
        kinds[cursor] = 0
        coefficients[cursor] = np.asarray(coeffs, dtype=np.float64)
        distance_vectors[cursor] = np.asarray(distance_vector, dtype=np.float64)
        cursor += 1
    for candidate in face_vertex:
        face = faces[int(candidate.face_id)]
        candidate_ids = np.array([candidate.vertex_id, face[0], face[1], face[2]], dtype=np.int64)
        positions = np.ascontiguousarray(vertices[candidate_ids]).reshape(-1)
        distance_vector, coeffs = candidate.compute_distance_vector_with_coefficients(positions)
        ids[cursor] = candidate_ids
        kinds[cursor] = 1
        coefficients[cursor] = np.asarray(coeffs, dtype=np.float64)
        distance_vectors[cursor] = np.asarray(distance_vector, dtype=np.float64)
        cursor += 1
    coefficient_seconds = time.monotonic() - started

    if cursor != len(candidates):
        blockers.append("candidate_count_mismatch")
    if not (
        np.isfinite(coefficients).all()
        and np.isfinite(distance_vectors).all()
        and np.all(ids >= 0)
        and np.all(ids < len(vertices))
    ):
        blockers.append("invalid_candidate_batch")
    return (
        ids,
        kinds,
        coefficients,
        distance_vectors,
        coefficient_seconds,
        tuple(blockers),
    )


def _isotropic_betas(
    vertices: np.ndarray,
    displacements: np.ndarray,
    centers: np.ndarray,
    radius: float,
) -> np.ndarray:
    betas = np.ones(len(vertices), dtype=np.float64)
    offset = vertices - centers
    endpoint_offset = offset + displacements
    endpoint_norm = np.linalg.norm(endpoint_offset, axis=1)
    outside = endpoint_norm > radius
    moving = np.einsum("ij,ij->i", displacements, displacements) != 0.0
    solve = outside & moving
    if not np.any(solve):
        return betas

    dx = displacements[solve]
    local_offset = offset[solve]
    a = np.einsum("ij,ij->i", dx, dx)
    b = 2.0 * np.einsum("ij,ij->i", dx, local_offset)
    c = np.einsum("ij,ij->i", local_offset, local_offset) - radius * radius
    discriminant = np.maximum(b * b - 4.0 * a * c, 0.0)
    square_root = np.sqrt(discriminant)
    positive_b = b >= 0.0
    local_betas = np.empty_like(a)
    local_betas[positive_b] = -2.0 * c[positive_b] / (b[positive_b] + square_root[positive_b])
    local_betas[~positive_b] = (-b[~positive_b] + square_root[~positive_b]) / (2.0 * a[~positive_b])
    betas[solve] = local_betas
    return betas


def _deterministic_segmented_minimum(
    vertex_ids: np.ndarray,
    candidate_values: np.ndarray,
    vertex_count: int,
) -> np.ndarray:
    flat_ids = vertex_ids.reshape(-1)
    flat_values = candidate_values.reshape(-1)
    candidate_ordinals = np.repeat(np.arange(len(vertex_ids), dtype=np.int64), 4)
    local_ordinals = np.tile(np.arange(4, dtype=np.int64), len(vertex_ids))
    order = np.lexsort((local_ordinals, candidate_ordinals, flat_ids))
    sorted_ids = flat_ids[order]
    sorted_values = flat_values[order]
    starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
    result = np.ones(vertex_count, dtype=np.float64)
    result[sorted_ids[starts]] = np.minimum.reduceat(sorted_values, starts)
    return result


def _batched_candidate_contributions(
    vertices: np.ndarray,
    displacements: np.ndarray,
    ids: np.ndarray,
    kinds: np.ndarray,
    coefficients: np.ndarray,
    distance_vectors: np.ndarray,
) -> CandidateContributions:
    started = time.monotonic()
    if len(ids) == 0:
        shape = (0, 4)
        return CandidateContributions(
            values=np.ones(shape, dtype=np.float64),
            skipped=np.ones(0, dtype=np.bool_),
            constrained=np.zeros(shape, dtype=np.bool_),
            elapsed_seconds=time.monotonic() - started,
        )

    positions = vertices[ids]
    local_displacements = displacements[ids]
    distances = np.linalg.norm(distance_vectors, axis=1)
    valid = distances >= DISTANCE_EPSILON
    normals = np.zeros_like(distance_vectors)
    normals[valid] = distance_vectors[valid] / distances[valid, None]

    positive = coefficients > 0.0
    negative = coefficients < 0.0
    weighted = coefficients[:, :, None] * positions
    closest_second = np.sum(np.where(negative[:, :, None], -weighted, 0.0), axis=1)
    velocity_dot_normal = np.einsum("mjd,md->mj", local_displacements, normals, optimize=False)
    delta_first = np.max(np.where(positive, -velocity_dot_normal, 0.0), axis=1, initial=0.0)
    delta_second = np.max(np.where(negative, velocity_dot_normal, 0.0), axis=1, initial=0.0)

    skip = ~valid
    edge_edge = kinds == 0
    negligible = delta_first + delta_second <= APPROACH_TOLERANCE * distances
    first_edges = positions[:, 1] - positions[:, 0]
    second_edges = positions[:, 3] - positions[:, 2]
    cross_squared = np.einsum(
        "ij,ij->i", np.cross(first_edges, second_edges), np.cross(first_edges, second_edges)
    )
    edge_norm_product = np.einsum("ij,ij->i", first_edges, first_edges) * np.einsum(
        "ij,ij->i", second_edges, second_edges
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_angle_squared = cross_squared / edge_norm_product
    skip |= edge_edge & negligible & (sin_angle_squared < EE_PARALLEL_TOLERANCE_SQUARED)

    total_delta = delta_first + delta_second
    lambdas = np.full(len(ids), 0.5, dtype=np.float64)
    moving = total_delta != 0.0
    lambdas[moving] = delta_second[moving] / total_delta[moving]
    plane_points = closest_second + lambdas[:, None] * distance_vectors

    denominator = velocity_dot_normal
    numerator = -np.einsum(
        "mjd,md->mj", positions - plane_points[:, None, :], normals, optimize=False
    )
    values = np.ones_like(denominator)
    nonparallel = np.abs(denominator) >= DENOMINATOR_EPSILON
    crossing_times = np.zeros_like(denominator)
    crossing_times[nonparallel] = numerator[nonparallel] / denominator[nonparallel]
    constrain = (
        nonparallel
        & (crossing_times >= CROSSING_TIME_EPSILON)
        & (crossing_times < 1.0 / RELAXED_RADIUS_SCALING)
        & ~skip[:, None]
    )
    values[constrain] = RELAXED_RADIUS_SCALING * crossing_times[constrain]
    return CandidateContributions(
        values=values,
        skipped=skip,
        constrained=constrain,
        elapsed_seconds=time.monotonic() - started,
    )


def _batched_truncation_ratios(
    vertices: np.ndarray,
    displacements: np.ndarray,
    ids: np.ndarray,
    kinds: np.ndarray,
    coefficients: np.ndarray,
    distance_vectors: np.ndarray,
    centers: np.ndarray,
    inflation_radius: float,
) -> np.ndarray:
    contributions = _batched_candidate_contributions(
        vertices,
        displacements,
        ids,
        kinds,
        coefficients,
        distance_vectors,
    )

    ratios = _deterministic_segmented_minimum(ids, contributions.values, len(vertices))
    return np.asarray(
        np.minimum(
            ratios,
            _isotropic_betas(vertices, displacements, centers, inflation_radius),
        ),
        dtype=np.float64,
    )


def batched_planar_dat_path(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dhat: float,
    dmin: float = 0.0,
) -> BatchedPlanarDATResult:
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("P8 requires the pinned collision extra: ipctk==1.6.0") from error

    started = time.monotonic()
    start = np.asfortranarray(vertices_t0, dtype=np.float64)
    proposal = np.asfortranarray(vertices_t1, dtype=np.float64)
    if start.shape != proposal.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("vertices_t0 and vertices_t1 must have equal shape (n, 3)")
    if start.shape[0] != int(collision_mesh.num_vertices):
        raise ValueError("P8 does not support hidden collision-mesh DOFs")
    if not np.isfinite(start).all() or not np.isfinite(proposal).all():
        raise ValueError("trajectory vertices must be finite")
    if not math.isfinite(dhat) or dhat <= 0.0:
        raise ValueError("dhat must be finite and positive")
    if dmin != 0.0:
        raise ValueError("P8 is registered only for dmin == 0")

    trust_region = ipctk.ogc.TrustRegion(dhat)
    trust_region.relaxed_radius_scaling = RELAXED_RADIUS_SCALING
    trust_region.update_threshold = UPDATE_THRESHOLD
    collisions = ipctk.NormalCollisions()
    trust_region.update(
        collision_mesh,
        start,
        collisions,
        dmin,
        ipctk.SweepAndPrune(),
    )
    centers = np.asarray(trust_region.trust_region_centers, dtype=np.float64).copy()
    radii = np.asarray(trust_region.trust_region_radii, dtype=np.float64).copy()
    candidate_keys = _candidate_keys(trust_region.candidates)
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

    reduction_started = time.monotonic()
    displacement = np.asfortranarray(proposal - start, dtype=np.float64)
    ratios = _batched_truncation_ratios(
        start,
        displacement,
        ids,
        kinds,
        coefficients,
        distance_vectors,
        centers,
        float(trust_region.trust_region_inflation_radius),
    )
    filtered = np.asfortranarray(displacement * ratios[:, None], dtype=np.float64)
    accepted = np.asfortranarray(start + filtered, dtype=np.float64)
    reduction_seconds = time.monotonic() - reduction_started

    proposed_norm = float(np.linalg.norm(displacement))
    filtered_norm = float(np.linalg.norm(filtered))
    retained_ratio = 1.0 if proposed_norm == 0.0 else filtered_norm / proposed_norm
    blockers = list(extraction_blockers)
    if not (
        np.isfinite(ratios).all()
        and np.isfinite(filtered).all()
        and np.isfinite(accepted).all()
        and np.all((ratios >= 0.0) & (ratios <= 1.0))
    ):
        blockers.append("invalid_filtered_path")
    if not math.isfinite(retained_ratio):
        blockers.append("invalid_retained_displacement_ratio")

    return BatchedPlanarDATResult(
        status="pass" if not blockers else "fail",
        accepted_vertices=accepted,
        filtered_displacements=filtered,
        truncation_ratios=ratios,
        trust_region_centers=centers,
        trust_region_radii=radii,
        candidate_ids=ids,
        candidate_kinds=kinds,
        candidate_keys=candidate_keys,
        candidate_count=len(ids),
        edge_edge_count=int(np.count_nonzero(kinds == 0)),
        face_vertex_count=int(np.count_nonzero(kinds == 1)),
        restricted_vertex_count=int(np.count_nonzero(ratios < 1.0)),
        proposed_displacement_norm=proposed_norm,
        filtered_displacement_norm=filtered_norm,
        retained_displacement_ratio=retained_ratio,
        coefficient_seconds=coefficient_seconds,
        reduction_seconds=reduction_seconds,
        mechanism_seconds=time.monotonic() - started,
        blockers=tuple(blockers),
    )


def candidate_batch_sha256(result: BatchedPlanarDATResult) -> str:
    payload = {
        "ids": _array_sha256(result.candidate_ids),
        "kinds": _array_sha256(result.candidate_kinds),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
