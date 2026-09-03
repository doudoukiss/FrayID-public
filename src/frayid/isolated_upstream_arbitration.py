from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from frayid.batched_planar_dat import (
    _batched_candidate_contributions,
    _deterministic_segmented_minimum,
    _extract_candidate_batch,
    _isotropic_betas,
)
from frayid.candidate_contribution_certificate import _scalar_candidate_contributions
from frayid.planar_dat_certificate import (
    RELAXED_RADIUS_SCALING,
    UPDATE_THRESHOLD,
    _array_sha256,
)

CORRECTNESS_ID = "postv1_p10_isolated_upstream_arbitration_r01"
SAMPLE_SEED = 20260831
UNCONSTRAINED_SAMPLE_COUNT = 1024
EQUIVALENCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class IsolatedUpstreamArbitration:
    status: str
    accepted_vertices: np.ndarray
    filtered_displacements: np.ndarray
    truncation_ratios: np.ndarray
    candidate_ids: np.ndarray
    candidate_kinds: np.ndarray
    candidate_keys: tuple[str, ...]
    selected_contributions: np.ndarray
    candidate_count: int
    edge_edge_count: int
    face_vertex_count: int
    disagreement_candidate_count: int
    constrained_candidate_count: int
    sampled_unconstrained_candidate_count: int
    arbitration_candidate_count: int
    singleton_failures: int
    consensus_ratio_mismatches: int
    maximum_consensus_ratio_difference: float
    retained_displacement_ratio: float
    filtered_displacement_norm: float
    restricted_vertex_count: int
    coefficient_seconds: float
    scalar_seconds: float
    batched_seconds: tuple[float, ...]
    arbitration_seconds: float
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
            "selected_contributions",
        ):
            result.pop(field)
        result.update(
            {
                "schema_version": "post_v1_p10_isolated_upstream_arbitration.v1",
                "correctness_id": CORRECTNESS_ID,
                "accepted_vertices_sha256": _array_sha256(self.accepted_vertices),
                "filtered_displacements_sha256": _array_sha256(self.filtered_displacements),
                "truncation_ratios_sha256": _array_sha256(self.truncation_ratios),
                "candidate_ids_sha256": _array_sha256(self.candidate_ids),
                "candidate_kinds_sha256": _array_sha256(self.candidate_kinds),
                "selected_contributions_sha256": _array_sha256(self.selected_contributions),
                "candidate_keys_sha256": hashlib.sha256(
                    json.dumps(self.candidate_keys, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
        return result


def _isolated_upstream_ratios(
    positions: np.ndarray,
    displacements: np.ndarray,
    kind: int,
    dhat: float,
) -> tuple[np.ndarray | None, int]:
    import ipctk  # type: ignore[import-not-found]

    local_positions = np.asfortranarray(positions, dtype=np.float64)
    if kind == 0:
        edges = np.asfortranarray([[0, 1], [2, 3]], dtype=np.int32)
        faces = np.empty((0, 3), dtype=np.int32, order="F")
    elif kind == 1:
        edges = np.asfortranarray([[1, 2], [2, 3], [1, 3]], dtype=np.int32)
        faces = np.asfortranarray([[1, 2, 3]], dtype=np.int32)
    else:
        return None, 0
    mesh = ipctk.CollisionMesh(local_positions, edges, faces)
    trust_region = ipctk.ogc.TrustRegion(dhat)
    trust_region.relaxed_radius_scaling = RELAXED_RADIUS_SCALING
    trust_region.update_threshold = UPDATE_THRESHOLD
    collisions = ipctk.NormalCollisions()
    trust_region.update(
        mesh,
        local_positions,
        collisions,
        0.0,
        ipctk.SweepAndPrune(),
    )
    candidate_count = len(trust_region.candidates)
    expected_type_count = (
        len(trust_region.candidates.ee_candidates)
        if kind == 0
        else len(trust_region.candidates.fv_candidates)
    )
    if candidate_count != 1 or expected_type_count != 1:
        return None, candidate_count
    filtered = np.asfortranarray(displacements, dtype=np.float64).copy(order="F")
    trust_region.planar_filter_step(mesh, local_positions, filtered)
    ratios = np.ones(4, dtype=np.float64)
    for local_index in range(4):
        component = int(np.argmax(np.abs(displacements[local_index])))
        denominator = float(displacements[local_index, component])
        if denominator != 0.0:
            ratios[local_index] = filtered[local_index, component] / denominator
    return ratios, candidate_count


def _sample_unconstrained(candidate_keys: tuple[str, ...], eligible: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    ranked = sorted(
        indices.tolist(),
        key=lambda index: hashlib.sha256(
            f"{SAMPLE_SEED}:{candidate_keys[index]}".encode("ascii")
        ).digest(),
    )
    return np.asarray(ranked[:UNCONSTRAINED_SAMPLE_COUNT], dtype=np.int64)


def arbitrate_isolated_upstream(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dhat: float,
    dmin: float = 0.0,
) -> IsolatedUpstreamArbitration:
    try:
        import ipctk
    except ModuleNotFoundError as error:
        raise RuntimeError("P10 requires the pinned collision extra: ipctk==1.6.0") from error

    start = np.asfortranarray(vertices_t0, dtype=np.float64)
    proposal = np.asfortranarray(vertices_t1, dtype=np.float64)
    if start.shape != proposal.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("vertices_t0 and vertices_t1 must have equal shape (n, 3)")
    if start.shape[0] != int(collision_mesh.num_vertices):
        raise ValueError("P10 does not support hidden collision-mesh DOFs")
    if not np.isfinite(start).all() or not np.isfinite(proposal).all():
        raise ValueError("trajectory vertices must be finite")
    if not math.isfinite(dhat) or dhat <= 0.0:
        raise ValueError("dhat must be finite and positive")
    if dmin != 0.0:
        raise ValueError("P10 is registered only for dmin == 0")

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
    candidate_keys = tuple(
        f"{int(kind)}:" + ":".join(str(int(value)) for value in local_ids)
        for kind, local_ids in zip(kinds, ids, strict=True)
    )
    displacement = np.asfortranarray(proposal - start, dtype=np.float64)
    scalar = _scalar_candidate_contributions(
        start, displacement, ids, kinds, coefficients, distance_vectors
    )
    batched = _batched_candidate_contributions(
        start, displacement, ids, kinds, coefficients, distance_vectors
    )
    decision_disagreement = np.any(
        (scalar.skipped != batched.skipped)[:, None] | (scalar.constrained != batched.constrained),
        axis=1,
    )
    numeric_disagreement = np.any(
        np.abs(scalar.values - batched.values) > EQUIVALENCE_TOLERANCE,
        axis=1,
    )
    disagreement = decision_disagreement | numeric_disagreement
    constrained = np.any(scalar.constrained | batched.constrained, axis=1)
    sampled = _sample_unconstrained(candidate_keys, ~(disagreement | constrained))
    arbitration_indices = np.unique(
        np.concatenate((np.flatnonzero(disagreement), np.flatnonzero(constrained), sampled))
    )

    selected = batched.values.copy()
    singleton_failures = 0
    consensus_mismatches = 0
    maximum_consensus_difference = 0.0
    arbitration_started = time.monotonic()
    for candidate_index in arbitration_indices:
        local_ids = ids[candidate_index]
        upstream_ratios, _candidate_count = _isolated_upstream_ratios(
            start[local_ids],
            displacement[local_ids],
            int(kinds[candidate_index]),
            dhat,
        )
        if upstream_ratios is None:
            singleton_failures += 1
            continue
        if disagreement[candidate_index]:
            selected[candidate_index] = upstream_ratios
            continue
        local_isotropic = _isotropic_betas(
            start[local_ids],
            displacement[local_ids],
            start[local_ids],
            2.0 * dhat,
        )
        expected = np.minimum(selected[candidate_index], local_isotropic)
        difference = float(np.max(np.abs(expected - upstream_ratios), initial=0.0))
        maximum_consensus_difference = max(maximum_consensus_difference, difference)
        if difference > EQUIVALENCE_TOLERANCE:
            consensus_mismatches += 1
    arbitration_seconds = time.monotonic() - arbitration_started

    ratios = _deterministic_segmented_minimum(ids, selected, len(start))
    centers = np.asarray(trust_region.trust_region_centers, dtype=np.float64)
    ratios = np.minimum(
        ratios,
        _isotropic_betas(
            start,
            displacement,
            centers,
            float(trust_region.trust_region_inflation_radius),
        ),
    )
    filtered = np.asfortranarray(displacement * ratios[:, None])
    accepted = np.asfortranarray(start + filtered)
    proposed_norm = float(np.linalg.norm(displacement))
    filtered_norm = float(np.linalg.norm(filtered))
    retention = 1.0 if proposed_norm == 0.0 else filtered_norm / proposed_norm
    blockers = list(extraction_blockers)
    if singleton_failures:
        blockers.append("isolated_candidate_not_singleton")
    if consensus_mismatches:
        blockers.append("isolated_consensus_ratio")
    if not np.isfinite(filtered).all() or not np.isfinite(accepted).all():
        blockers.append("nonfinite_filtered_path")

    return IsolatedUpstreamArbitration(
        status="pass" if not blockers else "fail",
        accepted_vertices=accepted,
        filtered_displacements=filtered,
        truncation_ratios=ratios,
        candidate_ids=ids,
        candidate_kinds=kinds,
        candidate_keys=candidate_keys,
        selected_contributions=selected,
        candidate_count=len(ids),
        edge_edge_count=int(np.count_nonzero(kinds == 0)),
        face_vertex_count=int(np.count_nonzero(kinds == 1)),
        disagreement_candidate_count=int(np.count_nonzero(disagreement)),
        constrained_candidate_count=int(np.count_nonzero(constrained)),
        sampled_unconstrained_candidate_count=len(sampled),
        arbitration_candidate_count=len(arbitration_indices),
        singleton_failures=singleton_failures,
        consensus_ratio_mismatches=consensus_mismatches,
        maximum_consensus_ratio_difference=maximum_consensus_difference,
        retained_displacement_ratio=retention,
        filtered_displacement_norm=filtered_norm,
        restricted_vertex_count=int(np.count_nonzero(ratios < 1.0)),
        coefficient_seconds=coefficient_seconds,
        scalar_seconds=scalar.elapsed_seconds,
        batched_seconds=(batched.elapsed_seconds,),
        arbitration_seconds=arbitration_seconds,
        blockers=tuple(blockers),
    )
