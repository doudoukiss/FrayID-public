from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from frayid.batched_planar_dat import (
    _deterministic_segmented_minimum,
    _extract_candidate_batch,
    _isotropic_betas,
)
from frayid.isolated_upstream_arbitration import _isolated_upstream_ratios
from frayid.planar_dat_certificate import RELAXED_RADIUS_SCALING, UPDATE_THRESHOLD, _array_sha256

CORRECTNESS_ID = "postv1_p11_full_isolated_upstream_filter_r01"


@dataclass(frozen=True)
class FullIsolatedUpstreamFilter:
    status: str
    accepted_vertices: np.ndarray
    filtered_displacements: np.ndarray
    truncation_ratios: np.ndarray
    candidate_ids: np.ndarray
    candidate_kinds: np.ndarray
    candidate_keys: tuple[str, ...]
    isolated_contributions: np.ndarray
    candidate_count: int
    edge_edge_count: int
    face_vertex_count: int
    singleton_failures: int
    retained_displacement_ratio: float
    filtered_displacement_norm: float
    restricted_vertex_count: int
    coefficient_seconds: float
    scalar_seconds: float
    batched_seconds: tuple[float, ...]
    full_isolated_filter_seconds: float
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
            "isolated_contributions",
        ):
            result.pop(field)
        result.update(
            {
                "schema_version": "post_v1_p11_full_isolated_upstream_filter.v1",
                "correctness_id": CORRECTNESS_ID,
                "accepted_vertices_sha256": _array_sha256(self.accepted_vertices),
                "filtered_displacements_sha256": _array_sha256(self.filtered_displacements),
                "truncation_ratios_sha256": _array_sha256(self.truncation_ratios),
                "candidate_ids_sha256": _array_sha256(self.candidate_ids),
                "candidate_kinds_sha256": _array_sha256(self.candidate_kinds),
                "isolated_contributions_sha256": _array_sha256(self.isolated_contributions),
                "candidate_keys_sha256": hashlib.sha256(
                    json.dumps(self.candidate_keys, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
        return result


def full_isolated_upstream_filter(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dhat: float,
    dmin: float = 0.0,
) -> FullIsolatedUpstreamFilter:
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("P11 requires the pinned collision extra: ipctk==1.6.0") from error

    start = np.asfortranarray(vertices_t0, dtype=np.float64)
    proposal = np.asfortranarray(vertices_t1, dtype=np.float64)
    if start.shape != proposal.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("vertices_t0 and vertices_t1 must have equal shape (n, 3)")
    if start.shape[0] != int(collision_mesh.num_vertices):
        raise ValueError("P11 does not support hidden collision-mesh DOFs")
    if not np.isfinite(start).all() or not np.isfinite(proposal).all():
        raise ValueError("trajectory vertices must be finite")
    if not math.isfinite(dhat) or dhat <= 0.0:
        raise ValueError("dhat must be finite and positive")
    if dmin != 0.0:
        raise ValueError("P11 is registered only for dmin == 0")

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
        _coefficients,
        _distance_vectors,
        coefficient_seconds,
        extraction_blockers,
    ) = _extract_candidate_batch(trust_region.candidates, start, edges, faces)
    candidate_keys = tuple(
        f"{int(kind)}:" + ":".join(str(int(value)) for value in local_ids)
        for kind, local_ids in zip(kinds, ids, strict=True)
    )
    displacement = np.asfortranarray(proposal - start, dtype=np.float64)
    isolated = np.ones((len(ids), 4), dtype=np.float64)
    singleton_failures = 0
    filter_started = time.monotonic()
    for candidate_index in range(len(ids)):
        local_ids = ids[candidate_index]
        ratios, _candidate_count = _isolated_upstream_ratios(
            start[local_ids],
            displacement[local_ids],
            int(kinds[candidate_index]),
            dhat,
        )
        if ratios is None:
            singleton_failures += 1
            continue
        isolated[candidate_index] = ratios
    filter_seconds = time.monotonic() - filter_started

    ratios = _deterministic_segmented_minimum(ids, isolated, len(start))
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
    if not np.isfinite(filtered).all() or not np.isfinite(accepted).all():
        blockers.append("nonfinite_filtered_path")
    return FullIsolatedUpstreamFilter(
        status="pass" if not blockers else "fail",
        accepted_vertices=accepted,
        filtered_displacements=filtered,
        truncation_ratios=ratios,
        candidate_ids=ids,
        candidate_kinds=kinds,
        candidate_keys=candidate_keys,
        isolated_contributions=isolated,
        candidate_count=len(ids),
        edge_edge_count=int(np.count_nonzero(kinds == 0)),
        face_vertex_count=int(np.count_nonzero(kinds == 1)),
        singleton_failures=singleton_failures,
        retained_displacement_ratio=retention,
        filtered_displacement_norm=filtered_norm,
        restricted_vertex_count=int(np.count_nonzero(ratios < 1.0)),
        coefficient_seconds=coefficient_seconds,
        scalar_seconds=0.0,
        batched_seconds=(filter_seconds,),
        full_isolated_filter_seconds=filter_seconds,
        blockers=tuple(blockers),
    )
