from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CORRECTNESS_ID = "postv1_p3_conservative_collision_partition_r01"
NEAR_INFLATION_FRACTION = 0.5
ACCEPTED_FRACTION = 0.8
TIGHT_INCLUSION_TOLERANCE = 1e-6
TIGHT_INCLUSION_MAX_ITERATIONS = 10_000_000
TIGHT_INCLUSION_CONSERVATIVE_RESCALING = 0.8


@dataclass(frozen=True)
class CollisionPartitionCertificate:
    status: str
    certified_fraction: float
    near_fraction: float
    far_fraction: float
    maximum_vertex_displacement: float
    near_candidate_count: int
    full_swept_candidate_count: int
    near_to_full_swept_candidate_ratio: float
    near_candidate_keys: tuple[str, ...]
    full_oracle_safe: bool
    partition_elapsed_seconds: float
    elapsed_seconds: float
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        candidate_bytes = json.dumps(self.near_candidate_keys, separators=(",", ":")).encode(
            "utf-8"
        )
        result = asdict(self)
        result.pop("near_candidate_keys")
        result.update(
            {
                "schema_version": "post_v1_p3_collision_partition_certificate.v1",
                "correctness_id": CORRECTNESS_ID,
                "near_candidate_keys_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            }
        )
        return result


@dataclass(frozen=True)
class CollisionCandidateSummary:
    near_candidate_count: int
    full_swept_candidate_count: int
    near_to_full_swept_candidate_ratio: float
    near_candidate_keys: tuple[str, ...]
    elapsed_seconds: float

    def report(self) -> dict[str, Any]:
        candidate_bytes = json.dumps(self.near_candidate_keys, separators=(",", ":")).encode(
            "utf-8"
        )
        return {
            "schema_version": "post_v1_p3_collision_candidate_summary.v1",
            "correctness_id": CORRECTNESS_ID,
            "near_candidate_count": self.near_candidate_count,
            "full_swept_candidate_count": self.full_swept_candidate_count,
            "near_to_full_swept_candidate_ratio": self.near_to_full_swept_candidate_ratio,
            "near_candidate_keys_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "elapsed_seconds": self.elapsed_seconds,
        }


def _candidate_keys(candidates: Any) -> tuple[str, ...]:
    keys: list[str] = []
    for value in candidates.vv_candidates:
        a, b = sorted((int(value.vertex0_id), int(value.vertex1_id)))
        keys.append(f"vv:{a}:{b}")
    for value in candidates.ev_candidates:
        keys.append(f"ev:{int(value.edge_id)}:{int(value.vertex_id)}")
    for value in candidates.ee_candidates:
        a, b = sorted((int(value.edge0_id), int(value.edge1_id)))
        keys.append(f"ee:{a}:{b}")
    for value in candidates.fv_candidates:
        keys.append(f"fv:{int(value.face_id)}:{int(value.vertex_id)}")
    for index, value in enumerate(candidates.pv_candidates):
        keys.append(f"pv:{index}:{value}")
    return tuple(sorted(keys))


def collision_candidate_summary(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dhat: float,
) -> CollisionCandidateSummary:
    """Build the registered static-near and full-swept candidate sets."""
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("P3 requires the pinned collision extra: ipctk==1.6.0") from error
    started = time.monotonic()
    start = np.asfortranarray(vertices_t0, dtype=np.float64)
    proposal = np.asfortranarray(vertices_t1, dtype=np.float64)
    if start.shape != proposal.shape:
        raise ValueError("candidate trajectory shapes must match")
    if not math.isfinite(dhat) or dhat <= 0.0:
        raise ValueError("dhat must be finite and positive")
    broad_phase = ipctk.SweepAndPrune()
    near = ipctk.Candidates()
    near.build(collision_mesh, start, NEAR_INFLATION_FRACTION * dhat, broad_phase)
    full_swept = ipctk.Candidates()
    full_swept.build(collision_mesh, start, proposal, 0.0, broad_phase)
    full_count = len(full_swept)
    return CollisionCandidateSummary(
        near_candidate_count=len(near),
        full_swept_candidate_count=full_count,
        near_to_full_swept_candidate_ratio=float(len(near) / full_count) if full_count else 0.0,
        near_candidate_keys=_candidate_keys(near),
        elapsed_seconds=time.monotonic() - started,
    )


def conservative_collision_partition(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dhat: float,
    dmin: float = 0.0,
    verify_full_path: bool = True,
) -> CollisionPartitionCertificate:
    """Certify a linear step using near-pair TI CCD and a global far bound.

    The far-pair proof uses AABBs inflated by ``dhat / 2`` at the start.
    Any pair omitted by that broad phase has an axis separation of at least
    ``dhat``. Each primitive can close that separation by at most the maximum
    vertex displacement, so ``dhat / (2 * vmax)`` is conservative for every
    omitted pair. The proof is restricted to ``dmin == 0`` and linear vertex
    trajectories.
    """
    try:
        import ipctk
    except ModuleNotFoundError as error:
        raise RuntimeError("P3 requires the pinned collision extra: ipctk==1.6.0") from error

    started = time.monotonic()
    start = np.asfortranarray(vertices_t0, dtype=np.float64)
    proposal = np.asfortranarray(vertices_t1, dtype=np.float64)
    if start.shape != proposal.shape or start.ndim != 2 or start.shape[1] not in (2, 3):
        raise ValueError("vertices_t0 and vertices_t1 must have equal shape (n, 2|3)")
    if start.shape[0] != int(collision_mesh.num_vertices):
        raise ValueError("trajectory vertex count does not match collision mesh")
    if not np.isfinite(start).all() or not np.isfinite(proposal).all():
        raise ValueError("trajectory vertices must be finite")
    if not math.isfinite(dhat) or dhat <= 0.0:
        raise ValueError("dhat must be finite and positive")
    if dmin != 0.0:
        raise ValueError("P3 is registered only for dmin == 0")

    broad_phase = ipctk.SweepAndPrune()
    tight_inclusion = ipctk.TightInclusionCCD(
        TIGHT_INCLUSION_TOLERANCE,
        TIGHT_INCLUSION_MAX_ITERATIONS,
        TIGHT_INCLUSION_CONSERVATIVE_RESCALING,
    )
    near = ipctk.Candidates()
    near.build(
        collision_mesh,
        start,
        NEAR_INFLATION_FRACTION * dhat,
        broad_phase,
    )
    keys = _candidate_keys(near)
    near_fraction = float(
        near.compute_collision_free_stepsize(
            collision_mesh,
            start,
            proposal,
            dmin,
            tight_inclusion,
        )
    )
    displacements = proposal - start
    maximum_displacement = float(np.max(np.linalg.norm(displacements, axis=1), initial=0.0))
    far_fraction = (
        1.0 if maximum_displacement == 0.0 else min(1.0, dhat / (2.0 * maximum_displacement))
    )
    certified_fraction = min(
        1.0,
        ACCEPTED_FRACTION * near_fraction,
        ACCEPTED_FRACTION * far_fraction,
    )
    accepted = np.asfortranarray(
        start + certified_fraction * displacements,
        dtype=np.float64,
    )
    partition_elapsed_seconds = time.monotonic() - started

    full_proposal = ipctk.Candidates()
    full_proposal.build(collision_mesh, start, proposal, 0.0, broad_phase)
    full_swept_count = len(full_proposal)
    ratio = float(len(near) / full_swept_count) if full_swept_count else 0.0

    full_oracle_safe = True
    if verify_full_path:
        full_accepted = ipctk.Candidates()
        full_accepted.build(collision_mesh, start, accepted, 0.0, broad_phase)
        full_oracle_safe = bool(
            full_accepted.is_step_collision_free(
                collision_mesh,
                start,
                accepted,
                dmin,
                tight_inclusion,
            )
        )

    blockers: list[str] = []
    if not (0.0 <= near_fraction <= 1.0):
        blockers.append("near_fraction_out_of_range")
    if not (0.0 <= far_fraction <= 1.0):
        blockers.append("far_fraction_out_of_range")
    if not (0.0 <= certified_fraction <= 1.0):
        blockers.append("certified_fraction_out_of_range")
    if len(keys) != len(near):
        blockers.append("candidate_identity_count_mismatch")
    if verify_full_path and not full_oracle_safe:
        blockers.append("complete_dynamic_tight_inclusion_oracle")
    return CollisionPartitionCertificate(
        status="pass" if not blockers else "fail",
        certified_fraction=certified_fraction,
        near_fraction=near_fraction,
        far_fraction=far_fraction,
        maximum_vertex_displacement=maximum_displacement,
        near_candidate_count=len(near),
        full_swept_candidate_count=full_swept_count,
        near_to_full_swept_candidate_ratio=ratio,
        near_candidate_keys=keys,
        full_oracle_safe=full_oracle_safe,
        partition_elapsed_seconds=partition_elapsed_seconds,
        elapsed_seconds=time.monotonic() - started,
        blockers=tuple(blockers),
    )
