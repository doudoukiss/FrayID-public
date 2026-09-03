from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CORRECTNESS_ID = "postv1_p4_planar_dat_path_certificate_r01"
RELAXED_RADIUS_SCALING = 0.9
UPDATE_THRESHOLD = 0.01
TIGHT_INCLUSION_TOLERANCE = 1e-6
TIGHT_INCLUSION_MAX_ITERATIONS = 10_000_000
TIGHT_INCLUSION_CONSERVATIVE_RESCALING = 0.8


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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


@dataclass(frozen=True)
class PlanarDATCertificate:
    status: str
    accepted_vertices: np.ndarray
    filtered_displacements: np.ndarray
    trust_region_centers: np.ndarray
    trust_region_radii: np.ndarray
    candidate_keys: tuple[str, ...]
    candidate_count: int
    restricted_vertex_count: int
    proposed_displacement_norm: float
    filtered_displacement_norm: float
    retained_displacement_ratio: float
    minimum_trust_region_radius: float
    maximum_trust_region_radius: float
    full_swept_candidate_count: int
    full_oracle_safe: bool | None
    mechanism_elapsed_seconds: float
    oracle_elapsed_seconds: float
    elapsed_seconds: float
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        result = asdict(self)
        for field in (
            "accepted_vertices",
            "filtered_displacements",
            "trust_region_centers",
            "trust_region_radii",
            "candidate_keys",
        ):
            result.pop(field)
        candidate_bytes = json.dumps(self.candidate_keys, separators=(",", ":")).encode("utf-8")
        result.update(
            {
                "schema_version": "post_v1_p4_planar_dat_certificate.v1",
                "correctness_id": CORRECTNESS_ID,
                "candidate_keys_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                "accepted_vertices_sha256": _array_sha256(self.accepted_vertices),
                "filtered_displacements_sha256": _array_sha256(self.filtered_displacements),
                "trust_region_centers_sha256": _array_sha256(self.trust_region_centers),
                "trust_region_radii_sha256": _array_sha256(self.trust_region_radii),
            }
        )
        return result


def planar_dat_path_certificate(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dhat: float,
    dmin: float = 0.0,
    verify_full_path: bool = True,
) -> PlanarDATCertificate:
    """Filter one linear path with pinned Planar-DAT and independently judge it."""
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("P4 requires the pinned collision extra: ipctk==1.6.0") from error

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
        raise ValueError("P4 is registered only for dmin == 0")

    mechanism_started = time.monotonic()
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
    keys = _candidate_keys(trust_region.candidates)
    displacement = np.asfortranarray(proposal - start, dtype=np.float64)
    filtered = displacement.copy(order="F")
    trust_region.planar_filter_step(collision_mesh, start, filtered)
    accepted = np.asfortranarray(start + filtered, dtype=np.float64)
    mechanism_elapsed = time.monotonic() - mechanism_started

    proposed_norm = float(np.linalg.norm(displacement))
    filtered_norm = float(np.linalg.norm(filtered))
    retained_ratio = 1.0 if proposed_norm == 0.0 else filtered_norm / proposed_norm
    per_vertex_changed = np.any(filtered != displacement, axis=1)
    finite_radii = radii[np.isfinite(radii)]
    minimum_radius = float(np.min(finite_radii, initial=math.inf))
    maximum_radius = float(np.max(finite_radii, initial=0.0))

    full_swept_count = 0
    full_oracle_safe: bool | None = None
    oracle_elapsed = 0.0
    if verify_full_path:
        oracle_started = time.monotonic()
        broad_phase = ipctk.SweepAndPrune()
        full_swept = ipctk.Candidates()
        full_swept.build(collision_mesh, start, accepted, 0.0, broad_phase)
        full_swept_count = len(full_swept)
        full_oracle_safe = bool(
            full_swept.is_step_collision_free(
                collision_mesh,
                start,
                accepted,
                dmin,
                ipctk.TightInclusionCCD(
                    TIGHT_INCLUSION_TOLERANCE,
                    TIGHT_INCLUSION_MAX_ITERATIONS,
                    TIGHT_INCLUSION_CONSERVATIVE_RESCALING,
                ),
            )
        )
        oracle_elapsed = time.monotonic() - oracle_started

    blockers: list[str] = []
    if len(keys) != len(trust_region.candidates):
        blockers.append("candidate_identity_count_mismatch")
    if not np.isfinite(filtered).all() or not np.isfinite(accepted).all():
        blockers.append("nonfinite_filtered_path")
    if np.isnan(radii).any() or np.any(radii < 0.0):
        blockers.append("invalid_trust_region_radius")
    if not math.isfinite(retained_ratio) or retained_ratio < 0.0:
        blockers.append("invalid_retained_displacement_ratio")
    if verify_full_path and full_oracle_safe is not True:
        blockers.append("complete_dynamic_tight_inclusion_oracle")
    return PlanarDATCertificate(
        status="pass" if not blockers else "fail",
        accepted_vertices=accepted,
        filtered_displacements=filtered,
        trust_region_centers=centers,
        trust_region_radii=radii,
        candidate_keys=keys,
        candidate_count=len(keys),
        restricted_vertex_count=int(np.count_nonzero(per_vertex_changed)),
        proposed_displacement_norm=proposed_norm,
        filtered_displacement_norm=filtered_norm,
        retained_displacement_ratio=retained_ratio,
        minimum_trust_region_radius=minimum_radius,
        maximum_trust_region_radius=maximum_radius,
        full_swept_candidate_count=full_swept_count,
        full_oracle_safe=full_oracle_safe,
        mechanism_elapsed_seconds=mechanism_elapsed,
        oracle_elapsed_seconds=oracle_elapsed,
        elapsed_seconds=time.monotonic() - started,
        blockers=tuple(blockers),
    )
