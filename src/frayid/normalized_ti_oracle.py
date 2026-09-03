from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CORRECTNESS_ID = "postv1_p5_normalized_ti_oracle_r01"
NORMALIZED_TOLERANCE = 1e-10
MAX_ITERATIONS = 10_000_000
CONSERVATIVE_RESCALING = 0.8


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class NormalizedTrajectory:
    start: np.ndarray
    end: np.ndarray
    center: np.ndarray
    scale: float

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "post_v1_p5_normalized_trajectory.v1",
            "center": self.center.tolist(),
            "scale": self.scale,
            "start_sha256": _array_sha256(self.start),
            "end_sha256": _array_sha256(self.end),
        }


@dataclass(frozen=True)
class NormalizedTICertificate:
    status: str
    collision_free: bool
    candidate_count: int
    normalized: NormalizedTrajectory
    normalized_minimum_distance: float
    elapsed_seconds: float
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("normalized")
        result.update(
            {
                "schema_version": "post_v1_p5_normalized_ti_certificate.v1",
                "correctness_id": CORRECTNESS_ID,
                "normalized_trajectory": self.normalized.report(),
            }
        )
        return result


def normalize_linear_trajectory(
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
) -> NormalizedTrajectory:
    start = np.asarray(vertices_t0, dtype=np.float64)
    end = np.asarray(vertices_t1, dtype=np.float64)
    if start.shape != end.shape or start.ndim != 2 or start.shape[1] not in (2, 3):
        raise ValueError("vertices_t0 and vertices_t1 must have equal shape (n, 2|3)")
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError("trajectory vertices must be finite")
    joint_minimum = np.minimum(start.min(axis=0), end.min(axis=0))
    joint_maximum = np.maximum(start.max(axis=0), end.max(axis=0))
    center = (joint_minimum + joint_maximum) / 2.0
    scale = float(np.max(joint_maximum - joint_minimum))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("joint trajectory AABB scale must be finite and positive")
    normalized_start = np.asfortranarray((start - center) / scale, dtype=np.float64)
    normalized_end = np.asfortranarray((end - center) / scale, dtype=np.float64)
    if not np.isfinite(normalized_start).all() or not np.isfinite(normalized_end).all():
        raise ValueError("normalized trajectory must be finite")
    return NormalizedTrajectory(
        start=normalized_start,
        end=normalized_end,
        center=np.asarray(center, dtype=np.float64),
        scale=scale,
    )


def normalized_ti_path_oracle(
    collision_mesh: Any,
    vertices_t0: np.ndarray,
    vertices_t1: np.ndarray,
    *,
    dmin: float = 0.0,
) -> NormalizedTICertificate:
    """Judge a linear path after deterministic joint-AABB normalization."""
    try:
        import ipctk  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("P5 requires the pinned collision extra: ipctk==1.6.0") from error

    started = time.monotonic()
    normalized = normalize_linear_trajectory(vertices_t0, vertices_t1)
    if normalized.start.shape[0] != int(collision_mesh.num_vertices):
        raise ValueError("trajectory vertex count does not match collision mesh")
    if not math.isfinite(dmin) or dmin < 0.0:
        raise ValueError("minimum distance must be finite and nonnegative")
    normalized_dmin = dmin / normalized.scale
    candidates = ipctk.Candidates()
    candidates.build(
        collision_mesh,
        normalized.start,
        normalized.end,
        normalized_dmin,
        ipctk.SweepAndPrune(),
    )
    collision_free = bool(
        candidates.is_step_collision_free(
            collision_mesh,
            normalized.start,
            normalized.end,
            normalized_dmin,
            ipctk.TightInclusionCCD(
                NORMALIZED_TOLERANCE,
                MAX_ITERATIONS,
                CONSERVATIVE_RESCALING,
            ),
        )
    )
    blockers: list[str] = []
    if not math.isfinite(normalized_dmin) or normalized_dmin < 0.0:
        blockers.append("invalid_normalized_minimum_distance")
    return NormalizedTICertificate(
        status="pass" if not blockers else "fail",
        collision_free=collision_free,
        candidate_count=len(candidates),
        normalized=normalized,
        normalized_minimum_distance=normalized_dmin,
        elapsed_seconds=time.monotonic() - started,
        blockers=tuple(blockers),
    )
