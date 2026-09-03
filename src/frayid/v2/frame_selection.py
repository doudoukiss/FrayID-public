from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from frayid.io import write_json
from frayid.v2.contracts import reject_sealed_capability


def select_phase_uniform_frames(
    evidence_master_path: Path,
    phase_solution_path: Path,
    output_path: Path,
    *,
    count: int,
    eligible_source_indices: set[int] | None = None,
    minimum_confidence: float = 0.0,
) -> Path:
    """Select by T05 yaw; never silently substitute timestamps for missing phase."""
    reject_sealed_capability([evidence_master_path, phase_solution_path, output_path])
    if output_path.exists():
        raise FileExistsError(f"phase selection output is immutable: {output_path}")
    if count < 4:
        raise ValueError("phase-uniform selection requires at least four frames")
    evidence = _read_object(evidence_master_path)
    phase = _read_object(phase_solution_path)
    phase_rows = phase.get("frames")
    if not isinstance(phase_rows, list):
        raise ValueError("T05 phase solution must contain a frames list")
    evidence_rows = evidence.get("frames")
    if not isinstance(evidence_rows, list):
        raise ValueError("evidence master must contain a frames list")
    evidence_indices = {
        int(row["source_frame_index"])
        for row in evidence_rows
        if isinstance(row, dict) and "source_frame_index" in row
    }
    candidates: list[tuple[int, float, float]] = []
    for row in phase_rows:
        if not isinstance(row, dict):
            continue
        if "source_frame_index" not in row or "yaw_radians" not in row:
            continue
        source_index = int(row["source_frame_index"])
        confidence_value = row.get("confidence")
        if confidence_value is None:
            confidence_value = row.get("yaw_confidence", 1.0)
        confidence = float(confidence_value)
        yaw = float(row["yaw_radians"])
        if not math.isfinite(yaw) or not math.isfinite(confidence):
            continue
        if source_index not in evidence_indices or confidence < minimum_confidence:
            continue
        if eligible_source_indices is not None and source_index not in eligible_source_indices:
            continue
        candidates.append((source_index, yaw, confidence))
    if len(candidates) < count:
        raise ValueError("insufficient eligible T05 phase observations")
    candidates.sort(key=lambda item: item[1])
    yaws = np.asarray([item[1] for item in candidates], dtype=np.float64)
    if np.any(np.diff(yaws) < 0.0):
        raise ValueError("T05 yaw must be monotonic after sorting")
    coverage = float(yaws[-1] - yaws[0])
    if coverage < math.radians(300.0):
        raise ValueError("T05 phase coverage is below 300 degrees")
    target_yaws = np.linspace(yaws[0], yaws[-1], count, endpoint=False, dtype=np.float64)
    selected: list[dict[str, float | int]] = []
    previous_slot = -1
    for target_index, target in enumerate(target_yaws):
        remaining_targets = count - target_index - 1
        candidate_slots = range(previous_slot + 1, len(candidates) - remaining_targets)
        slot = min(
            candidate_slots,
            key=lambda index: (abs(candidates[index][1] - target), index),
        )
        previous_slot = slot
        source_index, yaw, confidence = candidates[slot]
        selected.append(
            {
                "source_frame_index": source_index,
                "yaw_radians": yaw,
                "confidence": confidence,
                "target_yaw_radians": float(target),
                "absolute_phase_error_radians": abs(yaw - float(target)),
            }
        )
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_phase_uniform_selection.v1",
            "evidence_master_path": str(evidence_master_path),
            "phase_solution_path": str(phase_solution_path),
            "selection_policy": "t05_monotonic_yaw_uniform_no_time_fallback",
            "eligible_index_restriction_applied": eligible_source_indices is not None,
            "minimum_confidence": minimum_confidence,
            "requested_count": count,
            "selected_count": len(selected),
            "phase_coverage_radians": coverage,
            "frames": selected,
        },
    )


def eligible_indices_from_dataset_manifest(path: Path, *, split: str = "train") -> set[int]:
    payload = _read_object(path)
    rows = payload.get("frames")
    if not isinstance(rows, list):
        raise ValueError("dataset manifest must contain a frames list")
    return {
        int(row["source_frame_index"])
        for row in rows
        if isinstance(row, dict) and row.get("split") == split and "source_frame_index" in row
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload
