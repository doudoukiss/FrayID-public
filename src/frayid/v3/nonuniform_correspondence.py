from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability
from frayid.v3.controlled_target import ControlledMethodCaseContract
from frayid.v3.method_source_selection import (
    MethodSourceSelectionReport,
    _decode_candidate_span,
    _descriptors,
    _dynamic_mask,
    _nearest_indices,
)

EXPERIMENT_ID: Literal["postv3_m03_ordered_nonuniform_cycle_correspondence_r01"] = (
    "postv3_m03_ordered_nonuniform_cycle_correspondence_r01"
)
_MINIMUM_MATCHED_VIEWS = 10
_MINIMUM_PATH_COVERAGE = 0.80
_MINIMUM_CONTROL_MARGIN = 0.30
_MAXIMUM_REPEATED_STATE_RATIO = 0.50


class StrictCorrespondenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderedMatch(StrictCorrespondenceModel):
    first_window_index: int = Field(ge=0)
    second_window_index: int = Field(ge=0)
    first_seconds: float = Field(ge=0.0)
    second_seconds: float = Field(ge=0.0)
    descriptor_distance: float = Field(ge=0.0)


class NonuniformCycleCorrespondenceReport(StrictCorrespondenceModel):
    schema_version: Literal["frayid_v3_nonuniform_cycle_correspondence.v1"] = (
        "frayid_v3_nonuniform_cycle_correspondence.v1"
    )
    experiment_id: Literal["postv3_m03_ordered_nonuniform_cycle_correspondence_r01"] = EXPERIMENT_ID
    method_case_id: str
    method_case_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    m02_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_window_role: Literal["m02_frozen_low_motion_proposal"] = "m02_frozen_low_motion_proposal"
    ordered_matches: list[OrderedMatch]
    matched_view_count: int = Field(ge=0)
    first_cycle_path_coverage: float = Field(ge=0.0, le=1.0)
    second_cycle_path_coverage: float = Field(ge=0.0, le=1.0)
    treatment_alignment_cost: float = Field(ge=0.0)
    reversed_order_control_cost: float = Field(ge=0.0)
    half_cycle_control_cost: float = Field(ge=0.0)
    single_cycle_control_cost: float = Field(ge=0.0)
    minimum_control_margin: float
    repeated_state_distance_ratio: float = Field(ge=0.0)
    monotonic_order: Literal[True] = True
    case_a_pixels_read: Literal[0] = 0
    cross_person_geometry_reads: Literal[0] = 0
    evaluator_files_read: Literal[0] = 0
    sealed_test_accesses: Literal[0] = 0
    promotion_eligible: Literal[False] = False
    output_role: Literal["audited_phase_proposal", "rejected_phase_proposal"]
    status: Literal["pass_audited_phase_proposal", "fail_rejected_phase_proposal"]
    blockers: list[str]


class _QualificationMetrics(TypedDict):
    treatment: float
    path: list[tuple[int, int]]
    costs: np.ndarray
    reversed_cost: float
    half_cost: float
    single_cost: float
    margin: float
    first_coverage: float
    second_coverage: float
    matched_views: int
    repeated_ratio: float
    blockers: list[str]


def _pairwise_cost(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.mean(np.abs(first[:, None, :] - second[None, :, :]), axis=2),
        dtype=np.float64,
    )


def ordered_dtw(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, list[tuple[int, int]], np.ndarray]:
    """Return deterministic monotonic DTW cost and path."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("ordered DTW requires two nonempty descriptor matrices")
    if len(first) == 0 or len(second) == 0:
        raise ValueError("ordered DTW cannot align an empty sequence")
    costs = _pairwise_cost(first, second)
    cumulative = np.full((len(first), len(second)), np.inf, dtype=np.float64)
    predecessor = np.full((len(first), len(second), 2), -1, dtype=np.int32)
    cumulative[0, 0] = costs[0, 0]
    for i in range(len(first)):
        for j in range(len(second)):
            if i == 0 and j == 0:
                continue
            candidates: list[tuple[float, int, int]] = []
            if i > 0 and j > 0:
                candidates.append((float(cumulative[i - 1, j - 1]), i - 1, j - 1))
            if i > 0:
                candidates.append((float(cumulative[i - 1, j]), i - 1, j))
            if j > 0:
                candidates.append((float(cumulative[i, j - 1]), i, j - 1))
            best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
            cumulative[i, j] = costs[i, j] + best[0]
            predecessor[i, j] = (best[1], best[2])
    path = [(len(first) - 1, len(second) - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        previous = predecessor[i, j]
        path.append((int(previous[0]), int(previous[1])))
    path.reverse()
    normalized = float(cumulative[-1, -1] / len(path))
    return normalized, path, costs


def _control_metrics(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    reversed_cost, _, _ = ordered_dtw(first, second[::-1])
    half_cost, _, _ = ordered_dtw(second, np.roll(second, len(second) // 2, axis=0))
    single_cost, _, _ = ordered_dtw(first, np.roll(first, len(first) // 2, axis=0))
    return reversed_cost, half_cost, single_cost


def qualify_ordered_correspondence(
    first: np.ndarray,
    second: np.ndarray,
) -> _QualificationMetrics:
    """Apply frozen M03 gates to two descriptor sequences."""
    treatment, path, costs = ordered_dtw(first, second)
    reversed_cost, half_cost, single_cost = _control_metrics(first, second)
    minimum_control = min(reversed_cost, half_cost, single_cost)
    margin = 1.0 - treatment / max(minimum_control, 1e-8)
    first_coverage = len({i for i, _ in path}) / len(first)
    second_coverage = len({j for _, j in path}) / len(second)
    matched_views = min(len({i for i, _ in path}), len({j for _, j in path}))
    cross_distances = _pairwise_cost(first, second)
    repeated_ratio = float(np.min(cross_distances) / max(np.median(cross_distances), 1e-8))
    blockers: list[str] = []
    if matched_views < _MINIMUM_MATCHED_VIEWS:
        blockers.append("matched_view_count_below_10")
    if min(first_coverage, second_coverage) < _MINIMUM_PATH_COVERAGE:
        blockers.append("bidirectional_path_coverage_below_0_80")
    if margin < _MINIMUM_CONTROL_MARGIN:
        blockers.append("control_cost_margin_below_0_30")
    if repeated_ratio > _MAXIMUM_REPEATED_STATE_RATIO:
        blockers.append("repeated_state_distance_ratio_above_0_50")
    return {
        "treatment": treatment,
        "path": path,
        "costs": costs,
        "reversed_cost": reversed_cost,
        "half_cost": half_cost,
        "single_cost": single_cost,
        "margin": margin,
        "first_coverage": first_coverage,
        "second_coverage": second_coverage,
        "matched_views": matched_views,
        "repeated_ratio": repeated_ratio,
        "blockers": blockers,
    }


def audit_nonuniform_cycle_correspondence(
    *,
    method_case_contract_path: Path,
    m02_report_path: Path,
) -> NonuniformCycleCorrespondenceReport:
    """Align M02-frozen stable windows without assuming constant rotation speed."""
    reject_sealed_capability([method_case_contract_path, m02_report_path])
    contract = ControlledMethodCaseContract.model_validate(read_json(method_case_contract_path))
    m02 = MethodSourceSelectionReport.model_validate(read_json(m02_report_path))
    if m02.source_video_sha256 != contract.source_video_sha256:
        raise ValueError("M02 and method-case source hashes differ")
    if len(m02.interval_audits) != 2:
        raise ValueError("M03 r01 requires two M02 interval audits")
    video_path = Path(contract.source_video_path)
    sequence = _decode_candidate_span(
        video_path,
        start_seconds=min(item.proposal_start_seconds for item in m02.interval_audits),
        end_seconds=max(item.proposal_end_seconds for item in m02.interval_audits),
        source_fps=30000.0 / 1001.0,
    )
    mask = _dynamic_mask(sequence.frames)
    descriptors = _descriptors(sequence.frames, mask)
    window_times = [
        [window.representative_seconds for window in audit.stable_windows]
        for audit in m02.interval_audits
    ]
    indices = [_nearest_indices(sequence.times, np.asarray(times)) for times in window_times]
    first = descriptors[indices[0]]
    second = descriptors[indices[1]]
    metrics = qualify_ordered_correspondence(first, second)
    path = metrics["path"]
    costs = metrics["costs"]
    matches = [
        OrderedMatch(
            first_window_index=i,
            second_window_index=j,
            first_seconds=window_times[0][i],
            second_seconds=window_times[1][j],
            descriptor_distance=float(costs[i, j]),
        )
        for i, j in path
    ]
    blockers = list(metrics["blockers"])
    passed = not blockers
    return NonuniformCycleCorrespondenceReport(
        method_case_id=contract.case_id,
        method_case_contract_sha256=sha256_file(method_case_contract_path),
        m02_report_sha256=sha256_file(m02_report_path),
        source_video_sha256=contract.source_video_sha256,
        ordered_matches=matches,
        matched_view_count=int(metrics["matched_views"]),
        first_cycle_path_coverage=float(metrics["first_coverage"]),
        second_cycle_path_coverage=float(metrics["second_coverage"]),
        treatment_alignment_cost=float(metrics["treatment"]),
        reversed_order_control_cost=float(metrics["reversed_cost"]),
        half_cycle_control_cost=float(metrics["half_cost"]),
        single_cycle_control_cost=float(metrics["single_cost"]),
        minimum_control_margin=float(metrics["margin"]),
        repeated_state_distance_ratio=float(metrics["repeated_ratio"]),
        output_role="audited_phase_proposal" if passed else "rejected_phase_proposal",
        status="pass_audited_phase_proposal" if passed else "fail_rejected_phase_proposal",
        blockers=blockers,
    )


__all__ = [
    "NonuniformCycleCorrespondenceReport",
    "OrderedMatch",
    "audit_nonuniform_cycle_correspondence",
    "ordered_dtw",
    "qualify_ordered_correspondence",
]
