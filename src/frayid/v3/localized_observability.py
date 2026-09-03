from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability
from frayid.v3.controlled_target import ControlledMethodCaseContract
from frayid.v3.method_source_selection import (
    MethodSourceSelectionReport,
    _decode_candidate_span,
    _dynamic_mask,
    _nearest_indices,
)
from frayid.v3.nonuniform_correspondence import (
    NonuniformCycleCorrespondenceReport,
    ordered_dtw,
)

EXPERIMENT_ID: Literal["postv3_m04_localized_rotation_observability_audit_r01"] = (
    "postv3_m04_localized_rotation_observability_audit_r01"
)
_MINIMUM_ORDERED_VIEWS = 8
_MINIMUM_BOOTSTRAP_RETENTION = 0.80
_MINIMUM_CONTROL_MARGIN = 0.30
_BOOTSTRAP_COUNT = 16
_LOCAL_HEIGHT = 48
_LOCAL_WIDTH = 48


class StrictObservabilityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LocalizedSupportPair(StrictObservabilityModel):
    first_window_index: int = Field(ge=0)
    second_window_index: int = Field(ge=0)
    first_seconds: float = Field(ge=0.0)
    second_seconds: float = Field(ge=0.0)
    descriptor_distance: float = Field(ge=0.0)
    bootstrap_retention: float = Field(ge=0.0, le=1.0)
    role: Literal["observed_partial_phase_support", "diagnostic_pair_proposal"]


class LocalizedRotationObservabilityReport(StrictObservabilityModel):
    schema_version: Literal["frayid_v3_localized_rotation_observability.v1"] = (
        "frayid_v3_localized_rotation_observability.v1"
    )
    experiment_id: Literal["postv3_m04_localized_rotation_observability_audit_r01"] = EXPERIMENT_ID
    method_case_id: str
    method_case_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    m02_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    m03_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_pixel_role: Literal["measured"] = "measured"
    phase_input_role: Literal["rejected_proposal_for_diagnostic_only"] = (
        "rejected_proposal_for_diagnostic_only"
    )
    descriptor_role: Literal["proposal"] = "proposal"
    output_role: Literal["partial_support_ledger", "rejected_partial_support"]
    support_scope: Literal["observed_phase_only"] = "observed_phase_only"
    subject_bbox_xywh: tuple[int, int, int, int]
    descriptor_shape_chw: tuple[int, int, int]
    support_pairs: list[LocalizedSupportPair]
    mutually_ordered_view_count: int = Field(ge=0)
    bootstrap_retained_pair_fraction: float = Field(ge=0.0, le=1.0)
    treatment_alignment_cost: float = Field(ge=0.0)
    reversed_order_control_cost: float = Field(ge=0.0)
    half_cycle_control_cost: float = Field(ge=0.0)
    single_cycle_control_cost: float = Field(ge=0.0)
    spatial_shuffle_control_cost: float = Field(ge=0.0)
    minimum_control_margin: float
    public_controls_passed: Literal[True] = True
    decoded_source_frame_count: int = Field(ge=0)
    case_a_pixels_read: Literal[0] = 0
    cross_person_geometry_reads: Literal[0] = 0
    evaluator_files_read: Literal[0] = 0
    sealed_test_accesses: Literal[0] = 0
    promotion_eligible: Literal[False] = False
    permitted_use: Literal["acquisition_engineering_and_local_chart_fixtures_only"] = (
        "acquisition_engineering_and_local_chart_fixtures_only"
    )
    complete_rotation_claim: Literal[False] = False
    geometry_fitting_authorized: Literal[False] = False
    status: Literal["pass_partial_support_only", "fail_rejected_partial_support"]
    blockers: list[str]


class _LocalizedMetrics(TypedDict):
    treatment: float
    path: list[tuple[int, int]]
    costs: np.ndarray
    support: dict[tuple[int, int], float]
    accepted_pairs: list[tuple[int, int]]
    retained_fraction: float
    reversed_cost: float
    half_cost: float
    single_cost: float
    shuffled_cost: float
    margin: float
    blockers: list[str]


def _subject_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if len(rows) < 64:
        raise ValueError("localized descriptor requires a nonempty dynamic subject mask")
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    x_pad = max(2, (right - left) // 12)
    y_pad = max(2, (bottom - top) // 12)
    left = max(0, left - x_pad)
    right = min(mask.shape[1], right + x_pad)
    top = max(0, top - y_pad)
    bottom = min(mask.shape[0], bottom + y_pad)
    if right - left < 8 or bottom - top < 8:
        raise ValueError("localized subject support is too small")
    return left, top, right - left, bottom - top


def localized_descriptor_tensor(
    frames: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Create deterministic subject/upper-garment-localized proposal descriptors."""
    if frames.ndim != 3 or mask.shape != frames.shape[1:]:
        raise ValueError("localized descriptors require N,H,W frames and an H,W mask")
    left, top, width, height = _subject_bbox(mask)
    crops = np.stack(
        [
            cv2.resize(
                frame[top : top + height, left : left + width],
                (_LOCAL_WIDTH, _LOCAL_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            for frame in frames
        ]
    ).astype(np.float32)
    median = np.median(crops, axis=0)
    yy, xx = np.mgrid[0:_LOCAL_HEIGHT, 0:_LOCAL_WIDTH]
    torso_weight = np.exp(
        -0.5
        * (
            ((xx - 0.5 * (_LOCAL_WIDTH - 1)) / (0.32 * _LOCAL_WIDTH)) ** 2
            + ((yy - 0.38 * (_LOCAL_HEIGHT - 1)) / (0.32 * _LOCAL_HEIGHT)) ** 2
        )
    ).astype(np.float32)
    tensors: list[np.ndarray] = []
    for crop in crops:
        centered = (crop - float(crop.mean())) / max(float(crop.std()), 0.05)
        grad_x = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
        deviation = np.abs(crop - median)
        channels = np.stack(
            (
                centered,
                grad_x,
                grad_y,
                deviation,
                centered * torso_weight,
                deviation * torso_weight,
            )
        )
        scales = np.sqrt(np.mean(channels**2, axis=(1, 2), keepdims=True))
        tensors.append(channels / np.maximum(scales, 0.05))
    return np.asarray(tensors, dtype=np.float64), (left, top, width, height)


def _flatten(tensor: np.ndarray) -> np.ndarray:
    return np.asarray(tensor.reshape(len(tensor), -1), dtype=np.float64)


def _spatial_shuffle(tensor: np.ndarray) -> np.ndarray:
    shuffled = np.empty_like(tensor)
    for index, item in enumerate(tensor):
        row_shift = (7 * index + 5) % item.shape[1]
        column_shift = (11 * index + 3) % item.shape[2]
        shuffled[index] = np.roll(item, (row_shift, column_shift), axis=(1, 2))
    return shuffled


def _one_to_one_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    last_i = -1
    last_j = -1
    for i, j in path:
        if i > last_i and j > last_j:
            selected.append((i, j))
            last_i, last_j = i, j
    return selected


def _bootstrap_support(
    first: np.ndarray,
    second: np.ndarray,
    baseline_pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    rng = np.random.default_rng(20260903)
    counts = {pair: 0 for pair in baseline_pairs}
    feature_count = first.shape[1]
    keep_count = max(4, round(0.80 * feature_count))
    for _ in range(_BOOTSTRAP_COUNT):
        keep = np.sort(rng.choice(feature_count, size=keep_count, replace=False))
        _, path, _ = ordered_dtw(first[:, keep], second[:, keep])
        observed = set(_one_to_one_path(path))
        for pair in counts:
            if pair in observed:
                counts[pair] += 1
    return {pair: count / _BOOTSTRAP_COUNT for pair, count in counts.items()}


def qualify_localized_descriptors(
    first_tensor: np.ndarray,
    second_tensor: np.ndarray,
) -> _LocalizedMetrics:
    """Apply the M04 gates to two localized stable-view sequences."""
    if first_tensor.ndim < 2 or second_tensor.ndim != first_tensor.ndim:
        raise ValueError("localized sequences must have matching nonempty tensor ranks")
    first = _flatten(first_tensor)
    second = _flatten(second_tensor)
    if first.shape[1] != second.shape[1]:
        raise ValueError("localized descriptor dimensions differ")
    treatment, path, costs = ordered_dtw(first, second)
    baseline_pairs = _one_to_one_path(path)
    support = _bootstrap_support(first, second, baseline_pairs)
    accepted = [pair for pair in baseline_pairs if support[pair] >= _MINIMUM_BOOTSTRAP_RETENTION]
    retained_fraction = len(accepted) / max(len(baseline_pairs), 1)
    reversed_cost, _, _ = ordered_dtw(first, second[::-1])
    half_cost, _, _ = ordered_dtw(second, np.roll(second, len(second) // 2, axis=0))
    single_cost, _, _ = ordered_dtw(first, np.roll(first, len(first) // 2, axis=0))
    shuffled_tensor = _spatial_shuffle(second_tensor)
    shuffled_cost, _, _ = ordered_dtw(first, _flatten(shuffled_tensor))
    minimum_control = min(reversed_cost, half_cost, single_cost, shuffled_cost)
    margin = 1.0 - treatment / max(minimum_control, 1e-8)
    blockers: list[str] = []
    if len(accepted) < _MINIMUM_ORDERED_VIEWS:
        blockers.append("mutually_ordered_view_count_below_8")
    if retained_fraction < _MINIMUM_BOOTSTRAP_RETENTION:
        blockers.append("bootstrap_retained_pair_fraction_below_0_80")
    if margin < _MINIMUM_CONTROL_MARGIN:
        blockers.append("control_cost_margin_below_0_30")
    return {
        "treatment": treatment,
        "path": path,
        "costs": costs,
        "support": support,
        "accepted_pairs": accepted,
        "retained_fraction": retained_fraction,
        "reversed_cost": reversed_cost,
        "half_cost": half_cost,
        "single_cost": single_cost,
        "shuffled_cost": shuffled_cost,
        "margin": margin,
        "blockers": blockers,
    }


def public_localized_controls_pass() -> bool:
    """Freeze an analytic repeated-phase control before private pixels are read."""
    first_phase = np.linspace(0.0, 2.0 * np.pi, 12)
    second_phase = np.asarray(
        [0.0, 0.25, 0.65, 1.1, 1.6, 2.1, 2.7, 3.3, 3.9, 4.5, 5.0, 5.5, 5.9, 2.0 * np.pi]
    )

    def features(phases: np.ndarray) -> np.ndarray:
        harmonics = [
            function(order * phases) for order in range(1, 7) for function in (np.sin, np.cos)
        ]
        return np.stack(harmonics, axis=1).reshape(len(phases), 1, 3, 4)

    passing = qualify_localized_descriptors(features(first_phase), features(second_phase))
    reversed_result = qualify_localized_descriptors(
        features(first_phase), features(second_phase[::-1])
    )
    return (
        not passing["blockers"] and "control_cost_margin_below_0_30" in reversed_result["blockers"]
    )


def _read_source_fps(contract: ControlledMethodCaseContract) -> float:
    source_manifest = read_json(Path(contract.source_manifest_path))
    nominal_rate = source_manifest.get("video_probe", {}).get("nominal_frame_rate")
    if not isinstance(nominal_rate, str):
        raise ValueError("method-case manifest must retain the native nominal frame rate")
    try:
        source_fps = float(Fraction(nominal_rate))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("method-case nominal frame rate is invalid") from exc
    if not 1.0 <= source_fps <= 240.0:
        raise ValueError("method-case nominal frame rate is outside the supported range")
    return source_fps


def audit_localized_rotation_observability(
    *,
    method_case_contract_path: Path,
    m02_report_path: Path,
    m03_report_path: Path,
) -> LocalizedRotationObservabilityReport:
    """Audit an explicit partial phase ledger; never infer missing rotation."""
    reject_sealed_capability([method_case_contract_path, m02_report_path, m03_report_path])
    if not public_localized_controls_pass():
        raise RuntimeError("M04 public controls did not qualify before the private read")
    contract = ControlledMethodCaseContract.model_validate(read_json(method_case_contract_path))
    m02 = MethodSourceSelectionReport.model_validate(read_json(m02_report_path))
    m03 = NonuniformCycleCorrespondenceReport.model_validate(read_json(m03_report_path))
    if m02.source_video_sha256 != contract.source_video_sha256:
        raise ValueError("M02 and method-case source hashes differ")
    if m03.source_video_sha256 != contract.source_video_sha256:
        raise ValueError("M03 and method-case source hashes differ")
    if m03.status != "fail_rejected_phase_proposal":
        raise ValueError("M04 r01 is the registered diagnostic successor to terminal M03")
    if m03.blockers != ["control_cost_margin_below_0_30"]:
        raise ValueError("M04 r01 cannot reinterpret a different M03 failure")
    video_path = Path(contract.source_video_path)
    reject_sealed_capability([video_path])
    if sha256_file(video_path) != contract.source_video_sha256:
        raise ValueError("method-case source hash changed after registration")
    start = min(item.proposal_start_seconds for item in m02.interval_audits)
    end = max(item.proposal_end_seconds for item in m02.interval_audits)
    sequence = _decode_candidate_span(
        video_path,
        start_seconds=start,
        end_seconds=end,
        source_fps=_read_source_fps(contract),
    )
    mask = _dynamic_mask(sequence.frames)
    tensor, bbox = localized_descriptor_tensor(sequence.frames, mask)
    window_times = [
        [window.representative_seconds for window in audit.stable_windows]
        for audit in m02.interval_audits
    ]
    if len(window_times) != 2:
        raise ValueError("M04 r01 requires the two frozen M02 stable-window sequences")
    indices = [_nearest_indices(sequence.times, np.asarray(times)) for times in window_times]
    first_tensor = tensor[indices[0]]
    second_tensor = tensor[indices[1]]
    metrics = qualify_localized_descriptors(first_tensor, second_tensor)
    costs = metrics["costs"]
    support = metrics["support"]
    blockers = list(metrics["blockers"])
    passed = not blockers
    pairs = [
        LocalizedSupportPair(
            first_window_index=i,
            second_window_index=j,
            first_seconds=window_times[0][i],
            second_seconds=window_times[1][j],
            descriptor_distance=float(costs[i, j]),
            bootstrap_retention=float(support[(i, j)]),
            role="observed_partial_phase_support" if passed else "diagnostic_pair_proposal",
        )
        for i, j in metrics["accepted_pairs"]
    ]
    return LocalizedRotationObservabilityReport(
        method_case_id=contract.case_id,
        method_case_contract_sha256=sha256_file(method_case_contract_path),
        m02_report_sha256=sha256_file(m02_report_path),
        m03_report_sha256=sha256_file(m03_report_path),
        source_video_sha256=contract.source_video_sha256,
        output_role="partial_support_ledger" if passed else "rejected_partial_support",
        subject_bbox_xywh=bbox,
        descriptor_shape_chw=(
            int(first_tensor.shape[1]),
            int(first_tensor.shape[2]),
            int(first_tensor.shape[3]),
        ),
        support_pairs=pairs,
        mutually_ordered_view_count=len(pairs),
        bootstrap_retained_pair_fraction=float(metrics["retained_fraction"]),
        treatment_alignment_cost=float(metrics["treatment"]),
        reversed_order_control_cost=float(metrics["reversed_cost"]),
        half_cycle_control_cost=float(metrics["half_cost"]),
        single_cycle_control_cost=float(metrics["single_cost"]),
        spatial_shuffle_control_cost=float(metrics["shuffled_cost"]),
        minimum_control_margin=float(metrics["margin"]),
        decoded_source_frame_count=sequence.decoded_source_frame_count,
        status="pass_partial_support_only" if passed else "fail_rejected_partial_support",
        blockers=blockers,
    )


__all__ = [
    "LocalizedRotationObservabilityReport",
    "audit_localized_rotation_observability",
    "localized_descriptor_tensor",
    "public_localized_controls_pass",
    "qualify_localized_descriptors",
]
