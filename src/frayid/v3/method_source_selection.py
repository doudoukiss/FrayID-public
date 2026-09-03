from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability
from frayid.v3.controlled_target import ControlledMethodCaseContract

EXPERIMENT_ID: Literal["postv3_m02_deterministic_source_selection_audit_r01"] = (
    "postv3_m02_deterministic_source_selection_audit_r01"
)
_SAMPLE_RATE_HZ = 5.0
_DESCRIPTOR_WIDTH = 96
_DESCRIPTOR_HEIGHT = 54
_MINIMUM_STABLE_WINDOWS = 8
_MINIMUM_STABLE_SECONDS = 0.8
_MAXIMUM_CLOSURE_RATIO = 0.80
_MINIMUM_VIEW_DIVERSITY = 0.025
_MAXIMUM_REPEATABILITY_RATIO = 0.85


class StrictSelectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StableWindow(StrictSelectionModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    representative_seconds: float = Field(ge=0.0)
    median_motion: float = Field(ge=0.0)


class RotationIntervalAudit(StrictSelectionModel):
    proposal_start_seconds: float = Field(ge=0.0)
    proposal_end_seconds: float = Field(gt=0.0)
    sampled_frame_count: int = Field(ge=0)
    stable_windows: list[StableWindow]
    endpoint_closure_distance: float = Field(ge=0.0)
    midpoint_separation_distance: float = Field(ge=0.0)
    closure_ratio: float = Field(ge=0.0)
    view_diversity: float = Field(ge=0.0)
    status: Literal["pass", "fail"]
    blockers: list[str]


class MethodSourceSelectionReport(StrictSelectionModel):
    schema_version: Literal["frayid_v3_method_source_selection.v1"] = (
        "frayid_v3_method_source_selection.v1"
    )
    experiment_id: Literal["postv3_m02_deterministic_source_selection_audit_r01"] = EXPERIMENT_ID
    method_case_id: str
    source_video_path: str
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_case_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_pixel_role: Literal["measured"] = "measured"
    input_interval_role: Literal["proposal"] = "proposal"
    output_interval_role: Literal["audited_proposal", "rejected_proposal"]
    sample_rate_hz: float = _SAMPLE_RATE_HZ
    descriptor_resolution: tuple[int, int] = (
        _DESCRIPTOR_WIDTH,
        _DESCRIPTOR_HEIGHT,
    )
    dynamic_mask_fraction: float = Field(ge=0.0, le=1.0)
    stable_motion_threshold: float = Field(ge=0.0)
    interval_audits: list[RotationIntervalAudit]
    cross_cycle_same_phase_distance: float = Field(ge=0.0)
    cross_cycle_half_shift_distance: float = Field(ge=0.0)
    cross_cycle_repeatability_ratio: float = Field(ge=0.0)
    decoded_source_frame_count: int = Field(ge=0)
    case_a_pixels_read: Literal[0] = 0
    cross_person_geometry_reads: Literal[0] = 0
    evaluator_files_read: Literal[0] = 0
    sealed_test_accesses: Literal[0] = 0
    promotion_eligible: Literal[False] = False
    status: Literal["pass_audited_proposal", "fail_rejected_proposal"]
    blockers: list[str]


@dataclass(frozen=True)
class _SampledSequence:
    times: np.ndarray
    frames: np.ndarray
    decoded_source_frame_count: int


def _decode_candidate_span(
    video_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    sample_rate_hz: float = _SAMPLE_RATE_HZ,
    source_fps: float | None = None,
) -> _SampledSequence:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not decode method-case video: {video_path}")
    fps = float(source_fps) if source_fps is not None else float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise ValueError("method-case video must report a positive frame rate")
    stride = max(1, round(fps / sample_rate_hz))
    start_frame = max(0, int(np.floor(start_seconds * fps)))
    end_frame = int(np.ceil(end_seconds * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    frames: list[np.ndarray] = []
    times: list[float] = []
    decoded = 0
    try:
        frame_index = start_frame
        while frame_index <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if (frame_index - start_frame) % stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(
                    gray,
                    (_DESCRIPTOR_WIDTH, _DESCRIPTOR_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(small.astype(np.float32) / 255.0)
                times.append(frame_index / fps)
            frame_index += 1
    finally:
        capture.release()
    if len(frames) < 3:
        raise ValueError("method-case candidate span produced too few sampled frames")
    return _SampledSequence(
        times=np.asarray(times, dtype=np.float64),
        frames=np.stack(frames),
        decoded_source_frame_count=decoded,
    )


def _dynamic_mask(frames: np.ndarray) -> np.ndarray:
    variation = np.std(frames, axis=0)
    positive = variation[variation > 0.005]
    if positive.size < 64:
        raise ValueError("candidate span has insufficient dynamic image support")
    threshold = float(np.quantile(positive, 0.65))
    mask = variation >= threshold
    if int(mask.sum()) < 64:
        raise ValueError("dynamic support mask is too small")
    return np.asarray(mask, dtype=np.bool_)


def _descriptors(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = frames[:, mask].astype(np.float64)
    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    return np.asarray((values - means) / np.maximum(scales, 0.05), dtype=np.float64)


def _motion_scores(descriptors: np.ndarray) -> np.ndarray:
    differences = np.mean(np.abs(np.diff(descriptors, axis=0)), axis=1)
    return np.asarray(np.concatenate(([differences[0]], differences)), dtype=np.float64)


def _stable_threshold(motion: np.ndarray) -> float:
    return min(0.35, float(np.quantile(motion, 0.40) * 1.8 + 0.01))


def _stable_windows(
    times: np.ndarray,
    motion: np.ndarray,
    *,
    threshold: float,
) -> list[StableWindow]:
    stable = motion <= threshold
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, accepted in enumerate(stable):
        if accepted and start is None:
            start = index
        if start is not None and (not accepted or index == len(stable) - 1):
            end = index if accepted and index == len(stable) - 1 else index - 1
            duration = times[end] - times[start] + 1.0 / _SAMPLE_RATE_HZ
            if duration >= _MINIMUM_STABLE_SECONDS - 1e-9:
                groups.append((start, end))
            start = None
    windows = []
    for left, right in groups:
        representative = (left + right) // 2
        windows.append(
            StableWindow(
                start_seconds=float(times[left]),
                end_seconds=float(times[right]),
                representative_seconds=float(times[representative]),
                median_motion=float(np.median(motion[left : right + 1])),
            )
        )
    return windows


def _normalized_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def _nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(times - target))) for target in targets], dtype=int)


def _audit_interval(
    sequence: _SampledSequence,
    descriptors: np.ndarray,
    motion: np.ndarray,
    *,
    start_seconds: float,
    end_seconds: float,
    stable_threshold: float,
) -> RotationIntervalAudit:
    indices = np.flatnonzero((sequence.times >= start_seconds) & (sequence.times <= end_seconds))
    times = sequence.times[indices]
    local_motion = motion[indices]
    windows = _stable_windows(times, local_motion, threshold=stable_threshold)
    fractions = np.linspace(0.0, 1.0, 9)
    target_times = start_seconds + fractions * (end_seconds - start_seconds)
    phase_indices = _nearest_indices(sequence.times, target_times)
    phase_descriptors = descriptors[phase_indices]
    closure = _normalized_distance(phase_descriptors[0], phase_descriptors[-1])
    midpoint = _normalized_distance(phase_descriptors[0], phase_descriptors[4])
    closure_ratio = closure / max(midpoint, 1e-8)
    pair_distances = [
        _normalized_distance(left, right) for left, right in pairwise(phase_descriptors)
    ]
    diversity = float(np.median(pair_distances))
    blockers: list[str] = []
    if len(windows) < _MINIMUM_STABLE_WINDOWS:
        blockers.append("stable_window_count_below_8")
    if closure_ratio > _MAXIMUM_CLOSURE_RATIO:
        blockers.append("endpoint_closure_ratio_above_0_80")
    if diversity < _MINIMUM_VIEW_DIVERSITY:
        blockers.append("view_diversity_below_0_025")
    return RotationIntervalAudit(
        proposal_start_seconds=start_seconds,
        proposal_end_seconds=end_seconds,
        sampled_frame_count=len(indices),
        stable_windows=windows,
        endpoint_closure_distance=closure,
        midpoint_separation_distance=midpoint,
        closure_ratio=closure_ratio,
        view_diversity=diversity,
        status="pass" if not blockers else "fail",
        blockers=blockers,
    )


def audit_method_case_source_selection(
    *,
    method_case_contract_path: Path,
) -> MethodSourceSelectionReport:
    """Audit proposed cycles deterministically without reading another person's case."""
    reject_sealed_capability([method_case_contract_path])
    contract = ControlledMethodCaseContract.model_validate(read_json(method_case_contract_path))
    video_path = Path(contract.source_video_path)
    reject_sealed_capability([video_path])
    if sha256_file(video_path) != contract.source_video_sha256:
        raise ValueError("method-case source hash changed after registration")
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
    start = min(interval.start_seconds for interval in contract.candidate_intervals)
    end = max(interval.end_seconds for interval in contract.candidate_intervals)
    sequence = _decode_candidate_span(
        video_path,
        start_seconds=start,
        end_seconds=end,
        source_fps=source_fps,
    )
    mask = _dynamic_mask(sequence.frames)
    descriptors = _descriptors(sequence.frames, mask)
    motion = _motion_scores(descriptors)
    stable_threshold = _stable_threshold(motion)
    audits = [
        _audit_interval(
            sequence,
            descriptors,
            motion,
            start_seconds=interval.start_seconds,
            end_seconds=interval.end_seconds,
            stable_threshold=stable_threshold,
        )
        for interval in contract.candidate_intervals
    ]
    if len(contract.candidate_intervals) != 2:
        raise ValueError("M02 r01 requires exactly two proposed cycles")
    phase_count = 9
    first, second = contract.candidate_intervals
    fractions = np.linspace(0.0, 1.0, phase_count)
    first_indices = _nearest_indices(
        sequence.times,
        first.start_seconds + fractions * (first.end_seconds - first.start_seconds),
    )
    second_indices = _nearest_indices(
        sequence.times,
        second.start_seconds + fractions * (second.end_seconds - second.start_seconds),
    )
    first_descriptors = descriptors[first_indices]
    second_descriptors = descriptors[second_indices]
    same_phase = float(np.median(np.mean(np.abs(first_descriptors - second_descriptors), axis=1)))
    half_shifted = np.roll(second_descriptors, phase_count // 2, axis=0)
    half_shift = float(np.median(np.mean(np.abs(first_descriptors - half_shifted), axis=1)))
    repeatability_ratio = same_phase / max(half_shift, 1e-8)
    blockers = [
        f"interval_{index:02d}:{blocker}"
        for index, audit in enumerate(audits)
        for blocker in audit.blockers
    ]
    if repeatability_ratio > _MAXIMUM_REPEATABILITY_RATIO:
        blockers.append("cross_cycle_repeatability_ratio_above_0_85")
    status: Literal["pass_audited_proposal", "fail_rejected_proposal"] = (
        "pass_audited_proposal" if not blockers else "fail_rejected_proposal"
    )
    return MethodSourceSelectionReport(
        method_case_id=contract.case_id,
        source_video_path=str(video_path),
        source_video_sha256=contract.source_video_sha256,
        method_case_contract_sha256=sha256_file(method_case_contract_path),
        output_interval_role="audited_proposal" if not blockers else "rejected_proposal",
        dynamic_mask_fraction=float(mask.mean()),
        stable_motion_threshold=stable_threshold,
        interval_audits=audits,
        cross_cycle_same_phase_distance=same_phase,
        cross_cycle_half_shift_distance=half_shift,
        cross_cycle_repeatability_ratio=repeatability_ratio,
        decoded_source_frame_count=sequence.decoded_source_frame_count,
        status=status,
        blockers=blockers,
    )


def synthetic_cycle_features(
    *,
    repeat_second_cycle: bool,
    include_holds: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Public analytic control for closure, diversity, and repeated-cycle tests."""
    phases = np.linspace(0.0, 2.0 * np.pi, 61)
    first = np.stack((np.sin(phases), np.cos(phases), np.sin(2.0 * phases)), axis=1)
    second = first.copy() if repeat_second_cycle else np.roll(first, 20, axis=0)
    descriptors = np.concatenate((first, second), axis=0)
    times = np.arange(len(descriptors), dtype=np.float64) / _SAMPLE_RATE_HZ
    if include_holds:
        motion = np.tile(np.asarray([0.02, 0.02, 0.02, 0.02, 0.30, 0.30]), 21)[: len(times)]
    else:
        motion = np.full(len(times), 0.30)
    return times, descriptors, motion


__all__ = [
    "MethodSourceSelectionReport",
    "RotationIntervalAudit",
    "StableWindow",
    "audit_method_case_source_selection",
    "synthetic_cycle_features",
]
