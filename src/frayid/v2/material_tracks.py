from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.correspondence import _bind_training_frames
from frayid.v2.evidence import SAPIENS2_DOME29_LAYER_IDS

SEMANTIC_NAMES = tuple(sorted(SAPIENS2_DOME29_LAYER_IDS))
SEMANTIC_CODEBOOK = {name: index for index, name in enumerate(SEMANTIC_NAMES)}


@dataclass(frozen=True)
class MaterialTrackGate:
    """Frozen Q02a/Q02b gates chosen before the real semantic/photometric scan."""

    minimum_observations: int = 8
    minimum_semantic_stability: float = 0.80
    minimum_median_semantic_confidence: float = 0.50
    minimum_endpoint_patch_ncc: float = 0.35
    minimum_30_degree_tracks: int = 100
    minimum_90_degree_tracks: int = 20
    minimum_supported_semantic_layers: int = 3
    reverse_audit_maximum_tracks: int = 128
    reverse_return_error_p95_pixels: float = 3.0
    reverse_pass_fraction: float = 0.80
    photometric_minimum_90_degree_tracks: int = 20
    photometric_median_harmonic_improvement: float = 0.10
    photometric_positive_track_fraction: float = 0.60
    photometric_shuffled_margin: float = 0.05

    def as_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


@dataclass
class _RawTrack:
    frame_slots: list[int]
    points: list[np.ndarray]
    local_fb_errors: list[float]


@dataclass
class _QualifiedTrack:
    raw: _RawTrack
    semantic_code: int
    semantic_stability: float
    median_semantic_confidence: float
    endpoint_patch_ncc: float
    semantic_confidences: list[float]
    normalized_luminance: list[float]


@dataclass(frozen=True)
class VisibilityBoundedMaterialTracks:
    track_offsets: np.ndarray
    frame_ordinals: np.ndarray
    source_frame_indices: np.ndarray
    pixels: np.ndarray
    semantic_codes: np.ndarray
    semantic_confidence: np.ndarray
    normalized_luminance: np.ndarray
    local_forward_backward_error: np.ndarray
    track_span_degrees: np.ndarray
    track_weights: np.ndarray

    @property
    def track_count(self) -> int:
        return int(len(self.track_offsets) - 1)

    @property
    def observation_count(self) -> int:
        return len(self.frame_ordinals)

    def validate(self) -> None:
        if self.track_offsets.shape != (self.track_count + 1,):
            raise ValueError("Q02 track offsets have invalid shape")
        if self.track_offsets.dtype != np.int64:
            raise ValueError("Q02 track offsets must use int64")
        if int(self.track_offsets[0]) != 0 or int(self.track_offsets[-1]) != self.observation_count:
            raise ValueError("Q02 track offsets do not span observations")
        if np.any(self.track_offsets[1:] < self.track_offsets[:-1]):
            raise ValueError("Q02 track offsets must be monotonic")
        observation_vectors = (
            self.frame_ordinals,
            self.source_frame_indices,
            self.semantic_codes,
            self.semantic_confidence,
            self.normalized_luminance,
            self.local_forward_backward_error,
        )
        if any(value.shape != (self.observation_count,) for value in observation_vectors):
            raise ValueError("Q02 observation vectors must align")
        if self.pixels.shape != (self.observation_count, 2):
            raise ValueError("Q02 pixels must have shape [observation_count,2]")
        if self.track_span_degrees.shape != (self.track_count,) or self.track_weights.shape != (
            self.track_count,
        ):
            raise ValueError("Q02 track vectors must align")
        floating = (
            self.pixels,
            self.semantic_confidence,
            self.normalized_luminance,
            self.local_forward_backward_error,
            self.track_span_degrees,
            self.track_weights,
        )
        if any(not np.isfinite(value).all() for value in floating):
            raise ValueError("Q02 binding contains non-finite values")
        if np.any((self.semantic_confidence < 0) | (self.semantic_confidence > 1)):
            raise ValueError("Q02 semantic confidence must lie in [0,1]")
        if np.any((self.semantic_codes < 0) | (self.semantic_codes >= len(SEMANTIC_NAMES))):
            raise ValueError("Q02 semantic code is not registered")
        for start, stop in zip(self.track_offsets[:-1], self.track_offsets[1:], strict=True):
            if stop <= start or np.any(np.diff(self.frame_ordinals[start:stop]) <= 0):
                raise ValueError("Q02 observations must increase within every track")


def load_visibility_bounded_material_tracks(path: Path) -> VisibilityBoundedMaterialTracks:
    reject_sealed_capability([path])
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"]) != "frayid_v2_visibility_material_tracks.v1":
            raise ValueError("unsupported Q02 material-track schema")
        result = VisibilityBoundedMaterialTracks(
            track_offsets=archive["track_offsets"].astype(np.int64),
            frame_ordinals=archive["frame_ordinals"].astype(np.int64),
            source_frame_indices=archive["source_frame_indices"].astype(np.int64),
            pixels=archive["pixels"].astype(np.float32),
            semantic_codes=archive["semantic_codes"].astype(np.int16),
            semantic_confidence=archive["semantic_confidence"].astype(np.float32),
            normalized_luminance=archive["normalized_luminance"].astype(np.float32),
            local_forward_backward_error=archive["local_forward_backward_error"].astype(np.float32),
            track_span_degrees=archive["track_span_degrees"].astype(np.float32),
            track_weights=archive["track_weights"].astype(np.float32),
        )
    result.validate()
    return result


def _label_codebook() -> np.ndarray:
    codebook = np.full(29, -1, dtype=np.int16)
    for name, labels in SAPIENS2_DOME29_LAYER_IDS.items():
        codebook[np.asarray(labels, dtype=np.int64)] = SEMANTIC_CODEBOOK[name]
    return codebook


def _patch_ncc(
    first: np.ndarray,
    first_point: np.ndarray,
    second: np.ndarray,
    second_point: np.ndarray,
) -> float:
    first_patch = cv2.getRectSubPix(first, (11, 11), tuple(first_point)).astype(np.float32)
    second_patch = cv2.getRectSubPix(second, (11, 11), tuple(second_point)).astype(np.float32)
    first_patch -= float(first_patch.mean())
    second_patch -= float(second_patch.mean())
    denominator = float(np.linalg.norm(first_patch) * np.linalg.norm(second_patch))
    return float(np.sum(first_patch * second_patch) / denominator) if denominator > 1e-8 else 0.0


def _read_real_inputs(
    frames: list[Any],
    semantic_root: Path,
    semantic_hashes: dict[int, str],
    *,
    maximum_dimension: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[np.ndarray],
    list[float],
]:
    grays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    scales: list[float] = []
    labels: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    background_luminance: list[float] = []
    for frame in frames:
        gray = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(frame.mask_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or gray.shape != mask.shape:
            raise ValueError("Q02 image and mask evidence must align")
        height, width = gray.shape
        scale = min(1.0, maximum_dimension / float(max(height, width)))
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        kernel_size = max(3, round(min(mask.shape) * 0.01) | 1)
        mask = cv2.erode(mask, np.ones((kernel_size, kernel_size), dtype=np.uint8))
        background = gray[mask == 0]
        background_luminance.append(
            max(1.0, float(np.median(background)) if len(background) else float(np.median(gray)))
        )
        semantic_path = semantic_root / f"{frame.image_path.stem}.npz"
        if not semantic_path.is_file():
            raise FileNotFoundError(f"Q02 semantic frame is absent: {semantic_path}")
        if sha256_file(semantic_path) != semantic_hashes[frame.record.source_frame_index]:
            raise ValueError("Q02 semantic frame hash does not match S01 qualification")
        with np.load(semantic_path, allow_pickle=False) as archive:
            frame_labels = archive["labels"].astype(np.uint8)
            frame_confidence = archive["confidence"].astype(np.float32)
            source = int(archive["source_frame_index"])
        if (
            frame_labels.shape != (height, width)
            or frame_confidence.shape != (height, width)
            or source != frame.record.source_frame_index
        ):
            raise ValueError("Q02 semantic evidence does not align with its training frame")
        grays.append(gray)
        masks.append(mask)
        scales.append(scale)
        labels.append(frame_labels)
        confidences.append(frame_confidence)
    return grays, masks, scales, labels, confidences, background_luminance


def _track_lk(
    grays: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    maximum_corners: int,
    start_stride: int,
    maximum_track_steps: int,
) -> list[_RawTrack]:
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        "minEigThreshold": 1.0e-4,
    }
    tracks: list[_RawTrack] = []
    for start in range(0, len(grays) - 1, start_stride):
        seeds = cv2.goodFeaturesToTrack(
            grays[start],
            maxCorners=maximum_corners,
            qualityLevel=0.01,
            minDistance=7,
            mask=masks[start],
            blockSize=7,
        )
        if seeds is None:
            continue
        current = seeds.reshape(-1, 2)
        active = np.arange(len(current))
        points = [[point.copy()] for point in current]
        errors = [[0.0] for _ in current]
        last_slot = min(len(grays) - 1, start + maximum_track_steps)
        for slot in range(start, last_slot):
            if not len(current):
                break
            following, forward_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                grays[slot], grays[slot + 1], current.reshape(-1, 1, 2), None, **lk
            )
            if following is None or forward_status is None:
                break
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                grays[slot + 1], grays[slot], following, None, **lk
            )
            if backward is None or backward_status is None:
                break
            following = following.reshape(-1, 2)
            backward = backward.reshape(-1, 2)
            forward_status = forward_status.reshape(-1).astype(bool)
            backward_status = backward_status.reshape(-1).astype(bool)
            fb_error = np.linalg.norm(backward - current, axis=1)
            x = np.rint(following[:, 0]).astype(np.int64)
            y = np.rint(following[:, 1]).astype(np.int64)
            inside = (
                (x >= 0)
                & (x < masks[slot + 1].shape[1])
                & (y >= 0)
                & (y < masks[slot + 1].shape[0])
            )
            mask_ok = np.zeros(len(current), dtype=bool)
            mask_ok[inside] = masks[slot + 1][y[inside], x[inside]] > 0
            keep = forward_status & backward_status & (fb_error <= 1.0) & mask_ok
            for local_index, passed in enumerate(keep):
                original_index = int(active[local_index])
                if passed:
                    points[original_index].append(following[local_index].copy())
                    errors[original_index].append(float(fb_error[local_index]))
                elif len(points[original_index]) >= 2:
                    length = len(points[original_index])
                    tracks.append(
                        _RawTrack(
                            frame_slots=list(range(start, start + length)),
                            points=points[original_index],
                            local_fb_errors=errors[original_index],
                        )
                    )
            active = active[keep]
            current = following[keep]
        for active_index in active:
            original_index = int(active_index)
            length = len(points[original_index])
            if length >= 2:
                tracks.append(
                    _RawTrack(
                        frame_slots=list(range(start, start + length)),
                        points=points[original_index],
                        local_fb_errors=errors[original_index],
                    )
                )
    return tracks


def _sample_semantic(
    labels: np.ndarray,
    confidence: np.ndarray,
    point: np.ndarray,
    scale: float,
    codebook: np.ndarray,
) -> tuple[int, float]:
    original = point / scale
    x = int(np.clip(round(float(original[0])), 0, labels.shape[1] - 1))
    y = int(np.clip(round(float(original[1])), 0, labels.shape[0] - 1))
    label = int(labels[y, x])
    return int(codebook[label]), float(confidence[y, x])


def _qualify_tracks(
    raw_tracks: list[_RawTrack],
    frames: list[Any],
    grays: list[np.ndarray],
    scales: list[float],
    labels: list[np.ndarray],
    confidences: list[np.ndarray],
    background_luminance: list[float],
    gate: MaterialTrackGate,
) -> list[_QualifiedTrack]:
    codebook = _label_codebook()
    qualified: list[_QualifiedTrack] = []
    for track in raw_tracks:
        if len(track.points) < gate.minimum_observations:
            continue
        codes: list[int] = []
        semantic_confidence: list[float] = []
        luminance: list[float] = []
        for slot, point in zip(track.frame_slots, track.points, strict=True):
            code, confidence = _sample_semantic(
                labels[slot], confidences[slot], point, scales[slot], codebook
            )
            codes.append(code)
            semantic_confidence.append(confidence)
            patch = cv2.getRectSubPix(grays[slot], (5, 5), tuple(point)).astype(np.float32)
            luminance.append(float(patch.mean()) / background_luminance[slot])
        valid_codes = [code for code in codes if code >= 0]
        if not valid_codes:
            continue
        counts = np.bincount(np.asarray(valid_codes), minlength=len(SEMANTIC_NAMES))
        semantic_code = int(np.argmax(counts))
        stability = codes.count(semantic_code) / len(codes)
        median_confidence = float(np.median(semantic_confidence))
        endpoint_ncc = _patch_ncc(
            grays[track.frame_slots[0]],
            track.points[0],
            grays[track.frame_slots[-1]],
            track.points[-1],
        )
        if (
            stability < gate.minimum_semantic_stability
            or median_confidence < gate.minimum_median_semantic_confidence
            or endpoint_ncc < gate.minimum_endpoint_patch_ncc
        ):
            continue
        qualified.append(
            _QualifiedTrack(
                raw=track,
                semantic_code=semantic_code,
                semantic_stability=stability,
                median_semantic_confidence=median_confidence,
                endpoint_patch_ncc=endpoint_ncc,
                semantic_confidences=semantic_confidence,
                normalized_luminance=luminance,
            )
        )
    return qualified


def _span_ordinals(track: _QualifiedTrack, frames: list[Any]) -> int:
    return int(
        frames[track.raw.frame_slots[-1]].record.ordinal
        - frames[track.raw.frame_slots[0]].record.ordinal
    )


def _reverse_audit(
    tracks: list[_QualifiedTrack],
    frames: list[Any],
    grays: list[np.ndarray],
    gate: MaterialTrackGate,
) -> dict[str, float | int | bool | None]:
    candidates = [track for track in tracks if _span_ordinals(track, frames) >= 15]
    candidates.sort(
        key=lambda track: (
            -_span_ordinals(track, frames),
            track.raw.frame_slots[0],
            float(track.raw.points[0][0]),
            float(track.raw.points[0][1]),
        )
    )
    selected = candidates[: gate.reverse_audit_maximum_tracks]
    errors: list[float] = []
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        "minEigThreshold": 1.0e-4,
    }
    groups: dict[tuple[int, int], list[_QualifiedTrack]] = {}
    for track in selected:
        key = (track.raw.frame_slots[0], track.raw.frame_slots[-1])
        groups.setdefault(key, []).append(track)
    for (start, stop), group in groups.items():
        current = np.stack([track.raw.points[-1] for track in group]).astype(np.float32)
        valid = np.ones(len(group), dtype=bool)
        for slot in range(stop - 1, start - 1, -1):
            previous, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                grays[slot + 1], grays[slot], current.reshape(-1, 1, 2), None, **lk
            )
            if previous is None or status is None:
                valid[:] = False
                break
            previous = previous.reshape(-1, 2)
            valid &= status.reshape(-1).astype(bool)
            current = previous
        returned = np.linalg.norm(
            current - np.stack([track.raw.points[0] for track in group]), axis=1
        )
        errors.extend(
            float(value) if passed else math.inf
            for value, passed in zip(returned, valid, strict=True)
        )
    finite = np.asarray([value for value in errors if math.isfinite(value)], dtype=np.float64)
    median = float(np.median(finite)) if len(finite) else None
    p95 = float(np.quantile(finite, 0.95)) if len(finite) else None
    pass_fraction = (
        float(np.mean(np.asarray(errors) <= gate.reverse_return_error_p95_pixels))
        if errors
        else 0.0
    )
    return {
        "audited_track_count": len(errors),
        "finite_track_count": len(finite),
        "median_return_error_pixels": median,
        "p95_return_error_pixels": p95,
        "pass_fraction": pass_fraction,
        "pass": p95 is not None
        and p95 <= gate.reverse_return_error_p95_pixels
        and pass_fraction >= gate.reverse_pass_fraction,
    }


def _linear_prediction(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    regularizer = 1.0e-5 * np.eye(train_x.shape[1])
    coefficients = np.linalg.solve(train_x.T @ train_x + regularizer, train_x.T @ train_y)
    return test_x @ coefficients


def _photometric_observability(
    tracks: list[_QualifiedTrack],
    frames: list[Any],
    gate: MaterialTrackGate,
    *,
    seed: int,
) -> dict[str, Any]:
    candidates = [
        track
        for track in tracks
        if _span_ordinals(track, frames) >= 45 and len(track.raw.points) >= 12
    ]
    improvements: list[float] = []
    shuffled_improvements: list[float] = []
    for track_index, track in enumerate(candidates):
        ordinals = np.asarray(
            [frames[slot].record.ordinal for slot in track.raw.frame_slots], dtype=np.float64
        )
        theta = ordinals * (2.0 * math.pi / 180.0)
        values = np.log(np.clip(np.asarray(track.normalized_luminance), 0.02, None))
        design = np.column_stack(
            (
                np.ones(len(theta)),
                np.cos(theta),
                np.sin(theta),
                np.cos(2 * theta),
                np.sin(2 * theta),
            )
        )
        train = np.arange(len(theta)) % 2 == 0
        test = ~train
        if int(train.sum()) < design.shape[1] or int(test.sum()) < 3:
            continue
        baseline = np.full(int(test.sum()), float(values[train].mean()))
        harmonic = _linear_prediction(design[train], values[train], design[test])
        baseline_rmse = float(np.sqrt(np.mean(np.square(values[test] - baseline))))
        harmonic_rmse = float(np.sqrt(np.mean(np.square(values[test] - harmonic))))
        if baseline_rmse <= 1.0e-6:
            continue
        improvements.append((baseline_rmse - harmonic_rmse) / baseline_rmse)
        rng = np.random.default_rng(seed + track_index)
        shuffled_theta = theta[rng.permutation(len(theta))]
        shuffled_design = np.column_stack(
            (
                np.ones(len(theta)),
                np.cos(shuffled_theta),
                np.sin(shuffled_theta),
                np.cos(2 * shuffled_theta),
                np.sin(2 * shuffled_theta),
            )
        )
        shuffled = _linear_prediction(shuffled_design[train], values[train], shuffled_design[test])
        shuffled_rmse = float(np.sqrt(np.mean(np.square(values[test] - shuffled))))
        shuffled_improvements.append((baseline_rmse - shuffled_rmse) / baseline_rmse)
    median_improvement = float(np.median(improvements)) if improvements else None
    median_shuffled = float(np.median(shuffled_improvements)) if shuffled_improvements else None
    positive_fraction = float(np.mean(np.asarray(improvements) > 0)) if improvements else 0.0
    margin = (
        median_improvement - median_shuffled
        if median_improvement is not None and median_shuffled is not None
        else None
    )
    blockers: list[str] = []
    if len(improvements) < gate.photometric_minimum_90_degree_tracks:
        blockers.append("insufficient_scored_90_degree_tracks")
    if (
        median_improvement is None
        or median_improvement < gate.photometric_median_harmonic_improvement
    ):
        blockers.append("harmonic_held_out_improvement_gate_failed")
    if positive_fraction < gate.photometric_positive_track_fraction:
        blockers.append("positive_track_fraction_gate_failed")
    if margin is None or margin < gate.photometric_shuffled_margin:
        blockers.append("shuffled_angle_control_gate_failed")
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "model": "per_track_varpro_second_order_rotation_harmonics",
        "camera_exposure_control": "background_median_normalization",
        "rotation_parameterization": "two_degrees_per_frozen_candidate_ordinal",
        "candidate_90_degree_track_count": len(candidates),
        "scored_track_count": len(improvements),
        "median_held_out_relative_improvement": median_improvement,
        "positive_track_fraction": positive_fraction,
        "median_shuffled_relative_improvement": median_shuffled,
        "real_minus_shuffled_margin": margin,
        "geometry_frozen": True,
        "albedo_coefficients_eliminated": True,
    }


def _write_binding(
    path: Path,
    tracks: list[_QualifiedTrack],
    frames: list[Any],
    scales: list[float],
) -> Path:
    offsets = [0]
    ordinals: list[int] = []
    sources: list[int] = []
    pixels: list[np.ndarray] = []
    semantic_codes: list[int] = []
    semantic_confidence: list[float] = []
    luminance: list[float] = []
    fb_errors: list[float] = []
    spans: list[float] = []
    weights: list[float] = []
    codebook = _label_codebook()
    for track in tracks:
        spans.append(2.0 * _span_ordinals(track, frames))
        weights.append(
            track.semantic_stability
            * track.median_semantic_confidence
            * max(0.0, track.endpoint_patch_ncc)
        )
        for slot, point, confidence, value, fb_error in zip(
            track.raw.frame_slots,
            track.raw.points,
            track.semantic_confidences,
            track.normalized_luminance,
            track.raw.local_fb_errors,
            strict=True,
        ):
            frame = frames[slot]
            ordinals.append(frame.record.ordinal)
            sources.append(frame.record.source_frame_index)
            pixels.append(point / scales[slot])
            semantic_codes.append(track.semantic_code)
            semantic_confidence.append(confidence)
            luminance.append(value)
            fb_errors.append(fb_error)
        offsets.append(len(ordinals))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray("frayid_v2_visibility_material_tracks.v1"),
        track_offsets=np.asarray(offsets, dtype=np.int64),
        frame_ordinals=np.asarray(ordinals, dtype=np.int64),
        source_frame_indices=np.asarray(sources, dtype=np.int64),
        pixels=np.asarray(pixels, dtype=np.float32).reshape(-1, 2),
        semantic_codes=np.asarray(semantic_codes, dtype=np.int16),
        semantic_confidence=np.asarray(semantic_confidence, dtype=np.float32),
        normalized_luminance=np.asarray(luminance, dtype=np.float32),
        local_forward_backward_error=np.asarray(fb_errors, dtype=np.float32),
        track_span_degrees=np.asarray(spans, dtype=np.float32),
        track_weights=np.asarray(weights, dtype=np.float32),
        semantic_names=np.asarray(json.dumps(SEMANTIC_NAMES)),
        label_codebook=codebook,
        role=np.asarray("uncertain_visibility_bounded_material_track_proposals_not_truth"),
    )
    load_visibility_bounded_material_tracks(path)
    return path


def scan_visibility_bounded_material_tracks(
    manifest_path: Path,
    semantic_root: Path,
    semantic_qualification_path: Path,
    output_path: Path,
    photometric_output_path: Path,
    binding_path: Path,
    *,
    source_revision: str,
    validation_path: Path | None = None,
    maximum_dimension: int = 720,
    maximum_corners: int = 600,
    start_stride: int = 8,
    maximum_track_steps: int = 75,
    seed: int = 20260902,
    gate: MaterialTrackGate | None = None,
) -> tuple[Path, Path]:
    """Q02: qualify material tracks and photometry independently on train-only evidence."""

    gate = gate or MaterialTrackGate()
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("Q02 source revision must be a 40-character lowercase Git commit")
    validation_path = validation_path or manifest_path.parent / "dataset_validation.json"
    reject_sealed_capability(
        [
            manifest_path,
            semantic_root,
            semantic_qualification_path,
            validation_path,
            output_path,
            photometric_output_path,
            binding_path,
        ]
    )
    if output_path.exists() or photometric_output_path.exists() or binding_path.exists():
        raise FileExistsError("Q02 outputs are immutable")
    validation = read_json(validation_path)
    if validation.get("status") != "ready" or validation.get("blockers"):
        raise ValueError("Q02 requires a ready train dataset validation")
    semantic_report = read_json(semantic_qualification_path)
    if semantic_report.get("status") != "pass":
        raise ValueError("Q02 requires passing S01 semantic qualification")
    semantic_hashes = {
        int(record["source_frame_index"]): str(record["semantic_sha256"])
        for record in semantic_report.get("frame_records", [])
    }
    frames, held_out_record_count = _bind_training_frames(manifest_path)
    sources = [frame.record.source_frame_index for frame in frames]
    if sorted(semantic_hashes) != sorted(sources):
        raise ValueError("Q02 S01 qualification must exactly cover accepted train frames")
    grays, masks, scales, labels, confidences, background = _read_real_inputs(
        frames,
        semantic_root,
        semantic_hashes,
        maximum_dimension=maximum_dimension,
    )
    raw_tracks = _track_lk(
        grays,
        masks,
        maximum_corners=maximum_corners,
        start_stride=start_stride,
        maximum_track_steps=maximum_track_steps,
    )
    qualified = _qualify_tracks(
        raw_tracks,
        frames,
        grays,
        scales,
        labels,
        confidences,
        background,
        gate,
    )
    spans = np.asarray([_span_ordinals(track, frames) for track in qualified])
    tracks_30 = [track for track, span in zip(qualified, spans, strict=True) if span >= 15]
    tracks_90 = [track for track, span in zip(qualified, spans, strict=True) if span >= 45]
    tracks_180 = [track for track, span in zip(qualified, spans, strict=True) if span >= 90]
    supported_layers = sorted(
        {SEMANTIC_NAMES[track.semantic_code] for track in tracks_30 if track.semantic_code >= 0}
    )
    reverse = _reverse_audit(tracks_30, frames, grays, gate)
    material_blockers: list[str] = []
    if len(tracks_30) < gate.minimum_30_degree_tracks:
        material_blockers.append("insufficient_30_degree_material_tracks")
    if len(tracks_90) < gate.minimum_90_degree_tracks:
        material_blockers.append("insufficient_90_degree_material_tracks")
    if len(supported_layers) < gate.minimum_supported_semantic_layers:
        material_blockers.append("insufficient_semantic_layer_track_support")
    if reverse["pass"] is not True:
        material_blockers.append("global_reverse_cycle_gate_failed")
    material_eligible = not material_blockers
    photometric = _photometric_observability(tracks_90, frames, gate, seed=seed)
    binding_tracks = tracks_30 if material_eligible else qualified
    _write_binding(binding_path, binding_tracks, frames, scales)
    layer_counts = {
        name: sum(track.semantic_code == SEMANTIC_CODEBOOK[name] for track in tracks_30)
        for name in SEMANTIC_NAMES
    }
    material_report = {
        "schema_version": "frayid_v2_q02a_material_track_qualification.v1",
        "status": "pass" if material_eligible else "fail",
        "qualification_id": "postv2_q02a_visibility_material_tracklets_r01",
        "source_revision": source_revision,
        "purpose": "material_observation_model_selection_not_geometry_or_photometric_science",
        "frozen_gate": gate.as_dict(),
        "algorithm": {
            "proposal": "opencv_pyramidal_lucas_kanade",
            "seed": "mask_eroded_shi_tomasi",
            "local_consistency": "one_pixel_forward_backward",
            "visibility": "terminate_on_mask_or_flow_failure",
            "material_control": "s01_semantic_stability_and_confidence",
            "appearance_control": "endpoint_patch_ncc",
            "maximum_dimension": maximum_dimension,
            "maximum_corners": maximum_corners,
            "start_stride": start_stride,
            "maximum_track_steps": maximum_track_steps,
        },
        "track_metrics": {
            "raw_track_count": len(raw_tracks),
            "qualified_track_count": len(qualified),
            "qualified_30_degree_track_count": len(tracks_30),
            "qualified_90_degree_track_count": len(tracks_90),
            "qualified_180_degree_track_count": len(tracks_180),
            "supported_semantic_layers": supported_layers,
            "layer_30_degree_track_counts": layer_counts,
            "median_semantic_stability": (
                float(np.median([track.semantic_stability for track in tracks_30]))
                if tracks_30
                else 0.0
            ),
            "median_semantic_confidence": (
                float(np.median([track.median_semantic_confidence for track in tracks_30]))
                if tracks_30
                else 0.0
            ),
            "median_endpoint_patch_ncc": (
                float(np.median([track.endpoint_patch_ncc for track in tracks_30]))
                if tracks_30
                else 0.0
            ),
            "reverse_cycle_audit": reverse,
        },
        "material_track_route": {
            "eligible": material_eligible,
            "role": "uncertain_visibility_bounded_material_track_proposals_not_truth",
            "blockers": material_blockers,
        },
        "photometric_route": {
            "separately_reported": True,
            "qualification_id": "postv2_q02b_rotation_photometric_observability_r01",
            "failure_does_not_invalidate_q02a": True,
        },
        "binding": {
            "path": str(binding_path),
            "sha256": sha256_file(binding_path),
            "track_count": len(binding_tracks),
            "role": "uncertain_visibility_bounded_material_track_proposals_not_truth",
        },
        "source_hashes": {
            "manifest": sha256_file(manifest_path),
            "validation": sha256_file(validation_path),
            "semantic_qualification": sha256_file(semantic_qualification_path),
        },
        "access_counters": {
            "accepted_training_records_bound": len(frames),
            "training_images_read": len(frames),
            "training_masks_read": len(frames),
            "training_semantic_frames_read": len(frames),
            "held_out_records_present_but_not_bound": held_out_record_count,
            "held_out_images_read": 0,
            "development_metrics_read": 0,
            "sealed_test_accesses": 0,
        },
        "optimizer_steps": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "Thirty, ninety, and 180 degree counts are reported separately; occlusion is a boundary.",
            "Material-track eligibility cannot enable photometry when the harmonic control fails.",
            "The binding contains uncertain proposals, never authoritative cross-frame identity.",
        ],
    }
    material_report_path = write_json(output_path, material_report)
    photometric_report = {
        "schema_version": "frayid_v2_q02b_photometric_observability_qualification.v1",
        "status": "pass" if photometric["eligible"] else "fail",
        "qualification_id": "postv2_q02b_rotation_photometric_observability_r01",
        "source_revision": source_revision,
        "purpose": "frozen_geometry_rotation_photometric_observability_not_normal_science",
        "frozen_gate": {
            key: value for key, value in gate.as_dict().items() if key.startswith("photometric_")
        },
        "material_route": {
            "qualification_id": "postv2_q02a_visibility_material_tracklets_r01",
            "status": "pass" if material_eligible else "fail",
            "report_sha256": sha256_file(material_report_path),
        },
        "photometric_normal_route": photometric,
        "activation_eligible": material_eligible and bool(photometric["eligible"]),
        "q02b_failure_invalidates_q02a": False,
        "binding": {
            "sha256": sha256_file(binding_path),
            "track_count": len(binding_tracks),
            "role": "read_only_material_proposals_with_profiled_per_track_albedo",
        },
        "access_counters": material_report["access_counters"],
        "optimizer_steps": 0,
        "geometry_optimizer_steps": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
    }
    return material_report_path, write_json(photometric_output_path, photometric_report)
