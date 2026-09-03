from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class FrameTimestamp(BaseModel):
    """The native timing fields for one sequentially decoded video frame."""

    model_config = ConfigDict(extra="forbid")

    decode_index: int = Field(ge=0)
    pts_seconds: float | None = None
    best_effort_timestamp_seconds: float | None = None
    dts_seconds: float | None = None
    duration_seconds: float | None = None
    selected_timestamp_seconds: float | None = None
    selected_timestamp_source: Literal["pts", "best_effort", "dts", "missing"]
    key_frame: bool = False
    picture_type: str | None = None


class VideoProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codec: str
    profile: str | None = None
    pixel_format: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    reported_frame_count: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    average_frame_rate: str | None = None
    nominal_frame_rate: str | None = None
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    field_order: str | None = None
    container_format: str | None = None
    source_size_bytes: int = Field(ge=0)


def executable_version(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"required executable is unavailable: {executable}")
    result = subprocess.run([resolved, "-version"], check=True, capture_output=True, text=True)
    return result.stdout.splitlines()[0]


def probe_video_forensics(
    path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
) -> tuple[VideoProbe, list[FrameTimestamp], dict[str, Any]]:
    """Read native per-frame timing without random seeking or timestamp synthesis."""
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-show_frames",
        "-show_entries",
        (
            "stream=codec_name,profile,pix_fmt,width,height,nb_frames,duration,"
            "avg_frame_rate,r_frame_rate,color_range,color_space,color_transfer,"
            "color_primaries,field_order:"
            "format=format_name,duration,size:"
            "frame=media_type,pts_time,best_effort_timestamp_time,pkt_dts_time,"
            "pkt_duration_time,key_frame,pict_type"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("ffprobe payload must be an object")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ValueError(f"no video stream found: {path}")
    stream: dict[str, Any] = streams[0]
    format_payload = payload.get("format")
    container = format_payload if isinstance(format_payload, dict) else {}
    probe = VideoProbe(
        codec=str(stream.get("codec_name") or "unknown"),
        profile=_optional_string(stream.get("profile")),
        pixel_format=_optional_string(stream.get("pix_fmt")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        reported_frame_count=_optional_int(stream.get("nb_frames")),
        duration_seconds=_optional_float(stream.get("duration") or container.get("duration")),
        average_frame_rate=_optional_string(stream.get("avg_frame_rate")),
        nominal_frame_rate=_optional_string(stream.get("r_frame_rate")),
        color_range=_optional_string(stream.get("color_range")),
        color_space=_optional_string(stream.get("color_space")),
        color_transfer=_optional_string(stream.get("color_transfer")),
        color_primaries=_optional_string(stream.get("color_primaries")),
        field_order=_optional_string(stream.get("field_order")),
        container_format=_optional_string(container.get("format_name")),
        source_size_bytes=int(container.get("size") or path.stat().st_size),
    )
    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, list):
        raise ValueError("ffprobe did not return frame timing")
    timestamps = parse_frame_timestamps(frames_payload)
    return (
        probe,
        timestamps,
        {
            "command": command,
            "ffprobe_version": executable_version(ffprobe_bin),
        },
    )


def parse_frame_timestamps(frames: Sequence[object]) -> list[FrameTimestamp]:
    """Parse timing while retaining missing PTS instead of inventing CFR timestamps."""
    parsed: list[FrameTimestamp] = []
    for raw in frames:
        if not isinstance(raw, dict) or raw.get("media_type", "video") != "video":
            continue
        pts = _optional_float(raw.get("pts_time"))
        best_effort = _optional_float(raw.get("best_effort_timestamp_time"))
        dts = _optional_float(raw.get("pkt_dts_time"))
        if pts is not None:
            selected = pts
            source: Literal["pts", "best_effort", "dts", "missing"] = "pts"
        elif best_effort is not None:
            selected = best_effort
            source = "best_effort"
        elif dts is not None:
            selected = dts
            source = "dts"
        else:
            selected = None
            source = "missing"
        parsed.append(
            FrameTimestamp(
                decode_index=len(parsed),
                pts_seconds=pts,
                best_effort_timestamp_seconds=best_effort,
                dts_seconds=dts,
                duration_seconds=_optional_float(raw.get("pkt_duration_time")),
                selected_timestamp_seconds=selected,
                selected_timestamp_source=source,
                key_frame=bool(int(raw.get("key_frame", 0))),
                picture_type=_optional_string(raw.get("pict_type")),
            )
        )
    return parsed


def summarize_timestamps(timestamps: Sequence[FrameTimestamp]) -> dict[str, Any]:
    selected = [item.selected_timestamp_seconds for item in timestamps]
    missing_count = sum(value is None for value in selected)
    finite = np.asarray([value for value in selected if value is not None], dtype=np.float64)
    deltas = np.diff(finite)
    monotonic = bool(missing_count == 0 and np.all(deltas > 0.0))
    if deltas.size:
        median_delta = float(np.median(deltas))
        anomaly_count = int(np.sum(np.abs(deltas - median_delta) > 0.001))
        minimum_delta: float | None = float(deltas.min())
        maximum_delta: float | None = float(deltas.max())
    else:
        median_delta = None
        anomaly_count = 0
        minimum_delta = None
        maximum_delta = None
    return {
        "frame_timestamp_count": len(timestamps),
        "missing_timestamp_count": missing_count,
        "native_pts_count": sum(item.pts_seconds is not None for item in timestamps),
        "best_effort_fallback_count": sum(
            item.selected_timestamp_source == "best_effort" for item in timestamps
        ),
        "dts_fallback_count": sum(item.selected_timestamp_source == "dts" for item in timestamps),
        "strictly_monotonic": monotonic,
        "median_delta_seconds": median_delta,
        "minimum_delta_seconds": minimum_delta,
        "maximum_delta_seconds": maximum_delta,
        "delta_anomaly_count_over_1ms": anomaly_count,
    }


def iter_sequential_rgb_frames(
    path: Path,
    *,
    width: int,
    height: int,
    ffmpeg_bin: str = "ffmpeg",
) -> Iterator[np.ndarray]:
    """Yield decoded RGB frames from one forward-only ffmpeg process."""
    resolved = shutil.which(ffmpeg_bin)
    if resolved is None:
        raise FileNotFoundError(f"required executable is unavailable: {ffmpeg_bin}")
    command = [
        resolved,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    assert process.stderr is not None
    frame_size = width * height * 3
    completed = False
    try:
        while True:
            raw = _read_exact_or_eof(process.stdout, frame_size)
            if raw is None:
                break
            if len(raw) != frame_size:
                raise RuntimeError(
                    f"truncated decoded frame: expected {frame_size} bytes, received {len(raw)}"
                )
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        completed = True
        if return_code != 0:
            raise RuntimeError(f"ffmpeg sequential decode failed: {stderr.strip()}")
    finally:
        process.stdout.close()
        process.stderr.close()
        if not completed and process.poll() is None:
            process.terminate()
            process.wait()


def decoded_frame_metrics(
    rgb: np.ndarray,
    *,
    previous_rgb: np.ndarray | None,
) -> dict[str, float | str | None]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    adjacent_motion = (
        None
        if previous_rgb is None
        else float(
            np.mean(
                np.abs(
                    rgb.astype(np.float32, copy=False) - previous_rgb.astype(np.float32, copy=False)
                )
            )
        )
    )
    return {
        "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_luminance": float(gray.mean()),
        "near_black_fraction": float(np.mean(gray <= 5)),
        "near_white_fraction": float(np.mean(gray >= 250)),
        "adjacent_rgb_absolute_difference_mean": adjacent_motion,
    }


def background_border_mask(height: int, width: int, *, border_fraction: float = 0.22) -> np.ndarray:
    if not 0.05 <= border_fraction <= 0.45:
        raise ValueError("border_fraction must be between 0.05 and 0.45")
    mask = np.full((height, width), 255, dtype=np.uint8)
    x_margin = round(width * border_fraction)
    y_margin = round(height * border_fraction)
    mask[y_margin : height - y_margin, x_margin : width - x_margin] = 0
    return mask


def estimate_background_transforms(
    rgb_frames: Sequence[np.ndarray],
    *,
    source_indices: Sequence[int] | None = None,
    border_fraction: float = 0.22,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Estimate repeatable background similarity motion relative to the first frame."""
    if len(rgb_frames) < 2:
        raise ValueError("background audit requires at least two frames")
    first_shape = rgb_frames[0].shape
    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError("background audit expects RGB images")
    if any(frame.shape != first_shape for frame in rgb_frames):
        raise ValueError("background audit frames must have identical dimensions")
    height, width = first_shape[:2]
    mask = background_border_mask(height, width, border_fraction=border_fraction)
    reference = cv2.cvtColor(rgb_frames[0], cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(
        reference,
        maxCorners=5000,
        qualityLevel=0.005,
        minDistance=4,
        mask=mask,
        blockSize=5,
    )
    if points is None or len(points) < 12:
        raise ValueError("insufficient static-background features")
    indices = list(source_indices) if source_indices is not None else list(range(len(rgb_frames)))
    if len(indices) != len(rgb_frames):
        raise ValueError("source_indices must match rgb_frames")
    records: list[dict[str, float | int | bool | None]] = []
    for slot, (source_index, rgb) in enumerate(zip(indices, rgb_frames, strict=True)):
        if slot == 0:
            records.append(
                {
                    "source_frame_index": source_index,
                    "success": True,
                    "inlier_count": len(points),
                    "dx_pixels": 0.0,
                    "dy_pixels": 0.0,
                    "scale": 1.0,
                    "rotation_degrees": 0.0,
                    "median_reprojection_error_pixels": 0.0,
                }
            )
            continue
        target = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            reference,
            target,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
        )
        if tracked is None or status is None:
            records.append(_failed_transform_record(source_index))
            continue
        selected = status.reshape(-1).astype(bool)
        source_points = points.reshape(-1, 2)[selected]
        target_points = tracked.reshape(-1, 2)[selected]
        if len(source_points) < 12:
            records.append(_failed_transform_record(source_index))
            continue
        cv2.setRNGSeed(seed + slot)
        matrix, inliers = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=0.75,
            maxIters=5000,
            confidence=0.999,
            refineIters=20,
        )
        if matrix is None or inliers is None:
            records.append(_failed_transform_record(source_index))
            continue
        inlier_mask = inliers.reshape(-1).astype(bool)
        if int(inlier_mask.sum()) < 12:
            records.append(_failed_transform_record(source_index))
            continue
        homogeneous = np.concatenate(
            [source_points, np.ones((len(source_points), 1), dtype=np.float32)], axis=1
        )
        projected = homogeneous @ matrix.T
        residuals = np.linalg.norm(projected - target_points, axis=1)
        a = float(matrix[0, 0])
        b = float(matrix[1, 0])
        records.append(
            {
                "source_frame_index": source_index,
                "success": True,
                "inlier_count": int(inlier_mask.sum()),
                "dx_pixels": float(matrix[0, 2]),
                "dy_pixels": float(matrix[1, 2]),
                "scale": float(math.hypot(a, b)),
                "rotation_degrees": float(math.degrees(math.atan2(b, a))),
                "median_reprojection_error_pixels": float(np.median(residuals[inlier_mask])),
            }
        )
    successful = [record for record in records if record["success"]]
    return {
        "reference_source_frame_index": indices[0],
        "sample_count": len(records),
        "successful_count": len(successful),
        "border_fraction": border_fraction,
        "records": records,
        "summary": {
            "median_reprojection_error_pixels": _median_record_value(
                successful, "median_reprojection_error_pixels"
            ),
            "maximum_absolute_translation_pixels": max(
                (
                    math.hypot(float(item["dx_pixels"]), float(item["dy_pixels"]))
                    for item in successful
                    if item["dx_pixels"] is not None and item["dy_pixels"] is not None
                ),
                default=None,
            ),
            "maximum_absolute_rotation_degrees": max(
                (
                    abs(float(item["rotation_degrees"]))
                    for item in successful
                    if item["rotation_degrees"] is not None
                ),
                default=None,
            ),
            "maximum_absolute_scale_change": max(
                (
                    abs(float(item["scale"]) - 1.0)
                    for item in successful
                    if item["scale"] is not None
                ),
                default=None,
            ),
        },
    }


def camera_verdict(background_audit: dict[str, Any]) -> str:
    summary = background_audit.get("summary")
    if not isinstance(summary, dict):
        return "indeterminate"
    reprojection = _optional_float(summary.get("median_reprojection_error_pixels"))
    translation = _optional_float(summary.get("maximum_absolute_translation_pixels"))
    rotation = _optional_float(summary.get("maximum_absolute_rotation_degrees"))
    scale_change = _optional_float(summary.get("maximum_absolute_scale_change"))
    if None in (reprojection, translation, rotation, scale_change):
        return "indeterminate"
    assert reprojection is not None
    assert translation is not None
    assert rotation is not None
    assert scale_change is not None
    if reprojection <= 0.3 and translation <= 0.5 and rotation <= 0.02 and scale_change <= 0.001:
        return "fixed_to_subpixel_precision"
    return "background_motion_detected"


def _read_exact_or_eof(stream: Any, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if not chunks:
        return None
    return b"".join(chunks)


def _failed_transform_record(source_index: int) -> dict[str, float | int | bool | None]:
    return {
        "source_frame_index": source_index,
        "success": False,
        "inlier_count": 0,
        "dx_pixels": None,
        "dy_pixels": None,
        "scale": None,
        "rotation_degrees": None,
        "median_reprojection_error_pixels": None,
    }


def _median_record_value(records: Sequence[dict[str, Any]], name: str) -> float | None:
    values = [_optional_float(record.get(name)) for record in records]
    finite = [value for value in values if value is not None]
    return float(np.median(np.asarray(finite, dtype=np.float64))) if finite else None


def _optional_string(value: object) -> str | None:
    if value is None or value == "N/A":
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "N/A":
        return None
    result = float(str(value))
    return result if math.isfinite(result) else None


def _optional_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    return int(str(value))
