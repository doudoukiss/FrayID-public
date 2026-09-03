from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.video_forensics import probe_video_forensics, summarize_timestamps
from frayid.v3.controlled_camera_calibration import ControlledCharucoBoardSpec

EXPERIMENT_ID: Literal["postv3_v01_controlled_recapture_evidence_master_r01"] = (
    "postv3_v01_controlled_recapture_evidence_master_r01"
)
_WINDOW_NAME = "FrayID guided calibration - Q or Escape stops"
_MINIMUM_GUIDANCE_CORNERS = 12
_DEFAULT_STABLE_SECONDS = 0.8
_DEFAULT_TARGET_TOLERANCE = 0.085
_ROTATION_PRE_ROLL_SECONDS = 15.0
_ROTATION_HOLD_PERIOD_SECONDS = 6.0
_ROTATION_STABLE_SECONDS = 3.0
_ROTATION_TAIL_SECONDS = 1.0
_REHEARSAL_STEP_COUNT = 4
FramingMode = Literal["full_body", "upper_garment_complete"]


class StrictGuidanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalibrationTarget(StrictGuidanceModel):
    target_id: str = Field(pattern=r"^[a-z0-9_]+$")
    instruction: str = Field(min_length=1)
    quad_normalized: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]


class BoardObservation(StrictGuidanceModel):
    corner_count: int = Field(ge=0)
    quad_normalized: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    center_normalized: tuple[float, float]
    area_fraction: float = Field(gt=0.0)
    sharpness: float = Field(ge=0.0)


class GuidanceResult(StrictGuidanceModel):
    target_index: int = Field(ge=0)
    target_count: int = Field(gt=0)
    target_id: str
    instruction: str
    status: Literal[
        "board_not_found",
        "more_corners_needed",
        "match_green_outline",
        "hold_still",
        "accepted",
        "complete",
    ]
    target_error: float | None = Field(default=None, ge=0.0)
    stable_seconds: float = Field(ge=0.0)
    accepted: bool = False
    complete: bool = False


class RotationCueState(StrictGuidanceModel):
    phase: Literal["ready", "countdown", "hold", "turn", "complete"]
    direction: Literal["clockwise", "counter_clockwise"]
    target_angle_degrees: int = Field(ge=0, le=350, multiple_of=10)
    slot_index: int = Field(ge=0, le=35)
    seconds_remaining: float = Field(ge=0.0)
    angle_role: Literal["proposal_not_measurement"] = "proposal_not_measurement"


@dataclass(frozen=True)
class DetectedBoard:
    observation: BoardObservation
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    quad_pixels: np.ndarray


def _quad(
    top_left: tuple[float, float],
    top_right: tuple[float, float],
    bottom_right: tuple[float, float],
    bottom_left: tuple[float, float],
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return top_left, top_right, bottom_right, bottom_left


def guided_calibration_targets() -> tuple[CalibrationTarget, ...]:
    """Return the fixed visual poses used by the owner-operated calibration UI."""
    return (
        CalibrationTarget(
            target_id="center_medium",
            instruction="Center the board inside the green outline",
            quad_normalized=_quad((0.33, 0.28), (0.67, 0.28), (0.67, 0.72), (0.33, 0.72)),
        ),
        CalibrationTarget(
            target_id="upper_left",
            instruction="Move the complete board into the green outline",
            quad_normalized=_quad((0.10, 0.08), (0.37, 0.08), (0.37, 0.43), (0.10, 0.43)),
        ),
        CalibrationTarget(
            target_id="upper_right",
            instruction="Move the complete board into the green outline",
            quad_normalized=_quad((0.63, 0.08), (0.90, 0.08), (0.90, 0.43), (0.63, 0.43)),
        ),
        CalibrationTarget(
            target_id="lower_left",
            instruction="Move the complete board into the green outline",
            quad_normalized=_quad((0.10, 0.57), (0.37, 0.57), (0.37, 0.92), (0.10, 0.92)),
        ),
        CalibrationTarget(
            target_id="lower_right",
            instruction="Move the complete board into the green outline",
            quad_normalized=_quad((0.63, 0.57), (0.90, 0.57), (0.90, 0.92), (0.63, 0.92)),
        ),
        CalibrationTarget(
            target_id="center_close",
            instruction="Bring the phone closer and fill the green outline",
            quad_normalized=_quad((0.24, 0.17), (0.76, 0.17), (0.76, 0.83), (0.24, 0.83)),
        ),
        CalibrationTarget(
            target_id="center_far",
            instruction="Move back slightly; keep the whole board visible",
            quad_normalized=_quad((0.40, 0.37), (0.60, 0.37), (0.60, 0.63), (0.40, 0.63)),
        ),
        CalibrationTarget(
            target_id="perspective_left",
            instruction="Tilt until the detected outline matches the green trapezoid",
            quad_normalized=_quad((0.35, 0.38), (0.70, 0.27), (0.70, 0.73), (0.35, 0.62)),
        ),
        CalibrationTarget(
            target_id="perspective_right",
            instruction="Tilt until the detected outline matches the green trapezoid",
            quad_normalized=_quad((0.30, 0.27), (0.65, 0.38), (0.65, 0.62), (0.30, 0.73)),
        ),
        CalibrationTarget(
            target_id="perspective_up",
            instruction="Tilt until the detected outline matches the green trapezoid",
            quad_normalized=_quad((0.38, 0.30), (0.62, 0.30), (0.72, 0.72), (0.28, 0.72)),
        ),
        CalibrationTarget(
            target_id="perspective_down",
            instruction="Tilt until the detected outline matches the green trapezoid",
            quad_normalized=_quad((0.28, 0.28), (0.72, 0.28), (0.62, 0.70), (0.38, 0.70)),
        ),
        CalibrationTarget(
            target_id="rolled_final",
            instruction="Rotate in the image until both outlines agree",
            quad_normalized=_quad((0.38, 0.24), (0.70, 0.37), (0.62, 0.76), (0.30, 0.63)),
        ),
    )


def _dictionary() -> Any:
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)


def _board(spec: ControlledCharucoBoardSpec) -> Any:
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        _dictionary(),
    )


def _order_image_quad(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(4, 2)
    sums = values.sum(axis=1)
    differences = values[:, 1] - values[:, 0]
    ordered = np.stack(
        (
            values[int(np.argmin(sums))],
            values[int(np.argmin(differences))],
            values[int(np.argmax(sums))],
            values[int(np.argmax(differences))],
        )
    )
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("detected board quadrilateral is degenerate")
    return ordered


def detect_charuco_board(
    frame: np.ndarray,
    spec: ControlledCharucoBoardSpec,
) -> DetectedBoard | None:
    """Detect a board and derive its full projected outline without camera intrinsics."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("guided calibration frames must be BGR images")
    board = _board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(frame)
    if corners is None or ids is None or len(ids) < 4:
        return None
    objects, pixels = board.matchImagePoints(corners, ids)
    object_xy = np.asarray(objects, dtype=np.float64).reshape(-1, 3)[:, :2]
    image_xy = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(object_xy) < 4:
        return None
    homography, _ = cv2.findHomography(object_xy, image_xy, cv2.RANSAC, 2.0)
    if homography is None or not np.all(np.isfinite(homography)):
        return None
    width_m = spec.squares_x * spec.square_length_m
    height_m = spec.squares_y * spec.square_length_m
    board_corners = np.asarray(
        [[[0.0, 0.0], [width_m, 0.0], [width_m, height_m], [0.0, height_m]]],
        dtype=np.float64,
    )
    projected = cv2.perspectiveTransform(board_corners, homography).reshape(4, 2)
    try:
        quad_pixels = _order_image_quad(projected)
    except ValueError:
        return None
    height, width = frame.shape[:2]
    normalizer = np.asarray([width, height], dtype=np.float64)
    normalized = quad_pixels / normalizer
    center = normalized.mean(axis=0)
    area = abs(float(cv2.contourArea(quad_pixels.astype(np.float32)))) / float(width * height)
    if not np.isfinite(area) or area <= 0.0:
        return None
    x0, y0 = np.floor(quad_pixels.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad_pixels.max(axis=0)).astype(int)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[y0:y1, x0:x1]
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
    normalized_quad = (
        (float(normalized[0, 0]), float(normalized[0, 1])),
        (float(normalized[1, 0]), float(normalized[1, 1])),
        (float(normalized[2, 0]), float(normalized[2, 1])),
        (float(normalized[3, 0]), float(normalized[3, 1])),
    )
    observation = BoardObservation(
        corner_count=len(ids),
        quad_normalized=normalized_quad,
        center_normalized=(float(center[0]), float(center[1])),
        area_fraction=area,
        sharpness=sharpness,
    )
    return DetectedBoard(
        observation=observation,
        charuco_corners=np.asarray(corners, dtype=np.float32),
        charuco_ids=np.asarray(ids, dtype=np.int32),
        quad_pixels=quad_pixels,
    )


def _quad_error(observation: BoardObservation, target: CalibrationTarget) -> float:
    observed = np.asarray(observation.quad_normalized, dtype=np.float64)
    expected = np.asarray(target.quad_normalized, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((observed - expected) ** 2, axis=1))))


class GuidedCalibrationController:
    """State machine that accepts a pose only after visual match and stability."""

    def __init__(
        self,
        *,
        targets: tuple[CalibrationTarget, ...] | None = None,
        minimum_corners: int = _MINIMUM_GUIDANCE_CORNERS,
        stable_seconds: float = _DEFAULT_STABLE_SECONDS,
        target_tolerance: float = _DEFAULT_TARGET_TOLERANCE,
    ) -> None:
        self.targets = targets or guided_calibration_targets()
        if not self.targets:
            raise ValueError("guided calibration requires at least one target")
        if minimum_corners < 4 or stable_seconds <= 0.0 or target_tolerance <= 0.0:
            raise ValueError("guided calibration thresholds must be positive")
        self.minimum_corners = minimum_corners
        self.required_stable_seconds = stable_seconds
        self.target_tolerance = target_tolerance
        self.target_index = 0
        self._matching: deque[tuple[float, BoardObservation]] = deque()

    @property
    def complete(self) -> bool:
        return self.target_index >= len(self.targets)

    def reset(self) -> None:
        self.target_index = 0
        self._matching.clear()

    def update(
        self,
        timestamp_seconds: float,
        observation: BoardObservation | None,
    ) -> GuidanceResult:
        if self.complete:
            final = self.targets[-1]
            return GuidanceResult(
                target_index=len(self.targets),
                target_count=len(self.targets),
                target_id=final.target_id,
                instruction="Calibration image collection complete",
                status="complete",
                stable_seconds=self.required_stable_seconds,
                complete=True,
            )
        target = self.targets[self.target_index]
        base: dict[str, Any] = {
            "target_index": self.target_index,
            "target_count": len(self.targets),
            "target_id": target.target_id,
            "instruction": target.instruction,
        }
        if observation is None:
            self._matching.clear()
            return GuidanceResult(**base, status="board_not_found", stable_seconds=0.0)
        if observation.corner_count < self.minimum_corners:
            self._matching.clear()
            return GuidanceResult(
                **base,
                status="more_corners_needed",
                target_error=_quad_error(observation, target),
                stable_seconds=0.0,
            )
        error = _quad_error(observation, target)
        if error > self.target_tolerance:
            self._matching.clear()
            return GuidanceResult(
                **base,
                status="match_green_outline",
                target_error=error,
                stable_seconds=0.0,
            )
        if self._matching:
            previous_quad = np.asarray(self._matching[-1][1].quad_normalized, dtype=np.float64)
            current_quad = np.asarray(observation.quad_normalized, dtype=np.float64)
            frame_motion = float(
                np.sqrt(np.mean(np.sum((current_quad - previous_quad) ** 2, axis=1)))
            )
            if frame_motion > 0.012:
                self._matching.clear()
        self._matching.append((timestamp_seconds, observation))
        while self._matching and timestamp_seconds - self._matching[0][0] > (
            self.required_stable_seconds + 0.25
        ):
            self._matching.popleft()
        stable_for = max(0.0, timestamp_seconds - self._matching[0][0])
        if stable_for < self.required_stable_seconds:
            return GuidanceResult(
                **base,
                status="hold_still",
                target_error=error,
                stable_seconds=stable_for,
            )
        self.target_index += 1
        self._matching.clear()
        return GuidanceResult(
            **base,
            status="accepted",
            target_error=error,
            stable_seconds=stable_for,
            accepted=True,
            complete=self.complete,
        )


def render_guided_calibration_frame(
    frame: np.ndarray,
    result: GuidanceResult,
    detected: DetectedBoard | None,
    *,
    targets: tuple[CalibrationTarget, ...] | None = None,
    mirrored: bool = True,
) -> np.ndarray:
    """Render an unambiguous visual target; mirroring affects preview only."""
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    available = targets or guided_calibration_targets()
    target = available[min(result.target_index, len(available) - 1)]
    target_pixels = np.rint(
        np.asarray(target.quad_normalized, dtype=np.float64) * np.asarray([width, height])
    ).astype(np.int32)
    cv2.polylines(rendered, [target_pixels], True, (40, 220, 40), 6, cv2.LINE_AA)
    if detected is not None:
        observed_color = (
            (40, 220, 40) if result.status in {"hold_still", "accepted"} else (0, 190, 255)
        )
        cv2.polylines(
            rendered,
            [np.rint(detected.quad_pixels).astype(np.int32)],
            True,
            observed_color,
            5,
            cv2.LINE_AA,
        )
        cv2.aruco.drawDetectedCornersCharuco(
            rendered,
            detected.charuco_corners,
            detected.charuco_ids,
            (255, 80, 80),
        )
    cv2.rectangle(rendered, (0, 0), (width, 118), (16, 16, 16), -1)
    step = min(result.target_index + 1, result.target_count)
    heading = f"STEP {step}/{result.target_count}  {target.target_id.upper()}"
    status = result.status.replace("_", " ").upper()
    if detected is not None:
        status += f"  CORNERS {detected.observation.corner_count}/24"
    if result.status == "hold_still":
        remaining = max(0.0, _DEFAULT_STABLE_SECONDS - result.stable_seconds)
        status += f"  HOLD {remaining:.1f}s"
    cv2.putText(rendered, heading, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(
        rendered,
        target.instruction,
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (225, 225, 225),
        2,
    )
    cv2.putText(
        rendered,
        status,
        (24, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 255, 255),
        2,
    )
    cv2.putText(
        rendered,
        "Match outlines; accepted frames save automatically. Q/ESC stops.",
        (24, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    return cv2.flip(rendered, 1) if mirrored else rendered


def _open_capture(
    *,
    camera_index: int | None,
    replay_video: Path | None,
    width: int,
    height: int,
    fps: float,
) -> tuple[cv2.VideoCapture, Literal["live_camera", "replay_video"]]:
    if (camera_index is None) == (replay_video is None):
        raise ValueError("select exactly one live camera or replay video")
    if replay_video is not None:
        reject_sealed_capability([replay_video])
        if not replay_video.is_file():
            raise FileNotFoundError(f"guided-calibration replay is missing: {replay_video}")
        capture = cv2.VideoCapture(str(replay_video))
        mode: Literal["live_camera", "replay_video"] = "replay_video"
    else:
        if camera_index is None:
            raise RuntimeError("live-camera selection lost its camera index")
        capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        capture.set(cv2.CAP_PROP_FPS, float(fps))
        mode = "live_camera"
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("guided calibration could not open its selected video source")
    return capture, mode


def run_guided_calibration_capture(
    *,
    board_spec_path: Path,
    output_root: Path,
    camera_index: int | None = None,
    replay_video: Path | None = None,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    display: bool = True,
    maximum_frames: int | None = None,
) -> Path:
    """Collect write-once calibration stills with live visual detection and auto-gating."""
    if camera_index is not None and not display:
        raise ValueError("live guided calibration cannot disable its safety preview")
    reject_sealed_capability([board_spec_path, output_root])
    if output_root.exists():
        raise FileExistsError(f"guided calibration capture is immutable: {output_root}")
    partials = list(output_root.parent.glob(f".{output_root.name}.building-*"))
    if partials:
        raise FileExistsError("a prior guided-calibration partial must be audited separately")
    if not board_spec_path.is_file():
        raise FileNotFoundError(f"guided calibration board spec is missing: {board_spec_path}")
    spec_sha = sha256_file(board_spec_path)
    spec = ControlledCharucoBoardSpec.model_validate(read_json(board_spec_path))
    capture, source_mode = _open_capture(
        camera_index=camera_index,
        replay_video=replay_video,
        width=width,
        height=height,
        fps=fps,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    controller = GuidedCalibrationController()
    accepted: list[dict[str, Any]] = []
    frame_index = 0
    start = time.monotonic()
    stopped_by_owner = False
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = fps
    try:
        if display:
            cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_WINDOW_NAME, 1280, 720)
        while maximum_frames is None or frame_index < maximum_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = (
                frame_index / source_fps
                if source_mode == "replay_video"
                else time.monotonic() - start
            )
            detected = detect_charuco_board(frame, spec)
            result = controller.update(
                timestamp,
                None if detected is None else detected.observation,
            )
            if result.accepted and detected is not None:
                image_path = stage / f"intrinsics_{len(accepted):02d}.png"
                if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError("guided calibration could not encode an accepted PNG")
                accepted.append(
                    {
                        "target_id": result.target_id,
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "path": str(output_root / image_path.name),
                        "sha256": sha256_file(image_path),
                        "observation": detected.observation.model_dump(mode="json"),
                    }
                )
            if display:
                preview = render_guided_calibration_frame(frame, result, detected)
                cv2.imshow(_WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q"), ord("Q")}:
                    stopped_by_owner = True
                    break
                if cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1.0:
                    stopped_by_owner = True
                    break
            frame_index += 1
            if controller.complete:
                break
    except KeyboardInterrupt:
        stopped_by_owner = True
    finally:
        capture.release()
        if display:
            with suppress(cv2.error):
                cv2.destroyWindow(_WINDOW_NAME)
    status = "complete_candidate_images" if controller.complete else "incomplete_not_evidence"
    manifest = {
        "schema_version": "frayid_v3_guided_calibration_capture.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "scientific_status": "acquisition_only_requires_registered_calibration_solver",
        "source_mode": source_mode,
        "camera_index": camera_index,
        "replay_video": str(replay_video) if replay_video is not None else None,
        "board_spec_path": str(board_spec_path),
        "board_spec_sha256": spec_sha,
        "target_count": len(controller.targets),
        "accepted_count": len(accepted),
        "accepted_images": accepted,
        "decoded_frame_count": frame_index,
        "stopped_by_owner": stopped_by_owner,
        "preview_mirrored": True,
        "saved_images_mirrored": False,
        "minimum_corners": controller.minimum_corners,
        "stable_seconds": controller.required_stable_seconds,
        "target_tolerance_normalized_rms": controller.target_tolerance,
        "audio_captured": False,
        "metric_scale_claimed": False,
        "evaluator_files_read": 0,
        "historical_development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_geometry_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
    }
    write_json(stage / "guided_calibration_manifest.json", manifest)
    os.rename(stage, output_root)
    return output_root / "guided_calibration_manifest.json"


def render_training_camera_preview(
    frame: np.ndarray,
    *,
    mirrored: bool = True,
    framing: FramingMode = "full_body",
) -> np.ndarray:
    """Render a framing guide without recording or analyzing identity."""
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    if framing == "upper_garment_complete":
        left, right = int(width * 0.08), int(width * 0.92)
        top, bottom = int(height * 0.06), int(height * 0.94)
        instruction = "Keep neck, both armholes, and the entire hem visible"
    else:
        left, right = int(width * 0.16), int(width * 0.84)
        top, bottom = int(height * 0.06), int(height * 0.96)
        instruction = "Place your full body inside the green frame"
    cv2.rectangle(rendered, (left, top), (right, bottom), (40, 220, 40), 5, cv2.LINE_AA)
    cv2.line(rendered, (width // 2, top), (width // 2, bottom), (80, 180, 255), 2)
    cv2.line(rendered, (left, bottom), (right, bottom), (80, 180, 255), 4)
    if framing == "upper_garment_complete":
        shoulder_y = int(height * 0.24)
        hem_y = int(height * 0.82)
        cv2.line(rendered, (left, shoulder_y), (right, shoulder_y), (80, 180, 255), 2)
        cv2.line(rendered, (left, hem_y), (right, hem_y), (80, 180, 255), 2)
        cv2.putText(
            rendered,
            "NECK + SHOULDERS",
            (left + 12, shoulder_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 180, 255),
            2,
        )
        cv2.putText(
            rendered,
            "ENTIRE HEM ABOVE THIS LINE",
            (left + 12, hem_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 180, 255),
            2,
        )
    cv2.rectangle(rendered, (0, 0), (width, 92), (16, 16, 16), -1)
    cv2.putText(
        rendered,
        "LIVE PREVIEW ONLY - NOTHING IS SAVED",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        rendered,
        f"{instruction}. Q/ESC closes preview.",
        (24, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (225, 225, 225),
        2,
    )
    return cv2.flip(rendered, 1) if mirrored else rendered


def run_training_camera_preview(
    *,
    camera_index: int,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    framing: FramingMode = "full_body",
) -> None:
    """Open an always-on-top, mirrored, zero-save preview using AVFoundation."""
    ffplay_bin = shutil.which("ffplay")
    if ffplay_bin is None:
        raise FileNotFoundError("training camera preview requires ffplay")
    command = ffplay_training_camera_preview_command(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        framing=framing,
        ffplay_bin=ffplay_bin,
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(command)
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"training camera preview failed with status {returncode}")
    except KeyboardInterrupt:
        _stop_process(process)
    finally:
        _stop_process(process)


def _ffmpeg_preview_guide(framing: FramingMode) -> str:
    if framing == "upper_garment_complete":
        return (
            "drawbox=x=iw*0.08:y=ih*0.06:w=iw*0.84:h=ih*0.88:color=green@0.9:t=5,"
            "drawbox=x=iw*0.08:y=ih*0.24:w=iw*0.84:h=2:color=orange@0.9:t=fill,"
            "drawbox=x=iw*0.08:y=ih*0.82:w=iw*0.84:h=3:color=orange@0.9:t=fill"
        )
    return "drawbox=x=iw*0.16:y=ih*0.06:w=iw*0.68:h=ih*0.90:color=green@0.9:t=5"


def ffplay_training_camera_preview_command(
    *,
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    framing: FramingMode,
    ffplay_bin: str = "ffplay",
) -> list[str]:
    """Build a zero-save AVFoundation preview command."""
    if width < 640 or height < 480 or fps <= 0.0:
        raise ValueError("preview dimensions and frame rate must be positive")
    preview_filter = f"hflip,scale=1280:720,{_ffmpeg_preview_guide(framing)}"
    return [
        ffplay_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "avfoundation",
        "-framerate",
        f"{fps:g}",
        "-video_size",
        f"{width}x{height}",
        "-pixel_format",
        "uyvy422",
        "-i",
        f"{camera_index}:none",
        "-vf",
        preview_filter,
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-alwaysontop",
        "-window_title",
        "FrayID LIVE UPPER-GARMENT PREVIEW - NOTHING SAVED - Q CLOSES",
    ]


def _rotation_angles(
    direction: Literal["clockwise", "counter_clockwise"],
) -> tuple[int, ...]:
    if direction == "clockwise":
        return tuple(range(0, 360, 10))
    return (0, *range(350, 0, -10))


def rotation_cue_state(
    elapsed_seconds: float | None,
    direction: Literal["clockwise", "counter_clockwise"],
) -> RotationCueState:
    """Map recorder time to a visible/audible proposal; it never measures pose."""
    angles = _rotation_angles(direction)
    if elapsed_seconds is None:
        return RotationCueState(
            phase="ready",
            direction=direction,
            target_angle_degrees=angles[0],
            slot_index=0,
            seconds_remaining=0.0,
        )
    elapsed = max(0.0, elapsed_seconds)
    if elapsed < _ROTATION_PRE_ROLL_SECONDS:
        return RotationCueState(
            phase="countdown",
            direction=direction,
            target_angle_degrees=angles[0],
            slot_index=0,
            seconds_remaining=_ROTATION_PRE_ROLL_SECONDS - elapsed,
        )
    rotation_elapsed = elapsed - _ROTATION_PRE_ROLL_SECONDS
    slot_index = min(int(rotation_elapsed // _ROTATION_HOLD_PERIOD_SECONDS), 35)
    if rotation_elapsed >= len(angles) * _ROTATION_HOLD_PERIOD_SECONDS:
        return RotationCueState(
            phase="complete",
            direction=direction,
            target_angle_degrees=angles[-1],
            slot_index=35,
            seconds_remaining=0.0,
        )
    in_slot = rotation_elapsed - slot_index * _ROTATION_HOLD_PERIOD_SECONDS
    if in_slot < _ROTATION_STABLE_SECONDS:
        phase: Literal["hold", "turn"] = "hold"
        remaining = _ROTATION_STABLE_SECONDS - in_slot
    else:
        phase = "turn"
        remaining = _ROTATION_HOLD_PERIOD_SECONDS - in_slot
    return RotationCueState(
        phase=phase,
        direction=direction,
        target_angle_degrees=angles[slot_index],
        slot_index=slot_index,
        seconds_remaining=max(0.0, remaining),
    )


def render_guided_rotation_frame(
    frame: np.ndarray,
    state: RotationCueState,
    *,
    recording: bool,
    mirrored: bool = True,
    framing: FramingMode = "full_body",
) -> np.ndarray:
    """Overlay live framing and cue state on a preview copy, never on saved pixels."""
    rendered = render_training_camera_preview(frame, mirrored=False, framing=framing)
    height, width = rendered.shape[:2]
    cv2.rectangle(rendered, (0, 0), (width, 118), (16, 16, 16), -1)
    if state.phase == "ready":
        primary = "LIVE VIEW - PRESS SPACE WHEN FRAMING IS CORRECT"
        secondary = "Nothing is being recorded"
    elif state.phase == "countdown":
        primary = f"REC  START POSITION  {state.seconds_remaining:04.1f}s"
        secondary = "Walk to the center mark and face the camera"
    elif state.phase == "hold":
        primary = f"REC  {state.target_angle_degrees:03d} DEG  HOLD STILL"
        secondary = f"Stable interval {state.seconds_remaining:03.1f}s"
    elif state.phase == "turn":
        next_index = min(state.slot_index + 1, 35)
        next_angle = _rotation_angles(state.direction)[next_index]
        primary = f"REC  TURN TO {next_angle:03d} DEG"
        secondary = f"Move during the next {state.seconds_remaining:03.1f}s"
    else:
        primary = "CAPTURE COMPLETE"
        secondary = "Remain still until the window closes"
    color = (40, 40, 255) if recording else (255, 255, 255)
    cv2.putText(rendered, primary, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(
        rendered,
        secondary,
        (24, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (225, 225, 225),
        2,
    )
    cv2.putText(
        rendered,
        "Target angles are cues, not measured pose. Q/ESC stops immediately.",
        (24, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    return cv2.flip(rendered, 1) if mirrored else rendered


def _speak_rotation_transition(
    state: RotationCueState,
    previous: tuple[str, int] | None,
    previous_process: subprocess.Popen[bytes] | None,
) -> tuple[tuple[str, int], subprocess.Popen[bytes] | None]:
    """Speak one short cue, stopping any prior cue before it can overlap."""
    current = (state.phase, state.slot_index)
    if current == previous or state.phase not in {"countdown", "hold", "turn", "complete"}:
        return current, previous_process
    phrase = rotation_spoken_phrase(state)
    _stop_process(previous_process)
    process: subprocess.Popen[bytes] | None = None
    with suppress(OSError):
        process = subprocess.Popen(
            ["say", "-v", "Tingting", "-r", "210", phrase],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return current, process


def rotation_spoken_phrase(state: RotationCueState) -> str:
    """Return a usable motion cue without pretending the owner can measure angles."""
    if state.phase == "countdown":
        return "开始。十五秒准备。不用对准角度。"
    if state.phase == "turn":
        side = "右手" if state.direction == "clockwise" else "左手"
        return f"慢慢向你的{side}方向转一小步。"
    if state.phase == "complete":
        return "完成一圈。请面对摄像头。"
    if state.phase == "ready":
        return "准备。"
    landmarks = {
        0: "正面",
        9: "大约四分之一圈",
        18: "大约半圈",
        27: "大约四分之三圈",
    }
    landmark = landmarks.get(state.slot_index)
    if landmark is not None:
        return f"停。{landmark}。保持三秒。"
    return "停。保持三秒。"


def run_guided_rotation_rehearsal(
    *,
    direction: Literal["clockwise", "counter_clockwise"],
    camera_index: int,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    framing: FramingMode = "upper_garment_complete",
) -> None:
    """Rehearse four slow steps with a zero-save, always-on-top camera preview."""
    ffplay_bin = shutil.which("ffplay")
    if ffplay_bin is None:
        raise FileNotFoundError("guided rotation rehearsal requires ffplay")
    preview_command = ffplay_training_camera_preview_command(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        framing=framing,
        ffplay_bin=ffplay_bin,
    )
    preview_process: subprocess.Popen[bytes] | None = None
    speech_process: subprocess.Popen[bytes] | None = None
    previous_state: tuple[str, int] | None = None
    try:
        preview_process = subprocess.Popen(preview_command)
        rehearsal_start = time.monotonic()
        while time.monotonic() - rehearsal_start < 5.0:
            if preview_process.poll() is not None:
                return
            time.sleep(0.05)
        for slot_index in range(_REHEARSAL_STEP_COUNT):
            hold_elapsed = _ROTATION_PRE_ROLL_SECONDS + (slot_index * _ROTATION_HOLD_PERIOD_SECONDS)
            hold_state = rotation_cue_state(hold_elapsed, direction)
            previous_state, speech_process = _speak_rotation_transition(
                hold_state,
                previous_state,
                speech_process,
            )
            hold_start = time.monotonic()
            while time.monotonic() - hold_start < _ROTATION_STABLE_SECONDS:
                if preview_process.poll() is not None:
                    return
                time.sleep(0.05)
            turn_state = rotation_cue_state(
                hold_elapsed + _ROTATION_STABLE_SECONDS,
                direction,
            )
            previous_state, speech_process = _speak_rotation_transition(
                turn_state,
                previous_state,
                speech_process,
            )
            turn_start = time.monotonic()
            turn_seconds = _ROTATION_HOLD_PERIOD_SECONDS - _ROTATION_STABLE_SECONDS
            while time.monotonic() - turn_start < turn_seconds:
                if preview_process.poll() is not None:
                    return
                time.sleep(0.05)
        _stop_process(speech_process)
        speech_process = subprocess.Popen(
            ["say", "-v", "Tingting", "-r", "210", "排练结束。请回到正面。"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        speech_process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        _stop_process(speech_process)
        _stop_process(preview_process)


def _rotation_hold_coverage(
    timestamps: list[float],
    *,
    fps: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    array = np.asarray(timestamps, dtype=np.float64)
    reports: list[dict[str, Any]] = []
    blockers: list[str] = []
    required = max(1, int(np.floor(_ROTATION_STABLE_SECONDS * fps * 0.8)))
    for index in range(36):
        start = _ROTATION_PRE_ROLL_SECONDS + index * _ROTATION_HOLD_PERIOD_SECONDS
        end = start + _ROTATION_STABLE_SECONDS
        frame_indices = np.flatnonzero((array >= start) & (array <= end))
        if len(frame_indices) < required:
            blockers.append(f"stable_hold_frame_coverage:{index:02d}")
        reports.append(
            {
                "slot_index": index,
                "stable_start_seconds": start,
                "stable_end_seconds": end,
                "first_frame_index": int(frame_indices[0]) if len(frame_indices) else None,
                "last_frame_index": int(frame_indices[-1]) if len(frame_indices) else None,
                "frame_count": len(frame_indices),
                "minimum_required_frame_count": required,
            }
        )
    return reports, blockers


def ffmpeg_rotation_capture_commands(
    *,
    video_path: Path,
    direction: Literal["clockwise", "counter_clockwise"],
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    framing: FramingMode,
    duration_seconds: float,
    ffmpeg_bin: str = "ffmpeg",
    ffplay_bin: str = "ffplay",
) -> tuple[list[str], list[str]]:
    """Build the one-source lossless-recording and always-on-top preview pipeline."""
    if width < 640 or height < 480 or fps <= 0.0 or duration_seconds <= 0.0:
        raise ValueError("rotation capture dimensions, rate, and duration must be positive")
    guide = _ffmpeg_preview_guide(framing)
    filter_graph = (
        f"[0:v]split=2[record][preview];"
        f"[record]format=yuv422p[record_out];"
        f"[preview]hflip,scale=1280:720,{guide}[preview_out]"
    )
    ffmpeg = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-f",
        "avfoundation",
        "-framerate",
        f"{fps:g}",
        "-video_size",
        f"{width}x{height}",
        "-pixel_format",
        "uyvy422",
        "-i",
        f"{camera_index}:none",
        "-filter_complex",
        filter_graph,
        "-map",
        "[record_out]",
        "-an",
        "-t",
        f"{duration_seconds:g}",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-slicecrc",
        "1",
        "-fps_mode",
        "passthrough",
        str(video_path),
        "-map",
        "[preview_out]",
        "-an",
        "-t",
        f"{duration_seconds:g}",
        "-c:v",
        "mjpeg",
        "-q:v",
        "6",
        "-f",
        "nut",
        "pipe:1",
    ]
    ffplay = [
        ffplay_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-alwaysontop",
        "-window_title",
        f"FrayID {direction} - LIVE UPPER-GARMENT VIEW - Q STOPS",
        "-autoexit",
        "-i",
        "pipe:0",
    ]
    return ffmpeg, ffplay


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def run_guided_rotation_capture(
    *,
    output_root: Path,
    direction: Literal["clockwise", "counter_clockwise"],
    camera_index: int,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    speak_cues: bool = True,
    framing: FramingMode = "full_body",
) -> Path:
    """Record losslessly with FFmpeg while ffplay shows a low-cost mirrored live view."""
    reject_sealed_capability([output_root])
    if output_root.exists():
        raise FileExistsError(f"guided rotation capture is immutable: {output_root}")
    partials = list(output_root.parent.glob(f".{output_root.name}.building-*"))
    if partials:
        raise FileExistsError("a prior guided-rotation partial must be audited separately")
    ffmpeg_bin = shutil.which("ffmpeg")
    ffplay_bin = shutil.which("ffplay")
    if ffmpeg_bin is None or ffplay_bin is None:
        raise FileNotFoundError("guided rotation requires both ffmpeg and ffplay")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    video_stage = stage / f"{direction}.mkv"
    runtime_log = stage / "capture_runtime.log"
    stopped_by_owner = False
    previous_spoken_state: tuple[str, int] | None = None
    speech_process: subprocess.Popen[bytes] | None = None
    total_seconds = (
        _ROTATION_PRE_ROLL_SECONDS + 36 * _ROTATION_HOLD_PERIOD_SECONDS + _ROTATION_TAIL_SECONDS
    )
    ffmpeg_command, ffplay_command = ffmpeg_rotation_capture_commands(
        video_path=video_stage,
        direction=direction,
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        framing=framing,
        duration_seconds=total_seconds,
        ffmpeg_bin=ffmpeg_bin,
        ffplay_bin=ffplay_bin,
    )
    ffmpeg_process: subprocess.Popen[bytes] | None = None
    ffplay_process: subprocess.Popen[bytes] | None = None
    ffmpeg_returncode: int | None = None
    ffplay_returncode: int | None = None
    started_at = time.monotonic()
    with runtime_log.open("wb") as log:
        try:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=log,
            )
            if ffmpeg_process.stdout is None:
                raise RuntimeError("FFmpeg preview pipe was not created")
            ffplay_process = subprocess.Popen(
                ffplay_command,
                stdin=ffmpeg_process.stdout,
                stdout=subprocess.DEVNULL,
                stderr=log,
            )
            ffmpeg_process.stdout.close()
            while ffmpeg_process.poll() is None:
                if ffplay_process.poll() is not None:
                    stopped_by_owner = True
                    _stop_process(ffmpeg_process)
                    break
                elapsed = time.monotonic() - started_at
                if speak_cues:
                    state = rotation_cue_state(elapsed, direction)
                    previous_spoken_state, speech_process = _speak_rotation_transition(
                        state,
                        previous_spoken_state,
                        speech_process,
                    )
                time.sleep(0.05)
            ffmpeg_returncode = ffmpeg_process.wait()
        except KeyboardInterrupt:
            stopped_by_owner = True
            _stop_process(ffmpeg_process)
        finally:
            _stop_process(speech_process)
            _stop_process(ffplay_process)
            if ffmpeg_process is not None:
                ffmpeg_returncode = ffmpeg_process.poll()
            if ffplay_process is not None:
                ffplay_returncode = ffplay_process.poll()
    blockers: list[str] = []
    hold_reports: list[dict[str, Any]] = []
    timestamps: list[float] = []
    probe_payload: dict[str, Any] | None = None
    timestamp_summary: dict[str, Any] | None = None
    probe_provenance: dict[str, Any] | None = None
    if not video_stage.is_file() or video_stage.stat().st_size == 0:
        blockers.append("capture_did_not_create_video")
    else:
        probe, frame_timestamps, probe_provenance = probe_video_forensics(video_stage)
        selected = [
            item.selected_timestamp_seconds
            for item in frame_timestamps
            if item.selected_timestamp_seconds is not None
        ]
        if selected:
            origin = selected[0]
            timestamps = [value - origin for value in selected]
        probe_payload = probe.model_dump(mode="json")
        timestamp_summary = summarize_timestamps(frame_timestamps)
        hold_reports, hold_blockers = _rotation_hold_coverage(timestamps, fps=fps)
        blockers.extend(hold_blockers)
        if probe.codec != "ffv1":
            blockers.append("capture_codec_is_not_ffv1")
        if (probe.width, probe.height) != (width, height):
            blockers.append("capture_resolution_mismatch")
        if not timestamps or timestamps[-1] < total_seconds - 0.5:
            blockers.append("capture_ended_before_registered_duration")
        if len(timestamps) >= 2:
            deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
            if np.any(deltas <= 0.0):
                blockers.append("capture_timestamps_not_strictly_monotonic")
            if float(np.max(deltas)) > 0.1:
                blockers.append("capture_timestamp_gap_above_100ms")
    if ffmpeg_returncode not in {0, None}:
        blockers.append("ffmpeg_capture_process_failed")
    manifest = {
        "schema_version": "frayid_v3_guided_rotation_capture.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "complete_candidate_clip" if not blockers else "incomplete_not_evidence",
        "scientific_status": "acquisition_only_requires_v01_source_audit",
        "direction": direction,
        "framing": framing,
        "target_scope": (
            "neck_both_armholes_and_entire_hem"
            if framing == "upper_garment_complete"
            else "full_body"
        ),
        "camera_index": camera_index,
        "requested_resolution": [width, height],
        "observed_resolution": (
            [probe_payload["width"], probe_payload["height"]] if probe_payload is not None else None
        ),
        "requested_fps": fps,
        "codec": "ffv1",
        "video_path": str(output_root / video_stage.name) if video_stage.is_file() else None,
        "video_sha256": sha256_file(video_stage) if video_stage.is_file() else None,
        "video_size_bytes": video_stage.stat().st_size if video_stage.is_file() else None,
        "frame_count": len(timestamps),
        "frame_monotonic_elapsed_seconds": timestamps,
        "timestamp_summary": timestamp_summary,
        "video_probe": probe_payload,
        "probe_provenance": probe_provenance,
        "hold_reports": hold_reports,
        "blockers": blockers,
        "stopped_by_owner": stopped_by_owner,
        "preview_visible_during_entire_recording": True,
        "preview_window_requested_always_on_top": True,
        "recording_starts_before_ten_second_positioning_countdown": True,
        "preview_mirrored": True,
        "saved_video_mirrored": False,
        "preview_overlay_written_to_video": False,
        "angle_role": "proposal_not_measurement",
        "angle_measurement_claimed": False,
        "audio_captured": False,
        "spoken_cues_recorded_as_evidence": False,
        "captured_camera_frames_encoded_losslessly": True,
        "native_camera_container_bytes_available": False,
        "evaluator_files_read": 0,
        "historical_development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_geometry_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "runtime_log_path": str(output_root / runtime_log.name),
        "runtime_log_sha256": sha256_file(runtime_log),
        "ffmpeg_returncode": ffmpeg_returncode,
        "ffplay_returncode": ffplay_returncode,
        "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
    }
    write_json(stage / "guided_rotation_manifest.json", manifest)
    os.rename(stage, output_root)
    return output_root / "guided_rotation_manifest.json"


def create_guided_display_board_kit(board_spec_path: Path, output_root: Path) -> Path:
    """Create a write-once, offline phone page whose scale cannot change after locking."""
    reject_sealed_capability([board_spec_path, output_root])
    if output_root.exists():
        raise FileExistsError(f"guided display-board kit is immutable: {output_root}")
    partials = list(output_root.parent.glob(f".{output_root.name}.building-*"))
    if partials:
        raise FileExistsError("a prior guided display-board partial must be audited separately")
    if not board_spec_path.is_file():
        raise FileNotFoundError(f"guided display-board spec is missing: {board_spec_path}")
    spec = ControlledCharucoBoardSpec.model_validate(read_json(board_spec_path))
    board_image = Path(spec.board_image_path)
    if not board_image.is_file() or sha256_file(board_image) != spec.board_image_sha256:
        raise ValueError("guided display-board image does not match its bound specification")
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>FrayID locked calibration board</title>
<style>
html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#777;color:#fff;font-family:-apple-system,sans-serif}
body{display:grid;place-items:center}.panel{position:fixed;inset:0;display:grid;place-items:center;background:#151515;z-index:3}
.card{max-width:34rem;padding:1.4rem;text-align:center}.card h1{font-size:1.35rem}.card p{line-height:1.5}
button{font:inherit;font-weight:700;padding:.9rem 1.2rem;border:0;border-radius:.7rem;background:#37d25a;color:#07140a}
#board{display:none;width:min(94vw,calc(88vh * 1.4));height:auto;aspect-ratio:7/5;object-fit:contain;touch-action:none}
#warning{position:fixed;bottom:.4rem;left:0;right:0;text-align:center;font-size:.72rem;color:#fff;background:#222b;padding:.3rem}
body.locked #setup{display:none}body.locked #board{display:block}
</style>
</head>
<body>
<div id="setup" class="panel"><div class="card"><h1>先锁定横屏, 再测量</h1>
<p>点击后不要旋转手机、缩放页面或隐藏/显示浏览器工具栏。进入标定画面后, 再实测三个方格边长。</p>
<button id="lock">进入横屏标定画面</button></div></div>
<img id="board" src="controlled_charuco_board.png" alt="ChArUco calibration board" draggable="false">
<div id="warning">LOCKED LANDSCAPE - DO NOT ROTATE OR PINCH ZOOM</div>
<script>
document.getElementById('lock').addEventListener('click',async()=>{
  try{if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen()}catch(e){}
  try{if(screen.orientation&&screen.orientation.lock)await screen.orientation.lock('landscape')}catch(e){}
  document.body.classList.add('locked');
});
document.addEventListener('gesturestart',e=>e.preventDefault(),{passive:false});
document.addEventListener('touchmove',e=>{if(e.touches.length>1)e.preventDefault()},{passive:false});
</script>
</body>
</html>
"""
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    image_stage = stage / "controlled_charuco_board.png"
    image_stage.write_bytes(board_image.read_bytes())
    html_stage = stage / "display_board.html"
    html_stage.write_text(html, encoding="utf-8")
    manifest = {
        "schema_version": "frayid_v3_guided_display_board.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "planning_only_not_evidence",
        "board_spec_path": str(board_spec_path),
        "board_spec_sha256": sha256_file(board_spec_path),
        "board_image_path": str(output_root / image_stage.name),
        "board_image_sha256": sha256_file(image_stage),
        "html_path": str(output_root / html_stage.name),
        "html_sha256": sha256_file(html_stage),
        "orientation_lock_requested": "landscape",
        "pinch_zoom_disabled": True,
        "measurement_timing": "after_fullscreen_and_orientation_lock",
        "network_dependencies": [],
        "scientific_result_claimed": False,
    }
    write_json(stage / "display_board_manifest.json", manifest)
    os.rename(stage, output_root)
    return output_root / "display_board_manifest.json"


__all__ = [
    "BoardObservation",
    "CalibrationTarget",
    "DetectedBoard",
    "GuidanceResult",
    "GuidedCalibrationController",
    "RotationCueState",
    "create_guided_display_board_kit",
    "detect_charuco_board",
    "ffmpeg_rotation_capture_commands",
    "ffplay_training_camera_preview_command",
    "guided_calibration_targets",
    "render_guided_calibration_frame",
    "render_guided_rotation_frame",
    "render_training_camera_preview",
    "rotation_cue_state",
    "rotation_spoken_phrase",
    "run_guided_calibration_capture",
    "run_guided_rotation_capture",
    "run_guided_rotation_rehearsal",
    "run_training_camera_preview",
]
