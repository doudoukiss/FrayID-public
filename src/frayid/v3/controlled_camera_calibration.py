from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v3.controlled_capture import (
    CalibratedIntrinsics,
    DirectionSynchronization,
    EvaluatorStereoCalibration,
    TrainingCameraCalibration,
)

EXPERIMENT_ID: Literal["postv3_v01_controlled_recapture_evidence_master_r01"] = (
    "postv3_v01_controlled_recapture_evidence_master_r01"
)
_DICTIONARY_NAME: Literal["DICT_5X5_1000"] = "DICT_5X5_1000"
_MINIMUM_CORNERS = 8
_MAXIMUM_INTRINSIC_RMS_PIXELS = 1.0
_MAXIMUM_VIEW_RMS_PIXELS = 1.5
_MINIMUM_NORMAL_SPAN_DEGREES = 15.0
_MAXIMUM_STEREO_RMS_PIXELS = 1.5
_MAXIMUM_SETUP_RMS_PIXELS = 1.5


class StrictCalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ControlledCharucoBoardSpec(StrictCalibrationModel):
    schema_version: Literal["frayid_v3_controlled_charuco_board.v1"] = (
        "frayid_v3_controlled_charuco_board.v1"
    )
    dictionary: Literal["DICT_5X5_1000"] = _DICTIONARY_NAME
    squares_x: int = Field(ge=5)
    squares_y: int = Field(ge=5)
    square_length_m: float = Field(gt=0.0)
    marker_length_m: float = Field(gt=0.0)
    printed_width_m: float = Field(gt=0.0)
    printed_height_m: float = Field(gt=0.0)
    image_width_pixels: int = Field(gt=0)
    image_height_pixels: int = Field(gt=0)
    border_bits: int = Field(ge=1)
    layout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    board_image_path: str = Field(min_length=1)
    board_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _physical_layout_is_consistent(self) -> ControlledCharucoBoardSpec:
        if self.marker_length_m >= self.square_length_m:
            raise ValueError("ChArUco marker length must be smaller than square length")
        if not np.isclose(
            self.printed_width_m,
            self.squares_x * self.square_length_m,
            atol=1.0e-12,
            rtol=0.0,
        ) or not np.isclose(
            self.printed_height_m,
            self.squares_y * self.square_length_m,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("printed board dimensions do not match the metric layout")
        if self.layout_sha256 != _canonical_sha256(_layout_payload(self)):
            raise ValueError("ChArUco layout SHA-256 does not match its parameters")
        return self


class StereoImagePair(StrictCalibrationModel):
    training_image: str = Field(min_length=1)
    evaluator_image: str = Field(min_length=1)


class SynchronizationEvents(StrictCalibrationModel):
    direction: Literal["clockwise", "counter_clockwise"]
    method: Literal["hardware_trigger", "shared_timecode", "audible_visual_sync_event"]
    training_event_seconds: list[float] = Field(min_length=3)
    evaluator_event_seconds: list[float] = Field(min_length=3)

    @model_validator(mode="after")
    def _events_match_and_increase(self) -> SynchronizationEvents:
        if len(self.training_event_seconds) != len(self.evaluator_event_seconds):
            raise ValueError("synchronization event lists must have equal length")
        for values in (self.training_event_seconds, self.evaluator_event_seconds):
            array = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(array)) or np.any(np.diff(array) <= 0.0):
                raise ValueError("synchronization event times must be finite and increasing")
        return self


class ControlledCalibrationSession(StrictCalibrationModel):
    schema_version: Literal["frayid_v3_controlled_calibration_session.v1"] = (
        "frayid_v3_controlled_calibration_session.v1"
    )
    experiment_id: Literal["postv3_v01_controlled_recapture_evidence_master_r01"] = EXPERIMENT_ID
    board_spec_path: str = Field(min_length=1)
    board_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_only: Literal[False] = False
    printed_square_measurements_m: list[float] = Field(min_length=3)
    training_intrinsic_images: list[str] = Field(min_length=10)
    evaluator_intrinsic_images: list[str] = Field(min_length=10)
    stereo_pairs: list[StereoImagePair] = Field(min_length=10)
    training_setup_image: str = Field(min_length=1)
    rotation_axis_origin_board_m: tuple[float, float, float]
    floor_board_verified_level: Literal[True] = True
    rotation_axis_mark_verified: Literal[True] = True
    vertical_axis_board_normal_sign: Literal[-1, 1]
    synchronization: list[SynchronizationEvents]

    @model_validator(mode="after")
    def _session_is_complete(self) -> ControlledCalibrationSession:
        directions = [item.direction for item in self.synchronization]
        if len(directions) != 2 or set(directions) != {"clockwise", "counter_clockwise"}:
            raise ValueError("calibration session requires synchronization for both directions")
        origin = np.asarray(self.rotation_axis_origin_board_m, dtype=np.float64)
        if not np.all(np.isfinite(origin)) or not np.isclose(origin[2], 0.0, atol=1.0e-9, rtol=0.0):
            raise ValueError("rotation-axis origin must lie on the measured floor board")
        measurements = np.asarray(self.printed_square_measurements_m, dtype=np.float64)
        if not np.all(np.isfinite(measurements)) or np.any(measurements <= 0.0):
            raise ValueError("printed ChArUco square measurements must be finite and positive")
        return self


class ControlledTrainingCalibrationSession(StrictCalibrationModel):
    """One-camera metric calibration session with no evaluator capability."""

    schema_version: Literal["frayid_v3_controlled_training_calibration_session.v1"] = (
        "frayid_v3_controlled_training_calibration_session.v1"
    )
    experiment_id: Literal["postv3_v01_controlled_recapture_evidence_master_r01"] = EXPERIMENT_ID
    capture_mode: Literal["single_camera_evidence_consistent"] = "single_camera_evidence_consistent"
    board_spec_path: str = Field(min_length=1)
    board_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_only: Literal[False] = False
    printed_square_measurements_m: list[float] = Field(min_length=3)
    training_intrinsic_images: list[str] = Field(min_length=10)
    training_setup_image: str = Field(min_length=1)
    rotation_axis_origin_board_m: tuple[float, float, float]
    floor_board_verified_level: Literal[True] = True
    rotation_axis_mark_verified: Literal[True] = True
    vertical_axis_board_normal_sign: Literal[-1, 1]
    independent_evaluator_available: Literal[False] = False
    metric_accuracy_claim_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _session_is_complete(self) -> ControlledTrainingCalibrationSession:
        origin = np.asarray(self.rotation_axis_origin_board_m, dtype=np.float64)
        if not np.all(np.isfinite(origin)) or not np.isclose(origin[2], 0.0, atol=1.0e-9, rtol=0.0):
            raise ValueError("rotation-axis origin must lie on the measured floor board")
        measurements = np.asarray(self.printed_square_measurements_m, dtype=np.float64)
        if not np.all(np.isfinite(measurements)) or np.any(measurements <= 0.0):
            raise ValueError("printed ChArUco square measurements must be finite and positive")
        return self


def _layout_payload(spec: ControlledCharucoBoardSpec | dict[str, Any]) -> dict[str, Any]:
    read = spec if isinstance(spec, dict) else spec.model_dump(mode="json")
    return {
        "dictionary": read["dictionary"],
        "squares_x": read["squares_x"],
        "squares_y": read["squares_y"],
        "square_length_m": read["square_length_m"],
        "marker_length_m": read["marker_length_m"],
    }


def _dictionary() -> Any:
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)


def _opencv_runtime() -> dict[str, Any]:
    package_root = Path(cv2.__file__).resolve().parent
    binaries = sorted(package_root.glob("cv2*.so")) + sorted(package_root.glob("cv2*.pyd"))
    if len(binaries) != 1:
        raise RuntimeError(f"expected one OpenCV extension binary, found {len(binaries)}")
    return {
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "opencv_threads": 1,
        "opencv_binary_path": str(binaries[0]),
        "opencv_binary_sha256": sha256_file(binaries[0]),
    }


def _board(spec: ControlledCharucoBoardSpec) -> Any:
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        _dictionary(),
    )


def create_controlled_charuco_board(
    output_root: Path,
    *,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length_m: float = 0.04,
    marker_length_m: float = 0.03,
    pixels_per_square: int = 240,
) -> Path:
    """Create one write-once metric ChArUco board and its self-hashing spec."""
    reject_sealed_capability([output_root])
    if output_root.exists():
        raise FileExistsError(f"controlled calibration board is immutable: {output_root}")
    partials = list(output_root.parent.glob(f".{output_root.name}.building-*"))
    if partials:
        raise FileExistsError("a prior partial board build must be audited separately")
    if squares_x < 5 or squares_y < 5 or pixels_per_square < 100:
        raise ValueError("controlled board resolution/layout is too small")
    if not 0.0 < marker_length_m < square_length_m:
        raise ValueError("controlled board marker length must be smaller than square length")
    layout = {
        "dictionary": _DICTIONARY_NAME,
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_length_m": square_length_m,
        "marker_length_m": marker_length_m,
    }
    layout_sha = _canonical_sha256(layout)
    width = squares_x * pixels_per_square
    height = squares_y * pixels_per_square
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_m,
        marker_length_m,
        _dictionary(),
    )
    rendered = board.generateImage((width, height), marginSize=0, borderBits=1)
    replay = board.generateImage((width, height), marginSize=0, borderBits=1)
    if not np.array_equal(rendered, replay):
        raise RuntimeError("controlled ChArUco board did not render deterministically")
    ok, encoded = cv2.imencode(".png", rendered)
    replay_ok, replay_encoded = cv2.imencode(".png", replay)
    if not ok or not replay_ok or not np.array_equal(encoded, replay_encoded):
        raise RuntimeError("controlled ChArUco PNG did not encode deterministically")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    image_stage = stage / "controlled_charuco_board.png"
    image_stage.write_bytes(encoded.tobytes())
    image_final = output_root / image_stage.name
    spec = ControlledCharucoBoardSpec(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_length_m,
        marker_length_m=marker_length_m,
        printed_width_m=squares_x * square_length_m,
        printed_height_m=squares_y * square_length_m,
        image_width_pixels=width,
        image_height_pixels=height,
        border_bits=1,
        layout_sha256=layout_sha,
        board_image_path=str(image_final),
        board_image_sha256=sha256_file(image_stage),
    )
    write_json(stage / "board_spec.json", spec.model_dump(mode="json"))
    os.rename(stage, output_root)
    return output_root / "board_spec.json"


def controlled_calibration_session_template(board_spec_path: Path) -> dict[str, Any]:
    """Return a fail-closed fill-in template bound to one generated board."""
    reject_sealed_capability([board_spec_path])
    if not board_spec_path.is_file():
        raise FileNotFoundError(f"controlled board spec is missing: {board_spec_path}")
    spec = ControlledCharucoBoardSpec.model_validate(read_json(board_spec_path))
    training_root = "data/private/postv3_v01/calibration/training"
    evaluator_root = "data/private/postv3_v01/calibration/evaluator"
    stereo_root = "data/private/postv3_v01/calibration/stereo"
    session = ControlledCalibrationSession(
        board_spec_path=str(board_spec_path),
        board_spec_sha256=sha256_file(board_spec_path),
        printed_square_measurements_m=[spec.square_length_m] * 3,
        training_intrinsic_images=[
            f"{training_root}/intrinsics_{index:02d}.png" for index in range(12)
        ],
        evaluator_intrinsic_images=[
            f"{evaluator_root}/intrinsics_{index:02d}.png" for index in range(12)
        ],
        stereo_pairs=[
            StereoImagePair(
                training_image=f"{stereo_root}/training_{index:02d}.png",
                evaluator_image=f"{stereo_root}/evaluator_{index:02d}.png",
            )
            for index in range(12)
        ],
        training_setup_image=f"{training_root}/locked_floor_board.png",
        rotation_axis_origin_board_m=(
            spec.printed_width_m / 2.0,
            spec.printed_height_m / 2.0,
            0.0,
        ),
        vertical_axis_board_normal_sign=1,
        synchronization=[
            SynchronizationEvents(
                direction=direction,
                method="audible_visual_sync_event",
                training_event_seconds=[0.0, 54.0, 108.0],
                evaluator_event_seconds=[0.0, 54.0, 108.0],
            )
            for direction in ("clockwise", "counter_clockwise")
        ],
    )
    payload = session.model_dump(mode="json")
    payload["template_only"] = True
    payload["printed_square_measurements_m"] = [
        "REPLACE_WITH_MEASURED_SQUARE_LENGTH_METRES",
        "REPLACE_WITH_MEASURED_SQUARE_LENGTH_METRES",
        "REPLACE_WITH_MEASURED_SQUARE_LENGTH_METRES",
    ]
    for record in payload["synchronization"]:
        record["training_event_seconds"] = ["REPLACE_WITH_THREE_OR_MORE_NATIVE_TRAINING_TIMESTAMPS"]
        record["evaluator_event_seconds"] = ["REPLACE_WITH_MATCHED_NATIVE_EVALUATOR_TIMESTAMPS"]
    return payload


def controlled_training_calibration_session_template(
    board_spec_path: Path,
) -> dict[str, Any]:
    """Return a fail-closed one-camera calibration template bound to the board."""
    reject_sealed_capability([board_spec_path])
    if not board_spec_path.is_file():
        raise FileNotFoundError(f"controlled board spec is missing: {board_spec_path}")
    spec = ControlledCharucoBoardSpec.model_validate(read_json(board_spec_path))
    training_root = "data/private/postv3_v01/calibration/training"
    session = ControlledTrainingCalibrationSession(
        board_spec_path=str(board_spec_path),
        board_spec_sha256=sha256_file(board_spec_path),
        printed_square_measurements_m=[spec.square_length_m] * 3,
        training_intrinsic_images=[
            f"{training_root}/intrinsics_{index:02d}.png" for index in range(12)
        ],
        training_setup_image=f"{training_root}/locked_floor_board.png",
        rotation_axis_origin_board_m=(
            spec.printed_width_m / 2.0,
            spec.printed_height_m / 2.0,
            0.0,
        ),
        vertical_axis_board_normal_sign=1,
    )
    payload = session.model_dump(mode="json")
    payload["template_only"] = True
    payload["printed_square_measurements_m"] = [
        "REPLACE_WITH_MEASURED_SQUARE_LENGTH_METRES",
        "REPLACE_WITH_MEASURED_SQUARE_LENGTH_METRES",
        "REPLACE_WITH_MEASURED_SQUARE_LENGTH_METRES",
    ]
    return payload


def _load_image(path: Path) -> np.ndarray:
    reject_sealed_capability([path])
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"calibration image cannot be decoded: {path}")
    return image


def _detect_board_points(
    image: np.ndarray,
    board: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(image)
    if corners is None or ids is None or len(ids) < _MINIMUM_CORNERS:
        raise ValueError("fewer than eight ChArUco corners were detected")
    object_points, image_points = board.matchImagePoints(corners, ids)
    objects = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    identifiers = np.asarray(ids, dtype=np.int64).reshape(-1)
    if len(objects) != len(pixels) or len(pixels) != len(identifiers):
        raise RuntimeError("ChArUco point matching returned inconsistent lengths")
    return objects, pixels, identifiers


def _per_view_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: Sequence[Any],
    tvecs: Sequence[Any],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for objects, pixels, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(objects, rvec, tvec, camera_matrix, distortion)
        residual = projected.reshape(-1, 2) - pixels.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    return errors


def _normal_span_degrees(rvecs: Sequence[Any]) -> float:
    normals = []
    for rvec in rvecs:
        rotation, _ = cv2.Rodrigues(rvec)
        normal = rotation[:, 2]
        normals.append(normal / np.linalg.norm(normal))
    maximum = 0.0
    for first_index, first in enumerate(normals):
        for second in normals[first_index + 1 :]:
            cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
            maximum = max(maximum, float(np.rad2deg(np.arccos(cosine))))
    return maximum


def _calibrate_intrinsics(
    paths: list[Path],
    board: Any,
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    detected_corner_counts: list[int] = []
    for path in paths:
        image = _load_image(path)
        current_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError("all intrinsic calibration images must share one resolution")
        objects, pixels, _ = _detect_board_points(image, board)
        object_points.append(objects)
        image_points.append(pixels)
        detected_corner_counts.append(len(pixels))
    if image_size is None or len(object_points) < 10:
        raise ValueError("intrinsic calibration requires at least ten usable views")
    cv2.setNumThreads(1)
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1.0e-12),
    )
    per_view = _per_view_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        distortion,
    )
    normal_span = _normal_span_degrees(rvecs)
    if not np.isfinite(rms) or rms > _MAXIMUM_INTRINSIC_RMS_PIXELS:
        raise ValueError("intrinsic ChArUco reprojection RMS exceeds 1.0 pixels")
    if max(per_view) > _MAXIMUM_VIEW_RMS_PIXELS:
        raise ValueError("an intrinsic ChArUco view exceeds 1.5 pixels RMS")
    if normal_span < _MINIMUM_NORMAL_SPAN_DEGREES:
        raise ValueError("intrinsic ChArUco viewpoints span fewer than 15 degrees")
    width, height = image_size
    focal_limits = (0.2 * max(width, height), 10.0 * max(width, height))
    focal_values = (float(camera_matrix[0, 0]), float(camera_matrix[1, 1]))
    if not all(focal_limits[0] <= value <= focal_limits[1] for value in focal_values):
        raise ValueError("calibrated focal length is outside the registered plausibility range")
    if not (0.0 <= camera_matrix[0, 2] < width and 0.0 <= camera_matrix[1, 2] < height):
        raise ValueError("calibrated principal point lies outside the image")
    if float(np.max(np.abs(distortion))) > 2.0:
        raise ValueError("calibrated distortion exceeds the registered plausibility range")
    return (
        {
            "image_width": width,
            "image_height": height,
            "camera_matrix": np.asarray(camera_matrix, dtype=np.float64),
            "distortion": np.asarray(distortion, dtype=np.float64).reshape(-1),
            "rms_pixels": float(rms),
            "per_view_rms_pixels": per_view,
            "normal_span_degrees": normal_span,
            "detected_corner_counts": detected_corner_counts,
        },
        object_points,
        image_points,
    )


def _common_pair_points(
    training_image: np.ndarray,
    evaluator_image: np.ndarray,
    board: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, training_pixels, training_ids = _detect_board_points(training_image, board)
    _, evaluator_pixels, evaluator_ids = _detect_board_points(evaluator_image, board)
    training_by_id = {
        int(identifier): pixel
        for identifier, pixel in zip(training_ids, training_pixels, strict=True)
    }
    evaluator_by_id = {
        int(identifier): pixel
        for identifier, pixel in zip(evaluator_ids, evaluator_pixels, strict=True)
    }
    common = sorted(set(training_by_id) & set(evaluator_by_id))
    if len(common) < _MINIMUM_CORNERS:
        raise ValueError("a stereo pair has fewer than eight common ChArUco corners")
    board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
    objects = board_corners[np.asarray(common, dtype=np.int64)]
    training = np.asarray([training_by_id[index] for index in common], dtype=np.float32)
    evaluator = np.asarray([evaluator_by_id[index] for index in common], dtype=np.float32)
    return objects, training, evaluator


def _calibrate_stereo(
    pairs: list[StereoImagePair],
    board: Any,
    training: dict[str, Any],
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    if (training["image_width"], training["image_height"]) != (
        evaluator["image_width"],
        evaluator["image_height"],
    ):
        raise ValueError("controlled stereo calibration requires equal camera resolutions")
    object_points: list[np.ndarray] = []
    training_points: list[np.ndarray] = []
    evaluator_points: list[np.ndarray] = []
    common_corner_counts: list[int] = []
    for pair in pairs:
        objects, left, right = _common_pair_points(
            _load_image(Path(pair.training_image)),
            _load_image(Path(pair.evaluator_image)),
            board,
        )
        object_points.append(objects)
        training_points.append(left)
        evaluator_points.append(right)
        common_corner_counts.append(len(objects))
    image_size = (training["image_width"], training["image_height"])
    result = cv2.stereoCalibrate(
        object_points,
        training_points,
        evaluator_points,
        training["camera_matrix"].copy(),
        training["distortion"].copy(),
        evaluator["camera_matrix"].copy(),
        evaluator["distortion"].copy(),
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1.0e-12),
    )
    rms = float(result[0])
    evaluator_from_training_rotation = np.asarray(result[5], dtype=np.float64)
    evaluator_from_training_translation = np.asarray(result[6], dtype=np.float64).reshape(3)
    if not np.isfinite(rms) or rms > _MAXIMUM_STEREO_RMS_PIXELS:
        raise ValueError("stereo ChArUco reprojection RMS exceeds 1.5 pixels")
    training_from_evaluator_rotation = evaluator_from_training_rotation.T
    training_from_evaluator_translation = -(
        training_from_evaluator_rotation @ evaluator_from_training_translation
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = training_from_evaluator_rotation
    transform[:3, 3] = training_from_evaluator_translation
    baseline = float(np.linalg.norm(training_from_evaluator_translation))
    if baseline < 0.05:
        raise ValueError("controlled evaluator stereo baseline is below 50 mm")
    return {
        "rms_pixels": rms,
        "training_camera_from_evaluator": transform,
        "baseline_m": baseline,
        "common_corner_counts": common_corner_counts,
    }


def _setup_pose(
    path: Path,
    board: Any,
    intrinsics: dict[str, Any],
) -> dict[str, Any]:
    objects, pixels, _ = _detect_board_points(_load_image(path), board)
    valid, rvec, tvec = cv2.solvePnP(
        objects,
        pixels,
        intrinsics["camera_matrix"],
        intrinsics["distortion"],
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not valid:
        raise ValueError("training setup ChArUco pose could not be recovered")
    rotation, _ = cv2.Rodrigues(rvec)
    camera_from_board = np.eye(4, dtype=np.float64)
    camera_from_board[:3, :3] = rotation
    camera_from_board[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    world_from_camera = np.linalg.inv(camera_from_board)
    projected, _ = cv2.projectPoints(
        objects,
        rvec,
        tvec,
        intrinsics["camera_matrix"],
        intrinsics["distortion"],
    )
    residual = projected.reshape(-1, 2) - pixels
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    if rms > _MAXIMUM_SETUP_RMS_PIXELS:
        raise ValueError("training setup pose exceeds 1.5 pixels RMS")
    return {
        "world_from_camera": world_from_camera,
        "rms_pixels": rms,
        "corner_count": len(objects),
    }


def _fit_synchronization(events: SynchronizationEvents) -> tuple[dict[str, Any], dict[str, Any]]:
    training = np.asarray(events.training_event_seconds, dtype=np.float64)
    evaluator = np.asarray(events.evaluator_event_seconds, dtype=np.float64)
    design = np.column_stack((training, np.ones(len(training))))
    coefficients, *_ = np.linalg.lstsq(design, evaluator, rcond=None)
    scale, offset = float(coefficients[0]), float(coefficients[1])
    residual_ms = (design @ coefficients - evaluator) * 1000.0
    maximum_residual = float(np.max(np.abs(residual_ms)))
    if scale <= 0.0 or maximum_residual > 8.0:
        raise ValueError("controlled synchronization affine fit exceeds 8 ms")
    output = DirectionSynchronization(
        direction=events.direction,
        evaluator_time_from_training_scale=scale,
        evaluator_time_from_training_offset_seconds=offset,
        synchronization_residual_ms=maximum_residual,
        method=events.method,
    ).model_dump(mode="json")
    diagnostics = {
        "direction": events.direction,
        "event_count": len(training),
        "maximum_absolute_residual_ms": maximum_residual,
        "rms_residual_ms": float(np.sqrt(np.mean(residual_ms**2))),
    }
    return output, diagnostics


def _intrinsics_model(
    result: dict[str, Any],
    *,
    image_count: int,
    spec: ControlledCharucoBoardSpec,
) -> CalibratedIntrinsics:
    return CalibratedIntrinsics(
        image_width=result["image_width"],
        image_height=result["image_height"],
        camera_matrix=result["camera_matrix"].tolist(),
        distortion_coefficients=result["distortion"].tolist(),
        reprojection_rms_pixels=result["rms_pixels"],
        calibration_image_count=image_count,
        fiducial_layout_sha256=spec.layout_sha256,
        fiducial_square_size_m=spec.square_length_m,
    )


def _solve_session(
    session: ControlledCalibrationSession,
    spec: ControlledCharucoBoardSpec,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    board = _board(spec)
    training_paths = [Path(value) for value in session.training_intrinsic_images]
    evaluator_paths = [Path(value) for value in session.evaluator_intrinsic_images]
    training, _, _ = _calibrate_intrinsics(training_paths, board)
    evaluator, _, _ = _calibrate_intrinsics(evaluator_paths, board)
    stereo = _calibrate_stereo(session.stereo_pairs, board, training, evaluator)
    setup = _setup_pose(Path(session.training_setup_image), board, training)
    synchronization: list[dict[str, Any]] = []
    sync_diagnostics: list[dict[str, Any]] = []
    for events in sorted(session.synchronization, key=lambda item: item.direction):
        output, diagnostics = _fit_synchronization(events)
        synchronization.append(output)
        sync_diagnostics.append(diagnostics)
    training_output = TrainingCameraCalibration(
        intrinsics=_intrinsics_model(
            training,
            image_count=len(training_paths),
            spec=spec,
        ),
        world_from_camera=setup["world_from_camera"].tolist(),
        rotation_axis_origin_world_m=session.rotation_axis_origin_board_m,
        rotation_axis_direction_world=(
            0.0,
            0.0,
            float(session.vertical_axis_board_normal_sign),
        ),
        rotation_axis_registration_method=("known_scale_floor_fiducial_and_vertical_gravity"),
    ).model_dump(mode="json")
    evaluator_output = EvaluatorStereoCalibration(
        intrinsics=_intrinsics_model(
            evaluator,
            image_count=len(evaluator_paths),
            spec=spec,
        ),
        training_camera_from_evaluator=stereo["training_camera_from_evaluator"].tolist(),
        stereo_baseline_m=stereo["baseline_m"],
        synchronization=[DirectionSynchronization.model_validate(item) for item in synchronization],
    ).model_dump(mode="json")
    diagnostics = {
        "training_intrinsics": {
            key: value
            for key, value in training.items()
            if key not in {"camera_matrix", "distortion"}
        },
        "evaluator_intrinsics": {
            key: value
            for key, value in evaluator.items()
            if key not in {"camera_matrix", "distortion"}
        },
        "stereo": {
            "rms_pixels": stereo["rms_pixels"],
            "baseline_m": stereo["baseline_m"],
            "common_corner_counts": stereo["common_corner_counts"],
        },
        "training_setup": {
            "rms_pixels": setup["rms_pixels"],
            "corner_count": setup["corner_count"],
        },
        "synchronization": sync_diagnostics,
    }
    return training_output, evaluator_output, diagnostics


def _solve_training_session(
    session: ControlledTrainingCalibrationSession,
    spec: ControlledCharucoBoardSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    board = _board(spec)
    training_paths = [Path(value) for value in session.training_intrinsic_images]
    training, _, _ = _calibrate_intrinsics(training_paths, board)
    setup = _setup_pose(Path(session.training_setup_image), board, training)
    output = TrainingCameraCalibration(
        intrinsics=_intrinsics_model(
            training,
            image_count=len(training_paths),
            spec=spec,
        ),
        world_from_camera=setup["world_from_camera"].tolist(),
        rotation_axis_origin_world_m=session.rotation_axis_origin_board_m,
        rotation_axis_direction_world=(
            0.0,
            0.0,
            float(session.vertical_axis_board_normal_sign),
        ),
        rotation_axis_registration_method=("known_scale_floor_fiducial_and_vertical_gravity"),
    ).model_dump(mode="json")
    diagnostics = {
        "training_intrinsics": {
            key: value
            for key, value in training.items()
            if key not in {"camera_matrix", "distortion"}
        },
        "training_setup": {
            "rms_pixels": setup["rms_pixels"],
            "corner_count": setup["corner_count"],
        },
    }
    return output, diagnostics


def _source_inventory(session: ControlledCalibrationSession) -> list[dict[str, Any]]:
    roles_by_path: dict[str, set[str]] = {}

    def register(value: str, role: str) -> None:
        path = str(Path(value))
        roles_by_path.setdefault(path, set()).add(role)

    for value in session.training_intrinsic_images:
        register(value, "training_intrinsics")
    for value in session.evaluator_intrinsic_images:
        register(value, "evaluator_intrinsics")
    for pair in session.stereo_pairs:
        register(pair.training_image, "stereo_training")
        register(pair.evaluator_image, "stereo_evaluator")
    register(session.training_setup_image, "training_setup_floor_board")
    inventory: list[dict[str, Any]] = []
    for value in sorted(roles_by_path):
        path = Path(value)
        reject_sealed_capability([path])
        if not path.is_file():
            raise FileNotFoundError(f"controlled calibration source is missing: {path}")
        inventory.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "roles": sorted(roles_by_path[value]),
            }
        )
    return inventory


def _training_source_inventory(
    session: ControlledTrainingCalibrationSession,
) -> list[dict[str, Any]]:
    roles_by_path: dict[str, set[str]] = {}
    for value in session.training_intrinsic_images:
        roles_by_path.setdefault(str(Path(value)), set()).add("training_intrinsics")
    roles_by_path.setdefault(str(Path(session.training_setup_image)), set()).add(
        "training_setup_floor_board"
    )
    inventory: list[dict[str, Any]] = []
    for value in sorted(roles_by_path):
        path = Path(value)
        reject_sealed_capability([path])
        if not path.is_file():
            raise FileNotFoundError(f"controlled calibration source is missing: {path}")
        inventory.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "roles": sorted(roles_by_path[value]),
            }
        )
    return inventory


def calibrate_controlled_cameras(
    session_path: Path,
    output_root: Path,
) -> Path:
    """Solve metric training/stereo calibration twice and emit immutable evidence."""
    reject_sealed_capability([session_path, output_root])
    if output_root.exists():
        raise FileExistsError(f"controlled camera calibration is immutable: {output_root}")
    partials = list(output_root.parent.glob(f".{output_root.name}.building-*"))
    if partials:
        raise FileExistsError("a prior partial calibration run must be audited separately")
    session_sha = sha256_file(session_path)
    session = ControlledCalibrationSession.model_validate(read_json(session_path))
    board_spec_path = Path(session.board_spec_path)
    reject_sealed_capability([board_spec_path])
    if not board_spec_path.is_file() or sha256_file(board_spec_path) != session.board_spec_sha256:
        raise ValueError("controlled calibration board-spec hash mismatch")
    spec = ControlledCharucoBoardSpec.model_validate(read_json(board_spec_path))
    measurements = np.asarray(session.printed_square_measurements_m, dtype=np.float64)
    if np.max(np.abs(measurements - spec.square_length_m)) > 0.001:
        raise ValueError("printed ChArUco square measurements differ from the metric spec")
    board_image = Path(spec.board_image_path)
    reject_sealed_capability([board_image])
    if not board_image.is_file() or sha256_file(board_image) != spec.board_image_sha256:
        raise ValueError("controlled calibration board-image hash mismatch")
    source_inventory = _source_inventory(session)
    first_training, first_evaluator, first_diagnostics = _solve_session(session, spec)
    replay_training, replay_evaluator, replay_diagnostics = _solve_session(session, spec)
    if (
        first_training != replay_training
        or first_evaluator != replay_evaluator
        or first_diagnostics != replay_diagnostics
    ):
        raise RuntimeError("controlled camera calibration did not replay exactly")
    if source_inventory != _source_inventory(session):
        raise RuntimeError("controlled calibration source bytes changed during execution")
    if (
        sha256_file(session_path) != session_sha
        or sha256_file(board_spec_path) != session.board_spec_sha256
        or sha256_file(board_image) != spec.board_image_sha256
    ):
        raise RuntimeError("controlled calibration manifest or board changed during execution")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    training_stage = write_json(stage / "training_camera_calibration.json", first_training)
    evaluator_stage = write_json(stage / "evaluator_stereo_calibration.json", first_evaluator)
    training_final = output_root / training_stage.name
    evaluator_final = output_root / evaluator_stage.name
    report = {
        "schema_version": "frayid_v3_controlled_camera_calibration_report.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "evidence_scope": "controlled_calibration_only",
        "session_path": str(session_path),
        "session_sha256": session_sha,
        "board_spec_path": str(board_spec_path),
        "board_spec_sha256": session.board_spec_sha256,
        "board_layout_sha256": spec.layout_sha256,
        "source_inventory": source_inventory,
        "diagnostics": first_diagnostics,
        "thresholds": {
            "minimum_corners_per_view": _MINIMUM_CORNERS,
            "maximum_intrinsic_rms_pixels": _MAXIMUM_INTRINSIC_RMS_PIXELS,
            "maximum_view_rms_pixels": _MAXIMUM_VIEW_RMS_PIXELS,
            "minimum_normal_span_degrees": _MINIMUM_NORMAL_SPAN_DEGREES,
            "maximum_stereo_rms_pixels": _MAXIMUM_STEREO_RMS_PIXELS,
            "maximum_setup_rms_pixels": _MAXIMUM_SETUP_RMS_PIXELS,
            "maximum_synchronization_residual_ms": 8.0,
        },
        "training_camera_calibration": {
            "path": str(training_final),
            "sha256": sha256_file(training_stage),
        },
        "evaluator_stereo_calibration": {
            "path": str(evaluator_final),
            "sha256": sha256_file(evaluator_stage),
        },
        "exact_same_input_replay": True,
        "runtime": _opencv_runtime(),
        "training_camera_role": "measured_training_camera",
        "evaluator_camera_role": "evaluator_only",
        "evaluator_fitting_access": False,
        "evaluator_parameter_selection_access": False,
        "project_training_records_read": 0,
        "historical_development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_geometry_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
    }
    write_json(stage / "calibration_report.json", report)
    os.rename(stage, output_root)
    return output_root / "calibration_report.json"


def calibrate_controlled_training_camera(
    session_path: Path,
    output_root: Path,
) -> Path:
    """Solve one-camera metric calibration twice and emit immutable evidence."""
    reject_sealed_capability([session_path, output_root])
    if output_root.exists():
        raise FileExistsError(f"controlled camera calibration is immutable: {output_root}")
    partials = list(output_root.parent.glob(f".{output_root.name}.building-*"))
    if partials:
        raise FileExistsError("a prior partial calibration run must be audited separately")
    session_sha = sha256_file(session_path)
    session = ControlledTrainingCalibrationSession.model_validate(read_json(session_path))
    board_spec_path = Path(session.board_spec_path)
    reject_sealed_capability([board_spec_path])
    if not board_spec_path.is_file() or sha256_file(board_spec_path) != session.board_spec_sha256:
        raise ValueError("controlled calibration board-spec hash mismatch")
    spec = ControlledCharucoBoardSpec.model_validate(read_json(board_spec_path))
    measurements = np.asarray(session.printed_square_measurements_m, dtype=np.float64)
    if np.max(np.abs(measurements - spec.square_length_m)) > 0.001:
        raise ValueError("printed ChArUco square measurements differ from the metric spec")
    board_image = Path(spec.board_image_path)
    reject_sealed_capability([board_image])
    if not board_image.is_file() or sha256_file(board_image) != spec.board_image_sha256:
        raise ValueError("controlled calibration board-image hash mismatch")
    source_inventory = _training_source_inventory(session)
    first_training, first_diagnostics = _solve_training_session(session, spec)
    replay_training, replay_diagnostics = _solve_training_session(session, spec)
    if first_training != replay_training or first_diagnostics != replay_diagnostics:
        raise RuntimeError("controlled training-camera calibration did not replay exactly")
    if source_inventory != _training_source_inventory(session):
        raise RuntimeError("controlled calibration source bytes changed during execution")
    if (
        sha256_file(session_path) != session_sha
        or sha256_file(board_spec_path) != session.board_spec_sha256
        or sha256_file(board_image) != spec.board_image_sha256
    ):
        raise RuntimeError("controlled calibration manifest or board changed during execution")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    training_stage = write_json(stage / "training_camera_calibration.json", first_training)
    training_final = output_root / training_stage.name
    report = {
        "schema_version": "frayid_v3_controlled_training_camera_calibration_report.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "capture_mode": "single_camera_evidence_consistent",
        "scientific_claim_ceiling": "evidence_consistent_mantle_reconstruction",
        "metric_accuracy_claim_allowed": False,
        "evidence_scope": "controlled_calibration_only",
        "session_path": str(session_path),
        "session_sha256": session_sha,
        "board_spec_path": str(board_spec_path),
        "board_spec_sha256": session.board_spec_sha256,
        "board_layout_sha256": spec.layout_sha256,
        "source_inventory": source_inventory,
        "diagnostics": first_diagnostics,
        "thresholds": {
            "minimum_corners_per_view": _MINIMUM_CORNERS,
            "maximum_intrinsic_rms_pixels": _MAXIMUM_INTRINSIC_RMS_PIXELS,
            "maximum_view_rms_pixels": _MAXIMUM_VIEW_RMS_PIXELS,
            "minimum_normal_span_degrees": _MINIMUM_NORMAL_SPAN_DEGREES,
            "maximum_setup_rms_pixels": _MAXIMUM_SETUP_RMS_PIXELS,
        },
        "training_camera_calibration": {
            "path": str(training_final),
            "sha256": sha256_file(training_stage),
        },
        "independent_evaluator_available": False,
        "evaluator_files_read": 0,
        "exact_same_input_replay": True,
        "runtime": _opencv_runtime(),
        "training_camera_role": "measured_training_camera",
        "project_training_records_read": 0,
        "historical_development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_geometry_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
    }
    write_json(stage / "calibration_report.json", report)
    os.rename(stage, output_root)
    return output_root / "calibration_report.json"


__all__ = [
    "ControlledCalibrationSession",
    "ControlledCharucoBoardSpec",
    "ControlledTrainingCalibrationSession",
    "StereoImagePair",
    "SynchronizationEvents",
    "calibrate_controlled_cameras",
    "calibrate_controlled_training_camera",
    "controlled_calibration_session_template",
    "controlled_training_calibration_session_template",
    "create_controlled_charuco_board",
]
