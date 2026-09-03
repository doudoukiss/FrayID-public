from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.evidence_master import proxy_coordinate_contract, render_analysis_proxy
from frayid.v2.video_forensics import (
    FrameTimestamp,
    camera_verdict,
    decoded_frame_metrics,
    estimate_background_transforms,
    executable_version,
    iter_sequential_rgb_frames,
    probe_video_forensics,
    summarize_timestamps,
)

EXPERIMENT_ID: Literal["postv3_v01_controlled_recapture_evidence_master_r01"] = (
    "postv3_v01_controlled_recapture_evidence_master_r01"
)
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class StrictCaptureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HoldAnnotation(StrictCaptureModel):
    angle_degrees: int = Field(ge=0, le=350, multiple_of=10)
    stable_start_seconds: float = Field(ge=0.0)
    stable_end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _minimum_stable_duration(self) -> HoldAnnotation:
        if self.stable_end_seconds - self.stable_start_seconds < 2.0:
            raise ValueError("every angle requires at least two stable seconds")
        return self


class TrainingCaptureClip(StrictCaptureModel):
    direction: Literal["clockwise", "counter_clockwise"]
    path: str = Field(min_length=1)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    holds: list[HoldAnnotation]

    @model_validator(mode="after")
    def _complete_ordered_turn(self) -> TrainingCaptureClip:
        if len(self.holds) != 36:
            raise ValueError("each direction must contain exactly 36 holds")
        angles = [hold.angle_degrees for hold in self.holds]
        if set(angles) != set(range(0, 360, 10)):
            raise ValueError("each direction must contain every ten-degree angle exactly once")
        if angles[0] != 0:
            raise ValueError("each direction must start from the registered zero-degree pose")
        sign = 1 if self.direction == "clockwise" else -1
        for previous_angle, current_angle in pairwise(angles):
            if (current_angle - previous_angle) % 360 != (10 * sign) % 360:
                raise ValueError(f"hold order does not match {self.direction}")
        for previous_hold, current_hold in pairwise(self.holds):
            if current_hold.stable_start_seconds < previous_hold.stable_end_seconds:
                raise ValueError("stable hold intervals cannot overlap or run backward")
        return self


class ManualCameraSettings(StrictCaptureModel):
    exposure_mode: Literal["manual"] = "manual"
    focus_mode: Literal["manual"] = "manual"
    white_balance_mode: Literal["manual"] = "manual"
    shutter_seconds: float = Field(gt=0.0)
    iso: float = Field(gt=0.0)
    focal_length_mm: float = Field(gt=0.0)
    electronic_stabilization: Literal[False] = False
    optical_stabilization: Literal[False] = False
    auto_framing: Literal[False] = False
    dynamic_hdr: Literal[False] = False


class DeviceManagedCameraSettings(StrictCaptureModel):
    """Fail-closed settings for a camera whose physical controls are not exposed."""

    exposure_mode: Literal["device_managed_control_unavailable"] = (
        "device_managed_control_unavailable"
    )
    focus_mode: Literal["device_managed_control_unavailable"] = "device_managed_control_unavailable"
    white_balance_mode: Literal["device_managed_control_unavailable"] = (
        "device_managed_control_unavailable"
    )
    shutter_seconds: None = None
    iso: None = None
    focal_length_mm: None = None
    electronic_stabilization: Literal[False] = False
    optical_stabilization: Literal[False] = False
    auto_framing: Literal[False] = False
    dynamic_hdr: Literal[False] = False
    control_unavailability_reason: str = Field(min_length=1)
    exposure_and_focus_stability_diagnostics_required: Literal[True] = True
    photometric_geometry_allowed: Literal[False] = False


class CalibrationEvidenceFile(StrictCaptureModel):
    path: str = Field(min_length=1)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class CalibratedIntrinsics(StrictCaptureModel):
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    reprojection_rms_pixels: float = Field(ge=0.0)
    calibration_image_count: int = Field(ge=10)
    fiducial_layout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fiducial_square_size_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _valid_intrinsic_matrix(self) -> CalibratedIntrinsics:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("calibrated focal lengths must be positive")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1.0e-9):
            raise ValueError("camera_matrix must have the standard homogeneous final row")
        if distortion.ndim != 1 or not np.all(np.isfinite(distortion)):
            raise ValueError("distortion coefficients must be one finite vector")
        return self


def _validate_rigid_transform(matrix_raw: list[list[float]], *, name: str) -> np.ndarray:
    matrix = np.asarray(matrix_raw, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError(f"{name} must have a rigid homogeneous final row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-3):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-3):
        raise ValueError(f"{name} rotation must have determinant one")
    return matrix


class TrainingCameraCalibration(StrictCaptureModel):
    schema_version: Literal[
        "frayid_v3_training_camera_calibration.v1",
        "frayid_v3_training_camera_calibration.v2",
    ] = "frayid_v3_training_camera_calibration.v2"
    role: Literal["measured_training_camera"] = "measured_training_camera"
    fixed_extrinsics: Literal[True] = True
    method: Literal["known_scale_fiducial_bundle_adjustment"] = (
        "known_scale_fiducial_bundle_adjustment"
    )
    intrinsics: CalibratedIntrinsics
    world_from_camera: list[list[float]]
    rotation_axis_origin_world_m: tuple[float, float, float] | None = None
    rotation_axis_direction_world: tuple[float, float, float] | None = None
    rotation_axis_registration_method: (
        Literal["known_scale_floor_fiducial_and_vertical_gravity"] | None
    ) = None

    @model_validator(mode="after")
    def _valid_extrinsics(self) -> TrainingCameraCalibration:
        _validate_rigid_transform(self.world_from_camera, name="world_from_camera")
        if self.schema_version == "frayid_v3_training_camera_calibration.v2":
            if (
                self.rotation_axis_origin_world_m is None
                or self.rotation_axis_direction_world is None
                or self.rotation_axis_registration_method is None
            ):
                raise ValueError("training calibration v2 requires a registered rotation axis")
            origin = np.asarray(self.rotation_axis_origin_world_m, dtype=np.float64)
            direction = np.asarray(self.rotation_axis_direction_world, dtype=np.float64)
            if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
                raise ValueError("registered rotation axis must be finite")
            if not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-6, rtol=0.0):
                raise ValueError("registered rotation-axis direction must have unit norm")
        return self


class DirectionSynchronization(StrictCaptureModel):
    direction: Literal["clockwise", "counter_clockwise"]
    evaluator_time_from_training_scale: float = Field(gt=0.0)
    evaluator_time_from_training_offset_seconds: float
    synchronization_residual_ms: float = Field(ge=0.0)
    method: Literal["hardware_trigger", "shared_timecode", "audible_visual_sync_event"]


class EvaluatorStereoCalibration(StrictCaptureModel):
    schema_version: Literal["frayid_v3_evaluator_stereo_calibration.v1"] = (
        "frayid_v3_evaluator_stereo_calibration.v1"
    )
    role: Literal["evaluator_only"] = "evaluator_only"
    intrinsics: CalibratedIntrinsics
    training_camera_from_evaluator: list[list[float]]
    stereo_baseline_m: float = Field(gt=0.0)
    synchronization: list[DirectionSynchronization]

    @model_validator(mode="after")
    def _valid_stereo_and_sync(self) -> EvaluatorStereoCalibration:
        matrix = _validate_rigid_transform(
            self.training_camera_from_evaluator,
            name="training_camera_from_evaluator",
        )
        observed_baseline = float(np.linalg.norm(matrix[:3, 3]))
        tolerance = max(0.001, 0.01 * self.stereo_baseline_m)
        if abs(observed_baseline - self.stereo_baseline_m) > tolerance:
            raise ValueError("stereo baseline disagrees with the relative translation")
        if len(self.synchronization) != 2 or {item.direction for item in self.synchronization} != {
            "clockwise",
            "counter_clockwise",
        }:
            raise ValueError("evaluator calibration requires synchronization for both clips")
        return self


class EvaluatorCameraDeclaration(StrictCaptureModel):
    role: Literal["evaluator_only"] = "evaluator_only"
    synchronized: Literal[True] = True
    path: str = Field(min_length=1)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stereo_calibration: CalibrationEvidenceFile | None = None
    fitting_access: Literal[False] = False
    parameter_selection_access: Literal[False] = False
    prior_access: Literal[False] = False


class ControlledCaptureDeclaration(StrictCaptureModel):
    schema_version: Literal[
        "frayid_v3_controlled_capture_declaration.v1",
        "frayid_v3_controlled_capture_declaration.v2",
        "frayid_v3_controlled_capture_declaration.v3",
    ] = "frayid_v3_controlled_capture_declaration.v2"
    experiment_id: Literal["postv3_v01_controlled_recapture_evidence_master_r01"] = EXPERIMENT_ID
    capture_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*-r[0-9]{2}$")
    training_camera_count: Literal[1] = 1
    tripod_locked: bool = True
    camera_support: Literal["tripod_locked", "rigid_stationary_computer"] = "tripod_locked"
    fixed_diffuse_lighting: Literal[True] = True
    visible_background_fiducials: Literal[True] = True
    native_source_bytes_preserved: Literal[True] = True
    pose_instruction: Literal["comfortable_a_pose"] = "comfortable_a_pose"
    width: int = Field(ge=1280)
    height: int = Field(ge=720)
    below_4k_reason: str | None = None
    capture_mode: Literal[
        "dual_camera_metric_evaluation",
        "single_camera_evidence_consistent",
    ] = "dual_camera_metric_evaluation"
    metric_accuracy_claim_allowed: Literal[False] = False
    independent_evaluator_unavailable_reason: str | None = None
    camera_settings: ManualCameraSettings | DeviceManagedCameraSettings
    training_camera_calibration: CalibrationEvidenceFile | None = None
    training_clips: list[TrainingCaptureClip]
    evaluator_camera: EvaluatorCameraDeclaration | None = None
    stabilization_applied: Literal[False] = False
    denoising_applied: Literal[False] = False
    interpolation_applied: Literal[False] = False
    generated_views_applied: Literal[False] = False
    baked_normalization_applied: Literal[False] = False
    temporary_garment_texture_applied: Literal[False] = False

    @model_validator(mode="after")
    def _capture_is_complete_and_role_separated(self) -> ControlledCaptureDeclaration:
        if (self.width < 3840 or self.height < 2160) and (
            not self.below_4k_reason or not self.below_4k_reason.strip()
        ):
            raise ValueError("sub-4K capture requires a recorded availability reason")
        if len(self.training_clips) != 2:
            raise ValueError("V01 requires exactly two directional training clips")
        by_direction = {clip.direction: clip for clip in self.training_clips}
        if set(by_direction) != {"clockwise", "counter_clockwise"}:
            raise ValueError("V01 requires one clockwise and one counter-clockwise clip")
        paths = [clip.path for clip in self.training_clips]
        if len(set(paths)) != 2:
            raise ValueError("directional clips must be distinct files")
        if self.evaluator_camera is not None and self.evaluator_camera.path in paths:
            raise ValueError("the evaluator camera cannot alias a training-camera clip")
        reject_sealed_capability([Path(path) for path in paths])
        if self.schema_version == "frayid_v3_controlled_capture_declaration.v2":
            if self.capture_mode != "dual_camera_metric_evaluation":
                raise ValueError("capture declaration schema v2 is the dual-camera mode")
            if not self.tripod_locked or self.camera_support != "tripod_locked":
                raise ValueError("capture declaration schema v2 requires a locked tripod")
            if not isinstance(self.camera_settings, ManualCameraSettings):
                raise ValueError("capture declaration schema v2 requires manual camera controls")
            if self.training_camera_calibration is None:
                raise ValueError(
                    "capture declaration schema v2 requires training-camera calibration"
                )
            if self.evaluator_camera is None or self.evaluator_camera.stereo_calibration is None:
                raise ValueError(
                    "capture declaration schema v2 requires evaluator stereo calibration"
                )
            reject_sealed_capability([Path(self.training_camera_calibration.path)])
        if self.schema_version == "frayid_v3_controlled_capture_declaration.v3":
            if self.capture_mode != "single_camera_evidence_consistent":
                raise ValueError("capture declaration schema v3 is the single-camera mode")
            if self.tripod_locked or self.camera_support != "rigid_stationary_computer":
                raise ValueError("single-camera Mac mode requires a rigid stationary computer")
            if self.training_camera_calibration is None:
                raise ValueError(
                    "capture declaration schema v3 requires training-camera calibration"
                )
            if self.evaluator_camera is not None:
                raise ValueError("single-camera mode cannot declare an independent evaluator")
            if (
                not self.independent_evaluator_unavailable_reason
                or not self.independent_evaluator_unavailable_reason.strip()
            ):
                raise ValueError("single-camera mode requires the evaluator-unavailable reason")
            reject_sealed_capability([Path(self.training_camera_calibration.path)])
        if (
            self.schema_version != "frayid_v3_controlled_capture_declaration.v3"
            and self.independent_evaluator_unavailable_reason is not None
        ):
            raise ValueError("the evaluator-unavailable reason belongs only to schema v3")
        if self.evaluator_camera is not None:
            reject_sealed_capability([Path(self.evaluator_camera.path)])
            if self.evaluator_camera.stereo_calibration is not None:
                reject_sealed_capability([Path(self.evaluator_camera.stereo_calibration.path)])
        declared_paths = [*paths]
        if self.training_camera_calibration is not None:
            declared_paths.append(self.training_camera_calibration.path)
        if self.evaluator_camera is not None:
            declared_paths.append(self.evaluator_camera.path)
            if self.evaluator_camera.stereo_calibration is not None:
                declared_paths.append(self.evaluator_camera.stereo_calibration.path)
        if len(declared_paths) != len(set(declared_paths)):
            raise ValueError("capture video and calibration evidence paths must be distinct")
        return self


def _holds(direction: Literal["clockwise", "counter_clockwise"]) -> list[dict[str, Any]]:
    angles = list(range(0, 360, 10)) if direction == "clockwise" else [0, *range(350, 0, -10)]
    return [
        {
            "angle_degrees": angle,
            "stable_start_seconds": float(index * 3),
            "stable_end_seconds": float(index * 3 + 2),
        }
        for index, angle in enumerate(angles)
    ]


def controlled_capture_template() -> dict[str, Any]:
    """Return a valid declaration template; placeholder paths are never evidence."""
    return ControlledCaptureDeclaration(
        capture_id="controlled-upper-garment-r01",
        width=3840,
        height=2160,
        camera_settings=ManualCameraSettings(
            shutter_seconds=1.0 / 250.0,
            iso=200.0,
            focal_length_mm=26.0,
        ),
        training_camera_calibration=CalibrationEvidenceFile(
            path="data/private/postv3_v01/REPLACE_training_camera_calibration.json"
        ),
        training_clips=[
            TrainingCaptureClip(
                direction="clockwise",
                path="data/private/postv3_v01/REPLACE_clockwise_native_video",
                holds=[HoldAnnotation.model_validate(item) for item in _holds("clockwise")],
            ),
            TrainingCaptureClip(
                direction="counter_clockwise",
                path="data/private/postv3_v01/REPLACE_counter_clockwise_native_video",
                holds=[HoldAnnotation.model_validate(item) for item in _holds("counter_clockwise")],
            ),
        ],
        evaluator_camera=EvaluatorCameraDeclaration(
            path="data/private/postv3_v01/REPLACE_evaluator_camera_native_video",
            stereo_calibration=CalibrationEvidenceFile(
                path="data/private/postv3_v01/REPLACE_evaluator_stereo_calibration.json"
            ),
        ),
    ).model_dump(mode="json")


def controlled_single_camera_capture_template(
    cue_detection_path: Path | None = None,
) -> dict[str, Any]:
    """Return a strict one-camera template with an evidence-consistent claim ceiling."""
    clips: list[TrainingCaptureClip] = []
    detected: dict[str, Any] | None = None
    if cue_detection_path is not None:
        reject_sealed_capability([cue_detection_path])
        detected = read_json(cue_detection_path)
        if (
            detected.get("schema_version") != "frayid_v3_controlled_capture_cue_detection.v1"
            or detected.get("status") != "pass"
            or detected.get("capture_mode") != "single_camera_evidence_consistent"
        ):
            raise ValueError("single-camera template requires a passing single-camera cue report")
        directions = detected.get("directions")
        if not isinstance(directions, list) or len(directions) != 2:
            raise ValueError("single-camera cue report must contain both directions")
        source_audio = detected.get("source_audio")
        if not isinstance(source_audio, dict):
            raise ValueError("single-camera cue report has no bound source inventory")
        for record in directions:
            direction = str(record.get("direction"))
            if direction not in {"clockwise", "counter_clockwise"}:
                raise ValueError("single-camera cue report has an unknown direction")
            direction_value = cast(Literal["clockwise", "counter_clockwise"], direction)
            source = source_audio.get(direction)
            if not isinstance(source, dict):
                raise ValueError(f"single-camera cue report is missing {direction} source")
            clips.append(
                TrainingCaptureClip(
                    direction=direction_value,
                    path=str(source["path"]),
                    expected_sha256=str(source["sha256"]),
                    holds=[HoldAnnotation.model_validate(item) for item in record["holds"]],
                )
            )
    else:
        template_directions: tuple[
            Literal["clockwise", "counter_clockwise"],
            Literal["clockwise", "counter_clockwise"],
        ] = ("clockwise", "counter_clockwise")
        for direction in template_directions:
            holds = [
                {
                    "angle_degrees": angle,
                    "stable_start_seconds": 3.0 + index * 4.0,
                    "stable_end_seconds": 5.5 + index * 4.0,
                }
                for index, angle in enumerate(
                    list(range(0, 360, 10))
                    if direction == "clockwise"
                    else [0, *range(350, 0, -10)]
                )
            ]
            clips.append(
                TrainingCaptureClip(
                    direction=direction,
                    path=f"data/private/postv3_v01/REPLACE_{direction}_native_video",
                    holds=[HoldAnnotation.model_validate(item) for item in holds],
                )
            )
    return ControlledCaptureDeclaration(
        schema_version="frayid_v3_controlled_capture_declaration.v3",
        capture_id="controlled-upper-garment-single-camera-r01",
        capture_mode="single_camera_evidence_consistent",
        tripod_locked=False,
        camera_support="rigid_stationary_computer",
        independent_evaluator_unavailable_reason=(
            "only one physical MacBook Pro camera is available; Desk View is not "
            "an independent optical viewpoint"
        ),
        width=1920,
        height=1080,
        below_4k_reason="the available MacBook Pro camera is not a verified 4K source",
        camera_settings=DeviceManagedCameraSettings(
            control_unavailability_reason=(
                "AVFoundation does not expose manual exposure, focus, white balance, "
                "shutter, ISO, or focal-length controls for the built-in camera"
            )
        ),
        training_camera_calibration=CalibrationEvidenceFile(
            path="data/private/postv3_v01/REPLACE_training_camera_calibration.json"
        ),
        training_clips=clips,
        evaluator_camera=None,
    ).model_dump(mode="json")


def controlled_calibration_templates() -> dict[str, Any]:
    """Return typed metric calibration templates; placeholders are never evidence."""
    intrinsics = CalibratedIntrinsics(
        image_width=3840,
        image_height=2160,
        camera_matrix=[
            [3000.0, 0.0, 1920.0],
            [0.0, 3000.0, 1080.0],
            [0.0, 0.0, 1.0],
        ],
        distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
        reprojection_rms_pixels=0.0,
        calibration_image_count=20,
        fiducial_layout_sha256="0" * 64,
        fiducial_square_size_m=0.04,
    )
    training = TrainingCameraCalibration(
        intrinsics=intrinsics,
        world_from_camera=np.eye(4, dtype=np.float64).tolist(),
        rotation_axis_origin_world_m=(0.0, 0.0, 0.0),
        rotation_axis_direction_world=(0.0, 1.0, 0.0),
        rotation_axis_registration_method=("known_scale_floor_fiducial_and_vertical_gravity"),
    )
    evaluator_transform = np.eye(4, dtype=np.float64)
    evaluator_transform[0, 3] = 0.5
    evaluator = EvaluatorStereoCalibration(
        intrinsics=intrinsics,
        training_camera_from_evaluator=evaluator_transform.tolist(),
        stereo_baseline_m=0.5,
        synchronization=[
            DirectionSynchronization(
                direction=direction,
                evaluator_time_from_training_scale=1.0,
                evaluator_time_from_training_offset_seconds=0.0,
                synchronization_residual_ms=0.0,
                method="hardware_trigger",
            )
            for direction in ("clockwise", "counter_clockwise")
        ],
    )
    return {
        "schema_version": "frayid_v3_controlled_calibration_templates.v2",
        "placeholder_values_are_evidence": False,
        "training_camera_calibration": training.model_dump(mode="json"),
        "evaluator_stereo_calibration": evaluator.model_dump(mode="json"),
    }


def validate_controlled_capture_declaration(payload: dict[str, Any]) -> dict[str, Any]:
    declaration = ControlledCaptureDeclaration.model_validate(payload)
    return {
        "schema_version": "frayid_v3_controlled_capture_declaration_validation.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "ready_for_physical_capture"
            if declaration.schema_version
            in {
                "frayid_v3_controlled_capture_declaration.v2",
                "frayid_v3_controlled_capture_declaration.v3",
            }
            else "legacy_declaration_requires_v2_calibration"
        ),
        "capture_id": declaration.capture_id,
        "training_camera_count": declaration.training_camera_count,
        "direction_count": len(declaration.training_clips),
        "hold_count": sum(len(clip.holds) for clip in declaration.training_clips),
        "unique_angles_per_direction": 36,
        "minimum_hold_seconds": min(
            hold.stable_end_seconds - hold.stable_start_seconds
            for clip in declaration.training_clips
            for hold in clip.holds
        ),
        "manual_camera_controls": isinstance(declaration.camera_settings, ManualCameraSettings),
        "camera_control_mode": (
            "manual"
            if isinstance(declaration.camera_settings, ManualCameraSettings)
            else "device_managed_control_unavailable"
        ),
        "capture_mode": declaration.capture_mode,
        "camera_support": declaration.camera_support,
        "metric_accuracy_claim_allowed": declaration.metric_accuracy_claim_allowed,
        "scientific_claim_ceiling": (
            "evidence_consistent_mantle_reconstruction"
            if declaration.capture_mode == "single_camera_evidence_consistent"
            else "metric_accuracy_only_after_independent_evaluator_gates"
        ),
        "metric_training_camera_calibration_declared": (
            declaration.training_camera_calibration is not None
        ),
        "metric_evaluator_stereo_calibration_declared": (
            declaration.evaluator_camera is not None
            and declaration.evaluator_camera.stereo_calibration is not None
        ),
        "forbidden_processing_disabled": True,
        "native_source_bytes_preserved": declaration.native_source_bytes_preserved,
        "evaluator_camera_role": (
            declaration.evaluator_camera.role if declaration.evaluator_camera else None
        ),
        "evaluator_camera_fitting_access": False,
        "independent_evaluator_available": declaration.evaluator_camera is not None,
        "physical_capture_completed": False,
        "promotion_eligible": False,
        "project_evidence_reads": 0,
        "sealed_test_accesses": 0,
    }


def audit_controlled_capture_sources(declaration_path: Path) -> dict[str, Any]:
    """Audit native clip hashes/timing without granting evaluator-camera fit access."""
    reject_sealed_capability([declaration_path])
    declaration = ControlledCaptureDeclaration.model_validate(read_json(declaration_path))
    clip_reports: list[dict[str, Any]] = []
    blockers: list[str] = []
    if declaration.schema_version not in {
        "frayid_v3_controlled_capture_declaration.v2",
        "frayid_v3_controlled_capture_declaration.v3",
    }:
        blockers.append("capture_declaration_v2_calibration_required")
    for clip in declaration.training_clips:
        path = Path(clip.path)
        if not path.is_file():
            blockers.append(f"training_clip_missing:{clip.direction}")
            continue
        observed_sha256 = sha256_file(path)
        if clip.expected_sha256 is not None and observed_sha256 != clip.expected_sha256:
            blockers.append(f"training_clip_hash_mismatch:{clip.direction}")
        probe, timestamps, provenance = probe_video_forensics(path)
        timing = summarize_timestamps(timestamps)
        if not timing["strictly_monotonic"]:
            blockers.append(f"native_timing_invalid:{clip.direction}")
        if probe.width != declaration.width or probe.height != declaration.height:
            blockers.append(f"resolution_mismatch:{clip.direction}")
        duration = probe.duration_seconds
        if duration is None or any(hold.stable_end_seconds > duration for hold in clip.holds):
            blockers.append(f"hold_outside_clip_duration:{clip.direction}")
        clip_reports.append(
            {
                "direction": clip.direction,
                "path": str(path),
                "sha256": observed_sha256,
                "probe": probe.model_dump(mode="json"),
                "timing_summary": timing,
                "probe_provenance": provenance,
                "hold_count": len(clip.holds),
                "role": "measured_training_camera",
            }
        )

    training_calibration_report: dict[str, Any] | None = None
    calibration_file = declaration.training_camera_calibration
    if calibration_file is not None:
        calibration_path = Path(calibration_file.path)
        if not calibration_path.is_file():
            blockers.append("training_camera_calibration_missing")
        else:
            observed_sha256 = sha256_file(calibration_path)
            if (
                calibration_file.expected_sha256 is not None
                and observed_sha256 != calibration_file.expected_sha256
            ):
                blockers.append("training_camera_calibration_hash_mismatch")
            calibration = TrainingCameraCalibration.model_validate(read_json(calibration_path))
            if calibration.schema_version != "frayid_v3_training_camera_calibration.v2":
                blockers.append("training_camera_calibration_v2_rotation_axis_required")
            if calibration.intrinsics.fiducial_layout_sha256 == "0" * 64:
                blockers.append("training_camera_calibration_contains_placeholder_fiducial_hash")
            if (
                calibration.intrinsics.image_width != declaration.width
                or calibration.intrinsics.image_height != declaration.height
            ):
                blockers.append("training_camera_calibration_resolution_mismatch")
            training_calibration_report = {
                "path": str(calibration_path),
                "sha256": observed_sha256,
                "schema_version": calibration.schema_version,
                "role": calibration.role,
                "fixed_extrinsics": calibration.fixed_extrinsics,
                "intrinsics": calibration.intrinsics.model_dump(mode="json"),
                "world_from_camera": calibration.world_from_camera,
                "rotation_axis_origin_world_m": calibration.rotation_axis_origin_world_m,
                "rotation_axis_direction_world": calibration.rotation_axis_direction_world,
                "rotation_axis_registration_method": (
                    calibration.rotation_axis_registration_method
                ),
            }

    evaluator_report: dict[str, Any] | None = None
    evaluator = declaration.evaluator_camera
    if evaluator is not None:
        path = Path(evaluator.path)
        if not path.is_file():
            blockers.append("evaluator_clip_missing")
        else:
            observed_sha256 = sha256_file(path)
            if (
                evaluator.expected_sha256 is not None
                and observed_sha256 != evaluator.expected_sha256
            ):
                blockers.append("evaluator_clip_hash_mismatch")
            probe, timestamps, provenance = probe_video_forensics(path)
            timing_summary = summarize_timestamps(timestamps)
            if not timing_summary["strictly_monotonic"]:
                blockers.append("evaluator_native_timing_invalid")
            evaluator_report = {
                "path": str(path),
                "sha256": observed_sha256,
                "probe": probe.model_dump(mode="json"),
                "timing_summary": timing_summary,
                "probe_provenance": provenance,
                "role": "evaluator_only",
                "decoded_for_fitting": False,
                "fitting_access": False,
                "parameter_selection_access": False,
                "prior_access": False,
            }
            stereo_file = evaluator.stereo_calibration
            if stereo_file is None:
                blockers.append("evaluator_stereo_calibration_missing")
            else:
                stereo_path = Path(stereo_file.path)
                if not stereo_path.is_file():
                    blockers.append("evaluator_stereo_calibration_missing")
                else:
                    stereo_sha256 = sha256_file(stereo_path)
                    if (
                        stereo_file.expected_sha256 is not None
                        and stereo_sha256 != stereo_file.expected_sha256
                    ):
                        blockers.append("evaluator_stereo_calibration_hash_mismatch")
                    stereo = EvaluatorStereoCalibration.model_validate(read_json(stereo_path))
                    if stereo.intrinsics.fiducial_layout_sha256 == "0" * 64:
                        blockers.append("evaluator_calibration_contains_placeholder_fiducial_hash")
                    if (
                        stereo.intrinsics.image_width != probe.width
                        or stereo.intrinsics.image_height != probe.height
                    ):
                        blockers.append("evaluator_stereo_calibration_resolution_mismatch")
                    evaluator_delta = timing_summary.get("median_delta_seconds")
                    training_delta_by_direction = {
                        str(report["direction"]): report["timing_summary"].get(
                            "median_delta_seconds"
                        )
                        for report in clip_reports
                    }
                    synchronization = []
                    for item in stereo.synchronization:
                        training_delta = training_delta_by_direction.get(item.direction)
                        finite_deltas = [
                            float(value)
                            for value in (training_delta, evaluator_delta)
                            if value is not None and float(value) > 0.0
                        ]
                        maximum_residual_ms = 500.0 * min(finite_deltas) if finite_deltas else 0.0
                        if item.synchronization_residual_ms > maximum_residual_ms:
                            blockers.append(
                                f"evaluator_sync_residual_above_half_frame:{item.direction}"
                            )
                        synchronization.append(
                            {
                                **item.model_dump(mode="json"),
                                "maximum_allowed_residual_ms": maximum_residual_ms,
                            }
                        )
                    evaluator_report["stereo_calibration"] = {
                        "path": str(stereo_path),
                        "sha256": stereo_sha256,
                        "schema_version": stereo.schema_version,
                        "role": stereo.role,
                        "intrinsics": stereo.intrinsics.model_dump(mode="json"),
                        "training_camera_from_evaluator": (stereo.training_camera_from_evaluator),
                        "stereo_baseline_m": stereo.stereo_baseline_m,
                        "synchronization": synchronization,
                    }

    return {
        "schema_version": "frayid_v3_controlled_capture_source_audit.v1",
        "experiment_id": EXPERIMENT_ID,
        "capture_id": declaration.capture_id,
        "capture_mode": declaration.capture_mode,
        "camera_support": declaration.camera_support,
        "metric_accuracy_claim_allowed": declaration.metric_accuracy_claim_allowed,
        "scientific_claim_ceiling": (
            "evidence_consistent_mantle_reconstruction"
            if declaration.capture_mode == "single_camera_evidence_consistent"
            else "metric_accuracy_only_after_independent_evaluator_gates"
        ),
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "training_clips": clip_reports,
        "training_camera_calibration": training_calibration_report,
        "evaluator_camera": evaluator_report,
        "native_source_bytes_preserved": declaration.native_source_bytes_preserved,
        "decoded_training_frames_authoritative": False,
        "next_state_if_pass": "imported_not_yet_data_bound",
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
    }


def _hold_index_at_time(holds: list[HoldAnnotation], seconds: float) -> int | None:
    for index, hold in enumerate(holds):
        if hold.stable_start_seconds <= seconds < hold.stable_end_seconds:
            return index
    return None


def _timestamp_seconds(timestamp: FrameTimestamp, origin_seconds: float) -> float | None:
    value = timestamp.selected_timestamp_seconds
    return None if value is None else float(value - origin_seconds)


def _background_audit(
    frames: list[np.ndarray],
    source_indices: list[int],
) -> tuple[dict[str, Any] | None, bool, str]:
    try:
        first = estimate_background_transforms(frames, source_indices=source_indices)
        second = estimate_background_transforms(frames, source_indices=source_indices)
    except ValueError as error:
        return ({"error": str(error)}, False, "indeterminate")
    repeatable = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    return (first, repeatable, camera_verdict(first))


def _hold_summary(
    hold: HoldAnnotation,
    frame_records: list[dict[str, Any]],
    *,
    edge_tolerance_seconds: float,
) -> dict[str, Any]:
    relative_times = [float(record["clip_time_seconds"]) for record in frame_records]
    motion = [
        float(record["adjacent_rgb_absolute_difference_mean"])
        for record in frame_records
        if record["adjacent_rgb_absolute_difference_mean"] is not None
    ]
    sharpness = [float(record["laplacian_variance"]) for record in frame_records]
    luminance = [float(record["mean_luminance"]) for record in frame_records]
    near_black = [float(record["near_black_fraction"]) for record in frame_records]
    near_white = [float(record["near_white_fraction"]) for record in frame_records]
    first_time = min(relative_times) if relative_times else None
    last_time = max(relative_times) if relative_times else None
    starts_covered = (
        first_time is not None and first_time <= hold.stable_start_seconds + edge_tolerance_seconds
    )
    ends_covered = (
        last_time is not None and last_time >= hold.stable_end_seconds - edge_tolerance_seconds
    )
    return {
        "angle_degrees": hold.angle_degrees,
        "stable_start_seconds": hold.stable_start_seconds,
        "stable_end_seconds": hold.stable_end_seconds,
        "decoded_frame_count": len(frame_records),
        "first_decoded_clip_time_seconds": first_time,
        "last_decoded_clip_time_seconds": last_time,
        "start_edge_covered": starts_covered,
        "end_edge_covered": ends_covered,
        "interval_coverage_passed": len(frame_records) >= 2 and starts_covered and ends_covered,
        "median_adjacent_rgb_absolute_difference_mean": (
            float(np.median(np.asarray(motion, dtype=np.float64))) if motion else None
        ),
        "p95_adjacent_rgb_absolute_difference_mean": (
            float(np.percentile(np.asarray(motion, dtype=np.float64), 95)) if motion else None
        ),
        "minimum_laplacian_variance": min(sharpness) if sharpness else None,
        "median_laplacian_variance": (
            float(np.median(np.asarray(sharpness, dtype=np.float64))) if sharpness else None
        ),
        "median_mean_luminance": (
            float(np.median(np.asarray(luminance, dtype=np.float64))) if luminance else None
        ),
        "luminance_range": max(luminance) - min(luminance) if luminance else None,
        "maximum_near_black_fraction": max(near_black) if near_black else None,
        "maximum_near_white_fraction": max(near_white) if near_white else None,
        "stability_and_sharpness_role": "diagnostic_not_thresholded",
    }


def build_controlled_capture_evidence_master(
    declaration_path: Path,
    output_root: Path,
    *,
    source_revision: str,
    storage: Literal["png", "hashes_only"] = "hashes_only",
    proxy_size: tuple[int, int] = (640, 640),
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Path:
    """Build an atomic V01 master while keeping evaluator pixels outside fitting."""
    reject_sealed_capability([declaration_path, output_root])
    if not _REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError("source_revision must be a full lowercase Git commit")
    if output_root.exists():
        raise FileExistsError(f"controlled evidence master output is immutable: {output_root}")
    if min(proxy_size) <= 0:
        raise ValueError("proxy dimensions must be positive")
    declaration = ControlledCaptureDeclaration.model_validate(read_json(declaration_path))
    source_audit = audit_controlled_capture_sources(declaration_path)
    reject_sealed_capability(
        [Path(clip.path) for clip in declaration.training_clips]
        + ([Path(declaration.evaluator_camera.path)] if declaration.evaluator_camera else [])
        + (
            [Path(declaration.training_camera_calibration.path)]
            if declaration.training_camera_calibration
            else []
        )
        + (
            [Path(declaration.evaluator_camera.stereo_calibration.path)]
            if declaration.evaluator_camera and declaration.evaluator_camera.stereo_calibration
            else []
        )
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    if storage == "png":
        (stage / "frames").mkdir(parents=True)

    blockers = list(source_audit["blockers"])
    if (
        declaration.capture_mode == "dual_camera_metric_evaluation"
        and declaration.evaluator_camera is None
    ):
        blockers.append("synchronized_evaluator_camera_required")
    clip_reports: list[dict[str, Any]] = []
    zero_degree_proxies: list[np.ndarray] = []
    zero_degree_indices: list[int] = []
    total_training_records = 0
    try:
        for clip_slot, clip in enumerate(declaration.training_clips):
            path = Path(clip.path)
            if not path.is_file():
                continue
            probe, timestamps, probe_provenance = probe_video_forensics(
                path, ffprobe_bin=ffprobe_bin
            )
            timestamp_summary = summarize_timestamps(timestamps)
            finite_timestamps = [
                item.selected_timestamp_seconds
                for item in timestamps
                if item.selected_timestamp_seconds is not None
            ]
            origin_seconds = float(finite_timestamps[0]) if finite_timestamps else 0.0
            median_delta = timestamp_summary["median_delta_seconds"]
            edge_tolerance = max(
                0.1,
                2.0 * float(median_delta) if median_delta is not None else 0.1,
            )
            hold_records: list[list[dict[str, Any]]] = [[] for _ in clip.holds]
            midpoint_distances = [float("inf")] * len(clip.holds)
            midpoint_proxies: list[np.ndarray | None] = [None] * len(clip.holds)
            midpoint_indices: list[int | None] = [None] * len(clip.holds)
            previous_hold_index: int | None = None
            previous_hold_rgb: np.ndarray | None = None
            decoded_count = 0
            for decode_index, rgb in enumerate(
                iter_sequential_rgb_frames(
                    path,
                    width=probe.width,
                    height=probe.height,
                    ffmpeg_bin=ffmpeg_bin,
                )
            ):
                decoded_count += 1
                if decode_index >= len(timestamps):
                    continue
                timestamp = timestamps[decode_index]
                clip_time = _timestamp_seconds(timestamp, origin_seconds)
                if clip_time is None:
                    continue
                hold_index = _hold_index_at_time(clip.holds, clip_time)
                if hold_index is None:
                    previous_hold_index = None
                    previous_hold_rgb = None
                    continue
                if previous_hold_index != hold_index:
                    previous_hold_rgb = None
                metrics = decoded_frame_metrics(rgb, previous_rgb=previous_hold_rgb)
                record: dict[str, Any] = {
                    "direction": clip.direction,
                    "angle_degrees": clip.holds[hold_index].angle_degrees,
                    "decode_index": decode_index,
                    "source_frame_index": decode_index,
                    "native_timing": timestamp.model_dump(mode="json"),
                    "clip_time_seconds": clip_time,
                    "evidence_role": "measured_training_camera",
                    "source_bytes_authoritative": True,
                    "decoded_pixels_replayable_measured_derivative": True,
                    "decoded_pixel_format": "rgb24",
                    **metrics,
                }
                if storage == "png":
                    relative = (
                        Path("frames")
                        / clip.direction
                        / f"angle_{clip.holds[hold_index].angle_degrees:03d}"
                        / f"frame_{decode_index:06d}.png"
                    )
                    frame_path = stage / relative
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb, mode="RGB").save(
                        frame_path, format="PNG", compress_level=6, optimize=False
                    )
                    record["lossless_frame_path"] = relative.as_posix()
                    record["lossless_frame_sha256"] = sha256_file(frame_path)
                else:
                    record["lossless_frame_path"] = None
                    record["lossless_frame_sha256"] = None
                hold_records[hold_index].append(record)
                midpoint = (
                    clip.holds[hold_index].stable_start_seconds
                    + clip.holds[hold_index].stable_end_seconds
                ) / 2.0
                distance = abs(clip_time - midpoint)
                if distance < midpoint_distances[hold_index]:
                    proxy_contract = proxy_coordinate_contract(
                        (probe.width, probe.height), proxy_size
                    )
                    midpoint_proxies[hold_index] = render_analysis_proxy(rgb, proxy_contract)
                    midpoint_indices[hold_index] = decode_index
                    midpoint_distances[hold_index] = distance
                previous_hold_index = hold_index
                previous_hold_rgb = rgb.copy()

            if decoded_count != len(timestamps):
                blockers.append(f"decoded_frame_count_mismatch:{clip.direction}")
            if (
                probe.reported_frame_count is not None
                and decoded_count != probe.reported_frame_count
            ):
                blockers.append(f"stream_frame_count_mismatch:{clip.direction}")
            if not bool(timestamp_summary["strictly_monotonic"]):
                blockers.append(f"native_timing_invalid:{clip.direction}")
            summaries = [
                _hold_summary(hold, records, edge_tolerance_seconds=edge_tolerance)
                for hold, records in zip(clip.holds, hold_records, strict=True)
            ]
            for summary in summaries:
                if not summary["interval_coverage_passed"]:
                    blockers.append(
                        f"stable_hold_frame_coverage_failed:{clip.direction}:"
                        f"{summary['angle_degrees']}"
                    )
            material_chart_anchor_records: list[dict[str, Any]] = []
            for hold_index, (hold, midpoint_index) in enumerate(
                zip(clip.holds, midpoint_indices, strict=True)
            ):
                matches = [
                    record
                    for record in hold_records[hold_index]
                    if record["decode_index"] == midpoint_index
                ]
                if len(matches) != 1:
                    blockers.append(
                        f"material_chart_anchor_selection_failed:{clip.direction}:"
                        f"{hold.angle_degrees}"
                    )
                    continue
                selected = matches[0]
                material_chart_anchor_records.append(
                    {
                        "controlled_record_index": clip_slot * 36 + hold_index,
                        "direction": clip.direction,
                        "angle_degrees": hold.angle_degrees,
                        "source_decode_index": int(selected["decode_index"]),
                        "source_frame_key": (
                            f"{clip.direction}:{int(selected['decode_index']):06d}"
                        ),
                        "clip_time_seconds": float(selected["clip_time_seconds"]),
                        "decoded_rgb_sha256": selected["decoded_rgb_sha256"],
                        "lossless_frame_path": selected["lossless_frame_path"],
                        "lossless_frame_sha256": selected["lossless_frame_sha256"],
                        "evidence_role": "measured_training_camera",
                        "selection_policy": "nearest_native_timestamp_to_registered_hold_midpoint",
                    }
                )
            available_proxies = [proxy for proxy in midpoint_proxies if proxy is not None]
            available_indices = [index for index in midpoint_indices if index is not None]
            if len(available_proxies) >= 2:
                background, repeatable, verdict = _background_audit(
                    available_proxies,
                    [int(index) for index in available_indices],
                )
            else:
                background, repeatable, verdict = None, False, "indeterminate"
            if not repeatable:
                blockers.append(f"background_audit_not_repeatable:{clip.direction}")
            if verdict != "fixed_to_subpixel_precision":
                blockers.append(f"fixed_camera_gate_failed:{clip.direction}")
            if midpoint_proxies[0] is not None and midpoint_indices[0] is not None:
                zero_degree_proxies.append(midpoint_proxies[0])
                zero_degree_indices.append(clip_slot * 1_000_000 + int(midpoint_indices[0]))
            records = [record for group in hold_records for record in group]
            total_training_records += len(records)
            clip_reports.append(
                {
                    "direction": clip.direction,
                    "source": {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "probe": probe.model_dump(mode="json"),
                    },
                    "decode": {
                        "policy": "single_forward_only_sequential_decode",
                        "random_seek_count": 0,
                        "timestamp_synthesis_allowed": False,
                        "decoded_frame_count": decoded_count,
                        "selected_stable_hold_frame_count": len(records),
                        "storage": storage,
                        "ffmpeg_version": executable_version(ffmpeg_bin),
                        **probe_provenance,
                    },
                    "timing_summary": timestamp_summary,
                    "timestamp_origin_seconds": origin_seconds,
                    "hold_edge_tolerance_seconds": edge_tolerance,
                    "hold_summaries": summaries,
                    "material_chart_anchor_records": material_chart_anchor_records,
                    "background_audit_on_nonauthoritative_proxies": background,
                    "background_audit_repeatable": repeatable,
                    "physical_camera_verdict": verdict,
                    "frames": records,
                }
            )

        if len(zero_degree_proxies) == 2:
            cross_background, cross_repeatable, cross_verdict = _background_audit(
                zero_degree_proxies, zero_degree_indices
            )
        else:
            cross_background, cross_repeatable, cross_verdict = None, False, "indeterminate"
        if not cross_repeatable:
            blockers.append("cross_direction_background_audit_not_repeatable")
        if cross_verdict != "fixed_to_subpixel_precision":
            blockers.append("cross_direction_fixed_camera_gate_failed")
        blockers = list(dict.fromkeys(blockers))
        proxy_contract = proxy_coordinate_contract(
            (declaration.width, declaration.height), proxy_size
        )
        stability_diagnostics_recorded = len(clip_reports) == 2 and all(
            summary["median_mean_luminance"] is not None
            and summary["minimum_laplacian_variance"] is not None
            for report in clip_reports
            for summary in report["hold_summaries"]
        )
        manifest: dict[str, Any] = {
            "schema_version": "frayid_v3_controlled_capture_evidence_master.v1",
            "experiment_id": EXPERIMENT_ID,
            "evidence_scope": "train_real",
            "capture_id": declaration.capture_id,
            "capture_mode": declaration.capture_mode,
            "camera_support": declaration.camera_support,
            "metric_accuracy_claim_allowed": declaration.metric_accuracy_claim_allowed,
            "scientific_claim_ceiling": (
                "evidence_consistent_mantle_reconstruction"
                if declaration.capture_mode == "single_camera_evidence_consistent"
                else "metric_accuracy_only_after_independent_evaluator_gates"
            ),
            "source_revision": source_revision,
            "status": "pass" if not blockers else "blocked",
            "blockers": blockers,
            "declaration_path": str(declaration_path),
            "declaration_sha256": sha256_file(declaration_path),
            "source_audit": source_audit,
            "training_camera_calibration": source_audit.get("training_camera_calibration"),
            "training_clips": clip_reports,
            "cross_direction_zero_degree_background_audit": cross_background,
            "cross_direction_background_audit_repeatable": cross_repeatable,
            "cross_direction_physical_camera_verdict": cross_verdict,
            "proxy_coordinate_contract": proxy_contract,
            "evaluator_camera": {
                "role": "evaluator_only",
                "source_audit": source_audit.get("evaluator_camera"),
                "stereo_calibration": (
                    source_audit.get("evaluator_camera", {}).get("stereo_calibration")
                    if isinstance(source_audit.get("evaluator_camera"), dict)
                    else None
                ),
                "decoded_for_fitting": False,
                "training_record_count": 0,
                "fitting_access": False,
                "parameter_selection_access": False,
                "prior_access": False,
            }
            if declaration.evaluator_camera is not None
            else None,
            "evidence_policy": {
                "native_source_bytes_authoritative": True,
                "stable_hold_decoded_pixels_are_replayable_measured_derivatives": True,
                "transition_frames_fitting_access": False,
                "analysis_proxy_authoritative": False,
                "generated_pixels_in_measured_evidence": False,
                "stabilization_applied": False,
                "denoising_applied": False,
                "interpolation_applied": False,
                "generated_views_applied": False,
                "baked_normalization_applied": False,
                "camera_control_mode": (
                    "manual"
                    if isinstance(declaration.camera_settings, ManualCameraSettings)
                    else "device_managed_control_unavailable"
                ),
                "exposure_and_focus_stability_diagnostics_recorded": (
                    stability_diagnostics_recorded
                ),
                "photometric_geometry_allowed": isinstance(
                    declaration.camera_settings, ManualCameraSettings
                ),
            },
            "training_record_count": total_training_records,
            "development_records_read": 0,
            "sealed_test_accesses": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
        }
        manifest_path = write_json(stage / "evidence_master.json", manifest)
        checks = {
            "source_audit_passed": source_audit["status"] == "pass",
            "metric_training_camera_calibration_bound": source_audit.get(
                "training_camera_calibration"
            )
            is not None,
            "two_directional_clips_decoded": len(clip_reports) == 2,
            "all_72_holds_have_timestamp_edge_coverage": len(clip_reports) == 2
            and all(
                summary["interval_coverage_passed"]
                for report in clip_reports
                for summary in report["hold_summaries"]
            ),
            "exactly_one_material_chart_anchor_per_hold": len(clip_reports) == 2
            and sum(len(report["material_chart_anchor_records"]) for report in clip_reports) == 72,
            "directional_fixed_camera_gates_passed": len(clip_reports) == 2
            and all(
                report["physical_camera_verdict"] == "fixed_to_subpixel_precision"
                for report in clip_reports
            ),
            "cross_direction_zero_degree_fixed_camera_gate_passed": cross_verdict
            == "fixed_to_subpixel_precision",
            "background_audits_repeatable": cross_repeatable
            and len(clip_reports) == 2
            and all(report["background_audit_repeatable"] for report in clip_reports),
            "exposure_and_focus_stability_diagnostics_recorded": (stability_diagnostics_recorded),
            "photometric_geometry_policy_satisfied": (
                isinstance(declaration.camera_settings, ManualCameraSettings)
                or declaration.camera_settings.photometric_geometry_allowed is False
            ),
            "synchronized_evaluator_present_and_excluded_from_fitting": (
                declaration.evaluator_camera is not None
                and manifest["evaluator_camera"] is not None
                and source_audit.get("evaluator_camera") is not None
                and source_audit["evaluator_camera"].get("stereo_calibration") is not None
                and source_audit["evaluator_camera"]["timing_summary"]["strictly_monotonic"] is True
                and manifest["evaluator_camera"]["decoded_for_fitting"] is False
                and manifest["evaluator_camera"]["training_record_count"] == 0
            ),
            "capture_mode_evaluator_policy_satisfied": (
                declaration.capture_mode == "single_camera_evidence_consistent"
                and declaration.evaluator_camera is None
                and declaration.metric_accuracy_claim_allowed is False
            )
            or (
                declaration.capture_mode == "dual_camera_metric_evaluation"
                and declaration.evaluator_camera is not None
                and manifest["evaluator_camera"] is not None
                and manifest["evaluator_camera"]["decoded_for_fitting"] is False
            ),
            "sealed_and_development_evidence_excluded": True,
        }
        write_json(
            stage / "qualification.json",
            {
                "schema_version": "frayid_v3_v01_qualification.v1",
                "experiment_id": EXPERIMENT_ID,
                "capture_id": declaration.capture_id,
                "source_revision": source_revision,
                "status": manifest["status"],
                "blockers": blockers,
                "checks": checks,
                "evidence_master_sha256": sha256_file(manifest_path),
                "training_record_count": total_training_records,
                "development_records_read": 0,
                "sealed_test_accesses": 0,
                "optimizer_steps": 0,
                "paid_jobs": 0,
                "automatic_retries": 0,
            },
        )
        os.replace(stage, output_root)
        return output_root / "evidence_master.json"
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "CalibratedIntrinsics",
    "ControlledCaptureDeclaration",
    "DeviceManagedCameraSettings",
    "EvaluatorStereoCalibration",
    "TrainingCameraCalibration",
    "audit_controlled_capture_sources",
    "build_controlled_capture_evidence_master",
    "controlled_calibration_templates",
    "controlled_capture_template",
    "controlled_single_camera_capture_template",
    "validate_controlled_capture_declaration",
]
