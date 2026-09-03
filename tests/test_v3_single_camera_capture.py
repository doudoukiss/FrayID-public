from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import frayid.v3.controlled_camera_calibration as calibration_module
import frayid.v3.controlled_capture as capture_module
import frayid.v3.controlled_capture_cues as cue_module
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.video_forensics import FrameTimestamp, VideoProbe
from frayid.v3.controlled_camera_calibration import (
    ControlledTrainingCalibrationSession,
    calibrate_controlled_training_camera,
    controlled_training_calibration_session_template,
    create_controlled_charuco_board,
)
from frayid.v3.controlled_capture import (
    CalibratedIntrinsics,
    ControlledCaptureDeclaration,
    TrainingCameraCalibration,
    audit_controlled_capture_sources,
    build_controlled_capture_evidence_master,
    controlled_single_camera_capture_template,
    validate_controlled_capture_declaration,
)
from frayid.v3.controlled_capture_cues import (
    ControlledCaptureCueManifest,
    CueEvent,
    DirectionCue,
    create_controlled_capture_cue_kit,
    detect_controlled_capture_cues,
)


def _training_calibration(width: int, height: int) -> TrainingCameraCalibration:
    return TrainingCameraCalibration(
        intrinsics=CalibratedIntrinsics(
            image_width=width,
            image_height=height,
            camera_matrix=[
                [900.0, 0.0, width / 2.0],
                [0.0, 900.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            distortion_coefficients=[0.0] * 5,
            reprojection_rms_pixels=0.25,
            calibration_image_count=12,
            fiducial_layout_sha256="b" * 64,
            fiducial_square_size_m=0.04,
        ),
        world_from_camera=np.eye(4).tolist(),
        rotation_axis_origin_world_m=(0.0, 0.0, 0.0),
        rotation_axis_direction_world=(0.0, 0.0, 1.0),
        rotation_axis_registration_method=("known_scale_floor_fiducial_and_vertical_gravity"),
    )


def test_single_camera_declaration_is_explicitly_nonmetric_and_fail_closed() -> None:
    template = controlled_single_camera_capture_template()
    declaration = ControlledCaptureDeclaration.model_validate(template)
    report = validate_controlled_capture_declaration(template)
    assert declaration.schema_version == "frayid_v3_controlled_capture_declaration.v3"
    assert declaration.capture_mode == "single_camera_evidence_consistent"
    assert declaration.evaluator_camera is None
    assert declaration.metric_accuracy_claim_allowed is False
    assert report["status"] == "ready_for_physical_capture"
    assert report["manual_camera_controls"] is False
    assert report["camera_control_mode"] == "device_managed_control_unavailable"
    assert report["scientific_claim_ceiling"] == "evidence_consistent_mantle_reconstruction"

    injected_evaluator = controlled_single_camera_capture_template()
    injected_evaluator["evaluator_camera"] = {
        "path": "data/private/postv3_v01/not-independent.mov",
    }
    with pytest.raises(ValidationError, match="cannot declare an independent evaluator"):
        ControlledCaptureDeclaration.model_validate(injected_evaluator)

    missing_reason = controlled_single_camera_capture_template()
    missing_reason["independent_evaluator_unavailable_reason"] = None
    with pytest.raises(ValidationError, match="evaluator-unavailable reason"):
        ControlledCaptureDeclaration.model_validate(missing_reason)

    false_manual_claim = controlled_single_camera_capture_template()
    false_manual_claim["camera_settings"] = {
        "exposure_mode": "manual",
        "focus_mode": "manual",
        "white_balance_mode": "manual",
        "shutter_seconds": None,
        "iso": None,
        "focal_length_mm": None,
        "electronic_stabilization": False,
        "optical_stabilization": False,
        "auto_framing": False,
        "dynamic_hdr": False,
    }
    with pytest.raises(ValidationError):
        ControlledCaptureDeclaration.model_validate(false_manual_claim)


def test_single_camera_source_audit_needs_no_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clockwise = tmp_path / "clockwise.native"
    counter_clockwise = tmp_path / "counter-clockwise.native"
    clockwise.write_bytes(b"clockwise")
    counter_clockwise.write_bytes(b"counter-clockwise")
    calibration_path = write_json(
        tmp_path / "training-calibration.json",
        _training_calibration(1920, 1080).model_dump(mode="json"),
    )
    template = controlled_single_camera_capture_template()
    template["training_clips"][0]["path"] = str(clockwise)
    template["training_clips"][0]["expected_sha256"] = sha256_file(clockwise)
    template["training_clips"][1]["path"] = str(counter_clockwise)
    template["training_clips"][1]["expected_sha256"] = sha256_file(counter_clockwise)
    template["training_camera_calibration"] = {
        "path": str(calibration_path),
        "expected_sha256": sha256_file(calibration_path),
    }
    declaration_path = write_json(tmp_path / "declaration.json", template)
    timestamps = [
        FrameTimestamp(
            decode_index=index,
            pts_seconds=index / 30.0,
            selected_timestamp_seconds=index / 30.0,
            selected_timestamp_source="pts",
        )
        for index in range(4441)
    ]
    probe = VideoProbe(
        codec="fixture",
        pixel_format="rgb24",
        width=1920,
        height=1080,
        reported_frame_count=len(timestamps),
        duration_seconds=148.1,
        source_size_bytes=10,
    )
    monkeypatch.setattr(
        capture_module,
        "probe_video_forensics",
        lambda _: (probe, timestamps, {"command": ["fixture"]}),
    )
    audit = audit_controlled_capture_sources(declaration_path)
    assert audit["status"] == "pass"
    assert audit["blockers"] == []
    assert audit["evaluator_camera"] is None
    assert audit["metric_accuracy_claim_allowed"] is False
    assert audit["scientific_claim_ceiling"] == "evidence_consistent_mantle_reconstruction"


def test_single_camera_evidence_master_passes_without_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = controlled_single_camera_capture_template()
    clockwise = tmp_path / "clockwise.native"
    counter_clockwise = tmp_path / "counter-clockwise.native"
    clockwise.write_bytes(b"clockwise")
    counter_clockwise.write_bytes(b"counter-clockwise")
    template["training_clips"][0]["path"] = str(clockwise)
    template["training_clips"][1]["path"] = str(counter_clockwise)
    declaration_path = write_json(tmp_path / "declaration.json", template)
    timestamps: list[FrameTimestamp] = [
        FrameTimestamp(
            decode_index=0,
            pts_seconds=0.0,
            selected_timestamp_seconds=0.0,
            selected_timestamp_source="pts",
        )
    ]
    for hold_index in range(36):
        for offset in (3.01, 5.49):
            seconds = hold_index * 4.0 + offset
            timestamps.append(
                FrameTimestamp(
                    decode_index=len(timestamps),
                    pts_seconds=seconds,
                    selected_timestamp_seconds=seconds,
                    selected_timestamp_source="pts",
                )
            )
    probe = VideoProbe(
        codec="fixture",
        pixel_format="rgb24",
        width=1920,
        height=1080,
        reported_frame_count=len(timestamps),
        duration_seconds=148.0,
        source_size_bytes=10,
    )

    monkeypatch.setattr(
        capture_module,
        "audit_controlled_capture_sources",
        lambda _: {
            "status": "pass",
            "blockers": [],
            "training_camera_calibration": {
                "role": "measured_training_camera",
                "sha256": "e" * 64,
            },
            "evaluator_camera": None,
        },
    )
    monkeypatch.setattr(
        capture_module,
        "probe_video_forensics",
        lambda path, **_: (probe, timestamps, {"command": [str(path)]}),
    )

    def fake_frames(path: Path, **_: Any) -> Any:
        rng = np.random.default_rng(7 if path == clockwise else 8)
        for _timestamp in timestamps:
            yield rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(capture_module, "iter_sequential_rgb_frames", fake_frames)
    monkeypatch.setattr(capture_module, "executable_version", lambda _: "fixture")
    monkeypatch.setattr(
        capture_module,
        "_background_audit",
        lambda frames, indices: (
            {"sample_count": len(frames), "source_indices": indices},
            True,
            "fixed_to_subpixel_precision",
        ),
    )
    output_root = tmp_path / "master"
    manifest_path = build_controlled_capture_evidence_master(
        declaration_path,
        output_root,
        source_revision="a" * 40,
        proxy_size=(16, 16),
    )
    manifest = read_json(manifest_path)
    qualification = read_json(output_root / "qualification.json")
    assert manifest["status"] == "pass"
    assert manifest["capture_mode"] == "single_camera_evidence_consistent"
    assert manifest["evaluator_camera"] is None
    assert manifest["metric_accuracy_claim_allowed"] is False
    assert qualification["checks"]["capture_mode_evaluator_policy_satisfied"] is True
    assert (
        qualification["checks"]["synchronized_evaluator_present_and_excluded_from_fitting"] is False
    )


def test_training_only_calibration_template_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = create_controlled_charuco_board(tmp_path / "board")
    template = controlled_training_calibration_session_template(spec_path)
    assert template["capture_mode"] == "single_camera_evidence_consistent"
    assert template["independent_evaluator_available"] is False
    assert "evaluator_intrinsic_images" not in template
    with pytest.raises(ValidationError, match="template_only"):
        ControlledTrainingCalibrationSession.model_validate(template)

    sources = []
    for index in range(12):
        source = tmp_path / f"intrinsic-{index:02d}.png"
        source.write_bytes(f"view-{index}".encode())
        sources.append(source)
    setup = tmp_path / "setup.png"
    setup.write_bytes(b"setup")
    template["template_only"] = False
    template["printed_square_measurements_m"] = [0.04, 0.0401, 0.0399]
    template["training_intrinsic_images"] = [str(path) for path in sources]
    template["training_setup_image"] = str(setup)
    session_path = write_json(tmp_path / "training-session.json", template)
    calibration = _training_calibration(1920, 1080).model_dump(mode="json")
    diagnostics = {
        "training_intrinsics": {"rms_pixels": 0.25},
        "training_setup": {"rms_pixels": 0.2, "corner_count": 20},
    }
    monkeypatch.setattr(
        calibration_module,
        "_solve_training_session",
        lambda *_: (calibration, diagnostics),
    )
    monkeypatch.setattr(
        calibration_module,
        "_opencv_runtime",
        lambda: {"opencv": "fixture", "numpy": "fixture", "opencv_threads": 1},
    )
    output_root = tmp_path / "training-calibration"
    report_path = calibrate_controlled_training_camera(session_path, output_root)
    report = read_json(report_path)
    assert report["status"] == "pass"
    assert report["exact_same_input_replay"] is True
    assert report["independent_evaluator_available"] is False
    assert report["evaluator_files_read"] == 0
    assert report["metric_accuracy_claim_allowed"] is False
    assert TrainingCameraCalibration.model_validate(
        read_json(output_root / "training_camera_calibration.json")
    )


def test_single_camera_cue_detection_binds_holds_without_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cue_audio = {}
    directions = []
    for direction in ("clockwise", "counter_clockwise"):
        path = tmp_path / f"{direction}.wav"
        path.write_bytes(direction.encode())
        cue_audio[direction] = path
        directions.append(
            DirectionCue(
                direction=direction,
                audio_path=str(path),
                audio_sha256=sha256_file(path),
                holds=cue_module._holds(direction),
                synchronization_events=[
                    CueEvent(cue_seconds=seconds, frequency_hz=frequency)
                    for seconds, frequency in zip(
                        cue_module.SYNC_EVENT_SECONDS,
                        cue_module.SYNC_FREQUENCIES[direction],
                        strict=True,
                    )
                ],
            )
        )
    html = tmp_path / "capture_cue.html"
    html.write_text("fixture", encoding="utf-8")
    manifest_path = write_json(
        tmp_path / "cue-manifest.json",
        ControlledCaptureCueManifest(
            directions=directions,
            html_path=str(html),
            html_sha256=sha256_file(html),
        ).model_dump(mode="json"),
    )
    clockwise = tmp_path / "clockwise.mov"
    counter_clockwise = tmp_path / "counter-clockwise.mov"
    clockwise.write_bytes(b"clockwise-video")
    counter_clockwise.write_bytes(b"counter-clockwise-video")

    def fake_decode(path: Path, *, sample_rate: int) -> np.ndarray:
        marker = 1.0 if path == clockwise else 2.0
        return np.full(sample_rate, marker, dtype=np.float64)

    def fake_events(
        audio: np.ndarray,
        direction: cue_module.Direction,
        *,
        sample_rate: int,
    ) -> list[dict[str, float]]:
        assert sample_rate == cue_module.DETECTION_SAMPLE_RATE
        offset = 0.2 if float(audio[0]) == 1.0 else 0.35
        return [
            {
                "frequency_hz": frequency,
                "video_seconds": seconds + offset,
                "snr_db": 30.0,
            }
            for seconds, frequency in zip(
                cue_module.SYNC_EVENT_SECONDS,
                cue_module.SYNC_FREQUENCIES[direction],
                strict=True,
            )
        ]

    monkeypatch.setattr(cue_module, "_decode_audio", fake_decode)
    monkeypatch.setattr(cue_module, "_events_for_direction", fake_events)
    output_path = tmp_path / "cue-detection.json"
    detect_controlled_capture_cues(
        cue_manifest_path=manifest_path,
        clockwise_path=clockwise,
        counter_clockwise_path=counter_clockwise,
        evaluator_path=None,
        output_path=output_path,
    )
    report = read_json(output_path)
    assert report["status"] == "pass"
    assert report["capture_mode"] == "single_camera_evidence_consistent"
    assert report["independent_evaluator_available"] is False
    assert report["evaluator_audio_streams_read_for_sync_only"] == 0
    assert report["metric_accuracy_blockers"] == ["independent_evaluator_camera_unavailable"]
    assert all(len(record["holds"]) == 36 for record in report["directions"])
    assert report["directions"][0]["holds"][0]["stable_start_seconds"] == pytest.approx(3.2)
    bound = controlled_single_camera_capture_template(output_path)
    assert bound["training_clips"][0]["expected_sha256"] == sha256_file(clockwise)
    assert bound["training_clips"][1]["expected_sha256"] == sha256_file(counter_clockwise)


def test_controlled_cue_kit_is_deterministic_and_immutable(tmp_path: Path) -> None:
    output_root = tmp_path / "cue-kit"
    manifest_path = create_controlled_capture_cue_kit(output_root)
    manifest = ControlledCaptureCueManifest.model_validate(read_json(manifest_path))
    assert manifest.duration_seconds == 148.0
    assert [item.direction for item in manifest.directions] == [
        "clockwise",
        "counter_clockwise",
    ]
    assert all(Path(item.audio_path).stat().st_size > 14_000_000 for item in manifest.directions)
    assert "Start Mac recording first" in Path(manifest.html_path).read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        create_controlled_capture_cue_kit(output_root)
