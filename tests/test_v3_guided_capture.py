from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import frayid.v3.guided_capture as guided_capture_module
from frayid.io import read_json, sha256_file
from frayid.v3.controlled_camera_calibration import (
    ControlledCharucoBoardSpec,
    create_controlled_charuco_board,
)
from frayid.v3.guided_capture import (
    BoardObservation,
    GuidedCalibrationController,
    create_guided_display_board_kit,
    detect_charuco_board,
    ffmpeg_rotation_capture_commands,
    ffplay_training_camera_preview_command,
    guided_calibration_targets,
    render_guided_calibration_frame,
    render_guided_rotation_frame,
    render_training_camera_preview,
    rotation_cue_state,
    rotation_spoken_phrase,
    run_guided_calibration_capture,
)


def _observation(
    quad: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    *,
    corners: int = 24,
) -> BoardObservation:
    values = np.asarray(quad, dtype=np.float64)
    return BoardObservation(
        corner_count=corners,
        quad_normalized=quad,
        center_normalized=tuple(values.mean(axis=0)),
        area_fraction=0.15,
        sharpness=500.0,
    )


def test_guided_targets_are_complete_visible_and_unique() -> None:
    targets = guided_calibration_targets()
    assert len(targets) == 12
    assert len({target.target_id for target in targets}) == len(targets)
    for target in targets:
        quad = np.asarray(target.quad_normalized)
        assert quad.shape == (4, 2)
        assert np.all((quad >= 0.05) & (quad <= 0.95))
        assert cv2.contourArea(quad.astype(np.float32)) > 0.04


def test_guided_controller_requires_detection_match_corners_and_stability() -> None:
    targets = guided_calibration_targets()[:1]
    controller = GuidedCalibrationController(
        targets=targets,
        stable_seconds=0.8,
        target_tolerance=0.02,
    )
    assert controller.update(0.0, None).status == "board_not_found"
    partial = _observation(targets[0].quad_normalized, corners=8)
    assert controller.update(0.1, partial).status == "more_corners_needed"
    wrong_quad = tuple((x + 0.2, y) for x, y in targets[0].quad_normalized)
    assert controller.update(0.2, _observation(wrong_quad)).status == "match_green_outline"
    assert controller.update(1.0, _observation(targets[0].quad_normalized)).status == "hold_still"
    assert controller.update(1.4, _observation(targets[0].quad_normalized)).status == "hold_still"
    accepted = controller.update(1.81, _observation(targets[0].quad_normalized))
    assert accepted.status == "accepted"
    assert accepted.accepted is True
    assert accepted.complete is True
    assert controller.update(2.0, _observation(targets[0].quad_normalized)).status == "complete"


def test_controller_resets_stability_when_board_moves() -> None:
    target = guided_calibration_targets()[0]
    controller = GuidedCalibrationController(targets=(target,), stable_seconds=0.8)
    base = np.asarray(target.quad_normalized)
    shifted = tuple(tuple(point) for point in base + np.asarray([0.02, 0.0]))
    assert controller.update(0.0, _observation(target.quad_normalized)).status == "hold_still"
    result = controller.update(0.5, _observation(shifted))
    assert result.status == "hold_still"
    assert result.stable_seconds == 0.0


def test_charuco_detection_recovers_projected_board_outline(tmp_path: Path) -> None:
    spec_path = create_controlled_charuco_board(tmp_path / "board")
    spec = ControlledCharucoBoardSpec.model_validate(read_json(spec_path))
    source = cv2.imread(spec.board_image_path, cv2.IMREAD_COLOR)
    assert source is not None
    source_quad = np.asarray(
        [
            [0.0, 0.0],
            [source.shape[1] - 1.0, 0.0],
            [source.shape[1] - 1.0, source.shape[0] - 1.0],
            [0.0, source.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    target_quad = np.asarray(
        [[360.0, 180.0], [1220.0, 220.0], [1160.0, 840.0], [410.0, 800.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_quad, target_quad)
    frame = cv2.warpPerspective(source, homography, (1600, 1000), borderValue=(255, 255, 255))
    detected = detect_charuco_board(frame, spec)
    assert detected is not None
    assert detected.observation.corner_count >= 20
    np.testing.assert_allclose(detected.quad_pixels, target_quad, atol=6.0)
    assert detected.observation.area_fraction > 0.25


def test_renderers_are_preview_only_and_preserve_input_pixels() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    controller = GuidedCalibrationController(targets=guided_calibration_targets()[:1])
    result = controller.update(0.0, None)
    guidance = render_guided_calibration_frame(frame, result, None)
    framing = render_training_camera_preview(frame)
    assert guidance.shape == frame.shape
    assert framing.shape == frame.shape
    assert np.any(guidance != 0)
    assert np.any(framing != 0)
    np.testing.assert_array_equal(frame, original)


def test_display_board_locks_orientation_before_measurement_and_is_immutable(
    tmp_path: Path,
) -> None:
    spec_path = create_controlled_charuco_board(tmp_path / "board")
    output_root = tmp_path / "display-kit"
    manifest_path = create_guided_display_board_kit(spec_path, output_root)
    manifest = read_json(manifest_path)
    html_path = Path(manifest["html_path"])
    html = html_path.read_text(encoding="utf-8")
    assert manifest["orientation_lock_requested"] == "landscape"
    assert manifest["measurement_timing"] == "after_fullscreen_and_orientation_lock"
    assert manifest["network_dependencies"] == []
    assert "screen.orientation.lock('landscape')" in html
    assert "maximum-scale=1" in html
    assert manifest["html_sha256"] == sha256_file(html_path)
    with pytest.raises(FileExistsError, match="immutable"):
        create_guided_display_board_kit(spec_path, output_root)


def test_live_guided_calibration_cannot_disable_its_preview(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot disable its safety preview"):
        run_guided_calibration_capture(
            board_spec_path=tmp_path / "missing.json",
            output_root=tmp_path / "capture",
            camera_index=0,
            display=False,
        )


def test_rotation_cue_state_keeps_preview_schedule_explicit() -> None:
    assert rotation_cue_state(None, "clockwise").phase == "ready"
    countdown = rotation_cue_state(2.0, "clockwise")
    assert countdown.phase == "countdown"
    assert countdown.seconds_remaining == 13.0
    first_hold = rotation_cue_state(15.0, "clockwise")
    assert first_hold.phase == "hold"
    assert first_hold.target_angle_degrees == 0
    assert rotation_cue_state(18.01, "clockwise").phase == "turn"
    second_hold = rotation_cue_state(21.0, "clockwise")
    assert second_hold.phase == "hold"
    assert second_hold.target_angle_degrees == 10
    assert rotation_cue_state(21.0, "counter_clockwise").target_angle_degrees == 350
    complete = rotation_cue_state(231.0, "clockwise")
    assert complete.phase == "complete"
    assert complete.angle_role == "proposal_not_measurement"


def test_rotation_preview_is_mirrored_but_never_mutates_saved_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :50] = (20, 40, 80)
    original = frame.copy()
    state = rotation_cue_state(15.0, "clockwise")
    preview = render_guided_rotation_frame(frame, state, recording=True)
    assert preview.shape == frame.shape
    assert np.any(preview != frame)
    np.testing.assert_array_equal(frame, original)


def test_upper_garment_preview_marks_neck_armholes_and_complete_hem_region() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    full_body = render_training_camera_preview(frame, framing="full_body", mirrored=False)
    upper = render_training_camera_preview(
        frame,
        framing="upper_garment_complete",
        mirrored=False,
    )
    assert np.any(upper != full_body)
    assert np.count_nonzero(upper) > np.count_nonzero(full_body)


def test_ffmpeg_rotation_pipeline_separates_lossless_evidence_from_preview(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clockwise.mkv"
    ffmpeg, ffplay = ffmpeg_rotation_capture_commands(
        video_path=video,
        direction="clockwise",
        camera_index=0,
        width=1920,
        height=1080,
        fps=30.0,
        framing="upper_garment_complete",
        duration_seconds=155.0,
    )
    graph = ffmpeg[ffmpeg.index("-filter_complex") + 1]
    assert "[record]format=yuv422p[record_out]" in graph
    assert "[preview]hflip,scale=1280:720" in graph
    assert "drawbox" in graph
    assert ffmpeg[ffmpeg.index("-c:v") + 1] == "ffv1"
    assert str(video) in ffmpeg
    assert "mjpeg" in ffmpeg
    assert "-alwaysontop" in ffplay
    assert "pipe:0" in ffplay


def test_zero_save_preview_is_direct_avfoundation_and_always_on_top() -> None:
    command = ffplay_training_camera_preview_command(
        camera_index=0,
        width=1920,
        height=1080,
        fps=30.0,
        framing="upper_garment_complete",
    )
    assert "avfoundation" in command
    assert "0:none" in command
    preview_filter = command[command.index("-vf") + 1]
    assert "hflip,scale=1280:720" in preview_filter
    assert "drawbox" in preview_filter
    assert "-alwaysontop" in command
    assert not any(suffix in command for suffix in (".mkv", ".mov", ".mp4"))


def test_rotation_speech_stops_previous_process_and_uses_short_cues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_process = object()
    stopped: list[object] = []
    launched: list[list[str]] = []
    replacement_process = object()
    monkeypatch.setattr(
        guided_capture_module,
        "_stop_process",
        lambda process: stopped.append(process),
    )
    monkeypatch.setattr(
        guided_capture_module.subprocess,
        "Popen",
        lambda command, **_kwargs: launched.append(command) or replacement_process,
    )
    state = rotation_cue_state(15.0, "clockwise")
    key, process = guided_capture_module._speak_rotation_transition(
        state,
        None,
        previous_process,  # type: ignore[arg-type]
    )
    assert key == ("hold", 0)
    assert stopped == [previous_process]
    assert launched[0][-1] == "停。正面。保持三秒。"
    assert process is replacement_process

    same_key, same_process = guided_capture_module._speak_rotation_transition(
        state,
        key,
        process,  # type: ignore[arg-type]
    )
    assert same_key == key
    assert same_process is process
    assert len(launched) == 1


def test_rotation_speech_uses_landmarks_and_never_demands_numeric_angles() -> None:
    samples = [
        rotation_cue_state(0.0, "clockwise"),
        rotation_cue_state(15.0, "clockwise"),
        rotation_cue_state(18.0, "clockwise"),
        rotation_cue_state(69.0, "clockwise"),
        rotation_cue_state(123.0, "clockwise"),
        rotation_cue_state(177.0, "clockwise"),
        rotation_cue_state(231.0, "clockwise"),
    ]
    phrases = [rotation_spoken_phrase(state) for state in samples]
    assert any("四分之一圈" in phrase for phrase in phrases)
    assert any("半圈" in phrase for phrase in phrases)
    assert any("四分之三圈" in phrase for phrase in phrases)
    assert any("右手" in phrase for phrase in phrases)
    assert all("10度" not in phrase and "20度" not in phrase for phrase in phrases)
