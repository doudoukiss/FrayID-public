from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import frayid.v3.controlled_capture as controlled_capture_module
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.video_forensics import FrameTimestamp, VideoProbe
from frayid.v3.controlled_camera_calibration import (
    ControlledCalibrationSession,
    ControlledCharucoBoardSpec,
    SynchronizationEvents,
    _fit_synchronization,
    calibrate_controlled_cameras,
    controlled_calibration_session_template,
    create_controlled_charuco_board,
)
from frayid.v3.controlled_capture import (
    EvaluatorStereoCalibration,
    TrainingCameraCalibration,
    audit_controlled_capture_sources,
    controlled_capture_template,
)


def _render_board(
    board_image: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    board_width_m: float,
    board_height_m: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    object_corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [board_width_m, 0.0, 0.0],
            [board_width_m, board_height_m, 0.0],
            [0.0, board_height_m, 0.0],
        ],
        dtype=np.float32,
    )
    destination, _ = cv2.projectPoints(
        object_corners,
        rvec,
        tvec,
        camera_matrix,
        np.zeros(5),
    )
    source = np.asarray(
        [
            [0.0, 0.0],
            [board_image.shape[1] - 1.0, 0.0],
            [board_image.shape[1] - 1.0, board_image.shape[0] - 1.0],
            [0.0, board_image.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        source,
        destination.reshape(4, 2).astype(np.float32),
    )
    width, height = image_size
    rendered = cv2.warpPerspective(
        board_image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)


def test_controlled_charuco_board_and_session_template_are_bound_and_immutable(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "board"
    spec_path = create_controlled_charuco_board(output_root)
    spec = ControlledCharucoBoardSpec.model_validate(read_json(spec_path))
    assert spec.printed_width_m == pytest.approx(0.28)
    assert spec.printed_height_m == pytest.approx(0.20)
    assert sha256_file(Path(spec.board_image_path)) == spec.board_image_sha256
    template = controlled_calibration_session_template(spec_path)
    assert template["board_spec_sha256"] == sha256_file(spec_path)
    assert template["template_only"] is True
    assert len(template["training_intrinsic_images"]) == 12
    assert len(template["evaluator_intrinsic_images"]) == 12
    assert len(template["stereo_pairs"]) == 12
    with pytest.raises(ValueError, match="template_only"):
        ControlledCalibrationSession.model_validate(template)
    with pytest.raises(FileExistsError, match="immutable"):
        create_controlled_charuco_board(output_root)


def test_controlled_camera_calibration_recovers_metric_intrinsics_stereo_and_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = create_controlled_charuco_board(tmp_path / "board")
    spec = ControlledCharucoBoardSpec.model_validate(read_json(spec_path))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )
    board_image = board.generateImage((1400, 1000), marginSize=0, borderBits=1)
    image_size = (1280, 720)
    training_matrix = np.asarray([[900.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]])
    evaluator_matrix = np.asarray([[920.0, 0.0, 638.0], [0.0, 915.0, 362.0], [0.0, 0.0, 1.0]])
    evaluator_from_training_rvec = np.asarray([0.01, -0.02, 0.005])
    evaluator_from_training_rotation, _ = cv2.Rodrigues(evaluator_from_training_rvec)
    evaluator_from_training_translation = np.asarray([-0.09, 0.004, 0.008])
    training_paths: list[Path] = []
    evaluator_paths: list[Path] = []
    for index in range(12):
        training_rvec = np.asarray(
            [
                0.15 * np.sin(index * 0.7),
                0.22 * np.cos(index * 0.5),
                0.05 * np.sin(index * 0.3),
            ]
        )
        training_rotation, _ = cv2.Rodrigues(training_rvec)
        training_translation = np.asarray(
            [-0.14 + 0.012 * index, -0.10 + 0.006 * index, 0.68 + 0.025 * (index % 3)]
        )
        evaluator_rotation = evaluator_from_training_rotation @ training_rotation
        evaluator_translation = (
            evaluator_from_training_rotation @ training_translation
            + evaluator_from_training_translation
        )
        evaluator_rvec, _ = cv2.Rodrigues(evaluator_rotation)
        training_image = _render_board(
            board_image,
            camera_matrix=training_matrix,
            rvec=training_rvec,
            tvec=training_translation,
            board_width_m=spec.printed_width_m,
            board_height_m=spec.printed_height_m,
            image_size=image_size,
        )
        evaluator_image = _render_board(
            board_image,
            camera_matrix=evaluator_matrix,
            rvec=evaluator_rvec,
            tvec=evaluator_translation,
            board_width_m=spec.printed_width_m,
            board_height_m=spec.printed_height_m,
            image_size=image_size,
        )
        training_path = tmp_path / f"training-{index:02d}.png"
        evaluator_path = tmp_path / f"evaluator-{index:02d}.png"
        assert cv2.imwrite(str(training_path), training_image)
        assert cv2.imwrite(str(evaluator_path), evaluator_image)
        training_paths.append(training_path)
        evaluator_paths.append(evaluator_path)

    synchronization = []
    for direction in ("clockwise", "counter_clockwise"):
        training_events = np.asarray([1.0, 31.0, 61.0, 91.0])
        evaluator_events = 1.0002 * training_events + 0.012
        evaluator_events += np.asarray([0.0, 0.0004, -0.0004, 0.0])
        synchronization.append(
            {
                "direction": direction,
                "method": "audible_visual_sync_event",
                "training_event_seconds": training_events.tolist(),
                "evaluator_event_seconds": evaluator_events.tolist(),
            }
        )
    session_path = write_json(
        tmp_path / "session.json",
        {
            "schema_version": "frayid_v3_controlled_calibration_session.v1",
            "experiment_id": "postv3_v01_controlled_recapture_evidence_master_r01",
            "board_spec_path": str(spec_path),
            "board_spec_sha256": sha256_file(spec_path),
            "template_only": False,
            "printed_square_measurements_m": [0.04, 0.0401, 0.0399],
            "training_intrinsic_images": [str(path) for path in training_paths],
            "evaluator_intrinsic_images": [str(path) for path in evaluator_paths],
            "stereo_pairs": [
                {
                    "training_image": str(training_path),
                    "evaluator_image": str(evaluator_path),
                }
                for training_path, evaluator_path in zip(
                    training_paths,
                    evaluator_paths,
                    strict=True,
                )
            ],
            "training_setup_image": str(training_paths[0]),
            "rotation_axis_origin_board_m": [0.14, 0.10, 0.0],
            "floor_board_verified_level": True,
            "rotation_axis_mark_verified": True,
            "vertical_axis_board_normal_sign": 1,
            "synchronization": synchronization,
        },
    )
    output_root = tmp_path / "calibration"
    report_path = calibrate_controlled_cameras(session_path, output_root)
    report = read_json(report_path)
    assert report["status"] == "pass"
    assert report["exact_same_input_replay"] is True
    assert len(report["source_inventory"]) == 24
    training = TrainingCameraCalibration.model_validate(
        read_json(output_root / "training_camera_calibration.json")
    )
    evaluator = EvaluatorStereoCalibration.model_validate(
        read_json(output_root / "evaluator_stereo_calibration.json")
    )
    observed_training_matrix = np.asarray(training.intrinsics.camera_matrix)
    assert observed_training_matrix[0, 0] == pytest.approx(training_matrix[0, 0], rel=0.05)
    assert observed_training_matrix[1, 1] == pytest.approx(training_matrix[1, 1], rel=0.05)
    expected_baseline = float(np.linalg.norm(evaluator_from_training_translation))
    assert evaluator.stereo_baseline_m == pytest.approx(expected_baseline, abs=0.01)
    assert training.rotation_axis_origin_world_m == pytest.approx((0.14, 0.10, 0.0))
    assert training.rotation_axis_direction_world == pytest.approx((0.0, 0.0, 1.0))
    assert max(item.synchronization_residual_ms for item in evaluator.synchronization) < 1.0

    clockwise = tmp_path / "clockwise.native"
    counter_clockwise = tmp_path / "counter-clockwise.native"
    evaluator_video = tmp_path / "evaluator.native"
    for path, payload in (
        (clockwise, b"clockwise"),
        (counter_clockwise, b"counter-clockwise"),
        (evaluator_video, b"evaluator"),
    ):
        path.write_bytes(payload)
    declaration = controlled_capture_template()
    declaration["width"] = image_size[0]
    declaration["height"] = image_size[1]
    declaration["below_4k_reason"] = "public calibration integration fixture"
    declaration["training_clips"][0]["path"] = str(clockwise)
    declaration["training_clips"][0]["expected_sha256"] = sha256_file(clockwise)
    declaration["training_clips"][1]["path"] = str(counter_clockwise)
    declaration["training_clips"][1]["expected_sha256"] = sha256_file(counter_clockwise)
    declaration["training_camera_calibration"] = {
        "path": str(output_root / "training_camera_calibration.json"),
        "expected_sha256": sha256_file(output_root / "training_camera_calibration.json"),
    }
    declaration["evaluator_camera"]["path"] = str(evaluator_video)
    declaration["evaluator_camera"]["expected_sha256"] = sha256_file(evaluator_video)
    declaration["evaluator_camera"]["stereo_calibration"] = {
        "path": str(output_root / "evaluator_stereo_calibration.json"),
        "expected_sha256": sha256_file(output_root / "evaluator_stereo_calibration.json"),
    }
    declaration_path = write_json(tmp_path / "capture-declaration.json", declaration)
    timestamps = [
        FrameTimestamp(
            decode_index=index,
            pts_seconds=index / 60.0,
            selected_timestamp_seconds=index / 60.0,
            selected_timestamp_source="pts",
        )
        for index in range(6481)
    ]
    probe = VideoProbe(
        codec="fixture",
        pixel_format="rgb24",
        width=image_size[0],
        height=image_size[1],
        reported_frame_count=len(timestamps),
        duration_seconds=108.1,
        source_size_bytes=10,
    )
    monkeypatch.setattr(
        controlled_capture_module,
        "probe_video_forensics",
        lambda _: (probe, timestamps, {"command": ["fixture"]}),
    )
    capture_audit = audit_controlled_capture_sources(declaration_path)
    assert capture_audit["status"] == "pass"
    assert capture_audit["training_camera_calibration"]["schema_version"].endswith(".v2")
    assert capture_audit["evaluator_camera"]["stereo_calibration"]["role"] == "evaluator_only"

    with pytest.raises(FileExistsError, match="immutable"):
        calibrate_controlled_cameras(session_path, output_root)


def test_controlled_sync_fit_rejects_unregistered_timing_error() -> None:
    events = SynchronizationEvents(
        direction="clockwise",
        method="audible_visual_sync_event",
        training_event_seconds=[0.0, 10.0, 20.0, 30.0],
        evaluator_event_seconds=[0.0, 10.0, 20.1, 30.0],
    )
    with pytest.raises(ValueError, match="exceeds 8 ms"):
        _fit_synchronization(events)
