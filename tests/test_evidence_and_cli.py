from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from frayid.cli import app
from frayid.config import load_config
from frayid.dataset import validate_dataset
from frayid.evidence import (
    build_observed_pose_sequence,
    build_sequence_initialization,
    write_sapiens2_evidence,
)
from frayid.io import write_json
from frayid.schemas import (
    CameraHMRFrame,
    DatasetManifest,
    FrameRecord,
    ObservedPoseFrame,
    ObservedPoseSequence,
    SequenceInitialization,
    VideoMetadata,
)


def test_cli_exposes_only_v1_surface() -> None:
    runner = CliRunner()
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    for name in ("doctor", "assets", "dataset", "initialize", "reconstruct"):
        assert name in root.stdout
    assert "measure" not in root.stdout
    reconstruct = runner.invoke(app, ["reconstruct", "--help"])
    assert reconstruct.exit_code == 0
    for name in ("smoke", "plan-modal", "evaluate"):
        assert name in reconstruct.stdout


def test_dataset_validation_rejects_proxy_evidence(tmp_path: Path) -> None:
    config = load_config()
    dataset_root = tmp_path / "dataset"
    image_root = dataset_root / "images"
    mask_root = dataset_root / "masks"
    normal_root = dataset_root / "normals"
    for path in (image_root, mask_root, normal_root):
        path.mkdir(parents=True)
    records = []
    for ordinal in range(2):
        filename = f"frame_{ordinal:04d}_source_{ordinal:06d}.png"
        image = np.full((24, 16, 3), 100, dtype=np.uint8)
        mask = np.zeros((24, 16), dtype=np.uint8)
        mask[4:20, 4:12] = 255
        cv2.imwrite(str(image_root / filename), image)
        cv2.imwrite(str(mask_root / filename), mask)
        cv2.imwrite(str(normal_root / filename), np.full_like(image, 127))
        records.append(
            FrameRecord(
                ordinal=ordinal,
                source_frame_index=ordinal,
                timestamp_seconds=float(ordinal),
                image_path=str(image_root / filename),
                split="held_out" if ordinal == 0 else "train",
                blur_variance=100.0,
                mean_luminance=100.0,
                quality_accepted=True,
            )
        )
    manifest = DatasetManifest(
        status="rgb_ready",
        run_id="test",
        input_video_path="local.mp4",
        input_video_sha256="a" * 64,
        video=VideoMetadata(
            path="local.mp4",
            codec="h264",
            width=16,
            height=24,
            frame_count=2,
            frame_rate=1.0,
            duration_seconds=2.0,
            size_bytes=1,
        ),
        dataset_root=str(dataset_root),
        frames=records,
        train_frame_count=1,
        held_out_frame_count=1,
        rejected_candidate_count=0,
    )
    write_json(dataset_root / "dataset_manifest.json", manifest)
    local_config = config.model_copy(
        update={
            "paths": config.paths.model_copy(update={"dataset_root": dataset_root}),
            "dataset": config.dataset.model_copy(
                update={"minimum_usable_frame_count": 2, "target_frame_count": 2}
            ),
        }
    )
    report = validate_dataset(local_config)
    assert report.status == "blocked"
    assert "evidence_complete_frame_count_below_minimum" in report.blockers

    for ordinal in range(2):
        filename = f"frame_{ordinal:04d}_source_{ordinal:06d}.png"
        normal = np.zeros((24, 16, 3), dtype=np.uint8)
        normal[..., 0] = np.arange(16, dtype=np.uint8)[None, :] * 8
        normal[..., 1] = 127
        normal[..., 2] = 255 - normal[..., 0]
        cv2.imwrite(str(normal_root / filename), normal)
    frames = [
        CameraHMRFrame(
            source_frame_index=index,
            betas=[0.0] * 10,
            body_pose=[0.01] + [0.0] * 68,
            global_orient=[0.0, 0.1 * (index + 1), 0.0],
            translation=[0.0, 0.0, 3.0],
            focal_length_px=500.0,
            principal_point_px=[8.0, 12.0],
            keypoints_2d=[[8.0, 12.0, 1.0]],
            bounding_box_xyxy=[2.0, 2.0, 14.0, 22.0],
            detection_score=0.99,
        )
        for index in range(2)
    ]
    initialization = SequenceInitialization(
        status="raw",
        shared_betas=[0.0] * 10,
        shared_focal_length_px=500.0,
        shared_principal_point_px=[8.0, 12.0],
        image_width=16,
        image_height=24,
        frames=frames,
        source_revision="test-revision",
        checkpoint_sha256="b" * 64,
        camera_checkpoint_sha256="c" * 64,
        detector_checkpoint_sha256="d" * 64,
    )
    write_json(dataset_root / "sequence_initialization.json", initialization)
    write_json(
        dataset_root / "sapiens2_pose_sequence.json",
        ObservedPoseSequence(
            image_width=16,
            image_height=24,
            frames=[
                ObservedPoseFrame(
                    source_frame_index=index,
                    keypoints_body12=[[8.0, 12.0, 0.9] for _ in range(12)],
                    bounding_box_xyxy=[2.0, 2.0, 14.0, 22.0],
                )
                for index in range(2)
            ],
            source_revision="source-revision",
            model_revision="model-revision",
            detector_revision="detector-revision",
            checkpoint_sha256="e" * 64,
            detector_checkpoint_sha256="f" * 64,
        ),
    )
    ready = validate_dataset(local_config)
    assert ready.status == "ready", json.dumps(ready.model_dump(), indent=2)


def test_sapiens2_arrays_are_encoded_with_camera_xyz_contract(tmp_path: Path) -> None:
    labels = np.zeros((3, 4), dtype=np.int64)
    labels[1:, 1:3] = 1
    normals = np.zeros((3, 4, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    labels_path = tmp_path / "labels.npy"
    normals_path = tmp_path / "normals.npy"
    mask_path = tmp_path / "mask.png"
    normal_path = tmp_path / "normal.png"
    np.save(labels_path, labels)
    np.save(normals_path, normals)

    write_sapiens2_evidence(
        labels_path=labels_path,
        normals_path=normals_path,
        mask_output_path=mask_path,
        normal_output_path=normal_path,
        expected_size=(3, 4),
    )

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    encoded_bgr = cv2.imread(str(normal_path), cv2.IMREAD_COLOR)
    assert mask is not None and encoded_bgr is not None
    assert mask[1, 1] == 255
    np.testing.assert_array_equal(encoded_bgr[1, 1], np.asarray([255, 128, 128]))
    np.testing.assert_array_equal(encoded_bgr[0, 0], np.asarray([0, 0, 0]))


def test_camerahmr_frames_consolidate_shared_sequence_values(tmp_path: Path) -> None:
    records = [
        FrameRecord(
            ordinal=index,
            source_frame_index=10 + index,
            timestamp_seconds=float(index),
            image_path=f"frame_{index}.png",
            split="train",
            blur_variance=100.0,
            mean_luminance=100.0,
            quality_accepted=True,
        )
        for index in range(2)
    ]
    raw_frames = [
        {
            "source_frame_index": 10 + index,
            "betas": [float(index)] * 10,
            "body_pose": [0.01] + [0.0] * 68,
            "global_orient": [0.0, 0.1, 0.0],
            "translation": [0.0, 0.0, 3.0],
            "focal_length_px": 500.0 + index * 20.0,
            "principal_point_px": [8.0, 12.0],
            "keypoints_2d": [[8.0, 12.0, 1.0]],
            "joints_3d": [[0.0, 0.0, 0.0]],
            "bounding_box_xyxy": [2.0, 2.0, 14.0, 22.0],
            "detection_score": 0.99,
        }
        for index in range(2)
    ]
    output = tmp_path / "sequence_initialization.json"
    result = build_sequence_initialization(
        frame_records=records,
        raw_frames=raw_frames,
        image_width=16,
        image_height=24,
        source_revision="abc123",
        checkpoint_sha256="a" * 64,
        camera_checkpoint_sha256="b" * 64,
        detector_checkpoint_sha256="c" * 64,
        output_path=output,
    )
    assert output.is_file()
    assert result.shared_betas == [0.5] * 10
    assert result.shared_focal_length_px == 510.0


def test_sapiens2_pose_is_consolidated_as_independent_observation(tmp_path: Path) -> None:
    records = [
        FrameRecord(
            ordinal=index,
            source_frame_index=10 + index,
            timestamp_seconds=float(index),
            image_path=f"frame_{index}.png",
            split="train",
            blur_variance=100.0,
            mean_luminance=100.0,
            quality_accepted=True,
        )
        for index in range(2)
    ]
    raw_frames = []
    for index in range(2):
        raw_frames.append(
            {
                "image_name": f"frame_{index}.png",
                "instances": [
                    {
                        "bbox": [1.0, 2.0, 9.0, 20.0],
                        "keypoints": [[float(k), float(k + 1)] for k in range(308)],
                        "keypoint_scores": [0.8] * 5 + [1.02] + [0.8] * 302,
                    }
                ],
            }
        )
    output = tmp_path / "sapiens2_pose_sequence.json"
    sequence = build_observed_pose_sequence(
        frame_records=records,
        raw_payload={"frames": raw_frames},
        image_width=16,
        image_height=24,
        source_revision="source",
        model_revision="model",
        detector_revision="detector",
        checkpoint_sha256="a" * 64,
        detector_checkpoint_sha256="b" * 64,
        output_path=output,
    )
    assert output.is_file()
    assert [frame.source_frame_index for frame in sequence.frames] == [10, 11]
    assert sequence.frames[0].keypoints_body12[0][2] == 1.0
    assert sequence.frames[0].keypoints_body12[4] == [62.0, 63.0, 0.8]
    assert sequence.frames[0].keypoints_body12[5] == [41.0, 42.0, 0.8]
