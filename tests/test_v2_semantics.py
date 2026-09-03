from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from frayid.io import read_json, write_json
from frayid.v2.evidence import SAPIENS2_DOME29_LAYER_IDS
from frayid.v2.semantics import (
    audit_s01_qualification_lifecycle,
    qualify_sapiens2_semantic_directory,
)


def test_dome29_layer_mapping_does_not_confuse_right_hand_with_footwear() -> None:
    assert SAPIENS2_DOME29_LAYER_IDS["footwear"] == [9, 10, 18, 19]
    assert 15 not in SAPIENS2_DOME29_LAYER_IDS["footwear"]
    assert 15 in SAPIENS2_DOME29_LAYER_IDS["body_parts"]


def test_train_semantic_qualification_preserves_labels_and_confidence(tmp_path: Path) -> None:
    frame_count = 4
    manifest = {
        "schema_version": "canonical_dataset.v1",
        "status": "evidence_ready",
        "run_id": "semantic-test",
        "input_video_path": "local_input.mp4",
        "input_video_sha256": "0" * 64,
        "video": {
            "path": "local_input.mp4",
            "codec": "h264",
            "width": 8,
            "height": 8,
            "frame_count": 4,
            "frame_rate": 30.0,
            "duration_seconds": 1.0,
            "size_bytes": 100,
        },
        "dataset_root": "outputs/test/dataset",
        "frames": [
            {
                "ordinal": slot,
                "source_frame_index": slot,
                "timestamp_seconds": slot / 30.0,
                "image_path": f"images/frame_{slot:04d}.png",
                "split": "train",
                "blur_variance": 100.0,
                "mean_luminance": 128.0,
                "quality_accepted": True,
                "rejection_reasons": [],
            }
            for slot in range(frame_count)
        ],
        "train_frame_count": frame_count,
        "held_out_frame_count": 0,
        "rejected_candidate_count": 0,
        "blockers": [],
    }
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    mask_root = tmp_path / "masks"
    semantic_root = tmp_path / "semantics"
    mask_root.mkdir()
    semantic_root.mkdir()
    for slot in range(frame_count):
        labels = np.zeros((8, 8), dtype=np.uint8)
        labels[1:3, 1:3] = 23
        labels[3:5, 1:3] = 13
        labels[1:3, 3:5] = 4
        labels[3:5, 3:5] = 15
        labels[5:7, 2:4] = 9
        confidence = np.full((8, 8), 0.9, dtype=np.float16)
        np.savez_compressed(
            semantic_root / f"frame_{slot:04d}.npz",
            labels=labels,
            confidence=confidence,
            source_frame_index=np.asarray(slot, dtype=np.int64),
        )
        assert cv2.imwrite(
            str(mask_root / f"frame_{slot:04d}.png"),
            (labels > 0).astype(np.uint8) * 255,
        )
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"fixture")
    report_path = qualify_sapiens2_semantic_directory(
        manifest_path,
        mask_root,
        semantic_root,
        checkpoint,
        tmp_path / "qualification.json",
        source_revision="fixture-revision",
    )
    report = read_json(report_path)
    assert report["status"] == "pass"
    assert report["qualified_frame_count"] == 4
    assert report["foreground_iou_against_preserved_v1_masks"]["minimum"] == 1.0
    assert report["layer_pixel_counts"]["footwear"] > 0
    assert report["sealed_test_accesses"] == 0


def test_s01_lifecycle_advances_in_order_for_passing_real_shape_records(
    tmp_path: Path,
) -> None:
    extraction = write_json(
        tmp_path / "extraction.json",
        {
            "frame_count": 144,
            "device": "mps",
            "same_device_first_frame_replay_exact": True,
            "checkpoint_sha256": "a" * 64,
        },
    )
    qualification = write_json(
        tmp_path / "qualification.json",
        {
            "status": "pass",
            "qualified_frame_count": 144,
            "input_hashes": {"checkpoint": "a" * 64},
            "foreground_iou_against_preserved_v1_masks": {"minimum": 0.999},
            "legacy_development_images_read": 0,
            "sealed_test_accesses": 0,
            "modal_jobs": 0,
        },
    )
    lifecycle = read_json(
        audit_s01_qualification_lifecycle(
            extraction,
            qualification,
            tmp_path / "lifecycle.json",
        )
    )
    assert lifecycle["state"] == "qualified"
    assert [transition["to"] for transition in lifecycle["transitions"]] == [
        "imported",
        "data_bound",
        "device_validated",
        "one_step_passed",
        "checkpoint_restored",
        "evaluator_dry",
        "qualified",
    ]
