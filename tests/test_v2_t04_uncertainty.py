from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.evidence import bind_t04_hull_inputs
from frayid.v2.t04_uncertainty import (
    audit_t04_qualification_lifecycle,
    leave_one_out_camera_uncertainty,
    qualify_uncertainty_tagged_dynamic_camera,
)


def test_leave_one_out_uncertainty_detects_camera_outlier() -> None:
    sources = np.arange(9, dtype=np.float64)
    orientations = np.zeros((9, 3), dtype=np.float64)
    orientations[:, 1] = 0.03 * sources
    translations = np.column_stack(
        (0.01 * sources, np.zeros_like(sources), np.full_like(sources, 2.2))
    )
    clean = leave_one_out_camera_uncertainty(sources, orientations, translations)
    corrupted_orientations = orientations.copy()
    corrupted_orientations[4, 0] += math.radians(15.0)
    corrupted_translations = translations.copy()
    corrupted_translations[4, 0] += 0.10
    corrupted = leave_one_out_camera_uncertainty(
        sources, corrupted_orientations, corrupted_translations
    )
    assert corrupted.rotation_inconsistency_degrees[4] > 10.0
    assert corrupted.translation_inconsistency_metres[4] > 0.075
    assert corrupted.confidence[4] < clean.confidence[4] - 0.5


def _write_t04_fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    frame_count = 144
    video = {
        "path": "local_input.mp4",
        "codec": "h264",
        "width": 1120,
        "height": 720,
        "frame_count": 720,
        "frame_rate": 30.0,
        "duration_seconds": 24.0,
        "size_bytes": 1000,
    }
    manifest = {
        "schema_version": "canonical_dataset.v1",
        "status": "evidence_ready",
        "run_id": "t04-test",
        "input_video_path": "local_input.mp4",
        "input_video_sha256": "0" * 64,
        "video": video,
        "dataset_root": "outputs/test/dataset",
        "frames": [
            {
                "ordinal": slot,
                "source_frame_index": slot * 4,
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
    camera_frames = [
        {
            "source_frame_index": slot * 4,
            "betas": [0.0] * 10,
            "body_pose": [0.0] * 63,
            "global_orient": [0.0, slot * 0.01, 0.0],
            "translation": [slot * 0.001, 0.0, 2.2],
            "focal_length_px": 1000.25,
            "principal_point_px": [559.5, 359.5],
            "keypoints_2d": [],
            "joints_3d": [],
            "bounding_box_xyxy": [100.0, 50.0, 900.0, 700.0],
            "detection_score": 0.99,
        }
        for slot in range(frame_count)
    ]
    initialization: dict[str, object] = {
        "schema_version": "sequence_initialization.v1",
        "status": "refined",
        "shared_betas": [0.0] * 10,
        "shared_focal_length_px": 1000.25,
        "shared_principal_point_px": [559.5, 359.5],
        "image_width": 1120,
        "image_height": 720,
        "frames": camera_frames,
        "source_revision": "fixture",
        "checkpoint_sha256": "1" * 64,
        "camera_checkpoint_sha256": "2" * 64,
        "detector_checkpoint_sha256": "3" * 64,
        "proxy_camera": False,
        "zero_pose": False,
        "blockers": [],
    }
    manifest_path = write_json(root / "dataset_manifest.json", manifest)
    initialization_path = write_json(root / "sequence_initialization.json", initialization)
    return manifest_path, initialization_path, initialization


def test_t04_qualification_preserves_every_camera_parameter_exactly(tmp_path: Path) -> None:
    manifest_path, initialization_path, initialization = _write_t04_fixture(tmp_path)
    solution_path = tmp_path / "qualification" / "dynamic_camera_solution.json"
    report_path = tmp_path / "qualification" / "report.json"
    qualify_uncertainty_tagged_dynamic_camera(
        initialization_path,
        manifest_path,
        solution_path,
        report_path,
    )
    report = read_json(report_path)
    solution = read_json(solution_path)
    source_frames = {
        int(frame["source_frame_index"]): frame
        for frame in initialization["frames"]  # type: ignore[index,union-attr]
    }
    assert report["status"] == "pass"
    assert report["camera_initialization_parameters_exact"] is True
    assert report["same_device_replay_exact"] is True
    assert report["training_frame_count"] == 144
    assert report["legacy_development_images_read"] == 0
    assert report["sealed_test_accesses"] == 0
    assert solution["shared_intrinsics"] == [
        [1000.25, 0.0, 559.5],
        [0.0, 1000.25, 359.5],
        [0.0, 0.0, 1.0],
    ]
    for frame in solution["frames"]:
        source = source_frames[int(frame["source_frame_index"])]
        assert frame["global_orient"] == source["global_orient"]
        assert frame["translation"] == source["translation"]
    lifecycle_path = audit_t04_qualification_lifecycle(
        solution_path,
        report_path,
        tmp_path / "qualification" / "lifecycle.json",
    )
    lifecycle = read_json(lifecycle_path)
    assert lifecycle["status"] == "pass"
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


def test_t04_hull_binding_uses_solution_confidence_and_exact_source_order(
    tmp_path: Path,
) -> None:
    manifest_path, initialization_path, _ = _write_t04_fixture(tmp_path)
    solution_path = tmp_path / "qualification" / "dynamic_camera_solution.json"
    report_path = tmp_path / "qualification" / "report.json"
    qualify_uncertainty_tagged_dynamic_camera(
        initialization_path,
        manifest_path,
        solution_path,
        report_path,
    )
    mask_root = tmp_path / "masks"
    mask_root.mkdir()
    mask = np.zeros((64, 32), dtype=np.uint8)
    mask[8:56, 6:26] = 255
    for slot in range(144):
        assert cv2.imwrite(str(mask_root / f"frame_{slot:04d}.png"), mask)
    binding_path = bind_t04_hull_inputs(
        manifest_path,
        mask_root,
        solution_path,
        tmp_path / "hull" / "train_inputs.npz",
        maximum_dimension=32,
    )
    solution = read_json(solution_path)
    with np.load(binding_path, allow_pickle=False) as archive:
        assert archive["silhouettes"].shape == (144, 32, 16)
        assert archive["source_frame_indices"].tolist() == [slot * 4 for slot in range(144)]
        np.testing.assert_allclose(
            archive["motion_uncertainty"],
            np.asarray([1.0 - frame["confidence"] for frame in solution["frames"]]),
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            archive["original_intrinsics"],
            np.asarray(solution["shared_intrinsics"]),
        )
        assert (
            str(archive["camera_parameter_policy"])
            == "exact_t04_values_with_deterministic_image_coordinate_rescaling"
        )
    semantic_root = tmp_path / "semantics"
    semantic_root.mkdir()
    semantic_records = []
    semantic_labels = np.zeros((64, 32), dtype=np.uint8)
    semantic_labels[8:24, 6:26] = 23
    semantic_labels[24:48, 6:26] = 13
    semantic_labels[48:56, 8:24] = 9
    semantic_confidence = np.full((64, 32), 0.9, dtype=np.float16)
    for slot in range(144):
        path = semantic_root / f"frame_{slot:04d}.npz"
        np.savez_compressed(
            path,
            labels=semantic_labels,
            confidence=semantic_confidence,
            source_frame_index=np.asarray(slot * 4, dtype=np.int64),
        )
        semantic_records.append(
            {"source_frame_index": slot * 4, "semantic_sha256": sha256_file(path)}
        )
    semantic_qualification = write_json(
        tmp_path / "semantic_qualification.json",
        {"status": "pass", "frame_records": semantic_records},
    )
    semantic_binding = bind_t04_hull_inputs(
        manifest_path,
        mask_root,
        solution_path,
        tmp_path / "hull" / "semantic_train_inputs.npz",
        maximum_dimension=32,
        semantic_root=semantic_root,
        semantic_qualification_path=semantic_qualification,
    )
    with np.load(semantic_binding, allow_pickle=False) as archive:
        assert archive["semantic__upper_clothing"].shape == (144, 32, 16)
        assert float(archive["semantic__upper_clothing"].max()) > 0.8
        assert float(archive["semantic__body_parts"].max()) <= 1.0
        assert "semantic__body_parts" in archive
