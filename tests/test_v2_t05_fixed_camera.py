from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.t05_fixed_camera import (
    FixedCameraHumanSolution,
    audit_t05_qualification_lifecycle,
    audit_t05_training_background,
    decompose_fixed_camera_human_motion,
    fit_t05_fixed_camera_solution,
    reconstruct_t05_rotations,
    weighted_isotonic,
    write_t05_public_benchmark,
)


def test_weighted_isotonic_pools_only_violating_neighbors() -> None:
    values = np.asarray([0.0, 1.0, 0.5, 2.0, 1.8, 3.0])
    weights = np.asarray([1.0, 1.0, 3.0, 1.0, 1.0, 1.0])
    projected = weighted_isotonic(values, weights)
    assert np.all(np.diff(projected) >= 0.0)
    assert projected[1] == projected[2]
    assert projected[3] == projected[4]
    np.testing.assert_allclose(projected[[0, 5]], values[[0, 5]])


def test_fixed_camera_human_decomposition_replays_full_root_motion() -> None:
    frame_count = 20
    phase = np.linspace(0.0, 2.0 * math.pi, frame_count)
    base = Rotation.from_euler("x", math.pi - 0.03)
    axis = np.asarray([0.01, 1.0, -0.02])
    axis /= np.linalg.norm(axis)
    residual = np.stack(
        (0.005 * np.sin(phase), np.zeros(frame_count), 0.004 * np.cos(phase)), axis=1
    )
    rotations = (
        base * Rotation.from_rotvec(phase[:, None] * axis[None]) * Rotation.from_rotvec(residual)
    ).as_matrix()
    translations = np.stack(
        (
            0.02 * np.sin(phase),
            0.1 + 0.01 * np.cos(phase),
            2.4 + 0.03 * np.sin(2.0 * phase),
        ),
        axis=1,
    )
    result = decompose_fixed_camera_human_motion(
        rotations,
        translations,
        np.arange(frame_count, dtype=np.float64),
        np.ones(frame_count),
    )
    assert np.all(np.diff(result["yaw"]) >= 0.0)
    replay_error = Rotation.from_matrix(
        np.einsum("tji,tjk->tik", result["reconstructed_rotations"], rotations)
    ).magnitude()
    assert float(np.degrees(replay_error.max())) < 1.0e-8
    np.testing.assert_allclose(result["reconstructed_translations"], translations, atol=1.0e-12)
    np.testing.assert_allclose(result["root_residual_translation"].mean(axis=0), 0.0, atol=1e-12)


def _write_t05_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    frame_count = 144
    source_indices = list(range(frame_count))
    video = {
        "path": "docs/assets/fixture.mp4",
        "codec": "h264",
        "width": 64,
        "height": 48,
        "frame_count": frame_count,
        "frame_rate": 30.0,
        "duration_seconds": frame_count / 30.0,
        "size_bytes": 1000,
    }
    records = [
        {
            "ordinal": slot,
            "source_frame_index": source,
            "timestamp_seconds": source / 30.0,
            "image_path": f"images/frame_{slot:04d}.png",
            "split": "train",
            "blur_variance": 100.0,
            "mean_luminance": 128.0,
            "quality_accepted": True,
            "rejection_reasons": [],
        }
        for slot, source in enumerate(source_indices)
    ]
    manifest = {
        "schema_version": "canonical_dataset.v1",
        "status": "evidence_ready",
        "run_id": "t05-fixture",
        "input_video_path": "docs/assets/fixture.mp4",
        "input_video_sha256": "0" * 64,
        "video": video,
        "dataset_root": str(root),
        "frames": records,
        "train_frame_count": frame_count,
        "held_out_frame_count": 0,
        "rejected_candidate_count": 0,
        "blockers": [],
    }
    base = Rotation.from_euler("x", math.pi - 0.02)
    angles = np.linspace(0.0, 2.0 * math.pi, frame_count)
    rotations = (base * Rotation.from_rotvec(angles[:, None] * np.asarray([0, 1, 0]))).as_rotvec()
    initialization = {
        "schema_version": "sequence_initialization.v1",
        "status": "refined",
        "shared_betas": [0.0] * 10,
        "shared_focal_length_px": 80.0,
        "shared_principal_point_px": [31.5, 23.5],
        "image_width": 64,
        "image_height": 48,
        "frames": [
            {
                "source_frame_index": source,
                "betas": [0.0] * 10,
                "body_pose": [0.01 * math.sin(angle)] * 69,
                "global_orient": rotation.tolist(),
                "translation": [
                    0.01 * math.sin(angle),
                    0.1,
                    2.4 + 0.01 * math.cos(angle),
                ],
                "focal_length_px": 80.0,
                "principal_point_px": [31.5, 23.5],
                "keypoints_2d": [],
                "joints_3d": [],
                "bounding_box_xyxy": [8.0, 4.0, 56.0, 44.0],
                "detection_score": 0.99,
            }
            for source, angle, rotation in zip(source_indices, angles, rotations, strict=True)
        ],
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
    v00_lifecycle = write_json(
        root / "v00_lifecycle.json",
        {"status": "pass", "state": "qualified"},
    )
    static = np.random.default_rng(11).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    static_path = root / "static.png"
    Image.fromarray(static, mode="RGB").save(static_path)
    static_hash = sha256_file(static_path)
    evidence = write_json(
        root / "evidence_master.json",
        {
            "frames": [
                {
                    "source_frame_index": source,
                    "lossless_frame_path": "static.png",
                    "lossless_frame_sha256": static_hash,
                }
                for source in source_indices
            ]
        },
    )
    return manifest_path, initialization_path, v00_lifecycle, evidence


def test_t05_solution_public_background_and_lifecycle_qualify_fixture(tmp_path: Path) -> None:
    manifest, initialization, v00_lifecycle, evidence = _write_t05_fixture(tmp_path)
    public = write_t05_public_benchmark(tmp_path / "public.json")
    solution_path = fit_t05_fixed_camera_solution(
        initialization,
        manifest,
        v00_lifecycle,
        tmp_path / "solution.json",
        source_revision="a" * 40,
    )
    solution = FixedCameraHumanSolution.model_validate(read_json(solution_path))
    assert solution.training_frame_count == 144
    assert np.all(np.diff([frame.yaw_radians for frame in solution.frames]) >= 0.0)
    observed = Rotation.from_rotvec(
        np.asarray([frame.observed_global_orient_rotvec for frame in solution.frames])
    ).as_matrix()
    replay = reconstruct_t05_rotations(solution)
    error = Rotation.from_matrix(np.einsum("tji,tjk->tik", replay, observed)).magnitude()
    assert float(np.degrees(error.max())) < 1.0e-8
    # The registered translation gauge is confidence weighted. Exercise a case
    # whose ordinary mean is non-zero so the lifecycle cannot silently audit a
    # different gauge than the fitter enforces.
    solution.frames[0].yaw_confidence = 1.0
    solution.frames[1].yaw_confidence = 0.5
    confidence = np.asarray([frame.yaw_confidence for frame in solution.frames])
    residual = np.asarray([frame.root_residual_translation_metres for frame in solution.frames])
    residual -= np.average(residual, axis=0, weights=confidence)
    residual[0, 0] += 1.0e-4
    residual[1, 0] -= 2.0e-4
    for slot, frame in enumerate(solution.frames):
        frame.root_residual_translation_metres = residual[slot].tolist()
        frame.low_frequency_root_translation_metres = (
            np.asarray(frame.root_translation_metres) - residual[slot]
        ).tolist()
    write_json(solution_path, solution)
    residual = np.asarray([frame.root_residual_translation_metres for frame in solution.frames])
    confidence = np.asarray([frame.yaw_confidence for frame in solution.frames])
    assert abs(float(np.mean(residual, axis=0)[0])) > 1.0e-8
    np.testing.assert_allclose(
        np.average(residual, axis=0, weights=confidence),
        0.0,
        atol=1.0e-12,
    )
    background = audit_t05_training_background(
        evidence,
        manifest,
        tmp_path / "background.json",
    )
    assert read_json(background)["status"] == "pass"
    development = write_json(
        tmp_path / "development.json",
        {
            "status": "pass",
            "device": "cpu",
            "development_records_used_for_fit": 0,
            "development_masks_read": 36,
            "sealed_test_reads": 0,
        },
    )
    lifecycle = audit_t05_qualification_lifecycle(
        public,
        solution_path,
        background,
        development,
        tmp_path / "lifecycle.json",
    )
    report = read_json(lifecycle)
    assert report["status"] == "pass"
    assert report["state"] == "qualified"
    assert [transition["to"] for transition in report["transitions"]] == [
        "imported",
        "data_bound",
        "device_validated",
        "one_step_passed",
        "checkpoint_restored",
        "evaluator_dry",
        "qualified",
    ]
