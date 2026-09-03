from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from frayid.cli import app
from frayid.io import sha256_file, write_json
from frayid.v2.evidence_master import (
    audit_v00_qualification_lifecycle,
    build_evidence_master,
    proxy_coordinate_contract,
    render_analysis_proxy,
)
from frayid.v2.frame_selection import select_phase_uniform_frames
from frayid.v2.video_forensics import (
    camera_verdict,
    estimate_background_transforms,
    parse_frame_timestamps,
    summarize_timestamps,
)


def _make_lossless_video(path: Path, frames: list[np.ndarray], *, rate: int = 5) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    height, width = frames[0].shape[:2]
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(rate),
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            str(path),
        ],
        input=b"".join(frame.tobytes(order="C") for frame in frames),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_native_timestamp_parser_preserves_vfr_and_missing_pts() -> None:
    vfr = parse_frame_timestamps(
        [
            {"media_type": "video", "pts_time": "0.000000", "key_frame": 1},
            {"media_type": "video", "pts_time": "0.040000", "key_frame": 0},
            {"media_type": "video", "pts_time": "0.100000", "key_frame": 0},
        ]
    )
    report = summarize_timestamps(vfr)
    assert report["strictly_monotonic"] is True
    assert report["median_delta_seconds"] == pytest.approx(0.05)
    assert report["delta_anomaly_count_over_1ms"] == 2

    missing = parse_frame_timestamps(
        [
            {"media_type": "video", "pts_time": "0.0"},
            {"media_type": "video", "best_effort_timestamp_time": "0.04"},
            {"media_type": "video"},
        ]
    )
    assert missing[1].selected_timestamp_source == "best_effort"
    assert missing[2].selected_timestamp_source == "missing"
    missing_report = summarize_timestamps(missing)
    assert missing_report["missing_timestamp_count"] == 1
    assert missing_report["strictly_monotonic"] is False


def test_proxy_homography_is_isotropic_reversible_and_nonauthoritative() -> None:
    contract = proxy_coordinate_contract((720, 1120), (512, 768))
    matrix = np.asarray(contract["source_to_proxy_homography"])
    assert matrix[0, 0] == pytest.approx(matrix[1, 1])
    assert contract["maximum_round_trip_pixel_error"] < 1.0e-4
    assert contract["measured_evidence"] is False
    image = np.zeros((1120, 720, 3), dtype=np.uint8)
    proxy = render_analysis_proxy(image, contract)
    assert proxy.shape == (768, 512, 3)


def test_background_audit_separates_static_and_moving_camera() -> None:
    rng = np.random.default_rng(20260903)
    texture = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    cv2.rectangle(texture, (2, 2), (42, 30), (255, 255, 255), 2)
    static = estimate_background_transforms([texture, texture.copy()], source_indices=[0, 1])
    assert static == estimate_background_transforms(
        [texture, texture.copy()], source_indices=[0, 1]
    )
    assert camera_verdict(static) == "fixed_to_subpixel_precision"

    affine = np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float32)
    shifted = cv2.warpAffine(texture, affine, (128, 96))
    moving = estimate_background_transforms([texture, shifted], source_indices=[0, 1])
    assert float(moving["summary"]["maximum_absolute_translation_pixels"]) > 2.0
    assert camera_verdict(moving) == "background_motion_detected"


def test_evidence_master_is_atomic_sequential_and_pixel_hash_bound(tmp_path: Path) -> None:
    rng = np.random.default_rng(44)
    first = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    frames = [first, first.copy()]
    for shift in (1, 2, 3, 4):
        changed = first.copy()
        cv2.circle(changed, (28 + shift, 24), 6, (255, 0, 0), -1)
        frames.append(changed)
    video = tmp_path / "fixture.mkv"
    _make_lossless_video(video, frames)
    output_root = tmp_path / "master"
    manifest_path = build_evidence_master(
        video,
        output_root,
        source_revision="a" * 40,
        run_id="fixture-v00-r01",
        storage="png",
        proxy_size=(40, 40),
        background_sample_count=6,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert manifest["decode"]["policy"] == "single_forward_only_sequential_decode"
    assert manifest["decode"]["random_seek_count"] == 0
    assert manifest["decode"]["decoded_frame_count"] == len(frames)
    assert manifest["timing_summary"]["strictly_monotonic"] is True
    assert manifest["duplicate_decoded_frame_count"] >= 1
    assert manifest["background_audit_repeatable"] is True
    assert manifest["evidence_policy"]["generated_pixels_in_measured_evidence"] is False
    assert set(manifest["color_metadata"]) == {
        "range",
        "space",
        "transfer",
        "primaries",
        "field_order",
    }
    first_record = manifest["frames"][0]
    stored = np.asarray(
        Image.open(output_root / first_record["lossless_frame_path"]).convert("RGB")
    )
    assert (
        hashlib.sha256(stored.tobytes(order="C")).hexdigest() == first_record["decoded_rgb_sha256"]
    )
    assert (output_root / "qualification.json").is_file()
    lifecycle_path = audit_v00_qualification_lifecycle(
        manifest_path,
        output_root / "qualification.json",
        output_root / "qualification_lifecycle.json",
        expected_source_sha256=sha256_file(video),
        expected_frame_count=len(frames),
    )
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["status"] == "pass"
    assert lifecycle["state"] == "qualified"
    assert len(lifecycle["transitions"]) == 7
    with pytest.raises(FileExistsError, match="immutable"):
        build_evidence_master(
            video,
            output_root,
            source_revision="a" * 40,
            run_id="fixture-v00-r01",
            storage="hashes_only",
        )


def test_phase_selection_requires_t05_yaw_and_respects_eligible_indices(tmp_path: Path) -> None:
    master = write_json(
        tmp_path / "master.json",
        {
            "frames": [
                {"source_frame_index": index, "timing": {"selected_timestamp_seconds": index / 10}}
                for index in range(12)
            ]
        },
    )
    phase = write_json(
        tmp_path / "phase.json",
        {
            "frames": [
                {
                    "source_frame_index": index,
                    "yaw_radians": index * 2.0 * np.pi / 11.0,
                    "yaw_confidence": 0.95,
                }
                for index in range(12)
            ]
        },
    )
    output = select_phase_uniform_frames(
        master,
        phase,
        tmp_path / "selection.json",
        count=4,
        eligible_source_indices=set(range(0, 12, 2)),
        minimum_confidence=0.9,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selection_policy"] == "t05_monotonic_yaw_uniform_no_time_fallback"
    assert all(row["source_frame_index"] % 2 == 0 for row in payload["frames"])
    assert payload["selected_count"] == 4
    assert np.all(np.diff([row["yaw_radians"] for row in payload["frames"]]) >= 0.0)
    assert np.all(np.diff([row["target_yaw_radians"] for row in payload["frames"]]) > 0.0)

    missing_phase = write_json(tmp_path / "missing_phase.json", {"frames": []})
    with pytest.raises(ValueError, match="insufficient eligible T05 phase"):
        select_phase_uniform_frames(
            master,
            missing_phase,
            tmp_path / "must_not_exist.json",
            count=4,
        )


def test_v2_cli_exposes_capture_forensics_commands() -> None:
    result = CliRunner().invoke(app, ["v2", "--help"])
    assert result.exit_code == 0
    assert "video-audit" in result.stdout
    assert "build-evidence-master" in result.stdout
    assert "select-phase-frames" in result.stdout
