from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

from frayid.io import sha256_file, write_json
from frayid.v2.contracts import QualificationState, advance_qualification, reject_sealed_capability
from frayid.v2.video_forensics import (
    camera_verdict,
    decoded_frame_metrics,
    estimate_background_transforms,
    executable_version,
    iter_sequential_rgb_frames,
    probe_video_forensics,
    summarize_timestamps,
)

V00_EXPERIMENT_ID = "postv2_v00_capture_forensics_evidence_master_r01"
CONTROLLED_IMPORTANT_SHA256 = "unavailable-in-public-snapshot"
CONTROLLED_IMPORTANT_FRAME_COUNT = 935
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def proxy_coordinate_contract(
    source_size: tuple[int, int],
    proxy_size: tuple[int, int],
) -> dict[str, Any]:
    """Create one isotropic, invertible source-to-letterbox coordinate map."""
    source_width, source_height = source_size
    proxy_width, proxy_height = proxy_size
    if min(source_width, source_height, proxy_width, proxy_height) <= 0:
        raise ValueError("source and proxy dimensions must be positive")
    scale = min(proxy_width / source_width, proxy_height / source_height)
    tx = (proxy_width - scale * source_width) / 2.0
    ty = (proxy_height - scale * source_height) / 2.0
    forward = np.asarray([[scale, 0.0, tx], [0.0, scale, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    inverse = np.linalg.inv(forward)
    points = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [source_width - 1.0, 0.0, 1.0],
            [0.0, source_height - 1.0, 1.0],
            [source_width - 1.0, source_height - 1.0, 1.0],
            [(source_width - 1.0) / 2.0, (source_height - 1.0) / 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    projected = (forward @ points.T).T
    restored = (inverse @ projected.T).T
    error = float(np.max(np.linalg.norm(restored[:, :2] - points[:, :2], axis=1)))
    return {
        "role": "non_authoritative_analysis_proxy",
        "measured_evidence": False,
        "generated_padding_is_evidence": False,
        "source_width": source_width,
        "source_height": source_height,
        "proxy_width": proxy_width,
        "proxy_height": proxy_height,
        "isotropic_scale": scale,
        "source_to_proxy_homography": forward.tolist(),
        "proxy_to_source_homography": inverse.tolist(),
        "maximum_round_trip_pixel_error": error,
    }


def render_analysis_proxy(rgb: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    """Render a visibly non-authoritative proxy using the registered homography."""
    matrix = np.asarray(contract["source_to_proxy_homography"], dtype=np.float64)
    return cv2.warpPerspective(
        rgb,
        matrix,
        (int(contract["proxy_width"]), int(contract["proxy_height"])),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def build_evidence_master(
    video_path: Path,
    output_root: Path,
    *,
    source_revision: str,
    run_id: str,
    storage: Literal["png", "hashes_only"] = "png",
    proxy_size: tuple[int, int] = (512, 768),
    background_sample_count: int = 24,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Path:
    """Build one atomic, immutable, sequentially decoded V00 evidence master."""
    reject_sealed_capability([video_path, output_root])
    if not _REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError("source_revision must be a full lowercase Git commit")
    if output_root.exists():
        raise FileExistsError(f"evidence master output is immutable: {output_root}")
    if background_sample_count < 2:
        raise ValueError("background_sample_count must be at least two")
    probe, timestamps, probe_provenance = probe_video_forensics(video_path, ffprobe_bin=ffprobe_bin)
    timestamp_summary = summarize_timestamps(timestamps)
    sample_count = min(background_sample_count, max(len(timestamps), 2))
    sample_indices = set(
        np.linspace(0, max(len(timestamps) - 1, 1), sample_count, dtype=np.int64).tolist()
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    frames_root = stage / "frames"
    if storage == "png":
        frames_root.mkdir(parents=True)
    frame_records: list[dict[str, Any]] = []
    sampled_frames: list[np.ndarray] = []
    sampled_indices: list[int] = []
    previous: np.ndarray | None = None
    decoded_count = 0
    try:
        for decode_index, rgb in enumerate(
            iter_sequential_rgb_frames(
                video_path,
                width=probe.width,
                height=probe.height,
                ffmpeg_bin=ffmpeg_bin,
            )
        ):
            metrics = decoded_frame_metrics(rgb, previous_rgb=previous)
            timestamp = timestamps[decode_index] if decode_index < len(timestamps) else None
            record: dict[str, Any] = {
                "decode_index": decode_index,
                "source_frame_index": decode_index,
                "authoritative_measured_pixels": True,
                "decoded_pixel_format": "rgb24",
                **metrics,
                "timing": timestamp.model_dump(mode="json") if timestamp is not None else None,
            }
            if storage == "png":
                relative = Path("frames") / f"frame_{decode_index:06d}.png"
                frame_path = stage / relative
                Image.fromarray(rgb, mode="RGB").save(
                    frame_path, format="PNG", compress_level=6, optimize=False
                )
                record["lossless_frame_path"] = relative.as_posix()
                record["lossless_frame_sha256"] = sha256_file(frame_path)
            else:
                record["lossless_frame_path"] = None
                record["lossless_frame_sha256"] = None
            frame_records.append(record)
            if decode_index in sample_indices:
                sampled_frames.append(rgb.copy())
                sampled_indices.append(decode_index)
            previous = rgb
            decoded_count += 1
        if len(sampled_frames) < 2:
            raise ValueError("video does not contain enough frames for background audit")
        background_first = estimate_background_transforms(
            sampled_frames, source_indices=sampled_indices
        )
        background_second = estimate_background_transforms(
            sampled_frames, source_indices=sampled_indices
        )
        background_repeatable = _canonical_json(background_first) == _canonical_json(
            background_second
        )
        proxy = proxy_coordinate_contract(
            (probe.width, probe.height),
            proxy_size,
        )
        hash_counts: dict[str, int] = {}
        for record in frame_records:
            digest = str(record["decoded_rgb_sha256"])
            hash_counts[digest] = hash_counts.get(digest, 0) + 1
        duplicate_count = sum(count - 1 for count in hash_counts.values() if count > 1)
        blockers: list[str] = []
        if decoded_count != len(timestamps):
            blockers.append("decoded_frame_count_does_not_match_frame_probe")
        if probe.reported_frame_count is not None and decoded_count != probe.reported_frame_count:
            blockers.append("decoded_frame_count_does_not_match_stream_probe")
        if not bool(timestamp_summary["strictly_monotonic"]):
            blockers.append("native_frame_timestamps_missing_or_nonmonotonic")
        if not background_repeatable:
            blockers.append("background_audit_not_repeatable")
        if float(proxy["maximum_round_trip_pixel_error"]) > 1.0e-4:
            blockers.append("proxy_coordinate_round_trip_above_tolerance")
        manifest: dict[str, Any] = {
            "schema_version": "frayid_v2_evidence_master.v1",
            "experiment_id": V00_EXPERIMENT_ID,
            "run_id": run_id,
            "source_revision": source_revision,
            "status": "pass" if not blockers else "blocked",
            "blockers": blockers,
            "source": {
                "path": str(video_path),
                "sha256": sha256_file(video_path),
                "probe": probe.model_dump(mode="json"),
            },
            "decode": {
                "policy": "single_forward_only_sequential_decode",
                "random_seek_count": 0,
                "timestamp_synthesis_allowed": False,
                "decoded_frame_count": decoded_count,
                "decoded_pixel_format": "rgb24",
                "storage": storage,
                "ffmpeg_version": executable_version(ffmpeg_bin),
                **probe_provenance,
            },
            "timing_summary": timestamp_summary,
            "color_metadata": {
                "range": probe.color_range,
                "space": probe.color_space,
                "transfer": probe.color_transfer,
                "primaries": probe.color_primaries,
                "field_order": probe.field_order,
            },
            "duplicate_decoded_frame_count": duplicate_count,
            "background_audit": background_first,
            "background_audit_repeatable": background_repeatable,
            "physical_camera_verdict": camera_verdict(background_first),
            "proxy_coordinate_contract": proxy,
            "evidence_policy": {
                "raw_decoded_frames_authoritative": True,
                "analysis_proxy_authoritative": False,
                "generated_pixels_in_measured_evidence": False,
                "blind_stabilization_applied": False,
                "blind_deblur_or_denoise_applied": False,
                "generated_super_resolution_applied": False,
                "frame_interpolation_applied": False,
                "exposure_normalization_baked_into_evidence": False,
            },
            "frames": frame_records,
        }
        manifest_path = write_json(stage / "evidence_master.json", manifest)
        manifest_sha = sha256_file(manifest_path)
        write_json(
            stage / "qualification.json",
            {
                "schema_version": "frayid_v2_v00_qualification.v1",
                "experiment_id": V00_EXPERIMENT_ID,
                "run_id": run_id,
                "source_revision": source_revision,
                "status": manifest["status"],
                "blockers": blockers,
                "checks": {
                    "decoded_frame_count_matches_probe": decoded_count == len(timestamps)
                    and (
                        probe.reported_frame_count is None
                        or decoded_count == probe.reported_frame_count
                    ),
                    "pts_monotonic": bool(timestamp_summary["strictly_monotonic"]),
                    "proxy_round_trip_pixel_error_at_most_1e_4": float(
                        proxy["maximum_round_trip_pixel_error"]
                    )
                    <= 1.0e-4,
                    "background_audit_repeatable": background_repeatable,
                    "generated_pixels_in_measured_evidence": False,
                },
                "decoded_frame_count": decoded_count,
                "evidence_master_sha256": manifest_sha,
                "development_reads": 0,
                "sealed_test_reads": 0,
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


def audit_video(
    video_path: Path,
    output_path: Path,
    *,
    source_revision: str,
    run_id: str = "audit-20260903-r01",
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Path:
    """Run V00 without retaining frames; useful for an immutable diagnostic report."""
    if output_path.exists():
        raise FileExistsError(f"video audit output is immutable: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}-", dir=output_path.parent
    ) as temporary:
        root = Path(temporary) / "evidence-master"
        manifest_path = build_evidence_master(
            video_path,
            root,
            source_revision=source_revision,
            run_id=run_id,
            storage="hashes_only",
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_json(output_path, payload)
    return output_path


def verify_evidence_master(
    evidence_master_path: Path,
    qualification_report_path: Path,
    *,
    expected_source_sha256: str = CONTROLLED_IMPORTANT_SHA256,
    expected_frame_count: int = CONTROLLED_IMPORTANT_FRAME_COUNT,
) -> dict[str, Any]:
    """Restore every retained measured frame and independently verify its hashes."""
    reject_sealed_capability([evidence_master_path, qualification_report_path])
    manifest = json.loads(evidence_master_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_report_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(qualification, dict):
        raise ValueError("V00 manifest and qualification must be JSON objects")
    rows = manifest.get("frames")
    if not isinstance(rows, list):
        raise ValueError("V00 evidence master has no frame list")
    root = evidence_master_path.parent.resolve()
    restored_count = 0
    file_hash_failures = 0
    pixel_hash_failures = 0
    unsafe_paths = 0
    for raw in rows:
        if not isinstance(raw, dict):
            pixel_hash_failures += 1
            continue
        relative_raw = raw.get("lossless_frame_path")
        if not isinstance(relative_raw, str):
            pixel_hash_failures += 1
            continue
        relative = Path(relative_raw)
        candidate = (root / relative).resolve()
        if relative.is_absolute() or root not in candidate.parents:
            unsafe_paths += 1
            continue
        if not candidate.is_file():
            file_hash_failures += 1
            continue
        if sha256_file(candidate) != raw.get("lossless_frame_sha256"):
            file_hash_failures += 1
            continue
        rgb = np.asarray(Image.open(candidate).convert("RGB"))
        digest = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
        if digest != raw.get("decoded_rgb_sha256"):
            pixel_hash_failures += 1
            continue
        restored_count += 1
    source = manifest.get("source")
    decode = manifest.get("decode")
    timing = manifest.get("timing_summary")
    checks = {
        "source_hash_matches_controlled_input": isinstance(source, dict)
        and source.get("sha256") == expected_source_sha256,
        "decoded_frame_count_matches_controlled_input": isinstance(decode, dict)
        and decode.get("decoded_frame_count") == expected_frame_count
        and len(rows) == expected_frame_count,
        "native_timestamps_strictly_monotonic": isinstance(timing, dict)
        and timing.get("strictly_monotonic") is True,
        "lossless_frame_storage_required": isinstance(decode, dict)
        and decode.get("storage") == "png",
        "all_lossless_frames_restored": restored_count == expected_frame_count,
        "manifest_bound_to_qualification": qualification.get("evidence_master_sha256")
        == sha256_file(evidence_master_path),
        "background_audit_repeatable": manifest.get("background_audit_repeatable") is True,
        "fixed_camera_diagnostic_passed": manifest.get("physical_camera_verdict")
        == "fixed_to_subpixel_precision",
        "proxy_coordinate_round_trip_passed": float(
            manifest.get("proxy_coordinate_contract", {}).get(
                "maximum_round_trip_pixel_error", float("inf")
            )
        )
        <= 1.0e-4,
        "generated_pixels_excluded_from_measured_evidence": manifest.get("evidence_policy", {}).get(
            "generated_pixels_in_measured_evidence"
        )
        is False,
        "access_boundary_passed": qualification.get("development_reads") == 0
        and qualification.get("sealed_test_reads") == 0
        and qualification.get("paid_jobs") == 0,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not blockers else "fail",
        "checks": checks,
        "blockers": blockers,
        "restored_frame_count": restored_count,
        "file_hash_failure_count": file_hash_failures,
        "pixel_hash_failure_count": pixel_hash_failures,
        "unsafe_path_count": unsafe_paths,
    }


def audit_v00_qualification_lifecycle(
    evidence_master_path: Path,
    qualification_report_path: Path,
    output_path: Path,
    *,
    expected_source_sha256: str = CONTROLLED_IMPORTANT_SHA256,
    expected_frame_count: int = CONTROLLED_IMPORTANT_FRAME_COUNT,
) -> Path:
    """Restore V00 and record every ordered local qualification transition."""
    reject_sealed_capability([evidence_master_path, qualification_report_path, output_path])
    if output_path.exists():
        raise FileExistsError("V00 lifecycle records are immutable")
    verification = verify_evidence_master(
        evidence_master_path,
        qualification_report_path,
        expected_source_sha256=expected_source_sha256,
        expected_frame_count=expected_frame_count,
    )
    checks = {
        "module_imported": True,
        "controlled_video_data_bound": verification["checks"][
            "source_hash_matches_controlled_input"
        ]
        and verification["checks"]["decoded_frame_count_matches_controlled_input"],
        "local_cpu_decode_validated": verification["checks"][
            "native_timestamps_strictly_monotonic"
        ],
        "deterministic_sequential_decode_passed": verification["checks"][
            "background_audit_repeatable"
        ],
        "immutable_evidence_restored": verification["checks"]["all_lossless_frames_restored"]
        and verification["checks"]["manifest_bound_to_qualification"],
        "camera_and_proxy_evaluator_dry_run_passed": verification["checks"][
            "fixed_camera_diagnostic_passed"
        ]
        and verification["checks"]["proxy_coordinate_round_trip_passed"],
        "access_boundary_passed": verification["checks"]["access_boundary_passed"]
        and verification["checks"]["generated_pixels_excluded_from_measured_evidence"],
    }
    blockers = [name for name, passed in checks.items() if not passed]
    blockers.extend(str(item) for item in verification["blockers"] if item not in blockers)
    state = QualificationState.BUILT
    transitions: list[dict[str, str]] = []
    transition_evidence = {
        QualificationState.IMPORTED: "module_imported",
        QualificationState.DATA_BOUND: "controlled_video_data_bound",
        QualificationState.DEVICE_VALIDATED: "local_cpu_decode_validated",
        QualificationState.ONE_STEP_PASSED: "deterministic_sequential_decode_passed",
        QualificationState.CHECKPOINT_RESTORED: "immutable_evidence_restored",
        QualificationState.EVALUATOR_DRY: "camera_and_proxy_evaluator_dry_run_passed",
        QualificationState.QUALIFIED: "access_boundary_passed",
    }
    if not blockers:
        for requested, evidence in transition_evidence.items():
            previous = state
            state = advance_qualification(state, requested)
            transitions.append({"from": previous.value, "to": state.value, "evidence": evidence})
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_v00_qualification_lifecycle.v1",
            "experiment_id": V00_EXPERIMENT_ID,
            "status": "pass" if state is QualificationState.QUALIFIED else "fail",
            "state": state.value,
            "checks": checks,
            "transitions": transitions,
            "verification": verification,
            "evidence_master_sha256": sha256_file(evidence_master_path),
            "qualification_report_sha256": sha256_file(qualification_report_path),
            "auditor_source_sha256": sha256_file(Path(__file__)),
            "blockers": blockers,
            "development_reads": 0,
            "sealed_test_reads": 0,
            "paid_jobs": 0,
            "attempt_marker_created": False,
            "optimizer_steps": 0,
            "note": (
                "ONE_STEP_PASSED denotes one deterministic full sequential decode pass; "
                "CHECKPOINT_RESTORED verifies every immutable retained frame and manifest hash."
            ),
        },
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
