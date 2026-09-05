"""Prepare a sparse, provenance-preserving capture bundle for upstream probes."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, cast

import cv2
import numpy as np

from frayid.dataset import read_dataset_manifest
from frayid.io import read_json, write_json
from frayid.schemas import (
    DatasetManifest,
    FrameRecord,
    ObservedPoseSequence,
    SequenceInitialization,
    VideoMetadata,
)


def safe_relative_root(value: str) -> PurePosixPath:
    """Validate a private-volume-relative dataset root."""
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("dataset_relative_root must be a non-empty relative path without '..'")
    return relative


def _load_probe(path: Path) -> dict[str, Any]:
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError("adapter probe manifest must be a JSON object")
    supported = {
        "frayid_capture_adapter_probe.v1",
        "frayid_capture_adapter_interval.v1",
    }
    if payload.get("schema_version") not in supported:
        raise ValueError("unsupported capture-adapter manifest")
    if not isinstance(payload.get("frames"), list) or not payload["frames"]:
        raise ValueError("adapter probe has no frames")
    return cast(dict[str, Any], payload)


def build_probe_dataset_manifest(
    probe_root: Path,
    *,
    dataset_relative_root: str,
) -> tuple[DatasetManifest, Path]:
    """Translate a probe manifest into the pinned upstream jobs' dataset contract."""
    probe_root = probe_root.resolve()
    relative_root = safe_relative_root(dataset_relative_root)
    probe = _load_probe(probe_root / "manifest.json")
    source = probe["source"]
    runtime_dataset_root = PurePosixPath("/workspace/outputs") / relative_root
    frame_records: list[FrameRecord] = []
    for ordinal, frame in enumerate(probe["frames"]):
        local_image = probe_root / frame["path"]
        if not local_image.is_file():
            raise FileNotFoundError(local_image)
        if _sha256(local_image) != frame["sha256"]:
            raise ValueError(f"probe frame hash mismatch: {local_image.name}")
        image = cv2.imread(str(local_image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"probe frame is unreadable: {local_image}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        runtime_image = runtime_dataset_root / "images" / local_image.name
        frame_records.append(
            FrameRecord(
                ordinal=ordinal,
                source_frame_index=frame["source_frame_index"],
                timestamp_seconds=frame["source_timestamp_seconds"],
                image_path=str(runtime_image.relative_to("/workspace")),
                split=frame.get("split", "train"),
                blur_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                mean_luminance=float(gray.mean()),
                quality_accepted=True,
            )
        )

    manifest = DatasetManifest(
        status="rgb_ready",
        run_id=relative_root.name,
        input_video_path=source["path"],
        input_video_sha256=source["sha256"],
        video=VideoMetadata(
            path=source["path"],
            codec=source["codec"],
            width=source["width"],
            height=source["height"],
            frame_count=source["packet_count"],
            frame_rate=float(Fraction(source["r_frame_rate"])),
            duration_seconds=source["duration_seconds"],
            size_bytes=source["size_bytes"],
        ),
        dataset_root=str(runtime_dataset_root),
        frames=frame_records,
        train_frame_count=sum(frame.split == "train" for frame in frame_records),
        held_out_frame_count=sum(frame.split == "held_out" for frame in frame_records),
        rejected_candidate_count=0,
        blockers=[
            "real_sapiens2_masks_required",
            "real_sapiens2_normals_required",
            "real_sapiens2_observed_pose_required",
            "real_camerahmr_smpl_initialization_required",
        ],
    )
    output_path = probe_root / "dataset_manifest.json"
    write_json(output_path, manifest)
    return manifest, output_path


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    top_left = np.maximum(first[:2], second[:2])
    bottom_right = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(bottom_right - top_left, 0.0)))
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def qualify_probe_results(results_root: Path) -> Path:
    """Qualify real sparse upstream evidence without claiming reconstruction success."""
    results_root = results_root.resolve()
    manifest = read_dataset_manifest(results_root / "dataset_manifest.json")
    camera = SequenceInitialization.model_validate(read_json(results_root / "camerahmr_raw_sequence.json"))
    pose = ObservedPoseSequence.model_validate(read_json(results_root / "sapiens2_pose_sequence.json"))
    camera_provenance = read_json(results_root / "camerahmr_evidence_provenance.json")
    dense_provenance = read_json(results_root / "sapiens2_evidence_provenance.json")
    pose_provenance = read_json(results_root / "sapiens2_pose_provenance.json")
    adapter_manifest = _load_probe(results_root / "adapter_probe_manifest.json")

    expected_indices = {frame.source_frame_index for frame in manifest.frames}
    camera_by_index = {frame.source_frame_index: frame for frame in camera.frames}
    pose_by_index = {frame.source_frame_index: frame for frame in pose.frames}
    blockers: list[str] = []
    if set(camera_by_index) != expected_indices:
        blockers.append("camerahmr_frame_coverage_mismatch")
    if set(pose_by_index) != expected_indices:
        blockers.append("sapiens2_pose_frame_coverage_mismatch")
    for name, provenance in (
        ("camerahmr", camera_provenance),
        ("sapiens2_dense", dense_provenance),
        ("sapiens2_pose", pose_provenance),
    ):
        if provenance.get("status") != "complete":
            blockers.append(f"{name}_incomplete")
        if provenance.get("proxy_evidence") is True or provenance.get("proxy_camera") is True:
            blockers.append(f"{name}_proxy_output_forbidden")

    frame_metrics: list[dict[str, Any]] = []
    for record in manifest.frames:
        source_index = record.source_frame_index
        if source_index not in camera_by_index or source_index not in pose_by_index:
            continue
        name = Path(record.image_path).name
        image = cv2.imread(str(results_root / "images" / name), cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(str(results_root / "masks" / name), cv2.IMREAD_GRAYSCALE)
        normal_bgr = cv2.imread(str(results_root / "normals" / name), cv2.IMREAD_COLOR)
        if image is None or mask_u8 is None or normal_bgr is None:
            blockers.append(f"missing_or_unreadable_dense_evidence:{source_index}")
            continue
        if image.shape[:2] != mask_u8.shape[:2] or image.shape != normal_bgr.shape:
            blockers.append(f"dense_evidence_dimensions_mismatch:{source_index}")
            continue

        mask = mask_u8 > 127
        foreground_fraction = float(mask.mean())
        if not 0.01 < foreground_fraction < 0.90:
            blockers.append(f"implausible_foreground_fraction:{source_index}")
            continue
        ys, xs = np.where(mask)
        mask_box = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        component_areas = stats[1:component_count, cv2.CC_STAT_AREA]
        largest_component_fraction = float(component_areas.max() / mask.sum())

        normals_rgb = normal_bgr[..., ::-1].astype(np.float32) / 127.5 - 1.0
        foreground_norms = np.linalg.norm(normals_rgb[mask], axis=-1)
        normal_unit_error = float(np.mean(np.abs(foreground_norms - 1.0)))
        camera_frame = camera_by_index[source_index]
        pose_frame = pose_by_index[source_index]
        confident_joint_count = sum(point[2] >= 0.35 for point in pose_frame.keypoints_body12)
        camera_box_iou = _bbox_iou(
            mask_box, np.asarray(camera_frame.bounding_box_xyxy, dtype=np.float64)
        )
        pose_box_iou = _bbox_iou(
            mask_box, np.asarray(pose_frame.bounding_box_xyxy, dtype=np.float64)
        )
        if camera_frame.detection_score < 0.5:
            blockers.append(f"weak_camerahmr_detection:{source_index}")
        if confident_joint_count < 6:
            blockers.append(f"insufficient_sapiens2_pose_joints:{source_index}")
        if largest_component_fraction < 0.95:
            blockers.append(f"fragmented_sapiens2_mask:{source_index}")
        if normal_unit_error > 0.02:
            blockers.append(f"invalid_sapiens2_normals:{source_index}")
        if min(camera_box_iou, pose_box_iou) < 0.80:
            blockers.append(f"detector_mask_disagreement:{source_index}")
        frame_metrics.append(
            {
                "ordinal": record.ordinal,
                "source_frame_index": source_index,
                "camerahmr_detection_score": camera_frame.detection_score,
                "sapiens2_confident_body_joint_count": confident_joint_count,
                "mask_foreground_fraction": foreground_fraction,
                "mask_largest_component_fraction": largest_component_fraction,
                "mask_touches_bottom_edge": bool(mask[-1].any()),
                "mask_camerahmr_box_iou": camera_box_iou,
                "mask_sapiens2_pose_box_iou": pose_box_iou,
                "normal_mean_unit_length_error": normal_unit_error,
            }
        )

    if len(frame_metrics) != len(manifest.frames):
        blockers.append("qualified_frame_count_mismatch")
    beta_array = np.asarray([frame.betas[:10] for frame in camera.frames], dtype=np.float64)
    focal_array = np.asarray([frame.focal_length_px for frame in camera.frames], dtype=np.float64)
    maximum_beta_standard_deviation = float(beta_array.std(axis=0).max())
    focal_coefficient_of_variation = float(focal_array.std() / focal_array.mean())
    warnings: list[str] = []
    if frame_metrics and all(metric["mask_touches_bottom_edge"] for metric in frame_metrics):
        warnings.append("upper_body_capture_is_cropped_at_bottom_in_every_view")
    if maximum_beta_standard_deviation > 0.25:
        warnings.append("framewise_camerahmr_shape_varies; shared_shape_refinement_is_required")

    report = {
        "schema_version": "frayid_adapter_probe_qualification.v1",
        "status": (
            "blocked"
            if blockers
            else (
                "qualified_for_initialization_refinement"
                if adapter_manifest["schema_version"] == "frayid_capture_adapter_interval.v1"
                else "qualified_for_full_interval_evidence_processing"
            )
        ),
        "scientific_scope": (
            "Qualifies sparse upstream detection, pose, mask, and normal evidence only; "
            "does not establish a trained or evaluated canonical surface."
        ),
        "dataset_manifest_path": str(results_root / "dataset_manifest.json"),
        "frame_count": len(frame_metrics),
        "metrics": {
            "minimum_camerahmr_detection_score": min(
                (metric["camerahmr_detection_score"] for metric in frame_metrics), default=None
            ),
            "minimum_sapiens2_confident_body_joint_count": min(
                (
                    metric["sapiens2_confident_body_joint_count"]
                    for metric in frame_metrics
                ),
                default=None,
            ),
            "minimum_mask_largest_component_fraction": min(
                (metric["mask_largest_component_fraction"] for metric in frame_metrics),
                default=None,
            ),
            "minimum_mask_detector_box_iou": min(
                (
                    min(metric["mask_camerahmr_box_iou"], metric["mask_sapiens2_pose_box_iou"])
                    for metric in frame_metrics
                ),
                default=None,
            ),
            "maximum_normal_mean_unit_length_error": max(
                (metric["normal_mean_unit_length_error"] for metric in frame_metrics),
                default=None,
            ),
            "maximum_framewise_beta_standard_deviation": maximum_beta_standard_deviation,
            "framewise_focal_length_coefficient_of_variation": focal_coefficient_of_variation,
        },
        "frame_metrics": frame_metrics,
        "warnings": warnings,
        "blockers": sorted(set(blockers)),
    }
    output_path = results_root / "qualification.json"
    write_json(output_path, report)
    return output_path


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def export_selfrecon_dataset(results_root: Path, output_dir: Path) -> Path:
    """Export qualified evidence to the unmodified SelfRecon dataset interface."""
    if output_dir.exists():
        raise FileExistsError(f"output path already exists: {output_dir}")
    results_root = results_root.resolve()
    qualification = read_json(results_root / "qualification.json")
    if qualification.get("status") != "qualified_for_initialization_refinement":
        raise ValueError("full-interval qualification is required before SelfRecon export")
    evaluation = read_json(results_root / "initialization_evaluation.json")
    if evaluation.get("status") != "pass":
        raise ValueError("passing shared initialization is required before SelfRecon export")
    manifest = read_dataset_manifest(results_root / "dataset_manifest.json")
    initialization = SequenceInitialization.model_validate(
        read_json(results_root / "sequence_initialization.json")
    )
    if initialization.status != "refined":
        raise ValueError("SelfRecon export requires refined initialization")
    frames_by_index = {frame.source_frame_index: frame for frame in initialization.frames}
    expected_indices = {frame.source_frame_index for frame in manifest.frames}
    if set(frames_by_index) != expected_indices:
        raise ValueError("initialization does not cover the exported interval exactly")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent)
    )
    try:
        image_root = temporary / "imgs"
        mask_root = temporary / "masks"
        normal_root = temporary / "normals"
        for directory in (image_root, mask_root, normal_root):
            directory.mkdir()

        poses: list[list[list[float]]] = []
        translations: list[list[float]] = []
        frame_bindings: list[dict[str, Any]] = []
        ordered = sorted(manifest.frames, key=lambda record: record.ordinal)
        for export_index, record in enumerate(ordered):
            source_name = Path(record.image_path).name
            image_source = results_root / "images" / source_name
            mask_source = results_root / "masks" / source_name
            normal_source = results_root / "normals" / source_name
            for source in (image_source, mask_source, normal_source):
                if not source.is_file():
                    raise FileNotFoundError(source)
            export_name = f"{export_index:06d}.png"
            _link_or_copy(image_source, image_root / export_name)
            _link_or_copy(mask_source, mask_root / export_name)
            _link_or_copy(normal_source, normal_root / export_name)

            mask = cv2.imread(str(mask_source), cv2.IMREAD_GRAYSCALE)
            if mask is None or not np.any(mask > 127):
                raise ValueError(f"invalid SelfRecon export mask: {mask_source}")
            ys, xs = np.where(mask > 127)
            rectangle = np.asarray(
                [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1],
                dtype=np.int64,
            )
            np.savetxt(image_root / f"{export_index:06d}_rect.txt", rectangle[None], fmt="%d")

            frame = frames_by_index[record.source_frame_index]
            pose = np.asarray([*frame.global_orient, *frame.body_pose], dtype=np.float32)
            if pose.shape != (72,):
                raise ValueError(
                    f"refined pose has {pose.size} values for frame {record.source_frame_index}; "
                    "SelfRecon requires 72"
                )
            poses.append(pose.reshape(24, 3).tolist())
            translations.append(frame.translation)
            frame_bindings.append(
                {
                    "selfrecon_index": export_index,
                    "source_frame_index": record.source_frame_index,
                    "source_timestamp_seconds": record.timestamp_seconds,
                    "split": record.split,
                    "source_image_name": source_name,
                }
            )

        # SelfRecon's screen projection negates camera X/Y. A 180-degree Z
        # rotation preserves the refined CameraHMR pixel convention exactly.
        np.savez(
            temporary / "camera.npz",
            fx=np.float32(initialization.shared_focal_length_px),
            fy=np.float32(initialization.shared_focal_length_px),
            cx=np.float32(initialization.shared_principal_point_px[0]),
            cy=np.float32(initialization.shared_principal_point_px[1]),
            quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
        )
        np.savez(
            temporary / "smpl_rec.npz",
            poses=np.asarray(poses, dtype=np.float32),
            shape=np.asarray(initialization.shared_betas[:10], dtype=np.float32),
            trans=np.asarray(translations, dtype=np.float32),
            gender=np.asarray("neutral"),
        )
        adapter_manifest = {
            "schema_version": "frayid_selfrecon_dataset_adapter.v1",
            "status": "ready_for_paused_runtime_binding",
            "scientific_scope": (
                "Unmodified SelfRecon input adapter for a unified canonical outer surface; "
                "this export is not a trained or evaluated reconstruction."
            ),
            "upstream_revision": "344b86fc3e7617b94b5c9da3741c764ae93cacaa",
            "source_video_sha256": manifest.input_video_sha256,
            "frame_count": len(ordered),
            "train_frame_count": manifest.train_frame_count,
            "held_out_frame_count": manifest.held_out_frame_count,
            "camera_convention": "Rz_pi_then_SelfRecon_negative_xy_projection",
            "camera_sha256": _sha256(temporary / "camera.npz"),
            "smpl_rec_sha256": _sha256(temporary / "smpl_rec.npz"),
            "imgs_tree_sha256": _tree_sha256(image_root),
            "masks_tree_sha256": _tree_sha256(mask_root),
            "normals_tree_sha256": _tree_sha256(normal_root),
            "qualification_sha256": _sha256(results_root / "qualification.json"),
            "initialization_evaluation_sha256": _sha256(
                results_root / "initialization_evaluation.json"
            ),
            "sequence_initialization_sha256": _sha256(
                results_root / "sequence_initialization.json"
            ),
            "frames": frame_bindings,
        }
        manifest_path = temporary / "adapter_manifest.json"
        write_json(manifest_path, adapter_manifest)
        temporary.rename(output_dir)
        return output_dir / manifest_path.name
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object from ffprobe")
    return payload


def probe_video(video: Path, *, ffprobe: str) -> tuple[dict[str, Any], list[float]]:
    """Return stream metadata and ordered packet timestamps for the first video stream."""
    metadata = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration,size:stream=codec_name,codec_type,width,height,"
            "pix_fmt,r_frame_rate,time_base",
            "-of",
            "json",
            str(video),
        ]
    )
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    timestamps = [float(line) for line in result.stdout.splitlines() if line.strip()]
    if not timestamps:
        raise ValueError("video has no timestamped packets")
    if any(right <= left for left, right in pairwise(timestamps)):
        raise ValueError("video packet timestamps are not strictly increasing")
    return metadata, timestamps


def select_probe_frames(
    timestamps: list[float], *, start_seconds: float, end_seconds: float, samples: int
) -> list[tuple[int, float]]:
    """Select nearest source frames at bin centers within a usable time interval."""
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if samples < 1:
        raise ValueError("samples must be positive")
    if end_seconds > timestamps[-1]:
        raise ValueError(
            f"end_seconds ({end_seconds:.3f}) exceeds the last packet timestamp "
            f"({timestamps[-1]:.3f})"
        )

    step = (end_seconds - start_seconds) / samples
    selected: list[tuple[int, float]] = []
    for ordinal in range(samples):
        target = start_seconds + (ordinal + 0.5) * step
        insertion = bisect.bisect_left(timestamps, target)
        candidates = [min(insertion, len(timestamps) - 1)]
        if insertion > 0:
            candidates.append(insertion - 1)
        index = min(candidates, key=lambda candidate: abs(timestamps[candidate] - target))
        selected.append((index, timestamps[index]))

    indices = [index for index, _ in selected]
    if len(indices) != len(set(indices)):
        raise ValueError("sampling interval is too short to select distinct source frames")
    return selected


def _extract_selected_frames(
    video: Path,
    output_directory: Path,
    selected: list[tuple[int, float]],
    *,
    ffmpeg: str,
) -> None:
    if len(selected) <= 64:
        expression = "+".join(f"eq(n\\,{index})" for index, _ in selected)
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(video),
                "-vf",
                f"select={expression}",
                "-fps_mode",
                "passthrough",
                "-start_number",
                "0",
                str(output_directory / "frame_%04d.png"),
            ],
            check=True,
        )
        return

    selected_by_index = {source_index: ordinal for ordinal, (source_index, _) in enumerate(selected)}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video for sequential extraction: {video}")
    extracted = 0
    try:
        source_index = 0
        last_selected = selected[-1][0]
        while source_index <= last_selected:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"video decode failed at source frame {source_index}")
            ordinal = selected_by_index.get(source_index)
            if ordinal is not None:
                output_path = output_directory / f"frame_{ordinal:04d}.png"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"failed to write extracted frame: {output_path}")
                extracted += 1
            source_index += 1
    finally:
        capture.release()
    if extracted != len(selected):
        raise RuntimeError(f"extracted {extracted} frames; expected {len(selected)}")


def build_adapter_probe(
    video: Path,
    output_dir: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    samples: int,
    ffmpeg: str,
    ffprobe: str,
) -> Path:
    """Extract a sparse lossless frame bundle and write its provenance manifest."""
    if output_dir.exists():
        raise FileExistsError(f"output path already exists: {output_dir}")

    video = video.resolve()
    metadata, timestamps = probe_video(video, ffprobe=ffprobe)
    selected = select_probe_frames(
        timestamps,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        samples=samples,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent)
    )
    try:
        _extract_selected_frames(video, temporary, selected, ffmpeg=ffmpeg)

        frame_paths = sorted(temporary.glob("frame_*.png"))
        if len(frame_paths) != samples:
            raise RuntimeError(f"ffmpeg extracted {len(frame_paths)} frames; expected {samples}")

        stream = metadata.get("streams", [{}])[0]
        format_metadata = metadata.get("format", {})
        records = []
        for ordinal, (frame_path, (source_index, timestamp)) in enumerate(
            zip(frame_paths, selected, strict=True)
        ):
            records.append(
                {
                    "ordinal": ordinal,
                    "source_frame_index": source_index,
                    "source_timestamp_seconds": timestamp,
                    "path": frame_path.name,
                    "sha256": _sha256(frame_path),
                }
            )

        manifest = {
            "schema_version": "frayid_capture_adapter_probe.v1",
            "purpose": "sparse CameraHMR/Sapiens qualification before full reconstruction",
            "scientific_scope": (
                "Input evidence for a unified canonical outer surface only; this bundle does "
                "not establish a trained or evaluated reconstruction."
            ),
            "source": {
                "path": str(video),
                "sha256": _sha256(video),
                "size_bytes": int(format_metadata["size"]),
                "duration_seconds": float(format_metadata["duration"]),
                "codec": stream.get("codec_name"),
                "pixel_format": stream.get("pix_fmt"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "r_frame_rate": stream.get("r_frame_rate"),
                "packet_count": len(timestamps),
                "first_timestamp_seconds": timestamps[0],
                "last_timestamp_seconds": timestamps[-1],
            },
            "sampling": {
                "strategy": "nearest source frame to each equal-width temporal bin center",
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "sample_count": samples,
            },
            "frames": records,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output_dir)
        return output_dir / manifest_path.name
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_adapter_interval(
    video: Path,
    output_dir: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    frame_count: int,
    held_out_stride: int,
    ffmpeg: str,
    ffprobe: str,
) -> Path:
    """Extract a deterministic full interval with an interleaved held-out split."""
    if held_out_stride < 2:
        raise ValueError("held_out_stride must be at least 2")
    manifest_path = build_adapter_probe(
        video,
        output_dir,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        samples=frame_count,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    payload = _load_probe(manifest_path)
    payload["schema_version"] = "frayid_capture_adapter_interval.v1"
    payload["purpose"] = "full-interval CameraHMR/Sapiens evidence for reconstruction"
    payload["scientific_scope"] = (
        "Upstream evidence for a unified canonical outer surface only; this interval does "
        "not establish a trained or evaluated reconstruction."
    )
    payload["sampling"]["held_out_stride"] = held_out_stride
    for frame in payload["frames"]:
        frame["split"] = (
            "held_out" if frame["ordinal"] % held_out_stride == 0 else "train"
        )
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path
