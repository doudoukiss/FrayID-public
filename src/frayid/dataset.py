from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import TypeAdapter

from frayid.assets import probe_video
from frayid.config import ReconstructionConfig
from frayid.io import read_json, sha256_file, write_json
from frayid.schemas import (
    DatasetManifest,
    DatasetValidationReport,
    EvidenceFrameCheck,
    FrameRecord,
    ObservedPoseSequence,
    SequenceInitialization,
)

DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
DATASET_VALIDATION_FILENAME = "dataset_validation.json"


def prepare_dataset(config: ReconstructionConfig, *, overwrite: bool = False) -> DatasetManifest:
    video_path = config.paths.input_video
    dataset_root = config.paths.dataset_root
    image_root = dataset_root / "images"
    manifest_path = dataset_root / DATASET_MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        return read_dataset_manifest(manifest_path)
    image_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / config.evidence.masks_subdirectory).mkdir(parents=True, exist_ok=True)
    (dataset_root / config.evidence.normals_subdirectory).mkdir(parents=True, exist_ok=True)

    metadata = probe_video(video_path)
    candidate_count = min(
        metadata.frame_count,
        config.dataset.target_frame_count * config.dataset.candidate_multiplier,
    )
    candidate_indices = np.linspace(0, metadata.frame_count - 1, candidate_count, dtype=np.int64)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input video: {video_path}")
    accepted: list[tuple[int, np.ndarray, float, float]] = []
    rejected_count = 0
    try:
        for source_index in candidate_indices.tolist():
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                rejected_count += 1
                continue
            blur, luminance = _frame_quality(frame)
            if not _quality_accepted(blur, luminance, config):
                rejected_count += 1
                continue
            accepted.append((source_index, frame, blur, luminance))
    finally:
        capture.release()

    selected = _evenly_select(accepted, config.dataset.target_frame_count)
    frames: list[FrameRecord] = []
    for ordinal, (source_index, frame, blur, luminance) in enumerate(selected):
        filename = f"frame_{ordinal:04d}_source_{source_index:06d}.png"
        output_path = image_root / filename
        resized = cv2.resize(
            frame,
            (config.dataset.output_width, config.dataset.output_height),
            interpolation=cv2.INTER_AREA,
        )
        if not cv2.imwrite(str(output_path), resized):
            raise RuntimeError(f"Failed to write frame: {output_path}")
        split: Literal["train", "held_out"] = (
            "held_out" if ordinal % config.dataset.held_out_stride == 0 else "train"
        )
        frames.append(
            FrameRecord(
                ordinal=ordinal,
                source_frame_index=source_index,
                timestamp_seconds=source_index / metadata.frame_rate,
                image_path=str(output_path),
                split=split,
                blur_variance=blur,
                mean_luminance=luminance,
                quality_accepted=True,
            )
        )

    blockers: list[str] = []
    if len(frames) < config.dataset.minimum_usable_frame_count:
        blockers.append("usable_frame_count_below_minimum")
    blockers.extend(
        [
            "real_sapiens2_masks_required",
            "real_sapiens2_normals_required",
            "real_sapiens2_observed_pose_required",
            "real_camerahmr_smpl_initialization_required",
        ]
    )
    manifest = DatasetManifest(
        status="blocked"
        if len(frames) < config.dataset.minimum_usable_frame_count
        else "rgb_ready",
        run_id=config.run_id,
        input_video_path=str(video_path),
        input_video_sha256=sha256_file(video_path),
        video=metadata,
        dataset_root=str(dataset_root),
        frames=frames,
        train_frame_count=sum(frame.split == "train" for frame in frames),
        held_out_frame_count=sum(frame.split == "held_out" for frame in frames),
        rejected_candidate_count=rejected_count + max(0, len(accepted) - len(selected)),
        blockers=blockers,
    )
    write_json(manifest_path, manifest)
    return manifest


def validate_dataset(
    config: ReconstructionConfig,
    *,
    manifest_path: Path | None = None,
) -> DatasetValidationReport:
    path = manifest_path or config.paths.dataset_root / DATASET_MANIFEST_FILENAME
    manifest = read_dataset_manifest(path)
    dataset_root = Path(manifest.dataset_root)
    mask_root = dataset_root / config.evidence.masks_subdirectory
    normal_root = dataset_root / config.evidence.normals_subdirectory
    frame_checks: list[EvidenceFrameCheck] = []
    complete_count = 0
    for frame in manifest.frames:
        image_path = Path(frame.image_path)
        mask_path = mask_root / image_path.name
        normal_path = normal_root / image_path.name
        check = _validate_frame_evidence(frame, image_path, mask_path, normal_path)
        frame_checks.append(check)
        if not check.blockers:
            complete_count += 1

    initialization_path = dataset_root / config.evidence.initialization_filename
    observed_pose_path = dataset_root / config.evidence.observed_pose_filename
    blockers: list[str] = []
    if len(manifest.frames) < config.dataset.minimum_usable_frame_count:
        blockers.append("selected_frame_count_below_minimum")
    if complete_count < config.dataset.minimum_usable_frame_count:
        blockers.append("evidence_complete_frame_count_below_minimum")
    initialization_present = initialization_path.is_file()
    observed_pose_present = observed_pose_path.is_file()
    expected_indices = {frame.source_frame_index for frame in manifest.frames}
    if not observed_pose_present:
        blockers.append("missing_real_sapiens2_observed_pose")
    else:
        try:
            observed_pose = ObservedPoseSequence.model_validate(read_json(observed_pose_path))
            blockers.extend(
                _observed_pose_evidence_blockers(
                    observed_pose,
                    expected_indices,
                    expected_size=(manifest.video.width, manifest.video.height),
                )
            )
        except (OSError, ValueError) as exc:
            blockers.append(f"invalid_sapiens2_observed_pose:{type(exc).__name__}")
    if not initialization_present:
        blockers.append("missing_real_camerahmr_smpl_initialization")
    else:
        try:
            initialization = SequenceInitialization.model_validate(read_json(initialization_path))
            blockers.extend(
                _initialization_evidence_blockers(
                    initialization,
                    expected_indices,
                )
            )
        except (OSError, ValueError) as exc:
            blockers.append(f"invalid_camerahmr_smpl_initialization:{type(exc).__name__}")
    report = DatasetValidationReport(
        status="blocked" if blockers else "ready",
        dataset_manifest_path=str(path),
        selected_frame_count=len(manifest.frames),
        evidence_complete_frame_count=complete_count,
        minimum_usable_frame_count=config.dataset.minimum_usable_frame_count,
        initialization_present=initialization_present,
        observed_pose_present=observed_pose_present,
        frame_checks=frame_checks,
        blockers=blockers,
    )
    write_json(dataset_root / DATASET_VALIDATION_FILENAME, report)
    return report


def read_dataset_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate(read_json(path))


def _frame_quality(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())


def _quality_accepted(blur: float, luminance: float, config: ReconstructionConfig) -> bool:
    return (
        blur >= config.dataset.minimum_blur_variance
        and config.dataset.minimum_mean_luminance
        <= luminance
        <= config.dataset.maximum_mean_luminance
    )


def _evenly_select(
    values: list[tuple[int, np.ndarray, float, float]],
    count: int,
) -> list[tuple[int, np.ndarray, float, float]]:
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return [values[index] for index in positions.tolist()]


def _validate_frame_evidence(
    frame: FrameRecord,
    image_path: Path,
    mask_path: Path,
    normal_path: Path,
) -> EvidenceFrameCheck:
    blockers: list[str] = []
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
    normal = cv2.imread(str(normal_path), cv2.IMREAD_COLOR) if normal_path.is_file() else None
    if image is None:
        blockers.append("missing_or_unreadable_rgb")
    if mask is None:
        blockers.append("missing_real_sapiens2_mask")
    if normal is None:
        blockers.append("missing_real_sapiens2_normal")
    dimensions_match = bool(
        image is not None
        and mask is not None
        and normal is not None
        and image.shape[:2] == mask.shape[:2] == normal.shape[:2]
    )
    if image is not None and mask is not None and image.shape[:2] != mask.shape[:2]:
        blockers.append("mask_dimensions_mismatch")
    if image is not None and normal is not None and image.shape[:2] != normal.shape[:2]:
        blockers.append("normal_dimensions_mismatch")
    foreground_fraction = float(np.mean(mask > 127)) if mask is not None else None
    if foreground_fraction is not None and not 0.01 < foreground_fraction < 0.90:
        blockers.append("invalid_mask_foreground_fraction")
    normal_variance = float(np.var(normal.astype(np.float32))) if normal is not None else None
    if normal_variance is not None and normal_variance < 1.0:
        blockers.append("constant_or_degenerate_normal_map")
    return EvidenceFrameCheck(
        ordinal=frame.ordinal,
        source_frame_index=frame.source_frame_index,
        image_path=str(image_path),
        mask_path=str(mask_path),
        normal_path=str(normal_path),
        mask_present=mask is not None,
        normal_present=normal is not None,
        dimensions_match=dimensions_match,
        mask_foreground_fraction=foreground_fraction,
        normal_variance=normal_variance,
        blockers=blockers,
    )


def parse_frame_records(value: object) -> list[FrameRecord]:
    return TypeAdapter(list[FrameRecord]).validate_python(value)


def _initialization_evidence_blockers(
    initialization: SequenceInitialization,
    expected_indices: set[int],
) -> list[str]:
    blockers: list[str] = []
    if initialization.proxy_camera:
        blockers.append("proxy_camera_forbidden")
    if initialization.zero_pose:
        blockers.append("zero_pose_packet_forbidden")
    observed = {frame.source_frame_index for frame in initialization.frames}
    missing = expected_indices.difference(observed)
    if missing:
        blockers.append(f"initialization_missing_selected_frames:{len(missing)}")
    shared_betas = np.asarray(initialization.shared_betas)
    for frame in initialization.frames:
        if np.linalg.norm(frame.body_pose) + np.linalg.norm(frame.global_orient) < 1e-6:
            blockers.append(f"zero_pose_forbidden:{frame.source_frame_index}")
        if initialization.status == "refined" and not np.allclose(
            frame.betas[: shared_betas.size], shared_betas, atol=1e-4
        ):
            blockers.append(f"framewise_shape_drift:{frame.source_frame_index}")
        if initialization.status == "refined" and not np.isclose(
            frame.focal_length_px, initialization.shared_focal_length_px, rtol=1e-5
        ):
            blockers.append(f"framewise_focal_drift:{frame.source_frame_index}")
    return sorted(set(blockers))


def _observed_pose_evidence_blockers(
    sequence: ObservedPoseSequence,
    expected_indices: set[int],
    *,
    expected_size: tuple[int, int],
) -> list[str]:
    blockers: list[str] = []
    if sequence.proxy_evidence:
        blockers.append("proxy_observed_pose_forbidden")
    if (sequence.image_width, sequence.image_height) != expected_size:
        blockers.append("observed_pose_dimensions_mismatch")
    indices = [frame.source_frame_index for frame in sequence.frames]
    if len(indices) != len(set(indices)):
        blockers.append("duplicate_observed_pose_frame_indices")
    missing = expected_indices.difference(indices)
    extra = set(indices).difference(expected_indices)
    if missing:
        blockers.append(f"observed_pose_missing_selected_frames:{len(missing)}")
    if extra:
        blockers.append(f"observed_pose_has_extra_frames:{len(extra)}")
    for frame in sequence.frames:
        keypoints = np.asarray(frame.keypoints_body12, dtype=np.float64)
        if keypoints.shape != (12, 3) or not np.isfinite(keypoints).all():
            blockers.append(f"invalid_observed_pose:{frame.source_frame_index}")
        elif np.any((keypoints[:, 2] < 0.0) | (keypoints[:, 2] > 1.0)):
            blockers.append(f"invalid_observed_pose_confidence:{frame.source_frame_index}")
    return sorted(set(blockers))
