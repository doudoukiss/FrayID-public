from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frayid.io import write_json
from frayid.schemas import (
    CameraHMRFrame,
    FrameRecord,
    ObservedPoseFrame,
    ObservedPoseSequence,
    SequenceInitialization,
)

SAPIENS2_BODY12_INDICES = (5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)


def write_sapiens2_evidence(
    *,
    labels_path: Path,
    normals_path: Path,
    mask_output_path: Path,
    normal_output_path: Path,
    expected_size: tuple[int, int],
) -> None:
    """Convert official Sapiens2 arrays into the V1 loss contract.

    ``expected_size`` is ``(height, width)``. Sapiens2 normals are XYZ unit
    vectors in the camera frame; the PNG stores RGB=(XYZ+1)/2 and OpenCV writes
    that RGB payload through its BGR array convention.
    """
    labels = np.load(labels_path, allow_pickle=False)
    normals = np.load(normals_path, allow_pickle=False)
    if labels.shape != expected_size:
        raise ValueError(f"Sapiens2 labels have shape {labels.shape}, expected {expected_size}")
    if normals.shape != (*expected_size, 3):
        raise ValueError(
            f"Sapiens2 normals have shape {normals.shape}, expected {(*expected_size, 3)}"
        )
    if not np.isfinite(normals).all():
        raise ValueError("Sapiens2 normals contain non-finite values")
    mask = labels > 0
    foreground_fraction = float(mask.mean())
    if not 0.01 < foreground_fraction < 0.90:
        raise ValueError(f"Sapiens2 foreground fraction is implausible: {foreground_fraction:.6f}")
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normalized = normals / np.clip(lengths, 1e-8, None)
    normalized[~mask] = -1.0
    encoded_rgb = np.rint((np.clip(normalized, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    normal_output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_output_path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write Sapiens2 mask: {mask_output_path}")
    if not cv2.imwrite(str(normal_output_path), encoded_rgb[..., ::-1]):
        raise RuntimeError(f"Failed to write Sapiens2 normal: {normal_output_path}")


def convert_sapiens2_directory(
    *,
    frame_records: Sequence[FrameRecord],
    segmentation_directory: Path,
    normal_directory: Path,
    mask_output_directory: Path,
    normal_output_directory: Path,
    image_directory: Path | None = None,
) -> int:
    converted = 0
    for record in frame_records:
        manifest_image_path = Path(record.image_path)
        image_path = (
            image_directory / manifest_image_path.name
            if image_directory is not None
            else manifest_image_path
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Prepared RGB frame is unreadable: {image_path}")
        stem = image_path.stem
        write_sapiens2_evidence(
            labels_path=segmentation_directory / f"{stem}_seg.npy",
            normals_path=normal_directory / f"{stem}.npy",
            mask_output_path=mask_output_directory / manifest_image_path.name,
            normal_output_path=normal_output_directory / manifest_image_path.name,
            expected_size=image.shape[:2],
        )
        converted += 1
    return converted


def build_sequence_initialization(
    *,
    frame_records: Sequence[FrameRecord],
    raw_frames: Sequence[dict[str, Any]],
    image_width: int,
    image_height: int,
    source_revision: str,
    checkpoint_sha256: str,
    camera_checkpoint_sha256: str,
    detector_checkpoint_sha256: str,
    output_path: Path | None = None,
) -> SequenceInitialization:
    """Validate and consolidate one real CameraHMR result per selected frame."""
    parsed = [CameraHMRFrame.model_validate(item) for item in raw_frames]
    expected = {record.source_frame_index for record in frame_records}
    observed = {frame.source_frame_index for frame in parsed}
    if len(observed) != len(parsed):
        raise ValueError("CameraHMR output contains duplicate source frame indices")
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if missing or extra:
        raise ValueError(
            f"CameraHMR frame coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    frames = sorted(parsed, key=lambda frame: frame.source_frame_index)
    shared_betas = np.median(
        np.asarray([frame.betas[:10] for frame in frames], dtype=np.float64), axis=0
    ).tolist()
    shared_focal = float(np.median([frame.focal_length_px for frame in frames]))
    shared_principal = np.median(
        np.asarray([frame.principal_point_px for frame in frames], dtype=np.float64), axis=0
    ).tolist()
    initialization = SequenceInitialization(
        status="raw",
        shared_betas=shared_betas,
        shared_focal_length_px=shared_focal,
        shared_principal_point_px=shared_principal,
        image_width=image_width,
        image_height=image_height,
        frames=frames,
        source_revision=source_revision,
        checkpoint_sha256=checkpoint_sha256,
        camera_checkpoint_sha256=camera_checkpoint_sha256,
        detector_checkpoint_sha256=detector_checkpoint_sha256,
    )
    if output_path is not None:
        write_json(output_path, initialization)
    return initialization


def build_observed_pose_sequence(
    *,
    frame_records: Sequence[FrameRecord],
    raw_payload: dict[str, Any],
    image_width: int,
    image_height: int,
    source_revision: str,
    model_revision: str,
    detector_revision: str,
    checkpoint_sha256: str,
    detector_checkpoint_sha256: str,
    output_path: Path | None = None,
) -> ObservedPoseSequence:
    """Consolidate official Sapiens2 detections into independent COCO-17 evidence."""
    raw_frames = raw_payload.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("Sapiens2 pose payload is missing its frame list")
    records_by_name = {Path(record.image_path).name: record for record in frame_records}
    observed: list[ObservedPoseFrame] = []
    for item in raw_frames:
        if not isinstance(item, dict):
            raise ValueError("Sapiens2 pose frame must be an object")
        image_name = item.get("image_name")
        if not isinstance(image_name, str) or image_name not in records_by_name:
            raise ValueError(f"Unexpected Sapiens2 pose image: {image_name!r}")
        instances = item.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError(f"Sapiens2 pose found no person in {image_name}")
        instance = max(instances, key=_pose_instance_area)
        if not isinstance(instance, dict):
            raise ValueError(f"Invalid Sapiens2 pose instance in {image_name}")
        coordinates = np.asarray(instance.get("keypoints"), dtype=np.float64)
        scores = np.asarray(instance.get("keypoint_scores"), dtype=np.float64).reshape(-1)
        bbox = np.asarray(instance.get("bbox"), dtype=np.float64).reshape(-1)
        if coordinates.shape != (308, 2) or scores.shape != (308,):
            raise ValueError(f"Sapiens2 pose did not emit 308 keypoints for {image_name}")
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError(f"Sapiens2 pose emitted an invalid box for {image_name}")
        body_indices = list(SAPIENS2_BODY12_INDICES)
        raw_body12 = np.concatenate(
            (coordinates[body_indices], scores[body_indices, None]), axis=-1
        )
        if not np.isfinite(raw_body12).all():
            raise ValueError(f"Sapiens2 pose emitted non-finite keypoints for {image_name}")
        # UDP heatmap maxima are evidence weights, not calibrated probabilities,
        # and can marginally exceed one. Clamp only for a stable [0, 1] loss weight.
        body12 = raw_body12.copy()
        body12[:, 2] = np.clip(body12[:, 2], 0.0, 1.0)
        observed.append(
            ObservedPoseFrame(
                source_frame_index=records_by_name[image_name].source_frame_index,
                keypoints_body12=body12.tolist(),
                bounding_box_xyxy=bbox.tolist(),
            )
        )
    expected = {record.source_frame_index for record in frame_records}
    indices = [frame.source_frame_index for frame in observed]
    if len(indices) != len(set(indices)):
        raise ValueError("Sapiens2 pose output contains duplicate source frame indices")
    missing = expected.difference(indices)
    extra = set(indices).difference(expected)
    if missing or extra:
        raise ValueError(
            f"Sapiens2 pose coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    sequence = ObservedPoseSequence(
        image_width=image_width,
        image_height=image_height,
        frames=sorted(observed, key=lambda frame: frame.source_frame_index),
        source_revision=source_revision,
        model_revision=model_revision,
        detector_revision=detector_revision,
        checkpoint_sha256=checkpoint_sha256,
        detector_checkpoint_sha256=detector_checkpoint_sha256,
    )
    if output_path is not None:
        write_json(output_path, sequence)
    return sequence


def _pose_instance_area(value: object) -> float:
    if not isinstance(value, dict):
        return -1.0
    bbox = np.asarray(value.get("bbox"), dtype=np.float64).reshape(-1)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return -1.0
    return float(max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0))
