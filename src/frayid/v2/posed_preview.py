from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from frayid.config import ReconstructionConfig
from frayid.dataset import read_dataset_manifest
from frayid.geometry import linear_blend_skinning, rigid_transform_from_axis_angle
from frayid.io import read_json, sha256_file, write_json
from frayid.training import CanonicalGeometryModel
from frayid.v2.contracts import reject_sealed_capability

POSED_PREVIEW_REPORT_SCHEMA = "frayid_v2_posed_preview.v1"


def _interpolate_state(source_index: int, trained_indices: list[int], values: Tensor) -> Tensor:
    order = np.argsort(trained_indices)
    indices = np.asarray(trained_indices, dtype=np.int64)[order]
    ordered = values[torch.as_tensor(order, dtype=torch.long, device=values.device)]
    position = int(np.searchsorted(indices, source_index))
    if position == 0:
        return ordered[0]
    if position == len(indices):
        return ordered[-1]
    left = int(indices[position - 1])
    right = int(indices[position])
    fraction = (source_index - left) / max(right - left, 1)
    return ordered[position - 1] * (1.0 - fraction) + ordered[position] * fraction


def _project_vertices(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = source_size
    output_height, output_width = output_size
    scale_x = output_width / source_width
    scale_y = output_height / source_height
    safe_depth = np.maximum(vertices[:, 2], 1.0e-5)
    return np.stack(
        (
            (intrinsics[0, 0] * vertices[:, 0] / safe_depth + intrinsics[0, 2]) * scale_x,
            (intrinsics[1, 1] * vertices[:, 1] / safe_depth + intrinsics[1, 2]) * scale_y,
        ),
        axis=-1,
    )


def render_shaded_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Render a deterministic neutral diagnostic with painter depth ordering."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("posed preview vertices must have shape [V,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("posed preview faces must have shape [F,3]")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("posed preview faces contain invalid indices")
    height, width = output_size
    if height <= 0 or width <= 0 or np.any(vertices[:, 2] <= 0.0):
        raise ValueError("posed preview requires positive image size and camera depth")
    pixels = _project_vertices(
        vertices,
        np.asarray(intrinsics, dtype=np.float64),
        source_size=source_size,
        output_size=output_size,
    )
    triangles_3d = vertices[faces]
    triangle_depth = triangles_3d[..., 2].mean(axis=1)
    edges_a = triangles_3d[:, 1] - triangles_3d[:, 0]
    edges_b = triangles_3d[:, 2] - triangles_3d[:, 0]
    normals = np.cross(edges_a, edges_b)
    normal_length = np.linalg.norm(normals, axis=1)
    valid = normal_length > 1.0e-12
    normals[valid] /= normal_length[valid, None]
    light = np.asarray([-0.35, -0.45, -0.82], dtype=np.float64)
    light /= np.linalg.norm(light)
    intensity = np.clip(np.abs(normals @ light), 0.0, 1.0)
    intensity = 0.28 + 0.72 * intensity
    canvas = np.full((height, width, 3), 244, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    # Draw far to near. This is a presentation diagnostic, never the scientific
    # evaluator or evidence renderer.
    for face_index in np.argsort(triangle_depth)[::-1]:
        if not valid[face_index]:
            continue
        polygon_float = pixels[faces[face_index]]
        if (
            polygon_float[:, 0].max() < 0
            or polygon_float[:, 1].max() < 0
            or polygon_float[:, 0].min() >= width
            or polygon_float[:, 1].min() >= height
        ):
            continue
        polygon = np.rint(polygon_float).astype(np.int32)
        value = float(intensity[face_index])
        color = (
            int(138 * value),
            int(183 * value),
            int(222 * value),
        )
        cv2.fillConvexPoly(canvas, polygon, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_8)
    return canvas, mask


def load_frozen_v1_model(
    checkpoint_path: Path,
    config: ReconstructionConfig,
) -> tuple[CanonicalGeometryModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "canonical_checkpoint.v1":
        raise ValueError("posed preview received an incompatible V1 checkpoint")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("posed preview checkpoint has no model state")
    required = {"base_vertices", "faces", "skinning_weights", "root_rotation_corrections_raw"}
    if not required.issubset(state):
        raise ValueError("posed preview checkpoint state is incomplete")
    frame_count = int(state["root_rotation_corrections_raw"].shape[0])
    model = CanonicalGeometryModel(
        state["base_vertices"],
        state["faces"],
        state["skinning_weights"],
        frame_count,
        config,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def frozen_v1_posed_vertices(
    model: CanonicalGeometryModel,
    transform: Tensor,
    source_index: int,
    trained_indices: list[int],
    trained_slot: dict[int, int],
) -> Tensor:
    slot = trained_slot.get(source_index)
    if slot is not None:
        return model.posed_vertices(slot, transform)[0]
    code = _interpolate_state(source_index, trained_indices, model.deformer.frame_codes.weight)
    rotation = _interpolate_state(source_index, trained_indices, model.root_rotation_corrections)
    translation = _interpolate_state(
        source_index, trained_indices, model.root_translation_corrections
    )
    correction = rigid_transform_from_axis_angle(rotation, translation)
    residual = model.deformer.forward_with_code(model.canonical_vertices, code)
    return linear_blend_skinning(
        model.canonical_vertices + residual,
        model.skinning_weights,
        correction.unsqueeze(0) @ transform,
    )


def open_video_writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create posed preview: {path}")
    return writer


def annotate_panel(image: np.ndarray, label: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(
        image,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def render_posed_preview(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    image_root: Path,
    mask_root: Path,
    output_root: Path,
    output_width: int = 288,
    fps: float = 30.0,
) -> Path:
    """Write neutral and side-by-side pose replays of a frozen V1 endpoint."""

    paths = [
        checkpoint_path,
        manifest_path,
        joint_transforms_path,
        image_root,
        mask_root,
        output_root,
    ]
    reject_sealed_capability(paths)
    if output_root.exists():
        raise FileExistsError("posed preview output is immutable")
    if output_width < 64 or fps <= 0.0 or not math.isfinite(fps):
        raise ValueError("posed preview dimensions or frame rate are invalid")
    manifest = read_dataset_manifest(manifest_path)
    model, _checkpoint = load_frozen_v1_model(checkpoint_path, config)
    with np.load(joint_transforms_path, allow_pickle=False) as archive:
        source_indices = archive["source_frame_indices"].astype(np.int64)
        transforms = archive["transforms"].astype(np.float32)
    if transforms.shape != (len(source_indices), 24, 4, 4):
        raise ValueError("posed preview joint transform archive is invalid")
    transform_lookup = {int(source): slot for slot, source in enumerate(source_indices)}
    trained_indices = [
        record.source_frame_index for record in manifest.frames if record.split == "train"
    ]
    if len(trained_indices) != model.root_rotation_corrections.shape[0]:
        raise ValueError("posed preview train split does not match its checkpoint")
    trained_slot = {source: slot for slot, source in enumerate(trained_indices)}
    source_height = config.dataset.output_height
    source_width = config.dataset.output_width
    output_height = round(output_width * source_height / source_width)
    # The V1 config does not duplicate initialization camera values. Resolve
    # them from the dataset initialization record bound by the caller's root.
    initialization_path = manifest_path.with_name(config.evidence.initialization_filename)
    initialization = read_json(initialization_path)
    focal = float(initialization["shared_focal_length_px"])
    principal = initialization["shared_principal_point_px"]
    intrinsics = np.asarray(
        [[focal, 0.0, principal[0]], [0.0, focal, principal[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    neutral_path = output_root / "posed_neutral_replay.mp4"
    comparison_path = output_root / "posed_source_comparison.mp4"
    neutral_writer = open_video_writer(neutral_path, (output_width, output_height), fps)
    comparison_writer = open_video_writer(
        comparison_path,
        (output_width * 3, output_height),
        fps,
    )
    split_ious: dict[str, list[float]] = {"train": [], "held_out": []}
    try:
        with torch.no_grad():
            for record in manifest.frames:
                transform_slot = transform_lookup.get(record.source_frame_index)
                if transform_slot is None:
                    raise ValueError("posed preview lacks a frozen frame transform")
                posed = frozen_v1_posed_vertices(
                    model,
                    torch.from_numpy(transforms[transform_slot]),
                    record.source_frame_index,
                    trained_indices,
                    trained_slot,
                )
                render, render_mask = render_shaded_mesh(
                    posed.cpu().numpy(),
                    model.faces.cpu().numpy(),
                    intrinsics,
                    source_size=(source_height, source_width),
                    output_size=(output_height, output_width),
                )
                name = Path(record.image_path).name
                source = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
                target_mask = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
                if source is None or target_mask is None:
                    raise FileNotFoundError(f"posed preview evidence is absent: {name}")
                source = cv2.resize(
                    source, (output_width, output_height), interpolation=cv2.INTER_AREA
                )
                target_mask = cv2.resize(
                    target_mask,
                    (output_width, output_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                predicted = render_mask > 0
                target = target_mask > 127
                union = np.logical_or(predicted, target).sum()
                iou = float(np.logical_and(predicted, target).sum() / max(union, 1))
                split_ious[record.split].append(iou)
                overlay = source.copy()
                foreground = predicted
                overlay[foreground] = (
                    0.45 * overlay[foreground].astype(np.float32)
                    + 0.55 * render[foreground].astype(np.float32)
                ).astype(np.uint8)
                annotate_panel(source, "source display (not redistributed)")
                annotate_panel(render, "frozen V1 geometry, posed replay")
                annotate_panel(overlay, "registration overlay")
                neutral_writer.write(render)
                comparison_writer.write(np.concatenate((source, render, overlay), axis=1))
    finally:
        neutral_writer.release()
        comparison_writer.release()
    report = {
        "schema_version": POSED_PREVIEW_REPORT_SCHEMA,
        "status": "diagnostic_only",
        "representation": "frozen_v1_canonical_geometry_with_frozen_pose_replay",
        "frame_count": len(manifest.frames),
        "fps": fps,
        "render_width": output_width,
        "render_height": output_height,
        "train_median_hard_raster_iou": float(np.median(split_ious["train"])),
        "held_out_median_hard_raster_iou": float(np.median(split_ious["held_out"])),
        "development_records_read_for_display_and_diagnosis": len(split_ious["held_out"]),
        "optimizer_steps": 0,
        "sealed_test_accesses": 0,
        "authoritative_result_claimed": False,
        "renderer_role": "presentation_diagnostic_not_scientific_evaluator",
        "source_hashes": {
            "checkpoint": sha256_file(checkpoint_path),
            "manifest": sha256_file(manifest_path),
            "joint_transforms": sha256_file(joint_transforms_path),
        },
        "artifacts": {
            "neutral_replay": {
                "path": str(neutral_path),
                "sha256": sha256_file(neutral_path),
            },
            "source_comparison": {
                "path": str(comparison_path),
                "sha256": sha256_file(comparison_path),
            },
        },
        "limitations": [
            "no RGB appearance model",
            "no audio",
            "software painter renderer is not the scientific evaluator",
            "held-out frames interpolate learned residual and root-correction states",
        ],
    }
    return write_json(output_root / "posed_preview_report.json", report)
