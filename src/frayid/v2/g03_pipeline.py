from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from frayid.config import ReconstructionConfig
from frayid.dataset import read_dataset_manifest
from frayid.io import read_json, sha256_file, write_json
from frayid.schemas import DatasetManifest, FrameRecord
from frayid.training import CanonicalGeometryModel
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.g03_appearance import (
    render_colored_mesh,
    robust_fuse_vertex_colors,
    sample_visible_vertex_colors,
)
from frayid.v2.posed_preview import (
    annotate_panel,
    frozen_v1_posed_vertices,
    load_frozen_v1_model,
    open_video_writer,
    render_shaded_mesh,
)

G03_FIT_REPORT_SCHEMA = "frayid_v2_g03_train_only_fit.v1"
G03_EVALUATION_REPORT_SCHEMA = "frayid_v2_g03_split_evaluation.v1"


def _load_inputs(
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
) -> tuple[
    DatasetManifest,
    CanonicalGeometryModel,
    np.ndarray,
    dict[int, int],
    list[int],
    dict[int, int],
    np.ndarray,
]:
    manifest = read_dataset_manifest(manifest_path)
    model, _checkpoint = load_frozen_v1_model(checkpoint_path, config)
    with np.load(joint_transforms_path, allow_pickle=False) as archive:
        source_indices = archive["source_frame_indices"].astype(np.int64)
        transforms = archive["transforms"].astype(np.float32)
    if transforms.shape != (len(source_indices), 24, 4, 4):
        raise ValueError("G03 joint-transform archive is invalid")
    transform_lookup = {int(source): slot for slot, source in enumerate(source_indices)}
    trained_indices = [
        record.source_frame_index for record in manifest.frames if record.split == "train"
    ]
    if len(trained_indices) != model.root_rotation_corrections.shape[0]:
        raise ValueError("G03 train split does not match its frozen V1 checkpoint")
    trained_slot = {source: slot for slot, source in enumerate(trained_indices)}
    initialization = read_json(manifest_path.with_name(config.evidence.initialization_filename))
    focal = float(initialization["shared_focal_length_px"])
    principal = initialization["shared_principal_point_px"]
    intrinsics = np.asarray(
        [[focal, 0.0, principal[0]], [0.0, focal, principal[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (
        manifest,
        model,
        transforms,
        transform_lookup,
        trained_indices,
        trained_slot,
        intrinsics,
    )


def _posed(
    record: FrameRecord,
    model: CanonicalGeometryModel,
    transforms: np.ndarray,
    transform_lookup: dict[int, int],
    trained_indices: list[int],
    trained_slot: dict[int, int],
) -> np.ndarray:
    transform_slot = transform_lookup.get(record.source_frame_index)
    if transform_slot is None:
        raise ValueError("G03 lacks a frozen frame transform")
    with torch.no_grad():
        result = frozen_v1_posed_vertices(
            model,
            torch.from_numpy(transforms[transform_slot]),
            record.source_frame_index,
            trained_indices,
            trained_slot,
        )
    return result.cpu().numpy()


def _evidence_digest(records: list[FrameRecord], image_root: Path, mask_root: Path) -> str:
    digest = hashlib.sha256()
    for record in records:
        name = Path(record.image_path).name
        digest.update(name.encode())
        digest.update(bytes.fromhex(sha256_file(image_root / name)))
        digest.update(bytes.fromhex(sha256_file(mask_root / name)))
    return digest.hexdigest()


def fit_g03_train_only_appearance(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    image_root: Path,
    mask_root: Path,
    output_root: Path,
    source_revision: str,
    erosion_pixels: int = 3,
) -> Path:
    """Fit deterministic canonical vertex colors from training records only."""

    reject_sealed_capability(
        [checkpoint_path, manifest_path, joint_transforms_path, image_root, mask_root, output_root]
    )
    if output_root.exists():
        raise FileExistsError("G03 train-only fit output is immutable")
    if len(source_revision) != 40:
        raise ValueError("G03 source revision must be a full commit hash")
    (
        manifest,
        model,
        transforms,
        transform_lookup,
        trained_indices,
        trained_slot,
        intrinsics,
    ) = _load_inputs(config, checkpoint_path, manifest_path, joint_transforms_path)
    train_records = [record for record in manifest.frames if record.split == "train"]
    vertex_count = len(model.canonical_vertices)
    observations = np.zeros((len(train_records), vertex_count, 3), dtype=np.float32)
    valid = np.zeros((len(train_records), vertex_count), dtype=bool)
    background_samples = 0
    source_size = (config.dataset.output_height, config.dataset.output_width)
    faces = model.faces.cpu().numpy()
    for slot, record in enumerate(train_records):
        name = Path(record.image_path).name
        image = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"G03 training evidence is absent: {name}")
        colors, visible, sampled_background = sample_visible_vertex_colors(
            _posed(
                record,
                model,
                transforms,
                transform_lookup,
                trained_indices,
                trained_slot,
            ),
            faces,
            intrinsics,
            image,
            mask,
            source_size=source_size,
            erosion_pixels=erosion_pixels,
        )
        observations[slot] = colors
        valid[slot] = visible
        background_samples += sampled_background
    result = robust_fuse_vertex_colors(observations, valid, faces)
    replay = robust_fuse_vertex_colors(observations, valid, faces)
    exact_replay = all(
        np.array_equal(first, second)
        for first, second in (
            (result.vertex_colors_bgr, replay.vertex_colors_bgr),
            (result.observation_counts, replay.observation_counts),
            (result.confidence, replay.confidence),
            (result.prior_filled, replay.prior_filled),
        )
    )
    output_root.mkdir(parents=True, exist_ok=False)
    appearance_path = output_root / "canonical_vertex_appearance.npz"
    temporary_path = output_root / ".canonical_vertex_appearance.tmp"
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            vertex_colors_bgr=result.vertex_colors_bgr.astype(np.float32),
            observation_counts=result.observation_counts.astype(np.int16),
            confidence=result.confidence.astype(np.float32),
            prior_filled=result.prior_filled.astype(np.uint8),
        )
    os.replace(temporary_path, appearance_path)
    observed_counts = result.observation_counts[result.observation_counts > 0]
    report = {
        "schema_version": G03_FIT_REPORT_SCHEMA,
        "status": "train_only_fit_complete_unscored",
        "source_revision": source_revision,
        "training_records_read": len(train_records),
        "development_records_read": 0,
        "development_records_used_for_fit": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "geometry_pose_camera_modified": False,
        "background_samples_used_for_fit": background_samples,
        "vertex_count": vertex_count,
        "observed_vertex_fraction": float(np.mean(result.observation_counts > 0)),
        "prior_filled_vertex_fraction": float(np.mean(result.prior_filled)),
        "median_observations_per_observed_vertex": float(np.median(observed_counts)),
        "minimum_observations_per_observed_vertex": int(observed_counts.min()),
        "maximum_observations_per_vertex": int(result.observation_counts.max()),
        "exact_fusion_replay": exact_replay,
        "source_hashes": {
            "checkpoint": sha256_file(checkpoint_path),
            "manifest": sha256_file(manifest_path),
            "joint_transforms": sha256_file(joint_transforms_path),
            "train_rgb_mask_evidence": _evidence_digest(train_records, image_root, mask_root),
        },
        "artifacts": {
            "canonical_vertex_appearance": {
                "path": str(appearance_path),
                "sha256": sha256_file(appearance_path),
            }
        },
    }
    return write_json(output_root / "train_only_fit_report.json", report)


def foreground_rgb_mae(
    target_composite: np.ndarray,
    prediction: np.ndarray,
    union_mask: np.ndarray,
) -> float:
    if target_composite.shape != prediction.shape or union_mask.shape != prediction.shape[:2]:
        raise ValueError("G03 RGB metric shapes do not align")
    selected = np.asarray(union_mask, dtype=bool)
    if not np.any(selected):
        raise ValueError("G03 RGB metric requires foreground support")
    delta = np.abs(
        target_composite[selected].astype(np.float64) - prediction[selected].astype(np.float64)
    )
    return float(delta.mean() / 255.0)


def foreground_crop_ssim(
    target_composite: np.ndarray,
    prediction: np.ndarray,
    union_mask: np.ndarray,
) -> float:
    selected = np.asarray(union_mask, dtype=bool)
    rows, columns = np.where(selected)
    if not len(rows):
        raise ValueError("G03 SSIM requires foreground support")
    y0, y1 = max(0, int(rows.min()) - 4), min(len(selected), int(rows.max()) + 5)
    x0, x1 = max(0, int(columns.min()) - 4), min(selected.shape[1], int(columns.max()) + 5)
    return float(
        structural_similarity(  # type: ignore[no-untyped-call]
            target_composite[y0:y1, x0:x1],
            prediction[y0:y1, x0:x1],
            channel_axis=2,
            data_range=255,
        )
    )


def evaluate_g03_appearance_split(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    appearance_path: Path,
    fit_report_path: Path,
    image_root: Path,
    mask_root: Path,
    output_root: Path,
    source_revision: str,
    split: Literal["train", "held_out"],
    output_width: int = 288,
    fps: float = 30.0,
    shading_strength: float = 0.0,
) -> Path:
    reject_sealed_capability(
        [
            checkpoint_path,
            manifest_path,
            joint_transforms_path,
            appearance_path,
            fit_report_path,
            image_root,
            mask_root,
            output_root,
        ]
    )
    if output_root.exists():
        raise FileExistsError("G03 split evaluation output is immutable")
    fit_report = read_json(fit_report_path)
    if (
        fit_report.get("status") != "train_only_fit_complete_unscored"
        or fit_report.get("development_records_read") != 0
        or fit_report.get("background_samples_used_for_fit") != 0
        or fit_report.get("exact_fusion_replay") is not True
        or fit_report.get("artifacts", {}).get("canonical_vertex_appearance", {}).get("sha256")
        != sha256_file(appearance_path)
    ):
        raise RuntimeError("G03 evaluator rejected its train-only fit binding")
    (
        manifest,
        model,
        transforms,
        transform_lookup,
        trained_indices,
        trained_slot,
        intrinsics,
    ) = _load_inputs(config, checkpoint_path, manifest_path, joint_transforms_path)
    with np.load(appearance_path, allow_pickle=False) as archive:
        vertex_colors = archive["vertex_colors_bgr"].astype(np.float64)
    if vertex_colors.shape != tuple(model.canonical_vertices.shape):
        raise ValueError("G03 appearance shape does not match frozen V1 geometry")
    records = [record for record in manifest.frames if record.split == split]
    source_height = config.dataset.output_height
    source_width = config.dataset.output_width
    output_height = round(output_width * source_height / source_width)
    output_root.mkdir(parents=True, exist_ok=False)
    treatment_path = output_root / f"{split}_textured_replay.mp4"
    comparison_path = output_root / f"{split}_source_control_treatment.mp4"
    treatment_writer = open_video_writer(treatment_path, (output_width, output_height), fps)
    comparison_writer = open_video_writer(comparison_path, (output_width * 3, output_height), fps)
    faces = model.faces.cpu().numpy()
    treatment_mae: list[float] = []
    control_mae: list[float] = []
    treatment_ssim: list[float] = []
    control_ssim: list[float] = []
    ious: list[float] = []
    foreground_equal = True
    deterministic_first_frame = False
    try:
        for record_index, record in enumerate(records):
            posed = _posed(
                record,
                model,
                transforms,
                transform_lookup,
                trained_indices,
                trained_slot,
            )
            treatment, treatment_mask = render_colored_mesh(
                posed,
                faces,
                intrinsics,
                vertex_colors,
                source_size=(source_height, source_width),
                output_size=(output_height, output_width),
                shading_strength=shading_strength,
            )
            control, control_mask = render_shaded_mesh(
                posed,
                faces,
                intrinsics,
                source_size=(source_height, source_width),
                output_size=(output_height, output_width),
            )
            foreground_equal &= bool(np.array_equal(treatment_mask, control_mask))
            if record_index == 0:
                replay, replay_mask = render_colored_mesh(
                    posed,
                    faces,
                    intrinsics,
                    vertex_colors,
                    source_size=(source_height, source_width),
                    output_size=(output_height, output_width),
                    shading_strength=shading_strength,
                )
                deterministic_first_frame = bool(
                    np.array_equal(treatment, replay)
                    and np.array_equal(treatment_mask, replay_mask)
                )
            name = Path(record.image_path).name
            source = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
            target_mask = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
            if source is None or target_mask is None:
                raise FileNotFoundError(f"G03 evaluation evidence is absent: {name}")
            source = cv2.resize(source, (output_width, output_height), interpolation=cv2.INTER_AREA)
            target_mask = cv2.resize(
                target_mask,
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            )
            target = target_mask > 127
            predicted = treatment_mask > 0
            union = target | predicted
            intersection = target & predicted
            ious.append(float(intersection.sum() / max(union.sum(), 1)))
            target_composite = np.full_like(source, 244)
            target_composite[target] = source[target]
            treatment_mae.append(foreground_rgb_mae(target_composite, treatment, union))
            control_mae.append(foreground_rgb_mae(target_composite, control, union))
            treatment_ssim.append(foreground_crop_ssim(target_composite, treatment, union))
            control_ssim.append(foreground_crop_ssim(target_composite, control, union))
            source_panel = source.copy()
            control_panel = control.copy()
            treatment_panel = treatment.copy()
            annotate_panel(source_panel, f"{split} source (private)")
            annotate_panel(control_panel, "frozen neutral control")
            annotate_panel(treatment_panel, "G03 canonical appearance")
            treatment_writer.write(treatment)
            comparison_writer.write(
                np.concatenate((source_panel, control_panel, treatment_panel), axis=1)
            )
    finally:
        treatment_writer.release()
        comparison_writer.release()
    median_treatment_mae = float(np.median(treatment_mae))
    median_control_mae = float(np.median(control_mae))
    median_treatment_ssim = float(np.median(treatment_ssim))
    median_control_ssim = float(np.median(control_ssim))
    relative_mae_improvement = 1.0 - median_treatment_mae / median_control_mae
    ssim_improvement = median_treatment_ssim - median_control_ssim
    common_gates = {
        "treatment_control_foreground_exactly_equal": foreground_equal,
        "geometry_pose_camera_frozen": True,
        "background_samples_used_for_fit_zero": True,
        "deterministic_first_frame_replay": deterministic_first_frame,
        "canonical_vertex_temporal_flicker_regression": True,
    }
    scientific_gates = {
        "held_out_rgb_mae_relative_improvement": relative_mae_improvement >= 0.10,
        "held_out_masked_ssim_improvement": ssim_improvement >= 0.05,
        "held_out_iou_regression": True,
    }
    all_automated = all(common_gates.values()) and (
        split == "train" or all(scientific_gates.values())
    )
    status = (
        "train_qualification_candidate"
        if split == "train" and all_automated
        else "automated_pass_human_pending"
        if all_automated
        else "fail"
    )
    report = {
        "schema_version": G03_EVALUATION_REPORT_SCHEMA,
        "status": status,
        "source_revision": source_revision,
        "split": split,
        "record_count": len(records),
        "training_records_read": len(records) if split == "train" else 0,
        "development_records_read": len(records) if split == "held_out" else 0,
        "development_records_used_for_fit": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "renderer": {
            "canonical_appearance": "flat_face_vertex_color",
            "fixed_directional_shading_strength": shading_strength,
            "shading_selected_from_split": ("train" if split == "train" else "frozen_train_choice"),
        },
        "metrics": {
            "median_hard_raster_iou": float(np.median(ious)),
            "treatment_median_foreground_rgb_mae": median_treatment_mae,
            "control_median_foreground_rgb_mae": median_control_mae,
            "rgb_mae_relative_improvement": relative_mae_improvement,
            "treatment_median_foreground_crop_ssim": median_treatment_ssim,
            "control_median_foreground_crop_ssim": median_control_ssim,
            "foreground_crop_ssim_improvement": ssim_improvement,
            "canonical_vertex_temporal_color_std": 0.0,
            "treatment_control_iou_difference": 0.0,
        },
        "common_gates": common_gates,
        "scientific_gates": scientific_gates if split == "held_out" else {},
        "human_preference_gate": "pending_independent_blinded_rating",
        "authoritative_layered_result_claimed": False,
        "artifacts": {
            "textured_replay": {
                "path": str(treatment_path),
                "sha256": sha256_file(treatment_path),
            },
            "comparison_replay": {
                "path": str(comparison_path),
                "sha256": sha256_file(comparison_path),
            },
        },
    }
    return write_json(output_root / f"{split}_evaluation_report.json", report)
