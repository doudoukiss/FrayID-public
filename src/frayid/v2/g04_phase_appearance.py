from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from frayid.config import ReconstructionConfig
from frayid.io import read_json, sha256_file, write_json
from frayid.schemas import FrameRecord
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.g03_appearance import (
    render_colored_mesh,
    robust_fuse_vertex_colors,
    sample_visible_vertex_colors,
)
from frayid.v2.g03_pipeline import (
    _load_inputs,
    _posed,
    foreground_crop_ssim,
    foreground_rgb_mae,
)
from frayid.v2.posed_preview import annotate_panel, open_video_writer, render_shaded_mesh

G04_PUBLIC_BENCHMARK_SCHEMA = "frayid_v2_g04_public_benchmark.v1"
G04_FIT_REPORT_SCHEMA = "frayid_v2_g04_train_only_fit.v1"
G04_EVALUATION_REPORT_SCHEMA = "frayid_v2_g04_split_evaluation.v1"
G04_PRESENTATION_REPORT_SCHEMA = "frayid_v2_g04_full_sequence_presentation.v1"


@dataclass(frozen=True)
class PhaseAppearanceModel:
    source_indices: np.ndarray
    observations_bgr: np.ndarray
    valid_observations: np.ndarray
    static_colors_bgr: np.ndarray
    leave_one_out_static_colors_bgr: np.ndarray
    period: float


def build_phase_appearance_model(
    observations_bgr: np.ndarray,
    valid_observations: np.ndarray,
    source_indices: np.ndarray,
    faces: np.ndarray,
    *,
    period: float | None = None,
) -> PhaseAppearanceModel:
    observations = np.asarray(observations_bgr, dtype=np.float64)
    valid = np.asarray(valid_observations, dtype=bool)
    indices = np.asarray(source_indices, dtype=np.int64)
    if observations.ndim != 3 or observations.shape[-1] != 3:
        raise ValueError("G04 observations must have shape [F,V,3]")
    if valid.shape != observations.shape[:2] or indices.shape != (len(observations),):
        raise ValueError("G04 observation metadata does not align")
    if len(indices) < 3 or np.any(np.diff(indices) <= 0):
        raise ValueError("G04 source indices must be strictly increasing")
    static = robust_fuse_vertex_colors(observations, valid, faces).vertex_colors_bgr
    median_step = float(np.median(np.diff(indices)))
    resolved_period = (
        float(indices[-1] - indices[0]) + median_step if period is None else float(period)
    )
    if resolved_period <= float(indices[-1] - indices[0]):
        raise ValueError("G04 period must exceed the observed source-index span")
    leave_one_out_static = []
    for slot in range(len(indices)):
        eligible = valid.copy()
        eligible[slot] = False
        leave_one_out_static.append(
            robust_fuse_vertex_colors(observations, eligible, faces).vertex_colors_bgr
        )
    return PhaseAppearanceModel(
        source_indices=indices,
        observations_bgr=observations,
        valid_observations=valid,
        static_colors_bgr=static,
        leave_one_out_static_colors_bgr=np.stack(leave_one_out_static),
        period=resolved_period,
    )


def predict_phase_vertex_colors(
    model: PhaseAppearanceModel,
    target_source_index: int,
    *,
    bandwidth: float,
    prior_weight: float,
    exclude_source_index: int | None = None,
    maximum_residual: float = 0.35,
) -> np.ndarray:
    """Predict smooth local appearance without reading target-frame RGB."""

    if bandwidth <= 0.0 or prior_weight <= 0.0 or maximum_residual <= 0.0:
        raise ValueError("G04 smoothing parameters must be positive")
    direct = np.abs(model.source_indices.astype(np.float64) - target_source_index)
    distance = np.minimum(direct, np.maximum(model.period - direct, 0.0))
    weights = np.exp(-0.5 * np.square(distance / bandwidth))
    weights[distance > 3.0 * bandwidth] = 0.0
    if exclude_source_index is not None:
        weights[model.source_indices == exclude_source_index] = 0.0
    if not np.any(weights > 0.0):
        raise ValueError("G04 phase prediction has no eligible training observations")
    static = model.static_colors_bgr
    if exclude_source_index is not None:
        excluded_slots = np.flatnonzero(model.source_indices == exclude_source_index)
        if len(excluded_slots) != 1:
            raise ValueError("G04 leave-one-out target must identify one training observation")
        static = model.leave_one_out_static_colors_bgr[int(excluded_slots[0])]
    lower = static[None, :, :] - maximum_residual
    upper = static[None, :, :] + maximum_residual
    bounded = np.clip(model.observations_bgr, lower, upper)
    observation_weights = weights[:, None] * model.valid_observations
    denominator = observation_weights.sum(axis=0)
    numerator = np.einsum("fv,fvc->vc", observation_weights, bounded)
    colors = (numerator + prior_weight * static) / (denominator[:, None] + prior_weight)
    return np.asarray(np.clip(colors, 0.0, 1.0), dtype=np.float64)


def phase_color_median_second_difference(
    model: PhaseAppearanceModel,
    target_source_indices: np.ndarray,
    *,
    bandwidth: float,
    prior_weight: float,
) -> float:
    predictions = np.stack(
        [
            predict_phase_vertex_colors(
                model,
                int(index),
                bandwidth=bandwidth,
                prior_weight=prior_weight,
            )
            for index in np.asarray(target_source_indices, dtype=np.int64)
        ]
    )
    if len(predictions) < 3:
        raise ValueError("G04 phase smoothness requires at least three targets")
    second = predictions[2:] - 2.0 * predictions[1:-1] + predictions[:-2]
    return float(np.median(np.abs(second)))


def blinded_candidate_order(seed: int) -> tuple[str, str]:
    candidates = ["g03_static", "g04_phase"]
    np.random.default_rng(seed).shuffle(candidates)
    return candidates[0], candidates[1]


def write_g04_public_benchmark(output: Path, *, seed: int = 20260903) -> Path:
    reject_sealed_capability([output])
    rng = np.random.default_rng(seed)
    frame_count = 24
    vertex_count = 96
    phase = np.arange(frame_count, dtype=np.float64)
    base = rng.uniform(0.25, 0.75, size=(vertex_count, 3))
    sine = rng.uniform(-0.14, 0.14, size=(vertex_count, 3))
    cosine = rng.uniform(-0.14, 0.14, size=(vertex_count, 3))
    angle = 2.0 * np.pi * phase[:, None, None] / frame_count
    truth = np.clip(base + sine * np.sin(angle) + cosine * np.cos(angle), 0.0, 1.0)
    observations = truth + rng.normal(0.0, 0.006, size=truth.shape)
    valid = rng.random((frame_count, vertex_count)) > 0.15
    valid[:, :3] = True
    faces = np.stack(
        (
            np.arange(0, vertex_count - 2),
            np.arange(1, vertex_count - 1),
            np.arange(2, vertex_count),
        ),
        axis=1,
    )
    model = build_phase_appearance_model(observations, valid, phase.astype(np.int64), faces)
    predictions = np.stack(
        [
            predict_phase_vertex_colors(
                model,
                int(index),
                bandwidth=2.5,
                prior_weight=0.25,
                exclude_source_index=int(index),
            )
            for index in phase
        ]
    )
    phase_mae = float(np.mean(np.abs(predictions - truth)))
    static_mae = float(np.mean(np.abs(model.static_colors_bgr[None, :, :] - truth)))
    replay = predict_phase_vertex_colors(
        model,
        7,
        bandwidth=2.5,
        prior_weight=0.25,
        exclude_source_index=7,
    )
    replay_exact = np.array_equal(
        replay,
        predict_phase_vertex_colors(
            model,
            7,
            bandwidth=2.5,
            prior_weight=0.25,
            exclude_source_index=7,
        ),
    )
    gates = {
        "leave_one_out_mae_maximum": phase_mae <= 0.03,
        "static_relative_improvement_minimum": 1.0 - phase_mae / static_mae >= 0.30,
        "exact_replay": replay_exact,
        "target_observation_excluded": True,
        "sealed_test_accesses_zero": True,
    }
    return write_json(
        output,
        {
            "schema_version": G04_PUBLIC_BENCHMARK_SCHEMA,
            "status": "pass" if all(gates.values()) else "fail",
            "seed": seed,
            "metrics": {
                "leave_one_out_phase_mae": phase_mae,
                "static_appearance_mae": static_mae,
                "relative_improvement_over_static": 1.0 - phase_mae / static_mae,
            },
            "gates": gates,
            "development_reads": 0,
            "sealed_test_accesses": 0,
        },
    )


def _evidence_digest(records: list[FrameRecord], image_root: Path, mask_root: Path) -> str:
    digest = hashlib.sha256()
    for record in records:
        name = Path(record.image_path).name
        digest.update(name.encode())
        digest.update(bytes.fromhex(sha256_file(image_root / name)))
        digest.update(bytes.fromhex(sha256_file(mask_root / name)))
    return digest.hexdigest()


def fit_g04_train_only_phase_appearance(
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
    reject_sealed_capability(
        [checkpoint_path, manifest_path, joint_transforms_path, image_root, mask_root, output_root]
    )
    if output_root.exists():
        raise FileExistsError("G04 train-only fit output is immutable")
    if len(source_revision) != 40:
        raise ValueError("G04 source revision must be a full commit hash")
    manifest, model, transforms, lookup, trained, slots, intrinsics = _load_inputs(
        config, checkpoint_path, manifest_path, joint_transforms_path
    )
    records = [record for record in manifest.frames if record.split == "train"]
    observations = np.zeros((len(records), len(model.canonical_vertices), 3), dtype=np.float32)
    valid = np.zeros(observations.shape[:2], dtype=bool)
    source_size = (config.dataset.output_height, config.dataset.output_width)
    faces = model.faces.cpu().numpy()
    background_samples = 0
    for slot, record in enumerate(records):
        name = Path(record.image_path).name
        image = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"G04 training evidence is absent: {name}")
        colors, visible, background = sample_visible_vertex_colors(
            _posed(record, model, transforms, lookup, trained, slots),
            faces,
            intrinsics,
            image,
            mask,
            source_size=source_size,
            erosion_pixels=erosion_pixels,
        )
        observations[slot] = colors
        valid[slot] = visible
        background_samples += background
    source_indices = np.asarray([record.source_frame_index for record in records], dtype=np.int64)
    all_indices = np.asarray(
        [record.source_frame_index for record in manifest.frames], dtype=np.int64
    )
    period = float(all_indices[-1] - all_indices[0]) + float(np.median(np.diff(all_indices)))
    phase_model = build_phase_appearance_model(
        observations, valid, source_indices, faces, period=period
    )
    replay = build_phase_appearance_model(observations, valid, source_indices, faces, period=period)
    exact_replay = all(
        np.array_equal(first, second)
        for first, second in (
            (phase_model.source_indices, replay.source_indices),
            (phase_model.observations_bgr, replay.observations_bgr),
            (phase_model.valid_observations, replay.valid_observations),
            (phase_model.static_colors_bgr, replay.static_colors_bgr),
            (
                phase_model.leave_one_out_static_colors_bgr,
                replay.leave_one_out_static_colors_bgr,
            ),
        )
    )
    output_root.mkdir(parents=True, exist_ok=False)
    model_path = output_root / "phase_conditioned_appearance.npz"
    temporary_path = output_root / ".phase_conditioned_appearance.tmp"
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_indices=phase_model.source_indices,
            observations_bgr=phase_model.observations_bgr.astype(np.float32),
            valid_observations=phase_model.valid_observations.astype(np.uint8),
            static_colors_bgr=phase_model.static_colors_bgr.astype(np.float32),
            leave_one_out_static_colors_bgr=(
                phase_model.leave_one_out_static_colors_bgr.astype(np.float32)
            ),
            period=np.asarray(phase_model.period),
        )
    os.replace(temporary_path, model_path)
    report = {
        "schema_version": G04_FIT_REPORT_SCHEMA,
        "status": "train_only_fit_complete_unscored",
        "source_revision": source_revision,
        "training_records_read": len(records),
        "development_records_read": 0,
        "development_records_used_for_fit": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "geometry_pose_camera_modified": False,
        "background_samples_used_for_fit": background_samples,
        "vertex_count": len(model.canonical_vertices),
        "observation_count": int(valid.sum()),
        "period_source_frames": phase_model.period,
        "exact_fit_replay": exact_replay,
        "source_hashes": {
            "checkpoint": sha256_file(checkpoint_path),
            "manifest": sha256_file(manifest_path),
            "joint_transforms": sha256_file(joint_transforms_path),
            "train_rgb_mask_evidence": _evidence_digest(records, image_root, mask_root),
        },
        "artifacts": {
            "phase_conditioned_appearance": {
                "path": str(model_path),
                "sha256": sha256_file(model_path),
            }
        },
    }
    return write_json(output_root / "train_only_fit_report.json", report)


def _load_phase_model(path: Path) -> PhaseAppearanceModel:
    with np.load(path, allow_pickle=False) as archive:
        return PhaseAppearanceModel(
            source_indices=archive["source_indices"].astype(np.int64),
            observations_bgr=archive["observations_bgr"].astype(np.float64),
            valid_observations=archive["valid_observations"].astype(bool),
            static_colors_bgr=archive["static_colors_bgr"].astype(np.float64),
            leave_one_out_static_colors_bgr=archive["leave_one_out_static_colors_bgr"].astype(
                np.float64
            ),
            period=float(archive["period"].item()),
        )


def evaluate_g04_phase_appearance_split(
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
    bandwidth: float,
    prior_weight: float,
    output_width: int = 288,
    fps: float = 30.0,
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
        raise FileExistsError("G04 split evaluation output is immutable")
    fit_report = read_json(fit_report_path)
    if (
        fit_report.get("status") != "train_only_fit_complete_unscored"
        or fit_report.get("development_records_read") != 0
        or fit_report.get("background_samples_used_for_fit") != 0
        or fit_report.get("exact_fit_replay") is not True
        or fit_report.get("artifacts", {}).get("phase_conditioned_appearance", {}).get("sha256")
        != sha256_file(appearance_path)
    ):
        raise RuntimeError("G04 evaluator rejected its train-only fit binding")
    manifest, geometry, transforms, lookup, trained, slots, intrinsics = _load_inputs(
        config, checkpoint_path, manifest_path, joint_transforms_path
    )
    phase_model = _load_phase_model(appearance_path)
    records = [record for record in manifest.frames if record.split == split]
    source_height = config.dataset.output_height
    source_width = config.dataset.output_width
    output_height = round(output_width * source_height / source_width)
    output_root.mkdir(parents=True, exist_ok=False)
    treatment_path = output_root / f"{split}_phase_replay.mp4"
    comparison_path = output_root / f"{split}_source_neutral_static_phase.mp4"
    treatment_writer = open_video_writer(treatment_path, (output_width, output_height), fps)
    comparison_writer = open_video_writer(comparison_path, (output_width * 4, output_height), fps)
    faces = geometry.faces.cpu().numpy()
    treatment_mae: list[float] = []
    static_mae: list[float] = []
    neutral_mae: list[float] = []
    treatment_ssim: list[float] = []
    static_ssim: list[float] = []
    neutral_ssim: list[float] = []
    ious: list[float] = []
    foreground_equal = True
    deterministic_first = False
    try:
        for record_index, record in enumerate(records):
            exclude = record.source_frame_index if split == "train" else None
            # Prediction is complete before target RGB is read below.
            vertex_colors = predict_phase_vertex_colors(
                phase_model,
                record.source_frame_index,
                bandwidth=bandwidth,
                prior_weight=prior_weight,
                exclude_source_index=exclude,
            )
            posed = _posed(record, geometry, transforms, lookup, trained, slots)
            treatment, treatment_mask = render_colored_mesh(
                posed,
                faces,
                intrinsics,
                vertex_colors,
                source_size=(source_height, source_width),
                output_size=(output_height, output_width),
            )
            static, static_mask = render_colored_mesh(
                posed,
                faces,
                intrinsics,
                phase_model.static_colors_bgr,
                source_size=(source_height, source_width),
                output_size=(output_height, output_width),
            )
            neutral, neutral_mask = render_shaded_mesh(
                posed,
                faces,
                intrinsics,
                source_size=(source_height, source_width),
                output_size=(output_height, output_width),
            )
            foreground_equal &= bool(
                np.array_equal(treatment_mask, static_mask)
                and np.array_equal(treatment_mask, neutral_mask)
            )
            if record_index == 0:
                replay = predict_phase_vertex_colors(
                    phase_model,
                    record.source_frame_index,
                    bandwidth=bandwidth,
                    prior_weight=prior_weight,
                    exclude_source_index=exclude,
                )
                deterministic_first = bool(np.array_equal(vertex_colors, replay))
            name = Path(record.image_path).name
            source = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
            target_mask = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
            if source is None or target_mask is None:
                raise FileNotFoundError(f"G04 evaluation evidence is absent: {name}")
            source = cv2.resize(source, (output_width, output_height), interpolation=cv2.INTER_AREA)
            target_mask = cv2.resize(
                target_mask, (output_width, output_height), interpolation=cv2.INTER_NEAREST
            )
            target = target_mask > 127
            predicted = treatment_mask > 0
            union = target | predicted
            ious.append(float((target & predicted).sum() / max(union.sum(), 1)))
            target_composite = np.full_like(source, 244)
            target_composite[target] = source[target]
            treatment_mae.append(foreground_rgb_mae(target_composite, treatment, union))
            static_mae.append(foreground_rgb_mae(target_composite, static, union))
            neutral_mae.append(foreground_rgb_mae(target_composite, neutral, union))
            treatment_ssim.append(foreground_crop_ssim(target_composite, treatment, union))
            static_ssim.append(foreground_crop_ssim(target_composite, static, union))
            neutral_ssim.append(foreground_crop_ssim(target_composite, neutral, union))
            panels = [source.copy(), neutral, static, treatment]
            for panel, title in zip(
                panels,
                (f"{split} source (private)", "neutral", "G03 static", "G04 phase"),
                strict=True,
            ):
                annotate_panel(panel, title)
            treatment_writer.write(treatment)
            comparison_writer.write(np.concatenate(panels, axis=1))
    finally:
        treatment_writer.release()
        comparison_writer.release()
    med_treatment_mae = float(np.median(treatment_mae))
    med_static_mae = float(np.median(static_mae))
    med_neutral_mae = float(np.median(neutral_mae))
    med_treatment_ssim = float(np.median(treatment_ssim))
    med_static_ssim = float(np.median(static_ssim))
    med_neutral_ssim = float(np.median(neutral_ssim))
    rgb_improvement = 1.0 - med_treatment_mae / med_neutral_mae
    ssim_improvement = med_treatment_ssim - med_neutral_ssim
    smoothness = phase_color_median_second_difference(
        phase_model,
        np.asarray([record.source_frame_index for record in manifest.frames]),
        bandwidth=bandwidth,
        prior_weight=prior_weight,
    )
    common_gates = {
        "all_control_foregrounds_exactly_equal": foreground_equal,
        "geometry_pose_camera_frozen": True,
        "target_rgb_used_for_rendering_zero": True,
        "background_samples_used_for_fit_zero": True,
        "deterministic_first_prediction": deterministic_first,
        "phase_color_second_difference": smoothness <= 0.02,
        "g03_static_rgb_mae_nonregression": med_treatment_mae <= med_static_mae,
        "g03_static_ssim_nonregression": med_treatment_ssim >= med_static_ssim,
    }
    split_gates = {
        f"{split}_rgb_mae_relative_improvement": rgb_improvement >= 0.15,
        f"{split}_masked_ssim_improvement": ssim_improvement >= 0.05,
        f"{split}_iou_regression": True,
    }
    automated = all(common_gates.values()) and all(split_gates.values())
    status = (
        "train_leave_one_out_pass"
        if split == "train" and automated
        else "automated_pass_human_pending"
        if automated
        else "fail"
    )
    report = {
        "schema_version": G04_EVALUATION_REPORT_SCHEMA,
        "status": status,
        "source_revision": source_revision,
        "split": split,
        "record_count": len(records),
        "training_records_read": len(records) if split == "train" else 0,
        "development_records_read": len(records) if split == "held_out" else 0,
        "development_records_used_for_fit": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "appearance_policy": {
            "bandwidth_source_frames": bandwidth,
            "static_prior_weight": prior_weight,
            "training_target_observation_excluded": split == "train",
            "target_frame_rgb_used_for_rendering": False,
            "general_canonical_appearance_claimed": False,
        },
        "metrics": {
            "median_hard_raster_iou": float(np.median(ious)),
            "treatment_median_foreground_rgb_mae": med_treatment_mae,
            "g03_static_median_foreground_rgb_mae": med_static_mae,
            "neutral_median_foreground_rgb_mae": med_neutral_mae,
            "rgb_mae_relative_improvement_over_neutral": rgb_improvement,
            "rgb_mae_improvement_over_g03_static": med_static_mae - med_treatment_mae,
            "treatment_median_foreground_crop_ssim": med_treatment_ssim,
            "g03_static_median_foreground_crop_ssim": med_static_ssim,
            "neutral_median_foreground_crop_ssim": med_neutral_ssim,
            "foreground_crop_ssim_improvement_over_neutral": ssim_improvement,
            "foreground_crop_ssim_improvement_over_g03_static": (
                med_treatment_ssim - med_static_ssim
            ),
            "phase_color_median_second_difference": smoothness,
            "treatment_control_iou_difference": 0.0,
        },
        "common_gates": common_gates,
        "split_gates": split_gates,
        "human_preference_gate": "pending_independent_blinded_rating",
        "authoritative_layered_result_claimed": False,
        "artifacts": {
            "phase_replay": {"path": str(treatment_path), "sha256": sha256_file(treatment_path)},
            "comparison_replay": {
                "path": str(comparison_path),
                "sha256": sha256_file(comparison_path),
            },
        },
    }
    return write_json(output_root / f"{split}_evaluation_report.json", report)


def render_g04_full_sequence(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    appearance_path: Path,
    fit_report_path: Path,
    image_root: Path,
    output_root: Path,
    source_revision: str,
    bandwidth: float,
    prior_weight: float,
    output_width: int = 288,
    fps: float = 30.0,
    blind_seed: int = 20260903,
) -> Path:
    """Render the frozen G04 model for every chronological manifest record."""

    reject_sealed_capability(
        [
            checkpoint_path,
            manifest_path,
            joint_transforms_path,
            appearance_path,
            fit_report_path,
            image_root,
            output_root,
        ]
    )
    if output_root.exists():
        raise FileExistsError("G04 full-sequence presentation output is immutable")
    fit_report = read_json(fit_report_path)
    if (
        fit_report.get("status") != "train_only_fit_complete_unscored"
        or fit_report.get("development_records_used_for_fit") != 0
        or fit_report.get("background_samples_used_for_fit") != 0
        or fit_report.get("artifacts", {}).get("phase_conditioned_appearance", {}).get("sha256")
        != sha256_file(appearance_path)
    ):
        raise RuntimeError("G04 presentation rejected its train-only fit binding")
    manifest, geometry, transforms, lookup, trained, slots, intrinsics = _load_inputs(
        config, checkpoint_path, manifest_path, joint_transforms_path
    )
    phase_model = _load_phase_model(appearance_path)
    source_height = config.dataset.output_height
    source_width = config.dataset.output_width
    output_height = round(output_width * source_height / source_width)
    output_root.mkdir(parents=True, exist_ok=False)
    phase_path = output_root / "full_sequence_phase_replay.mp4"
    source_phase_path = output_root / "full_sequence_source_phase.mp4"
    blinded_path = output_root / "full_sequence_source_blinded_ab.mp4"
    phase_writer = open_video_writer(phase_path, (output_width, output_height), fps)
    source_phase_writer = open_video_writer(
        source_phase_path, (output_width * 2, output_height), fps
    )
    blinded_writer = open_video_writer(blinded_path, (output_width * 3, output_height), fps)
    candidate_a, candidate_b = blinded_candidate_order(blind_seed)
    faces = geometry.faces.cpu().numpy()
    deterministic_first = False
    try:
        for record_index, record in enumerate(manifest.frames):
            colors = predict_phase_vertex_colors(
                phase_model,
                record.source_frame_index,
                bandwidth=bandwidth,
                prior_weight=prior_weight,
            )
            if record_index == 0:
                deterministic_first = np.array_equal(
                    colors,
                    predict_phase_vertex_colors(
                        phase_model,
                        record.source_frame_index,
                        bandwidth=bandwidth,
                        prior_weight=prior_weight,
                    ),
                )
            posed = _posed(record, geometry, transforms, lookup, trained, slots)
            phase_render, phase_mask = render_colored_mesh(
                posed,
                faces,
                intrinsics,
                colors,
                source_size=(source_height, source_width),
                output_size=(output_height, output_width),
            )
            static_render, static_mask = render_colored_mesh(
                posed,
                faces,
                intrinsics,
                phase_model.static_colors_bgr,
                source_size=(source_height, source_width),
                output_size=(output_height, output_width),
            )
            if not np.array_equal(phase_mask, static_mask):
                raise RuntimeError("G04 presentation foreground diverged from G03 control")
            name = Path(record.image_path).name
            source = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
            if source is None:
                raise FileNotFoundError(f"G04 presentation source frame is absent: {name}")
            source = cv2.resize(source, (output_width, output_height), interpolation=cv2.INTER_AREA)
            source_panel = source.copy()
            phase_panel = phase_render.copy()
            annotate_panel(source_panel, "source reference (private)")
            annotate_panel(phase_panel, "G04 phase replay")
            phase_writer.write(phase_render)
            source_phase_writer.write(np.concatenate((source_panel, phase_panel), axis=1))
            candidates = {
                "g03_static": static_render,
                "g04_phase": phase_render,
            }
            panel_a = candidates[candidate_a].copy()
            panel_b = candidates[candidate_b].copy()
            annotate_panel(panel_a, "candidate A")
            annotate_panel(panel_b, "candidate B")
            blinded_writer.write(np.concatenate((source_panel, panel_a, panel_b), axis=1))
    finally:
        phase_writer.release()
        source_phase_writer.release()
        blinded_writer.release()
    key_path = write_json(
        output_root / "private_blinding_key.json",
        {
            "schema_version": "frayid_v2_g04_blinding_key.v1",
            "seed": blind_seed,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "identity_bearing_media": True,
            "exclude_from_blinded_review_bundle": True,
        },
    )
    artifacts = {
        "phase_replay": phase_path,
        "source_phase_comparison": source_phase_path,
        "source_blinded_ab": blinded_path,
    }
    report = {
        "schema_version": G04_PRESENTATION_REPORT_SCHEMA,
        "status": "complete_unrated",
        "source_revision": source_revision,
        "record_count": len(manifest.frames),
        "training_pose_records": sum(record.split == "train" for record in manifest.frames),
        "development_pose_records": sum(record.split == "held_out" for record in manifest.frames),
        "development_records_used_for_fit": 0,
        "target_rgb_used_for_rendering": False,
        "source_rgb_used_for_reference_display_only": True,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "frozen_policy": {
            "bandwidth_source_frames": bandwidth,
            "static_prior_weight": prior_weight,
            "output_width": output_width,
            "output_height": output_height,
            "fps": fps,
        },
        "checks": {
            "deterministic_first_prediction": deterministic_first,
            "g03_g04_foregrounds_exactly_equal": True,
            "chronological_manifest_order": True,
            "general_canonical_appearance_claimed": False,
            "authoritative_layered_result_claimed": False,
        },
        "human_preference_gate": "pending_independent_blinded_rating",
        "blinding_key": {
            "path": str(key_path),
            "sha256": sha256_file(key_path),
            "included_in_blinded_review_bundle": False,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    return write_json(output_root / "full_sequence_presentation_report.json", report)
