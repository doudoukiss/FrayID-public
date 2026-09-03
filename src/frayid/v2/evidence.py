from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from torch import Tensor

from frayid.camera import axis_angle_to_matrix, project_points
from frayid.dataset import read_dataset_manifest
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import DynamicCameraSolution, EvidenceVolumeMetadata

SAPIENS2_DOME29_LAYER_IDS: dict[str, list[int]] = {
    "lower_clothing": [13],
    "upper_clothing": [23],
    "hair": [4],
    "footwear": [9, 10, 18, 19],
    "body_parts": [3, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 27, 28],
    "other": [1, 2],
}


def bind_t04_hull_inputs(
    manifest_path: Path,
    mask_root: Path,
    camera_solution_path: Path,
    output_path: Path,
    *,
    maximum_dimension: int = 256,
    semantic_root: Path | None = None,
    semantic_qualification_path: Path | None = None,
) -> Path:
    """Bind accepted train masks to T04 cameras without changing camera geometry."""

    optional_paths = [path for path in (semantic_root, semantic_qualification_path) if path]
    reject_sealed_capability(
        [manifest_path, mask_root, camera_solution_path, output_path, *optional_paths]
    )
    if output_path.exists():
        raise FileExistsError("T04 hull-input bindings are immutable")
    if maximum_dimension < 32:
        raise ValueError("hull-input maximum dimension must be at least 32")
    if (semantic_root is None) != (semantic_qualification_path is None):
        raise ValueError("semantic root and qualification must be supplied together")
    manifest = read_dataset_manifest(manifest_path)
    solution = DynamicCameraSolution.model_validate(read_json(camera_solution_path))
    if solution.status != "pass":
        raise ValueError("hull binding requires a passing T04 camera solution")
    training_by_source = {
        frame.source_frame_index: frame
        for frame in manifest.frames
        if frame.split == "train" and frame.quality_accepted
    }
    solution_sources = [frame.source_frame_index for frame in solution.frames]
    if solution_sources != sorted(training_by_source):
        raise ValueError("T04 camera solution must exactly cover accepted train sources")
    semantic_hashes: dict[int, str] = {}
    if semantic_qualification_path is not None:
        semantic_qualification = read_json(semantic_qualification_path)
        if semantic_qualification.get("status") != "pass":
            raise ValueError("hull binding requires passing S01 semantic qualification")
        semantic_hashes = {
            int(frame["source_frame_index"]): str(frame["semantic_sha256"])
            for frame in semantic_qualification.get("frame_records", [])
        }
        if sorted(semantic_hashes) != solution_sources:
            raise ValueError("S01 semantic qualification must exactly cover T04 sources")

    masks: list[np.ndarray] = []
    semantic_support: dict[str, list[np.ndarray]] = {name: [] for name in SAPIENS2_DOME29_LAYER_IDS}
    mask_hashes: dict[str, str] = {}
    source_shape: tuple[int, int] | None = None
    target_shape: tuple[int, int] | None = None
    for frame in solution.frames:
        record = training_by_source[frame.source_frame_index]
        mask_path = mask_root / Path(record.image_path).name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"T04 train mask is absent: {mask_path}")
        shape = (int(mask.shape[0]), int(mask.shape[1]))
        if source_shape is None:
            source_shape = shape
            scale = min(1.0, maximum_dimension / max(shape))
            target_shape = (
                max(1, round(shape[0] * scale)),
                max(1, round(shape[1] * scale)),
            )
        if shape != source_shape or target_shape is None:
            raise ValueError("T04 train masks must share one image shape")
        resized = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        masks.append(resized.astype(np.float32) / 255.0)
        mask_hashes[str(frame.source_frame_index)] = sha256_file(mask_path)
        if semantic_root is not None:
            semantic_path = semantic_root / f"{Path(record.image_path).stem}.npz"
            if sha256_file(semantic_path) != semantic_hashes[frame.source_frame_index]:
                raise ValueError(f"S01 semantic hash mismatch for frame {frame.source_frame_index}")
            with np.load(semantic_path, allow_pickle=False) as archive:
                labels = archive["labels"]
                semantic_confidence = archive["confidence"].astype(np.float32)
                semantic_source = int(archive["source_frame_index"])
            if (
                labels.shape != source_shape
                or semantic_confidence.shape != source_shape
                or semantic_source != frame.source_frame_index
            ):
                raise ValueError("S01 semantic evidence does not align with the train mask")
            for name, class_ids in SAPIENS2_DOME29_LAYER_IDS.items():
                support = np.isin(labels, class_ids).astype(np.float32) * semantic_confidence
                semantic_support[name].append(
                    np.clip(
                        cv2.resize(
                            support,
                            (target_shape[1], target_shape[0]),
                            interpolation=cv2.INTER_AREA,
                        ),
                        0.0,
                        1.0,
                    )
                )
    if source_shape is None or target_shape is None:
        raise ValueError("T04 hull input binding found no train masks")

    orientations = torch.tensor(
        [frame.global_orient for frame in solution.frames], dtype=torch.float64
    )
    rotations = axis_angle_to_matrix(orientations).cpu().numpy()
    translations = np.asarray([frame.translation for frame in solution.frames], dtype=np.float64)
    confidences = np.asarray([frame.confidence for frame in solution.frames], dtype=np.float64)
    intrinsics = np.asarray(solution.shared_intrinsics, dtype=np.float64)
    scale_x = target_shape[1] / source_shape[1]
    scale_y = target_shape[0] / source_shape[0]
    resized_intrinsics = intrinsics.copy()
    resized_intrinsics[0] *= scale_x
    resized_intrinsics[1] *= scale_y
    mask_hash_manifest = json.dumps(mask_hashes, sort_keys=True)
    source_hashes = {
        "camera_solution": sha256_file(camera_solution_path),
        "manifest": sha256_file(manifest_path),
        "train_mask_hash_manifest": hashlib.sha256(mask_hash_manifest.encode("utf-8")).hexdigest(),
    }
    if semantic_qualification_path is not None:
        source_hashes["semantic_qualification"] = sha256_file(semantic_qualification_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "silhouettes": np.stack(masks).astype(np.float32),
        "intrinsics": resized_intrinsics.astype(np.float32),
        "rotations": rotations.astype(np.float32),
        "translations": translations.astype(np.float32),
        "motion_uncertainty": (1.0 - confidences).astype(np.float32),
        "source_frame_indices": np.asarray(solution_sources, dtype=np.int64),
        "original_intrinsics": intrinsics,
        "original_image_shape": np.asarray(source_shape, dtype=np.int64),
        "bound_image_shape": np.asarray(target_shape, dtype=np.int64),
        "camera_parameter_policy": np.asarray(
            "exact_t04_values_with_deterministic_image_coordinate_rescaling"
        ),
        "source_hashes": np.asarray(json.dumps(source_hashes, sort_keys=True)),
        "train_mask_hashes": np.asarray(mask_hash_manifest),
    }
    if semantic_root is not None:
        payload.update(
            {
                f"semantic__{name}": np.stack(values).astype(np.float32)
                for name, values in semantic_support.items()
            }
        )
    np.savez_compressed(output_path, **payload)  # type: ignore[arg-type]
    return output_path


@dataclass(frozen=True)
class EvidenceVolume:
    signed_distance: Tensor
    support_count: Tensor
    angular_coverage: Tensor
    mask_uncertainty: Tensor
    motion_uncertainty: Tensor
    prior_contribution: Tensor
    unsupported: Tensor
    semantic_support: dict[str, Tensor]
    metadata: EvidenceVolumeMetadata

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signed_distance": self.signed_distance.detach().cpu().numpy(),
            "support_count": self.support_count.detach().cpu().numpy(),
            "angular_coverage": self.angular_coverage.detach().cpu().numpy(),
            "mask_uncertainty": self.mask_uncertainty.detach().cpu().numpy(),
            "motion_uncertainty": self.motion_uncertainty.detach().cpu().numpy(),
            "prior_contribution": self.prior_contribution.detach().cpu().numpy(),
            "unsupported": self.unsupported.detach().cpu().numpy(),
            "metadata": np.asarray(self.metadata.model_dump_json()),
        }
        payload.update(
            {
                f"semantic__{name}": values.detach().cpu().numpy()
                for name, values in sorted(self.semantic_support.items())
            }
        )
        np.savez_compressed(path, **payload)  # type: ignore[arg-type]
        return path

    @classmethod
    def load(cls, path: Path, *, device: torch.device | str = "cpu") -> EvidenceVolume:
        with np.load(path, allow_pickle=False) as archive:
            metadata = EvidenceVolumeMetadata.model_validate_json(str(archive["metadata"]))

            def tensor(name: str) -> Tensor:
                return torch.as_tensor(archive[name], device=device)

            return cls(
                signed_distance=tensor("signed_distance"),
                support_count=tensor("support_count"),
                angular_coverage=tensor("angular_coverage"),
                mask_uncertainty=tensor("mask_uncertainty"),
                motion_uncertainty=tensor("motion_uncertainty"),
                prior_contribution=tensor("prior_contribution"),
                unsupported=tensor("unsupported").bool(),
                semantic_support={
                    name.removeprefix("semantic__"): tensor(name)
                    for name in archive.files
                    if name.startswith("semantic__")
                },
                metadata=metadata,
            )


def evidence_volume_agreement(first: EvidenceVolume, second: EvidenceVolume) -> dict[str, float]:
    if first.metadata.resolution != second.metadata.resolution:
        raise ValueError("evidence agreement requires equal grid resolution")
    if not math.isclose(first.metadata.extent, second.metadata.extent):
        raise ValueError("evidence agreement requires equal grid extent")

    def surface_points(volume: EvidenceVolume) -> tuple[np.ndarray, np.ndarray]:
        occupied = volume.signed_distance.detach().cpu().numpy() <= 0
        surface = occupied & ~binary_erosion(occupied)
        supported_surface = surface & ~volume.unsupported.detach().cpu().numpy()
        all_indices = np.argwhere(surface)
        if not len(all_indices):
            raise ValueError("evidence agreement requires nonempty supported surfaces")
        supported_indices = np.argwhere(supported_surface)
        if not len(supported_indices):
            supported_indices = all_indices
        return all_indices.astype(np.float64), supported_indices.astype(np.float64)

    first_surface, first_supported = surface_points(first)
    second_surface, second_supported = surface_points(second)
    first_to_second = cKDTree(second_surface).query(first_supported, workers=1)[0]
    second_to_first = cKDTree(first_surface).query(second_supported, workers=1)[0]
    return {
        "maximum_supported_surface_displacement_voxels": float(
            max(first_to_second.max(), second_to_first.max())
        ),
        "mean_supported_surface_displacement_voxels": float(
            0.5 * (first_to_second.mean() + second_to_first.mean())
        ),
        "unsupported_label_disagreement_fraction": float(
            torch.logical_xor(first.unsupported, second.unsupported).float().mean()
        ),
    }


def preserve_sapiens2_semantics(
    labels_path: Path,
    output_path: Path,
    *,
    confidence_path: Path | None = None,
) -> Path:
    labels = np.load(labels_path, allow_pickle=False)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Sapiens2 semantic labels must be one integer image")
    if labels.size == 0 or int(labels.min()) < 0 or int(labels.max()) > 28:
        raise ValueError("Sapiens2 labels must use the registered 29-class palette")
    confidence: np.ndarray
    if confidence_path is None:
        confidence = np.ones(labels.shape, dtype=np.float32)
    else:
        confidence = np.load(confidence_path, allow_pickle=False).astype(np.float32)
        if confidence.shape != labels.shape or not np.isfinite(confidence).all():
            raise ValueError("semantic confidence must be finite and align with labels")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("semantic confidence must lie in [0,1]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "labels": labels.astype(np.uint8),
        "confidence": confidence,
        "palette": np.asarray("sapiens2_dome29"),
    }
    payload.update(
        {
            f"support__{name}": np.isin(labels, class_ids).astype(np.float32) * confidence
            for name, class_ids in SAPIENS2_DOME29_LAYER_IDS.items()
        }
    )
    np.savez_compressed(output_path, **payload)  # type: ignore[arg-type]
    return output_path


def _weighted_quantile(values: Tensor, weights: Tensor, quantile: float) -> Tensor:
    order = torch.argsort(values, dim=0)
    ordered_values = torch.gather(values, 0, order)
    ordered_weights = torch.gather(weights, 0, order)
    cumulative = torch.cumsum(ordered_weights, dim=0)
    target = quantile * ordered_weights.sum(dim=0).clamp_min(1.0e-8)
    index = (cumulative < target.unsqueeze(0)).sum(dim=0).clamp_max(values.shape[0] - 1)
    return torch.gather(ordered_values, 0, index.unsqueeze(0)).squeeze(0)


def _aggregate_violations(
    values: Tensor,
    weights: Tensor,
    *,
    aggregation: str,
    quantile: float,
) -> Tensor:
    if aggregation == "weighted_quantile":
        return _weighted_quantile(values, weights, quantile)
    order = torch.argsort(values, dim=0)
    ordered = torch.gather(values, 0, order)
    ordered_weights = torch.gather(weights, 0, order)
    views = values.shape[0]
    trim = 1 if aggregation == "trimmed_weighted_quantile" else max(1, views // 10)
    if views > 2 * trim:
        selected = ordered[trim:-trim]
        selected_weights = ordered_weights[trim:-trim]
    else:
        selected = ordered
        selected_weights = ordered_weights
    if aggregation == "trimmed_weighted_quantile":
        return _weighted_quantile(selected, selected_weights, quantile)
    if aggregation == "trimmed_mean":
        return (selected * selected_weights).sum(dim=0) / selected_weights.sum(dim=0).clamp_min(
            1.0e-8
        )
    tail_start = max(0, int(selected.shape[0] * quantile))
    tail = selected[tail_start:]
    tail_weights = selected_weights[tail_start:]
    return (tail * tail_weights).sum(dim=0) / tail_weights.sum(dim=0).clamp_min(1.0e-8)


def build_confidence_aware_visual_hull(
    silhouettes: Tensor,
    intrinsics: Tensor,
    rotations: Tensor,
    translations: Tensor,
    *,
    resolution: int,
    extent: float,
    mask_confidence: Tensor | None = None,
    motion_uncertainty_per_view: Tensor | None = None,
    prior_contribution: Tensor | None = None,
    aggregation: str = "weighted_quantile",
    quantile: float = 0.8,
    minimum_view_support: int = 3,
    occupancy_violation_threshold: float = 0.5,
    source_hashes: dict[str, str] | None = None,
    semantic_masks: dict[str, Tensor] | None = None,
) -> EvidenceVolume:
    if silhouettes.ndim != 3:
        raise ValueError("silhouettes must have shape [V,H,W]")
    views, height, width = silhouettes.shape
    if rotations.shape != (views, 3, 3) or translations.shape != (views, 3):
        raise ValueError("visual-hull camera arrays do not align")
    if intrinsics.shape not in ((3, 3), (views, 3, 3)):
        raise ValueError("intrinsics must be shared or per-view")
    if resolution < 3 or extent <= 0 or minimum_view_support < 1:
        raise ValueError("visual-hull grid contract is invalid")
    if aggregation not in {
        "weighted_quantile",
        "trimmed_weighted_quantile",
        "trimmed_mean",
        "cvar",
    }:
        raise ValueError("unsupported robust visual-hull aggregation")
    confidence = mask_confidence if mask_confidence is not None else torch.ones_like(silhouettes)
    if confidence.shape != silhouettes.shape:
        raise ValueError("mask confidence must align with silhouettes")
    semantics = semantic_masks or {}
    for name, values in semantics.items():
        if name not in SAPIENS2_DOME29_LAYER_IDS:
            raise ValueError(f"unregistered semantic layer: {name}")
        if values.shape != silhouettes.shape or not torch.isfinite(values).all():
            raise ValueError("semantic masks must be finite and align with silhouettes")
        if torch.any((values < 0) | (values > 1)):
            raise ValueError("semantic support must lie in [0,1]")
    motion = (
        motion_uncertainty_per_view
        if motion_uncertainty_per_view is not None
        else torch.zeros(views, dtype=silhouettes.dtype, device=silhouettes.device)
    )
    if motion.shape != (views,) or torch.any((motion < 0) | (motion > 1)):
        raise ValueError("motion uncertainty must contain one [0,1] value per view")

    coordinates = torch.linspace(
        -extent,
        extent,
        resolution,
        dtype=silhouettes.dtype,
        device=silhouettes.device,
    )
    xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    violations: list[Tensor] = []
    weights: list[Tensor] = []
    direction_samples: list[Tensor] = []
    mask_uncertainties: list[Tensor] = []
    sampled_semantics: dict[str, list[Tensor]] = {name: [] for name in semantics}
    for view in range(views):
        camera = points @ rotations[view].T + translations[view]
        matrix = intrinsics if intrinsics.ndim == 2 else intrinsics[view]
        pixels = project_points(camera, matrix)
        normalized_x = 2.0 * (pixels[:, 0] + 0.5) / width - 1.0
        normalized_y = 2.0 * (pixels[:, 1] + 0.5) / height - 1.0
        grid = torch.stack((normalized_x, normalized_y), dim=-1).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            silhouettes[view][None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(-1)
        sampled_confidence = F.grid_sample(
            confidence[view][None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(-1)
        visible = (
            (camera[:, 2] > 0)
            & (normalized_x >= -1)
            & (normalized_x <= 1)
            & (normalized_y >= -1)
            & (normalized_y <= 1)
        )
        weight = sampled_confidence * (1.0 - motion[view]) * visible.to(sampled.dtype)
        violations.append(1.0 - sampled)
        weights.append(weight)
        mask_uncertainties.append(4.0 * sampled * (1.0 - sampled))
        for name, values in semantics.items():
            semantic_sample = F.grid_sample(
                values[view][None, None],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).reshape(-1)
            sampled_semantics[name].append(semantic_sample)
        camera_center = (-translations[view]) @ rotations[view]
        direction_samples.append(F.normalize(points - camera_center, dim=-1))
    violation_stack = torch.stack(violations)
    weight_stack = torch.stack(weights)
    aggregate = _aggregate_violations(
        violation_stack,
        weight_stack,
        aggregation=aggregation,
        quantile=quantile,
    )
    support = (weight_stack > 0.05).sum(dim=0)
    low_support = support < minimum_view_support
    base_occupancy = aggregate <= occupancy_violation_threshold
    positive_weight = weight_stack > 0
    inside = violation_stack <= occupancy_violation_threshold
    negative_candidate_weight = torch.where(
        inside & positive_weight,
        weight_stack,
        torch.full_like(weight_stack, -1.0),
    )
    negative_slot = negative_candidate_weight.argmax(dim=0, keepdim=True)
    has_negative_candidate = negative_candidate_weight.max(dim=0).values >= 0
    false_negative_values = violation_stack.clone()
    false_negative_values.scatter_(0, negative_slot, 1.0)
    false_negative_aggregate = _aggregate_violations(
        false_negative_values,
        weight_stack,
        aggregation=aggregation,
        quantile=quantile,
    )
    positive_candidate_weight = torch.where(
        ~inside & positive_weight,
        weight_stack,
        torch.full_like(weight_stack, -1.0),
    )
    positive_slot = positive_candidate_weight.argmax(dim=0, keepdim=True)
    has_positive_candidate = positive_candidate_weight.max(dim=0).values >= 0
    false_positive_values = violation_stack.clone()
    false_positive_values.scatter_(0, positive_slot, 0.0)
    false_positive_aggregate = _aggregate_violations(
        false_positive_values,
        weight_stack,
        aggregation=aggregation,
        quantile=quantile,
    )
    corruption_sensitive = (
        has_negative_candidate
        & ((false_negative_aggregate <= occupancy_violation_threshold) != base_occupancy)
    ) | (
        has_positive_candidate
        & ((false_positive_aggregate <= occupancy_violation_threshold) != base_occupancy)
    )
    unsupported = low_support | corruption_sensitive
    occupancy = base_occupancy & ~low_support
    if not torch.any(occupancy):
        raise ValueError("robust visual hull is empty")
    occupancy_grid = occupancy.reshape(resolution, resolution, resolution)
    boundary = torch.zeros_like(occupancy_grid)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    if torch.any(occupancy_grid & boundary):
        raise ValueError("robust visual hull touches its fixed outer boundary")
    occupancy_numpy = occupancy_grid.detach().cpu().numpy()
    pitch = 2.0 * extent / (resolution - 1)
    signed = (
        distance_transform_edt(~occupancy_numpy) - distance_transform_edt(occupancy_numpy)
    ) * pitch
    weighted_directions = (torch.stack(direction_samples) * weight_stack[..., None]).sum(
        dim=0
    ) / weight_stack.sum(dim=0).clamp_min(1.0e-8)[..., None]
    angular = 1.0 - torch.linalg.vector_norm(weighted_directions, dim=-1).clamp(0, 1)
    mask_uncertainty = (torch.stack(mask_uncertainties) * weight_stack).sum(
        dim=0
    ) / weight_stack.sum(dim=0).clamp_min(1.0e-8)
    motion_grid = (motion[:, None] * weight_stack).sum(dim=0) / weight_stack.sum(dim=0).clamp_min(
        1.0e-8
    )
    prior = (
        torch.zeros_like(occupancy_grid, dtype=silhouettes.dtype)
        if prior_contribution is None
        else prior_contribution
    )
    if prior.shape != occupancy_grid.shape:
        raise ValueError("prior contribution must match the evidence grid")
    shape = (resolution, resolution, resolution)
    metadata = EvidenceVolumeMetadata(
        resolution=resolution,
        extent=extent,
        aggregation=aggregation,  # type: ignore[arg-type]
        training_view_count=views,
        minimum_view_support=minimum_view_support,
        semantic_layer_ids=SAPIENS2_DOME29_LAYER_IDS,
        source_hashes=source_hashes or {},
    )
    semantic_support = {
        name: (torch.stack(samples) * weight_stack)
        .sum(dim=0)
        .div(weight_stack.sum(dim=0).clamp_min(1.0e-8))
        .reshape(shape)
        for name, samples in sampled_semantics.items()
    }
    return EvidenceVolume(
        signed_distance=torch.as_tensor(signed, dtype=silhouettes.dtype, device=silhouettes.device),
        support_count=support.reshape(shape),
        angular_coverage=angular.reshape(shape),
        mask_uncertainty=mask_uncertainty.reshape(shape),
        motion_uncertainty=motion_grid.reshape(shape),
        prior_contribution=prior,
        unsupported=unsupported.reshape(shape),
        semantic_support=semantic_support,
        metadata=metadata,
    )


def qualify_visual_hull_robustness(
    binding_path: Path,
    reference_volume_path: Path,
    public_benchmark_path: Path,
    report_output_path: Path,
) -> Path:
    """Apply the registered replay, sparse-view, and corrupted-mask hull gates."""

    reject_sealed_capability(
        [binding_path, reference_volume_path, public_benchmark_path, report_output_path]
    )
    if report_output_path.exists():
        raise FileExistsError("visual-hull qualification reports are immutable")
    reference = EvidenceVolume.load(reference_volume_path)
    with np.load(binding_path, allow_pickle=False) as archive:
        silhouettes = torch.as_tensor(archive["silhouettes"], dtype=torch.float32)
        intrinsics = torch.as_tensor(archive["intrinsics"], dtype=torch.float32)
        rotations = torch.as_tensor(archive["rotations"], dtype=torch.float32)
        translations = torch.as_tensor(archive["translations"], dtype=torch.float32)
        mask_confidence = (
            torch.as_tensor(archive["mask_confidence"], dtype=torch.float32)
            if "mask_confidence" in archive
            else None
        )
        motion = torch.as_tensor(archive["motion_uncertainty"], dtype=torch.float32)
        semantic_masks = {
            name.removeprefix("semantic__"): torch.as_tensor(archive[name], dtype=torch.float32)
            for name in archive.files
            if name.startswith("semantic__")
        }
        source_hashes = (
            json.loads(str(archive["source_hashes"]))
            if "source_hashes" in archive
            else {"input_binding": sha256_file(binding_path)}
        )
    build_arguments = {
        "resolution": reference.metadata.resolution,
        "extent": reference.metadata.extent,
        "aggregation": reference.metadata.aggregation,
        "minimum_view_support": reference.metadata.minimum_view_support,
    }
    replay = build_confidence_aware_visual_hull(
        silhouettes,
        intrinsics,
        rotations,
        translations,
        mask_confidence=mask_confidence,
        motion_uncertainty_per_view=motion,
        semantic_masks=semantic_masks,
        source_hashes=source_hashes,
        **build_arguments,  # type: ignore[arg-type]
    )
    replay_exact = all(
        torch.equal(getattr(reference, name), getattr(replay, name))
        for name in (
            "signed_distance",
            "support_count",
            "angular_coverage",
            "mask_uncertainty",
            "motion_uncertainty",
            "prior_contribution",
            "unsupported",
        )
    ) and all(
        torch.equal(reference.semantic_support[name], replay.semantic_support[name])
        for name in reference.semantic_support
    )
    sparse_slots = torch.arange(0, silhouettes.shape[0], 2)
    sparse = build_confidence_aware_visual_hull(
        silhouettes[sparse_slots],
        intrinsics if intrinsics.ndim == 2 else intrinsics[sparse_slots],
        rotations[sparse_slots],
        translations[sparse_slots],
        mask_confidence=(mask_confidence[sparse_slots] if mask_confidence is not None else None),
        motion_uncertainty_per_view=motion[sparse_slots],
        semantic_masks={name: values[sparse_slots] for name, values in semantic_masks.items()},
        source_hashes=source_hashes,
        **build_arguments,  # type: ignore[arg-type]
    )
    dense_sparse_agreement = evidence_volume_agreement(reference, sparse)
    corrupted_slot = int(torch.argmin(motion))
    corrupted_silhouettes = silhouettes.clone()
    corrupted_silhouettes[corrupted_slot] = 0
    corrupted = build_confidence_aware_visual_hull(
        corrupted_silhouettes,
        intrinsics,
        rotations,
        translations,
        mask_confidence=mask_confidence,
        motion_uncertainty_per_view=motion,
        semantic_masks=semantic_masks,
        source_hashes=source_hashes,
        **build_arguments,  # type: ignore[arg-type]
    )
    corrupted_mask_agreement = evidence_volume_agreement(reference, corrupted)
    benchmark = read_json(public_benchmark_path)
    thin_gap = benchmark.get("cases", {}).get("thin_gap_hairpin", {})
    thin_gap_gate = (
        benchmark.get("status") == "pass"
        and float(thin_gap.get("relative_treatment_improvement", -math.inf)) >= 0.10
    )
    cleanup_free = reference.metadata.cleanup_operations == []
    unsupported_fraction = float(reference.unsupported.float().mean())
    blockers: list[str] = []
    if not replay_exact:
        blockers.append("visual_hull_exact_replay_failed")
    if dense_sparse_agreement["maximum_supported_surface_displacement_voxels"] > 1.0:
        blockers.append("dense_sparse_surface_disagreement_exceeds_one_voxel")
    if corrupted_mask_agreement["maximum_supported_surface_displacement_voxels"] > 2.0:
        blockers.append("single_corrupted_mask_moves_surface_beyond_two_voxels")
    if not cleanup_free:
        blockers.append("visual_hull_contains_unregistered_cleanup")
    if not thin_gap_gate:
        blockers.append("independent_thin_gap_benchmark_failed")
    if unsupported_fraction > 0.25:
        blockers.append("unsupported_fraction_exceeds_0_25")
    report = {
        "schema_version": "frayid_v2_visual_hull_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "training_view_count": int(silhouettes.shape[0]),
        "sparse_view_count": int(sparse_slots.numel()),
        "same_device_replay_exact": replay_exact,
        "dense_sparse_agreement": dense_sparse_agreement,
        "corrupted_mask_slot": corrupted_slot,
        "corrupted_mask_agreement": corrupted_mask_agreement,
        "independent_thin_gap_gate": thin_gap_gate,
        "cleanup_operations": reference.metadata.cleanup_operations,
        "unsupported_fraction": unsupported_fraction,
        "unsupported_fraction_maximum": 0.25,
        "semantic_layer_evidence_present": bool(semantic_masks),
        "semantic_layer_status": (
            "bound" if semantic_masks else "absent_layering_must_remain_blocked"
        ),
        "source_hashes": {
            "binding": sha256_file(binding_path),
            "reference_volume": sha256_file(reference_volume_path),
            "public_benchmark": sha256_file(public_benchmark_path),
        },
        "blockers": blockers,
        "optimizer_steps": 0,
        "training_masks_read": int(silhouettes.shape[0]),
        "legacy_development_images_read": 0,
        "sealed_test_accesses": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
    }
    return write_json(report_output_path, report)
