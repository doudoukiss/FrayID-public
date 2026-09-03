from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation, Slerp  # type: ignore[import-untyped]
from torch import Tensor

from frayid.camera import make_intrinsics
from frayid.config import ReconstructionConfig
from frayid.dataset import DATASET_MANIFEST_FILENAME, read_dataset_manifest
from frayid.geometry import (
    canonical_face_orientation_report,
    linear_blend_skinning,
    rigid_transform_from_axis_angle,
)
from frayid.initialization import load_initialization
from frayid.io import read_json, write_json
from frayid.renderer import (
    normalized_boundary_error,
    render_soft_mesh,
    soft_silhouette_iou,
)
from frayid.replay_state import CHECKPOINT_SCHEMA_V2, CheckpointStateV2
from frayid.schemas import ReconstructionEvaluation
from frayid.training import CanonicalGeometryModel, load_canonical_model_state


@dataclass(frozen=True)
class EvaluationCheckpoint:
    """Read-only checkpoint view used by evaluation without restoring runtime state."""

    schema_version: str
    model_state: dict[str, Any]
    next_step_replay_capable: bool


@dataclass(frozen=True)
class EvaluatedGeometry:
    """The one geometry object shared by rendering, export, and topology checks."""

    vertices: Tensor
    faces: Tensor
    weights: Tensor
    schema_version: str
    source_kind: str

    def numpy_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.vertices.detach().cpu().numpy(),
            self.faces.detach().cpu().numpy(),
            self.weights.detach().cpu().numpy(),
        )


def load_evaluation_checkpoint(path: Path, device: torch.device) -> EvaluationCheckpoint:
    """Load v1/v2 model state without claiming or restoring next-step execution state."""
    payload = torch.load(path, map_location=device, weights_only=False)
    schema = payload.get("schema_version")
    if schema == CHECKPOINT_SCHEMA_V2:
        state = CheckpointStateV2.from_state_dict(payload)
        return EvaluationCheckpoint(
            schema_version=CHECKPOINT_SCHEMA_V2,
            model_state=state.model_state,
            next_step_replay_capable=True,
        )
    if schema == "canonical_checkpoint.v1":
        state = payload.get("model")
        if not isinstance(state, dict):
            raise ValueError("canonical_checkpoint.v1 does not contain model state")
        return EvaluationCheckpoint(
            schema_version="canonical_checkpoint.v1",
            model_state=state,
            next_step_replay_capable=False,
        )
    raise ValueError("Unsupported checkpoint schema")


def evaluated_geometry_topology_report(
    geometry: EvaluatedGeometry,
    *,
    checkpoint_reference_vertices: np.ndarray,
    checkpoint_faces: np.ndarray,
    minimum_area_ratio: float,
) -> dict[str, Any]:
    """Check the actual evaluated arrays without applying unrelated face references."""
    vertices, faces, _ = geometry.numpy_arrays()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    same_connectivity = bool(
        vertices.shape == checkpoint_reference_vertices.shape
        and np.array_equal(faces, checkpoint_faces)
    )
    relative = (
        canonical_face_orientation_report(
            checkpoint_reference_vertices,
            vertices,
            faces,
            minimum_area_ratio=minimum_area_ratio,
        )
        if same_connectivity
        else None
    )
    blockers: list[str] = []
    if not mesh.is_watertight:
        blockers.append("actual_geometry_not_watertight")
    if not mesh.is_winding_consistent:
        blockers.append("actual_geometry_inconsistent_winding")
    if len(components) != 1:
        blockers.append("actual_geometry_component_count_not_one")
    if int(mesh.euler_number) != 2:
        blockers.append("actual_geometry_euler_not_two")
    if not np.isfinite(vertices).all() or not np.isfinite(mesh.volume) or float(mesh.volume) <= 0:
        blockers.append("actual_geometry_nonpositive_or_nonfinite_volume")
    if relative is not None:
        blockers.extend(str(value) for value in relative["blockers"])
    return {
        "schema_version": "evaluated_geometry_topology.v1",
        "status": "pass" if not blockers else "fail",
        "geometry_schema_version": geometry.schema_version,
        "source_kind": geometry.source_kind,
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "same_connectivity_as_checkpoint": same_connectivity,
        "reference_contract": "checkpoint_base"
        if same_connectivity
        else "intrinsic_new_connectivity",
        "relative_face_report": relative,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "component_count": len(components),
        "euler_number": int(mesh.euler_number),
        "signed_volume": float(mesh.volume),
        "blockers": blockers,
    }


def evaluate_checkpoint(
    config: ReconstructionConfig,
    *,
    checkpoint_path: Path,
    output_directory: Path,
    device_name: str | None = None,
    geometry_mesh_path: Path | None = None,
    geometry_archive_path: Path | None = None,
    residual_archive_path: Path | None = None,
    interpolate_held_out_transforms: bool = False,
    robust_transform_mad_multiplier: float | None = None,
    initialization_path: Path | None = None,
    renderer: Callable[..., tuple[Tensor, Tensor]] = render_soft_mesh,
) -> ReconstructionEvaluation:
    """Render one checkpoint against train and untouched held-out evidence.

    ``geometry_archive_path`` is the topology-preserving carrier contract: its
    NPZ must contain ``vertices``, ``faces``, and exact interpolated ``weights``.
    ``geometry_mesh_path`` remains available for SDF-extracted meshes whose
    weights must be transferred from the scaffold by nearest canonical vertex.
    """
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_evaluation_checkpoint(checkpoint_path, device)
    state = checkpoint.model_state
    frame_code_key = "deformer.frame_codes.weight"
    if frame_code_key not in state:
        raise ValueError("Checkpoint does not contain residual frame codes")
    model = CanonicalGeometryModel(
        state["base_vertices"].to(device),
        state["faces"].to(device),
        state["skinning_weights"].to(device),
        int(state[frame_code_key].shape[0]),
        config,
    ).to(device)
    load_canonical_model_state(model, state)
    model.eval()
    checkpoint_topology_report = canonical_face_orientation_report(
        model.base_vertices.detach().cpu().numpy(),
        model.canonical_vertices.detach().cpu().numpy(),
        model.faces.detach().cpu().numpy(),
        minimum_area_ratio=config.model.canonical_minimum_area_ratio,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "checkpoint_topology_report.json", checkpoint_topology_report)
    geometry = EvaluatedGeometry(
        vertices=model.canonical_vertices,
        faces=model.faces,
        weights=model.skinning_weights,
        schema_version="checkpoint_geometry.v1",
        source_kind="checkpoint",
    )
    if geometry_mesh_path is not None and geometry_archive_path is not None:
        raise ValueError("Geometry mesh and carrier archive are mutually exclusive")
    if geometry_archive_path is not None:
        archive = np.load(geometry_archive_path)
        required = {"vertices", "faces", "weights"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("Geometry archive is missing: " + ", ".join(missing))
        vertices = np.asarray(archive["vertices"], dtype=np.float32)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        weights = np.asarray(archive["weights"], dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Geometry archive vertices must have shape [V, 3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Geometry archive faces must have shape [F, 3]")
        if weights.ndim != 2 or weights.shape[0] != vertices.shape[0]:
            raise ValueError("Geometry archive weights must have shape [V, J]")
        if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
            raise ValueError("Geometry archive faces reference an invalid vertex")
        weight_sums = weights.sum(axis=1)
        if not np.isfinite(vertices).all() or not np.isfinite(weights).all():
            raise ValueError("Geometry archive contains non-finite values")
        if not np.allclose(weight_sums, 1.0, atol=1e-4):
            raise ValueError("Geometry archive skinning weights must sum to one")
        archive_schema = (
            str(np.asarray(archive["schema_version"]).item())
            if "schema_version" in archive.files
            else "external_geometry_archive.v1"
        )
        geometry = EvaluatedGeometry(
            vertices=torch.tensor(vertices, dtype=torch.float32, device=device),
            faces=torch.tensor(faces, dtype=torch.long, device=device),
            weights=torch.tensor(weights, dtype=torch.float32, device=device),
            schema_version=archive_schema,
            source_kind="archive",
        )
    elif geometry_mesh_path is not None:
        extracted = trimesh.load_mesh(geometry_mesh_path, process=False)
        if not isinstance(extracted, trimesh.Trimesh):
            raise ValueError("SDF geometry path must contain exactly one mesh")
        geometry_vertices = torch.tensor(extracted.vertices, dtype=torch.float32, device=device)
        geometry_faces = torch.tensor(extracted.faces, dtype=torch.long, device=device)
        nearest = cKDTree(model.canonical_vertices.detach().cpu().numpy()).query(
            extracted.vertices, k=1
        )[1]
        geometry_weights = model.skinning_weights[
            torch.tensor(nearest, dtype=torch.long, device=device)
        ]
        geometry = EvaluatedGeometry(
            vertices=geometry_vertices,
            faces=geometry_faces,
            weights=geometry_weights,
            schema_version="external_mesh_nearest_vertex_weights.v1",
            source_kind="mesh",
        )
    geometry_vertices = geometry.vertices
    geometry_faces = geometry.faces
    geometry_weights = geometry.weights
    topology_report = evaluated_geometry_topology_report(
        geometry,
        checkpoint_reference_vertices=model.base_vertices.detach().cpu().numpy(),
        checkpoint_faces=model.faces.detach().cpu().numpy(),
        minimum_area_ratio=config.model.canonical_minimum_area_ratio,
    )
    write_json(output_directory / "canonical_topology_report.json", topology_report)
    direct_residuals: Tensor | None = None
    direct_residual_indices: list[int] | None = None
    if residual_archive_path is not None:
        residual_archive = np.load(residual_archive_path)
        required = {"source_frame_indices", "residual_offsets"}
        missing = sorted(required - set(residual_archive.files))
        if missing:
            raise ValueError("Residual archive is missing: " + ", ".join(missing))
        residual_indices = np.asarray(residual_archive["source_frame_indices"], dtype=np.int64)
        residual_offsets = np.asarray(residual_archive["residual_offsets"], dtype=np.float32)
        if residual_indices.ndim != 1 or residual_offsets.ndim != 3:
            raise ValueError("Residual archive must have indices [T] and offsets [T, V, 3]")
        if residual_offsets.shape != (len(residual_indices), len(geometry_vertices), 3):
            raise ValueError("Residual archive shape does not match evaluated geometry")
        if len(set(residual_indices.tolist())) != len(residual_indices):
            raise ValueError("Residual archive source indices must be unique")
        order = np.argsort(residual_indices)
        direct_residual_indices = residual_indices[order].tolist()
        direct_residuals = torch.tensor(residual_offsets[order], dtype=torch.float32, device=device)
    manifest = read_dataset_manifest(config.paths.dataset_root / DATASET_MANIFEST_FILENAME)
    initialization = load_initialization(
        initialization_path or config.paths.dataset_root / config.evidence.initialization_filename
    )
    if not initialization.joint_transforms_path:
        raise ValueError("Initialization has no real per-frame joint transforms")
    transform_data = np.load(initialization.joint_transforms_path)
    if interpolate_held_out_transforms and robust_transform_mad_multiplier is not None:
        raise ValueError("Transform interpolation and robust smoothing are mutually exclusive")
    transform_arrays = transform_data["transforms"]
    transform_smoothing: dict[str, object] | None = None
    repaired_transform_ordinals: set[int] = set()
    if robust_transform_mad_multiplier is not None:
        transform_arrays, transform_smoothing = robustly_smooth_joint_transforms(
            transform_data["source_frame_indices"],
            transform_arrays,
            mad_multiplier=robust_transform_mad_multiplier,
        )
        repaired_values = transform_smoothing["repaired_ordinals"]
        if not isinstance(repaired_values, list):
            raise TypeError("Robust transform report has invalid repaired ordinals")
        repaired_transform_ordinals = {int(value) for value in repaired_values}
    transform_lookup = {
        int(value): index
        for index, value in enumerate(transform_data["source_frame_indices"].tolist())
    }
    provenance_path = checkpoint_path.parent / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("Checkpoint provenance is required for held-out evaluation")
    provenance = read_json(provenance_path)
    trained_indices = [int(value) for value in provenance.get("frame_indices", [])]
    if len(trained_indices) != model.deformer.frame_codes.num_embeddings:
        raise ValueError("Provenance frame indices do not match checkpoint frame codes")
    intrinsics = make_intrinsics(
        initialization.shared_focal_length_px,
        initialization.shared_principal_point_px,
        device=device,
    )
    resolution = config.model.renderer_resolution
    split_ious: dict[str, list[float]] = {"train": [], "held_out": []}
    held_out_initial_ious: list[float] = []
    held_out_boundary: list[float] = []
    held_out_normal_errors: list[float] = []
    per_frame_metrics: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for record in manifest.frames:
            source_index = record.source_frame_index
            if source_index not in transform_lookup:
                raise ValueError(f"Missing SMPL transform for source frame {source_index}")
            transform_array = transform_arrays[transform_lookup[source_index]]
            if interpolate_held_out_transforms and record.split == "held_out":
                transform_array = _interpolate_joint_transforms(
                    source_index,
                    trained_indices,
                    transform_arrays,
                    transform_lookup,
                )
            transform = torch.tensor(transform_array, dtype=torch.float32, device=device)
            code = _interpolate_frame_code(
                source_index, trained_indices, model.deformer.frame_codes.weight
            )
            root_rotation = _interpolate_frame_code(
                source_index, trained_indices, model.root_rotation_corrections
            )
            root_translation = _interpolate_frame_code(
                source_index, trained_indices, model.root_translation_corrections
            )
            root_correction = rigid_transform_from_axis_angle(root_rotation, root_translation)
            corrected_transform = root_correction.unsqueeze(0) @ transform
            residual = (
                model.deformer.forward_with_code(geometry_vertices, code)
                if direct_residuals is None or direct_residual_indices is None
                else _interpolate_frame_code(
                    source_index, direct_residual_indices, direct_residuals
                )
            )
            posed = linear_blend_skinning(
                geometry_vertices + residual,
                geometry_weights,
                corrected_transform,
            )
            initial_posed = linear_blend_skinning(
                model.base_vertices, model.skinning_weights, transform
            )
            name = Path(record.image_path).name
            mask_image = cv2.imread(
                str(config.paths.dataset_root / config.evidence.masks_subdirectory / name),
                cv2.IMREAD_GRAYSCALE,
            )
            normal_image = cv2.imread(
                str(config.paths.dataset_root / config.evidence.normals_subdirectory / name),
                cv2.IMREAD_COLOR,
            )
            if mask_image is None or normal_image is None:
                raise FileNotFoundError(f"Missing validated evidence for {name}")
            mask = torch.tensor(mask_image / 255.0, dtype=torch.float32, device=device)
            mask = torch.nn.functional.interpolate(
                mask[None, None],
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            normal = cv2.resize(
                normal_image, (resolution, resolution), interpolation=cv2.INTER_LINEAR
            )
            target_normal = torch.tensor(
                normal[..., ::-1].copy() / 127.5 - 1.0,
                dtype=torch.float32,
                device=device,
            )
            target_normal = torch.nn.functional.normalize(target_normal, dim=-1, eps=1e-8)
            torch.manual_seed(config.seed + record.ordinal)
            silhouette, predicted_normal = renderer(
                posed,
                geometry_faces,
                intrinsics,
                (resolution, resolution),
                source_image_size=(initialization.image_height, initialization.image_width),
                sigma_pixels=config.model.renderer_sigma_pixels,
                sample_count=config.model.renderer_max_vertices,
                reference_sample_count=config.model.renderer_reference_sample_count,
                depth_temperature_m=config.model.renderer_depth_temperature_m,
            )
            iou = float(soft_silhouette_iou(silhouette, mask).cpu())
            split_ious[record.split].append(iou)
            frame_metrics: dict[str, float | int | str] = {
                "ordinal": record.ordinal,
                "source_frame_index": source_index,
                "split": record.split,
                "silhouette_iou": iou,
                "transform_source": (
                    "interpolated_train_neighbors"
                    if interpolate_held_out_transforms and record.split == "held_out"
                    else (
                        "robust_sequence_repair"
                        if transform_smoothing is not None
                        and record.ordinal in repaired_transform_ordinals
                        else "observed_initialization"
                    )
                ),
            }
            if record.split == "held_out":
                torch.manual_seed(config.seed + record.ordinal)
                initial_silhouette, _ = renderer(
                    initial_posed,
                    model.faces,
                    intrinsics,
                    (resolution, resolution),
                    source_image_size=(initialization.image_height, initialization.image_width),
                    sigma_pixels=config.model.renderer_sigma_pixels,
                    sample_count=config.model.renderer_max_vertices,
                    reference_sample_count=config.model.renderer_reference_sample_count,
                    depth_temperature_m=config.model.renderer_depth_temperature_m,
                )
                initial_iou = float(soft_silhouette_iou(initial_silhouette, mask).cpu())
                boundary = normalized_boundary_error(silhouette, mask)
                held_out_initial_ious.append(initial_iou)
                held_out_boundary.append(boundary)
                frame_metrics["initialization_silhouette_iou"] = initial_iou
                frame_metrics["silhouette_iou_improvement"] = iou - initial_iou
                frame_metrics["normalized_boundary_error"] = boundary
                valid = mask > 0.5
                cosine = (
                    (
                        torch.nn.functional.normalize(predicted_normal[valid], dim=-1, eps=1e-8)
                        * target_normal[valid]
                    )
                    .sum(-1)
                    .clamp(-1.0, 1.0)
                )
                if cosine.numel():
                    normal_errors = torch.rad2deg(torch.acos(cosine)).cpu()
                    held_out_normal_errors.extend(normal_errors.tolist())
                    frame_metrics["median_normal_error_degrees"] = float(
                        torch.median(normal_errors)
                    )
            per_frame_metrics.append(frame_metrics)
    if not split_ious["train"] or not split_ious["held_out"]:
        raise ValueError("Both train and held-out frames are required for evaluation")
    if not held_out_normal_errors:
        raise ValueError("No valid held-out normal pixels were evaluated")
    output_directory.mkdir(parents=True, exist_ok=True)
    mesh_path = output_directory / "canonical_mesh.ply"
    trimesh.Trimesh(
        vertices=geometry_vertices.detach().cpu().numpy(),
        faces=geometry_faces.detach().cpu().numpy(),
        process=False,
    ).export(mesh_path)
    metrics_path = output_directory / "held_out_metrics.json"
    write_json(
        metrics_path,
        {
            "schema_version": "held_out_metrics.v1",
            "train_silhouette_iou": float(np.median(split_ious["train"])),
            "held_out_silhouette_iou": float(np.median(split_ious["held_out"])),
            "initialization_held_out_iou": float(np.median(held_out_initial_ious)),
            "normalized_boundary_error": float(np.median(held_out_boundary)),
            "median_normal_error_degrees": float(np.median(held_out_normal_errors)),
            "train_frame_count": len(split_ious["train"]),
            "held_out_frame_count": len(split_ious["held_out"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_schema_version": checkpoint.schema_version,
            "next_step_replay_capable": checkpoint.next_step_replay_capable,
            "evaluated_geometry_schema_version": geometry.schema_version,
            "evaluated_geometry_source_kind": geometry.source_kind,
        },
    )
    write_json(
        output_directory / "per_frame_metrics.json",
        {
            "schema_version": "per_frame_geometry_metrics.v1",
            "checkpoint_path": str(checkpoint_path),
            "transform_smoothing": transform_smoothing,
            "frames": per_frame_metrics,
        },
    )
    report = evaluate_reconstruction(
        config,
        metrics_path=metrics_path,
        mesh_path=mesh_path,
        output_path=output_directory / "reconstruction_evaluation.json",
    )
    topology_blockers = [str(value) for value in topology_report["blockers"]]
    relative_face_report = topology_report["relative_face_report"]
    report = report.model_copy(
        update={
            "status": "fail" if topology_blockers else report.status,
            "canonical_topology_valid": not topology_blockers,
            "canonical_flipped_face_fraction": (
                relative_face_report["flipped_face_fraction"]
                if relative_face_report is not None
                else None
            ),
            "canonical_collapsed_face_fraction": (
                relative_face_report["collapsed_face_fraction"]
                if relative_face_report is not None
                else None
            ),
            "blockers": [*report.blockers, *topology_blockers],
        }
    )
    write_json(output_directory / "reconstruction_evaluation.json", report)
    return report


def evaluate_reconstruction(
    config: ReconstructionConfig,
    *,
    metrics_path: Path,
    mesh_path: Path,
    output_path: Path | None = None,
) -> ReconstructionEvaluation:
    """Apply the fixed held-out metric and connected-component gates."""
    blockers: list[str] = []
    if not metrics_path.is_file():
        blockers.append("missing_held_out_metrics")
    if not mesh_path.is_file():
        blockers.append("missing_canonical_mesh")
    if blockers:
        report = ReconstructionEvaluation(status="blocked", blockers=blockers)
        if output_path:
            write_json(output_path, report)
        return report
    payload = read_json(metrics_path)
    required = (
        "train_silhouette_iou",
        "held_out_silhouette_iou",
        "initialization_held_out_iou",
        "normalized_boundary_error",
        "median_normal_error_degrees",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        report = ReconstructionEvaluation(
            status="blocked", blockers=[f"missing_metric:{name}" for name in missing]
        )
        if output_path:
            write_json(output_path, report)
        return report
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Canonical mesh file must contain exactly one mesh")
    components = mesh.split(only_watertight=False)
    total_area = float(sum(component.area for component in components))
    dominant_fraction = (
        float(max((component.area for component in components), default=0.0) / total_area)
        if total_area > 0
        else 0.0
    )
    train_iou = float(payload["train_silhouette_iou"])
    held_out_iou = float(payload["held_out_silhouette_iou"])
    initial_iou = float(payload["initialization_held_out_iou"])
    boundary = float(payload["normalized_boundary_error"])
    normal = float(payload["median_normal_error_degrees"])
    checks = {
        "held_out_iou_below_gate": held_out_iou < config.evaluation.held_out_silhouette_iou,
        "held_out_improvement_below_gate": held_out_iou - initial_iou
        < config.evaluation.minimum_iou_improvement,
        "boundary_error_above_gate": boundary > config.evaluation.maximum_normalized_boundary_error,
        "normal_error_above_gate": normal > config.evaluation.maximum_median_normal_error_degrees,
        "train_held_out_gap_above_gate": train_iou - held_out_iou
        > config.evaluation.maximum_train_held_out_iou_gap,
        "dominant_component_below_gate": dominant_fraction
        < config.evaluation.minimum_dominant_component_area_fraction,
        "canonical_mesh_not_watertight": not mesh.is_watertight,
    }
    blockers = [name for name, failed in checks.items() if failed]
    report = ReconstructionEvaluation(
        status="pass" if not blockers else "fail",
        train_silhouette_iou=train_iou,
        held_out_silhouette_iou=held_out_iou,
        initialization_held_out_iou=initial_iou,
        normalized_boundary_error=boundary,
        median_normal_error_degrees=normal,
        dominant_component_area_fraction=dominant_fraction,
        canonical_mesh_watertight=mesh.is_watertight,
        blockers=blockers,
    )
    destination = output_path or metrics_path.parent / "reconstruction_evaluation.json"
    write_json(destination, report)
    return report


def _interpolate_frame_code(
    source_index: int,
    trained_indices: list[int],
    codes: torch.Tensor,
) -> torch.Tensor:
    order = np.argsort(trained_indices)
    sorted_indices = np.asarray(trained_indices, dtype=np.int64)[order]
    sorted_codes = codes[torch.tensor(order, dtype=torch.long, device=codes.device)]
    position = int(np.searchsorted(sorted_indices, source_index))
    if position == 0:
        return sorted_codes[0]
    if position == len(sorted_indices):
        return sorted_codes[-1]
    left_index = int(sorted_indices[position - 1])
    right_index = int(sorted_indices[position])
    fraction = (source_index - left_index) / max(right_index - left_index, 1)
    return sorted_codes[position - 1] * (1.0 - fraction) + sorted_codes[position] * fraction


def _interpolate_joint_transforms(
    source_index: int,
    trained_indices: list[int],
    transforms: np.ndarray,
    transform_lookup: dict[int, int],
) -> np.ndarray:
    """Interpolate held-out SMPL/LBS transforms from neighboring train frames."""
    sorted_indices = sorted(trained_indices)
    position = int(np.searchsorted(sorted_indices, source_index))
    if position == 0:
        return np.asarray(transforms[transform_lookup[sorted_indices[0]]], dtype=np.float32)
    if position == len(sorted_indices):
        return np.asarray(transforms[transform_lookup[sorted_indices[-1]]], dtype=np.float32)
    left_index = sorted_indices[position - 1]
    right_index = sorted_indices[position]
    fraction = (source_index - left_index) / max(right_index - left_index, 1)
    left = np.asarray(transforms[transform_lookup[left_index]], dtype=np.float64)
    right = np.asarray(transforms[transform_lookup[right_index]], dtype=np.float64)
    result = np.zeros_like(left)
    for joint in range(left.shape[0]):
        rotations = Rotation.from_matrix(
            np.stack((left[joint, :3, :3], right[joint, :3, :3]), axis=0)
        )
        result[joint, :3, :3] = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
        result[joint, :3, 3] = (
            left[joint, :3, 3] * (1.0 - fraction) + right[joint, :3, 3] * fraction
        )
        result[joint, 3, 3] = 1.0
    return result.astype(np.float32)


def robustly_smooth_joint_transforms(
    source_frame_indices: np.ndarray,
    transforms: np.ndarray,
    *,
    mad_multiplier: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Replace sequence transform outliers using a mask-independent robust rule.

    Each interior frame is compared with rigid SE(3) interpolation of its immediate
    temporal neighbors. The score is the median geodesic rotation error across all
    joints. Frames above ``median + mad_multiplier * MAD`` are replaced in one pass;
    predictions and scores always use the untouched input sequence.
    """
    indices = np.asarray(source_frame_indices, dtype=np.int64)
    values = np.asarray(transforms)
    if indices.ndim != 1 or values.ndim != 4 or values.shape[0] != indices.shape[0]:
        raise ValueError("Transform sequence must have shape [frame, joint, 4, 4]")
    if len(indices) < 3:
        raise ValueError("Robust transform smoothing requires at least three frames")
    if not np.all(np.diff(indices) > 0):
        raise ValueError("Source frame indices must be strictly increasing")
    if mad_multiplier <= 0:
        raise ValueError("MAD multiplier must be positive")

    lookup = {int(value): position for position, value in enumerate(indices.tolist())}
    predictions: dict[int, np.ndarray] = {}
    scores = np.full(len(indices), np.nan, dtype=np.float64)
    for ordinal in range(1, len(indices) - 1):
        source_index = int(indices[ordinal])
        prediction = _interpolate_joint_transforms(
            source_index,
            [int(indices[ordinal - 1]), int(indices[ordinal + 1])],
            values,
            lookup,
        )
        predictions[ordinal] = prediction
        angular_errors = []
        for joint in range(values.shape[1]):
            relative = prediction[joint, :3, :3].T @ values[ordinal, joint, :3, :3]
            angular_errors.append(float(np.rad2deg(Rotation.from_matrix(relative).magnitude())))
        scores[ordinal] = float(np.median(angular_errors))

    finite_scores = scores[np.isfinite(scores)]
    score_median = float(np.median(finite_scores))
    score_mad = float(np.median(np.abs(finite_scores - score_median)))
    threshold = score_median + mad_multiplier * score_mad
    repaired_ordinals = [
        ordinal for ordinal in range(1, len(indices) - 1) if float(scores[ordinal]) > threshold
    ]
    smoothed = values.astype(np.float32, copy=True)
    for ordinal in repaired_ordinals:
        smoothed[ordinal] = predictions[ordinal]
    report: dict[str, object] = {
        "method": "median_joint_rotation_neighbor_se3_mad_v1",
        "mad_multiplier": float(mad_multiplier),
        "score_median_degrees": score_median,
        "score_mad_degrees": score_mad,
        "threshold_degrees": threshold,
        "repaired_ordinals": repaired_ordinals,
        "repaired_source_frame_indices": [int(indices[value]) for value in repaired_ordinals],
        "repaired_scores_degrees": [float(scores[value]) for value in repaired_ordinals],
    }
    return smoothed, report
