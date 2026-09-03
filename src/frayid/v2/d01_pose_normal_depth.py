from __future__ import annotations

import hashlib
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]
from scipy.sparse import coo_matrix, vstack  # type: ignore[import-untyped]
from scipy.sparse.linalg import lsqr  # type: ignore[import-untyped]

from frayid.config import ReconstructionConfig
from frayid.geometry import canonical_face_orientation_report
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.g03_appearance import painter_visibility
from frayid.v2.g03_pipeline import _load_inputs, _posed

D01_EXPERIMENT_ID = "postv2_d01_pose_stabilized_normal_depth_fusion_r01"
D01_PUBLIC_SCHEMA = "frayid_v2_d01_public_pose_normal_depth_benchmark.v1"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(lengths <= 1.0e-12) or not np.all(np.isfinite(lengths)):
        raise ValueError("normal vectors must be finite and nonzero")
    return np.asarray(values / lengths, dtype=np.float64)


def normals_from_height(height: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    """Return +z-oriented unit normals for a regularly sampled height field."""

    values = np.asarray(height, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 3:
        raise ValueError("height field must be a two-dimensional grid of at least 3 by 3")
    spacing_y, spacing_x = spacing
    if spacing_y <= 0.0 or spacing_x <= 0.0:
        raise ValueError("height-field spacing must be positive")
    gradient_y, gradient_x = np.gradient(values, spacing_y, spacing_x, edge_order=2)
    return _normalize(np.stack((-gradient_x, -gradient_y, np.ones_like(values)), axis=-1))


def pose_normals_from_canonical(
    canonical_normals: np.ndarray,
    jacobians: np.ndarray,
) -> np.ndarray:
    """Apply the inverse-transpose normal rule from canonical to posed space."""

    normals = np.asarray(canonical_normals, dtype=np.float64)
    transforms = np.asarray(jacobians, dtype=np.float64)
    if normals.shape[-1] != 3 or transforms.shape[-2:] != (3, 3):
        raise ValueError("normal and Jacobian shapes are incompatible")
    if transforms.shape[:-2] != normals.shape[:-1]:
        transforms = np.broadcast_to(transforms, (*normals.shape[:-1], 3, 3))
    transported = np.linalg.solve(np.swapaxes(transforms, -1, -2), normals[..., None])[..., 0]
    return _normalize(transported)


def transport_normals_to_canonical(
    posed_normals: np.ndarray,
    jacobians: np.ndarray,
) -> np.ndarray:
    """Pull posed-space covectors back through the frozen pose Jacobian."""

    normals = np.asarray(posed_normals, dtype=np.float64)
    transforms = np.asarray(jacobians, dtype=np.float64)
    if normals.shape[-1] != 3 or transforms.shape[-2:] != (3, 3):
        raise ValueError("normal and Jacobian shapes are incompatible")
    if transforms.shape[:-2] != normals.shape[:-1]:
        transforms = np.broadcast_to(transforms, (*normals.shape[:-1], 3, 3))
    transported = np.einsum("...ji,...j->...i", transforms, normals)
    return _normalize(transported)


def robust_fuse_normals(
    observations: np.ndarray,
    confidence: np.ndarray,
    *,
    reference: np.ndarray,
    huber_degrees: float = 8.0,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly aggregate aligned unit normals without allowing sign flips."""

    raw = np.asarray(observations, dtype=np.float64)
    weights = np.asarray(confidence, dtype=np.float64)
    prior = _normalize(reference)
    if raw.ndim < 3 or raw.shape[-1] != 3:
        raise ValueError("normal observations must have shape [view,...,3]")
    if weights.shape != raw.shape[:-1]:
        raise ValueError("normal confidence must match observation rows")
    if prior.shape != raw.shape[1:]:
        raise ValueError("reference normal shape does not match observations")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("normal confidence must be finite and nonnegative")
    active = weights > 0.0
    if not np.any(active):
        raise ValueError("normal fusion requires at least one supported observation")
    if np.any(~np.isfinite(raw[active])):
        raise ValueError("supported normal observations must be finite")
    safe = raw.copy()
    safe[~active] = np.broadcast_to(prior, raw.shape)[~active]
    values = _normalize(safe)
    aligned = values.copy()
    flip = np.sum(aligned * prior[None], axis=-1) < 0.0
    aligned[flip] *= -1.0
    estimate = _normalize(np.sum(aligned * weights[..., None], axis=0) + prior * 1.0e-8)
    cutoff = math.radians(huber_degrees)
    if cutoff <= 0.0 or iterations < 1:
        raise ValueError("robust-normal settings are invalid")
    effective = weights.copy()
    for _ in range(iterations):
        cosine = np.clip(np.sum(aligned * estimate[None], axis=-1), -1.0, 1.0)
        angle = np.arccos(cosine)
        robust = np.minimum(1.0, cutoff / np.maximum(angle, 1.0e-12))
        effective = weights * robust
        estimate = _normalize(np.sum(aligned * effective[..., None], axis=0) + prior * 1.0e-8)
    support = np.sum(effective, axis=0)
    estimate[support <= 0.0] = prior[support <= 0.0]
    return estimate, support


def decode_sapiens_normal_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Decode the frozen Sapiens/E25 convention into camera geometry axes."""

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Sapiens normal image must have three BGR channels")
    sapiens_xyz = image[..., ::-1].astype(np.float64) / 127.5 - 1.0
    geometry_xyz = sapiens_xyz * np.asarray([1.0, -1.0, -1.0])
    lengths = np.linalg.norm(geometry_xyz, axis=-1)
    valid = lengths >= 0.25
    result = np.zeros_like(geometry_xyz)
    result[valid] = geometry_xyz[valid] / lengths[valid, None]
    return np.asarray(result, dtype=np.float64)


def _face_frames(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(first, second)
    lengths = np.linalg.norm(cross, axis=-1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("D01 cannot transport through degenerate faces")
    normals = cross / lengths[:, None]
    scale = 0.5 * (np.linalg.norm(first, axis=-1) + np.linalg.norm(second, axis=-1))
    frames = np.stack((first, second, normals * scale[:, None]), axis=-1)
    return frames, normals


def pull_back_face_normals(
    observed_posed_normals: np.ndarray,
    canonical_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Pull camera-space face normals through per-face deformation Jacobians."""

    observed = _normalize(observed_posed_normals)
    canonical_frames, canonical_normals = _face_frames(canonical_vertices, faces)
    posed_frames, _ = _face_frames(posed_vertices, faces)
    if observed.shape != canonical_normals.shape:
        raise ValueError("observed face normals do not match mesh faces")
    jacobian_transpose = np.linalg.solve(
        np.swapaxes(canonical_frames, -1, -2),
        np.swapaxes(posed_frames, -1, -2),
    )
    pulled = _normalize(np.einsum("fij,fj->fi", jacobian_transpose, observed))
    flip = np.sum(pulled * canonical_normals, axis=-1) < 0.0
    pulled[flip] *= -1.0
    return pulled


def sample_pose_stabilized_vertex_normals(
    canonical_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    normal_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    source_size: tuple[int, int],
    erosion_pixels: int = 3,
    minimum_face_pixels: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Sample observed face normals and pull them to canonical mesh vertices."""

    canonical = np.asarray(canonical_vertices, dtype=np.float64)
    posed = np.asarray(posed_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    image = np.asarray(normal_bgr)
    mask = np.asarray(foreground_mask)
    if canonical.shape != posed.shape or canonical.ndim != 2 or canonical.shape[1] != 3:
        raise ValueError("D01 canonical and posed vertices must share shape [V,3]")
    if image.shape[:2] != source_size or mask.shape != source_size:
        raise ValueError("D01 normal and mask evidence do not match source size")
    if erosion_pixels < 0 or minimum_face_pixels < 1:
        raise ValueError("D01 sampling settings are invalid")
    foreground = mask > 127
    if erosion_pixels:
        kernel_size = 2 * erosion_pixels + 1
        foreground = cv2.erode(
            foreground.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        ).astype(bool)
    face_ids, _ = painter_visibility(
        posed,
        triangles,
        intrinsics,
        source_size=source_size,
        output_size=source_size,
    )
    decoded = decode_sapiens_normal_bgr(image)
    decoded_length = np.linalg.norm(decoded, axis=-1)
    valid_pixels = foreground & (face_ids >= 0) & (decoded_length > 0.5)
    pixel_faces = face_ids[valid_pixels].astype(np.int64)
    pixel_normals = decoded[valid_pixels]
    face_count = len(triangles)
    counts = np.bincount(pixel_faces, minlength=face_count).astype(np.int64)
    sums = np.stack(
        [
            np.bincount(pixel_faces, weights=pixel_normals[:, axis], minlength=face_count)
            for axis in range(3)
        ],
        axis=-1,
    )
    sum_length = np.linalg.norm(sums, axis=-1)
    concentration = np.divide(
        sum_length,
        counts,
        out=np.zeros(face_count, dtype=np.float64),
        where=counts > 0,
    )
    face_valid = (counts >= minimum_face_pixels) & (concentration >= 0.55)
    if not np.any(face_valid):
        raise ValueError("D01 frame has no reliable observed face normals")
    posed_face_normals = np.zeros((face_count, 3), dtype=np.float64)
    posed_face_normals[face_valid] = sums[face_valid] / sum_length[face_valid, None]
    # Invalid rows receive a harmless nonzero placeholder and are discarded
    # after the vectorized pullback.
    _, posed_prior_normals = _face_frames(posed, triangles)
    posed_face_normals[~face_valid] = posed_prior_normals[~face_valid]
    canonical_face_normals = pull_back_face_normals(
        posed_face_normals,
        canonical,
        posed,
        triangles,
    )
    face_weight = np.clip(counts / 12.0, 0.0, 1.0) * concentration
    face_weight[~face_valid] = 0.0
    vertex_sum = np.zeros_like(canonical)
    vertex_weight = np.zeros(len(canonical), dtype=np.float64)
    vertex_contributions = np.zeros(len(canonical), dtype=np.int64)
    for corner in range(3):
        indices = triangles[:, corner]
        np.add.at(vertex_sum, indices, canonical_face_normals * face_weight[:, None])
        np.add.at(vertex_weight, indices, face_weight)
        np.add.at(vertex_contributions, indices, face_valid.astype(np.int64))
    vertex_valid = vertex_weight > 0.0
    vertex_normals = np.zeros_like(canonical)
    vertex_normals[vertex_valid] = _normalize(vertex_sum[vertex_valid])
    vertex_confidence = np.zeros(len(canonical), dtype=np.float64)
    vertex_confidence[vertex_valid] = np.clip(
        vertex_weight[vertex_valid] / np.maximum(vertex_contributions[vertex_valid], 1),
        0.0,
        1.0,
    )
    _, canonical_prior_face_normals = _face_frames(canonical, triangles)
    face_angles = _angular_error_degrees(
        canonical_face_normals[face_valid], canonical_prior_face_normals[face_valid]
    )
    return (
        vertex_normals,
        vertex_confidence,
        {
            "foreground_normal_pixel_count": int(np.count_nonzero(valid_pixels)),
            "valid_face_count": int(np.count_nonzero(face_valid)),
            "valid_vertex_count": int(np.count_nonzero(vertex_valid)),
            "median_face_observation_pixels": float(np.median(counts[face_valid])),
            "median_face_concentration": float(np.median(concentration[face_valid])),
            "median_pulled_normal_prior_degrees": float(np.median(face_angles)),
        },
    )


def _training_normal_mask_digest(records: list[Any], normal_root: Path, mask_root: Path) -> str:
    digest = hashlib.sha256()
    for record in records:
        name = Path(record.image_path).name
        digest.update(name.encode())
        digest.update(bytes.fromhex(sha256_file(normal_root / name)))
        digest.update(bytes.fromhex(sha256_file(mask_root / name)))
    return digest.hexdigest()


def bind_d01_train_only_normal_evidence(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    t05_solution_path: Path,
    q03_binding_path: Path,
    normal_root: Path,
    mask_root: Path,
    output_root: Path,
    source_revision: str,
) -> Path:
    """Create the immutable train-only canonical-normal evidence binding."""

    paths = [
        checkpoint_path,
        manifest_path,
        joint_transforms_path,
        t05_solution_path,
        q03_binding_path,
        normal_root,
        mask_root,
        output_root,
    ]
    reject_sealed_capability(paths)
    if output_root.exists():
        raise FileExistsError("D01 train evidence binding is immutable")
    if len(source_revision) != 40:
        raise ValueError("D01 source revision must be a full commit hash")
    (
        manifest,
        model,
        transforms,
        transform_lookup,
        trained_indices,
        trained_slot,
        initialization_intrinsics,
    ) = _load_inputs(config, checkpoint_path, manifest_path, joint_transforms_path)
    t05 = read_json(t05_solution_path)
    q03 = read_json(q03_binding_path)
    if t05.get("status") != "qualification_candidate" or t05.get("training_frame_count") != 144:
        raise ValueError("D01 requires the qualified 144-frame T05 solution")
    if q03.get("status") != "pass" or q03.get("binding", {}).get("accepted_track_count") != 249:
        raise ValueError("D01 requires the passing 249-track Q03 binding")
    train_records = [record for record in manifest.frames if record.split == "train"]
    if len(train_records) != 144:
        raise ValueError("D01 requires the complete frozen training split")
    if [record.source_frame_index for record in train_records] != [
        int(frame["source_frame_index"]) for frame in t05["frames"]
    ]:
        raise ValueError("D01 T05 and manifest training frames differ")
    intrinsics = np.asarray(t05["shared_intrinsics"], dtype=np.float64)
    if not np.allclose(intrinsics, initialization_intrinsics, atol=1.0e-6, rtol=0.0):
        raise ValueError("D01 T05 and initialization intrinsics differ")
    canonical = model.canonical_vertices.detach().cpu().numpy().astype(np.float64)
    faces = model.faces.detach().cpu().numpy().astype(np.int64)
    prior_vertex_normals = np.zeros_like(canonical)
    _, face_normals = _face_frames(canonical, faces)
    for corner in range(3):
        np.add.at(prior_vertex_normals, faces[:, corner], face_normals)
    prior_vertex_normals = _normalize(prior_vertex_normals)
    observations = np.broadcast_to(
        prior_vertex_normals[None], (len(train_records), *prior_vertex_normals.shape)
    ).copy()
    confidence = np.zeros(observations.shape[:-1], dtype=np.float64)
    frame_stats: list[dict[str, float | int]] = []
    source_size = (config.dataset.output_height, config.dataset.output_width)
    for slot, record in enumerate(train_records):
        name = Path(record.image_path).name
        normal_image = cv2.imread(str(normal_root / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
        if normal_image is None or mask is None:
            raise FileNotFoundError(f"D01 training normal/mask evidence is absent: {name}")
        vertex_normals, vertex_confidence, stats = sample_pose_stabilized_vertex_normals(
            canonical,
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
            normal_image,
            mask,
            source_size=source_size,
        )
        selected = vertex_confidence > 0.0
        observations[slot, selected] = vertex_normals[selected]
        confidence[slot] = vertex_confidence
        stats["source_frame_index"] = record.source_frame_index
        frame_stats.append(stats)
    fused, support = robust_fuse_normals(
        observations,
        confidence,
        reference=prior_vertex_normals,
    )
    observed = support > 0.0
    angular_delta = _angular_error_degrees(fused[observed], prior_vertex_normals[observed])
    view_counts = np.count_nonzero(confidence > 0.0, axis=0)
    output_root.mkdir(parents=True, exist_ok=False)
    binding_path = output_root / "train_pose_stabilized_normals.npz"
    temporary_path = output_root / ".train_pose_stabilized_normals.tmp"
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            observations=observations.astype(np.float32),
            confidence=confidence.astype(np.float32),
            fused_normals=fused.astype(np.float32),
            robust_support=support.astype(np.float32),
            prior_vertex_normals=prior_vertex_normals.astype(np.float32),
            source_frame_indices=np.asarray(
                [record.source_frame_index for record in train_records], dtype=np.int64
            ),
        )
    os.replace(temporary_path, binding_path)
    report = {
        "schema_version": "frayid_v2_d01_train_normal_binding.v1",
        "experiment_id": D01_EXPERIMENT_ID,
        "status": "train_only_evidence_bound",
        "source_revision": source_revision,
        "training_records_read": len(train_records),
        "development_records_read": 0,
        "sealed_test_reads": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "normal_convention": "sapiens_e25_rgb_xyz_then_camera_geometry_x_minus_y_minus_z",
        "depth_evidence": "none_scaffold_depth_reserved_for_prior_interval_only",
        "vertex_count": len(canonical),
        "observed_vertex_fraction": float(np.mean(observed)),
        "median_views_per_observed_vertex": float(np.median(view_counts[observed])),
        "minimum_views_per_observed_vertex": int(np.min(view_counts[observed])),
        "median_fused_normal_change_degrees": float(np.median(angular_delta)),
        "p95_fused_normal_change_degrees": float(np.percentile(angular_delta, 95.0)),
        "median_frame_valid_vertex_count": float(
            np.median([row["valid_vertex_count"] for row in frame_stats])
        ),
        "median_frame_pulled_normal_prior_degrees": float(
            np.median([row["median_pulled_normal_prior_degrees"] for row in frame_stats])
        ),
        "source_hashes": {
            "checkpoint": sha256_file(checkpoint_path),
            "manifest": sha256_file(manifest_path),
            "joint_transforms": sha256_file(joint_transforms_path),
            "t05_solution": sha256_file(t05_solution_path),
            "q03_binding": sha256_file(q03_binding_path),
            "train_normal_mask_evidence": _training_normal_mask_digest(
                train_records, normal_root, mask_root
            ),
        },
        "artifacts": {
            "train_pose_stabilized_normals": {
                "path": str(binding_path),
                "sha256": sha256_file(binding_path),
            }
        },
    }
    return write_json(output_root / "train_normal_binding_report.json", report)


D01_TRAIN_CANDIDATE_SCHEDULE: dict[str, float] = {
    "maximum_normal_displacement_metres": 0.035,
    "normal_equation_weight": 1.0,
    "prior_anchor_weight": 0.08,
    "edge_smoothness_weight": 0.04,
    "minimum_face_area_ratio": 0.10,
}
D01_TRAIN_CANDIDATE_GATES: dict[str, float] = {
    "median_normal_improvement_degrees_minimum": 1.0,
    "median_hard_iou_regression_maximum": 0.005,
    "median_boundary_error_regression_maximum": 0.001,
}


def write_d01_train_candidate_plan(
    evidence_report_path: Path,
    output_path: Path,
    *,
    source_revision: str,
) -> Path:
    """Freeze the train-only candidate schedule and evaluator before fitting."""

    reject_sealed_capability([evidence_report_path, output_path])
    if output_path.exists():
        raise FileExistsError("D01 train candidate plan is immutable")
    evidence = read_json(evidence_report_path)
    if (
        evidence.get("status") != "train_only_evidence_bound"
        or evidence.get("development_records_read") != 0
        or evidence.get("sealed_test_reads") != 0
    ):
        raise ValueError("D01 candidate plan requires a clean train-only evidence binding")
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_d01_train_candidate_plan.v1",
            "experiment_id": D01_EXPERIMENT_ID,
            "status": "frozen_before_candidate_fit",
            "source_revision": source_revision,
            "evidence_report_sha256": sha256_file(evidence_report_path),
            "schedule": D01_TRAIN_CANDIDATE_SCHEDULE,
            "training_gates": D01_TRAIN_CANDIDATE_GATES,
            "depth_role": "prior_derived_signed_normal_displacement_interval_not_measurement",
            "candidate_parameterization": "one_scalar_displacement_per_frozen_v1_vertex_along_frozen_v1_vertex_normal",
            "connectivity_policy": "frozen_exactly",
            "development_records_authorized": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "automatic_retries": 0,
        },
    )


def integrate_mesh_normals_along_prior(
    prior_vertices: np.ndarray,
    faces: np.ndarray,
    prior_vertex_normals: np.ndarray,
    target_vertex_normals: np.ndarray,
    robust_support: np.ndarray,
    *,
    maximum_displacement_metres: float,
    normal_equation_weight: float,
    prior_anchor_weight: float,
    edge_smoothness_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a bounded scalar normal-displacement field on fixed connectivity."""

    vertices = np.asarray(prior_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    prior_normals = _normalize(prior_vertex_normals)
    target_normals = _normalize(target_vertex_normals)
    support = np.asarray(robust_support, dtype=np.float64)
    if (
        vertices.shape != prior_normals.shape
        or vertices.shape != target_normals.shape
        or support.shape != (len(vertices),)
    ):
        raise ValueError("D01 mesh-normal integration inputs are not aligned")
    parameters = (
        maximum_displacement_metres,
        normal_equation_weight,
        prior_anchor_weight,
        edge_smoothness_weight,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in parameters):
        raise ValueError("D01 mesh-normal integration parameters must be positive")
    if np.any(support < 0.0):
        raise ValueError("D01 robust support must be nonnegative")
    vertex_count = len(vertices)
    face_targets = _normalize(target_normals[triangles].sum(axis=1))
    face_support = np.min(support[triangles], axis=1)
    face_weight = normal_equation_weight * np.sqrt(np.clip(face_support / 4.0, 0.0, 1.0))
    active_faces = face_weight > 0.0
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    targets: list[float] = []
    row_index = 0
    for edge_corner in (1, 2):
        first = triangles[active_faces, 0]
        second = triangles[active_faces, edge_corner]
        desired = face_targets[active_faces]
        weights = face_weight[active_faces]
        prior_edge = vertices[second] - vertices[first]
        first_coefficient = -np.sum(prior_normals[first] * desired, axis=-1) * weights
        second_coefficient = np.sum(prior_normals[second] * desired, axis=-1) * weights
        right_hand_side = -np.sum(prior_edge * desired, axis=-1) * weights
        count = len(first)
        row_ids = np.arange(row_index, row_index + count, dtype=np.int64)
        rows.extend(np.repeat(row_ids, 2).tolist())
        columns.extend(np.column_stack((first, second)).reshape(-1).tolist())
        data.extend(np.column_stack((first_coefficient, second_coefficient)).reshape(-1).tolist())
        targets.extend(right_hand_side.tolist())
        row_index += count
    normal_matrix = coo_matrix((data, (rows, columns)), shape=(row_index, vertex_count)).tocsr()
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    smooth_scale = math.sqrt(edge_smoothness_weight)
    smooth_rows = np.repeat(np.arange(len(edges)), 2)
    smooth_columns = edges.reshape(-1)
    smooth_data = np.tile(np.asarray([-smooth_scale, smooth_scale]), len(edges))
    smooth_matrix = coo_matrix(
        (smooth_data, (smooth_rows, smooth_columns)),
        shape=(len(edges), vertex_count),
    ).tocsr()
    anchor_scale = math.sqrt(prior_anchor_weight)
    anchor_matrix = coo_matrix(
        (
            np.full(vertex_count, anchor_scale),
            (np.arange(vertex_count), np.arange(vertex_count)),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    system = vstack((normal_matrix, smooth_matrix, anchor_matrix), format="csr")
    target = np.concatenate(
        (np.asarray(targets), np.zeros(len(edges) + vertex_count, dtype=np.float64))
    )
    displacement = np.asarray(
        lsqr(system, target, atol=1.0e-10, btol=1.0e-10, iter_lim=6000)[0],
        dtype=np.float64,
    )
    displacement = np.clip(
        displacement,
        -maximum_displacement_metres,
        maximum_displacement_metres,
    )
    candidate = vertices + displacement[:, None] * prior_normals
    return candidate, displacement


def fit_d01_train_only_mesh_candidate(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    evidence_binding_path: Path,
    evidence_report_path: Path,
    candidate_plan_path: Path,
    output_root: Path,
    source_revision: str,
) -> Path:
    """Fit one deterministic bounded candidate without reading development data."""

    paths = [
        checkpoint_path,
        manifest_path,
        joint_transforms_path,
        evidence_binding_path,
        evidence_report_path,
        candidate_plan_path,
        output_root,
    ]
    reject_sealed_capability(paths)
    if output_root.exists():
        raise FileExistsError("D01 train mesh candidate output is immutable")
    evidence_report = read_json(evidence_report_path)
    plan = read_json(candidate_plan_path)
    if (
        plan.get("status") != "frozen_before_candidate_fit"
        or plan.get("evidence_report_sha256") != sha256_file(evidence_report_path)
        or plan.get("schedule") != D01_TRAIN_CANDIDATE_SCHEDULE
        or plan.get("training_gates") != D01_TRAIN_CANDIDATE_GATES
    ):
        raise ValueError("D01 candidate rejected an altered or stale frozen plan")
    if (
        evidence_report.get("development_records_read") != 0
        or evidence_report.get("sealed_test_reads") != 0
        or evidence_report.get("artifacts", {})
        .get("train_pose_stabilized_normals", {})
        .get("sha256")
        != sha256_file(evidence_binding_path)
    ):
        raise ValueError("D01 candidate rejected its train evidence provenance")
    (
        manifest,
        model,
        _transforms,
        _transform_lookup,
        _trained_indices,
        _trained_slot,
        _intrinsics,
    ) = _load_inputs(config, checkpoint_path, manifest_path, joint_transforms_path)
    if sum(record.split == "train" for record in manifest.frames) != 144:
        raise ValueError("D01 candidate requires the frozen 144-frame training split")
    with np.load(evidence_binding_path, allow_pickle=False) as archive:
        fused = archive["fused_normals"].astype(np.float64)
        support = archive["robust_support"].astype(np.float64)
        prior_normals = archive["prior_vertex_normals"].astype(np.float64)
    prior = model.canonical_vertices.detach().cpu().numpy().astype(np.float64)
    faces = model.faces.detach().cpu().numpy().astype(np.int64)
    candidate, displacement = integrate_mesh_normals_along_prior(
        prior,
        faces,
        prior_normals,
        fused,
        support,
        maximum_displacement_metres=D01_TRAIN_CANDIDATE_SCHEDULE[
            "maximum_normal_displacement_metres"
        ],
        normal_equation_weight=D01_TRAIN_CANDIDATE_SCHEDULE["normal_equation_weight"],
        prior_anchor_weight=D01_TRAIN_CANDIDATE_SCHEDULE["prior_anchor_weight"],
        edge_smoothness_weight=D01_TRAIN_CANDIDATE_SCHEDULE["edge_smoothness_weight"],
    )
    replay_candidate, replay_displacement = integrate_mesh_normals_along_prior(
        prior,
        faces,
        prior_normals,
        fused,
        support,
        maximum_displacement_metres=D01_TRAIN_CANDIDATE_SCHEDULE[
            "maximum_normal_displacement_metres"
        ],
        normal_equation_weight=D01_TRAIN_CANDIDATE_SCHEDULE["normal_equation_weight"],
        prior_anchor_weight=D01_TRAIN_CANDIDATE_SCHEDULE["prior_anchor_weight"],
        edge_smoothness_weight=D01_TRAIN_CANDIDATE_SCHEDULE["edge_smoothness_weight"],
    )
    exact_replay = np.array_equal(candidate, replay_candidate) and np.array_equal(
        displacement, replay_displacement
    )
    topology = canonical_face_orientation_report(
        prior,
        candidate,
        faces,
        minimum_area_ratio=D01_TRAIN_CANDIDATE_SCHEDULE["minimum_face_area_ratio"],
    )
    blockers: list[str] = []
    if not exact_replay:
        blockers.append("candidate_solve_replay")
    if topology["status"] != "pass":
        blockers.extend(str(value) for value in topology["blockers"])
    output_root.mkdir(parents=True, exist_ok=False)
    candidate_path = output_root / "bounded_canonical_mesh_candidate.npz"
    temporary_path = output_root / ".bounded_canonical_mesh_candidate.tmp"
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            vertices=candidate.astype(np.float32),
            faces=faces,
            signed_normal_displacement_metres=displacement.astype(np.float32),
        )
    os.replace(temporary_path, candidate_path)
    report = {
        "schema_version": "frayid_v2_d01_train_mesh_candidate.v1",
        "experiment_id": D01_EXPERIMENT_ID,
        "status": "candidate_complete" if not blockers else "candidate_failed_precheck",
        "source_revision": source_revision,
        "training_records_bound": 144,
        "development_records_read": 0,
        "sealed_test_reads": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "schedule": D01_TRAIN_CANDIDATE_SCHEDULE,
        "exact_solve_replay": exact_replay,
        "connectivity_exactly_frozen": bool(np.array_equal(faces, model.faces.cpu().numpy())),
        "displacement_metres": {
            "minimum": float(displacement.min()),
            "median_absolute": float(np.median(np.abs(displacement))),
            "p95_absolute": float(np.percentile(np.abs(displacement), 95.0)),
            "maximum": float(displacement.max()),
            "clamped_vertex_fraction": float(
                np.mean(
                    np.abs(displacement)
                    >= D01_TRAIN_CANDIDATE_SCHEDULE["maximum_normal_displacement_metres"] - 1.0e-12
                )
            ),
        },
        "topology_precheck": topology,
        "blockers": blockers,
        "input_hashes": {
            "checkpoint": sha256_file(checkpoint_path),
            "manifest": sha256_file(manifest_path),
            "joint_transforms": sha256_file(joint_transforms_path),
            "evidence_binding": sha256_file(evidence_binding_path),
            "evidence_report": sha256_file(evidence_report_path),
            "candidate_plan": sha256_file(candidate_plan_path),
        },
        "artifacts": {
            "bounded_canonical_mesh_candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            }
        },
    }
    return write_json(output_root / "train_mesh_candidate_report.json", report)


def _render_hard_geometry_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    face_ids, _ = painter_visibility(
        vertices,
        faces,
        intrinsics,
        source_size=source_size,
        output_size=source_size,
    )
    _, face_normals = _face_frames(vertices, faces)
    normal_map = np.zeros((*source_size, 3), dtype=np.float64)
    foreground = face_ids >= 0
    normal_map[foreground] = face_normals[face_ids[foreground]]
    return foreground, normal_map


def _hard_mask_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    first = np.asarray(predicted, dtype=bool)
    second = np.asarray(target, dtype=bool)
    union = int(np.count_nonzero(first | second))
    intersection = int(np.count_nonzero(first & second))
    iou = float(intersection / max(union, 1))
    kernel = np.ones((3, 3), dtype=np.uint8)
    first_edge = first & ~cv2.erode(first.astype(np.uint8), kernel).astype(bool)
    second_edge = second & ~cv2.erode(second.astype(np.uint8), kernel).astype(bool)
    first_distance = cv2.distanceTransform((~first_edge).astype(np.uint8), cv2.DIST_L2, 3)
    second_distance = cv2.distanceTransform((~second_edge).astype(np.uint8), cv2.DIST_L2, 3)
    distances = []
    if np.any(first_edge):
        distances.append(float(np.mean(second_distance[first_edge])))
    if np.any(second_edge):
        distances.append(float(np.mean(first_distance[second_edge])))
    boundary = float(np.mean(distances) / math.hypot(*first.shape)) if distances else 1.0
    return iou, boundary


def evaluate_d01_train_mesh_candidate(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    t05_solution_path: Path,
    candidate_path: Path,
    candidate_report_path: Path,
    candidate_plan_path: Path,
    normal_root: Path,
    mask_root: Path,
    output_path: Path,
    source_revision: str,
    experiment_id: str = D01_EXPERIMENT_ID,
    report_schema_version: str = "frayid_v2_d01_train_candidate_evaluation.v1",
    training_gates: dict[str, float] | None = None,
) -> Path:
    """Independently compare candidate and control on training evidence only."""

    paths = [
        checkpoint_path,
        manifest_path,
        joint_transforms_path,
        t05_solution_path,
        candidate_path,
        candidate_report_path,
        candidate_plan_path,
        normal_root,
        mask_root,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("D01 train candidate evaluation is immutable")
    candidate_report = read_json(candidate_report_path)
    plan = read_json(candidate_plan_path)
    frozen_gates = D01_TRAIN_CANDIDATE_GATES if training_gates is None else training_gates
    if (
        candidate_report.get("status") != "candidate_complete"
        or candidate_report.get("development_records_read") != 0
        or candidate_report.get("artifacts", {})
        .get("bounded_canonical_mesh_candidate", {})
        .get("sha256")
        != sha256_file(candidate_path)
        or plan.get("training_gates") != frozen_gates
    ):
        raise ValueError("D01 evaluator rejected its candidate or plan binding")
    (
        manifest,
        control_model,
        transforms,
        transform_lookup,
        trained_indices,
        trained_slot,
        _initialization_intrinsics,
    ) = _load_inputs(config, checkpoint_path, manifest_path, joint_transforms_path)
    t05 = read_json(t05_solution_path)
    intrinsics = np.asarray(t05["shared_intrinsics"], dtype=np.float64)
    with np.load(candidate_path, allow_pickle=False) as archive:
        candidate_vertices = archive["vertices"].astype(np.float32)
        candidate_faces = archive["faces"].astype(np.int64)
    if not np.array_equal(candidate_faces, control_model.faces.cpu().numpy()):
        raise ValueError("D01 candidate connectivity differs from its control")
    candidate_model = deepcopy(control_model)
    with torch.no_grad():
        candidate_model.canonical_offsets.copy_(
            torch.from_numpy(candidate_vertices) - candidate_model.base_vertices.cpu()
        )
    train_records = [record for record in manifest.frames if record.split == "train"]
    source_size = (config.dataset.output_height, config.dataset.output_width)
    faces = candidate_faces
    control_ious: list[float] = []
    treatment_ious: list[float] = []
    control_boundaries: list[float] = []
    treatment_boundaries: list[float] = []
    control_normal_degrees: list[float] = []
    treatment_normal_degrees: list[float] = []
    for record in train_records:
        name = Path(record.image_path).name
        normal_image = cv2.imread(str(normal_root / name), cv2.IMREAD_COLOR)
        mask_image = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
        if normal_image is None or mask_image is None:
            raise FileNotFoundError(f"D01 evaluator lacks training evidence: {name}")
        target_mask = mask_image > 127
        target_normal = decode_sapiens_normal_bgr(normal_image)
        control_vertices = _posed(
            record,
            control_model,
            transforms,
            transform_lookup,
            trained_indices,
            trained_slot,
        )
        treatment_vertices = _posed(
            record,
            candidate_model,
            transforms,
            transform_lookup,
            trained_indices,
            trained_slot,
        )
        control_mask, control_normal = _render_hard_geometry_normals(
            control_vertices, faces, intrinsics, source_size
        )
        treatment_mask, treatment_normal = _render_hard_geometry_normals(
            treatment_vertices, faces, intrinsics, source_size
        )
        control_iou, control_boundary = _hard_mask_metrics(control_mask, target_mask)
        treatment_iou, treatment_boundary = _hard_mask_metrics(treatment_mask, target_mask)
        control_ious.append(control_iou)
        treatment_ious.append(treatment_iou)
        control_boundaries.append(control_boundary)
        treatment_boundaries.append(treatment_boundary)
        eroded = cv2.erode(target_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)).astype(
            bool
        )
        valid_control = eroded & control_mask & (np.linalg.norm(target_normal, axis=-1) > 0.5)
        valid_treatment = eroded & treatment_mask & (np.linalg.norm(target_normal, axis=-1) > 0.5)
        control_normal_degrees.append(
            float(
                np.median(
                    _angular_error_degrees(
                        control_normal[valid_control], target_normal[valid_control]
                    )
                )
            )
        )
        treatment_normal_degrees.append(
            float(
                np.median(
                    _angular_error_degrees(
                        treatment_normal[valid_treatment], target_normal[valid_treatment]
                    )
                )
            )
        )
    metrics = {
        "control_median_hard_iou": float(np.median(control_ious)),
        "treatment_median_hard_iou": float(np.median(treatment_ious)),
        "control_median_boundary_error": float(np.median(control_boundaries)),
        "treatment_median_boundary_error": float(np.median(treatment_boundaries)),
        "control_median_normal_degrees": float(np.median(control_normal_degrees)),
        "treatment_median_normal_degrees": float(np.median(treatment_normal_degrees)),
    }
    normal_improvement = (
        metrics["control_median_normal_degrees"] - metrics["treatment_median_normal_degrees"]
    )
    iou_regression = metrics["control_median_hard_iou"] - metrics["treatment_median_hard_iou"]
    boundary_regression = (
        metrics["treatment_median_boundary_error"] - metrics["control_median_boundary_error"]
    )
    gates = {
        "median_normal_improvement": normal_improvement
        >= frozen_gates["median_normal_improvement_degrees_minimum"],
        "median_hard_iou_nonregression": iou_regression
        <= frozen_gates["median_hard_iou_regression_maximum"],
        "median_boundary_nonregression": boundary_regression
        <= frozen_gates["median_boundary_error_regression_maximum"],
        "candidate_topology_precheck": candidate_report["topology_precheck"]["status"] == "pass",
    }
    return write_json(
        output_path,
        {
            "schema_version": report_schema_version,
            "experiment_id": experiment_id,
            "status": "pass" if all(gates.values()) else "fail",
            "source_revision": source_revision,
            "training_records_read": len(train_records),
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "metrics": metrics,
            "normal_improvement_degrees": normal_improvement,
            "hard_iou_regression": iou_regression,
            "boundary_error_regression": boundary_regression,
            "gates": gates,
            "input_hashes": {
                "checkpoint": sha256_file(checkpoint_path),
                "manifest": sha256_file(manifest_path),
                "joint_transforms": sha256_file(joint_transforms_path),
                "t05_solution": sha256_file(t05_solution_path),
                "candidate": sha256_file(candidate_path),
                "candidate_report": sha256_file(candidate_report_path),
                "candidate_plan": sha256_file(candidate_plan_path),
            },
        },
    )


def audit_d01_terminal_qualification(
    public_benchmark_path: Path,
    evidence_report_path: Path,
    candidate_plan_path: Path,
    candidate_report_path: Path,
    output_path: Path,
) -> Path:
    """Close D01 after its frozen candidate fails the topology precheck."""

    paths = [
        public_benchmark_path,
        evidence_report_path,
        candidate_plan_path,
        candidate_report_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("D01 terminal qualification audit is immutable")
    public = read_json(public_benchmark_path)
    evidence = read_json(evidence_report_path)
    plan = read_json(candidate_plan_path)
    candidate = read_json(candidate_report_path)
    checks = {
        "public_benchmark_passed": public.get("status") == "pass",
        "public_had_no_private_reads": public.get("provenance", {}).get("private_records_read")
        == 0,
        "train_evidence_bound": evidence.get("status") == "train_only_evidence_bound",
        "complete_training_split_bound": evidence.get("training_records_read") == 144,
        "candidate_plan_frozen_before_fit": plan.get("status") == "frozen_before_candidate_fit",
        "candidate_failed_precheck": candidate.get("status") == "candidate_failed_precheck",
        "candidate_replay_exact": candidate.get("exact_solve_replay") is True,
        "connectivity_frozen": candidate.get("connectivity_exactly_frozen") is True,
        "face_flip_detected": candidate.get("topology_precheck", {}).get("flipped_face_count", 0)
        > 0,
        "face_collapse_detected": candidate.get("topology_precheck", {}).get(
            "collapsed_face_count", 0
        )
        > 0,
        "training_evaluation_not_run": not candidate_report_path.with_name(
            "train_candidate_evaluation.json"
        ).exists(),
        "development_reads_zero": evidence.get("development_records_read") == 0
        and candidate.get("development_records_read") == 0,
        "sealed_reads_zero": evidence.get("sealed_test_reads") == 0
        and candidate.get("sealed_test_reads") == 0,
        "optimizer_steps_zero": candidate.get("optimizer_steps") == 0,
        "paid_jobs_zero": candidate.get("paid_jobs") == 0,
    }
    blockers = (
        [] if all(checks.values()) else [name for name, passed in checks.items() if not passed]
    )
    if blockers:
        raise ValueError("D01 terminal audit inputs are inconsistent: " + ",".join(blockers))
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_d01_terminal_qualification.v1",
            "experiment_id": D01_EXPERIMENT_ID,
            "status": "pass",
            "qualification_state": "unbuilt",
            "decision": "terminal_failed_train_topology_precheck",
            "checks": checks,
            "scientific_attempt_marker_created": False,
            "scientific_attempts_started": 0,
            "training_evaluation_run": False,
            "development_reads": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "topology_failure": candidate["topology_precheck"],
            "artifact_hashes": {
                "public_benchmark": sha256_file(public_benchmark_path),
                "train_evidence_report": sha256_file(evidence_report_path),
                "candidate_plan": sha256_file(candidate_plan_path),
                "candidate_report": sha256_file(candidate_report_path),
            },
            "next_rule": "D01_must_not_be_retuned_or_retried_register_a_materially_new_topology_constrained_successor",
        },
    )


def integrate_normals_in_prior_interval(
    normals: np.ndarray,
    support: np.ndarray,
    prior_height: np.ndarray,
    half_width: np.ndarray,
    spacing: tuple[float, float],
    *,
    prior_weight: float = 0.025,
) -> np.ndarray:
    """Integrate normals while using depth only as a bounded scaffold prior.

    No measured depth enters this solve. The prior supplies the otherwise
    unobservable global height gauge and a per-sample admissible interval.
    """

    unit = _normalize(normals)
    weights = np.asarray(support, dtype=np.float64)
    prior = np.asarray(prior_height, dtype=np.float64)
    widths = np.asarray(half_width, dtype=np.float64)
    if unit.shape != (*prior.shape, 3) or weights.shape != prior.shape:
        raise ValueError("normal-integration inputs are not aligned")
    if widths.shape != prior.shape or np.any(widths <= 0.0):
        raise ValueError("every prior-derived depth interval must have positive width")
    if prior_weight <= 0.0 or np.any(weights <= 0.0):
        raise ValueError("integration weights must be positive")
    spacing_y, spacing_x = spacing
    height, width = prior.shape
    count = height * width
    safe_z = np.maximum(unit[..., 2], 0.15)
    gradient_x = -unit[..., 0] / safe_z
    gradient_y = -unit[..., 1] / safe_z

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    targets: list[float] = []
    row_index = 0
    for y in range(height):
        for x in range(width - 1):
            edge_weight = math.sqrt(float(min(weights[y, x], weights[y, x + 1])))
            rows.extend((row_index, row_index))
            columns.extend((y * width + x, y * width + x + 1))
            data.extend((-edge_weight / spacing_x, edge_weight / spacing_x))
            targets.append(
                edge_weight * 0.5 * (float(gradient_x[y, x]) + float(gradient_x[y, x + 1]))
            )
            row_index += 1
    for y in range(height - 1):
        for x in range(width):
            edge_weight = math.sqrt(float(min(weights[y, x], weights[y + 1, x])))
            rows.extend((row_index, row_index))
            columns.extend((y * width + x, (y + 1) * width + x))
            data.extend((-edge_weight / spacing_y, edge_weight / spacing_y))
            targets.append(
                edge_weight * 0.5 * (float(gradient_y[y, x]) + float(gradient_y[y + 1, x]))
            )
            row_index += 1
    gradient_matrix = coo_matrix((data, (rows, columns)), shape=(row_index, count)).tocsr()
    anchor_scale = math.sqrt(prior_weight)
    anchor_matrix = coo_matrix(
        (
            np.full(count, anchor_scale, dtype=np.float64),
            (np.arange(count), np.arange(count)),
        ),
        shape=(count, count),
    ).tocsr()
    system = vstack((gradient_matrix, anchor_matrix), format="csr")
    target = np.concatenate((np.asarray(targets), anchor_scale * prior.reshape(-1)))
    solution = lsqr(system, target, atol=1.0e-10, btol=1.0e-10, iter_lim=4000)[0]
    unconstrained = np.asarray(solution, dtype=np.float64).reshape(prior.shape)
    return np.clip(unconstrained, prior - widths, prior + widths)


def _rotation_y(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _angular_error_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine = np.clip(np.sum(_normalize(first) * _normalize(second), axis=-1), -1.0, 1.0)
    return np.asarray(np.degrees(np.arccos(cosine)), dtype=np.float64)


def _surface_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    true_normals: np.ndarray,
    spacing: tuple[float, float],
) -> dict[str, float]:
    error = np.asarray(estimate) - np.asarray(truth)
    estimated_normals = normals_from_height(estimate, spacing)
    interior = np.s_[1:-1, 1:-1]
    return {
        "position_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "point_to_plane_rmse": float(np.sqrt(np.mean(np.square(error * true_normals[..., 2])))),
        "median_normal_degrees": float(
            np.median(_angular_error_degrees(estimated_normals[interior], true_normals[interior]))
        ),
        "maximum_interval_violation": 0.0,
    }


def _transport_finite_difference_audit() -> dict[str, float | bool]:
    gradient_x = 0.23
    gradient_y = -0.17
    tangent_x = np.asarray([1.0, 0.0, gradient_x])
    tangent_y = np.asarray([0.0, 1.0, gradient_y])
    normal = _normalize(np.asarray([-gradient_x, -gradient_y, 1.0]))
    jacobian = _rotation_y(math.radians(37.0)) @ np.asarray(
        [[1.07, 0.03, 0.02], [0.0, 0.94, -0.04], [0.01, 0.02, 1.03]]
    )
    posed = pose_normals_from_canonical(normal, jacobian)
    recovered = transport_normals_to_canonical(posed, jacobian)
    orthogonality = max(
        abs(float(np.dot(posed, jacobian @ tangent_x))),
        abs(float(np.dot(posed, jacobian @ tangent_y))),
    )
    round_trip = float(np.linalg.norm(recovered - normal))
    return {
        "maximum_posed_tangent_dot": orthogonality,
        "canonical_round_trip_error": round_trip,
        "pass": orthogonality <= 1.0e-10 and round_trip <= 1.0e-10,
    }


def run_d01_public_benchmark(*, seed: int = 20260903) -> dict[str, Any]:
    """Evaluate pose-stabilized fusion without private or generated-view evidence."""

    generator = np.random.default_rng(seed)
    height, width = 25, 31
    y = np.linspace(-0.8, 0.8, height)
    x = np.linspace(-1.0, 1.0, width)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    truth = (
        0.09 * np.sin(2.1 * math.pi * xx) * np.cos(1.7 * math.pi * yy)
        + 0.055 * np.exp(-((xx - 0.28) ** 2 + (yy + 0.12) ** 2) / 0.035)
        - 0.045 * np.exp(-((xx + 0.37) ** 2 + (yy - 0.18) ** 2) / 0.025)
        + 0.018 * xx
    )
    spacing = (float(y[1] - y[0]), float(x[1] - x[0]))
    true_normals = normals_from_height(truth, spacing)
    prior = gaussian_filter(truth, sigma=2.4) + 0.018 * np.cos(math.pi * yy)
    interval_half_width = np.full_like(prior, 0.14)
    prior_normals = normals_from_height(prior, spacing)

    frame_angles = np.radians(np.asarray([-62.0, -28.0, 8.0, 39.0, 73.0]))
    jacobians = []
    clean_observations = []
    for frame_index, angle in enumerate(frame_angles):
        local = np.asarray(
            [
                [1.0 + 0.025 * math.sin(frame_index), 0.018, 0.012],
                [0.0, 0.98 + 0.015 * math.cos(frame_index), -0.018],
                [0.012, 0.014, 1.02],
            ]
        )
        jacobian = _rotation_y(float(angle)) @ local
        tiled = np.broadcast_to(jacobian, (*true_normals.shape[:-1], 3, 3)).copy()
        posed = pose_normals_from_canonical(true_normals, tiled)
        noise = generator.normal(0.0, 0.006, size=posed.shape)
        clean_observations.append(_normalize(posed + noise))
        jacobians.append(tiled)
    clean = np.stack(clean_observations)
    transforms = np.stack(jacobians)
    confidence = np.ones(clean.shape[:-1], dtype=np.float64)

    corrupted = clean.copy()
    corrupt_mask = generator.random((height, width)) < 0.28
    corrupted[1, corrupt_mask] = _normalize(generator.normal(size=(int(corrupt_mask.sum()), 3)))
    corrupted_confidence = confidence.copy()
    corrupted_confidence[1, corrupt_mask] = 0.35

    canonical_clean = transport_normals_to_canonical(clean, transforms)
    fused_clean, clean_support = robust_fuse_normals(
        canonical_clean, confidence, reference=prior_normals
    )
    clean_height = integrate_normals_in_prior_interval(
        fused_clean,
        clean_support,
        prior,
        interval_half_width,
        spacing,
    )
    canonical_corrupt = transport_normals_to_canonical(corrupted, transforms)
    fused_treatment, treatment_support = robust_fuse_normals(
        canonical_corrupt, corrupted_confidence, reference=prior_normals
    )
    treatment_height = integrate_normals_in_prior_interval(
        fused_treatment,
        treatment_support,
        prior,
        interval_half_width,
        spacing,
    )
    fused_control, control_support = robust_fuse_normals(
        corrupted, corrupted_confidence, reference=prior_normals
    )
    control_height = integrate_normals_in_prior_interval(
        fused_control,
        control_support,
        prior,
        interval_half_width,
        spacing,
    )

    treatment = _surface_metrics(treatment_height, truth, true_normals, spacing)
    control = _surface_metrics(control_height, truth, true_normals, spacing)
    clean_metrics = _surface_metrics(clean_height, truth, true_normals, spacing)
    prior_metrics = _surface_metrics(prior, truth, true_normals, spacing)
    position_improvement = 1.0 - treatment["position_rmse"] / control["position_rmse"]
    plane_improvement = 1.0 - treatment["point_to_plane_rmse"] / control["point_to_plane_rmse"]
    corruption_regression = treatment["position_rmse"] - clean_metrics["position_rmse"]
    biased_height = np.clip(
        treatment_height + 0.08,
        prior - interval_half_width,
        prior + interval_half_width,
    )
    biased_metrics = _surface_metrics(biased_height, truth, true_normals, spacing)
    transport_audit = _transport_finite_difference_audit()
    gates = {
        "position_rmse_improvement": position_improvement >= 0.20,
        "point_to_plane_rmse_improvement": plane_improvement >= 0.20,
        "median_normal_degrees": treatment["median_normal_degrees"] <= 5.0,
        "corrupted_observation_regression": corruption_regression <= 0.01,
        "global_bias_shortcut_rejected": (
            biased_metrics["position_rmse"] > treatment["position_rmse"] * 1.5
            and abs(biased_metrics["median_normal_degrees"] - treatment["median_normal_degrees"])
            <= 1.0e-10
        ),
        "inverse_pose_transport": bool(transport_audit["pass"]),
        "truth_inside_every_prior_interval": bool(
            np.all(truth >= prior - interval_half_width)
            and np.all(truth <= prior + interval_half_width)
        ),
    }
    return {
        "schema_version": D01_PUBLIC_SCHEMA,
        "experiment_id": D01_EXPERIMENT_ID,
        "status": "pass" if all(gates.values()) else "fail",
        "seed": seed,
        "fixture": {
            "height": height,
            "width": width,
            "view_count": len(frame_angles),
            "corrupted_view_index": 1,
            "corrupted_sample_fraction": float(np.mean(corrupt_mask)),
            "depth_source": "analytic_scaffold_prior_interval_not_measured_depth",
            "generated_views_used_as_project_evidence": False,
        },
        "treatment": treatment,
        "matched_frame_space_control": control,
        "uncorrupted_pose_stabilized_treatment": clean_metrics,
        "scaffold_prior": prior_metrics,
        "global_bias_shortcut": biased_metrics,
        "position_rmse_relative_improvement": position_improvement,
        "point_to_plane_rmse_relative_improvement": plane_improvement,
        "corrupted_observation_position_regression": corruption_regression,
        "transport_audit": transport_audit,
        "gates": gates,
        "provenance": {
            "private_records_read": 0,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "measured_depth_claimed": False,
        },
    }


def write_d01_public_benchmark(output_path: Path, *, seed: int = 20260903) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("D01 public benchmark output is immutable")
    return write_json(output_path, run_d01_public_benchmark(seed=seed))
