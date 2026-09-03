from __future__ import annotations

import hashlib
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from skimage.measure import marching_cubes

from frayid.dataset import read_dataset_manifest
from frayid.geometry import linear_blend_skinning
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.d01_pose_normal_depth import (
    _hard_mask_metrics,
    _render_hard_geometry_normals,
    decode_sapiens_normal_bgr,
    integrate_mesh_normals_along_prior,
    robust_fuse_normals,
    sample_pose_stabilized_vertex_normals,
)
from frayid.v2.d02_topology_projection import (
    exact_face_constraint_audit,
    ipctk_has_self_intersections,
    topology_constrained_local_trust_projection,
)

D03_EXPERIMENT_ID = "postv2_d03_capsule_tree_implicit_body_r01"
D03_PUBLIC_SCHEMA = "frayid_v2_d03_public_capsule_tree_implicit_body.v1"
D03_REAL_PLAN_SCHEMA = "frayid_v2_d03_real_initialization_plan.v1"
D03_REAL_INITIALIZATION_SCHEMA = "frayid_v2_d03_real_initialization.v1"
D03_TRAIN_EVIDENCE_PLAN_SCHEMA = "frayid_v2_d03_train_evidence_plan.v1"
D03_TRAIN_EVIDENCE_SCHEMA = "frayid_v2_d03_train_evidence.v1"
D03_CONTINUATION_PLAN_SCHEMA = "frayid_v2_d03_implicit_continuation_plan.v1"
D03_CONTINUATION_SCHEMA = "frayid_v2_d03_implicit_continuation.v1"
D03_TRAIN_EVALUATION_PLAN_SCHEMA = "frayid_v2_d03_train_evaluation_plan.v1"
D03_DEVELOPMENT_PLAN_SCHEMA = "frayid_v2_d03_development_evaluation_plan.v1"

D03_CONTINUATION_SCHEDULE: dict[str, float | int] = {
    "maximum_normal_displacement_metres": 0.015,
    "normal_equation_weight": 1.0,
    "prior_anchor_weight": 0.20,
    "edge_smoothness_weight": 0.10,
    "minimum_mesh_area_ratio": 0.25,
    "maximum_local_projection_rejections": 48,
    "local_projection_shrink_factor": 0.5,
    "minimum_local_vertex_scale": 2.0**-24,
    "field_interpolation_neighbours": 8,
    "field_interpolation_epsilon_metres": 0.005,
    "field_update_band_metres": 0.12,
}
D03_TRAIN_EVALUATION_GATES: dict[str, float] = {
    "median_normal_improvement_degrees_minimum": 1.0,
    "median_hard_iou_regression_maximum": 0.005,
    "median_boundary_error_regression_maximum": 0.001,
}
D03_DEVELOPMENT_GATES: dict[str, float] = {
    "held_out_iou_minimum": 0.8778329789638519,
    "boundary_error_maximum": 0.004556156724774018,
    "median_normal_degrees_maximum": 21.374711990356445,
    "train_held_out_gap_maximum": 0.05,
    "relative_iou_regression_maximum": 0.005,
    "relative_boundary_regression_maximum": 0.001,
    "relative_normal_regression_degrees_maximum": 0.0,
}

_REAL_CAPSULE_SPECS = (
    ("torso", 0, 12, 0.105, 0.190),
    ("head", 12, 15, 0.100, 0.150),
    ("left_upper_arm", 13, 16, 0.055, 0.105),
    ("left_arm", 16, 18, 0.050, 0.090),
    ("left_forearm", 18, 20, 0.045, 0.080),
    ("left_hand", 20, 22, 0.045, 0.075),
    ("right_upper_arm", 14, 17, 0.055, 0.105),
    ("right_arm", 17, 19, 0.050, 0.090),
    ("right_forearm", 19, 21, 0.045, 0.080),
    ("right_hand", 21, 23, 0.045, 0.075),
    ("left_thigh", 0, 1, 0.075, 0.140),
    ("left_lower_thigh", 1, 4, 0.075, 0.130),
    ("left_shin", 4, 7, 0.060, 0.105),
    ("left_foot", 7, 10, 0.055, 0.095),
    ("right_thigh", 0, 2, 0.075, 0.140),
    ("right_lower_thigh", 2, 5, 0.075, 0.130),
    ("right_shin", 5, 8, 0.060, 0.105),
    ("right_foot", 8, 11, 0.055, 0.095),
)


@dataclass(frozen=True)
class Capsule:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float
    label: str

    def __post_init__(self) -> None:
        if self.radius <= 0.0 or not math.isfinite(self.radius):
            raise ValueError("capsule radius must be positive and finite")
        if not all(math.isfinite(value) for value in (*self.start, *self.end)):
            raise ValueError("capsule endpoints must be finite")


def capsule_signed_distance(points: np.ndarray, capsule: Capsule) -> np.ndarray:
    """Return the exact Euclidean signed distance to one closed capsule."""

    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("capsule SDF points must end in three coordinates")
    start = np.asarray(capsule.start, dtype=np.float64)
    end = np.asarray(capsule.end, dtype=np.float64)
    axis = end - start
    denominator = float(np.dot(axis, axis))
    if denominator <= 1.0e-12:
        return np.asarray(np.linalg.norm(values - start, axis=-1) - capsule.radius)
    parameter = np.sum((values - start) * axis, axis=-1) / denominator
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = start + parameter[..., None] * axis
    return np.asarray(np.linalg.norm(values - closest, axis=-1) - capsule.radius)


def capsule_tree_signed_distance(points: np.ndarray, capsules: tuple[Capsule, ...]) -> np.ndarray:
    """Exact hard-union field; no occupancy cleanup or component filtering."""

    if not capsules:
        raise ValueError("capsule tree cannot be empty")
    distances = np.stack([capsule_signed_distance(points, capsule) for capsule in capsules])
    return np.asarray(np.min(distances, axis=0), dtype=np.float64)


def _fit_real_capsules(vertices: np.ndarray, weights: np.ndarray) -> tuple[Capsule, ...]:
    points = np.asarray(vertices, dtype=np.float64)
    skinning = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("D03 scaffold vertices must have shape [V,3]")
    if skinning.shape != (len(points), 24):
        raise ValueError("D03 requires 24-joint SMPL skinning weights")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(skinning)):
        raise ValueError("D03 scaffold inputs must be finite")
    if np.any(skinning < 0.0) or not np.allclose(skinning.sum(axis=1), 1.0, atol=1.0e-5):
        raise ValueError("D03 skinning weights must be nonnegative and sum to one")
    joint_mass = skinning.sum(axis=0)
    if np.any(joint_mass <= 1.0e-8):
        raise ValueError("D03 requires nonzero support for every SMPL joint")
    centers = (skinning.T @ points) / joint_mass[:, None]
    capsules: list[Capsule] = []
    for label, start_joint, end_joint, lower_radius, upper_radius in _REAL_CAPSULE_SPECS:
        start = centers[start_joint]
        end = centers[end_joint]
        axis = end - start
        denominator = float(np.dot(axis, axis))
        if denominator <= 1.0e-12:
            raise ValueError(f"D03 capsule {label} has coincident scaffold joints")
        support = skinning[:, start_joint] + skinning[:, end_joint] > 0.15
        if np.count_nonzero(support) < 2:
            raise ValueError(f"D03 capsule {label} has insufficient scaffold support")
        supported_points = points[support]
        parameter = np.clip(((supported_points - start) @ axis) / denominator, 0.0, 1.0)
        distance = np.linalg.norm(supported_points - (start + parameter[:, None] * axis), axis=1)
        radius = float(np.clip(np.quantile(distance, 0.75), lower_radius, upper_radius))
        capsules.append(Capsule(tuple(start), tuple(end), radius, label))
    return tuple(capsules)


def build_d03_real_initialization_plan(
    *,
    scaffold_mesh_path: Path,
    skinning_weights_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Freeze the real-sequence capsule field before constructing its surface."""

    reject_sealed_capability([scaffold_mesh_path, skinning_weights_path])
    if len(source_revision) != 40:
        raise ValueError("D03 source revision must be a full commit hash")
    scaffold = np.load(scaffold_mesh_path)
    skinning = np.load(skinning_weights_path)
    if set(scaffold.files) != {"vertices", "faces"} or set(skinning.files) != {"weights"}:
        raise ValueError("D03 scaffold archives have unexpected members")
    vertices = np.asarray(scaffold["vertices"], dtype=np.float64)
    faces = np.asarray(scaffold["faces"], dtype=np.int64)
    weights = np.asarray(skinning["weights"], dtype=np.float64)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("D03 scaffold faces are invalid")
    capsules = _fit_real_capsules(vertices, weights)
    maximum_absolute_coordinate = float(np.max(np.abs(vertices)))
    extent = max(1.4, math.ceil((maximum_absolute_coordinate + 0.15) * 100.0) / 100.0)
    return {
        "schema_version": D03_REAL_PLAN_SCHEMA,
        "experiment_id": D03_EXPERIMENT_ID,
        "status": "real_initialization_planned",
        "source_revision": source_revision,
        "source": {
            "scaffold_mesh_path": str(scaffold_mesh_path),
            "scaffold_mesh_sha256": sha256_file(scaffold_mesh_path),
            "skinning_weights_path": str(skinning_weights_path),
            "skinning_weights_sha256": sha256_file(skinning_weights_path),
            "role": "geometry_and_rig_prior_only_never_topology_reference",
            "vertex_count": len(vertices),
            "face_count": len(faces),
        },
        "field": {
            "representation": "hard_union_of_closed_capsules",
            "resolution": 128,
            "symmetric_extent_metres": extent,
            "surface_level": 0.0,
            "extraction": "skimage_marching_cubes_allow_degenerate_false",
            "cleanup_operations_allowed": 0,
        },
        "capsule_fit": {
            "joint_center": "normalized_skinning_weighted_scaffold_centroid",
            "support_weight_threshold": 0.15,
            "radius_quantile": 0.75,
            "capsules": [
                {
                    "label": capsule.label,
                    "start": list(capsule.start),
                    "end": list(capsule.end),
                    "radius_metres": capsule.radius,
                }
                for capsule in capsules
            ],
        },
        "rig_transfer": {
            "method": "nearest_scaffold_vertex_skinning_weights",
            "topology_inherited": False,
        },
        "gates": {
            "component_count": 1,
            "euler_number": 2,
            "exact_self_intersections": 0,
            "watertight": True,
            "winding_consistent": True,
            "outward": True,
            "exact_replay": True,
        },
        "provenance": {
            "training_records_read": 0,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
        },
    }


def write_d03_real_initialization_plan(
    *,
    scaffold_mesh_path: Path,
    skinning_weights_path: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("D03 real initialization plan is immutable")
    return write_json(
        output_path,
        build_d03_real_initialization_plan(
            scaffold_mesh_path=scaffold_mesh_path,
            skinning_weights_path=skinning_weights_path,
            source_revision=source_revision,
        ),
    )


def _capsules_from_real_plan(plan: dict[str, Any]) -> tuple[Capsule, ...]:
    if plan.get("schema_version") != D03_REAL_PLAN_SCHEMA:
        raise ValueError("D03 real initialization plan schema is invalid")
    if plan.get("experiment_id") != D03_EXPERIMENT_ID:
        raise ValueError("D03 real initialization plan has the wrong experiment")
    if plan.get("status") != "real_initialization_planned":
        raise ValueError("D03 real initialization plan is not frozen")
    records = plan.get("capsule_fit", {}).get("capsules")
    if not isinstance(records, list) or len(records) != len(_REAL_CAPSULE_SPECS):
        raise ValueError("D03 real initialization plan has invalid capsules")
    capsules: list[Capsule] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("D03 real initialization capsule record is invalid")
        start = record.get("start")
        end = record.get("end")
        if (
            not isinstance(start, list)
            or len(start) != 3
            or not isinstance(end, list)
            or len(end) != 3
        ):
            raise ValueError("D03 real initialization capsule endpoints are invalid")
        capsules.append(
            Capsule(
                (float(start[0]), float(start[1]), float(start[2])),
                (float(end[0]), float(end[1]), float(end[2])),
                float(record["radius_metres"]),
                str(record["label"]),
            )
        )
    expected_labels = [spec[0] for spec in _REAL_CAPSULE_SPECS]
    if [capsule.label for capsule in capsules] != expected_labels:
        raise ValueError("D03 real initialization capsule order changed after planning")
    return tuple(capsules)


def build_d03_real_initialization(*, plan_path: Path, output_root: Path) -> Path:
    """Construct and exactly audit D03's frozen real-sequence initial field."""

    reject_sealed_capability([plan_path, output_root])
    if output_root.exists():
        raise FileExistsError("D03 real initialization output is immutable")
    plan = read_json(plan_path)
    capsules = _capsules_from_real_plan(plan)
    source = plan["source"]
    scaffold_path = Path(source["scaffold_mesh_path"])
    weights_path = Path(source["skinning_weights_path"])
    reject_sealed_capability([scaffold_path, weights_path])
    if sha256_file(scaffold_path) != source["scaffold_mesh_sha256"]:
        raise ValueError("D03 scaffold mesh changed after planning")
    if sha256_file(weights_path) != source["skinning_weights_sha256"]:
        raise ValueError("D03 skinning weights changed after planning")
    scaffold = np.load(scaffold_path)
    skinning = np.load(weights_path)
    source_vertices = np.asarray(scaffold["vertices"], dtype=np.float64)
    source_faces = np.asarray(scaffold["faces"], dtype=np.int64)
    source_weights = np.asarray(skinning["weights"], dtype=np.float64)
    if len(source_vertices) != source["vertex_count"] or len(source_faces) != source["face_count"]:
        raise ValueError("D03 scaffold counts changed after planning")
    if source_weights.shape != (len(source_vertices), 24):
        raise ValueError("D03 planned rig has an invalid shape")
    field_plan = plan["field"]
    resolution = int(field_plan["resolution"])
    extent = float(field_plan["symmetric_extent_metres"])
    if resolution < 48 or float(field_plan["surface_level"]) != 0.0:
        raise ValueError("D03 planned field parameters are invalid")
    if int(field_plan["cleanup_operations_allowed"]) != 0:
        raise ValueError("D03 real initialization cannot use cleanup")
    field = _field_for_capsules(capsules, resolution=resolution, extent=extent)
    vertices, faces = _extract_field_surface(field, extent=extent)
    topology = _mesh_audit(vertices, faces)
    replay_field = _field_for_capsules(capsules, resolution=resolution, extent=extent)
    replay_vertices, replay_faces = _extract_field_surface(replay_field, extent=extent)
    exact_replay = bool(
        np.array_equal(field, replay_field)
        and np.array_equal(vertices, replay_vertices)
        and np.array_equal(faces, replay_faces)
    )
    source_tree = cKDTree(source_vertices)
    candidate_tree = cKDTree(vertices)
    candidate_to_source_distance, nearest_source = source_tree.query(vertices, workers=1)
    source_to_candidate_distance = candidate_tree.query(source_vertices, workers=1)[0]
    transferred_weights = source_weights[np.asarray(nearest_source, dtype=np.int64)]
    transferred_weights /= transferred_weights.sum(axis=1, keepdims=True)
    expected_gates = plan["gates"]
    gates = {
        "component_count": topology["component_count"] == int(expected_gates["component_count"]),
        "euler_number": topology["euler_number"] == int(expected_gates["euler_number"]),
        "exact_self_intersections": topology["exact_self_intersections"] is False,
        "watertight": topology["watertight"] is bool(expected_gates["watertight"]),
        "winding_consistent": topology["winding_consistent"]
        is bool(expected_gates["winding_consistent"]),
        "outward": topology["outward"] is bool(expected_gates["outward"]),
        "exact_replay": exact_replay is bool(expected_gates["exact_replay"]),
        "skinning_weights_valid": bool(
            np.all(transferred_weights >= 0.0)
            and np.allclose(transferred_weights.sum(axis=1), 1.0, atol=1.0e-7)
        ),
    }
    status = "initial_field_qualified" if all(gates.values()) else "fail"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".d03-real-init-", dir=output_root.parent))
    try:
        field_path = staging / "canonical_capsule_field.npz"
        mesh_path = staging / "canonical_capsule_mesh.npz"
        np.savez_compressed(
            field_path,
            values=field,
            coordinates=np.linspace(-extent, extent, resolution),
            surface_level=np.asarray(0.0, dtype=np.float32),
        )
        np.savez_compressed(
            mesh_path,
            vertices=vertices.astype(np.float32),
            faces=faces,
            skinning_weights=transferred_weights.astype(np.float32),
            nearest_scaffold_vertex=np.asarray(nearest_source, dtype=np.int64),
        )
        final_field_path = output_root / field_path.name
        final_mesh_path = output_root / mesh_path.name
        report = {
            "schema_version": D03_REAL_INITIALIZATION_SCHEMA,
            "experiment_id": D03_EXPERIMENT_ID,
            "status": status,
            "source_revision": plan["source_revision"],
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "source": source,
            "topology": topology,
            "gates": gates,
            "field": {
                **field_plan,
                "minimum_value": float(field.min()),
                "maximum_value": float(field.max()),
                "path": str(final_field_path),
                "sha256": sha256_file(field_path),
            },
            "surface": {
                "path": str(final_mesh_path),
                "sha256": sha256_file(mesh_path),
                "vertex_count": len(vertices),
                "face_count": len(faces),
                "topology_inherited_from_scaffold": False,
                "cleanup_operations": 0,
                "bidirectional_scaffold_vertex_error_metres": float(
                    0.5
                    * (
                        np.mean(candidate_to_source_distance)
                        + np.mean(source_to_candidate_distance)
                    )
                ),
                "candidate_to_scaffold_p95_metres": float(
                    np.quantile(candidate_to_source_distance, 0.95)
                ),
                "scaffold_to_candidate_p95_metres": float(
                    np.quantile(source_to_candidate_distance, 0.95)
                ),
                "rig_transfer": plan["rig_transfer"],
            },
            "capsules": plan["capsule_fit"]["capsules"],
            "provenance": {
                **plan["provenance"],
                "source_scaffold_faces_used_as_topology": 0,
                "cleanup_operations": 0,
            },
        }
        report_path = staging / "real_initialization_report.json"
        write_json(report_path, report)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / "real_initialization_report.json"


def _d03_training_evidence_digest(records: list[Any], normal_root: Path, mask_root: Path) -> str:
    digest = hashlib.sha256()
    for record in records:
        name = Path(record.image_path).name
        digest.update(name.encode())
        digest.update(bytes.fromhex(sha256_file(normal_root / name)))
        digest.update(bytes.fromhex(sha256_file(mask_root / name)))
    return digest.hexdigest()


def build_d03_train_evidence_plan(
    *,
    real_initialization_report_path: Path,
    real_mesh_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    t05_solution_path: Path,
    normal_root: Path,
    mask_root: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Bind all and only training normals/masks before D03 evidence transfer."""

    paths = [
        real_initialization_report_path,
        real_mesh_path,
        manifest_path,
        joint_transforms_path,
        t05_solution_path,
        normal_root,
        mask_root,
    ]
    reject_sealed_capability(paths)
    if len(source_revision) != 40:
        raise ValueError("D03 train-evidence source revision must be a full commit hash")
    initialization = read_json(real_initialization_report_path)
    if initialization.get("status") != "initial_field_qualified":
        raise ValueError("D03 train evidence requires the qualified real initial field")
    if initialization.get("surface", {}).get("sha256") != sha256_file(real_mesh_path):
        raise ValueError("D03 real mesh does not match its initialization report")
    with np.load(real_mesh_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "vertices",
            "faces",
            "skinning_weights",
            "nearest_scaffold_vertex",
        }:
            raise ValueError("D03 real mesh archive has unexpected members")
        vertex_count = len(archive["vertices"])
        face_count = len(archive["faces"])
    manifest = read_dataset_manifest(manifest_path)
    train_records = [record for record in manifest.frames if record.split == "train"]
    if len(train_records) != 144:
        raise ValueError("D03 train evidence requires the frozen 144-frame split")
    training_sources = np.asarray(
        [record.source_frame_index for record in train_records], dtype=np.int64
    )
    t05 = read_json(t05_solution_path)
    t05_sources = np.asarray(
        [frame["source_frame_index"] for frame in t05.get("frames", [])], dtype=np.int64
    )
    if (
        t05.get("status") != "qualification_candidate"
        or t05.get("training_frame_count") != 144
        or not np.array_equal(training_sources, t05_sources)
    ):
        raise ValueError("D03 train evidence does not match the qualified T05 solution")
    with np.load(joint_transforms_path, allow_pickle=False) as transforms:
        transform_sources = transforms["source_frame_indices"].astype(np.int64)
        transform_values = transforms["transforms"]
    if transform_values.shape != (len(transform_sources), 24, 4, 4):
        raise ValueError("D03 train evidence has invalid joint transforms")
    if not set(training_sources.tolist()).issubset(set(transform_sources.tolist())):
        raise ValueError("D03 train evidence is missing training joint transforms")
    evidence_digest = _d03_training_evidence_digest(train_records, normal_root, mask_root)
    return {
        "schema_version": D03_TRAIN_EVIDENCE_PLAN_SCHEMA,
        "experiment_id": D03_EXPERIMENT_ID,
        "status": "train_evidence_transfer_planned",
        "source_revision": source_revision,
        "input_hashes": {
            "real_initialization_report": sha256_file(real_initialization_report_path),
            "real_mesh": sha256_file(real_mesh_path),
            "manifest": sha256_file(manifest_path),
            "joint_transforms": sha256_file(joint_transforms_path),
            "t05_solution": sha256_file(t05_solution_path),
            "training_normal_mask_evidence": evidence_digest,
        },
        "input_paths": {
            "real_initialization_report": str(real_initialization_report_path),
            "real_mesh": str(real_mesh_path),
            "manifest": str(manifest_path),
            "joint_transforms": str(joint_transforms_path),
            "t05_solution": str(t05_solution_path),
            "normal_root": str(normal_root),
            "mask_root": str(mask_root),
        },
        "mesh": {"vertex_count": vertex_count, "face_count": face_count},
        "transfer": {
            "pose_method": "frozen_smpl_joint_transforms_and_d03_transferred_weights",
            "normal_convention": "sapiens_rgb_xyz_then_camera_geometry_x_minus_y_minus_z",
            "foreground_mask_threshold": 127,
            "mask_erosion_pixels": 3,
            "minimum_face_pixels": 3,
            "minimum_face_normal_concentration": 0.55,
            "robust_fusion": "five_iteration_eight_degree_huber_with_prior_sign_alignment",
            "development_evidence_allowed": False,
        },
        "gates": {
            "training_frame_count": 144,
            "observed_vertex_fraction_minimum": 0.90,
            "median_views_per_observed_vertex_minimum": 20.0,
            "exact_fusion_replay": True,
            "development_records_read": 0,
            "sealed_test_reads": 0,
        },
        "provenance": {
            "training_records_bound": 144,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
        },
    }


def write_d03_train_evidence_plan(
    *,
    real_initialization_report_path: Path,
    real_mesh_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    t05_solution_path: Path,
    normal_root: Path,
    mask_root: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("D03 train evidence plan is immutable")
    return write_json(
        output_path,
        build_d03_train_evidence_plan(
            real_initialization_report_path=real_initialization_report_path,
            real_mesh_path=real_mesh_path,
            manifest_path=manifest_path,
            joint_transforms_path=joint_transforms_path,
            t05_solution_path=t05_solution_path,
            normal_root=normal_root,
            mask_root=mask_root,
            source_revision=source_revision,
        ),
    )


def _load_and_verify_d03_train_plan(plan_path: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    if plan.get("schema_version") != D03_TRAIN_EVIDENCE_PLAN_SCHEMA:
        raise ValueError("D03 train evidence plan schema is invalid")
    if plan.get("experiment_id") != D03_EXPERIMENT_ID:
        raise ValueError("D03 train evidence plan has the wrong experiment")
    if plan.get("status") != "train_evidence_transfer_planned":
        raise ValueError("D03 train evidence plan is not frozen")
    paths = {name: Path(path) for name, path in plan["input_paths"].items()}
    reject_sealed_capability([*paths.values(), plan_path])
    direct_hashes = {
        "real_initialization_report": sha256_file(paths["real_initialization_report"]),
        "real_mesh": sha256_file(paths["real_mesh"]),
        "manifest": sha256_file(paths["manifest"]),
        "joint_transforms": sha256_file(paths["joint_transforms"]),
        "t05_solution": sha256_file(paths["t05_solution"]),
    }
    for name, digest in direct_hashes.items():
        if digest != plan["input_hashes"][name]:
            raise ValueError(f"D03 train evidence input changed after planning: {name}")
    manifest = read_dataset_manifest(paths["manifest"])
    train_records = [record for record in manifest.frames if record.split == "train"]
    evidence_digest = _d03_training_evidence_digest(
        train_records, paths["normal_root"], paths["mask_root"]
    )
    if evidence_digest != plan["input_hashes"]["training_normal_mask_evidence"]:
        raise ValueError("D03 training normal/mask evidence changed after planning")
    return plan


def bind_d03_train_normal_evidence(*, plan_path: Path, output_root: Path) -> Path:
    """Pull all train-only normal evidence onto the embedded D03 body surface."""

    reject_sealed_capability([plan_path, output_root])
    if output_root.exists():
        raise FileExistsError("D03 train evidence output is immutable")
    plan = _load_and_verify_d03_train_plan(plan_path)
    paths = {name: Path(path) for name, path in plan["input_paths"].items()}
    manifest = read_dataset_manifest(paths["manifest"])
    records = [record for record in manifest.frames if record.split == "train"]
    t05 = read_json(paths["t05_solution"])
    intrinsics = np.asarray(t05["shared_intrinsics"], dtype=np.float64)
    with np.load(paths["real_mesh"], allow_pickle=False) as archive:
        vertices = archive["vertices"].astype(np.float32)
        faces = archive["faces"].astype(np.int64)
        weights = archive["skinning_weights"].astype(np.float32)
    if len(vertices) != plan["mesh"]["vertex_count"] or len(faces) != plan["mesh"]["face_count"]:
        raise ValueError("D03 train evidence mesh counts changed after planning")
    with np.load(paths["joint_transforms"], allow_pickle=False) as transform_archive:
        source_indices = transform_archive["source_frame_indices"].astype(np.int64)
        transforms = transform_archive["transforms"].astype(np.float32)
    transform_lookup = {int(source): slot for slot, source in enumerate(source_indices)}
    source_size = (manifest.video.height, manifest.video.width)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    prior_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    observations = np.zeros((len(records), len(vertices), 3), dtype=np.float32)
    confidence = np.zeros((len(records), len(vertices)), dtype=np.float32)
    valid_face_counts: list[int] = []
    valid_vertex_counts: list[int] = []
    median_face_changes: list[float] = []
    vertex_tensor = torch.from_numpy(vertices)
    weight_tensor = torch.from_numpy(weights)
    for record_slot, record in enumerate(records):
        transform_slot = transform_lookup.get(record.source_frame_index)
        if transform_slot is None:
            raise ValueError("D03 train evidence lacks a planned joint transform")
        with torch.no_grad():
            posed = linear_blend_skinning(
                vertex_tensor,
                weight_tensor,
                torch.from_numpy(transforms[transform_slot]),
            ).numpy()
        name = Path(record.image_path).name
        normal_image = cv2.imread(str(paths["normal_root"] / name), cv2.IMREAD_COLOR)
        mask_image = cv2.imread(str(paths["mask_root"] / name), cv2.IMREAD_GRAYSCALE)
        if normal_image is None or mask_image is None:
            raise FileNotFoundError(f"D03 lacks planned training evidence: {name}")
        frame_normals, frame_confidence, diagnostic = sample_pose_stabilized_vertex_normals(
            vertices,
            posed,
            faces,
            intrinsics,
            normal_image,
            mask_image,
            source_size=source_size,
            erosion_pixels=int(plan["transfer"]["mask_erosion_pixels"]),
            minimum_face_pixels=int(plan["transfer"]["minimum_face_pixels"]),
        )
        observations[record_slot] = frame_normals.astype(np.float32)
        confidence[record_slot] = frame_confidence.astype(np.float32)
        valid_face_counts.append(int(diagnostic["valid_face_count"]))
        valid_vertex_counts.append(int(diagnostic["valid_vertex_count"]))
        median_face_changes.append(float(diagnostic["median_pulled_normal_prior_degrees"]))
    fused, support = robust_fuse_normals(
        observations,
        confidence,
        reference=prior_normals,
        huber_degrees=8.0,
        iterations=5,
    )
    replay_fused, replay_support = robust_fuse_normals(
        observations,
        confidence,
        reference=prior_normals,
        huber_degrees=8.0,
        iterations=5,
    )
    exact_replay = bool(
        np.array_equal(fused, replay_fused) and np.array_equal(support, replay_support)
    )
    views = np.count_nonzero(confidence > 0.0, axis=0)
    observed = views > 0
    observed_fraction = float(np.mean(observed))
    median_views = float(np.median(views[observed])) if np.any(observed) else 0.0
    gates = {
        "training_frame_count": len(records) == int(plan["gates"]["training_frame_count"]),
        "observed_vertex_fraction": observed_fraction
        >= float(plan["gates"]["observed_vertex_fraction_minimum"]),
        "median_views_per_observed_vertex": median_views
        >= float(plan["gates"]["median_views_per_observed_vertex_minimum"]),
        "exact_fusion_replay": exact_replay is bool(plan["gates"]["exact_fusion_replay"]),
        "development_records_read": int(plan["gates"]["development_records_read"]) == 0,
        "sealed_test_reads": int(plan["gates"]["sealed_test_reads"]) == 0,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".d03-train-evidence-", dir=output_root.parent))
    try:
        evidence_path = staging / "train_pose_stabilized_normals.npz"
        np.savez_compressed(
            evidence_path,
            observations=observations,
            confidence=confidence,
            fused_normals=fused.astype(np.float32),
            robust_support=support.astype(np.float32),
            prior_vertex_normals=prior_normals.astype(np.float32),
            source_frame_indices=np.asarray(
                [record.source_frame_index for record in records], dtype=np.int64
            ),
        )
        final_evidence_path = output_root / evidence_path.name
        report = {
            "schema_version": D03_TRAIN_EVIDENCE_SCHEMA,
            "experiment_id": D03_EXPERIMENT_ID,
            "status": "train_evidence_bound" if all(gates.values()) else "fail",
            "source_revision": plan["source_revision"],
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "artifact": {
                "path": str(final_evidence_path),
                "sha256": sha256_file(evidence_path),
            },
            "training_records_read": len(records),
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "observed_vertex_fraction": observed_fraction,
            "median_views_per_observed_vertex": median_views,
            "minimum_views_per_observed_vertex": int(views[observed].min())
            if np.any(observed)
            else 0,
            "median_frame_valid_face_count": float(np.median(valid_face_counts)),
            "median_frame_valid_vertex_count": float(np.median(valid_vertex_counts)),
            "median_frame_pulled_normal_prior_degrees": float(np.median(median_face_changes)),
            "median_fused_normal_change_degrees": float(
                np.median(
                    np.degrees(
                        np.arccos(np.clip(np.sum(fused * prior_normals, axis=-1), -1.0, 1.0))
                    )[observed]
                )
            ),
            "gates": gates,
            "input_hashes": plan["input_hashes"],
        }
        report_path = staging / "train_normal_binding_report.json"
        write_json(report_path, report)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / "train_normal_binding_report.json"


def build_d03_implicit_continuation_plan(
    *,
    public_benchmark_path: Path,
    real_initialization_report_path: Path,
    real_field_path: Path,
    real_mesh_path: Path,
    train_evidence_report_path: Path,
    train_evidence_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Freeze one train-only D03 field continuation before creating a candidate."""

    paths = [
        public_benchmark_path,
        real_initialization_report_path,
        real_field_path,
        real_mesh_path,
        train_evidence_report_path,
        train_evidence_path,
    ]
    reject_sealed_capability(paths)
    if len(source_revision) != 40:
        raise ValueError("D03 continuation source revision must be a full commit hash")
    public = read_json(public_benchmark_path)
    initialization = read_json(real_initialization_report_path)
    evidence = read_json(train_evidence_report_path)
    if public.get("status") != "pass" or public.get("experiment_id") != D03_EXPERIMENT_ID:
        raise ValueError("D03 continuation requires its passing public benchmark")
    if (
        initialization.get("status") != "initial_field_qualified"
        or initialization.get("field", {}).get("sha256") != sha256_file(real_field_path)
        or initialization.get("surface", {}).get("sha256") != sha256_file(real_mesh_path)
    ):
        raise ValueError("D03 continuation requires its qualified real initial field")
    if (
        evidence.get("status") != "train_evidence_bound"
        or evidence.get("artifact", {}).get("sha256") != sha256_file(train_evidence_path)
        or evidence.get("training_records_read") != 144
        or evidence.get("development_records_read") != 0
    ):
        raise ValueError("D03 continuation requires its passing train-only evidence binding")
    return {
        "schema_version": D03_CONTINUATION_PLAN_SCHEMA,
        "experiment_id": D03_EXPERIMENT_ID,
        "status": "frozen_before_implicit_continuation",
        "source_revision": source_revision,
        "input_paths": {
            "public_benchmark": str(public_benchmark_path),
            "real_initialization_report": str(real_initialization_report_path),
            "real_field": str(real_field_path),
            "real_mesh": str(real_mesh_path),
            "train_evidence_report": str(train_evidence_report_path),
            "train_evidence": str(train_evidence_path),
        },
        "input_hashes": {
            "public_benchmark": sha256_file(public_benchmark_path),
            "real_initialization_report": sha256_file(real_initialization_report_path),
            "real_field": sha256_file(real_field_path),
            "real_mesh": sha256_file(real_mesh_path),
            "train_evidence_report": sha256_file(train_evidence_report_path),
            "train_evidence": sha256_file(train_evidence_path),
        },
        "schedule": D03_CONTINUATION_SCHEDULE,
        "candidate_policy": {
            "raw_mesh_update": "bounded_scalar_normal_integration",
            "mesh_projection": "local_orientation_and_area_trust_region",
            "field_update": "band_limited_inverse_distance_scalar_offset_of_capsule_field",
            "surface_extraction": "fresh_zero_level_marching_cubes",
            "source_faces_used_as_topology": 0,
            "cleanup_operations_allowed": 0,
            "automatic_retries": 0,
        },
        "gates": {
            "canonical_median_normal_improvement_degrees_minimum": 1.0,
            "initial_mesh_projection_constraints_pass": True,
            "candidate_component_count": 1,
            "candidate_euler_number": 2,
            "candidate_exact_self_intersections": 0,
            "candidate_watertight": True,
            "candidate_winding_consistent": True,
            "candidate_outward": True,
            "exact_candidate_replay": True,
            "development_records_read": 0,
            "sealed_test_reads": 0,
        },
        "provenance": {
            "training_records_bound": 144,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
        },
    }


def write_d03_implicit_continuation_plan(
    *,
    public_benchmark_path: Path,
    real_initialization_report_path: Path,
    real_field_path: Path,
    real_mesh_path: Path,
    train_evidence_report_path: Path,
    train_evidence_path: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("D03 implicit continuation plan is immutable")
    return write_json(
        output_path,
        build_d03_implicit_continuation_plan(
            public_benchmark_path=public_benchmark_path,
            real_initialization_report_path=real_initialization_report_path,
            real_field_path=real_field_path,
            real_mesh_path=real_mesh_path,
            train_evidence_report_path=train_evidence_report_path,
            train_evidence_path=train_evidence_path,
            source_revision=source_revision,
        ),
    )


def offset_field_from_surface_displacement(
    field: np.ndarray,
    coordinates: np.ndarray,
    surface_vertices: np.ndarray,
    signed_displacement: np.ndarray,
    *,
    neighbours: int,
    epsilon_metres: float,
    update_band_metres: float,
) -> np.ndarray:
    """Apply a deterministic band-limited inverse-distance zero-set offset."""

    values = np.asarray(field, dtype=np.float64)
    axis = np.asarray(coordinates, dtype=np.float64)
    vertices = np.asarray(surface_vertices, dtype=np.float64)
    displacement = np.asarray(signed_displacement, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape != (len(axis), len(axis), len(axis))
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or displacement.shape != (len(vertices),)
    ):
        raise ValueError("D03 field offset inputs are not aligned")
    if neighbours < 1 or neighbours > len(vertices):
        raise ValueError("D03 field offset neighbour count is invalid")
    if epsilon_metres <= 0.0 or update_band_metres <= 0.0:
        raise ValueError("D03 field offset scales must be positive")
    tree = cKDTree(vertices)
    result = values.copy()
    yy, zz = np.meshgrid(axis, axis, indexing="ij")
    for x_index, x in enumerate(axis):
        points = np.stack((np.full_like(yy, x), yy, zz), axis=-1).reshape(-1, 3)
        distance, indices = tree.query(points, k=neighbours, workers=1)
        if neighbours == 1:
            distance = distance[:, None]
            indices = indices[:, None]
        inverse_square = 1.0 / np.square(distance + epsilon_metres)
        interpolated = np.sum(inverse_square * displacement[indices], axis=1) / np.sum(
            inverse_square, axis=1
        )
        fade = np.clip(1.0 - np.abs(values[x_index].reshape(-1)) / update_band_metres, 0.0, 1.0)
        result[x_index] -= (fade * interpolated).reshape(values.shape[1:])
    return result.astype(np.float32)


def _angular_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    first_values /= np.linalg.norm(first_values, axis=-1, keepdims=True)
    second_values /= np.linalg.norm(second_values, axis=-1, keepdims=True)
    return np.asarray(
        np.degrees(np.arccos(np.clip(np.sum(first_values * second_values, axis=-1), -1.0, 1.0))),
        dtype=np.float64,
    )


def build_d03_implicit_continuation(*, plan_path: Path, output_root: Path) -> Path:
    """Create one frozen D03 field candidate and run every exact acceptance gate."""

    reject_sealed_capability([plan_path, output_root])
    if output_root.exists():
        raise FileExistsError("D03 implicit continuation output is immutable")
    plan = read_json(plan_path)
    if (
        plan.get("schema_version") != D03_CONTINUATION_PLAN_SCHEMA
        or plan.get("experiment_id") != D03_EXPERIMENT_ID
        or plan.get("status") != "frozen_before_implicit_continuation"
        or plan.get("schedule") != D03_CONTINUATION_SCHEDULE
    ):
        raise ValueError("D03 implicit continuation rejected its frozen plan")
    paths = {name: Path(path) for name, path in plan["input_paths"].items()}
    reject_sealed_capability([*paths.values()])
    for name, path in paths.items():
        if sha256_file(path) != plan["input_hashes"][name]:
            raise ValueError(f"D03 implicit continuation input changed: {name}")
    with np.load(paths["real_mesh"], allow_pickle=False) as mesh_archive:
        vertices = mesh_archive["vertices"].astype(np.float64)
        faces = mesh_archive["faces"].astype(np.int64)
        weights = mesh_archive["skinning_weights"].astype(np.float64)
    with np.load(paths["real_field"], allow_pickle=False) as field_archive:
        initial_field = field_archive["values"].astype(np.float32)
        coordinates = field_archive["coordinates"].astype(np.float64)
    with np.load(paths["train_evidence"], allow_pickle=False) as evidence_archive:
        fused_normals = evidence_archive["fused_normals"].astype(np.float64)
        robust_support = evidence_archive["robust_support"].astype(np.float64)
        prior_normals = evidence_archive["prior_vertex_normals"].astype(np.float64)
    schedule = plan["schedule"]
    raw_vertices, raw_signed_displacement = integrate_mesh_normals_along_prior(
        vertices,
        faces,
        prior_normals,
        fused_normals,
        robust_support,
        maximum_displacement_metres=float(schedule["maximum_normal_displacement_metres"]),
        normal_equation_weight=float(schedule["normal_equation_weight"]),
        prior_anchor_weight=float(schedule["prior_anchor_weight"]),
        edge_smoothness_weight=float(schedule["edge_smoothness_weight"]),
    )
    projected_vertices, local_scales, projection = topology_constrained_local_trust_projection(
        vertices,
        raw_vertices,
        faces,
        minimum_area_ratio=float(schedule["minimum_mesh_area_ratio"]),
        maximum_rejections=int(schedule["maximum_local_projection_rejections"]),
        shrink_factor=float(schedule["local_projection_shrink_factor"]),
        minimum_vertex_scale=float(schedule["minimum_local_vertex_scale"]),
    )
    projection_audit = exact_face_constraint_audit(
        vertices,
        projected_vertices,
        faces,
        minimum_area_ratio=float(schedule["minimum_mesh_area_ratio"]),
    )
    projected_signed_displacement = np.sum((projected_vertices - vertices) * prior_normals, axis=-1)

    def construct_field() -> np.ndarray:
        return offset_field_from_surface_displacement(
            initial_field,
            coordinates,
            vertices,
            projected_signed_displacement,
            neighbours=int(schedule["field_interpolation_neighbours"]),
            epsilon_metres=float(schedule["field_interpolation_epsilon_metres"]),
            update_band_metres=float(schedule["field_update_band_metres"]),
        )

    candidate_field = construct_field()
    extent = float(max(abs(coordinates[0]), abs(coordinates[-1])))
    candidate_vertices, candidate_faces = _extract_field_surface(candidate_field, extent=extent)
    candidate_topology = _mesh_audit(candidate_vertices, candidate_faces)
    replay_field = construct_field()
    replay_vertices, replay_faces = _extract_field_surface(replay_field, extent=extent)
    exact_replay = bool(
        np.array_equal(candidate_field, replay_field)
        and np.array_equal(candidate_vertices, replay_vertices)
        and np.array_equal(candidate_faces, replay_faces)
    )
    initial_tree = cKDTree(vertices)
    _, nearest_initial = initial_tree.query(candidate_vertices, workers=1)
    nearest_initial = np.asarray(nearest_initial, dtype=np.int64)
    candidate_mesh = trimesh.Trimesh(
        vertices=candidate_vertices, faces=candidate_faces, process=False
    )
    candidate_normals = np.asarray(candidate_mesh.vertex_normals, dtype=np.float64)
    supported = robust_support[nearest_initial] > 0.0
    initial_normal_error = _angular_degrees(
        prior_normals[nearest_initial][supported], fused_normals[nearest_initial][supported]
    )
    candidate_normal_error = _angular_degrees(
        candidate_normals[supported], fused_normals[nearest_initial][supported]
    )
    initial_median_normal = float(np.median(initial_normal_error))
    candidate_median_normal = float(np.median(candidate_normal_error))
    normal_improvement = initial_median_normal - candidate_median_normal
    candidate_weights = weights[nearest_initial]
    expected = plan["gates"]
    gates = {
        "canonical_median_normal_improvement": normal_improvement
        >= float(expected["canonical_median_normal_improvement_degrees_minimum"]),
        "initial_mesh_projection_constraints_pass": projection_audit["status"] == "pass",
        "candidate_component_count": candidate_topology["component_count"]
        == int(expected["candidate_component_count"]),
        "candidate_euler_number": candidate_topology["euler_number"]
        == int(expected["candidate_euler_number"]),
        "candidate_exact_self_intersections": candidate_topology["exact_self_intersections"]
        is False,
        "candidate_watertight": candidate_topology["watertight"]
        is bool(expected["candidate_watertight"]),
        "candidate_winding_consistent": candidate_topology["winding_consistent"]
        is bool(expected["candidate_winding_consistent"]),
        "candidate_outward": candidate_topology["outward"] is bool(expected["candidate_outward"]),
        "exact_candidate_replay": exact_replay is bool(expected["exact_candidate_replay"]),
        "development_records_read": int(expected["development_records_read"]) == 0,
        "sealed_test_reads": int(expected["sealed_test_reads"]) == 0,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".d03-continuation-", dir=output_root.parent))
    try:
        field_path = staging / "continued_canonical_field.npz"
        mesh_path = staging / "continued_canonical_mesh.npz"
        np.savez_compressed(
            field_path,
            values=candidate_field,
            coordinates=coordinates,
            surface_level=np.asarray(0.0, dtype=np.float32),
        )
        np.savez_compressed(
            mesh_path,
            vertices=candidate_vertices.astype(np.float32),
            faces=candidate_faces,
            skinning_weights=candidate_weights.astype(np.float32),
            nearest_initial_vertex=nearest_initial,
        )
        final_field_path = output_root / field_path.name
        final_mesh_path = output_root / mesh_path.name
        displacement_magnitude = np.linalg.norm(projected_vertices - vertices, axis=-1)
        report = {
            "schema_version": D03_CONTINUATION_SCHEMA,
            "experiment_id": D03_EXPERIMENT_ID,
            "status": "pass" if all(gates.values()) else "fail",
            "decision": "eligible_for_train_image_evaluation"
            if all(gates.values())
            else "terminal_failed_train_implicit_continuation",
            "source_revision": plan["source_revision"],
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "schedule": schedule,
            "raw_signed_displacement_metres": {
                "minimum": float(raw_signed_displacement.min()),
                "median_absolute": float(np.median(np.abs(raw_signed_displacement))),
                "maximum": float(raw_signed_displacement.max()),
            },
            "projected_displacement_metres": {
                "median": float(np.median(displacement_magnitude)),
                "p95": float(np.quantile(displacement_magnitude, 0.95)),
                "maximum": float(displacement_magnitude.max()),
            },
            "local_projection": projection,
            "local_projection_scales": {
                "minimum": float(local_scales.min()),
                "median": float(np.median(local_scales)),
                "full_scale_vertex_fraction": float(np.mean(local_scales == 1.0)),
            },
            "local_projection_audit": projection_audit,
            "candidate_topology": candidate_topology,
            "canonical_normal_diagnostic": {
                "initial_median_degrees_on_candidate_support": initial_median_normal,
                "candidate_median_degrees": candidate_median_normal,
                "improvement_degrees": normal_improvement,
                "supported_candidate_vertex_count": int(np.count_nonzero(supported)),
            },
            "gates": gates,
            "artifacts": {
                "continued_field": {
                    "path": str(final_field_path),
                    "sha256": sha256_file(field_path),
                },
                "continued_mesh": {
                    "path": str(final_mesh_path),
                    "sha256": sha256_file(mesh_path),
                },
            },
            "provenance": {
                **plan["provenance"],
                "cleanup_operations": 0,
                "source_scaffold_faces_used_as_topology": 0,
            },
        }
        report_path = staging / "implicit_continuation_report.json"
        write_json(report_path, report)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / "implicit_continuation_report.json"


def build_d03_train_evaluation_plan(
    *,
    real_initialization_report_path: Path,
    initial_mesh_path: Path,
    continuation_report_path: Path,
    candidate_mesh_path: Path,
    train_evidence_plan_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Freeze the independent train-image comparison before evaluation."""

    paths = [
        real_initialization_report_path,
        initial_mesh_path,
        continuation_report_path,
        candidate_mesh_path,
        train_evidence_plan_path,
    ]
    reject_sealed_capability(paths)
    if len(source_revision) != 40:
        raise ValueError("D03 train-evaluation source revision must be a full commit hash")
    initialization = read_json(real_initialization_report_path)
    continuation = read_json(continuation_report_path)
    evidence_plan = read_json(train_evidence_plan_path)
    if initialization.get("status") != "initial_field_qualified" or initialization.get(
        "surface", {}
    ).get("sha256") != sha256_file(initial_mesh_path):
        raise ValueError("D03 train evaluation requires its initial mesh control")
    if (
        continuation.get("status") != "pass"
        or continuation.get("decision") != "eligible_for_train_image_evaluation"
        or continuation.get("artifacts", {}).get("continued_mesh", {}).get("sha256")
        != sha256_file(candidate_mesh_path)
    ):
        raise ValueError("D03 train evaluation requires its passing continuation candidate")
    if (
        evidence_plan.get("status") != "train_evidence_transfer_planned"
        or evidence_plan.get("provenance", {}).get("training_records_bound") != 144
        or evidence_plan.get("provenance", {}).get("development_records_read") != 0
    ):
        raise ValueError("D03 train evaluation requires its frozen train-only evidence plan")
    evaluation_inputs = {
        name: evidence_plan["input_paths"][name]
        for name in ("manifest", "joint_transforms", "t05_solution", "normal_root", "mask_root")
    }
    return {
        "schema_version": D03_TRAIN_EVALUATION_PLAN_SCHEMA,
        "experiment_id": D03_EXPERIMENT_ID,
        "status": "frozen_before_train_image_evaluation",
        "source_revision": source_revision,
        "input_paths": {
            "real_initialization_report": str(real_initialization_report_path),
            "initial_mesh": str(initial_mesh_path),
            "continuation_report": str(continuation_report_path),
            "candidate_mesh": str(candidate_mesh_path),
            "train_evidence_plan": str(train_evidence_plan_path),
            **evaluation_inputs,
        },
        "input_hashes": {
            "real_initialization_report": sha256_file(real_initialization_report_path),
            "initial_mesh": sha256_file(initial_mesh_path),
            "continuation_report": sha256_file(continuation_report_path),
            "candidate_mesh": sha256_file(candidate_mesh_path),
            "train_evidence_plan": sha256_file(train_evidence_plan_path),
            "manifest": evidence_plan["input_hashes"]["manifest"],
            "joint_transforms": evidence_plan["input_hashes"]["joint_transforms"],
            "t05_solution": evidence_plan["input_hashes"]["t05_solution"],
            "training_normal_mask_evidence": evidence_plan["input_hashes"][
                "training_normal_mask_evidence"
            ],
        },
        "matched_comparison": {
            "control": "raw_embedded_d03_capsule_initialization",
            "treatment": "single_frozen_normal_continued_d03_field",
            "pose_and_rasterizer_shared": True,
            "training_frames_only": True,
        },
        "training_gates": D03_TRAIN_EVALUATION_GATES,
        "provenance": {
            "training_records_authorized": 144,
            "development_records_authorized": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
        },
    }


def write_d03_train_evaluation_plan(
    *,
    real_initialization_report_path: Path,
    initial_mesh_path: Path,
    continuation_report_path: Path,
    candidate_mesh_path: Path,
    train_evidence_plan_path: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("D03 train evaluation plan is immutable")
    return write_json(
        output_path,
        build_d03_train_evaluation_plan(
            real_initialization_report_path=real_initialization_report_path,
            initial_mesh_path=initial_mesh_path,
            continuation_report_path=continuation_report_path,
            candidate_mesh_path=candidate_mesh_path,
            train_evidence_plan_path=train_evidence_plan_path,
            source_revision=source_revision,
        ),
    )


def evaluate_d03_train_images(*, plan_path: Path, output_path: Path) -> Path:
    """Compare D03 initial and continued fields on all and only training images."""

    reject_sealed_capability([plan_path, output_path])
    if output_path.exists():
        raise FileExistsError("D03 train image evaluation is immutable")
    plan = read_json(plan_path)
    if (
        plan.get("schema_version") != D03_TRAIN_EVALUATION_PLAN_SCHEMA
        or plan.get("experiment_id") != D03_EXPERIMENT_ID
        or plan.get("status") != "frozen_before_train_image_evaluation"
        or plan.get("training_gates") != D03_TRAIN_EVALUATION_GATES
    ):
        raise ValueError("D03 train image evaluator rejected its frozen plan")
    paths = {name: Path(path) for name, path in plan["input_paths"].items()}
    reject_sealed_capability([*paths.values()])
    for name in (
        "real_initialization_report",
        "initial_mesh",
        "continuation_report",
        "candidate_mesh",
        "train_evidence_plan",
        "manifest",
        "joint_transforms",
        "t05_solution",
    ):
        if sha256_file(paths[name]) != plan["input_hashes"][name]:
            raise ValueError(f"D03 train image input changed after planning: {name}")
    manifest = read_dataset_manifest(paths["manifest"])
    records = [record for record in manifest.frames if record.split == "train"]
    if len(records) != plan["provenance"]["training_records_authorized"]:
        raise ValueError("D03 train image evaluator requires all 144 training records")
    evidence_digest = _d03_training_evidence_digest(
        records, paths["normal_root"], paths["mask_root"]
    )
    if evidence_digest != plan["input_hashes"]["training_normal_mask_evidence"]:
        raise ValueError("D03 train image evidence changed after planning")
    with np.load(paths["initial_mesh"], allow_pickle=False) as archive:
        control_vertices = archive["vertices"].astype(np.float32)
        control_faces = archive["faces"].astype(np.int64)
        control_weights = archive["skinning_weights"].astype(np.float32)
    with np.load(paths["candidate_mesh"], allow_pickle=False) as archive:
        treatment_vertices = archive["vertices"].astype(np.float32)
        treatment_faces = archive["faces"].astype(np.int64)
        treatment_weights = archive["skinning_weights"].astype(np.float32)
    with np.load(paths["joint_transforms"], allow_pickle=False) as archive:
        source_indices = archive["source_frame_indices"].astype(np.int64)
        transforms = archive["transforms"].astype(np.float32)
    transform_lookup = {int(source): slot for slot, source in enumerate(source_indices)}
    t05 = read_json(paths["t05_solution"])
    intrinsics = np.asarray(t05["shared_intrinsics"], dtype=np.float64)
    source_size = (manifest.video.height, manifest.video.width)
    control_vertex_tensor = torch.from_numpy(control_vertices)
    control_weight_tensor = torch.from_numpy(control_weights)
    treatment_vertex_tensor = torch.from_numpy(treatment_vertices)
    treatment_weight_tensor = torch.from_numpy(treatment_weights)
    control_ious: list[float] = []
    treatment_ious: list[float] = []
    control_boundaries: list[float] = []
    treatment_boundaries: list[float] = []
    control_normal_degrees: list[float] = []
    treatment_normal_degrees: list[float] = []
    for record in records:
        transform_slot = transform_lookup.get(record.source_frame_index)
        if transform_slot is None:
            raise ValueError("D03 train image evaluator lacks a joint transform")
        transform = torch.from_numpy(transforms[transform_slot])
        with torch.no_grad():
            posed_control = linear_blend_skinning(
                control_vertex_tensor, control_weight_tensor, transform
            ).numpy()
            posed_treatment = linear_blend_skinning(
                treatment_vertex_tensor, treatment_weight_tensor, transform
            ).numpy()
        name = Path(record.image_path).name
        normal_image = cv2.imread(str(paths["normal_root"] / name), cv2.IMREAD_COLOR)
        mask_image = cv2.imread(str(paths["mask_root"] / name), cv2.IMREAD_GRAYSCALE)
        if normal_image is None or mask_image is None:
            raise FileNotFoundError(f"D03 train image evaluator lacks evidence: {name}")
        target_mask = mask_image > 127
        target_normal = decode_sapiens_normal_bgr(normal_image)
        control_mask, control_normal = _render_hard_geometry_normals(
            posed_control, control_faces, intrinsics, source_size
        )
        treatment_mask, treatment_normal = _render_hard_geometry_normals(
            posed_treatment, treatment_faces, intrinsics, source_size
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
        target_valid = np.linalg.norm(target_normal, axis=-1) > 0.5
        control_valid = eroded & control_mask & target_valid
        treatment_valid = eroded & treatment_mask & target_valid
        if not np.any(control_valid) or not np.any(treatment_valid):
            raise ValueError("D03 train image evaluator found no valid normal pixels")
        control_normal_degrees.append(
            float(
                np.median(
                    _angular_degrees(control_normal[control_valid], target_normal[control_valid])
                )
            )
        )
        treatment_normal_degrees.append(
            float(
                np.median(
                    _angular_degrees(
                        treatment_normal[treatment_valid], target_normal[treatment_valid]
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
    thresholds = plan["training_gates"]
    gates = {
        "median_normal_improvement": normal_improvement
        >= float(thresholds["median_normal_improvement_degrees_minimum"]),
        "median_hard_iou_nonregression": iou_regression
        <= float(thresholds["median_hard_iou_regression_maximum"]),
        "median_boundary_nonregression": boundary_regression
        <= float(thresholds["median_boundary_error_regression_maximum"]),
        "training_records_only": len(records) == 144,
        "development_records_read": int(plan["provenance"]["development_records_authorized"]) == 0,
        "sealed_test_reads": int(plan["provenance"]["sealed_test_reads"]) == 0,
    }
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_d03_train_image_evaluation.v1",
            "experiment_id": D03_EXPERIMENT_ID,
            "status": "pass" if all(gates.values()) else "fail",
            "decision": "eligible_for_development_evaluation_freeze"
            if all(gates.values())
            else "terminal_failed_train_image_evaluation",
            "source_revision": plan["source_revision"],
            "metrics": metrics,
            "normal_improvement_degrees": normal_improvement,
            "hard_iou_regression": iou_regression,
            "boundary_error_regression": boundary_regression,
            "gates": gates,
            "training_records_read": len(records),
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "input_hashes": plan["input_hashes"],
            "plan_sha256": sha256_file(plan_path),
        },
    )


def _d03_split_evidence_digest(records: list[Any], normal_root: Path, mask_root: Path) -> str:
    return _d03_training_evidence_digest(records, normal_root, mask_root)


def build_d03_development_evaluation_plan(
    *,
    train_evaluation_path: Path,
    continuation_report_path: Path,
    initial_mesh_path: Path,
    candidate_mesh_path: Path,
    train_evidence_plan_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Freeze D03's single development read after all training gates pass."""

    paths = [
        train_evaluation_path,
        continuation_report_path,
        initial_mesh_path,
        candidate_mesh_path,
        train_evidence_plan_path,
    ]
    reject_sealed_capability(paths)
    if len(source_revision) != 40:
        raise ValueError("D03 development-plan source revision must be a full commit hash")
    train = read_json(train_evaluation_path)
    continuation = read_json(continuation_report_path)
    evidence_plan = read_json(train_evidence_plan_path)
    if train.get("status") != "pass" or train.get("decision") != (
        "eligible_for_development_evaluation_freeze"
    ):
        raise ValueError("D03 development evaluation requires the passing train gate")
    if continuation.get("status") != "pass" or continuation.get("artifacts", {}).get(
        "continued_mesh", {}
    ).get("sha256") != sha256_file(candidate_mesh_path):
        raise ValueError("D03 development evaluation requires its continued mesh")
    initialization_report = read_json(
        Path(evidence_plan["input_paths"]["real_initialization_report"])
    )
    if initialization_report.get("surface", {}).get("sha256") != sha256_file(initial_mesh_path):
        raise ValueError("D03 development evaluation requires its initial mesh control")
    manifest_path = Path(evidence_plan["input_paths"]["manifest"])
    normal_root = Path(evidence_plan["input_paths"]["normal_root"])
    mask_root = Path(evidence_plan["input_paths"]["mask_root"])
    manifest = read_dataset_manifest(manifest_path)
    development_records = [record for record in manifest.frames if record.split == "held_out"]
    if len(development_records) != 36:
        raise ValueError("D03 development evaluation requires the frozen 36-frame split")
    development_digest = _d03_split_evidence_digest(development_records, normal_root, mask_root)
    return {
        "schema_version": D03_DEVELOPMENT_PLAN_SCHEMA,
        "experiment_id": D03_EXPERIMENT_ID,
        "status": "frozen_before_single_development_evaluation",
        "source_revision": source_revision,
        "input_paths": {
            "train_evaluation": str(train_evaluation_path),
            "continuation_report": str(continuation_report_path),
            "initial_mesh": str(initial_mesh_path),
            "candidate_mesh": str(candidate_mesh_path),
            "train_evidence_plan": str(train_evidence_plan_path),
            "manifest": str(manifest_path),
            "joint_transforms": evidence_plan["input_paths"]["joint_transforms"],
            "t05_solution": evidence_plan["input_paths"]["t05_solution"],
            "normal_root": str(normal_root),
            "mask_root": str(mask_root),
        },
        "input_hashes": {
            "train_evaluation": sha256_file(train_evaluation_path),
            "continuation_report": sha256_file(continuation_report_path),
            "initial_mesh": sha256_file(initial_mesh_path),
            "candidate_mesh": sha256_file(candidate_mesh_path),
            "train_evidence_plan": sha256_file(train_evidence_plan_path),
            "manifest": evidence_plan["input_hashes"]["manifest"],
            "joint_transforms": evidence_plan["input_hashes"]["joint_transforms"],
            "t05_solution": evidence_plan["input_hashes"]["t05_solution"],
            "development_normal_mask_evidence": development_digest,
        },
        "development_gates": D03_DEVELOPMENT_GATES,
        "evaluation_policy": {
            "development_records_bound": 36,
            "development_records_used_for_fit": 0,
            "single_evaluation_only": True,
            "post_development_tuning_allowed": False,
            "sealed_test_access": False,
        },
        "provenance": {
            "training_records_previously_scored": 144,
            "development_records_bound": 36,
            "development_records_used_for_fit": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
        },
    }


def write_d03_development_evaluation_plan(
    *,
    train_evaluation_path: Path,
    continuation_report_path: Path,
    initial_mesh_path: Path,
    candidate_mesh_path: Path,
    train_evidence_plan_path: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("D03 development evaluation plan is immutable")
    return write_json(
        output_path,
        build_d03_development_evaluation_plan(
            train_evaluation_path=train_evaluation_path,
            continuation_report_path=continuation_report_path,
            initial_mesh_path=initial_mesh_path,
            candidate_mesh_path=candidate_mesh_path,
            train_evidence_plan_path=train_evidence_plan_path,
            source_revision=source_revision,
        ),
    )


def evaluate_d03_development_images(*, plan_path: Path, output_path: Path) -> Path:
    """Run D03's one frozen 36-frame development evaluation without fitting."""

    reject_sealed_capability([plan_path, output_path])
    if output_path.exists():
        raise FileExistsError("D03 development evaluation is immutable")
    plan = read_json(plan_path)
    if (
        plan.get("schema_version") != D03_DEVELOPMENT_PLAN_SCHEMA
        or plan.get("experiment_id") != D03_EXPERIMENT_ID
        or plan.get("status") != "frozen_before_single_development_evaluation"
        or plan.get("development_gates") != D03_DEVELOPMENT_GATES
        or plan.get("evaluation_policy", {}).get("post_development_tuning_allowed") is not False
    ):
        raise ValueError("D03 development evaluator rejected its frozen plan")
    paths = {name: Path(path) for name, path in plan["input_paths"].items()}
    reject_sealed_capability([*paths.values()])
    for name in (
        "train_evaluation",
        "continuation_report",
        "initial_mesh",
        "candidate_mesh",
        "train_evidence_plan",
        "manifest",
        "joint_transforms",
        "t05_solution",
    ):
        if sha256_file(paths[name]) != plan["input_hashes"][name]:
            raise ValueError(f"D03 development input changed after planning: {name}")
    manifest = read_dataset_manifest(paths["manifest"])
    records = [record for record in manifest.frames if record.split == "held_out"]
    if len(records) != plan["evaluation_policy"]["development_records_bound"]:
        raise ValueError("D03 development evaluator requires all 36 held-out records")
    development_digest = _d03_split_evidence_digest(
        records, paths["normal_root"], paths["mask_root"]
    )
    if development_digest != plan["input_hashes"]["development_normal_mask_evidence"]:
        raise ValueError("D03 development evidence changed after planning")
    with np.load(paths["initial_mesh"], allow_pickle=False) as archive:
        control_vertices = archive["vertices"].astype(np.float32)
        control_faces = archive["faces"].astype(np.int64)
        control_weights = archive["skinning_weights"].astype(np.float32)
    with np.load(paths["candidate_mesh"], allow_pickle=False) as archive:
        treatment_vertices = archive["vertices"].astype(np.float32)
        treatment_faces = archive["faces"].astype(np.int64)
        treatment_weights = archive["skinning_weights"].astype(np.float32)
    with np.load(paths["joint_transforms"], allow_pickle=False) as archive:
        source_indices = archive["source_frame_indices"].astype(np.int64)
        transforms = archive["transforms"].astype(np.float32)
    transform_lookup = {int(source): slot for slot, source in enumerate(source_indices)}
    t05 = read_json(paths["t05_solution"])
    intrinsics = np.asarray(t05["shared_intrinsics"], dtype=np.float64)
    source_size = (manifest.video.height, manifest.video.width)
    control_vertex_tensor = torch.from_numpy(control_vertices)
    control_weight_tensor = torch.from_numpy(control_weights)
    treatment_vertex_tensor = torch.from_numpy(treatment_vertices)
    treatment_weight_tensor = torch.from_numpy(treatment_weights)
    control_ious: list[float] = []
    treatment_ious: list[float] = []
    control_boundaries: list[float] = []
    treatment_boundaries: list[float] = []
    control_normal_degrees: list[float] = []
    treatment_normal_degrees: list[float] = []
    for record in records:
        transform_slot = transform_lookup.get(record.source_frame_index)
        if transform_slot is None:
            raise ValueError("D03 development evaluator lacks a joint transform")
        transform = torch.from_numpy(transforms[transform_slot])
        with torch.no_grad():
            posed_control = linear_blend_skinning(
                control_vertex_tensor, control_weight_tensor, transform
            ).numpy()
            posed_treatment = linear_blend_skinning(
                treatment_vertex_tensor, treatment_weight_tensor, transform
            ).numpy()
        name = Path(record.image_path).name
        normal_image = cv2.imread(str(paths["normal_root"] / name), cv2.IMREAD_COLOR)
        mask_image = cv2.imread(str(paths["mask_root"] / name), cv2.IMREAD_GRAYSCALE)
        if normal_image is None or mask_image is None:
            raise FileNotFoundError(f"D03 development evaluator lacks evidence: {name}")
        target_mask = mask_image > 127
        target_normal = decode_sapiens_normal_bgr(normal_image)
        control_mask, control_normal = _render_hard_geometry_normals(
            posed_control, control_faces, intrinsics, source_size
        )
        treatment_mask, treatment_normal = _render_hard_geometry_normals(
            posed_treatment, treatment_faces, intrinsics, source_size
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
        target_valid = np.linalg.norm(target_normal, axis=-1) > 0.5
        control_valid = eroded & control_mask & target_valid
        treatment_valid = eroded & treatment_mask & target_valid
        if not np.any(control_valid) or not np.any(treatment_valid):
            raise ValueError("D03 development evaluator found no valid normal pixels")
        control_normal_degrees.append(
            float(
                np.median(
                    _angular_degrees(control_normal[control_valid], target_normal[control_valid])
                )
            )
        )
        treatment_normal_degrees.append(
            float(
                np.median(
                    _angular_degrees(
                        treatment_normal[treatment_valid], target_normal[treatment_valid]
                    )
                )
            )
        )
    metrics = {
        "control_held_out_iou": float(np.median(control_ious)),
        "treatment_held_out_iou": float(np.median(treatment_ious)),
        "control_boundary_error": float(np.median(control_boundaries)),
        "treatment_boundary_error": float(np.median(treatment_boundaries)),
        "control_median_normal_degrees": float(np.median(control_normal_degrees)),
        "treatment_median_normal_degrees": float(np.median(treatment_normal_degrees)),
    }
    train = read_json(paths["train_evaluation"])
    treatment_train_iou = float(train["metrics"]["treatment_median_hard_iou"])
    train_held_out_gap = abs(treatment_train_iou - metrics["treatment_held_out_iou"])
    iou_regression = metrics["control_held_out_iou"] - metrics["treatment_held_out_iou"]
    boundary_regression = metrics["treatment_boundary_error"] - metrics["control_boundary_error"]
    normal_regression = (
        metrics["treatment_median_normal_degrees"] - metrics["control_median_normal_degrees"]
    )
    thresholds = plan["development_gates"]
    continuation = read_json(paths["continuation_report"])
    gates = {
        "held_out_iou": metrics["treatment_held_out_iou"]
        >= float(thresholds["held_out_iou_minimum"]),
        "boundary_error": metrics["treatment_boundary_error"]
        <= float(thresholds["boundary_error_maximum"]),
        "median_normal_degrees": metrics["treatment_median_normal_degrees"]
        <= float(thresholds["median_normal_degrees_maximum"]),
        "train_held_out_gap": train_held_out_gap <= float(thresholds["train_held_out_gap_maximum"]),
        "relative_iou_nonregression": iou_regression
        <= float(thresholds["relative_iou_regression_maximum"]),
        "relative_boundary_nonregression": boundary_regression
        <= float(thresholds["relative_boundary_regression_maximum"]),
        "relative_normal_nonregression": normal_regression
        <= float(thresholds["relative_normal_regression_degrees_maximum"]),
        "exact_topology_prepassed": continuation["candidate_topology"]["status"] == "pass",
        "development_not_used_for_fit": plan["evaluation_policy"][
            "development_records_used_for_fit"
        ]
        == 0,
        "sealed_test_reads": plan["evaluation_policy"]["sealed_test_access"] is False,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_d03_development_evaluation.v1",
            "experiment_id": D03_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "decision": "eligible_for_d03_qualification_audit"
            if not blockers
            else "terminal_failed_single_development_evaluation",
            "source_revision": plan["source_revision"],
            "metrics": metrics,
            "treatment_train_iou": treatment_train_iou,
            "train_held_out_iou_gap": train_held_out_gap,
            "relative_iou_regression": iou_regression,
            "relative_boundary_regression": boundary_regression,
            "relative_normal_regression_degrees": normal_regression,
            "gates": gates,
            "blockers": blockers,
            "training_records_previously_read": 144,
            "development_records_read": len(records),
            "development_records_used_for_fit": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "post_development_tuning_allowed": False,
            "input_hashes": plan["input_hashes"],
            "plan_sha256": sha256_file(plan_path),
        },
    )


def _human_capsules(*, treatment: bool) -> tuple[Capsule, ...]:
    perturb = 1.0 if treatment else 0.0
    return (
        Capsule((0.0, -0.38, 0.0), (0.0, 0.38, 0.0), 0.255 - 0.008 * perturb, "torso"),
        Capsule((0.0, 0.34, 0.0), (0.0, 0.67, 0.0), 0.115, "neck"),
        Capsule((0.0, 0.66, 0.0), (0.0, 0.88, 0.0), 0.185 - 0.004 * perturb, "head"),
        Capsule((-0.17, 0.30, 0.0), (-0.62, 0.23, 0.01), 0.105, "left_upper_arm"),
        Capsule((-0.59, 0.23, 0.01), (-0.98, 0.04, 0.025), 0.085, "left_lower_arm"),
        Capsule((0.17, 0.30, 0.0), (0.62, 0.23, -0.01), 0.105, "right_upper_arm"),
        Capsule((0.59, 0.23, -0.01), (0.98, 0.04, -0.025), 0.085, "right_lower_arm"),
        Capsule((-0.115, -0.30, 0.0), (-0.17, -0.78, 0.01), 0.13, "left_upper_leg"),
        Capsule((-0.17, -0.74, 0.01), (-0.18, -1.17, 0.04), 0.105, "left_lower_leg"),
        Capsule((0.115, -0.30, 0.0), (0.17, -0.78, -0.01), 0.13, "right_upper_leg"),
        Capsule((0.17, -0.74, -0.01), (0.18, -1.17, -0.04), 0.105, "right_lower_leg"),
    )


def _extract_field_surface(field: np.ndarray, *, extent: float) -> tuple[np.ndarray, np.ndarray]:
    if field.ndim != 3 or field.shape[0] != field.shape[1] or field.shape[1] != field.shape[2]:
        raise ValueError("D03 extraction requires one cubic scalar grid")
    if float(field.min()) >= 0.0 or float(field.max()) <= 0.0:
        raise ValueError("D03 field does not cross zero")
    pitch = 2.0 * extent / (field.shape[0] - 1)
    vertices, faces, _, _ = marching_cubes(  # type: ignore[no-untyped-call]
        field.astype(np.float32),
        level=0.0,
        spacing=(pitch, pitch, pitch),
        gradient_direction="descent",
        allow_degenerate=False,
    )
    vertices += np.asarray([-extent, -extent, -extent])
    return vertices.astype(np.float64), faces.astype(np.int64)


def _field_for_capsules(
    capsules: tuple[Capsule, ...], *, resolution: int, extent: float
) -> np.ndarray:
    coordinates = np.linspace(-extent, extent, resolution)
    field = np.empty((resolution, resolution, resolution), dtype=np.float32)
    yy, zz = np.meshgrid(coordinates, coordinates, indexing="ij")
    for x_index, x in enumerate(coordinates):
        points = np.stack((np.full_like(yy, x), yy, zz), axis=-1)
        field[x_index] = capsule_tree_signed_distance(points, capsules).astype(np.float32)
    return field


def _ellipsoid_field(*, resolution: int, extent: float) -> np.ndarray:
    coordinates = np.linspace(-extent, extent, resolution)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    return (
        np.square(xx / 1.08) + np.square((yy + 0.14) / 1.18) + np.square(zz / 0.27) - 1.0
    ).astype(np.float32)


def _mesh_audit(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    intersections = ipctk_has_self_intersections(vertices, faces)
    outward = bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0.0)
    checks = {
        "one_component": len(components) == 1,
        "euler_two": int(mesh.euler_number) == 2,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "outward": outward,
        "exact_self_intersection_free": not intersections,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "component_count": len(components),
        "euler_number": int(mesh.euler_number),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "outward": outward,
        "exact_self_intersections": intersections,
        "exact_collision_backend": "ipctk_1.6.0_has_intersections",
        "checks": checks,
        "cleanup_operations": 0,
    }


def _bidirectional_vertex_error(first: np.ndarray, second: np.ndarray) -> float:
    first_tree = cKDTree(np.asarray(first, dtype=np.float64))
    second_tree = cKDTree(np.asarray(second, dtype=np.float64))
    first_to_second = second_tree.query(first, workers=1)[0]
    second_to_first = first_tree.query(second, workers=1)[0]
    return float(0.5 * (np.mean(first_to_second) + np.mean(second_to_first)))


def run_d03_public_benchmark(*, resolution: int = 80, extent: float = 1.35) -> dict[str, Any]:
    """Qualify a new implicit topology lineage on analytic public geometry."""

    if resolution < 48 or extent <= 1.2:
        raise ValueError("D03 public grid does not cover the registered fixture")
    truth_capsules = _human_capsules(treatment=False)
    treatment_capsules = _human_capsules(treatment=True)
    truth_field = _field_for_capsules(truth_capsules, resolution=resolution, extent=extent)
    treatment_field = _field_for_capsules(treatment_capsules, resolution=resolution, extent=extent)
    ellipsoid_field = _ellipsoid_field(resolution=resolution, extent=extent)
    truth_vertices, _truth_faces = _extract_field_surface(truth_field, extent=extent)
    treatment_vertices, treatment_faces = _extract_field_surface(treatment_field, extent=extent)
    ellipsoid_vertices, ellipsoid_faces = _extract_field_surface(ellipsoid_field, extent=extent)
    treatment_audit = _mesh_audit(treatment_vertices, treatment_faces)
    ellipsoid_audit = _mesh_audit(ellipsoid_vertices, ellipsoid_faces)
    treatment_error = _bidirectional_vertex_error(treatment_vertices, truth_vertices)
    ellipsoid_error = _bidirectional_vertex_error(ellipsoid_vertices, truth_vertices)
    error_improvement = 1.0 - treatment_error / ellipsoid_error

    adversarial_capsules = (
        *treatment_capsules,
        Capsule((1.18, 1.12, 1.08), (1.18, 1.12, 1.08), 0.10, "disconnected_adversary"),
    )
    adversarial_field = _field_for_capsules(
        adversarial_capsules, resolution=resolution, extent=extent
    )
    adversarial_vertices, adversarial_faces = _extract_field_surface(
        adversarial_field, extent=extent
    )
    adversarial_audit = _mesh_audit(adversarial_vertices, adversarial_faces)
    identity_capsule = Capsule((0.0, -0.2, 0.0), (0.0, 0.2, 0.0), 0.3, "identity")
    identity_points = np.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0]])
    identity_expected = np.asarray([-0.3, 0.0, 0.2])
    identity_error = float(
        np.max(
            np.abs(capsule_signed_distance(identity_points, identity_capsule) - identity_expected)
        )
    )
    replay_field = _field_for_capsules(treatment_capsules, resolution=resolution, extent=extent)
    replay_vertices, replay_faces = _extract_field_surface(replay_field, extent=extent)
    exact_replay = (
        np.array_equal(treatment_field, replay_field)
        and np.array_equal(treatment_vertices, replay_vertices)
        and np.array_equal(treatment_faces, replay_faces)
    )
    gates = {
        "exact_capsule_sdf_identity": identity_error <= 1.0e-12,
        "capsule_tree_closed_embedded_euler2": treatment_audit["status"] == "pass",
        "ellipsoid_control_topology_valid": ellipsoid_audit["status"] == "pass",
        "truth_error_improvement": error_improvement >= 0.20,
        "topology_changing_proposal_rejected": adversarial_audit["status"] == "fail"
        and adversarial_audit["component_count"] > 1,
        "exact_replay": exact_replay,
    }
    return {
        "schema_version": D03_PUBLIC_SCHEMA,
        "experiment_id": D03_EXPERIMENT_ID,
        "status": "pass" if all(gates.values()) else "fail",
        "fixture": {
            "resolution": resolution,
            "extent": extent,
            "capsule_count": len(treatment_capsules),
            "generated_views_used_as_project_evidence": False,
        },
        "exact_capsule_identity_maximum_error": identity_error,
        "treatment_topology": treatment_audit,
        "ellipsoid_control_topology": ellipsoid_audit,
        "adversarial_topology": adversarial_audit,
        "treatment_bidirectional_vertex_error": treatment_error,
        "ellipsoid_bidirectional_vertex_error": ellipsoid_error,
        "truth_error_relative_improvement": error_improvement,
        "gates": gates,
        "provenance": {
            "private_records_read": 0,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "cleanup_operations": 0,
        },
    }


def write_d03_public_benchmark(output_path: Path) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("D03 public benchmark output is immutable")
    return write_json(output_path, run_d03_public_benchmark())
