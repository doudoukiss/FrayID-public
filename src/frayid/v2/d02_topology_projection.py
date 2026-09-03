from __future__ import annotations

import math
import os
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

from frayid.config import ReconstructionConfig
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.d01_pose_normal_depth import evaluate_d01_train_mesh_candidate
from frayid.v2.g03_pipeline import _load_inputs
from frayid.v2.posed_preview import load_frozen_v1_model

D02_EXPERIMENT_ID = "postv2_d02_topology_constrained_normal_projection_r01"
D02_PUBLIC_SCHEMA = "frayid_v2_d02_public_topology_constrained_projection.v1"
D02_TRAIN_PROJECTION_SCHEDULE: dict[str, float | int] = {
    "minimum_area_ratio": 0.10,
    "maximum_rejections": 48,
    "shrink_factor": 0.5,
    "minimum_vertex_scale": 2.0**-24,
}
D02_TRAIN_GATES: dict[str, float] = {
    "median_normal_improvement_degrees_minimum": 1.0,
    "median_hard_iou_regression_maximum": 0.005,
    "median_boundary_error_regression_maximum": 0.001,
}


def _normalize(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths <= 1.0e-12) or not np.all(np.isfinite(lengths)):
        raise ValueError("D02 vectors must be finite and nonzero")
    return np.asarray(vectors / lengths, dtype=np.float64)


def _face_constraint_quantities(
    reference_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(reference_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("D02 reference and candidate vertices must share shape [V,3]")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("D02 faces must have shape [F,3]")
    reference_triangles = reference[triangles]
    candidate_triangles = candidate[triangles]
    reference_area = np.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
    )
    candidate_area = np.cross(
        candidate_triangles[:, 1] - candidate_triangles[:, 0],
        candidate_triangles[:, 2] - candidate_triangles[:, 0],
    )
    reference_length = np.linalg.norm(reference_area, axis=-1)
    candidate_length = np.linalg.norm(candidate_area, axis=-1)
    if np.any(reference_length <= 1.0e-12):
        raise ValueError("D02 reference contains a degenerate face")
    cosine = np.sum(reference_area * candidate_area, axis=-1) / np.maximum(
        reference_length * candidate_length,
        1.0e-20,
    )
    unsigned_ratio = candidate_length / reference_length
    signed_ratio = np.sum(reference_area * candidate_area, axis=-1) / np.square(reference_length)
    return cosine, unsigned_ratio, signed_ratio


def exact_face_constraint_audit(
    reference_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_area_ratio: float = 0.10,
) -> dict[str, Any]:
    """Audit orientation and relative area without mesh cleanup or remeshing."""

    if minimum_area_ratio <= 0.0:
        raise ValueError("D02 area floor must be positive")
    cosine, unsigned_ratio, signed_ratio = _face_constraint_quantities(
        reference_vertices, candidate_vertices, faces
    )
    invalid_orientation = cosine <= 0.0
    invalid_area = unsigned_ratio < minimum_area_ratio
    invalid = invalid_orientation | invalid_area
    return {
        "status": "pass" if not np.any(invalid) else "fail",
        "face_count": len(cosine),
        "invalid_face_count": int(np.count_nonzero(invalid)),
        "flipped_face_count": int(np.count_nonzero(invalid_orientation)),
        "collapsed_face_count": int(np.count_nonzero(invalid_area)),
        "minimum_orientation_cosine": float(cosine.min()),
        "minimum_unsigned_area_ratio": float(unsigned_ratio.min()),
        "minimum_signed_area_ratio": float(signed_ratio.min()),
        "minimum_area_ratio_gate": minimum_area_ratio,
        "cleanup_operations": 0,
    }


def topology_constrained_local_trust_projection(
    reference_vertices: np.ndarray,
    raw_target_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_area_ratio: float = 0.10,
    maximum_rejections: int = 48,
    shrink_factor: float = 0.5,
    minimum_vertex_scale: float = 2.0**-24,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Project one raw update through deterministic local topology trust radii."""

    reference = np.asarray(reference_vertices, dtype=np.float64)
    raw = np.asarray(raw_target_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if reference.shape != raw.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("D02 trust projection inputs must share shape [V,3]")
    if maximum_rejections < 1 or not 0.0 < shrink_factor < 1.0:
        raise ValueError("D02 trust projection settings are invalid")
    if not 0.0 < minimum_vertex_scale < 1.0:
        raise ValueError("D02 minimum vertex scale is invalid")
    scales = np.ones(len(reference), dtype=np.float64)
    rejected: list[dict[str, Any]] = []
    candidate = raw.copy()
    for attempt in range(maximum_rejections + 1):
        candidate = reference + scales[:, None] * (raw - reference)
        cosine, area_ratio, _ = _face_constraint_quantities(reference, candidate, triangles)
        invalid = (cosine <= 0.0) | (area_ratio < minimum_area_ratio)
        audit = exact_face_constraint_audit(
            reference,
            candidate,
            triangles,
            minimum_area_ratio=minimum_area_ratio,
        )
        if not np.any(invalid):
            accepted = {
                "attempt_index": attempt,
                "audit": audit,
                "minimum_vertex_scale": float(scales.min()),
                "median_vertex_scale": float(np.median(scales)),
                "full_scale_vertex_fraction": float(np.mean(scales == 1.0)),
            }
            return (
                candidate,
                scales,
                {
                    "status": "pass",
                    "rejected_proposal_count": len(rejected),
                    "rejected_proposals": rejected,
                    "accepted_steps": [accepted],
                    "every_accepted_step_safe": audit["status"] == "pass",
                    "cleanup_operations": 0,
                    "connectivity_changes": 0,
                },
            )
        if attempt == maximum_rejections:
            break
        affected = np.unique(triangles[invalid].reshape(-1))
        previous = scales[affected].copy()
        scales[affected] *= shrink_factor
        rejected.append(
            {
                "attempt_index": attempt,
                "invalid_face_count": int(np.count_nonzero(invalid)),
                "affected_vertex_count": len(affected),
                "minimum_area_ratio": float(area_ratio.min()),
                "minimum_orientation_cosine": float(cosine.min()),
            }
        )
        if np.any(scales[affected] < minimum_vertex_scale) or np.array_equal(
            previous, scales[affected]
        ):
            break
    raise RuntimeError("D02 local trust projection could not find a topology-safe update")


def _height_mesh(height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(height, dtype=np.float64)
    rows, columns = values.shape
    yy, xx = np.meshgrid(
        np.linspace(-0.8, 0.8, rows),
        np.linspace(-1.0, 1.0, columns),
        indexing="ij",
    )
    vertices = np.stack((xx, yy, values), axis=-1).reshape(-1, 3)
    faces: list[tuple[int, int, int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            top_left = row * columns + column
            top_right = top_left + 1
            bottom_left = top_left + columns
            bottom_right = bottom_left + 1
            faces.extend(
                (
                    (top_left, top_right, bottom_right),
                    (top_left, bottom_right, bottom_left),
                )
            )
    return vertices, np.asarray(faces, dtype=np.int64)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    result = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(result, faces[:, corner], face_normals)
    return _normalize(result)


def _geometry_metrics(
    candidate: np.ndarray,
    truth: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float]:
    position = np.linalg.norm(np.asarray(candidate) - np.asarray(truth), axis=-1)
    normal_error = np.degrees(
        np.arccos(
            np.clip(
                np.sum(_vertex_normals(candidate, faces) * _vertex_normals(truth, faces), axis=-1),
                -1.0,
                1.0,
            )
        )
    )
    return {
        "position_rmse": float(np.sqrt(np.mean(np.square(position)))),
        "median_normal_degrees": float(np.median(normal_error)),
        "p95_normal_degrees": float(np.percentile(normal_error, 95.0)),
    }


def run_d02_public_benchmark(*, seed: int = 20260903) -> dict[str, Any]:
    """Test local topology constraints on an analytic surface with an injected defect."""

    generator = np.random.default_rng(seed)
    rows, columns = 23, 29
    y = np.linspace(-0.8, 0.8, rows)
    x = np.linspace(-1.0, 1.0, columns)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    truth_height = (
        0.085 * np.sin(2.0 * math.pi * xx) * np.cos(1.6 * math.pi * yy)
        + 0.04 * np.exp(-((xx - 0.25) ** 2 + (yy + 0.15) ** 2) / 0.04)
        + 0.014 * xx
    )
    prior_height = gaussian_filter(truth_height, sigma=2.3)
    prior, faces = _height_mesh(prior_height)
    truth, truth_faces = _height_mesh(truth_height)
    if not np.array_equal(faces, truth_faces):
        raise AssertionError("D02 public fixture connectivity changed")
    raw = truth.copy()
    raw += generator.normal(0.0, 0.0002, size=raw.shape)
    center = (rows // 2) * columns + columns // 2
    raw[center, 0] += 0.42
    raw[center, 1] -= 0.31
    raw[center, 2] += 0.24
    collapsed = center + columns + 5
    raw[collapsed] = raw[collapsed - 1]
    raw_audit = exact_face_constraint_audit(prior, raw, faces)
    treatment, scales, projection = topology_constrained_local_trust_projection(
        prior,
        raw,
        faces,
    )
    replay, replay_scales, replay_projection = topology_constrained_local_trust_projection(
        prior,
        raw,
        faces,
    )
    treatment_audit = exact_face_constraint_audit(prior, treatment, faces)
    prior_metrics = _geometry_metrics(prior, truth, faces)
    treatment_metrics = _geometry_metrics(treatment, truth, faces)
    position_improvement = 1.0 - treatment_metrics["position_rmse"] / prior_metrics["position_rmse"]
    normal_improvement = (
        prior_metrics["median_normal_degrees"] - treatment_metrics["median_normal_degrees"]
    )
    exact_replay = (
        np.array_equal(treatment, replay)
        and np.array_equal(scales, replay_scales)
        and projection == replay_projection
    )
    gates = {
        "injected_flip_and_collapse_detected": raw_audit["status"] == "fail"
        and raw_audit["flipped_face_count"] > 0
        and raw_audit["collapsed_face_count"] > 0,
        "unsafe_proposal_rejected": projection["rejected_proposal_count"] > 0,
        "every_accepted_step_safe": projection["every_accepted_step_safe"] is True,
        "final_topology_safe": treatment_audit["status"] == "pass",
        "position_error_improvement": position_improvement >= 0.10,
        "normal_improvement": normal_improvement >= 2.0,
        "exact_replay": exact_replay,
        "connectivity_frozen": projection["connectivity_changes"] == 0,
    }
    return {
        "schema_version": D02_PUBLIC_SCHEMA,
        "experiment_id": D02_EXPERIMENT_ID,
        "status": "pass" if all(gates.values()) else "fail",
        "seed": seed,
        "fixture": {
            "vertex_count": len(prior),
            "face_count": len(faces),
            "injected_vertex_index": center,
            "collapsed_vertex_index": collapsed,
            "generated_views_used_as_project_evidence": False,
            "private_records_read": 0,
        },
        "raw_unconstrained_audit": raw_audit,
        "topology_constrained_audit": treatment_audit,
        "projection": projection,
        "prior_control": prior_metrics,
        "treatment": treatment_metrics,
        "position_rmse_relative_improvement": position_improvement,
        "median_normal_improvement_degrees": normal_improvement,
        "gates": gates,
        "provenance": {
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "cleanup_operations": 0,
        },
    }


def write_d02_public_benchmark(output_path: Path, *, seed: int = 20260903) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("D02 public benchmark output is immutable")
    return write_json(output_path, run_d02_public_benchmark(seed=seed))


def write_d02_train_projection_plan(
    public_benchmark_path: Path,
    d01_terminal_path: Path,
    d01_evidence_report_path: Path,
    output_path: Path,
    *,
    source_revision: str,
) -> Path:
    """Freeze D02's private projection and evaluator before its candidate exists."""

    paths = [public_benchmark_path, d01_terminal_path, d01_evidence_report_path, output_path]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("D02 train projection plan is immutable")
    public = read_json(public_benchmark_path)
    terminal = read_json(d01_terminal_path)
    evidence = read_json(d01_evidence_report_path)
    if public.get("status") != "pass" or public.get("experiment_id") != D02_EXPERIMENT_ID:
        raise ValueError("D02 plan requires its passing public benchmark")
    if terminal.get("decision") != "terminal_failed_train_topology_precheck":
        raise ValueError("D02 plan requires the immutable terminal D01 control")
    if (
        evidence.get("status") != "train_only_evidence_bound"
        or evidence.get("training_records_read") != 144
        or evidence.get("development_records_read") != 0
    ):
        raise ValueError("D02 plan requires the clean D01 train evidence binding")
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_d02_train_projection_plan.v1",
            "experiment_id": D02_EXPERIMENT_ID,
            "status": "frozen_before_real_projection",
            "source_revision": source_revision,
            "projection_schedule": D02_TRAIN_PROJECTION_SCHEDULE,
            "training_gates": D02_TRAIN_GATES,
            "raw_proposal_role": "immutable_failed_d01_candidate_not_a_passing_dependency",
            "connectivity_policy": "frozen_exactly",
            "cleanup_operations_authorized": 0,
            "development_records_authorized": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "automatic_retries": 0,
            "input_hashes": {
                "public_benchmark": sha256_file(public_benchmark_path),
                "d01_terminal": sha256_file(d01_terminal_path),
                "d01_evidence_report": sha256_file(d01_evidence_report_path),
            },
        },
    )


def fit_d02_train_topology_projection(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    d01_raw_candidate_path: Path,
    d01_candidate_report_path: Path,
    d01_terminal_path: Path,
    projection_plan_path: Path,
    output_root: Path,
    source_revision: str,
) -> Path:
    """Project the frozen D01 proposal through D02's in-loop topology gate."""

    paths = [
        checkpoint_path,
        manifest_path,
        joint_transforms_path,
        d01_raw_candidate_path,
        d01_candidate_report_path,
        d01_terminal_path,
        projection_plan_path,
        output_root,
    ]
    reject_sealed_capability(paths)
    if output_root.exists():
        raise FileExistsError("D02 train projection output is immutable")
    raw_report = read_json(d01_candidate_report_path)
    terminal = read_json(d01_terminal_path)
    plan = read_json(projection_plan_path)
    if (
        raw_report.get("status") != "candidate_failed_precheck"
        or raw_report.get("artifacts", {}).get("bounded_canonical_mesh_candidate", {}).get("sha256")
        != sha256_file(d01_raw_candidate_path)
        or terminal.get("artifact_hashes", {}).get("candidate_report")
        != sha256_file(d01_candidate_report_path)
    ):
        raise ValueError("D02 rejected its immutable D01 failure control")
    if (
        plan.get("status") != "frozen_before_real_projection"
        or plan.get("projection_schedule") != D02_TRAIN_PROJECTION_SCHEDULE
        or plan.get("training_gates") != D02_TRAIN_GATES
    ):
        raise ValueError("D02 rejected an altered projection plan")
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
        raise ValueError("D02 requires the frozen 144-frame training split")
    reference = model.canonical_vertices.detach().cpu().numpy().astype(np.float64)
    reference_faces = model.faces.detach().cpu().numpy().astype(np.int64)
    with np.load(d01_raw_candidate_path, allow_pickle=False) as archive:
        raw = archive["vertices"].astype(np.float64)
        raw_faces = archive["faces"].astype(np.int64)
    if not np.array_equal(reference_faces, raw_faces):
        raise ValueError("D02 raw proposal connectivity differs from V1")
    schedule = D02_TRAIN_PROJECTION_SCHEDULE
    candidate, scales, projection = topology_constrained_local_trust_projection(
        reference,
        raw,
        reference_faces,
        minimum_area_ratio=float(schedule["minimum_area_ratio"]),
        maximum_rejections=int(schedule["maximum_rejections"]),
        shrink_factor=float(schedule["shrink_factor"]),
        minimum_vertex_scale=float(schedule["minimum_vertex_scale"]),
    )
    replay, replay_scales, replay_projection = topology_constrained_local_trust_projection(
        reference,
        raw,
        reference_faces,
        minimum_area_ratio=float(schedule["minimum_area_ratio"]),
        maximum_rejections=int(schedule["maximum_rejections"]),
        shrink_factor=float(schedule["shrink_factor"]),
        minimum_vertex_scale=float(schedule["minimum_vertex_scale"]),
    )
    exact_replay = (
        np.array_equal(candidate, replay)
        and np.array_equal(scales, replay_scales)
        and projection == replay_projection
    )
    audit = exact_face_constraint_audit(
        reference,
        candidate,
        reference_faces,
        minimum_area_ratio=float(schedule["minimum_area_ratio"]),
    )
    blockers = []
    if audit["status"] != "pass":
        blockers.append("exact_face_constraint_audit")
    if not exact_replay:
        blockers.append("projection_replay")
    if projection["rejected_proposal_count"] < 1:
        blockers.append("unsafe_d01_proposal_not_rejected")
    output_root.mkdir(parents=True, exist_ok=False)
    candidate_path = output_root / "topology_constrained_canonical_candidate.npz"
    temporary_path = output_root / ".topology_constrained_canonical_candidate.tmp"
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            vertices=candidate.astype(np.float32),
            faces=reference_faces,
            local_trust_scales=scales.astype(np.float32),
        )
    os.replace(temporary_path, candidate_path)
    displacement = np.linalg.norm(candidate - reference, axis=-1)
    report = {
        "schema_version": "frayid_v2_d02_train_projection_candidate.v1",
        "experiment_id": D02_EXPERIMENT_ID,
        "status": "candidate_complete" if not blockers else "candidate_failed_precheck",
        "source_revision": source_revision,
        "training_records_bound": 144,
        "development_records_read": 0,
        "sealed_test_reads": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "projection_schedule": schedule,
        "projection": projection,
        "exact_projection_replay": exact_replay,
        "connectivity_exactly_frozen": bool(np.array_equal(reference_faces, raw_faces)),
        "topology_precheck": audit,
        "displacement_metres": {
            "median": float(np.median(displacement)),
            "p95": float(np.percentile(displacement, 95.0)),
            "maximum": float(displacement.max()),
        },
        "local_trust_scale": {
            "minimum": float(scales.min()),
            "median": float(np.median(scales)),
            "full_scale_vertex_fraction": float(np.mean(scales == 1.0)),
        },
        "blockers": blockers,
        "input_hashes": {
            "checkpoint": sha256_file(checkpoint_path),
            "manifest": sha256_file(manifest_path),
            "joint_transforms": sha256_file(joint_transforms_path),
            "d01_raw_candidate": sha256_file(d01_raw_candidate_path),
            "d01_candidate_report": sha256_file(d01_candidate_report_path),
            "d01_terminal": sha256_file(d01_terminal_path),
            "projection_plan": sha256_file(projection_plan_path),
        },
        "artifacts": {
            "bounded_canonical_mesh_candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            }
        },
    }
    return write_json(output_root / "train_projection_candidate_report.json", report)


def evaluate_d02_train_candidate(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    manifest_path: Path,
    joint_transforms_path: Path,
    t05_solution_path: Path,
    candidate_path: Path,
    candidate_report_path: Path,
    projection_plan_path: Path,
    normal_root: Path,
    mask_root: Path,
    output_path: Path,
    source_revision: str,
) -> Path:
    """Use the frozen independent raster evaluator with D02 identity and gates."""

    return evaluate_d01_train_mesh_candidate(
        config=config,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        joint_transforms_path=joint_transforms_path,
        t05_solution_path=t05_solution_path,
        candidate_path=candidate_path,
        candidate_report_path=candidate_report_path,
        candidate_plan_path=projection_plan_path,
        normal_root=normal_root,
        mask_root=mask_root,
        output_path=output_path,
        source_revision=source_revision,
        experiment_id=D02_EXPERIMENT_ID,
        report_schema_version="frayid_v2_d02_train_candidate_evaluation.v1",
        training_gates=D02_TRAIN_GATES,
    )


def ipctk_has_self_intersections(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Run IPC Toolkit's exact static self-intersection predicate."""

    ipctk = import_module("ipctk")
    points = np.asfortranarray(vertices, dtype=np.float64)
    triangles = np.asfortranarray(faces, dtype=np.int32)
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    edges = np.asfortranarray(np.unique(np.sort(edges, axis=1), axis=0), dtype=np.int32)
    collision_mesh = ipctk.CollisionMesh(points, edges, triangles)
    return bool(ipctk.has_intersections(collision_mesh, points))


def audit_d02_exact_topology(
    *,
    config: ReconstructionConfig,
    checkpoint_path: Path,
    candidate_path: Path,
    candidate_report_path: Path,
    train_evaluation_path: Path,
    output_path: Path,
) -> Path:
    """Fail D02 before development when the exact collision predicate is positive."""

    paths = [
        checkpoint_path,
        candidate_path,
        candidate_report_path,
        train_evaluation_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("D02 exact topology audit is immutable")
    candidate_report = read_json(candidate_report_path)
    train_evaluation = read_json(train_evaluation_path)
    if (
        candidate_report.get("status") != "candidate_complete"
        or candidate_report.get("artifacts", {})
        .get("bounded_canonical_mesh_candidate", {})
        .get("sha256")
        != sha256_file(candidate_path)
        or train_evaluation.get("status") != "pass"
        or train_evaluation.get("development_records_read") != 0
    ):
        raise ValueError("D02 exact topology audit rejected its candidate lifecycle")
    control_model, _checkpoint = load_frozen_v1_model(checkpoint_path, config)
    reference = control_model.canonical_vertices.detach().cpu().numpy().astype(np.float64)
    reference_faces = control_model.faces.detach().cpu().numpy().astype(np.int64)
    with np.load(candidate_path, allow_pickle=False) as archive:
        candidate = archive["vertices"].astype(np.float64)
        candidate_faces = archive["faces"].astype(np.int64)
    if not np.array_equal(candidate_faces, reference_faces):
        raise ValueError("D02 exact topology audit detected changed connectivity")
    reference_intersections = ipctk_has_self_intersections(reference, reference_faces)
    candidate_intersections = ipctk_has_self_intersections(candidate, candidate_faces)
    mesh = trimesh.Trimesh(vertices=candidate, faces=candidate_faces, process=False)
    components = mesh.split(only_watertight=False)
    structural = {
        "component_count": len(components),
        "euler_number": int(mesh.euler_number),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "outward": bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0.0),
    }
    checks = {
        "train_evaluation_passed": True,
        "development_not_read": train_evaluation["development_records_read"] == 0,
        "connectivity_frozen": True,
        "component_policy": structural["component_count"] == 1,
        "euler_policy": structural["euler_number"] == 2,
        "watertight": structural["watertight"],
        "winding_consistent": structural["winding_consistent"],
        "outward": structural["outward"],
        "candidate_has_no_self_intersections": not candidate_intersections,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    decision = (
        "eligible_for_development_freeze"
        if not blockers
        else "terminal_failed_exact_self_intersection_predevelopment"
    )
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_d02_exact_topology_audit.v1",
            "experiment_id": D02_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "decision": decision,
            "exact_collision_backend": "ipctk_1.6.0_has_intersections",
            "reference_has_self_intersections": reference_intersections,
            "candidate_has_self_intersections": candidate_intersections,
            "exact_intersection_pair_count": None,
            "exact_intersection_pair_count_note": "backend_returns_an_exact_boolean_predicate_not_a_pair_enumeration",
            "structural_topology": structural,
            "checks": checks,
            "blockers": blockers,
            "training_records_read": train_evaluation["training_records_read"],
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "input_hashes": {
                "checkpoint": sha256_file(checkpoint_path),
                "candidate": sha256_file(candidate_path),
                "candidate_report": sha256_file(candidate_report_path),
                "train_evaluation": sha256_file(train_evaluation_path),
            },
        },
    )
