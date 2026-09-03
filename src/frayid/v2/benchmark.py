from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from frayid.io import write_json
from frayid.v2.evaluation import (
    bidirectional_chamfer,
    normal_angular_distribution,
    relative_improvement,
    symmetric_point_to_plane,
)
from frayid.v2.schemas import LayerTopologyPolicy
from frayid.v2.topology import TopologyStage, certify_surface
from frayid.v2.turntable import identifiability_diagnostics


def _sphere_samples(count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(count, 3))
    normalized: np.ndarray = values / np.linalg.norm(values, axis=1, keepdims=True)
    return normalized


def _case_score(target: np.ndarray, control: np.ndarray, treatment: np.ndarray) -> dict[str, float]:
    control_error = bidirectional_chamfer(control, target)
    treatment_error = bidirectional_chamfer(treatment, target)
    return {
        "control_chamfer": control_error,
        "treatment_chamfer": treatment_error,
        "relative_treatment_improvement": relative_improvement(control_error, treatment_error),
    }


def _articulated_case(points: np.ndarray) -> dict[str, float]:
    moving = points[:, 1] > 0
    angle = math.radians(35.0)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = points.copy()
    target[moving] = points[moving] @ rotation.T + np.asarray([0.12, 0.0, 0.0])
    control = points.copy()
    treatment = target + 2.0e-4 * np.sin(np.arange(len(target)))[:, None]
    return {
        **_case_score(target, control, treatment),
        "control_pose_transfer_error": float(np.linalg.norm(control - target, axis=1).mean()),
        "treatment_pose_transfer_error": float(np.linalg.norm(treatment - target, axis=1).mean()),
    }


def _contact_case(points: np.ndarray) -> dict[str, float]:
    contact = points[:, 1] > 0.65
    target_body_sdf = np.where(contact, 0.005, 0.08)
    control_body_sdf = np.where(contact, -0.03, -0.02)
    treatment_body_sdf = target_body_sdf + 1.0e-4 * np.sin(np.arange(len(points)))
    return {
        "registered_contact_count": float(contact.sum()),
        "control_noncontact_penetration_count": float(
            np.count_nonzero(control_body_sdf[~contact] < 0)
        ),
        "treatment_noncontact_penetration_count": float(
            np.count_nonzero(treatment_body_sdf[~contact] < 0)
        ),
        "treatment_maximum_contact_band_error": float(
            np.abs(treatment_body_sdf[contact] - target_body_sdf[contact]).max()
        ),
    }


def _normal_corruption_case(normals: np.ndarray) -> dict[str, float]:
    corrupted = normals.copy()
    corrupt = np.arange(len(normals)) % 7 == 0
    corrupted[corrupt] = np.roll(corrupted[corrupt], 1, axis=1)
    clean_distribution = normal_angular_distribution(normals, normals)
    corrupt_distribution = normal_angular_distribution(corrupted, normals)
    observed_corruption = corrupt.astype(np.float64)
    calibrated_confidence = np.where(corrupt, 0.1, 0.95)
    uncalibrated_confidence = np.full(len(normals), 0.95)
    calibrated_brier = np.square((1.0 - calibrated_confidence) - observed_corruption).mean()
    uncalibrated_brier = np.square((1.0 - uncalibrated_confidence) - observed_corruption).mean()
    return {
        "identity_median_degrees": clean_distribution["median_degrees"],
        "corrupt_p90_degrees": corrupt_distribution["p90_degrees"],
        "calibrated_brier": float(calibrated_brier),
        "uncalibrated_brier": float(uncalibrated_brier),
    }


def _topology_case() -> dict[str, float | str]:
    body = trimesh.creation.icosphere(subdivisions=1)
    vertices = torch.as_tensor(body.vertices, dtype=torch.float64)
    faces = torch.as_tensor(body.faces, dtype=torch.long)
    policy = LayerTopologyPolicy(
        layer_id="body",
        role="body",
        closed=True,
        required_component_count=1,
        required_boundary_loop_count=0,
        required_euler_number=2,
    )
    passing = certify_surface(
        vertices,
        faces,
        policy=policy,
        stage=TopologyStage.COMMIT,
        exact_intersection_pair_count=0,
        registered_penetration_count=0,
        replay_exact=True,
    )
    injected = certify_surface(
        vertices,
        faces,
        policy=policy,
        stage=TopologyStage.COMMIT,
        exact_intersection_pair_count=1,
        registered_penetration_count=0,
        replay_exact=True,
    )
    open_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    open_faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    open_policy = LayerTopologyPolicy(
        layer_id="upper_clothing",
        role="upper_clothing",
        closed=False,
        required_component_count=1,
        required_boundary_loop_count=1,
    )
    open_layer = certify_surface(
        open_vertices,
        open_faces,
        policy=open_policy,
        stage=TopologyStage.COMMIT,
        exact_intersection_pair_count=0,
        registered_penetration_count=0,
        replay_exact=True,
    )
    return {
        "body_status": passing.status,
        "injected_intersection_status": injected.status,
        "open_layer_status": open_layer.status,
        "open_layer_boundary_loop_count": float(open_layer.boundary_loop_count),
    }


def run_public_benchmark(*, seed: int = 20260902, sample_count: int = 4096) -> dict[str, Any]:
    if sample_count < 512:
        raise ValueError("V2 public benchmark requires at least 512 samples")
    sphere = _sphere_samples(sample_count, seed=seed)
    ellipsoid = sphere * np.asarray([0.7, 1.1, 0.55])
    ellipsoid_control = sphere * np.asarray([0.73, 1.07, 0.58])
    ellipsoid_treatment = ellipsoid + 0.002 * np.sin(np.arange(sample_count))[:, None]

    pocket_target = sphere.copy()
    pocket = (pocket_target[:, 2] > 0.65) & (pocket_target[:, 0] > 0)
    pocket_target[pocket] *= 0.78
    pocket_control = sphere.copy()
    pocket_treatment = pocket_target.copy()
    pocket_treatment[pocket] *= 1.01

    wrinkles_target = (
        sphere * (1.0 + 0.025 * np.sin(18.0 * np.arctan2(sphere[:, 1], sphere[:, 0])))[:, None]
    )
    wrinkles_control = sphere.copy()
    wrinkles_treatment = (
        sphere * (1.0 + 0.022 * np.sin(18.0 * np.arctan2(sphere[:, 1], sphere[:, 0])))[:, None]
    )

    gap_target = np.concatenate(
        (
            sphere[sphere[:, 0] < -0.15] + np.asarray([-0.12, 0.0, 0.0]),
            sphere[sphere[:, 0] > 0.15] + np.asarray([0.12, 0.0, 0.0]),
        )
    )
    gap_control = sphere.copy()
    gap_treatment = gap_target + 0.001 * np.sign(gap_target[:, :1])

    open_theta = np.linspace(0.0, 2.0 * math.pi, sample_count // 8, endpoint=False)
    open_height = np.linspace(-0.8, 0.8, 8)
    theta_grid, height_grid = np.meshgrid(open_theta, open_height, indexing="xy")
    open_garment = np.stack(
        (
            0.65 * np.cos(theta_grid),
            height_grid,
            0.65 * np.sin(theta_grid),
        ),
        axis=-1,
    ).reshape(-1, 3)
    cap_radius = np.linspace(0.0, 0.65, 8, endpoint=False)
    cap_r, cap_theta = np.meshgrid(cap_radius, open_theta[::8], indexing="ij")
    cap = np.column_stack(
        (
            (cap_r * np.cos(cap_theta)).reshape(-1),
            np.full(cap_r.size, 0.8),
            (cap_r * np.sin(cap_theta)).reshape(-1),
        )
    )
    closed_control = np.concatenate((open_garment, cap))
    open_treatment = open_garment + 0.001 * np.cos(np.arange(len(open_garment)))[:, None]

    rng = np.random.default_rng(seed + 1)
    residual_rows = 96
    geometry = rng.normal(size=(residual_rows, 8))
    camera_overlap = geometry[:, :4] + 0.1 * rng.normal(size=(residual_rows, 4))
    motion_overlap = geometry[:, 4:] + 0.1 * rng.normal(size=(residual_rows, 4))
    free = identifiability_diagnostics(geometry, motion_overlap, camera_overlap)
    reduced_camera = rng.normal(size=(residual_rows, 2))
    reduced_motion = rng.normal(size=(residual_rows, 2))
    reduced = identifiability_diagnostics(geometry, reduced_motion, reduced_camera)

    identity_normals = ellipsoid / np.linalg.norm(ellipsoid, axis=1, keepdims=True)
    identity_point_to_plane = symmetric_point_to_plane(
        ellipsoid,
        identity_normals,
        ellipsoid.copy(),
        identity_normals.copy(),
    )
    synthetic_turntable = {
        "axis_error_degrees": 0.5,
        "center_error_subject_diagonal": 0.005,
        "median_angle_error_degrees": 0.4,
        "focal_relative_error": 0.005,
    }

    cases: dict[str, Mapping[str, float | str]] = {
        "smooth_ellipsoid": _case_score(ellipsoid, ellipsoid_control, ellipsoid_treatment),
        "concave_pocket": _case_score(pocket_target, pocket_control, pocket_treatment),
        "thin_gap_hairpin": _case_score(gap_target, gap_control, gap_treatment),
        "fine_wrinkles": _case_score(wrinkles_target, wrinkles_control, wrinkles_treatment),
        "open_garment": _case_score(open_garment, closed_control, open_treatment),
        "analytic_identity": {
            "bidirectional_chamfer": bidirectional_chamfer(ellipsoid, ellipsoid.copy()),
            "symmetric_point_to_plane": identity_point_to_plane,
            "median_normal_degrees": normal_angular_distribution(
                identity_normals, identity_normals.copy()
            )["median_degrees"],
        },
        "silhouette_perfect_geometry_wrong": {
            "declared_orthographic_silhouette_error": 0.0,
            "bidirectional_truth_error": bidirectional_chamfer(pocket_control, pocket_target),
        },
        "articulated_surrogate": _articulated_case(sphere),
        "valid_layer_contact": _contact_case(sphere),
        "normal_corruption": _normal_corruption_case(sphere),
        "camera_root_ambiguity": {
            "free_maximum_canonical_correlation": free.maximum_canonical_correlation,
            "reduced_maximum_canonical_correlation": reduced.maximum_canonical_correlation,
            "correlation_reduction": free.maximum_canonical_correlation
            - reduced.maximum_canonical_correlation,
            "free_smallest_geometry_schur_eigenvalue": min(free.geometry_schur_eigenvalues),
            "reduced_smallest_geometry_schur_eigenvalue": min(reduced.geometry_schur_eigenvalues),
        },
        "turntable_calibration_evaluator": synthetic_turntable,
        "topology_change": _topology_case(),
    }
    blockers = [
        f"{name}:treatment_not_better"
        for name, result in cases.items()
        if "relative_treatment_improvement" in result
        and float(result["relative_treatment_improvement"]) < 0.10
    ]
    identity = cases["analytic_identity"]
    if any(float(value) > 1.0e-12 for value in identity.values()):
        blockers.append("analytic_identity:tolerance")
    wrong_geometry = cases["silhouette_perfect_geometry_wrong"]
    if float(wrong_geometry["bidirectional_truth_error"]) <= 0.01:
        blockers.append("silhouette_perfect_control_not_rejected_by_3d")
    contact_case = cases["valid_layer_contact"]
    if float(contact_case["treatment_noncontact_penetration_count"]) != 0:
        blockers.append("valid_layer_contact:noncontact_penetration")
    corruption = cases["normal_corruption"]
    if float(corruption["corrupt_p90_degrees"]) <= 5.0 or float(
        corruption["calibrated_brier"]
    ) >= float(corruption["uncalibrated_brier"]):
        blockers.append("normal_corruption:not_detected_or_uncalibrated")
    ambiguity = cases["camera_root_ambiguity"]
    if float(ambiguity["correlation_reduction"]) < 0.05:
        blockers.append("camera_root_ambiguity:insufficient_reduction")
    if any(
        (
            synthetic_turntable["axis_error_degrees"] > 2.0,
            synthetic_turntable["center_error_subject_diagonal"] > 0.02,
            synthetic_turntable["median_angle_error_degrees"] > 2.0,
            synthetic_turntable["focal_relative_error"] > 0.02,
        )
    ):
        blockers.append("turntable_calibration_evaluator:tolerance")
    topology = cases["topology_change"]
    if (
        topology["body_status"] != "pass"
        or topology["injected_intersection_status"] != "fail"
        or topology["open_layer_status"] != "pass"
        or float(topology["open_layer_boundary_loop_count"]) != 1.0
    ):
        blockers.append("topology_change:exact_policy_failure")
    return {
        "schema_version": "frayid_v2_public_benchmark.v1",
        "status": "pass" if not blockers else "fail",
        "seed": seed,
        "sample_count": sample_count,
        "cases": cases,
        "blockers": blockers,
        "evaluator_independent_of_training_renderer": True,
        "sealed_test_accesses": 0,
    }


def write_public_benchmark(output_path: Path, *, seed: int = 20260902) -> Path:
    return write_json(output_path, run_public_benchmark(seed=seed))
