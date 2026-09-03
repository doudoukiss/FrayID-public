from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import TurntableSolution
from frayid.v2.track_factors import load_pairwise_tracklet_factors, pairwise_sampson_loss
from frayid.v2.turntable import (
    axis_angle_rotation,
    turntable_edge_slots,
    turntable_fundamental_matrices,
)


@dataclass(frozen=True)
class SyntheticTurntableProblem:
    canonical_points: np.ndarray
    observations: np.ndarray
    principal_point: tuple[float, float]
    center_y: float
    true_axis: np.ndarray
    true_center: np.ndarray
    true_angles: np.ndarray
    true_focal: float


def _axis(axis_x: float, axis_z: float) -> np.ndarray:
    value = np.asarray([axis_x, 1.0, axis_z], dtype=np.float64)
    return value / np.linalg.norm(value)


def _rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return np.asarray(
        [
            identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
            for angle in angles
        ]
    )


def _project(
    points: np.ndarray,
    *,
    axis: np.ndarray,
    center: np.ndarray,
    angles: np.ndarray,
    focal: float | np.ndarray,
    principal_point: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    rotations = _rotations(axis, angles)
    trajectories = np.einsum("tij,pj->tpi", rotations, points - center) + center
    depth = trajectories[..., 2]
    if np.any(depth <= 0.1):
        raise ValueError("synthetic turntable projected behind the camera")
    focal_array = np.asarray(focal, dtype=np.float64)
    if focal_array.ndim == 0:
        focal_array = np.full((len(angles),), float(focal_array), dtype=np.float64)
    cx, cy = principal_point
    pixels = np.stack(
        (
            focal_array[:, None] * trajectories[..., 0] / depth + cx,
            focal_array[:, None] * trajectories[..., 1] / depth + cy,
        ),
        axis=-1,
    )
    return pixels, trajectories


def make_synthetic_turntable_problem(seed: int = 20260902) -> SyntheticTurntableProblem:
    rng = np.random.default_rng(seed)
    point_count = 72
    directions = rng.normal(size=(point_count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.asarray([0.48, 0.92, 0.34], dtype=np.float64)
    center = np.asarray([0.08, -0.04, 4.2], dtype=np.float64)
    points = center + directions * radii
    # Break bilateral symmetry so axis direction and angular phase are observable.
    points[:8, 0] += np.linspace(0.03, 0.16, 8)
    true_axis = _axis(0.018, -0.014)
    increments = np.linspace(0.30, 0.40, 17)
    true_angles = np.concatenate((np.zeros(1), np.cumsum(increments)))
    true_focal = 820.0
    principal_point = (360.0, 560.0)
    clean, _ = _project(
        points,
        axis=true_axis,
        center=center,
        angles=true_angles,
        focal=true_focal,
        principal_point=principal_point,
    )
    observations = clean + rng.normal(scale=0.18, size=clean.shape)
    return SyntheticTurntableProblem(
        canonical_points=points,
        observations=observations,
        principal_point=principal_point,
        center_y=float(center[1]),
        true_axis=true_axis,
        true_center=center,
        true_angles=true_angles,
        true_focal=true_focal,
    )


def _unpack(
    parameters: np.ndarray,
    *,
    frame_count: int,
    center_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    axis = _axis(float(parameters[0]), float(parameters[1]))
    center = np.asarray([parameters[2], center_y, parameters[3]], dtype=np.float64)
    focal = math.exp(float(parameters[4]))
    increments = np.exp(parameters[5 : 5 + frame_count - 1])
    angles = np.concatenate((np.zeros(1), np.cumsum(increments)))
    return axis, center, angles, focal


def fit_reduced_turntable(problem: SyntheticTurntableProblem) -> dict[str, Any]:
    frame_count = int(problem.observations.shape[0])
    initial_increments = np.diff(problem.true_angles) * np.linspace(0.88, 1.12, frame_count - 1)
    initial = np.concatenate(
        (
            np.asarray([0.08, -0.06]),
            problem.true_center[[0, 2]] + np.asarray([0.08, -0.12]),
            np.asarray([math.log(problem.true_focal * 1.06)]),
            np.log(initial_increments),
        )
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        axis, center, angles, focal = _unpack(
            parameters,
            frame_count=frame_count,
            center_y=problem.center_y,
        )
        predicted, _ = _project(
            problem.canonical_points,
            axis=axis,
            center=center,
            angles=angles,
            focal=focal,
            principal_point=problem.principal_point,
        )
        reprojection = (predicted - problem.observations).reshape(-1)
        smoothness = np.diff(np.diff(angles)) * 0.5
        return np.concatenate((reprojection, smoothness))

    result = least_squares(
        residual,
        initial,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=500,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    axis, center, angles, focal = _unpack(
        result.x,
        frame_count=frame_count,
        center_y=problem.center_y,
    )
    axis_error = math.degrees(math.acos(float(np.clip(axis @ problem.true_axis, -1.0, 1.0))))
    subject_diagonal = float(np.linalg.norm(np.ptp(problem.canonical_points, axis=0)))
    center_error_fraction = float(np.linalg.norm(center - problem.true_center) / subject_diagonal)
    angle_error = np.degrees(np.abs(angles - problem.true_angles))
    focal_error_fraction = abs(focal - problem.true_focal) / problem.true_focal
    return {
        "success": bool(result.success),
        "termination": str(result.message),
        "function_evaluations": int(result.nfev),
        "axis": axis.tolist(),
        "center": center.tolist(),
        "angles_radians": angles.tolist(),
        "focal": focal,
        "axis_error_degrees": axis_error,
        "center_error_fraction_subject_diagonal": center_error_fraction,
        "median_angle_error_degrees": float(np.median(angle_error)),
        "maximum_angle_error_degrees": float(np.max(angle_error)),
        "focal_error_fraction": focal_error_fraction,
        "final_median_reprojection_pixels": float(
            np.median(np.abs(residual(result.x)[: problem.observations.size]))
        ),
    }


def _finite_jacobian(function: Any, value: np.ndarray, epsilon: float = 1.0e-5) -> np.ndarray:
    baseline = np.asarray(function(value), dtype=np.float64).reshape(-1)
    jacobian = np.empty((len(baseline), len(value)), dtype=np.float64)
    for column in range(len(value)):
        step = np.zeros_like(value)
        step[column] = epsilon
        jacobian[:, column] = (
            np.asarray(function(value + step)).reshape(-1)
            - np.asarray(function(value - step)).reshape(-1)
        ) / (2.0 * epsilon)
    return jacobian


def _geometry_modes(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    relative = points - center
    x, y, z = relative.T
    modes = np.stack(
        (
            np.stack((y, np.zeros_like(y), np.zeros_like(y)), axis=-1),
            np.stack((np.zeros_like(x), x, np.zeros_like(x)), axis=-1),
            np.stack((z * y, np.zeros_like(y), np.zeros_like(y)), axis=-1),
            np.stack((np.zeros_like(x), x * z, np.zeros_like(x)), axis=-1),
            np.stack((x * y, -0.5 * np.square(x), np.zeros_like(x)), axis=-1),
            np.stack((np.zeros_like(z), y * z, -0.5 * np.square(y)), axis=-1),
        ),
        axis=0,
    )
    norms = np.linalg.norm(modes.reshape(len(modes), -1), axis=1)
    return np.asarray(modes / norms[:, None, None], dtype=np.float64)


def _range(matrix: np.ndarray) -> np.ndarray:
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = max(float(singular[0]) if len(singular) else 0.0, 1.0) * 1.0e-8
    return left[:, singular > threshold]


def _geometry_nuisance_diagnostics(
    geometry: np.ndarray,
    nuisance: np.ndarray,
) -> dict[str, Any]:
    geometry_range = _range(geometry)
    nuisance_range = _range(nuisance)
    singular = np.linalg.svd(geometry_range.T @ nuisance_range, compute_uv=False)
    correlation = float(np.clip(singular[0], 0.0, 1.0)) if len(singular) else 0.0
    h_gg = geometry.T @ geometry
    h_gn = geometry.T @ nuisance
    h_nn = nuisance.T @ nuisance + 1.0e-8 * np.eye(nuisance.shape[1])
    schur = h_gg - h_gn @ np.linalg.solve(h_nn, h_gn.T)
    eigenvalues = np.linalg.eigvalsh(0.5 * (schur + schur.T))
    rank_threshold = max(float(eigenvalues[-1]), 1.0) * 1.0e-8
    return {
        "maximum_geometry_nuisance_canonical_correlation": correlation,
        "geometry_schur_eigenvalues": eigenvalues.tolist(),
        "registered_geometry_rank": int(np.linalg.matrix_rank(geometry)),
        "schur_informative_rank": int(np.sum(eigenvalues > rank_threshold)),
        # Every geometry column above is a normalized, preregistered non-gauge mode.
        # A near-zero control eigenvalue is the loss of observability being measured;
        # filtering it independently in each arm would make the comparison invalid.
        "smallest_informative_geometry_schur_eigenvalue": max(float(eigenvalues[0]), 0.0),
    }


def turntable_identifiability_benchmark(
    problem: SyntheticTurntableProblem,
) -> dict[str, Any]:
    points = problem.canonical_points
    modes = _geometry_modes(points, problem.true_center)
    frame_count = len(problem.true_angles)

    def geometry_prediction(coefficients: np.ndarray) -> np.ndarray:
        changed = points + np.einsum("m,mpd->pd", coefficients, modes)
        pixels, _ = _project(
            changed,
            axis=problem.true_axis,
            center=problem.true_center,
            angles=problem.true_angles,
            focal=problem.true_focal,
            principal_point=problem.principal_point,
        )
        return pixels

    structured_base = np.concatenate(
        (
            problem.true_axis[[0, 2]] / problem.true_axis[1],
            problem.true_center[[0, 2]],
            problem.true_angles[1:],
        )
    )

    def structured_motion(parameters: np.ndarray) -> np.ndarray:
        axis = _axis(float(parameters[0]), float(parameters[1]))
        center = np.asarray([parameters[2], problem.center_y, parameters[3]])
        angles = np.concatenate((np.zeros(1), parameters[4:]))
        pixels, _ = _project(
            points,
            axis=axis,
            center=center,
            angles=angles,
            focal=problem.true_focal,
            principal_point=problem.principal_point,
        )
        return pixels

    def shared_camera(parameters: np.ndarray) -> np.ndarray:
        pixels, _ = _project(
            points,
            axis=problem.true_axis,
            center=problem.true_center,
            angles=problem.true_angles,
            focal=float(parameters[0]),
            principal_point=problem.principal_point,
        )
        return pixels

    _, trajectories = _project(
        points,
        axis=problem.true_axis,
        center=problem.true_center,
        angles=problem.true_angles,
        focal=problem.true_focal,
        principal_point=problem.principal_point,
    )

    def free_motion(parameters: np.ndarray) -> np.ndarray:
        values = parameters.reshape(frame_count, 6)
        moved = trajectories.copy()
        for frame, (rx, ry, rz, tx, ty, tz) in enumerate(values):
            rotation_axis = np.asarray([rx, ry, rz], dtype=np.float64)
            magnitude = float(np.linalg.norm(rotation_axis))
            if magnitude > 0:
                rotation = _rotations(rotation_axis / magnitude, np.asarray([magnitude]))[0]
                moved[frame] = (
                    np.einsum(
                        "ij,pj->pi",
                        rotation,
                        moved[frame] - problem.true_center,
                    )
                    + problem.true_center
                )
            moved[frame] += np.asarray([tx, ty, tz])
        cx, cy = problem.principal_point
        return np.stack(
            (
                problem.true_focal * moved[..., 0] / moved[..., 2] + cx,
                problem.true_focal * moved[..., 1] / moved[..., 2] + cy,
            ),
            axis=-1,
        )

    def free_camera(parameters: np.ndarray) -> np.ndarray:
        cx, cy = problem.principal_point
        return np.stack(
            (
                parameters[:, None] * trajectories[..., 0] / trajectories[..., 2] + cx,
                parameters[:, None] * trajectories[..., 1] / trajectories[..., 2] + cy,
            ),
            axis=-1,
        )

    geometry_jacobian = _finite_jacobian(geometry_prediction, np.zeros(len(modes)))
    structured_motion_jacobian = _finite_jacobian(structured_motion, structured_base)
    structured_camera_jacobian = _finite_jacobian(
        shared_camera,
        np.asarray([problem.true_focal]),
    )
    free_motion_jacobian = _finite_jacobian(
        free_motion,
        np.zeros(frame_count * 6),
    )
    free_camera_jacobian = _finite_jacobian(
        free_camera,
        np.full((frame_count,), problem.true_focal),
    )
    structured = _geometry_nuisance_diagnostics(
        geometry_jacobian,
        np.concatenate((structured_motion_jacobian, structured_camera_jacobian), axis=1),
    )
    control = _geometry_nuisance_diagnostics(
        geometry_jacobian,
        np.concatenate((free_motion_jacobian, free_camera_jacobian), axis=1),
    )
    correlation_drop = float(
        control["maximum_geometry_nuisance_canonical_correlation"]
        - structured["maximum_geometry_nuisance_canonical_correlation"]
    )
    structured_eigenvalue = float(structured["smallest_informative_geometry_schur_eigenvalue"])
    control_eigenvalue = float(control["smallest_informative_geometry_schur_eigenvalue"])
    eigenvalue_rise = structured_eigenvalue / max(control_eigenvalue, 1.0e-12) - 1.0
    return {
        "structured": structured,
        "free_camera_root_control": control,
        "geometry_nuisance_correlation_drop": correlation_drop,
        "smallest_informative_schur_eigenvalue_rise_fraction": eigenvalue_rise,
    }


def write_turntable_ba_benchmark(output_path: Path, *, seed: int = 20260902) -> Path:
    reject_sealed_capability([output_path])
    problem = make_synthetic_turntable_problem(seed)
    fit = fit_reduced_turntable(problem)
    identifiability = turntable_identifiability_benchmark(problem)
    blockers: list[str] = []
    gates = {
        "axis_error_degrees_at_most_2": float(fit["axis_error_degrees"]) <= 2.0,
        "center_error_fraction_at_most_0_02": float(fit["center_error_fraction_subject_diagonal"])
        <= 0.02,
        "median_angle_error_degrees_at_most_2": float(fit["median_angle_error_degrees"]) <= 2.0,
        "focal_error_fraction_at_most_0_02": float(fit["focal_error_fraction"]) <= 0.02,
        "geometry_nuisance_correlation_drop_at_least_0_05": float(
            identifiability["geometry_nuisance_correlation_drop"]
        )
        >= 0.05,
        "smallest_informative_schur_eigenvalue_rise_at_least_0_25": float(
            identifiability["smallest_informative_schur_eigenvalue_rise_fraction"]
        )
        >= 0.25,
    }
    blockers.extend(name for name, passed in gates.items() if not passed)
    report = {
        "schema_version": "frayid_v2_turntable_ba_public_benchmark.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t01_public_reduced_ba_qualification_r01",
        "seed": seed,
        "fit": fit,
        "identifiability": identifiability,
        "gates": gates,
        "blockers": blockers,
        "center_axis_gauge": "center_component_parallel_to_axis_fixed_from_scaffold",
        "scientific_attempt_marker_created": False,
        "private_evidence_reads": 0,
        "development_metrics_read": 0,
        "sealed_test_accesses": 0,
        "modal_jobs": 0,
        "optimizer_role": "public_synthetic_engineering_qualification_only",
    }
    return write_json(output_path, report)


def diagnose_real_turntable_factor_route(
    solution_path: Path,
    factor_binding_path: Path,
    output_path: Path,
    *,
    image_size: tuple[int, int],
) -> Path:
    """Backpropagate real train-only factors once without taking an optimizer step."""

    reject_sealed_capability([solution_path, factor_binding_path, output_path])
    solution = TurntableSolution.model_validate(read_json(solution_path))
    factors = load_pairwise_tracklet_factors(factor_binding_path)
    dtype = torch.float32
    axis_tangent = torch.tensor(solution.axis, dtype=dtype, requires_grad=True)
    center = torch.tensor(solution.center, dtype=dtype, requires_grad=True)
    angles = torch.tensor(solution.angles_radians, dtype=dtype, requires_grad=True)
    initial_focal = float(solution.shared_intrinsics[0][0])
    log_focal = torch.tensor(math.log(initial_focal), dtype=dtype, requires_grad=True)
    principal_x = float(solution.shared_intrinsics[0][2])
    principal_y = float(solution.shared_intrinsics[1][2])

    def intrinsics(value: torch.Tensor) -> torch.Tensor:
        focal = torch.exp(value)
        zero = focal * 0.0
        one = zero + 1.0
        return torch.stack(
            (
                torch.stack((focal, zero, zero + principal_x)),
                torch.stack((zero, focal, zero + principal_y)),
                torch.stack((zero, zero, one)),
            )
        )

    first_slots, second_slots = turntable_edge_slots(
        solution.source_frame_indices,
        factors.first_source_frame_indices,
        factors.second_source_frame_indices,
    )

    def loss_for(
        axis_value: torch.Tensor,
        center_value: torch.Tensor,
        angle_value: torch.Tensor,
        log_focal_value: torch.Tensor,
    ) -> torch.Tensor:
        normalized_axis = F.normalize(axis_value, dim=0, eps=1.0e-12)
        rotations = axis_angle_rotation(normalized_axis, angle_value)
        fundamental = turntable_fundamental_matrices(
            rotations,
            center_value,
            intrinsics(log_focal_value),
            first_slots,
            second_slots,
        )
        return pairwise_sampson_loss(
            fundamental,
            factors,
            image_size=image_size,
        )

    loss = loss_for(axis_tangent, center, angles, log_focal)
    loss.backward()  # type: ignore[no-untyped-call]
    variables = {
        "axis_tangent": axis_tangent,
        "center": center,
        "angles": angles,
        "log_focal": log_focal,
    }
    gradient_norms: dict[str, float] = {}
    gradient_finite: dict[str, bool] = {}
    for name, value in variables.items():
        gradient = value.grad
        gradient_finite[name] = gradient is not None and bool(torch.isfinite(gradient).all())
        gradient_norms[name] = (
            float(torch.linalg.vector_norm(gradient)) if gradient is not None else 0.0
        )
    with torch.no_grad():
        base_axis = axis_tangent.detach()
        base_center = center.detach()
        base_angles = angles.detach()
        base_focal = log_focal.detach()
        phase_pattern = torch.sin(torch.linspace(0.0, math.pi, len(base_angles))) * math.radians(
            3.0
        )
        stresses = {
            "axis_tilt_3_degrees": loss_for(
                F.normalize(
                    base_axis + torch.tensor([math.tan(math.radians(3.0)), 0.0, 0.0]), dim=0
                ),
                base_center,
                base_angles,
                base_focal,
            ),
            "center_x_plus_0_03": loss_for(
                base_axis,
                base_center + torch.tensor([0.03, 0.0, 0.0]),
                base_angles,
                base_focal,
            ),
            "phase_sine_3_degrees": loss_for(
                base_axis,
                base_center,
                base_angles + phase_pattern,
                base_focal,
            ),
            "focal_plus_5_percent": loss_for(
                base_axis,
                base_center,
                base_angles,
                base_focal + math.log(1.05),
            ),
        }
    base_loss = float(loss.detach())
    stress_values = {name: float(value) for name, value in stresses.items()}
    blockers: list[str] = []
    if not all(gradient_finite.values()):
        blockers.append("nonfinite_turntable_factor_gradient")
    if sum(gradient_norms.values()) <= 0:
        blockers.append("zero_turntable_factor_gradient")
    if factors.factor_count < 1000:
        blockers.append("insufficient_pairwise_factor_count")
    report = {
        "schema_version": "frayid_v2_real_turntable_factor_route.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t01_real_factor_forward_backward_q01",
        "solution_path": str(solution_path),
        "solution_sha256": sha256_file(solution_path),
        "factor_binding_path": str(factor_binding_path),
        "factor_binding_sha256": sha256_file(factor_binding_path),
        "edge_count": factors.edge_count,
        "factor_count": factors.factor_count,
        "base_robust_sampson_loss": base_loss,
        "gradient_norms": gradient_norms,
        "gradient_finite": gradient_finite,
        "capacity_stress_losses": stress_values,
        "capacity_stress_loss_ratios": {
            name: value / max(base_loss, 1.0e-12) for name, value in stress_values.items()
        },
        "blockers": blockers,
        "optimizer_steps": 0,
        "scientific_attempt_marker_created": False,
        "training_images_read": 0,
        "development_metrics_read": 0,
        "held_out_images_read": 0,
        "sealed_test_accesses": 0,
        "modal_jobs": 0,
        "notes": [
            "This checks a differentiable route through fixed Q01 proposals; it does not update T01.",
            "Stress losses are diagnostic before fitting and are not promotion metrics.",
        ],
    }
    return write_json(output_path, report)
