from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

EXPERIMENT_ID = "postv3_p01_rotation_photometric_varpro_r01"


def _fit_profiled(
    intensity: np.ndarray,
    basis: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize shared lighting while profiling per-chart albedo analytically."""
    chart_count = intensity.shape[1]

    def decode(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lighting = np.concatenate([np.ones(1), parameters])
        illumination = basis @ lighting
        train_light = illumination[train]
        denominator = max(float(train_light @ train_light), 1e-12)
        albedo = np.sum(train_light[:, None] * intensity[train], axis=0) / denominator
        prediction = illumination[:, None] * albedo[None, :]
        return lighting, albedo, prediction

    def residual(parameters: np.ndarray) -> np.ndarray:
        _, _, prediction = decode(parameters)
        regularizer = 0.01 * parameters
        return np.concatenate([(prediction[train] - intensity[train]).reshape(-1), regularizer])

    result = least_squares(residual, np.zeros(basis.shape[1] - 1), method="trf", loss="soft_l1")
    lighting, albedo, prediction = decode(result.x)
    if albedo.shape != (chart_count,):
        raise RuntimeError("profiled albedo shape mismatch")
    return lighting, albedo, prediction


def diagnose_photometry(payload: dict[str, Any]) -> dict[str, Any]:
    intensity = np.asarray(payload["linear_bt709_intensity"], dtype=np.float64)
    basis = np.asarray(payload["spherical_harmonic_basis"], dtype=np.float64)
    train = np.asarray(payload["train_frame_indices"], dtype=np.int64)
    held = np.asarray(payload["held_out_frame_indices"], dtype=np.int64)
    if intensity.ndim != 2 or basis.shape[0] != intensity.shape[0]:
        raise ValueError("intensity and lighting basis must share the frame dimension")
    lighting, albedo, prediction = _fit_profiled(intensity, basis, train)
    baseline = np.repeat(np.mean(intensity[train], axis=0, keepdims=True), len(intensity), axis=0)
    error = float(np.mean(np.abs(prediction[held] - intensity[held])))
    baseline_error = float(np.mean(np.abs(baseline[held] - intensity[held])))
    improvement = 1.0 - error / max(baseline_error, 1e-12)

    shuffled_basis = basis.copy()
    permutation = np.random.default_rng(90210).permutation(len(basis))
    shuffled_basis[:, 1:] = shuffled_basis[permutation, 1:]
    _, _, shuffled_prediction = _fit_profiled(intensity, shuffled_basis, train)
    shuffled_error = float(np.mean(np.abs(shuffled_prediction[held] - intensity[held])))
    shuffled_improvement = 1.0 - shuffled_error / max(baseline_error, 1e-12)
    normal_improvement = float(payload["independent_normal_improvement_degrees"])
    blockers: list[str] = []
    if improvement < 0.1:
        blockers.append("held_out_intensity_improvement_below_10_percent")
    if normal_improvement < 1.0:
        blockers.append("independent_normal_improvement_below_1_degree")
    if bool(payload.get("geometry_regression", True)):
        blockers.append("geometry_regression")
    if shuffled_improvement >= 0.1:
        blockers.append("benefit_survives_shuffled_phase")
    return {
        "schema_version": "frayid_v3_rotation_photometric_varpro.v1",
        "experiment_id": EXPERIMENT_ID,
        "input_color_representation": "decoded_bt709_documented_linear_light_not_raw_sensor_rgb",
        "evidence_scope": str(payload.get("evidence_scope", "public_synthetic")),
        "status": "pass" if not blockers else "fail",
        "promotion_eligible": not blockers and payload.get("evidence_scope") == "train_real",
        "profiled_chart_albedo": albedo.tolist(),
        "profiled_low_order_lighting": lighting.tolist(),
        "profiled_exposure_model": "fixed_manual_exposure_or_precalibrated_low_order_input",
        "held_out_intensity_improvement_fraction": improvement,
        "shuffled_phase_improvement_fraction": shuffled_improvement,
        "independent_normal_improvement_degrees": normal_improvement,
        "geometry_regression": bool(payload.get("geometry_regression", True)),
        "blockers": blockers,
    }


def public_photometry_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(31)
    frame_count = 30
    chart_count = 16
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count, endpoint=False)
    basis = np.column_stack([np.ones(frame_count), np.cos(phase), np.sin(phase)])
    lighting = np.asarray([1.0, 0.25, -0.15])
    albedo = rng.uniform(0.25, 0.85, chart_count)
    intensity = (basis @ lighting)[:, None] * albedo[None, :]
    intensity += rng.normal(0.0, 0.002, intensity.shape)
    held = np.arange(4, frame_count, 5)
    train = np.asarray([index for index in range(frame_count) if index not in set(held)])
    return {
        "linear_bt709_intensity": intensity.tolist(),
        "spherical_harmonic_basis": basis.tolist(),
        "train_frame_indices": train.tolist(),
        "held_out_frame_indices": held.tolist(),
        "independent_normal_improvement_degrees": 1.5,
        "geometry_regression": False,
        "evidence_scope": "public_synthetic",
    }
