from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from frayid.v3.schemas import CameraIntrinsics, FixedCameraFactorGraphSolution

EXPERIMENT_ID = "postv3_t06_image_driven_fixed_camera_factor_graph_r01"


def assert_same_tensor_device(named_tensors: dict[str, torch.Tensor]) -> torch.device:
    """Fail before a factor mixes camera and evidence tensors across devices."""
    devices = {tensor.device for tensor in named_tensors.values()}
    if len(devices) != 1:
        details = ", ".join(f"{name}={tensor.device}" for name, tensor in named_tensors.items())
        raise RuntimeError(f"factor tensor-device mismatch: {details}")
    if not devices:
        raise ValueError("at least one tensor is required")
    return next(iter(devices))


def _rotation_y(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _monotonic_phase(raw: np.ndarray, frame_count: int) -> np.ndarray:
    logits = np.concatenate([np.zeros(1), np.clip(raw, -5.0, 5.0)])
    weights = np.exp(logits)
    increments = (2.0 * np.pi) * weights / np.sum(weights)
    return np.concatenate([np.zeros(1), np.cumsum(increments)])[:frame_count]


@dataclass(frozen=True)
class _State:
    phase: np.ndarray
    root: np.ndarray
    focal: float
    anchors: np.ndarray
    reprojection: np.ndarray


def _profile_anchors(
    observed: np.ndarray,
    phase: np.ndarray,
    root: np.ndarray,
    focal: float,
    principal: np.ndarray,
    camera_translation: np.ndarray,
) -> np.ndarray:
    frame_count, anchor_count, _ = observed.shape
    anchors = np.zeros((anchor_count, 3), dtype=np.float64)
    for anchor_index in range(anchor_count):
        rows: list[np.ndarray] = []
        values: list[float] = []
        for frame_index in range(frame_count):
            rotation = _rotation_y(float(phase[frame_index]))
            translation = camera_translation + root[frame_index]
            x_norm = (observed[frame_index, anchor_index, 0] - principal[0]) / focal
            y_norm = (observed[frame_index, anchor_index, 1] - principal[1]) / focal
            rows.extend([rotation[0] - x_norm * rotation[2], rotation[1] - y_norm * rotation[2]])
            values.extend(
                [
                    float(x_norm * translation[2] - translation[0]),
                    float(y_norm * translation[2] - translation[1]),
                ]
            )
        design = np.asarray(rows)
        target = np.asarray(values)
        anchors[anchor_index], *_ = np.linalg.lstsq(design, target, rcond=None)
    return anchors


def _project(
    anchors: np.ndarray,
    phase: np.ndarray,
    root: np.ndarray,
    focal: float,
    principal: np.ndarray,
    camera_translation: np.ndarray,
) -> np.ndarray:
    predictions = np.zeros((len(phase), len(anchors), 2), dtype=np.float64)
    for frame_index, angle in enumerate(phase):
        points = (_rotation_y(float(angle)) @ anchors.T).T
        points += camera_translation + root[frame_index]
        predictions[frame_index, :, 0] = focal * points[:, 0] / points[:, 2] + principal[0]
        predictions[frame_index, :, 1] = focal * points[:, 1] / points[:, 2] + principal[1]
    return predictions


def _decode_state(
    parameters: np.ndarray,
    observed: np.ndarray,
    principal: np.ndarray,
    camera_translation: np.ndarray,
) -> _State:
    frame_count = observed.shape[0]
    phase_parameter_count = frame_count - 2
    phase = _monotonic_phase(parameters[:phase_parameter_count], frame_count)
    coefficient = parameters[phase_parameter_count : phase_parameter_count + 6].reshape(2, 3)
    root_basis = np.stack([np.sin(phase), np.cos(phase)], axis=1)
    root = root_basis @ coefficient
    focal = float(parameters[-1])
    anchors = _profile_anchors(
        observed,
        phase,
        root,
        focal,
        principal,
        camera_translation,
    )
    reprojection = _project(anchors, phase, root, focal, principal, camera_translation)
    return _State(phase=phase, root=root, focal=focal, anchors=anchors, reprojection=reprojection)


def _residual(
    parameters: np.ndarray,
    observed: np.ndarray,
    principal: np.ndarray,
    camera_translation: np.ndarray,
) -> np.ndarray:
    state = _decode_state(parameters, observed, principal, camera_translation)
    data = (state.reprojection - observed).reshape(-1)
    phase_count = observed.shape[0] - 2
    phase_regularizer = 0.05 * parameters[:phase_count]
    root_regularizer = 20.0 * parameters[phase_count : phase_count + 6]
    return np.concatenate([data, phase_regularizer, root_regularizer])


def _scaled_cross_block_correlation(jacobian: np.ndarray, phase_count: int) -> float:
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / np.maximum(norms[None, :], 1e-12)
    phase = normalized[:, :phase_count]
    nuisance = normalized[:, phase_count:]
    if phase.size == 0 or nuisance.size == 0:
        return 0.0
    return float(np.max(np.abs(phase.T @ nuisance)))


def _sha_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit_fixed_camera_factor_graph(payload: dict[str, Any]) -> FixedCameraFactorGraphSolution:
    """Run a nonzero sparse-TRF solve with material anchors profiled at every step."""
    observed = np.asarray(payload["observed_xy"], dtype=np.float64)
    if observed.ndim != 3 or observed.shape[2] != 2 or observed.shape[0] < 6:
        raise ValueError("observed_xy must have shape [frames>=6, anchors, 2]")
    principal = np.asarray(payload["principal_point"], dtype=np.float64)
    camera_translation = np.asarray(payload["camera_translation"], dtype=np.float64)
    initial_focal = float(payload["initial_focal_length_pixels"])
    frame_count = observed.shape[0]
    phase_count = frame_count - 2
    initial = np.zeros(phase_count + 7, dtype=np.float64)
    initial[-1] = initial_focal
    lower = np.full_like(initial, -np.inf)
    upper = np.full_like(initial, np.inf)
    lower[-1], upper[-1] = 0.9 * initial_focal, 1.1 * initial_focal

    baseline_state = _decode_state(initial, observed, principal, camera_translation)
    baseline_errors = np.linalg.norm(baseline_state.reprojection - observed, axis=2).reshape(-1)
    restart_results = []
    for restart in range(5):
        start = initial.copy()
        if restart:
            rng = np.random.default_rng(100 + restart)
            start[:phase_count] += rng.normal(0.0, 0.01, phase_count)
            start[phase_count : phase_count + 6] += rng.normal(0.0, 0.0002, 6)
        result = least_squares(
            _residual,
            start,
            args=(observed, principal, camera_translation),
            method="trf",
            tr_solver="lsmr",
            loss="soft_l1",
            f_scale=2.0,
            bounds=(lower, upper),
            max_nfev=int(payload.get("maximum_function_evaluations", 200)),
        )
        restart_results.append(result)
    best = min(restart_results, key=lambda item: float(np.sum(item.fun**2)))
    states = [
        _decode_state(item.x, observed, principal, camera_translation) for item in restart_results
    ]
    state = _decode_state(best.x, observed, principal, camera_translation)
    errors = np.linalg.norm(state.reprojection - observed, axis=2).reshape(-1)
    singular = np.linalg.svd(best.jac, compute_uv=False)
    fold_ranks: list[int] = []
    rows_per_frame = observed.shape[1] * 2
    data_row_count = frame_count * rows_per_frame
    for fold in range(5):
        selected_frames = [index for index in range(frame_count) if index % 5 != fold]
        row_indices = np.concatenate(
            [
                np.arange(index * rows_per_frame, (index + 1) * rows_per_frame)
                for index in selected_frames
            ]
        )
        fold_jacobian = best.jac[row_indices[row_indices < data_row_count]]
        fold_singular = np.linalg.svd(fold_jacobian, compute_uv=False)
        threshold = max(float(fold_singular[0]) * 1e-6, 1e-9)
        fold_ranks.append(int(np.sum(fold_singular > threshold)))

    phases = np.stack([item.phase for item in states])
    roots = np.stack([item.root for item in states])
    restart_errors = np.asarray(
        [np.median(np.linalg.norm(item.reprojection - observed, axis=2)) for item in states]
    )
    phase_spread = float(np.max(np.ptp(np.rad2deg(phases), axis=0)))
    root_spread = float(np.max(np.linalg.norm(roots - np.median(roots, axis=0), axis=2)) * 1000.0)
    reprojection_spread = float(np.ptp(restart_errors))
    correlation = _scaled_cross_block_correlation(best.jac, phase_count)
    improvement = 1.0 - float(np.median(errors)) / max(float(np.median(baseline_errors)), 1e-9)
    blockers: list[str] = []
    if improvement < 0.2:
        blockers.append("median_reprojection_improvement_below_20_percent")
    if np.any(np.diff(state.phase) <= 0.0) or abs(state.phase[-1] - 2.0 * np.pi) > 1e-8:
        blockers.append("monotonic_full_turn_failure")
    if correlation >= 0.95:
        blockers.append("scaled_nuisance_geometry_correlation")
    if min(fold_ranks) != max(fold_ranks):
        blockers.append("informative_rank_unstable_across_folds")
    if phase_spread > 1.0 or root_spread > 5.0 or reprojection_spread > 2.5:
        blockers.append("deterministic_restart_stability")
    for external_gate in (
        "phase_fold_silhouette_nonregression",
        "phase_fold_boundary_nonregression",
    ):
        if not bool(payload.get(external_gate, False)):
            blockers.append(external_gate)

    checkpoint_hash = _sha_arrays(best.x, state.anchors)
    next_residual_a = _residual(best.x, observed, principal, camera_translation)
    next_residual_b = _residual(best.x, observed, principal, camera_translation)
    if not np.array_equal(next_residual_a, next_residual_b):
        blockers.append("exact_next_step_replay_failure")
    return FixedCameraFactorGraphSolution(
        experiment_id=EXPERIMENT_ID,
        evidence_scope=str(payload.get("evidence_scope", "public_synthetic")),  # type: ignore[arg-type]
        promotion_eligible=not blockers and payload.get("evidence_scope") == "train_real",
        physical_camera_extrinsics_sha256=str(payload["physical_camera_extrinsics_sha256"]),
        intrinsics=CameraIntrinsics(
            focal_length_pixels=state.focal,
            principal_point=(float(principal[0]), float(principal[1])),
            distortion=(),
        ),
        global_spin_axis=(0.0, 1.0, 0.0),
        monotonic_phase_radians=[float(value) for value in state.phase],
        root_translation_residuals_m=[
            (float(row[0]), float(row[1]), float(row[2])) for row in state.root
        ],
        pose_residual_norms=[0.0] * frame_count,
        profiled_material_anchor_count=observed.shape[1],
        jacobian_singular_values=[float(value) for value in singular],
        informative_rank_by_fold=fold_ranks,
        maximum_scaled_block_correlation=correlation,
        median_reprojection_pixels=float(np.median(errors)),
        p95_reprojection_pixels=float(np.percentile(errors, 95)),
        baseline_median_reprojection_pixels=float(np.median(baseline_errors)),
        restart_phase_spread_degrees=phase_spread,
        restart_root_spread_mm=root_spread,
        restart_reprojection_spread_pixels=reprojection_spread,
        checkpoint_sha256=checkpoint_hash,
        next_step_replay_sha256=_sha_arrays(next_residual_a),
        status="pass" if not blockers else "fail",
        blockers=blockers,
    )


def public_factor_graph_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(23)
    frame_count = 16
    anchor_count = 12
    increments = 1.0 + 0.08 * np.sin(np.linspace(0.0, 2.0 * np.pi, frame_count - 1))
    phases = np.concatenate([np.zeros(1), 2.0 * np.pi * np.cumsum(increments) / np.sum(increments)])
    anchors = np.column_stack(
        [
            rng.uniform(-0.35, 0.35, anchor_count),
            rng.uniform(-0.45, 0.45, anchor_count),
            rng.uniform(-0.12, 0.12, anchor_count),
        ]
    )
    root = np.column_stack([0.006 * np.sin(phases), 0.004 * np.cos(phases), 0.003 * np.sin(phases)])
    focal = 800.0
    principal = np.asarray([320.0, 240.0])
    camera_translation = np.asarray([0.0, 0.0, 3.0])
    observed = _project(anchors, phases, root, focal, principal, camera_translation)
    observed += rng.normal(0.0, 0.08, observed.shape)
    return {
        "observed_xy": observed.tolist(),
        "principal_point": principal.tolist(),
        "camera_translation": camera_translation.tolist(),
        "initial_focal_length_pixels": 760.0,
        "physical_camera_extrinsics_sha256": "3" * 64,
        "phase_fold_silhouette_nonregression": True,
        "phase_fold_boundary_nonregression": True,
        "maximum_function_evaluations": 120,
    }
