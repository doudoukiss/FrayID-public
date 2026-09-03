from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

EXPERIMENT_ID = "postv3_t07_controlled_fixed_camera_factor_graph_r01"
DIRECTIONS = ("clockwise", "counter_clockwise")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _rotation_y(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _registered_angles() -> dict[str, np.ndarray]:
    return {
        "clockwise": np.arange(0.0, 360.0, 10.0, dtype=np.float64),
        "counter_clockwise": np.asarray([0.0, *range(350, 0, -10)], dtype=np.float64),
    }


def _unwrapped_nominal_phase(direction: str, angles_degrees: np.ndarray) -> np.ndarray:
    sign = 1.0 if direction == "clockwise" else -1.0
    return sign * np.deg2rad(np.arange(len(angles_degrees), dtype=np.float64) * 10.0)


@dataclass(frozen=True)
class _ControlledState:
    phases: np.ndarray
    roots: np.ndarray
    focal: float
    anchors: np.ndarray
    reprojection: np.ndarray


def _phase_and_root(
    parameters: np.ndarray,
    nominal_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phase_coefficients = parameters[:4].reshape(2, 2)
    root_coefficients = parameters[4:16].reshape(2, 2, 3)
    phases = np.empty_like(nominal_phase)
    roots = np.empty((*nominal_phase.shape, 3), dtype=np.float64)
    for direction_index in range(2):
        nominal = nominal_phase[direction_index]
        phase_basis = np.column_stack([np.sin(nominal), 1.0 - np.cos(nominal)])
        phases[direction_index] = nominal + phase_basis @ phase_coefficients[direction_index]
        root_basis = np.column_stack([np.sin(nominal), 1.0 - np.cos(nominal)])
        roots[direction_index] = root_basis @ root_coefficients[direction_index]
    return phases, roots


def _profile_anchors(
    observed: np.ndarray,
    visibility: np.ndarray,
    phases: np.ndarray,
    roots: np.ndarray,
    focal: float,
    principal: np.ndarray,
    camera_translation: np.ndarray,
) -> np.ndarray:
    anchor_count = observed.shape[2]
    anchors = np.zeros((anchor_count, 3), dtype=np.float64)
    for anchor_index in range(anchor_count):
        rows: list[np.ndarray] = []
        values: list[float] = []
        for direction_index in range(2):
            for frame_index in range(36):
                if not visibility[direction_index, frame_index, anchor_index]:
                    continue
                rotation = _rotation_y(float(phases[direction_index, frame_index]))
                translation = camera_translation + roots[direction_index, frame_index]
                xy = observed[direction_index, frame_index, anchor_index]
                x_norm = (xy[0] - principal[0]) / focal
                y_norm = (xy[1] - principal[1]) / focal
                rows.extend(
                    [rotation[0] - x_norm * rotation[2], rotation[1] - y_norm * rotation[2]]
                )
                values.extend(
                    [
                        float(x_norm * translation[2] - translation[0]),
                        float(y_norm * translation[2] - translation[1]),
                    ]
                )
        if len(rows) < 8:
            raise ValueError(f"material anchor {anchor_index} has fewer than four observations")
        anchors[anchor_index], *_ = np.linalg.lstsq(
            np.asarray(rows), np.asarray(values), rcond=None
        )
    return anchors


def _project(
    anchors: np.ndarray,
    phases: np.ndarray,
    roots: np.ndarray,
    focal: float,
    principal: np.ndarray,
    camera_translation: np.ndarray,
) -> np.ndarray:
    predictions = np.zeros((2, 36, len(anchors), 2), dtype=np.float64)
    for direction_index in range(2):
        for frame_index in range(36):
            points = (_rotation_y(float(phases[direction_index, frame_index])) @ anchors.T).T
            points += camera_translation + roots[direction_index, frame_index]
            predictions[direction_index, frame_index, :, 0] = (
                focal * points[:, 0] / points[:, 2] + principal[0]
            )
            predictions[direction_index, frame_index, :, 1] = (
                focal * points[:, 1] / points[:, 2] + principal[1]
            )
    return predictions


def _decode_state(
    parameters: np.ndarray,
    observed: np.ndarray,
    visibility: np.ndarray,
    nominal_phase: np.ndarray,
    principal: np.ndarray,
    camera_translation: np.ndarray,
) -> _ControlledState:
    phases, roots = _phase_and_root(parameters, nominal_phase)
    focal = float(parameters[-1])
    anchors = _profile_anchors(
        observed,
        visibility,
        phases,
        roots,
        focal,
        principal,
        camera_translation,
    )
    reprojection = _project(
        anchors,
        phases,
        roots,
        focal,
        principal,
        camera_translation,
    )
    return _ControlledState(phases, roots, focal, anchors, reprojection)


def _residual(
    parameters: np.ndarray,
    observed: np.ndarray,
    visibility: np.ndarray,
    nominal_phase: np.ndarray,
    principal: np.ndarray,
    camera_translation: np.ndarray,
) -> np.ndarray:
    state = _decode_state(
        parameters,
        observed,
        visibility,
        nominal_phase,
        principal,
        camera_translation,
    )
    data = (state.reprojection - observed)[visibility].reshape(-1)
    phase_regularizer = 5.0 * parameters[:4]
    root_regularizer = 20.0 * parameters[4:16]
    return np.concatenate([data, phase_regularizer, root_regularizer])


def _sha_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _scaled_cross_block_correlation(jacobian: np.ndarray) -> float:
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / np.maximum(norms[None, :], 1.0e-12)
    phase = normalized[:, :4]
    nuisance = normalized[:, 4:]
    return float(np.max(np.abs(phase.T @ nuisance)))


def _fold_ranks(
    jacobian: np.ndarray,
    visibility: np.ndarray,
) -> list[int]:
    observation_rows: list[tuple[int, int]] = []
    row = 0
    for direction_index in range(2):
        for frame_index in range(36):
            for anchor_index in range(visibility.shape[2]):
                if visibility[direction_index, frame_index, anchor_index]:
                    observation_rows.append((frame_index, row))
                    row += 2
    ranks: list[int] = []
    for fold in range(5):
        selected: list[int] = []
        for frame_index, start in observation_rows:
            if frame_index % 5 != fold:
                selected.extend((start, start + 1))
        fold_jacobian = jacobian[np.asarray(selected, dtype=np.int64)]
        singular = np.linalg.svd(fold_jacobian, compute_uv=False)
        threshold = max(float(singular[0]) * 1.0e-6, 1.0e-9)
        ranks.append(int(np.sum(singular > threshold)))
    return ranks


def _validate_payload(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = payload.get("observed_xy_by_direction")
    angles = payload.get("angle_degrees_by_direction")
    if not isinstance(observations, dict) or set(observations) != set(DIRECTIONS):
        raise ValueError("T07 requires clockwise and counter_clockwise observations")
    if not isinstance(angles, dict) or set(angles) != set(DIRECTIONS):
        raise ValueError("T07 requires both registered directional angle lists")
    expected = _registered_angles()
    stacked_observations: list[np.ndarray] = []
    nominal: list[np.ndarray] = []
    for direction in DIRECTIONS:
        observed = np.asarray(observations[direction], dtype=np.float64)
        supplied_angles = np.asarray(angles[direction], dtype=np.float64)
        if observed.ndim != 3 or observed.shape[0] != 36 or observed.shape[2] != 2:
            raise ValueError(f"{direction} observations must have shape [36, anchors, 2]")
        if not np.array_equal(supplied_angles, expected[direction]):
            raise ValueError(f"{direction} angles do not match the registered ten-degree order")
        stacked_observations.append(observed)
        nominal.append(_unwrapped_nominal_phase(direction, supplied_angles))
    observed_array = np.stack(stacked_observations)
    raw_visibility = payload.get("visibility_by_direction")
    if raw_visibility is None:
        visibility = np.all(np.isfinite(observed_array), axis=3)
    else:
        if not isinstance(raw_visibility, dict) or set(raw_visibility) != set(DIRECTIONS):
            raise ValueError("visibility_by_direction must contain both directions")
        visibility = np.stack(
            [np.asarray(raw_visibility[direction], dtype=bool) for direction in DIRECTIONS]
        )
        if visibility.shape != observed_array.shape[:3]:
            raise ValueError("visibility must have shape [2, 36, anchors]")
    if np.any(~np.isfinite(observed_array[visibility])):
        raise ValueError("visible observations must be finite")
    if min(np.sum(visibility, axis=(0, 1))) < 4:
        raise ValueError("every material anchor requires at least four visible observations")
    return observed_array, visibility, np.stack(nominal)


def fit_controlled_fixed_camera_factor_graph(payload: dict[str, Any]) -> dict[str, Any]:
    """Qualify the bidirectional known-angle T07 solve without evaluator access."""
    observed, visibility, nominal_phase = _validate_payload(payload)
    evidence_scope = str(payload.get("evidence_scope", "public_synthetic"))
    if evidence_scope not in {"public_synthetic", "train_real"}:
        raise ValueError("unsupported T07 evidence_scope")
    if bool(payload.get("evaluator_camera_fitting_access", True)):
        raise ValueError("evaluator camera cannot enter T07 fitting")
    physical_camera_hash = str(payload.get("physical_camera_extrinsics_sha256", ""))
    t05_hash = str(payload.get("t05_initialization_sha256", ""))
    if not _SHA256_PATTERN.fullmatch(physical_camera_hash):
        raise ValueError("physical_camera_extrinsics_sha256 must be a SHA-256 digest")
    if not _SHA256_PATTERN.fullmatch(t05_hash):
        raise ValueError("t05_initialization_sha256 must be a SHA-256 digest")
    source_hashes = payload.get("source_hashes", {})
    if evidence_scope == "train_real":
        if observed.shape[2] < 100:
            raise ValueError("real T07 requires at least 100 Q05 material anchors")
        if not isinstance(source_hashes, dict) or set(source_hashes) != {"v01", "q05", "t05"}:
            raise ValueError("real T07 must bind V01, Q05, and T05 source hashes")
        if not all(_SHA256_PATTERN.fullmatch(str(value)) for value in source_hashes.values()):
            raise ValueError("real T07 source hashes must be SHA-256 digests")
        if int(payload.get("training_records_read", -1)) != 72:
            raise ValueError("real T07 must read exactly 72 controlled hold records")
        if int(payload.get("development_records_read", -1)) != 0:
            raise ValueError("real T07 cannot read historical development records")
        if int(payload.get("sealed_test_accesses", -1)) != 0:
            raise ValueError("real T07 cannot read sealed evidence")
    principal = np.asarray(payload["principal_point"], dtype=np.float64)
    camera_translation = np.asarray(payload["camera_translation"], dtype=np.float64)
    if principal.shape != (2,) or camera_translation.shape != (3,):
        raise ValueError("principal_point and camera_translation have invalid shape")
    initial_focal = float(payload["initial_focal_length_pixels"])
    initial = np.zeros(17, dtype=np.float64)
    initial[-1] = initial_focal
    lower = np.concatenate(
        [
            np.full(4, np.deg2rad(-3.0)),
            np.full(12, -0.03),
            np.asarray([0.9 * initial_focal]),
        ]
    )
    upper = np.concatenate(
        [
            np.full(4, np.deg2rad(3.0)),
            np.full(12, 0.03),
            np.asarray([1.1 * initial_focal]),
        ]
    )
    baseline = _decode_state(
        initial,
        observed,
        visibility,
        nominal_phase,
        principal,
        camera_translation,
    )
    baseline_errors = np.linalg.norm(baseline.reprojection - observed, axis=3)[visibility]
    results = []
    for restart in range(5):
        start = initial.copy()
        if restart:
            rng = np.random.default_rng(700 + restart)
            start[:4] += rng.normal(0.0, np.deg2rad(0.05), 4)
            start[4:16] += rng.normal(0.0, 0.0001, 12)
        result = least_squares(
            _residual,
            start,
            args=(observed, visibility, nominal_phase, principal, camera_translation),
            method="trf",
            tr_solver="lsmr",
            loss="soft_l1",
            f_scale=2.0,
            bounds=(lower, upper),
            max_nfev=int(payload.get("maximum_function_evaluations", 160)),
        )
        results.append(result)
    best = min(results, key=lambda result: float(np.sum(result.fun**2)))
    states = [
        _decode_state(
            result.x,
            observed,
            visibility,
            nominal_phase,
            principal,
            camera_translation,
        )
        for result in results
    ]
    state = _decode_state(
        best.x,
        observed,
        visibility,
        nominal_phase,
        principal,
        camera_translation,
    )
    errors = np.linalg.norm(state.reprojection - observed, axis=3)[visibility]
    improvement = 1.0 - float(np.median(errors)) / max(float(np.median(baseline_errors)), 1.0e-9)
    singular = np.linalg.svd(best.jac, compute_uv=False)
    fold_ranks = _fold_ranks(best.jac, visibility)
    restart_phases = np.stack([candidate.phases for candidate in states])
    restart_roots = np.stack([candidate.roots for candidate in states])
    restart_errors = np.asarray(
        [
            np.median(np.linalg.norm(candidate.reprojection - observed, axis=3)[visibility])
            for candidate in states
        ]
    )
    phase_spread = float(np.max(np.ptp(np.rad2deg(restart_phases), axis=0)))
    median_roots = np.median(restart_roots, axis=0)
    root_spread = float(
        np.max(np.linalg.norm(restart_roots - median_roots[None, ...], axis=3)) * 1000.0
    )
    reprojection_spread = float(np.ptp(restart_errors))
    correlation = _scaled_cross_block_correlation(best.jac)
    progress = np.stack([state.phases[0], -state.phases[1]])
    blockers: list[str] = []
    if improvement < 0.2:
        blockers.append("median_reprojection_improvement_below_20_percent")
    if np.any(np.diff(progress, axis=1) <= 0.0):
        blockers.append("positive_directional_phase_increments_failed")
    if any(
        not np.array_equal(
            np.asarray(payload["angle_degrees_by_direction"][direction]),
            _registered_angles()[direction],
        )
        for direction in DIRECTIONS
    ):
        blockers.append("full_turn_orientation_changed")
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
    dependency_expected = "pass" if evidence_scope == "train_real" else "public_pass"
    if payload.get("v01_status") != dependency_expected:
        blockers.append("v01_dependency_not_qualified")
    if payload.get("q05_status") != dependency_expected:
        blockers.append("q05_dependency_not_qualified")
    replay_a = _residual(best.x, observed, visibility, nominal_phase, principal, camera_translation)
    replay_b = _residual(best.x, observed, visibility, nominal_phase, principal, camera_translation)
    if not np.array_equal(replay_a, replay_b):
        blockers.append("exact_next_step_replay_failure")
    blockers = sorted(set(blockers))
    result_payload = {
        "schema_version": "frayid_v3_controlled_fixed_camera_factor_graph_solution.v1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_scope": evidence_scope,
        "status": "pass" if not blockers else "fail",
        "promotion_eligible": not blockers and evidence_scope == "train_real",
        "physical_camera_extrinsics_sha256": physical_camera_hash,
        "t05_initialization_sha256": t05_hash,
        "source_hashes": source_hashes,
        "intrinsics": {
            "focal_length_pixels": state.focal,
            "principal_point": principal.tolist(),
            "distortion": [],
        },
        "angle_degrees_by_direction": payload["angle_degrees_by_direction"],
        "phase_radians_by_direction": {
            direction: state.phases[index].tolist() for index, direction in enumerate(DIRECTIONS)
        },
        "root_translation_residuals_m_by_direction": {
            direction: state.roots[index].tolist() for index, direction in enumerate(DIRECTIONS)
        },
        "profiled_material_anchor_count": observed.shape[2],
        "visible_observation_count": int(np.sum(visibility)),
        "baseline_median_reprojection_pixels": float(np.median(baseline_errors)),
        "median_reprojection_pixels": float(np.median(errors)),
        "p95_reprojection_pixels": float(np.percentile(errors, 95)),
        "median_reprojection_improvement": improvement,
        "jacobian_singular_values": singular.tolist(),
        "informative_rank_by_fold": fold_ranks,
        "maximum_scaled_block_correlation": correlation,
        "restart_phase_spread_degrees": phase_spread,
        "restart_root_spread_mm": root_spread,
        "restart_reprojection_spread_pixels": reprojection_spread,
        "checkpoint_sha256": _sha_arrays(best.x, state.anchors),
        "next_step_replay_sha256": _sha_arrays(replay_a),
        "exact_same_device_next_step_replay": np.array_equal(replay_a, replay_b),
        "v01_status": payload.get("v01_status"),
        "q05_status": payload.get("q05_status"),
        "evaluator_camera_decoded_for_fitting": False,
        "evaluator_camera_fitting_access": False,
        "training_records_read": 0 if evidence_scope == "public_synthetic" else 72,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer": "scipy_sparse_trust_region_lsmr_profiled_material_anchors",
        "restart_count": 5,
        "blockers": blockers,
    }
    result_payload["exact_replay_hash"] = hashlib.sha256(
        json.dumps(result_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result_payload


def public_controlled_factor_graph_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(71)
    registered = _registered_angles()
    nominal = np.stack(
        [_unwrapped_nominal_phase(direction, registered[direction]) for direction in DIRECTIONS]
    )
    true_parameters = np.zeros(17, dtype=np.float64)
    true_parameters[:4] = np.deg2rad([0.8, -0.5, -0.7, 0.4])
    true_parameters[4:16] = np.asarray(
        [
            [0.005, 0.0, 0.002],
            [0.001, 0.003, -0.001],
            [-0.004, 0.001, 0.002],
            [0.002, -0.002, 0.001],
        ]
    ).reshape(-1)
    true_parameters[-1] = 800.0
    phases, roots = _phase_and_root(true_parameters, nominal)
    anchor_count = 10
    anchors = np.column_stack(
        [
            rng.uniform(-0.34, 0.34, anchor_count),
            rng.uniform(-0.42, 0.42, anchor_count),
            rng.uniform(-0.14, 0.14, anchor_count),
        ]
    )
    principal = np.asarray([320.0, 240.0], dtype=np.float64)
    camera_translation = np.asarray([0.0, 0.0, 3.0], dtype=np.float64)
    observed = _project(
        anchors,
        phases,
        roots,
        true_parameters[-1],
        principal,
        camera_translation,
    )
    observed += rng.normal(0.0, 0.08, observed.shape)
    return {
        "evidence_scope": "public_synthetic",
        "observed_xy_by_direction": {
            direction: observed[index].tolist() for index, direction in enumerate(DIRECTIONS)
        },
        "angle_degrees_by_direction": {
            direction: registered[direction].tolist() for direction in DIRECTIONS
        },
        "principal_point": principal.tolist(),
        "camera_translation": camera_translation.tolist(),
        "initial_focal_length_pixels": 760.0,
        "physical_camera_extrinsics_sha256": "7" * 64,
        "t05_initialization_sha256": "5" * 64,
        "source_hashes": {},
        "phase_fold_silhouette_nonregression": True,
        "phase_fold_boundary_nonregression": True,
        "evaluator_camera_fitting_access": False,
        "v01_status": "public_pass",
        "q05_status": "public_pass",
        "maximum_function_evaluations": 80,
    }


__all__ = [
    "fit_controlled_fixed_camera_factor_graph",
    "public_controlled_factor_graph_fixture",
]
