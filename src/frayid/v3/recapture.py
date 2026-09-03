from __future__ import annotations

from typing import Any

import numpy as np

EXPERIMENT_ID = "postv3_a01_information_gain_recapture_r01"
TRIGGER_STAGES = ("postv3_q04", "postv3_t06", "postv3_l04", "postv3_l05")


def plan_recapture(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a planning-only recapture branch from a frozen failed gate."""
    payload = payload or {}
    trigger_experiment = str(payload.get("trigger_experiment_id", ""))
    failed_gates = [str(value) for value in payload.get("failed_gates", [])]
    active = bool(failed_gates) and trigger_experiment.startswith(TRIGGER_STAGES)
    ranked_angles: list[dict[str, float]] = []
    current_raw = payload.get("current_information_matrix")
    candidates_raw = payload.get("candidate_jacobians", [])
    information_gain_basis: dict[str, Any] | None = None
    observed_phase_bins_raw = payload.get("observed_phase_bins")
    if current_raw is None and trigger_experiment.startswith("postv3_q04"):
        if not isinstance(observed_phase_bins_raw, list):
            raise ValueError("failed Q04 recapture planning requires observed_phase_bins")
        observed_phase_bins = sorted({int(value) for value in observed_phase_bins_raw})
        if not observed_phase_bins or any(value < 0 or value > 11 for value in observed_phase_bins):
            raise ValueError("observed_phase_bins must be a non-empty subset of [0,11]")

        def periodic_support_jacobian(angle_degrees: float) -> np.ndarray:
            angle = np.deg2rad(angle_degrees)
            return np.asarray(
                [
                    1.0,
                    np.cos(angle),
                    np.sin(angle),
                    np.cos(2.0 * angle),
                    np.sin(2.0 * angle),
                ]
            )

        ridge = 1.0e-3
        current = np.eye(5, dtype=np.float64) * ridge
        for phase_bin in observed_phase_bins:
            row = periodic_support_jacobian(30.0 * phase_bin)
            current += np.outer(row, row)
        current_raw = current.tolist()
        candidates_raw = [
            {
                "angle_degrees": float(angle),
                "jacobian": [periodic_support_jacobian(float(angle)).tolist()],
            }
            for angle in range(0, 360, 10)
        ]
        information_gain_basis = {
            "kind": "q04_periodic_chart_support_design_jacobian",
            "parameter_basis": ["constant", "cos_phase", "sin_phase", "cos_2phase", "sin_2phase"],
            "observed_phase_bins": observed_phase_bins,
            "ridge": ridge,
            "interpretation": "capture-order_prioritization_not_a_geometry_accuracy_claim",
        }
    if current_raw is not None:
        current = np.asarray(current_raw, dtype=np.float64)
        if current.ndim != 2 or current.shape[0] != current.shape[1]:
            raise ValueError("current_information_matrix must be square")
        sign, base_logdet = np.linalg.slogdet(current)
        if sign <= 0:
            raise ValueError("current information matrix must be positive definite")
        if not isinstance(candidates_raw, list):
            raise ValueError("candidate_jacobians must be a list")
        for item in candidates_raw:
            if not isinstance(item, dict):
                raise ValueError("each candidate Jacobian must be an object")
            jacobian = np.asarray(item["jacobian"], dtype=np.float64)
            if jacobian.ndim != 2 or jacobian.shape[1] != current.shape[0]:
                raise ValueError("candidate Jacobian has incompatible parameter dimension")
            candidate_sign, candidate_logdet = np.linalg.slogdet(current + jacobian.T @ jacobian)
            if candidate_sign <= 0:
                raise ValueError("candidate update produced non-positive information")
            ranked_angles.append(
                {
                    "angle_degrees": float(item["angle_degrees"]),
                    "expected_logdet_gain": float(candidate_logdet - base_logdet),
                }
            )
        ranked_angles.sort(key=lambda item: item["expected_logdet_gain"], reverse=True)
        if information_gain_basis is None:
            information_gain_basis = {
                "kind": "caller_supplied_route_a_observation_jacobian",
                "interpretation": "capture-order_prioritization_not_a_geometry_accuracy_claim",
            }

    return {
        "schema_version": "frayid_v3_information_gain_recapture_plan.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "active_planning" if active else "dormant",
        "planning_only": True,
        "capture_authorized": False,
        "trigger_experiment_id": trigger_experiment or None,
        "failed_gates": failed_gates,
        "protocol": {
            "training_camera": "one_fixed_tripod_camera",
            "angles_degrees": list(range(0, 360, 10)),
            "directions": ["clockwise", "counter_clockwise"],
            "hold_seconds_per_angle": 2.0,
            "resolution": "4K_where_available",
            "short_shutter": True,
            "manual_exposure_focus_white_balance": True,
            "fixed_diffuse_lighting": True,
            "visible_background_fiducials": True,
            "native_timing_and_lossless_evidence": True,
            "forbidden_processing": [
                "stabilization",
                "denoising",
                "interpolation",
                "generated_views",
                "baked_normalization",
            ],
            "temporary_garment_texture": "excluded_requires_separate_owner_authorization",
        },
        "second_camera": {
            "role": "evaluator_only",
            "allowed_outputs": ["sparse_stereo_surface_points", "boundary_points"],
            "fitting_access": False,
            "parameter_selection_access": False,
            "prior_access": False,
            "metric_claim_gates_mm": {"surface_median": 5.0, "surface_p95": 10.0, "boundary": 5.0},
        },
        "additional_angle_ranking": ranked_angles,
        "information_gain_basis": information_gain_basis,
        "successor_contracts": [
            "postv3_v01_controlled_recapture_evidence_master_r01",
            "postv3_q05_controlled_material_chart_graph_r01",
            "postv3_t07_controlled_fixed_camera_factor_graph_r01",
            "postv3_l06_controlled_upper_garment_material_atlas_r01",
        ],
        "sealed_test_accesses": 0,
    }
