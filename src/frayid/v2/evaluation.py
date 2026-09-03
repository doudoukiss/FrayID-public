from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability

R03_HELD_OUT_IOU = 0.8778329789638519
R03_NORMALIZED_BOUNDARY_ERROR = 0.004556156724774018
R03_MEDIAN_NORMAL_ERROR_DEGREES = 21.374711990356445
R03_MAXIMUM_TRAIN_HELD_OUT_GAP = 0.05


def bidirectional_chamfer(first: np.ndarray, second: np.ndarray) -> float:
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != (3,) or second.shape[1:] != (3,):
        raise ValueError("Chamfer inputs must be nonempty [N,3] point arrays")
    if (
        not len(first)
        or not len(second)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("Chamfer inputs must be finite and nonempty")
    first_distance = cKDTree(second).query(first, workers=1)[0]
    second_distance = cKDTree(first).query(second, workers=1)[0]
    return float(0.5 * (first_distance.mean() + second_distance.mean()))


def symmetric_point_to_plane(
    first: np.ndarray,
    first_normals: np.ndarray,
    second: np.ndarray,
    second_normals: np.ndarray,
) -> float:
    if first.shape != first_normals.shape or second.shape != second_normals.shape:
        raise ValueError("point-to-plane normals must align with points")
    second_tree = cKDTree(second)
    first_tree = cKDTree(first)
    _, second_index = second_tree.query(first, workers=1)
    _, first_index = first_tree.query(second, workers=1)
    forward = np.abs(np.sum((first - second[second_index]) * second_normals[second_index], axis=1))
    backward = np.abs(np.sum((second - first[first_index]) * first_normals[first_index], axis=1))
    return float(0.5 * (forward.mean() + backward.mean()))


def normal_angular_distribution(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("normal arrays must have identical [N,3] shape")
    first_unit = first / np.clip(np.linalg.norm(first, axis=1, keepdims=True), 1e-12, None)
    second_unit = second / np.clip(np.linalg.norm(second, axis=1, keepdims=True), 1e-12, None)
    angles = np.degrees(np.arccos(np.clip(np.sum(first_unit * second_unit, axis=1), -1.0, 1.0)))
    return {
        "mean_degrees": float(angles.mean()),
        "median_degrees": float(np.median(angles)),
        "p90_degrees": float(np.quantile(angles, 0.9)),
        "p95_degrees": float(np.quantile(angles, 0.95)),
    }


def relative_improvement(control: float, treatment: float) -> float:
    if not math.isfinite(control) or not math.isfinite(treatment) or control <= 0:
        raise ValueError("relative improvement requires positive finite control")
    return (control - treatment) / control


def inherited_real_gate(metrics: dict[str, float]) -> list[str]:
    blockers: list[str] = []
    if metrics.get("held_out_iou", -math.inf) < R03_HELD_OUT_IOU:
        blockers.append("held_out_iou_below_r03")
    if metrics.get("normalized_boundary_error", math.inf) > R03_NORMALIZED_BOUNDARY_ERROR:
        blockers.append("boundary_worse_than_r03")
    if metrics.get("median_normal_error_degrees", math.inf) > R03_MEDIAN_NORMAL_ERROR_DEGREES:
        blockers.append("normal_worse_than_r03")
    if metrics.get("train_held_out_iou_gap", math.inf) > R03_MAXIMUM_TRAIN_HELD_OUT_GAP:
        blockers.append("train_held_out_gap")
    return blockers


def qualify_g01_evaluator_routes(
    public_benchmark_path: Path,
    r03_control_contract_path: Path,
    output_path: Path,
) -> Path:
    """Dry-run G01's independent public and historical real-control gates.

    No candidate is scored here. The result only proves that frozen fixtures,
    historical development controls, and fail-closed gate sensitivity are
    available before a scientific attempt.
    """

    reject_sealed_capability([public_benchmark_path, r03_control_contract_path, output_path])
    if output_path.exists():
        raise FileExistsError("G01 evaluator qualification reports are immutable")
    public = read_json(public_benchmark_path)
    control = yaml.safe_load(r03_control_contract_path.read_text(encoding="utf-8"))
    if not isinstance(control, dict):
        raise ValueError("r03 control contract must be a mapping")
    replay = control.get("modal_replay", {})
    if not isinstance(replay, dict):
        raise ValueError("r03 control contract has no modal replay record")
    metrics = {
        "held_out_iou": float(replay.get("r03_held_out_iou", -math.inf)),
        "normalized_boundary_error": float(replay.get("r03_boundary_error", math.inf)),
        "median_normal_error_degrees": float(replay.get("r03_normal_degrees", math.inf)),
        "train_held_out_iou_gap": float(replay.get("train_iou", math.inf))
        - float(replay.get("r03_held_out_iou", -math.inf)),
    }
    real_gate_blockers = inherited_real_gate(metrics)
    cases = public.get("cases", {})
    required_cases = {
        "analytic_identity",
        "articulated_surrogate",
        "camera_root_ambiguity",
        "concave_pocket",
        "fine_wrinkles",
        "normal_corruption",
        "open_garment",
        "silhouette_perfect_geometry_wrong",
        "smooth_ellipsoid",
        "thin_gap_hairpin",
        "topology_change",
        "turntable_calibration_evaluator",
        "valid_layer_contact",
    }
    public_cases_complete = isinstance(cases, dict) and required_cases <= set(cases)
    public_distance_cases = (
        [
            name
            for name, values in cases.items()
            if isinstance(values, dict)
            and "control_chamfer" in values
            and "treatment_chamfer" in values
        ]
        if isinstance(cases, dict)
        else []
    )
    sensitivity = {
        "held_out_iou": "held_out_iou_below_r03"
        in inherited_real_gate({**metrics, "held_out_iou": R03_HELD_OUT_IOU - 1.0e-6}),
        "normalized_boundary_error": "boundary_worse_than_r03"
        in inherited_real_gate(
            {
                **metrics,
                "normalized_boundary_error": R03_NORMALIZED_BOUNDARY_ERROR + 1.0e-6,
            }
        ),
        "median_normal_error_degrees": "normal_worse_than_r03"
        in inherited_real_gate(
            {
                **metrics,
                "median_normal_error_degrees": R03_MEDIAN_NORMAL_ERROR_DEGREES + 1.0e-3,
            }
        ),
        "train_held_out_iou_gap": "train_held_out_gap"
        in inherited_real_gate(
            {**metrics, "train_held_out_iou_gap": R03_MAXIMUM_TRAIN_HELD_OUT_GAP + 1.0e-6}
        ),
    }
    blockers: list[str] = []
    if public.get("status") != "pass":
        blockers.append("public_benchmark_not_passing")
    if public.get("evaluator_independent_of_training_renderer") is not True:
        blockers.append("public_evaluator_not_independent")
    if not public_cases_complete or len(public_distance_cases) < 5:
        blockers.append("public_geometry_fixture_coverage_incomplete")
    if control.get("experiment_id") != "postv1_e00_original_reference_projection_safety_margin_r03":
        blockers.append("historical_control_identity_mismatch")
    if control.get("status") != "pass" or replay.get("status") != "pass":
        blockers.append("historical_control_not_passing")
    blockers.extend(real_gate_blockers)
    blockers.extend(name for name, passed in sensitivity.items() if not passed)
    report = {
        "schema_version": "frayid_v2_g01_evaluator_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "dry_run_only": True,
        "candidate_scored": False,
        "scientific_result_claimed": False,
        "public_benchmark_status": public.get("status"),
        "public_required_cases_complete": public_cases_complete,
        "public_distance_case_count": len(public_distance_cases),
        "public_evaluator_independent_of_training_renderer": public.get(
            "evaluator_independent_of_training_renderer"
        ),
        "historical_r03_control_metrics": metrics,
        "historical_r03_gate_blockers": real_gate_blockers,
        "fail_closed_sensitivity": sensitivity,
        "source_hashes": {
            "public_benchmark": sha256_file(public_benchmark_path),
            "r03_control_contract": sha256_file(r03_control_contract_path),
        },
        "blockers": blockers,
        "training_images_read": 0,
        "legacy_development_images_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "scientific_attempt_marker_created": False,
    }
    return write_json(output_path, report)
