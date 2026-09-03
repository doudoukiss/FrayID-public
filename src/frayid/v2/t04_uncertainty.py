from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from frayid.dataset import read_dataset_manifest
from frayid.io import read_json, sha256_file, write_json
from frayid.schemas import SequenceInitialization
from frayid.v2.contracts import (
    QualificationState,
    advance_qualification,
    reject_sealed_capability,
)
from frayid.v2.schemas import DynamicCameraFrame, DynamicCameraSolution
from frayid.v2.t02_geodesic import _linear_with_extrapolation, _slerp_with_extrapolation


@dataclass(frozen=True)
class CameraUncertainty:
    rotation_inconsistency_degrees: np.ndarray
    translation_inconsistency_metres: np.ndarray
    confidence: np.ndarray


def leave_one_out_camera_uncertainty(
    source_frame_indices: np.ndarray,
    global_orient: np.ndarray,
    translations: np.ndarray,
    *,
    rotation_scale_degrees: float = 5.0,
    translation_scale_metres: float = 0.05,
    confidence_floor: float = 0.05,
) -> CameraUncertainty:
    """Score each frozen camera against a trajectory fit without that camera."""

    sources = np.asarray(source_frame_indices, dtype=np.float64)
    orientations = np.asarray(global_orient, dtype=np.float64)
    positions = np.asarray(translations, dtype=np.float64)
    count = sources.size
    if count < 4:
        raise ValueError("leave-one-out camera uncertainty requires at least four frames")
    if sources.shape != (count,) or orientations.shape != (count, 3):
        raise ValueError("camera source/orientation arrays have invalid shape")
    if positions.shape != (count, 3):
        raise ValueError("camera translation array has invalid shape")
    if not np.all(np.diff(sources) > 0):
        raise ValueError("camera source indices must be unique and increasing")
    if not np.isfinite(orientations).all() or not np.isfinite(positions).all():
        raise ValueError("camera arrays must be finite")
    if rotation_scale_degrees <= 0 or translation_scale_metres <= 0:
        raise ValueError("camera uncertainty scales must be positive")
    if not 0 <= confidence_floor < 1:
        raise ValueError("camera confidence floor must lie in [0, 1)")

    rotations = Rotation.from_rotvec(orientations)
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for slot in range(count):
        keep = np.arange(count) != slot
        predicted_rotation = _slerp_with_extrapolation(
            sources[keep],
            rotations[keep],
            sources[slot : slot + 1],
        )
        relative = predicted_rotation.inv() * rotations[slot]
        rotation_errors.append(float(np.degrees(relative.magnitude()[0])))
        predicted_translation = _linear_with_extrapolation(
            sources[keep],
            positions[keep],
            sources[slot : slot + 1],
        )[0]
        translation_errors.append(float(np.linalg.norm(predicted_translation - positions[slot])))

    rotation_array = np.asarray(rotation_errors, dtype=np.float64)
    translation_array = np.asarray(translation_errors, dtype=np.float64)
    normalized_energy = np.square(rotation_array / rotation_scale_degrees) + np.square(
        translation_array / translation_scale_metres
    )
    confidence = np.maximum(confidence_floor, np.exp(-0.5 * normalized_energy))
    return CameraUncertainty(
        rotation_inconsistency_degrees=rotation_array,
        translation_inconsistency_metres=translation_array,
        confidence=confidence,
    )


def _synthetic_outlier_stress() -> dict[str, float | bool | int]:
    sources = np.arange(9, dtype=np.float64)
    clean_orientations = np.zeros((9, 3), dtype=np.float64)
    clean_orientations[:, 1] = 0.03 * sources
    clean_translations = np.column_stack(
        (0.01 * sources, np.zeros_like(sources), np.full_like(sources, 2.2))
    )
    outlier_slot = 4
    corrupted_orientations = clean_orientations.copy()
    clean_rotation = Rotation.from_rotvec(clean_orientations[outlier_slot])
    corruption = Rotation.from_rotvec(np.array([math.radians(15.0), 0.0, 0.0]))
    corrupted_orientations[outlier_slot] = (corruption * clean_rotation).as_rotvec()
    corrupted_translations = clean_translations.copy()
    corrupted_translations[outlier_slot, 0] += 0.10
    clean = leave_one_out_camera_uncertainty(sources, clean_orientations, clean_translations)
    corrupted = leave_one_out_camera_uncertainty(
        sources, corrupted_orientations, corrupted_translations
    )
    rotation_increase = (
        corrupted.rotation_inconsistency_degrees[outlier_slot]
        - clean.rotation_inconsistency_degrees[outlier_slot]
    )
    translation_increase = (
        corrupted.translation_inconsistency_metres[outlier_slot]
        - clean.translation_inconsistency_metres[outlier_slot]
    )
    confidence_decrease = clean.confidence[outlier_slot] - corrupted.confidence[outlier_slot]
    return {
        "outlier_slot": outlier_slot,
        "rotation_inconsistency_increase_degrees": float(rotation_increase),
        "translation_inconsistency_increase_metres": float(translation_increase),
        "confidence_decrease": float(confidence_decrease),
        "pass": bool(
            rotation_increase > 10.0 and translation_increase > 0.075 and confidence_decrease > 0.5
        ),
    }


def _build_solution(
    initialization: SequenceInitialization,
    training_source_indices: list[int],
    *,
    initialization_sha256: str,
    manifest_sha256: str,
) -> DynamicCameraSolution:
    initialization_by_source = {frame.source_frame_index: frame for frame in initialization.frames}
    missing = [
        source for source in training_source_indices if source not in initialization_by_source
    ]
    if missing:
        raise ValueError(f"T04 initialization is missing training frames: {missing[:5]}")
    source_array = np.asarray(training_source_indices, dtype=np.float64)
    orientation_array = np.asarray(
        [initialization_by_source[source].global_orient for source in training_source_indices],
        dtype=np.float64,
    )
    translation_array = np.asarray(
        [initialization_by_source[source].translation for source in training_source_indices],
        dtype=np.float64,
    )
    uncertainty = leave_one_out_camera_uncertainty(
        source_array, orientation_array, translation_array
    )
    focal = initialization.shared_focal_length_px
    principal = initialization.shared_principal_point_px
    return DynamicCameraSolution(
        status="qualification_candidate",
        shared_intrinsics=[
            [focal, 0.0, principal[0]],
            [0.0, focal, principal[1]],
            [0.0, 0.0, 1.0],
        ],
        frames=[
            DynamicCameraFrame(
                source_frame_index=source,
                global_orient=list(initialization_by_source[source].global_orient),
                translation=list(initialization_by_source[source].translation),
                rotation_inconsistency_degrees=float(
                    uncertainty.rotation_inconsistency_degrees[slot]
                ),
                translation_inconsistency_metres=float(
                    uncertainty.translation_inconsistency_metres[slot]
                ),
                confidence=float(uncertainty.confidence[slot]),
            )
            for slot, source in enumerate(training_source_indices)
        ],
        gauge_policy={
            "camera_parameters": "frozen_camerahmr_initialization_exact",
            "optimization": "none",
            "downstream_use": "confidence_weighted_evidence_only",
        },
        uncertainty_policy={
            "estimator": "leave_one_out_slerp_and_linear_interpolation",
            "rotation_scale_degrees": 5.0,
            "translation_scale_metres": 0.05,
            "confidence_floor": 0.05,
        },
        source_provenance={
            "initialization_sha256": initialization_sha256,
            "manifest_sha256": manifest_sha256,
            "camera_source": "camerahmr",
            "interpretation": "initialization_evidence_not_calibrated_truth",
        },
    )


def qualify_uncertainty_tagged_dynamic_camera(
    initialization_path: Path,
    manifest_path: Path,
    solution_output_path: Path,
    report_output_path: Path,
    *,
    device: str = "cpu",
) -> Path:
    """Write exact frozen cameras with deterministic train-only uncertainty."""

    paths = [
        initialization_path,
        manifest_path,
        solution_output_path,
        report_output_path,
    ]
    reject_sealed_capability(paths)
    if device != "cpu":
        raise ValueError("T04 qualification is registered for deterministic Mac CPU")
    if solution_output_path.exists() or report_output_path.exists():
        raise FileExistsError("T04 outputs are immutable")
    initialization = SequenceInitialization.model_validate(read_json(initialization_path))
    manifest = read_dataset_manifest(manifest_path)
    training = sorted(
        (frame for frame in manifest.frames if frame.split == "train" and frame.quality_accepted),
        key=lambda frame: frame.source_frame_index,
    )
    if len(training) != manifest.train_frame_count:
        raise ValueError("T04 must cover the complete accepted training set")
    training_sources = [frame.source_frame_index for frame in training]
    if len(training_sources) != len(set(training_sources)):
        raise ValueError("T04 training source frames must be unique")
    initialization_hash = sha256_file(initialization_path)
    manifest_hash = sha256_file(manifest_path)
    solution = _build_solution(
        initialization,
        training_sources,
        initialization_sha256=initialization_hash,
        manifest_sha256=manifest_hash,
    )
    replay = _build_solution(
        initialization,
        training_sources,
        initialization_sha256=initialization_hash,
        manifest_sha256=manifest_hash,
    )
    same_device_replay_exact = solution.model_dump(mode="json") == replay.model_dump(mode="json")
    initialization_by_source = {frame.source_frame_index: frame for frame in initialization.frames}
    expected_intrinsics = [
        [initialization.shared_focal_length_px, 0.0, initialization.shared_principal_point_px[0]],
        [0.0, initialization.shared_focal_length_px, initialization.shared_principal_point_px[1]],
        [0.0, 0.0, 1.0],
    ]
    initialization_parameters_exact = solution.shared_intrinsics == expected_intrinsics and all(
        output.global_orient == initialization_by_source[output.source_frame_index].global_orient
        and output.translation == initialization_by_source[output.source_frame_index].translation
        for output in solution.frames
    )
    confidence = np.asarray([frame.confidence for frame in solution.frames])
    rotation_uncertainty = np.asarray(
        [frame.rotation_inconsistency_degrees for frame in solution.frames]
    )
    translation_uncertainty = np.asarray(
        [frame.translation_inconsistency_metres for frame in solution.frames]
    )
    uncertainty_finite_and_bounded = bool(
        np.isfinite(confidence).all()
        and np.isfinite(rotation_uncertainty).all()
        and np.isfinite(translation_uncertainty).all()
        and np.all((confidence >= 0.0) & (confidence <= 1.0))
        and np.all(rotation_uncertainty >= 0.0)
        and np.all(translation_uncertainty >= 0.0)
    )
    synthetic_stress = _synthetic_outlier_stress()
    blockers: list[str] = []
    if not initialization_parameters_exact:
        blockers.append("camera_initialization_parameters_changed")
    if not uncertainty_finite_and_bounded:
        blockers.append("camera_uncertainty_not_finite_and_bounded")
    if not synthetic_stress["pass"]:
        blockers.append("synthetic_camera_outlier_not_detected")
    if len(solution.frames) != 144 or len(solution.frames) != manifest.train_frame_count:
        blockers.append("complete_training_frame_coverage_failed")
    if not same_device_replay_exact:
        blockers.append("same_device_replay_mismatch")
    solution.status = "pass" if not blockers else "fail"
    write_json(solution_output_path, solution)
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_t04_uncertainty_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_id": "postv2_t04_uncertainty_tagged_dynamic_camera_r01",
        "device": device,
        "dtype": "float64_uncertainty_json_source_values",
        "training_frame_count": len(solution.frames),
        "training_frame_coverage_fraction": len(solution.frames)
        / max(manifest.train_frame_count, 1),
        "camera_initialization_parameters_exact": initialization_parameters_exact,
        "uncertainty_finite_and_bounded": uncertainty_finite_and_bounded,
        "rotation_inconsistency_degrees": {
            "minimum": float(rotation_uncertainty.min()),
            "median": float(np.median(rotation_uncertainty)),
            "maximum": float(rotation_uncertainty.max()),
        },
        "translation_inconsistency_metres": {
            "minimum": float(translation_uncertainty.min()),
            "median": float(np.median(translation_uncertainty)),
            "maximum": float(translation_uncertainty.max()),
        },
        "confidence": {
            "minimum": float(confidence.min()),
            "median": float(np.median(confidence)),
            "maximum": float(confidence.max()),
        },
        "synthetic_outlier_stress": synthetic_stress,
        "same_device_replay_exact": same_device_replay_exact,
        "solution_path": str(solution_output_path),
        "solution_sha256": sha256_file(solution_output_path),
        "input_hashes": {
            "initialization": initialization_hash,
            "manifest": manifest_hash,
        },
        "blockers": blockers,
        "optimizer_steps": 0,
        "training_masks_read": 0,
        "training_rgb_images_read": 0,
        "training_tracks_read": 0,
        "legacy_development_images_read": 0,
        "sealed_test_accesses": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "CameraHMR parameters are copied exactly and are initialization evidence, not truth.",
            "Uncertainty changes downstream evidence weight; it never changes a camera parameter.",
            "Qualification reads only the accepted train manifest records and initialization JSON.",
        ],
    }
    return write_json(report_output_path, report)


def audit_t04_qualification_lifecycle(
    solution_path: Path,
    qualification_report_path: Path,
    output_path: Path,
) -> Path:
    """Restore T04 artifacts and record every fail-closed lifecycle transition."""

    reject_sealed_capability([solution_path, qualification_report_path, output_path])
    if output_path.exists():
        raise FileExistsError("T04 lifecycle records are immutable")
    solution = DynamicCameraSolution.model_validate(read_json(solution_path))
    report = read_json(qualification_report_path)
    checks = {
        "module_imported": True,
        "real_train_data_bound": report.get("training_frame_count") == 144,
        "mac_cpu_device_validated": report.get("device") == "cpu",
        "deterministic_transform_passed": report.get("optimizer_steps") == 0
        and report.get("camera_initialization_parameters_exact") is True,
        "immutable_solution_restored": solution.status == "pass"
        and report.get("solution_sha256") == sha256_file(solution_path),
        "evaluator_dry_run_passed": report.get("synthetic_outlier_stress", {}).get("pass") is True,
        "access_boundary_passed": report.get("legacy_development_images_read") == 0
        and report.get("sealed_test_accesses") == 0,
        "same_device_replay_exact": report.get("same_device_replay_exact") is True,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if report.get("status") != "pass":
        blockers.append("qualification_report_not_passing")
    state = QualificationState.BUILT
    transitions: list[dict[str, str]] = []
    transition_evidence = {
        QualificationState.IMPORTED: "module_imported",
        QualificationState.DATA_BOUND: "real_train_data_bound",
        QualificationState.DEVICE_VALIDATED: "mac_cpu_device_validated",
        QualificationState.ONE_STEP_PASSED: "deterministic_transform_passed",
        QualificationState.CHECKPOINT_RESTORED: "immutable_solution_restored",
        QualificationState.EVALUATOR_DRY: "evaluator_dry_run_passed",
        QualificationState.QUALIFIED: "access_boundary_passed",
    }
    if not blockers:
        for requested, evidence in transition_evidence.items():
            previous = state
            state = advance_qualification(state, requested)
            transitions.append(
                {
                    "from": previous.value,
                    "to": state.value,
                    "evidence": evidence,
                }
            )
    record: dict[str, Any] = {
        "schema_version": "frayid_v2_t04_qualification_lifecycle.v1",
        "experiment_id": "postv2_t04_uncertainty_tagged_dynamic_camera_r01",
        "status": "pass" if state is QualificationState.QUALIFIED else "fail",
        "state": state.value,
        "checks": checks,
        "transitions": transitions,
        "solution_sha256": sha256_file(solution_path),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "blockers": blockers,
        "private_reads": 2,
        "training_images_read": 0,
        "development_reads": 0,
        "sealed_test_reads": 0,
        "attempt_marker_created": False,
        "optimizer_steps": 0,
        "note": (
            "ONE_STEP_PASSED denotes one deterministic uncertainty transform pass; "
            "T04 has no optimizer or checkpoint by registered design."
        ),
    }
    return write_json(output_path, record)
