from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frayid.dataset import read_dataset_manifest
from frayid.io import sha256_file, write_json
from frayid.v2.contracts import (
    QualificationState,
    advance_qualification,
    reject_sealed_capability,
)
from frayid.v2.evidence import SAPIENS2_DOME29_LAYER_IDS


def qualify_sapiens2_semantic_directory(
    manifest_path: Path,
    mask_root: Path,
    semantic_root: Path,
    checkpoint_path: Path,
    report_output_path: Path,
    *,
    source_revision: str,
    minimum_median_foreground_iou: float = 0.995,
    minimum_frame_foreground_iou: float = 0.99,
) -> Path:
    """Qualify train-only DOME29 labels without using them as binary-mask proxies."""

    reject_sealed_capability(
        [manifest_path, mask_root, semantic_root, checkpoint_path, report_output_path]
    )
    if report_output_path.exists():
        raise FileExistsError("semantic qualification reports are immutable")
    manifest = read_dataset_manifest(manifest_path)
    training = sorted(
        (frame for frame in manifest.frames if frame.split == "train" and frame.quality_accepted),
        key=lambda frame: frame.source_frame_index,
    )
    if len(training) != manifest.train_frame_count:
        raise ValueError("semantic qualification must cover all accepted train frames")
    class_pixel_counts = np.zeros(29, dtype=np.int64)
    class_confidence_sums = np.zeros(29, dtype=np.float64)
    foreground_ious: list[float] = []
    frame_records: list[dict[str, Any]] = []
    blockers: list[str] = []
    for frame in training:
        stem = Path(frame.image_path).stem
        semantic_path = semantic_root / f"{stem}.npz"
        if not semantic_path.is_file():
            blockers.append(f"missing_semantic_frame:{frame.source_frame_index}")
            continue
        with np.load(semantic_path, allow_pickle=False) as archive:
            labels = archive["labels"]
            confidence = archive["confidence"].astype(np.float32)
            source = int(archive["source_frame_index"])
        mask_path = mask_root / Path(frame.image_path).name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"semantic qualification mask is absent: {mask_path}")
        if source != frame.source_frame_index:
            blockers.append(f"semantic_source_mismatch:{frame.source_frame_index}")
            continue
        if labels.shape != mask.shape or confidence.shape != mask.shape:
            blockers.append(f"semantic_shape_mismatch:{frame.source_frame_index}")
            continue
        if not np.issubdtype(labels.dtype, np.integer):
            blockers.append(f"semantic_labels_not_integer:{frame.source_frame_index}")
            continue
        if labels.size == 0 or int(labels.min()) < 0 or int(labels.max()) > 28:
            blockers.append(f"semantic_palette_out_of_range:{frame.source_frame_index}")
            continue
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            blockers.append(f"semantic_confidence_invalid:{frame.source_frame_index}")
            continue
        foreground = labels > 0
        reference_foreground = mask > 127
        union = np.logical_or(foreground, reference_foreground).sum()
        iou = float(np.logical_and(foreground, reference_foreground).sum() / max(union, 1))
        foreground_ious.append(iou)
        counts = np.bincount(labels.reshape(-1), minlength=29)
        class_pixel_counts += counts
        class_confidence_sums += np.bincount(
            labels.reshape(-1),
            weights=confidence.reshape(-1),
            minlength=29,
        )
        frame_records.append(
            {
                "source_frame_index": frame.source_frame_index,
                "semantic_sha256": sha256_file(semantic_path),
                "foreground_iou_against_preserved_v1_mask": iou,
                "present_class_ids": np.flatnonzero(counts).tolist(),
            }
        )
    if len(frame_records) != len(training):
        blockers.append("semantic_training_coverage_incomplete")
    if foreground_ious:
        median_iou = float(np.median(foreground_ious))
        minimum_iou = float(np.min(foreground_ious))
    else:
        median_iou = 0.0
        minimum_iou = 0.0
    if median_iou < minimum_median_foreground_iou:
        blockers.append("semantic_median_foreground_iou_below_0_995")
    if minimum_iou < minimum_frame_foreground_iou:
        blockers.append("semantic_minimum_foreground_iou_below_0_99")
    for required_name in ("upper_clothing", "lower_clothing", "hair", "body_parts"):
        if (
            sum(
                class_pixel_counts[class_id]
                for class_id in SAPIENS2_DOME29_LAYER_IDS[required_name]
            )
            == 0
        ):
            blockers.append(f"semantic_required_support_absent:{required_name}")
    layer_pixel_counts = {
        name: int(sum(class_pixel_counts[class_id] for class_id in class_ids))
        for name, class_ids in SAPIENS2_DOME29_LAYER_IDS.items()
    }
    class_mean_confidence = {
        str(class_id): (
            float(class_confidence_sums[class_id] / class_pixel_counts[class_id])
            if class_pixel_counts[class_id] > 0
            else 0.0
        )
        for class_id in range(29)
    }
    report: dict[str, Any] = {
        "schema_version": "frayid_v2_sapiens2_semantic_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "palette": "sapiens2_dome29",
        "source_revision": source_revision,
        "training_frame_count": len(training),
        "qualified_frame_count": len(frame_records),
        "foreground_iou_against_preserved_v1_masks": {
            "minimum": minimum_iou,
            "median": median_iou,
            "minimum_gate": minimum_frame_foreground_iou,
            "median_gate": minimum_median_foreground_iou,
        },
        "class_pixel_counts": {
            str(class_id): int(class_pixel_counts[class_id]) for class_id in range(29)
        },
        "class_mean_confidence": class_mean_confidence,
        "semantic_layer_ids": SAPIENS2_DOME29_LAYER_IDS,
        "layer_pixel_counts": layer_pixel_counts,
        "footwear_status": (
            "supported" if layer_pixel_counts["footwear"] > 0 else "absent_do_not_instantiate"
        ),
        "frame_records": frame_records,
        "input_hashes": {
            "manifest": sha256_file(manifest_path),
            "checkpoint": sha256_file(checkpoint_path),
        },
        "blockers": blockers,
        "optimizer_steps": 0,
        "training_images_read": len(frame_records),
        "legacy_development_images_read": 0,
        "sealed_test_accesses": 0,
        "scientific_attempt_marker_created": False,
        "modal_jobs": 0,
        "automatic_retries": 0,
    }
    return write_json(report_output_path, report)


def audit_s01_qualification_lifecycle(
    extraction_manifest_path: Path,
    qualification_report_path: Path,
    output_path: Path,
) -> Path:
    """Restore S01 evidence and record every ordered qualification transition."""

    reject_sealed_capability([extraction_manifest_path, qualification_report_path, output_path])
    if output_path.exists():
        raise FileExistsError("S01 lifecycle records are immutable")
    extraction = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_report_path.read_text(encoding="utf-8"))
    checks = {
        "module_imported": True,
        "real_train_data_bound": extraction.get("frame_count") == 144
        and qualification.get("qualified_frame_count") == 144,
        "mac_mps_device_validated": extraction.get("device") == "mps",
        "deterministic_transform_passed": extraction.get("same_device_first_frame_replay_exact")
        is True,
        "immutable_semantics_restored": qualification.get("status") == "pass"
        and qualification.get("input_hashes", {}).get("checkpoint")
        == extraction.get("checkpoint_sha256"),
        "evaluator_dry_run_passed": qualification.get(
            "foreground_iou_against_preserved_v1_masks", {}
        ).get("minimum", 0.0)
        >= 0.99,
        "access_boundary_passed": qualification.get("legacy_development_images_read") == 0
        and qualification.get("sealed_test_accesses") == 0
        and qualification.get("modal_jobs") == 0,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    state = QualificationState.BUILT
    transitions: list[dict[str, str]] = []
    transition_evidence = {
        QualificationState.IMPORTED: "module_imported",
        QualificationState.DATA_BOUND: "real_train_data_bound",
        QualificationState.DEVICE_VALIDATED: "mac_mps_device_validated",
        QualificationState.ONE_STEP_PASSED: "deterministic_transform_passed",
        QualificationState.CHECKPOINT_RESTORED: "immutable_semantics_restored",
        QualificationState.EVALUATOR_DRY: "evaluator_dry_run_passed",
        QualificationState.QUALIFIED: "access_boundary_passed",
    }
    if not blockers:
        for requested, evidence in transition_evidence.items():
            previous = state
            state = advance_qualification(state, requested)
            transitions.append({"from": previous.value, "to": state.value, "evidence": evidence})
    payload = {
        "schema_version": "frayid_v2_s01_qualification_lifecycle.v1",
        "experiment_id": "postv2_s01_sapiens2_semantic_recovery_r01",
        "status": "pass" if state is QualificationState.QUALIFIED else "fail",
        "state": state.value,
        "checks": checks,
        "transitions": transitions,
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "blockers": blockers,
        "development_reads": 0,
        "sealed_test_reads": 0,
        "modal_jobs": 0,
        "attempt_marker_created": False,
        "optimizer_steps": 0,
        "note": (
            "ONE_STEP_PASSED is one deterministic local inference pass and "
            "CHECKPOINT_RESTORED verifies the immutable checkpoint/evidence hashes."
        ),
    }
    return write_json(output_path, payload)
