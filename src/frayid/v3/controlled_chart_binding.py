from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.evidence import SAPIENS2_DOME29_LAYER_IDS

EXPERIMENT_ID = "postv3_q05_controlled_material_chart_graph_r01"
SOURCES = ("lk", "tapir", "cotracker3")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _require_report(path: Path, *, expected_status: str = "pass") -> dict[str, Any]:
    report = read_json(path)
    if report.get("status") != expected_status:
        raise ValueError(f"required report is not {expected_status}: {path}")
    return report


def _anchor_records(master: dict[str, Any]) -> list[dict[str, Any]]:
    clips = master.get("training_clips")
    if not isinstance(clips, list) or len(clips) != 2:
        raise ValueError("V01 master must contain two training clips")
    directions: set[str] = set()
    records: list[dict[str, Any]] = []
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("V01 training clip record must be an object")
        direction = str(clip.get("direction"))
        directions.add(direction)
        source = clip.get("source")
        if not isinstance(source, dict):
            raise ValueError("V01 clip has no native source binding")
        source_path = Path(str(source.get("path", "")))
        reject_sealed_capability([source_path])
        if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
            raise ValueError(f"V01 native source hash mismatch: {direction}")
        anchors = clip.get("material_chart_anchor_records")
        if not isinstance(anchors, list) or len(anchors) != 36:
            raise ValueError(f"V01 {direction} must expose exactly 36 material anchors")
        for anchor in anchors:
            if not isinstance(anchor, dict) or anchor.get("direction") != direction:
                raise ValueError("V01 material anchor direction mismatch")
            records.append({**anchor, "native_source_path": str(source_path)})
    if directions != {"clockwise", "counter_clockwise"}:
        raise ValueError("V01 material anchors require both directions")
    if [int(item["controlled_record_index"]) for item in records] != list(range(72)):
        raise ValueError("V01 controlled record indices must be exactly 0 through 71")
    if len({str(item["source_frame_key"]) for item in records}) != 72:
        raise ValueError("V01 source frame keys must be globally unique")
    return records


def _validate_tracker_audit(report: dict[str, Any], *, evidence_scope: str) -> None:
    sources = report.get("sources")
    if not isinstance(sources, list):
        raise ValueError("tracker source audit has no source inventory")
    by_source = {str(item.get("source")): item for item in sources if isinstance(item, dict)}
    if set(by_source) != set(SOURCES):
        raise ValueError("tracker source audit must cover LK, TAPIR, and CoTracker3")
    if evidence_scope == "train_real":
        if report.get("real_execution_ready") is not True or report.get("blockers"):
            raise ValueError("tracker source audit is not ready for real Q05 execution")
        for source, item in by_source.items():
            if item.get("license_ready_for_real_use") is not True:
                raise ValueError(f"tracker license is not ready for real Q05: {source}")


def _semantic_records(
    manifest: dict[str, Any],
    anchors: list[dict[str, Any]],
    *,
    evidence_scope: str,
    v01_sha256: str,
    proxy_coordinate_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    if manifest.get("schema_version") != "frayid_v3_controlled_semantic_evidence.v1":
        raise ValueError("unexpected controlled semantic evidence schema")
    if manifest.get("evidence_scope") != evidence_scope:
        raise ValueError("semantic evidence scope does not match the requested Q05 scope")
    if manifest.get("v01_evidence_master_sha256") != v01_sha256:
        raise ValueError("semantic evidence is not bound to this V01 master")
    if manifest.get("palette") != "sapiens2_dome29":
        raise ValueError("Q05 requires the preserved Sapiens2 DOME29 palette")
    if manifest.get("proxy_coordinate_contract") != proxy_coordinate_contract:
        raise ValueError("controlled semantics do not share V01 proxy coordinates")
    if int(manifest.get("evaluator_camera_reads", -1)) != 0:
        raise ValueError("evaluator camera cannot enter controlled semantic extraction")
    if (
        int(manifest.get("development_records_read", -1)) != 0
        or int(manifest.get("sealed_test_accesses", -1)) != 0
    ):
        raise ValueError("controlled semantics cannot read development or sealed evidence")
    checkpoint_sha = str(manifest.get("checkpoint_sha256", ""))
    if not _SHA256_PATTERN.fullmatch(checkpoint_sha):
        raise ValueError("controlled semantic evidence must bind its checkpoint SHA-256")
    if evidence_scope == "train_real" and checkpoint_sha == "0" * 64:
        raise ValueError("real controlled semantic evidence cannot use a placeholder checkpoint")
    if not str(manifest.get("source_revision", "")).strip():
        raise ValueError("controlled semantic evidence must bind its source revision")
    if manifest.get("exact_same_device_replay") is not True:
        raise ValueError("controlled semantic evidence must replay exactly on the same device")
    raw_records = manifest.get("frames")
    if not isinstance(raw_records, list) or len(raw_records) != 72:
        raise ValueError("controlled semantic evidence must contain exactly 72 frames")
    by_index = {
        int(record["controlled_record_index"]): record
        for record in raw_records
        if isinstance(record, dict)
    }
    if set(by_index) != set(range(72)):
        raise ValueError("controlled semantic frame indices must be exactly 0 through 71")
    verified: list[dict[str, Any]] = []
    semantic_paths: set[Path] = set()
    upper_ids = SAPIENS2_DOME29_LAYER_IDS["upper_clothing"]
    for anchor in anchors:
        index = int(anchor["controlled_record_index"])
        record = by_index[index]
        if record.get("source_frame_key") != anchor["source_frame_key"]:
            raise ValueError(f"semantic source-frame key mismatch: {index}")
        path = Path(str(record.get("semantic_path", "")))
        reject_sealed_capability([path])
        if path in semantic_paths:
            raise ValueError("each controlled anchor requires distinct semantic evidence")
        semantic_paths.add(path)
        if not path.is_file() or sha256_file(path) != record.get("semantic_sha256"):
            raise ValueError(f"semantic evidence hash mismatch: {index}")
        with np.load(path, allow_pickle=False) as archive:
            labels = np.asarray(archive["labels"])
            confidence = np.asarray(archive["confidence"], dtype=np.float32)
        if labels.shape != confidence.shape or labels.ndim != 2:
            raise ValueError(f"semantic evidence shape mismatch: {index}")
        expected_shape = (
            int(proxy_coordinate_contract["proxy_height"]),
            int(proxy_coordinate_contract["proxy_width"]),
        )
        if labels.shape != expected_shape:
            raise ValueError(f"semantic evidence does not use V01 proxy dimensions: {index}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"semantic labels are not integer: {index}")
        if labels.size == 0 or int(labels.min()) < 0 or int(labels.max()) > 28:
            raise ValueError(f"semantic palette is invalid: {index}")
        if not np.all(np.isfinite(confidence)) or np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError(f"semantic confidence is invalid: {index}")
        upper = np.isin(labels, upper_ids) & (confidence >= 0.5)
        upper_count = int(np.sum(upper))
        if upper_count < 64:
            raise ValueError(f"upper-garment semantic support is insufficient: {index}")
        verified.append(
            {
                "controlled_record_index": index,
                "source_frame_key": anchor["source_frame_key"],
                "semantic_path": str(path),
                "semantic_sha256": record["semantic_sha256"],
                "upper_garment_confident_pixel_count": upper_count,
            }
        )
    return verified, checkpoint_sha


def prepare_controlled_chart_requests(
    *,
    v01_master_path: Path,
    v01_qualification_path: Path,
    semantic_manifest_path: Path,
    tracker_source_audit_path: Path,
    output_path: Path,
) -> Path:
    """Bind 72 immutable Q05 anchor requests without running or trusting a tracker."""
    paths = [
        v01_master_path,
        v01_qualification_path,
        semantic_manifest_path,
        tracker_source_audit_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError(f"controlled chart request output is immutable: {output_path}")
    master = _require_report(v01_master_path)
    qualification = _require_report(v01_qualification_path)
    if master.get("schema_version") != "frayid_v3_controlled_capture_evidence_master.v1":
        raise ValueError("unexpected V01 evidence-master schema")
    if master.get("experiment_id") != "postv3_v01_controlled_recapture_evidence_master_r01":
        raise ValueError("unexpected V01 experiment")
    master_sha = sha256_file(v01_master_path)
    if qualification.get("evidence_master_sha256") != master_sha:
        raise ValueError("V01 qualification does not bind this evidence master")
    if (
        qualification.get("checks", {}).get("exactly_one_material_chart_anchor_per_hold")
        is not True
    ):
        raise ValueError("V01 did not qualify exactly one material anchor per hold")
    source_audit = master.get("source_audit")
    if not isinstance(source_audit, dict) or source_audit.get("status") != "pass":
        raise ValueError("V01 source audit did not pass")
    calibration = source_audit.get("training_camera_calibration")
    if not isinstance(calibration, dict) or calibration.get("role") != "measured_training_camera":
        raise ValueError("V01 has no measured training-camera calibration")
    evidence_scope = str(master.get("evidence_scope", "train_real"))
    if evidence_scope not in {"public_synthetic", "train_real"}:
        raise ValueError("unsupported controlled evidence scope")
    anchors = _anchor_records(master)
    proxy_coordinate_contract = master.get("proxy_coordinate_contract")
    if not isinstance(proxy_coordinate_contract, dict):
        raise ValueError("V01 master has no proxy coordinate contract")
    required_coordinate_fields = {
        "source_width",
        "source_height",
        "proxy_width",
        "proxy_height",
        "source_to_proxy_homography",
        "proxy_to_source_homography",
    }
    if not required_coordinate_fields.issubset(proxy_coordinate_contract):
        raise ValueError("V01 proxy coordinate contract is incomplete")
    semantic_manifest = read_json(semantic_manifest_path)
    semantic, semantic_checkpoint_sha = _semantic_records(
        semantic_manifest,
        anchors,
        evidence_scope=evidence_scope,
        v01_sha256=master_sha,
        proxy_coordinate_contract=proxy_coordinate_contract,
    )
    semantic_by_index = {int(item["controlled_record_index"]): item for item in semantic}
    tracker_audit = _require_report(tracker_source_audit_path)
    _validate_tracker_audit(tracker_audit, evidence_scope=evidence_scope)
    requests = []
    for anchor in anchors:
        index = int(anchor["controlled_record_index"])
        requests.append(
            {
                **anchor,
                "semantic_evidence": semantic_by_index[index],
                "tracker_roles": {source: "proposal_only" for source in SOURCES},
                "upper_garment_seed_policy": {
                    "confidence_minimum": 0.5,
                    "boundary_erosion_pixels": 8,
                    "minimum_seed_spacing_pixels": 16,
                    "maximum_seed_count": 64,
                    "truth_write_access": False,
                },
            }
        )
    payload = {
        "schema_version": "frayid_v3_controlled_chart_request_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_scope": evidence_scope,
        "capture_mode": master.get("capture_mode", "dual_camera_metric_evaluation"),
        "metric_accuracy_claim_allowed": bool(master.get("metric_accuracy_claim_allowed", False)),
        "scientific_claim_ceiling": master.get(
            "scientific_claim_ceiling",
            "metric_accuracy_only_after_independent_evaluator_gates",
        ),
        "status": "pass",
        "promotion_eligible": False,
        "request_count": len(requests),
        "directions": sorted({str(item["direction"]) for item in requests}),
        "v01_evidence_master_path": str(v01_master_path),
        "v01_evidence_master_sha256": master_sha,
        "v01_qualification_sha256": sha256_file(v01_qualification_path),
        "training_camera_calibration": calibration,
        "proxy_coordinate_contract": proxy_coordinate_contract,
        "semantic_manifest_sha256": sha256_file(semantic_manifest_path),
        "semantic_checkpoint_sha256": semantic_checkpoint_sha,
        "semantic_source_revision": semantic_manifest["source_revision"],
        "semantic_exact_same_device_replay": True,
        "tracker_source_audit_sha256": sha256_file(tracker_source_audit_path),
        "tracker_source_audit_path": str(tracker_source_audit_path),
        "tracker_sources": list(SOURCES),
        "requests": requests,
        "transition_frames_role": "proposal_context_only_never_measured_fit_truth",
        "evaluator_camera_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "blockers": [],
    }
    return _write_json_exclusive(output_path, payload)


__all__ = ["prepare_controlled_chart_requests"]
