from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability

V01_EXPERIMENT_ID = "postv3_v01_controlled_recapture_evidence_master_r01"
Q05_EXPERIMENT_ID = "postv3_q05_controlled_material_chart_graph_r01"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _anchor_keys(master: dict[str, Any]) -> dict[int, str]:
    clips = master.get("training_clips")
    if not isinstance(clips, list) or len(clips) != 2:
        raise ValueError("controlled semantic binding requires two V01 clips")
    anchors = [
        anchor
        for clip in clips
        if isinstance(clip, dict)
        for anchor in clip.get("material_chart_anchor_records", [])
        if isinstance(anchor, dict)
    ]
    if len(anchors) != 72:
        raise ValueError("controlled semantic binding requires exactly 72 V01 anchors")
    by_index = {
        int(item["controlled_record_index"]): str(item["source_frame_key"]) for item in anchors
    }
    if set(by_index) != set(range(72)) or len(set(by_index.values())) != 72:
        raise ValueError("controlled semantic V01 anchor identity is invalid")
    return by_index


def _validate_extraction(
    report: dict[str, Any],
    *,
    label: str,
    v01_sha256: str,
    evidence_scope: str,
    anchor_keys: dict[int, str],
    proxy_shape: tuple[int, int],
) -> tuple[dict[int, dict[str, Any]], tuple[str, str, str]]:
    expected = {
        "schema_version": "frayid_v3_controlled_sapiens2_extraction.v1",
        "experiment_id": Q05_EXPERIMENT_ID,
        "status": "complete",
        "evidence_scope": evidence_scope,
        "v01_evidence_master_sha256": v01_sha256,
        "palette": "sapiens2_dome29",
        "evaluator_camera_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "exact_same_device_replay": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"controlled semantic {label} extraction mismatch: {key}")
    source_revision = str(report.get("source_revision", ""))
    checkpoint_sha256 = str(report.get("checkpoint_sha256", ""))
    device = str(report.get("device", ""))
    if not source_revision or not device:
        raise ValueError(f"controlled semantic {label} extraction lacks provenance")
    if not _SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise ValueError(f"controlled semantic {label} checkpoint hash is invalid")
    if evidence_scope == "train_real" and checkpoint_sha256 == "0" * 64:
        raise ValueError("real controlled semantics cannot use placeholder weights")
    if int(report.get("controlled_records_processed", -1)) != 72:
        raise ValueError(f"controlled semantic {label} extraction did not process 72 records")
    if int(report.get("training_records_read", -1)) != (
        72 if evidence_scope == "train_real" else 0
    ):
        raise ValueError(f"controlled semantic {label} training-read count is invalid")
    frames = report.get("frames")
    if not isinstance(frames, list) or len(frames) != 72:
        raise ValueError(f"controlled semantic {label} extraction must contain 72 frames")
    by_index = {
        int(frame["controlled_record_index"]): frame for frame in frames if isinstance(frame, dict)
    }
    if set(by_index) != set(range(72)) or len(by_index) != len(frames):
        raise ValueError(f"controlled semantic {label} frame indices are invalid")
    semantic_paths: set[Path] = set()
    for index in range(72):
        frame = by_index[index]
        if frame.get("source_frame_key") != anchor_keys[index]:
            raise ValueError(f"controlled semantic {label} source key mismatch: {index}")
        path = Path(str(frame.get("semantic_path", "")))
        reject_sealed_capability([path])
        if path in semantic_paths:
            raise ValueError(f"controlled semantic {label} reused a frame artifact")
        semantic_paths.add(path)
        if not path.is_file() or sha256_file(path) != frame.get("semantic_sha256"):
            raise ValueError(f"controlled semantic {label} frame hash mismatch: {index}")
        with np.load(path, allow_pickle=False) as archive:
            labels = np.asarray(archive["labels"])
            confidence = np.asarray(archive["confidence"], dtype=np.float32)
        if labels.shape != proxy_shape or confidence.shape != proxy_shape:
            raise ValueError(f"controlled semantic {label} frame shape mismatch: {index}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"controlled semantic {label} labels are not integer: {index}")
        if labels.size == 0 or int(labels.min()) < 0 or int(labels.max()) > 28:
            raise ValueError(f"controlled semantic {label} palette is invalid: {index}")
        if not np.all(np.isfinite(confidence)) or np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError(f"controlled semantic {label} confidence is invalid: {index}")
    return by_index, (source_revision, checkpoint_sha256, device)


def bind_controlled_semantic_replays(
    *,
    v01_master_path: Path,
    primary_extraction_path: Path,
    replay_extraction_path: Path,
    output_path: Path,
) -> Path:
    """Bind two independent same-device Sapiens2 passes into Q05 semantics."""
    paths = [v01_master_path, primary_extraction_path, replay_extraction_path, output_path]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError(f"controlled semantic manifest is immutable: {output_path}")
    if primary_extraction_path == replay_extraction_path:
        raise ValueError("controlled semantic replay must be a distinct execution report")
    master = read_json(v01_master_path)
    if (
        master.get("schema_version") != "frayid_v3_controlled_capture_evidence_master.v1"
        or master.get("experiment_id") != V01_EXPERIMENT_ID
        or master.get("status") != "pass"
    ):
        raise ValueError("controlled semantic binding requires a passing V01 master")
    evidence_scope = str(master.get("evidence_scope"))
    if evidence_scope not in {"public_synthetic", "train_real"}:
        raise ValueError("unsupported controlled semantic evidence scope")
    coordinate_contract = master.get("proxy_coordinate_contract")
    if not isinstance(coordinate_contract, dict):
        raise ValueError("V01 master has no proxy coordinate contract")
    proxy_shape = (
        int(coordinate_contract.get("proxy_height", 0)),
        int(coordinate_contract.get("proxy_width", 0)),
    )
    if min(proxy_shape) <= 0:
        raise ValueError("V01 proxy coordinate dimensions are invalid")
    anchor_keys = _anchor_keys(master)
    v01_sha = sha256_file(v01_master_path)
    primary = read_json(primary_extraction_path)
    replay = read_json(replay_extraction_path)
    primary_frames, primary_provenance = _validate_extraction(
        primary,
        label="primary",
        v01_sha256=v01_sha,
        evidence_scope=evidence_scope,
        anchor_keys=anchor_keys,
        proxy_shape=proxy_shape,
    )
    replay_frames, replay_provenance = _validate_extraction(
        replay,
        label="replay",
        v01_sha256=v01_sha,
        evidence_scope=evidence_scope,
        anchor_keys=anchor_keys,
        proxy_shape=proxy_shape,
    )
    if primary_provenance != replay_provenance:
        raise ValueError("controlled semantic replay changed source, checkpoint, or device")
    frames: list[dict[str, Any]] = []
    for index in range(72):
        primary_frame = primary_frames[index]
        replay_frame = replay_frames[index]
        primary_path = Path(str(primary_frame["semantic_path"]))
        replay_path = Path(str(replay_frame["semantic_path"]))
        if primary_path == replay_path:
            raise ValueError("controlled semantic replay must use distinct frame artifacts")
        with np.load(primary_path, allow_pickle=False) as primary_archive:
            primary_labels = np.asarray(primary_archive["labels"])
            primary_confidence = np.asarray(primary_archive["confidence"])
        with np.load(replay_path, allow_pickle=False) as replay_archive:
            replay_labels = np.asarray(replay_archive["labels"])
            replay_confidence = np.asarray(replay_archive["confidence"])
        if not np.array_equal(primary_labels, replay_labels) or not np.array_equal(
            primary_confidence, replay_confidence
        ):
            raise ValueError(f"controlled semantic same-device replay differs: {index}")
        frames.append(
            {
                "controlled_record_index": index,
                "source_frame_key": anchor_keys[index],
                "semantic_path": str(primary_path),
                "semantic_sha256": primary_frame["semantic_sha256"],
                "replay_semantic_path": str(replay_path),
                "replay_semantic_sha256": replay_frame["semantic_sha256"],
                "exact_array_replay": True,
            }
        )
    source_revision, checkpoint_sha256, device = primary_provenance
    payload = {
        "schema_version": "frayid_v3_controlled_semantic_evidence.v1",
        "experiment_id": Q05_EXPERIMENT_ID,
        "evidence_scope": evidence_scope,
        "status": "pass",
        "promotion_eligible": False,
        "v01_evidence_master_sha256": v01_sha,
        "palette": "sapiens2_dome29",
        "proxy_coordinate_contract": coordinate_contract,
        "checkpoint_sha256": checkpoint_sha256,
        "source_revision": source_revision,
        "device": device,
        "primary_extraction_sha256": sha256_file(primary_extraction_path),
        "replay_extraction_sha256": sha256_file(replay_extraction_path),
        "exact_same_device_replay": True,
        "frames": frames,
        "evaluator_camera_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "blockers": [],
    }
    return _write_json_exclusive(output_path, payload)


__all__ = ["bind_controlled_semantic_replays"]
