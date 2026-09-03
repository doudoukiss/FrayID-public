from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.evidence_master import render_analysis_proxy
from frayid.v2.video_forensics import iter_sequential_rgb_frames
from frayid.v3.controlled_chart_execution import (
    EXPERIMENT_ID,
    _coordinate_contract,
    _require_q05_request,
    _seed_inventory,
)

SourceName = Literal["lk", "tapir", "cotracker3"]
WINDOW_LENGTH = 4


def track_controlled_lk_window(
    rgb_frames: np.ndarray,
    query_xy: np.ndarray,
    query_frame_indices: np.ndarray,
    *,
    maximum_forward_backward_error_pixels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track mixed-time queries bidirectionally with an audited LK visibility test."""
    frames = np.asarray(rgb_frames)
    queries = np.asarray(query_xy, dtype=np.float32)
    query_times = np.asarray(query_frame_indices, dtype=np.int64)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 2:
        raise ValueError("controlled LK expects RGB frames with shape [time,height,width,3]")
    if queries.ndim != 2 or queries.shape[1] != 2 or len(queries) == 0:
        raise ValueError("controlled LK queries must have shape [queries,2]")
    if query_times.shape != (len(queries),) or np.any(
        (query_times < 0) | (query_times >= len(frames))
    ):
        raise ValueError("controlled LK query times are invalid")
    if maximum_forward_backward_error_pixels <= 0.0:
        raise ValueError("controlled LK forward/backward threshold must be positive")
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames])
    tracks = np.repeat(queries[None], len(frames), axis=0)
    visibility = np.zeros((len(frames), len(queries)), dtype=bool)
    forward_backward = np.full((len(frames), len(queries)), np.inf, dtype=np.float32)
    lk_parameters = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50,
            1.0e-4,
        ),
    }
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260903)

    def propagate(
        indices: np.ndarray,
        source_time: int,
        target_times: range,
    ) -> None:
        points = queries[indices].reshape(-1, 1, 2).copy()
        active = np.ones(len(indices), dtype=bool)
        previous_time = source_time
        for target_time in target_times:
            predicted, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                gray[previous_time], gray[target_time], points, None, **lk_parameters
            )
            if predicted is None or status is None:
                active[:] = False
                break
            restored, reverse_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                gray[target_time], gray[previous_time], predicted, None, **lk_parameters
            )
            if restored is None or reverse_status is None:
                active[:] = False
                break
            error = np.linalg.norm(restored.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
            active &= status.reshape(-1).astype(bool)
            active &= reverse_status.reshape(-1).astype(bool)
            active &= np.isfinite(error)
            active &= error <= maximum_forward_backward_error_pixels
            current = predicted.reshape(-1, 2)
            tracks[target_time, indices] = current
            visibility[target_time, indices] = active
            forward_backward[target_time, indices] = error
            points = predicted
            previous_time = target_time

    for query_time in sorted(set(query_times.tolist())):
        indices = np.flatnonzero(query_times == query_time)
        tracks[query_time, indices] = queries[indices]
        visibility[query_time, indices] = True
        forward_backward[query_time, indices] = 0.0
        propagate(indices, query_time, range(query_time + 1, len(frames)))
        propagate(indices, query_time, range(query_time - 1, -1, -1))
    return tracks.astype(np.float64), visibility, forward_backward.astype(np.float64)


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracker_source_record(request: dict[str, Any], source: SourceName) -> dict[str, Any]:
    audit_path = Path(str(request.get("tracker_source_audit_path", "")))
    reject_sealed_capability([audit_path])
    if not audit_path.is_file() or sha256_file(audit_path) != request.get(
        "tracker_source_audit_sha256"
    ):
        raise ValueError("Q05 tracker source audit hash mismatch")
    audit = read_json(audit_path)
    if audit.get("status") != "pass" or audit.get("real_execution_ready") is not True:
        raise ValueError("Q05 tracker source audit is not real-execution ready")
    records = {
        str(item.get("source")): item for item in audit.get("sources", []) if isinstance(item, dict)
    }
    record = records.get(source)
    if record is None:
        raise ValueError(f"Q05 source audit has no {source} record")
    if (
        record.get("license_ready_for_real_use") is not True
        or record.get("material_truth_write_access") is not False
    ):
        raise ValueError(f"Q05 tracker source is not role/license qualified: {source}")
    return record


def _calibration_record(path: Path, source: SourceName) -> tuple[dict[str, Any], float]:
    reject_sealed_capability([path])
    report = read_json(path)
    if source == "lk" and isinstance(report.get("lk_calibration"), dict):
        record = report["lk_calibration"]
        if report.get("status") != "pass":
            raise ValueError("Q05 LK calibration bundle did not pass")
    else:
        record = report
    if record.get("source") != source or record.get("status") != "pass":
        raise ValueError(f"Q05 public tracker calibration did not pass: {source}")
    if (
        int(record.get("project_evidence_reads", -1)) != 0
        or int(record.get("sealed_test_accesses", -1)) != 0
    ):
        raise ValueError("Q05 tracker calibration used project or sealed evidence")
    p95 = float(record.get("p95_visible_error_pixels", float("inf")))
    if not np.isfinite(p95) or p95 > 5.0:
        raise ValueError(f"Q05 tracker calibration p95 is invalid: {source}")
    if source != "lk" and record.get("exact_same_device_replay") is not True:
        raise ValueError(f"Q05 learned tracker calibration did not replay: {source}")
    return record, max(p95, 0.5)


def _decode_proxy_anchors(request: dict[str, Any]) -> np.ndarray:
    master_path = Path(str(request.get("v01_evidence_master_path", "")))
    reject_sealed_capability([master_path])
    if not master_path.is_file() or sha256_file(master_path) != request.get(
        "v01_evidence_master_sha256"
    ):
        raise ValueError("Q05 tracker runner cannot restore its V01 master")
    master = read_json(master_path)
    if master.get("status") != "pass" or master.get("evidence_scope") != "train_real":
        raise ValueError("Q05 tracker runner requires a passing real V01 master")
    _, source_width, source_height, proxy_width, proxy_height = _coordinate_contract(request)
    selected: dict[int, np.ndarray] = {}
    requests_by_direction = {
        direction: {
            int(record["source_decode_index"]): record
            for record in request["requests"]
            if record["direction"] == direction
        }
        for direction in ("clockwise", "counter_clockwise")
    }
    clips = {
        str(clip["direction"]): clip
        for clip in master.get("training_clips", [])
        if isinstance(clip, dict)
    }
    for direction in ("clockwise", "counter_clockwise"):
        clip = clips.get(direction)
        if clip is None or not isinstance(clip.get("source"), dict):
            raise ValueError(f"V01 master has no {direction} native source")
        source_record = clip["source"]
        source_path = Path(str(source_record.get("path", "")))
        reject_sealed_capability([source_path])
        if not source_path.is_file() or sha256_file(source_path) != source_record.get("sha256"):
            raise ValueError(f"V01 {direction} native source hash mismatch")
        for decode_index, rgb in enumerate(
            iter_sequential_rgb_frames(
                source_path,
                width=source_width,
                height=source_height,
            )
        ):
            anchor = requests_by_direction[direction].get(decode_index)
            if anchor is None:
                continue
            digest = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
            if digest != anchor["decoded_rgb_sha256"]:
                raise ValueError(
                    f"Q05 tracker decoded RGB hash mismatch: {direction}:{decode_index}"
                )
            proxy = render_analysis_proxy(rgb, request["proxy_coordinate_contract"])
            if proxy.shape != (proxy_height, proxy_width, 3):
                raise RuntimeError("Q05 tracker proxy coordinate contract changed")
            selected[int(anchor["controlled_record_index"])] = proxy
    if set(selected) != set(range(72)):
        raise ValueError("Q05 tracker runner did not recover all 72 midpoint frames")
    return np.stack([selected[index] for index in range(72)])


def _window_groups(
    seed_inventory: dict[str, dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for seed in seed_inventory.values():
        record_index = int(seed["controlled_record_index"])
        direction_slot = 0 if record_index < 36 else 1
        local_index = record_index % 36
        window_start = min(local_index, 36 - WINDOW_LENGTH)
        groups[(direction_slot, window_start)].append(seed)
    for seeds in groups.values():
        seeds.sort(key=lambda item: str(item["seed_id"]))
    return dict(sorted(groups.items()))


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for Q05 but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested for Q05 but is unavailable")
    return device


def _load_tapir(
    source_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("tapnet.torch.tapir_model")
    module_file = getattr(module, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(source_root.resolve()):
        raise ValueError("TAPIR import did not resolve to the audited source")
    model = module.TAPIR(pyramid_level=0, extra_convs=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _load_cotracker(
    source_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("cotracker.predictor")
    module_file = getattr(module, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(source_root.resolve()):
        raise ValueError("CoTracker3 import did not resolve to the audited source")
    predictor = module.CoTrackerPredictor(
        checkpoint=None,
        offline=True,
        v2=False,
        window_len=60,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    predictor.model.load_state_dict(state)
    return predictor.to(device).eval()


def _infer_tapir_window(
    model: Any,
    frames: np.ndarray,
    query_xy: np.ndarray,
    query_times: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_size = 256
    resized = np.stack(
        [
            cv2.resize(frame, (model_size, model_size), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
    )
    scale_x = model_size / frames.shape[2]
    scale_y = model_size / frames.shape[1]
    queries = np.column_stack(
        (query_times, query_xy[:, 1] * scale_y, query_xy[:, 0] * scale_x)
    ).astype(np.float32)
    video = torch.from_numpy(resized).to(device=device, dtype=torch.float32)
    video = video / 255.0 * 2.0 - 1.0
    query = torch.from_numpy(queries).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(video[None], query[None])
        score = (1.0 - torch.sigmoid(output["occlusion"][0])) * (
            1.0 - torch.sigmoid(output["expected_dist"][0])
        )
    tracks = np.transpose(output["tracks"][0].detach().cpu().numpy(), (1, 0, 2)).copy()
    tracks[..., 0] /= scale_x
    tracks[..., 1] /= scale_y
    visibility = np.transpose((score > 0.5).detach().cpu().numpy(), (1, 0)).copy()
    uncertainty = np.broadcast_to(np.asarray(1.0, dtype=np.float64), visibility.shape).copy()
    return tracks.astype(np.float64), visibility.astype(bool), uncertainty


def _infer_cotracker_window(
    predictor: Any,
    frames: np.ndarray,
    query_xy: np.ndarray,
    query_times: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_height = 384
    input_width = round(frames.shape[2] * input_height / frames.shape[1])
    resized = np.stack(
        [
            cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
    )
    scale_x = input_width / frames.shape[2]
    scale_y = input_height / frames.shape[1]
    queries = np.column_stack(
        (query_times, query_xy[:, 0] * scale_x, query_xy[:, 1] * scale_y)
    ).astype(np.float32)
    video = (
        torch.from_numpy(resized).permute(0, 3, 1, 2)[None].to(device=device, dtype=torch.float32)
    )
    query = torch.from_numpy(queries)[None].to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        tracks_tensor, visibility_tensor = predictor(video, queries=query)
    tracks = tracks_tensor[0].detach().cpu().numpy().copy()
    tracks[..., 0] /= scale_x
    tracks[..., 1] /= scale_y
    visibility = visibility_tensor[0].detach().cpu().numpy().astype(bool).copy()
    uncertainty = np.broadcast_to(np.asarray(1.0, dtype=np.float64), visibility.shape).copy()
    return tracks.astype(np.float64), visibility, uncertainty


def _proxy_to_source(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(len(flat))))
    projected = (homography @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1.0e-12):
        raise ValueError("Q05 tracker output encountered a singular coordinate map")
    result = projected[:, :2] / projected[:, 2, None]
    return result.reshape(points.shape)


def _infer_tracker_window(
    *,
    source: SourceName,
    model: Any,
    frames: np.ndarray,
    query_xy: np.ndarray,
    query_times: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source == "lk":
        return track_controlled_lk_window(
            frames,
            query_xy,
            query_times,
            maximum_forward_backward_error_pixels=5.0,
        )
    if source == "tapir":
        return _infer_tapir_window(model, frames, query_xy, query_times, device)
    return _infer_cotracker_window(model, frames, query_xy, query_times, device)


def _validate_tracker_arrays(
    *,
    source: SourceName,
    tracks: np.ndarray,
    visibility: np.ndarray,
    uncertainty: np.ndarray,
    frame_count: int,
    seed_count: int,
) -> None:
    if tracks.shape != (frame_count, seed_count, 2):
        raise ValueError(f"Q05 {source} emitted an invalid track array shape")
    if visibility.shape != (frame_count, seed_count):
        raise ValueError(f"Q05 {source} emitted an invalid visibility array shape")
    if uncertainty.shape != (frame_count, seed_count):
        raise ValueError(f"Q05 {source} emitted an invalid uncertainty array shape")
    if not np.all(np.isfinite(tracks)):
        raise ValueError(f"Q05 {source} emitted non-finite tracker coordinates")


def _serialize_seed_track(
    *,
    predicted: np.ndarray,
    visibility: np.ndarray,
    uncertainty: np.ndarray,
    seed_slot: int,
    seed: dict[str, Any],
    global_indices: list[int],
    requests_by_index: dict[int, dict[str, Any]],
    base_covariance: np.ndarray,
) -> dict[str, Any]:
    seed_id = str(seed["seed_id"])
    seed_record = int(seed["controlled_record_index"])
    observations: list[dict[str, Any]] = []
    for time_slot, record_index in enumerate(global_indices):
        request = requests_by_index[record_index]
        xy = predicted[time_slot, seed_slot].copy()
        visible = bool(visibility[time_slot, seed_slot])
        if record_index == seed_record:
            xy = np.asarray(seed["source_xy_pixels"], dtype=np.float64)
            visible = True
        uncertainty_value = float(uncertainty[time_slot, seed_slot])
        uncertainty_multiplier = (
            min(max(uncertainty_value, 1.0), 25.0) if np.isfinite(uncertainty_value) else 25.0
        )
        covariance = base_covariance * uncertainty_multiplier
        observations.append(
            {
                "controlled_record_index": record_index,
                "source_frame_index": request["source_decode_index"],
                "source_frame_key": request["source_frame_key"],
                "xy": xy.tolist(),
                "covariance": covariance.tolist(),
                "visible": visible,
                "evidence_role": "proposal_only",
            }
        )
    return {
        "seed_id": seed_id,
        "seed_controlled_record_index": seed_record,
        "chart_id": seed["chart_id"],
        "query_xy_source_pixels": seed["source_xy_pixels"],
        "observations": observations,
    }


def _payload_tracks(
    *,
    groups: dict[tuple[int, int], list[dict[str, Any]]],
    frames: np.ndarray,
    requests_by_index: dict[int, dict[str, Any]],
    source: SourceName,
    device: torch.device,
    model: Any,
    calibration_sigma: float,
    proxy_to_source: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_by_seed: dict[str, dict[str, Any]] = {}
    replay_by_seed: dict[str, dict[str, Any]] = {}
    covariance_transform = proxy_to_source[:2, :2]
    base_covariance = calibration_sigma**2 * (covariance_transform @ covariance_transform.T)
    base_covariance += np.eye(2) * 1.0e-6
    for (direction_slot, window_start), seeds in groups.items():
        offset = 36 * direction_slot
        global_indices = [offset + index for index in range(window_start, window_start + 4)]
        window = frames[np.asarray(global_indices)]
        query_proxy = np.asarray([item["semantic_xy_pixels"] for item in seeds], dtype=np.float64)
        query_times = np.asarray(
            [int(item["controlled_record_index"]) % 36 - window_start for item in seeds],
            dtype=np.int64,
        )

        inference_arguments = {
            "source": source,
            "model": model,
            "frames": window,
            "query_xy": query_proxy,
            "query_times": query_times,
            "device": device,
        }
        primary_tracks, primary_visibility, primary_uncertainty = _infer_tracker_window(
            **inference_arguments
        )
        replay_tracks, replay_visibility, replay_uncertainty = _infer_tracker_window(
            **inference_arguments
        )
        for tracks, visibility, uncertainty in (
            (primary_tracks, primary_visibility, primary_uncertainty),
            (replay_tracks, replay_visibility, replay_uncertainty),
        ):
            _validate_tracker_arrays(
                source=source,
                tracks=tracks,
                visibility=visibility,
                uncertainty=uncertainty,
                frame_count=len(window),
                seed_count=len(seeds),
            )
        if not (
            np.array_equal(primary_tracks, replay_tracks)
            and np.array_equal(primary_visibility, replay_visibility)
            and np.array_equal(primary_uncertainty, replay_uncertainty)
        ):
            raise RuntimeError(
                f"Q05 {source} same-device replay differs at window {direction_slot}:{window_start}"
            )
        primary_source = _proxy_to_source(primary_tracks, proxy_to_source)
        replay_source = _proxy_to_source(replay_tracks, proxy_to_source)
        for seed_slot, seed in enumerate(seeds):
            seed_id = str(seed["seed_id"])
            serialization_arguments = {
                "seed_slot": seed_slot,
                "seed": seed,
                "global_indices": global_indices,
                "requests_by_index": requests_by_index,
                "base_covariance": base_covariance,
            }
            primary_by_seed[seed_id] = _serialize_seed_track(
                predicted=primary_source,
                visibility=primary_visibility,
                uncertainty=primary_uncertainty,
                **serialization_arguments,
            )
            replay_by_seed[seed_id] = _serialize_seed_track(
                predicted=replay_source,
                visibility=replay_visibility,
                uncertainty=replay_uncertainty,
                **serialization_arguments,
            )
    return (
        [primary_by_seed[key] for key in sorted(primary_by_seed)],
        [replay_by_seed[key] for key in sorted(replay_by_seed)],
    )


def run_controlled_tracker(
    *,
    request_manifest_path: Path,
    seed_manifest_path: Path,
    calibration_report_path: Path,
    source: SourceName,
    output_root: Path,
    source_root: Path | None = None,
    checkpoint_path: Path | None = None,
    device_name: str = "cpu",
) -> Path:
    """Run one audited tracker twice and atomically emit its proposal-only bundle."""
    paths = [
        request_manifest_path,
        seed_manifest_path,
        calibration_report_path,
        output_root,
        *([source_root] if source_root is not None else []),
        *([checkpoint_path] if checkpoint_path is not None else []),
    ]
    reject_sealed_capability(paths)
    if output_root.exists():
        raise FileExistsError(f"controlled tracker output is immutable: {output_root}")
    prior_partial = list(output_root.parent.glob(f".{output_root.name}.building-*"))
    if prior_partial:
        raise FileExistsError("a prior partial tracker run must be audited separately")
    request = _require_q05_request(request_manifest_path)
    if request.get("evidence_scope") != "train_real":
        raise ValueError("the controlled tracker runner is restricted to real V01 evidence")
    request_sha = sha256_file(request_manifest_path)
    seed_manifest = read_json(seed_manifest_path)
    if (
        seed_manifest.get("schema_version") != "frayid_v3_controlled_chart_seed_manifest.v1"
        or seed_manifest.get("request_manifest_sha256") != request_sha
        or seed_manifest.get("status") != "pass"
        or seed_manifest.get("evidence_scope") != "train_real"
        or seed_manifest.get("exact_same_input_replay") is not True
    ):
        raise ValueError("Q05 tracker runner requires a qualified seed manifest")
    seed_sha = sha256_file(seed_manifest_path)
    seed_inventory = _seed_inventory(seed_manifest)
    if len(seed_inventory) < 100:
        raise ValueError("Q05 tracker runner requires at least 100 registered seeds")
    source_record = _tracker_source_record(request, source)
    calibration, calibration_sigma = _calibration_record(calibration_report_path, source)
    expected_revision = str(source_record["source_revision"])
    expected_checkpoint = source_record.get("checkpoint_observed_sha256")
    device = _device(device_name)
    model: Any = None
    weights_executed = False
    if source == "lk":
        if source_root is not None or checkpoint_path is not None or device.type != "cpu":
            raise ValueError("controlled LK uses CPU OpenCV and no source/checkpoint arguments")
        runtime = f"opencv-{cv2.__version__}"
    else:
        if source_root is None or checkpoint_path is None:
            raise ValueError(f"controlled {source} requires source and checkpoint paths")
        if _git_revision(source_root) != expected_revision:
            raise ValueError(f"controlled {source} source revision mismatch")
        observed_checkpoint = sha256_file(checkpoint_path)
        if observed_checkpoint != expected_checkpoint:
            raise ValueError(f"controlled {source} checkpoint SHA-256 mismatch")
        if (
            calibration.get("source_revision") != expected_revision
            or calibration.get("checkpoint_sha256") != expected_checkpoint
        ):
            raise ValueError(f"controlled {source} calibration provenance mismatch")
        torch.manual_seed(20260903)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        model = (
            _load_tapir(source_root, checkpoint_path, device)
            if source == "tapir"
            else _load_cotracker(source_root, checkpoint_path, device)
        )
        weights_executed = True
        runtime = f"torch-{torch.__version__}"
    frames = _decode_proxy_anchors(request)
    groups = _window_groups(seed_inventory)
    proxy_to_source, _, _, _, _ = _coordinate_contract(request)
    requests_by_index = {int(item["controlled_record_index"]): item for item in request["requests"]}
    primary_tracks, replay_tracks = _payload_tracks(
        groups=groups,
        frames=frames,
        requests_by_index=requests_by_index,
        source=source,
        device=device,
        model=model,
        calibration_sigma=calibration_sigma,
        proxy_to_source=proxy_to_source,
    )
    if primary_tracks != replay_tracks:
        raise RuntimeError(f"controlled {source} serialized replay differs")
    payload_common = {
        "schema_version": "frayid_v3_controlled_tracker_proposals.v1",
        "experiment_id": EXPERIMENT_ID,
        "source": source,
        "role": "proposal_only",
        "request_manifest_sha256": request_sha,
        "seed_manifest_sha256": seed_sha,
        "transition_frames_role": "proposal_context_only_never_measured_fit_truth",
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    primary_stage = write_json(stage / "primary.json", {**payload_common, "tracks": primary_tracks})
    replay_stage = write_json(stage / "replay.json", {**payload_common, "tracks": replay_tracks})
    if sha256_file(primary_stage) != sha256_file(replay_stage):
        raise RuntimeError(f"controlled {source} replay files are not byte-identical")
    primary_final = output_root / "primary.json"
    replay_final = output_root / "replay.json"
    audit_path = Path(str(request["tracker_source_audit_path"]))
    bundle = {
        "schema_version": "frayid_v3_controlled_tracker_output_bundle.v1",
        "experiment_id": EXPERIMENT_ID,
        "source": source,
        "role": "proposal_only",
        "request_manifest_sha256": request_sha,
        "seed_manifest_sha256": seed_sha,
        "tracker_source_audit_sha256": request["tracker_source_audit_sha256"],
        "calibration_report_sha256": sha256_file(calibration_report_path),
        "source_revision": expected_revision,
        "license": source_record["license"],
        "checkpoint_sha256": expected_checkpoint,
        "weights_executed": weights_executed,
        "runtime": runtime,
        "device": str(device),
        "primary_payload": {
            "path": str(primary_final),
            "sha256": sha256_file(primary_stage),
        },
        "replay_payload": {
            "path": str(replay_final),
            "sha256": sha256_file(replay_stage),
        },
        "exact_same_device_replay": True,
        "material_truth_write_access": False,
        "query_identity_is_registered_input_not_tracker_truth": True,
        "window_length_registered_holds": WINDOW_LENGTH,
        "window_span_degrees": 30,
        "model_input_record_count": 72,
        "transition_frame_model_inputs": 0,
        "transition_frames_role": "proposal_context_only_never_measured_fit_truth",
        "controlled_records_processed": 72,
        "training_records_read": 72,
        "evaluator_camera_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "source_audit_path": str(audit_path),
    }
    write_json(stage / "bundle.json", bundle)
    os.rename(stage, output_root)
    return output_root / "bundle.json"


__all__ = ["run_controlled_tracker", "track_controlled_lk_window"]
