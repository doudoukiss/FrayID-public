from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from frayid.io import sha256_file
from frayid.v3.material_charts import build_material_chart_graph, public_chart_fixture


def _synthetic_cloth_sequence() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Render public cloth-like texture with occlusion, reappearance, and repetition."""
    rng = np.random.default_rng(404)
    size = 128
    texture: np.ndarray = np.asarray(rng.integers(0, 256, (size, size), dtype=np.uint8))
    texture = cv2.GaussianBlur(texture, (3, 3), 0.8)
    yy, xx = np.indices((size, size))
    checker = (((xx // 8 + yy // 8) % 2) * 150 + 50).astype(np.uint8)
    texture[:, : size // 2] = (
        0.45 * texture[:, : size // 2] + 0.55 * checker[:, : size // 2]
    ).astype(np.uint8)
    garment = np.zeros((size, size), dtype=np.uint8)
    garment[16:112, 16:112] = 255
    texture = cv2.bitwise_and(texture, garment)

    query_x, query_y = np.meshgrid(np.linspace(28, 92, 9), np.linspace(28, 92, 9))
    queries = np.column_stack([query_x.reshape(-1), query_y.reshape(-1)]).astype(np.float32)
    frame_count = 12
    truth = np.zeros((frame_count, len(queries), 2), dtype=np.float32)
    visibility = np.ones((frame_count, len(queries)), dtype=bool)
    frames: list[np.ndarray] = []
    for frame_index in range(frame_count):
        dx = 1.35 * frame_index
        dy = 2.0 * np.sin(frame_index * 0.45)
        transform = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        frame = cv2.warpAffine(
            texture,
            transform,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        truth[frame_index] = queries + np.asarray([dx, dy], dtype=np.float32)
        if 4 <= frame_index <= 6:
            left, right, top, bottom = 48, 86, 24, 106
            frame[top:bottom, left:right] = 15
            points = truth[frame_index]
            visibility[frame_index] &= ~(
                (points[:, 0] >= left)
                & (points[:, 0] < right)
                & (points[:, 1] >= top)
                & (points[:, 1] < bottom)
            )
        points = truth[frame_index]
        visibility[frame_index] &= (
            (points[:, 0] >= 2)
            & (points[:, 0] < size - 2)
            & (points[:, 1] >= 2)
            & (points[:, 1] < size - 2)
        )
        frames.append(frame)
    return frames, truth, visibility


def calibrate_lk_public_cloth() -> dict[str, Any]:
    frames, truth, visibility = _synthetic_cloth_sequence()
    point_count = truth.shape[1]
    estimates = truth[0].copy()
    estimate_valid = visibility[0].copy()
    accepted_errors: list[float] = []
    forward_backward: list[float] = []
    drift_by_step: list[float] = []
    invisible_proposals = 0
    invisible_count = 0
    reappearance_events = 0
    interval_splits = 0
    for frame_index in range(1, len(frames)):
        previous_truth_visible = visibility[frame_index - 1]
        current_truth_visible = visibility[frame_index]
        active = np.flatnonzero(estimate_valid & previous_truth_visible)
        next_estimates = estimates.copy()
        next_valid = np.zeros(point_count, dtype=bool)
        if len(active):
            previous_points = estimates[active].reshape(-1, 1, 2).astype(np.float32)
            predicted, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                frames[frame_index - 1],
                frames[frame_index],
                previous_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if predicted is None or status is None:
                raise RuntimeError("OpenCV LK returned no prediction")
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                frames[frame_index],
                frames[frame_index - 1],
                predicted,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if backward is None or backward_status is None:
                raise RuntimeError("OpenCV backward LK returned no prediction")
            fb = np.linalg.norm(backward[:, 0] - previous_points[:, 0], axis=1)
            valid = (status[:, 0] == 1) & (backward_status[:, 0] == 1) & (fb <= 1.5)
            next_estimates[active] = predicted[:, 0]
            next_valid[active] = valid
            forward_backward.extend(float(value) for value in fb[valid])

        visible_active = current_truth_visible & next_valid
        errors = np.linalg.norm(
            next_estimates[visible_active] - truth[frame_index, visible_active], axis=1
        )
        accepted_errors.extend(float(value) for value in errors)
        if errors.size:
            drift_by_step.append(float(np.median(errors)))
        invisible = ~current_truth_visible
        invisible_count += int(np.sum(invisible))
        invisible_proposals += int(np.sum(invisible & next_valid))
        next_valid &= current_truth_visible
        reappeared = ~previous_truth_visible & current_truth_visible
        if np.any(reappeared):
            # Q04/Q05 do not claim identity through an occlusion. Public truth
            # initializes a new interval query, but that seed is not counted as
            # a propagated LK observation.
            next_estimates[reappeared] = truth[frame_index, reappeared]
            next_valid[reappeared] = True
            reappearance_events += int(np.sum(reappeared))
            interval_splits += int(np.sum(reappeared))
        estimates = next_estimates
        estimate_valid = next_valid

    errors_array = np.asarray(accepted_errors)
    fb_array = np.asarray(forward_backward)
    if errors_array.size == 0 or fb_array.size == 0:
        raise RuntimeError("LK calibration produced no accepted public observations")
    median_error = float(np.median(errors_array))
    p95_error = float(np.percentile(errors_array, 95))
    corruption_clean = build_material_chart_graph(public_chart_fixture())
    corruption_treatment = build_material_chart_graph(public_chart_fixture(corrupt=True))
    clean_capacity = sum(track.accepted for track in corruption_clean.tracks)
    corrupt_capacity = sum(track.accepted for track in corruption_treatment.tracks)
    capacity_regression = max(clean_capacity - corrupt_capacity, 0)
    blockers: list[str] = []
    if median_error > 2.0 or p95_error > 5.0:
        blockers.append("lk_visible_track_error_gate")
    if capacity_regression != 0:
        blockers.append("corrupted_proposal_capacity_regression")
    return {
        "schema_version": "frayid_v3_q04_public_tracker_calibration.v1",
        "source": "lk",
        "status": "pass" if not blockers else "fail",
        "runtime": f"opencv-{cv2.__version__}",
        "conditions": {
            "occlusion": True,
            "reappearance": True,
            "repeated_texture": True,
            "drift": True,
            "deliberately_corrupted_proposals": True,
        },
        "query_count": point_count,
        "frame_count": len(frames),
        "accepted_observation_count": len(accepted_errors),
        "median_visible_error_pixels": median_error,
        "p95_visible_error_pixels": p95_error,
        "median_forward_backward_error_pixels": float(np.median(fb_array)),
        "p95_forward_backward_error_pixels": float(np.percentile(fb_array, 95)),
        "maximum_step_median_drift_pixels": max(drift_by_step, default=0.0),
        "occluded_proposal_fraction_before_truth_filter": (
            invisible_proposals / max(invisible_count, 1)
        ),
        "reappearance_events": reappearance_events,
        "visibility_interval_splits": interval_splits,
        "accepted_track_capacity_clean": clean_capacity,
        "accepted_track_capacity_with_corruption": corrupt_capacity,
        "corrupted_proposal_capacity_regression": capacity_regression,
        "truth_used_only_for_new_interval_query_initialization": True,
        "truth_substitution_in_propagated_observations": False,
        "project_evidence_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "blockers": blockers,
    }


def calibrate_tapir_public_cloth(
    *,
    source_root: Path,
    checkpoint_path: Path,
    expected_source_revision: str,
    expected_checkpoint_sha256: str,
    device_name: str = "cpu",
    verify_replay: bool = True,
) -> dict[str, Any]:
    """Execute the pinned official PyTorch TAPIR on the public cloth sequence."""
    if not source_root.is_dir() or not (source_root / "tapnet/torch/tapir_model.py").is_file():
        raise FileNotFoundError(f"pinned TAPIR source is unavailable: {source_root}")
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_source_revision:
        raise ValueError(f"TAPIR source revision mismatch: {revision}")
    observed_checkpoint_sha256 = sha256_file(checkpoint_path)
    if observed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("TAPIR checkpoint SHA-256 mismatch")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("tapnet.torch.tapir_model")
    # The official plain TAPIR panning checkpoint has no BootsTAPIR extra
    # convolutions and its mixer input (486) identifies pyramid_level=0.
    model = module.TAPIR(pyramid_level=0, extra_convs=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    frames, truth, visibility = _synthetic_cloth_sequence()
    model_size = 256
    rgb = np.stack(
        [
            cv2.cvtColor(cv2.resize(frame, (model_size, model_size)), cv2.COLOR_GRAY2RGB)
            for frame in frames
        ]
    )
    scale_x = model_size / frames[0].shape[1]
    scale_y = model_size / frames[0].shape[0]
    query_points = np.zeros((truth.shape[1], 3), dtype=np.float32)
    query_points[:, 1] = truth[0, :, 1] * scale_y
    query_points[:, 2] = truth[0, :, 0] * scale_x
    video_tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    video_tensor = video_tensor / 255.0 * 2.0 - 1.0
    query_tensor = torch.from_numpy(query_points).to(device=device, dtype=torch.float32)

    def infer() -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            outputs = model(video_tensor[None], query_tensor[None])
            tracks_tensor = outputs["tracks"][0]
            visibility_score = (1.0 - torch.sigmoid(outputs["occlusion"][0])) * (
                1.0 - torch.sigmoid(outputs["expected_dist"][0])
            )
            predicted_visibility_tensor = visibility_score > 0.5
        inferred_tracks = np.transpose(tracks_tensor.detach().cpu().numpy(), (1, 0, 2)).copy()
        inferred_visibility = np.transpose(
            predicted_visibility_tensor.detach().cpu().numpy(), (1, 0)
        ).copy()
        inferred_tracks[..., 0] /= scale_x
        inferred_tracks[..., 1] /= scale_y
        return inferred_tracks, inferred_visibility

    tracks, predicted_visibility = infer()
    exact_same_device_replay = True
    if verify_replay:
        replay_tracks, replay_visibility = infer()
        exact_same_device_replay = np.array_equal(tracks, replay_tracks) and np.array_equal(
            predicted_visibility, replay_visibility
        )
    error = np.linalg.norm(tracks - truth, axis=2)
    accepted = visibility & predicted_visibility
    accepted_error = error[accepted]
    if accepted_error.size == 0:
        raise RuntimeError("TAPIR accepted no visible public truth points")
    true_positive = int(np.sum(visibility & predicted_visibility))
    false_negative = int(np.sum(visibility & ~predicted_visibility))
    false_positive = int(np.sum(~visibility & predicted_visibility))
    true_negative = int(np.sum(~visibility & ~predicted_visibility))
    reappeared_points = (~visibility[6]) & visibility[7]
    reappearance_accepted = reappeared_points & predicted_visibility[7]
    reappearance_error = error[7, reappearance_accepted]
    median_error = float(np.median(accepted_error))
    p95_error = float(np.percentile(accepted_error, 95))
    blockers: list[str] = []
    if median_error > 2.0 or p95_error > 5.0:
        blockers.append("tapir_visible_track_error_gate")
    if reappearance_error.size == 0 or float(np.median(reappearance_error)) > 5.0:
        blockers.append("tapir_reappearance_gate")
    if true_positive / max(true_positive + false_negative, 1) < 0.8:
        blockers.append("tapir_visibility_recall_gate")
    corruption_clean = build_material_chart_graph(public_chart_fixture())
    corruption_treatment = build_material_chart_graph(public_chart_fixture(corrupt=True))
    clean_capacity = sum(track.accepted for track in corruption_clean.tracks)
    corrupt_capacity = sum(track.accepted for track in corruption_treatment.tracks)
    capacity_regression = max(clean_capacity - corrupt_capacity, 0)
    if capacity_regression != 0:
        blockers.append("corrupted_proposal_capacity_regression")
    if not exact_same_device_replay:
        blockers.append("tapir_exact_same_device_replay_failure")
    replay_digest = hashlib.sha256()
    replay_digest.update(np.ascontiguousarray(tracks).tobytes())
    replay_digest.update(np.ascontiguousarray(predicted_visibility).tobytes())
    return {
        "schema_version": "frayid_v3_q04_public_tracker_calibration.v1",
        "source": "tapir",
        "status": "pass" if not blockers else "fail",
        "source_revision": revision,
        "checkpoint_sha256": observed_checkpoint_sha256,
        "device": str(device),
        "runtime": f"torch-{torch.__version__}",
        "conditions": {
            "occlusion": True,
            "reappearance": True,
            "repeated_texture": True,
            "drift": True,
            "deliberately_corrupted_proposals": True,
        },
        "query_count": truth.shape[1],
        "frame_count": len(frames),
        "accepted_observation_count": int(np.sum(accepted)),
        "median_visible_error_pixels": median_error,
        "p95_visible_error_pixels": p95_error,
        "visibility_recall": true_positive / max(true_positive + false_negative, 1),
        "visibility_precision": true_positive / max(true_positive + false_positive, 1),
        "true_negative_count": true_negative,
        "reappearance_accepted_count": int(np.sum(reappearance_accepted)),
        "median_reappearance_error_pixels": (
            float(np.median(reappearance_error)) if reappearance_error.size else None
        ),
        "accepted_track_capacity_clean": clean_capacity,
        "accepted_track_capacity_with_corruption": corrupt_capacity,
        "corrupted_proposal_capacity_regression": capacity_regression,
        "output_sha256": replay_digest.hexdigest(),
        "exact_same_device_replay": exact_same_device_replay,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_thread_count": torch.get_num_threads(),
        "weights_imported": True,
        "weights_executed": True,
        "material_truth_write_access": False,
        "project_evidence_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "blockers": blockers,
    }


def calibrate_cotracker_public_cloth(
    *,
    source_root: Path,
    checkpoint_path: Path,
    expected_source_revision: str,
    expected_checkpoint_sha256: str,
    device_name: str = "cpu",
    verify_replay: bool = True,
) -> dict[str, Any]:
    """Execute pinned official CoTracker3 on public cloth as proposals only."""
    predictor_path = source_root / "cotracker/predictor.py"
    if not source_root.is_dir() or not predictor_path.is_file():
        raise FileNotFoundError(f"pinned CoTracker3 source is unavailable: {source_root}")
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_source_revision:
        raise ValueError(f"CoTracker3 source revision mismatch: {revision}")
    observed_checkpoint_sha256 = sha256_file(checkpoint_path)
    if observed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("CoTracker3 checkpoint SHA-256 mismatch")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("cotracker.predictor")
    predictor = module.CoTrackerPredictor(
        checkpoint=None,
        offline=True,
        v2=False,
        window_len=60,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    predictor.model.load_state_dict(state)
    predictor = predictor.to(device).eval()

    frames, truth, visibility = _synthetic_cloth_sequence()
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in frames])
    video_tensor = (
        torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].to(device=device, dtype=torch.float32)
    )
    query_points = np.zeros((truth.shape[1], 3), dtype=np.float32)
    query_points[:, 1:] = truth[0]
    query_tensor = torch.from_numpy(query_points)[None].to(device=device, dtype=torch.float32)

    def infer() -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            tracks_tensor, visibility_tensor = predictor(video_tensor, queries=query_tensor)
        inferred_tracks = tracks_tensor[0].detach().cpu().numpy().copy()
        inferred_visibility = visibility_tensor[0].detach().cpu().numpy().astype(bool).copy()
        return inferred_tracks, inferred_visibility

    tracks, predicted_visibility = infer()
    exact_same_device_replay = True
    if verify_replay:
        replay_tracks, replay_visibility = infer()
        exact_same_device_replay = np.array_equal(tracks, replay_tracks) and np.array_equal(
            predicted_visibility, replay_visibility
        )

    error = np.linalg.norm(tracks - truth, axis=2)
    accepted = visibility & predicted_visibility
    accepted_error = error[accepted]
    if accepted_error.size == 0:
        raise RuntimeError("CoTracker3 accepted no visible public truth points")
    true_positive = int(np.sum(visibility & predicted_visibility))
    false_negative = int(np.sum(visibility & ~predicted_visibility))
    false_positive = int(np.sum(~visibility & predicted_visibility))
    true_negative = int(np.sum(~visibility & ~predicted_visibility))
    reappeared_points = (~visibility[6]) & visibility[7]
    reappearance_accepted = reappeared_points & predicted_visibility[7]
    reappearance_error = error[7, reappearance_accepted]
    median_error = float(np.median(accepted_error))
    p95_error = float(np.percentile(accepted_error, 95))
    blockers: list[str] = []
    if median_error > 2.0 or p95_error > 5.0:
        blockers.append("cotracker3_visible_track_error_gate")
    if reappearance_error.size == 0 or float(np.median(reappearance_error)) > 5.0:
        blockers.append("cotracker3_reappearance_gate")
    if true_positive / max(true_positive + false_negative, 1) < 0.8:
        blockers.append("cotracker3_visibility_recall_gate")
    corruption_clean = build_material_chart_graph(public_chart_fixture())
    corruption_treatment = build_material_chart_graph(public_chart_fixture(corrupt=True))
    clean_capacity = sum(track.accepted for track in corruption_clean.tracks)
    corrupt_capacity = sum(track.accepted for track in corruption_treatment.tracks)
    capacity_regression = max(clean_capacity - corrupt_capacity, 0)
    if capacity_regression != 0:
        blockers.append("corrupted_proposal_capacity_regression")
    if not exact_same_device_replay:
        blockers.append("cotracker3_exact_same_device_replay_failure")
    replay_digest = hashlib.sha256()
    replay_digest.update(np.ascontiguousarray(tracks).tobytes())
    replay_digest.update(np.ascontiguousarray(predicted_visibility).tobytes())
    return {
        "schema_version": "frayid_v3_q04_public_tracker_calibration.v1",
        "source": "cotracker3",
        "status": "pass" if not blockers else "fail",
        "source_revision": revision,
        "checkpoint_sha256": observed_checkpoint_sha256,
        "device": str(device),
        "runtime": f"torch-{torch.__version__}",
        "conditions": {
            "occlusion": True,
            "reappearance": True,
            "repeated_texture": True,
            "drift": True,
            "deliberately_corrupted_proposals": True,
        },
        "query_count": truth.shape[1],
        "frame_count": len(frames),
        "accepted_observation_count": int(np.sum(accepted)),
        "median_visible_error_pixels": median_error,
        "p95_visible_error_pixels": p95_error,
        "visibility_recall": true_positive / max(true_positive + false_negative, 1),
        "visibility_precision": true_positive / max(true_positive + false_positive, 1),
        "true_negative_count": true_negative,
        "reappearance_accepted_count": int(np.sum(reappearance_accepted)),
        "median_reappearance_error_pixels": (
            float(np.median(reappearance_error)) if reappearance_error.size else None
        ),
        "accepted_track_capacity_clean": clean_capacity,
        "accepted_track_capacity_with_corruption": corrupt_capacity,
        "corrupted_proposal_capacity_regression": capacity_regression,
        "output_sha256": replay_digest.hexdigest(),
        "exact_same_device_replay": exact_same_device_replay,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_thread_count": torch.get_num_threads(),
        "weights_imported": True,
        "weights_executed": True,
        "material_truth_write_access": False,
        "project_evidence_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "blockers": blockers,
    }


def public_tracker_calibration_status(
    lk_report: dict[str, Any],
    source_audit: dict[str, object],
    learned_reports: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    sources = source_audit.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source audit has no source reports")
    learned_by_source = {str(item.get("source")): item for item in (learned_reports or [])}
    source_states: dict[str, str] = {"lk": str(lk_report["status"])}
    blockers: list[str] = []
    if lk_report["status"] != "pass":
        blockers.append("lk_public_calibration_failed")
    for raw in sources:
        if not isinstance(raw, dict):
            raise ValueError("invalid tracker source report")
        source = str(raw["source"])
        if source == "lk":
            continue
        checkpoint_ready = raw.get("checkpoint_status") == "verified"
        license_ready = bool(raw.get("license_ready_for_real_use", False))
        learned_report = learned_by_source.get(source)
        source_blockers: list[str] = []
        if not checkpoint_ready:
            source_blockers.append("checkpoint_not_verified")
            blockers.append(f"public_calibration_checkpoint_not_verified:{source}")
        if not license_ready:
            source_blockers.append("license_confirmation")
            blockers.append(f"public_calibration_license_confirmation:{source}")
        if source_blockers:
            source_states[source] = "blocked_" + "_and_".join(source_blockers)
        elif learned_report is not None and learned_report.get("status") == "pass":
            source_states[source] = "pass"
        elif learned_report is not None:
            source_states[source] = "failed_public_calibration"
            blockers.append(f"public_calibration_failed:{source}")
        else:
            source_states[source] = "ready_not_executed"
            blockers.append(f"public_calibration_not_executed:{source}")
    return {
        "schema_version": "frayid_v3_q04_public_tracker_calibration_status.v1",
        "experiment_id": "postv3_q04_local_material_chart_graph_r01",
        "status": "blocked" if blockers else "pass",
        "source_states": source_states,
        "blockers": blockers,
        "promotion_eligible": False,
        "project_evidence_reads": 0,
        "sealed_test_accesses": 0,
    }
