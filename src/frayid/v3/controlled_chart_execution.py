from __future__ import annotations

import hashlib
import json
import math
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.evidence import SAPIENS2_DOME29_LAYER_IDS
from frayid.v3.schemas import MaterialChartGraph

EXPERIMENT_ID = "postv3_q05_controlled_material_chart_graph_r01"
SOURCES = ("lk", "tapir", "cotracker3")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_q05_request(path: Path) -> dict[str, Any]:
    request = read_json(path)
    if request.get("schema_version") != "frayid_v3_controlled_chart_request_manifest.v1":
        raise ValueError("unexpected controlled chart request schema")
    if request.get("experiment_id") != EXPERIMENT_ID or request.get("status") != "pass":
        raise ValueError("Q05 request manifest is not qualified")
    if request.get("promotion_eligible") is not False:
        raise ValueError("Q05 tracker requests cannot be promotion evidence")
    records = request.get("requests")
    if not isinstance(records, list) or len(records) != 72:
        raise ValueError("Q05 request manifest must contain exactly 72 anchors")
    if int(request.get("evaluator_camera_reads", -1)) != 0:
        raise ValueError("evaluator evidence cannot enter Q05 requests")
    return request


def _coordinate_contract(request: dict[str, Any]) -> tuple[np.ndarray, int, int, int, int]:
    contract = request.get("proxy_coordinate_contract")
    if not isinstance(contract, dict):
        raise ValueError("Q05 request has no proxy coordinate contract")
    source_width = int(contract.get("source_width", 0))
    source_height = int(contract.get("source_height", 0))
    proxy_width = int(contract.get("proxy_width", 0))
    proxy_height = int(contract.get("proxy_height", 0))
    if min(source_width, source_height, proxy_width, proxy_height) <= 0:
        raise ValueError("Q05 coordinate dimensions must be positive")
    forward = np.asarray(contract.get("source_to_proxy_homography"), dtype=np.float64)
    inverse = np.asarray(contract.get("proxy_to_source_homography"), dtype=np.float64)
    if forward.shape != (3, 3) or inverse.shape != (3, 3):
        raise ValueError("Q05 coordinate homographies must be 3 by 3")
    if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(inverse)):
        raise ValueError("Q05 coordinate homographies must be finite")
    if not np.allclose(forward @ inverse, np.eye(3), atol=1.0e-9, rtol=0.0):
        raise ValueError("Q05 proxy coordinate transforms are not inverse")
    return inverse, source_width, source_height, proxy_width, proxy_height


def _select_seed_pixels(
    mask: np.ndarray,
    *,
    erosion_pixels: int,
    spacing_pixels: float,
    maximum_count: int,
) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("seed mask must be a two-dimensional Boolean array")
    if erosion_pixels < 0 or spacing_pixels <= 0.0 or maximum_count <= 0:
        raise ValueError("invalid controlled seed policy")
    eroded = binary_erosion(mask, iterations=erosion_pixels) if erosion_pixels else mask.copy()
    candidates = np.argwhere(eroded)
    if candidates.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    boundary_distance = distance_transform_edt(mask)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -float(boundary_distance[tuple(candidates[index])]),
            int(candidates[index, 0]),
            int(candidates[index, 1]),
        ),
    )
    accepted: list[np.ndarray] = []
    minimum_squared_distance = spacing_pixels**2
    for index in order:
        y, x = candidates[index]
        point = np.asarray([float(x), float(y)], dtype=np.float64)
        if all(
            float(np.sum((point - previous) ** 2)) >= minimum_squared_distance
            for previous in accepted
        ):
            accepted.append(point)
            if len(accepted) == maximum_count:
                break
    return np.asarray(accepted, dtype=np.float64).reshape(-1, 2)


def _source_point(proxy_xy: np.ndarray, proxy_to_source: np.ndarray) -> np.ndarray:
    homogeneous = proxy_to_source @ np.asarray([proxy_xy[0], proxy_xy[1], 1.0])
    if not np.all(np.isfinite(homogeneous)) or abs(float(homogeneous[2])) < 1.0e-12:
        raise ValueError("proxy-to-source projection is singular")
    return np.asarray(homogeneous[:2] / homogeneous[2], dtype=np.float64)


def materialize_controlled_chart_seeds(
    *,
    request_manifest_path: Path,
    output_path: Path,
) -> Path:
    """Freeze deterministic upper-garment queries before any tracker executes."""
    reject_sealed_capability([request_manifest_path, output_path])
    if output_path.exists():
        raise FileExistsError(f"controlled chart seed output is immutable: {output_path}")
    request = _require_q05_request(request_manifest_path)
    proxy_to_source, source_width, source_height, proxy_width, proxy_height = _coordinate_contract(
        request
    )
    upper_ids = SAPIENS2_DOME29_LAYER_IDS["upper_clothing"]
    frames: list[dict[str, Any]] = []
    all_seed_ids: set[str] = set()
    for raw in request["requests"]:
        if not isinstance(raw, dict):
            raise ValueError("each Q05 request must be an object")
        index = int(raw["controlled_record_index"])
        semantic = raw.get("semantic_evidence")
        policy = raw.get("upper_garment_seed_policy")
        if not isinstance(semantic, dict) or not isinstance(policy, dict):
            raise ValueError(f"Q05 request has no semantic seed policy: {index}")
        semantic_path = Path(str(semantic.get("semantic_path", "")))
        reject_sealed_capability([semantic_path])
        if not semantic_path.is_file() or sha256_file(semantic_path) != semantic.get(
            "semantic_sha256"
        ):
            raise ValueError(f"semantic evidence hash mismatch while seeding: {index}")
        with np.load(semantic_path, allow_pickle=False) as archive:
            labels = np.asarray(archive["labels"])
            confidence = np.asarray(archive["confidence"], dtype=np.float64)
        if labels.shape != (proxy_height, proxy_width) or confidence.shape != labels.shape:
            raise ValueError(f"semantic coordinates changed before seeding: {index}")
        confidence_minimum = float(policy["confidence_minimum"])
        mask = np.isin(labels, upper_ids) & (confidence >= confidence_minimum)
        erosion_pixels = int(policy["boundary_erosion_pixels"])
        spacing_pixels = float(policy["minimum_seed_spacing_pixels"])
        maximum_count = int(policy["maximum_seed_count"])
        selected = _select_seed_pixels(
            mask,
            erosion_pixels=erosion_pixels,
            spacing_pixels=spacing_pixels,
            maximum_count=maximum_count,
        )
        replay = _select_seed_pixels(
            mask,
            erosion_pixels=erosion_pixels,
            spacing_pixels=spacing_pixels,
            maximum_count=maximum_count,
        )
        if not np.array_equal(selected, replay):
            raise RuntimeError("controlled seed selection did not replay exactly")
        if len(selected) < 2:
            raise ValueError(f"fewer than two interior upper-garment seeds: {index}")
        phase_bin = math.floor((float(raw["angle_degrees"]) % 360.0) / 30.0)
        seeds: list[dict[str, Any]] = []
        for seed_index, proxy_xy in enumerate(selected):
            source_xy = _source_point(proxy_xy, proxy_to_source)
            if not (0.0 <= source_xy[0] < source_width and 0.0 <= source_xy[1] < source_height):
                raise ValueError(f"controlled seed maps outside native pixels: {index}")
            seed_id = f"controlled-{index:02d}-seed-{seed_index:03d}"
            if seed_id in all_seed_ids:
                raise RuntimeError("controlled seed IDs are not globally unique")
            all_seed_ids.add(seed_id)
            seeds.append(
                {
                    "seed_id": seed_id,
                    "chart_id": f"controlled-chart-{phase_bin:02d}",
                    "semantic_xy_pixels": proxy_xy.tolist(),
                    "source_xy_pixels": source_xy.tolist(),
                    "semantic_posterior": {
                        "upper_garment": confidence[int(proxy_xy[1]), int(proxy_xy[0])]
                    },
                    "tracker_role": "proposal_only",
                }
            )
        frames.append(
            {
                "controlled_record_index": index,
                "direction": raw["direction"],
                "angle_degrees": raw["angle_degrees"],
                "source_frame_key": raw["source_frame_key"],
                "source_decode_index": raw["source_decode_index"],
                "semantic_sha256": semantic["semantic_sha256"],
                "seed_count": len(seeds),
                "seeds": seeds,
            }
        )
    if [int(frame["controlled_record_index"]) for frame in frames] != list(range(72)):
        raise ValueError("controlled seed records must remain ordered 0 through 71")
    if len(all_seed_ids) < 100:
        raise ValueError("controlled chart seed capacity is below 100 tracks")
    replay_payload = {
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "frames": frames,
    }
    payload = {
        "schema_version": "frayid_v3_controlled_chart_seed_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_scope": request["evidence_scope"],
        "status": "pass",
        "promotion_eligible": False,
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "proxy_coordinate_contract": request["proxy_coordinate_contract"],
        "selection_algorithm": (
            "confidence_mask_then_binary_erosion_then_boundary_distance_ordered_poisson"
        ),
        "tie_break_order": "descending_boundary_distance_then_y_then_x",
        "frame_count": len(frames),
        "seed_count": len(all_seed_ids),
        "frames": frames,
        "exact_same_input_replay": True,
        "exact_replay_hash": _canonical_sha256(replay_payload),
        "tracker_truth_write_access": False,
        "evaluator_camera_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "blockers": [],
    }
    return _write_json_exclusive(output_path, payload)


def _seed_inventory(seed_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frames = seed_manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != 72:
        raise ValueError("controlled seed manifest must contain 72 frames")
    inventory: dict[str, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("controlled seed frame must be an object")
        for seed in frame.get("seeds", []):
            if not isinstance(seed, dict):
                raise ValueError("controlled seed must be an object")
            seed_id = str(seed["seed_id"])
            if seed_id in inventory:
                raise ValueError("controlled seed IDs must be unique")
            inventory[seed_id] = {
                **seed,
                **{
                    key: frame[key]
                    for key in (
                        "controlled_record_index",
                        "direction",
                        "angle_degrees",
                        "source_frame_key",
                        "source_decode_index",
                    )
                },
            }
    if len(inventory) != int(seed_manifest.get("seed_count", -1)):
        raise ValueError("controlled seed manifest count mismatch")
    return inventory


def _validate_covariance(raw: Any) -> None:
    covariance = np.asarray(raw, dtype=np.float64)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("tracker covariance must be a finite 2 by 2 matrix")
    if not np.allclose(covariance, covariance.T, atol=1.0e-8, rtol=0.0):
        raise ValueError("tracker covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
        raise ValueError("tracker covariance must be positive definite")


def _validate_tracker_payload(
    payload: dict[str, Any],
    *,
    source: str,
    request_sha256: str,
    seed_sha256: str,
    seeds: dict[str, dict[str, Any]],
    requests_by_index: dict[int, dict[str, Any]],
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    if payload.get("schema_version") != "frayid_v3_controlled_tracker_proposals.v1":
        raise ValueError("unexpected controlled tracker proposal schema")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "source": source,
        "role": "proposal_only",
        "request_manifest_sha256": request_sha256,
        "seed_manifest_sha256": seed_sha256,
        "transition_frames_role": "proposal_context_only_never_measured_fit_truth",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"controlled tracker payload binding mismatch: {key}")
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("controlled tracker payload tracks must be a list")
    by_seed = {str(track.get("seed_id")): track for track in tracks if isinstance(track, dict)}
    if set(by_seed) != set(seeds) or len(by_seed) != len(tracks):
        raise ValueError("controlled tracker payload must report every seed exactly once")
    observation_count = 0
    visible_count = 0
    for seed_id, expected_seed in seeds.items():
        track = by_seed[seed_id]
        seed_record_index = int(expected_seed["controlled_record_index"])
        if int(track.get("seed_controlled_record_index", -1)) != seed_record_index:
            raise ValueError(f"tracker seed record mismatch: {seed_id}")
        if track.get("chart_id") != expected_seed["chart_id"]:
            raise ValueError(f"tracker chart assignment mismatch: {seed_id}")
        query = np.asarray(track.get("query_xy_source_pixels"), dtype=np.float64)
        expected_query = np.asarray(expected_seed["source_xy_pixels"], dtype=np.float64)
        if query.shape != (2,) or not np.array_equal(query, expected_query):
            raise ValueError(f"tracker query changed after registration: {seed_id}")
        observations = track.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"tracker emitted no seed observation: {seed_id}")
        observation_indices = [int(item["controlled_record_index"]) for item in observations]
        if len(set(observation_indices)) != len(observation_indices):
            raise ValueError(f"tracker duplicated a controlled observation: {seed_id}")
        seed_direction_slot = 0 if seed_record_index < 36 else 1
        if any((index < 36) != (seed_direction_slot == 0) for index in observation_indices):
            raise ValueError(f"tracker crossed directional clips: {seed_id}")
        local_indices = [index % 36 for index in observation_indices]
        if max(local_indices) - min(local_indices) > 9:
            raise ValueError(f"tracker interval exceeds 90 degrees: {seed_id}")
        seed_observation: dict[str, Any] | None = None
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError("tracker observation must be an object")
            index = int(observation["controlled_record_index"])
            request = requests_by_index[index]
            if observation.get("source_frame_key") != request["source_frame_key"]:
                raise ValueError(f"tracker source-frame key mismatch: {seed_id}:{index}")
            if int(observation.get("source_frame_index", -1)) != int(
                request["source_decode_index"]
            ):
                raise ValueError(f"tracker native frame index mismatch: {seed_id}:{index}")
            if observation.get("evidence_role") != "proposal_only":
                raise ValueError("tracker observation attempted to claim measured truth")
            xy = np.asarray(observation.get("xy"), dtype=np.float64)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                raise ValueError(f"tracker coordinates are invalid: {seed_id}:{index}")
            if bool(observation.get("visible")) and not (
                0.0 <= xy[0] < source_width and 0.0 <= xy[1] < source_height
            ):
                raise ValueError(f"visible tracker point lies outside source pixels: {seed_id}")
            _validate_covariance(observation.get("covariance"))
            if index == seed_record_index:
                seed_observation = observation
            observation_count += 1
            visible_count += int(bool(observation.get("visible")))
        if seed_observation is None or seed_observation.get("visible") is not True:
            raise ValueError(f"tracker did not preserve its visible query: {seed_id}")
        seed_xy = np.asarray(seed_observation["xy"], dtype=np.float64)
        if float(np.linalg.norm(seed_xy - expected_query)) > 0.25:
            raise ValueError(f"tracker moved its registered query point: {seed_id}")
    return observation_count, visible_count


def validate_controlled_tracker_output_bundle(
    *,
    request_manifest_path: Path,
    seed_manifest_path: Path,
    bundle_path: Path,
    output_path: Path,
) -> Path:
    """Verify one source's byte-identical Q05 replay and proposal-only ownership."""
    reject_sealed_capability([request_manifest_path, seed_manifest_path, bundle_path, output_path])
    if output_path.exists():
        raise FileExistsError(f"controlled tracker validation is immutable: {output_path}")
    request = _require_q05_request(request_manifest_path)
    request_sha = sha256_file(request_manifest_path)
    seed_manifest = read_json(seed_manifest_path)
    if seed_manifest.get("schema_version") != "frayid_v3_controlled_chart_seed_manifest.v1":
        raise ValueError("unexpected controlled chart seed schema")
    if seed_manifest.get("request_manifest_sha256") != request_sha:
        raise ValueError("controlled seeds are not bound to this request manifest")
    if (
        seed_manifest.get("status") != "pass"
        or seed_manifest.get("exact_same_input_replay") is not True
    ):
        raise ValueError("controlled chart seeds are not qualified")
    seeds = _seed_inventory(seed_manifest)
    seed_sha = sha256_file(seed_manifest_path)
    bundle = read_json(bundle_path)
    if bundle.get("schema_version") != "frayid_v3_controlled_tracker_output_bundle.v1":
        raise ValueError("unexpected controlled tracker output bundle schema")
    source = str(bundle.get("source"))
    if source not in SOURCES:
        raise ValueError("controlled tracker output has an unknown source")
    bindings = {
        "experiment_id": EXPERIMENT_ID,
        "role": "proposal_only",
        "request_manifest_sha256": request_sha,
        "seed_manifest_sha256": seed_sha,
        "tracker_source_audit_sha256": request["tracker_source_audit_sha256"],
    }
    for key, value in bindings.items():
        if bundle.get(key) != value:
            raise ValueError(f"controlled tracker bundle binding mismatch: {key}")
    if bundle.get("material_truth_write_access") is not False:
        raise ValueError("a tracker cannot write controlled material truth")
    evidence_scope = str(request["evidence_scope"])
    expected_training_reads = 72 if evidence_scope == "train_real" else 0
    counters = {
        "controlled_records_processed": 72,
        "training_records_read": expected_training_reads,
        "evaluator_camera_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "automatic_retries": 0,
    }
    for key, value in counters.items():
        if int(bundle.get(key, -1)) != value:
            raise ValueError(f"controlled tracker bundle counter mismatch: {key}")
    if bundle.get("transition_frames_role") != ("proposal_context_only_never_measured_fit_truth"):
        raise ValueError("controlled transition frames have an invalid evidence role")
    source_audit_path = Path(str(request.get("tracker_source_audit_path", "")))
    reject_sealed_capability([source_audit_path])
    if not source_audit_path.is_file() or sha256_file(source_audit_path) != request.get(
        "tracker_source_audit_sha256"
    ):
        raise ValueError("controlled tracker source audit hash mismatch")
    audit = read_json(source_audit_path)
    audit_by_source = {
        str(item.get("source")): item for item in audit.get("sources", []) if isinstance(item, dict)
    }
    source_record = audit_by_source.get(source)
    if source_record is None:
        raise ValueError("controlled tracker source is absent from its audit")
    for key in ("source_revision", "license"):
        if bundle.get(key) != source_record.get(key):
            raise ValueError(f"controlled tracker provenance mismatch: {key}")
    checkpoint = bundle.get("checkpoint_sha256")
    expected_checkpoint = source_record.get("checkpoint_observed_sha256")
    if checkpoint != expected_checkpoint:
        raise ValueError("controlled tracker checkpoint hash mismatch")
    if source == "lk":
        if bundle.get("weights_executed") is not False or checkpoint is not None:
            raise ValueError("LK cannot claim a learned checkpoint")
    elif evidence_scope == "train_real":
        if bundle.get("weights_executed") is not True:
            raise ValueError("real learned tracker output did not execute weights")
        if not _SHA256_PATTERN.fullmatch(str(checkpoint)):
            raise ValueError("real learned tracker output has no checkpoint hash")
        if source_record.get("license_ready_for_real_use") is not True:
            raise ValueError("real tracker output is not license-authorized")
    primary = bundle.get("primary_payload")
    replay = bundle.get("replay_payload")
    if not isinstance(primary, dict) or not isinstance(replay, dict):
        raise ValueError("controlled tracker bundle requires primary and replay payloads")
    primary_path = Path(str(primary.get("path", "")))
    replay_path = Path(str(replay.get("path", "")))
    reject_sealed_capability([primary_path, replay_path])
    if primary_path == replay_path:
        raise ValueError("exact replay must be a distinct execution artifact")
    for label, reference, path in (
        ("primary", primary, primary_path),
        ("replay", replay, replay_path),
    ):
        if not path.is_file() or sha256_file(path) != reference.get("sha256"):
            raise ValueError(f"controlled tracker {label} payload hash mismatch")
    if primary["sha256"] != replay["sha256"]:
        raise ValueError("controlled tracker same-device replay is not byte-identical")
    if bundle.get("exact_same_device_replay") is not True:
        raise ValueError("controlled tracker did not declare exact same-device replay")
    primary_payload = read_json(primary_path)
    replay_payload = read_json(replay_path)
    if primary_payload != replay_payload:
        raise ValueError("controlled tracker replay payload content differs")
    requests_by_index = {int(item["controlled_record_index"]): item for item in request["requests"]}
    _, source_width, source_height, _, _ = _coordinate_contract(request)
    observation_count, visible_count = _validate_tracker_payload(
        primary_payload,
        source=source,
        request_sha256=request_sha,
        seed_sha256=seed_sha,
        seeds=seeds,
        requests_by_index=requests_by_index,
        source_width=source_width,
        source_height=source_height,
    )
    payload = {
        "schema_version": "frayid_v3_controlled_tracker_output_validation.v1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_scope": evidence_scope,
        "source": source,
        "status": "pass",
        "promotion_eligible": False,
        "role": "proposal_only",
        "request_manifest_sha256": request_sha,
        "seed_manifest_sha256": seed_sha,
        "bundle_sha256": sha256_file(bundle_path),
        "model_output_sha256": primary["sha256"],
        "exact_same_device_replay": True,
        "track_count": len(seeds),
        "observation_count": observation_count,
        "visible_observation_count": visible_count,
        "material_truth_write_access": False,
        "evaluator_camera_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "blockers": [],
    }
    return _write_json_exclusive(output_path, payload)


def _rotation_about_axis(
    angle_radians: float,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    rotation = Rotation.from_rotvec(direction * angle_radians).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin - rotation @ origin
    return transform


def _projection_by_record(
    request: dict[str, Any],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    calibration = request.get("training_camera_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Q05 cannot profile anchors without training-camera calibration")
    if calibration.get("schema_version") != "frayid_v3_training_camera_calibration.v2":
        raise ValueError("Q05 anchor profiling requires rotation-axis calibration v2")
    intrinsics = calibration.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError("Q05 training-camera intrinsics are missing")
    camera_matrix = np.asarray(intrinsics.get("camera_matrix"), dtype=np.float64)
    distortion = np.asarray(
        intrinsics.get("distortion_coefficients", []), dtype=np.float64
    ).reshape(-1)
    world_from_camera = np.asarray(calibration.get("world_from_camera"), dtype=np.float64)
    origin = np.asarray(calibration.get("rotation_axis_origin_world_m"), dtype=np.float64)
    direction = np.asarray(calibration.get("rotation_axis_direction_world"), dtype=np.float64)
    if (
        camera_matrix.shape != (3, 3)
        or world_from_camera.shape != (4, 4)
        or origin.shape != (3,)
        or direction.shape != (3,)
        or not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(world_from_camera))
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(direction))
    ):
        raise ValueError("Q05 camera/rotation-axis calibration is invalid")
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("Q05 rotation-axis direction is not normalized")
    camera_from_world = np.linalg.inv(world_from_camera)
    projections: dict[int, np.ndarray] = {}
    for record in request["requests"]:
        index = int(record["controlled_record_index"])
        rotation = _rotation_about_axis(
            math.radians(float(record["angle_degrees"])), origin, direction
        )
        projections[index] = camera_matrix @ camera_from_world[:3] @ rotation
    return projections, camera_matrix, distortion


def _profile_anchor_reprojection(
    tracks: list[dict[str, Any]],
    projections: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for track in tracks:
        if not bool(track.get("accepted")):
            continue
        observations = [item for item in track["observations"] if bool(item["visible"])]
        if len(observations) < 4:
            continue
        pixels = np.asarray([item["xy"] for item in observations], dtype=np.float64)
        undistorted = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2),
            camera_matrix,
            distortion,
            P=camera_matrix,
        ).reshape(-1, 2)
        rows: list[np.ndarray] = []
        for observation, xy in zip(observations, undistorted, strict=True):
            projection = projections[int(observation["frame_index"])]
            covariance = np.asarray(observation["covariance"], dtype=np.float64)
            weight = 1.0 / max(float(np.trace(covariance)), 1.0e-9)
            rows.extend(
                [
                    math.sqrt(weight) * (xy[0] * projection[2] - projection[0]),
                    math.sqrt(weight) * (xy[1] * projection[2] - projection[1]),
                ]
            )
        _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        homogeneous = vh[-1]
        if abs(float(homogeneous[3])) < 1.0e-12:
            continue
        point = homogeneous[:3] / homogeneous[3]
        for observation, target in zip(observations, undistorted, strict=True):
            projection = projections[int(observation["frame_index"])]
            projected = projection @ np.asarray([*point, 1.0])
            if projected[2] <= 1.0e-9:
                continue
            xy = projected[:2] / projected[2]
            errors.append(float(np.linalg.norm(xy - target)))
    return errors


def _fit_transition(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[list[list[float]], float]:
    if len(source) < 3:
        raise ValueError("controlled chart transition needs at least three overlap tracks")
    design = np.column_stack((source, np.ones(len(source))))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    residual = np.linalg.norm(prediction - target, axis=1)
    return (
        [
            [
                float(coefficients[0, 0]),
                float(coefficients[1, 0]),
                float(coefficients[2, 0]),
            ],
            [
                float(coefficients[0, 1]),
                float(coefficients[1, 1]),
                float(coefficients[2, 1]),
            ],
        ],
        float(np.median(residual)),
    )


def _controlled_transitions(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_bin_by_record = {
        **{index: index // 3 for index in range(36)},
        **{36 + index: ((36 - index) % 36) // 3 for index in range(36)},
    }
    sequences = [list(range(12)), [0, *range(11, 0, -1)]]
    transitions: list[dict[str, Any]] = []
    for sequence in sequences:
        for source_bin, target_bin in pairwise(sequence):
            source_points: list[np.ndarray] = []
            target_points: list[np.ndarray] = []
            overlap_ids: list[str] = []
            for track in tracks:
                if not bool(track.get("accepted")):
                    continue
                by_bin: dict[int, list[np.ndarray]] = {}
                for observation in track["observations"]:
                    if not bool(observation["visible"]):
                        continue
                    phase_bin = phase_bin_by_record[int(observation["frame_index"])]
                    by_bin.setdefault(phase_bin, []).append(
                        np.asarray(observation["xy"], dtype=np.float64)
                    )
                if source_bin in by_bin and target_bin in by_bin:
                    source_points.append(np.median(by_bin[source_bin], axis=0))
                    target_points.append(np.median(by_bin[target_bin], axis=0))
                    overlap_ids.append(str(track["track_id"]))
            if len(source_points) < 3:
                continue
            affine, residual = _fit_transition(np.asarray(source_points), np.asarray(target_points))
            transitions.append(
                {
                    "source_chart_id": f"controlled-chart-{source_bin:02d}",
                    "target_chart_id": f"controlled-chart-{target_bin:02d}",
                    "overlap_track_ids": overlap_ids,
                    "affine_map": affine,
                    "cycle_residual_pixels": residual,
                }
            )
    return transitions


def _connected_chart_count(transitions: list[dict[str, Any]]) -> int:
    adjacency: dict[str, set[str]] = {}
    for transition in transitions:
        source = str(transition["source_chart_id"])
        target = str(transition["target_chart_id"])
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    maximum = 0
    unseen = set(adjacency)
    while unseen:
        frontier = [unseen.pop()]
        size = 0
        while frontier:
            node = frontier.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    frontier.append(neighbor)
        maximum = max(maximum, size)
    return maximum


def build_controlled_material_chart_graph(
    *,
    request_manifest_path: Path,
    seed_manifest_path: Path,
    bundle_paths: list[Path],
    validation_paths: list[Path],
    public_robustness_path: Path,
) -> MaterialChartGraph:
    """Assemble three audited proposal sources into the real controlled Q05 graph."""
    all_paths = [
        request_manifest_path,
        seed_manifest_path,
        public_robustness_path,
        *bundle_paths,
        *validation_paths,
    ]
    reject_sealed_capability(all_paths)
    request = _require_q05_request(request_manifest_path)
    request_sha = sha256_file(request_manifest_path)
    seed_manifest = read_json(seed_manifest_path)
    seed_sha = sha256_file(seed_manifest_path)
    if (
        seed_manifest.get("schema_version") != "frayid_v3_controlled_chart_seed_manifest.v1"
        or seed_manifest.get("request_manifest_sha256") != request_sha
        or seed_manifest.get("status") != "pass"
    ):
        raise ValueError("Q05 seed manifest is not bound and qualified")
    seed_inventory = _seed_inventory(seed_manifest)
    validations: dict[str, dict[str, Any]] = {}
    for path in validation_paths:
        report = read_json(path)
        source = str(report.get("source"))
        if (
            report.get("schema_version") != "frayid_v3_controlled_tracker_output_validation.v1"
            or report.get("status") != "pass"
            or report.get("request_manifest_sha256") != request_sha
            or report.get("seed_manifest_sha256") != seed_sha
        ):
            raise ValueError(f"Q05 tracker validation is not qualified: {path}")
        if source in validations:
            raise ValueError("duplicate Q05 tracker validation source")
        validations[source] = report
    bundles: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for path in bundle_paths:
        bundle = read_json(path)
        source = str(bundle.get("source"))
        validation = validations.get(source)
        if validation is None or validation.get("bundle_sha256") != sha256_file(path):
            raise ValueError(f"Q05 bundle has no matching immutable validation: {source}")
        primary = bundle.get("primary_payload")
        if not isinstance(primary, dict):
            raise ValueError("Q05 bundle has no primary proposal payload")
        primary_path = Path(str(primary.get("path", "")))
        reject_sealed_capability([primary_path])
        if not primary_path.is_file() or sha256_file(primary_path) != primary.get("sha256"):
            raise ValueError(f"Q05 primary tracker payload hash mismatch: {source}")
        if source in bundles:
            raise ValueError("duplicate Q05 tracker bundle source")
        bundles[source] = bundle
        payloads[source] = read_json(primary_path)
    if set(bundles) != set(SOURCES) or set(validations) != set(SOURCES):
        raise ValueError("Q05 graph requires validated LK, TAPIR, and CoTracker3 bundles")
    public_report = read_json(public_robustness_path)
    if (
        public_report.get("schema_version") != "frayid_v3_q05_controlled_chart_robustness.v1"
        or public_report.get("status") != "pass"
        or int(public_report.get("project_evidence_reads", -1)) != 0
        or int(public_report.get("sealed_test_accesses", -1)) != 0
    ):
        raise ValueError("Q05 public single-source robustness qualification did not pass")
    public_graph = public_report.get("graph")
    if not isinstance(public_graph, dict):
        raise ValueError("Q05 public robustness report has no graph")
    requests_by_index = {int(item["controlled_record_index"]): item for item in request["requests"]}
    proposals: list[dict[str, Any]] = []
    proposal_counts: dict[str, int] = {}
    audits: list[dict[str, Any]] = []
    for source in SOURCES:
        bundle = bundles[source]
        tracks = payloads[source]["tracks"]
        count = 0
        for track in tracks:
            seed = seed_inventory[str(track["seed_id"])]
            for observation in track["observations"]:
                if not bool(observation["visible"]):
                    continue
                proposals.append(
                    {
                        "source": source,
                        "track_id": track["seed_id"],
                        "chart_id": track["chart_id"],
                        "frame_index": observation["controlled_record_index"],
                        "source_frame_index": observation["source_frame_index"],
                        "xy": observation["xy"],
                        "covariance": observation["covariance"],
                        "visible": True,
                        "semantic_posterior": seed["semantic_posterior"],
                    }
                )
                count += 1
        proposal_counts[source] = count
        audits.append(
            {
                "source": source,
                "source_revision": bundle["source_revision"],
                "license": bundle["license"],
                "checkpoint_sha256": bundle["checkpoint_sha256"],
                "runtime": bundle["runtime"],
                "weights_executed": bundle["weights_executed"],
                "real_use_authorized": (
                    request["evidence_scope"] != "train_real"
                    or source == "lk"
                    or bundle["checkpoint_sha256"] is not None
                ),
            }
        )
    phase_by_frame = {
        **{index: float(index * 10) for index in range(36)},
        **{36 + index: float(-index * 10) for index in range(36)},
    }
    input_hashes = {
        "request_manifest": request_sha,
        "seed_manifest": seed_sha,
        "public_single_source_robustness": sha256_file(public_robustness_path),
        **{f"{source}_bundle": validations[source]["bundle_sha256"] for source in SOURCES},
    }
    base_payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "evidence_scope": request["evidence_scope"],
        "tracker_audits": audits,
        "proposals": proposals,
        "phase_degrees_by_frame": phase_by_frame,
        "transitions": [],
        "anchor_reprojection_pixels": [1.0e9],
        "public_truth_benchmark": public_graph["public_truth_benchmark"],
        "corrupted_proposal_capacity_regression": 0.0,
        "minimum_proposal_sources_per_observation": 2,
        "input_hashes": input_hashes,
        "source_frame_indices": [
            int(requests_by_index[index]["source_decode_index"]) for index in range(72)
        ],
        "proposal_count_by_source": proposal_counts,
        "model_output_sha256_by_source": {
            source: validations[source]["model_output_sha256"] for source in SOURCES
        },
        "exact_same_device_replay_by_source": {source: True for source in SOURCES},
        "training_records_read": 72 if request["evidence_scope"] == "train_real" else 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
    }
    from frayid.v3.controlled_charts import qualify_controlled_chart_robustness
    from frayid.v3.material_charts import build_material_chart_graph

    preliminary = build_material_chart_graph(base_payload)
    preliminary_tracks = [track.model_dump(mode="json") for track in preliminary.tracks]
    projections, camera_matrix, distortion = _projection_by_record(request)
    anchor_errors = _profile_anchor_reprojection(
        preliminary_tracks, projections, camera_matrix, distortion
    )
    if not anchor_errors:
        anchor_errors = [1.0e9]
    base_payload["anchor_reprojection_pixels"] = anchor_errors
    base_payload["transitions"] = _controlled_transitions(preliminary_tracks)
    qualified = qualify_controlled_chart_robustness(base_payload)
    graph = MaterialChartGraph.model_validate(qualified["graph"])
    accepted_bins = sorted(
        {
            math.floor((phase_by_frame[observation.frame_index] % 360.0) / 30.0)
            for track in graph.tracks
            if track.accepted
            for observation in track.observations
            if observation.visible
        }
    )
    controlled_blockers = list(graph.blockers)
    transition_payload = [item.model_dump(mode="json") for item in graph.transitions]
    if _connected_chart_count(transition_payload) < 10:
        controlled_blockers.append("controlled_chart_transition_component_below_10_bins")
    if len(accepted_bins) < 10:
        controlled_blockers.append("accepted_controlled_tracks_span_below_10_phase_bins")
    controlled_blockers = sorted(set(controlled_blockers))
    return graph.model_copy(
        update={
            "phase_bins_spanned": accepted_bins,
            "status": "pass" if not controlled_blockers else "fail",
            "promotion_eligible": (
                not controlled_blockers and graph.evidence_scope == "train_real"
            ),
            "blockers": controlled_blockers,
        }
    )


__all__ = [
    "build_controlled_material_chart_graph",
    "materialize_controlled_chart_seeds",
    "validate_controlled_tracker_output_bundle",
]
