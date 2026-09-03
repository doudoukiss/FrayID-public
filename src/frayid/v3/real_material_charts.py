from __future__ import annotations

import copy
import hashlib
import importlib
import subprocess
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.material_tracks import SEMANTIC_NAMES
from frayid.v2.q03_interval_tracks import (
    IntervalMaterialTrackGraph,
    load_interval_material_track_graph,
    robust_material_anchor,
)
from frayid.v2.t05_fixed_camera import FixedCameraHumanSolution
from frayid.v3.material_charts import build_material_chart_graph
from frayid.v3.schemas import MaterialChartGraph

# The 72-frame bridge is required by the observed Q03 interval support. The
# earlier 48 -> 84 jump had no track whose complete interval was contained in
# both windows, so it could not define an audited transition map.
WINDOW_STARTS = (0, 24, 48, 72, 84)
WINDOW_LENGTH = 60
TAPIR_SOURCE_REVISION = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPIR_CHECKPOINT_SHA256 = "unavailable-in-public-snapshot"
COTRACKER_SOURCE_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_CHECKPOINT_SHA256 = "unavailable-in-public-snapshot"


@dataclass(frozen=True)
class RealChartSeed:
    track_id: str
    q03_track_index: int
    local_positions: np.ndarray
    source_frame_indices: np.ndarray
    pixels: np.ndarray
    covariances: np.ndarray
    semantic_posterior: dict[str, float]
    valid_window_starts: tuple[int, ...]
    primary_window_start: int


@dataclass(frozen=True)
class RealQ04Inputs:
    solution: FixedCameraHumanSolution
    graph: IntervalMaterialTrackGraph
    seeds: tuple[RealChartSeed, ...]
    frame_paths: tuple[Path, ...]
    phase_degrees_by_frame: dict[int, float]
    input_hashes: dict[str, str]
    audits: list[dict[str, Any]]
    tapir_calibration: dict[str, Any]
    cotracker_calibration: dict[str, Any]
    public_truth_benchmark: dict[str, Any]


WindowPredictions = dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_report(path: Path, *, status: str = "pass") -> dict[str, Any]:
    report = read_json(path)
    if report.get("status") != status:
        raise ValueError(f"required report is not {status}: {path}")
    return report


def _tracker_audits(
    source_audit: dict[str, Any],
    tapir_calibration: dict[str, Any],
    cotracker_calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    if source_audit.get("real_execution_ready") is not True or source_audit.get("blockers"):
        raise ValueError("Q04 tracker source audit is not real-execution ready")
    by_source = {str(item["source"]): item for item in source_audit["sources"]}
    if set(by_source) != {"lk", "tapir", "cotracker3"}:
        raise ValueError("Q04 source audit does not cover all trackers")
    reports = {"tapir": tapir_calibration, "cotracker3": cotracker_calibration}
    audits: list[dict[str, Any]] = []
    for source in ("lk", "tapir", "cotracker3"):
        source_record = by_source[source]
        if source_record.get("license_ready_for_real_use") is not True:
            raise ValueError(f"tracker license is not authorized: {source}")
        calibration = reports.get(source)
        if calibration is not None and (
            calibration.get("status") != "pass"
            or calibration.get("exact_same_device_replay") is not True
            or calibration.get("checkpoint_sha256")
            != source_record.get("checkpoint_observed_sha256")
        ):
            raise ValueError(f"tracker public calibration is incomplete: {source}")
        audits.append(
            {
                "source": source,
                "source_revision": source_record["source_revision"],
                "license": source_record["license"],
                "checkpoint_sha256": (
                    source_record["checkpoint_observed_sha256"] if source != "lk" else None
                ),
                "runtime": (
                    str(calibration["runtime"])
                    if calibration is not None
                    else f"opencv-{source_audit['runtime']['opencv']}"
                ),
                "weights_executed": source != "lk",
                "real_use_authorized": True,
            }
        )
    return audits


def _load_inputs(
    *,
    v00_master_path: Path,
    v00_qualification_path: Path,
    t05_solution_path: Path,
    q03_binding_path: Path,
    q03_report_path: Path,
    source_audit_path: Path,
    tapir_calibration_path: Path,
    cotracker_calibration_path: Path,
    public_graph_path: Path,
) -> RealQ04Inputs:
    paths = [
        v00_master_path,
        v00_qualification_path,
        t05_solution_path,
        q03_binding_path,
        q03_report_path,
        source_audit_path,
        tapir_calibration_path,
        cotracker_calibration_path,
        public_graph_path,
    ]
    reject_sealed_capability(paths)
    master = _require_report(v00_master_path)
    qualification = _require_report(v00_qualification_path)
    q03_report = _require_report(q03_report_path)
    source_audit = _require_report(source_audit_path)
    tapir_calibration = _require_report(tapir_calibration_path)
    cotracker_calibration = _require_report(cotracker_calibration_path)
    public_graph = _require_report(public_graph_path)
    if master.get("experiment_id") != "postv2_v00_capture_forensics_evidence_master_r01":
        raise ValueError("unexpected V00 evidence master")
    if qualification.get("evidence_master_sha256") != sha256_file(v00_master_path):
        raise ValueError("V00 evidence-master hash mismatch")
    if master.get("evidence_policy", {}).get("raw_decoded_frames_authoritative") is not True:
        raise ValueError("V00 decoded frames are not authoritative measured evidence")
    if q03_report.get("binding", {}).get("sha256") != sha256_file(q03_binding_path):
        raise ValueError("Q03 binding hash mismatch")
    if q03_report.get("input_hashes", {}).get("t05_solution") != sha256_file(t05_solution_path):
        raise ValueError("Q03-to-T05 lineage mismatch")
    if public_graph.get("evidence_scope") != "public_synthetic":
        raise ValueError("Q04 public graph has the wrong evidence scope")
    public_truth_benchmark = public_graph.get("public_truth_benchmark")
    if (
        not isinstance(public_truth_benchmark, dict)
        or public_truth_benchmark.get("status") != "pass"
    ):
        raise ValueError("Q04 public graph has no passing measured truth benchmark")

    audits = _tracker_audits(source_audit, tapir_calibration, cotracker_calibration)
    solution = FixedCameraHumanSolution.model_validate(read_json(t05_solution_path))
    if (
        solution.training_frame_count != 144
        or solution.development_images_read != 0
        or solution.development_records_used_for_fit != 0
        or solution.sealed_test_reads != 0
    ):
        raise ValueError("T05 solution does not expose exactly 144 clean training records")
    graph = load_interval_material_track_graph(q03_binding_path)
    state_by_source = {int(frame.source_frame_index): frame for frame in solution.frames}
    if len(state_by_source) != 144:
        raise ValueError("T05 training source indices are not unique")

    master_by_source = {int(item["source_frame_index"]): item for item in master["frames"]}
    evidence_root = v00_master_path.parent
    frame_paths: list[Path] = []
    frame_digests: list[str] = []
    source_indices = [int(frame.source_frame_index) for frame in solution.frames]
    for source_index in source_indices:
        record = master_by_source.get(source_index)
        if record is None or record.get("authoritative_measured_pixels") is not True:
            raise ValueError(f"missing authoritative V00 training frame {source_index}")
        frame_path = evidence_root / str(record["lossless_frame_path"])
        observed = sha256_file(frame_path)
        if observed != record.get("lossless_frame_sha256"):
            raise ValueError(f"V00 frame hash mismatch: {source_index}")
        frame_paths.append(frame_path)
        frame_digests.append(observed)

    local_by_source = {source: index for index, source in enumerate(source_indices)}
    upper_index = SEMANTIC_NAMES.index("upper_clothing")
    seeds: list[RealChartSeed] = []
    for q03_track_index in np.flatnonzero(graph.accepted):
        if int(np.argmax(graph.layer_posterior[q03_track_index])) != upper_index:
            continue
        start = int(graph.track_offsets[q03_track_index])
        stop = int(graph.track_offsets[q03_track_index + 1])
        sources = graph.source_frame_indices[start:stop]
        local_positions = np.asarray([local_by_source[int(value)] for value in sources])
        lo = int(np.min(local_positions))
        hi = int(np.max(local_positions))
        valid_windows = tuple(
            window_start
            for window_start in WINDOW_STARTS
            if window_start <= lo and hi <= window_start + WINDOW_LENGTH - 1
        )
        if not valid_windows:
            raise ValueError(
                f"upper-garment track does not fit a registered chart window: {q03_track_index}"
            )
        midpoint = 0.5 * (lo + hi)
        primary = min(
            valid_windows,
            key=lambda value: abs(value + 0.5 * (WINDOW_LENGTH - 1) - midpoint),
        )
        posterior = {
            "upper_garment": float(graph.layer_posterior[q03_track_index, upper_index]),
            "body_parts": float(
                graph.layer_posterior[q03_track_index, SEMANTIC_NAMES.index("body_parts")]
            ),
            "lower_clothing": float(
                graph.layer_posterior[q03_track_index, SEMANTIC_NAMES.index("lower_clothing")]
            ),
        }
        seeds.append(
            RealChartSeed(
                track_id=f"q03-upper-{q03_track_index:04d}",
                q03_track_index=int(q03_track_index),
                local_positions=local_positions,
                source_frame_indices=sources.astype(np.int64),
                pixels=graph.pixels[start:stop].astype(np.float32),
                covariances=graph.observation_covariance_pixels2[start:stop].astype(np.float32),
                semantic_posterior=posterior,
                valid_window_starts=valid_windows,
                primary_window_start=primary,
            )
        )
    if len(seeds) < 100:
        raise ValueError("Q03 supplies fewer than 100 accepted upper-garment seeds")

    first_phase = float(solution.frames[0].yaw_radians)
    phase = {
        index: float(np.degrees(float(frame.yaw_radians) - first_phase))
        for index, frame in enumerate(solution.frames)
    }
    if min(phase.values()) < -1.0e-6 or max(phase.values()) < 350.0:
        raise ValueError("T05 phase does not preserve the full positive turn")
    frame_set_digest = hashlib.sha256("".join(frame_digests).encode()).hexdigest()
    input_hashes = {
        "v00_evidence_master": sha256_file(v00_master_path),
        "v00_qualification": sha256_file(v00_qualification_path),
        "v00_selected_frame_set": frame_set_digest,
        "t05_solution": sha256_file(t05_solution_path),
        "q03_binding": sha256_file(q03_binding_path),
        "q03_report": sha256_file(q03_report_path),
        "tracker_source_audit": sha256_file(source_audit_path),
        "tapir_public_calibration": sha256_file(tapir_calibration_path),
        "cotracker3_public_calibration": sha256_file(cotracker_calibration_path),
        "public_chart_graph": sha256_file(public_graph_path),
    }
    return RealQ04Inputs(
        solution=solution,
        graph=graph,
        seeds=tuple(seeds),
        frame_paths=tuple(frame_paths),
        phase_degrees_by_frame=phase,
        input_hashes=input_hashes,
        audits=audits,
        tapir_calibration=tapir_calibration,
        cotracker_calibration=cotracker_calibration,
        public_truth_benchmark=public_truth_benchmark,
    )


def _load_rgb_window(frame_paths: tuple[Path, ...], window_start: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    stop = min(window_start + WINDOW_LENGTH, len(frame_paths))
    for path in frame_paths[window_start:stop]:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"failed to decode verified V00 frame: {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if len(frames) != WINDOW_LENGTH:
        raise ValueError("every registered Q04 chart must contain exactly 60 training frames")
    return np.stack(frames)


def _window_seeds(seeds: tuple[RealChartSeed, ...], window_start: int) -> list[RealChartSeed]:
    return [seed for seed in seeds if window_start in seed.valid_window_starts]


def _infer_tapir(
    model: Any,
    video: torch.Tensor,
    query: torch.Tensor,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        output = model(video[None], query[None])
        score = (1.0 - torch.sigmoid(output["occlusion"][0])) * (
            1.0 - torch.sigmoid(output["expected_dist"][0])
        )
    tracks = np.transpose(output["tracks"][0].detach().cpu().numpy(), (1, 0, 2)).copy()
    visibility = np.transpose((score > 0.5).detach().cpu().numpy(), (1, 0)).copy()
    tracks[..., 0] /= scale_x
    tracks[..., 1] /= scale_y
    return tracks, visibility


def _infer_cotracker(
    predictor: Any,
    video: torch.Tensor,
    query: torch.Tensor,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        tracks_tensor, visibility_tensor = predictor(video, queries=query)
    tracks = tracks_tensor[0].detach().cpu().numpy().copy()
    visibility = visibility_tensor[0].detach().cpu().numpy().astype(bool).copy()
    tracks[..., 0] /= scale_x
    tracks[..., 1] /= scale_y
    return tracks, visibility


def _run_tapir(
    inputs: RealQ04Inputs,
    *,
    source_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[WindowPredictions, str]:
    if _git_revision(source_root) != TAPIR_SOURCE_REVISION:
        raise ValueError("TAPIR source revision mismatch")
    if sha256_file(checkpoint_path) != TAPIR_CHECKPOINT_SHA256:
        raise ValueError("TAPIR checkpoint hash mismatch")
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("tapnet.torch.tapir_model")
    module_file = getattr(module, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(source_root.resolve()):
        raise ValueError("TAPIR import did not resolve to the pinned source")
    model = module.TAPIR(pyramid_level=0, extra_convs=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    predictions: WindowPredictions = {}
    digest = hashlib.sha256()
    for window_start in WINDOW_STARTS:
        seeds = _window_seeds(inputs.seeds, window_start)
        frames = _load_rgb_window(inputs.frame_paths, window_start)
        original_height, original_width = frames.shape[1:3]
        model_size = 256
        resized = np.stack(
            [
                cv2.resize(frame, (model_size, model_size), interpolation=cv2.INTER_AREA)
                for frame in frames
            ]
        )
        scale_x = model_size / original_width
        scale_y = model_size / original_height
        queries = np.zeros((len(seeds), 3), dtype=np.float32)
        for index, seed in enumerate(seeds):
            queries[index, 0] = float(seed.local_positions[0] - window_start)
            queries[index, 1] = float(seed.pixels[0, 1] * scale_y)
            queries[index, 2] = float(seed.pixels[0, 0] * scale_x)
        video = torch.from_numpy(resized).to(device=device, dtype=torch.float32)
        video = video / 255.0 * 2.0 - 1.0
        query = torch.from_numpy(queries).to(device=device, dtype=torch.float32)

        tracks, visibility = _infer_tapir(model, video, query, scale_x, scale_y)
        replay_tracks, replay_visibility = _infer_tapir(model, video, query, scale_x, scale_y)
        if not np.array_equal(tracks, replay_tracks) or not np.array_equal(
            visibility, replay_visibility
        ):
            raise RuntimeError(f"TAPIR exact replay failed for chart {window_start}")
        digest.update(window_start.to_bytes(2, "little"))
        digest.update(np.ascontiguousarray(tracks).tobytes())
        digest.update(np.ascontiguousarray(visibility).tobytes())
        for index, seed in enumerate(seeds):
            predictions[(window_start, seed.track_id)] = (
                tracks[:, index].copy(),
                visibility[:, index].copy(),
            )
    return predictions, digest.hexdigest()


def _run_cotracker(
    inputs: RealQ04Inputs,
    *,
    source_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[WindowPredictions, str]:
    if _git_revision(source_root) != COTRACKER_SOURCE_REVISION:
        raise ValueError("CoTracker3 source revision mismatch")
    if sha256_file(checkpoint_path) != COTRACKER_CHECKPOINT_SHA256:
        raise ValueError("CoTracker3 checkpoint hash mismatch")
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("cotracker.predictor")
    module_file = getattr(module, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(source_root.resolve()):
        raise ValueError("CoTracker3 import did not resolve to the pinned source")
    predictor = module.CoTrackerPredictor(
        checkpoint=None,
        offline=True,
        v2=False,
        window_len=60,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    predictor.model.load_state_dict(state)
    predictor = predictor.to(device).eval()
    predictions: WindowPredictions = {}
    digest = hashlib.sha256()
    for window_start in WINDOW_STARTS:
        seeds = _window_seeds(inputs.seeds, window_start)
        frames = _load_rgb_window(inputs.frame_paths, window_start)
        original_height, original_width = frames.shape[1:3]
        input_height = 384
        input_width = round(original_width * input_height / original_height)
        resized = np.stack(
            [
                cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
                for frame in frames
            ]
        )
        scale_x = input_width / original_width
        scale_y = input_height / original_height
        queries = np.zeros((len(seeds), 3), dtype=np.float32)
        for index, seed in enumerate(seeds):
            queries[index, 0] = float(seed.local_positions[0] - window_start)
            queries[index, 1] = float(seed.pixels[0, 0] * scale_x)
            queries[index, 2] = float(seed.pixels[0, 1] * scale_y)
        video = (
            torch.from_numpy(resized)
            .permute(0, 3, 1, 2)[None]
            .to(device=device, dtype=torch.float32)
        )
        query = torch.from_numpy(queries)[None].to(device=device, dtype=torch.float32)

        tracks, visibility = _infer_cotracker(predictor, video, query, scale_x, scale_y)
        replay_tracks, replay_visibility = _infer_cotracker(
            predictor, video, query, scale_x, scale_y
        )
        if not np.array_equal(tracks, replay_tracks) or not np.array_equal(
            visibility, replay_visibility
        ):
            raise RuntimeError(f"CoTracker3 exact replay failed for chart {window_start}")
        digest.update(window_start.to_bytes(2, "little"))
        digest.update(np.ascontiguousarray(tracks).tobytes())
        digest.update(np.ascontiguousarray(visibility).tobytes())
        for index, seed in enumerate(seeds):
            predictions[(window_start, seed.track_id)] = (
                tracks[:, index].copy(),
                visibility[:, index].copy(),
            )
    return predictions, digest.hexdigest()


def _chart_id(window_start: int) -> str:
    return f"chart-{window_start:03d}-{window_start + WINDOW_LENGTH - 1:03d}"


def _proposal_covariance(
    prediction: np.ndarray,
    q03_pixel: np.ndarray,
    calibration_p95: float,
) -> list[list[float]]:
    disagreement = float(np.linalg.norm(prediction - q03_pixel))
    sigma = float(np.clip(max(0.5, calibration_p95, 0.25 * disagreement), 0.5, 5.0))
    variance = sigma * sigma
    return [[variance, 0.0], [0.0, variance]]


def _main_proposals(
    inputs: RealQ04Inputs,
    tapir: WindowPredictions,
    cotracker: WindowPredictions,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    tapir_p95 = float(inputs.tapir_calibration["p95_visible_error_pixels"])
    cotracker_p95 = float(inputs.cotracker_calibration["p95_visible_error_pixels"])
    for seed in inputs.seeds:
        window_start = seed.primary_window_start
        tapir_tracks, tapir_visibility = tapir[(window_start, seed.track_id)]
        cotracker_tracks, cotracker_visibility = cotracker[(window_start, seed.track_id)]
        for observation_index, local_position in enumerate(seed.local_positions):
            source_frame_index = int(seed.source_frame_indices[observation_index])
            common = {
                "track_id": seed.track_id,
                "chart_id": _chart_id(window_start),
                "frame_index": int(local_position),
                "source_frame_index": source_frame_index,
                "semantic_posterior": seed.semantic_posterior,
            }
            proposals.append(
                {
                    **common,
                    "source": "lk",
                    "xy": seed.pixels[observation_index].tolist(),
                    "covariance": seed.covariances[observation_index].tolist(),
                    "visible": True,
                }
            )
            relative = int(local_position - window_start)
            tapir_xy = tapir_tracks[relative]
            if bool(tapir_visibility[relative]):
                proposals.append(
                    {
                        **common,
                        "source": "tapir",
                        "xy": tapir_xy.tolist(),
                        "covariance": _proposal_covariance(
                            tapir_xy, seed.pixels[observation_index], tapir_p95
                        ),
                        "visible": True,
                    }
                )
            cotracker_xy = cotracker_tracks[relative]
            if bool(cotracker_visibility[relative]):
                proposals.append(
                    {
                        **common,
                        "source": "cotracker3",
                        "xy": cotracker_xy.tolist(),
                        "covariance": _proposal_covariance(
                            cotracker_xy, seed.pixels[observation_index], cotracker_p95
                        ),
                        "visible": True,
                    }
                )
    return proposals


def _fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[list[list[float]], float]:
    if len(source) < 3:
        raise ValueError("chart transition requires at least three visible overlap points")
    design = np.column_stack((source, np.ones(len(source))))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    residual = np.linalg.norm(predicted - target, axis=1)
    affine = [
        [float(coefficients[0, 0]), float(coefficients[1, 0]), float(coefficients[2, 0])],
        [float(coefficients[0, 1]), float(coefficients[1, 1]), float(coefficients[2, 1])],
    ]
    return affine, float(np.median(residual))


def _transitions(
    inputs: RealQ04Inputs,
    tapir: WindowPredictions,
    cotracker: WindowPredictions,
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for source_start, target_start in pairwise(WINDOW_STARTS):
        overlap = [
            seed
            for seed in inputs.seeds
            if source_start in seed.valid_window_starts and target_start in seed.valid_window_starts
        ]
        source_points: list[np.ndarray] = []
        target_points: list[np.ndarray] = []
        for seed in overlap:
            for predictions in (tapir, cotracker):
                source_tracks, source_visibility = predictions[(source_start, seed.track_id)]
                target_tracks, target_visibility = predictions[(target_start, seed.track_id)]
                for local_position in seed.local_positions:
                    source_offset = int(local_position - source_start)
                    target_offset = int(local_position - target_start)
                    if bool(source_visibility[source_offset]) and bool(
                        target_visibility[target_offset]
                    ):
                        source_points.append(source_tracks[source_offset])
                        target_points.append(target_tracks[target_offset])
        source_array = np.asarray(source_points, dtype=np.float64)
        target_array = np.asarray(target_points, dtype=np.float64)
        try:
            affine, residual = _fit_affine(source_array, target_array)
        except ValueError as exc:
            raise ValueError(
                f"unobservable chart transition {source_start}->{target_start}: {exc}"
            ) from exc
        transitions.append(
            {
                "source_chart_id": _chart_id(source_start),
                "target_chart_id": _chart_id(target_start),
                "overlap_track_ids": [seed.track_id for seed in overlap],
                "affine_map": affine,
                "cycle_residual_pixels": residual,
            }
        )
    return transitions


def _track_anchor_medians(
    graph: MaterialChartGraph,
    inputs: RealQ04Inputs,
) -> list[float]:
    state_by_local = {index: state for index, state in enumerate(inputs.solution.frames)}
    intrinsics = np.asarray(inputs.solution.shared_intrinsics, dtype=np.float64)
    medians: list[float] = []
    for track in graph.tracks:
        if not track.accepted:
            continue
        states = [state_by_local[item.frame_index] for item in track.observations]
        rotations = Rotation.from_rotvec(
            np.asarray([state.observed_global_orient_rotvec for state in states])
        ).as_matrix()
        translations = np.asarray([state.root_translation_metres for state in states])
        pixels = np.asarray([item.xy for item in track.observations], dtype=np.float64)
        covariance = np.asarray([item.covariance for item in track.observations])
        weights = 1.0 / np.maximum(np.trace(covariance, axis1=1, axis2=2), 1.0e-6)
        fit = robust_material_anchor(rotations, translations, pixels, intrinsics, weights)
        medians.append(float(np.median(fit.reprojection_error_pixels)))
    return medians


def _q03_upper_anchor_medians(inputs: RealQ04Inputs) -> list[float]:
    values: list[float] = []
    for seed in inputs.seeds:
        start = int(inputs.graph.track_offsets[seed.q03_track_index])
        stop = int(inputs.graph.track_offsets[seed.q03_track_index + 1])
        values.append(float(np.median(inputs.graph.anchor_reprojection_error_pixels[start:stop])))
    return values


def build_real_material_chart_graph(
    *,
    v00_master_path: Path,
    v00_qualification_path: Path,
    t05_solution_path: Path,
    q03_binding_path: Path,
    q03_report_path: Path,
    source_audit_path: Path,
    tapir_calibration_path: Path,
    cotracker_calibration_path: Path,
    public_graph_path: Path,
    tapir_source_root: Path,
    tapir_checkpoint_path: Path,
    cotracker_source_root: Path,
    cotracker_checkpoint_path: Path,
    device_name: str = "cpu",
) -> MaterialChartGraph:
    """Construct the registered real Q04 graph from training evidence only."""
    inputs = _load_inputs(
        v00_master_path=v00_master_path,
        v00_qualification_path=v00_qualification_path,
        t05_solution_path=t05_solution_path,
        q03_binding_path=q03_binding_path,
        q03_report_path=q03_report_path,
        source_audit_path=source_audit_path,
        tapir_calibration_path=tapir_calibration_path,
        cotracker_calibration_path=cotracker_calibration_path,
        public_graph_path=public_graph_path,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    tapir, tapir_digest = _run_tapir(
        inputs,
        source_root=tapir_source_root,
        checkpoint_path=tapir_checkpoint_path,
        device=device,
    )
    cotracker, cotracker_digest = _run_cotracker(
        inputs,
        source_root=cotracker_source_root,
        checkpoint_path=cotracker_checkpoint_path,
        device=device,
    )
    proposals = _main_proposals(inputs, tapir, cotracker)
    counts = {
        source: sum(item["source"] == source for item in proposals)
        for source in ("lk", "tapir", "cotracker3")
    }
    payload: dict[str, Any] = {
        "evidence_scope": "train_real",
        "tracker_audits": inputs.audits,
        "proposals": proposals,
        "phase_degrees_by_frame": inputs.phase_degrees_by_frame,
        "transitions": _transitions(inputs, tapir, cotracker),
        "anchor_reprojection_pixels": [0.0],
        "public_truth_benchmark": inputs.public_truth_benchmark,
        "corrupted_proposal_capacity_regression": 0.0,
        "minimum_proposal_sources_per_observation": 2,
        "input_hashes": inputs.input_hashes,
        "source_frame_indices": [int(frame.source_frame_index) for frame in inputs.solution.frames],
        "proposal_count_by_source": counts,
        "model_output_sha256_by_source": {
            "tapir": tapir_digest,
            "cotracker3": cotracker_digest,
        },
        "exact_same_device_replay_by_source": {"tapir": True, "cotracker3": True},
        "training_records_read": 144,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
    }
    preliminary = build_material_chart_graph(payload)
    anchor_medians = _track_anchor_medians(preliminary, inputs)
    if not anchor_medians:
        raise RuntimeError("real Q04 produced no anchor-qualified upper-garment tracks")
    q03_medians = _q03_upper_anchor_medians(inputs)
    q03_baseline = float(np.median(q03_medians))
    treatment = float(np.median(anchor_medians))
    improvement = (q03_baseline - treatment) / max(q03_baseline, 1.0e-9)
    payload["anchor_reprojection_pixels"] = anchor_medians
    payload["q03_anchor_reprojection_improvement"] = improvement

    corrupted_payload = copy.deepcopy(payload)
    for proposal in corrupted_payload["proposals"]:
        if proposal["source"] == "cotracker3" and int(proposal["track_id"][-4:]) % 5 == 0:
            proposal["xy"][0] += 50.0
            proposal["xy"][1] -= 50.0
    clean = build_material_chart_graph(payload)
    corrupted = build_material_chart_graph(corrupted_payload)
    clean_capacity = sum(track.accepted for track in clean.tracks)
    corrupt_capacity = sum(track.accepted for track in corrupted.tracks)
    payload["corrupted_proposal_capacity_regression"] = float(
        max(clean_capacity - corrupt_capacity, 0)
    )
    return build_material_chart_graph(payload)


def real_q04_summary(graph: MaterialChartGraph) -> dict[str, Any]:
    """Produce a compact factual summary without adding a second artifact authority."""
    return {
        "schema_version": "frayid_v3_q04_real_chart_summary.v1",
        "status": graph.status,
        "promotion_eligible": graph.promotion_eligible,
        "accepted_track_count": sum(track.accepted for track in graph.tracks),
        "rejected_track_count": sum(not track.accepted for track in graph.tracks),
        "chart_count": len(graph.charts),
        "transition_count": len(graph.transitions),
        "phase_bins_spanned": graph.phase_bins_spanned,
        "median_cycle_residual_pixels": graph.median_cycle_residual_pixels,
        "p95_cycle_residual_pixels": graph.p95_cycle_residual_pixels,
        "median_anchor_reprojection_pixels": graph.median_anchor_reprojection_pixels,
        "p95_anchor_reprojection_pixels": graph.p95_anchor_reprojection_pixels,
        "q03_anchor_reprojection_improvement": graph.q03_anchor_reprojection_improvement,
        "proposal_count_by_source": graph.proposal_count_by_source,
        "model_output_sha256_by_source": graph.model_output_sha256_by_source,
        "exact_replay_hash": graph.exact_replay_hash,
        "blockers": graph.blockers,
        "training_records_read": graph.training_records_read,
        "development_records_read": graph.development_records_read,
        "sealed_test_accesses": graph.sealed_test_accesses,
        "input_hashes": graph.input_hashes,
    }
