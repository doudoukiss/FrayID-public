from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np

from frayid.v3.schemas import (
    ChartObservation,
    ChartTransition,
    MaterialChartGraph,
    MaterialTrack,
    PublicChartTruthBenchmark,
    TrackSourceAudit,
)

EXPERIMENT_ID = "postv3_q04_local_material_chart_graph_r01"
SUPPORTED_EXPERIMENT_IDS = {
    EXPERIMENT_ID,
    "postv3_q05_controlled_material_chart_graph_r01",
}
SOURCES = ("lk", "tapir", "cotracker3")


def _replay_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _triangulate_rotating_points(
    observations: np.ndarray,
    projection_matrices: np.ndarray,
    fit_phase_indices: np.ndarray,
) -> np.ndarray:
    """Recover canonical points from fixed-camera, rotating-object observations."""
    result = np.empty((observations.shape[1], 3), dtype=np.float64)
    for point_index in range(observations.shape[1]):
        rows: list[np.ndarray] = []
        for phase_index in fit_phase_indices:
            projection = projection_matrices[phase_index]
            x, y = observations[phase_index, point_index]
            rows.extend((x * projection[2] - projection[0], y * projection[2] - projection[1]))
        _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        homogeneous = vh[-1]
        result[point_index] = homogeneous[:3] / homogeneous[3]
    return result


def _truth_reprojection_errors(
    points: np.ndarray,
    truth_pixels: np.ndarray,
    projection_matrices: np.ndarray,
    evaluator_phase_indices: np.ndarray,
) -> np.ndarray:
    errors: list[np.ndarray] = []
    homogeneous_points = np.column_stack((points, np.ones(len(points))))
    for phase_index in evaluator_phase_indices:
        projected = (projection_matrices[phase_index] @ homogeneous_points.T).T
        pixels = projected[:, :2] / projected[:, 2, None]
        errors.append(np.linalg.norm(pixels - truth_pixels[phase_index], axis=1))
    return np.concatenate(errors)


def public_chart_truth_benchmark() -> dict[str, Any]:
    """Measure chart consensus against public non-radial 3D and held-phase truth."""
    rng = np.random.default_rng(904)
    grid_x, grid_y = np.meshgrid(np.linspace(-0.55, 0.55, 12), np.linspace(-0.75, 0.75, 10))
    x = grid_x.reshape(-1)
    y = grid_y.reshape(-1)
    z = 0.12 * np.sin(4.0 * x) * np.cos(3.0 * y) + 0.04 * x * y
    truth_points = np.column_stack((x, y, z))
    intrinsics = np.asarray([[240.0, 0.0, 128.0], [0.0, 240.0, 128.0], [0.0, 0.0, 1.0]])
    projections: list[np.ndarray] = []
    truth_pixels: list[np.ndarray] = []
    for phase_index in range(12):
        angle = np.deg2rad(30.0 * phase_index)
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        projection = intrinsics @ np.column_stack((rotation, np.asarray([0.0, 0.0, 3.0])))
        homogeneous = projection @ np.column_stack((truth_points, np.ones(len(truth_points)))).T
        projections.append(projection)
        truth_pixels.append((homogeneous[:2] / homogeneous[2]).T)
    projection_array = np.asarray(projections)
    truth_pixel_array = np.asarray(truth_pixels)

    phase = np.arange(12, dtype=np.float64)[:, None]
    point = np.arange(len(truth_points), dtype=np.float64)[None, :]
    lk_bias = np.stack(
        (
            0.22 * phase + 0.35 * np.sin(0.4 * point),
            -0.13 * phase + 0.25 * np.cos(0.3 * point),
        ),
        axis=-1,
    )
    lk = truth_pixel_array + lk_bias + rng.normal(0.0, 0.35, truth_pixel_array.shape)
    tapir = truth_pixel_array + rng.normal(0.0, 0.32, truth_pixel_array.shape)
    cotracker = truth_pixel_array + rng.normal(0.0, 0.28, truth_pixel_array.shape)
    corrupted = (np.arange(12)[:, None] + np.arange(len(truth_points))[None, :]) % 17 == 0
    cotracker[corrupted] += np.asarray([9.0, -7.0])

    proposals = np.stack((lk, tapir, cotracker), axis=2)
    center = np.median(proposals, axis=2)
    distances = np.linalg.norm(proposals - center[:, :, None, :], axis=-1)
    accepted = distances <= 5.0
    inverse_variances = np.asarray([1.0 / 0.35**2, 1.0 / 0.32**2, 1.0 / 0.28**2])
    weights = accepted * inverse_variances[None, None, :]
    consensus = np.sum(proposals * weights[..., None], axis=2) / np.sum(weights, axis=2)[..., None]

    fit_indices = np.asarray([0, 2, 4, 6, 8, 10])
    evaluator_indices = np.asarray([1, 3, 5, 7, 9, 11])
    control_points = _triangulate_rotating_points(lk, projection_array, fit_indices)
    ensemble_points = _triangulate_rotating_points(consensus, projection_array, fit_indices)
    control_surface = np.linalg.norm(control_points - truth_points, axis=1)
    ensemble_surface = np.linalg.norm(ensemble_points - truth_points, axis=1)
    control_reprojection = _truth_reprojection_errors(
        control_points, truth_pixel_array, projection_array, evaluator_indices
    )
    ensemble_reprojection = _truth_reprojection_errors(
        ensemble_points, truth_pixel_array, projection_array, evaluator_indices
    )
    control_surface_median = float(np.median(control_surface))
    ensemble_surface_median = float(np.median(ensemble_surface))
    control_reprojection_median = float(np.median(control_reprojection))
    ensemble_reprojection_median = float(np.median(ensemble_reprojection))
    geometry_improvement = (
        control_surface_median - ensemble_surface_median
    ) / control_surface_median
    reprojection_improvement = (
        control_reprojection_median - ensemble_reprojection_median
    ) / control_reprojection_median
    blockers: list[str] = []
    if geometry_improvement < 0.2:
        blockers.append("public_geometry_improvement_below_20_percent")
    if reprojection_improvement < 0.2:
        blockers.append("public_reprojection_improvement_below_20_percent")
    digest = hashlib.sha256()
    for value in (
        truth_points,
        truth_pixel_array,
        lk,
        tapir,
        cotracker,
        consensus,
        control_points,
        ensemble_points,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return {
        "point_count": len(truth_points),
        "phase_count": 12,
        "fit_phase_indices": fit_indices.tolist(),
        "evaluator_phase_indices": evaluator_indices.tolist(),
        "control_median_surface_error": control_surface_median,
        "ensemble_median_surface_error": ensemble_surface_median,
        "geometry_improvement": geometry_improvement,
        "control_median_reprojection_error_pixels": control_reprojection_median,
        "ensemble_median_reprojection_error_pixels": ensemble_reprojection_median,
        "reprojection_improvement": reprojection_improvement,
        "nonradial_surface": True,
        "corrupted_source_present": bool(np.any(corrupted)),
        "exact_replay_hash": digest.hexdigest(),
        "project_evidence_reads": 0,
        "sealed_test_accesses": 0,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


def _visibility_intervals(frame_indices: list[int]) -> list[tuple[int, int]]:
    if not frame_indices:
        return []
    values = sorted(set(frame_indices))
    intervals: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            intervals.append((start, previous))
            start = value
        previous = value
    intervals.append((start, previous))
    return intervals


def _validate_audits(raw: list[dict[str, Any]]) -> list[TrackSourceAudit]:
    audits = [TrackSourceAudit.model_validate(item) for item in raw]
    by_source = {item.source: item for item in audits}
    if set(by_source) != set(SOURCES):
        raise ValueError("Q04 requires audited LK, TAPIR, and CoTracker3 proposal sources")
    return audits


def build_material_chart_graph(payload: dict[str, Any]) -> MaterialChartGraph:
    """Merge tracker proposals without granting any tracker material truth."""
    experiment_id = str(payload.get("experiment_id", EXPERIMENT_ID))
    if experiment_id not in SUPPORTED_EXPERIMENT_IDS:
        raise ValueError(f"unsupported material-chart experiment: {experiment_id}")
    audits_raw = payload.get("tracker_audits")
    proposals_raw = payload.get("proposals")
    phase_by_frame_raw = payload.get("phase_degrees_by_frame")
    if not isinstance(audits_raw, list) or not isinstance(proposals_raw, list):
        raise ValueError("tracker_audits and proposals must be lists")
    if not isinstance(phase_by_frame_raw, dict):
        raise ValueError("phase_degrees_by_frame must be an object")
    audits = _validate_audits(audits_raw)
    evidence_scope = str(payload.get("evidence_scope", "public_synthetic"))
    benchmark = PublicChartTruthBenchmark.model_validate(payload.get("public_truth_benchmark"))
    if evidence_scope == "train_real" and any(
        item.source != "lk" and (not item.weights_executed or not item.real_use_authorized)
        for item in audits
    ):
        raise ValueError(
            "real Q04 use requires executed, hashed, license-authorized tracker audits"
        )
    phase_by_frame = {int(key): float(value) for key, value in phase_by_frame_raw.items()}
    minimum_sources = int(payload.get("minimum_proposal_sources_per_observation", 1))
    if minimum_sources < 1 or minimum_sources > len(SOURCES):
        raise ValueError("minimum proposal-source count must lie in [1,3]")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    chart_votes: dict[str, list[str]] = defaultdict(list)
    semantic: dict[str, dict[str, float]] = {}
    for raw in proposals_raw:
        if not isinstance(raw, dict):
            raise ValueError("each proposal must be an object")
        source = str(raw["source"])
        if source not in SOURCES:
            raise ValueError(f"unknown proposal source: {source}")
        track_id = str(raw["track_id"])
        frame_index = int(raw["frame_index"])
        grouped[(track_id, frame_index)].append(raw)
        chart_votes[track_id].append(str(raw["chart_id"]))
        posterior = raw.get("semantic_posterior", {"upper_garment": 1.0})
        if not isinstance(posterior, dict):
            raise ValueError("semantic_posterior must be an object")
        semantic[track_id] = {str(key): float(value) for key, value in posterior.items()}

    observations_by_track: dict[str, list[ChartObservation]] = defaultdict(list)
    consensus_residuals: list[float] = []
    for (track_id, frame_index), group in sorted(grouped.items()):
        xy = np.asarray([item["xy"] for item in group], dtype=np.float64)
        center = np.median(xy, axis=0)
        distances = np.linalg.norm(xy - center[None, :], axis=1)
        accepted_indices = np.flatnonzero(distances <= 5.0)
        if accepted_indices.size == 0:
            continue
        accepted_source_names = {str(group[int(index)]["source"]) for index in accepted_indices}
        if len(accepted_source_names) < minimum_sources:
            continue
        precisions: list[np.ndarray] = []
        weighted: list[np.ndarray] = []
        sources: list[str] = []
        for index in accepted_indices:
            item = group[int(index)]
            covariance = np.asarray(item.get("covariance", [[1.0, 0.0], [0.0, 1.0]]))
            precision = np.linalg.inv(covariance)
            precisions.append(precision)
            weighted.append(precision @ xy[int(index)])
            sources.append(str(item["source"]))
        combined_precision = np.sum(precisions, axis=0)
        covariance = np.linalg.inv(combined_precision)
        combined_xy = covariance @ np.sum(weighted, axis=0)
        consensus_residuals.extend(float(value) for value in distances[accepted_indices])
        observations_by_track[track_id].append(
            ChartObservation(
                frame_index=frame_index,
                source_frame_index=(
                    int(group[int(accepted_indices[0])]["source_frame_index"])
                    if group[int(accepted_indices[0])].get("source_frame_index") is not None
                    else None
                ),
                xy=(float(combined_xy[0]), float(combined_xy[1])),
                covariance=(
                    (float(covariance[0, 0]), float(covariance[0, 1])),
                    (float(covariance[1, 0]), float(covariance[1, 1])),
                ),
                visible=any(
                    bool(group[int(index)].get("visible", True)) for index in accepted_indices
                ),
                proposal_sources=sorted(set(sources)),  # type: ignore[arg-type]
            )
        )

    tracks: list[MaterialTrack] = []
    bins: set[int] = set()
    for track_id, observations in sorted(observations_by_track.items()):
        visible_frames = [item.frame_index for item in observations if item.visible]
        phases = [phase_by_frame[index] for index in visible_frames if index in phase_by_frame]
        phase_span = max(phases) - min(phases) if phases else 0.0
        rejection: list[str] = []
        if phase_span < 10.0 or phase_span > 90.0:
            rejection.append("visibility_interval_outside_10_90_degrees")
        if len(visible_frames) < 2:
            rejection.append("insufficient_visible_observations")
        for phase in phases:
            bins.add(int(np.floor((phase % 360.0) / 30.0)))
        votes = chart_votes[track_id]
        chart_id = max(sorted(set(votes)), key=votes.count)
        tracks.append(
            MaterialTrack(
                track_id=track_id,
                chart_id=chart_id,
                semantic_posterior=semantic[track_id],
                visibility_intervals=_visibility_intervals(visible_frames),
                observations=observations,
                accepted=not rejection,
                rejection_reasons=rejection,
            )
        )

    transitions = [ChartTransition.model_validate(item) for item in payload.get("transitions", [])]
    cycle_values = [item.cycle_residual_pixels for item in transitions]
    if not cycle_values:
        cycle_values = consensus_residuals or [float("inf")]
    anchor_values = np.asarray(payload.get("anchor_reprojection_pixels", consensus_residuals))
    if anchor_values.size == 0:
        anchor_values = np.asarray([float("inf")])
    cycle = np.asarray(cycle_values, dtype=np.float64)
    accepted_count = sum(item.accepted for item in tracks)
    public_geometry_improvement = benchmark.geometry_improvement
    public_reprojection_improvement = benchmark.reprojection_improvement
    corrupted_regression = float(payload.get("corrupted_proposal_capacity_regression", 0.0))
    blockers: list[str] = []
    if accepted_count < 100:
        blockers.append("accepted_upper_garment_tracks_below_100")
    if len(bins) < 10:
        blockers.append("chart_connectivity_below_10_phase_bins")
    if float(np.median(cycle)) > 2.0 or float(np.percentile(cycle, 95)) > 5.0:
        blockers.append("cycle_residual_gate")
    if float(np.median(anchor_values)) > 2.5 or float(np.percentile(anchor_values, 95)) > 7.22:
        blockers.append("anchor_reprojection_gate")
    if public_geometry_improvement < 0.2 or public_reprojection_improvement < 0.2:
        blockers.append("public_truth_improvement_below_20_percent")
    if corrupted_regression > 0.0:
        blockers.append("corrupted_proposal_capacity_regression")

    replay_payload = {
        "tracks": [item.model_dump(mode="json") for item in tracks],
        "transitions": [item.model_dump(mode="json") for item in transitions],
        "phase_bins": sorted(bins),
    }
    chart_ids = {item.chart_id for item in tracks}
    for transition in transitions:
        chart_ids.add(transition.source_chart_id)
        chart_ids.add(transition.target_chart_id)
    return MaterialChartGraph(
        experiment_id=experiment_id,
        evidence_scope=evidence_scope,  # type: ignore[arg-type]
        promotion_eligible=not blockers and evidence_scope == "train_real",
        tracker_audits=audits,
        input_hashes={
            str(key): str(value) for key, value in payload.get("input_hashes", {}).items()
        },
        source_frame_indices=[int(value) for value in payload.get("source_frame_indices", [])],
        proposal_count_by_source={
            str(key): int(value)
            for key, value in payload.get("proposal_count_by_source", {}).items()
        },
        model_output_sha256_by_source={
            str(key): str(value)
            for key, value in payload.get("model_output_sha256_by_source", {}).items()
        },
        exact_same_device_replay_by_source={
            str(key): bool(value)
            for key, value in payload.get("exact_same_device_replay_by_source", {}).items()
        },
        training_records_read=int(payload.get("training_records_read", 0)),
        development_records_read=int(payload.get("development_records_read", 0)),
        sealed_test_accesses=int(payload.get("sealed_test_accesses", 0)),
        public_truth_geometry_improvement=public_geometry_improvement,
        public_truth_reprojection_improvement=public_reprojection_improvement,
        public_truth_benchmark=benchmark,
        q03_anchor_reprojection_improvement=(
            float(payload["q03_anchor_reprojection_improvement"])
            if payload.get("q03_anchor_reprojection_improvement") is not None
            else None
        ),
        charts=sorted(chart_ids),
        tracks=tracks,
        transitions=transitions,
        phase_bins_spanned=sorted(bins),
        median_cycle_residual_pixels=float(np.median(cycle)),
        p95_cycle_residual_pixels=float(np.percentile(cycle, 95)),
        median_anchor_reprojection_pixels=float(np.median(anchor_values)),
        p95_anchor_reprojection_pixels=float(np.percentile(anchor_values, 95)),
        corrupted_proposal_capacity_regression=corrupted_regression,
        exact_replay_hash=_replay_hash(replay_payload),
        status="pass" if not blockers else "fail",
        blockers=blockers,
    )


def public_chart_fixture(*, corrupt: bool = False) -> dict[str, Any]:
    """Deterministic 12-bin chart fixture with one optional adversarial source."""
    rng = np.random.default_rng(17)
    audits = [
        {
            "source": "lk",
            "source_revision": "opencv-public-api",
            "license": "Apache-2.0",
            "checkpoint_sha256": None,
            "runtime": "cpu",
            "weights_executed": False,
            "real_use_authorized": True,
        },
        {
            "source": "tapir",
            "source_revision": "public-synthetic-stub-no-weight-execution",
            "license": "Apache-2.0",
            "checkpoint_sha256": "1" * 64,
            "runtime": "proposal-fixture",
            "weights_executed": False,
            "real_use_authorized": False,
        },
        {
            "source": "cotracker3",
            "source_revision": "public-synthetic-stub-no-weight-execution",
            "license": "CC-BY-NC-4.0-review-required-before-real-use",
            "checkpoint_sha256": "2" * 64,
            "runtime": "proposal-fixture",
            "weights_executed": False,
            "real_use_authorized": False,
        },
    ]
    proposals: list[dict[str, Any]] = []
    phase = {index: float(index * 10) for index in range(36)}
    for track_index in range(120):
        start = track_index % 33
        chart = f"chart-{track_index % 12:02d}"
        for offset in range(4):
            frame = start + offset
            truth = np.array([40.0 + track_index * 0.3 + offset, 70.0 + track_index * 0.1])
            for source_index, source in enumerate(SOURCES):
                noise = rng.normal(0.0, 0.15, 2)
                if corrupt and source == "cotracker3" and track_index % 5 == 0:
                    noise += 50.0
                proposals.append(
                    {
                        "track_id": f"track-{track_index:03d}",
                        "chart_id": chart,
                        "frame_index": frame,
                        "xy": (truth + noise).tolist(),
                        "covariance": [[0.25 + source_index * 0.1, 0.0], [0.0, 0.25]],
                        "visible": True,
                        "source": source,
                        "semantic_posterior": {"upper_garment": 0.99},
                    }
                )
    transitions = [
        {
            "source_chart_id": f"chart-{index:02d}",
            "target_chart_id": f"chart-{(index + 1) % 12:02d}",
            "overlap_track_ids": [f"track-{index:03d}"],
            "affine_map": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "cycle_residual_pixels": 0.5,
        }
        for index in range(12)
    ]
    benchmark = public_chart_truth_benchmark()
    return {
        "tracker_audits": audits,
        "proposals": proposals,
        "phase_degrees_by_frame": phase,
        "transitions": transitions,
        "anchor_reprojection_pixels": [0.5] * 120,
        "public_truth_benchmark": benchmark,
        "corrupted_proposal_capacity_regression": 0.0,
    }
