from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frayid.dataset import read_dataset_manifest
from frayid.io import read_json, sha256_file, write_json
from frayid.schemas import FrameRecord
from frayid.v2.contracts import reject_sealed_capability


@dataclass(frozen=True)
class CorrespondenceGate:
    """Frozen engineering thresholds for the pre-T01 observability scan."""

    local_median_inliers: int = 40
    local_median_inlier_ratio: float = 0.25
    local_median_coverage: float = 0.05
    quarter_median_inliers: int = 12
    quarter_median_inlier_ratio: float = 0.15
    quarter_median_coverage: float = 0.025
    loop_median_inliers: int = 20
    loop_median_inlier_ratio: float = 0.20
    loop_median_coverage: float = 0.03

    def as_dict(self) -> dict[str, int | float]:
        return {
            "local_median_inliers": self.local_median_inliers,
            "local_median_inlier_ratio": self.local_median_inlier_ratio,
            "local_median_coverage": self.local_median_coverage,
            "quarter_median_inliers": self.quarter_median_inliers,
            "quarter_median_inlier_ratio": self.quarter_median_inlier_ratio,
            "quarter_median_coverage": self.quarter_median_coverage,
            "loop_median_inliers": self.loop_median_inliers,
            "loop_median_inlier_ratio": self.loop_median_inlier_ratio,
            "loop_median_coverage": self.loop_median_coverage,
        }


@dataclass(frozen=True)
class TemporalGraphGate:
    """Frozen thresholds for a rotating, visibility-limited tracklet factor graph."""

    minimum_edge_inliers: int = 30
    minimum_edge_inlier_ratio: float = 0.25
    minimum_edge_coverage: float = 0.03
    minimum_passing_edge_fraction: float = 0.90
    minimum_largest_component_fraction: float = 0.95
    minimum_loop_inliers: int = 20
    minimum_loop_inlier_ratio: float = 0.20
    minimum_loop_coverage: float = 0.03

    def as_dict(self) -> dict[str, int | float]:
        return {
            "minimum_edge_inliers": self.minimum_edge_inliers,
            "minimum_edge_inlier_ratio": self.minimum_edge_inlier_ratio,
            "minimum_edge_coverage": self.minimum_edge_coverage,
            "minimum_passing_edge_fraction": self.minimum_passing_edge_fraction,
            "minimum_largest_component_fraction": self.minimum_largest_component_fraction,
            "minimum_loop_inliers": self.minimum_loop_inliers,
            "minimum_loop_inlier_ratio": self.minimum_loop_inlier_ratio,
            "minimum_loop_coverage": self.minimum_loop_coverage,
        }


@dataclass(frozen=True)
class _BoundFrame:
    record: FrameRecord
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class _Features:
    points: np.ndarray
    descriptors: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    scale: float
    foreground_area: float


@dataclass(frozen=True)
class _PairEvaluation:
    metrics: dict[str, int | float | str | None]
    inlier_pairs: list[tuple[int, int]]


PAIR_BINS: dict[str, tuple[int, int]] = {
    "local": (1, 3),
    "short_arc": (12, 18),
    "quarter_turn": (42, 48),
    "half_turn": (87, 93),
    "loop_closure": (174, 179),
}


def _resolve_record_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in manifest_path.resolve().parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"unable to resolve dataset record path: {value}")


def _bind_training_frames(manifest_path: Path) -> tuple[list[_BoundFrame], int]:
    manifest = read_dataset_manifest(manifest_path)
    held_out_records = sum(frame.split == "held_out" for frame in manifest.frames)
    frames: list[_BoundFrame] = []
    for record in sorted(manifest.frames, key=lambda item: item.ordinal):
        if record.split != "train" or not record.quality_accepted:
            continue
        image_path = _resolve_record_path(manifest_path, record.image_path)
        mask_path = manifest_path.parent / "masks" / image_path.name
        reject_sealed_capability([image_path, mask_path])
        if not mask_path.is_file():
            raise FileNotFoundError(f"training mask is absent: {mask_path}")
        frames.append(_BoundFrame(record=record, image_path=image_path, mask_path=mask_path))
    if len(frames) < 3:
        raise ValueError("correspondence scan requires at least three accepted training frames")
    if any(frame.record.split != "train" for frame in frames):
        raise AssertionError("held-out record entered the correspondence binding")
    return frames, held_out_records


def _pair_schedule(
    frames: list[_BoundFrame],
    *,
    maximum_pairs_per_bin: int,
) -> dict[str, list[tuple[_BoundFrame, _BoundFrame]]]:
    if maximum_pairs_per_bin < 1:
        raise ValueError("maximum_pairs_per_bin must be positive")
    result: dict[str, list[tuple[_BoundFrame, _BoundFrame]]] = {}
    for name, (minimum_delta, maximum_delta) in PAIR_BINS.items():
        candidates = [
            (first, second)
            for index, first in enumerate(frames)
            for second in frames[index + 1 :]
            if minimum_delta <= second.record.ordinal - first.record.ordinal <= maximum_delta
        ]
        if len(candidates) <= maximum_pairs_per_bin:
            result[name] = candidates
            continue
        selected_indices = np.linspace(
            0,
            len(candidates) - 1,
            maximum_pairs_per_bin,
            dtype=np.int64,
        )
        result[name] = [candidates[int(index)] for index in selected_indices]
    return result


def _read_features(
    frame: _BoundFrame,
    *,
    detector: Any,
    maximum_dimension: int,
) -> _Features:
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(frame.mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"unable to read bound training evidence: {frame.image_path}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"training image/mask dimensions differ: {frame.image_path}")
    height, width = mask.shape
    scale = min(1.0, maximum_dimension / float(max(height, width)))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
    kernel_size = max(3, round(min(mask.shape) * 0.01) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, eroded)
    if descriptors is None or not keypoints:
        descriptor_size = int(detector.descriptorSize())
        descriptors = np.empty((0, descriptor_size), dtype=np.float32)
        points = np.empty((0, 2), dtype=np.float32)
    else:
        descriptors = np.asarray(descriptors)
        points = np.asarray([point.pt for point in keypoints], dtype=np.float32)
    foreground_area = float(np.count_nonzero(eroded))
    return _Features(
        points=points,
        descriptors=descriptors,
        gray=gray,
        mask=eroded,
        scale=scale,
        foreground_area=foreground_area,
    )


def _ratio_matches(first: np.ndarray, second: np.ndarray, *, norm: int) -> list[Any]:
    if len(first) < 2 or len(second) < 2:
        return []
    matcher = cv2.BFMatcher(normType=norm, crossCheck=False)
    candidates = matcher.knnMatch(first, second, k=2)
    return [
        pair[0]
        for pair in candidates
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]


def _mutual_matches(first: _Features, second: _Features, *, norm: int) -> list[tuple[int, int]]:
    forward = _ratio_matches(first.descriptors, second.descriptors, norm=norm)
    backward = _ratio_matches(second.descriptors, first.descriptors, norm=norm)
    reverse = {(int(match.queryIdx), int(match.trainIdx)) for match in backward}
    return [
        (int(match.queryIdx), int(match.trainIdx))
        for match in forward
        if (int(match.trainIdx), int(match.queryIdx)) in reverse
    ]


def _coverage(points: np.ndarray, foreground_area: float) -> float:
    if len(points) < 3 or foreground_area <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(np.clip(cv2.contourArea(hull) / foreground_area, 0.0, 1.0))


def _patch_ncc(first: _Features, second: _Features, pairs: list[tuple[int, int]]) -> float | None:
    values: list[float] = []
    for first_index, second_index in pairs:
        first_patch = cv2.getRectSubPix(
            first.gray,
            (9, 9),
            tuple(float(value) for value in first.points[first_index]),
        ).astype(np.float32)
        second_patch = cv2.getRectSubPix(
            second.gray,
            (9, 9),
            tuple(float(value) for value in second.points[second_index]),
        ).astype(np.float32)
        first_patch -= float(first_patch.mean())
        second_patch -= float(second_patch.mean())
        denominator = float(np.linalg.norm(first_patch) * np.linalg.norm(second_patch))
        if denominator > 1.0e-8:
            values.append(float(np.sum(first_patch * second_patch) / denominator))
    return float(np.median(values)) if values else None


def _evaluate_pair_details(
    first: _Features,
    second: _Features,
    *,
    descriptor_norm: int,
    ransac_seed: int,
) -> _PairEvaluation:
    pairs = _mutual_matches(first, second, norm=descriptor_norm)
    fundamental_selected = np.zeros((len(pairs),), dtype=bool)
    homography_selected = np.zeros((len(pairs),), dtype=bool)
    if len(pairs) >= 8:
        first_points = np.asarray([first.points[item[0]] for item in pairs], dtype=np.float32)
        second_points = np.asarray([second.points[item[1]] for item in pairs], dtype=np.float32)
        cv2.setRNGSeed(ransac_seed)
        _, mask = cv2.findFundamentalMat(
            first_points,
            second_points,
            cv2.FM_RANSAC,
            1.5,
            0.995,
            4000,
        )
        if mask is not None:
            fundamental_selected = np.asarray(mask).reshape(-1).astype(bool)
    if len(pairs) >= 4:
        first_points = np.asarray([first.points[item[0]] for item in pairs], dtype=np.float32)
        second_points = np.asarray([second.points[item[1]] for item in pairs], dtype=np.float32)
        cv2.setRNGSeed(ransac_seed + 1)
        _, mask = cv2.findHomography(
            first_points,
            second_points,
            cv2.RANSAC,
            2.0,
            maxIters=4000,
            confidence=0.995,
        )
        if mask is not None:
            homography_selected = np.asarray(mask).reshape(-1).astype(bool)
    if int(fundamental_selected.sum()) >= int(homography_selected.sum()):
        selected = fundamental_selected
        selected_model = "fundamental"
    else:
        selected = homography_selected
        selected_model = "homography"
    inlier_pairs = [pair for pair, keep in zip(pairs, selected, strict=True) if keep]
    inlier_points = np.asarray(
        [first.points[item[0]] for item in inlier_pairs],
        dtype=np.float32,
    ).reshape(-1, 2)
    inlier_count = len(inlier_pairs)
    return _PairEvaluation(
        metrics={
            "first_keypoint_count": len(first.points),
            "second_keypoint_count": len(second.points),
            "mutual_ratio_match_count": len(pairs),
            "fundamental_inlier_count": int(fundamental_selected.sum()),
            "homography_inlier_count": int(homography_selected.sum()),
            "selected_geometric_model": selected_model,
            "geometric_inlier_count": inlier_count,
            "geometric_inlier_ratio": inlier_count / len(pairs) if pairs else 0.0,
            "foreground_coverage": _coverage(inlier_points, first.foreground_area),
            "median_patch_ncc": _patch_ncc(first, second, inlier_pairs),
        },
        inlier_pairs=inlier_pairs,
    )


def _evaluate_pair(
    first: _Features,
    second: _Features,
    *,
    descriptor_norm: int,
    ransac_seed: int,
) -> dict[str, int | float | str | None]:
    return _evaluate_pair_details(
        first,
        second,
        descriptor_norm=descriptor_norm,
        ransac_seed=ransac_seed,
    ).metrics


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _aggregate(pairs: list[dict[str, Any]]) -> dict[str, int | float | None]:
    ncc = [
        float(pair["median_patch_ncc"]) for pair in pairs if pair["median_patch_ncc"] is not None
    ]
    return {
        "pair_count": len(pairs),
        "median_geometric_inliers": _median(
            [float(pair["geometric_inlier_count"]) for pair in pairs]
        ),
        "median_geometric_inlier_ratio": _median(
            [float(pair["geometric_inlier_ratio"]) for pair in pairs]
        ),
        "median_foreground_coverage": _median(
            [float(pair["foreground_coverage"]) for pair in pairs]
        ),
        "median_patch_ncc": _median(ncc) if ncc else None,
    }


def _passes(
    aggregate: dict[str, int | float | None],
    *,
    inliers: int,
    ratio: float,
    coverage: float,
) -> bool:
    return (
        int(aggregate["pair_count"] or 0) > 0
        and float(aggregate["median_geometric_inliers"] or 0.0) >= inliers
        and float(aggregate["median_geometric_inlier_ratio"] or 0.0) >= ratio
        and float(aggregate["median_foreground_coverage"] or 0.0) >= coverage
    )


def _numeric_metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics[name]
    if not isinstance(value, (int, float)):
        raise TypeError(f"correspondence metric {name} is not numeric")
    return float(value)


def _edge_passes(metrics: dict[str, Any], gate: TemporalGraphGate) -> bool:
    return (
        _numeric_metric(metrics, "geometric_inlier_count") >= gate.minimum_edge_inliers
        and _numeric_metric(metrics, "geometric_inlier_ratio") >= gate.minimum_edge_inlier_ratio
        and _numeric_metric(metrics, "foreground_coverage") >= gate.minimum_edge_coverage
    )


def _largest_component_fraction(ordinals: list[int], edges: list[tuple[int, int]]) -> float:
    adjacency: dict[int, set[int]] = {ordinal: set() for ordinal in ordinals}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    maximum_size = 0
    remaining = set(ordinals)
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbors = adjacency[current] & remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        maximum_size = max(maximum_size, size)
    return maximum_size / len(ordinals) if ordinals else 0.0


def scan_temporal_track_graph(
    manifest_path: Path,
    output_path: Path,
    *,
    validation_path: Path | None = None,
    binding_path: Path | None = None,
    maximum_dimension: int = 720,
    seed: int = 20260902,
    gate: TemporalGraphGate | None = None,
) -> Path:
    """Test whether local train-only tracklet factors span and close the rotation."""

    gate = gate or TemporalGraphGate()
    validation_path = validation_path or manifest_path.parent / "dataset_validation.json"
    protected_paths = [manifest_path, validation_path, output_path]
    if binding_path is not None:
        protected_paths.append(binding_path)
    reject_sealed_capability(protected_paths)
    validation = read_json(validation_path)
    if validation.get("status") != "ready" or validation.get("blockers"):
        raise ValueError("temporal track graph requires a ready dataset validation report")
    frames, held_out_record_count = _bind_training_frames(manifest_path)
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("the frozen track-graph qualification requires OpenCV SIFT")
    detector = cv2.SIFT_create(
        nfeatures=2400,
        contrastThreshold=0.02,
        edgeThreshold=12,
    )
    features = {
        frame.record.ordinal: _read_features(
            frame,
            detector=detector,
            maximum_dimension=maximum_dimension,
        )
        for frame in frames
    }
    edge_reports: list[dict[str, Any]] = []
    passing_edges: list[tuple[int, int]] = []
    binding_offsets = [0]
    binding_first_points: list[np.ndarray] = []
    binding_second_points: list[np.ndarray] = []
    binding_weights: list[np.ndarray] = []
    binding_first_ordinals: list[int] = []
    binding_second_ordinals: list[int] = []
    binding_first_sources: list[int] = []
    binding_second_sources: list[int] = []
    binding_model_codes: list[int] = []
    for edge_number, (first, second) in enumerate(pairwise(frames)):
        evaluation = _evaluate_pair_details(
            features[first.record.ordinal],
            features[second.record.ordinal],
            descriptor_norm=cv2.NORM_L2,
            ransac_seed=seed + edge_number,
        )
        metrics = evaluation.metrics
        passed = _edge_passes(metrics, gate)
        if passed:
            passing_edges.append((first.record.ordinal, second.record.ordinal))
            first_features = features[first.record.ordinal]
            second_features = features[second.record.ordinal]
            first_points = np.asarray(
                [first_features.points[item[0]] for item in evaluation.inlier_pairs],
                dtype=np.float32,
            ).reshape(-1, 2)
            second_points = np.asarray(
                [second_features.points[item[1]] for item in evaluation.inlier_pairs],
                dtype=np.float32,
            ).reshape(-1, 2)
            first_points /= first_features.scale
            second_points /= second_features.scale
            edge_weight = min(1.0, _numeric_metric(metrics, "geometric_inlier_ratio"))
            binding_first_points.append(first_points)
            binding_second_points.append(second_points)
            binding_weights.append(np.full((len(first_points),), edge_weight, dtype=np.float32))
            binding_first_ordinals.append(first.record.ordinal)
            binding_second_ordinals.append(second.record.ordinal)
            binding_first_sources.append(first.record.source_frame_index)
            binding_second_sources.append(second.record.source_frame_index)
            binding_model_codes.append(
                0 if metrics["selected_geometric_model"] == "fundamental" else 1
            )
            binding_offsets.append(binding_offsets[-1] + len(first_points))
        edge_reports.append(
            {
                "first_ordinal": first.record.ordinal,
                "second_ordinal": second.record.ordinal,
                "ordinal_delta": second.record.ordinal - first.record.ordinal,
                "passed": passed,
                **metrics,
            }
        )
    loop_metrics = _evaluate_pair(
        features[frames[0].record.ordinal],
        features[frames[-1].record.ordinal],
        descriptor_norm=cv2.NORM_L2,
        ransac_seed=seed + len(edge_reports),
    )
    loop_pass = (
        _numeric_metric(loop_metrics, "geometric_inlier_count") >= gate.minimum_loop_inliers
        and _numeric_metric(loop_metrics, "geometric_inlier_ratio")
        >= gate.minimum_loop_inlier_ratio
        and _numeric_metric(loop_metrics, "foreground_coverage") >= gate.minimum_loop_coverage
    )
    passing_edge_fraction = len(passing_edges) / len(edge_reports)
    component_fraction = _largest_component_fraction(
        [frame.record.ordinal for frame in frames],
        passing_edges,
    )
    graph_pass = (
        passing_edge_fraction >= gate.minimum_passing_edge_fraction
        and component_fraction >= gate.minimum_largest_component_fraction
        and loop_pass
    )
    blockers: list[str] = []
    if passing_edge_fraction < gate.minimum_passing_edge_fraction:
        blockers.append("temporal_edge_pass_fraction_below_gate")
    if component_fraction < gate.minimum_largest_component_fraction:
        blockers.append("temporal_graph_component_coverage_below_gate")
    if not loop_pass:
        blockers.append("full_rotation_loop_closure_below_gate")
    proposal_binding: dict[str, Any] | None = None
    if binding_path is not None:
        if binding_path.exists():
            raise FileExistsError(f"track-factor binding already exists: {binding_path}")
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            binding_path,
            schema_version=np.asarray("frayid_v2_pairwise_tracklet_factors.v1"),
            first_ordinals=np.asarray(binding_first_ordinals, dtype=np.int64),
            second_ordinals=np.asarray(binding_second_ordinals, dtype=np.int64),
            first_source_frame_indices=np.asarray(binding_first_sources, dtype=np.int64),
            second_source_frame_indices=np.asarray(binding_second_sources, dtype=np.int64),
            edge_offsets=np.asarray(binding_offsets, dtype=np.int64),
            first_pixels=(
                np.concatenate(binding_first_points, axis=0).astype(np.float32)
                if binding_first_points
                else np.empty((0, 2), dtype=np.float32)
            ),
            second_pixels=(
                np.concatenate(binding_second_points, axis=0).astype(np.float32)
                if binding_second_points
                else np.empty((0, 2), dtype=np.float32)
            ),
            observation_weights=(
                np.concatenate(binding_weights, axis=0).astype(np.float32)
                if binding_weights
                else np.empty((0,), dtype=np.float32)
            ),
            geometric_model_codes=np.asarray(binding_model_codes, dtype=np.uint8),
        )
        proposal_binding = {
            "path": str(binding_path),
            "sha256": sha256_file(binding_path),
            "edge_count": len(binding_first_ordinals),
            "factor_count": binding_offsets[-1],
            "role": "uncertain_pairwise_observation_proposals_not_material_track_truth",
            "model_codebook": {"0": "fundamental", "1": "homography"},
        }
    report = {
        "schema_version": "frayid_v2_temporal_track_graph.v1",
        "status": "pass",
        "qualification_id": "postv2_q01_temporal_track_graph_r01",
        "purpose": "train_only_t01_factor_graph_selection_not_material_track_truth",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_validation": str(validation_path),
        "source_validation_sha256": sha256_file(validation_path),
        "detector": "opencv_sift",
        "opencv_version": cv2.__version__,
        "seed": seed,
        "maximum_dimension": maximum_dimension,
        "frozen_gate": gate.as_dict(),
        "access_counters": {
            "accepted_training_records_bound": len(frames),
            "training_images_read": len(frames),
            "training_masks_read": len(frames),
            "held_out_records_present_but_not_bound": held_out_record_count,
            "held_out_images_read": 0,
            "development_metrics_read": 0,
            "sealed_test_accesses": 0,
        },
        "graph_metrics": {
            "frame_node_count": len(frames),
            "temporal_edge_count": len(edge_reports),
            "passing_temporal_edge_count": len(passing_edges),
            "passing_temporal_edge_fraction": passing_edge_fraction,
            "largest_passing_component_fraction": component_fraction,
            "median_edge_geometric_inliers": _median(
                [float(report["geometric_inlier_count"]) for report in edge_reports]
            ),
            "median_edge_geometric_inlier_ratio": _median(
                [float(report["geometric_inlier_ratio"]) for report in edge_reports]
            ),
            "median_edge_foreground_coverage": _median(
                [float(report["foreground_coverage"]) for report in edge_reports]
            ),
            "loop_closure_pass": loop_pass,
            "loop_closure": loop_metrics,
        },
        "edges": edge_reports,
        "gate_results": {
            "temporal_track_graph_eligible_for_t01": graph_pass,
            "direct_quarter_turn_identity_factor_enabled": False,
            "long_material_track_factor_enabled": False,
            "photometric_factor_enabled": False,
        },
        "proposal_binding": proposal_binding,
        "recommended_t01_route": (
            "mask_boundary_plus_robust_local_tracklet_factor_graph"
            if graph_pass
            else "mask_boundary_only_then_one_learned_long_track_qualification"
        ),
        "route_blockers": blockers,
        "scientific_attempt_marker_created": False,
        "optimizer_steps": 0,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "Passing edges supply local pairwise observation factors, not persistent material identity.",
            "Occlusion makes missing direct quarter-turn matches expected; they remain disabled.",
            "Photometric normal recovery requires separately qualified long material track chains.",
        ],
    }
    write_json(output_path, report)
    return output_path


def scan_correspondence_viability(
    manifest_path: Path,
    output_path: Path,
    *,
    validation_path: Path | None = None,
    maximum_pairs_per_bin: int = 8,
    maximum_dimension: int = 720,
    seed: int = 20260902,
    gate: CorrespondenceGate | None = None,
) -> Path:
    """Measure train-only identity evidence before selecting the T01 observation model."""

    gate = gate or CorrespondenceGate()
    validation_path = validation_path or manifest_path.parent / "dataset_validation.json"
    reject_sealed_capability([manifest_path, validation_path, output_path])
    validation = read_json(validation_path)
    if validation.get("status") != "ready" or validation.get("blockers"):
        raise ValueError("correspondence scan requires a ready dataset validation report")
    frames, held_out_record_count = _bind_training_frames(manifest_path)
    schedule = _pair_schedule(frames, maximum_pairs_per_bin=maximum_pairs_per_bin)
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("the frozen correspondence qualification requires OpenCV SIFT")
    detector = cv2.SIFT_create(
        nfeatures=2400,
        contrastThreshold=0.02,
        edgeThreshold=12,
    )
    detector_name = "opencv_sift"
    descriptor_norm = cv2.NORM_L2
    features: dict[int, _Features] = {}
    reads: set[int] = set()
    pair_reports: dict[str, list[dict[str, Any]]] = {}
    pair_number = 0
    for bin_name, pairs in schedule.items():
        reports: list[dict[str, Any]] = []
        for first, second in pairs:
            if first.record.ordinal not in features:
                features[first.record.ordinal] = _read_features(
                    first,
                    detector=detector,
                    maximum_dimension=maximum_dimension,
                )
                reads.add(first.record.ordinal)
            if second.record.ordinal not in features:
                features[second.record.ordinal] = _read_features(
                    second,
                    detector=detector,
                    maximum_dimension=maximum_dimension,
                )
                reads.add(second.record.ordinal)
            metrics = _evaluate_pair(
                features[first.record.ordinal],
                features[second.record.ordinal],
                descriptor_norm=descriptor_norm,
                ransac_seed=seed + pair_number,
            )
            reports.append(
                {
                    "first_ordinal": first.record.ordinal,
                    "second_ordinal": second.record.ordinal,
                    "ordinal_delta": second.record.ordinal - first.record.ordinal,
                    "first_source_frame_index": first.record.source_frame_index,
                    "second_source_frame_index": second.record.source_frame_index,
                    **metrics,
                }
            )
            pair_number += 1
        pair_reports[bin_name] = reports
    aggregates = {name: _aggregate(reports) for name, reports in pair_reports.items()}
    local_pass = _passes(
        aggregates["local"],
        inliers=gate.local_median_inliers,
        ratio=gate.local_median_inlier_ratio,
        coverage=gate.local_median_coverage,
    )
    quarter_pass = _passes(
        aggregates["quarter_turn"],
        inliers=gate.quarter_median_inliers,
        ratio=gate.quarter_median_inlier_ratio,
        coverage=gate.quarter_median_coverage,
    )
    loop_pass = _passes(
        aggregates["loop_closure"],
        inliers=gate.loop_median_inliers,
        ratio=gate.loop_median_inlier_ratio,
        coverage=gate.loop_median_coverage,
    )
    track_driven_t01 = local_pass and quarter_pass and loop_pass
    if track_driven_t01:
        route = "classical_tracks_may_enter_t01_with_robust_uncertainty"
        blockers: list[str] = []
    elif local_pass:
        route = "mask_boundary_t01_then_one_learned_long_track_qualification"
        blockers = [
            name
            for name, passed in (
                ("quarter_turn_correspondence_gate_failed", quarter_pass),
                ("loop_closure_correspondence_gate_failed", loop_pass),
            )
            if not passed
        ]
    else:
        route = "mask_boundary_only_t01_until_correspondence_observation_is_repaired"
        blockers = ["local_correspondence_gate_failed"]
    report = {
        "schema_version": "frayid_v2_correspondence_viability.v1",
        "status": "pass",
        "qualification_id": "postv2_q00_train_correspondence_viability_r01",
        "purpose": "observation_model_selection_not_scientific_geometry_result",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_validation": str(validation_path),
        "source_validation_sha256": sha256_file(validation_path),
        "detector": detector_name,
        "opencv_version": cv2.__version__,
        "seed": seed,
        "maximum_dimension": maximum_dimension,
        "maximum_pairs_per_bin": maximum_pairs_per_bin,
        "frozen_gate": gate.as_dict(),
        "access_counters": {
            "accepted_training_records_bound": len(frames),
            "training_images_read": len(reads),
            "training_masks_read": len(reads),
            "held_out_records_present_but_not_bound": held_out_record_count,
            "held_out_images_read": 0,
            "development_metrics_read": 0,
            "sealed_test_accesses": 0,
        },
        "aggregates": aggregates,
        "pairs": pair_reports,
        "gate_results": {
            "local": local_pass,
            "quarter_turn": quarter_pass,
            "loop_closure": loop_pass,
            "track_driven_t01_eligible": track_driven_t01,
        },
        "recommended_t01_route": route,
        "route_blockers": blockers,
        "photometric_normal_route": {
            "eligible": False,
            "reason": "pairwise_matches_do_not_establish_long_material_track_chains",
            "next_gate": "train_only_long_tracks_with_visibility_uncertainty_and_low_rank_intensity_test",
        },
        "scientific_attempt_marker_created": False,
        "optimizer_steps": 0,
        "modal_jobs": 0,
        "automatic_retries": 0,
        "notes": [
            "Robust fundamental/homography inliers are a viability diagnostic, not ground-truth tracks.",
            "No development or sealed frame is available to the feature extractor.",
            "A learned tracker may propose observations later; the geometric optimizer remains judge.",
        ],
    }
    write_json(output_path, report)
    return output_path
