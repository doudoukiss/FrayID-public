from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.t05_fixed_camera import FixedCameraHumanSolution
from frayid.v3.schemas import MaterialChartGraph

EXPERIMENT_ID = "postv3_a01_information_gain_recapture_r01"
COMPARISON_VIDEO_SHA256 = "unavailable-in-public-snapshot"
REGIONS = (
    "head_hair",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_arm_and_hand",
    "right_arm_and_hand",
    "front_torso",
    "back_torso",
    "upper_garment_hem",
    "waist",
    "shorts",
    "left_leg",
    "right_leg",
    "left_foot",
    "right_foot",
)
UNCERTAINTY_ONLY_REGIONS = (
    "head_hair",
    "left_arm_and_hand",
    "right_arm_and_hand",
    "shorts",
    "left_leg",
    "right_leg",
    "left_foot",
    "right_foot",
)


def _select_anchor_indices(phases_degrees: np.ndarray) -> list[int]:
    if phases_degrees.ndim != 1 or len(phases_degrees) < 12:
        raise ValueError("regional atlas requires at least 12 phase-tagged training records")
    chosen: list[int] = []
    for target in range(0, 360, 30):
        distance = np.abs((phases_degrees - target + 180.0) % 360.0 - 180.0)
        candidates = np.argsort(distance, kind="stable")
        selected = next((int(index) for index in candidates if int(index) not in chosen), None)
        if selected is None:
            raise ValueError("could not choose 12 distinct phase anchors")
        chosen.append(selected)
    return chosen


def _dataset_index_by_source(dataset_root: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    pattern = re.compile(r"^frame_(\d+)_source_(\d+)\.png$")
    for path in (dataset_root / "images").glob("*.png"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            result[int(match.group(2))] = int(match.group(1))
    if not result:
        raise ValueError("legacy dataset has no source-indexed images")
    return result


def _indexed_evidence(dataset_root: Path, kind: str, dataset_index: int, source: int) -> Path:
    path = dataset_root / kind / f"frame_{dataset_index:04d}_source_{source:06d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing {kind} evidence for source frame {source}: {path}")
    return path


def build_regional_geometry_failure_atlas(
    *,
    t05_solution_path: Path,
    v00_master_path: Path,
    q04_graph_path: Path,
    dataset_root: Path,
    comparison_video_path: Path,
    output_image_path: Path,
) -> dict[str, Any]:
    """Freeze a train-only 12-phase evaluator storyboard after terminal Q04."""
    reject_sealed_capability(
        [
            t05_solution_path,
            v00_master_path,
            q04_graph_path,
            dataset_root,
            comparison_video_path,
            output_image_path,
        ]
    )
    if output_image_path.exists():
        raise FileExistsError(f"regional diagnostic image already exists: {output_image_path}")
    if sha256_file(comparison_video_path) != COMPARISON_VIDEO_SHA256:
        raise ValueError("B-GEO-V1 source-comparison video hash mismatch")
    solution = FixedCameraHumanSolution.model_validate(read_json(t05_solution_path))
    if solution.training_frame_count != 144 or solution.development_images_read != 0:
        raise ValueError("regional train-fit atlas requires exactly 144 clean T05 records")
    graph = MaterialChartGraph.model_validate(read_json(q04_graph_path))
    if graph.status != "fail" or graph.promotion_eligible:
        raise ValueError("regional failure atlas requires the immutable terminal Q04 failure")
    master = read_json(v00_master_path)
    master_by_source = {int(item["source_frame_index"]): item for item in master["frames"]}
    source_to_dataset = _dataset_index_by_source(dataset_root)

    first_phase = float(solution.frames[0].yaw_radians)
    phases = np.asarray(
        [np.degrees(float(frame.yaw_radians) - first_phase) % 360.0 for frame in solution.frames]
    )
    selected_indices = _select_anchor_indices(phases)
    observation_count_by_frame: dict[int, int] = {}
    for track in graph.tracks:
        if not track.accepted:
            continue
        for observation in track.observations:
            observation_count_by_frame[observation.frame_index] = (
                observation_count_by_frame.get(observation.frame_index, 0) + 1
            )

    capture = cv2.VideoCapture(str(comparison_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open B-GEO-V1 comparison: {comparison_video_path}")
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != 180:
        raise ValueError("B-GEO-V1 comparison must contain exactly 180 frames")
    panels: list[np.ndarray] = []
    anchors: list[dict[str, Any]] = []
    try:
        for target, local_index in zip(range(0, 360, 30), selected_indices, strict=True):
            state = solution.frames[local_index]
            source_index = int(state.source_frame_index)
            record = master_by_source[source_index]
            source_path = v00_master_path.parent / str(record["lossless_frame_path"])
            if sha256_file(source_path) != record["lossless_frame_sha256"]:
                raise ValueError(f"V00 source hash mismatch at {source_index}")
            dataset_index = source_to_dataset[source_index]
            mask_path = _indexed_evidence(dataset_root, "masks", dataset_index, source_index)
            normal_path = _indexed_evidence(dataset_root, "normals", dataset_index, source_index)

            capture.set(cv2.CAP_PROP_POS_FRAMES, float(dataset_index))
            ok, panel = capture.read()
            if not ok or panel is None:
                raise RuntimeError(f"could not read comparison frame {dataset_index}")
            panel = cv2.resize(panel, (432, 224), interpolation=cv2.INTER_AREA)
            supported = (target // 30) in graph.phase_bins_spanned
            color = (40, 170, 40) if supported else (30, 30, 220)
            cv2.rectangle(panel, (0, 0), (431, 27), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"target {target:03d} deg | observed {phases[local_index]:06.2f} | "
                f"tracks {observation_count_by_frame.get(local_index, 0):03d}",
                (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(panel, (0, 0), (431, 223), color, 3)
            panels.append(panel)
            anchors.append(
                {
                    "target_phase_degrees": target,
                    "observed_phase_degrees": float(phases[local_index]),
                    "absolute_phase_error_degrees": float(
                        abs((phases[local_index] - target + 180.0) % 360.0 - 180.0)
                    ),
                    "t05_training_index": local_index,
                    "source_frame_index": source_index,
                    "comparison_video_frame_index": dataset_index,
                    "q04_phase_bin_supported": supported,
                    "accepted_q04_observations": observation_count_by_frame.get(local_index, 0),
                    "source_rgb": {"path": str(source_path), "sha256": sha256_file(source_path)},
                    "source_mask": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
                    "observed_normal": {
                        "path": str(normal_path),
                        "sha256": sha256_file(normal_path),
                    },
                }
            )
    finally:
        capture.release()

    rows = [np.concatenate(panels[index : index + 3], axis=1) for index in range(0, 12, 3)]
    storyboard = np.concatenate(rows, axis=0)
    encoded, buffer = cv2.imencode(".png", storyboard)
    if not encoded:
        raise RuntimeError("could not encode regional diagnostic storyboard")
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    with output_image_path.open("xb") as handle:
        handle.write(buffer.tobytes())

    return {
        "schema_version": "frayid_v3_regional_geometry_failure_atlas.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "partial_evaluator_diagnostic",
        "promotion_eligible": False,
        "evidence_scope": "training_records_evaluator_only",
        "baseline_id": "B-GEO-V1",
        "regions": list(REGIONS),
        "uncertainty_only_regions": list(UNCERTAINTY_ONLY_REGIONS),
        "first_promotable_region": "sleeveless_upper_garment_only",
        "available_channels": [
            "source_rgb",
            "source_mask",
            "neutral_render",
            "registration_overlay",
            "observed_normal",
            "q04_track_support",
        ],
        "deferred_channels": [
            "rendered_normal",
            "normal_disagreement",
            "regional_semantic_labels",
            "boundary_distance_heatmap",
            "body_garment_ownership",
        ],
        "deferred_channel_policy": "must_be_computed_by_future_qualified_stage_never_filled_with_proxy_truth",
        "anchors": anchors,
        "storyboard": {
            "path": str(output_image_path),
            "sha256": sha256_file(output_image_path),
            "width": int(storyboard.shape[1]),
            "height": int(storyboard.shape[0]),
        },
        "input_hashes": {
            "t05_solution": sha256_file(t05_solution_path),
            "v00_master": sha256_file(v00_master_path),
            "q04_terminal_graph": sha256_file(q04_graph_path),
            "b_geo_v1_comparison": sha256_file(comparison_video_path),
        },
        "training_records_read": 12,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "fitting_access": False,
    }


__all__ = ["build_regional_geometry_failure_atlas"]
