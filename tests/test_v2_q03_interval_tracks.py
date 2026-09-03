from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.q03_interval_tracks import (
    audit_q03_qualification_lifecycle,
    load_interval_material_track_graph,
    qualify_q03_interval_track_graph,
    robust_material_anchor,
    write_q03_public_benchmark,
)


def _project(
    point: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    camera = np.einsum("tij,j->ti", rotations, point) + translations
    return np.column_stack(
        (
            intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2],
        )
    )


def test_q03_public_partial_tracks_survive_failed_global_cycle(tmp_path: Path) -> None:
    output = write_q03_public_benchmark(tmp_path / "public.json")
    report = read_json(output)
    assert report["status"] == "pass"
    assert report["clean_interval_treatment"]["geometry_improvement_fraction"] >= 0.10
    assert report["clean_interval_treatment"]["reprojection_improvement_fraction"] >= 0.10
    assert report["corrupted_capacity_stress"]["regression_count"] == 0
    assert report["partial_track_control"]["local_interval_pass"] is True
    assert report["partial_track_control"]["global_reverse_cycle_pass"] is False
    assert report["partial_track_control"]["local_photometric_real_minus_shuffled_margin"] >= 0.10


def test_robust_material_anchor_rejects_invalid_or_nonpositive_weights() -> None:
    rotations = np.repeat(np.eye(3)[None], 2, axis=0)
    translations = np.asarray([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
    pixels = np.asarray([[16.0, 16.0], [17.0, 16.0]])
    intrinsics = np.asarray([[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]])
    with np.testing.assert_raises_regex(ValueError, "positively weighted"):
        robust_material_anchor(
            rotations,
            translations,
            pixels,
            intrinsics,
            np.asarray([1.0, 0.0]),
        )


def _write_q03_fixture(root: Path) -> dict[str, Path]:
    frame_count = 144
    phases = np.linspace(0.0, 2.0 * math.pi, frame_count)
    rotations = Rotation.from_rotvec(
        np.column_stack((np.zeros(frame_count), phases, np.zeros(frame_count)))
    ).as_matrix()
    translations = np.repeat(np.asarray([[0.0, 0.0, 3.0]]), frame_count, axis=0)
    intrinsics = np.asarray([[500.0, 0.0, 160.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    t05 = {
        "schema_version": "frayid_v2_t05_fixed_camera_human_solution.v1",
        "experiment_id": "postv2_t05_background_anchored_fixed_camera_human_ba_r01",
        "status": "qualification_candidate",
        "source_revision": "a" * 40,
        "shared_intrinsics": intrinsics.tolist(),
        "distortion_coefficients": [0.0] * 5,
        "physical_camera_rotation": np.eye(3).tolist(),
        "physical_camera_translation": [0.0, 0.0, 0.0],
        "spin_axis_camera": [0.0, 1.0, 0.0],
        "root_center_camera_metres": [0.0, 0.0, 3.0],
        "base_orientation_rotvec": [0.0, 0.0, 0.0],
        "micromotion_basis": [],
        "micromotion_codes": [[] for _ in range(frame_count)],
        "micromotion_retained_variance": 0.0,
        "frames": [
            {
                "source_frame_index": index,
                "yaw_radians": float(phases[index]),
                "observed_yaw_radians": float(phases[index]),
                "yaw_confidence": 1.0,
                "root_translation_metres": translations[index].tolist(),
                "low_frequency_root_translation_metres": translations[index].tolist(),
                "root_residual_translation_metres": [0.0, 0.0, 0.0],
                "observed_global_orient_rotvec": [0.0, float(phases[index]), 0.0],
                "residual_rotation_rotvec": [0.0, 0.0, 0.0],
                "body_pose": [],
                "body_pose_source": "frozen_camerahmr_smpl_scaffold",
            }
            for index in range(frame_count)
        ],
        "gauge_policy": {},
        "uncertainty": {},
        "provenance": {},
        "source_hashes": {},
        "training_frame_count": frame_count,
        "development_records_used_for_fit": 0,
        "development_images_read": 0,
        "sealed_test_reads": 0,
        "optimizer_steps": 0,
        "paid_jobs": 0,
    }
    t05_solution = write_json(root / "t05_solution.json", t05)
    t05_lifecycle = write_json(
        root / "t05_lifecycle.json",
        {
            "status": "pass",
            "state": "qualified",
            "input_hashes": {"solution": sha256_file(t05_solution)},
        },
    )
    semantic = write_json(root / "semantic.json", {"status": "pass"})
    q01_binding = root / "q01.npz"
    np.savez_compressed(q01_binding, proposal=np.asarray([1], dtype=np.int64))
    q01_report = write_json(
        root / "q01.json",
        {
            "status": "pass",
            "gate_results": {"temporal_track_graph_eligible_for_t01": True},
            "proposal_binding": {"sha256": sha256_file(q01_binding)},
        },
    )

    rng = np.random.default_rng(41)
    offsets = [0]
    ordinals: list[int] = []
    sources: list[int] = []
    pixels: list[np.ndarray] = []
    semantic_codes: list[int] = []
    semantic_confidence: list[float] = []
    luminance: list[float] = []
    fb_error: list[float] = []
    spans: list[float] = []
    track_weights: list[float] = []
    for track_index in range(130):
        step_count = 9 if track_index < 105 else 21
        start = track_index % (frame_count - step_count - 1)
        slots = np.arange(start, start + step_count + 1)
        point = np.asarray(
            [
                rng.uniform(-0.30, 0.30),
                rng.uniform(-0.55, 0.55),
                rng.uniform(-0.12, 0.12),
            ]
        )
        observed = _project(point, rotations[slots], translations[slots], intrinsics)
        observed += rng.normal(0.0, 0.08, size=observed.shape)
        code = track_index % 3
        ordinals.extend(map(int, slots))
        sources.extend(map(int, slots))
        pixels.extend(observed)
        semantic_codes.extend([code] * len(slots))
        semantic_confidence.extend([0.98] * len(slots))
        luminance.extend([1.0] * len(slots))
        fb_error.extend([0.05] * len(slots))
        spans.append(float(np.degrees(phases[slots[-1]] - phases[slots[0]])))
        track_weights.append(0.95)
        offsets.append(len(ordinals))
    q02_binding = root / "q02.npz"
    np.savez_compressed(
        q02_binding,
        schema_version=np.asarray("frayid_v2_visibility_material_tracks.v1"),
        track_offsets=np.asarray(offsets, dtype=np.int64),
        frame_ordinals=np.asarray(ordinals, dtype=np.int64),
        source_frame_indices=np.asarray(sources, dtype=np.int64),
        pixels=np.asarray(pixels, dtype=np.float32),
        semantic_codes=np.asarray(semantic_codes, dtype=np.int16),
        semantic_confidence=np.asarray(semantic_confidence, dtype=np.float32),
        normalized_luminance=np.asarray(luminance, dtype=np.float32),
        local_forward_backward_error=np.asarray(fb_error, dtype=np.float32),
        track_span_degrees=np.asarray(spans, dtype=np.float32),
        track_weights=np.asarray(track_weights, dtype=np.float32),
    )
    q02_report = write_json(
        root / "q02.json",
        {
            "status": "fail",
            "material_track_route": {
                "blockers": ["global_reverse_cycle_gate_failed"],
            },
            "binding": {"sha256": sha256_file(q02_binding)},
            "source_hashes": {"semantic_qualification": sha256_file(semantic)},
            "access_counters": {"training_images_read": 144},
        },
    )
    q02_photometric = write_json(
        root / "q02_photometric.json",
        {
            "status": "pass",
            "photometric_normal_route": {"eligible": True},
        },
    )
    public = write_q03_public_benchmark(root / "public.json")
    return {
        "public": public,
        "t05_solution": t05_solution,
        "t05_lifecycle": t05_lifecycle,
        "q01_report": q01_report,
        "q01_binding": q01_binding,
        "q02_report": q02_report,
        "q02_photometric": q02_photometric,
        "q02_binding": q02_binding,
        "semantic": semantic,
    }


def test_q03_qualification_promotes_intervals_not_global_identity(tmp_path: Path) -> None:
    paths = _write_q03_fixture(tmp_path)
    report_path, binding_path = qualify_q03_interval_track_graph(
        paths["public"],
        paths["t05_solution"],
        paths["t05_lifecycle"],
        paths["q01_report"],
        paths["q01_binding"],
        paths["q02_report"],
        paths["q02_photometric"],
        paths["q02_binding"],
        paths["semantic"],
        tmp_path / "qualification.json",
        tmp_path / "interval_graph.npz",
        source_revision="b" * 40,
    )
    report = read_json(report_path)
    assert report["status"] == "pass"
    assert report["gates"]["q02_terminal_global_cycle_failure_preserved"] is True
    assert report["track_metrics"]["accepted_short_track_count"] >= 100
    assert report["track_metrics"]["accepted_medium_track_count"] >= 20
    assert report["adversarial_corrupted_track_control"]["regression_count"] == 0
    graph = load_interval_material_track_graph(binding_path)
    assert graph.track_count == 130
    assert int(graph.accepted.sum()) == 130
    assert np.allclose(graph.layer_posterior.sum(axis=1), 1.0)
    assert graph.visible_interval_frame_ordinals.shape == (130, 2)
    assert graph.visible_interval_source_indices.shape == (130, 2)
    assert graph.local_descriptor_summary.shape == (130, 4)
    assert np.all(graph.interval_boundary_occlusion_codes == 1)
    lifecycle = audit_q03_qualification_lifecycle(
        paths["public"],
        report_path,
        binding_path,
        tmp_path / "lifecycle.json",
    )
    lifecycle_report = read_json(lifecycle)
    assert lifecycle_report["status"] == "pass"
    assert lifecycle_report["state"] == "qualified"
