from __future__ import annotations

import numpy as np

from frayid.io import read_json
from frayid.v2.g04_phase_appearance import (
    blinded_candidate_order,
    build_phase_appearance_model,
    phase_color_median_second_difference,
    predict_phase_vertex_colors,
    write_g04_public_benchmark,
)


def _triangle_faces() -> np.ndarray:
    return np.asarray([[0, 1, 2]], dtype=np.int64)


def test_leave_one_out_prediction_cannot_use_target_observation() -> None:
    indices = np.arange(5, dtype=np.int64)
    base = np.asarray([[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])
    observations = np.broadcast_to(base, (5, 3, 3)).copy()
    valid = np.ones((5, 3), dtype=bool)
    clean = build_phase_appearance_model(observations, valid, indices, _triangle_faces())
    observations[2] = 1.0
    corrupted = build_phase_appearance_model(observations, valid, indices, _triangle_faces())
    clean_prediction = predict_phase_vertex_colors(
        clean,
        2,
        bandwidth=1.5,
        prior_weight=0.25,
        exclude_source_index=2,
    )
    corrupted_prediction = predict_phase_vertex_colors(
        corrupted,
        2,
        bandwidth=1.5,
        prior_weight=0.25,
        exclude_source_index=2,
    )
    np.testing.assert_array_equal(clean_prediction, corrupted_prediction)


def test_phase_prediction_is_smooth_and_deterministic() -> None:
    indices = np.arange(8, dtype=np.int64)
    phase = 2.0 * np.pi * indices[:, None, None] / len(indices)
    base = np.full((3, 3), 0.5)
    observations = base + 0.1 * np.sin(phase)
    observations = np.broadcast_to(observations, (8, 3, 3)).copy()
    valid = np.ones((8, 3), dtype=bool)
    model = build_phase_appearance_model(observations, valid, indices, _triangle_faces())
    first = predict_phase_vertex_colors(model, 3, bandwidth=1.5, prior_weight=0.25)
    second = predict_phase_vertex_colors(model, 3, bandwidth=1.5, prior_weight=0.25)
    np.testing.assert_array_equal(first, second)
    assert (
        phase_color_median_second_difference(
            model,
            indices,
            bandwidth=1.5,
            prior_weight=0.25,
        )
        < 0.05
    )


def test_g04_public_benchmark_passes(tmp_path) -> None:
    report_path = write_g04_public_benchmark(tmp_path / "g04_public.json")
    report = read_json(report_path)
    assert report["status"] == "pass"
    assert report["development_reads"] == 0
    assert report["sealed_test_accesses"] == 0
    assert all(report["gates"].values())


def test_g04_blinded_candidate_order_is_deterministic() -> None:
    first = blinded_candidate_order(20260903)
    assert first == blinded_candidate_order(20260903)
    assert set(first) == {"g03_static", "g04_phase"}
