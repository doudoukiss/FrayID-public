from __future__ import annotations

import numpy as np

from frayid.io import read_json
from frayid.v2.g03_appearance import (
    render_colored_mesh,
    robust_fuse_vertex_colors,
    write_g03_public_benchmark,
)
from frayid.v2.posed_preview import render_shaded_mesh


def test_robust_vertex_color_fusion_rejects_one_corrupted_view() -> None:
    truth = np.asarray([[0.2, 0.4, 0.8], [0.7, 0.3, 0.1], [0.5, 0.6, 0.2]], dtype=np.float64)
    observations = np.broadcast_to(truth, (5, 3, 3)).copy()
    observations[0] = 1.0 - truth
    valid = np.ones((5, 3), dtype=bool)
    result = robust_fuse_vertex_colors(
        observations,
        valid,
        np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    np.testing.assert_allclose(result.vertex_colors_bgr, truth)
    assert np.all(result.observation_counts == 5)
    assert not np.any(result.prior_filled)


def test_colored_and_neutral_renderers_freeze_foreground() -> None:
    vertices = np.asarray([[-0.4, -0.4, 2.0], [0.4, -0.4, 2.0], [0.0, 0.4, 2.0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    intrinsics = np.asarray([[60.0, 0.0, 31.5], [0.0, 60.0, 31.5], [0.0, 0.0, 1.0]])
    colored, colored_mask = render_colored_mesh(
        vertices,
        faces,
        intrinsics,
        np.asarray([[0.1, 0.2, 0.9], [0.2, 0.8, 0.1], [0.8, 0.1, 0.2]]),
        source_size=(64, 64),
        output_size=(64, 64),
    )
    neutral, neutral_mask = render_shaded_mesh(
        vertices,
        faces,
        intrinsics,
        source_size=(64, 64),
        output_size=(64, 64),
    )
    assert np.array_equal(colored_mask, neutral_mask)
    assert np.any(colored[colored_mask > 0] != neutral[colored_mask > 0])


def test_fixed_shading_changes_only_colored_foreground() -> None:
    vertices = np.asarray([[-0.4, -0.4, 2.0], [0.4, -0.4, 2.0], [0.0, 0.4, 2.0]])
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    intrinsics = np.asarray([[60.0, 0.0, 31.5], [0.0, 60.0, 31.5], [0.0, 0.0, 1.0]])
    colors = np.full((3, 3), 0.5)
    flat, flat_mask = render_colored_mesh(
        vertices,
        faces,
        intrinsics,
        colors,
        source_size=(64, 64),
        output_size=(64, 64),
    )
    shaded, shaded_mask = render_colored_mesh(
        vertices,
        faces,
        intrinsics,
        colors,
        source_size=(64, 64),
        output_size=(64, 64),
        shading_strength=0.25,
    )
    assert np.array_equal(flat_mask, shaded_mask)
    assert np.any(flat[flat_mask > 0] != shaded[shaded_mask > 0])
    assert np.array_equal(flat[0, 0], shaded[0, 0])


def test_g03_public_benchmark_passes_registered_gates(tmp_path) -> None:
    output = write_g03_public_benchmark(tmp_path / "g03_public.json")
    report = read_json(output)
    assert report["status"] == "pass"
    assert report["metrics"]["background_samples_used"] == 0
    assert all(report["gates"].values())
