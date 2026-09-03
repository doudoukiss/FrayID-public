from __future__ import annotations

import numpy as np

from frayid.v2.posed_preview import render_shaded_mesh


def test_posed_preview_rasterizes_front_facing_triangle() -> None:
    vertices = np.asarray([[-0.4, -0.4, 2.0], [0.4, -0.4, 2.0], [0.0, 0.4, 2.0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    intrinsics = np.asarray([[60.0, 0.0, 31.5], [0.0, 60.0, 31.5], [0.0, 0.0, 1.0]])
    image, mask = render_shaded_mesh(
        vertices,
        faces,
        intrinsics,
        source_size=(64, 64),
        output_size=(64, 64),
    )
    assert image.shape == (64, 64, 3)
    assert mask.shape == (64, 64)
    assert int(mask.sum()) > 0
    assert np.any(image[mask > 0] != 244)
