from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from frayid.source_exclusion_carrier import uniform_conforming_subdivide


def test_two_midpoint_rounds_preserve_dyadic_lattice_exactly() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    grid = 2.0**-30
    parent = np.rint(np.asarray(mesh.vertices) / grid) * grid
    vertices, faces = uniform_conforming_subdivide(parent, np.asarray(mesh.faces), rounds=2)
    refined_grid = grid / 4.0
    assert np.array_equal(vertices, np.rint(vertices / refined_grid) * refined_grid)
    assert np.array_equal(vertices[: len(parent)], parent)
    assert len(faces) == len(mesh.faces) * 16


def test_e14_tool_freezes_binary_ratio_and_grid_guard() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "tools/e14_cgal/dyadic_envelope.cpp"
    ).read_text()
    assert "Kernel::FT(129) / Kernel::FT(128)" in source
    assert "std::floor(std::log2(magnitude))" in source
    assert "- 40" in source
    assert "35184372088832.0" in source
