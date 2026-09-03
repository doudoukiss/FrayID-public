from __future__ import annotations

import numpy as np
import pytest
import trimesh

from frayid.source_exclusion_carrier import (
    EXPERIMENT_ID,
    source_exclusion_shrinkwrap,
    uniform_conforming_subdivide,
)


def test_uniform_conforming_subdivision_preserves_surface_and_topology() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    vertices, faces = uniform_conforming_subdivide(
        np.asarray(mesh.vertices), np.asarray(mesh.faces)
    )
    refined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert len(faces) == len(mesh.faces) * 16
    assert refined.is_watertight
    assert refined.euler_number == 2
    _, distances, _ = trimesh.proximity.closest_point(mesh, vertices)  # type: ignore[no-untyped-call]
    assert float(np.max(distances)) < 1e-12


def test_source_exclusion_pressure_is_deterministic() -> None:
    pytest.importorskip("ipctk")
    assert EXPERIMENT_ID == "postv1_e13_source_exclusion_shrinkwrap_r01"
    source = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    initial = source.copy()
    initial.vertices = np.asarray(initial.vertices) * 1.1
    arguments = (
        np.asarray(initial.vertices),
        np.asarray(initial.faces),
        np.asarray(source.vertices),
        np.asarray(source.faces),
    )
    first = source_exclusion_shrinkwrap(*arguments, pitch=0.5)
    second = source_exclusion_shrinkwrap(*arguments, pitch=0.5)
    assert first.status == "pass"
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)
    assert first.report()["steps"] == second.report()["steps"]
    assert first.report()["source_source_pairs_filtered"] is True
