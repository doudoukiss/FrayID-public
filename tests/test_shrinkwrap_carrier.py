from __future__ import annotations

import numpy as np
import pytest
import trimesh

from frayid.shrinkwrap_carrier import (
    EXPERIMENT_ID,
    PRESSURE_ITERATIONS,
    TARGET_OFFSET_PITCH,
    pressure_shrinkwrap,
)


def test_e12_registration_constants_are_frozen() -> None:
    assert EXPERIMENT_ID == "postv1_e12_ccd_shrinkwrap_carrier_r01"
    assert PRESSURE_ITERATIONS == 96
    assert TARGET_OFFSET_PITCH == 0.0025


def test_pressure_shrinkwrap_is_deterministic_and_keeps_connectivity() -> None:
    pytest.importorskip("ipctk")
    source = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    initial = source.copy()
    initial.vertices = np.asarray(initial.vertices) * 1.1
    first = pressure_shrinkwrap(
        np.asarray(initial.vertices),
        np.asarray(initial.faces),
        np.asarray(source.vertices),
        np.asarray(source.faces),
        pitch=0.5,
    )
    second = pressure_shrinkwrap(
        np.asarray(initial.vertices),
        np.asarray(initial.faces),
        np.asarray(source.vertices),
        np.asarray(source.faces),
        pitch=0.5,
    )
    assert first.status == "pass"
    assert len(first.steps) <= PRESSURE_ITERATIONS
    assert first.converged or len(first.steps) == PRESSURE_ITERATIONS
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)
    assert first.report()["steps"] == second.report()["steps"]
    output = trimesh.Trimesh(vertices=first.vertices, faces=first.faces, process=False)
    assert output.is_watertight
    assert output.euler_number == 2
