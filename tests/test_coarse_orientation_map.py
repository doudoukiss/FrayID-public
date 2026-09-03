from __future__ import annotations

import json

import numpy as np
import trimesh

from frayid.certified_tet_path import certify_piecewise_affine_path
from frayid.coarse_bilipschitz import FreudenthalLatticeV1, refine_surface_to_lattice
from frayid.coarse_orientation_map import (
    fit_and_certify_coarse_orientation_step,
    run_coarse_orientation_controls,
)


def _lattice() -> FreudenthalLatticeV1:
    return FreudenthalLatticeV1.create(
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        nodes_per_axis=4,
    )


def test_exact_piecewise_affine_path_accepts_shear_and_truncates_fold() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    shear = np.zeros_like(vertices)
    shear[2, 0] = 1.5
    accepted = certify_piecewise_affine_path(
        vertices, tetrahedra, shear, vertices, faces, shear, timeout_seconds=None
    )
    assert accepted.status == "pass"
    assert accepted.accepted_alpha == 1.0

    fold = np.zeros_like(vertices)
    fold[1, 0] = -2.0
    truncated = certify_piecewise_affine_path(
        vertices, tetrahedra, fold, vertices, faces, fold, timeout_seconds=None
    )
    assert truncated.status == "pass"
    assert 0.0 < truncated.accepted_alpha < 0.5


def test_coarse_orientation_fit_is_fixed_boundary_and_replayable() -> None:
    lattice = _lattice()
    mesh = trimesh.creation.box(extents=(1.0, 0.9, 0.8))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    refined = refine_surface_to_lattice(lattice, vertices, faces)
    proposal = np.zeros_like(vertices)
    proposal[:, 0] = 0.01 * (1.0 - vertices[:, 0] ** 2) * (1.0 - vertices[:, 1] ** 2)
    first = fit_and_certify_coarse_orientation_step(
        lattice,
        vertices,
        faces,
        refined,
        proposal,
        minimum_retained_displacement_ratio=0.0,
        timeout_seconds=None,
    )
    second = fit_and_certify_coarse_orientation_step(
        lattice,
        vertices,
        faces,
        refined,
        proposal,
        minimum_retained_displacement_ratio=0.0,
        timeout_seconds=None,
    )
    assert first.status == "pass"
    assert first.accepted_alpha == 1.0
    assert first.accepted_path.accepted_alpha == 1.0
    assert first.decision_sha256 == second.decision_sha256
    assert np.array_equal(first.accepted_controls, second.accepted_controls)
    assert np.all(first.accepted_controls[lattice.boundary_mask] == 0.0)


def test_piecewise_affine_timeout_is_unknown() -> None:
    lattice = _lattice()
    triangle = np.asarray([[-0.4, -0.2, 0.0], [0.4, -0.2, 0.0], [0.0, 0.4, 0.0]], dtype=np.float64)
    result = certify_piecewise_affine_path(
        lattice.vertices,
        lattice.tetrahedra,
        np.zeros_like(lattice.vertices),
        triangle,
        np.asarray([[0, 1, 2]], dtype=np.int64),
        np.zeros_like(triangle),
        timeout_seconds=0.0,
    )
    assert result.status == "unknown"
    assert result.accepted_alpha == 0.0


def test_registered_coarse_orientation_controls_pass() -> None:
    report = run_coarse_orientation_controls()
    assert report["status"] == "pass"
    assert all(report["checks"].values())
    json.dumps(report)
