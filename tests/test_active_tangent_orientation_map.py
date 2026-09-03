from __future__ import annotations

import json

import numpy as np
import trimesh

from frayid.active_tangent_orientation_map import (
    MAXIMUM_NORMALIZED_KKT_RESIDUAL,
    MAXIMUM_TANGENT_RESIDUAL,
    fit_active_determinant_tangent_controls,
    fit_and_certify_active_tangent_step,
    run_active_tangent_controls,
    tetrahedron_determinant_vertex_gradients,
    tetrahedron_determinants,
)
from frayid.coarse_bilipschitz import FreudenthalLatticeV1, refine_surface_to_lattice


def _lattice() -> FreudenthalLatticeV1:
    return FreudenthalLatticeV1.create(
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        nodes_per_axis=4,
    )


def test_determinant_vertex_gradient_matches_centered_difference() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    direction = np.asarray(
        [[0.1, -0.2, 0.05], [-0.1, 0.3, 0.2], [0.2, 0.1, -0.2], [-0.2, -0.1, 0.1]],
        dtype=np.float64,
    )
    analytic = float(np.sum(tetrahedron_determinant_vertex_gradients(vertices) * direction))
    epsilon = 1.0e-6
    tetrahedron = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    finite = float(
        (
            tetrahedron_determinants(vertices + epsilon * direction, tetrahedron)[0]
            - tetrahedron_determinants(vertices - epsilon * direction, tetrahedron)[0]
        )
        / (2.0 * epsilon)
    )
    assert abs(analytic - finite) <= 1.0e-8


def test_active_tangent_kkt_solve_is_deterministic_and_fixed_boundary() -> None:
    lattice = _lattice()
    current = lattice.vertices.copy()
    current[np.flatnonzero(~lattice.boundary_mask)[0], 0] += 0.4
    points = np.asarray(
        [[x, y, z] for x in (-0.4, 0.0, 0.4) for y in (-0.4, 0.0, 0.4) for z in (-0.4, 0.0, 0.4)],
        dtype=np.float64,
    )
    residual = np.zeros_like(points)
    residual[:, 0] = 0.1
    first = fit_active_determinant_tangent_controls(
        lattice, points, residual, current, active_ratio=0.5, deadline=None
    )
    second = fit_active_determinant_tangent_controls(
        lattice, points, residual, current, active_ratio=0.5, deadline=None
    )
    assert first.status == "pass"
    assert first.active_tetrahedron_indices.size > 0
    assert first.normalized_kkt_residual <= MAXIMUM_NORMALIZED_KKT_RESIDUAL
    assert first.maximum_tangent_residual <= MAXIMUM_TANGENT_RESIDUAL
    assert np.array_equal(first.controls, second.controls)
    assert np.all(first.controls[lattice.boundary_mask] == 0.0)


def test_active_tangent_composition_is_certified_and_serializable() -> None:
    lattice = _lattice()
    mesh = trimesh.creation.box(extents=(1.0, 0.9, 0.8))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    refined = refine_surface_to_lattice(lattice, vertices, faces)
    proposal = np.zeros_like(vertices)
    proposal[:, 0] = 0.08 * vertices[:, 1]
    proposal[:, 1] = -0.04 * vertices[:, 0]
    result = fit_and_certify_active_tangent_step(
        lattice,
        vertices,
        faces,
        refined,
        proposal,
        block_count=2,
        minimum_retained_displacement_ratio=0.0,
        timeout_seconds_per_block=None,
    )
    assert result.status == "pass"
    assert len(result.blocks) == 2
    assert all(block.status == "pass" for block in result.blocks)
    json.dumps(result.report())


def test_registered_active_tangent_controls_pass() -> None:
    report = run_active_tangent_controls()
    assert report["status"] == "pass"
    assert all(report["checks"].values())
    json.dumps(report)
