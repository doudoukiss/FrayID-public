from __future__ import annotations

from fractions import Fraction

import numpy as np
import trimesh

from frayid.coarse_bilipschitz import (
    DEFAULT_KAPPA,
    FreudenthalLatticeV1,
    exact_gradient_frobenius_squared,
    exact_max_gradient_frobenius_squared,
    fit_and_certify_bilipschitz_step,
    parent_area_path_report,
    refine_surface_to_lattice,
    run_bilipschitz_controls,
)


def _lattice(nodes: int = 4) -> FreudenthalLatticeV1:
    return FreudenthalLatticeV1.create(
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        nodes_per_axis=nodes,
    )


def test_freudenthal_lattice_has_complete_fixed_boundary() -> None:
    lattice = _lattice(8)
    assert lattice.vertices.shape == (512, 3)
    assert lattice.tetrahedra.shape == (2058, 4)
    assert np.count_nonzero(~lattice.boundary_mask) == 216
    assert np.unique(np.sort(lattice.tetrahedra, axis=1), axis=0).shape[0] == 2058


def test_exact_gradient_rejects_large_shear() -> None:
    tetrahedron = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    shear = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert exact_gradient_frobenius_squared(tetrahedron, shear) == Fraction(9, 4)
    assert Fraction(9, 4) > DEFAULT_KAPPA * DEFAULT_KAPPA


def test_lift_is_deterministic_fixed_boundary_and_certified() -> None:
    lattice = _lattice()
    points = np.asarray(
        [[x, y, z] for x in (-0.4, 0.0, 0.4) for y in (-0.4, 0.0, 0.4) for z in (-0.4, 0.0, 0.4)]
    )
    proposal = np.zeros_like(points)
    proposal[:, 0] = 0.01 * (1.0 - points[:, 0] ** 2) * (1.0 - points[:, 1] ** 2)
    first = fit_and_certify_bilipschitz_step(
        lattice, points, proposal, minimum_retained_displacement_ratio=0.0
    )
    second = fit_and_certify_bilipschitz_step(
        lattice, points, proposal, minimum_retained_displacement_ratio=0.0
    )
    assert first.status == "pass"
    assert first.decision_sha256 == second.decision_sha256
    assert np.array_equal(first.accepted_controls, second.accepted_controls)
    assert np.all(first.accepted_controls[lattice.boundary_mask] == 0.0)
    maximum, _ = exact_max_gradient_frobenius_squared(lattice, first.accepted_controls)
    assert maximum <= DEFAULT_KAPPA * DEFAULT_KAPPA


def test_exact_refinement_is_replayable_watertight_and_parent_bound() -> None:
    lattice = _lattice()
    mesh = trimesh.creation.box(extents=(1.2, 1.0, 0.8))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    first = refine_surface_to_lattice(lattice, vertices, faces)
    second = refine_surface_to_lattice(lattice, vertices, faces)
    assert first.surface_sha256 == second.surface_sha256
    assert first.provenance_sha256 == second.provenance_sha256
    assert first.faces.shape[0] > faces.shape[0]
    assert np.unique(first.parent_face_indices).size == faces.shape[0]
    assert first.corner_barycentric_text.shape == (first.faces.shape[0], 3, 3)
    for record in first.corner_barycentric_text.reshape(-1, 3):
        assert sum(Fraction(value) for value in record) == 1
    refined_mesh = trimesh.Trimesh(
        vertices=first.reference_vertices, faces=first.faces, process=False
    )
    assert refined_mesh.is_watertight
    assert int(refined_mesh.euler_number) == 2
    report = parent_area_path_report(
        vertices, faces, first, first.mapped_vertices(lattice, np.zeros_like(lattice.vertices))
    )
    assert report["status"] == "pass"
    assert np.isclose(report["minimum_signed_parent_area_ratio"], 1.0)
    assert np.isclose(report["minimum_unsigned_parent_area_ratio"], 1.0)


def test_registered_controls_pass() -> None:
    report = run_bilipschitz_controls()
    assert report["status"] == "pass"
