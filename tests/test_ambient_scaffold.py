from __future__ import annotations

import hashlib
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from frayid.ambient_scaffold import (
    AmbientScaffoldV1,
    solve_harmonic_direction,
)
from frayid.certified_tet_path import (
    certify_exact_polynomial_path,
    certify_tet_step,
)
from frayid.global_path_controls import run_global_path_controls


def _regular_grid_scaffold() -> tuple[AmbientScaffoldV1, np.ndarray, np.ndarray]:
    side = 4
    coordinates = np.linspace(0.0, 1.0, side, dtype=np.float64)
    vertices = np.asarray(
        [(x, y, z) for x in coordinates for y in coordinates for z in coordinates],
        dtype=np.float64,
    )

    def index(x: int, y: int, z: int) -> int:
        return x * side * side + y * side + z

    tetrahedra: list[list[int]] = []
    for x in range(side - 1):
        for y in range(side - 1):
            for z in range(side - 1):
                v000 = index(x, y, z)
                v100 = index(x + 1, y, z)
                v010 = index(x, y + 1, z)
                v110 = index(x + 1, y + 1, z)
                v001 = index(x, y, z + 1)
                v101 = index(x + 1, y, z + 1)
                v011 = index(x, y + 1, z + 1)
                v111 = index(x + 1, y + 1, z + 1)
                tetrahedra.extend(
                    (
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                        [v000, v101, v100, v111],
                    )
                )
    tetrahedra_array = np.asarray(tetrahedra, dtype=np.int64)
    points = vertices[tetrahedra_array]
    determinants = np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )
    negative = determinants < 0.0
    tetrahedra_array[negative, :2] = tetrahedra_array[negative, 1::-1]
    points = vertices[tetrahedra_array]
    determinants = np.asarray(
        np.einsum(
            "ij,ij->i",
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            points[:, 3] - points[:, 0],
        ),
        dtype=np.float64,
    )
    interior_tet = tetrahedra_array[6 * ((1 * (side - 1) + 1) * (side - 1) + 1)]
    carrier_face = interior_tet[[0, 1, 2]][None, :]
    carrier_vertices = np.unique(carrier_face)
    source_vertices = vertices[carrier_vertices]
    source_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    barycentric = np.eye(3, dtype=np.float64)
    outer = np.any((vertices == 0.0) | (vertices == 1.0), axis=1)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(source_vertices, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(source_faces, dtype="<i8").tobytes())
    scaffold = AmbientScaffoldV1(
        vertices=vertices,
        tetrahedra=tetrahedra_array,
        carrier_faces=carrier_face,
        carrier_face_parent_indices=np.asarray([0], dtype=np.int64),
        carrier_vertex_indices=carrier_vertices,
        carrier_vertex_parent_faces=np.zeros(3, dtype=np.int64),
        carrier_vertex_parent_barycentrics=barycentric,
        fixed_source_faces=np.empty((0, 3), dtype=np.int64),
        outer_boundary_mask=outer,
        fixed_source_vertex_mask=np.zeros(vertices.shape[0], dtype=np.bool_),
        region_labels=np.zeros(tetrahedra_array.shape[0], dtype=np.int16),
        reference_determinants=determinants,
        bounds_lower=np.zeros(3, dtype=np.float64),
        bounds_upper=np.ones(3, dtype=np.float64),
        source_carrier_vertex_count=3,
        source_carrier_face_count=1,
        source_carrier_sha256=digest.hexdigest(),
        constructor_bindings={"fixture": "regular_grid"},
    )
    scaffold.validate()
    return scaffold, source_faces, source_vertices


def test_harmonic_direction_and_certified_shared_step_are_deterministic() -> None:
    scaffold, faces, _ = _regular_grid_scaffold()
    proposal = np.repeat(np.asarray([[0.01, -0.005, 0.002]], dtype=np.float64), 3, axis=0)
    first_direction = solve_harmonic_direction(scaffold, faces, proposal)
    second_direction = solve_harmonic_direction(scaffold, faces, proposal)
    assert np.array_equal(first_direction.displacement, second_direction.displacement)
    assert first_direction.normalized_residual <= 1e-10
    assert np.all(first_direction.displacement[scaffold.outer_boundary_mask] == 0.0)

    first = certify_tet_step(scaffold, faces, proposal, first_direction, timeout_seconds=None)
    second = certify_tet_step(scaffold, faces, proposal, second_direction, timeout_seconds=None)
    assert first.status == "pass"
    assert first.accepted_alpha == 1.0
    assert first.retained_displacement_ratio == pytest.approx(1.0)
    assert first.decision_sha256 == second.decision_sha256
    assert np.array_equal(first.accepted_vertices, second.accepted_vertices)


def test_ambient_scaffold_round_trip_is_hash_checked(tmp_path: Path) -> None:
    scaffold, _, _ = _regular_grid_scaffold()
    path = tmp_path / "scaffold.npz"
    scaffold.save(path)
    restored = AmbientScaffoldV1.load(path)
    assert restored.scaffold_sha256 == scaffold.scaffold_sha256
    assert np.array_equal(restored.tetrahedra, scaffold.tetrahedra)
    with pytest.raises(FileExistsError, match="immutable"):
        scaffold.save(path)


def test_invalid_boundary_and_nonconforming_carrier_are_rejected() -> None:
    scaffold, _, _ = _regular_grid_scaffold()
    invalid_boundary = scaffold.outer_boundary_mask.copy()
    invalid_boundary[np.flatnonzero(invalid_boundary)[0]] = False
    with pytest.raises(ValueError, match="boundary"):
        replace(scaffold, outer_boundary_mask=invalid_boundary).validate()

    interior = np.flatnonzero(~scaffold.outer_boundary_mask)
    nonconforming = np.asarray([interior[:3]], dtype=np.int64)
    with pytest.raises(ValueError, match="conforming"):
        replace(
            scaffold,
            carrier_faces=nonconforming,
            carrier_vertex_indices=np.unique(nonconforming),
        ).validate()


def test_positive_endpoints_do_not_hide_interior_inversion() -> None:
    report = certify_exact_polynomial_path([1, -5, 6, 0])
    assert report["endpoint_positive"] is True
    assert report["status"] == "fail"
    root = report["first_root_lower_bound"]
    assert root is not None
    lower = Fraction(root["numerator"], root["denominator"])
    assert lower < Fraction(1, 3)
    assert Fraction(1, 3) - lower <= Fraction(1, 2**64)


def test_first_singularity_selects_one_safe_dyadic_alpha() -> None:
    scaffold, faces, _ = _regular_grid_scaffold()
    proposal = np.repeat(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)
    harmonic = solve_harmonic_direction(scaffold, faces, proposal)
    step = certify_tet_step(scaffold, faces, proposal, harmonic, timeout_seconds=None)
    assert step.status == "pass"
    assert 0.25 <= step.accepted_alpha < 1.0
    assert step.accepted_alpha * 2**40 == round(step.accepted_alpha * 2**40)
    assert step.retained_displacement_ratio == pytest.approx(step.accepted_alpha)
    assert step.determinant_report["full_proposal_scan"]["exact_rational_fallback_count"] > 0


def test_certificate_timeout_is_unknown_and_never_passes() -> None:
    scaffold, faces, _ = _regular_grid_scaffold()
    proposal = np.repeat(np.asarray([[0.01, 0.0, 0.0]], dtype=np.float64), 3, axis=0)
    harmonic = solve_harmonic_direction(scaffold, faces, proposal)
    step = certify_tet_step(scaffold, faces, proposal, harmonic, timeout_seconds=0.0)
    assert step.status == "unknown"
    assert step.accepted_alpha == 0.0


def test_coordinatewise_velocity_truncation_can_create_collision() -> None:
    original_velocities = (Fraction(2), Fraction(2))
    truncated_velocities = (Fraction(2), Fraction(1, 2))
    assert Fraction(1) + original_velocities[1] - original_velocities[0] == 1
    collision_time = Fraction(1) / (truncated_velocities[0] - truncated_velocities[1])
    assert collision_time == Fraction(2, 3)


def test_registered_global_path_controls_pass_without_private_inputs() -> None:
    report = run_global_path_controls()
    assert report["status"] == "pass"
    assert report["uses_private_data"] is False
    assert all(report["checks"].values())
