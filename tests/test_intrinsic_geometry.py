from __future__ import annotations

import torch

from frayid.eulerian_reconstruction import public_eulerian_fixture
from frayid.intrinsic_geometry import (
    IntrinsicGeometryTransform,
    project_intrinsic_step,
)


def test_intrinsic_transform_is_full_rank_spd_and_round_trips_public_surface() -> None:
    fixture = public_eulerian_fixture()
    field = fixture.initial_field()
    vertices = field.surface_vertices().detach().double()
    transform = IntrinsicGeometryTransform.from_mesh(vertices, field.surface_faces)
    report = transform.report(vertices)
    assert report.status == "pass"
    assert report.rank == vertices.shape[0]
    assert report.minimum_eigenvalue > 0.0
    assert report.condition_number < 100.0
    assert report.relative_round_trip_error <= 1.0e-12
    assert report.relative_solve_residual <= 1.0e-12
    assert transform.encode(vertices).numel() == vertices.numel()


def test_intrinsic_transform_retains_translation_and_local_deformation() -> None:
    fixture = public_eulerian_fixture()
    field = fixture.initial_field()
    vertices = field.surface_vertices().detach().double()
    transform = IntrinsicGeometryTransform.from_mesh(vertices, field.surface_faces)
    translation = vertices + vertices.new_tensor([0.03, -0.02, 0.01])
    local = vertices.clone()
    local[17] += vertices.new_tensor([0.01, -0.015, 0.02])
    torch.testing.assert_close(
        transform.decode(transform.encode(translation)), translation, rtol=1.0e-12, atol=1.0e-12
    )
    torch.testing.assert_close(
        transform.decode(transform.encode(local)), local, rtol=1.0e-12, atol=1.0e-12
    )


def test_intrinsic_projection_uses_shared_complete_surface_path_gate() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    transform = IntrinsicGeometryTransform.from_mesh(vertices, faces)
    previous = transform.encode(vertices)
    coordinates = torch.nn.Parameter(previous.clone())
    proposed = vertices.clone()
    proposed[2, 1] = -1.0
    with torch.no_grad():
        coordinates.copy_(transform.encode(proposed))
    result = project_intrinsic_step(
        transform, coordinates, previous, vertices, faces, maximum_backtracks=8
    )
    assert result.rejected is False
    assert 0.0 < result.accepted_scale < 0.5
    assert result.certificate.status == "pass"
