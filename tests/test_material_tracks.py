from __future__ import annotations

import pytest
import torch

from frayid.camera import make_intrinsics
from frayid.material_tracks import (
    MaterialTrackObservations,
    material_track_points,
    material_track_reprojection_loss,
    pseudo_huber,
)


def _observations(pixels: torch.Tensor) -> MaterialTrackObservations:
    return MaterialTrackObservations(
        face_indices=torch.tensor([0], dtype=torch.long),
        barycentric_coordinates=torch.tensor([[0.2, 0.3, 0.5]]),
        observed_pixels=pixels,
        observation_weights=torch.ones(pixels.shape[:2]),
        valid=torch.ones(pixels.shape[:2], dtype=torch.bool),
    )


def test_material_point_retains_fixed_face_and_barycentric_identity() -> None:
    vertices = torch.tensor(
        [
            [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
            [[1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [1.0, 1.0, 2.0]],
        ]
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    points = material_track_points(
        vertices,
        faces,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[0.2, 0.3, 0.5]]),
    )
    torch.testing.assert_close(points[:, 0], torch.tensor([[0.3, 0.5, 2.0], [1.3, 0.5, 2.0]]))


def test_zero_material_reprojection_has_finite_gradient() -> None:
    vertices = torch.tensor(
        [[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]],
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    intrinsics = make_intrinsics(40.0, (16.0, 16.0))
    point = material_track_points(
        vertices.detach(),
        faces,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[0.2, 0.3, 0.5]]),
    )
    pixels = torch.stack(
        (
            intrinsics[0, 0] * point[..., 0] / point[..., 2] + intrinsics[0, 2],
            intrinsics[1, 1] * point[..., 1] / point[..., 2] + intrinsics[1, 2],
        ),
        dim=-1,
    )
    loss = material_track_reprojection_loss(
        vertices,
        faces,
        intrinsics,
        _observations(pixels),
        source_image_size=(32, 32),
        robust_delta_fraction_of_diagonal=0.0025,
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert vertices.grad is not None
    assert bool(torch.isfinite(vertices.grad).all())


def test_material_reprojection_gradient_moves_shared_surface_point() -> None:
    vertices = torch.tensor(
        [[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]],
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    intrinsics = make_intrinsics(40.0, (16.0, 16.0))
    observed = torch.tensor([[[24.0, 28.0]]])
    loss = material_track_reprojection_loss(
        vertices,
        faces,
        intrinsics,
        _observations(observed),
        source_image_size=(32, 32),
        robust_delta_fraction_of_diagonal=0.0025,
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert vertices.grad is not None
    assert float(vertices.grad[:, :, 0].sum()) < 0
    assert float(vertices.grad[:, :, 1].sum()) < 0


def test_pseudo_huber_limits_outlier_influence() -> None:
    values = torch.tensor([0.001, 0.1], requires_grad=True)
    pseudo_huber(values, delta=0.0025).sum().backward()  # type: ignore[no-untyped-call]
    assert values.grad is not None
    assert float(values.grad.abs().max()) <= 1.0
    assert float(values.grad[1]) < 1.0


def test_material_track_validation_rejects_boundary_anchor() -> None:
    observations = MaterialTrackObservations(
        face_indices=torch.tensor([0], dtype=torch.long),
        barycentric_coordinates=torch.tensor([[0.01, 0.49, 0.50]]),
        observed_pixels=torch.zeros((1, 1, 2)),
        observation_weights=torch.ones((1, 1)),
        valid=torch.ones((1, 1), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="too close"):
        observations.validate(
            face_count=1,
            frame_count=1,
            minimum_barycentric_coordinate=0.05,
        )


def test_minibatch_validation_allows_temporarily_unobserved_track() -> None:
    observations = MaterialTrackObservations(
        face_indices=torch.tensor([0, 0], dtype=torch.long),
        barycentric_coordinates=torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]]),
        observed_pixels=torch.zeros((1, 2, 2)),
        observation_weights=torch.ones((1, 2)),
        valid=torch.tensor([[True, False]]),
    )
    with pytest.raises(ValueError, match="every material track"):
        observations.validate(face_count=1, frame_count=1)
    observations.validate(
        face_count=1,
        frame_count=1,
        require_each_track_observed=False,
    )
