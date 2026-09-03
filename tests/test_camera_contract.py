from __future__ import annotations

import cv2
import numpy as np
import torch

from frayid.camera import (
    axis_angle_to_matrix,
    camera_to_world,
    make_intrinsics,
    project_points,
    transform_intrinsics_for_crop,
    world_to_camera,
    yaw_matrix,
)


def test_axis_angle_zero_has_finite_correct_gradient() -> None:
    axis_angle = torch.zeros(3, requires_grad=True)
    rotation = axis_angle_to_matrix(axis_angle)
    rotation[0, 1].backward()
    assert axis_angle.grad is not None
    assert torch.isfinite(axis_angle.grad).all()
    torch.testing.assert_close(axis_angle.grad, torch.tensor([0.0, 0.0, -1.0]))


def test_world_camera_round_trip() -> None:
    points = torch.tensor([[0.2, -0.1, 2.0], [1.0, 0.5, 4.0]])
    rotation = yaw_matrix(27.0)
    translation = torch.tensor([0.1, -0.2, 0.3])
    camera = world_to_camera(points, rotation, translation)
    assert torch.allclose(camera_to_world(camera, rotation, translation), points, atol=1e-6)


def test_projection_matches_opencv() -> None:
    points = np.asarray([[0.1, 0.2, 2.0], [-0.4, 0.1, 3.0]], dtype=np.float64)
    intrinsics = np.asarray([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    expected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsics, np.zeros(5))
    actual = project_points(torch.tensor(points), torch.tensor(intrinsics)).numpy()
    np.testing.assert_allclose(actual, expected[:, 0], atol=1e-7)


def test_crop_intrinsics_preserve_projected_location() -> None:
    intrinsics = np.asarray([[900.0, 0.0, 360.0], [0.0, 900.0, 560.0], [0.0, 0.0, 1.0]])
    point = torch.tensor([[0.1, -0.2, 2.5]], dtype=torch.float64)
    original = project_points(point, torch.tensor(intrinsics))[0].numpy()
    crop = (60.0, 160.0, 600.0, 800.0)
    output = (300, 400)
    transformed = transform_intrinsics_for_crop(intrinsics, crop, output)
    projected = project_points(point, torch.tensor(transformed))[0].numpy()
    expected = np.asarray(
        [
            (original[0] - crop[0]) * output[0] / crop[2],
            (original[1] - crop[1]) * output[1] / crop[3],
        ]
    )
    np.testing.assert_allclose(projected, expected, atol=1e-7)


def test_root_rotation_equals_world_to_virtual_camera_rotation() -> None:
    points = torch.tensor([[0.2, 0.1, 2.0], [-0.3, 0.4, 2.5]])
    rotation = yaw_matrix(65.0)
    translation = torch.tensor([0.0, 0.0, 3.0])
    object_rotated = points @ rotation.T + translation
    virtual_camera = world_to_camera(points, rotation, translation)
    intrinsics = make_intrinsics(500.0, (128.0, 128.0))
    assert torch.allclose(
        project_points(object_rotated, intrinsics),
        project_points(virtual_camera, intrinsics),
        atol=1e-6,
    )
