from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert (..., 3) Rodrigues vectors to rotation matrices."""
    theta = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    # torch.sinc uses sin(pi*x)/(pi*x), giving the analytic small-angle
    # limits without the singular zero-axis normalization.
    sine_over_theta = torch.sinc(theta / math.pi)[..., None]
    one_minus_cosine_over_theta_squared = (0.5 * torch.sinc(theta / (2.0 * math.pi)).square())[
        ..., None
    ]
    return eye + sine_over_theta * skew + one_minus_cosine_over_theta_squared * (skew @ skew)


def world_to_camera(points: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    """Apply the OpenCV world-to-camera convention: Xc = R Xw + t."""
    return torch.matmul(points, rotation.transpose(-1, -2)) + translation.unsqueeze(-2)


def camera_to_world(points: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    """Invert :func:`world_to_camera`."""
    return torch.matmul(points - translation.unsqueeze(-2), rotation)


def project_points(points_camera: Tensor, intrinsics: Tensor, eps: float = 1e-6) -> Tensor:
    """Project OpenCV camera-space points (+x right, +y down, +z forward)."""
    z = points_camera[..., 2:3].clamp_min(eps)
    normalized = points_camera[..., :2] / z
    fx = intrinsics[..., 0, 0].unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)
    u = normalized[..., 0] * fx + cx
    v = normalized[..., 1] * fy + cy
    return torch.stack((u, v), dim=-1)


def make_intrinsics(
    focal_length_px: float | Tensor,
    principal_point_px: tuple[float, float] | list[float] | Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    focal = torch.as_tensor(focal_length_px, dtype=dtype, device=device)
    principal = torch.as_tensor(principal_point_px, dtype=dtype, device=device)
    result = torch.zeros((*focal.shape, 3, 3), dtype=dtype, device=device)
    result[..., 0, 0] = focal
    result[..., 1, 1] = focal
    result[..., 0, 2] = principal[..., 0]
    result[..., 1, 2] = principal[..., 1]
    result[..., 2, 2] = 1.0
    return result


def resize_intrinsics(
    intrinsics: Tensor,
    source_image_size: tuple[int, int],
    output_image_size: tuple[int, int],
) -> Tensor:
    """Scale intrinsics between explicit ``(height, width)`` image sizes."""
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end in shape (3, 3)")
    source_height, source_width = source_image_size
    output_height, output_width = output_image_size
    if min(source_height, source_width, output_height, output_width) <= 0:
        raise ValueError("source and output image dimensions must be positive")
    scale = intrinsics.new_tensor((output_width / source_width, output_height / source_height))
    resized = intrinsics.clone()
    resized[..., 0, 0] = resized[..., 0, 0] * scale[0]
    resized[..., 0, 2] = resized[..., 0, 2] * scale[0]
    resized[..., 1, 1] = resized[..., 1, 1] * scale[1]
    resized[..., 1, 2] = resized[..., 1, 2] * scale[1]
    return resized


def transform_intrinsics_for_crop(
    intrinsics: np.ndarray,
    crop_xywh: tuple[float, float, float, float],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Map full-image intrinsics into a cropped and resized image.

    ``output_size`` is ``(width, height)``. No hidden half-pixel offset is added;
    OpenCV and the renderer therefore use the same pixel-center convention.
    """
    x, y, width, height = crop_xywh
    output_width, output_height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("crop width and height must be positive")
    scale = np.array(
        [
            [output_width / width, 0.0, -x * output_width / width],
            [0.0, output_height / height, -y * output_height / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scale @ np.asarray(intrinsics, dtype=np.float64)


def rotation_angle_degrees(first: Tensor, second: Tensor) -> Tensor:
    relative = first.transpose(-1, -2) @ second
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def yaw_matrix(degrees: float, *, dtype: torch.dtype = torch.float32) -> Tensor:
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=dtype,
    )
