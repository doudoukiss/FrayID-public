from __future__ import annotations

import math

import torch
from torch import Tensor

from frayid.camera import project_points, resize_intrinsics
from frayid.geometry import vertex_normals


def sample_mesh_surface(
    vertices: Tensor, faces: Tensor, sample_count: int
) -> tuple[Tensor, Tensor]:
    """Deterministically sample triangle interiors and interpolate normals."""
    triangles = vertices[faces]
    areas = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
    ).norm(dim=-1)
    probabilities = areas / areas.sum().clamp_min(1e-8)
    indices = torch.multinomial(probabilities, sample_count, replacement=True)
    selected = triangles[indices]
    random = torch.rand((sample_count, 2), dtype=vertices.dtype, device=vertices.device)
    root_u = torch.sqrt(random[:, :1])
    barycentric = torch.cat(
        (1.0 - root_u, root_u * (1.0 - random[:, 1:]), root_u * random[:, 1:]), dim=-1
    )
    points = (selected * barycentric.unsqueeze(-1)).sum(dim=1)
    normals = vertex_normals(vertices, faces)
    sampled_normals = torch.nn.functional.normalize(
        (normals[faces[indices]] * barycentric.unsqueeze(-1)).sum(dim=1), dim=-1, eps=1e-8
    )
    return points, sampled_normals


def render_soft_mesh(
    vertices_camera: Tensor,
    faces: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    source_image_size: tuple[int, int],
    sigma_pixels: float = 1.75,
    sample_count: int = 2048,
    reference_sample_count: int | None = None,
    chunk_size: int = 256,
    depth_temperature_m: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Differentiable soft surface-splat silhouette and visible-normal renderer.

    This renderer is intentionally dependency-free and suited to smoke tests and
    coarse supervision. Fine training can switch to an accelerated rasterizer
    without changing the camera or loss contract.
    """
    if depth_temperature_m <= 0:
        raise ValueError("depth_temperature_m must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if reference_sample_count is not None and reference_sample_count <= 0:
        raise ValueError("reference_sample_count must be positive")
    points, normals = sample_mesh_surface(vertices_camera, faces, sample_count)
    visible = points[:, 2] > 1e-5
    points, normals = points[visible], normals[visible]
    if points.shape[0] == 0:
        height, width = image_size
        return (
            vertices_camera.new_zeros((height, width)),
            vertices_camera.new_zeros((height, width, 3)),
        )
    render_intrinsics = resize_intrinsics(intrinsics, source_image_size, image_size)
    projected = project_points(points, render_intrinsics)
    height, width = image_size
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=vertices_camera.dtype, device=vertices_camera.device),
        torch.arange(width, dtype=vertices_camera.dtype, device=vertices_camera.device),
        indexing="ij",
    )
    occupancy_sum = torch.zeros(
        (height, width), dtype=vertices_camera.dtype, device=vertices_camera.device
    )
    normal_sum = torch.zeros(
        (height, width, 3), dtype=vertices_camera.dtype, device=vertices_camera.device
    )
    # Sapiens2 camera-space normals use +x right, +y up, +z toward the
    # camera. OpenCV geometry uses +x right, +y down, +z away from it.
    sapiens_normals = normals * normals.new_tensor((1.0, -1.0, -1.0))
    minimum_depth = points[:, 2].amin().detach()
    depth_confidence = torch.exp(-(points[:, 2] - minimum_depth) / depth_temperature_m)
    # Each splat is a Monte Carlo surface sample. Without 1/N normalization,
    # opacity grows with sample_count and evaluation cannot converge. Preserve
    # the calibrated reference density while making larger estimates unbiased.
    sample_weight = (reference_sample_count or sample_count) / sample_count
    support_radius_squared = (3.0 * sigma_pixels) ** 2
    for point_chunk, normal_chunk, depth_chunk in zip(
        projected.split(chunk_size),  # type: ignore[no-untyped-call]
        sapiens_normals.split(chunk_size),  # type: ignore[no-untyped-call]
        depth_confidence.split(chunk_size),  # type: ignore[no-untyped-call]
        strict=True,
    ):
        dx = xx.unsqueeze(0) - point_chunk[:, 0, None, None]
        dy = yy.unsqueeze(0) - point_chunk[:, 1, None, None]
        squared_distance = dx.square() + dy.square()
        weights = torch.exp(-squared_distance / (2.0 * sigma_pixels**2))
        occupancy_sum = occupancy_sum + sample_weight * weights.sum(dim=0)
        visibility_weights = (
            weights * (squared_distance <= support_radius_squared) * depth_chunk[:, None, None]
        )
        normal_sum = normal_sum + sample_weight * torch.einsum(
            "nhw,nc->hwc", visibility_weights, normal_chunk
        )
    silhouette = 1.0 - torch.exp(-occupancy_sum)
    normal_map = torch.nn.functional.normalize(normal_sum, dim=-1, eps=1e-8)
    return silhouette, normal_map


def scaled_splat_parameters(
    image_size: tuple[int, int],
    *,
    reference_resolution: int,
    reference_sigma_pixels: float,
    reference_sample_count: int,
) -> tuple[float, int]:
    """Keep splat width and surface density invariant across render scales."""
    scale = max(image_size) / reference_resolution
    return (
        reference_sigma_pixels * scale,
        max(256, round(reference_sample_count * scale * scale)),
    )


def soft_silhouette_iou(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    intersection = (prediction * target).sum()
    union = (prediction + target - prediction * target).sum()
    return (intersection + eps) / (union + eps)


def silhouette_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return 1.0 - soft_silhouette_iou(prediction, target)


def differentiable_boundary_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Sobel edge agreement used as a differentiable boundary surrogate."""
    if prediction.shape != target.shape:
        raise ValueError("boundary prediction and target must have identical shapes")
    if prediction.ndim == 2:
        pred = prediction[None, None]
        truth = target[None, None]
    elif prediction.ndim == 3:
        pred = prediction[:, None]
        truth = target[:, None]
    else:
        raise ValueError("boundary tensors must have shape [H, W] or [B, H, W]")
    kernel_x = prediction.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = kernel_x.transpose(0, 1)
    pred_edge = torch.sqrt(
        torch.nn.functional.conv2d(pred, kernel_x[None, None], padding=1).square()
        + torch.nn.functional.conv2d(pred, kernel_y[None, None], padding=1).square()
        + 1e-8
    )
    truth_edge = torch.sqrt(
        torch.nn.functional.conv2d(truth, kernel_x[None, None], padding=1).square()
        + torch.nn.functional.conv2d(truth, kernel_y[None, None], padding=1).square()
        + 1e-8
    )
    return torch.nn.functional.l1_loss(pred_edge, truth_edge)


def normal_cosine_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return prediction.sum() * 0.0
    pred = torch.nn.functional.normalize(prediction[valid], dim=-1, eps=1e-8)
    truth = torch.nn.functional.normalize(target[valid], dim=-1, eps=1e-8)
    return (1.0 - (pred * truth).sum(dim=-1)).mean()


def normalized_boundary_error(prediction: Tensor, target: Tensor) -> float:
    """Symmetric boundary Chamfer distance normalized by image diagonal."""
    from scipy.ndimage import binary_erosion, distance_transform_edt  # type: ignore[import-untyped]

    pred = prediction.detach().cpu().numpy() > 0.5
    truth = target.detach().cpu().numpy() > 0.5
    pred_boundary = pred ^ binary_erosion(pred)
    truth_boundary = truth ^ binary_erosion(truth)
    if not pred_boundary.any() or not truth_boundary.any():
        return math.inf
    truth_distance = distance_transform_edt(~truth_boundary)
    pred_distance = distance_transform_edt(~pred_boundary)
    error = 0.5 * (truth_distance[pred_boundary].mean() + pred_distance[truth_boundary].mean())
    return float(error / math.hypot(*pred.shape))
