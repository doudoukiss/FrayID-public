from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt  # type: ignore[import-untyped]
from torch import Tensor, nn

from frayid.camera import project_points
from frayid.renderer import (
    differentiable_boundary_loss,
    normal_cosine_loss,
    silhouette_loss,
)


class ScalarField(Protocol):
    def __call__(self, points: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class VisualHullGrid:
    values: Tensor
    extent: float

    @property
    def resolution(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class NeuSRender:
    silhouette: Tensor
    normals: Tensor
    expected_depth: Tensor
    accumulated_weight: Tensor


def _check_grid(values: Tensor, extent: float) -> None:
    if values.ndim != 3 or len(set(values.shape)) != 1 or values.shape[0] < 3:
        raise ValueError("SDF grid must be cubic with resolution at least three")
    if extent <= 0:
        raise ValueError("SDF extent must be positive")
    if not torch.isfinite(values).all():
        raise ValueError("SDF grid values must be finite")


def trilinear_grid_sample(values: Tensor, points: Tensor, *, extent: float) -> Tensor:
    """Sample an `[x,y,z]` scalar grid without changing its axis convention."""
    _check_grid(values, extent)
    if points.shape[-1] != 3 or not torch.isfinite(points).all():
        raise ValueError("sample points must be finite and end in dimension three")
    resolution = values.shape[0]
    flat = points.reshape(-1, 3)
    scaled = ((flat / extent + 1.0) * 0.5 * (resolution - 1)).clamp(0.0, float(resolution - 1))
    lower = torch.floor(scaled).to(torch.long)
    upper = (lower + 1).clamp_max(resolution - 1)
    fraction = scaled - lower.to(scaled.dtype)
    result = torch.zeros(flat.shape[0], dtype=values.dtype, device=values.device)
    for x_side in (0, 1):
        for y_side in (0, 1):
            for z_side in (0, 1):
                x_index = upper[:, 0] if x_side else lower[:, 0]
                y_index = upper[:, 1] if y_side else lower[:, 1]
                z_index = upper[:, 2] if z_side else lower[:, 2]
                x_weight = fraction[:, 0] if x_side else 1.0 - fraction[:, 0]
                y_weight = fraction[:, 1] if y_side else 1.0 - fraction[:, 1]
                z_weight = fraction[:, 2] if z_side else 1.0 - fraction[:, 2]
                result = result + (
                    values[x_index, y_index, z_index] * x_weight * y_weight * z_weight
                )
    return result.reshape(points.shape[:-1])


def visual_hull_sdf(
    silhouettes: Tensor,
    intrinsics: Tensor,
    rotations: Tensor,
    translations: Tensor,
    *,
    resolution: int,
    extent: float,
    threshold: float = 0.5,
) -> VisualHullGrid:
    """Build a deterministic intersection-of-silhouette signed-distance grid."""
    if silhouettes.ndim != 3:
        raise ValueError("silhouettes must have shape [V, H, W]")
    view_count, height, width = silhouettes.shape
    if intrinsics.shape not in ((3, 3), (view_count, 3, 3)):
        raise ValueError("intrinsics must be shared or have one matrix per view")
    if rotations.shape != (view_count, 3, 3) or translations.shape != (view_count, 3):
        raise ValueError("camera transforms must have one entry per view")
    if resolution < 3 or extent <= 0:
        raise ValueError("visual-hull grid contract is invalid")
    if not 0.0 < threshold < 1.0:
        raise ValueError("visual-hull threshold must lie strictly between zero and one")
    tensors = (silhouettes, intrinsics, rotations, translations)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("visual-hull evidence must be finite")

    coordinates = torch.linspace(
        -extent,
        extent,
        resolution,
        dtype=silhouettes.dtype,
        device=silhouettes.device,
    )
    xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    inside = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for view in range(view_count):
        camera = points @ rotations[view].T + translations[view]
        matrix = intrinsics if intrinsics.ndim == 2 else intrinsics[view]
        pixels = project_points(camera, matrix)
        normalized_x = 2.0 * (pixels[:, 0] + 0.5) / width - 1.0
        normalized_y = 2.0 * (pixels[:, 1] + 0.5) / height - 1.0
        grid = torch.stack((normalized_x, normalized_y), dim=-1).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            silhouettes[view][None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(-1)
        visible = (
            (camera[:, 2] > 0)
            & (normalized_x >= -1.0)
            & (normalized_x <= 1.0)
            & (normalized_y >= -1.0)
            & (normalized_y <= 1.0)
        )
        inside = inside & visible & (sampled >= threshold)
    occupancy = inside.reshape(resolution, resolution, resolution)
    if not torch.any(occupancy):
        raise ValueError("silhouette intersection produced an empty visual hull")
    boundary = torch.zeros_like(occupancy)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    if torch.any(occupancy & boundary):
        raise ValueError("visual hull touches the fixed outer boundary")
    occupancy_numpy = occupancy.detach().cpu().numpy()
    outside = distance_transform_edt(~occupancy_numpy)
    inside_distance = distance_transform_edt(occupancy_numpy)
    pitch = 2.0 * extent / (resolution - 1)
    signed = (outside - inside_distance) * pitch
    values = torch.as_tensor(signed, dtype=silhouettes.dtype, device=silhouettes.device)
    return VisualHullGrid(values=values, extent=extent)


class MultiresolutionHashEncoding(nn.Module):
    def __init__(
        self,
        *,
        level_count: int = 8,
        features_per_level: int = 2,
        base_resolution: int = 8,
        maximum_resolution: int = 96,
        table_size: int = 2**15,
        seed: int = 25,
    ) -> None:
        super().__init__()
        if level_count < 1 or features_per_level < 1 or table_size < 8:
            raise ValueError("hash-encoding dimensions are invalid")
        if base_resolution < 2 or maximum_resolution < base_resolution:
            raise ValueError("hash-encoding resolutions are invalid")
        resolutions: tuple[int, ...]
        if level_count == 1:
            resolutions = (maximum_resolution,)
        else:
            growth = math.exp(math.log(maximum_resolution / base_resolution) / (level_count - 1))
            resolutions = tuple(
                max(2, round(base_resolution * growth**level)) for level in range(level_count)
            )
        self.resolutions = resolutions
        self.features_per_level = features_per_level
        self.table_size = table_size
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        initial = torch.empty(level_count, table_size, features_per_level)
        initial.uniform_(-1.0e-4, 1.0e-4, generator=generator)
        self.tables = nn.Parameter(initial)

    @property
    def output_dimension(self) -> int:
        return len(self.resolutions) * self.features_per_level

    def _hash(self, indices: Tensor) -> Tensor:
        x, y, z = indices.unbind(-1)
        return torch.remainder(
            torch.bitwise_xor(torch.bitwise_xor(x * 73_856_093, y * 19_349_663), z * 83_492_791),
            self.table_size,
        )

    def forward(self, normalized_points: Tensor) -> Tensor:
        if normalized_points.shape[-1] != 3:
            raise ValueError("hash-encoding points must end in dimension three")
        flat = normalized_points.reshape(-1, 3).clamp(0.0, 1.0)
        encoded: list[Tensor] = []
        for level, resolution in enumerate(self.resolutions):
            scaled = flat * resolution
            lower = torch.floor(scaled).to(torch.long)
            fraction = scaled - lower.to(scaled.dtype)
            value = torch.zeros(
                flat.shape[0],
                self.features_per_level,
                dtype=self.tables.dtype,
                device=self.tables.device,
            )
            for x_side in (0, 1):
                for y_side in (0, 1):
                    for z_side in (0, 1):
                        offset = lower.new_tensor((x_side, y_side, z_side))
                        corner = lower + offset
                        weight = (
                            (fraction[:, 0] if x_side else 1.0 - fraction[:, 0])
                            * (fraction[:, 1] if y_side else 1.0 - fraction[:, 1])
                            * (fraction[:, 2] if z_side else 1.0 - fraction[:, 2])
                        )
                        value = value + self.tables[level, self._hash(corner)] * weight[:, None]
            encoded.append(value)
        result = torch.cat(encoded, dim=-1)
        return result.reshape(*normalized_points.shape[:-1], self.output_dimension)


class NormalIntegrableNeuralSDF(nn.Module):
    visual_hull_values: Tensor

    def __init__(
        self,
        visual_hull: VisualHullGrid,
        *,
        hidden_width: int = 64,
        hidden_layers: int = 2,
        maximum_hash_resolution: int = 96,
    ) -> None:
        super().__init__()
        _check_grid(visual_hull.values, visual_hull.extent)
        if hidden_width < 4 or hidden_layers < 1:
            raise ValueError("SDF network dimensions are invalid")
        self.extent = float(visual_hull.extent)
        self.register_buffer("visual_hull_values", visual_hull.values.clone())
        self.encoding = MultiresolutionHashEncoding(maximum_resolution=maximum_hash_resolution)
        layers: list[nn.Module] = []
        input_width = self.encoding.output_dimension + 3
        for layer in range(hidden_layers):
            layers.append(nn.Linear(input_width if layer == 0 else hidden_width, hidden_width))
            layers.append(nn.Softplus(beta=100.0))
        output = nn.Linear(hidden_width, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.residual = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        if points.shape[-1] != 3:
            raise ValueError("SDF points must end in dimension three")
        normalized = (points / self.extent + 1.0) * 0.5
        encoded = self.encoding(normalized)
        residual: Tensor = self.residual(torch.cat((normalized, encoded), dim=-1)).squeeze(-1)
        base: Tensor = trilinear_grid_sample(self.visual_hull_values, points, extent=self.extent)
        result: Tensor = base + residual
        return result


def spatial_sdf_gradient(field: ScalarField, points: Tensor, *, create_graph: bool) -> Tensor:
    if points.shape[-1] != 3:
        raise ValueError("SDF gradient points must end in dimension three")
    differentiable = points if points.requires_grad else points.detach().requires_grad_(True)
    values = field(differentiable)
    if values.shape != differentiable.shape[:-1]:
        raise ValueError("scalar field returned an incompatible shape")
    gradient: Tensor = torch.autograd.grad(
        values.sum(), differentiable, create_graph=create_graph, retain_graph=create_graph
    )[0]
    if not torch.isfinite(gradient).all():
        raise RuntimeError("SDF gradient is non-finite")
    return gradient


def eikonal_loss(field: ScalarField, points: Tensor) -> Tensor:
    gradient = spatial_sdf_gradient(field, points, create_graph=True)
    result: Tensor = (torch.linalg.vector_norm(gradient, dim=-1) - 1.0).square().mean()
    return result


def transport_normals_inverse_transpose(
    normals: Tensor, jacobians: Tensor, *, determinant_floor: float = 1.0e-8
) -> Tensor:
    if normals.shape[-1] != 3 or jacobians.shape[-2:] != (3, 3):
        raise ValueError("normal transport expects [...,3] normals and [...,3,3] Jacobians")
    if determinant_floor <= 0:
        raise ValueError("determinant floor must be positive")
    determinant = torch.linalg.det(jacobians)
    if not torch.isfinite(determinant).all() or torch.any(determinant.abs() <= determinant_floor):
        raise ValueError("normal transport Jacobian is singular or non-finite")
    transformed = torch.linalg.solve(jacobians.transpose(-2, -1), normals.unsqueeze(-1)).squeeze(-1)
    return F.normalize(transformed, dim=-1, eps=1.0e-8)


def camera_rays(
    intrinsics: Tensor,
    rotation: Tensor,
    translation: Tensor,
    image_size: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    if intrinsics.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("camera ray matrices have invalid shapes")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("camera ray image size must be positive")
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=intrinsics.dtype, device=intrinsics.device) + 0.5,
        torch.arange(width, dtype=intrinsics.dtype, device=intrinsics.device) + 0.5,
        indexing="ij",
    )
    direction_camera = torch.stack(
        (
            (xx - intrinsics[0, 2]) / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(xx),
        ),
        dim=-1,
    )
    direction_world = F.normalize(direction_camera @ rotation, dim=-1)
    origin_world = (-translation) @ rotation
    origins = origin_world.expand(height, width, 3)
    return origins, direction_world


def neus_interval_weights(sdf_samples: Tensor, inverse_sharpness: Tensor | float) -> Tensor:
    if sdf_samples.ndim < 1 or sdf_samples.shape[-1] < 2:
        raise ValueError("NeuS needs at least two ordered SDF samples per ray")
    sharpness = torch.as_tensor(
        inverse_sharpness, dtype=sdf_samples.dtype, device=sdf_samples.device
    )
    if sharpness.numel() != 1 or not torch.isfinite(sharpness) or sharpness <= 0:
        raise ValueError("NeuS inverse sharpness must be one positive finite scalar")
    cdf = torch.sigmoid(sdf_samples * sharpness)
    previous, following = cdf[..., :-1], cdf[..., 1:]
    alpha = ((previous - following) / previous.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1.0e-7), dim=-1),
        dim=-1,
    )[..., :-1]
    return alpha * transmittance


def hierarchical_depth_samples(
    boundary_depths: Tensor,
    interval_weights: Tensor,
    *,
    sample_count: int,
) -> Tensor:
    """Deterministically invert an interval PDF for NeuS fine samples."""
    if boundary_depths.ndim != 1 or boundary_depths.numel() < 2:
        raise ValueError("hierarchical boundary depths must be one ordered vector")
    if interval_weights.shape[-1] != boundary_depths.numel() - 1:
        raise ValueError("hierarchical interval weights do not match the depth bins")
    if sample_count < 1 or not torch.all(boundary_depths[1:] > boundary_depths[:-1]):
        raise ValueError("hierarchical depth sampling contract is invalid")
    weights = interval_weights.detach().clamp_min(0.0) + 1.0e-8
    probability = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(probability, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1)
    quantiles = (
        torch.arange(sample_count, dtype=weights.dtype, device=weights.device) + 0.5
    ) / sample_count
    quantiles = quantiles.expand(*weights.shape[:-1], sample_count).contiguous()
    upper = torch.searchsorted(cdf.contiguous(), quantiles, right=True).clamp(
        1, boundary_depths.numel() - 1
    )
    lower = upper - 1
    lower_cdf = torch.gather(cdf, -1, lower)
    upper_cdf = torch.gather(cdf, -1, upper)
    depths = boundary_depths.expand(*weights.shape[:-1], boundary_depths.numel())
    lower_depth = torch.gather(depths, -1, lower)
    upper_depth = torch.gather(depths, -1, upper)
    fraction = (quantiles - lower_cdf) / (upper_cdf - lower_cdf).clamp_min(1.0e-8)
    return lower_depth + fraction * (upper_depth - lower_depth)


def render_neus_sdf(
    field: ScalarField,
    ray_origins: Tensor,
    ray_directions: Tensor,
    *,
    near: float,
    far: float,
    sample_count: int,
    hierarchical_sample_count: int = 0,
    inverse_sharpness: Tensor | float,
    deformation_jacobian: Tensor,
    create_graph: bool,
    ray_chunk_size: int = 4096,
) -> NeuSRender:
    if ray_origins.shape != ray_directions.shape or ray_origins.shape[-1] != 3:
        raise ValueError("ray origins and directions must have the same [...,3] shape")
    if (
        near < 0
        or far <= near
        or sample_count < 4
        or hierarchical_sample_count < 0
        or ray_chunk_size < 1
    ):
        raise ValueError("NeuS ray sampling contract is invalid")
    if not torch.isfinite(ray_origins).all() or not torch.isfinite(ray_directions).all():
        raise ValueError("NeuS rays must be finite")
    image_shape = ray_origins.shape[:-1]
    flat_origins = ray_origins.reshape(-1, 3)
    flat_directions = F.normalize(ray_directions.reshape(-1, 3), dim=-1)
    depths = torch.linspace(
        near,
        far,
        sample_count + 1,
        dtype=flat_origins.dtype,
        device=flat_origins.device,
    )
    silhouettes: list[Tensor] = []
    normal_maps: list[Tensor] = []
    expected_depths: list[Tensor] = []
    accumulated: list[Tensor] = []
    for origins, directions in zip(
        flat_origins.split(ray_chunk_size),  # type: ignore[no-untyped-call]
        flat_directions.split(ray_chunk_size),  # type: ignore[no-untyped-call]
        strict=True,
    ):
        boundary_points = origins[:, None] + directions[:, None] * depths[None, :, None]
        boundary_sdf = field(boundary_points)
        weights = neus_interval_weights(boundary_sdf, inverse_sharpness)
        render_depths = depths.expand(origins.shape[0], depths.numel())
        if hierarchical_sample_count:
            fine_depths = hierarchical_depth_samples(
                depths,
                weights,
                sample_count=hierarchical_sample_count,
            )
            render_depths = torch.sort(torch.cat((render_depths, fine_depths), dim=-1), dim=-1)[0]
            boundary_points = origins[:, None] + directions[:, None] * render_depths[..., None]
            boundary_sdf = field(boundary_points)
            weights = neus_interval_weights(boundary_sdf, inverse_sharpness)
        midpoint_depths = 0.5 * (render_depths[..., :-1] + render_depths[..., 1:])
        midpoint_points = origins[:, None] + directions[:, None] * midpoint_depths[..., None]
        gradients = spatial_sdf_gradient(field, midpoint_points, create_graph=create_graph)
        canonical_normals = F.normalize(gradients, dim=-1, eps=1.0e-8)
        transported = transport_normals_inverse_transpose(canonical_normals, deformation_jacobian)
        sapiens_normals = transported * transported.new_tensor((1.0, -1.0, -1.0))
        normal = F.normalize((weights[..., None] * sapiens_normals).sum(dim=-2), dim=-1, eps=1.0e-8)
        opacity = weights.sum(dim=-1).clamp(0.0, 1.0)
        silhouettes.append(opacity)
        accumulated.append(opacity)
        normal_maps.append(normal)
        expected_depths.append((weights * midpoint_depths).sum(dim=-1))
    silhouette = torch.cat(silhouettes).reshape(image_shape)
    normals = torch.cat(normal_maps).reshape(*image_shape, 3)
    expected_depth = torch.cat(expected_depths).reshape(image_shape)
    accumulated_weight = torch.cat(accumulated).reshape(image_shape)
    return NeuSRender(silhouette, normals, expected_depth, accumulated_weight)


def normal_integrable_image_loss(
    rendered: NeuSRender,
    target_silhouette: Tensor,
    target_normals: Tensor,
    *,
    silhouette_weight: float = 1.0,
    boundary_weight: float = 0.5,
    normal_weight: float = 0.25,
    integrability_weight: float = 0.1,
) -> Tensor:
    if rendered.silhouette.shape != target_silhouette.shape:
        raise ValueError("E25 rendered and target silhouettes must match")
    if rendered.normals.shape != target_normals.shape:
        raise ValueError("E25 rendered and target normals must match")
    weights = (silhouette_weight, boundary_weight, normal_weight, integrability_weight)
    if any(weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("E25 image-loss weights must be finite and nonnegative")
    return (
        silhouette_weight * silhouette_loss(rendered.silhouette, target_silhouette)
        + boundary_weight * differentiable_boundary_loss(rendered.silhouette, target_silhouette)
        + normal_weight * normal_cosine_loss(rendered.normals, target_normals, target_silhouette)
        + integrability_weight
        * bilateral_normal_integrability_loss(
            rendered.normals,
            target_normals,
            target_silhouette,
        )
    )


def bilateral_normal_integrability_loss(
    predicted_gradient_normals: Tensor,
    observed_normals: Tensor,
    mask: Tensor,
    *,
    angular_scale: float = 0.25,
) -> Tensor:
    """Match local normal variation without smoothing across observed creases."""
    if predicted_gradient_normals.shape != observed_normals.shape:
        raise ValueError("predicted and observed normal maps must match")
    if predicted_gradient_normals.ndim != 3 or predicted_gradient_normals.shape[-1] != 3:
        raise ValueError("normal maps must have shape [H, W, 3]")
    if mask.shape != predicted_gradient_normals.shape[:2]:
        raise ValueError("normal-integrability mask has the wrong shape")
    if angular_scale <= 0:
        raise ValueError("normal-integrability angular scale must be positive")
    predicted = F.normalize(predicted_gradient_normals, dim=-1, eps=1.0e-8)
    observed = F.normalize(observed_normals, dim=-1, eps=1.0e-8)
    valid = mask > 0.5
    terms: list[Tensor] = []
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis] = slice(None, -1)
        second[axis] = slice(1, None)
        first_index = tuple(first)
        second_index = tuple(second)
        pair_valid = valid[first_index] & valid[second_index]
        if not torch.any(pair_valid):
            continue
        observed_first = observed[first_index]
        observed_second = observed[second_index]
        cosine = (observed_first * observed_second).sum(dim=-1).clamp(-1.0, 1.0)
        bilateral = torch.exp(-(1.0 - cosine) / angular_scale)
        predicted_difference = predicted[second_index] - predicted[first_index]
        observed_difference = observed_second - observed_first
        residual = (predicted_difference - observed_difference).square().sum(dim=-1)
        terms.append((bilateral[pair_valid] * residual[pair_valid]).mean())
    if not terms:
        return predicted_gradient_normals.sum() * 0.0
    return torch.stack(terms).mean()


def directional_normal_error_degrees(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if predicted.shape != target.shape or mask.shape != predicted.shape[:-1]:
        raise ValueError("directional normal metric shapes do not match")
    valid = mask > 0.5
    if not torch.any(valid):
        raise ValueError("directional normal metric has no valid pixels")
    first = F.normalize(predicted[valid], dim=-1, eps=1.0e-8)
    second = F.normalize(target[valid], dim=-1, eps=1.0e-8)
    cosine = (first * second).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))
