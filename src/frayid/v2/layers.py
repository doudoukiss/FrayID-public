from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ClippedImplicitLayer(nn.Module):
    """Open surface h(x)=0 restricted to q(x)>=0 with explicit boundary q=0."""

    def __init__(
        self, support_field: nn.Module, clipping_field: nn.Module, *, layer_id: str
    ) -> None:
        super().__init__()
        if not layer_id:
            raise ValueError("layer_id is required")
        self.support_field = support_field
        self.clipping_field = clipping_field
        self.layer_id = layer_id

    def support_values(self, points: Tensor) -> Tensor:
        values: Tensor = self.support_field(points)
        return values

    def clipping_values(self, points: Tensor) -> Tensor:
        values: Tensor = self.clipping_field(points)
        return values

    def active_weight(self, points: Tensor, *, sharpness: float = 50.0) -> Tensor:
        if sharpness <= 0:
            raise ValueError("clipping sharpness must be positive")
        return torch.sigmoid(self.clipping_values(points) * sharpness)


class LayeredCanonicalModel(nn.Module):
    def __init__(
        self,
        body_field: nn.Module,
        surface_layers: Mapping[str, ClippedImplicitLayer],
    ) -> None:
        super().__init__()
        if not surface_layers:
            raise ValueError("authoritative layered model requires at least one surface layer")
        self.body_field = body_field
        self.surface_layers = nn.ModuleDict(surface_layers)

    def body_values(self, points: Tensor) -> Tensor:
        values: Tensor = self.body_field(points)
        return values

    def layer_values(self, points: Tensor) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        for name, module in self.surface_layers.items():
            if not isinstance(module, ClippedImplicitLayer):
                raise TypeError(f"unexpected layer module for {name}: {type(module)}")
            values[name] = module.support_values(points)
        return values

    def visibility_ownership(self, signed_ray_distances: dict[str, Tensor]) -> Tensor:
        names = ["body", *self.surface_layers.keys()]
        if set(signed_ray_distances) != set(names):
            raise ValueError("visibility ownership requires body and every registered layer")
        distances = torch.stack([signed_ray_distances[name] for name in names], dim=-1)
        valid = torch.any(distances >= 0, dim=-1)
        ownership = torch.argmin(torch.where(distances >= 0, distances, torch.inf), dim=-1)
        return torch.where(valid, ownership, torch.full_like(ownership, -1))


def body_garment_nonpenetration_loss(
    body_sdf_values_at_garment: Tensor,
    *,
    contact_band: float,
) -> Tensor:
    if contact_band < 0:
        raise ValueError("contact band cannot be negative")
    penetration = F.relu(-(body_sdf_values_at_garment + contact_band))
    return penetration.square().mean()


def layer_order_loss(front_depth: Tensor, back_depth: Tensor, *, minimum_gap: float) -> Tensor:
    if front_depth.shape != back_depth.shape or minimum_gap < 0:
        raise ValueError("layer-order depth contract is invalid")
    return F.relu(front_depth + minimum_gap - back_depth).square().mean()


def bounded_residual_deformation(
    raw_displacement: Tensor,
    *,
    maximum_norm: float,
) -> Tensor:
    if raw_displacement.shape[-1] != 3 or maximum_norm <= 0:
        raise ValueError("bounded layer deformation contract is invalid")
    norm = torch.linalg.vector_norm(raw_displacement, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.tanh(norm / maximum_norm) * maximum_norm / norm
    result: Tensor = raw_displacement * scale
    return result
