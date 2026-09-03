from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn


class LearnableDenseSDF(nn.Module):
    """Inspectable trilinear canonical SDF initialized from an x/y/z NumPy grid."""

    origin: Tensor
    shape_xyz: Tensor

    def __init__(self, values_xyz: Tensor, origin: Tensor, pitch: float) -> None:
        super().__init__()
        if values_xyz.ndim != 3 or min(values_xyz.shape) < 2:
            raise ValueError("Dense SDF values must have shape [x, y, z] with all axes >= 2")
        if origin.shape != (3,) or pitch <= 0:
            raise ValueError("Dense SDF origin/pitch are invalid")
        # torch.grid_sample stores volumes as [N,C,z,y,x].
        self.values_zyx = nn.Parameter(values_xyz.permute(2, 1, 0)[None, None].contiguous())
        self.register_buffer("origin", origin.detach().clone())
        self.register_buffer(
            "shape_xyz",
            torch.tensor(values_xyz.shape, dtype=origin.dtype, device=origin.device),
        )
        self.pitch = float(pitch)

    @property
    def values_xyz(self) -> Tensor:
        return self.values_zyx[0, 0].permute(2, 1, 0)

    def forward(self, points: Tensor) -> Tensor:
        if points.shape[-1] != 3:
            raise ValueError("Dense SDF query points must end in xyz")
        flat = points.reshape(-1, 3)
        grid_indices = (flat - self.origin) / self.pitch
        normalized = 2.0 * grid_indices / (self.shape_xyz - 1.0) - 1.0
        query = normalized.reshape(1, -1, 1, 1, 3)
        sampled = torch.nn.functional.grid_sample(
            self.values_zyx,
            query,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0, :, 0, 0]
        return sampled.reshape(points.shape[:-1])

    def spatial_gradient(self, points: Tensor, *, create_graph: bool = True) -> Tensor:
        query = points if points.requires_grad else points.detach().requires_grad_(True)
        values = self(query)
        gradient = torch.autograd.grad(
            values.sum(), query, create_graph=create_graph, retain_graph=create_graph
        )[0]
        return gradient

    def finite_difference_gradient(self, points: Tensor, *, step: float | None = None) -> Tensor:
        """Field gradient with first-order autograd support for grid parameters."""
        epsilon = step or self.pitch
        if epsilon <= 0:
            raise ValueError("Finite-difference step must be positive")
        basis = torch.eye(3, dtype=points.dtype, device=points.device) * epsilon
        components = [
            (self(points + basis[axis]) - self(points - basis[axis])) / (2.0 * epsilon)
            for axis in range(3)
        ]
        return torch.stack(components, dim=-1)

    def total_variation(self) -> Tensor:
        grid = self.values_zyx
        dx = (grid[..., 1:] - grid[..., :-1]).abs().mean()
        dy = (grid[..., 1:, :] - grid[..., :-1, :]).abs().mean()
        dz = (grid[..., 1:, :, :] - grid[..., :-1, :, :]).abs().mean()
        return (dx + dy + dz) / self.pitch

    def numpy_values_xyz(self) -> np.ndarray:
        return self.values_xyz.detach().cpu().numpy().astype(np.float32, copy=True)


def dense_eikonal_loss(field: LearnableDenseSDF, points: Tensor) -> Tensor:
    return cast(
        Tensor,
        (torch.linalg.vector_norm(field.spatial_gradient(points), dim=-1) - 1.0).square().mean(),
    )


def project_points_to_zero_level(
    field: LearnableDenseSDF,
    points: Tensor,
    *,
    iteration_count: int = 2,
    maximum_step_voxels: float = 0.5,
) -> Tensor:
    """Differentiably project carrier points toward the learnable zero level."""
    if iteration_count <= 0 or maximum_step_voxels <= 0:
        raise ValueError("Projection iteration and step limits must be positive")
    projected = points
    maximum_step = maximum_step_voxels * field.pitch
    for _ in range(iteration_count):
        values = field(projected)
        gradients = field.finite_difference_gradient(projected)
        update = (
            -values[..., None]
            * gradients
            / gradients.square().sum(-1, keepdim=True).clamp_min(1e-8)
        )
        lengths = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
        update = update * (maximum_step / lengths.clamp_min(maximum_step)).clamp_max(1.0)
        projected = projected + update
    return projected


def sign_anchor_loss(
    field: LearnableDenseSDF,
    inside_points: Tensor,
    outside_points: Tensor,
    *,
    margin: float,
) -> Tensor:
    if margin <= 0:
        raise ValueError("Sign-anchor margin must be positive")
    inside = torch.relu(field(inside_points) + margin).square().mean()
    outside = torch.relu(margin - field(outside_points)).square().mean()
    return inside + outside


def smooth_sign_anchor_loss(
    field: LearnableDenseSDF,
    inside_points: Tensor,
    outside_points: Tensor,
    *,
    margin: float,
    temperature: float,
) -> Tensor:
    """Nonzero smooth barrier that preserves inside/outside sign margins."""
    if margin <= 0 or temperature <= 0:
        raise ValueError("Smooth sign-anchor margin and temperature must be positive")
    inside_violation = (
        torch.nn.functional.softplus((field(inside_points) + margin) / temperature) * temperature
    )
    outside_violation = (
        torch.nn.functional.softplus((margin - field(outside_points)) / temperature) * temperature
    )
    return inside_violation.square().mean() + outside_violation.square().mean()


def save_dense_sdf_checkpoint(
    path: Path,
    field: LearnableDenseSDF,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "learnable_dense_sdf_checkpoint.v1",
            "step": step,
            "field": field.state_dict(),
            "optimizer": optimizer.state_dict(),
            "shape_xyz": [int(value) for value in field.values_xyz.shape],
            "pitch": field.pitch,
        },
        path,
    )


def load_dense_sdf_checkpoint(
    path: Path,
    field: LearnableDenseSDF,
    optimizer: torch.optim.Optimizer,
) -> int:
    payload = torch.load(path, map_location=field.values_zyx.device, weights_only=False)
    if payload.get("schema_version") != "learnable_dense_sdf_checkpoint.v1":
        raise ValueError("Unsupported learnable dense SDF checkpoint")
    if list(field.values_xyz.shape) != list(payload["shape_xyz"]):
        raise ValueError("Dense SDF checkpoint shape mismatch")
    if abs(field.pitch - float(payload["pitch"])) > 1e-12:
        raise ValueError("Dense SDF checkpoint pitch mismatch")
    field.load_state_dict(payload["field"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"])
