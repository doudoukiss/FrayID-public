from __future__ import annotations

import torch
from torch import Tensor, nn

from frayid.geometry import canonical_topology_is_valid, vertex_normals


class DiffusedNormalResidual(nn.Module):
    """Scalar normal displacements smoothed over fixed one-ring connectivity."""

    def __init__(
        self,
        reference_vertices: Tensor,
        faces: Tensor,
        *,
        diffusion_steps: int = 4,
        diffusion_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if reference_vertices.ndim != 2 or reference_vertices.shape[1] != 3:
            raise ValueError("reference vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= len(reference_vertices)):
            raise ValueError("faces reference an invalid vertex")
        if diffusion_steps <= 0:
            raise ValueError("diffusion steps must be positive")
        if not 0.0 < diffusion_weight <= 1.0:
            raise ValueError("diffusion weight must be in (0, 1]")

        face_edges = torch.cat(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
            dim=0,
        )
        edges = torch.unique(torch.sort(face_edges.long(), dim=1).values, dim=0)
        if not len(edges):
            raise ValueError("normal residual requires at least one topology edge")
        sources = torch.cat((edges[:, 0], edges[:, 1]))
        targets = torch.cat((edges[:, 1], edges[:, 0]))
        degree = torch.zeros(
            len(reference_vertices),
            dtype=reference_vertices.dtype,
            device=reference_vertices.device,
        )
        degree.index_add_(0, targets, torch.ones_like(targets, dtype=reference_vertices.dtype))
        if torch.any(degree <= 0):
            raise ValueError("normal residual topology contains an isolated vertex")

        self.diffusion_steps = diffusion_steps
        self.diffusion_weight = diffusion_weight
        self.reference_vertices: Tensor
        self.reference_normals: Tensor
        self.edges: Tensor
        self.adjacency_sources: Tensor
        self.adjacency_targets: Tensor
        self.degree: Tensor
        self.register_buffer("reference_vertices", reference_vertices.detach().clone())
        self.register_buffer(
            "reference_normals",
            vertex_normals(reference_vertices.detach(), faces).detach(),
        )
        self.register_buffer("edges", edges)
        self.register_buffer("adjacency_sources", sources)
        self.register_buffer("adjacency_targets", targets)
        self.register_buffer("degree", degree)
        self.raw_displacements = nn.Parameter(
            torch.zeros(
                len(reference_vertices),
                dtype=reference_vertices.dtype,
                device=reference_vertices.device,
            )
        )

    def diffused_displacements(self) -> Tensor:
        values: Tensor = self.raw_displacements
        for _ in range(self.diffusion_steps):
            neighbor_sum = torch.zeros_like(values)
            neighbor_sum.index_add_(
                0,
                self.adjacency_targets,
                values[self.adjacency_sources],
            )
            neighbor_mean = neighbor_sum / self.degree
            values = (1.0 - self.diffusion_weight) * values + self.diffusion_weight * neighbor_mean
        return values

    def vertex_offsets(self) -> Tensor:
        return self.reference_normals * self.diffused_displacements().unsqueeze(-1)

    def deformed_vertices(self) -> Tensor:
        return self.reference_vertices + self.vertex_offsets()

    def smoothness_loss(self) -> Tensor:
        values = self.diffused_displacements()
        edge_lengths = torch.linalg.vector_norm(
            self.reference_vertices[self.edges[:, 1]] - self.reference_vertices[self.edges[:, 0]],
            dim=-1,
        ).clamp_min(1e-8)
        differences = values[self.edges[:, 1]] - values[self.edges[:, 0]]
        result: Tensor = (differences / edge_lengths).square().mean()
        return result


def project_normal_residual_step(
    field: DiffusedNormalResidual,
    previous_raw_displacements: Tensor,
    original_reference_vertices: Tensor,
    faces: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    minimum_signed_area_ratio: float = 0.01,
    minimum_area_ratio: float = 0.1,
    maximum_backtracks: int = 16,
) -> float:
    """Project a scalar normal update through the original explicit topology gate."""
    if previous_raw_displacements.shape != field.raw_displacements.shape:
        raise ValueError("previous normal residual shape does not match the field")
    if original_reference_vertices.shape != field.reference_vertices.shape:
        raise ValueError("original topology reference does not match the residual field")
    proposed = field.raw_displacements.detach().clone()
    accepted_scale = 0.0
    with torch.no_grad():
        for backtrack in range(maximum_backtracks + 1):
            scale = 0.5**backtrack
            field.raw_displacements.copy_(
                previous_raw_displacements + scale * (proposed - previous_raw_displacements)
            )
            if canonical_topology_is_valid(
                original_reference_vertices,
                field.deformed_vertices(),
                faces,
                minimum_signed_area_ratio=minimum_signed_area_ratio,
                minimum_area_ratio=minimum_area_ratio,
            ):
                accepted_scale = scale
                break
        if accepted_scale == 0.0:
            field.raw_displacements.copy_(previous_raw_displacements)
        if accepted_scale < 1.0:
            state = optimizer.state.get(field.raw_displacements, {})
            first_moment = state.get("exp_avg")
            second_moment = state.get("exp_avg_sq")
            if isinstance(first_moment, Tensor):
                first_moment.mul_(accepted_scale)
            if isinstance(second_moment, Tensor):
                second_moment.mul_(accepted_scale * accepted_scale)
    return accepted_scale
