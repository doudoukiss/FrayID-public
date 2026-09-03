from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from frayid.geometry import canonical_topology_is_valid, canonical_topology_quantities


class TrilinearDeformationCage(nn.Module):
    """Low-dimensional smooth displacement field over fixed reference vertices."""

    def __init__(
        self,
        reference_vertices: Tensor,
        resolution: tuple[int, int, int] = (8, 8, 4),
    ) -> None:
        super().__init__()
        if reference_vertices.ndim != 2 or reference_vertices.shape[1] != 3:
            raise ValueError("reference vertices must have shape (V, 3)")
        if any(axis < 2 for axis in resolution):
            raise ValueError("every cage axis must contain at least two controls")
        low = reference_vertices.detach().amin(dim=0)
        high = reference_vertices.detach().amax(dim=0)
        extent = high - low
        if torch.any(extent <= 0):
            raise ValueError("reference vertices must span every coordinate axis")
        self.resolution = resolution
        self.reference_vertices: Tensor
        self.corner_indices: Tensor
        self.corner_weights: Tensor
        self.register_buffer("reference_vertices", reference_vertices.detach().clone())

        scale = torch.tensor(
            [axis - 1 for axis in resolution],
            dtype=reference_vertices.dtype,
            device=reference_vertices.device,
        )
        coordinates = (reference_vertices.detach() - low) / extent * scale
        lower = torch.floor(coordinates).long()
        maximum_lower = torch.tensor(
            [axis - 2 for axis in resolution],
            dtype=torch.long,
            device=reference_vertices.device,
        )
        lower = torch.minimum(lower, maximum_lower)
        lower = torch.clamp(lower, min=0)
        fractions = (coordinates - lower.to(coordinates.dtype)).clamp(0.0, 1.0)
        indices: list[Tensor] = []
        weights: list[Tensor] = []
        y_count, z_count = resolution[1], resolution[2]
        for dx, dy, dz in (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ):
            corner = lower + torch.tensor(
                [dx, dy, dz], dtype=torch.long, device=reference_vertices.device
            )
            indices.append(corner[:, 0] * y_count * z_count + corner[:, 1] * z_count + corner[:, 2])
            factors = torch.where(
                torch.tensor([dx, dy, dz], device=reference_vertices.device).bool(),
                fractions,
                1.0 - fractions,
            )
            weights.append(factors.prod(dim=-1))
        self.register_buffer("corner_indices", torch.stack(indices, dim=1))
        self.register_buffer("corner_weights", torch.stack(weights, dim=1))
        self.controls = nn.Parameter(
            torch.zeros(
                (*resolution, 3), dtype=reference_vertices.dtype, device=reference_vertices.device
            )
        )

    def vertex_offsets(self) -> Tensor:
        flat_controls = self.controls.reshape(-1, 3)
        return (flat_controls[self.corner_indices] * self.corner_weights[..., None]).sum(dim=1)

    def deformed_vertices(self) -> Tensor:
        return self.reference_vertices + self.vertex_offsets()

    def smoothness_loss(self) -> Tensor:
        differences = [
            self.controls[1:] - self.controls[:-1],
            self.controls[:, 1:] - self.controls[:, :-1],
            self.controls[:, :, 1:] - self.controls[:, :, :-1],
        ]
        return torch.stack([difference.square().mean() for difference in differences]).mean()


def project_cage_step(
    cage: TrilinearDeformationCage,
    previous_controls: Tensor,
    faces: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    minimum_signed_area_ratio: float = 0.01,
    minimum_area_ratio: float = 0.1,
    maximum_backtracks: int = 16,
    active_slack_tolerance: float = 1e-4,
    diagnostics: dict[str, Any] | None = None,
) -> float:
    """Backtrack a cage update until its fixed-connectivity surface stays valid."""
    if active_slack_tolerance < 0:
        raise ValueError("Active slack tolerance cannot be negative")
    proposed = cage.controls.detach().clone()
    proposed_vertices = cage.deformed_vertices().detach().clone()
    accepted_scale = 0.0
    with torch.no_grad():
        for backtrack in range(maximum_backtracks + 1):
            scale = 0.5**backtrack
            cage.controls.copy_(previous_controls + scale * (proposed - previous_controls))
            if canonical_topology_is_valid(
                cage.reference_vertices,
                cage.deformed_vertices(),
                faces,
                minimum_signed_area_ratio=minimum_signed_area_ratio,
                minimum_area_ratio=minimum_area_ratio,
            ):
                accepted_scale = scale
                break
        if accepted_scale == 0.0:
            cage.controls.copy_(previous_controls)
        if accepted_scale < 1.0:
            state = optimizer.state.get(cage.controls, {})
            first_moment = state.get("exp_avg")
            second_moment = state.get("exp_avg_sq")
            if isinstance(first_moment, Tensor):
                first_moment.mul_(accepted_scale)
            if isinstance(second_moment, Tensor):
                second_moment.mul_(accepted_scale * accepted_scale)
        if diagnostics is not None:
            proposed_signed, proposed_unsigned = canonical_topology_quantities(
                cage.reference_vertices, proposed_vertices, faces
            )
            accepted_signed, accepted_unsigned = canonical_topology_quantities(
                cage.reference_vertices, cage.deformed_vertices(), faces
            )

            def constraint_report(signed: Tensor, unsigned: Tensor) -> dict[str, float | int]:
                signed_slack = signed - minimum_signed_area_ratio
                unsigned_slack = unsigned - minimum_area_ratio
                return {
                    "minimum_signed_slack": float(signed_slack.min().cpu()),
                    "minimum_unsigned_slack": float(unsigned_slack.min().cpu()),
                    "active_signed_constraint_count": int(
                        torch.count_nonzero(signed_slack <= active_slack_tolerance).cpu()
                    ),
                    "active_unsigned_constraint_count": int(
                        torch.count_nonzero(unsigned_slack <= active_slack_tolerance).cpu()
                    ),
                    "violated_signed_constraint_count": int(
                        torch.count_nonzero(signed_slack < 0).cpu()
                    ),
                    "violated_unsigned_constraint_count": int(
                        torch.count_nonzero(unsigned_slack < 0).cpu()
                    ),
                }

            diagnostics.update(
                {
                    "accepted_scale": accepted_scale,
                    "backtracking_count": (
                        maximum_backtracks + 1
                        if accepted_scale == 0.0
                        else round(-torch.log2(torch.tensor(accepted_scale)).item())
                    ),
                    "moment_damping_scale": accepted_scale,
                    "proposed_control_delta_l2": float(
                        torch.linalg.vector_norm(proposed - previous_controls).cpu()
                    ),
                    "active_slack_tolerance": active_slack_tolerance,
                    "proposed": constraint_report(proposed_signed, proposed_unsigned),
                    "accepted": constraint_report(accepted_signed, accepted_unsigned),
                }
            )
    return accepted_scale
