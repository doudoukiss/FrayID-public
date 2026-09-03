from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from torch import Tensor

from frayid.differentiable_isosurface import (
    SurfacePathCertificate,
    certify_zero_set_scalar_path,
    interpolate_zero_crossings,
    surface_endpoint_area_ratios,
)
from frayid.hybrid_tetrahedral import FixedSignTetrahedralField, tetrahedron_signed_volumes


def fixed_box_boundary_mask(positions: Tensor) -> Tensor:
    minimum = positions.amin(dim=0)
    maximum = positions.amax(dim=0)
    return torch.any((positions == minimum) | (positions == maximum), dim=1)


class EulerianImageField(FixedSignTetrahedralField):
    """Fixed-domain scalar field whose actual PL zero set receives image gradients."""

    outer_boundary_mask: Tensor

    def __init__(
        self,
        positions: Tensor,
        tetrahedra: Tensor,
        initial_values: Tensor,
        *,
        minimum_absolute_value: float = 1e-3,
        maximum_absolute_value: float = 4.0,
    ) -> None:
        super().__init__(
            positions,
            tetrahedra,
            initial_values,
            minimum_absolute_value=minimum_absolute_value,
            maximum_absolute_value=maximum_absolute_value,
        )
        boundary = fixed_box_boundary_mask(self.positions)
        if torch.any(self.signs[boundary] <= 0):
            raise ValueError("outer boundary scalar signs must be strictly positive")
        volumes = tetrahedron_signed_volumes(self.positions, self.tetrahedra)
        if torch.any(volumes == 0):
            raise ValueError("fixed tetrahedral domain contains a degenerate cell")
        self.register_buffer("outer_boundary_mask", boundary)

    def surface_vertices(self) -> Tensor:
        return interpolate_zero_crossings(self.positions, self.surface_edges, self.field_values)

    @property
    def serialized_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for tensor in (
            self.positions,
            self.tetrahedra,
            self.signs,
            self.surface_edges,
            self.surface_faces,
            self.field_values,
        ):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class EulerianStepResult:
    accepted_scale: float
    certificate: SurfacePathCertificate
    rejected: bool


def project_eulerian_step(
    field: EulerianImageField,
    previous_logits: Tensor,
    reference_vertices: Tensor,
    *,
    signed_area_floor: float = 0.01,
    unsigned_area_floor: float = 0.10,
    maximum_backtracks: int = 32,
    optimizer: torch.optim.Optimizer | None = None,
) -> EulerianStepResult:
    candidate = field.magnitude_logits.detach().clone()
    delta = candidate - previous_logits
    with torch.no_grad():
        field.magnitude_logits.copy_(previous_logits)
        start_values = field.field_values.detach().cpu().double().numpy()
    last = SurfacePathCertificate(
        "unknown",
        signed_area_floor,
        unsigned_area_floor,
        -np.inf,
        0.0,
        0,
        1,
        "candidate_not_checked",
    )
    scale = 1.0
    for _ in range(maximum_backtracks + 1):
        with torch.no_grad():
            field.magnitude_logits.copy_(previous_logits + scale * delta)
            end_values = field.field_values.detach().cpu().double().numpy()
            endpoint_vertices = field.surface_vertices().detach().cpu().double().numpy()
        minimum_signed, minimum_unsigned = surface_endpoint_area_ratios(
            endpoint_vertices,
            reference_vertices.detach().cpu().double().numpy(),
            field.surface_faces.detach().cpu().numpy(),
        )
        if minimum_signed < signed_area_floor or minimum_unsigned < unsigned_area_floor:
            last = SurfacePathCertificate(
                "fail",
                signed_area_floor,
                unsigned_area_floor,
                minimum_signed,
                minimum_unsigned,
                0,
                1,
                "endpoint_area_floor",
            )
            scale *= 0.5
            continue
        last = certify_zero_set_scalar_path(
            field.positions.detach().cpu().double().numpy(),
            field.surface_edges.detach().cpu().numpy(),
            field.surface_faces.detach().cpu().numpy(),
            start_values,
            end_values,
            reference_vertices.detach().cpu().double().numpy(),
            signed_area_floor=signed_area_floor,
            unsigned_area_floor=unsigned_area_floor,
        )
        if last.status == "pass":
            if optimizer is not None and scale < 1.0:
                _damp_optimizer_state(optimizer, field.magnitude_logits, scale)
            return EulerianStepResult(scale, last, False)
        scale *= 0.5
    with torch.no_grad():
        field.magnitude_logits.copy_(previous_logits)
    if optimizer is not None:
        _damp_optimizer_state(optimizer, field.magnitude_logits, 0.0)
    return EulerianStepResult(0.0, last, True)


def _damp_optimizer_state(
    optimizer: torch.optim.Optimizer, parameter: Tensor, scale: float
) -> None:
    state = optimizer.state.get(parameter, {})
    first_moment = state.get("exp_avg")
    second_moment = state.get("exp_avg_sq")
    if isinstance(first_moment, Tensor):
        first_moment.mul_(scale)
    if isinstance(second_moment, Tensor):
        second_moment.mul_(scale * scale)


def conventional_surface_audit(vertices: Tensor, faces: Tensor) -> dict[str, object]:
    array = vertices.detach().cpu().double().numpy()
    indices = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=array, faces=indices, process=False)
    components = mesh.split(only_watertight=False)
    return {
        "finite": bool(np.isfinite(array).all()),
        "vertex_count": int(array.shape[0]),
        "face_count": int(indices.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "component_count": len(components),
        "euler_number": int(mesh.euler_number),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "positive_volume": bool(mesh.volume > 0),
        "status": (
            "pass"
            if np.isfinite(array).all()
            and mesh.is_watertight
            and len(components) == 1
            and int(mesh.euler_number) == 2
            and mesh.is_winding_consistent
            and mesh.volume > 0
            else "fail"
        ),
    }
