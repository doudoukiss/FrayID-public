from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from frayid.config import ReconstructionConfig
from frayid.geometry import (
    canonical_topology_is_valid,
    canonical_topology_losses,
    deformation_jacobian,
    eikonal_loss,
    jacobian_regularization,
    linear_blend_skinning,
    rigid_transform_from_axis_angle,
    safe_root_mean_square,
    signed_sdf_mesh_consistency,
    temporal_second_difference,
)
from frayid.renderer import (
    differentiable_boundary_loss,
    normal_cosine_loss,
    render_soft_mesh,
    silhouette_loss,
)
from frayid.training import CanonicalGeometryModel


class CarrierEvidence(Protocol):
    masks: Tensor
    normals: Tensor
    transforms: Tensor
    intrinsics: Tensor
    source_image_size: tuple[int, int]


def carrier_loss_values(
    model: CanonicalGeometryModel,
    base: Tensor,
    offsets: Tensor,
    faces: Tensor,
    weights: Tensor,
    edges: Tensor,
    evidence: CarrierEvidence,
    config: ReconstructionConfig,
    slot: int,
    *,
    residual_enabled: bool,
    transform_override: Tensor | None = None,
) -> dict[str, Tensor]:
    """Evaluate the native objective on an external topology-safe carrier."""
    canonical = base + offsets
    code_slot = torch.tensor(slot, device=canonical.device)
    residual = (
        model.deformer(canonical, code_slot) if residual_enabled else torch.zeros_like(canonical)
    )
    root = rigid_transform_from_axis_angle(
        model.root_rotation_corrections[slot], model.root_translation_corrections[slot]
    )
    frame_transform = (
        evidence.transforms[slot] if transform_override is None else transform_override
    )
    posed = linear_blend_skinning(
        canonical + residual, weights, root.unsqueeze(0) @ frame_transform
    )
    resolution = int(evidence.masks.shape[-1])
    torch.manual_seed(config.seed + 100_000 + slot)
    silhouette, normals = render_soft_mesh(
        posed,
        faces,
        evidence.intrinsics,
        (resolution, resolution),
        source_image_size=evidence.source_image_size,
        sigma_pixels=config.model.renderer_sigma_pixels * resolution / 128.0,
        sample_count=min(
            len(canonical),
            round(config.model.renderer_max_vertices * (resolution / 128.0) ** 2),
        ),
        reference_sample_count=round(
            config.model.renderer_reference_sample_count * (resolution / 128.0) ** 2
        ),
        depth_temperature_m=config.model.renderer_depth_temperature_m,
    )
    bounds_low = canonical.detach().amin(0) - 0.1
    bounds_high = canonical.detach().amax(0) + 0.1
    random_points = bounds_low + torch.rand(
        (config.model.eikonal_sample_count, 3),
        dtype=canonical.dtype,
        device=canonical.device,
    ) * (bounds_high - bounds_low)
    jacobian_points = canonical[:: max(1, len(canonical) // 64)].detach().requires_grad_(True)
    jacobian_displacement = model.deformer(jacobian_points, code_slot)
    topology = canonical_topology_losses(
        base,
        canonical,
        faces,
        edges,
        offsets,
        orientation_margin=config.model.canonical_orientation_margin,
        minimum_area_ratio=config.model.canonical_minimum_area_ratio,
    )
    root_rotation = model.root_rotation_corrections
    root_translation = model.root_translation_corrections
    return {
        "silhouette": silhouette_loss(silhouette, evidence.masks[slot]),
        "boundary": differentiable_boundary_loss(silhouette, evidence.masks[slot]),
        "normal": normal_cosine_loss(normals, evidence.normals[slot], evidence.masks[slot]),
        "eikonal": eikonal_loss(model.sdf, random_points),
        "sdf_mesh_consistency": signed_sdf_mesh_consistency(
            model.sdf,
            canonical,
            faces,
            maximum_sample_count=config.model.eikonal_sample_count,
        ),
        "deformation": safe_root_mean_square(residual),
        "jacobian": jacobian_regularization(
            deformation_jacobian(jacobian_displacement, jacobian_points)
        ).clamp_min(1e-12),
        "temporal": temporal_second_difference(model.deformer.frame_codes.weight).clamp_min(1e-12),
        "root_rotation_correction": (root_rotation[slot].square().mean() + 1e-12).sqrt(),
        "root_translation_correction": (root_translation[slot].square().mean() + 1e-12).sqrt(),
        "root_correction_temporal": (
            temporal_second_difference(root_rotation) + temporal_second_difference(root_translation)
        ).clamp_min(1e-12),
        **topology,
    }


def weighted_geometry_objective(values: dict[str, Tensor], config: ReconstructionConfig) -> Tensor:
    first = next(iter(values.values()))
    total = torch.zeros((), dtype=first.dtype, device=first.device)
    for name, value in values.items():
        total = total + getattr(config.losses, name) * value
    return total


def project_carrier_step(
    offsets: Tensor,
    previous_offsets: Tensor,
    base: Tensor,
    faces: Tensor,
    optimizer: torch.optim.Optimizer,
    config: ReconstructionConfig,
) -> float:
    """Backtrack a carrier step and damp Adam moments by its accepted scale."""
    proposed = offsets.detach().clone()
    accepted_scale = 0.0
    for backtrack in range(config.model.canonical_backtracking_steps + 1):
        scale = 0.5**backtrack
        candidate = previous_offsets + scale * (proposed - previous_offsets)
        if canonical_topology_is_valid(
            base,
            base + candidate,
            faces,
            minimum_signed_area_ratio=config.model.canonical_minimum_signed_area_ratio,
            minimum_area_ratio=config.model.canonical_minimum_area_ratio,
        ):
            offsets.data.copy_(candidate)
            accepted_scale = scale
            break
    if accepted_scale == 0.0:
        offsets.data.copy_(previous_offsets)
    if accepted_scale < 1.0:
        state = optimizer.state.get(offsets, {})
        first_moment = state.get("exp_avg")
        second_moment = state.get("exp_avg_sq")
        if isinstance(first_moment, Tensor):
            first_moment.mul_(accepted_scale)
        if isinstance(second_moment, Tensor):
            second_moment.mul_(accepted_scale * accepted_scale)
    return accepted_scale
