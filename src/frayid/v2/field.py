from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from frayid.flexicubes_adapter import FlexiCubesMesh, PinnedFlexiCubes
from frayid.normal_integrable_sdf import (
    NeuSRender,
    NormalIntegrableNeuralSDF,
    VisualHullGrid,
    render_neus_sdf,
    spatial_sdf_gradient,
)
from frayid.v2.evidence import EvidenceVolume


@runtime_checkable
class CanonicalField(Protocol):
    def values(self, points: Tensor) -> Tensor: ...

    def gradients(self, points: Tensor, *, create_graph: bool) -> Tensor: ...

    def render_rays(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        *,
        near: float,
        far: float,
        sample_count: int,
        hierarchical_sample_count: int,
        deformation_jacobian: Tensor,
        create_graph: bool,
    ) -> NeuSRender: ...

    def checkpoint_state(self) -> dict[str, Tensor]: ...

    def adaptive_extract(
        self,
        extractor: PinnedFlexiCubes,
        *,
        resolution: int,
        extent: float,
        mode: str,
    ) -> ExtractionResult: ...

    def evidence_gradients(self, losses: dict[str, Tensor]) -> dict[str, float]: ...


class V2NeuralSDF(nn.Module):
    """V2 adapter around the proven E25 neural-field primitives.

    The adapter has a new state namespace and binds an uncertainty-aware V2
    evidence volume.  It never loads an E25 attempt, endpoint, or run state.
    """

    def __init__(
        self,
        evidence: EvidenceVolume,
        *,
        hidden_width: int = 64,
        hidden_layers: int = 2,
        maximum_hash_resolution: int = 96,
    ) -> None:
        super().__init__()
        self.metadata = evidence.metadata
        self.field = NormalIntegrableNeuralSDF(
            VisualHullGrid(evidence.signed_distance, evidence.metadata.extent),
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            maximum_hash_resolution=maximum_hash_resolution,
        )
        self.register_buffer("evidence_support", evidence.support_count.clone())
        self.register_buffer("evidence_uncertainty", evidence.mask_uncertainty.clone())
        self.register_buffer("prior_contribution", evidence.prior_contribution.clone())
        self.semantic_layer_names = tuple(sorted(evidence.semantic_support))
        for name in self.semantic_layer_names:
            self.register_buffer(
                f"semantic_support_{name}",
                evidence.semantic_support[name].clone(),
            )

    def forward(self, points: Tensor) -> Tensor:
        result: Tensor = self.field(points)
        return result

    def values(self, points: Tensor) -> Tensor:
        result: Tensor = self.field(points)
        return result

    def gradients(self, points: Tensor, *, create_graph: bool) -> Tensor:
        result: Tensor = spatial_sdf_gradient(self.field, points, create_graph=create_graph)
        return result

    def render_rays(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        *,
        near: float,
        far: float,
        sample_count: int,
        hierarchical_sample_count: int,
        deformation_jacobian: Tensor,
        create_graph: bool,
    ) -> NeuSRender:
        return render_neus_sdf(
            self.field,
            ray_origins,
            ray_directions,
            near=near,
            far=far,
            sample_count=sample_count,
            hierarchical_sample_count=hierarchical_sample_count,
            inverse_sharpness=64.0,
            deformation_jacobian=deformation_jacobian,
            create_graph=create_graph,
        )

    def checkpoint_state(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.state_dict().items()}

    def adaptive_extract(
        self,
        extractor: PinnedFlexiCubes,
        *,
        resolution: int,
        extent: float,
        mode: str,
    ) -> ExtractionResult:
        return extract_field(
            self,
            extractor,
            resolution=resolution,
            extent=extent,
            mode=mode,
        )

    def evidence_gradients(self, losses: dict[str, Tensor]) -> dict[str, float]:
        return evidence_gradient_diagnostics(losses, self)


@dataclass(frozen=True)
class ExtractionResult:
    mesh: FlexiCubesMesh
    search_only: bool
    resolution: int


def extract_field(
    field: CanonicalField,
    extractor: PinnedFlexiCubes,
    *,
    resolution: int,
    extent: float,
    mode: str,
) -> ExtractionResult:
    if mode not in {"search", "commit", "refine"}:
        raise ValueError("extraction mode must be search, commit, or refine")
    vertices, cubes = extractor.voxel_grid(resolution, extent=extent)
    values = field.values(vertices)

    def extraction_gradient(points: Tensor) -> Tensor:
        # Official FlexiCubes requests this callback inside torch.no_grad().
        # Enable only the local derivative used to select a quad diagonal.
        with torch.enable_grad():
            query = points.detach().requires_grad_(True)
            return field.gradients(query, create_graph=False)

    mesh = extractor.extract(
        vertices,
        values,
        cubes,
        resolution,
        training=mode == "search",
        gradient_function=extraction_gradient,
    )
    return ExtractionResult(mesh=mesh, search_only=mode == "search", resolution=resolution)


def evidence_gradient_diagnostics(losses: dict[str, Tensor], model: nn.Module) -> dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    diagnostics: dict[str, float] = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        finite = [gradient for gradient in gradients if gradient is not None]
        if any(not torch.isfinite(gradient).all() for gradient in finite):
            raise RuntimeError(f"non-finite image-to-field gradient: {name}")
        if finite:
            total = torch.stack([gradient.square().sum() for gradient in finite]).sum()
            diagnostics[name] = float(torch.sqrt(total).detach())
        else:
            diagnostics[name] = 0.0
    return diagnostics
