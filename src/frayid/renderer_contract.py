from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from torch import Tensor

from frayid.renderer import render_soft_mesh
from frayid.triangle_rasterizer import NvdiffrastRenderer


class RendererBackend(StrEnum):
    LEGACY_SOFT_SPLAT = "legacy_soft_splat"
    OPAQUE_NVDIFFRAST = "opaque_nvdiffrast"


@dataclass(frozen=True)
class RendererBackendContract:
    backend: RendererBackend
    opaque_visibility: bool
    differentiable: bool
    training_permitted: bool
    legacy_evaluation_permitted: bool
    report_namespace: str


CONTRACTS = {
    RendererBackend.LEGACY_SOFT_SPLAT: RendererBackendContract(
        backend=RendererBackend.LEGACY_SOFT_SPLAT,
        opaque_visibility=False,
        differentiable=True,
        training_permitted=True,
        legacy_evaluation_permitted=True,
        report_namespace="legacy_evaluator",
    ),
    RendererBackend.OPAQUE_NVDIFFRAST: RendererBackendContract(
        backend=RendererBackend.OPAQUE_NVDIFFRAST,
        opaque_visibility=True,
        differentiable=True,
        training_permitted=True,
        legacy_evaluation_permitted=False,
        report_namespace="opaque_reference",
    ),
}

Renderer = Callable[..., tuple[Tensor, Tensor]]


def renderer_contract(backend: RendererBackend) -> RendererBackendContract:
    return CONTRACTS[backend]


def create_training_renderer(backend: RendererBackend) -> Renderer:
    contract = renderer_contract(backend)
    if not contract.training_permitted:
        raise ValueError(f"renderer is not permitted for training: {backend}")
    if backend is RendererBackend.LEGACY_SOFT_SPLAT:
        return render_soft_mesh
    return NvdiffrastRenderer()


def require_legacy_evaluator_backend(backend: RendererBackend) -> None:
    if not renderer_contract(backend).legacy_evaluation_permitted:
        raise ValueError("opaque-reference metrics cannot replace legacy evaluator metrics")
