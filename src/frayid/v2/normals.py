from __future__ import annotations

from enum import StrEnum

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor

from frayid.normal_integrable_sdf import (
    bilateral_normal_integrability_loss,
    spatial_sdf_gradient,
    transport_normals_inverse_transpose,
)


class NormalSpace(StrEnum):
    CAMERA_OPENCV = "camera_opencv"
    CAMERA_OPENGL = "camera_opengl"
    WORLD = "world"
    CANONICAL = "canonical"


class NormalObservationConvention(BaseModel):
    space: NormalSpace
    channel_order: str
    encoded_minimum: float
    encoded_maximum: float
    front_axis: str
    crop_scale_x: float = 1.0
    crop_scale_y: float = 1.0


def decode_normal_observation(values: Tensor, convention: NormalObservationConvention) -> Tensor:
    if values.shape[-1] != 3:
        raise ValueError("normal observation must end in three channels")
    if convention.encoded_maximum <= convention.encoded_minimum:
        raise ValueError("normal encoding range is invalid")
    channel_indices = {channel: index for index, channel in enumerate(convention.channel_order)}
    if set(channel_indices) != {"x", "y", "z"} or len(convention.channel_order) != 3:
        raise ValueError("normal channel order must be one permutation of xyz")
    decoded = (values - convention.encoded_minimum) / (
        convention.encoded_maximum - convention.encoded_minimum
    )
    decoded = decoded * 2.0 - 1.0
    decoded = decoded[..., [channel_indices["x"], channel_indices["y"], channel_indices["z"]]]
    if convention.space is NormalSpace.CAMERA_OPENGL:
        decoded = decoded * decoded.new_tensor((1.0, -1.0, -1.0))
    return F.normalize(decoded, dim=-1, eps=1.0e-8)


def continuous_canonical_normals(field: object, points: Tensor) -> tuple[Tensor, Tensor]:
    gradient = spatial_sdf_gradient(field, points, create_graph=True)  # type: ignore[arg-type]
    magnitudes = torch.linalg.vector_norm(gradient, dim=-1)
    normals = F.normalize(gradient, dim=-1, eps=1.0e-8)
    return normals, magnitudes


def normal_transport_ablation(
    normals: Tensor,
    jacobians: Tensor,
    *,
    mode: str,
) -> Tensor:
    if mode == "inverse_transpose":
        return transport_normals_inverse_transpose(normals, jacobians)
    if mode == "rotation_only":
        u, _, vh = torch.linalg.svd(jacobians)
        rotation = u @ vh
        return F.normalize((rotation @ normals.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-8)
    if mode == "facet_control":
        return F.normalize(normals, dim=-1, eps=1e-8)
    raise ValueError("unknown normal-transport ablation")


def normal_integration_residual(predicted: Tensor, observed: Tensor, mask: Tensor) -> Tensor:
    return bilateral_normal_integrability_loss(predicted, observed, mask)
