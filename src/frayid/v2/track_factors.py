from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from frayid.v2.contracts import reject_sealed_capability


@dataclass(frozen=True)
class PairwiseTrackletFactors:
    """Uncertain train-only image factors; these do not assert persistent material identity."""

    first_ordinals: Tensor
    second_ordinals: Tensor
    first_source_frame_indices: Tensor
    second_source_frame_indices: Tensor
    edge_offsets: Tensor
    first_pixels: Tensor
    second_pixels: Tensor
    observation_weights: Tensor
    geometric_model_codes: Tensor

    @property
    def edge_count(self) -> int:
        return int(self.first_ordinals.numel())

    @property
    def factor_count(self) -> int:
        return int(self.first_pixels.shape[0])

    def validate(self) -> None:
        edge_count = self.edge_count
        factor_count = self.factor_count
        edge_vectors = (
            self.first_ordinals,
            self.second_ordinals,
            self.first_source_frame_indices,
            self.second_source_frame_indices,
            self.geometric_model_codes,
        )
        if any(value.shape != (edge_count,) for value in edge_vectors):
            raise ValueError("track-factor edge vectors must share one length")
        if any(value.dtype != torch.long for value in edge_vectors):
            raise ValueError("track-factor edge vectors must use torch.long")
        if self.edge_offsets.shape != (edge_count + 1,) or self.edge_offsets.dtype != torch.long:
            raise ValueError("edge_offsets must be torch.long with edge_count + 1 entries")
        if int(self.edge_offsets[0]) != 0 or int(self.edge_offsets[-1]) != factor_count:
            raise ValueError("edge_offsets do not span the factor array")
        if bool(torch.any(self.edge_offsets[1:] < self.edge_offsets[:-1])):
            raise ValueError("edge_offsets must be monotonic")
        if self.first_pixels.shape != (factor_count, 2) or self.second_pixels.shape != (
            factor_count,
            2,
        ):
            raise ValueError("track-factor pixels must have shape [factor_count, 2]")
        if self.observation_weights.shape != (factor_count,):
            raise ValueError("track-factor weights must have shape [factor_count]")
        finite = (self.first_pixels, self.second_pixels, self.observation_weights)
        if any(not bool(torch.isfinite(value).all()) for value in finite):
            raise ValueError("track-factor floating arrays must be finite")
        if bool(torch.any(self.observation_weights <= 0)):
            raise ValueError("track-factor weights must be positive")
        if bool(torch.any(self.first_ordinals >= self.second_ordinals)):
            raise ValueError("track-factor edges must point forward in training time")
        if bool(torch.any((self.geometric_model_codes < 0) | (self.geometric_model_codes > 1))):
            raise ValueError("unknown proposal geometric-model code")

    def factor_edge_indices(self) -> Tensor:
        counts = self.edge_offsets[1:] - self.edge_offsets[:-1]
        indices = torch.arange(edge_count := self.edge_count, device=counts.device)
        result = torch.repeat_interleave(indices, counts)
        if result.shape != (self.factor_count,) or edge_count != len(counts):
            raise AssertionError("track-factor edge expansion is inconsistent")
        return result


def load_pairwise_tracklet_factors(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> PairwiseTrackletFactors:
    reject_sealed_capability([path])
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"]) != "frayid_v2_pairwise_tracklet_factors.v1":
            raise ValueError("unsupported pairwise track-factor schema")
        result = PairwiseTrackletFactors(
            first_ordinals=torch.as_tensor(
                archive["first_ordinals"], dtype=torch.long, device=device
            ),
            second_ordinals=torch.as_tensor(
                archive["second_ordinals"], dtype=torch.long, device=device
            ),
            first_source_frame_indices=torch.as_tensor(
                archive["first_source_frame_indices"], dtype=torch.long, device=device
            ),
            second_source_frame_indices=torch.as_tensor(
                archive["second_source_frame_indices"], dtype=torch.long, device=device
            ),
            edge_offsets=torch.as_tensor(archive["edge_offsets"], dtype=torch.long, device=device),
            first_pixels=torch.as_tensor(
                archive["first_pixels"], dtype=torch.float32, device=device
            ),
            second_pixels=torch.as_tensor(
                archive["second_pixels"], dtype=torch.float32, device=device
            ),
            observation_weights=torch.as_tensor(
                archive["observation_weights"], dtype=torch.float32, device=device
            ),
            geometric_model_codes=torch.as_tensor(
                archive["geometric_model_codes"], dtype=torch.long, device=device
            ),
        )
    result.validate()
    return result


def pairwise_sampson_loss(
    fundamental_matrices: Tensor,
    factors: PairwiseTrackletFactors,
    *,
    image_size: tuple[int, int],
    robust_delta_fraction_of_diagonal: float = 0.0025,
) -> Tensor:
    """Differentiate robust pairwise observation factors through T01 camera geometry."""

    factors.validate()
    if fundamental_matrices.shape != (factors.edge_count, 3, 3):
        raise ValueError("one 3x3 fundamental matrix is required for each track-factor edge")
    if not bool(torch.isfinite(fundamental_matrices).all()):
        raise ValueError("fundamental matrices must be finite")
    height, width = image_size
    if height <= 0 or width <= 0 or robust_delta_fraction_of_diagonal <= 0:
        raise ValueError("image dimensions and robust delta must be positive")
    edge_indices = factors.factor_edge_indices()
    matrices = fundamental_matrices[edge_indices]
    ones = torch.ones(
        (factors.factor_count, 1),
        dtype=factors.first_pixels.dtype,
        device=factors.first_pixels.device,
    )
    first = torch.cat((factors.first_pixels, ones), dim=-1)
    second = torch.cat((factors.second_pixels, ones), dim=-1)
    first_lines = torch.einsum("nij,nj->ni", matrices, first)
    second_lines = torch.einsum("nji,nj->ni", matrices, second)
    numerator = torch.einsum("ni,ni->n", second, first_lines)
    denominator = (
        first_lines[:, :2].square().sum(dim=-1) + second_lines[:, :2].square().sum(dim=-1)
    ).clamp_min(1.0e-12)
    sampson_pixels = torch.sqrt(numerator.square() / denominator + 1.0e-12)
    diagonal = float((height * height + width * width) ** 0.5)
    normalized = sampson_pixels / diagonal
    delta = normalized.new_tensor(robust_delta_fraction_of_diagonal)
    penalty = delta * (torch.sqrt(1.0 + (normalized / delta).square()) - 1.0)
    weights = factors.observation_weights.to(dtype=penalty.dtype, device=penalty.device)
    return (weights * penalty).sum() / weights.sum()
