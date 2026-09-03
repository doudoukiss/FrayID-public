from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from frayid.camera import project_points


@dataclass(frozen=True)
class MaterialTrackObservations:
    """Fixed material-point identities and their uncertain image observations."""

    face_indices: Tensor
    barycentric_coordinates: Tensor
    observed_pixels: Tensor
    observation_weights: Tensor
    valid: Tensor

    def validate(
        self,
        *,
        face_count: int,
        frame_count: int,
        minimum_barycentric_coordinate: float = 0.0,
        require_each_track_observed: bool = True,
    ) -> None:
        track_count = int(self.face_indices.numel())
        if self.face_indices.shape != (track_count,):
            raise ValueError("face_indices must have shape [track_count]")
        if self.face_indices.dtype != torch.long:
            raise ValueError("face_indices must use torch.long")
        if bool(torch.any(self.face_indices < 0) or torch.any(self.face_indices >= face_count)):
            raise ValueError("face_indices contain an out-of-range face")
        if self.barycentric_coordinates.shape != (track_count, 3):
            raise ValueError("barycentric_coordinates must have shape [track_count, 3]")
        if self.observed_pixels.shape != (frame_count, track_count, 2):
            raise ValueError("observed_pixels must have shape [frame_count, track_count, 2]")
        if self.observation_weights.shape != (frame_count, track_count):
            raise ValueError("observation_weights must have shape [frame_count, track_count]")
        if self.valid.shape != (frame_count, track_count) or self.valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor with shape [frame_count, track_count]")
        tensors = (
            self.barycentric_coordinates,
            self.observed_pixels,
            self.observation_weights,
        )
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("material-track arrays must be finite")
        if bool(torch.any(self.observation_weights < 0)):
            raise ValueError("observation weights must be nonnegative")
        if bool(torch.any(self.barycentric_coordinates < minimum_barycentric_coordinate)):
            raise ValueError("a material point is too close to an anchor triangle boundary")
        sums = self.barycentric_coordinates.sum(dim=-1)
        if not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0)):
            raise ValueError("barycentric coordinates must sum to one")
        if (
            require_each_track_observed
            and track_count
            and not bool(torch.all(self.valid.any(dim=0)))
        ):
            raise ValueError("every material track must contain at least one valid observation")


@dataclass(frozen=True)
class TrackletAssignment:
    """Train-only segmentation of long identities into contiguous reliable pieces."""

    ids: Tensor
    tracklet_count: int

    def validate(self, shape: tuple[int, int]) -> None:
        if self.ids.shape != shape or self.ids.dtype != torch.long:
            raise ValueError("tracklet ids must be torch.long with shape [frame, track]")
        if self.tracklet_count < 0:
            raise ValueError("tracklet_count must be nonnegative")
        valid_ids = self.ids[self.ids >= 0]
        if valid_ids.numel() and int(valid_ids.max()) >= self.tracklet_count:
            raise ValueError("tracklet ids are outside the declared range")


def segment_tracklets(
    valid: Tensor,
    forward_backward_error: Tensor,
    cycle_error: Tensor,
    occlusion_break: Tensor,
    *,
    maximum_forward_backward_error: float,
    maximum_cycle_error: float,
) -> TrackletAssignment:
    """Split identities using train-only visibility, consistency, cycle, and occlusion rules."""
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [frame, track]")
    if forward_backward_error.shape != valid.shape or cycle_error.shape != valid.shape:
        raise ValueError("consistency arrays must match valid")
    if occlusion_break.shape != valid.shape or occlusion_break.dtype != torch.bool:
        raise ValueError("occlusion_break must be boolean and match valid")
    if maximum_forward_backward_error <= 0 or maximum_cycle_error <= 0:
        raise ValueError("tracklet consistency thresholds must be positive")
    reliable = (
        valid
        & torch.isfinite(forward_backward_error)
        & torch.isfinite(cycle_error)
        & (forward_backward_error <= maximum_forward_backward_error)
        & (cycle_error <= maximum_cycle_error)
    )
    ids = torch.full(valid.shape, -1, dtype=torch.long, device=valid.device)
    next_id = 0
    for track in range(int(valid.shape[1])):
        active_id = -1
        for frame in range(int(valid.shape[0])):
            if not bool(reliable[frame, track]):
                active_id = -1
                continue
            if active_id < 0 or bool(occlusion_break[frame, track]):
                active_id = next_id
                next_id += 1
            ids[frame, track] = active_id
    result = TrackletAssignment(ids=ids, tracklet_count=next_id)
    result.validate((int(valid.shape[0]), int(valid.shape[1])))
    return result


def material_track_points(
    posed_vertices: Tensor,
    faces: Tensor,
    face_indices: Tensor,
    barycentric_coordinates: Tensor,
) -> Tensor:
    """Evaluate fixed surface material points on one or more posed meshes."""
    squeeze = posed_vertices.ndim == 2
    if squeeze:
        posed_vertices = posed_vertices.unsqueeze(0)
    if posed_vertices.ndim != 3 or posed_vertices.shape[-1] != 3:
        raise ValueError("posed_vertices must have shape [V, 3] or [T, V, 3]")
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be a torch.long tensor with shape [F, 3]")
    if face_indices.ndim != 1 or face_indices.dtype != torch.long:
        raise ValueError("face_indices must be one-dimensional torch.long")
    if barycentric_coordinates.shape != (face_indices.numel(), 3):
        raise ValueError("barycentric_coordinates must match face_indices")
    selected_faces = faces[face_indices]
    corners = posed_vertices[:, selected_faces]
    points = (corners * barycentric_coordinates[None, :, :, None]).sum(dim=2)
    return points[0] if squeeze else points


def pseudo_huber(values: Tensor, *, delta: float) -> Tensor:
    """Dimension-preserving pseudo-Huber penalty with bounded influence."""
    if delta <= 0:
        raise ValueError("pseudo-Huber delta must be positive")
    return delta * (torch.sqrt(1.0 + (values / delta).square()) - 1.0)


def material_track_reprojection_loss(
    posed_vertices: Tensor,
    faces: Tensor,
    intrinsics: Tensor,
    observations: MaterialTrackObservations,
    *,
    source_image_size: tuple[int, int],
    robust_delta_fraction_of_diagonal: float,
) -> Tensor:
    """Robust train-only reprojection loss for frozen material identities."""
    if posed_vertices.ndim != 3:
        raise ValueError("posed_vertices must have shape [T, V, 3]")
    frame_count = int(posed_vertices.shape[0])
    observations.validate(
        face_count=int(faces.shape[0]),
        frame_count=frame_count,
        require_each_track_observed=False,
    )
    if robust_delta_fraction_of_diagonal <= 0:
        raise ValueError("robust delta fraction must be positive")
    points = material_track_points(
        posed_vertices,
        faces,
        observations.face_indices,
        observations.barycentric_coordinates,
    )
    projected = project_points(points, intrinsics)
    height, width = source_image_size
    if height <= 0 or width <= 0:
        raise ValueError("source image dimensions must be positive")
    diagonal = float((height * height + width * width) ** 0.5)
    pixel_delta = projected - observations.observed_pixels
    normalized_residual = torch.sqrt(pixel_delta.square().sum(dim=-1) + 1e-12) / diagonal
    penalty = pseudo_huber(
        normalized_residual,
        delta=robust_delta_fraction_of_diagonal,
    )
    weights = observations.observation_weights * observations.valid.to(penalty.dtype)
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        return posed_vertices.sum() * 0.0
    return (weights * penalty).sum() / denominator


def tracklet_redescending_loss(
    penalties: Tensor,
    weights: Tensor,
    assignment: TrackletAssignment,
    *,
    lambda_value: float,
) -> tuple[Tensor, Tensor]:
    """Eliminate one outlier scale per tracklet using lambda*S/(lambda+S)."""
    if penalties.shape != weights.shape or penalties.ndim != 2:
        raise ValueError("penalties and weights must share shape [frame, track]")
    assignment.validate((int(penalties.shape[0]), int(penalties.shape[1])))
    if lambda_value <= 0:
        raise ValueError("lambda_value must be positive")
    if not bool(torch.isfinite(penalties).all() and torch.isfinite(weights).all()):
        raise ValueError("penalties and weights must be finite")
    if bool(torch.any(penalties < 0) or torch.any(weights < 0)):
        raise ValueError("penalties and weights must be nonnegative")
    if assignment.tracklet_count == 0:
        return penalties.sum() * 0.0, penalties.new_zeros((0,))
    sums: list[Tensor] = []
    for tracklet_id in range(assignment.tracklet_count):
        selected = assignment.ids == tracklet_id
        sums.append((penalties[selected] * weights[selected]).sum())
    tracklet_sums = torch.stack(sums)
    lambda_tensor = penalties.new_tensor(lambda_value)
    reduced = lambda_tensor * tracklet_sums / (lambda_tensor + tracklet_sums)
    return reduced.mean(), tracklet_sums


def tracklet_reliability(tracklet_sums: Tensor, *, lambda_value: float) -> Tensor:
    """Return the normalized derivative of the eliminated redescending penalty."""
    if lambda_value <= 0:
        raise ValueError("lambda_value must be positive")
    if bool(torch.any(tracklet_sums < 0) or not torch.isfinite(tracklet_sums).all()):
        raise ValueError("tracklet sums must be finite and nonnegative")
    lambda_tensor = tracklet_sums.new_tensor(lambda_value)
    return (lambda_tensor / (lambda_tensor + tracklet_sums)).square()
