from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from skimage.measure import marching_cubes
from torch import Tensor, nn

_VERTEX_FACE_ADJACENCY_CACHE: dict[
    tuple[str, int | None, int, int, int],
    tuple[weakref.ReferenceType[Tensor], Tensor, Tensor],
] = {}


def clear_vertex_face_adjacency_cache() -> None:
    """Drop derived adjacency so checkpoint restore can rebuild it deterministically."""
    _VERTEX_FACE_ADJACENCY_CACHE.clear()


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray


def safe_root_mean_square(values: Tensor, epsilon: float = 1e-12) -> Tensor:
    """Return a differentiable RMS with a finite gradient at zero."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return torch.sqrt(values.square().mean() + epsilon)


def canonical_face_orientation_report(
    reference_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_area_ratio: float = 0.1,
) -> dict[str, Any]:
    """Detect canonical triangle flips and collapses relative to its base topology."""
    reference = np.asarray(reference_vertices, dtype=np.float64)
    deformed = np.asarray(deformed_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if reference.shape != deformed.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Reference and deformed vertices must share shape (V, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Faces must have shape (F, 3)")
    if minimum_area_ratio <= 0:
        raise ValueError("Minimum canonical face-area ratio must be positive")

    reference_edges_a = reference[triangles[:, 1]] - reference[triangles[:, 0]]
    reference_edges_b = reference[triangles[:, 2]] - reference[triangles[:, 0]]
    deformed_edges_a = deformed[triangles[:, 1]] - deformed[triangles[:, 0]]
    deformed_edges_b = deformed[triangles[:, 2]] - deformed[triangles[:, 0]]
    reference_area_vectors = np.cross(reference_edges_a, reference_edges_b)
    deformed_area_vectors = np.cross(deformed_edges_a, deformed_edges_b)
    reference_double_areas = np.linalg.norm(reference_area_vectors, axis=1)
    deformed_double_areas = np.linalg.norm(deformed_area_vectors, axis=1)
    if np.any(reference_double_areas <= 1e-12):
        raise ValueError("Reference topology contains a degenerate triangle")
    cosine = np.sum(reference_area_vectors * deformed_area_vectors, axis=1) / np.maximum(
        reference_double_areas * deformed_double_areas,
        1e-20,
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    area_ratios = deformed_double_areas / reference_double_areas
    flipped = cosine <= 0.0
    collapsed = area_ratios < minimum_area_ratio
    blockers: list[str] = []
    if np.any(flipped):
        blockers.append("canonical_face_orientation_flip")
    if np.any(collapsed):
        blockers.append("canonical_face_area_collapse")
    return {
        "schema_version": "canonical_face_orientation.v1",
        "status": "pass" if not blockers else "fail",
        "face_count": int(triangles.shape[0]),
        "flipped_face_count": int(np.count_nonzero(flipped)),
        "flipped_face_fraction": float(np.mean(flipped)),
        "collapsed_face_count": int(np.count_nonzero(collapsed)),
        "collapsed_face_fraction": float(np.mean(collapsed)),
        "median_orientation_error_degrees": float(np.median(np.rad2deg(np.arccos(cosine)))),
        "minimum_area_ratio": float(np.min(area_ratios)),
        "median_area_ratio": float(np.median(area_ratios)),
        "blockers": blockers,
    }


def canonical_topology_quantities(
    reference_vertices: Tensor,
    deformed_vertices: Tensor,
    faces: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return signed and unsigned face-area ratios to a reference topology."""
    reference_triangles = reference_vertices[faces]
    deformed_triangles = deformed_vertices[faces]
    reference_area_vectors = torch.linalg.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
        dim=-1,
    )
    deformed_area_vectors = torch.linalg.cross(
        deformed_triangles[:, 1] - deformed_triangles[:, 0],
        deformed_triangles[:, 2] - deformed_triangles[:, 0],
        dim=-1,
    )
    reference_double_area_squared = reference_area_vectors.square().sum(dim=-1).clamp_min(1e-20)
    reference_double_area = torch.sqrt(reference_double_area_squared)
    deformed_double_area = torch.linalg.vector_norm(deformed_area_vectors, dim=-1)
    signed_area_ratio = (reference_area_vectors * deformed_area_vectors).sum(
        dim=-1
    ) / reference_double_area_squared
    unsigned_area_ratio = deformed_double_area / reference_double_area
    return signed_area_ratio, unsigned_area_ratio


def canonical_topology_losses(
    reference_vertices: Tensor,
    deformed_vertices: Tensor,
    faces: Tensor,
    edges: Tensor,
    offsets: Tensor,
    *,
    orientation_margin: float,
    minimum_area_ratio: float,
) -> dict[str, Tensor]:
    """Differentiable barriers and distortion penalties for canonical geometry."""
    signed_area_ratio, unsigned_area_ratio = canonical_topology_quantities(
        reference_vertices,
        deformed_vertices,
        faces,
    )
    reference_edge_vectors = reference_vertices[edges[:, 1]] - reference_vertices[edges[:, 0]]
    deformed_edge_vectors = deformed_vertices[edges[:, 1]] - deformed_vertices[edges[:, 0]]
    reference_edge_lengths = torch.linalg.vector_norm(reference_edge_vectors, dim=-1).clamp_min(
        1e-8
    )
    deformed_edge_lengths = torch.linalg.vector_norm(deformed_edge_vectors, dim=-1).clamp_min(1e-8)
    log_edge_ratio = torch.log(deformed_edge_lengths / reference_edge_lengths)
    offset_difference = offsets[edges[:, 1]] - offsets[edges[:, 0]]
    normalized_offset_difference = offset_difference / reference_edge_lengths[:, None]
    return {
        "canonical_orientation": torch.relu(orientation_margin - signed_area_ratio)
        .square()
        .mean()
        .clamp_min(1e-12),
        "canonical_area": torch.relu(minimum_area_ratio - unsigned_area_ratio)
        .square()
        .mean()
        .clamp_min(1e-12),
        "canonical_edge_strain": log_edge_ratio.square().mean().clamp_min(1e-12),
        "canonical_smoothness": normalized_offset_difference.square().mean().clamp_min(1e-12),
    }


def canonical_topology_is_valid(
    reference_vertices: Tensor,
    deformed_vertices: Tensor,
    faces: Tensor,
    *,
    minimum_signed_area_ratio: float,
    minimum_area_ratio: float,
) -> bool:
    """Return whether every face remains oriented and non-collapsed."""
    with torch.no_grad():
        signed_area_ratio, unsigned_area_ratio = canonical_topology_quantities(
            reference_vertices,
            deformed_vertices,
            faces,
        )
        return bool(
            torch.all(signed_area_ratio >= minimum_signed_area_ratio)
            and torch.all(unsigned_area_ratio >= minimum_area_ratio)
        )


def rigid_transform_from_axis_angle(
    rotation_vector: Tensor,
    translation: Tensor,
) -> Tensor:
    """Build differentiable SE(3) matrices from axis-angle and translation."""
    if rotation_vector.shape != translation.shape or rotation_vector.shape[-1] != 3:
        raise ValueError("Rotation vectors and translations must share shape (..., 3)")
    x, y, z = rotation_vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros),
        dim=-1,
    ).reshape(*rotation_vector.shape[:-1], 3, 3)
    rotation = torch.matrix_exp(skew)
    upper = torch.cat((rotation, translation.unsqueeze(-1)), dim=-1)
    bottom = torch.zeros(
        (*rotation_vector.shape[:-1], 1, 4),
        dtype=rotation_vector.dtype,
        device=rotation_vector.device,
    )
    bottom[..., 0, 3] = 1.0
    return torch.cat((upper, bottom), dim=-2)


def positional_encoding(points: Tensor, frequency_count: int) -> Tensor:
    if frequency_count == 0:
        return points
    frequencies = 2.0 ** torch.arange(frequency_count, device=points.device, dtype=points.dtype)
    angles = points.unsqueeze(-2) * frequencies[:, None] * torch.pi
    return torch.cat((points, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)), dim=-1)


class CanonicalSDF(nn.Module):
    """Small native implicit surface network with an analytic sphere prior."""

    def __init__(
        self,
        hidden_dim: int = 128,
        layer_count: int = 4,
        frequency_count: int = 6,
        initial_radius: float = 0.7,
    ) -> None:
        super().__init__()
        self.frequency_count = frequency_count
        encoded_dim = 3 * (1 + 2 * frequency_count)
        layers: list[nn.Module] = []
        for index in range(layer_count):
            input_dim = encoded_dim if index == 0 else hidden_dim
            output_dim = 1 if index == layer_count - 1 else hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if index != layer_count - 1:
                layers.append(nn.Softplus(beta=100))
        self.network = nn.Sequential(*layers)
        self.radius = nn.Parameter(torch.tensor(float(initial_radius)))
        final = next(layer for layer in reversed(layers) if isinstance(layer, nn.Linear))
        nn.init.normal_(final.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(final.bias)

    def forward(self, points: Tensor) -> Tensor:
        sphere_prior = torch.linalg.vector_norm(points, dim=-1, keepdim=True) - self.radius.abs()
        return cast(
            Tensor,
            sphere_prior + self.network(positional_encoding(points, self.frequency_count)),
        )


class ResidualDeformer(nn.Module):
    """Frame-conditioned non-rigid residual displacement field."""

    def __init__(
        self,
        frame_count: int,
        code_dim: int = 32,
        hidden_dim: int = 128,
        layer_count: int = 3,
        frequency_count: int = 4,
    ) -> None:
        super().__init__()
        self.frequency_count = frequency_count
        self.frame_codes = nn.Embedding(frame_count, code_dim)
        nn.init.normal_(self.frame_codes.weight, std=0.01)
        input_dim = 3 * (1 + 2 * frequency_count) + code_dim
        layers: list[nn.Module] = []
        for index in range(layer_count):
            layers.append(
                nn.Linear(
                    input_dim if index == 0 else hidden_dim,
                    3 if index == layer_count - 1 else hidden_dim,
                )
            )
            if index != layer_count - 1:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)
        final = next(layer for layer in reversed(layers) if isinstance(layer, nn.Linear))
        nn.init.normal_(final.weight, std=1e-5)
        nn.init.zeros_(final.bias)

    def forward(self, points: Tensor, frame_indices: Tensor) -> Tensor:
        if frame_indices.ndim == 0:
            frame_indices = frame_indices.expand(points.shape[0])
        code = self.frame_codes(frame_indices)
        if code.ndim == points.ndim - 1:
            code = code.unsqueeze(-2).expand(*points.shape[:-1], code.shape[-1])
        encoded = positional_encoding(points, self.frequency_count)
        return cast(Tensor, self.network(torch.cat((encoded, code), dim=-1)))

    def forward_with_code(self, points: Tensor, code: Tensor) -> Tensor:
        if code.ndim == 1:
            code = code.expand(*points.shape[:-1], code.shape[-1])
        encoded = positional_encoding(points, self.frequency_count)
        return cast(Tensor, self.network(torch.cat((encoded, code), dim=-1)))


def sphere_sdf(points: Tensor, radius: float = 1.0) -> Tensor:
    return cast(Tensor, torch.linalg.vector_norm(points, dim=-1) - radius)


def eikonal_loss(sdf: nn.Module, points: Tensor) -> Tensor:
    sample = points.detach().requires_grad_(True)
    values = sdf(sample)
    gradients = torch.autograd.grad(
        values,
        sample,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
    )[0]
    return cast(Tensor, ((torch.linalg.vector_norm(gradients, dim=-1) - 1.0) ** 2).mean())


def signed_sdf_mesh_consistency(
    sdf: nn.Module,
    vertices: Tensor,
    faces: Tensor,
    *,
    offset_distance: float = 0.02,
    maximum_sample_count: int = 2048,
) -> Tensor:
    """Anchor zero level-set and enforce opposite signs across the proxy surface."""
    if offset_distance <= 0:
        raise ValueError("offset_distance must be positive")
    if maximum_sample_count <= 0:
        raise ValueError("maximum_sample_count must be positive")
    stride = max(1, int(np.ceil(vertices.shape[0] / maximum_sample_count)))
    sample_indices = torch.arange(0, vertices.shape[0], stride, device=vertices.device)[
        :maximum_sample_count
    ]
    surface = vertices[sample_indices]
    normals = vertex_normals(vertices, faces)[sample_indices].detach()
    surface_values = sdf(surface).reshape(-1)
    outside_values = sdf(surface + normals * offset_distance).reshape(-1)
    inside_values = sdf(surface - normals * offset_distance).reshape(-1)
    margin = offset_distance * 0.5
    return cast(
        Tensor,
        surface_values.abs().mean()
        + torch.relu(margin - outside_values).mean()
        + torch.relu(margin + inside_values).mean(),
    )


def extract_sdf_mesh(
    sdf: nn.Module,
    *,
    resolution: int = 64,
    bounds: tuple[float | Sequence[float], float | Sequence[float]] = (-1.2, 1.2),
    chunk_size: int = 65_536,
    device: torch.device | str = "cpu",
) -> Mesh:
    low_raw, high_raw = bounds
    low = np.broadcast_to(np.asarray(low_raw, dtype=np.float32), (3,)).copy()
    high = np.broadcast_to(np.asarray(high_raw, dtype=np.float32), (3,)).copy()
    if np.any(high <= low):
        raise ValueError("Every SDF extraction upper bound must exceed its lower bound")
    coordinates = [
        torch.linspace(float(low[axis]), float(high[axis]), resolution, device=device)
        for axis in range(3)
    ]
    grid = torch.stack(torch.meshgrid(*coordinates, indexing="ij"), dim=-1)
    flat = grid.reshape(-1, 3)
    values: list[Tensor] = []
    with torch.no_grad():
        for chunk in flat.split(chunk_size):  # type: ignore[no-untyped-call]
            values.append(sdf(chunk).reshape(-1).cpu())
    volume = torch.cat(values).reshape(resolution, resolution, resolution).numpy()
    if not float(volume.min()) <= 0.0 <= float(volume.max()):
        raise ValueError("SDF level zero is outside the sampled volume")
    spacing = tuple(((high - low) / (resolution - 1)).tolist())
    vertices, faces, normals, _ = marching_cubes(  # type: ignore[no-untyped-call]
        volume, level=0.0, spacing=spacing
    )
    vertices += low
    return Mesh(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int64),
        normals=normals.astype(np.float32),
    )


def linear_blend_skinning(vertices: Tensor, weights: Tensor, joint_transforms: Tensor) -> Tensor:
    """Apply LBS to ``(V,3)`` vertices for transforms ``(...,J,4,4)``."""
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if weights.shape != (vertices.shape[0], joint_transforms.shape[-3]):
        raise ValueError("weights must have shape (V, J)")
    ones = torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    homogeneous = torch.cat((vertices, ones), dim=-1)
    transformed = torch.einsum("...jab,vb->...jva", joint_transforms, homogeneous)[..., :3]
    return torch.einsum("vj,...jvk->...vk", weights, transformed)


def subdivide_triangular_mesh(
    vertices: Tensor,
    faces: Tensor,
    vertex_attributes: dict[str, Tensor] | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
    """Deterministically split every triangle into four using unique edge midpoints."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Subdivision vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Subdivision faces must have shape [F, 3]")
    attributes = vertex_attributes or {}
    for name, values in attributes.items():
        if values.shape[0] != vertices.shape[0]:
            raise ValueError(f"Subdivision attribute {name!r} has the wrong vertex count")
    face_edges = torch.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=1)
    sorted_edges = torch.sort(face_edges.reshape(-1, 2), dim=-1).values
    unique_edges, inverse = torch.unique(sorted_edges, dim=0, sorted=True, return_inverse=True)
    midpoints = 0.5 * (vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]])
    subdivided_vertices = torch.cat((vertices, midpoints), dim=0)
    midpoint_indices = inverse.reshape(-1, 3) + vertices.shape[0]
    v0, v1, v2 = faces.unbind(-1)
    m01, m12, m20 = midpoint_indices.unbind(-1)
    subdivided_faces = torch.stack(
        (
            torch.stack((v0, m01, m20), dim=-1),
            torch.stack((m01, v1, m12), dim=-1),
            torch.stack((m20, m12, v2), dim=-1),
            torch.stack((m01, m12, m20), dim=-1),
        ),
        dim=1,
    ).reshape(-1, 3)
    subdivided_attributes = {
        name: torch.cat(
            (values, 0.5 * (values[unique_edges[:, 0]] + values[unique_edges[:, 1]])),
            dim=0,
        )
        for name, values in attributes.items()
    }
    return subdivided_vertices, subdivided_faces, subdivided_attributes, unique_edges


def deformation_jacobian(displacements: Tensor, points: Tensor) -> Tensor:
    rows: list[Tensor] = []
    for axis in range(3):
        gradient = torch.autograd.grad(
            displacements[..., axis].sum(),
            points,
            create_graph=True,
            retain_graph=True,
        )[0]
        rows.append(gradient)
    return torch.stack(rows, dim=-2)


def jacobian_regularization(jacobian: Tensor, maximum_singular_value: float = 1.5) -> Tensor:
    identity = torch.eye(3, dtype=jacobian.dtype, device=jacobian.device)
    deformation_gradient = identity + jacobian
    singular_values = torch.linalg.svdvals(deformation_gradient)
    rigidity = (
        (deformation_gradient.transpose(-1, -2) @ deformation_gradient - identity) ** 2
    ).mean()
    foldover = torch.relu(1e-3 - torch.linalg.det(deformation_gradient)).mean()
    stretch = torch.relu(singular_values - maximum_singular_value).square().mean()
    return rigidity + foldover + stretch


def temporal_second_difference(values: Tensor) -> Tensor:
    if values.shape[0] < 3:
        return values.sum() * 0.0
    return (values[2:] - 2.0 * values[1:-1] + values[:-2]).square().mean()


def vertex_face_adjacency(faces: Tensor, vertex_count: int) -> tuple[Tensor, Tensor]:
    """Build fixed-order padded vertex-to-face adjacency for deterministic sums."""
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be a torch.long tensor with shape [F, 3]")
    if vertex_count <= 0:
        raise ValueError("vertex_count must be positive")
    if faces.numel() and (bool(torch.any(faces < 0)) or bool(torch.any(faces >= vertex_count))):
        raise ValueError("faces contain an out-of-range vertex")
    key = (
        faces.device.type,
        faces.device.index,
        faces.data_ptr(),
        int(faces.shape[0]),
        vertex_count,
    )
    cached = _VERTEX_FACE_ADJACENCY_CACHE.get(key)
    if cached is not None and cached[0]() is faces:
        return cached[1], cached[2]
    incident: list[list[int]] = [[] for _ in range(vertex_count)]
    for face_index, face in enumerate(faces.detach().cpu().tolist()):
        for vertex_index in face:
            incident[int(vertex_index)].append(face_index)
    width = max(max((len(values) for values in incident), default=0), 1)
    indices = torch.zeros((vertex_count, width), dtype=torch.long, device=faces.device)
    valid = torch.zeros((vertex_count, width), dtype=torch.bool, device=faces.device)
    for vertex_index, values in enumerate(incident):
        if not values:
            continue
        count = len(values)
        indices[vertex_index, :count] = torch.tensor(values, dtype=torch.long, device=faces.device)
        valid[vertex_index, :count] = True
    _VERTEX_FACE_ADJACENCY_CACHE[key] = (weakref.ref(faces), indices, valid)
    return indices, valid


def vertex_normals(
    vertices: Tensor,
    faces: Tensor,
    adjacency: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    triangles = vertices[faces]
    face_normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
    )
    indices, valid = adjacency or vertex_face_adjacency(faces, int(vertices.shape[0]))
    if indices.device != vertices.device or valid.device != vertices.device:
        raise ValueError("vertex-face adjacency must be on the vertex device")
    contributions = face_normals[indices] * valid.unsqueeze(-1).to(face_normals.dtype)
    result = contributions.sum(dim=1)
    return torch.nn.functional.normalize(result, dim=-1, eps=1e-8)
