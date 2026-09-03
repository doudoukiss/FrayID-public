from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from frayid.geometry import vertex_normals


def regular_tetrahedral_grid(
    resolution: int, *, extent: float = 1.4, device: torch.device | str | None = None
) -> tuple[Tensor, Tensor]:
    """Create a conforming six-tetrahedra-per-cube Cartesian grid."""
    if resolution < 3:
        raise ValueError("tetrahedral grid resolution must be at least 3")
    if extent <= 0:
        raise ValueError("tetrahedral grid extent must be positive")
    coordinates = torch.linspace(-extent, extent, resolution, device=device)
    return tetrahedral_grid_from_axes((coordinates, coordinates, coordinates))


def tetrahedral_grid_from_axes(axes: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Create a conforming grid from x/y/z coordinate axes."""
    if any(axis.ndim != 1 or len(axis) < 3 for axis in axes):
        raise ValueError("each tetrahedral coordinate axis must have at least three values")
    x_coordinates, y_coordinates, z_coordinates = axes
    xx, yy, zz = torch.meshgrid(x_coordinates, y_coordinates, z_coordinates, indexing="ij")
    positions = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    x_count, y_count, z_count = (len(axis) for axis in axes)

    def index(x: int, y: int, z: int) -> int:
        return x * y_count * z_count + y * z_count + z

    local_tetrahedra = (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    )
    tetrahedra: list[list[int]] = []
    for x in range(x_count - 1):
        for y in range(y_count - 1):
            for z in range(z_count - 1):
                corners = (
                    index(x, y, z),
                    index(x + 1, y, z),
                    index(x + 1, y + 1, z),
                    index(x, y + 1, z),
                    index(x, y, z + 1),
                    index(x + 1, y, z + 1),
                    index(x + 1, y + 1, z + 1),
                    index(x, y + 1, z + 1),
                )
                tetrahedra.extend([[corners[i] for i in tetra] for tetra in local_tetrahedra])
    return positions, torch.tensor(tetrahedra, dtype=torch.long, device=positions.device)


def tetrahedron_signed_volumes(positions: Tensor, tetrahedra: Tensor) -> Tensor:
    vertices = positions[tetrahedra]
    volumes: Tensor = (
        torch.linalg.det(
            torch.stack(
                (
                    vertices[:, 1] - vertices[:, 0],
                    vertices[:, 2] - vertices[:, 0],
                    vertices[:, 3] - vertices[:, 0],
                ),
                dim=-1,
            )
        )
        / 6.0
    )
    return volumes


def fixed_sign_surface_connectivity(
    positions: Tensor, tetrahedra: Tensor, signs: Tensor
) -> tuple[Tensor, Tensor]:
    """Build one deterministic marching-tetrahedra topology for fixed signs."""
    if signs.shape != (positions.shape[0],):
        raise ValueError("signs must have one value per grid vertex")
    if torch.any(signs == 0):
        raise ValueError("fixed signs cannot contain zero")
    edge_lookup: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    faces: list[list[int]] = []
    outward_directions: list[Tensor] = []

    def edge_vertex(first: int, second: int) -> int:
        key = (min(first, second), max(first, second))
        if key not in edge_lookup:
            edge_lookup[key] = len(edges)
            edges.append(key)
        return edge_lookup[key]

    sign_values = signs.detach().cpu().tolist()
    for tetrahedron in tetrahedra.detach().cpu().tolist():
        inside = [vertex for vertex in tetrahedron if sign_values[vertex] < 0]
        outside = [vertex for vertex in tetrahedron if sign_values[vertex] > 0]
        if not inside or not outside:
            continue
        if len(inside) in (1, 3):
            singleton = inside[0] if len(inside) == 1 else outside[0]
            opposite = outside if len(inside) == 1 else inside
            triangle = [edge_vertex(singleton, vertex) for vertex in opposite]
            faces.append(triangle)
            inside_center = positions[torch.tensor(inside, device=positions.device)].mean(dim=0)
            outside_center = positions[torch.tensor(outside, device=positions.device)].mean(dim=0)
            outward_directions.append(outside_center - inside_center)
            continue
        first_inside, second_inside = inside
        first_outside, second_outside = outside
        a = edge_vertex(first_inside, first_outside)
        b = edge_vertex(first_inside, second_outside)
        c = edge_vertex(second_inside, first_outside)
        d = edge_vertex(second_inside, second_outside)
        faces.extend(([a, b, c], [b, d, c]))
        inside_center = positions[torch.tensor(inside, device=positions.device)].mean(dim=0)
        outside_center = positions[torch.tensor(outside, device=positions.device)].mean(dim=0)
        outward = outside_center - inside_center
        outward_directions.extend((outward, outward))
    if not faces:
        raise ValueError("fixed sign pattern does not intersect the tetrahedral grid")
    edge_tensor = torch.tensor(edges, dtype=torch.long, device=positions.device)
    face_tensor = torch.tensor(faces, dtype=torch.long, device=positions.device)
    midpoint_vertices = positions[edge_tensor].mean(dim=1)
    triangles = midpoint_vertices[face_tensor]
    normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
    )
    outward = torch.stack(outward_directions)
    flip = (normals * outward).sum(dim=-1) < 0
    face_tensor[flip] = face_tensor[flip][:, [0, 2, 1]]
    return edge_tensor, face_tensor


class FixedSignTetrahedralField(nn.Module):
    """Bounded implicit values with immutable signs and surface connectivity."""

    positions: Tensor
    tetrahedra: Tensor
    signs: Tensor
    surface_edges: Tensor
    surface_faces: Tensor
    magnitude_logits: nn.Parameter

    def __init__(
        self,
        positions: Tensor,
        tetrahedra: Tensor,
        initial_values: Tensor,
        *,
        minimum_absolute_value: float = 1e-3,
        maximum_absolute_value: float = 4.0,
    ) -> None:
        super().__init__()
        if initial_values.shape != (positions.shape[0],):
            raise ValueError("initial_values must have one value per grid vertex")
        if minimum_absolute_value <= 0 or maximum_absolute_value <= minimum_absolute_value:
            raise ValueError("invalid fixed-sign magnitude bounds")
        if torch.any(initial_values == 0):
            raise ValueError("initial implicit values cannot be zero")
        signs = torch.sign(initial_values)
        edges, faces = fixed_sign_surface_connectivity(positions, tetrahedra, signs)
        self.register_buffer("positions", positions.detach().clone())
        self.register_buffer("tetrahedra", tetrahedra.detach().clone())
        self.register_buffer("signs", signs.detach().clone())
        self.register_buffer("surface_edges", edges)
        self.register_buffer("surface_faces", faces)
        self.minimum_absolute_value = minimum_absolute_value
        self.maximum_absolute_value = maximum_absolute_value
        normalized = (
            (initial_values.abs() - minimum_absolute_value)
            / (maximum_absolute_value - minimum_absolute_value)
        ).clamp(1e-5, 1.0 - 1e-5)
        self.magnitude_logits = nn.Parameter(torch.logit(normalized))

    @property
    def field_values(self) -> Tensor:
        magnitudes = self.minimum_absolute_value + (
            self.maximum_absolute_value - self.minimum_absolute_value
        ) * torch.sigmoid(self.magnitude_logits)
        return self.signs * magnitudes

    def surface_vertices(self) -> Tensor:
        values = self.field_values
        endpoints = self.positions[self.surface_edges]
        edge_values = values[self.surface_edges]
        interpolation = edge_values[:, 0] / (edge_values[:, 0] - edge_values[:, 1])
        return endpoints[:, 0] + interpolation[:, None] * (endpoints[:, 1] - endpoints[:, 0])

    def minimum_sign_margin(self) -> Tensor:
        return self.field_values.abs().amin()


def symmetric_chamfer_distance(first: Tensor, second: Tensor) -> Tensor:
    distances = torch.cdist(first, second).square()
    return distances.amin(dim=1).mean() + distances.amin(dim=0).mean()


def corresponding_normal_loss(vertices: Tensor, target_vertices: Tensor, faces: Tensor) -> Tensor:
    normals = vertex_normals(vertices, faces)
    targets = vertex_normals(target_vertices, faces)
    return (1.0 - (normals * targets).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def minimum_face_area_ratio(vertices: Tensor, reference: Tensor, faces: Tensor) -> Tensor:
    triangles = vertices[faces]
    reference_triangles = reference[faces]
    areas = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
    ).norm(dim=-1)
    reference_areas = torch.linalg.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
        dim=-1,
    ).norm(dim=-1)
    minimum_ratio: Tensor = (areas / reference_areas.clamp_min(torch.finfo(areas.dtype).eps)).amin()
    return minimum_ratio


def surface_orientation_counts(
    vertices: Tensor, reference: Tensor, faces: Tensor, *, minimum_area_ratio: float
) -> tuple[int, int, float]:
    triangles = vertices[faces]
    reference_triangles = reference[faces]
    normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
    )
    reference_normals = torch.linalg.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
        dim=-1,
    )
    areas = normals.norm(dim=-1)
    reference_areas = reference_normals.norm(dim=-1)
    ratios = areas / reference_areas.clamp_min(torch.finfo(areas.dtype).eps)
    flips = int(((normals * reference_normals).sum(dim=-1) <= 0).sum().detach().cpu())
    collapses = int((ratios < minimum_area_ratio).sum().detach().cpu())
    return flips, collapses, float(ratios.amin().detach().cpu())


def project_fixed_sign_step(
    field: FixedSignTetrahedralField,
    previous_logits: Tensor,
    reference_vertices: Tensor,
    *,
    minimum_area_ratio: float = 0.1,
    maximum_backtracks: int = 16,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Backtrack one implicit update until its explicit zero set is valid."""
    candidate = field.magnitude_logits.detach().clone()
    delta = candidate - previous_logits
    scale = 1.0
    with torch.no_grad():
        for _ in range(maximum_backtracks + 1):
            field.magnitude_logits.copy_(previous_logits + scale * delta)
            flips, collapses, _ = surface_orientation_counts(
                field.surface_vertices(),
                reference_vertices,
                field.surface_faces,
                minimum_area_ratio=minimum_area_ratio,
            )
            if flips == 0 and collapses == 0:
                if optimizer is not None and scale < 1.0:
                    _damp_optimizer_state(optimizer, field.magnitude_logits, scale)
                return scale
            scale *= 0.5
        field.magnitude_logits.copy_(previous_logits)
        if optimizer is not None:
            _damp_optimizer_state(optimizer, field.magnitude_logits, 0.0)
    return 0.0


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


def ellipsoid_implicit_values(positions: Tensor, axes: Tensor) -> Tensor:
    if axes.shape != (3,) or torch.any(axes <= 0):
        raise ValueError("ellipsoid axes must be a positive length-three tensor")
    values: Tensor = torch.linalg.vector_norm(positions / axes, dim=-1) - 1.0
    return values


def distorted_fixed_sign_initialization(target_values: Tensor) -> Tensor:
    """Create a deterministic, same-sign synthetic starting field."""
    phase = torch.arange(
        target_values.numel(), device=target_values.device, dtype=target_values.dtype
    )
    scale = 1.0 + 0.45 * torch.sin(phase * math.pi * (math.sqrt(5.0) - 1.0))
    return torch.sign(target_values) * target_values.abs().clamp_min(0.02) * scale
