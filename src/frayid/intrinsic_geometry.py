"""Full-rank intrinsic coordinates for explicit surface optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from frayid.eulerian_reconstruction import ExplicitStepResult, project_explicit_step


def _unique_edges(faces: Tensor) -> Tensor:
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0)
    unique_edges: Tensor = torch.unique(torch.sort(edges, dim=1).values, dim=0)
    return unique_edges


@dataclass(frozen=True)
class IntrinsicTransformReport:
    vertex_count: int
    rank: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    relative_round_trip_error: float
    relative_solve_residual: float
    median_edge_length: float
    mean_lumped_mass: float
    stiffness_scale: float
    lambda_value: float
    status: str

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "vertex_count": self.vertex_count,
            "rank": self.rank,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "condition_number": self.condition_number,
            "relative_round_trip_error": self.relative_round_trip_error,
            "relative_solve_residual": self.relative_solve_residual,
            "median_edge_length": self.median_edge_length,
            "mean_lumped_mass": self.mean_lumped_mass,
            "stiffness_scale": self.stiffness_scale,
            "lambda": self.lambda_value,
            "status": self.status,
        }


@dataclass(frozen=True)
class IntrinsicGeometryTransform:
    """Frozen invertible map ``U=A V`` with ``A=M+lambda*L``."""

    matrix: Tensor
    cholesky: Tensor
    lumped_mass: Tensor
    stiffness: Tensor
    lambda_value: float
    median_edge_length: float
    mean_lumped_mass: float
    stiffness_scale: float

    @classmethod
    def from_mesh(
        cls, vertices: Tensor, faces: Tensor, *, lambda_value: float = 1.0
    ) -> IntrinsicGeometryTransform:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if vertices.dtype != torch.float64:
            raise ValueError("intrinsic coordinates require float64 vertices")
        if not torch.isfinite(vertices).all():
            raise ValueError("vertices must be finite")
        if lambda_value < 0.0:
            raise ValueError("lambda must be nonnegative")
        vertex_count = vertices.shape[0]
        if torch.any(faces < 0) or torch.any(faces >= vertex_count):
            raise ValueError("face index is out of range")

        triangles = vertices[faces]
        edge_01 = triangles[:, 1] - triangles[:, 0]
        edge_02 = triangles[:, 2] - triangles[:, 0]
        double_area = torch.linalg.vector_norm(torch.cross(edge_01, edge_02, dim=1), dim=1)
        if torch.any(double_area <= 0.0):
            raise ValueError("intrinsic transform requires nondegenerate faces")
        area = 0.5 * double_area
        lumped = torch.zeros(vertex_count, dtype=vertices.dtype, device=vertices.device)
        share = area / 3.0
        for corner in range(3):
            lumped.scatter_add_(0, faces[:, corner], share)
        if torch.any(lumped <= 0.0):
            raise ValueError("every vertex must have positive lumped mass")
        mean_mass = lumped.mean()
        normalized_mass = lumped / mean_mass

        cotangent_0 = (edge_01 * edge_02).sum(dim=1) / double_area
        edge_10 = triangles[:, 0] - triangles[:, 1]
        edge_12 = triangles[:, 2] - triangles[:, 1]
        cotangent_1 = (edge_10 * edge_12).sum(dim=1) / double_area
        edge_20 = triangles[:, 0] - triangles[:, 2]
        edge_21 = triangles[:, 1] - triangles[:, 2]
        cotangent_2 = (edge_20 * edge_21).sum(dim=1) / double_area

        stiffness = torch.zeros(
            (vertex_count, vertex_count), dtype=vertices.dtype, device=vertices.device
        )
        for left, right, weight in (
            (faces[:, 1], faces[:, 2], 0.5 * cotangent_0),
            (faces[:, 2], faces[:, 0], 0.5 * cotangent_1),
            (faces[:, 0], faces[:, 1], 0.5 * cotangent_2),
        ):
            stiffness.index_put_((left, left), weight, accumulate=True)
            stiffness.index_put_((right, right), weight, accumulate=True)
            stiffness.index_put_((left, right), -weight, accumulate=True)
            stiffness.index_put_((right, left), -weight, accumulate=True)

        edges = _unique_edges(faces)
        edge_lengths = torch.linalg.vector_norm(
            vertices[edges[:, 1]] - vertices[edges[:, 0]], dim=1
        )
        median_edge = torch.median(edge_lengths)
        stiffness_scale = median_edge.square() / mean_mass
        scaled_stiffness = stiffness_scale * stiffness
        matrix = torch.diag(normalized_mass) + lambda_value * scaled_stiffness
        matrix = 0.5 * (matrix + matrix.T)
        cholesky = torch.linalg.cholesky(matrix)
        return cls(
            matrix=matrix,
            cholesky=cholesky,
            lumped_mass=normalized_mass,
            stiffness=scaled_stiffness,
            lambda_value=float(lambda_value),
            median_edge_length=float(median_edge),
            mean_lumped_mass=float(mean_mass),
            stiffness_scale=float(stiffness_scale),
        )

    def encode(self, vertices: Tensor) -> Tensor:
        if vertices.shape != (self.matrix.shape[0], 3):
            raise ValueError("vertex array shape does not match intrinsic transform")
        return self.matrix @ vertices

    def decode(self, coordinates: Tensor) -> Tensor:
        if coordinates.shape != (self.matrix.shape[0], 3):
            raise ValueError("coordinate array shape does not match intrinsic transform")
        return torch.cholesky_solve(coordinates, self.cholesky)

    def report(self, reference_vertices: Tensor) -> IntrinsicTransformReport:
        eigenvalues = torch.linalg.eigvalsh(self.matrix)
        decoded = self.decode(self.encode(reference_vertices))
        denominator = torch.linalg.vector_norm(reference_vertices).clamp_min(
            torch.finfo(reference_vertices.dtype).eps
        )
        round_trip = torch.linalg.vector_norm(decoded - reference_vertices) / denominator
        encoded = self.encode(reference_vertices)
        solved = self.decode(encoded)
        residual = torch.linalg.vector_norm(
            self.matrix @ solved - encoded
        ) / torch.linalg.vector_norm(encoded).clamp_min(torch.finfo(encoded.dtype).eps)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        rank = int(torch.linalg.matrix_rank(self.matrix))
        status = bool(
            rank == self.matrix.shape[0]
            and minimum > 0.0
            and float(round_trip) <= 1.0e-12
            and float(residual) <= 1.0e-12
        )
        return IntrinsicTransformReport(
            vertex_count=self.matrix.shape[0],
            rank=rank,
            minimum_eigenvalue=minimum,
            maximum_eigenvalue=maximum,
            condition_number=maximum / minimum,
            relative_round_trip_error=float(round_trip),
            relative_solve_residual=float(residual),
            median_edge_length=self.median_edge_length,
            mean_lumped_mass=self.mean_lumped_mass,
            stiffness_scale=self.stiffness_scale,
            lambda_value=self.lambda_value,
            status="pass" if status else "fail",
        )


def _damp_adam_state(optimizer: torch.optim.Optimizer, parameter: Tensor, scale: float) -> None:
    state = optimizer.state.get(parameter, {})
    first_moment = state.get("exp_avg")
    second_moment = state.get("exp_avg_sq")
    if isinstance(first_moment, Tensor):
        first_moment.mul_(scale)
    if isinstance(second_moment, Tensor):
        second_moment.mul_(scale * scale)


def project_intrinsic_step(
    transform: IntrinsicGeometryTransform,
    coordinates: Tensor,
    previous_coordinates: Tensor,
    reference_vertices: Tensor,
    faces: Tensor,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    signed_area_floor: float = 0.01,
    unsigned_area_floor: float = 0.10,
    maximum_backtracks: int = 32,
) -> ExplicitStepResult:
    """Project a proposed intrinsic step using the explicit complete-path judge."""
    previous_vertices = transform.decode(previous_coordinates)
    candidate_vertices = transform.decode(coordinates.detach().clone())
    result = project_explicit_step(
        candidate_vertices,
        previous_vertices,
        reference_vertices,
        faces,
        signed_area_floor=signed_area_floor,
        unsigned_area_floor=unsigned_area_floor,
        maximum_backtracks=maximum_backtracks,
    )
    with torch.no_grad():
        if result.rejected:
            coordinates.copy_(previous_coordinates)
        elif result.accepted_scale < 1.0:
            coordinates.copy_(transform.encode(candidate_vertices))
    if optimizer is not None and result.accepted_scale < 1.0:
        _damp_adam_state(optimizer, coordinates, result.accepted_scale)
    return result
