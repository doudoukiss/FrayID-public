from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor


def interpolate_zero_crossings(positions: Tensor, edges: Tensor, values: Tensor) -> Tensor:
    """Extract shared PL zero crossings while preserving gradients to ``values``."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape [E,2]")
    if values.shape != (positions.shape[0],):
        raise ValueError("values must have shape [N]")
    edge_values = values[edges]
    if torch.any(edge_values[:, 0] * edge_values[:, 1] >= 0):
        raise ValueError("every extracted edge must have strictly opposite signs")
    endpoints = positions[edges]
    interpolation = edge_values[:, 0] / (edge_values[:, 0] - edge_values[:, 1])
    return endpoints[:, 0] + interpolation[:, None] * (endpoints[:, 1] - endpoints[:, 0])


def same_open_sign_chamber(start: Tensor, end: Tensor) -> bool:
    if start.shape != end.shape or start.ndim != 1:
        raise ValueError("sign-chamber values must have identical vector shapes")
    if not torch.isfinite(start).all() or not torch.isfinite(end).all():
        return False
    return bool(torch.all(start != 0) and torch.all(end != 0) and torch.equal(start < 0, end < 0))


@dataclass(frozen=True)
class SurfacePathCertificate:
    status: str
    signed_area_floor: float
    unsigned_area_floor: float
    minimum_signed_area_lower_bound: float
    minimum_unsigned_area_lower_bound: float
    subdivision_depth: int
    interval_count: int
    blocker: str | None = None

    def report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "signed_area_floor": self.signed_area_floor,
            "unsigned_area_floor": self.unsigned_area_floor,
            "minimum_signed_area_lower_bound": self.minimum_signed_area_lower_bound,
            "minimum_unsigned_area_lower_bound": self.minimum_unsigned_area_lower_bound,
            "subdivision_depth": self.subdivision_depth,
            "interval_count": self.interval_count,
            "blocker": self.blocker,
        }


def _down(values: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.nextafter(values, -np.inf))


def _up(values: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.nextafter(values, np.inf))


def _interval_add(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return _down(first[0] + second[0]), _up(first[1] + second[1])


def _interval_subtract(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return _down(first[0] - second[1]), _up(first[1] - second[0])


def _interval_multiply(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    products = np.stack(
        (
            first[0] * second[0],
            first[0] * second[1],
            first[1] * second[0],
            first[1] * second[1],
        ),
        axis=0,
    )
    return _down(products.min(axis=0)), _up(products.max(axis=0))


def _interval_divide(
    numerator: tuple[np.ndarray, np.ndarray], denominator: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    if np.any((denominator[0] <= 0) & (denominator[1] >= 0)):
        raise ZeroDivisionError("interval denominator contains zero")
    reciprocal = (1.0 / denominator[1], 1.0 / denominator[0])
    reciprocal = (np.minimum(*reciprocal), np.maximum(*reciprocal))
    return _interval_multiply(numerator, reciprocal)


def _affine_ranges(
    start: np.ndarray, end: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta = end - start
    scale_shape = (bounds.shape[0],) + (1,) * start.ndim
    first = start[None, ...] + bounds[:, 0].reshape(scale_shape) * delta[None, ...]
    second = start[None, ...] + bounds[:, 1].reshape(scale_shape) * delta[None, ...]
    return _down(np.minimum(first, second)), _up(np.maximum(first, second))


def _surface_interval_lowers(
    vertex_lower: np.ndarray,
    vertex_upper: np.ndarray,
    faces: np.ndarray,
    reference_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    triangles_lower = vertex_lower[:, faces]
    triangles_upper = vertex_upper[:, faces]
    first = _interval_subtract(
        (triangles_lower[:, :, 1], triangles_upper[:, :, 1]),
        (triangles_lower[:, :, 0], triangles_upper[:, :, 0]),
    )
    second = _interval_subtract(
        (triangles_lower[:, :, 2], triangles_upper[:, :, 2]),
        (triangles_lower[:, :, 0], triangles_upper[:, :, 0]),
    )
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b, c, d in ((1, 2, 2, 1), (2, 0, 0, 2), (0, 1, 1, 0)):
        product_a = _interval_multiply(
            (first[0][..., a], first[1][..., a]), (second[0][..., b], second[1][..., b])
        )
        product_b = _interval_multiply(
            (first[0][..., c], first[1][..., c]), (second[0][..., d], second[1][..., d])
        )
        component = _interval_subtract(product_a, product_b)
        components.append(component)
    normal_lower = np.stack([value[0] for value in components], axis=-1)
    normal_upper = np.stack([value[1] for value in components], axis=-1)

    coefficient = reference_normals[None, :, :]
    signed_terms = _interval_multiply(
        (normal_lower, normal_upper),
        (coefficient, coefficient),
    )
    signed_lower = _down(signed_terms[0].sum(axis=-1))
    reference_squared = np.square(reference_normals).sum(axis=-1)[None, :]
    signed_ratio_lower = _down(signed_lower / reference_squared)

    crosses_zero = (normal_lower <= 0) & (normal_upper >= 0)
    minimum_absolute = np.where(
        crosses_zero,
        0.0,
        np.minimum(np.abs(normal_lower), np.abs(normal_upper)),
    )
    norm_squared_lower = _down(np.square(minimum_absolute).sum(axis=-1))
    unsigned_ratio_lower = _down(
        np.sqrt(np.maximum(norm_squared_lower, 0.0)) / np.sqrt(reference_squared)
    )
    return signed_ratio_lower, unsigned_ratio_lower


def surface_endpoint_area_ratios(
    vertices: np.ndarray, reference_vertices: np.ndarray, faces: np.ndarray
) -> tuple[float, float]:
    endpoint = np.asarray(vertices, dtype=np.float64)
    reference = np.asarray(reference_vertices, dtype=np.float64)
    triangles = endpoint[np.asarray(faces, dtype=np.int64)]
    reference_triangles = reference[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    reference_normals = np.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
    )
    reference_squared = np.square(reference_normals).sum(axis=-1)
    if np.any(reference_squared == 0.0):
        raise ValueError("reference surface contains a degenerate face")
    signed = np.sum(normals * reference_normals, axis=-1) / reference_squared
    unsigned = np.linalg.norm(normals, axis=-1) / np.sqrt(reference_squared)
    return float(np.min(signed)), float(np.min(unsigned))


def _certify_interval_vertices(
    vertex_interval_builder: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    reference_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    signed_area_floor: float,
    unsigned_area_floor: float,
    maximum_subdivision_depth: int,
) -> SurfacePathCertificate:
    triangles = reference_vertices[faces]
    reference_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    if np.any(np.linalg.norm(reference_normals, axis=1) == 0):
        return SurfacePathCertificate(
            "fail",
            signed_area_floor,
            unsigned_area_floor,
            -np.inf,
            0.0,
            0,
            1,
            "degenerate_reference_face",
        )
    minimum_signed = -np.inf
    minimum_unsigned = 0.0
    for depth in range(maximum_subdivision_depth + 1):
        count = 1 << depth
        grid = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
        bounds = np.stack((grid[:-1], grid[1:]), axis=1)
        try:
            lower, upper = vertex_interval_builder(bounds)
        except ZeroDivisionError:
            return SurfacePathCertificate(
                "fail",
                signed_area_floor,
                unsigned_area_floor,
                -np.inf,
                0.0,
                depth,
                count,
                "zero_crossing_denominator",
            )
        signed, unsigned = _surface_interval_lowers(lower, upper, faces, reference_normals)
        minimum_signed = float(np.min(signed))
        minimum_unsigned = float(np.min(unsigned))
        if minimum_signed >= signed_area_floor and minimum_unsigned >= unsigned_area_floor:
            return SurfacePathCertificate(
                "pass",
                signed_area_floor,
                unsigned_area_floor,
                minimum_signed,
                minimum_unsigned,
                depth,
                count,
            )
    return SurfacePathCertificate(
        "unknown",
        signed_area_floor,
        unsigned_area_floor,
        minimum_signed,
        minimum_unsigned,
        maximum_subdivision_depth,
        1 << maximum_subdivision_depth,
        "interval_subdivision_exhausted",
    )


def certify_zero_set_scalar_path(
    positions: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    start_values: np.ndarray,
    end_values: np.ndarray,
    reference_vertices: np.ndarray,
    *,
    signed_area_floor: float = 0.01,
    unsigned_area_floor: float = 0.10,
    maximum_subdivision_depth: int = 12,
) -> SurfacePathCertificate:
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    start = np.asarray(start_values, dtype=np.float64)
    end = np.asarray(end_values, dtype=np.float64)
    if start.shape != end.shape or start.shape != (positions.shape[0],):
        raise ValueError("scalar endpoints must have one value per domain vertex")
    if np.any(start == 0) or np.any(end == 0) or not np.array_equal(start < 0, end < 0):
        return SurfacePathCertificate(
            "fail",
            signed_area_floor,
            unsigned_area_floor,
            -np.inf,
            0.0,
            0,
            1,
            "sign_chamber_violation",
        )
    edge_start = start[edges]
    edge_end = end[edges]
    if np.any(edge_start[:, 0] * edge_start[:, 1] >= 0) or np.any(
        edge_end[:, 0] * edge_end[:, 1] >= 0
    ):
        return SurfacePathCertificate(
            "fail",
            signed_area_floor,
            unsigned_area_floor,
            -np.inf,
            0.0,
            0,
            1,
            "surface_edge_sign_violation",
        )

    def builder(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = _affine_ranges(edge_start[:, 0], edge_end[:, 0], bounds)
        second = _affine_ranges(edge_start[:, 1], edge_end[:, 1], bounds)
        denominator = _interval_subtract(first, second)
        interpolation = _interval_divide(first, denominator)
        delta = positions[edges[:, 1]] - positions[edges[:, 0]]
        product = _interval_multiply(
            (interpolation[0][..., None], interpolation[1][..., None]),
            (delta[None, ...], delta[None, ...]),
        )
        base = positions[edges[:, 0]][None, ...]
        return _interval_add((base, base), product)

    return _certify_interval_vertices(
        builder,
        np.asarray(reference_vertices, dtype=np.float64),
        faces,
        signed_area_floor=signed_area_floor,
        unsigned_area_floor=unsigned_area_floor,
        maximum_subdivision_depth=maximum_subdivision_depth,
    )


def certify_linear_surface_path(
    start_vertices: np.ndarray,
    end_vertices: np.ndarray,
    faces: np.ndarray,
    reference_vertices: np.ndarray,
    *,
    signed_area_floor: float = 0.01,
    unsigned_area_floor: float = 0.10,
    maximum_subdivision_depth: int = 12,
) -> SurfacePathCertificate:
    start = np.asarray(start_vertices, dtype=np.float64)
    end = np.asarray(end_vertices, dtype=np.float64)

    def builder(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _affine_ranges(start, end, bounds)

    return _certify_interval_vertices(
        builder,
        np.asarray(reference_vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        signed_area_floor=signed_area_floor,
        unsigned_area_floor=unsigned_area_floor,
        maximum_subdivision_depth=maximum_subdivision_depth,
    )
