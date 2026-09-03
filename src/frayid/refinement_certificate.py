from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from frayid.source_exclusion_carrier import MAXIMUM_FACE_COUNT


@dataclass(frozen=True)
class RefinementProvenance:
    vertices: np.ndarray
    faces: np.ndarray
    parent_face_indices: np.ndarray
    barycentric_numerators: np.ndarray
    denominator: int


@dataclass(frozen=True)
class ExactRefinementCertificate:
    status: str
    rounds: int
    denominator: int
    parent_face_count: int
    child_face_count: int
    children_per_parent: int
    parent_vertices_retained_bitwise: bool
    parent_on_grid: bool
    refined_on_grid: bool
    nonnegative_barycentric_numerators: bool
    barycentric_row_sums_valid: bool
    parent_child_counts_valid: bool
    standard_partition_valid: bool
    exact_integer_affine_reconstruction: bool
    blockers: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "post_v1_p2_exact_refinement_certificate.v1",
            "status": self.status,
            "rounds": self.rounds,
            "denominator": self.denominator,
            "parent_face_count": self.parent_face_count,
            "child_face_count": self.child_face_count,
            "children_per_parent": self.children_per_parent,
            "parent_vertices_retained_bitwise": self.parent_vertices_retained_bitwise,
            "parent_on_grid": self.parent_on_grid,
            "refined_on_grid": self.refined_on_grid,
            "nonnegative_barycentric_numerators": (self.nonnegative_barycentric_numerators),
            "barycentric_row_sums_valid": self.barycentric_row_sums_valid,
            "parent_child_counts_valid": self.parent_child_counts_valid,
            "standard_partition_valid": self.standard_partition_valid,
            "exact_integer_affine_reconstruction": (self.exact_integer_affine_reconstruction),
            "blockers": list(self.blockers),
        }


def _midpoint(
    points: list[np.ndarray],
    midpoints: dict[tuple[int, int], int],
    first: int,
    second: int,
) -> int:
    edge = (min(first, second), max(first, second))
    if edge not in midpoints:
        midpoints[edge] = len(points)
        points.append(0.5 * (points[edge[0]] + points[edge[1]]))
    return midpoints[edge]


def _barycentric_children(barycentric: np.ndarray) -> tuple[np.ndarray, ...]:
    a, b, c = barycentric
    ab = a + b
    bc = b + c
    ca = c + a
    return (
        np.stack((2 * a, ab, ca)),
        np.stack((ab, 2 * b, bc)),
        np.stack((ca, bc, 2 * c)),
        np.stack((ab, bc, ca)),
    )


def subdivide_with_exact_provenance(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    rounds: int = 2,
) -> RefinementProvenance:
    """Uniformly subdivide while retaining per-child original-face provenance."""
    if rounds < 0:
        raise ValueError("subdivision rounds cannot be negative")
    points = [row.copy() for row in np.asarray(vertices, dtype=np.float64)]
    triangles = np.asarray(faces, dtype=np.int64)
    parent_indices = np.arange(len(triangles), dtype=np.int64)
    identity = np.eye(3, dtype=np.int64)
    barycentrics = np.repeat(identity[None, :, :], len(triangles), axis=0)
    denominator = 1

    for _ in range(rounds):
        midpoints: dict[tuple[int, int], int] = {}
        children: list[tuple[int, int, int]] = []
        child_parents: list[int] = []
        child_barycentrics: list[np.ndarray] = []
        for face, parent_index, barycentric in zip(
            triangles, parent_indices, barycentrics, strict=True
        ):
            a, b, c = (int(value) for value in face)
            ab = _midpoint(points, midpoints, a, b)
            bc = _midpoint(points, midpoints, b, c)
            ca = _midpoint(points, midpoints, c, a)
            children.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
            child_parents.extend((int(parent_index),) * 4)
            child_barycentrics.extend(_barycentric_children(barycentric))
        triangles = np.asarray(children, dtype=np.int64)
        parent_indices = np.asarray(child_parents, dtype=np.int64)
        barycentrics = np.asarray(child_barycentrics, dtype=np.int64)
        denominator *= 2
        if len(triangles) > MAXIMUM_FACE_COUNT:
            raise ValueError("subdivision_face_cap")

    return RefinementProvenance(
        vertices=np.asarray(points, dtype=np.float64),
        faces=triangles,
        parent_face_indices=parent_indices,
        barycentric_numerators=barycentrics,
        denominator=denominator,
    )


def _canonical_partition(rounds: int) -> tuple[tuple[int, ...], ...]:
    triangles = [np.eye(3, dtype=np.int64)]
    for _ in range(rounds):
        triangles = [child for triangle in triangles for child in _barycentric_children(triangle)]
    return tuple(sorted(tuple(int(value) for value in triangle.flat) for triangle in triangles))


def _on_grid(values: np.ndarray, grid: float) -> tuple[bool, np.ndarray]:
    if not np.isfinite(grid) or grid <= 0.0:
        return False, np.zeros_like(values, dtype=np.int64)
    scaled = values / grid
    safe = bool(
        np.all(np.isfinite(scaled))
        and np.max(np.abs(scaled), initial=0.0) <= np.iinfo(np.int64).max // 16
    )
    if not safe:
        return False, np.zeros_like(values, dtype=np.int64)
    integers = np.rint(scaled).astype(np.int64)
    return bool(np.array_equal(values, integers.astype(np.float64) * grid)), integers


def certify_exact_dyadic_refinement(
    parent_vertices: np.ndarray,
    parent_faces: np.ndarray,
    refinement: RefinementProvenance,
    *,
    parent_grid: float,
    rounds: int = 2,
) -> ExactRefinementCertificate:
    """Certify surface identity using dyadic integers, not proximity queries."""
    vertices = np.asarray(parent_vertices, dtype=np.float64)
    faces = np.asarray(parent_faces, dtype=np.int64)
    expected_denominator = 2**rounds
    expected_children = 4**rounds
    blockers: list[str] = []

    retained = bool(
        len(refinement.vertices) >= len(vertices)
        and np.array_equal(refinement.vertices[: len(vertices)], vertices)
    )
    parent_on_grid, parent_integers = _on_grid(vertices, parent_grid)
    refined_on_grid, refined_integers = _on_grid(
        refinement.vertices, parent_grid / expected_denominator
    )
    numerators = np.asarray(refinement.barycentric_numerators, dtype=np.int64)
    parents = np.asarray(refinement.parent_face_indices, dtype=np.int64)
    shape_valid = bool(
        refinement.denominator == expected_denominator
        and numerators.shape == (len(refinement.faces), 3, 3)
        and parents.shape == (len(refinement.faces),)
        and np.asarray(refinement.faces).shape == (len(refinement.faces), 3)
        and np.all((parents >= 0) & (parents < len(faces)))
    )
    nonnegative = bool(shape_valid and np.all(numerators >= 0))
    row_sums = bool(shape_valid and np.all(np.sum(numerators, axis=2) == expected_denominator))
    counts = (
        np.bincount(parents, minlength=len(faces))
        if shape_valid
        else np.zeros(len(faces), dtype=np.int64)
    )
    counts_valid = bool(
        shape_valid
        and len(refinement.faces) == len(faces) * expected_children
        and np.all(counts == expected_children)
    )

    expected_partition = _canonical_partition(rounds)
    partition_valid = shape_valid and counts_valid
    if partition_valid:
        for parent_index in range(len(faces)):
            observed = tuple(
                sorted(
                    tuple(int(value) for value in triangle.flat)
                    for triangle in numerators[parents == parent_index]
                )
            )
            if observed != expected_partition:
                partition_valid = False
                break

    reconstruction_valid = bool(
        shape_valid and parent_on_grid and refined_on_grid and nonnegative and row_sums
    )
    if reconstruction_valid:
        child_faces = np.asarray(refinement.faces, dtype=np.int64)
        for child_index, parent_index in enumerate(parents):
            parent_coordinates = parent_integers[faces[int(parent_index)]]
            expected = numerators[child_index] @ parent_coordinates
            observed = refined_integers[child_faces[child_index]]
            if not np.array_equal(expected, observed):
                reconstruction_valid = False
                break

    checks = (
        ("denominator_or_shape", shape_valid),
        ("parent_vertex_retention", retained),
        ("parent_grid", parent_on_grid),
        ("refined_grid", refined_on_grid),
        ("negative_barycentric_numerator", nonnegative),
        ("barycentric_row_sum", row_sums),
        ("parent_child_count", counts_valid),
        ("standard_partition", partition_valid),
        ("integer_affine_reconstruction", reconstruction_valid),
    )
    blockers.extend(name for name, passed in checks if not passed)
    return ExactRefinementCertificate(
        status="pass" if not blockers else "fail",
        rounds=rounds,
        denominator=refinement.denominator,
        parent_face_count=len(faces),
        child_face_count=len(refinement.faces),
        children_per_parent=expected_children,
        parent_vertices_retained_bitwise=retained,
        parent_on_grid=parent_on_grid,
        refined_on_grid=refined_on_grid,
        nonnegative_barycentric_numerators=nonnegative,
        barycentric_row_sums_valid=row_sums,
        parent_child_counts_valid=counts_valid,
        standard_partition_valid=bool(partition_valid),
        exact_integer_affine_reconstruction=reconstruction_valid,
        blockers=tuple(blockers),
    )
