from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

import numpy as np

DYADIC_DENOMINATOR = 1 << 40
DEFAULT_KAPPA = Fraction(1, 2)
_PERMUTATIONS = tuple(itertools.permutations(range(3)))
_PERMUTATION_INDEX = {value: index for index, value in enumerate(_PERMUTATIONS)}


def _fraction(value: float | np.floating[Any]) -> Fraction:
    return Fraction.from_float(float(value))


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _determinant3(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _inverse3(matrix: tuple[tuple[Fraction, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    determinant = _determinant3(matrix)
    if determinant == 0:
        raise ValueError("tetrahedron edge matrix is singular")
    cofactors = tuple(
        tuple(
            ((-1) ** (row + column))
            * (
                matrix[(row + 1) % 3][(column + 1) % 3] * matrix[(row + 2) % 3][(column + 2) % 3]
                - matrix[(row + 1) % 3][(column + 2) % 3] * matrix[(row + 2) % 3][(column + 1) % 3]
            )
            for column in range(3)
        )
        for row in range(3)
    )
    return tuple(
        tuple(cofactors[column][row] / determinant for column in range(3)) for row in range(3)
    )


def exact_gradient_frobenius_squared(
    tetrahedron_vertices: np.ndarray, tetrahedron_displacements: np.ndarray
) -> Fraction:
    """Return the exact squared Frobenius norm for a binary64 affine tetra field."""
    vertices = np.asarray(tetrahedron_vertices, dtype=np.float64)
    displacements = np.asarray(tetrahedron_displacements, dtype=np.float64)
    if vertices.shape != (4, 3) or displacements.shape != (4, 3):
        raise ValueError("tetrahedron vertices and displacements must have shape [4, 3]")
    edges = tuple(
        tuple(
            _fraction(vertices[column + 1, row]) - _fraction(vertices[0, row])
            for column in range(3)
        )
        for row in range(3)
    )
    inverse = _inverse3(edges)
    delta = tuple(
        tuple(
            _fraction(displacements[column + 1, row]) - _fraction(displacements[0, row])
            for column in range(3)
        )
        for row in range(3)
    )
    gradient = tuple(
        tuple(
            sum(delta[row][inner] * inverse[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    squared = Fraction(0)
    for row in gradient:
        for value in row:
            squared += value * value
    return squared


@dataclass(frozen=True)
class FreudenthalLatticeV1:
    lower: np.ndarray
    upper: np.ndarray
    nodes_per_axis: int
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]
    vertices: np.ndarray
    tetrahedra: np.ndarray
    boundary_mask: np.ndarray

    @classmethod
    def create(
        cls, lower: np.ndarray, upper: np.ndarray, *, nodes_per_axis: int = 8
    ) -> FreudenthalLatticeV1:
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,) or np.any(low >= high):
            raise ValueError("lattice bounds must be strictly ordered three-vectors")
        if nodes_per_axis < 2:
            raise ValueError("lattice requires at least two nodes per axis")
        generated_axes = tuple(
            np.asarray(
                [
                    float(low[axis] + (high[axis] - low[axis]) * index / (nodes_per_axis - 1))
                    for index in range(nodes_per_axis)
                ],
                dtype=np.float64,
            )
            for axis in range(3)
        )
        axes = cast(tuple[np.ndarray, np.ndarray, np.ndarray], generated_axes)
        vertices = np.asarray(
            [
                [axes[0][i], axes[1][j], axes[2][k]]
                for i in range(nodes_per_axis)
                for j in range(nodes_per_axis)
                for k in range(nodes_per_axis)
            ],
            dtype=np.float64,
        )

        def node(i: int, j: int, k: int) -> int:
            return (i * nodes_per_axis + j) * nodes_per_axis + k

        tetrahedra: list[list[int]] = []
        for i in range(nodes_per_axis - 1):
            for j in range(nodes_per_axis - 1):
                for k in range(nodes_per_axis - 1):
                    base = np.asarray([i, j, k], dtype=np.int64)
                    for permutation in _PERMUTATIONS:
                        indices = [base.copy()]
                        current = base.copy()
                        for axis in permutation:
                            current = current.copy()
                            current[axis] += 1
                            indices.append(current)
                        tet = [node(*(int(value) for value in index)) for index in indices]
                        points = vertices[tet]
                        determinant = float(
                            np.dot(
                                np.cross(points[1] - points[0], points[2] - points[0]),
                                points[3] - points[0],
                            )
                        )
                        if determinant < 0.0:
                            tet[1], tet[2] = tet[2], tet[1]
                        tetrahedra.append(tet)
        boundary = np.zeros((nodes_per_axis, nodes_per_axis, nodes_per_axis), dtype=np.bool_)
        boundary[[0, -1], :, :] = True
        boundary[:, [0, -1], :] = True
        boundary[:, :, [0, -1]] = True
        result = cls(
            lower=low,
            upper=high,
            nodes_per_axis=nodes_per_axis,
            coordinates=axes,
            vertices=vertices,
            tetrahedra=np.asarray(tetrahedra, dtype=np.int64),
            boundary_mask=boundary.reshape(-1),
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected_nodes = self.nodes_per_axis**3
        expected_tetrahedra = 6 * (self.nodes_per_axis - 1) ** 3
        if self.vertices.shape != (expected_nodes, 3) or not np.isfinite(self.vertices).all():
            raise ValueError("lattice vertices are malformed")
        if self.tetrahedra.shape != (expected_tetrahedra, 4):
            raise ValueError("Freudenthal tetrahedra are incomplete")
        if self.boundary_mask.shape != (expected_nodes,) or self.boundary_mask.dtype != np.bool_:
            raise ValueError("boundary mask is malformed")
        if np.count_nonzero(~self.boundary_mask) != max(self.nodes_per_axis - 2, 0) ** 3:
            raise ValueError("interior lattice node count is inconsistent")
        points = self.vertices[self.tetrahedra]
        determinants = np.einsum(
            "ij,ij->i",
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            points[:, 3] - points[:, 0],
        )
        if np.any(determinants <= 0.0):
            raise ValueError("lattice contains a nonpositive serialized tetrahedron")

    def locate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("points must be finite with shape [N, 3]")
        if np.any(values < self.lower) or np.any(values > self.upper):
            raise ValueError("point lies outside the lattice box")
        cell_count = self.nodes_per_axis - 1
        cells = np.empty((values.shape[0], 3), dtype=np.int64)
        fractions = np.empty_like(values)
        for axis in range(3):
            coordinate = self.coordinates[axis]
            cells[:, axis] = np.clip(
                np.searchsorted(coordinate, values[:, axis], side="right") - 1,
                0,
                cell_count - 1,
            )
            left = coordinate[cells[:, axis]]
            right = coordinate[cells[:, axis] + 1]
            fractions[:, axis] = (values[:, axis] - left) / (right - left)
        nodes = np.empty((values.shape[0], 4), dtype=np.int64)
        weights = np.empty((values.shape[0], 4), dtype=np.float64)
        tetrahedron_indices = np.empty(values.shape[0], dtype=np.int64)
        for row, (cell, local) in enumerate(zip(cells, fractions, strict=True)):
            permutation = tuple(int(value) for value in np.argsort(-local, kind="stable"))
            ordered = local[list(permutation)]
            weights[row] = [
                1.0 - ordered[0],
                ordered[0] - ordered[1],
                ordered[1] - ordered[2],
                ordered[2],
            ]
            current = cell.copy()
            coordinates = [current.copy()]
            for axis in permutation:
                current = current.copy()
                current[axis] += 1
                coordinates.append(current)
            nodes[row] = [
                (int(index[0]) * self.nodes_per_axis + int(index[1])) * self.nodes_per_axis
                + int(index[2])
                for index in coordinates
            ]
            cube_index = (int(cell[0]) * cell_count + int(cell[1])) * cell_count + int(cell[2])
            tetrahedron_indices[row] = 6 * cube_index + _PERMUTATION_INDEX[permutation]
        return nodes, weights, tetrahedron_indices

    def evaluate(self, controls: np.ndarray, points: np.ndarray) -> np.ndarray:
        values = np.asarray(controls, dtype=np.float64)
        if values.shape != self.vertices.shape or not np.isfinite(values).all():
            raise ValueError("controls must match lattice vertices")
        nodes, weights, _ = self.locate(points)
        return np.asarray(np.einsum("ni,nij->nj", weights, values[nodes]), dtype=np.float64)

    def content_sha256(self) -> str:
        return _sha256_arrays(
            self.lower.astype("<f8"),
            self.upper.astype("<f8"),
            self.vertices.astype("<f8"),
            self.tetrahedra.astype("<i8"),
            self.boundary_mask.astype(np.uint8),
        )


def exact_max_gradient_frobenius_squared(
    lattice: FreudenthalLatticeV1,
    controls: np.ndarray,
    *,
    deadline: float | None = None,
) -> tuple[Fraction, int]:
    values = np.asarray(controls, dtype=np.float64)
    if values.shape != lattice.vertices.shape or not np.isfinite(values).all():
        raise ValueError("controls must match lattice vertices")
    maximum = Fraction(0)
    maximum_index = -1
    for index, tetrahedron in enumerate(lattice.tetrahedra):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("exact every-cell Lipschitz certificate timed out")
        squared = exact_gradient_frobenius_squared(
            lattice.vertices[tetrahedron], values[tetrahedron]
        )
        if squared > maximum:
            maximum = squared
            maximum_index = index
    return maximum, maximum_index


@dataclass(frozen=True)
class CertifiedBilipschitzStepV1:
    raw_controls: np.ndarray
    accepted_controls: np.ndarray
    dyadic_scale_numerator: int
    accepted_scale: float
    exact_max_gradient_squared: str
    maximum_gradient_tetrahedron: int
    retained_displacement_ratio: float
    accepted_displacements: np.ndarray
    status: str
    blockers: tuple[str, ...]
    decision_sha256: str

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "certified_bilipschitz_step.v1",
            "status": self.status,
            "dyadic_scale_numerator": self.dyadic_scale_numerator,
            "dyadic_scale_denominator": DYADIC_DENOMINATOR,
            "accepted_scale": self.accepted_scale,
            "exact_max_gradient_frobenius_squared": self.exact_max_gradient_squared,
            "maximum_gradient_tetrahedron": self.maximum_gradient_tetrahedron,
            "certified_kappa": float(DEFAULT_KAPPA),
            "complete_path_minimum_distance_factor": 1.0 - float(DEFAULT_KAPPA),
            "retained_displacement_ratio": self.retained_displacement_ratio,
            "decision_sha256": self.decision_sha256,
            "blockers": list(self.blockers),
        }


def fit_and_certify_bilipschitz_step(
    lattice: FreudenthalLatticeV1,
    carrier_vertices: np.ndarray,
    proposed_displacements: np.ndarray,
    *,
    minimum_retained_displacement_ratio: float = 0.25,
    tikhonov: float = 1.0e-10,
    rcond: float = 1.0e-12,
    timeout_seconds: float | None = None,
) -> CertifiedBilipschitzStepV1:
    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    vertices = np.asarray(carrier_vertices, dtype=np.float64)
    proposal = np.asarray(proposed_displacements, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or proposal.shape != vertices.shape:
        raise ValueError("carrier vertices and proposal must share finite shape [V, 3]")
    if not np.isfinite(vertices).all() or not np.isfinite(proposal).all():
        raise ValueError("carrier vertices and proposal must be finite")
    if tikhonov < 0.0 or rcond < 0.0:
        raise ValueError("least-squares controls must be nonnegative")
    nodes, weights, _ = lattice.locate(vertices)
    free = np.flatnonzero(~lattice.boundary_mask)
    free_lookup = np.full(lattice.vertices.shape[0], -1, dtype=np.int64)
    free_lookup[free] = np.arange(free.size, dtype=np.int64)
    design = np.zeros((vertices.shape[0], free.size), dtype=np.float64)
    for column in range(4):
        free_columns = free_lookup[nodes[:, column]]
        rows = np.flatnonzero(free_columns >= 0)
        design[rows, free_columns[rows]] += weights[rows, column]
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("coarse lift construction timed out")
    augmented_design = np.vstack((design, math.sqrt(tikhonov) * np.eye(free.size)))
    augmented_target = np.vstack((proposal, np.zeros((free.size, 3), dtype=np.float64)))
    fitted, _, _, _ = np.linalg.lstsq(augmented_design, augmented_target, rcond=rcond)
    raw_controls = np.zeros_like(lattice.vertices)
    raw_controls[free] = fitted
    if np.any(raw_controls[lattice.boundary_mask] != 0.0):
        raise AssertionError("boundary controls changed during fitting")
    raw_squared, _ = exact_max_gradient_frobenius_squared(lattice, raw_controls, deadline=deadline)
    target_squared = DEFAULT_KAPPA * DEFAULT_KAPPA
    if raw_squared <= target_squared:
        numerator = DYADIC_DENOMINATOR
    else:
        ratio = target_squared / raw_squared
        numerator = math.isqrt(
            (ratio.numerator * DYADIC_DENOMINATOR * DYADIC_DENOMINATOR) // ratio.denominator
        )
    maximum_index = -1
    accepted_squared = Fraction(0)
    accepted_controls = np.zeros_like(raw_controls)
    for _ in range(1024):
        scale = numerator / DYADIC_DENOMINATOR
        accepted_controls = np.asarray(raw_controls * scale, dtype=np.float64)
        accepted_squared, maximum_index = exact_max_gradient_frobenius_squared(
            lattice, accepted_controls, deadline=deadline
        )
        if accepted_squared <= target_squared:
            break
        if numerator == 0:
            break
        numerator -= 1
    else:
        raise RuntimeError("serialized dyadic-scale recertification did not converge")
    accepted = lattice.evaluate(accepted_controls, vertices)
    proposed_norm = float(np.linalg.norm(proposal))
    accepted_norm = float(np.linalg.norm(accepted))
    retention = accepted_norm / proposed_norm if proposed_norm > 0.0 else 0.0
    blockers: list[str] = []
    if accepted_squared > target_squared:
        blockers.append("lipschitz_bound")
    if retention < minimum_retained_displacement_ratio:
        blockers.append("motion_retention")
    if not np.any(accepted != 0.0):
        blockers.append("zero_motion")
    scale = numerator / DYADIC_DENOMINATOR
    decision = _sha256_arrays(
        lattice.vertices.astype("<f8"),
        lattice.tetrahedra.astype("<i8"),
        raw_controls.astype("<f8"),
        accepted_controls.astype("<f8"),
        np.asarray([numerator], dtype="<i8"),
        accepted.astype("<f8"),
    )
    return CertifiedBilipschitzStepV1(
        raw_controls=raw_controls,
        accepted_controls=accepted_controls,
        dyadic_scale_numerator=numerator,
        accepted_scale=scale,
        exact_max_gradient_squared=f"{accepted_squared.numerator}/{accepted_squared.denominator}",
        maximum_gradient_tetrahedron=maximum_index,
        retained_displacement_ratio=retention,
        accepted_displacements=accepted,
        status="pass" if not blockers else "fail",
        blockers=tuple(blockers),
        decision_sha256=decision,
    )


_Barycentric = tuple[Fraction, Fraction, Fraction]
_Polygon = list[_Barycentric]


def _barycentric_signed(barycentric: _Barycentric, source_signs: tuple[Fraction, ...]) -> Fraction:
    return sum(
        (weight * sign for weight, sign in zip(barycentric, source_signs, strict=True)), Fraction(0)
    )


def _clean_polygon(polygon: _Polygon) -> _Polygon:
    cleaned: _Polygon = []
    for value in polygon:
        if not cleaned or value != cleaned[-1]:
            cleaned.append(value)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def _clip_polygon(
    polygon: _Polygon, source_signs: tuple[Fraction, ...], *, keep_positive: bool
) -> _Polygon:
    result: _Polygon = []
    for current, following in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        current_sign = _barycentric_signed(current, source_signs)
        following_sign = _barycentric_signed(following, source_signs)
        current_inside = current_sign >= 0 if keep_positive else current_sign <= 0
        following_inside = following_sign >= 0 if keep_positive else following_sign <= 0
        if current_inside:
            result.append(current)
        if current_inside != following_inside:
            fraction = current_sign / (current_sign - following_sign)
            result.append(
                tuple(
                    (1 - fraction) * left + fraction * right
                    for left, right in zip(current, following, strict=True)
                )  # type: ignore[arg-type]
            )
    return _clean_polygon(result)


def _split_polygons(polygons: list[_Polygon], source_signs: tuple[Fraction, ...]) -> list[_Polygon]:
    if not any(value > 0 for value in source_signs) or not any(value < 0 for value in source_signs):
        return polygons
    result: list[_Polygon] = []
    for polygon in polygons:
        signs = [_barycentric_signed(value, source_signs) for value in polygon]
        if any(value > 0 for value in signs) and any(value < 0 for value in signs):
            positive = _clip_polygon(polygon, source_signs, keep_positive=True)
            negative = _clip_polygon(polygon, source_signs, keep_positive=False)
            if len(positive) >= 3:
                result.append(positive)
            if len(negative) >= 3:
                result.append(negative)
        else:
            result.append(polygon)
    return result


def _plane_signs(
    triangle: tuple[tuple[Fraction, Fraction, Fraction], ...],
    coefficients: tuple[Fraction, Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction]:
    a, b, c, d = coefficients
    return tuple(a * point[0] + b * point[1] + c * point[2] + d for point in triangle)  # type: ignore[return-value]


def _fraction_point(
    triangle: tuple[tuple[Fraction, Fraction, Fraction], ...], barycentric: _Barycentric
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(
        sum(
            (barycentric[corner] * triangle[corner][axis] for corner in range(3)),
            Fraction(0),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class ConformingSurfaceV1:
    reference_vertices: np.ndarray
    faces: np.ndarray
    parent_face_indices: np.ndarray
    corner_barycentric_text: np.ndarray
    provenance_sha256: str
    surface_sha256: str

    def mapped_vertices(self, lattice: FreudenthalLatticeV1, controls: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.reference_vertices + lattice.evaluate(controls, self.reference_vertices),
            dtype=np.float64,
        )

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "conforming_surface.v1",
            "vertex_count": int(self.reference_vertices.shape[0]),
            "face_count": int(self.faces.shape[0]),
            "parent_face_count": int(np.unique(self.parent_face_indices).size),
            "provenance_sha256": self.provenance_sha256,
            "surface_sha256": self.surface_sha256,
        }


def refine_surface_to_lattice(
    lattice: FreudenthalLatticeV1,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    timeout_seconds: float | None = None,
) -> ConformingSurfaceV1:
    """Exactly partition each triangle by every crossed Freudenthal boundary."""
    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("surface must contain vertices [V,3] and faces [F,3]")
    if np.any(triangles < 0) or np.any(triangles >= points.shape[0]):
        raise ValueError("surface face index is out of range")
    exact_points = [
        cast(tuple[Fraction, Fraction, Fraction], tuple(_fraction(value) for value in point))
        for point in points
    ]
    coordinate_planes: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
    for axis in range(3):
        for coordinate in lattice.coordinates[axis][1:-1]:
            coefficients = [Fraction(0), Fraction(0), Fraction(0), -_fraction(coordinate)]
            coefficients[axis] = Fraction(1)
            coordinate_planes.append(tuple(coefficients))  # type: ignore[arg-type]

    vertex_lookup: dict[tuple[Fraction, Fraction, Fraction], int] = {}
    refined_vertices: list[tuple[Fraction, Fraction, Fraction]] = []
    refined_faces: list[tuple[int, int, int]] = []
    parent_indices: list[int] = []
    barycentric_records: list[tuple[str, str, str]] = []
    initial_polygon: _Polygon = [
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    ]
    for parent_index, face in enumerate(triangles):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("exact conforming surface refinement timed out")
        triangle = cast(
            tuple[
                tuple[Fraction, Fraction, Fraction],
                tuple[Fraction, Fraction, Fraction],
                tuple[Fraction, Fraction, Fraction],
            ],
            tuple(exact_points[int(index)] for index in face),
        )
        polygons = [initial_polygon]
        for plane in coordinate_planes:
            polygons = _split_polygons(polygons, _plane_signs(triangle, plane))
        cube_polygons: list[_Polygon] = []
        for polygon in polygons:
            centroid = tuple(
                sum((_fraction_point(triangle, value)[axis] for value in polygon), Fraction(0))
                / len(polygon)
                for axis in range(3)
            )
            cell: list[int] = []
            for axis in range(3):
                coordinate = float(centroid[axis])
                cell.append(
                    int(
                        np.clip(
                            np.searchsorted(lattice.coordinates[axis], coordinate, side="right")
                            - 1,
                            0,
                            lattice.nodes_per_axis - 2,
                        )
                    )
                )
            internal_planes: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
            for first, second in ((0, 1), (0, 2), (1, 2)):
                first_low = _fraction(lattice.coordinates[first][cell[first]])
                first_width = _fraction(lattice.coordinates[first][cell[first] + 1]) - first_low
                second_low = _fraction(lattice.coordinates[second][cell[second]])
                second_width = _fraction(lattice.coordinates[second][cell[second] + 1]) - second_low
                coefficients = [Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
                coefficients[first] = Fraction(1) / first_width
                coefficients[second] = -Fraction(1) / second_width
                coefficients[3] = -first_low / first_width + second_low / second_width
                internal_planes.append(tuple(coefficients))  # type: ignore[arg-type]
            local_polygons = [polygon]
            for plane in internal_planes:
                local_polygons = _split_polygons(local_polygons, _plane_signs(triangle, plane))
            cube_polygons.extend(local_polygons)
        for polygon in cube_polygons:
            for corner in range(1, len(polygon) - 1):
                barycentrics = (polygon[0], polygon[corner], polygon[corner + 1])
                determinant = (barycentrics[1][0] - barycentrics[0][0]) * (
                    barycentrics[2][1] - barycentrics[0][1]
                ) - (barycentrics[1][1] - barycentrics[0][1]) * (
                    barycentrics[2][0] - barycentrics[0][0]
                )
                if determinant == 0:
                    continue
                output_face: list[int] = []
                output_barycentric_records: list[tuple[str, str, str]] = []
                for barycentric in barycentrics:
                    exact_point = _fraction_point(triangle, barycentric)
                    output_index = vertex_lookup.get(exact_point)
                    if output_index is None:
                        output_index = len(refined_vertices)
                        vertex_lookup[exact_point] = output_index
                        refined_vertices.append(exact_point)
                    output_face.append(output_index)
                    output_barycentric_records.append(
                        tuple(_fraction_text(value) for value in barycentric)  # type: ignore[arg-type]
                    )
                refined_faces.append(tuple(output_face))  # type: ignore[arg-type]
                parent_indices.append(parent_index)
                barycentric_records.extend(output_barycentric_records)
    output_vertices = np.asarray(
        [[float(value) for value in point] for point in refined_vertices], dtype=np.float64
    )
    output_faces = np.asarray(refined_faces, dtype=np.int64)
    output_parents = np.asarray(parent_indices, dtype=np.int64)
    maximum_text = max(
        (len(value) for record in barycentric_records for value in record), default=1
    )
    barycentric_array = np.asarray(barycentric_records, dtype=f"<U{maximum_text}").reshape(-1, 3, 3)
    provenance_digest = hashlib.sha256()
    provenance_digest.update(output_parents.astype("<i8").tobytes())
    provenance_digest.update(
        "\n".join(value for record in barycentric_records for value in record).encode()
    )
    surface_hash = _sha256_arrays(output_vertices.astype("<f8"), output_faces.astype("<i8"))
    return ConformingSurfaceV1(
        reference_vertices=output_vertices,
        faces=output_faces,
        parent_face_indices=output_parents,
        corner_barycentric_text=barycentric_array,
        provenance_sha256=provenance_digest.hexdigest(),
        surface_sha256=surface_hash,
    )


def parent_area_path_report(
    original_vertices: np.ndarray,
    original_faces: np.ndarray,
    refined: ConformingSurfaceV1,
    mapped_vertices: np.ndarray,
) -> dict[str, Any]:
    original = np.asarray(original_vertices, dtype=np.float64)
    faces = np.asarray(original_faces, dtype=np.int64)
    mapped = np.asarray(mapped_vertices, dtype=np.float64)
    reference_cross = np.cross(
        original[faces[:, 1]] - original[faces[:, 0]],
        original[faces[:, 2]] - original[faces[:, 0]],
    )
    mapped_faces = refined.faces
    mapped_cross = np.cross(
        mapped[mapped_faces[:, 1]] - mapped[mapped_faces[:, 0]],
        mapped[mapped_faces[:, 2]] - mapped[mapped_faces[:, 0]],
    )
    signed_sum = np.zeros_like(reference_cross)
    unsigned_sum = np.zeros(faces.shape[0], dtype=np.float64)
    np.add.at(signed_sum, refined.parent_face_indices, mapped_cross)
    np.add.at(unsigned_sum, refined.parent_face_indices, np.linalg.norm(mapped_cross, axis=1))
    reference_squared = np.einsum("ij,ij->i", reference_cross, reference_cross)
    reference_norm = np.sqrt(reference_squared)
    signed_ratio = np.einsum("ij,ij->i", signed_sum, reference_cross) / reference_squared
    unsigned_ratio = unsigned_sum / reference_norm
    blockers: list[str] = []
    if not np.isfinite(signed_ratio).all() or float(np.min(signed_ratio)) < 0.01:
        blockers.append("signed_parent_area_floor")
    if not np.isfinite(unsigned_ratio).all() or float(np.min(unsigned_ratio)) < 0.10:
        blockers.append("unsigned_parent_area_floor")
    return {
        "schema_version": "bilipschitz_parent_area_path.v1",
        "minimum_signed_parent_area_ratio": float(np.min(signed_ratio)),
        "minimum_unsigned_parent_area_ratio": float(np.min(unsigned_ratio)),
        "signed_floor": 0.01,
        "unsigned_floor": 0.10,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


def run_bilipschitz_controls() -> dict[str, Any]:
    lattice = FreudenthalLatticeV1.create(
        np.asarray([-1.0, -1.0, -1.0]), np.asarray([1.0, 1.0, 1.0]), nodes_per_axis=4
    )
    positive = np.zeros_like(lattice.vertices)
    interior = np.flatnonzero(~lattice.boundary_mask)
    positive[interior, 0] = 0.02
    positive_squared, _ = exact_max_gradient_frobenius_squared(lattice, positive)
    unit_tetrahedron = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    shear = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    fold = np.asarray([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    shear_squared = exact_gradient_frobenius_squared(unit_tetrahedron, shear)
    fold_squared = exact_gradient_frobenius_squared(unit_tetrahedron, fold)
    triangle_vertices = np.asarray(
        [[-0.8, -0.2, 0.0], [0.8, -0.2, 0.0], [0.0, 0.8, 0.0]], dtype=np.float64
    )
    triangle_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    first = refine_surface_to_lattice(lattice, triangle_vertices, triangle_faces)
    second = refine_surface_to_lattice(lattice, triangle_vertices, triangle_faces)
    controls: dict[str, dict[str, Any]] = {
        "fixed_zero_boundary_positive_interior_field": {
            "status": "pass"
            if positive_squared <= DEFAULT_KAPPA * DEFAULT_KAPPA and positive_squared > 0
            else "fail",
            "exact_gradient_squared": str(positive_squared),
        },
        "excessive_large_shear_rejected": {
            "status": "pass" if shear_squared > DEFAULT_KAPPA * DEFAULT_KAPPA else "fail",
            "exact_gradient_squared": str(shear_squared),
        },
        "between_sample_fold_rejected": {
            "status": "pass" if fold_squared > DEFAULT_KAPPA * DEFAULT_KAPPA else "fail",
            "exact_gradient_squared": str(fold_squared),
        },
        "exact_conforming_refinement_replay": {
            "status": "pass"
            if first.surface_sha256 == second.surface_sha256
            and first.provenance_sha256 == second.provenance_sha256
            and first.faces.shape[0] > triangle_faces.shape[0]
            else "fail",
            "surface_sha256": first.surface_sha256,
            "provenance_sha256": first.provenance_sha256,
            "refined_face_count": int(first.faces.shape[0]),
        },
    }
    return {
        "schema_version": "post_v1_e17_public_controls.v1",
        "status": "pass"
        if all(value["status"] == "pass" for value in controls.values())
        else "fail",
        "controls": controls,
    }
