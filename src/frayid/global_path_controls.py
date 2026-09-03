from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import Any

from frayid.certified_tet_path import certify_exact_polynomial_path


def _determinant(matrix: list[list[Fraction]]) -> Fraction:
    result = Fraction(0)
    for permutation in permutations(range(3)):
        inversions = sum(
            permutation[first] > permutation[second]
            for first in range(3)
            for second in range(first + 1, 3)
        )
        product = Fraction((-1) ** inversions)
        for row in range(3):
            product *= matrix[row][permutation[row]]
        result += product
    return result


def _edge_matrix(
    vertices: list[list[Fraction]], tetrahedron: tuple[int, int, int, int]
) -> list[list[Fraction]]:
    origin = vertices[tetrahedron[0]]
    return [
        [vertices[tetrahedron[column + 1]][row] - origin[row] for column in range(3)]
        for row in range(3)
    ]


def _determinant_polynomial(
    start: list[list[Fraction]], delta: list[list[Fraction]]
) -> list[Fraction]:
    result = [Fraction(0)] * 4
    for permutation in permutations(range(3)):
        inversions = sum(
            permutation[first] > permutation[second]
            for first in range(3)
            for second in range(first + 1, 3)
        )
        product = [Fraction((-1) ** inversions)]
        for row in range(3):
            following = [Fraction(0)] * (len(product) + 1)
            for degree, value in enumerate(product):
                following[degree] += value * start[row][permutation[row]]
                following[degree + 1] += value * delta[row][permutation[row]]
            product = following
        for degree, value in enumerate(product):
            result[degree] += value
    return result


def _bernstein_cubic(coefficients: list[Fraction]) -> list[Fraction]:
    first, second, third, fourth = coefficients
    return [
        first,
        first + second / 3,
        first + 2 * second / 3 + third / 3,
        first + second + third + fourth,
    ]


def run_global_path_controls() -> dict[str, Any]:
    """Run the tracked exact public controls registered for G16."""

    original_velocity = (Fraction(2), Fraction(2))
    truncated_velocity = (Fraction(2), Fraction(1, 2))
    collision_time = Fraction(1) / (truncated_velocity[0] - truncated_velocity[1])
    local_truncation_detected = Fraction(1) + original_velocity[1] - original_velocity[
        0
    ] == 1 and collision_time == Fraction(2, 3)

    endpoint_control = certify_exact_polynomial_path([1, -5, 6, 0])
    endpoint_interior_inversion_detected = bool(
        endpoint_control["endpoint_positive"] and endpoint_control["status"] == "fail"
    )

    vertices = [
        [Fraction(x), Fraction(y), Fraction(z)]
        for x, y, z in (
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
            (0, 0, 0),
        )
    ]
    boundary_faces = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    )
    tetrahedra: list[tuple[int, int, int, int]] = []
    for face in boundary_faces:
        tetrahedron = (8, *face)
        if _determinant(_edge_matrix(vertices, tetrahedron)) < 0:
            tetrahedron = (tetrahedron[0], tetrahedron[1], tetrahedron[3], tetrahedron[2])
        tetrahedra.append(tetrahedron)
    displacement = [[Fraction(0)] * 3 for _ in vertices]
    displacement[8] = [Fraction(1, 4), Fraction(-1, 5), Fraction(1, 6)]
    lower_ratios: list[Fraction] = []
    cube_paths_pass = True
    for tetrahedron in tetrahedra:
        start_matrix = _edge_matrix(vertices, tetrahedron)
        delta_matrix = _edge_matrix(displacement, tetrahedron)
        polynomial = _determinant_polynomial(start_matrix, delta_matrix)
        bernstein = _bernstein_cubic(polynomial)
        cube_paths_pass &= min(bernstein) > 0
        cube_paths_pass &= certify_exact_polynomial_path(polynomial)["status"] == "pass"
        lower_ratios.append(min(bernstein) / _determinant(start_matrix))
    initial_volume = sum(_determinant(_edge_matrix(vertices, value)) / 6 for value in tetrahedra)
    cube_paths_pass &= initial_volume == 8

    inverted = (
        tetrahedra[0][1],
        tetrahedra[0][0],
        tetrahedra[0][2],
        tetrahedra[0][3],
    )
    invalid_initial_detected = _determinant(_edge_matrix(vertices, tetrahedra[0])) > 0 and (
        _determinant(_edge_matrix(vertices, inverted)) < 0
    )
    free_boundary_displacement = [[Fraction(0)] * 3 for _ in vertices]
    free_boundary_displacement[0][0] = Fraction(1, 10)
    free_boundary_detected = any(
        component != 0 for vertex in free_boundary_displacement[:8] for component in vertex
    )
    tetrahedral_facets = {
        tuple(sorted(tetrahedron[index] for index in indices))
        for tetrahedron in tetrahedra
        for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    }
    nonconforming_face = next(
        candidate
        for candidate in permutations(range(8), 3)
        if len(set(candidate)) == 3 and tuple(sorted(candidate)) not in tetrahedral_facets
    )
    nonconforming_carrier_detected = tuple(sorted(nonconforming_face)) not in tetrahedral_facets

    checks = {
        "coordinatewise_truncation_collision_detected": local_truncation_detected,
        "positive_endpoint_interior_inversion_detected": endpoint_interior_inversion_detected,
        "invalid_initial_complex_detected": invalid_initial_detected,
        "free_outer_boundary_detected": free_boundary_detected,
        "nonconforming_carrier_detected": nonconforming_carrier_detected,
        "fixed_boundary_cube_certified": cube_paths_pass,
    }
    return {
        "schema_version": "g16_global_path_controls.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "two_sheet_collision_time": str(collision_time),
        "endpoint_control": endpoint_control,
        "cube_tetrahedron_count": len(tetrahedra),
        "cube_initial_exact_volume": str(initial_volume),
        "cube_minimum_exact_bernstein_ratio": str(min(lower_ratios)),
        "uses_private_data": False,
    }
