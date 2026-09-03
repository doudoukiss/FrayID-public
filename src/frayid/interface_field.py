from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class InterfaceField:
    """A fixed tetrahedral PL field with explicit source-interface provenance."""

    vertices: np.ndarray
    values: np.ndarray
    interface_vertices: np.ndarray
    tetrahedra: np.ndarray
    cell_regions: np.ndarray
    interface_faces: np.ndarray
    source_face_indices: np.ndarray
    outside_cell_count: int
    inside_cell_count: int
    source_face_count: int

    def validate(self) -> None:
        vertex_count = int(self.vertices.shape[0])
        tetrahedron_count = int(self.tetrahedra.shape[0])
        interface_face_count = int(self.interface_faces.shape[0])
        if self.vertices.shape != (vertex_count, 3) or not np.isfinite(self.vertices).all():
            raise ValueError("field vertices must be finite with shape [V, 3]")
        if self.values.shape != (vertex_count,) or not np.isfinite(self.values).all():
            raise ValueError("field values must be finite with shape [V]")
        if self.interface_vertices.shape != (vertex_count,):
            raise ValueError("interface vertex mask must have shape [V]")
        if self.interface_vertices.dtype != np.bool_:
            raise ValueError("interface vertex mask must be boolean")
        if self.tetrahedra.shape != (tetrahedron_count, 4):
            raise ValueError("tetrahedra must have shape [T, 4]")
        if self.cell_regions.shape != (tetrahedron_count,):
            raise ValueError("cell regions must have shape [T]")
        if not np.isin(self.cell_regions, (-1, 1)).all():
            raise ValueError("cell regions must be exactly -1 or +1")
        if self.interface_faces.shape != (interface_face_count, 3):
            raise ValueError("interface faces must have shape [F, 3]")
        if self.source_face_indices.shape != (interface_face_count,):
            raise ValueError("source face provenance must have shape [F]")
        for name, indices, width in (
            ("tetrahedra", self.tetrahedra, 4),
            ("interface_faces", self.interface_faces, 3),
        ):
            if indices.dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integer indices")
            if indices.shape[1] != width or np.any(indices < 0) or np.any(indices >= vertex_count):
                raise ValueError(f"{name} contains an out-of-range vertex")
            if np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0):
                raise ValueError(f"{name} contains a repeated vertex")
        if np.any(self.source_face_indices < 0) or np.any(
            self.source_face_indices >= self.source_face_count
        ):
            raise ValueError("interface face has invalid source-face provenance")
        if self.outside_cell_count != int(np.count_nonzero(self.cell_regions == 1)):
            raise ValueError("outside cell count does not match labels")
        if self.inside_cell_count != int(np.count_nonzero(self.cell_regions == -1)):
            raise ValueError("inside cell count does not match labels")
        if np.any(self.values[self.interface_vertices] != 0.0):
            raise ValueError("every interface vertex must have exactly zero value")
        if np.any(self.values[~self.interface_vertices] == 0.0):
            raise ValueError("a non-interface field vertex has zero value")


def write_interface_mesh(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> None:
    """Write the strict text interchange consumed by the pinned CGAL helper."""
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("source vertices must be finite with shape [V, 3]")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("source faces must have shape [F, 3]")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("outer bounds must be strictly ordered three-vectors")
    lines = [
        "FRAYID_E6_MESH 1",
        f"{points.shape[0]} {triangles.shape[0]}",
        " ".join(format(float(value), ".17g") for value in np.r_[lower, upper]),
    ]
    lines.extend(" ".join(format(float(value), ".17g") for value in row) for row in points)
    lines.extend(" ".join(str(int(value)) for value in row) for row in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_interface_field(path: Path) -> InterfaceField:
    """Read the bounded streaming output of the CGAL field builder."""
    with path.open("r", encoding="utf-8") as stream:
        magic = stream.readline().split()
        if magic != ["FRAYID_E6_FIELD", "1"]:
            raise ValueError("unsupported interface-field format")
        counts = [int(value) for value in stream.readline().split()]
        if len(counts) != 4:
            raise ValueError("interface-field count record is malformed")
        vertex_count, tetrahedron_count, interface_face_count, source_face_count = counts
        region_counts = [int(value) for value in stream.readline().split()]
        if len(region_counts) != 2:
            raise ValueError("interface-field region count record is malformed")
        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        values = np.empty(vertex_count, dtype=np.float64)
        interface_vertices = np.empty(vertex_count, dtype=np.bool_)
        for index in range(vertex_count):
            record = stream.readline().split()
            if len(record) != 5:
                raise ValueError("interface-field vertex record is malformed")
            vertices[index] = [float(value) for value in record[:3]]
            values[index] = float(record[3])
            interface_vertices[index] = bool(int(record[4]))
        tetrahedra = np.empty((tetrahedron_count, 4), dtype=np.int64)
        cell_regions = np.empty(tetrahedron_count, dtype=np.int8)
        for index in range(tetrahedron_count):
            record = stream.readline().split()
            if len(record) != 5:
                raise ValueError("interface-field tetrahedron record is malformed")
            tetrahedra[index] = [int(value) for value in record[:4]]
            cell_regions[index] = int(record[4])
        tetrahedron_vertices = vertices[tetrahedra]
        serialized_determinants = np.einsum(
            "ij,ij->i",
            np.cross(
                tetrahedron_vertices[:, 1] - tetrahedron_vertices[:, 0],
                tetrahedron_vertices[:, 2] - tetrahedron_vertices[:, 0],
            ),
            tetrahedron_vertices[:, 3] - tetrahedron_vertices[:, 0],
        )
        negative = serialized_determinants < 0.0
        tetrahedra[negative, :2] = tetrahedra[negative, 1::-1]
        interface_faces = np.empty((interface_face_count, 3), dtype=np.int64)
        source_face_indices = np.empty(interface_face_count, dtype=np.int64)
        for index in range(interface_face_count):
            record = stream.readline().split()
            if len(record) != 4:
                raise ValueError("interface-field face record is malformed")
            interface_faces[index] = [int(value) for value in record[:3]]
            source_face_indices[index] = int(record[3])
        if stream.read().strip():
            raise ValueError("interface-field file has trailing records")
    result = InterfaceField(
        vertices=vertices,
        values=values,
        interface_vertices=interface_vertices,
        tetrahedra=tetrahedra,
        cell_regions=cell_regions,
        interface_faces=interface_faces,
        source_face_indices=source_face_indices,
        outside_cell_count=region_counts[0],
        inside_cell_count=region_counts[1],
        source_face_count=source_face_count,
    )
    result.validate()
    return result


def tetrahedron_volumes(field: InterfaceField) -> np.ndarray:
    tetrahedra = field.vertices[field.tetrahedra]
    return np.asarray(
        np.einsum(
            "ij,ij->i",
            np.cross(tetrahedra[:, 1] - tetrahedra[:, 0], tetrahedra[:, 2] - tetrahedra[:, 0]),
            tetrahedra[:, 3] - tetrahedra[:, 0],
        )
        / 6.0,
        dtype=np.float64,
    )


def _unique_rows(values: np.ndarray) -> np.ndarray:
    if not values.size:
        return np.empty((0, values.shape[1]), dtype=values.dtype)
    return np.unique(np.ascontiguousarray(values), axis=0)


def _row_set(values: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(int(value) for value in row) for row in values}


def certify_zero_subcomplex(field: InterfaceField) -> dict[str, Any]:
    """Enumerate the complete zero subcomplex and reject every spurious simplex."""
    field.validate()
    zero = field.values == 0.0
    expected_faces = _unique_rows(np.sort(field.interface_faces, axis=1))
    expected_edges = _unique_rows(
        np.sort(
            field.interface_faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2),
            axis=1,
        )
    )
    tetrahedra = field.tetrahedra
    tetrahedron_faces = tetrahedra[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]
    all_zero_faces = _unique_rows(
        np.sort(tetrahedron_faces[zero[tetrahedron_faces].all(axis=-1)], axis=1)
    )
    tetrahedron_edges = tetrahedra[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]]
    all_zero_edges = _unique_rows(
        np.sort(tetrahedron_edges[zero[tetrahedron_edges].all(axis=-1)], axis=1)
    )
    expected_face_set = _row_set(expected_faces)
    expected_edge_set = _row_set(expected_edges)
    zero_face_set = _row_set(all_zero_faces)
    zero_edge_set = _row_set(all_zero_edges)
    interface_vertex_set = set(int(value) for value in np.unique(field.interface_faces))
    zero_vertex_set = set(int(value) for value in np.flatnonzero(zero))
    report = {
        "schema_version": "interface_zero_subcomplex.v1",
        "expected_interface_face_count": int(expected_faces.shape[0]),
        "enumerated_all_zero_face_count": int(all_zero_faces.shape[0]),
        "noninterface_all_zero_vertex_count": len(zero_vertex_set - interface_vertex_set),
        "noninterface_all_zero_edge_count": len(zero_edge_set - expected_edge_set),
        "noninterface_all_zero_face_count": len(zero_face_set - expected_face_set),
        "missing_interface_zero_face_count": len(expected_face_set - zero_face_set),
        "all_zero_tetrahedron_count": int(np.count_nonzero(zero[tetrahedra].all(axis=1))),
    }
    failure_counts = (
        "noninterface_all_zero_vertex_count",
        "noninterface_all_zero_edge_count",
        "noninterface_all_zero_face_count",
        "missing_interface_zero_face_count",
        "all_zero_tetrahedron_count",
    )
    blockers = [key for key in failure_counts if report[key]]
    report["blockers"] = blockers
    report["status"] = "pass" if not blockers else "fail"
    return report


def _canonical_surface_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    remapped = inverse[faces]
    for index, face in enumerate(remapped):
        offset = int(np.argmin(face))
        remapped[index] = np.roll(face, -offset)
    face_order = np.lexsort((remapped[:, 2], remapped[:, 1], remapped[:, 0]))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices[order], dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(remapped[face_order], dtype="<i8").tobytes())
    return digest.hexdigest()


def certify_interface_surface(
    field: InterfaceField,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Certify that enumerated zero facets exactly tile their immutable source faces."""
    field.validate()
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    triangles = source_vertices[source_faces[field.source_face_indices]]
    points = field.vertices[field.interface_faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normal = np.cross(edges_a, edges_b)
    candidate_normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    oriented = np.einsum("ij,ij->i", normal, candidate_normal) > 0
    plane_residuals: list[np.ndarray] = []
    minimum_barycentric = 1.0
    for corner in range(3):
        relative = points[:, corner] - triangles[:, 0]
        dot00 = np.einsum("ij,ij->i", edges_a, edges_a)
        dot01 = np.einsum("ij,ij->i", edges_a, edges_b)
        dot11 = np.einsum("ij,ij->i", edges_b, edges_b)
        dot20 = np.einsum("ij,ij->i", relative, edges_a)
        dot21 = np.einsum("ij,ij->i", relative, edges_b)
        denominator = dot00 * dot11 - dot01**2
        first = (dot11 * dot20 - dot01 * dot21) / denominator
        second = (dot00 * dot21 - dot01 * dot20) / denominator
        barycentric = np.stack((1.0 - first - second, first, second), axis=1)
        minimum_barycentric = min(minimum_barycentric, float(barycentric.min()))
        reconstructed = np.einsum("fbc,fb->fc", triangles, barycentric)
        plane_residuals.append(np.linalg.norm(reconstructed - points[:, corner], axis=1))
    source_double_area = np.linalg.norm(
        np.cross(
            source_vertices[source_faces[:, 1]] - source_vertices[source_faces[:, 0]],
            source_vertices[source_faces[:, 2]] - source_vertices[source_faces[:, 0]],
        ),
        axis=1,
    )
    extracted_double_area = np.linalg.norm(candidate_normal, axis=1)
    coverage = np.bincount(
        field.source_face_indices,
        weights=extracted_double_area,
        minlength=source_faces.shape[0],
    )
    relative_coverage_error = np.abs(coverage - source_double_area) / source_double_area
    mesh = trimesh.Trimesh(vertices=field.vertices, faces=field.interface_faces, process=False)
    components = mesh.split(only_watertight=False)
    lower, upper = (np.asarray(value, dtype=np.float64) for value in bounds)
    diagonal = float(np.linalg.norm(upper - lower))
    boundary_vertex = np.isclose(field.vertices, lower, atol=diagonal * 1e-12).any(
        axis=1
    ) | np.isclose(field.vertices, upper, atol=diagonal * 1e-12).any(axis=1)
    max_residual = float(np.max(np.concatenate(plane_residuals), initial=0.0))
    max_residual_fraction = max_residual / diagonal
    max_coverage_error = float(relative_coverage_error.max(initial=0.0))
    report = {
        "schema_version": "interface_surface_certificate.v1",
        "surface_vertex_count": int(np.unique(field.interface_faces).size),
        "surface_face_count": int(field.interface_faces.shape[0]),
        "component_count": len(components),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "positive_volume": bool(mesh.volume > 0),
        "maximum_source_face_residual": max_residual,
        "maximum_source_face_residual_fraction_of_domain_diagonal": max_residual_fraction,
        "minimum_source_barycentric_coordinate": minimum_barycentric,
        "maximum_source_face_area_coverage_relative_error": max_coverage_error,
        "source_faces_without_coverage": int(np.count_nonzero(coverage == 0.0)),
        "misoriented_interface_face_count": int(np.count_nonzero(~oriented)),
        "nonpositive_outer_boundary_node_count": int(
            np.count_nonzero(field.values[boundary_vertex] <= 0.0)
        ),
        "minimum_outer_boundary_value": float(field.values[boundary_vertex].min(initial=np.inf)),
        "canonical_extraction_sha256": _canonical_surface_hash(
            field.vertices, field.interface_faces
        ),
    }
    blockers: list[str] = []
    if report["component_count"] != 1:
        blockers.append("zero_set_component_count")
    if not report["watertight"] or not report["winding_consistent"]:
        blockers.append("zero_set_not_watertight_or_consistently_oriented")
    if report["euler_number"] != 2 or not report["positive_volume"]:
        blockers.append("zero_set_not_outward_euler2")
    if max_residual_fraction > 1e-10:
        blockers.append("surface_conformance_residual")
    if minimum_barycentric < -1e-10:
        blockers.append("steiner_vertex_outside_source_face")
    if max_coverage_error > 1e-10:
        blockers.append("source_face_coverage")
    if report["source_faces_without_coverage"]:
        blockers.append("source_face_missing")
    if report["misoriented_interface_face_count"]:
        blockers.append("interface_orientation")
    if report["nonpositive_outer_boundary_node_count"]:
        blockers.append("outer_boundary_not_strictly_positive")
    report["blockers"] = blockers
    report["status"] = "pass" if not blockers else "fail"
    return report


def sample_field_points(
    field: InterfaceField,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample deterministic volume-weighted points and PL values from finite cells."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    volumes = tetrahedron_volumes(field)
    if np.any(volumes <= 0):
        raise ValueError("field contains non-positive tetrahedron orientation")
    generator = np.random.default_rng(seed)
    chosen = generator.choice(volumes.size, size=count, p=volumes / volumes.sum())
    exponential = generator.exponential(size=(count, 4))
    barycentric = exponential / exponential.sum(axis=1, keepdims=True)
    tetrahedra = field.tetrahedra[chosen]
    points = np.einsum("bi,bij->bj", barycentric, field.vertices[tetrahedra])
    values = np.einsum("bi,bi->b", barycentric, field.values[tetrahedra])
    return points, values, field.cell_regions[chosen]


def certify_independent_signs(
    field: InterfaceField,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    """Compare flood-filled PL signs with an independent solid-angle winding oracle."""
    generator = np.random.default_rng(seed)
    inside_cells = np.flatnonzero(field.cell_regions < 0)
    outside_cells = np.flatnonzero(field.cell_regions > 0)
    if not inside_cells.size or not outside_cells.size:
        raise ValueError("sign certificate requires both field regions")
    mesh = trimesh.Trimesh(vertices=source_vertices, faces=source_faces, process=False)
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(source_vertices), axis=0)))
    boundary_exclusion = diagonal * 1e-10

    def sample_region(cells: np.ndarray, target_count: int) -> tuple[np.ndarray, np.ndarray]:
        retained_points: list[np.ndarray] = []
        retained_values: list[np.ndarray] = []
        retained_count = 0
        for _ in range(8):
            candidate_count = max(4096, (target_count - retained_count) * 3)
            chosen = generator.choice(cells, size=candidate_count, replace=True)
            exponential = generator.exponential(size=(candidate_count, 4))
            barycentric = exponential / exponential.sum(axis=1, keepdims=True)
            tetrahedra = field.tetrahedra[chosen]
            points = np.einsum("bi,bij->bj", barycentric, field.vertices[tetrahedra])
            values = np.einsum("bi,bi->b", barycentric, field.values[tetrahedra])
            _, distances, _ = trimesh.proximity.closest_point(mesh, points)  # type: ignore[no-untyped-call]
            keep = distances > boundary_exclusion
            retained_points.append(points[keep])
            retained_values.append(values[keep])
            retained_count += int(np.count_nonzero(keep))
            if retained_count >= target_count:
                break
        if retained_count < target_count:
            raise ValueError("insufficient off-interface samples for independent sign certificate")
        return (
            np.concatenate(retained_points, axis=0)[:target_count],
            np.concatenate(retained_values, axis=0)[:target_count],
        )

    inside_count = sample_count // 2
    inside_points, inside_values = sample_region(inside_cells, inside_count)
    outside_points, outside_values = sample_region(outside_cells, sample_count - inside_count)
    points = np.concatenate((inside_points, outside_points), axis=0)
    values = np.concatenate((inside_values, outside_values), axis=0)
    regions = np.concatenate(
        (
            np.full(inside_count, -1, dtype=np.int8),
            np.full(sample_count - inside_count, 1, dtype=np.int8),
        )
    )
    triangles = np.asarray(source_vertices, dtype=np.float64)[
        np.asarray(source_faces, dtype=np.int64)
    ]
    winding_numbers = np.empty(sample_count, dtype=np.float64)
    for start in range(0, sample_count, 128):
        batch = points[start : start + 128]
        first = triangles[None, :, 0] - batch[:, None, :]
        second = triangles[None, :, 1] - batch[:, None, :]
        third = triangles[None, :, 2] - batch[:, None, :]
        first_norm = np.linalg.norm(first, axis=-1)
        second_norm = np.linalg.norm(second, axis=-1)
        third_norm = np.linalg.norm(third, axis=-1)
        numerator = np.einsum("bfi,bfi->bf", first, np.cross(second, third))
        denominator = (
            first_norm * second_norm * third_norm
            + np.einsum("bfi,bfi->bf", first, second) * third_norm
            + np.einsum("bfi,bfi->bf", second, third) * first_norm
            + np.einsum("bfi,bfi->bf", third, first) * second_norm
        )
        winding_numbers[start : start + batch.shape[0]] = np.sum(
            2.0 * np.arctan2(numerator, denominator), axis=1
        ) / (4.0 * np.pi)
    independently_inside = np.abs(winding_numbers) > 0.5
    ray_inside = mesh.contains(points)
    field_inside = values < 0.0
    region_inside = regions < 0
    return {
        "schema_version": "interface_sign_certificate.v1",
        "sample_count": sample_count,
        "inside_sample_count": inside_count,
        "outside_sample_count": sample_count - inside_count,
        "boundary_exclusion_fraction_of_source_bbox_diagonal": 1e-10,
        "field_region_disagreement_count": int(np.count_nonzero(field_inside != region_inside)),
        "independent_sign_mismatch_count": int(
            np.count_nonzero(field_inside != independently_inside)
        ),
        "trimesh_ray_mismatch_count_diagnostic": int(np.count_nonzero(field_inside != ray_inside)),
        "minimum_absolute_inside_winding_number": float(
            np.abs(winding_numbers[field_inside]).min(initial=np.inf)
        ),
        "maximum_absolute_outside_winding_number": float(
            np.abs(winding_numbers[~field_inside]).max(initial=0.0)
        ),
        "status": (
            "pass"
            if np.array_equal(field_inside, region_inside)
            and np.array_equal(field_inside, independently_inside)
            else "fail"
        ),
    }
