from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]
from scipy.sparse import linalg as sparse_linalg  # type: ignore[import-untyped]

from frayid.interface_field import InterfaceField

AMBIENT_SCAFFOLD_SCHEMA_V1 = "ambient_scaffold.v1"
HARMONIC_DIRECTION_SCHEMA_V1 = "ambient_harmonic_direction.v1"


@dataclass(frozen=True)
class ConstrainedAmbientComplex:
    vertices: np.ndarray
    tetrahedra: np.ndarray
    region_labels: np.ndarray
    interface_faces: np.ndarray
    source_face_indices: np.ndarray
    source_face_count: int
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray

    def validate(self) -> None:
        vertex_count = int(self.vertices.shape[0])
        tetrahedron_count = int(self.tetrahedra.shape[0])
        interface_face_count = int(self.interface_faces.shape[0])
        if self.vertices.dtype != np.float64 or self.vertices.shape != (vertex_count, 3):
            raise ValueError("constrained complex vertices must be float64 [V, 3]")
        if not np.isfinite(self.vertices).all():
            raise ValueError("constrained complex vertices must be finite")
        if self.tetrahedra.dtype != np.int64 or self.tetrahedra.shape != (
            tetrahedron_count,
            4,
        ):
            raise ValueError("constrained complex tetrahedra must be int64 [T, 4]")
        if self.region_labels.shape != (tetrahedron_count,):
            raise ValueError("constrained complex region labels must have shape [T]")
        if self.interface_faces.dtype != np.int64 or self.interface_faces.shape != (
            interface_face_count,
            3,
        ):
            raise ValueError("constrained complex interface faces must be int64 [F, 3]")
        if self.source_face_indices.shape != (interface_face_count,):
            raise ValueError("constrained complex source provenance must have shape [F]")
        if np.any(self.source_face_indices < 0) or np.any(
            self.source_face_indices >= self.source_face_count
        ):
            raise ValueError("constrained complex has invalid source provenance")
        if np.unique(self.source_face_indices).size != self.source_face_count:
            raise ValueError("constrained complex does not cover every source face")
        if (
            self.bounds_lower.shape != (3,)
            or self.bounds_upper.shape != (3,)
            or np.any(self.bounds_lower >= self.bounds_upper)
        ):
            raise ValueError("constrained complex bounds are invalid")


def read_constrained_ambient_complex(path: Path) -> ConstrainedAmbientComplex:
    """Read the bounded, streaming E16 CGAL interchange format."""

    with path.open("r", encoding="utf-8") as stream:
        if stream.readline().split() != ["FRAYID_E16_SCAFFOLD", "1"]:
            raise ValueError("unsupported constrained ambient complex format")
        counts = [int(value) for value in stream.readline().split()]
        if len(counts) != 4:
            raise ValueError("constrained ambient complex count record is malformed")
        vertex_count, tetrahedron_count, interface_face_count, source_face_count = counts
        bounds = np.asarray([float(value) for value in stream.readline().split()], dtype=np.float64)
        if bounds.shape != (6,):
            raise ValueError("constrained ambient complex bounds record is malformed")
        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        for index in range(vertex_count):
            record = stream.readline().split()
            if len(record) != 3:
                raise ValueError("constrained ambient complex vertex record is malformed")
            vertices[index] = [float(value) for value in record]
        tetrahedra = np.empty((tetrahedron_count, 4), dtype=np.int64)
        regions = np.empty(tetrahedron_count, dtype=np.int16)
        for index in range(tetrahedron_count):
            record = stream.readline().split()
            if len(record) != 5:
                raise ValueError("constrained ambient complex cell record is malformed")
            tetrahedra[index] = [int(value) for value in record[:4]]
            regions[index] = int(record[4])
        interface_faces = np.empty((interface_face_count, 3), dtype=np.int64)
        source_face_indices = np.empty(interface_face_count, dtype=np.int64)
        for index in range(interface_face_count):
            record = stream.readline().split()
            if len(record) != 4:
                raise ValueError("constrained ambient complex face record is malformed")
            interface_faces[index] = [int(value) for value in record[:3]]
            source_face_indices[index] = int(record[3])
        if stream.read().strip():
            raise ValueError("constrained ambient complex has trailing records")
    result = ConstrainedAmbientComplex(
        vertices=vertices,
        tetrahedra=tetrahedra,
        region_labels=regions,
        interface_faces=interface_faces,
        source_face_indices=source_face_indices,
        source_face_count=source_face_count,
        bounds_lower=bounds[:3],
        bounds_upper=bounds[3:],
    )
    result.validate()
    return result


def _array_hash(array: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def _surface_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def _tetrahedron_determinants(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = vertices[tetrahedra]
    return np.asarray(
        np.einsum(
            "ij,ij->i",
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            points[:, 3] - points[:, 0],
        ),
        dtype=np.float64,
    )


def _rows_are_members(sorted_unique_rows: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    """Vectorized exact row membership without a memory-heavy Python tuple set."""

    row_dtype = np.dtype((np.void, sorted_unique_rows.dtype.itemsize * sorted_unique_rows.shape[1]))
    haystack = np.sort(np.ascontiguousarray(sorted_unique_rows).view(row_dtype).reshape(-1))
    needles = np.ascontiguousarray(query_rows).view(row_dtype).reshape(-1)
    positions = np.searchsorted(haystack, needles)
    in_range = positions < haystack.size
    result = np.zeros(needles.size, dtype=np.bool_)
    result[in_range] = haystack[positions[in_range]] == needles[in_range]
    return result


def _barycentric_coordinates(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    edges = np.stack((triangle[1] - triangle[0], triangle[2] - triangle[0]), axis=1)
    coefficients, _residuals, rank, _ = np.linalg.lstsq(edges, point - triangle[0], rcond=None)
    if rank != 2:
        raise ValueError("carrier parent triangle is degenerate")
    barycentric = np.asarray(
        (1.0 - coefficients[0] - coefficients[1], coefficients[0], coefficients[1]),
        dtype=np.float64,
    )
    reconstructed = barycentric @ triangle
    scale = max(float(np.linalg.norm(np.ptp(triangle, axis=0))), 1.0)
    residual = max(
        float(np.linalg.norm(reconstructed - point)) / scale,
        max(0.0, -float(np.min(barycentric))),
    )
    if residual > 1e-9:
        raise ValueError("carrier vertex does not lie on its declared parent face")
    return barycentric


@dataclass(frozen=True)
class AmbientScaffoldV1:
    """A complete fixed-box tetrahedral complex with conforming surface subcomplexes."""

    vertices: np.ndarray
    tetrahedra: np.ndarray
    carrier_faces: np.ndarray
    carrier_face_parent_indices: np.ndarray
    carrier_vertex_indices: np.ndarray
    carrier_vertex_parent_faces: np.ndarray
    carrier_vertex_parent_barycentrics: np.ndarray
    fixed_source_faces: np.ndarray
    outer_boundary_mask: np.ndarray
    fixed_source_vertex_mask: np.ndarray
    region_labels: np.ndarray
    reference_determinants: np.ndarray
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    source_carrier_vertex_count: int
    source_carrier_face_count: int
    source_carrier_sha256: str
    constructor_bindings: dict[str, str] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        return AMBIENT_SCAFFOLD_SCHEMA_V1

    @property
    def content_hashes(self) -> dict[str, str]:
        arrays = {
            "vertices": (self.vertices, "<f8"),
            "tetrahedra": (self.tetrahedra, "<i8"),
            "carrier_faces": (self.carrier_faces, "<i8"),
            "carrier_face_parent_indices": (self.carrier_face_parent_indices, "<i8"),
            "carrier_vertex_indices": (self.carrier_vertex_indices, "<i8"),
            "carrier_vertex_parent_faces": (self.carrier_vertex_parent_faces, "<i8"),
            "carrier_vertex_parent_barycentrics": (
                self.carrier_vertex_parent_barycentrics,
                "<f8",
            ),
            "fixed_source_faces": (self.fixed_source_faces, "<i8"),
            "outer_boundary_mask": (self.outer_boundary_mask, "|b1"),
            "fixed_source_vertex_mask": (self.fixed_source_vertex_mask, "|b1"),
            "region_labels": (self.region_labels, "<i2"),
            "reference_determinants": (self.reference_determinants, "<f8"),
        }
        return {name: _array_hash(array, dtype) for name, (array, dtype) in arrays.items()}

    @property
    def scaffold_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "array_hashes": self.content_hashes,
            "bounds_lower": self.bounds_lower.tolist(),
            "bounds_upper": self.bounds_upper.tolist(),
            "source_carrier_vertex_count": self.source_carrier_vertex_count,
            "source_carrier_face_count": self.source_carrier_face_count,
            "source_carrier_sha256": self.source_carrier_sha256,
            "constructor_bindings": dict(sorted(self.constructor_bindings.items())),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        vertices = np.asarray(self.vertices)
        tetrahedra = np.asarray(self.tetrahedra)
        vertex_count = int(vertices.shape[0])
        tetrahedron_count = int(tetrahedra.shape[0])
        if vertices.dtype != np.float64 or vertices.shape != (vertex_count, 3):
            raise ValueError("ambient vertices must be float64 with shape [V, 3]")
        if not np.isfinite(vertices).all():
            raise ValueError("ambient vertices must be finite")
        if tetrahedra.dtype != np.int64 or tetrahedra.shape != (tetrahedron_count, 4):
            raise ValueError("ambient tetrahedra must be int64 with shape [T, 4]")
        if np.any(tetrahedra < 0) or np.any(tetrahedra >= vertex_count):
            raise ValueError("ambient tetrahedron contains an out-of-range vertex")
        if np.any(np.diff(np.sort(tetrahedra, axis=1), axis=1) == 0):
            raise ValueError("ambient tetrahedron contains a repeated vertex")
        lower = np.asarray(self.bounds_lower, dtype=np.float64)
        upper = np.asarray(self.bounds_upper, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("ambient bounds must be strictly ordered three-vectors")
        if np.any(vertices < lower) or np.any(vertices > upper):
            raise ValueError("ambient vertex lies outside the closed box")
        determinants = _tetrahedron_determinants(vertices, tetrahedra)
        if self.reference_determinants.shape != (tetrahedron_count,) or not np.array_equal(
            determinants, self.reference_determinants
        ):
            raise ValueError("reference determinant record does not match serialized vertices")
        if np.any(determinants <= 0.0) or not np.isfinite(determinants).all():
            raise ValueError("ambient complex contains a non-positive tetrahedron")

        for name, faces in (
            ("carrier", self.carrier_faces),
            ("fixed source", self.fixed_source_faces),
        ):
            if faces.dtype != np.int64 or faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"{name} faces must be int64 with shape [F, 3]")
            if np.any(faces < 0) or np.any(faces >= vertex_count):
                raise ValueError(f"{name} face contains an out-of-range vertex")
            if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
                raise ValueError(f"{name} face contains a repeated vertex")

        if self.outer_boundary_mask.dtype != np.bool_ or self.outer_boundary_mask.shape != (
            vertex_count,
        ):
            raise ValueError("outer-boundary mask must be boolean with shape [V]")
        if self.fixed_source_vertex_mask.dtype != np.bool_ or (
            self.fixed_source_vertex_mask.shape != (vertex_count,)
        ):
            raise ValueError("fixed-source mask must be boolean with shape [V]")
        if np.any(self.outer_boundary_mask & self.fixed_source_vertex_mask):
            raise ValueError("fixed-source and outer-boundary masks overlap")
        if self.region_labels.shape != (tetrahedron_count,):
            raise ValueError("region labels must have shape [T]")

        carrier_vertex_indices = np.asarray(self.carrier_vertex_indices)
        if carrier_vertex_indices.dtype != np.int64 or carrier_vertex_indices.ndim != 1:
            raise ValueError("carrier vertex indices must be int64 with shape [C]")
        if not np.array_equal(carrier_vertex_indices, np.unique(self.carrier_faces)):
            raise ValueError("carrier vertex indices do not exactly enumerate carrier faces")
        if np.any(self.outer_boundary_mask[carrier_vertex_indices]):
            raise ValueError("moving carrier touches the fixed outer boundary")
        if np.any(self.fixed_source_vertex_mask[carrier_vertex_indices]):
            raise ValueError("moving carrier and fixed source share a vertex")
        if self.carrier_face_parent_indices.shape != (self.carrier_faces.shape[0],):
            raise ValueError("carrier face parent indices must have shape [F]")
        if np.any(self.carrier_face_parent_indices < 0) or np.any(
            self.carrier_face_parent_indices >= self.source_carrier_face_count
        ):
            raise ValueError("carrier face has invalid parent provenance")
        if np.unique(self.carrier_face_parent_indices).size != self.source_carrier_face_count:
            raise ValueError("carrier subcomplex does not cover every source carrier face")
        carrier_vertex_count = carrier_vertex_indices.size
        if self.carrier_vertex_parent_faces.shape != (carrier_vertex_count,):
            raise ValueError("carrier vertex parent faces must have shape [C]")
        if self.carrier_vertex_parent_barycentrics.shape != (carrier_vertex_count, 3):
            raise ValueError("carrier vertex barycentrics must have shape [C, 3]")
        barycentric = self.carrier_vertex_parent_barycentrics
        if (
            not np.isfinite(barycentric).all()
            or np.max(np.abs(barycentric.sum(axis=1) - 1.0)) > 1e-9
        ):
            raise ValueError("carrier vertex barycentrics are invalid")
        if float(np.min(barycentric, initial=0.0)) < -1e-9:
            raise ValueError("carrier vertex barycentric lies outside its parent")

        facets = tetrahedra[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]
        sorted_facets = np.sort(facets.reshape(-1, 3), axis=1)
        unique_facets, incidence = np.unique(sorted_facets, axis=0, return_counts=True)
        if np.any((incidence != 1) & (incidence != 2)):
            raise ValueError("ambient complex has inconsistent facet incidence")
        for name, faces in (
            ("carrier", self.carrier_faces),
            ("fixed source", self.fixed_source_faces),
        ):
            contained = _rows_are_members(unique_facets, np.sort(faces, axis=1))
            if not np.all(contained):
                raise ValueError(f"{name} is not a conforming tetrahedral subcomplex")
        boundary_facets = unique_facets[incidence == 1]
        if boundary_facets.size == 0 or not np.all(self.outer_boundary_mask[boundary_facets]):
            raise ValueError("ambient complex has a non-box boundary facet")
        boundary_points = vertices[boundary_facets]
        on_plane = np.zeros(boundary_facets.shape[0], dtype=np.bool_)
        for axis in range(3):
            on_plane |= np.all(boundary_points[:, :, axis] == lower[axis], axis=1)
            on_plane |= np.all(boundary_points[:, :, axis] == upper[axis], axis=1)
        if not np.all(on_plane):
            raise ValueError("ambient boundary does not conform to the declared box")
        box_volume = float(np.prod(upper - lower))
        relative_volume_error = abs(float(determinants.sum()) / 6.0 - box_volume) / box_volume
        if relative_volume_error > 1e-9:
            raise ValueError("ambient tetrahedra do not fill the complete box volume")

    def report(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "status": "pass",
            "vertex_count": int(self.vertices.shape[0]),
            "tetrahedron_count": int(self.tetrahedra.shape[0]),
            "carrier_face_count": int(self.carrier_faces.shape[0]),
            "fixed_source_face_count": int(self.fixed_source_faces.shape[0]),
            "outer_boundary_vertex_count": int(np.count_nonzero(self.outer_boundary_mask)),
            "fixed_source_vertex_count": int(np.count_nonzero(self.fixed_source_vertex_mask)),
            "minimum_reference_determinant": float(self.reference_determinants.min()),
            "scaffold_sha256": self.scaffold_sha256,
            "content_hashes": self.content_hashes,
            "constructor_bindings": dict(sorted(self.constructor_bindings.items())),
        }

    def save(self, path: Path) -> None:
        self.validate()
        if path.exists():
            raise FileExistsError(f"immutable ambient scaffold exists: {path}")
        np.savez_compressed(
            path,
            schema_version=np.asarray(self.schema_version),
            vertices=self.vertices,
            tetrahedra=self.tetrahedra,
            carrier_faces=self.carrier_faces,
            carrier_face_parent_indices=self.carrier_face_parent_indices,
            carrier_vertex_indices=self.carrier_vertex_indices,
            carrier_vertex_parent_faces=self.carrier_vertex_parent_faces,
            carrier_vertex_parent_barycentrics=self.carrier_vertex_parent_barycentrics,
            fixed_source_faces=self.fixed_source_faces,
            outer_boundary_mask=self.outer_boundary_mask,
            fixed_source_vertex_mask=self.fixed_source_vertex_mask,
            region_labels=self.region_labels,
            reference_determinants=self.reference_determinants,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
            source_carrier_vertex_count=np.asarray(self.source_carrier_vertex_count),
            source_carrier_face_count=np.asarray(self.source_carrier_face_count),
            source_carrier_sha256=np.asarray(self.source_carrier_sha256),
            constructor_bindings=np.asarray(
                json.dumps(self.constructor_bindings, sort_keys=True, separators=(",", ":"))
            ),
            scaffold_sha256=np.asarray(self.scaffold_sha256),
        )

    @classmethod
    def load(cls, path: Path) -> AmbientScaffoldV1:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["schema_version"]) != AMBIENT_SCAFFOLD_SCHEMA_V1:
                raise ValueError("unsupported ambient scaffold schema")
            result = cls(
                vertices=np.asarray(payload["vertices"], dtype=np.float64),
                tetrahedra=np.asarray(payload["tetrahedra"], dtype=np.int64),
                carrier_faces=np.asarray(payload["carrier_faces"], dtype=np.int64),
                carrier_face_parent_indices=np.asarray(
                    payload["carrier_face_parent_indices"], dtype=np.int64
                ),
                carrier_vertex_indices=np.asarray(
                    payload["carrier_vertex_indices"], dtype=np.int64
                ),
                carrier_vertex_parent_faces=np.asarray(
                    payload["carrier_vertex_parent_faces"], dtype=np.int64
                ),
                carrier_vertex_parent_barycentrics=np.asarray(
                    payload["carrier_vertex_parent_barycentrics"], dtype=np.float64
                ),
                fixed_source_faces=np.asarray(payload["fixed_source_faces"], dtype=np.int64),
                outer_boundary_mask=np.asarray(payload["outer_boundary_mask"], dtype=np.bool_),
                fixed_source_vertex_mask=np.asarray(
                    payload["fixed_source_vertex_mask"], dtype=np.bool_
                ),
                region_labels=np.asarray(payload["region_labels"], dtype=np.int16),
                reference_determinants=np.asarray(
                    payload["reference_determinants"], dtype=np.float64
                ),
                bounds_lower=np.asarray(payload["bounds_lower"], dtype=np.float64),
                bounds_upper=np.asarray(payload["bounds_upper"], dtype=np.float64),
                source_carrier_vertex_count=int(payload["source_carrier_vertex_count"]),
                source_carrier_face_count=int(payload["source_carrier_face_count"]),
                source_carrier_sha256=str(payload["source_carrier_sha256"]),
                constructor_bindings=json.loads(str(payload["constructor_bindings"])),
            )
            expected_hash = str(payload["scaffold_sha256"])
        result.validate()
        if result.scaffold_sha256 != expected_hash:
            raise ValueError("ambient scaffold content hash mismatch")
        return result


def ambient_scaffold_from_interface_field(
    field: InterfaceField,
    *,
    source_carrier_vertices: np.ndarray,
    source_carrier_faces: np.ndarray,
    source_carrier_face_count: int,
    bounds: tuple[np.ndarray, np.ndarray],
    constructor_bindings: dict[str, str],
) -> AmbientScaffoldV1:
    """Promote an exact constrained-field complex into the E16 scaffold contract."""

    field.validate()
    source_vertices = np.asarray(source_carrier_vertices, dtype=np.float64)
    source_faces = np.asarray(source_carrier_faces, dtype=np.int64)
    if source_carrier_face_count != source_faces.shape[0]:
        raise ValueError("source carrier face count does not match source faces")
    carrier_select = field.source_face_indices < source_carrier_face_count
    fixed_select = ~carrier_select
    carrier_faces = np.asarray(field.interface_faces[carrier_select], dtype=np.int64)
    carrier_parents = np.asarray(field.source_face_indices[carrier_select], dtype=np.int64)
    fixed_faces = np.asarray(field.interface_faces[fixed_select], dtype=np.int64)
    carrier_vertices = np.unique(carrier_faces).astype(np.int64, copy=False)
    parent_by_vertex = np.empty(carrier_vertices.size, dtype=np.int64)
    barycentric = np.empty((carrier_vertices.size, 3), dtype=np.float64)
    child_face_lookup: dict[int, int] = {}
    for child_index, child in enumerate(carrier_faces):
        for vertex in child:
            child_face_lookup.setdefault(int(vertex), child_index)
    for slot, vertex in enumerate(carrier_vertices):
        child_index = child_face_lookup[int(vertex)]
        parent = int(carrier_parents[child_index])
        parent_by_vertex[slot] = parent
        barycentric[slot] = _barycentric_coordinates(
            field.vertices[vertex], source_vertices[source_faces[parent]]
        )

    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    scale = float(np.linalg.norm(upper - lower))
    tolerance = max(scale * 1e-12, np.finfo(np.float64).eps)
    outer = np.zeros(field.vertices.shape[0], dtype=np.bool_)
    for axis in range(3):
        outer |= np.isclose(field.vertices[:, axis], lower[axis], rtol=0.0, atol=tolerance)
        outer |= np.isclose(field.vertices[:, axis], upper[axis], rtol=0.0, atol=tolerance)
    fixed_vertices = np.zeros(field.vertices.shape[0], dtype=np.bool_)
    if fixed_faces.size:
        fixed_vertices[np.unique(fixed_faces)] = True
    result = AmbientScaffoldV1(
        vertices=np.asarray(field.vertices, dtype=np.float64).copy(),
        tetrahedra=np.asarray(field.tetrahedra, dtype=np.int64).copy(),
        carrier_faces=carrier_faces.copy(),
        carrier_face_parent_indices=carrier_parents.copy(),
        carrier_vertex_indices=carrier_vertices.copy(),
        carrier_vertex_parent_faces=parent_by_vertex,
        carrier_vertex_parent_barycentrics=barycentric,
        fixed_source_faces=fixed_faces.copy(),
        outer_boundary_mask=outer,
        fixed_source_vertex_mask=fixed_vertices,
        region_labels=np.asarray(field.cell_regions, dtype=np.int16).copy(),
        reference_determinants=_tetrahedron_determinants(field.vertices, field.tetrahedra),
        bounds_lower=lower.copy(),
        bounds_upper=upper.copy(),
        source_carrier_vertex_count=int(source_vertices.shape[0]),
        source_carrier_face_count=int(source_faces.shape[0]),
        source_carrier_sha256=_surface_hash(source_vertices, source_faces),
        constructor_bindings=dict(constructor_bindings),
    )
    result.validate()
    return result


def ambient_scaffold_from_constrained_complex(
    complex_: ConstrainedAmbientComplex,
    *,
    source_carrier_vertices: np.ndarray,
    source_carrier_faces: np.ndarray,
    source_carrier_face_count: int,
    constructor_bindings: dict[str, str],
) -> AmbientScaffoldV1:
    """Assign moving/fixed roles to the exact E16 base complex."""

    complex_.validate()
    interface_mask = np.zeros(complex_.vertices.shape[0], dtype=np.bool_)
    interface_mask[np.unique(complex_.interface_faces)] = True
    values = np.ones(complex_.vertices.shape[0], dtype=np.float64)
    values[interface_mask] = 0.0
    field = InterfaceField(
        vertices=complex_.vertices,
        values=values,
        interface_vertices=interface_mask,
        tetrahedra=complex_.tetrahedra,
        cell_regions=np.asarray(complex_.region_labels, dtype=np.int8),
        interface_faces=complex_.interface_faces,
        source_face_indices=complex_.source_face_indices,
        outside_cell_count=int(np.count_nonzero(complex_.region_labels == 1)),
        inside_cell_count=int(np.count_nonzero(complex_.region_labels == -1)),
        source_face_count=complex_.source_face_count,
    )
    return ambient_scaffold_from_interface_field(
        field,
        source_carrier_vertices=source_carrier_vertices,
        source_carrier_faces=source_carrier_faces,
        source_carrier_face_count=source_carrier_face_count,
        bounds=(complex_.bounds_lower, complex_.bounds_upper),
        constructor_bindings=constructor_bindings,
    )


@dataclass(frozen=True)
class HarmonicDirectionV1:
    displacement: np.ndarray
    normalized_residual: float
    free_vertex_count: int
    constrained_vertex_count: int
    source_proposal_sha256: str

    @property
    def direction_sha256(self) -> str:
        return _array_hash(self.displacement, "<f8")

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": HARMONIC_DIRECTION_SCHEMA_V1,
            "normalized_residual": self.normalized_residual,
            "free_vertex_count": self.free_vertex_count,
            "constrained_vertex_count": self.constrained_vertex_count,
            "source_proposal_sha256": self.source_proposal_sha256,
            "direction_sha256": self.direction_sha256,
        }


def lift_carrier_proposal(
    scaffold: AmbientScaffoldV1,
    source_carrier_faces: np.ndarray,
    source_proposal: np.ndarray,
) -> np.ndarray:
    scaffold.validate()
    faces = np.asarray(source_carrier_faces, dtype=np.int64)
    proposal = np.asarray(source_proposal, dtype=np.float64)
    if faces.shape != (scaffold.source_carrier_face_count, 3):
        raise ValueError("source carrier faces do not match the scaffold binding")
    if proposal.shape != (scaffold.source_carrier_vertex_count, 3):
        raise ValueError("source proposal must have shape [source vertices, 3]")
    if not np.isfinite(proposal).all():
        raise ValueError("source proposal must be finite")
    parents = faces[scaffold.carrier_vertex_parent_faces]
    return np.asarray(
        np.einsum("ci,cij->cj", scaffold.carrier_vertex_parent_barycentrics, proposal[parents]),
        dtype=np.float64,
    )


def solve_harmonic_direction(
    scaffold: AmbientScaffoldV1,
    source_carrier_faces: np.ndarray,
    source_proposal: np.ndarray,
    *,
    maximum_normalized_residual: float = 1e-10,
) -> HarmonicDirectionV1:
    """Solve the fixed Dirichlet volumetric harmonic extension in float64."""

    scaffold.validate()
    carrier_displacement = lift_carrier_proposal(scaffold, source_carrier_faces, source_proposal)
    vertices = scaffold.vertices
    tetrahedra = scaffold.tetrahedra
    points = vertices[tetrahedra]
    first_edge = points[:, 1] - points[:, 0]
    second_edge = points[:, 2] - points[:, 0]
    third_edge = points[:, 3] - points[:, 0]
    gradients = np.empty((tetrahedra.shape[0], 4, 3), dtype=np.float64)
    gradients[:, 1] = np.cross(second_edge, third_edge) / scaffold.reference_determinants[:, None]
    gradients[:, 2] = np.cross(third_edge, first_edge) / scaffold.reference_determinants[:, None]
    gradients[:, 3] = np.cross(first_edge, second_edge) / scaffold.reference_determinants[:, None]
    gradients[:, 0] = -np.sum(gradients[:, 1:], axis=1)
    volumes = scaffold.reference_determinants / 6.0
    local = volumes[:, None, None] * np.einsum("tid,tjd->tij", gradients, gradients)
    rows = np.repeat(tetrahedra, 4, axis=1).reshape(-1)
    columns = np.tile(tetrahedra, (1, 4)).reshape(-1)
    stiffness = sparse.coo_matrix(
        (local.reshape(-1), (rows, columns)), shape=(vertices.shape[0], vertices.shape[0])
    ).tocsr()

    constrained = scaffold.outer_boundary_mask | scaffold.fixed_source_vertex_mask
    constrained = constrained.copy()
    constrained[scaffold.carrier_vertex_indices] = True
    free = np.flatnonzero(~constrained)
    fixed = np.flatnonzero(constrained)
    direction = np.zeros_like(vertices)
    direction[scaffold.carrier_vertex_indices] = carrier_displacement
    if free.size:
        system = stiffness[free][:, free].tocsc()
        right_hand_side = -(stiffness[free][:, fixed] @ direction[fixed])
        for axis in range(3):
            direction[free, axis] = sparse_linalg.spsolve(system, right_hand_side[:, axis])
        residual = system @ direction[free] - right_hand_side
        normalized_residual = float(
            np.linalg.norm(residual) / max(float(np.linalg.norm(right_hand_side)), 1.0)
        )
    else:
        normalized_residual = 0.0
    if not np.isfinite(direction).all() or not np.isfinite(normalized_residual):
        raise ValueError("harmonic solve produced a non-finite direction")
    if normalized_residual > maximum_normalized_residual:
        raise ValueError(
            f"harmonic normalized residual {normalized_residual} exceeds "
            f"{maximum_normalized_residual}"
        )
    if np.any(direction[scaffold.outer_boundary_mask] != 0.0):
        raise ValueError("harmonic direction moved the fixed outer boundary")
    if np.any(direction[scaffold.fixed_source_vertex_mask] != 0.0):
        raise ValueError("harmonic direction moved the fixed source")
    if not np.array_equal(direction[scaffold.carrier_vertex_indices], carrier_displacement):
        raise ValueError("harmonic direction did not retain the complete carrier proposal")
    return HarmonicDirectionV1(
        displacement=np.asarray(direction, dtype=np.float64),
        normalized_residual=normalized_residual,
        free_vertex_count=int(free.size),
        constrained_vertex_count=int(fixed.size),
        source_proposal_sha256=_array_hash(np.asarray(source_proposal), "<f8"),
    )
