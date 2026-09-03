from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

import numpy as np
import trimesh

EXPERIMENT_ID = "postv1_e11_genus_controlled_carrier_r01"
FIXTURE_SCHEMA = "e11_public_fixture.v1"
EXPANSION_NUMERATOR = 101
EXPANSION_DENOMINATOR = 100
PUBLIC_FIXTURE_DEFINITION = {
    "schema_version": "e11_public_fixture_definition.v1",
    "fixtures": [
        "sphere_control",
        "rotated_ellipsoid",
        "concave_star",
        "genus_one_torus",
        "two_component_spheres",
    ],
    "constructor": "CGAL_6.2_EPECK_convex_hull_fixed_general_position",
    "expansion_ratio": [EXPANSION_NUMERATOR, EXPANSION_DENOMINATOR],
}
PUBLIC_FIXTURE_INPUT_SHA256 = hashlib.sha256(
    json.dumps(PUBLIC_FIXTURE_DEFINITION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

FIDELITY_SEED = 20260831
FIDELITY_SAMPLE_COUNT = 20_000
FIDELITY_MAXIMUM_MEAN_DISTANCE_PITCH = 0.5
FIDELITY_MAXIMUM_P95_DISTANCE_PITCH = 1.0
FIDELITY_MAXIMUM_MEDIAN_NORMAL_DEGREES = 5.0
# A 1% radial expansion has an analytic 3.0301% volume floor. The 3.1% gate
# leaves less than 0.07 percentage point for any additional volume distortion.
FIDELITY_MAXIMUM_RELATIVE_VOLUME_ERROR = 0.031
FIDELITY_MAXIMUM_FEATURE_P95_DISTANCE_PITCH = 1.0
FIDELITY_MAXIMUM_INVARIANCE_DELTA = 1e-5


@dataclass(frozen=True)
class GenusCarrierFixture:
    """Public procedural source for the genus-by-construction gate."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    source_euler_number: int
    source_component_count: int

    def validate(self) -> None:
        vertices = np.asarray(self.vertices)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError(f"{self.name}: vertices must be finite [V,3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{self.name}: faces must have shape [F,3]")
        if len(vertices) < 4 or len(faces) < 4:
            raise ValueError(f"{self.name}: fixture is too small")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError(f"{self.name}: face index is out of range")
        centered = vertices - vertices.mean(axis=0)
        if int(np.linalg.matrix_rank(centered)) != 3:
            raise ValueError(f"{self.name}: vertices do not span three dimensions")

    def as_public_record(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": FIXTURE_SCHEMA,
            "name": self.name,
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "source_euler_number": self.source_euler_number,
            "source_component_count": self.source_component_count,
        }


@dataclass(frozen=True)
class GenusCarrierFidelityFixture:
    """Public clean reference, constructor input, and registered feature probes."""

    name: str
    source_vertices: np.ndarray
    source_faces: np.ndarray
    reference_vertices: np.ndarray
    reference_faces: np.ndarray
    pitch: float
    exterior_probes: np.ndarray
    feature_face_indices: np.ndarray
    invariance_group: str | None = None

    def validate(self) -> None:
        for label, vertices, faces in (
            ("source", self.source_vertices, self.source_faces),
            ("reference", self.reference_vertices, self.reference_faces),
        ):
            vertices = np.asarray(vertices)
            faces = np.asarray(faces)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError(f"{self.name}: {label} vertices must be finite [V,3]")
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"{self.name}: {label} faces must be [F,3]")
            if np.any(faces < 0) or np.any(faces >= len(vertices)):
                raise ValueError(f"{self.name}: {label} face index is out of range")
        if not np.isfinite(self.pitch) or self.pitch <= 0:
            raise ValueError(f"{self.name}: pitch must be positive")
        probes = np.asarray(self.exterior_probes)
        if probes.ndim != 2 or probes.shape[1] != 3 or not np.isfinite(probes).all():
            raise ValueError(f"{self.name}: exterior probes must be finite [P,3]")
        indices = np.asarray(self.feature_face_indices)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.reference_faces)):
            raise ValueError(f"{self.name}: feature face indices are invalid")
        reference = trimesh.Trimesh(
            vertices=np.asarray(self.reference_vertices),
            faces=np.asarray(self.reference_faces),
            process=False,
        )
        if not reference.is_watertight or not reference.is_winding_consistent:
            raise ValueError(f"{self.name}: reference must be watertight and consistently wound")
        if int(reference.euler_number) != 2 or len(reference.split(only_watertight=False)) != 1:
            raise ValueError(f"{self.name}: reference must be one Euler-2 component")
        if len(probes) and bool(np.any(reference.contains(probes))):
            raise ValueError(f"{self.name}: registered exterior probe is inside the reference")

    def as_public_record(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "e11_public_fidelity_fixture.v1",
            "name": self.name,
            "source_vertex_count": len(self.source_vertices),
            "source_face_count": len(self.source_faces),
            "reference_vertex_count": len(self.reference_vertices),
            "reference_face_count": len(self.reference_faces),
            "pitch": self.pitch,
            "exterior_probe_count": len(self.exterior_probes),
            "feature_face_count": len(self.feature_face_indices),
            "invariance_group": self.invariance_group,
        }


def _star_surface() -> trimesh.Trimesh:
    mesh = cast(trimesh.Trimesh, trimesh.creation.icosphere(subdivisions=2, radius=1.0))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    radius = np.linalg.norm(vertices, axis=1)
    azimuth = np.arctan2(vertices[:, 1], vertices[:, 0])
    elevation_weight = 1.0 - np.square(vertices[:, 2] / radius)
    scale = 1.0 + 0.22 * np.cos(5.0 * azimuth) * elevation_weight
    mesh.vertices = vertices * scale[:, None]
    return mesh


def _fixture(name: str, mesh: trimesh.Trimesh) -> GenusCarrierFixture:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return GenusCarrierFixture(
        name=name,
        vertices=vertices,
        faces=faces,
        source_euler_number=int(mesh.euler_number),
        source_component_count=len(mesh.split(only_watertight=False)),
    )


def public_genus_carrier_fixtures() -> tuple[GenusCarrierFixture, ...]:
    """Return public sources spanning concavity, handles, and disconnected parts."""
    sphere = cast(trimesh.Trimesh, trimesh.creation.icosphere(subdivisions=2, radius=1.0))
    ellipsoid = sphere.copy()
    ellipsoid.apply_scale((1.0, 0.68, 1.31))  # type: ignore[no-untyped-call]
    ellipsoid.apply_transform(
        trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
            np.deg2rad(29.0), (0.3, 0.8, 0.5)
        )
    )
    torus = cast(
        trimesh.Trimesh,
        trimesh.creation.torus(
            major_radius=0.85, minor_radius=0.24, major_sections=32, minor_sections=16
        ),
    )
    first_component = cast(trimesh.Trimesh, trimesh.creation.icosphere(subdivisions=1, radius=0.52))
    second_component = first_component.copy()
    first_component.apply_translation((-0.72, 0.0, 0.0))
    second_component.apply_translation((0.72, 0.0, 0.0))
    disconnected = cast(
        trimesh.Trimesh, trimesh.util.concatenate((first_component, second_component))
    )
    fixtures = (
        _fixture("sphere_control", sphere),
        _fixture("rotated_ellipsoid", ellipsoid),
        _fixture("concave_star", _star_surface()),
        _fixture("genus_one_torus", torus),
        _fixture("two_component_spheres", disconnected),
    )
    for fixture in fixtures:
        fixture.validate()
    return fixtures


def _capped_hairpin_surface() -> trimesh.Trimesh:
    arm_half_separation = 0.35
    tube_radius = 0.12
    arm_height = 1.0
    left = np.column_stack(
        (
            np.full(14, -arm_half_separation),
            np.linspace(arm_height, 0.0, 14),
            np.zeros(14),
        )
    )
    theta = np.linspace(np.pi, 2.0 * np.pi, 28)
    bend = np.column_stack(
        (
            arm_half_separation * np.cos(theta),
            arm_half_separation * np.sin(theta),
            np.zeros_like(theta),
        )
    )
    right = np.column_stack(
        (
            np.full(14, arm_half_separation),
            np.linspace(0.0, arm_height, 14),
            np.zeros(14),
        )
    )
    centers = np.vstack((left, bend[1:], right[1:]))
    tangents = np.empty_like(centers)
    tangents[0] = centers[1] - centers[0]
    tangents[-1] = centers[-1] - centers[-2]
    tangents[1:-1] = centers[2:] - centers[:-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    binormal = np.asarray((0.0, 0.0, 1.0))
    planar_normals = np.cross(np.broadcast_to(binormal, tangents.shape), tangents)
    planar_normals /= np.linalg.norm(planar_normals, axis=1, keepdims=True)
    ring_angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    rings = np.asarray(
        [
            center
            + tube_radius
            * (
                np.cos(ring_angles)[:, None] * normal[None]
                + np.sin(ring_angles)[:, None] * binormal[None]
            )
            for center, normal in zip(centers, planar_normals, strict=True)
        ]
    )
    vertices = rings.reshape(-1, 3)
    faces: list[tuple[int, int, int]] = []
    ring_size = rings.shape[1]
    for path_index in range(len(centers) - 1):
        current = path_index * ring_size
        following = (path_index + 1) * ring_size
        for ring_index in range(ring_size):
            next_ring = (ring_index + 1) % ring_size
            faces.append((current + ring_index, following + ring_index, following + next_ring))
            faces.append((current + ring_index, following + next_ring, current + next_ring))
    start_center = len(vertices)
    end_center = start_center + 1
    vertices = np.vstack((vertices, centers[0], centers[-1]))
    for ring_index in range(ring_size):
        next_ring = (ring_index + 1) % ring_size
        faces.append((start_center, next_ring, ring_index))
        end_base = (len(centers) - 1) * ring_size
        faces.append((end_center, end_base + ring_index, end_base + next_ring))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.fix_normals()
    return mesh


def _concave_pocket_surface() -> trimesh.Trimesh:
    mesh = cast(trimesh.Trimesh, trimesh.creation.icosphere(subdivisions=3, radius=1.0))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    directions = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    pocket_weight = np.square(np.clip((directions[:, 0] - 0.30) / 0.70, 0.0, 1.0))
    mesh.vertices = vertices * (1.0 - 0.48 * pocket_weight)[:, None]
    mesh.fix_normals()
    return mesh


def _fidelity_fixture(
    name: str,
    reference: trimesh.Trimesh,
    *,
    pitch: float,
    exterior_probes: np.ndarray | None = None,
    feature_face_indices: np.ndarray | None = None,
    source_faces: np.ndarray | None = None,
    invariance_group: str | None = None,
) -> GenusCarrierFidelityFixture:
    reference_vertices = np.asarray(reference.vertices, dtype=np.float64)
    reference_faces = np.asarray(reference.faces, dtype=np.int64)
    fixture = GenusCarrierFidelityFixture(
        name=name,
        source_vertices=reference_vertices.copy(),
        source_faces=(
            np.asarray(source_faces, dtype=np.int64).copy()
            if source_faces is not None
            else reference_faces.copy()
        ),
        reference_vertices=reference_vertices.copy(),
        reference_faces=reference_faces.copy(),
        pitch=float(pitch),
        exterior_probes=(
            np.empty((0, 3), dtype=np.float64)
            if exterior_probes is None
            else np.asarray(exterior_probes, dtype=np.float64)
        ),
        feature_face_indices=(
            np.empty(0, dtype=np.int64)
            if feature_face_indices is None
            else np.asarray(feature_face_indices, dtype=np.int64)
        ),
        invariance_group=invariance_group,
    )
    fixture.validate()
    return fixture


def public_genus_fidelity_fixtures() -> tuple[GenusCarrierFidelityFixture, ...]:
    """Return frozen public references for the E11 geometric-fidelity gate."""
    sphere = cast(trimesh.Trimesh, trimesh.creation.icosphere(subdivisions=2, radius=1.0))
    sphere_pitch = float(np.linalg.norm(np.diff(sphere.bounds, axis=0)[0]) / 40.0)
    sphere_small = sphere.copy()
    sphere_small.apply_scale(0.1)  # type: ignore[no-untyped-call]
    sphere_large = sphere.copy()
    sphere_large.apply_scale(10.0)  # type: ignore[no-untyped-call]

    ellipsoid = sphere.copy()
    ellipsoid.apply_scale((1.0, 0.68, 1.31))  # type: ignore[no-untyped-call]
    ellipsoid_pitch = float(np.linalg.norm(np.diff(ellipsoid.bounds, axis=0)[0]) / 40.0)
    rigid_ellipsoid = ellipsoid.copy()
    rigid_ellipsoid.apply_transform(
        trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
            np.deg2rad(37.0), (0.2, 0.9, 0.3)
        )
    )
    rigid_ellipsoid.apply_translation((0.7, -0.4, 0.2))

    pocket = _concave_pocket_surface()
    pocket_pitch = float(np.linalg.norm(np.diff(pocket.bounds, axis=0)[0]) / 60.0)
    pocket_centroids = np.asarray(pocket.triangles_center)
    pocket_features = np.flatnonzero(pocket_centroids[:, 0] > 0.20)
    pocket_probes = np.asarray(((0.72, 0.0, 0.0), (0.80, 0.08, 0.0)), dtype=np.float64)
    defective_faces = np.vstack(
        (
            np.asarray(pocket.faces, dtype=np.int64),
            np.asarray(pocket.faces[:4, ::-1], dtype=np.int64),
            np.asarray([[0, len(pocket.vertices) // 3, 2 * len(pocket.vertices) // 3]]),
        )
    )

    hairpin = _capped_hairpin_surface()
    hairpin_centroids = np.asarray(hairpin.triangles_center)
    hairpin_features = np.flatnonzero(
        (np.abs(hairpin_centroids[:, 0]) < 0.35) & (hairpin_centroids[:, 1] > 0.10)
    )
    hairpin_probes = np.asarray(
        ((0.0, 0.25, 0.0), (0.0, 0.50, 0.0), (0.0, 0.75, 0.0)), dtype=np.float64
    )
    hairpin_pitch = (2.0 * (0.35 - 0.12)) / 2.2

    fixtures = (
        _fidelity_fixture(
            "sphere_control", sphere, pitch=sphere_pitch, invariance_group="sphere_scale"
        ),
        _fidelity_fixture(
            "sphere_scale_0_1",
            sphere_small,
            pitch=sphere_pitch * 0.1,
            invariance_group="sphere_scale",
        ),
        _fidelity_fixture(
            "sphere_scale_10",
            sphere_large,
            pitch=sphere_pitch * 10.0,
            invariance_group="sphere_scale",
        ),
        _fidelity_fixture(
            "rotated_ellipsoid",
            ellipsoid,
            pitch=ellipsoid_pitch,
            invariance_group="ellipsoid_rigid",
        ),
        _fidelity_fixture(
            "rigid_ellipsoid",
            rigid_ellipsoid,
            pitch=ellipsoid_pitch,
            invariance_group="ellipsoid_rigid",
        ),
        _fidelity_fixture(
            "concave_pocket",
            pocket,
            pitch=pocket_pitch,
            exterior_probes=pocket_probes,
            feature_face_indices=pocket_features,
        ),
        _fidelity_fixture(
            "concave_pocket_defective_soup",
            pocket,
            pitch=pocket_pitch,
            exterior_probes=pocket_probes,
            feature_face_indices=pocket_features,
            source_faces=defective_faces,
        ),
        _fidelity_fixture(
            "near_contact_hairpin",
            hairpin,
            pitch=hairpin_pitch,
            exterior_probes=hairpin_probes,
            feature_face_indices=hairpin_features,
        ),
    )
    for fixture in fixtures:
        fixture.validate()
    return fixtures


def _public_fidelity_input_sha256() -> str:
    definition = {
        "schema_version": "e11_public_fidelity_definition.v1",
        "constructor": PUBLIC_FIXTURE_DEFINITION["constructor"],
        "expansion_ratio": [EXPANSION_NUMERATOR, EXPANSION_DENOMINATOR],
        "seed": FIDELITY_SEED,
        "sample_count_per_direction": FIDELITY_SAMPLE_COUNT,
        "thresholds": {
            "maximum_mean_distance_pitch": FIDELITY_MAXIMUM_MEAN_DISTANCE_PITCH,
            "maximum_p95_distance_pitch": FIDELITY_MAXIMUM_P95_DISTANCE_PITCH,
            "maximum_median_normal_degrees": FIDELITY_MAXIMUM_MEDIAN_NORMAL_DEGREES,
            "maximum_relative_volume_error": FIDELITY_MAXIMUM_RELATIVE_VOLUME_ERROR,
            "maximum_feature_p95_distance_pitch": FIDELITY_MAXIMUM_FEATURE_P95_DISTANCE_PITCH,
            "maximum_invariance_delta": FIDELITY_MAXIMUM_INVARIANCE_DELTA,
        },
    }
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode())
    for fixture in public_genus_fidelity_fixtures():
        digest.update(fixture.name.encode())
        digest.update(np.asarray(fixture.source_vertices, dtype="<f8").tobytes())
        digest.update(np.asarray(fixture.source_faces, dtype="<i8").tobytes())
        digest.update(np.asarray(fixture.reference_vertices, dtype="<f8").tobytes())
        digest.update(np.asarray(fixture.reference_faces, dtype="<i8").tobytes())
        digest.update(np.asarray([fixture.pitch], dtype="<f8").tobytes())
        digest.update(np.asarray(fixture.exterior_probes, dtype="<f8").tobytes())
        digest.update(np.asarray(fixture.feature_face_indices, dtype="<i8").tobytes())
        digest.update((fixture.invariance_group or "").encode())
    return digest.hexdigest()


PUBLIC_FIDELITY_INPUT_SHA256 = _public_fidelity_input_sha256()
