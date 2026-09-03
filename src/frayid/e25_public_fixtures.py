from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from frayid.camera import make_intrinsics
from frayid.hybrid_tetrahedral import (
    fixed_sign_surface_connectivity,
    regular_tetrahedral_grid,
)
from frayid.renderer import render_soft_mesh

E25_PUBLIC_FIXTURE_NAMES = (
    "rotated_ellipsoid",
    "concave_pocket_thin_bridge_near_gap",
    "articulated_two_part_body_with_known_jacobians",
    "fine_wrinkle",
    "cross_cell_surface_motion",
)
E25_PUBLIC_MODALITIES = frozenset(("mask", "boundary", "normal"))
E25_PUBLIC_VIEW_COUNT = 18
E25_PUBLIC_TRAIN_VIEW_COUNT = 12
E25_PUBLIC_HELD_OUT_VIEW_COUNT = 6
E25_PUBLIC_EXTENT = 1.25
E25_PUBLIC_CAMERA_DISTANCE = 3.2
E25_PUBLIC_FOCAL_PIXELS = 115.0

_PROTECTED_PREFIXES = (
    "data/private/",
    "models/private/",
    "models/checkpoints/",
    "docs/assets/",
    "outputs/canonical_clothed_surface_v1/",
    "outputs/development/",
    "outputs/sealed/",
)


@dataclass(frozen=True)
class E25PublicFixture:
    name: str
    extent: float
    field: Callable[[Tensor], Tensor]
    description: str


@dataclass(frozen=True)
class E25PublicMesh:
    vertices: Tensor
    faces: Tensor
    grid_positions: Tensor
    grid_values: Tensor


@dataclass(frozen=True)
class E25PublicEvidence:
    silhouettes: Tensor
    normals: Tensor
    intrinsics: Tensor
    rotations: Tensor
    translations: Tensor


def _ellipsoid(points: Tensor) -> Tensor:
    x, y, z = points.unbind(-1)
    return torch.sqrt((x / 0.68).square() + (y / 0.91).square() + (z / 0.49).square()) - 1.0


def _concave_pocket(points: Tensor) -> Tensor:
    x, y, z = points.unbind(-1)
    main = torch.sqrt((x / 0.66).square() + (y / 0.90).square() + (z / 0.48).square()) - 1.0
    lobe = (
        torch.sqrt(
            ((x - 0.76) / 0.30).square() + ((y - 0.16) / 0.42).square() + (z / 0.24).square()
        )
        - 1.0
    )
    bridge = (
        torch.maximum(
            torch.maximum(torch.abs(x - 0.48) / 0.34, torch.abs(y + 0.26) / 0.12),
            torch.abs(z) / 0.14,
        )
        - 1.0
    )
    cutter = (
        torch.sqrt(
            ((x - 0.48) / 0.30).square()
            + ((y - 0.22) / 0.32).square()
            + ((z - 0.34) / 0.28).square()
        )
        - 1.0
    )
    return torch.maximum(torch.minimum(torch.minimum(main, lobe), bridge), -cutter)


def _articulated_two_part(points: Tensor) -> Tensor:
    x, y, z = points.unbind(-1)
    torso = torch.sqrt((x / 0.48).square() + ((y + 0.20) / 0.72).square() + (z / 0.36).square())
    upper = torch.sqrt(
        ((x - 0.29) / 0.34).square() + ((y - 0.52) / 0.47).square() + ((z + 0.04) / 0.31).square()
    )
    return torch.minimum(torso - 1.0, upper - 1.0)


def _fine_wrinkle(points: Tensor) -> Tensor:
    x, y, z = points.unbind(-1)
    radius = torch.sqrt((x / 0.63).square() + (y / 0.88).square() + (z / 0.47).square())
    azimuth = torch.atan2(z, x)
    wrinkle = 0.035 * torch.sin(10.0 * azimuth + 3.0 * y) * torch.exp(-2.0 * y.square())
    return radius - 1.0 + wrinkle


def _cross_cell(points: Tensor) -> Tensor:
    center = points.new_tensor((0.073, -0.041, 0.058))
    result: Tensor = torch.linalg.vector_norm(points - center, dim=-1) - 0.61
    return result


def public_fixture_registry() -> tuple[E25PublicFixture, ...]:
    return (
        E25PublicFixture(
            "rotated_ellipsoid",
            E25_PUBLIC_EXTENT,
            _ellipsoid,
            "anisotropic smooth surface used for continuous-normal checks",
        ),
        E25PublicFixture(
            "concave_pocket_thin_bridge_near_gap",
            E25_PUBLIC_EXTENT,
            _concave_pocket,
            "G22-compatible pocket, bridge, and near-gap geometry",
        ),
        E25PublicFixture(
            "articulated_two_part_body_with_known_jacobians",
            E25_PUBLIC_EXTENT,
            _articulated_two_part,
            "overlapping two-part genus-zero body with an analytic affine pose Jacobian",
        ),
        E25PublicFixture(
            "fine_wrinkle",
            E25_PUBLIC_EXTENT,
            _fine_wrinkle,
            "high-frequency directional-normal fixture",
        ),
        E25PublicFixture(
            "cross_cell_surface_motion",
            E25_PUBLIC_EXTENT,
            _cross_cell,
            "off-grid sphere whose zero set crosses extraction cells under translation",
        ),
    )


def fixture_by_name(name: str) -> E25PublicFixture:
    matches = [fixture for fixture in public_fixture_registry() if fixture.name == name]
    if len(matches) != 1:
        raise KeyError(f"unknown E25 public fixture: {name}")
    return matches[0]


def extract_public_truth_mesh(
    fixture: E25PublicFixture,
    *,
    resolution: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> E25PublicMesh:
    if resolution < 8:
        raise ValueError("E25 truth extraction resolution must be at least eight")
    positions, tetrahedra = regular_tetrahedral_grid(
        resolution,
        extent=fixture.extent,
        device=device,
    )
    positions = positions.to(dtype=dtype)
    values = fixture.field(positions)
    if values.shape != (positions.shape[0],) or not torch.isfinite(values).all():
        raise ValueError("E25 public fixture returned an invalid scalar field")
    tolerance = torch.finfo(values.dtype).eps * 32.0
    zero = values.abs() <= tolerance
    if torch.any(zero):
        ordinal = torch.arange(values.numel(), device=values.device)
        perturbation = torch.where(ordinal.remainder(2) == 0, tolerance, -tolerance)
        values = torch.where(zero, perturbation, values)
    edges, faces = fixed_sign_surface_connectivity(positions, tetrahedra, torch.sign(values))
    endpoints = positions[edges]
    edge_values = values[edges]
    interpolation = edge_values[:, 0] / (edge_values[:, 0] - edge_values[:, 1])
    vertices = endpoints[:, 0] + interpolation[:, None] * (endpoints[:, 1] - endpoints[:, 0])
    return E25PublicMesh(vertices, faces, positions, values)


def public_camera_bundle(
    *,
    image_size: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("E25 public image size must be positive")
    focal = E25_PUBLIC_FOCAL_PIXELS * width / 128.0
    intrinsics = make_intrinsics(
        focal,
        (width / 2.0, height / 2.0),
        dtype=dtype,
        device=device,
    )
    rotations: list[Tensor] = []
    for view in range(E25_PUBLIC_VIEW_COUNT):
        yaw = 2.0 * math.pi * view / E25_PUBLIC_VIEW_COUNT
        pitch = math.radians((-12.0, 0.0, 12.0)[view % 3])
        cosine_yaw, sine_yaw = math.cos(yaw), math.sin(yaw)
        cosine_pitch, sine_pitch = math.cos(pitch), math.sin(pitch)
        rotate_y = torch.tensor(
            ((cosine_yaw, 0.0, sine_yaw), (0.0, 1.0, 0.0), (-sine_yaw, 0.0, cosine_yaw)),
            dtype=dtype,
            device=device,
        )
        rotate_x = torch.tensor(
            ((1.0, 0.0, 0.0), (0.0, cosine_pitch, -sine_pitch), (0.0, sine_pitch, cosine_pitch)),
            dtype=dtype,
            device=device,
        )
        rotations.append(rotate_x @ rotate_y)
    rotation_tensor = torch.stack(rotations)
    translations = torch.tensor(
        (0.0, 0.0, E25_PUBLIC_CAMERA_DISTANCE), dtype=dtype, device=device
    ).repeat(E25_PUBLIC_VIEW_COUNT, 1)
    return intrinsics, rotation_tensor, translations


def render_public_mesh_evidence(
    mesh: E25PublicMesh,
    *,
    image_size: tuple[int, int] = (128, 128),
    target_sample_count: int = 2048,
    reference_sample_count: int = 2048,
) -> E25PublicEvidence:
    intrinsics, rotations, translations = public_camera_bundle(
        image_size=image_size,
        dtype=mesh.vertices.dtype,
        device=mesh.vertices.device,
    )
    silhouettes: list[Tensor] = []
    normals: list[Tensor] = []
    sigma_pixels = 1.75 * max(image_size) / 128.0
    with torch.no_grad():
        for view in range(E25_PUBLIC_VIEW_COUNT):
            torch.manual_seed(25_000 + view)
            posed = mesh.vertices @ rotations[view].T + translations[view]
            silhouette, normal = render_soft_mesh(
                posed,
                mesh.faces,
                intrinsics,
                image_size,
                source_image_size=image_size,
                sigma_pixels=sigma_pixels,
                sample_count=target_sample_count,
                reference_sample_count=reference_sample_count,
                chunk_size=256,
            )
            silhouettes.append(silhouette.detach())
            normals.append(normal.detach())
    evidence = E25PublicEvidence(
        torch.stack(silhouettes),
        torch.stack(normals),
        intrinsics,
        rotations,
        translations,
    )
    validate_public_evidence(evidence)
    return evidence


def validate_public_evidence(evidence: E25PublicEvidence) -> None:
    if evidence.silhouettes.ndim != 3:
        raise ValueError("E25 silhouettes must have shape [view,height,width]")
    view_count, height, width = evidence.silhouettes.shape
    if view_count != E25_PUBLIC_VIEW_COUNT:
        raise ValueError("E25 evidence must retain the registered 12/6 view split")
    if evidence.normals.shape != (view_count, height, width, 3):
        raise ValueError("E25 normal evidence shape does not match silhouettes")
    if evidence.intrinsics.shape != (3, 3):
        raise ValueError("E25 public intrinsics must be shared across views")
    if evidence.rotations.shape != (view_count, 3, 3) or evidence.translations.shape != (
        view_count,
        3,
    ):
        raise ValueError("E25 public camera bundle has invalid shape")
    tensors = (
        evidence.silhouettes,
        evidence.normals,
        evidence.intrinsics,
        evidence.rotations,
        evidence.translations,
    )
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("E25 public evidence must be finite")
    if torch.any((evidence.silhouettes < 0.0) | (evidence.silhouettes > 1.0)):
        raise ValueError("E25 public silhouettes must lie in [0,1]")
    valid = evidence.silhouettes > 0.5
    if not torch.any(valid):
        raise ValueError("E25 public evidence contains no foreground")
    lengths = torch.linalg.vector_norm(evidence.normals[valid], dim=-1)
    supported = lengths > 0.5
    if float(supported.to(torch.float32).mean()) < 0.5:
        raise ValueError("E25 foreground normal coverage is insufficient")
    if torch.any((lengths[supported] < 0.95) | (lengths[supported] > 1.05)):
        raise ValueError("E25 foreground normals must be unit length")


def validate_public_modalities(modalities: Iterable[str]) -> None:
    supplied = frozenset(modalities)
    if supplied != E25_PUBLIC_MODALITIES:
        added = sorted(supplied - E25_PUBLIC_MODALITIES)
        missing = sorted(E25_PUBLIC_MODALITIES - supplied)
        raise ValueError(f"E25 public modalities changed; added={added}, missing={missing}")


def assert_public_read_allowed(project_root: Path, candidate: Path) -> None:
    root = project_root.resolve()
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return
    normalized = relative.rstrip("/") + ("/" if candidate.is_dir() else "")
    if any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in _PROTECTED_PREFIXES
    ):
        raise PermissionError(f"E25 public stage rejected protected read: {relative}")


def articulated_pose_jacobian(
    points: Tensor,
    *,
    shear: float = 0.18,
    vertical_scale: float = 1.08,
) -> Tensor:
    if points.shape[-1] != 3 or not torch.isfinite(points).all():
        raise ValueError("articulated public points must be finite [...,3]")
    base = points.new_tensor(((1.0, shear, 0.0), (0.0, vertical_scale, 0.0), (0.0, 0.0, 0.94)))
    return base.expand(*points.shape[:-1], 3, 3)


def move_cross_cell_fixture(points: Tensor, *, grid_pitch: float) -> Tensor:
    if grid_pitch <= 0:
        raise ValueError("cross-cell grid pitch must be positive")
    displacement = points.new_tensor((0.6 * grid_pitch, -0.4 * grid_pitch, 0.5 * grid_pitch))
    return points + displacement


def finite_difference_field_normal(
    field: Callable[[Tensor], Tensor], points: Tensor, *, epsilon: float = 1.0e-4
) -> Tensor:
    if epsilon <= 0 or points.shape[-1] != 3:
        raise ValueError("finite-difference normal contract is invalid")
    differences: list[Tensor] = []
    for axis in range(3):
        offset = torch.zeros_like(points)
        offset[..., axis] = epsilon
        differences.append((field(points + offset) - field(points - offset)) / (2.0 * epsilon))
    return F.normalize(torch.stack(differences, dim=-1), dim=-1, eps=1.0e-8)
