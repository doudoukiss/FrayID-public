from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from torch import Tensor

from frayid.camera import make_intrinsics
from frayid.differentiable_isosurface import (
    SurfacePathCertificate,
    certify_linear_surface_path,
    surface_endpoint_area_ratios,
)
from frayid.embedded_carrier import embedded_surface_fidelity
from frayid.eulerian_field import EulerianImageField
from frayid.hybrid_tetrahedral import regular_tetrahedral_grid
from frayid.renderer import (
    differentiable_boundary_loss,
    normal_cosine_loss,
    normalized_boundary_error,
    render_soft_mesh,
    silhouette_loss,
    soft_silhouette_iou,
)

PUBLIC_GRID_RESOLUTION = 12
PUBLIC_BOX_EXTENT = 1.25
PUBLIC_IMAGE_SIZE = (128, 128)
PUBLIC_VIEW_COUNT = 18
PUBLIC_TRAIN_VIEW_COUNT = 12
PUBLIC_HELD_OUT_VIEW_COUNT = 6
PUBLIC_FOCAL_PIXELS = 115.0
PUBLIC_CAMERA_DISTANCE = 3.2
PUBLIC_TARGET_SAMPLE_COUNT = 2048
PUBLIC_TRAIN_SAMPLE_COUNT = 256
PUBLIC_REFERENCE_SAMPLE_COUNT = 2048
PUBLIC_EVALUATION_SAMPLE_COUNT = 2048


@dataclass(frozen=True)
class PublicEulerianFixture:
    positions: Tensor
    tetrahedra: Tensor
    target_values: Tensor
    initial_values: Tensor
    pitch: float
    exterior_probes: np.ndarray
    interior_probes: np.ndarray

    def target_field(self) -> EulerianImageField:
        return EulerianImageField(
            self.positions.clone(),
            self.tetrahedra.clone(),
            self.target_values.clone(),
            maximum_absolute_value=8.0,
        )

    def initial_field(self) -> EulerianImageField:
        return EulerianImageField(
            self.positions.clone(),
            self.tetrahedra.clone(),
            self.initial_values.clone(),
            maximum_absolute_value=8.0,
        )


@dataclass(frozen=True)
class PublicImageEvidence:
    silhouettes: tuple[Tensor, ...]
    normals: tuple[Tensor, ...]
    intrinsics: Tensor


@dataclass(frozen=True)
class ExplicitStepResult:
    accepted_scale: float
    certificate: SurfacePathCertificate
    rejected: bool


def public_eulerian_fixture() -> PublicEulerianFixture:
    positions, tetrahedra = regular_tetrahedral_grid(
        PUBLIC_GRID_RESOLUTION, extent=PUBLIC_BOX_EXTENT
    )
    x, y, z = positions.T
    main = torch.sqrt((x / 0.66) ** 2 + (y / 0.90) ** 2 + (z / 0.48) ** 2) - 1.0
    lobe = torch.sqrt(((x - 0.76) / 0.30) ** 2 + ((y - 0.16) / 0.42) ** 2 + (z / 0.24) ** 2) - 1.0
    bridge = (
        torch.maximum(
            torch.maximum(torch.abs(x - 0.48) / 0.34, torch.abs(y + 0.26) / 0.12),
            torch.abs(z) / 0.14,
        )
        - 1.0
    )
    cutter = (
        torch.sqrt(((x - 0.48) / 0.30) ** 2 + ((y - 0.22) / 0.32) ** 2 + ((z - 0.34) / 0.28) ** 2)
        - 1.0
    )
    target = torch.maximum(torch.minimum(torch.minimum(main, lobe), bridge), -cutter)
    if torch.any(target == 0):
        raise RuntimeError("public target unexpectedly crosses a grid vertex")

    # The initialization uses only the public sign template and vertex ordinal,
    # never target magnitudes or evaluator geometry. This freezes an in-chamber
    # but deliberately biased starting surface.
    phase = torch.arange(target.numel(), dtype=target.dtype)
    variation = 1.0 + 0.12 * torch.sin(phase * (1.0 + math.sqrt(5.0)) / 2.0)
    initial_magnitudes = torch.where(target < 0, 0.04 * variation, 0.80 * variation)
    initial = torch.sign(target) * initial_magnitudes
    return PublicEulerianFixture(
        positions=positions,
        tetrahedra=tetrahedra,
        target_values=target,
        initial_values=initial,
        pitch=2.0 * PUBLIC_BOX_EXTENT / (PUBLIC_GRID_RESOLUTION - 1),
        exterior_probes=np.asarray(
            [
                [0.48, 0.22, 0.34],
                [0.48, 0.22, 0.15],
                [0.48, -0.05, 0.22],
                [0.52, 0.05, 0.28],
                [0.62, 0.10, 0.30],
            ],
            dtype=np.float64,
        ),
        interior_probes=np.asarray(
            [[0.75, 0.16, 0.0], [0.0, 0.0, 0.0], [0.48, -0.26, 0.0]],
            dtype=np.float64,
        ),
    )


def pose_public_view(vertices: Tensor, view: int) -> Tensor:
    if view < 0 or view >= PUBLIC_VIEW_COUNT:
        raise ValueError("public view ordinal is out of range")
    yaw = vertices.new_tensor(2.0 * math.pi * view / PUBLIC_VIEW_COUNT)
    pitch = vertices.new_tensor(math.radians((-12.0, 0.0, 12.0)[view % 3]))
    cosine_yaw, sine_yaw = torch.cos(yaw), torch.sin(yaw)
    cosine_pitch, sine_pitch = torch.cos(pitch), torch.sin(pitch)
    zero = torch.zeros_like(yaw)
    one = torch.ones_like(yaw)
    rotate_y = torch.stack(
        (
            torch.stack((cosine_yaw, zero, sine_yaw)),
            torch.stack((zero, one, zero)),
            torch.stack((-sine_yaw, zero, cosine_yaw)),
        )
    )
    rotate_x = torch.stack(
        (
            torch.stack((one, zero, zero)),
            torch.stack((zero, cosine_pitch, -sine_pitch)),
            torch.stack((zero, sine_pitch, cosine_pitch)),
        )
    )
    return vertices @ (rotate_x @ rotate_y).T + vertices.new_tensor(
        [0.0, 0.0, PUBLIC_CAMERA_DISTANCE]
    )


def public_intrinsics(*, dtype: torch.dtype = torch.float32) -> Tensor:
    return make_intrinsics(PUBLIC_FOCAL_PIXELS, (64.0, 64.0)).to(dtype=dtype)


def render_public_evidence(vertices: Tensor, faces: Tensor) -> PublicImageEvidence:
    intrinsics = public_intrinsics(dtype=vertices.dtype)
    silhouettes: list[Tensor] = []
    normals: list[Tensor] = []
    with torch.no_grad():
        for view in range(PUBLIC_VIEW_COUNT):
            torch.manual_seed(10_000 + view)
            silhouette, normal = render_soft_mesh(
                pose_public_view(vertices, view),
                faces,
                intrinsics,
                PUBLIC_IMAGE_SIZE,
                source_image_size=PUBLIC_IMAGE_SIZE,
                sample_count=PUBLIC_TARGET_SAMPLE_COUNT,
                reference_sample_count=PUBLIC_REFERENCE_SAMPLE_COUNT,
                chunk_size=256,
            )
            silhouettes.append(silhouette.detach())
            normals.append(normal.detach())
    return PublicImageEvidence(tuple(silhouettes), tuple(normals), intrinsics)


def public_image_loss(
    vertices: Tensor,
    faces: Tensor,
    evidence: PublicImageEvidence,
    view: int,
    *,
    seed: int,
) -> Tensor:
    torch.manual_seed(seed)
    silhouette, normals = render_soft_mesh(
        pose_public_view(vertices, view),
        faces,
        evidence.intrinsics,
        PUBLIC_IMAGE_SIZE,
        source_image_size=PUBLIC_IMAGE_SIZE,
        sample_count=PUBLIC_TRAIN_SAMPLE_COUNT,
        reference_sample_count=PUBLIC_REFERENCE_SAMPLE_COUNT,
        chunk_size=128,
    )
    target_silhouette = evidence.silhouettes[view]
    return (
        silhouette_loss(silhouette, target_silhouette)
        + 0.5 * differentiable_boundary_loss(silhouette, target_silhouette)
        + 0.25 * normal_cosine_loss(normals, evidence.normals[view], target_silhouette)
    )


def evaluate_public_images(
    vertices: Tensor,
    faces: Tensor,
    evidence: PublicImageEvidence,
) -> dict[str, float]:
    ious: list[float] = []
    boundaries: list[float] = []
    normal_errors: list[float] = []
    with torch.no_grad():
        for view in range(PUBLIC_VIEW_COUNT):
            torch.manual_seed(20_000 + view)
            silhouette, normals = render_soft_mesh(
                pose_public_view(vertices, view),
                faces,
                evidence.intrinsics,
                PUBLIC_IMAGE_SIZE,
                source_image_size=PUBLIC_IMAGE_SIZE,
                sample_count=PUBLIC_EVALUATION_SAMPLE_COUNT,
                reference_sample_count=PUBLIC_REFERENCE_SAMPLE_COUNT,
                chunk_size=256,
            )
            target_silhouette = evidence.silhouettes[view]
            ious.append(float(soft_silhouette_iou(silhouette, target_silhouette)))
            boundaries.append(normalized_boundary_error(silhouette, target_silhouette))
            valid = target_silhouette > 0.5
            cosine = (normals[valid] * evidence.normals[view][valid]).sum(dim=-1).clamp(-1.0, 1.0)
            normal_errors.extend(torch.rad2deg(torch.acos(cosine)).cpu().tolist())
    train_iou = float(np.median(ious[:PUBLIC_TRAIN_VIEW_COUNT]))
    held_out_iou = float(np.median(ious[PUBLIC_TRAIN_VIEW_COUNT:]))
    return {
        "train_silhouette_iou": train_iou,
        "held_out_silhouette_iou": held_out_iou,
        "normalized_boundary_error": float(np.median(boundaries[PUBLIC_TRAIN_VIEW_COUNT:])),
        "pooled_normal_error_degrees": float(np.median(normal_errors)),
        "signed_train_held_out_gap": train_iou - held_out_iou,
    }


def geometry_fidelity(
    target_vertices: Tensor,
    candidate_vertices: Tensor,
    faces: Tensor,
    *,
    pitch: float,
    sample_count: int = 20_000,
) -> dict[str, object]:
    target = trimesh.Trimesh(
        vertices=target_vertices.detach().cpu().double().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    candidate = trimesh.Trimesh(
        vertices=candidate_vertices.detach().cpu().double().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    report = embedded_surface_fidelity(
        target,
        candidate,
        pitch=pitch,
        sample_count=sample_count,
        seed=20260902,
        maximum_relative_volume_error=0.031,
    )
    first = report["source_to_target"]
    second = report["target_to_source"]
    assert isinstance(first, dict) and isinstance(second, dict)
    report["bidirectional_mean_distance_pitch"] = 0.5 * (
        float(first["mean_distance_pitch"]) + float(second["mean_distance_pitch"])
    )
    return report


def probe_classification(
    vertices: Tensor, faces: Tensor, fixture: PublicEulerianFixture
) -> dict[str, object]:
    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().double().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    exterior_inside = np.asarray(mesh.contains(fixture.exterior_probes), dtype=np.bool_)
    interior_inside = np.asarray(mesh.contains(fixture.interior_probes), dtype=np.bool_)
    return {
        "exterior_probe_count": len(exterior_inside),
        "exterior_inside_count": int(np.count_nonzero(exterior_inside)),
        "interior_probe_count": len(interior_inside),
        "interior_inside_count": int(np.count_nonzero(interior_inside)),
        "status": (
            "pass" if not np.any(exterior_inside) and bool(np.all(interior_inside)) else "fail"
        ),
    }


def project_explicit_step(
    vertices: Tensor,
    previous_vertices: Tensor,
    reference_vertices: Tensor,
    faces: Tensor,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    signed_area_floor: float = 0.01,
    unsigned_area_floor: float = 0.10,
    maximum_backtracks: int = 32,
) -> ExplicitStepResult:
    candidate = vertices.detach().clone()
    delta = candidate - previous_vertices
    last = SurfacePathCertificate(
        "unknown",
        signed_area_floor,
        unsigned_area_floor,
        -np.inf,
        0.0,
        0,
        1,
        "candidate_not_checked",
    )
    scale = 1.0
    for _ in range(maximum_backtracks + 1):
        endpoint = previous_vertices + scale * delta
        minimum_signed, minimum_unsigned = surface_endpoint_area_ratios(
            endpoint.detach().cpu().double().numpy(),
            reference_vertices.detach().cpu().double().numpy(),
            faces.detach().cpu().numpy(),
        )
        if minimum_signed < signed_area_floor or minimum_unsigned < unsigned_area_floor:
            last = SurfacePathCertificate(
                "fail",
                signed_area_floor,
                unsigned_area_floor,
                minimum_signed,
                minimum_unsigned,
                0,
                1,
                "endpoint_area_floor",
            )
            scale *= 0.5
            continue
        last = certify_linear_surface_path(
            previous_vertices.detach().cpu().double().numpy(),
            endpoint.detach().cpu().double().numpy(),
            faces.detach().cpu().numpy(),
            reference_vertices.detach().cpu().double().numpy(),
            signed_area_floor=signed_area_floor,
            unsigned_area_floor=unsigned_area_floor,
        )
        if last.status == "pass":
            with torch.no_grad():
                vertices.copy_(endpoint)
            if optimizer is not None and scale < 1.0:
                _damp_optimizer_state(optimizer, vertices, scale)
            return ExplicitStepResult(scale, last, False)
        scale *= 0.5
    with torch.no_grad():
        vertices.copy_(previous_vertices)
    if optimizer is not None:
        _damp_optimizer_state(optimizer, vertices, 0.0)
    return ExplicitStepResult(0.0, last, True)


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
