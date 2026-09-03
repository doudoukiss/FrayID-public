from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from frayid.carrier_optimization import project_carrier_step
from frayid.config import load_config
from frayid.geometry import (
    canonical_face_orientation_report,
    canonical_topology_is_valid,
    canonical_topology_losses,
    deformation_jacobian,
    eikonal_loss,
    extract_sdf_mesh,
    jacobian_regularization,
    linear_blend_skinning,
    rigid_transform_from_axis_angle,
    safe_root_mean_square,
    signed_sdf_mesh_consistency,
    sphere_sdf,
    subdivide_triangular_mesh,
)


class Sphere(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return sphere_sdf(points, 0.75).unsqueeze(-1)


def test_topology_preserving_subdivision_is_deterministic_and_watertight() -> None:
    import trimesh

    source = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    vertices = torch.tensor(source.vertices, dtype=torch.float32)
    faces = torch.tensor(source.faces, dtype=torch.long)
    weights = torch.rand((len(vertices), 4), generator=torch.Generator().manual_seed(9))
    weights = weights / weights.sum(-1, keepdim=True)
    first = subdivide_triangular_mesh(vertices, faces, {"weights": weights})
    second = subdivide_triangular_mesh(vertices, faces, {"weights": weights})
    for left, right in zip(first[:2], second[:2], strict=True):
        torch.testing.assert_close(left, right)
    subdivided_vertices, subdivided_faces, attributes, unique_edges = first
    assert len(subdivided_faces) == 4 * len(faces)
    assert len(subdivided_vertices) == len(vertices) + len(unique_edges)
    torch.testing.assert_close(attributes["weights"].sum(-1), torch.ones(len(subdivided_vertices)))
    mesh = trimesh.Trimesh(
        vertices=subdivided_vertices.numpy(), faces=subdivided_faces.numpy(), process=False
    )
    assert mesh.is_watertight
    assert mesh.euler_number == source.euler_number == 2


def test_subdivision_preserves_identity_lbs_and_receives_gradient() -> None:
    vertices = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], requires_grad=True)
    faces = torch.tensor([[0, 1, 2]])
    weights = torch.tensor([[0.7, 0.3], [0.7, 0.3], [0.7, 0.3]])
    subdivided, subdivided_faces, attributes, _ = subdivide_triangular_mesh(
        vertices, faces, {"weights": weights}
    )
    transforms = torch.eye(4).repeat(2, 1, 1)
    posed = linear_blend_skinning(subdivided, attributes["weights"], transforms)
    torch.testing.assert_close(posed, subdivided)
    assert subdivided_faces.shape == (4, 3)
    posed.square().sum().backward()
    assert vertices.grad is not None
    assert float(vertices.grad.abs().sum()) > 0


def test_canonical_face_orientation_report_detects_foldover() -> None:
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    preserved = canonical_face_orientation_report(reference, reference + 0.25, faces)
    assert preserved["status"] == "pass"
    folded = reference.copy()
    folded[2, 1] = -1.0
    rejected = canonical_face_orientation_report(reference, folded, faces)
    assert rejected["status"] == "fail"
    assert rejected["flipped_face_fraction"] == pytest.approx(1.0)


def test_canonical_topology_barriers_are_differentiable() -> None:
    reference = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.5, 0.0]],
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]])
    edges = torch.tensor([[0, 1], [0, 2], [1, 2]])
    losses = canonical_topology_losses(
        reference,
        reference + offsets,
        faces,
        edges,
        offsets,
        orientation_margin=0.2,
        minimum_area_ratio=0.1,
    )
    assert float(losses["canonical_orientation"]) > 0.0
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert offsets.grad is not None
    assert bool(torch.isfinite(offsets.grad).all())
    assert not canonical_topology_is_valid(
        reference,
        reference + offsets,
        faces,
        minimum_signed_area_ratio=0.01,
        minimum_area_ratio=0.1,
    )


def test_carrier_projection_backtracks_and_damps_adam_moments() -> None:
    config = load_config()
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    offsets = nn.Parameter(torch.zeros_like(base))
    optimizer = torch.optim.Adam([offsets], lr=1e-3)
    optimizer.state[offsets] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.ones_like(offsets),
        "exp_avg_sq": torch.ones_like(offsets),
    }
    previous = offsets.detach().clone()
    offsets.data[2, 1] = -2.0
    accepted_scale = project_carrier_step(offsets, previous, base, faces, optimizer, config)
    assert accepted_scale == pytest.approx(0.25)
    assert float(offsets[2, 1].detach()) == pytest.approx(-0.5)
    torch.testing.assert_close(optimizer.state[offsets]["exp_avg"], torch.full_like(offsets, 0.25))
    torch.testing.assert_close(
        optimizer.state[offsets]["exp_avg_sq"], torch.full_like(offsets, 0.0625)
    )


def test_safe_root_mean_square_has_finite_zero_gradient() -> None:
    values = torch.zeros((8, 3), requires_grad=True)
    loss = safe_root_mean_square(values)
    loss.backward()
    assert values.grad is not None
    assert bool(torch.isfinite(values.grad).all())


def test_sphere_sdf_sign_and_eikonal_gradient() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = sphere_sdf(points, 0.75)
    assert values[0] < 0
    assert values[1] == pytest.approx(0.0)
    assert values[2] > 0
    samples = torch.randn(512, 3)
    assert float(eikonal_loss(Sphere(), samples)) < 1e-10


def test_marching_cubes_extracts_sphere() -> None:
    mesh = extract_sdf_mesh(Sphere(), resolution=28, bounds=(-1.0, 1.0))
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert mesh.faces.shape[0] > 100
    assert np.median(radii) == pytest.approx(0.75, abs=0.02)


def test_marching_cubes_supports_axis_aligned_bounds() -> None:
    mesh = extract_sdf_mesh(
        Sphere(),
        resolution=30,
        bounds=((-1.0, -1.1, -1.2), (1.0, 1.1, 1.2)),
    )
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert np.median(radii) == pytest.approx(0.75, abs=0.03)


def test_signed_sdf_mesh_consistency_rejects_unsigned_interior() -> None:
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    class UnsignedSphere(nn.Module):
        def forward(self, points: torch.Tensor) -> torch.Tensor:
            return sphere_sdf(points, 0.75).abs().unsqueeze(-1)

    signed_loss = signed_sdf_mesh_consistency(Sphere(), vertices, faces)
    unsigned_loss = signed_sdf_mesh_consistency(UnsignedSphere(), vertices, faces)
    assert float(signed_loss) < 1e-3
    assert float(unsigned_loss) > float(signed_loss) + 1e-2


def test_two_bone_linear_blend_skinning() -> None:
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    transforms = torch.eye(4).repeat(2, 1, 1)
    transforms[0, 0, 3] = 1.0
    transforms[1, 1, 3] = 2.0
    result = linear_blend_skinning(vertices, weights, transforms)
    expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
    assert torch.allclose(result, expected)


def test_axis_angle_se3_recovers_known_rigid_transform() -> None:
    transform = rigid_transform_from_axis_angle(
        torch.tensor([0.0, 0.0, torch.pi / 2]),
        torch.tensor([1.0, 2.0, 3.0]),
    )
    point = torch.tensor([1.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(transform @ point, torch.tensor([1.0, 3.0, 3.0, 1.0]))


def test_axis_angle_se3_is_optimizable_from_point_correspondences() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    target = rigid_transform_from_axis_angle(
        torch.tensor([0.03, -0.02, 0.04]),
        torch.tensor([0.02, -0.01, 0.03]),
    ).detach()
    target_points = (target @ points.T).T
    rotation = torch.zeros(3, requires_grad=True)
    translation = torch.zeros(3, requires_grad=True)
    optimizer = torch.optim.Adam((rotation, translation), lr=0.03)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        predicted = (rigid_transform_from_axis_angle(rotation, translation) @ points.T).T
        loss = (predicted - target_points).square().mean()
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(rotation, torch.tensor([0.03, -0.02, 0.04]), atol=2e-4, rtol=0)
    torch.testing.assert_close(translation, torch.tensor([0.02, -0.01, 0.03]), atol=2e-4, rtol=0)


def test_deformation_jacobian_and_regularizer_are_differentiable() -> None:
    points = torch.randn(8, 3, requires_grad=True)
    scale = torch.tensor([0.1, -0.2, 0.05], requires_grad=True)
    displacement = points * scale
    jacobian = deformation_jacobian(displacement, points)
    expected = torch.diag(torch.tensor([0.1, -0.2, 0.05])).expand(8, -1, -1)
    assert torch.allclose(jacobian, expected)
    penalty = jacobian_regularization(jacobian)
    assert torch.isfinite(penalty)
    penalty.backward()
    assert scale.grad is not None
