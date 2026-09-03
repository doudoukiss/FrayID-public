from __future__ import annotations

import pytest
import torch

from frayid.camera import make_intrinsics
from frayid.initialization import pose_shared_displacements, shared_normal_envelope
from frayid.renderer import (
    differentiable_boundary_loss,
    render_soft_mesh,
    scaled_splat_parameters,
    silhouette_loss,
)


def test_differentiable_boundary_loss_supports_batches() -> None:
    prediction = torch.zeros((2, 8, 8), requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 2:6, 2:6] = 1.0
    batched = differentiable_boundary_loss(prediction, target)
    separate = torch.stack(
        [differentiable_boundary_loss(prediction[index], target[index]) for index in range(2)]
    ).mean()
    torch.testing.assert_close(batched, separate)
    batched.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_differentiable_rendering_reaches_geometry() -> None:
    vertices = torch.tensor(
        [[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.6, 2.0], [0.0, 0.0, 2.7]],
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    intrinsics = make_intrinsics(45.0, (16.0, 16.0))
    torch.manual_seed(7)
    silhouette, normals = render_soft_mesh(
        vertices,
        faces,
        intrinsics,
        (32, 32),
        source_image_size=(32, 32),
        sample_count=96,
        sigma_pixels=1.4,
    )
    target = torch.zeros_like(silhouette)
    target[8:25, 8:25] = 1.0
    loss = silhouette_loss(silhouette, target) + normals.square().mean() * 1e-3
    loss.backward()
    assert vertices.grad is not None
    assert float(vertices.grad.abs().sum()) > 0


def test_visible_normal_uses_front_surface_and_sapiens_axes() -> None:
    vertices = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.0, 0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.75, -0.75, 3.0],
            [0.75, -0.75, 3.0],
            [0.0, 0.75, 3.0],
        ]
    )
    # The front face has OpenCV normal -z, which is Sapiens2 normal +z.
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    torch.manual_seed(3)
    _, normals = render_soft_mesh(
        vertices,
        faces,
        make_intrinsics(40.0, (16.0, 16.0)),
        (32, 32),
        source_image_size=(32, 32),
        sample_count=1024,
        sigma_pixels=1.5,
        depth_temperature_m=0.05,
    )
    assert float(normals[16, 16, 2]) > 0.9


def test_splat_parameters_preserve_normalized_scale() -> None:
    assert scaled_splat_parameters(
        (256, 256),
        reference_resolution=128,
        reference_sigma_pixels=1.75,
        reference_sample_count=2048,
    ) == (3.5, 8192)


def test_surface_splat_opacity_converges_with_sample_count() -> None:
    vertices = torch.tensor([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.6, 2.0], [0.0, 0.0, 2.7]])
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    intrinsics = make_intrinsics(45.0, (32.0, 32.0))
    outputs = []
    for sample_count in (2048, 4096, 8192):
        torch.manual_seed(7)
        outputs.append(
            render_soft_mesh(
                vertices,
                faces,
                intrinsics,
                (64, 64),
                source_image_size=(64, 64),
                sample_count=sample_count,
                reference_sample_count=2048,
                sigma_pixels=1.4,
            )[0]
        )
    assert float((outputs[-1] - outputs[-2]).abs().mean()) < 0.003
    assert float(((outputs[-1] > 0.5) != (outputs[-2] > 0.5)).float().mean()) < 0.002


def test_shared_normal_envelope_is_bounded_and_uses_lbs_rotations() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]])
    logits = torch.tensor([10.0, -10.0, 0.0], requires_grad=True)
    envelope, offsets = shared_normal_envelope(vertices, faces, logits, 0.12)
    assert float(offsets.abs().max()) <= 0.12
    envelope.sum().backward()
    assert logits.grad is not None

    displacement = torch.tensor([[1.0, 0.0, 0.0]])
    weights = torch.tensor([[1.0]])
    transform = torch.eye(4).reshape(1, 4, 4)
    transform[0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    posed = pose_shared_displacements(displacement, weights, transform)
    torch.testing.assert_close(posed, torch.tensor([[0.0, 1.0, 0.0]]))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_rotating_clothed_ellipsoid_end_to_end(device: str) -> None:
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    base = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    base = base * base.new_tensor([0.55, 0.95, 0.35])
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    intrinsics = make_intrinsics(30.0, (12.0, 12.0), device=device)
    angles = torch.deg2rad(torch.tensor([0.0, 70.0, 145.0], device=device))
    transforms = []
    for angle in angles:
        cosine, sine = torch.cos(angle), torch.sin(angle)
        transform = torch.eye(4, device=device)
        transform[:3, :3] = torch.stack(
            (
                torch.stack((cosine, torch.zeros_like(angle), sine)),
                torch.tensor([0.0, 1.0, 0.0], device=device),
                torch.stack((-sine, torch.zeros_like(angle), cosine)),
            )
        )
        transform[2, 3] = 3.0
        transforms.append(transform)
    targets = []
    for index, transform in enumerate(transforms):
        homogeneous = torch.cat(
            (base * 1.15, torch.ones((base.shape[0], 1), device=device)), dim=-1
        )
        posed = (transform @ homogeneous.T).T[:, :3]
        torch.manual_seed(100 + index)
        targets.append(
            render_soft_mesh(
                posed,
                faces,
                intrinsics,
                (24, 24),
                source_image_size=(24, 24),
                sample_count=96,
            )[0].detach()
        )
    log_scale = torch.nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.Adam([log_scale], lr=0.08)
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = torch.zeros((), device=device)
        for index, (target, transform) in enumerate(zip(targets, transforms, strict=True)):
            homogeneous = torch.cat(
                (base * log_scale.exp(), torch.ones((base.shape[0], 1), device=device)), dim=-1
            )
            posed = (transform @ homogeneous.T).T[:, :3]
            torch.manual_seed(100 + index)
            prediction = render_soft_mesh(
                posed,
                faces,
                intrinsics,
                (24, 24),
                source_image_size=(24, 24),
                sample_count=96,
            )[0]
            loss = loss + silhouette_loss(prediction, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert losses[-1] < losses[0]
    assert float(log_scale.exp().detach().cpu()) == pytest.approx(1.15, abs=0.12)
