from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from frayid.camera import make_intrinsics, yaw_matrix
from frayid.normal_integrable_sdf import (
    MultiresolutionHashEncoding,
    NeuSRender,
    NormalIntegrableNeuralSDF,
    VisualHullGrid,
    bilateral_normal_integrability_loss,
    camera_rays,
    directional_normal_error_degrees,
    eikonal_loss,
    hierarchical_depth_samples,
    neus_interval_weights,
    normal_integrable_image_loss,
    render_neus_sdf,
    spatial_sdf_gradient,
    transport_normals_inverse_transpose,
    trilinear_grid_sample,
    visual_hull_sdf,
)


def test_trilinear_grid_sample_is_exact_for_affine_fields() -> None:
    coordinates = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    values = 2.0 * xx - 3.0 * yy + 0.5 * zz + 0.7
    points = torch.tensor(
        [[-0.82, 0.14, 0.71], [0.0, 0.0, 0.0], [0.93, -0.76, -0.22]],
        dtype=torch.float64,
    )
    expected = 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.5 * points[:, 2] + 0.7
    observed = trilinear_grid_sample(values, points, extent=1.0)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=1.0e-12)


def test_visual_hull_builds_a_bounded_negative_chamber() -> None:
    image_size = 64
    views = 4
    intrinsics = make_intrinsics(42.0, (32.0, 32.0), dtype=torch.float64)
    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float64) + 0.5,
        torch.arange(image_size, dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    radius_pixels = 42.0 * 0.55 / 3.0
    circle = (((xx - 32.0) ** 2 + (yy - 32.0) ** 2) <= radius_pixels**2).to(torch.float64)
    silhouettes = circle.repeat(views, 1, 1)
    rotations = torch.stack([yaw_matrix(view * 90.0, dtype=torch.float64) for view in range(views)])
    translations = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float64).repeat(views, 1)
    hull = visual_hull_sdf(
        silhouettes,
        intrinsics,
        rotations,
        translations,
        resolution=17,
        extent=1.2,
    )
    assert hull.values.shape == (17, 17, 17)
    assert hull.values[8, 8, 8] < 0
    boundary = torch.cat(
        (
            hull.values[0].reshape(-1),
            hull.values[-1].reshape(-1),
            hull.values[:, 0].reshape(-1),
            hull.values[:, -1].reshape(-1),
            hull.values[:, :, 0].reshape(-1),
            hull.values[:, :, -1].reshape(-1),
        )
    )
    assert torch.all(boundary > 0)


def test_hash_encoding_repeats_and_has_coordinate_and_parameter_gradients() -> None:
    first = MultiresolutionHashEncoding(
        level_count=3,
        features_per_level=2,
        base_resolution=4,
        maximum_resolution=12,
        table_size=128,
        seed=91,
    )
    second = MultiresolutionHashEncoding(
        level_count=3,
        features_per_level=2,
        base_resolution=4,
        maximum_resolution=12,
        table_size=128,
        seed=91,
    )
    points = torch.tensor([[0.21, 0.42, 0.63], [0.77, 0.31, 0.54]], requires_grad=True)
    encoded = first(points)
    torch.testing.assert_close(encoded, second(points), rtol=0.0, atol=0.0)
    encoded.square().sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert points.grad.abs().sum() > 0
    assert first.tables.grad is not None and first.tables.grad.abs().sum() > 0


def test_neural_sdf_starts_at_visual_hull_and_is_trainable() -> None:
    coordinates = torch.linspace(-1.0, 1.0, 13)
    xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    base = torch.sqrt(xx.square() + yy.square() + zz.square()) - 0.55
    model = NormalIntegrableNeuralSDF(
        VisualHullGrid(base, 1.0), hidden_width=16, hidden_layers=2, maximum_hash_resolution=24
    )
    points = torch.tensor([[0.1, 0.2, 0.3], [0.7, -0.1, 0.0]], requires_grad=True)
    expected = trilinear_grid_sample(base, points, extent=1.0)
    torch.testing.assert_close(model(points), expected)
    gradient = spatial_sdf_gradient(model, points, create_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    loss = model(points).square().mean() + 0.01 * eikonal_loss(model, points)
    loss.backward()
    final_linear = model.residual[-1]
    assert isinstance(final_linear, nn.Linear)
    assert final_linear.weight.grad is not None
    assert final_linear.weight.grad.abs().sum() > 0


def test_inverse_transpose_normal_transport_preserves_tangent_orthogonality() -> None:
    jacobian = torch.tensor(
        [[1.2, 0.3, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]],
        dtype=torch.float64,
    )
    normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    tangent = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    transported_normal = transport_normals_inverse_transpose(normal, jacobian)
    transported_tangent = jacobian @ tangent
    assert abs(float(transported_normal @ transported_tangent)) < 1.0e-12
    torch.testing.assert_close(
        torch.linalg.vector_norm(transported_normal),
        torch.tensor(1.0, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="singular"):
        transport_normals_inverse_transpose(normal, torch.zeros(3, 3, dtype=torch.float64))


def test_neus_weights_select_the_first_positive_to_negative_crossing() -> None:
    sdf = torch.tensor([[0.8, 0.2, -0.2, -0.7, 0.1, 0.8]])
    weights = neus_interval_weights(sdf, 40.0)
    assert weights.shape == (1, 5)
    assert int(weights.argmax(dim=-1)) == 1
    assert float(weights.sum()) <= 1.0 + 1.0e-6
    assert weights[0, 3] < 1.0e-6


def test_hierarchical_samples_concentrate_in_the_dominant_interval() -> None:
    depths = torch.linspace(0.0, 1.0, 6)
    weights = torch.tensor([[0.01, 0.01, 0.94, 0.02, 0.02]])
    samples = hierarchical_depth_samples(depths, weights, sample_count=32)
    assert samples.shape == (1, 32)
    assert torch.all(samples[:, 1:] >= samples[:, :-1])
    assert float(((samples >= 0.4) & (samples <= 0.6)).float().mean()) > 0.85


class _SphereField(nn.Module):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = nn.Parameter(torch.tensor(radius))

    def forward(self, points: Tensor) -> Tensor:
        return torch.linalg.vector_norm(points, dim=-1) - self.radius


def test_neus_renderer_uses_continuous_gradient_normals_and_backpropagates() -> None:
    intrinsics = make_intrinsics(20.0, (8.0, 8.0))
    origins, directions = camera_rays(
        intrinsics,
        torch.eye(3),
        torch.tensor([0.0, 0.0, 2.0]),
        (16, 16),
    )
    field = _SphereField(0.55)
    rendered: NeuSRender = render_neus_sdf(
        field,
        origins,
        directions,
        near=0.5,
        far=3.5,
        sample_count=64,
        hierarchical_sample_count=32,
        inverse_sharpness=80.0,
        deformation_jacobian=torch.eye(3),
        create_graph=True,
        ray_chunk_size=64,
    )
    assert rendered.silhouette[8, 8] > 0.9
    assert rendered.silhouette[0, 0] < 0.1
    assert rendered.normals[8, 8, 2] > 0.9
    assert torch.isfinite(rendered.normals).all()
    rendered.silhouette.mean().backward()
    assert field.radius.grad is not None and abs(float(field.radius.grad)) > 0


def test_registered_image_loss_is_finite_and_uses_normal_integrability() -> None:
    silhouette = torch.full((5, 6), 0.8, requires_grad=True)
    normal = torch.zeros(5, 6, 3)
    normal[..., 2] = 1.0
    normal.requires_grad_()
    rendered = NeuSRender(silhouette, normal, torch.zeros(5, 6), silhouette)
    target_silhouette = torch.ones(5, 6)
    target_normal = normal.detach().clone()
    target_normal[:, 3:, 0] = 1.0
    target_normal[:, 3:, 2] = 0.0
    loss = normal_integrable_image_loss(rendered, target_silhouette, target_normal)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert silhouette.grad is not None and silhouette.grad.abs().sum() > 0
    assert normal.grad is not None and normal.grad.abs().sum() > 0


def test_bilateral_integrability_matches_local_observed_normal_changes() -> None:
    observed = torch.zeros(6, 7, 3)
    observed[..., 2] = 1.0
    observed[:, 4:, 0] = 1.0
    observed[:, 4:, 2] = 0.0
    mask = torch.ones(6, 7)
    exact = bilateral_normal_integrability_loss(observed, observed, mask)
    assert exact == 0
    perturbed = observed.clone()
    perturbed[2:5, 1:3] = torch.tensor([0.0, 1.0, 0.0])
    loss = bilateral_normal_integrability_loss(perturbed, observed, mask)
    assert torch.isfinite(loss) and loss > 0
    errors = directional_normal_error_degrees(perturbed, observed, mask)
    assert errors.shape == (42,)
    assert math.isclose(float(errors.max()), 90.0, abs_tol=1.0e-4)
