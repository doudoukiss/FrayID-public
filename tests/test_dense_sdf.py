from __future__ import annotations

import numpy as np
import pytest
import torch
import trimesh

from frayid.dense_sdf import (
    LearnableDenseSDF,
    dense_eikonal_loss,
    load_dense_sdf_checkpoint,
    project_points_to_zero_level,
    save_dense_sdf_checkpoint,
    sign_anchor_loss,
    smooth_sign_anchor_loss,
)
from frayid.sdf_distillation import extract_voxel_sdf_mesh


def _sphere_grid(resolution: int = 32) -> tuple[torch.Tensor, torch.Tensor, float]:
    origin = torch.tensor([-1.25, -1.25, -1.25])
    pitch = 2.5 / (resolution - 1)
    axes = [origin[index] + torch.arange(resolution) * pitch for index in range(3)]
    xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
    return torch.sqrt(xx.square() + yy.square() + zz.square()) - 0.8, origin, pitch


def test_dense_sdf_axis_contract_values_gradients_and_eikonal() -> None:
    values, origin, pitch = _sphere_grid()
    field = LearnableDenseSDF(values, origin, pitch)
    points = torch.tensor([[0.8, 0.0, 0.0], [0.0, -0.8, 0.0]], requires_grad=True)
    sampled = field(points)
    assert float(sampled.abs().max()) < 0.005
    gradients = field.spatial_gradient(points, create_graph=True)
    torch.testing.assert_close(gradients[0], torch.tensor([1.0, 0.0, 0.0]), atol=0.01, rtol=0.01)
    torch.testing.assert_close(gradients[1], torch.tensor([0.0, -1.0, 0.0]), atol=0.01, rtol=0.01)
    assert float(dense_eikonal_loss(field, points)) < 0.001
    dense_eikonal_loss(field, points).backward()
    assert field.values_zyx.grad is not None
    assert torch.isfinite(field.values_zyx.grad).all()


def test_dense_sdf_sign_anchors_and_checkpoint_resume(tmp_path) -> None:
    values, origin, pitch = _sphere_grid(20)
    field = LearnableDenseSDF(values + 1.0, origin, pitch)
    optimizer = torch.optim.Adam(field.parameters(), lr=0.03)
    inside = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]])
    outside = torch.tensor([[1.1, 0.0, 0.0], [0.0, -1.1, 0.0]])
    initial = float(sign_anchor_loss(field, inside, outside, margin=0.1))
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = sign_anchor_loss(field, inside, outside, margin=0.1) + 1e-4 * field.total_variation()
        loss.backward()
        optimizer.step()
    assert float(sign_anchor_loss(field, inside, outside, margin=0.1)) < initial * 0.1
    checkpoint = tmp_path / "dense_sdf.pt"
    save_dense_sdf_checkpoint(checkpoint, field, optimizer, 19)
    restored = LearnableDenseSDF(values, origin, pitch)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.03)
    assert load_dense_sdf_checkpoint(checkpoint, restored, restored_optimizer) == 19
    torch.testing.assert_close(restored.values_xyz, field.values_xyz)


def test_zero_level_projection_moves_points_and_reaches_grid_gradient() -> None:
    values, origin, pitch = _sphere_grid(24)
    field = LearnableDenseSDF(values, origin, pitch)
    points = torch.tensor([[0.95, 0.0, 0.0], [0.0, -0.95, 0.0]])
    projected = project_points_to_zero_level(field, points, iteration_count=3)
    assert float(field(projected).abs().max()) < float(field(points).abs().max())
    projected.square().sum().backward()
    assert field.values_zyx.grad is not None
    assert float(field.values_zyx.grad.abs().sum()) > 0


def test_smooth_sign_anchor_remains_active_when_margin_is_satisfied() -> None:
    values, origin, pitch = _sphere_grid(20)
    field = LearnableDenseSDF(values, origin, pitch)
    loss = smooth_sign_anchor_loss(
        field,
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.1, 0.0, 0.0]]),
        margin=pitch,
        temperature=pitch,
    )
    assert 0 < float(loss) < pitch**2
    loss.backward()
    assert field.values_zyx.grad is not None
    assert float(field.values_zyx.grad.abs().sum()) > 0


@pytest.mark.parametrize("scales", [(1.0, 1.0, 1.0), (0.65, 1.0, 0.45)])
def test_dense_sdf_extracts_watertight_sphere_and_rotated_ellipsoid(scales) -> None:
    resolution = 40
    origin = np.array([-1.4, -1.4, -1.4], dtype=np.float32)
    pitch = 2.8 / (resolution - 1)
    axes = [origin[index] + np.arange(resolution) * pitch for index in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    angle = np.deg2rad(31.0)
    xr = np.cos(angle) * xx + np.sin(angle) * zz
    zr = -np.sin(angle) * xx + np.cos(angle) * zz
    sx, sy, sz = scales
    sdf = np.sqrt((xr / sx) ** 2 + (yy / sy) ** 2 + (zr / sz) ** 2) - 0.8
    mesh = extract_voxel_sdf_mesh(sdf.astype(np.float32), origin, pitch)
    assert mesh.is_watertight
    assert len(mesh.split(only_watertight=False)) == 1
    assert trimesh.repair.broken_faces(mesh).size == 0
