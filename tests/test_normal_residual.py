from __future__ import annotations

from pathlib import Path

import torch
import trimesh

from frayid.geometry import canonical_face_orientation_report
from frayid.normal_residual import DiffusedNormalResidual, project_normal_residual_step


def _sphere() -> tuple[torch.Tensor, torch.Tensor]:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    return (
        torch.tensor(mesh.vertices, dtype=torch.float32),
        torch.tensor(mesh.faces, dtype=torch.long),
    )


def test_constant_normal_offset_is_preserved_and_differentiable() -> None:
    vertices, faces = _sphere()
    field = DiffusedNormalResidual(vertices, faces, diffusion_steps=4)
    with torch.no_grad():
        field.raw_displacements.fill_(0.08)
    torch.testing.assert_close(
        field.diffused_displacements(),
        torch.full_like(field.raw_displacements, 0.08),
    )
    expected = vertices + 0.08 * field.reference_normals
    torch.testing.assert_close(field.deformed_vertices(), expected)
    field.deformed_vertices().square().sum().backward()
    assert field.raw_displacements.grad is not None
    assert bool(torch.isfinite(field.raw_displacements.grad).all())


def test_one_ring_diffusion_spreads_an_impulse_deterministically() -> None:
    vertices, faces = _sphere()
    field = DiffusedNormalResidual(
        vertices,
        faces,
        diffusion_steps=1,
        diffusion_weight=1.0,
    )
    with torch.no_grad():
        field.raw_displacements[0] = 1.0
    first = field.diffused_displacements()
    repeated = field.diffused_displacements()
    neighbors = field.adjacency_targets[field.adjacency_sources == 0]
    assert float(first[0]) == 0.0
    assert bool(torch.all(first[neighbors] > 0))
    assert int(torch.count_nonzero(first)) == len(neighbors)
    torch.testing.assert_close(first, repeated)


def test_normal_residual_projection_backtracks_and_damps_adam() -> None:
    vertices, faces = _sphere()
    field = DiffusedNormalResidual(vertices, faces, diffusion_steps=1, diffusion_weight=0.25)
    optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
    optimizer.state[field.raw_displacements] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.ones_like(field.raw_displacements),
        "exp_avg_sq": torch.ones_like(field.raw_displacements),
    }
    previous = field.raw_displacements.detach().clone()
    with torch.no_grad():
        field.raw_displacements[0] = -2.0
    scale = project_normal_residual_step(field, previous, vertices, faces, optimizer)
    assert 0.0 < scale < 1.0
    torch.testing.assert_close(
        optimizer.state[field.raw_displacements]["exp_avg"],
        torch.full_like(field.raw_displacements, scale),
    )
    torch.testing.assert_close(
        optimizer.state[field.raw_displacements]["exp_avg_sq"],
        torch.full_like(field.raw_displacements, scale * scale),
    )


def test_normal_residual_checkpoint_resume_is_exact(tmp_path: Path) -> None:
    vertices, faces = _sphere()
    field = DiffusedNormalResidual(vertices, faces, diffusion_steps=3)
    optimizer = torch.optim.Adam(field.parameters(), lr=2e-4)
    loss = (field.deformed_vertices() - 1.05 * vertices).square().mean()
    loss.backward()
    optimizer.step()
    path = tmp_path / "normal_residual.pt"
    torch.save(
        {"field": field.state_dict(), "optimizer": optimizer.state_dict()},
        path,
    )
    restored = DiffusedNormalResidual(vertices, faces, diffusion_steps=3)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=2e-4)
    payload = torch.load(path, weights_only=False)
    restored.load_state_dict(payload["field"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    torch.testing.assert_close(restored.raw_displacements, field.raw_displacements)
    torch.testing.assert_close(restored.deformed_vertices(), field.deformed_vertices())
    assert restored_optimizer.state_dict()["state"]


def test_clothed_ellipsoid_normal_refinement_preserves_topology() -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = torch.tensor(source.vertices, dtype=torch.float32) * torch.tensor([0.65, 1.0, 0.5])
    faces = torch.tensor(source.faces, dtype=torch.long)
    field = DiffusedNormalResidual(vertices, faces, diffusion_steps=3)
    vertical = vertices[:, 1]
    clothing_profile = 0.025 + 0.02 * torch.exp(-((vertical - 0.15) / 0.25).square())
    target = vertices + field.reference_normals * clothing_profile[:, None]
    optimizer = torch.optim.Adam(field.parameters(), lr=0.02)

    def objective() -> torch.Tensor:
        return (field.deformed_vertices() - target).square().mean()

    initial = float(objective().detach())
    for _ in range(40):
        previous = field.raw_displacements.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = objective() + 0.01 * field.smoothness_loss()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        scale = project_normal_residual_step(field, previous, vertices, faces, optimizer)
        assert scale > 0.0
    final = float(objective().detach())
    assert final < initial * 0.05
    deformed = field.deformed_vertices().detach()
    topology = canonical_face_orientation_report(
        vertices.numpy(),
        deformed.numpy(),
        faces.numpy(),
        minimum_area_ratio=0.1,
    )
    mesh = trimesh.Trimesh(vertices=deformed.numpy(), faces=faces.numpy(), process=False)
    assert topology["status"] == "pass"
    assert topology["flipped_face_count"] == 0
    assert topology["collapsed_face_count"] == 0
    assert mesh.is_watertight
    assert int(mesh.euler_number) == 2
