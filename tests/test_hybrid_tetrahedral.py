from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from frayid.eulerian_field import EulerianImageField, conventional_surface_audit
from frayid.hybrid_tetrahedral import (
    FixedSignTetrahedralField,
    corresponding_normal_loss,
    distorted_fixed_sign_initialization,
    ellipsoid_implicit_values,
    minimum_face_area_ratio,
    project_fixed_sign_step,
    regular_tetrahedral_grid,
    symmetric_chamfer_distance,
    tetrahedral_grid_from_axes,
    tetrahedron_signed_volumes,
)


def _field(resolution: int = 8) -> tuple[FixedSignTetrahedralField, torch.Tensor]:
    positions, tetrahedra = regular_tetrahedral_grid(resolution)
    target_values = ellipsoid_implicit_values(positions, torch.tensor([0.68, 0.94, 0.52]))
    assert not torch.any(target_values == 0)
    field = FixedSignTetrahedralField(
        positions, tetrahedra, distorted_fixed_sign_initialization(target_values)
    )
    return field, target_values


def test_fixed_sign_tetrahedral_surface_is_watertight_and_noninverted() -> None:
    field, _ = _field()
    vertices = field.surface_vertices().detach().numpy()
    faces = field.surface_faces.detach().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    volumes = tetrahedron_signed_volumes(field.positions, field.tetrahedra)
    assert bool(torch.all(volumes.abs() > 1e-8))
    assert mesh.is_watertight
    assert int(mesh.euler_number) == 2
    assert len(mesh.split(only_watertight=False)) == 1
    assert np.isfinite(vertices).all()


def test_concave_fixed_sign_surface_has_consistent_outward_winding() -> None:
    positions, tetrahedra = regular_tetrahedral_grid(12, extent=1.25)
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
    values = torch.maximum(torch.minimum(torch.minimum(main, lobe), bridge), -cutter)
    field = EulerianImageField(positions.double(), tetrahedra, values.double())
    assert conventional_surface_audit(field.surface_vertices(), field.surface_faces)["status"] == (
        "pass"
    )


def test_rectangular_tetrahedral_grid_preserves_axis_order_and_volume() -> None:
    axes = (
        torch.tensor([-1.0, -0.25, 1.0]),
        torch.tensor([-2.0, -0.5, 0.5, 2.0]),
        torch.tensor([3.0, 4.0, 6.0, 9.0, 13.0]),
    )
    positions, tetrahedra = tetrahedral_grid_from_axes(axes)
    assert positions.shape == (3 * 4 * 5, 3)
    assert tetrahedra.shape == (2 * 3 * 4 * 6, 4)
    torch.testing.assert_close(positions[0], torch.tensor([-1.0, -2.0, 3.0]))
    torch.testing.assert_close(positions[-1], torch.tensor([1.0, 2.0, 13.0]))
    assert bool(torch.all(tetrahedron_signed_volumes(positions, tetrahedra).abs() > 1e-8))


def test_fixed_sign_hybrid_optimization_and_resume(tmp_path: Path) -> None:
    field, target_values = _field()
    target = FixedSignTetrahedralField(field.positions, field.tetrahedra, target_values)
    target_vertices = target.surface_vertices().detach()
    reference = field.surface_vertices().detach()
    optimizer = torch.optim.Adam(field.parameters(), lr=0.04)

    def objective() -> torch.Tensor:
        vertices = field.surface_vertices()
        return symmetric_chamfer_distance(
            vertices, target_vertices
        ) + 0.2 * corresponding_normal_loss(vertices, target_vertices, field.surface_faces)

    initial = float(objective().detach())
    for _ in range(30):
        previous = field.magnitude_logits.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        scale = project_fixed_sign_step(field, previous, reference, minimum_area_ratio=0.1)
        assert scale > 0
        assert float(field.minimum_sign_margin()) >= field.minimum_absolute_value
        assert (
            float(minimum_face_area_ratio(field.surface_vertices(), reference, field.surface_faces))
            > 0.1
        )
    final = float(objective().detach())
    assert final < initial * 0.95

    checkpoint = Path(str(tmp_path)) / "hybrid_checkpoint.pt"
    torch.save({"model": field.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    restored, _ = _field()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.04)
    payload = torch.load(checkpoint, weights_only=False)
    restored.load_state_dict(payload["model"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    torch.testing.assert_close(restored.surface_vertices(), field.surface_vertices())
    assert restored_optimizer.state_dict()["state"]
