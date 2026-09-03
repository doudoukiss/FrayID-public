from __future__ import annotations

import torch

from frayid.deformation_cage import TrilinearDeformationCage, project_cage_step


def test_trilinear_cage_reproduces_constant_translation_and_gradient() -> None:
    vertices = torch.tensor(
        [[-1.0, -2.0, -0.5], [1.0, -2.0, 0.5], [1.0, 2.0, -0.5], [-1.0, 2.0, 0.5]]
    )
    cage = TrilinearDeformationCage(vertices, (3, 4, 2))
    translation = torch.tensor([0.2, -0.1, 0.3])
    with torch.no_grad():
        cage.controls.copy_(translation.expand_as(cage.controls))
    torch.testing.assert_close(cage.vertex_offsets(), translation.expand_as(vertices))
    cage.deformed_vertices().square().sum().backward()  # type: ignore[no-untyped-call]
    assert cage.controls.grad is not None
    assert bool(torch.isfinite(cage.controls.grad).all())


def test_cage_projection_backtracks_before_triangle_flip() -> None:
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [0.0, 1.0, 0.2], [1.0, 1.0, 0.3]])
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    cage = TrilinearDeformationCage(vertices, (2, 2, 2))
    optimizer = torch.optim.Adam(cage.parameters(), lr=1e-3)
    optimizer.state[cage.controls] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.ones_like(cage.controls),
        "exp_avg_sq": torch.ones_like(cage.controls),
    }
    previous = cage.controls.detach().clone()
    with torch.no_grad():
        cage.controls[..., 1] = -2.0 * cage.controls.new_tensor(
            [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]
        )
    diagnostics: dict[str, object] = {}
    scale = project_cage_step(cage, previous, faces, optimizer, diagnostics=diagnostics)
    assert 0.0 < scale < 1.0
    torch.testing.assert_close(
        optimizer.state[cage.controls]["exp_avg"], torch.full_like(cage.controls, scale)
    )
    torch.testing.assert_close(
        optimizer.state[cage.controls]["exp_avg_sq"],
        torch.full_like(cage.controls, scale * scale),
    )
    assert diagnostics["accepted_scale"] == scale
    assert diagnostics["moment_damping_scale"] == scale
    assert diagnostics["backtracking_count"] > 0  # type: ignore[operator]
    proposed = diagnostics["proposed"]
    accepted = diagnostics["accepted"]
    assert isinstance(proposed, dict)
    assert isinstance(accepted, dict)
    assert proposed["violated_signed_constraint_count"] > 0
    assert accepted["violated_signed_constraint_count"] == 0
    assert accepted["violated_unsigned_constraint_count"] == 0


def test_cage_projection_reports_full_scale_without_moment_damping() -> None:
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [0.0, 1.0, 0.2], [1.0, 1.0, 0.3]])
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    cage = TrilinearDeformationCage(vertices, (2, 2, 2))
    optimizer = torch.optim.Adam(cage.parameters(), lr=1e-3)
    previous = cage.controls.detach().clone()
    with torch.no_grad():
        cage.controls[..., 0] += 1e-5
    diagnostics: dict[str, object] = {}
    scale = project_cage_step(cage, previous, faces, optimizer, diagnostics=diagnostics)
    assert scale == 1.0
    assert diagnostics["backtracking_count"] == 0
    assert diagnostics["moment_damping_scale"] == 1.0
