from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from frayid.deformation_cage import TrilinearDeformationCage, project_cage_step
from frayid.feasible_cage import (
    FeasibleCageProjectionFailure,
    archive_qp_failure,
    linearized_feasible_cage_direction,
    project_halfspace_qp,
)
from frayid.geometry import canonical_topology_quantities


def test_halfspace_qp_preserves_distant_candidate_component() -> None:
    candidate = torch.tensor([-2.0, 3.0], dtype=torch.float64)
    matrix = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    lower = torch.tensor([-0.1], dtype=torch.float64)
    result = project_halfspace_qp(candidate, matrix, lower, trust_region_radius=4.0)
    assert result.certified
    assert result.final_direction is not None
    torch.testing.assert_close(
        result.final_direction, torch.tensor([-0.1, 3.0], dtype=torch.float64), atol=1e-8, rtol=0.0
    )
    assert result.active_constraint_count == 1


def test_halfspace_qp_trust_region_preserves_feasibility() -> None:
    candidate = torch.tensor([-2.0, 3.0], dtype=torch.float64)
    matrix = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    lower = torch.tensor([-0.1], dtype=torch.float64)
    result = project_halfspace_qp(candidate, matrix, lower, trust_region_radius=1.0)
    assert result.certified
    assert result.final_direction is not None
    assert float(torch.linalg.vector_norm(result.final_direction)) == pytest.approx(1.0)
    assert bool(torch.all(matrix @ result.final_direction >= lower - 1e-8))
    assert result.trust_scale < 1.0


def test_halfspace_qp_reports_contradictory_zero_row() -> None:
    result = project_halfspace_qp(
        torch.zeros(1), torch.zeros((1, 1)), torch.ones(1), trust_region_radius=1.0
    )
    assert result.status == "contradictory_zero_row"
    assert result.contradictory_zero_row_count == 1


def test_halfspace_qp_handles_redundant_constraints() -> None:
    result = project_halfspace_qp(
        torch.tensor([-3.0], dtype=torch.float64),
        torch.tensor([[2.0], [1.0]], dtype=torch.float64),
        torch.tensor([-2.0, 0.0], dtype=torch.float64),
        trust_region_radius=10.0,
    )
    assert result.certified
    assert result.final_direction is not None
    torch.testing.assert_close(
        result.final_direction, torch.zeros(1, dtype=torch.float64), atol=1e-8, rtol=0.0
    )
    assert result.scaled_certificate is not None
    assert result.scaled_certificate.maximum <= 1e-8


@pytest.mark.parametrize("order", ([0, 1], [1, 0]))
def test_halfspace_qp_is_invariant_to_row_order_and_positive_scaling(
    order: list[int],
) -> None:
    matrix = torch.tensor([[2.0, 0.0], [1.0, 0.0]], dtype=torch.float64)[order]
    lower = torch.tensor([-2.0, 0.0], dtype=torch.float64)[order]
    result = project_halfspace_qp(
        torch.tensor([-3.0, 2.0], dtype=torch.float64),
        matrix,
        lower,
        trust_region_radius=4.0,
    )
    assert result.certified
    assert result.final_direction is not None
    torch.testing.assert_close(
        result.final_direction,
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        atol=1e-8,
        rtol=0.0,
    )


def test_halfspace_qp_normalized_five_row_regression() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "normalized_qp_failure.json").read_text()
    )
    result = project_halfspace_qp(
        torch.tensor(fixture["candidate"], dtype=torch.float64),
        torch.tensor(fixture["A"], dtype=torch.float64),
        torch.tensor(fixture["b"], dtype=torch.float64),
        trust_region_radius=8.0,
    )
    assert result.certified
    assert result.final_direction is not None
    torch.testing.assert_close(
        result.final_direction,
        torch.tensor(fixture["oracle"], dtype=torch.float64),
        atol=1e-8,
        rtol=0.0,
    )


def test_halfspace_qp_reports_tautological_zero_rows() -> None:
    result = project_halfspace_qp(
        torch.tensor([-2.0], dtype=torch.float64),
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        torch.tensor([-1.0, 0.0], dtype=torch.float64),
        trust_region_radius=4.0,
    )
    assert result.certified
    assert result.tautological_zero_row_count == 1
    assert result.final_direction is not None
    torch.testing.assert_close(
        result.final_direction, torch.zeros(1, dtype=torch.float64), atol=1e-8, rtol=0.0
    )


def test_qp_failure_archive_preserves_problem_and_rejects_fallback(tmp_path: Path) -> None:
    candidate = torch.zeros(1, dtype=torch.float64)
    matrix = torch.zeros((1, 1), dtype=torch.float64)
    lower = torch.ones(1, dtype=torch.float64)
    result = project_halfspace_qp(candidate, matrix, lower, trust_region_radius=1.0)
    failure = FeasibleCageProjectionFailure(result, candidate, matrix, lower)
    destination = tmp_path / "failure_0001"
    report = archive_qp_failure(
        destination,
        failure,
        last_valid_checkpoint=None,
        context={"experiment_id": "injected_failure"},
    )
    assert (destination / "qp_problem.npz").is_file()
    assert (destination / "failure_report.json").is_file()
    assert report["fallback_used"] is False
    assert report["candidate_accepted"] is False
    with pytest.raises(FileExistsError):
        archive_qp_failure(
            destination,
            failure,
            last_valid_checkpoint=None,
            context={"experiment_id": "injected_failure"},
        )


def test_cage_fixture_propagates_control_gradients() -> None:
    vertices = torch.tensor(
        [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    cage = TrilinearDeformationCage(vertices, (2, 2, 2))
    loss = cage.deformed_vertices().square().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert cage.controls.grad is not None
    assert bool(torch.isfinite(cage.controls.grad).all())


def test_linearized_direction_retains_distant_update_under_local_pressure() -> None:
    import numpy as np
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float64)
    vertices *= torch.tensor([0.55, 0.95, 0.42], dtype=torch.float64)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.long)
    cage = TrilinearDeformationCage(vertices, (8, 8, 4))
    low, high = 0.0, 0.3
    for _ in range(50):
        amplitude = 0.5 * (low + high)
        with torch.no_grad():
            cage.controls.zero_()
            cage.controls[3, 7, 1, 2] = amplitude
            signed, _ = canonical_topology_quantities(vertices, cage.deformed_vertices(), faces)
        if float(signed.min()) > 0.011:
            low = amplitude
        else:
            high = amplitude
    with torch.no_grad():
        cage.controls.zero_()
        cage.controls[3, 7, 1, 2] = low
    previous = cage.controls.detach().clone()
    candidate = torch.zeros_like(previous)
    candidate[4, 0, 2, 0] = 0.2
    candidate[3, 7, 1, 2] = 0.08

    with torch.no_grad():
        cage.controls.copy_(previous + candidate)
    optimizer = torch.optim.Adam(cage.parameters(), lr=0.002)
    control_scale = project_cage_step(cage, previous, faces, optimizer)
    assert control_scale < 0.01

    with torch.no_grad():
        cage.controls.copy_(previous)
    projected, report = linearized_feasible_cage_direction(cage, candidate, faces)
    assert report.active_set_size >= 1
    assert float(projected[4, 0, 2, 0]) == pytest.approx(0.2, abs=1e-6)
    with torch.no_grad():
        cage.controls.copy_(previous + projected)
    assert project_cage_step(cage, previous, faces, optimizer) == 1.0
