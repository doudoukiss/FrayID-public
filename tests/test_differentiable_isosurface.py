from __future__ import annotations

import numpy as np
import torch

from frayid.differentiable_isosurface import (
    certify_linear_surface_path,
    certify_zero_set_scalar_path,
    interpolate_zero_crossings,
    same_open_sign_chamber,
)
from frayid.eulerian_field import EulerianImageField, conventional_surface_audit
from frayid.hybrid_tetrahedral import (
    distorted_fixed_sign_initialization,
    ellipsoid_implicit_values,
    regular_tetrahedral_grid,
)


def _field() -> EulerianImageField:
    positions, tetrahedra = regular_tetrahedral_grid(7, extent=1.25)
    target = ellipsoid_implicit_values(positions, torch.tensor([0.66, 0.88, 0.50]))
    return EulerianImageField(
        positions.double(), tetrahedra, distorted_fixed_sign_initialization(target).double()
    )


def test_image_only_zero_set_gradient_matches_directional_difference() -> None:
    field = _field()
    target = field.surface_vertices().detach() + torch.tensor([0.015, -0.01, 0.005])
    loss = (field.surface_vertices() - target).square().mean()
    (gradient,) = torch.autograd.grad(loss, (field.magnitude_logits,))
    assert torch.isfinite(gradient).all()
    assert float(torch.linalg.vector_norm(gradient)) > 0
    direction = torch.linspace(-1.0, 1.0, gradient.numel(), dtype=gradient.dtype)
    direction = direction / torch.linalg.vector_norm(direction)
    analytic = float((gradient * direction).sum())
    original = field.magnitude_logits.detach().clone()
    epsilon = 1e-5
    with torch.no_grad():
        field.magnitude_logits.copy_(original + epsilon * direction)
        plus = float((field.surface_vertices() - target).square().mean())
        field.magnitude_logits.copy_(original - epsilon * direction)
        minus = float((field.surface_vertices() - target).square().mean())
        field.magnitude_logits.copy_(original)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert analytic * finite_difference > 0
    assert abs(analytic - finite_difference) <= 2e-5


def test_zero_crossings_reject_non_crossing_edges() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = torch.tensor([[0, 1]])
    with torch.no_grad():
        values = torch.tensor([1.0, 2.0])
    try:
        interpolate_zero_crossings(positions, edges, values)
    except ValueError as error:
        assert "opposite signs" in str(error)
    else:
        raise AssertionError("non-crossing edge was accepted")


def test_sign_chamber_surface_path_certificate_and_serialized_audit() -> None:
    field = _field()
    start = field.field_values.detach().cpu().numpy()
    end = start.copy()
    end[start < 0] *= 1.03
    end[start > 0] *= 0.97
    reference = field.surface_vertices().detach().cpu().numpy()
    certificate = certify_zero_set_scalar_path(
        field.positions.cpu().numpy(),
        field.surface_edges.cpu().numpy(),
        field.surface_faces.cpu().numpy(),
        start,
        end,
        reference,
    )
    assert certificate.status == "pass"
    assert certificate.minimum_signed_area_lower_bound >= 0.01
    assert certificate.minimum_unsigned_area_lower_bound >= 0.10
    assert conventional_surface_audit(field.surface_vertices(), field.surface_faces)["status"] == (
        "pass"
    )
    assert same_open_sign_chamber(torch.from_numpy(start), torch.from_numpy(end))


def test_sign_change_and_interior_linear_collapse_do_not_pass() -> None:
    field = _field()
    start = field.field_values.detach().cpu().numpy()
    end = start.copy()
    end[0] *= -1
    certificate = certify_zero_set_scalar_path(
        field.positions.cpu().numpy(),
        field.surface_edges.cpu().numpy(),
        field.surface_faces.cpu().numpy(),
        start,
        end,
        field.surface_vertices().detach().cpu().numpy(),
    )
    assert certificate.status == "fail"
    assert certificate.blocker == "sign_chamber_violation"

    linear = certify_linear_surface_path(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        np.asarray([[0, 1, 2]], dtype=np.int64),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        maximum_subdivision_depth=8,
    )
    assert linear.status != "pass"
