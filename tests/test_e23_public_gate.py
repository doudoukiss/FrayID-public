from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from frayid.eulerian_reconstruction import (
    public_eulerian_fixture,
    public_image_loss,
    render_public_evidence,
)
from frayid.intrinsic_geometry import IntrinsicGeometryTransform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_post_v1_e23_public_gate.py"
STATUS = PROJECT_ROOT / "configs/evaluation/post_v1_e23_intrinsic_full_rank_geometry_r01.yaml"


def test_e23_status_freezes_only_full_rank_coordinate_change() -> None:
    if not STATUS.exists():
        pytest.skip("private status record is intentionally absent from public snapshot")
    status = yaml.safe_load(STATUS.read_text())
    assert status["experiment_id"] == "postv1_e23_intrinsic_full_rank_geometry_r01"
    assert status["only_changed_mechanism"].startswith("optimize_full_rank_u_equals")
    assert status["coordinates"]["lambda"] == 1.0
    assert status["representation"]["trainable_geometric_degrees_of_freedom_per_arm"] == 1662
    assert status["matched_public_comparison"]["learning_rate_per_arm"] == 0.006
    assert status["matched_public_comparison"]["optimizer_steps_per_arm"] == 300
    assert status["matched_public_comparison"]["extra_smoothing_or_regularization_loss"] is False
    assert status["limits"]["public_runs"] == 1
    assert status["limits"]["automatic_retries"] == 0


def test_intrinsic_image_only_gradient_is_finite_nonzero_and_directional() -> None:
    fixture = public_eulerian_fixture()
    initial_field = fixture.initial_field()
    target_field = fixture.target_field()
    faces = initial_field.surface_faces
    initial = initial_field.surface_vertices().detach().double()
    target = target_field.surface_vertices().detach().double()
    evidence = render_public_evidence(target, faces)
    transform = IntrinsicGeometryTransform.from_mesh(initial, faces)
    coordinates = torch.nn.Parameter(transform.encode(initial).detach().clone())
    loss = public_image_loss(transform.decode(coordinates), faces, evidence, 0, seed=71_000)
    (gradient,) = torch.autograd.grad(loss, (coordinates,))
    norm = torch.linalg.vector_norm(gradient)
    assert torch.isfinite(gradient).all()
    assert float(norm) > 0.0
    direction = gradient / norm
    epsilon = 1.0e-5
    with torch.no_grad():
        positive = public_image_loss(
            transform.decode(coordinates + epsilon * direction),
            faces,
            evidence,
            0,
            seed=71_000,
        )
        negative = public_image_loss(
            transform.decode(coordinates - epsilon * direction),
            faces,
            evidence,
            0,
            seed=71_000,
        )
    finite_difference = float((positive - negative) / (2.0 * epsilon))
    analytic = float((gradient * direction).sum())
    assert analytic * finite_difference > 0.0
    assert abs(analytic - finite_difference) / abs(analytic) <= 0.10


def test_e23_runner_is_public_only_matched_and_immutable() -> None:
    source = RUNNER.read_text()
    assert "LAMBDA_VALUE = 1.0" in source
    assert "LEARNING_RATE = 0.006" in source
    assert "OPTIMIZER_STEPS = 300" in source
    assert "PREFLIGHT_STEPS = 40" in source
    assert "REPLAY_STEP = 37" in source
    assert "extra_smoothing_or_regularization_loss" in source
    assert "immutable E23 report exists" in source
    assert "immutable E23 artifact directory exists" in source
    assert "_comparison_blockers" in source
    assert "_exact_surface_audit" in source
    assert "_p2_hairpin_regression" in source
    for counter in (
        "private_input_reads",
        "development_evidence_reads",
        "sealed_test_accesses",
        "gpu_hours",
        "cloud_invocations",
        "spend_usd",
        "automatic_paid_retries",
    ):
        assert f'"{counter}": 0' in source
    assert "data/private" not in source
    assert "docs/assets" not in source
    assert "modal.run" not in source
