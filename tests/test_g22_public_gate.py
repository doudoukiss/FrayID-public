from __future__ import annotations

import math
import runpy
from pathlib import Path

import pytest
import torch
import yaml

from frayid.eulerian_field import conventional_surface_audit
from frayid.eulerian_reconstruction import (
    PUBLIC_HELD_OUT_VIEW_COUNT,
    PUBLIC_IMAGE_SIZE,
    PUBLIC_TRAIN_VIEW_COUNT,
    PUBLIC_VIEW_COUNT,
    probe_classification,
    public_eulerian_fixture,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_post_v1_g22_public_gate.py"
POSTMORTEM = PROJECT_ROOT / "scripts/audit_post_v1_g22_retained_artifact.py"
STATUS = PROJECT_ROOT / "configs/evaluation/post_v1_e22_eulerian_image_active_surface_r01.yaml"


def test_g22_status_freezes_matched_public_image_geometry_gate() -> None:
    if not STATUS.exists():
        return
    status = yaml.safe_load(STATUS.read_text())
    assert status["experiment_id"] == "postv1_e22_eulerian_image_active_surface_r01"
    assert status["correctness_id"] == "postv1_g22_eulerian_image_reconstruction_r01"
    assert status["public_comparison"] == {
        "train_views": 12,
        "held_out_views": 6,
        "image_long_side": 128,
        "optimizer_steps_per_arm": 300,
        "truth_geometry_used_for_training": False,
        "same_seed_renderer_images_cameras_losses_sampling_and_compute": True,
        "minimum_relative_bidirectional_geometry_error_improvement": 0.10,
        "held_out_image_metrics_may_worsen": False,
    }
    assert status["unchanged_gates"]["directional_median_normal_degrees_maximum"] == 5.0
    assert status["unchanged_gates"]["relative_volume_error_maximum"] == 0.031
    assert status["limits"]["automatic_retries"] == 0


def test_public_fixture_is_concave_valid_and_uses_sign_template_initializer() -> None:
    fixture = public_eulerian_fixture()
    target = fixture.target_field()
    initial = fixture.initial_field()
    assert PUBLIC_VIEW_COUNT == PUBLIC_TRAIN_VIEW_COUNT + PUBLIC_HELD_OUT_VIEW_COUNT
    assert PUBLIC_IMAGE_SIZE == (128, 128)
    assert torch.equal(target.surface_edges, initial.surface_edges)
    assert torch.equal(target.surface_faces, initial.surface_faces)
    assert (
        conventional_surface_audit(target.surface_vertices(), target.surface_faces)["status"]
        == "pass"
    )
    assert (
        conventional_surface_audit(initial.surface_vertices(), initial.surface_faces)["status"]
        == "pass"
    )
    assert (
        probe_classification(target.surface_vertices(), target.surface_faces, fixture)["status"]
        == "pass"
    )
    assert (
        probe_classification(initial.surface_vertices(), initial.surface_faces, fixture)["status"]
        == "pass"
    )

    phase = torch.arange(fixture.target_values.numel(), dtype=fixture.target_values.dtype)
    variation = 1.0 + 0.12 * torch.sin(phase * (1.0 + math.sqrt(5.0)) / 2.0)
    expected = torch.sign(fixture.target_values) * torch.where(
        fixture.target_values < 0, 0.04 * variation, 0.80 * variation
    )
    torch.testing.assert_close(fixture.initial_values, expected)


def test_g22_runner_is_immutable_public_only_and_complete() -> None:
    source = RUNNER.read_text()
    assert "OPTIMIZER_STEPS = 300" in source
    assert "PUBLIC_TRAIN_VIEW_COUNT" in source
    assert "PUBLIC_HELD_OUT_VIEW_COUNT" in source
    assert "MINIMUM_RELATIVE_GEOMETRY_IMPROVEMENT = 0.10" in source
    assert "worker.join(MAXIMUM_TOTAL_SECONDS)" in source
    assert "immutable G22 report exists" in source
    assert "immutable G22 artifact directory exists" in source
    assert "_p2_hairpin_regression" in source
    assert "_exact_surface_audit" in source
    assert "_comparison_blockers" in source
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


def test_g22_report_path_accepts_repository_and_external_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    namespace = runpy.run_path(str(RUNNER), run_name="g22_runner")
    report_path = namespace["_portable_report_path"]
    repository_artifact = PROJECT_ROOT / "outputs" / "public" / "artifact.npz"
    assert report_path(repository_artifact) == "outputs/public/artifact.npz"

    external_artifact = tmp_path / "artifact.npz"
    assert report_path(external_artifact) == str(external_artifact.resolve())


def test_g22_retained_artifact_audit_is_read_only_and_nonpromotable() -> None:
    if not POSTMORTEM.exists():
        pytest.skip("hash-bound private-artifact auditor is absent from public snapshot")
    source = POSTMORTEM.read_text()
    assert '"optimizer_rerun": False' in source
    assert '"promotion_permitted": False' in source
    assert '"public_runs": 0' in source
    assert "immutable G22 postmortem exists" in source
    assert "data/private" not in source
    assert "docs/assets" not in source
    assert "modal.run" not in source
