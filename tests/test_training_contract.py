from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frayid.config import load_config
from frayid.io import write_json
from frayid.modal_plan import build_modal_smoke_plan
from frayid.schemas import SmokeRunReport
from frayid.training import (
    CanonicalGeometryModel,
    TrainingEvidence,
    _backtrack_canonical_update,
    _evaluate_fixed_objective,
    _load_checkpoint,
    _load_checkpoint_runtime,
    _renderer_sample_count,
    _renderer_sigma_pixels,
    _save_checkpoint,
    load_canonical_model_state,
    training_stage_for_epoch,
)


def test_fixed_objective_is_repeatable_and_preserves_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model = CanonicalGeometryModel(
        vertices,
        torch.tensor([[0, 1, 2]]),
        torch.ones((3, 1)),
        2,
        config,
    )
    evidence = TrainingEvidence(
        masks=torch.zeros((2, 1, 1)),
        normals=torch.zeros((2, 1, 1, 3)),
        transforms=torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1),
        frame_indices=torch.arange(2),
        intrinsics=torch.eye(3),
        source_image_size=(1, 1),
    )

    def random_losses(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {name: torch.rand(()) for name in config.losses.__class__.model_fields}

    monkeypatch.setattr("frayid.training._loss_values_for_slot", random_losses)
    torch.manual_seed(901)
    initial_rng = torch.random.get_rng_state().clone()
    first = _evaluate_fixed_objective(model, evidence, config)
    assert torch.equal(torch.random.get_rng_state(), initial_rng)
    second = _evaluate_fixed_objective(model, evidence, config)
    assert first == second
    assert torch.equal(torch.random.get_rng_state(), initial_rng)


def test_modal_plan_requires_manual_authorization_after_failed_smoke(tmp_path: Path) -> None:
    config = load_config()
    config = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={
                    "dataset_root": tmp_path / "dataset",
                    "run_root": tmp_path / "runs",
                }
            )
        }
    )
    smoke_path = config.paths.run_root / config.run_id / "smoke/smoke_report.json"
    write_json(
        smoke_path,
        SmokeRunReport(
            run_id=config.run_id,
            status="fail",
            frame_count=24,
            epoch_count=2,
            optimizer_steps=48,
            checkpoint_resume_verified=True,
            enabled_losses=[],
            gradient_parameter_groups={},
            epoch_metrics=[],
            blockers=["geometry_loss_reduction_below_gate"],
        ),
    )

    plan = build_modal_smoke_plan(config, config_path=Path("config.yaml"))
    assert "previous_smoke_failed_manual_retry_authorization_required" in plan.blockers


def test_coarse_medium_fine_schedule_is_fixed() -> None:
    config = load_config()
    assert config.smoke.learning_rate == 2e-4
    assert config.training.learning_rate == 1e-4
    assert training_stage_for_epoch(config, 0).name == "coarse"
    assert training_stage_for_epoch(config, 11).name == "coarse"
    assert training_stage_for_epoch(config, 12).name == "medium"
    assert training_stage_for_epoch(config, 59).name == "medium"
    assert training_stage_for_epoch(config, 60).name == "fine"
    assert training_stage_for_epoch(config, 199).name == "fine"
    assert _renderer_sample_count(config, (128, 128)) == 2048
    assert _renderer_sample_count(config, (192, 192)) == 4608
    assert _renderer_sample_count(config, (256, 256)) == 8192
    assert _renderer_sigma_pixels(config, (128, 128)) == 1.75
    assert _renderer_sigma_pixels(config, (192, 192)) == 2.625
    assert _renderer_sigma_pixels(config, (256, 256)) == 3.5


def test_canonical_only_pose_disables_frame_residual() -> None:
    config = load_config()
    vertices = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
    model = CanonicalGeometryModel(
        vertices,
        torch.tensor([[0, 1, 2]]),
        torch.ones((3, 1)),
        2,
        config,
    )
    posed, residual = model.posed_vertices(
        0,
        torch.eye(4).reshape(1, 4, 4),
        residual_enabled=False,
    )
    torch.testing.assert_close(posed, vertices)
    torch.testing.assert_close(residual, torch.zeros_like(vertices))


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path: Path) -> None:
    config = load_config()
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    weights = torch.ones((3, 1))
    model = CanonicalGeometryModel(vertices, faces, weights, 3, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model.canonical_offsets.square().sum() + model.sdf(vertices).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint, model, optimizer, 7, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "canonical_checkpoint.v2"
    assert payload["next_step_replay_capable"] is True

    restored = CanonicalGeometryModel(vertices, faces, weights, 3, config)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    assert _load_checkpoint(checkpoint, restored, restored_optimizer, torch.device("cpu")) == 7
    for expected, actual in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_legacy_v1_checkpoint_loads_but_is_not_replay_proof(tmp_path: Path) -> None:
    config = load_config()
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    weights = torch.ones((3, 1))
    model = CanonicalGeometryModel(vertices, faces, weights, 1, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "legacy-v1.pt"
    torch.save(
        {
            "schema_version": "canonical_checkpoint.v1",
            "epoch": 4,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint,
    )

    clone = CanonicalGeometryModel(vertices, faces, weights, 1, config)
    clone_optimizer = torch.optim.Adam(clone.parameters(), lr=1e-3)
    with pytest.warns(RuntimeWarning, match="cannot prove exact next-step replay"):
        loaded = _load_checkpoint_runtime(checkpoint, clone, clone_optimizer, torch.device("cpu"))
    assert loaded.epoch == 4
    assert loaded.sampler_state is None
    assert loaded.next_step_replay_capable is False


def test_canonical_optimizer_step_is_backtracked_before_foldover() -> None:
    config = load_config()
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model = CanonicalGeometryModel(
        vertices,
        torch.tensor([[0, 1, 2]]),
        torch.ones((3, 1)),
        1,
        config,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    previous = model.canonical_offsets.detach().clone()
    with torch.no_grad():
        model.canonical_offsets[2, 1] = -2.0
    scale = _backtrack_canonical_update(model, previous, optimizer, config)
    assert 0.0 < scale < 1.0
    assert float(model.canonical_vertices[2, 1]) >= 0.1


def test_root_corrections_are_bounded_and_receive_geometry_gradient() -> None:
    config = load_config()
    vertices = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
    model = CanonicalGeometryModel(
        vertices,
        torch.tensor([[0, 1, 2]]),
        torch.ones((3, 1)),
        1,
        config,
    )
    with torch.no_grad():
        model.root_rotation_corrections_raw.fill_(100.0)
        model.root_translation_corrections_raw.fill_(100.0)
    assert float(
        torch.linalg.vector_norm(model.root_rotation_corrections, dim=-1).max()
    ) == pytest.approx(torch.deg2rad(torch.tensor(5.0)).item(), abs=1e-7)
    assert float(
        torch.linalg.vector_norm(model.root_translation_corrections, dim=-1).max()
    ) == pytest.approx(0.05)
    model.root_rotation_corrections_raw.data.zero_()
    model.root_translation_corrections_raw.data.zero_()
    zero_regularizer = (model.root_rotation_corrections.square().mean() + 1e-12).sqrt() + (
        model.root_translation_corrections.square().mean() + 1e-12
    ).sqrt()
    zero_regularizer.backward()
    assert model.root_rotation_corrections_raw.grad is not None
    assert bool(torch.isfinite(model.root_rotation_corrections_raw.grad).all())
    assert model.root_translation_corrections_raw.grad is not None
    assert bool(torch.isfinite(model.root_translation_corrections_raw.grad).all())
    model.zero_grad(set_to_none=True)
    posed, _ = model.posed_vertices(
        0,
        torch.eye(4).reshape(1, 4, 4),
        residual_enabled=False,
    )
    posed.sum().backward()
    assert model.root_rotation_corrections_raw.grad is not None
    assert model.root_translation_corrections_raw.grad is not None


def test_pre_root_correction_checkpoint_loads_with_zero_corrections() -> None:
    config = load_config()
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model = CanonicalGeometryModel(
        vertices,
        torch.tensor([[0, 1, 2]]),
        torch.ones((3, 1)),
        2,
        config,
    )
    legacy_state = dict(model.state_dict())
    legacy_state.pop("root_rotation_corrections_raw")
    legacy_state.pop("root_translation_corrections_raw")
    restored = CanonicalGeometryModel(
        vertices,
        torch.tensor([[0, 1, 2]]),
        torch.ones((3, 1)),
        2,
        config,
    )
    load_canonical_model_state(restored, legacy_state)
    torch.testing.assert_close(restored.root_rotation_corrections, torch.zeros((2, 3)))
    torch.testing.assert_close(restored.root_translation_corrections, torch.zeros((2, 3)))
