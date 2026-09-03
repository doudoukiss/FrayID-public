from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import pytest
import torch

from frayid.geometry import vertex_normals
from frayid.replay_state import (
    CHECKPOINT_SCHEMA_V2,
    CHECKPOINT_SCHEMA_V3,
    CheckpointStateV2,
    CheckpointStateV3,
    SamplerState,
    capture_checkpoint_state,
    capture_checkpoint_state_v3,
    checkpoint_state_from_dict,
    configure_deterministic_execution,
    nested_state_equal,
    restore_checkpoint_state,
)


def _make_pair() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 7),
        torch.nn.Tanh(),
        torch.nn.Linear(7, 1),
    ).double()
    return model, torch.optim.Adam(model.parameters(), lr=1e-3)


def _advance(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: SamplerState,
    generator: torch.Generator,
) -> dict[str, Any]:
    batch_id = sampler.take()
    scale = 1.0 + 0.01 * random.random() + 0.01 * float(np.random.random())
    values = torch.randn((12, 3), generator=generator, dtype=torch.float64)
    target = 0.3 * values[:, :1] - 0.2 * values[:, 1:2] + 0.1 * values[:, 2:3]
    optimizer.zero_grad(set_to_none=True)
    loss = (model(values * scale) - target).square().mean() + batch_id * 1e-8
    loss.backward()
    gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    return {
        "batch_id": batch_id,
        "loss": loss.detach().clone(),
        "gradients": gradients,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }


def test_sampler_state_round_trip_and_validation() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    sampler = SamplerState.shuffled(8, generator)
    first = sampler.take()
    restored = SamplerState.from_state_dict(sampler.state_dict())
    assert first not in restored.permutation[restored.cursor :]
    assert restored.take() == sampler.take()
    with pytest.raises(ValueError, match="duplicate"):
        SamplerState([0, 0]).validate()


def test_complete_checkpoint_restores_exact_next_step() -> None:
    configure_deterministic_execution()
    random.seed(741)
    np.random.seed(741)
    torch.manual_seed(741)
    generator = torch.Generator(device="cpu").manual_seed(991)
    model, optimizer = _make_pair()
    sampler = SamplerState.shuffled(5, generator)
    _advance(model, optimizer, sampler, generator)
    state = capture_checkpoint_state(
        model,
        optimizer,
        epoch=2,
        global_step=7,
        stage="coarse",
        sampler_state=sampler,
        named_generators={"batch": generator},
        auxiliary_state={"accepted_scale": 0.5},
        immutable_bindings={"source": "public-fixture"},
    )
    payload = state.state_dict()
    restored_state = CheckpointStateV2.from_state_dict(payload)
    expected = _advance(model, optimizer, sampler, generator)

    clone, clone_optimizer = _make_pair()
    clone_generator = torch.Generator(device="cpu")
    clone_sampler = restore_checkpoint_state(
        restored_state,
        clone,
        clone_optimizer,
        named_generators={"batch": clone_generator},
        expected_immutable_bindings={"source": "public-fixture"},
    )
    actual = _advance(clone, clone_optimizer, clone_sampler, clone_generator)
    assert nested_state_equal(expected, actual)


def test_weights_without_rng_and_cursor_are_not_next_step_replay() -> None:
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    generator = torch.Generator(device="cpu").manual_seed(31)
    model, optimizer = _make_pair()
    sampler = SamplerState([0, 1, 2])
    _advance(model, optimizer, sampler, generator)
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    expected = _advance(model, optimizer, sampler, generator)

    clone, clone_optimizer = _make_pair()
    clone.load_state_dict(model_state)
    clone_optimizer.load_state_dict(optimizer_state)
    wrong_generator = torch.Generator(device="cpu").manual_seed(999)
    wrong_sampler = SamplerState([2, 1, 0])
    actual = _advance(clone, clone_optimizer, wrong_sampler, wrong_generator)
    assert not nested_state_equal(expected, actual)


def test_optimizer_parameter_order_mismatch_is_rejected() -> None:
    model, optimizer = _make_pair()
    state = capture_checkpoint_state(
        model,
        optimizer,
        epoch=0,
        global_step=0,
        stage="coarse",
        sampler_state=SamplerState([], 0),
    )
    reversed_optimizer = torch.optim.Adam(list(reversed(list(model.parameters()))), lr=1e-3)
    with pytest.raises(ValueError, match="parameter-group ordering"):
        restore_checkpoint_state(state, model, reversed_optimizer)


def test_vertex_normals_are_repeatable_and_have_finite_gradients() -> None:
    vertices = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [0.5, 0.5, 2.0],
            [-0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    first = vertex_normals(vertices, faces)
    second = vertex_normals(vertices, faces)
    assert torch.equal(first, second)
    first.square().sum().backward()
    assert vertices.grad is not None
    assert bool(torch.isfinite(vertices.grad).all())


def test_checkpoint_v3_round_trip_preserves_v2_compatibility() -> None:
    model, optimizer = _make_pair()
    ambient = {
        "scaffold_sha256": "a" * 64,
        "scaffold_ordering_sha256": "b" * 64,
        "solver_state": {"normalized_residual": 1e-12, "ordering": "ascending_dof"},
        "proposed_direction_sha256": "c" * 64,
        "accepted_alpha_hex": (0.5).hex(),
        "certificate_sha256": "d" * 64,
        "immutable_report_paths": ["outputs/public/e16/report.json"],
    }
    state = capture_checkpoint_state_v3(
        model,
        optimizer,
        epoch=1,
        global_step=3,
        stage="e16_public",
        sampler_state=SamplerState([0, 1], 1),
        ambient_state=ambient,
    )
    payload = state.state_dict()
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_V3
    restored = checkpoint_state_from_dict(payload)
    assert isinstance(restored, CheckpointStateV3)
    assert restored.ambient_state == ambient

    v2 = capture_checkpoint_state(
        model,
        optimizer,
        epoch=0,
        global_step=0,
        stage="legacy",
        sampler_state=SamplerState([], 0),
    )
    v2_payload = v2.state_dict()
    assert v2_payload["schema_version"] == CHECKPOINT_SCHEMA_V2
    assert isinstance(checkpoint_state_from_dict(v2_payload), CheckpointStateV2)


def test_checkpoint_v3_rejects_incomplete_ambient_state() -> None:
    model, optimizer = _make_pair()
    with pytest.raises(ValueError, match="incomplete"):
        capture_checkpoint_state_v3(
            model,
            optimizer,
            epoch=0,
            global_step=0,
            stage="e16_public",
            sampler_state=SamplerState([], 0),
            ambient_state={"scaffold_sha256": "missing-fields"},
        )
