from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from frayid.e25_stage import (
    E25_CHECKPOINT_SCHEMA,
    assert_frozen_connectivity,
    capture_checkpoint,
    commit_stage_surface,
    restore_checkpoint,
)
from frayid.hybrid_tetrahedral import (
    fixed_sign_surface_connectivity,
    regular_tetrahedral_grid,
)


def _sphere_surface() -> tuple[torch.Tensor, torch.Tensor]:
    positions, tetrahedra = regular_tetrahedral_grid(9, extent=1.0)
    values = torch.linalg.vector_norm(positions, dim=-1) - 0.62
    edges, faces = fixed_sign_surface_connectivity(positions, tetrahedra, torch.sign(values))
    edge_values = values[edges]
    interpolation = edge_values[:, 0] / (edge_values[:, 0] - edge_values[:, 1])
    endpoints = positions[edges]
    vertices = endpoints[:, 0] + interpolation[:, None] * (endpoints[:, 1] - endpoints[:, 0])
    return vertices, faces


def test_stage_commitment_requires_every_exact_boundary_gate() -> None:
    vertices, faces = _sphere_surface()
    commitment = commit_stage_surface(
        vertices,
        faces,
        resolution=24,
        exact_intersection_pair_count=0,
        probes_preserved=True,
        replay_exact=True,
    )
    assert commitment.status == "pass"
    assert commitment.conventional_topology["euler_number"] == 2
    assert_frozen_connectivity(commitment, faces.clone())
    changed = faces.clone()
    changed[0] = changed[0, [0, 2, 1]]
    with pytest.raises(ValueError, match="connectivity changed"):
        assert_frozen_connectivity(commitment, changed)
    for arguments, message in (
        ({"exact_intersection_pair_count": 1}, "exact_self_intersection"),
        ({"probes_preserved": False}, "probe_or_gap"),
        ({"replay_exact": False}, "replay"),
    ):
        values = {
            "resolution": 24,
            "exact_intersection_pair_count": 0,
            "probes_preserved": True,
            "replay_exact": True,
            **arguments,
        }
        with pytest.raises(ValueError, match=message):
            commit_stage_surface(vertices, faces, **values)


def _transition(model: nn.Linear, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    sample = torch.rand(5, 3)
    numpy_scale = float(np.random.uniform(0.8, 1.2))
    python_shift = random.uniform(-0.1, 0.1)
    optimizer.zero_grad(set_to_none=True)
    output = model(sample).square().mean() * numpy_scale + python_shift
    output.backward()
    optimizer.step()
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def test_e25_checkpoint_restores_complete_next_step_state_bitwise() -> None:
    torch.manual_seed(17)
    np.random.seed(18)
    random.seed(19)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    _transition(model, optimizer)
    data = capture_checkpoint(
        model,
        optimizer,
        resolution=48,
        step=37,
        committed_connectivity_digest="frozen",
    )
    expected = _transition(model, optimizer)

    restored_model = nn.Linear(3, 2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=9.0)
    payload = restore_checkpoint(data, restored_model, restored_optimizer)
    observed = _transition(restored_model, restored_optimizer)
    assert payload["schema_version"] == E25_CHECKPOINT_SCHEMA
    assert payload["resolution"] == 48
    assert payload["step"] == 37
    assert payload["committed_connectivity_digest"] == "frozen"
    assert "cuda_rng_state_all" in payload
    assert torch.equal(observed, expected)


def test_e25_checkpoint_rejects_wrong_schema() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match="schema"):
        restore_checkpoint(b"not a checkpoint", model, optimizer)
