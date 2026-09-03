from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frayid.flexicubes_adapter import (
    FLEXICUBES_REVISION,
    PinnedFlexiCubes,
    validate_flexicubes_repository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "external/FlexiCubes"


def _require_upstream() -> None:
    if not UPSTREAM.is_dir():
        pytest.skip("non-vendored official FlexiCubes checkout is unavailable")


def test_pinned_flexicubes_extracts_deterministically_and_backpropagates() -> None:
    _require_upstream()
    validate_flexicubes_repository(UPSTREAM, expected_revision=FLEXICUBES_REVISION)
    adapter = PinnedFlexiCubes(UPSTREAM, device="cpu")
    vertices, cubes = adapter.voxel_grid(8, extent=1.0)
    values = (torch.linalg.vector_norm(vertices, dim=-1) - 0.55).detach().requires_grad_(True)
    first = adapter.extract(vertices, values, cubes, 8, training=True)
    second = adapter.extract(vertices, values, cubes, 8, training=True)
    torch.testing.assert_close(first.vertices, second.vertices, rtol=0.0, atol=0.0)
    assert torch.equal(first.faces, second.faces)
    assert first.faces.shape[0] > 100
    loss = first.vertices.square().mean() + first.developability.mean()
    loss.backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


def test_flexicubes_adapter_rejects_wrong_revision_and_empty_surface() -> None:
    _require_upstream()
    with pytest.raises(RuntimeError, match="registered revision"):
        validate_flexicubes_repository(UPSTREAM, expected_revision="0" * 40)
    adapter = PinnedFlexiCubes(UPSTREAM, device="cpu")
    vertices, cubes = adapter.voxel_grid(4, extent=1.0)
    with pytest.raises(ValueError, match="no extractable surface"):
        adapter.extract(
            vertices,
            torch.ones(vertices.shape[0]),
            cubes,
            4,
            training=False,
        )
