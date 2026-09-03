from __future__ import annotations

import pytest

from frayid.renderer import render_soft_mesh
from frayid.renderer_contract import (
    RendererBackend,
    create_training_renderer,
    renderer_contract,
    require_legacy_evaluator_backend,
)


def test_renderer_contract_keeps_legacy_and_opaque_reports_separate() -> None:
    legacy = renderer_contract(RendererBackend.LEGACY_SOFT_SPLAT)
    opaque = renderer_contract(RendererBackend.OPAQUE_NVDIFFRAST)
    assert legacy.legacy_evaluation_permitted is True
    assert legacy.opaque_visibility is False
    assert opaque.legacy_evaluation_permitted is False
    assert opaque.opaque_visibility is True
    assert legacy.report_namespace != opaque.report_namespace
    assert create_training_renderer(RendererBackend.LEGACY_SOFT_SPLAT) is render_soft_mesh
    require_legacy_evaluator_backend(RendererBackend.LEGACY_SOFT_SPLAT)
    with pytest.raises(ValueError, match="cannot replace legacy"):
        require_legacy_evaluator_backend(RendererBackend.OPAQUE_NVDIFFRAST)
