from __future__ import annotations

import torch

from frayid.camera import axis_angle_to_matrix
from frayid.v2.t03_dynamic import (
    sequence_regularized_dynamic_prediction,
    t03_internal_slots,
)


def test_t03_split_uses_new_fold_and_excludes_t02_fold() -> None:
    fit, validation, excluded = t03_internal_slots(144)
    assert (fit.numel(), validation.numel(), excluded.numel()) == (86, 29, 29)
    assert validation.tolist()[:3] == [1, 6, 11]
    assert excluded.tolist()[:3] == [0, 5, 10]
    assert set(fit.tolist()).isdisjoint(validation.tolist())
    assert set(fit.tolist()).isdisjoint(excluded.tolist())


def test_t03_prediction_is_bounded_and_retains_initialization() -> None:
    sources = torch.arange(20, dtype=torch.float32)
    fit, validation, _ = t03_internal_slots(20)
    angles = 0.04 * sources
    rotations = axis_angle_to_matrix(
        torch.stack((torch.zeros_like(angles), angles, torch.zeros_like(angles)), dim=-1)
    )
    translations = torch.stack(
        (0.01 * sources, torch.zeros_like(sources), torch.full_like(sources, 2.2)), dim=-1
    )
    prediction = sequence_regularized_dynamic_prediction(
        sources[fit],
        rotations[fit],
        translations[fit],
        sources[validation],
        rotations[validation],
        translations[validation],
    )
    assert float(prediction.rotation_correction_degrees.max()) <= 2.0
    assert float(prediction.translation_correction_metres.max()) <= 0.02
    assert torch.isfinite(prediction.rotations).all()
    assert torch.isfinite(prediction.translations).all()
