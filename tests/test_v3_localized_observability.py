from __future__ import annotations

import numpy as np

from frayid.v3.localized_observability import (
    localized_descriptor_tensor,
    public_localized_controls_pass,
    qualify_localized_descriptors,
)


def _features(phases: np.ndarray) -> np.ndarray:
    values = [function(order * phases) for order in range(1, 7) for function in (np.sin, np.cos)]
    return np.stack(values, axis=1).reshape(len(phases), 1, 3, 4)


def test_public_localized_controls_pass_before_private_read() -> None:
    assert public_localized_controls_pass()


def test_reversed_localized_sequence_fails_separation() -> None:
    phases = np.linspace(0.0, 2.0 * np.pi, 12)
    result = qualify_localized_descriptors(_features(phases), _features(phases[::-1]))
    assert "control_cost_margin_below_0_30" in result["blockers"]


def test_localized_descriptor_uses_dynamic_subject_bbox() -> None:
    frames = np.zeros((12, 54, 96), dtype=np.float32)
    for index in range(len(frames)):
        left = 35 + index % 4
        frames[index, 10:48, left : left + 20] = 0.4 + 0.03 * index
    mask = np.std(frames, axis=0) > 0.01
    tensor, bbox = localized_descriptor_tensor(frames, mask)
    assert tensor.shape == (12, 6, 48, 48)
    assert bbox[0] > 0
    assert bbox[2] < 96
    assert np.isfinite(tensor).all()
