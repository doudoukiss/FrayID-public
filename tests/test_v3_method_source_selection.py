from __future__ import annotations

import numpy as np

from frayid.v3.method_source_selection import (
    _audit_interval,
    _dynamic_mask,
    _SampledSequence,
    _stable_windows,
    synthetic_cycle_features,
)


def test_public_repeated_cycle_control_passes_interval_gates() -> None:
    times, descriptors, motion = synthetic_cycle_features(
        repeat_second_cycle=True,
        include_holds=True,
    )
    sequence = _SampledSequence(
        times=times,
        frames=np.zeros((len(times), 2, 2), dtype=np.float32),
        decoded_source_frame_count=len(times),
    )
    first = _audit_interval(
        sequence,
        descriptors,
        motion,
        start_seconds=0.0,
        end_seconds=12.0,
        stable_threshold=0.1,
    )
    second = _audit_interval(
        sequence,
        descriptors,
        motion,
        start_seconds=12.2,
        end_seconds=24.2,
        stable_threshold=0.1,
    )
    assert first.status == "pass"
    assert second.status == "pass"
    assert first.closure_ratio < 0.01
    assert len(first.stable_windows) >= 8


def test_continuous_motion_control_fails_stable_window_gate() -> None:
    times, descriptors, motion = synthetic_cycle_features(
        repeat_second_cycle=True,
        include_holds=False,
    )
    sequence = _SampledSequence(
        times=times,
        frames=np.zeros((len(times), 2, 2), dtype=np.float32),
        decoded_source_frame_count=len(times),
    )
    audit = _audit_interval(
        sequence,
        descriptors,
        motion,
        start_seconds=0.0,
        end_seconds=12.0,
        stable_threshold=0.1,
    )
    assert audit.status == "fail"
    assert "stable_window_count_below_8" in audit.blockers


def test_dynamic_mask_rejects_static_input_and_selects_varying_pixels() -> None:
    static = np.zeros((10, 8, 8), dtype=np.float32)
    try:
        _dynamic_mask(static)
    except ValueError as exc:
        assert "insufficient dynamic image support" in str(exc)
    else:
        raise AssertionError("static source must fail dynamic support")

    changing = static.copy()
    changing[:, :, :] = np.linspace(0.0, 1.0, 10)[:, None, None]
    mask = _dynamic_mask(changing)
    assert mask.shape == (8, 8)
    assert np.all(mask)


def test_stable_windows_require_minimum_contiguous_duration() -> None:
    times = np.arange(20, dtype=np.float64) / 5.0
    motion = np.full(20, 0.4)
    motion[2:6] = 0.01
    motion[10:12] = 0.01
    windows = _stable_windows(times, motion, threshold=0.1)
    assert len(windows) == 1
    assert windows[0].start_seconds == 0.4
    assert windows[0].end_seconds == 1.0
