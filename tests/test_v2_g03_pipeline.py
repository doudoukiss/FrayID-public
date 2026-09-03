from __future__ import annotations

import numpy as np

from frayid.v2.g03_pipeline import foreground_crop_ssim, foreground_rgb_mae


def test_g03_foreground_metrics_prefer_matching_color() -> None:
    target = np.full((16, 16, 3), 244, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:13, 4:12] = True
    target[mask] = (30, 120, 220)
    treatment = target.copy()
    control = np.full_like(target, 244)
    control[mask] = (120, 160, 180)
    assert foreground_rgb_mae(target, treatment, mask) == 0.0
    assert foreground_rgb_mae(target, control, mask) > 0.1
    assert foreground_crop_ssim(target, treatment, mask) == 1.0
    assert foreground_crop_ssim(target, treatment, mask) > foreground_crop_ssim(
        target, control, mask
    )
