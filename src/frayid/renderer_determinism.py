from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from frayid.triangle_rasterizer import (
    clip_to_opencv_pixels,
    opencv_camera_to_clip,
    rasterize_reference,
)


@dataclass(frozen=True)
class TensorDifference:
    name: str
    shape_equal: bool
    dtype_equal: bool
    first_flat_index: int | None
    maximum_absolute_difference: float | None


def first_bitwise_tensor_difference(
    reference: Mapping[str, Tensor], candidate: Mapping[str, Tensor]
) -> TensorDifference | None:
    """Return the first stable-name tensor difference for a P1 trace."""
    names = sorted(set(reference) | set(candidate))
    for name in names:
        if name not in reference or name not in candidate:
            return TensorDifference(name, False, False, None, None)
        expected = reference[name].detach().cpu().contiguous()
        actual = candidate[name].detach().cpu().contiguous()
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return TensorDifference(
                name,
                expected.shape == actual.shape,
                expected.dtype == actual.dtype,
                None,
                None,
            )
        equal = torch.eq(expected, actual)
        if bool(torch.all(equal)):
            continue
        first = int(torch.nonzero(~equal.reshape(-1), as_tuple=False)[0, 0])
        maximum = (
            float(torch.max(torch.abs(expected - actual))) if expected.is_floating_point() else None
        )
        return TensorDifference(name, True, True, first, maximum)
    return None


def cpu_reference_trace(
    vertices_camera: Tensor,
    faces: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    source_image_size: tuple[int, int],
) -> dict[str, Tensor]:
    """Materialize the deterministic CPU forward contract; no gradient claim."""
    clip = opencv_camera_to_clip(
        vertices_camera,
        intrinsics,
        image_size,
        source_image_size=source_image_size,
    )
    silhouette, normals, depth = rasterize_reference(
        vertices_camera,
        faces,
        intrinsics,
        image_size,
        source_image_size=source_image_size,
    )
    return {
        "clip_vertices": clip,
        "opencv_pixels": clip_to_opencv_pixels(clip, image_size),
        "point_sampled_coverage": silhouette,
        "interpolated_normals": normals,
        "depth": depth,
    }
