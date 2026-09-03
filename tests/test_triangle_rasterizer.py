from __future__ import annotations

import pytest
import torch
import trimesh

from frayid.camera import make_intrinsics, project_points, resize_intrinsics
from frayid.normal_residual import DiffusedNormalResidual
from frayid.renderer_determinism import cpu_reference_trace, first_bitwise_tensor_difference
from frayid.triangle_rasterizer import (
    NvdiffrastRenderer,
    clip_to_opencv_pixels,
    opencv_camera_to_clip,
    opencv_faces_to_opengl,
    rasterize_reference,
)


def test_opencv_clip_projection_preserves_pixel_centers() -> None:
    vertices = torch.tensor([[-0.4, 0.2, 2.0], [0.5, -0.3, 3.0]])
    intrinsics = make_intrinsics(40.0, (32.0, 32.0))
    clip = opencv_camera_to_clip(vertices, intrinsics, (64, 64), source_image_size=(64, 64))
    torch.testing.assert_close(
        clip_to_opencv_pixels(clip, (64, 64)), project_points(vertices, intrinsics)
    )
    assert bool(torch.all(clip[:, 2] / clip[:, 3] > -1.0))
    assert bool(torch.all(clip[:, 2] / clip[:, 3] < 1.0))


def test_off_centre_intrinsics_use_explicit_source_dimensions() -> None:
    vertices = torch.tensor([[-0.4, 0.2, 2.0], [0.5, -0.3, 3.0]])
    intrinsics = make_intrinsics(800.0, (231.0, 317.0))
    source_size = (720, 1120)
    output_size = (144, 224)
    clip = opencv_camera_to_clip(
        vertices,
        intrinsics,
        output_size,
        source_image_size=source_size,
    )
    expected = project_points(vertices, resize_intrinsics(intrinsics, source_size, output_size))
    torch.testing.assert_close(clip_to_opencv_pixels(clip, output_size), expected)


def test_opencv_to_opengl_reflection_reverses_face_winding() -> None:
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    torch.testing.assert_close(opencv_faces_to_opengl(faces), torch.tensor([[0, 2, 1], [3, 5, 4]]))


def test_reference_rasterizer_uses_nearest_surface_and_sapiens_axes() -> None:
    vertices = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.0, 0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.7, -0.7, 3.0],
            [0.7, -0.7, 3.0],
            [0.0, 0.7, 3.0],
        ]
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    mask, normals, depth = rasterize_reference(
        vertices,
        faces,
        make_intrinsics(40.0, (16.0, 16.0)),
        (32, 32),
        source_image_size=(32, 32),
    )
    assert float(mask[16, 16]) == 1.0
    assert float(depth[16, 16]) == 2.0
    # OpenCV -z points toward the viewer and maps to Sapiens2 +z.
    assert float(normals[16, 16, 2]) > 0.99


def test_reference_rasterizer_has_exact_triangle_coverage() -> None:
    vertices = torch.tensor([[-0.4, -0.4, 2.0], [0.4, -0.4, 2.0], [0.0, 0.4, 2.0]])
    faces = torch.tensor([[0, 1, 2]])
    mask, _, _ = rasterize_reference(
        vertices,
        faces,
        make_intrinsics(40.0, (16.0, 16.0)),
        (32, 32),
        source_image_size=(32, 32),
    )
    assert 100 < int(mask.sum()) < 150
    assert float(mask[16, 16]) == 1.0
    assert float(mask[2, 2]) == 0.0


def test_cpu_reference_trace_is_bitwise_repeatable_and_reports_first_difference() -> None:
    vertices = torch.tensor([[-0.4, -0.4, 2.0], [0.4, -0.4, 2.0], [0.0, 0.4, 2.0]])
    faces = torch.tensor([[0, 1, 2]])
    intrinsics = make_intrinsics(40.0, (16.0, 16.0))
    reference = cpu_reference_trace(
        vertices,
        faces,
        intrinsics,
        (32, 32),
        source_image_size=(32, 32),
    )
    for _ in range(100):
        repeated = cpu_reference_trace(
            vertices,
            faces,
            intrinsics,
            (32, 32),
            source_image_size=(32, 32),
        )
        assert first_bitwise_tensor_difference(reference, repeated) is None
    changed = {name: value.clone() for name, value in reference.items()}
    changed["interpolated_normals"].reshape(-1)[17] += 1.0
    difference = first_bitwise_tensor_difference(reference, changed)
    assert difference is not None
    assert difference.name == "interpolated_normals"
    assert difference.first_flat_index == 17


@pytest.mark.skipif(not torch.cuda.is_available(), reason="nvdiffrast requires CUDA")
def test_exact_rasterizer_backpropagates_to_diffused_normal_residual() -> None:
    source = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    vertices = torch.tensor(source.vertices, dtype=torch.float32, device="cuda")
    vertices[:, 2] += 2.5
    faces = torch.tensor(source.faces, dtype=torch.long, device="cuda")
    field = DiffusedNormalResidual(vertices, faces, diffusion_steps=2).cuda()
    renderer = NvdiffrastRenderer()
    intrinsics = make_intrinsics(80.0, (32.0, 32.0), device="cuda")
    with torch.no_grad():
        field.raw_displacements.fill_(0.08)
        target, _ = renderer(
            field.deformed_vertices(),
            faces,
            intrinsics,
            (64, 64),
            source_image_size=(64, 64),
        )
        field.raw_displacements.zero_()
    predicted, _ = renderer(
        field.deformed_vertices(),
        faces,
        intrinsics,
        (64, 64),
        source_image_size=(64, 64),
    )
    trace = renderer.diagnostic_trace(
        field.deformed_vertices(),
        faces,
        intrinsics,
        (64, 64),
        source_image_size=(64, 64),
        final_parameter_inputs={"final_cage_parameters": field.raw_displacements},
    )
    assert trace.raster.shape[-1] == 4
    assert trace.point_sampled_coverage.shape[-1] == 1
    for name in ("geometry", "interpolated_attributes", "final_cage_parameters"):
        assert trace.gradients[name] is not None
        assert bool(torch.isfinite(trace.gradients[name]).all())  # type: ignore[union-attr]
    loss = (predicted - target).square().mean()
    loss.backward()  # type: ignore[no-untyped-call]
    gradient = field.raw_displacements.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0
    with torch.no_grad():
        field.raw_displacements -= 0.01 * gradient / gradient.abs().max().clamp_min(1e-8)
    improved, _ = renderer(
        field.deformed_vertices(),
        faces,
        intrinsics,
        (64, 64),
        source_image_size=(64, 64),
    )
    improved_loss = (improved - target).square().mean()
    assert float(improved_loss) < float(loss)
