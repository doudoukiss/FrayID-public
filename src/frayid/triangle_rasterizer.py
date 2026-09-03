from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from frayid.camera import project_points, resize_intrinsics
from frayid.geometry import vertex_normals


@dataclass(frozen=True)
class OpaqueRendererTrace:
    """Named stage tensors and diagnostic gradients from one opaque render."""

    clip_vertices: Tensor
    triangle_indices: Tensor
    raster: Tensor
    raster_derivatives: Tensor
    point_sampled_coverage: Tensor
    interpolated_normals: Tensor
    antialiased_coverage: Tensor
    antialiased_normals: Tensor
    project_silhouette: Tensor
    project_normals: Tensor
    gradients: dict[str, Tensor | None]

    def tensors(self) -> dict[str, Tensor]:
        values = {
            "clip_vertices": self.clip_vertices,
            "triangle_indices": self.triangle_indices,
            "raster": self.raster,
            "raster_derivatives": self.raster_derivatives,
            "point_sampled_coverage": self.point_sampled_coverage,
            "interpolated_normals": self.interpolated_normals,
            "antialiased_coverage": self.antialiased_coverage,
            "antialiased_normals": self.antialiased_normals,
            "project_silhouette": self.project_silhouette,
            "project_normals": self.project_normals,
        }
        values.update(
            {
                f"gradient:{name}": value
                for name, value in self.gradients.items()
                if value is not None
            }
        )
        return values


def opencv_camera_to_clip(
    vertices_camera: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    source_image_size: tuple[int, int],
    near: float = 0.01,
    far: float = 100.0,
) -> Tensor:
    """Map OpenCV camera coordinates to OpenGL clip coordinates.

    The project uses pixel-center coordinates with a top-left image origin.
    nvdiffrast consumes OpenGL clip coordinates and produces bottom-up image
    tensors, so :class:`NvdiffrastRenderer` flips the rendered image exactly
    once when returning to the project contract.
    """
    if vertices_camera.ndim != 2 or vertices_camera.shape[-1] != 3:
        raise ValueError("vertices_camera must have shape [V, 3]")
    if near <= 0 or far <= near:
        raise ValueError("clip planes must satisfy 0 < near < far")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if torch.any(vertices_camera[:, 2] <= 0):
        raise ValueError("all rasterized vertices must be in front of the camera")

    render_intrinsics = resize_intrinsics(intrinsics, source_image_size, image_size)
    pixels = project_points(vertices_camera, render_intrinsics)
    ndc_x = 2.0 * (pixels[:, 0] + 0.5) / width - 1.0
    ndc_y = 1.0 - 2.0 * (pixels[:, 1] + 0.5) / height
    z = vertices_camera[:, 2]
    depth_scale = (far + near) / (far - near)
    depth_offset = -2.0 * far * near / (far - near)
    return torch.stack((ndc_x * z, ndc_y * z, depth_scale * z + depth_offset, z), dim=-1)


def clip_to_opencv_pixels(clip_vertices: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Invert the clip-space x/y mapping for contract tests and CPU reference."""
    height, width = image_size
    ndc = clip_vertices[:, :2] / clip_vertices[:, 3:4]
    x = (ndc[:, 0] + 1.0) * width / 2.0 - 0.5
    y = (1.0 - ndc[:, 1]) * height / 2.0 - 0.5
    return torch.stack((x, y), dim=-1)


def opencv_faces_to_opengl(faces: Tensor) -> Tensor:
    """Reverse winding after the OpenCV-down to OpenGL-up axis reflection."""
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    return faces[:, [0, 2, 1]]


def rasterize_reference(
    vertices_camera: Tensor,
    faces: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    source_image_size: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor]:
    """Small exact CPU z-buffer used only as a forward-contract oracle.

    This implementation intentionally has no differentiability claim. It is
    suitable for tiny synthetic fixtures, not project-data evaluation.
    """
    if vertices_camera.device.type != "cpu" or faces.device.type != "cpu":
        raise ValueError("reference rasterization is CPU-only")
    height, width = image_size
    clip = opencv_camera_to_clip(
        vertices_camera,
        intrinsics,
        image_size,
        source_image_size=source_image_size,
    )
    pixels = clip_to_opencv_pixels(clip, image_size)
    normals = vertex_normals(vertices_camera, faces) * vertices_camera.new_tensor((1.0, -1.0, -1.0))
    depth = vertices_camera.new_full((height, width), float("inf"))
    triangle_ids = torch.full((height, width), -1, dtype=torch.long)
    normal_image = vertices_camera.new_zeros((height, width, 3))
    epsilon = 1e-7
    for face_index, face in enumerate(faces):
        triangle = pixels[face]
        minimum = torch.floor(triangle.amin(dim=0)).to(torch.long)
        maximum = torch.ceil(triangle.amax(dim=0)).to(torch.long)
        x0 = max(0, int(minimum[0]))
        x1 = min(width - 1, int(maximum[0]))
        y0 = max(0, int(minimum[1]))
        y1 = min(height - 1, int(maximum[1]))
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(float(denominator)) <= epsilon:
            continue
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                point = triangle.new_tensor((float(x), float(y)))
                first = (
                    (b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])
                ) / denominator
                second = (
                    (c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])
                ) / denominator
                third = 1.0 - first - second
                barycentric = torch.stack((first, second, third))
                if bool(torch.any(barycentric < -epsilon)):
                    continue
                reciprocal_depth = (barycentric / vertices_camera[face, 2].clamp_min(epsilon)).sum()
                candidate_depth = 1.0 / reciprocal_depth.clamp_min(epsilon)
                if candidate_depth >= depth[y, x]:
                    continue
                perspective = (
                    barycentric / vertices_camera[face, 2].clamp_min(epsilon)
                ) * candidate_depth
                interpolated = (normals[face] * perspective[:, None]).sum(dim=0)
                depth[y, x] = candidate_depth
                triangle_ids[y, x] = face_index
                normal_image[y, x] = torch.nn.functional.normalize(interpolated, dim=0, eps=1e-8)
    return (triangle_ids >= 0).to(vertices_camera.dtype), normal_image, depth


class NvdiffrastRenderer:
    """Exact triangle coverage/depth renderer with analytic edge gradients."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("nvdiffrast audit requires CUDA")
        try:
            import nvdiffrast.torch as dr  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("nvdiffrast is not installed") from error
        self._dr: Any = dr
        self._context: Any = dr.RasterizeCudaContext()

    def __call__(
        self,
        vertices_camera: Tensor,
        faces: Tensor,
        intrinsics: Tensor,
        image_size: tuple[int, int],
        *,
        source_image_size: tuple[int, int],
        **_: object,
    ) -> tuple[Tensor, Tensor]:
        clip, triangles, rast, _rast_db, coverage, interpolated = self._point_sampled_buffers(
            vertices_camera, faces, intrinsics, image_size, source_image_size
        )
        coverage = self._dr.antialias(coverage, rast, clip.unsqueeze(0), triangles)
        interpolated = self._dr.antialias(interpolated, rast, clip.unsqueeze(0), triangles)
        return self._to_project_image(coverage, interpolated)

    def render_point_sampled(
        self,
        vertices_camera: Tensor,
        faces: Tensor,
        intrinsics: Tensor,
        image_size: tuple[int, int],
        *,
        source_image_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        """Return raw point-sampled coverage for a CPU/CUDA contract audit."""
        _, _, _, _, coverage, interpolated = self._point_sampled_buffers(
            vertices_camera, faces, intrinsics, image_size, source_image_size
        )
        return self._to_project_image(coverage, interpolated)

    def diagnostic_trace(
        self,
        vertices_camera: Tensor,
        faces: Tensor,
        intrinsics: Tensor,
        image_size: tuple[int, int],
        *,
        source_image_size: tuple[int, int],
        final_parameter_inputs: dict[str, Tensor] | None = None,
    ) -> OpaqueRendererTrace:
        """Materialize every P1 stage and gradients without changing normal rendering."""
        if not vertices_camera.requires_grad:
            raise ValueError("diagnostic geometry must require gradients")
        clip, triangles, rast, rast_db, coverage, interpolated = self._point_sampled_buffers(
            vertices_camera,
            faces,
            intrinsics,
            image_size,
            source_image_size,
            grad_db=True,
        )
        coverage_aa = self._dr.antialias(coverage, rast, clip.unsqueeze(0), triangles)
        normals_aa = self._dr.antialias(interpolated, rast, clip.unsqueeze(0), triangles)
        silhouette, normal_image = self._to_project_image(coverage_aa, normals_aa)
        objective = silhouette.square().mean() + normal_image.square().mean()
        named_inputs: dict[str, Tensor] = {
            "geometry": vertices_camera,
            "interpolated_attributes": interpolated,
        }
        named_inputs.update(final_parameter_inputs or {})
        active = {name: value for name, value in named_inputs.items() if value.requires_grad}
        computed = torch.autograd.grad(
            objective,
            tuple(active.values()),
            retain_graph=True,
            allow_unused=True,
        )
        gradients = dict.fromkeys(named_inputs)
        gradients.update(dict(zip(active, computed, strict=True)))
        return OpaqueRendererTrace(
            clip_vertices=clip,
            triangle_indices=triangles,
            raster=rast,
            raster_derivatives=rast_db,
            point_sampled_coverage=coverage,
            interpolated_normals=interpolated,
            antialiased_coverage=coverage_aa,
            antialiased_normals=normals_aa,
            project_silhouette=silhouette,
            project_normals=normal_image,
            gradients=gradients,
        )

    def _point_sampled_buffers(
        self,
        vertices_camera: Tensor,
        faces: Tensor,
        intrinsics: Tensor,
        image_size: tuple[int, int],
        source_image_size: tuple[int, int],
        *,
        grad_db: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if vertices_camera.device.type != "cuda":
            raise ValueError("nvdiffrast renderer requires CUDA tensors")
        clip = opencv_camera_to_clip(
            vertices_camera,
            intrinsics,
            image_size,
            source_image_size=source_image_size,
        ).contiguous()
        triangles = opencv_faces_to_opengl(faces).to(dtype=torch.int32).contiguous()
        rast, rast_db = self._dr.rasterize(
            self._context,
            clip.unsqueeze(0),
            triangles,
            resolution=list(image_size),
            grad_db=grad_db,
        )
        coverage = (rast[..., 3:4] > 0).to(vertices_camera.dtype)
        normals = vertex_normals(vertices_camera, faces) * vertices_camera.new_tensor(
            (1.0, -1.0, -1.0)
        )
        interpolated, _ = self._dr.interpolate(normals.unsqueeze(0), rast, triangles)
        interpolated = interpolated * (rast[..., 3:4] > 0)
        return clip, triangles, rast, rast_db, coverage, interpolated

    @staticmethod
    def _to_project_image(coverage: Tensor, interpolated: Tensor) -> tuple[Tensor, Tensor]:
        # nvdiffrast tensors use a bottom-up image convention.
        silhouette = torch.flip(coverage[0, ..., 0], dims=(0,))
        normal_image = torch.flip(interpolated[0], dims=(0,))
        normal_image = torch.nn.functional.normalize(normal_image, dim=-1, eps=1e-8)
        return silhouette, normal_image
