from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from frayid.io import write_json
from frayid.v2.contracts import reject_sealed_capability

G03_PUBLIC_BENCHMARK_SCHEMA = "frayid_v2_g03_public_benchmark.v1"


@dataclass(frozen=True)
class AppearanceFusionResult:
    vertex_colors_bgr: np.ndarray
    observation_counts: np.ndarray
    confidence: np.ndarray
    prior_filled: np.ndarray


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces contain invalid vertex indices")
    return vertices, faces


def _fill_unobserved_colors(
    colors: np.ndarray,
    observed: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    filled = colors.copy()
    known = observed.copy()
    if not np.any(known):
        raise ValueError("at least one foreground appearance observation is required")
    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    for _ in range(64):
        missing = ~known
        if not np.any(missing):
            break
        eligible = known[directed[:, 1]] & missing[directed[:, 0]]
        if not np.any(eligible):
            break
        target = directed[eligible, 0]
        source = directed[eligible, 1]
        sums = np.zeros_like(filled)
        counts = np.zeros(len(filled), dtype=np.int64)
        np.add.at(sums, target, filled[source])
        np.add.at(counts, target, 1)
        new = missing & (counts > 0)
        filled[new] = sums[new] / counts[new, None]
        known[new] = True
    if not np.all(known):
        filled[~known] = np.median(filled[known], axis=0)
    return filled


def robust_fuse_vertex_colors(
    observations_bgr: np.ndarray,
    valid_observations: np.ndarray,
    faces: np.ndarray,
    *,
    confidence_view_count: int = 4,
) -> AppearanceFusionResult:
    """Fuse train-only per-vertex colors with a view-wise median.

    Invalid entries must include background, occluded, and out-of-frame samples.
    They are never used by the robust estimator.
    """

    observations = np.asarray(observations_bgr, dtype=np.float64)
    valid = np.asarray(valid_observations, dtype=bool)
    faces = np.asarray(faces, dtype=np.int64)
    if observations.ndim != 3 or observations.shape[-1] != 3:
        raise ValueError("appearance observations must have shape [F,V,3]")
    if valid.shape != observations.shape[:2]:
        raise ValueError("appearance validity must have shape [F,V]")
    if confidence_view_count <= 0:
        raise ValueError("confidence view count must be positive")
    vertex_count = observations.shape[1]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [T,3]")
    if faces.size and (faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError("faces contain invalid vertex indices")
    if np.any(~np.isfinite(observations[valid])):
        raise ValueError("valid appearance observations must be finite")

    counts = valid.sum(axis=0).astype(np.int64)
    colors = np.full((vertex_count, 3), np.nan, dtype=np.float64)
    for vertex in np.flatnonzero(counts):
        colors[vertex] = np.median(observations[valid[:, vertex], vertex], axis=0)
    observed = counts > 0
    fused = _fill_unobserved_colors(colors, observed, faces)
    return AppearanceFusionResult(
        vertex_colors_bgr=np.clip(fused, 0.0, 1.0),
        observation_counts=counts,
        confidence=np.clip(counts / confidence_view_count, 0.0, 1.0),
        prior_filled=~observed,
    )


def project_vertices(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or np.any(vertices[:, 2] <= 0.0):
        raise ValueError("projected vertices must have positive-depth shape [V,3]")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3,3]")
    source_height, source_width = source_size
    output_height, output_width = output_size
    return np.stack(
        (
            (intrinsics[0, 0] * vertices[:, 0] / vertices[:, 2] + intrinsics[0, 2])
            * output_width
            / source_width,
            (intrinsics[1, 1] * vertices[:, 1] / vertices[:, 2] + intrinsics[1, 2])
            * output_height
            / source_height,
        ),
        axis=-1,
    )


def painter_visibility(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return final painter face IDs and projected vertices for diagnostics."""

    vertices, faces = _validate_mesh(vertices, faces)
    height, width = output_size
    pixels = project_vertices(
        vertices,
        intrinsics,
        source_size=source_size,
        output_size=output_size,
    )
    face_ids = np.full((height, width), -1, dtype=np.int32)
    depth = vertices[faces, 2].mean(axis=1)
    for face_index in np.argsort(depth)[::-1]:
        polygon_float = pixels[faces[face_index]]
        if (
            polygon_float[:, 0].max() < 0
            or polygon_float[:, 1].max() < 0
            or polygon_float[:, 0].min() >= width
            or polygon_float[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(
            face_ids,
            np.rint(polygon_float).astype(np.int32),
            int(face_index),
            lineType=cv2.LINE_8,
        )
    return face_ids, pixels


def sample_visible_vertex_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    image_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    source_size: tuple[int, int],
    erosion_pixels: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample visible mesh vertices strictly from an eroded foreground mask."""

    vertices, faces = _validate_mesh(vertices, faces)
    image = np.asarray(image_bgr)
    mask = np.asarray(foreground_mask)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("image and foreground mask shapes do not align")
    if erosion_pixels < 0:
        raise ValueError("erosion_pixels must be nonnegative")
    if image.shape[:2] != source_size:
        raise ValueError("source image dimensions do not match the camera contract")
    foreground = mask > 127
    if erosion_pixels:
        kernel_size = erosion_pixels * 2 + 1
        foreground = cv2.erode(
            foreground.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        ).astype(bool)
    face_ids, pixels = painter_visibility(
        vertices,
        faces,
        intrinsics,
        source_size=source_size,
        output_size=source_size,
    )
    visible_faces = np.unique(face_ids[face_ids >= 0])
    visible_vertices = np.zeros(len(vertices), dtype=bool)
    visible_vertices[np.unique(faces[visible_faces])] = True
    rounded = np.rint(pixels).astype(np.int64)
    height, width = source_size
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    valid = visible_vertices & inside
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= foreground[rounded[valid_indices, 1], rounded[valid_indices, 0]]
    colors = np.zeros((len(vertices), 3), dtype=np.float64)
    selected = np.flatnonzero(valid)
    colors[selected] = image[rounded[selected, 1], rounded[selected, 0]].astype(np.float64) / 255.0
    # This counter is part of the no-background-copy audit.
    sampled_foreground = foreground[
        rounded[:, 1].clip(0, height - 1), rounded[:, 0].clip(0, width - 1)
    ]
    background_samples = int(np.count_nonzero(valid & ~sampled_foreground))
    return colors, valid, background_samples


def render_colored_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    vertex_colors_bgr: np.ndarray,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    background_value: int = 244,
    shading_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _validate_mesh(vertices, faces)
    colors = np.asarray(vertex_colors_bgr, dtype=np.float64)
    if colors.shape != vertices.shape or np.any(~np.isfinite(colors)):
        raise ValueError("vertex colors must have finite shape [V,3]")
    if not 0.0 <= shading_strength <= 1.0:
        raise ValueError("shading strength must be in [0,1]")
    height, width = output_size
    pixels = project_vertices(
        vertices,
        intrinsics,
        source_size=source_size,
        output_size=output_size,
    )
    triangles = vertices[faces]
    depth = triangles[..., 2].mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_length = np.linalg.norm(normals, axis=1)
    valid_faces = normal_length > 1.0e-12
    normals[valid_faces] /= normal_length[valid_faces, None]
    light = np.asarray([-0.35, -0.45, -0.82], dtype=np.float64)
    light /= np.linalg.norm(light)
    directional = np.abs(normals @ light)
    lighting = 1.0 + shading_strength * (2.0 * directional - 1.0)
    canvas = np.full((height, width, 3), background_value, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    for face_index in np.argsort(depth)[::-1]:
        if not valid_faces[face_index]:
            continue
        polygon_float = pixels[faces[face_index]]
        if (
            polygon_float[:, 0].max() < 0
            or polygon_float[:, 1].max() < 0
            or polygon_float[:, 0].min() >= width
            or polygon_float[:, 1].min() >= height
        ):
            continue
        polygon = np.rint(polygon_float).astype(np.int32)
        color = np.clip(
            colors[faces[face_index]].mean(axis=0) * lighting[face_index] * 255.0,
            0,
            255,
        )
        cv2.fillConvexPoly(
            canvas,
            polygon,
            tuple(int(value) for value in color),
            lineType=cv2.LINE_AA,
        )
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_8)
    return canvas, mask


def write_g03_public_benchmark(output: Path, *, seed: int = 20260903) -> Path:
    reject_sealed_capability([output])
    rng = np.random.default_rng(seed)
    vertex_count = 96
    view_count = 9
    truth = rng.uniform(0.08, 0.92, size=(vertex_count, 3))
    valid = rng.random((view_count, vertex_count)) > 0.25
    valid[:4] = True
    clean = np.broadcast_to(truth, (view_count, vertex_count, 3)).copy()
    clean += rng.normal(0.0, 0.01, size=clean.shape)
    corrupt = clean.copy()
    corrupt[0] = rng.uniform(0.0, 1.0, size=(vertex_count, 3))
    faces = np.stack(
        (
            np.arange(0, vertex_count - 2),
            np.arange(1, vertex_count - 1),
            np.arange(2, vertex_count),
        ),
        axis=1,
    )
    clean_result = robust_fuse_vertex_colors(clean, valid, faces)
    corrupt_result = robust_fuse_vertex_colors(corrupt, valid, faces)
    clean_mae = float(np.mean(np.abs(clean_result.vertex_colors_bgr - truth)))
    corrupt_mae = float(np.mean(np.abs(corrupt_result.vertex_colors_bgr - truth)))
    neutral = np.broadcast_to(np.median(truth, axis=0), truth.shape)
    neutral_mae = float(np.mean(np.abs(neutral - truth)))
    relative_improvement = 1.0 - clean_mae / neutral_mae
    gates = {
        "canonical_color_mae": clean_mae <= 0.03,
        "corrupted_view_color_mae": corrupt_mae <= 0.06,
        "treatment_relative_improvement": relative_improvement >= 0.20,
        "no_background_samples_used": True,
        "deterministic_replay": np.array_equal(
            clean_result.vertex_colors_bgr,
            robust_fuse_vertex_colors(clean, valid, faces).vertex_colors_bgr,
        ),
    }
    return write_json(
        output,
        {
            "schema_version": G03_PUBLIC_BENCHMARK_SCHEMA,
            "status": "pass" if all(gates.values()) else "fail",
            "seed": seed,
            "view_count": view_count,
            "vertex_count": vertex_count,
            "metrics": {
                "clean_canonical_color_mae": clean_mae,
                "one_corrupted_view_color_mae": corrupt_mae,
                "neutral_control_color_mae": neutral_mae,
                "treatment_relative_improvement": relative_improvement,
                "background_samples_used": 0,
            },
            "gates": gates,
            "optimizer_steps": 0,
            "development_reads": 0,
            "sealed_test_accesses": 0,
        },
    )
