from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from frayid.embedded_carrier import (
    EMBEDDED_CARRIER_SCHEMA,
    build_barycentric_transfer,
    embedded_surface_fidelity,
    interpolate_vertex_field,
    read_embedded_carrier,
    write_embedded_carrier,
)


def _two_layer_source() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    upper = lower.copy()
    upper[:, 2] = 0.02
    vertices = np.vstack((lower, upper))
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    weights = np.zeros((8, 2), dtype=np.float64)
    weights[:4, 0] = 1.0
    weights[4:, 1] = 1.0
    return vertices, faces, weights


def test_barycentric_transfer_preserves_weights_and_selects_nearest_layer() -> None:
    vertices, faces, weights = _two_layer_source()
    targets = np.asarray([[0.25, 0.25, 0.001], [-0.4, 0.2, 0.019]])
    transfer = build_barycentric_transfer(
        vertices,
        faces,
        weights,
        targets,
        target_normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        pitch=0.1,
    )
    np.testing.assert_allclose(transfer.weights, np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(transfer.source_barycentrics.sum(axis=1), 1.0)
    assert not bool(np.any(transfer.ambiguous))


def test_barycentric_transfer_marks_equal_distance_wrong_layer_ambiguity() -> None:
    vertices, faces, weights = _two_layer_source()
    transfer = build_barycentric_transfer(
        vertices,
        faces,
        weights,
        np.asarray([[0.0, 0.0, 0.01]]),
        target_normals=np.asarray([[0.0, 0.0, 1.0]]),
        pitch=0.1,
    )
    assert bool(transfer.ambiguous[0])


def test_wrong_layer_fixture_selects_at_least_995_percent() -> None:
    vertices, faces, weights = _two_layer_source()
    axis = np.linspace(-0.8, 0.8, 20)
    xy = np.asarray([(x, y) for x in axis for y in axis], dtype=np.float64)
    lower = np.column_stack((xy, np.full(len(xy), 0.001)))
    upper = np.column_stack((xy, np.full(len(xy), 0.019)))
    targets = np.vstack((lower, upper))
    transfer = build_barycentric_transfer(
        vertices,
        faces,
        weights,
        targets,
        target_normals=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(targets), 1)),
        pitch=0.1,
    )
    predicted = np.argmax(transfer.weights, axis=1)
    expected = np.concatenate(
        (np.zeros(len(lower), dtype=np.int64), np.ones(len(upper), dtype=np.int64))
    )
    assert float(np.mean(predicted == expected)) >= 0.995


def test_residual_field_uses_the_same_frozen_triangle_map() -> None:
    vertices, faces, weights = _two_layer_source()
    target = np.asarray([[0.25, 0.25, 0.001]])
    transfer = build_barycentric_transfer(
        vertices,
        faces,
        weights,
        target,
        target_normals=np.asarray([[0.0, 0.0, 1.0]]),
        pitch=0.1,
    )
    residuals = np.column_stack((vertices[:, 0], vertices[:, 1], vertices[:, 2]))
    mapped = interpolate_vertex_field(
        residuals,
        faces,
        transfer.source_face_indices,
        transfer.source_barycentrics,
    )
    np.testing.assert_allclose(mapped[0], np.asarray([0.25, 0.25, 0.0]), atol=1e-12)


def test_embedded_carrier_archive_round_trip(tmp_path: Path) -> None:
    vertices, faces, weights = _two_layer_source()
    target = np.asarray([[0.25, 0.25, 0.001], [-0.2, -0.3, 0.019], [0.35, -0.2, 0.001]])
    transfer = build_barycentric_transfer(
        vertices,
        faces,
        weights,
        target,
        target_normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        pitch=0.1,
    )
    target_faces = np.asarray([[0, 1, 2]])
    archive = tmp_path / "embedded.npz"
    write_embedded_carrier(
        archive,
        target,
        target_faces,
        transfer,
        source_face_count=len(faces),
        pitch=0.1,
        alpha_over_pitch=1.0,
        offset_over_alpha=0.05,
    )
    loaded_vertices, loaded_faces, loaded_transfer, metadata = read_embedded_carrier(archive)
    np.testing.assert_array_equal(loaded_vertices, target)
    np.testing.assert_array_equal(loaded_faces, target_faces)
    np.testing.assert_allclose(loaded_transfer.weights, transfer.weights)
    assert metadata == {"pitch": 0.1, "alpha_over_pitch": 1.0, "offset_over_alpha": 0.05}
    with np.load(archive, allow_pickle=False) as payload:
        assert str(payload["schema_version"].item()) == EMBEDDED_CARRIER_SCHEMA


def test_identical_embedded_surface_passes_fidelity() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    report = embedded_surface_fidelity(mesh, mesh.copy(), pitch=0.1, sample_count=2_000)
    assert report["status"] == "pass"
    assert report["legacy_relative_volume_error"] == 0.0


def test_embedded_surface_fidelity_accepts_explicit_volume_tolerance() -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    target = source.copy()
    target.apply_scale(1.01)  # type: ignore[no-untyped-call]
    report = embedded_surface_fidelity(
        source,
        target,
        pitch=0.1,
        sample_count=2_000,
        maximum_relative_volume_error=0.031,
    )
    assert report["maximum_relative_volume_error"] == 0.031
    assert "legacy_relative_volume" not in report["blockers"]
