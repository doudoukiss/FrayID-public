from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frayid.e25_public_fixtures import (
    E25_PUBLIC_FIXTURE_NAMES,
    E25_PUBLIC_HELD_OUT_VIEW_COUNT,
    E25_PUBLIC_TRAIN_VIEW_COUNT,
    E25PublicEvidence,
    articulated_pose_jacobian,
    assert_public_read_allowed,
    extract_public_truth_mesh,
    finite_difference_field_normal,
    fixture_by_name,
    move_cross_cell_fixture,
    public_camera_bundle,
    public_fixture_registry,
    validate_public_evidence,
    validate_public_modalities,
)
from frayid.eulerian_field import conventional_surface_audit
from frayid.normal_integrable_sdf import spatial_sdf_gradient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fixture_registry_is_complete_unique_and_extracts_genus_zero_surfaces() -> None:
    fixtures = public_fixture_registry()
    assert tuple(fixture.name for fixture in fixtures) == E25_PUBLIC_FIXTURE_NAMES
    assert len({fixture.name for fixture in fixtures}) == len(fixtures)
    for fixture in fixtures:
        mesh = extract_public_truth_mesh(fixture, resolution=18, dtype=torch.float64)
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        audit = conventional_surface_audit(mesh.vertices, mesh.faces)
        assert audit["status"] == "pass", (fixture.name, audit)


def test_public_camera_bundle_retains_registered_twelve_six_split() -> None:
    intrinsics, rotations, translations = public_camera_bundle(
        image_size=(64, 64), dtype=torch.float64
    )
    assert E25_PUBLIC_TRAIN_VIEW_COUNT == 12
    assert E25_PUBLIC_HELD_OUT_VIEW_COUNT == 6
    assert rotations.shape[0] == E25_PUBLIC_TRAIN_VIEW_COUNT + E25_PUBLIC_HELD_OUT_VIEW_COUNT
    assert intrinsics.shape == (3, 3)
    assert translations.shape == (18, 3)
    torch.testing.assert_close(
        torch.linalg.det(rotations), torch.ones(18, dtype=torch.float64), rtol=0.0, atol=1.0e-12
    )


def test_modality_and_evidence_guards_reject_rgb_and_corrupted_normals() -> None:
    validate_public_modalities(("mask", "boundary", "normal"))
    with pytest.raises(ValueError, match=r"added=.*rgb"):
        validate_public_modalities(("mask", "boundary", "normal", "rgb"))

    intrinsics, rotations, translations = public_camera_bundle(
        image_size=(4, 4), dtype=torch.float32
    )
    silhouettes = torch.ones(18, 4, 4)
    normals = torch.zeros(18, 4, 4, 3)
    normals[..., 2] = 1.0
    valid = E25PublicEvidence(silhouettes, normals, intrinsics, rotations, translations)
    validate_public_evidence(valid)
    corrupted = E25PublicEvidence(
        silhouettes,
        normals.clone(),
        intrinsics,
        rotations,
        translations,
    )
    corrupted.normals[3, 1, 2] = torch.tensor([4.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="unit length"):
        validate_public_evidence(corrupted)


@pytest.mark.parametrize(
    "relative",
    (
        "data/private/example.bin",
        "models/private/model.bin",
        "models/checkpoints/state.pt",
        "docs/assets/subject_video.mp4",
        "outputs/development/report.json",
        "outputs/sealed/result.json",
    ),
)
def test_public_read_guard_rejects_every_protected_class(relative: str) -> None:
    with pytest.raises(PermissionError, match="protected read"):
        assert_public_read_allowed(PROJECT_ROOT, PROJECT_ROOT / relative)
    assert_public_read_allowed(PROJECT_ROOT, PROJECT_ROOT / "configs/evaluation/public.yaml")


def test_continuous_normal_matches_central_finite_difference() -> None:
    fixture = fixture_by_name("rotated_ellipsoid")
    points = torch.tensor(
        [[0.68, 0.0, 0.0], [0.0, 0.91, 0.0], [0.2, 0.3, 0.35]],
        dtype=torch.float64,
        requires_grad=True,
    )
    analytic = torch.nn.functional.normalize(
        spatial_sdf_gradient(fixture.field, points, create_graph=False), dim=-1
    )
    finite = finite_difference_field_normal(fixture.field, points.detach(), epsilon=1.0e-5)
    cosine = (analytic * finite).sum(dim=-1)
    assert torch.all(cosine > 1.0 - 1.0e-8)


def test_articulated_jacobian_is_full_rank_and_cross_cell_motion_crosses_half_pitch() -> None:
    points = torch.zeros(7, 3, dtype=torch.float64)
    jacobians = articulated_pose_jacobian(points)
    assert torch.all(torch.linalg.det(jacobians) > 0.9)
    moved = move_cross_cell_fixture(points, grid_pitch=0.1)
    assert torch.linalg.vector_norm(moved - points, dim=-1).min() > 0.05
