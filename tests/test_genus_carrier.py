from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from frayid.genus_carrier import (
    EXPANSION_DENOMINATOR,
    EXPANSION_NUMERATOR,
    EXPERIMENT_ID,
    FIDELITY_MAXIMUM_RELATIVE_VOLUME_ERROR,
    PUBLIC_FIDELITY_INPUT_SHA256,
    PUBLIC_FIXTURE_INPUT_SHA256,
    public_genus_carrier_fixtures,
    public_genus_fidelity_fixtures,
)


def _reference_envelope(points: np.ndarray) -> trimesh.Trimesh:
    hull = trimesh.convex.convex_hull(points)
    vertices = np.asarray(hull.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    ratio = EXPANSION_NUMERATOR / EXPANSION_DENOMINATOR
    hull.vertices = center + ratio * (vertices - center)
    return hull


def test_public_fixtures_cover_non_spherical_source_topologies() -> None:
    fixtures = public_genus_carrier_fixtures()
    assert EXPERIMENT_ID == "postv1_e11_genus_controlled_carrier_r01"
    assert len(PUBLIC_FIXTURE_INPUT_SHA256) == 64
    assert {fixture.name for fixture in fixtures} == {
        "sphere_control",
        "rotated_ellipsoid",
        "concave_star",
        "genus_one_torus",
        "two_component_spheres",
    }
    assert any(fixture.source_euler_number != 2 for fixture in fixtures)
    assert any(fixture.source_component_count > 1 for fixture in fixtures)
    for fixture in fixtures:
        fixture.validate()
        assert fixture.as_public_record()["schema_version"] == "e11_public_fixture.v1"


def test_reference_convex_envelopes_are_genus_zero_by_construction() -> None:
    for fixture in public_genus_carrier_fixtures():
        envelope = _reference_envelope(fixture.vertices)
        assert envelope.is_watertight
        assert envelope.is_winding_consistent
        assert int(envelope.euler_number) == 2
        assert len(envelope.split(only_watertight=False)) == 1
        assert float(envelope.volume) > 0.0


def test_public_fidelity_fixtures_bind_clean_references_and_feature_strata() -> None:
    fixtures = public_genus_fidelity_fixtures()
    assert len(PUBLIC_FIDELITY_INPUT_SHA256) == 64
    assert {fixture.name for fixture in fixtures} == {
        "sphere_control",
        "sphere_scale_0_1",
        "sphere_scale_10",
        "rotated_ellipsoid",
        "rigid_ellipsoid",
        "concave_pocket",
        "concave_pocket_defective_soup",
        "near_contact_hairpin",
    }
    assert (EXPANSION_NUMERATOR / EXPANSION_DENOMINATOR) ** 3 - 1.0 < (
        FIDELITY_MAXIMUM_RELATIVE_VOLUME_ERROR
    )
    assert {fixture.invariance_group for fixture in fixtures if fixture.invariance_group} == {
        "sphere_scale",
        "ellipsoid_rigid",
    }
    assert any(len(fixture.exterior_probes) for fixture in fixtures)
    assert any(len(fixture.feature_face_indices) for fixture in fixtures)
    for fixture in fixtures:
        fixture.validate()


def test_public_fidelity_runner_is_public_only_and_immutable() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_post_v1_e11_fidelity_gate.py"
    ).read_text(encoding="utf-8")
    assert "public_procedural_geometry_only" in source
    assert "private_input_reads" in source
    assert "development_evidence_reads" in source
    assert "sealed_test_accesses" in source
    assert "if arguments.output.exists()" in source
    assert "automatic" not in source.lower()


def test_exact_constructor_freezes_rational_expansion_and_cgal_minor() -> None:
    source = (Path(__file__).resolve().parents[1] / "tools/e11_cgal/convex_envelope.cpp").read_text(
        encoding="utf-8"
    )
    assert "Kernel::FT(101) / Kernel::FT(100)" in source
    assert "CGAL_VERSION_NR >= 1060200000" in source
    assert "CGAL_VERSION_NR < 1060300000" in source
    assert "CGAL::convex_hull_3" in source


def test_modal_runner_is_public_only_and_zero_retry() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/modal_post_v1_e11_genus_carrier.py"
    if not source_path.is_file():
        pytest.skip("operator-only Modal entrypoint is excluded from the public snapshot")
    source = source_path.read_text(encoding="utf-8")
    assert "cpu=16.0" in source
    assert "memory=65536" in source
    assert "retries=0" in source
    assert "private_input_volume" not in source
    assert '"public_fixture_definition": PUBLIC_FIXTURE_INPUT_SHA256' in source
    assert "claim_attempt(" in source
    assert source.index("claim_attempt(") < source.index("subprocess.run(")
    assert "immutable_output_volume.commit()" in source
