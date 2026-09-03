from __future__ import annotations

from dataclasses import replace

import numpy as np
import trimesh

from frayid.refinement_certificate import (
    certify_exact_dyadic_refinement,
    subdivide_with_exact_provenance,
)
from frayid.source_exclusion_carrier import uniform_conforming_subdivide


def _fixture() -> tuple[np.ndarray, np.ndarray, float]:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    grid = 2.0**-30
    vertices = np.rint(np.asarray(mesh.vertices) / grid) * grid
    return vertices, np.asarray(mesh.faces, dtype=np.int64), grid


def test_exact_certificate_matches_existing_refinement() -> None:
    vertices, faces, grid = _fixture()
    refinement = subdivide_with_exact_provenance(vertices, faces, rounds=2)
    prior_vertices, prior_faces = uniform_conforming_subdivide(vertices, faces, rounds=2)
    assert np.array_equal(refinement.vertices, prior_vertices)
    assert np.array_equal(refinement.faces, prior_faces)
    certificate = certify_exact_dyadic_refinement(
        vertices, faces, refinement, parent_grid=grid, rounds=2
    )
    assert certificate.status == "pass"
    assert certificate.denominator == 4
    assert certificate.children_per_parent == 16
    assert certificate.exact_integer_affine_reconstruction


def test_coordinate_fault_fails_integer_reconstruction() -> None:
    vertices, faces, grid = _fixture()
    refinement = subdivide_with_exact_provenance(vertices, faces, rounds=2)
    corrupted = refinement.vertices.copy()
    corrupted[-1, 0] += grid / 4.0
    certificate = certify_exact_dyadic_refinement(
        vertices,
        faces,
        replace(refinement, vertices=corrupted),
        parent_grid=grid,
        rounds=2,
    )
    assert certificate.status == "fail"
    assert "integer_affine_reconstruction" in certificate.blockers


def test_provenance_fault_fails_partition() -> None:
    vertices, faces, grid = _fixture()
    refinement = subdivide_with_exact_provenance(vertices, faces, rounds=2)
    corrupted = refinement.barycentric_numerators.copy()
    corrupted[0] = corrupted[1]
    certificate = certify_exact_dyadic_refinement(
        vertices,
        faces,
        replace(refinement, barycentric_numerators=corrupted),
        parent_grid=grid,
        rounds=2,
    )
    assert certificate.status == "fail"
    assert "standard_partition" in certificate.blockers


def test_parent_assignment_fault_fails_child_counts() -> None:
    vertices, faces, grid = _fixture()
    refinement = subdivide_with_exact_provenance(vertices, faces, rounds=2)
    corrupted = refinement.parent_face_indices.copy()
    corrupted[0] = 1
    certificate = certify_exact_dyadic_refinement(
        vertices,
        faces,
        replace(refinement, parent_face_indices=corrupted),
        parent_grid=grid,
        rounds=2,
    )
    assert certificate.status == "fail"
    assert "parent_child_count" in certificate.blockers


def test_p2_runner_has_public_only_execution_counters() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/run_post_v1_p2_public_gate.py"
    ).read_text()
    for counter in (
        '"private_input_reads": 0',
        '"development_evidence_reads": 0',
        '"image_loads": 0',
        '"optimizer_steps": 0',
        '"modal_invocations": 0',
        '"sealed_test_accesses": 0',
    ):
        assert counter in source
