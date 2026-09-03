from __future__ import annotations

import json

import numpy as np
import trimesh

from frayid.coarse_bilipschitz import FreudenthalLatticeV1, refine_surface_to_lattice
from frayid.composed_orientation_map import (
    MaterialEmbeddingV1,
    apply_material_control_blocks,
    fit_and_certify_composed_orientation_step,
    run_composed_orientation_controls,
)


def _lattice(nodes: int = 4) -> FreudenthalLatticeV1:
    return FreudenthalLatticeV1.create(
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        nodes_per_axis=nodes,
    )


def test_material_embedding_replays_ordered_blocks_bitwise() -> None:
    lattice = _lattice()
    points = np.asarray([[-0.3, -0.2, -0.1], [0.1, 0.2, 0.3], [0.4, -0.1, 0.2]], dtype=np.float64)
    embedding = MaterialEmbeddingV1.create(lattice, points)
    first = np.zeros_like(lattice.vertices)
    second = np.zeros_like(lattice.vertices)
    first[~lattice.boundary_mask, 0] = 0.01
    second[~lattice.boundary_mask, 1] = -0.02
    expected = points.copy()
    for controls in (first, second):
        expected = np.asarray(expected + embedding.evaluate(controls), dtype=np.float64)
    replay = apply_material_control_blocks(lattice, points, np.asarray([first, second]))
    assert np.array_equal(expected, replay)


def test_composed_fit_is_deterministic_and_every_block_is_certified() -> None:
    lattice = _lattice()
    mesh = trimesh.creation.box(extents=(1.0, 0.9, 0.8))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    refined = refine_surface_to_lattice(lattice, vertices, faces)
    proposal = np.zeros_like(vertices)
    proposal[:, 0] = 0.08 * vertices[:, 1]
    proposal[:, 1] = -0.04 * vertices[:, 0]
    first = fit_and_certify_composed_orientation_step(
        lattice,
        vertices,
        faces,
        refined,
        proposal,
        block_count=2,
        minimum_retained_displacement_ratio=0.0,
        timeout_seconds_per_block=None,
    )
    second = fit_and_certify_composed_orientation_step(
        lattice,
        vertices,
        faces,
        refined,
        proposal,
        block_count=2,
        minimum_retained_displacement_ratio=0.0,
        timeout_seconds_per_block=None,
    )
    assert first.status == "pass"
    assert len(first.blocks) == 2
    assert all(block.status == "pass" for block in first.blocks)
    assert all(block.accepted_path.accepted_alpha == 1.0 for block in first.blocks)
    assert first.decision_sha256 == second.decision_sha256
    assert np.array_equal(first.accepted_control_blocks, second.accepted_control_blocks)
    replay = apply_material_control_blocks(
        lattice, refined.reference_vertices, first.accepted_control_blocks
    )
    assert np.array_equal(replay, first.final_refined_surface_vertices)
    json.dumps(first.report())


def test_registered_composed_orientation_controls_are_json_serializable() -> None:
    report = run_composed_orientation_controls()
    assert report["status"] == "pass"
    assert all(report["checks"].values())
    json.dumps(report)
