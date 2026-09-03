from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from frayid.io import sha256_file
from frayid.v3.schemas import (
    FrameEmbedding,
    RestMetricFace,
    TopologyCertificateV3,
    UpperGarmentAtlas,
)

EXPERIMENT_ID = "postv3_l05_intrinsic_upper_garment_material_atlas_r01"
LOOP_NAMES = ("neck", "left_armhole", "right_armhole", "hem")


def _write_json_exclusive(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def intrinsic_four_boundary_mesh(grid_size: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Construct a deterministic disk-with-three-holes material domain."""
    if grid_size < 15:
        raise ValueError("grid_size must be at least 15")
    coordinates = np.linspace(-1.0, 1.0, grid_size)
    vertices = np.asarray([(x, y) for y in coordinates for x in coordinates], dtype=np.float64)
    holes = (
        (range(6, 8), range(10, 12)),
        (range(2, 4), range(6, 8)),
        (range(10, 12), range(6, 8)),
    )

    def removed(cell_x: int, cell_y: int) -> bool:
        return any(cell_x in x_values and cell_y in y_values for x_values, y_values in holes)

    faces: list[tuple[int, int, int]] = []
    for cell_y in range(grid_size - 1):
        for cell_x in range(grid_size - 1):
            if removed(cell_x, cell_y):
                continue
            lower_left = cell_y * grid_size + cell_x
            lower_right = lower_left + 1
            upper_left = lower_left + grid_size
            upper_right = upper_left + 1
            faces.extend(
                [(lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)]
            )
    face_array = np.asarray(faces, dtype=np.int64)
    used = np.unique(face_array)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[face_array]


def boundary_cycles(faces: np.ndarray) -> list[list[int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(int(start), int(end)), max(int(start), int(end)))
            counts[edge] += 1
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in boundary_edges:
        adjacency[start].append(end)
        adjacency[end].append(start)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("material domain boundary is not a disjoint union of cycles")
    unused = set(boundary_edges)
    cycles: list[list[int]] = []
    while unused:
        first_edge = min(unused)
        start, current = first_edge
        cycle = [start]
        previous = start
        while current != start:
            cycle.append(current)
            edge = (min(previous, current), max(previous, current))
            unused.discard(edge)
            candidates = [value for value in adjacency[current] if value != previous]
            if not candidates:
                raise ValueError("open boundary chain encountered")
            previous, current = current, candidates[0]
        unused.discard((min(previous, current), max(previous, current)))
        cycles.append(cycle)
    return cycles


def _named_cycles(vertices: np.ndarray, cycles: list[list[int]]) -> dict[str, list[int]]:
    outer = max(cycles, key=len)
    holes = [cycle for cycle in cycles if cycle is not outer]
    if len(holes) != 3:
        raise ValueError("expected one outer cycle and three hole cycles")
    centers = {id(cycle): np.mean(vertices[cycle], axis=0) for cycle in holes}
    neck = max(holes, key=lambda cycle: float(centers[id(cycle)][1]))
    armholes = [cycle for cycle in holes if cycle is not neck]
    left = min(armholes, key=lambda cycle: float(centers[id(cycle)][0]))
    right = max(armholes, key=lambda cycle: float(centers[id(cycle)][0]))
    return {"neck": neck, "left_armhole": left, "right_armhole": right, "hem": outer}


def _topology(
    vertices: np.ndarray, faces: np.ndarray, cycles: list[list[int]]
) -> TopologyCertificateV3:
    edges = {
        tuple(sorted((int(start), int(end))))
        for face in faces
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    }
    euler = len(vertices) - len(edges) + len(faces)
    if len(cycles) != 4 or euler != -2:
        raise ValueError(f"four-boundary topology failed: cycles={len(cycles)}, euler={euler}")
    return TopologyCertificateV3(
        connected_components=1,
        genus=0,
        boundary_loops=4,
        euler_number=-2,
        self_intersections=0,
        unregistered_body_penetrations=0,
        collapsed_triangles=0,
        flipped_triangles=0,
        winding_consistent=True,
    )


def _edge_strain(reference: np.ndarray, posed: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges = np.asarray(
        sorted(
            {
                tuple(sorted((int(start), int(end))))
                for face in faces
                for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            }
        ),
        dtype=np.int64,
    )
    reference_lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    posed_lengths = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
    result: np.ndarray = np.asarray(
        np.abs(posed_lengths / np.maximum(reference_lengths, 1e-12) - 1.0)
    )
    return result


def fit_public_atlas(output_root: Path) -> UpperGarmentAtlas:
    """Build and certify the public atlas fixture; it cannot promote a real result."""
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"immutable atlas output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    uv, faces = intrinsic_four_boundary_mesh()
    cycles = boundary_cycles(faces)
    named = _named_cycles(uv, cycles)
    topology = _topology(uv, faces, cycles)
    intrinsic_path = _write_json_exclusive(
        output_root / "intrinsic_domain.json",
        {"vertices": uv.tolist(), "faces": faces.tolist(), "boundary_cycles": named},
    )

    reference = np.column_stack([0.35 * uv[:, 0], 0.45 * uv[:, 1], np.zeros(len(uv))])
    embeddings: list[FrameEmbedding] = []
    strains: list[float] = []
    for frame_index, phase in enumerate(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)):
        posed = reference.copy()
        posed[:, 0] += 0.002 * np.sin(phase) * uv[:, 1]
        posed[:, 1] += 0.001 * np.cos(phase) * uv[:, 0]
        posed[:, 2] += 0.004 * np.sin(np.pi * uv[:, 0]) * np.cos(phase)
        strains.extend(float(value) for value in _edge_strain(reference, posed, faces))
        embedding_path = _write_json_exclusive(
            output_root / f"embedding_{frame_index:03d}.json",
            {"frame_index": frame_index, "vertices": posed.tolist()},
        )
        embeddings.append(
            FrameEmbedding(
                frame_index=frame_index,
                vertices_path=str(embedding_path),
                vertices_sha256=sha256_file(embedding_path),
                deformation_regime_counts={
                    "attached": 0,
                    "sliding": len(uv) // 4,
                    "free": len(uv) - len(uv) // 4,
                    "contact": 0,
                },
            )
        )

    sdf_path = _write_json_exclusive(
        output_root / "atlas_conditioned_clipped_sdf.json",
        {
            "representation": "atlas_conditioned_clipped_narrow_band_sdf",
            "band_half_width_m": 0.02,
            "samples": [
                {"material_coordinate": [0.0, 0.0], "normal_offset_m": value, "sdf_m": value}
                for value in (-0.02, -0.01, 0.0, 0.01, 0.02)
            ],
        },
    )
    contact_path = _write_json_exclusive(
        output_root / "body_contact_posterior.json",
        {
            "scope": "public_synthetic",
            "d03_role": "immutable_prior_derived_collision_body",
            "contact_probability": [0.0] * len(uv),
        },
    )
    uncertainty_path = _write_json_exclusive(
        output_root / "uncertainty_support_ledger.json",
        {
            "scope": "public_synthetic",
            "observed_vertex_ids": list(range(len(uv))),
            "hidden_vertex_ids": [],
            "silent_completion_forbidden": True,
        },
    )
    metric = [
        RestMetricFace(face_index=index, metric=((1.0, 0.0), (0.0, 1.0)))
        for index in range(len(faces))
    ]
    blockers: list[str] = []
    median_strain = float(np.median(strains))
    p95_strain = float(np.percentile(strains, 95))
    if median_strain >= 0.05 or p95_strain >= 0.15:
        blockers.append("public_strain_gate")
    return UpperGarmentAtlas(
        experiment_id=EXPERIMENT_ID,
        evidence_scope="public_synthetic",
        promotion_eligible=False,
        intrinsic_vertices_path=str(intrinsic_path),
        intrinsic_faces_path=str(intrinsic_path),
        intrinsic_domain_sha256=sha256_file(intrinsic_path),
        boundary_cycles=cast(Any, named),
        seam_hypotheses=[],
        rest_metric=metric,
        frame_embeddings=embeddings,
        clipped_sdf_path=str(sdf_path),
        clipped_sdf_sha256=sha256_file(sdf_path),
        body_contact_posterior_path=str(contact_path),
        body_contact_posterior_sha256=sha256_file(contact_path),
        uncertainty_support_ledger_path=str(uncertainty_path),
        uncertainty_support_ledger_sha256=sha256_file(uncertainty_path),
        topology=topology,
        median_absolute_in_plane_strain=median_strain,
        p95_absolute_in_plane_strain=p95_strain,
        restart_observed_median_spread_mm=0.0,
        restart_observed_p95_spread_mm=0.0,
        status="pass" if not blockers else "fail",
        blockers=blockers,
    )
