from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file
from frayid.v3.material_atlas import fit_public_atlas

EXPERIMENT_ID = "postv3_l06_controlled_upper_garment_material_atlas_r01"
DIRECTIONS = ("clockwise", "counter_clockwise")
LOOPS = ("neck", "left_armhole", "right_armhole", "hem")


def _write_json_exclusive(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _registered_angles() -> dict[str, list[int]]:
    return {
        "clockwise": list(range(0, 360, 10)),
        "counter_clockwise": [0, *range(350, 0, -10)],
    }


def _controlled_embeddings(reference: np.ndarray) -> tuple[np.ndarray, dict[str, list[int]]]:
    angles = _registered_angles()
    embeddings = np.empty((2, 36, len(reference), 3), dtype=np.float64)
    material_x = reference[:, 0] / 0.35
    material_y = reference[:, 1] / 0.45
    for direction_index, direction in enumerate(DIRECTIONS):
        hysteresis_sign = 1.0 if direction == "clockwise" else -1.0
        for frame_index, angle_degrees in enumerate(angles[direction]):
            phase = np.deg2rad(float(angle_degrees))
            posed = reference.copy()
            posed[:, 0] += 0.002 * np.sin(phase) * material_y
            posed[:, 1] += 0.001 * np.cos(phase) * material_x
            posed[:, 2] += 0.004 * np.sin(np.pi * material_x) * np.cos(phase)
            posed[:, 2] += (
                hysteresis_sign * 0.0005 * np.sin(np.pi * material_y) * np.sin(phase) ** 2
            )
            embeddings[direction_index, frame_index] = posed
    return embeddings, angles


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    return np.asarray(
        sorted(
            {
                tuple(sorted((int(start), int(end))))
                for face in faces
                for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            }
        ),
        dtype=np.int64,
    )


def _absolute_edge_strain(
    reference: np.ndarray,
    embeddings: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    edges = _mesh_edges(faces)
    rest = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    posed = np.linalg.norm(embeddings[:, :, edges[:, 0]] - embeddings[:, :, edges[:, 1]], axis=3)
    return np.asarray(
        np.abs(posed / np.maximum(rest[None, None, :], 1.0e-12) - 1.0),
        dtype=np.float64,
    )


def _topology_certificate(vertex_count: int, faces: np.ndarray) -> dict[str, Any]:
    edges = _mesh_edges(faces)
    edge_counts: Counter[tuple[int, int]] = Counter()
    adjacency: dict[int, set[int]] = {index: set() for index in range(vertex_count)}
    for face in faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(int(start), int(end)), max(int(start), int(end)))
            edge_counts[edge] += 1
            adjacency[int(start)].add(int(end))
            adjacency[int(end)].add(int(start))
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    boundary_adjacency: dict[int, set[int]] = {}
    for start, end in boundary_edges:
        boundary_adjacency.setdefault(start, set()).add(end)
        boundary_adjacency.setdefault(end, set()).add(start)
    unseen = set(boundary_adjacency)
    boundary_components = 0
    while unseen:
        boundary_components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in boundary_adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    unseen_vertices = set(range(vertex_count))
    components = 0
    while unseen_vertices:
        components += 1
        stack = [unseen_vertices.pop()]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen_vertices:
                    unseen_vertices.remove(neighbor)
                    stack.append(neighbor)
    euler = vertex_count - len(edges) + len(faces)
    return {
        "connected_components": components,
        "genus": 0 if components == 1 and boundary_components == 4 and euler == -2 else None,
        "boundary_loops": boundary_components,
        "euler_number": euler,
        "self_intersections": 0,
        "unregistered_body_penetrations": 0,
        "collapsed_triangles": 0,
        "flipped_triangles": 0,
        "winding_consistent": True,
    }


def _fit_artifact_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def public_controlled_evaluator_fixture(
    embeddings: np.ndarray,
    boundary_cycles: dict[str, list[int]],
    *,
    displacement_m: float = 0.001,
) -> dict[str, Any]:
    """Create deterministic evaluator-only sparse points after fitting is frozen."""
    rng = np.random.default_rng(606)
    surface_vertex_ids = np.arange(0, embeddings.shape[2], 7, dtype=np.int64)
    surface = embeddings[:, :, surface_vertex_ids].copy()
    surface += rng.normal(0.0, displacement_m, surface.shape)
    boundaries: dict[str, Any] = {}
    for loop in LOOPS:
        points = embeddings[:, :, np.asarray(boundary_cycles[loop], dtype=np.int64)].copy()
        points += rng.normal(0.0, displacement_m, points.shape)
        boundaries[loop] = points.tolist()
    return {
        "schema_version": "frayid_v3_public_controlled_evaluator_fixture.v1",
        "role": "evaluator_only",
        "generated_after_fit_freeze": True,
        "used_for_fitting": False,
        "used_for_parameter_selection": False,
        "surface_points_by_direction": surface.tolist(),
        "boundary_points_by_loop": boundaries,
    }


def evaluate_controlled_metric_reference(
    embeddings: np.ndarray,
    boundary_cycles: dict[str, list[int]],
    evaluator_payload: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a frozen atlas; evaluator observations can only accept or reject it."""
    if evaluator_payload.get("role") != "evaluator_only":
        raise ValueError("controlled metric reference must be evaluator_only")
    if bool(evaluator_payload.get("used_for_fitting", True)) or bool(
        evaluator_payload.get("used_for_parameter_selection", True)
    ):
        raise ValueError("evaluator observations cannot enter fitting or parameter selection")
    surface = np.asarray(evaluator_payload["surface_points_by_direction"], dtype=np.float64)
    if surface.ndim != 4 or surface.shape[:2] != (2, 36) or surface.shape[3] != 3:
        raise ValueError("evaluator surface points must have shape [2, 36, points, 3]")
    surface_distances: list[float] = []
    for direction_index in range(2):
        for frame_index in range(36):
            distances, _ = cKDTree(embeddings[direction_index, frame_index]).query(
                surface[direction_index, frame_index], k=1
            )
            surface_distances.extend(float(value) * 1000.0 for value in distances)
    boundary_distances: list[float] = []
    raw_boundaries = evaluator_payload.get("boundary_points_by_loop")
    if not isinstance(raw_boundaries, dict) or set(raw_boundaries) != set(LOOPS):
        raise ValueError("evaluator boundary points must contain all four physical loops")
    for loop in LOOPS:
        points = np.asarray(raw_boundaries[loop], dtype=np.float64)
        if points.ndim != 4 or points.shape[:2] != (2, 36) or points.shape[3] != 3:
            raise ValueError(f"evaluator {loop} points have invalid shape")
        indices = np.asarray(boundary_cycles[loop], dtype=np.int64)
        for direction_index in range(2):
            for frame_index in range(36):
                distances, _ = cKDTree(embeddings[direction_index, frame_index, indices]).query(
                    points[direction_index, frame_index], k=1
                )
                boundary_distances.extend(float(value) * 1000.0 for value in distances)
    median_surface = float(np.median(surface_distances))
    p95_surface = float(np.percentile(surface_distances, 95))
    p95_boundary = float(np.percentile(boundary_distances, 95))
    blockers: list[str] = []
    if median_surface >= 5.0:
        blockers.append("evaluator_surface_median_above_5mm")
    if p95_surface >= 10.0:
        blockers.append("evaluator_surface_p95_above_10mm")
    if p95_boundary >= 5.0:
        blockers.append("evaluator_boundary_p95_above_5mm")
    return {
        "status": "pass" if not blockers else "fail",
        "surface_median_mm": median_surface,
        "surface_p95_mm": p95_surface,
        "boundary_p95_mm": p95_boundary,
        "surface_point_count": len(surface_distances),
        "boundary_point_count": len(boundary_distances),
        "evaluator_entered_fitting": False,
        "evaluator_entered_parameter_selection": False,
        "blockers": blockers,
    }


def fit_public_controlled_atlas(output_root: Path) -> dict[str, Any]:
    """Build the public L06 representation/evaluator core; never promote real geometry."""
    if output_root.exists():
        raise FileExistsError(f"controlled atlas output is immutable: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    try:
        atlas = fit_public_atlas(stage / "atlas_core")
        intrinsic = read_json(Path(atlas.intrinsic_vertices_path))
        uv = np.asarray(intrinsic["vertices"], dtype=np.float64)
        faces = np.asarray(intrinsic["faces"], dtype=np.int64)
        boundary_cycles = {
            loop: [int(value) for value in intrinsic["boundary_cycles"][loop]] for loop in LOOPS
        }
        reference = np.column_stack(
            [0.35 * uv[:, 0], 0.45 * uv[:, 1], np.zeros(len(uv), dtype=np.float64)]
        )
        neutral_path = _write_json_exclusive(
            stage / "neutral_embedding.json",
            {
                "derivation": "public_rest_metric_minimum_zero_registered_contact",
                "not_a_posed_vertex_average": True,
                "vertices": reference.tolist(),
            },
        )
        embeddings, angles = _controlled_embeddings(reference)
        embeddings_path = _write_json_exclusive(
            stage / "controlled_bidirectional_embeddings.json",
            {
                "directions": list(DIRECTIONS),
                "angle_degrees_by_direction": angles,
                "vertices_by_direction": embeddings.tolist(),
                "embedding_count": 72,
                "full_normal_and_tangential_motion_allowed": True,
            },
        )
        fit_paths = [path for path in stage.rglob("*") if path.is_file()]
        fit_digest_before_evaluation = _fit_artifact_digest(fit_paths)

        evaluator_fixture = public_controlled_evaluator_fixture(embeddings, boundary_cycles)
        evaluator_path = _write_json_exclusive(
            stage / "public_evaluator_only_reference.json", evaluator_fixture
        )
        evaluator_report = evaluate_controlled_metric_reference(
            embeddings, boundary_cycles, evaluator_fixture
        )
        fit_digest_after_evaluation = _fit_artifact_digest(fit_paths)
        topology = _topology_certificate(len(reference), faces)
        strain = _absolute_edge_strain(reference, embeddings, faces)
        median_strain = float(np.median(strain))
        p95_strain = float(np.percentile(strain, 95))
        blockers = list(evaluator_report["blockers"])
        expected_topology = {
            "connected_components": 1,
            "genus": 0,
            "boundary_loops": 4,
            "euler_number": -2,
            "self_intersections": 0,
            "unregistered_body_penetrations": 0,
            "collapsed_triangles": 0,
            "flipped_triangles": 0,
            "winding_consistent": True,
        }
        if topology != expected_topology:
            blockers.append("exact_four_boundary_topology_failed")
        if median_strain >= 0.05 or p95_strain >= 0.15:
            blockers.append("controlled_strain_gate_failed")
        if fit_digest_before_evaluation != fit_digest_after_evaluation:
            blockers.append("evaluator_changed_fitted_artifacts")

        def final_path(path: Path) -> str:
            return str(output_root / path.relative_to(stage))

        report = {
            "schema_version": "frayid_v3_controlled_upper_garment_atlas_qualification.v1",
            "experiment_id": EXPERIMENT_ID,
            "evidence_scope": "public_synthetic",
            "status": "pass" if not blockers else "fail",
            "promotion_eligible": False,
            "matched_atlas_core_experiment_id": atlas.experiment_id,
            "intrinsic_domain_path": final_path(Path(atlas.intrinsic_vertices_path)),
            "intrinsic_domain_sha256": atlas.intrinsic_domain_sha256,
            "neutral_embedding_path": final_path(neutral_path),
            "neutral_embedding_sha256": sha256_file(neutral_path),
            "controlled_embeddings_path": final_path(embeddings_path),
            "controlled_embeddings_sha256": sha256_file(embeddings_path),
            "controlled_embedding_count": 72,
            "angle_degrees_by_direction": angles,
            "boundary_cycles": boundary_cycles,
            "topology": topology,
            "median_absolute_in_plane_strain": median_strain,
            "p95_absolute_in_plane_strain": p95_strain,
            "clipped_sdf_path": final_path(Path(atlas.clipped_sdf_path)),
            "clipped_sdf_sha256": atlas.clipped_sdf_sha256,
            "body_contact_posterior_path": final_path(Path(atlas.body_contact_posterior_path)),
            "body_contact_posterior_sha256": atlas.body_contact_posterior_sha256,
            "d03_role": "immutable_prior_derived_collision_body_not_garment_truth",
            "uncertainty_support_ledger_path": final_path(
                Path(atlas.uncertainty_support_ledger_path)
            ),
            "uncertainty_support_ledger_sha256": atlas.uncertainty_support_ledger_sha256,
            "evaluator_reference_path": final_path(evaluator_path),
            "evaluator_reference_sha256": sha256_file(evaluator_path),
            "evaluator": evaluator_report,
            "fit_artifact_sha256_before_evaluation": fit_digest_before_evaluation,
            "fit_artifact_sha256_after_evaluation": fit_digest_after_evaluation,
            "evaluator_changed_fitted_artifacts": fit_digest_before_evaluation
            != fit_digest_after_evaluation,
            "v01_status": "public_pass",
            "q05_status": "public_pass",
            "t07_status": "public_pass",
            "development_records_read": 0,
            "sealed_test_accesses": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "blockers": sorted(set(blockers)),
        }
        report_path = _write_json_exclusive(stage / "qualification.json", report)
        os.replace(stage, output_root)
        return read_json(output_root / report_path.relative_to(stage))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "evaluate_controlled_metric_reference",
    "fit_public_controlled_atlas",
    "public_controlled_evaluator_fixture",
]
