from __future__ import annotations

from typing import Any

import numpy as np

from frayid.v3.material_atlas import intrinsic_four_boundary_mesh

EXPERIMENT_ID = "postv3_g05_envelope_differential_surface_r01"


def _graph_laplacian(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    adjacency = np.zeros((vertex_count, vertex_count), dtype=np.float64)
    for face in faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            adjacency[int(start), int(end)] = 1.0
            adjacency[int(end), int(start)] = 1.0
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def refine_differential_surface(payload: dict[str, Any]) -> dict[str, Any]:
    """Split envelope/detail with a frozen public spectral cutoff and fail closed."""
    vertices = np.asarray(payload["vertices"], dtype=np.float64)
    faces = np.asarray(payload["faces"], dtype=np.int64)
    displacement = np.asarray(payload["displacement"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or displacement.shape != vertices.shape:
        raise ValueError("vertices and displacement must both have shape [N, 3]")
    cutoff = int(payload["laplace_beltrami_cutoff_modes"])
    if cutoff < 1 or cutoff >= len(vertices):
        raise ValueError("spectral cutoff must retain between 1 and N-1 modes")
    laplacian = _graph_laplacian(len(vertices), faces)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    low_basis = eigenvectors[:, :cutoff]
    envelope = low_basis @ (low_basis.T @ displacement)
    detail = displacement - envelope
    detail -= np.mean(detail, axis=0, keepdims=True)
    mask_before = float(payload["full_mask_metric_before"])
    mask_after = float(payload["full_mask_metric_after"])
    normal_improvement = float(payload["median_normal_improvement_degrees"])
    boundary_improvement = float(payload["boundary_error_improvement_fraction"])
    topology = payload.get("topology", {})
    exact_topology = (
        isinstance(topology, dict)
        and topology.get("connected_components") == 1
        and topology.get("boundary_loops") == 4
        and topology.get("euler_number") == -2
        and topology.get("self_intersections") == 0
        and topology.get("unregistered_body_penetrations") == 0
    )
    blockers: list[str] = []
    if mask_after + 1e-12 < mask_before:
        blockers.append("full_mask_regression")
    if not exact_topology:
        blockers.append("exact_topology_regression")
    if normal_improvement < 2.0 and boundary_improvement < 0.1:
        blockers.append("detail_evidence_improvement_gate")
    if str(payload.get("cutoff_source")) != "public_synthetic_frozen":
        blockers.append("laplace_beltrami_cutoff_not_synthetic_frozen")
    return {
        "schema_version": "frayid_v3_envelope_differential_surface.v1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_scope": str(payload.get("evidence_scope", "public_synthetic")),
        "status": "pass" if not blockers else "fail",
        "promotion_eligible": not blockers and payload.get("evidence_scope") == "train_real",
        "envelope_update_evidence": ["masks", "physical_boundaries", "free_space"],
        "detail_update_evidence": ["normals", "chart_residuals", "bending_coordinates"],
        "eigenvalues": eigenvalues[: cutoff + 1].tolist(),
        "envelope": envelope.tolist(),
        "zero_mean_detail": detail.tolist(),
        "detail_mean_norm": float(np.linalg.norm(np.mean(detail, axis=0))),
        "topology_checked_at_block_commit": True,
        "hidden_cleanup_operations": 0,
        "blockers": blockers,
    }


def public_detail_fixture() -> dict[str, Any]:
    uv, faces = intrinsic_four_boundary_mesh()
    vertices = np.column_stack([0.35 * uv[:, 0], 0.45 * uv[:, 1], np.zeros(len(uv))])
    displacement = np.column_stack(
        [np.zeros(len(uv)), np.zeros(len(uv)), 0.01 * np.sin(6.0 * uv[:, 0])]
    )
    return {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "displacement": displacement.tolist(),
        "laplace_beltrami_cutoff_modes": 8,
        "cutoff_source": "public_synthetic_frozen",
        "full_mask_metric_before": 0.9,
        "full_mask_metric_after": 0.9,
        "median_normal_improvement_degrees": 2.5,
        "boundary_error_improvement_fraction": 0.0,
        "topology": {
            "connected_components": 1,
            "boundary_loops": 4,
            "euler_number": -2,
            "self_intersections": 0,
            "unregistered_body_penetrations": 0,
        },
        "evidence_scope": "public_synthetic",
    }
