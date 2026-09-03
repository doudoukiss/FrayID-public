from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.measure import marching_cubes

from frayid.io import sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import LayerTopologyPolicy
from frayid.v2.topology import TopologyStage, certify_surface

G02_RAW_FIELD_SCHEMA = "frayid_v2_g02_raw_canonical_field.v1"
G02_TOPOLOGY_REPORT_SCHEMA = "frayid_v2_g02_raw_topology_audit.v1"


def _load_raw_field(path: Path) -> tuple[np.ndarray, float, int, dict[str, str]]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"]) != G02_RAW_FIELD_SCHEMA:
            raise ValueError("G02 topology audit received an incompatible raw field")
        field = archive["signed_distance"].astype(np.float32, copy=True)
        extent = float(archive["extent"])
        resolution = int(archive["resolution"])
        provenance = {
            "source_revision": str(archive["source_revision"]),
            "arm_binding_sha256": str(archive["arm_binding_sha256"]),
            "representation": str(archive["representation"]),
            "topology_state": str(archive["topology_state"]),
        }
    if field.shape != (resolution, resolution, resolution):
        raise ValueError("G02 raw field resolution metadata does not match its grid")
    if resolution < 16 or extent <= 0.0 or not np.isfinite(field).all():
        raise ValueError("G02 raw field is not finite and extractable")
    if float(field.min()) > 0.0 or float(field.max()) < 0.0:
        raise ValueError("G02 raw field has no zero level set")
    if provenance["topology_state"] != "search_not_committed":
        raise ValueError("G02 raw field topology state was unexpectedly changed")
    return field, extent, resolution, provenance


def _raw_marching_cubes(field: np.ndarray, extent: float) -> tuple[torch.Tensor, torch.Tensor]:
    pitch = 2.0 * extent / (field.shape[0] - 1)
    # The field uses the conventional negative-inside/positive-outside SDF
    # sign.  skimage's `descent` convention selects outward winding for this
    # ordering (verified by the analytic-sphere test);
    # no component filtering, filling, smoothing, remeshing, or repair occurs.
    vertices, faces, _, _ = marching_cubes(  # type: ignore[no-untyped-call]
        field,
        level=0.0,
        spacing=(pitch, pitch, pitch),
        gradient_direction="descent",
        allow_degenerate=False,
    )
    vertices = vertices + np.asarray([-extent, -extent, -extent])
    return torch.from_numpy(vertices.copy()), torch.from_numpy(faces.astype(np.int64, copy=True))


def audit_g02_raw_topology(raw_field_path: Path, output_path: Path) -> Path:
    """Fail closed on the frozen raw G02 field without presentation cleanup."""

    reject_sealed_capability([raw_field_path, output_path])
    if output_path.exists():
        raise FileExistsError("G02 raw topology report is immutable")
    field, extent, resolution, provenance = _load_raw_field(raw_field_path)
    vertices, faces = _raw_marching_cubes(field, extent)
    policy = LayerTopologyPolicy(
        layer_id="g02_outer_field_candidate",
        role="body",
        closed=True,
        required_component_count=1,
        required_boundary_loop_count=0,
        required_euler_number=2,
    )
    search = certify_surface(
        vertices,
        faces,
        policy=policy,
        stage=TopologyStage.SEARCH,
        exact_intersection_pair_count=None,
        replay_exact=True,
    )
    # Exact pair enumeration cannot rescue a component/Euler/winding failure.
    # Short-circuit it explicitly and leave COMMIT blocked instead of claiming
    # an invented zero count.
    commit = certify_surface(
        vertices,
        faces,
        policy=policy,
        stage=TopologyStage.COMMIT,
        exact_intersection_pair_count=None,
        replay_exact=True,
    )
    preintersection_failures = [
        blocker for blocker in commit.blockers if blocker != "exact_intersection_audit_missing"
    ]
    status = "fail" if preintersection_failures else "blocked"
    payload: dict[str, Any] = {
        "schema_version": G02_TOPOLOGY_REPORT_SCHEMA,
        "status": status,
        "experiment_id": "postv2_g02_direct_multires_field_matched_science_r01",
        "attempt_id": "scientific-attempt-r02",
        "raw_field_sha256": sha256_file(raw_field_path),
        "raw_field_resolution": resolution,
        "raw_field_extent_metres": extent,
        "raw_field_provenance": provenance,
        "extraction": {
            "method": "skimage_marching_cubes_zero_level",
            "gradient_direction": "descent",
            "cleanup_operations": 0,
            "component_filtering": False,
            "hole_filling": False,
            "smoothing": False,
            "remeshing": False,
        },
        "search_certificate": search.model_dump(mode="json"),
        "commit_precheck_certificate": commit.model_dump(mode="json"),
        "exact_intersection_pair_count": None,
        "exact_intersection_short_circuited": bool(preintersection_failures),
        "preintersection_failures": preintersection_failures,
        "authoritative_result_claimed": False,
        "presentation_mesh_generated": False,
        "sealed_test_accesses": 0,
        "blockers": preintersection_failures or ["exact_intersection_audit_required_before_commit"],
    }
    return write_json(output_path, payload)


def topology_report_replay_exact(raw_field_path: Path) -> bool:
    """Extract the frozen grid twice and compare exact vertices and faces."""

    field, extent, _, _ = _load_raw_field(raw_field_path)
    first_vertices, first_faces = _raw_marching_cubes(field, extent)
    second_vertices, second_faces = _raw_marching_cubes(field, extent)
    return bool(
        torch.equal(first_vertices, second_vertices) and torch.equal(first_faces, second_faces)
    )
