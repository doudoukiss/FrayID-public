from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import trimesh
import yaml

from frayid.io import read_json, sha256_file
from frayid.sdf_distillation import extract_voxel_sdf_mesh

EXPERIMENT_ID = "postv1_e00_incumbent_binding_r01"
ACCEPTED_RUN_RELATIVE = Path(
    "outputs/canonical_clothed_surface_v1/runs/"
    "important_canonical_clothed_surface_v1_r4m_motion_bounded12_0ebff81c68a7"
)
SDF_RELATIVE = ACCEPTED_RUN_RELATIVE / ("sdf_distillation/topology_safe_61247567032b_r384_ss3")
ACCEPTED_CONFIG_RELATIVE = Path("configs/evaluation/accepted_canonical_surface_v1.yaml")
RECONSTRUCTION_CONFIG_RELATIVE = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml")
DATASET_MANIFEST_RELATIVE = Path(
    "outputs/canonical_clothed_surface_v1/dataset/dataset_manifest.json"
)
INITIALIZATION_RELATIVE = Path(
    "outputs/canonical_clothed_surface_v1/dataset/sequence_initialization.json"
)
OUTPUT_RELATIVE = Path("outputs/canonical_clothed_surface_v1/post_v1") / EXPERIMENT_ID
BRIDGE_SOURCE_REVISION = "61247567032b854729c4e614c5fb4245f02760a5"
P10_CONTRACT = {
    "estimator": "numpy.quantile",
    "q": 0.10,
    "method": "linear",
    "population": "all_36_development_validation_frames",
}
PAIRED_BOOTSTRAP_CONTRACT = {
    "seed": 20260831,
    "replicates": 10_000,
    "resampling_unit": "development_validation_frame",
    "purpose": "descriptive_uncertainty_only",
}
RELEVANT_EVALUATOR_PATHS = (
    "src/frayid/evaluation.py",
    "src/frayid/geometry.py",
    "src/frayid/triangle_rasterizer.py",
    "src/frayid/dataset.py",
    "src/frayid/config.py",
    str(RECONSTRUCTION_CONFIG_RELATIVE),
)


def assert_not_sealed_private_path(path: Path) -> None:
    """Reject access to private sealed-test evidence and audit directories."""
    lowered = [part.lower() for part in path.parts]
    if "sealed_test_v1" in lowered or any(
        part.startswith("sealed_test_audit_") for part in lowered
    ):
        raise ValueError(f"E0 forbids sealed-test private path access: {path}")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return cast(dict[str, Any], payload)


def _as_numpy(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return np.asarray(result, dtype=dtype) if dtype is not None else np.asarray(result)


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(_as_numpy(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def topology_report(
    reference_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_signed_area_ratio: float = 0.01,
    minimum_unsigned_area_ratio: float = 0.1,
) -> dict[str, Any]:
    """Audit face floors plus the structure of one actual same-connectivity mesh."""
    reference = np.asarray(reference_vertices, dtype=np.float64)
    deformed = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if reference.shape != deformed.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Reference and deformed vertices must share shape [V, 3]")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Faces must have shape [F, 3]")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(reference)):
        raise ValueError("Faces reference an invalid vertex")

    reference_triangles = reference[triangles]
    deformed_triangles = deformed[triangles]
    reference_area_vectors = np.cross(
        reference_triangles[:, 1] - reference_triangles[:, 0],
        reference_triangles[:, 2] - reference_triangles[:, 0],
    )
    deformed_area_vectors = np.cross(
        deformed_triangles[:, 1] - deformed_triangles[:, 0],
        deformed_triangles[:, 2] - deformed_triangles[:, 0],
    )
    reference_norm_squared = np.sum(reference_area_vectors**2, axis=1)
    if np.any(reference_norm_squared <= 1e-24):
        raise ValueError("Original topology reference contains a degenerate face")
    reference_norm = np.sqrt(reference_norm_squared)
    deformed_norm = np.linalg.norm(deformed_area_vectors, axis=1)
    signed_ratios = np.sum(reference_area_vectors * deformed_area_vectors, axis=1) / (
        reference_norm_squared
    )
    unsigned_ratios = deformed_norm / reference_norm
    flipped = signed_ratios <= 0.0
    signed_floor = signed_ratios < minimum_signed_area_ratio
    collapsed = unsigned_ratios < minimum_unsigned_area_ratio

    mesh = trimesh.Trimesh(vertices=deformed, faces=triangles, process=False)
    components = mesh.split(only_watertight=False)
    blockers: list[str] = []
    if np.any(flipped):
        blockers.append("face_flip")
    if np.any(signed_floor):
        blockers.append("signed_area_floor")
    if np.any(collapsed):
        blockers.append("unsigned_area_floor")
    if len(components) != 1:
        blockers.append("component_count_not_one")
    if not mesh.is_watertight:
        blockers.append("not_watertight")
    if int(mesh.euler_number) != 2:
        blockers.append("euler_number_not_two")
    if not mesh.is_winding_consistent:
        blockers.append("inconsistent_winding")
    return {
        "status": "pass" if not blockers else "fail",
        "vertex_count": len(deformed),
        "face_count": len(triangles),
        "flipped_face_count": int(np.count_nonzero(flipped)),
        "signed_area_floor_violation_count": int(np.count_nonzero(signed_floor)),
        "collapsed_face_count": int(np.count_nonzero(collapsed)),
        "minimum_signed_area_ratio": float(np.min(signed_ratios)),
        "minimum_unsigned_area_ratio": float(np.min(unsigned_ratios)),
        "component_count": len(components),
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "blockers": blockers,
    }


def structural_mesh_report(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Describe an unrelated tessellation without applying source face ratios."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    return {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "component_count": len(components),
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def development_metrics_from_evaluation(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "train_iou": float(payload["train_silhouette_iou"]),
        "held_out_iou": float(payload["held_out_silhouette_iou"]),
        "held_out_initialization_iou": float(payload["initialization_held_out_iou"]),
        "held_out_improvement": float(payload["held_out_silhouette_iou"])
        - float(payload["initialization_held_out_iou"]),
        "boundary_error": float(payload["normalized_boundary_error"]),
        "median_normal_error_degrees": float(payload["median_normal_error_degrees"]),
    }


def exact_metric_comparison(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "train_iou",
        "held_out_iou",
        "held_out_initialization_iou",
        "held_out_improvement",
        "boundary_error",
        "median_normal_error_degrees",
    )
    comparisons = {
        key: {
            "expected": float(expected[key]),
            "observed": float(observed[key]),
            "exact": float(expected[key]) == float(observed[key]),
        }
        for key in keys
    }
    mismatches = [key for key, value in comparisons.items() if not value["exact"]]
    return {
        "status": "pass" if not mismatches else "fail",
        "tolerance": 0.0,
        "comparisons": comparisons,
        "mismatches": mismatches,
    }


def _artifact_record(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    assert_not_sealed_private_path(path)
    exists = path.is_file()
    observed = sha256_file(path) if exists else None
    matches = exists and (expected_sha256 is None or observed == expected_sha256)
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "matches": matches,
    }


def build_local_binding_report(project_root: Path, *, source_revision: str) -> dict[str, Any]:
    """Bind the accepted development context without reading sealed private data."""
    root = project_root.resolve()
    run_root = root / ACCEPTED_RUN_RELATIVE
    sdf_root = root / SDF_RELATIVE
    accepted_path = root / ACCEPTED_CONFIG_RELATIVE
    reconstruction_config_path = root / RECONSTRUCTION_CONFIG_RELATIVE
    dataset_manifest_path = root / DATASET_MANIFEST_RELATIVE
    initialization_path = root / INITIALIZATION_RELATIVE
    bridge_path = sdf_root / "accepted_canonical_sdf_modal_report.json"
    provenance_path = run_root / "provenance.json"

    accepted = _load_yaml_mapping(accepted_path)
    bridge = read_json(bridge_path)
    provenance = read_json(provenance_path)
    candidate = cast(dict[str, Any], accepted["candidate"])
    accepted_reports = cast(dict[str, Any], accepted["reports"])

    expected_hashes: dict[str, tuple[Path, str | None]] = {
        "motion_checkpoint": (
            run_root / "motion_coadaptation_bounded_checkpoint.pt",
            str(candidate["motion_checkpoint_sha256"]),
        ),
        "explicit_checkpoint": (
            run_root / "explicit_motion_surface_checkpoint.pt",
            str(candidate["explicit_checkpoint_sha256"]),
        ),
        "explicit_carrier": (
            run_root / "explicit_motion_carrier.npz",
            str(bridge["source_carrier_sha256"]),
        ),
        "explicit_mesh": (
            run_root / "explicit_motion_surface.ply",
            str(candidate["explicit_mesh_sha256"]),
        ),
        "sdf_grid": (sdf_root / "canonical_sdf_grid.npz", str(candidate["sdf_grid_sha256"])),
        "sdf_carrier": (
            sdf_root / "canonical_sdf_carrier.npz",
            str(candidate["sdf_carrier_sha256"]),
        ),
        "sdf_mesh": (sdf_root / "canonical_sdf_mesh.ply", str(candidate["sdf_mesh_sha256"])),
        "bridge_report": (bridge_path, str(accepted_reports["modal_sdf_bridge_sha256"])),
        "reconstruction_config": (
            reconstruction_config_path,
            str(provenance["config_sha256"]),
        ),
        "dataset_manifest_split": (
            dataset_manifest_path,
            str(provenance["dataset_manifest_sha256"]),
        ),
        "development_initialization_comparator": (initialization_path, None),
        "accepted_v1_manifest": (accepted_path, None),
    }
    artifacts = {
        name: _artifact_record(path, expected) for name, (path, expected) in expected_hashes.items()
    }
    blockers = [
        f"artifact_binding:{name}" for name, value in artifacts.items() if not value["matches"]
    ]

    motion_payload = torch.load(
        expected_hashes["motion_checkpoint"][0], map_location="cpu", weights_only=False
    )
    motion_state = cast(dict[str, Any], motion_payload["model"])
    checkpoint_reference = _as_numpy(motion_state["base_vertices"], dtype=np.dtype(np.float32))
    checkpoint_vertices = checkpoint_reference + _as_numpy(
        motion_state["canonical_offsets"], dtype=np.dtype(np.float32)
    )
    checkpoint_faces = _as_numpy(motion_state["faces"], dtype=np.dtype(np.int64))

    explicit_payload = torch.load(
        expected_hashes["explicit_checkpoint"][0], map_location="cpu", weights_only=False
    )
    original_reference = _as_numpy(explicit_payload["base"], dtype=np.dtype(np.float32))
    explicit_checkpoint_faces = _as_numpy(explicit_payload["faces"], dtype=np.dtype(np.int64))
    with np.load(expected_hashes["explicit_carrier"][0]) as archive:
        explicit_vertices = np.asarray(archive["vertices"], dtype=np.float32)
        explicit_faces = np.asarray(archive["faces"], dtype=np.int64)
        explicit_weights = np.asarray(archive["weights"], dtype=np.float32)
    with np.load(expected_hashes["sdf_carrier"][0]) as archive:
        sdf_vertices = np.asarray(archive["vertices"], dtype=np.float32)
        sdf_faces = np.asarray(archive["faces"], dtype=np.int64)
        sdf_weights = np.asarray(archive["weights"], dtype=np.float32)

    explicit_mesh = trimesh.load_mesh(expected_hashes["explicit_mesh"][0], process=False)
    sdf_mesh = trimesh.load_mesh(expected_hashes["sdf_mesh"][0], process=False)
    if not isinstance(explicit_mesh, trimesh.Trimesh) or not isinstance(sdf_mesh, trimesh.Trimesh):
        raise ValueError("Accepted PLY paths must each contain exactly one mesh")

    array_bindings = {
        "explicit_checkpoint_faces_equal_carrier": bool(
            np.array_equal(explicit_checkpoint_faces, explicit_faces)
        ),
        "explicit_ply_vertices_equal_carrier": bool(
            np.array_equal(np.asarray(explicit_mesh.vertices, dtype=np.float32), explicit_vertices)
        ),
        "explicit_ply_faces_equal_carrier": bool(
            np.array_equal(np.asarray(explicit_mesh.faces, dtype=np.int64), explicit_faces)
        ),
        "sdf_ply_vertices_equal_carrier": bool(
            np.array_equal(np.asarray(sdf_mesh.vertices, dtype=np.float32), sdf_vertices)
        ),
        "sdf_ply_faces_equal_carrier": bool(
            np.array_equal(np.asarray(sdf_mesh.faces, dtype=np.int64), sdf_faces)
        ),
        "explicit_and_sdf_faces_equal": bool(np.array_equal(explicit_faces, sdf_faces)),
        "explicit_and_sdf_weights_equal": bool(np.array_equal(explicit_weights, sdf_weights)),
    }
    blockers.extend(f"array_binding:{name}" for name, value in array_bindings.items() if not value)

    topology = {
        "checkpoint_13776": topology_report(
            checkpoint_reference, checkpoint_vertices, checkpoint_faces
        ),
        "explicit_carrier_55104": topology_report(
            original_reference, explicit_vertices, explicit_faces
        ),
        "sdf_carrier_55104_actual_output": topology_report(
            original_reference, sdf_vertices, sdf_faces
        ),
    }
    blockers.extend(
        f"topology:{name}" for name, value in topology.items() if value["status"] != "pass"
    )

    with np.load(expected_hashes["sdf_grid"][0]) as grid:
        sdf = np.asarray(grid["sdf"], dtype=np.float32)
        origin = np.asarray(grid["origin"], dtype=np.float32)
        pitch = float(np.asarray(grid["pitch"]).reshape(()))
    raw_mesh = extract_voxel_sdf_mesh(sdf, origin, pitch)
    boundary_values = np.concatenate(
        (
            sdf[0].ravel(),
            sdf[-1].ravel(),
            sdf[:, 0].ravel(),
            sdf[:, -1].ravel(),
            sdf[:, :, 0].ravel(),
            sdf[:, :, -1].ravel(),
        )
    )
    raw_field_diagnostic = {
        **structural_mesh_report(
            np.asarray(raw_mesh.vertices), np.asarray(raw_mesh.faces, dtype=np.int64)
        ),
        "minimum_boundary_sdf": float(np.min(boundary_values)),
        "positive_domain_boundary": bool(np.all(boundary_values > 0.0)),
        "role": "diagnostic_unrelated_tessellation_not_the_promoted_v1_carrier",
        "prospective_e3_one_component_euler2_certificate": bool(
            len(raw_mesh.split(only_watertight=False)) == 1
            and raw_mesh.is_watertight
            and int(raw_mesh.euler_number) == 2
        ),
    }
    if not raw_field_diagnostic["positive_domain_boundary"]:
        blockers.append("raw_field:nonpositive_domain_boundary")

    expected_development = cast(dict[str, Any], accepted["development_validation"])
    stored_development = development_metrics_from_evaluation(
        cast(dict[str, Any], bridge["sdf_exact_evaluation"])
    )
    stored_metric_match = exact_metric_comparison(expected_development, stored_development)
    if stored_metric_match["status"] != "pass":
        blockers.append("stored_development_metrics_mismatch")
    if bridge.get("status") != "pass":
        blockers.append("stored_bridge_report_not_pass")
    fidelity = cast(dict[str, Any], bridge["fidelity"])
    if fidelity.get("status") != "pass":
        blockers.append("stored_fidelity_report_not_pass")

    revision_comparison = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            f"{BRIDGE_SOURCE_REVISION}..{source_revision}",
            "--",
            *RELEVANT_EVALUATOR_PATHS,
        ],
        cwd=root,
        check=False,
    )
    working_tree_comparison = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *RELEVANT_EVALUATOR_PATHS],
        cwd=root,
        check=False,
    )
    relevant_source_unchanged = revision_comparison.returncode == 0
    relevant_working_tree_clean = working_tree_comparison.returncode == 0
    if not relevant_source_unchanged:
        blockers.append("evaluator_contract_changed_since_bridge")
    if not relevant_working_tree_clean:
        blockers.append("evaluator_contract_has_uncommitted_changes")
    relevant_source_files = {
        relative: sha256_file(root / relative) for relative in RELEVANT_EVALUATOR_PATHS
    }

    return {
        "schema_version": "post_v1_e0_binding.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "local_pass" if not blockers else "fail",
        "blockers": blockers,
        "source_revision": source_revision,
        "bridge_source_revision": BRIDGE_SOURCE_REVISION,
        "source_bindings": {
            "relevant_paths": list(RELEVANT_EVALUATOR_PATHS),
            "relevant_file_sha256": relevant_source_files,
            "unchanged_from_bridge_revision": relevant_source_unchanged,
            "working_tree_clean_for_relevant_paths": relevant_working_tree_clean,
        },
        "scope": "development_only_zero_training",
        "artifacts": artifacts,
        "original_topology_reference": {
            "vertex_count": len(original_reference),
            "sha256_array": sha256_array(original_reference),
            "faces_sha256_array": sha256_array(explicit_faces),
        },
        "array_bindings": array_bindings,
        "topology": topology,
        "raw_field_diagnostic": raw_field_diagnostic,
        "face_count_explanation": {
            "13776": "checkpoint model base topology; not the external evaluated carrier",
            "55104": "one subdivision of the explicit source and same-connectivity SDF carrier",
            "raw_marching_cubes": "independent field diagnostic; source face-ratio checks do not apply",
        },
        "development_metrics": {
            "expected_final_exact_sdf": expected_development,
            "stored_bridge_exact_sdf": stored_development,
            "stored_exact_comparison": stored_metric_match,
        },
        "evaluation_contract": {
            "renderer": "NvdiffrastRenderer",
            "gpu": "L40S",
            "torch": "2.7.1+cu126",
            "nvdiffrast_revision": "253ac4fcea7de5f396371124af597e6cc957bfae",
            "config_seed": 42,
            "training_provenance_seed": 20260831,
            "development_initialization_sha256": artifacts["development_initialization_comparator"][
                "observed_sha256"
            ],
            "training_motion_initialization_sha256": provenance["initialization_sha256"],
            "note": "training and final-development initialization contexts are intentionally distinct",
        },
        "local_execution": {
            "command": [
                "uv",
                "run",
                "python",
                "scripts/audit_post_v1_e0.py",
                "--output",
                str(OUTPUT_RELATIVE / "e0_report.json"),
            ],
            "environment": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "trimesh": trimesh.__version__,
                "device": "cpu",
            },
            "optimizer_steps": 0,
            "training_frames_fitted": 0,
            "modal_jobs_launched": 0,
            "automatic_paid_retries": 0,
            "immutable_output": str(OUTPUT_RELATIVE / "e0_report.json"),
        },
        "future_e1_statistics": {
            "p10": P10_CONTRACT,
            "paired_bootstrap": PAIRED_BOOTSTRAP_CONTRACT,
        },
        "sealed_test_isolation": {
            "private_evidence_paths_accessed": [],
            "per_frame_rows_accessed": False,
            "metric_reruns": 0,
            "selection_uses_frozen_test_aggregate": False,
            "tracked_acceptance_manifest_read": True,
            "accepted_manifest_sections_used": [
                "candidate",
                "development_validation",
                "reports.modal_sdf_bridge_sha256",
            ],
        },
        "replay_required": True,
    }


def finalize_replay_report(
    binding: dict[str, Any],
    *,
    replay_evaluation: dict[str, Any],
    environment: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    result = dict(binding)
    blockers = [str(value) for value in binding["blockers"]]
    replay_metrics = development_metrics_from_evaluation(replay_evaluation)
    expected = cast(
        dict[str, Any],
        cast(dict[str, Any], binding["development_metrics"])["expected_final_exact_sdf"],
    )
    comparison = exact_metric_comparison(expected, replay_metrics)
    if comparison["status"] != "pass":
        blockers.append("development_replay_not_bit_exact")
    if replay_evaluation.get("status") != "pass":
        blockers.append("development_replay_gate_failed")
    result["development_replay"] = {
        "evaluation": replay_evaluation,
        "normalized_metrics": replay_metrics,
        "exact_comparison": comparison,
    }
    result["execution"] = {
        "environment": environment,
        "command": command,
        "optimizer_steps": 0,
        "training_frames_fitted": 0,
        "automatic_paid_retries": 0,
    }
    result["blockers"] = blockers
    result["status"] = "pass" if not blockers else "fail"
    result["replay_required"] = False
    return result
