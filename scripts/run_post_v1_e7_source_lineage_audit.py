"""Select the latest same-connectivity exact non-self-intersecting E7 source."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import trimesh

from frayid.interface_field import write_interface_mesh
from frayid.io import sha256_file, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e07_collision_preserving_canonical_r01"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/canonical_clothed_surface_v1/post_v1" / EXPERIMENT_ID
RUNS = PROJECT_ROOT / "outputs/canonical_clothed_surface_v1/runs"
R4K = RUNS / "important_canonical_clothed_surface_v1_r4k_explicit_cage_bounded12_4fbdfd56f7b7"
R4M = RUNS / "important_canonical_clothed_surface_v1_r4m_motion_bounded12_0ebff81c68a7"
R03 = PROJECT_ROOT / (
    "outputs/canonical_clothed_surface_v1/post_v1/"
    "postv1_e00_original_reference_projection_safety_margin_r03/canonical_sdf_carrier_r03.npz"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    path: Path
    source_kind: str


CANDIDATES = (
    Candidate(
        "scaffold",
        PROJECT_ROOT
        / "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/shared_smpl_canonical.npz",
        "npz",
    ),
    Candidate(
        "shared_initialization",
        PROJECT_ROOT
        / (
            "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
            "initialization_artifacts/shared_clothing_envelope.npz"
        ),
        "npz",
    ),
    Candidate(
        "r4j_shared_explicit_initialization",
        RUNS
        / (
            "important_canonical_clothed_surface_v1_r4j_explicit_exact_smoke_bfbf28e5b06f/"
            "explicit_exact_smoke_surface.ply"
        ),
        "ply",
    ),
    Candidate(
        "r4k_smoke",
        RUNS
        / (
            "important_canonical_clothed_surface_v1_r4k_explicit_cage_smoke_51e5e448acd1/"
            "explicit_cage_smoke_surface.ply"
        ),
        "ply",
    ),
    Candidate("r4k_reference", R4K / "explicit_cage_bounded_checkpoint.pt", "checkpoint_base"),
    Candidate("r4k_epoch_3", R4K / "cage_stage_checkpoint_epoch_0003.pt", "checkpoint_cage"),
    Candidate("r4k_epoch_7", R4K / "cage_stage_checkpoint_epoch_0007.pt", "checkpoint_cage"),
    Candidate("r4k_epoch_11", R4K / "cage_stage_checkpoint_epoch_0011.pt", "checkpoint_cage"),
    Candidate("r4k_final", R4K / "explicit_cage_bounded_carrier.npz", "npz"),
    Candidate("r4m_accepted_explicit", R4M / "explicit_motion_carrier.npz", "npz"),
    Candidate("r03_sdf_projected", R03, "npz"),
)


def _build_auditor() -> Path:
    build = PROJECT_ROOT / "build/e7_cgal"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT / "tools/e7_cgal"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "8"], cwd=PROJECT_ROOT, check=True
    )
    executable = build / "frayid_e7_collision_audit"
    if not executable.is_file():
        raise FileNotFoundError("E7 CGAL auditor was not produced")
    return executable


def _load_candidate(candidate: Candidate) -> tuple[np.ndarray, np.ndarray]:
    if not candidate.path.is_file():
        raise FileNotFoundError(candidate.path)
    if candidate.source_kind == "npz":
        with np.load(candidate.path, allow_pickle=False) as archive:
            return (
                np.asarray(archive["vertices"], dtype=np.float64),
                np.asarray(archive["faces"], dtype=np.int64),
            )
    if candidate.source_kind == "ply":
        mesh = cast(trimesh.Trimesh, trimesh.load(candidate.path, process=False))
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    checkpoint = torch.load(candidate.path, map_location="cpu", weights_only=False)
    faces = checkpoint["faces"].detach().cpu().numpy().astype(np.int64, copy=False)
    if candidate.source_kind == "checkpoint_base":
        vertices = checkpoint["base"].detach().cpu().numpy()
        return np.asarray(vertices, dtype=np.float64), faces
    if candidate.source_kind == "checkpoint_cage":
        cage = checkpoint["cage"]
        reference = cage["reference_vertices"]
        controls = cage["controls"].reshape(-1, 3)
        indices = cage["corner_indices"]
        weights = cage["corner_weights"]
        vertices = reference + (controls[indices] * weights[..., None]).sum(dim=1)
        return vertices.detach().cpu().numpy().astype(np.float64, copy=False), faces
    raise ValueError(f"unsupported candidate kind: {candidate.source_kind}")


def _audit(
    candidate: Candidate,
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: np.ndarray,
    executable: Path,
    temporary: Path,
) -> dict[str, Any]:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    padding = max(float(np.max(upper - lower)) * 0.25, 1e-3)
    mesh_path = temporary / f"{candidate.name}.e6mesh"
    report_path = temporary / f"{candidate.name}.json"
    write_interface_mesh(mesh_path, vertices, faces, (lower - padding, upper + padding))
    completed = subprocess.run(
        [str(executable), str(mesh_path), str(report_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if completed.returncode or not report_path.is_file():
        return {
            "name": candidate.name,
            "source_path": str(candidate.path.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_file(candidate.path),
            "status": "audit_error",
            "diagnostic": (completed.stdout + "\n" + completed.stderr).strip(),
            "same_connectivity": bool(np.array_equal(faces, target_faces)),
            "eligible": False,
        }
    report = cast(dict[str, Any], json.loads(report_path.read_text()))
    report.update(
        {
            "name": candidate.name,
            "source_path": str(candidate.path.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_file(candidate.path),
            "same_connectivity": bool(np.array_equal(faces, target_faces)),
        }
    )
    report["exact_self_intersection_free"] = report["intersection_pair_count"] == 0
    report["eligible"] = bool(
        report["same_connectivity"] and report["closed"] and report["exact_self_intersection_free"]
    )
    report["status"] = "pass" if report["exact_self_intersection_free"] else "fail"
    return report


def run_audit(source_revision: str) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    executable = _build_auditor()
    _, target_faces = _load_candidate(CANDIDATES[-2])
    private_candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="frayid-e7-lineage-") as temporary_name:
        temporary = Path(temporary_name)
        for candidate in CANDIDATES:
            vertices, faces = _load_candidate(candidate)
            private_candidates.append(
                _audit(candidate, vertices, faces, target_faces, executable, temporary)
            )
    eligible = [item for item in private_candidates if item.get("eligible") is True]
    selected = eligible[-1]["name"] if eligible else None
    status = "pass" if selected is not None else "fail"
    private = {
        "schema_version": "post_v1_e7_source_lineage_audit.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_revision": source_revision,
        "status": status,
        "selected_source": selected,
        "candidates": private_candidates,
        "blockers": [] if selected else ["no_same_connectivity_exact_embedded_source"],
        "elapsed_seconds": time.monotonic() - started,
        "execution": {
            "source_geometry_reads": len(private_candidates),
            "mesh_repairs": 0,
            "image_loads": 0,
            "optimizer_steps": 0,
            "development_evaluations": 0,
            "sealed_test_accesses": 0,
        },
    }
    public_candidates = [
        {
            "name": item["name"],
            "vertex_count": item.get("vertex_count"),
            "face_count": item.get("face_count"),
            "same_connectivity": item.get("same_connectivity"),
            "closed": item.get("closed"),
            "intersection_pair_count": item.get("intersection_pair_count"),
            "classification": item.get("classification"),
            "eligible": item.get("eligible"),
            "status": item["status"],
        }
        for item in private_candidates
    ]
    public = {
        "schema_version": "post_v1_e7_source_lineage_audit.public.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "selected_source": selected,
        "candidates": public_candidates,
        "blockers": private["blockers"],
        "privacy": "counts_only_no_paths_hashes_face_ids_or_coordinates",
    }
    return private, public


def main() -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relevant = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            "scripts/run_post_v1_e7_source_lineage_audit.py",
            "tools/e7_cgal",
            "configs/evaluation/post_v1_e7_collision_preserving_canonical_r01.yaml",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if relevant:
        raise RuntimeError("E7 source audit requires a clean relevant revision")
    destination = OUTPUT_ROOT / f"source_lineage_audit_{revision[:12]}"
    if destination.exists():
        raise FileExistsError(f"immutable E7 audit already exists: {destination}")
    private, public = run_audit(revision)
    destination.mkdir(parents=True)
    write_json(destination / "private_source_lineage_audit.json", private)
    write_json(destination / "public_source_lineage_counts.json", public)
    print(json.dumps(public, indent=2, sort_keys=True))
    if private["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
