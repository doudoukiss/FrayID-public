"""Run the local topology-safe SDF source-mesh fidelity gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from frayid.io import sha256_file, write_json
from frayid.sdf_distillation import (
    build_topology_safe_sdf_grid,
    evaluate_topology_safe_sdf_fidelity,
    extract_topology_constrained_sdf_mesh,
)


def validate_source_mesh(
    source_mesh_path: Path,
    output_directory: Path,
    *,
    longest_axis_resolution: int,
    occupancy_supersampling: int,
    narrow_band_voxels: float,
    sample_count: int,
) -> dict[str, Any]:
    """Build, extract, measure, and persist one immutable local fidelity run."""
    if not source_mesh_path.is_file():
        raise FileNotFoundError(f"Source mesh does not exist: {source_mesh_path}")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    loaded = trimesh.load_mesh(source_mesh_path, process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("Source mesh path must contain exactly one triangle mesh")
    sdf, origin, pitch = build_topology_safe_sdf_grid(
        loaded,
        longest_axis_resolution=longest_axis_resolution,
        occupancy_supersampling=occupancy_supersampling,
        narrow_band_voxels=narrow_band_voxels,
    )
    extracted, projection = extract_topology_constrained_sdf_mesh(loaded, sdf, origin, pitch)
    repeated, repeated_projection = extract_topology_constrained_sdf_mesh(
        loaded, sdf, origin, pitch
    )
    deterministic = bool(
        np.array_equal(extracted.vertices, repeated.vertices)
        and np.array_equal(extracted.faces, repeated.faces)
        and projection == repeated_projection
    )
    fidelity = evaluate_topology_safe_sdf_fidelity(
        loaded,
        extracted,
        sdf,
        pitch,
        sample_count=sample_count,
    )
    if not deterministic:
        fidelity["status"] = "fail"
        fidelity["blockers"].append("extraction_not_deterministic")

    grid_path = output_directory / "canonical_sdf_grid.npz"
    np.savez_compressed(
        grid_path,
        schema_version="canonical_topology_safe_sdf_grid.v1",
        sdf=sdf,
        origin=origin,
        pitch=np.asarray(pitch, dtype=np.float32),
    )
    mesh_path = output_directory / "canonical_sdf_mesh.ply"
    extracted.export(mesh_path)
    report: dict[str, Any] = {
        "schema_version": "topology_safe_sdf_local_gate.v1",
        "status": fidelity["status"],
        "source": {
            "path": str(source_mesh_path),
            "sha256": sha256_file(source_mesh_path),
            "vertices": len(loaded.vertices),
            "faces": len(loaded.faces),
            "watertight": bool(loaded.is_watertight),
        },
        "method": {
            "sign": "supersampled_conservative_filled_occupancy",
            "distance": "unsigned_closest_triangle_in_narrow_band",
            "longest_axis_resolution": longest_axis_resolution,
            "occupancy_supersampling": occupancy_supersampling,
            "narrow_band_voxels": narrow_band_voxels,
            "extraction": "source_topology_sdf_gradient_projection",
        },
        "projection": projection,
        "grid": {
            "shape": list(sdf.shape),
            "pitch_m": pitch,
            "minimum_sdf_m": float(sdf.min()),
            "maximum_sdf_m": float(sdf.max()),
            "artifact_path": str(grid_path),
            "artifact_sha256": sha256_file(grid_path),
        },
        "mesh": {
            "vertices": len(extracted.vertices),
            "faces": len(extracted.faces),
            "watertight": bool(extracted.is_watertight),
            "deterministic_extraction": deterministic,
            "artifact_path": str(mesh_path),
            "artifact_sha256": sha256_file(mesh_path),
        },
        "fidelity": fidelity,
    }
    write_json(output_directory / "topology_safe_sdf_local_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_mesh", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--occupancy-supersampling", type=int, default=4)
    parser.add_argument("--narrow-band-voxels", type=float, default=3.0)
    parser.add_argument("--sample-count", type=int, default=50_000)
    arguments = parser.parse_args()
    report = validate_source_mesh(
        arguments.source_mesh,
        arguments.output_directory,
        longest_axis_resolution=arguments.resolution,
        occupancy_supersampling=arguments.occupancy_supersampling,
        narrow_band_voxels=arguments.narrow_band_voxels,
        sample_count=arguments.sample_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
