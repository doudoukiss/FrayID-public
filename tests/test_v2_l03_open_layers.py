from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from frayid.cli import app
from frayid.io import read_json, sha256_file, write_json
from frayid.v2.l03_open_layers import (
    _extract_semantic_offset_layers,
    audit_l03_semantic_support,
    boundary_loop_audit,
    build_l03_real_initialization_plan,
    build_l03_semantic_support_plan,
    make_open_wrinkled_tube,
    run_l03_public_benchmark,
    write_l03_public_benchmark,
    write_l03_semantic_support_plan,
)


def test_l03_open_tube_has_two_registered_boundary_loops() -> None:
    pytest.importorskip("ipctk")
    vertices, faces = make_open_wrinkled_tube(
        y_minimum=-0.4,
        y_maximum=0.4,
        base_radius=0.3,
        wrinkle_amplitude=0.02,
        phase=0.5,
    )
    audit = boundary_loop_audit(vertices, faces)
    assert audit["boundary_loop_count"] == 2
    assert audit["boundary_vertex_degrees_two"] is True
    assert audit["watertight"] is False
    assert audit["exact_self_intersections"] is False


def test_l03_public_benchmark_passes_and_rejects_controls() -> None:
    pytest.importorskip("ipctk")
    report = run_l03_public_benchmark()
    assert report["status"] == "pass"
    assert report["gates"]["topology_changing_cap_rejected"] is True
    assert report["gates"]["penetrating_proposal_rejected"] is True
    assert report["aggregate_curve_error"]["relative_improvement"] >= 0.20


def test_l03_public_writer_and_cli_are_immutable(tmp_path: Path) -> None:
    pytest.importorskip("ipctk")
    direct = tmp_path / "direct.json"
    write_l03_public_benchmark(direct)
    assert read_json(direct)["status"] == "pass"
    with pytest.raises(FileExistsError, match="immutable"):
        write_l03_public_benchmark(direct)
    cli_output = tmp_path / "cli.json"
    result = CliRunner().invoke(
        app, ["v2", "benchmark-l03-open-layers", "--output", str(cli_output)]
    )
    assert result.exit_code == 0, result.stdout
    assert read_json(cli_output)["status"] == "pass"


def _write_semantic_support_inputs(tmp_path: Path) -> dict[str, Path]:
    public = tmp_path / "public.json"
    write_json(
        public, {"experiment_id": "postv2_l03_semantic_open_clothing_layers_r01", "status": "pass"}
    )
    s01 = tmp_path / "s01.json"
    write_json(
        s01,
        {
            "status": "pass",
            "training_frame_count": 144,
            "training_images_read": 144,
            "legacy_development_images_read": 0,
            "sealed_test_accesses": 0,
        },
    )
    sources = np.arange(144, dtype=np.int64)
    upper = np.zeros((144, 40, 30), dtype=np.float32)
    lower = np.zeros_like(upper)
    upper[:, 4:14, :] = 0.9
    lower[:, 24:34, :] = 0.8
    semantic_inputs = tmp_path / "semantic_inputs.npz"
    np.savez_compressed(
        semantic_inputs,
        source_frame_indices=sources,
        semantic__upper_clothing=upper,
        semantic__lower_clothing=lower,
        source_hashes=np.asarray('{"semantic_qualification":"' + sha256_file(s01) + '"}'),
    )
    t05 = tmp_path / "t05.json"
    write_json(
        t05,
        {
            "status": "qualification_candidate",
            "training_frame_count": 144,
            "development_records_used_for_fit": 0,
            "sealed_test_reads": 0,
            "frames": [
                {"source_frame_index": int(source), "yaw_radians": float(yaw)}
                for source, yaw in zip(sources, np.linspace(0.0, 2.0 * np.pi, 144), strict=True)
            ],
        },
    )
    return {
        "public_report_path": public,
        "semantic_inputs_path": semantic_inputs,
        "t05_solution_path": t05,
        "s01_qualification_path": s01,
    }


def test_l03_semantic_support_plan_and_audit_pass(tmp_path: Path) -> None:
    inputs = _write_semantic_support_inputs(tmp_path)
    plan = build_l03_semantic_support_plan(**inputs, source_revision="a" * 40)
    assert plan["audit"]["minimum_supported_frames_per_layer"] == 120
    plan_path = tmp_path / "plan.json"
    write_l03_semantic_support_plan(
        **inputs,
        source_revision="a" * 40,
        output_path=plan_path,
    )
    report_path = tmp_path / "report.json"
    audit_l03_semantic_support(plan_path=plan_path, output_path=report_path)
    report = read_json(report_path)
    assert report["status"] == "pass"
    assert report["layers"]["upper_clothing"]["supported_phase_bin_count"] == 12
    assert report["joint_diagnostics"]["upper_above_lower_fraction"] == 1.0
    with pytest.raises(FileExistsError, match="immutable"):
        audit_l03_semantic_support(plan_path=plan_path, output_path=report_path)


def test_l03_semantic_support_rejects_sparse_layer(tmp_path: Path) -> None:
    inputs = _write_semantic_support_inputs(tmp_path)
    with np.load(inputs["semantic_inputs_path"], allow_pickle=False) as archive:
        upper = archive["semantic__upper_clothing"]
        lower = archive["semantic__lower_clothing"]
        source_hashes = archive["source_hashes"]
        sources = archive["source_frame_indices"]
    lower[20:] = 0.0
    np.savez_compressed(
        inputs["semantic_inputs_path"],
        source_frame_indices=sources,
        semantic__upper_clothing=upper,
        semantic__lower_clothing=lower,
        source_hashes=source_hashes,
    )
    plan_path = tmp_path / "plan.json"
    write_l03_semantic_support_plan(
        **inputs,
        source_revision="b" * 40,
        output_path=plan_path,
    )
    report_path = tmp_path / "report.json"
    audit_l03_semantic_support(plan_path=plan_path, output_path=report_path)
    report = read_json(report_path)
    assert report["status"] == "fail"
    assert report["gates"]["minimum_supported_frames"] is False
    assert report["gates"]["complete_supported_phase_coverage"] is False


def _write_real_initialization_inputs(tmp_path: Path) -> dict[str, Path]:
    coordinates = np.linspace(-1.0, 1.0, 32)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    field = np.sqrt(xx**2 + yy**2 + zz**2).astype(np.float32) - 0.45
    d03_field = tmp_path / "d03_field.npz"
    np.savez_compressed(
        d03_field,
        values=field,
        coordinates=coordinates,
        surface_level=np.asarray(0.0, dtype=np.float32),
    )
    from skimage.measure import marching_cubes

    vertices, faces, _, _ = marching_cubes(
        field,
        level=0.0,
        spacing=(coordinates[1] - coordinates[0],) * 3,
        allow_degenerate=False,
    )
    vertices += coordinates[0]
    weights = np.zeros((len(vertices), 24), dtype=np.float32)
    weights[:, 0] = 1.0
    d03_mesh = tmp_path / "d03_mesh.npz"
    np.savez_compressed(d03_mesh, vertices=vertices, faces=faces, skinning_weights=weights)
    d03_report = tmp_path / "d03_report.json"
    write_json(
        d03_report,
        {
            "status": "pass",
            "candidate_topology": {"status": "pass"},
            "artifacts": {
                "continued_field": {"sha256": sha256_file(d03_field)},
                "continued_mesh": {"sha256": sha256_file(d03_mesh)},
            },
        },
    )
    semantic_volume = tmp_path / "semantic_volume.npz"
    upper = np.broadcast_to((yy > -0.05).astype(np.float32), field.shape)
    lower = np.broadcast_to((yy <= 0.05).astype(np.float32), field.shape)
    np.savez_compressed(
        semantic_volume,
        semantic__upper_clothing=upper,
        semantic__lower_clothing=lower,
        metadata=np.asarray('{"resolution":32,"extent":1.0}'),
    )
    hull = tmp_path / "hull.json"
    write_json(
        hull,
        {
            "status": "pass",
            "semantic_layer_status": "bound",
            "source_hashes": {"reference_volume": sha256_file(semantic_volume)},
        },
    )
    support = tmp_path / "support.json"
    write_json(support, {"status": "pass"})
    return {
        "semantic_support_report_path": support,
        "d03_report_path": d03_report,
        "d03_field_path": d03_field,
        "d03_mesh_path": d03_mesh,
        "hull_qualification_path": hull,
        "semantic_volume_path": semantic_volume,
    }


def test_l03_real_initialization_plan_binds_body_and_semantics(tmp_path: Path) -> None:
    inputs = _write_real_initialization_inputs(tmp_path)
    plan = build_l03_real_initialization_plan(**inputs, source_revision="c" * 40)
    assert plan["construction"]["body_clearance_level_metres"] == 0.01
    assert plan["construction"]["registered_boundary_loop_counts"] == {
        "upper_clothing": 1,
        "lower_clothing": 2,
    }
    assert plan["input_hashes"]["d03_mesh"] == sha256_file(inputs["d03_mesh_path"])


def test_l03_semantic_offset_partition_is_open_and_contact_registered(tmp_path: Path) -> None:
    pytest.importorskip("ipctk")
    inputs = _write_real_initialization_inputs(tmp_path)
    with np.load(inputs["d03_field_path"], allow_pickle=False) as archive:
        field = archive["values"]
        coordinates = archive["coordinates"]
    with np.load(inputs["d03_mesh_path"], allow_pickle=False) as archive:
        body_vertices = archive["vertices"]
        body_weights = archive["skinning_weights"]
    with np.load(inputs["semantic_volume_path"], allow_pickle=False) as archive:
        semantics = {
            "upper_clothing": archive["semantic__upper_clothing"],
            "lower_clothing": archive["semantic__lower_clothing"],
        }
    _, _, layers, contact_edges = _extract_semantic_offset_layers(
        field=field,
        coordinates=coordinates,
        semantic_volumes=semantics,
        semantic_extent=1.0,
        body_vertices=body_vertices,
        body_weights=body_weights,
        level=0.01,
        threshold=0.25,
    )
    assert contact_edges > 0
    for layer in layers.values():
        audit = boundary_loop_audit(layer["vertices"], layer["faces"])
        assert audit["watertight"] is False
        assert audit["boundary_loop_count"] >= 1
        assert audit["exact_self_intersections"] is False
        assert np.allclose(layer["skinning_weights"].sum(axis=1), 1.0)
