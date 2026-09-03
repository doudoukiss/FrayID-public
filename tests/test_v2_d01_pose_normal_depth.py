from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from frayid.cli import app
from frayid.io import read_json, write_json
from frayid.v2.d01_pose_normal_depth import (
    D01_EXPERIMENT_ID,
    D01_TRAIN_CANDIDATE_GATES,
    D01_TRAIN_CANDIDATE_SCHEDULE,
    audit_d01_terminal_qualification,
    decode_sapiens_normal_bgr,
    integrate_mesh_normals_along_prior,
    pose_normals_from_canonical,
    run_d01_public_benchmark,
    sample_pose_stabilized_vertex_normals,
    transport_normals_to_canonical,
    write_d01_public_benchmark,
    write_d01_train_candidate_plan,
)


def test_d01_samples_sapiens_normals_and_pulls_them_to_canonical_vertices() -> None:
    vertices = np.asarray(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [0.5, 0.5, 2.0],
            [-0.5, 0.5, 2.0],
        ]
    )
    faces = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    intrinsics = np.asarray([[30.0, 0.0, 15.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]])
    normal_bgr = np.empty((32, 32, 3), dtype=np.uint8)
    normal_bgr[...] = np.asarray([255, 128, 128], dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    decoded = decode_sapiens_normal_bgr(normal_bgr)
    assert np.median(decoded[..., 2]) < -0.99
    normals, confidence, stats = sample_pose_stabilized_vertex_normals(
        vertices,
        vertices,
        faces,
        intrinsics,
        normal_bgr,
        mask,
        source_size=(32, 32),
        erosion_pixels=0,
    )
    assert np.all(confidence > 0.0)
    assert np.all(normals[:, 2] < -0.99)
    assert stats["valid_face_count"] == 2


def test_d01_normal_transport_round_trip_handles_nonrigid_jacobians() -> None:
    normals = np.asarray(
        [[0.2, -0.3, 0.93], [-0.4, 0.1, 0.88]],
        dtype=np.float64,
    )
    angle = math.radians(41.0)
    rotation = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    jacobian = rotation @ np.asarray([[1.08, 0.04, 0.02], [0.0, 0.93, -0.03], [0.01, 0.02, 1.04]])
    transforms = np.broadcast_to(jacobian, (2, 3, 3))
    posed = pose_normals_from_canonical(normals, transforms)
    recovered = transport_normals_to_canonical(posed, transforms)
    expected = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    assert np.allclose(recovered, expected, atol=1.0e-12)


def test_d01_public_benchmark_rejects_frame_and_bias_shortcuts() -> None:
    report = run_d01_public_benchmark()
    assert report["status"] == "pass"
    assert report["experiment_id"] == D01_EXPERIMENT_ID
    assert report["position_rmse_relative_improvement"] >= 0.20
    assert report["point_to_plane_rmse_relative_improvement"] >= 0.20
    assert report["treatment"]["median_normal_degrees"] <= 5.0
    assert report["gates"]["global_bias_shortcut_rejected"] is True
    assert report["gates"]["inverse_pose_transport"] is True
    assert report["provenance"]["measured_depth_claimed"] is False
    assert report["provenance"]["development_records_read"] == 0
    assert report["provenance"]["sealed_test_reads"] == 0


def test_d01_public_benchmark_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "d01_public.json"
    assert write_d01_public_benchmark(path) == path
    assert read_json(path)["status"] == "pass"
    with pytest.raises(FileExistsError, match="immutable"):
        write_d01_public_benchmark(path)


def test_d01_public_benchmark_cli(tmp_path: Path) -> None:
    output = tmp_path / "cli_d01_public.json"
    result = CliRunner().invoke(
        app,
        ["v2", "benchmark-d01-pose-normal-depth", "--output", str(output)],
    )
    assert result.exit_code == 0, result.stdout
    assert read_json(output)["status"] == "pass"


def test_d01_mesh_integration_is_bounded_and_deterministic() -> None:
    vertices = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (4, 1))
    first, displacement = integrate_mesh_normals_along_prior(
        vertices,
        faces,
        normals,
        normals,
        np.ones(4),
        maximum_displacement_metres=0.035,
        normal_equation_weight=1.0,
        prior_anchor_weight=0.08,
        edge_smoothness_weight=0.04,
    )
    second, second_displacement = integrate_mesh_normals_along_prior(
        vertices,
        faces,
        normals,
        normals,
        np.ones(4),
        maximum_displacement_metres=0.035,
        normal_equation_weight=1.0,
        prior_anchor_weight=0.08,
        edge_smoothness_weight=0.04,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(displacement, second_displacement)
    assert np.max(np.abs(displacement)) <= 0.035
    assert np.allclose(first, vertices)


def test_d01_train_candidate_plan_freezes_schedule_and_gates(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    write_json(
        evidence,
        {
            "status": "train_only_evidence_bound",
            "development_records_read": 0,
            "sealed_test_reads": 0,
        },
    )
    output = tmp_path / "plan.json"
    write_d01_train_candidate_plan(evidence, output, source_revision="a" * 40)
    plan = read_json(output)
    assert plan["schedule"] == D01_TRAIN_CANDIDATE_SCHEDULE
    assert plan["training_gates"] == D01_TRAIN_CANDIDATE_GATES
    assert plan["development_records_authorized"] == 0


def test_d01_terminal_audit_requires_topology_failure_and_zero_private_progress(
    tmp_path: Path,
) -> None:
    public = tmp_path / "public.json"
    evidence = tmp_path / "evidence.json"
    plan = tmp_path / "plan.json"
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate = candidate_root / "candidate_report.json"
    write_json(public, {"status": "pass", "provenance": {"private_records_read": 0}})
    write_json(
        evidence,
        {
            "status": "train_only_evidence_bound",
            "training_records_read": 144,
            "development_records_read": 0,
            "sealed_test_reads": 0,
        },
    )
    write_json(plan, {"status": "frozen_before_candidate_fit"})
    write_json(
        candidate,
        {
            "status": "candidate_failed_precheck",
            "exact_solve_replay": True,
            "connectivity_exactly_frozen": True,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "topology_precheck": {"flipped_face_count": 2, "collapsed_face_count": 1},
        },
    )
    output = tmp_path / "terminal.json"
    audit_d01_terminal_qualification(public, evidence, plan, candidate, output)
    result = read_json(output)
    assert result["decision"] == "terminal_failed_train_topology_precheck"
    assert result["training_evaluation_run"] is False
    assert result["scientific_attempts_started"] == 0
