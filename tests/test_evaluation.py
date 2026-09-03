from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from frayid.config import load_config
from frayid.evaluation import (
    EvaluatedGeometry,
    _interpolate_frame_code,
    _interpolate_joint_transforms,
    evaluate_reconstruction,
    evaluated_geometry_topology_report,
    load_evaluation_checkpoint,
    robustly_smooth_joint_transforms,
)
from frayid.io import write_json
from frayid.replay_state import SamplerState, capture_checkpoint_state


def test_held_out_frame_code_is_temporally_interpolated() -> None:
    codes = torch.tensor([[0.0, 2.0], [10.0, 12.0], [20.0, 22.0]])
    assert torch.equal(_interpolate_frame_code(0, [0, 10, 20], codes), codes[0])
    assert torch.allclose(_interpolate_frame_code(5, [0, 10, 20], codes), torch.tensor([5.0, 7.0]))
    assert torch.equal(_interpolate_frame_code(30, [0, 10, 20], codes), codes[-1])


def test_held_out_joint_transforms_use_rigid_interpolation() -> None:
    transforms = torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1).numpy()
    transforms[1, 0, 0, 0] = 0.0
    transforms[1, 0, 0, 2] = 1.0
    transforms[1, 0, 2, 0] = -1.0
    transforms[1, 0, 2, 2] = 0.0
    transforms[1, 0, 0, 3] = 2.0
    midpoint = _interpolate_joint_transforms(5, [0, 10], transforms, {0: 0, 10: 1})
    assert midpoint[0, 0, 3] == pytest.approx(1.0)
    assert midpoint[0, 0, 0] == pytest.approx(2**-0.5, abs=1e-6)
    assert midpoint[0, 0, 2] == pytest.approx(2**-0.5, abs=1e-6)


def test_robust_transform_smoothing_repairs_rotation_outlier_without_touching_sequence_ends() -> (
    None
):
    transforms = torch.eye(4).reshape(1, 1, 4, 4).repeat(9, 2, 1, 1).numpy()
    # Give the non-outlier frames small, non-zero second differences so MAD is defined.
    angles = [0.0, 1.0, 0.5, 1.5, 60.0, 2.0, 1.5, 2.5, 2.0]
    for ordinal, angle in enumerate(angles):
        radians = torch.deg2rad(torch.tensor(angle)).item()
        rotation = torch.tensor(
            [
                [torch.cos(torch.tensor(radians)), -torch.sin(torch.tensor(radians)), 0.0],
                [torch.sin(torch.tensor(radians)), torch.cos(torch.tensor(radians)), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transforms[ordinal, :, :3, :3] = rotation.numpy()
    smoothed, report = robustly_smooth_joint_transforms(
        torch.arange(9).numpy(), transforms, mad_multiplier=6.0
    )
    assert 4 in report["repaired_ordinals"]
    assert smoothed[4, 0, 0, 0] > 0.999
    assert smoothed[0, 0, 0, 0] == pytest.approx(transforms[0, 0, 0, 0])
    assert smoothed[-1, 0, 0, 0] == pytest.approx(transforms[-1, 0, 0, 0])


def test_reconstruction_gate_uses_held_out_metrics_and_mesh_components(tmp_path: Path) -> None:
    config = load_config()
    mesh_path = tmp_path / "canonical.ply"
    trimesh.creation.icosphere(subdivisions=1).export(mesh_path)
    metrics_path = tmp_path / "metrics.json"
    write_json(
        metrics_path,
        {
            "train_silhouette_iou": 0.88,
            "held_out_silhouette_iou": 0.86,
            "initialization_held_out_iou": 0.74,
            "normalized_boundary_error": 0.01,
            "median_normal_error_degrees": 20.0,
        },
    )
    report = evaluate_reconstruction(config, metrics_path=metrics_path, mesh_path=mesh_path)
    assert report.status == "pass"
    assert report.dominant_component_area_fraction == 1.0
    assert report.canonical_mesh_watertight is True

    payload = {
        "train_silhouette_iou": 0.95,
        "held_out_silhouette_iou": 0.80,
        "initialization_held_out_iou": 0.75,
        "normalized_boundary_error": 0.03,
        "median_normal_error_degrees": 35.0,
    }
    write_json(metrics_path, payload)
    failed = evaluate_reconstruction(config, metrics_path=metrics_path, mesh_path=mesh_path)
    assert failed.status == "fail"
    assert "held_out_iou_below_gate" in failed.blockers
    assert "train_held_out_gap_above_gate" in failed.blockers


def test_evaluation_checkpoint_reads_v1_and_v2_without_runtime_restore(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    legacy = tmp_path / "legacy.pt"
    torch.save({"schema_version": "canonical_checkpoint.v1", "model": model.state_dict()}, legacy)
    legacy_view = load_evaluation_checkpoint(legacy, torch.device("cpu"))
    assert legacy_view.schema_version == "canonical_checkpoint.v1"
    assert legacy_view.next_step_replay_capable is False

    current = tmp_path / "current.pt"
    state = capture_checkpoint_state(
        model,
        optimizer,
        epoch=2,
        global_step=3,
        stage="fixture",
        sampler_state=SamplerState([0], 0),
    )
    torch.save(state.state_dict(), current)
    current_view = load_evaluation_checkpoint(current, torch.device("cpu"))
    assert current_view.schema_version == "canonical_checkpoint.v2"
    assert current_view.next_step_replay_capable is True
    assert current_view.model_state.keys() == legacy_view.model_state.keys()


def test_actual_external_geometry_is_audited_without_checkpoint_face_reference() -> None:
    checkpoint = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    external = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    geometry = EvaluatedGeometry(
        vertices=torch.tensor(external.vertices, dtype=torch.float32),
        faces=torch.tensor(external.faces, dtype=torch.long),
        weights=torch.ones((len(external.vertices), 1)),
        schema_version="embedded_carrier.v1",
        source_kind="archive",
    )
    report = evaluated_geometry_topology_report(
        geometry,
        checkpoint_reference_vertices=checkpoint.vertices,
        checkpoint_faces=checkpoint.faces,
        minimum_area_ratio=0.1,
    )
    assert report["status"] == "pass"
    assert report["same_connectivity_as_checkpoint"] is False
    assert report["relative_face_report"] is None
    exported_vertices, exported_faces, _ = geometry.numpy_arrays()
    assert np.array_equal(exported_vertices, geometry.vertices.numpy())
    assert np.array_equal(exported_faces, geometry.faces.numpy())


def test_actual_external_geometry_component_failure_is_not_hidden() -> None:
    checkpoint = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    first = checkpoint.copy()
    second = checkpoint.copy()
    second.apply_translation((3.0, 0.0, 0.0))
    external = trimesh.util.concatenate((first, second))
    geometry = EvaluatedGeometry(
        vertices=torch.tensor(external.vertices, dtype=torch.float32),
        faces=torch.tensor(external.faces, dtype=torch.long),
        weights=torch.ones((len(external.vertices), 1)),
        schema_version="embedded_carrier.v1",
        source_kind="archive",
    )
    report = evaluated_geometry_topology_report(
        geometry,
        checkpoint_reference_vertices=checkpoint.vertices,
        checkpoint_faces=checkpoint.faces,
        minimum_area_ratio=0.1,
    )
    assert report["status"] == "fail"
    assert "actual_geometry_component_count_not_one" in report["blockers"]
