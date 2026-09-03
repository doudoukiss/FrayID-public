from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from frayid.io import write_json
from frayid.schemas import (
    CameraHMRFrame,
    DatasetManifest,
    FrameRecord,
    SequenceInitialization,
    VideoMetadata,
)
from frayid.v2.track_factors import PairwiseTrackletFactors, pairwise_sampson_loss
from frayid.v2.turntable import (
    axis_angle_rotation,
    initialize_cooperative_turntable_solution,
    turntable_edge_slots,
    turntable_fundamental_matrices,
)
from frayid.v2.turntable_ba import (
    fit_reduced_turntable,
    make_synthetic_turntable_problem,
    write_turntable_ba_benchmark,
)


def test_public_reduced_turntable_ba_recovers_registered_parameters() -> None:
    problem = make_synthetic_turntable_problem(seed=11)
    result = fit_reduced_turntable(problem)
    assert result["success"] is True
    assert float(result["axis_error_degrees"]) <= 2.0
    assert float(result["center_error_fraction_subject_diagonal"]) <= 0.02
    assert float(result["median_angle_error_degrees"]) <= 2.0
    assert float(result["focal_error_fraction"]) <= 0.02
    assert np.all(np.diff(np.asarray(result["angles_radians"])) > 0)


def test_public_turntable_ba_report_runs_without_private_or_modal_access(tmp_path: Path) -> None:
    output = write_turntable_ba_benchmark(tmp_path / "turntable_ba.json", seed=17)
    report = json.loads(output.read_text())
    assert report["status"] == "pass"
    assert all(report["gates"].values())
    assert report["private_evidence_reads"] == 0
    assert report["development_metrics_read"] == 0
    assert report["sealed_test_accesses"] == 0
    assert report["modal_jobs"] == 0


def test_cooperative_initializer_recovers_full_phase_from_rotation_matrices(
    tmp_path: Path,
) -> None:
    angles = np.linspace(0.0, -2.0 * np.pi, 10)
    rotvec = Rotation.from_euler("z", angles[:, None]).as_rotvec()
    initialization_frames = [
        CameraHMRFrame(
            source_frame_index=index * 5,
            betas=[0.0] * 10,
            body_pose=[0.01 * index] + [0.0] * 68,
            global_orient=rotvec[index].tolist(),
            translation=[0.1 + 0.001 * index, 0.2, 2.3 - 0.001 * index],
            focal_length_px=900.0,
            principal_point_px=[360.0, 560.0],
            bounding_box_xyxy=[100.0, 100.0, 600.0, 1000.0],
            detection_score=0.99,
        )
        for index in range(10)
    ]
    initialization = SequenceInitialization(
        status="refined",
        shared_betas=[0.0] * 10,
        shared_focal_length_px=900.0,
        shared_principal_point_px=[360.0, 560.0],
        image_width=720,
        image_height=1120,
        frames=initialization_frames,
        source_revision="public-fixture",
        checkpoint_sha256="a" * 64,
        camera_checkpoint_sha256="b" * 64,
        detector_checkpoint_sha256="c" * 64,
    )
    records = [
        FrameRecord(
            ordinal=index,
            source_frame_index=index * 5,
            timestamp_seconds=float(index),
            image_path=f"public/frame_{index:04d}.png",
            split="held_out" if index in (0, 5) else "train",
            blur_variance=100.0,
            mean_luminance=120.0,
            quality_accepted=True,
        )
        for index in range(10)
    ]
    manifest = DatasetManifest(
        status="evidence_ready",
        run_id="public-fixture",
        input_video_path="public.mp4",
        input_video_sha256="d" * 64,
        video=VideoMetadata(
            path="public.mp4",
            codec="synthetic",
            width=720,
            height=1120,
            frame_count=50,
            frame_rate=5.0,
            duration_seconds=10.0,
            size_bytes=1,
        ),
        dataset_root="public",
        frames=records,
        train_frame_count=8,
        held_out_frame_count=2,
        rejected_candidate_count=0,
    )
    initialization_path = write_json(tmp_path / "initialization.json", initialization)
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    validation_path = write_json(tmp_path / "validation.json", {"status": "ready", "blockers": []})
    graph_path = write_json(
        tmp_path / "graph.json",
        {"gate_results": {"temporal_track_graph_eligible_for_t01": True}},
    )
    solution = initialize_cooperative_turntable_solution(
        initialization_path,
        manifest_path,
        validation_path,
        graph_path,
        micromotion_rank=2,
    )
    assert solution.status == "qualification_candidate"
    assert len(solution.source_frame_indices) == 8
    assert 1.5 * np.pi < solution.angles_radians[-1] < 2.5 * np.pi
    assert solution.source_provenance["selected_unwrapped_euler_component"] == "z"
    assert solution.angles_radians == sorted(solution.angles_radians)
    increments = np.diff(solution.angles_radians)
    assert np.all(increments > 0)
    assert float(increments.min()) >= 0.25 * float(increments.mean()) - 1.0e-10
    assert float(increments.max()) <= 3.0 * float(increments.mean()) + 1.0e-10


def test_turntable_fundamental_route_matches_object_centric_rotation() -> None:
    import torch

    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    truth_angles = torch.tensor([0.0, 0.25, 0.55], dtype=torch.float64)
    truth_rotations = axis_angle_rotation(axis, truth_angles)
    truth_center = torch.tensor([0.1, -0.05, 4.0], dtype=torch.float64)
    truth_intrinsics = torch.tensor(
        [[700.0, 0.0, 360.0], [0.0, 700.0, 560.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    points = torch.tensor(
        [[-0.3, -0.5, 3.8], [0.4, -0.2, 4.1], [0.2, 0.6, 3.9], [-0.2, 0.4, 4.2]],
        dtype=torch.float64,
    )
    trajectories = (
        torch.einsum("tij,pj->tpi", truth_rotations, points - truth_center) + truth_center
    )
    pixels = torch.stack(
        (
            truth_intrinsics[0, 0] * trajectories[..., 0] / trajectories[..., 2]
            + truth_intrinsics[0, 2],
            truth_intrinsics[1, 1] * trajectories[..., 1] / trajectories[..., 2]
            + truth_intrinsics[1, 2],
        ),
        dim=-1,
    )
    first_slots = torch.tensor([0, 1], dtype=torch.long)
    second_slots = torch.tensor([1, 2], dtype=torch.long)
    exact_fundamental = turntable_fundamental_matrices(
        truth_rotations,
        truth_center,
        truth_intrinsics,
        first_slots,
        second_slots,
    )
    factors = PairwiseTrackletFactors(
        first_ordinals=torch.tensor([1, 2]),
        second_ordinals=torch.tensor([2, 3]),
        first_source_frame_indices=torch.tensor([5, 10]),
        second_source_frame_indices=torch.tensor([10, 15]),
        edge_offsets=torch.tensor([0, 4, 8]),
        first_pixels=torch.cat((pixels[0], pixels[1])).float().detach(),
        second_pixels=torch.cat((pixels[1], pixels[2])).float().detach(),
        observation_weights=torch.ones(8),
        geometric_model_codes=torch.zeros(2, dtype=torch.long),
    )
    bound_first, bound_second = turntable_edge_slots(
        [5, 10, 15],
        factors.first_source_frame_indices,
        factors.second_source_frame_indices,
    )
    assert torch.equal(bound_first, first_slots)
    assert torch.equal(bound_second, second_slots)
    exact_loss = pairwise_sampson_loss(exact_fundamental.float(), factors, image_size=(1120, 720))
    assert float(exact_loss) < 1.0e-8

    angles = torch.tensor([0.0, 0.27, 0.52], dtype=torch.float64, requires_grad=True)
    center = torch.tensor([0.12, -0.05, 4.0], dtype=torch.float64, requires_grad=True)
    intrinsics = truth_intrinsics.clone().requires_grad_(True)
    fundamental = turntable_fundamental_matrices(
        axis_angle_rotation(axis, angles),
        center,
        intrinsics,
        first_slots,
        second_slots,
    )
    loss = pairwise_sampson_loss(fundamental.float(), factors, image_size=(1120, 720))
    loss.backward()
    assert float(loss) > float(exact_loss)
    assert angles.grad is not None and bool(torch.isfinite(angles.grad).all())
    assert float(angles.grad.abs().sum()) > 0
    assert center.grad is not None and bool(torch.isfinite(center.grad).all())
    assert intrinsics.grad is not None and bool(torch.isfinite(intrinsics.grad).all())
