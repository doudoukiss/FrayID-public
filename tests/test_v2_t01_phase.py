from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from frayid.io import write_json
from frayid.v2.schemas import TurntableSolution
from frayid.v2.t01_phase import (
    BoundedPhaseAxisBlock,
    qualify_real_phase_axis_step,
    take_phase_axis_trust_region_step,
)
from frayid.v2.track_factors import PairwiseTrackletFactors
from frayid.v2.turntable import axis_angle_rotation


def _fixture() -> tuple[TurntableSolution, PairwiseTrackletFactors]:
    torch.manual_seed(7)
    frame_count = 12
    point_count = 32
    truth_angles = torch.linspace(0.0, 2.0 * math.pi, frame_count)
    base_angles = truth_angles + 0.055 * torch.sin(2.0 * truth_angles)
    axis = torch.tensor([0.0, 1.0, 0.0])
    center = torch.tensor([0.08, -0.03, 4.0])
    intrinsics = torch.tensor([[700.0, 0.0, 360.0], [0.0, 700.0, 560.0], [0.0, 0.0, 1.0]])
    points = center + (torch.rand(point_count, 3) - 0.5) * torch.tensor([0.8, 1.6, 0.5])
    rotations = axis_angle_rotation(axis, truth_angles)
    trajectories = torch.einsum("tij,pj->tpi", rotations, points - center) + center
    pixels = torch.stack(
        (
            intrinsics[0, 0] * trajectories[..., 0] / trajectories[..., 2] + intrinsics[0, 2],
            intrinsics[1, 1] * trajectories[..., 1] / trajectories[..., 2] + intrinsics[1, 2],
        ),
        dim=-1,
    )
    edge_count = frame_count - 1
    factors = PairwiseTrackletFactors(
        first_ordinals=torch.arange(edge_count, dtype=torch.long),
        second_ordinals=torch.arange(1, frame_count, dtype=torch.long),
        first_source_frame_indices=torch.arange(edge_count, dtype=torch.long) * 5,
        second_source_frame_indices=torch.arange(1, frame_count, dtype=torch.long) * 5,
        edge_offsets=torch.arange(0, frame_count * point_count, point_count, dtype=torch.long),
        first_pixels=pixels[:-1].reshape(-1, 2).detach(),
        second_pixels=pixels[1:].reshape(-1, 2).detach(),
        observation_weights=torch.ones(edge_count * point_count),
        geometric_model_codes=torch.zeros(edge_count, dtype=torch.long),
    )
    solution = TurntableSolution(
        status="qualification_candidate",
        shared_intrinsics=intrinsics.tolist(),
        axis=axis.tolist(),
        center=center.tolist(),
        angles_radians=base_angles.tolist(),
        residual_twists=[[0.0] * 6 for _ in range(frame_count)],
        micromotion_basis=[],
        micromotion_codes=[[] for _ in range(frame_count)],
        source_frame_indices=(torch.arange(frame_count) * 5).tolist(),
        gauge_policy={"fixture": "public"},
        uncertainty={},
        source_provenance={"role": "public_fixture"},
    )
    return solution, factors


def _write_binding(path: Path, factors: PairwiseTrackletFactors) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray("frayid_v2_pairwise_tracklet_factors.v1"),
        first_ordinals=factors.first_ordinals.numpy(),
        second_ordinals=factors.second_ordinals.numpy(),
        first_source_frame_indices=factors.first_source_frame_indices.numpy(),
        second_source_frame_indices=factors.second_source_frame_indices.numpy(),
        edge_offsets=factors.edge_offsets.numpy(),
        first_pixels=factors.first_pixels.numpy(),
        second_pixels=factors.second_pixels.numpy(),
        observation_weights=factors.observation_weights.numpy(),
        geometric_model_codes=factors.geometric_model_codes.numpy().astype(np.uint8),
    )


def test_bounded_phase_step_improves_public_factors_and_preserves_full_turn() -> None:
    solution, factors = _fixture()
    first = BoundedPhaseAxisBlock(solution, factors)
    second = BoundedPhaseAxisBlock(solution, factors)
    first_step = take_phase_axis_trust_region_step(first, factors, image_size=(1120, 720))
    second_step = take_phase_axis_trust_region_step(second, factors, image_size=(1120, 720))
    assert first_step == second_step
    assert first_step.evidence_improvement_fraction >= 0.01
    assert first_step.maximum_increment_relative_change <= 0.20
    assert first_step.axis_tilt_degrees <= 5.0
    torch.testing.assert_close(first.angles[-1], first.base_angles[-1], rtol=0.0, atol=1e-6)
    assert bool(torch.all(first.angles.diff() > 0))
    for first_value, second_value in zip(
        first.state_dict().values(), second.state_dict().values(), strict=True
    ):
        assert torch.equal(first_value, second_value)


def test_phase_qualification_checkpoint_restores_and_replays(tmp_path: Path) -> None:
    solution, factors = _fixture()
    solution_path = write_json(tmp_path / "solution.json", solution)
    binding_path = tmp_path / "factors.npz"
    _write_binding(binding_path, factors)
    report_path = qualify_real_phase_axis_step(
        solution_path,
        binding_path,
        tmp_path / "report.json",
        tmp_path / "checkpoint.pt",
        image_size=(1120, 720),
    )
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert report["checkpoint_restore_exact"] is True
    assert report["same_device_replay_exact"] is True
    assert report["optimizer_steps"] == 1
    assert report["scientific_attempt_marker_created"] is False
    assert report["sealed_test_accesses"] == 0
