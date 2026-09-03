"""Run the public-only E9 tracklet outlier-process gate."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from frayid.geometry import vertex_normals
from frayid.io import write_json
from frayid.material_tracks import (
    TrackletAssignment,
    pseudo_huber,
    segment_tracklets,
    tracklet_redescending_loss,
    tracklet_reliability,
)
from frayid.replay_state import (
    SamplerState,
    capture_checkpoint_state,
    nested_state_equal,
    restore_checkpoint_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e09_track_outlier_process_r01"
SEED = 20260831
FRAME_COUNT = 18
GRID_HEIGHT = 6
GRID_WIDTH = 8
STEPS = 120
LEARNING_RATE = 0.025
TRACK_WEIGHT = 7.0
PSEUDO_HUBER_DELTA = 0.025
LAMBDA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SCHEDULE_MULTIPLIERS = (4.0, 2.0, 1.0)


@dataclass(frozen=True)
class Fixture:
    base: Tensor
    truth: Tensor
    truth_motion: Tensor
    faces: Tensor
    image_centroids: Tensor
    image_spread: Tensor
    clean_observations: Tensor
    corrupted_observations: Tensor
    weights: Tensor
    assignment: TrackletAssignment
    corrupted_tracklets: Tensor


class FixtureModel(nn.Module):
    def __init__(self, fixture: Fixture) -> None:
        super().__init__()
        self.canonical = nn.Parameter(fixture.base.clone())
        self.motion = nn.Parameter(torch.zeros_like(fixture.truth_motion))

    def posed(self, frames: Tensor) -> Tensor:
        return self.canonical[None] + self.motion[frames, None]


def _faces() -> Tensor:
    result: list[tuple[int, int, int]] = []
    for row in range(GRID_HEIGHT - 1):
        for column in range(GRID_WIDTH - 1):
            a = row * GRID_WIDTH + column
            b = a + 1
            c = a + GRID_WIDTH
            d = c + 1
            result.extend(((a, b, d), (a, d, c)))
    return torch.tensor(result, dtype=torch.long)


def make_fixture() -> Fixture:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, GRID_HEIGHT, dtype=torch.float64),
        torch.linspace(-1.2, 1.2, GRID_WIDTH, dtype=torch.float64),
        indexing="ij",
    )
    base = torch.stack((x, y, 0.04 * torch.sin(2.0 * x)), dim=-1).reshape(-1, 3)
    fold = 0.20 * torch.sin(math.pi * x / 1.2) * torch.cos(1.5 * math.pi * y)
    shear = 0.055 * torch.sin(math.pi * y) * torch.exp(-0.7 * x.square())
    truth = base.clone()
    truth[:, 0] += shear.reshape(-1)
    truth[:, 2] += fold.reshape(-1)
    time = torch.linspace(0.0, 1.0, FRAME_COUNT, dtype=torch.float64)
    truth_motion = torch.stack(
        (
            0.11 * torch.sin(2.0 * math.pi * time),
            0.07 * torch.cos(4.0 * math.pi * time + 0.2),
            0.09 * torch.sin(3.0 * math.pi * time - 0.3),
        ),
        dim=-1,
    )
    clean = truth[None] + truth_motion[:, None]
    valid = torch.ones((FRAME_COUNT, truth.shape[0]), dtype=torch.bool)
    valid[7:9, ::7] = False
    forward_backward = torch.zeros_like(valid, dtype=torch.float64)
    cycle = torch.zeros_like(forward_backward)
    forward_backward[5, 3::11] = 2.0
    cycle[12, 4::13] = 2.0
    occlusion = torch.zeros_like(valid)
    occlusion[9, ::5] = True
    assignment = segment_tracklets(
        valid,
        forward_backward,
        cycle,
        occlusion,
        maximum_forward_backward_error=0.5,
        maximum_cycle_error=0.5,
    )
    weights = (assignment.ids >= 0).to(torch.float64)
    corrupted = clean.clone()
    corrupt_points = torch.arange(truth.shape[0]) % 5 == 0
    phase = torch.linspace(0.0, 2.0 * math.pi, FRAME_COUNT, dtype=torch.float64)
    offset = torch.stack(
        (
            0.65 + 0.18 * torch.sin(phase),
            -0.52 + 0.14 * torch.cos(2.0 * phase),
            0.48 * torch.cos(phase - 0.4),
        ),
        dim=-1,
    )
    corrupted[:, corrupt_points] += offset[:, None]
    corrupted_tracklets = torch.zeros(assignment.tracklet_count, dtype=torch.bool)
    for tracklet_id in range(assignment.tracklet_count):
        selected = assignment.ids == tracklet_id
        corrupted_tracklets[tracklet_id] = bool(selected[:, corrupt_points].any())
    centroid_noise = 0.035 * torch.randn((FRAME_COUNT, 3), generator=generator, dtype=torch.float64)
    image_centroids = clean.mean(dim=1) + centroid_noise
    centered = clean - clean.mean(dim=1, keepdim=True)
    image_spread = torch.sqrt(centered.square().mean(dim=1) + 1e-12)
    return Fixture(
        base=base,
        truth=truth,
        truth_motion=truth_motion,
        faces=_faces(),
        image_centroids=image_centroids,
        image_spread=image_spread,
        clean_observations=clean,
        corrupted_observations=corrupted,
        weights=weights,
        assignment=assignment,
        corrupted_tracklets=corrupted_tracklets,
    )


def _observation_penalties(predicted: Tensor, observed: Tensor) -> Tensor:
    residual = torch.sqrt((predicted - observed).square().sum(dim=-1) + 1e-12)
    return pseudo_huber(residual, delta=PSEUDO_HUBER_DELTA)


def _lambda_for_step(base_lambda: float, step: int) -> float:
    section = min(2, 3 * step // STEPS)
    return base_lambda * SCHEDULE_MULTIPLIERS[section]


def _objective(
    model: FixtureModel,
    fixture: Fixture,
    frames: Tensor,
    *,
    observations: Tensor,
    mode: Literal["no_track", "mean", "redescending"],
    lambda_value: float,
) -> Tensor:
    posed = model.posed(frames)
    centroid = posed.mean(dim=1)
    centered = posed - centroid[:, None]
    spread = torch.sqrt(centered.square().mean(dim=1) + 1e-12)
    image_loss = (centroid - fixture.image_centroids[frames]).square().mean()
    image_loss = image_loss + 0.5 * (spread - fixture.image_spread[frames]).square().mean()
    regularizer = 0.004 * (model.canonical - fixture.base).square().mean()
    motion_smoothness = 0.015 * (model.motion[1:] - model.motion[:-1]).square().mean()
    gauge = 0.2 * model.motion.mean(dim=0).square().mean()
    objective = image_loss + regularizer + motion_smoothness + gauge
    if mode == "no_track":
        return objective
    penalties = _observation_penalties(posed, observations[frames])
    weights = fixture.weights[frames]
    if mode == "mean":
        track_loss = (penalties * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        local_ids = fixture.assignment.ids[frames]
        local_assignment = TrackletAssignment(local_ids, fixture.assignment.tracklet_count)
        track_loss, _ = tracklet_redescending_loss(
            penalties,
            weights,
            local_assignment,
            lambda_value=lambda_value,
        )
    return objective + TRACK_WEIGHT * track_loss


def _metrics(model: FixtureModel, fixture: Fixture) -> dict[str, float]:
    canonical = model.canonical.detach()
    distances = torch.cdist(canonical, fixture.truth)
    chamfer = 0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
    motion_rmse = torch.sqrt((model.motion.detach() - fixture.truth_motion).square().mean() + 1e-12)
    predicted_normals = vertex_normals(canonical, fixture.faces)
    truth_normals = vertex_normals(fixture.truth, fixture.faces)
    cosine = (predicted_normals * truth_normals).sum(dim=-1).clamp(-1.0, 1.0)
    normal = torch.rad2deg(torch.acos(cosine)).median()
    return {
        "canonical_chamfer": float(chamfer),
        "motion_rmse": float(motion_rmse),
        "pooled_normal_error_degrees": float(normal),
    }


def _train(
    fixture: Fixture,
    *,
    observations: Tensor,
    mode: Literal["no_track", "mean", "redescending"],
    base_lambda: float,
    verify_replay: bool = False,
) -> tuple[FixtureModel, dict[str, Any]]:
    torch.manual_seed(SEED)
    model = FixtureModel(fixture)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    sampler = SamplerState(list(range(STEPS)))

    def advance(step: int) -> dict[str, Any]:
        frames = torch.randperm(FRAME_COUNT, generator=generator)[:6]
        optimizer.zero_grad(set_to_none=True)
        loss = _objective(
            model,
            fixture,
            frames,
            observations=observations,
            mode=mode,
            lambda_value=_lambda_for_step(base_lambda, step),
        )
        loss.backward()  # type: ignore[no-untyped-call]
        gradients: list[Tensor] = []
        for parameter in model.parameters():
            if parameter.grad is None:
                raise RuntimeError("E9 public fixture produced a missing gradient")
            gradients.append(parameter.grad.detach().clone())
        optimizer.step()
        return {
            "frames": frames.clone(),
            "loss": loss.detach().clone(),
            "gradients": gradients,
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }

    replay_exact = True
    while not sampler.complete:
        step = sampler.take()
        if verify_replay and step == 23:
            checkpoint = capture_checkpoint_state(
                model,
                optimizer,
                epoch=0,
                global_step=step,
                stage="e9_public_fixture",
                sampler_state=sampler,
                named_generators={"batch": generator},
                immutable_bindings={"experiment": EXPERIMENT_ID},
            )
            uninterrupted = advance(step)
            restored_model = FixtureModel(fixture)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=LEARNING_RATE)
            restored_generator = torch.Generator(device="cpu")
            restore_checkpoint_state(
                checkpoint,
                restored_model,
                restored_optimizer,
                named_generators={"batch": restored_generator},
                expected_immutable_bindings={"experiment": EXPERIMENT_ID},
            )
            original_model, original_optimizer, original_generator = model, optimizer, generator
            model, optimizer, generator = restored_model, restored_optimizer, restored_generator
            replayed = advance(step)
            replay_exact = nested_state_equal(uninterrupted, replayed)
            model, optimizer, generator = original_model, original_optimizer, original_generator
            continue
        advance(step)
    endpoint = _metrics(model, fixture)
    endpoint["exact_next_step_replay"] = replay_exact
    return model, endpoint


def _endpoint_classification(
    model: FixtureModel,
    fixture: Fixture,
    observations: Tensor,
    *,
    lambda_value: float,
) -> tuple[float, float, list[float]]:
    penalties = _observation_penalties(model.posed(torch.arange(FRAME_COUNT)), observations)
    _, sums = tracklet_redescending_loss(
        penalties,
        fixture.weights,
        fixture.assignment,
        lambda_value=lambda_value,
    )
    reliability = tracklet_reliability(sums, lambda_value=lambda_value)
    retained = reliability >= 0.25
    clean = ~fixture.corrupted_tracklets
    clean_retention = float(retained[clean].to(torch.float64).mean())
    wrong_rejection = float((~retained[fixture.corrupted_tracklets]).to(torch.float64).mean())
    return clean_retention, wrong_rejection, reliability.tolist()


def _no_worse(candidate: dict[str, Any], comparator: dict[str, Any]) -> bool:
    return all(
        candidate[key] <= comparator[key] + 1e-12
        for key in ("canonical_chamfer", "motion_rmse", "pooled_normal_error_degrees")
    )


def run_gate(source_revision: str) -> dict[str, Any]:
    fixture = make_fixture()
    initial = FixtureModel(fixture)
    initial_penalties = _observation_penalties(
        initial.posed(torch.arange(FRAME_COUNT)), fixture.clean_observations
    )
    _, initial_sums = tracklet_redescending_loss(
        initial_penalties,
        fixture.weights,
        fixture.assignment,
        lambda_value=1.0,
    )
    lambda_scale = float(initial_sums.detach().median())
    no_track_model, no_track = _train(
        fixture,
        observations=fixture.corrupted_observations,
        mode="no_track",
        base_lambda=lambda_scale,
    )
    del no_track_model
    _, clean = _train(
        fixture,
        observations=fixture.clean_observations,
        mode="mean",
        base_lambda=lambda_scale,
        verify_replay=True,
    )
    mean_corrupt_model, mean_corrupt = _train(
        fixture,
        observations=fixture.corrupted_observations,
        mode="mean",
        base_lambda=lambda_scale,
    )
    del mean_corrupt_model
    candidate_reports: list[dict[str, Any]] = []
    selected: tuple[float, FixtureModel, dict[str, Any], float, float] | None = None
    for multiplier in LAMBDA_MULTIPLIERS:
        base_lambda = lambda_scale * multiplier
        model, metrics = _train(
            fixture,
            observations=fixture.corrupted_observations,
            mode="redescending",
            base_lambda=base_lambda,
            verify_replay=True,
        )
        clean_retention, wrong_rejection, reliability = _endpoint_classification(
            model,
            fixture,
            fixture.corrupted_observations,
            lambda_value=base_lambda,
        )
        qualifies = (
            clean_retention >= 0.80
            and wrong_rejection >= 0.80
            and bool(metrics["exact_next_step_replay"])
            and _no_worse(metrics, no_track)
            and _no_worse(metrics, mean_corrupt)
        )
        candidate_reports.append(
            {
                "multiplier": multiplier,
                "lambda": base_lambda,
                "metrics": metrics,
                "clean_observation_retention_fraction": clean_retention,
                "wrong_observation_rejection_fraction": wrong_rejection,
                "reliability_sha256": hashlib.sha256(
                    json.dumps(reliability, separators=(",", ":")).encode()
                ).hexdigest(),
                "qualifies": qualifies,
            }
        )
        if qualifies:
            selected = (multiplier, model, metrics, clean_retention, wrong_rejection)

    chamfer_improvement = (no_track["canonical_chamfer"] - clean["canonical_chamfer"]) / no_track[
        "canonical_chamfer"
    ]
    motion_improvement = (no_track["motion_rmse"] - clean["motion_rmse"]) / no_track["motion_rmse"]
    normal_improvement = (
        no_track["pooled_normal_error_degrees"] - clean["pooled_normal_error_degrees"]
    )
    blockers: list[str] = []
    if chamfer_improvement < 0.20:
        blockers.append("clean_chamfer_improvement_below_twenty_percent")
    if motion_improvement < 0.10:
        blockers.append("clean_motion_improvement_below_ten_percent")
    if normal_improvement < 2.0:
        blockers.append("clean_pooled_normal_improvement_below_two_degrees")
    if not clean["exact_next_step_replay"]:
        blockers.append("clean_exact_replay_failed")
    if selected is None:
        blockers.append("no_lambda_multiplier_qualified")
    selected_report: dict[str, Any] | None = None
    if selected is not None:
        multiplier, _, metrics, clean_retention, wrong_rejection = selected
        selected_report = {
            "multiplier": multiplier,
            "lambda": lambda_scale * multiplier,
            "metrics": metrics,
            "clean_observation_retention_fraction": clean_retention,
            "wrong_observation_rejection_fraction": wrong_rejection,
        }
    return {
        "schema_version": "post_v1_e9_public_gate.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "source_revision": source_revision,
        "seed": SEED,
        "fixture": {
            "description": "public_folded_grid_with_train_only_tracklet_breaks_and_consistent_identity_outliers",
            "frames": FRAME_COUNT,
            "vertices": int(fixture.truth.shape[0]),
            "tracklets": fixture.assignment.tracklet_count,
            "corrupted_tracklets": int(fixture.corrupted_tracklets.sum()),
            "private_inputs": 0,
            "human_evidence_accesses": 0,
        },
        "registered_selection": {
            "lambda_scale": lambda_scale,
            "multiplier_grid": list(LAMBDA_MULTIPLIERS),
            "selection": "largest_qualifying_multiplier",
            "graduated_schedule_multipliers": list(SCHEDULE_MULTIPLIERS),
        },
        "metrics": {
            "no_track_control": no_track,
            "clean_mean_pseudo_huber": clean,
            "corrupted_mean_pseudo_huber_control": mean_corrupt,
            "clean_chamfer_improvement_fraction": chamfer_improvement,
            "clean_motion_improvement_fraction": motion_improvement,
            "clean_pooled_normal_improvement_degrees": normal_improvement,
            "lambda_candidates": candidate_reports,
            "selected": selected_report,
        },
        "execution": {
            "device": "cpu_float64",
            "public_synthetic_only": True,
            "private_rgb_frames_processed": 0,
            "development_evaluations": 0,
            "sealed_test_accesses": 0,
            "automatic_paid_retries": 0,
        },
    }


def main() -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "src", "configs", __file__],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode
    if dirty:
        raise RuntimeError("Refusing official E9 gate from a dirty relevant worktree")
    result = run_gate(revision)
    destination = (
        PROJECT_ROOT
        / "outputs/canonical_clothed_surface_v1/post_v1"
        / EXPERIMENT_ID
        / f"public_gate_{revision[:12]}"
    )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable E9 result: {destination}")
    destination.mkdir(parents=True)
    write_json(destination / "public_gate_report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
