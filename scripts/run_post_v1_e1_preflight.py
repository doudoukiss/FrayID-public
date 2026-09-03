"""Run the public synthetic E1 continuous-time motion preflight."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch import Tensor, nn

from frayid.continuous_time import (
    CubicControlTrajectory,
    initialize_cubic_controls,
    interpolate_slots,
)
from frayid.geometry import linear_blend_skinning, rigid_transform_from_axis_angle
from frayid.io import write_json

EXPERIMENT_ID = "postv1_e01_continuous_time_motion_r01"
OUTPUT_RELATIVE = Path("outputs/canonical_clothed_surface_v1/post_v1") / EXPERIMENT_ID
SEED = 20260831
CONTROL_COUNT = 32
STEP_COUNT = 400
LEARNING_RATE = 0.01


def _tensor_hash(value: Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _normalized_irregular_times() -> Tensor:
    ordinal = torch.arange(180, dtype=torch.float64)
    intervals = 1.0 + 0.22 * torch.sin(ordinal[1:] * 0.37) + 0.11 * torch.cos(ordinal[1:] * 0.19)
    times = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(intervals, dim=0)))
    return times / times[-1]


def _true_motion(times: Tensor) -> Tensor:
    phase = 2 * torch.pi * times
    residual = torch.column_stack(
        (
            0.018 * torch.sin(phase),
            0.012 * torch.cos(2 * phase + 0.2),
            0.009 * torch.sin(3 * phase - 0.4),
        )
    )
    root_rotation = torch.column_stack(
        (
            0.018 * torch.sin(phase + 0.3),
            0.012 * torch.cos(phase - 0.1),
            0.035 * torch.sin(2 * phase),
        )
    )
    root_translation = torch.column_stack(
        (
            0.012 * torch.sin(phase - 0.2),
            0.006 * torch.cos(2 * phase),
            0.014 * torch.sin(phase + 0.5),
        )
    )
    return torch.cat((residual, root_rotation, root_translation), dim=1)


def _deterministic_error(times: Tensor, *, phase_offset: float) -> Tensor:
    phase = 2 * torch.pi * times
    columns = []
    scales = (0.004, 0.003, 0.0025, 0.004, 0.003, 0.004, 0.003, 0.002, 0.003)
    for index, scale in enumerate(scales):
        columns.append(scale * torch.sin((7 + index % 4) * phase + phase_offset + 0.31 * index))
    return torch.column_stack(columns)


def _bounded_vectors(raw: Tensor, maximum: float) -> Tensor:
    norm = torch.sqrt(raw.square().sum(dim=-1, keepdim=True) + 1e-12)
    return raw * (maximum * torch.tanh(norm) / norm)


def _physical_parameters(raw: Tensor) -> Tensor:
    return torch.cat(
        (
            raw[:, :3],
            _bounded_vectors(raw[:, 3:6], float(np.deg2rad(5.0))),
            _bounded_vectors(raw[:, 6:9], 0.05),
        ),
        dim=1,
    )


def _articulated_ellipsoid() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float64)
    vertices = vertices * torch.tensor([0.55, 0.95, 0.42], dtype=torch.float64)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.long)
    upper = torch.sigmoid(8.0 * vertices[:, 1])
    weights = torch.column_stack((1.0 - upper, upper))
    bases = torch.stack(
        (
            torch.column_stack(
                (vertices[:, 0], torch.zeros(len(vertices)), torch.zeros(len(vertices)))
            ),
            torch.column_stack(
                (torch.zeros(len(vertices)), vertices[:, 1].square(), torch.zeros(len(vertices)))
            ),
            torch.column_stack((-vertices[:, 2], torch.zeros(len(vertices)), vertices[:, 0])),
        ),
        dim=0,
    )
    return vertices, faces, weights, bases


def _posed_vertices(times: Tensor, raw_parameters: Tensor) -> Tensor:
    vertices, _, weights, bases = _articulated_ellipsoid()
    parameters = _physical_parameters(raw_parameters)
    outputs = []
    for time, values in zip(times, parameters, strict=True):
        residual = torch.einsum("k,kvi->vi", values[:3], bases)
        joint_angle = 0.32 * torch.sin(2 * torch.pi * time)
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1] = rigid_transform_from_axis_angle(
            torch.stack(
                (torch.zeros_like(joint_angle), torch.zeros_like(joint_angle), joint_angle)
            ),
            torch.zeros(3, dtype=torch.float64),
        )
        root = rigid_transform_from_axis_angle(values[3:6], values[6:9])
        outputs.append(
            linear_blend_skinning(vertices + residual, weights, root.unsqueeze(0) @ base)
        )
    return torch.stack(outputs)


class _SlotTrajectory(nn.Module):
    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.values = nn.Parameter(values.clone())

    def training_values(self, _: Tensor) -> Tensor:
        return self.values

    def query(self, times: Tensor, train_times: Tensor) -> Tensor:
        return interpolate_slots(times, train_times, self.values)


def _train(
    model: nn.Module,
    train_times: Tensor,
    targets: Tensor,
    *,
    steps: int,
    optimizer_state: dict[str, Any] | None = None,
) -> tuple[torch.optim.Adam, bool]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    finite_gradient = False
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = (
            model.training_values(train_times)
            if isinstance(model, _SlotTrajectory)
            else model(train_times)
        )
        loss = (_physical_parameters(predicted) - _physical_parameters(targets)).square().mean()
        loss.backward()  # type: ignore[no-untyped-call]
        finite_gradient |= all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        optimizer.step()
    return optimizer, finite_gradient


def _resume_check(initial_controls: Tensor, train_times: Tensor, targets: Tensor) -> dict[str, Any]:
    uninterrupted = CubicControlTrajectory(initial_controls)
    _train(uninterrupted, train_times, targets, steps=STEP_COUNT)

    staged = CubicControlTrajectory(initial_controls)
    staged_optimizer, _ = _train(staged, train_times, targets, steps=STEP_COUNT // 2)
    buffer = io.BytesIO()
    torch.save({"model": staged.state_dict(), "optimizer": staged_optimizer.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=False)
    resumed = CubicControlTrajectory(initial_controls)
    resumed.load_state_dict(payload["model"])
    _train(
        resumed,
        train_times,
        targets,
        steps=STEP_COUNT // 2,
        optimizer_state=payload["optimizer"],
    )
    return {
        "exact": bool(torch.equal(uninterrupted.controls, resumed.controls)),
        "uninterrupted_hash": _tensor_hash(uninterrupted.controls),
        "resumed_hash": _tensor_hash(resumed.controls),
    }


def run(project_root: Path, output_root: Path, *, source_revision: str) -> dict[str, Any]:
    destination = output_root.resolve()
    if "sealed_test_v1" in {part.lower() for part in destination.parts}:
        raise ValueError("E1 synthetic output may not enter sealed-test storage")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable E1 preflight: {destination}")
    torch.manual_seed(SEED)
    times = _normalized_irregular_times()
    held_out_mask = torch.arange(len(times)) % 5 == 0
    train_times = times[~held_out_mask]
    held_out_times = times[held_out_mask]
    truth = _true_motion(times)
    train_truth = truth[~held_out_mask]
    held_out_truth = truth[held_out_mask]
    incumbent = train_truth + _deterministic_error(train_times, phase_offset=0.2)
    evidence_targets = train_truth + _deterministic_error(train_times, phase_offset=1.1)

    camera = torch.tensor([1300.0, 360.0, 560.0], dtype=torch.float64)
    shape = torch.linspace(-0.04, 0.04, 10, dtype=torch.float64)
    frozen_before = {"camera": _tensor_hash(camera), "shape": _tensor_hash(shape)}

    control = _SlotTrajectory(incumbent)
    _, control_gradient = _train(control, train_times, evidence_targets, steps=STEP_COUNT)
    treatment_initial = initialize_cubic_controls(
        train_times, incumbent, control_count=CONTROL_COUNT
    )
    treatment = CubicControlTrajectory(treatment_initial)
    _, treatment_gradient = _train(treatment, train_times, evidence_targets, steps=STEP_COUNT)
    control_held_out = control.query(held_out_times, train_times)
    treatment_held_out = treatment(held_out_times)
    truth_vertices = _posed_vertices(held_out_times, held_out_truth)
    control_vertices = _posed_vertices(held_out_times, control_held_out)
    treatment_vertices = _posed_vertices(held_out_times, treatment_held_out)
    control_rmse = float(torch.sqrt((control_vertices - truth_vertices).square().mean()).detach())
    treatment_rmse = float(
        torch.sqrt((treatment_vertices - truth_vertices).square().mean()).detach()
    )
    improvement = (control_rmse - treatment_rmse) / max(control_rmse, 1e-12)

    zero_times = train_times[:24]
    zero_slots = _SlotTrajectory(torch.zeros((len(zero_times), 9), dtype=torch.float64))
    _, zero_slot_gradient = _train(
        zero_slots, zero_times, torch.zeros_like(zero_slots.values), steps=1
    )
    zero_spline = CubicControlTrajectory(torch.zeros((CONTROL_COUNT, 9), dtype=torch.float64))
    _, zero_spline_gradient = _train(
        zero_spline,
        zero_times,
        torch.zeros((len(zero_times), 9), dtype=torch.float64),
        steps=1,
    )
    zero_unchanged = bool(
        torch.equal(zero_slots.values, torch.zeros_like(zero_slots.values))
        and torch.equal(zero_spline.controls, torch.zeros_like(zero_spline.controls))
    )
    resume = _resume_check(treatment_initial, train_times, evidence_targets)
    frozen_after = {"camera": _tensor_hash(camera), "shape": _tensor_hash(shape)}

    blockers = []
    if improvement <= 0:
        blockers.append("treatment_did_not_improve_known_motion")
    if not control_gradient or not treatment_gradient:
        blockers.append("missing_or_nonfinite_gradient")
    if not zero_slot_gradient or not zero_spline_gradient or not zero_unchanged:
        blockers.append("rigid_zero_residual_control")
    if frozen_before != frozen_after:
        blockers.append("camera_or_shape_drift")
    if not resume["exact"]:
        blockers.append("resume_not_exact")

    report: dict[str, Any] = {
        "schema_version": "post_v1_e1_synthetic_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "source_revision": source_revision,
        "seed": SEED,
        "fixture": "public_procedural_two_bone_clothed_ellipsoid",
        "time_sampling": {
            "irregular": True,
            "train_count": len(train_times),
            "omitted_count": len(held_out_times),
            "omission_rule": "every_fifth_source_time",
        },
        "changed_mechanism": "32_control_clamped_open_uniform_cubic_bspline",
        "control": "irregular_train_slots_with_piecewise_linear_held_out_interpolation",
        "optimizer": {
            "steps_per_arm": STEP_COUNT,
            "learning_rate": LEARNING_RATE,
            "same_budget": True,
        },
        "known_motion": {
            "control_held_out_vertex_rmse": control_rmse,
            "treatment_held_out_vertex_rmse": treatment_rmse,
            "relative_improvement": improvement,
        },
        "finite_gradients": {
            "control": control_gradient,
            "treatment": treatment_gradient,
            "rigid_zero_control": zero_slot_gradient and zero_spline_gradient,
        },
        "rigid_zero_residual_control_unchanged": zero_unchanged,
        "frozen_state": {
            "before": frozen_before,
            "after": frozen_after,
            "unchanged": frozen_before == frozen_after,
        },
        "resume": resume,
        "execution": {
            "human_evidence_accesses": 0,
            "development_evaluations": 0,
            "optimizer_steps_on_project_data": 0,
            "modal_jobs_launched": 0,
            "automatic_paid_retries": 0,
        },
        "sealed_test_isolation": {"private_evidence_paths_accessed": []},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e1-preflight-", dir=destination.parent))
    try:
        write_json(staging / "synthetic_preflight_report.json", report)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relevant = (
        "src/frayid/continuous_time.py",
        "scripts/run_post_v1_e1_preflight.py",
        "configs/evaluation/post_v1_e1_continuous_time_motion_r01.yaml",
    )
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *relevant],
        cwd=project_root,
        check=False,
    ).returncode:
        raise RuntimeError("Refusing E1 preflight from a dirty relevant worktree")
    report = run(
        project_root,
        project_root / OUTPUT_RELATIVE / "synthetic_preflight",
        source_revision=revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
