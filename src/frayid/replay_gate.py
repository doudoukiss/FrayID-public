from __future__ import annotations

import copy
import os
import platform
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from frayid.geometry import (
    canonical_topology_is_valid,
    clear_vertex_face_adjacency_cache,
    vertex_normals,
)
from frayid.replay_state import (
    SamplerState,
    capture_checkpoint_state,
    configure_deterministic_execution,
    nested_state_equal,
    restore_checkpoint_state,
)

SEED = 20260831
CHECKPOINT_STEPS = (1, 7, 12, 23)


class ReplayFixture(nn.Module):
    """Small public mesh workload exercising every replay-sensitive state class."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        dtype = torch.float64 if device.type == "cpu" else torch.float32
        vertices = torch.tensor(
            [
                [-0.8, -0.6, 2.0],
                [0.8, -0.6, 2.0],
                [0.8, 0.6, 2.0],
                [-0.8, 0.6, 2.0],
            ],
            dtype=dtype,
            device=device,
        )
        self.reference_vertices: Tensor
        self.faces: Tensor
        self.stage_index: Tensor
        self.barrier_stiffness: Tensor
        self.ccd_cache: Tensor
        self.register_buffer("reference_vertices", vertices)
        self.register_buffer(
            "faces",
            torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device),
        )
        self.register_buffer("stage_index", torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer("barrier_stiffness", torch.tensor(1.0, dtype=dtype, device=device))
        self.register_buffer("ccd_cache", torch.zeros((2, 3), dtype=dtype, device=device))
        self.offsets = nn.Parameter(torch.zeros_like(vertices))
        self.regressor: nn.Sequential = nn.Sequential(
            nn.Linear(3, 9), nn.Tanh(), nn.Linear(9, 1)
        ).to(device=device, dtype=dtype)

    @property
    def vertices(self) -> Tensor:
        return self.reference_vertices + self.offsets


@dataclass
class FixtureRuntime:
    model: ReplayFixture
    optimizer: torch.optim.Adam
    scheduler: torch.optim.lr_scheduler.StepLR
    generator: torch.Generator
    sampler: SamplerState
    global_step: int = 0
    stage: str = "coarse"


def _make_runtime(device: torch.device) -> FixtureRuntime:
    model = ReplayFixture(device=device)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.offsets], "lr": 0.04},
            {"params": list(model.regressor.parameters()), "lr": 0.003},
        ]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    generator = torch.Generator(device=device).manual_seed(SEED + 19)
    permutation_generator = torch.Generator(device="cpu").manual_seed(SEED + 23)
    sampler = SamplerState.shuffled(40, permutation_generator)
    return FixtureRuntime(model, optimizer, scheduler, generator, sampler)


def _project_topology(runtime: FixtureRuntime, previous: Tensor, *, force: bool) -> float:
    model = runtime.model
    candidate = model.offsets.detach().clone()
    if force:
        with torch.no_grad():
            candidate[2, 1] -= 2.0
    scale = 1.0
    for _ in range(17):
        with torch.no_grad():
            model.offsets.copy_(previous + scale * (candidate - previous))
        if canonical_topology_is_valid(
            model.reference_vertices,
            model.vertices,
            model.faces,
            minimum_signed_area_ratio=0.01,
            minimum_area_ratio=0.10,
        ):
            break
        scale *= 0.5
    else:
        scale = 0.0
        with torch.no_grad():
            model.offsets.copy_(previous)
    if scale < 1.0:
        state = runtime.optimizer.state[model.offsets]
        if "exp_avg" in state:
            state["exp_avg"].mul_(scale)
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(scale * scale)
    return scale


def _refresh_stage_and_mesh(runtime: FixtureRuntime) -> None:
    runtime.stage = "fine"
    with torch.no_grad():
        runtime.model.stage_index.fill_(1)
        runtime.model.barrier_stiffness.mul_(1.25)
        runtime.model.ccd_cache.copy_(runtime.model.vertices[:2] - runtime.model.vertices[2:])
    clear_vertex_face_adjacency_cache()
    vertex_normals(runtime.model.vertices, runtime.model.faces)


def _step(runtime: FixtureRuntime) -> dict[str, Any]:
    batch_id = runtime.sampler.take()
    model = runtime.model
    dtype = model.offsets.dtype
    device = model.offsets.device
    random_scale = 1.0 + 0.01 * random.random() + 0.01 * float(np.random.random())
    inputs = torch.randn((16, 3), dtype=dtype, device=device, generator=runtime.generator)
    target = 0.17 * inputs[:, :1] - 0.11 * inputs[:, 1:2] + 0.07 * inputs[:, 2:]
    runtime.optimizer.zero_grad(set_to_none=True)
    normals = vertex_normals(model.vertices, model.faces)
    loss = (
        (model.regressor(inputs * random_scale) - target).square().mean()
        + 0.02 * normals.square().mean()
        + 0.001 * model.offsets.square().mean()
        + batch_id * 1e-9
    )
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    previous = model.offsets.detach().clone()
    runtime.optimizer.step()
    projection_scale = _project_topology(runtime, previous, force=runtime.global_step == 6)
    runtime.scheduler.step()
    runtime.global_step += 1
    if runtime.global_step == 12:
        _refresh_stage_and_mesh(runtime)
    return {
        "batch": batch_id,
        "loss": loss.detach().clone(),
        "gradients": gradients,
        "parameters_and_buffers": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(runtime.optimizer.state_dict()),
        "scheduler": copy.deepcopy(runtime.scheduler.state_dict()),
        "projection_scale": projection_scale,
        "stage": runtime.stage,
        "sampler": runtime.sampler.state_dict(),
    }


def _seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _checkpoint_check(device: torch.device, checkpoint_step: int) -> dict[str, Any]:
    _seed_all()
    runtime = _make_runtime(device)
    projection_scales: list[float] = []
    for _ in range(checkpoint_step):
        projection_scales.append(_step(runtime)["projection_scale"])
    state = capture_checkpoint_state(
        runtime.model,
        runtime.optimizer,
        epoch=runtime.global_step // 40,
        global_step=runtime.global_step,
        stage=runtime.stage,
        sampler_state=runtime.sampler,
        named_generators={"batch": runtime.generator},
        scheduler=runtime.scheduler,
        auxiliary_state={
            "barrier_stiffness": runtime.model.barrier_stiffness.detach().clone(),
            "ccd_cache": runtime.model.ccd_cache.detach().clone(),
            "projection_scales": projection_scales,
        },
        immutable_bindings={"fixture": "public-replay-state-v2"},
    )
    expected = _step(runtime)

    clone = _make_runtime(device)

    def rebuild() -> None:
        clear_vertex_face_adjacency_cache()
        vertex_normals(clone.model.vertices, clone.model.faces)

    clone.sampler = restore_checkpoint_state(
        state,
        clone.model,
        clone.optimizer,
        named_generators={"batch": clone.generator},
        scheduler=clone.scheduler,
        rebuild_caches=rebuild,
        expected_immutable_bindings={"fixture": "public-replay-state-v2"},
    )
    clone.global_step = state.global_step
    clone.stage = state.stage
    actual = _step(clone)
    exact = nested_state_equal(expected, actual)
    return {
        "checkpoint_step": checkpoint_step,
        "status": "pass" if exact else "fail",
        "bitwise_next_step": exact,
        "batch_equal": expected["batch"] == actual["batch"],
        "loss_equal": torch.equal(expected["loss"], actual["loss"]),
        "gradients_equal": nested_state_equal(expected["gradients"], actual["gradients"]),
        "parameters_equal": nested_state_equal(
            expected["parameters_and_buffers"], actual["parameters_and_buffers"]
        ),
        "adam_moments_equal": nested_state_equal(expected["optimizer"], actual["optimizer"]),
        "projection_decision_equal": expected["projection_scale"] == actual["projection_scale"],
        "stage_equal": expected["stage"] == actual["stage"],
        "forced_projection_seen": any(scale < 1.0 for scale in projection_scales),
        "stage_refresh_seen": checkpoint_step >= 12,
    }


def _negative_check(device: torch.device) -> dict[str, Any]:
    _seed_all()
    runtime = _make_runtime(device)
    _step(runtime)
    model_state = copy.deepcopy(runtime.model.state_dict())
    optimizer_state = copy.deepcopy(runtime.optimizer.state_dict())
    expected = _step(runtime)

    clone = _make_runtime(device)
    clone.model.load_state_dict(model_state)
    clone.optimizer.load_state_dict(optimizer_state)
    clone.sampler = SamplerState(
        list(reversed(runtime.sampler.permutation)), runtime.sampler.cursor
    )
    clone.generator.manual_seed(SEED + 999)
    actual = _step(clone)
    detected = not nested_state_equal(expected, actual)
    return {"status": "pass" if detected else "fail", "mismatch_detected": detected}


def run_replay_gate(device_name: str) -> dict[str, Any]:
    configure_deterministic_execution()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay gate requested but CUDA is unavailable")
    device = torch.device(device_name)
    checks = [_checkpoint_check(device, step) for step in CHECKPOINT_STEPS]
    negative = _negative_check(device)
    blockers: list[str] = []
    for check in checks:
        if check["status"] != "pass":
            blockers.append(f"next_step_mismatch_at_{check['checkpoint_step']}")
    if not checks[1]["forced_projection_seen"]:
        blockers.append("topology_projection_not_exercised")
    if not checks[2]["stage_refresh_seen"]:
        blockers.append("stage_refresh_not_exercised")
    if negative["status"] != "pass":
        blockers.append("negative_replay_fixture_not_rejected")
    return {
        "schema_version": "post_v1_replay_state_preflight.v1",
        "status": "pass" if not blockers else "fail",
        "seed": SEED,
        "device": device_name,
        "device_name": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else platform.processor(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "checks": checks,
        "negative_control": negative,
        "blockers": blockers,
    }
