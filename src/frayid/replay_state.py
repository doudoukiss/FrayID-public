from __future__ import annotations

import copy
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

CHECKPOINT_SCHEMA_V2 = "canonical_checkpoint.v2"
CHECKPOINT_SCHEMA_V3 = "canonical_checkpoint.v3"


@dataclass
class SamplerState:
    """Serializable permutation and cursor for an interruptible training epoch."""

    permutation: list[int]
    cursor: int = 0

    def validate(self, *, item_count: int | None = None) -> None:
        if self.cursor < 0 or self.cursor > len(self.permutation):
            raise ValueError("sampler cursor is outside the permutation")
        if len(set(self.permutation)) != len(self.permutation):
            raise ValueError("sampler permutation contains duplicate entries")
        if item_count is not None and sorted(self.permutation) != list(range(item_count)):
            raise ValueError("sampler permutation does not cover the expected items")

    @property
    def complete(self) -> bool:
        return self.cursor == len(self.permutation)

    def take(self) -> int:
        self.validate()
        if self.complete:
            raise StopIteration
        value = self.permutation[self.cursor]
        self.cursor += 1
        return value

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return {"permutation": list(self.permutation), "cursor": self.cursor}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> SamplerState:
        result = cls(
            permutation=[int(value) for value in state["permutation"]],
            cursor=int(state["cursor"]),
        )
        result.validate()
        return result

    @classmethod
    def shuffled(cls, item_count: int, generator: torch.Generator) -> SamplerState:
        if item_count <= 0:
            raise ValueError("item_count must be positive")
        return cls(torch.randperm(item_count, generator=generator, device="cpu").tolist())


@dataclass
class RNGState:
    """All process and named-generator random state that affects an update."""

    python_state: tuple[Any, ...]
    numpy_state: Any
    torch_cpu_state: Tensor
    torch_cuda_states: list[Tensor]
    named_generator_states: dict[str, Tensor] = field(default_factory=dict)

    @classmethod
    def capture(cls, named_generators: Mapping[str, torch.Generator] | None = None) -> RNGState:
        return cls(
            python_state=copy.deepcopy(random.getstate()),
            numpy_state=copy.deepcopy(np.random.get_state()),
            torch_cpu_state=torch.random.get_rng_state().clone(),
            torch_cuda_states=(
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            ),
            named_generator_states={
                name: generator.get_state().clone()
                for name, generator in sorted((named_generators or {}).items())
            },
        )

    def restore(self, named_generators: Mapping[str, torch.Generator] | None = None) -> None:
        generators = named_generators or {}
        if set(generators) != set(self.named_generator_states):
            raise ValueError(
                "named generator mismatch: "
                f"checkpoint={sorted(self.named_generator_states)}, runtime={sorted(generators)}"
            )
        if self.torch_cuda_states:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
            if len(self.torch_cuda_states) != torch.cuda.device_count():
                raise RuntimeError("checkpoint CUDA RNG device count does not match runtime")
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_cpu_state)
        if self.torch_cuda_states:
            torch.cuda.set_rng_state_all(self.torch_cuda_states)
        for name, generator in generators.items():
            generator.set_state(self.named_generator_states[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "python": copy.deepcopy(self.python_state),
            "numpy": copy.deepcopy(self.numpy_state),
            "torch_cpu": self.torch_cpu_state.clone(),
            "torch_cuda": [state.clone() for state in self.torch_cuda_states],
            "named_generators": {
                name: state.clone() for name, state in self.named_generator_states.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> RNGState:
        return cls(
            python_state=cast(tuple[Any, ...], copy.deepcopy(state["python"])),
            numpy_state=copy.deepcopy(tuple(state["numpy"])),
            torch_cpu_state=state["torch_cpu"].clone(),
            torch_cuda_states=[value.clone() for value in state["torch_cuda"]],
            named_generator_states={
                str(name): value.clone() for name, value in state["named_generators"].items()
            },
        )


@dataclass
class CheckpointStateV2:
    """Complete state transition snapshot for exact same-device next-step replay."""

    epoch: int
    global_step: int
    stage: str
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    optimizer_parameter_names: list[list[str]]
    sampler_state: SamplerState
    rng_state: RNGState
    scheduler_state: dict[str, Any] | None = None
    scaler_state: dict[str, Any] | None = None
    auxiliary_state: dict[str, Any] = field(default_factory=dict)
    immutable_bindings: dict[str, str] = field(default_factory=dict)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_V2,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stage": self.stage,
            "model": copy.deepcopy(self.model_state),
            "optimizer": copy.deepcopy(self.optimizer_state),
            "optimizer_parameter_names": copy.deepcopy(self.optimizer_parameter_names),
            "sampler": self.sampler_state.state_dict(),
            "rng": self.rng_state.state_dict(),
            "scheduler": copy.deepcopy(self.scheduler_state),
            "scaler": copy.deepcopy(self.scaler_state),
            "auxiliary": copy.deepcopy(self.auxiliary_state),
            "immutable_bindings": dict(self.immutable_bindings),
            "next_step_replay_capable": True,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> CheckpointStateV2:
        if state.get("schema_version") != CHECKPOINT_SCHEMA_V2:
            raise ValueError("checkpoint is not canonical_checkpoint.v2")
        if state.get("next_step_replay_capable") is not True:
            raise ValueError("checkpoint does not claim complete next-step replay state")
        return cls(
            epoch=int(state["epoch"]),
            global_step=int(state["global_step"]),
            stage=str(state["stage"]),
            model_state=copy.deepcopy(state["model"]),
            optimizer_state=copy.deepcopy(state["optimizer"]),
            optimizer_parameter_names=[
                [str(name) for name in group] for group in state["optimizer_parameter_names"]
            ],
            sampler_state=SamplerState.from_state_dict(state["sampler"]),
            rng_state=RNGState.from_state_dict(state["rng"]),
            scheduler_state=copy.deepcopy(state.get("scheduler")),
            scaler_state=copy.deepcopy(state.get("scaler")),
            auxiliary_state=copy.deepcopy(state.get("auxiliary", {})),
            immutable_bindings={
                str(key): str(value) for key, value in state.get("immutable_bindings", {}).items()
            },
        )


@dataclass
class CheckpointStateV3(CheckpointStateV2):
    """V2 replay state plus the complete E16 ambient-path transition state."""

    ambient_state: dict[str, Any] = field(default_factory=dict)

    def validate_ambient_state(self) -> None:
        required = {
            "scaffold_sha256",
            "scaffold_ordering_sha256",
            "solver_state",
            "proposed_direction_sha256",
            "accepted_alpha_hex",
            "certificate_sha256",
            "immutable_report_paths",
        }
        missing = required - self.ambient_state.keys()
        if missing:
            raise ValueError(f"checkpoint v3 ambient state is incomplete: {sorted(missing)}")
        if not isinstance(self.ambient_state["solver_state"], Mapping):
            raise ValueError("checkpoint v3 solver state must be a mapping")
        report_paths = self.ambient_state["immutable_report_paths"]
        if not isinstance(report_paths, list) or not all(
            isinstance(path, str) and path for path in report_paths
        ):
            raise ValueError("checkpoint v3 immutable report paths must be non-empty strings")
        try:
            float.fromhex(str(self.ambient_state["accepted_alpha_hex"]))
        except ValueError as error:
            raise ValueError("checkpoint v3 accepted alpha is not a float hex value") from error

    def state_dict(self) -> dict[str, Any]:
        self.validate_ambient_state()
        result = super().state_dict()
        result["schema_version"] = CHECKPOINT_SCHEMA_V3
        result["ambient"] = copy.deepcopy(self.ambient_state)
        return result

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> CheckpointStateV3:
        if state.get("schema_version") != CHECKPOINT_SCHEMA_V3:
            raise ValueError("checkpoint is not canonical_checkpoint.v3")
        base_payload = copy.deepcopy(dict(state))
        base_payload["schema_version"] = CHECKPOINT_SCHEMA_V2
        base_payload.pop("ambient", None)
        base = CheckpointStateV2.from_state_dict(base_payload)
        result = cls(
            epoch=base.epoch,
            global_step=base.global_step,
            stage=base.stage,
            model_state=base.model_state,
            optimizer_state=base.optimizer_state,
            optimizer_parameter_names=base.optimizer_parameter_names,
            sampler_state=base.sampler_state,
            rng_state=base.rng_state,
            scheduler_state=base.scheduler_state,
            scaler_state=base.scaler_state,
            auxiliary_state=base.auxiliary_state,
            immutable_bindings=base.immutable_bindings,
            ambient_state=copy.deepcopy(dict(state.get("ambient", {}))),
        )
        result.validate_ambient_state()
        return result


def checkpoint_state_from_dict(state: Mapping[str, Any]) -> CheckpointStateV2 | CheckpointStateV3:
    """Load either replay schema while keeping historical v2 payloads unchanged."""

    schema = state.get("schema_version")
    if schema == CHECKPOINT_SCHEMA_V2:
        return CheckpointStateV2.from_state_dict(state)
    if schema == CHECKPOINT_SCHEMA_V3:
        return CheckpointStateV3.from_state_dict(state)
    raise ValueError(f"unsupported checkpoint schema: {schema}")


def configure_deterministic_execution() -> None:
    """Enable strict same-device determinism before constructing CUDA workloads."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _optimizer_parameter_names(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> list[list[str]]:
    by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    result: list[list[str]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = by_identity.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter not owned by the model")
            names.append(name)
        result.append(names)
    return result


def capture_checkpoint_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    stage: str,
    sampler_state: SamplerState,
    named_generators: Mapping[str, torch.Generator] | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    auxiliary_state: Mapping[str, Any] | None = None,
    immutable_bindings: Mapping[str, str] | None = None,
) -> CheckpointStateV2:
    sampler_state.validate()
    return CheckpointStateV2(
        epoch=epoch,
        global_step=global_step,
        stage=stage,
        model_state=copy.deepcopy(model.state_dict()),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        optimizer_parameter_names=_optimizer_parameter_names(model, optimizer),
        sampler_state=SamplerState.from_state_dict(sampler_state.state_dict()),
        rng_state=RNGState.capture(named_generators),
        scheduler_state=copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        scaler_state=copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        auxiliary_state=copy.deepcopy(dict(auxiliary_state or {})),
        immutable_bindings=dict(immutable_bindings or {}),
    )


def capture_checkpoint_state_v3(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    stage: str,
    sampler_state: SamplerState,
    ambient_state: Mapping[str, Any],
    named_generators: Mapping[str, torch.Generator] | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    auxiliary_state: Mapping[str, Any] | None = None,
    immutable_bindings: Mapping[str, str] | None = None,
) -> CheckpointStateV3:
    base = capture_checkpoint_state(
        model,
        optimizer,
        epoch=epoch,
        global_step=global_step,
        stage=stage,
        sampler_state=sampler_state,
        named_generators=named_generators,
        scheduler=scheduler,
        scaler=scaler,
        auxiliary_state=auxiliary_state,
        immutable_bindings=immutable_bindings,
    )
    result = CheckpointStateV3(
        epoch=base.epoch,
        global_step=base.global_step,
        stage=base.stage,
        model_state=base.model_state,
        optimizer_state=base.optimizer_state,
        optimizer_parameter_names=base.optimizer_parameter_names,
        sampler_state=base.sampler_state,
        rng_state=base.rng_state,
        scheduler_state=base.scheduler_state,
        scaler_state=base.scaler_state,
        auxiliary_state=base.auxiliary_state,
        immutable_bindings=base.immutable_bindings,
        ambient_state=copy.deepcopy(dict(ambient_state)),
    )
    result.validate_ambient_state()
    return result


def restore_checkpoint_state(
    state: CheckpointStateV2 | CheckpointStateV3,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    named_generators: Mapping[str, torch.Generator] | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    rebuild_caches: Callable[[], None] | None = None,
    expected_immutable_bindings: Mapping[str, str] | None = None,
) -> SamplerState:
    if dict(expected_immutable_bindings or {}) != state.immutable_bindings:
        raise ValueError("checkpoint immutable bindings do not match runtime")
    runtime_names = _optimizer_parameter_names(model, optimizer)
    if runtime_names != state.optimizer_parameter_names:
        raise ValueError("optimizer parameter-group ordering does not match checkpoint")
    model.load_state_dict(state.model_state)
    optimizer.load_state_dict(state.optimizer_state)
    if (scheduler is None) != (state.scheduler_state is None):
        raise ValueError("scheduler presence does not match checkpoint")
    if scheduler is not None:
        scheduler.load_state_dict(state.scheduler_state)
    if (scaler is None) != (state.scaler_state is None):
        raise ValueError("AMP scaler presence does not match checkpoint")
    if scaler is not None:
        scaler.load_state_dict(state.scaler_state)
    if rebuild_caches is not None:
        rebuild_caches()
    # Object construction and all deterministic cache rebuilding must happen before RNG restore.
    state.rng_state.restore(named_generators)
    return SamplerState.from_state_dict(state.sampler_state.state_dict())


def nested_state_equal(first: Any, second: Any) -> bool:
    if isinstance(first, Tensor) and isinstance(second, Tensor):
        return torch.equal(first, second)
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(
            nested_state_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            nested_state_equal(left, right) for left, right in zip(first, second, strict=True)
        )
    return bool(first == second)
