from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
import torch
from torch import nn

V2_CHECKPOINT_SCHEMA = "frayid_v2_checkpoint.v1"


def _cpu_cuda_rng_states(value: object) -> list[torch.Tensor]:
    """Return the CPU byte tensors required by torch.cuda.set_rng_state_all."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("V2 checkpoint CUDA RNG state is invalid")
    states: list[torch.Tensor] = []
    for state in value:
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
            raise ValueError("V2 checkpoint CUDA RNG state is invalid")
        states.append(state.detach().cpu().contiguous())
    return states


def capture_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    topology_connectivity_sha256: str | None,
) -> bytes:
    if step < 0:
        raise ValueError("V2 checkpoint step cannot be negative")
    payload = {
        "schema_version": V2_CHECKPOINT_SCHEMA,
        "step": step,
        "topology_connectivity_sha256": topology_connectivity_sha256,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def restore_checkpoint(
    data: bytes,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    try:
        payload = torch.load(
            io.BytesIO(data), map_location=torch.device(device), weights_only=False
        )
    except Exception as error:
        raise ValueError("V2 checkpoint schema is invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != V2_CHECKPOINT_SCHEMA:
        raise ValueError("V2 checkpoint schema is invalid")
    if not isinstance(payload.get("step"), int):
        raise ValueError("V2 checkpoint cursor is invalid")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_states = payload.get("cuda_rng_state_all")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("V2 checkpoint requires unavailable CUDA RNG state")
        # torch.load(map_location="cuda") also maps serialized RNG tensors to
        # CUDA, while PyTorch's RNG restore API explicitly requires CPU byte
        # tensors. Normalize the transport representation before restoration.
        torch.cuda.set_rng_state_all(_cpu_cuda_rng_states(cuda_states))
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return payload
