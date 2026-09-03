from __future__ import annotations

import hashlib
import io
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from frayid.eulerian_field import conventional_surface_audit

E25_CHECKPOINT_SCHEMA = "post_v1_e25_checkpoint.v1"


def tensor_digest(*values: Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        canonical = value.detach().cpu().contiguous()
        digest.update(str(canonical.dtype).encode())
        digest.update(str(tuple(canonical.shape)).encode())
        digest.update(canonical.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class StageCommitment:
    resolution: int
    vertex_count: int
    face_count: int
    surface_digest: str
    connectivity_digest: str
    exact_intersection_pair_count: int
    probes_preserved: bool
    replay_exact: bool
    conventional_topology: dict[str, object]

    @property
    def status(self) -> str:
        return "pass"

    def as_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "resolution": self.resolution,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "surface_digest": self.surface_digest,
            "connectivity_digest": self.connectivity_digest,
            "exact_intersection_pair_count": self.exact_intersection_pair_count,
            "probes_preserved": self.probes_preserved,
            "replay_exact": self.replay_exact,
            "conventional_topology": self.conventional_topology,
        }


def commit_stage_surface(
    vertices: Tensor,
    faces: Tensor,
    *,
    resolution: int,
    exact_intersection_pair_count: int,
    probes_preserved: bool,
    replay_exact: bool,
) -> StageCommitment:
    if resolution not in (24, 48, 96):
        raise ValueError("E25 stage resolution is not registered")
    if exact_intersection_pair_count < 0:
        raise ValueError("exact intersection count cannot be negative")
    topology = conventional_surface_audit(vertices, faces)
    blockers: list[str] = []
    if topology.get("status") != "pass":
        blockers.append("conventional_topology")
    if exact_intersection_pair_count != 0:
        blockers.append("exact_self_intersection")
    if not probes_preserved:
        blockers.append("probe_or_gap_classification")
    if not replay_exact:
        blockers.append("next_step_replay")
    if blockers:
        raise ValueError("E25 stage commitment failed: " + ",".join(blockers))
    return StageCommitment(
        resolution=resolution,
        vertex_count=int(vertices.shape[0]),
        face_count=int(faces.shape[0]),
        surface_digest=tensor_digest(vertices, faces),
        connectivity_digest=tensor_digest(faces),
        exact_intersection_pair_count=exact_intersection_pair_count,
        probes_preserved=probes_preserved,
        replay_exact=replay_exact,
        conventional_topology=topology,
    )


def assert_frozen_connectivity(commitment: StageCommitment, faces: Tensor) -> None:
    if tensor_digest(faces) != commitment.connectivity_digest:
        raise ValueError("committed E25 connectivity changed during certified refinement")


def capture_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    resolution: int,
    step: int,
    committed_connectivity_digest: str | None,
) -> bytes:
    if resolution not in (24, 48, 96) or step < 0:
        raise ValueError("E25 checkpoint cursor is invalid")
    payload = {
        "schema_version": E25_CHECKPOINT_SCHEMA,
        "resolution": resolution,
        "step": step,
        "committed_connectivity_digest": committed_connectivity_digest,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def restore_checkpoint(
    data: bytes,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("E25 checkpoint schema is invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != E25_CHECKPOINT_SCHEMA:
        raise ValueError("E25 checkpoint schema is invalid")
    if payload.get("resolution") not in (24, 48, 96) or not isinstance(payload.get("step"), int):
        raise ValueError("E25 checkpoint cursor is invalid")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_states = payload.get("cuda_rng_state_all")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("E25 checkpoint requires unavailable CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return payload
