from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

FLEXICUBES_REVISION = "4cc7d6c3d0cee83c011ce36721b81adff0dd7db6"


@dataclass(frozen=True)
class FlexiCubesMesh:
    vertices: Tensor
    faces: Tensor
    developability: Tensor


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def validate_flexicubes_repository(
    repository: Path, *, expected_revision: str = FLEXICUBES_REVISION
) -> None:
    repository = repository.resolve()
    if not (repository / "flexicubes.py").is_file() or not (repository / "tables.py").is_file():
        raise FileNotFoundError("official FlexiCubes source files are absent")
    if _git_head(repository) != expected_revision:
        raise RuntimeError("FlexiCubes checkout does not match the registered revision")
    status_lines = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    disallowed = [
        line
        for line in status_lines
        if not (line.startswith("?? __pycache__/") and line.endswith((".pyc", ".pyo")))
    ]
    if disallowed:
        raise RuntimeError("FlexiCubes checkout must remain clean")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_flexicubes_class(repository: Path) -> type[Any]:
    """Load the pinned upstream class without vendoring or mutating its source."""
    validate_flexicubes_repository(repository)
    repository = repository.resolve()
    previous_tables = sys.modules.get("tables")
    try:
        sys.modules["tables"] = _load_module("tables", repository / "tables.py")
        module = _load_module("frayid_pinned_flexicubes", repository / "flexicubes.py")
    finally:
        if previous_tables is None:
            sys.modules.pop("tables", None)
        else:
            sys.modules["tables"] = previous_tables
    value = getattr(module, "FlexiCubes", None)
    if not isinstance(value, type):
        raise ImportError("official module does not expose FlexiCubes")
    return value


class PinnedFlexiCubes:
    """Thin verified adapter around the official non-vendored implementation."""

    def __init__(self, repository: Path, *, device: torch.device | str) -> None:
        flexicubes = load_flexicubes_class(repository)
        self._implementation = flexicubes(device=str(device))
        self.device = torch.device(device)

    def voxel_grid(self, resolution: int, *, extent: float) -> tuple[Tensor, Tensor]:
        if resolution < 2:
            raise ValueError("FlexiCubes resolution must be at least two")
        if extent <= 0:
            raise ValueError("FlexiCubes extent must be positive")
        vertices, cubes = self._implementation.construct_voxel_grid(resolution)
        return vertices * (2.0 * extent), cubes

    def extract(
        self,
        vertices: Tensor,
        values: Tensor,
        cubes: Tensor,
        resolution: int,
        *,
        beta: Tensor | None = None,
        alpha: Tensor | None = None,
        gamma: Tensor | None = None,
        training: bool,
        gradient_function: Callable[[Tensor], Tensor] | None = None,
    ) -> FlexiCubesMesh:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("FlexiCubes vertices must have shape [N, 3]")
        if values.shape != (vertices.shape[0],):
            raise ValueError("FlexiCubes values must have one scalar per vertex")
        if cubes.ndim != 2 or cubes.shape[1] != 8:
            raise ValueError("FlexiCubes cubes must have shape [F, 8]")
        if not torch.isfinite(vertices).all() or not torch.isfinite(values).all():
            raise ValueError("FlexiCubes input must be finite")
        result = self._implementation(
            vertices,
            values,
            cubes,
            resolution,
            beta_fx12=beta,
            alpha_fx8=alpha,
            gamma_f=gamma,
            training=training,
            grad_func=gradient_function,
        )
        mesh_vertices, faces, developability = result
        if mesh_vertices.numel() == 0 or faces.numel() == 0:
            raise ValueError("FlexiCubes field contains no extractable surface")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise RuntimeError("FlexiCubes returned non-triangular surface output")
        if not torch.isfinite(mesh_vertices).all() or not torch.isfinite(developability).all():
            raise RuntimeError("FlexiCubes returned non-finite output")
        return FlexiCubesMesh(mesh_vertices, faces, developability)
