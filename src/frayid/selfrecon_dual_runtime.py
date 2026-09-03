from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InterpreterIdentity:
    executable: str
    prefix: str
    version: tuple[int, int, int]

    @classmethod
    def from_json(cls, payload: str) -> InterpreterIdentity:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("interpreter probe must return a JSON object")
        executable = value.get("executable")
        prefix = value.get("prefix")
        version = value.get("version")
        if (
            not isinstance(executable, str)
            or not isinstance(prefix, str)
            or not isinstance(version, list)
            or len(version) != 3
            or any(not isinstance(part, int) for part in version)
        ):
            raise ValueError("interpreter probe returned an invalid identity")
        return cls(executable=executable, prefix=prefix, version=tuple(version))


@dataclass(frozen=True)
class DualRuntimeBoundary:
    control: InterpreterIdentity
    child: InterpreterIdentity
    legacy_prefix: str
    global_path_entries: tuple[str, ...]

    def as_report(self) -> dict[str, Any]:
        return asdict(self)


INTERPRETER_PROBE = (
    "import json,sys;"
    "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
    "'version':list(sys.version_info[:3])},sort_keys=True))"
)


def _resolved(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError("runtime interpreter paths must be absolute")
    return value.resolve(strict=False)


def path_contains_prefix(path_value: str, prefix: str | Path) -> bool:
    resolved_prefix = _resolved(prefix)
    for entry in path_value.split(os.pathsep):
        if not entry:
            continue
        try:
            _resolved(entry).relative_to(resolved_prefix)
        except ValueError:
            continue
        return True
    return False


def build_isolated_child_environment(
    environment: dict[str, str], *, legacy_prefix: str | Path
) -> dict[str, str]:
    """Copy the control environment while proving legacy paths stay unexported."""
    result = dict(environment)
    path_value = result.get("PATH", "")
    if path_contains_prefix(path_value, legacy_prefix):
        raise ValueError("legacy runtime prefix is present in global PATH")
    for name in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME"):
        result.pop(name, None)
    result["PYTHONNOUSERSITE"] = "1"
    return result


def probe_interpreter(
    executable: str | Path,
    *,
    environment: dict[str, str],
    timeout_seconds: int = 30,
) -> InterpreterIdentity:
    executable_path = _resolved(executable)
    completed = subprocess.run(
        [str(executable_path), "-I", "-c", INTERPRETER_PROBE],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=environment,
    )
    return InterpreterIdentity.from_json(completed.stdout)


def validate_dual_runtime_boundary(
    *,
    control: InterpreterIdentity,
    child: InterpreterIdentity,
    legacy_prefix: str | Path,
    path_value: str,
) -> DualRuntimeBoundary:
    resolved_prefix = _resolved(legacy_prefix)
    control_executable = _resolved(control.executable)
    child_executable = _resolved(child.executable)
    child_prefix = _resolved(child.prefix)
    if control.version < (3, 10, 0):
        raise ValueError("control-plane Python must be at least 3.10")
    if child.version[:2] != (3, 8):
        raise ValueError("SelfRecon child must use Python 3.8")
    try:
        control_executable.relative_to(resolved_prefix)
    except ValueError:
        pass
    else:
        raise ValueError("control-plane interpreter is inside the legacy prefix")
    try:
        child_executable.relative_to(resolved_prefix)
        child_prefix.relative_to(resolved_prefix)
    except ValueError as exc:
        raise ValueError("SelfRecon child is outside the registered legacy prefix") from exc
    if path_contains_prefix(path_value, resolved_prefix):
        raise ValueError("legacy runtime prefix is present in global PATH")
    return DualRuntimeBoundary(
        control=control,
        child=child,
        legacy_prefix=str(resolved_prefix),
        global_path_entries=tuple(entry for entry in path_value.split(os.pathsep) if entry),
    )


def run_dual_runtime_preflight(
    *,
    control_executable: str | Path,
    child_executable: str | Path,
    legacy_prefix: str | Path,
    environment: dict[str, str],
) -> DualRuntimeBoundary:
    isolated = build_isolated_child_environment(environment, legacy_prefix=legacy_prefix)
    control = probe_interpreter(control_executable, environment=isolated)
    child = probe_interpreter(child_executable, environment=isolated)
    return validate_dual_runtime_boundary(
        control=control,
        child=child,
        legacy_prefix=legacy_prefix,
        path_value=isolated.get("PATH", ""),
    )
