from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from frayid.selfrecon_dual_runtime import (
    InterpreterIdentity,
    build_isolated_child_environment,
    path_contains_prefix,
    probe_interpreter,
    validate_dual_runtime_boundary,
)


def test_interpreter_identity_parser_is_strict() -> None:
    parsed = InterpreterIdentity.from_json(
        '{"executable":"/usr/bin/python3","prefix":"/usr","version":[3,11,9]}'
    )
    assert parsed.version == (3, 11, 9)
    with pytest.raises(ValueError, match="invalid identity"):
        InterpreterIdentity.from_json('{"executable":"python","version":[3,11]}')


def test_global_path_prefix_detection_is_component_aware() -> None:
    assert path_contains_prefix("/usr/bin:/opt/selfrecon/bin", "/opt/selfrecon")
    assert not path_contains_prefix("/usr/bin:/opt/selfrecon-other/bin", "/opt/selfrecon")


def test_isolated_child_environment_removes_activation_state_not_control_path() -> None:
    source = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "CONDA_PREFIX": "/another/environment",
        "CONDA_DEFAULT_ENV": "another",
        "PYTHONHOME": "/another/python",
        "CUDA_HOME": "/usr/local/cuda",
    }
    result = build_isolated_child_environment(source, legacy_prefix="/opt/selfrecon")
    assert result["PATH"] == source["PATH"]
    assert result["CUDA_HOME"] == source["CUDA_HOME"]
    assert result["PYTHONNOUSERSITE"] == "1"
    assert "CONDA_PREFIX" not in result
    assert "CONDA_DEFAULT_ENV" not in result
    assert "PYTHONHOME" not in result
    with pytest.raises(ValueError, match="global PATH"):
        build_isolated_child_environment(
            {"PATH": "/usr/bin:/opt/selfrecon/bin"}, legacy_prefix="/opt/selfrecon"
        )


def test_current_control_interpreter_can_be_probed_by_absolute_path() -> None:
    identity = probe_interpreter(
        Path(sys.executable), environment=dict(os.environ), timeout_seconds=10
    )
    assert Path(identity.executable).resolve() == Path(sys.executable).resolve()
    assert identity.version[:2] == sys.version_info[:2]


def test_registered_boundary_accepts_only_new_control_and_legacy_child() -> None:
    control = InterpreterIdentity("/usr/local/bin/python3.11", "/usr/local", (3, 11, 15))
    child = InterpreterIdentity("/opt/selfrecon/bin/python3.8", "/opt/selfrecon", (3, 8, 12))
    boundary = validate_dual_runtime_boundary(
        control=control,
        child=child,
        legacy_prefix="/opt/selfrecon",
        path_value="/usr/local/bin:/usr/bin:/bin",
    )
    assert boundary.control.version == (3, 11, 15)
    assert boundary.child.version == (3, 8, 12)
    assert boundary.legacy_prefix == "/opt/selfrecon"


@pytest.mark.parametrize(
    ("control", "child", "path_value", "message"),
    [
        (
            InterpreterIdentity("/usr/bin/python3.9", "/usr", (3, 9, 19)),
            InterpreterIdentity("/opt/selfrecon/bin/python", "/opt/selfrecon", (3, 8, 12)),
            "/usr/bin",
            "at least 3.10",
        ),
        (
            InterpreterIdentity("/usr/bin/python3.11", "/usr", (3, 11, 9)),
            InterpreterIdentity("/opt/selfrecon/bin/python", "/opt/selfrecon", (3, 9, 0)),
            "/usr/bin",
            "Python 3.8",
        ),
        (
            InterpreterIdentity("/opt/selfrecon/bin/python3.11", "/opt/selfrecon", (3, 11, 9)),
            InterpreterIdentity("/opt/selfrecon/bin/python", "/opt/selfrecon", (3, 8, 12)),
            "/usr/bin",
            "control-plane",
        ),
        (
            InterpreterIdentity("/usr/bin/python3.11", "/usr", (3, 11, 9)),
            InterpreterIdentity("/other/bin/python", "/other", (3, 8, 12)),
            "/usr/bin",
            "outside",
        ),
        (
            InterpreterIdentity("/usr/bin/python3.11", "/usr", (3, 11, 9)),
            InterpreterIdentity("/opt/selfrecon/bin/python", "/opt/selfrecon", (3, 8, 12)),
            "/usr/bin:/opt/selfrecon/bin",
            "global PATH",
        ),
    ],
)
def test_registered_boundary_rejects_shadowing_and_wrong_versions(
    control: InterpreterIdentity,
    child: InterpreterIdentity,
    path_value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_dual_runtime_boundary(
            control=control,
            child=child,
            legacy_prefix="/opt/selfrecon",
            path_value=path_value,
        )
