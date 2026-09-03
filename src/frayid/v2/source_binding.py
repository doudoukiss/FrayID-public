from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast


class GitRunner(Protocol):
    def __call__(
        self,
        arguments: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]: ...


_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def resolve_packaged_source_revision(
    project_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    git_runner: GitRunner | None = None,
) -> str:
    """Resolve a source commit locally or from immutable remote image metadata."""

    environment = os.environ if environ is None else environ
    if environment.get("MODAL_IS_REMOTE") == "1":
        revision = environment.get("FRAYID_PACKAGED_SOURCE_REVISION", "").strip()
        if not _COMMIT_PATTERN.fullmatch(revision):
            raise RuntimeError("remote image is missing a valid packaged source revision")
        return revision
    runner = cast(GitRunner, subprocess.run) if git_runner is None else git_runner
    result = runner(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    revision = result.stdout.strip()
    if not _COMMIT_PATTERN.fullmatch(revision):
        raise RuntimeError("local Git HEAD is not a full source revision")
    return revision
