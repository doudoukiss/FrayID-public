from __future__ import annotations

import shutil
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch

from frayid.assets import verify_assets
from frayid.config import load_config


def diagnose(config_path: Path) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for executable in ("git", "ffmpeg", "ffprobe"):
        resolved = shutil.which(executable)
        checks[executable] = {"ok": resolved is not None, "path": resolved}
    checks["python"] = {
        "ok": (3, 11) <= sys.version_info[:2] < (3, 13),
        "version": sys.version.split()[0],
    }
    try:
        config = load_config(config_path)
        checks["config"] = {"ok": True, "schema_version": config.schema_version}
    except Exception as exc:
        checks["config"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"status": "blocked", "checks": checks}
    asset_report = verify_assets(config.paths.asset_manifest)
    checks["local_assets"] = {
        "ok": asset_report.status == "ready",
        "status": asset_report.status,
        "errors": [error for check in asset_report.checks for error in check.errors],
    }
    checks["torch"] = {
        "ok": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }
    model_root = Path("models/private/camerahmr_assets/models/SMPL")
    checks["smpl_assets"] = {
        "ok": model_root.is_dir(),
        "path": str(model_root),
        "required_for": "initialize fit/evaluate and reconstruction",
    }
    checks["smplx_dependency"] = {
        "ok": find_spec("smplx") is not None,
        "install": "uv sync --extra smpl",
        "required_for": "initialize fit/evaluate and reconstruction",
    }
    core_names = ("git", "ffmpeg", "ffprobe", "python", "config", "local_assets", "torch")
    core_ready = all(bool(checks[name]["ok"]) for name in core_names)
    reconstruction_ready = (
        core_ready and bool(checks["smpl_assets"]["ok"]) and bool(checks["smplx_dependency"]["ok"])
    )
    return {
        "status": "ready" if core_ready else "blocked",
        "reconstruction_status": "ready_for_evidence_validation"
        if reconstruction_ready
        else "blocked",
        "checks": checks,
    }
