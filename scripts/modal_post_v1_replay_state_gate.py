"""Run the public P0 exact next-step replay gate on one L40S."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11")
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy>=1.26,<3", "scikit-image>=0.24")
    .add_local_dir(PROJECT_ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
)
app = modal.App("frayid-postv1-p0-replay-state-l40s-r01", image=image)


@app.function(
    gpu="L40S",
    cpu=2.0,
    memory=8192,
    timeout=600,
    retries=0,
    env={"PYTHONPATH": str(REMOTE_ROOT / "src")},
)
def run_gate(source_revision: str) -> dict[str, Any]:
    from frayid.replay_gate import run_replay_gate

    report = run_replay_gate("cuda")
    report["source_revision"] = source_revision
    report["maximum_gpu_seconds"] = 600
    report["automatic_paid_retries"] = 0
    report["private_inputs_loaded"] = 0
    report["sealed_test_accesses"] = 0
    return report


@app.local_entrypoint()
def main(output: str) -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("P0 L40S gate requires a clean source revision")
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"immutable output already exists: {output_path}")
    report = run_gate.remote(revision)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)
