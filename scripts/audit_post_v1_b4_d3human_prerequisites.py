"""Audit the frozen, public-only B4 D3-Human prerequisites without execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "postv1_b4_d3human_public_baseline_r01"
AUDIT_CLOSED = True
EXPECTED_UPSTREAM_REVISION = "8841b45bc8adb3d0790d68164ffa1f1fee140f5a"
EXPECTED_COMMAND = (
    "CUDA_VISIBLE_DEVICES=0 python train.py -o res/f3c "
    "--folder_name female-3-casual --config configs/f3c.json"
)
RESTRICTIVE_CORE_FILES = (
    "train.py",
    "dataset/dataset_split.py",
    "geometry/hmsdf.py",
)
SMPLX_LICENSE_FILE = "deform/smplx_exavatar/body_models.py"
LICENSE_NAMES = {"license", "copying", "notice"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _license_paths(repo: Path) -> list[str]:
    paths: list[str] = []
    for path in repo.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        stem = path.name.lower().split(".", 1)[0]
        if stem in LICENSE_NAMES:
            paths.append(path.relative_to(repo).as_posix())
    return sorted(paths)


def audit(repo: Path, status_path: Path, source_revision: str) -> dict[str, Any]:
    status = yaml.safe_load(status_path.read_text(encoding="utf-8"))
    if status["baseline_id"] != BASELINE_ID:
        raise ValueError("B4 status file has the wrong baseline ID")
    if status["status"] != "registered_public_prerequisite_pending":
        raise RuntimeError("B4 prerequisite audit is not eligible")

    actual_revision = _git(repo, "rev-parse", "HEAD")
    git_clean = _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    readme = (repo / "README.md").read_text(encoding="utf-8")
    config = json.loads((repo / "configs/f3c.json").read_text(encoding="utf-8"))
    license_paths = _license_paths(repo)
    restrictive_headers = {
        relative: "strictly prohibited" in (repo / relative).read_text(encoding="utf-8")
        for relative in RESTRICTIVE_CORE_FILES
    }
    smplx_license_required = "without a valid license is prohibited" in (
        repo / SMPLX_LICENSE_FILE
    ).read_text(encoding="utf-8")

    checks: dict[str, str] = {
        "pinned_revision": ("pass" if actual_revision == EXPECTED_UPSTREAM_REVISION else "fail"),
        "clean_upstream_tree": "pass" if git_clean else "fail",
        "official_command": "pass" if EXPECTED_COMMAND in readme else "fail",
        "official_config": (
            "pass"
            if config.get("iter") == 10000
            and config.get("train_res") == [1080, 1080]
            and config.get("batch") == 1
            else "fail"
        ),
        "official_public_sequence": (
            "pass"
            if "female-3-casual" in readme and "1-OY5X7pnt45XBMURVTM55xhOrKKUi7BX" in readme
            else "fail"
        ),
        "project_code_license": "pass" if license_paths else "fail",
        "bundled_core_license_coverage": (
            "fail" if all(restrictive_headers.values()) else "unknown"
        ),
        "smplx_license_binding": "fail" if smplx_license_required else "unknown",
        "public_data_terms": (
            "fail"
            if "license" not in readme.lower() and "terms" not in readme.lower()
            else "unknown"
        ),
        "evaluator_mapping": "fail",
    }
    blockers = [name for name, result in checks.items() if result != "pass"]

    return {
        "schema_version": "post_v1_b4_prerequisite_audit.v1",
        "baseline_id": BASELINE_ID,
        "stage": "public_source_only_prerequisite",
        "status": "fail" if blockers else "pass",
        "source_revision": source_revision,
        "upstream": {
            "repository": status["upstream"]["repository"],
            "expected_revision": EXPECTED_UPSTREAM_REVISION,
            "actual_revision": actual_revision,
            "clean": git_clean,
            "readme_sha256": _sha256(repo / "README.md"),
            "config_sha256": _sha256(repo / "configs/f3c.json"),
            "train_sha256": _sha256(repo / "train.py"),
        },
        "official_binding": {
            "command": EXPECTED_COMMAND,
            "sequence": "female-3-casual",
            "iterations": config.get("iter"),
            "train_resolution": config.get("train_res"),
            "batch_size": config.get("batch"),
            "processed_data_listing": status["official_public_binding"]["processed_data_listing"],
            "data_downloaded": False,
            "licensed_model_read": False,
        },
        "license_audit": {
            "github_spdx": None,
            "repository_license_paths": license_paths,
            "restrictive_core_headers": restrictive_headers,
            "smplx_valid_license_required": smplx_license_required,
        },
        "checks": checks,
        "blockers": blockers,
        "execution": {
            "prerequisite_audit_attempts": 1,
            "public_data_downloads": 0,
            "licensed_model_reads": 0,
            "gpu_attempts": 0,
            "public_training_attempts": 0,
            "completed_iterations": 0,
            "endpoint_created": False,
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "sealed_test_accesses": 0,
            "automatic_paid_retries": 0,
        },
        "disposition": (
            "close_b4_before_data_model_or_gpu_without_license_runtime_data_or_schedule_substitute"
            if blockers
            else "eligible_for_separate_full_public_execution_registration"
        ),
    }


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to((PROJECT_ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("B4 audit reports must remain under ignored outputs/") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    if AUDIT_CLOSED:
        raise RuntimeError("B4 is terminally closed after its single prerequisite audit")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--status-config", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo.resolve(), args.status_config.resolve(), args.source_revision)
    _write_once(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}))


if __name__ == "__main__":
    main()
