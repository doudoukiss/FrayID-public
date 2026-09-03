"""Audit the frozen, public-only B5 REC-MV prerequisites without execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "postv1_b5_recmv_public_baseline_r01"
AUDIT_CLOSED = True
EXPECTED_UPSTREAM_REVISION = "5898020efc3a7b9fb50c255a59ee638e3b9676e7"
EXPECTED_WRAPPER_COMMAND = (
    "bash ./scripts/people_snapshot/train_female-3-casual.sh 0 ${exp_name} ${wandb name}"
)
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
        raise ValueError("B5 status file has the wrong baseline ID")
    if status["status"] != "registered_public_prerequisite_pending":
        raise RuntimeError("B5 prerequisite audit is not eligible")

    actual_revision = _git(repo, "rev-parse", "HEAD")
    git_clean = _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    readme = (repo / "README.md").read_text(encoding="utf-8")
    license_text = (repo / "LICENSE").read_text(encoding="utf-8")
    config_text = (repo / "configs/people_snapshot/female-3-casual.conf").read_text(
        encoding="utf-8"
    )
    wrapper_text = (repo / "scripts/people_snapshot/train_female-3-casual.sh").read_text(
        encoding="utf-8"
    )
    license_paths = _license_paths(repo)

    source_license_present = "MIT License" in license_text and "LICENSE" in license_paths
    badge_conflicts = "License-Apache_2.0" in readme and source_license_present
    official_command = EXPECTED_WRAPPER_COMMAND in readme
    official_wrapper = all(
        token in wrapper_text
        for token in (
            "configs/people_snapshot/female-3-casual.conf",
            "people_snapshot_public_proprecess/female-3-casual/",
            "female_3_casual_fl",
            "--data_type people_snapshot",
            "--a_pose",
        )
    )
    official_schedule = all(
        token in config_text
        for token in (
            "nepoch = 200",
            "start_epoch = 0",
            "start_epoch = 8",
            "start_epoch = 12",
            "color_weight = 0.5",
            "color_weight = 1.0",
            "normal_weight = 0.1",
            "opt_pose = true",
            "opt_trans = true",
        )
    )

    bindings = status["required_external_bindings"]
    processed = bindings["processed_people_snapshot"]
    templates = bindings["labeled_garment_templates"]
    smpl = bindings["smpl"]
    preprocess = bindings["preprocess_mirror"]
    checks: dict[str, str] = {
        "pinned_revision": "pass" if actual_revision == EXPECTED_UPSTREAM_REVISION else "fail",
        "clean_upstream_tree": "pass" if git_clean else "fail",
        "root_project_license": "pass" if source_license_present else "fail",
        "unambiguous_project_license": "fail" if badge_conflicts else "pass",
        "official_command": "pass" if official_command and official_wrapper else "fail",
        "official_config_and_full_schedule": "pass" if official_schedule else "fail",
        "public_sequence_archive_hash_and_terms": (
            "pass" if processed["archive_hash_bound"] and processed["terms_bound"] else "fail"
        ),
        "labeled_template_bundle_hash_and_terms": (
            "pass"
            if templates["recmv_labeled_bundle_hash_bound"]
            and templates["recmv_labeled_bundle_terms_bound"]
            else "fail"
        ),
        "licensed_smpl_binding": "pass" if smpl["licensed_model_bound"] else "fail",
        "preprocessing_dependency_license_coverage": (
            "pass"
            if preprocess["license_metadata"] != "absent"
            and not preprocess["redistributed_smpl_pkls_present"]
            else "fail"
        ),
        "evaluator_mapping": "fail",
    }
    blockers = [name for name, result in checks.items() if result != "pass"]

    return {
        "schema_version": "post_v1_b5_prerequisite_audit.v1",
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
            "license_sha256": _sha256(repo / "LICENSE"),
            "config_sha256": _sha256(repo / "configs/people_snapshot/female-3-casual.conf"),
            "wrapper_sha256": _sha256(repo / "scripts/people_snapshot/train_female-3-casual.sh"),
        },
        "official_binding": {
            "command": EXPECTED_WRAPPER_COMMAND,
            "sequence": "female-3-casual",
            "epochs": 200,
            "data_downloaded": False,
            "licensed_model_read": False,
            "dependency_build_started": False,
        },
        "license_and_access_audit": {
            "repository_license_paths": license_paths,
            "root_license_spdx": "MIT" if source_license_present else None,
            "readme_badge_spdx": "Apache-2.0" if badge_conflicts else None,
            "processed_people_snapshot": processed,
            "labeled_garment_templates": templates,
            "smpl": smpl,
            "preprocess_mirror": preprocess,
        },
        "checks": checks,
        "blockers": blockers,
        "execution": {
            "prerequisite_audit_attempts": 1,
            "public_data_downloads": 0,
            "licensed_model_reads": 0,
            "dependency_build_attempts": 0,
            "gpu_attempts": 0,
            "public_training_attempts": 0,
            "completed_epochs": 0,
            "endpoint_created": False,
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "sealed_test_accesses": 0,
            "automatic_paid_retries": 0,
        },
        "disposition": (
            "close_b5_before_data_model_dependency_or_gpu_without_asset_license_or_data_substitute"
            if blockers
            else "eligible_for_separate_full_200_epoch_public_execution_registration"
        ),
    }


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to((PROJECT_ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("B5 audit reports must remain under ignored outputs/") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    if AUDIT_CLOSED:
        raise RuntimeError("B5 is terminally closed after its single prerequisite audit")
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
