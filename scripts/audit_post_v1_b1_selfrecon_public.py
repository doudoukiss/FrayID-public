"""Audit the isolated public prerequisites for the pinned SelfRecon B1 baseline.

This command is deliberately read-only except for its immutable JSON report. It
does not download licensed assets, build CUDA extensions, launch training, or
inspect any FrayID private/development/sealed path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_b1_selfrecon_public_baseline_r01"
REPORT_SCHEMA = "post_v1_b1_selfrecon_public_prerequisite_audit.v1"
UPSTREAM_REVISION = "344b86fc3e7617b94b5c9da3741c764ae93cacaa"
ALLOWED_UNTRACKED_PREFIX = "docs/0901/"
ISOLATED_EXTERNAL_ROOT = PROJECT_ROOT / "external"
ISOLATED_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PROTECTED_ROOTS = (
    PROJECT_ROOT / "data/private",
    PROJECT_ROOT / "models/private",
    PROJECT_ROOT / "models/checkpoints",
)
SOURCE_FILES = (
    "README.md",
    "config.conf",
    "environment.yml",
    "install.sh",
    "train.py",
    "dataset/dataset.py",
    "people_snapshot_process.py",
    "smpl_pytorch/SMPL.py",
)
RAW_DATA_FILES = (
    "masks.hdf5",
    "camera.pkl",
    "reconstructed_poses.hdf5",
)
PROCESSED_DATA_FILES = ("camera.npz", "smpl_rec.npz")
CUDA_EXTENSION_SOURCES = (
    "FastMinv/Matrix3x3InvKernels.cu",
    "MCGpu/CudaKernels.cu",
    "MCAcc/cuda/GridSamplerMineKernel.cu",
)
CUDA_EXTENSION_MODULE_PATTERNS = (
    "FastMinv*.so",
    "MCGpu*.so",
    "interp2x_boundary2d*.so",
    "interp2x_boundary3d*.so",
    "GridSamplerMine*.so",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _allowed_subprocesses() -> tuple[str, ...]:
    """Declare the complete command surface used by this read-only auditor."""
    return ("git", "nvidia-smi")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _assert_isolated_input(path: Path, label: str) -> Path:
    resolved = path.resolve()
    for protected in PROTECTED_ROOTS:
        if _is_within(resolved, protected):
            raise ValueError(f"{label} may not read protected path {protected}")
    if not _is_within(resolved, ISOLATED_EXTERNAL_ROOT):
        raise ValueError(f"{label} must remain under ignored external/")
    return resolved


def _assert_isolated_output(path: Path) -> Path:
    resolved = path.resolve()
    if not _is_within(resolved, ISOLATED_OUTPUT_ROOT):
        raise ValueError("B1 audit reports must remain under ignored outputs/")
    return resolved


def _project_git_binding() -> dict[str, Any]:
    revision = _run(["git", "rev-parse", "HEAD"], PROJECT_ROOT)[1]
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"], PROJECT_ROOT)[
        1
    ].splitlines()
    disallowed = [
        record
        for record in status
        if not (record.startswith("?? ") and record[3:].startswith(ALLOWED_UNTRACKED_PREFIX))
    ]
    return {
        "revision": revision,
        "implementation_tree_clean": not disallowed,
        "allowed_untracked_advisory_prefix": ALLOWED_UNTRACKED_PREFIX,
        "disallowed_status_records": disallowed,
    }


def _upstream_binding(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {
            "present": False,
            "revision": None,
            "revision_matches": False,
            "tree_clean": False,
            "source_hashes": {},
            "missing_source_files": list(SOURCE_FILES),
        }
    revision = _run(["git", "rev-parse", "HEAD"], root)[1]
    tree_status = _run(["git", "status", "--porcelain", "--untracked-files=all"], root)[
        1
    ].splitlines()
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for relative in SOURCE_FILES:
        path = root / relative
        if path.is_file():
            hashes[relative] = _sha256(path)
        else:
            missing.append(relative)
    return {
        "present": True,
        "revision": revision,
        "revision_matches": revision == UPSTREAM_REVISION,
        "tree_clean": not tree_status,
        "tree_status": tree_status,
        "source_hashes": hashes,
        "missing_source_files": missing,
    }


def _source_contract(root: Path) -> dict[str, Any]:
    config_path = root / "config.conf"
    environment_path = root / "environment.yml"
    readme_path = root / "README.md"
    if not all(path.is_file() for path in (config_path, environment_path, readme_path)):
        return {"status": "fail", "checks": {}}
    config = config_path.read_text()
    environment = environment_path.read_text()
    readme = readme_path.read_text()
    checks = {
        "nepoch_200": bool(re.search(r"(?m)^\s*nepoch\s*=\s*200\s*$", config)),
        "rgb_weights_0_5_1_0_1_0": re.findall(r"(?m)^\s*color_weight\s*=\s*([0-9.]+)\s*$", config)
        == ["0.5", "1.0", "1.0"],
        "normal_weights_0_1": re.findall(r"(?m)^\s*normal_weight\s*=\s*([0-9.]+)\s*$", config)
        == ["0.1", "0.1", "0.1"],
        "pose_optimized": bool(re.search(r"(?m)^\s*opt_pose\s*=\s*true\s*$", config)),
        "translation_optimized": bool(re.search(r"(?m)^\s*opt_trans\s*=\s*true\s*$", config)),
        "camera_focal_optimized": bool(re.search(r"(?m)^\s*focal_length\s*=\s*true\s*$", config)),
        "camera_principal_optimized": bool(
            re.search(r"(?m)^\s*princeple_points\s*=\s*true\s*$", config)
        ),
        "camera_translation_optimized": bool(re.search(r"(?m)^\s*T\s*=\s*true\s*$", config)),
        "python_3_8_12": "python=3.8.12" in environment,
        "cuda_11_3_1": "cudatoolkit=11.3.1" in environment,
        "pytorch_1_10_2": "pytorch=1.10.2" in environment,
        "torchvision_0_11_3": "torchvision=0.11.3" in environment,
        "pytorch3d_0_4_0_documented": "pytorch3d-0.4.0" in readme,
        "research_only_terms_present": (
            "only used for research purposes" in readme
            and "For non-commercial research use only" in readme
        ),
        "full_training_documented": (
            "python train.py" in readme and "this may take one day to finish" in readme
        ),
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def _file_inventory(root: Path, relatives: tuple[str, ...]) -> dict[str, Any]:
    present = [relative for relative in relatives if (root / relative).is_file()]
    missing = [relative for relative in relatives if relative not in present]
    return {
        "root": str(root),
        "present": present,
        "missing": missing,
        "status": "pass" if not missing else "fail",
    }


def _dataset_contract(raw_root: Path, processed_root: Path) -> dict[str, Any]:
    raw = _file_inventory(raw_root, RAW_DATA_FILES)
    videos = sorted(path.name for path in raw_root.glob("*.mp4")) if raw_root.is_dir() else []
    raw["videos"] = videos
    if not videos:
        raw["status"] = "fail"

    processed = _file_inventory(processed_root, PROCESSED_DATA_FILES)
    counts: dict[str, int] = {}
    for folder in ("imgs", "masks", "normals"):
        directory = processed_root / folder
        counts[folder] = (
            sum(1 for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".png"})
            if directory.is_dir()
            else 0
        )
    aligned_nonzero = counts["imgs"] > 0 and len(set(counts.values())) == 1
    processed["frame_counts"] = counts
    processed["rgb_mask_normal_alignment"] = aligned_nonzero
    if not aligned_nonzero:
        processed["status"] = "fail"
    return {
        "sequence": "male-3-casual",
        "raw": raw,
        "processed": processed,
        "status": "pass" if raw["status"] == processed["status"] == "pass" else "fail",
    }


def _runtime_contract(upstream_root: Path) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")
    conda = shutil.which("conda")
    gpu_probe = _run([nvidia_smi, "-L"], PROJECT_ROOT) if nvidia_smi else (127, "", "not found")
    source_presence = {
        relative: (upstream_root / relative).is_file() for relative in CUDA_EXTENSION_SOURCES
    }
    modules: dict[str, list[str]] = {}
    for pattern in CUDA_EXTENSION_MODULE_PATTERNS:
        modules[pattern] = sorted(str(path) for path in upstream_root.rglob(pattern))
    checks = {
        "linux_x86_64": platform.system() == "Linux" and platform.machine() == "x86_64",
        "nvidia_smi_available": nvidia_smi is not None,
        "nvidia_gpu_visible": gpu_probe[0] == 0 and bool(gpu_probe[1]),
        "nvcc_available": nvcc is not None,
        "conda_available": conda is not None,
        "cuda_extension_sources_complete": all(source_presence.values()),
        "cuda_extension_modules_built": all(modules.values()),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "commands": {"nvidia_smi": nvidia_smi, "nvcc": nvcc, "conda": conda},
        "gpu_probe": {"returncode": gpu_probe[0], "stdout": gpu_probe[1], "stderr": gpu_probe[2]},
        "extension_sources": source_presence,
        "extension_modules": modules,
        "checks": checks,
    }


def audit(
    upstream_root: Path,
    raw_data_root: Path,
    processed_data_root: Path,
    *,
    license_authorized: bool,
) -> dict[str, Any]:
    upstream_root = _assert_isolated_input(upstream_root, "upstream root")
    raw_data_root = _assert_isolated_input(raw_data_root, "raw public data root")
    processed_data_root = _assert_isolated_input(processed_data_root, "processed public data root")
    project = _project_git_binding()
    upstream = _upstream_binding(upstream_root)
    source = _source_contract(upstream_root)
    smpl = _file_inventory(
        upstream_root / "smpl_pytorch/model", ("male_smpl_with_cocoplus_reg.pkl",)
    )
    dataset = _dataset_contract(raw_data_root, processed_data_root)
    runtime = _runtime_contract(upstream_root)
    blockers: list[str] = []
    if not project["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")
    if not upstream["present"] or not upstream["revision_matches"] or not upstream["tree_clean"]:
        blockers.append("pinned_upstream_revision_not_cleanly_bound")
    if upstream["missing_source_files"] or source["status"] != "pass":
        blockers.append("official_source_or_configuration_contract_failed")
    if not license_authorized:
        blockers.append("research_only_license_not_owner_authorized")
    if smpl["status"] != "pass":
        blockers.append("licensed_smpl_model_missing")
    if dataset["status"] != "pass":
        blockers.append("male_3_casual_public_dataset_or_normals_missing")
    if runtime["status"] != "pass":
        blockers.append("official_cuda_runtime_unavailable")
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "stage": "public_prerequisite_audit",
        "status": "pass" if not blockers else "fail",
        "terminal_disposition": (
            "eligible_for_one_full_200_epoch_public_attempt"
            if not blockers
            else "close_before_public_reconstruction"
        ),
        "blockers": blockers,
        "project_binding": project,
        "upstream": upstream,
        "official_source_contract": source,
        "license": {
            "terms_present": source.get("checks", {}).get("research_only_terms_present", False),
            "owner_authorized_for_this_run": license_authorized,
        },
        "compatibility_overlays": {
            "count": 0,
            "files": [],
            "upstream_tree_modified": False,
        },
        "smpl": smpl,
        "dataset": dataset,
        "runtime": runtime,
        "execution": {
            "public_reconstruction_attempts": 0,
            "epochs_completed": 0,
            "evaluator_adapter_runs": 0,
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "sealed_test_accesses": 0,
            "gpu_hours": 0,
            "cloud_invocations": 0,
            "spend_usd": 0,
            "automatic_paid_retries": 0,
        },
        "comparison_scope": {
            "selfrecon_uses_rgb": True,
            "e16_and_e18_use_rgb": False,
            "causal_ablation": False,
            "external_evidence_rich_baseline": True,
        },
    }


def _default_output() -> Path:
    revision = _run(["git", "rev-parse", "--short=8", "HEAD"], PROJECT_ROOT)[1]
    return (
        ISOLATED_OUTPUT_ROOT / "post_v1_b1_selfrecon_public_prereq_r01" / f"{revision}_report.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root", type=Path, default=ISOLATED_EXTERNAL_ROOT / "SelfReconCode"
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        default=ISOLATED_EXTERNAL_ROOT / "people_snapshot_public/male-3-casual",
    )
    parser.add_argument(
        "--processed-data-root",
        type=Path,
        default=ISOLATED_EXTERNAL_ROOT / "selfrecon_processed/male-3-casual",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--license-authorized",
        action="store_true",
        help="record explicit owner authorization for the upstream research-only terms",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = _assert_isolated_output(args.output or _default_output())
    report = audit(
        args.upstream_root,
        args.raw_data_root,
        args.processed_data_root,
        license_authorized=args.license_authorized,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {"output": str(output), "status": report["status"], "blockers": report["blockers"]},
            indent=2,
        )
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
