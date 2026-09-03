from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundFile(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReferenceDatasetContract(StrictModel):
    sequence: Literal["male-3-casual"] = "male-3-casual"
    raw_archive: BoundFile
    processed_archive: BoundFile
    processed_relative_root: str
    image_count: int = Field(gt=0)
    rectangle_count: int = Field(gt=0)
    mask_count: int = Field(gt=0)
    normal_count: int = Field(gt=0)


class OfficialSchedule(StrictModel):
    epochs: Literal[200] = 200
    rgb_weights: tuple[float, float, float] = (0.5, 1.0, 1.0)
    normal_weight: float = 0.1
    optimize_pose: Literal[True] = True
    optimize_translation: Literal[True] = True
    optimize_camera: Literal[True] = True

    @model_validator(mode="after")
    def _weights_match_official_schedule(self) -> OfficialSchedule:
        if self.rgb_weights != (0.5, 1.0, 1.0) or self.normal_weight != 0.1:
            raise ValueError("SelfRecon loss weights must match the official schedule")
        return self


class RuntimeRequirements(StrictModel):
    manifest_relative_path: str
    allowed_artifact_kinds: tuple[Literal["oci_image", "apptainer_sif"], ...]
    control_python: Literal["3.11"] = "3.11"
    legacy_child_python: Literal["3.8.12"] = "3.8.12"
    cuda: Literal["11.3.1"] = "11.3.1"
    pytorch: Literal["1.10.2"] = "1.10.2"
    torchvision: Literal["0.11.3"] = "0.11.3"
    pytorch3d: Literal["0.4.0"] = "0.4.0"
    prebuilt_before_remote_job: Literal[True] = True
    modal_layer_assembly_allowed: Literal[False] = False
    b2_or_b3_image_reuse_allowed: Literal[False] = False
    licensed_assets_embedded: Literal[False] = False
    dataset_embedded: Literal[False] = False


class EvaluationOutputContract(StrictModel):
    required_endpoint_epoch: Literal[200] = 200
    checkpoint_state_required: Literal[True] = True
    canonical_geometry_required: Literal[True] = True
    posed_geometry_required: Literal[True] = True
    geometry_only_render_required: Literal[True] = True
    exact_replay_manifest_required: Literal[True] = True
    pretrained_endpoint_allowed: Literal[False] = False
    shortened_schedule_allowed: Literal[False] = False
    file_existence_is_success: Literal[False] = False


class SelfReconReferenceSpec(StrictModel):
    schema_version: Literal["frayid_selfrecon_reference_spec.v1"]
    experiment_id: Literal["postv3_r01_selfrecon_reference_reproduction_r01"]
    upstream_repository: Literal["https://github.com/jby1993/SelfReconCode"]
    upstream_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    upstream_relative_root: str
    research_only_terms_acknowledged: Literal[True]
    patent_notice_acknowledged: Literal[True]
    owner_authorized_noncommercial_research_use: Literal[True]
    licensed_smpl: BoundFile
    dataset: ReferenceDatasetContract
    schedule: OfficialSchedule
    runtime: RuntimeRequirements
    evaluation: EvaluationOutputContract


class PrebuiltRuntimeManifest(StrictModel):
    schema_version: Literal["frayid_selfrecon_prebuilt_runtime.v1"]
    experiment_id: Literal["postv3_r01_selfrecon_reference_reproduction_r01"]
    artifact_kind: Literal["oci_image", "apptainer_sif"]
    artifact_locator: str = Field(min_length=1)
    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_attestation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    control_python: Literal["3.11"]
    legacy_child_python: Literal["3.8.12"]
    cuda: Literal["11.3.1"]
    pytorch: Literal["1.10.2"]
    torchvision: Literal["0.11.3"]
    pytorch3d: Literal["0.4.0"]
    prebuilt_before_remote_job: Literal[True]
    modal_layer_assembly: Literal[False]
    reuses_b2_or_b3_image: Literal[False]
    scientific_source_modified: Literal[False]
    licensed_assets_embedded: Literal[False]
    dataset_embedded: Literal[False]

    @model_validator(mode="after")
    def _artifact_is_content_addressed(self) -> PrebuiltRuntimeManifest:
        if "@sha256:" not in self.artifact_locator:
            raise ValueError("runtime artifact locator must contain an immutable sha256 digest")
        if not self.artifact_locator.endswith(self.artifact_digest):
            raise ValueError("runtime artifact locator and digest must agree")
        return self


def load_reference_spec(path: Path) -> SelfReconReferenceSpec:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SelfRecon reference spec must be a YAML mapping")
    return SelfReconReferenceSpec.model_validate(payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _bound_path(evidence_root: Path, relative_path: str) -> Path:
    path = (evidence_root / relative_path).resolve()
    if not _inside(path, evidence_root):
        raise ValueError("reference evidence path escapes the registered external root")
    return path


def _git(command: list[str], root: Path) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", *command],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    return completed.returncode, completed.stdout.strip()


def _source_audit(root: Path, expected_revision: str) -> dict[str, Any]:
    if not root.is_dir():
        return {"status": "fail", "present": False}
    revision_code, revision = _git(["rev-parse", "HEAD"], root)
    status_code, tree_status = _git(["status", "--porcelain", "--untracked-files=all"], root)
    readme = root / "README.md"
    config = root / "config.conf"
    environment = root / "environment.yml"
    required = (readme, config, environment, root / "train.py", root / "infer.py")
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    config_text = config.read_text(encoding="utf-8") if config.is_file() else ""
    environment_text = environment.read_text(encoding="utf-8") if environment.is_file() else ""
    checks = {
        "revision_exact": revision_code == 0 and revision == expected_revision,
        "tree_clean": status_code == 0 and not tree_status,
        "required_entrypoints_present": all(path.is_file() for path in required),
        "research_only_notice": "only used for research purposes" in text
        and "For non-commercial research use only" in text,
        "patent_notice": "protected under patent" in text,
        "epoch_200": "nepoch = 200" in config_text,
        "python_3_8_12": "python=3.8.12" in environment_text,
        "cuda_11_3_1": "cudatoolkit=11.3.1" in environment_text,
        "pytorch_1_10_2": "pytorch=1.10.2" in environment_text,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "present": True,
        "revision": revision,
        "tree_status": tree_status.splitlines(),
        "checks": checks,
    }


def _file_audit(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path) if path.is_file() else None
    return {
        "path": str(path),
        "present": path.is_file(),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "status": "pass" if observed == expected_sha256 else "fail",
    }


def _processed_dataset_audit(root: Path, spec: ReferenceDatasetContract) -> dict[str, Any]:
    suffix_by_folder = {"imgs": ".png", "masks": ".png", "normals": ".png"}
    stems: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for folder, suffix in suffix_by_folder.items():
        paths = sorted((root / folder).glob(f"*{suffix}")) if (root / folder).is_dir() else []
        stems[folder] = {path.stem for path in paths}
        counts[folder] = len(paths)
    rectangles = sorted((root / "imgs").glob("*_rect.txt")) if (root / "imgs").is_dir() else []
    counts["rectangles"] = len(rectangles)
    expected = {
        "imgs": spec.image_count,
        "rectangles": spec.rectangle_count,
        "masks": spec.mask_count,
        "normals": spec.normal_count,
    }
    checks = {
        "counts_exact": counts == expected,
        "rgb_mask_normal_stems_aligned": bool(stems["imgs"])
        and stems["imgs"] == stems["masks"] == stems["normals"],
        "camera_present": (root / "camera.npz").is_file(),
        "smpl_initialization_present": (root / "smpl_rec.npz").is_file(),
    }
    return {
        "root": str(root),
        "counts": counts,
        "expected_counts": expected,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def _runtime_audit(path: Path, spec: SelfReconReferenceSpec) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "blocked", "present": False, "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = PrebuiltRuntimeManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        return {
            "status": "fail",
            "present": True,
            "path": str(path),
            "validation_error": str(exc),
        }
    requirements = spec.runtime
    checks = {
        "artifact_kind_allowed": manifest.artifact_kind in requirements.allowed_artifact_kinds,
        "upstream_revision_exact": manifest.upstream_revision == spec.upstream_revision,
        "control_python_exact": manifest.control_python == requirements.control_python,
        "legacy_child_python_exact": (
            manifest.legacy_child_python == requirements.legacy_child_python
        ),
        "cuda_exact": manifest.cuda == requirements.cuda,
        "pytorch_exact": manifest.pytorch == requirements.pytorch,
        "torchvision_exact": manifest.torchvision == requirements.torchvision,
        "pytorch3d_exact": manifest.pytorch3d == requirements.pytorch3d,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "present": True,
        "path": str(path),
        "manifest": manifest.model_dump(mode="json"),
        "checks": checks,
    }


def audit_reference_binding(*, spec: SelfReconReferenceSpec, evidence_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    source = _source_audit(
        _bound_path(evidence_root, spec.upstream_relative_root), spec.upstream_revision
    )
    raw_archive = _file_audit(
        _bound_path(evidence_root, spec.dataset.raw_archive.relative_path),
        spec.dataset.raw_archive.sha256,
    )
    processed_archive = _file_audit(
        _bound_path(evidence_root, spec.dataset.processed_archive.relative_path),
        spec.dataset.processed_archive.sha256,
    )
    smpl = _file_audit(
        _bound_path(evidence_root, spec.licensed_smpl.relative_path), spec.licensed_smpl.sha256
    )
    processed = _processed_dataset_audit(
        _bound_path(evidence_root, spec.dataset.processed_relative_root), spec.dataset
    )
    runtime = _runtime_audit(_bound_path(evidence_root, spec.runtime.manifest_relative_path), spec)
    blockers: list[str] = []
    for name, result in (
        ("pinned_source", source),
        ("raw_archive", raw_archive),
        ("processed_archive", processed_archive),
        ("licensed_smpl", smpl),
        ("processed_dataset", processed),
    ):
        if result["status"] != "pass":
            blockers.append(f"{name}_binding_failed")
    if runtime["status"] == "blocked":
        blockers.append("prebuilt_dual_runtime_artifact_manifest_missing")
    elif runtime["status"] != "pass":
        blockers.append("prebuilt_dual_runtime_artifact_invalid")
    return {
        "schema_version": "frayid_post_v3_r01_local_binding_audit.v1",
        "experiment_id": spec.experiment_id,
        "stage": "local_zero_gpu_binding",
        "status": "pass" if not blockers else "blocked",
        "qualification_state": "built",
        "eligible_for_runtime_import": not blockers,
        "eligible_for_device_qualification": False,
        "eligible_for_scientific_attempt": False,
        "blockers": blockers,
        "source": source,
        "license": {
            "research_only_terms_acknowledged": spec.research_only_terms_acknowledged,
            "patent_notice_acknowledged": spec.patent_notice_acknowledged,
            "owner_authorized_noncommercial_research_use": (
                spec.owner_authorized_noncommercial_research_use
            ),
        },
        "raw_archive": raw_archive,
        "processed_archive": processed_archive,
        "licensed_smpl": smpl,
        "processed_dataset": processed,
        "runtime": runtime,
        "official_schedule": spec.schedule.model_dump(mode="json"),
        "evaluation_output_contract": spec.evaluation.model_dump(mode="json"),
        "execution": {
            "downloads": 0,
            "camera_opens": 0,
            "gpu_workers": 0,
            "training_attempts": 0,
            "optimizer_steps": 0,
            "private_project_video_reads": 0,
            "development_reads": 0,
            "sealed_test_reads": 0,
            "spend_usd": 0,
            "automatic_paid_retries": 0,
        },
    }
