from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from frayid.selfrecon_reference import (
    BoundFile,
    EvaluationOutputContract,
    OfficialSchedule,
    PrebuiltRuntimeManifest,
    ReferenceDatasetContract,
    RuntimeRequirements,
    SelfReconReferenceSpec,
    audit_reference_binding,
    sha256_file,
)


def _write(path: Path, value: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[SelfReconReferenceSpec, Path]:
    evidence = tmp_path / "external"
    source = evidence / "SelfReconCode"
    source.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "fixture.invalid"], cwd=source, check=True)
    _write(
        source / "README.md",
        "This code is protected under patent, and it can be only used for research "
        "purposes.\nFor non-commercial research use only.\n",
    )
    _write(source / "config.conf", "nepoch = 200\n")
    _write(
        source / "environment.yml",
        "python=3.8.12\ncudatoolkit=11.3.1\npytorch=1.10.2\n",
    )
    _write(source / "train.py")
    _write(source / "infer.py")
    smpl = _write(source / "smpl_pytorch/model/male_smpl_with_cocoplus_reg.pkl")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    raw_archive = _write(evidence / "downloads/raw.zip", "raw")
    processed_archive = _write(evidence / "downloads/processed.zip", "processed")
    processed = evidence / "processed/male-3-casual"
    for index in range(2):
        for folder in ("imgs", "masks", "normals"):
            _write(processed / folder / f"{index}.png")
        _write(processed / "imgs" / f"{index}_rect.txt")
    _write(processed / "camera.npz")
    _write(processed / "smpl_rec.npz")

    spec = SelfReconReferenceSpec(
        schema_version="frayid_selfrecon_reference_spec.v1",
        experiment_id="postv3_r01_selfrecon_reference_reproduction_r01",
        upstream_repository="https://github.com/jby1993/SelfReconCode",
        upstream_revision=revision,
        upstream_relative_root="SelfReconCode",
        research_only_terms_acknowledged=True,
        patent_notice_acknowledged=True,
        owner_authorized_noncommercial_research_use=True,
        licensed_smpl=BoundFile(
            relative_path="SelfReconCode/smpl_pytorch/model/male_smpl_with_cocoplus_reg.pkl",
            sha256=sha256_file(smpl),
        ),
        dataset=ReferenceDatasetContract(
            raw_archive=BoundFile(
                relative_path="downloads/raw.zip", sha256=sha256_file(raw_archive)
            ),
            processed_archive=BoundFile(
                relative_path="downloads/processed.zip",
                sha256=sha256_file(processed_archive),
            ),
            processed_relative_root="processed/male-3-casual",
            image_count=2,
            rectangle_count=2,
            mask_count=2,
            normal_count=2,
        ),
        schedule=OfficialSchedule(),
        runtime=RuntimeRequirements(
            manifest_relative_path="runtime/r01.json",
            allowed_artifact_kinds=("oci_image", "apptainer_sif"),
        ),
        evaluation=EvaluationOutputContract(),
    )
    return spec, evidence


def _runtime_manifest(spec: SelfReconReferenceSpec) -> dict[str, object]:
    digest = f"sha256:{'a' * 64}"
    return {
        "schema_version": "frayid_selfrecon_prebuilt_runtime.v1",
        "experiment_id": spec.experiment_id,
        "artifact_kind": "oci_image",
        "artifact_locator": f"registry.example/frayid/selfrecon@{digest}",
        "artifact_digest": digest,
        "recipe_sha256": "b" * 64,
        "provenance_attestation_sha256": "c" * 64,
        "upstream_revision": spec.upstream_revision,
        "control_python": "3.11",
        "legacy_child_python": "3.8.12",
        "cuda": "11.3.1",
        "pytorch": "1.10.2",
        "torchvision": "0.11.3",
        "pytorch3d": "0.4.0",
        "prebuilt_before_remote_job": True,
        "modal_layer_assembly": False,
        "reuses_b2_or_b3_image": False,
        "scientific_source_modified": False,
        "licensed_assets_embedded": False,
        "dataset_embedded": False,
    }


def test_local_binding_passes_source_and_data_but_blocks_missing_runtime(
    tmp_path: Path,
) -> None:
    spec, evidence = _fixture(tmp_path)
    report = audit_reference_binding(spec=spec, evidence_root=evidence)
    assert report["status"] == "blocked"
    assert report["blockers"] == ["prebuilt_dual_runtime_artifact_manifest_missing"]
    assert report["source"]["status"] == "pass"
    assert report["processed_dataset"]["status"] == "pass"
    assert report["eligible_for_runtime_import"] is False
    assert report["eligible_for_scientific_attempt"] is False
    assert set(report["execution"].values()) == {0}


def test_content_addressed_prebuilt_runtime_unblocks_only_import(
    tmp_path: Path,
) -> None:
    spec, evidence = _fixture(tmp_path)
    manifest = evidence / spec.runtime.manifest_relative_path
    _write(manifest, json.dumps(_runtime_manifest(spec)))
    report = audit_reference_binding(spec=spec, evidence_root=evidence)
    assert report["status"] == "pass"
    assert report["blockers"] == []
    assert report["eligible_for_runtime_import"] is True
    assert report["eligible_for_device_qualification"] is False
    assert report["eligible_for_scientific_attempt"] is False


def test_runtime_manifest_rejects_mutable_locator_and_old_image_reuse(
    tmp_path: Path,
) -> None:
    spec, _ = _fixture(tmp_path)
    mutable = _runtime_manifest(spec)
    mutable["artifact_locator"] = "registry.example/frayid/selfrecon:latest"
    with pytest.raises(ValidationError, match="immutable sha256 digest"):
        PrebuiltRuntimeManifest.model_validate(mutable)
    reused = _runtime_manifest(spec)
    reused["reuses_b2_or_b3_image"] = True
    with pytest.raises(ValidationError):
        PrebuiltRuntimeManifest.model_validate(reused)


def test_reference_evidence_cannot_escape_registered_external_root(tmp_path: Path) -> None:
    spec, evidence = _fixture(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["dataset"]["raw_archive"]["relative_path"] = "../private.zip"
    escaped = SelfReconReferenceSpec.model_validate(payload)
    with pytest.raises(ValueError, match="escapes"):
        audit_reference_binding(spec=escaped, evidence_root=evidence)


def test_r01_auditor_has_no_network_camera_gpu_or_training_surface() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/audit_post_v3_r01_selfrecon_reference.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("requests", "urllib", "modal", "ffmpeg", "cv2", "train.py"):
        assert forbidden not in source
    assert "write_immutable_json" in source
    assert 'Path("qualification/local_binding_audit.json")' in source
