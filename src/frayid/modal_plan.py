from __future__ import annotations

from pathlib import Path

from frayid.config import ReconstructionConfig
from frayid.dataset import DATASET_MANIFEST_FILENAME, DATASET_VALIDATION_FILENAME
from frayid.initialization import INITIALIZATION_EVALUATION_FILENAME
from frayid.io import read_json, write_json
from frayid.schemas import (
    DatasetManifest,
    DatasetValidationReport,
    InitializationEvaluation,
    ModalSmokePlan,
    SmokeRunReport,
)


def build_modal_smoke_plan(
    config: ReconstructionConfig,
    *,
    config_path: Path,
    output_path: Path | None = None,
) -> ModalSmokePlan:
    blockers: list[str] = []
    dataset_root = config.paths.dataset_root
    manifest_path = dataset_root / DATASET_MANIFEST_FILENAME
    validation_path = dataset_root / DATASET_VALIDATION_FILENAME
    initialization_evaluation_path = dataset_root / INITIALIZATION_EVALUATION_FILENAME
    smoke_report_path = config.paths.run_root / config.run_id / "smoke/smoke_report.json"
    if not manifest_path.is_file():
        blockers.append("dataset_not_prepared")
    else:
        manifest = DatasetManifest.model_validate(read_json(manifest_path))
        if manifest.train_frame_count < config.smoke.frame_count:
            blockers.append("insufficient_training_frames_for_smoke")
    if not validation_path.is_file():
        blockers.append("dataset_evidence_not_validated")
    else:
        validation = DatasetValidationReport.model_validate(read_json(validation_path))
        if validation.status != "ready":
            blockers.append("dataset_evidence_gate_not_passed")
    if not initialization_evaluation_path.is_file():
        blockers.append("initialization_not_evaluated")
    else:
        initialization = InitializationEvaluation.model_validate(
            read_json(initialization_evaluation_path)
        )
        if initialization.status != "pass":
            blockers.append("initialization_gate_not_passed")
    if smoke_report_path.is_file():
        previous_smoke = SmokeRunReport.model_validate(read_json(smoke_report_path))
        if previous_smoke.status == "fail":
            blockers.append("previous_smoke_failed_manual_retry_authorization_required")
    command = [
        "modal",
        "run",
        "scripts/modal_geometry_smoke.py",
        "--config-path",
        "/workspace/configs/reconstruction/canonical_clothed_surface_v1.yaml",
    ]
    plan = ModalSmokePlan(
        run_id=config.run_id,
        gpu="L40S",
        timeout_seconds=config.smoke.timeout_seconds,
        frame_count=24,
        epoch_count=2,
        automatic_retry_count=0,
        full_training_authorized=config.smoke.full_training_authorized,
        config_path=str(config_path),
        dataset_manifest_path=str(manifest_path),
        command=command,
        status="ready" if not blockers else "blocked",
        blockers=blockers,
    )
    if output_path:
        write_json(output_path, plan)
    return plan
