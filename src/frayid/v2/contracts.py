from __future__ import annotations

import json
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from frayid.io import write_json

EXPERIMENT_ID_PATTERN = re.compile(r"^postv2_[a-z][a-z0-9_]+_r[0-9]{2}$")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{7,127}$")
SEALED_TOKENS = frozenset({"sealed", "sealed_test", "sealed-test", "test_v1"})
GateValue: TypeAlias = bool | float | int | str


class QualificationState(StrEnum):
    UNBUILT = "unbuilt"
    BUILT = "built"
    IMPORTED = "imported"
    DATA_BOUND = "data_bound"
    DEVICE_VALIDATED = "device_validated"
    ONE_STEP_PASSED = "one_step_passed"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    EVALUATOR_DRY = "evaluator_dry"
    QUALIFIED = "qualified"


QUALIFICATION_SEQUENCE = tuple(QualificationState)


class ScientificAttemptState(StrEnum):
    REGISTERED = "registered"
    ATTEMPT_STARTED = "attempt_started"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    AUDITED = "audited"
    TERMINAL = "terminal"


class ComputeCap(BaseModel):
    qualification_gpu_hours: float = Field(ge=0)
    scientific_gpu_hours: float = Field(ge=0)
    wall_time_seconds: int = Field(gt=0)
    maximum_cost_usd: float | None = Field(default=None, gt=0)
    price_rate_checked_at: str | None = None
    automatic_retries: Literal[0] = 0

    @model_validator(mode="after")
    def require_priced_scientific_dispatch(self) -> ComputeCap:
        if (self.maximum_cost_usd is None) != (self.price_rate_checked_at is None):
            raise ValueError("cost cap and price-check timestamp must be recorded together")
        return self


class EvidenceBoundary(BaseModel):
    train_masks: bool = True
    train_normals: bool = True
    train_semantics: bool = False
    train_tracks: bool = False
    train_rgb: bool = False
    development_evaluator_only: bool = True
    sealed_test_access: Literal[False] = False
    generated_views_as_evidence: Literal[False] = False


class V2ExperimentContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["frayid_v2_experiment_contract.v1"] = "frayid_v2_experiment_contract.v1"
    experiment_id: str
    run_id: str
    hypothesis: str = Field(min_length=20)
    changed_mechanism: str = Field(min_length=5)
    matched_control: str = Field(min_length=5)
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    immutable_output_root: Path
    evidence: EvidenceBoundary
    compute_cap: ComputeCap
    qualification_state: QualificationState = QualificationState.UNBUILT
    scientific_state: ScientificAttemptState = ScientificAttemptState.REGISTERED
    dependencies: list[str] = Field(default_factory=list)
    promotion_gates: dict[str, GateValue] = Field(default_factory=dict)
    stop_conditions: list[str] = Field(default_factory=list)
    historical_records_immutable: Literal[True] = True
    automatic_paid_retries: Literal[0] = 0

    @model_validator(mode="after")
    def validate_identity_and_path(self) -> V2ExperimentContract:
        if EXPERIMENT_ID_PATTERN.fullmatch(self.experiment_id) is None:
            raise ValueError("experiment_id must use the postv2_*_rNN namespace")
        if RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a stable lowercase identifier")
        expected = Path("outputs") / "post_v2" / self.experiment_id / self.run_id
        if self.immutable_output_root != expected:
            raise ValueError(f"immutable output must be exactly {expected}")
        if self.experiment_id in self.dependencies:
            raise ValueError("experiment cannot depend on itself")
        if any(EXPERIMENT_ID_PATTERN.fullmatch(item) is None for item in self.dependencies):
            raise ValueError("dependencies must use registered postv2 experiment IDs")
        reject_sealed_capability([self.immutable_output_root])
        return self

    @property
    def priced_for_scientific_dispatch(self) -> bool:
        return self.compute_cap.maximum_cost_usd is not None


class QualificationRecord(BaseModel):
    schema_version: Literal["frayid_v2_qualification_record.v1"] = (
        "frayid_v2_qualification_record.v1"
    )
    experiment_id: str
    run_id: str
    state: QualificationState
    checks: dict[str, bool]
    blockers: list[str] = Field(default_factory=list)
    private_reads: int = Field(default=0, ge=0)
    development_reads: int = Field(default=0, ge=0)
    sealed_test_reads: Literal[0] = 0
    attempt_marker_created: Literal[False] = False


def advance_qualification(
    current: QualificationState, requested: QualificationState
) -> QualificationState:
    current_index = QUALIFICATION_SEQUENCE.index(current)
    requested_index = QUALIFICATION_SEQUENCE.index(requested)
    if requested_index != current_index + 1:
        raise ValueError("qualification transitions must advance exactly one registered state")
    return requested


def reject_sealed_capability(paths: list[Path]) -> None:
    for path in paths:
        tokens = {part.lower() for part in path.parts}
        if any(token in part for part in tokens for token in SEALED_TOKENS):
            raise ValueError(f"V2 capability cannot reference sealed-test path: {path}")


def write_contract(path: Path, contract: V2ExperimentContract) -> Path:
    if path.exists():
        raise FileExistsError(f"experiment contract is immutable: {path}")
    reject_sealed_capability([path])
    return write_json(path, contract)


def load_contract(path: Path) -> V2ExperimentContract:
    reject_sealed_capability([path])
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"V2 contract must be a mapping: {path}")
    return V2ExperimentContract.model_validate(payload)


def claim_scientific_attempt(
    contract: V2ExperimentContract,
    *,
    optimizer_step: int,
    attempt_root: Path,
    passing_dependencies: set[str] | None = None,
) -> Path:
    """Claim a scientific attempt immediately before optimizer step one.

    Builds, imports, data binding, device validation, one-step qualification,
    checkpoint tests, and report dry-runs must all happen before this function.
    """
    if contract.qualification_state is not QualificationState.QUALIFIED:
        raise ValueError("scientific attempt requires a fully qualified contract")
    if contract.scientific_state is not ScientificAttemptState.REGISTERED:
        raise ValueError("scientific attempt is no longer in registered state")
    if optimizer_step != 1:
        raise ValueError("attempt marker must be claimed immediately before optimizer step 1")
    if not contract.priced_for_scientific_dispatch:
        raise ValueError("scientific attempt requires a current fixed dollar cap")
    missing_dependencies = set(contract.dependencies).difference(passing_dependencies or set())
    if missing_dependencies:
        raise ValueError(
            "scientific attempt has nonpassing dependencies: "
            + ",".join(sorted(missing_dependencies))
        )
    reject_sealed_capability([attempt_root, contract.immutable_output_root])
    attempt_root.mkdir(parents=True, exist_ok=True)
    marker = attempt_root / f"{contract.run_id}.json"
    payload = {
        "schema_version": "frayid_v2_scientific_attempt.v1",
        "event": "attempt_started",
        "experiment_id": contract.experiment_id,
        "run_id": contract.run_id,
        "source_commit": contract.source_commit,
        "optimizer_step": optimizer_step,
        "compute_cap": contract.compute_cap.model_dump(mode="json"),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return marker
