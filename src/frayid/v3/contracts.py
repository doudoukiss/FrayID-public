from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from frayid.io import write_json
from frayid.v2.contracts import (
    QualificationState,
    ScientificAttemptState,
    advance_qualification,
    reject_sealed_capability,
)

V3_ID = re.compile(r"^postv3_[a-z][a-z0-9_]*_r[0-9]{2}$")
DEPENDENCY_ID = re.compile(r"^postv[23]_[a-z][a-z0-9_]*_r[0-9]{2}$")


class EvidenceRole(StrEnum):
    MEASURED = "measured"
    PROPOSAL = "proposal"
    PRIOR = "prior"
    EVALUATOR_ONLY = "evaluator_only"
    FORBIDDEN = "forbidden"


class EvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    role: EvidenceRole
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def _forbidden_path_is_declared(self) -> EvidenceInput:
        lowered = self.locator.lower()
        if (
            "sealed" in lowered or "test_v1" in lowered
        ) and self.role is not EvidenceRole.FORBIDDEN:
            raise ValueError("sealed evidence must be declared forbidden")
        return self


class V3ComputeCap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qualification_gpu_hours: float = Field(ge=0.0)
    scientific_gpu_hours: float = Field(ge=0.0)
    wall_time_seconds: int = Field(gt=0)
    estimated_base_cost_usd: float | None = Field(default=None, ge=0.0)
    maximum_cost_usd: float | None = Field(default=None, ge=0.0)
    price_rate_checked_at: str | None = None
    contingency_fraction: float = Field(default=0.2, ge=0.2, le=0.2)
    automatic_retries: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_pricing(self) -> V3ComputeCap:
        priced = (
            self.estimated_base_cost_usd,
            self.maximum_cost_usd,
            self.price_rate_checked_at,
        )
        if any(value is not None for value in priced) and any(value is None for value in priced):
            raise ValueError("base cost, dollar cap, and price timestamp must be set together")
        if self.maximum_cost_usd is not None and self.estimated_base_cost_usd is not None:
            minimum_cap = self.estimated_base_cost_usd * (1.0 + self.contingency_fraction)
            if self.maximum_cost_usd + 1e-9 < minimum_cap:
                raise ValueError("dollar cap must include the fixed 20% contingency")
        if self.price_rate_checked_at is not None:
            checked = datetime.fromisoformat(self.price_rate_checked_at.replace("Z", "+00:00"))
            if checked.tzinfo is None:
                raise ValueError("price timestamp must include a timezone")
        return self


class V3ExperimentContract(BaseModel):
    """Immutable V3 registration; V2 models and records remain untouched."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["frayid_v3_experiment_contract.v1"] = "frayid_v3_experiment_contract.v1"
    experiment_id: str
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*-r[0-9]{2}$")
    hypothesis: str = Field(min_length=1)
    changed_mechanism: str = Field(min_length=1)
    matched_control: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    immutable_output_root: Path
    evidence_inputs: list[EvidenceInput]
    compute_cap: V3ComputeCap
    qualification_state: QualificationState = QualificationState.UNBUILT
    scientific_state: ScientificAttemptState = ScientificAttemptState.REGISTERED
    dependencies: list[str] = Field(default_factory=list)
    promotion_gates: dict[str, bool | int | float | str]
    stop_conditions: list[str]
    historical_records_immutable: Literal[True] = True
    automatic_paid_retries: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_namespace_and_root(self) -> V3ExperimentContract:
        if not V3_ID.fullmatch(self.experiment_id):
            raise ValueError("V3 experiment IDs must match postv3_*_rNN")
        expected = Path("outputs/post_v3") / self.experiment_id / self.run_id
        if self.immutable_output_root != expected:
            raise ValueError(f"immutable output root must be {expected}")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("dependencies must be unique")
        if any(not DEPENDENCY_ID.fullmatch(item) for item in self.dependencies):
            raise ValueError("dependencies must be immutable post-V2 or post-V3 experiment IDs")
        evidence_ids = [item.evidence_id for item in self.evidence_inputs]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence IDs must be unique")
        if not any(item.role is EvidenceRole.FORBIDDEN for item in self.evidence_inputs):
            raise ValueError("the permanently sealed evidence boundary must be explicit")
        return self

    @property
    def priced_for_scientific_dispatch(self) -> bool:
        return (
            self.compute_cap.maximum_cost_usd is not None
            and self.compute_cap.price_rate_checked_at is not None
        )


def load_contract(path: Path) -> V3ExperimentContract:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return V3ExperimentContract.model_validate(payload)


def write_contract(path: Path, contract: V3ExperimentContract) -> Path:
    reject_sealed_capability([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = contract.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def fitting_evidence(contract: V3ExperimentContract) -> list[EvidenceInput]:
    """Return only inputs authorized to influence fitting."""
    return [
        item
        for item in contract.evidence_inputs
        if item.role in {EvidenceRole.MEASURED, EvidenceRole.PROPOSAL, EvidenceRole.PRIOR}
    ]


def reject_nonfitting_evidence(inputs: list[EvidenceInput]) -> None:
    denied = [
        item.evidence_id
        for item in inputs
        if item.role in {EvidenceRole.EVALUATOR_ONLY, EvidenceRole.FORBIDDEN}
    ]
    if denied:
        raise ValueError(f"evaluator-only or forbidden evidence cannot enter fitting: {denied}")
    reject_sealed_capability([Path(item.locator) for item in inputs])


def claim_scientific_attempt(
    contract: V3ExperimentContract,
    *,
    optimizer_step: int,
    attempt_root: Path,
    passing_dependencies: set[str] | None = None,
) -> Path:
    if contract.qualification_state is not QualificationState.QUALIFIED:
        raise ValueError("scientific attempts require a fully qualified V3 contract")
    if contract.scientific_state is not ScientificAttemptState.REGISTERED:
        raise ValueError("scientific attempt is not in registered state")
    if optimizer_step < 1:
        raise ValueError("attempt marker can only be claimed at the first optimizer step")
    if not contract.priced_for_scientific_dispatch:
        raise ValueError("scientific dispatch requires a current price timestamp and dollar cap")
    checked_raw = contract.compute_cap.price_rate_checked_at
    if checked_raw is None:
        raise ValueError("scientific dispatch requires a current price timestamp")
    checked_at = datetime.fromisoformat(checked_raw.replace("Z", "+00:00"))
    age = datetime.now(UTC) - checked_at
    if age > timedelta(hours=24) or age < -timedelta(minutes=5):
        raise ValueError("scientific dispatch price timestamp is not current")
    missing = set(contract.dependencies) - (passing_dependencies or set())
    if missing:
        raise ValueError(f"nonpassing dependencies: {sorted(missing)}")
    reject_sealed_capability([attempt_root])
    marker = attempt_root / "SCIENTIFIC_ATTEMPT.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "frayid_v3_scientific_attempt.v1",
        "experiment_id": contract.experiment_id,
        "run_id": contract.run_id,
        "optimizer_step": optimizer_step,
        "claimed_at": datetime.now(UTC).isoformat(),
        "automatic_retries": 0,
        "maximum_cost_usd": contract.compute_cap.maximum_cost_usd,
        "wall_time_seconds": contract.compute_cap.wall_time_seconds,
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise FileExistsError("the exclusive V3 scientific attempt is already claimed") from exc
    return marker


def write_immutable_json(
    root: Path, relative_path: Path, payload: BaseModel | dict[str, Any]
) -> Path:
    """Write once beneath an explicit V3 immutable run root."""
    root_parts = root.parts
    if len(root_parts) < 4 or root_parts[:2] != ("outputs", "post_v3"):
        raise ValueError("V3 writes require an outputs/post_v3/<experiment>/<run> root")
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("relative output path cannot escape its immutable root")
    destination = root / relative_path
    reject_sealed_capability([destination])
    if destination.exists():
        raise FileExistsError(f"immutable V3 output already exists: {destination}")
    return write_json(destination, payload)


__all__ = [
    "EvidenceInput",
    "EvidenceRole",
    "QualificationState",
    "ScientificAttemptState",
    "V3ComputeCap",
    "V3ExperimentContract",
    "advance_qualification",
    "claim_scientific_attempt",
    "fitting_evidence",
    "load_contract",
    "reject_nonfitting_evidence",
    "write_contract",
    "write_immutable_json",
]
