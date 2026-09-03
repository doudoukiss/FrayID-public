from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import scipy  # type: ignore[import-untyped]
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from frayid.io import sha256_file


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckpointRegistration(StrictModel):
    repository: str
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    url: str


class TrackerSourceRegistration(StrictModel):
    source: Literal["lk", "tapir", "cotracker3"]
    implementation: str
    source_url: str
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_descriptor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: Literal["Apache-2.0", "CC-BY-NC-4.0"]
    license_url: str
    license_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint: CheckpointRegistration | None
    checkpoint_path: Path | None
    real_use_policy: Literal["allowed", "conditional_noncommercial_owner_confirmation"]
    owner_authorization_recorded: bool
    owner_authorization_record: str | None = None

    @model_validator(mode="after")
    def _checkpoint_fields_match(self) -> TrackerSourceRegistration:
        if (self.checkpoint is None) != (self.checkpoint_path is None):
            raise ValueError("checkpoint and checkpoint_path must be set together")
        if self.real_use_policy != "allowed":
            if self.owner_authorization_recorded and not self.owner_authorization_record:
                raise ValueError("conditional tracker authorization needs a decision record")
            if (
                not self.owner_authorization_recorded
                and self.owner_authorization_record is not None
            ):
                raise ValueError("unauthorized tracker cannot cite an authorization record")
        return self


class TrackerSourceRegistry(StrictModel):
    schema_version: Literal["frayid_v3_q04_tracker_source_registry.v1"]
    checked_at: str
    sources: list[TrackerSourceRegistration]

    @model_validator(mode="after")
    def _all_sources_are_unique(self) -> TrackerSourceRegistry:
        names = [item.source for item in self.sources]
        if set(names) != {"lk", "tapir", "cotracker3"} or len(names) != 3:
            raise ValueError("registry must contain exactly LK, TAPIR, and CoTracker3")
        return self


def load_tracker_registry(path: Path) -> TrackerSourceRegistry:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tracker registry must be a YAML mapping")
    return TrackerSourceRegistry.model_validate(payload)


def _runtime_binary() -> tuple[Path, str]:
    package_root = Path(cv2.__file__).resolve().parent
    binaries = sorted(package_root.glob("cv2*.so")) + sorted(package_root.glob("cv2*.pyd"))
    if len(binaries) != 1:
        raise RuntimeError(f"expected exactly one OpenCV extension binary, found {len(binaries)}")
    return binaries[0], sha256_file(binaries[0])


def audit_tracker_sources(registry_path: Path) -> dict[str, object]:
    """Audit pinned provenance and local readiness without importing learned trackers."""
    registry = load_tracker_registry(registry_path)
    binary_path, binary_sha256 = _runtime_binary()
    source_reports: list[dict[str, object]] = []
    blockers: list[str] = []
    for source in registry.sources:
        checkpoint_status = "not_applicable"
        checkpoint_observed_sha256: str | None = None
        if source.checkpoint is not None and source.checkpoint_path is not None:
            if not source.checkpoint_path.is_file():
                checkpoint_status = "missing"
                blockers.append(f"checkpoint_missing:{source.source}")
            else:
                checkpoint_observed_sha256 = sha256_file(source.checkpoint_path)
                size_matches = source.checkpoint_path.stat().st_size == source.checkpoint.size_bytes
                hash_matches = checkpoint_observed_sha256 == source.checkpoint.sha256
                checkpoint_status = "verified" if size_matches and hash_matches else "mismatch"
                if checkpoint_status != "verified":
                    blockers.append(f"checkpoint_integrity:{source.source}")
        license_ready = source.real_use_policy == "allowed" or source.owner_authorization_recorded
        if not license_ready:
            blockers.append(f"owner_license_confirmation_required:{source.source}")
        source_reports.append(
            {
                "source": source.source,
                "implementation": source.implementation,
                "source_url": source.source_url,
                "source_revision": source.source_revision,
                "source_descriptor_sha256": source.source_descriptor_sha256,
                "license": source.license,
                "license_url": source.license_url,
                "license_sha256": source.license_sha256,
                "real_use_policy": source.real_use_policy,
                "owner_authorization_recorded": source.owner_authorization_recorded,
                "owner_authorization_record": source.owner_authorization_record,
                "license_ready_for_real_use": license_ready,
                "checkpoint": (
                    source.checkpoint.model_dump(mode="json")
                    if source.checkpoint is not None
                    else None
                ),
                "checkpoint_path": (
                    str(source.checkpoint_path) if source.checkpoint_path is not None else None
                ),
                "checkpoint_status": checkpoint_status,
                "checkpoint_observed_sha256": checkpoint_observed_sha256,
                "weights_imported": False,
                "weights_executed": False,
                "material_truth_write_access": False,
            }
        )
    registry_digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    return {
        "schema_version": "frayid_v3_q04_tracker_source_audit.v1",
        "experiment_id": "postv3_q04_local_material_chart_graph_r01",
        "status": "pass",
        "metadata_audit_passed": True,
        "real_execution_ready": not blockers,
        "registry_path": str(registry_path),
        "registry_sha256": registry_digest,
        "checked_at": registry.checked_at,
        "sources": source_reports,
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "opencv_binary_path": str(binary_path),
            "opencv_binary_sha256": binary_sha256,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "blockers": sorted(blockers),
        "project_evidence_reads": 0,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "paid_jobs_launched": 0,
    }
