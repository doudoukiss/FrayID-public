from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{7,127}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40}$")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def image_definition_sha256(definition: dict[str, Any]) -> str:
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RunBinding:
    """Complete immutable metadata bound before a Modal workload reads inputs."""

    schema_version: str
    run_id: str
    git_commit: str
    config_sha256: str
    input_hashes: dict[str, str]
    image_name: Literal["frayid-cpu-exact", "frayid-cuda"]
    image_definition_sha256: str
    resource_type: str
    random_seed: int

    def validate(
        self,
        *,
        expected_image_definition_sha256: str | None = None,
        expected_resource_type: str | None = None,
    ) -> None:
        if self.schema_version != "frayid_modal_run_binding.v1":
            raise ValueError("unsupported Modal run-binding schema")
        if RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a stable lowercase immutable identifier")
        if GIT_COMMIT_PATTERN.fullmatch(self.git_commit) is None:
            raise ValueError("git_commit must be a full 40-character commit")
        for name, digest in {"config": self.config_sha256, **self.input_hashes}.items():
            if not name or SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"invalid SHA-256 binding: {name}")
        if SHA256_PATTERN.fullmatch(self.image_definition_sha256) is None:
            raise ValueError("invalid image-definition SHA-256")
        if self.random_seed < 0:
            raise ValueError("random seed must be nonnegative")
        if expected_image_definition_sha256 is not None and (
            self.image_definition_sha256 != expected_image_definition_sha256
        ):
            raise ValueError("Modal image definition does not match the run binding")
        if expected_resource_type is not None and self.resource_type != expected_resource_type:
            raise ValueError("Modal resource type does not match the run binding")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> RunBinding:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("Modal run binding must be a JSON object")
        binding = cls(
            schema_version=str(payload.get("schema_version", "")),
            run_id=str(payload.get("run_id", "")),
            git_commit=str(payload.get("git_commit", "")),
            config_sha256=str(payload.get("config_sha256", "")),
            input_hashes={
                str(name): str(digest)
                for name, digest in dict(payload.get("input_hashes", {})).items()
            },
            image_name=str(payload.get("image_name", "")),  # type: ignore[arg-type]
            image_definition_sha256=str(payload.get("image_definition_sha256", "")),
            resource_type=str(payload.get("resource_type", "")),
            random_seed=int(payload.get("random_seed", -1)),
        )
        binding.validate()
        return binding


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        try:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise


def claim_attempt(
    binding: RunBinding,
    *,
    attempt_root: Path,
    output_root: Path,
) -> Path:
    """Atomically claim one run; any reschedule sees the directory and refuses."""
    binding.validate()
    attempt_directory = attempt_root / binding.run_id
    run_output = output_root / binding.run_id
    attempt_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    if attempt_directory.exists():
        raise FileExistsError(
            f"Modal attempt already claimed; refusing automatic continuation: {binding.run_id}"
        )
    if run_output.exists():
        raise FileExistsError(f"immutable Modal output already exists: {binding.run_id}")
    try:
        attempt_directory.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"Modal attempt already claimed; refusing automatic continuation: {binding.run_id}"
        ) from error
    _exclusive_json(
        attempt_directory / "claimed.json",
        {
            "schema_version": "frayid_modal_attempt_event.v1",
            "event": "claimed",
            "binding": asdict(binding),
        },
    )
    run_output.mkdir()
    return run_output


def write_attempt_event(
    binding: RunBinding,
    *,
    attempt_root: Path,
    event_name: Literal["completed", "failed"],
    payload: dict[str, Any],
) -> Path:
    binding.validate()
    attempt_directory = attempt_root / binding.run_id
    claimed = attempt_directory / "claimed.json"
    if not claimed.is_file():
        raise FileNotFoundError("Modal attempt has not been claimed")
    destination = attempt_directory / f"{event_name}.json"
    _exclusive_json(
        destination,
        {
            "schema_version": "frayid_modal_attempt_event.v1",
            "event": event_name,
            "run_id": binding.run_id,
            "payload": payload,
        },
    )
    return destination
