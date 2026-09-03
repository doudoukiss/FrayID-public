from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from frayid.io import sha256_file, write_json
from frayid.schemas import (
    AssetCheck,
    AssetVerificationReport,
    LocalAsset,
    LocalAssetManifest,
    VideoMetadata,
)


def probe_video(path: Path, *, ffprobe_bin: str = "ffprobe") -> VideoMetadata:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames,duration:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {path}")
    stream = streams[0]
    format_payload = payload.get("format", {})
    frame_rate = _parse_fraction(str(stream["r_frame_rate"]))
    duration = float(stream.get("duration") or format_payload.get("duration") or 0)
    frame_count_raw = stream.get("nb_frames")
    frame_count = (
        int(frame_count_raw)
        if frame_count_raw not in (None, "N/A")
        else round(duration * frame_rate)
    )
    return VideoMetadata(
        path=str(path),
        codec=str(stream["codec_name"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_count=frame_count,
        frame_rate=frame_rate,
        duration_seconds=duration,
        size_bytes=int(format_payload.get("size") or path.stat().st_size),
    )


def load_asset_manifest(path: Path) -> LocalAssetManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return LocalAssetManifest.model_validate(payload)


def verify_assets(
    manifest_path: Path,
    *,
    output_path: Path | None = None,
) -> AssetVerificationReport:
    manifest = load_asset_manifest(manifest_path)
    checks = [_verify_asset(asset) for asset in manifest.assets]
    report = AssetVerificationReport(
        status="ready" if all(_asset_ready(check) for check in checks) else "blocked",
        privacy=manifest.privacy,
        checks=checks,
    )
    if output_path is not None:
        write_json(output_path, report)
    return report


def _verify_asset(asset: LocalAsset) -> AssetCheck:
    path = asset.path
    if not path.exists():
        return AssetCheck(
            asset_id=asset.asset_id,
            path=str(path),
            exists=False,
            sha256_matches=False,
            metadata_matches=False,
            errors=["missing_local_asset"],
        )
    errors: list[str] = []
    observed_sha256 = sha256_file(path)
    if observed_sha256 != asset.sha256:
        errors.append("sha256_mismatch")
    try:
        metadata = probe_video(path)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        return AssetCheck(
            asset_id=asset.asset_id,
            path=str(path),
            exists=True,
            sha256_matches=observed_sha256 == asset.sha256,
            metadata_matches=False,
            observed_sha256=observed_sha256,
            errors=[*errors, f"video_probe_failed:{type(exc).__name__}"],
        )
    metadata_errors = _metadata_errors(asset, metadata)
    errors.extend(metadata_errors)
    if asset.tracked_in_git:
        if not _path_is_git_tracked(path):
            errors.append("authorized_media_not_git_tracked")
        if not _path_uses_git_lfs(path):
            errors.append("authorized_media_not_git_lfs")
    elif not _path_is_git_ignored(path):
        errors.append("local_media_not_git_ignored")
    return AssetCheck(
        asset_id=asset.asset_id,
        path=str(path),
        exists=True,
        sha256_matches=observed_sha256 == asset.sha256,
        metadata_matches=not metadata_errors,
        observed_sha256=observed_sha256,
        metadata=metadata,
        errors=errors,
    )


def _metadata_errors(asset: LocalAsset, observed: VideoMetadata) -> list[str]:
    expected = asset.expected
    errors: list[str] = []
    for field_name in ("codec", "width", "height", "frame_count"):
        if getattr(observed, field_name) != getattr(expected, field_name):
            errors.append(f"metadata_mismatch:{field_name}")
    if abs(observed.duration_seconds - expected.duration_seconds) > 0.05:
        errors.append("metadata_mismatch:duration_seconds")
    return errors


def _path_is_git_ignored(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _path_is_git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _path_uses_git_lfs(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-attr", "filter", "--", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.rstrip().endswith(": lfs")


def _asset_ready(check: AssetCheck) -> bool:
    return check.exists and check.sha256_matches and check.metadata_matches and not check.errors


def _parse_fraction(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    denominator_value = float(denominator)
    if denominator_value == 0:
        raise ValueError(f"Invalid frame rate: {value}")
    return float(numerator) / denominator_value


def manifest_summary(manifest_path: Path) -> dict[str, Any]:
    manifest = load_asset_manifest(manifest_path)
    return {
        "schema_version": manifest.schema_version,
        "privacy": manifest.privacy,
        "asset_ids": [asset.asset_id for asset in manifest.assets],
    }
