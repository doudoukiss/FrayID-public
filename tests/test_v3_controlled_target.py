from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from frayid.io import sha256_file, write_json
from frayid.v3.controlled_target import (
    GarmentBoundaryProfile,
    register_controlled_method_case,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "candidate.mkv"
    video.write_bytes(b"measured pixels fixture")
    digest = sha256_file(video)
    manifest = tmp_path / "guided_rotation_manifest.json"
    write_json(
        manifest,
        {
            "status": "incomplete_not_evidence",
            "video_path": str(video),
            "video_sha256": digest,
        },
    )
    audit = tmp_path / "content_audit.json"
    write_json(
        audit,
        {
            "source_video_sha256": digest,
            "audit_role": "post_hoc_visual_proposal_not_promotion_evidence",
            "observed_complete_rotation_proposals_seconds": [[10.0, 70.0], [70.0, 130.0]],
        },
    )
    return manifest, audit


def test_method_case_keeps_people_and_geometry_isolated(tmp_path: Path) -> None:
    manifest, audit = _inputs(tmp_path)
    contract = register_controlled_method_case(
        source_manifest_path=manifest,
        content_audit_path=audit,
        owner_confirmed=True,
    )
    assert contract.source_pixel_role == "measured"
    assert all(interval.role == "proposal" for interval in contract.candidate_intervals)
    assert contract.case_a_and_case_b_are_distinct_people is True
    assert contract.cross_person_pixels_shared is False
    assert contract.cross_person_tracks_shared is False
    assert contract.cross_person_geometry_shared is False
    assert contract.shared_method_code_and_frozen_gates_only is True
    assert contract.boundary_profile.generic_boundary_loops == (
        "neck",
        "left_distal_opening",
        "right_distal_opening",
        "hem",
    )
    assert contract.boundary_profile.left_distal_opening_semantics == "left_sleeve_cuff"
    assert contract.promotion_eligible is False


def test_method_case_requires_explicit_confirmation(tmp_path: Path) -> None:
    manifest, audit = _inputs(tmp_path)
    with pytest.raises(ValueError, match="explicit owner confirmation"):
        register_controlled_method_case(
            source_manifest_path=manifest,
            content_audit_path=audit,
            owner_confirmed=False,
        )


def test_boundary_profile_maps_distal_openings_by_variant() -> None:
    sleeveless = GarmentBoundaryProfile(
        garment_variant="sleeveless",
        left_distal_opening_semantics="left_armhole",
        right_distal_opening_semantics="right_armhole",
    )
    assert sleeveless.boundary_loop_count == 4
    with pytest.raises(ValidationError, match="semantics must match"):
        GarmentBoundaryProfile(
            garment_variant="short_sleeve",
            left_distal_opening_semantics="left_armhole",
            right_distal_opening_semantics="right_armhole",
        )


def test_method_case_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest, audit = _inputs(tmp_path)
    payload = audit.read_text(encoding="utf-8").replace(
        sha256_file(tmp_path / "candidate.mkv"), "0" * 64
    )
    audit.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="content audit"):
        register_controlled_method_case(
            source_manifest_path=manifest,
            content_audit_path=audit,
            owner_confirmed=True,
        )
