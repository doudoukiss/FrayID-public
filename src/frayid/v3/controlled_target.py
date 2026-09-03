from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from frayid.io import read_json, sha256_file
from frayid.v2.contracts import reject_sealed_capability

EXPERIMENT_ID: Literal["postv3_m01_instance_isolated_upper_garment_method_r01"] = (
    "postv3_m01_instance_isolated_upper_garment_method_r01"
)
GENERIC_BOUNDARY_LOOPS: tuple[
    Literal["neck"],
    Literal["left_distal_opening"],
    Literal["right_distal_opening"],
    Literal["hem"],
] = (
    "neck",
    "left_distal_opening",
    "right_distal_opening",
    "hem",
)


class StrictTargetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateRotationInterval(StrictTargetModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    role: Literal["proposal"] = "proposal"

    @model_validator(mode="after")
    def _ordered(self) -> CandidateRotationInterval:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("candidate rotation interval must have positive duration")
        return self


class GarmentBoundaryProfile(StrictTargetModel):
    garment_variant: Literal["sleeveless", "short_sleeve"]
    generic_boundary_loops: tuple[
        Literal["neck"],
        Literal["left_distal_opening"],
        Literal["right_distal_opening"],
        Literal["hem"],
    ] = GENERIC_BOUNDARY_LOOPS
    left_distal_opening_semantics: Literal["left_armhole", "left_sleeve_cuff"]
    right_distal_opening_semantics: Literal["right_armhole", "right_sleeve_cuff"]
    connected_components: Literal[1] = 1
    genus: Literal[0] = 0
    boundary_loop_count: Literal[4] = 4
    euler_number: Literal[-2] = -2

    @model_validator(mode="after")
    def _semantics_match_variant(self) -> GarmentBoundaryProfile:
        expected = (
            ("left_armhole", "right_armhole")
            if self.garment_variant == "sleeveless"
            else ("left_sleeve_cuff", "right_sleeve_cuff")
        )
        observed = (
            self.left_distal_opening_semantics,
            self.right_distal_opening_semantics,
        )
        if observed != expected:
            raise ValueError("distal-opening semantics must match the garment variant")
        if tuple(self.generic_boundary_loops) != GENERIC_BOUNDARY_LOOPS:
            raise ValueError("upper-garment method requires exactly four generic boundary loops")
        return self


class ControlledMethodCaseContract(StrictTargetModel):
    schema_version: Literal["frayid_v3_controlled_method_case.v1"] = (
        "frayid_v3_controlled_method_case.v1"
    )
    experiment_id: Literal["postv3_m01_instance_isolated_upper_garment_method_r01"] = EXPERIMENT_ID
    case_id: Literal["controlled_case_b_short_sleeve_r01"] = "controlled_case_b_short_sleeve_r01"
    owner_confirmed_method_case: Literal[True]
    owner_confirmation_date: Literal["2026-09-03"] = "2026-09-03"
    purpose: Literal["method_development_case_not_cross_instance_geometry"] = (
        "method_development_case_not_cross_instance_geometry"
    )
    source_video_path: str
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_path: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_audit_path: str
    content_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_pixel_role: Literal["measured"] = "measured"
    scheduled_angle_role: Literal["proposal"] = "proposal"
    candidate_intervals: list[CandidateRotationInterval]
    candidate_interval_role: Literal["proposal"] = "proposal"
    boundary_profile: GarmentBoundaryProfile
    case_a_and_case_b_are_distinct_people: Literal[True] = True
    cross_person_pixels_shared: Literal[False] = False
    cross_person_tracks_shared: Literal[False] = False
    cross_person_geometry_shared: Literal[False] = False
    shared_method_code_and_frozen_gates_only: Literal[True] = True
    source_manifest_status: Literal["incomplete_not_evidence"] = "incomplete_not_evidence"
    promotion_eligible: Literal[False] = False
    blockers: list[
        Literal[
            "candidate_intervals_require_deterministic_source_selection_audit",
            "counterclockwise_source_missing",
            "camera_intrinsics_not_yet_qualified",
        ]
    ]
    claim_ceiling: Literal["evidence-consistent MANTLE reconstruction"] = (
        "evidence-consistent MANTLE reconstruction"
    )
    broad_generalization_claim_allowed: Literal[False] = False
    measurements_sizing_tailoring_claims_allowed: Literal[False] = False
    sealed_test_accesses: Literal[0] = 0
    evaluator_files_read: Literal[0] = 0

    @model_validator(mode="after")
    def _case_is_isolated(self) -> ControlledMethodCaseContract:
        if len(self.candidate_intervals) < 1:
            raise ValueError("controlled method case requires a proposed rotation interval")
        ordered = sorted(self.candidate_intervals, key=lambda item: item.start_seconds)
        if ordered != self.candidate_intervals:
            raise ValueError("candidate rotation intervals must be time ordered")
        if any(right.start_seconds < left.end_seconds for left, right in pairwise(ordered)):
            raise ValueError("candidate rotation intervals cannot overlap")
        expected_blockers = {
            "candidate_intervals_require_deterministic_source_selection_audit",
            "counterclockwise_source_missing",
            "camera_intrinsics_not_yet_qualified",
        }
        if set(self.blockers) != expected_blockers:
            raise ValueError("unqualified controlled case must retain every frozen blocker")
        return self


def register_controlled_method_case(
    *,
    source_manifest_path: Path,
    content_audit_path: Path,
    owner_confirmed: bool,
) -> ControlledMethodCaseContract:
    """Bind one method-development case while forbidding cross-person evidence mixing."""
    if not owner_confirmed:
        raise ValueError("method-case registration requires explicit owner confirmation")
    reject_sealed_capability([source_manifest_path, content_audit_path])
    manifest = read_json(source_manifest_path)
    audit = read_json(content_audit_path)
    if manifest.get("status") != "incomplete_not_evidence":
        raise ValueError("method-case registration expects the preserved incomplete manifest")
    video_path_value = manifest.get("video_path")
    if not isinstance(video_path_value, str):
        raise ValueError("source manifest must name its captured video")
    video_path = Path(video_path_value)
    reject_sealed_capability([video_path])
    if not video_path.is_file():
        raise FileNotFoundError(f"controlled method-case video is missing: {video_path}")
    observed_video_sha = sha256_file(video_path)
    if manifest.get("video_sha256") != observed_video_sha:
        raise ValueError("method-case video hash does not match its source manifest")
    if audit.get("source_video_sha256") != observed_video_sha:
        raise ValueError("method-case video hash does not match its content audit")
    if audit.get("audit_role") != "post_hoc_visual_proposal_not_promotion_evidence":
        raise ValueError("content audit must retain its proposal-only role")
    raw_intervals = audit.get("observed_complete_rotation_proposals_seconds")
    if not isinstance(raw_intervals, list):
        raise ValueError("content audit must provide proposed rotation intervals")
    intervals = [
        CandidateRotationInterval(start_seconds=float(value[0]), end_seconds=float(value[1]))
        for value in raw_intervals
        if isinstance(value, list) and len(value) == 2
    ]
    if len(intervals) != len(raw_intervals):
        raise ValueError("every proposed interval must contain exactly two timestamps")
    return ControlledMethodCaseContract(
        owner_confirmed_method_case=True,
        source_video_path=str(video_path),
        source_video_sha256=observed_video_sha,
        source_manifest_path=str(source_manifest_path),
        source_manifest_sha256=sha256_file(source_manifest_path),
        content_audit_path=str(content_audit_path),
        content_audit_sha256=sha256_file(content_audit_path),
        candidate_intervals=intervals,
        boundary_profile=GarmentBoundaryProfile(
            garment_variant="short_sleeve",
            left_distal_opening_semantics="left_sleeve_cuff",
            right_distal_opening_semantics="right_sleeve_cuff",
        ),
        blockers=[
            "candidate_intervals_require_deterministic_source_selection_audit",
            "counterclockwise_source_missing",
            "camera_intrinsics_not_yet_qualified",
        ],
    )


__all__ = [
    "GENERIC_BOUNDARY_LOOPS",
    "CandidateRotationInterval",
    "ControlledMethodCaseContract",
    "GarmentBoundaryProfile",
    "register_controlled_method_case",
]
