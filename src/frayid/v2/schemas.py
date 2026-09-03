from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TurntableSolution(BaseModel):
    schema_version: Literal["frayid_v2_turntable_solution.v1"] = "frayid_v2_turntable_solution.v1"
    status: Literal["qualification_candidate", "pass", "fail"]
    shared_intrinsics: list[list[float]]
    axis: list[float]
    center: list[float]
    angles_radians: list[float]
    residual_twists: list[list[float]]
    micromotion_basis: list[list[float]]
    micromotion_codes: list[list[float]]
    source_frame_indices: list[int]
    gauge_policy: dict[str, str]
    uncertainty: dict[str, float]
    source_provenance: dict[str, str]

    @model_validator(mode="after")
    def validate_shapes(self) -> TurntableSolution:
        count = len(self.source_frame_indices)
        if len(self.shared_intrinsics) != 3 or any(len(row) != 3 for row in self.shared_intrinsics):
            raise ValueError("turntable intrinsics must be 3x3")
        if len(self.axis) != 3 or len(self.center) != 3:
            raise ValueError("turntable axis and center must each contain three values")
        if not math.isclose(sum(value * value for value in self.axis), 1.0, abs_tol=1.0e-5):
            raise ValueError("turntable axis must be unit length")
        if self.shared_intrinsics[0][0] <= 0 or self.shared_intrinsics[1][1] <= 0:
            raise ValueError("turntable focal lengths must be positive")
        if len(self.angles_radians) != count or len(self.residual_twists) != count:
            raise ValueError("turntable frame arrays must align")
        if len(self.micromotion_codes) != count:
            raise ValueError("turntable micromotion codes must align with frames")
        rank = len(self.micromotion_basis)
        if any(len(code) != rank for code in self.micromotion_codes):
            raise ValueError("turntable micromotion codes must align with the basis rank")
        if rank and len({len(vector) for vector in self.micromotion_basis}) != 1:
            raise ValueError("turntable micromotion basis rows must have equal dimension")
        if self.source_frame_indices != sorted(set(self.source_frame_indices)):
            raise ValueError("turntable source frames must be unique and increasing")
        if any(len(value) != 6 for value in self.residual_twists):
            raise ValueError("each residual twist must contain six values")
        if any(
            following < previous
            for previous, following in zip(
                self.angles_radians, self.angles_radians[1:], strict=False
            )
        ):
            raise ValueError("turntable angles must be monotonic")
        return self


class DynamicCameraFrame(BaseModel):
    source_frame_index: int = Field(ge=0)
    global_orient: list[float] = Field(min_length=3, max_length=3)
    translation: list[float] = Field(min_length=3, max_length=3)
    rotation_inconsistency_degrees: float = Field(ge=0)
    translation_inconsistency_metres: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    provenance: Literal["frozen_camerahmr_initialization"] = "frozen_camerahmr_initialization"


class DynamicCameraSolution(BaseModel):
    schema_version: Literal["frayid_v2_dynamic_camera_solution.v1"] = (
        "frayid_v2_dynamic_camera_solution.v1"
    )
    status: Literal["qualification_candidate", "pass", "fail"]
    shared_intrinsics: list[list[float]]
    frames: list[DynamicCameraFrame]
    gauge_policy: dict[str, str]
    uncertainty_policy: dict[str, float | str]
    source_provenance: dict[str, str]

    @model_validator(mode="after")
    def validate_camera_sequence(self) -> DynamicCameraSolution:
        if len(self.shared_intrinsics) != 3 or any(len(row) != 3 for row in self.shared_intrinsics):
            raise ValueError("dynamic-camera intrinsics must be 3x3")
        if self.shared_intrinsics[0][0] <= 0 or self.shared_intrinsics[1][1] <= 0:
            raise ValueError("dynamic-camera focal lengths must be positive")
        sources = [frame.source_frame_index for frame in self.frames]
        if not sources or sources != sorted(set(sources)):
            raise ValueError("dynamic-camera source frames must be unique and increasing")
        return self


class EvidenceVolumeMetadata(BaseModel):
    schema_version: Literal["frayid_v2_evidence_volume.v1"] = "frayid_v2_evidence_volume.v1"
    resolution: int = Field(ge=3)
    extent: float = Field(gt=0)
    aggregation: Literal["weighted_quantile", "trimmed_weighted_quantile", "trimmed_mean", "cvar"]
    training_view_count: int = Field(gt=0)
    minimum_view_support: int = Field(gt=0)
    semantic_layer_ids: dict[str, list[int]] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    prior_is_separate: Literal[True] = True
    cleanup_operations: list[str] = Field(default_factory=list, max_length=0)


class LayerTopologyPolicy(BaseModel):
    layer_id: str
    role: Literal["body", "upper_clothing", "lower_clothing", "hair", "footwear", "other"]
    closed: bool
    required_component_count: int | None = Field(default=None, gt=0)
    required_boundary_loop_count: int | None = Field(default=None, ge=0)
    required_euler_number: int | None = None
    allow_registered_contact: bool = True


class LayerArtifact(BaseModel):
    layer_id: str
    field_kind: Literal["sdf", "clipped_implicit", "udf", "template_surface"]
    topology_policy: LayerTopologyPolicy
    field_checkpoint: str
    raw_surface_path: str
    boundary_curve_path: str | None = None
    deformation_checkpoint: str
    provenance_path: str
    observed_support_fraction: float = Field(ge=0, le=1)
    prior_support_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_support(self) -> LayerArtifact:
        if self.observed_support_fraction + self.prior_support_fraction > 1.000001:
            raise ValueError("layer support fractions cannot exceed one")
        if self.topology_policy.role != "body":
            if self.topology_policy.closed:
                raise ValueError("exterior surface layers must use an open topology policy")
            if self.boundary_curve_path is None:
                raise ValueError("exterior surface layers require an explicit boundary curve")
        return self


class LayeredCanonicalArtifact(BaseModel):
    schema_version: Literal["frayid_v2_layered_canonical_artifact.v1"] = (
        "frayid_v2_layered_canonical_artifact.v1"
    )
    status: Literal["candidate", "promoted", "failed"]
    body: LayerArtifact
    surface_layers: list[LayerArtifact]
    camera_solution_path: str
    turntable_solution_path: str | None = None
    posed_sequence_path: str
    topology_certificate_paths: dict[str, str]
    visibility_ownership_path: str
    contact_order_report_path: str
    derived_compatibility_export: str | None = None
    compatibility_export_is_authoritative: Literal[False] = False
    sealed_test_accesses: Literal[0] = 0

    @model_validator(mode="after")
    def validate_body(self) -> LayeredCanonicalArtifact:
        if (
            self.body.topology_policy.role != "body"
            or not self.body.topology_policy.closed
            or self.body.field_kind != "sdf"
        ):
            raise ValueError("authoritative layered artifact requires one closed body layer")
        ids = [self.body.layer_id, *(layer.layer_id for layer in self.surface_layers)]
        if len(ids) != len(set(ids)):
            raise ValueError("layer identifiers must be unique")
        return self


class TopologyCertificate(BaseModel):
    schema_version: Literal["frayid_v2_topology_certificate.v1"] = (
        "frayid_v2_topology_certificate.v1"
    )
    layer_id: str
    stage: Literal["search", "commit", "refine"]
    status: Literal["nonpromotable", "pass", "fail", "blocked"]
    vertex_count: int = Field(ge=0)
    face_count: int = Field(ge=0)
    component_count: int = Field(ge=0)
    boundary_loop_count: int = Field(ge=0)
    euler_number: int
    watertight: bool
    winding_consistent: bool
    outward: bool
    exact_intersection_pair_count: int | None = Field(default=None, ge=0)
    registered_penetration_count: int | None = Field(default=None, ge=0)
    connectivity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    surface_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    replay_exact: bool
    blockers: list[str] = Field(default_factory=list)


class V2EvaluationReport(BaseModel):
    schema_version: Literal["frayid_v2_evaluation_report.v1"] = "frayid_v2_evaluation_report.v1"
    status: Literal["pass", "fail", "blocked"]
    experiment_id: str
    run_id: str
    historical_image_metrics: dict[str, float]
    geometry_metrics: dict[str, float]
    layer_metrics: dict[str, float]
    topology_certificates: dict[str, str]
    capacity_stress_metrics: dict[str, float]
    provenance_coverage: dict[str, float]
    replay_exact: bool
    hidden_cleanup_operations: Literal[0] = 0
    sealed_test_accesses: Literal[0] = 0
    blockers: list[str] = Field(default_factory=list)
