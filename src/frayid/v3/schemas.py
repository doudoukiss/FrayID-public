from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrackSourceAudit(StrictModel):
    source: Literal["lk", "tapir", "cotracker3"]
    source_revision: str
    license: str
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime: str
    role: Literal["proposal"] = "proposal"
    weights_executed: bool
    real_use_authorized: bool

    @model_validator(mode="after")
    def _executed_weight_is_hashed(self) -> TrackSourceAudit:
        if self.weights_executed and self.checkpoint_sha256 is None:
            raise ValueError("every executed tracker checkpoint must be hashed")
        return self


class ChartObservation(StrictModel):
    frame_index: int = Field(ge=0)
    source_frame_index: int | None = Field(default=None, ge=0)
    xy: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    visible: bool
    proposal_sources: list[Literal["lk", "tapir", "cotracker3"]]

    @model_validator(mode="after")
    def _covariance_is_spd(self) -> ChartObservation:
        a, b = self.covariance[0]
        c, d = self.covariance[1]
        if abs(b - c) > 1e-8 or a <= 0.0 or d <= 0.0 or a * d - b * c <= 0.0:
            raise ValueError("observation covariance must be symmetric positive definite")
        return self


class MaterialTrack(StrictModel):
    track_id: str
    chart_id: str
    semantic_posterior: dict[str, float]
    visibility_intervals: list[tuple[int, int]]
    observations: list[ChartObservation]
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class ChartTransition(StrictModel):
    source_chart_id: str
    target_chart_id: str
    overlap_track_ids: list[str]
    affine_map: tuple[tuple[float, float, float], tuple[float, float, float]]
    cycle_residual_pixels: float = Field(ge=0.0)


class PublicChartTruthBenchmark(StrictModel):
    schema_version: Literal["frayid_q04_public_truth_benchmark.v1"] = (
        "frayid_q04_public_truth_benchmark.v1"
    )
    baseline_id: Literal["lk_q03_control"] = "lk_q03_control"
    point_count: int = Field(ge=100)
    phase_count: int = Field(ge=10)
    fit_phase_indices: list[int]
    evaluator_phase_indices: list[int]
    control_median_surface_error: float = Field(gt=0.0)
    ensemble_median_surface_error: float = Field(ge=0.0)
    geometry_improvement: float
    control_median_reprojection_error_pixels: float = Field(gt=0.0)
    ensemble_median_reprojection_error_pixels: float = Field(ge=0.0)
    reprojection_improvement: float
    nonradial_surface: bool
    corrupted_source_present: bool
    exact_replay_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_evidence_reads: int = Field(default=0, ge=0)
    sealed_test_accesses: int = Field(default=0, ge=0)
    status: Literal["pass", "fail"]
    blockers: list[str]


class MaterialChartGraph(StrictModel):
    schema_version: Literal["frayid_material_chart_graph.v1"] = "frayid_material_chart_graph.v1"
    experiment_id: str
    evidence_scope: Literal["public_synthetic", "train_real"]
    promotion_eligible: bool
    tracker_audits: list[TrackSourceAudit]
    input_hashes: dict[str, str] = Field(default_factory=dict)
    source_frame_indices: list[int] = Field(default_factory=list)
    proposal_count_by_source: dict[str, int] = Field(default_factory=dict)
    model_output_sha256_by_source: dict[str, str] = Field(default_factory=dict)
    exact_same_device_replay_by_source: dict[str, bool] = Field(default_factory=dict)
    training_records_read: int = Field(default=0, ge=0)
    development_records_read: int = Field(default=0, ge=0)
    sealed_test_accesses: int = Field(default=0, ge=0)
    public_truth_geometry_improvement: float = 0.0
    public_truth_reprojection_improvement: float = 0.0
    public_truth_benchmark: PublicChartTruthBenchmark
    q03_anchor_reprojection_improvement: float | None = None
    charts: list[str]
    tracks: list[MaterialTrack]
    transitions: list[ChartTransition]
    phase_bins_spanned: list[int]
    median_cycle_residual_pixels: float = Field(ge=0.0)
    p95_cycle_residual_pixels: float = Field(ge=0.0)
    median_anchor_reprojection_pixels: float = Field(ge=0.0)
    p95_anchor_reprojection_pixels: float = Field(ge=0.0)
    corrupted_proposal_capacity_regression: float = Field(ge=0.0)
    exact_replay_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pass", "fail", "blocked"]
    blockers: list[str]

    @model_validator(mode="after")
    def _real_graph_is_audited(self) -> MaterialChartGraph:
        if self.evidence_scope == "train_real":
            expected_records = {
                "postv3_q04_local_material_chart_graph_r01": 144,
                "postv3_q05_controlled_material_chart_graph_r01": 72,
            }.get(self.experiment_id)
            if expected_records is None:
                raise ValueError("unsupported real material-chart experiment")
            if self.training_records_read != expected_records:
                raise ValueError(
                    f"real {self.experiment_id} must read exactly "
                    f"{expected_records} training records"
                )
            if self.development_records_read or self.sealed_test_accesses:
                raise ValueError("real material charts cannot read development or sealed evidence")
            if set(self.proposal_count_by_source) != {"lk", "tapir", "cotracker3"}:
                raise ValueError("real Q04 must report all three proposal sources")
            if not all(
                self.exact_same_device_replay_by_source.get(source, False)
                for source in ("tapir", "cotracker3")
            ):
                raise ValueError("real learned-tracker predictions must replay exactly")
        return self


class CameraIntrinsics(StrictModel):
    focal_length_pixels: float = Field(gt=0.0)
    principal_point: tuple[float, float]
    distortion: tuple[float, ...] = ()


class FixedCameraFactorGraphSolution(StrictModel):
    schema_version: Literal["frayid_fixed_camera_factor_graph_solution.v1"] = (
        "frayid_fixed_camera_factor_graph_solution.v1"
    )
    experiment_id: str
    evidence_scope: Literal["public_synthetic", "train_real"]
    promotion_eligible: bool
    physical_camera_extrinsics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    intrinsics: CameraIntrinsics
    global_spin_axis: tuple[float, float, float]
    monotonic_phase_radians: list[float]
    root_translation_residuals_m: list[tuple[float, float, float]]
    pose_residual_norms: list[float]
    profiled_material_anchor_count: int = Field(ge=0)
    jacobian_singular_values: list[float]
    informative_rank_by_fold: list[int]
    maximum_scaled_block_correlation: float = Field(ge=0.0, le=1.0)
    median_reprojection_pixels: float = Field(ge=0.0)
    p95_reprojection_pixels: float = Field(ge=0.0)
    baseline_median_reprojection_pixels: float = Field(ge=0.0)
    restart_phase_spread_degrees: float = Field(ge=0.0)
    restart_root_spread_mm: float = Field(ge=0.0)
    restart_reprojection_spread_pixels: float = Field(ge=0.0)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    next_step_replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pass", "fail", "blocked"]
    blockers: list[str]

    @model_validator(mode="after")
    def _phase_is_monotonic(self) -> FixedCameraFactorGraphSolution:
        if any(
            right <= left
            for left, right in zip(
                self.monotonic_phase_radians,
                self.monotonic_phase_radians[1:],
                strict=False,
            )
        ):
            raise ValueError("phase must be strictly monotonic")
        return self


class BoundaryClass(StrEnum):
    PHYSICAL_BOUNDARY = "physical_boundary"
    SEAM = "seam"
    APPARENT_CONTOUR = "apparent_contour"
    OCCLUSION_BOUNDARY = "occlusion_boundary"
    SEMANTIC_UNCERTAIN = "semantic_uncertain"


class BoundaryCurveHypothesis(StrictModel):
    curve_id: str
    label: BoundaryClass
    garment_loop: Literal["neck", "left_armhole", "right_armhole", "hem"] | None = None
    phase_bins: list[int]
    independent_chart_ids: list[str]
    median_reprojection_pixels: float = Field(ge=0.0)
    alternative_explanation_pixels: float = Field(ge=0.0)
    accepted: bool
    rejection_reasons: list[str]


class BoundaryHypothesisSet(StrictModel):
    schema_version: Literal["frayid_boundary_hypothesis_set.v1"] = (
        "frayid_boundary_hypothesis_set.v1"
    )
    experiment_id: str
    evidence_scope: Literal["public_synthetic", "train_real"]
    promotion_eligible: bool
    garment_hypothesis: Literal["sleeveless_upper_genus0_four_boundaries"]
    curves: list[BoundaryCurveHypothesis]
    promoted_physical_loops: list[Literal["neck", "left_armhole", "right_armhole", "hem"]]
    l03_partition_edges_used_as_contact_truth: Literal[False] = False
    l03_shared_edges_used_as_contact_truth: Literal[False] = False
    status: Literal["pass", "fail", "blocked"]
    blockers: list[str]

    @model_validator(mode="after")
    def _passing_set_has_exact_loops(self) -> BoundaryHypothesisSet:
        expected = {"neck", "left_armhole", "right_armhole", "hem"}
        if self.status == "pass" and set(self.promoted_physical_loops) != expected:
            raise ValueError("passing boundary set requires all four physical loops")
        return self


class RestMetricFace(StrictModel):
    face_index: int = Field(ge=0)
    metric: tuple[tuple[float, float], tuple[float, float]]

    @model_validator(mode="after")
    def _metric_is_spd(self) -> RestMetricFace:
        a, b = self.metric[0]
        c, d = self.metric[1]
        if abs(b - c) > 1e-8 or a <= 0.0 or d <= 0.0 or a * d - b * c <= 0.0:
            raise ValueError("rest metric must be symmetric positive definite")
        return self


class FrameEmbedding(StrictModel):
    frame_index: int = Field(ge=0)
    vertices_path: str
    vertices_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deformation_regime_counts: dict[Literal["attached", "sliding", "free", "contact"], int]


class TopologyCertificateV3(StrictModel):
    connected_components: Literal[1]
    genus: Literal[0]
    boundary_loops: Literal[4]
    euler_number: Literal[-2]
    self_intersections: Literal[0]
    unregistered_body_penetrations: Literal[0]
    collapsed_triangles: Literal[0]
    flipped_triangles: Literal[0]
    winding_consistent: Literal[True]


class UpperGarmentAtlas(StrictModel):
    schema_version: Literal["frayid_upper_garment_atlas.v1"] = "frayid_upper_garment_atlas.v1"
    experiment_id: str
    evidence_scope: Literal["public_synthetic", "train_real"]
    promotion_eligible: bool
    intrinsic_vertices_path: str
    intrinsic_faces_path: str
    intrinsic_domain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    boundary_cycles: dict[Literal["neck", "left_armhole", "right_armhole", "hem"], list[int]]
    seam_hypotheses: list[list[int]]
    rest_metric: list[RestMetricFace]
    frame_embeddings: list[FrameEmbedding]
    clipped_sdf_path: str
    clipped_sdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    body_contact_posterior_path: str
    body_contact_posterior_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uncertainty_support_ledger_path: str
    uncertainty_support_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topology: TopologyCertificateV3
    median_absolute_in_plane_strain: float = Field(ge=0.0)
    p95_absolute_in_plane_strain: float = Field(ge=0.0)
    restart_observed_median_spread_mm: float = Field(ge=0.0)
    restart_observed_p95_spread_mm: float = Field(ge=0.0)
    status: Literal["pass", "fail", "blocked"]
    blockers: list[str]

    @model_validator(mode="after")
    def _boundary_names_are_exact(self) -> UpperGarmentAtlas:
        expected = {"neck", "left_armhole", "right_armhole", "hem"}
        if set(self.boundary_cycles) != expected:
            raise ValueError("upper-garment atlas requires exactly the four registered loops")
        return self


class DerivedSurfaceExport(StrictModel):
    role: Literal["derived_non_authoritative"] = "derived_non_authoritative"
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MantleArtifact(StrictModel):
    schema_version: Literal["frayid_mantle_artifact.v1"] = "frayid_mantle_artifact.v1"
    experiment_id: str
    authority: Literal["intrinsic_atlas_conditioned_clipped_sdf"]
    claim: Literal["evidence-consistent MANTLE reconstruction"]
    d03_collision_body_role: Literal["immutable_prior_derived_collision_body"]
    atlas: UpperGarmentAtlas
    neutral_embedding: DerivedSurfaceExport
    posed_exports: list[DerivedSurfaceExport]
    excluded_products: list[
        Literal[
            "measurements",
            "sizing",
            "tailoring_patterns",
            "textures",
            "avatars",
            "3dgs",
            "virtual_try_on",
        ]
    ]
    status: Literal["pass", "fail", "blocked"]
    blockers: list[str]

    @model_validator(mode="after")
    def _excluded_scope_is_complete(self) -> MantleArtifact:
        expected = {
            "measurements",
            "sizing",
            "tailoring_patterns",
            "textures",
            "avatars",
            "3dgs",
            "virtual_try_on",
        }
        if set(self.excluded_products) != expected:
            raise ValueError("MANTLE must explicitly exclude every out-of-scope product")
        return self
