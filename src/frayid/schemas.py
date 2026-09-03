from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class VideoMetadata(BaseModel):
    path: str
    codec: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    frame_rate: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    size_bytes: int = Field(gt=0)


class ExpectedMedia(BaseModel):
    codec: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)


class LocalAsset(BaseModel):
    asset_id: str
    role: Literal["reconstruction_input", "qualitative_reference_only"]
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected: ExpectedMedia
    allowed_uses: list[str]
    forbidden_uses: list[str] = Field(default_factory=list)
    tracked_in_git: bool


class LocalAssetManifest(BaseModel):
    schema_version: Literal["local_media.v1"]
    privacy: Literal["local_only", "private_git_remote"]
    assets: list[LocalAsset]


class AssetCheck(BaseModel):
    asset_id: str
    path: str
    exists: bool
    sha256_matches: bool
    metadata_matches: bool
    observed_sha256: str | None = None
    metadata: VideoMetadata | None = None
    errors: list[str] = Field(default_factory=list)


class AssetVerificationReport(BaseModel):
    schema_version: Literal["asset_verification.v1"] = "asset_verification.v1"
    status: Literal["ready", "blocked"]
    privacy: Literal["local_only", "private_git_remote"]
    checks: list[AssetCheck]


class FrameRecord(BaseModel):
    ordinal: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    image_path: str
    split: Literal["train", "held_out"]
    blur_variance: float = Field(ge=0)
    mean_luminance: float = Field(ge=0, le=255)
    quality_accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    schema_version: Literal["canonical_dataset.v1"] = "canonical_dataset.v1"
    status: Literal["rgb_ready", "evidence_ready", "blocked"]
    run_id: str
    input_video_path: str
    input_video_sha256: str
    video: VideoMetadata
    dataset_root: str
    frames: list[FrameRecord]
    train_frame_count: int = Field(ge=0)
    held_out_frame_count: int = Field(ge=0)
    rejected_candidate_count: int = Field(ge=0)
    blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> DatasetManifest:
        if self.train_frame_count != sum(frame.split == "train" for frame in self.frames):
            raise ValueError("train_frame_count does not match frame records")
        if self.held_out_frame_count != sum(frame.split == "held_out" for frame in self.frames):
            raise ValueError("held_out_frame_count does not match frame records")
        return self


class EvidenceFrameCheck(BaseModel):
    ordinal: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    image_path: str
    mask_path: str
    normal_path: str
    mask_present: bool
    normal_present: bool
    dimensions_match: bool
    mask_foreground_fraction: float | None = None
    normal_variance: float | None = None
    blockers: list[str] = Field(default_factory=list)


class DatasetValidationReport(BaseModel):
    schema_version: Literal["dataset_validation.v1"] = "dataset_validation.v1"
    status: Literal["ready", "blocked"]
    dataset_manifest_path: str
    selected_frame_count: int = Field(ge=0)
    evidence_complete_frame_count: int = Field(ge=0)
    minimum_usable_frame_count: int = Field(gt=0)
    initialization_present: bool
    observed_pose_present: bool = False
    frame_checks: list[EvidenceFrameCheck]
    blockers: list[str] = Field(default_factory=list)


class CameraHMRFrame(BaseModel):
    source_frame_index: int = Field(ge=0)
    betas: list[float] = Field(min_length=10)
    body_pose: list[float] = Field(min_length=63)
    global_orient: list[float] = Field(min_length=3, max_length=3)
    translation: list[float] = Field(min_length=3, max_length=3)
    focal_length_px: float = Field(gt=0)
    principal_point_px: list[float] = Field(min_length=2, max_length=2)
    keypoints_2d: list[list[float]] = Field(default_factory=list)
    joints_3d: list[list[float]] = Field(default_factory=list)
    bounding_box_xyxy: list[float] = Field(min_length=4, max_length=4)
    detection_score: float = Field(ge=0, le=1)
    keypoints_source: Literal["camerahmr_projected_smpl_joints"] = "camerahmr_projected_smpl_joints"
    source: Literal["camerahmr"] = "camerahmr"


class ObservedPoseFrame(BaseModel):
    source_frame_index: int = Field(ge=0)
    keypoints_body12: list[list[float]] = Field(min_length=12, max_length=12)
    bounding_box_xyxy: list[float] = Field(min_length=4, max_length=4)
    source: Literal["sapiens2_pose"] = "sapiens2_pose"


class ObservedPoseSequence(BaseModel):
    schema_version: Literal["observed_pose_sequence.v1"] = "observed_pose_sequence.v1"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    frames: list[ObservedPoseFrame]
    source_revision: str
    model_revision: str
    detector_revision: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proxy_evidence: Literal[False] = False


class SequenceInitialization(BaseModel):
    schema_version: Literal["sequence_initialization.v1"] = "sequence_initialization.v1"
    status: Literal["raw", "refined", "blocked"]
    shared_betas: list[float] = Field(min_length=10)
    shared_focal_length_px: float = Field(gt=0)
    shared_principal_point_px: list[float] = Field(min_length=2, max_length=2)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    frames: list[CameraHMRFrame]
    canonical_mesh_path: str | None = None
    canonical_mesh_role: Literal["shared_smpl", "shared_clothing_envelope"] | None = None
    skinning_weights_path: str | None = None
    joint_transforms_path: str | None = None
    envelope_maximum_offset_m: float | None = Field(default=None, ge=0)
    envelope_rms_offset_m: float | None = Field(default=None, ge=0)
    envelope_laplacian_rms_m: float | None = Field(default=None, ge=0)
    source_revision: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    camera_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proxy_camera: Literal[False] = False
    zero_pose: Literal[False] = False
    blockers: list[str] = Field(default_factory=list)


class InitializationEvaluation(BaseModel):
    schema_version: Literal["initialization_evaluation.v1"] = "initialization_evaluation.v1"
    status: Literal["pass", "blocked", "fail"]
    median_silhouette_iou: float | None = None
    median_normalized_boundary_error: float | None = None
    median_keypoint_reprojection_error_px: float | None = None
    keypoint_source: Literal["sapiens2_pose"] | None = None
    maximum_root_rotation_jump_degrees: float | None = None
    shared_focal: bool
    shared_shape: bool
    evaluated_frame_count: int = Field(ge=0)
    blockers: list[str] = Field(default_factory=list)


class ModalSmokePlan(BaseModel):
    schema_version: Literal["modal_smoke_plan.v1"] = "modal_smoke_plan.v1"
    run_id: str
    mode: Literal["geometry_smoke"] = "geometry_smoke"
    gpu: Literal["L40S"]
    timeout_seconds: int = Field(gt=0, le=1800)
    frame_count: Literal[24]
    epoch_count: Literal[2]
    automatic_retry_count: Literal[0]
    full_training_authorized: Literal[True]
    config_path: str
    dataset_manifest_path: str
    command: list[str]
    status: Literal["ready", "blocked"]
    blockers: list[str] = Field(default_factory=list)


class RunProvenance(BaseModel):
    schema_version: Literal["run_provenance.v1"] = "run_provenance.v1"
    run_id: str
    created_at_utc: str
    git_commit: str
    command: list[str]
    environment: dict[str, str]
    input_video_sha256: str
    asset_manifest_sha256: str
    config_sha256: str
    dataset_manifest_sha256: str
    initialization_sha256: str
    seed: int
    frame_indices: list[int]
    enabled_losses: dict[str, float]
    optimizer_steps: int = Field(ge=0)
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None


class ReconstructionEvaluation(BaseModel):
    schema_version: Literal["reconstruction_evaluation.v1"] = "reconstruction_evaluation.v1"
    status: Literal["pass", "blocked", "fail"]
    train_silhouette_iou: float | None = None
    held_out_silhouette_iou: float | None = None
    initialization_held_out_iou: float | None = None
    normalized_boundary_error: float | None = None
    median_normal_error_degrees: float | None = None
    dominant_component_area_fraction: float | None = None
    canonical_mesh_watertight: bool | None = None
    canonical_topology_valid: bool | None = None
    canonical_flipped_face_fraction: float | None = Field(default=None, ge=0, le=1)
    canonical_collapsed_face_fraction: float | None = Field(default=None, ge=0, le=1)
    loss_reduction_fraction: float | None = None
    blockers: list[str] = Field(default_factory=list)


class EpochMetrics(BaseModel):
    epoch: int = Field(ge=0)
    stage: Literal["coarse", "medium", "fine"]
    total: float
    losses: dict[str, float]
    parameter_delta_norm: float = Field(ge=0)


class SmokeRunReport(BaseModel):
    schema_version: Literal["geometry_smoke.v1"] = "geometry_smoke.v1"
    run_id: str
    status: Literal["pass", "blocked", "fail"]
    frame_count: int = Field(ge=0)
    epoch_count: int = Field(ge=0)
    optimizer_steps: int = Field(ge=0)
    checkpoint_resume_verified: bool
    enabled_losses: list[str]
    gradient_parameter_groups: dict[str, bool]
    epoch_metrics: list[EpochMetrics]
    initial_fixed_objective: float | None = None
    final_fixed_objective: float | None = None
    loss_reduction_fraction: float | None = None
    checkpoint_path: str | None = None
    provenance_path: str | None = None
    topology_report_path: str | None = None
    blockers: list[str] = Field(default_factory=list)


class GeometryTrainingReport(BaseModel):
    schema_version: Literal["geometry_training_report.v1"] = "geometry_training_report.v1"
    run_id: str
    status: Literal["complete", "blocked", "fail"]
    frame_count: int = Field(ge=0)
    epoch_count: int = Field(ge=0)
    optimizer_steps: int = Field(ge=0)
    epoch_metrics: list[EpochMetrics]
    checkpoint_path: str | None = None
    provenance_path: str | None = None
    explicit_mesh_path: str | None = None
    sdf_mesh_path: str | None = None
    topology_report_path: str | None = None
    blockers: list[str] = Field(default_factory=list)
