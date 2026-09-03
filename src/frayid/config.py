from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    asset_manifest: Path
    input_video: Path
    reference_video: Path
    dataset_root: Path
    run_root: Path


class DatasetConfig(BaseModel):
    target_frame_count: int = Field(gt=0)
    minimum_usable_frame_count: int = Field(gt=0)
    candidate_multiplier: int = Field(ge=1)
    held_out_stride: int = Field(ge=2)
    minimum_blur_variance: float = Field(ge=0)
    minimum_mean_luminance: float = Field(ge=0, le=255)
    maximum_mean_luminance: float = Field(ge=0, le=255)
    output_width: int = Field(gt=0)
    output_height: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> DatasetConfig:
        if self.minimum_usable_frame_count > self.target_frame_count:
            raise ValueError("minimum_usable_frame_count cannot exceed target_frame_count")
        if self.minimum_mean_luminance >= self.maximum_mean_luminance:
            raise ValueError("minimum luminance must be below maximum luminance")
        return self


class EvidenceConfig(BaseModel):
    masks_subdirectory: str
    normals_subdirectory: str
    observed_pose_filename: str
    raw_initialization_filename: str
    initialization_filename: str
    allow_proxy_masks: Literal[False]
    allow_constant_normals: Literal[False]
    allow_proxy_camera: Literal[False]
    allow_zero_pose: Literal[False]


class InitializationGateConfig(BaseModel):
    minimum_median_silhouette_iou: float = Field(ge=0, le=1)
    overlay_frame_count: int = Field(gt=0)
    maximum_root_rotation_jump_degrees: float = Field(gt=0)
    shared_focal_required: bool
    shared_shape_required: bool
    observed_pose_minimum_confidence: float = Field(ge=0, le=1)
    envelope_maximum_offset_m: float = Field(gt=0)
    envelope_smoothness_weight: float = Field(ge=0)
    envelope_magnitude_weight: float = Field(ge=0)


class PoseRefitConfig(BaseModel):
    """Fixed R4B.3 sequence-pose refinement contract."""

    steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    silhouette_batch_size: int = Field(gt=0)
    render_resolution: int = Field(gt=0)
    robust_scale_pixels: float = Field(gt=0)
    maximum_joint_correction_degrees: float = Field(gt=0, le=45)
    maximum_root_correction_degrees: float = Field(gt=0, le=20)
    maximum_translation_correction_m: float = Field(gt=0, le=0.2)
    observed_joint_weight: float = Field(ge=0)
    camerahmr_joint_weight: float = Field(ge=0)
    joint_3d_anchor_weight: float = Field(ge=0)
    silhouette_weight: float = Field(ge=0)
    boundary_weight: float = Field(ge=0)
    pose_anchor_weight: float = Field(ge=0)
    root_anchor_weight: float = Field(ge=0)
    translation_anchor_weight: float = Field(ge=0)
    rotation_acceleration_weight: float = Field(ge=0)
    translation_acceleration_weight: float = Field(ge=0)


class ModelConfig(BaseModel):
    positional_encoding_frequencies: int = Field(ge=0)
    sdf_hidden_dim: int = Field(gt=0)
    sdf_layer_count: int = Field(ge=2)
    residual_hidden_dim: int = Field(gt=0)
    residual_layer_count: int = Field(ge=2)
    frame_code_dim: int = Field(gt=0)
    eikonal_sample_count: int = Field(gt=0)
    renderer_resolution: int = Field(gt=0)
    renderer_sigma_pixels: float = Field(gt=0)
    renderer_max_vertices: int = Field(gt=0)
    renderer_reference_sample_count: int = Field(gt=0)
    renderer_depth_temperature_m: float = Field(gt=0)
    canonical_orientation_margin: float = Field(gt=0)
    canonical_minimum_signed_area_ratio: float = Field(gt=0)
    canonical_minimum_area_ratio: float = Field(gt=0)
    canonical_backtracking_steps: int = Field(gt=0)
    maximum_root_rotation_correction_degrees: float = Field(gt=0)
    maximum_root_translation_correction_m: float = Field(gt=0)


class LossWeights(BaseModel):
    silhouette: float = Field(ge=0)
    boundary: float = Field(ge=0)
    normal: float = Field(ge=0)
    eikonal: float = Field(ge=0)
    sdf_mesh_consistency: float = Field(ge=0)
    deformation: float = Field(ge=0)
    jacobian: float = Field(ge=0)
    temporal: float = Field(ge=0)
    canonical_orientation: float = Field(ge=0)
    canonical_area: float = Field(ge=0)
    canonical_edge_strain: float = Field(ge=0)
    canonical_smoothness: float = Field(ge=0)
    root_rotation_correction: float = Field(ge=0)
    root_translation_correction: float = Field(ge=0)
    root_correction_temporal: float = Field(ge=0)


class TrainingStageConfig(BaseModel):
    name: Literal["coarse", "medium", "fine"]
    start_epoch: int = Field(ge=0)
    grid_resolution: int = Field(gt=0)
    mask_resolution: int = Field(gt=0)


class TrainingConfig(BaseModel):
    learning_rate: float = Field(gt=0)
    checkpoint_every_epochs: int = Field(gt=0)
    refresh_mesh_every_epochs: int = Field(gt=0)
    stages: list[TrainingStageConfig]
    full_epoch_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_stages(self) -> TrainingConfig:
        if [stage.name for stage in self.stages] != ["coarse", "medium", "fine"]:
            raise ValueError("training stages must be coarse, medium, fine")
        starts = [stage.start_epoch for stage in self.stages]
        if starts != sorted(starts) or starts[-1] >= self.full_epoch_count:
            raise ValueError("training stage schedule is unreachable")
        return self


class SmokeConfig(BaseModel):
    frame_count: int = Field(gt=0)
    epoch_count: int = Field(gt=0)
    stage: Literal["coarse"]
    learning_rate: float = Field(gt=0)
    required_loss_reduction_fraction: float = Field(gt=0, lt=1)
    gpu: str
    timeout_seconds: int = Field(gt=0, le=1800)
    automatic_retry_count: Literal[0]
    full_training_authorized: Literal[True]


class EvaluationConfig(BaseModel):
    held_out_silhouette_iou: float = Field(ge=0, le=1)
    minimum_iou_improvement: float = Field(ge=0, le=1)
    maximum_normalized_boundary_error: float = Field(ge=0)
    maximum_median_normal_error_degrees: float = Field(ge=0, le=180)
    maximum_train_held_out_iou_gap: float = Field(ge=0, le=1)
    minimum_dominant_component_area_fraction: float = Field(ge=0, le=1)


class ReconstructionConfig(BaseModel):
    schema_version: Literal["reconstruction_config.v1"]
    run_id: str
    seed: int
    paths: PathsConfig
    dataset: DatasetConfig
    evidence: EvidenceConfig
    initialization_gate: InitializationGateConfig
    pose_refit: PoseRefitConfig
    model: ModelConfig
    losses: LossWeights
    training: TrainingConfig
    smoke: SmokeConfig
    evaluation: EvaluationConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAYID_", env_file=".env", extra="ignore")

    env: str = "development"
    output_root: Path = Path("outputs")
    model_root: Path = Path("models")
    privacy_mode: Literal["local_only"] = "local_only"
    config_path: Path = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml")
    smpl_model_root: Path = Path("models/private/camerahmr_assets/models/SMPL")


def load_config(path: Path | None = None) -> ReconstructionConfig:
    config_path = path or Settings().config_path
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object: {config_path}")
    return ReconstructionConfig.model_validate(payload)
