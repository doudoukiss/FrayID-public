from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from torch import Tensor, nn

from frayid.dataset import read_dataset_manifest
from frayid.io import read_json, sha256_file
from frayid.schemas import SequenceInitialization
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.schemas import TurntableSolution


def monotonic_angles(raw_increments: Tensor, *, direction: int = 1) -> Tensor:
    if raw_increments.ndim != 1 or raw_increments.numel() < 1:
        raise ValueError("turntable increments must be one nonempty vector")
    if direction not in (-1, 1):
        raise ValueError("turntable direction must be chosen once as -1 or 1")
    increments = F.softplus(raw_increments)
    angles = torch.cat((increments.new_zeros(1), torch.cumsum(increments, dim=0)))
    return angles * direction


def axis_angle_rotation(axis: Tensor, angles: Tensor) -> Tensor:
    if axis.shape != (3,) or angles.ndim != 1:
        raise ValueError("axis-angle rotation expects one axis and angle vector")
    normalized = F.normalize(axis, dim=0, eps=1.0e-12)
    x, y, z = normalized
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    sine = torch.sin(angles)[:, None, None]
    cosine = torch.cos(angles)[:, None, None]
    return identity + sine * skew + (1.0 - cosine) * (skew @ skew)


def turntable_fundamental_matrices(
    rotations: Tensor,
    center: Tensor,
    intrinsics: Tensor,
    first_frame_slots: Tensor,
    second_frame_slots: Tensor,
) -> Tensor:
    """Derive pairwise image geometry from object-centric turntable motion."""

    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("turntable rotations must have shape [frame, 3, 3]")
    if center.shape != (3,) or intrinsics.shape != (3, 3):
        raise ValueError("turntable center and intrinsics have invalid shape")
    if (
        first_frame_slots.ndim != 1
        or second_frame_slots.shape != first_frame_slots.shape
        or first_frame_slots.dtype != torch.long
        or second_frame_slots.dtype != torch.long
    ):
        raise ValueError("turntable edge slots must be aligned torch.long vectors")
    frame_count = int(rotations.shape[0])
    if bool(
        torch.any(first_frame_slots < 0)
        or torch.any(second_frame_slots < 0)
        or torch.any(first_frame_slots >= frame_count)
        or torch.any(second_frame_slots >= frame_count)
    ):
        raise ValueError("turntable edge slot is outside the rotation sequence")
    first = rotations[first_frame_slots]
    second = rotations[second_frame_slots]
    relative = second @ first.transpose(-1, -2)
    translation = center[None, :] - torch.einsum("nij,j->ni", relative, center)
    x, y, z = translation.unbind(dim=-1)
    zero = torch.zeros_like(x)
    translation_skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(-1, 3, 3)
    essential = translation_skew @ relative
    inverse = torch.linalg.inv(intrinsics)
    fundamental = inverse.transpose(-1, -2)[None, :, :] @ essential @ inverse[None, :, :]
    scale = torch.linalg.vector_norm(fundamental, dim=(-2, -1), keepdim=True).clamp_min(1.0e-12)
    normalized: Tensor = fundamental / scale
    return normalized


def turntable_edge_slots(
    solution_source_frame_indices: list[int],
    first_source_frame_indices: Tensor,
    second_source_frame_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Bind train-only factor edges to a TurntableSolution frame order."""

    if first_source_frame_indices.shape != second_source_frame_indices.shape:
        raise ValueError("track-factor source-index vectors must align")
    slots = {source_index: slot for slot, source_index in enumerate(solution_source_frame_indices)}
    try:
        first = [slots[int(value)] for value in first_source_frame_indices.cpu().tolist()]
        second = [slots[int(value)] for value in second_source_frame_indices.cpu().tolist()]
    except KeyError as error:
        raise ValueError("track-factor edge is absent from the turntable solution") from error
    device = first_source_frame_indices.device
    return (
        torch.tensor(first, dtype=torch.long, device=device),
        torch.tensor(second, dtype=torch.long, device=device),
    )


class TurntableModel(nn.Module):
    """Reduced object-centric motion model for one cooperative rotation sequence."""

    def __init__(
        self,
        frame_count: int,
        *,
        micromotion_rank: int = 4,
        micromotion_dimension: int = 69,
    ) -> None:
        super().__init__()
        if frame_count < 3 or micromotion_rank < 0 or micromotion_dimension < micromotion_rank:
            raise ValueError("turntable model dimensions are invalid")
        self.frame_count = frame_count
        self.micromotion_rank = micromotion_rank
        self.axis_tangent = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]))
        self.center = nn.Parameter(torch.zeros(3))
        self.raw_angle_increments = nn.Parameter(torch.full((frame_count - 1,), -1.0))
        self.residual_twists = nn.Parameter(torch.zeros(frame_count, 6))
        self.micromotion_codes = nn.Parameter(torch.zeros(frame_count, micromotion_rank))
        self.micromotion_basis = nn.Parameter(torch.eye(micromotion_rank, micromotion_dimension))
        self.log_focal = nn.Parameter(torch.tensor(math.log(1000.0)))
        self.principal_point = nn.Parameter(torch.zeros(2))

    @property
    def axis(self) -> Tensor:
        return F.normalize(self.axis_tangent, dim=0, eps=1.0e-12)

    @property
    def angles(self) -> Tensor:
        return monotonic_angles(self.raw_angle_increments)

    def intrinsics(self) -> Tensor:
        focal = torch.exp(self.log_focal)
        matrix = torch.eye(3, dtype=focal.dtype, device=focal.device)
        matrix[0, 0] = focal
        matrix[1, 1] = focal
        matrix[0, 2] = self.principal_point[0]
        matrix[1, 2] = self.principal_point[1]
        return matrix

    def rotations(self) -> Tensor:
        return axis_angle_rotation(self.axis, self.angles)

    def micromotion(self) -> Tensor:
        return self.micromotion_codes @ self.micromotion_basis

    def gauge_penalty(self) -> Tensor:
        residual_mean = self.residual_twists.mean(dim=0).square().sum()
        if self.micromotion_rank:
            code_mean = self.micromotion_codes.mean(dim=0).square().sum()
            basis_gram = self.micromotion_basis @ self.micromotion_basis.T
            basis_identity = torch.eye(
                self.micromotion_rank,
                dtype=basis_gram.dtype,
                device=basis_gram.device,
            )
            basis_gauge = (basis_gram - basis_identity).square().mean()
        else:
            code_mean = self.axis_tangent.new_zeros(())
            basis_gauge = self.axis_tangent.new_zeros(())
        angle_smoothness = self.angles.diff().diff().square().mean()
        return residual_mean + code_mean + basis_gauge + angle_smoothness


@dataclass(frozen=True)
class IdentifiabilityDiagnostics:
    geometry_motion_principal_angle_degrees: float
    geometry_camera_principal_angle_degrees: float
    motion_camera_principal_angle_degrees: float
    maximum_canonical_correlation: float
    geometry_schur_eigenvalues: list[float]
    geometry_rank: int
    motion_rank: int
    camera_rank: int

    def as_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "geometry_motion_principal_angle_degrees": self.geometry_motion_principal_angle_degrees,
            "geometry_camera_principal_angle_degrees": self.geometry_camera_principal_angle_degrees,
            "motion_camera_principal_angle_degrees": self.motion_camera_principal_angle_degrees,
            "maximum_canonical_correlation": self.maximum_canonical_correlation,
            "geometry_schur_eigenvalues": self.geometry_schur_eigenvalues,
            "geometry_rank": self.geometry_rank,
            "motion_rank": self.motion_rank,
            "camera_rank": self.camera_rank,
        }


def _orthonormal_range(matrix: np.ndarray, tolerance: float) -> tuple[np.ndarray, int]:
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = tolerance * max(float(singular[0]) if singular.size else 0.0, 1.0)
    rank = int(np.sum(singular > threshold))
    return left[:, :rank], rank


def _principal_angle(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    if first.shape[1] == 0 or second.shape[1] == 0:
        return 90.0, 0.0
    singular = np.linalg.svd(first.T @ second, compute_uv=False)
    correlation = float(np.clip(singular[0], 0.0, 1.0))
    return math.degrees(math.acos(correlation)), correlation


def identifiability_diagnostics(
    geometry_jacobian: np.ndarray,
    motion_jacobian: np.ndarray,
    camera_jacobian: np.ndarray,
    *,
    damping: float = 1.0e-8,
    rank_tolerance: float = 1.0e-8,
) -> IdentifiabilityDiagnostics:
    matrices = (geometry_jacobian, motion_jacobian, camera_jacobian)
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("Jacobian blocks must be matrices")
    if len({matrix.shape[0] for matrix in matrices}) != 1:
        raise ValueError("Jacobian blocks must share residual rows")
    ranges = [_orthonormal_range(matrix, rank_tolerance) for matrix in matrices]
    gm_angle, gm_corr = _principal_angle(ranges[0][0], ranges[1][0])
    gc_angle, gc_corr = _principal_angle(ranges[0][0], ranges[2][0])
    mc_angle, mc_corr = _principal_angle(ranges[1][0], ranges[2][0])
    nuisance = np.concatenate((motion_jacobian, camera_jacobian), axis=1)
    h_gg = geometry_jacobian.T @ geometry_jacobian
    h_gn = geometry_jacobian.T @ nuisance
    h_nn = nuisance.T @ nuisance + damping * np.eye(nuisance.shape[1])
    schur = h_gg - h_gn @ np.linalg.solve(h_nn, h_gn.T)
    eigenvalues = np.linalg.eigvalsh(0.5 * (schur + schur.T))
    return IdentifiabilityDiagnostics(
        geometry_motion_principal_angle_degrees=gm_angle,
        geometry_camera_principal_angle_degrees=gc_angle,
        motion_camera_principal_angle_degrees=mc_angle,
        maximum_canonical_correlation=max(gm_corr, gc_corr, mc_corr),
        geometry_schur_eigenvalues=eigenvalues.tolist(),
        geometry_rank=ranges[0][1],
        motion_rank=ranges[1][1],
        camera_rank=ranges[2][1],
    )


def initialize_turntable_solution(
    initialization_path: Path,
    *,
    micromotion_rank: int = 4,
) -> TurntableSolution:
    initialization = SequenceInitialization.model_validate(read_json(initialization_path))
    if initialization.status == "blocked" or not initialization.frames:
        raise ValueError("turntable initialization requires a usable CameraHMR sequence")
    frames = sorted(initialization.frames, key=lambda item: item.source_frame_index)
    orientations = np.asarray([frame.global_orient for frame in frames], dtype=np.float64)
    yaw = np.unwrap(orientations[:, 1])
    direction = 1.0 if yaw[-1] >= yaw[0] else -1.0
    angles = direction * (yaw - yaw[0])
    angles = np.maximum.accumulate(angles)
    body_pose = np.asarray([frame.body_pose for frame in frames], dtype=np.float64)
    centered = body_pose - body_pose.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    rank = min(micromotion_rank, len(frames) - 1, right.shape[0])
    basis = right[:rank]
    codes = centered @ basis.T
    total = float(np.square(singular).sum())
    retained = float(np.square(singular[:rank]).sum() / total) if total > 0 else 1.0
    focal = initialization.shared_focal_length_px
    cx, cy = initialization.shared_principal_point_px
    intrinsics = [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]
    return TurntableSolution(
        status="qualification_candidate",
        shared_intrinsics=intrinsics,
        axis=[0.0, direction, 0.0],
        center=[0.0, 0.0, 0.0],
        angles_radians=angles.tolist(),
        residual_twists=[[0.0] * 6 for _ in frames],
        micromotion_basis=basis.tolist(),
        micromotion_codes=codes.tolist(),
        source_frame_indices=[frame.source_frame_index for frame in frames],
        gauge_policy={
            "angle_origin": "first_training_frame_zero",
            "axis_sign": "chosen_once_for_monotonic_positive_angles",
            "residual_twist_mean": "zero",
            "micromotion_mean": "zero",
            "scale": "inherited_declared_canonical_scale",
        },
        uncertainty={
            "micromotion_variance_retained": retained,
            "axis_uncertainty_degrees": 180.0,
            "center_uncertainty_fraction_bbox": 1.0,
        },
        source_provenance={
            "initialization_path": str(initialization_path),
            "camerahmr_revision": initialization.source_revision,
            "role": "initialization_only_not_scientific_t01_result",
        },
    )


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("isotonic turntable input must be one finite nonempty vector")
    levels: list[float] = []
    weights: list[int] = []
    for value in values:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            pooled = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_weight
            levels[-2:] = [pooled]
            weights[-2:] = [total_weight]
    result = np.concatenate(
        [
            np.full((weight,), level, dtype=np.float64)
            for level, weight in zip(levels, weights, strict=True)
        ]
    )
    if len(result) != len(values):
        raise AssertionError("isotonic turntable projection changed vector length")
    return result


def _project_bounded_increments(
    increments: np.ndarray,
    *,
    total: float,
    minimum_mean_ratio: float = 0.25,
    maximum_mean_ratio: float = 3.0,
) -> np.ndarray:
    """Project increments to a positive bounded simplex without changing total phase."""

    if increments.ndim != 1 or len(increments) == 0 or not np.isfinite(increments).all():
        raise ValueError("phase increments must be one finite nonempty vector")
    if total <= 0 or not 0 < minimum_mean_ratio < 1 < maximum_mean_ratio:
        raise ValueError("phase increment projection bounds are invalid")
    mean = total / len(increments)
    lower = minimum_mean_ratio * mean
    upper = maximum_mean_ratio * mean
    low_shift = lower - float(np.max(increments)) - total
    high_shift = upper - float(np.min(increments)) + total
    for _ in range(100):
        shift = 0.5 * (low_shift + high_shift)
        candidate = np.clip(increments + shift, lower, upper)
        if float(candidate.sum()) < total:
            low_shift = shift
        else:
            high_shift = shift
    result = np.clip(increments + 0.5 * (low_shift + high_shift), lower, upper)
    residual = total - float(result.sum())
    free = (result > lower + 1.0e-12) & (result < upper - 1.0e-12)
    if np.any(free):
        result[free] += residual / int(np.sum(free))
    else:
        result += residual / len(result)
    if (
        not np.isfinite(result).all()
        or np.any(result <= 0)
        or not math.isclose(float(result.sum()), total, rel_tol=0.0, abs_tol=1.0e-10)
    ):
        raise AssertionError("bounded phase projection failed")
    return result


def initialize_cooperative_turntable_solution(
    initialization_path: Path,
    manifest_path: Path,
    validation_path: Path,
    track_graph_report_path: Path,
    *,
    micromotion_rank: int = 4,
) -> TurntableSolution:
    """Build a full-turn train-only initializer without claiming a T01 fit."""

    paths = [initialization_path, manifest_path, validation_path, track_graph_report_path]
    reject_sealed_capability(paths)
    validation = read_json(validation_path)
    if validation.get("status") != "ready" or validation.get("blockers"):
        raise ValueError("cooperative turntable initialization requires ready dataset evidence")
    track_graph = read_json(track_graph_report_path)
    gate_results = track_graph.get("gate_results", {})
    if not isinstance(gate_results, dict) or not gate_results.get(
        "temporal_track_graph_eligible_for_t01"
    ):
        raise ValueError("cooperative turntable initialization requires a passing Q01 graph")
    manifest = read_dataset_manifest(manifest_path)
    train_source_indices = {
        frame.source_frame_index
        for frame in manifest.frames
        if frame.split == "train" and frame.quality_accepted
    }
    initialization = SequenceInitialization.model_validate(read_json(initialization_path))
    frames = sorted(
        (
            frame
            for frame in initialization.frames
            if frame.source_frame_index in train_source_indices
        ),
        key=lambda item: item.source_frame_index,
    )
    if len(frames) != len(train_source_indices) or len(frames) < 3:
        raise ValueError("train-only CameraHMR initialization does not match the manifest")
    orientations = np.asarray([frame.global_orient for frame in frames], dtype=np.float64)
    euler = Rotation.from_rotvec(orientations).as_euler("xyz")
    unwrapped = np.unwrap(euler, axis=0)
    spans = unwrapped[-1] - unwrapped[0]
    selected_component = int(np.argmax(np.abs(spans)))
    selected_span = float(spans[selected_component])
    if abs(selected_span) < math.pi:
        raise ValueError("CameraHMR rotation does not expose a cooperative full-turn phase")
    direction = 1.0 if selected_span >= 0 else -1.0
    raw_progress = direction * (unwrapped[:, selected_component] - unwrapped[0, selected_component])
    isotonic = _isotonic_increasing(raw_progress)
    isotonic -= isotonic[0]
    projected_increments = _project_bounded_increments(
        np.diff(isotonic),
        total=abs(selected_span),
    )
    angles = np.concatenate((np.zeros(1), np.cumsum(projected_increments)))
    if angles[-1] < 1.5 * math.pi or angles[-1] > 2.5 * math.pi:
        raise ValueError("cooperative phase span is inconsistent with one full turn")
    translations = np.asarray([frame.translation for frame in frames], dtype=np.float64)
    center = np.median(translations, axis=0)
    center_mad = np.median(np.abs(translations - center), axis=0)
    body_pose = np.asarray([frame.body_pose for frame in frames], dtype=np.float64)
    centered_pose = body_pose - body_pose.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centered_pose, full_matrices=False)
    rank = min(micromotion_rank, len(frames) - 1, right.shape[0])
    basis = right[:rank]
    codes = centered_pose @ basis.T
    total = float(np.square(singular).sum())
    retained = float(np.square(singular[:rank]).sum() / total) if total > 0 else 1.0
    increments = np.diff(angles)
    median_increment = float(np.median(increments))
    phase_mad = float(np.median(np.abs(increments - median_increment)))
    focal = initialization.shared_focal_length_px
    cx, cy = initialization.shared_principal_point_px
    intrinsics = [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]
    component_names = ("x", "y", "z")
    return TurntableSolution(
        status="qualification_candidate",
        shared_intrinsics=intrinsics,
        axis=[0.0, direction, 0.0],
        center=center.tolist(),
        angles_radians=angles.tolist(),
        residual_twists=[[0.0] * 6 for _ in frames],
        micromotion_basis=basis.tolist(),
        micromotion_codes=codes.tolist(),
        source_frame_indices=[frame.source_frame_index for frame in frames],
        gauge_policy={
            "angle_origin": "first_accepted_training_frame_zero",
            "phase_source": "largest_unwrapped_xyz_euler_component_from_rotation_matrices",
            "phase_speed_projection": "closest_simplex_with_0.25x_to_3x_mean_increment_bounds",
            "axis_sign": "cooperative_phase_direction_chosen_once",
            "axis_geometry": "camera_vertical_initialization_only",
            "center": "train_only_componentwise_median_camerahmr_translation",
            "center_parallel_to_axis": "fixed_initialization_gauge",
            "residual_twist_mean": "zero",
            "micromotion_mean": "zero",
            "scale": "inherited_declared_canonical_scale",
        },
        uncertainty={
            "micromotion_variance_retained": retained,
            "phase_increment_mad_radians": phase_mad,
            "center_mad_x": float(center_mad[0]),
            "center_mad_y": float(center_mad[1]),
            "center_mad_z": float(center_mad[2]),
            "axis_uncertainty_degrees": 10.0,
        },
        source_provenance={
            "initialization_path": str(initialization_path),
            "initialization_sha256": sha256_file(initialization_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "validation_sha256": sha256_file(validation_path),
            "track_graph_report_sha256": sha256_file(track_graph_report_path),
            "selected_unwrapped_euler_component": component_names[selected_component],
            "selected_signed_span_radians": str(selected_span),
            "role": "train_only_cooperative_initialization_not_scientific_t01_result",
        },
    )
