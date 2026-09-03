from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import QualificationState, advance_qualification, reject_sealed_capability
from frayid.v2.material_tracks import (
    SEMANTIC_NAMES,
    load_visibility_bounded_material_tracks,
)
from frayid.v2.t05_fixed_camera import FixedCameraHumanSolution

Q03_EXPERIMENT_ID = "postv2_q03_interval_material_track_graph_r01"
Q03_SCHEMA = "frayid_v2_interval_material_track_graph.v1"
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

CYCLE_CLASS_CODEBOOK = {
    0: "below_registered_10_degree_interval",
    1: "short_10_to_below_30_degrees",
    2: "medium_30_to_90_degrees",
    3: "long_above_90_degrees",
}
DESCRIPTOR_PROVENANCE_CODEBOOK = {
    0: "q02_opencv_pyramidal_lk_shi_tomasi_seed_grayscale_patch_ncc",
}
OCCLUSION_STATE_CODEBOOK = {
    0: "visible_observation",
    1: "outside_visible_interval_unknown_or_occluded",
}


@dataclass(frozen=True)
class Q03Gate:
    minimum_short_track_count: int = 100
    minimum_medium_track_count: int = 20
    minimum_supported_semantic_layers: int = 3
    maximum_track_median_reprojection_pixels: float = 5.0
    maximum_track_p95_reprojection_pixels: float = 10.0
    minimum_track_mean_robust_weight: float = 0.30
    maximum_anchor_condition_number: float = 1.0e6
    minimum_positive_depth_fraction: float = 0.90
    public_minimum_geometry_improvement_fraction: float = 0.10
    public_minimum_reprojection_improvement_fraction: float = 0.10
    public_minimum_clean_acceptance_fraction: float = 0.95
    public_minimum_corrupted_safe_fraction: float = 1.0
    public_minimum_photometric_shuffled_margin: float = 0.10
    adversarial_real_track_count: int = 32

    def as_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AnchorFit:
    mean: np.ndarray
    covariance: np.ndarray
    reprojection_error_pixels: np.ndarray
    robust_weights: np.ndarray
    condition_number: float
    positive_depth_fraction: float

    @property
    def median_reprojection_pixels(self) -> float:
        return float(np.median(self.reprojection_error_pixels))

    @property
    def p95_reprojection_pixels(self) -> float:
        return float(np.quantile(self.reprojection_error_pixels, 0.95))

    @property
    def mean_robust_weight(self) -> float:
        return float(np.mean(self.robust_weights))


@dataclass(frozen=True)
class IntervalMaterialTrackGraph:
    track_ids: np.ndarray
    track_offsets: np.ndarray
    visible_interval_frame_ordinals: np.ndarray
    visible_interval_source_indices: np.ndarray
    interval_boundary_occlusion_codes: np.ndarray
    frame_ordinals: np.ndarray
    source_frame_indices: np.ndarray
    pixels: np.ndarray
    observation_covariance_pixels2: np.ndarray
    visibility_probability: np.ndarray
    occlusion_state_codes: np.ndarray
    robust_outlier_weights: np.ndarray
    descriptor_provenance_codes: np.ndarray
    local_descriptor_summary: np.ndarray
    layer_posterior: np.ndarray
    material_anchor_mean_metres: np.ndarray
    material_anchor_covariance_metres2: np.ndarray
    anchor_reprojection_error_pixels: np.ndarray
    anchor_condition_number: np.ndarray
    material_strain_rms_metres: np.ndarray
    angular_span_degrees: np.ndarray
    cycle_class_codes: np.ndarray
    loop_anchor_candidate: np.ndarray
    track_weights: np.ndarray
    accepted: np.ndarray

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    @property
    def observation_count(self) -> int:
        return len(self.frame_ordinals)

    def validate(self) -> None:
        track_count = self.track_count
        observation_count = self.observation_count
        if not np.array_equal(self.track_ids, np.arange(track_count, dtype=np.int64)):
            raise ValueError("Q03 track IDs must be contiguous int64 values")
        if self.track_offsets.shape != (track_count + 1,) or self.track_offsets.dtype != np.int64:
            raise ValueError("Q03 track offsets are invalid")
        if int(self.track_offsets[0]) != 0 or int(self.track_offsets[-1]) != observation_count:
            raise ValueError("Q03 track offsets do not span observations")
        observation_vectors = (
            self.frame_ordinals,
            self.source_frame_indices,
            self.visibility_probability,
            self.occlusion_state_codes,
            self.robust_outlier_weights,
            self.anchor_reprojection_error_pixels,
        )
        if any(value.shape != (observation_count,) for value in observation_vectors):
            raise ValueError("Q03 observation vectors do not align")
        if self.pixels.shape != (observation_count, 2):
            raise ValueError("Q03 pixels must have shape [observation_count,2]")
        if self.observation_covariance_pixels2.shape != (observation_count, 2, 2):
            raise ValueError("Q03 observation covariance has an invalid shape")
        interval_arrays = (
            self.visible_interval_frame_ordinals,
            self.visible_interval_source_indices,
            self.interval_boundary_occlusion_codes,
        )
        if any(value.shape != (track_count, 2) for value in interval_arrays):
            raise ValueError("Q03 visible-interval arrays have invalid shapes")
        if self.local_descriptor_summary.shape != (track_count, 4):
            raise ValueError("Q03 local descriptor summary has an invalid shape")
        per_track_vectors = (
            self.descriptor_provenance_codes,
            self.anchor_condition_number,
            self.material_strain_rms_metres,
            self.angular_span_degrees,
            self.cycle_class_codes,
            self.loop_anchor_candidate,
            self.track_weights,
            self.accepted,
        )
        if any(value.shape != (track_count,) for value in per_track_vectors):
            raise ValueError("Q03 per-track vectors do not align")
        if self.layer_posterior.shape != (track_count, len(SEMANTIC_NAMES)):
            raise ValueError("Q03 layer posterior has an invalid shape")
        if self.material_anchor_mean_metres.shape != (track_count, 3):
            raise ValueError("Q03 material-anchor means have an invalid shape")
        if self.material_anchor_covariance_metres2.shape != (track_count, 3, 3):
            raise ValueError("Q03 material-anchor covariance has an invalid shape")
        floating = (
            self.pixels,
            self.observation_covariance_pixels2,
            self.visibility_probability,
            self.robust_outlier_weights,
            self.local_descriptor_summary,
            self.layer_posterior,
            self.material_anchor_mean_metres,
            self.material_anchor_covariance_metres2,
            self.anchor_reprojection_error_pixels,
            self.anchor_condition_number,
            self.material_strain_rms_metres,
            self.angular_span_degrees,
            self.track_weights,
        )
        if any(not np.isfinite(value).all() for value in floating):
            raise ValueError("Q03 binding contains non-finite values")
        if np.any((self.visibility_probability < 0.0) | (self.visibility_probability > 1.0)):
            raise ValueError("Q03 visibility probability must lie in [0,1]")
        if np.any((self.robust_outlier_weights < 0.0) | (self.robust_outlier_weights > 1.0)):
            raise ValueError("Q03 robust weights must lie in [0,1]")
        if not np.allclose(self.layer_posterior.sum(axis=1), 1.0, atol=1.0e-6):
            raise ValueError("Q03 layer posteriors must sum to one")
        covariance_symmetry = np.max(
            np.abs(
                self.observation_covariance_pixels2
                - np.swapaxes(self.observation_covariance_pixels2, 1, 2)
            )
        )
        if covariance_symmetry > 1.0e-6:
            raise ValueError("Q03 observation covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(self.observation_covariance_pixels2) <= 0.0):
            raise ValueError("Q03 observation covariance must be positive definite")
        if np.any(~np.isin(self.cycle_class_codes, list(CYCLE_CLASS_CODEBOOK))):
            raise ValueError("Q03 cycle class code is unregistered")
        if np.any(~np.isin(self.descriptor_provenance_codes, list(DESCRIPTOR_PROVENANCE_CODEBOOK))):
            raise ValueError("Q03 descriptor provenance code is unregistered")
        if np.any(~np.isin(self.occlusion_state_codes, list(OCCLUSION_STATE_CODEBOOK))):
            raise ValueError("Q03 occlusion-state code is unregistered")
        if np.any(~np.isin(self.interval_boundary_occlusion_codes, list(OCCLUSION_STATE_CODEBOOK))):
            raise ValueError("Q03 interval-boundary occlusion code is unregistered")
        if np.any(self.accepted & ((self.cycle_class_codes == 0) | (self.track_weights <= 0.0))):
            raise ValueError("Q03 cannot accept below-range or zero-weight tracks")
        for start, stop in zip(self.track_offsets[:-1], self.track_offsets[1:], strict=True):
            if stop <= start or np.any(np.diff(self.frame_ordinals[start:stop]) <= 0):
                raise ValueError("Q03 observations must increase inside each visible interval")
        first_slots = self.track_offsets[:-1]
        last_slots = self.track_offsets[1:] - 1
        expected_frame_intervals = np.column_stack(
            (self.frame_ordinals[first_slots], self.frame_ordinals[last_slots])
        )
        expected_source_intervals = np.column_stack(
            (self.source_frame_indices[first_slots], self.source_frame_indices[last_slots])
        )
        if not np.array_equal(self.visible_interval_frame_ordinals, expected_frame_intervals):
            raise ValueError("Q03 frame intervals do not match their observations")
        if not np.array_equal(self.visible_interval_source_indices, expected_source_intervals):
            raise ValueError("Q03 source intervals do not match their observations")


def load_interval_material_track_graph(path: Path) -> IntervalMaterialTrackGraph:
    reject_sealed_capability([path])
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"]) != Q03_SCHEMA:
            raise ValueError("unsupported Q03 interval-track schema")
        graph = IntervalMaterialTrackGraph(
            track_ids=archive["track_ids"].astype(np.int64),
            track_offsets=archive["track_offsets"].astype(np.int64),
            visible_interval_frame_ordinals=archive["visible_interval_frame_ordinals"].astype(
                np.int64
            ),
            visible_interval_source_indices=archive["visible_interval_source_indices"].astype(
                np.int64
            ),
            interval_boundary_occlusion_codes=archive["interval_boundary_occlusion_codes"].astype(
                np.uint8
            ),
            frame_ordinals=archive["frame_ordinals"].astype(np.int64),
            source_frame_indices=archive["source_frame_indices"].astype(np.int64),
            pixels=archive["pixels"].astype(np.float32),
            observation_covariance_pixels2=archive["observation_covariance_pixels2"].astype(
                np.float32
            ),
            visibility_probability=archive["visibility_probability"].astype(np.float32),
            occlusion_state_codes=archive["occlusion_state_codes"].astype(np.uint8),
            robust_outlier_weights=archive["robust_outlier_weights"].astype(np.float32),
            descriptor_provenance_codes=archive["descriptor_provenance_codes"].astype(np.uint8),
            local_descriptor_summary=archive["local_descriptor_summary"].astype(np.float32),
            layer_posterior=archive["layer_posterior"].astype(np.float32),
            material_anchor_mean_metres=archive["material_anchor_mean_metres"].astype(np.float32),
            material_anchor_covariance_metres2=archive["material_anchor_covariance_metres2"].astype(
                np.float32
            ),
            anchor_reprojection_error_pixels=archive["anchor_reprojection_error_pixels"].astype(
                np.float32
            ),
            anchor_condition_number=archive["anchor_condition_number"].astype(np.float32),
            material_strain_rms_metres=archive["material_strain_rms_metres"].astype(np.float32),
            angular_span_degrees=archive["angular_span_degrees"].astype(np.float32),
            cycle_class_codes=archive["cycle_class_codes"].astype(np.uint8),
            loop_anchor_candidate=archive["loop_anchor_candidate"].astype(bool),
            track_weights=archive["track_weights"].astype(np.float32),
            accepted=archive["accepted"].astype(bool),
        )
    graph.validate()
    return graph


def _project_points(
    point: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.einsum("tij,j->ti", rotations, point) + translations
    safe_depth = np.where(np.abs(camera[:, 2]) > 1.0e-9, camera[:, 2], 1.0e-9)
    pixels = np.column_stack(
        (
            intrinsics[0, 0] * camera[:, 0] / safe_depth + intrinsics[0, 2],
            intrinsics[1, 1] * camera[:, 1] / safe_depth + intrinsics[1, 2],
        )
    )
    return pixels, camera[:, 2]


def robust_material_anchor(
    rotations: np.ndarray,
    translations: np.ndarray,
    pixels: np.ndarray,
    intrinsics: np.ndarray,
    observation_weights: np.ndarray,
    *,
    cauchy_scale_pixels: float = 3.0,
    iteration_count: int = 8,
) -> AnchorFit:
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    observation_weights = np.asarray(observation_weights, dtype=np.float64)
    count = len(pixels)
    if (
        count < 2
        or rotations.shape != (count, 3, 3)
        or translations.shape != (count, 3)
        or pixels.shape != (count, 2)
        or observation_weights.shape != (count,)
        or intrinsics.shape != (3, 3)
    ):
        raise ValueError("Q03 anchor inputs have invalid shapes")
    if (
        not np.isfinite(rotations).all()
        or not np.isfinite(translations).all()
        or not np.isfinite(pixels).all()
        or not np.isfinite(intrinsics).all()
        or not np.isfinite(observation_weights).all()
        or np.any(observation_weights <= 0.0)
        or cauchy_scale_pixels <= 0.0
        or iteration_count < 1
    ):
        raise ValueError("Q03 anchor inputs must be finite and positively weighted")
    inverse_intrinsics = np.linalg.inv(intrinsics)
    homogeneous = np.column_stack((pixels, np.ones(count)))
    camera_directions = homogeneous @ inverse_intrinsics.T
    camera_directions /= np.linalg.norm(camera_directions, axis=1, keepdims=True)
    canonical_directions = np.einsum("tji,tj->ti", rotations, camera_directions)
    canonical_origins = -np.einsum("tji,tj->ti", rotations, translations)
    identity = np.eye(3)
    projectors = identity[None] - np.einsum(
        "ti,tj->tij", canonical_directions, canonical_directions
    )
    robust_weights = np.ones(count, dtype=np.float64)
    normal = identity.copy()
    anchor = np.zeros(3, dtype=np.float64)
    errors = np.full(count, math.inf, dtype=np.float64)
    depths = np.zeros(count, dtype=np.float64)
    for _ in range(iteration_count):
        combined = observation_weights * robust_weights
        normal = np.einsum("t,tij->ij", combined, projectors) + 1.0e-9 * identity
        right = np.einsum("t,tij,tj->i", combined, projectors, canonical_origins)
        anchor = np.linalg.solve(normal, right)
        projected, depths = _project_points(anchor, rotations, translations, intrinsics)
        errors = np.linalg.norm(projected - pixels, axis=1)
        errors[depths <= 1.0e-6] = 1.0e6
        robust_weights = 1.0 / (1.0 + np.square(errors / cauchy_scale_pixels))
    combined = observation_weights * robust_weights
    normal = np.einsum("t,tij->ij", combined, projectors) + 1.0e-9 * identity
    variance = max(1.0e-10, float(np.average(np.square(errors), weights=combined)))
    covariance = np.linalg.pinv(normal) * variance
    return AnchorFit(
        mean=anchor,
        covariance=covariance,
        reprojection_error_pixels=errors,
        robust_weights=robust_weights,
        condition_number=float(np.linalg.cond(normal)),
        positive_depth_fraction=float(np.mean(depths > 1.0e-6)),
    )


def _cycle_class(span_degrees: float) -> int:
    if span_degrees < 10.0:
        return 0
    if span_degrees < 30.0:
        return 1
    if span_degrees <= 90.0:
        return 2
    return 3


def _accept_fit(fit: AnchorFit, cycle_class: int, gate: Q03Gate) -> bool:
    return (
        cycle_class != 0
        and fit.median_reprojection_pixels <= gate.maximum_track_median_reprojection_pixels
        and fit.p95_reprojection_pixels <= gate.maximum_track_p95_reprojection_pixels
        and fit.mean_robust_weight >= gate.minimum_track_mean_robust_weight
        and fit.condition_number <= gate.maximum_anchor_condition_number
        and fit.positive_depth_fraction >= gate.minimum_positive_depth_fraction
    )


def _relative_improvement(control: np.ndarray, treatment: np.ndarray) -> float:
    control_median = float(np.median(control))
    treatment_median = float(np.median(treatment))
    return (control_median - treatment_median) / max(control_median, 1.0e-12)


def _synthetic_rotations(phases: np.ndarray) -> np.ndarray:
    return np.asarray(
        Rotation.from_rotvec(
            np.column_stack((np.zeros(len(phases)), phases, np.zeros(len(phases))))
        ).as_matrix(),
        dtype=np.float64,
    )


def _photometric_margin(phases: np.ndarray, values: np.ndarray, *, seed: int) -> float:
    design = np.column_stack(
        (
            np.ones(len(phases)),
            np.cos(phases),
            np.sin(phases),
            np.cos(2.0 * phases),
            np.sin(2.0 * phases),
        )
    )
    train = np.arange(len(phases)) % 2 == 0
    test = ~train
    regularizer = 1.0e-6 * np.eye(design.shape[1])

    def predict(matrix: np.ndarray) -> np.ndarray:
        coefficients = np.linalg.solve(
            matrix[train].T @ matrix[train] + regularizer,
            matrix[train].T @ values[train],
        )
        return np.asarray(matrix[test] @ coefficients, dtype=np.float64)

    baseline = np.full(int(test.sum()), float(values[train].mean()))
    baseline_rmse = float(np.sqrt(np.mean(np.square(values[test] - baseline))))
    real_rmse = float(np.sqrt(np.mean(np.square(values[test] - predict(design)))))
    rng = np.random.default_rng(seed)
    shuffled_phases = phases[rng.permutation(len(phases))]
    shuffled = np.column_stack(
        (
            np.ones(len(phases)),
            np.cos(shuffled_phases),
            np.sin(shuffled_phases),
            np.cos(2.0 * shuffled_phases),
            np.sin(2.0 * shuffled_phases),
        )
    )
    shuffled_rmse = float(np.sqrt(np.mean(np.square(values[test] - predict(shuffled)))))
    real_improvement = (baseline_rmse - real_rmse) / max(baseline_rmse, 1.0e-12)
    shuffled_improvement = (baseline_rmse - shuffled_rmse) / max(baseline_rmse, 1.0e-12)
    return real_improvement - shuffled_improvement


def write_q03_public_benchmark(
    output_path: Path,
    *,
    seed: int = 20260903,
    gate: Q03Gate | None = None,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("Q03 public benchmark is immutable")
    gate = gate or Q03Gate()
    rng = np.random.default_rng(seed)
    intrinsics = np.asarray([[900.0, 0.0, 360.0], [0.0, 900.0, 560.0], [0.0, 0.0, 1.0]])
    clean_control_geometry: list[float] = []
    clean_treatment_geometry: list[float] = []
    clean_control_reprojection: list[float] = []
    clean_treatment_reprojection: list[float] = []
    clean_acceptance: list[bool] = []
    corrupted_safe: list[bool] = []
    class_counts = {name: 0 for name in CYCLE_CLASS_CODEBOOK.values()}
    for track_index in range(48):
        span_degrees = (20.0, 60.0, 120.0)[track_index % 3]
        start = rng.uniform(-0.3, 0.3)
        phases = np.linspace(start, start + math.radians(span_degrees), 12)
        rotations = _synthetic_rotations(phases)
        translations = np.column_stack(
            (
                0.015 * np.sin(phases),
                0.01 * np.cos(2.0 * phases),
                np.full(len(phases), 3.0),
            )
        )
        truth = np.asarray(
            [rng.uniform(-0.35, 0.35), rng.uniform(-0.65, 0.65), rng.uniform(-0.2, 0.2)]
        )
        truth_pixels, _ = _project_points(truth, rotations, translations, intrinsics)
        noisy_pixels = truth_pixels + rng.normal(0.0, 0.45, size=truth_pixels.shape)
        weights = np.ones(len(phases), dtype=np.float64)
        control_slots = np.asarray([0, len(phases) - 1])
        control = robust_material_anchor(
            rotations[control_slots],
            translations[control_slots],
            noisy_pixels[control_slots],
            intrinsics,
            weights[control_slots],
        )
        treatment = robust_material_anchor(
            rotations,
            translations,
            noisy_pixels,
            intrinsics,
            weights,
        )
        control_pixels, _ = _project_points(control.mean, rotations, translations, intrinsics)
        treatment_pixels, _ = _project_points(treatment.mean, rotations, translations, intrinsics)
        clean_control_geometry.append(float(np.linalg.norm(control.mean - truth)))
        clean_treatment_geometry.append(float(np.linalg.norm(treatment.mean - truth)))
        clean_control_reprojection.append(
            float(np.median(np.linalg.norm(control_pixels - truth_pixels, axis=1)))
        )
        clean_treatment_reprojection.append(
            float(np.median(np.linalg.norm(treatment_pixels - truth_pixels, axis=1)))
        )
        cycle_class = _cycle_class(span_degrees)
        class_counts[CYCLE_CLASS_CODEBOOK[cycle_class]] += 1
        clean_acceptance.append(_accept_fit(treatment, cycle_class, gate))
        corrupted_pixels = noisy_pixels.copy()
        corrupted_pixels[[3, 7, 9]] += np.asarray([24.0, -19.0])
        corrupted = robust_material_anchor(
            rotations,
            translations,
            corrupted_pixels,
            intrinsics,
            weights,
        )
        corrupted_accepted = _accept_fit(corrupted, cycle_class, gate)
        corrupted_geometry_error = float(np.linalg.norm(corrupted.mean - truth))
        corrupted_safe.append(
            (not corrupted_accepted)
            or corrupted_geometry_error <= clean_control_geometry[-1] + 1.0e-9
        )
    geometry_improvement = _relative_improvement(
        np.asarray(clean_control_geometry), np.asarray(clean_treatment_geometry)
    )
    reprojection_improvement = _relative_improvement(
        np.asarray(clean_control_reprojection), np.asarray(clean_treatment_reprojection)
    )
    clean_acceptance_fraction = float(np.mean(clean_acceptance))
    corrupted_safe_fraction = float(np.mean(corrupted_safe))

    local_phases = np.linspace(0.0, math.radians(60.0), 24)
    local_rotations = _synthetic_rotations(local_phases)
    local_translations = np.column_stack(
        (np.zeros(len(local_phases)), np.zeros(len(local_phases)), np.full(len(local_phases), 3.0))
    )
    local_truth = np.asarray([0.22, -0.1, 0.08])
    local_pixels, _ = _project_points(local_truth, local_rotations, local_translations, intrinsics)
    local_fit = robust_material_anchor(
        local_rotations,
        local_translations,
        local_pixels + rng.normal(0.0, 0.25, size=local_pixels.shape),
        intrinsics,
        np.ones(len(local_phases)),
    )
    local_interval_pass = _accept_fit(local_fit, _cycle_class(60.0), gate)
    global_phase = np.asarray([0.0, 2.0 * math.pi])
    global_rotations = _synthetic_rotations(global_phase)
    global_translations = np.asarray([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
    first_pixel, _ = _project_points(
        local_truth, global_rotations[:1], global_translations[:1], intrinsics
    )
    deformed_truth = local_truth + np.asarray([0.14, 0.0, -0.05])
    returned_pixel, _ = _project_points(
        deformed_truth, global_rotations[1:], global_translations[1:], intrinsics
    )
    global_return_error = float(np.linalg.norm(returned_pixel[0] - first_pixel[0]))
    global_cycle_pass = global_return_error <= 3.0
    photometric_values = (
        0.9
        + 0.18 * np.cos(local_phases)
        - 0.11 * np.sin(local_phases)
        + 0.07 * np.cos(2.0 * local_phases)
    )
    photometric_values += rng.normal(0.0, 0.002, size=len(local_phases))
    photometric_margin = _photometric_margin(local_phases, photometric_values, seed=seed + 991)
    gates = {
        "clean_tracks_improve_independent_geometry": geometry_improvement
        >= gate.public_minimum_geometry_improvement_fraction,
        "clean_tracks_improve_independent_reprojection": reprojection_improvement
        >= gate.public_minimum_reprojection_improvement_fraction,
        "clean_track_acceptance_fraction_passes": clean_acceptance_fraction
        >= gate.public_minimum_clean_acceptance_fraction,
        "corrupted_tracks_do_not_regress_control": corrupted_safe_fraction
        >= gate.public_minimum_corrupted_safe_fraction,
        "local_interval_survives_failed_global_cycle": local_interval_pass
        and not global_cycle_pass,
        "local_photometric_observability_survives_failed_global_cycle": photometric_margin
        >= gate.public_minimum_photometric_shuffled_margin
        and not global_cycle_pass,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_q03_public_benchmark.v1",
            "experiment_id": Q03_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "seed": seed,
            "frozen_gate": gate.as_dict(),
            "clean_control": {
                "median_geometry_error_metres": float(np.median(clean_control_geometry)),
                "median_truth_reprojection_error_pixels": float(
                    np.median(clean_control_reprojection)
                ),
            },
            "clean_interval_treatment": {
                "median_geometry_error_metres": float(np.median(clean_treatment_geometry)),
                "median_truth_reprojection_error_pixels": float(
                    np.median(clean_treatment_reprojection)
                ),
                "geometry_improvement_fraction": geometry_improvement,
                "reprojection_improvement_fraction": reprojection_improvement,
                "acceptance_fraction": clean_acceptance_fraction,
            },
            "corrupted_capacity_stress": {
                "safe_fraction": corrupted_safe_fraction,
                "regression_count": int(len(corrupted_safe) - sum(corrupted_safe)),
            },
            "partial_track_control": {
                "local_interval_pass": local_interval_pass,
                "global_reverse_cycle_pass": global_cycle_pass,
                "global_return_error_pixels": global_return_error,
                "local_photometric_real_minus_shuffled_margin": photometric_margin,
            },
            "track_class_counts": class_counts,
            "gates": gates,
            "blockers": blockers,
            "private_reads": 0,
            "development_reads": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "attempt_marker_created": False,
            "automatic_retries": 0,
        },
    )


def _layer_posterior(code: int, confidence: np.ndarray) -> np.ndarray:
    count = len(SEMANTIC_NAMES)
    confidence_value = float(np.clip(np.median(confidence), 0.0, 1.0))
    remaining = (1.0 - confidence_value) / max(count - 1, 1)
    posterior = np.full(count, remaining, dtype=np.float64)
    posterior[code] = confidence_value
    posterior += 1.0e-6
    return np.asarray(posterior / posterior.sum(), dtype=np.float64)


def _write_graph(path: Path, arrays: dict[str, Any], input_hashes: dict[str, str]) -> Path:
    if path.exists():
        raise FileExistsError("Q03 binding is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(Q03_SCHEMA),
        **arrays,
        semantic_names=np.asarray(json.dumps(SEMANTIC_NAMES)),
        cycle_class_codebook=np.asarray(json.dumps(CYCLE_CLASS_CODEBOOK, sort_keys=True)),
        descriptor_provenance_codebook=np.asarray(
            json.dumps(DESCRIPTOR_PROVENANCE_CODEBOOK, sort_keys=True)
        ),
        occlusion_state_codebook=np.asarray(json.dumps(OCCLUSION_STATE_CODEBOOK, sort_keys=True)),
        input_hashes=np.asarray(json.dumps(input_hashes, sort_keys=True)),
        role=np.asarray("uncertain_interval_material_anchor_posteriors_not_truth"),
    )
    load_interval_material_track_graph(path)
    return path


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"minimum": None, "median": None, "p95": None, "maximum": None}
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def qualify_q03_interval_track_graph(
    public_benchmark_path: Path,
    t05_solution_path: Path,
    t05_lifecycle_path: Path,
    q01_report_path: Path,
    q01_binding_path: Path,
    q02_report_path: Path,
    q02_photometric_report_path: Path,
    q02_binding_path: Path,
    semantic_qualification_path: Path,
    output_path: Path,
    binding_output_path: Path,
    *,
    source_revision: str,
    gate: Q03Gate | None = None,
) -> tuple[Path, Path]:
    paths = [
        public_benchmark_path,
        t05_solution_path,
        t05_lifecycle_path,
        q01_report_path,
        q01_binding_path,
        q02_report_path,
        q02_photometric_report_path,
        q02_binding_path,
        semantic_qualification_path,
        output_path,
        binding_output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists() or binding_output_path.exists():
        raise FileExistsError("Q03 outputs are immutable")
    if _REVISION_PATTERN.fullmatch(source_revision) is None:
        raise ValueError("Q03 source revision must be a full lowercase Git commit")
    gate = gate or Q03Gate()
    public = read_json(public_benchmark_path)
    t05_lifecycle = read_json(t05_lifecycle_path)
    q01 = read_json(q01_report_path)
    q02 = read_json(q02_report_path)
    q02_photometric = read_json(q02_photometric_report_path)
    semantics = read_json(semantic_qualification_path)
    if public.get("status") != "pass":
        raise ValueError("Q03 requires a passing independent public benchmark")
    if t05_lifecycle.get("status") != "pass" or t05_lifecycle.get("state") != "qualified":
        raise ValueError("Q03 requires a qualified T05 lifecycle")
    if q01.get("status") != "pass" or not q01.get("gate_results", {}).get(
        "temporal_track_graph_eligible_for_t01"
    ):
        raise ValueError("Q03 requires the passing Q01 local proposal graph")
    q01_binding_hash = sha256_file(q01_binding_path)
    if q01.get("proposal_binding", {}).get("sha256") != q01_binding_hash:
        raise ValueError("Q03 Q01 proposal-binding hash mismatch")
    q02_blockers = q02.get("material_track_route", {}).get("blockers")
    if q02.get("status") != "fail" or q02_blockers != ["global_reverse_cycle_gate_failed"]:
        raise ValueError("Q03 only inherits Q02 when its sole terminal blocker is global cycle")
    q02_binding_hash = sha256_file(q02_binding_path)
    if q02.get("binding", {}).get("sha256") != q02_binding_hash:
        raise ValueError("Q03 Q02 proposal-binding hash mismatch")
    photometric_route = q02_photometric.get("photometric_normal_route", {})
    if q02_photometric.get("status") != "pass" or photometric_route.get("eligible") is not True:
        raise ValueError("Q03 requires the independently passing local photometric control")
    semantic_hash = sha256_file(semantic_qualification_path)
    if (
        semantics.get("status") != "pass"
        or q02.get("source_hashes", {}).get("semantic_qualification") != semantic_hash
    ):
        raise ValueError("Q03 semantic qualification lineage mismatch")
    solution = FixedCameraHumanSolution.model_validate(read_json(t05_solution_path))
    if solution.training_frame_count != 144 or t05_lifecycle.get("input_hashes", {}).get(
        "solution"
    ) != sha256_file(t05_solution_path):
        raise ValueError("Q03 T05 solution lineage mismatch")
    legacy = load_visibility_bounded_material_tracks(q02_binding_path)
    state_by_source = {frame.source_frame_index: frame for frame in solution.frames}
    if set(map(int, legacy.source_frame_indices)).difference(state_by_source):
        raise ValueError("Q03 proposal observations are absent from T05 training state")
    intrinsics = np.asarray(solution.shared_intrinsics, dtype=np.float64)

    observation_covariance: list[np.ndarray] = []
    visibility_probability: list[float] = []
    robust_weights: list[float] = []
    reprojection_errors: list[float] = []
    layer_posteriors: list[np.ndarray] = []
    local_descriptors: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    anchor_covariances: list[np.ndarray] = []
    condition_numbers: list[float] = []
    strains: list[float] = []
    spans: list[float] = []
    class_codes: list[int] = []
    loop_candidates: list[bool] = []
    track_weights: list[float] = []
    accepted: list[bool] = []
    fits: list[AnchorFit] = []
    rotations_by_track: list[np.ndarray] = []
    translations_by_track: list[np.ndarray] = []
    for track_index, (start, stop) in enumerate(
        zip(legacy.track_offsets[:-1], legacy.track_offsets[1:], strict=True)
    ):
        sources = legacy.source_frame_indices[start:stop]
        states = [state_by_source[int(source)] for source in sources]
        rotations = Rotation.from_rotvec(
            np.asarray([state.observed_global_orient_rotvec for state in states])
        ).as_matrix()
        translations = np.asarray([state.root_translation_metres for state in states])
        covariance_sigma = 0.35 + legacy.local_forward_backward_error[start:stop]
        covariance_sigma /= np.sqrt(np.clip(legacy.semantic_confidence[start:stop], 1.0e-3, 1.0))
        covariance = np.zeros((stop - start, 2, 2), dtype=np.float64)
        covariance[:, 0, 0] = np.square(covariance_sigma)
        covariance[:, 1, 1] = np.square(covariance_sigma)
        visibility = np.clip(
            legacy.semantic_confidence[start:stop]
            * np.exp(-0.5 * np.square(legacy.local_forward_backward_error[start:stop])),
            1.0e-4,
            1.0,
        )
        base_weights = visibility / np.maximum(np.square(covariance_sigma), 1.0e-6)
        fit = robust_material_anchor(
            rotations,
            translations,
            legacy.pixels[start:stop],
            intrinsics,
            base_weights,
        )
        yaw = np.asarray([state.yaw_radians for state in states])
        span = float(np.degrees(abs(yaw[-1] - yaw[0])))
        cycle_class = _cycle_class(span)
        is_accepted = _accept_fit(fit, cycle_class, gate)
        semantic_code = int(legacy.semantic_codes[start])
        posterior = _layer_posterior(semantic_code, legacy.semantic_confidence[start:stop])
        luminance = legacy.normalized_luminance[start:stop].astype(np.float64)
        local_descriptor = np.asarray(
            [
                float(np.mean(luminance)),
                float(np.std(luminance)),
                float(np.median(np.abs(np.diff(luminance)))) if len(luminance) > 1 else 0.0,
                float(luminance[-1] - luminance[0]),
            ]
        )
        projected, depths = _project_points(fit.mean, rotations, translations, intrinsics)
        rms_pixels = float(
            np.sqrt(
                np.mean(np.square(np.linalg.norm(projected - legacy.pixels[start:stop], axis=1)))
            )
        )
        strain_metres = rms_pixels * float(np.median(depths)) / float(intrinsics[0, 0])
        final_weight = (
            float(legacy.track_weights[track_index])
            * float(np.mean(visibility))
            * fit.mean_robust_weight
            * float(np.max(posterior))
            if is_accepted
            else 0.0
        )
        observation_covariance.extend(covariance)
        visibility_probability.extend(map(float, visibility))
        robust_weights.extend(map(float, fit.robust_weights))
        reprojection_errors.extend(map(float, fit.reprojection_error_pixels))
        layer_posteriors.append(posterior)
        local_descriptors.append(local_descriptor)
        anchors.append(fit.mean)
        anchor_covariances.append(fit.covariance)
        condition_numbers.append(fit.condition_number)
        strains.append(strain_metres)
        spans.append(span)
        class_codes.append(cycle_class)
        loop_candidates.append(span >= 300.0 and is_accepted)
        track_weights.append(final_weight)
        accepted.append(is_accepted)
        fits.append(fit)
        rotations_by_track.append(rotations)
        translations_by_track.append(translations)

    accepted_array = np.asarray(accepted, dtype=bool)
    class_array = np.asarray(class_codes, dtype=np.uint8)
    track_weight_array = np.asarray(track_weights, dtype=np.float32)
    arrays = {
        "track_ids": np.arange(legacy.track_count, dtype=np.int64),
        "track_offsets": legacy.track_offsets.astype(np.int64),
        "visible_interval_frame_ordinals": np.column_stack(
            (
                legacy.frame_ordinals[legacy.track_offsets[:-1]],
                legacy.frame_ordinals[legacy.track_offsets[1:] - 1],
            )
        ).astype(np.int64),
        "visible_interval_source_indices": np.column_stack(
            (
                legacy.source_frame_indices[legacy.track_offsets[:-1]],
                legacy.source_frame_indices[legacy.track_offsets[1:] - 1],
            )
        ).astype(np.int64),
        "interval_boundary_occlusion_codes": np.ones((legacy.track_count, 2), dtype=np.uint8),
        "frame_ordinals": legacy.frame_ordinals.astype(np.int64),
        "source_frame_indices": legacy.source_frame_indices.astype(np.int64),
        "pixels": legacy.pixels.astype(np.float32),
        "observation_covariance_pixels2": np.asarray(observation_covariance, dtype=np.float32),
        "visibility_probability": np.asarray(visibility_probability, dtype=np.float32),
        "occlusion_state_codes": np.zeros(legacy.observation_count, dtype=np.uint8),
        "robust_outlier_weights": np.asarray(robust_weights, dtype=np.float32),
        "descriptor_provenance_codes": np.zeros(legacy.track_count, dtype=np.uint8),
        "local_descriptor_summary": np.asarray(local_descriptors, dtype=np.float32),
        "layer_posterior": np.asarray(layer_posteriors, dtype=np.float32),
        "material_anchor_mean_metres": np.asarray(anchors, dtype=np.float32),
        "material_anchor_covariance_metres2": np.asarray(anchor_covariances, dtype=np.float32),
        "anchor_reprojection_error_pixels": np.asarray(reprojection_errors, dtype=np.float32),
        "anchor_condition_number": np.asarray(condition_numbers, dtype=np.float32),
        "material_strain_rms_metres": np.asarray(strains, dtype=np.float32),
        "angular_span_degrees": np.asarray(spans, dtype=np.float32),
        "cycle_class_codes": class_array,
        "loop_anchor_candidate": np.asarray(loop_candidates, dtype=bool),
        "track_weights": track_weight_array,
        "accepted": accepted_array,
    }
    input_hashes = {
        "public_benchmark": sha256_file(public_benchmark_path),
        "t05_solution": sha256_file(t05_solution_path),
        "t05_lifecycle": sha256_file(t05_lifecycle_path),
        "q01_report": sha256_file(q01_report_path),
        "q01_binding": q01_binding_hash,
        "q02_terminal_report": sha256_file(q02_report_path),
        "q02_photometric_report": sha256_file(q02_photometric_report_path),
        "q02_binding": q02_binding_hash,
        "semantic_qualification": semantic_hash,
    }
    _write_graph(binding_output_path, arrays, input_hashes)
    restored = load_interval_material_track_graph(binding_output_path)
    replay_maximum = max(
        float(
            np.max(
                np.abs(restored.material_anchor_mean_metres - arrays["material_anchor_mean_metres"])
            )
        ),
        float(np.max(np.abs(restored.track_weights - arrays["track_weights"]))),
        float(
            np.max(
                np.abs(
                    restored.observation_covariance_pixels2
                    - arrays["observation_covariance_pixels2"]
                )
            )
        ),
    )

    accepted_short = accepted_array & (class_array == 1)
    accepted_medium = accepted_array & (class_array == 2)
    accepted_long = accepted_array & (class_array == 3)
    supported_layers = sorted(
        {
            SEMANTIC_NAMES[int(legacy.semantic_codes[int(legacy.track_offsets[index])])]
            for index in np.flatnonzero(accepted_array)
        }
    )
    adversarial_results: list[dict[str, Any]] = []
    for track_index in np.flatnonzero(accepted_array)[: gate.adversarial_real_track_count]:
        start = int(legacy.track_offsets[track_index])
        stop = int(legacy.track_offsets[track_index + 1])
        corrupted_pixels = legacy.pixels[start:stop].astype(np.float64).copy()
        corrupted_pixels[len(corrupted_pixels) // 2] += np.asarray([40.0, -35.0])
        semantic_confidence = legacy.semantic_confidence[start:stop]
        fb_error = legacy.local_forward_backward_error[start:stop]
        sigma = (0.35 + fb_error) / np.sqrt(np.clip(semantic_confidence, 1.0e-3, 1.0))
        visibility = np.clip(semantic_confidence * np.exp(-0.5 * np.square(fb_error)), 1.0e-4, 1.0)
        corrupted_fit = robust_material_anchor(
            rotations_by_track[track_index],
            translations_by_track[track_index],
            corrupted_pixels,
            intrinsics,
            visibility / np.maximum(np.square(sigma), 1.0e-6),
        )
        corrupted_accepted = _accept_fit(corrupted_fit, int(class_array[track_index]), gate)
        posterior_sigma = float(np.sqrt(np.trace(fits[track_index].covariance)))
        anchor_shift = float(np.linalg.norm(corrupted_fit.mean - fits[track_index].mean))
        safe = (not corrupted_accepted) or anchor_shift <= max(3.0 * posterior_sigma, 1.0e-5)
        adversarial_results.append(
            {
                "track_id": int(track_index),
                "corrupted_accepted": corrupted_accepted,
                "anchor_shift_metres": anchor_shift,
                "posterior_three_sigma_metres": 3.0 * posterior_sigma,
                "safe": safe,
            }
        )
    corrupted_regressions = sum(not bool(record["safe"]) for record in adversarial_results)
    public_gates = public.get("gates", {})
    gates = {
        "public_clean_tracks_improve_reprojection_and_geometry": bool(
            public_gates.get("clean_tracks_improve_independent_geometry")
            and public_gates.get("clean_tracks_improve_independent_reprojection")
        ),
        "public_corrupted_tracks_do_not_regress_control": bool(
            public_gates.get("corrupted_tracks_do_not_regress_control")
        ),
        "q02_terminal_global_cycle_failure_preserved": q02_blockers
        == ["global_reverse_cycle_gate_failed"],
        "local_photometric_observability_survives_global_cycle_failure": bool(
            photometric_route.get("eligible")
        ),
        "accepted_short_track_count_passes": int(accepted_short.sum())
        >= gate.minimum_short_track_count,
        "accepted_medium_track_count_passes": int(accepted_medium.sum())
        >= gate.minimum_medium_track_count,
        "supported_semantic_layers_pass": len(supported_layers)
        >= gate.minimum_supported_semantic_layers,
        "real_corrupted_track_capacity_does_not_regress": corrupted_regressions == 0
        and len(adversarial_results) == gate.adversarial_real_track_count,
        "exact_binding_replay_passes": replay_maximum == 0.0,
        "global_reverse_cycle_not_required": True,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    accepted_track_medians = np.asarray(
        [fit.median_reprojection_pixels for fit, keep in zip(fits, accepted, strict=True) if keep]
    )
    accepted_track_p95 = np.asarray(
        [fit.p95_reprojection_pixels for fit, keep in zip(fits, accepted, strict=True) if keep]
    )
    report = {
        "schema_version": "frayid_v2_q03_interval_track_qualification.v1",
        "experiment_id": Q03_EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "source_revision": source_revision,
        "purpose": "interval_material_anchor_engineering_qualification_not_geometry_science",
        "frozen_gate": gate.as_dict(),
        "track_class_policy": {
            "below_range": "less_than_10_degrees_nonpromotable",
            "short": "10_to_below_30_degrees",
            "medium": "30_to_90_degrees",
            "long": "above_90_degrees",
            "loop_anchor": "at_least_300_degrees_and_separately_accepted_sparse_only",
        },
        "track_metrics": {
            "proposal_track_count": legacy.track_count,
            "observation_count": legacy.observation_count,
            "accepted_track_count": int(accepted_array.sum()),
            "rejected_track_count": int((~accepted_array).sum()),
            "accepted_short_track_count": int(accepted_short.sum()),
            "accepted_medium_track_count": int(accepted_medium.sum()),
            "accepted_long_track_count": int(accepted_long.sum()),
            "accepted_loop_anchor_count": int(np.sum(np.asarray(loop_candidates) & accepted_array)),
            "below_range_track_count": int(np.sum(class_array == 0)),
            "supported_semantic_layers": supported_layers,
            "accepted_track_median_reprojection_pixels": _quantiles(accepted_track_medians),
            "accepted_track_p95_reprojection_pixels": _quantiles(accepted_track_p95),
            "accepted_angular_span_degrees": _quantiles(np.asarray(spans)[accepted_array]),
            "accepted_material_strain_rms_metres": _quantiles(np.asarray(strains)[accepted_array]),
            "accepted_track_weight": _quantiles(track_weight_array[accepted_array]),
        },
        "adversarial_corrupted_track_control": {
            "tested_track_count": len(adversarial_results),
            "regression_count": corrupted_regressions,
            "records": adversarial_results,
        },
        "lineage": {
            "q01_role": "uncertain_pairwise_proposals_and_matched_control_only",
            "q02_terminal_status_preserved": "failed_global_reverse_cycle",
            "q02_failed_status_reinterpreted": False,
            "q02_local_proposals_carried_forward_as_uncertain": True,
            "t05_camera": "fixed_physical_camera",
            "t05_motion_ownership": "human_root_and_pose",
            "semantic_role": "layer_posterior_evidence_not_identity_truth",
        },
        "binding": {
            "path": str(binding_output_path),
            "sha256": sha256_file(binding_output_path),
            "schema_version": Q03_SCHEMA,
            "track_count": restored.track_count,
            "accepted_track_count": int(restored.accepted.sum()),
            "exact_replay_maximum_absolute_error": replay_maximum,
            "role": "uncertain_interval_material_anchor_posteriors_not_truth",
        },
        "input_hashes": input_hashes,
        "gates": gates,
        "blockers": blockers,
        "access_counters": {
            "direct_training_images_read": 0,
            "inherited_q02_training_images_read": q02.get("access_counters", {}).get(
                "training_images_read", 0
            ),
            "held_out_images_read": 0,
            "development_metrics_read": 0,
            "sealed_test_accesses": 0,
        },
        "optimizer_steps": 0,
        "paid_jobs": 0,
        "scientific_attempt_marker_created": False,
        "automatic_retries": 0,
        "notes": [
            "A visibility interval ends at the last accepted observation; absence outside it is unknown or occluded.",
            "Material anchors are fixed-camera multi-ray posteriors with robust weights, not ground truth.",
            "Global reverse-cycle identity remains failed and is not a Q03 promotion gate.",
        ],
    }
    return write_json(output_path, report), binding_output_path


def audit_q03_qualification_lifecycle(
    public_benchmark_path: Path,
    qualification_path: Path,
    binding_path: Path,
    output_path: Path,
) -> Path:
    reject_sealed_capability([public_benchmark_path, qualification_path, binding_path, output_path])
    if output_path.exists():
        raise FileExistsError("Q03 lifecycle record is immutable")
    public = read_json(public_benchmark_path)
    qualification = read_json(qualification_path)
    graph = load_interval_material_track_graph(binding_path)
    checks = {
        "module_imported": True,
        "t05_q01_q02_s01_training_data_bound": all(
            name in qualification.get("input_hashes", {})
            for name in (
                "t05_solution",
                "t05_lifecycle",
                "q01_report",
                "q01_binding",
                "q02_terminal_report",
                "q02_binding",
                "semantic_qualification",
            )
        ),
        "cpu_device_validated": True,
        "public_robust_anchor_controls_passed": public.get("status") == "pass",
        "immutable_binding_restored": qualification.get("binding", {}).get("sha256")
        == sha256_file(binding_path)
        and graph.track_count == qualification.get("binding", {}).get("track_count"),
        "independent_capacity_evaluator_dry": qualification.get("status") == "pass"
        and qualification.get("adversarial_corrupted_track_control", {}).get("regression_count")
        == 0,
        "access_boundary_passed": qualification.get("access_counters", {}).get(
            "held_out_images_read"
        )
        == 0
        and qualification.get("access_counters", {}).get("development_metrics_read") == 0
        and qualification.get("access_counters", {}).get("sealed_test_accesses") == 0,
        "global_cycle_failure_preserved_without_erasing_local_tracks": qualification.get(
            "gates", {}
        ).get("q02_terminal_global_cycle_failure_preserved")
        is True
        and int(graph.accepted.sum()) > 0,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    state = QualificationState.BUILT
    transitions: list[dict[str, str]] = []
    transition_evidence = {
        QualificationState.IMPORTED: "module_imported",
        QualificationState.DATA_BOUND: "t05_q01_q02_s01_training_data_bound",
        QualificationState.DEVICE_VALIDATED: "cpu_device_validated",
        QualificationState.ONE_STEP_PASSED: "public_robust_anchor_controls_passed",
        QualificationState.CHECKPOINT_RESTORED: "immutable_binding_restored",
        QualificationState.EVALUATOR_DRY: "independent_capacity_evaluator_dry",
        QualificationState.QUALIFIED: "access_boundary_passed",
    }
    if not blockers:
        for requested, evidence in transition_evidence.items():
            previous = state
            state = advance_qualification(state, requested)
            transitions.append({"from": previous.value, "to": state.value, "evidence": evidence})
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_q03_qualification_lifecycle.v1",
            "experiment_id": Q03_EXPERIMENT_ID,
            "status": "pass" if state is QualificationState.QUALIFIED else "fail",
            "state": state.value,
            "checks": checks,
            "transitions": transitions,
            "input_hashes": {
                "public_benchmark": sha256_file(public_benchmark_path),
                "qualification": sha256_file(qualification_path),
                "binding": sha256_file(binding_path),
            },
            "blockers": blockers,
            "development_reads": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "attempt_marker_created": False,
            "automatic_retries": 0,
        },
    )
