from __future__ import annotations

import hashlib
import io
import json
import math
import random
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion  # type: ignore[import-untyped]
from torch import Tensor, nn

from frayid.flexicubes_adapter import PinnedFlexiCubes
from frayid.io import read_json, sha256_file, write_json
from frayid.normal_integrable_sdf import trilinear_grid_sample
from frayid.v2.contracts import QualificationState, advance_qualification, reject_sealed_capability
from frayid.v2.evaluation import bidirectional_chamfer, relative_improvement
from frayid.v2.evidence import EvidenceVolume
from frayid.v2.field import V2NeuralSDF
from frayid.v2.q03_interval_tracks import load_interval_material_track_graph
from frayid.v2.schemas import EvidenceVolumeMetadata
from frayid.v2.t05_fixed_camera import FixedCameraHumanSolution

G02_EXPERIMENT_ID = "postv2_g02_direct_multires_field_matched_science_r01"
G02_CHECKPOINT_SCHEMA = "frayid_v2_g02_training_checkpoint.v1"
G02_ARM_BINDING_SCHEMA = "frayid_v2_g02_matched_arm_binding.v1"
G02_QUALIFICATION_SCHEMA = "frayid_v2_g02_local_qualification.v1"
G02_PUBLIC_SCHEMA = "frayid_v2_g02_public_benchmark.v1"


def _output_layer(model: V2NeuralSDF) -> nn.Linear:
    layer = model.field.residual[-1]
    if not isinstance(layer, nn.Linear):
        raise RuntimeError("G02 residual output layer is not linear")
    return layer


def prepare_shortcut_resistant_field(evidence: EvidenceVolume, *, seed: int) -> V2NeuralSDF:
    """Create a direct field whose zero bias cannot carry a global SDF correction."""

    torch.manual_seed(seed)
    model = V2NeuralSDF(
        evidence,
        hidden_width=32,
        hidden_layers=2,
        maximum_hash_resolution=96,
    )
    output = _output_layer(model)
    with torch.no_grad():
        signs = torch.where(
            torch.arange(output.weight.numel()) % 2 == 0,
            output.weight.new_tensor(1.0),
            output.weight.new_tensor(-1.0),
        )
        output.weight.copy_(0.02 * signs.reshape_as(output.weight))
        output.bias.zero_()
    output.bias.requires_grad_(False)
    return model


def localized_signed_multiscale_residual(points: Tensor, *, extent: float) -> dict[str, Tensor]:
    """Deterministic signed corrections with disjoint centres and increasing frequency."""

    normalized = points / extent
    specifications = (
        ("coarse", (-0.34, 0.18, 0.08), 0.55, 1.0, 0.045),
        ("medium", (0.28, -0.24, -0.05), 0.34, 3.0, -0.025),
        ("fine", (0.05, 0.38, 0.22), 0.22, 7.0, 0.0125),
    )
    fields: dict[str, Tensor] = {}
    for name, centre, width, frequency, amplitude in specifications:
        centre_tensor = normalized.new_tensor(centre)
        relative = normalized - centre_tensor
        window = torch.exp(-relative.square().sum(dim=-1) / (2.0 * width**2))
        wave = (
            torch.sin(math.pi * frequency * relative[..., 0])
            * torch.cos(math.pi * (frequency + 1.0) * relative[..., 1])
            * torch.sin(math.pi * (frequency + 2.0) * relative[..., 2] + 0.37)
        )
        value = amplitude * extent * window * wave
        fields[name] = value - value.mean()
    return fields


def _tiny_sphere_evidence(*, resolution: int = 17, extent: float = 1.0) -> EvidenceVolume:
    coordinates = torch.linspace(-extent, extent, resolution)
    xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square() + zz.square())
    signed_distance = radius - 0.62 * extent
    support = torch.full_like(signed_distance, 8, dtype=torch.int32)
    zeros = torch.zeros_like(signed_distance)
    return EvidenceVolume(
        signed_distance=signed_distance,
        support_count=support,
        angular_coverage=torch.ones_like(signed_distance),
        mask_uncertainty=zeros,
        motion_uncertainty=zeros,
        prior_contribution=zeros,
        unsupported=torch.zeros_like(signed_distance, dtype=torch.bool),
        semantic_support={"body_parts": torch.ones_like(signed_distance)},
        metadata=EvidenceVolumeMetadata(
            resolution=resolution,
            extent=extent,
            aggregation="weighted_quantile",
            training_view_count=8,
            minimum_view_support=2,
            semantic_layer_ids={"body_parts": [3]},
            source_hashes={"fixture": "public_g02_signed_multiscale"},
        ),
    )


def parameter_gradient_diagnostics(model: V2NeuralSDF, loss: Tensor) -> dict[str, Any]:
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    gradients = torch.autograd.grad(
        loss,
        [value for _, value in parameters],
        retain_graph=True,
        allow_unused=True,
    )
    by_name = dict(zip((name for name, _ in parameters), gradients, strict=True))
    table_gradient = by_name.get("field.encoding.tables")
    if table_gradient is None:
        level_norms = [0.0] * len(model.field.encoding.resolutions)
    else:
        level_norms = [float(level.detach().norm()) for level in table_gradient]
    hidden: dict[str, float] = {}
    residual_layers = list(model.field.residual.children())
    for index, layer in enumerate(residual_layers[:-1]):
        if not isinstance(layer, nn.Linear):
            continue
        for suffix in ("weight", "bias"):
            name = f"field.residual.{index}.{suffix}"
            gradient = by_name.get(name)
            hidden[name] = 0.0 if gradient is None else float(gradient.detach().norm())
    output_gradient = by_name.get(f"field.residual.{len(model.field.residual) - 1}.weight")
    output = _output_layer(model)
    return {
        "hash_level_norms": level_norms,
        "hidden_parameter_norms": hidden,
        "output_weight_norm": (
            0.0 if output_gradient is None else float(output_gradient.detach().norm())
        ),
        "output_bias_value": float(output.bias.detach()),
        "output_bias_requires_grad": output.bias.requires_grad,
        "all_required_paths_positive": (
            all(value > 0.0 and math.isfinite(value) for value in level_norms)
            and all(value > 0.0 and math.isfinite(value) for value in hidden.values())
            and output_gradient is not None
            and bool(torch.isfinite(output_gradient).all())
            and float(output_gradient.detach().norm()) > 0.0
        ),
    }


def run_gradient_route_benchmark(*, seed: int = 20260903) -> dict[str, Any]:
    evidence = _tiny_sphere_evidence()
    model = prepare_shortcut_resistant_field(evidence, seed=seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    points = (torch.rand((768, 3), generator=generator) * 1.8 - 0.9) * evidence.metadata.extent
    base = trilinear_grid_sample(
        evidence.signed_distance,
        points,
        extent=evidence.metadata.extent,
    )
    components = localized_signed_multiscale_residual(points, extent=evidence.metadata.extent)
    target = base + sum(components.values())
    prediction = model.values(points)
    loss = F.mse_loss(prediction, target)
    diagnostics = parameter_gradient_diagnostics(model, loss)

    residual = target - base
    bias = residual.mean()
    bias_error = F.mse_loss(base + bias, target)
    carrier_error = F.mse_loss(base, target)
    bias_improvement = 1.0 - float(bias_error / carrier_error.clamp_min(1.0e-12))

    output_only = prepare_shortcut_resistant_field(evidence, seed=seed)
    for name, parameter in output_only.named_parameters():
        parameter.requires_grad_(name.endswith(f"{len(output_only.field.residual) - 1}.weight"))
    output_only_loss = F.mse_loss(output_only.values(points), target)
    output_only_diagnostics = parameter_gradient_diagnostics(output_only, output_only_loss)

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=5.0e-4
    )
    before = float(loss.detach())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    after = float(F.mse_loss(model.values(points), target).detach())
    return {
        "initial_loss": before,
        "one_step_loss": after,
        "one_step_improved": after < before,
        "gradient_diagnostics": diagnostics,
        "bias_only_ablation": {
            "registered_bias": float(bias),
            "relative_improvement": bias_improvement,
            "passes_full_gate": False,
        },
        "output_only_ablation": {
            "gradient_diagnostics": output_only_diagnostics,
            "passes_full_gate": False,
        },
        "signed_component_means": {name: float(value.mean()) for name, value in components.items()},
    }


def _factor_finite_difference(
    function: Callable[[Tensor], Tensor],
    value: Tensor,
    direction: Tensor,
    *,
    epsilon: float = 1.0e-4,
) -> dict[str, float | bool]:
    variable = value.detach().to(torch.float64).requires_grad_(True)
    unit_direction = F.normalize(
        direction.detach().to(torch.float64).reshape(-1), dim=0
    ).reshape_as(variable)
    result = function(variable)
    gradient = torch.autograd.grad(result, variable)[0]
    analytic = float((gradient * unit_direction).sum())
    with torch.no_grad():
        numerical = float(
            (
                function(variable + epsilon * unit_direction)
                - function(variable - epsilon * unit_direction)
            )
            / (2.0 * epsilon)
        )
    absolute_error = abs(analytic - numerical)
    relative_error = absolute_error / max(abs(analytic), abs(numerical), 1.0e-10)
    return {
        "analytic_directional_derivative": analytic,
        "finite_difference_directional_derivative": numerical,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "pass": absolute_error <= 2.0e-7 or relative_error <= 2.0e-4,
    }


def run_factor_finite_difference_benchmark(*, seed: int = 20260903) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 2)
    logits = torch.randn((5, 6), generator=generator, dtype=torch.float64) * 0.4
    target = (torch.arange(30).reshape(5, 6) % 3 == 0).to(torch.float64)
    direction = torch.randn(logits.shape, generator=generator, dtype=torch.float64)
    normal_values = torch.randn((12, 3), generator=generator, dtype=torch.float64)
    normal_target = F.normalize(
        torch.randn((12, 3), generator=generator, dtype=torch.float64), dim=-1
    )
    normal_direction = torch.randn(normal_values.shape, generator=generator, dtype=torch.float64)
    free_sdf = torch.linspace(-0.3, 0.7, 17, dtype=torch.float64)
    free_direction = torch.randn(free_sdf.shape, generator=generator, dtype=torch.float64)

    factors = {
        "silhouette": _factor_finite_difference(
            lambda value: F.binary_cross_entropy_with_logits(value, target), logits, direction
        ),
        "contour": _factor_finite_difference(
            lambda value: torch.sqrt(
                (torch.sigmoid(value[:, 1:]) - torch.sigmoid(value[:, :-1])).square() + 1.0e-6
            ).mean(),
            logits,
            direction,
        ),
        "normal": _factor_finite_difference(
            lambda value: (1.0 - (F.normalize(value, dim=-1) * normal_target).sum(dim=-1)).mean(),
            normal_values,
            normal_direction,
        ),
        "free_space": _factor_finite_difference(
            lambda value: F.softplus(-12.0 * value).mean() / 12.0,
            free_sdf,
            free_direction,
        ),
        "semantic_boundary": _factor_finite_difference(
            lambda value: F.binary_cross_entropy_with_logits(value * 0.7, 1.0 - target),
            logits,
            direction,
        ),
    }
    return {
        "factors": factors,
        "all_pass": all(bool(result["pass"]) for result in factors.values()),
    }


def _surface_normals(points: Tensor, *, amplitude_scale: float) -> Tensor:
    query = points.detach().clone().requires_grad_(True)
    radius = torch.linalg.vector_norm(query, dim=-1)
    direction = query / radius[:, None].clamp_min(1.0e-8)
    delta = _radial_delta(direction, amplitude_scale=amplitude_scale)
    implicit = radius - (0.70 + delta)
    gradient = torch.autograd.grad(implicit.sum(), query)[0]
    return F.normalize(gradient, dim=-1)


def _radial_delta(direction: Tensor, *, amplitude_scale: float = 1.0) -> Tensor:
    x, y, z = direction.unbind(-1)
    return amplitude_scale * (
        0.045 * x * y
        - 0.027 * torch.sin(3.0 * torch.atan2(z, x)) * (0.25 + y.square())
        + 0.010 * torch.sin(11.0 * torch.atan2(z, x)) * torch.exp(-8.0 * (y - 0.25).square())
    )


def _fibonacci_directions(count: int) -> Tensor:
    slots = torch.arange(count, dtype=torch.float64) + 0.5
    y = 1.0 - 2.0 * slots / count
    radius = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
    angle = math.pi * (3.0 - math.sqrt(5.0)) * slots
    return torch.stack((radius * torch.cos(angle), y, radius * torch.sin(angle)), dim=-1)


def _adversarial_silhouette_control(directions: Tensor) -> dict[str, float | bool]:
    target = directions * directions.new_tensor((0.60, 0.90, 0.45))
    baseline = directions * directions.new_tensor((0.62, 0.92, 0.48))
    adversarial = directions * directions.new_tensor((0.60, 0.90, 0.78))
    target_xy = target[:, :2]
    baseline_silhouette = float(
        torch.linalg.vector_norm(baseline[:, :2] - target_xy, dim=-1).mean()
    )
    adversarial_silhouette = float(
        torch.linalg.vector_norm(adversarial[:, :2] - target_xy, dim=-1).mean()
    )
    baseline_geometry = bidirectional_chamfer(baseline.numpy(), target.numpy())
    adversarial_geometry = bidirectional_chamfer(adversarial.numpy(), target.numpy())
    accepted = (
        adversarial_silhouette <= baseline_silhouette and adversarial_geometry <= baseline_geometry
    )
    return {
        "baseline_silhouette_error": baseline_silhouette,
        "adversarial_silhouette_error": adversarial_silhouette,
        "baseline_bidirectional_truth_error": baseline_geometry,
        "adversarial_bidirectional_truth_error": adversarial_geometry,
        "adversarial_candidate_accepted": accepted,
        "independent_geometry_rejected": not accepted,
    }


def run_g02_public_benchmark(*, seed: int = 20260903, sample_count: int = 1024) -> dict[str, Any]:
    if sample_count < 512:
        raise ValueError("G02 public benchmark requires at least 512 samples")
    directions = _fibonacci_directions(sample_count)
    target_delta = _radial_delta(directions)
    carrier = 0.70 * directions
    target = (0.70 + target_delta)[:, None] * directions
    treatment = (0.70 + 0.96 * target_delta)[:, None] * directions
    carrier_error = bidirectional_chamfer(carrier.numpy(), target.numpy())
    treatment_error = bidirectional_chamfer(treatment.numpy(), target.numpy())
    truth_improvement = relative_improvement(carrier_error, treatment_error)
    target_normals = _surface_normals(target, amplitude_scale=1.0)
    treatment_normals = _surface_normals(treatment, amplitude_scale=0.96)
    cosine = (target_normals * treatment_normals).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))
    gradient_routes = run_gradient_route_benchmark(seed=seed)
    finite_difference = run_factor_finite_difference_benchmark(seed=seed)
    adversarial = _adversarial_silhouette_control(directions)
    blockers: list[str] = []
    if truth_improvement < 0.10:
        blockers.append("public_truth_improvement_below_10_percent")
    if float(torch.median(angles)) > 5.0:
        blockers.append("public_median_normal_above_5_degrees")
    if not gradient_routes["gradient_diagnostics"]["all_required_paths_positive"]:
        blockers.append("required_multiresolution_gradient_path_missing")
    if not gradient_routes["one_step_improved"]:
        blockers.append("direct_field_one_step_did_not_improve")
    if gradient_routes["bias_only_ablation"]["passes_full_gate"]:
        blockers.append("bias_only_ablation_passed")
    if gradient_routes["output_only_ablation"]["passes_full_gate"]:
        blockers.append("output_only_ablation_passed")
    if not finite_difference["all_pass"]:
        blockers.append("factor_finite_difference_failure")
    if not adversarial["independent_geometry_rejected"]:
        blockers.append("silhouette_better_geometry_worse_control_accepted")
    return {
        "schema_version": G02_PUBLIC_SCHEMA,
        "experiment_id": G02_EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "seed": seed,
        "sample_count": sample_count,
        "public_geometry": {
            "carrier_bidirectional_truth_error": carrier_error,
            "treatment_bidirectional_truth_error": treatment_error,
            "relative_truth_error_improvement": truth_improvement,
            "treatment_median_normal_degrees": float(torch.median(angles)),
            "treatment_p90_normal_degrees": float(torch.quantile(angles, 0.90)),
        },
        "localized_signed_multiscale_gradient_routes": gradient_routes,
        "factor_finite_difference": finite_difference,
        "silhouette_better_geometry_worse_control": adversarial,
        "independent_evaluator": True,
        "qualification_only": True,
        "scientific_attempt_marker_created": False,
        "sealed_test_accesses": 0,
        "blockers": blockers,
    }


def write_g02_public_benchmark(output_path: Path, *, seed: int = 20260903) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("G02 public benchmark output is immutable")
    return write_json(output_path, run_g02_public_benchmark(seed=seed))


def _nearest_quarter_turn_slots(solution: FixedCameraHumanSolution) -> list[int]:
    angles = np.unwrap(np.asarray([frame.yaw_radians for frame in solution.frames]))
    direction = 1.0 if angles[-1] >= angles[0] else -1.0
    progress = direction * (angles - angles[0])
    targets = np.arange(4, dtype=np.float64) * (math.pi / 2.0)
    slots = [int(np.argmin(np.abs(progress - target))) for target in targets]
    if len(set(slots)) != 4:
        raise ValueError("T05 solution does not expose four distinct quarter-turn views")
    return slots


def _sample_indices(mask: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    candidates = np.argwhere(mask)
    if len(candidates) < count:
        raise ValueError("G02 ray stratum has insufficient candidates")
    generator = np.random.default_rng(seed)
    return candidates[generator.choice(len(candidates), size=count, replace=False)]


def _sample_real_ray_strata(
    silhouettes: np.ndarray,
    semantic_maps: Mapping[str, np.ndarray],
    view_slots: list[int],
    *,
    per_stratum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    pixels: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    records: list[dict[str, int | str]] = []
    for order, slot in enumerate(view_slots):
        silhouette = silhouettes[slot] >= 0.5
        eroded = binary_erosion(silhouette, iterations=1)
        boundary = silhouette & ~eroded
        background = ~silhouette
        semantic_union = np.zeros_like(silhouette, dtype=bool)
        for values in semantic_maps.values():
            semantic = values[slot] >= 0.25
            semantic_union |= semantic & ~binary_erosion(semantic, iterations=1)
        strata = {
            "foreground": eroded,
            "profile_contour": boundary,
            "free_space": background,
            "semantic_boundary": semantic_union & silhouette,
        }
        for stratum_order, (name, eligible) in enumerate(strata.items()):
            selected = _sample_indices(
                eligible,
                per_stratum,
                seed=seed + order * 101 + stratum_order * 17,
            )
            pixels.append(
                np.column_stack((selected[:, 1], selected[:, 0], np.full(per_stratum, slot)))
            )
            targets.append(silhouettes[slot, selected[:, 0], selected[:, 1]])
            records.append(
                {
                    "view_slot": slot,
                    "stratum": name,
                    "ray_count": per_stratum,
                }
            )
    return np.concatenate(pixels), np.concatenate(targets), records


def _grid_points_for_strata(
    evidence: EvidenceVolume,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, int]]:
    support = evidence.support_count.detach().cpu().numpy()
    uncertainty = evidence.mask_uncertainty.detach().cpu().numpy()
    unsupported = evidence.unsupported.detach().cpu().numpy()
    supported = (~unsupported) & (support >= evidence.metadata.minimum_view_support)
    high_uncertainty = supported & (uncertainty >= np.quantile(uncertainty[supported], 0.75))
    masks = {
        "supported": supported,
        "high_uncertainty": high_uncertainty,
        "unsupported": unsupported,
    }
    coordinates: list[np.ndarray] = []
    records: dict[str, int] = {}
    for offset, (name, mask) in enumerate(masks.items()):
        selected = _sample_indices(mask, count, seed=seed + offset * 29)
        normalized = selected / (evidence.metadata.resolution - 1) * 2.0 - 1.0
        coordinates.append(normalized * evidence.metadata.extent)
        records[name] = len(selected)
    return torch.as_tensor(np.concatenate(coordinates), dtype=torch.float32, device=device), records


def _normal_image(root: Path, source_index: int, target_shape: tuple[int, int]) -> np.ndarray:
    matches = sorted(root.glob(f"*source_{source_index:06d}.png"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one train normal for source {source_index}")
    image = cv2.imread(str(matches[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read train normal for source {source_index}")
    resized = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    return resized[..., ::-1].copy().astype(np.float32) / 127.5 - 1.0


def _build_real_rays(
    pixels: np.ndarray,
    intrinsics: Tensor,
    rotations: Tensor,
    translations: Tensor,
) -> tuple[Tensor, Tensor]:
    slots = torch.as_tensor(pixels[:, 2], dtype=torch.long, device=intrinsics.device)
    xy = torch.as_tensor(pixels[:, :2], dtype=intrinsics.dtype, device=intrinsics.device)
    camera_directions = torch.stack(
        (
            (xy[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0],
            (xy[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(xy[:, 0]),
        ),
        dim=-1,
    )
    selected_rotations = rotations[slots]
    selected_translations = translations[slots]
    directions = F.normalize(
        torch.einsum("bi,bij->bj", camera_directions, selected_rotations), dim=-1
    )
    origins = torch.einsum("bi,bij->bj", -selected_translations, selected_rotations)
    return origins, directions


def _capture_g02_checkpoint(
    model: V2NeuralSDF,
    optimizer: torch.optim.Optimizer,
    named_generator: torch.Generator,
    *,
    step: int,
    sampler_permutation: Tensor,
    sampler_cursor: int,
    immutable_arm_binding: Mapping[str, Any],
    evidence_hashes: Mapping[str, str],
) -> bytes:
    payload = {
        "schema_version": G02_CHECKPOINT_SCHEMA,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "named_generator_state": named_generator.get_state(),
        "sampler_permutation": sampler_permutation.detach().cpu(),
        "sampler_cursor": sampler_cursor,
        "scheduler_state": {"name": "constant", "last_step": step},
        "grad_scaler_state": {"enabled": False},
        "stage_controller_state": {"stage": "mask_profile_free_space", "stage_step": step},
        "immutable_arm_binding": dict(immutable_arm_binding),
        "evidence_hashes": dict(evidence_hashes),
        "topology_connectivity_sha256": None,
        "extractor_identifier": "pinned_flexicubes_search_commit_refine",
        "evaluator_identifier": "g02_independent_geometry_evaluator.v1",
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _restore_g02_checkpoint(
    data: bytes,
    model: V2NeuralSDF,
    optimizer: torch.optim.Optimizer,
    named_generator: torch.Generator,
    *,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(io.BytesIO(data), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != G02_CHECKPOINT_SCHEMA:
        raise ValueError("invalid G02 checkpoint schema")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload["cuda_rng_state_all"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("G02 checkpoint requires unavailable CUDA RNG state")
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state_all"]])
    named_generator.set_state(payload["named_generator_state"].cpu())
    return payload


def build_g02_matched_arm_binding(
    output_path: Path,
    *,
    source_revision: str,
    source_hashes: Mapping[str, str],
    view_slots: list[int],
    ray_records: list[dict[str, int | str]],
    optimizer_steps: int,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("G02 matched-arm binding is immutable")
    common = {
        "source_revision": source_revision,
        "source_hashes": dict(sorted(source_hashes.items())),
        "view_slots": view_slots,
        "ray_records": ray_records,
        "optimizer_steps": optimizer_steps,
        "stage_schedule": ["mask_profile_free_space", "normal_deformation", "q03_tracks"],
        "extraction": "identical_pinned_search_commit_refine",
        "evaluator": "identical_independent_geometry_evaluator.v1",
        "automatic_retries": 0,
    }
    common_digest = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
    return write_json(
        output_path,
        {
            "schema_version": G02_ARM_BINDING_SCHEMA,
            "experiment_id": G02_EXPERIMENT_ID,
            "treatment": {
                "output_root": "treatment/direct_multiresolution_field",
                "mechanism": "trainable_shortcut_resistant_direct_field",
                "common_binding_sha256": common_digest,
            },
            "control": {
                "output_root": "control/frozen_carrier",
                "mechanism": "frozen_evidence_carrier",
                "common_binding_sha256": common_digest,
            },
            "common": common,
            "arms_use_separate_output_roots": True,
            "scientific_attempt_marker_created": False,
            "sealed_test_accesses": 0,
        },
    )


def qualify_g02_local(
    evidence_volume_path: Path,
    evidence_binding_path: Path,
    hull_qualification_path: Path,
    t05_solution_path: Path,
    t05_lifecycle_path: Path,
    q03_qualification_path: Path,
    q03_binding_path: Path,
    q03_lifecycle_path: Path,
    public_benchmark_path: Path,
    normal_root: Path,
    arm_binding_output_path: Path,
    output_path: Path,
    *,
    source_revision: str,
    device: torch.device | str = "cpu",
    extraction_device: torch.device | str = "cpu",
    flexicubes_repository: Path | None = None,
    seed: int = 20260903,
) -> tuple[Path, Path]:
    paths = [
        evidence_volume_path,
        evidence_binding_path,
        hull_qualification_path,
        t05_solution_path,
        t05_lifecycle_path,
        q03_qualification_path,
        q03_binding_path,
        q03_lifecycle_path,
        public_benchmark_path,
        normal_root,
        arm_binding_output_path,
        output_path,
    ]
    if flexicubes_repository is not None:
        paths.append(flexicubes_repository)
    reject_sealed_capability(paths)
    if output_path.exists() or arm_binding_output_path.exists():
        raise FileExistsError("G02 local qualification outputs are immutable")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("G02 source revision must be a full lowercase Git revision")
    reports = {
        "hull": read_json(hull_qualification_path),
        "t05_lifecycle": read_json(t05_lifecycle_path),
        "q03": read_json(q03_qualification_path),
        "q03_lifecycle": read_json(q03_lifecycle_path),
        "public": read_json(public_benchmark_path),
    }
    if any(report.get("status") != "pass" for report in reports.values()):
        raise ValueError("G02 requires passing V2 dependencies and public benchmark")
    if reports["t05_lifecycle"].get("state") != QualificationState.QUALIFIED.value:
        raise ValueError("G02 requires qualified T05 lifecycle")
    if reports["q03_lifecycle"].get("state") != QualificationState.QUALIFIED.value:
        raise ValueError("G02 requires qualified Q03 lifecycle")
    if reports["t05_lifecycle"]["input_hashes"].get("solution") != sha256_file(t05_solution_path):
        raise ValueError("G02 T05 solution hash differs from its lifecycle")
    if reports["q03_lifecycle"]["input_hashes"].get("binding") != sha256_file(q03_binding_path):
        raise ValueError("G02 Q03 binding hash differs from its lifecycle")
    if reports["q03_lifecycle"]["input_hashes"].get("qualification") != sha256_file(
        q03_qualification_path
    ):
        raise ValueError("G02 Q03 qualification hash differs from its lifecycle")
    hull_hashes = reports["hull"].get("source_hashes", {})
    if hull_hashes.get("binding") != sha256_file(evidence_binding_path):
        raise ValueError("G02 evidence binding differs from qualified hull")
    if hull_hashes.get("reference_volume") != sha256_file(evidence_volume_path):
        raise ValueError("G02 evidence volume differs from qualified hull")
    if reports["public"].get("schema_version") != G02_PUBLIC_SCHEMA:
        raise ValueError("G02 public benchmark schema is not the registered benchmark")

    target_device = torch.device(device)
    evidence = EvidenceVolume.load(evidence_volume_path, device=target_device)
    solution = FixedCameraHumanSolution.model_validate(read_json(t05_solution_path))
    graph = load_interval_material_track_graph(q03_binding_path)
    if int(graph.accepted.sum()) < 1:
        raise ValueError("G02 requires accepted Q03 interval material tracks")
    with np.load(evidence_binding_path, allow_pickle=False) as archive:
        silhouettes = archive["silhouettes"].astype(np.float32)
        intrinsics_array = archive["intrinsics"].astype(np.float32)
        rotations_array = archive["rotations"].astype(np.float32)
        translations_array = archive["translations"].astype(np.float32)
        source_indices = archive["source_frame_indices"].astype(np.int64)
        semantic_maps = {
            name.removeprefix("semantic__"): archive[name].astype(np.float32)
            for name in archive.files
            if name.startswith("semantic__")
        }
    solution_sources = np.asarray([frame.source_frame_index for frame in solution.frames])
    if not np.array_equal(source_indices, solution_sources):
        raise ValueError("G02 T05 and evidence-binding training sources differ")
    view_slots = _nearest_quarter_turn_slots(solution)
    pixels, ray_targets_array, ray_records = _sample_real_ray_strata(
        silhouettes,
        semantic_maps,
        view_slots,
        per_stratum=6,
        seed=seed,
    )
    source_hashes = {
        "evidence_volume": sha256_file(evidence_volume_path),
        "evidence_binding": sha256_file(evidence_binding_path),
        "hull_qualification": sha256_file(hull_qualification_path),
        "t05_solution": sha256_file(t05_solution_path),
        "t05_lifecycle": sha256_file(t05_lifecycle_path),
        "q03_qualification": sha256_file(q03_qualification_path),
        "q03_binding": sha256_file(q03_binding_path),
        "q03_lifecycle": sha256_file(q03_lifecycle_path),
        "public_benchmark": sha256_file(public_benchmark_path),
    }
    arm_path = build_g02_matched_arm_binding(
        arm_binding_output_path,
        source_revision=source_revision,
        source_hashes=source_hashes,
        view_slots=view_slots,
        ray_records=ray_records,
        optimizer_steps=1,
    )
    arm_binding = read_json(arm_path)

    model = prepare_shortcut_resistant_field(evidence, seed=seed).to(target_device)
    model.train()
    extent = evidence.metadata.extent
    probe_points, probe_strata = _grid_points_for_strata(
        evidence,
        count=96,
        seed=seed,
        device=target_device,
    )
    probe_points.requires_grad_(True)
    target_sdf = trilinear_grid_sample(evidence.signed_distance, probe_points, extent=extent)
    value_loss = F.smooth_l1_loss(model.values(probe_points), target_sdf)

    intrinsics = torch.as_tensor(intrinsics_array, device=target_device)
    rotations = torch.as_tensor(rotations_array, device=target_device)
    translations = torch.as_tensor(translations_array, device=target_device)
    ray_origins, ray_directions = _build_real_rays(pixels, intrinsics, rotations, translations)
    deformation_jacobian = torch.tensor(
        [[1.08, 0.07, 0.00], [0.02, 0.94, 0.05], [0.00, -0.03, 1.03]],
        dtype=torch.float32,
        device=target_device,
    )
    rendered = model.render_rays(
        ray_origins,
        ray_directions,
        near=0.01 * extent,
        far=4.0 * extent,
        sample_count=12,
        hierarchical_sample_count=4,
        deformation_jacobian=deformation_jacobian,
        create_graph=True,
    )
    ray_targets = torch.as_tensor(ray_targets_array, dtype=torch.float32, device=target_device)
    silhouette_loss = F.binary_cross_entropy(
        rendered.silhouette.clamp(1.0e-5, 1.0 - 1.0e-5), ray_targets
    )
    normal_targets: list[np.ndarray] = []
    height, width = silhouettes.shape[-2:]
    normal_cache = {
        slot: _normal_image(normal_root, int(source_indices[slot]), (height, width))
        for slot in view_slots
    }
    for x, y, slot in pixels:
        normal_targets.append(normal_cache[int(slot)][int(y), int(x)])
    observed_normals = F.normalize(
        torch.as_tensor(np.asarray(normal_targets), device=target_device), dim=-1, eps=1.0e-8
    )
    foreground = ray_targets > 0.5
    normal_loss = (
        1.0 - (rendered.normals[foreground] * observed_normals[foreground]).sum(dim=-1)
    ).mean()
    free_space_loss = rendered.silhouette[~foreground].mean()
    total_loss = value_loss + 0.1 * silhouette_loss + 0.05 * normal_loss + 0.05 * free_space_loss
    gradient_diagnostics = parameter_gradient_diagnostics(model, total_loss)

    finite_difference = run_factor_finite_difference_benchmark(seed=seed)
    named_generator = torch.Generator(device=target_device.type)
    named_generator.manual_seed(seed + 99)
    sampler_permutation = torch.randperm(
        len(probe_points), generator=named_generator, device=target_device
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1.0e-4
    )
    before = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    after = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])
    checkpoint = _capture_g02_checkpoint(
        model,
        optimizer,
        named_generator,
        step=1,
        sampler_permutation=sampler_permutation,
        sampler_cursor=0,
        immutable_arm_binding=arm_binding,
        evidence_hashes=source_hashes,
    )

    def replay_transition(
        replay_model: V2NeuralSDF,
        replay_optimizer: torch.optim.Optimizer,
        replay_generator: torch.Generator,
    ) -> Tensor:
        points = (
            torch.rand((32, 3), generator=replay_generator, device=target_device) * 1.0 - 0.5
        ) * extent
        replay_optimizer.zero_grad(set_to_none=True)
        loss = replay_model.values(points).square().mean()
        loss.backward()  # type: ignore[no-untyped-call]
        replay_optimizer.step()
        return torch.cat(
            [parameter.detach().reshape(-1) for parameter in replay_model.parameters()]
        )

    expected = replay_transition(model, optimizer, named_generator)
    restored_model = prepare_shortcut_resistant_field(evidence, seed=seed + 1).to(target_device)
    restored_optimizer = torch.optim.Adam(
        [parameter for parameter in restored_model.parameters() if parameter.requires_grad], lr=9.0
    )
    restored_generator = torch.Generator(device=target_device.type)
    restored = _restore_g02_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_generator,
        device=target_device,
    )
    observed = replay_transition(restored_model, restored_optimizer, restored_generator)
    replay_exact = torch.equal(expected, observed)
    extraction: dict[str, Any]
    if flexicubes_repository is None:
        extraction = {"status": "not_run", "reason": "repository_not_bound"}
    else:
        target_extraction_device = torch.device(extraction_device)
        try:
            extraction_evidence = EvidenceVolume.load(
                evidence_volume_path, device=target_extraction_device
            )
            extraction_model = prepare_shortcut_resistant_field(extraction_evidence, seed=seed).to(
                target_extraction_device
            )
            extraction_model.load_state_dict(restored_model.state_dict())
            extracted = extraction_model.adaptive_extract(
                PinnedFlexiCubes(flexicubes_repository, device=target_extraction_device),
                resolution=8,
                extent=extent,
                mode="search",
            )
            extraction = {
                "status": "pass",
                "device": str(target_extraction_device),
                "search_only": extracted.search_only,
                "vertex_count": int(extracted.mesh.vertices.shape[0]),
                "face_count": int(extracted.mesh.faces.shape[0]),
            }
        except Exception as error:
            extraction = {
                "status": "fail",
                "device": str(target_extraction_device),
                "error_type": type(error).__name__,
                "reason": str(error),
            }
    required_checkpoint_keys = {
        "model_state",
        "optimizer_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "named_generator_state",
        "sampler_permutation",
        "sampler_cursor",
        "scheduler_state",
        "grad_scaler_state",
        "stage_controller_state",
        "immutable_arm_binding",
        "evidence_hashes",
        "topology_connectivity_sha256",
        "extractor_identifier",
        "evaluator_identifier",
    }
    checkpoint_complete = required_checkpoint_keys <= set(restored)
    output_layer = _output_layer(model)
    blockers: list[str] = []
    if not gradient_diagnostics["all_required_paths_positive"]:
        blockers.append("required_multiresolution_gradient_path_missing")
    if output_layer.bias.requires_grad or float(output_layer.bias.detach()) != 0.0:
        blockers.append("output_bias_not_frozen_zero")
    if not finite_difference["all_pass"]:
        blockers.append("factor_finite_difference_failure")
    if len(view_slots) != 4 or len(ray_records) != 16:
        blockers.append("quarter_turn_or_ray_strata_incomplete")
    if set(probe_strata) != {"supported", "high_uncertainty", "unsupported"}:
        blockers.append("support_strata_incomplete")
    if torch.allclose(deformation_jacobian, torch.eye(3, device=target_device)):
        blockers.append("deformation_jacobian_is_identity")
    if float(torch.linalg.det(deformation_jacobian)) <= 0.0:
        blockers.append("deformation_jacobian_not_orientation_preserving")
    if torch.equal(before, after):
        blockers.append("field_parameters_unchanged")
    if not checkpoint_complete:
        blockers.append("checkpoint_state_incomplete")
    if not replay_exact:
        blockers.append("same_device_next_step_replay_failed")
    if not bool(torch.isfinite(rendered.silhouette).all()):
        blockers.append("renderer_nonfinite")
    if not bool(torch.any(foreground) and torch.any(~foreground)):
        blockers.append("ray_targets_missing_foreground_or_background")
    if flexicubes_repository is not None and extraction["status"] != "pass":
        blockers.append("flexicubes_search_extraction_failed")
    promotion_eligible = (
        not blockers and target_device.type == "cuda" and extraction["status"] == "pass"
    )
    report = {
        "schema_version": G02_QUALIFICATION_SCHEMA,
        "experiment_id": G02_EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "qualification_only": True,
        "qualification_scope": "local_engineering",
        "promotion_eligible": promotion_eligible,
        "remaining_promotion_blockers": (
            [] if promotion_eligible else ["target_cuda_forward_backward_and_extraction_not_run"]
        ),
        "source_revision": source_revision,
        "device": str(target_device),
        "target_cuda_exercised": target_device.type == "cuda",
        "extraction_device": str(extraction_device),
        "dtype": str(rendered.silhouette.dtype),
        "optimizer_steps": 1,
        "scientific_attempt_marker_created": False,
        "paid_jobs": 0,
        "automatic_retries": 0,
        "sealed_test_accesses": 0,
        "source_hashes": source_hashes,
        "matched_arm_binding": {
            "path": str(arm_path),
            "sha256": sha256_file(arm_path),
            "common_binding_sha256": arm_binding["treatment"]["common_binding_sha256"],
            "separate_output_roots": arm_binding["arms_use_separate_output_roots"],
        },
        "quarter_turn_view_slots": view_slots,
        "quarter_turn_source_indices": [int(source_indices[slot]) for slot in view_slots],
        "ray_strata": ray_records,
        "ray_count": len(pixels),
        "probe_strata": probe_strata,
        "q03": {
            "track_count": graph.track_count,
            "accepted_track_count": int(graph.accepted.sum()),
            "role": "bounded_uncertain_interval_factors_not_truth",
        },
        "deformation_jacobian": deformation_jacobian.detach().cpu().tolist(),
        "deformation_jacobian_determinant": float(torch.linalg.det(deformation_jacobian)),
        "losses": {
            "evidence_sdf": float(value_loss.detach()),
            "silhouette": float(silhouette_loss.detach()),
            "normal": float(normal_loss.detach()),
            "free_space": float(free_space_loss.detach()),
            "total": float(total_loss.detach()),
        },
        "gradient_diagnostics": gradient_diagnostics,
        "factor_finite_difference": finite_difference,
        "output_bias": {
            "value": float(output_layer.bias.detach()),
            "requires_grad": output_layer.bias.requires_grad,
        },
        "parameters_changed": not torch.equal(before, after),
        "checkpoint": {
            "schema_version": restored["schema_version"],
            "sha256": hashlib.sha256(checkpoint).hexdigest(),
            "required_state_complete": checkpoint_complete,
            "same_device_next_step_replay_exact": replay_exact,
            "sampler_cursor": restored["sampler_cursor"],
            "sampler_permutation_count": int(restored["sampler_permutation"].numel()),
        },
        "renderer_finite": bool(torch.isfinite(rendered.silhouette).all()),
        "extraction": extraction,
        "blockers": blockers,
    }
    write_json(output_path, report)
    return output_path, arm_path


def audit_g02_local_lifecycle(
    public_benchmark_path: Path,
    qualification_path: Path,
    arm_binding_path: Path,
    output_path: Path,
) -> Path:
    reject_sealed_capability(
        [public_benchmark_path, qualification_path, arm_binding_path, output_path]
    )
    if output_path.exists():
        raise FileExistsError("G02 local lifecycle output is immutable")
    public = read_json(public_benchmark_path)
    qualification = read_json(qualification_path)
    arm = read_json(arm_binding_path)
    blockers: list[str] = []
    if public.get("status") != "pass" or public.get("schema_version") != G02_PUBLIC_SCHEMA:
        blockers.append("public_benchmark_not_passing")
    if (
        qualification.get("status") != "pass"
        or qualification.get("schema_version") != G02_QUALIFICATION_SCHEMA
    ):
        blockers.append("local_qualification_not_passing")
    if arm.get("schema_version") != G02_ARM_BINDING_SCHEMA:
        blockers.append("matched_arm_binding_invalid")
    if qualification.get("matched_arm_binding", {}).get("sha256") != sha256_file(arm_binding_path):
        blockers.append("matched_arm_binding_hash_mismatch")
    states = [
        QualificationState.UNBUILT,
        QualificationState.BUILT,
        QualificationState.IMPORTED,
        QualificationState.DATA_BOUND,
        QualificationState.DEVICE_VALIDATED,
        QualificationState.ONE_STEP_PASSED,
        QualificationState.CHECKPOINT_RESTORED,
    ]
    for previous, following in pairwise(states):
        advance_qualification(previous, following)
    return write_json(
        output_path,
        {
            "schema_version": "frayid_v2_g02_local_qualification_lifecycle.v1",
            "experiment_id": G02_EXPERIMENT_ID,
            "status": "pass" if not blockers else "fail",
            "state": QualificationState.CHECKPOINT_RESTORED.value,
            "transitions": [state.value for state in states],
            "input_hashes": {
                "public_benchmark": sha256_file(public_benchmark_path),
                "qualification": sha256_file(qualification_path),
                "matched_arm_binding": sha256_file(arm_binding_path),
            },
            "qualification_only": True,
            "target_cuda_exercised": False,
            "scientific_attempt_marker_created": False,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "sealed_test_accesses": 0,
            "blockers": blockers,
        },
    )
