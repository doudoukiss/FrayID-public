from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from frayid.geometry import linear_blend_skinning, vertex_normals
from frayid.io import read_json, sha256_file, write_json
from frayid.renderer import (
    differentiable_boundary_loss,
    render_soft_mesh,
    silhouette_loss,
)
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.l03_open_layers import (
    L03_EXPERIMENT_ID,
    L03_GARMENT_LAYERS,
    boundary_loop_audit,
    make_open_wrinkled_tube,
)

L03_TRAINING_PUBLIC_SCHEMA = "frayid_v2_l03_training_public_benchmark.v1"
L03_TRAINING_QUALIFICATION_PLAN_SCHEMA = "frayid_v2_l03_training_qualification_plan.v1"
L03_TRAINING_QUALIFICATION_SCHEMA = "frayid_v2_l03_training_qualification.v1"


class LayerDisplacementModel(nn.Module):
    """Fixed-connectivity open surface with bounded outward scalar displacement."""

    base_vertices: Tensor
    faces: Tensor
    directions: Tensor

    def __init__(
        self,
        vertices: Tensor,
        faces: Tensor,
        *,
        maximum_displacement_metres: float,
        outward_directions: Tensor | None = None,
    ) -> None:
        super().__init__()
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("L03 displacement vertices must have shape [V,3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("L03 displacement faces must have shape [F,3]")
        if maximum_displacement_metres <= 0.0:
            raise ValueError("L03 maximum displacement must be positive")
        normals = (
            vertex_normals(vertices, faces)
            if outward_directions is None
            else F.normalize(outward_directions, dim=-1, eps=1.0e-8)
        )
        if normals.shape != vertices.shape:
            raise ValueError("L03 outward directions must match vertices")
        if not torch.isfinite(normals).all():
            raise ValueError("L03 displacement directions must be finite")
        self.register_buffer("base_vertices", vertices.detach().clone())
        self.register_buffer("faces", faces.detach().clone())
        self.register_buffer("directions", normals.detach().clone())
        self.raw_displacement = nn.Parameter(vertices.new_zeros(len(vertices)))
        self.maximum_displacement_metres = float(maximum_displacement_metres)

    def bounded_displacement(self) -> Tensor:
        return self.raw_displacement.clamp(0.0, self.maximum_displacement_metres)

    def forward(self) -> Tensor:
        return self.base_vertices + self.directions * self.bounded_displacement()[:, None]

    @torch.no_grad()
    def project_parameters(self) -> None:
        self.raw_displacement.clamp_(0.0, self.maximum_displacement_metres)


def _unique_edges(faces: Tensor) -> Tensor:
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0)
    unique_edges: Tensor = torch.unique(torch.sort(edges, dim=1).values, dim=0)
    return unique_edges


def layer_offset_regularization(model: LayerDisplacementModel) -> tuple[Tensor, Tensor]:
    displacement = model.bounded_displacement()
    edges = _unique_edges(model.faces)
    smoothness = (displacement[edges[:, 0]] - displacement[edges[:, 1]]).square().mean()
    magnitude = displacement.square().mean()
    return smoothness, magnitude


def run_l03_training_public_benchmark() -> dict[str, Any]:
    """Prove bounded outward fitting without changing registered open topology."""

    vertices, faces = make_open_wrinkled_tube(
        y_minimum=-0.5,
        y_maximum=0.5,
        base_radius=0.28,
        wrinkle_amplitude=0.015,
        phase=0.4,
        ring_count=9,
        segment_count=24,
    )
    base = torch.as_tensor(vertices, dtype=torch.float64)
    triangles = torch.as_tensor(faces, dtype=torch.long)
    radial = base.clone()
    radial[:, 1] = 0.0
    radial = F.normalize(radial, dim=-1)
    model = LayerDisplacementModel(
        base,
        triangles,
        maximum_displacement_metres=0.08,
        outward_directions=radial,
    )
    target = model.base_vertices + model.directions * 0.04
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = float(F.mse_loss(model(), target).detach())
    finite_gradients = True
    positive_gradient = False
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        prediction = model()
        smoothness, magnitude = layer_offset_regularization(model)
        loss = F.mse_loss(prediction, target) + 0.01 * smoothness + 0.001 * magnitude
        loss.backward()  # type: ignore[no-untyped-call]
        gradient = model.raw_displacement.grad
        finite_gradients &= gradient is not None and bool(torch.isfinite(gradient).all())
        positive_gradient |= (
            gradient is not None and float(torch.linalg.vector_norm(gradient)) > 0.0
        )
        optimizer.step()
        model.project_parameters()
    final = float(F.mse_loss(model(), target).detach())
    final_vertices = model().detach().cpu().numpy()
    final_audit = boundary_loop_audit(final_vertices, faces)
    state = copy.deepcopy(model.state_dict())
    restored = LayerDisplacementModel(
        base,
        triangles,
        maximum_displacement_metres=0.08,
        outward_directions=radial,
    )
    restored.load_state_dict(state)
    exact_restore = torch.equal(model(), restored())
    minimum_radial_change = float(
        np.min(
            np.linalg.norm(final_vertices[:, [0, 2]], axis=1)
            - np.linalg.norm(vertices[:, [0, 2]], axis=1)
        )
    )
    gates = {
        "objective_relative_reduction": 1.0 - final / initial >= 0.90,
        "finite_gradients": finite_gradients,
        "active_displacement_gradient": positive_gradient,
        "outward_only": float(model.bounded_displacement().min().detach()) >= 0.0
        and minimum_radial_change >= 0.0,
        "maximum_displacement_respected": float(model.bounded_displacement().max().detach())
        <= 0.08,
        "registered_open_topology_preserved": final_audit["component_count"] == 1
        and final_audit["boundary_loop_count"] == 2
        and not final_audit["watertight"]
        and final_audit["winding_consistent"]
        and not final_audit["exact_self_intersections"],
        "exact_checkpoint_restore": exact_restore,
    }
    return {
        "schema_version": L03_TRAINING_PUBLIC_SCHEMA,
        "experiment_id": L03_EXPERIMENT_ID,
        "status": "pass" if all(gates.values()) else "fail",
        "objective": {
            "initial": initial,
            "final": final,
            "relative_reduction": 1.0 - final / initial,
        },
        "displacement": {
            "minimum_metres": float(model.bounded_displacement().min().detach()),
            "median_metres": float(model.bounded_displacement().median().detach()),
            "maximum_metres": float(model.bounded_displacement().max().detach()),
            "minimum_radial_change_metres": minimum_radial_change,
        },
        "topology": final_audit,
        "gates": gates,
        "provenance": {
            "private_records_read": 0,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "synthetic_optimizer_steps": 40,
            "scientific_attempt_marker_created": False,
        },
    }


def write_l03_training_public_benchmark(output_path: Path) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("L03 training public benchmark output is immutable")
    return write_json(output_path, run_l03_training_public_benchmark())


def _load_initial_layer(path: Path) -> dict[str, np.ndarray]:
    required = {
        "vertices",
        "faces",
        "skinning_weights",
        "nearest_body_vertex",
        "source_level_set_vertex_indices",
    }
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError("L03 training layer artifact is incomplete")
        values = {name: np.asarray(archive[name]) for name in required}
    vertices = values["vertices"]
    faces = values["faces"]
    weights = values["skinning_weights"]
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or weights.shape != (len(vertices), 24)
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5)
    ):
        raise ValueError("L03 training layer arrays are invalid")
    return values


def build_l03_training_qualification_plan(
    *,
    public_training_report_path: Path,
    initialization_report_path: Path,
    upper_layer_path: Path,
    lower_layer_path: Path,
    semantic_support_plan_path: Path,
    semantic_inputs_path: Path,
    d03_train_evidence_plan_path: Path,
    d03_mesh_path: Path,
    joint_transforms_path: Path,
    t05_solution_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Bind a four-phase real engineering step without starting L03 science."""

    paths = [
        public_training_report_path,
        initialization_report_path,
        upper_layer_path,
        lower_layer_path,
        semantic_support_plan_path,
        semantic_inputs_path,
        d03_train_evidence_plan_path,
        d03_mesh_path,
        joint_transforms_path,
        t05_solution_path,
    ]
    reject_sealed_capability(paths)
    if len(source_revision) != 40:
        raise ValueError("L03 training qualification source revision must be a full commit hash")
    public = read_json(public_training_report_path)
    initialization = read_json(initialization_report_path)
    support_plan = read_json(semantic_support_plan_path)
    evidence_plan = read_json(d03_train_evidence_plan_path)
    t05 = read_json(t05_solution_path)
    if public.get("status") != "pass" or public.get("schema_version") != L03_TRAINING_PUBLIC_SCHEMA:
        raise ValueError("L03 training qualification requires its passing public benchmark")
    if initialization.get("status") != "pass":
        raise ValueError("L03 training qualification requires its passing real initialization")
    for layer, path in zip(L03_GARMENT_LAYERS, (upper_layer_path, lower_layer_path), strict=True):
        if initialization["artifacts"][layer]["sha256"] != sha256_file(path):
            raise ValueError(f"L03 training {layer} artifact differs from initialization")
        _load_initial_layer(path)
    if (
        support_plan.get("status") != "real_semantic_support_planned"
        or support_plan["input_hashes"]["semantic_inputs"] != sha256_file(semantic_inputs_path)
        or support_plan["input_hashes"]["t05_solution"] != sha256_file(t05_solution_path)
    ):
        raise ValueError("L03 training qualification semantic binding changed")
    if (
        evidence_plan.get("status") != "train_evidence_transfer_planned"
        or initialization["immutable_body"]["mesh_sha256"] != sha256_file(d03_mesh_path)
        or evidence_plan["input_hashes"]["joint_transforms"] != sha256_file(joint_transforms_path)
        or evidence_plan["input_hashes"]["t05_solution"] != sha256_file(t05_solution_path)
    ):
        raise ValueError("L03 training qualification pose binding changed")
    with np.load(semantic_inputs_path, allow_pickle=False) as semantics:
        sources = semantics["source_frame_indices"].astype(np.int64)
        image_shape = tuple(int(value) for value in semantics["bound_image_shape"])
        if any(f"semantic__{layer}" not in semantics.files for layer in L03_GARMENT_LAYERS):
            raise ValueError("L03 training qualification lacks garment semantic maps")
    t05_sources = np.asarray([frame["source_frame_index"] for frame in t05["frames"]])
    if not np.array_equal(sources, t05_sources) or image_shape != (256, 165):
        raise ValueError("L03 training qualification semantics and T05 frames do not align")
    with np.load(joint_transforms_path, allow_pickle=False) as transforms:
        transform_sources = transforms["source_frame_indices"].astype(np.int64)
        transform_values = transforms["transforms"]
    if transform_values.shape[1:] != (24, 4, 4) or not set(sources).issubset(
        set(transform_sources)
    ):
        raise ValueError("L03 training qualification joint transforms are incomplete")
    return {
        "schema_version": L03_TRAINING_QUALIFICATION_PLAN_SCHEMA,
        "experiment_id": L03_EXPERIMENT_ID,
        "status": "local_training_qualification_planned",
        "source_revision": source_revision,
        "input_paths": {
            "public_training_report": str(public_training_report_path),
            "initialization_report": str(initialization_report_path),
            "upper_clothing": str(upper_layer_path),
            "lower_clothing": str(lower_layer_path),
            "semantic_support_plan": str(semantic_support_plan_path),
            "semantic_inputs": str(semantic_inputs_path),
            "d03_train_evidence_plan": str(d03_train_evidence_plan_path),
            "d03_mesh": str(d03_mesh_path),
            "joint_transforms": str(joint_transforms_path),
            "t05_solution": str(t05_solution_path),
        },
        "input_hashes": {
            name: sha256_file(path)
            for name, path in zip(
                (
                    "public_training_report",
                    "initialization_report",
                    "upper_clothing",
                    "lower_clothing",
                    "semantic_support_plan",
                    "semantic_inputs",
                    "d03_train_evidence_plan",
                    "d03_mesh",
                    "joint_transforms",
                    "t05_solution",
                ),
                paths,
                strict=True,
            )
        },
        "schedule": {
            "frame_slots": [0, 36, 72, 108],
            "render_image_shape": [64, 41],
            "source_image_shape": list(image_shape),
            "sample_count_per_layer": 512,
            "reference_sample_count": 512,
            "sigma_pixels": 1.2,
            "depth_temperature_metres": 0.05,
            "maximum_displacement_metres": 0.08,
            "optimizer": "Adam",
            "learning_rate": 0.002,
            "optimizer_steps": 1,
            "silhouette_weight": 1.0,
            "boundary_weight": 0.05,
            "smoothness_weight": 0.10,
            "magnitude_weight": 0.02,
            "contact_weight": 0.25,
            "seed": 20260903,
        },
        "gates": {
            "finite_objective": True,
            "both_layer_gradients_active": True,
            "both_layer_parameters_change": True,
            "bounded_outward_displacement": True,
            "fixed_connectivity": True,
            "same_device_checkpoint_restore_exact": True,
            "development_records_read": 0,
            "sealed_test_reads": 0,
        },
        "provenance": {
            "training_records_bound": 144,
            "training_records_exercised": 4,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "qualification_optimizer_steps": 1,
            "scientific_optimizer_steps": 0,
            "scientific_attempt_marker_created": False,
            "paid_jobs": 0,
            "automatic_retries": 0,
        },
    }


def write_l03_training_qualification_plan(
    *,
    public_training_report_path: Path,
    initialization_report_path: Path,
    upper_layer_path: Path,
    lower_layer_path: Path,
    semantic_support_plan_path: Path,
    semantic_inputs_path: Path,
    d03_train_evidence_plan_path: Path,
    d03_mesh_path: Path,
    joint_transforms_path: Path,
    t05_solution_path: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("L03 training qualification plan is immutable")
    return write_json(
        output_path,
        build_l03_training_qualification_plan(
            public_training_report_path=public_training_report_path,
            initialization_report_path=initialization_report_path,
            upper_layer_path=upper_layer_path,
            lower_layer_path=lower_layer_path,
            semantic_support_plan_path=semantic_support_plan_path,
            semantic_inputs_path=semantic_inputs_path,
            d03_train_evidence_plan_path=d03_train_evidence_plan_path,
            d03_mesh_path=d03_mesh_path,
            joint_transforms_path=joint_transforms_path,
            t05_solution_path=t05_solution_path,
            source_revision=source_revision,
        ),
    )


def _load_and_verify_training_qualification_plan(
    plan_path: Path, *, input_overrides: dict[str, Path] | None = None
) -> tuple[dict[str, Any], dict[str, Path]]:
    plan = read_json(plan_path)
    if plan.get("schema_version") != L03_TRAINING_QUALIFICATION_PLAN_SCHEMA:
        raise ValueError("L03 training qualification plan schema is invalid")
    if plan.get("experiment_id") != L03_EXPERIMENT_ID:
        raise ValueError("L03 training qualification plan has the wrong experiment")
    if plan.get("status") != "local_training_qualification_planned":
        raise ValueError("L03 training qualification plan is not frozen")
    expected_names = set(plan["input_paths"])
    if input_overrides is not None and set(input_overrides) != expected_names:
        raise ValueError("L03 training qualification input override names are incomplete")
    paths = (
        {name: Path(value) for name, value in plan["input_paths"].items()}
        if input_overrides is None
        else input_overrides
    )
    reject_sealed_capability([plan_path, *paths.values()])
    for name, path in paths.items():
        if sha256_file(path) != plan["input_hashes"][name]:
            raise ValueError(f"L03 training qualification input changed after planning: {name}")
    return plan, paths


def _contact_pairs(upper: dict[str, np.ndarray], lower: dict[str, np.ndarray]) -> np.ndarray:
    upper_sources = upper["source_level_set_vertex_indices"].astype(np.int64)
    lower_sources = lower["source_level_set_vertex_indices"].astype(np.int64)
    lower_lookup = {int(source): slot for slot, source in enumerate(lower_sources)}
    pairs = [
        (slot, lower_lookup[int(source)])
        for slot, source in enumerate(upper_sources)
        if int(source) in lower_lookup
    ]
    if not pairs:
        raise ValueError("L03 training qualification lacks registered layer contact vertices")
    return np.asarray(pairs, dtype=np.int64)


def _qualification_objective(
    models: dict[str, LayerDisplacementModel],
    layer_inputs: dict[str, dict[str, Tensor]],
    targets: dict[str, Tensor],
    transforms: Tensor,
    intrinsics: Tensor,
    contact_pairs: Tensor,
    schedule: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    device = intrinsics.device
    total = torch.zeros((), dtype=torch.float32, device=device)
    silhouette_total = torch.zeros_like(total)
    boundary_total = torch.zeros_like(total)
    for frame_index, transform in enumerate(transforms):
        for layer_index, layer in enumerate(L03_GARMENT_LAYERS):
            torch.manual_seed(int(schedule["seed"]) + 100 * layer_index + frame_index)
            posed = linear_blend_skinning(
                models[layer](), layer_inputs[layer]["weights"], transform
            )
            prediction, _ = render_soft_mesh(
                posed,
                layer_inputs[layer]["faces"],
                intrinsics,
                tuple(schedule["render_image_shape"]),
                source_image_size=tuple(schedule["source_image_shape"]),
                sigma_pixels=float(schedule["sigma_pixels"]),
                sample_count=int(schedule["sample_count_per_layer"]),
                reference_sample_count=int(schedule["reference_sample_count"]),
                depth_temperature_m=float(schedule["depth_temperature_metres"]),
            )
            target = targets[layer][frame_index]
            silhouette_total = silhouette_total + silhouette_loss(prediction, target)
            boundary_total = boundary_total + differentiable_boundary_loss(prediction, target)
    divisor = len(transforms) * len(L03_GARMENT_LAYERS)
    silhouette_total = silhouette_total / divisor
    boundary_total = boundary_total / divisor
    smoothness = torch.zeros_like(total)
    magnitude = torch.zeros_like(total)
    for model in models.values():
        layer_smoothness, layer_magnitude = layer_offset_regularization(model)
        smoothness = smoothness + layer_smoothness
        magnitude = magnitude + layer_magnitude
    upper_offsets = models["upper_clothing"].bounded_displacement()
    lower_offsets = models["lower_clothing"].bounded_displacement()
    contact = (
        (upper_offsets[contact_pairs[:, 0]] - lower_offsets[contact_pairs[:, 1]]).square().mean()
    )
    total = (
        float(schedule["silhouette_weight"]) * silhouette_total
        + float(schedule["boundary_weight"]) * boundary_total
        + float(schedule["smoothness_weight"]) * smoothness
        + float(schedule["magnitude_weight"]) * magnitude
        + float(schedule["contact_weight"]) * contact
    )
    components = {
        "silhouette": float(silhouette_total.detach()),
        "boundary": float(boundary_total.detach()),
        "smoothness": float(smoothness.detach()),
        "magnitude": float(magnitude.detach()),
        "contact": float(contact.detach()),
        "total": float(total.detach()),
    }
    return total, components


def run_l03_local_training_qualification(
    *,
    plan_path: Path,
    output_root: Path,
    device: str,
    input_overrides: dict[str, Path] | None = None,
) -> Path:
    """Exercise one real train-only step and exact same-device restore."""

    reject_sealed_capability([plan_path, output_root])
    if output_root.exists():
        raise FileExistsError("L03 local training qualification output is immutable")
    plan, paths = _load_and_verify_training_qualification_plan(
        plan_path, input_overrides=input_overrides
    )
    target_device = torch.device(device)
    if target_device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("L03 requested MPS qualification but MPS is unavailable")
    schedule = plan["schedule"]
    layer_arrays = {layer: _load_initial_layer(paths[layer]) for layer in L03_GARMENT_LAYERS}
    with np.load(paths["d03_mesh"], allow_pickle=False) as body_archive:
        body_vertices = body_archive["vertices"].astype(np.float32)
    models: dict[str, LayerDisplacementModel] = {}
    layer_inputs: dict[str, dict[str, Tensor]] = {}
    for layer in L03_GARMENT_LAYERS:
        arrays = layer_arrays[layer]
        vertices = torch.as_tensor(arrays["vertices"], dtype=torch.float32, device=target_device)
        faces = torch.as_tensor(arrays["faces"], dtype=torch.long, device=target_device)
        nearest = arrays.get("nearest_body_vertex")
        if nearest is None:
            raise ValueError("L03 training layer lacks its outward body correspondence")
        directions = vertices - torch.as_tensor(
            body_vertices[nearest.astype(np.int64)],
            dtype=torch.float32,
            device=target_device,
        )
        models[layer] = LayerDisplacementModel(
            vertices,
            faces,
            maximum_displacement_metres=float(schedule["maximum_displacement_metres"]),
            outward_directions=directions,
        ).to(target_device)
        layer_inputs[layer] = {
            "faces": faces,
            "weights": torch.as_tensor(
                arrays["skinning_weights"], dtype=torch.float32, device=target_device
            ),
        }
    frame_slots = np.asarray(schedule["frame_slots"], dtype=np.int64)
    with np.load(paths["semantic_inputs"], allow_pickle=False) as archive:
        source_indices = archive["source_frame_indices"].astype(np.int64)
        intrinsics = torch.as_tensor(
            archive["intrinsics"], dtype=torch.float32, device=target_device
        )
        target_maps = {
            layer: torch.as_tensor(
                archive[f"semantic__{layer}"][frame_slots],
                dtype=torch.float32,
                device=target_device,
            )
            for layer in L03_GARMENT_LAYERS
        }
    output_shape = tuple(int(value) for value in schedule["render_image_shape"])
    targets = {
        layer: F.interpolate(
            maps[:, None], size=output_shape, mode="bilinear", align_corners=False
        )[:, 0].clamp(0.0, 1.0)
        for layer, maps in target_maps.items()
    }
    with np.load(paths["joint_transforms"], allow_pickle=False) as archive:
        transform_sources = archive["source_frame_indices"].astype(np.int64)
        transform_values = archive["transforms"].astype(np.float32)
    lookup = {int(source): slot for slot, source in enumerate(transform_sources)}
    selected_transforms = torch.as_tensor(
        np.stack([transform_values[lookup[int(source_indices[slot])]] for slot in frame_slots]),
        dtype=torch.float32,
        device=target_device,
    )
    contact_pairs = torch.as_tensor(
        _contact_pairs(layer_arrays["upper_clothing"], layer_arrays["lower_clothing"]),
        dtype=torch.long,
        device=target_device,
    )
    parameters = [parameter for model in models.values() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=float(schedule["learning_rate"]))
    optimizer.zero_grad(set_to_none=True)
    objective, before = _qualification_objective(
        models,
        layer_inputs,
        targets,
        selected_transforms,
        intrinsics,
        contact_pairs,
        schedule,
    )
    objective.backward()  # type: ignore[no-untyped-call]
    gradient_norms = {
        layer: float(torch.linalg.vector_norm(model.raw_displacement.grad).detach())
        if model.raw_displacement.grad is not None
        else 0.0
        for layer, model in models.items()
    }
    parameter_before = {
        layer: model.raw_displacement.detach().clone() for layer, model in models.items()
    }
    optimizer.step()
    for model in models.values():
        model.project_parameters()
    parameter_changes = {
        layer: float(
            torch.linalg.vector_norm(model.raw_displacement - parameter_before[layer]).detach()
        )
        for layer, model in models.items()
    }
    _, after = _qualification_objective(
        models,
        layer_inputs,
        targets,
        selected_transforms,
        intrinsics,
        contact_pairs,
        schedule,
    )
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError("L03 local training qualification staging output already exists")
    staging.mkdir(parents=True)
    try:
        checkpoint_path = staging / "qualification_checkpoint.pt"
        checkpoint = {
            "models": {layer: model.state_dict() for layer, model in models.items()},
            "optimizer": optimizer.state_dict(),
            "schedule": schedule,
            "source_revision": plan["source_revision"],
        }
        torch.save(checkpoint, checkpoint_path)
        restored_models: dict[str, LayerDisplacementModel] = {}
        for layer in L03_GARMENT_LAYERS:
            arrays = layer_arrays[layer]
            restored = LayerDisplacementModel(
                torch.as_tensor(arrays["vertices"], dtype=torch.float32, device=target_device),
                torch.as_tensor(arrays["faces"], dtype=torch.long, device=target_device),
                maximum_displacement_metres=float(schedule["maximum_displacement_metres"]),
                outward_directions=models[layer].directions,
            ).to(target_device)
            restored.load_state_dict(checkpoint["models"][layer])
            restored_models[layer] = restored
        exact_restore = all(
            torch.equal(models[layer](), restored_models[layer]()) for layer in L03_GARMENT_LAYERS
        )
        finite = all(math.isfinite(value) for value in (*before.values(), *after.values()))
        bounded = all(
            float(model.bounded_displacement().min().detach()) >= 0.0
            and float(model.bounded_displacement().max().detach())
            <= float(schedule["maximum_displacement_metres"])
            for model in models.values()
        )
        gates = {
            "finite_objective": finite,
            "both_layer_gradients_active": all(value > 0.0 for value in gradient_norms.values()),
            "both_layer_parameters_change": all(
                value > 0.0 for value in parameter_changes.values()
            ),
            "bounded_outward_displacement": bounded,
            "fixed_connectivity": all(
                torch.equal(models[layer].faces, layer_inputs[layer]["faces"])
                for layer in L03_GARMENT_LAYERS
            ),
            "same_device_checkpoint_restore_exact": exact_restore,
            "development_records_read": plan["provenance"]["development_records_read"] == 0,
            "sealed_test_reads": plan["provenance"]["sealed_test_reads"] == 0,
        }
        report = {
            "schema_version": L03_TRAINING_QUALIFICATION_SCHEMA,
            "experiment_id": L03_EXPERIMENT_ID,
            "status": "pass" if all(gates.values()) else "fail",
            "decision": (
                "local_training_qualification_passed_target_gpu_pending"
                if all(gates.values())
                else "local_training_qualification_failed"
            ),
            "source_revision": plan["source_revision"],
            "plan_sha256": sha256_file(plan_path),
            "device": str(target_device),
            "dtype": "float32",
            "objective_before": before,
            "objective_after": after,
            "gradient_norms": gradient_norms,
            "parameter_change_norms": parameter_changes,
            "maximum_displacements_metres": {
                layer: float(model.bounded_displacement().max().detach())
                for layer, model in models.items()
            },
            "registered_contact_vertex_count": len(contact_pairs),
            "checkpoint": {
                "path": str(output_root / checkpoint_path.name),
                "sha256": sha256_file(checkpoint_path),
            },
            "gates": gates,
            "provenance": plan["provenance"],
        }
        report_path = staging / "local_training_qualification.json"
        write_json(report_path, report)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / "local_training_qualification.json"
