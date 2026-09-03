from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from skimage.measure import marching_cubes

from frayid.io import read_json, sha256_file, write_json
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.d02_topology_projection import ipctk_has_self_intersections
from frayid.v2.d03_capsule_tree import (
    Capsule,
    _extract_field_surface,
    _field_for_capsules,
    _mesh_audit,
    capsule_tree_signed_distance,
)

L03_EXPERIMENT_ID = "postv2_l03_semantic_open_clothing_layers_r01"
L03_PUBLIC_SCHEMA = "frayid_v2_l03_public_open_layers.v1"
L03_SEMANTIC_SUPPORT_PLAN_SCHEMA = "frayid_v2_l03_semantic_support_plan.v1"
L03_SEMANTIC_SUPPORT_REPORT_SCHEMA = "frayid_v2_l03_semantic_support_report.v1"
L03_REAL_INITIALIZATION_PLAN_SCHEMA = "frayid_v2_l03_real_initialization_plan.v1"
L03_REAL_INITIALIZATION_REPORT_SCHEMA = "frayid_v2_l03_real_initialization_report.v1"
L03_GARMENT_LAYERS = ("upper_clothing", "lower_clothing")


def make_open_wrinkled_tube(
    *,
    y_minimum: float,
    y_maximum: float,
    base_radius: float,
    wrinkle_amplitude: float,
    phase: float,
    ring_count: int = 15,
    segment_count: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an oriented open tube with two explicit circular boundary loops."""

    if y_maximum <= y_minimum or base_radius <= 0.0 or wrinkle_amplitude < 0.0:
        raise ValueError("L03 open-tube dimensions are invalid")
    if ring_count < 3 or segment_count < 8:
        raise ValueError("L03 open tube requires at least three rings and eight segments")
    vertices: list[tuple[float, float, float]] = []
    for ring in range(ring_count):
        fraction = ring / (ring_count - 1)
        y = (1.0 - fraction) * y_minimum + fraction * y_maximum
        vertical_envelope = math.sin(math.pi * fraction)
        for segment in range(segment_count):
            theta = 2.0 * math.pi * segment / segment_count
            radius = base_radius + wrinkle_amplitude * vertical_envelope * math.sin(
                5.0 * theta + phase
            )
            vertices.append((radius * math.cos(theta), y, radius * math.sin(theta)))
    faces: list[tuple[int, int, int]] = []
    for ring in range(ring_count - 1):
        for segment in range(segment_count):
            following = (segment + 1) % segment_count
            lower = ring * segment_count + segment
            lower_next = ring * segment_count + following
            upper = (ring + 1) * segment_count + segment
            upper_next = (ring + 1) * segment_count + following
            faces.extend(((lower, upper_next, upper), (lower, lower_next, upper_next)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def boundary_loop_audit(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Count boundary loops without filling, welding, or deleting geometry."""

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("L03 layer vertices must have shape [V,3]")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("L03 layer faces must have shape [F,3]")
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    sorted_edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("L03 layer is nonmanifold")
    boundary_edges = unique_edges[counts == 1]
    adjacency: dict[int, set[int]] = {}
    for first, second in boundary_edges.tolist():
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    degrees_valid = all(len(neighbours) == 2 for neighbours in adjacency.values())
    remaining = set(adjacency)
    loop_count = 0
    while remaining:
        loop_count += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
    mesh = trimesh.Trimesh(vertices=points, faces=triangles, process=False)
    return {
        "component_count": len(mesh.split(only_watertight=False)),
        "boundary_edge_count": len(boundary_edges),
        "boundary_vertex_count": len(adjacency),
        "boundary_loop_count": loop_count,
        "boundary_vertex_degrees_two": degrees_valid,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "exact_self_intersections": ipctk_has_self_intersections(points, triangles),
        "cleanup_operations": 0,
    }


def _combined_intersections(
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    layer_vertices: np.ndarray,
    layer_faces: np.ndarray,
) -> bool:
    vertices = np.concatenate((body_vertices, layer_vertices), axis=0)
    faces = np.concatenate((body_faces, layer_faces + len(body_vertices)), axis=0)
    return ipctk_has_self_intersections(vertices, faces)


def _cap_top_boundary(
    vertices: np.ndarray, faces: np.ndarray, *, segment_count: int
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    top = np.arange(len(points) - segment_count, len(points), dtype=np.int64)
    center = np.asarray([[0.0, float(np.mean(points[top, 1])), 0.0]])
    center_index = len(points)
    caps = np.asarray(
        [
            (center_index, int(top[(slot + 1) % segment_count]), int(top[slot]))
            for slot in range(segment_count)
        ],
        dtype=np.int64,
    )
    return np.concatenate((points, center), axis=0), np.concatenate((triangles, caps), axis=0)


def _curve_error(candidate: np.ndarray, truth: np.ndarray) -> float:
    candidate_radius = np.linalg.norm(np.asarray(candidate)[:, [0, 2]], axis=1)
    truth_radius = np.linalg.norm(np.asarray(truth)[:, [0, 2]], axis=1)
    return float(np.mean(np.abs(candidate_radius - truth_radius)))


def run_l03_public_benchmark() -> dict[str, Any]:
    """Qualify explicit open boundaries, ordering, contact, and failure controls."""

    body_capsules = (Capsule((0.0, -1.02, 0.0), (0.0, 0.72, 0.0), 0.24, "body"),)
    body_field = _field_for_capsules(body_capsules, resolution=72, extent=1.32)
    body_vertices, body_faces = _extract_field_surface(body_field, extent=1.32)
    body_audit = _mesh_audit(body_vertices, body_faces)
    layer_specs = {
        "upper_clothing": (-0.12, 0.52, 0.315, 0.026, 0.3),
        "lower_clothing": (-0.88, -0.16, 0.295, 0.020, 1.1),
    }
    layer_reports: dict[str, Any] = {}
    truth_errors = 0.0
    control_errors = 0.0
    replay_equal = True
    for label, (low, high, radius, amplitude, phase) in layer_specs.items():
        truth_vertices, truth_faces = make_open_wrinkled_tube(
            y_minimum=low,
            y_maximum=high,
            base_radius=radius,
            wrinkle_amplitude=amplitude,
            phase=phase,
        )
        treatment_vertices, treatment_faces = make_open_wrinkled_tube(
            y_minimum=low,
            y_maximum=high,
            base_radius=radius,
            wrinkle_amplitude=0.85 * amplitude,
            phase=phase,
        )
        control_vertices, _ = make_open_wrinkled_tube(
            y_minimum=low,
            y_maximum=high,
            base_radius=radius,
            wrinkle_amplitude=0.0,
            phase=phase,
        )
        replay_vertices, replay_faces = make_open_wrinkled_tube(
            y_minimum=low,
            y_maximum=high,
            base_radius=radius,
            wrinkle_amplitude=0.85 * amplitude,
            phase=phase,
        )
        replay_equal &= np.array_equal(treatment_vertices, replay_vertices) and np.array_equal(
            treatment_faces, replay_faces
        )
        audit = boundary_loop_audit(treatment_vertices, treatment_faces)
        clearance = capsule_tree_signed_distance(treatment_vertices, body_capsules)
        body_layer_intersections = _combined_intersections(
            body_vertices, body_faces, treatment_vertices, treatment_faces
        )
        treatment_error = _curve_error(treatment_vertices, truth_vertices)
        control_error = _curve_error(control_vertices, truth_vertices)
        truth_errors += treatment_error
        control_errors += control_error
        layer_reports[label] = {
            "vertex_count": len(treatment_vertices),
            "face_count": len(treatment_faces),
            "registered_boundary_loop_count": 2,
            "audit": audit,
            "minimum_body_clearance": float(clearance.min()),
            "maximum_body_clearance": float(clearance.max()),
            "body_layer_intersections": body_layer_intersections,
            "treatment_curve_error": treatment_error,
            "smooth_control_curve_error": control_error,
            "truth_face_connectivity_matches": np.array_equal(treatment_faces, truth_faces),
        }
    curve_improvement = 1.0 - truth_errors / control_errors
    upper = make_open_wrinkled_tube(
        y_minimum=-0.12,
        y_maximum=0.52,
        base_radius=0.315,
        wrinkle_amplitude=0.85 * 0.026,
        phase=0.3,
    )
    capped_vertices, capped_faces = _cap_top_boundary(*upper, segment_count=48)
    capped_audit = boundary_loop_audit(capped_vertices, capped_faces)
    penetrating_vertices, penetrating_faces = make_open_wrinkled_tube(
        y_minimum=-0.12,
        y_maximum=0.52,
        base_radius=0.20,
        wrinkle_amplitude=0.0,
        phase=0.3,
    )
    penetrating_clearance = capsule_tree_signed_distance(penetrating_vertices, body_capsules)
    gates = {
        "closed_body_embedded": body_audit["status"] == "pass",
        "registered_boundary_loops_exact": all(
            report["audit"]["boundary_loop_count"] == report["registered_boundary_loop_count"]
            and report["audit"]["boundary_vertex_degrees_two"]
            for report in layer_reports.values()
        ),
        "open_layers_not_watertight": all(
            report["audit"]["watertight"] is False for report in layer_reports.values()
        ),
        "layer_self_intersection_free": all(
            report["audit"]["exact_self_intersections"] is False
            for report in layer_reports.values()
        ),
        "body_layer_intersection_free": all(
            report["body_layer_intersections"] is False for report in layer_reports.values()
        ),
        "nonnegative_clearance": all(
            report["minimum_body_clearance"] >= 0.0 for report in layer_reports.values()
        ),
        "registered_contact_band": all(
            report["minimum_body_clearance"] <= 0.10 for report in layer_reports.values()
        ),
        "garment_curve_error_improvement": curve_improvement >= 0.20,
        "topology_changing_cap_rejected": capped_audit["boundary_loop_count"] != 2,
        "penetrating_proposal_rejected": float(penetrating_clearance.min()) < 0.0,
        "exact_replay": replay_equal,
    }
    return {
        "schema_version": L03_PUBLIC_SCHEMA,
        "experiment_id": L03_EXPERIMENT_ID,
        "status": "pass" if all(gates.values()) else "fail",
        "body": body_audit,
        "layers": layer_reports,
        "aggregate_curve_error": {
            "treatment": truth_errors,
            "smooth_control": control_errors,
            "relative_improvement": curve_improvement,
        },
        "adversarial_topology": capped_audit,
        "adversarial_penetration_minimum_clearance": float(penetrating_clearance.min()),
        "adversarial_penetration_face_count": len(penetrating_faces),
        "gates": gates,
        "provenance": {
            "private_records_read": 0,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "cleanup_operations": 0,
            "generated_views_used_as_evidence": False,
        },
    }


def write_l03_public_benchmark(output_path: Path) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("L03 public benchmark output is immutable")
    return write_json(output_path, run_l03_public_benchmark())


def _load_l03_semantic_archive(path: Path) -> dict[str, np.ndarray]:
    required = {
        "source_frame_indices",
        "semantic__upper_clothing",
        "semantic__lower_clothing",
        "source_hashes",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"L03 semantic input is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    sources = arrays["source_frame_indices"]
    if sources.shape != (144,) or len(np.unique(sources)) != 144:
        raise ValueError("L03 semantic input must contain the frozen 144 unique train frames")
    for layer in L03_GARMENT_LAYERS:
        values = arrays[f"semantic__{layer}"]
        if values.ndim != 3 or values.shape[0] != 144:
            raise ValueError(f"L03 {layer} evidence must have shape [144,H,W]")
        if not np.isfinite(values).all() or float(values.min()) < 0.0 or float(values.max()) > 1.0:
            raise ValueError(f"L03 {layer} confidence must be finite and lie in [0,1]")
    if arrays["semantic__upper_clothing"].shape != arrays["semantic__lower_clothing"].shape:
        raise ValueError("L03 upper/lower semantic evidence must share image coordinates")
    return arrays


def build_l03_semantic_support_plan(
    *,
    public_report_path: Path,
    semantic_inputs_path: Path,
    t05_solution_path: Path,
    s01_qualification_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Bind immutable train-only semantic evidence before measuring layer support."""

    paths = [
        public_report_path,
        semantic_inputs_path,
        t05_solution_path,
        s01_qualification_path,
    ]
    reject_sealed_capability(paths)
    if len(source_revision) != 40:
        raise ValueError("L03 semantic support source revision must be a full commit hash")
    public = read_json(public_report_path)
    if public.get("status") != "pass" or public.get("experiment_id") != L03_EXPERIMENT_ID:
        raise ValueError("L03 semantic support requires the passing public open-layer report")
    semantics = _load_l03_semantic_archive(semantic_inputs_path)
    s01 = read_json(s01_qualification_path)
    if (
        s01.get("status") != "pass"
        or s01.get("training_frame_count") != 144
        or s01.get("training_images_read") != 144
        or s01.get("legacy_development_images_read") != 0
        or s01.get("sealed_test_accesses") != 0
    ):
        raise ValueError("L03 semantic support requires the qualified train-only S01 result")
    source_hashes = json.loads(str(semantics["source_hashes"].item()))
    if source_hashes.get("semantic_qualification") != sha256_file(s01_qualification_path):
        raise ValueError("L03 semantic archive does not bind the supplied S01 qualification")
    t05 = read_json(t05_solution_path)
    t05_sources = np.asarray(
        [frame["source_frame_index"] for frame in t05.get("frames", [])], dtype=np.int64
    )
    yaws = np.asarray([frame["yaw_radians"] for frame in t05.get("frames", [])])
    if (
        t05.get("status") != "qualification_candidate"
        or t05.get("training_frame_count") != 144
        or t05.get("development_records_used_for_fit") != 0
        or t05.get("sealed_test_reads") != 0
        or not np.array_equal(semantics["source_frame_indices"], t05_sources)
        or not np.isfinite(yaws).all()
        or np.any(np.diff(yaws) < 0.0)
        or float(yaws[-1] - yaws[0]) < 2.0 * math.pi
    ):
        raise ValueError("L03 semantic evidence does not match the qualified monotonic T05 turn")
    return {
        "schema_version": L03_SEMANTIC_SUPPORT_PLAN_SCHEMA,
        "experiment_id": L03_EXPERIMENT_ID,
        "status": "real_semantic_support_planned",
        "source_revision": source_revision,
        "input_hashes": {
            "public_open_layers": sha256_file(public_report_path),
            "semantic_inputs": sha256_file(semantic_inputs_path),
            "t05_solution": sha256_file(t05_solution_path),
            "s01_qualification": sha256_file(s01_qualification_path),
        },
        "input_paths": {
            "public_open_layers": str(public_report_path),
            "semantic_inputs": str(semantic_inputs_path),
            "t05_solution": str(t05_solution_path),
            "s01_qualification": str(s01_qualification_path),
        },
        "audit": {
            "layers": list(L03_GARMENT_LAYERS),
            "confidence_threshold": 0.25,
            "minimum_pixels_per_supported_frame": 200,
            "minimum_supported_frames_per_layer": 120,
            "phase_bin_count": 12,
            "minimum_supported_phase_bins_per_layer": 12,
            "minimum_vertical_order_fraction": 0.90,
            "maximum_median_overlap_fraction": 0.05,
            "phase_coordinate": "monotonic_unwrapped_t05_yaw_minimum_to_maximum",
            "component_connectivity": 8,
            "development_evidence_allowed": False,
        },
        "provenance": {
            "training_records_bound": 144,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
        },
    }


def write_l03_semantic_support_plan(
    *,
    public_report_path: Path,
    semantic_inputs_path: Path,
    t05_solution_path: Path,
    s01_qualification_path: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("L03 semantic support plan is immutable")
    return write_json(
        output_path,
        build_l03_semantic_support_plan(
            public_report_path=public_report_path,
            semantic_inputs_path=semantic_inputs_path,
            t05_solution_path=t05_solution_path,
            s01_qualification_path=s01_qualification_path,
            source_revision=source_revision,
        ),
    )


def _load_and_verify_l03_support_plan(plan_path: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    if plan.get("schema_version") != L03_SEMANTIC_SUPPORT_PLAN_SCHEMA:
        raise ValueError("L03 semantic support plan schema is invalid")
    if plan.get("experiment_id") != L03_EXPERIMENT_ID:
        raise ValueError("L03 semantic support plan has the wrong experiment")
    if plan.get("status") != "real_semantic_support_planned":
        raise ValueError("L03 semantic support plan is not frozen")
    paths = {name: Path(value) for name, value in plan["input_paths"].items()}
    reject_sealed_capability([plan_path, *paths.values()])
    for name, path in paths.items():
        if sha256_file(path) != plan["input_hashes"][name]:
            raise ValueError(f"L03 semantic support input changed after planning: {name}")
    return plan


def _largest_component_fraction(mask: np.ndarray) -> tuple[int, float]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.reshape(-1))[1:]
    return int(count), float(sizes.max() / sizes.sum())


def _semantic_layer_diagnostics(
    values: np.ndarray,
    yaws: np.ndarray,
    *,
    confidence_threshold: float,
    minimum_pixels: int,
    phase_bin_count: int,
) -> dict[str, Any]:
    masks = values >= confidence_threshold
    pixel_counts = masks.sum(axis=(1, 2)).astype(np.int64)
    supported = pixel_counts >= minimum_pixels
    phase_edges = np.linspace(float(yaws[0]), float(yaws[-1]), phase_bin_count + 1)
    phase_slots = np.minimum(
        np.searchsorted(phase_edges, yaws, side="right") - 1, phase_bin_count - 1
    )
    phase_counts = np.bincount(phase_slots[supported], minlength=phase_bin_count)
    centroids: list[float | None] = []
    component_counts: list[int] = []
    largest_fractions: list[float] = []
    confidence_means: list[float | None] = []
    for frame, mask in zip(values, masks, strict=True):
        if not np.any(mask):
            centroids.append(None)
            component_counts.append(0)
            largest_fractions.append(0.0)
            confidence_means.append(None)
            continue
        rows = np.nonzero(mask)[0]
        centroids.append(float(np.mean(rows)))
        component_count, largest_fraction = _largest_component_fraction(mask)
        component_counts.append(component_count)
        largest_fractions.append(largest_fraction)
        confidence_means.append(float(np.mean(frame[mask])))
    valid_confidences = [value for value in confidence_means if value is not None]
    return {
        "supported_frame_count": int(supported.sum()),
        "supported_frame_indices": np.flatnonzero(supported).tolist(),
        "pixel_count_minimum": int(pixel_counts.min()),
        "pixel_count_median": float(np.median(pixel_counts)),
        "pixel_count_maximum": int(pixel_counts.max()),
        "supported_phase_bin_count": int(np.count_nonzero(phase_counts)),
        "supported_frames_per_phase_bin": phase_counts.tolist(),
        "centroid_rows": centroids,
        "component_count_median": float(np.median(component_counts)),
        "component_count_maximum": int(np.max(component_counts)),
        "largest_component_fraction_median": float(np.median(largest_fractions)),
        "positive_confidence_mean": float(np.mean(valid_confidences)),
    }


def audit_l03_semantic_support(*, plan_path: Path, output_path: Path) -> Path:
    """Measure whether train-only semantics justify attempting real open layers."""

    reject_sealed_capability([plan_path, output_path])
    if output_path.exists():
        raise FileExistsError("L03 semantic support report is immutable")
    plan = _load_and_verify_l03_support_plan(plan_path)
    paths = {name: Path(value) for name, value in plan["input_paths"].items()}
    semantics = _load_l03_semantic_archive(paths["semantic_inputs"])
    t05 = read_json(paths["t05_solution"])
    yaws = np.asarray([frame["yaw_radians"] for frame in t05["frames"]], dtype=np.float64)
    settings = plan["audit"]

    def calculate() -> dict[str, Any]:
        return {
            layer: _semantic_layer_diagnostics(
                semantics[f"semantic__{layer}"],
                yaws,
                confidence_threshold=settings["confidence_threshold"],
                minimum_pixels=settings["minimum_pixels_per_supported_frame"],
                phase_bin_count=settings["phase_bin_count"],
            )
            for layer in L03_GARMENT_LAYERS
        }

    layers = calculate()
    exact_replay = layers == calculate()
    upper_centroids = layers["upper_clothing"]["centroid_rows"]
    lower_centroids = layers["lower_clothing"]["centroid_rows"]
    paired = [
        (upper, lower)
        for upper, lower in zip(upper_centroids, lower_centroids, strict=True)
        if upper is not None and lower is not None
    ]
    vertical_order_fraction = float(
        np.mean([upper < lower for upper, lower in paired]) if paired else 0.0
    )
    threshold = settings["confidence_threshold"]
    upper_masks = semantics["semantic__upper_clothing"] >= threshold
    lower_masks = semantics["semantic__lower_clothing"] >= threshold
    intersections = np.logical_and(upper_masks, lower_masks).sum(axis=(1, 2))
    smaller = np.minimum(upper_masks.sum(axis=(1, 2)), lower_masks.sum(axis=(1, 2)))
    overlap = np.divide(
        intersections,
        smaller,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=smaller > 0,
    )
    median_overlap_fraction = float(np.median(overlap))
    gates = {
        "public_open_layer_prerequisite_passed": True,
        "s01_train_only_qualification_bound": True,
        "t05_monotonic_full_turn_bound": True,
        "minimum_supported_frames": all(
            layers[layer]["supported_frame_count"] >= settings["minimum_supported_frames_per_layer"]
            for layer in L03_GARMENT_LAYERS
        ),
        "complete_supported_phase_coverage": all(
            layers[layer]["supported_phase_bin_count"]
            >= settings["minimum_supported_phase_bins_per_layer"]
            for layer in L03_GARMENT_LAYERS
        ),
        "upper_lower_vertical_order": (
            vertical_order_fraction >= settings["minimum_vertical_order_fraction"]
        ),
        "upper_lower_overlap_bounded": (
            median_overlap_fraction <= settings["maximum_median_overlap_fraction"]
        ),
        "exact_replay": exact_replay,
    }
    for diagnostics in layers.values():
        diagnostics.pop("centroid_rows")
    report = {
        "schema_version": L03_SEMANTIC_SUPPORT_REPORT_SCHEMA,
        "experiment_id": L03_EXPERIMENT_ID,
        "status": "pass" if all(gates.values()) else "fail",
        "plan_sha256": sha256_file(plan_path),
        "source_revision": plan["source_revision"],
        "settings": settings,
        "layers": layers,
        "joint_diagnostics": {
            "co_present_frame_count": len(paired),
            "upper_above_lower_fraction": vertical_order_fraction,
            "median_overlap_fraction": median_overlap_fraction,
        },
        "gates": gates,
        "provenance": {
            "training_records_read": 144,
            "development_records_read": 0,
            "development_records_used_for_fit": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "generated_views_used_as_evidence": False,
        },
    }
    return write_json(output_path, report)


def _load_d03_body_inputs(
    field_path: Path, mesh_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(field_path, allow_pickle=False) as field_archive:
        if set(field_archive.files) != {"values", "coordinates", "surface_level"}:
            raise ValueError("L03 D03 body field has unexpected arrays")
        field = field_archive["values"].astype(np.float32)
        coordinates = field_archive["coordinates"].astype(np.float64)
        surface_level = float(field_archive["surface_level"])
    if (
        field.ndim != 3
        or field.shape[0] != field.shape[1]
        or field.shape[1] != field.shape[2]
        or coordinates.shape != (field.shape[0],)
        or not np.allclose(np.diff(coordinates), np.diff(coordinates)[0])
        or not math.isclose(surface_level, 0.0, abs_tol=1.0e-12)
    ):
        raise ValueError("L03 D03 body field is not the registered regular zero-level grid")
    with np.load(mesh_path, allow_pickle=False) as mesh_archive:
        required = {"vertices", "faces", "skinning_weights"}
        if not required.issubset(mesh_archive.files):
            raise ValueError("L03 D03 body mesh is missing geometry or skinning weights")
        vertices = mesh_archive["vertices"].astype(np.float64)
        faces = mesh_archive["faces"].astype(np.int64)
        weights = mesh_archive["skinning_weights"].astype(np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or weights.shape != (len(vertices), 24)
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5)
    ):
        raise ValueError("L03 D03 body mesh arrays are invalid")
    return field, coordinates, vertices, faces, weights


def _load_semantic_evidence_volume(
    path: Path,
) -> tuple[dict[str, np.ndarray], float, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {f"semantic__{layer}" for layer in L03_GARMENT_LAYERS} | {"metadata"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"L03 semantic evidence volume is missing arrays: {sorted(missing)}")
        metadata = json.loads(str(archive["metadata"]))
        semantics = {
            layer: archive[f"semantic__{layer}"].astype(np.float32) for layer in L03_GARMENT_LAYERS
        }
    resolution = int(metadata["resolution"])
    extent = float(metadata["extent"])
    if (
        resolution < 16
        or extent <= 0.0
        or any(
            values.shape != (resolution, resolution, resolution) for values in semantics.values()
        )
        or any(not np.isfinite(values).all() for values in semantics.values())
    ):
        raise ValueError("L03 semantic evidence volume is invalid")
    return semantics, extent, metadata


def build_l03_real_initialization_plan(
    *,
    semantic_support_report_path: Path,
    d03_report_path: Path,
    d03_field_path: Path,
    d03_mesh_path: Path,
    hull_qualification_path: Path,
    semantic_volume_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Freeze L03's offset-shell semantic partition before real extraction."""

    paths = [
        semantic_support_report_path,
        d03_report_path,
        d03_field_path,
        d03_mesh_path,
        hull_qualification_path,
        semantic_volume_path,
    ]
    reject_sealed_capability(paths)
    if len(source_revision) != 40:
        raise ValueError("L03 real initialization source revision must be a full commit hash")
    support = read_json(semantic_support_report_path)
    d03 = read_json(d03_report_path)
    hull = read_json(hull_qualification_path)
    if support.get("status") != "pass":
        raise ValueError("L03 real initialization requires passing semantic support")
    if d03.get("status") != "pass" or d03.get("candidate_topology", {}).get("status") != "pass":
        raise ValueError("L03 real initialization requires the passing embedded D03 body")
    if hull.get("status") != "pass" or hull.get("semantic_layer_status") != "bound":
        raise ValueError("L03 real initialization requires the qualified semantic visual hull")
    if d03["artifacts"]["continued_field"]["sha256"] != sha256_file(d03_field_path):
        raise ValueError("L03 D03 field differs from its passing continuation report")
    if d03["artifacts"]["continued_mesh"]["sha256"] != sha256_file(d03_mesh_path):
        raise ValueError("L03 D03 mesh differs from its passing continuation report")
    if hull["source_hashes"].get("reference_volume") != sha256_file(semantic_volume_path):
        raise ValueError("L03 semantic volume differs from its qualification report")
    _load_d03_body_inputs(d03_field_path, d03_mesh_path)
    _, semantic_extent, semantic_metadata = _load_semantic_evidence_volume(semantic_volume_path)
    return {
        "schema_version": L03_REAL_INITIALIZATION_PLAN_SCHEMA,
        "experiment_id": L03_EXPERIMENT_ID,
        "status": "real_initialization_planned",
        "source_revision": source_revision,
        "input_hashes": {
            "semantic_support_report": sha256_file(semantic_support_report_path),
            "d03_report": sha256_file(d03_report_path),
            "d03_field": sha256_file(d03_field_path),
            "d03_mesh": sha256_file(d03_mesh_path),
            "hull_qualification": sha256_file(hull_qualification_path),
            "semantic_volume": sha256_file(semantic_volume_path),
        },
        "input_paths": {
            "semantic_support_report": str(semantic_support_report_path),
            "d03_report": str(d03_report_path),
            "d03_field": str(d03_field_path),
            "d03_mesh": str(d03_mesh_path),
            "hull_qualification": str(hull_qualification_path),
            "semantic_volume": str(semantic_volume_path),
        },
        "construction": {
            "body_role": "immutable_prior_derived_closed_inner_layer",
            "garment_source": "positive_level_set_of_d03_continuous_field",
            "body_clearance_level_metres": 0.01,
            "semantic_sampling": "trilinear_on_qualified_r32_evidence_volume",
            "semantic_volume_extent_metres": semantic_extent,
            "semantic_volume_resolution": int(semantic_metadata["resolution"]),
            "face_score": "mean_of_three_vertex_semantic_support_values",
            "face_ownership": "argmax_upper_lower_if_maximum_score_at_least_threshold",
            "semantic_face_threshold": 0.25,
            "registered_component_counts": {
                "upper_clothing": 1,
                "lower_clothing": 1,
            },
            "registered_boundary_loop_counts": {
                "upper_clothing": 1,
                "lower_clothing": 2,
            },
            "registered_contact": "shared_source_level_set_edges_between_layer_partitions",
            "skinning_transfer": "nearest_d03_body_vertex_weights_without_geometry_transfer",
            "cleanup_operations_allowed": 0,
            "hole_filling_allowed": False,
            "largest_component_filter_allowed": False,
        },
        "gates": {
            "minimum_layer_vertices": 500,
            "minimum_layer_faces": 800,
            "minimum_body_clearance_metres": 0.005,
            "maximum_body_clearance_metres": 0.015,
            "minimum_registered_contact_edges": 1,
            "exact_self_intersections": 0,
            "unregistered_body_layer_intersections": 0,
            "winding_consistent": True,
            "exact_replay": True,
            "development_records_read": 0,
            "sealed_test_reads": 0,
        },
        "provenance": {
            "training_semantic_records_bound": 144,
            "development_records_read": 0,
            "sealed_test_reads": 0,
            "optimizer_steps": 0,
            "paid_jobs": 0,
            "automatic_retries": 0,
            "cleanup_operations": 0,
        },
    }


def write_l03_real_initialization_plan(
    *,
    semantic_support_report_path: Path,
    d03_report_path: Path,
    d03_field_path: Path,
    d03_mesh_path: Path,
    hull_qualification_path: Path,
    semantic_volume_path: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    reject_sealed_capability([output_path])
    if output_path.exists():
        raise FileExistsError("L03 real initialization plan is immutable")
    return write_json(
        output_path,
        build_l03_real_initialization_plan(
            semantic_support_report_path=semantic_support_report_path,
            d03_report_path=d03_report_path,
            d03_field_path=d03_field_path,
            d03_mesh_path=d03_mesh_path,
            hull_qualification_path=hull_qualification_path,
            semantic_volume_path=semantic_volume_path,
            source_revision=source_revision,
        ),
    )


def _load_and_verify_l03_real_initialization_plan(plan_path: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    if plan.get("schema_version") != L03_REAL_INITIALIZATION_PLAN_SCHEMA:
        raise ValueError("L03 real initialization plan schema is invalid")
    if plan.get("experiment_id") != L03_EXPERIMENT_ID:
        raise ValueError("L03 real initialization plan has the wrong experiment")
    if plan.get("status") != "real_initialization_planned":
        raise ValueError("L03 real initialization plan is not frozen")
    paths = {name: Path(value) for name, value in plan["input_paths"].items()}
    reject_sealed_capability([plan_path, *paths.values()])
    for name, path in paths.items():
        if sha256_file(path) != plan["input_hashes"][name]:
            raise ValueError(f"L03 real initialization input changed after planning: {name}")
    return plan


def _extract_level_surface(
    field: np.ndarray, coordinates: np.ndarray, *, level: float
) -> tuple[np.ndarray, np.ndarray]:
    pitch = float(coordinates[1] - coordinates[0])
    vertices, faces, _, _ = marching_cubes(  # type: ignore[no-untyped-call]
        field,
        level=level,
        spacing=(pitch, pitch, pitch),
        gradient_direction="descent",
        allow_degenerate=False,
    )
    vertices += float(coordinates[0])
    return vertices.astype(np.float64), faces.astype(np.int64)


def _sample_regular_volume(
    values: np.ndarray, vertices: np.ndarray, *, extent: float
) -> np.ndarray:
    grid_coordinates = ((vertices + extent) / (2.0 * extent) * (values.shape[0] - 1)).T
    return np.asarray(
        ndimage.map_coordinates(values, grid_coordinates, order=1, mode="constant", cval=0.0),
        dtype=np.float64,
    )


def _edge_set(faces: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = sorted((int(first), int(second)))
            edges.add((edge[0], edge[1]))
    return edges


def _extract_semantic_offset_layers(
    *,
    field: np.ndarray,
    coordinates: np.ndarray,
    semantic_volumes: dict[str, np.ndarray],
    semantic_extent: float,
    body_vertices: np.ndarray,
    body_weights: np.ndarray,
    level: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]], int]:
    shell_vertices, shell_faces = _extract_level_surface(field, coordinates, level=level)
    vertex_scores = np.stack(
        [
            _sample_regular_volume(semantic_volumes[layer], shell_vertices, extent=semantic_extent)
            for layer in L03_GARMENT_LAYERS
        ],
        axis=1,
    )
    face_scores = vertex_scores[shell_faces].mean(axis=1)
    ownership = np.argmax(face_scores, axis=1)
    confidence = np.max(face_scores, axis=1)
    tree = cKDTree(body_vertices)
    nearest_distance, nearest_body_vertex = tree.query(shell_vertices, k=1)
    layers: dict[str, dict[str, np.ndarray]] = {}
    source_faces: dict[str, np.ndarray] = {}
    for layer_index, layer in enumerate(L03_GARMENT_LAYERS):
        selected_indices = np.flatnonzero((ownership == layer_index) & (confidence >= threshold))
        selected_faces = shell_faces[selected_indices]
        source_faces[layer] = selected_faces
        source_vertices = np.unique(selected_faces)
        remap = np.full(len(shell_vertices), -1, dtype=np.int64)
        remap[source_vertices] = np.arange(len(source_vertices))
        layers[layer] = {
            "vertices": shell_vertices[source_vertices],
            "faces": remap[selected_faces],
            "skinning_weights": body_weights[nearest_body_vertex[source_vertices]],
            "source_level_set_vertex_indices": source_vertices,
            "source_level_set_face_indices": selected_indices,
            "semantic_vertex_scores": vertex_scores[source_vertices, layer_index],
            "nearest_body_vertex": nearest_body_vertex[source_vertices].astype(np.int64),
            "nearest_body_vertex_distance_metres": nearest_distance[source_vertices],
        }
    registered_contact_edges = len(
        _edge_set(source_faces[L03_GARMENT_LAYERS[0]])
        & _edge_set(source_faces[L03_GARMENT_LAYERS[1]])
    )
    return shell_vertices, shell_faces, layers, registered_contact_edges


def build_l03_real_initialization(*, plan_path: Path, output_root: Path) -> Path:
    """Construct and exactly audit L03's first real open garment surfaces."""

    reject_sealed_capability([plan_path, output_root])
    if output_root.exists():
        raise FileExistsError("L03 real initialization output is immutable")
    plan = _load_and_verify_l03_real_initialization_plan(plan_path)
    paths = {name: Path(value) for name, value in plan["input_paths"].items()}
    field, coordinates, body_vertices, body_faces, body_weights = _load_d03_body_inputs(
        paths["d03_field"], paths["d03_mesh"]
    )
    semantics, semantic_extent, _ = _load_semantic_evidence_volume(paths["semantic_volume"])
    settings = plan["construction"]
    extracted = _extract_semantic_offset_layers(
        field=field,
        coordinates=coordinates,
        semantic_volumes=semantics,
        semantic_extent=semantic_extent,
        body_vertices=body_vertices,
        body_weights=body_weights,
        level=settings["body_clearance_level_metres"],
        threshold=settings["semantic_face_threshold"],
    )
    replay = _extract_semantic_offset_layers(
        field=field,
        coordinates=coordinates,
        semantic_volumes=semantics,
        semantic_extent=semantic_extent,
        body_vertices=body_vertices,
        body_weights=body_weights,
        level=settings["body_clearance_level_metres"],
        threshold=settings["semantic_face_threshold"],
    )
    shell_vertices, shell_faces, layers, registered_contact_edges = extracted
    exact_replay = (
        np.array_equal(shell_vertices, replay[0])
        and np.array_equal(shell_faces, replay[1])
        and registered_contact_edges == replay[3]
        and all(
            all(np.array_equal(values[name], replay[2][layer][name]) for name in values)
            for layer, values in layers.items()
        )
    )
    field_extent = float(max(abs(coordinates[0]), abs(coordinates[-1])))
    layer_reports: dict[str, Any] = {}
    gates: dict[str, bool] = {"exact_replay": exact_replay}
    for layer, values in layers.items():
        audit = boundary_loop_audit(values["vertices"], values["faces"])
        clearance = _sample_regular_volume(field, values["vertices"], extent=field_extent)
        body_intersections = _combined_intersections(
            body_vertices, body_faces, values["vertices"], values["faces"]
        )
        layer_reports[layer] = {
            "vertex_count": len(values["vertices"]),
            "face_count": len(values["faces"]),
            "registered_component_count": settings["registered_component_counts"][layer],
            "registered_boundary_loop_count": settings["registered_boundary_loop_counts"][layer],
            "audit": audit,
            "minimum_body_clearance_metres": float(clearance.min()),
            "median_body_clearance_metres": float(np.median(clearance)),
            "maximum_body_clearance_metres": float(clearance.max()),
            "body_layer_intersections": body_intersections,
            "nearest_body_vertex_distance_median_metres": float(
                np.median(values["nearest_body_vertex_distance_metres"])
            ),
            "nearest_body_vertex_distance_maximum_metres": float(
                np.max(values["nearest_body_vertex_distance_metres"])
            ),
        }
        gates[f"{layer}_minimum_size"] = (
            len(values["vertices"]) >= plan["gates"]["minimum_layer_vertices"]
            and len(values["faces"]) >= plan["gates"]["minimum_layer_faces"]
        )
        gates[f"{layer}_registered_topology"] = (
            audit["component_count"] == settings["registered_component_counts"][layer]
            and audit["boundary_loop_count"] == settings["registered_boundary_loop_counts"][layer]
            and audit["boundary_vertex_degrees_two"]
            and not audit["watertight"]
            and audit["winding_consistent"]
        )
        gates[f"{layer}_exact_self_intersection_free"] = not audit["exact_self_intersections"]
        gates[f"{layer}_body_intersection_free"] = not body_intersections
        gates[f"{layer}_clearance_band"] = (
            float(clearance.min()) >= plan["gates"]["minimum_body_clearance_metres"]
            and float(clearance.max()) <= plan["gates"]["maximum_body_clearance_metres"]
        )
    gates["registered_interlayer_contact_present"] = (
        registered_contact_edges >= plan["gates"]["minimum_registered_contact_edges"]
    )
    gates["body_hash_immutable"] = (
        sha256_file(paths["d03_mesh"]) == plan["input_hashes"]["d03_mesh"]
    )
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError("L03 real initialization staging output already exists")
    staging.mkdir(parents=True)
    try:
        artifacts: dict[str, Any] = {}
        for layer, values in layers.items():
            artifact_path = staging / f"{layer}_initial_surface.npz"
            np.savez_compressed(artifact_path, **values)  # type: ignore[arg-type]
            artifacts[layer] = {
                "path": str(output_root / artifact_path.name),
                "sha256": sha256_file(artifact_path),
            }
        report = {
            "schema_version": L03_REAL_INITIALIZATION_REPORT_SCHEMA,
            "experiment_id": L03_EXPERIMENT_ID,
            "status": "pass" if all(gates.values()) else "fail",
            "decision": (
                "real_open_layer_initialization_qualified"
                if all(gates.values())
                else "real_open_layer_initialization_failed"
            ),
            "source_revision": plan["source_revision"],
            "plan_sha256": sha256_file(plan_path),
            "immutable_body": {
                "field_path": str(paths["d03_field"]),
                "field_sha256": plan["input_hashes"]["d03_field"],
                "mesh_path": str(paths["d03_mesh"]),
                "mesh_sha256": plan["input_hashes"]["d03_mesh"],
                "role": settings["body_role"],
            },
            "derived_partition_source": {
                "positive_level_metres": settings["body_clearance_level_metres"],
                "vertex_count": len(shell_vertices),
                "face_count": len(shell_faces),
                "exported_as_authoritative_layer": False,
            },
            "layers": layer_reports,
            "registered_interlayer_contact_edge_count": registered_contact_edges,
            "artifacts": artifacts,
            "gates": gates,
            "provenance": plan["provenance"],
        }
        report_path = staging / "real_initialization_report.json"
        write_json(report_path, report)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / "real_initialization_report.json"
