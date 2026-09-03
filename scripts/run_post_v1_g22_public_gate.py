"""Run the frozen public G22 Eulerian image-to-geometry comparison once."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch import Tensor

import run_post_v1_e12_public_gate as e12
import run_post_v1_e17_public_gate as e17
from frayid.differentiable_isosurface import (
    certify_linear_surface_path,
    certify_zero_set_scalar_path,
    same_open_sign_chamber,
)
from frayid.eulerian_field import (
    EulerianImageField,
    conventional_surface_audit,
    fixed_box_boundary_mask,
    project_eulerian_step,
)
from frayid.eulerian_reconstruction import (
    PUBLIC_HELD_OUT_VIEW_COUNT,
    PUBLIC_IMAGE_SIZE,
    PUBLIC_TRAIN_VIEW_COUNT,
    PublicEulerianFixture,
    PublicImageEvidence,
    evaluate_public_images,
    geometry_fidelity,
    probe_classification,
    project_explicit_step,
    public_eulerian_fixture,
    public_image_loss,
    render_public_evidence,
)
from frayid.interface_field import write_interface_mesh
from frayid.io import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e22_eulerian_image_active_surface_r01"
CORRECTNESS_ID = "postv1_g22_eulerian_image_reconstruction_r01"
REPORT_SCHEMA = "post_v1_g22_eulerian_image_reconstruction_gate.v1"
SEED = 20260902
OPTIMIZER_STEPS = 300
REPLAY_STEP = 37
TREATMENT_LEARNING_RATE = 0.15
CONTROL_VERTEX_LEARNING_RATE = 0.006
CONTROL_FIELD_LEARNING_RATE = 0.15
CONTROL_FIELD_CONSISTENCY_WEIGHT = 0.10
SIGNED_AREA_FLOOR = 0.01
UNSIGNED_AREA_FLOOR = 0.10
MINIMUM_RELATIVE_GEOMETRY_IMPROVEMENT = 0.10
MINIMUM_HELD_OUT_IOU = 0.85
MINIMUM_INITIALIZATION_IOU_IMPROVEMENT = 0.10
MAXIMUM_BOUNDARY_ERROR = 0.015
MAXIMUM_NORMAL_ERROR_DEGREES = 25.0
MAXIMUM_SIGNED_TRAIN_HELD_OUT_GAP = 0.05
MAXIMUM_TOTAL_SECONDS = 21_600.0
MAXIMUM_ENDPOINT_AUDIT_SECONDS = 120.0
MAXIMUM_MEMORY_GIB = 16.0
CPU_CORE_LIMIT = 8
ALLOWED_UNTRACKED_PREFIXES = ("docs/0901/", "docs/0902/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_report_path(path: Path) -> str:
    """Use a project-relative report path when the artifact is in the repository."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _git_binding() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    records = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    disallowed = [
        record
        for record in records
        if not (
            record.startswith("?? ")
            and any(record[3:].startswith(prefix) for prefix in ALLOWED_UNTRACKED_PREFIXES)
        )
    ]
    return {
        "revision": revision,
        "implementation_tree_clean": not disallowed,
        "allowed_untracked_advisory_prefixes": list(ALLOWED_UNTRACKED_PREFIXES),
        "disallowed_status_records": disallowed,
    }


def _peak_memory_gib() -> float:
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return maximum / (1024.0**3)
    return maximum * 1024.0 / (1024.0**3)


def _state_equal(first: Any, second: Any) -> bool:
    if isinstance(first, Tensor) and isinstance(second, Tensor):
        return bool(torch.equal(first, second))
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _state_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(
            _state_equal(left, right) for left, right in zip(first, second, strict=True)
        )
    return bool(first == second)


def _certificate_chain_update(digest: Any, step: int, report: dict[str, object]) -> None:
    digest.update(
        json.dumps(
            {"step": step, "certificate": report},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )


def _accumulate_image_gradients(
    vertices: Callable[[], Tensor],
    faces: Tensor,
    evidence: PublicImageEvidence,
    step: int,
) -> float:
    losses: list[float] = []
    for view in range(PUBLIC_TRAIN_VIEW_COUNT):
        loss = public_image_loss(
            vertices(),
            faces,
            evidence,
            view,
            seed=30_000 + step * PUBLIC_TRAIN_VIEW_COUNT + view,
        )
        (loss / PUBLIC_TRAIN_VIEW_COUNT).backward()  # type: ignore[no-untyped-call]
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _treatment_step(
    field: EulerianImageField,
    optimizer: torch.optim.Optimizer,
    evidence: PublicImageEvidence,
    reference_vertices: Tensor,
    step: int,
) -> tuple[float, Any]:
    previous = field.magnitude_logits.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = _accumulate_image_gradients(field.surface_vertices, field.surface_faces, evidence, step)
    optimizer.step()
    result = project_eulerian_step(
        field,
        previous,
        reference_vertices,
        signed_area_floor=SIGNED_AREA_FLOOR,
        unsigned_area_floor=UNSIGNED_AREA_FLOOR,
        optimizer=optimizer,
    )
    return loss, result


def _control_optimizer(vertices: Tensor, field: EulerianImageField) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": [vertices], "lr": CONTROL_VERTEX_LEARNING_RATE},
            {"params": [field.magnitude_logits], "lr": CONTROL_FIELD_LEARNING_RATE},
        ]
    )


def _control_step(
    vertices: Tensor,
    indirect_field: EulerianImageField,
    optimizer: torch.optim.Optimizer,
    evidence: PublicImageEvidence,
    reference_vertices: Tensor,
    indirect_reference: Tensor,
    step: int,
) -> tuple[float, Any, Any]:
    previous_vertices = vertices.detach().clone()
    previous_logits = indirect_field.magnitude_logits.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    image_loss = _accumulate_image_gradients(
        lambda: vertices, indirect_field.surface_faces, evidence, step
    )
    consistency = (vertices - indirect_field.surface_vertices()).square().mean()
    (CONTROL_FIELD_CONSISTENCY_WEIGHT * consistency).backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    explicit = project_explicit_step(
        vertices,
        previous_vertices,
        reference_vertices,
        indirect_field.surface_faces,
        optimizer=optimizer,
        signed_area_floor=SIGNED_AREA_FLOOR,
        unsigned_area_floor=UNSIGNED_AREA_FLOOR,
    )
    indirect = project_eulerian_step(
        indirect_field,
        previous_logits,
        indirect_reference,
        signed_area_floor=SIGNED_AREA_FLOOR,
        unsigned_area_floor=UNSIGNED_AREA_FLOOR,
        optimizer=optimizer,
    )
    return (
        image_loss + CONTROL_FIELD_CONSISTENCY_WEIGHT * float(consistency.detach()),
        explicit,
        indirect,
    )


def _train_treatment(
    fixture: PublicEulerianFixture, evidence: PublicImageEvidence
) -> tuple[EulerianImageField, dict[str, Any]]:
    field = fixture.initial_field()
    reference = field.surface_vertices().detach().clone()
    optimizer = torch.optim.Adam([field.magnitude_logits], lr=TREATMENT_LEARNING_RATE)
    losses: list[float] = []
    scales: list[float] = []
    chain = hashlib.sha256()
    replay_exact = False
    rejected = 0
    replay_checkpoint: dict[str, Any] | None = None
    for step in range(OPTIMIZER_STEPS):
        if step == REPLAY_STEP:
            replay_checkpoint = {
                "field": copy.deepcopy(field.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
        loss, result = _treatment_step(field, optimizer, evidence, reference, step)
        losses.append(loss)
        scales.append(float(result.accepted_scale))
        _certificate_chain_update(chain, step, result.certificate.report())
        if result.rejected:
            rejected += 1
            break
        if step == REPLAY_STEP and replay_checkpoint is not None:
            replay_field = fixture.initial_field()
            replay_field.load_state_dict(replay_checkpoint["field"])
            replay_optimizer = torch.optim.Adam(
                [replay_field.magnitude_logits], lr=TREATMENT_LEARNING_RATE
            )
            replay_optimizer.load_state_dict(replay_checkpoint["optimizer"])
            _, replay_result = _treatment_step(
                replay_field, replay_optimizer, evidence, reference, step
            )
            replay_exact = bool(
                _state_equal(field.state_dict(), replay_field.state_dict())
                and _state_equal(optimizer.state_dict(), replay_optimizer.state_dict())
                and result == replay_result
            )
    return field, {
        "completed_steps": len(losses),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "minimum_accepted_scale": min(scales, default=0.0),
        "rejected_step_count": rejected,
        "certificate_chain_sha256": chain.hexdigest(),
        "exact_next_step_replay": replay_exact,
        "status": (
            "pass" if len(losses) == OPTIMIZER_STEPS and rejected == 0 and replay_exact else "fail"
        ),
    }


def _train_control(
    fixture: PublicEulerianFixture, evidence: PublicImageEvidence
) -> tuple[Tensor, EulerianImageField, dict[str, Any]]:
    indirect_field = fixture.initial_field()
    vertices = torch.nn.Parameter(indirect_field.surface_vertices().detach().clone())
    reference = vertices.detach().clone()
    indirect_reference = indirect_field.surface_vertices().detach().clone()
    optimizer = _control_optimizer(vertices, indirect_field)
    losses: list[float] = []
    explicit_scales: list[float] = []
    indirect_scales: list[float] = []
    chain = hashlib.sha256()
    replay_exact = False
    rejected = 0
    replay_checkpoint: dict[str, Any] | None = None
    for step in range(OPTIMIZER_STEPS):
        if step == REPLAY_STEP:
            replay_checkpoint = {
                "vertices": vertices.detach().clone(),
                "field": copy.deepcopy(indirect_field.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
        loss, explicit, indirect = _control_step(
            vertices,
            indirect_field,
            optimizer,
            evidence,
            reference,
            indirect_reference,
            step,
        )
        losses.append(loss)
        explicit_scales.append(float(explicit.accepted_scale))
        indirect_scales.append(float(indirect.accepted_scale))
        _certificate_chain_update(
            chain,
            step,
            {"explicit": explicit.certificate.report(), "indirect": indirect.certificate.report()},
        )
        if explicit.rejected or indirect.rejected:
            rejected += 1
            break
        if step == REPLAY_STEP and replay_checkpoint is not None:
            replay_field = fixture.initial_field()
            replay_field.load_state_dict(replay_checkpoint["field"])
            replay_vertices = torch.nn.Parameter(replay_checkpoint["vertices"].clone())
            replay_optimizer = _control_optimizer(replay_vertices, replay_field)
            replay_optimizer.load_state_dict(replay_checkpoint["optimizer"])
            _, replay_explicit, replay_indirect = _control_step(
                replay_vertices,
                replay_field,
                replay_optimizer,
                evidence,
                reference,
                indirect_reference,
                step,
            )
            replay_exact = bool(
                torch.equal(vertices, replay_vertices)
                and _state_equal(indirect_field.state_dict(), replay_field.state_dict())
                and _state_equal(optimizer.state_dict(), replay_optimizer.state_dict())
                and explicit == replay_explicit
                and indirect == replay_indirect
            )
    return (
        vertices,
        indirect_field,
        {
            "completed_steps": len(losses),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "minimum_explicit_accepted_scale": min(explicit_scales, default=0.0),
            "minimum_indirect_accepted_scale": min(indirect_scales, default=0.0),
            "rejected_step_count": rejected,
            "certificate_chain_sha256": chain.hexdigest(),
            "exact_next_step_replay": replay_exact,
            "status": (
                "pass"
                if len(losses) == OPTIMIZER_STEPS and rejected == 0 and replay_exact
                else "fail"
            ),
        },
    )


def _exact_surface_audit(
    auditor: Path,
    vertices: Tensor,
    faces: Tensor,
    root: Path,
    name: str,
) -> dict[str, Any]:
    source_path = root / f"{name}_interior_probe.e6mesh"
    mesh_path = root / f"{name}.e10mesh"
    report_path = root / f"{name}_exact_audit.json"
    array = vertices.detach().cpu().double().numpy()
    lower = array.min(axis=0) - 0.25
    upper = array.max(axis=0) + 0.25
    write_interface_mesh(
        source_path,
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
        (lower, upper),
    )
    e12._write_mesh(mesh_path, array, faces.detach().cpu().numpy())
    started = time.monotonic()
    completed = subprocess.run(
        [str(auditor), str(source_path), str(mesh_path), str(report_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=MAXIMUM_ENDPOINT_AUDIT_SECONDS,
    )
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    report["elapsed_seconds"] = time.monotonic() - started
    report["returncode"] = completed.returncode
    report["diagnostic"] = (completed.stdout + completed.stderr).strip()
    return report


def _p2_hairpin_regression(constructor: Path, auditor: Path, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    fixture, vertices, faces, certificate = e17._fixture_refinement(constructor, root)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    exterior_inside = np.asarray(mesh.contains(fixture.exterior_probes), dtype=np.bool_)
    exact, diagnostic, elapsed = e17._exact_endpoint_audit(
        auditor,
        fixture.source_vertices,
        fixture.source_faces,
        vertices,
        faces,
        root,
        "p2_hairpin",
    )
    blockers: list[str] = []
    if faces.shape[0] != 10_592:
        blockers.append("full_resolution_face_count")
    if certificate.get("status") != "pass":
        blockers.append("parent_provenance_certificate")
    expected_inside = np.ones_like(exterior_inside, dtype=np.bool_)
    if not np.array_equal(exterior_inside, expected_inside):
        blockers.append("historical_p2_probe_classification")
    if exact.get("status") != "pass":
        blockers.append("independent_exact_audit")
    return {
        "status": "pass" if not blockers else "fail",
        "face_count": int(faces.shape[0]),
        "vertex_count": int(vertices.shape[0]),
        "provenance_certificate": certificate,
        "exterior_probe_count": len(exterior_inside),
        "exterior_inside_count": int(np.count_nonzero(exterior_inside)),
        "historical_expected_inside_count": len(exterior_inside),
        "matches_historical_probe_classification": bool(
            np.array_equal(exterior_inside, expected_inside)
        ),
        "exact_audit": exact,
        "exact_diagnostic": diagnostic,
        "exact_audit_elapsed_seconds": elapsed,
        "blockers": blockers,
    }


def _image_gradient_control(
    fixture: PublicEulerianFixture, evidence: PublicImageEvidence
) -> dict[str, Any]:
    field = fixture.initial_field()
    loss = public_image_loss(
        field.surface_vertices(), field.surface_faces, evidence, 0, seed=70_000
    )
    (gradient,) = torch.autograd.grad(loss, (field.magnitude_logits,))
    norm = torch.linalg.vector_norm(gradient)
    direction = gradient / norm.clamp_min(torch.finfo(gradient.dtype).eps)
    original = field.magnitude_logits.detach().clone()
    epsilon = 1.0e-4
    values: list[float] = []
    with torch.no_grad():
        for sign in (1.0, -1.0):
            field.magnitude_logits.copy_(original + sign * epsilon * direction)
            values.append(
                float(
                    public_image_loss(
                        field.surface_vertices(),
                        field.surface_faces,
                        evidence,
                        0,
                        seed=70_000,
                    )
                )
            )
        field.magnitude_logits.copy_(original)
    analytic = float((gradient * direction).sum())
    finite_difference = (values[0] - values[1]) / (2.0 * epsilon)
    status = bool(
        torch.isfinite(gradient).all()
        and float(norm) > 0.0
        and np.isfinite(finite_difference)
        and analytic * finite_difference > 0.0
    )
    return {
        "status": "pass" if status else "fail",
        "loss": float(loss.detach()),
        "gradient_norm": float(norm),
        "analytic_directional_derivative": analytic,
        "finite_difference_directional_derivative": finite_difference,
        "epsilon": epsilon,
    }


def _tangent_shape_distinction() -> dict[str, Any]:
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    points = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)
    tangent_endpoint = np.roll(points, 1, axis=0)
    tangent_material_rms = float(np.sqrt(np.mean(np.square(tangent_endpoint - points))))
    tangent_set_error = float(
        np.mean(
            np.min(np.linalg.norm(points[:, None] - tangent_endpoint[None, :], axis=-1), axis=1)
        )
    )
    shape_endpoint = points * 1.1
    shape_set_error = float(
        np.mean(np.min(np.linalg.norm(points[:, None] - shape_endpoint[None, :], axis=-1), axis=1))
    )
    status = tangent_material_rms > 0.0 and tangent_set_error == 0.0 and shape_set_error > 0.0
    return {
        "status": "pass" if status else "fail",
        "tangent_material_rms": tangent_material_rms,
        "tangent_geometric_set_error": tangent_set_error,
        "shape_geometric_set_error": shape_set_error,
    }


def _mathematical_controls(
    fixture: PublicEulerianFixture, evidence: PublicImageEvidence
) -> dict[str, Any]:
    initial = fixture.initial_field()
    reference = initial.surface_vertices().detach().cpu().double().numpy()
    start = initial.field_values.detach().cpu().double().numpy()
    end = start.copy()
    end[start < 0] *= 1.01
    end[start > 0] *= 0.99
    valid_path = certify_zero_set_scalar_path(
        initial.positions.cpu().double().numpy(),
        initial.surface_edges.cpu().numpy(),
        initial.surface_faces.cpu().numpy(),
        start,
        end,
        reference,
    )
    sign_change = end.copy()
    sign_change[0] *= -1.0
    invalid_path = certify_zero_set_scalar_path(
        initial.positions.cpu().double().numpy(),
        initial.surface_edges.cpu().numpy(),
        initial.surface_faces.cpu().numpy(),
        start,
        sign_change,
        reference,
    )
    collapse = certify_linear_surface_path(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        np.asarray([[0, 1, 2]], dtype=np.int64),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        maximum_subdivision_depth=8,
    )

    boundary_rejected = False
    invalid_boundary = fixture.initial_values.clone()
    boundary = fixed_box_boundary_mask(fixture.positions)
    invalid_boundary[torch.nonzero(boundary, as_tuple=False)[0, 0]] *= -1.0
    try:
        EulerianImageField(fixture.positions, fixture.tetrahedra, invalid_boundary)
    except ValueError:
        boundary_rejected = True

    domain_rejected = False
    invalid_tetrahedra = fixture.tetrahedra.clone()
    invalid_tetrahedra[0, 1] = invalid_tetrahedra[0, 0]
    try:
        EulerianImageField(fixture.positions, invalid_tetrahedra, fixture.initial_values)
    except ValueError:
        domain_rejected = True

    target = fixture.target_values.clone()
    target[0] *= -1.0
    out_of_chamber_unsupported = not same_open_sign_chamber(fixture.initial_values, target)
    image_gradient = _image_gradient_control(fixture, evidence)
    tangent_shape = _tangent_shape_distinction()
    controls = {
        "image_only_gradient": image_gradient,
        "valid_scalar_path": valid_path.report(),
        "sign_change_path": invalid_path.report(),
        "interior_explicit_collapse": collapse.report(),
        "invalid_outer_boundary_rejected": boundary_rejected,
        "invalid_domain_rejected": domain_rejected,
        "out_of_chamber_reported_unsupported": out_of_chamber_unsupported,
        "tangent_shape_distinction": tangent_shape,
    }
    status = bool(
        image_gradient["status"] == "pass"
        and valid_path.status == "pass"
        and invalid_path.status == "fail"
        and collapse.status != "pass"
        and boundary_rejected
        and domain_rejected
        and out_of_chamber_unsupported
        and tangent_shape["status"] == "pass"
    )
    return {"status": "pass" if status else "fail", **controls}


def _arm_blockers(
    name: str,
    training: dict[str, Any],
    images: dict[str, float],
    initial_images: dict[str, float],
    topology: dict[str, object],
    exact: dict[str, Any],
    probes: dict[str, object],
) -> list[str]:
    blockers: list[str] = []
    if training["status"] != "pass":
        blockers.append(f"{name}:training_or_replay")
    if images["held_out_silhouette_iou"] < MINIMUM_HELD_OUT_IOU:
        blockers.append(f"{name}:held_out_iou")
    if (
        images["held_out_silhouette_iou"] - initial_images["held_out_silhouette_iou"]
        < MINIMUM_INITIALIZATION_IOU_IMPROVEMENT
    ):
        blockers.append(f"{name}:initialization_iou_improvement")
    if images["normalized_boundary_error"] > MAXIMUM_BOUNDARY_ERROR:
        blockers.append(f"{name}:boundary")
    if images["pooled_normal_error_degrees"] > MAXIMUM_NORMAL_ERROR_DEGREES:
        blockers.append(f"{name}:normal")
    if images["signed_train_held_out_gap"] > MAXIMUM_SIGNED_TRAIN_HELD_OUT_GAP:
        blockers.append(f"{name}:train_held_out_gap")
    if topology["status"] != "pass":
        blockers.append(f"{name}:conventional_topology")
    if exact.get("status") != "pass":
        blockers.append(f"{name}:exact_endpoint")
    if probes["status"] != "pass":
        blockers.append(f"{name}:gap_probes")
    return blockers


def _comparison_blockers(
    control_images: dict[str, float],
    treatment_images: dict[str, float],
    control_geometry: dict[str, Any],
    treatment_geometry: dict[str, Any],
) -> tuple[list[str], float]:
    control_error = float(control_geometry["bidirectional_mean_distance_pitch"])
    treatment_error = float(treatment_geometry["bidirectional_mean_distance_pitch"])
    improvement = (control_error - treatment_error) / max(control_error, 1.0e-12)
    blockers: list[str] = []
    if improvement < MINIMUM_RELATIVE_GEOMETRY_IMPROVEMENT:
        blockers.append("treatment_truth_geometry_benefit")
    if treatment_geometry["status"] != "pass":
        blockers.append("treatment_inherited_public_fidelity")
    for metric in ("held_out_silhouette_iou",):
        if treatment_images[metric] < control_images[metric]:
            blockers.append(f"treatment_nonregression:{metric}")
    for metric in ("normalized_boundary_error", "pooled_normal_error_degrees"):
        if treatment_images[metric] > control_images[metric]:
            blockers.append(f"treatment_nonregression:{metric}")
    if treatment_images["signed_train_held_out_gap"] > control_images["signed_train_held_out_gap"]:
        blockers.append("treatment_nonregression:signed_train_held_out_gap")
    return blockers, improvement


def _write_artifact(
    path: Path,
    fixture: PublicEulerianFixture,
    control_vertices: Tensor,
    control_field: EulerianImageField,
    treatment: EulerianImageField,
) -> None:
    if path.exists():
        raise FileExistsError(f"immutable G22 artifact exists: {path}")
    np.savez_compressed(
        path,
        positions=fixture.positions.cpu().numpy(),
        tetrahedra=fixture.tetrahedra.cpu().numpy(),
        surface_faces=treatment.surface_faces.cpu().numpy(),
        initial_values=fixture.initial_values.cpu().numpy(),
        target_values=fixture.target_values.cpu().numpy(),
        control_vertices=control_vertices.detach().cpu().numpy(),
        control_indirect_values=control_field.field_values.detach().cpu().numpy(),
        treatment_values=treatment.field_values.detach().cpu().numpy(),
        treatment_vertices=treatment.surface_vertices().detach().cpu().numpy(),
    )


def run_public_gate(artifact_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_CORE_LIMIT))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_CORE_LIMIT))
    torch.set_num_threads(CPU_CORE_LIMIT)
    torch.manual_seed(SEED)
    git = _git_binding()
    blockers: list[str] = []
    if not git["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")

    fixture = public_eulerian_fixture()
    target = fixture.target_field()
    initial = fixture.initial_field()
    evidence = render_public_evidence(target.surface_vertices().detach(), target.surface_faces)
    controls = _mathematical_controls(fixture, evidence)
    if controls["status"] != "pass":
        blockers.append("mathematical_controls")
    target_topology = conventional_surface_audit(target.surface_vertices(), target.surface_faces)
    initial_topology = conventional_surface_audit(initial.surface_vertices(), initial.surface_faces)
    target_probes = probe_classification(target.surface_vertices(), target.surface_faces, fixture)
    initial_probes = probe_classification(
        initial.surface_vertices(), initial.surface_faces, fixture
    )
    if target_topology["status"] != "pass" or initial_topology["status"] != "pass":
        blockers.append("initial_or_target_topology")
    if target_probes["status"] != "pass" or initial_probes["status"] != "pass":
        blockers.append("initial_or_target_gap_probes")

    constructor, auditor = e17._build_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-g22-public-") as directory:
        root = Path(directory)
        p2_regression = _p2_hairpin_regression(constructor, auditor, root / "p2")
        if p2_regression["status"] != "pass":
            blockers.append("p2_hairpin_regression")
        initial_exact = _exact_surface_audit(
            auditor, initial.surface_vertices(), initial.surface_faces, root, "initial"
        )
        target_exact = _exact_surface_audit(
            auditor, target.surface_vertices(), target.surface_faces, root, "target"
        )
        if initial_exact.get("status") != "pass" or target_exact.get("status") != "pass":
            blockers.append("initial_or_target_exact_audit")

        initial_images = evaluate_public_images(
            initial.surface_vertices(), initial.surface_faces, evidence
        )
        control_vertices, control_field, control_training = _train_control(fixture, evidence)
        treatment, treatment_training = _train_treatment(fixture, evidence)
        control_images = evaluate_public_images(
            control_vertices, control_field.surface_faces, evidence
        )
        treatment_images = evaluate_public_images(
            treatment.surface_vertices(), treatment.surface_faces, evidence
        )
        control_geometry = geometry_fidelity(
            target.surface_vertices(),
            control_vertices,
            control_field.surface_faces,
            pitch=fixture.pitch,
        )
        treatment_geometry = geometry_fidelity(
            target.surface_vertices(),
            treatment.surface_vertices(),
            treatment.surface_faces,
            pitch=fixture.pitch,
        )
        control_topology = conventional_surface_audit(control_vertices, control_field.surface_faces)
        treatment_topology = conventional_surface_audit(
            treatment.surface_vertices(), treatment.surface_faces
        )
        control_probes = probe_classification(
            control_vertices, control_field.surface_faces, fixture
        )
        treatment_probes = probe_classification(
            treatment.surface_vertices(), treatment.surface_faces, fixture
        )
        control_exact = _exact_surface_audit(
            auditor, control_vertices, control_field.surface_faces, root, "control"
        )
        treatment_exact = _exact_surface_audit(
            auditor,
            treatment.surface_vertices(),
            treatment.surface_faces,
            root,
            "treatment",
        )
    blockers.extend(
        _arm_blockers(
            "control",
            control_training,
            control_images,
            initial_images,
            control_topology,
            control_exact,
            control_probes,
        )
    )
    blockers.extend(
        _arm_blockers(
            "treatment",
            treatment_training,
            treatment_images,
            initial_images,
            treatment_topology,
            treatment_exact,
            treatment_probes,
        )
    )
    comparison_blockers, improvement = _comparison_blockers(
        control_images, treatment_images, control_geometry, treatment_geometry
    )
    blockers.extend(comparison_blockers)
    artifact_root.mkdir(parents=True, exist_ok=False)
    artifact_path = artifact_root / "g22_endpoint_state.npz"
    _write_artifact(artifact_path, fixture, control_vertices, control_field, treatment)
    elapsed = time.monotonic() - started
    peak_memory = _peak_memory_gib()
    if elapsed > MAXIMUM_TOTAL_SECONDS:
        blockers.append("total_wall_time")
    if peak_memory > MAXIMUM_MEMORY_GIB:
        blockers.append("resident_memory")
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "correctness_id": CORRECTNESS_ID,
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_images_and_geometry_only",
        "git": git,
        "seed": SEED,
        "fixture": {
            "grid_vertex_count": int(fixture.positions.shape[0]),
            "tetrahedron_count": int(fixture.tetrahedra.shape[0]),
            "surface_vertex_count": int(target.surface_vertices().shape[0]),
            "surface_face_count": int(target.surface_faces.shape[0]),
            "pitch": fixture.pitch,
            "initialization_source": "public_sign_template_and_vertex_ordinal_only",
            "truth_geometry_training_accesses": 0,
            "target_topology": target_topology,
            "initial_topology": initial_topology,
            "target_probes": target_probes,
            "initial_probes": initial_probes,
            "initial_exact_audit": initial_exact,
            "target_exact_audit": target_exact,
        },
        "controls": controls,
        "p2_hairpin_regression": p2_regression,
        "initial_images": initial_images,
        "control": {
            "training": control_training,
            "images": control_images,
            "geometry": control_geometry,
            "topology": control_topology,
            "probes": control_probes,
            "exact_endpoint_audit": control_exact,
        },
        "treatment": {
            "training": treatment_training,
            "images": treatment_images,
            "geometry": treatment_geometry,
            "topology": treatment_topology,
            "probes": treatment_probes,
            "exact_endpoint_audit": treatment_exact,
        },
        "comparison": {
            "relative_bidirectional_geometry_error_improvement": improvement,
            "required_minimum": MINIMUM_RELATIVE_GEOMETRY_IMPROVEMENT,
            "blockers": comparison_blockers,
        },
        "artifact": _portable_report_path(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "elapsed_seconds": elapsed,
        "peak_resident_memory_gib": peak_memory,
        "limits": {
            "cpu_cores": CPU_CORE_LIMIT,
            "resident_memory_gib": MAXIMUM_MEMORY_GIB,
            "total_wall_seconds": MAXIMUM_TOTAL_SECONDS,
            "endpoint_audit_seconds": MAXIMUM_ENDPOINT_AUDIT_SECONDS,
            "image_size": list(PUBLIC_IMAGE_SIZE),
            "train_views": PUBLIC_TRAIN_VIEW_COUNT,
            "held_out_views": PUBLIC_HELD_OUT_VIEW_COUNT,
            "optimizer_steps_per_arm": OPTIMIZER_STEPS,
            "automatic_retries": 0,
        },
        "optimizer_contract": {
            "optimizer": "Adam",
            "treatment_learning_rate": TREATMENT_LEARNING_RATE,
            "control_vertex_learning_rate": CONTROL_VERTEX_LEARNING_RATE,
            "control_indirect_field_learning_rate": CONTROL_FIELD_LEARNING_RATE,
            "control_field_consistency_weight": CONTROL_FIELD_CONSISTENCY_WEIGHT,
            "silhouette_weight": 1.0,
            "boundary_weight": 0.5,
            "normal_weight": 0.25,
        },
        "bindings": {
            "eulerian_reconstruction_source_sha256": _sha256(
                PROJECT_ROOT / "src/frayid/eulerian_reconstruction.py"
            ),
            "eulerian_field_source_sha256": _sha256(PROJECT_ROOT / "src/frayid/eulerian_field.py"),
            "isosurface_source_sha256": _sha256(
                PROJECT_ROOT / "src/frayid/differentiable_isosurface.py"
            ),
            "runner_source_sha256": _sha256(Path(__file__)),
        },
        "execution_counters": {
            "public_runs": 1,
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "sealed_test_accesses": 0,
            "gpu_hours": 0,
            "cloud_invocations": 0,
            "spend_usd": 0,
            "automatic_paid_retries": 0,
        },
        "blockers": blockers,
    }


def _worker(report_path: str, artifact_root: str) -> None:
    write_json(Path(report_path), run_public_gate(Path(artifact_root)))


def _failure_report(failure: str, started: float, exitcode: int | None) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "correctness_id": CORRECTNESS_ID,
        "status": "fail",
        "failure_class": failure,
        "worker_exitcode": exitcode,
        "elapsed_seconds": time.monotonic() - started,
        "automatic_retry_count": 0,
        "partial_results_promoted": False,
        "blockers": [failure],
        "execution_counters": {
            "public_runs": 1,
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "sealed_test_accesses": 0,
            "gpu_hours": 0,
            "cloud_invocations": 0,
            "spend_usd": 0,
            "automatic_paid_retries": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    artifact_root = arguments.output.parent / f"{arguments.output.stem}_artifacts"
    if arguments.output.exists():
        raise FileExistsError(f"immutable G22 report exists: {arguments.output}")
    if artifact_root.exists():
        raise FileExistsError(f"immutable G22 artifact directory exists: {artifact_root}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-g22-supervisor-") as directory:
        worker_report = Path(directory) / "worker_report.json"
        worker = multiprocessing.get_context("spawn").Process(
            target=_worker, args=(str(worker_report), str(artifact_root))
        )
        worker.start()
        worker.join(MAXIMUM_TOTAL_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(30)
            report = _failure_report("total_wall_time", started, worker.exitcode)
        elif worker.exitcode != 0 or not worker_report.is_file():
            report = _failure_report("worker_failure", started, worker.exitcode)
        else:
            report = json.loads(worker_report.read_text())
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
