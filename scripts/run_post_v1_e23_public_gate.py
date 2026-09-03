"""Run the frozen E23 full-rank intrinsic-coordinate public comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

import run_post_v1_e17_public_gate as e17
import run_post_v1_g22_public_gate as g22
from frayid.eulerian_field import conventional_surface_audit
from frayid.eulerian_reconstruction import (
    PUBLIC_HELD_OUT_VIEW_COUNT,
    PUBLIC_IMAGE_SIZE,
    PUBLIC_TRAIN_VIEW_COUNT,
    ExplicitStepResult,
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
from frayid.intrinsic_geometry import (
    IntrinsicGeometryTransform,
    project_intrinsic_step,
)
from frayid.io import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e23_intrinsic_full_rank_geometry_r01"
REPORT_SCHEMA = "post_v1_e23_intrinsic_full_rank_geometry_gate.v1"
PREFLIGHT_SCHEMA = "post_v1_e23_intrinsic_full_rank_geometry_preflight.v1"
SEED = 20260902
LAMBDA_VALUE = 1.0
LEARNING_RATE = 0.006
OPTIMIZER_STEPS = 300
PREFLIGHT_STEPS = 40
REPLAY_STEP = 37
SIGNED_AREA_FLOOR = 0.01
UNSIGNED_AREA_FLOOR = 0.10
MINIMUM_RELATIVE_GEOMETRY_IMPROVEMENT = 0.10
MAXIMUM_TOTAL_SECONDS = 14_400.0
MAXIMUM_MEMORY_GIB = 16.0
CPU_CORE_LIMIT = 8


def _optimizer(parameter: Tensor) -> torch.optim.Optimizer:
    return torch.optim.Adam([parameter], lr=LEARNING_RATE)


def _identity(value: Tensor) -> Tensor:
    return value


def _direct_step(
    vertices: Tensor,
    optimizer: torch.optim.Optimizer,
    evidence: PublicImageEvidence,
    faces: Tensor,
    reference_vertices: Tensor,
    step: int,
) -> tuple[float, ExplicitStepResult]:
    previous = vertices.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = g22._accumulate_image_gradients(lambda: vertices, faces, evidence, step)
    optimizer.step()
    result = project_explicit_step(
        vertices,
        previous,
        reference_vertices,
        faces,
        optimizer=optimizer,
        signed_area_floor=SIGNED_AREA_FLOOR,
        unsigned_area_floor=UNSIGNED_AREA_FLOOR,
    )
    return loss, result


def _intrinsic_step(
    coordinates: Tensor,
    transform: IntrinsicGeometryTransform,
    optimizer: torch.optim.Optimizer,
    evidence: PublicImageEvidence,
    faces: Tensor,
    reference_vertices: Tensor,
    step: int,
) -> tuple[float, ExplicitStepResult]:
    previous = coordinates.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = g22._accumulate_image_gradients(
        lambda: transform.decode(coordinates), faces, evidence, step
    )
    optimizer.step()
    result = project_intrinsic_step(
        transform,
        coordinates,
        previous,
        reference_vertices,
        faces,
        optimizer=optimizer,
        signed_area_floor=SIGNED_AREA_FLOOR,
        unsigned_area_floor=UNSIGNED_AREA_FLOOR,
    )
    return loss, result


def _train_parameterization(
    initial_vertices: Tensor,
    faces: Tensor,
    evidence: PublicImageEvidence,
    *,
    transform: IntrinsicGeometryTransform | None,
    steps: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    if transform is None:
        parameter = torch.nn.Parameter(initial_vertices.clone())
        decode: Callable[[Tensor], Tensor] = _identity
    else:
        parameter = torch.nn.Parameter(transform.encode(initial_vertices).detach().clone())
        decode = transform.decode
    optimizer = _optimizer(parameter)
    losses: list[float] = []
    scales: list[float] = []
    chain = hashlib.sha256()
    rejected = 0
    replay_exact = False
    replay_checkpoint: dict[str, Any] | None = None
    for step in range(steps):
        if step == REPLAY_STEP:
            replay_checkpoint = {
                "parameter": parameter.detach().clone(),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
        if transform is None:
            loss, result = _direct_step(
                parameter, optimizer, evidence, faces, initial_vertices, step
            )
        else:
            loss, result = _intrinsic_step(
                parameter,
                transform,
                optimizer,
                evidence,
                faces,
                initial_vertices,
                step,
            )
        losses.append(loss)
        scales.append(float(result.accepted_scale))
        g22._certificate_chain_update(chain, step, result.certificate.report())
        if result.rejected:
            rejected += 1
            break
        if step == REPLAY_STEP and replay_checkpoint is not None:
            replay_parameter = torch.nn.Parameter(replay_checkpoint["parameter"].clone())
            replay_optimizer = _optimizer(replay_parameter)
            replay_optimizer.load_state_dict(replay_checkpoint["optimizer"])
            if transform is None:
                _, replay_result = _direct_step(
                    replay_parameter,
                    replay_optimizer,
                    evidence,
                    faces,
                    initial_vertices,
                    step,
                )
            else:
                _, replay_result = _intrinsic_step(
                    replay_parameter,
                    transform,
                    replay_optimizer,
                    evidence,
                    faces,
                    initial_vertices,
                    step,
                )
            replay_exact = bool(
                torch.equal(parameter, replay_parameter)
                and torch.equal(decode(parameter), decode(replay_parameter))
                and g22._state_equal(optimizer.state_dict(), replay_optimizer.state_dict())
                and result == replay_result
            )
    endpoint = decode(parameter).detach().clone()
    status = bool(len(losses) == steps and rejected == 0 and replay_exact)
    return (
        endpoint,
        parameter.detach().clone(),
        {
            "parameterization": "direct_v" if transform is None else "intrinsic_u",
            "parameter_scalar_count": parameter.numel(),
            "completed_steps": len(losses),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "minimum_accepted_scale": min(scales, default=0.0),
            "rejected_step_count": rejected,
            "certificate_chain_sha256": chain.hexdigest(),
            "exact_next_step_replay": replay_exact,
            "status": "pass" if status else "fail",
        },
    )


def _gradient_control(
    initial_vertices: Tensor,
    faces: Tensor,
    evidence: PublicImageEvidence,
    transform: IntrinsicGeometryTransform,
) -> dict[str, float | str]:
    coordinates = torch.nn.Parameter(transform.encode(initial_vertices).detach().clone())
    loss = public_image_loss(transform.decode(coordinates), faces, evidence, 0, seed=71_000)
    (gradient,) = torch.autograd.grad(loss, (coordinates,))
    norm = torch.linalg.vector_norm(gradient)
    direction = gradient / norm.clamp_min(torch.finfo(gradient.dtype).eps)
    original = coordinates.detach().clone()
    epsilon = 1.0e-5
    values: list[float] = []
    with torch.no_grad():
        for sign in (1.0, -1.0):
            coordinates.copy_(original + sign * epsilon * direction)
            values.append(
                float(
                    public_image_loss(
                        transform.decode(coordinates), faces, evidence, 0, seed=71_000
                    )
                )
            )
        coordinates.copy_(original)
    analytic = float((gradient * direction).sum())
    finite_difference = (values[0] - values[1]) / (2.0 * epsilon)
    relative_error = abs(analytic - finite_difference) / max(abs(analytic), 1.0e-12)
    passed = bool(
        torch.isfinite(gradient).all()
        and float(norm) > 0.0
        and np.isfinite(finite_difference)
        and analytic * finite_difference > 0.0
        and relative_error <= 0.10
    )
    return {
        "status": "pass" if passed else "fail",
        "loss": float(loss.detach()),
        "gradient_norm": float(norm),
        "analytic_directional_derivative": analytic,
        "finite_difference_directional_derivative": finite_difference,
        "relative_error": relative_error,
        "epsilon": epsilon,
    }


def _reachability_control(
    initial_vertices: Tensor, transform: IntrinsicGeometryTransform
) -> dict[str, float | int | str]:
    translation = initial_vertices + initial_vertices.new_tensor([0.03, -0.02, 0.01])
    local = initial_vertices.clone()
    local[17] += initial_vertices.new_tensor([0.01, -0.015, 0.02])
    denominator = torch.linalg.vector_norm(initial_vertices).clamp_min(
        torch.finfo(initial_vertices.dtype).eps
    )
    translation_error = float(
        torch.linalg.vector_norm(transform.decode(transform.encode(translation)) - translation)
        / denominator
    )
    local_error = float(
        torch.linalg.vector_norm(transform.decode(transform.encode(local)) - local) / denominator
    )
    scalar_dof = initial_vertices.numel()
    passed = bool(translation_error <= 1.0e-12 and local_error <= 1.0e-12)
    return {
        "status": "pass" if passed else "fail",
        "geometric_scalar_dof": scalar_dof,
        "coordinate_scalar_dof": scalar_dof,
        "translation_relative_error": translation_error,
        "local_deformation_relative_error": local_error,
    }


def _public_inputs() -> tuple[
    PublicEulerianFixture,
    Tensor,
    Tensor,
    Tensor,
    PublicImageEvidence,
    IntrinsicGeometryTransform,
]:
    fixture = public_eulerian_fixture()
    target_field = fixture.target_field()
    initial_field = fixture.initial_field()
    faces = initial_field.surface_faces
    target_vertices = target_field.surface_vertices().detach().double()
    initial_vertices = initial_field.surface_vertices().detach().double()
    evidence = render_public_evidence(target_vertices, faces)
    transform = IntrinsicGeometryTransform.from_mesh(
        initial_vertices, faces, lambda_value=LAMBDA_VALUE
    )
    return fixture, target_vertices, initial_vertices, faces, evidence, transform


def run_preflight() -> dict[str, Any]:
    started = time.monotonic()
    torch.manual_seed(SEED)
    git = g22._git_binding()
    fixture, _, initial, faces, evidence, transform = _public_inputs()
    matrix = transform.report(initial).as_dict()
    gradient = _gradient_control(initial, faces, evidence, transform)
    reachability = _reachability_control(initial, transform)
    control_vertices, _, control_training = _train_parameterization(
        initial, faces, evidence, transform=None, steps=PREFLIGHT_STEPS
    )
    treatment_vertices, _, treatment_training = _train_parameterization(
        initial, faces, evidence, transform=transform, steps=PREFLIGHT_STEPS
    )
    control_topology = conventional_surface_audit(control_vertices, faces)
    treatment_topology = conventional_surface_audit(treatment_vertices, faces)
    control_probes = probe_classification(control_vertices, faces, fixture)
    treatment_probes = probe_classification(treatment_vertices, faces, fixture)
    _, auditor = e17._build_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-e23-preflight-") as directory:
        root = Path(directory)
        control_exact = g22._exact_surface_audit(auditor, control_vertices, faces, root, "control")
        treatment_exact = g22._exact_surface_audit(
            auditor, treatment_vertices, faces, root, "treatment"
        )
    blockers: list[str] = []
    if not git["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")
    for name, audit_record in (
        ("matrix", matrix),
        ("gradient", gradient),
        ("reachability", reachability),
        ("control_training", control_training),
        ("treatment_training", treatment_training),
        ("control_topology", control_topology),
        ("treatment_topology", treatment_topology),
        ("control_probes", control_probes),
        ("treatment_probes", treatment_probes),
        ("control_exact", control_exact),
        ("treatment_exact", treatment_exact),
    ):
        if audit_record.get("status") != "pass":
            blockers.append(name)
    elapsed = time.monotonic() - started
    peak_memory = g22._peak_memory_gib()
    if elapsed > MAXIMUM_TOTAL_SECONDS:
        blockers.append("total_wall_time")
    if peak_memory > MAXIMUM_MEMORY_GIB:
        blockers.append("resident_memory")
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "scope": "public_preflight_no_truth_endpoint_metrics",
        "git": git,
        "seed": SEED,
        "matrix": matrix,
        "gradient": gradient,
        "reachability": reachability,
        "control": {
            "training": control_training,
            "topology": control_topology,
            "probes": control_probes,
            "exact_endpoint_audit": control_exact,
        },
        "treatment": {
            "training": treatment_training,
            "topology": treatment_topology,
            "probes": treatment_probes,
            "exact_endpoint_audit": treatment_exact,
        },
        "blockers": blockers,
        "elapsed_seconds": elapsed,
        "peak_resident_memory_gib": peak_memory,
        "execution_counters": {
            "public_runs": 0,
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "sealed_test_accesses": 0,
            "gpu_hours": 0,
            "cloud_invocations": 0,
            "spend_usd": 0,
            "automatic_paid_retries": 0,
        },
    }


def _write_artifact(
    path: Path,
    fixture: PublicEulerianFixture,
    faces: Tensor,
    control_vertices: Tensor,
    treatment_vertices: Tensor,
    treatment_coordinates: Tensor,
    transform: IntrinsicGeometryTransform,
) -> None:
    if path.exists():
        raise FileExistsError(f"immutable E23 artifact exists: {path}")
    np.savez_compressed(
        path,
        target_values=fixture.target_values.numpy(),
        initial_values=fixture.initial_values.numpy(),
        surface_faces=faces.numpy(),
        control_vertices=control_vertices.numpy(),
        treatment_vertices=treatment_vertices.numpy(),
        treatment_coordinates=treatment_coordinates.numpy(),
        intrinsic_matrix=transform.matrix.numpy(),
    )


def run_public_gate(artifact_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    torch.manual_seed(SEED)
    fixture, target, initial, faces, evidence, transform = _public_inputs()
    matrix = transform.report(initial).as_dict()
    gradient = _gradient_control(initial, faces, evidence, transform)
    reachability = _reachability_control(initial, transform)
    blockers: list[str] = []
    git = g22._git_binding()
    if not git["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")
    for name, record in (
        ("matrix", matrix),
        ("gradient", gradient),
        ("reachability", reachability),
    ):
        if record["status"] != "pass":
            blockers.append(name)

    initial_images = evaluate_public_images(initial, faces, evidence)
    target_topology = conventional_surface_audit(target, faces)
    initial_topology = conventional_surface_audit(initial, faces)
    target_probes = probe_classification(target, faces, fixture)
    initial_probes = probe_classification(initial, faces, fixture)
    control_vertices, _, control_training = _train_parameterization(
        initial, faces, evidence, transform=None, steps=OPTIMIZER_STEPS
    )
    treatment_vertices, treatment_coordinates, treatment_training = _train_parameterization(
        initial, faces, evidence, transform=transform, steps=OPTIMIZER_STEPS
    )
    control_images = evaluate_public_images(control_vertices, faces, evidence)
    treatment_images = evaluate_public_images(treatment_vertices, faces, evidence)
    control_geometry = geometry_fidelity(target, control_vertices, faces, pitch=fixture.pitch)
    treatment_geometry = geometry_fidelity(target, treatment_vertices, faces, pitch=fixture.pitch)
    control_topology = conventional_surface_audit(control_vertices, faces)
    treatment_topology = conventional_surface_audit(treatment_vertices, faces)
    control_probes = probe_classification(control_vertices, faces, fixture)
    treatment_probes = probe_classification(treatment_vertices, faces, fixture)
    constructor, auditor = e17._build_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-e23-public-") as directory:
        root = Path(directory)
        p2_regression = g22._p2_hairpin_regression(constructor, auditor, root / "p2")
        initial_exact = g22._exact_surface_audit(auditor, initial, faces, root, "initial")
        target_exact = g22._exact_surface_audit(auditor, target, faces, root, "target")
        control_exact = g22._exact_surface_audit(auditor, control_vertices, faces, root, "control")
        treatment_exact = g22._exact_surface_audit(
            auditor, treatment_vertices, faces, root, "treatment"
        )
    for audit_name, audit_record in (
        ("target_topology", target_topology),
        ("initial_topology", initial_topology),
        ("target_probes", target_probes),
        ("initial_probes", initial_probes),
        ("p2_hairpin_regression", p2_regression),
        ("initial_exact", initial_exact),
        ("target_exact", target_exact),
    ):
        if audit_record.get("status") != "pass":
            blockers.append(audit_name)
    blockers.extend(
        g22._arm_blockers(
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
        g22._arm_blockers(
            "treatment",
            treatment_training,
            treatment_images,
            initial_images,
            treatment_topology,
            treatment_exact,
            treatment_probes,
        )
    )
    comparison_blockers, improvement = g22._comparison_blockers(
        control_images, treatment_images, control_geometry, treatment_geometry
    )
    blockers.extend(comparison_blockers)
    artifact_root.mkdir(parents=True, exist_ok=False)
    artifact_path = artifact_root / "e23_endpoint_state.npz"
    _write_artifact(
        artifact_path,
        fixture,
        faces,
        control_vertices,
        treatment_vertices,
        treatment_coordinates,
        transform,
    )
    elapsed = time.monotonic() - started
    peak_memory = g22._peak_memory_gib()
    if elapsed > MAXIMUM_TOTAL_SECONDS:
        blockers.append("total_wall_time")
    if peak_memory > MAXIMUM_MEMORY_GIB:
        blockers.append("resident_memory")
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_images_and_geometry_only",
        "git": git,
        "seed": SEED,
        "matrix": matrix,
        "gradient": gradient,
        "reachability": reachability,
        "fixture": {
            "vertex_count": int(initial.shape[0]),
            "face_count": int(faces.shape[0]),
            "pitch": fixture.pitch,
            "truth_geometry_training_accesses": 0,
            "target_topology": target_topology,
            "initial_topology": initial_topology,
            "target_probes": target_probes,
            "initial_probes": initial_probes,
            "target_exact_audit": target_exact,
            "initial_exact_audit": initial_exact,
        },
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
        "artifact": g22._portable_report_path(artifact_path),
        "artifact_sha256": g22._sha256(artifact_path),
        "elapsed_seconds": elapsed,
        "peak_resident_memory_gib": peak_memory,
        "limits": {
            "cpu_cores": CPU_CORE_LIMIT,
            "resident_memory_gib": MAXIMUM_MEMORY_GIB,
            "total_wall_seconds": MAXIMUM_TOTAL_SECONDS,
            "image_size": list(PUBLIC_IMAGE_SIZE),
            "train_views": PUBLIC_TRAIN_VIEW_COUNT,
            "held_out_views": PUBLIC_HELD_OUT_VIEW_COUNT,
            "optimizer_steps_per_arm": OPTIMIZER_STEPS,
            "automatic_retries": 0,
        },
        "optimizer_contract": {
            "optimizer": "Adam",
            "learning_rate_per_arm": LEARNING_RATE,
            "silhouette_weight": 1.0,
            "boundary_weight": 0.5,
            "normal_weight": 0.25,
            "extra_smoothing_or_regularization_loss": False,
        },
        "bindings": {
            "intrinsic_geometry_source_sha256": g22._sha256(
                PROJECT_ROOT / "src/frayid/intrinsic_geometry.py"
            ),
            "eulerian_reconstruction_source_sha256": g22._sha256(
                PROJECT_ROOT / "src/frayid/eulerian_reconstruction.py"
            ),
            "runner_source_sha256": g22._sha256(Path(__file__)),
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


def _failure_report(
    mode: str, failure: str, started: float, exitcode: int | None
) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_SCHEMA if mode == "preflight" else REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "fail",
        "failure_class": failure,
        "worker_exitcode": exitcode,
        "elapsed_seconds": time.monotonic() - started,
        "automatic_retry_count": 0,
        "partial_results_promoted": False,
        "blockers": [failure],
        "execution_counters": {
            "public_runs": 0 if mode == "preflight" else 1,
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "sealed_test_accesses": 0,
            "gpu_hours": 0,
            "cloud_invocations": 0,
            "spend_usd": 0,
            "automatic_paid_retries": 0,
        },
    }


def _worker(mode: str, report_path: str, artifact_root: str | None) -> None:
    if mode == "preflight":
        report = run_preflight()
    elif artifact_root is not None:
        report = run_public_gate(Path(artifact_root))
    else:
        raise ValueError("official E23 run requires an artifact root")
    write_json(Path(report_path), report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "official"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"immutable E23 report exists: {arguments.output}")
    artifact_root = (
        arguments.output.parent / f"{arguments.output.stem}_artifacts"
        if arguments.mode == "official"
        else None
    )
    if artifact_root is not None and artifact_root.exists():
        raise FileExistsError(f"immutable E23 artifact directory exists: {artifact_root}")
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_CORE_LIMIT))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_CORE_LIMIT))
    torch.set_num_threads(CPU_CORE_LIMIT)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-e23-supervisor-") as directory:
        worker_report = Path(directory) / "worker_report.json"
        worker = multiprocessing.get_context("spawn").Process(
            target=_worker,
            args=(
                arguments.mode,
                str(worker_report),
                str(artifact_root) if artifact_root is not None else None,
            ),
        )
        worker.start()
        worker.join(MAXIMUM_TOTAL_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join(30)
            report = _failure_report(arguments.mode, "total_wall_time", started, worker.exitcode)
        elif worker.exitcode != 0 or not worker_report.is_file():
            report = _failure_report(arguments.mode, "worker_failure", started, worker.exitcode)
        else:
            report = json.loads(worker_report.read_text())
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
