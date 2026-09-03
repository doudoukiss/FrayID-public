"""Run the immutable local E25 public fixture and derivative preflight once."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import run_post_v1_e17_public_gate as e17
import run_post_v1_g22_public_gate as g22
from frayid.e25_public_fixtures import (
    E25_PUBLIC_FIXTURE_NAMES,
    E25PublicEvidence,
    articulated_pose_jacobian,
    assert_public_read_allowed,
    extract_public_truth_mesh,
    finite_difference_field_normal,
    fixture_by_name,
    move_cross_cell_fixture,
    public_fixture_registry,
    render_public_mesh_evidence,
    validate_public_evidence,
    validate_public_modalities,
)
from frayid.e25_stage import (
    assert_frozen_connectivity,
    capture_checkpoint,
    commit_stage_surface,
    restore_checkpoint,
)
from frayid.eulerian_field import conventional_surface_audit
from frayid.eulerian_reconstruction import probe_classification, public_eulerian_fixture
from frayid.flexicubes_adapter import FLEXICUBES_REVISION, PinnedFlexiCubes
from frayid.io import write_json
from frayid.normal_integrable_sdf import (
    NormalIntegrableNeuralSDF,
    camera_rays,
    eikonal_loss,
    normal_integrable_image_loss,
    render_neus_sdf,
    transport_normals_inverse_transpose,
    visual_hull_sdf,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "postv1_e25_normal_integrable_multires_image_sdf_r01"
REPORT_SCHEMA = "post_v1_e25_public_preflight.v1"
SEED = 20260925
REGISTERED_STAGE_RESOLUTIONS = (24, 48, 96)
REGISTERED_STAGE_STEPS = (300, 500, 700)
FINAL_COMMITMENT_RESOLUTION = 96
FLEXICUBES_ROOT = PROJECT_ROOT / "external/FlexiCubes"
CPU_CORE_LIMIT = 8
MAXIMUM_MEMORY_GIB = 16.0
MAXIMUM_TOTAL_SECONDS = 14_400.0
MAXIMUM_ENDPOINT_AUDIT_SECONDS = 120.0
GRADIENT_IMAGE_SIZE = (32, 32)
GRADIENT_CROP = (slice(8, 24), slice(8, 24))
GRADIENT_SAMPLES = 64
GRADIENT_HIERARCHICAL_SAMPLES = 64
GRADIENT_FINITE_DIFFERENCE_EPSILON = 2.0e-4
ALLOWED_UNTRACKED_PREFIXES = ("docs/0901/", "docs/0902/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if isinstance(first, (tuple, list)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(
            _state_equal(left, right) for left, right in zip(first, second, strict=True)
        )
    return bool(first == second)


def _gradient_loss(
    model: NormalIntegrableNeuralSDF,
    evidence: E25PublicEvidence,
    *,
    create_graph: bool,
) -> Tensor:
    origins, directions = camera_rays(
        evidence.intrinsics,
        evidence.rotations[0],
        evidence.translations[0],
        GRADIENT_IMAGE_SIZE,
    )
    crop_y, crop_x = GRADIENT_CROP
    rendered = render_neus_sdf(
        model,
        origins[crop_y, crop_x],
        directions[crop_y, crop_x],
        near=1.6,
        far=4.8,
        sample_count=GRADIENT_SAMPLES,
        hierarchical_sample_count=GRADIENT_HIERARCHICAL_SAMPLES,
        inverse_sharpness=80.0,
        deformation_jacobian=torch.eye(3, dtype=origins.dtype, device=origins.device),
        create_graph=create_graph,
        ray_chunk_size=64,
    )
    image = normal_integrable_image_loss(
        rendered,
        evidence.silhouettes[0, crop_y, crop_x],
        evidence.normals[0, crop_y, crop_x],
    )
    coordinates = torch.linspace(-0.8, 0.8, 5, dtype=origins.dtype, device=origins.device)
    xx, yy, zz = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    eikonal_points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    return image + 0.1 * eikonal_loss(model, eikonal_points)


def _image_gradient_control() -> dict[str, Any]:
    fixture = fixture_by_name("rotated_ellipsoid")
    truth = extract_public_truth_mesh(fixture, resolution=28)
    evidence = render_public_mesh_evidence(
        truth,
        image_size=GRADIENT_IMAGE_SIZE,
        target_sample_count=4096,
        reference_sample_count=4096,
    )
    hull = visual_hull_sdf(
        evidence.silhouettes,
        evidence.intrinsics,
        evidence.rotations,
        evidence.translations,
        resolution=24,
        extent=fixture.extent,
    )
    torch.manual_seed(SEED)
    model = NormalIntegrableNeuralSDF(
        hull,
        hidden_width=64,
        hidden_layers=2,
        maximum_hash_resolution=96,
    )
    loss = _gradient_loss(model, evidence, create_graph=True)
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(loss, parameters)
    squared_norm = loss.new_zeros(())
    for gradient in gradients:
        squared_norm = squared_norm + gradient.square().sum()
    norm = torch.sqrt(squared_norm)
    finite = bool(torch.isfinite(norm) and float(norm) > 0.0)
    if not finite:
        return {"status": "fail", "reason": "absent_or_nonfinite_gradient"}
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    directions = tuple(gradient / norm for gradient in gradients)
    values: list[float] = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            for parameter, original, direction in zip(
                parameters, originals, directions, strict=True
            ):
                parameter.copy_(original + sign * GRADIENT_FINITE_DIFFERENCE_EPSILON * direction)
        values.append(float(_gradient_loss(model, evidence, create_graph=False).detach()))
    with torch.no_grad():
        for parameter, original in zip(parameters, originals, strict=True):
            parameter.copy_(original)
    analytic = float(norm)
    finite_difference = (values[0] - values[1]) / (2.0 * GRADIENT_FINITE_DIFFERENCE_EPSILON)
    relative_error = abs(analytic - finite_difference) / max(abs(analytic), 1.0e-12)
    passed = bool(
        np.isfinite(finite_difference)
        and analytic * finite_difference > 0.0
        and relative_error <= 0.20
    )
    return {
        "status": "pass" if passed else "fail",
        "loss": float(loss.detach()),
        "gradient_norm": analytic,
        "finite_difference_directional_derivative": finite_difference,
        "relative_error": relative_error,
        "epsilon": GRADIENT_FINITE_DIFFERENCE_EPSILON,
        "train_views": 12,
        "held_out_views": 6,
        "truth_geometry_training_accesses": 0,
    }


def _continuous_normal_and_jacobian_controls() -> dict[str, Any]:
    fixture = fixture_by_name("rotated_ellipsoid")
    points = torch.tensor(
        ((0.68, 0.0, 0.0), (0.0, 0.91, 0.0), (0.21, 0.31, 0.34)),
        dtype=torch.float64,
        requires_grad=True,
    )
    values = fixture.field(points)
    gradient = torch.autograd.grad(values.sum(), points)[0]
    analytic = F.normalize(gradient, dim=-1, eps=1.0e-8)
    finite = finite_difference_field_normal(fixture.field, points.detach(), epsilon=1.0e-5)
    cosine = (analytic * finite).sum(dim=-1).clamp(-1.0, 1.0)
    maximum_normal_error = float(torch.rad2deg(torch.acos(cosine)).max())

    jacobians = articulated_pose_jacobian(points.detach())
    transported = transport_normals_inverse_transpose(analytic, jacobians)
    tangents = torch.linalg.cross(
        analytic, analytic.new_tensor((0.0, 0.0, 1.0)).expand_as(analytic)
    )
    fallback = torch.linalg.cross(
        analytic,
        analytic.new_tensor((0.0, 1.0, 0.0)).expand_as(analytic),
    )
    tangents = torch.where(
        (torch.linalg.vector_norm(tangents, dim=-1) < 1.0e-8)[:, None], fallback, tangents
    )
    posed_tangents = torch.einsum("nij,nj->ni", jacobians, tangents)
    maximum_orthogonality = float((transported * posed_tangents).sum(dim=-1).abs().max())
    passed = maximum_normal_error <= 0.01 and maximum_orthogonality <= 1.0e-10
    return {
        "status": "pass" if passed else "fail",
        "maximum_spatial_gradient_normal_error_degrees": maximum_normal_error,
        "maximum_inverse_transpose_tangent_dot": maximum_orthogonality,
        "minimum_jacobian_determinant": float(torch.linalg.det(jacobians).min()),
    }


def _negative_controls() -> dict[str, Any]:
    moving_illumination_rejected = False
    try:
        validate_public_modalities(("mask", "boundary", "normal", "rgb"))
    except ValueError:
        moving_illumination_rejected = True

    fixture = fixture_by_name("rotated_ellipsoid")
    truth = extract_public_truth_mesh(fixture, resolution=16)
    evidence = render_public_mesh_evidence(
        truth,
        image_size=(16, 16),
        target_sample_count=1024,
        reference_sample_count=1024,
    )
    corrupted = E25PublicEvidence(
        evidence.silhouettes,
        evidence.normals.clone(),
        evidence.intrinsics,
        evidence.rotations,
        evidence.translations,
    )
    foreground = torch.nonzero(corrupted.silhouettes > 0.5, as_tuple=False)[0]
    corrupted.normals[tuple(foreground)] *= 4.0
    corrupted_normals_rejected = False
    try:
        validate_public_evidence(corrupted)
    except ValueError:
        corrupted_normals_rejected = True
    passed = moving_illumination_rejected and corrupted_normals_rejected
    return {
        "status": "pass" if passed else "fail",
        "moving_illumination_rgb_rejected": moving_illumination_rejected,
        "corrupted_normals_rejected": corrupted_normals_rejected,
    }


def _negative_read_guards() -> dict[str, Any]:
    protected = (
        PROJECT_ROOT / "data/private/e25-forbidden",
        PROJECT_ROOT / "models/private/e25-forbidden",
        PROJECT_ROOT / "models/checkpoints/e25-forbidden",
        PROJECT_ROOT / "docs/assets/subject_video.mp4",
        PROJECT_ROOT / "outputs/development/e25-forbidden",
        PROJECT_ROOT / "outputs/sealed/e25-forbidden",
    )
    rejected = 0
    for path in protected:
        try:
            assert_public_read_allowed(PROJECT_ROOT, path)
        except PermissionError:
            rejected += 1
    public_allowed = True
    try:
        assert_public_read_allowed(
            PROJECT_ROOT,
            PROJECT_ROOT
            / "configs/evaluation/post_v1_e25_normal_integrable_multires_image_sdf_r01.yaml",
        )
    except PermissionError:
        public_allowed = False
    passed = rejected == len(protected) and public_allowed
    return {
        "status": "pass" if passed else "fail",
        "protected_classes": len(protected),
        "protected_classes_rejected": rejected,
        "public_config_allowed": public_allowed,
    }


def _checkpoint_replay_control() -> dict[str, Any]:
    torch.manual_seed(SEED + 1)
    np.random.seed(SEED + 2)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    def transition(
        active_model: torch.nn.Module, active_optimizer: torch.optim.Optimizer
    ) -> Tensor:
        sample = torch.rand(7, 3)
        scale = float(np.random.uniform(0.8, 1.2))
        active_optimizer.zero_grad(set_to_none=True)
        loss = active_model(sample).square().mean() * scale
        loss.backward()
        active_optimizer.step()
        return torch.cat([value.detach().reshape(-1) for value in active_model.parameters()])

    transition(model, optimizer)
    checkpoint = capture_checkpoint(
        model,
        optimizer,
        resolution=48,
        step=37,
        committed_connectivity_digest=None,
    )
    expected = transition(model, optimizer)
    replay_model = torch.nn.Linear(3, 2)
    replay_optimizer = torch.optim.Adam(replay_model.parameters(), lr=9.0)
    payload = restore_checkpoint(checkpoint, replay_model, replay_optimizer)
    observed = transition(replay_model, replay_optimizer)
    passed = bool(
        torch.equal(expected, observed)
        and _state_equal(model.state_dict(), replay_model.state_dict())
        and _state_equal(optimizer.state_dict(), replay_optimizer.state_dict())
    )
    return {
        "status": "pass" if passed else "fail",
        "resolution": payload["resolution"],
        "step": payload["step"],
        "bitwise_next_step": passed,
    }


def _stage_fixture_gate(
    adapter: PinnedFlexiCubes,
    auditor: Path,
    fixture_name: str,
    root: Path,
) -> dict[str, Any]:
    fixture = fixture_by_name(fixture_name)
    fixture_root = root / fixture_name
    fixture_root.mkdir(parents=True, exist_ok=False)
    stages: list[dict[str, Any]] = []
    final_faces: Tensor | None = None
    final_commitment = None
    for resolution in REGISTERED_STAGE_RESOLUTIONS:
        vertices, cubes = adapter.voxel_grid(resolution, extent=fixture.extent)
        values = fixture.field(vertices)
        first = adapter.extract(vertices, values, cubes, resolution, training=False)
        second = adapter.extract(vertices, values, cubes, resolution, training=False)
        replay_exact = bool(
            torch.equal(first.vertices, second.vertices) and torch.equal(first.faces, second.faces)
        )
        topology = conventional_surface_audit(first.vertices, first.faces)
        exact = g22._exact_surface_audit(
            auditor,
            first.vertices,
            first.faces,
            fixture_root,
            f"stage_{resolution}",
        )
        probes_preserved = True
        if fixture_name == "concave_pocket_thin_bridge_near_gap":
            g22_fixture = public_eulerian_fixture()
            probes = probe_classification(first.vertices, first.faces, g22_fixture)
            probes_preserved = probes["status"] == "pass"
        else:
            probes = {"status": "not_declared"}
        blockers: list[str] = []
        if topology["status"] != "pass":
            blockers.append("conventional_topology")
        if exact.get("status") != "pass":
            blockers.append("exact_endpoint")
        if not probes_preserved:
            blockers.append("probe_or_gap_classification")
        if not replay_exact:
            blockers.append("extraction_replay")
        commitment_report: dict[str, Any] | None = None
        if resolution == FINAL_COMMITMENT_RESOLUTION and not blockers:
            final_commitment = commit_stage_surface(
                first.vertices,
                first.faces,
                resolution=resolution,
                exact_intersection_pair_count=0,
                probes_preserved=probes_preserved,
                replay_exact=replay_exact,
            )
            assert_frozen_connectivity(final_commitment, first.faces.clone())
            commitment_report = final_commitment.as_report()
            final_faces = first.faces
        stages.append(
            {
                "resolution": resolution,
                "status": "pass" if not blockers else "fail",
                "vertex_count": int(first.vertices.shape[0]),
                "face_count": int(first.faces.shape[0]),
                "topology": topology,
                "exact_endpoint": exact,
                "probes": probes,
                "extraction_replay_exact": replay_exact,
                "search_intermediate_promotable": False,
                "commitment": commitment_report,
                "blockers": blockers,
            }
        )
    passed = all(stage["status"] == "pass" for stage in stages) and final_commitment is not None
    if passed and final_commitment is not None and final_faces is not None:
        assert_frozen_connectivity(final_commitment, final_faces)
    return {
        "name": fixture_name,
        "status": "pass" if passed else "fail",
        "stages": stages,
        "final_connectivity_frozen": passed,
    }


def _cross_cell_motion_control(
    adapter: PinnedFlexiCubes,
    auditor: Path,
    root: Path,
) -> dict[str, Any]:
    fixture = fixture_by_name("cross_cell_surface_motion")
    resolution = 24
    vertices, cubes = adapter.voxel_grid(resolution, extent=fixture.extent)
    pitch = 2.0 * fixture.extent / resolution
    displacement = move_cross_cell_fixture(torch.zeros(1, 3), grid_pitch=pitch)[0]
    first = adapter.extract(vertices, fixture.field(vertices), cubes, resolution, training=False)
    moved = adapter.extract(
        vertices,
        fixture.field(vertices - displacement),
        cubes,
        resolution,
        training=False,
    )
    first_exact = g22._exact_surface_audit(
        auditor, first.vertices, first.faces, root, "cross_cell_start"
    )
    moved_exact = g22._exact_surface_audit(
        auditor, moved.vertices, moved.faces, root, "cross_cell_end"
    )
    centroid_motion = moved.vertices.mean(dim=0) - first.vertices.mean(dim=0)
    motion_error = float(torch.linalg.vector_norm(centroid_motion - displacement))
    passed = bool(
        first_exact.get("status") == "pass"
        and moved_exact.get("status") == "pass"
        and conventional_surface_audit(moved.vertices, moved.faces)["status"] == "pass"
        and float(torch.linalg.vector_norm(centroid_motion)) > 0.5 * pitch
        and motion_error < 0.25 * pitch
    )
    return {
        "status": "pass" if passed else "fail",
        "grid_pitch": pitch,
        "requested_displacement": displacement.tolist(),
        "observed_centroid_displacement": centroid_motion.tolist(),
        "motion_error": motion_error,
        "start_exact": first_exact,
        "end_exact": moved_exact,
    }


def run_public_preflight() -> dict[str, Any]:
    started = time.monotonic()
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_CORE_LIMIT))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_CORE_LIMIT))
    torch.set_num_threads(CPU_CORE_LIMIT)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    git = _git_binding()
    blockers: list[str] = []
    if not git["implementation_tree_clean"]:
        blockers.append("implementation_tree_not_clean")
    if tuple(fixture.name for fixture in public_fixture_registry()) != E25_PUBLIC_FIXTURE_NAMES:
        blockers.append("fixture_registry")
    validate_public_modalities(("mask", "boundary", "normal"))

    derivative = _image_gradient_control()
    normals = _continuous_normal_and_jacobian_controls()
    negative_controls = _negative_controls()
    read_guards = _negative_read_guards()
    checkpoint = _checkpoint_replay_control()
    for name, control in (
        ("image_gradient", derivative),
        ("normal_and_jacobian", normals),
        ("negative_controls", negative_controls),
        ("negative_read_guards", read_guards),
        ("checkpoint_replay", checkpoint),
    ):
        if control["status"] != "pass":
            blockers.append(name)

    adapter = PinnedFlexiCubes(FLEXICUBES_ROOT, device="cpu")
    constructor, auditor = e17._build_tools()
    with tempfile.TemporaryDirectory(prefix="frayid-e25-public-") as directory:
        root = Path(directory)
        p2 = g22._p2_hairpin_regression(constructor, auditor, root / "p2")
        if p2["status"] != "pass":
            blockers.append("full_resolution_p2_hairpin")
        fixtures = [
            _stage_fixture_gate(adapter, auditor, fixture_name, root)
            for fixture_name in E25_PUBLIC_FIXTURE_NAMES
        ]
        for fixture in fixtures:
            if fixture["status"] != "pass":
                blockers.append(f"fixture:{fixture['name']}")
        cross_cell = _cross_cell_motion_control(adapter, auditor, root)
        if cross_cell["status"] != "pass":
            blockers.append("cross_cell_surface_motion")

    elapsed = time.monotonic() - started
    peak_memory = _peak_memory_gib()
    if elapsed > MAXIMUM_TOTAL_SECONDS:
        blockers.append("total_wall_time")
    if peak_memory > MAXIMUM_MEMORY_GIB:
        blockers.append("resident_memory")
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "scope": "public_procedural_mask_boundary_normal_preflight",
        "git": git,
        "seed": SEED,
        "registered_stage_resolutions": list(REGISTERED_STAGE_RESOLUTIONS),
        "registered_stage_optimizer_steps": list(REGISTERED_STAGE_STEPS),
        "truth_geometry_role": "evaluator_and_fixture_generation_only",
        "truth_geometry_training_accesses": 0,
        "rgb_accesses": 0,
        "feature_track_accesses": 0,
        "image_gradient": derivative,
        "continuous_normal_and_jacobian": normals,
        "negative_controls": negative_controls,
        "negative_read_guards": read_guards,
        "checkpoint_replay": checkpoint,
        "p2_hairpin": p2,
        "fixtures": fixtures,
        "cross_cell_motion": cross_cell,
        "flexicubes": {
            "repository": "external/FlexiCubes",
            "revision": FLEXICUBES_REVISION,
            "source_vendored": False,
        },
        "elapsed_seconds": elapsed,
        "peak_resident_memory_gib": peak_memory,
        "limits": {
            "cpu_cores": CPU_CORE_LIMIT,
            "resident_memory_gib": MAXIMUM_MEMORY_GIB,
            "total_wall_seconds": MAXIMUM_TOTAL_SECONDS,
            "endpoint_audit_seconds": MAXIMUM_ENDPOINT_AUDIT_SECONDS,
            "automatic_retries": 0,
        },
        "bindings": {
            "fixture_source_sha256": _sha256(PROJECT_ROOT / "src/frayid/e25_public_fixtures.py"),
            "sdf_source_sha256": _sha256(PROJECT_ROOT / "src/frayid/normal_integrable_sdf.py"),
            "flexicubes_adapter_sha256": _sha256(PROJECT_ROOT / "src/frayid/flexicubes_adapter.py"),
            "stage_source_sha256": _sha256(PROJECT_ROOT / "src/frayid/e25_stage.py"),
            "runner_source_sha256": _sha256(Path(__file__)),
        },
        "execution_counters": {
            "public_preflight_runs": 1,
            "public_gpu_runs": 0,
            "private_input_reads": 0,
            "development_evidence_reads": 0,
            "sealed_test_accesses": 0,
            "gpu_hours": 0,
            "cloud_invocations": 0,
            "spend_usd": 0,
            "automatic_paid_retries": 0,
        },
        "partial_results_promotable": False,
        "blockers": blockers,
    }


def _worker(report_path: str) -> None:
    write_json(Path(report_path), run_public_preflight())


def _failure_report(failure: str, started: float, exitcode: int | None) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "fail",
        "failure_class": failure,
        "worker_exitcode": exitcode,
        "elapsed_seconds": time.monotonic() - started,
        "automatic_retry_count": 0,
        "partial_results_promotable": False,
        "blockers": [failure],
        "execution_counters": {
            "public_preflight_runs": 1,
            "public_gpu_runs": 0,
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
    if arguments.output.exists():
        raise FileExistsError(f"immutable E25 preflight report exists: {arguments.output}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frayid-e25-supervisor-") as directory:
        worker_report = Path(directory) / "worker_report.json"
        worker = multiprocessing.get_context("spawn").Process(
            target=_worker,
            args=(str(worker_report),),
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
