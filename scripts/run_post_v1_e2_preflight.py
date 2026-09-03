"""Run the public synthetic E2 feasible-cage-direction preflight exactly once."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch import Tensor

from frayid.camera import make_intrinsics
from frayid.deformation_cage import TrilinearDeformationCage, project_cage_step
from frayid.feasible_cage import linearized_feasible_cage_direction
from frayid.geometry import (
    canonical_face_orientation_report,
    canonical_topology_is_valid,
    canonical_topology_quantities,
)
from frayid.io import write_json
from frayid.renderer import (
    normalized_boundary_error,
    render_soft_mesh,
    soft_silhouette_iou,
)

EXPERIMENT_ID = "postv1_e02_feasible_cage_step_r01"
OUTPUT_RELATIVE = Path("outputs/canonical_clothed_surface_v1/post_v1") / EXPERIMENT_ID
SEED = 20260831
CAGE_RESOLUTION = (8, 8, 4)
SIGNED_FLOOR = 0.01
UNSIGNED_FLOOR = 0.10
ACTIVE_SLACK_TOLERANCE = 1e-4
MAXIMUM_BACKTRACKS = 16
HELD_OUT_ANGLES = (35.0, 105.0, 215.0)


def _ellipsoid() -> tuple[Tensor, Tensor]:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float64)
    vertices *= torch.tensor([0.55, 0.95, 0.42], dtype=torch.float64)
    return vertices, torch.tensor(np.asarray(mesh.faces), dtype=torch.long)


def _set_nearly_active_incumbent(cage: TrilinearDeformationCage, faces: Tensor) -> float:
    low, high = 0.0, 0.3
    for _ in range(60):
        amplitude = 0.5 * (low + high)
        with torch.no_grad():
            cage.controls.zero_()
            cage.controls[3, 7, 1, 2] = amplitude
            signed, _ = canonical_topology_quantities(
                cage.reference_vertices, cage.deformed_vertices(), faces
            )
        if float(signed.min()) > 0.011:
            low = amplitude
        else:
            high = amplitude
    with torch.no_grad():
        cage.controls.zero_()
        cage.controls[3, 7, 1, 2] = low
    return low


def _candidate_directions(cage: TrilinearDeformationCage) -> tuple[Tensor, Tensor]:
    x = torch.arange(8, dtype=cage.controls.dtype)[:, None, None]
    y = torch.arange(8, dtype=cage.controls.dtype)[None, :, None]
    z = torch.arange(4, dtype=cage.controls.dtype)[None, None, :]
    distant = torch.exp(-((x - 4.0).square() + y.square() + (z - 2.0).square()) / (2 * 1.8**2))
    bulge = torch.zeros_like(cage.controls)
    bulge[..., 0] = 0.3 * distant
    harmful = torch.zeros_like(cage.controls)
    harmful[3, 7, 1, 2] = 0.08
    return bulge, bulge + harmful


def _pose(vertices: Tensor, angle_degrees: float) -> Tensor:
    angle = vertices.new_tensor(np.deg2rad(angle_degrees))
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros_like(angle)
    rotation = torch.stack(
        (
            torch.stack((cosine, zero, sine)),
            torch.stack((zero, torch.ones_like(angle), zero)),
            torch.stack((-sine, zero, cosine)),
        )
    )
    return vertices @ rotation.T + vertices.new_tensor([0.0, 0.0, 3.0])


def _held_out_metrics(vertices: Tensor, target: Tensor, faces: Tensor) -> dict[str, float]:
    intrinsics = make_intrinsics(70.0, (32.0, 32.0), device="cpu").to(torch.float64)
    ious: list[float] = []
    boundaries: list[float] = []
    angular_errors: list[float] = []
    for ordinal, angle in enumerate(HELD_OUT_ANGLES):
        torch.manual_seed(700 + ordinal)
        target_silhouette, target_normals = render_soft_mesh(
            _pose(target, angle),
            faces,
            intrinsics,
            (64, 64),
            source_image_size=(64, 64),
            sample_count=8192,
            chunk_size=512,
        )
        torch.manual_seed(700 + ordinal)
        silhouette, normals = render_soft_mesh(
            _pose(vertices, angle),
            faces,
            intrinsics,
            (64, 64),
            source_image_size=(64, 64),
            sample_count=8192,
            chunk_size=512,
        )
        valid = target_silhouette > 0.5
        cosine = (normals[valid] * target_normals[valid]).sum(-1).clamp(-1.0, 1.0)
        angular_errors.extend(torch.rad2deg(torch.acos(cosine)).tolist())
        ious.append(float(soft_silhouette_iou(silhouette, target_silhouette)))
        boundaries.append(normalized_boundary_error(silhouette, target_silhouette))
    signed, _ = canonical_topology_quantities(target, vertices, faces)
    return {
        "pooled_held_out_median_normal_error_degrees": float(np.median(angular_errors)),
        "mean_held_out_iou": float(np.mean(ious)),
        "mean_held_out_boundary_error": float(np.mean(boundaries)),
        "mean_absolute_signed_gap": float(torch.mean(torch.abs(signed - 1.0))),
    }


def _topology(vertices: Tensor, deformed: Tensor, faces: Tensor) -> dict[str, Any]:
    return canonical_face_orientation_report(
        vertices.numpy(), deformed.numpy(), faces.numpy(), minimum_area_ratio=UNSIGNED_FLOOR
    )


def _auxiliary_fixtures() -> dict[str, Any]:
    sphere_vertices, sphere_faces = _ellipsoid()
    sphere_vertices = sphere_vertices / sphere_vertices.new_tensor([0.55, 0.95, 0.42]) * 0.7
    sphere_cage = TrilinearDeformationCage(sphere_vertices, CAGE_RESOLUTION)
    sphere_delta = torch.zeros_like(sphere_cage.controls)
    sphere_delta[..., 2] = 0.005
    sphere_projected, sphere_report = linearized_feasible_cage_direction(
        sphere_cage, sphere_delta, sphere_faces
    )
    with torch.no_grad():
        sphere_cage.controls.add_(sphere_projected)
    sphere_valid = canonical_topology_is_valid(
        sphere_vertices,
        sphere_cage.deformed_vertices(),
        sphere_faces,
        minimum_signed_area_ratio=SIGNED_FLOOR,
        minimum_area_ratio=UNSIGNED_FLOOR,
    )

    bridge = trimesh.creation.box(extents=[1.4, 0.18, 0.12])
    rotation = trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
        np.deg2rad(31.0), [0.3, 0.8, 0.5]
    )
    bridge.apply_transform(rotation)
    bridge_vertices = torch.tensor(np.asarray(bridge.vertices), dtype=torch.float64)
    bridge_faces = torch.tensor(np.asarray(bridge.faces), dtype=torch.long)
    bridge_cage = TrilinearDeformationCage(bridge_vertices, CAGE_RESOLUTION)
    bridge_delta = torch.zeros_like(bridge_cage.controls)
    bridge_delta[..., 1] = 0.003
    bridge_projected, bridge_report = linearized_feasible_cage_direction(
        bridge_cage, bridge_delta, bridge_faces
    )
    with torch.no_grad():
        bridge_cage.controls.add_(bridge_projected)
    bridge_valid = canonical_topology_is_valid(
        bridge_vertices,
        bridge_cage.deformed_vertices(),
        bridge_faces,
        minimum_signed_area_ratio=SIGNED_FLOOR,
        minimum_area_ratio=UNSIGNED_FLOOR,
    )
    return {
        "sphere": {
            "valid": sphere_valid,
            "finite": bool(torch.isfinite(sphere_projected).all()),
            "qp": sphere_report.__dict__,
        },
        "rotated_thin_bridge": {
            "valid": bridge_valid,
            "finite": bool(torch.isfinite(bridge_projected).all()),
            "qp": bridge_report.__dict__,
        },
    }


def _resume_check(cage: TrilinearDeformationCage, target: Tensor) -> bool:
    optimizer = torch.optim.Adam(cage.parameters(), lr=0.002)
    optimizer.zero_grad(set_to_none=True)
    loss = (cage.deformed_vertices() - target).square().mean()
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    buffer = io.BytesIO()
    torch.save({"cage": cage.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    restored_cage = TrilinearDeformationCage(cage.reference_vertices, cage.resolution)
    restored_optimizer = torch.optim.Adam(restored_cage.parameters(), lr=0.002)
    payload = torch.load(buffer, weights_only=False)
    restored_cage.load_state_dict(payload["cage"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    return torch.equal(cage.controls, restored_cage.controls) and bool(
        restored_optimizer.state_dict()["state"]
    )


def run(destination: Path, *, source_revision: str) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable E2 preflight: {destination}")
    torch.manual_seed(SEED)
    reference, faces = _ellipsoid()
    cage = TrilinearDeformationCage(reference, CAGE_RESOLUTION)
    incumbent_amplitude = _set_nearly_active_incumbent(cage, faces)
    previous = cage.controls.detach().clone()
    incumbent = cage.deformed_vertices().detach()
    bulge, candidate = _candidate_directions(cage)
    with torch.no_grad():
        cage.controls.copy_(previous + bulge)
    target = cage.deformed_vertices().detach()

    with torch.no_grad():
        cage.controls.copy_(previous)
    gradient_loss = (cage.deformed_vertices() - target).square().mean()
    gradient_loss.backward()  # type: ignore[no-untyped-call]
    gradient_valid = cage.controls.grad is not None and bool(
        torch.isfinite(cage.controls.grad).all() and torch.count_nonzero(cage.controls.grad)
    )
    cage.controls.grad = None

    control_optimizer = torch.optim.Adam(cage.parameters(), lr=0.002)
    with torch.no_grad():
        cage.controls.copy_(previous + candidate)
    control_scale = project_cage_step(
        cage,
        previous,
        faces,
        control_optimizer,
        minimum_signed_area_ratio=SIGNED_FLOOR,
        minimum_area_ratio=UNSIGNED_FLOOR,
        maximum_backtracks=MAXIMUM_BACKTRACKS,
    )
    control = cage.deformed_vertices().detach()

    with torch.no_grad():
        cage.controls.copy_(previous)
    treatment_delta, direction_report = linearized_feasible_cage_direction(
        cage,
        candidate,
        faces,
        minimum_signed_area_ratio=SIGNED_FLOOR,
        minimum_area_ratio=UNSIGNED_FLOOR,
        active_slack_tolerance=ACTIVE_SLACK_TOLERANCE,
    )
    with torch.no_grad():
        cage.controls.copy_(previous + treatment_delta)
    treatment_optimizer = torch.optim.Adam(cage.parameters(), lr=0.002)
    treatment_scale = project_cage_step(
        cage,
        previous,
        faces,
        treatment_optimizer,
        minimum_signed_area_ratio=SIGNED_FLOOR,
        minimum_area_ratio=UNSIGNED_FLOOR,
        maximum_backtracks=MAXIMUM_BACKTRACKS,
    )
    treatment = cage.deformed_vertices().detach()

    metrics = {
        "incumbent": _held_out_metrics(incumbent, target, faces),
        "global_backtracking_control": _held_out_metrics(control, target, faces),
        "linearized_qp_treatment": _held_out_metrics(treatment, target, faces),
    }
    treatment_normal = metrics["linearized_qp_treatment"][
        "pooled_held_out_median_normal_error_degrees"
    ]
    normal_improvement = min(
        metrics[name]["pooled_held_out_median_normal_error_degrees"] - treatment_normal
        for name in ("incumbent", "global_backtracking_control")
    )
    auxiliary = _auxiliary_fixtures()
    resume_cage = TrilinearDeformationCage(reference, CAGE_RESOLUTION)
    resume_verified = _resume_check(resume_cage, reference + 0.002)
    treatment_topology = _topology(reference, treatment, faces)
    blockers: list[str] = []
    if normal_improvement < 1.0:
        blockers.append("normal_improvement_below_one_degree")
    for metric, maximize in (
        ("mean_held_out_iou", True),
        ("mean_held_out_boundary_error", False),
        ("mean_absolute_signed_gap", False),
    ):
        treatment_value = metrics["linearized_qp_treatment"][metric]
        control_values = [
            metrics[name][metric] for name in ("incumbent", "global_backtracking_control")
        ]
        if (maximize and treatment_value < max(control_values)) or (
            not maximize and treatment_value > min(control_values)
        ):
            blockers.append(f"treatment_worse_{metric}")
    if treatment_scale != 1.0 or treatment_topology["status"] != "pass":
        blockers.append("nonlinear_topology_validation_failed")
    if not gradient_valid:
        blockers.append("missing_or_nonfinite_gradient")
    if not all(item["valid"] and item["finite"] for item in auxiliary.values()):
        blockers.append("sphere_or_rotated_thin_bridge_failed")
    if not resume_verified:
        blockers.append("resume_not_exact")

    report: dict[str, Any] = {
        "schema_version": "post_v1_e2_synthetic_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "source_revision": source_revision,
        "seed": SEED,
        "fixture": {
            "name": "reachable_same_8x8x4_cage_nearly_active_triangle_and_distant_smooth_bulge",
            "incumbent_control_amplitude": incumbent_amplitude,
            "held_out_view_angles_degrees": list(HELD_OUT_ANGLES),
        },
        "changed_mechanism": "candidate_direction_only_linearized_active_set_trust_region_qp",
        "frozen": {
            "cage_resolution": list(CAGE_RESOLUTION),
            "reference_connectivity": True,
            "original_reference": True,
            "nonlinear_backtracking_steps": MAXIMUM_BACKTRACKS,
            "signed_area_floor": SIGNED_FLOOR,
            "unsigned_area_floor": UNSIGNED_FLOOR,
        },
        "control": {"accepted_scale": control_scale, "mechanism": "global_backtracking"},
        "treatment": {
            "accepted_scale": treatment_scale,
            "direction_report": direction_report.__dict__,
            "exact_nonlinear_validation_retained": True,
        },
        "metrics": metrics,
        "minimum_normal_improvement_degrees": normal_improvement,
        "topology": treatment_topology,
        "auxiliary_fixtures": auxiliary,
        "gradient_valid": gradient_valid,
        "predicate_valid": treatment_topology["status"] == "pass",
        "resume_verified": resume_verified,
        "global_self_intersection_claim": False,
        "global_self_intersection_caveat": "local face predicates do not prove absence of all global self-intersections",
        "execution": {
            "human_evidence_accesses": 0,
            "development_evaluations": 0,
            "optimizer_steps_on_project_data": 0,
            "modal_jobs_launched": 0,
            "automatic_paid_retries": 0,
        },
        "sealed_test_isolation": {"private_evidence_paths_accessed": []},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e2-preflight-", dir=destination.parent))
    try:
        write_json(staging / "synthetic_preflight_report.json", report)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relevant = (
        "src/frayid/feasible_cage.py",
        "scripts/run_post_v1_e2_preflight.py",
        "configs/evaluation/post_v1_e2_feasible_cage_step_r01.yaml",
    )
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *relevant],
        cwd=project_root,
        check=False,
    ).returncode:
        raise RuntimeError("Refusing E2 preflight from a dirty relevant worktree")
    report = run(OUTPUT_RELATIVE / "synthetic_preflight", source_revision=revision)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
