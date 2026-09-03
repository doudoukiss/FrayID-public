"""Run the public-only E8 opaque visibility and known-shape CUDA gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace")
EXPERIMENT_ID = "postv1_e08_opaque_visibility_training_r01"
NVDIFFRAST_REVISION = "253ac4fcea7de5f396371124af597e6cc957bfae"
SEED = 20260831

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11")
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    .apt_install("build-essential", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("ninja", "numpy>=1.26,<3", "setuptools", "wheel")
    .run_commands(
        f"pip install git+https://github.com/NVlabs/nvdiffrast.git@{NVDIFFRAST_REVISION} "
        "--no-build-isolation"
    )
    .pip_install("numpy>=1.26,<3", "scikit-image>=0.24", "scipy>=1.12", "trimesh>=4.4")
    .add_local_dir(PROJECT_ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
)
app = modal.App("frayid-postv1-e8-opaque-public-gate-r01", image=image)


@app.function(
    gpu="L40S",
    cpu=4.0,
    memory=16384,
    timeout=600,
    retries=0,
    env={"PYTHONPATH": str(REMOTE_ROOT / "src")},
)
def run_gate(source_revision: str) -> dict[str, Any]:
    import math

    import numpy as np
    import torch
    import trimesh

    from frayid.camera import make_intrinsics
    from frayid.renderer import normal_cosine_loss, render_soft_mesh
    from frayid.renderer_contract import (
        RendererBackend,
        create_training_renderer,
        renderer_contract,
        require_legacy_evaluator_backend,
    )
    from frayid.replay_state import configure_deterministic_execution
    from frayid.triangle_rasterizer import NvdiffrastRenderer, rasterize_reference

    configure_deterministic_execution()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    renderer = NvdiffrastRenderer()

    def box(depth: float) -> tuple[Any, Any]:
        vertices = torch.tensor(
            [
                [-0.6, -0.6, 2.0],
                [0.6, -0.6, 2.0],
                [0.6, 0.6, 2.0],
                [-0.6, 0.6, 2.0],
                [-0.6, -0.6, 2.0 + depth],
                [0.6, -0.6, 2.0 + depth],
                [0.6, 0.6, 2.0 + depth],
                [-0.6, 0.6, 2.0 + depth],
            ],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [3, 7, 6],
                [3, 6, 2],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ],
            dtype=torch.long,
            device=device,
        )
        return vertices, faces

    intrinsics = make_intrinsics(70.0, (32.0, 32.0), device=device)
    shallow_vertices, box_faces = box(0.2)
    deep_vertices, _ = box(2.0)
    shallow_mask, _ = renderer(
        shallow_vertices,
        box_faces,
        intrinsics,
        (64, 64),
        source_image_size=(64, 64),
    )
    deep_mask, _ = renderer(
        deep_vertices,
        box_faces,
        intrinsics,
        (64, 64),
        source_image_size=(64, 64),
    )
    hidden_surface_max_delta = float((shallow_mask - deep_mask).abs().max())

    quad = torch.tensor(
        [[-0.7, -0.5, 2.0], [0.7, -0.5, 2.0], [0.7, 0.5, 2.0], [-0.7, 0.5, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    quad_faces = torch.tensor([[0, 2, 1], [0, 3, 2]], dtype=torch.long, device=device)
    center = quad.mean(dim=0, keepdim=True)
    subdivided = torch.cat((quad, center), dim=0)
    subdivided_faces = torch.tensor(
        [[0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]], dtype=torch.long, device=device
    )
    quad_mask, _ = renderer(quad, quad_faces, intrinsics, (64, 64), source_image_size=(64, 64))
    subdivided_mask, _ = renderer(
        subdivided,
        subdivided_faces.flip(0),
        intrinsics,
        (64, 64),
        source_image_size=(64, 64),
    )
    subdivision_max_delta = float((quad_mask - subdivided_mask).abs().max())

    offcentre_vertices = torch.tensor(
        [[-0.5, -0.4, 2.0], [0.7, -0.4, 2.2], [0.1, 0.6, 2.1]], dtype=torch.float32
    )
    offcentre_faces = torch.tensor([[0, 2, 1]], dtype=torch.long)
    offcentre_intrinsics_cpu = make_intrinsics(150.0, (91.0, 43.0))
    reference_mask, reference_normals, _ = rasterize_reference(
        offcentre_vertices,
        offcentre_faces,
        offcentre_intrinsics_cpu,
        (48, 80),
        source_image_size=(120, 200),
    )
    cuda_mask, cuda_normals = renderer.render_point_sampled(
        offcentre_vertices.cuda(),
        offcentre_faces.cuda(),
        offcentre_intrinsics_cpu.cuda(),
        (48, 80),
        source_image_size=(120, 200),
    )
    point_mask_equal = bool(torch.equal(cuda_mask.cpu(), reference_mask))
    common = reference_mask > 0.5
    normal_axis_error = float((cuda_normals.cpu()[common] - reference_normals[common]).abs().max())

    gap_vertices = torch.tensor(
        [
            [-0.7, -0.5, 2.0],
            [-0.08, -0.5, 2.0],
            [-0.08, 0.5, 2.0],
            [-0.7, 0.5, 2.0],
            [0.08, -0.5, 2.0],
            [0.7, -0.5, 2.0],
            [0.7, 0.5, 2.0],
            [0.08, 0.5, 2.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    gap_faces = torch.tensor(
        [[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]], dtype=torch.long, device=device
    )
    gap_mask, _ = renderer(
        gap_vertices, gap_faces, intrinsics, (64, 64), source_image_size=(64, 64)
    )
    thin_gap_max_coverage = float(gap_mask[:, 31:33].max())

    scale = torch.tensor(0.92, device=device, requires_grad=True)
    target_quad = quad.detach().clone()
    target_quad[:, 0] *= 1.08
    target_mask, _ = renderer(
        target_quad, quad_faces, intrinsics, (64, 64), source_image_size=(64, 64)
    )

    def scale_loss(value: Any) -> Any:
        candidate = quad * torch.stack((value, value.new_tensor(1.0), value.new_tensor(1.0)))
        mask, _ = renderer(candidate, quad_faces, intrinsics, (64, 64), source_image_size=(64, 64))
        return (mask - target_mask).square().mean()

    differentiable_loss = scale_loss(scale)
    differentiable_loss.backward()
    assert scale.grad is not None
    analytic = float(scale.grad)
    epsilon = 1e-3
    with torch.no_grad():
        finite_difference = float(
            (scale_loss(scale + epsilon) - scale_loss(scale - epsilon)) / (2.0 * epsilon)
        )
    finite_difference_same_direction = analytic * finite_difference > 0
    finite_difference_relative_error = abs(analytic - finite_difference) / max(
        abs(finite_difference), 1e-8
    )

    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    base = torch.tensor(np.asarray(source.vertices), dtype=torch.float32, device=device)
    shape_faces = torch.tensor(np.asarray(source.faces), dtype=torch.long, device=device)
    target_scale = torch.tensor([0.52, 0.83, 0.41], dtype=torch.float32, device=device)
    initial_scale = torch.tensor([0.72, 0.63, 0.61], dtype=torch.float32, device=device)
    angles = (0.0, 70.0, 140.0)

    def pose(vertices: Any, degrees: float) -> Any:
        angle = vertices.new_tensor(math.radians(degrees))
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack(
            (
                torch.stack((cosine, angle.new_tensor(0.0), sine)),
                torch.stack((angle.new_tensor(0.0), angle.new_tensor(1.0), angle.new_tensor(0.0))),
                torch.stack((-sine, angle.new_tensor(0.0), cosine)),
            )
        )
        return vertices @ rotation.T + vertices.new_tensor([0.0, 0.0, 3.0])

    oracle_intrinsics = make_intrinsics(62.0, (24.0, 24.0), device=device)
    target_vertices = base * target_scale
    targets = [
        renderer(
            pose(target_vertices, angle),
            shape_faces,
            oracle_intrinsics,
            (48, 48),
            source_image_size=(48, 48),
        )
        for angle in angles
    ]

    def optimize(backend: str) -> tuple[Any, list[float]]:
        parameter = initial_scale.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([parameter], lr=0.04)
        history: list[float] = []
        selected = renderer if backend == "opaque" else render_soft_mesh
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            values = []
            for view, angle in enumerate(angles):
                if backend == "soft":
                    torch.manual_seed(SEED + view)
                    torch.cuda.manual_seed_all(SEED + view)
                predicted_mask, predicted_normals = selected(
                    pose(base * parameter, angle),
                    shape_faces,
                    oracle_intrinsics,
                    (48, 48),
                    source_image_size=(48, 48),
                    sample_count=1024,
                    reference_sample_count=1024,
                    sigma_pixels=1.4,
                )
                target_view_mask, target_view_normals = targets[view]
                values.append(
                    (predicted_mask - target_view_mask).square().mean()
                    + 0.2
                    * normal_cosine_loss(predicted_normals, target_view_normals, target_view_mask)
                )
            objective = torch.stack(values).mean()
            objective.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            with torch.no_grad():
                parameter.clamp_(0.2, 1.2)
            history.append(float(objective.detach()))
        return parameter.detach(), history

    soft_scale, soft_history = optimize("soft")
    opaque_scale, opaque_history = optimize("opaque")

    def canonical_chamfer(scale_value: Any) -> float:
        candidate = base * scale_value
        distances = torch.cdist(candidate, target_vertices)
        return float(0.5 * (distances.min(0).values.mean() + distances.min(1).values.mean()))

    soft_chamfer = canonical_chamfer(soft_scale)
    opaque_chamfer = canonical_chamfer(opaque_scale)
    chamfer_improvement = (soft_chamfer - opaque_chamfer) / max(soft_chamfer, 1e-8)

    legacy_contract = renderer_contract(RendererBackend.LEGACY_SOFT_SPLAT)
    opaque_contract = renderer_contract(RendererBackend.OPAQUE_NVDIFFRAST)
    require_legacy_evaluator_backend(RendererBackend.LEGACY_SOFT_SPLAT)
    backend_instances_valid = (
        create_training_renderer(RendererBackend.LEGACY_SOFT_SPLAT) is render_soft_mesh
        and opaque_contract.report_namespace != legacy_contract.report_namespace
    )
    blockers = []
    if hidden_surface_max_delta > 1e-6:
        blockers.append("hidden_surface_invariance")
    if subdivision_max_delta > 1e-5:
        blockers.append("subdivision_or_triangle_order_invariance")
    if not point_mask_equal or normal_axis_error > 1e-5:
        blockers.append("offcentre_crop_or_normal_axis_contract")
    if thin_gap_max_coverage > 1e-4:
        blockers.append("thin_gap_closed")
    if not finite_difference_same_direction or finite_difference_relative_error > 0.15:
        blockers.append("finite_difference_gradient")
    if chamfer_improvement < 0.10:
        blockers.append("known_shape_chamfer_improvement_below_10_percent")
    if not backend_instances_valid:
        blockers.append("renderer_backend_contract")
    return {
        "schema_version": "post_v1_e8_opaque_public_gate.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_revision": source_revision,
        "status": "pass" if not blockers else "fail",
        "seed": SEED,
        "device_name": torch.cuda.get_device_name(0),
        "invariants": {
            "hidden_surface_max_delta": hidden_surface_max_delta,
            "subdivision_max_delta": subdivision_max_delta,
            "offcentre_crop_point_mask_equal": point_mask_equal,
            "normal_axis_max_error": normal_axis_error,
            "thin_gap_max_coverage": thin_gap_max_coverage,
            "finite_difference_same_direction": finite_difference_same_direction,
            "finite_difference_relative_error": finite_difference_relative_error,
        },
        "known_shape": {
            "soft_control_scale": soft_scale.cpu().tolist(),
            "opaque_treatment_scale": opaque_scale.cpu().tolist(),
            "target_scale": target_scale.cpu().tolist(),
            "soft_control_chamfer": soft_chamfer,
            "opaque_treatment_chamfer": opaque_chamfer,
            "chamfer_improvement_fraction": chamfer_improvement,
            "soft_initial_objective": soft_history[0],
            "soft_final_objective": soft_history[-1],
            "opaque_initial_objective": opaque_history[0],
            "opaque_final_objective": opaque_history[-1],
        },
        "renderer_contract": {
            "legacy_namespace": legacy_contract.report_namespace,
            "opaque_namespace": opaque_contract.report_namespace,
            "legacy_evaluator_unchanged": True,
        },
        "blockers": blockers,
        "execution": {
            "private_inputs_loaded": 0,
            "development_evaluations": 0,
            "sealed_test_accesses": 0,
            "automatic_paid_retries": 0,
        },
    }


@app.local_entrypoint()
def main(output: str) -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("E8 public gate requires a clean source revision")
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"immutable output already exists: {output_path}")
    report = run_gate.remote(revision)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)
