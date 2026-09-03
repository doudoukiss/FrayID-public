from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from frayid.flexicubes_adapter import PinnedFlexiCubes
from frayid.io import read_json, sha256_file, write_json
from frayid.normal_integrable_sdf import trilinear_grid_sample
from frayid.v2.checkpoint import capture_checkpoint, restore_checkpoint
from frayid.v2.contracts import QualificationState, advance_qualification, reject_sealed_capability
from frayid.v2.evidence import EvidenceVolume
from frayid.v2.field import V2NeuralSDF
from frayid.v2.layers import (
    ClippedImplicitLayer,
    LayeredCanonicalModel,
    body_garment_nonpenetration_loss,
    bounded_residual_deformation,
)
from frayid.v2.normals import normal_transport_ablation


def qualify_outer_field(
    evidence: EvidenceVolume,
    *,
    evidence_volume_path: Path,
    evidence_binding_path: Path,
    hull_qualification_path: Path,
    device: torch.device | str = "cpu",
    extraction_device: torch.device | str | None = None,
    flexicubes_repository: Path | None = None,
    seed: int = 20260902,
) -> dict[str, object]:
    """Exercise G01 once against the qualified real T04+S01 evidence.

    This is deliberately an engineering qualification, not a scientific
    attempt: it proves that the actual hull, uncertainty, semantics, cameras,
    and silhouettes reach differentiable field code and checkpoint replay.
    """

    reject_sealed_capability([evidence_volume_path, evidence_binding_path, hull_qualification_path])
    hull_report = read_json(hull_qualification_path)
    if hull_report.get("status") != "pass":
        raise ValueError("G01 qualification requires a passing hull qualification")
    report_hashes = hull_report.get("source_hashes", {})
    if report_hashes.get("binding") != sha256_file(evidence_binding_path):
        raise ValueError("G01 evidence binding does not match the qualified hull report")
    if report_hashes.get("reference_volume") != sha256_file(evidence_volume_path):
        raise ValueError("G01 evidence volume does not match the qualified hull report")

    torch.manual_seed(seed)
    target_device = torch.device(device)
    target_extraction_device = torch.device(extraction_device or target_device)
    model = V2NeuralSDF(evidence).to(target_device)
    extent = evidence.metadata.extent
    voxel_pitch = 2.0 * extent / (evidence.metadata.resolution - 1)
    registered_sdf_offset = 0.25 * voxel_pitch
    output_layer = model.field.residual[-1]
    if not isinstance(output_layer, nn.Linear):
        raise RuntimeError("G01 residual output layer is not the registered linear layer")
    with torch.no_grad():
        output_layer.bias.add_(registered_sdf_offset)
    points = (torch.rand(2048, 3, device=target_device) * 2.0 - 1.0) * extent
    points.requires_grad_(True)
    values = model.values(points)

    target_sdf = trilinear_grid_sample(evidence.signed_distance, points, extent=extent)
    support = trilinear_grid_sample(evidence.support_count.to(torch.float32), points, extent=extent)
    mask_uncertainty = trilinear_grid_sample(
        evidence.mask_uncertainty, points, extent=extent
    ).clamp(0.0, 1.0)
    motion_uncertainty = trilinear_grid_sample(
        evidence.motion_uncertainty, points, extent=extent
    ).clamp(0.0, 1.0)
    unsupported = (
        trilinear_grid_sample(evidence.unsupported.to(torch.float32), points, extent=extent) >= 0.5
    )
    evidence_weights = (
        (support / float(evidence.metadata.training_view_count)).clamp(0.0, 1.0)
        * (1.0 - mask_uncertainty)
        * (1.0 - motion_uncertainty)
        * (~unsupported).to(values.dtype)
    )
    if not torch.any(evidence_weights > 0):
        raise ValueError("G01 evidence contains no supported optimization samples")

    with np.load(evidence_binding_path, allow_pickle=False) as archive:
        silhouettes = torch.as_tensor(archive["silhouettes"], device=target_device)
        intrinsics = torch.as_tensor(archive["intrinsics"], device=target_device)
        rotations = torch.as_tensor(archive["rotations"], device=target_device)
        translations = torch.as_tensor(archive["translations"], device=target_device)
        motion_per_view = torch.as_tensor(archive["motion_uncertainty"], device=target_device)
        binding_source_hashes = json.loads(str(archive["source_hashes"]))
        binding_semantics = sorted(
            name.removeprefix("semantic__")
            for name in archive.files
            if name.startswith("semantic__")
        )
    if binding_source_hashes != evidence.metadata.source_hashes:
        raise ValueError("G01 hull and binding source provenance differ")
    if silhouettes.shape[0] != evidence.metadata.training_view_count:
        raise ValueError("G01 binding view count does not match the evidence volume")
    if binding_semantics != sorted(evidence.semantic_support):
        raise ValueError("G01 semantic channels do not match the evidence volume")

    view = int(torch.argmin(motion_per_view))
    height, width = silhouettes.shape[-2:]
    sample_y = torch.linspace(0, height - 1, 10, device=target_device).round().long()
    sample_x = torch.linspace(0, width - 1, 8, device=target_device).round().long()
    pixel_y, pixel_x = torch.meshgrid(sample_y, sample_x, indexing="ij")
    matrix = intrinsics if intrinsics.ndim == 2 else intrinsics[view]
    camera_directions = torch.stack(
        (
            (pixel_x.to(matrix.dtype) - matrix[0, 2]) / matrix[0, 0],
            (pixel_y.to(matrix.dtype) - matrix[1, 2]) / matrix[1, 1],
            torch.ones_like(pixel_x, dtype=matrix.dtype),
        ),
        dim=-1,
    )
    ray_directions = F.normalize(camera_directions @ rotations[view], dim=-1)
    camera_center = (-translations[view]) @ rotations[view]
    ray_origins = camera_center.expand_as(ray_directions)
    ray_targets = silhouettes[view][pixel_y, pixel_x].to(values.dtype)
    rendered = model.render_rays(
        ray_origins,
        ray_directions,
        near=0.01 * extent,
        far=4.0 * extent,
        sample_count=16,
        hierarchical_sample_count=8,
        deformation_jacobian=torch.eye(3, device=target_device),
        create_graph=True,
    )
    losses = {
        "evidence_sdf": (
            F.smooth_l1_loss(values, target_sdf, reduction="none") * evidence_weights
        ).sum()
        / evidence_weights.sum().clamp_min(1.0e-8),
        "eikonal_probe": (
            torch.linalg.vector_norm(model.gradients(points, create_graph=True), dim=-1) - 1
        )
        .square()
        .mean(),
        "real_silhouette_renderer": F.binary_cross_entropy(
            rendered.silhouette.clamp(1.0e-5, 1.0 - 1.0e-5),
            ray_targets,
        ),
    }
    gradients = model.evidence_gradients(losses)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])
    optimizer.zero_grad(set_to_none=True)
    total = sum(losses.values())
    total.backward()
    optimizer.step()
    after = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])
    checkpoint_state = model.checkpoint_state()
    checkpoint = capture_checkpoint(
        model,
        optimizer,
        step=1,
        topology_connectivity_sha256=None,
    )
    replay_points = torch.linspace(
        -0.4 * extent,
        0.4 * extent,
        24,
        device=target_device,
    ).reshape(8, 3)

    def replay_transition(
        replay_model: V2NeuralSDF, replay_optimizer: torch.optim.Optimizer
    ) -> Tensor:
        replay_optimizer.zero_grad(set_to_none=True)
        replay_loss = replay_model.values(replay_points).square().mean()
        replay_loss.backward()  # type: ignore[no-untyped-call]
        replay_optimizer.step()
        return torch.cat(
            [parameter.detach().reshape(-1) for parameter in replay_model.parameters()]
        )

    expected_replay = replay_transition(model, optimizer)
    restored_model = V2NeuralSDF(evidence).to(target_device)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=9.0)
    restored = restore_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        device=target_device,
    )
    observed_replay = replay_transition(restored_model, restored_optimizer)
    replay_exact = torch.equal(observed_replay, expected_replay)

    blockers = [name for name, value in gradients.items() if value <= 0]
    if torch.equal(before, after):
        blockers.append("field_parameters_unchanged")
    if not checkpoint_state:
        blockers.append("canonical_field_checkpoint_state_empty")
    if not replay_exact:
        blockers.append("same_device_checkpoint_replay")
    if not bool(torch.isfinite(rendered.silhouette).all()):
        blockers.append("real_silhouette_renderer_nonfinite")
    if not bool(torch.any(ray_targets > 0.5) and torch.any(ray_targets < 0.5)):
        blockers.append("real_ray_probe_missing_foreground_or_background")
    if evidence.metadata.cleanup_operations:
        blockers.append("evidence_volume_contains_cleanup")
    if not binding_semantics:
        blockers.append("semantic_evidence_not_bound")
    extraction: dict[str, object]
    if flexicubes_repository is None:
        extraction = {"status": "not_run", "reason": "repository_not_bound"}
        blockers.append("flexicubes_runtime_not_run")
    else:
        try:
            extraction_evidence = EvidenceVolume.load(
                evidence_volume_path, device=target_extraction_device
            )
            extraction_model = V2NeuralSDF(extraction_evidence).to(target_extraction_device)
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
            blockers.append("flexicubes_runtime_failed")
    return {
        "schema_version": "frayid_v2_outer_field_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_only": True,
        "qualification_scope": "local_engineering",
        "promotion_eligible": not blockers and target_device.type == "cuda",
        "remaining_promotion_blockers": (
            [] if target_device.type == "cuda" else ["target_gpu_forward_backward_not_run"]
        ),
        "scientific_attempt_marker_created": False,
        "optimizer_steps": 1,
        "device": str(target_device),
        "dtype": str(values.dtype),
        "target_gpu_exercised": target_device.type == "cuda",
        "apple_gpu_exercised": target_device.type == "mps",
        "extraction_device": str(target_extraction_device),
        "real_training_view_count": int(silhouettes.shape[0]),
        "real_ray_view_slot": view,
        "real_ray_count": int(ray_targets.numel()),
        "real_ray_foreground_count": int((ray_targets > 0.5).sum()),
        "real_ray_background_count": int((ray_targets <= 0.5).sum()),
        "supported_probe_count": int((evidence_weights > 0).sum()),
        "unsupported_probe_count": int(unsupported.sum()),
        "registered_sdf_recovery_offset": registered_sdf_offset,
        "semantic_layer_names": binding_semantics,
        "source_hashes": {
            "evidence_volume": sha256_file(evidence_volume_path),
            "evidence_binding": sha256_file(evidence_binding_path),
            "hull_qualification": sha256_file(hull_qualification_path),
            **binding_source_hashes,
        },
        "gradient_norms": gradients,
        "parameters_changed": not torch.equal(before, after),
        "renderer_finite": bool(torch.isfinite(rendered.silhouette).all()),
        "checkpoint_schema": restored["schema_version"],
        "checkpoint_tensor_count": len(checkpoint_state),
        "same_device_replay_exact": replay_exact,
        "extraction": extraction,
        "blockers": blockers,
        "sealed_test_accesses": 0,
    }


def audit_g01_local_qualification_lifecycle(
    report_path: Path,
    output_path: Path,
    *,
    evaluator_report_path: Path | None = None,
) -> Path:
    """Record G01's achieved local state without claiming target-GPU qualification."""

    paths = [report_path, output_path]
    if evaluator_report_path is not None:
        paths.append(evaluator_report_path)
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G01 lifecycle records are immutable")
    report = read_json(report_path)
    gradient_norms = report.get("gradient_norms", {})
    checks = {
        "module_imported": True,
        "real_t04_s01_data_bound": report.get("real_training_view_count") == 144
        and len(report.get("semantic_layer_names", [])) == 6
        and len(report.get("source_hashes", {})) >= 7,
        "local_device_validated": report.get("device") in {"cpu", "mps"}
        and report.get("dtype") == "torch.float32"
        and report.get("renderer_finite") is True,
        "one_step_real_evidence_passed": report.get("status") == "pass"
        and report.get("optimizer_steps") == 1
        and report.get("parameters_changed") is True
        and all(float(value) > 0 for value in gradient_norms.values()),
        "same_device_checkpoint_restored": report.get("same_device_replay_exact") is True
        and report.get("checkpoint_schema") == "frayid_v2_checkpoint.v1",
        "sealed_boundary_passed": report.get("sealed_test_accesses") == 0
        and report.get("scientific_attempt_marker_created") is False,
    }
    evaluator_report = (
        read_json(evaluator_report_path) if evaluator_report_path is not None else None
    )
    if evaluator_report is not None:
        checks["independent_evaluator_dry_run_passed"] = (
            evaluator_report.get("status") == "pass"
            and evaluator_report.get("dry_run_only") is True
            and evaluator_report.get("candidate_scored") is False
            and evaluator_report.get("legacy_development_images_read") == 0
            and evaluator_report.get("sealed_test_accesses") == 0
        )
    blockers = [name for name, passed in checks.items() if not passed]
    state = QualificationState.BUILT
    transitions: list[dict[str, str]] = []
    transition_evidence = {
        QualificationState.IMPORTED: "module_imported",
        QualificationState.DATA_BOUND: "real_t04_s01_data_bound",
        QualificationState.DEVICE_VALIDATED: "local_device_validated",
        QualificationState.ONE_STEP_PASSED: "one_step_real_evidence_passed",
        QualificationState.CHECKPOINT_RESTORED: "same_device_checkpoint_restored",
    }
    if evaluator_report is not None:
        transition_evidence[QualificationState.EVALUATOR_DRY] = (
            "independent_evaluator_dry_run_passed"
        )
    if not blockers:
        for requested, evidence_name in transition_evidence.items():
            previous = state
            state = advance_qualification(state, requested)
            transitions.append(
                {"from": previous.value, "to": state.value, "evidence": evidence_name}
            )
    payload = {
        "schema_version": "frayid_v2_g01_local_qualification_lifecycle.v1",
        "experiment_id": "postv2_g01_direct_multires_field_outer_r01",
        "status": (
            "pass"
            if state
            is (
                QualificationState.EVALUATOR_DRY
                if evaluator_report is not None
                else QualificationState.CHECKPOINT_RESTORED
            )
            else "fail"
        ),
        "state": state.value,
        "checks": checks,
        "transitions": transitions,
        "qualification_report_sha256": sha256_file(report_path),
        "evaluator_report_sha256": (
            sha256_file(evaluator_report_path) if evaluator_report_path is not None else None
        ),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "blockers": blockers,
        "remaining_qualification_work": (
            ["one_step_target_gpu_forward_backward_restore_replay"]
            if evaluator_report is not None
            else [
                "one_step_target_gpu_forward_backward_restore_replay",
                "independent_public_and_real_evaluator_dry_run",
            ]
        ),
        "development_reads": 0,
        "sealed_test_reads": 0,
        "attempt_marker_created": False,
        "scientific_attempt_authorized": False,
        "optimizer_steps": 1,
    }
    return write_json(output_path, payload)


def audit_g01_target_cuda_qualification(
    report_path: Path,
    claim_path: Path,
    plan_path: Path,
    evaluator_lifecycle_path: Path,
    local_qualification_path: Path,
    output_path: Path,
) -> Path:
    """Promote G01 exactly once from evaluator-dry to CUDA-qualified evidence."""

    paths = [
        report_path,
        claim_path,
        plan_path,
        evaluator_lifecycle_path,
        local_qualification_path,
        output_path,
    ]
    reject_sealed_capability(paths)
    if output_path.exists():
        raise FileExistsError("G01 target-CUDA lifecycle records are immutable")
    report = read_json(report_path)
    claim = read_json(claim_path)
    plan = read_json(plan_path)
    lifecycle = read_json(evaluator_lifecycle_path)
    plan_hashes = plan.get("input_hashes", {})
    report_hashes = report.get("source_hashes", {})
    gradient_norms = report.get("gradient_norms", {})
    extraction = report.get("extraction", {})
    checks = {
        "evaluator_dry_predecessor_passed": lifecycle.get("status") == "pass"
        and lifecycle.get("state") == QualificationState.EVALUATOR_DRY.value
        and sha256_file(evaluator_lifecycle_path) == plan_hashes.get("lifecycle"),
        "local_qualification_bound": sha256_file(local_qualification_path)
        == plan_hashes.get("local_qualification"),
        "ready_zero_retry_qualification_plan": plan.get("status") == "ready"
        and plan.get("qualification_only") is True
        and plan.get("scientific_attempt") is False
        and plan.get("automatic_retries") == 0
        and plan.get("dispatch_authorized") is True
        and plan.get("blockers") == [],
        "exclusive_claim_matches_plan": claim.get("qualification_run_id")
        == plan.get("qualification_run_id")
        and claim.get("source_revision") == plan.get("source_commit")
        and claim.get("gpu") == plan.get("gpu")
        and claim.get("timeout_seconds") == plan.get("timeout_seconds")
        and claim.get("provider_rate_usd_per_hour") == plan.get("provider_rate_usd_per_hour")
        and claim.get("price_checked_at") == plan.get("price_checked_at")
        and claim.get("maximum_cost_usd") == plan.get("maximum_cost_usd")
        and claim.get("automatic_retries") == 0
        and claim.get("scientific_attempt") is False,
        "qualified_inputs_match_report": report_hashes.get("evidence_volume")
        == plan_hashes.get("evidence_volume")
        and report_hashes.get("evidence_binding") == plan_hashes.get("evidence_binding")
        and report_hashes.get("hull_qualification") == plan_hashes.get("hull_qualification"),
        "target_cuda_forward_backward_passed": report.get("status") == "pass"
        and report.get("device") == "cuda"
        and report.get("dtype") == "torch.float32"
        and report.get("target_gpu_exercised") is True
        and report.get("optimizer_steps") == 1
        and report.get("parameters_changed") is True
        and report.get("renderer_finite") is True
        and bool(gradient_norms)
        and all(float(value) > 0 for value in gradient_norms.values()),
        "target_cuda_checkpoint_replay_passed": report.get("same_device_replay_exact") is True
        and report.get("checkpoint_schema") == "frayid_v2_checkpoint.v1",
        "cpu_search_extraction_passed": extraction.get("status") == "pass"
        and extraction.get("device") == "cpu"
        and extraction.get("search_only") is True
        and int(extraction.get("vertex_count", 0)) > 0
        and int(extraction.get("face_count", 0)) > 0,
        "qualification_promotable_without_science": report.get("promotion_eligible") is True
        and report.get("blockers") == []
        and report.get("remaining_promotion_blockers") == []
        and report.get("qualification_only") is True
        and report.get("scientific_attempt_marker_created") is False
        and report.get("sealed_test_accesses") == 0,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    state = QualificationState.EVALUATOR_DRY
    transitions: list[dict[str, str]] = []
    if not blockers:
        previous = state
        state = advance_qualification(state, QualificationState.QUALIFIED)
        transitions.append(
            {
                "from": previous.value,
                "to": state.value,
                "evidence": "target_cuda_forward_backward_restore_replay_and_cpu_search",
            }
        )
    payload = {
        "schema_version": "frayid_v2_g01_target_cuda_qualification_lifecycle.v1",
        "experiment_id": "postv2_g01_direct_multires_field_outer_r01",
        "qualification_run_id": report.get("qualification_run_id"),
        "status": "pass" if state is QualificationState.QUALIFIED else "fail",
        "state": state.value,
        "checks": checks,
        "transitions": transitions,
        "blockers": blockers,
        "target_cuda_report_sha256": sha256_file(report_path),
        "qualification_claim_sha256": sha256_file(claim_path),
        "qualification_plan_sha256": sha256_file(plan_path),
        "evaluator_lifecycle_sha256": sha256_file(evaluator_lifecycle_path),
        "local_qualification_sha256": sha256_file(local_qualification_path),
        "development_reads": 0,
        "sealed_test_reads": 0,
        "attempt_marker_created": False,
        "scientific_attempt_authorized": False,
        "remaining_qualification_work": [],
    }
    return write_json(output_path, payload)


class _SphereField(nn.Module):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = nn.Parameter(torch.tensor(radius))

    def forward(self, points: Tensor) -> Tensor:
        result: Tensor = torch.linalg.vector_norm(points, dim=-1) - self.radius
        return result


class _ClipField(nn.Module):
    def forward(self, points: Tensor) -> Tensor:
        return 0.7 - points[..., 1].abs()


def qualify_layered_model(*, device: torch.device | str = "cpu") -> dict[str, object]:
    body = _SphereField(0.55).to(device)
    support = _SphereField(0.68).to(device)
    garment = ClippedImplicitLayer(support, _ClipField().to(device), layer_id="upper_clothing")
    model = LayeredCanonicalModel(body, {"upper_clothing": garment})
    theta = torch.linspace(0, 2 * torch.pi, 128, device=device)
    y = torch.linspace(-0.65, 0.65, 128, device=device)
    points = torch.stack((0.68 * torch.cos(theta), y, 0.68 * torch.sin(theta)), dim=-1)
    body_values = model.body_values(points)
    loss = body_garment_nonpenetration_loss(body_values, contact_band=0.01)
    loss = (
        loss
        + garment.active_weight(points).mean() * 0.001
        + garment.support_values(points).square().mean()
    )
    loss.backward()  # type: ignore[no-untyped-call]
    displacement = bounded_residual_deformation(torch.full_like(points, 100.0), maximum_norm=0.05)
    blockers: list[str] = []
    if support.radius.grad is None or not torch.isfinite(support.radius.grad):
        blockers.append("garment_gradient")
    if float(torch.linalg.vector_norm(displacement, dim=-1).max()) > 0.050001:
        blockers.append("deformation_bound")
    ownership = model.visibility_ownership(
        {
            "body": torch.full((8,), 1.0, device=device),
            "upper_clothing": torch.full((8,), 0.8, device=device),
        }
    )
    if not torch.all(ownership == 1):
        blockers.append("visibility_ownership")
    no_hit = model.visibility_ownership(
        {
            "body": torch.full((2,), -1.0, device=device),
            "upper_clothing": torch.full((2,), -0.5, device=device),
        }
    )
    if not torch.all(no_hit == -1):
        blockers.append("visibility_no_hit_sentinel")
    return {
        "schema_version": "frayid_v2_layered_qualification.v1",
        "status": "pass" if not blockers else "fail",
        "qualification_only": True,
        "scientific_attempt_marker_created": False,
        "maximum_residual_norm": float(torch.linalg.vector_norm(displacement, dim=-1).max()),
        "blockers": blockers,
        "sealed_test_accesses": 0,
    }


def qualify_normal_transport(*, device: torch.device | str = "cpu") -> dict[str, object]:
    normals = torch.tensor([[0.0, 1.0, 1.0]], device=device)
    jacobian = torch.diag(torch.tensor([2.0, 0.5, 1.5], device=device))[None]
    correct = normal_transport_ablation(normals, jacobian, mode="inverse_transpose")
    wrong = normal_transport_ablation(normals, jacobian, mode="rotation_only")
    difference = torch.rad2deg(torch.acos((correct * wrong).sum(dim=-1).clamp(-1.0, 1.0)))
    return {
        "schema_version": "frayid_v2_normal_qualification.v1",
        "status": "pass" if float(difference.max()) > 1.0 else "fail",
        "inverse_transpose_vs_rotation_difference_degrees": float(difference.max()),
        "sealed_test_accesses": 0,
    }
