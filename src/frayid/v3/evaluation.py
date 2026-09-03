from __future__ import annotations

from typing import Any

from frayid.v3.schemas import (
    DerivedSurfaceExport,
    MantleArtifact,
    TopologyCertificateV3,
    UpperGarmentAtlas,
)

EXPERIMENT_ID = "postv3_h03_material_atlas_joint_inverse_capture_r01"
REQUIRED_STAGES = (
    "postv3_q04_local_material_chart_graph_r01",
    "postv3_t06_image_driven_fixed_camera_factor_graph_r01",
    "postv3_l04_physical_boundary_ontology_r01",
    "postv3_l05_intrinsic_upper_garment_material_atlas_r01",
    "postv3_g05_envelope_differential_surface_r01",
)


def evaluate_mantle(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply H03's fail-closed aggregate gates without opening sealed evidence."""
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("stages must be an object keyed by experiment ID")
    blockers: list[str] = []
    for experiment_id in REQUIRED_STAGES:
        stage = stages.get(experiment_id)
        if not isinstance(stage, dict) or stage.get("status") != "pass":
            blockers.append(f"required_stage_not_passing:{experiment_id}")
        elif not bool(stage.get("promotion_eligible", False)):
            blockers.append(f"required_stage_not_real_promotion_eligible:{experiment_id}")
    if bool(payload.get("photometry_activated", False)):
        photometry = stages.get("postv3_p01_rotation_photometric_varpro_r01")
        if not isinstance(photometry, dict) or photometry.get("status") != "pass":
            blockers.append("activated_photometry_not_passing")

    global_gates = payload.get("global_gates", {})
    if not isinstance(global_gates, dict):
        raise ValueError("global_gates must be an object")
    required_global = (
        "historical_iou_unchanged",
        "historical_boundary_unchanged",
        "historical_normal_unchanged",
        "train_held_gap_unchanged",
        "upper_boundary_improvement_at_least_20_percent",
        "chart_median_reprojection_at_most_2_5_pixels",
        "restart_gate",
        "uncertainty_gate",
        "provenance_gate",
        "exact_replay_gate",
        "privacy_gate",
        "capacity_ablation_gate",
    )
    for gate in required_global:
        if not bool(global_gates.get(gate, False)):
            blockers.append(f"global_gate:{gate}")

    topology = payload.get("topology", {})
    exact_topology = {
        "connected_components": 1,
        "genus": 0,
        "boundary_loops": 4,
        "euler_number": -2,
        "self_intersections": 0,
        "unregistered_body_penetrations": 0,
        "collapsed_triangles": 0,
        "flipped_triangles": 0,
        "winding_consistent": True,
    }
    if not isinstance(topology, dict) or any(
        topology.get(key) != value for key, value in exact_topology.items()
    ):
        blockers.append("exact_four_boundary_topology_gate")
    if int(payload.get("sealed_test_accesses", 0)) != 0:
        blockers.append("sealed_test_access_forbidden")
    if int(payload.get("hidden_cleanup_operations", 0)) != 0:
        blockers.append("hidden_cleanup_forbidden")
    evidence_scope = str(payload.get("evidence_scope", "public_synthetic"))
    if evidence_scope != "train_real":
        blockers.append("real_train_evidence_required_for_h03_promotion")

    capture_mode = str(payload.get("capture_mode", "existing_video_evidence_consistent"))
    independent_reference = bool(payload.get("independent_3d_reference", False))
    metric_evaluator_gates_passed = bool(
        payload.get("controlled_metric_evaluator_gates_passed", False)
    )
    if capture_mode == "single_camera_evidence_consistent" and independent_reference:
        blockers.append("single_camera_capture_cannot_declare_independent_3d_reference")
    if independent_reference and capture_mode != "dual_camera_metric_evaluation":
        blockers.append("dual_camera_metric_evaluation_mode_required")
    if independent_reference and not metric_evaluator_gates_passed:
        blockers.append("controlled_metric_evaluator_gates_required")
    metric_claim_allowed = (
        independent_reference
        and metric_evaluator_gates_passed
        and capture_mode == "dual_camera_metric_evaluation"
    )
    claim = (
        "independently evaluated metric MANTLE reconstruction"
        if metric_claim_allowed
        else "evidence-consistent MANTLE reconstruction"
    )
    return {
        "schema_version": "frayid_v3_mantle_evaluation_report.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "blocked",
        "promotion_eligible": not blockers,
        "claim": claim,
        "capture_mode": capture_mode,
        "metric_accuracy_claim_allowed": metric_claim_allowed,
        "authority": "intrinsic_atlas_conditioned_clipped_sdf",
        "derived_meshes_authoritative": False,
        "g04_geometry_promotion_allowed": False,
        "sealed_test_accesses": int(payload.get("sealed_test_accesses", 0)),
        "blockers": blockers,
    }


def report_dry_run() -> dict[str, Any]:
    digest = "0" * 64
    topology = TopologyCertificateV3(
        connected_components=1,
        genus=0,
        boundary_loops=4,
        euler_number=-2,
        self_intersections=0,
        unregistered_body_penetrations=0,
        collapsed_triangles=0,
        flipped_triangles=0,
        winding_consistent=True,
    )
    atlas = UpperGarmentAtlas(
        experiment_id="postv3_l05_intrinsic_upper_garment_material_atlas_r01",
        evidence_scope="public_synthetic",
        promotion_eligible=False,
        intrinsic_vertices_path="DRY_RUN_NOT_AN_ARTIFACT",
        intrinsic_faces_path="DRY_RUN_NOT_AN_ARTIFACT",
        intrinsic_domain_sha256=digest,
        boundary_cycles={"neck": [], "left_armhole": [], "right_armhole": [], "hem": []},
        seam_hypotheses=[],
        rest_metric=[],
        frame_embeddings=[],
        clipped_sdf_path="DRY_RUN_NOT_AN_ARTIFACT",
        clipped_sdf_sha256=digest,
        body_contact_posterior_path="DRY_RUN_NOT_AN_ARTIFACT",
        body_contact_posterior_sha256=digest,
        uncertainty_support_ledger_path="DRY_RUN_NOT_AN_ARTIFACT",
        uncertainty_support_ledger_sha256=digest,
        topology=topology,
        median_absolute_in_plane_strain=0.0,
        p95_absolute_in_plane_strain=0.0,
        restart_observed_median_spread_mm=0.0,
        restart_observed_p95_spread_mm=0.0,
        status="blocked",
        blockers=["dry_run_only_no_scientific_result"],
    )
    artifact = MantleArtifact(
        experiment_id=EXPERIMENT_ID,
        authority="intrinsic_atlas_conditioned_clipped_sdf",
        claim="evidence-consistent MANTLE reconstruction",
        d03_collision_body_role="immutable_prior_derived_collision_body",
        atlas=atlas,
        neutral_embedding=DerivedSurfaceExport(path="DRY_RUN_NOT_AN_ARTIFACT", sha256=digest),
        posed_exports=[],
        excluded_products=[
            "measurements",
            "sizing",
            "tailoring_patterns",
            "textures",
            "avatars",
            "3dgs",
            "virtual_try_on",
        ],
        status="blocked",
        blockers=["dry_run_only_no_scientific_result"],
    )
    return {
        "schema_version": "frayid_v3_mantle_report_dry_run.v1",
        "status": "pass",
        "dry_run": True,
        "scientific_result_claimed": False,
        "artifact": artifact.model_dump(mode="json"),
        "sealed_test_accesses": 0,
    }
