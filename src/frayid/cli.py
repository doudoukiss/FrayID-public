from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
import typer

from frayid.assets import verify_assets
from frayid.config import load_config
from frayid.dataset import prepare_dataset, validate_dataset
from frayid.doctor import diagnose
from frayid.evaluation import evaluate_checkpoint, evaluate_reconstruction
from frayid.initialization import evaluate_initialization, fit_initialization
from frayid.io import read_json, sha256_file, write_json
from frayid.modal_plan import build_modal_smoke_plan
from frayid.training import run_geometry_smoke
from frayid.v2.audit import write_source_audit
from frayid.v2.benchmark import write_public_benchmark
from frayid.v2.contracts import reject_sealed_capability
from frayid.v2.correspondence import (
    scan_correspondence_viability,
    scan_temporal_track_graph,
)
from frayid.v2.d01_pose_normal_depth import (
    audit_d01_terminal_qualification,
    bind_d01_train_only_normal_evidence,
    evaluate_d01_train_mesh_candidate,
    fit_d01_train_only_mesh_candidate,
    write_d01_public_benchmark,
    write_d01_train_candidate_plan,
)
from frayid.v2.d02_topology_projection import (
    audit_d02_exact_topology,
    evaluate_d02_train_candidate,
    fit_d02_train_topology_projection,
    write_d02_public_benchmark,
    write_d02_train_projection_plan,
)
from frayid.v2.d03_capsule_tree import (
    bind_d03_train_normal_evidence,
    build_d03_implicit_continuation,
    build_d03_real_initialization,
    evaluate_d03_development_images,
    evaluate_d03_train_images,
    write_d03_development_evaluation_plan,
    write_d03_implicit_continuation_plan,
    write_d03_public_benchmark,
    write_d03_real_initialization_plan,
    write_d03_train_evaluation_plan,
    write_d03_train_evidence_plan,
)
from frayid.v2.evaluation import inherited_real_gate, qualify_g01_evaluator_routes
from frayid.v2.evidence import (
    EvidenceVolume,
    bind_t04_hull_inputs,
    build_confidence_aware_visual_hull,
    preserve_sapiens2_semantics,
    qualify_visual_hull_robustness,
)
from frayid.v2.evidence_master import (
    audit_v00_qualification_lifecycle,
    audit_video,
    build_evidence_master,
)
from frayid.v2.frame_selection import (
    eligible_indices_from_dataset_manifest,
    select_phase_uniform_frames,
)
from frayid.v2.g02_endpoint_evaluator import (
    evaluate_g02_frozen_endpoint,
    prepare_g02_endpoint_evidence,
)
from frayid.v2.g02_modal import (
    audit_g02_target_cuda_qualification,
    build_g02_cuda_qualification_plan,
)
from frayid.v2.g02_science import (
    G02ScienceSchedule,
    build_g02_science_arm_binding,
    g02_science_preflight_schedule,
    g02_science_target_preflight_schedule,
    prepare_g02_science_training_evidence,
    run_g02_science_training,
)
from frayid.v2.g02_science_modal import (
    audit_g02_endpoint_evaluation,
    audit_g02_target_science_preflight,
    build_g02_endpoint_evaluation_plan,
    build_g02_scientific_attempt_plan,
    build_g02_target_science_preflight_plan,
)
from frayid.v2.g02_shortcut_resistant import (
    audit_g02_local_lifecycle,
    qualify_g02_local,
    write_g02_public_benchmark,
)
from frayid.v2.g02_topology import audit_g02_raw_topology
from frayid.v2.g03_appearance import write_g03_public_benchmark
from frayid.v2.g03_pipeline import (
    evaluate_g03_appearance_split,
    fit_g03_train_only_appearance,
)
from frayid.v2.g04_phase_appearance import (
    evaluate_g04_phase_appearance_split,
    fit_g04_train_only_phase_appearance,
    render_g04_full_sequence,
    write_g04_public_benchmark,
)
from frayid.v2.l03_modal import (
    audit_l03_target_cuda_qualification,
    build_l03_cuda_qualification_plan,
)
from frayid.v2.l03_open_layers import (
    audit_l03_semantic_support,
    build_l03_real_initialization,
    write_l03_public_benchmark,
    write_l03_real_initialization_plan,
    write_l03_semantic_support_plan,
)
from frayid.v2.l03_training import (
    run_l03_local_training_qualification,
    write_l03_training_public_benchmark,
    write_l03_training_qualification_plan,
)
from frayid.v2.material_tracks import scan_visibility_bounded_material_tracks
from frayid.v2.modal_qualification import build_g01_cuda_qualification_plan
from frayid.v2.posed_preview import render_posed_preview
from frayid.v2.q03_interval_tracks import (
    audit_q03_qualification_lifecycle,
    qualify_q03_interval_track_graph,
    write_q03_public_benchmark,
)
from frayid.v2.qualification import (
    audit_g01_local_qualification_lifecycle,
    audit_g01_target_cuda_qualification,
    qualify_layered_model,
    qualify_normal_transport,
    qualify_outer_field,
)
from frayid.v2.schemas import V2EvaluationReport
from frayid.v2.semantics import (
    audit_s01_qualification_lifecycle,
    qualify_sapiens2_semantic_directory,
)
from frayid.v2.source_binding import resolve_packaged_source_revision
from frayid.v2.t01_capacity import qualify_real_residual_capacity
from frayid.v2.t01_evaluator import evaluate_real_turntable_development
from frayid.v2.t01_joint import qualify_real_joint_schur_step
from frayid.v2.t01_phase import qualify_real_phase_axis_step
from frayid.v2.t01_silhouette import qualify_real_center_focal_step
from frayid.v2.t02_geodesic import qualify_real_t02_internal_validation
from frayid.v2.t03_dynamic import qualify_real_t03_internal_validation
from frayid.v2.t04_uncertainty import (
    audit_t04_qualification_lifecycle,
    qualify_uncertainty_tagged_dynamic_camera,
)
from frayid.v2.t05_fixed_camera import (
    audit_t05_qualification_lifecycle,
    audit_t05_training_background,
    evaluate_t05_development_nonregression,
    fit_t05_fixed_camera_solution,
    write_t05_public_benchmark,
)
from frayid.v2.turntable import (
    initialize_cooperative_turntable_solution,
    initialize_turntable_solution,
)
from frayid.v2.turntable_ba import (
    diagnose_real_turntable_factor_route,
    write_turntable_ba_benchmark,
)

app = typer.Typer(help="FrayID canonical clothed-surface reconstruction.", no_args_is_help=True)
assets_app = typer.Typer(help="Verify controlled local assets.", no_args_is_help=True)
dataset_app = typer.Typer(help="Prepare and validate sequence evidence.", no_args_is_help=True)
initialize_app = typer.Typer(
    help="Fit and evaluate shared SMPL/camera initialization.", no_args_is_help=True
)
reconstruct_app = typer.Typer(
    help="Plan, smoke-test, and evaluate canonical geometry.", no_args_is_help=True
)
v2_app = typer.Typer(
    help="Build and qualify the layered FrayID V2 successor without mutating V1.",
    no_args_is_help=True,
)
app.add_typer(assets_app, name="assets")
app.add_typer(dataset_app, name="dataset")
app.add_typer(initialize_app, name="initialize")
app.add_typer(reconstruct_app, name="reconstruct")
app.add_typer(v2_app, name="v2")

ConfigOption = Annotated[
    Path,
    typer.Option("--config", exists=True, dir_okay=False, readable=True, help="V1 YAML config."),
]
DEFAULT_SMPL_MODEL_ROOT = Path("models/private/camerahmr_assets/models/SMPL")
DEFAULT_V00_OUTPUT_ROOT = Path(
    "outputs/post_v2/postv2_v00_capture_forensics_evidence_master_r01/registered-20260903-r01"
)
DEFAULT_T05_OUTPUT_ROOT = Path(
    "outputs/post_v2/postv2_t05_background_anchored_fixed_camera_human_ba_r01/"
    "registered-20260903-r01/qualification"
)
DEFAULT_Q03_OUTPUT_ROOT = Path(
    "outputs/post_v2/postv2_q03_interval_material_track_graph_r01/"
    "registered-20260903-r01/qualification"
)
DEFAULT_G02_OUTPUT_ROOT = Path(
    "outputs/post_v2/postv2_g02_direct_multires_field_matched_science_r01/"
    "registered-20260903-r01/qualification"
)
DEFAULT_G02_EXPERIMENT_ROOT = DEFAULT_G02_OUTPUT_ROOT.parent
DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT = DEFAULT_G02_EXPERIMENT_ROOT / "science_preflight"
DEFAULT_G02_CONTRACT = Path(
    "configs/evaluation/post_v2_g02_direct_multires_field_matched_science_r01.yaml"
)
DEFAULT_G02_ATTEMPT_R02_ROOT = DEFAULT_G02_EXPERIMENT_ROOT / "scientific-attempt-r02"
DEFAULT_G02_RECOVERY_R02_ROOT = (
    DEFAULT_G02_ATTEMPT_R02_ROOT / "volume_recovery_r01/scientific-attempt-r02"
)
DEFAULT_D01_OUTPUT_ROOT = Path(
    "outputs/post_v2/postv2_d01_pose_stabilized_normal_depth_fusion_r01/"
    "registered-20260903-r01/qualification"
)
DEFAULT_D02_OUTPUT_ROOT = Path(
    "outputs/post_v2/postv2_d02_topology_constrained_normal_projection_r01/"
    "registered-20260903-r01/qualification"
)
DEFAULT_D03_OUTPUT_ROOT = Path(
    "outputs/post_v2/postv2_d03_capsule_tree_implicit_body_r01/"
    "registered-20260903-r01/qualification"
)
DEFAULT_L03_OUTPUT_ROOT = Path(
    "outputs/post_v2/postv2_l03_semantic_open_clothing_layers_r01/"
    "registered-20260903-r01/qualification"
)


def _emit(value: Any) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def doctor(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
) -> None:
    """Check the local runtime without creating reconstruction artifacts."""
    report = diagnose(config)
    _emit(report)
    if report["status"] != "ready":
        raise typer.Exit(2)


@assets_app.command("verify")
def assets_verify(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    settings = load_config(config)
    report = verify_assets(settings.paths.asset_manifest, output_path=output)
    _emit(report)
    if report.status != "ready":
        raise typer.Exit(2)


@dataset_app.command("prepare")
def dataset_prepare(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    _emit(prepare_dataset(load_config(config), overwrite=overwrite))


@dataset_app.command("validate")
def dataset_validate(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
) -> None:
    report = validate_dataset(load_config(config), manifest_path=manifest)
    _emit(report)
    if report.status != "ready":
        raise typer.Exit(2)


@initialize_app.command("fit")
def initialize_fit(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    input_path: Annotated[Path | None, typer.Option("--input")] = None,
    output_path: Annotated[Path | None, typer.Option("--output")] = None,
    model_root: Annotated[Path, typer.Option("--model-root")] = DEFAULT_SMPL_MODEL_ROOT,
    steps: Annotated[int, typer.Option(min=1)] = 100,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    result = fit_initialization(
        load_config(config),
        input_path=input_path,
        output_path=output_path,
        model_root=model_root,
        steps=steps,
        device_name=device,
    )
    _emit(result)


@initialize_app.command("evaluate")
def initialize_evaluate(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    initialization: Annotated[Path | None, typer.Option("--initialization")] = None,
    model_root: Annotated[Path, typer.Option("--model-root")] = DEFAULT_SMPL_MODEL_ROOT,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    report = evaluate_initialization(
        load_config(config),
        initialization_path=initialization,
        model_root=model_root,
        device_name=device,
    )
    _emit(report)
    if report.status != "pass":
        raise typer.Exit(2)


@reconstruct_app.command("smoke")
def reconstruct_smoke(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
    device: Annotated[str | None, typer.Option()] = None,
) -> None:
    settings = load_config(config)
    if settings.smoke.epoch_count != 2 or settings.smoke.frame_count != 24:
        raise typer.BadParameter("V1 smoke is fixed to exactly 24 frames and 2 epochs")
    report = run_geometry_smoke(
        settings,
        device_name=device,
        resume_path=resume,
        config_source_path=config,
    )
    _emit(report)
    if report.status != "pass":
        raise typer.Exit(2)


@reconstruct_app.command("plan-modal")
def reconstruct_plan_modal(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    report = build_modal_smoke_plan(load_config(config), config_path=config, output_path=output)
    _emit(report)


@reconstruct_app.command("evaluate")
def reconstruct_evaluate(
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", exists=True, dir_okay=False)
    ] = None,
    metrics: Annotated[Path | None, typer.Option("--metrics", exists=True, dir_okay=False)] = None,
    mesh: Annotated[Path | None, typer.Option("--mesh", exists=True, dir_okay=False)] = None,
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    output: Annotated[Path | None, typer.Option("--output")] = None,
    device: Annotated[str | None, typer.Option()] = None,
) -> None:
    settings = load_config(config)
    if checkpoint is not None:
        output_directory = output or checkpoint.parent / "evaluation"
        report = evaluate_checkpoint(
            settings,
            checkpoint_path=checkpoint,
            output_directory=output_directory,
            device_name=device,
        )
    elif metrics is not None and mesh is not None:
        report = evaluate_reconstruction(
            settings, metrics_path=metrics, mesh_path=mesh, output_path=output
        )
    else:
        raise typer.BadParameter("Provide --checkpoint, or provide both --metrics and --mesh")
    _emit(report)
    if report.status != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-pipeline")
def v2_audit_pipeline(
    source_root: Annotated[
        Path, typer.Option("--source-root", exists=True, file_okay=False)
    ] = Path("src/frayid"),
    output: Annotated[Path, typer.Option("--output")] = Path(
        "outputs/post_v2/qualification/source_audit.json"
    ),
) -> None:
    """Create a commit-bound source/gradient-route inventory without reading evidence."""
    reject_sealed_capability([source_root, output])
    path = write_source_audit(source_root, output)
    _emit(read_json(path))


@v2_app.command("video-audit")
def v2_video_audit(
    source_revision: Annotated[str, typer.Option("--source-revision")],
    video: Annotated[
        Path, typer.Option("--video", exists=True, dir_okay=False, readable=True)
    ] = Path("local_input.mp4"),
    output: Annotated[Path, typer.Option("--output")] = Path(
        "outputs/post_v2/qualification/v00_video_audit.json"
    ),
) -> None:
    """Audit native PTS, decoded pixels, quality, duplicates, and background motion."""
    path = audit_video(
        video,
        output,
        source_revision=source_revision,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("build-evidence-master")
def v2_build_evidence_master(
    source_revision: Annotated[str, typer.Option("--source-revision")],
    video: Annotated[
        Path, typer.Option("--video", exists=True, dir_okay=False, readable=True)
    ] = Path("local_input.mp4"),
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_V00_OUTPUT_ROOT,
    run_id: Annotated[str, typer.Option("--run-id")] = "registered-20260903-r01",
    hashes_only: Annotated[
        bool,
        typer.Option(
            "--hashes-only",
            help="Audit decoded pixels without retaining lossless PNG frames.",
        ),
    ] = False,
    proxy_width: Annotated[int, typer.Option("--proxy-width", min=1)] = 512,
    proxy_height: Annotated[int, typer.Option("--proxy-height", min=1)] = 768,
) -> None:
    """Atomically build the immutable V00 raw evidence master by sequential decode."""
    path = build_evidence_master(
        video,
        output_root,
        source_revision=source_revision,
        run_id=run_id,
        storage="hashes_only" if hashes_only else "png",
        proxy_size=(proxy_width, proxy_height),
    )
    report = read_json(path)
    _emit(
        {
            "status": report["status"],
            "blockers": report["blockers"],
            "evidence_master": str(path),
            "decoded_frame_count": report["decode"]["decoded_frame_count"],
            "physical_camera_verdict": report["physical_camera_verdict"],
        }
    )
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-v00-lifecycle")
def v2_audit_v00_lifecycle(
    evidence_master: Annotated[
        Path, typer.Option("--evidence-master", exists=True, dir_okay=False)
    ] = DEFAULT_V00_OUTPUT_ROOT / "evidence_master.json",
    qualification: Annotated[
        Path, typer.Option("--qualification", exists=True, dir_okay=False)
    ] = DEFAULT_V00_OUTPUT_ROOT / "qualification.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_V00_OUTPUT_ROOT
    / "qualification_lifecycle.json",
) -> None:
    """Restore all V00 frames and audit ordered qualification transitions."""
    path = audit_v00_qualification_lifecycle(evidence_master, qualification, output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("benchmark-t05")
def v2_benchmark_t05(
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_T05_OUTPUT_ROOT
    / "public_fixed_camera_benchmark.json",
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260903,
) -> None:
    """Run T05 fixed-camera/root/yaw recovery and identifiability controls."""
    path = write_t05_public_benchmark(output, seed=seed)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("fit-t05")
def v2_fit_t05(
    source_revision: Annotated[str, typer.Option("--source-revision")],
    initialization: Annotated[Path, typer.Option("--initialization", exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    v00_lifecycle: Annotated[
        Path, typer.Option("--v00-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_V00_OUTPUT_ROOT / "qualification_lifecycle.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_T05_OUTPUT_ROOT
    / "fixed_camera_human_solution.json",
    t04_solution: Annotated[
        Path | None, typer.Option("--t04-solution", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Fit train-only monotonic human yaw/root while fixing the physical camera."""
    path = fit_t05_fixed_camera_solution(
        initialization,
        manifest,
        v00_lifecycle,
        output,
        source_revision=source_revision,
        t04_solution_path=t04_solution,
    )
    solution = read_json(path)
    _emit(
        {
            "status": solution["status"],
            "solution": str(path),
            "training_frame_count": solution["training_frame_count"],
            "spin_axis_camera": solution["spin_axis_camera"],
            "uncertainty": solution["uncertainty"],
        }
    )


@v2_app.command("audit-t05-background")
def v2_audit_t05_background(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    evidence_master: Annotated[
        Path, typer.Option("--evidence-master", exists=True, dir_okay=False)
    ] = DEFAULT_V00_OUTPUT_ROOT / "evidence_master.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_T05_OUTPUT_ROOT
    / "training_background_audit.json",
) -> None:
    """Recheck the fixed physical camera using only frozen training frames."""
    path = audit_t05_training_background(evidence_master, manifest, output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("evaluate-t05")
def v2_evaluate_t05(
    solution: Annotated[Path, typer.Option("--solution", exists=True, dir_okay=False)],
    initialization: Annotated[Path, typer.Option("--initialization", exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    canonical_mesh: Annotated[Path, typer.Option("--canonical-mesh", exists=True, dir_okay=False)],
    skinning_weights: Annotated[
        Path, typer.Option("--skinning-weights", exists=True, dir_okay=False)
    ],
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_T05_OUTPUT_ROOT
    / "development_nonregression.json",
    render_resolution: Annotated[int, typer.Option("--render-resolution", min=16)] = 64,
) -> None:
    """Score T05 reownership against the identical held-out free-root scaffold."""
    path = evaluate_t05_development_nonregression(
        solution,
        initialization,
        manifest,
        mask_root,
        canonical_mesh,
        skinning_weights,
        joint_transforms,
        output,
        render_resolution=render_resolution,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-t05-lifecycle")
def v2_audit_t05_lifecycle(
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "public_fixed_camera_benchmark.json",
    solution: Annotated[
        Path, typer.Option("--solution", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "fixed_camera_human_solution.json",
    background_report: Annotated[
        Path, typer.Option("--background-report", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "training_background_audit.json",
    development_report: Annotated[
        Path, typer.Option("--development-report", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "development_nonregression.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_T05_OUTPUT_ROOT
    / "qualification_lifecycle.json",
) -> None:
    """Audit T05's complete ordered local qualification lifecycle."""
    path = audit_t05_qualification_lifecycle(
        public_benchmark,
        solution,
        background_report,
        development_report,
        output,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("select-phase-frames")
def v2_select_phase_frames(
    evidence_master: Annotated[
        Path, typer.Option("--evidence-master", exists=True, dir_okay=False)
    ],
    phase_solution: Annotated[Path, typer.Option("--phase-solution", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    count: Annotated[int, typer.Option("--count", min=4)] = 144,
    eligible_manifest: Annotated[
        Path | None,
        typer.Option("--eligible-manifest", exists=True, dir_okay=False),
    ] = None,
    minimum_confidence: Annotated[
        float, typer.Option("--minimum-confidence", min=0.0, max=1.0)
    ] = 0.0,
) -> None:
    """Select train-eligible source frames uniformly in T05 yaw, never in time."""
    eligible = (
        None
        if eligible_manifest is None
        else eligible_indices_from_dataset_manifest(eligible_manifest, split="train")
    )
    path = select_phase_uniform_frames(
        evidence_master,
        phase_solution,
        output,
        count=count,
        eligible_source_indices=eligible,
        minimum_confidence=minimum_confidence,
    )
    _emit(read_json(path))


@v2_app.command("benchmark-q03")
def v2_benchmark_q03(
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_Q03_OUTPUT_ROOT
    / "public_interval_track_benchmark.json",
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260903,
) -> None:
    """Run Q03 clean, corrupted, partial-cycle, and photometric controls."""
    path = write_q03_public_benchmark(output, seed=seed)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-q03")
def v2_qualify_q03(
    source_revision: Annotated[str, typer.Option("--source-revision")],
    q01_report: Annotated[Path, typer.Option("--q01-report", exists=True, dir_okay=False)],
    q01_binding: Annotated[Path, typer.Option("--q01-binding", exists=True, dir_okay=False)],
    q02_report: Annotated[Path, typer.Option("--q02-report", exists=True, dir_okay=False)],
    q02_photometric_report: Annotated[
        Path, typer.Option("--q02-photometric-report", exists=True, dir_okay=False)
    ],
    q02_binding: Annotated[Path, typer.Option("--q02-binding", exists=True, dir_okay=False)],
    semantic_qualification: Annotated[
        Path, typer.Option("--semantic-qualification", exists=True, dir_okay=False)
    ],
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "public_interval_track_benchmark.json",
    t05_solution: Annotated[
        Path, typer.Option("--t05-solution", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "fixed_camera_human_solution.json",
    t05_lifecycle: Annotated[
        Path, typer.Option("--t05-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "qualification_lifecycle.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_Q03_OUTPUT_ROOT
    / "interval_track_qualification.json",
    binding_output: Annotated[Path, typer.Option("--binding-output")] = DEFAULT_Q03_OUTPUT_ROOT
    / "interval_material_track_graph.npz",
) -> None:
    """Promote only robust T05-phased local and medium material intervals."""
    report_path, binding_path = qualify_q03_interval_track_graph(
        public_benchmark,
        t05_solution,
        t05_lifecycle,
        q01_report,
        q01_binding,
        q02_report,
        q02_photometric_report,
        q02_binding,
        semantic_qualification,
        output,
        binding_output,
        source_revision=source_revision,
    )
    report = read_json(report_path)
    _emit(
        {
            "status": report["status"],
            "blockers": report["blockers"],
            "track_metrics": report["track_metrics"],
            "binding": str(binding_path),
        }
    )
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-q03-lifecycle")
def v2_audit_q03_lifecycle(
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "public_interval_track_benchmark.json",
    qualification: Annotated[
        Path, typer.Option("--qualification", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "interval_track_qualification.json",
    binding: Annotated[Path, typer.Option("--binding", exists=True, dir_okay=False)] = (
        DEFAULT_Q03_OUTPUT_ROOT / "interval_material_track_graph.npz"
    ),
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_Q03_OUTPUT_ROOT
    / "qualification_lifecycle.json",
) -> None:
    """Audit Q03's ordered qualification and fail-closed evidence boundary."""
    path = audit_q03_qualification_lifecycle(
        public_benchmark,
        qualification,
        binding,
        output,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("benchmark-g02")
def v2_benchmark_g02(
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_OUTPUT_ROOT
    / "public_shortcut_resistant_field_benchmark.json",
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260903,
) -> None:
    """Run G02 multiscale-gradient, factor, and adversarial public controls."""
    path = write_g02_public_benchmark(output, seed=seed)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-g02-local")
def v2_qualify_g02_local(
    source_revision: Annotated[str, typer.Option("--source-revision")],
    evidence_volume: Annotated[
        Path, typer.Option("--evidence-volume", exists=True, dir_okay=False)
    ],
    evidence_binding: Annotated[
        Path, typer.Option("--evidence-binding", exists=True, dir_okay=False)
    ],
    hull_qualification: Annotated[
        Path, typer.Option("--hull-qualification", exists=True, dir_okay=False)
    ],
    normal_root: Annotated[Path, typer.Option("--normal-root", exists=True, file_okay=False)],
    t05_solution: Annotated[
        Path, typer.Option("--t05-solution", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "fixed_camera_human_solution.json",
    t05_lifecycle: Annotated[
        Path, typer.Option("--t05-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "qualification_lifecycle.json",
    q03_qualification: Annotated[
        Path, typer.Option("--q03-qualification", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "interval_track_qualification.json",
    q03_binding: Annotated[
        Path, typer.Option("--q03-binding", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "interval_material_track_graph.npz",
    q03_lifecycle: Annotated[
        Path, typer.Option("--q03-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "qualification_lifecycle.json",
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "public_shortcut_resistant_field_benchmark.json",
    arm_binding_output: Annotated[
        Path, typer.Option("--arm-binding-output")
    ] = DEFAULT_G02_OUTPUT_ROOT / "matched_arm_binding.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_OUTPUT_ROOT
    / "local_engineering_qualification.json",
    device: Annotated[str, typer.Option("--device")] = "cpu",
    extraction_device: Annotated[str, typer.Option("--extraction-device")] = "cpu",
    flexicubes_repository: Annotated[
        Path | None,
        typer.Option("--flexicubes-repository", exists=True, file_okay=False),
    ] = None,
) -> None:
    """Exercise G02 once on real train evidence without starting science."""
    report_path, arm_path = qualify_g02_local(
        evidence_volume,
        evidence_binding,
        hull_qualification,
        t05_solution,
        t05_lifecycle,
        q03_qualification,
        q03_binding,
        q03_lifecycle,
        public_benchmark,
        normal_root,
        arm_binding_output,
        output,
        source_revision=source_revision,
        device=device,
        extraction_device=extraction_device,
        flexicubes_repository=flexicubes_repository,
    )
    report = read_json(report_path)
    _emit(
        {
            "status": report["status"],
            "blockers": report["blockers"],
            "matched_arm_binding": str(arm_path),
            "checkpoint": report["checkpoint"],
        }
    )
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-g02-local-lifecycle")
def v2_audit_g02_local_lifecycle(
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "public_shortcut_resistant_field_benchmark.json",
    qualification: Annotated[
        Path, typer.Option("--qualification", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "local_engineering_qualification.json",
    arm_binding: Annotated[
        Path, typer.Option("--arm-binding", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "matched_arm_binding.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_OUTPUT_ROOT
    / "local_qualification_lifecycle.json",
) -> None:
    """Audit G02 only through checkpoint-restored local engineering state."""
    path = audit_g02_local_lifecycle(
        public_benchmark,
        qualification,
        arm_binding,
        output,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-g02-cuda")
def v2_plan_g02_cuda(
    provider_rate_usd_per_hour: Annotated[
        float, typer.Option("--provider-rate-usd-per-hour", min=0.0)
    ],
    price_checked_at: Annotated[str, typer.Option("--price-checked-at")],
    maximum_cost_usd: Annotated[float, typer.Option("--maximum-cost-usd", min=0.0)],
    contract: Annotated[Path, typer.Option("--contract", exists=True, dir_okay=False)] = Path(
        "configs/evaluation/post_v2_g02_direct_multires_field_matched_science_r01.yaml"
    ),
    local_qualification: Annotated[
        Path, typer.Option("--local-qualification", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "local_engineering_qualification_r02.json",
    local_lifecycle: Annotated[
        Path, typer.Option("--local-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "local_qualification_lifecycle_r02.json",
    arm_binding: Annotated[
        Path, typer.Option("--arm-binding", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "matched_arm_binding_r02.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_OUTPUT_ROOT
    / "target_cuda_plan_r01.json",
    dispatch_authorized: Annotated[bool, typer.Option("--dispatch-authorized")] = False,
) -> None:
    """Create the priced, zero-retry G02 target-CUDA qualification plan."""
    input_root = DEFAULT_G02_OUTPUT_ROOT / "inputs"
    inputs = {
        "evidence_volume": input_root / "t04_s01_visual_hull_r32_r07.npz",
        "evidence_binding": input_root / "t04_s01_hull_inputs_256_r02.npz",
        "hull_qualification": input_root / "t04_s01_visual_hull_qualification_r32_r07.json",
        "t05_solution": DEFAULT_T05_OUTPUT_ROOT / "fixed_camera_human_solution.json",
        "t05_lifecycle": DEFAULT_T05_OUTPUT_ROOT / "qualification_lifecycle.json",
        "q03_qualification": DEFAULT_Q03_OUTPUT_ROOT / "interval_track_qualification.json",
        "q03_binding": DEFAULT_Q03_OUTPUT_ROOT / "interval_material_track_graph.npz",
        "q03_lifecycle": DEFAULT_Q03_OUTPUT_ROOT / "qualification_lifecycle.json",
        "public_benchmark": DEFAULT_G02_OUTPUT_ROOT
        / "public_shortcut_resistant_field_benchmark.json",
    }
    normal_root = input_root / "normals"
    for index, path in enumerate(sorted(normal_root.glob("*.png"))):
        inputs[f"normal_{index}"] = path
    path = build_g02_cuda_qualification_plan(
        project_root=Path.cwd(),
        contract_path=contract,
        local_qualification_path=local_qualification,
        local_lifecycle_path=local_lifecycle,
        matched_arm_binding_path=arm_binding,
        input_paths=inputs,
        output_path=output,
        provider_rate_usd_per_hour=provider_rate_usd_per_hour,
        price_checked_at=price_checked_at,
        maximum_cost_usd=maximum_cost_usd,
        dispatch_authorized=dispatch_authorized,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "ready":
        raise typer.Exit(2)


@v2_app.command("audit-g02-cuda-lifecycle")
def v2_audit_g02_cuda_lifecycle(
    envelope: Annotated[
        Path, typer.Option("--envelope", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "target_cuda_qualification_envelope_r01.json",
    claim: Annotated[
        Path, typer.Option("--claim", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "target_cuda_qualification_claim_r01.json",
    plan: Annotated[
        Path, typer.Option("--plan", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "target_cuda_plan_r01.json",
    local_lifecycle: Annotated[
        Path, typer.Option("--local-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "local_qualification_lifecycle_r02.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_OUTPUT_ROOT
    / "target_cuda_qualification_lifecycle_r01.json",
) -> None:
    """Promote G02 engineering qualification only after audited CUDA replay."""
    path = audit_g02_target_cuda_qualification(
        envelope,
        claim,
        plan,
        local_lifecycle,
        output,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("prepare-g02-science-evidence")
def v2_prepare_g02_science_evidence(
    normal_root: Annotated[Path, typer.Option("--normal-root", exists=True, file_okay=False)],
    hull_binding: Annotated[
        Path, typer.Option("--hull-binding", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "inputs/t04_s01_hull_inputs_256_r02.npz",
    t05_solution: Annotated[
        Path, typer.Option("--t05-solution", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "fixed_camera_human_solution.json",
    q03_binding: Annotated[
        Path, typer.Option("--q03-binding", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "interval_material_track_graph.npz",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT
    / "science_training_evidence_r01.npz",
    rays_per_view_stratum: Annotated[int, typer.Option("--rays-per-view-stratum", min=4)] = 16,
) -> None:
    """Freeze all train-only G02 science rays, normals, and Q03 anchors."""
    path = prepare_g02_science_training_evidence(
        hull_binding,
        normal_root,
        t05_solution,
        q03_binding,
        output,
        rays_per_view_stratum=rays_per_view_stratum,
    )
    _emit({"status": "pass", "path": str(path), "sha256": sha256_file(path)})


@v2_app.command("bind-g02-science-arms")
def v2_bind_g02_science_arms(
    source_revision: Annotated[str, typer.Option("--source-revision")],
    qualification_lifecycle: Annotated[
        Path, typer.Option("--qualification-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "target_cuda_qualification_lifecycle_r01.json",
    training_evidence: Annotated[
        Path, typer.Option("--training-evidence", exists=True, dir_okay=False)
    ] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "science_training_evidence_r01.npz",
    evidence_volume: Annotated[
        Path, typer.Option("--evidence-volume", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "inputs/t04_s01_visual_hull_r32_r07.npz",
    q03_binding: Annotated[
        Path, typer.Option("--q03-binding", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "interval_material_track_graph.npz",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT
    / "science_arm_binding_r01.json",
    preflight: Annotated[bool, typer.Option("--preflight")] = False,
    target_cuda_preflight: Annotated[bool, typer.Option("--target-cuda-preflight")] = False,
    attempt_id: Annotated[str, typer.Option("--attempt-id")] = "scientific-attempt-r01",
) -> None:
    """Bind immutable matched treatment/control inputs and schedule."""
    if preflight and target_cuda_preflight:
        raise typer.BadParameter("choose at most one G02 preflight schedule")
    schedule = (
        g02_science_preflight_schedule()
        if preflight
        else (
            g02_science_target_preflight_schedule()
            if target_cuda_preflight
            else G02ScienceSchedule()
        )
    )
    path = build_g02_science_arm_binding(
        output,
        source_revision=source_revision,
        qualification_lifecycle_path=qualification_lifecycle,
        training_evidence_path=training_evidence,
        evidence_volume_path=evidence_volume,
        q03_binding_path=q03_binding,
        schedule=schedule,
        attempt_id=attempt_id,
    )
    _emit({"status": "pass", "path": str(path), "sha256": sha256_file(path)})


@v2_app.command("dry-run-g02-science")
def v2_dry_run_g02_science(
    source_revision: Annotated[str, typer.Option("--source-revision")],
    arm_binding: Annotated[
        Path, typer.Option("--arm-binding", exists=True, dir_okay=False)
    ] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "science_arm_binding_r01.json",
    training_evidence: Annotated[
        Path, typer.Option("--training-evidence", exists=True, dir_okay=False)
    ] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "science_training_evidence_r01.npz",
    evidence_volume: Annotated[
        Path, typer.Option("--evidence-volume", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "inputs/t04_s01_visual_hull_r32_r07.npz",
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "local_dry_run_r01"
    ),
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Exercise all G02 science stages without creating an attempt marker."""
    path = run_g02_science_training(
        evidence_volume,
        training_evidence,
        arm_binding,
        output_root,
        source_revision=source_revision,
        device=device,
        schedule=g02_science_preflight_schedule(),
        raw_field_resolution=16,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "endpoint_frozen_unscored":
        raise typer.Exit(2)


@v2_app.command("plan-g02-science-preflight")
def v2_plan_g02_science_preflight(
    provider_rate_usd_per_hour: Annotated[float, typer.Option("--provider-rate")],
    price_checked_at: Annotated[str, typer.Option("--price-checked-at")],
    maximum_cost_usd: Annotated[float, typer.Option("--maximum-cost-usd")],
    contract: Annotated[
        Path, typer.Option("--contract", exists=True, dir_okay=False)
    ] = DEFAULT_G02_CONTRACT,
    qualification_lifecycle: Annotated[
        Path, typer.Option("--qualification-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "target_cuda_qualification_lifecycle_r01.json",
    local_preflight_report: Annotated[
        Path, typer.Option("--local-preflight-report", exists=True, dir_okay=False)
    ] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "local_dry_run_r03/training_report.json",
    preflight_arm_binding: Annotated[
        Path, typer.Option("--preflight-arm-binding", exists=True, dir_okay=False)
    ] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "science_arm_binding_target_preflight_r02.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT
    / "target_preflight_plan_r02.json",
    dispatch_authorized: Annotated[bool, typer.Option("--dispatch-authorized")] = False,
) -> None:
    """Fail closed before the paid three-step L40S science preflight."""
    path = build_g02_target_science_preflight_plan(
        project_root=Path.cwd(),
        contract_path=contract,
        qualification_lifecycle_path=qualification_lifecycle,
        local_preflight_report_path=local_preflight_report,
        preflight_arm_binding_path=preflight_arm_binding,
        output_path=output,
        provider_rate_usd_per_hour=provider_rate_usd_per_hour,
        price_checked_at=price_checked_at,
        maximum_cost_usd=maximum_cost_usd,
        dispatch_authorized=dispatch_authorized,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "ready":
        raise typer.Exit(2)


@v2_app.command("audit-g02-science-preflight")
def v2_audit_g02_science_preflight(
    envelope: Annotated[Path, typer.Option("--envelope", exists=True, dir_okay=False)],
    plan: Annotated[
        Path, typer.Option("--plan", exists=True, dir_okay=False)
    ] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "target_preflight_plan_r02.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT
    / "target_preflight_audit_r02.json",
) -> None:
    """Audit the target preflight without promoting a scientific result."""
    path = audit_g02_target_science_preflight(envelope, plan, output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-g02-science")
def v2_plan_g02_science(
    provider_rate_usd_per_hour: Annotated[float, typer.Option("--provider-rate")],
    price_checked_at: Annotated[str, typer.Option("--price-checked-at")],
    maximum_cost_usd: Annotated[float, typer.Option("--maximum-cost-usd")],
    contract: Annotated[
        Path, typer.Option("--contract", exists=True, dir_okay=False)
    ] = DEFAULT_G02_CONTRACT,
    qualification_lifecycle: Annotated[
        Path, typer.Option("--qualification-lifecycle", exists=True, dir_okay=False)
    ] = DEFAULT_G02_OUTPUT_ROOT / "target_cuda_qualification_lifecycle_r01.json",
    target_preflight_audit: Annotated[
        Path, typer.Option("--target-preflight-audit", exists=True, dir_okay=False)
    ] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "target_preflight_audit_r02.json",
    science_arm_binding: Annotated[
        Path, typer.Option("--science-arm-binding", exists=True, dir_okay=False)
    ] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT / "science_arm_binding_scientific_r02.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_SCIENCE_PREFLIGHT_ROOT
    / "scientific_attempt_plan_r02.json",
    dispatch_authorized: Annotated[bool, typer.Option("--dispatch-authorized")] = False,
) -> None:
    """Fail closed before consuming the one-shot G02 scientific attempt."""
    path = build_g02_scientific_attempt_plan(
        project_root=Path.cwd(),
        contract_path=contract,
        qualification_lifecycle_path=qualification_lifecycle,
        target_preflight_audit_path=target_preflight_audit,
        science_arm_binding_path=science_arm_binding,
        output_path=output,
        provider_rate_usd_per_hour=provider_rate_usd_per_hour,
        price_checked_at=price_checked_at,
        maximum_cost_usd=maximum_cost_usd,
        dispatch_authorized=dispatch_authorized,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "ready":
        raise typer.Exit(2)


@v2_app.command("prepare-g02-endpoint-evidence")
def v2_prepare_g02_endpoint_evidence(
    scientific_envelope: Annotated[
        Path, typer.Option("--scientific-envelope", exists=True, dir_okay=False)
    ],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    evidence_volume: Annotated[
        Path, typer.Option("--evidence-volume", exists=True, dir_okay=False)
    ],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    initialization: Annotated[Path, typer.Option("--initialization", exists=True, dir_okay=False)],
    t05_solution: Annotated[Path, typer.Option("--t05-solution", exists=True, dir_okay=False)],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    normal_root: Annotated[Path, typer.Option("--normal-root", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    render_resolution: Annotated[int, typer.Option("--render-resolution", min=16)] = 128,
) -> None:
    """Bind all 144/36 endpoint evidence after science training is frozen."""
    path = prepare_g02_endpoint_evidence(
        scientific_envelope,
        checkpoint,
        evidence_volume,
        manifest,
        initialization,
        t05_solution,
        mask_root,
        normal_root,
        output,
        render_resolution=render_resolution,
    )
    _emit({"status": "pass", "path": str(path), "sha256": sha256_file(path)})


@v2_app.command("evaluate-g02-endpoint")
def v2_evaluate_g02_endpoint(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    evidence_volume: Annotated[
        Path, typer.Option("--evidence-volume", exists=True, dir_okay=False)
    ],
    endpoint_evidence: Annotated[
        Path, typer.Option("--endpoint-evidence", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")],
    device: Annotated[str, typer.Option("--device")] = "cpu",
    ray_batch_size: Annotated[int, typer.Option("--ray-batch-size", min=1)] = 512,
) -> None:
    """Independently score the frozen G02 treatment and matched control."""
    path = evaluate_g02_frozen_endpoint(
        checkpoint,
        evidence_volume,
        endpoint_evidence,
        output,
        device=device,
        ray_batch_size=ray_batch_size,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-g02-endpoint-evaluation")
def v2_plan_g02_endpoint_evaluation(
    provider_rate_usd_per_hour: Annotated[float, typer.Option("--provider-rate")],
    price_checked_at: Annotated[str, typer.Option("--price-checked-at")],
    maximum_cost_usd: Annotated[float, typer.Option("--maximum-cost-usd")],
    scientific_envelope: Annotated[
        Path, typer.Option("--scientific-envelope", exists=True, dir_okay=False)
    ] = DEFAULT_G02_RECOVERY_R02_ROOT / "scientific_attempt_envelope.json",
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", exists=True, dir_okay=False)
    ] = DEFAULT_G02_RECOVERY_R02_ROOT / "treatment/final_checkpoint.pt",
    endpoint_evidence: Annotated[
        Path, typer.Option("--endpoint-evidence", exists=True, dir_okay=False)
    ] = DEFAULT_G02_ATTEMPT_R02_ROOT / "evaluation/endpoint_evidence_r01.npz",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_ATTEMPT_R02_ROOT
    / "evaluation/endpoint_evaluation_plan_r02.json",
    dispatch_authorized: Annotated[bool, typer.Option("--dispatch-authorized")] = False,
) -> None:
    """Fail closed before the read-only paid G02 endpoint evaluation."""
    path = build_g02_endpoint_evaluation_plan(
        project_root=Path.cwd(),
        scientific_envelope_path=scientific_envelope,
        checkpoint_path=checkpoint,
        endpoint_evidence_path=endpoint_evidence,
        output_path=output,
        provider_rate_usd_per_hour=provider_rate_usd_per_hour,
        price_checked_at=price_checked_at,
        maximum_cost_usd=maximum_cost_usd,
        dispatch_authorized=dispatch_authorized,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "ready":
        raise typer.Exit(2)


@v2_app.command("audit-g02-endpoint-evaluation")
def v2_audit_g02_endpoint_evaluation(
    envelope: Annotated[Path, typer.Option("--envelope", exists=True, dir_okay=False)],
    plan: Annotated[
        Path, typer.Option("--plan", exists=True, dir_okay=False)
    ] = DEFAULT_G02_ATTEMPT_R02_ROOT / "evaluation/endpoint_evaluation_plan_r02.json",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_ATTEMPT_R02_ROOT
    / "evaluation/endpoint_evaluation_audit_r01.json",
) -> None:
    """Audit evaluator isolation separately from its scientific verdict."""
    path = audit_g02_endpoint_evaluation(envelope, plan, output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-g02-raw-topology")
def v2_audit_g02_raw_topology(
    raw_field: Annotated[
        Path, typer.Option("--raw-field", exists=True, dir_okay=False)
    ] = DEFAULT_G02_RECOVERY_R02_ROOT / "treatment/raw_canonical_field.npz",
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_G02_ATTEMPT_R02_ROOT
    / "evaluation/raw_topology_audit_r01.json",
) -> None:
    """Audit the unmodified frozen G02 zero set before exact COMMIT."""
    path = audit_g02_raw_topology(raw_field, output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("render-posed-preview")
def v2_render_posed_preview(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ],
    image_root: Annotated[Path, typer.Option("--image-root", exists=True, file_okay=False)],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root")],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/reconstruction/canonical_clothed_surface_v1.yaml"
    ),
    output_width: Annotated[int, typer.Option("--output-width", min=64)] = 288,
    fps: Annotated[float, typer.Option("--fps", min=1.0)] = 30.0,
) -> None:
    """Render the frozen V1 endpoint in its observed poses for diagnosis."""
    report_path = render_posed_preview(
        config=load_config(config_path),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        image_root=image_root,
        mask_root=mask_root,
        output_root=output_root,
        output_width=output_width,
        fps=fps,
    )
    _emit(read_json(report_path))


@v2_app.command("benchmark-g03-appearance")
def v2_benchmark_g03_appearance(
    output: Annotated[Path, typer.Option("--output")] = Path(
        "outputs/post_v2/postv2_g03_pose_stable_canonical_appearance_r01/"
        "registered-20260903-r01/qualification/public_appearance_benchmark.json"
    ),
    seed: Annotated[int, typer.Option(min=0)] = 20260903,
) -> None:
    """Qualify robust pose-stable appearance on public synthetic evidence."""
    report_path = write_g03_public_benchmark(output, seed=seed)
    report = read_json(report_path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("fit-g03-appearance")
def v2_fit_g03_appearance(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ],
    image_root: Annotated[Path, typer.Option("--image-root", exists=True, file_okay=False)],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "outputs/post_v2/postv2_g03_pose_stable_canonical_appearance_r01/"
        "registered-20260903-r01/qualification/train-only-fit-r01"
    ),
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/reconstruction/canonical_clothed_surface_v1.yaml"
    ),
    erosion_pixels: Annotated[int, typer.Option("--erosion-pixels", min=0)] = 3,
) -> None:
    """Fuse canonical appearance from the frozen 144-frame training split."""
    report_path = fit_g03_train_only_appearance(
        config=load_config(config_path),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        image_root=image_root,
        mask_root=mask_root,
        output_root=output_root,
        source_revision=source_revision,
        erosion_pixels=erosion_pixels,
    )
    _emit(read_json(report_path))


@v2_app.command("evaluate-g03-appearance")
def v2_evaluate_g03_appearance(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ],
    appearance: Annotated[Path, typer.Option("--appearance", exists=True, dir_okay=False)],
    fit_report: Annotated[Path, typer.Option("--fit-report", exists=True, dir_okay=False)],
    image_root: Annotated[Path, typer.Option("--image-root", exists=True, file_okay=False)],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root")],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    split: Annotated[Literal["train", "held_out"], typer.Option("--split")],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/reconstruction/canonical_clothed_surface_v1.yaml"
    ),
    output_width: Annotated[int, typer.Option("--output-width", min=64)] = 288,
    fps: Annotated[float, typer.Option("--fps", min=1.0)] = 30.0,
    shading_strength: Annotated[float, typer.Option("--shading-strength", min=0.0, max=1.0)] = 0.0,
) -> None:
    """Evaluate a frozen G03 appearance on one explicitly selected split."""
    report_path = evaluate_g03_appearance_split(
        config=load_config(config_path),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        appearance_path=appearance,
        fit_report_path=fit_report,
        image_root=image_root,
        mask_root=mask_root,
        output_root=output_root,
        source_revision=source_revision,
        split=split,
        output_width=output_width,
        fps=fps,
        shading_strength=shading_strength,
    )
    report = read_json(report_path)
    _emit(report)
    if report["status"] == "fail":
        raise typer.Exit(2)


@v2_app.command("benchmark-g04-phase-appearance")
def v2_benchmark_g04_phase_appearance(
    output: Annotated[Path, typer.Option("--output")] = Path(
        "outputs/post_v2/postv2_g04_phase_conditioned_canonical_appearance_r01/"
        "registered-20260903-r01/qualification/public_phase_appearance_benchmark.json"
    ),
    seed: Annotated[int, typer.Option(min=0)] = 20260903,
) -> None:
    """Qualify phase-conditioned appearance on public synthetic evidence."""
    report_path = write_g04_public_benchmark(output, seed=seed)
    report = read_json(report_path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("fit-g04-phase-appearance")
def v2_fit_g04_phase_appearance(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ],
    image_root: Annotated[Path, typer.Option("--image-root", exists=True, file_okay=False)],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "outputs/post_v2/postv2_g04_phase_conditioned_canonical_appearance_r01/"
        "registered-20260903-r01/qualification/train-only-fit-r01"
    ),
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/reconstruction/canonical_clothed_surface_v1.yaml"
    ),
    erosion_pixels: Annotated[int, typer.Option("--erosion-pixels", min=0)] = 3,
) -> None:
    """Fit phase-indexed canonical appearance from training RGB only."""
    report_path = fit_g04_train_only_phase_appearance(
        config=load_config(config_path),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        image_root=image_root,
        mask_root=mask_root,
        output_root=output_root,
        source_revision=source_revision,
        erosion_pixels=erosion_pixels,
    )
    _emit(read_json(report_path))


@v2_app.command("benchmark-d01-pose-normal-depth")
def v2_benchmark_d01_pose_normal_depth(
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D01_OUTPUT_ROOT / "public_pose_normal_depth_benchmark.json"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 20260903,
) -> None:
    """Qualify pose-stabilized normal fusion on public analytic evidence."""

    report_path = write_d01_public_benchmark(output, seed=seed)
    report = read_json(report_path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("bind-d01-train-normals")
def v2_bind_d01_train_normals(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/runs/"
        "important_canonical_clothed_surface_v1_r4m_motion_bounded12_0ebff81c68a7/"
        "motion_coadaptation_bounded_checkpoint.pt"
    ),
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/dataset_manifest.json"
    ),
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
        "initialization_artifacts/smpl_joint_transforms.npz"
    ),
    t05_solution: Annotated[
        Path, typer.Option("--t05-solution", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "fixed_camera_human_solution.json",
    q03_binding: Annotated[
        Path, typer.Option("--q03-binding", exists=True, dir_okay=False)
    ] = DEFAULT_Q03_OUTPUT_ROOT / "interval_track_qualification.json",
    normal_root: Annotated[
        Path, typer.Option("--normal-root", exists=True, file_okay=False)
    ] = Path("outputs/canonical_clothed_surface_v1/dataset/normals"),
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/masks"
    ),
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_D01_OUTPUT_ROOT / "train-normal-binding-r01"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
) -> None:
    """Bind only training normal/mask evidence through the frozen human pose."""

    settings = load_config(config)
    revision = source_revision or resolve_packaged_source_revision(Path.cwd())
    report_path = bind_d01_train_only_normal_evidence(
        config=settings,
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        t05_solution_path=t05_solution,
        q03_binding_path=q03_binding,
        normal_root=normal_root,
        mask_root=mask_root,
        output_root=output_root,
        source_revision=revision,
    )
    _emit(read_json(report_path))


@v2_app.command("plan-d01-train-candidate")
def v2_plan_d01_train_candidate(
    evidence_report: Annotated[
        Path, typer.Option("--evidence-report", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-normal-binding-r01/train_normal_binding_report.json",
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D01_OUTPUT_ROOT / "train-candidate-plan-r01.json"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
) -> None:
    """Freeze D01's bounded train candidate and evaluator gates."""

    revision = source_revision or resolve_packaged_source_revision(Path.cwd())
    path = write_d01_train_candidate_plan(
        evidence_report,
        output,
        source_revision=revision,
    )
    _emit(read_json(path))


@v2_app.command("fit-d01-train-candidate")
def v2_fit_d01_train_candidate(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/runs/"
        "important_canonical_clothed_surface_v1_r4m_motion_bounded12_0ebff81c68a7/"
        "motion_coadaptation_bounded_checkpoint.pt"
    ),
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/dataset_manifest.json"
    ),
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
        "initialization_artifacts/smpl_joint_transforms.npz"
    ),
    evidence_binding: Annotated[
        Path, typer.Option("--evidence-binding", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-normal-binding-r01/train_pose_stabilized_normals.npz",
    evidence_report: Annotated[
        Path, typer.Option("--evidence-report", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-normal-binding-r01/train_normal_binding_report.json",
    candidate_plan: Annotated[
        Path, typer.Option("--candidate-plan", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-candidate-plan-r01.json",
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_D01_OUTPUT_ROOT / "train-mesh-candidate-r01"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
) -> None:
    """Fit the one frozen bounded train-only D01 mesh candidate."""

    revision = source_revision or resolve_packaged_source_revision(Path.cwd())
    path = fit_d01_train_only_mesh_candidate(
        config=load_config(config),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        evidence_binding_path=evidence_binding,
        evidence_report_path=evidence_report,
        candidate_plan_path=candidate_plan,
        output_root=output_root,
        source_revision=revision,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "candidate_complete":
        raise typer.Exit(2)


@v2_app.command("evaluate-d01-train-candidate")
def v2_evaluate_d01_train_candidate(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/runs/"
        "important_canonical_clothed_surface_v1_r4m_motion_bounded12_0ebff81c68a7/"
        "motion_coadaptation_bounded_checkpoint.pt"
    ),
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/dataset_manifest.json"
    ),
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
        "initialization_artifacts/smpl_joint_transforms.npz"
    ),
    t05_solution: Annotated[
        Path, typer.Option("--t05-solution", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "fixed_camera_human_solution.json",
    candidate: Annotated[Path, typer.Option("--candidate", exists=True, dir_okay=False)] = (
        DEFAULT_D01_OUTPUT_ROOT / "train-mesh-candidate-r01/bounded_canonical_mesh_candidate.npz"
    ),
    candidate_report: Annotated[
        Path, typer.Option("--candidate-report", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-mesh-candidate-r01/train_mesh_candidate_report.json",
    candidate_plan: Annotated[
        Path, typer.Option("--candidate-plan", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-candidate-plan-r01.json",
    normal_root: Annotated[
        Path, typer.Option("--normal-root", exists=True, file_okay=False)
    ] = Path("outputs/canonical_clothed_surface_v1/dataset/normals"),
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/masks"
    ),
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D01_OUTPUT_ROOT / "train-mesh-candidate-r01/train_candidate_evaluation.json"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
) -> None:
    """Score D01's frozen candidate against V1 on all training records."""

    revision = source_revision or resolve_packaged_source_revision(Path.cwd())
    path = evaluate_d01_train_mesh_candidate(
        config=load_config(config),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        t05_solution_path=t05_solution,
        candidate_path=candidate,
        candidate_report_path=candidate_report,
        candidate_plan_path=candidate_plan,
        normal_root=normal_root,
        mask_root=mask_root,
        output_path=output,
        source_revision=revision,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-d01-terminal")
def v2_audit_d01_terminal(
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "public_pose_normal_depth_benchmark.json",
    evidence_report: Annotated[
        Path, typer.Option("--evidence-report", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-normal-binding-r01/train_normal_binding_report.json",
    candidate_plan: Annotated[
        Path, typer.Option("--candidate-plan", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-candidate-plan-r01.json",
    candidate_report: Annotated[
        Path, typer.Option("--candidate-report", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-mesh-candidate-r01/train_mesh_candidate_report.json",
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D01_OUTPUT_ROOT / "terminal_qualification.json"
    ),
) -> None:
    """Audit and close the frozen D01 qualification route."""

    path = audit_d01_terminal_qualification(
        public_benchmark,
        evidence_report,
        candidate_plan,
        candidate_report,
        output,
    )
    _emit(read_json(path))


@v2_app.command("benchmark-d02-topology-projection")
def v2_benchmark_d02_topology_projection(
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D02_OUTPUT_ROOT / "public_topology_constrained_projection.json"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 20260903,
) -> None:
    """Qualify D02's local topology trust projection on public evidence."""

    path = write_d02_public_benchmark(output, seed=seed)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-d02-train-projection")
def v2_plan_d02_train_projection(
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ] = DEFAULT_D02_OUTPUT_ROOT / "public_topology_constrained_projection.json",
    d01_terminal: Annotated[
        Path, typer.Option("--d01-terminal", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "terminal_qualification.json",
    d01_evidence_report: Annotated[
        Path, typer.Option("--d01-evidence-report", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-normal-binding-r01/train_normal_binding_report.json",
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D02_OUTPUT_ROOT / "train-projection-plan-r01.json"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
) -> None:
    """Freeze D02's real topology projection and training gates."""

    revision = source_revision or resolve_packaged_source_revision(Path.cwd())
    path = write_d02_train_projection_plan(
        public_benchmark,
        d01_terminal,
        d01_evidence_report,
        output,
        source_revision=revision,
    )
    _emit(read_json(path))


@v2_app.command("fit-d02-train-projection")
def v2_fit_d02_train_projection(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/runs/"
        "important_canonical_clothed_surface_v1_r4m_motion_bounded12_0ebff81c68a7/"
        "motion_coadaptation_bounded_checkpoint.pt"
    ),
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/dataset_manifest.json"
    ),
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
        "initialization_artifacts/smpl_joint_transforms.npz"
    ),
    d01_raw_candidate: Annotated[
        Path, typer.Option("--d01-raw-candidate", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-mesh-candidate-r01/bounded_canonical_mesh_candidate.npz",
    d01_candidate_report: Annotated[
        Path, typer.Option("--d01-candidate-report", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "train-mesh-candidate-r01/train_mesh_candidate_report.json",
    d01_terminal: Annotated[
        Path, typer.Option("--d01-terminal", exists=True, dir_okay=False)
    ] = DEFAULT_D01_OUTPUT_ROOT / "terminal_qualification.json",
    projection_plan: Annotated[
        Path, typer.Option("--projection-plan", exists=True, dir_okay=False)
    ] = DEFAULT_D02_OUTPUT_ROOT / "train-projection-plan-r01.json",
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_D02_OUTPUT_ROOT / "train-projection-candidate-r01"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
) -> None:
    """Project D01's frozen raw proposal through D02's topology constraints."""

    revision = source_revision or resolve_packaged_source_revision(Path.cwd())
    path = fit_d02_train_topology_projection(
        config=load_config(config),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        d01_raw_candidate_path=d01_raw_candidate,
        d01_candidate_report_path=d01_candidate_report,
        d01_terminal_path=d01_terminal,
        projection_plan_path=projection_plan,
        output_root=output_root,
        source_revision=revision,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "candidate_complete":
        raise typer.Exit(2)


@v2_app.command("evaluate-d02-train-candidate")
def v2_evaluate_d02_train_candidate(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/runs/"
        "important_canonical_clothed_surface_v1_r4m_motion_bounded12_0ebff81c68a7/"
        "motion_coadaptation_bounded_checkpoint.pt"
    ),
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/dataset_manifest.json"
    ),
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
        "initialization_artifacts/smpl_joint_transforms.npz"
    ),
    t05_solution: Annotated[
        Path, typer.Option("--t05-solution", exists=True, dir_okay=False)
    ] = DEFAULT_T05_OUTPUT_ROOT / "fixed_camera_human_solution.json",
    candidate: Annotated[Path, typer.Option("--candidate", exists=True, dir_okay=False)] = (
        DEFAULT_D02_OUTPUT_ROOT
        / "train-projection-candidate-r01/topology_constrained_canonical_candidate.npz"
    ),
    candidate_report: Annotated[
        Path, typer.Option("--candidate-report", exists=True, dir_okay=False)
    ] = (
        DEFAULT_D02_OUTPUT_ROOT
        / "train-projection-candidate-r01/train_projection_candidate_report.json"
    ),
    projection_plan: Annotated[
        Path, typer.Option("--projection-plan", exists=True, dir_okay=False)
    ] = DEFAULT_D02_OUTPUT_ROOT / "train-projection-plan-r01.json",
    normal_root: Annotated[
        Path, typer.Option("--normal-root", exists=True, file_okay=False)
    ] = Path("outputs/canonical_clothed_surface_v1/dataset/normals"),
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/masks"
    ),
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D02_OUTPUT_ROOT / "train-projection-candidate-r01/train_candidate_evaluation.json"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
) -> None:
    """Score the frozen D02 candidate on training records only."""

    revision = source_revision or resolve_packaged_source_revision(Path.cwd())
    path = evaluate_d02_train_candidate(
        config=load_config(config),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        t05_solution_path=t05_solution,
        candidate_path=candidate,
        candidate_report_path=candidate_report,
        projection_plan_path=projection_plan,
        normal_root=normal_root,
        mask_root=mask_root,
        output_path=output,
        source_revision=revision,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-d02-exact-topology")
def v2_audit_d02_exact_topology(
    config: ConfigOption = Path("configs/reconstruction/canonical_clothed_surface_v1.yaml"),
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/runs/"
        "important_canonical_clothed_surface_v1_r4m_motion_bounded12_0ebff81c68a7/"
        "motion_coadaptation_bounded_checkpoint.pt"
    ),
    candidate: Annotated[Path, typer.Option("--candidate", exists=True, dir_okay=False)] = (
        DEFAULT_D02_OUTPUT_ROOT
        / "train-projection-candidate-r01/topology_constrained_canonical_candidate.npz"
    ),
    candidate_report: Annotated[
        Path, typer.Option("--candidate-report", exists=True, dir_okay=False)
    ] = (
        DEFAULT_D02_OUTPUT_ROOT
        / "train-projection-candidate-r01/train_projection_candidate_report.json"
    ),
    train_evaluation: Annotated[
        Path, typer.Option("--train-evaluation", exists=True, dir_okay=False)
    ] = (
        DEFAULT_D02_OUTPUT_ROOT / "train-projection-candidate-r01/train_candidate_evaluation.json"
    ),
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D02_OUTPUT_ROOT / "exact_topology_audit.json"
    ),
) -> None:
    """Run D02's exact collision and closed-body topology gate."""

    path = audit_d02_exact_topology(
        config=load_config(config),
        checkpoint_path=checkpoint,
        candidate_path=candidate,
        candidate_report_path=candidate_report,
        train_evaluation_path=train_evaluation,
        output_path=output,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("benchmark-d03-capsule-tree")
def v2_benchmark_d03_capsule_tree(
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D03_OUTPUT_ROOT / "public_capsule_tree_implicit_body.json"
    ),
) -> None:
    """Qualify D03's new embedded implicit-body topology lineage."""

    path = write_d03_public_benchmark(output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-d03-real-initialization")
def v2_plan_d03_real_initialization(
    scaffold_mesh: Annotated[Path, typer.Option("--scaffold-mesh", exists=True)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
        "shared_smpl_canonical.npz"
    ),
    skinning_weights: Annotated[Path, typer.Option("--skinning-weights", exists=True)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
        "smpl_skinning_weights.npz"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D03_OUTPUT_ROOT / "real-initialization-plan-r01.json"
    ),
) -> None:
    """Freeze D03's real-sequence capsule field before surface construction."""

    path = write_d03_real_initialization_plan(
        scaffold_mesh_path=scaffold_mesh,
        skinning_weights_path=skinning_weights,
        source_revision=source_revision,
        output_path=output,
    )
    _emit(read_json(path))


@v2_app.command("build-d03-real-initialization")
def v2_build_d03_real_initialization(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "real-initialization-plan-r01.json"
    ),
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01"
    ),
) -> None:
    """Construct D03's frozen real field and run the exact topology gate."""

    report_path = build_d03_real_initialization(plan_path=plan, output_root=output_root)
    report = read_json(report_path)
    _emit(report)
    if report["status"] != "initial_field_qualified":
        raise typer.Exit(2)


@v2_app.command("plan-d03-train-evidence")
def v2_plan_d03_train_evidence(
    real_initialization_report: Annotated[
        Path, typer.Option("--real-initialization-report", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01/real_initialization_report.json",
    real_mesh: Annotated[Path, typer.Option("--real-mesh", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01/canonical_capsule_mesh.npz"
    ),
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/dataset_manifest.json"
    ),
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/initialization_artifacts/"
        "smpl_joint_transforms.npz"
    ),
    t05_solution: Annotated[
        Path, typer.Option("--t05-solution", exists=True, dir_okay=False)
    ] = Path(
        "outputs/post_v2/postv2_t05_background_anchored_fixed_camera_human_ba_r01/"
        "registered-20260903-r01/qualification/fixed_camera_human_solution.json"
    ),
    normal_root: Annotated[Path, typer.Option("--normal-root", exists=True, file_okay=False)] = (
        Path("outputs/canonical_clothed_surface_v1/dataset/normals")
    ),
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)] = Path(
        "outputs/canonical_clothed_surface_v1/dataset/masks"
    ),
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D03_OUTPUT_ROOT / "train-evidence-plan-r01.json"
    ),
) -> None:
    """Freeze D03's all-and-only-training normal/mask evidence transfer."""

    path = write_d03_train_evidence_plan(
        real_initialization_report_path=real_initialization_report,
        real_mesh_path=real_mesh,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        t05_solution_path=t05_solution,
        normal_root=normal_root,
        mask_root=mask_root,
        source_revision=source_revision,
        output_path=output,
    )
    _emit(read_json(path))


@v2_app.command("bind-d03-train-normal-evidence")
def v2_bind_d03_train_normal_evidence(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "train-evidence-plan-r01.json"
    ),
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_D03_OUTPUT_ROOT / "train-normal-binding-r01"
    ),
) -> None:
    """Transfer frozen train-only normals to D03's embedded surface."""

    report_path = bind_d03_train_normal_evidence(plan_path=plan, output_root=output_root)
    report = read_json(report_path)
    _emit(report)
    if report["status"] != "train_evidence_bound":
        raise typer.Exit(2)


@v2_app.command("plan-d03-implicit-continuation")
def v2_plan_d03_implicit_continuation(
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "public_capsule_tree_implicit_body.json",
    real_initialization_report: Annotated[
        Path, typer.Option("--real-initialization-report", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01/real_initialization_report.json",
    real_field: Annotated[Path, typer.Option("--real-field", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01/canonical_capsule_field.npz"
    ),
    real_mesh: Annotated[Path, typer.Option("--real-mesh", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01/canonical_capsule_mesh.npz"
    ),
    train_evidence_report: Annotated[
        Path, typer.Option("--train-evidence-report", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "train-normal-binding-r01/train_normal_binding_report.json",
    train_evidence: Annotated[
        Path, typer.Option("--train-evidence", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "train-normal-binding-r01/train_pose_stabilized_normals.npz",
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D03_OUTPUT_ROOT / "implicit-continuation-plan-r01.json"
    ),
) -> None:
    """Freeze one topology-certified train-only D03 field continuation."""

    path = write_d03_implicit_continuation_plan(
        public_benchmark_path=public_benchmark,
        real_initialization_report_path=real_initialization_report,
        real_field_path=real_field,
        real_mesh_path=real_mesh,
        train_evidence_report_path=train_evidence_report,
        train_evidence_path=train_evidence,
        source_revision=source_revision,
        output_path=output,
    )
    _emit(read_json(path))


@v2_app.command("build-d03-implicit-continuation")
def v2_build_d03_implicit_continuation(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "implicit-continuation-plan-r01.json"
    ),
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_D03_OUTPUT_ROOT / "implicit-continuation-r01"
    ),
) -> None:
    """Build and exactly audit D03's single frozen implicit continuation."""

    report_path = build_d03_implicit_continuation(plan_path=plan, output_root=output_root)
    report = read_json(report_path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-d03-train-evaluation")
def v2_plan_d03_train_evaluation(
    real_initialization_report: Annotated[
        Path, typer.Option("--real-initialization-report", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01/real_initialization_report.json",
    initial_mesh: Annotated[Path, typer.Option("--initial-mesh", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01/canonical_capsule_mesh.npz"
    ),
    continuation_report: Annotated[
        Path, typer.Option("--continuation-report", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "implicit-continuation-r01/implicit_continuation_report.json",
    candidate_mesh: Annotated[
        Path, typer.Option("--candidate-mesh", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "implicit-continuation-r01/continued_canonical_mesh.npz",
    train_evidence_plan: Annotated[
        Path, typer.Option("--train-evidence-plan", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "train-evidence-plan-r01.json",
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D03_OUTPUT_ROOT / "train-evaluation-plan-r01.json"
    ),
) -> None:
    """Freeze D03's matched train-only image evaluator and its gates."""

    path = write_d03_train_evaluation_plan(
        real_initialization_report_path=real_initialization_report,
        initial_mesh_path=initial_mesh,
        continuation_report_path=continuation_report,
        candidate_mesh_path=candidate_mesh,
        train_evidence_plan_path=train_evidence_plan,
        source_revision=source_revision,
        output_path=output,
    )
    _emit(read_json(path))


@v2_app.command("evaluate-d03-train-images")
def v2_evaluate_d03_train_images(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "train-evaluation-plan-r01.json"
    ),
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D03_OUTPUT_ROOT / "train-image-evaluation-r01.json"
    ),
) -> None:
    """Evaluate D03's continued field on the frozen 144 training images."""

    report_path = evaluate_d03_train_images(plan_path=plan, output_path=output)
    report = read_json(report_path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-d03-development-evaluation")
def v2_plan_d03_development_evaluation(
    train_evaluation: Annotated[
        Path, typer.Option("--train-evaluation", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "train-image-evaluation-r01.json",
    continuation_report: Annotated[
        Path, typer.Option("--continuation-report", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "implicit-continuation-r01/implicit_continuation_report.json",
    initial_mesh: Annotated[Path, typer.Option("--initial-mesh", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "real-initialization-r01/canonical_capsule_mesh.npz"
    ),
    candidate_mesh: Annotated[
        Path, typer.Option("--candidate-mesh", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "implicit-continuation-r01/continued_canonical_mesh.npz",
    train_evidence_plan: Annotated[
        Path, typer.Option("--train-evidence-plan", exists=True, dir_okay=False)
    ] = DEFAULT_D03_OUTPUT_ROOT / "train-evidence-plan-r01.json",
    source_revision: Annotated[str, typer.Option("--source-revision")] = "",
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D03_OUTPUT_ROOT / "development-evaluation-plan-r01.json"
    ),
) -> None:
    """Freeze D03's single 36-frame development evaluation."""

    path = write_d03_development_evaluation_plan(
        train_evaluation_path=train_evaluation,
        continuation_report_path=continuation_report,
        initial_mesh_path=initial_mesh,
        candidate_mesh_path=candidate_mesh,
        train_evidence_plan_path=train_evidence_plan,
        source_revision=source_revision,
        output_path=output,
    )
    _emit(read_json(path))


@v2_app.command("evaluate-d03-development")
def v2_evaluate_d03_development(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_D03_OUTPUT_ROOT / "development-evaluation-plan-r01.json"
    ),
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_D03_OUTPUT_ROOT / "development-evaluation-r01.json"
    ),
) -> None:
    """Run D03's single frozen 36-frame development evaluation."""

    report_path = evaluate_d03_development_images(plan_path=plan, output_path=output)
    report = read_json(report_path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("benchmark-l03-open-layers")
def v2_benchmark_l03_open_layers(
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_L03_OUTPUT_ROOT / "public_open_layers.json"
    ),
) -> None:
    """Qualify L03 open boundaries, ordering, contact, and rejection controls."""

    path = write_l03_public_benchmark(output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-l03-semantic-support")
def v2_plan_l03_semantic_support(
    public_report: Annotated[Path, typer.Option("--public-report", exists=True, dir_okay=False)],
    semantic_inputs: Annotated[
        Path, typer.Option("--semantic-inputs", exists=True, dir_okay=False)
    ],
    t05_solution: Annotated[Path, typer.Option("--t05-solution", exists=True, dir_okay=False)],
    s01_qualification: Annotated[
        Path, typer.Option("--s01-qualification", exists=True, dir_okay=False)
    ],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_L03_OUTPUT_ROOT / "real-semantic-support-plan-r01.json"
    ),
) -> None:
    """Freeze L03's train-only semantic-support audit inputs and gates."""

    path = write_l03_semantic_support_plan(
        public_report_path=public_report,
        semantic_inputs_path=semantic_inputs,
        t05_solution_path=t05_solution,
        s01_qualification_path=s01_qualification,
        source_revision=source_revision,
        output_path=output,
    )
    _emit(read_json(path))


@v2_app.command("audit-l03-semantic-support")
def v2_audit_l03_semantic_support(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_L03_OUTPUT_ROOT / "real-semantic-support-plan-r01.json"
    ),
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_L03_OUTPUT_ROOT / "real-semantic-support-r01.json"
    ),
) -> None:
    """Audit whether train-only semantics justify real upper/lower layers."""

    path = audit_l03_semantic_support(plan_path=plan, output_path=output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-l03-real-initialization")
def v2_plan_l03_real_initialization(
    semantic_support_report: Annotated[
        Path, typer.Option("--semantic-support-report", exists=True, dir_okay=False)
    ],
    d03_report: Annotated[Path, typer.Option("--d03-report", exists=True, dir_okay=False)],
    d03_field: Annotated[Path, typer.Option("--d03-field", exists=True, dir_okay=False)],
    d03_mesh: Annotated[Path, typer.Option("--d03-mesh", exists=True, dir_okay=False)],
    hull_qualification: Annotated[
        Path, typer.Option("--hull-qualification", exists=True, dir_okay=False)
    ],
    semantic_volume: Annotated[
        Path, typer.Option("--semantic-volume", exists=True, dir_okay=False)
    ],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_L03_OUTPUT_ROOT / "real-initialization-plan-r01.json"
    ),
) -> None:
    """Freeze L03's real semantic offset-surface initialization."""

    path = write_l03_real_initialization_plan(
        semantic_support_report_path=semantic_support_report,
        d03_report_path=d03_report,
        d03_field_path=d03_field,
        d03_mesh_path=d03_mesh,
        hull_qualification_path=hull_qualification,
        semantic_volume_path=semantic_volume,
        source_revision=source_revision,
        output_path=output,
    )
    _emit(read_json(path))


@v2_app.command("build-l03-real-initialization")
def v2_build_l03_real_initialization(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_L03_OUTPUT_ROOT / "real-initialization-plan-r01.json"
    ),
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_L03_OUTPUT_ROOT / "real-initialization-r01"
    ),
) -> None:
    """Build and exactly audit L03's real open garment initialization."""

    path = build_l03_real_initialization(plan_path=plan, output_root=output_root)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("benchmark-l03-training")
def v2_benchmark_l03_training(
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_L03_OUTPUT_ROOT / "public_training_benchmark.json"
    ),
) -> None:
    """Qualify fixed-topology bounded outward layer displacement publicly."""

    path = write_l03_training_public_benchmark(output)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-l03-training-qualification")
def v2_plan_l03_training_qualification(
    public_training_report: Annotated[
        Path, typer.Option("--public-training-report", exists=True, dir_okay=False)
    ],
    initialization_report: Annotated[
        Path, typer.Option("--initialization-report", exists=True, dir_okay=False)
    ],
    upper_layer: Annotated[Path, typer.Option("--upper-layer", exists=True, dir_okay=False)],
    lower_layer: Annotated[Path, typer.Option("--lower-layer", exists=True, dir_okay=False)],
    semantic_support_plan: Annotated[
        Path, typer.Option("--semantic-support-plan", exists=True, dir_okay=False)
    ],
    semantic_inputs: Annotated[
        Path, typer.Option("--semantic-inputs", exists=True, dir_okay=False)
    ],
    d03_train_evidence_plan: Annotated[
        Path, typer.Option("--d03-train-evidence-plan", exists=True, dir_okay=False)
    ],
    d03_mesh: Annotated[Path, typer.Option("--d03-mesh", exists=True, dir_okay=False)],
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ],
    t05_solution: Annotated[Path, typer.Option("--t05-solution", exists=True, dir_okay=False)],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_L03_OUTPUT_ROOT / "training-qualification-plan-r01.json"
    ),
) -> None:
    """Freeze L03's real four-phase one-step engineering qualification."""

    path = write_l03_training_qualification_plan(
        public_training_report_path=public_training_report,
        initialization_report_path=initialization_report,
        upper_layer_path=upper_layer,
        lower_layer_path=lower_layer,
        semantic_support_plan_path=semantic_support_plan,
        semantic_inputs_path=semantic_inputs,
        d03_train_evidence_plan_path=d03_train_evidence_plan,
        d03_mesh_path=d03_mesh,
        joint_transforms_path=joint_transforms,
        t05_solution_path=t05_solution,
        source_revision=source_revision,
        output_path=output,
    )
    _emit(read_json(path))


@v2_app.command("qualify-l03-training-local")
def v2_qualify_l03_training_local(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_L03_OUTPUT_ROOT / "training-qualification-plan-r01.json"
    ),
    output_root: Annotated[Path, typer.Option("--output-root")] = (
        DEFAULT_L03_OUTPUT_ROOT / "local-training-qualification-r01"
    ),
    device: Annotated[str, typer.Option("--device")] = "mps",
) -> None:
    """Run L03's one-step local device and checkpoint qualification."""

    path = run_l03_local_training_qualification(
        plan_path=plan, output_root=output_root, device=device
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-l03-cuda-qualification")
def v2_plan_l03_cuda_qualification(
    contract: Annotated[Path, typer.Option("--contract", exists=True, dir_okay=False)] = Path(
        "configs/evaluation/post_v2_l03_semantic_open_clothing_layers_r01.yaml"
    ),
    local_qualification: Annotated[
        Path, typer.Option("--local-qualification", exists=True, dir_okay=False)
    ] = DEFAULT_L03_OUTPUT_ROOT
    / "local-training-qualification-r01/local_training_qualification.json",
    training_plan: Annotated[
        Path, typer.Option("--training-plan", exists=True, dir_okay=False)
    ] = DEFAULT_L03_OUTPUT_ROOT / "training-qualification-plan-r01.json",
    provider_rate_usd_per_hour: Annotated[
        float | None, typer.Option("--provider-rate-usd-per-hour")
    ] = None,
    price_checked_at: Annotated[str | None, typer.Option("--price-checked-at")] = None,
    maximum_cost_usd: Annotated[float | None, typer.Option("--maximum-cost-usd")] = None,
    dispatch_authorized: Annotated[bool, typer.Option("--dispatch-authorized")] = False,
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_L03_OUTPUT_ROOT / "target-cuda-plan-r01.json"
    ),
) -> None:
    """Build L03's fail-closed target-CUDA qualification plan."""

    path = build_l03_cuda_qualification_plan(
        project_root=Path.cwd(),
        contract_path=contract,
        local_qualification_path=local_qualification,
        training_qualification_plan_path=training_plan,
        output_path=output,
        provider_rate_usd_per_hour=provider_rate_usd_per_hour,
        price_checked_at=price_checked_at,
        maximum_cost_usd=maximum_cost_usd,
        dispatch_authorized=dispatch_authorized,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "ready":
        raise typer.Exit(2)


@v2_app.command("audit-l03-cuda-qualification")
def v2_audit_l03_cuda_qualification(
    envelope: Annotated[Path, typer.Option("--envelope", exists=True, dir_okay=False)] = (
        DEFAULT_L03_OUTPUT_ROOT / "target-cuda-envelope-r01.json"
    ),
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)] = (
        DEFAULT_L03_OUTPUT_ROOT / "target-cuda-plan-r01.json"
    ),
    output: Annotated[Path, typer.Option("--output")] = (
        DEFAULT_L03_OUTPUT_ROOT / "target-cuda-audit-r01.json"
    ),
) -> None:
    """Audit L03's target-CUDA envelope and advance only on exact binding."""

    path = audit_l03_target_cuda_qualification(
        envelope_path=envelope, plan_path=plan, output_path=output
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("evaluate-g04-phase-appearance")
def v2_evaluate_g04_phase_appearance(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ],
    appearance: Annotated[Path, typer.Option("--appearance", exists=True, dir_okay=False)],
    fit_report: Annotated[Path, typer.Option("--fit-report", exists=True, dir_okay=False)],
    image_root: Annotated[Path, typer.Option("--image-root", exists=True, file_okay=False)],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root")],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    split: Annotated[Literal["train", "held_out"], typer.Option("--split")],
    bandwidth: Annotated[float, typer.Option("--bandwidth", min=0.1)] = 25.0,
    prior_weight: Annotated[float, typer.Option("--prior-weight", min=0.001)] = 0.25,
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/reconstruction/canonical_clothed_surface_v1.yaml"
    ),
    output_width: Annotated[int, typer.Option("--output-width", min=64)] = 288,
    fps: Annotated[float, typer.Option("--fps", min=1.0)] = 30.0,
) -> None:
    """Evaluate G04 with train leave-one-out or frozen development scoring."""
    report_path = evaluate_g04_phase_appearance_split(
        config=load_config(config_path),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        appearance_path=appearance,
        fit_report_path=fit_report,
        image_root=image_root,
        mask_root=mask_root,
        output_root=output_root,
        source_revision=source_revision,
        split=split,
        bandwidth=bandwidth,
        prior_weight=prior_weight,
        output_width=output_width,
        fps=fps,
    )
    report = read_json(report_path)
    _emit(report)
    if report["status"] == "fail":
        raise typer.Exit(2)


@v2_app.command("render-g04-full-sequence")
def v2_render_g04_full_sequence(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    joint_transforms: Annotated[
        Path, typer.Option("--joint-transforms", exists=True, dir_okay=False)
    ],
    appearance: Annotated[Path, typer.Option("--appearance", exists=True, dir_okay=False)],
    fit_report: Annotated[Path, typer.Option("--fit-report", exists=True, dir_okay=False)],
    image_root: Annotated[Path, typer.Option("--image-root", exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root")],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    bandwidth: Annotated[float, typer.Option("--bandwidth", min=0.1)] = 12.5,
    prior_weight: Annotated[float, typer.Option("--prior-weight", min=0.001)] = 0.25,
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/reconstruction/canonical_clothed_surface_v1.yaml"
    ),
    output_width: Annotated[int, typer.Option("--output-width", min=64)] = 288,
    fps: Annotated[float, typer.Option("--fps", min=1.0)] = 30.0,
    blind_seed: Annotated[int, typer.Option("--blind-seed", min=0)] = 20260903,
) -> None:
    """Render the frozen G04 model across all 180 chronological records."""
    report_path = render_g04_full_sequence(
        config=load_config(config_path),
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        joint_transforms_path=joint_transforms,
        appearance_path=appearance,
        fit_report_path=fit_report,
        image_root=image_root,
        output_root=output_root,
        source_revision=source_revision,
        bandwidth=bandwidth,
        prior_weight=prior_weight,
        output_width=output_width,
        fps=fps,
        blind_seed=blind_seed,
    )
    _emit(read_json(report_path))


@v2_app.command("benchmark")
def v2_benchmark(
    output: Annotated[Path, typer.Option("--output")] = Path(
        "outputs/post_v2/qualification/public_benchmark.json"
    ),
    seed: Annotated[int, typer.Option(min=0)] = 20260902,
) -> None:
    """Run the independent public/synthetic V2 geometry scorecard."""
    reject_sealed_capability([output])
    path = write_public_benchmark(output, seed=seed)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("fit-turntable")
def v2_fit_turntable(
    initialization: Annotated[
        Path, typer.Option("--initialization", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    micromotion_rank: Annotated[int, typer.Option(min=0, max=16)] = 4,
) -> None:
    """Write a qualification candidate; this does not claim the T01 scientific attempt."""
    reject_sealed_capability([initialization, output])
    solution = initialize_turntable_solution(
        initialization,
        micromotion_rank=micromotion_rank,
    )
    write_json(output, solution)
    _emit(solution)


@v2_app.command("benchmark-turntable-ba")
def v2_benchmark_turntable_ba(
    output: Annotated[Path, typer.Option("--output")],
    seed: Annotated[int, typer.Option(min=0)] = 20260902,
) -> None:
    """Run the public reduced-BA recovery and identifiability gates on the Mac."""
    path = write_turntable_ba_benchmark(output, seed=seed)
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("initialize-cooperative-turntable")
def v2_initialize_cooperative_turntable(
    initialization: Annotated[
        Path, typer.Option("--initialization", exists=True, dir_okay=False, readable=True)
    ],
    manifest: Annotated[
        Path, typer.Option("--manifest", exists=True, dir_okay=False, readable=True)
    ],
    validation: Annotated[
        Path, typer.Option("--validation", exists=True, dir_okay=False, readable=True)
    ],
    track_graph_report: Annotated[
        Path, typer.Option("--track-graph-report", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    micromotion_rank: Annotated[int, typer.Option(min=0, max=16)] = 4,
) -> None:
    """Build the Q01-bound full-turn initializer; this is not a T01 fit."""
    solution = initialize_cooperative_turntable_solution(
        initialization,
        manifest,
        validation,
        track_graph_report,
        micromotion_rank=micromotion_rank,
    )
    write_json(output, solution)
    _emit(solution)


@v2_app.command("diagnose-turntable-factors")
def v2_diagnose_turntable_factors(
    solution: Annotated[
        Path, typer.Option("--solution", exists=True, dir_okay=False, readable=True)
    ],
    factors: Annotated[Path, typer.Option("--factors", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    image_height: Annotated[int, typer.Option(min=1)] = 1120,
    image_width: Annotated[int, typer.Option(min=1)] = 720,
) -> None:
    """Backpropagate Q01 factors through T01 once without updating parameters."""
    path = diagnose_real_turntable_factor_route(
        solution,
        factors,
        output,
        image_size=(image_height, image_width),
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-turntable-phase")
def v2_qualify_turntable_phase(
    solution: Annotated[
        Path, typer.Option("--solution", exists=True, dir_okay=False, readable=True)
    ],
    factors: Annotated[Path, typer.Option("--factors", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    image_height: Annotated[int, typer.Option(min=1)] = 1120,
    image_width: Annotated[int, typer.Option(min=1)] = 720,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Take and exactly replay one bounded Mac-CPU phase/axis qualification step."""
    path = qualify_real_phase_axis_step(
        solution,
        factors,
        output,
        checkpoint,
        image_size=(image_height, image_width),
        device=device,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-turntable-silhouette")
def v2_qualify_turntable_silhouette(
    solution: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    factors: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    phase_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    initialization: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    mask_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    canonical_mesh: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    skinning_weights: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    joint_transforms: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    checkpoint: Annotated[Path, typer.Option()],
    sample_frame_count: Annotated[int, typer.Option(min=3, max=32)] = 8,
    render_resolution: Annotated[int, typer.Option(min=32, max=128)] = 64,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Take one bounded center/focal step using train masks and a fixed scaffold."""
    path = qualify_real_center_focal_step(
        solution,
        factors,
        phase_checkpoint,
        initialization,
        manifest,
        mask_root,
        canonical_mesh,
        skinning_weights,
        joint_transforms,
        output,
        checkpoint,
        sample_frame_count=sample_frame_count,
        render_resolution=render_resolution,
        device=device,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-turntable-joint")
def v2_qualify_turntable_joint(
    solution: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    factors: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    phase_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    center_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    checkpoint: Annotated[Path, typer.Option()],
    image_height: Annotated[int, typer.Option(min=1)] = 1120,
    image_width: Annotated[int, typer.Option(min=1)] = 720,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Take one Pareto-guarded joint block-Schur camera/motion step."""
    path = qualify_real_joint_schur_step(
        solution,
        factors,
        phase_checkpoint,
        center_checkpoint,
        output,
        checkpoint,
        image_size=(image_height, image_width),
        device=device,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-turntable-capacity")
def v2_qualify_turntable_capacity(
    solution: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    factors: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    phase_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    center_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    joint_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    image_height: Annotated[int, typer.Option(min=1)] = 1120,
    image_width: Annotated[int, typer.Option(min=1)] = 720,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Measure and reject bounded residual capacity under two stress controls."""
    path = qualify_real_residual_capacity(
        solution,
        factors,
        phase_checkpoint,
        center_checkpoint,
        joint_checkpoint,
        output,
        image_size=(image_height, image_width),
        device=device,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("evaluate-turntable-development")
def v2_evaluate_turntable_development(
    solution: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    factors: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    phase_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    center_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    joint_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    initialization: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    mask_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    canonical_mesh: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    skinning_weights: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    joint_transforms: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    render_resolution: Annotated[int, typer.Option(min=32, max=128)] = 64,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Compare T01 with the frozen free-camera control on development masks."""
    path = evaluate_real_turntable_development(
        solution,
        factors,
        phase_checkpoint,
        center_checkpoint,
        joint_checkpoint,
        initialization,
        manifest,
        mask_root,
        canonical_mesh,
        skinning_weights,
        joint_transforms,
        output,
        render_resolution=render_resolution,
        device=device,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-turntable-geodesic")
def v2_qualify_turntable_geodesic(
    solution: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    factors: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    phase_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    center_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    joint_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    initialization: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    mask_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    canonical_mesh: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    skinning_weights: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    joint_transforms: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    split_output: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    render_resolution: Annotated[int, typer.Option(min=32, max=128)] = 64,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Cross-validate bounded T02 geodesic residuals on training evidence."""
    path = qualify_real_t02_internal_validation(
        solution,
        factors,
        phase_checkpoint,
        center_checkpoint,
        joint_checkpoint,
        initialization,
        manifest,
        mask_root,
        canonical_mesh,
        skinning_weights,
        joint_transforms,
        split_output,
        output,
        render_resolution=render_resolution,
        device=device,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-dynamic-camera")
def v2_qualify_dynamic_camera(
    factors: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    initialization: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    mask_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    canonical_mesh: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    skinning_weights: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    joint_transforms: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    split_output: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    render_resolution: Annotated[int, typer.Option(min=32, max=128)] = 64,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Cross-validate fixed T03 dynamic-camera sequence regularization."""
    path = qualify_real_t03_internal_validation(
        factors,
        initialization,
        manifest,
        mask_root,
        canonical_mesh,
        skinning_weights,
        joint_transforms,
        split_output,
        output,
        render_resolution=render_resolution,
        device=device,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-uncertain-dynamic-camera")
def v2_qualify_uncertain_dynamic_camera(
    initialization: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    solution_output: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Preserve CameraHMR cameras and qualify train-only temporal uncertainty."""
    path = qualify_uncertainty_tagged_dynamic_camera(
        initialization,
        manifest,
        solution_output,
        output,
        device=device,
    )
    report = read_json(path)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-dynamic-camera-qualification")
def v2_audit_dynamic_camera_qualification(
    solution: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Restore T04 artifacts and audit every qualification state transition."""
    path = audit_t04_qualification_lifecycle(solution, report, output)
    result = read_json(path)
    _emit(result)
    if result["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("scan-correspondence")
def v2_scan_correspondence(
    manifest: Annotated[
        Path, typer.Option("--manifest", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    validation: Annotated[
        Path | None,
        typer.Option("--validation", exists=True, dir_okay=False, readable=True),
    ] = None,
    maximum_pairs_per_bin: Annotated[int, typer.Option(min=1, max=32)] = 8,
    maximum_dimension: Annotated[int, typer.Option(min=128, max=2048)] = 720,
    seed: Annotated[int, typer.Option(min=0)] = 20260902,
) -> None:
    """Select T01's observation route using accepted training evidence only."""
    path = scan_correspondence_viability(
        manifest,
        output,
        validation_path=validation,
        maximum_pairs_per_bin=maximum_pairs_per_bin,
        maximum_dimension=maximum_dimension,
        seed=seed,
    )
    _emit(read_json(path))


@v2_app.command("scan-track-graph")
def v2_scan_track_graph(
    manifest: Annotated[
        Path, typer.Option("--manifest", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    validation: Annotated[
        Path | None,
        typer.Option("--validation", exists=True, dir_okay=False, readable=True),
    ] = None,
    binding_output: Annotated[Path | None, typer.Option("--binding-output")] = None,
    maximum_dimension: Annotated[int, typer.Option(min=128, max=2048)] = 720,
    seed: Annotated[int, typer.Option(min=0)] = 20260902,
) -> None:
    """Test whether local tracklet factors span and close the train rotation."""
    path = scan_temporal_track_graph(
        manifest,
        output,
        validation_path=validation,
        binding_path=binding_output,
        maximum_dimension=maximum_dimension,
        seed=seed,
    )
    _emit(read_json(path))


@v2_app.command("scan-material-tracks")
def v2_scan_material_tracks(
    manifest: Annotated[
        Path, typer.Option("--manifest", exists=True, dir_okay=False, readable=True)
    ],
    semantic_root: Annotated[
        Path, typer.Option("--semantic-root", exists=True, file_okay=False, readable=True)
    ],
    semantic_qualification: Annotated[
        Path,
        typer.Option("--semantic-qualification", exists=True, dir_okay=False, readable=True),
    ],
    material_output: Annotated[Path, typer.Option("--material-output")],
    photometric_output: Annotated[Path, typer.Option("--photometric-output")],
    binding_output: Annotated[Path, typer.Option("--binding-output")],
    source_revision: Annotated[str, typer.Option("--source-revision")],
    validation: Annotated[
        Path | None,
        typer.Option("--validation", exists=True, dir_okay=False, readable=True),
    ] = None,
    maximum_dimension: Annotated[int, typer.Option(min=128, max=2048)] = 720,
    maximum_corners: Annotated[int, typer.Option(min=32, max=4096)] = 600,
    start_stride: Annotated[int, typer.Option(min=1, max=64)] = 8,
    maximum_track_steps: Annotated[int, typer.Option(min=8, max=180)] = 75,
    seed: Annotated[int, typer.Option(min=0)] = 20260902,
) -> None:
    """Qualify Q02a material tracks and Q02b photometry as separate decisions."""
    material_path, photometric_path = scan_visibility_bounded_material_tracks(
        manifest,
        semantic_root,
        semantic_qualification,
        material_output,
        photometric_output,
        binding_output,
        source_revision=source_revision,
        validation_path=validation,
        maximum_dimension=maximum_dimension,
        maximum_corners=maximum_corners,
        start_stride=start_stride,
        maximum_track_steps=maximum_track_steps,
        seed=seed,
    )
    material = read_json(material_path)
    photometric = read_json(photometric_path)
    _emit({"material": material, "photometric": photometric})
    if material["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("preserve-semantics")
def v2_preserve_semantics(
    labels: Annotated[Path, typer.Option("--labels", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    confidence: Annotated[
        Path | None, typer.Option("--confidence", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Retain Sapiens2's 29-class evidence instead of collapsing it to a body mask."""
    reject_sealed_capability([labels, output, *([confidence] if confidence else [])])
    path = preserve_sapiens2_semantics(labels, output, confidence_path=confidence)
    _emit({"status": "pass", "output": str(path), "sealed_test_accesses": 0})


@v2_app.command("bind-hull-inputs")
def v2_bind_hull_inputs(
    manifest: Annotated[
        Path, typer.Option("--manifest", exists=True, dir_okay=False, readable=True)
    ],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    camera_solution: Annotated[
        Path, typer.Option("--camera-solution", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    maximum_dimension: Annotated[int, typer.Option(min=32, max=1120)] = 256,
    semantic_root: Annotated[
        Path | None, typer.Option("--semantic-root", exists=True, file_okay=False)
    ] = None,
    semantic_qualification: Annotated[
        Path | None,
        typer.Option("--semantic-qualification", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Bind accepted train masks to exact T04 cameras and uncertainty."""
    path = bind_t04_hull_inputs(
        manifest,
        mask_root,
        camera_solution,
        output,
        maximum_dimension=maximum_dimension,
        semantic_root=semantic_root,
        semantic_qualification_path=semantic_qualification,
    )
    with np.load(path, allow_pickle=False) as archive:
        result = {
            "status": "pass",
            "output": str(path),
            "training_view_count": int(archive["silhouettes"].shape[0]),
            "bound_image_shape": archive["bound_image_shape"].tolist(),
            "camera_parameter_policy": str(archive["camera_parameter_policy"]),
            "semantic_layer_names": sorted(
                name.removeprefix("semantic__")
                for name in archive.files
                if name.startswith("semantic__")
            ),
            "legacy_development_images_read": 0,
            "sealed_test_accesses": 0,
        }
    _emit(result)


@v2_app.command("qualify-semantics")
def v2_qualify_semantics(
    manifest: Annotated[
        Path, typer.Option("--manifest", exists=True, dir_okay=False, readable=True)
    ],
    mask_root: Annotated[Path, typer.Option("--mask-root", exists=True, file_okay=False)],
    semantic_root: Annotated[Path, typer.Option("--semantic-root", exists=True, file_okay=False)],
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    source_revision: Annotated[str, typer.Option("--source-revision")],
) -> None:
    """Qualify lossless train-only Sapiens2 DOME29 labels and confidence."""
    path = qualify_sapiens2_semantic_directory(
        manifest,
        mask_root,
        semantic_root,
        checkpoint,
        output,
        source_revision=source_revision,
    )
    result = read_json(path)
    _emit(result)
    if result["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-semantic-qualification")
def v2_audit_semantic_qualification(
    extraction_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Restore S01 artifacts and audit every qualification transition."""
    path = audit_s01_qualification_lifecycle(extraction_manifest, report, output)
    result = read_json(path)
    _emit(result)
    if result["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("build-hull")
def v2_build_hull(
    inputs: Annotated[Path, typer.Option("--inputs", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    resolution: Annotated[int, typer.Option(min=3)] = 48,
    extent: Annotated[float, typer.Option(min=0.001)] = 1.5,
    aggregation: Annotated[str, typer.Option()] = "weighted_quantile",
) -> None:
    """Build an uncertainty-preserving hull from a train-only NPZ evidence binding."""
    reject_sealed_capability([inputs, output])
    if output.exists():
        raise FileExistsError("V2 hull outputs are immutable")
    with np.load(inputs, allow_pickle=False) as archive:
        silhouettes = torch.as_tensor(archive["silhouettes"], dtype=torch.float32)
        intrinsics = torch.as_tensor(archive["intrinsics"], dtype=torch.float32)
        rotations = torch.as_tensor(archive["rotations"], dtype=torch.float32)
        translations = torch.as_tensor(archive["translations"], dtype=torch.float32)
        confidence = (
            torch.as_tensor(archive["mask_confidence"], dtype=torch.float32)
            if "mask_confidence" in archive
            else None
        )
        motion = (
            torch.as_tensor(archive["motion_uncertainty"], dtype=torch.float32)
            if "motion_uncertainty" in archive
            else None
        )
        semantic_masks = {
            name.removeprefix("semantic__"): torch.as_tensor(archive[name], dtype=torch.float32)
            for name in archive.files
            if name.startswith("semantic__")
        }
        source_hashes = (
            json.loads(str(archive["source_hashes"]))
            if "source_hashes" in archive
            else {"input_binding": sha256_file(inputs)}
        )
    volume = build_confidence_aware_visual_hull(
        silhouettes,
        intrinsics,
        rotations,
        translations,
        resolution=resolution,
        extent=extent,
        mask_confidence=confidence,
        motion_uncertainty_per_view=motion,
        aggregation=aggregation,
        semantic_masks=semantic_masks,
        source_hashes=source_hashes,
    )
    volume.save(output)
    _emit(
        {
            "status": "pass",
            "output": str(output),
            "metadata": volume.metadata.model_dump(mode="json"),
            "unsupported_fraction": float(volume.unsupported.float().mean()),
            "scientific_attempt_marker_created": False,
        }
    )


@v2_app.command("qualify-hull")
def v2_qualify_hull(
    inputs: Annotated[Path, typer.Option("--inputs", exists=True, dir_okay=False)],
    volume: Annotated[Path, typer.Option("--volume", exists=True, dir_okay=False)],
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Qualify exact replay, view sparsity, and corrupted-mask hull robustness."""
    path = qualify_visual_hull_robustness(inputs, volume, public_benchmark, output)
    result = read_json(path)
    _emit(result)
    if result["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("train-outer")
def v2_train_outer(
    evidence_volume: Annotated[
        Path, typer.Option("--evidence-volume", exists=True, dir_okay=False)
    ],
    evidence_binding: Annotated[
        Path, typer.Option("--evidence-binding", exists=True, dir_okay=False)
    ],
    hull_qualification: Annotated[
        Path, typer.Option("--hull-qualification", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")],
    flexicubes_repository: Annotated[
        Path,
        typer.Option(
            "--flexicubes-repository",
            exists=True,
            file_okay=False,
            help="Pinned official FlexiCubes checkout used for the runtime route probe.",
        ),
    ],
    device: Annotated[str, typer.Option()] = "cpu",
    extraction_device: Annotated[
        str,
        typer.Option(
            "--extraction-device",
            help="Explicit extraction/audit device; CPU avoids unsupported MPS linear algebra.",
        ),
    ] = "cpu",
) -> None:
    """Run the one-step G01 qualification path, never a scientific attempt."""
    reject_sealed_capability(
        [
            evidence_volume,
            evidence_binding,
            hull_qualification,
            output,
            flexicubes_repository,
        ]
    )
    if output.exists():
        raise FileExistsError("V2 outer-field qualification reports are immutable")
    report = qualify_outer_field(
        EvidenceVolume.load(evidence_volume, device=device),
        evidence_volume_path=evidence_volume,
        evidence_binding_path=evidence_binding,
        hull_qualification_path=hull_qualification,
        device=device,
        extraction_device=extraction_device,
        flexicubes_repository=flexicubes_repository,
    )
    write_json(output, report)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-outer-qualification")
def v2_audit_outer_qualification(
    report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    evaluator_report: Annotated[
        Path | None, typer.Option("--evaluator-report", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Audit G01 through its achieved local checkpoint-restored state."""
    path = audit_g01_local_qualification_lifecycle(
        report,
        output,
        evaluator_report_path=evaluator_report,
    )
    result = read_json(path)
    _emit(result)
    if result["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("audit-outer-cuda-qualification")
def v2_audit_outer_cuda_qualification(
    report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    claim: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    plan: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    evaluator_lifecycle: Annotated[
        Path, typer.Option("--evaluator-lifecycle", exists=True, dir_okay=False)
    ],
    local_qualification: Annotated[
        Path, typer.Option("--local-qualification", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Audit the final EVALUATOR_DRY to QUALIFIED G01 CUDA transition."""
    path = audit_g01_target_cuda_qualification(
        report,
        claim,
        plan,
        evaluator_lifecycle,
        local_qualification,
        output,
    )
    result = read_json(path)
    _emit(result)
    if result["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("qualify-outer-evaluators")
def v2_qualify_outer_evaluators(
    public_benchmark: Annotated[
        Path, typer.Option("--public-benchmark", exists=True, dir_okay=False)
    ],
    r03_control_contract: Annotated[
        Path, typer.Option("--r03-control-contract", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Dry-run G01's independent public and frozen r03 evaluation routes."""
    path = qualify_g01_evaluator_routes(public_benchmark, r03_control_contract, output)
    result = read_json(path)
    _emit(result)
    if result["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("plan-outer-cuda")
def v2_plan_outer_cuda(
    contract: Annotated[
        Path,
        typer.Option(
            "--contract",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("configs/evaluation/post_v2_g01_direct_multires_field_outer_r01.yaml"),
    evidence_volume: Annotated[
        Path, typer.Option("--evidence-volume", exists=True, dir_okay=False)
    ] = Path(
        "outputs/post_v2/postv2_g01_direct_multires_field_outer_r01/"
        "registered-20260902-r01/qualification/t04_s01_visual_hull_r32_r07.npz"
    ),
    evidence_binding: Annotated[
        Path, typer.Option("--evidence-binding", exists=True, dir_okay=False)
    ] = Path(
        "outputs/post_v2/postv2_g01_direct_multires_field_outer_r01/"
        "registered-20260902-r01/qualification/t04_s01_hull_inputs_256_r02.npz"
    ),
    hull_qualification: Annotated[
        Path, typer.Option("--hull-qualification", exists=True, dir_okay=False)
    ] = Path(
        "outputs/post_v2/postv2_g01_direct_multires_field_outer_r01/"
        "registered-20260902-r01/qualification/"
        "t04_s01_visual_hull_qualification_r32_r07.json"
    ),
    local_qualification: Annotated[
        Path, typer.Option("--local-qualification", exists=True, dir_okay=False)
    ] = Path(
        "outputs/post_v2/postv2_g01_direct_multires_field_outer_r01/"
        "registered-20260902-r01/qualification/"
        "real_t04_s01_outer_field_local_mps_cpu_extract_r05.json"
    ),
    lifecycle: Annotated[Path, typer.Option("--lifecycle", exists=True, dir_okay=False)] = Path(
        "outputs/post_v2/postv2_g01_direct_multires_field_outer_r01/"
        "registered-20260902-r01/qualification/"
        "real_t04_s01_outer_field_local_lifecycle_r03.json"
    ),
    output: Annotated[Path, typer.Option("--output")] = Path(
        "outputs/post_v2/postv2_g01_direct_multires_field_outer_r01/"
        "registered-20260902-r01/qualification/cuda_plan_r01.json"
    ),
    provider_rate_usd_per_hour: Annotated[
        float | None, typer.Option("--provider-rate-usd-per-hour", min=0.0)
    ] = None,
    price_checked_at: Annotated[str | None, typer.Option("--price-checked-at")] = None,
    maximum_cost_usd: Annotated[float | None, typer.Option("--maximum-cost-usd", min=0.0)] = None,
    dispatch_authorized: Annotated[
        bool,
        typer.Option(
            "--dispatch-authorized",
            help="Record explicit owner authorization in the plan; this command never dispatches.",
        ),
    ] = False,
) -> None:
    """Plan the one-shot G01 CUDA qualification without invoking Modal."""
    path = build_g01_cuda_qualification_plan(
        project_root=Path.cwd(),
        contract_path=contract,
        evidence_volume_path=evidence_volume,
        evidence_binding_path=evidence_binding,
        hull_qualification_path=hull_qualification,
        local_qualification_path=local_qualification,
        lifecycle_path=lifecycle,
        output_path=output,
        provider_rate_usd_per_hour=provider_rate_usd_per_hour,
        price_checked_at=price_checked_at,
        maximum_cost_usd=maximum_cost_usd,
        dispatch_authorized=dispatch_authorized,
    )
    _emit(read_json(path))


@v2_app.command("diagnose-normals")
def v2_diagnose_normals(
    output: Annotated[Path, typer.Option("--output")],
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Run analytic inverse-transpose versus rotation-only qualification."""
    reject_sealed_capability([output])
    report = qualify_normal_transport(device=device)
    write_json(output, report)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("train-layers")
def v2_train_layers(
    output: Annotated[Path, typer.Option("--output")],
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Run the L01 layer/contact/visibility qualification path only."""
    reject_sealed_capability([output])
    report = qualify_layered_model(device=device)
    write_json(output, report)
    _emit(report)
    if report["status"] != "pass":
        raise typer.Exit(2)


@v2_app.command("evaluate")
def v2_evaluate(
    metrics: Annotated[Path, typer.Option("--metrics", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Apply the frozen r03 non-regression gates without sealed-test capability."""
    reject_sealed_capability([metrics, output])
    payload = read_json(metrics)
    values = payload.get("historical_image_metrics", payload)
    if not isinstance(values, dict):
        raise typer.BadParameter("metrics payload must contain a numeric object")
    blockers = inherited_real_gate({str(name): float(value) for name, value in values.items()})
    report = {
        "schema_version": "frayid_v2_real_gate.v1",
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "sealed_test_accesses": 0,
    }
    write_json(output, report)
    _emit(report)
    if blockers:
        raise typer.Exit(2)


@v2_app.command("report-dry-run")
def v2_report_dry_run(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Validate the complete report schema without claiming a scientific result."""
    reject_sealed_capability([output])
    report = V2EvaluationReport(
        status="blocked",
        experiment_id="postv2_h01_turntable_layered_multimodal_r01",
        run_id="registered-20260902-r01",
        historical_image_metrics={
            "held_out_iou": 0.0,
            "normalized_boundary_error": 0.0,
            "median_normal_error_degrees": 0.0,
            "train_held_out_iou_gap": 0.0,
        },
        geometry_metrics={
            "bidirectional_chamfer": 0.0,
            "symmetric_point_to_plane": 0.0,
            "median_normal_error_degrees": 0.0,
            "cross_section_error": 0.0,
            "gap_survival": 0.0,
            "curve_survival": 0.0,
            "pose_transfer_error": 0.0,
        },
        layer_metrics={
            "boundary_loop_error": 0.0,
            "penetration_count": 0.0,
            "contact_band_error": 0.0,
            "layer_order_error": 0.0,
            "uncertainty_calibration_error": 0.0,
        },
        topology_certificates={},
        capacity_stress_metrics={
            "camera": 0.0,
            "deformation": 0.0,
            "visibility": 0.0,
            "appearance": 0.0,
        },
        provenance_coverage={"observed": 0.0, "prior_derived": 0.0},
        replay_exact=False,
        blockers=["dry_run_only_no_scientific_result"],
    )
    write_json(output, report)
    _emit(
        {
            "status": "pass",
            "dry_run": True,
            "report_status": report.status,
            "output": str(output),
            "sealed_test_accesses": 0,
        }
    )


if __name__ == "__main__":
    app()
