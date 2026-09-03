from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from frayid.cli import app
from frayid.io import read_json, write_json
from frayid.v2 import l03_modal
from frayid.v2.l03_modal import (
    audit_l03_target_cuda_qualification,
    build_l03_cuda_qualification_plan,
)
from frayid.v2.l03_open_layers import make_open_wrinkled_tube
from frayid.v2.l03_training import (
    LayerDisplacementModel,
    run_l03_training_public_benchmark,
    write_l03_training_public_benchmark,
)


def test_l03_displacement_model_is_fixed_topology_and_outward_bounded() -> None:
    vertices, faces = make_open_wrinkled_tube(
        y_minimum=-0.3,
        y_maximum=0.3,
        base_radius=0.25,
        wrinkle_amplitude=0.01,
        phase=0.0,
    )
    points = torch.as_tensor(vertices, dtype=torch.float64)
    triangles = torch.as_tensor(faces, dtype=torch.long)
    radial = points.clone()
    radial[:, 1] = 0.0
    radial = torch.nn.functional.normalize(radial, dim=-1)
    model = LayerDisplacementModel(
        points,
        triangles,
        maximum_displacement_metres=0.05,
        outward_directions=radial,
    )
    with torch.no_grad():
        model.raw_displacement.copy_(torch.linspace(-0.1, 0.1, len(points)))
    displaced = model()
    displacement = model.bounded_displacement()
    assert float(displacement.min()) == 0.0
    assert float(displacement.max()) == pytest.approx(0.05)
    assert torch.equal(model.faces, triangles)
    assert np.all(
        np.linalg.norm(displaced.detach().numpy()[:, [0, 2]], axis=1)
        >= np.linalg.norm(vertices[:, [0, 2]], axis=1)
    )


def test_l03_training_public_benchmark_passes() -> None:
    pytest.importorskip("ipctk")
    report = run_l03_training_public_benchmark()
    assert report["status"] == "pass"
    assert report["objective"]["relative_reduction"] >= 0.90
    assert report["gates"]["registered_open_topology_preserved"] is True


def test_l03_training_public_writer_and_cli_are_immutable(tmp_path: Path) -> None:
    pytest.importorskip("ipctk")
    direct = tmp_path / "direct.json"
    write_l03_training_public_benchmark(direct)
    assert read_json(direct)["status"] == "pass"
    with pytest.raises(FileExistsError, match="immutable"):
        write_l03_training_public_benchmark(direct)
    cli_output = tmp_path / "cli.json"
    result = CliRunner().invoke(app, ["v2", "benchmark-l03-training", "--output", str(cli_output)])
    assert result.exit_code == 0, result.stdout
    assert read_json(cli_output)["status"] == "pass"


def test_l03_cuda_plan_is_ready_only_with_bound_local_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "contract.yaml"
    write_json(
        contract,
        {
            "schema_version": "frayid_v2_experiment_contract.v1",
            "experiment_id": "postv2_l03_semantic_open_clothing_layers_r01",
            "run_id": "registered-20260903-r01",
            "hypothesis": "Self-contained public L03 target-CUDA plan fixture.",
            "changed_mechanism": "bounded semantic open layers",
            "matched_control": "frozen implicit body",
            "source_commit": "a" * 40,
            "immutable_output_root": (
                "outputs/post_v2/postv2_l03_semantic_open_clothing_layers_r01/"
                "registered-20260903-r01"
            ),
            "evidence": {},
            "compute_cap": {
                "qualification_gpu_hours": 8,
                "scientific_gpu_hours": 48,
                "wall_time_seconds": 172800,
                "automatic_retries": 0,
            },
            "qualification_state": "checkpoint_restored",
            "scientific_state": "registered",
            "dependencies": [],
            "promotion_gates": {},
            "stop_conditions": [],
            "historical_records_immutable": True,
            "automatic_paid_retries": 0,
        },
    )
    training_plan = tmp_path / "training_plan.json"
    write_json(
        training_plan,
        {
            "schema_version": "frayid_v2_l03_training_qualification_plan.v1",
            "status": "local_training_qualification_planned",
        },
    )
    local = tmp_path / "local.json"
    write_json(
        local,
        {
            "status": "pass",
            "decision": "local_training_qualification_passed_target_gpu_pending",
            "source_revision": "a" * 40,
            "plan_sha256": hashlib.sha256(training_plan.read_bytes()).hexdigest(),
            "gates": {"same_device_checkpoint_restore_exact": True},
            "provenance": {
                "scientific_attempt_marker_created": False,
                "development_records_read": 0,
                "sealed_test_reads": 0,
            },
        },
    )
    monkeypatch.setattr(
        l03_modal,
        "_git_output",
        lambda _root, *args: "b" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        l03_modal.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    output = tmp_path / "cuda_plan.json"
    build_l03_cuda_qualification_plan(
        project_root=Path.cwd(),
        contract_path=contract,
        local_qualification_path=local,
        training_qualification_plan_path=training_plan,
        output_path=output,
        provider_rate_usd_per_hour=1.95,
        price_checked_at="2026-09-03T00:00:00Z",
        maximum_cost_usd=0.59,
        dispatch_authorized=True,
    )
    report = read_json(output)
    assert report["status"] == "ready"
    assert report["automatic_retries"] == 0
    assert report["scientific_attempt"] is False


def test_l03_target_cuda_envelope_audit_passes(tmp_path: Path) -> None:
    revision = "c" * 40
    plan = tmp_path / "plan.json"
    write_json(plan, {"status": "ready", "automatic_retries": 0, "source_commit": revision})
    envelope = tmp_path / "envelope.json"
    write_json(
        envelope,
        {
            "status": "pass",
            "scientific_attempt": False,
            "automatic_retries": 0,
            "source_revision": revision,
            "qualification_report": {
                "status": "pass",
                "device": "cuda",
                "packaged_source_revision": revision,
                "gates": {
                    "both_layer_gradients_active": True,
                    "both_layer_parameters_change": True,
                    "same_device_checkpoint_restore_exact": True,
                },
            },
        },
    )
    output = tmp_path / "audit.json"
    audit_l03_target_cuda_qualification(envelope_path=envelope, plan_path=plan, output_path=output)
    report = read_json(output)
    assert report["status"] == "pass"
    assert report["state"] == "qualified"
