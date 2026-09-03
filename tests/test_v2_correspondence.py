from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from frayid.cli import app
from frayid.io import sha256_file, write_json
from frayid.schemas import DatasetManifest, FrameRecord, VideoMetadata
from frayid.v2.correspondence import (
    CorrespondenceGate,
    TemporalGraphGate,
    scan_correspondence_viability,
    scan_temporal_track_graph,
)
from frayid.v2.material_tracks import (
    MaterialTrackGate,
    load_visibility_bounded_material_tracks,
    scan_visibility_bounded_material_tracks,
)
from frayid.v2.track_factors import (
    PairwiseTrackletFactors,
    load_pairwise_tracklet_factors,
    pairwise_sampson_loss,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    image_root = dataset_root / "images"
    mask_root = dataset_root / "masks"
    image_root.mkdir(parents=True)
    mask_root.mkdir()
    rng = np.random.default_rng(19)
    texture = rng.integers(0, 256, size=(144, 112, 3), dtype=np.uint8)
    cv2.circle(texture, (56, 72), 38, (0, 255, 30), 3)
    cv2.putText(texture, "FRAY", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    mask = np.full((144, 112), 255, dtype=np.uint8)
    records: list[FrameRecord] = []
    for ordinal in range(180):
        filename = f"frame_{ordinal:04d}_source_{ordinal:06d}.png"
        split = "held_out" if ordinal % 5 == 0 else "train"
        if split == "train":
            assert cv2.imwrite(str(image_root / filename), texture)
            assert cv2.imwrite(str(mask_root / filename), mask)
            image_path = str(image_root / filename)
        else:
            # A nonexistent held-out path makes accidental binding fail the test loudly.
            image_path = str(dataset_root / "held_out_must_not_be_read" / filename)
        records.append(
            FrameRecord(
                ordinal=ordinal,
                source_frame_index=ordinal,
                timestamp_seconds=ordinal / 12.0,
                image_path=image_path,
                split=split,
                blur_variance=100.0,
                mean_luminance=120.0,
                quality_accepted=True,
            )
        )
    manifest = DatasetManifest(
        status="evidence_ready",
        run_id="public-observability-fixture",
        input_video_path="public-fixture.mp4",
        input_video_sha256="a" * 64,
        video=VideoMetadata(
            path="public-fixture.mp4",
            codec="synthetic",
            width=112,
            height=144,
            frame_count=180,
            frame_rate=12.0,
            duration_seconds=15.0,
            size_bytes=1,
        ),
        dataset_root=str(dataset_root),
        frames=records,
        train_frame_count=144,
        held_out_frame_count=36,
        rejected_candidate_count=0,
    )
    manifest_path = write_json(dataset_root / "dataset_manifest.json", manifest)
    validation_path = write_json(
        dataset_root / "dataset_validation.json",
        {"status": "ready", "blockers": [], "evidence_complete_frame_count": 180},
    )
    return manifest_path, validation_path


def _write_semantic_fixture(manifest_path: Path) -> tuple[Path, Path]:
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    semantic_root = manifest_path.parent / "semantic_frames"
    semantic_root.mkdir()
    frame_records: list[dict[str, int | str]] = []
    for record in manifest.frames:
        if record.split != "train":
            continue
        labels = np.full((144, 112), 23, dtype=np.uint8)
        confidence = np.ones((144, 112), dtype=np.float16)
        semantic_path = semantic_root / f"{Path(record.image_path).stem}.npz"
        np.savez_compressed(
            semantic_path,
            labels=labels,
            confidence=confidence,
            source_frame_index=np.asarray(record.source_frame_index, dtype=np.int64),
        )
        frame_records.append(
            {
                "source_frame_index": record.source_frame_index,
                "semantic_sha256": sha256_file(semantic_path),
            }
        )
    report_path = write_json(
        semantic_root / "semantic_qualification.json",
        {"status": "pass", "frame_records": frame_records},
    )
    return semantic_root, report_path


def test_train_only_correspondence_scan_never_binds_held_out(tmp_path: Path) -> None:
    manifest, validation = _write_fixture(tmp_path)
    output = tmp_path / "qualification.json"
    low_fixture_gate = CorrespondenceGate(
        local_median_inliers=8,
        local_median_inlier_ratio=0.5,
        local_median_coverage=0.05,
        quarter_median_inliers=8,
        quarter_median_inlier_ratio=0.5,
        quarter_median_coverage=0.05,
        loop_median_inliers=8,
        loop_median_inlier_ratio=0.5,
        loop_median_coverage=0.05,
    )
    scan_correspondence_viability(
        manifest,
        output,
        validation_path=validation,
        maximum_pairs_per_bin=1,
        maximum_dimension=160,
        gate=low_fixture_gate,
    )
    report = json.loads(output.read_text())
    assert report["status"] == "pass"
    assert report["access_counters"]["accepted_training_records_bound"] == 144
    assert report["access_counters"]["held_out_records_present_but_not_bound"] == 36
    assert report["access_counters"]["held_out_images_read"] == 0
    assert report["access_counters"]["sealed_test_accesses"] == 0
    assert report["gate_results"]["track_driven_t01_eligible"] is True
    assert report["photometric_normal_route"]["eligible"] is False
    assert report["optimizer_steps"] == 0


def test_correspondence_scan_fails_closed_on_unvalidated_dataset(tmp_path: Path) -> None:
    manifest, validation = _write_fixture(tmp_path)
    write_json(validation, {"status": "blocked", "blockers": ["missing_normals"]})
    with pytest.raises(ValueError, match="ready dataset validation"):
        scan_correspondence_viability(
            manifest,
            tmp_path / "qualification.json",
            validation_path=validation,
            maximum_pairs_per_bin=1,
        )


def test_temporal_track_graph_spans_training_rotation_without_held_out_reads(
    tmp_path: Path,
) -> None:
    manifest, validation = _write_fixture(tmp_path)
    output = tmp_path / "track_graph.json"
    fixture_gate = TemporalGraphGate(
        minimum_edge_inliers=8,
        minimum_edge_inlier_ratio=0.5,
        minimum_edge_coverage=0.05,
        minimum_passing_edge_fraction=0.99,
        minimum_largest_component_fraction=0.99,
        minimum_loop_inliers=8,
        minimum_loop_inlier_ratio=0.5,
        minimum_loop_coverage=0.05,
    )
    scan_temporal_track_graph(
        manifest,
        output,
        validation_path=validation,
        binding_path=tmp_path / "track_factors.npz",
        maximum_dimension=160,
        gate=fixture_gate,
    )
    report = json.loads(output.read_text())
    assert report["graph_metrics"]["frame_node_count"] == 144
    assert report["graph_metrics"]["temporal_edge_count"] == 143
    assert report["graph_metrics"]["largest_passing_component_fraction"] == 1.0
    assert report["gate_results"]["temporal_track_graph_eligible_for_t01"] is True
    assert report["gate_results"]["direct_quarter_turn_identity_factor_enabled"] is False
    assert report["access_counters"]["held_out_images_read"] == 0
    with np.load(tmp_path / "track_factors.npz", allow_pickle=False) as binding:
        assert str(binding["schema_version"]) == "frayid_v2_pairwise_tracklet_factors.v1"
        assert len(binding["first_ordinals"]) == 143
        assert int(binding["edge_offsets"][-1]) == len(binding["first_pixels"])
        assert len(binding["first_pixels"]) == len(binding["second_pixels"])
        assert len(binding["first_pixels"]) == len(binding["observation_weights"])
    factors = load_pairwise_tracklet_factors(tmp_path / "track_factors.npz")
    assert factors.edge_count == 143
    assert factors.factor_count > 0


def test_v2_cli_exposes_correspondence_observability_gate() -> None:
    result = CliRunner().invoke(app, ["v2", "--help"])
    assert result.exit_code == 0
    assert "scan-correspondence" in result.stdout
    assert "scan-track-graph" in result.stdout
    assert "scan-material-tracks" in result.stdout
    assert "benchmark-q03" in result.stdout
    assert "qualify-q03" in result.stdout
    assert "audit-q03-lifecycle" in result.stdout


def test_q02a_material_tracks_survive_when_q02b_photometry_fails(tmp_path: Path) -> None:
    manifest, validation = _write_fixture(tmp_path)
    semantic_root, semantic_report = _write_semantic_fixture(manifest)
    material_output = tmp_path / "q02a.json"
    photometric_output = tmp_path / "q02b.json"
    binding_output = tmp_path / "q02_material_tracks.npz"
    fixture_gate = MaterialTrackGate(
        minimum_observations=8,
        minimum_semantic_stability=0.95,
        minimum_median_semantic_confidence=0.95,
        minimum_endpoint_patch_ncc=0.8,
        minimum_30_degree_tracks=1,
        minimum_90_degree_tracks=1,
        minimum_supported_semantic_layers=1,
        reverse_audit_maximum_tracks=16,
        reverse_return_error_p95_pixels=1.0,
        reverse_pass_fraction=0.95,
        photometric_minimum_90_degree_tracks=1,
        photometric_median_harmonic_improvement=0.1,
        photometric_positive_track_fraction=0.6,
        photometric_shuffled_margin=0.05,
    )
    material_path, photometric_path = scan_visibility_bounded_material_tracks(
        manifest,
        semantic_root,
        semantic_report,
        material_output,
        photometric_output,
        binding_output,
        source_revision="0" * 40,
        validation_path=validation,
        maximum_dimension=160,
        maximum_corners=80,
        start_stride=24,
        maximum_track_steps=50,
        gate=fixture_gate,
    )
    material = json.loads(material_path.read_text(encoding="utf-8"))
    photometric = json.loads(photometric_path.read_text(encoding="utf-8"))
    assert material["status"] == "pass"
    assert material["material_track_route"]["eligible"] is True
    assert material["access_counters"]["held_out_images_read"] == 0
    assert material["access_counters"]["sealed_test_accesses"] == 0
    assert photometric["status"] == "fail"
    assert photometric["activation_eligible"] is False
    assert photometric["q02b_failure_invalidates_q02a"] is False
    binding = load_visibility_bounded_material_tracks(binding_output)
    assert binding.track_count > 0
    assert binding.observation_count > binding.track_count
    assert np.allclose(binding.semantic_confidence, 1.0)


def test_q02_binding_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad_q02.npz"
    np.savez_compressed(path, schema_version=np.asarray("not-q02"))
    with pytest.raises(ValueError, match="unsupported Q02"):
        load_visibility_bounded_material_tracks(path)


def test_pairwise_sampson_factor_is_zero_at_epipolar_identity_and_differentiable() -> None:
    factors = PairwiseTrackletFactors(
        first_ordinals=torch.tensor([1]),
        second_ordinals=torch.tensor([2]),
        first_source_frame_indices=torch.tensor([5]),
        second_source_frame_indices=torch.tensor([10]),
        edge_offsets=torch.tensor([0, 2]),
        first_pixels=torch.tensor([[10.0, 12.0], [20.0, 18.0]]),
        second_pixels=torch.tensor([[11.0, 12.0], [21.0, 18.0]]),
        observation_weights=torch.ones(2),
        geometric_model_codes=torch.zeros(1, dtype=torch.long),
    )
    # Horizontal translation has horizontal epipolar lines: corresponding y values agree.
    fundamental = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    loss = pairwise_sampson_loss(fundamental, factors, image_size=(32, 32))
    loss.backward()
    assert float(loss) < 1.0e-8
    assert fundamental.grad is not None
    assert bool(torch.isfinite(fundamental.grad).all())

    perturbed = PairwiseTrackletFactors(
        **{
            **factors.__dict__,
            "second_pixels": factors.second_pixels + torch.tensor([[0.0, 2.0], [0.0, -1.0]]),
        }
    )
    assert float(pairwise_sampson_loss(fundamental.detach(), perturbed, image_size=(32, 32))) > 0
