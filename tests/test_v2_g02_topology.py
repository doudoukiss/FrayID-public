from __future__ import annotations

from pathlib import Path

import numpy as np

from frayid.io import read_json
from frayid.v2.g02_topology import audit_g02_raw_topology, topology_report_replay_exact


def _write_field(path: Path, *, disconnected: bool) -> Path:
    resolution = 24
    extent = 1.2
    coordinates = np.linspace(-extent, extent, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    first = np.sqrt((xx + (0.45 if disconnected else 0.0)) ** 2 + yy**2 + zz**2) - 0.3
    field = first
    if disconnected:
        second = np.sqrt((xx - 0.45) ** 2 + yy**2 + zz**2) - 0.3
        field = np.minimum(first, second)
    np.savez_compressed(
        path,
        schema_version=np.asarray("frayid_v2_g02_raw_canonical_field.v1"),
        signed_distance=field.astype(np.float32),
        extent=np.asarray(extent, dtype=np.float32),
        resolution=np.asarray(resolution, dtype=np.int64),
        source_revision=np.asarray("a" * 40),
        arm_binding_sha256=np.asarray("b" * 64),
        representation=np.asarray("authoritative_raw_direct_field_candidate"),
        topology_state=np.asarray("search_not_committed"),
    )
    return path


def test_single_component_raw_field_waits_for_exact_intersection_audit(tmp_path: Path) -> None:
    field = _write_field(tmp_path / "field.npz", disconnected=False)
    report_path = audit_g02_raw_topology(field, tmp_path / "report.json")
    report = read_json(report_path)
    assert report["status"] == "blocked"
    assert report["preintersection_failures"] == []
    assert report["exact_intersection_short_circuited"] is False
    assert report["extraction"]["cleanup_operations"] == 0
    assert topology_report_replay_exact(field)


def test_disconnected_raw_field_fails_before_intersection_backend(tmp_path: Path) -> None:
    field = _write_field(tmp_path / "field.npz", disconnected=True)
    report = read_json(audit_g02_raw_topology(field, tmp_path / "report.json"))
    assert report["status"] == "fail"
    assert report["exact_intersection_short_circuited"] is True
    assert "component_policy" in report["blockers"]
    assert report["commit_precheck_certificate"]["component_count"] == 2
