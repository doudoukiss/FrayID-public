from pathlib import Path

import numpy as np
import pytest

from frayid.post_v1_e0 import (
    RELEVANT_EVALUATOR_PATHS,
    assert_not_sealed_private_path,
    exact_metric_comparison,
    topology_report,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0)),
        dtype=np.float32,
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64)
    return vertices, faces


def test_e0_rejects_sealed_private_paths() -> None:
    with pytest.raises(ValueError, match="forbids sealed-test"):
        assert_not_sealed_private_path(Path("outputs/canonical/sealed_test_v1/frame.png"))
    with pytest.raises(ValueError, match="forbids sealed-test"):
        assert_not_sealed_private_path(Path("outputs/runs/sealed_test_audit_abc/report.json"))
    assert_not_sealed_private_path(Path("outputs/canonical/post_v1/e0/report.json"))


def test_e0_relevant_source_bindings_exist() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert all((project_root / relative).is_file() for relative in RELEVANT_EVALUATOR_PATHS)


def test_topology_report_checks_actual_mesh_and_area_floors() -> None:
    vertices, faces = _tetrahedron()
    passing = topology_report(vertices, vertices.copy(), faces)
    assert passing["status"] == "pass"
    assert passing["component_count"] == 1
    assert passing["watertight"] is True
    assert passing["euler_number"] == 2

    collapsed = vertices.copy()
    collapsed[0] = collapsed[1] + (vertices[0] - vertices[1]) * 0.01
    failing = topology_report(vertices, collapsed, faces)
    assert failing["status"] == "fail"
    assert failing["collapsed_face_count"] > 0


def test_exact_metric_comparison_has_zero_tolerance() -> None:
    expected = {
        "train_iou": 0.8,
        "held_out_iou": 0.7,
        "held_out_initialization_iou": 0.5,
        "held_out_improvement": 0.2,
        "boundary_error": 0.01,
        "median_normal_error_degrees": 20.0,
    }
    assert exact_metric_comparison(expected, dict(expected))["status"] == "pass"
    observed = dict(expected)
    observed["held_out_iou"] += 1e-12
    comparison = exact_metric_comparison(expected, observed)
    assert comparison["status"] == "fail"
    assert comparison["mismatches"] == ["held_out_iou"]
