from __future__ import annotations

import numpy as np

from frayid.interface_field import InterfaceField, certify_zero_subcomplex


def _single_interface_tetrahedron() -> InterfaceField:
    return InterfaceField(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        values=np.array([0.0, 0.0, 0.0, 1.0]),
        interface_vertices=np.array([True, True, True, False]),
        tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_regions=np.array([1], dtype=np.int8),
        interface_faces=np.array([[0, 1, 2]], dtype=np.int64),
        source_face_indices=np.array([0], dtype=np.int64),
        outside_cell_count=1,
        inside_cell_count=0,
        source_face_count=1,
    )


def test_zero_subcomplex_accepts_only_registered_interface_face() -> None:
    report = certify_zero_subcomplex(_single_interface_tetrahedron())
    assert report["status"] == "pass"


def test_zero_subcomplex_rejects_noninterface_zero_edge_and_face() -> None:
    original = _single_interface_tetrahedron()
    field = InterfaceField(
        **{
            **original.__dict__,
            "values": np.zeros(4),
            "interface_vertices": np.ones(4, dtype=np.bool_),
        }
    )
    report = certify_zero_subcomplex(field)
    assert report["status"] == "fail"
    assert report["noninterface_all_zero_edge_count"] == 3
    assert report["noninterface_all_zero_face_count"] == 3
    assert report["all_zero_tetrahedron_count"] == 1
