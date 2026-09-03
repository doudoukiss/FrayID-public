from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import trimesh
from torch import Tensor

from frayid.v2.schemas import LayerTopologyPolicy, TopologyCertificate


class TopologyStage(StrEnum):
    SEARCH = "search"
    COMMIT = "commit"
    REFINE = "refine"


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(str(canonical.dtype).encode())
        digest.update(str(canonical.shape).encode())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _boundary_loop_audit(faces: np.ndarray) -> tuple[int, bool]:
    if faces.size == 0:
        return 0, True
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if not len(boundary):
        return 0, True
    adjacency: dict[int, set[int]] = {}
    for first, second in boundary.tolist():
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    visited: set[int] = set()
    components = 0
    for start in adjacency:
        if start in visited:
            continue
        components += 1
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency[current].difference(visited))
    loops_are_registered_cycles = all(len(neighbors) == 2 for neighbors in adjacency.values())
    return components, loops_are_registered_cycles


def certify_surface(
    vertices: Tensor,
    faces: Tensor,
    *,
    policy: LayerTopologyPolicy,
    stage: TopologyStage,
    exact_intersection_pair_count: int | None,
    registered_penetration_count: int | None = None,
    replay_exact: bool,
) -> TopologyCertificate:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("surface vertices must have shape [N,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("surface faces must have shape [F,3]")
    if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("surface faces contain an invalid vertex index")
    vertex_array = vertices.detach().cpu().double().numpy()
    face_array = faces.detach().cpu().long().numpy()
    if not np.isfinite(vertex_array).all():
        raise ValueError("surface vertices must be finite")
    mesh = trimesh.Trimesh(vertices=vertex_array, faces=face_array, process=False)
    components = mesh.split(only_watertight=False)
    boundary_loops, boundaries_are_loops = _boundary_loop_audit(face_array)
    blockers: list[str] = []
    if stage is not TopologyStage.SEARCH and exact_intersection_pair_count is None:
        blockers.append("exact_intersection_audit_missing")
    elif exact_intersection_pair_count not in (None, 0):
        blockers.append("exact_self_intersection")
    if not mesh.is_winding_consistent:
        blockers.append("inconsistent_winding")
    if not boundaries_are_loops:
        blockers.append("boundary_not_closed_loops")
    if policy.closed and not mesh.is_watertight:
        blockers.append("body_not_watertight")
    outward = bool(mesh.is_watertight and mesh.is_winding_consistent and float(mesh.volume) > 0)
    if policy.closed and not outward:
        blockers.append("body_not_outward")
    if not policy.closed and mesh.is_watertight:
        blockers.append("open_layer_unexpectedly_watertight")
    if (
        policy.required_component_count is not None
        and len(components) != policy.required_component_count
    ):
        blockers.append("component_policy")
    if (
        policy.required_boundary_loop_count is not None
        and boundary_loops != policy.required_boundary_loop_count
    ):
        blockers.append("boundary_loop_policy")
    if (
        policy.required_euler_number is not None
        and int(mesh.euler_number) != policy.required_euler_number
    ):
        blockers.append("euler_policy")
    if registered_penetration_count not in (None, 0):
        blockers.append("unregistered_penetration")
    if stage is not TopologyStage.SEARCH and not replay_exact:
        blockers.append("certificate_replay")
    status: str
    if stage is TopologyStage.SEARCH:
        status = "nonpromotable"
    elif "exact_intersection_audit_missing" in blockers:
        status = "blocked"
    else:
        status = "fail" if blockers else "pass"
    return TopologyCertificate(
        layer_id=policy.layer_id,
        stage=stage.value,
        status=status,  # type: ignore[arg-type]
        vertex_count=len(vertex_array),
        face_count=len(face_array),
        component_count=len(components),
        boundary_loop_count=boundary_loops,
        euler_number=int(mesh.euler_number),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        outward=outward,
        exact_intersection_pair_count=exact_intersection_pair_count,
        registered_penetration_count=registered_penetration_count,
        connectivity_sha256=_array_digest(face_array),
        surface_sha256=_array_digest(vertex_array, face_array),
        replay_exact=replay_exact,
        blockers=blockers,
    )


@dataclass
class TopologyStateMachine:
    policy: LayerTopologyPolicy
    stage: TopologyStage = TopologyStage.SEARCH
    committed_connectivity_sha256: str | None = None

    def commit(self, certificate: TopologyCertificate) -> None:
        if self.stage is not TopologyStage.SEARCH:
            raise ValueError("topology may commit only from search")
        if certificate.layer_id != self.policy.layer_id or certificate.stage != "commit":
            raise ValueError("certificate does not bind the state-machine layer/stage")
        if certificate.status != "pass":
            raise ValueError("failed or blocked topology cannot be committed")
        self.stage = TopologyStage.COMMIT
        self.committed_connectivity_sha256 = certificate.connectivity_sha256

    def begin_refine(self) -> None:
        if self.stage is not TopologyStage.COMMIT or self.committed_connectivity_sha256 is None:
            raise ValueError("refine requires a passing committed topology")
        self.stage = TopologyStage.REFINE

    def assert_refine_connectivity(self, faces: Tensor) -> None:
        if self.stage is not TopologyStage.REFINE or self.committed_connectivity_sha256 is None:
            raise ValueError("connectivity lock applies only during refine")
        observed = _array_digest(faces.detach().cpu().long().numpy())
        if observed != self.committed_connectivity_sha256:
            raise ValueError("refine connectivity or boundary topology changed")
