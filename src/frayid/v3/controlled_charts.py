from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from typing import Any

from frayid.v3.material_charts import build_material_chart_graph, public_chart_fixture

EXPERIMENT_ID = "postv3_q05_controlled_material_chart_graph_r01"
SOURCES = ("lk", "tapir", "cotracker3")


def public_controlled_chart_fixture() -> dict[str, Any]:
    payload = public_chart_fixture()
    payload["experiment_id"] = EXPERIMENT_ID
    payload["minimum_proposal_sources_per_observation"] = 2
    return payload


def _accepted_capacity(payload: dict[str, Any]) -> int:
    graph = build_material_chart_graph(payload)
    return sum(track.accepted for track in graph.tracks)


def qualify_controlled_chart_robustness(payload: dict[str, Any]) -> dict[str, Any]:
    """Require three-source observability and survive any one-source failure."""
    working = copy.deepcopy(payload)
    working["experiment_id"] = EXPERIMENT_ID
    working["minimum_proposal_sources_per_observation"] = 2
    proposals = working.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("controlled chart proposals must be a list")

    sources_by_observation: dict[tuple[str, int], set[str]] = defaultdict(set)
    observations_by_track: dict[str, set[int]] = defaultdict(set)
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("each controlled chart proposal must be an object")
        key = (str(proposal["track_id"]), int(proposal["frame_index"]))
        sources_by_observation[key].add(str(proposal["source"]))
        observations_by_track[key[0]].add(key[1])
    incomplete = [
        {"track_id": track_id, "frame_index": frame_index, "sources": sorted(sources)}
        for (track_id, frame_index), sources in sorted(sources_by_observation.items())
        if sources != set(SOURCES)
    ]
    fully_redundant_tracks = {
        track_id
        for track_id, frames in observations_by_track.items()
        if all(sources_by_observation[(track_id, frame)] == set(SOURCES) for frame in frames)
    }

    working["corrupted_proposal_capacity_regression"] = 0.0
    clean_capacity = _accepted_capacity(working)
    corruption_capacity: dict[str, int] = {}
    dropout_capacity: dict[str, int] = {}
    for source in SOURCES:
        corrupted = copy.deepcopy(working)
        for proposal in corrupted["proposals"]:
            track_number = int(str(proposal["track_id"]).rsplit("-", maxsplit=1)[-1])
            if proposal["source"] == source and track_number % 5 == 0:
                proposal["xy"][0] += 50.0
                proposal["xy"][1] -= 50.0
        corruption_capacity[source] = _accepted_capacity(corrupted)

        dropped = copy.deepcopy(working)
        dropped["proposals"] = [
            proposal for proposal in dropped["proposals"] if proposal["source"] != source
        ]
        dropout_capacity[source] = _accepted_capacity(dropped)

    maximum_regression = max(
        [clean_capacity - value for value in corruption_capacity.values()]
        + [clean_capacity - value for value in dropout_capacity.values()]
    )
    working["corrupted_proposal_capacity_regression"] = float(max(maximum_regression, 0))
    graph = build_material_chart_graph(working)
    blockers = list(graph.blockers)
    if incomplete:
        blockers.append("incomplete_three_source_observation_support")
    if len(fully_redundant_tracks) < 100:
        blockers.append("fully_three_source_redundant_tracks_below_100")
    if maximum_regression > 0:
        blockers.append("single_source_failure_capacity_regression")
    blockers = sorted(set(blockers))
    replay_payload = {
        "graph_replay_hash": graph.exact_replay_hash,
        "fully_redundant_tracks": sorted(fully_redundant_tracks),
        "corruption_capacity": corruption_capacity,
        "dropout_capacity": dropout_capacity,
    }
    replay_hash = hashlib.sha256(
        json.dumps(replay_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "frayid_v3_q05_controlled_chart_robustness.v1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_scope": graph.evidence_scope,
        "status": "pass" if not blockers else "fail",
        "promotion_eligible": False,
        "graph": graph.model_dump(mode="json"),
        "clean_accepted_track_capacity": clean_capacity,
        "complete_three_source_observation_count": len(sources_by_observation) - len(incomplete),
        "incomplete_three_source_observation_count": len(incomplete),
        "incomplete_three_source_observations": incomplete,
        "fully_three_source_redundant_track_count": len(fully_redundant_tracks),
        "corruption_accepted_capacity_by_source": corruption_capacity,
        "dropout_accepted_capacity_by_source": dropout_capacity,
        "maximum_single_source_failure_capacity_regression": max(maximum_regression, 0),
        "exact_replay_hash": replay_hash,
        "blockers": blockers,
        "project_evidence_reads": 0 if graph.evidence_scope == "public_synthetic" else 72,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
    }


__all__ = ["public_controlled_chart_fixture", "qualify_controlled_chart_robustness"]
