from __future__ import annotations

from frayid.replay_gate import run_replay_gate


def test_cpu_replay_gate_passes_all_registered_boundaries() -> None:
    report = run_replay_gate("cpu")
    assert report["status"] == "pass"
    assert [item["checkpoint_step"] for item in report["checks"]] == [1, 7, 12, 23]
    assert report["checks"][1]["forced_projection_seen"] is True
    assert report["checks"][2]["stage_refresh_seen"] is True
    assert report["negative_control"]["mismatch_detected"] is True
