from __future__ import annotations

import torch

from frayid.material_tracks import (
    TrackletAssignment,
    segment_tracklets,
    tracklet_redescending_loss,
    tracklet_reliability,
)


def test_tracklet_assignment_allows_global_id_subset_in_batch() -> None:
    assignment = TrackletAssignment(torch.tensor([[3, -1], [3, 5]]), tracklet_count=6)
    assignment.validate((2, 2))


def test_tracklet_segmentation_breaks_on_occlusion_and_consistency() -> None:
    valid = torch.ones((5, 2), dtype=torch.bool)
    forward_backward = torch.zeros((5, 2))
    cycle = torch.zeros((5, 2))
    occlusion = torch.zeros((5, 2), dtype=torch.bool)
    occlusion[3, 0] = True
    forward_backward[2, 1] = 2.0
    assignment = segment_tracklets(
        valid,
        forward_backward,
        cycle,
        occlusion,
        maximum_forward_backward_error=0.5,
        maximum_cycle_error=0.5,
    )
    assert assignment.tracklet_count == 4
    assert assignment.ids[2, 1] == -1
    assert assignment.ids[2, 0] != assignment.ids[3, 0]


def test_redescending_tracklet_penalty_saturates_and_downweights_outlier() -> None:
    assignment_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    assignment = TrackletAssignment(assignment_ids, 2)
    penalties = torch.tensor([[0.01, 10.0], [0.01, 10.0]], requires_grad=True)
    loss, sums = tracklet_redescending_loss(
        penalties, torch.ones_like(penalties), assignment, lambda_value=0.1
    )
    loss.backward()
    assert float(loss) < 0.1
    reliability = tracklet_reliability(sums.detach(), lambda_value=0.1)
    assert float(reliability[0]) > 0.5
    assert float(reliability[1]) < 1e-3
    assert penalties.grad is not None
    assert float(penalties.grad[0, 0]) > float(penalties.grad[0, 1])
