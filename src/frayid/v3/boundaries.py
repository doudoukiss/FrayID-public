from __future__ import annotations

from typing import Any

from frayid.v3.schemas import (
    BoundaryClass,
    BoundaryCurveHypothesis,
    BoundaryHypothesisSet,
)

EXPERIMENT_ID = "postv3_l04_physical_boundary_ontology_r01"
PHYSICAL_LOOPS = ("neck", "left_armhole", "right_armhole", "hem")


def infer_boundary_hypotheses(payload: dict[str, Any]) -> BoundaryHypothesisSet:
    """Classify synchronized curves without equating silhouettes with material edges."""
    raw_curves = payload.get("curves")
    if not isinstance(raw_curves, list):
        raise ValueError("curves must be a list")
    l03_boundary_error = float(payload["l03_boundary_error_pixels"])
    curves: list[BoundaryCurveHypothesis] = []
    for raw in raw_curves:
        if not isinstance(raw, dict):
            raise ValueError("each curve must be an object")
        label = BoundaryClass(str(raw["label"]))
        loop = raw.get("garment_loop")
        phase_bins = sorted(set(int(value) for value in raw.get("phase_bins", [])))
        chart_ids = sorted(set(str(value) for value in raw.get("independent_chart_ids", [])))
        reprojection = float(raw["median_reprojection_pixels"])
        alternative = float(raw["alternative_explanation_pixels"])
        rejection: list[str] = []
        if label is BoundaryClass.PHYSICAL_BOUNDARY:
            if loop not in PHYSICAL_LOOPS:
                rejection.append("unregistered_physical_loop")
            if len(phase_bins) < 4:
                rejection.append("support_below_four_separated_phase_bins")
            if len(chart_ids) < 2:
                rejection.append("reappearance_below_two_independent_charts")
            if reprojection >= alternative:
                rejection.append("apparent_or_occlusion_explanation_not_rejected")
            if l03_boundary_error <= 0.0 or 1.0 - reprojection / l03_boundary_error < 0.2:
                rejection.append("cross_view_improvement_below_20_percent")
        else:
            rejection.append("not_a_physical_boundary")
        curves.append(
            BoundaryCurveHypothesis(
                curve_id=str(raw["curve_id"]),
                label=label,
                garment_loop=loop,
                phase_bins=phase_bins,
                independent_chart_ids=chart_ids,
                median_reprojection_pixels=reprojection,
                alternative_explanation_pixels=alternative,
                accepted=not rejection,
                rejection_reasons=rejection,
            )
        )

    promoted = sorted(
        {
            curve.garment_loop
            for curve in curves
            if curve.accepted
            and curve.label is BoundaryClass.PHYSICAL_BOUNDARY
            and curve.garment_loop is not None
        }
    )
    blockers = [
        f"unsupported_physical_loop:{loop}" for loop in PHYSICAL_LOOPS if loop not in promoted
    ]
    evidence_scope = str(payload.get("evidence_scope", "public_synthetic"))
    return BoundaryHypothesisSet(
        experiment_id=EXPERIMENT_ID,
        evidence_scope=evidence_scope,  # type: ignore[arg-type]
        promotion_eligible=not blockers and evidence_scope == "train_real",
        garment_hypothesis="sleeveless_upper_genus0_four_boundaries",
        curves=curves,
        promoted_physical_loops=promoted,
        status="pass" if not blockers else "fail",
        blockers=blockers,
    )


def public_boundary_fixture(*, omit_loop: str | None = None) -> dict[str, Any]:
    curves = []
    for index, loop in enumerate(PHYSICAL_LOOPS):
        if loop == omit_loop:
            continue
        curves.append(
            {
                "curve_id": f"physical-{loop}",
                "label": "physical_boundary",
                "garment_loop": loop,
                "phase_bins": [index, index + 3, index + 6, index + 9],
                "independent_chart_ids": [f"chart-{index}", f"chart-{index + 4}"],
                "median_reprojection_pixels": 1.5,
                "alternative_explanation_pixels": 3.0,
            }
        )
    curves.extend(
        [
            {
                "curve_id": "view-silhouette",
                "label": "apparent_contour",
                "garment_loop": None,
                "phase_bins": [0, 1, 2],
                "independent_chart_ids": ["chart-0"],
                "median_reprojection_pixels": 1.0,
                "alternative_explanation_pixels": 0.5,
            },
            {
                "curve_id": "side-seam-hypothesis",
                "label": "seam",
                "garment_loop": None,
                "phase_bins": [2, 5, 8],
                "independent_chart_ids": ["chart-2"],
                "median_reprojection_pixels": 1.0,
                "alternative_explanation_pixels": 0.8,
            },
        ]
    )
    return {
        "curves": curves,
        "l03_boundary_error_pixels": 2.0,
        "evidence_scope": "public_synthetic",
    }
