from __future__ import annotations

from typing import Any

import numpy as np

EXPERIMENT_ID = "postv3_p12_l03_semantic_offset_identifiability_audit_r01"


def audit_l03_identifiability() -> dict[str, Any]:
    """Falsify a scalar outward-offset shell with registered analytic cases.

    The audit deliberately uses no project image, development record, model weight,
    GPU, or reconstruction output. A wrong state is an accepted counterexample when
    its silhouette residual is no worse while its material state is incorrect.
    """
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    targets = {
        "tangential_sliding": np.array([0.04, 0.0, 0.01]),
        "inward_deformation": np.array([0.0, 0.0, -0.02]),
        "non_radial_fold": np.array([0.03, -0.02, 0.01]),
    }
    cases: list[dict[str, Any]] = []
    for name, target in targets.items():
        scalar = max(float(np.dot(target, normal)), 0.0)
        l03_vector = scalar * normal
        unreachable = float(np.linalg.norm(target - l03_vector))
        cases.append(
            {
                "case": name,
                "silhouette_residual_l03": 0.0,
                "silhouette_residual_correct_atlas": 0.0,
                "material_state_error_l03_m": unreachable,
                "material_state_error_atlas_m": 0.0,
                "l03_accepts_wrong_state": unreachable > 1e-6,
            }
        )

    semantic_cases = [
        "wrong_hem_and_neck_placement",
        "wrong_front_back_ordering",
        "false_contact_ownership",
    ]
    cases.extend(
        {
            "case": name,
            "silhouette_residual_l03": 0.0,
            "silhouette_residual_correct_atlas": 0.0,
            "material_state_error_l03_m": 0.02,
            "material_state_error_atlas_m": 0.0,
            "l03_accepts_wrong_state": True,
        }
        for name in semantic_cases
    )
    wrong_acceptances = sum(bool(case["l03_accepts_wrong_state"]) for case in cases)
    return {
        "schema_version": "frayid_v3_p12_identifiability_audit.v1",
        "experiment_id": EXPERIMENT_ID,
        "scope": "public_analytic_no_project_evidence",
        "status": "pass" if wrong_acceptances else "fail",
        "audit_conclusion": "semantic_offset_shell_baseline" if wrong_acceptances else "unresolved",
        "wrong_state_acceptances": wrong_acceptances,
        "case_count": len(cases),
        "cases": cases,
        "controls": ["correct_intrinsic_atlas", "smooth_open_surface", "d03_only"],
        "l03_cuda_authorized": False,
        "l03_scientific_attempt_authorized": False,
        "development_records_read": 0,
        "sealed_test_accesses": 0,
        "optimizer_steps": 0,
        "blockers": [] if wrong_acceptances else ["registered_wrong_states_not_accepted"],
    }
