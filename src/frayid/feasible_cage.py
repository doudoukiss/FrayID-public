from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import osqp  # type: ignore[import-untyped]
import torch
from scipy import sparse  # type: ignore[import-untyped]
from torch import Tensor

from frayid.deformation_cage import TrilinearDeformationCage
from frayid.geometry import canonical_topology_quantities
from frayid.io import sha256_file, write_json

OSQP_EPSILON = 1e-10
CERTIFICATE_TOLERANCE = 1e-8
OSQP_MAXIMUM_ITERATIONS = 20_000
ZERO_ROW_TOLERANCE = 1e-14
POST_CAST_FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class KKTResiduals:
    primal_feasibility: float
    dual_feasibility: float
    stationarity: float
    complementarity: float

    @property
    def maximum(self) -> float:
        return max(
            self.primal_feasibility,
            self.dual_feasibility,
            self.stationarity,
            self.complementarity,
        )


@dataclass(frozen=True)
class CertifiedQPResult:
    status: str
    message: str
    unclipped_direction: Tensor | None
    final_direction: Tensor | None
    multipliers: Tensor | None
    solver_name: str
    solver_version: str
    solver_status: str
    solver_status_value: int | None
    iteration_count: int
    setup_time_seconds: float
    solve_time_seconds: float
    polish_time_seconds: float
    run_time_seconds: float
    constraint_count: int
    normalized_row_count: int
    tautological_zero_row_count: int
    contradictory_zero_row_count: int
    active_constraint_count: int
    trust_region_radius: float
    trust_scale: float
    raw_certificate: KKTResiduals | None
    scaled_certificate: KKTResiduals | None
    unclipped_minimum_slack: float
    final_minimum_slack: float
    final_scaled_primal_residual: float
    settings: dict[str, bool | float | int | str]

    @property
    def certified(self) -> bool:
        return self.status == "certified"

    def to_dict(self, *, include_vectors: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("unclipped_direction", "final_direction", "multipliers"):
            value = getattr(self, key)
            payload[key] = value.tolist() if include_vectors and value is not None else None
        return payload


class FeasibleCageProjectionFailure(RuntimeError):
    """Structured cage failure whose complete QP input can be archived."""

    def __init__(
        self,
        result: CertifiedQPResult,
        candidate: Tensor,
        matrix: Tensor,
        lower_bound: Tensor,
    ) -> None:
        super().__init__(
            f"certified halfspace projection failed: {result.status}: {result.message}"
        )
        self.result = result
        self.candidate = candidate.detach().cpu().to(torch.float64)
        self.matrix = matrix.detach().cpu().to(torch.float64)
        self.lower_bound = lower_bound.detach().cpu().to(torch.float64)


def _tensor_sha256(value: Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def archive_qp_failure(
    destination: Path,
    failure: FeasibleCageProjectionFailure,
    *,
    last_valid_checkpoint: Path | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Atomically archive a rejected QP problem without touching the checkpoint."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable QP failure: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".qp-failure-", dir=destination.parent))
    try:
        problem_path = staging / "qp_problem.npz"
        upper_bound = torch.full_like(failure.lower_bound, float("inf"))
        np.savez_compressed(
            problem_path,
            candidate=failure.candidate.numpy(),
            matrix=failure.matrix.numpy(),
            lower_bound=failure.lower_bound.numpy(),
            upper_bound=upper_bound.numpy(),
        )
        checkpoint_path = str(last_valid_checkpoint) if last_valid_checkpoint is not None else None
        checkpoint_hash = (
            sha256_file(last_valid_checkpoint)
            if last_valid_checkpoint is not None and last_valid_checkpoint.is_file()
            else None
        )
        report: dict[str, Any] = {
            "schema_version": "certified_qp_failure.v1",
            "status": "rejected_without_fallback",
            "solver_result": failure.result.to_dict(),
            "problem": {
                "candidate_shape": list(failure.candidate.shape),
                "matrix_shape": list(failure.matrix.shape),
                "lower_bound_shape": list(failure.lower_bound.shape),
                "dtype": "float64",
                "candidate_sha256": _tensor_sha256(failure.candidate),
                "matrix_sha256": _tensor_sha256(failure.matrix),
                "lower_bound_sha256": _tensor_sha256(failure.lower_bound),
                "upper_bound_sha256": _tensor_sha256(upper_bound),
                "archive_sha256": sha256_file(problem_path),
            },
            "last_valid_checkpoint": {
                "path": checkpoint_path,
                "sha256": checkpoint_hash,
                "preserved": True,
            },
            "context": context,
            "fallback_used": False,
            "candidate_accepted": False,
        }
        write_json(staging / "failure_report.json", report)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


@dataclass(frozen=True)
class FeasibleDirectionReport:
    active_signed_constraint_count: int
    active_unsigned_constraint_count: int
    active_set_size: int
    iteration_count: int
    candidate_norm: float
    projected_norm: float
    trust_region_radius: float
    trust_scale: float
    minimum_linearized_slack: float
    solver_version: str
    solver_status: str
    certificate_maximum: float
    tautological_zero_row_count: int


@dataclass(frozen=True)
class LinearizedCageConstraints:
    matrix: Tensor
    lower_bound: Tensor
    active_signed_constraint_count: int
    active_unsigned_constraint_count: int


def _vertices_for_controls(cage: TrilinearDeformationCage, controls: Tensor) -> Tensor:
    flat = controls.reshape(-1, 3)
    offsets = (flat[cage.corner_indices] * cage.corner_weights[..., None]).sum(dim=1)
    return cage.reference_vertices + offsets


def _settings() -> dict[str, bool | float | int | str]:
    return {
        "dtype": "float64",
        "eps_abs": OSQP_EPSILON,
        "eps_rel": OSQP_EPSILON,
        "max_iter": OSQP_MAXIMUM_ITERATIONS,
        "polishing": True,
        "adaptive_rho": False,
        "warm_starting": False,
        "check_termination": 1,
        "certificate_tolerance": CERTIFICATE_TOLERANCE,
        "zero_row_tolerance": ZERO_ROW_TOLERANCE,
        "post_cast_feasibility_tolerance": POST_CAST_FEASIBILITY_TOLERANCE,
    }


def _empty_result(
    *,
    status: str,
    message: str,
    constraint_count: int,
    normalized_row_count: int,
    tautological_zero_row_count: int,
    contradictory_zero_row_count: int,
    trust_region_radius: float,
    solver_status: str = "not_run",
) -> CertifiedQPResult:
    return CertifiedQPResult(
        status=status,
        message=message,
        unclipped_direction=None,
        final_direction=None,
        multipliers=None,
        solver_name="osqp",
        solver_version=osqp.__version__,
        solver_status=solver_status,
        solver_status_value=None,
        iteration_count=0,
        setup_time_seconds=0.0,
        solve_time_seconds=0.0,
        polish_time_seconds=0.0,
        run_time_seconds=0.0,
        constraint_count=constraint_count,
        normalized_row_count=normalized_row_count,
        tautological_zero_row_count=tautological_zero_row_count,
        contradictory_zero_row_count=contradictory_zero_row_count,
        active_constraint_count=0,
        trust_region_radius=trust_region_radius,
        trust_scale=0.0,
        raw_certificate=None,
        scaled_certificate=None,
        unclipped_minimum_slack=float("nan"),
        final_minimum_slack=float("nan"),
        final_scaled_primal_residual=float("inf"),
        settings=_settings(),
    )


def _residuals(
    candidate: np.ndarray,
    solution: np.ndarray,
    matrix: np.ndarray,
    lower_bound: np.ndarray,
    multipliers: np.ndarray,
) -> tuple[KKTResiduals, KKTResiduals]:
    slack = matrix @ solution - lower_bound
    stationarity_vector = solution - candidate - matrix.T @ multipliers
    primal = float(np.maximum(-slack, 0.0).max(initial=0.0))
    dual = float(np.maximum(-multipliers, 0.0).max(initial=0.0))
    stationarity = float(np.abs(stationarity_vector).max(initial=0.0))
    complementarity = float(np.abs(multipliers * slack).max(initial=0.0))
    raw = KKTResiduals(primal, dual, stationarity, complementarity)
    primal_scale = max(
        1.0,
        float(np.abs(lower_bound).max(initial=0.0)),
        float(np.abs(matrix @ solution).max(initial=0.0)),
    )
    dual_scale = max(1.0, float(np.abs(multipliers).max(initial=0.0)))
    stationarity_scale = max(
        1.0,
        float(np.abs(solution - candidate).max(initial=0.0)),
        float(np.abs(matrix.T @ multipliers).max(initial=0.0)),
    )
    complementarity_scale = max(
        1.0,
        float(np.abs(multipliers).max(initial=0.0))
        * max(
            1.0,
            float(np.abs(lower_bound).max(initial=0.0)),
            float(np.abs(matrix @ solution).max(initial=0.0)),
        ),
    )
    scaled = KKTResiduals(
        primal / primal_scale,
        dual / dual_scale,
        stationarity / stationarity_scale,
        complementarity / complementarity_scale,
    )
    return raw, scaled


def project_halfspace_qp(
    candidate: Tensor,
    matrix: Tensor,
    lower_bound: Tensor,
    *,
    trust_region_radius: float,
) -> CertifiedQPResult:
    """Certify the Euclidean halfspace projection before radial trust clipping.

    Nonzero rows are positively normalized before OSQP. The independent KKT
    certificate is computed outside the solver in both original and normalized
    coordinates. Radial clipping is only a feasibility-preserving trust step;
    this function makes no claim that the clipped result is the optimum of a
    ball-constrained QP.
    """
    if candidate.ndim != 1:
        raise ValueError("candidate must be one-dimensional")
    if matrix.ndim != 2 or matrix.shape[1] != candidate.numel():
        raise ValueError("matrix must have shape (constraint_count, candidate_size)")
    if lower_bound.shape != (matrix.shape[0],):
        raise ValueError("lower_bound must match the constraint count")
    if not np.isfinite(trust_region_radius) or trust_region_radius <= 0:
        raise ValueError("trust_region_radius must be positive and finite")

    candidate_np = np.asarray(candidate.detach().cpu(), dtype=np.float64)
    matrix_np = np.asarray(matrix.detach().cpu(), dtype=np.float64)
    lower_np = np.asarray(lower_bound.detach().cpu(), dtype=np.float64)
    if not (
        np.isfinite(candidate_np).all()
        and np.isfinite(matrix_np).all()
        and np.isfinite(lower_np).all()
    ):
        return _empty_result(
            status="invalid_problem",
            message="candidate, matrix, and bounds must be finite",
            constraint_count=matrix.shape[0],
            normalized_row_count=0,
            tautological_zero_row_count=0,
            contradictory_zero_row_count=0,
            trust_region_radius=trust_region_radius,
        )

    row_norms = np.linalg.norm(matrix_np, axis=1)
    zero_rows = row_norms <= ZERO_ROW_TOLERANCE
    contradictory_zero_rows = zero_rows & (lower_np > 0.0)
    tautological_zero_rows = zero_rows & ~contradictory_zero_rows
    nonzero_rows = ~zero_rows
    contradiction_count = int(contradictory_zero_rows.sum())
    tautology_count = int(tautological_zero_rows.sum())
    if contradiction_count:
        return _empty_result(
            status="contradictory_zero_row",
            message="a zero coefficient row has a strictly positive lower bound",
            constraint_count=matrix.shape[0],
            normalized_row_count=int(nonzero_rows.sum()),
            tautological_zero_row_count=tautology_count,
            contradictory_zero_row_count=contradiction_count,
            trust_region_radius=trust_region_radius,
        )

    normalized_matrix = matrix_np[nonzero_rows] / row_norms[nonzero_rows, None]
    normalized_lower = lower_np[nonzero_rows] / row_norms[nonzero_rows]
    if normalized_matrix.shape[0] == 0:
        norm = float(np.linalg.norm(candidate_np))
        trust_scale = min(1.0, trust_region_radius / max(norm, 1e-300))
        final = candidate_np * trust_scale
        zero_certificate = KKTResiduals(0.0, 0.0, 0.0, 0.0)
        return CertifiedQPResult(
            status="certified",
            message="unconstrained projection; all rows are tautological zeros",
            unclipped_direction=torch.from_numpy(candidate_np.copy()),
            final_direction=torch.from_numpy(final.copy()),
            multipliers=torch.zeros(matrix.shape[0], dtype=torch.float64),
            solver_name="osqp",
            solver_version=osqp.__version__,
            solver_status="not_needed",
            solver_status_value=None,
            iteration_count=0,
            setup_time_seconds=0.0,
            solve_time_seconds=0.0,
            polish_time_seconds=0.0,
            run_time_seconds=0.0,
            constraint_count=matrix.shape[0],
            normalized_row_count=0,
            tautological_zero_row_count=tautology_count,
            contradictory_zero_row_count=0,
            active_constraint_count=0,
            trust_region_radius=trust_region_radius,
            trust_scale=trust_scale,
            raw_certificate=zero_certificate,
            scaled_certificate=zero_certificate,
            unclipped_minimum_slack=float("inf"),
            final_minimum_slack=float("inf"),
            final_scaled_primal_residual=0.0,
            settings=_settings(),
        )

    solver = osqp.OSQP()
    try:
        solver.setup(
            P=sparse.eye(candidate_np.size, format="csc"),
            q=-candidate_np,
            A=sparse.csc_matrix(normalized_matrix),
            l=normalized_lower,
            u=np.full(normalized_lower.shape, np.inf, dtype=np.float64),
            eps_abs=OSQP_EPSILON,
            eps_rel=OSQP_EPSILON,
            max_iter=OSQP_MAXIMUM_ITERATIONS,
            polishing=True,
            adaptive_rho=False,
            warm_starting=False,
            check_termination=1,
            verbose=False,
        )
        solution = solver.solve()
    except Exception as error:  # OSQP converts backend errors to several exception types.
        return _empty_result(
            status="solver_error",
            message=f"{type(error).__name__}: {error}",
            constraint_count=matrix.shape[0],
            normalized_row_count=normalized_matrix.shape[0],
            tautological_zero_row_count=tautology_count,
            contradictory_zero_row_count=0,
            trust_region_radius=trust_region_radius,
            solver_status="exception",
        )

    info = solution.info
    solver_status = str(info.status).strip().lower()
    base_kwargs: dict[str, Any] = {
        "solver_name": "osqp",
        "solver_version": osqp.__version__,
        "solver_status": solver_status,
        "solver_status_value": int(info.status_val),
        "iteration_count": int(info.iter),
        "setup_time_seconds": float(info.setup_time),
        "solve_time_seconds": float(info.solve_time),
        "polish_time_seconds": float(info.polish_time),
        "run_time_seconds": float(info.run_time),
        "constraint_count": matrix.shape[0],
        "normalized_row_count": normalized_matrix.shape[0],
        "tautological_zero_row_count": tautology_count,
        "contradictory_zero_row_count": 0,
        "trust_region_radius": trust_region_radius,
        "settings": _settings(),
    }
    if solver_status != "solved" or solution.x is None or solution.y is None:
        return CertifiedQPResult(
            status="solver_failed",
            message=f"OSQP did not return exact solved status: {solver_status}",
            unclipped_direction=None,
            final_direction=None,
            multipliers=None,
            active_constraint_count=0,
            trust_scale=0.0,
            raw_certificate=None,
            scaled_certificate=None,
            unclipped_minimum_slack=float("nan"),
            final_minimum_slack=float("nan"),
            final_scaled_primal_residual=float("inf"),
            **base_kwargs,
        )

    unclipped = np.asarray(solution.x, dtype=np.float64)
    normalized_multipliers = -np.asarray(solution.y, dtype=np.float64)
    original_multipliers = np.zeros(matrix_np.shape[0], dtype=np.float64)
    original_multipliers[nonzero_rows] = normalized_multipliers / row_norms[nonzero_rows]
    raw_certificate, _ = _residuals(
        candidate_np,
        unclipped,
        matrix_np,
        lower_np,
        original_multipliers,
    )
    _, scaled_certificate = _residuals(
        candidate_np,
        unclipped,
        normalized_matrix,
        normalized_lower,
        normalized_multipliers,
    )
    unclipped_slack = matrix_np @ unclipped - lower_np
    unclipped_minimum_slack = float(unclipped_slack.min(initial=np.inf))
    norm = float(np.linalg.norm(unclipped))
    trust_scale = min(1.0, trust_region_radius / max(norm, 1e-300))
    final = unclipped * trust_scale
    final_slack = matrix_np @ final - lower_np
    final_minimum_slack = float(final_slack.min(initial=np.inf))
    final_normalized_slack = normalized_matrix @ final - normalized_lower
    final_scale = max(
        1.0,
        float(np.abs(normalized_lower).max(initial=0.0)),
        float(np.abs(normalized_matrix @ final).max(initial=0.0)),
    )
    final_scaled_primal = float(
        np.maximum(-final_normalized_slack, 0.0).max(initial=0.0) / final_scale
    )
    failure_reasons: list[str] = []
    if scaled_certificate.maximum > CERTIFICATE_TOLERANCE:
        failure_reasons.append(
            f"scaled KKT residual {scaled_certificate.maximum:.3e} exceeds "
            f"{CERTIFICATE_TOLERANCE:.1e}"
        )
    if final_scaled_primal > CERTIFICATE_TOLERANCE:
        failure_reasons.append(
            f"radially clipped direction has scaled primal residual {final_scaled_primal:.3e}"
        )
    result_status = "certificate_failed" if failure_reasons else "certified"
    return CertifiedQPResult(
        status=result_status,
        message="; ".join(failure_reasons) if failure_reasons else "independent certificate passed",
        unclipped_direction=torch.from_numpy(unclipped.copy()),
        final_direction=torch.from_numpy(final.copy()),
        multipliers=torch.from_numpy(original_multipliers.copy()),
        active_constraint_count=int(
            np.count_nonzero(np.abs(normalized_matrix @ unclipped - normalized_lower) <= 1e-8)
        ),
        trust_scale=trust_scale,
        raw_certificate=raw_certificate,
        scaled_certificate=scaled_certificate,
        unclipped_minimum_slack=unclipped_minimum_slack,
        final_minimum_slack=final_minimum_slack,
        final_scaled_primal_residual=final_scaled_primal,
        **base_kwargs,
    )


def build_linearized_cage_constraints(
    cage: TrilinearDeformationCage,
    candidate_delta: Tensor,
    faces: Tensor,
    *,
    minimum_signed_area_ratio: float = 0.01,
    minimum_area_ratio: float = 0.1,
    active_slack_tolerance: float = 1e-4,
) -> LinearizedCageConstraints:
    """Build the frozen E4 local signed/unsigned face inequalities."""
    if candidate_delta.shape != cage.controls.shape:
        raise ValueError("candidate_delta must match cage controls")
    if active_slack_tolerance < 0:
        raise ValueError("active_slack_tolerance cannot be negative")
    current_vertices = cage.deformed_vertices()
    current_signed, current_unsigned = canonical_topology_quantities(
        cage.reference_vertices, current_vertices, faces
    )
    with torch.no_grad():
        candidate_vertices = _vertices_for_controls(cage, cage.controls + candidate_delta)
        candidate_signed, candidate_unsigned = canonical_topology_quantities(
            cage.reference_vertices, candidate_vertices, faces
        )
        signed_mask = (current_signed <= minimum_signed_area_ratio + active_slack_tolerance) | (
            candidate_signed <= minimum_signed_area_ratio + active_slack_tolerance
        )
        unsigned_mask = (current_unsigned <= minimum_area_ratio + active_slack_tolerance) | (
            candidate_unsigned <= minimum_area_ratio + active_slack_tolerance
        )
    constraints: list[Tensor] = []
    bounds: list[Tensor] = []
    for values, mask, floor in (
        (current_signed, signed_mask, minimum_signed_area_ratio),
        (current_unsigned, unsigned_mask, minimum_area_ratio),
    ):
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            gradient = torch.autograd.grad(
                values[index], cage.controls, retain_graph=True, create_graph=False
            )[0]
            constraints.append(gradient.detach().reshape(-1))
            bounds.append(values[index].detach().new_tensor(floor) - values[index].detach())

    flat_candidate = candidate_delta.detach().reshape(-1)
    if constraints:
        matrix = torch.stack(constraints)
        lower_bound = torch.stack(bounds)
    else:
        matrix = flat_candidate.new_empty((0, flat_candidate.numel()))
        lower_bound = flat_candidate.new_empty((0,))
    return LinearizedCageConstraints(
        matrix=matrix,
        lower_bound=lower_bound,
        active_signed_constraint_count=int(torch.count_nonzero(signed_mask)),
        active_unsigned_constraint_count=int(torch.count_nonzero(unsigned_mask)),
    )


def linearized_feasible_cage_direction(
    cage: TrilinearDeformationCage,
    candidate_delta: Tensor,
    faces: Tensor,
    *,
    minimum_signed_area_ratio: float = 0.01,
    minimum_area_ratio: float = 0.1,
    active_slack_tolerance: float = 1e-4,
    trust_region_radius: float | None = None,
) -> tuple[Tensor, FeasibleDirectionReport]:
    """Project an existing-cage candidate direction under local face constraints."""
    candidate_norm = float(torch.linalg.vector_norm(candidate_delta.detach()))
    radius = candidate_norm if trust_region_radius is None else trust_region_radius
    if radius <= 0:
        raise ValueError("the trust-region radius must be positive")
    problem = build_linearized_cage_constraints(
        cage,
        candidate_delta,
        faces,
        minimum_signed_area_ratio=minimum_signed_area_ratio,
        minimum_area_ratio=minimum_area_ratio,
        active_slack_tolerance=active_slack_tolerance,
    )
    flat_candidate = candidate_delta.detach().reshape(-1)
    result = project_halfspace_qp(
        flat_candidate,
        problem.matrix,
        problem.lower_bound,
        trust_region_radius=radius,
    )
    if not result.certified or result.final_direction is None or result.scaled_certificate is None:
        raise FeasibleCageProjectionFailure(
            result, flat_candidate, problem.matrix, problem.lower_bound
        )
    projected = result.final_direction.to(
        device=candidate_delta.device, dtype=candidate_delta.dtype
    )
    if problem.matrix.shape[0]:
        row_norms = torch.linalg.vector_norm(problem.matrix, dim=1)
        nonzero = row_norms > ZERO_ROW_TOLERANCE
        normalized_matrix = problem.matrix[nonzero] / row_norms[nonzero, None]
        normalized_lower = problem.lower_bound[nonzero] / row_norms[nonzero]
        scaled_values = normalized_matrix @ projected - normalized_lower
        post_cast_scale = max(
            1.0,
            float(normalized_lower.abs().max()) if normalized_lower.numel() else 0.0,
            float((normalized_matrix @ projected).abs().max())
            if normalized_matrix.shape[0]
            else 0.0,
        )
        post_cast_residual = (
            float(torch.relu(-scaled_values).max()) / post_cast_scale
            if scaled_values.numel()
            else 0.0
        )
        if post_cast_residual > POST_CAST_FEASIBILITY_TOLERANCE:
            failed = replace(
                result,
                status="post_cast_feasibility_failed",
                message=(
                    f"returned {candidate_delta.dtype} direction has scaled primal residual "
                    f"{post_cast_residual:.3e}"
                ),
            )
            raise FeasibleCageProjectionFailure(
                failed, flat_candidate, problem.matrix, problem.lower_bound
            )
        minimum_linearized_slack = float((problem.matrix @ projected - problem.lower_bound).min())
    else:
        minimum_linearized_slack = float("inf")
    report = FeasibleDirectionReport(
        active_signed_constraint_count=problem.active_signed_constraint_count,
        active_unsigned_constraint_count=problem.active_unsigned_constraint_count,
        active_set_size=result.active_constraint_count,
        iteration_count=result.iteration_count,
        candidate_norm=candidate_norm,
        projected_norm=float(torch.linalg.vector_norm(projected)),
        trust_region_radius=radius,
        trust_scale=result.trust_scale,
        minimum_linearized_slack=minimum_linearized_slack,
        solver_version=result.solver_version,
        solver_status=result.solver_status,
        certificate_maximum=result.scaled_certificate.maximum,
        tautological_zero_row_count=result.tautological_zero_row_count,
    )
    return projected.reshape_as(candidate_delta), report
