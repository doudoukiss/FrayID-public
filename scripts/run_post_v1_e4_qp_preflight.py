"""Run the fixed 1,000-case public E4 certified-QP numerical gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import clarabel  # type: ignore[import-untyped]
import numpy as np
import osqp  # type: ignore[import-untyped]
import torch
from scipy import sparse  # type: ignore[import-untyped]

from frayid.feasible_cage import CERTIFICATE_TOLERANCE, project_halfspace_qp
from frayid.io import write_json

EXPERIMENT_ID = "postv1_e04_certified_convex_projection_r01"
SEED = 20260831
CASE_COUNT = 1_000
DIRECTION_TOLERANCE = 1e-7
OBJECTIVE_TOLERANCE = 1e-7
REGRESSION_TOLERANCE = 1e-8
WALL_TIME_LIMIT_SECONDS = 2 * 60 * 60
OUTPUT_RELATIVE = (
    Path("outputs/canonical_clothed_surface_v1/post_v1") / EXPERIMENT_ID / "numerical_preflight"
)
CATEGORIES = (
    "redundant",
    "rescaled",
    "permuted",
    "near_collinear",
    "overdetermined",
    "zero_row",
    "zero_slack",
    "coefficient_scale",
)


@dataclass(frozen=True)
class QPCase:
    name: str
    category: str
    candidate: np.ndarray
    matrix: np.ndarray
    lower_bound: np.ndarray

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for value in (self.candidate, self.matrix, self.lower_bound):
            digest.update(np.asarray(value, dtype=np.float64).tobytes())
        return digest.hexdigest()


def _unit_rows(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    rows = rng.normal(size=(count, dimension))
    return np.asarray(rows / np.linalg.norm(rows, axis=1, keepdims=True), dtype=np.float64)


def _case(rng: np.random.Generator, index: int) -> QPCase:
    category = CATEGORIES[index % len(CATEGORIES)]
    dimension = int(rng.integers(2, 9))
    count = int(rng.integers(dimension + 1, 3 * dimension + 5))
    matrix = _unit_rows(rng, count, dimension)
    lower = -rng.uniform(0.0, 1.5, size=count)
    candidate = -3.0 * matrix[0] + rng.normal(scale=0.75, size=dimension)

    if category == "redundant":
        direction = _unit_rows(rng, 1, dimension)[0]
        scale = float(10.0 ** rng.uniform(-6.0, 6.0))
        matrix = np.vstack((direction, scale * direction, matrix))
        lower = np.concatenate((np.array([-1.0, 0.0]), lower))
        candidate = -3.0 * direction + rng.normal(scale=0.25, size=dimension)
    elif category == "rescaled":
        scales = 10.0 ** rng.uniform(-6.0, 6.0, size=count)
        matrix = matrix * scales[:, None]
        lower = lower * scales
    elif category == "permuted":
        order = rng.permutation(count)
        matrix = matrix[order]
        lower = lower[order]
    elif category == "near_collinear":
        direction = _unit_rows(rng, 1, dimension)[0]
        perturbation = rng.normal(scale=10.0 ** rng.uniform(-9.0, -5.0), size=(count, dimension))
        matrix = direction + perturbation
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        lower = -np.linspace(0.0, 1.0, count)
        candidate = -4.0 * direction + rng.normal(scale=0.2, size=dimension)
    elif category == "overdetermined":
        count = 5 * dimension + 11
        matrix = _unit_rows(rng, count, dimension)
        # Keep the deliberately overdetermined system well inside Slater's
        # condition; near-active degeneracy is exercised by its own category.
        lower = -rng.uniform(0.5, 1.5, size=count)
        candidate = -1.5 * matrix[0] + rng.normal(scale=0.3, size=dimension)
    elif category == "zero_row":
        matrix = np.vstack((matrix, np.zeros((1, dimension))))
        lower = np.concatenate((lower, np.array([-rng.uniform(0.0, 1.0)])))
    elif category == "zero_slack":
        lower[0] = 0.0
        candidate = -4.0 * matrix[0] + rng.normal(scale=0.2, size=dimension)
    elif category == "coefficient_scale":
        scales = np.geomspace(1e-6, 1e6, count)
        rng.shuffle(scales)
        matrix = matrix * scales[:, None]
        lower = lower * scales

    return QPCase(
        name=f"random_{index:04d}_{category}",
        category=category,
        candidate=np.asarray(candidate, dtype=np.float64),
        matrix=np.asarray(matrix, dtype=np.float64),
        lower_bound=np.asarray(lower, dtype=np.float64),
    )


def _oracle(case: QPCase) -> tuple[np.ndarray, float, str]:
    row_norms = np.linalg.norm(case.matrix, axis=1)
    active = row_norms > 1e-14
    matrix = case.matrix[active] / row_norms[active, None]
    lower = case.lower_bound[active] / row_norms[active]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 20_000
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 1e-10
    settings.tol_feas = 1e-10
    solver = clarabel.DefaultSolver(
        sparse.eye(case.candidate.size, format="csc"),
        -case.candidate,
        sparse.csc_matrix(-matrix),
        -lower,
        [clarabel.NonnegativeConeT(matrix.shape[0])],
        settings,
    )
    result = solver.solve()
    direction = np.asarray(result.x, dtype=np.float64)
    objective = 0.5 * float(np.square(direction - case.candidate).sum())
    return direction, objective, str(result.status)


def _fixed_cases(project_root: Path) -> list[tuple[QPCase, np.ndarray]]:
    fixture = json.loads((project_root / "tests/fixtures/normalized_qp_failure.json").read_text())
    return [
        (
            QPCase(
                "redundant_one_dimensional",
                "fixed_regression",
                np.array([-3.0]),
                np.array([[2.0], [1.0]]),
                np.array([-2.0, 0.0]),
            ),
            np.array([0.0]),
        ),
        (
            QPCase(
                "redundant_two_dimensional_order_01",
                "fixed_regression",
                np.array([-3.0, 2.0]),
                np.array([[2.0, 0.0], [1.0, 0.0]]),
                np.array([-2.0, 0.0]),
            ),
            np.array([0.0, 2.0]),
        ),
        (
            QPCase(
                "redundant_two_dimensional_order_10",
                "fixed_regression",
                np.array([-3.0, 2.0]),
                np.array([[1.0, 0.0], [2.0, 0.0]]),
                np.array([0.0, -2.0]),
            ),
            np.array([0.0, 2.0]),
        ),
        (
            QPCase(
                "normalized_five_halfspace",
                "fixed_regression",
                np.asarray(fixture["candidate"], dtype=np.float64),
                np.asarray(fixture["A"], dtype=np.float64),
                np.asarray(fixture["b"], dtype=np.float64),
            ),
            np.asarray(fixture["oracle"], dtype=np.float64),
        ),
    ]


def _evaluate(case: QPCase) -> dict[str, Any]:
    result = project_halfspace_qp(
        torch.from_numpy(case.candidate),
        torch.from_numpy(case.matrix),
        torch.from_numpy(case.lower_bound),
        trust_region_radius=1e9,
    )
    record: dict[str, Any] = {
        "name": case.name,
        "category": case.category,
        "input_sha256": case.digest,
        "osqp_status": result.status,
        "solver_status": result.solver_status,
        "iteration_count": result.iteration_count,
        "certificate_maximum": (
            result.scaled_certificate.maximum
            if result.scaled_certificate is not None
            else float("inf")
        ),
        "tautological_zero_rows": result.tautological_zero_row_count,
    }
    if not result.certified or result.final_direction is None:
        record["blocker"] = result.message
        return record
    oracle_direction, oracle_objective, oracle_status = _oracle(case)
    direction = result.final_direction.numpy()
    objective = 0.5 * float(np.square(direction - case.candidate).sum())
    record.update(
        {
            "clarabel_status": oracle_status,
            "direction_maximum_absolute_difference": float(
                np.abs(direction - oracle_direction).max(initial=0.0)
            ),
            "objective_absolute_difference": abs(objective - oracle_objective),
        }
    )
    return record


def run(destination: Path, *, source_revision: str, project_root: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable E4 preflight: {destination}")
    started = time.monotonic()
    rng = np.random.default_rng(SEED)
    random_cases = [_case(rng, index) for index in range(CASE_COUNT)]
    records = [_evaluate(case) for case in random_cases]
    fixed_records: list[dict[str, Any]] = []
    for case, expected in _fixed_cases(project_root):
        record = _evaluate(case)
        result = project_halfspace_qp(
            torch.from_numpy(case.candidate),
            torch.from_numpy(case.matrix),
            torch.from_numpy(case.lower_bound),
            trust_region_radius=1e9,
        )
        regression_error = (
            float(np.abs(result.final_direction.numpy() - expected).max(initial=0.0))
            if result.final_direction is not None
            else float("inf")
        )
        record["fixed_regression_maximum_absolute_difference"] = regression_error
        fixed_records.append(record)
    elapsed = time.monotonic() - started

    blockers: list[str] = []
    all_records = records + fixed_records
    if len(records) < CASE_COUNT:
        blockers.append("fewer_than_1000_random_cases")
    if any(record["osqp_status"] != "certified" for record in all_records):
        blockers.append("osqp_or_independent_certificate_failure")
    if any(record.get("clarabel_status") != "Solved" for record in all_records):
        blockers.append("clarabel_oracle_failure")
    if any(
        record.get("direction_maximum_absolute_difference", float("inf")) > DIRECTION_TOLERANCE
        for record in all_records
    ):
        blockers.append("direction_disagreement_above_1e-7")
    if any(
        record.get("objective_absolute_difference", float("inf")) > OBJECTIVE_TOLERANCE
        for record in all_records
    ):
        blockers.append("objective_disagreement_above_1e-7")
    if any(record["certificate_maximum"] > CERTIFICATE_TOLERANCE for record in all_records):
        blockers.append("certificate_above_1e-8")
    if any(
        record.get("fixed_regression_maximum_absolute_difference", 0.0) > REGRESSION_TOLERANCE
        for record in fixed_records
    ):
        blockers.append("fixed_regression_above_1e-8")
    if elapsed > WALL_TIME_LIMIT_SECONDS:
        blockers.append("local_wall_time_above_two_hours")

    report: dict[str, Any] = {
        "schema_version": "post_v1_e4_qp_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "source_revision": source_revision,
        "dirty_worktree": False,
        "seed": SEED,
        "random_case_count": len(records),
        "fixed_regression_count": len(fixed_records),
        "category_counts": dict(Counter(case.category for case in random_cases)),
        "osqp_version": osqp.__version__,
        "clarabel_version": clarabel.__version__,
        "direction_tolerance": DIRECTION_TOLERANCE,
        "objective_tolerance": OBJECTIVE_TOLERANCE,
        "fixed_regression_tolerance": REGRESSION_TOLERANCE,
        "certificate_tolerance": CERTIFICATE_TOLERANCE,
        "elapsed_seconds": elapsed,
        "wall_time_limit_seconds": WALL_TIME_LIMIT_SECONDS,
        "worst_direction_maximum_absolute_difference": max(
            record.get("direction_maximum_absolute_difference", float("inf"))
            for record in all_records
        ),
        "worst_objective_absolute_difference": max(
            record.get("objective_absolute_difference", float("inf")) for record in all_records
        ),
        "worst_certificate_maximum": max(record["certificate_maximum"] for record in all_records),
        "fixed_regressions": fixed_records,
        "cases": records,
        "execution": {
            "human_evidence_accesses": 0,
            "development_evaluations": 0,
            "optimizer_steps_on_project_data": 0,
            "modal_jobs_launched": 0,
            "automatic_paid_retries": 0,
            "sealed_test_accesses": 0,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e4-qp-preflight-", dir=destination.parent))
    try:
        write_json(staging / "numerical_preflight_report.json", report)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=OUTPUT_RELATIVE)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relevant = (
        "src/frayid/feasible_cage.py",
        "scripts/run_post_v1_e4_qp_preflight.py",
        "tests/test_feasible_cage.py",
        "tests/fixtures/normalized_qp_failure.json",
        "configs/evaluation/post_v1_e4_certified_convex_projection_r01.yaml",
        "pyproject.toml",
        "uv.lock",
    )
    dirty = (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *relevant],
            cwd=project_root,
            check=False,
        ).returncode
        != 0
    )
    if dirty and not args.allow_dirty:
        raise RuntimeError("Refusing official E4 numerical preflight from a dirty relevant tree")
    destination = args.destination
    if not destination.is_absolute():
        destination = project_root / destination
    report = run(destination, source_revision=revision, project_root=project_root)
    report["dirty_worktree"] = dirty
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
