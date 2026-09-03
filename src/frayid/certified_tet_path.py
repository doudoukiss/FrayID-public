from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from typing import Any, Literal

import numpy as np

from frayid.ambient_scaffold import (
    AmbientScaffoldV1,
    HarmonicDirectionV1,
    lift_carrier_proposal,
)

CERTIFIED_TET_STEP_SCHEMA_V1 = "certified_tet_step.v1"
DYADIC_ALPHA_BITS = 40
SIGNED_AREA_FLOOR = Fraction(1, 100)
UNSIGNED_AREA_SQUARED_FLOOR = Fraction(1, 100)


class CertificateTimeout(RuntimeError):
    pass


Polynomial = list[Fraction]
IntervalPolynomial = tuple[np.ndarray, np.ndarray]


def _trim(polynomial: Polynomial) -> Polynomial:
    result = list(polynomial)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _poly_add(first: Polynomial, second: Polynomial) -> Polynomial:
    result = [Fraction(0) for _ in range(max(len(first), len(second)))]
    for index, value in enumerate(first):
        result[index] += value
    for index, value in enumerate(second):
        result[index] += value
    return _trim(result)


def _poly_sub(first: Polynomial, second: Polynomial) -> Polynomial:
    return _poly_add(first, [-value for value in second])


def _poly_mul(first: Polynomial, second: Polynomial) -> Polynomial:
    result = [Fraction(0) for _ in range(len(first) + len(second) - 1)]
    for left, first_value in enumerate(first):
        for right, second_value in enumerate(second):
            result[left + right] += first_value * second_value
    return _trim(result)


def _poly_scale(polynomial: Polynomial, scalar: Fraction) -> Polynomial:
    return _trim([value * scalar for value in polynomial])


def _poly_value(polynomial: Polynomial, argument: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(polynomial):
        result = result * argument + coefficient
    return result


def _poly_derivative(polynomial: Polynomial) -> Polynomial:
    if len(polynomial) <= 1:
        return [Fraction(0)]
    return _trim([index * polynomial[index] for index in range(1, len(polynomial))])


def _poly_divmod(numerator: Polynomial, denominator: Polynomial) -> tuple[Polynomial, Polynomial]:
    numerator = _trim(numerator)
    denominator = _trim(denominator)
    if denominator == [0]:
        raise ZeroDivisionError("polynomial division by zero")
    if len(numerator) < len(denominator):
        return [Fraction(0)], numerator
    quotient = [Fraction(0) for _ in range(len(numerator) - len(denominator) + 1)]
    remainder = list(numerator)
    while len(remainder) >= len(denominator) and remainder != [0]:
        offset = len(remainder) - len(denominator)
        scale = remainder[-1] / denominator[-1]
        quotient[offset] = scale
        for index, value in enumerate(denominator):
            remainder[index + offset] -= scale * value
        remainder = _trim(remainder)
    return _trim(quotient), _trim(remainder)


def _sturm_sequence(polynomial: Polynomial) -> list[Polynomial]:
    polynomial = _trim(polynomial)
    if polynomial == [0]:
        raise ValueError("identically zero constraint polynomial")
    derivative = _poly_derivative(polynomial)
    if derivative == [0]:
        return [polynomial]
    sequence = [polynomial, derivative]
    while sequence[-1] != [0]:
        _, remainder = _poly_divmod(sequence[-2], sequence[-1])
        if remainder == [0]:
            break
        sequence.append([-value for value in remainder])
    return sequence


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _variations(sequence: list[Polynomial], argument: Fraction) -> int:
    signs = [_sign(_poly_value(polynomial, argument)) for polynomial in sequence]
    nonzero = [value for value in signs if value]
    return sum(first != second for first, second in pairwise(nonzero))


def _root_count(sequence: list[Polynomial], lower: Fraction, upper: Fraction) -> int:
    return _variations(sequence, lower) - _variations(sequence, upper)


def _first_root_lower_bound(polynomial: Polynomial, *, bits: int = 64) -> Fraction | None:
    polynomial = _trim(polynomial)
    if _poly_value(polynomial, Fraction(0)) <= 0:
        return Fraction(0)
    sequence = _sturm_sequence(polynomial)
    if _root_count(sequence, Fraction(0), Fraction(1)) <= 0:
        return None
    lower = Fraction(0)
    upper = Fraction(1)
    for _ in range(bits):
        midpoint = (lower + upper) / 2
        if _root_count(sequence, Fraction(0), midpoint) > 0:
            upper = midpoint
        else:
            lower = midpoint
    return lower


def certify_exact_polynomial_path(coefficients: Sequence[int | Fraction]) -> dict[str, Any]:
    """Exact public control for endpoint-only and between-sample fold regressions."""

    polynomial = _trim([Fraction(value) for value in coefficients])
    start = _poly_value(polynomial, Fraction(0))
    end = _poly_value(polynomial, Fraction(1))
    root = _first_root_lower_bound(polynomial) if start > 0 else Fraction(0)
    return {
        "schema_version": "exact_polynomial_path_control.v1",
        "status": "pass" if start > 0 and root is None else "fail",
        "start": str(start),
        "end": str(end),
        "endpoint_positive": bool(start > 0 and end > 0),
        "first_root_lower_bound": (
            None if root is None else {"numerator": root.numerator, "denominator": root.denominator}
        ),
    }


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_linear(start: float, end: float) -> Polynomial:
    first = _fraction(start)
    return [first, _fraction(end) - first]


def _exact_cross(
    first: tuple[Polynomial, Polynomial, Polynomial],
    second: tuple[Polynomial, Polynomial, Polynomial],
) -> tuple[Polynomial, Polynomial, Polynomial]:
    return (
        _poly_sub(_poly_mul(first[1], second[2]), _poly_mul(first[2], second[1])),
        _poly_sub(_poly_mul(first[2], second[0]), _poly_mul(first[0], second[2])),
        _poly_sub(_poly_mul(first[0], second[1]), _poly_mul(first[1], second[0])),
    )


def _exact_dot(
    first: tuple[Polynomial, Polynomial, Polynomial],
    second: tuple[Polynomial, Polynomial, Polynomial],
) -> Polynomial:
    result = [Fraction(0)]
    for left, right in zip(first, second, strict=True):
        result = _poly_add(result, _poly_mul(left, right))
    return result


def _exact_edge(
    start: np.ndarray, end: np.ndarray, first: int, second: int
) -> tuple[Polynomial, Polynomial, Polynomial]:
    return tuple(
        _poly_sub(
            _exact_linear(start[second, axis], end[second, axis]),
            _exact_linear(start[first, axis], end[first, axis]),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _exact_determinant_polynomial(
    start: np.ndarray, end: np.ndarray, tetrahedron: np.ndarray
) -> Polynomial:
    local_start = start[tetrahedron]
    local_end = end[tetrahedron]
    first = _exact_edge(local_start, local_end, 0, 1)
    second = _exact_edge(local_start, local_end, 0, 2)
    third = _exact_edge(local_start, local_end, 0, 3)
    return _exact_dot(_exact_cross(first, second), third)


def _exact_area_polynomials(
    start: np.ndarray, end: np.ndarray, face: np.ndarray
) -> tuple[Polynomial, Polynomial]:
    local_start = start[face]
    local_end = end[face]
    first = _exact_edge(local_start, local_end, 0, 1)
    second = _exact_edge(local_start, local_end, 0, 2)
    cross = _exact_cross(first, second)
    reference = ([cross[0][0]], [cross[1][0]], [cross[2][0]])
    reference_norm_squared = _exact_dot(reference, reference)
    signed = _poly_sub(
        _exact_dot(cross, reference),
        _poly_scale(reference_norm_squared, SIGNED_AREA_FLOOR),
    )
    unsigned = _poly_sub(
        _exact_dot(cross, cross),
        _poly_scale(reference_norm_squared, UNSIGNED_AREA_SQUARED_FLOOR),
    )
    return signed, unsigned


def _outward_add(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.nextafter(first[0] + second[0], -np.inf),
        np.nextafter(first[1] + second[1], np.inf),
    )


def _outward_sub(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.nextafter(first[0] - second[1], -np.inf),
        np.nextafter(first[1] - second[0], np.inf),
    )


def _outward_mul(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    products = np.stack(
        (
            first[0] * second[0],
            first[0] * second[1],
            first[1] * second[0],
            first[1] * second[1],
        ),
        axis=0,
    )
    return (
        np.nextafter(np.min(products, axis=0), -np.inf),
        np.nextafter(np.max(products, axis=0), np.inf),
    )


def _interval_constant(values: np.ndarray) -> IntervalPolynomial:
    return values[..., None].copy(), values[..., None].copy()


def _interval_poly_add(first: IntervalPolynomial, second: IntervalPolynomial) -> IntervalPolynomial:
    degree = max(first[0].shape[-1], second[0].shape[-1])
    shape = (*first[0].shape[:-1], degree)
    first_lo = np.zeros(shape, dtype=np.float64)
    first_hi = np.zeros(shape, dtype=np.float64)
    second_lo = np.zeros(shape, dtype=np.float64)
    second_hi = np.zeros(shape, dtype=np.float64)
    first_lo[..., : first[0].shape[-1]] = first[0]
    first_hi[..., : first[1].shape[-1]] = first[1]
    second_lo[..., : second[0].shape[-1]] = second[0]
    second_hi[..., : second[1].shape[-1]] = second[1]
    return _outward_add((first_lo, first_hi), (second_lo, second_hi))


def _interval_poly_sub(first: IntervalPolynomial, second: IntervalPolynomial) -> IntervalPolynomial:
    return _interval_poly_add(first, (-second[1], -second[0]))


def _interval_poly_mul(first: IntervalPolynomial, second: IntervalPolynomial) -> IntervalPolynomial:
    result_shape = (
        *first[0].shape[:-1],
        first[0].shape[-1] + second[0].shape[-1] - 1,
    )
    lower = np.zeros(result_shape, dtype=np.float64)
    upper = np.zeros(result_shape, dtype=np.float64)
    for left in range(first[0].shape[-1]):
        for right in range(second[0].shape[-1]):
            product = _outward_mul(
                (first[0][..., left], first[1][..., left]),
                (second[0][..., right], second[1][..., right]),
            )
            existing = (lower[..., left + right], upper[..., left + right])
            added = _outward_add(existing, product)
            lower[..., left + right] = added[0]
            upper[..., left + right] = added[1]
    return lower, upper


def _interval_scale_fraction(
    polynomial: IntervalPolynomial, scalar: Fraction
) -> IntervalPolynomial:
    approximation = float(scalar)
    constant = (
        np.full(polynomial[0].shape[:-1], np.nextafter(approximation, -np.inf)),
        np.full(polynomial[0].shape[:-1], np.nextafter(approximation, np.inf)),
    )
    lower = np.empty_like(polynomial[0])
    upper = np.empty_like(polynomial[1])
    for index in range(polynomial[0].shape[-1]):
        lower[..., index], upper[..., index] = _outward_mul(
            (polynomial[0][..., index], polynomial[1][..., index]), constant
        )
    return lower, upper


def _interval_linear(start: np.ndarray, end: np.ndarray) -> IntervalPolynomial:
    delta = _outward_sub((end, end), (start, start))
    return np.stack((start, delta[0]), axis=-1), np.stack((start, delta[1]), axis=-1)


def _interval_edge(
    start: np.ndarray, end: np.ndarray, first: int, second: int
) -> tuple[IntervalPolynomial, IntervalPolynomial, IntervalPolynomial]:
    return tuple(
        _interval_poly_sub(
            _interval_linear(start[:, second, axis], end[:, second, axis]),
            _interval_linear(start[:, first, axis], end[:, first, axis]),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _interval_cross(
    first: tuple[IntervalPolynomial, IntervalPolynomial, IntervalPolynomial],
    second: tuple[IntervalPolynomial, IntervalPolynomial, IntervalPolynomial],
) -> tuple[IntervalPolynomial, IntervalPolynomial, IntervalPolynomial]:
    return (
        _interval_poly_sub(
            _interval_poly_mul(first[1], second[2]),
            _interval_poly_mul(first[2], second[1]),
        ),
        _interval_poly_sub(
            _interval_poly_mul(first[2], second[0]),
            _interval_poly_mul(first[0], second[2]),
        ),
        _interval_poly_sub(
            _interval_poly_mul(first[0], second[1]),
            _interval_poly_mul(first[1], second[0]),
        ),
    )


def _interval_dot(
    first: tuple[IntervalPolynomial, IntervalPolynomial, IntervalPolynomial],
    second: tuple[IntervalPolynomial, IntervalPolynomial, IntervalPolynomial],
) -> IntervalPolynomial:
    result = _interval_poly_mul(first[0], second[0])
    for left, right in zip(first[1:], second[1:], strict=True):
        result = _interval_poly_add(result, _interval_poly_mul(left, right))
    return result


def _bernstein_lower_bounds(polynomial: IntervalPolynomial) -> np.ndarray:
    degree = polynomial[0].shape[-1] - 1
    lower = np.full(polynomial[0].shape[:-1], np.inf, dtype=np.float64)
    for bernstein_index in range(degree + 1):
        coefficient = _interval_constant(np.zeros(polynomial[0].shape[:-1]))
        for power in range(bernstein_index + 1):
            weight = Fraction(math.comb(bernstein_index, power), math.comb(degree, power))
            term = _interval_scale_fraction(
                (polynomial[0][..., power : power + 1], polynomial[1][..., power : power + 1]),
                weight,
            )
            coefficient = _interval_poly_add(coefficient, term)
        lower = np.minimum(lower, coefficient[0][..., 0])
    return lower


def _determinant_bernstein_lowers(
    start: np.ndarray, end: np.ndarray, tetrahedra: np.ndarray
) -> np.ndarray:
    local_start = start[tetrahedra]
    local_end = end[tetrahedra]
    first = _interval_edge(local_start, local_end, 0, 1)
    second = _interval_edge(local_start, local_end, 0, 2)
    third = _interval_edge(local_start, local_end, 0, 3)
    return _bernstein_lower_bounds(_interval_dot(_interval_cross(first, second), third))


def _area_bernstein_lowers(
    start: np.ndarray, end: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    local_start = start[faces]
    local_end = end[faces]
    first = _interval_edge(local_start, local_end, 0, 1)
    second = _interval_edge(local_start, local_end, 0, 2)
    cross = _interval_cross(first, second)
    reference = tuple((component[0][..., :1], component[1][..., :1]) for component in cross)
    reference_norm_squared = _interval_dot(reference, reference)  # type: ignore[arg-type]
    signed = _interval_poly_sub(
        _interval_dot(cross, reference),  # type: ignore[arg-type]
        _interval_scale_fraction(reference_norm_squared, SIGNED_AREA_FLOOR),
    )
    unsigned = _interval_poly_sub(
        _interval_dot(cross, cross),
        _interval_scale_fraction(reference_norm_squared, UNSIGNED_AREA_SQUARED_FLOOR),
    )
    return _bernstein_lower_bounds(signed), _bernstein_lower_bounds(unsigned)


@dataclass(frozen=True)
class _ConstraintScan:
    minimum_bernstein_lower: float
    fast_certificate_count: int
    exact_fallback_count: int
    first_root_lower_bound: Fraction | None
    initial_failure_count: int


@dataclass
class _MutableConstraintScan:
    minimum: float = math.inf
    fast: int = 0
    exact: int = 0
    root: Fraction | None = None
    initial: int = 0


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise CertificateTimeout("whole-path certificate exceeded its fixed deadline")


def _scan_tetrahedra(
    start: np.ndarray,
    end: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    deadline: float | None,
    chunk_size: int = 65_536,
) -> _ConstraintScan:
    fast = 0
    exact = 0
    initial_failures = 0
    earliest: Fraction | None = None
    minimum_lower = math.inf
    for offset in range(0, tetrahedra.shape[0], chunk_size):
        _check_deadline(deadline)
        chunk = tetrahedra[offset : offset + chunk_size]
        lowers = _determinant_bernstein_lowers(start, end, chunk)
        minimum_lower = min(minimum_lower, float(np.min(lowers, initial=math.inf)))
        certified = lowers > 0.0
        fast += int(np.count_nonzero(certified))
        for local in np.flatnonzero(~certified):
            _check_deadline(deadline)
            exact += 1
            polynomial = _exact_determinant_polynomial(start, end, chunk[local])
            if _poly_value(polynomial, Fraction(0)) <= 0:
                initial_failures += 1
                continue
            root = _first_root_lower_bound(polynomial)
            if root is not None and (earliest is None or root < earliest):
                earliest = root
    return _ConstraintScan(minimum_lower, fast, exact, earliest, initial_failures)


def _scan_faces(
    start: np.ndarray,
    end: np.ndarray,
    faces: np.ndarray,
    *,
    deadline: float | None,
    chunk_size: int = 65_536,
) -> tuple[_ConstraintScan, _ConstraintScan]:
    accumulators = [_MutableConstraintScan(), _MutableConstraintScan()]
    for offset in range(0, faces.shape[0], chunk_size):
        _check_deadline(deadline)
        chunk = faces[offset : offset + chunk_size]
        lower_sets = _area_bernstein_lowers(start, end, chunk)
        for kind, lowers in enumerate(lower_sets):
            accumulator = accumulators[kind]
            accumulator.minimum = min(accumulator.minimum, float(np.min(lowers, initial=math.inf)))
            certified = lowers > 0.0
            accumulator.fast += int(np.count_nonzero(certified))
            for local in np.flatnonzero(~certified):
                _check_deadline(deadline)
                accumulator.exact += 1
                polynomials = _exact_area_polynomials(start, end, chunk[local])
                polynomial = polynomials[kind]
                if _poly_value(polynomial, Fraction(0)) <= 0:
                    accumulator.initial += 1
                    continue
                root = _first_root_lower_bound(polynomial)
                current = accumulator.root
                if root is not None and (current is None or root < current):
                    accumulator.root = root
    return tuple(
        _ConstraintScan(
            minimum_bernstein_lower=value.minimum,
            fast_certificate_count=value.fast,
            exact_fallback_count=value.exact,
            first_root_lower_bound=value.root,
            initial_failure_count=value.initial,
        )
        for value in accumulators
    )  # type: ignore[return-value]


def _scan_report(scan: _ConstraintScan) -> dict[str, Any]:
    root = scan.first_root_lower_bound
    return {
        "minimum_outward_bernstein_lower_bound": scan.minimum_bernstein_lower,
        "fast_interval_certificate_count": scan.fast_certificate_count,
        "exact_rational_fallback_count": scan.exact_fallback_count,
        "initial_failure_count": scan.initial_failure_count,
        "first_root_lower_bound": (
            None if root is None else {"numerator": root.numerator, "denominator": root.denominator}
        ),
    }


@dataclass(frozen=True)
class CertifiedTetStepV1:
    proposed_carrier_displacement: np.ndarray
    global_direction: np.ndarray
    accepted_vertices: np.ndarray
    accepted_alpha: float
    retained_displacement_ratio: float
    harmonic_report: dict[str, Any]
    determinant_report: dict[str, Any]
    signed_area_report: dict[str, Any]
    unsigned_area_report: dict[str, Any]
    status: Literal["pass", "fail", "unknown"]
    blockers: tuple[str, ...]
    elapsed_seconds: float
    endpoint_audit: dict[str, Any] | None = None

    @property
    def decision_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            np.ascontiguousarray(self.proposed_carrier_displacement, dtype="<f8").tobytes()
        )
        digest.update(np.ascontiguousarray(self.global_direction, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(self.accepted_vertices, dtype="<f8").tobytes())
        digest.update(float(self.accepted_alpha).hex().encode())
        digest.update(self.status.encode())
        digest.update(json.dumps(self.blockers, separators=(",", ":")).encode())
        return digest.hexdigest()

    def with_endpoint_audit(self, audit: dict[str, Any]) -> CertifiedTetStepV1:
        blockers = list(self.blockers)
        status: Literal["pass", "fail", "unknown"] = self.status
        if audit.get("status") != "pass":
            blockers.append("independent_endpoint_exact_audit")
            status = "fail"
        return replace(self, endpoint_audit=dict(audit), blockers=tuple(blockers), status=status)

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": CERTIFIED_TET_STEP_SCHEMA_V1,
            "status": self.status,
            "accepted_alpha": self.accepted_alpha,
            "accepted_alpha_hex": self.accepted_alpha.hex(),
            "retained_displacement_ratio": self.retained_displacement_ratio,
            "proposed_carrier_displacement_sha256": hashlib.sha256(
                np.ascontiguousarray(self.proposed_carrier_displacement, dtype="<f8").tobytes()
            ).hexdigest(),
            "global_direction_sha256": hashlib.sha256(
                np.ascontiguousarray(self.global_direction, dtype="<f8").tobytes()
            ).hexdigest(),
            "accepted_vertices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.accepted_vertices, dtype="<f8").tobytes()
            ).hexdigest(),
            "decision_sha256": self.decision_sha256,
            "harmonic": self.harmonic_report,
            "determinants": self.determinant_report,
            "signed_area": self.signed_area_report,
            "unsigned_area": self.unsigned_area_report,
            "independent_endpoint_exact_audit": self.endpoint_audit,
            "elapsed_seconds": self.elapsed_seconds,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class CertifiedPLPathV1:
    """Exact complete-path certificate for a serialized volumetric PL map."""

    accepted_volume_vertices: np.ndarray
    accepted_surface_vertices: np.ndarray
    accepted_alpha: float
    determinant_report: dict[str, Any]
    signed_area_report: dict[str, Any]
    unsigned_area_report: dict[str, Any]
    status: Literal["pass", "fail", "unknown"]
    blockers: tuple[str, ...]
    elapsed_seconds: float

    @property
    def decision_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.accepted_volume_vertices, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(self.accepted_surface_vertices, dtype="<f8").tobytes())
        digest.update(float(self.accepted_alpha).hex().encode())
        digest.update(self.status.encode())
        digest.update(json.dumps(self.blockers, separators=(",", ":")).encode())
        return digest.hexdigest()

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "certified_pl_path.v1",
            "status": self.status,
            "accepted_alpha": self.accepted_alpha,
            "accepted_alpha_hex": self.accepted_alpha.hex(),
            "dyadic_alpha_bits": DYADIC_ALPHA_BITS,
            "determinants": self.determinant_report,
            "signed_area": self.signed_area_report,
            "unsigned_area": self.unsigned_area_report,
            "accepted_volume_vertices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.accepted_volume_vertices, dtype="<f8").tobytes()
            ).hexdigest(),
            "accepted_surface_vertices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.accepted_surface_vertices, dtype="<f8").tobytes()
            ).hexdigest(),
            "decision_sha256": self.decision_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "blockers": list(self.blockers),
        }


def _earliest_root(scans: tuple[_ConstraintScan, ...]) -> Fraction | None:
    roots = [
        scan.first_root_lower_bound for scan in scans if scan.first_root_lower_bound is not None
    ]
    return min(roots) if roots else None


def _accepted_alpha(root: Fraction | None) -> Fraction:
    if root is None:
        return Fraction(1)
    scaled = root * Fraction(4, 5) * (1 << DYADIC_ALPHA_BITS)
    numerator = scaled.numerator // scaled.denominator
    return Fraction(numerator, 1 << DYADIC_ALPHA_BITS)


def certify_piecewise_affine_path(
    volume_vertices: np.ndarray,
    tetrahedra: np.ndarray,
    volume_direction: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    surface_direction: np.ndarray,
    *,
    timeout_seconds: float | None = 60.0,
) -> CertifiedPLPathV1:
    """Certify one shared dyadic step for a full-box PL map and exact surface image."""

    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    volume = np.asarray(volume_vertices, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    direction = np.asarray(volume_direction, dtype=np.float64)
    surface = np.asarray(surface_vertices, dtype=np.float64)
    faces = np.asarray(surface_faces, dtype=np.int64)
    surface_delta = np.asarray(surface_direction, dtype=np.float64)
    if volume.ndim != 2 or volume.shape[1] != 3 or direction.shape != volume.shape:
        raise ValueError("volume vertices and direction must have finite shape [V,3]")
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.size == 0:
        raise ValueError("tetrahedra must have nonempty shape [T,4]")
    if surface.ndim != 2 or surface.shape[1] != 3 or surface_delta.shape != surface.shape:
        raise ValueError("surface vertices and direction must have finite shape [S,3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        raise ValueError("surface faces must have nonempty shape [F,3]")
    if not all(np.isfinite(values).all() for values in (volume, direction, surface, surface_delta)):
        raise ValueError("piecewise-affine path coordinates must be finite")
    if np.any(cells < 0) or np.any(cells >= volume.shape[0]):
        raise ValueError("tetrahedron index is out of range")
    if np.any(faces < 0) or np.any(faces >= surface.shape[0]):
        raise ValueError("surface face index is out of range")
    full_volume_endpoint = np.asarray(volume + direction, dtype=np.float64)
    full_surface_endpoint = np.asarray(surface + surface_delta, dtype=np.float64)
    blockers: list[str] = []
    try:
        determinant_full = _scan_tetrahedra(volume, full_volume_endpoint, cells, deadline=deadline)
        signed_full, unsigned_full = _scan_faces(
            surface, full_surface_endpoint, faces, deadline=deadline
        )
        full_scans = (determinant_full, signed_full, unsigned_full)
        if any(scan.initial_failure_count for scan in full_scans):
            blockers.append("invalid_initial_path_premise")
        alpha_fraction = _accepted_alpha(_earliest_root(full_scans))
        alpha = float(alpha_fraction)
        accepted_volume = np.asarray(volume + alpha * direction, dtype=np.float64)
        accepted_surface = np.asarray(surface + alpha * surface_delta, dtype=np.float64)
        determinant = _scan_tetrahedra(volume, accepted_volume, cells, deadline=deadline)
        signed, unsigned = _scan_faces(surface, accepted_surface, faces, deadline=deadline)
        accepted_scans = (determinant, signed, unsigned)
        if any(
            scan.initial_failure_count or scan.first_root_lower_bound is not None
            for scan in accepted_scans
        ):
            blockers.append("accepted_path_not_strictly_certified")
        status: Literal["pass", "fail", "unknown"] = "pass" if not blockers else "fail"
        return CertifiedPLPathV1(
            accepted_volume_vertices=accepted_volume,
            accepted_surface_vertices=accepted_surface,
            accepted_alpha=alpha,
            determinant_report={
                "full_proposal_scan": _scan_report(determinant_full),
                "accepted_path_scan": _scan_report(determinant),
            },
            signed_area_report={
                "floor": float(SIGNED_AREA_FLOOR),
                "full_proposal_scan": _scan_report(signed_full),
                "accepted_path_scan": _scan_report(signed),
            },
            unsigned_area_report={
                "floor": math.sqrt(float(UNSIGNED_AREA_SQUARED_FLOOR)),
                "full_proposal_scan": _scan_report(unsigned_full),
                "accepted_path_scan": _scan_report(unsigned),
            },
            status=status,
            blockers=tuple(blockers),
            elapsed_seconds=time.monotonic() - started,
        )
    except CertificateTimeout as error:
        return CertifiedPLPathV1(
            accepted_volume_vertices=volume.copy(),
            accepted_surface_vertices=surface.copy(),
            accepted_alpha=0.0,
            determinant_report={},
            signed_area_report={"floor": float(SIGNED_AREA_FLOOR)},
            unsigned_area_report={"floor": math.sqrt(float(UNSIGNED_AREA_SQUARED_FLOOR))},
            status="unknown",
            blockers=(str(error),),
            elapsed_seconds=time.monotonic() - started,
        )


def certify_tet_step(
    scaffold: AmbientScaffoldV1,
    source_carrier_faces: np.ndarray,
    source_proposal: np.ndarray,
    harmonic: HarmonicDirectionV1,
    *,
    minimum_retained_displacement_ratio: float = 0.25,
    timeout_seconds: float | None = 60.0,
) -> CertifiedTetStepV1:
    """Choose and prove one shared dyadic step for the actual serialized PL path."""

    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    scaffold.validate()
    proposed = np.asarray(source_proposal, dtype=np.float64)
    direction = np.asarray(harmonic.displacement, dtype=np.float64)
    if direction.shape != scaffold.vertices.shape or not np.isfinite(direction).all():
        raise ValueError("global direction must be finite with shape [ambient vertices, 3]")
    lifted = lift_carrier_proposal(scaffold, source_carrier_faces, proposed)
    if not np.array_equal(direction[scaffold.carrier_vertex_indices], lifted):
        raise ValueError("global direction does not retain the complete carrier proposal")
    full_endpoint = np.asarray(scaffold.vertices + direction, dtype=np.float64)
    blockers: list[str] = []
    try:
        determinant_full = _scan_tetrahedra(
            scaffold.vertices, full_endpoint, scaffold.tetrahedra, deadline=deadline
        )
        signed_full, unsigned_full = _scan_faces(
            scaffold.vertices, full_endpoint, scaffold.carrier_faces, deadline=deadline
        )
        if any(
            scan.initial_failure_count for scan in (determinant_full, signed_full, unsigned_full)
        ):
            blockers.append("invalid_initial_path_premise")
        alpha_fraction = _accepted_alpha(
            _earliest_root((determinant_full, signed_full, unsigned_full))
        )
        alpha = float(alpha_fraction)
        accepted = np.asarray(scaffold.vertices + alpha * direction, dtype=np.float64)
        determinant = _scan_tetrahedra(
            scaffold.vertices, accepted, scaffold.tetrahedra, deadline=deadline
        )
        signed, unsigned = _scan_faces(
            scaffold.vertices, accepted, scaffold.carrier_faces, deadline=deadline
        )
        if any(
            scan.initial_failure_count or scan.first_root_lower_bound is not None
            for scan in (determinant, signed, unsigned)
        ):
            blockers.append("accepted_path_not_strictly_certified")
        proposed_norm = float(np.linalg.norm(lifted))
        accepted_norm = float(
            np.linalg.norm(
                accepted[scaffold.carrier_vertex_indices]
                - scaffold.vertices[scaffold.carrier_vertex_indices]
            )
        )
        retained = accepted_norm / proposed_norm if proposed_norm > 0.0 else 0.0
        if proposed_norm <= 0.0:
            blockers.append("nonpositive_proposed_carrier_motion")
        if retained < minimum_retained_displacement_ratio:
            blockers.append("carrier_motion_retention")
        status: Literal["pass", "fail", "unknown"] = "pass" if not blockers else "fail"
        return CertifiedTetStepV1(
            proposed_carrier_displacement=proposed.copy(),
            global_direction=direction.copy(),
            accepted_vertices=accepted,
            accepted_alpha=alpha,
            retained_displacement_ratio=retained,
            harmonic_report=harmonic.report(),
            determinant_report={
                "full_proposal_scan": _scan_report(determinant_full),
                "accepted_path_scan": _scan_report(determinant),
            },
            signed_area_report={
                "floor": float(SIGNED_AREA_FLOOR),
                "full_proposal_scan": _scan_report(signed_full),
                "accepted_path_scan": _scan_report(signed),
            },
            unsigned_area_report={
                "floor": math.sqrt(float(UNSIGNED_AREA_SQUARED_FLOOR)),
                "full_proposal_scan": _scan_report(unsigned_full),
                "accepted_path_scan": _scan_report(unsigned),
            },
            status=status,
            blockers=tuple(blockers),
            elapsed_seconds=time.monotonic() - started,
        )
    except CertificateTimeout as error:
        return CertifiedTetStepV1(
            proposed_carrier_displacement=proposed.copy(),
            global_direction=direction.copy(),
            accepted_vertices=scaffold.vertices.copy(),
            accepted_alpha=0.0,
            retained_displacement_ratio=0.0,
            harmonic_report=harmonic.report(),
            determinant_report={},
            signed_area_report={"floor": float(SIGNED_AREA_FLOOR)},
            unsigned_area_report={"floor": math.sqrt(float(UNSIGNED_AREA_SQUARED_FLOOR))},
            status="unknown",
            blockers=(str(error),),
            elapsed_seconds=time.monotonic() - started,
        )
