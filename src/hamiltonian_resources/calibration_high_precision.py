"""Arbitrary-precision reference kernels for offline MPF calibration.

This module is deliberately absent from runtime planning imports.  Its mpmath
dependency is optional and is loaded only when a calibration function runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import time as wall_time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .hamiltonians import PauliHamiltonian
from .multiproduct import (
    MPFSchedule,
    mpf_richardson_diagnostics,
)
from .trotter import suzuki_group_factors

if TYPE_CHECKING:
    from mpmath.ctx_mp import MPContext
    from mpmath.matrices.matrices import _matrix


@dataclass(frozen=True)
class HighPrecisionOperatorNormEstimate:
    """One converged or precision-limited MPF operator-norm observation."""

    value_decimal: str
    backend: str
    backend_version: str
    decimal_digits: int
    attempted_digits: tuple[int, ...]
    converged: bool
    relative_precision_change: float
    eigensystem_residual: float
    interval_relative_width: float
    interval_certified: bool
    branch_count: int
    formal_order: int
    segments: int
    schedule: MPFSchedule
    symmetry_reduction: str
    schedule_digest: str
    term_order_digest: str
    wall_seconds: float

    @property
    def value(self) -> float:
        """Return a binary64 compatibility view of the high-precision value."""
        return float(self.value_decimal)


def _require_mpmath() -> "MPContext":
    try:
        import mpmath as mp
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "high-precision calibration requires the 'calibration' optional extra"
        ) from error
    return mp


def _require_flint() -> Any:
    try:
        import flint
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "FLINT calibration requires the 'calibration-flint' optional extra"
        ) from error
    return flint


def _mp_number(mp: "MPContext", value: int | float | str) -> Any:
    if isinstance(value, int):
        return mp.mpf(value)
    return mp.mpf(str(value))


def _matrix_max_abs(mp: "MPContext", matrix: "_matrix") -> Any:
    return max((abs(value) for value in matrix), default=mp.mpf("0"))


def _matrix_power(mp: "MPContext", matrix: "_matrix", exponent: int) -> "_matrix":
    if exponent < 0:
        raise ValueError("matrix exponent must be nonnegative")
    result = mp.eye(matrix.rows)
    factor = matrix.copy()
    power = exponent
    while power:
        if power & 1:
            result = factor * result
        power >>= 1
        if power:
            factor = factor * factor
    return result


def _pauli_matrix(mp: "MPContext", label: str) -> "_matrix":
    dimension = 2 ** len(label)
    x_mask = 0
    z_mask = 0
    y_count = 0
    for offset, pauli in enumerate(reversed(label)):
        if pauli in "XY":
            x_mask |= 1 << offset
        if pauli in "YZ":
            z_mask |= 1 << offset
        if pauli == "Y":
            y_count += 1
    result = mp.zeros(dimension)
    y_phase = mp.j**y_count
    for column in range(dimension):
        row = column ^ x_mask
        parity = (column & z_mask).bit_count() & 1
        result[row, column] = y_phase * (-1 if parity else 1)
    return result


def _hamiltonian_and_terms(
    mp: "MPContext",
    hamiltonian: PauliHamiltonian,
) -> tuple["_matrix", tuple[tuple["_matrix", Any], ...]]:
    dimension = 2**hamiltonian.num_qubits
    dense_hamiltonian = mp.zeros(dimension)
    terms: list[tuple["_matrix", Any]] = []
    for label, coefficient in hamiltonian.terms:
        pauli = _pauli_matrix(mp, label)
        weight = _mp_number(mp, coefficient)
        dense_hamiltonian += weight * pauli
        terms.append((pauli, weight))
    return dense_hamiltonian, tuple(terms)


def _strang_step(
    mp: "MPContext",
    terms: tuple[tuple["_matrix", Any], ...],
    step_time: Any,
) -> "_matrix":
    result = mp.eye(terms[0][0].rows)
    for group, factor in suzuki_group_factors(len(terms), 2):
        pauli, weight = terms[group]
        angle = _mp_number(mp, factor) * step_time * weight
        unitary = mp.cos(angle) * mp.eye(pauli.rows) - mp.j * mp.sin(angle) * pauli
        result = unitary * result
    return result


def _exact_evolution(
    mp: "MPContext",
    hamiltonian_matrix: "_matrix",
    time: Any,
) -> tuple["_matrix", Any]:
    eigenvalues, eigenvectors = mp.eigh(hamiltonian_matrix)
    phases = mp.diag([mp.exp(-mp.j * time * value) for value in eigenvalues])
    exact = eigenvectors * phases * eigenvectors.H
    residual_matrix = (
        hamiltonian_matrix * eigenvectors
        - eigenvectors * mp.diag(list(eigenvalues))
    )
    scale = max(_matrix_max_abs(mp, hamiltonian_matrix), mp.mpf("1"))
    return exact, _matrix_max_abs(mp, residual_matrix) / scale


def _mpmath_operator_norm_at_precision(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    branch_count: int,
    *,
    schedule: MPFSchedule,
    decimal_digits: int,
) -> tuple[str, float, float, bool]:
    mp = _require_mpmath()
    with mp.workdps(decimal_digits):
        time_mp = _mp_number(mp, time)
        hamiltonian_matrix, terms = _hamiltonian_and_terms(mp, hamiltonian)
        exact, eigensystem_residual = _exact_evolution(
            mp,
            hamiltonian_matrix,
            time_mp,
        )
        diagnostics = mpf_richardson_diagnostics(branch_count, schedule=schedule)
        step_time = time_mp / segments
        mpf_step = mp.zeros(hamiltonian_matrix.rows)
        for coefficient, exponent in zip(
            diagnostics.coefficients,
            diagnostics.exponents,
            strict=True,
        ):
            base = _strang_step(mp, terms, step_time / exponent)
            coefficient_mp = mp.mpf(coefficient.numerator) / coefficient.denominator
            mpf_step += coefficient_mp * _matrix_power(mp, base, exponent)
        approximate = _matrix_power(mp, mpf_step, segments)
        difference = approximate - exact
        gram = difference.H * difference
        eigenvalues = mp.eigh(gram, eigvals_only=True)
        largest = max(mp.re(value) for value in eigenvalues)
        negative_tolerance = mp.power(10, -(decimal_digits - 8))
        if largest < -negative_tolerance:
            raise ArithmeticError("D^dagger D has a negative eigenvalue at working precision")
        value = mp.sqrt(max(largest, mp.mpf("0")))
        return mp.nstr(value, n=decimal_digits), float(eigensystem_residual), 0.0, False


def _flint_identity(flint: Any, dimension: int) -> Any:
    result = flint.acb_mat(dimension, dimension)
    for index in range(dimension):
        result[index, index] = 1
    return result


def _flint_pauli_matrix(flint: Any, label: str) -> Any:
    dimension = 2 ** len(label)
    x_mask = 0
    z_mask = 0
    y_count = 0
    for offset, pauli in enumerate(reversed(label)):
        if pauli in "XY":
            x_mask |= 1 << offset
        if pauli in "YZ":
            z_mask |= 1 << offset
        if pauli == "Y":
            y_count += 1
    result = flint.acb_mat(dimension, dimension)
    y_phase = flint.acb(0, 1) ** y_count
    for column in range(dimension):
        row = column ^ x_mask
        parity = (column & z_mask).bit_count() & 1
        result[row, column] = y_phase * (-1 if parity else 1)
    return result


def _flint_pauli_block_matrix(
    flint: Any,
    label: str,
    basis_states: tuple[int, ...],
    basis_index: dict[int, int],
) -> Any:
    dimension = len(basis_states)
    x_mask = 0
    z_mask = 0
    y_count = 0
    for offset, pauli in enumerate(reversed(label)):
        if pauli in "XY":
            x_mask |= 1 << offset
        if pauli in "YZ":
            z_mask |= 1 << offset
        if pauli == "Y":
            y_count += 1
    if x_mask.bit_count() & 1:
        raise ValueError(f"Pauli term {label!r} does not preserve the selected parity")
    result = flint.acb_mat(dimension, dimension)
    y_phase = flint.acb(0, 1) ** y_count
    for column, state in enumerate(basis_states):
        row_state = state ^ x_mask
        row = basis_index[row_state]
        parity = (state & z_mask).bit_count() & 1
        result[row, column] = y_phase * (-1 if parity else 1)
    return result


def _parity_labels_and_coefficients(
    hamiltonian: PauliHamiltonian,
) -> tuple[tuple[str, float], ...]:
    metadata = hamiltonian.model_metadata
    if metadata is None:
        raise ValueError("parity reduction requires built-in Hamiltonian metadata")
    if metadata.model == "heisenberg_chain":
        return hamiltonian.terms
    if metadata.model == "transverse_field_ising":
        transformed: list[tuple[str, float]] = []
        translation = {"I": "I", "X": "Z", "Y": "Y", "Z": "X"}
        for label, coefficient in hamiltonian.terms:
            transformed.append(
                (
                    "".join(translation[value] for value in label),
                    coefficient * (-1 if label.count("Y") & 1 else 1),
                )
            )
        return tuple(transformed)
    raise ValueError(f"no parity reduction is registered for {metadata.model!r}")


def _flint_matrix_power(flint: Any, matrix: Any, exponent: int) -> Any:
    result = _flint_identity(flint, matrix.nrows())
    factor = matrix
    power = exponent
    while power:
        if power & 1:
            result = factor * result
        power >>= 1
        if power:
            factor = factor * factor
    return result


def _flint_strang_step(
    flint: Any,
    terms: tuple[tuple[Any, Any], ...],
    step_time: Any,
) -> Any:
    identity = _flint_identity(flint, terms[0][0].nrows())
    result = identity
    minus_i = flint.acb(0, -1)
    for group, factor in suzuki_group_factors(len(terms), 2):
        pauli, weight = terms[group]
        angle = flint.arb(str(factor)) * step_time * weight
        unitary = angle.cos() * identity + minus_i * angle.sin() * pauli
        result = unitary * result
    return result


def _flatten_eigenvalues(values: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten_eigenvalues(value))
        else:
            flattened.append(value)
    return flattened


def _flint_sector_norm(
    flint: Any,
    terms: tuple[tuple[Any, Any], ...],
    time_ball: Any,
    segments: int,
    diagnostics: Any,
) -> tuple[Any, float, bool]:
    dimension = terms[0][0].nrows()
    hamiltonian_matrix = flint.acb_mat(dimension, dimension)
    for pauli, weight in terms:
        hamiltonian_matrix += weight * pauli
    exact = (flint.acb(0, -1) * time_ball * hamiltonian_matrix).exp()
    step_time = time_ball / segments
    mpf_step = flint.acb_mat(dimension, dimension)
    for coefficient, exponent in zip(
        diagnostics.coefficients,
        diagnostics.exponents,
        strict=True,
    ):
        base = _flint_strang_step(flint, terms, step_time / exponent)
        coefficient_ball = flint.arb(coefficient.numerator) / coefficient.denominator
        mpf_step += coefficient_ball * _flint_matrix_power(
            flint,
            base,
            exponent,
        )
    approximate = _flint_matrix_power(flint, mpf_step, segments)
    difference = approximate - exact
    gram = difference.conjugate().transpose() * difference
    try:
        eigenvalues = _flatten_eigenvalues(gram.eig(multiple=True))
        interval_certified = True
    except ValueError:
        eigenvalues = _flatten_eigenvalues(gram.eig(algorithm="approx"))
        interval_certified = False
    largest = max(eigenvalues, key=lambda value: float(value.real.mid()))
    if interval_certified and not largest.imag.contains(0):
        raise ArithmeticError("D^dagger D eigenvalue enclosure is not real")
    if (
        interval_certified
        and float(largest.real.mid()) < 0
        and not largest.real.contains(0)
    ):
        raise ArithmeticError("D^dagger D has a negative eigenvalue enclosure")
    nonnegative = largest.real if float(largest.real.mid()) >= 0 else flint.arb(0)
    value = nonnegative.sqrt()
    midpoint = abs(float(value.mid()))
    radius = float(value.rad())
    relative_width = math.inf if midpoint == 0 else 2 * radius / midpoint
    return value, relative_width, interval_certified


def _flint_operator_norm_at_precision(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    branch_count: int,
    *,
    schedule: MPFSchedule,
    decimal_digits: int,
    symmetry_reduction: Literal["none", "parity"],
) -> tuple[str, float, float, bool]:
    flint = _require_flint()
    previous_dps = flint.ctx.dps
    flint.ctx.dps = decimal_digits
    try:
        time_ball = flint.arb(str(time))
        diagnostics = mpf_richardson_diagnostics(branch_count, schedule=schedule)
        if symmetry_reduction == "none":
            term_sets = (
                tuple(
                    (
                        _flint_pauli_matrix(flint, label),
                        flint.arb(str(coefficient)),
                    )
                    for label, coefficient in hamiltonian.terms
                ),
            )
        elif symmetry_reduction == "parity":
            labels = _parity_labels_and_coefficients(hamiltonian)
            full_dimension = 2**hamiltonian.num_qubits
            blocks: list[tuple[tuple[Any, Any], ...]] = []
            for sector in (0, 1):
                basis_states = tuple(
                    state
                    for state in range(full_dimension)
                    if state.bit_count() % 2 == sector
                )
                basis_index = {
                    state: index for index, state in enumerate(basis_states)
                }
                blocks.append(
                    tuple(
                        (
                            _flint_pauli_block_matrix(
                                flint,
                                label,
                                basis_states,
                                basis_index,
                            ),
                            flint.arb(str(coefficient)),
                        )
                        for label, coefficient in labels
                    )
                )
            term_sets = tuple(blocks)
        else:
            raise ValueError("symmetry_reduction must be 'none' or 'parity'")
        sector_results = tuple(
            _flint_sector_norm(
                flint,
                terms,
                time_ball,
                segments,
                diagnostics,
            )
            for terms in term_sets
        )
        value, relative_width, interval_certified = max(
            sector_results,
            key=lambda result: float(result[0].mid()),
        )
        return (
            value.str(decimal_digits, radius=False),
            0.0,
            relative_width,
            interval_certified,
        )
    finally:
        flint.ctx.dps = previous_dps


def recommended_initial_digits(
    branch_count: int,
    time: float,
    segments: int,
    *,
    schedule: MPFSchedule = "new",
) -> int:
    """Return the cancellation-aware starting precision from the study plan."""
    if not math.isfinite(time) or time <= 0:
        raise ValueError("time must be positive and finite")
    if segments < 1:
        raise ValueError("segments must be positive")
    diagnostics = mpf_richardson_diagnostics(branch_count, schedule=schedule)
    cancellation_digits = math.log10(
        diagnostics.leading_omitted_moment.denominator
    )
    step_digits = 2 * branch_count * math.log10(max(segments / time, 1.0))
    return max(64, math.ceil(cancellation_digits + step_digits + 25))


def adaptive_mpf_operator_norm_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    branch_count: int,
    *,
    backend: Literal["mpmath", "flint"] = "mpmath",
    symmetry_reduction: Literal["none", "parity"] = "none",
    schedule: MPFSchedule = "new",
    initial_digits: int | None = None,
    digit_increment: int = 32,
    max_digits: int = 512,
    relative_tolerance: float = 1e-8,
) -> HighPrecisionOperatorNormEstimate:
    """Evaluate an MPF error until successive working precisions agree."""
    if segments < 1:
        raise ValueError("segments must be positive")
    if digit_increment < 1 or max_digits < 2:
        raise ValueError("digit limits must be positive")
    if not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    if initial_digits is not None and initial_digits < 2:
        raise ValueError("initial_digits must be at least two")
    if backend == "mpmath" and symmetry_reduction != "none":
        raise ValueError("symmetry reduction is currently available only with FLINT")
    start_digits = (
        initial_digits
        if initial_digits is not None
        else recommended_initial_digits(
            branch_count,
            time,
            segments,
            schedule=schedule,
        )
    )
    if start_digits > max_digits:
        start_digits = max_digits

    attempted: list[int] = []
    previous: str | None = None
    relative_change = math.inf
    residual = math.inf
    interval_relative_width = math.inf
    interval_certified = False
    value_decimal = "nan"
    started = wall_time.perf_counter()
    digits = start_digits
    while True:
        attempted.append(digits)
        if backend == "mpmath":
            value_decimal, residual, interval_relative_width, interval_certified = (
                _mpmath_operator_norm_at_precision(
                    hamiltonian,
                    time,
                    segments,
                    branch_count,
                    schedule=schedule,
                    decimal_digits=digits,
                )
            )
        elif backend == "flint":
            value_decimal, residual, interval_relative_width, interval_certified = (
                _flint_operator_norm_at_precision(
                    hamiltonian,
                    time,
                    segments,
                    branch_count,
                    schedule=schedule,
                    decimal_digits=digits,
                    symmetry_reduction=symmetry_reduction,
                )
            )
        else:
            raise ValueError("backend must be 'mpmath' or 'flint'")
        if previous is not None:
            mp = _require_mpmath()
            with mp.workdps(digits):
                current_mp = mp.mpf(value_decimal)
                previous_mp = mp.mpf(previous)
                denominator = max(abs(current_mp), mp.power(10, -(digits - 8)))
                relative_change = float(abs(current_mp - previous_mp) / denominator)
            if (
                relative_change <= relative_tolerance
                and (
                    not interval_certified
                    or interval_relative_width <= relative_tolerance
                )
            ):
                break
        previous = value_decimal
        if digits == max_digits:
            break
        digits = min(digits + digit_increment, max_digits)
    converged = (
        len(attempted) >= 2
        and relative_change <= relative_tolerance
        and (
            not interval_certified
            or interval_relative_width <= relative_tolerance
        )
    )
    diagnostics = mpf_richardson_diagnostics(branch_count, schedule=schedule)
    schedule_payload = json.dumps(
        diagnostics.exponents,
        separators=(",", ":"),
    ).encode("ascii")
    term_payload = json.dumps(
        hamiltonian.terms,
        separators=(",", ":"),
    ).encode("ascii")
    if backend == "mpmath":
        backend_version = _require_mpmath().__version__
    else:
        backend_version = _require_flint().__version__
    return HighPrecisionOperatorNormEstimate(
        value_decimal=value_decimal,
        backend=backend,
        backend_version=backend_version,
        decimal_digits=attempted[-1],
        attempted_digits=tuple(attempted),
        converged=converged,
        relative_precision_change=relative_change,
        eigensystem_residual=residual,
        interval_relative_width=interval_relative_width,
        interval_certified=interval_certified,
        branch_count=branch_count,
        formal_order=2 * branch_count,
        segments=segments,
        schedule=schedule,
        symmetry_reduction=symmetry_reduction,
        schedule_digest=hashlib.sha256(schedule_payload).hexdigest(),
        term_order_digest=hashlib.sha256(term_payload).hexdigest(),
        wall_seconds=wall_time.perf_counter() - started,
    )
