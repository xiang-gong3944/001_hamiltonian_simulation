"""Operator-norm kernels used by the empirical calibration study.

This module is intentionally separate from runtime resource planning.  It
constructs ideal Trotter and ideal MPF operators; it never substitutes
state-dependent errors or the amplified reference circuit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, expm_multiply

from .hamiltonians import PauliHamiltonian
from .multiproduct import MPFSchedule, multiproduct_coefficients, optimal_mpf_exponents
from .trotter import TrotterPartition, resolve_trotter_structure, suzuki_group_factors


CalibrationAlgorithm = Literal["trotter", "multiproduct"]


@dataclass(frozen=True)
class OperatorNormEstimate:
    value: float
    backend: Literal["dense-svd", "sparse-power"]
    converged: bool
    iterations: int
    restarts: int
    relative_residual: float


@dataclass(frozen=True)
class AsymptoticPair:
    """Two consecutive observations accepted in the formal-order regime."""

    first_segments: int
    second_segments: int
    first_error: float
    second_error: float
    running_exponent: float


@dataclass(frozen=True)
class AffineCalibrationFit:
    """Diagnostics for a least-squares fit to ``B(N)=a*N+b``."""

    slope: float
    intercept: float
    r_squared: float
    root_mean_square_error: float
    slope_standard_error: float
    intercept_standard_error: float
    residuals: tuple[float, ...]


def effective_power(
    first_value: float,
    first_scale: float,
    second_value: float,
    second_scale: float,
) -> float:
    """Return ``beta`` for a local law ``value proportional to scale^-beta``."""
    values = (first_value, first_scale, second_value, second_scale)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("values and scales must be positive and finite")
    if second_scale <= first_scale:
        raise ValueError("scales must be strictly increasing")
    return float(
        math.log(first_value / second_value)
        / math.log(second_scale / first_scale)
    )


def select_asymptotic_pair(
    observations: tuple[tuple[int, float], ...],
    formal_order: int,
    *,
    relative_order_tolerance: float = 0.05,
    floating_point_floor: float = 1e-11,
) -> AsymptoticPair:
    """Select the latest consecutive segment pair in the accepted regime."""
    if formal_order < 1:
        raise ValueError("formal_order must be positive")
    if not 0 < relative_order_tolerance < 1:
        raise ValueError("relative_order_tolerance must lie in (0, 1)")
    if not np.isfinite(floating_point_floor) or floating_point_floor <= 0:
        raise ValueError("floating_point_floor must be positive and finite")
    if len(observations) < 2:
        raise ValueError("at least two observations are required")
    segments = tuple(point[0] for point in observations)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in segments
    ):
        raise ValueError("segment counts must be positive integers")
    if tuple(sorted(set(segments))) != segments:
        raise ValueError("segment counts must be strictly increasing")
    accepted: list[AsymptoticPair] = []
    for (first_segments, first_error), (second_segments, second_error) in zip(
        observations,
        observations[1:],
    ):
        if min(first_error, second_error) <= floating_point_floor:
            continue
        running = effective_power(
            first_error,
            float(first_segments),
            second_error,
            float(second_segments),
        )
        if abs(running - formal_order) <= relative_order_tolerance * formal_order:
            accepted.append(
                AsymptoticPair(
                    first_segments,
                    second_segments,
                    float(first_error),
                    float(second_error),
                    running,
                )
            )
    if not accepted:
        raise ValueError(
            "no consecutive observations satisfy the formal-order and "
            "floating-point-floor criteria"
        )
    return accepted[-1]


def observed_error_coefficient(
    error: float,
    segments: int,
    time: float,
    formal_order: int,
) -> float:
    """Remove the fixed ``r^-q T^(q+1)`` factors from one observation."""
    if segments < 1 or formal_order < 1:
        raise ValueError("segments and formal_order must be positive")
    if not np.isfinite((error, time)).all() or error <= 0 or time <= 0:
        raise ValueError("error and time must be positive and finite")
    return float(error * segments**formal_order / time ** (formal_order + 1))


def fit_affine_size_coefficient(
    sizes: tuple[int, ...],
    coefficients: tuple[float, ...],
) -> AffineCalibrationFit:
    """Fit a reviewed one-dimensional bulk-plus-boundary coefficient."""
    if len(sizes) != len(coefficients) or len(sizes) < 3:
        raise ValueError("at least three aligned sizes and coefficients are required")
    if tuple(sorted(set(sizes))) != sizes or sizes[0] < 1:
        raise ValueError("sizes must be positive and strictly increasing")
    if not np.isfinite(coefficients).all() or min(coefficients) <= 0:
        raise ValueError("coefficients must be positive and finite")
    design = np.column_stack((np.asarray(sizes, dtype=float), np.ones(len(sizes))))
    response = np.asarray(coefficients, dtype=float)
    slope, intercept = np.linalg.lstsq(design, response, rcond=None)[0]
    fitted = design @ np.asarray((slope, intercept))
    residuals = response - fitted
    squared_error = float(np.dot(residuals, residuals))
    centered = response - np.mean(response)
    total = float(np.dot(centered, centered))
    r_squared = 1.0 if total == 0 else 1.0 - squared_error / total
    variance = squared_error / (len(sizes) - 2)
    covariance = variance * np.linalg.inv(design.T @ design)
    return AffineCalibrationFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        root_mean_square_error=float(np.sqrt(squared_error / len(sizes))),
        slope_standard_error=float(np.sqrt(covariance[0, 0])),
        intercept_standard_error=float(np.sqrt(covariance[1, 1])),
        residuals=tuple(float(value) for value in residuals),
    )


@dataclass(frozen=True)
class _SparsePauliAction:
    """Vectorized action of one real-weighted Pauli word."""

    coefficient: float
    permutation: np.ndarray
    phase: np.ndarray

    def apply(self, vector: np.ndarray) -> np.ndarray:
        result = np.empty_like(vector)
        result[self.permutation] = self.phase * vector
        return result


def _sparse_pauli_action(label: str, coefficient: float) -> _SparsePauliAction:
    num_qubits = len(label)
    dimension = 2**num_qubits
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
    indices = np.arange(dimension, dtype=np.uint64)
    permutation = np.asarray(indices ^ x_mask, dtype=np.intp)
    parity = np.zeros(dimension, dtype=np.uint8)
    masked = indices & z_mask
    while np.any(masked):
        parity ^= np.asarray(masked & 1, dtype=np.uint8)
        masked >>= 1
    phase = (1j**y_count) * (1 - 2 * parity.astype(np.int8))
    return _SparsePauliAction(float(coefficient), permutation, phase.astype(complex))


@lru_cache(maxsize=None)
def _dense_term_matrices(hamiltonian: PauliHamiltonian) -> tuple[np.ndarray, ...]:
    matrices = []
    for label, coefficient in hamiltonian.terms:
        term = PauliHamiltonian.from_terms(
            hamiltonian.num_qubits,
            [(label, coefficient)],
        )
        matrices.append(np.asarray(term.matrix(), dtype=complex))
    return tuple(matrices)


@lru_cache(maxsize=None)
def _dense_groups(
    hamiltonian: PauliHamiltonian,
    order: int,
    partition: TrotterPartition,
) -> tuple[np.ndarray, ...]:
    terms = _dense_term_matrices(hamiltonian)
    structure = resolve_trotter_structure(hamiltonian, order, partition)
    return tuple(
        sum(
            (terms[index] for index in group),
            start=np.zeros_like(terms[0]),
        )
        for group in structure.group_term_indices
    )


def _dense_product_step(
    groups: tuple[np.ndarray, ...],
    step_time: float,
    order: int,
) -> np.ndarray:
    spectra = tuple(np.linalg.eigh(group) for group in groups)
    return _dense_product_step_from_spectra(spectra, step_time, order)


def _dense_product_step_from_spectra(
    spectra: tuple[tuple[np.ndarray, np.ndarray], ...],
    step_time: float,
    order: int,
) -> np.ndarray:
    result = np.eye(spectra[0][1].shape[0], dtype=complex)
    for group, coefficient in suzuki_group_factors(len(spectra), order):
        eigenvalues, eigenvectors = spectra[group]
        phases = np.exp(-1j * coefficient * step_time * eigenvalues)
        unitary = (eigenvectors * phases) @ eigenvectors.conj().T
        result = unitary @ result
    return result


def dense_trotter_operator(
    hamiltonian: PauliHamiltonian,
    time: float,
    repetitions: int,
    order: int,
    *,
    partition: TrotterPartition = "auto",
) -> np.ndarray:
    groups = _dense_groups(hamiltonian, order, partition)
    step = _dense_product_step(groups, float(time) / repetitions, order)
    return np.linalg.matrix_power(step, repetitions)


def dense_mpf_operator(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> np.ndarray:
    groups = _dense_groups(hamiltonian, 2, "individual")
    spectra = tuple(np.linalg.eigh(group) for group in groups)
    exponents = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = multiproduct_coefficients(m, schedule=schedule)
    step_time = float(time) / segments
    mpf_step = np.zeros_like(groups[0])
    for coefficient, exponent in zip(coefficients, exponents, strict=True):
        base = _dense_product_step_from_spectra(spectra, step_time / exponent, 2)
        mpf_step += float(coefficient) * np.linalg.matrix_power(base, exponent)
    return np.linalg.matrix_power(mpf_step, segments)


def dense_operator_norm_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    *,
    algorithm: CalibrationAlgorithm,
    formal_order: int,
    partition: TrotterPartition = "auto",
    schedule: MPFSchedule = "new",
) -> OperatorNormEstimate:
    exact = _dense_exact_evolution(hamiltonian, float(time))
    if algorithm == "trotter":
        approximate = dense_trotter_operator(
            hamiltonian,
            time,
            segments,
            formal_order,
            partition=partition,
        )
    elif algorithm == "multiproduct":
        if formal_order % 2:
            raise ValueError("MPF formal_order must be even")
        approximate = dense_mpf_operator(
            hamiltonian,
            time,
            segments,
            formal_order // 2,
            schedule=schedule,
        )
    else:
        raise ValueError("unknown calibration algorithm")
    value = float(np.linalg.norm(approximate - exact, ord=2))
    return OperatorNormEstimate(value, "dense-svd", True, 1, 1, 0.0)


@lru_cache(maxsize=None)
def _dense_exact_evolution(
    hamiltonian: PauliHamiltonian,
    time: float,
) -> np.ndarray:
    return expm(-1j * float(time) * hamiltonian.matrix())


def _sparse_groups(
    hamiltonian: PauliHamiltonian,
    order: int,
    partition: TrotterPartition,
):
    terms = tuple(
        _sparse_pauli_action(label, coefficient)
        for label, coefficient in hamiltonian.terms
    )
    structure = resolve_trotter_structure(hamiltonian, order, partition)
    return tuple(tuple(terms[index] for index in group) for group in structure.group_term_indices)


def _apply_product_step(
    groups,
    vector: np.ndarray,
    step_time: float,
    order: int,
    *,
    adjoint: bool,
) -> np.ndarray:
    factors = suzuki_group_factors(len(groups), order)
    if adjoint:
        factors = tuple(reversed(factors))
    result = vector
    sign = 1j if adjoint else -1j
    for group, coefficient in factors:
        for term in groups[group]:
            angle = coefficient * step_time * term.coefficient
            result = (
                np.cos(angle) * result
                + sign * np.sin(angle) * term.apply(result)
            )
    return np.asarray(result)


def _apply_repeated_product(
    groups,
    vector: np.ndarray,
    time: float,
    repetitions: int,
    order: int,
    *,
    adjoint: bool,
) -> np.ndarray:
    result = vector
    for _ in range(repetitions):
        result = _apply_product_step(
            groups,
            result,
            time / repetitions,
            order,
            adjoint=adjoint,
        )
    return result


def _apply_mpf_step(
    groups,
    vector: np.ndarray,
    step_time: float,
    exponents: tuple[int, ...],
    coefficients: np.ndarray,
    *,
    adjoint: bool,
) -> np.ndarray:
    result = np.zeros_like(vector)
    for coefficient, exponent in zip(coefficients, exponents, strict=True):
        branch = vector
        for _ in range(exponent):
            branch = _apply_product_step(
                groups,
                branch,
                step_time / exponent,
                2,
                adjoint=adjoint,
            )
        result += float(coefficient) * branch
    return result


def sparse_operator_norm_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    *,
    algorithm: CalibrationAlgorithm,
    formal_order: int,
    partition: TrotterPartition = "auto",
    schedule: MPFSchedule = "new",
    tolerance: float = 1e-8,
    max_iterations: int = 80,
    restarts: int = 3,
    seed: int = 2026,
) -> OperatorNormEstimate:
    """Estimate ``||U_approx-U||_2`` through power iteration on ``D^dagger D``."""
    if segments < 1 or max_iterations < 1 or restarts < 1:
        raise ValueError("segments, max_iterations, and restarts must be positive")
    if algorithm == "trotter":
        groups = _sparse_groups(hamiltonian, formal_order, partition)

        def approximate(vector: np.ndarray, adjoint: bool) -> np.ndarray:
            return _apply_repeated_product(
                groups,
                vector,
                float(time),
                segments,
                formal_order,
                adjoint=adjoint,
            )

    elif algorithm == "multiproduct":
        if formal_order % 2:
            raise ValueError("MPF formal_order must be even")
        groups = _sparse_groups(hamiltonian, 2, "individual")
        exponents = optimal_mpf_exponents(formal_order // 2, schedule=schedule)
        coefficients = multiproduct_coefficients(formal_order // 2, schedule=schedule)

        def approximate(vector: np.ndarray, adjoint: bool) -> np.ndarray:
            result = vector
            for _ in range(segments):
                result = _apply_mpf_step(
                    groups,
                    result,
                    float(time) / segments,
                    exponents,
                    coefficients,
                    adjoint=adjoint,
                )
            return result

    else:
        raise ValueError("unknown calibration algorithm")

    exact_matrix = hamiltonian.to_sparse_pauli_op().to_matrix(sparse=True).tocsr()

    def difference(vector: np.ndarray, adjoint: bool) -> np.ndarray:
        sign = 1j if adjoint else -1j
        exact = expm_multiply(sign * float(time) * exact_matrix, vector)
        return approximate(vector, adjoint) - exact

    dimension = 2**hamiltonian.num_qubits
    operator = LinearOperator(
        (dimension, dimension),
        matvec=lambda vector: difference(difference(vector, False), True),
        dtype=np.complex128,
    )
    rng = np.random.default_rng(seed)
    best = 0.0
    best_iterations = 0
    best_residual = np.inf
    any_converged = False
    for _ in range(restarts):
        vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)
        previous = 0.0
        residual = np.inf
        for iteration in range(1, max_iterations + 1):
            applied = np.asarray(operator.matvec(vector))
            norm = float(np.linalg.norm(applied))
            if norm == 0:
                eigenvalue = 0.0
                residual = 0.0
                converged = True
                break
            vector = applied / norm
            eigenvalue = max(0.0, norm)
            converged = previous > 0 and (
                abs(eigenvalue - previous) <= tolerance * max(eigenvalue, 1e-300)
            )
            previous = eigenvalue
            if converged:
                break
        reapplied = np.asarray(operator.matvec(vector))
        eigenvalue = max(0.0, float(np.vdot(vector, reapplied).real))
        residual = float(np.linalg.norm(reapplied - eigenvalue * vector)) / max(
            1.0,
            eigenvalue,
        )
        value = float(np.sqrt(eigenvalue))
        if value > best:
            best = value
            best_iterations = iteration
            best_residual = residual
        any_converged = any_converged or converged
    return OperatorNormEstimate(
        best,
        "sparse-power",
        any_converged,
        best_iterations,
        restarts,
        best_residual,
    )
