"""Well-conditioned multiproduct-formula (MPF) LCU circuits.

Implements the coherent LCU construction based on Childs & Wiebe-style
multiproduct formulas and the well-conditioned schedules of Low, Kliuchnikov,
and Wiebe, arXiv:1907.11679. Each normalized LCU step is robustly amplified
before the same branch register is reused for the next simulation segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

from ._commutator_execution import CommutatorExecution, execution_scope, validate_workers
from .circuit_utils import (
    build_three_step_oaa,
    index_state_phase_gate,
    state_preparation,
)
from .hamiltonians import PauliHamiltonian
from .trotter import (
    PauliNestedCommutatorBounds,
    build_trotter_circuit,
    pauli_nested_commutator_bounds,
    suzuki_commutator_bounds,
)


_OAA_NORMALIZATION = 2.0
MPFSchedule = Literal["new", "legacy"]
MPFErrorMethod: TypeAlias = Literal[
    "low2019-l1-ideal-rigorous",
    "mizuta2026-commutator-ideal-rigorous",
    "low-rigorous",
    "legacy-w2-proxy",
]
MPFErrorScope: TypeAlias = Literal["ideal-mpf", "amplified-shared-ancilla"]


@dataclass(frozen=True)
class MPFErrorEstimate:
    """Compatibility view of one family-specific MPF sizing estimate.

    ``rigorous`` certifies ``error`` only for ``scope``.  In particular, the
    Low--Kliuchnikov--Wiebe bound implemented here applies to the ideal MPF
    operator. The plan's canonical ``ErrorAnalysis`` derives one-segment and
    repeated-good-block claims separately; the legacy ``circuit_*`` fields on
    this adapter remain conservative for backward compatibility.
    """

    error: float
    prefactor: float
    time: float
    segments: int
    m: int
    formal_order: int
    schedule: MPFSchedule
    exponents: tuple[int, ...]
    coefficient_l1_norm: float
    method: MPFErrorMethod
    scope: MPFErrorScope
    rigorous: bool
    local_error: float | None = None
    local_error_rigorous: bool = False
    circuit_scope: MPFErrorScope = "amplified-shared-ancilla"
    circuit_rigorous: bool = False
    reference: str = ""
    theorem_or_equations: str = ""
    local_step_size: float = 0.0
    bound_components: tuple[tuple[str, float], ...] = ()
    hamiltonian_decomposition: str = "ordered individual Pauli terms"
    assumptions: tuple[str, ...] = ()
    fallback_reason: str | None = None
    max_nested_commutator_order: int = 0
    max_exact_nested_commutator_order: int = 0
    locality_compatible: bool = False
    commutator_bounds: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class MPFLCUStructure:
    """Logical branch structure shared by the circuit and resource model."""

    physical_branch_count: int
    negative_coefficient_count: int
    padding_branch_count: int
    sign_branch_count: int
    active_branch_count: int
    branch_bits: int
    unused_branch_state_count: int
    coefficient_l1_norm: float
    padding_weight: float


_NEW_MPF_EXPONENTS: dict[int, tuple[int, ...]] = {
    2: (1, 2),
    3: (1, 2, 4),
    4: (1, 2, 3, 7),
    5: (1, 2, 3, 5, 12),
    6: (1, 2, 3, 4, 6, 16),
    7: (1, 2, 3, 4, 5, 9, 22),
    8: (1, 2, 3, 4, 5, 6, 11, 29),
    9: (1, 2, 3, 4, 5, 6, 8, 14, 37),
    10: (1, 2, 3, 4, 5, 6, 7, 10, 18, 46),
    11: (1, 2, 3, 4, 5, 6, 7, 8, 12, 22, 56),
    12: (1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 26, 66),
    13: (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 30, 78),
    14: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 19, 35, 91),
    15: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 22, 40, 104),
}


_LEGACY_MPF_EXPONENTS: dict[int, tuple[int, ...]] = {
    2: (1, 2),
    3: (1, 2, 6),
    4: (1, 2, 3, 10),
    5: (1, 2, 3, 5, 17),
    6: (1, 2, 3, 4, 6, 21),
    7: (1, 2, 3, 4, 5, 9, 34),
    8: (1, 2, 3, 4, 5, 6, 12, 45),
    9: (1, 2, 3, 4, 5, 6, 8, 15, 58),
    10: (1, 2, 3, 4, 5, 6, 7, 10, 18, 72),
    11: (1, 2, 3, 4, 5, 6, 7, 8, 12, 22, 88),
    12: (1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 27, 106),
    13: (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 31, 121),
    14: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 19, 37, 147),
    15: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 22, 42, 170),
}


def optimal_mpf_exponents(
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> tuple[int, ...]:
    """Return one registered well-conditioned exponent schedule.

    ``new`` is the default table optimized for the three-query OAA construction:
    it reduces ``sum(k_j)`` while retaining coefficient 1-norm below two.
    ``legacy`` preserves the previous, more conservatively conditioned table.
    """
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError("m must be an integer")
    if schedule == "new":
        table = _NEW_MPF_EXPONENTS
    elif schedule == "legacy":
        table = _LEGACY_MPF_EXPONENTS
    else:
        raise ValueError("schedule must be 'new' or 'legacy'")
    try:
        return table[int(m)]
    except KeyError as error:
        raise ValueError("m must lie between 2 and 15") from error


def multiproduct_coefficients(
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> np.ndarray:
    """Return coefficients for one registered ``m``-term MPF schedule.

    The direct product formula avoids the ill-conditioned Vandermonde solve
    while satisfying the cancellation conditions through formal order ``2m``.
    """
    ks = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = np.ones(len(ks), dtype=float)
    for j, k_j in enumerate(ks):
        k_j_squared = k_j**2
        for q, k_q in enumerate(ks):
            if q != j:
                coefficients[j] *= k_j_squared / (k_j_squared - k_q**2)
    return coefficients


def mpf_lcu_structure(
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> MPFLCUStructure:
    """Return branch/sign counts for the implemented normalized MPF LCU."""
    coefficients = multiproduct_coefficients(m, schedule=schedule)
    coefficient_l1_norm = float(np.sum(np.abs(coefficients)))
    if coefficient_l1_norm >= _OAA_NORMALIZATION:
        raise ValueError("the MPF coefficient 1-norm must be less than 2")
    physical_branch_count = len(coefficients)
    padding_branch_count = 2
    active_branch_count = physical_branch_count + padding_branch_count
    branch_bits = max(1, math.ceil(math.log2(active_branch_count)))
    negative_coefficient_count = int(np.count_nonzero(coefficients < 0))
    return MPFLCUStructure(
        physical_branch_count=physical_branch_count,
        negative_coefficient_count=negative_coefficient_count,
        padding_branch_count=padding_branch_count,
        sign_branch_count=negative_coefficient_count + 1,
        active_branch_count=active_branch_count,
        branch_bits=branch_bits,
        unused_branch_state_count=2**branch_bits - active_branch_count,
        coefficient_l1_norm=coefficient_l1_norm,
        padding_weight=_OAA_NORMALIZATION - coefficient_l1_norm,
    )


def legacy_w2_proxy_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int,
    segments: int,
) -> tuple[float, float]:
    """Return the historical W2-calibrated MPF proxy and its prefactor.

    This is not a rigorous MPF error bound.  It is retained under an explicit
    name solely for reproducing earlier benchmark data.
    """
    if segments < 1:
        raise ValueError("segments must be positive")
    optimal_mpf_exponents(m)
    _, w2 = suzuki_commutator_bounds(hamiltonian)
    alpha_effective = min(hamiltonian.alpha, w2 ** (1 / 3))
    formal_order = 2 * m
    prefactor = alpha_effective ** (formal_order + 1)
    error = prefactor * abs(float(time)) ** (formal_order + 1) / segments**formal_order
    return error, prefactor


def legacy_w2_proxy_segments(
    hamiltonian: PauliHamiltonian,
    time: float,
    target_error: float,
    m: int,
) -> int:
    """Reproduce the historical non-rigorous MPF segment-selection rule."""
    if target_error <= 0:
        raise ValueError("target_error must be positive")
    one_segment_error, _ = legacy_w2_proxy_error(hamiltonian, time, m, 1)
    return max(1, math.ceil((one_segment_error / target_error) ** (1 / (2 * m))))


def _low_ideal_mpf_bound(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int,
    segments: int,
    coefficient_l1_norm: float,
) -> tuple[float, float]:
    """Evaluate Low--Kliuchnikov--Wiebe Eqs. (14)--(15)."""
    log_error, prefactor_log = _low_log_ideal_mpf_bound(
        hamiltonian,
        time,
        m,
        segments,
        coefficient_l1_norm,
    )
    prefactor = math.exp(prefactor_log) if prefactor_log < 709 else math.inf
    error = math.exp(log_error) if log_error < 709 else math.inf
    return error, prefactor


def _low_local_mpf_bound(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int,
    segments: int,
    coefficient_l1_norm: float,
) -> float:
    """Evaluate the one-step Low--Kliuchnikov--Wiebe Eq. (14) bound."""
    formal_order = 2 * m
    scaled_time = hamiltonian.alpha * abs(float(time)) / segments
    if scaled_time == 0:
        return 0.0
    log_error = (
        math.log(2 * coefficient_l1_norm)
        + (formal_order + 1) * math.log(scaled_time)
        - math.lgamma(formal_order + 2)
        + scaled_time
    )
    return math.exp(log_error) if log_error < 709 else math.inf


def _log1p_exp(value: float) -> float:
    """Return ``log(1 + exp(value))`` without overflow."""
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _low_log_ideal_mpf_bound(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int,
    segments: int,
    coefficient_l1_norm: float,
) -> tuple[float, float]:
    """Evaluate Low 2019 Eqs. (14)--(15) entirely in the log domain."""
    formal_order = 2 * m
    lambda_norm = hamiltonian.alpha
    factorial_log = math.lgamma(formal_order + 2)
    prefactor_log = (
        math.log(2 * coefficient_l1_norm)
        + (formal_order + 1) * math.log(lambda_norm)
        - factorial_log
        if lambda_norm > 0
        else -math.inf
    )
    scaled_time = lambda_norm * abs(float(time)) / segments
    if scaled_time == 0:
        return -math.inf, prefactor_log
    log_step_error = (
        math.log(2 * coefficient_l1_norm)
        + (formal_order + 1) * math.log(scaled_time)
        - factorial_log
        + scaled_time
    )
    log_global_error = (
        log_step_error + math.log(segments) + (segments - 1) * _log1p_exp(log_step_error)
    )
    return log_global_error, prefactor_log


def _logsumexp(values: list[float]) -> float:
    """Return the log of a positive exponential sum."""
    maximum = max(values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


@lru_cache(maxsize=None)
def _mizuta_mu_upper_bound(
    commutators: PauliNestedCommutatorBounds,
    *,
    base_order: int = 2,
) -> float:
    """Upper-bound Mizuta 2026 Eq. (61) by a finite polynomial root.

    Let ``A(x)=sum_q alpha_com,q x^q`` for ``base_order < q <= p0``.
    If ``A(x_*)=1``, every coefficient of ``A(x_*)^n`` is at most one.
    Therefore the supremum in Eq. (61) is at most ``1/x_*``. This retains
    every finite commutator order required by Theorem 4 without enumerating
    its unbounded repetition index ``n``.
    """
    log_terms = [
        (order, math.log(commutators.at(order)))
        for order in range(base_order + 1, commutators.max_order + 1)
        if commutators.at(order) > 0
    ]
    if not log_terms:
        return 0.0
    if any(math.isinf(log_value) for _, log_value in log_terms):
        return math.inf

    # Solve sum_q alpha_q / mu^q = 1 in log(mu). At the lower endpoint
    # at least one summand is one; the upper endpoint makes the whole sum <= 1.
    lower = max(log_value / order for order, log_value in log_terms)
    upper = lower + math.log(len(log_terms)) / min(order for order, _ in log_terms)

    def log_polynomial(log_mu: float) -> float:
        return _logsumexp([log_value - order * log_mu for order, log_value in log_terms])

    for _ in range(80):
        midpoint = (lower + upper) / 2
        if log_polynomial(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint
    if upper > 709:
        return math.inf
    return float(np.nextafter(math.exp(upper), np.inf))


def _repeated_step_error(step_error: float, segments: int) -> float:
    """Apply the telescoping bound used in Low 2019 Eq. (15)."""
    if step_error == 0:
        return 0.0
    if not math.isfinite(step_error):
        return math.inf
    log_error = math.log(step_error) + math.log(segments) + (segments - 1) * math.log1p(step_error)
    return math.exp(log_error) if log_error < 709 else math.inf


def _mizuta_ideal_mpf_bound(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int,
    segments: int,
    coefficient_l1_norm: float,
    exponents: tuple[int, ...],
    target_error: float,
    execution: CommutatorExecution | None = None,
) -> tuple[
    float,
    float,
    tuple[tuple[str, float], ...],
    PauliNestedCommutatorBounds,
    tuple[str, ...],
    bool,
]:
    """Evaluate Mizuta 2026 Theorem 4/Eqs. (61)--(63) for ``p=2``."""
    formal_order = 2 * m
    base_order = 2
    base_repetitions = 2  # c_p for the ordered Strang formula, paper Eq. (6)
    exponent_l1_norm = float(sum(exponents))
    step_time = abs(float(time)) / segments

    # Allocate half of the exact local budget implied by the repeated-step
    # target to the truncated-BCH remainder in Theorem 4.
    local_budget = math.expm1(math.log1p(target_error) / segments)
    auxiliary_error = local_budget / (2 * coefficient_l1_norm * exponent_l1_norm)
    if auxiliary_error <= 0:
        raise OverflowError("Mizuta auxiliary error underflowed float range")
    truncation_order = math.ceil(math.log(3 * hamiltonian.num_qubits / auxiliary_error))
    truncation_order = max(base_order + 1, truncation_order)
    commutators = pauli_nested_commutator_bounds(
        hamiltonian,
        truncation_order,
        workers=execution.workers if execution is not None else 1,
        _execution=execution,
    )

    if all(value == 0 for value in commutators.values):
        components = (
            ("mu_upper", 0.0),
            ("local_commutator_error", 0.0),
            ("local_truncated_bch_error", 0.0),
            ("auxiliary_error", auxiliary_error),
            ("truncation_order_p0", float(truncation_order)),
            ("locality_k", float(commutators.locality_k)),
            ("extensiveness_g", commutators.extensiveness_g),
        )
        return (
            0.0,
            0.0,
            components,
            commutators,
            ("all individual Pauli summands commute; the ordered Strang formula is exact",),
            True,
        )

    mu_upper = _mizuta_mu_upper_bound(commutators, base_order=base_order)
    if mu_upper == 0:
        prefactor = 0.0
        local_commutator_error = 0.0
    else:
        log_prefactor = (
            math.log(2)
            + 0.5
            + math.log(coefficient_l1_norm)
            + (formal_order + 1) * math.log(base_repetitions * mu_upper)
        )
        prefactor = math.exp(log_prefactor) if log_prefactor < 709 else math.inf
        if step_time == 0:
            local_commutator_error = 0.0
        else:
            log_commutator_error = log_prefactor + (formal_order + 1) * math.log(step_time)
            local_commutator_error = (
                math.exp(log_commutator_error) if log_commutator_error < 709 else math.inf
            )
    local_truncated_bch_error = coefficient_l1_norm * exponent_l1_norm * auxiliary_error
    local_step_error = local_commutator_error + local_truncated_bch_error

    k_value = commutators.locality_k
    g_value = commutators.extensiveness_g
    first_time_limit = (
        math.inf
        if k_value == 0 or g_value == 0
        else 1 / (8 * math.e**3 * base_repetitions * truncation_order * k_value * g_value)
    )
    second_time_limit = math.inf if mu_upper == 0 else 1 / (2 * base_repetitions * mu_upper)
    time_hypothesis_satisfied = step_time <= min(first_time_limit, second_time_limit)
    assumptions = (
        "individual Pauli summands define the ordered H_gamma decomposition",
        "Pauli support gives k-locality and per-site coefficient sums give g-extensiveness",
        "base formula is the symmetric second-order formula with c_p=2",
        (
            "Theorem 4 time hypothesis Eq. (62) is satisfied"
            if time_hypothesis_satisfied
            else "Theorem 4 time hypothesis Eq. (62) is not satisfied"
        ),
        "the repeated ideal MPF is composed with the Eq. (15) telescoping argument",
    )
    components = (
        ("mu_upper", mu_upper),
        ("local_commutator_error", local_commutator_error),
        ("local_truncated_bch_error", local_truncated_bch_error),
        ("local_step_error", local_step_error),
        ("auxiliary_error", auxiliary_error),
        ("truncation_order_p0", float(truncation_order)),
        ("first_time_limit", first_time_limit),
        ("second_time_limit", second_time_limit),
        ("locality_k", float(k_value)),
        ("extensiveness_g", g_value),
    )
    error = (
        _repeated_step_error(local_step_error, segments) if time_hypothesis_satisfied else math.inf
    )
    return (
        error,
        prefactor,
        components,
        commutators,
        assumptions,
        time_hypothesis_satisfied,
    )


def _normalize_mpf_error_method(method: MPFErrorMethod) -> MPFErrorMethod:
    """Map the historical Low method name to its explicit canonical name."""
    if method == "low-rigorous":
        return "low2019-l1-ideal-rigorous"
    return method


def estimate_mpf_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    m: int,
    *,
    schedule: MPFSchedule = "new",
    method: MPFErrorMethod = "low2019-l1-ideal-rigorous",
    target_error: float | None = None,
    workers: int = 1,
    _execution: CommutatorExecution | None = None,
) -> MPFErrorEstimate:
    """Estimate ideal-MPF error while preserving certification provenance."""
    validate_workers(workers)
    if _execution is not None and workers != _execution.workers:
        raise ValueError("workers must match the shared commutator execution")
    if isinstance(segments, bool) or not isinstance(segments, Integral):
        raise TypeError("segments must be an integer")
    if segments < 1:
        raise ValueError("segments must be positive")
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    exponents = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = multiproduct_coefficients(m, schedule=schedule)
    coefficient_l1_norm = float(np.sum(np.abs(coefficients)))
    method = _normalize_mpf_error_method(method)
    if method == "low2019-l1-ideal-rigorous":
        error, prefactor = _low_ideal_mpf_bound(
            hamiltonian,
            time,
            m,
            int(segments),
            coefficient_l1_norm,
        )
        rigorous = True
        local_error = _low_local_mpf_bound(
            hamiltonian,
            time,
            m,
            int(segments),
            coefficient_l1_norm,
        )
        local_error_rigorous = True
        reference = "Low, Kliuchnikov, and Wiebe, arXiv:1907.11679v2 (2019)"
        theorem_or_equations = "Theorem 2, Eqs. (13)--(15)"
        bound_components = (("worst_case_prefactor", prefactor),)
        assumptions = (
            "H is decomposed into the ordered individual Pauli summands",
            "lambda is upper-bounded by the Pauli coefficient 1-norm",
            "the bound certifies the repeated ideal MPF operator only",
        )
        commutators: PauliNestedCommutatorBounds | None = None
    elif method == "legacy-w2-proxy":
        error, prefactor = legacy_w2_proxy_error(
            hamiltonian,
            time,
            m,
            int(segments),
        )
        rigorous = False
        local_error = error / int(segments)
        local_error_rigorous = False
        reference = "historical repository W2 calibration; no MPF theorem"
        theorem_or_equations = "none (nonrigorous proxy)"
        bound_components = (("legacy_w2_prefactor", prefactor),)
        assumptions = ("alpha_eff=min(alpha,W2^(1/3)) is a heuristic MPF substitution",)
        commutators = None
    elif method == "mizuta2026-commutator-ideal-rigorous":
        if target_error is None or not 0 < target_error <= 1:
            raise ValueError("target_error in (0, 1] is required for the Mizuta estimator")
        (
            error,
            prefactor,
            bound_components,
            commutators,
            assumptions,
            rigorous,
        ) = _mizuta_ideal_mpf_bound(
            hamiltonian,
            time,
            m,
            int(segments),
            coefficient_l1_norm,
            exponents,
            target_error,
            _execution,
        )
        reference = "Mizuta, Quantum 10, 1974 (2026), arXiv:2507.06557v4"
        theorem_or_equations = "Theorem 4, Eqs. (61)--(63), with Theorem 3, Eqs. (47)--(49)"
        local_error = dict(bound_components).get("local_step_error", 0.0)
        local_error_rigorous = rigorous
    else:
        raise ValueError(
            "MPF error method must be 'low2019-l1-ideal-rigorous' "
            "(historical alias 'low-rigorous'), "
            "'mizuta2026-commutator-ideal-rigorous', or 'legacy-w2-proxy'"
        )
    return MPFErrorEstimate(
        error=error,
        prefactor=prefactor,
        time=float(time),
        segments=int(segments),
        m=int(m),
        formal_order=2 * int(m),
        schedule=schedule,
        exponents=exponents,
        coefficient_l1_norm=coefficient_l1_norm,
        method=method,
        scope="ideal-mpf",
        rigorous=rigorous,
        local_error=local_error,
        local_error_rigorous=local_error_rigorous,
        reference=reference,
        theorem_or_equations=theorem_or_equations,
        local_step_size=abs(float(time)) / int(segments),
        bound_components=bound_components,
        assumptions=assumptions,
        fallback_reason=(commutators.fallback_reason if commutators is not None else None),
        max_nested_commutator_order=(commutators.max_order if commutators is not None else 0),
        max_exact_nested_commutator_order=(
            commutators.max_exact_order if commutators is not None else 0
        ),
        locality_compatible=commutators is not None,
        commutator_bounds=(
            tuple((order, commutators.at(order)) for order in range(2, commutators.max_order + 1))
            if commutators is not None
            else ()
        ),
    )


def _select_mpf_segments(
    hamiltonian: PauliHamiltonian,
    time: float,
    target_error: float,
    m: int,
    *,
    schedule: MPFSchedule = "new",
    method: MPFErrorMethod = "low2019-l1-ideal-rigorous",
    execution: CommutatorExecution,
) -> MPFErrorEstimate:
    """Select segments and return the resulting scoped MPF estimate.

    ``low2019-l1-ideal-rigorous`` uses Eq. (16) of Low, Kliuchnikov, and
    Wiebe, arXiv:1907.11679, only as a safe upper bracket, then finds the
    smallest integer satisfying their direct Eqs. (14)--(15) bound by binary
    search. ``low-rigorous`` remains a backward-compatible alias.
    ``legacy-w2-proxy`` exactly reproduces the historical heuristic.
    """
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if not 0 < target_error <= 1:
        raise ValueError("target_error must lie in (0, 1]")
    optimal_mpf_exponents(m, schedule=schedule)
    method = _normalize_mpf_error_method(method)
    if method == "legacy-w2-proxy":
        segments = legacy_w2_proxy_segments(hamiltonian, time, target_error, m)
    elif method == "low2019-l1-ideal-rigorous":
        coefficients = multiproduct_coefficients(m, schedule=schedule)
        coefficient_l1_norm = float(np.sum(np.abs(coefficients)))
        scaled_time = hamiltonian.alpha * abs(float(time))
        if scaled_time == 0:
            segments = 1
        else:
            formal_order = 2 * m
            log_accuracy_factor = (
                math.log(8 * coefficient_l1_norm * scaled_time)
                - math.log(target_error)
                - math.lgamma(formal_order + 2)
            ) / formal_order
            log_multiplier = max(log_accuracy_factor, -math.log(math.log(2)))
            log_segments = math.log(scaled_time) + log_multiplier
            if log_segments > 709:
                raise OverflowError("required MPF segment count exceeds float range")
            segments = max(1, math.ceil(math.exp(log_segments)))

        target_log = math.log(target_error)

        def satisfies(candidate: int) -> bool:
            log_error, _ = _low_log_ideal_mpf_bound(
                hamiltonian,
                time,
                m,
                candidate,
                coefficient_l1_norm,
            )
            return log_error <= target_log

        # Eq. (16) is the analytical upper bracket. Doubling is retained as a
        # guard against floating-point rounding at its integer boundary.
        while not satisfies(segments):
            segments *= 2
            if math.log(segments) > 709:
                raise OverflowError("required MPF segment count exceeds float range")

        if segments > 1 and satisfies(1):
            segments = 1
        elif segments > 1:
            lower = 1
            upper = segments
            while lower + 1 < upper:
                midpoint = (lower + upper) // 2
                if satisfies(midpoint):
                    upper = midpoint
                else:
                    lower = midpoint
            segments = upper
    elif method == "mizuta2026-commutator-ideal-rigorous":
        estimates: dict[int, MPFErrorEstimate] = {}

        def candidate_estimate(candidate: int) -> MPFErrorEstimate:
            estimate = estimates.get(candidate)
            if estimate is None:
                estimate = estimate_mpf_error(
                    hamiltonian,
                    time,
                    candidate,
                    m,
                    schedule=schedule,
                    method=method,
                    target_error=target_error,
                    workers=execution.workers,
                    _execution=execution,
                )
                estimates[candidate] = estimate
            return estimate

        def satisfies_mizuta(candidate: int) -> bool:
            estimate = candidate_estimate(candidate)
            return estimate.rigorous and estimate.error <= target_error

        segments = 1
        while not satisfies_mizuta(segments):
            segments *= 2
            if math.log(segments) > 709:
                raise OverflowError("required MPF segment count exceeds float range")
        if segments > 1:
            lower = 1
            upper = segments
            while lower + 1 < upper:
                midpoint = (lower + upper) // 2
                if satisfies_mizuta(midpoint):
                    upper = midpoint
                else:
                    lower = midpoint
            segments = upper
    else:
        raise ValueError(
            "MPF error method must be 'low2019-l1-ideal-rigorous' "
            "(historical alias 'low-rigorous'), "
            "'mizuta2026-commutator-ideal-rigorous', or 'legacy-w2-proxy'"
        )
    if method == "mizuta2026-commutator-ideal-rigorous":
        estimate = candidate_estimate(segments)
    else:
        estimate = estimate_mpf_error(
            hamiltonian,
            time,
            segments,
            m,
            schedule=schedule,
            method=method,
            workers=execution.workers,
            _execution=execution,
        )
    return estimate


def select_mpf_segments(
    hamiltonian: PauliHamiltonian,
    time: float,
    target_error: float,
    m: int,
    *,
    schedule: MPFSchedule = "new",
    method: MPFErrorMethod = "low2019-l1-ideal-rigorous",
    workers: int = 1,
    _execution: CommutatorExecution | None = None,
) -> MPFErrorEstimate:
    """Select the smallest segment count satisfying the requested bound."""
    with execution_scope(workers, _execution) as execution:
        return _select_mpf_segments(
            hamiltonian,
            time,
            target_error,
            m,
            schedule=schedule,
            method=method,
            execution=execution,
        )


def _multiproduct_select_gate(
    hamiltonian: PauliHamiltonian,
    step_time: float,
    exponents: tuple[int, ...],
    branch_weights: np.ndarray,
    branch_width: int,
) -> Gate:
    """Return the named signed SELECT gate for one MPF LCU step.

    Physical MPF branches contain controlled second-order product formulas.
    The final two branches are cancelling positive and negative identities;
    any remaining computational states are unused identity branches.
    """
    branch = QuantumRegister(branch_width, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    select = QuantumCircuit(branch, system, name="SELECT_MPF")
    for j, weight in enumerate(branch_weights):
        if j < len(exponents):
            exponent = exponents[j]
            approximation = build_trotter_circuit(
                hamiltonian,
                step_time,
                reps=exponent,
                order=2,
            ).to_gate(label=f"S2({step_time:g}/{exponent})^{exponent}")
            select.append(
                approximation.control(branch_width, ctrl_state=j),
                [*branch, *system],
            )
        if weight < 0:
            sign = index_state_phase_gate(
                branch_width,
                j,
                np.pi,
                name="MPF_BRANCH_SIGN",
            )
            select.append(sign, branch)
    return select.to_gate(label="SELECT_MPF")


def _build_multiproduct_step_lcu(
    hamiltonian: PauliHamiltonian,
    step_time: float,
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> QuantumCircuit:
    """Build one normalized coherent MPF step before amplification.

    The all-zero branch block is exactly
    ``sum_j a_j S_2(step_time / k_j) ** k_j / 2``. Two cancelling identity
    branches pad the coefficient 1-norm to two, which is the normalization
    required by one round of three-step robust OAA.
    """
    if not np.isfinite(step_time):
        raise ValueError("step_time must be finite")

    exponents = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = multiproduct_coefficients(m, schedule=schedule)
    structure = mpf_lcu_structure(m, schedule=schedule)
    return _build_multiproduct_step_from_components(
        hamiltonian,
        step_time,
        m,
        schedule,
        exponents,
        tuple(float(value) for value in coefficients),
        structure,
    )


def _build_multiproduct_step_from_components(
    hamiltonian: PauliHamiltonian,
    step_time: float,
    m: int,
    schedule: MPFSchedule,
    exponents: tuple[int, ...],
    coefficients: tuple[float, ...],
    structure: MPFLCUStructure,
) -> QuantumCircuit:
    """Build one MPF LCU step from already-selected logical components."""
    coefficient_array = np.asarray(coefficients, dtype=float)
    coefficient_l1 = structure.coefficient_l1_norm
    padding_weight = structure.padding_weight
    branch_weights = np.concatenate((coefficient_array, [padding_weight / 2, -padding_weight / 2]))
    prepare = state_preparation(
        np.sqrt(np.abs(branch_weights) / _OAA_NORMALIZATION),
        name="PREPARE_MPF",
    )

    branch_width = prepare.num_qubits
    select = _multiproduct_select_gate(
        hamiltonian,
        step_time,
        exponents,
        branch_weights,
        branch_width,
    )
    branch = QuantumRegister(branch_width, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(branch, system, name=f"MPF_step_{2 * m}")
    circuit.append(prepare, branch)
    circuit.append(select, [*branch, *system])
    circuit.append(prepare.inverse(), branch)
    circuit.metadata = {
        "algorithm": "multiproduct-step-lcu",
        "m": int(m),
        "schedule": schedule,
        "exponents": exponents,
        "exponent_sum": int(sum(exponents)),
        "coefficients": list(coefficients),
        "coefficient_l1_norm": coefficient_l1,
        "padding_weight": padding_weight,
        "physical_branch_count": structure.physical_branch_count,
        "negative_coefficient_count": structure.negative_coefficient_count,
        "padding_branch_count": structure.padding_branch_count,
        "sign_branch_count": structure.sign_branch_count,
        "active_branch_count": structure.active_branch_count,
        "unused_branch_state_count": structure.unused_branch_state_count,
        "lcu_normalization": _OAA_NORMALIZATION,
        "formal_order": 2 * int(m),
        "step_time": float(step_time),
        "amplitude_amplification": False,
        "good_subspace": "branch register all-zero",
        "trotter_step_queries": int(sum(exponents)),
        "logical_gate_counts": {
            "prepare": 2,
            "select": 1,
            "good_reflection": 0,
            "controlled_u2": int(sum(exponents)),
        },
    }
    return circuit


def build_multiproduct_circuit(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int = 2,
    segments: int = 1,
    *,
    schedule: MPFSchedule = "new",
    amplitude_amplification: bool = True,
) -> QuantumCircuit:
    """Repeat robustly amplified MPF-step unitaries on shared ancillas.

    Before amplification the good block is ``B=M(step_time)/2``. One robust
    OAA round transforms it exactly to ``3B - 4 B B^dagger B``, which is close
    to ``M`` to the extent that the MPF approximation is unitary. The amplified
    step unitary is then repeated on the same branch register; its final good
    block is therefore not asserted to equal ``M**segments`` exactly. The
    unamplified form is exposed only for validating a single LCU step.
    """
    if isinstance(segments, bool) or not isinstance(segments, Integral):
        raise TypeError("segments must be an integer")
    if segments < 1:
        raise ValueError("segments must be positive")
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if not amplitude_amplification and segments != 1:
        raise ValueError("unamplified MPF is only supported for segments=1")

    exponents = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = tuple(float(value) for value in multiproduct_coefficients(m, schedule=schedule))
    structure = mpf_lcu_structure(m, schedule=schedule)
    return _build_multiproduct_circuit_from_components(
        hamiltonian,
        time,
        m,
        segments,
        schedule,
        amplitude_amplification,
        exponents,
        coefficients,
        structure,
    )


def _build_multiproduct_circuit_from_components(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int,
    segments: int,
    schedule: MPFSchedule,
    amplitude_amplification: bool,
    exponents: tuple[int, ...],
    coefficients: tuple[float, ...],
    structure: MPFLCUStructure,
) -> QuantumCircuit:
    """Compile a complete MPF circuit from selected logical components."""
    coefficient_l1 = structure.coefficient_l1_norm
    padding_weight = structure.padding_weight
    step_time = float(time) / int(segments)
    oaa_factor = 3 if amplitude_amplification else 1
    per_segment_queries = oaa_factor * sum(exponents)
    logical_counts_per_segment = {
        "prepare": 2 * oaa_factor,
        "select": oaa_factor,
        "good_reflection": 2 if amplitude_amplification else 0,
        "controlled_u2": int(per_segment_queries),
    }

    metadata = {
        "algorithm": "multiproduct",
        "construction": "robust-oaa-segments"
        if amplitude_amplification
        else "single-unamplified-step",
        "m": int(m),
        "schedule": schedule,
        "exponents": exponents,
        "exponent_sum": int(sum(exponents)),
        "segments": int(segments),
        "step_time": step_time,
        "coefficients": list(coefficients),
        "coefficient_l1_norm": coefficient_l1,
        "padding_weight": padding_weight,
        "physical_branch_count": structure.physical_branch_count,
        "negative_coefficient_count": structure.negative_coefficient_count,
        "padding_branch_count": structure.padding_branch_count,
        "sign_branch_count": structure.sign_branch_count,
        "active_branch_count": structure.active_branch_count,
        "unused_branch_state_count": structure.unused_branch_state_count,
        "lcu_normalization": _OAA_NORMALIZATION,
        "formal_order": 2 * int(m),
        "amplitude_amplification": bool(amplitude_amplification),
        "good_subspace": "branch register all-zero",
        "postselection": "measure branch register as all-zero",
        "trotter_step_queries_per_segment": int(per_segment_queries),
        "trotter_step_queries": int(segments * per_segment_queries),
        "base_lcu_uses_per_segment": oaa_factor,
        "logical_gate_counts_per_segment": logical_counts_per_segment,
        "logical_gate_counts": {
            key: int(segments * value) for key, value in logical_counts_per_segment.items()
        },
        "registers": {"branch": 0, "system": hamiltonian.num_qubits},
    }

    if time == 0:
        system = QuantumRegister(hamiltonian.num_qubits, "system")
        circuit = QuantumCircuit(system, name="MPF_identity")
        metadata.update(
            construction="zero-time-identity",
            good_subspace="no ancillas",
            postselection="none",
            trotter_step_queries_per_segment=0,
            trotter_step_queries=0,
            logical_gate_counts_per_segment={key: 0 for key in logical_counts_per_segment},
            logical_gate_counts={key: 0 for key in logical_counts_per_segment},
        )
        circuit.metadata = metadata
        return circuit

    base_step = _build_multiproduct_step_from_components(
        hamiltonian,
        step_time,
        m,
        schedule,
        exponents,
        coefficients,
        structure,
    )
    if amplitude_amplification:
        step = build_three_step_oaa(
            base_step,
            hamiltonian.num_qubits,
            name=f"MPF_step_{2 * m}_oaa",
            gate_label="MPF_step/2",
        )
    else:
        step = base_step

    branch_count = step.num_qubits - hamiltonian.num_qubits
    branch = QuantumRegister(branch_count, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(branch, system, name=f"MPF-{2 * m}")
    step_gate = step.to_gate(label=f"MPF step x{oaa_factor}")
    for _ in range(segments):
        circuit.append(step_gate, [*branch, *system])
    metadata["registers"]["branch"] = branch_count
    circuit.metadata = metadata
    return circuit


def build_multiproduct_circuit_from_plan(plan) -> QuantumCircuit:
    """Compile the logical branches and segments stored in an MPF plan."""
    from .planning import MPFPlan

    if not isinstance(plan, MPFPlan):
        raise TypeError("plan must be an MPFPlan")
    return _build_multiproduct_circuit_from_components(
        plan.hamiltonian,
        plan.time,
        plan.method.term_count,
        plan.segments,
        plan.method.schedule,
        True,
        plan.exponents,
        plan.coefficients,
        plan.lcu_structure,
    )
