"""Well-conditioned multiproduct-formula (MPF) LCU circuits.

Implements the coherent LCU construction based on Childs & Wiebe-style
multiproduct formulas and the well-conditioned schedules of Low, Kliuchnikov,
and Wiebe, arXiv:1907.11679. Each normalized LCU step is robustly amplified
before the same branch register is reused for the next simulation segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

from ._commutator_execution import (
    CommutatorExecution,
    CommutatorProgress,
    CommutatorProgressCallback,
    execution_scope,
)
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
    "childs2021-w2-triangle-ideal-rigorous",
    "mizuta2026-commutator-ideal-rigorous",
    "best-rigorous-ideal",
    "low-rigorous",
    "legacy-w2-proxy",
]
MPFErrorScope: TypeAlias = Literal["ideal-mpf", "amplified-shared-ancilla"]


@dataclass(frozen=True)
class MPFErrorEstimate:
    """Compatibility view of one family-specific MPF sizing estimate.

    ``m`` is the repository's backward-compatible name for the MPF branch
    count ``J``; ``formal_order`` is the paper's MPF order and equals ``2J``
    for the implemented symmetric second-order schedules.

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
    segment_diagnostics: MPFSegmentDiagnostics | None = None
    requested_method: MPFErrorMethod | None = None
    bound_candidates: tuple[MPFBoundCandidateSummary, ...] = ()


@dataclass(frozen=True)
class MPFBoundCandidateSummary:
    """Compact provenance for one estimator considered by a bound policy."""

    method: MPFErrorMethod
    segments: int
    error: float
    rigorous: bool
    fallback_reason: str | None = None
    max_exact_nested_commutator_order: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "segments": self.segments,
            "error": self.error,
            "rigorous": self.rigorous,
            "fallback_reason": self.fallback_reason,
            "max_exact_nested_commutator_order": self.max_exact_nested_commutator_order,
        }


@dataclass(frozen=True)
class MPFSegmentDiagnostics:
    """Exact lower-bound decomposition used by MPF segment selection.

    Each reported threshold is obtained by rerunning the corresponding
    candidate predicate.  In particular, Mizuta diagnostics recompute the
    candidate-dependent truncation order and commutator parameter rather than
    freezing their values at the selected row.
    """

    r_error: int
    r_time_1: int | None
    r_time_2: int | None
    active_constraints: tuple[str, ...]
    truncation_order_p0: int | None = None
    mu_upper: float | None = None
    auxiliary_error: float | None = None
    auxiliary_allocation_fraction: float | None = None
    local_commutator_error: float | None = None
    local_truncated_bch_error: float | None = None
    allocation_strategy: str = "not-applicable"
    additional_constraints: tuple[tuple[str, int], ...] = ()


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
    """Upper-bound Mizuta 2026 Eq. (47) by a finite polynomial root.

    Let ``A(x)=sum_q alpha_com,q x^q`` for ``base_order < q <= p0``.
    If ``A(x_*)=1``, every coefficient of ``A(x_*)^n`` is at most one.
    Therefore the supremum in Eq. (47) is at most ``1/x_*``. This retains
    every finite commutator order required by Theorem 4 without enumerating
    its unbounded repetition index ``n``.

    The formal-order restriction in Eq. (47) does not lower this supremum.
    The normalized weights ``alpha_com,q * x_*^q`` form a finite probability
    distribution.  A largest coefficient of its ``n``-fold convolution loses
    only a polynomial factor in ``n``, while its degree grows linearly and is
    eventually above every fixed formal-order cutoff.  Its degree root thus
    converges to ``1/x_*``.  The root is consequently sharp for nonzero
    supplied nonnegative data and remains a rigorous upper bound when some
    ``alpha_com,q`` values are themselves upper bounds.
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


def _w2_triangle_b2(
    coefficients: tuple[float, ...] | np.ndarray,
    exponents: tuple[int, ...],
) -> float:
    """Return the absolute-value branch factor ``sum_j |a_j| / k_j^2``."""
    if len(coefficients) != len(exponents) or not exponents:
        raise ValueError("coefficients and exponents must have the same nonzero length")
    if any(exponent < 1 for exponent in exponents):
        raise ValueError("MPF exponents must be positive")
    if any(not math.isfinite(float(coefficient)) for coefficient in coefficients):
        raise ValueError("MPF coefficients must be finite")
    return math.fsum(
        abs(float(coefficient)) / exponent**2
        for coefficient, exponent in zip(coefficients, exponents, strict=True)
    )


def _w2_triangle_log_ideal_mpf_bound(
    w2: float,
    time: float,
    segments: int,
    b2: float,
    *,
    unitary_single_branch: bool = False,
) -> tuple[float, float]:
    """Return log local and repeated W2-triangle errors without overflow."""
    if segments < 1:
        raise ValueError("segments must be positive")
    if w2 < 0 or b2 < 0:
        raise ValueError("W2 and B2 must be nonnegative")
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    if w2 == 0 or b2 == 0 or time == 0:
        return -math.inf, -math.inf
    log_segments = math.log(segments)
    log_delta = (
        math.log(w2)
        + 3 * (math.log(abs(float(time))) - log_segments)
        + math.log(b2)
    )
    if unitary_single_branch:
        return log_delta, log_segments + log_delta
    log_error = log_segments + log_delta + (segments - 1) * _log1p_exp(log_delta)
    return log_delta, log_error


def _w2_triangle_ideal_mpf_bound(
    w2: float,
    time: float,
    segments: int,
    coefficients: tuple[float, ...] | np.ndarray,
    exponents: tuple[int, ...],
) -> tuple[float, float, float, float]:
    """Return repeated error, prefactor, local error, and B2.

    A single coefficient-one branch is itself unitary, so its repeated bound
    uses the sharper unitary telescoping factor ``r * delta``. Registered MPF
    schedules have at least two branches; the special case supports the
    ordinary-Strang regression without extending the public LCU schedule API.
    """
    b2 = _w2_triangle_b2(coefficients, exponents)
    unitary_single_branch = len(coefficients) == 1 and float(coefficients[0]) == 1.0
    log_local, log_error = _w2_triangle_log_ideal_mpf_bound(
        w2,
        time,
        segments,
        b2,
        unitary_single_branch=unitary_single_branch,
    )
    prefactor = w2 * b2
    local_error = math.exp(log_local) if log_local < 709 else math.inf
    error = math.exp(log_error) if log_error < 709 else math.inf
    return error, prefactor, local_error, b2


@dataclass(frozen=True)
class _MizutaAllocationCandidate:
    """One exact printed-theorem evaluation at fixed ``(r, p0, rho)``."""

    allocation_fraction: float
    auxiliary_error: float
    truncation_order: int
    mu_upper: float
    prefactor: float
    local_commutator_error: float
    local_truncated_bch_error: float
    local_step_error: float
    repeated_error: float
    first_time_limit: float
    second_time_limit: float
    commutators: PauliNestedCommutatorBounds
    commuting_exact: bool = False
    zero_time_exact: bool = False


@dataclass(frozen=True)
class _MizutaCandidateEvaluation:
    """Allocation-aware predicates for one candidate segment count."""

    selected: _MizutaAllocationCandidate
    error_satisfied: bool
    first_time_satisfied: bool
    second_time_satisfied: bool
    full_satisfied: bool
    error_and_time_1_satisfied: bool
    error_and_time_2_satisfied: bool
    time_1_and_time_2_satisfied: bool


def _mizuta_truncation_order(
    hamiltonian: PauliHamiltonian,
    local_budget: float,
    allocation_scale: float,
    allocation_fraction: float,
) -> int:
    auxiliary_error = allocation_fraction * local_budget / allocation_scale
    if auxiliary_error <= 0:
        raise OverflowError("Mizuta auxiliary error underflowed float range")
    return max(3, math.ceil(math.log(3 * hamiltonian.num_qubits / auxiliary_error)))


def _mizuta_minimum_fraction_for_order(
    hamiltonian: PauliHamiltonian,
    local_budget: float,
    allocation_scale: float,
    truncation_order: int,
) -> float:
    """Return the smallest floating ``rho`` assigned to one discrete ``p0``."""
    log_scale = math.log(3 * hamiltonian.num_qubits * allocation_scale / local_budget)
    # The exact lower endpoint is exp(log_scale-p0).  Move the target exponent
    # one float below p0 so the production ceil calculation robustly maps the
    # returned fraction to p0 rather than p0+1 at a rounded boundary.
    target = float(np.nextafter(float(truncation_order), -np.inf))
    fraction = math.exp(log_scale - target)
    if not 0 < fraction < 1:
        raise ValueError("truncation order has no allocation fraction in (0, 1)")
    for _ in range(64):
        derived = _mizuta_truncation_order(
            hamiltonian,
            local_budget,
            allocation_scale,
            fraction,
        )
        if derived == truncation_order:
            return fraction
        direction = np.inf if derived > truncation_order else 0.0
        fraction = float(np.nextafter(fraction, direction))
    raise ArithmeticError("could not reproduce the discrete Mizuta truncation order")


def _mizuta_allocation_candidate(
    hamiltonian: PauliHamiltonian,
    time: float,
    term_count: int,
    segments: int,
    coefficient_l1_norm: float,
    exponent_l1_norm: float,
    target_error: float,
    allocation_fraction: float,
    truncation_order: int,
    execution: CommutatorExecution | None,
) -> _MizutaAllocationCandidate:
    """Evaluate Theorem 4 at one allocation without freezing any input."""
    mizuta_formal_order = 2 * term_count
    base_order = 2
    base_repetitions = 2
    step_time = abs(float(time)) / segments
    local_budget = math.expm1(math.log1p(target_error) / segments)
    allocation_scale = coefficient_l1_norm * exponent_l1_norm
    auxiliary_error = allocation_fraction * local_budget / allocation_scale
    if execution is not None:
        execution.report(
            CommutatorProgress(
                family="multiproduct",
                phase="segment-candidate",
                completed=0,
                total=None,
                commutator_order=truncation_order,
                max_commutator_order=truncation_order,
                formula_order=mizuta_formal_order,
                system_qubits=hamiltonian.num_qubits,
                segment_candidate=segments,
                target_error=target_error,
            )
        )
    commutators = pauli_nested_commutator_bounds(
        hamiltonian,
        truncation_order,
        workers=execution.workers if execution is not None else 1,
        _execution=execution,
        _formula_order=mizuta_formal_order,
        _segment_candidate=segments,
        _target_error=target_error,
    )
    commuting_exact = all(value == 0 for value in commutators.values)
    zero_time_exact = step_time == 0
    if commuting_exact:
        return _MizutaAllocationCandidate(
            allocation_fraction=allocation_fraction,
            auxiliary_error=auxiliary_error,
            truncation_order=truncation_order,
            mu_upper=0.0,
            prefactor=0.0,
            local_commutator_error=0.0,
            local_truncated_bch_error=0.0,
            local_step_error=0.0,
            repeated_error=0.0,
            first_time_limit=math.inf,
            second_time_limit=math.inf,
            commutators=commutators,
            commuting_exact=True,
            zero_time_exact=zero_time_exact,
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
            + (mizuta_formal_order + 1) * math.log(base_repetitions * mu_upper)
        )
        prefactor = math.exp(log_prefactor) if log_prefactor < 709 else math.inf
        if step_time == 0:
            local_commutator_error = 0.0
        else:
            log_commutator_error = log_prefactor + (mizuta_formal_order + 1) * math.log(
                step_time
            )
            local_commutator_error = (
                math.exp(log_commutator_error) if log_commutator_error < 709 else math.inf
            )
    local_truncated_bch_error = allocation_scale * auxiliary_error
    if zero_time_exact:
        # The allocation term only bounds a truncated representation. At
        # tau=0 the product formula, MPF, and exact evolution are identical.
        local_truncated_bch_error = 0.0
    local_step_error = local_commutator_error + local_truncated_bch_error
    k_value = commutators.locality_k
    g_value = commutators.extensiveness_g
    first_time_limit = (
        math.inf
        if k_value == 0 or g_value == 0
        else 1 / (8 * math.e**3 * base_repetitions * truncation_order * k_value * g_value)
    )
    second_time_limit = math.inf if mu_upper == 0 else 1 / (2 * base_repetitions * mu_upper)
    return _MizutaAllocationCandidate(
        allocation_fraction=allocation_fraction,
        auxiliary_error=auxiliary_error,
        truncation_order=truncation_order,
        mu_upper=mu_upper,
        prefactor=prefactor,
        local_commutator_error=local_commutator_error,
        local_truncated_bch_error=local_truncated_bch_error,
        local_step_error=local_step_error,
        repeated_error=_repeated_step_error(local_step_error, segments),
        first_time_limit=first_time_limit,
        second_time_limit=second_time_limit,
        commutators=commutators,
        zero_time_exact=zero_time_exact,
    )


def _evaluate_mizuta_candidate(
    hamiltonian: PauliHamiltonian,
    time: float,
    term_count: int,
    segments: int,
    coefficient_l1_norm: float,
    exponents: tuple[int, ...],
    target_error: float,
    execution: CommutatorExecution | None,
    allocation_fraction: float | None,
) -> _MizutaCandidateEvaluation:
    """Evaluate fixed or exactly optimized printed-theorem allocations."""
    exponent_l1_norm = float(sum(exponents))
    allocation_scale = coefficient_l1_norm * exponent_l1_norm
    local_budget = math.expm1(math.log1p(target_error) / segments)
    if local_budget <= 0:
        raise OverflowError("Mizuta local error budget underflowed float range")

    def evaluate(fraction: float, order: int) -> _MizutaAllocationCandidate:
        return _mizuta_allocation_candidate(
            hamiltonian,
            time,
            term_count,
            segments,
            coefficient_l1_norm,
            exponent_l1_norm,
            target_error,
            fraction,
            order,
            execution,
        )

    if allocation_fraction is not None:
        if not 0 < allocation_fraction < 1:
            raise ValueError("auxiliary_allocation_fraction must lie in (0, 1)")
        order = _mizuta_truncation_order(
            hamiltonian,
            local_budget,
            allocation_scale,
            allocation_fraction,
        )
        candidate = evaluate(allocation_fraction, order)
        error_ok = candidate.repeated_error <= target_error
        first_ok = abs(float(time)) / segments <= candidate.first_time_limit
        second_ok = abs(float(time)) / segments <= candidate.second_time_limit
        return _MizutaCandidateEvaluation(
            selected=candidate,
            error_satisfied=error_ok,
            first_time_satisfied=first_ok,
            second_time_satisfied=second_ok,
            full_satisfied=error_ok and first_ok and second_ok,
            error_and_time_1_satisfied=error_ok and first_ok,
            error_and_time_2_satisfied=error_ok and second_ok,
            time_1_and_time_2_satisfied=first_ok and second_ok,
        )

    nearly_one = float(np.nextafter(1.0, 0.0))
    minimum_order = _mizuta_truncation_order(
        hamiltonian,
        local_budget,
        allocation_scale,
        nearly_one,
    )
    first_candidate = evaluate(
        _mizuta_minimum_fraction_for_order(
            hamiltonian,
            local_budget,
            allocation_scale,
            minimum_order,
        ),
        minimum_order,
    )
    if first_candidate.commuting_exact or first_candidate.zero_time_exact:
        return _MizutaCandidateEvaluation(
            selected=first_candidate,
            error_satisfied=True,
            first_time_satisfied=True,
            second_time_satisfied=True,
            full_satisfied=True,
            error_and_time_1_satisfied=True,
            error_and_time_2_satisfied=True,
            time_1_and_time_2_satisfied=True,
        )

    step_time = abs(float(time)) / segments
    best_full: _MizutaAllocationCandidate | None = None
    best_time_feasible: _MizutaAllocationCandidate | None = None
    first_error: _MizutaAllocationCandidate | None = None
    first_ok_exists = False
    second_ok_exists = False
    error_time_1_exists = False
    error_time_2_exists = False
    time_1_time_2_exists = False
    order = minimum_order
    candidate = first_candidate
    while True:
        error_ok = candidate.repeated_error <= target_error
        first_ok = step_time <= candidate.first_time_limit
        second_ok = step_time <= candidate.second_time_limit
        first_ok_exists = first_ok_exists or first_ok
        second_ok_exists = second_ok_exists or second_ok
        error_time_1_exists = error_time_1_exists or (error_ok and first_ok)
        error_time_2_exists = error_time_2_exists or (error_ok and second_ok)
        time_1_time_2_exists = time_1_time_2_exists or (first_ok and second_ok)
        if error_ok and first_error is None:
            first_error = candidate
        if first_ok and second_ok and (
            best_time_feasible is None
            or candidate.repeated_error < best_time_feasible.repeated_error
        ):
            best_time_feasible = candidate
        if error_ok and first_ok and second_ok and (
            best_full is None or candidate.repeated_error < best_full.repeated_error
        ):
            best_full = candidate

        commutator_only_error = _repeated_step_error(
            candidate.local_commutator_error,
            segments,
        )
        no_future_error = commutator_only_error > target_error
        # The first hypothesis makes the full feasible p0 range finite.  Walk
        # through its first failing order even after a valid allocation is
        # found so every allowed discrete p0 participates in the minimization.
        full_search_done = not first_ok
        error_search_done = first_error is not None or no_future_error
        error_time_1_done = error_time_1_exists or no_future_error or not first_ok
        error_time_2_done = error_time_2_exists or no_future_error or not second_ok
        if full_search_done and error_search_done and error_time_1_done and error_time_2_done:
            break

        order += 1
        fraction = _mizuta_minimum_fraction_for_order(
            hamiltonian,
            local_budget,
            allocation_scale,
            order,
        )
        candidate = evaluate(fraction, order)

    selected = best_time_feasible or first_error or first_candidate
    return _MizutaCandidateEvaluation(
        selected=selected,
        error_satisfied=first_error is not None,
        first_time_satisfied=first_ok_exists,
        second_time_satisfied=second_ok_exists,
        full_satisfied=best_full is not None,
        error_and_time_1_satisfied=error_time_1_exists,
        error_and_time_2_satisfied=error_time_2_exists,
        time_1_and_time_2_satisfied=time_1_time_2_exists,
    )


def _mizuta_ideal_mpf_bound(
    hamiltonian: PauliHamiltonian,
    time: float,
    term_count: int,
    segments: int,
    coefficient_l1_norm: float,
    exponents: tuple[int, ...],
    target_error: float,
    execution: CommutatorExecution | None = None,
    allocation_fraction: float | None = None,
) -> tuple[
    float,
    float,
    tuple[tuple[str, float], ...],
    PauliNestedCommutatorBounds,
    tuple[str, ...],
    bool,
]:
    """Evaluate Mizuta 2026 Theorem 4/Eqs. (47)--(49) for ``p=2``."""
    evaluation = _evaluate_mizuta_candidate(
        hamiltonian,
        time,
        term_count,
        segments,
        coefficient_l1_norm,
        exponents,
        target_error,
        execution,
        allocation_fraction,
    )
    candidate = evaluation.selected
    selected_step_time = abs(float(time)) / segments
    time_hypothesis_satisfied = selected_step_time <= min(
        candidate.first_time_limit,
        candidate.second_time_limit,
    )
    assumptions = (
        "individual Pauli summands define the ordered H_gamma decomposition",
        "Pauli support gives k-locality and per-site coefficient sums give g-extensiveness",
        "base formula is the symmetric second-order formula with c_p=2",
        (
            "Theorem 4 time hypothesis Eq. (48) is satisfied"
            if time_hypothesis_satisfied
            else "Theorem 4 time hypothesis Eq. (48) is not satisfied"
        ),
        "the repeated ideal MPF is composed with the Eq. (15) telescoping argument",
    )
    components = (
        ("mu_upper", candidate.mu_upper),
        ("local_commutator_error", candidate.local_commutator_error),
        ("local_truncated_bch_error", candidate.local_truncated_bch_error),
        ("local_step_error", candidate.local_step_error),
        ("auxiliary_error", candidate.auxiliary_error),
        ("auxiliary_allocation_fraction", candidate.allocation_fraction),
        ("truncation_order_p0", float(candidate.truncation_order)),
        ("first_time_limit", candidate.first_time_limit),
        ("second_time_limit", candidate.second_time_limit),
        ("locality_k", float(candidate.commutators.locality_k)),
        ("extensiveness_g", candidate.commutators.extensiveness_g),
        ("error_predicate_satisfied", float(evaluation.error_satisfied)),
        ("time_1_predicate_satisfied", float(evaluation.first_time_satisfied)),
        ("time_2_predicate_satisfied", float(evaluation.second_time_satisfied)),
        ("joint_predicate_satisfied", float(evaluation.full_satisfied)),
        ("error_and_time_1_satisfied", float(evaluation.error_and_time_1_satisfied)),
        ("error_and_time_2_satisfied", float(evaluation.error_and_time_2_satisfied)),
        ("time_1_and_time_2_satisfied", float(evaluation.time_1_and_time_2_satisfied)),
    )
    if candidate.commuting_exact:
        assumptions = (
            "all individual Pauli summands commute; the ordered Strang formula is exact",
        )
    elif candidate.zero_time_exact:
        assumptions = ("zero-time evolution is exact",)
    error = candidate.repeated_error if time_hypothesis_satisfied else math.inf
    return (
        error,
        candidate.prefactor,
        components,
        candidate.commutators,
        assumptions,
        time_hypothesis_satisfied,
    )


def _normalize_mpf_error_method(method: MPFErrorMethod) -> MPFErrorMethod:
    """Map the historical Low method name to its explicit canonical name."""
    if method == "low-rigorous":
        return "low2019-l1-ideal-rigorous"
    return method


def _estimate_mpf_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    m: int,
    *,
    schedule: MPFSchedule = "new",
    method: MPFErrorMethod = "low2019-l1-ideal-rigorous",
    target_error: float | None = None,
    auxiliary_allocation_fraction: float | None = None,
    execution: CommutatorExecution,
) -> MPFErrorEstimate:
    """Estimate ideal-MPF error while preserving certification provenance.

    Mizuta estimates optimize the printed-theorem auxiliary allocation when
    ``auxiliary_allocation_fraction`` is ``None``. A value in ``(0, 1)`` fixes
    that fraction, primarily to reproduce the former 50/50 audit baseline.
    """
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
    w2_commutator_bound = False
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
    elif method == "childs2021-w2-triangle-ideal-rigorous":
        _, w2 = suzuki_commutator_bounds(hamiltonian)
        error, prefactor, local_error, b2 = _w2_triangle_ideal_mpf_bound(
            w2,
            time,
            int(segments),
            coefficients,
            exponents,
        )
        rigorous = True
        local_error_rigorous = True
        reference = (
            "Childs, Su, Tran, Wiebe, and Zhu, Phys. Rev. X 11, 011020 (2021)"
        )
        theorem_or_equations = (
            "Propositions 9--10 Strang bound; repository MPF triangle and "
            "repeated-step telescoping derivation"
        )
        local_step_size = abs(float(time)) / int(segments)
        bound_components = (
            ("w2", w2),
            ("b2", b2),
            ("local_step_size", local_step_size),
            ("local_step_error", local_error),
            ("repeated_ideal_mpf_error", error),
        )
        assumptions = (
            "H is decomposed into the ordered individual Pauli summands",
            "each branch uses the same ordered symmetric second-order formula",
            "branch and MPF errors are combined only by triangle inequalities",
            "no MPF cancellation condition is used",
            "the bound certifies the repeated ideal MPF operator only",
        )
        commutators = None
        w2_commutator_bound = True
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
            execution,
            auxiliary_allocation_fraction,
        )
        reference = "Mizuta, Quantum 10, 1974 (2026), arXiv:2507.06557v4"
        theorem_or_equations = (
            "Theorem 4, Eqs. (47)--(49), with Theorem 3, Eqs. (33)--(35)"
        )
        local_error = dict(bound_components).get("local_step_error", 0.0)
        local_error_rigorous = rigorous
    elif method == "best-rigorous-ideal":
        raise ValueError(
            "best-rigorous-ideal is a segment-selection policy; "
            "use select_mpf_segments rather than estimate_mpf_error"
        )
    else:
        raise ValueError(
            "MPF error method must be 'low2019-l1-ideal-rigorous' "
            "(historical alias 'low-rigorous'), "
            "'childs2021-w2-triangle-ideal-rigorous', "
            "'mizuta2026-commutator-ideal-rigorous', "
            "'best-rigorous-ideal', or 'legacy-w2-proxy'"
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
        max_nested_commutator_order=(
            commutators.max_order
            if commutators is not None
            else (3 if w2_commutator_bound else 0)
        ),
        max_exact_nested_commutator_order=(
            commutators.max_exact_order
            if commutators is not None
            else (3 if w2_commutator_bound else 0)
        ),
        locality_compatible=commutators is not None or w2_commutator_bound,
        commutator_bounds=(
            tuple((order, commutators.at(order)) for order in range(2, commutators.max_order + 1))
            if commutators is not None
            else ()
        ),
        requested_method=method,
    )


def estimate_mpf_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    m: int,
    *,
    schedule: MPFSchedule = "new",
    method: MPFErrorMethod = "low2019-l1-ideal-rigorous",
    target_error: float | None = None,
    auxiliary_allocation_fraction: float | None = None,
    workers: int = 1,
    progress: CommutatorProgressCallback | None = None,
    _execution: CommutatorExecution | None = None,
) -> MPFErrorEstimate:
    """Estimate ideal-MPF error while preserving certification provenance."""
    with execution_scope(workers, _execution, progress) as execution:
        return _estimate_mpf_error(
            hamiltonian,
            time,
            segments,
            m,
            schedule=schedule,
            method=method,
            target_error=target_error,
            auxiliary_allocation_fraction=auxiliary_allocation_fraction,
            execution=execution,
        )


def _select_mpf_segments(
    hamiltonian: PauliHamiltonian,
    time: float,
    target_error: float,
    m: int,
    *,
    schedule: MPFSchedule = "new",
    method: MPFErrorMethod = "low2019-l1-ideal-rigorous",
    auxiliary_allocation_fraction: float | None = None,
    execution: CommutatorExecution,
) -> MPFErrorEstimate:
    """Select segments and return the resulting scoped MPF estimate.

    ``low2019-l1-ideal-rigorous`` uses Eq. (16) of Low, Kliuchnikov, and
    Wiebe, arXiv:1907.11679, only as a safe upper bracket, then finds the
    smallest integer satisfying their direct Eqs. (14)--(15) bound by binary
    search. ``low-rigorous`` remains a backward-compatible alias.
    ``childs2021-w2-triangle-ideal-rigorous`` composes the rigorous Strang W2
    error through branch, MPF, and repeated-step triangle inequalities.
    ``legacy-w2-proxy`` exactly reproduces the historical heuristic.
    """
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if not 0 < target_error <= 1:
        raise ValueError("target_error must lie in (0, 1]")
    optimal_mpf_exponents(m, schedule=schedule)
    method = _normalize_mpf_error_method(method)
    if method == "best-rigorous-ideal":
        candidates = tuple(
            _select_mpf_segments(
                hamiltonian,
                time,
                target_error,
                m,
                schedule=schedule,
                method=candidate_method,
                auxiliary_allocation_fraction=auxiliary_allocation_fraction,
                execution=execution,
            )
            for candidate_method in (
                "low2019-l1-ideal-rigorous",
                "childs2021-w2-triangle-ideal-rigorous",
                "mizuta2026-commutator-ideal-rigorous",
            )
        )
        tie_break_priority = {
            "low2019-l1-ideal-rigorous": 0,
            "childs2021-w2-triangle-ideal-rigorous": 1,
            "mizuta2026-commutator-ideal-rigorous": 2,
        }
        chosen = min(
            candidates,
            key=lambda item: (
                item.segments,
                item.error,
                tie_break_priority[item.method],
            ),
        )
        summaries = tuple(
            MPFBoundCandidateSummary(
                method=item.method,
                segments=item.segments,
                error=item.error,
                rigorous=item.rigorous,
                fallback_reason=item.fallback_reason,
                max_exact_nested_commutator_order=item.max_exact_nested_commutator_order,
            )
            for item in candidates
        )
        return replace(
            chosen,
            requested_method="best-rigorous-ideal",
            bound_candidates=summaries,
        )
    if method == "legacy-w2-proxy":
        segments = legacy_w2_proxy_segments(hamiltonian, time, target_error, m)
    elif method == "childs2021-w2-triangle-ideal-rigorous":
        coefficients = multiproduct_coefficients(m, schedule=schedule)
        exponents = optimal_mpf_exponents(m, schedule=schedule)
        b2 = _w2_triangle_b2(coefficients, exponents)
        _, w2 = suzuki_commutator_bounds(hamiltonian)
        if w2 == 0 or time == 0:
            segments = 1
        else:
            target_log = math.log(target_error)

            def satisfies_w2(candidate: int) -> bool:
                _, log_error = _w2_triangle_log_ideal_mpf_bound(
                    w2,
                    time,
                    candidate,
                    b2,
                )
                return log_error <= target_log

            segments = 1
            while not satisfies_w2(segments):
                segments *= 2
                if math.log(segments) > 709:
                    raise OverflowError("required MPF segment count exceeds float range")
            if segments > 1:
                lower = 1
                upper = segments
                while lower + 1 < upper:
                    midpoint = (lower + upper) // 2
                    if satisfies_w2(midpoint):
                        upper = midpoint
                    else:
                        lower = midpoint
                segments = upper
            if not satisfies_w2(segments):
                raise AssertionError("selected W2-triangle segment fails its error bound")
            if segments > 1 and satisfies_w2(segments - 1):
                raise AssertionError("W2-triangle segment selection is not minimal")
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
                    auxiliary_allocation_fraction=auxiliary_allocation_fraction,
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
            "'childs2021-w2-triangle-ideal-rigorous', "
            "'mizuta2026-commutator-ideal-rigorous', "
            "'best-rigorous-ideal', or 'legacy-w2-proxy'"
        )
    if method == "mizuta2026-commutator-ideal-rigorous":
        estimate = candidate_estimate(segments)

        def predicate_value(candidate: int, name: str) -> bool:
            return bool(dict(candidate_estimate(candidate).bound_components)[name])

        def error_satisfied(candidate: int) -> bool:
            return predicate_value(candidate, "error_predicate_satisfied")

        def first_time_satisfied(candidate: int) -> bool:
            return predicate_value(candidate, "time_1_predicate_satisfied")

        def second_time_satisfied(candidate: int) -> bool:
            return predicate_value(candidate, "time_2_predicate_satisfied")

        def smallest_component(predicate) -> int:
            if predicate(1):
                return 1
            lower = 1
            upper = segments
            while lower + 1 < upper:
                midpoint = (lower + upper) // 2
                if predicate(midpoint):
                    upper = midpoint
                else:
                    lower = midpoint
            if not predicate(upper) or (upper > 1 and predicate(upper - 1)):
                raise AssertionError("Mizuta segment diagnostic is not independently minimal")
            return upper

        r_error = smallest_component(error_satisfied)
        r_time_1 = smallest_component(first_time_satisfied)
        r_time_2 = smallest_component(second_time_satisfied)
        components = dict(estimate.bound_components)
        commuting_exact = all(value == 0 for _, value in estimate.commutator_bounds)
        additional_constraints: tuple[tuple[str, int], ...] = ()
        if commuting_exact:
            active_constraints = ("commuting_exact",)
        else:
            thresholds = {
                "error": r_error,
                "time_1": r_time_1,
                "time_2": r_time_2,
            }
            independent_maximum = max(thresholds.values())
            if auxiliary_allocation_fraction is not None and independent_maximum != segments:
                raise AssertionError("fixed-allocation diagnostics do not recover selection")
            if independent_maximum > segments:
                raise AssertionError("Mizuta component threshold exceeds joint selection")
            if independent_maximum == segments:
                active_constraints = tuple(
                    name for name, threshold in thresholds.items() if threshold == segments
                )
            else:
                active_constraints = ("joint_allocation",)
                additional_constraints = (("joint_allocation", segments),)
            if not predicate_value(segments, "joint_predicate_satisfied"):
                raise AssertionError("selected Mizuta segment fails the complete predicate")
            if segments > 1:
                if predicate_value(segments - 1, "joint_predicate_satisfied"):
                    raise AssertionError("Mizuta segment selection is not jointly minimal")
                if active_constraints != ("joint_allocation",):
                    previous_failures = {
                        "error": not error_satisfied(segments - 1),
                        "time_1": not first_time_satisfied(segments - 1),
                        "time_2": not second_time_satisfied(segments - 1),
                    }
                    if not any(previous_failures[name] for name in active_constraints):
                        raise AssertionError("selected Mizuta segment has no active failing predicate")
        estimate = replace(
            estimate,
            segment_diagnostics=MPFSegmentDiagnostics(
                r_error=r_error,
                r_time_1=r_time_1,
                r_time_2=r_time_2,
                active_constraints=active_constraints,
                truncation_order_p0=int(components["truncation_order_p0"]),
                mu_upper=components["mu_upper"],
                auxiliary_error=components["auxiliary_error"],
                auxiliary_allocation_fraction=components["auxiliary_allocation_fraction"],
                local_commutator_error=components["local_commutator_error"],
                local_truncated_bch_error=components["local_truncated_bch_error"],
                allocation_strategy=(
                    "optimized-discrete-p0"
                    if auxiliary_allocation_fraction is None
                    else "fixed-local-budget-fraction"
                ),
                additional_constraints=additional_constraints,
            ),
        )
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
        estimate = replace(
            estimate,
            segment_diagnostics=MPFSegmentDiagnostics(
                r_error=segments,
                r_time_1=None,
                r_time_2=None,
                active_constraints=(
                    "error" if estimate.rigorous else "heuristic_error",
                ),
            ),
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
    auxiliary_allocation_fraction: float | None = None,
    workers: int = 1,
    progress: CommutatorProgressCallback | None = None,
    _execution: CommutatorExecution | None = None,
) -> MPFErrorEstimate:
    """Select the smallest segment count satisfying the requested bound.

    Mizuta selection exactly optimizes its discrete auxiliary allocation by
    default. Set ``auxiliary_allocation_fraction`` to a value in ``(0, 1)``
    only when a fixed-allocation comparison is required.
    """
    with execution_scope(workers, _execution, progress) as execution:
        return _select_mpf_segments(
            hamiltonian,
            time,
            target_error,
            m,
            schedule=schedule,
            method=method,
            auxiliary_allocation_fraction=auxiliary_allocation_fraction,
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
