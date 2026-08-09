"""Positive scalar majorants for Mizuta's finite-size BCH proof.

The routines in this module deliberately keep the proof's Suzuki schedule in
view.  In particular, they do not replace the schedule by the coarse factor
``c_p`` before applying the extensiveness estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .hamiltonians import PauliHamiltonian
from .trotter import suzuki_group_factors


_NEGATIVE_INFINITY = float("-inf")


def _round_up(value: float) -> float:
    """Round a nonnegative finite floating-point result toward ``+inf``."""
    if value == 0.0 or math.isinf(value):
        return value
    return float(np.nextafter(value, np.inf))


def _positive_sum_up(values: tuple[float, ...] | list[float]) -> float:
    """Accumulate nonnegative values with directed outward rounding."""
    total = 0.0
    for value in values:
        if value < 0.0:
            raise ValueError("positive majorants cannot contain negative values")
        total = _round_up(total + value)
    return total


def _logaddexp_up(left: float, right: float) -> float:
    """Return an outward-rounded log of ``exp(left) + exp(right)``."""
    if left == _NEGATIVE_INFINITY:
        return right
    if right == _NEGATIVE_INFINITY:
        return left
    maximum = max(left, right)
    minimum = min(left, right)
    value = maximum + math.log1p(math.exp(minimum - maximum))
    return _round_up(value)


@dataclass(frozen=True)
class MizutaScheduleWeights:
    """Absolute Suzuki schedule weights and their induced extensiveness.

    ``group_weights[gamma]`` is
    ``sum_{v: gamma_v=gamma} |alpha_v|``.  The weighted extensiveness is

    ``max_i sum_gamma group_weights[gamma]
                  sum_{X contains i} ||h_X^gamma||``.

    The current MPF analysis uses one Pauli term per Suzuki group, matching the
    decomposition used by the audited nested-commutator calculation.
    """

    group_weights: tuple[float, ...]
    weighted_extensiveness: float
    locality_k: int

    @property
    def maximum_group_weight(self) -> float:
        return max(self.group_weights, default=0.0)


def mizuta_schedule_weights(
    hamiltonian: PauliHamiltonian,
    *,
    suzuki_order: int = 2,
) -> MizutaScheduleWeights:
    """Resolve the actual Suzuki factors and form their proof weights.

    Every occurrence of a group in ``T_p(tau)`` contributes its absolute
    coefficient.  This is the quantity entering the triangle inequalities in
    Mizuta Lemma 9, Eqs. (109), (112), and (114).  For Strang splitting, a
    noncentral group occurs twice with coefficient ``1/2`` and therefore has
    total absolute weight one.
    """
    factors = suzuki_group_factors(hamiltonian.term_count, suzuki_order)
    weights = [0.0] * hamiltonian.term_count
    for group, coefficient in factors:
        weights[group] = _round_up(weights[group] + abs(coefficient))

    site_weights = [0.0] * hamiltonian.num_qubits
    locality_k = 0
    for group, (label, coefficient) in enumerate(hamiltonian.terms):
        support = [site for site, symbol in enumerate(reversed(label)) if symbol != "I"]
        locality_k = max(locality_k, len(support))
        weighted_norm = _round_up(weights[group] * abs(coefficient))
        for site in support:
            site_weights[site] = _round_up(site_weights[site] + weighted_norm)

    return MizutaScheduleWeights(
        group_weights=tuple(weights),
        weighted_extensiveness=max(site_weights, default=0.0),
        locality_k=locality_k,
    )


def lemma9_log_coefficients(
    locality_k: int,
    weighted_extensiveness: float,
    maximum_order: int,
) -> tuple[float, ...]:
    """Return logs of the Lemma-9 majorant coefficients through ``maximum_order``.

    With ``a=2*k*g_alpha`` and ``b=g_alpha``, the generating function is

    ``A(z) = exp(b*z/(1-a*z))``.

    Differentiating it gives the positive recurrence

    ``q A_q = b sum_{j=0}^{q-1} (j+1) a^j A_{q-1-j}``,

    which is evaluated in the log domain.  ``A_0=1`` is included in the
    returned tuple.
    """
    if locality_k < 0:
        raise ValueError("locality_k must be nonnegative")
    if not math.isfinite(weighted_extensiveness) or weighted_extensiveness < 0.0:
        raise ValueError("weighted_extensiveness must be finite and nonnegative")
    if maximum_order < 0:
        raise ValueError("maximum_order must be nonnegative")

    coefficients = [_NEGATIVE_INFINITY] * (maximum_order + 1)
    coefficients[0] = 0.0
    if locality_k == 0 or weighted_extensiveness == 0.0:
        return tuple(coefficients)

    log_a = math.log(2.0 * locality_k * weighted_extensiveness)
    log_b = math.log(weighted_extensiveness)
    for order in range(1, maximum_order + 1):
        log_total = _NEGATIVE_INFINITY
        for power in range(order):
            term = (
                log_b
                + math.log(power + 1.0)
                + power * log_a
                + coefficients[order - 1 - power]
            )
            log_total = _logaddexp_up(log_total, term)
        coefficients[order] = _round_up(log_total - math.log(order))
    return tuple(coefficients)


def lemma9_direct_log_coefficient(
    locality_k: int,
    weighted_extensiveness: float,
    order: int,
) -> float:
    """Evaluate the order-resolved finite sum used to audit the recurrence."""
    if order < 1:
        raise ValueError("order must be positive")
    if locality_k < 0:
        raise ValueError("locality_k must be nonnegative")
    if not math.isfinite(weighted_extensiveness) or weighted_extensiveness < 0.0:
        raise ValueError("weighted_extensiveness must be finite and nonnegative")
    if locality_k == 0 or weighted_extensiveness == 0.0:
        return _NEGATIVE_INFINITY

    log_a = math.log(2.0 * locality_k * weighted_extensiveness)
    log_b = math.log(weighted_extensiveness)
    log_total = _NEGATIVE_INFINITY
    for insertions in range(1, order + 1):
        log_binomial = (
            math.lgamma(order)
            - math.lgamma(insertions)
            - math.lgamma(order - insertions + 1)
        )
        term = (
            (order - insertions) * log_a
            + insertions * log_b
            - math.lgamma(insertions + 1)
            + log_binomial
        )
        log_total = _logaddexp_up(log_total, term)
    return log_total
