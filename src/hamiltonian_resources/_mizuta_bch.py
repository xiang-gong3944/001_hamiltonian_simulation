"""Positive scalar majorants for Mizuta's finite-size BCH proof.

The routines in this module deliberately keep the proof's Suzuki schedule in
view.  In particular, they do not replace the schedule by the coarse factor
``c_p`` before applying the extensiveness estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


def _round_down(value: float) -> float:
    """Round a positive floating-point result toward ``-inf``."""
    if value == 0.0 or math.isinf(value):
        return value
    return float(np.nextafter(value, -np.inf))


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


def _exp_log_up(log_value: float) -> float:
    """Exponentiate a log majorant without overflowing the recurrence."""
    if log_value == _NEGATIVE_INFINITY:
        return 0.0
    if log_value >= math.log(np.finfo(float).max):
        return math.inf
    return _round_up(math.exp(log_value))


def _log_polynomial_value(log_coefficients: tuple[float, ...], log_x: float) -> float:
    """Evaluate a nonnegative polynomial in the log domain."""
    total = _NEGATIVE_INFINITY
    for order, log_coefficient in enumerate(log_coefficients):
        if log_coefficient == _NEGATIVE_INFINITY:
            continue
        total = _logaddexp_up(total, log_coefficient + order * log_x)
    return total


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


@dataclass(frozen=True)
class MizutaLemma10Majorant:
    """Order-resolved positive majorants following Mizuta Lemma 10.

    ``log_g[s]`` bounds the extensiveness of the truncated-log coefficient
    ``Phi_s``. ``log_c_by_adjoint[ell][q]`` bounds order-``q`` terms with
    exactly ``ell`` nested adjoints. ``log_d`` is their exponential-series
    sum and majorizes the interaction-picture generator. Finally, ``log_b``
    majorizes the outer Dyson exponential.
    """

    log_g: tuple[float, ...]
    log_c_by_adjoint: tuple[tuple[float, ...], ...]
    log_d: tuple[float, ...]
    log_b: tuple[float, ...]
    truncation_order: int

    @property
    def maximum_order(self) -> int:
        return len(self.log_b) - 1


@lru_cache(maxsize=512)
def lemma10_log_g_coefficients(
    locality_k: int,
    weighted_extensiveness: float,
    truncation_order: int,
    maximum_order: int,
) -> tuple[float, ...]:
    """Bound ``g(Phi_s)`` with exact symmetric-Strang parity.

    Lemma 8's locality estimate gives ``K_s=s*k``.  Before the final coarse
    simplification in Lemma 10, its extensiveness estimate is retained as

    ``G_s = (s-1)!/s * (2*k*g_alpha)^(s-1) * g_alpha``.

    Strang symmetry makes every even ``Phi_s`` exactly zero.  Coefficients
    above ``p_0`` are absent because Lemma 10 applies to the truncated BCH
    polynomial.
    """
    if locality_k < 0:
        raise ValueError("locality_k must be nonnegative")
    if not math.isfinite(weighted_extensiveness) or weighted_extensiveness < 0.0:
        raise ValueError("weighted_extensiveness must be finite and nonnegative")
    if truncation_order < 1:
        raise ValueError("truncation_order must be positive")
    if maximum_order < 0:
        raise ValueError("maximum_order must be nonnegative")

    coefficients = [_NEGATIVE_INFINITY] * (maximum_order + 1)
    if locality_k == 0 or weighted_extensiveness == 0.0:
        return tuple(coefficients)
    log_scale = math.log(2.0 * locality_k * weighted_extensiveness)
    log_g_alpha = math.log(weighted_extensiveness)
    for order in range(1, min(truncation_order, maximum_order) + 1):
        if order % 2 == 0:
            continue
        if order == 1:
            coefficients[order] = log_g_alpha
        else:
            coefficients[order] = _round_up(
                math.lgamma(order)
                - math.log(order)
                + (order - 1) * log_scale
                + log_g_alpha
            )
    return tuple(coefficients)


@lru_cache(maxsize=512)
def lemma10_majorant(
    locality_k: int,
    weighted_extensiveness: float,
    truncation_order: int,
    maximum_order: int,
) -> MizutaLemma10Majorant:
    """Generate the Lemma-10 ``C``, ``D``, and ``B`` recurrences.

    If an accumulated order-``q-s`` operator is ``(q-s)k``-local, commuting
    it with a ``G_s``-extensive ``Phi_s`` costs at most
    ``2*k*(q-s)*G_s``.  Thus applying one more adjoint gives

    ``C_q^(ell+1) = sum_s 2*k*(q-s)*G_s*C_(q-s)^ell``.

    Summing ``C_q^ell/ell!`` produces ``D_q``, the interaction-picture
    generator majorant.  The subsystem difference ``Psi_s^i`` obeys the same
    local-insertion estimate ``G_s``: terms outside the subsystem cancel and
    every surviving schedule occurrence is already included in ``g_alpha``.
    The outer Dyson series is therefore majorized by ``B(z)=exp(D(z))``, whose
    differentiated generating function yields

    ``q B_q = sum_s s D_s B_(q-s)``.
    """
    if maximum_order < 0:
        raise ValueError("maximum_order must be nonnegative")
    log_g = lemma10_log_g_coefficients(
        locality_k,
        weighted_extensiveness,
        truncation_order,
        maximum_order,
    )
    log_c_rows: list[tuple[float, ...]] = [log_g]
    log_d = list(log_g)
    previous = log_g
    log_two_k = (
        math.log(2.0 * locality_k) if locality_k > 0 else _NEGATIVE_INFINITY
    )
    log_factorial = 0.0

    for adjoints in range(1, maximum_order + 1):
        current = [_NEGATIVE_INFINITY] * (maximum_order + 1)
        any_nonzero = False
        for order in range(1, maximum_order + 1):
            total = _NEGATIVE_INFINITY
            for phi_order in range(1, min(truncation_order, order - 1) + 1):
                remainder_order = order - phi_order
                if (
                    log_g[phi_order] == _NEGATIVE_INFINITY
                    or previous[remainder_order] == _NEGATIVE_INFINITY
                ):
                    continue
                term = (
                    log_two_k
                    + math.log(remainder_order)
                    + log_g[phi_order]
                    + previous[remainder_order]
                )
                total = _logaddexp_up(total, term)
            current[order] = total
            any_nonzero |= total != _NEGATIVE_INFINITY
        if not any_nonzero:
            break
        log_factorial = _round_up(log_factorial + math.log(adjoints))
        for order, value in enumerate(current):
            if value != _NEGATIVE_INFINITY:
                log_d[order] = _logaddexp_up(log_d[order], value - log_factorial)
        row = tuple(current)
        log_c_rows.append(row)
        previous = row

    log_b = [_NEGATIVE_INFINITY] * (maximum_order + 1)
    log_b[0] = 0.0
    for order in range(1, maximum_order + 1):
        total = _NEGATIVE_INFINITY
        for generator_order in range(1, order + 1):
            if log_d[generator_order] == _NEGATIVE_INFINITY:
                continue
            term = (
                math.log(generator_order)
                + log_d[generator_order]
                + log_b[order - generator_order]
            )
            total = _logaddexp_up(total, term)
        if total != _NEGATIVE_INFINITY:
            log_b[order] = _round_up(total - math.log(order))

    return MizutaLemma10Majorant(
        log_g=log_g,
        log_c_by_adjoint=tuple(log_c_rows),
        log_d=tuple(log_d),
        log_b=tuple(log_b),
        truncation_order=truncation_order,
    )


@dataclass(frozen=True)
class MizutaTailCertificate:
    """A scalar-flow certificate for the production Cauchy tail."""

    rho: float
    flow_radius: float
    integral_lower_bound: float
    log_b_at_rho_upper: float
    integration_intervals: int


def _flow_integral_certificate(
    log_g: tuple[float, ...],
    locality_k: int,
    rho: float,
    *,
    logarithmic_step: float = 1.0 / 64.0,
    maximum_intervals: int = 2048,
) -> MizutaTailCertificate | None:
    """Lower-bound ``integral_rho^R dx/(2*k*x*G(x))`` by right rectangles.

    In the logarithmic coordinate ``u=log(x)``, the integrand is
    ``1/(2*k*G(exp(u)))`` and decreases because ``G`` has nonnegative
    coefficients.  Right-endpoint rectangles are therefore rigorous lower
    bounds.  Every elementary operation is rounded in the conservative
    direction.
    """
    if locality_k <= 0 or rho <= 0.0:
        return None
    step_down = _round_down(logarithmic_step)
    log_two_k_up = _round_up(math.log(2.0 * locality_k))
    log_x = _round_up(math.log(rho))
    integral = 0.0

    for interval in range(1, maximum_intervals + 1):
        log_x = _round_up(log_x + logarithmic_step)
        log_g_value = _log_polynomial_value(log_g, log_x)
        if log_g_value == _NEGATIVE_INFINITY:
            return None
        log_contribution = math.log(step_down) - log_two_k_up - log_g_value
        contribution = 0.0 if log_contribution < -745.0 else math.exp(log_contribution)
        integral = _round_down(integral + _round_down(contribution))
        if integral >= 1.0:
            radius = _round_up(math.exp(log_x)) if log_x < 709.0 else math.inf
            log_g_radius = _log_polynomial_value(log_g, log_x)
            g_radius = _exp_log_up(log_g_radius)
            return MizutaTailCertificate(
                rho=_round_up(rho),
                flow_radius=radius,
                integral_lower_bound=integral,
                log_b_at_rho_upper=g_radius,
                integration_intervals=interval,
            )
        if contribution == 0.0:
            break
    return None


def certify_lemma10_tail(
    locality_k: int,
    weighted_extensiveness: float,
    truncation_order: int,
    x: float,
    *,
    maximum_order: int,
) -> MizutaTailCertificate | None:
    """Certify a common Cauchy radius for Lemmas 9 and 10.

    For ``G(z)=sum_{s<=p0} G_s z^s`` and
    ``L f=2*k*z*G(z)f'(z)``, the recurrence gives ``D=e^L G``.  The scalar
    characteristic satisfies ``y'=2*k*y*G(y)``.  Hence a pair ``rho<R`` with
    ``integral_rho^R dx/(2*k*x*G(x)) >= 1`` proves existence through unit flow
    time and ``D(rho)<=G(R)``.  Therefore ``B(rho)<=exp(G(R))``.

    Candidate radii are tried from the largest Lemma-9 analytic radius down
    toward ``x``.  Returning ``None`` is intentional: callers may then use the
    separately certified printed-Theorem-3 fallback.
    """
    if x < 0.0 or not math.isfinite(x):
        raise ValueError("x must be finite and nonnegative")
    if x == 0.0:
        return MizutaTailCertificate(
            rho=1.0,
            flow_radius=1.0,
            integral_lower_bound=math.inf,
            log_b_at_rho_upper=0.0,
            integration_intervals=0,
        )
    if locality_k <= 0 or weighted_extensiveness <= 0.0:
        return None

    lemma9_radius = 1.0 / (2.0 * locality_k * weighted_extensiveness)
    largest_rho = _round_down(0.95 * lemma9_radius)
    smallest_rho = _round_up(1.01 * x)
    if smallest_rho >= largest_rho:
        return None

    log_g = lemma10_log_g_coefficients(
        locality_k,
        weighted_extensiveness,
        truncation_order,
        maximum_order,
    )
    log_smallest = math.log(smallest_rho)
    log_largest = math.log(largest_rho)
    best: MizutaTailCertificate | None = None
    # Flow existence is monotone in the starting radius: if rho fails, every
    # larger radius fails as well.  Start next to x, retain the largest
    # successful grid point, and stop at the first failure.  This avoids
    # paying a full divergent-flow integration for every oversized radius.
    for index in range(24):
        fraction = index / 23.0
        log_rho = log_smallest + fraction * (log_largest - log_smallest)
        rho = _round_up(math.exp(log_rho))
        certificate = _flow_integral_certificate(log_g, locality_k, rho)
        if certificate is None:
            break
        best = certificate
    if best is not None:
        return best

    # A coarse right-endpoint grid can lose enough area to miss a radius very
    # close to the true flow boundary.  Retry only the smallest radius on the
    # original fine grid before declaring that the refined tail is unavailable.
    return _flow_integral_certificate(
        log_g,
        locality_k,
        smallest_rho,
        logarithmic_step=1.0 / 256.0,
        maximum_intervals=8192,
    )


@dataclass(frozen=True)
class MizutaRefinedRemainder:
    """Certified Lemma-9/Lemma-10 remainder at one time argument."""

    lemma9: float
    lemma10: float
    total: float
    explicit_order: int
    certificate: MizutaTailCertificate | None
    used_legacy_fallback: bool = False


def _positive_series_remainder(
    log_coefficients: tuple[float, ...],
    x: float,
    first_order: int,
    maximum_order: int,
    rho: float,
    log_function_at_rho_upper: float,
) -> float:
    """Add an explicit positive series prefix to its Cauchy tail."""
    if x == 0.0:
        return 0.0
    log_x = math.log(x)
    explicit_log = _NEGATIVE_INFINITY
    for order in range(first_order, maximum_order + 1):
        coefficient = log_coefficients[order]
        if coefficient != _NEGATIVE_INFINITY:
            explicit_log = _logaddexp_up(explicit_log, coefficient + order * log_x)
    explicit = _exp_log_up(explicit_log)

    ratio = _round_up(x / rho)
    if ratio >= 1.0:
        return math.inf
    log_tail = (
        log_function_at_rho_upper
        + (maximum_order + 1) * math.log(ratio)
        - math.log1p(-ratio)
    )
    tail = _exp_log_up(_round_up(log_tail))
    return _positive_sum_up([explicit, tail])


@lru_cache(maxsize=2048)
def refined_mizuta_remainder(
    num_qubits: int,
    locality_k: int,
    weighted_extensiveness: float,
    truncation_order: int,
    x: float,
) -> MizutaRefinedRemainder | None:
    """Certify ``N sum_{q>p0} (A_q+B_q)|x|^q``.

    Coefficients are generated through ``Q=2*p0+32``.  Positivity and Cauchy's
    estimate give, for either majorant ``F`` and ``x<rho``,

    ``sum_{q>Q} F_q x^q <= F(rho) (x/rho)^(Q+1)/(1-x/rho)``.

    Lemma 9 supplies its exact generating function at ``rho``; Lemma 10 uses
    the scalar-flow certificate above.  ``None`` means that this refined proof
    did not establish a tail at the requested point.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")
    if truncation_order < 1:
        raise ValueError("truncation_order must be positive")
    if x < 0.0 or not math.isfinite(x):
        raise ValueError("x must be finite and nonnegative")
    if x == 0.0 or locality_k == 0 or weighted_extensiveness == 0.0:
        return MizutaRefinedRemainder(0.0, 0.0, 0.0, 2 * truncation_order + 32, None)

    maximum_order = 2 * truncation_order + 32
    certificate = certify_lemma10_tail(
        locality_k,
        weighted_extensiveness,
        truncation_order,
        x,
        maximum_order=maximum_order,
    )
    if certificate is None or not x < certificate.rho:
        return None

    log_a = lemma9_log_coefficients(
        locality_k,
        weighted_extensiveness,
        maximum_order,
    )
    lemma10 = lemma10_majorant(
        locality_k,
        weighted_extensiveness,
        truncation_order,
        maximum_order,
    )
    denominator = _round_down(
        1.0 - 2.0 * locality_k * weighted_extensiveness * certificate.rho
    )
    if denominator <= 0.0:
        return None
    log_a_at_rho = _round_up(weighted_extensiveness * certificate.rho / denominator)

    lemma9_value = _positive_series_remainder(
        log_a,
        x,
        truncation_order + 1,
        maximum_order,
        certificate.rho,
        log_a_at_rho,
    )
    lemma10_value = _positive_series_remainder(
        lemma10.log_b,
        x,
        truncation_order + 1,
        maximum_order,
        certificate.rho,
        certificate.log_b_at_rho_upper,
    )
    lemma9_value = _round_up(num_qubits * lemma9_value)
    lemma10_value = _round_up(num_qubits * lemma10_value)
    return MizutaRefinedRemainder(
        lemma9=lemma9_value,
        lemma10=lemma10_value,
        total=_positive_sum_up([lemma9_value, lemma10_value]),
        explicit_order=maximum_order,
        certificate=certificate,
    )
