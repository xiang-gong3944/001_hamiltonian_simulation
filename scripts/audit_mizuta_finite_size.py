#!/usr/bin/env python3
"""Reproduce finite-size constants in Mizuta's 2026 MPF analysis.

This is a read-only audit utility.  It compares the repository's implemented
Theorem-4 estimator with (i) a direct optimization of the printed local bound
and (ii) a separately labelled tightening that retains the geometric tails
appearing immediately before the paper's final simplifications.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from hamiltonian_resources import (
    estimate_mpf_error,
    pauli_nested_commutator_bounds,
    select_mpf_segments,
    transverse_field_ising,
)
from hamiltonian_resources.multiproduct import (
    multiproduct_coefficients,
    optimal_mpf_exponents,
)


BASE_ORDER = 2
BASE_REPETITIONS = 2
DEFAULT_MAX_P0 = 96


@dataclass(frozen=True)
class Scenario:
    sites: int
    branches: int
    time: float
    epsilon: float


@dataclass(frozen=True)
class Candidate:
    segments: int
    p0: int
    eta: float | None
    mu: float


DEFAULT_SCENARIOS = (
    Scenario(4, 2, 0.01, 1e-4),
    Scenario(4, 3, 0.01, 1e-3),
    Scenario(4, 3, 0.01, 1e-4),
    Scenario(4, 3, 0.01, 1e-6),
    Scenario(4, 4, 0.01, 1e-4),
    Scenario(4, 3, 4.0, 1e-4),
)


def _ceil_positive(value: float) -> int:
    return max(1, math.ceil(value))


def _logsumexp(values: Iterable[float]) -> float:
    values = tuple(values)
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def mu_polynomial_root(alpha_by_order: dict[int, float]) -> float:
    """Solve sum_q alpha_com,q / mu**q = 1 in the log domain."""
    log_terms = tuple(
        (order, math.log(value))
        for order, value in sorted(alpha_by_order.items())
        if order > BASE_ORDER and value > 0
    )
    if not log_terms:
        return 0.0
    if any(math.isinf(log_value) for _, log_value in log_terms):
        return math.inf

    lower = max(log_value / order for order, log_value in log_terms)
    upper = lower + math.log(len(log_terms)) / min(order for order, _ in log_terms)

    def residual(log_mu: float) -> float:
        return _logsumexp(log_value - order * log_mu for order, log_value in log_terms)

    for _ in range(100):
        midpoint = (lower + upper) / 2
        if residual(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint
    return math.exp(upper)


def _alpha_data(hamiltonian: Any, p0: int, workers: int) -> tuple[dict[int, float], int, float]:
    bounds = pauli_nested_commutator_bounds(hamiltonian, p0, workers=workers)
    values = {order: bounds.at(order) for order in range(BASE_ORDER + 1, p0 + 1)}
    return values, bounds.locality_k, bounds.extensiveness_g


def _global_geometric_error(local_error: float, segments: int) -> float:
    exponent = segments * math.log1p(local_error)
    return math.expm1(exponent) if exponent < 709 else math.inf


def _smallest_integer_with_u_shaped_bound(
    lower: int,
    formal_order: int,
    commutator_constant: float,
    truncation_slope: float,
    budget: float,
) -> int | None:
    """Minimize C/r**m + D*r <= budget over integer r >= lower."""

    def value(segments: int) -> float:
        return commutator_constant / segments**formal_order + truncation_slope * segments

    if value(lower) <= budget:
        return lower
    if commutator_constant == 0 or truncation_slope == 0:
        return None
    stationary = (formal_order * commutator_constant / truncation_slope) ** (1 / (formal_order + 1))
    upper = max(lower, math.ceil(stationary))
    if value(upper) > budget:
        return None
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if value(midpoint) <= budget:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def theorem_minimum_segments(
    hamiltonian: Any,
    time: float,
    epsilon: float,
    formal_order: int,
    coefficient_l1: float,
    exponent_l1: int,
    workers: int,
    max_p0: int = DEFAULT_MAX_P0,
) -> Candidate:
    """Minimize the printed Theorem-4 bound under the paper's Eq. (75)."""
    best: Candidate | None = None
    sites = hamiltonian.num_qubits
    minimum_p0 = max(BASE_ORDER + 1, math.floor(math.log(3 * sites)) + 1)
    for p0 in range(minimum_p0, max_p0 + 1):
        eta = 3 * sites * math.exp(-p0)
        if not 0 < eta < 1:
            continue
        alpha, locality_k, extensiveness_g = _alpha_data(hamiltonian, p0, workers)
        mu = mu_polynomial_root(alpha)
        time_one = _ceil_positive(
            8 * math.e**3 * BASE_REPETITIONS * p0 * locality_k * extensiveness_g * abs(time)
        )
        time_two = _ceil_positive(2 * BASE_REPETITIONS * mu * abs(time))
        lower = max(time_one, time_two)
        if best is not None and lower >= best.segments:
            break
        commutator_constant = (
            2
            * math.sqrt(math.e)
            * coefficient_l1
            * (BASE_REPETITIONS * mu * abs(time)) ** (formal_order + 1)
        )
        truncation_slope = coefficient_l1 * exponent_l1 * eta
        segments = _smallest_integer_with_u_shaped_bound(
            lower,
            formal_order,
            commutator_constant,
            truncation_slope,
            epsilon / 2,
        )
        if segments is not None and (best is None or segments < best.segments):
            best = Candidate(segments, p0, eta, mu)
    if best is None:
        raise RuntimeError(f"no Theorem-4 candidate found through p0={max_p0}")
    return best


def _tight_local_error(
    segments: int,
    p0: int,
    mu: float,
    sites: int,
    time: float,
    coefficients: np.ndarray,
    exponents: tuple[int, ...],
    locality_k: int,
    extensiveness_g: float,
) -> float:
    """Proof-level tightening using Eqs. (59), (66), and (67)."""
    tau = abs(time) / segments
    x = BASE_REPETITIONS * mu * tau
    if x >= 1:
        return math.inf
    coefficient_l1 = float(np.sum(np.abs(coefficients)))
    commutator_error = (
        0.0 if x == 0 else coefficient_l1 * x ** (2 * len(exponents)) * math.expm1(x) / (1 - x)
    )

    truncation_error = 0.0
    for coefficient, exponent in zip(coefficients, exponents, strict=True):
        branch_tau = tau / exponent
        first_ratio = 4 * BASE_REPETITIONS * locality_k * extensiveness_g * branch_tau
        second_ratio = (
            8 * math.e**2 * BASE_REPETITIONS * p0 * locality_k * extensiveness_g * branch_tau
        )
        if first_ratio >= 1 or second_ratio >= 1:
            return math.inf
        bch_remainder = sites * (
            first_ratio ** (p0 + 1) / (1 - first_ratio)
            + math.e**2 * second_ratio ** (p0 + 1) / (2 * (1 - second_ratio))
        )
        truncation_error += abs(float(coefficient)) * exponent * bch_remainder
    return commutator_error + truncation_error


def tightened_minimum_segments(
    hamiltonian: Any,
    time: float,
    epsilon: float,
    branches: int,
    coefficients: np.ndarray,
    exponents: tuple[int, ...],
    workers: int,
    max_p0: int = DEFAULT_MAX_P0,
) -> Candidate:
    """Minimize the separately labelled same-ingredients tightening."""
    best: Candidate | None = None
    sites = hamiltonian.num_qubits
    for p0 in range(BASE_ORDER + 1, max_p0 + 1):
        alpha, locality_k, extensiveness_g = _alpha_data(hamiltonian, p0, workers)
        mu = mu_polynomial_root(alpha)
        denominator_lower = (
            math.floor(
                max(
                    BASE_REPETITIONS * mu * abs(time),
                    4 * BASE_REPETITIONS * locality_k * extensiveness_g * abs(time),
                    8
                    * math.e**2
                    * BASE_REPETITIONS
                    * p0
                    * locality_k
                    * extensiveness_g
                    * abs(time),
                )
            )
            + 1
        )
        lower = max(1, denominator_lower)
        if best is not None and lower >= best.segments:
            break

        def satisfies(segments: int) -> bool:
            local_error = _tight_local_error(
                segments,
                p0,
                mu,
                sites,
                time,
                coefficients,
                exponents,
                locality_k,
                extensiveness_g,
            )
            return _global_geometric_error(local_error, segments) <= epsilon

        upper = lower
        while not satisfies(upper):
            upper *= 2
            if best is not None and upper >= best.segments:
                upper = best.segments
                break
            if upper > 10**12:
                raise RuntimeError("tightened segment search exceeded 10^12")
        if not satisfies(upper):
            continue
        left = lower - 1
        while left + 1 < upper:
            midpoint = (left + upper) // 2
            if satisfies(midpoint):
                upper = midpoint
            else:
                left = midpoint
        if best is None or upper < best.segments:
            best = Candidate(upper, p0, None, mu)
    if best is None:
        raise RuntimeError(f"no tightened candidate found through p0={max_p0}")
    return best


def paper_coarse_segments(
    sites: int,
    time: float,
    epsilon: float,
    formal_order: int,
    coefficient_l1: float,
    exponent_l1: int,
    locality_k: int,
    extensiveness_g: float,
) -> dict[str, float | int | bool]:
    """Evaluate Mizuta Eqs. (77), (81), and (83) without asymptotics."""
    n_root = sites ** (1 / (BASE_ORDER + 1))
    r1 = (
        8
        * BASE_REPETITIONS
        * (BASE_ORDER + 1)
        * locality_k
        * n_root
        * extensiveness_g
        * abs(time)
        * (
            32
            * BASE_REPETITIONS
            * (BASE_ORDER + 1)
            * locality_k
            * coefficient_l1
            * n_root
            * extensiveness_g
            * abs(time)
            / epsilon
        )
        ** (1 / formal_order)
    )
    log_argument = (
        (8 * math.e**3 * BASE_REPETITIONS * locality_k * extensiveness_g * abs(time))
        * (12 * coefficient_l1 * exponent_l1 * sites)
        / epsilon
    )
    r2 = (
        40
        * math.e**4
        * BASE_REPETITIONS
        * locality_k
        * extensiveness_g
        * abs(time)
        * (formal_order + 1)
        * (
            160
            * math.e**3
            * coefficient_l1
            * BASE_REPETITIONS
            * locality_k
            * extensiveness_g
            * abs(time)
            / epsilon
        )
        ** (1 / formal_order)
        * math.log(log_argument) ** (1 + 1 / formal_order)
    )
    a_value = (
        epsilon
        / (
            4
            * coefficient_l1
            * (8 * math.e**3 * BASE_REPETITIONS * locality_k * extensiveness_g * abs(time))
            ** (formal_order + 1)
        )
        * (epsilon / (12 * coefficient_l1 * exponent_l1 * sites)) ** formal_order
    )
    return {
        "r1": r1,
        "r2": r2,
        "segments": math.ceil(max(r1, r2)),
        "a": a_value,
        "a_le_one_fifth": 0 < a_value <= 0.2,
    }


def audit_scenario(scenario: Scenario, workers: int) -> dict[str, Any]:
    hamiltonian = transverse_field_ising(
        scenario.sites,
        coupling=1.0,
        field=3.0,
        periodic=False,
    )
    coefficients = multiproduct_coefficients(scenario.branches)
    exponents = optimal_mpf_exponents(scenario.branches)
    coefficient_l1 = float(np.sum(np.abs(coefficients)))
    exponent_l1 = sum(exponents)
    weighted_exponent = float(np.dot(np.abs(coefficients), np.asarray(exponents)))
    formal_order = 2 * scenario.branches

    selected = select_mpf_segments(
        hamiltonian,
        scenario.time,
        scenario.epsilon,
        scenario.branches,
        method="mizuta2026-commutator-ideal-rigorous",
        workers=workers,
    )
    components = dict(selected.bound_components)
    p0 = int(components["truncation_order_p0"])
    mu = components["mu_upper"]
    locality_k = int(components["locality_k"])
    extensiveness_g = components["extensiveness_g"]
    r_time_one = _ceil_positive(scenario.time / components["first_time_limit"])
    r_time_two = _ceil_positive(scenario.time / components["second_time_limit"])
    r_error = _ceil_positive(
        (
            8
            * math.sqrt(math.e)
            * coefficient_l1
            * (BASE_REPETITIONS * mu * abs(scenario.time)) ** (formal_order + 1)
            / scenario.epsilon
        )
        ** (1 / formal_order)
    )
    component_counts = {
        "error": r_error,
        "time_1": r_time_one,
        "time_2": r_time_two,
    }
    active = max(component_counts, key=component_counts.__getitem__)

    theorem = theorem_minimum_segments(
        hamiltonian,
        scenario.time,
        scenario.epsilon,
        formal_order,
        coefficient_l1,
        exponent_l1,
        workers,
    )
    tightened = tightened_minimum_segments(
        hamiltonian,
        scenario.time,
        scenario.epsilon,
        scenario.branches,
        coefficients,
        exponents,
        workers,
    )
    coarse = paper_coarse_segments(
        scenario.sites,
        scenario.time,
        scenario.epsilon,
        formal_order,
        coefficient_l1,
        exponent_l1,
        locality_k,
        extensiveness_g,
    )
    alpha_by_order = {
        str(order): value for order, value in selected.commutator_bounds if order >= 3
    }
    term_count = len(hamiltonian.terms)
    controlled_t2_queries = 3 * exponent_l1 * selected.segments

    return {
        "sites": scenario.sites,
        "branches_J": scenario.branches,
        "formal_order_m": formal_order,
        "time": scenario.time,
        "epsilon": scenario.epsilon,
        "exponents": list(exponents),
        "coefficients": coefficients.tolist(),
        "coefficient_l1": coefficient_l1,
        "exponent_l1": exponent_l1,
        "weighted_exponent_sum": weighted_exponent,
        "locality_k": locality_k,
        "extensiveness_g": extensiveness_g,
        "p0": p0,
        "mu": mu,
        "repo_segments": selected.segments,
        "repo_error": selected.error,
        "repo_local_error": selected.local_error,
        "r_error": r_error,
        "r_time_1": r_time_one,
        "r_time_2": r_time_two,
        "r_trunc": None,
        "r_trunc_note": "absorbed into eta and p0; not an independent lower bound",
        "active_constraint": active,
        "paper_coarse": coarse,
        "theorem_minimum": {
            "segments": theorem.segments,
            "p0": theorem.p0,
            "eta": theorem.eta,
            "mu": theorem.mu,
        },
        "tightened": {
            "segments": tightened.segments,
            "p0": tightened.p0,
            "mu": tightened.mu,
        },
        "repository_structure": {
            "select_calls": 3 * selected.segments,
            "controlled_T2_queries": controlled_t2_queries,
            "controlled_Pauli_exponentials": (
                controlled_t2_queries * BASE_REPETITIONS * term_count
            ),
            "hamiltonian_term_count": term_count,
        },
        "auxiliary_error": components["auxiliary_error"],
        "local_commutator_error": components["local_commutator_error"],
        "local_truncated_bch_error": components["local_truncated_bch_error"],
        "first_time_limit": components["first_time_limit"],
        "second_time_limit": components["second_time_limit"],
        "alpha_com_through_p0": alpha_by_order,
    }


def _markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "| N | J | t | epsilon | p0 | mu | r_error | r_time,1 | r_time,2 | r_repo | active |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for item in results:
        lines.append(
            "| {sites} | {branches_J} | {time:g} | {epsilon:.1e} | {p0} | {mu:.6g} | "
            "{r_error} | {r_time_1} | {r_time_2} | {repo_segments} | {active_constraint} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "| N | J | t | epsilon | paper coarse | theorem minimum | tightened |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in results:
        lines.append(
            "| {sites} | {branches_J} | {time:g} | {epsilon:.1e} | {coarse} | {theorem} | {tight} |".format(
                **item,
                coarse=item["paper_coarse"]["segments"],
                theorem=item["theorem_minimum"]["segments"],
                tight=item["tightened"]["segments"],
            )
        )
    return "\n".join(lines)


def _check(results: list[dict[str, Any]], workers: int) -> None:
    for item in results:
        residual = math.fsum(
            value / item["mu"] ** int(order)
            for order, value in item["alpha_com_through_p0"].items()
            if value > 0
        )
        if item["mu"] > 0:
            assert math.isclose(residual, 1.0, rel_tol=2e-12, abs_tol=2e-12)
        assert item["repo_error"] <= item["epsilon"]
        assert item["repo_segments"] >= item["r_time_1"]
        assert item["repo_segments"] >= item["r_time_2"]
        if item["repo_segments"] > 1:
            previous = estimate_mpf_error(
                transverse_field_ising(item["sites"], coupling=1.0, field=3.0, periodic=False),
                item["time"],
                item["repo_segments"] - 1,
                item["branches_J"],
                method="mizuta2026-commutator-ideal-rigorous",
                target_error=item["epsilon"],
                workers=workers,
            )
            assert not (previous.rigorous and previous.error <= item["epsilon"])

    short = next(
        item
        for item in results
        if item["sites"] == 4
        and item["branches_J"] == 3
        and item["time"] == 0.01
        and item["epsilon"] == 1e-4
    )
    assert short["p0"] == 22
    assert short["repo_segments"] == 708
    assert math.isclose(short["mu"], 17.423049714315187, rel_tol=2e-12)
    assert short["active_constraint"] == "time_1"

    long = next(
        item
        for item in results
        if item["sites"] == 4
        and item["branches_J"] == 3
        and item["time"] == 4.0
        and item["epsilon"] == 1e-4
    )
    assert long["p0"] == 28
    assert long["repo_segments"] == 359933
    assert long["r_time_2"] == 280
    assert long["active_constraint"] == "time_1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--check", action="store_true", help="run internal consistency checks")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sites", type=int)
    parser.add_argument("--branches", type=int)
    parser.add_argument("--time", type=float)
    parser.add_argument("--epsilon", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    custom = any(
        value is not None for value in (args.sites, args.branches, args.time, args.epsilon)
    )
    scenarios = (
        (
            Scenario(
                args.sites or 4,
                args.branches or 3,
                args.time if args.time is not None else 0.01,
                args.epsilon if args.epsilon is not None else 1e-4,
            ),
        )
        if custom
        else DEFAULT_SCENARIOS
    )
    results = [audit_scenario(scenario, args.workers) for scenario in scenarios]
    if args.check:
        if custom:
            raise SystemExit("--check requires the built-in scenario set")
        _check(results, args.workers)
    if args.format == "json":
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(_markdown(results))


if __name__ == "__main__":
    main()
