import math

import pytest

from hamiltonian_resources import transverse_field_ising
from hamiltonian_resources.multiproduct import (
    estimate_mpf_error,
    legacy_w2_proxy_error,
    legacy_w2_proxy_segments,
    multiproduct_coefficients,
    select_mpf_segments,
)
from hamiltonian_resources.trotter import suzuki_commutator_bounds


@pytest.mark.parametrize("m", [2, 3, 5, 7])
def test_legacy_w2_proxy_exactly_reproduces_historical_rule(m):
    hamiltonian = transverse_field_ising(4, field=3.0)
    time = 1.7
    target_error = 9e-5
    _, w2 = suzuki_commutator_bounds(hamiltonian)
    alpha_effective = min(hamiltonian.alpha, w2 ** (1 / 3))
    formal_order = 2 * m
    historical_segments = max(
        1,
        math.ceil(
            (
                (alpha_effective * time) ** (formal_order + 1)
                / target_error
            )
            ** (1 / formal_order)
        ),
    )

    segments = legacy_w2_proxy_segments(hamiltonian, time, target_error, m)
    error, prefactor = legacy_w2_proxy_error(hamiltonian, time, m, segments)

    assert segments == historical_segments
    assert prefactor == pytest.approx(alpha_effective ** (formal_order + 1))
    assert error == pytest.approx(
        (alpha_effective * time) ** (formal_order + 1)
        / segments**formal_order
    )


def test_legacy_w2_proxy_is_explicitly_noncertifying_metadata():
    from hamiltonian_resources import BenchmarkConfig, MultiproductMethod, run_benchmark

    frame = run_benchmark(
        BenchmarkConfig(system_sizes=[2], methods=[MultiproductMethod(3)]),
        sweeps="system-size",
    )
    row = frame.iloc[0]

    assert row["bound_method"] == "legacy-w2-proxy"
    assert not bool(row["bound_rigorous"])


def test_low_bound_depends_on_schedule_coefficient_norm():
    hamiltonian = transverse_field_ising(2, field=0.7)
    new = estimate_mpf_error(
        hamiltonian, 0.4, 8, 3, schedule="new", method="low-rigorous"
    )
    legacy = estimate_mpf_error(
        hamiltonian, 0.4, 8, 3, schedule="legacy", method="low-rigorous"
    )

    assert new.coefficient_l1_norm == pytest.approx(
        sum(abs(value) for value in multiproduct_coefficients(3, schedule="new"))
    )
    assert legacy.coefficient_l1_norm == pytest.approx(
        sum(abs(value) for value in multiproduct_coefficients(3, schedule="legacy"))
    )
    assert new.coefficient_l1_norm != pytest.approx(legacy.coefficient_l1_norm)
    assert new.error / legacy.error == pytest.approx(
        new.coefficient_l1_norm / legacy.coefficient_l1_norm,
        rel=1e-6,
    )


def test_low_segment_selection_is_monotone_and_satisfies_bound():
    hamiltonian = transverse_field_ising(3, field=0.7)
    estimates = [
        select_mpf_segments(
            hamiltonian,
            0.8,
            target,
            3,
            schedule="new",
            method="low-rigorous",
        )
        for target in (1e-2, 1e-4, 1e-6)
    ]

    assert [estimate.segments for estimate in estimates] == sorted(
        estimate.segments for estimate in estimates
    )
    assert all(
        estimate.error <= target
        for estimate, target in zip(estimates, (1e-2, 1e-4, 1e-6))
    )
    assert all(estimate.rigorous for estimate in estimates)
    assert all(estimate.scope == "ideal-mpf" for estimate in estimates)
    assert all(not estimate.circuit_rigorous for estimate in estimates)
    assert all(
        estimate.circuit_scope == "amplified-shared-ancilla"
        for estimate in estimates
    )


def test_low_bound_matches_theorem_equations_14_and_15():
    hamiltonian = transverse_field_ising(2, field=0.7)
    time = 0.3
    segments = 7
    m = 2
    estimate = estimate_mpf_error(
        hamiltonian, time, segments, m, method="low-rigorous"
    )
    scaled_time = hamiltonian.alpha * time / segments
    step_error = (
        2
        * estimate.coefficient_l1_norm
        * scaled_time ** (2 * m + 1)
        * math.exp(scaled_time)
        / math.factorial(2 * m + 1)
    )
    expected = step_error * segments * (1 + step_error) ** (segments - 1)

    assert estimate.error == pytest.approx(expected)
