import math
import json
from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from hamiltonian_resources import (
    PauliHamiltonian,
    build_trotter_circuit,
    pauli_nested_commutator_bounds,
    transverse_field_ising,
)
from hamiltonian_resources.multiproduct import (
    estimate_mpf_error,
    legacy_w2_proxy_error,
    legacy_w2_proxy_segments,
    multiproduct_coefficients,
    select_mpf_segments,
)
from hamiltonian_resources.trotter import suzuki_commutator_bounds


def _ideal_mpf_operator(hamiltonian, time, segments, m, schedule="new"):
    step_time = time / segments
    step = sum(
        coefficient
        * Operator(
            build_trotter_circuit(
                hamiltonian,
                step_time,
                reps=exponent,
                order=2,
                partition="individual",
            )
        ).data
        for coefficient, exponent in zip(
            multiproduct_coefficients(m, schedule=schedule),
            estimate_mpf_error(
                hamiltonian,
                time,
                segments,
                m,
                schedule=schedule,
            ).exponents,
            strict=True,
        )
    )
    return np.linalg.matrix_power(step, segments)


def _ideal_mpf_operator_error(hamiltonian, time, segments, m, schedule="new"):
    approximation = _ideal_mpf_operator(
        hamiltonian, time, segments, m, schedule=schedule
    )
    exact = expm(-1j * time * hamiltonian.matrix())
    return float(np.linalg.norm(approximation - exact, ord=2))


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


def test_mpf_metadata_distinguishes_ideal_and_circuit_certification():
    from hamiltonian_resources import BenchmarkConfig, MultiproductMethod, run_benchmark

    frame = run_benchmark(
        BenchmarkConfig(system_sizes=[2], methods=[MultiproductMethod(3)]),
        sweeps="system-size",
    )
    row = frame.iloc[0]

    assert row["bound_method"] == "low2019-l1-ideal-rigorous"
    assert bool(row["bound_rigorous"])
    assert row["bound_scope"] == "ideal-mpf"
    assert bool(row["bound_target_satisfied"])
    assert row["circuit_bound_scope"] == "amplified-shared-ancilla"
    assert not bool(row["circuit_bound_rigorous"])
    assert not bool(row["circuit_target_satisfied"])
    coefficients = multiproduct_coefficients(3)
    assert row["mpf_physical_branch_count"] == 3
    assert row["mpf_negative_coefficient_count"] == int(sum(coefficients < 0))
    assert row["mpf_padding_branch_count"] == 2
    assert row["mpf_sign_branch_count"] == int(sum(coefficients < 0)) + 1
    assert row["mpf_active_branch_count"] == 5
    assert row["mpf_unused_branch_state_count"] == 3
    assert row["mpf_prepare_calls_per_segment"] == 6
    assert row["mpf_select_calls_per_segment"] == 3
    assert row["mpf_good_reflections_per_segment"] == 2
    assert row["mpf_base_lcu_uses_per_segment"] == 3

    legacy = run_benchmark(
        BenchmarkConfig(
            system_sizes=[2],
            methods=[MultiproductMethod(3, error_method="legacy-w2-proxy")],
        ),
        sweeps="system-size",
    ).iloc[0]
    assert legacy["bound_method"] == "legacy-w2-proxy"
    assert not bool(legacy["bound_rigorous"])
    assert not bool(legacy["bound_target_satisfied"])


def test_mizuta_theorem_metadata_propagates_to_benchmark_rows():
    from hamiltonian_resources import (
        BenchmarkConfig,
        MultiproductMethod,
        TimeScaling,
        run_benchmark,
    )

    row = run_benchmark(
        BenchmarkConfig(
            system_sizes=[2],
            time=TimeScaling("fixed", 0.01),
            methods=[
                MultiproductMethod(
                    2,
                    error_method="mizuta2026-commutator-ideal-rigorous",
                )
            ],
        ),
        sweeps="system-size",
    ).iloc[0]

    assert row["status"] == "ok"
    assert row["bound_method"] == "mizuta2026-commutator-ideal-rigorous"
    assert "Mizuta" in row["bound_reference"]
    assert row["bound_theorem_or_equations"].startswith("Theorem 4")
    assert row["hamiltonian_decomposition"] == "ordered individual Pauli terms"
    assert bool(row["bound_rigorous"])
    assert bool(row["locality_compatible"])
    assert row["max_nested_commutator_order"] >= 3
    assert json.loads(row["bound_components_json"])["mu_upper"] > 0
    assert json.loads(row["commutator_bounds_json"])
    assert not bool(row["circuit_bound_rigorous"])


def test_low_bound_depends_on_schedule_coefficient_norm():
    hamiltonian = transverse_field_ising(2, field=0.7)
    new = estimate_mpf_error(
        hamiltonian,
        0.4,
        8,
        3,
        schedule="new",
        method="low2019-l1-ideal-rigorous",
    )
    legacy = estimate_mpf_error(
        hamiltonian,
        0.4,
        8,
        3,
        schedule="legacy",
        method="low2019-l1-ideal-rigorous",
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
            method="low2019-l1-ideal-rigorous",
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
    for estimate, target in zip(estimates, (1e-2, 1e-4, 1e-6), strict=True):
        if estimate.segments > 1:
            previous = estimate_mpf_error(
                hamiltonian,
                0.8,
                estimate.segments - 1,
                3,
                schedule="new",
                method="low2019-l1-ideal-rigorous",
            )
            assert previous.error > target


def test_low_bound_matches_theorem_equations_14_and_15():
    hamiltonian = transverse_field_ising(2, field=0.7)
    time = 0.3
    segments = 7
    m = 2
    estimate = estimate_mpf_error(
        hamiltonian,
        time,
        segments,
        m,
        method="low2019-l1-ideal-rigorous",
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


def test_mpf_estimators_reject_invalid_policy_and_nonfinite_time():
    hamiltonian = transverse_field_ising(2)
    with pytest.raises(ValueError, match="method"):
        estimate_mpf_error(hamiltonian, 0.2, 2, 2, method="unknown")
    with pytest.raises(ValueError, match="finite"):
        select_mpf_segments(hamiltonian, np.inf, 1e-3, 2)


def test_historical_low_method_name_is_a_backward_compatible_alias():
    hamiltonian = transverse_field_ising(2)
    estimate = select_mpf_segments(
        hamiltonian,
        0.2,
        1e-4,
        2,
        method="low-rigorous",
    )

    assert estimate.method == "low2019-l1-ideal-rigorous"


def test_low_selector_reports_float_range_overflow_explicitly():
    with pytest.raises(OverflowError, match="segment count"):
        select_mpf_segments(
            transverse_field_ising(2),
            1e308,
            1e-6,
            2,
            method="low2019-l1-ideal-rigorous",
        )


def test_imbalanced_two_term_case_exposes_legacy_proxy_failure():
    hamiltonian = PauliHamiltonian.from_terms(
        1,
        [("Z", 30.0), ("X", 0.001)],
        name="imbalanced",
    )
    algorithm_budget = 9e-7
    legacy = select_mpf_segments(
        hamiltonian,
        1.0,
        algorithm_budget,
        3,
        method="legacy-w2-proxy",
    )
    exact_legacy_error = _ideal_mpf_operator_error(
        hamiltonian, 1.0, legacy.segments, 3
    )
    rigorous = select_mpf_segments(
        hamiltonian,
        1.0,
        algorithm_budget,
        3,
        method="low2019-l1-ideal-rigorous",
    )
    exact_rigorous_error = _ideal_mpf_operator_error(
        hamiltonian, 1.0, rigorous.segments, 3
    )

    assert legacy.segments == 5
    assert legacy.error <= algorithm_budget
    assert exact_legacy_error > 100 * algorithm_budget
    assert not legacy.rigorous
    assert rigorous.error <= algorithm_budget
    assert exact_rigorous_error <= rigorous.error


@pytest.mark.parametrize(("m", "schedule"), [(2, "new"), (3, "new"), (3, "legacy")])
def test_low_bound_upper_bounds_small_system_ideal_mpf_operator(m, schedule):
    hamiltonian = transverse_field_ising(2, field=0.7)
    estimate = select_mpf_segments(
        hamiltonian,
        0.6,
        1e-4,
        m,
        schedule=schedule,
        method="low2019-l1-ideal-rigorous",
    )
    exact_error = _ideal_mpf_operator_error(
        hamiltonian,
        0.6,
        estimate.segments,
        m,
        schedule=schedule,
    )

    assert exact_error <= estimate.error
    assert estimate.error <= 1e-4


def test_pauli_nested_commutators_vanish_for_commuting_decomposition():
    hamiltonian = PauliHamiltonian.from_terms(
        3,
        [("ZII", 0.3), ("IZI", -0.4), ("ZZI", 0.2), ("III", 1.7)],
    )
    bounds = pauli_nested_commutator_bounds(hamiltonian, 7)

    assert bounds.values == (0.0,) * 6
    assert bounds.max_exact_order == 7
    assert not bounds.used_locality_fallback


def test_exact_pauli_commutator_sums_agree_with_dense_tiny_validation():
    hamiltonian = PauliHamiltonian.from_terms(
        1,
        [("X", 0.3), ("Y", -0.2), ("Z", 0.7)],
    )
    term_matrices = [
        coefficient * PauliHamiltonian.from_terms(1, [(label, 1.0)]).matrix()
        for label, coefficient in hamiltonian.terms
    ]
    bounds = pauli_nested_commutator_bounds(hamiltonian, 5)

    for order in range(2, 6):
        dense_sum = 0.0
        for indices in product(range(len(term_matrices)), repeat=order):
            nested = term_matrices[indices[0]]
            for outer in indices[1:]:
                nested = term_matrices[outer] @ nested - nested @ term_matrices[outer]
            dense_sum += np.linalg.norm(nested, ord=2)
        assert bounds.at(order) == pytest.approx(dense_sum, rel=1e-12, abs=1e-14)


def test_pauli_commutator_cap_uses_explicit_rigorous_locality_fallback():
    hamiltonian = transverse_field_ising(4, field=0.7)
    bounds = pauli_nested_commutator_bounds(
        hamiltonian,
        6,
        transition_cap=1,
    )

    assert bounds.used_locality_fallback
    assert bounds.max_exact_order == 1
    assert "Mizuta 2026 Eq. (8)" in bounds.fallback_reason
    assert all(value > 0 for value in bounds.values)


def test_mizuta_bound_is_exact_zero_for_commuting_pauli_terms():
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("ZI", 0.7), ("IZ", -0.2), ("ZZ", 0.4)],
    )
    estimate = select_mpf_segments(
        hamiltonian,
        5.0,
        1e-8,
        3,
        method="mizuta2026-commutator-ideal-rigorous",
    )

    assert estimate.segments == 1
    assert estimate.error == 0.0
    assert estimate.rigorous
    assert estimate.locality_compatible
    assert estimate.max_nested_commutator_order >= 3


def test_mizuta_bound_upper_bounds_small_system_ideal_mpf_error():
    hamiltonian = transverse_field_ising(2, field=0.7)
    estimate = select_mpf_segments(
        hamiltonian,
        0.01,
        1e-3,
        2,
        method="mizuta2026-commutator-ideal-rigorous",
    )
    exact_error = _ideal_mpf_operator_error(
        hamiltonian,
        0.01,
        estimate.segments,
        2,
    )

    assert estimate.error <= 1e-3
    assert exact_error <= estimate.error
    assert estimate.rigorous
    assert estimate.theorem_or_equations.startswith("Theorem 4")
    assert estimate.max_exact_nested_commutator_order >= 3


def test_mizuta_analytical_path_never_constructs_dense_hamiltonian(monkeypatch):
    hamiltonian = transverse_field_ising(2, field=0.7)

    def fail_if_called(self):
        raise AssertionError("dense Hamiltonian matrix is forbidden")

    monkeypatch.setattr(PauliHamiltonian, "matrix", fail_if_called)
    estimate = estimate_mpf_error(
        hamiltonian,
        0.01,
        300,
        2,
        method="mizuta2026-commutator-ideal-rigorous",
        target_error=1e-3,
    )

    assert estimate.max_nested_commutator_order >= 3


def test_local_chain_nested_commutator_sum_is_extensive_at_fixed_order():
    small = pauli_nested_commutator_bounds(
        transverse_field_ising(6, field=0.7),
        3,
    ).at(3)
    large = pauli_nested_commutator_bounds(
        transverse_field_ising(12, field=0.7),
        3,
    ).at(3)

    assert 1.5 < large / small < 2.5
