import json
import math
from collections import defaultdict
from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from hamiltonian_resources import (
    MultiproductMethod,
    PauliHamiltonian,
    build_multiproduct_circuit,
    build_trotter_circuit,
    pauli_nested_commutator_bounds,
    plan_simulation,
    transverse_field_ising,
)
from hamiltonian_resources.multiproduct import (
    _mizuta_mu_upper_bound,
    _refined_mizuta_candidate,
    _w2_triangle_b2,
    _w2_triangle_ideal_mpf_bound,
    estimate_mpf_error,
    legacy_w2_proxy_error,
    legacy_w2_proxy_segments,
    multiproduct_coefficients,
    select_mpf_segments,
)
from hamiltonian_resources.trotter import (
    PauliNestedCommutatorBounds,
    estimate_suzuki_error,
    suzuki_commutator_bounds,
)


def _synthetic_commutator_bounds(
    values_by_order: dict[int, float],
) -> PauliNestedCommutatorBounds:
    max_order = max(values_by_order)
    values = tuple(values_by_order.get(order, 0.0) for order in range(2, max_order + 1))
    return PauliNestedCommutatorBounds(
        values=values,
        max_order=max_order,
        max_exact_order=max_order,
        state_counts=(0,) * len(values),
        used_locality_fallback=False,
        fallback_reason=None,
        locality_k=1,
        extensiveness_g=1.0,
    )


def _direct_mizuta_mu_candidates(
    values_by_order: dict[int, float],
    *,
    base_order: int,
    formal_order: int,
    max_repetitions: int,
) -> dict[tuple[int, int, int], float]:
    """Enumerate Eq. (47) through a finite repetition count.

    Keys are ``(q, n, q+n-1)``.  This helper deliberately implements the
    paper's index constraints independently of the polynomial-root routine.
    """
    assert all(order >= base_order + 1 for order in values_by_order)
    coefficients = {0: 1.0}
    candidates: dict[tuple[int, int, int], float] = {}
    for repetitions in range(1, max_repetitions + 1):
        following: defaultdict[int, float] = defaultdict(float)
        for total_degree, coefficient in coefficients.items():
            for order, commutator_bound in values_by_order.items():
                following[total_degree + order] += coefficient * commutator_bound
        coefficients = dict(following)
        for total_degree, coefficient in coefficients.items():
            q_value = total_degree - repetitions + 1
            if (
                coefficient > 0
                and q_value >= formal_order + 1
                and repetitions <= (q_value - 1) // base_order
            ):
                candidates[(q_value, repetitions, total_degree)] = coefficient ** (
                    1 / total_degree
                )
    return candidates


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


def _ideal_mpf_step(hamiltonian, step_time, m, schedule="new"):
    return sum(
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
                step_time,
                1,
                m,
                schedule=schedule,
            ).exponents,
            strict=True,
        )
    )


def _zero_ancilla_block(circuit, system_qubits):
    ancillas = circuit.num_qubits - system_qubits
    return Operator(circuit).data[:: 2**ancillas, :: 2**ancillas]


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


@pytest.mark.parametrize(
    ("m", "expected_b2"),
    [(2, 2 / 3), (3, 2 / 9), (5, 0.03546760370814992)],
)
def test_w2_triangle_b2_uses_absolute_branch_weights(m, expected_b2):
    coefficients = multiproduct_coefficients(m)
    exponents = estimate_mpf_error(
        transverse_field_ising(1),
        0.0,
        1,
        m,
    ).exponents

    signed_cancellation = math.fsum(
        float(coefficient) / exponent**2
        for coefficient, exponent in zip(coefficients, exponents, strict=True)
    )
    b2 = _w2_triangle_b2(coefficients, exponents)

    assert signed_cancellation == pytest.approx(0.0, abs=1e-15)
    assert b2 == pytest.approx(expected_b2)
    assert b2 > abs(signed_cancellation)


@pytest.mark.parametrize("m", [2, 3, 5])
@pytest.mark.parametrize("step_time", [0.03, 0.17])
def test_each_mpf_branch_obeys_w2_strang_telescoping(m, step_time):
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("XI", 0.31), ("YZ", -0.47), ("ZZ", 0.23), ("IX", -0.19)],
    )
    _, w2 = suzuki_commutator_bounds(hamiltonian)
    exact = expm(-1j * step_time * hamiltonian.matrix())
    exponents = estimate_mpf_error(hamiltonian, step_time, 1, m).exponents

    for exponent in exponents:
        branch = Operator(
            build_trotter_circuit(
                hamiltonian,
                step_time,
                reps=exponent,
                order=2,
                partition="individual",
            )
        ).data
        actual = float(np.linalg.norm(branch - exact, ord=2))
        bound = w2 * abs(step_time) ** 3 / exponent**2
        assert actual <= bound * (1 + 1e-12) + 1e-12


@pytest.mark.parametrize("m", [2, 3, 5])
@pytest.mark.parametrize(("time", "segments"), [(0.08, 1), (0.24, 3)])
def test_w2_triangle_bounds_exact_local_and_repeated_ideal_mpf(m, time, segments):
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("XI", 0.31), ("YZ", -0.47), ("ZZ", 0.23), ("IX", -0.19)],
    )
    estimate = estimate_mpf_error(
        hamiltonian,
        time,
        segments,
        m,
        method="childs2021-w2-triangle-ideal-rigorous",
    )
    step_time = time / segments
    step = _ideal_mpf_step(hamiltonian, step_time, m)
    exact_step = expm(-1j * step_time * hamiltonian.matrix())
    actual_local = float(np.linalg.norm(step - exact_step, ord=2))
    actual_repeated = _ideal_mpf_operator_error(hamiltonian, time, segments, m)

    assert estimate.rigorous
    assert estimate.scope == "ideal-mpf"
    assert estimate.local_error_rigorous
    assert actual_local <= estimate.local_error * (1 + 1e-12) + 1e-12
    assert actual_repeated <= estimate.error * (1 + 1e-12) + 1e-12


def test_w2_triangle_commuting_case_is_exact_and_selects_one_segment():
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("ZI", 0.7), ("IZ", -0.2), ("ZZ", 0.4)],
    )
    estimate = select_mpf_segments(
        hamiltonian,
        5.0,
        1e-12,
        3,
        method="childs2021-w2-triangle-ideal-rigorous",
    )

    assert estimate.segments == 1
    assert estimate.error == 0.0
    assert estimate.local_error == 0.0
    assert dict(estimate.bound_components)["w2"] == 0.0
    assert _ideal_mpf_operator_error(hamiltonian, 0.5, 1, 3) < 1e-12


def test_w2_triangle_segment_selection_is_minimal():
    hamiltonian = transverse_field_ising(2, field=3.0)
    target_error = 9e-4
    selected = select_mpf_segments(
        hamiltonian,
        2.0,
        target_error,
        3,
        method="childs2021-w2-triangle-ideal-rigorous",
    )
    previous = estimate_mpf_error(
        hamiltonian,
        2.0,
        selected.segments - 1,
        3,
        method="childs2021-w2-triangle-ideal-rigorous",
    )

    assert selected.segments == 161
    assert selected.error <= target_error
    assert previous.error > target_error


def test_w2_triangle_log_domain_handles_overflowing_direct_bound():
    repeated, _, local, _ = _w2_triangle_ideal_mpf_bound(
        1e300,
        1e100,
        1,
        (1.0,),
        (1,),
    )
    scaled, _, scaled_local, _ = _w2_triangle_ideal_mpf_bound(
        1e300,
        1e100,
        10**200,
        (1.0,),
        (1,),
    )

    assert math.isinf(local) and math.isinf(repeated)
    assert math.isfinite(scaled_local) and math.isfinite(scaled)


def test_w2_triangle_single_branch_reduces_to_second_order_trotter_bound():
    hamiltonian = transverse_field_ising(3, field=0.7)
    time = 0.4
    segments = 7
    _, w2 = suzuki_commutator_bounds(hamiltonian)
    repeated, prefactor, local, b2 = _w2_triangle_ideal_mpf_bound(
        w2,
        time,
        segments,
        (1.0,),
        (1,),
    )
    trotter = estimate_suzuki_error(
        hamiltonian,
        time,
        reps=segments,
        order=2,
        partition="individual",
    )

    assert b2 == 1.0
    assert prefactor == w2
    assert local == pytest.approx(w2 * abs(time / segments) ** 3)
    assert repeated == pytest.approx(w2 * abs(time) ** 3 / segments**2)
    assert repeated == pytest.approx(trotter.error)


@pytest.mark.parametrize(
    "hamiltonian",
    [
        transverse_field_ising(3, field=0.7),
        PauliHamiltonian.from_terms(
            2,
            [("XI", 0.31), ("YZ", -0.47), ("ZZ", 0.23), ("II", 0.9)],
        ),
    ],
)
def test_alpha_cubed_is_a_redundant_coarse_w2_prefactor(hamiltonian):
    _, w2 = suzuki_commutator_bounds(hamiltonian)

    assert w2 <= hamiltonian.alpha**3 / 2 * (1 + 1e-15)


def test_mpf_metadata_distinguishes_ideal_and_circuit_certification():
    from hamiltonian_resources import BenchmarkConfig, MultiproductMethod, run_benchmark

    frame = run_benchmark(
        BenchmarkConfig(
            system_sizes=[2],
            methods=[
                MultiproductMethod(
                    3,
                    error_method="mizuta2026-theorem3-legacy-ideal-rigorous",
                )
            ],
        ),
        sweeps="system-size",
    )
    row = frame.iloc[0]

    assert row["bound_method"] == "mizuta2026-theorem3-legacy-ideal-rigorous"
    assert bool(row["bound_rigorous"])
    assert row["bound_scope"] == "ideal-mpf"
    assert bool(row["bound_target_satisfied"])
    assert row["circuit_bound_scope"] == "repeated-shared-ancilla-good-block"
    assert bool(row["circuit_bound_rigorous"])
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


def test_w2_triangle_provenance_serializes_to_benchmark_rows():
    from hamiltonian_resources import BenchmarkConfig, MultiproductMethod, TimeScaling, run_benchmark

    row = run_benchmark(
        BenchmarkConfig(
            system_sizes=[2],
            time=TimeScaling("fixed", 0.2),
            methods=[
                MultiproductMethod(
                    3,
                    error_method="childs2021-w2-triangle-ideal-rigorous",
                )
            ],
        ),
        sweeps="system-size",
    ).iloc[0]
    components = json.loads(row["bound_components_json"])
    assumptions = json.loads(row["bound_assumptions_json"])

    assert row["method_id"] == "mpf-m3-childs2021-w2-triangle-ideal-rigorous"
    assert row["bound_method"] == "childs2021-w2-triangle-ideal-rigorous"
    assert bool(row["bound_rigorous"])
    assert row["bound_scope"] == "ideal-mpf"
    assert row["hamiltonian_decomposition"] == "ordered individual Pauli terms"
    assert "Childs" in row["bound_reference"]
    assert bool(row["locality_compatible"])
    assert row["max_nested_commutator_order"] == 3
    assert row["max_exact_nested_commutator_order"] == 3
    assert set(components) == {
        "w2",
        "b2",
        "local_step_size",
        "local_step_error",
        "repeated_ideal_mpf_error",
    }
    assert components["local_step_size"] == pytest.approx(
        row["evolution_time"] / row["segment_count"]
    )
    assert components["repeated_ideal_mpf_error"] == row["bound_value"]
    assert "no MPF cancellation condition is used" in assumptions


def test_representative_w2_triangle_comparison_retains_expected_tradeoffs():
    hamiltonian = transverse_field_ising(2, coupling=1.0, field=3.0, periodic=False)
    methods = (
        "legacy-w2-proxy",
        "low2019-l1-ideal-rigorous",
        "childs2021-w2-triangle-ideal-rigorous",
        "mizuta2026-theorem3-legacy-ideal-rigorous",
    )
    segments = {
        method: select_mpf_segments(
            hamiltonian,
            2.0,
            9e-4,
            3,
            method=method,
        ).segments
        for method in methods
    }

    assert segments == {
        "legacy-w2-proxy": 20,
        "low2019-l1-ideal-rigorous": 24,
        "childs2021-w2-triangle-ideal-rigorous": 161,
        "mizuta2026-theorem3-legacy-ideal-rigorous": 123_406,
    }


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
                    error_method="mizuta2026-theorem3-legacy-ideal-rigorous",
                )
            ],
        ),
        sweeps="system-size",
    ).iloc[0]

    assert row["status"] == "ok"
    assert row["bound_method"] == "mizuta2026-theorem3-legacy-ideal-rigorous"
    assert "Mizuta" in row["bound_reference"]
    assert row["bound_theorem_or_equations"] == (
        "Theorem 4, Eqs. (47)--(49), with Theorem 3, Eqs. (33)--(35)"
    )
    assert row["hamiltonian_decomposition"] == "ordered individual Pauli terms"
    assert bool(row["bound_rigorous"])
    assert bool(row["locality_compatible"])
    assert row["max_nested_commutator_order"] >= 3
    assert json.loads(row["bound_components_json"])["mu_upper"] > 0
    assert json.loads(row["commutator_bounds_json"])
    assert bool(row["circuit_bound_rigorous"])
    assert row["mpf_r_error"] >= 1
    assert row["mpf_r_time_1"] == row["segment_count"]
    assert row["mpf_r_time_2"] >= 1
    assert json.loads(row["mpf_active_constraints_json"]) == ["time_1"]
    assert row["mpf_mu_upper"] > 0
    assert row["mpf_truncation_order_p0"] >= 3
    assert row["mpf_auxiliary_error"] > 0
    assert 0 < row["mpf_auxiliary_allocation_fraction"] < 1
    assert row["mpf_local_commutator_error"] >= 0
    assert row["mpf_local_truncated_bch_error"] > 0
    assert row["mpf_bound_policy"] == "mizuta2026-theorem3-legacy-ideal-rigorous"
    assert json.loads(row["mpf_bound_candidates_json"]) == []


def test_mizuta_segment_diagnostics_recompute_each_candidate_predicate():
    hamiltonian = transverse_field_ising(
        4,
        coupling=1.0,
        field=3.0,
        periodic=False,
    )

    estimate = select_mpf_segments(
        hamiltonian,
        0.01,
        1e-4,
        3,
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
    )
    diagnostics = estimate.segment_diagnostics

    assert estimate.segments == 675
    assert diagnostics is not None
    assert (diagnostics.r_error, diagnostics.r_time_1, diagnostics.r_time_2) == (
        2,
        675,
        1,
    )
    assert diagnostics.active_constraints == ("time_1",)
    assert diagnostics.truncation_order_p0 == 21
    assert diagnostics.mu_upper == pytest.approx(17.412555296623527)
    assert diagnostics.auxiliary_allocation_fraction == pytest.approx(0.8121327656087732)
    assert diagnostics.allocation_strategy == "optimized-discrete-p0"


def test_mizuta_fixed_equal_allocation_remains_available_for_audit():
    hamiltonian = transverse_field_ising(4, coupling=1.0, field=3.0, periodic=False)

    estimate = select_mpf_segments(
        hamiltonian,
        0.01,
        1e-4,
        3,
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
        auxiliary_allocation_fraction=0.5,
    )
    diagnostics = estimate.segment_diagnostics

    assert estimate.segments == 708
    assert diagnostics is not None
    assert (diagnostics.r_error, diagnostics.r_time_1, diagnostics.r_time_2) == (3, 708, 1)
    assert diagnostics.truncation_order_p0 == 22
    assert diagnostics.mu_upper == pytest.approx(17.423049714315187)
    assert diagnostics.auxiliary_allocation_fraction == 0.5
    assert diagnostics.allocation_strategy == "fixed-local-budget-fraction"


@pytest.mark.parametrize(
    ("sites", "time", "epsilon", "branches", "fixed_segments", "optimized_segments", "rho"),
    [
        (4, 0.01, 1e-4, 2, 675, 643, 0.7952327646929407),
        (4, 0.01, 1e-4, 4, 740, 708, 0.5894035594342328),
        (4, 4.0, 1e-4, 3, 359_933, 359_933, 0.39489677289325875),
        (12, 12.0, 1e-3, 3, 1_079_799, 1_041_235, 0.9320125405347331),
        (50, 50.0, 1e-3, 3, 4_981_214, 4_820_529, 0.8951029956261423),
    ],
)
def test_mizuta_discrete_allocation_optimizer_representative_cases(
    sites, time, epsilon, branches, fixed_segments, optimized_segments, rho
):
    hamiltonian = transverse_field_ising(
        sites,
        coupling=1.0,
        field=3.0,
        periodic=False,
    )

    optimized = select_mpf_segments(
        hamiltonian,
        time,
        epsilon,
        branches,
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
    )
    fixed = select_mpf_segments(
        hamiltonian,
        time,
        epsilon,
        branches,
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
        auxiliary_allocation_fraction=0.5,
    )

    assert optimized.segments == optimized_segments
    assert fixed.segments == fixed_segments
    assert optimized.segments <= fixed.segments
    assert optimized.segment_diagnostics is not None
    assert optimized.segment_diagnostics.auxiliary_allocation_fraction == pytest.approx(rho)
    if optimized.segments > 1:
        previous = estimate_mpf_error(
            hamiltonian,
            time,
            optimized.segments - 1,
            branches,
            method="mizuta2026-theorem3-legacy-ideal-rigorous",
            target_error=epsilon,
        )
        assert not (previous.rigorous and previous.error <= epsilon)


def test_mizuta_selected_allocation_reproduces_production_formulas_and_dense_search():
    hamiltonian = transverse_field_ising(4, coupling=1.0, field=3.0, periodic=False)
    epsilon = 1e-4
    selected = select_mpf_segments(
        hamiltonian,
        0.01,
        epsilon,
        3,
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
    )
    diagnostics = selected.segment_diagnostics
    assert diagnostics is not None
    assert diagnostics.auxiliary_error is not None
    assert diagnostics.auxiliary_allocation_fraction is not None
    assert diagnostics.truncation_order_p0 is not None
    local_budget = math.expm1(math.log1p(epsilon) / selected.segments)

    assert math.ceil(
        math.log(3 * hamiltonian.num_qubits / diagnostics.auxiliary_error)
    ) == diagnostics.truncation_order_p0
    assert diagnostics.local_truncated_bch_error == pytest.approx(
        diagnostics.auxiliary_allocation_fraction * local_budget
    )

    dense_errors = []
    for fraction in np.linspace(0.001, 0.999, 200):
        estimate = estimate_mpf_error(
            hamiltonian,
            0.01,
            selected.segments,
            3,
            method="mizuta2026-theorem3-legacy-ideal-rigorous",
            target_error=epsilon,
            auxiliary_allocation_fraction=float(fraction),
        )
        if estimate.rigorous:
            dense_errors.append(estimate.error)
    assert dense_errors
    assert selected.error <= min(dense_errors) * (1 + 1e-12)


def test_refined_mizuta_selects_p0_directly_and_reports_branchwise_remainders():
    hamiltonian = transverse_field_ising(4, coupling=1.0, field=3.0, periodic=False)

    estimate = select_mpf_segments(
        hamiltonian,
        0.01,
        1e-4,
        3,
        method="mizuta2026-commutator-ideal-rigorous",
    )
    diagnostics = estimate.segment_diagnostics

    assert estimate.segments == 2
    assert estimate.error <= 1e-4
    assert diagnostics is not None
    assert diagnostics.truncation_order_p0 == 4
    assert diagnostics.auxiliary_error is None
    assert diagnostics.auxiliary_allocation_fraction is None
    assert diagnostics.allocation_strategy == "direct-p0-remainder-optimization"
    assert diagnostics.refined_lemma9_remainder > 0.0
    assert diagnostics.refined_lemma10_remainder > 0.0
    assert diagnostics.total_branchwise_bch_remainder == pytest.approx(
        diagnostics.refined_lemma9_remainder + diagnostics.refined_lemma10_remainder
    )
    assert diagnostics.local_step_error == pytest.approx(
        diagnostics.local_commutator_error + diagnostics.total_branchwise_bch_remainder
    )
    assert diagnostics.repeated_global_error == pytest.approx(estimate.error)
    assert diagnostics.legacy_first_condition_passed is False
    assert diagnostics.second_time_limit > estimate.local_step_size
    assert diagnostics.schedule_weighted_extensiveness == pytest.approx(5.0)
    assert np.allclose(diagnostics.schedule_weights, 1.0, atol=2e-15)
    assert diagnostics.refined_tail_fallback_status == "not-used"

    previous = estimate_mpf_error(
        hamiltonian,
        0.01,
        estimate.segments - 1,
        3,
        method="mizuta2026-commutator-ideal-rigorous",
        target_error=1e-4,
    )
    assert not (previous.rigorous and previous.error <= 1e-4)


def test_refined_mizuta_rejects_auxiliary_allocation_option():
    with pytest.raises(ValueError, match="not applicable to the refined Mizuta"):
        estimate_mpf_error(
            transverse_field_ising(2, field=0.7),
            0.01,
            2,
            2,
            method="mizuta2026-commutator-ideal-rigorous",
            target_error=1e-3,
            auxiliary_allocation_fraction=0.5,
        )


def test_refined_tail_failure_uses_legacy_fallback_only_under_first_condition(monkeypatch):
    monkeypatch.setattr(
        "hamiltonian_resources.multiproduct.refined_mizuta_remainder",
        lambda *args, **kwargs: None,
    )
    hamiltonian = transverse_field_ising(4, coupling=1.0, field=3.0)
    coefficients = multiproduct_coefficients(3)
    exponents = (1, 2, 4)

    fallback = _refined_mizuta_candidate(
        hamiltonian,
        0.01,
        3,
        1000,
        coefficients,
        exponents,
        1e-4,
        3,
        None,
    )
    rejected = _refined_mizuta_candidate(
        hamiltonian,
        0.01,
        3,
        1,
        coefficients,
        exponents,
        1e-4,
        3,
        None,
    )

    assert fallback.legacy_first_condition_passed
    assert fallback.used_legacy_tail_fallback
    assert fallback.tail_certified
    assert fallback.local_truncated_bch_error > 0.0
    assert not rejected.legacy_first_condition_passed
    assert not rejected.used_legacy_tail_fallback
    assert not rejected.tail_certified
    assert math.isinf(rejected.local_truncated_bch_error)


def test_refined_mizuta_bound_upper_bounds_dense_small_system_error():
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

    assert exact_error <= estimate.error <= 1e-3
    assert estimate.rigorous


def test_mpf_segment_diagnostics_preserve_tied_constraints_and_low_provenance():
    hamiltonian = transverse_field_ising(2, field=0.7)
    mizuta = select_mpf_segments(
        hamiltonian,
        0.0,
        1e-3,
        2,
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
    )
    low = select_mpf_segments(
        hamiltonian,
        0.2,
        1e-3,
        2,
        method="low2019-l1-ideal-rigorous",
    )

    assert mizuta.segment_diagnostics is not None
    assert mizuta.segment_diagnostics.active_constraints == (
        "error",
        "time_1",
        "time_2",
    )
    assert low.segment_diagnostics is not None
    assert low.segment_diagnostics.r_error == low.segments
    assert low.segment_diagnostics.r_time_1 is None
    assert low.segment_diagnostics.r_time_2 is None
    assert low.segment_diagnostics.active_constraints == ("error",)


def test_best_rigorous_ideal_policy_selects_low_and_retains_candidates():
    hamiltonian = transverse_field_ising(4, coupling=1.0, field=3.0, periodic=False)

    estimate = select_mpf_segments(
        hamiltonian,
        0.01,
        1e-4,
        3,
        method="best-rigorous-ideal",
    )

    assert estimate.requested_method == "best-rigorous-ideal"
    assert estimate.method == "low2019-l1-ideal-rigorous"
    assert estimate.segments == 1
    assert [(candidate.method, candidate.segments) for candidate in estimate.bound_candidates] == [
        ("low2019-l1-ideal-rigorous", 1),
        ("childs2021-w2-triangle-ideal-rigorous", 1),
        ("mizuta2026-commutator-ideal-rigorous", 2),
    ]
    assert all(candidate.rigorous for candidate in estimate.bound_candidates)


def test_best_rigorous_ideal_policy_can_select_w2_and_break_ties_with_low():
    identity_heavy = PauliHamiltonian.from_terms(
        1,
        [("I", 1000.0), ("X", 0.1), ("Z", 0.1)],
    )
    w2_triangle = select_mpf_segments(
        identity_heavy,
        0.01,
        1e-3,
        2,
        method="best-rigorous-ideal",
    )
    tie = select_mpf_segments(
        transverse_field_ising(2, field=0.7),
        0.0,
        1e-3,
        2,
        method="best-rigorous-ideal",
    )

    assert w2_triangle.method == "mizuta2026-commutator-ideal-rigorous"
    assert w2_triangle.segments == 1
    assert tie.method == "low2019-l1-ideal-rigorous"
    assert tie.segments == 1


def test_best_rigorous_ideal_policy_propagates_selected_bound_and_policy_to_benchmark():
    from hamiltonian_resources import BenchmarkConfig, TimeScaling, run_benchmark

    row = run_benchmark(
        BenchmarkConfig(
            system_sizes=[2],
            time=TimeScaling("fixed", 0.01),
            methods=[MultiproductMethod(2, error_method="best-rigorous-ideal")],
        ),
        sweeps="system-size",
    ).iloc[0]
    candidates = json.loads(row["mpf_bound_candidates_json"])

    assert row["method_id"] == "mpf-m2-best-rigorous-ideal"
    assert row["mpf_bound_policy"] == "best-rigorous-ideal"
    assert row["bound_method"] == "low2019-l1-ideal-rigorous"
    assert [candidate["method"] for candidate in candidates] == [
        "low2019-l1-ideal-rigorous",
        "childs2021-w2-triangle-ideal-rigorous",
        "mizuta2026-commutator-ideal-rigorous",
    ]
    assert all(candidate["rigorous"] for candidate in candidates)


def test_repository_branch_count_maps_to_mizuta_formal_order():
    method = MultiproductMethod(7)
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("ZI", 0.7), ("IZ", -0.2)],
    )
    estimate = estimate_mpf_error(
        hamiltonian,
        0.01,
        1,
        method.term_count,
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
        target_error=1e-3,
    )

    assert method.label.startswith("MPF J=7, formal order=14")
    assert estimate.m == 7
    assert estimate.formal_order == 14


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
    assert estimate.local_error == pytest.approx(step_error)
    assert estimate.local_error_rigorous


def test_repeated_shared_ancilla_claim_bounds_the_actual_projected_block():
    hamiltonian = transverse_field_ising(2, field=0.7)
    plan = plan_simulation(
        hamiltonian,
        MultiproductMethod(2, error_method="low2019-l1-ideal-rigorous"),
        0.5,
        1e-2,
    )
    circuit = build_multiproduct_circuit(
        hamiltonian,
        plan.time,
        m=plan.method.term_count,
        segments=plan.segments,
        schedule=plan.method.schedule,
    )
    projected = _zero_ancilla_block(circuit, hamiltonian.num_qubits)
    exact = expm(-1j * plan.time * hamiltonian.matrix())
    actual_error = np.linalg.norm(projected - exact, ord=2)
    repeated = plan.error_analysis.claim_for_scope(
        "repeated-shared-ancilla-good-block"
    )

    assert plan.segments == 2
    assert repeated is not None
    assert actual_error <= repeated.claim.value
    assert plan.error_analysis.ideal_algorithm_target_certified
    assert not plan.error_analysis.implemented_circuit_target_certified
    assert plan.error_analysis.implemented_circuit_target.outcome == "not_met"


def test_repeated_projected_block_is_not_silently_replaced_by_good_block_power():
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("XI", 0.7), ("ZZ", -0.9), ("IY", 0.4)],
    )
    total_time = 1.4
    segments = 2
    step = build_multiproduct_circuit(
        hamiltonian,
        total_time / segments,
        m=2,
    )
    repeated = build_multiproduct_circuit(
        hamiltonian,
        total_time,
        m=2,
        segments=segments,
    )
    one_step_good_block = _zero_ancilla_block(step, hamiltonian.num_qubits)
    repeated_good_block = _zero_ancilla_block(repeated, hamiltonian.num_qubits)

    reentry_term = repeated_good_block - one_step_good_block @ one_step_good_block
    assert np.linalg.norm(reentry_term, ord=2) > 1e-8


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


def test_pauli_commutator_recurrence_reuses_exact_prefixes():
    hamiltonian = transverse_field_ising(4, field=0.7)
    pauli_nested_commutator_bounds.cache_clear()

    through_five = pauli_nested_commutator_bounds(hamiltonian, 5)
    through_three = pauli_nested_commutator_bounds(hamiltonian, 3)
    through_seven = pauli_nested_commutator_bounds(hamiltonian, 7)

    assert through_three.values == through_five.values[:2]
    assert through_seven.values[:4] == through_five.values
    assert through_three.state_counts == through_five.state_counts[:2]
    assert through_seven.state_counts[:4] == through_five.state_counts


def test_duplicate_pauli_terms_preserve_nested_commutator_sums():
    duplicated = PauliHamiltonian(
        1,
        (("X", 0.2), ("X", -0.1), ("Y", 0.4), ("Z", -0.3)),
    )
    combined_weights = PauliHamiltonian(
        1,
        (("X", 0.3), ("Y", 0.4), ("Z", 0.3)),
    )
    pauli_nested_commutator_bounds.cache_clear()

    duplicated_bounds = pauli_nested_commutator_bounds(duplicated, 6)
    combined_bounds = pauli_nested_commutator_bounds(combined_weights, 6)

    assert duplicated_bounds.values == pytest.approx(
        combined_bounds.values,
        rel=1e-12,
        abs=1e-14,
    )
    assert duplicated_bounds.state_counts == combined_bounds.state_counts


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


@pytest.mark.parametrize("formal_order", [2, 4, 14])
def test_mizuta_single_support_mu_is_sharp_after_formal_order_cutoff(formal_order):
    values_by_order = {3: 8.0}
    bounds = _synthetic_commutator_bounds(values_by_order)
    mu_upper = _mizuta_mu_upper_bound(bounds, base_order=2)
    minimum_repetitions = math.ceil(formal_order / 2)
    candidates = _direct_mizuta_mu_candidates(
        values_by_order,
        base_order=2,
        formal_order=formal_order,
        max_repetitions=minimum_repetitions,
    )

    assert mu_upper == pytest.approx(2.0)
    assert min(repetitions for _, repetitions, _ in candidates) == minimum_repetitions
    assert max(candidates.values()) == pytest.approx(mu_upper)


def test_mizuta_finite_enumeration_is_bounded_by_and_approaches_polynomial_root():
    values_by_order = {3: 0.4, 4: 2.0, 5: 0.7}
    bounds = _synthetic_commutator_bounds(values_by_order)
    mu_upper = _mizuta_mu_upper_bound(bounds, base_order=2)
    short_candidates = _direct_mizuta_mu_candidates(
        values_by_order,
        base_order=2,
        formal_order=14,
        max_repetitions=10,
    )
    long_candidates = _direct_mizuta_mu_candidates(
        values_by_order,
        base_order=2,
        formal_order=14,
        max_repetitions=80,
    )

    assert short_candidates
    assert all(candidate <= mu_upper for candidate in long_candidates.values())
    assert max(long_candidates.values()) > max(short_candidates.values())
    assert mu_upper - max(long_candidates.values()) < 0.02


def test_mizuta_mu_upper_bound_is_monotone_in_commutator_upper_bounds():
    exact_data = _synthetic_commutator_bounds({3: 0.4, 4: 2.0, 5: 0.7})
    enlarged_data = _synthetic_commutator_bounds({3: 0.4, 4: 2.5, 5: 0.7})

    assert _mizuta_mu_upper_bound(
        exact_data,
        base_order=2,
    ) <= _mizuta_mu_upper_bound(enlarged_data, base_order=2)


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
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
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
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
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
    if estimate.segments > 1:
        previous = estimate_mpf_error(
            hamiltonian,
            0.01,
            estimate.segments - 1,
            2,
            method="mizuta2026-theorem3-legacy-ideal-rigorous",
            target_error=1e-3,
        )
        assert not previous.rigorous or previous.error > 1e-3


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
        method="mizuta2026-theorem3-legacy-ideal-rigorous",
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
