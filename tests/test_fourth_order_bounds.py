import math

import numpy as np
import pytest
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import expm

from hamiltonian_resources import (
    PauliHamiltonian,
    all_schubert_mendl_centers,
    build_fourth_order_bound_problem,
    build_trotter_circuit,
    childs_fourth_order_small_prefactor_bound,
    childs_general_commutator_bound,
    heisenberg_chain,
    minimizing_schubert_mendl_center,
    schubert_mendl_small_prefactor_bound,
    transverse_field_ising,
)


EXPECTED_M13 = {
    (0, 0, 0, 1, 0): 0.0047013343101698826,
    (0, 0, 1, 1, 0): 0.00570381876262935,
    (0, 1, 0, 1, 0): 0.004638910081589791,
    (0, 1, 1, 1, 0): 0.00737205664595221,
    (1, 0, 0, 1, 0): 0.00968966829012217,
    (1, 0, 1, 1, 0): 0.009726162358456834,
    (1, 1, 0, 1, 0): 0.01732815305710953,
    (1, 1, 1, 1, 0): 0.0283734344054259,
}


def _coefficient_map(result):
    return {
        contribution.hamiltonian_indices: contribution.prefactor
        for contribution in result.contributions
    }


def _two_term_hamiltonian(scale=1.0):
    return PauliHamiltonian.from_terms(
        1,
        [("X", 0.7 * scale), ("Z", -0.2 * scale)],
        f"two-term-scale-{scale}",
    )


def test_shared_merged_formula_matches_qiskit_and_all_evaluators_use_it():
    hamiltonian = _two_term_hamiltonian()
    problem = build_fourth_order_bound_problem(
        hamiltonian,
        partition="individual",
    )
    synthesis = SuzukiTrotter(order=4, reps=1, preserve_order=True)
    gate = PauliEvolutionGate(list(problem.groups), time=1.0, synthesis=synthesis)
    expanded = synthesis.expand(gate)
    labels = {"X": 0, "Z": 1}
    coefficients = {"X": 0.7, "Z": -0.2}
    qiskit_factors = tuple(
        (labels[label], float(angle) / (2 * coefficients[label]))
        for label, _, angle in expanded
    )
    merged = []
    for group, coefficient in qiskit_factors:
        if merged and merged[-1][0] == group:
            merged[-1] = (group, merged[-1][1] + coefficient)
        else:
            merged.append((group, coefficient))

    assert problem.order == 4
    assert problem.time_power == 5
    assert problem.exponential_count == 11
    assert problem.stage_count == 10
    assert [group for group, _ in problem.ordered_exponentials] == [
        group for group, _ in merged
    ]
    assert np.allclose(
        [coefficient for _, coefficient in problem.ordered_exponentials],
        [coefficient for _, coefficient in merged],
        atol=1e-15,
    )

    results = (
        childs_general_commutator_bound(problem),
        childs_fourth_order_small_prefactor_bound(problem),
        schubert_mendl_small_prefactor_bound(problem),
    )
    assert all(result.problem is problem for result in results)
    assert all(result.ordered_exponentials is problem.ordered_exponentials for result in results)


def test_schubert_mendl_reproduces_childs_m13_term_by_term_and_in_total():
    problem = build_fourth_order_bound_problem(
        _two_term_hamiltonian(),
        partition="individual",
    )
    childs = childs_fourth_order_small_prefactor_bound(problem)
    schubert = schubert_mendl_small_prefactor_bound(problem, center=6)

    assert _coefficient_map(childs) == EXPECTED_M13
    assert _coefficient_map(schubert).keys() == EXPECTED_M13.keys()
    for indices, expected in EXPECTED_M13.items():
        assert _coefficient_map(schubert)[indices] == pytest.approx(expected, abs=2e-17)
    assert schubert.one_step_coefficient == pytest.approx(
        childs.one_step_coefficient,
        rel=2e-15,
    )


def test_local_and_accumulated_bounds_have_required_time_and_segment_scaling():
    problem = build_fourth_order_bound_problem(
        _two_term_hamiltonian(),
        partition="individual",
    )
    result = schubert_mendl_small_prefactor_bound(problem)

    assert result.local_error_bound(0.2) / result.local_error_bound(0.1) == pytest.approx(
        2**5
    )
    assert result.accumulated_error_bound(0.4, 2) == pytest.approx(
        result.one_step_coefficient * 0.4**5 / 2**4
    )
    assert result.accumulated_error_bound(0.4, 4) == pytest.approx(
        result.accumulated_error_bound(0.4, 2) / 2**4
    )
    target = 1e-5
    segments = result.required_segments(0.4, target)
    assert result.accumulated_error_bound(0.4, segments) <= target
    if segments > 1:
        assert result.accumulated_error_bound(0.4, segments - 1) > target


def test_exact_local_product_formula_error_is_fifth_order():
    hamiltonian = _two_term_hamiltonian()
    errors = []
    for time in (0.2, 0.1):
        circuit = build_trotter_circuit(
            hamiltonian,
            time,
            1,
            4,
            partition="individual",
        )
        exact = expm(-1j * time * hamiltonian.matrix())
        errors.append(float(np.linalg.norm(Operator(circuit).data - exact, 2)))

    assert errors[0] / errors[1] == pytest.approx(2**5, rel=1e-3)


def test_three_term_appendix_m_is_s10_while_centered_and_minimum_are_retained():
    problem = build_fourth_order_bound_problem(heisenberg_chain(3, field_z=0.3))
    childs = childs_fourth_order_small_prefactor_bound(problem)
    s10 = schubert_mendl_small_prefactor_bound(problem, center=10)
    centered = schubert_mendl_small_prefactor_bound(problem, center=11)
    all_centers = all_schubert_mendl_centers(problem)
    minimum = minimizing_schubert_mendl_center(all_centers)

    assert problem.group_count == 3
    assert problem.exponential_count == 21
    assert childs.center_index == 10
    assert centered.center_index == problem.centered_index == 11
    assert len(all_centers) == problem.exponential_count
    assert _coefficient_map(s10).keys() == _coefficient_map(childs).keys()
    assert len(childs.contributions) == 81
    assert s10.one_step_coefficient == pytest.approx(
        childs.one_step_coefficient,
        rel=2e-15,
    )
    assert centered.one_step_coefficient != pytest.approx(childs.one_step_coefficient)
    assert minimum.one_step_coefficient <= centered.one_step_coefficient
    assert minimum.center_index is not None

    table = _coefficient_map(childs)
    assert table[(0, 0, 0, 1, 0)] == pytest.approx(0.0047013343101698826)
    assert table[(0, 0, 0, 2, 1)] == pytest.approx(0.004310439277389538)
    assert table[(1, 0, 0, 1, 0)] == pytest.approx(0.015024679162295496)


def test_expanding_bj_triangle_is_explicit_and_can_only_loosen_the_bound():
    problem = build_fourth_order_bound_problem(heisenberg_chain(3, field_z=0.3))
    expanded = schubert_mendl_small_prefactor_bound(
        problem,
        center=problem.centered_index,
        expand_base_triangle=True,
    )
    combined = schubert_mendl_small_prefactor_bound(
        problem,
        center=problem.centered_index,
        expand_base_triangle=False,
    )

    assert combined.one_step_coefficient <= expanded.one_step_coefficient
    assert "B_j retained" in combined.diagnostic_message
    assert "B_j expanded" in expanded.diagnostic_message


def test_merging_is_recorded_and_representation_dependence_is_measured():
    hamiltonian = heisenberg_chain(3, field_z=0.3)
    merged_problem = build_fourth_order_bound_problem(
        hamiltonian,
        merge_adjacent=True,
    )
    unmerged_problem = build_fourth_order_bound_problem(
        hamiltonian,
        merge_adjacent=False,
    )
    merged = schubert_mendl_small_prefactor_bound(merged_problem)
    unmerged = schubert_mendl_small_prefactor_bound(unmerged_problem)

    assert merged_problem.merged_consecutive
    assert not unmerged_problem.merged_consecutive
    assert merged_problem.exponential_count == 21
    assert unmerged_problem.exponential_count == 25
    assert merged.one_step_coefficient != pytest.approx(unmerged.one_step_coefficient)


def test_norm_evaluation_method_isolated_from_symbolic_prefactors():
    problem = build_fourth_order_bound_problem(heisenberg_chain(3, field_z=0.3))
    pauli_l1 = schubert_mendl_small_prefactor_bound(
        problem,
        center=10,
        norm_method="pauli-l1",
    )
    spectral = schubert_mendl_small_prefactor_bound(
        problem,
        center=10,
        norm_method="spectral",
    )

    assert _coefficient_map(pauli_l1) == _coefficient_map(spectral)
    assert spectral.one_step_coefficient <= pauli_l1.one_step_coefficient
    assert pauli_l1.additional_relaxations
    assert not spectral.additional_relaxations


def test_common_hamiltonian_rescaling_is_fifth_degree_and_preserves_ratios():
    base_problem = build_fourth_order_bound_problem(
        _two_term_hamiltonian(),
        partition="individual",
    )
    scaled_problem = build_fourth_order_bound_problem(
        _two_term_hamiltonian(scale=3.0),
        partition="individual",
    )
    base_childs = childs_fourth_order_small_prefactor_bound(base_problem)
    base_schubert = schubert_mendl_small_prefactor_bound(base_problem)
    scaled_childs = childs_fourth_order_small_prefactor_bound(scaled_problem)
    scaled_schubert = schubert_mendl_small_prefactor_bound(scaled_problem)

    assert scaled_childs.one_step_coefficient / base_childs.one_step_coefficient == pytest.approx(
        3**5
    )
    assert scaled_schubert.one_step_coefficient / base_schubert.one_step_coefficient == pytest.approx(
        3**5
    )
    assert (
        base_schubert.one_step_coefficient / base_childs.one_step_coefficient
    ) == pytest.approx(
        scaled_schubert.one_step_coefficient / scaled_childs.one_step_coefficient
    )


def test_unsupported_appendix_m_decomposition_is_reported_not_substituted():
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("XI", 0.2), ("YI", -0.3), ("IX", 0.4), ("IZ", 0.5)],
        "four-terms",
    )
    problem = build_fourth_order_bound_problem(
        hamiltonian,
        partition="individual",
    )
    result = childs_fourth_order_small_prefactor_bound(problem)

    assert problem.group_count == 4
    assert result.status == "unsupported"
    assert result.one_step_coefficient is None
    assert "only for decompositions with two or three" in result.diagnostic_message
    with pytest.raises(ValueError, match="only for decompositions"):
        result.local_error_bound(0.1)


@pytest.mark.parametrize(
    "evaluator",
    [
        childs_general_commutator_bound,
        childs_fourth_order_small_prefactor_bound,
        schubert_mendl_small_prefactor_bound,
    ],
)
def test_small_system_bounds_dominate_exact_operator_norm_error(evaluator):
    hamiltonian = _two_term_hamiltonian()
    problem = build_fourth_order_bound_problem(
        hamiltonian,
        partition="individual",
    )
    result = evaluator(problem, norm_method="spectral")
    time = 0.2
    circuit = build_trotter_circuit(
        hamiltonian,
        time,
        1,
        4,
        partition="individual",
    )
    exact = expm(-1j * time * hamiltonian.matrix())
    actual = float(np.linalg.norm(Operator(circuit).data - exact, 2))

    assert actual <= result.local_error_bound(time) * (1 + 1e-12) + 1e-12


def test_invalid_centers_norms_and_segment_inputs_are_rejected():
    problem = build_fourth_order_bound_problem(
        _two_term_hamiltonian(),
        partition="individual",
    )
    with pytest.raises(ValueError, match="center"):
        schubert_mendl_small_prefactor_bound(problem, center=0)
    with pytest.raises(TypeError, match="center"):
        schubert_mendl_small_prefactor_bound(problem, center=True)
    with pytest.raises(ValueError, match="norm_method"):
        childs_general_commutator_bound(problem, norm_method="bad")
    result = schubert_mendl_small_prefactor_bound(problem)
    with pytest.raises(ValueError, match="segments"):
        result.accumulated_error_bound(1.0, 0)
    with pytest.raises(ValueError, match="target_error"):
        result.required_segments(1.0, 0.0)


def test_suzuki_coefficients_are_the_standard_recursive_values():
    problem = build_fourth_order_bound_problem(
        transverse_field_ising(3, field=0.7)
    )
    expected_z1 = 1 / (4 - 4 ** (1 / 3))

    assert problem.z1 == pytest.approx(expected_z1)
    assert problem.z0 == pytest.approx(1 - 4 * expected_z1)
    assert math.fsum(
        coefficient
        for group, coefficient in problem.ordered_exponentials
        if group == 0
    ) == pytest.approx(1.0)
    assert math.fsum(
        coefficient
        for group, coefficient in problem.ordered_exponentials
        if group == 1
    ) == pytest.approx(1.0)
