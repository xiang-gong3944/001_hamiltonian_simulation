from dataclasses import FrozenInstanceError, replace

import pytest

from hamiltonian_resources.benchmark_suite import (
    MultiproductMethod,
    QSVTMethod,
    TrotterMethod,
)
from hamiltonian_resources.hamiltonians import PauliHamiltonian, transverse_field_ising
from hamiltonian_resources.planning import (
    ErrorBudget,
    MPFPlan,
    QSVTPlan,
    TrotterPlan,
    plan_simulation,
)


def test_error_budget_derives_consistent_components():
    budget = ErrorBudget(1e-3, 0.2)

    assert budget.algorithm_error == pytest.approx(8e-4)
    assert budget.synthesis_error == pytest.approx(2e-4)
    with pytest.raises(FrozenInstanceError):
        budget.target_error = 0.2


def test_plans_are_deterministic_and_deeply_immutable():
    hamiltonian = PauliHamiltonian(1, [("X", 0.7), ("Z", -0.3)])
    first = plan_simulation(hamiltonian, TrotterMethod(2), 0.2, 1e-3)
    second = plan_simulation(hamiltonian, TrotterMethod(2), 0.2, 1e-3)

    assert first == second
    assert isinstance(first.hamiltonian.terms, tuple)
    assert isinstance(first.group_term_indices, tuple)
    assert all(isinstance(group, tuple) for group in first.group_term_indices)
    with pytest.raises(FrozenInstanceError):
        first.repetitions = 10
    with pytest.raises(ValueError, match="repetitions"):
        replace(first, repetitions=first.repetitions + 1)


@pytest.mark.parametrize(
    ("method", "plan_type", "parameter"),
    [
        (TrotterMethod(2), TrotterPlan, "trotter_reps"),
        (MultiproductMethod(3), MPFPlan, "mpf_segments"),
        (QSVTMethod(), QSVTPlan, "qsvt_degree"),
    ],
)
def test_plan_simulation_selects_one_family(method, plan_type, parameter):
    plan = plan_simulation(transverse_field_ising(2, field=0.7), method, 0.1, 1e-2)

    assert isinstance(plan, plan_type)
    assert set(plan.selected_parameters) >= {parameter}
    assert plan.error_budget.algorithm_error == pytest.approx(9e-3)
    assert plan.error_metadata["bound_method"]
    assert plan.logical_counts.as_dict()["totals"]
    for backend_field in ("t_count", "cnot_count", "toffoli_count", "work_qubits"):
        assert not hasattr(plan, backend_field)


def test_trotter_plan_preserves_resolved_term_index_order():
    hamiltonian = transverse_field_ising(3, field=0.7)
    plan = plan_simulation(
        hamiltonian,
        TrotterMethod(4),
        0.2,
        1e-2,
    )

    assert plan.resolved_partition == "commuting"
    assert plan.group_term_indices == ((0, 1), (2, 3, 4))
    assert sorted(index for group in plan.group_term_indices for index in group) == list(
        range(hamiltonian.term_count)
    )
    assert plan.logical_counts.as_dict()["totals"]["pauli_evolution"] == 35


def test_mpf_plan_owns_logical_structure_not_backend_decomposition():
    plan = plan_simulation(
        transverse_field_ising(2, field=0.7),
        MultiproductMethod(3),
        0.1,
        1e-2,
    )
    counts = plan.logical_counts.as_dict()

    assert plan.exponents == (1, 2, 4)
    assert plan.lcu_structure.physical_branch_count == 3
    assert counts["per_segment"] == {
        "prepare": 6,
        "select": 3,
        "good_reflection": 2,
        "controlled_s2": 21,
        "pauli_evolution": 105,
    }
    assert "temporary_and" not in counts["totals"]


def test_dynamic_mpf_policy_resolves_once_from_the_algorithm_budget():
    hamiltonian = PauliHamiltonian.from_terms(1, [("Z", 1.0)])
    method = MultiproductMethod(
        None,
        error_method="low2019-l1-ideal-rigorous",
        branch_count_policy="mizuta2026-theorem6",
    )

    plan = plan_simulation(hamiltonian, method, 5.0, 0.1)

    # The raw target would give J=2, while the default 90% algorithm budget
    # gives J=3. This guards the policy input against target/auxiliary leakage.
    assert plan.error_budget.algorithm_error == pytest.approx(0.09)
    assert plan.term_count == plan.branch_count_selection.branch_count == 3
    assert plan.formal_order == 6
    assert plan.exponents == (1, 2, 4)
    assert plan.error_estimate.m == 3
    assert plan.selected_parameters["mpf_m"] == 3
    assert plan.selected_parameters["mpf_branch_count"] == 3
    assert plan.selected_parameters["mpf_formal_order"] == 6
    assert plan.selected_parameters["mpf_branch_count_policy"] == (
        "mizuta2026-theorem6"
    )
    sizing = plan.error_analysis.sizing_estimate
    assert sizing.term_count == 3
    assert sizing.formal_order == 6
    assert sizing.branch_count_policy == "mizuta2026-theorem6"
    assert sizing.branch_count_policy_target_error == pytest.approx(0.09)
    assert sizing.branch_count_policy_extensiveness_g == pytest.approx(1.0)


def test_all_mpf_bounds_consume_the_same_dynamic_selection():
    hamiltonian = PauliHamiltonian.from_terms(1, [("Z", 1.0)])
    error_methods = (
        "low2019-l1-ideal-rigorous",
        "mizuta2026-theorem3-legacy-ideal-rigorous",
        "mizuta2026-commutator-ideal-rigorous",
        "best-rigorous-ideal",
    )

    plans = tuple(
        plan_simulation(
            hamiltonian,
            MultiproductMethod(
                None,
                error_method=error_method,
                branch_count_policy="mizuta2026-theorem6",
            ),
            5.0,
            0.1,
        )
        for error_method in error_methods
    )

    assert {plan.term_count for plan in plans} == {3}
    assert {plan.exponents for plan in plans} == {(1, 2, 4)}
    assert len({plan.coefficients for plan in plans}) == 1
    assert len({plan.lcu_structure for plan in plans}) == 1
    refined = plans[2]
    assert refined.error_estimate.segment_diagnostics.truncation_order_p0 >= (
        2 * refined.term_count
    )


def test_sequential_dynamic_orders_do_not_reuse_stale_schedule_data():
    hamiltonian = PauliHamiltonian.from_terms(1, [("Z", 1.0)])
    method = MultiproductMethod(
        None,
        branch_count_policy="mizuta2026-theorem6",
    )

    plans = tuple(
        plan_simulation(hamiltonian, method, time, 0.1)
        for time in (0.5, 5.0, 0.5)
    )

    assert tuple(plan.term_count for plan in plans) == (2, 3, 2)
    assert tuple(plan.exponents for plan in plans) == ((1, 2), (1, 2, 4), (1, 2))
    assert plans[0].coefficients == plans[2].coefficients
    assert plans[0].lcu_structure == plans[2].lcu_structure
    assert plans[0].logical_counts == plans[2].logical_counts
    assert plans[0].lcu_structure != plans[1].lcu_structure


def test_qsvt_plan_describes_logical_response_slots_only():
    plan = plan_simulation(
        transverse_field_ising(2, field=0.7),
        QSVTMethod(),
        0.1,
        1e-2,
    )
    counts = plan.logical_counts.as_dict()["totals"]

    assert plan.degree == 3
    assert (plan.cosine_degree, plan.sine_degree) == (2, 3)
    assert counts["block_encoding_query_slot"] == 15
    assert counts["projector_phase_slot"] == 21
    assert "generic_qiskit_query" not in counts
    assert "structured_query" not in counts


def test_trotter_selection_does_not_touch_other_family_selectors(monkeypatch):
    import hamiltonian_resources.planning as planning

    def fail(*args, **kwargs):
        raise AssertionError("unrelated family selector was called")

    monkeypatch.setattr(planning, "select_mpf_segments", fail)
    monkeypatch.setattr(planning, "estimate_qsvt_degree", fail)

    plan = plan_simulation(
        transverse_field_ising(2),
        TrotterMethod(2),
        0.1,
        1e-2,
    )
    assert isinstance(plan, TrotterPlan)


@pytest.mark.parametrize("time", [0.0, -0.1])
def test_resource_planning_requires_positive_time(time):
    with pytest.raises(ValueError, match="positive"):
        plan_simulation(transverse_field_ising(1), TrotterMethod(2), time, 1e-2)
