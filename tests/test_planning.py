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
