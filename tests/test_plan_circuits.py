import numpy as np
import pytest
from qiskit.quantum_info import Operator

from hamiltonian_resources import (
    MultiproductMethod,
    QSVTMethod,
    TrotterMethod,
    build_hamiltonian_qsvt_circuit,
    build_multiproduct_circuit,
    build_simulation_circuit,
    build_trotter_circuit,
    plan_simulation,
    transverse_field_ising,
)


def test_trotter_plan_builder_uses_selected_groups_and_repetitions():
    hamiltonian = transverse_field_ising(3, field=0.7)
    plan = plan_simulation(hamiltonian, TrotterMethod(4), 0.2, 1e-2)
    planned = build_simulation_circuit(plan)
    legacy = build_trotter_circuit(
        hamiltonian,
        0.2,
        plan.repetitions,
        4,
        partition="auto",
    )

    assert planned.metadata["trotter_partition"] == plan.resolved_partition
    assert planned.metadata["trotter_group_sizes"] == tuple(
        len(group) for group in plan.group_term_indices
    )
    assert np.allclose(Operator(planned).data, Operator(legacy).data, atol=1e-12)


def test_mpf_plan_builder_consumes_stored_lcu_structure(monkeypatch):
    import hamiltonian_resources.planning as planning

    hamiltonian = transverse_field_ising(1, field=0.7)
    plan = plan_simulation(hamiltonian, MultiproductMethod(2), 0.001, 0.1)

    def fail(*args, **kwargs):
        raise AssertionError("MPF selection was repeated during concrete compilation")

    monkeypatch.setattr(planning, "select_mpf_segments", fail)
    planned = build_simulation_circuit(plan)
    legacy = build_multiproduct_circuit(
        hamiltonian,
        plan.time,
        m=2,
        segments=plan.segments,
    )

    assert planned.metadata["exponents"] == plan.exponents
    assert planned.metadata["segments"] == plan.segments
    assert planned.metadata["physical_branch_count"] == plan.lcu_structure.physical_branch_count
    assert np.allclose(Operator(planned).data, Operator(legacy).data, atol=1e-12)


def test_qsvt_plan_builder_uses_fixed_selected_degree(monkeypatch):
    import hamiltonian_resources.qsvt as qsvt

    hamiltonian = transverse_field_ising(1, field=0.7)
    plan = plan_simulation(hamiltonian, QSVTMethod(), 0.05, 2e-2)

    def fail(*args, **kwargs):
        raise AssertionError("QSVT degree selection was repeated during phase synthesis")

    monkeypatch.setattr(qsvt, "estimate_qsvt_degree", fail)
    circuit = build_simulation_circuit(plan)

    assert circuit.metadata["cosine_degree"] == plan.cosine_degree
    assert circuit.metadata["sine_degree"] == plan.sine_degree
    direct = build_hamiltonian_qsvt_circuit(
        hamiltonian,
        plan.time,
        plan.error_budget.algorithm_error,
    )
    assert circuit.num_qubits == direct.num_qubits


def test_plan_dispatch_rejects_nonplans():
    with pytest.raises(TypeError, match="plan"):
        build_simulation_circuit(object())
