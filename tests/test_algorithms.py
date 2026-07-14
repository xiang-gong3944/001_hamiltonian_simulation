import numpy as np
from qiskit import QuantumCircuit

from hamiltonian_resources import (
    BenchmarkConfig,
    benchmark_scaling,
    build_hamiltonian_qsvt_circuit,
    build_multiproduct_circuit,
    build_qsvt_circuit,
    build_trotter_circuit,
    compare_with_exact,
    multiproduct_coefficients,
    transverse_field_ising,
)
from hamiltonian_resources.resources import count_circuit_resources


def test_multiproduct_order_conditions():
    exponents = np.array([1, 2, 4])
    coefficients = multiproduct_coefficients(exponents)
    assert np.isclose(sum(coefficients), 1)
    assert np.isclose(sum(coefficients / exponents**2), 0)
    assert np.isclose(sum(coefficients / exponents**4), 0)


def test_all_builders_return_circuits():
    hamiltonian = transverse_field_ising(2)
    trotter = build_trotter_circuit(hamiltonian, 0.1, reps=1)
    mpf = build_multiproduct_circuit(hamiltonian, 0.1, (1, 2))
    qsvt = build_qsvt_circuit(hamiltonian, [0.1, 0.2, 0.1])
    hamsim_qsvt = build_hamiltonian_qsvt_circuit(
        hamiltonian, [0.1, 0.2, 0.1], [0.3, 0.4]
    )
    assert all(isinstance(c, QuantumCircuit) for c in (trotter, mpf, qsvt))
    assert trotter.num_qubits == 2
    assert mpf.num_qubits > 2
    assert qsvt.metadata["degree"] == 2
    assert hamsim_qsvt.num_qubits == qsvt.num_qubits + 1


def test_commuting_trotter_is_exact_on_state():
    from hamiltonian_resources import PauliHamiltonian

    hamiltonian = PauliHamiltonian.from_terms(2, [("ZI", 0.7), ("IZ", -0.2)])
    result = compare_with_exact(hamiltonian, 0.8, method="trotter", reps=1)
    assert result["state_error"] < 1e-10


def test_resource_counter_counts_cx():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.rz(0.123, 1)
    estimate = count_circuit_resources(circuit, total_synthesis_error=1e-4)
    assert estimate.cnot_count == 1
    assert estimate.t_count > 0


def test_analytical_benchmark_does_not_build_large_unitaries():
    frame = benchmark_scaling(
        [2, 4], transverse_field_ising, BenchmarkConfig(time=0.1, target_error=1e-2)
    )
    assert len(frame) == 6
    assert set(frame["counting_mode"]) == {"analytical-model"}
    assert (frame["t_count"] > 0).all()
    assert (frame["nominal_success_probability"] <= 1).all()
