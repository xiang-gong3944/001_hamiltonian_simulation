import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import cosm, sinm

from hamiltonian_resources import (
    BenchmarkConfig,
    benchmark_scaling,
    build_hamiltonian_qsvt_circuit,
    build_multiproduct_circuit,
    build_trotter_circuit,
    choose_parameters,
    compare_with_exact,
    estimate_resources_analytically,
    multiproduct_coefficients,
    optimal_mpf_exponents,
    transverse_field_ising,
)
from hamiltonian_resources.resources import count_circuit_resources
from hamiltonian_resources.circuit_utils import build_block_encoding
from hamiltonian_resources.qsvt import (
    _build_qsvt_component_circuit,
    synthesize_hamsim_phases,
)
from hamiltonian_resources.multiproduct import _build_multiproduct_step_lcu


OPTIMAL_MPF_EXPONENTS = {
    2: (1, 2),
    3: (1, 2, 6),
    4: (1, 2, 3, 10),
    5: (1, 2, 3, 5, 17),
    6: (1, 2, 3, 4, 6, 21),
    7: (1, 2, 3, 4, 5, 9, 34),
    8: (1, 2, 3, 4, 5, 6, 12, 45),
    9: (1, 2, 3, 4, 5, 6, 8, 15, 58),
    10: (1, 2, 3, 4, 5, 6, 7, 10, 18, 72),
    11: (1, 2, 3, 4, 5, 6, 7, 8, 12, 22, 88),
    12: (1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 27, 106),
    13: (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 31, 121),
    14: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 19, 37, 147),
    15: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 22, 42, 170),
}


def _zero_ancilla_block(circuit, system_qubits):
    unitary = Operator(circuit).data
    ancillas = circuit.num_qubits - system_qubits
    selected = np.arange(2**system_qubits) * 2**ancillas
    return unitary[np.ix_(selected, selected)]


def test_pauli_lcu_zero_block_matches_normalized_hamiltonian():
    from hamiltonian_resources import PauliHamiltonian

    hamiltonian = PauliHamiltonian.from_terms(
        1,
        [("I", 0.2), ("X", -0.7), ("Z", 0.3)],
    )
    block = _zero_ancilla_block(build_block_encoding(hamiltonian), 1)
    assert np.allclose(block, hamiltonian.matrix() / hamiltonian.alpha, atol=1e-12)


def test_qsvt_components_have_common_scale_and_correct_zero_blocks():
    from hamiltonian_resources import PauliHamiltonian

    hamiltonian = PauliHamiltonian.from_terms(1, [("Z", 0.3), ("X", 0.7)])
    time = 0.7
    phases = synthesize_hamsim_phases(hamiltonian.alpha * time, 1e-3)

    cosine = _build_qsvt_component_circuit(
        hamiltonian, phases.cosine, component="cos"
    )
    sine = _build_qsvt_component_circuit(hamiltonian, phases.sine, component="sin")
    expected_cosine = phases.scale * cosm(time * hamiltonian.matrix())
    expected_sine = phases.scale * sinm(time * hamiltonian.matrix())

    assert phases.cosine_tail_bound < 1e-3 / 18
    assert phases.sine_tail_bound < 1e-3 / 18
    assert phases.cosine_phase_residual < 1e-3 / 18
    assert phases.sine_phase_residual < 1e-3 / 18
    assert np.allclose(_zero_ancilla_block(cosine, 1), expected_cosine, atol=1e-5)
    assert np.allclose(_zero_ancilla_block(sine, 1), expected_sine, atol=1e-5)


@pytest.mark.parametrize(("m", "expected"), OPTIMAL_MPF_EXPONENTS.items())
def test_optimal_mpf_exponents(m, expected):
    exponents = optimal_mpf_exponents(m)
    assert exponents == expected
    assert len(exponents) == m
    assert all(isinstance(k, int) and k > 0 for k in exponents)
    assert len(set(exponents)) == m


@pytest.mark.parametrize("m", [1, 16])
def test_optimal_mpf_exponents_rejects_unsupported_orders(m):
    with pytest.raises(ValueError, match="between 2 and 15"):
        optimal_mpf_exponents(m)


@pytest.mark.parametrize("m", [2.0, "2", True])
def test_optimal_mpf_exponents_rejects_nonintegers(m):
    with pytest.raises(TypeError, match="integer"):
        optimal_mpf_exponents(m)


@pytest.mark.parametrize("m", OPTIMAL_MPF_EXPONENTS)
def test_multiproduct_order_conditions_are_stable(m):
    exponents = np.asarray(optimal_mpf_exponents(m), dtype=float)
    coefficients = multiproduct_coefficients(m)
    assert np.isclose(sum(coefficients), 1, atol=1e-14)
    for q in range(1, m):
        assert np.isclose(sum(coefficients / exponents ** (2 * q)), 0, atol=1e-14)
    assert sum(abs(coefficients)) < 2


@pytest.mark.parametrize("m", [2, 3])
def test_multiproduct_step_lcu_has_normalization_two(m):
    from hamiltonian_resources import PauliHamiltonian

    hamiltonian = PauliHamiltonian.from_terms(1, [("Z", 0.3), ("X", -0.7)])
    step_time = 0.2
    step = _build_multiproduct_step_lcu(hamiltonian, step_time, m)
    coefficients = multiproduct_coefficients(m)
    exponents = optimal_mpf_exponents(m)
    target = sum(
        coefficient
        * Operator(
            build_trotter_circuit(hamiltonian, step_time, exponent, order=2)
        ).data
        for coefficient, exponent in zip(coefficients, exponents, strict=True)
    )

    assert np.allclose(_zero_ancilla_block(step, 1), target / 2, atol=1e-12)
    assert step.metadata["coefficient_l1_norm"] == pytest.approx(
        sum(abs(coefficients))
    )
    assert step.metadata["padding_weight"] == pytest.approx(
        2 - sum(abs(coefficients))
    )
    assert step.metadata["lcu_normalization"] == 2.0
    assert step.metadata["trotter_step_queries"] == sum(exponents)


def test_all_builders_return_circuits():
    hamiltonian = transverse_field_ising(2)
    trotter = build_trotter_circuit(hamiltonian, 0.1, reps=1)
    mpf = build_multiproduct_circuit(hamiltonian, 0.1, m=2)
    hamsim_qsvt = build_hamiltonian_qsvt_circuit(hamiltonian, 0.1, 1e-2)
    assert all(isinstance(c, QuantumCircuit) for c in (trotter, mpf, hamsim_qsvt))
    assert trotter.num_qubits == 2
    assert mpf.num_qubits > 2
    assert hamsim_qsvt.num_qubits > 2
    assert hamsim_qsvt.metadata["amplitude_amplification"] is True


def test_qsvt_hamiltonian_lcu_and_oaa_zero_blocks():
    from scipy.linalg import expm

    from hamiltonian_resources import PauliHamiltonian

    hamiltonian = PauliHamiltonian.from_terms(1, [("Z", 0.3), ("X", 0.7)])
    time = 0.7
    epsilon = 1e-3
    unamplified = build_hamiltonian_qsvt_circuit(
        hamiltonian, time, epsilon, amplitude_amplification=False
    )
    amplified = build_hamiltonian_qsvt_circuit(hamiltonian, time, epsilon)
    target = expm(-1j * time * hamiltonian.matrix())
    scale = unamplified.metadata["polynomial_scale"]

    assert np.allclose(
        _zero_ancilla_block(unamplified, 1), scale * target / 2, atol=1e-5
    )
    assert np.allclose(_zero_ancilla_block(amplified, 1), target, atol=5e-4)
    assert amplified.metadata["base_circuit_uses"] == 3


def test_qsvt_supports_negative_and_zero_time():
    hamiltonian = transverse_field_ising(1)
    forward = build_hamiltonian_qsvt_circuit(hamiltonian, 0.2, 1e-2)
    backward = build_hamiltonian_qsvt_circuit(hamiltonian, -0.2, 1e-2)
    identity = build_hamiltonian_qsvt_circuit(hamiltonian, 0.0, 1e-2)

    assert np.allclose(
        _zero_ancilla_block(backward, 1),
        _zero_ancilla_block(forward, 1).conj().T,
        atol=5e-3,
    )
    assert identity.num_qubits == 1
    assert np.allclose(Operator(identity).data, np.eye(2))


@pytest.mark.parametrize("epsilon", [0.0, -1e-3, 0.5, 1.0])
def test_qsvt_rejects_invalid_precision(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        build_hamiltonian_qsvt_circuit(transverse_field_ising(1), 0.1, epsilon)


def test_qsvt_exact_comparison_reports_amplified_success():
    rng = np.random.default_rng(1234)
    initial_state = rng.normal(size=4) + 1j * rng.normal(size=4)
    initial_state /= np.linalg.norm(initial_state)
    result = compare_with_exact(
        transverse_field_ising(2),
        0.08,
        method="qsvt",
        initial_state=initial_state,
        qsvt_epsilon=2e-2,
    )

    assert result["state_error"] < 2e-2
    assert result["fidelity"] > 0.999
    assert result["success_probability"] > 0.99


def test_qsvt_unamplified_success_is_one_quarter_up_to_scale():
    rng = np.random.default_rng(5678)
    initial_state = rng.normal(size=2) + 1j * rng.normal(size=2)
    initial_state /= np.linalg.norm(initial_state)
    epsilon = 1e-2
    result = compare_with_exact(
        transverse_field_ising(1),
        0.2,
        method="qsvt",
        initial_state=initial_state,
        qsvt_epsilon=epsilon,
        amplitude_amplification=False,
    )
    expected_scale = 1 - epsilon / 18

    assert result["state_error"] < epsilon
    assert result["success_probability"] == pytest.approx(expected_scale**2 / 4, abs=1e-4)


def test_qsvt_builder_never_forms_dense_hamiltonian(monkeypatch):
    from hamiltonian_resources import PauliHamiltonian

    def fail_if_called(self):
        raise AssertionError("dense matrix construction is forbidden in circuit builders")

    monkeypatch.setattr(PauliHamiltonian, "matrix", fail_if_called)
    circuit = build_hamiltonian_qsvt_circuit(transverse_field_ising(2), 0.05, 2e-2)
    assert circuit.metadata["registers"]["system"] == 2


@pytest.mark.parametrize("time", [np.inf, -np.inf, np.nan])
def test_qsvt_rejects_nonfinite_time(time):
    with pytest.raises(ValueError, match="time"):
        build_hamiltonian_qsvt_circuit(transverse_field_ising(1), time, 1e-2)


def test_multiproduct_circuit_uses_registered_schedule_and_segments():
    circuit = build_multiproduct_circuit(
        transverse_field_ising(2), 0.1, m=3, segments=2
    )
    assert circuit.metadata["m"] == 3
    assert circuit.metadata["exponents"] == (1, 2, 6)
    assert circuit.metadata["segments"] == 2
    assert circuit.metadata["formal_order"] == 6
    branch_labels = [
        instruction.operation.base_gate.label
        for instruction in circuit.data
        if getattr(instruction.operation, "base_gate", None) is not None
        and instruction.operation.base_gate.label is not None
        and instruction.operation.base_gate.label.startswith("S2^")
    ]
    assert branch_labels == ["S2^2", "S2^4", "S2^12"]


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


def test_multiproduct_consumers_use_m_parameter():
    hamiltonian = transverse_field_ising(2)
    config = BenchmarkConfig(time=0.1, target_error=1e-2, mpf_m=3)
    parameters = choose_parameters(hamiltonian, config)
    estimate = estimate_resources_analytically(hamiltonian, config, "multiproduct")
    result = compare_with_exact(
        hamiltonian, 0.1, method="multiproduct", reps=1, mpf_m=3
    )
    branch_bits = 2
    base_rotations = (
        (2 * hamiltonian.term_count - 1)
        * parameters["mpf_segments"]
        * sum(OPTIMAL_MPF_EXPONENTS[3])
    )
    assert estimate.rotation_count == base_rotations * 2**branch_bits + 2 * (
        2**branch_bits - 1
    )
    assert result["fidelity"] > 0.99


def test_benchmark_config_rejects_unsupported_m():
    with pytest.raises(ValueError, match="between 2 and 15"):
        BenchmarkConfig(mpf_m=16)
