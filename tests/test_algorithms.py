import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
from scipy.linalg import cosm, sinm

from hamiltonian_resources import (
    BenchmarkConfig,
    HamiltonianSpec,
    MultiproductMethod,
    TimeScaling,
    build_hamiltonian_qsvt_circuit,
    build_multiproduct_circuit,
    build_trotter_circuit,
    compare_with_exact,
    multiproduct_coefficients,
    optimal_mpf_exponents,
    run_benchmark,
    transverse_field_ising,
)
from hamiltonian_resources.benchmark import (
    _EvaluationConfig,
    choose_parameters,
    estimate_resources_analytically,
)
from hamiltonian_resources.resources import count_circuit_resources
from hamiltonian_resources.circuit_utils import (
    build_block_encoding,
    index_state_phase_gate,
    state_preparation,
    zero_projector_phase_gate,
)
from hamiltonian_resources.qsvt import (
    _build_qsvt_component_circuit,
    synthesize_hamsim_phases,
)
from hamiltonian_resources.multiproduct import _build_multiproduct_step_lcu


NEW_MPF_EXPONENTS = {
    2: (1, 2),
    3: (1, 2, 4),
    4: (1, 2, 3, 7),
    5: (1, 2, 3, 5, 12),
    6: (1, 2, 3, 4, 6, 16),
    7: (1, 2, 3, 4, 5, 9, 22),
    8: (1, 2, 3, 4, 5, 6, 11, 29),
    9: (1, 2, 3, 4, 5, 6, 8, 14, 37),
    10: (1, 2, 3, 4, 5, 6, 7, 10, 18, 46),
    11: (1, 2, 3, 4, 5, 6, 7, 8, 12, 22, 56),
    12: (1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 26, 66),
    13: (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 30, 78),
    14: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 19, 35, 91),
    15: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 22, 40, 104),
}


LEGACY_MPF_EXPONENTS = {
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


def _classical_mpf_step(hamiltonian, step_time, m, schedule="new"):
    return sum(
        coefficient
        * Operator(
            build_trotter_circuit(hamiltonian, step_time, exponent, order=2)
        ).data
        for coefficient, exponent in zip(
            multiproduct_coefficients(m, schedule=schedule),
            optimal_mpf_exponents(m, schedule=schedule),
            strict=True,
        )
    )


def test_named_lcu_primitives_preserve_their_gate_definitions():
    prepare = state_preparation(np.array([np.sqrt(0.25), np.sqrt(0.75)]), "PREPARE_TEST")
    phase = index_state_phase_gate(2, 2, 0.37)
    reflection = zero_projector_phase_gate(2, np.pi, name="GOOD_REFLECTION")

    assert all(isinstance(gate, Gate) for gate in (prepare, phase, reflection))
    assert all(gate.definition is not None for gate in (prepare, phase, reflection))
    expected_phase = np.eye(4, dtype=complex)
    expected_phase[2, 2] = np.exp(0.37j)
    expected_reflection = np.diag([-1, 1, 1, 1])
    assert np.allclose(Operator(phase).data, expected_phase, atol=1e-12)
    assert np.allclose(Operator(reflection).data, expected_reflection, atol=1e-12)


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


@pytest.mark.parametrize(("m", "expected"), NEW_MPF_EXPONENTS.items())
def test_new_mpf_exponents_are_the_default(m, expected):
    exponents = optimal_mpf_exponents(m)
    assert exponents == expected
    assert len(exponents) == m
    assert all(isinstance(k, int) and k > 0 for k in exponents)
    assert len(set(exponents)) == m


@pytest.mark.parametrize(("m", "expected"), LEGACY_MPF_EXPONENTS.items())
def test_legacy_mpf_exponents_remain_selectable(m, expected):
    assert optimal_mpf_exponents(m, schedule="legacy") == expected


@pytest.mark.parametrize("m", range(3, 16))
def test_new_mpf_schedule_reduces_controlled_u2_queries(m):
    assert sum(NEW_MPF_EXPONENTS[m]) < sum(LEGACY_MPF_EXPONENTS[m])


def test_mpf_schedule_rejects_unknown_name():
    with pytest.raises(ValueError, match="schedule"):
        optimal_mpf_exponents(3, schedule="unknown")


@pytest.mark.parametrize("m", [1, 16])
def test_optimal_mpf_exponents_rejects_unsupported_orders(m):
    with pytest.raises(ValueError, match="between 2 and 15"):
        optimal_mpf_exponents(m)


@pytest.mark.parametrize("m", [2.0, "2", True])
def test_optimal_mpf_exponents_rejects_nonintegers(m):
    with pytest.raises(TypeError, match="integer"):
        optimal_mpf_exponents(m)


@pytest.mark.parametrize("schedule", ["new", "legacy"])
@pytest.mark.parametrize("m", NEW_MPF_EXPONENTS)
def test_multiproduct_order_conditions_are_stable(m, schedule):
    exponents = np.asarray(
        optimal_mpf_exponents(m, schedule=schedule),
        dtype=float,
    )
    coefficients = multiproduct_coefficients(m, schedule=schedule)
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
    target = _classical_mpf_step(hamiltonian, step_time, m)

    assert np.allclose(_zero_ancilla_block(step, 1), target / 2, atol=1e-12)
    assert step.metadata["coefficient_l1_norm"] == pytest.approx(
        sum(abs(coefficients))
    )
    assert step.metadata["padding_weight"] == pytest.approx(
        2 - sum(abs(coefficients))
    )
    assert step.metadata["lcu_normalization"] == 2.0
    assert step.metadata["trotter_step_queries"] == sum(exponents)
    assert [instruction.operation.name for instruction in step.data] == [
        "state_preparation",
        "SELECT_MPF",
        "state_preparation_dg",
    ]
    assert step.data[1].operation.definition is not None


def test_multiproduct_select_gate_contains_signed_and_padding_branches():
    from hamiltonian_resources import PauliHamiltonian

    hamiltonian = PauliHamiltonian.from_terms(1, [("X", 0.6), ("Z", -0.4)])
    step_time = 0.3
    m = 3
    step = _build_multiproduct_step_lcu(hamiltonian, step_time, m)
    select = step.data[1].operation
    select_matrix = Operator(select).data
    branch_width = select.num_qubits - hamiltonian.num_qubits
    system_dimension = 2**hamiltonian.num_qubits
    coefficients = multiproduct_coefficients(m)
    exponents = optimal_mpf_exponents(m)

    for branch_value in range(2**branch_width):
        selected = branch_value + 2**branch_width * np.arange(system_dimension)
        branch_block = select_matrix[np.ix_(selected, selected)]
        if branch_value < m:
            expected = np.sign(coefficients[branch_value]) * Operator(
                build_trotter_circuit(
                    hamiltonian,
                    step_time,
                    exponents[branch_value],
                    order=2,
                )
            ).data
        elif branch_value == m:
            expected = np.eye(system_dimension)
        elif branch_value == m + 1:
            expected = -np.eye(system_dimension)
        else:
            expected = np.eye(system_dimension)
        assert np.allclose(branch_block, expected, atol=1e-12)


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


def test_multiproduct_circuit_amplifies_each_registered_segment():
    circuit = build_multiproduct_circuit(
        transverse_field_ising(2), 0.1, m=3, segments=2
    )
    assert circuit.metadata["m"] == 3
    assert circuit.metadata["schedule"] == "new"
    assert circuit.metadata["exponents"] == (1, 2, 4)
    assert circuit.metadata["exponent_sum"] == 7
    assert circuit.metadata["segments"] == 2
    assert circuit.metadata["step_time"] == pytest.approx(0.05)
    assert circuit.metadata["formal_order"] == 6
    assert circuit.metadata["lcu_normalization"] == 2.0
    assert circuit.metadata["amplitude_amplification"] is True
    assert circuit.metadata["base_lcu_uses_per_segment"] == 3
    assert circuit.metadata["trotter_step_queries"] == 3 * 2 * (1 + 2 + 4)
    assert circuit.metadata["logical_gate_counts_per_segment"] == {
        "prepare": 6,
        "select": 3,
        "good_reflection": 2,
        "controlled_u2": 21,
    }
    assert circuit.metadata["logical_gate_counts"] == {
        "prepare": 12,
        "select": 6,
        "good_reflection": 4,
        "controlled_u2": 42,
    }
    assert len(circuit.data) == 2


def test_multiproduct_unamplified_public_block_is_one_half_step():
    from hamiltonian_resources import PauliHamiltonian

    hamiltonian = PauliHamiltonian.from_terms(1, [("X", 0.6), ("Z", -0.4)])
    circuit = build_multiproduct_circuit(
        hamiltonian, 0.2, m=3, amplitude_amplification=False
    )
    target = _classical_mpf_step(hamiltonian, 0.2, 3)

    assert np.allclose(_zero_ancilla_block(circuit, 1), target / 2, atol=1e-12)
    assert circuit.metadata["amplitude_amplification"] is False


@pytest.mark.parametrize(("m", "schedule"), [(2, "new"), (3, "new"), (3, "legacy")])
def test_multiproduct_oaa_has_exact_cubic_good_block(m, schedule):
    from hamiltonian_resources import PauliHamiltonian

    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("XI", 0.7), ("ZZ", -0.9), ("IY", 0.4)],
    )
    step_time = 0.7
    multiproduct = _classical_mpf_step(
        hamiltonian,
        step_time,
        m,
        schedule,
    )
    unamplified = build_multiproduct_circuit(
        hamiltonian,
        step_time,
        m=m,
        schedule=schedule,
        amplitude_amplification=False,
    )
    amplified = build_multiproduct_circuit(
        hamiltonian,
        step_time,
        m=m,
        schedule=schedule,
    )
    block = multiproduct / 2
    expected_amplified = 3 * block - 4 * block @ block.conj().T @ block

    assert np.allclose(
        _zero_ancilla_block(unamplified, 2),
        block,
        atol=1e-12,
    )
    assert np.allclose(
        _zero_ancilla_block(amplified, 2),
        expected_amplified,
        atol=5e-12,
    )


def test_multiproduct_random_state_success_matches_good_blocks():
    from hamiltonian_resources import PauliHamiltonian

    rng = np.random.default_rng(2468)
    initial_state = rng.normal(size=4) + 1j * rng.normal(size=4)
    initial_state /= np.linalg.norm(initial_state)
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("XI", 0.7), ("ZZ", -0.9), ("IY", 0.4)],
    )
    time = 0.7
    multiproduct = _classical_mpf_step(hamiltonian, time, 2)
    block_before = multiproduct / 2
    block_after = (
        3 * block_before
        - 4 * block_before @ block_before.conj().T @ block_before
    )
    before = compare_with_exact(
        hamiltonian,
        time,
        method="multiproduct",
        initial_state=initial_state,
        mpf_m=2,
        amplitude_amplification=False,
    )
    after = compare_with_exact(
        hamiltonian,
        time,
        method="multiproduct",
        initial_state=initial_state,
        mpf_m=2,
    )

    expected_before = np.linalg.norm(block_before @ initial_state) ** 2
    expected_after = np.linalg.norm(block_after @ initial_state) ** 2
    assert before["success_probability"] == pytest.approx(expected_before, abs=1e-12)
    assert after["success_probability"] == pytest.approx(expected_after, abs=1e-12)
    assert after["success_probability"] > before["success_probability"]


def test_multiproduct_shared_ancilla_segments_converge_to_exact_evolution():
    rng = np.random.default_rng(2026)
    initial_state = rng.normal(size=4) + 1j * rng.normal(size=4)
    initial_state /= np.linalg.norm(initial_state)
    hamiltonian = transverse_field_ising(2, coupling=1.0, field=0.7)
    segment_counts = (1, 2, 4)
    circuits = [
        build_multiproduct_circuit(hamiltonian, 0.6, m=2, segments=segments)
        for segments in segment_counts
    ]
    results = [
        compare_with_exact(
            hamiltonian,
            0.6,
            method="multiproduct",
            initial_state=initial_state,
            reps=segments,
            mpf_m=2,
        )
        for segments in segment_counts
    ]
    errors = [result["state_error"] for result in results]

    assert errors[0] > errors[1] > errors[2]
    assert all(result["success_probability"] > 0.999 for result in results)
    assert len({circuit.num_qubits for circuit in circuits}) == 1
    assert [len(circuit.data) for circuit in circuits] == list(segment_counts)


def test_named_multiproduct_gates_remain_transpilable_for_resource_counting():
    circuit = build_multiproduct_circuit(
        transverse_field_ising(1),
        0.1,
        m=2,
        amplitude_amplification=False,
    )
    estimate = count_circuit_resources(
        circuit,
        total_synthesis_error=1e-4,
        optimization_level=0,
    )

    assert estimate.cnot_count > 0
    assert estimate.rotation_count > 0


def test_multiproduct_supports_negative_and_zero_time():
    hamiltonian = transverse_field_ising(1)
    forward = build_multiproduct_circuit(hamiltonian, 0.1, segments=2)
    backward = build_multiproduct_circuit(hamiltonian, -0.1, segments=2)
    identity = build_multiproduct_circuit(hamiltonian, 0.0, segments=2)

    assert np.allclose(
        _zero_ancilla_block(backward, 1),
        _zero_ancilla_block(forward, 1).conj().T,
        atol=1e-12,
    )
    assert identity.num_qubits == 1
    assert np.allclose(Operator(identity).data, np.eye(2))
    assert identity.metadata["trotter_step_queries"] == 0


@pytest.mark.parametrize("segments", [0, -1])
def test_multiproduct_rejects_nonpositive_segments(segments):
    with pytest.raises(ValueError, match="segments"):
        build_multiproduct_circuit(transverse_field_ising(1), 0.1, segments=segments)


@pytest.mark.parametrize("segments", [1.5, True])
def test_multiproduct_rejects_noninteger_segments(segments):
    with pytest.raises(TypeError, match="segments"):
        build_multiproduct_circuit(transverse_field_ising(1), 0.1, segments=segments)


@pytest.mark.parametrize("time", [np.inf, -np.inf, np.nan])
def test_multiproduct_rejects_nonfinite_time(time):
    with pytest.raises(ValueError, match="time"):
        build_multiproduct_circuit(transverse_field_ising(1), time)


def test_multiproduct_rejects_unamplified_multiple_segments():
    with pytest.raises(ValueError, match="segments=1"):
        build_multiproduct_circuit(
            transverse_field_ising(1),
            0.1,
            segments=2,
            amplitude_amplification=False,
        )


def test_multiproduct_builder_never_forms_dense_hamiltonian(monkeypatch):
    from hamiltonian_resources import PauliHamiltonian

    def fail_if_called(self):
        raise AssertionError("dense matrix construction is forbidden in circuit builders")

    monkeypatch.setattr(PauliHamiltonian, "matrix", fail_if_called)
    circuit = build_multiproduct_circuit(transverse_field_ising(2), 0.05, m=3)
    assert circuit.metadata["registers"]["system"] == 2


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
    frame = run_benchmark(
        BenchmarkConfig(
            hamiltonian=HamiltonianSpec(
                model="tfim-test", parameters={}, factory=transverse_field_ising
            ),
            system_sizes=[2, 4],
            target_errors=[1e-2],
            time=TimeScaling("fixed", 0.1),
            fixed_target_error=1e-2,
            methods=[MultiproductMethod(3)],
        ),
        sweeps="system-size",
    )
    assert len(frame) == 2
    assert set(frame["counting_mode"]) == {"analytical-model"}
    assert (frame["t_count"] > 0).all()
    assert frame["nominal_success_probability"].isna().all()


def test_multiproduct_consumers_use_m_parameter():
    hamiltonian = transverse_field_ising(2)
    config = _EvaluationConfig(time=0.1, target_error=1e-2, mpf_m=3)
    parameters = choose_parameters(hamiltonian, config)
    estimate = estimate_resources_analytically(hamiltonian, config, "multiproduct")
    result = compare_with_exact(
        hamiltonian, 0.1, method="multiproduct", reps=1, mpf_m=3
    )
    # Mirrors the analytical model: per segment one robust-OAA round applies
    # SELECT three times (doubled rotations through the branch flag) and
    # PREPARE six times over ceil(log2(m + 2)) branch qubits.
    branch_bits = 3  # m + 2 = 5 branches, including identity padding
    select_rotations = (2 * hamiltonian.term_count - 1) * sum(NEW_MPF_EXPONENTS[3])
    expected_per_segment = 3 * 2 * select_rotations + 6 * (2**branch_bits - 1)
    assert estimate.rotation_count == parameters["mpf_segments"] * expected_per_segment
    assert estimate.toffoli_count > 0
    assert result["fidelity"] > 0.99


def test_multiproduct_schedule_propagates_through_consumers():
    hamiltonian = transverse_field_ising(1)
    new_config = _EvaluationConfig(mpf_m=3)
    legacy_config = _EvaluationConfig(mpf_m=3, mpf_schedule="legacy")
    new_estimate = estimate_resources_analytically(
        hamiltonian,
        new_config,
        "multiproduct",
    )
    legacy_estimate = estimate_resources_analytically(
        hamiltonian,
        legacy_config,
        "multiproduct",
    )
    result = compare_with_exact(
        hamiltonian,
        0.1,
        method="multiproduct",
        mpf_m=3,
        mpf_schedule="legacy",
    )

    assert new_estimate.rotation_count < legacy_estimate.rotation_count
    assert result["fidelity"] > 0.99


def test_benchmark_config_rejects_unsupported_m():
    with pytest.raises(ValueError, match="between 2 and 15"):
        _EvaluationConfig(mpf_m=16)


def test_suzuki_commutator_bounds_vanish_for_commuting_terms():
    from hamiltonian_resources import PauliHamiltonian, suzuki_commutator_bounds

    hamiltonian = PauliHamiltonian.from_terms(
        2, [("ZI", 0.7), ("IZ", -0.2), ("ZZ", 0.4)]
    )
    w1, w2 = suzuki_commutator_bounds(hamiltonian)
    parameters = choose_parameters(
        hamiltonian,
        _EvaluationConfig(
            time=5.0,
            target_error=1e-6,
            mpf_error_method="legacy-w2-proxy",
        ),
    )

    assert w1 == 0.0 and w2 == 0.0
    assert parameters["trotter_reps"] == 1
    assert parameters["mpf_segments"] == 1


def test_suzuki_commutator_bounds_scale_linearly_for_tfim():
    from hamiltonian_resources import suzuki_commutator_bounds

    _, w2_small = suzuki_commutator_bounds(transverse_field_ising(8, field=0.7))
    _, w2_large = suzuki_commutator_bounds(transverse_field_ising(16, field=0.7))

    # Nearest-neighbour chains have extensive (O(n)) commutator prefactors,
    # unlike the O(n^3) growth of the loose (alpha*t)^3 proxy.
    assert 1.5 < w2_large / w2_small < 2.5


def test_chosen_trotter_reps_meet_the_error_budget():
    hamiltonian = transverse_field_ising(3, field=0.7)
    config = _EvaluationConfig(time=0.5, target_error=1e-3)
    reps = choose_parameters(hamiltonian, config)["trotter_reps"]
    result = compare_with_exact(
        hamiltonian, 0.5, method="trotter", reps=reps, trotter_order=2
    )

    algorithm_budget = config.target_error * (1 - config.synthesis_error_fraction)
    assert result["state_error"] <= algorithm_budget


def test_chosen_mpf_segments_meet_the_error_budget():
    hamiltonian = transverse_field_ising(3, field=0.7)
    config = _EvaluationConfig(time=0.5, target_error=1e-3, mpf_m=2)
    segments = choose_parameters(hamiltonian, config)["mpf_segments"]
    result = compare_with_exact(
        hamiltonian, 0.5, method="multiproduct", reps=segments, mpf_m=2
    )

    algorithm_budget = config.target_error * (1 - config.synthesis_error_fraction)
    assert result["state_error"] <= algorithm_budget


def test_analytical_qsvt_model_includes_amplification_and_toffolis():
    hamiltonian = transverse_field_ising(2, field=0.7)
    config = _EvaluationConfig(time=0.5, target_error=1e-3)
    degree = choose_parameters(hamiltonian, config)["qsvt_degree"]
    estimate = estimate_resources_analytically(hamiltonian, config, "qsvt")

    index_bits = 2
    queries = 3 * ((degree - 1) + degree)
    phase_slots = 3 * (degree + degree + 1)
    expected_rotations = 2 * queries * (2**index_bits - 1) + 2 * phase_slots
    assert estimate.rotation_count == expected_rotations
    assert estimate.toffoli_count > queries  # SELECT control ladders dominate
    assert estimate.t_count > estimate.rotation_count


def test_new_and_legacy_schedules_have_comparable_accuracy():
    hamiltonian = transverse_field_ising(2, field=0.7)
    errors = {
        schedule: compare_with_exact(
            hamiltonian,
            0.3,
            method="multiproduct",
            reps=1,
            mpf_m=3,
            mpf_schedule=schedule,
        )["state_error"]
        for schedule in ("new", "legacy")
    }

    # The cheaper 'new' exponent table may lose a small constant factor in
    # accuracy but must stay within one order of magnitude of 'legacy'.
    assert errors["new"] < 10 * errors["legacy"]
    assert errors["new"] < 1e-5


def test_qsvt_degree_handles_large_alpha_time_without_overflow():
    from hamiltonian_resources import estimate_qsvt_degree

    degree = estimate_qsvt_degree(1.6e5, 1e-8)
    assert degree % 2 == 1
    assert degree > 1.6e5  # Jacobi--Anger needs q > alpha*t before decay


def test_method_specific_parameter_selection_is_isolated(monkeypatch):
    import hamiltonian_resources.benchmark as benchmark_module

    hamiltonian = transverse_field_ising(3)
    config = _EvaluationConfig(time=3.0, target_error=1e-3, trotter_order=2)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("QSVT degree should not be evaluated for Trotter")

    monkeypatch.setattr(benchmark_module, "estimate_qsvt_degree", fail_if_called)
    parameters = choose_parameters(hamiltonian, config, "trotter")

    assert set(parameters) == {"trotter_reps"}
