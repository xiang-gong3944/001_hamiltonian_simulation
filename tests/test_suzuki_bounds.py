import numpy as np
import pytest
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import expm

from hamiltonian_resources import (
    BenchmarkConfig,
    PauliHamiltonian,
    benchmark_scaling,
    build_trotter_circuit,
    choose_parameters,
    estimate_suzuki_error,
    heisenberg_chain,
    transverse_field_ising,
)
from hamiltonian_resources.trotter import (
    _resolve_suzuki_specification,
    _suzuki_group_factors,
    _suzuki_term_occurrences,
)


def _operator_error(hamiltonian, time, reps, order, partition="auto"):
    circuit = build_trotter_circuit(
        hamiltonian,
        time,
        reps,
        order,
        partition=partition,
    )
    exact = expm(-1j * time * hamiltonian.matrix())
    return float(np.linalg.norm(Operator(circuit).data - exact, 2))


@pytest.mark.parametrize("order", [1, 2])
def test_auto_preserves_low_order_individual_formula(order):
    hamiltonian = transverse_field_ising(3, field=0.7)
    automatic = build_trotter_circuit(hamiltonian, 0.2, 2, order)
    individual = build_trotter_circuit(
        hamiltonian,
        0.2,
        2,
        order,
        partition="individual",
    )

    assert automatic.metadata["trotter_partition"] == "individual"
    assert np.allclose(Operator(automatic).data, Operator(individual).data, atol=1e-12)


def test_auto_commuting_groups_are_exact_partitions_of_supported_models():
    cases = [
        (transverse_field_ising(4, field=0.7), 2),
        (heisenberg_chain(4, field_z=0.3), 3),
    ]
    for hamiltonian, expected_groups in cases:
        specification = _resolve_suzuki_specification(hamiltonian, 4, "auto")

        assert specification.partition == "commuting"
        assert len(specification.groups) == expected_groups
        assert np.allclose(
            sum((group.to_matrix() for group in specification.groups)),
            hamiltonian.matrix(),
            atol=1e-12,
        )
        for group in specification.groups:
            assert all(
                group.paulis[left].commutes(group.paulis[right])
                for left in range(group.size)
                for right in range(left)
            )


@pytest.mark.parametrize("order", [2, 4, 6])
def test_internal_suzuki_recursion_matches_qiskit_expand(order):
    groups = [
        SparsePauliOp.from_list([("X", 1.0)]),
        SparsePauliOp.from_list([("Z", 1.0)]),
    ]
    synthesis = SuzukiTrotter(order=order, reps=1, preserve_order=True)
    gate = PauliEvolutionGate(groups, time=1.0, synthesis=synthesis)
    expanded = synthesis.expand(gate)
    labels = {"X": 0, "Z": 1}
    qiskit_factors = tuple((labels[label], float(angle) / 2) for label, _, angle in expanded)

    assert len(qiskit_factors) == len(_suzuki_group_factors(2, order))
    assert [group for group, _ in qiskit_factors] == [
        group for group, _ in _suzuki_group_factors(2, order)
    ]
    assert np.allclose(
        [coefficient for _, coefficient in qiskit_factors],
        [coefficient for _, coefficient in _suzuki_group_factors(2, order)],
        atol=1e-15,
    )


@pytest.mark.parametrize("order", [4, 6])
@pytest.mark.parametrize("reps", [1, 2])
@pytest.mark.parametrize(
    "hamiltonian",
    [
        transverse_field_ising(3, field=0.7),
        heisenberg_chain(3, field_z=0.3),
        PauliHamiltonian.from_terms(
            2,
            [("XI", 0.31), ("YZ", -0.47), ("ZZ", 0.23), ("IX", -0.19)],
            "fixed-random-paulis",
        ),
    ],
)
def test_rigorous_higher_order_bound_dominates_exact_operator_error(
    hamiltonian, order, reps
):
    time = 0.2
    estimate = estimate_suzuki_error(hamiltonian, time, reps, order)
    actual_error = _operator_error(hamiltonian, time, reps, order)

    assert estimate.rigorous
    assert estimate.method == "schubert-mendl-commutator"
    assert actual_error <= estimate.error * (1 + 1e-12) + 1e-12


@pytest.mark.parametrize("order", [4, 6])
def test_commuting_higher_order_formula_has_zero_bound(order):
    hamiltonian = PauliHamiltonian.from_terms(
        2,
        [("ZI", 0.7), ("IZ", -0.2), ("ZZ", 0.4)],
    )
    estimate = estimate_suzuki_error(hamiltonian, 5.0, 1, order)

    assert estimate.error == 0.0
    assert estimate.rigorous
    assert estimate.method == "commuting-exact"
    assert _operator_error(hamiltonian, 0.5, 1, order) < 1e-12


@pytest.mark.parametrize("order", [4, 6])
def test_chosen_higher_order_reps_meet_operator_error_budget(order):
    hamiltonian = transverse_field_ising(3, field=0.7)
    config = BenchmarkConfig(time=0.2, target_error=1e-3, trotter_order=order)
    reps = choose_parameters(hamiltonian, config)["trotter_reps"]
    budget = config.target_error * (1 - config.synthesis_error_fraction)

    assert estimate_suzuki_error(hamiltonian, config.time, reps, order).error <= budget
    assert _operator_error(hamiltonian, config.time, reps, order) <= budget


def test_higher_order_fallbacks_are_explicitly_nonrigorous():
    hamiltonian = PauliHamiltonian.from_terms(
        3,
        [
            ("XII", 0.1),
            ("YII", 0.2),
            ("IXI", 0.3),
            ("IYI", 0.4),
            ("IIX", 0.5),
            ("IIY", 0.6),
        ],
    )
    too_many_individual = estimate_suzuki_error(
        hamiltonian,
        0.2,
        order=4,
        partition="individual",
    )
    unsupported_order = estimate_suzuki_error(hamiltonian, 0.2, order=8)

    for estimate in (too_many_individual, unsupported_order):
        assert not estimate.rigorous
        assert estimate.method == "alpha-proxy"
        assert estimate.error > 0


@pytest.mark.parametrize("order", [4, 6])
@pytest.mark.parametrize(
    "hamiltonian",
    [transverse_field_ising(3, field=0.7), heisenberg_chain(3, field_z=0.3)],
)
def test_analytical_occurrences_match_qiskit_grouped_expansion(hamiltonian, order):
    specification = _resolve_suzuki_specification(hamiltonian, order, "auto")
    synthesis = SuzukiTrotter(order=order, reps=1, preserve_order=True)
    gate = PauliEvolutionGate(list(specification.groups), time=1.0, synthesis=synthesis)

    assert _suzuki_term_occurrences(hamiltonian, 1, order) == len(
        synthesis.expand(gate)
    )


def test_benchmark_reports_higher_order_bound_provenance():
    frame = benchmark_scaling(
        [3],
        transverse_field_ising,
        BenchmarkConfig(time=0.1, target_error=1e-2, trotter_order=4),
    )
    trotter = frame[frame["algorithm"] == "trotter"].iloc[0]

    assert trotter["trotter_partition"] == "commuting"
    assert trotter["trotter_group_count"] == 2
    assert trotter["trotter_error_method"] == "schubert-mendl-commutator"
    assert bool(trotter["trotter_error_rigorous"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trotter_order", 3, "trotter_order"),
        ("trotter_partition", "bad", "trotter_partition"),
    ],
)
def test_benchmark_config_rejects_invalid_trotter_settings(field, value, message):
    with pytest.raises(ValueError, match=message):
        BenchmarkConfig(**{field: value})


def test_suzuki_error_estimate_rejects_invalid_inputs():
    hamiltonian = transverse_field_ising(2)
    with pytest.raises(ValueError, match="reps"):
        estimate_suzuki_error(hamiltonian, 0.2, reps=0, order=4)
    with pytest.raises(ValueError, match="time"):
        estimate_suzuki_error(hamiltonian, np.inf, order=4)
    with pytest.raises(ValueError, match="partition"):
        estimate_suzuki_error(hamiltonian, 0.2, order=4, partition="bad")
