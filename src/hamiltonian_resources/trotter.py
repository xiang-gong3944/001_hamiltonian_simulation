"""Circuit-level Lie/Suzuki product-formula simulation and error bounds."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from .hamiltonians import PauliHamiltonian


def _pauli_l1(operator: SparsePauliOp) -> float:
    return float(np.sum(np.abs(operator.simplify().coeffs)))


def _commutator(a: SparsePauliOp, b: SparsePauliOp) -> SparsePauliOp:
    return (a @ b - b @ a).simplify()


@lru_cache(maxsize=None)
def suzuki_commutator_bounds(hamiltonian: PauliHamiltonian) -> tuple[float, float]:
    """Return prefactors (W1, W2) of the commutator Trotter error bounds.

    For the term ordering used by ``build_trotter_circuit``, Childs, Su, Tran,
    Wiebe, and Zhu (PRX 11, 011020 (2021), Prop. 9/10) give

        ||S1(d) - exp(-i d H)|| <= W1 d^2,   W1 = (1/2) sum_g ||[T_g, H_g]||
        ||S2(d) - exp(-i d H)|| <= W2 d^3,
        W2 = sum_g ( ||[T_g, [T_g, H_g]]||/12 + ||[H_g, [H_g, T_g]]||/24 )

    with T_g the sum of all terms after H_g.  Spectral norms are upper-bounded
    by Pauli coefficient 1-norms of the exactly computed nested commutators, so
    both prefactors are rigorous and scale with the commutation structure
    (O(n) for local chains) instead of the loose 1-norm power alpha^(p+1).
    """
    operator = hamiltonian.to_sparse_pauli_op()
    terms = [
        SparsePauliOp(pauli, np.array([coeff]))
        for pauli, coeff in zip(operator.paulis, operator.coeffs)
    ]
    w1 = 0.0
    w2 = 0.0
    suffix = terms[-1]
    for gamma in range(len(terms) - 2, -1, -1):
        head = terms[gamma]
        inner = _commutator(suffix, head)
        if inner.size:
            w1 += _pauli_l1(inner) / 2
            w2 += _pauli_l1(_commutator(suffix, inner)) / 12
            w2 += _pauli_l1(_commutator(head, inner)) / 24
        suffix = (suffix + head).simplify()
    return w1, w2


def build_trotter_circuit(
    hamiltonian: PauliHamiltonian,
    time: float,
    reps: int,
    order: int = 2,
    *,
    insert_barriers: bool = False,
) -> QuantumCircuit:
    """Build exp(-i H time) as an actual product-formula circuit.

    This uses Pauli rotations; it deliberately does not exponentiate a dense
    matrix.  ``order=1`` selects Lie-Trotter. Higher orders must be even.
    """
    if reps < 1:
        raise ValueError("reps must be positive")
    if order == 1:
        synthesis = LieTrotter(reps=reps, insert_barriers=insert_barriers)
    elif order >= 2 and order % 2 == 0:
        synthesis = SuzukiTrotter(order=order, reps=reps, insert_barriers=insert_barriers)
    else:
        raise ValueError("order must be 1 or a positive even integer")
    gate = PauliEvolutionGate(
        hamiltonian.to_sparse_pauli_op(), time=float(time), synthesis=synthesis
    )
    circuit = QuantumCircuit(hamiltonian.num_qubits, name=f"Suzuki-{order}")
    circuit.append(gate, circuit.qubits)
    return circuit.decompose()

