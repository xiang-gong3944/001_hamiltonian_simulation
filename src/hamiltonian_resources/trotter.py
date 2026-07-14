"""Circuit-level Lie/Suzuki product-formula simulation."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from .hamiltonians import PauliHamiltonian


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

