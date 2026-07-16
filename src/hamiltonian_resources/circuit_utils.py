"""Reusable circuit primitives for LCU block encodings."""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import PauliGate, PhaseGate, StatePreparation

from .hamiltonians import PauliHamiltonian


def index_width(item_count: int) -> int:
    return max(1, math.ceil(math.log2(item_count)))


def state_preparation(amplitudes: np.ndarray, name: str = "PREPARE") -> Gate:
    """Return A with A|0> = amplitudes (zero-padding to a power of two)."""
    width = index_width(len(amplitudes))
    padded = np.zeros(2**width, dtype=complex)
    padded[: len(amplitudes)] = amplitudes
    if not np.isclose(np.linalg.norm(padded), 1.0):
        raise ValueError("state-preparation amplitudes must be normalized")
    return StatePreparation(padded, label=name)


def append_phase_on_index(circuit: QuantumCircuit, index, value: int, angle: float) -> None:
    """Apply exp(i angle) only to one index-register computational state."""
    width = len(index)
    zero_positions = [bit for bit in range(width) if not (value >> bit) & 1]
    for bit in zero_positions:
        circuit.x(index[bit])
    if width == 1:
        circuit.p(angle, index[0])
    else:
        circuit.append(PhaseGate(angle).control(width - 1), [*index[:-1], index[-1]])
    for bit in zero_positions:
        circuit.x(index[bit])


def append_zero_projector_phase(circuit: QuantumCircuit, ancillas, angle: float) -> None:
    """Apply exp(i*angle) on ancillas |0...0>, up to no other phases."""
    for qubit in ancillas:
        circuit.x(qubit)
    if len(ancillas) == 1:
        circuit.p(angle, ancillas[0])
    else:
        circuit.append(PhaseGate(angle).control(len(ancillas) - 1), list(ancillas))
    for qubit in ancillas:
        circuit.x(qubit)


def pauli_lcu_oracles(
    hamiltonian: PauliHamiltonian,
) -> tuple[Gate, QuantumCircuit]:
    """Construct PREPARE and SELECT for H/alpha = <G|SELECT|G>.

    SELECT includes each coefficient's sign.  No dense Hamiltonian matrix is
    formed; only controlled Pauli gates and state preparation are emitted.
    """
    alpha = hamiltonian.alpha
    if alpha == 0:
        raise ValueError("the Hamiltonian L1 norm must be nonzero")
    amplitudes = np.sqrt([abs(c) / alpha for _, c in hamiltonian.terms])
    prepare = state_preparation(amplitudes)

    width = prepare.num_qubits
    index = QuantumRegister(width, "index")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    select = QuantumCircuit(index, system, name="SELECT")
    for j, (label, coefficient) in enumerate(hamiltonian.terms):
        controlled_pauli = PauliGate(label).control(width, ctrl_state=j)
        select.append(controlled_pauli, [*index, *system])
        if coefficient < 0:
            append_phase_on_index(select, index, j, np.pi)
    return prepare, select


def build_block_encoding(hamiltonian: PauliHamiltonian) -> QuantumCircuit:
    """Circuit U_H satisfying <0|U_H|0> = H/alpha."""
    prepare, select = pauli_lcu_oracles(hamiltonian)
    anc = QuantumRegister(prepare.num_qubits, "index")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(anc, system, name="U_H")
    circuit.append(prepare, anc)
    circuit.append(select.to_gate(label="SELECT"), [*anc, *system])
    circuit.append(prepare.inverse(), anc)
    return circuit
