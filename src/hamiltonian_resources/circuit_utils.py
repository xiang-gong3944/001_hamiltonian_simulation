"""Named circuit primitives for LCU block encodings and robust OAA."""

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
    """Return a named gate ``A`` with ``A|0> = amplitudes``.

    The amplitudes are zero-padded to a power of two. Returning a gate rather
    than mutating a parent circuit keeps PREPARE visible as a logical resource
    boundary while retaining a complete definition for later transpilation.
    """
    width = index_width(len(amplitudes))
    padded = np.zeros(2**width, dtype=complex)
    padded[: len(amplitudes)] = amplitudes
    if not np.isclose(np.linalg.norm(padded), 1.0):
        raise ValueError("state-preparation amplitudes must be normalized")
    return StatePreparation(padded, label=name)


def index_state_phase_gate(
    width: int,
    value: int,
    angle: float,
    *,
    name: str = "INDEX_PHASE",
) -> Gate:
    """Return a gate applying ``exp(i*angle)`` only to ``|value>``."""
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width < 1:
        raise ValueError("width must be positive")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    if not 0 <= value < 2**width:
        raise ValueError("value must fit in the index register")

    index = QuantumRegister(width, "index")
    circuit = QuantumCircuit(index, name=name)
    zero_positions = [bit for bit in range(width) if not (value >> bit) & 1]
    for bit in zero_positions:
        circuit.x(index[bit])
    if width == 1:
        circuit.p(angle, index[0])
    else:
        circuit.append(PhaseGate(angle).control(width - 1), [*index[:-1], index[-1]])
    for bit in zero_positions:
        circuit.x(index[bit])
    return circuit.to_gate(label=name)


def zero_projector_phase_gate(
    width: int,
    angle: float,
    *,
    name: str = "ZERO_PROJECTOR_PHASE",
) -> Gate:
    """Return ``exp(i*angle |0...0><0...0|)`` as a named gate."""
    return index_state_phase_gate(width, 0, angle, name=name)


def build_three_step_oaa(
    base: QuantumCircuit,
    system_qubits: int,
    *,
    name: str = "robust_oaa",
    gate_label: str = "U/2",
) -> QuantumCircuit:
    """Return one robust oblivious-amplitude-amplification round.

    ``base`` must place all good-subspace ancillas before the system register,
    with the all-zero ancilla state defining the desired block ``B``. The
    returned ``-U R U^dagger R U`` construction has exact good block
    ``3B - 4 B B^dagger B``. For ``B=A/2`` this is close to ``A`` when ``A``
    is close to unitary.
    """
    if isinstance(system_qubits, bool) or not isinstance(system_qubits, int):
        raise TypeError("system_qubits must be an integer")
    if system_qubits < 1 or system_qubits >= base.num_qubits:
        raise ValueError("OAA requires system qubits and at least one ancilla")

    ancilla_count = base.num_qubits - system_qubits
    ancillas = QuantumRegister(ancilla_count, "ancilla")
    system = QuantumRegister(system_qubits, "system")
    amplified = QuantumCircuit(ancillas, system, name=name)
    base_gate = base.to_gate(label=gate_label)
    targets = [*ancillas, *system]

    reflection = zero_projector_phase_gate(
        ancilla_count,
        np.pi,
        name="GOOD_REFLECTION",
    )
    amplified.append(base_gate, targets)
    amplified.append(reflection, ancillas)
    amplified.append(base_gate.inverse(), targets)
    amplified.append(reflection, ancillas)
    amplified.append(base_gate, targets)
    # U R U^dagger R U has block 4a^3 - 3a; correct its known sign.
    amplified.global_phase += np.pi
    return amplified


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
            phase = index_state_phase_gate(width, j, np.pi, name="COEFFICIENT_SIGN")
            select.append(phase, index)
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
