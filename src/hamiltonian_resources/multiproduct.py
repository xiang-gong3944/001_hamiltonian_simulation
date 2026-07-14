"""Well-conditioned multiproduct-formula (MPF) LCU circuits.

Implements the coherent LCU construction based on Childs & Wiebe-style
multiproduct formulas and the well-conditioned schedules of Low, Kliuchnikov,
and Wiebe, arXiv:1907.11679.  Postselecting the branch register on zero yields
the desired weighted sum divided by its coefficient 1-norm.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from .circuit_utils import append_phase_on_index, state_preparation
from .hamiltonians import PauliHamiltonian
from .trotter import build_trotter_circuit


def multiproduct_coefficients(exponents: Sequence[int]) -> np.ndarray:
    """Solve the order conditions for sum_j a_j S_2^(k_j).

    For m distinct positive integers k_j, cancellation of powers
    k_j^-2, ..., k_j^-2(m-1) gives formal order 2m.
    """
    ks = np.asarray(exponents, dtype=float)
    if len(ks) < 1 or np.any(ks <= 0) or len(set(ks.tolist())) != len(ks):
        raise ValueError("exponents must be distinct positive integers")
    matrix = np.vstack([np.ones(len(ks)), *[ks ** (-2 * q) for q in range(1, len(ks))]])
    rhs = np.zeros(len(ks))
    rhs[0] = 1.0
    return np.linalg.solve(matrix, rhs)


def build_multiproduct_circuit(
    hamiltonian: PauliHamiltonian,
    time: float,
    exponents: Sequence[int] = (1, 2),
    segments: int = 1,
) -> QuantumCircuit:
    """Build a coherent MPF LCU circuit from second-order Suzuki branches.

    The returned circuit acts on branch ancillas followed by system qubits.
    Its zero-ancilla block is M/s where M=sum_j a_j S_2(t/k_j)^k_j and
    s=sum_j |a_j|. With ``segments>1`` each branch uses k_j*segments steps.
    """
    if segments < 1:
        raise ValueError("segments must be positive")
    ks = tuple(int(k) for k in exponents)
    coefficients = multiproduct_coefficients(ks)
    scale = float(np.sum(np.abs(coefficients)))
    prepare = state_preparation(np.sqrt(np.abs(coefficients) / scale), name="PREPARE_MPF")

    branch = QuantumRegister(prepare.num_qubits, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(branch, system, name=f"MPF-{2 * len(ks)}")
    circuit.append(prepare, branch)
    for j, (k, coefficient) in enumerate(zip(ks, coefficients, strict=True)):
        approximation = build_trotter_circuit(
            hamiltonian, time, reps=k * segments, order=2
        ).to_gate(label=f"S2^{k * segments}")
        circuit.append(
            approximation.control(len(branch), ctrl_state=j), [*branch, *system]
        )
        if coefficient < 0:
            append_phase_on_index(circuit, branch, j, np.pi)
    circuit.append(prepare.inverse(), branch)
    circuit.metadata = {
        "algorithm": "multiproduct",
        "exponents": ks,
        "coefficients": coefficients.tolist(),
        "lcu_scale": scale,
        "formal_order": 2 * len(ks),
        "postselection": "measure branch register as all-zero",
    }
    return circuit

