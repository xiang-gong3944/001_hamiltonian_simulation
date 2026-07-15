"""Well-conditioned multiproduct-formula (MPF) LCU circuits.

Implements the coherent LCU construction based on Childs & Wiebe-style
multiproduct formulas and the well-conditioned schedules of Low, Kliuchnikov,
and Wiebe, arXiv:1907.11679.  Postselecting the branch register on zero yields
the desired weighted sum divided by its coefficient 1-norm.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from .circuit_utils import append_phase_on_index, state_preparation
from .hamiltonians import PauliHamiltonian
from .trotter import build_trotter_circuit


_OPTIMAL_MPF_EXPONENTS: dict[int, tuple[int, ...]] = {
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


def optimal_mpf_exponents(m: int) -> tuple[int, ...]:
    """Return the registered well-conditioned MPF schedule with ``m`` terms."""
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError("m must be an integer")
    try:
        return _OPTIMAL_MPF_EXPONENTS[int(m)]
    except KeyError as error:
        raise ValueError("m must lie between 2 and 15") from error


def multiproduct_coefficients(m: int) -> np.ndarray:
    """Return the coefficients for the registered ``m``-term MPF.

    The direct product formula avoids the ill-conditioned Vandermonde solve
    while satisfying the cancellation conditions through formal order ``2m``.
    """
    ks = optimal_mpf_exponents(m)
    coefficients = np.ones(len(ks), dtype=float)
    for j, k_j in enumerate(ks):
        k_j_squared = k_j**2
        for q, k_q in enumerate(ks):
            if q != j:
                coefficients[j] *= k_j_squared / (k_j_squared - k_q**2)
    return coefficients


def build_multiproduct_circuit(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int = 2,
    segments: int = 1,
) -> QuantumCircuit:
    """Build a coherent MPF LCU circuit from second-order Suzuki branches.

    The returned circuit acts on branch ancillas followed by system qubits.
    Its zero-ancilla block is M/s where M=sum_j a_j S_2(t/k_j)^k_j and
    s=sum_j |a_j|. With ``segments>1`` each branch uses k_j*segments steps.
    """
    if isinstance(segments, bool) or not isinstance(segments, Integral):
        raise TypeError("segments must be an integer")
    if segments < 1:
        raise ValueError("segments must be positive")
    ks = optimal_mpf_exponents(m)
    coefficients = multiproduct_coefficients(m)
    scale = float(np.sum(np.abs(coefficients)))
    prepare = state_preparation(np.sqrt(np.abs(coefficients) / scale), name="PREPARE_MPF")

    branch = QuantumRegister(prepare.num_qubits, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(branch, system, name=f"MPF-{2 * m}")
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
        "m": int(m),
        "exponents": ks,
        "segments": int(segments),
        "coefficients": coefficients.tolist(),
        "lcu_scale": scale,
        "formal_order": 2 * int(m),
        "postselection": "measure branch register as all-zero",
    }
    return circuit
