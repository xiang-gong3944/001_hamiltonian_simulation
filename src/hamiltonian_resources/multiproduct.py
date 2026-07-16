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


_OAA_NORMALIZATION = 2.0


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


def _build_multiproduct_step_lcu(
    hamiltonian: PauliHamiltonian,
    step_time: float,
    m: int,
) -> QuantumCircuit:
    """Build one normalized coherent MPF step before amplification.

    The all-zero branch block is exactly
    ``sum_j a_j S_2(step_time / k_j) ** k_j / 2``. Two cancelling identity
    branches pad the coefficient 1-norm to two, which is the normalization
    required by one round of three-step robust OAA.
    """
    if not np.isfinite(step_time):
        raise ValueError("step_time must be finite")

    exponents = optimal_mpf_exponents(m)
    coefficients = multiproduct_coefficients(m)
    coefficient_l1 = float(np.sum(np.abs(coefficients)))
    if coefficient_l1 >= _OAA_NORMALIZATION:
        raise ValueError("the MPF coefficient 1-norm must be less than 2")

    padding_weight = _OAA_NORMALIZATION - coefficient_l1
    branch_weights = np.concatenate(
        (coefficients, [padding_weight / 2, -padding_weight / 2])
    )
    prepare = state_preparation(
        np.sqrt(np.abs(branch_weights) / _OAA_NORMALIZATION),
        name="PREPARE_MPF",
    )

    branch = QuantumRegister(prepare.num_qubits, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(branch, system, name=f"MPF_step_{2 * m}")
    circuit.append(prepare, branch)
    for j, weight in enumerate(branch_weights):
        if j < len(exponents):
            exponent = exponents[j]
            approximation = build_trotter_circuit(
                hamiltonian,
                step_time,
                reps=exponent,
                order=2,
            ).to_gate(label=f"S2({step_time:g}/{exponent})^{exponent}")
            circuit.append(
                approximation.control(len(branch), ctrl_state=j),
                [*branch, *system],
            )
        if weight < 0:
            append_phase_on_index(circuit, branch, j, np.pi)
    circuit.append(prepare.inverse(), branch)
    circuit.metadata = {
        "algorithm": "multiproduct-step-lcu",
        "m": int(m),
        "exponents": exponents,
        "coefficients": coefficients.tolist(),
        "coefficient_l1_norm": coefficient_l1,
        "padding_weight": padding_weight,
        "lcu_normalization": _OAA_NORMALIZATION,
        "formal_order": 2 * int(m),
        "step_time": float(step_time),
        "amplitude_amplification": False,
        "good_subspace": "branch register all-zero",
        "trotter_step_queries": int(sum(exponents)),
    }
    return circuit


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
