"""Well-conditioned multiproduct-formula (MPF) LCU circuits.

Implements the coherent LCU construction based on Childs & Wiebe-style
multiproduct formulas and the well-conditioned schedules of Low, Kliuchnikov,
and Wiebe, arXiv:1907.11679. Each normalized LCU step is robustly amplified
before the same branch register is reused for the next simulation segment.
"""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

from .circuit_utils import (
    build_three_step_oaa,
    index_state_phase_gate,
    state_preparation,
)
from .hamiltonians import PauliHamiltonian
from .trotter import build_trotter_circuit


_OAA_NORMALIZATION = 2.0
MPFSchedule = Literal["new", "legacy"]


_NEW_MPF_EXPONENTS: dict[int, tuple[int, ...]] = {
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


_LEGACY_MPF_EXPONENTS: dict[int, tuple[int, ...]] = {
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


def optimal_mpf_exponents(
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> tuple[int, ...]:
    """Return one registered well-conditioned exponent schedule.

    ``new`` is the default table optimized for the three-query OAA construction:
    it reduces ``sum(k_j)`` while retaining coefficient 1-norm below two.
    ``legacy`` preserves the previous, more conservatively conditioned table.
    """
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError("m must be an integer")
    if schedule == "new":
        table = _NEW_MPF_EXPONENTS
    elif schedule == "legacy":
        table = _LEGACY_MPF_EXPONENTS
    else:
        raise ValueError("schedule must be 'new' or 'legacy'")
    try:
        return table[int(m)]
    except KeyError as error:
        raise ValueError("m must lie between 2 and 15") from error


def multiproduct_coefficients(
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> np.ndarray:
    """Return coefficients for one registered ``m``-term MPF schedule.

    The direct product formula avoids the ill-conditioned Vandermonde solve
    while satisfying the cancellation conditions through formal order ``2m``.
    """
    ks = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = np.ones(len(ks), dtype=float)
    for j, k_j in enumerate(ks):
        k_j_squared = k_j**2
        for q, k_q in enumerate(ks):
            if q != j:
                coefficients[j] *= k_j_squared / (k_j_squared - k_q**2)
    return coefficients


def _multiproduct_select_gate(
    hamiltonian: PauliHamiltonian,
    step_time: float,
    exponents: tuple[int, ...],
    branch_weights: np.ndarray,
    branch_width: int,
) -> Gate:
    """Return the named signed SELECT gate for one MPF LCU step.

    Physical MPF branches contain controlled second-order product formulas.
    The final two branches are cancelling positive and negative identities;
    any remaining computational states are unused identity branches.
    """
    branch = QuantumRegister(branch_width, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    select = QuantumCircuit(branch, system, name="SELECT_MPF")
    for j, weight in enumerate(branch_weights):
        if j < len(exponents):
            exponent = exponents[j]
            approximation = build_trotter_circuit(
                hamiltonian,
                step_time,
                reps=exponent,
                order=2,
            ).to_gate(label=f"S2({step_time:g}/{exponent})^{exponent}")
            select.append(
                approximation.control(branch_width, ctrl_state=j),
                [*branch, *system],
            )
        if weight < 0:
            sign = index_state_phase_gate(
                branch_width,
                j,
                np.pi,
                name="MPF_BRANCH_SIGN",
            )
            select.append(sign, branch)
    return select.to_gate(label="SELECT_MPF")


def _build_multiproduct_step_lcu(
    hamiltonian: PauliHamiltonian,
    step_time: float,
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> QuantumCircuit:
    """Build one normalized coherent MPF step before amplification.

    The all-zero branch block is exactly
    ``sum_j a_j S_2(step_time / k_j) ** k_j / 2``. Two cancelling identity
    branches pad the coefficient 1-norm to two, which is the normalization
    required by one round of three-step robust OAA.
    """
    if not np.isfinite(step_time):
        raise ValueError("step_time must be finite")

    exponents = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = multiproduct_coefficients(m, schedule=schedule)
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

    branch_width = prepare.num_qubits
    select = _multiproduct_select_gate(
        hamiltonian,
        step_time,
        exponents,
        branch_weights,
        branch_width,
    )
    branch = QuantumRegister(branch_width, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(branch, system, name=f"MPF_step_{2 * m}")
    circuit.append(prepare, branch)
    circuit.append(select, [*branch, *system])
    circuit.append(prepare.inverse(), branch)
    circuit.metadata = {
        "algorithm": "multiproduct-step-lcu",
        "m": int(m),
        "schedule": schedule,
        "exponents": exponents,
        "exponent_sum": int(sum(exponents)),
        "coefficients": coefficients.tolist(),
        "coefficient_l1_norm": coefficient_l1,
        "padding_weight": padding_weight,
        "lcu_normalization": _OAA_NORMALIZATION,
        "formal_order": 2 * int(m),
        "step_time": float(step_time),
        "amplitude_amplification": False,
        "good_subspace": "branch register all-zero",
        "trotter_step_queries": int(sum(exponents)),
        "logical_gate_counts": {
            "prepare": 2,
            "select": 1,
            "good_reflection": 0,
            "controlled_u2": int(sum(exponents)),
        },
    }
    return circuit


def build_multiproduct_circuit(
    hamiltonian: PauliHamiltonian,
    time: float,
    m: int = 2,
    segments: int = 1,
    *,
    schedule: MPFSchedule = "new",
    amplitude_amplification: bool = True,
) -> QuantumCircuit:
    """Repeat robustly amplified MPF-step unitaries on shared ancillas.

    Before amplification the good block is ``B=M(step_time)/2``. One robust
    OAA round transforms it exactly to ``3B - 4 B B^dagger B``, which is close
    to ``M`` to the extent that the MPF approximation is unitary. The amplified
    step unitary is then repeated on the same branch register; its final good
    block is therefore not asserted to equal ``M**segments`` exactly. The
    unamplified form is exposed only for validating a single LCU step.
    """
    if isinstance(segments, bool) or not isinstance(segments, Integral):
        raise TypeError("segments must be an integer")
    if segments < 1:
        raise ValueError("segments must be positive")
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if not amplitude_amplification and segments != 1:
        raise ValueError("unamplified MPF is only supported for segments=1")

    exponents = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = multiproduct_coefficients(m, schedule=schedule)
    coefficient_l1 = float(np.sum(np.abs(coefficients)))
    if coefficient_l1 >= _OAA_NORMALIZATION:
        raise ValueError("the MPF coefficient 1-norm must be less than 2")
    padding_weight = _OAA_NORMALIZATION - coefficient_l1
    step_time = float(time) / int(segments)
    oaa_factor = 3 if amplitude_amplification else 1
    per_segment_queries = oaa_factor * sum(exponents)
    logical_counts_per_segment = {
        "prepare": 2 * oaa_factor,
        "select": oaa_factor,
        "good_reflection": 2 if amplitude_amplification else 0,
        "controlled_u2": int(per_segment_queries),
    }

    metadata = {
        "algorithm": "multiproduct",
        "construction": "robust-oaa-segments"
        if amplitude_amplification
        else "single-unamplified-step",
        "m": int(m),
        "schedule": schedule,
        "exponents": exponents,
        "exponent_sum": int(sum(exponents)),
        "segments": int(segments),
        "step_time": step_time,
        "coefficients": coefficients.tolist(),
        "coefficient_l1_norm": coefficient_l1,
        "padding_weight": padding_weight,
        "lcu_normalization": _OAA_NORMALIZATION,
        "formal_order": 2 * int(m),
        "amplitude_amplification": bool(amplitude_amplification),
        "good_subspace": "branch register all-zero",
        "postselection": "measure branch register as all-zero",
        "trotter_step_queries_per_segment": int(per_segment_queries),
        "trotter_step_queries": int(segments * per_segment_queries),
        "base_lcu_uses_per_segment": oaa_factor,
        "logical_gate_counts_per_segment": logical_counts_per_segment,
        "logical_gate_counts": {
            key: int(segments * value)
            for key, value in logical_counts_per_segment.items()
        },
        "registers": {"branch": 0, "system": hamiltonian.num_qubits},
    }

    if time == 0:
        system = QuantumRegister(hamiltonian.num_qubits, "system")
        circuit = QuantumCircuit(system, name="MPF_identity")
        metadata.update(
            construction="zero-time-identity",
            good_subspace="no ancillas",
            postselection="none",
            trotter_step_queries_per_segment=0,
            trotter_step_queries=0,
            logical_gate_counts_per_segment={key: 0 for key in logical_counts_per_segment},
            logical_gate_counts={key: 0 for key in logical_counts_per_segment},
        )
        circuit.metadata = metadata
        return circuit

    base_step = _build_multiproduct_step_lcu(
        hamiltonian,
        step_time,
        m,
        schedule=schedule,
    )
    if amplitude_amplification:
        step = build_three_step_oaa(
            base_step,
            hamiltonian.num_qubits,
            name=f"MPF_step_{2 * m}_oaa",
            gate_label="MPF_step/2",
        )
    else:
        step = base_step

    branch_count = step.num_qubits - hamiltonian.num_qubits
    branch = QuantumRegister(branch_count, "branch")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(branch, system, name=f"MPF-{2 * m}")
    step_gate = step.to_gate(label=f"MPF step x{oaa_factor}")
    for _ in range(segments):
        circuit.append(step_gate, [*branch, *system])
    metadata["registers"]["branch"] = branch_count
    circuit.metadata = metadata
    return circuit
