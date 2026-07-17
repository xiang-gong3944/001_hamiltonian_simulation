"""Small-system statevector validation against exact dense evolution."""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from .hamiltonians import PauliHamiltonian
from .multiproduct import MPFSchedule, build_multiproduct_circuit
from .qsvt import build_hamiltonian_qsvt_circuit
from .trotter import build_trotter_circuit


def zero_state(num_qubits: int) -> np.ndarray:
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1.0
    return state


def _phase_aligned_error(actual: np.ndarray, expected: np.ndarray) -> float:
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    return float(np.linalg.norm(actual - phase * expected))


def compare_with_exact(
    hamiltonian: PauliHamiltonian,
    time: float,
    *,
    method: Literal["trotter", "multiproduct", "qsvt"] = "trotter",
    initial_state: np.ndarray | None = None,
    trotter_order: int = 2,
    reps: int = 1,
    mpf_m: int = 2,
    mpf_schedule: MPFSchedule = "new",
    qsvt_epsilon: float = 1e-3,
    amplitude_amplification: bool = True,
) -> dict[str, float | str]:
    """Run a small circuit and compare its output with exp(-iHt)|psi>.

    MPF and QSVT results condition on all non-system registers being zero and
    report the corresponding success probability. ``mpf_schedule`` selects the
    exponent table independently of the MPF order ``mpf_m``. This routine is
    intentionally dense and should only be used for small systems.
    """
    psi = zero_state(hamiltonian.num_qubits) if initial_state is None else np.asarray(initial_state)
    if psi.shape != (2**hamiltonian.num_qubits,) or not np.isclose(np.linalg.norm(psi), 1):
        raise ValueError("initial_state must be a normalized 2**num_qubits vector")
    exact = expm(-1j * float(time) * hamiltonian.matrix()) @ psi

    if method == "trotter":
        circuit = build_trotter_circuit(hamiltonian, time, reps, trotter_order)
        actual = np.asarray(Statevector(psi).evolve(circuit).data)
        success_probability = 1.0
    elif method == "multiproduct":
        circuit = build_multiproduct_circuit(
            hamiltonian,
            time,
            mpf_m,
            segments=reps,
            schedule=mpf_schedule,
            amplitude_amplification=amplitude_amplification,
        )
        ancillas = circuit.num_qubits - hamiltonian.num_qubits
        if ancillas == 0:
            actual = np.asarray(Statevector(psi).evolve(circuit).data)
            success_probability = 1.0
        else:
            joint = np.zeros(2**circuit.num_qubits, dtype=complex)
            joint[:: 2**ancillas] = psi
            output = np.asarray(Statevector(joint).evolve(circuit).data)
            postselected = output[:: 2**ancillas]
            success_probability = float(np.vdot(postselected, postselected).real)
            if success_probability <= np.finfo(float).eps:
                raise RuntimeError(
                    "MPF all-zero postselection probability is numerically zero"
                )
            actual = postselected / np.sqrt(success_probability)
    elif method == "qsvt":
        circuit = build_hamiltonian_qsvt_circuit(
            hamiltonian,
            time,
            qsvt_epsilon,
            amplitude_amplification=amplitude_amplification,
        )
        ancillas = circuit.num_qubits - hamiltonian.num_qubits
        if ancillas == 0:
            actual = np.asarray(Statevector(psi).evolve(circuit).data)
            success_probability = 1.0
        else:
            joint = np.zeros(2**circuit.num_qubits, dtype=complex)
            joint[:: 2**ancillas] = psi
            output = np.asarray(Statevector(joint).evolve(circuit).data)
            postselected = output[:: 2**ancillas]
            success_probability = float(np.vdot(postselected, postselected).real)
            if success_probability <= np.finfo(float).eps:
                raise RuntimeError("QSVT all-zero postselection probability is numerically zero")
            actual = postselected / np.sqrt(success_probability)
    else:
        raise ValueError(f"unknown method: {method}")

    fidelity = float(abs(np.vdot(exact, actual)) ** 2)
    return {
        "method": method,
        "state_error": _phase_aligned_error(actual, exact),
        "fidelity": fidelity,
        "success_probability": success_probability,
    }
