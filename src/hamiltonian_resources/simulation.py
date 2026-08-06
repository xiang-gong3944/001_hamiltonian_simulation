"""Small-system statevector validation against exact dense evolution."""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from .hamiltonians import PauliHamiltonian
from .multiproduct import MPFSchedule, build_multiproduct_circuit
from .qsvt import build_hamiltonian_qsvt_circuit
from .trotter import TrotterPartition, build_trotter_circuit


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
    trotter_partition: TrotterPartition = "auto",
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
    if method == "trotter":
        circuit = build_trotter_circuit(
            hamiltonian,
            time,
            reps,
            trotter_order,
            partition=trotter_partition,
        )
    elif method == "multiproduct":
        circuit = build_multiproduct_circuit(
            hamiltonian,
            time,
            mpf_m,
            segments=reps,
            schedule=mpf_schedule,
            amplitude_amplification=amplitude_amplification,
        )
    elif method == "qsvt":
        circuit = build_hamiltonian_qsvt_circuit(
            hamiltonian,
            time,
            qsvt_epsilon,
            amplitude_amplification=amplitude_amplification,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    return _compare_circuit_with_exact(hamiltonian, float(time), circuit, method, psi)


def _compare_circuit_with_exact(
    hamiltonian: PauliHamiltonian,
    time: float,
    circuit,
    method: str,
    psi: np.ndarray,
) -> dict[str, float | str]:
    exact = expm(-1j * time * hamiltonian.matrix()) @ psi
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
            raise RuntimeError(f"{method.upper()} all-zero postselection probability is zero")
        actual = postselected / np.sqrt(success_probability)

    fidelity = float(abs(np.vdot(exact, actual)) ** 2)
    return {
        "method": method,
        "state_error": _phase_aligned_error(actual, exact),
        "fidelity": fidelity,
        "success_probability": success_probability,
    }


def compare_plan_with_exact(
    plan,
    *,
    initial_state: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Validate a selected plan with its reference Qiskit circuit."""
    from .evaluation import build_simulation_circuit
    from .planning import MPFPlan, QSVTPlan, TrotterPlan

    if not isinstance(plan, (TrotterPlan, MPFPlan, QSVTPlan)):
        raise TypeError("plan must be a supported simulation plan")
    psi = (
        zero_state(plan.hamiltonian.num_qubits)
        if initial_state is None
        else np.asarray(initial_state)
    )
    if psi.shape != (2**plan.hamiltonian.num_qubits,) or not np.isclose(
        np.linalg.norm(psi), 1
    ):
        raise ValueError("initial_state must be a normalized 2**num_qubits vector")
    circuit = build_simulation_circuit(plan)
    return _compare_circuit_with_exact(
        plan.hamiltonian,
        plan.time,
        circuit,
        plan.family,
        psi,
    )
