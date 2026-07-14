"""QSVT/QSP circuits built around a Pauli-LCU block encoding."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from .circuit_utils import append_zero_projector_phase, pauli_lcu_oracles
from .hamiltonians import PauliHamiltonian


def estimate_qsvt_degree(alpha_time: float, epsilon: float) -> int:
    """Practical Jacobi-Anger truncation degree for Hamiltonian simulation.

    The returned odd integer follows the asymptotic optimal query scaling
    O(alpha*t + log(1/epsilon)). It is a sizing heuristic; phase synthesis and
    a posteriori polynomial validation determine the actual approximation.
    """
    if alpha_time < 0 or not 0 < epsilon < 1:
        raise ValueError("require alpha_time >= 0 and 0 < epsilon < 1")
    degree = math.ceil(alpha_time + 1.5 * math.log(2 / epsilon))
    return degree if degree % 2 else degree + 1


def synthesize_hamsim_phases(
    alpha_time: float,
    epsilon: float,
    component: str,
) -> tuple[np.ndarray, float]:
    """Synthesize symmetric-QSP phases for cosine or sine using ``pyqsp``.

    Hamiltonian evolution is cos(alpha*t*x)-i sin(alpha*t*x); definite parity
    requires synthesizing its even and odd components separately. The returned
    scale (<1) prevents boundary saturation. Combine both components with one
    additional LCU qubit when a fully coherent propagator is required.
    """
    if component not in {"cos", "sin"}:
        raise ValueError("component must be 'cos' or 'sin'")
    try:
        from pyqsp import angle_sequence
        from pyqsp.poly import PolyTaylorSeries
    except ImportError as exc:
        raise ImportError("install the optional dependency with: pip install -e .[qsp]") from exc

    degree = estimate_qsvt_degree(alpha_time, epsilon)
    if component == "cos" and degree % 2:
        degree += 1
    if component == "sin" and degree % 2 == 0:
        degree += 1
    scale = 1.0 - min(0.1, epsilon / 4)
    function = np.cos if component == "cos" else np.sin
    polynomial = PolyTaylorSeries().taylor_series(
        func=lambda x: function(alpha_time * x),
        degree=degree,
        max_scale=scale,
        chebyshev_basis=True,
        cheb_samples=max(2 * degree, 20),
    )
    phases, _, _ = angle_sequence.QuantumSignalProcessingPhases(
        polynomial, method="sym_qsp", chebyshev_basis=True
    )
    return np.asarray(phases, dtype=float), scale


def build_qsvt_circuit(
    hamiltonian: PauliHamiltonian,
    phases: Sequence[float],
) -> QuantumCircuit:
    """Build the QSVT alternating walk/projector-phase circuit.

    ``phases`` must match the Wx/symmetric-QSP convention. The circuit contains
    PREPARE, SELECT, reflections, and phase rotations explicitly; it does not
    construct or exponentiate the Hamiltonian matrix.
    """
    phase_values = np.asarray(phases, dtype=float)
    if phase_values.ndim != 1 or len(phase_values) < 2:
        raise ValueError("at least two one-dimensional QSP phases are required")
    prepare, select = pauli_lcu_oracles(hamiltonian)
    select_gate = select.to_gate(label="SELECT")
    index = QuantumRegister(prepare.num_qubits, "index")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(index, system, name=f"QSVT-d{len(phase_values)-1}")

    # Projector phases are A exp(i*2phi |0><0|) A^dagger, up to global phase.
    def projector_phase(phi: float) -> None:
        circuit.append(prepare.inverse(), index)
        append_zero_projector_phase(circuit, index, 2 * float(phi))
        circuit.append(prepare, index)

    circuit.append(prepare, index)
    projector_phase(phase_values[0])
    for phi in phase_values[1:]:
        circuit.append(select_gate, [*index, *system])
        # Reflection 2|G><G|-I (the ignored global sign is QSP-convention safe).
        circuit.append(prepare.inverse(), index)
        append_zero_projector_phase(circuit, index, np.pi)
        circuit.append(prepare, index)
        projector_phase(phi)
    circuit.append(prepare.inverse(), index)
    circuit.metadata = {
        "algorithm": "qsvt",
        "degree": len(phase_values) - 1,
        "alpha": hamiltonian.alpha,
        "phase_convention": "Wx symmetric QSP; projector global phases omitted",
        "postselection": "index register all-zero selects the transformed block",
    }
    return circuit


def build_hamiltonian_qsvt_circuit(
    hamiltonian: PauliHamiltonian,
    cosine_phases: Sequence[float],
    sine_phases: Sequence[float],
) -> QuantumCircuit:
    """Coherently combine even/odd QSVT components for exp(-iHt).

    A branch Hadamard and postselection implement (U_cos - i U_sin)/2.
    Component polynomial scaling factors must be accounted for when interpreting
    the block; oblivious amplitude amplification can replace postselection.
    """
    cosine = build_qsvt_circuit(hamiltonian, cosine_phases).to_gate(label="QSVT_cos")
    sine = build_qsvt_circuit(hamiltonian, sine_phases).to_gate(label="QSVT_sin")
    branch = QuantumRegister(1, "component")
    index_count = cosine.num_qubits - hamiltonian.num_qubits
    index = QuantumRegister(index_count, "index")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(branch, index, system, name="QSVT_hamsim")
    circuit.h(branch)
    circuit.append(cosine.control(1, ctrl_state=0), [*branch, *index, *system])
    circuit.append(sine.control(1, ctrl_state=1), [*branch, *index, *system])
    circuit.sdg(branch)
    circuit.h(branch)
    circuit.metadata = {
        "algorithm": "qsvt-hamiltonian-simulation",
        "cosine_degree": len(cosine_phases) - 1,
        "sine_degree": len(sine_phases) - 1,
        "alpha": hamiltonian.alpha,
        "postselection": "component and index registers all-zero",
        "lcu_scale": 2.0,
    }
    return circuit
