"""QSVT Hamiltonian simulation using a Pauli-LCU block encoding.

The phase solver and the circuit use different conventions.  ``pyqsp``'s
``sym_qsp`` solver places the requested real polynomial in the imaginary part
of the Wx response.  We first convert the phases to the QSVT reflection
convention and then extract that imaginary part coherently with ``V`` and
``V.inverse()``.  Keeping those steps explicit prevents a phase list from
silently being interpreted in the wrong convention.
"""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from scipy.special import gammaln, jv

from .circuit_utils import (
    append_zero_projector_phase,
    build_block_encoding,
    pauli_lcu_oracles,
)
from .hamiltonians import PauliHamiltonian


@dataclass(frozen=True)
class HamiltonianSimulationPhases:
    """Auditable cosine/sine phases for one Hamiltonian-simulation time.

    ``cosine`` and ``sine`` are already converted from pyQSP's Wx convention
    to projector-phase QSVT angles.  Both polynomial approximations use the
    same ``scale`` so their relative normalization is preserved by the final
    linear combination.
    """

    cosine: tuple[float, ...]
    sine: tuple[float, ...]
    alpha_time: float
    epsilon: float
    scale: float
    cosine_tail_bound: float
    sine_tail_bound: float
    cosine_phase_residual: float
    sine_phase_residual: float

    @property
    def cosine_degree(self) -> int:
        return len(self.cosine) - 1

    @property
    def sine_degree(self) -> int:
        return len(self.sine) - 1


def _bessel_parity_tail_bound(abs_time: float, first_omitted_degree: int) -> float:
    """Bound one parity tail of the Jacobi--Anger expansion.

    For q >= |t|-1, the factorial Bessel bound gives
    2 sum_l |J_{q+2l}(t)| <= (8/3) (|t|/2)^q / q!.
    """
    if abs_time == 0:
        return 0.0
    if first_omitted_degree < abs_time - 1:
        return math.inf
    log_bound = (
        math.log(8 / 3)
        + first_omitted_degree * math.log(abs_time / 2)
        - float(gammaln(first_omitted_degree + 1))
    )
    return 0.0 if log_bound < math.log(np.finfo(float).tiny) else math.exp(log_bound)


def _jacobi_anger_polynomials(
    alpha_time: float, tail_budget: float, scale: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return equally scaled Chebyshev coefficients for cosine and sine."""
    abs_time = abs(alpha_time)
    truncation_order = 1
    while True:
        cosine_bound = _bessel_parity_tail_bound(abs_time, 2 * truncation_order + 2)
        sine_bound = _bessel_parity_tail_bound(abs_time, 2 * truncation_order + 3)
        if cosine_bound <= tail_budget and sine_bound <= tail_budget:
            break
        truncation_order += 1
        if truncation_order > 100_000:
            raise RuntimeError("failed to find a Jacobi-Anger truncation order")

    cosine = np.zeros(2 * truncation_order + 1)
    cosine[0] = jv(0, alpha_time)
    for k in range(1, truncation_order + 1):
        cosine[2 * k] = 2 * ((-1) ** k) * jv(2 * k, alpha_time)

    sine = np.zeros(2 * truncation_order + 2)
    for k in range(truncation_order + 1):
        sine[2 * k + 1] = 2 * ((-1) ** k) * jv(2 * k + 1, alpha_time)

    return scale * cosine, scale * sine, scale * cosine_bound, scale * sine_bound


def _qsp_to_qsvt_angles(phases: Sequence[float]) -> np.ndarray:
    """Convert Wx-QSP phases to the QSVT reflection convention.

    This is the phase map in Appendix A.2 of Martyn et al., arXiv:2105.02859.
    """
    converted = np.asarray(phases, dtype=float).copy()
    degree = len(converted) - 1
    if degree < 1:
        raise ValueError("at least two QSP phases are required")
    converted[0] += (2 * degree - 1) * np.pi / 4
    converted[1:degree] -= np.pi / 2
    converted[degree] -= np.pi / 4
    return converted


def _solve_symmetric_phases(coefficients: np.ndarray, tolerance: float) -> tuple[np.ndarray, float]:
    """Solve phases and verify the imaginary Wx response on a scalar grid."""
    try:
        from pyqsp import angle_sequence, response
    except ImportError as exc:
        raise ImportError("install the optional dependency with: pip install -e .[qsp]") from exc

    polynomial = np.polynomial.chebyshev.Chebyshev(coefficients)
    # pyQSP currently writes iteration diagnostics to stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        qsp_phases, _, _ = angle_sequence.QuantumSignalProcessingPhases(
            polynomial,
            method="sym_qsp",
            chebyshev_basis=True,
        )
    grid = np.cos(np.pi * np.arange(2049) / 2048)
    values = response.ComputeQSPResponse(
        grid, qsp_phases, signal_operator="Wx", sym_qsp=True
    )["pdat"]
    residual = float(np.max(np.abs(values.imag - polynomial(grid))))
    if residual > tolerance:
        raise RuntimeError(
            f"pyQSP phase reconstruction residual {residual:.3e} exceeds {tolerance:.3e}"
        )
    return _qsp_to_qsvt_angles(qsp_phases), residual


def estimate_qsvt_degree(alpha_time: float, epsilon: float) -> int:
    """Return the larger Jacobi--Anger component degree for a safe baseline."""
    if not np.isfinite(alpha_time) or alpha_time < 0 or not 0 < epsilon < 0.5:
        raise ValueError("require finite alpha_time >= 0 and 0 < epsilon < 1/2")
    if alpha_time == 0:
        return 0
    source_budget = epsilon / 18
    _, sine, _, _ = _jacobi_anger_polynomials(alpha_time, source_budget, 1 - source_budget)
    return len(sine) - 1


def synthesize_hamsim_phases(
    alpha_time: float,
    epsilon: float,
) -> HamiltonianSimulationPhases:
    """Synthesize a common-scale cosine/sine phase pair.

    The total target error is split between Jacobi--Anger truncation, the
    boundary safety scale, and numerical phase reconstruction.  The resulting
    scalar response is checked a posteriori before the phase set is returned.
    """
    if not np.isfinite(alpha_time) or alpha_time == 0:
        raise ValueError("alpha_time must be finite and nonzero")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 1/2)")

    source_budget = epsilon / 18
    scale = 1.0 - source_budget
    cosine_poly, sine_poly, cosine_tail, sine_tail = _jacobi_anger_polynomials(
        alpha_time, source_budget, scale
    )
    cosine, cosine_residual = _solve_symmetric_phases(cosine_poly, source_budget)
    sine, sine_residual = _solve_symmetric_phases(sine_poly, source_budget)
    return HamiltonianSimulationPhases(
        cosine=tuple(float(value) for value in cosine),
        sine=tuple(float(value) for value in sine),
        alpha_time=float(alpha_time),
        epsilon=float(epsilon),
        scale=scale,
        cosine_tail_bound=cosine_tail,
        sine_tail_bound=sine_tail,
        cosine_phase_residual=cosine_residual,
        sine_phase_residual=sine_residual,
    )


def _append_exact_projector_phase(
    circuit: QuantumCircuit, ancillas: QuantumRegister, phi: float
) -> None:
    """Append exp(i phi (2 Pi-I)), retaining its global phase."""
    circuit.global_phase -= float(phi)
    append_zero_projector_phase(circuit, ancillas, 2 * float(phi))


def _build_qsvt_response_circuit(
    hamiltonian: PauliHamiltonian,
    phases: Sequence[float],
) -> QuantumCircuit:
    """Build the complex QSVT response before quadrature extraction."""
    phase_values = np.asarray(phases, dtype=float)
    if phase_values.ndim != 1 or len(phase_values) < 2 or not np.isfinite(phase_values).all():
        raise ValueError("phases must be a finite one-dimensional sequence of length at least two")

    block_encoding = build_block_encoding(hamiltonian)
    index_count = block_encoding.num_qubits - hamiltonian.num_qubits
    index = QuantumRegister(index_count, "index")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(index, system, name=f"QSVT-response-d{len(phase_values)-1}")

    _append_exact_projector_phase(circuit, index, phase_values[0])
    for phi in phase_values[1:]:
        circuit.compose(block_encoding, [*index, *system], inplace=True)
        _append_exact_projector_phase(circuit, index, phi)
    return circuit


def _build_qsvt_component_circuit(
    hamiltonian: PauliHamiltonian,
    phases: Sequence[float],
    *,
    component: str,
) -> QuantumCircuit:
    """Extract the real target polynomial from a symmetric-QSP response."""
    if component not in {"cos", "sin"}:
        raise ValueError("component must be 'cos' or 'sin'")
    response = _build_qsvt_response_circuit(hamiltonian, phases).to_gate(
        label=f"V_{component}"
    )
    index_count = response.num_qubits - hamiltonian.num_qubits
    quadrature = QuantumRegister(1, "quadrature")
    index = QuantumRegister(index_count, "index")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(quadrature, index, system, name=f"QSVT_{component}")

    circuit.h(quadrature)
    targets = [*index, *system]
    circuit.append(response.control(1, ctrl_state=0), [*quadrature, *targets])
    circuit.append(response.inverse().control(1, ctrl_state=1), [*quadrature, *targets])
    # (-i V + i V^dagger)/2 extracts Im(V)'s selected block.
    circuit.z(quadrature)
    circuit.global_phase -= np.pi / 2
    circuit.h(quadrature)
    circuit.metadata = {
        "algorithm": "qsvt-component",
        "component": component,
        "degree": len(phases) - 1,
        "phase_convention": "QSVT reflection angles converted from pyQSP Wx/sym_qsp",
        "target_quadrature": "imaginary",
        "postselection": "quadrature and index registers all-zero",
    }
    return circuit


# Temporary compatibility builders.  The public raw-phase API is removed in
# the next implementation step, after the complete Hamiltonian circuit exists.
def build_qsvt_circuit(
    hamiltonian: PauliHamiltonian,
    phases: Sequence[float],
) -> QuantumCircuit:
    """Build the legacy alternating walk circuit (temporary compatibility)."""
    phase_values = np.asarray(phases, dtype=float)
    if phase_values.ndim != 1 or len(phase_values) < 2:
        raise ValueError("at least two one-dimensional QSP phases are required")
    prepare, select = pauli_lcu_oracles(hamiltonian)
    select_gate = select.to_gate(label="SELECT")
    index = QuantumRegister(prepare.num_qubits, "index")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(index, system, name=f"QSVT-d{len(phase_values)-1}")

    def projector_phase(phi: float) -> None:
        circuit.append(prepare.inverse(), index)
        append_zero_projector_phase(circuit, index, 2 * float(phi))
        circuit.append(prepare, index)

    circuit.append(prepare, index)
    projector_phase(phase_values[0])
    for phi in phase_values[1:]:
        circuit.append(select_gate, [*index, *system])
        circuit.append(prepare.inverse(), index)
        append_zero_projector_phase(circuit, index, np.pi)
        circuit.append(prepare, index)
        projector_phase(phi)
    circuit.append(prepare.inverse(), index)
    circuit.metadata = {"algorithm": "qsvt", "degree": len(phase_values) - 1}
    return circuit


def build_hamiltonian_qsvt_circuit(
    hamiltonian: PauliHamiltonian,
    cosine_phases: Sequence[float],
    sine_phases: Sequence[float],
) -> QuantumCircuit:
    """Build the legacy raw-phase Hamiltonian circuit (temporary compatibility)."""
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
    }
    return circuit
