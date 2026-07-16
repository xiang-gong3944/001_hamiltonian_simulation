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


def _build_hamiltonian_lcu_circuit(
    hamiltonian: PauliHamiltonian,
    phases: HamiltonianSimulationPhases,
) -> QuantumCircuit:
    """Build the pre-amplification block encoding of s exp(-iHt)/2."""
    cosine = _build_qsvt_component_circuit(
        hamiltonian, phases.cosine, component="cos"
    ).to_gate(label="QSVT_cos")
    sine = _build_qsvt_component_circuit(
        hamiltonian, phases.sine, component="sin"
    ).to_gate(label="QSVT_sin")
    component = QuantumRegister(1, "component")
    quadrature = QuantumRegister(1, "quadrature")
    index_count = cosine.num_qubits - hamiltonian.num_qubits - 1
    index = QuantumRegister(index_count, "index")
    system = QuantumRegister(hamiltonian.num_qubits, "system")
    circuit = QuantumCircuit(
        component, quadrature, index, system, name="QSVT_hamsim_unamplified"
    )

    targets = [*quadrature, *index, *system]
    circuit.h(component)
    circuit.append(cosine.control(1, ctrl_state=0), [*component, *targets])
    circuit.append(sine.control(1, ctrl_state=1), [*component, *targets])
    circuit.sdg(component)
    circuit.h(component)

    base_queries = 2 * (phases.cosine_degree + phases.sine_degree)
    circuit.metadata = {
        "algorithm": "qsvt-hamiltonian-simulation",
        "construction": "explicit-cos-sin-lcu",
        "cosine_degree": phases.cosine_degree,
        "sine_degree": phases.sine_degree,
        "alpha": hamiltonian.alpha,
        "alpha_time": phases.alpha_time,
        "epsilon": phases.epsilon,
        "polynomial_scale": phases.scale,
        "block_scale": 2 / phases.scale,
        "amplitude_amplification": False,
        "good_subspace": "component, quadrature, and index registers all-zero",
        "registers": {
            "component": 1,
            "quadrature": 1,
            "index": index_count,
            "system": hamiltonian.num_qubits,
        },
        "base_block_encoding_queries": base_queries,
        "base_circuit_uses": 1,
    }
    return circuit


def _apply_three_step_oaa(
    base: QuantumCircuit,
    system_qubits: int,
) -> QuantumCircuit:
    """Apply one robust OAA round to a block with amplitude close to 1/2."""
    ancilla_count = base.num_qubits - system_qubits
    if ancilla_count < 1:
        raise ValueError("OAA requires at least one good-subspace ancilla")

    ancillas = QuantumRegister(ancilla_count, "ancilla")
    system = QuantumRegister(system_qubits, "system")
    amplified = QuantumCircuit(ancillas, system, name="QSVT_hamsim_oaa")
    base_gate = base.to_gate(label="U_hamsim/2")
    targets = [*ancillas, *system]

    amplified.append(base_gate, targets)
    append_zero_projector_phase(amplified, ancillas, np.pi)
    amplified.append(base_gate.inverse(), targets)
    append_zero_projector_phase(amplified, ancillas, np.pi)
    amplified.append(base_gate, targets)
    # U R U^dagger R U has block 4a^3-3a; correct its known sign.
    amplified.global_phase += np.pi

    metadata = dict(base.metadata or {})
    metadata.update(
        amplitude_amplification=True,
        block_scale=1.0,
        base_circuit_uses=3,
        base_block_encoding_queries=3 * int(metadata["base_block_encoding_queries"]),
        oaa_sequence="-U R U^dagger R U",
    )
    amplified.metadata = metadata
    return amplified


def build_hamiltonian_qsvt_circuit(
    hamiltonian: PauliHamiltonian,
    time: float,
    epsilon: float,
    *,
    amplitude_amplification: bool = True,
) -> QuantumCircuit:
    """Build a coherent QSVT approximation of ``exp(-i H time)``.

    Without amplitude amplification the all-zero ancilla block is approximately
    ``scale * exp(-i H time) / 2``.  The default performs one robust OAA round,
    producing a near-deterministic block encoding of the propagator.
    """
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 1/2)")
    if time == 0:
        system = QuantumRegister(hamiltonian.num_qubits, "system")
        identity = QuantumCircuit(system, name="QSVT_hamsim_identity")
        identity.metadata = {
            "algorithm": "qsvt-hamiltonian-simulation",
            "construction": "zero-time-identity",
            "alpha": hamiltonian.alpha,
            "alpha_time": 0.0,
            "epsilon": float(epsilon),
            "polynomial_scale": 1.0,
            "block_scale": 1.0,
            "amplitude_amplification": bool(amplitude_amplification),
            "good_subspace": "no ancillas",
            "registers": {
                "component": 0,
                "quadrature": 0,
                "index": 0,
                "system": hamiltonian.num_qubits,
            },
            "base_block_encoding_queries": 0,
            "base_circuit_uses": 0,
        }
        return identity

    phases = synthesize_hamsim_phases(hamiltonian.alpha * float(time), epsilon)
    base = _build_hamiltonian_lcu_circuit(hamiltonian, phases)
    if not amplitude_amplification:
        return base
    return _apply_three_step_oaa(base, hamiltonian.num_qubits)
