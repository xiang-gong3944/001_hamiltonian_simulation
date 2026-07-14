"""Gate counting and explicit assumptions for fault-tolerant T estimates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile


@dataclass(frozen=True)
class ResourceEstimate:
    algorithm: str
    num_qubits: int
    cnot_count: int
    t_count: int
    rotation_count: int
    depth: int
    counting_mode: str = "transpiled"
    rotation_synthesis_error: float = 1e-10

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def _is_multiple(angle: float, unit: float, atol: float = 1e-10) -> bool:
    return bool(np.isclose(angle / unit, round(angle / unit), atol=atol))


def t_cost_for_z_rotation(angle: float, epsilon: float) -> int:
    """Estimate ancilla-free Clifford+T synthesis cost for one Rz.

    Exact Clifford and T-axis rotations are recognized. Generic rotations use
    ceil(3 log2(1/epsilon) + log2(log2(1/epsilon))), a transparent conservative
    approximation to modern number-theoretic synthesis scaling.
    """
    angle = float(angle) % (2 * np.pi)
    if _is_multiple(angle, np.pi / 2):
        return 0
    if _is_multiple(angle, np.pi / 4):
        return 1
    if not 0 < epsilon < 1:
        raise ValueError("rotation synthesis epsilon must lie in (0, 1)")
    log_precision = math.log2(1 / epsilon)
    return math.ceil(3 * log_precision + math.log2(max(log_precision, 1.0)))


def count_circuit_resources(
    circuit: QuantumCircuit,
    *,
    algorithm: str | None = None,
    total_synthesis_error: float = 1e-6,
    optimization_level: int = 1,
) -> ResourceEstimate:
    """Transpile a concrete circuit and count CX plus estimated T gates.

    Qiskit retains arbitrary ``rz`` rotations. We allocate the total synthesis
    error equally across non-Clifford rotations and convert each to an estimated
    Clifford+T cost, avoiding the misleading convention that a continuous Rz
    is a free gate.
    """
    if not 0 < total_synthesis_error < 1:
        raise ValueError("total_synthesis_error must lie in (0, 1)")
    decomposed = transpile(
        circuit,
        basis_gates=["cx", "rz", "sx", "x"],
        optimization_level=optimization_level,
    )
    angles: list[float] = []
    for instruction in decomposed.data:
        if instruction.operation.name == "rz":
            try:
                angles.append(float(instruction.operation.params[0]))
            except (TypeError, ValueError) as exc:
                raise ValueError("bind all circuit parameters before resource counting") from exc
    non_clifford = [a for a in angles if not _is_multiple(a, np.pi / 2)]
    per_rotation_error = total_synthesis_error / max(1, len(non_clifford))
    t_count = sum(t_cost_for_z_rotation(a, per_rotation_error) for a in angles)
    operations = decomposed.count_ops()
    label = algorithm or str((circuit.metadata or {}).get("algorithm", circuit.name))
    return ResourceEstimate(
        algorithm=label,
        num_qubits=decomposed.num_qubits,
        cnot_count=int(operations.get("cx", 0)),
        t_count=int(t_count),
        rotation_count=len(non_clifford),
        depth=int(decomposed.depth()),
        rotation_synthesis_error=total_synthesis_error,
    )

