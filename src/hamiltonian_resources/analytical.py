"""Structured analytical compilation of backend-independent simulation plans."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .planning import MPFPlan, QSVTPlan, SimulationPlan, TrotterPlan
from .resources import (
    T_PER_AND,
    ResourceEstimate,
    multicontrol_and_pairs,
    t_cost_for_z_rotation,
)


_CX_PER_AND = 6


@dataclass(frozen=True)
class ResourceModelProvenance:
    """Identity and assumptions of one resource-compilation backend."""

    backend: str
    model: str
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "assumptions": list(self.assumptions),
        }


def _analytical_provenance(plan: SimulationPlan) -> ResourceModelProvenance:
    common = (
        "arbitrary Rz synthesis error is divided equally across compiled rotations",
        "each temporary-AND compute/uncompute pair costs 4 T and 6 CX",
    )
    if isinstance(plan, TrotterPlan):
        assumptions = (
            "Pauli evolutions use the plan's resolved ordered Trotter groups",
            "each Pauli word uses a parity ladder with no cross-word cancellation",
        )
    elif isinstance(plan, MPFPlan):
        assumptions = (
            "MPF SELECT uses one reusable equality-flag ancilla per physical branch",
            "each segment uses three SELECT, six PREPARE, and two good reflections",
            "generic Qiskit controlled-gate decomposition is not modeled",
        )
    else:
        assumptions = (
            "controlled QSVT responses multiplex V and V-dagger in shared query slots",
            "only projector phases are selected on quadrature/component controls",
            "generic Qiskit .control() query duplication is not modeled",
        )
    return ResourceModelProvenance(
        backend="analytical",
        model="structured-analytical-v1",
        assumptions=common + assumptions,
    )


def compile_resources_analytically(
    plan: SimulationPlan,
) -> tuple[ResourceEstimate, ResourceModelProvenance]:
    """Compile a selected logical plan into the historical analytical model."""
    weights = [
        sum(character != "I" for character in label)
        for label, _ in plan.hamiltonian.terms
    ]
    mean_ladder_cx = sum(2 * max(0, weight - 1) for weight in weights) / len(weights)
    synth_error = plan.error_budget.synthesis_error
    term_count = plan.hamiltonian.term_count

    if isinstance(plan, TrotterPlan):
        rotations = plan.logical_counts.as_dict()["totals"]["pauli_evolution"]
        and_pairs = 0
        cnots = math.ceil(rotations * mean_ladder_cx)
        qubits = plan.hamiltonian.num_qubits
    elif isinstance(plan, MPFPlan):
        segments = plan.segments
        structure = plan.lcu_structure
        branch_bits = structure.branch_bits
        branch_flag_pairs = multicontrol_and_pairs(branch_bits)
        phase_pairs = multicontrol_and_pairs(branch_bits - 1)
        select_rotations = (
            plan.logical_counts.as_dict()["per_segment"]["pauli_evolution"] // 3
        )
        prepare_rotations = 2**branch_bits - 1
        rotations = segments * (3 * 2 * select_rotations + 6 * prepare_rotations)
        and_pairs = segments * (
            3 * structure.physical_branch_count * branch_flag_pairs
            + 3 * structure.sign_branch_count * phase_pairs
            + 2 * phase_pairs
        )
        cnots = math.ceil(
            segments
            * (
                3 * select_rotations * (mean_ladder_cx + 2)
                + 6 * max(0, 2**branch_bits - 2)
                + (3 * structure.sign_branch_count + 2)
            )
            + and_pairs * _CX_PER_AND
        )
        qubits = (
            plan.hamiltonian.num_qubits
            + branch_bits
            + max(branch_flag_pairs, phase_pairs)
        )
    elif isinstance(plan, QSVTPlan):
        index_bits = max(1, math.ceil(math.log2(term_count)))
        logical = plan.logical_counts.as_dict()["totals"]
        queries = logical["block_encoding_query_slot"]
        prepare_calls = logical["prepare_slot"]
        prepare_rotations = max(1, 2**index_bits - 1)
        phase_slots = logical["projector_phase_slot"]
        rotations = prepare_calls * prepare_rotations + 2 * phase_slots
        select_pairs = multicontrol_and_pairs(index_bits + 1)
        phase_pairs = multicontrol_and_pairs(index_bits + 2)
        and_pairs = (
            queries * term_count * select_pairs
            + 2 * phase_slots * phase_pairs
            + 2 * phase_pairs
        )
        cnots = (
            prepare_calls * max(0, 2**index_bits - 2)
            + queries * sum(weights)
            + and_pairs * _CX_PER_AND
        )
        qubits = plan.hamiltonian.num_qubits + index_bits + 2 + select_pairs
    else:  # pragma: no cover - the public union is exhaustively checked upstream
        raise TypeError("unsupported simulation plan")

    per_rotation = synth_error / max(1, rotations)
    t_count = rotations * t_cost_for_z_rotation(0.17320508075688773, per_rotation)
    t_count += and_pairs * T_PER_AND
    resources = ResourceEstimate(
        algorithm=plan.family,
        num_qubits=qubits,
        cnot_count=int(cnots),
        t_count=int(t_count),
        rotation_count=int(rotations),
        depth=-1,
        counting_mode="analytical-model",
        rotation_synthesis_error=synth_error,
        toffoli_count=int(and_pairs),
    )
    return resources, _analytical_provenance(plan)
