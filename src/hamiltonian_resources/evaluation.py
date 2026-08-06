"""Facade API for evaluating selected Hamiltonian-simulation plans."""

from __future__ import annotations

from dataclasses import dataclass

from .analytical import ResourceModelProvenance, compile_resources_analytically
from .hamiltonians import PauliHamiltonian
from .method_specs import MethodSpec
from .multiproduct import build_multiproduct_circuit_from_plan
from .planning import (
    ErrorBudget,
    LogicalOperationCounts,
    MPFPlan,
    QSVTPlan,
    SimulationPlan,
    TrotterPlan,
    plan_simulation,
)
from .qsvt import build_hamiltonian_qsvt_circuit_from_plan
from .resources import ResourceEstimate
from .trotter import TrotterPartition, build_trotter_circuit_from_plan


@dataclass(frozen=True)
class EvaluationReport:
    """A selected plan and one backend's resource result."""

    plan: SimulationPlan
    resources: ResourceEstimate
    resource_provenance: ResourceModelProvenance

    @property
    def logical_counts(self) -> LogicalOperationCounts:
        return self.plan.logical_counts

    @property
    def selected_parameters(self) -> dict[str, object]:
        return self.plan.selected_parameters

    @property
    def error_metadata(self) -> dict[str, object]:
        return self.plan.error_metadata

    @property
    def error_budget(self) -> ErrorBudget:
        return self.plan.error_budget


def estimate_plan_resources(plan: SimulationPlan) -> EvaluationReport:
    """Evaluate an already-selected plan without invoking parameter selection."""
    resources, provenance = compile_resources_analytically(plan)
    return EvaluationReport(plan, resources, provenance)


def estimate_resources(
    hamiltonian: PauliHamiltonian,
    method: MethodSpec,
    time: float,
    target_error: float,
    *,
    synthesis_error_fraction: float = 0.1,
    trotter_partition: TrotterPartition = "auto",
) -> EvaluationReport:
    """Select one logical algorithm plan and estimate its analytical resources."""
    plan = plan_simulation(
        hamiltonian,
        method,
        time,
        target_error,
        synthesis_error_fraction=synthesis_error_fraction,
        trotter_partition=trotter_partition,
    )
    return estimate_plan_resources(plan)


def build_simulation_circuit(plan: SimulationPlan):
    """Compile a selected plan with the reference Qiskit backend."""
    if isinstance(plan, TrotterPlan):
        circuit = build_trotter_circuit_from_plan(plan)
    elif isinstance(plan, MPFPlan):
        circuit = build_multiproduct_circuit_from_plan(plan)
    elif isinstance(plan, QSVTPlan):
        circuit = build_hamiltonian_qsvt_circuit_from_plan(plan)
    else:
        raise TypeError("plan must be a supported simulation plan")
    metadata = dict(circuit.metadata or {})
    metadata.update(serialize_plan_metadata(plan))
    metadata["resource_provenance"] = _qiskit_provenance(plan).as_dict()
    circuit.metadata = metadata
    return circuit


def serialize_plan_metadata(plan: SimulationPlan) -> dict[str, object]:
    """Return a derived metadata view without introducing another data owner."""
    return {
        "method_id": plan.method.method_id,
        "method_family": plan.family,
        "evolution_time": plan.time,
        "selected_parameters": plan.selected_parameters,
        "logical_operation_counts": plan.logical_counts.as_dict(),
        "error_budget": {
            "target_error": plan.error_budget.target_error,
            "synthesis_fraction": plan.error_budget.synthesis_fraction,
            "algorithm_error": plan.error_budget.algorithm_error,
            "synthesis_error": plan.error_budget.synthesis_error,
        },
        "error_metadata": plan.error_metadata,
    }


def _qiskit_provenance(plan: SimulationPlan) -> ResourceModelProvenance:
    if isinstance(plan, TrotterPlan):
        model = "qiskit-pauli-evolution-reference"
        assumptions = (
            "Qiskit PauliEvolutionGate preserves the plan's group order",
            "the circuit is a reference construction rather than a routed implementation",
        )
    elif isinstance(plan, MPFPlan):
        model = "qiskit-generic-controlled-mpf-reference"
        assumptions = (
            "MPF branches use Qiskit generic controlled product-formula gates",
            "no reusable equality-flag structured compilation is imposed",
        )
    else:
        model = "qiskit-generic-controlled-qsvt-reference"
        assumptions = (
            "QSVT response extraction uses generic controlled V and V-dagger gates",
            "the generic reference circuit does not impose analytical query sharing",
        )
    return ResourceModelProvenance("qiskit-reference", model, assumptions)
