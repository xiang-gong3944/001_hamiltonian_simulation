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
        return build_trotter_circuit_from_plan(plan)
    if isinstance(plan, MPFPlan):
        return build_multiproduct_circuit_from_plan(plan)
    if isinstance(plan, QSVTPlan):
        return build_hamiltonian_qsvt_circuit_from_plan(plan)
    raise TypeError("plan must be a supported simulation plan")
