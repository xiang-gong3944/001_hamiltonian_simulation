"""Facade API for evaluating selected Hamiltonian-simulation plans."""

from __future__ import annotations

from dataclasses import dataclass

from .analytical import ResourceModelProvenance, compile_resources_analytically
from .hamiltonians import PauliHamiltonian
from .method_specs import MethodSpec
from .planning import (
    ErrorBudget,
    LogicalOperationCounts,
    SimulationPlan,
    plan_simulation,
)
from .resources import ResourceEstimate
from .trotter import TrotterPartition


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
