"""Single-pass parameter selection into immutable algorithm plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .hamiltonians import PauliHamiltonian
from .method_specs import MethodSpec, MultiproductMethod, QSVTMethod, TrotterMethod
from .multiproduct import (
    MPFErrorEstimate,
    MPFLCUStructure,
    mpf_lcu_structure,
    multiproduct_coefficients,
    select_mpf_segments,
)
from .qsvt import estimate_qsvt_degree
from .trotter import (
    SuzukiErrorEstimate,
    TrotterPartition,
    estimate_suzuki_error,
    resolve_trotter_structure,
    suzuki_group_factors,
)


@dataclass(frozen=True)
class ErrorBudget:
    """One target-error split, with all component budgets derived."""

    target_error: float
    synthesis_fraction: float = 0.1

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_error) or not 0 < self.target_error < 1:
            raise ValueError("target_error must lie in (0, 1)")
        if not np.isfinite(self.synthesis_fraction) or not 0 < self.synthesis_fraction < 1:
            raise ValueError("synthesis_fraction must lie in (0, 1)")
        object.__setattr__(self, "target_error", float(self.target_error))
        object.__setattr__(self, "synthesis_fraction", float(self.synthesis_fraction))

    @property
    def algorithm_error(self) -> float:
        return self.target_error * (1 - self.synthesis_fraction)

    @property
    def synthesis_error(self) -> float:
        return self.target_error * self.synthesis_fraction


@dataclass(frozen=True)
class LogicalOperationCounts:
    """Immutable logical-operation totals derived from a selected plan."""

    totals: tuple[tuple[str, int], ...]
    per_segment: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        for entries in (self.totals, self.per_segment):
            names = [name for name, _ in entries]
            if len(names) != len(set(names)):
                raise ValueError("logical operation names must be unique")
            if any(not name or count < 0 for name, count in entries):
                raise ValueError("logical operation counts must be named and nonnegative")

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            "totals": dict(self.totals),
            "per_segment": dict(self.per_segment),
        }


@dataclass(frozen=True)
class TrotterPlan:
    hamiltonian: PauliHamiltonian
    method: TrotterMethod
    time: float
    error_budget: ErrorBudget
    repetitions: int
    requested_partition: TrotterPartition
    resolved_partition: str
    group_term_indices: tuple[tuple[int, ...], ...]
    suzuki_factors: tuple[tuple[int, float], ...]
    error_estimate: SuzukiErrorEstimate

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("Trotter repetitions must be positive")
        flattened = tuple(index for group in self.group_term_indices for index in group)
        if sorted(flattened) != list(range(self.hamiltonian.term_count)):
            raise ValueError("Trotter groups must partition Hamiltonian term indices")
        if self.error_estimate.reps != self.repetitions:
            raise ValueError("Trotter error estimate repetitions do not match the plan")
        if self.error_estimate.order != self.method.order:
            raise ValueError("Trotter error estimate order does not match the method")
        if self.error_estimate.partition != self.resolved_partition:
            raise ValueError("Trotter error estimate partition does not match the plan")
        if self.error_estimate.group_count != len(self.group_term_indices):
            raise ValueError("Trotter error estimate group count does not match the plan")
        if any(
            group < 0 or group >= len(self.group_term_indices)
            for group, _ in self.suzuki_factors
        ):
            raise ValueError("Suzuki factors must reference resolved Trotter groups")

    @property
    def family(self) -> str:
        return self.method.family

    @property
    def selected_parameters(self) -> dict[str, object]:
        return {
            "trotter_order": self.method.order,
            "trotter_reps": self.repetitions,
            "trotter_partition": self.resolved_partition,
            "trotter_group_count": len(self.group_term_indices),
        }

    @property
    def logical_counts(self) -> LogicalOperationCounts:
        per_step = sum(
            len(self.group_term_indices[group]) for group, _ in self.suzuki_factors
        )
        pauli_evolutions = self.repetitions * per_step
        return LogicalOperationCounts(
            totals=(
                ("suzuki_step", self.repetitions),
                ("pauli_evolution", pauli_evolutions),
            ),
            per_segment=(("pauli_evolution", per_step),),
        )

    @property
    def error_metadata(self) -> dict[str, object]:
        error = self.error_estimate
        target_satisfied = error.rigorous and error.error <= self.error_budget.algorithm_error
        return {
            "bound_value": error.error,
            "bound_prefactor": error.prefactor,
            "bound_method": error.method,
            "bound_rigorous": error.rigorous,
            "bound_scope": "implemented-product-formula",
            "bound_target_satisfied": target_satisfied,
            "circuit_bound_scope": "implemented-product-formula",
            "circuit_bound_rigorous": error.rigorous,
            "circuit_target_satisfied": target_satisfied,
        }


@dataclass(frozen=True)
class MPFPlan:
    hamiltonian: PauliHamiltonian
    method: MultiproductMethod
    time: float
    error_budget: ErrorBudget
    segments: int
    exponents: tuple[int, ...]
    coefficients: tuple[float, ...]
    lcu_structure: MPFLCUStructure
    base_formula_group_term_indices: tuple[tuple[int, ...], ...]
    oaa_rounds_per_segment: int
    error_estimate: MPFErrorEstimate

    def __post_init__(self) -> None:
        if self.segments < 1:
            raise ValueError("MPF segments must be positive")
        if self.error_estimate.segments != self.segments:
            raise ValueError("MPF error estimate segments do not match the plan")
        if self.error_estimate.m != self.method.term_count:
            raise ValueError("MPF error estimate term count does not match the method")
        if self.error_estimate.schedule != self.method.schedule:
            raise ValueError("MPF error estimate schedule does not match the method")
        if self.error_estimate.exponents != self.exponents:
            raise ValueError("MPF error estimate exponents do not match the plan")
        expected_indices = tuple((index,) for index in range(self.hamiltonian.term_count))
        if self.base_formula_group_term_indices != expected_indices:
            raise ValueError("MPF base formulas must use ordered individual Pauli terms")
        if len(self.coefficients) != len(self.exponents):
            raise ValueError("MPF coefficients and exponents must have equal length")
        if self.oaa_rounds_per_segment != 1:
            raise ValueError("the implemented MPF plan requires one OAA round per segment")

    @property
    def family(self) -> str:
        return self.method.family

    @property
    def step_time(self) -> float:
        return self.time / self.segments

    @property
    def selected_parameters(self) -> dict[str, object]:
        return {
            "mpf_m": self.method.term_count,
            "mpf_segments": self.segments,
            "mpf_schedule": self.method.schedule,
            "mpf_error_method": self.method.error_method,
            "mpf_exponents": self.exponents,
        }

    @property
    def logical_counts(self) -> LogicalOperationCounts:
        exponent_sum = sum(self.exponents)
        term_occurrences = (2 * self.hamiltonian.term_count - 1) * exponent_sum
        per_segment = (
            ("prepare", 6),
            ("select", 3),
            ("good_reflection", 2),
            ("controlled_s2", 3 * exponent_sum),
            ("pauli_evolution", 3 * term_occurrences),
        )
        return LogicalOperationCounts(
            totals=tuple((name, self.segments * count) for name, count in per_segment),
            per_segment=per_segment,
        )

    @property
    def error_metadata(self) -> dict[str, object]:
        error = self.error_estimate
        target_satisfied = error.rigorous and error.error <= self.error_budget.algorithm_error
        return {
            "bound_value": error.error,
            "bound_prefactor": error.prefactor,
            "bound_method": error.method,
            "bound_reference": error.reference,
            "bound_theorem_or_equations": error.theorem_or_equations,
            "bound_components": error.bound_components,
            "bound_rigorous": error.rigorous,
            "bound_scope": error.scope,
            "bound_target_satisfied": target_satisfied,
            "circuit_bound_scope": error.circuit_scope,
            "circuit_bound_rigorous": error.circuit_rigorous,
            "circuit_target_satisfied": False,
            "hamiltonian_decomposition": error.hamiltonian_decomposition,
            "bound_assumptions": error.assumptions,
            "bound_fallback_reason": error.fallback_reason,
            "max_nested_commutator_order": error.max_nested_commutator_order,
            "max_exact_nested_commutator_order": error.max_exact_nested_commutator_order,
            "locality_compatible": error.locality_compatible,
            "commutator_bounds": error.commutator_bounds,
        }


@dataclass(frozen=True)
class QSVTResponse:
    component: str
    degree: int
    parity: str

    def __post_init__(self) -> None:
        if self.component not in {"cosine", "sine"}:
            raise ValueError("QSVT response component must be cosine or sine")
        if self.parity not in {"even", "odd"} or self.degree < 0:
            raise ValueError("QSVT response degree/parity is invalid")

    @property
    def projector_phase_slots(self) -> int:
        return self.degree + 1


@dataclass(frozen=True)
class QSVTPlan:
    hamiltonian: PauliHamiltonian
    method: QSVTMethod
    time: float
    error_budget: ErrorBudget
    degree: int
    truncation_order: int
    responses: tuple[QSVTResponse, QSVTResponse]
    base_lcu_uses: int
    oaa_rounds: int

    def __post_init__(self) -> None:
        if self.degree < 1 or self.degree % 2 != 1:
            raise ValueError("QSVT plan degree must be a positive odd integer")
        if self.truncation_order != (self.degree - 1) // 2:
            raise ValueError("QSVT truncation order does not match the degree")
        cosine, sine = self.responses
        if (cosine.component, cosine.degree, cosine.parity) != (
            "cosine",
            self.degree - 1,
            "even",
        ):
            raise ValueError("QSVT cosine response does not match the selected degree")
        if (sine.component, sine.degree, sine.parity) != ("sine", self.degree, "odd"):
            raise ValueError("QSVT sine response does not match the selected degree")
        if self.base_lcu_uses != 3 or self.oaa_rounds != 1:
            raise ValueError("the implemented QSVT plan requires one three-use OAA round")

    @property
    def family(self) -> str:
        return self.method.family

    @property
    def cosine_degree(self) -> int:
        return self.responses[0].degree

    @property
    def sine_degree(self) -> int:
        return self.responses[1].degree

    @property
    def source_error_budget(self) -> float:
        return self.error_budget.algorithm_error / 18

    @property
    def polynomial_scale(self) -> float:
        return 1 - self.source_error_budget

    @property
    def selected_parameters(self) -> dict[str, object]:
        return {
            "qsvt_degree": self.degree,
            "qsvt_cosine_degree": self.cosine_degree,
            "qsvt_sine_degree": self.sine_degree,
        }

    @property
    def logical_counts(self) -> LogicalOperationCounts:
        query_slots = self.base_lcu_uses * (self.cosine_degree + self.sine_degree)
        projector_slots = self.base_lcu_uses * sum(
            response.projector_phase_slots for response in self.responses
        )
        return LogicalOperationCounts(
            totals=(
                ("base_hamsim_lcu", self.base_lcu_uses),
                ("response_pair", 2 * self.base_lcu_uses),
                ("block_encoding_query_slot", query_slots),
                ("prepare_slot", 2 * query_slots),
                ("select_slot", query_slots),
                ("projector_phase_slot", projector_slots),
                ("good_reflection", 2),
            )
        )

    @property
    def error_metadata(self) -> dict[str, object]:
        return {
            "bound_value": self.error_budget.algorithm_error,
            "bound_prefactor": None,
            "bound_method": "jacobi-anger-truncation",
            "bound_rigorous": True,
            "bound_scope": "implemented-algorithm",
            "bound_target_satisfied": True,
            "circuit_bound_scope": "implemented-algorithm",
            "circuit_bound_rigorous": True,
            "circuit_target_satisfied": True,
        }


SimulationPlan: TypeAlias = TrotterPlan | MPFPlan | QSVTPlan


def _canonical_hamiltonian(hamiltonian: PauliHamiltonian) -> PauliHamiltonian:
    if not isinstance(hamiltonian, PauliHamiltonian):
        raise TypeError("hamiltonian must be a PauliHamiltonian")
    return PauliHamiltonian(
        hamiltonian.num_qubits,
        tuple(hamiltonian.terms),
        hamiltonian.name,
    )


def plan_simulation(
    hamiltonian: PauliHamiltonian,
    method: MethodSpec,
    time: float,
    target_error: float,
    *,
    synthesis_error_fraction: float = 0.1,
    trotter_partition: TrotterPartition = "auto",
) -> SimulationPlan:
    """Select parameters once and return the complete logical algorithm plan."""
    if not np.isfinite(time) or float(time) <= 0:
        raise ValueError("resource-planning time must be positive and finite")
    if not isinstance(method, (TrotterMethod, MultiproductMethod, QSVTMethod)):
        raise TypeError("method must be a Hamiltonian-simulation method specification")
    method.validate()
    canonical = _canonical_hamiltonian(hamiltonian)
    evolution_time = float(time)
    budget = ErrorBudget(target_error, synthesis_error_fraction)

    if isinstance(method, TrotterMethod):
        structure = resolve_trotter_structure(canonical, method.order, trotter_partition)
        one_step = estimate_suzuki_error(
            canonical,
            evolution_time,
            reps=1,
            order=method.order,
            partition=trotter_partition,
        )
        repetitions = max(
            1,
            math.ceil((one_step.error / budget.algorithm_error) ** (1 / method.order)),
        )
        selected_error = estimate_suzuki_error(
            canonical,
            evolution_time,
            reps=repetitions,
            order=method.order,
            partition=trotter_partition,
        )
        return TrotterPlan(
            hamiltonian=canonical,
            method=method,
            time=evolution_time,
            error_budget=budget,
            repetitions=repetitions,
            requested_partition=trotter_partition,
            resolved_partition=structure.partition,
            group_term_indices=structure.group_term_indices,
            suzuki_factors=suzuki_group_factors(len(structure.group_term_indices), method.order),
            error_estimate=selected_error,
        )

    if isinstance(method, MultiproductMethod):
        selected_error = select_mpf_segments(
            canonical,
            evolution_time,
            budget.algorithm_error,
            method.term_count,
            schedule=method.schedule,
            method=method.error_method,
        )
        exponents = selected_error.exponents
        coefficients = tuple(
            float(value)
            for value in multiproduct_coefficients(
                method.term_count,
                schedule=method.schedule,
            )
        )
        return MPFPlan(
            hamiltonian=canonical,
            method=method,
            time=evolution_time,
            error_budget=budget,
            segments=selected_error.segments,
            exponents=exponents,
            coefficients=coefficients,
            lcu_structure=mpf_lcu_structure(method.term_count, schedule=method.schedule),
            base_formula_group_term_indices=tuple(
                (index,) for index in range(canonical.term_count)
            ),
            oaa_rounds_per_segment=1,
            error_estimate=selected_error,
        )

    if canonical.alpha <= 0:
        raise ValueError("QSVT resource planning requires a nonzero Hamiltonian L1 norm")
    degree = estimate_qsvt_degree(canonical.alpha * evolution_time, budget.algorithm_error)
    return QSVTPlan(
        hamiltonian=canonical,
        method=method,
        time=evolution_time,
        error_budget=budget,
        degree=degree,
        truncation_order=(degree - 1) // 2,
        responses=(
            QSVTResponse("cosine", degree - 1, "even"),
            QSVTResponse("sine", degree, "odd"),
        ),
        base_lcu_uses=3,
        oaa_rounds=1,
    )
