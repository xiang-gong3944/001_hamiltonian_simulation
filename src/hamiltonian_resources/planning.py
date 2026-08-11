"""Single-pass parameter selection into immutable algorithm plans."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from ._commutator_execution import (
    CommutatorExecution,
    CommutatorProgressCallback,
    execution_scope,
)
from .error_models import (
    AssumptionRecord,
    ErrorAnalysis,
    ErrorClaim,
    ErrorComponent,
    EstimateSupport,
    FallbackRecord,
    MPFSizingEstimate,
    ReferenceRecord,
    SupportedClaim,
    assess_claim,
    good_subspace_leakage_bound,
    oaa_good_block_error_bound,
    repeated_block_encoding_error_bound,
)
from .empirical import (
    EmpiricalErrorEstimate,
    EmpiricalCalibrationKey,
    default_empirical_calibrations,
    select_empirical_segments,
)
from .hamiltonians import PauliHamiltonian
from .method_specs import MethodSpec, MultiproductMethod, QSVTMethod, TrotterMethod
from .multiproduct import (
    MPFBranchCountSelection,
    MPFErrorEstimate,
    MPFScheduleCost,
    MPFLCUStructure,
    mpf_exponent_cost,
    mpf_lcu_structure,
    multiproduct_coefficients,
    resolve_mpf_branch_count,
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


def _empirical_runtime_metadata(error: EmpiricalErrorEstimate) -> dict[str, object]:
    """Expose the reviewed coefficient model and its finite review domain."""
    calibration = error.calibration
    return {
        "empirical_calibration_id": error.calibration_id,
        "empirical_calibration_model": calibration.key.model,
        "empirical_calibration_schema_version": calibration.schema_version,
        "empirical_coefficient_model": calibration.coefficient.model_name,
        "empirical_coefficient_parameters_json": json.dumps(
            dict(calibration.coefficient.parameters),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "empirical_coefficient_value": error.prefactor,
        "empirical_calibration_size_min": calibration.size_range[0],
        "empirical_calibration_size_max": calibration.size_range[1],
        "empirical_reviewed_size_max": calibration.reviewed_size_max,
        "empirical_calibration_time_min": calibration.time_range[0],
        "empirical_calibration_time_max": calibration.time_range[1],
        "empirical_size_extrapolated": error.size_extrapolated,
        "empirical_time_extrapolated": error.time_extrapolated,
        "empirical_active_constraint": error.active_constraint,
        "empirical_stability_diagnostics_json": json.dumps(
            dict(calibration.stability_diagnostics),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "empirical_external_validation_sizes_json": json.dumps(
            calibration.external_validation_sizes,
            separators=(",", ":"),
        ),
        "empirical_external_validation_status": (
            calibration.external_validation_status
        ),
        "empirical_precision_backend": calibration.precision_backend,
        "empirical_precision_digits": calibration.precision_digits,
    }


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
    error_estimate: SuzukiErrorEstimate | EmpiricalErrorEstimate

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("Trotter repetitions must be positive")
        flattened = tuple(index for group in self.group_term_indices for index in group)
        if sorted(flattened) != list(range(self.hamiltonian.term_count)):
            raise ValueError("Trotter groups must partition Hamiltonian term indices")
        estimated_repetitions = (
            self.error_estimate.segments
            if isinstance(self.error_estimate, EmpiricalErrorEstimate)
            else self.error_estimate.reps
        )
        if estimated_repetitions != self.repetitions:
            raise ValueError("Trotter error estimate repetitions do not match the plan")
        estimated_order = (
            self.error_estimate.formal_order
            if isinstance(self.error_estimate, EmpiricalErrorEstimate)
            else self.error_estimate.order
        )
        if estimated_order != self.method.order:
            raise ValueError("Trotter error estimate order does not match the method")
        if isinstance(self.error_estimate, EmpiricalErrorEstimate):
            if self.error_estimate.calibration.key.method != "trotter":
                raise ValueError("Trotter plan requires a Trotter empirical calibration")
            if self.error_estimate.calibration.key.partition != self.resolved_partition:
                raise ValueError("Trotter empirical partition does not match the plan")
        else:
            if self.error_estimate.partition != self.resolved_partition:
                raise ValueError("Trotter error estimate partition does not match the plan")
            if self.error_estimate.group_count != len(self.group_term_indices):
                raise ValueError("Trotter error estimate group count does not match the plan")
        if any(
            group < 0 or group >= len(self.group_term_indices) for group, _ in self.suzuki_factors
        ):
            raise ValueError("Suzuki factors must reference resolved Trotter groups")

    @property
    def family(self) -> str:
        return self.method.family

    @property
    def selected_parameters(self) -> dict[str, object]:
        result: dict[str, object] = {
            "trotter_order": self.method.order,
            "trotter_reps": self.repetitions,
            "trotter_partition": self.resolved_partition,
            "trotter_group_count": len(self.group_term_indices),
            "trotter_error_policy": self.method.error_policy,
        }
        if isinstance(self.error_estimate, EmpiricalErrorEstimate):
            result["empirical_calibration_id"] = self.error_estimate.calibration_id
        return result

    @property
    def logical_counts(self) -> LogicalOperationCounts:
        per_step = sum(len(self.group_term_indices[group]) for group, _ in self.suzuki_factors)
        pauli_evolutions = self.repetitions * per_step
        return LogicalOperationCounts(
            totals=(
                ("suzuki_step", self.repetitions),
                ("pauli_evolution", pauli_evolutions),
            ),
            per_segment=(("pauli_evolution", per_step),),
        )

    @property
    def error_analysis(self) -> ErrorAnalysis:
        from .error_models import SuzukiSizingEstimate

        error = self.error_estimate
        scope = "implemented-product-formula"
        empirical = isinstance(error, EmpiricalErrorEstimate)
        category = "empirical" if empirical else ("analytical" if error.rigorous else "proxy")
        certification = "rigorous" if error.rigorous else "nonrigorous"
        sizing = SuzukiSizingEstimate(
            value=error.error,
            method=error.method,
            category=category,
            certification=certification,
            quantity="evolution-operator approximation error",
            metric="operator-2-norm",
            scope=scope,
            target=self.error_budget.algorithm_error,
            repetitions=self.repetitions,
            order=self.method.order,
            prefactor=error.prefactor,
            partition=self.resolved_partition,
            group_count=len(self.group_term_indices),
            calibration_id=(error.calibration_id if empirical else None),
            calibration_size_extrapolated=(error.size_extrapolated if empirical else False),
            calibration_time_extrapolated=(error.time_extrapolated if empirical else False),
            active_constraint=(error.active_constraint if empirical else None),
        )
        fallback = None
        if not empirical and error.method == "alpha-proxy":
            reason = (
                "commutator evaluation exceeds the configured practical cap"
                if error.order in {4, 6}
                else "no rigorous estimator is implemented for this Suzuki order"
            )
            fallback = FallbackRecord(
                requested_method="higher-order-commutator-bound",
                used_method="alpha-proxy",
                reason=reason,
            )
        references = ()
        if empirical:
            references = (
                ReferenceRecord(
                    error.calibration.reference,
                    f"reviewed calibration {error.calibration_id}; "
                    f"artifact {error.calibration.source}",
                ),
            )
        elif error.method == "childs-commutator":
            references = (
                ReferenceRecord(
                    "Childs, Su, Tran, Wiebe, and Zhu, arXiv:1912.08854",
                    "Propositions 9--10",
                    "https://arxiv.org/abs/1912.08854",
                ),
            )
        elif error.method == "schubert-mendl-commutator":
            references = (
                ReferenceRecord(
                    "Schubert and Mendl, arXiv:2306.10603",
                    "Theorem 1",
                    "https://arxiv.org/abs/2306.10603",
                ),
            )
        assumptions = [
            AssumptionRecord(
                "the resolved ordered Hamiltonian groups match the implemented formula",
                True,
            ),
        ]
        if empirical:
            assumptions.extend(
                (
                    AssumptionRecord("the fixed formal-order powers in time and segments apply", None),
                    AssumptionRecord("the reviewed affine size calibration extrapolates as flagged", None),
                )
            )
        support = EstimateSupport(
            components=(ErrorComponent("prefactor", error.prefactor, "operator-error prefactor"),),
            assumptions=tuple(assumptions),
            references=references,
            fallback=fallback,
        )
        claim = (
            ErrorClaim(
                value=error.error,
                category="analytical",
                certification="rigorous",
                quantity="evolution-operator approximation error",
                metric="operator-2-norm",
                scope=scope,
            )
            if error.rigorous
            else None
        )
        claims = (
            (
                SupportedClaim(
                    claim,
                    error.method,
                    components=support.components,
                    references=support.references,
                    assumptions=support.assumptions,
                ),
            )
            if claim is not None
            else ()
        )
        assessment = assess_claim(claim, self.error_budget.algorithm_error, scope)
        return ErrorAnalysis(
            sizing_estimate=sizing,
            sizing_support=support,
            claims=claims,
            observations=(),
            selection_succeeded=True,
            ideal_algorithm_target=assessment,
            implemented_circuit_target=assessment,
        )

    @property
    def error_metadata(self) -> dict[str, object]:
        error = self.error_estimate
        analysis = self.error_analysis
        empirical = isinstance(error, EmpiricalErrorEstimate)
        result = {
            "estimate_category": "empirical" if empirical else (
                "analytical" if error.rigorous else "proxy"
            ),
            "error_policy": self.method.error_policy,
            "bound_value": error.error,
            "bound_prefactor": error.prefactor,
            "bound_method": error.method,
            "bound_rigorous": error.rigorous,
            "bound_scope": "implemented-product-formula",
            "bound_target_satisfied": analysis.ideal_algorithm_target_certified,
            "circuit_bound_scope": "implemented-product-formula",
            "circuit_bound_rigorous": error.rigorous,
            "circuit_target_satisfied": analysis.implemented_circuit_target_certified,
            "bound_reference": (
                analysis.sizing_support.references[0].citation
                if analysis.sizing_support.references
                else None
            ),
            "bound_theorem_or_equations": (
                analysis.sizing_support.references[0].locator
                if analysis.sizing_support.references
                else None
            ),
            "bound_components": tuple(
                (component.name, component.value)
                for component in analysis.sizing_support.components
            ),
            "bound_assumptions": tuple(
                assumption.description for assumption in analysis.sizing_support.assumptions
            ),
            "bound_fallback_reason": (
                analysis.sizing_support.fallback.reason
                if analysis.sizing_support.fallback is not None
                else None
            ),
        }
        if empirical:
            result.update(_empirical_runtime_metadata(error))
        return result


@dataclass(frozen=True)
class MPFImplementationData:
    """Explicit registered schedule data required for circuit construction."""

    exponents: tuple[int, ...]
    coefficients: tuple[float, ...]
    lcu_structure: MPFLCUStructure

    def __post_init__(self) -> None:
        if not self.exponents or len(self.coefficients) != len(self.exponents):
            raise ValueError("explicit MPF exponents and coefficients must align")
        if self.lcu_structure.physical_branch_count != len(self.exponents):
            raise ValueError("MPF LCU structure does not match its explicit schedule")


@dataclass(frozen=True)
class MPFPlan:
    hamiltonian: PauliHamiltonian
    method: MultiproductMethod
    time: float
    error_budget: ErrorBudget
    branch_count_selection: MPFBranchCountSelection
    segments: int
    schedule_cost: MPFScheduleCost
    implementation: MPFImplementationData | None
    base_formula_group_term_indices: tuple[tuple[int, ...], ...]
    oaa_rounds_per_segment: int
    error_estimate: MPFErrorEstimate | EmpiricalErrorEstimate

    def __post_init__(self) -> None:
        if self.segments < 1:
            raise ValueError("MPF segments must be positive")
        if self.error_estimate.segments != self.segments:
            raise ValueError("MPF error estimate segments do not match the plan")
        selection = self.branch_count_selection
        if selection.policy != self.method.branch_count_policy:
            raise ValueError("MPF branch-count selection policy does not match the method")
        if selection.schedule != self.method.schedule:
            raise ValueError("MPF branch-count selection schedule does not match the method")
        if selection.num_qubits != self.hamiltonian.num_qubits:
            raise ValueError("MPF branch-count selection system size does not match the plan")
        if selection.time != abs(self.time):
            raise ValueError("MPF branch-count selection time does not match the plan")
        if selection.target_error != self.error_budget.algorithm_error:
            raise ValueError("MPF branch-count selection target does not match the algorithm budget")
        if self.method.term_count is not None and selection.term_count != self.method.term_count:
            raise ValueError("fixed MPF term count does not match the branch-count selection")
        if self.schedule_cost.term_count != selection.term_count:
            raise ValueError("MPF schedule cost does not match the resolved branch count")
        if self.schedule_cost.schedule != self.method.schedule:
            raise ValueError("MPF schedule cost does not match the method")
        if isinstance(self.error_estimate, EmpiricalErrorEstimate):
            calibration = self.error_estimate.calibration
            if calibration.key.method != "multiproduct":
                raise ValueError("MPF plan requires an MPF empirical calibration")
            if calibration.key.formal_order != selection.formal_order:
                raise ValueError("MPF empirical calibration order does not match the plan")
            if calibration.key.schedule != self.method.schedule:
                raise ValueError("MPF empirical schedule does not match the plan")
        else:
            if self.error_estimate.m != selection.term_count:
                raise ValueError("MPF error estimate term count does not match the selection")
            if self.error_estimate.schedule != self.method.schedule:
                raise ValueError("MPF error estimate schedule does not match the method")
            if self.error_estimate.exponents != self.exponents:
                raise ValueError("MPF error estimate exponents do not match the plan")
        expected_indices = tuple((index,) for index in range(self.hamiltonian.term_count))
        if self.base_formula_group_term_indices != expected_indices:
            raise ValueError("MPF base formulas must use ordered individual Pauli terms")
        if self.implementation is None:
            if self.schedule_cost.explicit_schedule_available:
                raise ValueError("registered MPF schedule cost requires implementation data")
            if not isinstance(self.error_estimate, EmpiricalErrorEstimate):
                raise ValueError("aggregate-only MPF plans require empirical sizing")
        elif self.implementation.exponents != self.schedule_cost.exponents:
            raise ValueError("MPF implementation does not match its schedule cost")
        if self.oaa_rounds_per_segment != 1:
            raise ValueError("the implemented MPF plan requires one OAA round per segment")

    @property
    def family(self) -> str:
        return self.method.family

    @property
    def step_time(self) -> float:
        return self.time / self.segments

    @property
    def term_count(self) -> int:
        """Resolved MPF branch count ``J`` used by this plan."""
        return self.branch_count_selection.term_count

    @property
    def formal_order(self) -> int:
        return self.branch_count_selection.formal_order

    @property
    def exponents(self) -> tuple[int, ...] | None:
        return self.implementation.exponents if self.implementation is not None else None

    @property
    def coefficients(self) -> tuple[float, ...] | None:
        return self.implementation.coefficients if self.implementation is not None else None

    @property
    def lcu_structure(self) -> MPFLCUStructure | None:
        return self.implementation.lcu_structure if self.implementation is not None else None

    @property
    def selected_parameters(self) -> dict[str, object]:
        return {
            "mpf_m": self.term_count,
            "mpf_branch_count": self.term_count,
            "mpf_formal_order": self.formal_order,
            "mpf_branch_count_policy": self.branch_count_selection.policy,
            "mpf_segments": self.segments,
            "mpf_schedule": self.method.schedule,
            "mpf_error_method": self.method.error_method,
            "mpf_exponents": self.exponents,
            "mpf_exponent_sum": self.schedule_cost.exponent_sum,
            "mpf_exponent_sum_source": self.schedule_cost.source,
            "mpf_explicit_schedule_available": (
                self.schedule_cost.explicit_schedule_available
            ),
        }

    @property
    def logical_counts(self) -> LogicalOperationCounts:
        exponent_sum = self.schedule_cost.exponent_sum
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
    def error_analysis(self) -> ErrorAnalysis:
        error = self.error_estimate
        if isinstance(error, EmpiricalErrorEstimate):
            ideal_scope = "ideal-mpf"
            sizing = MPFSizingEstimate(
                value=error.error,
                method=error.method,
                category="empirical",
                certification="nonrigorous",
                quantity="evolution-operator approximation error",
                metric="operator-2-norm",
                scope=ideal_scope,
                target=self.error_budget.algorithm_error,
                segments=error.segments,
                term_count=self.term_count,
                formal_order=self.formal_order,
                branch_count_policy=self.branch_count_selection.policy,
                branch_count_policy_extensiveness_g=(
                    self.branch_count_selection.extensiveness_g
                ),
                branch_count_policy_target_error=self.branch_count_selection.target_error,
                schedule=self.method.schedule,
                exponents=self.exponents,
                coefficient_l1_norm=(
                    self.lcu_structure.coefficient_l1_norm
                    if self.lcu_structure is not None
                    else None
                ),
                exponent_sum=self.schedule_cost.exponent_sum,
                exponent_sum_source=self.schedule_cost.source,
                explicit_schedule_available=self.schedule_cost.explicit_schedule_available,
                calibration_id=error.calibration_id,
                calibration_size_extrapolated=error.size_extrapolated,
                calibration_time_extrapolated=error.time_extrapolated,
                active_constraint=error.active_constraint,
            )
            support = EstimateSupport(
                components=(
                    ErrorComponent(
                        "empirical-coefficient",
                        error.prefactor,
                        "operator-error prefactor",
                    ),
                ),
                references=(
                    ReferenceRecord(
                        error.calibration.reference,
                        f"reviewed calibration {error.calibration_id}; "
                        f"artifact {error.calibration.source}",
                    ),
                ),
                assumptions=(
                    AssumptionRecord(
                        "the fixed formal-order powers in time and segments apply",
                        None,
                    ),
                    AssumptionRecord(
                        "the empirical estimate concerns the repeated ideal MPF operator",
                        None,
                    ),
                    AssumptionRecord(
                        "the reviewed affine size calibration extrapolates as flagged",
                        None,
                    ),
                ),
            )
            return ErrorAnalysis(
                sizing_estimate=sizing,
                sizing_support=support,
                claims=(),
                observations=(),
                selection_succeeded=True,
                ideal_algorithm_target=assess_claim(
                    None,
                    self.error_budget.algorithm_error,
                    ideal_scope,
                ),
                implemented_circuit_target=assess_claim(
                    None,
                    self.error_budget.algorithm_error,
                    "repeated-shared-ancilla-good-block",
                ),
            )
        ideal_scope = "ideal-mpf"
        local_scope = "one-segment-ideal-mpf"
        amplified_scope = "one-segment-amplified-good-block"
        circuit_scope = "repeated-shared-ancilla-good-block"
        certification = "rigorous" if error.rigorous else "nonrigorous"
        sizing = MPFSizingEstimate(
            value=error.error,
            method=error.method,
            category="analytical" if error.rigorous else "proxy",
            certification=certification,
            quantity="evolution-operator approximation error",
            metric="operator-2-norm",
            scope=ideal_scope,
            target=self.error_budget.algorithm_error,
            segments=error.segments,
            term_count=error.m,
            formal_order=self.formal_order,
            branch_count_policy=self.branch_count_selection.policy,
            branch_count_policy_extensiveness_g=(
                self.branch_count_selection.extensiveness_g
            ),
            branch_count_policy_target_error=self.branch_count_selection.target_error,
            schedule=error.schedule,
            exponents=error.exponents,
            coefficient_l1_norm=error.coefficient_l1_norm,
            exponent_sum=self.schedule_cost.exponent_sum,
            exponent_sum_source=self.schedule_cost.source,
            explicit_schedule_available=self.schedule_cost.explicit_schedule_available,
        )
        components = tuple(
            ErrorComponent(name, value, name.replace("_", " "))
            for name, value in error.bound_components
            if np.isfinite(value) and value >= 0
        )
        reference = ReferenceRecord(
            error.reference,
            error.theorem_or_equations,
        )
        fallback = (
            FallbackRecord(
                requested_method="exact-pauli-commutator-recurrence",
                used_method="rigorous-locality-commutator-bound",
                reason=error.fallback_reason,
            )
            if error.fallback_reason is not None
            else None
        )
        assumptions = tuple(
            AssumptionRecord(description, True if error.rigorous else None)
            for description in error.assumptions
        )
        support = EstimateSupport(
            components=components,
            references=(reference,),
            assumptions=assumptions,
            fallback=fallback,
        )

        claims: list[SupportedClaim] = []
        ideal_claim = None
        if error.rigorous:
            ideal_claim = ErrorClaim(
                value=error.error,
                category="analytical",
                certification="rigorous",
                quantity="evolution-operator approximation error",
                metric="operator-2-norm",
                scope=ideal_scope,
            )
            claims.append(
                SupportedClaim(
                    ideal_claim,
                    error.method,
                    components=components,
                    references=(reference,),
                    assumptions=assumptions,
                    warnings=("this claim applies to the repeated ideal MPF operator",),
                )
            )

        circuit_claim = None
        if (
            error.local_error_rigorous
            and error.local_error is not None
            and np.isfinite(error.local_error)
        ):
            local_claim = ErrorClaim(
                value=error.local_error,
                category="analytical",
                certification="rigorous",
                quantity="one-segment evolution-operator approximation error",
                metric="operator-2-norm",
                scope=local_scope,
            )
            claims.append(
                SupportedClaim(
                    local_claim,
                    f"{error.method}-local-step",
                    references=(reference,),
                    assumptions=assumptions,
                )
            )
            amplified_error = oaa_good_block_error_bound(error.local_error)
            distortion = amplified_error - error.local_error
            leakage = good_subspace_leakage_bound(amplified_error)
            amplified_components = [
                ErrorComponent(
                    "mpf-formula-approximation",
                    error.local_error,
                    "one-segment operator error",
                ),
                ErrorComponent(
                    "oaa-unitarity-defect-distortion",
                    distortion,
                    "one-segment operator error",
                ),
            ]
            if leakage is not None:
                amplified_components.append(
                    ErrorComponent(
                        "good-subspace-leakage-amplitude",
                        leakage,
                        "leakage amplitude",
                    )
                )
            amplified_claim = ErrorClaim(
                value=amplified_error,
                category="derived",
                certification="rigorous",
                quantity="amplified good-block approximation error",
                metric="operator-2-norm",
                scope=amplified_scope,
            )
            claims.append(
                SupportedClaim(
                    amplified_claim,
                    "exact-cubic-oaa-unitarity-defect",
                    components=tuple(amplified_components),
                    assumptions=(
                        AssumptionRecord("the unamplified good block is exactly M/2", True),
                        AssumptionRecord("the OAA convention is -U R U-dagger R U", True),
                    ),
                )
            )
            repeated_error = repeated_block_encoding_error_bound(
                amplified_error,
                self.segments,
            )
            if repeated_error is not None:
                circuit_claim = ErrorClaim(
                    value=repeated_error,
                    category="derived",
                    certification="rigorous",
                    quantity="repeated projected good-block approximation error",
                    metric="operator-2-norm",
                    scope=circuit_scope,
                )
                claims.append(
                    SupportedClaim(
                        circuit_claim,
                        "gslw2019-reused-ancilla-product",
                        components=(
                            ErrorComponent(
                                "one-segment-amplified-good-block-error",
                                amplified_error,
                                "operator error",
                            ),
                            ErrorComponent(
                                "repeated-good-block-error",
                                repeated_error,
                                "operator error",
                            ),
                        ),
                        references=(
                            ReferenceRecord(
                                "Gilyen, Su, Low, and Wiebe, arXiv:1806.01838",
                                "Lemma 54 and Corollary 55",
                                "https://arxiv.org/abs/1806.01838",
                            ),
                        ),
                        assumptions=(
                            AssumptionRecord(
                                "each repeated W is a scale-one block encoding of the same unitary step",
                                True,
                            ),
                            AssumptionRecord(
                                "the same branch register is reused by every segment",
                                True,
                            ),
                        ),
                        warnings=(
                            "the claim is for P W^r P, not the full joint unitary",
                            "postselected normalization and success overhead are outside scope",
                        ),
                    )
                )

        ideal_assessment = assess_claim(
            ideal_claim,
            self.error_budget.algorithm_error,
            ideal_scope,
        )
        circuit_assessment = assess_claim(
            circuit_claim,
            self.error_budget.algorithm_error,
            circuit_scope,
        )
        return ErrorAnalysis(
            sizing_estimate=sizing,
            sizing_support=support,
            claims=tuple(claims),
            observations=(),
            selection_succeeded=True,
            ideal_algorithm_target=ideal_assessment,
            implemented_circuit_target=circuit_assessment,
        )

    @property
    def error_metadata(self) -> dict[str, object]:
        error = self.error_estimate
        analysis = self.error_analysis
        if isinstance(error, EmpiricalErrorEstimate):
            return {
                "estimate_category": "empirical",
                "error_policy": self.method.error_method,
                "bound_value": error.error,
                "bound_prefactor": error.prefactor,
                "bound_method": error.method,
                "bound_reference": error.calibration.reference,
                "bound_theorem_or_equations": (
                    f"reviewed calibration {error.calibration_id}; fixed formal powers"
                ),
                "bound_components": (("empirical_coefficient", error.prefactor),),
                "bound_rigorous": False,
                "bound_scope": "ideal-mpf",
                "bound_target_satisfied": False,
                "circuit_bound_value": None,
                "circuit_bound_scope": "repeated-shared-ancilla-good-block",
                "circuit_bound_rigorous": False,
                "circuit_target_satisfied": False,
                "hamiltonian_decomposition": "ordered individual Pauli terms",
                "bound_assumptions": tuple(
                    assumption.description
                    for assumption in analysis.sizing_support.assumptions
                ),
                "bound_fallback_reason": None,
                "max_nested_commutator_order": 0,
                "max_exact_nested_commutator_order": 0,
                "locality_compatible": False,
                "commutator_bounds": (),
                **_empirical_runtime_metadata(error),
                "mpf_exponent_sum": self.schedule_cost.exponent_sum,
                "mpf_exponent_sum_source": self.schedule_cost.source,
                "mpf_explicit_schedule_available": (
                    self.schedule_cost.explicit_schedule_available
                ),
            }
        circuit_entry = analysis.claim_for_scope("repeated-shared-ancilla-good-block")
        return {
            "estimate_category": "analytical" if error.rigorous else "proxy",
            "error_policy": self.method.error_method,
            "bound_value": error.error,
            "bound_prefactor": error.prefactor,
            "bound_method": error.method,
            "bound_reference": error.reference,
            "bound_theorem_or_equations": error.theorem_or_equations,
            "bound_components": error.bound_components,
            "bound_rigorous": error.rigorous,
            "bound_scope": error.scope,
            "bound_target_satisfied": analysis.ideal_algorithm_target_certified,
            "circuit_bound_value": (
                circuit_entry.claim.value if circuit_entry is not None else None
            ),
            "circuit_bound_scope": "repeated-shared-ancilla-good-block",
            "circuit_bound_rigorous": circuit_entry is not None,
            "circuit_target_satisfied": analysis.implemented_circuit_target_certified,
            "hamiltonian_decomposition": error.hamiltonian_decomposition,
            "bound_assumptions": error.assumptions,
            "bound_fallback_reason": error.fallback_reason,
            "max_nested_commutator_order": error.max_nested_commutator_order,
            "max_exact_nested_commutator_order": error.max_exact_nested_commutator_order,
            "locality_compatible": error.locality_compatible,
            "commutator_bounds": error.commutator_bounds,
            "mpf_exponent_sum": self.schedule_cost.exponent_sum,
            "mpf_exponent_sum_source": self.schedule_cost.source,
            "mpf_explicit_schedule_available": (
                self.schedule_cost.explicit_schedule_available
            ),
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
    def error_analysis(self) -> ErrorAnalysis:
        from .error_models import QSVTSizingEstimate
        from .qsvt import qsvt_polynomial_error_bound

        estimate = qsvt_polynomial_error_bound(
            self.hamiltonian.alpha * abs(self.time),
            self.error_budget.algorithm_error,
            self.degree,
        )
        polynomial_scope = "ideal-qsvt-scaled-polynomial"
        ideal_scope = "ideal-qsvt-oaa-good-block"
        circuit_scope = "implemented-qsvt-floating-phase-circuit"
        sizing = QSVTSizingEstimate(
            value=max(estimate.cosine_tail_bound, estimate.sine_tail_bound),
            method="jacobi-anger-parity-tail",
            category="analytical",
            certification="rigorous",
            quantity="component polynomial truncation error",
            metric="uniform-scalar-error",
            scope="ideal-qsvt-jacobi-anger-components",
            target=self.source_error_budget,
            truncation_order=self.truncation_order,
            degree=self.degree,
            cosine_degree=self.cosine_degree,
            sine_degree=self.sine_degree,
            cosine_first_omitted_degree=self.degree + 1,
            sine_first_omitted_degree=self.degree + 2,
            scale=estimate.scale,
            cosine_tail_bound=estimate.cosine_tail_bound,
            sine_tail_bound=estimate.sine_tail_bound,
        )
        reference = ReferenceRecord(
            "Martyn, Rossi, Tan, and Chuang, arXiv:2105.02859",
            "Eqs. (76)--(77)",
            "https://arxiv.org/abs/2105.02859",
        )
        components = (
            ErrorComponent(
                "boundary-scaling-error",
                estimate.scaling_error,
                "uniform scalar error",
            ),
            ErrorComponent(
                "scaled-cosine-tail",
                estimate.scale * estimate.cosine_tail_bound,
                "uniform scalar error",
            ),
            ErrorComponent(
                "scaled-sine-tail",
                estimate.scale * estimate.sine_tail_bound,
                "uniform scalar error",
            ),
        )
        support = EstimateSupport(
            components=components[1:],
            references=(reference,),
            assumptions=(
                AssumptionRecord("the encoded Hamiltonian spectrum lies in [-1, 1]", True),
            ),
        )
        polynomial_claim = ErrorClaim(
            value=estimate.polynomial_error,
            category="derived",
            certification="rigorous",
            quantity="scaled polynomial evolution approximation error",
            metric="operator-2-norm",
            scope=polynomial_scope,
        )
        ideal_claim = ErrorClaim(
            value=estimate.amplified_good_block_error,
            category="derived",
            certification="rigorous",
            quantity="ideal amplified good-block approximation error",
            metric="operator-2-norm",
            scope=ideal_scope,
        )
        claims = (
            SupportedClaim(
                polynomial_claim,
                "scaled-jacobi-anger-polynomial",
                components=components,
                references=(reference,),
                assumptions=support.assumptions,
            ),
            SupportedClaim(
                ideal_claim,
                "exact-cubic-oaa-unitarity-defect",
                components=(
                    ErrorComponent(
                        "scaled-polynomial-error",
                        estimate.polynomial_error,
                        "operator error",
                    ),
                    ErrorComponent(
                        "oaa-unitarity-defect-distortion",
                        estimate.amplified_good_block_error - estimate.polynomial_error,
                        "operator error",
                    ),
                ),
                references=(reference,),
                assumptions=(
                    AssumptionRecord(
                        "the cosine and sine polynomials are implemented exactly",
                        True,
                    ),
                    AssumptionRecord("the unamplified good block is exactly Q/2", True),
                ),
                warnings=(
                    "floating-point phase reconstruction is outside this claim",
                    "the constructed Qiskit circuit is not uniformly certified",
                ),
            ),
        )
        return ErrorAnalysis(
            sizing_estimate=sizing,
            sizing_support=support,
            claims=claims,
            observations=(),
            selection_succeeded=True,
            ideal_algorithm_target=assess_claim(
                ideal_claim,
                self.error_budget.algorithm_error,
                ideal_scope,
            ),
            implemented_circuit_target=assess_claim(
                None,
                self.error_budget.algorithm_error,
                circuit_scope,
            ),
        )

    @property
    def error_metadata(self) -> dict[str, object]:
        analysis = self.error_analysis
        ideal_entry = analysis.claim_for_scope("ideal-qsvt-oaa-good-block")
        if ideal_entry is None:
            raise RuntimeError("QSVT plan is missing its ideal OAA claim")
        return {
            "estimate_category": "analytical",
            "error_policy": "jacobi-anger-rigorous",
            "bound_value": ideal_entry.claim.value,
            "bound_prefactor": None,
            "bound_method": "jacobi-anger-polynomial-oaa",
            "bound_reference": "Martyn et al., arXiv:2105.02859, Eqs. (76)--(77)",
            "bound_theorem_or_equations": "Jacobi--Anger tails plus exact cubic OAA identity",
            "bound_components": tuple(
                (component.name, component.value) for component in ideal_entry.components
            ),
            "bound_rigorous": True,
            "bound_scope": "ideal-qsvt-oaa-good-block",
            "bound_target_satisfied": analysis.ideal_algorithm_target_certified,
            "circuit_bound_scope": "implemented-qsvt-floating-phase-circuit",
            "circuit_bound_rigorous": False,
            "circuit_target_satisfied": False,
            "bound_assumptions": tuple(
                assumption.description for assumption in ideal_entry.assumptions
            ),
            "bound_fallback_reason": None,
        }


SimulationPlan: TypeAlias = TrotterPlan | MPFPlan | QSVTPlan


def _canonical_hamiltonian(hamiltonian: PauliHamiltonian) -> PauliHamiltonian:
    if not isinstance(hamiltonian, PauliHamiltonian):
        raise TypeError("hamiltonian must be a PauliHamiltonian")
    return PauliHamiltonian(
        hamiltonian.num_qubits,
        tuple(hamiltonian.terms),
        hamiltonian.name,
        hamiltonian.model_metadata,
    )


def _plan_simulation(
    hamiltonian: PauliHamiltonian,
    method: MethodSpec,
    time: float,
    target_error: float,
    *,
    synthesis_error_fraction: float = 0.1,
    trotter_partition: TrotterPartition = "auto",
    execution: CommutatorExecution,
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
        if method.error_policy == "empirical-operator-norm":
            key = EmpiricalCalibrationKey.for_hamiltonian(
                canonical,
                method="trotter",
                formal_order=method.order,
                partition=structure.partition,
                formula="repository-suzuki-v1",
            )
            record = default_empirical_calibrations().lookup(key)
            selected_error = select_empirical_segments(
                record,
                canonical.num_qubits,
                evolution_time,
                budget.algorithm_error,
            )
            repetitions = selected_error.segments
        else:
            one_step = estimate_suzuki_error(
                canonical,
                evolution_time,
                reps=1,
                order=method.order,
                partition=trotter_partition,
                workers=execution.workers,
                _execution=execution,
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
                workers=execution.workers,
                _execution=execution,
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
        branch_count_selection = resolve_mpf_branch_count(
            canonical,
            evolution_time,
            budget.algorithm_error,
            policy=method.branch_count_policy,
            term_count=method.term_count,
            schedule=method.schedule,
        )
        term_count = branch_count_selection.term_count
        schedule_cost = mpf_exponent_cost(term_count, schedule=method.schedule)
        if method.error_method == "empirical-operator-norm":
            key = EmpiricalCalibrationKey.for_hamiltonian(
                canonical,
                method="multiproduct",
                formal_order=2 * term_count,
                schedule=method.schedule,
                formula="ordered-individual-pauli-strang-mpf-v1",
            )
            record = default_empirical_calibrations().lookup(key)
            selected_error = select_empirical_segments(
                record,
                canonical.num_qubits,
                evolution_time,
                budget.algorithm_error,
            )
        else:
            if not schedule_cost.explicit_schedule_available:
                raise ValueError(
                    f"resolved MPF J={term_count} for N={canonical.num_qubits} has "
                    f"aggregate {method.schedule!r} schedule cost only; rigorous and "
                    "coefficient-dependent MPF estimators require an explicit "
                    "registered schedule with 2 <= J <= 15"
                )
            selected_error = select_mpf_segments(
                canonical,
                evolution_time,
                budget.algorithm_error,
                term_count,
                schedule=method.schedule,
                method=method.error_method,
                workers=execution.workers,
                _execution=execution,
            )
        implementation = None
        if schedule_cost.exponents is not None:
            coefficients = tuple(
                float(value)
                for value in multiproduct_coefficients(
                    term_count,
                    schedule=method.schedule,
                )
            )
            implementation = MPFImplementationData(
                exponents=schedule_cost.exponents,
                coefficients=coefficients,
                lcu_structure=mpf_lcu_structure(term_count, schedule=method.schedule),
            )
        return MPFPlan(
            hamiltonian=canonical,
            method=method,
            time=evolution_time,
            error_budget=budget,
            branch_count_selection=branch_count_selection,
            segments=selected_error.segments,
            schedule_cost=schedule_cost,
            implementation=implementation,
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


def plan_simulation(
    hamiltonian: PauliHamiltonian,
    method: MethodSpec,
    time: float,
    target_error: float,
    *,
    synthesis_error_fraction: float = 0.1,
    trotter_partition: TrotterPartition = "auto",
    workers: int = 1,
    progress: CommutatorProgressCallback | None = None,
    _execution: CommutatorExecution | None = None,
) -> SimulationPlan:
    """Select parameters once and return the complete logical algorithm plan."""
    with execution_scope(workers, _execution, progress) as execution:
        return _plan_simulation(
            hamiltonian,
            method,
            time,
            target_error,
            synthesis_error_fraction=synthesis_error_fraction,
            trotter_partition=trotter_partition,
            execution=execution,
        )
