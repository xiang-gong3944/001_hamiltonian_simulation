"""Notebook-first analytical resource benchmark API."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd

from ._commutator_execution import (
    CommutatorExecution,
    CommutatorProgressCallback,
)
from ._progress import TqdmProgressRenderer, combine_callbacks
from .evaluation import EvaluationReport, estimate_resources
from .hamiltonians import PauliHamiltonian, heisenberg_chain, transverse_field_ising
from .method_specs import (
    MethodSpec,
    MultiproductMethod,
    QSVTMethod,
    TrotterMethod,
    default_methods,
)
from .planning import MPFPlan, QSVTPlan, TrotterPlan


BenchmarkSweep: TypeAlias = Literal["system-size", "target-error"]
ProgressCallback: TypeAlias = Callable[["BenchmarkProgress"], None]
SCHEMA_VERSION = "2.0"

BENCHMARK_COLUMNS = (
    "schema_version",
    "run_id",
    "generated_at_utc",
    "config_digest",
    "sweep",
    "hamiltonian_model",
    "hamiltonian_name",
    "model_parameters_json",
    "system_qubits",
    "evolution_time",
    "time_scaling_mode",
    "time_scaling_coefficient",
    "target_error",
    "hamiltonian_alpha",
    "hamiltonian_term_count",
    "method_id",
    "method_family",
    "method_label",
    "trotter_order",
    "mpf_term_count",
    "mpf_formal_order",
    "segment_count",
    "mpf_r_error",
    "mpf_r_time_1",
    "mpf_r_time_2",
    "mpf_active_constraints_json",
    "mpf_mu_upper",
    "mpf_truncation_order_p0",
    "mpf_auxiliary_error",
    "mpf_auxiliary_allocation_fraction",
    "mpf_local_commutator_error",
    "mpf_local_truncated_bch_error",
    "mpf_refined_lemma9_remainder",
    "mpf_refined_lemma10_remainder",
    "mpf_total_branchwise_bch_remainder",
    "mpf_local_step_error",
    "mpf_repeated_global_error",
    "mpf_legacy_first_time_limit",
    "mpf_legacy_first_condition_passed",
    "mpf_second_time_limit",
    "mpf_schedule_weights_json",
    "mpf_schedule_weighted_extensiveness",
    "mpf_exact_commutator_cutoff",
    "mpf_locality_fallback",
    "mpf_locality_fallback_reason",
    "mpf_refined_tail_fallback_status",
    "mpf_local_error_dominance",
    "mpf_bound_policy",
    "mpf_bound_candidates_json",
    "query_count",
    "qsvt_degree",
    "trotter_partition",
    "trotter_group_count",
    "bound_value",
    "bound_prefactor",
    "bound_method",
    "bound_reference",
    "bound_theorem_or_equations",
    "bound_components_json",
    "bound_rigorous",
    "bound_scope",
    "bound_target_satisfied",
    "hamiltonian_decomposition",
    "bound_assumptions_json",
    "bound_fallback_reason",
    "max_nested_commutator_order",
    "max_exact_nested_commutator_order",
    "locality_compatible",
    "commutator_cap_fallback",
    "commutator_bounds_json",
    "circuit_bound_scope",
    "circuit_bound_rigorous",
    "circuit_target_satisfied",
    "algorithm_error_budget",
    "mpf_schedule",
    "mpf_exponents_json",
    "mpf_coefficients_json",
    "mpf_coefficient_l1_norm",
    "mpf_padding_weight",
    "mpf_physical_branch_count",
    "mpf_negative_coefficient_count",
    "mpf_padding_branch_count",
    "mpf_sign_branch_count",
    "mpf_active_branch_count",
    "mpf_unused_branch_state_count",
    "mpf_prepare_calls_per_segment",
    "mpf_select_calls_per_segment",
    "mpf_good_reflections_per_segment",
    "mpf_base_lcu_uses_per_segment",
    "lcu_normalization",
    "amplitude_amplification",
    "amplitude_amplification_rounds",
    "good_subspace",
    "nominal_success_probability",
    "total_qubits",
    "rotation_count",
    "toffoli_count",
    "depth",
    "t_count",
    "cnot_count",
    "counting_mode",
    "rotation_synthesis_error",
    "package_version",
    "python_version",
    "qiskit_version",
    "git_commit",
    "git_dirty",
    "status",
    "error_type",
    "error_message",
)

_SCHEMA2_EXTENSION_COLUMNS = {
    "bound_reference",
    "bound_theorem_or_equations",
    "bound_components_json",
    "bound_scope",
    "bound_target_satisfied",
    "hamiltonian_decomposition",
    "bound_assumptions_json",
    "bound_fallback_reason",
    "max_nested_commutator_order",
    "max_exact_nested_commutator_order",
    "locality_compatible",
    "commutator_cap_fallback",
    "commutator_bounds_json",
    "mpf_r_error",
    "mpf_r_time_1",
    "mpf_r_time_2",
    "mpf_active_constraints_json",
    "mpf_mu_upper",
    "mpf_truncation_order_p0",
    "mpf_auxiliary_error",
    "mpf_auxiliary_allocation_fraction",
    "mpf_local_commutator_error",
    "mpf_local_truncated_bch_error",
    "mpf_refined_lemma9_remainder",
    "mpf_refined_lemma10_remainder",
    "mpf_total_branchwise_bch_remainder",
    "mpf_local_step_error",
    "mpf_repeated_global_error",
    "mpf_legacy_first_time_limit",
    "mpf_legacy_first_condition_passed",
    "mpf_second_time_limit",
    "mpf_schedule_weights_json",
    "mpf_schedule_weighted_extensiveness",
    "mpf_exact_commutator_cutoff",
    "mpf_locality_fallback",
    "mpf_locality_fallback_reason",
    "mpf_refined_tail_fallback_status",
    "mpf_local_error_dominance",
    "mpf_bound_policy",
    "mpf_bound_candidates_json",
    "mpf_physical_branch_count",
    "mpf_negative_coefficient_count",
    "mpf_padding_branch_count",
    "mpf_sign_branch_count",
    "mpf_active_branch_count",
    "mpf_unused_branch_state_count",
    "mpf_prepare_calls_per_segment",
    "mpf_select_calls_per_segment",
    "mpf_good_reflections_per_segment",
    "mpf_base_lcu_uses_per_segment",
    "circuit_bound_scope",
    "circuit_bound_rigorous",
    "circuit_target_satisfied",
}
_SCHEMA2_REQUIRED_COLUMNS = tuple(
    column for column in BENCHMARK_COLUMNS if column not in _SCHEMA2_EXTENSION_COLUMNS
)

_MODEL_PARAMETERS = {
    "transverse_field_ising": {"coupling", "field", "periodic"},
    "heisenberg_chain": {"coupling", "field_z"},
}
_MODEL_FACTORIES: dict[str, Callable[..., PauliHamiltonian]] = {
    "transverse_field_ising": transverse_field_ising,
    "heisenberg_chain": heisenberg_chain,
}


@dataclass
class HamiltonianSpec:
    """A stable model name, constructor parameters, and optional Python factory."""

    model: str = "transverse_field_ising"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    factory: Callable[..., PauliHamiltonian] | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        self.parameters = dict(self.parameters)
        if not self.model or not isinstance(self.model, str):
            raise ValueError("hamiltonian model must be a nonempty string")
        if self.factory is None:
            if self.model not in _MODEL_FACTORIES:
                supported = ", ".join(sorted(_MODEL_FACTORIES))
                raise ValueError(
                    f"unknown Hamiltonian model {self.model!r}; registered models: {supported}"
                )
            unknown = set(self.parameters) - _MODEL_PARAMETERS[self.model]
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"unsupported parameters for {self.model}: {names}")
        elif not callable(self.factory):
            raise TypeError("Hamiltonian factory must be callable")

    def build(self, system_qubits: int) -> PauliHamiltonian:
        self.validate()
        constructor = self.factory or _MODEL_FACTORIES[self.model]
        result = constructor(system_qubits, **dict(self.parameters))
        if not isinstance(result, PauliHamiltonian):
            raise TypeError("Hamiltonian factory must return PauliHamiltonian")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "parameters": dict(self.parameters)}


@dataclass
class TimeScaling:
    """Map a system size to physical evolution time."""

    mode: Literal["proportional", "fixed"] = "proportional"
    coefficient: float = 1.0

    def validate(self) -> None:
        if self.mode not in {"proportional", "fixed"}:
            raise ValueError("time mode must be 'proportional' or 'fixed'")
        if (
            isinstance(self.coefficient, bool)
            or not isinstance(self.coefficient, Real)
            or not np.isfinite(self.coefficient)
            or float(self.coefficient) <= 0
        ):
            raise ValueError("time coefficient must be positive and finite")
        self.coefficient = float(self.coefficient)

    def at(self, system_qubits: int) -> float:
        self.validate()
        if self.mode == "proportional":
            return self.coefficient * system_qubits
        return self.coefficient

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "coefficient": self.coefficient}


@dataclass
class BenchmarkConfig:
    """Mutable, notebook-friendly configuration for analytical sweeps."""

    hamiltonian: HamiltonianSpec = field(
        default_factory=lambda: HamiltonianSpec(
            parameters={"coupling": 1.0, "field": 3.0, "periodic": False}
        )
    )
    system_sizes: Sequence[int] = field(default_factory=lambda: [2, 4, 6, 8, 10, 12])
    target_errors: Sequence[float] = field(default_factory=lambda: [0.1, 0.03, 0.01, 0.003, 0.001])
    time: TimeScaling = field(default_factory=TimeScaling)
    fixed_system_size: int = 8
    fixed_target_error: float = 1e-3
    methods: Sequence[MethodSpec] = field(default_factory=default_methods)
    synthesis_error_fraction: float = 0.1
    trotter_partition: Literal["auto", "individual", "commuting"] = "auto"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.hamiltonian, HamiltonianSpec):
            raise TypeError("hamiltonian must be a HamiltonianSpec")
        if not isinstance(self.time, TimeScaling):
            raise TypeError("time must be a TimeScaling")
        self.hamiltonian.validate()
        self.time.validate()
        self.system_sizes = _normalize_sizes(self.system_sizes)
        self.target_errors = _normalize_errors(self.target_errors)
        if (
            isinstance(self.fixed_system_size, bool)
            or not isinstance(self.fixed_system_size, Integral)
            or int(self.fixed_system_size) < 1
        ):
            raise ValueError("fixed_system_size must be a positive integer")
        self.fixed_system_size = int(self.fixed_system_size)
        if not _is_probability(self.fixed_target_error):
            raise ValueError("fixed_target_error must lie in (0, 1)")
        self.fixed_target_error = float(self.fixed_target_error)
        if not _is_probability(self.synthesis_error_fraction):
            raise ValueError("synthesis_error_fraction must lie in (0, 1)")
        self.synthesis_error_fraction = float(self.synthesis_error_fraction)
        if self.trotter_partition not in {"auto", "individual", "commuting"}:
            raise ValueError("trotter_partition must be 'auto', 'individual', or 'commuting'")
        self.methods = list(self.methods)
        if not self.methods:
            raise ValueError("methods must not be empty")
        for method in self.methods:
            if not isinstance(method, (TrotterMethod, MultiproductMethod, QSVTMethod)):
                raise TypeError("methods must contain benchmark method specifications")
            method.validate()
        method_ids = [method.method_id for method in self.methods]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("methods must have unique method IDs")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "hamiltonian": self.hamiltonian.as_dict(),
            "system_sizes": list(self.system_sizes),
            "target_errors": list(self.target_errors),
            "time": self.time.as_dict(),
            "fixed_system_size": self.fixed_system_size,
            "fixed_target_error": self.fixed_target_error,
            "methods": [method.as_dict() for method in self.methods],
            "synthesis_error_fraction": self.synthesis_error_fraction,
            "trotter_partition": self.trotter_partition,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkProgress:
    completed: int
    total: int
    sweep: BenchmarkSweep
    system_qubits: int
    target_error: float
    method_id: str
    status: Literal["ok", "error"]


def _normalize_sizes(values: Sequence[int]) -> list[int]:
    normalized = list(values)
    if not normalized:
        raise ValueError("system_sizes must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1
        for value in normalized
    ):
        raise ValueError("system_sizes must contain positive integers")
    result = [int(value) for value in normalized]
    if len(result) != len(set(result)):
        raise ValueError("system_sizes must not contain duplicates")
    return result


def _is_probability(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and np.isfinite(value)
        and 0 < float(value) < 1
    )


def _normalize_errors(values: Sequence[float]) -> list[float]:
    normalized = list(values)
    if not normalized:
        raise ValueError("target_errors must not be empty")
    if not all(_is_probability(value) for value in normalized):
        raise ValueError("target_errors must contain values in (0, 1)")
    result = [float(value) for value in normalized]
    if len(result) != len(set(result)):
        raise ValueError("target_errors must not contain duplicates")
    return result


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _git_metadata() -> tuple[str, bool | None]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def _software_metadata() -> dict[str, Any]:
    git_commit, git_dirty = _git_metadata()
    return {
        "package_version": _package_version("hamiltonian-resources"),
        "python_version": platform.python_version(),
        "qiskit_version": _package_version("qiskit"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _evaluate_method(
    hamiltonian: PauliHamiltonian,
    config: BenchmarkConfig,
    evolution_time: float,
    target_error: float,
    method: MethodSpec,
    execution: CommutatorExecution,
) -> EvaluationReport:
    return estimate_resources(
        hamiltonian,
        method,
        evolution_time,
        target_error,
        synthesis_error_fraction=config.synthesis_error_fraction,
        trotter_partition=config.trotter_partition,
        workers=execution.workers,
        _execution=execution,
    )


def _report_metadata(report: EvaluationReport) -> dict[str, Any]:
    plan = report.plan
    result = dict(report.error_metadata)
    if isinstance(plan, TrotterPlan):
        result.update(
            segment_count=plan.repetitions,
            trotter_partition=plan.resolved_partition,
            trotter_group_count=len(plan.group_term_indices),
            lcu_normalization=1.0,
            amplitude_amplification="none",
            amplitude_amplification_rounds=0,
            good_subspace="system register",
            nominal_success_probability=1.0,
        )
    elif isinstance(plan, MPFPlan):
        error = plan.error_estimate
        diagnostics = error.segment_diagnostics
        structure = plan.lcu_structure
        per_segment = plan.logical_counts.as_dict()["per_segment"]
        if (
            diagnostics is None
            or diagnostics.local_commutator_error is None
            or diagnostics.total_branchwise_bch_remainder is None
        ):
            local_error_dominance = None
        elif diagnostics.local_commutator_error > diagnostics.total_branchwise_bch_remainder:
            local_error_dominance = "commutator"
        elif diagnostics.local_commutator_error < diagnostics.total_branchwise_bch_remainder:
            local_error_dominance = "bch"
        else:
            local_error_dominance = "tie"
        result.update(
            segment_count=plan.segments,
            mpf_r_error=(diagnostics.r_error if diagnostics is not None else None),
            mpf_r_time_1=(diagnostics.r_time_1 if diagnostics is not None else None),
            mpf_r_time_2=(diagnostics.r_time_2 if diagnostics is not None else None),
            mpf_active_constraints_json=json.dumps(
                diagnostics.active_constraints if diagnostics is not None else (),
                separators=(",", ":"),
            ),
            mpf_mu_upper=(diagnostics.mu_upper if diagnostics is not None else None),
            mpf_truncation_order_p0=(
                diagnostics.truncation_order_p0 if diagnostics is not None else None
            ),
            mpf_auxiliary_error=(
                diagnostics.auxiliary_error if diagnostics is not None else None
            ),
            mpf_auxiliary_allocation_fraction=(
                diagnostics.auxiliary_allocation_fraction
                if diagnostics is not None
                else None
            ),
            mpf_local_commutator_error=(
                diagnostics.local_commutator_error if diagnostics is not None else None
            ),
            mpf_local_truncated_bch_error=(
                diagnostics.local_truncated_bch_error if diagnostics is not None else None
            ),
            mpf_refined_lemma9_remainder=(
                diagnostics.refined_lemma9_remainder if diagnostics is not None else None
            ),
            mpf_refined_lemma10_remainder=(
                diagnostics.refined_lemma10_remainder if diagnostics is not None else None
            ),
            mpf_total_branchwise_bch_remainder=(
                diagnostics.total_branchwise_bch_remainder
                if diagnostics is not None
                else None
            ),
            mpf_local_step_error=(
                diagnostics.local_step_error if diagnostics is not None else None
            ),
            mpf_repeated_global_error=(
                diagnostics.repeated_global_error if diagnostics is not None else None
            ),
            mpf_legacy_first_time_limit=(
                diagnostics.legacy_first_time_limit if diagnostics is not None else None
            ),
            mpf_legacy_first_condition_passed=(
                diagnostics.legacy_first_condition_passed if diagnostics is not None else None
            ),
            mpf_second_time_limit=(
                diagnostics.second_time_limit if diagnostics is not None else None
            ),
            mpf_schedule_weights_json=json.dumps(
                diagnostics.schedule_weights if diagnostics is not None else (),
                separators=(",", ":"),
            ),
            mpf_schedule_weighted_extensiveness=(
                diagnostics.schedule_weighted_extensiveness
                if diagnostics is not None
                else None
            ),
            mpf_exact_commutator_cutoff=(
                diagnostics.max_exact_nested_commutator_order
                if diagnostics is not None
                else None
            ),
            mpf_locality_fallback=(
                diagnostics.used_locality_fallback if diagnostics is not None else None
            ),
            mpf_locality_fallback_reason=(
                diagnostics.locality_fallback_reason if diagnostics is not None else None
            ),
            mpf_refined_tail_fallback_status=(
                diagnostics.refined_tail_fallback_status if diagnostics is not None else None
            ),
            mpf_local_error_dominance=local_error_dominance,
            mpf_bound_policy=(error.requested_method or plan.method.error_method),
            mpf_bound_candidates_json=json.dumps(
                [candidate.as_dict() for candidate in error.bound_candidates],
                sort_keys=True,
                separators=(",", ":"),
            ),
            query_count=plan.logical_counts.as_dict()["totals"]["controlled_s2"],
            bound_components_json=json.dumps(
                dict(error.bound_components), sort_keys=True, separators=(",", ":")
            ),
            bound_assumptions_json=json.dumps(error.assumptions, separators=(",", ":")),
            commutator_cap_fallback=error.fallback_reason is not None,
            commutator_bounds_json=json.dumps(
                dict(error.commutator_bounds), sort_keys=True, separators=(",", ":")
            ),
            mpf_schedule=plan.method.schedule,
            mpf_exponents_json=json.dumps(plan.exponents, separators=(",", ":")),
            mpf_coefficients_json=json.dumps(plan.coefficients, separators=(",", ":")),
            mpf_coefficient_l1_norm=structure.coefficient_l1_norm,
            mpf_padding_weight=structure.padding_weight,
            mpf_physical_branch_count=structure.physical_branch_count,
            mpf_negative_coefficient_count=structure.negative_coefficient_count,
            mpf_padding_branch_count=structure.padding_branch_count,
            mpf_sign_branch_count=structure.sign_branch_count,
            mpf_active_branch_count=structure.active_branch_count,
            mpf_unused_branch_state_count=structure.unused_branch_state_count,
            mpf_prepare_calls_per_segment=per_segment["prepare"],
            mpf_select_calls_per_segment=per_segment["select"],
            mpf_good_reflections_per_segment=per_segment["good_reflection"],
            mpf_base_lcu_uses_per_segment=per_segment["select"],
            lcu_normalization=2.0,
            amplitude_amplification="one robust OAA round per segment",
            amplitude_amplification_rounds=plan.segments,
            good_subspace="branch register all-zero",
            nominal_success_probability=None,
        )
    elif isinstance(plan, QSVTPlan):
        result.update(
            query_count=plan.logical_counts.as_dict()["totals"]["block_encoding_query_slot"],
            qsvt_degree=plan.degree,
            lcu_normalization=2.0,
            amplitude_amplification="one robust OAA round",
            amplitude_amplification_rounds=plan.oaa_rounds,
            good_subspace="component, quadrature, and index registers all-zero",
            nominal_success_probability=None,
            bound_components_json=json.dumps(
                dict(result["bound_components"]),
                sort_keys=True,
                separators=(",", ":"),
            ),
            bound_assumptions_json=json.dumps(
                result["bound_assumptions"],
                separators=(",", ":"),
            ),
        )
    if "bound_components" in result and "bound_components_json" not in result:
        result["bound_components_json"] = json.dumps(
            dict(result["bound_components"]),
            sort_keys=True,
            separators=(",", ":"),
        )
    if "bound_assumptions" in result and "bound_assumptions_json" not in result:
        result["bound_assumptions_json"] = json.dumps(
            result["bound_assumptions"],
            separators=(",", ":"),
        )
    resource = report.resources
    result.update(
        total_qubits=resource.num_qubits,
        rotation_count=resource.rotation_count,
        toffoli_count=resource.toffoli_count,
        depth=resource.depth,
        t_count=resource.t_count,
        cnot_count=resource.cnot_count,
        counting_mode=resource.counting_mode,
        rotation_synthesis_error=resource.rotation_synthesis_error,
        status="ok",
    )
    return result


def _sweep_points(
    config: BenchmarkConfig, sweeps: Sequence[BenchmarkSweep]
) -> list[tuple[BenchmarkSweep, int, float, float]]:
    points: list[tuple[BenchmarkSweep, int, float, float]] = []
    for sweep in sweeps:
        if sweep == "system-size":
            points.extend(
                (
                    sweep,
                    size,
                    config.time.at(size),
                    config.fixed_target_error,
                )
                for size in config.system_sizes
            )
        elif sweep == "target-error":
            points.extend(
                (
                    sweep,
                    config.fixed_system_size,
                    config.time.at(config.fixed_system_size),
                    error,
                )
                for error in config.target_errors
            )
        else:
            raise ValueError(f"unsupported benchmark sweep: {sweep!r}")
    return points


def _base_record(
    config: BenchmarkConfig,
    run_metadata: Mapping[str, Any],
    sweep: BenchmarkSweep,
    system_qubits: int,
    evolution_time: float,
    target_error: float,
    method: MethodSpec,
) -> dict[str, Any]:
    record = {column: None for column in BENCHMARK_COLUMNS}
    record.update(
        schema_version=SCHEMA_VERSION,
        run_id=run_metadata["run_id"],
        generated_at_utc=run_metadata["generated_at_utc"],
        config_digest=config.digest,
        sweep=sweep,
        hamiltonian_model=config.hamiltonian.model,
        model_parameters_json=json.dumps(
            dict(config.hamiltonian.parameters), sort_keys=True, separators=(",", ":")
        ),
        system_qubits=system_qubits,
        evolution_time=evolution_time,
        time_scaling_mode=config.time.mode,
        time_scaling_coefficient=config.time.coefficient,
        target_error=target_error,
        method_id=method.method_id,
        method_family=method.family,
        method_label=method.label,
        trotter_order=method.order if isinstance(method, TrotterMethod) else None,
        mpf_term_count=(method.term_count if isinstance(method, MultiproductMethod) else None),
        mpf_formal_order=(
            2 * method.term_count if isinstance(method, MultiproductMethod) else None
        ),
        algorithm_error_budget=target_error * (1 - config.synthesis_error_fraction),
        **run_metadata["software"],
    )
    return record


def _run_benchmark(
    config: BenchmarkConfig,
    sweeps: Sequence[BenchmarkSweep] | BenchmarkSweep = (
        "system-size",
        "target-error",
    ),
    progress: ProgressCallback | None = None,
    *,
    execution: CommutatorExecution,
) -> pd.DataFrame:
    """Run analytical sweeps in memory without writing files."""
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")
    config.validate()
    selected_sweeps = [sweeps] if isinstance(sweeps, str) else list(sweeps)
    if not selected_sweeps or len(selected_sweeps) != len(set(selected_sweeps)):
        raise ValueError("sweeps must be nonempty and contain no duplicates")
    points = _sweep_points(config, selected_sweeps)
    run_metadata = {
        "run_id": str(uuid.uuid4()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": _software_metadata(),
    }
    total = len(points) * len(config.methods)
    completed = 0
    records: list[dict[str, Any]] = []
    hamiltonians: dict[int, PauliHamiltonian | Exception] = {}
    for sweep, system_qubits, evolution_time, target_error in points:
        if system_qubits not in hamiltonians:
            try:
                hamiltonians[system_qubits] = config.hamiltonian.build(system_qubits)
            except Exception as exc:
                hamiltonians[system_qubits] = exc
        hamiltonian_or_error = hamiltonians[system_qubits]
        for method in config.methods:
            record = _base_record(
                config,
                run_metadata,
                sweep,
                system_qubits,
                evolution_time,
                target_error,
                method,
            )
            try:
                if isinstance(hamiltonian_or_error, Exception):
                    raise hamiltonian_or_error
                hamiltonian = hamiltonian_or_error
                record.update(
                    hamiltonian_name=hamiltonian.name,
                    hamiltonian_alpha=hamiltonian.alpha,
                    hamiltonian_term_count=hamiltonian.term_count,
                )
                report = _evaluate_method(
                    hamiltonian,
                    config,
                    evolution_time,
                    target_error,
                    method,
                    execution,
                )
                record.update(_report_metadata(report))
            except Exception as exc:
                record.update(
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc) or repr(exc),
                )
            records.append(record)
            completed += 1
            if progress is not None:
                progress(
                    BenchmarkProgress(
                        completed=completed,
                        total=total,
                        sweep=sweep,
                        system_qubits=system_qubits,
                        target_error=target_error,
                        method_id=method.method_id,
                        status=record["status"],
                    )
                )
    frame = pd.DataFrame.from_records(records, columns=BENCHMARK_COLUMNS)
    validate_benchmark_frame(frame)
    return frame


def run_benchmark(
    config: BenchmarkConfig,
    sweeps: Sequence[BenchmarkSweep] | BenchmarkSweep = (
        "system-size",
        "target-error",
    ),
    progress: ProgressCallback | None = None,
    *,
    workers: int = 1,
    commutator_progress: CommutatorProgressCallback | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Run analytical sweeps in memory without writing files."""
    renderer = TqdmProgressRenderer() if show_progress else None
    try:
        outer_callback = combine_callbacks(
            progress,
            renderer.benchmark if renderer is not None else None,
        )
        inner_callback = combine_callbacks(
            commutator_progress,
            renderer.commutator if renderer is not None else None,
        )
        with CommutatorExecution(workers, inner_callback) as execution:
            return _run_benchmark(
                config,
                sweeps=sweeps,
                progress=outer_callback,
                execution=execution,
            )
    finally:
        if renderer is not None:
            renderer.close()


def validate_benchmark_frame(frame: pd.DataFrame) -> None:
    """Validate schema-2 data, including files predating scoped MPF metadata."""
    missing = set(_SCHEMA2_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"benchmark data is missing required columns: {names}")
    if frame.empty:
        raise ValueError("benchmark data must not be empty")
    versions = set(frame["schema_version"].astype(str))
    if versions != {SCHEMA_VERSION}:
        found = ", ".join(sorted(versions))
        raise ValueError(
            f"unsupported benchmark schema {found}; schema 1.x is not compatible, "
            f"expected {SCHEMA_VERSION}"
        )
    if not set(frame["sweep"]).issubset({"system-size", "target-error"}):
        raise ValueError("benchmark sweep values are invalid")
    if not set(frame["status"]).issubset({"ok", "error"}):
        raise ValueError("benchmark status must be 'ok' or 'error'")
    successful = frame[frame["status"] == "ok"]
    if successful[["t_count", "cnot_count"]].isna().any().any():
        raise ValueError("successful rows must contain T and CNOT counts")
    if (successful[["t_count", "cnot_count"]] < 0).any().any():
        raise ValueError("resource counts must be nonnegative")
    failed = frame[frame["status"] == "error"]
    if failed["error_message"].isna().any() or (failed["error_message"] == "").any():
        raise ValueError("failed rows must contain an error message")


@dataclass
class BenchmarkJob:
    """JSON/CLI settings kept separate from the in-memory benchmark API."""

    benchmark: BenchmarkConfig
    output_root: Path = Path("benchmark_outputs")
    output_formats: Sequence[str] = ("png", "pdf")
    generate_summary_plots: bool = False

    def validate(self) -> None:
        self.benchmark.validate()
        self.output_root = Path(self.output_root)
        self.output_formats = list(self.output_formats)
        supported = {"png", "pdf", "svg"}
        if not self.output_formats:
            raise ValueError("output_formats must not be empty")
        unknown = set(self.output_formats) - supported
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported output formats: {names}")
        if len(self.output_formats) != len(set(self.output_formats)):
            raise ValueError("output_formats must not contain duplicates")
        if not isinstance(self.generate_summary_plots, bool):
            raise TypeError("generate_summary_plots must be a boolean")


def _method_from_dict(raw: Mapping[str, Any]) -> MethodSpec:
    family = raw.get("family")
    if family == "trotter":
        unknown = set(raw) - {"family", "order"}
        if unknown or "order" not in raw:
            raise ValueError("Trotter method requires only family and order")
        return TrotterMethod(raw["order"])
    if family == "multiproduct":
        unknown = set(raw) - {"family", "term_count", "schedule", "error_method"}
        if unknown or "term_count" not in raw:
            raise ValueError(
                "multiproduct method requires family, term_count, and optional "
                "schedule/error_method"
            )
        return MultiproductMethod(
            raw["term_count"],
            raw.get("schedule", "new"),
            raw.get("error_method", "low2019-l1-ideal-rigorous"),
        )
    if family == "qsvt":
        if set(raw) != {"family"}:
            raise ValueError("QSVT method accepts only the family field")
        return QSVTMethod()
    raise ValueError(f"unknown benchmark method family: {family!r}")


def benchmark_config_from_dict(raw: Mapping[str, Any]) -> BenchmarkConfig:
    """Parse the schema-2 JSON representation of an in-memory configuration."""
    allowed = {
        "hamiltonian",
        "system_sizes",
        "target_errors",
        "time",
        "fixed_system_size",
        "fixed_target_error",
        "methods",
        "synthesis_error_fraction",
        "trotter_partition",
    }
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown benchmark configuration fields: {names}")
    hamiltonian_raw = raw.get("hamiltonian", {})
    time_raw = raw.get("time", {})
    methods_raw = raw.get("methods")
    if not isinstance(hamiltonian_raw, Mapping):
        raise TypeError("hamiltonian configuration must be an object")
    if not isinstance(time_raw, Mapping):
        raise TypeError("time configuration must be an object")
    if methods_raw is not None and (
        isinstance(methods_raw, (str, bytes)) or not isinstance(methods_raw, Sequence)
    ):
        raise TypeError("methods configuration must be an array")
    hamiltonian_unknown = set(hamiltonian_raw) - {"model", "parameters"}
    time_unknown = set(time_raw) - {"mode", "coefficient"}
    if hamiltonian_unknown:
        raise ValueError("unknown Hamiltonian configuration fields")
    if time_unknown:
        raise ValueError("unknown time configuration fields")
    kwargs = dict(raw)
    kwargs["hamiltonian"] = HamiltonianSpec(
        model=hamiltonian_raw.get("model", "transverse_field_ising"),
        parameters=hamiltonian_raw.get(
            "parameters", {"coupling": 1.0, "field": 3.0, "periodic": False}
        ),
    )
    kwargs["time"] = TimeScaling(
        mode=time_raw.get("mode", "proportional"),
        coefficient=time_raw.get("coefficient", 1.0),
    )
    if methods_raw is not None:
        if not all(isinstance(item, Mapping) for item in methods_raw):
            raise TypeError("each method configuration must be an object")
        kwargs["methods"] = [_method_from_dict(item) for item in methods_raw]
    return BenchmarkConfig(**kwargs)


def load_benchmark_job(path: str | Path) -> BenchmarkJob:
    """Load a schema-2 benchmark job used by the CLI."""
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark job must be a JSON object")
    unknown = set(raw) - {"benchmark", "output"}
    if unknown or "benchmark" not in raw:
        raise ValueError("benchmark job requires benchmark and optional output objects")
    output = raw.get("output", {})
    if not isinstance(raw["benchmark"], Mapping) or not isinstance(output, Mapping):
        raise TypeError("benchmark and output settings must be objects")
    output_unknown = set(output) - {
        "root",
        "formats",
        "generate_summary_plots",
    }
    if output_unknown:
        raise ValueError("unknown benchmark output fields")
    output_root = Path(output.get("root", "benchmark_outputs"))
    if not output_root.is_absolute():
        output_root = config_path.parent / output_root
    job = BenchmarkJob(
        benchmark=benchmark_config_from_dict(raw["benchmark"]),
        output_root=output_root.resolve(),
        output_formats=output.get("formats", ["png", "pdf"]),
        generate_summary_plots=output.get("generate_summary_plots", False),
    )
    job.validate()
    return job


def load_benchmark(path: str | Path) -> pd.DataFrame:
    """Load a benchmark CSV; a metadata sidecar is not required."""
    frame = pd.read_csv(path, keep_default_na=True)
    validate_benchmark_frame(frame)
    if _SCHEMA2_EXTENSION_COLUMNS - set(frame.columns):
        is_mpf = frame["method_family"] == "multiproduct"
        is_qsvt = frame["method_family"] == "qsvt"
        rigorous = frame["bound_rigorous"].fillna(False).astype(bool)
        within_bound = pd.to_numeric(frame["bound_value"], errors="coerce") <= pd.to_numeric(
            frame["algorithm_error_budget"], errors="coerce"
        )
        frame["bound_scope"] = np.select(
            [is_mpf, is_qsvt],
            ["ideal-mpf", "legacy-qsvt-unscoped"],
            default="implemented-product-formula",
        )
        frame["bound_target_satisfied"] = rigorous & within_bound & ~is_qsvt
        frame["circuit_bound_scope"] = np.select(
            [is_mpf, is_qsvt],
            [
                "repeated-shared-ancilla-good-block",
                "implemented-qsvt-floating-phase-circuit",
            ],
            default="implemented-product-formula",
        )
        frame["circuit_bound_rigorous"] = rigorous & ~is_mpf & ~is_qsvt
        frame["circuit_target_satisfied"] = rigorous & within_bound & ~is_mpf & ~is_qsvt
        for column in _SCHEMA2_EXTENSION_COLUMNS - set(frame.columns):
            frame[column] = None
    legacy_qsvt_claim = (frame["method_family"] == "qsvt") & (
        frame["bound_scope"] == "implemented-algorithm"
    )
    frame.loc[legacy_qsvt_claim, "bound_scope"] = "legacy-qsvt-unscoped"
    frame.loc[legacy_qsvt_claim, "bound_rigorous"] = False
    frame.loc[legacy_qsvt_claim, "bound_target_satisfied"] = False
    frame.loc[
        frame["method_family"] == "qsvt",
        "circuit_bound_scope",
    ] = "implemented-qsvt-floating-phase-circuit"
    frame.loc[frame["method_family"] == "qsvt", "circuit_bound_rigorous"] = False
    frame.loc[frame["method_family"] == "qsvt", "circuit_target_satisfied"] = False
    return frame


def save_benchmark(
    frame: pd.DataFrame,
    config: BenchmarkConfig,
    *,
    output_root: str | Path = "benchmark_outputs",
) -> tuple[Path, Path, Path]:
    """Persist one in-memory run in a new collision-resistant directory."""
    validate_benchmark_frame(frame)
    config.validate()
    run_ids = frame["run_id"].dropna().astype(str).unique()
    generated = frame["generated_at_utc"].dropna().astype(str).unique()
    digests = frame["config_digest"].dropna().astype(str).unique()
    if len(run_ids) != 1 or len(generated) != 1 or len(digests) != 1:
        raise ValueError("save_benchmark requires exactly one generated run")
    timestamp = datetime.fromisoformat(generated[0]).astimezone(timezone.utc)
    directory_name = (
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{digests[0][:8]}_{run_ids[0].replace('-', '')[:8]}"
    )
    run_directory = Path(output_root).resolve() / directory_name
    run_directory.mkdir(parents=True, exist_ok=False)
    csv_path = run_directory / "benchmark.csv"
    metadata_path = run_directory / "metadata.json"
    frame.to_csv(csv_path, index=False, na_rep="")
    first = frame.iloc[0]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "data_file": csv_path.name,
        "row_count": len(frame),
        "status_counts": {
            str(key): int(value) for key, value in frame["status"].value_counts().items()
        },
        "columns": list(frame.columns),
        "run": {
            "run_id": run_ids[0],
            "generated_at_utc": generated[0],
            "config_digest": digests[0],
        },
        "configuration": config.as_dict(),
        "software": {
            "package_version": first["package_version"],
            "python_version": first["python_version"],
            "qiskit_version": first["qiskit_version"],
            "git_commit": first["git_commit"],
            "git_dirty": None if pd.isna(first["git_dirty"]) else bool(first["git_dirty"]),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_directory, csv_path, metadata_path
