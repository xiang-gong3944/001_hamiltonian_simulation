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

from .benchmark import (
    _EvaluationConfig,
    choose_parameters,
    estimate_resources_analytically,
)
from .hamiltonians import PauliHamiltonian, heisenberg_chain, transverse_field_ising
from .multiproduct import (
    MPFErrorMethod,
    estimate_mpf_error,
    mpf_lcu_structure,
    multiproduct_coefficients,
    optimal_mpf_exponents,
)
from .trotter import estimate_suzuki_error


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
    factory: Callable[..., PauliHamiltonian] | None = field(
        default=None, repr=False, compare=False
    )

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


@dataclass(frozen=True)
class TrotterMethod:
    order: int

    @property
    def family(self) -> str:
        return "trotter"

    @property
    def method_id(self) -> str:
        return f"trotter-p{self.order}"

    @property
    def label(self) -> str:
        return f"Trotter p={self.order}"

    def validate(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, Integral):
            raise ValueError("Trotter order must be 1 or a positive even integer")
        if self.order != 1 and (self.order < 2 or self.order % 2):
            raise ValueError("Trotter order must be 1 or a positive even integer")

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "order": int(self.order)}


@dataclass(frozen=True)
class MultiproductMethod:
    term_count: int
    schedule: Literal["new", "legacy"] = "new"
    error_method: MPFErrorMethod = "low2019-l1-ideal-rigorous"

    @property
    def family(self) -> str:
        return "multiproduct"

    @property
    def method_id(self) -> str:
        suffix = "" if self.schedule == "new" else f"-{self.schedule}"
        if self.error_method not in (
            "low2019-l1-ideal-rigorous",
            "low-rigorous",
        ):
            suffix += f"-{self.error_method}"
        return f"mpf-m{self.term_count}{suffix}"

    @property
    def label(self) -> str:
        suffix = "" if self.schedule == "new" else f" ({self.schedule})"
        if self.error_method == "legacy-w2-proxy":
            suffix += " [legacy W2 heuristic]"
        elif self.error_method == "mizuta2026-commutator-ideal-rigorous":
            suffix += " [Mizuta 2026 commutator]"
        return f"MPF m={self.term_count}{suffix}"

    def validate(self) -> None:
        if isinstance(self.term_count, bool) or not isinstance(self.term_count, Integral):
            raise ValueError("MPF term count must be an integer")
        optimal_mpf_exponents(int(self.term_count), schedule=self.schedule)
        if self.error_method not in (
            "low2019-l1-ideal-rigorous",
            "mizuta2026-commutator-ideal-rigorous",
            "low-rigorous",
            "legacy-w2-proxy",
        ):
            raise ValueError(
                "MPF error method must be 'low2019-l1-ideal-rigorous' "
                "(historical alias 'low-rigorous'), "
                "'mizuta2026-commutator-ideal-rigorous', or 'legacy-w2-proxy'"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "term_count": int(self.term_count),
            "schedule": self.schedule,
            "error_method": self.error_method,
        }


@dataclass(frozen=True)
class QSVTMethod:
    @property
    def family(self) -> str:
        return "qsvt"

    @property
    def method_id(self) -> str:
        return "qsvt"

    @property
    def label(self) -> str:
        return "QSVT"

    def validate(self) -> None:
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family}


MethodSpec: TypeAlias = TrotterMethod | MultiproductMethod | QSVTMethod


def default_methods() -> list[MethodSpec]:
    return [
        *(TrotterMethod(order) for order in (1, 2, 4, 6)),
        *(MultiproductMethod(term_count) for term_count in (3, 5, 7)),
        QSVTMethod(),
    ]


@dataclass
class BenchmarkConfig:
    """Mutable, notebook-friendly configuration for analytical sweeps."""

    hamiltonian: HamiltonianSpec = field(
        default_factory=lambda: HamiltonianSpec(
            parameters={"coupling": 1.0, "field": 3.0, "periodic": False}
        )
    )
    system_sizes: Sequence[int] = field(default_factory=lambda: [2, 4, 6, 8, 10, 12])
    target_errors: Sequence[float] = field(
        default_factory=lambda: [0.1, 0.03, 0.01, 0.003, 0.001]
    )
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
            raise ValueError(
                "trotter_partition must be 'auto', 'individual', or 'commuting'"
            )
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
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
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


def _evaluation_config(
    config: BenchmarkConfig,
    evolution_time: float,
    target_error: float,
    method: MethodSpec,
) -> _EvaluationConfig:
    return _EvaluationConfig(
        time=evolution_time,
        target_error=target_error,
        synthesis_error_fraction=config.synthesis_error_fraction,
        trotter_order=method.order if isinstance(method, TrotterMethod) else 2,
        trotter_partition=config.trotter_partition,
        mpf_m=method.term_count if isinstance(method, MultiproductMethod) else 3,
        mpf_schedule=method.schedule if isinstance(method, MultiproductMethod) else "new",
        mpf_error_method=(
            method.error_method
            if isinstance(method, MultiproductMethod)
            else "low2019-l1-ideal-rigorous"
        ),
    )


def _method_metadata(
    hamiltonian: PauliHamiltonian,
    evaluation: _EvaluationConfig,
    method: MethodSpec,
    parameters: Mapping[str, int],
) -> dict[str, Any]:
    if isinstance(method, TrotterMethod):
        reps = parameters["trotter_reps"]
        error = estimate_suzuki_error(
            hamiltonian,
            evaluation.time,
            reps,
            evaluation.trotter_order,
            partition=evaluation.trotter_partition,
        )
        return {
            "segment_count": reps,
            "trotter_partition": error.partition,
            "trotter_group_count": error.group_count,
            "bound_value": error.error,
            "bound_prefactor": error.prefactor,
            "bound_method": error.method,
            "bound_rigorous": error.rigorous,
            "bound_scope": "implemented-product-formula",
            "bound_target_satisfied": error.rigorous
            and error.error
            <= evaluation.target_error * (1 - evaluation.synthesis_error_fraction),
            "circuit_bound_scope": "implemented-product-formula",
            "circuit_bound_rigorous": error.rigorous,
            "circuit_target_satisfied": error.rigorous
            and error.error
            <= evaluation.target_error * (1 - evaluation.synthesis_error_fraction),
            "lcu_normalization": 1.0,
            "amplitude_amplification": "none",
            "amplitude_amplification_rounds": 0,
            "good_subspace": "system register",
            "nominal_success_probability": 1.0,
        }
    if isinstance(method, MultiproductMethod):
        segments = parameters["mpf_segments"]
        exponents = optimal_mpf_exponents(method.term_count, schedule=method.schedule)
        coefficients = multiproduct_coefficients(method.term_count, schedule=method.schedule)
        structure = mpf_lcu_structure(method.term_count, schedule=method.schedule)
        coefficient_norm = float(np.sum(np.abs(coefficients)))
        error = estimate_mpf_error(
            hamiltonian,
            evaluation.time,
            segments,
            method.term_count,
            schedule=method.schedule,
            method=method.error_method,
            target_error=(
                evaluation.target_error * (1 - evaluation.synthesis_error_fraction)
                if method.error_method
                == "mizuta2026-commutator-ideal-rigorous"
                else None
            ),
        )
        algorithm_budget = evaluation.target_error * (
            1 - evaluation.synthesis_error_fraction
        )
        return {
            "segment_count": segments,
            "query_count": 3 * segments * sum(exponents),
            "bound_value": error.error,
            "bound_prefactor": error.prefactor,
            "bound_method": error.method,
            "bound_reference": error.reference,
            "bound_theorem_or_equations": error.theorem_or_equations,
            "bound_components_json": json.dumps(
                dict(error.bound_components), sort_keys=True, separators=(",", ":")
            ),
            "bound_rigorous": error.rigorous,
            "bound_scope": error.scope,
            "bound_target_satisfied": error.rigorous
            and error.error <= algorithm_budget,
            "circuit_bound_scope": error.circuit_scope,
            "circuit_bound_rigorous": error.circuit_rigorous,
            "circuit_target_satisfied": False,
            "hamiltonian_decomposition": error.hamiltonian_decomposition,
            "bound_assumptions_json": json.dumps(
                error.assumptions, separators=(",", ":")
            ),
            "bound_fallback_reason": error.fallback_reason,
            "max_nested_commutator_order": error.max_nested_commutator_order,
            "max_exact_nested_commutator_order": (
                error.max_exact_nested_commutator_order
            ),
            "locality_compatible": error.locality_compatible,
            "commutator_cap_fallback": error.fallback_reason is not None,
            "commutator_bounds_json": json.dumps(
                dict(error.commutator_bounds),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "mpf_schedule": method.schedule,
            "mpf_exponents_json": json.dumps(exponents, separators=(",", ":")),
            "mpf_coefficients_json": json.dumps(
                coefficients.tolist(), separators=(",", ":")
            ),
            "mpf_coefficient_l1_norm": coefficient_norm,
            "mpf_padding_weight": 2.0 - coefficient_norm,
            "mpf_physical_branch_count": structure.physical_branch_count,
            "mpf_negative_coefficient_count": structure.negative_coefficient_count,
            "mpf_padding_branch_count": structure.padding_branch_count,
            "mpf_sign_branch_count": structure.sign_branch_count,
            "mpf_active_branch_count": structure.active_branch_count,
            "mpf_unused_branch_state_count": structure.unused_branch_state_count,
            "mpf_prepare_calls_per_segment": 6,
            "mpf_select_calls_per_segment": 3,
            "mpf_good_reflections_per_segment": 2,
            "mpf_base_lcu_uses_per_segment": 3,
            "lcu_normalization": 2.0,
            "amplitude_amplification": "one robust OAA round per segment",
            "amplitude_amplification_rounds": segments,
            "good_subspace": "branch register all-zero",
            "nominal_success_probability": None,
        }
    degree = parameters["qsvt_degree"]
    queries = 0 if degree == 0 else 3 * ((degree - 1) + degree)
    return {
        "query_count": queries,
        "qsvt_degree": degree,
        "bound_value": evaluation.target_error
        * (1 - evaluation.synthesis_error_fraction),
        "bound_method": "jacobi-anger-truncation",
        "bound_rigorous": True,
        "bound_scope": "implemented-algorithm",
        "bound_target_satisfied": True,
        "circuit_bound_scope": "implemented-algorithm",
        "circuit_bound_rigorous": True,
        "circuit_target_satisfied": True,
        "lcu_normalization": 2.0,
        "amplitude_amplification": "one robust OAA round",
        "amplitude_amplification_rounds": 1,
        "good_subspace": "component, quadrature, and index registers all-zero",
        "nominal_success_probability": 1.0,
    }


def _evaluate_method(
    hamiltonian: PauliHamiltonian,
    config: BenchmarkConfig,
    evolution_time: float,
    target_error: float,
    method: MethodSpec,
) -> dict[str, Any]:
    evaluation = _evaluation_config(config, evolution_time, target_error, method)
    parameters = choose_parameters(hamiltonian, evaluation, method.family)
    resource = estimate_resources_analytically(hamiltonian, evaluation, method.family)
    result = _method_metadata(hamiltonian, evaluation, method, parameters)
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
        mpf_term_count=(
            method.term_count if isinstance(method, MultiproductMethod) else None
        ),
        mpf_formal_order=(
            2 * method.term_count if isinstance(method, MultiproductMethod) else None
        ),
        algorithm_error_budget=target_error * (1 - config.synthesis_error_fraction),
        **run_metadata["software"],
    )
    return record


def run_benchmark(
    config: BenchmarkConfig,
    sweeps: Sequence[BenchmarkSweep] | BenchmarkSweep = (
        "system-size",
        "target-error",
    ),
    progress: ProgressCallback | None = None,
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
                record.update(
                    _evaluate_method(
                        hamiltonian,
                        config,
                        evolution_time,
                        target_error,
                        method,
                    )
                )
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
        rigorous = frame["bound_rigorous"].fillna(False).astype(bool)
        within_bound = (
            pd.to_numeric(frame["bound_value"], errors="coerce")
            <= pd.to_numeric(frame["algorithm_error_budget"], errors="coerce")
        )
        frame["bound_scope"] = np.where(
            is_mpf, "ideal-mpf", "implemented-algorithm"
        )
        frame["bound_target_satisfied"] = rigorous & within_bound
        frame["circuit_bound_scope"] = np.where(
            is_mpf, "amplified-shared-ancilla", "implemented-algorithm"
        )
        frame["circuit_bound_rigorous"] = rigorous & ~is_mpf
        frame["circuit_target_satisfied"] = rigorous & within_bound & ~is_mpf
        for column in _SCHEMA2_EXTENSION_COLUMNS - set(frame.columns):
            frame[column] = None
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
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{digests[0][:8]}_"
        f"{run_ids[0].replace('-', '')[:8]}"
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
