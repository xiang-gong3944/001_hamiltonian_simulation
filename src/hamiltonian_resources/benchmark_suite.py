"""Configurable analytical resource-scaling benchmark data generation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np
import pandas as pd

from .benchmark import BenchmarkConfig, choose_parameters, estimate_resources_analytically
from .hamiltonians import PauliHamiltonian, heisenberg_chain, transverse_field_ising
from .multiproduct import multiproduct_coefficients, optimal_mpf_exponents
from .trotter import (
    _higher_order_commutator_work,
    estimate_suzuki_error,
    suzuki_commutator_bounds,
)


BenchmarkSweep: TypeAlias = Literal["system-size", "target-error"]
EvolutionTimeMode: TypeAlias = Literal["fixed", "system-size"]

SCHEMA_VERSION = "1.1"
_SUPPORTED_SCHEMA_VERSIONS = {"1.0", SCHEMA_VERSION}
_HIGHER_ORDER_COMMUTATOR_WORK_LIMIT = 32_768
TROTTER_ORDERS = (1, 2, 4, 6)
MPF_TERM_COUNTS = (3, 5, 7)
METHOD_LABELS = (
    "Trotter p=1",
    "Trotter p=2",
    "Trotter p=4",
    "Trotter p=6",
    "MPF m=3",
    "MPF m=5",
    "MPF m=7",
    "QSVT",
)

SWEEP_FILENAMES: dict[BenchmarkSweep, str] = {
    "system-size": "system_size_scaling.csv",
    "target-error": "target_error_scaling.csv",
}

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
    "target_error",
    "hamiltonian_alpha",
    "hamiltonian_term_count",
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
    "bound_rigorous",
    "algorithm_error_budget",
    "mpf_schedule",
    "mpf_exponents_json",
    "mpf_coefficients_json",
    "mpf_coefficient_l1_norm",
    "mpf_padding_weight",
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

_MODEL_PARAMETERS = {
    "transverse_field_ising": {"coupling", "field", "periodic"},
    "heisenberg_chain": {"coupling", "field_z"},
}
_OUTPUT_FORMATS = {"png", "pdf", "svg"}


@dataclass(frozen=True)
class ScalingBenchmarkConfig:
    """Resolved inputs for both analytical scaling sweeps."""

    hamiltonian_model: str = "transverse_field_ising"
    model_parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "coupling": 1.0,
            "field": 3.0,
            "periodic": False,
        }
    )
    system_qubit_values: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 500)
    target_error_values: tuple[float, ...] = (
        0.1,
        0.03,
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0001,
    )
    evolution_time: float = 1.0
    evolution_time_mode: str = "fixed"
    fixed_system_qubits_for_error_sweep: int = 8
    fixed_target_error_for_size_sweep: float = 1e-3
    synthesis_error_fraction: float = 0.1
    trotter_partition: str = "auto"
    mpf_schedule: str = "new"
    output_directory: Path = Path("benchmark_outputs")
    output_formats: tuple[str, ...] = ("png", "pdf")
    generate_summary_plots: bool = False
    skip_expensive_higher_order_bounds: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_parameters", dict(self.model_parameters))
        object.__setattr__(self, "system_qubit_values", tuple(self.system_qubit_values))
        object.__setattr__(self, "target_error_values", tuple(self.target_error_values))
        object.__setattr__(self, "output_formats", tuple(self.output_formats))
        object.__setattr__(self, "output_directory", Path(self.output_directory))

        if self.hamiltonian_model not in _MODEL_PARAMETERS:
            supported = ", ".join(sorted(_MODEL_PARAMETERS))
            raise ValueError(f"hamiltonian_model must be one of: {supported}")
        unknown_parameters = set(self.model_parameters) - _MODEL_PARAMETERS[
            self.hamiltonian_model
        ]
        if unknown_parameters:
            names = ", ".join(sorted(unknown_parameters))
            raise ValueError(
                f"unsupported parameters for {self.hamiltonian_model}: {names}"
            )
        _validate_sizes("system_qubit_values", self.system_qubit_values)
        _validate_errors("target_error_values", self.target_error_values)
        if (
            isinstance(self.fixed_system_qubits_for_error_sweep, bool)
            or not isinstance(self.fixed_system_qubits_for_error_sweep, int)
            or self.fixed_system_qubits_for_error_sweep < 1
        ):
            raise ValueError("fixed_system_qubits_for_error_sweep must be a positive integer")
        if not _is_probability(self.fixed_target_error_for_size_sweep):
            raise ValueError("fixed_target_error_for_size_sweep must lie in (0, 1)")
        if not np.isfinite(self.evolution_time) or self.evolution_time <= 0:
            raise ValueError("evolution_time must be positive and finite")
        if self.evolution_time_mode not in {"fixed", "system-size"}:
            raise ValueError(
                "evolution_time_mode must be 'fixed' or 'system-size'"
            )
        if not _is_probability(self.synthesis_error_fraction):
            raise ValueError("synthesis_error_fraction must lie in (0, 1)")
        if self.trotter_partition not in {"auto", "individual", "commuting"}:
            raise ValueError(
                "trotter_partition must be 'auto', 'individual', or 'commuting'"
            )
        if self.mpf_schedule not in {"new", "legacy"}:
            raise ValueError("mpf_schedule must be 'new' or 'legacy'")
        if not self.output_formats:
            raise ValueError("output_formats must not be empty")
        unknown_formats = set(self.output_formats) - _OUTPUT_FORMATS
        if unknown_formats:
            names = ", ".join(sorted(unknown_formats))
            raise ValueError(f"unsupported output formats: {names}")
        if len(set(self.output_formats)) != len(self.output_formats):
            raise ValueError("output_formats must not contain duplicates")
        if not isinstance(self.generate_summary_plots, bool):
            raise TypeError("generate_summary_plots must be a boolean")
        if not isinstance(self.skip_expensive_higher_order_bounds, bool):
            raise TypeError("skip_expensive_higher_order_bounds must be a boolean")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""
        return {
            "hamiltonian_model": self.hamiltonian_model,
            "model_parameters": dict(self.model_parameters),
            "system_qubit_values": list(self.system_qubit_values),
            "target_error_values": list(self.target_error_values),
            "evolution_time": self.evolution_time,
            "evolution_time_mode": self.evolution_time_mode,
            "fixed_system_qubits_for_error_sweep": (
                self.fixed_system_qubits_for_error_sweep
            ),
            "fixed_target_error_for_size_sweep": (
                self.fixed_target_error_for_size_sweep
            ),
            "synthesis_error_fraction": self.synthesis_error_fraction,
            "trotter_partition": self.trotter_partition,
            "mpf_schedule": self.mpf_schedule,
            "output_directory": str(self.output_directory),
            "output_formats": list(self.output_formats),
            "generate_summary_plots": self.generate_summary_plots,
            "skip_expensive_higher_order_bounds": (
                self.skip_expensive_higher_order_bounds
            ),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _MethodCase:
    family: str
    label: str
    trotter_order: int | None = None
    mpf_term_count: int | None = None


_METHOD_CASES = tuple(
    _MethodCase("trotter", f"Trotter p={order}", trotter_order=order)
    for order in TROTTER_ORDERS
) + tuple(
    _MethodCase("multiproduct", f"MPF m={term_count}", mpf_term_count=term_count)
    for term_count in MPF_TERM_COUNTS
) + (_MethodCase("qsvt", "QSVT"),)


def _is_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and 0 < float(value) < 1
    )


def _validate_sizes(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _validate_errors(name: str, values: tuple[float, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if not all(_is_probability(value) for value in values):
        raise ValueError(f"{name} must contain values in (0, 1)")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def load_benchmark_config(path: str | Path) -> ScalingBenchmarkConfig:
    """Load and validate a JSON configuration.

    A relative output directory is resolved relative to the configuration file.
    """
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("benchmark configuration must be a JSON object")

    allowed = {
        field_name
        for field_name in ScalingBenchmarkConfig.__dataclass_fields__
    }
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown benchmark configuration fields: {names}")
    if "output_directory" in raw:
        output_directory = Path(raw["output_directory"])
        if not output_directory.is_absolute():
            output_directory = config_path.parent / output_directory
        raw["output_directory"] = output_directory.resolve()
    return ScalingBenchmarkConfig(**raw)


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


def _build_hamiltonian(config: ScalingBenchmarkConfig, system_qubits: int) -> PauliHamiltonian:
    parameters = dict(config.model_parameters)
    if config.hamiltonian_model == "transverse_field_ising":
        return transverse_field_ising(system_qubits, **parameters)
    if config.hamiltonian_model == "heisenberg_chain":
        return heisenberg_chain(system_qubits, **parameters)
    raise AssertionError(f"unreachable model: {config.hamiltonian_model}")


def _empty_record() -> dict[str, Any]:
    return {column: None for column in BENCHMARK_COLUMNS}


def _base_record(
    config: ScalingBenchmarkConfig,
    sweep: BenchmarkSweep,
    run_metadata: Mapping[str, Any],
    system_qubits: int,
    evolution_time: float,
    target_error: float,
    method: _MethodCase,
) -> dict[str, Any]:
    record = _empty_record()
    record.update(
        schema_version=SCHEMA_VERSION,
        run_id=run_metadata["run_id"],
        generated_at_utc=run_metadata["generated_at_utc"],
        config_digest=config.digest,
        sweep=sweep,
        hamiltonian_model=config.hamiltonian_model,
        model_parameters_json=json.dumps(
            dict(config.model_parameters), sort_keys=True, separators=(",", ":")
        ),
        system_qubits=system_qubits,
        evolution_time=evolution_time,
        target_error=target_error,
        method_family=method.family,
        method_label=method.label,
        trotter_order=method.trotter_order,
        mpf_term_count=method.mpf_term_count,
        mpf_formal_order=(
            2 * method.mpf_term_count if method.mpf_term_count is not None else None
        ),
        algorithm_error_budget=target_error * (1 - config.synthesis_error_fraction),
        **run_metadata["software"],
    )
    return record


def _evaluation_config(
    config: ScalingBenchmarkConfig,
    evolution_time: float,
    target_error: float,
    method: _MethodCase,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        time=evolution_time,
        target_error=target_error,
        synthesis_error_fraction=config.synthesis_error_fraction,
        trotter_order=method.trotter_order or 2,
        trotter_partition=config.trotter_partition,  # type: ignore[arg-type]
        mpf_m=method.mpf_term_count or MPF_TERM_COUNTS[0],
        mpf_schedule=config.mpf_schedule,  # type: ignore[arg-type]
    )


def _method_metadata(
    hamiltonian: PauliHamiltonian,
    config: ScalingBenchmarkConfig,
    evaluation: BenchmarkConfig,
    method: _MethodCase,
    parameters: Mapping[str, int],
) -> dict[str, Any]:
    if method.family == "trotter":
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
            "lcu_normalization": 1.0,
            "amplitude_amplification": "none",
            "amplitude_amplification_rounds": 0,
            "good_subspace": "system register",
            "nominal_success_probability": 1.0,
        }
    if method.family == "multiproduct":
        assert method.mpf_term_count is not None
        segments = parameters["mpf_segments"]
        exponents = optimal_mpf_exponents(
            method.mpf_term_count, schedule=evaluation.mpf_schedule
        )
        coefficients = multiproduct_coefficients(
            method.mpf_term_count, schedule=evaluation.mpf_schedule
        )
        coefficient_norm = float(np.sum(np.abs(coefficients)))
        _, w2 = suzuki_commutator_bounds(hamiltonian)
        alpha_effective = min(hamiltonian.alpha, w2 ** (1 / 3))
        formal_order = 2 * method.mpf_term_count
        proxy = (
            (alpha_effective * evaluation.time) ** (formal_order + 1)
            / segments**formal_order
        )
        return {
            "segment_count": segments,
            "query_count": 3 * segments * sum(exponents),
            "bound_value": proxy,
            "bound_prefactor": alpha_effective ** (formal_order + 1),
            "bound_method": "commutator-calibrated-mpf-proxy",
            "bound_rigorous": False,
            "mpf_schedule": evaluation.mpf_schedule,
            "mpf_exponents_json": json.dumps(exponents, separators=(",", ":")),
            "mpf_coefficients_json": json.dumps(
                coefficients.tolist(), separators=(",", ":")
            ),
            "mpf_coefficient_l1_norm": coefficient_norm,
            "mpf_padding_weight": 2.0 - coefficient_norm,
            "lcu_normalization": 2.0,
            "amplitude_amplification": "one robust OAA round per segment",
            "amplitude_amplification_rounds": segments,
            "good_subspace": "branch register all-zero",
            "nominal_success_probability": 1.0,
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
        "lcu_normalization": 2.0,
        "amplitude_amplification": "one robust OAA round",
        "amplitude_amplification_rounds": 1,
        "good_subspace": "component, quadrature, and index registers all-zero",
        "nominal_success_probability": 1.0,
    }


def _evaluate_method(
    hamiltonian: PauliHamiltonian,
    config: ScalingBenchmarkConfig,
    evolution_time: float,
    target_error: float,
    method: _MethodCase,
) -> dict[str, Any]:
    evaluation = _evaluation_config(config, evolution_time, target_error, method)
    parameters = choose_parameters(hamiltonian, evaluation)
    resource = estimate_resources_analytically(
        hamiltonian, evaluation, method.family
    )
    result = _method_metadata(hamiltonian, config, evaluation, method, parameters)
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


def _expensive_higher_order_skip(
    hamiltonian: PauliHamiltonian,
    config: ScalingBenchmarkConfig,
    method: _MethodCase,
) -> dict[str, Any] | None:
    """Return row metadata when a rigorous higher-order bound should be skipped."""
    if (
        not config.skip_expensive_higher_order_bounds
        or method.family != "trotter"
        or method.trotter_order not in (4, 6)
    ):
        return None
    estimate = _higher_order_commutator_work(
        hamiltonian,
        method.trotter_order,
        config.trotter_partition,  # type: ignore[arg-type]
    )
    if estimate is None:
        return None
    work, specification = estimate
    if work <= _HIGHER_ORDER_COMMUTATOR_WORK_LIMIT:
        return None
    return {
        "trotter_partition": specification.partition,
        "trotter_group_count": len(specification.groups),
        "status": "skipped",
        "error_type": "HigherOrderBoundWorkLimit",
        "error_message": (
            f"skipped rigorous order-{method.trotter_order} commutator bound: "
            f"estimated work {work} exceeds limit "
            f"{_HIGHER_ORDER_COMMUTATOR_WORK_LIMIT}"
        ),
    }


def _resolved_evolution_time(
    config: ScalingBenchmarkConfig, system_qubits: int
) -> float:
    if config.evolution_time_mode == "system-size":
        return float(system_qubits)
    return float(config.evolution_time)


def _sweep_points(
    config: ScalingBenchmarkConfig, sweep: BenchmarkSweep
) -> tuple[tuple[int, float, float], ...]:
    if sweep == "system-size":
        return tuple(
            (
                size,
                _resolved_evolution_time(config, size),
                config.fixed_target_error_for_size_sweep,
            )
            for size in config.system_qubit_values
        )
    if sweep == "target-error":
        return tuple(
            (
                config.fixed_system_qubits_for_error_sweep,
                _resolved_evolution_time(
                    config, config.fixed_system_qubits_for_error_sweep
                ),
                error,
            )
            for error in config.target_error_values
        )
    raise ValueError("sweep must be 'system-size' or 'target-error'")


def generate_benchmark_sweep(
    config: ScalingBenchmarkConfig,
    sweep: BenchmarkSweep,
) -> pd.DataFrame:
    """Evaluate all eight fixed method configurations for one sweep.

    Each failed or intentionally skipped method remains present while other
    evaluations continue. The function never builds exact dense matrices or
    concrete circuits.
    """
    points = _sweep_points(config, sweep)
    run_metadata = {
        "run_id": str(uuid.uuid4()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": _software_metadata(),
    }
    records: list[dict[str, Any]] = []
    for system_qubits, evolution_time, target_error in points:
        try:
            hamiltonian = _build_hamiltonian(config, system_qubits)
            hamiltonian_error: Exception | None = None
        except Exception as exc:  # configuration-specific failures become rows
            hamiltonian = None
            hamiltonian_error = exc
        for method in _METHOD_CASES:
            record = _base_record(
                config,
                sweep,
                run_metadata,
                system_qubits,
                evolution_time,
                target_error,
                method,
            )
            try:
                if hamiltonian_error is not None:
                    raise hamiltonian_error
                assert hamiltonian is not None
                record.update(
                    hamiltonian_name=hamiltonian.name,
                    hamiltonian_alpha=hamiltonian.alpha,
                    hamiltonian_term_count=hamiltonian.term_count,
                )
                skip = _expensive_higher_order_skip(hamiltonian, config, method)
                if skip is None:
                    record.update(
                        _evaluate_method(
                            hamiltonian,
                            config,
                            evolution_time,
                            target_error,
                            method,
                        )
                    )
                else:
                    record.update(skip)
            except Exception as exc:  # every requested method retains a row
                record.update(
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc) or repr(exc),
                )
            records.append(record)
    frame = pd.DataFrame.from_records(records, columns=BENCHMARK_COLUMNS)
    validate_benchmark_frame(frame, expected_sweep=sweep)
    return frame


def validate_benchmark_frame(
    frame: pd.DataFrame,
    *,
    expected_sweep: BenchmarkSweep | None = None,
) -> None:
    """Validate the stable schema and required success/failure fields."""
    if tuple(frame.columns) != BENCHMARK_COLUMNS:
        raise ValueError("benchmark data columns do not match the supported schema")
    if frame.empty:
        raise ValueError("benchmark data must not be empty")
    versions = set(frame["schema_version"].astype(str))
    if len(versions) != 1 or not versions.issubset(_SUPPORTED_SCHEMA_VERSIONS):
        supported = ", ".join(sorted(_SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(f"unsupported benchmark schema; expected one of: {supported}")
    if expected_sweep is not None and set(frame["sweep"]) != {expected_sweep}:
        raise ValueError(f"benchmark data is not a {expected_sweep!r} sweep")
    schema_version = next(iter(versions))
    allowed_statuses = {"ok", "error"}
    if schema_version == SCHEMA_VERSION:
        allowed_statuses.add("skipped")
    if not set(frame["status"]).issubset(allowed_statuses):
        choices = ", ".join(sorted(allowed_statuses))
        raise ValueError(f"benchmark status must be one of: {choices}")
    successful = frame[frame["status"] == "ok"]
    if successful[["t_count", "cnot_count"]].isna().any().any():
        raise ValueError("successful benchmark rows must contain T and CNOT counts")
    if (successful[["t_count", "cnot_count"]] < 0).any().any():
        raise ValueError("resource counts must be nonnegative")
    diagnostic = frame[frame["status"].isin({"error", "skipped"})]
    if (
        diagnostic[["error_type", "error_message"]].isna().any().any()
        or (diagnostic[["error_type", "error_message"]] == "").any().any()
    ):
        raise ValueError(
            "failed or skipped benchmark rows must contain diagnostic details"
        )
    skipped = frame[frame["status"] == "skipped"]
    if skipped[["t_count", "cnot_count"]].notna().any().any():
        raise ValueError("skipped benchmark rows must not contain resource counts")


def load_benchmark_data(path: str | Path) -> pd.DataFrame:
    """Load and validate one persisted benchmark CSV."""
    frame = pd.read_csv(path, keep_default_na=True)
    validate_benchmark_frame(frame)
    return frame


def save_benchmark_data(
    frame: pd.DataFrame,
    config: ScalingBenchmarkConfig,
    *,
    path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Persist a benchmark CSV and a same-stem reproducibility sidecar."""
    validate_benchmark_frame(frame)
    sweeps = set(frame["sweep"])
    if len(sweeps) != 1:
        raise ValueError("save one sweep per benchmark CSV")
    sweep = next(iter(sweeps))
    if sweep not in SWEEP_FILENAMES:
        raise ValueError(f"unknown sweep in benchmark data: {sweep}")

    csv_path = Path(path) if path is not None else config.output_directory / SWEEP_FILENAMES[sweep]
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, na_rep="")
    metadata_path = csv_path.with_suffix(".metadata.json")
    first = frame.iloc[0]
    metadata = {
        "schema_version": str(first["schema_version"]),
        "data_file": csv_path.name,
        "row_count": len(frame),
        "status_counts": {
            str(key): int(value)
            for key, value in frame["status"].value_counts().items()
        },
        "columns": list(BENCHMARK_COLUMNS),
        "run": {
            "run_id": first["run_id"],
            "generated_at_utc": first["generated_at_utc"],
            "config_digest": first["config_digest"],
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
    return csv_path, metadata_path


def generate_and_save_benchmark(
    config: ScalingBenchmarkConfig,
    sweep: BenchmarkSweep,
) -> tuple[pd.DataFrame, Path, Path]:
    """Generate one sweep and persist its CSV and metadata sidecar."""
    frame = generate_benchmark_sweep(config, sweep)
    csv_path, metadata_path = save_benchmark_data(frame, config)
    return frame, csv_path, metadata_path
