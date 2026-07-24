"""Separate benchmark tables and plots for fourth-order error bounds."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .fourth_order_bounds import (
    FourthOrderBoundProblem,
    FourthOrderBoundResult,
    FourthOrderNormMethod,
    all_schubert_mendl_centers,
    build_fourth_order_bound_problem,
    childs_fourth_order_small_prefactor_bound,
    childs_general_commutator_bound,
    minimizing_schubert_mendl_center,
    schubert_mendl_small_prefactor_bound,
)
from .hamiltonians import PauliHamiltonian, heisenberg_chain, transverse_field_ising


FourthOrderComparisonSweep: TypeAlias = Literal["system-size", "target-error"]
FourthOrderComparisonMetric: TypeAlias = Literal[
    "one_step_coefficient",
    "required_segment_count",
    "ratio_to_reference",
]

FOURTH_ORDER_COMPARISON_SCHEMA_VERSION = "1.0"
FOURTH_ORDER_COMPARISON_COLUMNS = (
    "schema_version",
    "run_id",
    "generated_at_utc",
    "sweep",
    "decomposition_case",
    "hamiltonian_model",
    "hamiltonian_name",
    "system_qubits",
    "hamiltonian_decomposition_json",
    "term_ordering_json",
    "hamiltonian_group_count",
    "hamiltonian_group_sizes_json",
    "product_formula_order",
    "product_formula_coefficients_json",
    "ordered_exponentials_json",
    "exponential_count",
    "consecutive_exponentials_merged",
    "bound_family",
    "bound_variant",
    "specific_theorem_or_equation",
    "center_index_s",
    "centered_index_s",
    "is_centered_s",
    "is_minimizing_s",
    "norm_evaluation_method",
    "triangle_inequalities_json",
    "additional_relaxations_json",
    "commutator_contributions_json",
    "one_step_coefficient_c5",
    "time_step_power",
    "evolution_time",
    "segment_count",
    "accumulated_error_bound",
    "target_error",
    "required_segment_count",
    "reference_bound_family",
    "ratio_to_reference",
    "status",
    "diagnostic_message",
)

_OUTPUT_FORMATS = {"png", "pdf", "svg"}

_CURVE_STYLES: dict[str, dict[str, Any]] = {
    "Childs general proof relaxation": {
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "s",
    },
    "Childs Appendix M": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
    },
    "Schubert--Mendl centered": {
        "color": "#009E73",
        "linestyle": "-.",
        "marker": "^",
    },
    "Schubert--Mendl minimizing": {
        "color": "#CC79A7",
        "linestyle": ":",
        "marker": "D",
    },
}


@dataclass(frozen=True)
class FourthOrderComparisonConfig:
    """Configuration for the standalone bound-comparison sweeps."""

    system_qubit_values: tuple[int, ...] = (3, 4, 5, 6)
    target_error_values: tuple[float, ...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
    fixed_system_qubits_for_error_sweep: int = 4
    fixed_target_error_for_size_sweep: float = 1e-3
    evolution_time: float = 1.0
    segment_count: int = 1
    transverse_field_ising_parameters: dict[str, Any] | None = None
    heisenberg_parameters: dict[str, Any] | None = None
    norm_method: FourthOrderNormMethod = "pauli-l1"
    merge_adjacent: bool = True
    include_combined_base_diagnostic: bool = True
    output_directory: Path = Path("benchmark_outputs/fourth_order_bounds")
    output_formats: tuple[str, ...] = ("png", "pdf")

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_qubit_values", tuple(self.system_qubit_values))
        object.__setattr__(self, "target_error_values", tuple(self.target_error_values))
        object.__setattr__(self, "output_formats", tuple(self.output_formats))
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        object.__setattr__(
            self,
            "transverse_field_ising_parameters",
            dict(
                self.transverse_field_ising_parameters
                or {"coupling": 1.0, "field": 0.7, "periodic": False}
            ),
        )
        object.__setattr__(
            self,
            "heisenberg_parameters",
            dict(self.heisenberg_parameters or {"coupling": 1.0, "field_z": 0.3}),
        )

        if not self.system_qubit_values or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 3
            for size in self.system_qubit_values
        ):
            raise ValueError("system_qubit_values must contain integers at least three")
        if len(set(self.system_qubit_values)) != len(self.system_qubit_values):
            raise ValueError("system_qubit_values must not contain duplicates")
        if not self.target_error_values or any(
            not np.isfinite(error) or error <= 0 for error in self.target_error_values
        ):
            raise ValueError("target_error_values must be positive and finite")
        if len(set(self.target_error_values)) != len(self.target_error_values):
            raise ValueError("target_error_values must not contain duplicates")
        if (
            isinstance(self.fixed_system_qubits_for_error_sweep, bool)
            or not isinstance(self.fixed_system_qubits_for_error_sweep, int)
            or self.fixed_system_qubits_for_error_sweep < 3
        ):
            raise ValueError("fixed_system_qubits_for_error_sweep must be at least three")
        if (
            not np.isfinite(self.fixed_target_error_for_size_sweep)
            or self.fixed_target_error_for_size_sweep <= 0
        ):
            raise ValueError("fixed_target_error_for_size_sweep must be positive and finite")
        if not np.isfinite(self.evolution_time) or self.evolution_time <= 0:
            raise ValueError("evolution_time must be positive and finite")
        if (
            isinstance(self.segment_count, bool)
            or not isinstance(self.segment_count, int)
            or self.segment_count < 1
        ):
            raise ValueError("segment_count must be a positive integer")
        if self.norm_method not in ("pauli-l1", "spectral"):
            raise ValueError("norm_method must be 'pauli-l1' or 'spectral'")
        if not isinstance(self.merge_adjacent, bool):
            raise TypeError("merge_adjacent must be a boolean")
        if not isinstance(self.include_combined_base_diagnostic, bool):
            raise TypeError("include_combined_base_diagnostic must be a boolean")
        if not self.output_formats or set(self.output_formats) - _OUTPUT_FORMATS:
            raise ValueError("output_formats must contain only png, pdf, or svg")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""
        return {
            "system_qubit_values": list(self.system_qubit_values),
            "target_error_values": list(self.target_error_values),
            "fixed_system_qubits_for_error_sweep": (
                self.fixed_system_qubits_for_error_sweep
            ),
            "fixed_target_error_for_size_sweep": (
                self.fixed_target_error_for_size_sweep
            ),
            "evolution_time": self.evolution_time,
            "segment_count": self.segment_count,
            "transverse_field_ising_parameters": dict(
                self.transverse_field_ising_parameters or {}
            ),
            "heisenberg_parameters": dict(self.heisenberg_parameters or {}),
            "norm_method": self.norm_method,
            "merge_adjacent": self.merge_adjacent,
            "include_combined_base_diagnostic": self.include_combined_base_diagnostic,
            "output_directory": str(self.output_directory),
            "output_formats": list(self.output_formats),
        }


def load_fourth_order_comparison_config(
    path: str | Path,
) -> FourthOrderComparisonConfig:
    """Load a standalone comparison configuration from JSON."""
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("fourth-order comparison configuration must be a JSON object")
    allowed = set(FourthOrderComparisonConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown comparison configuration fields: {sorted(unknown)}")
    if "output_directory" in raw:
        output = Path(raw["output_directory"])
        if not output.is_absolute():
            output = config_path.parent / output
        raw["output_directory"] = output.resolve()
    return FourthOrderComparisonConfig(**raw)


def _model_cases(
    config: FourthOrderComparisonConfig,
    system_qubits: int,
) -> tuple[tuple[str, str, PauliHamiltonian], ...]:
    return (
        (
            "two-term",
            "transverse_field_ising",
            transverse_field_ising(
                system_qubits,
                **dict(config.transverse_field_ising_parameters or {}),
            ),
        ),
        (
            "multi-term",
            "heisenberg_chain",
            heisenberg_chain(
                system_qubits,
                **dict(config.heisenberg_parameters or {}),
            ),
        ),
    )


def _group_description(problem: FourthOrderBoundProblem) -> str:
    groups = []
    for index, group in enumerate(problem.groups):
        terms = [
            {
                "pauli": label,
                "coefficient_real": float(complex(coefficient).real),
                "coefficient_imag": float(complex(coefficient).imag),
            }
            for label, coefficient in group.to_list()
        ]
        groups.append({"hamiltonian_index": index + 1, "terms": terms})
    return json.dumps(groups, separators=(",", ":"), sort_keys=True)


def _factors_json(problem: FourthOrderBoundProblem) -> str:
    return json.dumps(
        [
            {"k": k, "hamiltonian_index": group + 1, "coefficient": coefficient}
            for k, (group, coefficient) in enumerate(
                problem.ordered_exponentials,
                start=1,
            )
        ],
        separators=(",", ":"),
    )


def _contributions_json(result: FourthOrderBoundResult) -> str:
    return json.dumps(
        [
            {
                "hamiltonian_indices": [
                    index + 1 for index in contribution.hamiltonian_indices
                ],
                "base_coefficients": [
                    [index + 1, coefficient]
                    for index, coefficient in contribution.base_coefficients
                ],
                "prefactor": contribution.prefactor,
                "commutator_norm": contribution.commutator_norm,
                "weighted_value": contribution.weighted_value,
                "source_term_count": contribution.source_term_count,
            }
            for contribution in result.contributions
        ],
        separators=(",", ":"),
    )


def _comparison_results(
    problem: FourthOrderBoundProblem,
    config: FourthOrderComparisonConfig,
) -> tuple[
    tuple[FourthOrderBoundResult, str, bool, bool], ...
]:
    general = childs_general_commutator_bound(problem, norm_method=config.norm_method)
    appendix = childs_fourth_order_small_prefactor_bound(
        problem,
        norm_method=config.norm_method,
    )
    centers = all_schubert_mendl_centers(
        problem,
        norm_method=config.norm_method,
        expand_base_triangle=True,
    )
    minimum = minimizing_schubert_mendl_center(centers)
    results: list[tuple[FourthOrderBoundResult, str, bool, bool]] = [
        (general, "general-proof-relaxation", False, False),
        (appendix, "appendix-m-expanded", False, False),
    ]
    results.extend(
        (
            result,
            "theorem-1-expanded-base",
            result.center_index == problem.centered_index,
            result is minimum,
        )
        for result in centers
    )
    if config.include_combined_base_diagnostic:
        combined = schubert_mendl_small_prefactor_bound(
            problem,
            center=problem.centered_index,
            norm_method=config.norm_method,
            expand_base_triangle=False,
        )
        results.append((combined, "theorem-1-combined-base", True, False))
    return tuple(results)


def _base_record(
    *,
    run_id: str,
    generated_at: str,
    sweep: FourthOrderComparisonSweep,
    decomposition_case: str,
    model_name: str,
    problem: FourthOrderBoundProblem,
    result: FourthOrderBoundResult,
    variant: str,
    is_centered: bool,
    is_minimizing: bool,
    config: FourthOrderComparisonConfig,
    target_error: float,
) -> dict[str, Any]:
    coefficient = result.one_step_coefficient
    if coefficient is None:
        accumulated = None
        required = None
    else:
        accumulated = result.accumulated_error_bound(
            config.evolution_time,
            config.segment_count,
        )
        required = result.required_segments(config.evolution_time, target_error)
    return {
        "schema_version": FOURTH_ORDER_COMPARISON_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "sweep": sweep,
        "decomposition_case": decomposition_case,
        "hamiltonian_model": model_name,
        "hamiltonian_name": problem.hamiltonian.name,
        "system_qubits": problem.hamiltonian.num_qubits,
        "hamiltonian_decomposition_json": _group_description(problem),
        "term_ordering_json": json.dumps(
            list(range(1, problem.group_count + 1)), separators=(",", ":")
        ),
        "hamiltonian_group_count": problem.group_count,
        "hamiltonian_group_sizes_json": json.dumps(
            problem.group_sizes, separators=(",", ":")
        ),
        "product_formula_order": problem.order,
        "product_formula_coefficients_json": json.dumps(
            {"z1": problem.z1, "z0": problem.z0}, separators=(",", ":")
        ),
        "ordered_exponentials_json": _factors_json(problem),
        "exponential_count": problem.exponential_count,
        "consecutive_exponentials_merged": problem.merged_consecutive,
        "bound_family": result.bound_family,
        "bound_variant": variant,
        "specific_theorem_or_equation": result.specific_result,
        "center_index_s": result.center_index,
        "centered_index_s": result.centered_index,
        "is_centered_s": is_centered,
        "is_minimizing_s": is_minimizing,
        "norm_evaluation_method": result.norm_method,
        "triangle_inequalities_json": json.dumps(result.triangle_inequalities),
        "additional_relaxations_json": json.dumps(result.additional_relaxations),
        "commutator_contributions_json": _contributions_json(result),
        "one_step_coefficient_c5": coefficient,
        "time_step_power": result.time_power,
        "evolution_time": config.evolution_time,
        "segment_count": config.segment_count,
        "accumulated_error_bound": accumulated,
        "target_error": target_error,
        "required_segment_count": required,
        "reference_bound_family": None,
        "ratio_to_reference": None,
        "status": result.status,
        "diagnostic_message": result.diagnostic_message,
    }


def _add_reference_ratios(records: list[dict[str, Any]]) -> None:
    by_case: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["decomposition_case"], record["system_qubits"])
        by_case.setdefault(key, []).append(record)
    for case_records in by_case.values():
        appendix = next(
            (
                record
                for record in case_records
                if record["bound_variant"] == "appendix-m-expanded"
                and record["status"] == "ok"
            ),
            None,
        )
        reference = appendix or next(
            record
            for record in case_records
            if record["bound_variant"] == "general-proof-relaxation"
        )
        reference_value = reference["one_step_coefficient_c5"]
        for record in case_records:
            record["reference_bound_family"] = reference["bound_family"]
            value = record["one_step_coefficient_c5"]
            if value is not None and reference_value is not None:
                if reference_value == 0:
                    record["ratio_to_reference"] = 1.0 if value == 0 else np.inf
                else:
                    record["ratio_to_reference"] = value / reference_value


def generate_fourth_order_comparison(
    config: FourthOrderComparisonConfig,
    sweep: FourthOrderComparisonSweep,
) -> pd.DataFrame:
    """Generate one standalone comparison sweep, retaining every valid center."""
    if sweep == "system-size":
        points = tuple(
            (size, config.fixed_target_error_for_size_sweep)
            for size in config.system_qubit_values
        )
    elif sweep == "target-error":
        points = tuple(
            (config.fixed_system_qubits_for_error_sweep, error)
            for error in config.target_error_values
        )
    else:
        raise ValueError("sweep must be 'system-size' or 'target-error'")

    run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    result_cache: dict[
        tuple[str, int],
        tuple[FourthOrderBoundProblem, tuple[tuple[FourthOrderBoundResult, str, bool, bool], ...]],
    ] = {}
    for system_qubits, target_error in points:
        for decomposition_case, model_name, hamiltonian in _model_cases(
            config,
            system_qubits,
        ):
            cache_key = (decomposition_case, system_qubits)
            if cache_key not in result_cache:
                problem = build_fourth_order_bound_problem(
                    hamiltonian,
                    partition="auto",
                    merge_adjacent=config.merge_adjacent,
                )
                if decomposition_case == "two-term" and problem.group_count != 2:
                    raise ValueError("the two-term benchmark did not resolve to two groups")
                if decomposition_case == "multi-term" and problem.group_count < 3:
                    raise ValueError("the multi-term benchmark did not resolve to at least three groups")
                result_cache[cache_key] = (
                    problem,
                    _comparison_results(problem, config),
                )
            problem, results = result_cache[cache_key]
            for result, variant, is_centered, is_minimizing in results:
                records.append(
                    _base_record(
                        run_id=run_id,
                        generated_at=generated_at,
                        sweep=sweep,
                        decomposition_case=decomposition_case,
                        model_name=model_name,
                        problem=problem,
                        result=result,
                        variant=variant,
                        is_centered=is_centered,
                        is_minimizing=is_minimizing,
                        config=config,
                        target_error=target_error,
                    )
                )
    _add_reference_ratios(records)
    frame = pd.DataFrame.from_records(records, columns=FOURTH_ORDER_COMPARISON_COLUMNS)
    validate_fourth_order_comparison_frame(frame, expected_sweep=sweep)
    return frame


def validate_fourth_order_comparison_frame(
    frame: pd.DataFrame,
    *,
    expected_sweep: FourthOrderComparisonSweep | None = None,
) -> None:
    """Validate the machine-readable comparison schema and invariants."""
    if tuple(frame.columns) != FOURTH_ORDER_COMPARISON_COLUMNS:
        raise ValueError("fourth-order comparison columns do not match the schema")
    if frame.empty:
        raise ValueError("fourth-order comparison data must not be empty")
    if set(frame["schema_version"].astype(str)) != {
        FOURTH_ORDER_COMPARISON_SCHEMA_VERSION
    }:
        raise ValueError("unsupported fourth-order comparison schema")
    if expected_sweep is not None and set(frame["sweep"]) != {expected_sweep}:
        raise ValueError(f"comparison data is not a {expected_sweep!r} sweep")
    if not set(frame["status"]).issubset({"ok", "unsupported"}):
        raise ValueError("comparison status must be 'ok' or 'unsupported'")
    successful = frame[frame["status"] == "ok"]
    required = [
        "one_step_coefficient_c5",
        "accumulated_error_bound",
        "required_segment_count",
        "ratio_to_reference",
    ]
    if successful[required].isna().any().any():
        raise ValueError("successful comparison rows must contain all bound values")
    if (successful["time_step_power"] != _TIME_POWER_FOR_SCHEMA).any():
        raise ValueError("all fourth-order rows must use time-step power five")


_TIME_POWER_FOR_SCHEMA = 5


def save_fourth_order_comparison(
    frame: pd.DataFrame,
    config: FourthOrderComparisonConfig,
) -> tuple[Path, Path]:
    """Persist a validated comparison table and resolved metadata."""
    validate_fourth_order_comparison_frame(frame)
    sweep = str(frame["sweep"].iloc[0])
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    stem = sweep.replace("-", "_")
    csv_path = output / f"{stem}_fourth_order_bounds.csv"
    metadata_path = output / f"{stem}_fourth_order_bounds.metadata.json"
    frame.to_csv(csv_path, index=False)
    metadata = {
        "schema_version": FOURTH_ORDER_COMPARISON_SCHEMA_VERSION,
        "sweep": sweep,
        "config": config.as_dict(),
        "row_count": len(frame),
        "status_counts": {
            str(key): int(value) for key, value in frame["status"].value_counts().items()
        },
        "ratio_denominator_policy": (
            "Childs Appendix M when supported; otherwise the explicit Childs general proof relaxation"
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, metadata_path


def _primary_curve_rows(
    frame: pd.DataFrame,
) -> tuple[tuple[str, pd.DataFrame], ...]:
    expanded = frame[frame["bound_variant"] == "theorem-1-expanded-base"]
    return (
        (
            "Childs general proof relaxation",
            frame[frame["bound_variant"] == "general-proof-relaxation"],
        ),
        (
            "Childs Appendix M",
            frame[frame["bound_variant"] == "appendix-m-expanded"],
        ),
        ("Schubert--Mendl centered", expanded[expanded["is_centered_s"]]),
        ("Schubert--Mendl minimizing", expanded[expanded["is_minimizing_s"]]),
    )


def create_fourth_order_comparison_figure(
    frame: pd.DataFrame,
    metric: FourthOrderComparisonMetric,
    decomposition_case: Literal["two-term", "multi-term"],
) -> Figure:
    """Create one bound-only figure from persisted comparison data."""
    validate_fourth_order_comparison_frame(frame)
    if metric not in (
        "one_step_coefficient",
        "required_segment_count",
        "ratio_to_reference",
    ):
        raise ValueError("unsupported fourth-order comparison metric")
    subset = frame[frame["decomposition_case"] == decomposition_case]
    if subset.empty:
        raise ValueError(f"no rows for decomposition case {decomposition_case!r}")
    sweep = str(subset["sweep"].iloc[0])
    if metric == "one_step_coefficient":
        y_column = "one_step_coefficient_c5"
        y_label = r"One-step coefficient $C_5$"
    elif metric == "required_segment_count":
        y_column = "required_segment_count"
        y_label = "Required fourth-order segments"
    else:
        y_column = "ratio_to_reference"
        y_label = "Ratio to Childs Appendix M"
    x_column = "system_qubits" if sweep == "system-size" else "target_error"
    x_label = "System qubits" if sweep == "system-size" else "Target error"

    figure = Figure(figsize=(7.2, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for label, rows in _primary_curve_rows(subset):
        rows = rows[rows["status"] == "ok"].sort_values(x_column)
        if rows.empty:
            continue
        axis.plot(
            rows[x_column],
            rows[y_column],
            label=label,
            linewidth=1.8,
            markersize=5,
            **_CURVE_STYLES[label],
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_yscale("log")
    if sweep == "target-error":
        axis.set_xscale("log")
        axis.invert_xaxis()
    model = str(subset["hamiltonian_model"].iloc[0])
    axis.set_title(f"Fourth-order bound comparison: {model} ({decomposition_case})")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    return figure


def plot_fourth_order_comparison(
    frame_or_path: pd.DataFrame | str | Path,
    *,
    output_directory: str | Path | None = None,
    output_formats: tuple[str, ...] = ("png", "pdf"),
) -> tuple[Path, ...]:
    """Write all requested standalone coefficient, segment, and ratio plots."""
    if isinstance(frame_or_path, pd.DataFrame):
        frame = frame_or_path
        source_path = None
    else:
        source_path = Path(frame_or_path)
        frame = pd.read_csv(source_path)
    validate_fourth_order_comparison_frame(frame)
    unknown = set(output_formats) - _OUTPUT_FORMATS
    if unknown:
        raise ValueError(f"unsupported plot formats: {sorted(unknown)}")
    if output_directory is None:
        output = source_path.parent if source_path is not None else Path.cwd()
    else:
        output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    sweep = str(frame["sweep"].iloc[0]).replace("-", "_")
    metrics: tuple[FourthOrderComparisonMetric, ...]
    if sweep == "system_size":
        metrics = (
            "one_step_coefficient",
            "required_segment_count",
            "ratio_to_reference",
        )
    else:
        metrics = ("required_segment_count", "ratio_to_reference")
    paths: list[Path] = []
    for decomposition_case in ("two-term", "multi-term"):
        for metric in metrics:
            figure = create_fourth_order_comparison_figure(
                frame,
                metric,
                decomposition_case,
            )
            stem = f"{sweep}_{decomposition_case.replace('-', '_')}_{metric}"
            for extension in output_formats:
                path = output / f"{stem}.{extension}"
                figure.savefig(path, dpi=180)
                paths.append(path)
    return tuple(paths)


def generate_and_save_fourth_order_comparison(
    config: FourthOrderComparisonConfig,
) -> tuple[Path, ...]:
    """Generate both standalone sweeps, tables, metadata, and plots."""
    outputs: list[Path] = []
    for sweep in ("system-size", "target-error"):
        frame = generate_fourth_order_comparison(config, sweep)
        csv_path, metadata_path = save_fourth_order_comparison(frame, config)
        outputs.extend((csv_path, metadata_path))
        outputs.extend(
            plot_fourth_order_comparison(
                frame,
                output_directory=config.output_directory,
                output_formats=config.output_formats,
            )
        )
    return tuple(outputs)
