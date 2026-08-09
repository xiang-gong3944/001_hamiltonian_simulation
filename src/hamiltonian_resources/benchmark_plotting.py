"""Configurable plots for schema-2 benchmark data."""

from __future__ import annotations

import itertools
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .benchmark_suite import BenchmarkSweep, validate_benchmark_frame


CertificationPolicy: TypeAlias = Literal[
    "implemented-circuit",
    "declared-bound-scope",
    "unconstrained",
]


FAMILY_COLORS = {
    "trotter": "#0072B2",
    "multiproduct": "#D55E00",
    "qsvt": "#009E73",
}
_FALLBACK_COLORS = ("#CC79A7", "#E69F00", "#56B4E9", "#000000")
_LINESTYLES = ("-", "--", "-.", ":")
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
_METRIC_LABELS = {
    "t_count": "T count",
    "cnot_count": "CNOT count",
    "total_qubits": "Total qubits",
    "query_count": "Query count",
    "rotation_count": "Rotation count",
    "toffoli_count": "Toffoli count",
    "depth": "Depth",
    "segment_count": "Segment count",
    "mpf_r_error": "MPF error-predicate segment threshold",
    "mpf_r_time_1": "Mizuta first-time-condition segment threshold",
    "mpf_r_time_2": "Mizuta mu-time-condition segment threshold",
}
_MPF_CONSTRAINT_MARKERS = {
    "error": "o",
    "time_1": "^",
    "time_2": "s",
    "joint_p0": "D",
    "commuting_exact": "P",
    "zero_time_exact": "P",
    "multiple": "D",
    "unknown": "X",
}


def _mpf_comparison_keys(frame: pd.DataFrame) -> list[str]:
    keys = [
        "sweep",
        "hamiltonian_model",
        "model_parameters_json",
        "system_qubits",
        "evolution_time",
        "time_scaling_mode",
        "time_scaling_coefficient",
        "target_error",
        "algorithm_error_budget",
        "mpf_term_count",
        "mpf_schedule",
        "rotation_synthesis_error",
    ]
    missing = set(keys) - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"benchmark data cannot pair MPF bounds without: {names}")
    return keys


def compare_mpf_bounds(
    frame: pd.DataFrame,
    metric: str = "segment_count",
) -> pd.DataFrame:
    """Pair rigorous Low and Mizuta rows for the same implemented MPF.

    The comparison is deliberately limited to concrete estimator rows. Policy
    rows are excluded so that a selected best-bound row cannot be paired with
    one of its own candidates.
    """
    validate_benchmark_frame(frame)
    values = frame.copy()
    if "mpf_bound_policy" not in values:
        values["mpf_bound_policy"] = values["bound_method"]
    if "mpf_active_constraints_json" not in values:
        values["mpf_active_constraints_json"] = "[]"
    if "commutator_cap_fallback" not in values:
        values["commutator_cap_fallback"] = False
    values[metric] = _metric_values(values, metric)
    policy = values["mpf_bound_policy"]
    concrete = {
        "low2019-l1-ideal-rigorous",
        "mizuta2026-commutator-ideal-rigorous",
    }
    values = values[
        (values["status"] == "ok")
        & (values["method_family"] == "multiproduct")
        & values["bound_method"].isin(concrete)
        & policy.isin(concrete)
        & values["bound_rigorous"].fillna(False).astype(bool)
        & values[metric].notna()
        & (values[metric] > 0)
    ].copy()
    keys = _mpf_comparison_keys(values)
    if values.duplicated([*keys, "bound_method"]).any():
        raise ValueError("benchmark data contains duplicate concrete MPF bound rows")

    shared = [
        *keys,
        "method_id",
        "bound_method",
        metric,
        "mpf_active_constraints_json",
        "commutator_cap_fallback",
    ]
    low = values[values["bound_method"] == "low2019-l1-ideal-rigorous"][shared]
    mizuta = values[
        values["bound_method"] == "mizuta2026-commutator-ideal-rigorous"
    ][shared]
    paired = low.merge(mizuta, on=keys, how="inner", suffixes=("_low", "_mizuta"))
    if paired.empty:
        raise ValueError("benchmark data contains no matched rigorous Low/Mizuta MPF rows")
    paired = paired.rename(
        columns={
            f"{metric}_low": "low_value",
            f"{metric}_mizuta": "mizuta_value",
            "method_id_low": "low_method_id",
            "method_id_mizuta": "mizuta_method_id",
            "mpf_active_constraints_json_mizuta": "mizuta_active_constraints_json",
            "commutator_cap_fallback_mizuta": "mizuta_commutator_cap_fallback",
        }
    )
    paired["mizuta_to_low_ratio"] = paired["mizuta_value"] / paired["low_value"]
    paired["tighter_bound_method"] = "tie"
    paired.loc[
        paired["low_value"] < paired["mizuta_value"], "tighter_bound_method"
    ] = "low2019-l1-ideal-rigorous"
    paired.loc[
        paired["mizuta_value"] < paired["low_value"], "tighter_bound_method"
    ] = "mizuta2026-commutator-ideal-rigorous"
    return paired.sort_values(keys).reset_index(drop=True)


def _active_constraint_label(raw: object) -> str:
    try:
        constraints = tuple(json.loads(str(raw)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return "unknown"
    if len(constraints) == 1:
        return str(constraints[0])
    if len(constraints) > 1:
        return "multiple"
    return "unknown"


def plot_mpf_crossover(
    frame: pd.DataFrame,
    *,
    sweep: BenchmarkSweep,
    metric: str = "segment_count",
    ax: Axes | None = None,
) -> Figure:
    """Plot the matched Mizuta/Low resource ratio for fixed MPF structures."""
    paired = compare_mpf_bounds(frame, metric)
    if sweep not in {"system-size", "target-error"}:
        raise ValueError("sweep must be 'system-size' or 'target-error'")
    paired = paired[paired["sweep"] == sweep].copy()
    if paired.empty:
        raise ValueError(f"benchmark data contains no matched {sweep!r} MPF rows")
    x_column = "system_qubits" if sweep == "system-size" else "target_error"
    paired["active_constraint"] = paired["mizuta_active_constraints_json"].map(
        _active_constraint_label
    )

    if ax is None:
        figure, axis = plt.subplots(figsize=(9.2, 5.8))
    else:
        axis = ax
        figure = axis.figure

    colors = itertools.cycle(("#D55E00", "#0072B2", "#009E73", "#CC79A7"))
    marker_labels: set[tuple[str, bool]] = set()
    for (term_count, schedule), values in paired.groupby(
        ["mpf_term_count", "mpf_schedule"], dropna=False, sort=True
    ):
        values = values.sort_values(x_column)
        color = next(colors)
        label = f"J={int(term_count)}, schedule={schedule}"
        axis.plot(
            values[x_column],
            values["mizuta_to_low_ratio"],
            color=color,
            linewidth=1.8,
            label=label,
        )
        for _, row in values.iterrows():
            constraint = str(row["active_constraint"])
            fallback = bool(row["mizuta_commutator_cap_fallback"])
            marker = _MPF_CONSTRAINT_MARKERS.get(constraint, "X")
            marker_key = (constraint, fallback)
            marker_label = None
            if marker_key not in marker_labels:
                marker_labels.add(marker_key)
                marker_label = f"active={constraint}" + (" [fallback]" if fallback else "")
            axis.scatter(
                [row[x_column]],
                [row["mizuta_to_low_ratio"]],
                marker=marker,
                s=55,
                facecolors="none" if fallback else color,
                edgecolors=color,
                linewidths=1.3,
                label=marker_label,
                zorder=3,
            )
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2, label="crossover")
    axis.set_xscale("log", base=10)
    axis.set_yscale("log", base=10)
    if sweep == "target-error":
        axis.invert_xaxis()
        axis.set_xlabel("Target simulation error epsilon")
    else:
        axis.set_xlabel("Number of system qubits")
    axis.set_ylabel(f"Mizuta / Low {_METRIC_LABELS.get(metric, metric)}")
    axis.set_title("Rigorous MPF finite-resource crossover")
    axis.grid(True, which="both", alpha=0.28)
    axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def _sweep_frame(frame: pd.DataFrame, sweep: BenchmarkSweep) -> pd.DataFrame:
    validate_benchmark_frame(frame)
    if sweep not in {"system-size", "target-error"}:
        raise ValueError("sweep must be 'system-size' or 'target-error'")
    selected = frame[frame["sweep"] == sweep].copy()
    if selected.empty:
        raise ValueError(f"benchmark data contains no {sweep!r} rows")
    return selected


def _metric_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in frame.columns:
        raise ValueError(f"benchmark metric column not found: {metric}")
    values = pd.to_numeric(frame[metric], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"benchmark metric must be numeric: {metric}")
    return values


def _bound_target_satisfied(frame: pd.DataFrame) -> pd.Series:
    """Return scoped certification status, including early schema-2 data."""
    if "bound_target_satisfied" in frame.columns:
        return frame["bound_target_satisfied"].fillna(False).astype(bool)
    rigorous = frame["bound_rigorous"].fillna(False).astype(bool)
    bound = pd.to_numeric(frame["bound_value"], errors="coerce")
    budget = pd.to_numeric(frame["algorithm_error_budget"], errors="coerce")
    return rigorous & bound.notna() & budget.notna() & (bound <= budget)


def _circuit_target_satisfied(frame: pd.DataFrame) -> pd.Series:
    """Return complete implemented-circuit certification conservatively."""
    if "circuit_target_satisfied" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["circuit_target_satisfied"].fillna(False).astype(bool)


def _certification_mask(
    frame: pd.DataFrame,
    policy: CertificationPolicy,
) -> pd.Series:
    if policy == "implemented-circuit":
        return _circuit_target_satisfied(frame)
    if policy == "declared-bound-scope":
        return _bound_target_satisfied(frame)
    if policy == "unconstrained":
        return pd.Series(True, index=frame.index)
    raise ValueError(
        "certification_policy must be 'implemented-circuit', "
        "'declared-bound-scope', or 'unconstrained'"
    )


def select_best_by_family(
    frame: pd.DataFrame,
    metric: str,
    *,
    sweep: BenchmarkSweep,
    certification_policy: CertificationPolicy = "implemented-circuit",
    rigorous_only: bool | None = None,
) -> pd.DataFrame:
    """Select the minimum method per family under an explicit scope policy.

    ``rigorous_only`` is the backward-compatible spelling: true maps to
    ``declared-bound-scope`` and false maps to ``unconstrained``.
    """
    selected = _sweep_frame(frame, sweep)
    selected[metric] = _metric_values(selected, metric)
    if rigorous_only is not None:
        if certification_policy != "implemented-circuit":
            raise ValueError(
                "do not combine rigorous_only with certification_policy"
            )
        warnings.warn(
            "rigorous_only is deprecated; use certification_policy explicitly",
            DeprecationWarning,
            stacklevel=2,
        )
        certification_policy = (
            "declared-bound-scope" if rigorous_only else "unconstrained"
        )
    selected = selected[_certification_mask(selected, certification_policy)]
    selected = selected[
        (selected["status"] == "ok")
        & selected[metric].notna()
        & (selected[metric] > 0)
    ]
    if selected.empty:
        raise ValueError(
            "no positive successful values for "
            f"metric {metric!r} under policy {certification_policy!r}"
        )
    x_column = "system_qubits" if sweep == "system-size" else "target_error"
    context_columns = [
        "hamiltonian_model",
        "model_parameters_json",
        "time_scaling_mode",
        "time_scaling_coefficient",
    ]
    context_columns.append("target_error" if sweep == "system-size" else "system_qubits")
    group_columns = [x_column, "method_family", *context_columns]
    indices = selected.groupby(group_columns, dropna=False, sort=True)[metric].idxmin()
    result = selected.loc[indices].copy().sort_values(group_columns)
    result["selected_method_id"] = result["method_id"]
    result["selected_method_label"] = result["method_label"]
    family_labels = {
        "trotter": "Trotter",
        "multiproduct": "MPF",
        "qsvt": "QSVT",
    }
    result["summary_label"] = result["method_family"].map(family_labels).fillna(
        "Best " + result["method_family"].astype(str)
    )
    result["certification_policy"] = certification_policy
    return result


def _context_title(frame: pd.DataFrame, sweep: BenchmarkSweep) -> str:
    details: list[str] = []
    models = frame["hamiltonian_model"].dropna().astype(str).unique()
    if len(models) == 1:
        details.append(models[0])
    if sweep == "system-size":
        modes = frame["time_scaling_mode"].dropna().astype(str).unique()
        coefficients = frame["time_scaling_coefficient"].dropna().unique()
        if len(modes) == 1 and len(coefficients) == 1:
            coefficient = float(coefficients[0])
            if modes[0] == "proportional":
                details.append(f"t={coefficient:g}×n")
            else:
                details.append(f"t={coefficient:g}")
        errors = frame["target_error"].dropna().unique()
        if len(errors) == 1:
            details.append(f"ε={float(errors[0]):g}")
    else:
        sizes = frame["system_qubits"].dropna().unique()
        times = frame["evolution_time"].dropna().unique()
        if len(sizes) == 1:
            details.append(f"n={int(sizes[0])}")
        if len(times) == 1:
            details.append(f"t={float(times[0]):g}")
    return ", ".join(details)


def _series_columns(series_by: str | Sequence[str], frame: pd.DataFrame) -> list[str]:
    columns = [series_by] if isinstance(series_by, str) else list(series_by)
    if not columns:
        raise ValueError("series_by must contain at least one column")
    missing = set(columns) - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"series columns not found: {names}")
    return columns


def _series_label(key: Any, columns: Sequence[str]) -> str:
    values = key if isinstance(key, tuple) else (key,)
    if len(columns) == 1:
        return str(values[0])
    return ", ".join(f"{column}={value}" for column, value in zip(columns, values))


def plot_benchmark(
    frame: pd.DataFrame,
    *,
    sweep: BenchmarkSweep,
    metric: str,
    xscale: str | None = None,
    xbase: float | None = None,
    yscale: str = "log",
    ybase: float = 10,
    summary: bool = False,
    selection_metric: str | None = None,
    certification_policy: CertificationPolicy = "implemented-circuit",
    series_by: str | Sequence[str] = "method_label",
    ax: Axes | None = None,
) -> Figure:
    """Plot a benchmark metric in detailed or best-by-family summary form.

    When ``summary`` is true, ``selection_metric`` chooses the resource used
    to select one method per family and x value. It defaults to ``metric`` for
    backward compatibility, while allowing (for example) a CNOT plot to show
    the same methods selected for a T-count summary.
    """
    selected = _sweep_frame(frame, sweep)
    selected[metric] = _metric_values(selected, metric)
    if summary:
        resolved_selection_metric = selection_metric or metric
        selected = select_best_by_family(
            selected,
            resolved_selection_metric,
            sweep=sweep,
            certification_policy=certification_policy,
        )
        if series_by == "method_label":
            series_by = "summary_label"
    x_column = "system_qubits" if sweep == "system-size" else "target_error"
    selected = selected[
        (selected["status"] == "ok")
        & selected[metric].notna()
        & (selected[metric] > 0)
    ].copy()
    if selected.empty:
        raise ValueError(f"no positive successful values for metric {metric!r}")
    columns = _series_columns(series_by, selected)

    if ax is None:
        figure, axis = plt.subplots(figsize=(9.2, 5.8))
    else:
        axis = ax
        figure = axis.figure

    color_cycle = itertools.cycle(_FALLBACK_COLORS)
    family_colors: dict[str, str] = dict(FAMILY_COLORS)
    family_variant: dict[str, int] = {}
    group_key: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, values in selected.groupby(group_key, dropna=False, sort=False):
        values = values.sort_values(x_column)
        family_values = values["method_family"].dropna().astype(str).unique()
        family = family_values[0] if len(family_values) == 1 else str(key)
        color = family_colors.setdefault(family, next(color_cycle))
        variant = family_variant.get(family, 0)
        family_variant[family] = variant + 1
        label = _series_label(key, columns)
        heuristic = not bool(_bound_target_satisfied(values).all())
        if not summary and heuristic and "heuristic" not in label.lower():
            label += " [heuristic/non-certified]"
        if not summary and family == "multiproduct":
            circuit_rigorous = (
                values["circuit_bound_rigorous"].fillna(False).astype(bool)
                if "circuit_bound_rigorous" in values
                else pd.Series(False, index=values.index)
            )
            if not bool(circuit_rigorous.all()):
                label += " [ideal bound; circuit unproven]"
        axis.plot(
            values[x_column],
            values[metric],
            label=label,
            color=color,
            linestyle=(
                ":" if heuristic else _LINESTYLES[variant % len(_LINESTYLES)]
            ),
            marker=_MARKERS[variant % len(_MARKERS)],
            linewidth=2.0 if summary else 1.7,
            alpha=0.7 if heuristic else 1.0,
        )
        if summary:
            for _, row in values.iterrows():
                annotation = None
                if family == "trotter" and pd.notna(row.get("trotter_order")):
                    annotation = f"p={int(row['trotter_order'])}"
                elif family == "multiproduct" and pd.notna(row.get("mpf_term_count")):
                    annotation = f"J={int(row['mpf_term_count'])}"
                if annotation is not None:
                    axis.annotate(
                        annotation,
                        (row[x_column], row[metric]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        clip_on=True,
                    )

    resolved_xscale = xscale or "log"
    if resolved_xscale not in {"linear", "log"}:
        raise ValueError("xscale must be 'linear' or 'log'")
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'")
    if resolved_xscale == "log":
        axis.set_xscale("log", base=xbase or 10)
    else:
        axis.set_xscale("linear")
    if yscale == "log":
        axis.set_yscale("log", base=ybase)
    else:
        axis.set_yscale("linear")
    if sweep == "target-error":
        axis.invert_xaxis()
        axis.set_xlabel("Target simulation error ε (smaller = higher precision)")
        sweep_title = "Target-error scaling"
    else:
        axis.set_xlabel("Number of system qubits")
        sweep_title = "System-size scaling"
    axis.set_ylabel(_METRIC_LABELS.get(metric, metric.replace("_", " ")))
    qualifier = (
        "best by "
        f"{_METRIC_LABELS.get(selection_metric or metric, selection_metric or metric)}; "
        f"policy={certification_policy}"
        if summary
        else "all methods; certification scope shown in labels"
    )
    context = _context_title(selected, sweep)
    title = f"{sweep_title}: {_METRIC_LABELS.get(metric, metric)} ({qualifier})"
    axis.set_title(title + (f"\n{context}" if context else ""))
    axis.grid(True, which="both", alpha=0.28)
    axis.legend(fontsize="small")

    source = _sweep_frame(frame, sweep)
    failed_count = int((source["status"] == "error").sum())
    nonpositive_count = int(
        (
            (source["status"] == "ok")
            & pd.to_numeric(source[metric], errors="coerce").notna()
            & (pd.to_numeric(source[metric], errors="coerce") <= 0)
        ).sum()
    )
    if failed_count or nonpositive_count:
        message = "; ".join(
            part
            for part in (
                f"{failed_count} failed rows" if failed_count else "",
                f"{nonpositive_count} nonpositive values" if nonpositive_count else "",
            )
            if part
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        axis.text(
            0.01,
            0.01,
            message,
            transform=axis.transAxes,
            fontsize=8,
            color="#7A1F1F",
            verticalalignment="bottom",
        )
    figure.tight_layout()
    return figure


def save_benchmark_plots(
    frame: pd.DataFrame,
    *,
    output_directory: str | Path,
    output_formats: Sequence[str] = ("png", "pdf"),
    summary: bool = False,
    certification_policy: CertificationPolicy = "implemented-circuit",
) -> tuple[Path, ...]:
    """Write standard T/CNOT figures for every sweep present in a DataFrame."""
    validate_benchmark_frame(frame)
    formats = list(output_formats)
    unknown = set(formats) - {"png", "pdf", "svg"}
    if not formats or unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported or empty output formats: {names}")
    target = Path(output_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for sweep, prefix in (
        ("system-size", "system_size"),
        ("target-error", "target_error"),
    ):
        if sweep not in set(frame["sweep"]):
            continue
        for metric in ("t_count", "cnot_count"):
            for is_summary in ((False, True) if summary else (False,)):
                figure = plot_benchmark(
                    frame,
                    sweep=sweep,
                    metric=metric,
                    summary=is_summary,
                    certification_policy=certification_policy,
                )
                suffix = "_summary" if is_summary else ""
                for output_format in formats:
                    output = target / f"{prefix}_{metric}{suffix}.{output_format}"
                    figure.savefig(
                        output,
                        dpi=220 if output_format == "png" else None,
                        bbox_inches="tight",
                    )
                    outputs.append(output)
                figure.clear()
    return tuple(outputs)
