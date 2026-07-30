"""Configurable plots for schema-2 benchmark data."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .benchmark_suite import BenchmarkSweep, validate_benchmark_frame


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
}


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


def select_best_by_family(
    frame: pd.DataFrame,
    metric: str,
    *,
    sweep: BenchmarkSweep,
) -> pd.DataFrame:
    """Select the minimum successful method and retain its identity at every point."""
    selected = _sweep_frame(frame, sweep)
    selected[metric] = _metric_values(selected, metric)
    selected = selected[
        (selected["status"] == "ok")
        & selected[metric].notna()
        & (selected[metric] > 0)
    ]
    if selected.empty:
        raise ValueError(f"no positive successful values for metric {metric!r}")
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
    result["summary_label"] = result["method_family"].map(
        {
            "trotter": "Best evaluated Trotter",
            "multiproduct": "Best evaluated MPF",
            "qsvt": "QSVT",
        }
    ).fillna("Best " + result["method_family"].astype(str))
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
    series_by: str | Sequence[str] = "method_label",
    ax: Axes | None = None,
) -> Figure:
    """Plot any positive numeric benchmark metric with configurable axes and grouping."""
    selected = _sweep_frame(frame, sweep)
    selected[metric] = _metric_values(selected, metric)
    if summary:
        selected = select_best_by_family(selected, metric, sweep=sweep)
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
        if summary and "selected_method_label" in values:
            methods = list(dict.fromkeys(values["selected_method_label"].astype(str)))
            if len(methods) > 1:
                label += " [" + " → ".join(methods) + "]"
        axis.plot(
            values[x_column],
            values[metric],
            label=label,
            color=color,
            linestyle=_LINESTYLES[variant % len(_LINESTYLES)],
            marker=_MARKERS[variant % len(_MARKERS)],
            linewidth=2.0 if summary else 1.7,
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
    qualifier = "best of evaluated methods" if summary else "all methods"
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
