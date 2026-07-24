"""Publication-friendly plots loaded from persisted benchmark data."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .benchmark_suite import METHOD_LABELS, load_benchmark_data


FAMILY_COLORS = {
    "trotter": "#0072B2",  # Okabe-Ito blue
    "multiproduct": "#D55E00",  # Okabe-Ito vermillion
    "qsvt": "#009E73",  # Okabe-Ito bluish green
}

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "Trotter p=1": {
        "color": FAMILY_COLORS["trotter"],
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.7,
    },
    "Trotter p=2": {
        "color": FAMILY_COLORS["trotter"],
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.7,
    },
    "Trotter p=4": {
        "color": FAMILY_COLORS["trotter"],
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 1.7,
    },
    "Trotter p=6": {
        "color": FAMILY_COLORS["trotter"],
        "linestyle": ":",
        "marker": "D",
        "linewidth": 1.9,
    },
    "MPF m=3": {
        "color": FAMILY_COLORS["multiproduct"],
        "linestyle": "-",
        "marker": "v",
        "linewidth": 1.7,
    },
    "MPF m=5": {
        "color": FAMILY_COLORS["multiproduct"],
        "linestyle": "--",
        "marker": "P",
        "linewidth": 1.7,
    },
    "MPF m=7": {
        "color": FAMILY_COLORS["multiproduct"],
        "linestyle": "-.",
        "marker": "X",
        "linewidth": 1.7,
    },
    "QSVT": {
        "color": FAMILY_COLORS["qsvt"],
        "linestyle": "-",
        "marker": "*",
        "linewidth": 2.4,
        "markersize": 9,
    },
}

SUMMARY_STYLES: dict[str, dict[str, Any]] = {
    "Best Trotter (p=1,2,4,6)": {
        "color": FAMILY_COLORS["trotter"],
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.1,
    },
    "Best MPF (m=3,5,7)": {
        "color": FAMILY_COLORS["multiproduct"],
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.1,
    },
    "QSVT": METHOD_STYLES["QSVT"],
}

_METRIC_LABELS = {"t_count": "T count", "cnot_count": "CNOT count"}
_SUPPORTED_FORMATS = {"png", "pdf", "svg"}


def _unique_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"plotting requires one {column}; found {len(values)}")
    return values[0]


def _format_number(value: float) -> str:
    return f"{float(value):g}"


def _plot_context(frame: pd.DataFrame) -> tuple[str, str, str]:
    sweep = str(_unique_value(frame, "sweep"))
    model = str(_unique_value(frame, "hamiltonian_model"))
    parameters = json.loads(str(_unique_value(frame, "model_parameters_json")))
    parameter_text = ", ".join(
        f"{key}={value}" for key, value in sorted(parameters.items())
    )
    display_model = {
        "transverse_field_ising": "Transverse-field Ising",
        "heisenberg_chain": "Heisenberg chain",
    }.get(model, model)
    model_text = f"{display_model} ({parameter_text})" if parameter_text else display_model
    time = _format_number(float(_unique_value(frame, "evolution_time")))
    if sweep == "system-size":
        epsilon = _format_number(float(_unique_value(frame, "target_error")))
        fixed_text = f"t={time}, target error ε={epsilon}"
        x_column = "system_qubits"
    elif sweep == "target-error":
        size = int(_unique_value(frame, "system_qubits"))
        fixed_text = f"n={size} system qubits, t={time}"
        x_column = "target_error"
    else:
        raise ValueError(f"unsupported sweep: {sweep}")
    return sweep, x_column, f"{model_text}\n{fixed_text}"


def _summary_series(
    frame: pd.DataFrame, x_column: str, metric: str
) -> tuple[tuple[str, pd.DataFrame], ...]:
    successful = frame[
        (frame["status"] == "ok")
        & frame[metric].notna()
        & (frame[metric] > 0)
    ]
    series: list[tuple[str, pd.DataFrame]] = []
    for family, label in (
        ("trotter", "Best Trotter (p=1,2,4,6)"),
        ("multiproduct", "Best MPF (m=3,5,7)"),
    ):
        family_rows = successful[successful["method_family"] == family]
        minimum = (
            family_rows.groupby(x_column, as_index=False, sort=True)[metric].min()
            if not family_rows.empty
            else family_rows[[x_column, metric]]
        )
        series.append((label, minimum))
    qsvt = successful[successful["method_label"] == "QSVT"][[x_column, metric]]
    series.append(("QSVT", qsvt.sort_values(x_column)))
    return tuple(series)


def create_benchmark_figure(
    frame: pd.DataFrame,
    metric: str,
    *,
    summary: bool = False,
) -> Figure:
    """Create one figure from an already loaded benchmark DataFrame."""
    if metric not in _METRIC_LABELS:
        raise ValueError("metric must be 't_count' or 'cnot_count'")
    sweep, x_column, context = _plot_context(frame)
    figure = Figure(figsize=(9.2, 5.8))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    missing_labels: list[str] = []

    if summary:
        plot_series = _summary_series(frame, x_column, metric)
        styles: Mapping[str, Mapping[str, Any]] = SUMMARY_STYLES
    else:
        plot_series = tuple(
            (
                label,
                frame[
                    (frame["method_label"] == label)
                    & (frame["status"] == "ok")
                    & frame[metric].notna()
                    & (frame[metric] > 0)
                ][[x_column, metric]].sort_values(x_column),
            )
            for label in METHOD_LABELS
        )
        styles = METHOD_STYLES

    for label, values in plot_series:
        style = dict(styles[label])
        if values.empty:
            missing_labels.append(label)
            axis.plot([], [], label=label, **style)
            continue
        axis.plot(values[x_column], values[metric], label=label, **style)

    axis.set_yscale("log")
    axis.set_ylabel(_METRIC_LABELS[metric])
    if sweep == "system-size":
        axis.set_xlabel("Number of system qubits")
        axis.xaxis.get_major_locator().set_params(integer=True)
        sweep_title = "System-size scaling"
    else:
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("Target simulation error ε (smaller = higher precision)")
        sweep_title = "Target-error scaling"
    qualifier = "best of evaluated configurations" if summary else "all configurations"
    axis.set_title(f"{sweep_title}: {_METRIC_LABELS[metric]} ({qualifier})\n{context}")
    axis.grid(True, which="both", alpha=0.28)
    axis.legend(ncol=2 if not summary else 1, fontsize="small")

    failed_count = int((frame["status"] == "error").sum())
    nonpositive_count = int(
        ((frame["status"] == "ok") & frame[metric].notna() & (frame[metric] <= 0)).sum()
    )
    if failed_count or nonpositive_count or missing_labels:
        details = []
        if failed_count:
            details.append(f"{failed_count} failed rows")
        if nonpositive_count:
            details.append(f"{nonpositive_count} nonpositive values")
        if missing_labels:
            details.append("missing curves: " + ", ".join(missing_labels))
        message = "; ".join(details) + "; see benchmark CSV"
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


def _load_sidecar(data_path: Path) -> dict[str, Any]:
    metadata_path = data_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"benchmark metadata sidecar not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or "configuration" not in metadata:
        raise ValueError(f"invalid benchmark metadata sidecar: {metadata_path}")
    return metadata


def plot_saved_benchmark(
    data_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    output_formats: tuple[str, ...] | list[str] | None = None,
    summary: bool | None = None,
) -> tuple[Path, ...]:
    """Load one saved sweep and write its T/CNOT plots.

    No resource estimator is called. Plot defaults are read from the metadata
    sidecar written alongside the CSV.
    """
    csv_path = Path(data_path).resolve()
    frame = load_benchmark_data(csv_path)
    metadata = _load_sidecar(csv_path)
    stored_config = metadata["configuration"]
    formats = tuple(output_formats or stored_config.get("output_formats", ("png", "pdf")))
    unknown_formats = set(formats) - _SUPPORTED_FORMATS
    if not formats or unknown_formats:
        names = ", ".join(sorted(unknown_formats))
        raise ValueError(f"unsupported or empty output formats: {names}")
    include_summary = (
        bool(stored_config.get("generate_summary_plots", False))
        if summary is None
        else summary
    )
    target_directory = (
        Path(output_directory).resolve()
        if output_directory is not None
        else csv_path.parent
    )
    target_directory.mkdir(parents=True, exist_ok=True)
    sweep = str(_unique_value(frame, "sweep"))
    prefix = "system_size" if sweep == "system-size" else "target_error"
    outputs: list[Path] = []
    for metric in _METRIC_LABELS:
        for is_summary in ((False, True) if include_summary else (False,)):
            figure = create_benchmark_figure(frame, metric, summary=is_summary)
            suffix = "_summary" if is_summary else ""
            for output_format in formats:
                output = target_directory / f"{prefix}_{metric}{suffix}.{output_format}"
                figure.savefig(
                    output,
                    dpi=220 if output_format == "png" else None,
                    bbox_inches="tight",
                )
                outputs.append(output)
            figure.clear()
    return tuple(outputs)
