"""Build the deterministic fail-closed N=9 calibration checkpoint report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from hamiltonian_resources.calibration_checkpoint import analyze_n9_checkpoint


MODEL_LABELS = {
    "transverse_field_ising": "TFIM",
    "heisenberg_chain": "Heisenberg",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _scientific(value: float) -> str:
    return f"{value:.8e}"


def _parameters(payload: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{name}={float(value):.6e}" for name, value in payload["parameters"].items()
    )


def _markdown(checkpoint: Mapping[str, Any]) -> str:
    rows = [
        row
        for row in checkpoint["rows"]
        if row["numerical_N4_through_N9_passed"]
    ]
    complete = [
        row
        for row in rows
        if row["all_current_numerical_and_time_law_gates_passed"]
    ]
    pending = [
        row
        for row in rows
        if row["time_law_gate"]["status"] == "pending-required-N8-expansion"
    ]
    rejected = [
        row
        for row in checkpoint["rows"]
        if not row["numerical_N4_through_N9_passed"]
    ]
    lines = [
        "# High-order empirical MPF calibration: N=9 checkpoint",
        "",
        "This is a fail-closed checkpoint. No N=10 task was launched, no acceptance "
        "criterion was relaxed, and no row is eligible for `reviewed` status because "
        "the required N=6..10 shifted window and N=10 holdout are unavailable.",
        "",
        "## Gate status",
        "",
        "Rows passing all currently required numerical and time-law gates: "
        + ", ".join(
            f"{MODEL_LABELS[row['model']]} 2J={row['formal_order']}"
            for row in complete
        )
        + ".",
        "",
    ]
    if pending:
        lines.extend(
            [
                "The following rows pass every N=4..9 numerical window but are not "
                "gate-complete: "
                + ", ".join(
                    f"{MODEL_LABELS[row['model']]} 2J={row['formal_order']}"
                    for row in pending
                )
                + ". The Heisenberg N=8 sentinel failure requires the predeclared "
                "all-order N=8 time-law expansion before these rows can pass.",
                "",
            ]
        )
    lines.extend(
        [
            "Rows removed before the fit checkpoint:",
            "",
            "| Model | 2J | Missing or rejected primary sizes | Time-law status |",
            "|---|---:|---|---|",
        ]
    )
    for row in rejected:
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {row['formal_order']} | "
            f"{', '.join(str(size) for size in row['missing_or_rejected_sizes'])} | "
            f"{row['time_law_gate']['status']} |"
        )
    lines.append("")
    lines.extend(
        [
            "## B_2J(N) observations",
            "",
            "| Model | 2J | N=4 | N=5 | N=6 | N=7 | N=8 | N=9 | Time gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        values = row["coefficients_b_2j"]
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS[row["model"]],
                    str(row["formal_order"]),
                    *[_scientific(float(values[str(size)])) for size in range(4, 10)],
                    row["time_law_gate"]["status"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Fits use the configured weighted log-residual objective. Relative weights "
            "combine precision convergence, accepted plateau spread, and accepted "
            "time-law spread in quadrature.",
            "",
            "## Candidate fits using N=4..8 with N=9 as the available holdout",
            "",
            "| Model | 2J | Law | Parameters | AICc | N=9 error | B(50) | B(100) | "
            "Two-window necessary conditions |",
            "|---|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        for candidate in row["fits"]:
            training = candidate["stages"][1]
            stability = candidate["available_shifted_window_stability"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        MODEL_LABELS[row["model"]],
                        str(row["formal_order"]),
                        candidate["model"],
                        _parameters(training),
                        (
                            "n/a"
                            if training["aicc"] is None
                            else f"{float(training['aicc']):.3f}"
                        ),
                        _percent(
                            float(
                                candidate[
                                    "n9_holdout_relative_error_from_N4_through_N8"
                                ]
                            )
                        ),
                        _scientific(float(training["predictions"]["50"])),
                        _scientific(float(training["predictions"]["100"])),
                        (
                            "pass (partial only)"
                            if stability["necessary_conditions_pass"]
                            else "fail"
                        ),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Sensitivity to adding the largest available points",
            "",
            "| Model | 2J | Law | Add N=8: ΔB(50) | Add N=8: ΔB(100) | "
            "Add N=9: ΔB(50) | Add N=9: ΔB(100) | Parameter changes on adding N=9 |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        for candidate in row["fits"]:
            add8, add9 = candidate["updates"]
            parameter_changes = ", ".join(
                f"{name} {_percent(change['relative_change'])}"
                for name, change in add9["parameter_changes"].items()
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        MODEL_LABELS[row["model"]],
                        str(row["formal_order"]),
                        candidate["model"],
                        _percent(add8["prediction_changes"]["50"]["relative_change"]),
                        _percent(add8["prediction_changes"]["100"]["relative_change"]),
                        _percent(add9["prediction_changes"]["50"]["relative_change"]),
                        _percent(add9["prediction_changes"]["100"]["relative_change"]),
                        parameter_changes,
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Available shifted-window diagnostics",
            "",
            "Only N=4..8 and N=5..9 can be compared. Every `pass` below is a necessary "
            "condition only; the full gate remains unevaluated until N=6..10 exists.",
            "",
            "| Model | 2J | Law | S(12) | S(20) | S(50) | S(100) | Parameter span / "
            "limit | Residual drift free | Partial result |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        for candidate in row["fits"]:
            stability = candidate["available_shifted_window_stability"]
            spreads = stability["prediction_spreads"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        MODEL_LABELS[row["model"]],
                        str(row["formal_order"]),
                        candidate["model"],
                        *[_percent(float(spreads[str(size)]["spread"])) for size in (12, 20, 50, 100)],
                        f"{float(stability['parameter_span']):.4f} / "
                        f"{float(stability['parameter_limit']):.4f}",
                        str(bool(stability["residual_drift_free"])),
                        (
                            "pass"
                            if stability["necessary_conditions_pass"]
                            else "fail"
                        ),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "No row has more than one provisionally viable model with the current data. "
            "TFIM 2J=24 and 28 each retain only the affine candidate. Thus their N=10 "
            "points would complete the holdout/stability evidence, not resolve a current "
            "affine-versus-power ambiguity.",
            "",
            "## Wall-time scaling and N=10 projection",
            "",
            "The row estimate repeats the measured N=8→9 task-time factor once. Task "
            "time is the sum of all precision attempts and segment points in the shard.",
            "",
            "| Model | 2J | N=8 task min | N=9 task min | Factor | Projected N=10 h | "
            "Scientific disposition |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        wall = row["wall_time"]
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS[row["model"]],
                    str(row["formal_order"]),
                    f"{float(wall['N8_seconds']) / 60:.2f}",
                    f"{float(wall['N9_seconds']) / 60:.2f}",
                    f"{float(wall['N9_over_N8_factor']):.2f}×",
                    f"{float(wall['projected_N10_seconds_repeating_last_factor']) / 3600:.2f}",
                    row["n10_scientific_need"],
                ]
            )
            + " |"
        )
    summary = checkpoint["summary"]
    lines.extend(
        [
            "",
            f"The measured two-batch N=9 makespan reconstructed from shard timings was "
            f"{float(summary['measured_N9_two_batch_makespan_seconds']) / 3600:.2f} h. "
            f"Running N=10 for all seven numerically surviving rows is projected at "
            f"{float(summary['projected_N10_all_numerical_rows_task_hours']):.1f} task-h, "
            f"or {float(summary['projected_N10_all_numerical_rows_three_worker_hours']):.1f} h "
            "under ideal three-worker scheduling. The narrower total for gate-complete "
            f"rows that remain scientifically informative is "
            f"{float(summary['projected_N10_informative_gate_complete_rows_task_hours']):.1f} "
            f"task-h / {float(summary['projected_N10_informative_gate_complete_rows_three_worker_hours']):.1f} "
            "ideal three-worker hours.",
            "",
            "No N=10 run should be started from this report. Heisenberg rows first need "
            "their required N=8 time-law expansion; TFIM N=10 candidates are identified "
            "row by row above. No high-order coefficient is promoted at this checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduced", type=Path, required=True)
    parser.add_argument("--assembled", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    reduced = json.loads(arguments.reduced.read_text(encoding="utf-8"))
    assembled = json.loads(arguments.assembled.read_text(encoding="utf-8"))
    checkpoint = analyze_n9_checkpoint(reduced, assembled)
    _write_json(arguments.output_json, checkpoint)
    arguments.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_markdown.write_text(
        _markdown(checkpoint), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
