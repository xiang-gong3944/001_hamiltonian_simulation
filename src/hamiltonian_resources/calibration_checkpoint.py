"""Deterministic partial-study diagnostics for the high-order N=9 checkpoint."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration_study import SizeLawFit, SizeLawModel, fit_size_law
from .empirical import canonical_json_digest


CHECKPOINT_MODELS: tuple[SizeLawModel, ...] = (
    "affine",
    "power",
    "power-plus-offset",
)
_ANCHOR_LIMITS = ((12, 0.10), (20, 0.15), (50, 0.25), (100, 0.35))


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _fit_payload(fit: SizeLawFit) -> dict[str, Any]:
    return {
        "model": fit.model,
        "parameters": dict(fit.parameters),
        "sizes": list(fit.sizes),
        "fitted_values": list(fit.fitted_values),
        "relative_residuals": list(fit.relative_residuals),
        "weighted_log_residual_sum_squares": fit.log_residual_sum_squares,
        "aicc": _finite(fit.aicc),
        "converged": fit.converged,
        "predictions": {str(size): fit.at(size) for size in (12, 20, 50, 100)},
    }


def _has_monotone_residual_drift(fit: SizeLawFit) -> bool:
    residuals = fit.relative_residuals[-3:]
    same_sign = all(value > 0 for value in residuals) or all(
        value < 0 for value in residuals
    )
    magnitudes = tuple(abs(value) for value in residuals)
    return max(magnitudes) > 1e-4 and same_sign and all(
        second >= first for first, second in zip(magnitudes, magnitudes[1:])
    )


def _fit_update(before: SizeLawFit, after: SizeLawFit, added_size: int) -> dict[str, Any]:
    before_parameters = dict(before.parameters)
    after_parameters = dict(after.parameters)
    parameter_changes = {}
    for name, old_value in before_parameters.items():
        new_value = after_parameters[name]
        parameter_changes[name] = {
            "before": old_value,
            "after": new_value,
            "absolute_change": new_value - old_value,
            "relative_change": (
                (new_value - old_value) / abs(old_value)
                if abs(old_value) > 1e-300
                else None
            ),
        }
    prediction_changes = {}
    for size in (50, 100):
        old_value = before.at(size)
        new_value = after.at(size)
        prediction_changes[str(size)] = {
            "before": old_value,
            "after": new_value,
            "relative_change": (new_value - old_value) / old_value,
        }
    return {
        "added_size": added_size,
        "parameter_changes": parameter_changes,
        "prediction_changes": prediction_changes,
    }


def available_shifted_window_stability(
    sizes: tuple[int, ...],
    coefficients_b_2j: tuple[float, ...],
    model: SizeLawModel,
    *,
    weights: tuple[float, ...] | None,
    reviewed_size_max: int,
) -> dict[str, Any]:
    """Evaluate necessary two-window conditions without claiming the N=4..10 gate."""
    lookup = dict(zip(sizes, coefficients_b_2j, strict=True))
    weight_lookup = (
        dict(zip(sizes, weights, strict=True)) if weights is not None else None
    )
    windows = (tuple(range(4, 9)), tuple(range(5, 10)))
    if any(any(size not in lookup for size in window) for window in windows):
        raise ValueError("the N=9 checkpoint requires observations for N=4..9")
    fits = tuple(
        fit_size_law(
            window,
            tuple(lookup[size] for size in window),
            model,
            weights=(
                tuple(weight_lookup[size] for size in window)
                if weight_lookup is not None
                else None
            ),
            reviewed_size_max=reviewed_size_max,
        )
        for window in windows
    )
    applicable_limits = {
        size: limit for size, limit in _ANCHOR_LIMITS if size <= reviewed_size_max
    }
    if reviewed_size_max not in applicable_limits:
        lower = max(
            (point for point in _ANCHOR_LIMITS if point[0] < reviewed_size_max),
            default=_ANCHOR_LIMITS[0],
        )
        upper = min(
            (point for point in _ANCHOR_LIMITS if point[0] > reviewed_size_max),
            default=_ANCHOR_LIMITS[-1],
        )
        if lower[0] == upper[0]:
            limit = lower[1]
        else:
            fraction = math.log(reviewed_size_max / lower[0]) / math.log(
                upper[0] / lower[0]
            )
            limit = lower[1] + fraction * (upper[1] - lower[1])
        applicable_limits[reviewed_size_max] = limit
    spreads = {}
    prediction_stable = True
    for size, limit in sorted(applicable_limits.items()):
        predictions = tuple(fit.at(size) for fit in fits)
        spread = (max(predictions) - min(predictions)) / float(np.median(predictions))
        spreads[str(size)] = {
            "spread": spread,
            "limit": limit,
            "within_limit": spread <= limit,
        }
        prediction_stable &= spread <= limit
    if model == "affine":
        parameters = tuple(dict(fit.parameters)["slope"] for fit in fits)
        parameter_span = (max(parameters) - min(parameters)) / float(
            np.median(parameters)
        )
        parameter_limit = 0.20
    else:
        parameters = tuple(dict(fit.parameters)["exponent"] for fit in fits)
        parameter_span = max(parameters) - min(parameters)
        parameter_limit = 0.15
    parameter_stable = parameter_span <= parameter_limit
    residual_drift_free = not any(_has_monotone_residual_drift(fit) for fit in fits)
    necessary_conditions_pass = (
        all(fit.converged for fit in fits)
        and prediction_stable
        and parameter_stable
        and residual_drift_free
    )
    return {
        "available_windows": [[4, 8], [5, 9]],
        "missing_required_window": [6, 10],
        "window_fits": [_fit_payload(fit) for fit in fits],
        "prediction_spreads": spreads,
        "parameter_span": parameter_span,
        "parameter_limit": parameter_limit,
        "parameter_stable": parameter_stable,
        "residual_drift_free": residual_drift_free,
        "necessary_conditions_pass": necessary_conditions_pass,
        "full_gate_evaluated": False,
        "full_gate_passed": False,
        "status": "incomplete-without-N10",
    }


def _time_gate_status(
    configuration: Mapping[str, Any],
    assembled: Mapping[str, Any],
    model: str,
    formal_order: int,
) -> dict[str, Any]:
    checks = {
        (str(row["model"]), int(row["formal_order"]), int(row["system_size"])): row
        for row in assembled["time_law_checks"]
    }
    configured_n8 = {
        int(row["formal_order"])
        for row in configuration.get("time_law_checks", [])
        if str(row["model"]) == model and int(row["system_size"]) == 8
    }
    sentinel_rows = [checks.get((model, order, 8)) for order in sorted(configured_n8)]
    sentinel_passed = bool(sentinel_rows) and all(
        row is not None and bool(row["accepted"]) for row in sentinel_rows
    )
    n4 = checks.get((model, formal_order, 4))
    n8 = checks.get((model, formal_order, 8))
    if n4 is None:
        status = "missing-N4-time-law"
    elif not bool(n4["accepted"]):
        status = "failed"
    elif sentinel_passed:
        status = "passed"
    elif n8 is not None and bool(n8["accepted"]):
        status = "passed"
    elif n8 is not None and not bool(n8["accepted"]):
        status = "failed"
    else:
        status = "pending-required-N8-expansion"
    return {
        "status": status,
        "n4": n4,
        "n8": n8,
        "n8_sentinel_orders": sorted(configured_n8),
        "n8_sentinel_set_passed": sentinel_passed,
        "expansion_required": not sentinel_passed,
    }


def _row_weights(
    rows: Sequence[Mapping[str, Any]],
    time_gate: Mapping[str, Any],
    precision_lookup: Mapping[tuple[str, int, int, int], float],
) -> tuple[tuple[float, ...], tuple[dict[str, float], ...]]:
    accepted_checks = [
        value
        for value in (time_gate.get("n4"), time_gate.get("n8"))
        if value is not None and bool(value["accepted"])
    ]
    time_spread = max(
        (
            float(row["maximum_relative_coefficient_deviation"])
            for row in accepted_checks
        ),
        default=0.0,
    )
    weights = []
    components = []
    for row in rows:
        key_prefix = (
            str(row["model"]),
            int(row["formal_order"]),
            int(row["system_size"]),
        )
        precision = max(
            (
                precision_lookup.get((*key_prefix, int(segment)), 0.0)
                for segment in row["segments"]
            ),
            default=0.0,
        )
        plateau = float(row["maximum_relative_deviation"])
        combined = max(math.sqrt(plateau**2 + precision**2 + time_spread**2), 1e-8)
        weights.append(1.0 / combined**2)
        components.append(
            {
                "plateau_spread": plateau,
                "precision_change": precision,
                "time_law_spread": time_spread,
                "combined_relative_uncertainty": combined,
            }
        )
    return tuple(weights), tuple(components)


def _primary_task_timings(reduced: Mapping[str, Any]) -> dict[tuple[str, int, int], float]:
    return {
        (
            str(task["task"]["model"]),
            int(task["task"]["formal_order"]),
            int(task["task"]["system_size"]),
        ): sum(float(row["wall_seconds"]) for row in task["observations"])
        for task in reduced["tasks"]
        if str(task["task"]["kind"]) == "primary"
    }


def _lpt_makespan(durations: Sequence[float], workers: int) -> float:
    loads = [0.0] * workers
    for duration in sorted(durations, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        loads[index] += duration
    return max(loads, default=0.0)


def analyze_n9_checkpoint(
    reduced: Mapping[str, Any],
    assembled: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble fail-closed N=9 fits, partial stability, and N=10 cost evidence."""
    reviewed_size_max = int(reduced["reviewed_size_max"])
    configuration = reduced["configuration"]
    accepted = {}
    for row in assembled["accepted_windows"]:
        accepted[(str(row["model"]), int(row["formal_order"]), int(row["system_size"]))] = row
    precision_lookup = {}
    for task in reduced["tasks"]:
        if str(task["task"]["kind"]) != "primary":
            continue
        prefix = (
            str(task["task"]["model"]),
            int(task["task"]["formal_order"]),
            int(task["task"]["system_size"]),
        )
        for row in task["observations"]:
            value = row.get("relative_precision_change")
            precision_lookup[(*prefix, int(row["segments"]))] = (
                float(value) if value is not None else 0.0
            )
    timings = _primary_task_timings(reduced)
    rows_payload = []
    for model in configuration["models"]:
        for formal_order in configuration["formal_orders"]:
            pair = (str(model), int(formal_order))
            size_rows = [accepted.get((*pair, size)) for size in range(4, 10)]
            numerical_passed = all(row is not None for row in size_rows)
            time_gate = _time_gate_status(
                configuration, assembled, pair[0], pair[1]
            )
            payload: dict[str, Any] = {
                "model": pair[0],
                "formal_order": pair[1],
                "branch_count": pair[1] // 2,
                "numerical_N4_through_N9_passed": numerical_passed,
                "time_law_gate": time_gate,
                "all_current_numerical_and_time_law_gates_passed": (
                    numerical_passed and time_gate["status"] == "passed"
                ),
            }
            if not numerical_passed:
                payload["missing_or_rejected_sizes"] = [
                    size for size, row in zip(range(4, 10), size_rows, strict=True) if row is None
                ]
                rows_payload.append(payload)
                continue
            complete_rows = [row for row in size_rows if row is not None]
            sizes = tuple(range(4, 10))
            coefficients = tuple(float(row["coefficient_b_2j"]) for row in complete_rows)
            weights, uncertainty = _row_weights(
                complete_rows, time_gate, precision_lookup
            )
            payload["coefficients_b_2j"] = {
                str(size): coefficient
                for size, coefficient in zip(sizes, coefficients, strict=True)
            }
            payload["relative_uncertainty_components"] = {
                str(size): components
                for size, components in zip(sizes, uncertainty, strict=True)
            }
            fits = []
            viable_models = []
            for candidate in CHECKPOINT_MODELS:
                stages = []
                stage_fits = []
                for last_size in (7, 8, 9):
                    length = last_size - 3
                    fit = fit_size_law(
                        sizes[:length],
                        coefficients[:length],
                        candidate,
                        weights=weights[:length],
                        reviewed_size_max=reviewed_size_max,
                    )
                    stage_fits.append(fit)
                    stages.append(_fit_payload(fit))
                training = stage_fits[1]
                n9_holdout = abs(training.at(9) - coefficients[-1]) / coefficients[-1]
                stability = available_shifted_window_stability(
                    sizes,
                    coefficients,
                    candidate,
                    weights=weights,
                    reviewed_size_max=reviewed_size_max,
                )
                viable = (
                    training.converged
                    # With one of the two holdouts available, only the 20%
                    # maximum-error gate can be evaluated.  The 10% median
                    # gate remains unresolved until N=10 exists.
                    and n9_holdout <= 0.20
                    and stability["necessary_conditions_pass"]
                )
                if viable:
                    viable_models.append(candidate)
                fits.append(
                    {
                        "model": candidate,
                        "stages": stages,
                        "n9_holdout_relative_error_from_N4_through_N8": n9_holdout,
                        "updates": [
                            _fit_update(stage_fits[0], stage_fits[1], 8),
                            _fit_update(stage_fits[1], stage_fits[2], 9),
                        ],
                        "available_shifted_window_stability": stability,
                        "provisionally_viable_but_not_promotable": viable,
                    }
                )
            if time_gate["status"] == "pending-required-N8-expansion":
                n10_need = "defer-until-required-N8-time-law-check"
            elif viable_models:
                n10_need = (
                    "needed-for-model-ambiguity-and-full-stability"
                    if len(viable_models) > 1
                    else "needed-for-second-holdout-and-full-stability"
                )
            else:
                n10_need = "not-informative-for-current-candidate-family"
            n8_wall = timings[(*pair, 8)]
            n9_wall = timings[(*pair, 9)]
            scaling = n9_wall / n8_wall
            payload["fits"] = fits
            payload["provisionally_viable_models"] = viable_models
            payload["model_ambiguity_with_current_data"] = len(viable_models) > 1
            payload["n10_scientific_need"] = n10_need
            payload["wall_time"] = {
                "N8_seconds": n8_wall,
                "N9_seconds": n9_wall,
                "N9_over_N8_factor": scaling,
                "projected_N10_seconds_repeating_last_factor": n9_wall * scaling,
            }
            rows_payload.append(payload)
    numerical_rows = [
        row for row in rows_payload if row["numerical_N4_through_N9_passed"]
    ]
    gate_complete = [
        row
        for row in numerical_rows
        if row["all_current_numerical_and_time_law_gates_passed"]
    ]
    informative = [
        row
        for row in gate_complete
        if str(row["n10_scientific_need"]).startswith("needed-")
    ]
    projected_all = [
        float(row["wall_time"]["projected_N10_seconds_repeating_last_factor"])
        for row in numerical_rows
    ]
    projected_informative = [
        float(row["wall_time"]["projected_N10_seconds_repeating_last_factor"])
        for row in informative
    ]
    n9_by_model = {}
    for row in numerical_rows:
        n9_by_model.setdefault(row["model"], []).append(
            float(row["wall_time"]["N9_seconds"])
        )
    payload = {
        "schema_version": "n9-checkpoint-1.0",
        "study_id": reduced["study_id"],
        "source_reduced_digest": reduced["reduced_digest"],
        "source_assembled_digest": assembled["assembled_digest"],
        "reviewed_size_max": reviewed_size_max,
        "acceptance_criteria_unchanged": True,
        "N10_tasks_launched": False,
        "full_finite_size_gate_evaluated": False,
        "full_finite_size_gate_reason": (
            "The required shifted window N=6..10 and second holdout N=10 are absent."
        ),
        "rows": rows_payload,
        "summary": {
            "gate_complete_rows": [
                {"model": row["model"], "formal_order": row["formal_order"]}
                for row in gate_complete
            ],
            "numerical_rows_pending_time_law_expansion": [
                {"model": row["model"], "formal_order": row["formal_order"]}
                for row in numerical_rows
                if row["time_law_gate"]["status"]
                == "pending-required-N8-expansion"
            ],
            "measured_N9_two_batch_makespan_seconds": sum(
                max(values) for values in n9_by_model.values()
            ),
            "projected_N10_all_numerical_rows_task_hours": sum(projected_all) / 3600,
            "projected_N10_all_numerical_rows_three_worker_hours": (
                _lpt_makespan(projected_all, 3) / 3600
            ),
            "projected_N10_informative_gate_complete_rows_task_hours": (
                sum(projected_informative) / 3600
            ),
            "projected_N10_informative_gate_complete_rows_three_worker_hours": (
                _lpt_makespan(projected_informative, 3) / 3600
            ),
        },
    }
    payload["checkpoint_digest"] = canonical_json_digest(payload)
    return payload
