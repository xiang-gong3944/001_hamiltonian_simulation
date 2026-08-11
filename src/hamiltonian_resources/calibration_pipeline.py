"""Resumable high-precision calibration tasks and deterministic reduction."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import mpmath as mp

from .calibration_high_precision import adaptive_mpf_operator_norm_error
from .calibration_study import (
    select_asymptotic_window,
    select_size_law_model,
    validate_time_law,
)
from .empirical import canonical_json_digest
from .hamiltonians import PauliHamiltonian, heisenberg_chain, transverse_field_ising
from .multiproduct import mpf_richardson_diagnostics


@dataclass(frozen=True)
class CalibrationPoint:
    time: float
    segments: int


@dataclass(frozen=True)
class CalibrationTask:
    kind: str
    model: str
    formal_order: int
    system_size: int
    points: tuple[CalibrationPoint, ...]

    @property
    def branch_count(self) -> int:
        return self.formal_order // 2

    @property
    def task_id(self) -> str:
        return canonical_json_digest(asdict(self))[:24]


def load_calibration_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("calibration configuration must be an object")
    required = {
        "study_id",
        "models",
        "model_parameters",
        "formal_orders",
        "sizes",
        "segment_ratios",
        "reviewed_size_max",
        "downstream_benchmark",
        "backend",
        "formula",
        "schedule",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"calibration configuration is missing {sorted(missing)}")
    downstream = raw["downstream_benchmark"]
    if not isinstance(downstream, Mapping):
        raise TypeError("downstream_benchmark must be an object")
    benchmark_sizes = tuple(int(value) for value in downstream["system_sizes"])  # type: ignore[arg-type]
    if not benchmark_sizes or min(benchmark_sizes) < 1:
        raise ValueError("downstream benchmark sizes must be positive")
    reviewed_size_max = max(benchmark_sizes)
    if int(raw["reviewed_size_max"]) != reviewed_size_max:
        raise ValueError(
            "reviewed_size_max must equal the downstream benchmark maximum"
        )
    raw["reviewed_size_max"] = reviewed_size_max
    if reviewed_size_max < max(int(value) for value in raw["sizes"]):
        raise ValueError("reviewed_size_max is below the observed size matrix")
    expected_parameters = {
        "transverse_field_ising": {
            "coupling": 1.0,
            "field": 3.0,
            "periodic": False,
        },
        "heisenberg_chain": {"coupling": 1.0, "field_z": 0.3},
    }
    if raw["model_parameters"] != {
        model: expected_parameters[str(model)] for model in raw["models"]
    }:
        raise ValueError("model_parameters must match the registered calibration models")
    if raw["formula"] != "ordered-individual-pauli-strang-mpf-v1":
        raise ValueError("unsupported calibration formula")
    if raw["schedule"] != "new":
        raise ValueError("high-order calibration requires the registered new schedule")
    return raw


def expand_calibration_tasks(config: Mapping[str, Any]) -> tuple[CalibrationTask, ...]:
    tasks: list[CalibrationTask] = []
    ratios_by_order = config["segment_ratios"]
    if not isinstance(ratios_by_order, Mapping):
        raise TypeError("segment_ratios must be an object keyed by formal order")
    for model in config["models"]:
        for formal_order_raw in config["formal_orders"]:
            formal_order = int(formal_order_raw)
            if formal_order % 2 or not 4 <= formal_order <= 30:
                raise ValueError("MPF formal orders must be even and lie in [4, 30]")
            for size_raw in config["sizes"]:
                size = int(size_raw)
                ratios = (
                    config.get("segment_ratio_overrides", {})
                    .get(str(model), {})
                    .get(str(formal_order), {})
                    .get(str(size), ratios_by_order[str(formal_order)])
                )
                points = tuple(
                    CalibrationPoint(float(size), max(1, round(float(ratio) * size)))
                    for ratio in ratios
                )
                tasks.append(
                    CalibrationTask("primary", str(model), formal_order, size, points)
                )
    for raw in config.get("time_law_checks", []):
        size = int(raw["system_size"])
        fixed_segments = int(raw["segments"])
        points = tuple(
            CalibrationPoint(float(factor) * size, fixed_segments)
            for factor in raw.get("time_factors", (0.8, 1.0, 1.2))
        )
        tasks.append(
            CalibrationTask(
                "time-law",
                str(raw["model"]),
                int(raw["formal_order"]),
                size,
                points,
            )
        )
    identifiers = [task.task_id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("calibration configuration expands to duplicate tasks")
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def _hamiltonian(model: str, size: int) -> PauliHamiltonian:
    if model == "transverse_field_ising":
        return transverse_field_ising(size, coupling=1.0, field=3.0, periodic=False)
    if model == "heisenberg_chain":
        return heisenberg_chain(size, coupling=1.0, field_z=0.3)
    raise ValueError(f"unsupported calibration model: {model!r}")


def _coefficient_strings(
    error_decimal: str,
    *,
    time: float,
    segments: int,
    formal_order: int,
    digits: int,
) -> tuple[str, str, str]:
    diagnostics = mpf_richardson_diagnostics(formal_order // 2)
    with mp.workdps(digits):
        error = mp.mpf(error_decimal)
        coefficient = error * segments**formal_order / mp.mpf(str(time)) ** (
            formal_order + 1
        )
        sigma = mp.mpf(abs(diagnostics.leading_omitted_moment.numerator)) / (
            diagnostics.leading_omitted_moment.denominator
        )
        normalized = coefficient / sigma
        gamma = normalized ** (mp.mpf(1) / (formal_order + 1))
        return tuple(
            mp.nstr(value, n=digits)
            for value in (coefficient, normalized, gamma)
        )  # type: ignore[return-value]


def task_execution_digest(
    config: Mapping[str, Any],
    task: CalibrationTask,
) -> str:
    """Hash only settings that can change the numerical result of one task."""
    return canonical_json_digest(
        {
            "task": asdict(task),
            "backend": config["backend"],
            "digit_increment": int(config.get("digit_increment", 32)),
            "max_digits": int(config.get("max_digits", 512)),
            "relative_tolerance": float(config.get("relative_tolerance", 1e-8)),
            "schedule": str(config.get("schedule", "new")),
            "symmetry_reduction": str(config.get("symmetry_reduction", "none")),
        }
    )


def run_calibration_task(
    config: Mapping[str, Any],
    task: CalibrationTask,
) -> dict[str, Any]:
    hamiltonian = _hamiltonian(task.model, task.system_size)
    observations: list[dict[str, Any]] = []
    for point in task.points:
        estimate = adaptive_mpf_operator_norm_error(
            hamiltonian,
            point.time,
            point.segments,
            task.branch_count,
            backend=str(config["backend"]),  # type: ignore[arg-type]
            symmetry_reduction=str(  # type: ignore[arg-type]
                config.get("symmetry_reduction", "none")
            ),
            digit_increment=int(config.get("digit_increment", 32)),
            max_digits=int(config.get("max_digits", 512)),
            relative_tolerance=float(config.get("relative_tolerance", 1e-8)),
        )
        coefficient, normalized, gamma = _coefficient_strings(
            estimate.value_decimal,
            time=point.time,
            segments=point.segments,
            formal_order=task.formal_order,
            digits=estimate.decimal_digits,
        )
        observations.append(
            {
                "time": str(point.time),
                "segments": point.segments,
                "error": estimate.value_decimal,
                "coefficient_b_2j": coefficient,
                "normalized_c_2j": normalized,
                "gamma_2j": gamma,
                "status": "converged" if estimate.converged else "precision-limited",
                "precision_converged": estimate.converged,
                "decimal_digits": estimate.decimal_digits,
                "attempted_digits": list(estimate.attempted_digits),
                "relative_precision_change": estimate.relative_precision_change,
                "backend": estimate.backend,
                "backend_version": estimate.backend_version,
                "interval_certified": estimate.interval_certified,
                "interval_relative_width": estimate.interval_relative_width,
                "schedule_digest": estimate.schedule_digest,
                "term_order_digest": estimate.term_order_digest,
                "wall_seconds": estimate.wall_seconds,
            }
        )
    payload = {
        "schema_version": "task-1.0",
        "study_id": config["study_id"],
        "configuration_digest": canonical_json_digest(config),
        "task_execution_digest": task_execution_digest(config, task),
        "task": asdict(task),
        "task_id": task.task_id,
        "observations": observations,
    }
    payload["scientific_digest"] = canonical_json_digest(
        {
            "schema_version": payload["schema_version"],
            "task": payload["task"],
            "task_id": payload["task_id"],
            "task_execution_digest": payload["task_execution_digest"],
            "observations": [
                {key: value for key, value in row.items() if key != "wall_seconds"}
                for row in observations
            ],
        }
    )
    return payload


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run_and_write_task(
    config: Mapping[str, Any],
    task: CalibrationTask,
    shard_directory: Path,
) -> Path:
    destination = shard_directory / f"{task.task_id}.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        execution_matches = existing.get("task_execution_digest") == (
            task_execution_digest(config, task)
        )
        legacy_matches = (
            existing.get("task_execution_digest") is None
            and canonical_json_digest(existing.get("task"))
            == canonical_json_digest(asdict(task))
            and all(
                row.get("backend") == config["backend"]
                for row in existing.get("observations", [])
            )
        )
        if existing.get("task_id") == task.task_id and (
            execution_matches or legacy_matches
        ):
            return destination
    _atomic_json_write(destination, run_calibration_task(config, task))
    return destination


def run_calibration_matrix(
    config: Mapping[str, Any],
    shard_directory: Path,
    *,
    workers: int = 1,
) -> tuple[Path, ...]:
    tasks = expand_calibration_tasks(config)
    if workers <= 1:
        return tuple(run_and_write_task(config, task, shard_directory) for task in tasks)
    completed: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_and_write_task, config, task, shard_directory): task
            for task in tasks
        }
        for future in as_completed(futures):
            completed.append(future.result())
    return tuple(sorted(completed))


def reduce_calibration_shards(
    config: Mapping[str, Any],
    shard_paths: Iterable[Path],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    expected = {task.task_id: task for task in expand_calibration_tasks(config)}
    configuration_digest = canonical_json_digest(config)
    shards: dict[str, dict[str, Any]] = {}
    ignored_task_ids: list[str] = []
    for path in shard_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(raw["task_id"])
        if task_id not in expected:
            ignored_task_ids.append(task_id)
            continue
        execution_digest = raw.get("task_execution_digest")
        if execution_digest is not None and execution_digest != task_execution_digest(
            config,
            expected[task_id],
        ):
            raise ValueError(f"shard {path} has a different task execution digest")
        if task_id in shards:
            raise ValueError(f"duplicate shard for task {task_id}")
        shards[task_id] = raw
    missing = sorted(expected.keys() - shards.keys())
    if require_complete and missing:
        raise ValueError(f"missing {len(missing)} configured calibration shards")
    reduced_tasks = []
    for task_id in sorted(shards):
        raw = shards[task_id]
        reduced_tasks.append(
            {
                "task_id": task_id,
                "task": raw["task"],
                "scientific_digest": raw["scientific_digest"],
                "observations": sorted(
                    raw["observations"],
                    key=lambda row: (float(row["time"]), int(row["segments"])),
                ),
            }
        )
    payload = {
        "schema_version": "reduced-1.0",
        "study_id": config["study_id"],
        "configuration": dict(config),
        "configuration_digest": configuration_digest,
        "reviewed_size_max": int(config["reviewed_size_max"]),
        "expected_task_ids": sorted(expected),
        "missing_task_ids": missing,
        "ignored_task_ids": sorted(ignored_task_ids),
        "tasks": reduced_tasks,
    }
    payload["reduced_digest"] = canonical_json_digest(payload)
    return payload


def _fit_payload(selection: Any) -> dict[str, Any]:
    return {
        "selected_model": selection.selected.model if selection.selected else None,
        "selected_parameters": (
            dict(selection.selected.parameters) if selection.selected else None
        ),
        "failure_reasons": list(selection.failure_reasons),
        "candidates": [
            {
                "model": fit.model,
                "parameters": dict(fit.parameters),
                "aicc": fit.aicc,
                "converged": fit.converged,
                "holdout_errors": list(dict(selection.holdout_errors)[fit.model]),
                "stability": {
                    "accepted": stability.accepted,
                    "prediction_spreads": dict(stability.prediction_spreads),
                    "parameter_stable": stability.parameter_stable,
                    "residual_drift_free": stability.residual_drift_free,
                    "failure_reasons": list(stability.failure_reasons),
                },
            }
            for fit, stability in zip(
                selection.candidates,
                selection.stability,
                strict=True,
            )
        ],
    }


def assemble_calibration_artifacts(reduced: Mapping[str, Any]) -> dict[str, Any]:
    accepted_windows: list[dict[str, Any]] = []
    time_checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_result in reduced["tasks"]:
        task = task_result["task"]
        observations = task_result["observations"]
        try:
            if task["kind"] == "primary":
                time = float(observations[0]["time"])
                window = select_asymptotic_window(
                    tuple(
                        (
                            int(row["segments"]),
                            float(row["error"]),
                            bool(row["precision_converged"]),
                        )
                        for row in observations
                    ),
                    time,
                    int(task["formal_order"]),
                )
                accepted_windows.append(
                    {
                        "model": task["model"],
                        "formal_order": int(task["formal_order"]),
                        "system_size": int(task["system_size"]),
                        "time": time,
                        "segments": [point[0] for point in window.observations],
                        "running_exponents": list(window.running_exponents),
                        "coefficient_b_2j": window.median_coefficient_b_2j,
                        "maximum_relative_deviation": (
                            window.maximum_relative_deviation
                        ),
                        "relative_mad": window.relative_median_absolute_deviation,
                        "status": "accepted-window",
                    }
                )
            else:
                validation = validate_time_law(
                    tuple(
                        (
                            float(row["time"]),
                            int(row["segments"]),
                            float(row["error"]),
                            bool(row["precision_converged"]),
                        )
                        for row in observations
                    ),
                    int(task["formal_order"]),
                )
                time_checks.append(
                    {
                        "model": task["model"],
                        "formal_order": int(task["formal_order"]),
                        "system_size": int(task["system_size"]),
                        "segments": int(observations[0]["segments"]),
                        "fitted_exponent": validation.fitted_exponent,
                        "maximum_relative_coefficient_deviation": (
                            validation.maximum_relative_coefficient_deviation
                        ),
                        "accepted": validation.accepted,
                        "status": (
                            "time-law-passed"
                            if validation.accepted
                            else "time-law-failed"
                        ),
                    }
                )
        except ValueError as error:
            precision_limited = any(
                not bool(row["precision_converged"]) for row in observations
            )
            failures.append(
                {
                    "task_id": task_result["task_id"],
                    "model": task["model"],
                    "formal_order": int(task["formal_order"]),
                    "system_size": int(task["system_size"]),
                    "reason": str(error),
                    "status": (
                        "precision-limited"
                        if precision_limited
                        else "truncation-dominated"
                    ),
                }
            )
    fits: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in accepted_windows:
        grouped.setdefault((row["model"], row["formal_order"]), []).append(row)
    configured_pairs = tuple(
        (str(model), int(formal_order))
        for model in reduced["configuration"]["models"]
        for formal_order in reduced["configuration"]["formal_orders"]
    )
    for model, formal_order in configured_pairs:
        rows = grouped.get((model, formal_order), [])
        rows.sort(key=lambda row: row["system_size"])
        sizes = tuple(int(row["system_size"]) for row in rows)
        if all(size in sizes for size in range(4, 11)):
            selection = select_size_law_model(
                sizes,
                tuple(float(row["coefficient_b_2j"]) for row in rows),
                reviewed_size_max=int(reduced["reviewed_size_max"]),
            )
            fit = {
                "model": model,
                "formal_order": formal_order,
                **_fit_payload(selection),
            }
            fit["status"] = (
                "size-fit-passed"
                if selection.selected is not None
                else (
                    "finite-size-unstable"
                    if any(
                        not stability.accepted for stability in selection.stability
                    )
                    else "size-fit-failed"
                )
            )
            fits.append(fit)
        else:
            fits.append(
                {
                    "model": model,
                    "formal_order": formal_order,
                    "selected_model": None,
                    "failure_reasons": ["N=4..10 observations are incomplete"],
                    "status": "insufficient-window",
                }
            )
    promotion: list[dict[str, Any]] = []
    failures_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for failure in failures:
        failures_by_pair.setdefault(
            (str(failure["model"]), int(failure["formal_order"])), []
        ).append(failure)
    time_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for check in time_checks:
        time_by_pair.setdefault(
            (str(check["model"]), int(check["formal_order"])), []
        ).append(check)
    for fit in fits:
        pair = (str(fit["model"]), int(fit["formal_order"]))
        failed_time = [
            row for row in time_by_pair.get(pair, []) if not bool(row["accepted"])
        ]
        pair_failures = failures_by_pair.get(pair, [])
        if failed_time:
            status = "time-law-failed"
        elif pair_failures:
            statuses = {str(row["status"]) for row in pair_failures}
            status = (
                "precision-limited"
                if "precision-limited" in statuses
                else "truncation-dominated"
            )
        elif fit["status"] != "size-fit-passed":
            status = str(fit["status"])
        else:
            status = "candidate-awaiting-human-review"
        promotion.append(
            {
                "model": pair[0],
                "formal_order": pair[1],
                "status": status,
                "automatic_promotion": False,
            }
        )
    payload = {
        "schema_version": "assembled-1.0",
        "study_id": reduced["study_id"],
        "configuration_digest": reduced["configuration_digest"],
        "reviewed_size_max": int(reduced["reviewed_size_max"]),
        "accepted_windows": accepted_windows,
        "time_law_checks": time_checks,
        "size_fits": fits,
        "promotion": promotion,
        "failures": failures,
    }
    payload["assembled_digest"] = canonical_json_digest(payload)
    return payload


def assemble_reproducibility_manifest(
    reduced: Mapping[str, Any],
    assembled: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize deterministic task hashes and numerical provenance."""
    completed_hashes = {
        str(row["task_id"]): str(row["scientific_digest"])
        for row in reduced["tasks"]
    }
    precision_rows = [
        observation
        for task in reduced["tasks"]
        for observation in task["observations"]
    ]
    status_counts: dict[str, int] = {}
    for row in precision_rows:
        status = str(row.get("status", "converged"))
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "schema_version": "provenance-1.0",
        "study_id": reduced["study_id"],
        "configuration_digest": reduced["configuration_digest"],
        "reduced_digest": reduced["reduced_digest"],
        "assembled_digest": assembled["assembled_digest"],
        "reviewed_size_max": int(reduced["reviewed_size_max"]),
        "environment_lock": reduced["configuration"].get("environment_lock", {}),
        "numerical_kernel_provenance": reduced["configuration"].get(
            "numerical_kernel_provenance", []
        ),
        "expected_task_ids": list(reduced["expected_task_ids"]),
        "completed_task_hashes": completed_hashes,
        "missing_task_ids": list(reduced["missing_task_ids"]),
        "precision_status_counts": dict(sorted(status_counts.items())),
        "maximum_decimal_digits": max(
            (int(row.get("decimal_digits", 0)) for row in precision_rows),
            default=0,
        ),
        "raw_shards_committed": False,
        "hash_definition": "SHA-256 of canonical parsed JSON",
    }
    payload["provenance_digest"] = canonical_json_digest(payload)
    return payload
