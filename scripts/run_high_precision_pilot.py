"""Run the deterministic TFIM high-order calibration pilot.

The output is a reduced observation artifact.  No matrices or per-task working
shards are serialized by this pilot runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mpmath as mp

from hamiltonian_resources import mpf_richardson_diagnostics, transverse_field_ising
from hamiltonian_resources.calibration_high_precision import (
    adaptive_mpf_operator_norm_error,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _coefficient_diagnostics(
    error_decimal: str,
    *,
    time: float,
    segments: int,
    branch_count: int,
    schedule: str,
    digits: int,
) -> tuple[str, str, str]:
    with mp.workdps(digits):
        error = mp.mpf(error_decimal)
        formal_order = 2 * branch_count
        coefficient = error * segments**formal_order / mp.mpf(str(time)) ** (
            formal_order + 1
        )
        richardson = mpf_richardson_diagnostics(branch_count, schedule=schedule)
        sigma = mp.mpf(abs(richardson.leading_omitted_moment.numerator)) / (
            richardson.leading_omitted_moment.denominator
        )
        normalized = coefficient / sigma
        gamma = normalized ** (mp.mpf(1) / (formal_order + 1))
        return (
            mp.nstr(coefficient, n=digits),
            mp.nstr(normalized, n=digits),
            mp.nstr(gamma, n=digits),
        )


def run_pilot(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"] != "transverse_field_ising":
        raise ValueError("the staged pilot is restricted to transverse_field_ising")
    backend = str(config["backend"])
    schedule = str(config.get("schedule", "new"))
    observations: list[dict[str, Any]] = []
    for size in config["sizes"]:
        hamiltonian = transverse_field_ising(
            int(size),
            coupling=float(config["model_parameters"]["coupling"]),
            field=float(config["model_parameters"]["field"]),
            periodic=bool(config["model_parameters"]["periodic"]),
        )
        time = float(size)
        for branch_count_text, ratios in config["segment_ratios"].items():
            branch_count = int(branch_count_text)
            for ratio in ratios:
                segments = max(1, round(float(ratio) * time))
                estimate = adaptive_mpf_operator_norm_error(
                    hamiltonian,
                    time,
                    segments,
                    branch_count,
                    backend=backend,
                    schedule=schedule,
                    digit_increment=int(config["digit_increment"]),
                    max_digits=int(config["max_digits"]),
                    relative_tolerance=float(config["relative_tolerance"]),
                )
                coefficient, normalized, gamma = _coefficient_diagnostics(
                    estimate.value_decimal,
                    time=time,
                    segments=segments,
                    branch_count=branch_count,
                    schedule=schedule,
                    digits=estimate.decimal_digits,
                )
                observations.append(
                    {
                        "model": config["model"],
                        "system_size": int(size),
                        "time": str(time),
                        "branch_count": branch_count,
                        "formal_order": 2 * branch_count,
                        "segments": segments,
                        "segment_ratio": float(ratio),
                        "error": estimate.value_decimal,
                        "coefficient_b_2j": coefficient,
                        "normalized_c_2j": normalized,
                        "gamma_2j": gamma,
                        "backend": estimate.backend,
                        "backend_version": estimate.backend_version,
                        "decimal_digits": estimate.decimal_digits,
                        "attempted_digits": list(estimate.attempted_digits),
                        "precision_converged": estimate.converged,
                        "relative_precision_change": (
                            estimate.relative_precision_change
                        ),
                        "interval_relative_width": (
                            estimate.interval_relative_width
                        ),
                        "interval_certified": estimate.interval_certified,
                        "schedule_digest": estimate.schedule_digest,
                        "term_order_digest": estimate.term_order_digest,
                        "wall_seconds": estimate.wall_seconds,
                    }
                )
    reference_checks: list[dict[str, Any]] = []
    for check in config.get("mpmath_reference_checks", []):
        size = int(check["system_size"])
        branch_count = int(check["branch_count"])
        segments = int(check["segments"])
        hamiltonian = transverse_field_ising(
            size,
            coupling=float(config["model_parameters"]["coupling"]),
            field=float(config["model_parameters"]["field"]),
            periodic=bool(config["model_parameters"]["periodic"]),
        )
        estimates = {
            comparison_backend: adaptive_mpf_operator_norm_error(
                hamiltonian,
                float(size),
                segments,
                branch_count,
                backend=comparison_backend,
                schedule=schedule,
                digit_increment=int(config["digit_increment"]),
                max_digits=int(config["max_digits"]),
                relative_tolerance=float(config["relative_tolerance"]),
            )
            for comparison_backend in ("mpmath", "flint")
        }
        with mp.workdps(max(value.decimal_digits for value in estimates.values())):
            reference = mp.mpf(estimates["mpmath"].value_decimal)
            accelerated = mp.mpf(estimates["flint"].value_decimal)
            relative_difference = abs(accelerated - reference) / abs(reference)
        reference_checks.append(
            {
                "system_size": size,
                "branch_count": branch_count,
                "formal_order": 2 * branch_count,
                "segments": segments,
                "mpmath_error": estimates["mpmath"].value_decimal,
                "flint_error": estimates["flint"].value_decimal,
                "relative_difference": mp.nstr(relative_difference, n=20),
                "mpmath_wall_seconds": estimates["mpmath"].wall_seconds,
                "flint_wall_seconds": estimates["flint"].wall_seconds,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "pilot-1.0",
        "study_id": config["study_id"],
        "configuration_digest": hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        "reviewed_size_max": int(config["reviewed_size_max"]),
        "configuration": config,
        "observations": observations,
        "reference_checks": reference_checks,
    }
    payload["observations_digest"] = hashlib.sha256(
        _canonical_bytes(observations)
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_pilot(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
