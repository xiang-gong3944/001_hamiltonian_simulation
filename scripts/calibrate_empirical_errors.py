#!/usr/bin/env python3
"""Generate a reviewable operator-norm calibration artifact bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hamiltonian_resources.calibration_study import (
    dense_operator_norm_error,
    fit_affine_size_coefficient,
    observed_error_coefficient,
    select_asymptotic_pair,
    sparse_operator_norm_error,
)
from hamiltonian_resources.hamiltonians import (
    heisenberg_chain,
    transverse_field_ising,
)


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(","))
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(","))
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tfim", "heisenberg"), required=True)
    parser.add_argument("--algorithm", choices=("trotter", "multiproduct"), required=True)
    parser.add_argument("--formal-order", type=int, required=True)
    parser.add_argument("--sizes", type=_parse_ints, default=tuple(range(4, 13)))
    parser.add_argument(
        "--segment-multipliers",
        type=_parse_floats,
        default=(1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0),
        help="candidate r/N values, evaluated in ascending order",
    )
    parser.add_argument("--dense-through", type=int, default=8)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--power-tolerance", type=float, default=1e-8)
    parser.add_argument("--power-iterations", type=int, default=80)
    parser.add_argument("--power-restarts", type=int, default=3)
    parser.add_argument("--power-seed", type=int, default=2026)
    parser.add_argument("--relative-order-tolerance", type=float, default=0.05)
    parser.add_argument("--floating-point-floor", type=float, default=1e-11)
    parser.add_argument(
        "--time-factors",
        type=_parse_floats,
        default=(0.75, 1.0, 1.25),
        help="representative fixed-r time-law checks relative to T=N",
    )
    parser.add_argument("--reviewed-by")
    parser.add_argument("--review-note", default="")
    return parser


def _model(arguments):
    if arguments.model == "tfim":
        return (
            lambda size: transverse_field_ising(
                size,
                coupling=1.0,
                field=3.0,
                periodic=False,
            ),
            {"coupling": 1.0, "field": 3.0, "periodic": False},
        )
    return (
        lambda size: heisenberg_chain(size, coupling=1.0, field_z=0.3),
        {"coupling": 1.0, "field_z": 0.3},
    )


def _estimate(arguments, hamiltonian, time: float, segments: int):
    common = {
        "algorithm": arguments.algorithm,
        "formal_order": arguments.formal_order,
    }
    if hamiltonian.num_qubits <= arguments.dense_through:
        return dense_operator_norm_error(hamiltonian, time, segments, **common)
    return sparse_operator_norm_error(
        hamiltonian,
        time,
        segments,
        tolerance=arguments.power_tolerance,
        max_iterations=arguments.power_iterations,
        restarts=arguments.power_restarts,
        seed=arguments.power_seed,
        **common,
    )


def _json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    arguments = _parser().parse_args()
    if tuple(sorted(set(arguments.sizes))) != arguments.sizes:
        raise ValueError("sizes must be strictly increasing")
    multipliers = tuple(sorted(set(arguments.segment_multipliers)))
    if len(multipliers) < 2:
        raise ValueError("at least two segment multipliers are required")
    if len(set(arguments.time_factors)) < 2:
        raise ValueError("at least two time factors are required")
    factory, model_parameters = _model(arguments)

    rows: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    for size in arguments.sizes:
        hamiltonian = factory(size)
        time = float(size)
        segments = tuple(sorted({max(1, math.ceil(value * size)) for value in multipliers}))
        observations: list[tuple[int, float]] = []
        by_segment: dict[int, dict[str, object]] = {}
        for segment_count in segments:
            estimate = _estimate(arguments, hamiltonian, time, segment_count)
            coefficient = observed_error_coefficient(
                estimate.value,
                segment_count,
                time,
                arguments.formal_order,
            )
            row = {
                "model": arguments.model,
                "model_parameters_json": json.dumps(
                    model_parameters,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "algorithm": arguments.algorithm,
                "formal_order": arguments.formal_order,
                "N": size,
                "time": time,
                "segments": segment_count,
                "step_size": time / segment_count,
                "operator_norm_error": estimate.value,
                "observed_coefficient": coefficient,
                "backend": estimate.backend,
                "converged": estimate.converged,
                "iterations": estimate.iterations,
                "restarts": estimate.restarts,
                "relative_residual": estimate.relative_residual,
            }
            rows.append(row)
            by_segment[segment_count] = row
            observations.append((segment_count, estimate.value))
            print(
                f"{arguments.model} {arguments.algorithm} q={arguments.formal_order} "
                f"N={size} r={segment_count}: error={estimate.value:.8e}, "
                f"B={coefficient:.8e} ({estimate.backend})",
                flush=True,
            )
        pair = select_asymptotic_pair(
            tuple(observations),
            arguments.formal_order,
            relative_order_tolerance=arguments.relative_order_tolerance,
            floating_point_floor=arguments.floating_point_floor,
        )
        coefficients = (
            float(by_segment[pair.first_segments]["observed_coefficient"]),
            float(by_segment[pair.second_segments]["observed_coefficient"]),
        )
        accepted.append(
            {
                "N": size,
                "time": time,
                "first_segments": pair.first_segments,
                "second_segments": pair.second_segments,
                "running_exponent": pair.running_exponent,
                "coefficient": float(np.mean(coefficients)),
                "max_step_size": time / pair.first_segments,
            }
        )

    fit = fit_affine_size_coefficient(
        tuple(int(row["N"]) for row in accepted),
        tuple(float(row["coefficient"]) for row in accepted),
    )
    representative = accepted[len(accepted) // 2]
    representative_size = int(representative["N"])
    representative_segments = int(representative["second_segments"])
    representative_hamiltonian = factory(representative_size)
    time_checks = []
    for factor in tuple(sorted(set(arguments.time_factors))):
        check_time = factor * representative_size
        estimate = _estimate(
            arguments,
            representative_hamiltonian,
            check_time,
            representative_segments,
        )
        time_checks.append(
            {
                "N": representative_size,
                "time": check_time,
                "segments": representative_segments,
                "operator_norm_error": estimate.value,
                "backend": estimate.backend,
            }
        )
    time_exponents = []
    for first, second in zip(time_checks, time_checks[1:]):
        time_exponents.append(
            math.log(
                float(second["operator_norm_error"])
                / float(first["operator_norm_error"])
            )
            / math.log(float(second["time"]) / float(first["time"]))
        )
    fit_payload = {
        "coefficient_model": "affine-aN-plus-b",
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_squared": fit.r_squared,
        "root_mean_square_error": fit.root_mean_square_error,
        "slope_standard_error": fit.slope_standard_error,
        "intercept_standard_error": fit.intercept_standard_error,
        "residuals": list(fit.residuals),
        "maximum_accepted_step_size": min(
            float(row["max_step_size"]) for row in accepted
        ),
        "time_exponent_min": min(time_exponents),
        "time_exponent_max": max(time_exponents),
    }
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": generated,
        "model": arguments.model,
        "model_parameters": model_parameters,
        "geometry": "1d-chain",
        "boundary_condition": "open",
        "algorithm": arguments.algorithm,
        "formal_order": arguments.formal_order,
        "partition": "auto" if arguments.algorithm == "trotter" else None,
        "schedule": "new" if arguments.algorithm == "multiproduct" else None,
        "formula": (
            "repository-suzuki-v1"
            if arguments.algorithm == "trotter"
            else "ordered-individual-pauli-strang-mpf-v1"
        ),
        "sizes": list(arguments.sizes),
        "time_rule": "T=N",
        "representative_time_factors": list(arguments.time_factors),
        "segment_multipliers": list(multipliers),
        "error_metric": "spectral operator 2-norm",
        "dense_backend": "explicit matrix and SVD",
        "sparse_backend": "expm_multiply plus deterministic D-dagger-D power iteration",
        "power_iteration": {
            "tolerance": arguments.power_tolerance,
            "max_iterations": arguments.power_iterations,
            "restarts": arguments.power_restarts,
            "seed": arguments.power_seed,
        },
        "acceptance": {
            "relative_order_tolerance": arguments.relative_order_tolerance,
            "floating_point_floor": arguments.floating_point_floor,
            "consecutive_points_required": True,
            "random_state_inputs_allowed": False,
        },
    }
    review = {
        "status": "reviewed" if arguments.reviewed_by else "candidate",
        "reviewed_by": arguments.reviewed_by,
        "reviewed_at_utc": generated if arguments.reviewed_by else None,
        "note": arguments.review_note,
        "manifest_digest": _json_digest(manifest),
        "accepted_observations_digest": _json_digest(accepted),
        "time_checks_digest": _json_digest(time_checks),
        "fit_digest": _json_digest(fit_payload),
    }

    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=False)
    with (output / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "accepted_observations.json").open("w", encoding="utf-8") as handle:
        json.dump(accepted, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output / "time_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(time_checks, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for name, payload in (
        ("manifest.json", manifest),
        ("fit.json", fit_payload),
        ("review.json", review),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
