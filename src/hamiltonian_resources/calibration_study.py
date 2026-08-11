"""Operator-norm kernels used by the empirical calibration study.

This module is intentionally separate from runtime resource planning.  It
constructs ideal Trotter and ideal MPF operators; it never substitutes
state-dependent errors or the amplified reference circuit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator, expm_multiply

from .hamiltonians import PauliHamiltonian
from .multiproduct import MPFSchedule, multiproduct_coefficients, optimal_mpf_exponents
from .trotter import TrotterPartition, resolve_trotter_structure, suzuki_group_factors


CalibrationAlgorithm = Literal["trotter", "multiproduct"]
SizeLawModel = Literal[
    "affine",
    "power",
    "power-plus-offset",
    "power-with-inverse-size-correction",
]


@dataclass(frozen=True)
class OperatorNormEstimate:
    value: float
    backend: Literal["dense-svd", "sparse-power"]
    converged: bool
    iterations: int
    restarts: int
    relative_residual: float


@dataclass(frozen=True)
class AsymptoticPair:
    """Two consecutive observations accepted in the formal-order regime."""

    first_segments: int
    second_segments: int
    first_error: float
    second_error: float
    running_exponent: float


@dataclass(frozen=True)
class AffineCalibrationFit:
    """Diagnostics for a least-squares fit to ``B(N)=a*N+b``."""

    slope: float
    intercept: float
    r_squared: float
    root_mean_square_error: float
    slope_standard_error: float
    intercept_standard_error: float
    residuals: tuple[float, ...]


@dataclass(frozen=True)
class AsymptoticWindow:
    """A consecutive precision-converged formal-order coefficient plateau."""

    observations: tuple[tuple[int, float], ...]
    running_exponents: tuple[float, ...]
    coefficients_b_2j: tuple[float, ...]
    median_coefficient_b_2j: float
    maximum_relative_deviation: float
    relative_median_absolute_deviation: float

    @property
    def max_step_size(self) -> float:
        """Return ``T/r`` only when a caller supplied dimensionless segments."""
        return 1.0 / self.observations[0][0]


@dataclass(frozen=True)
class SizeLawFit:
    """One relative-error fit of a candidate law for ``B_2J(N)``."""

    model: SizeLawModel
    parameters: tuple[tuple[str, float], ...]
    sizes: tuple[int, ...]
    coefficients_b_2j: tuple[float, ...]
    fitted_values: tuple[float, ...]
    relative_residuals: tuple[float, ...]
    log_residual_sum_squares: float
    aicc: float
    converged: bool

    def at(self, system_size: int | float) -> float:
        return evaluate_size_law(self.model, self.parameters, float(system_size))


@dataclass(frozen=True)
class FiniteSizeStability:
    """Shifted-window stability diagnostics for one fixed size-law family."""

    model: SizeLawModel
    reviewed_size_max: int
    window_fits: tuple[SizeLawFit, ...]
    prediction_spreads: tuple[tuple[int, float], ...]
    parameter_stable: bool
    residual_drift_free: bool
    accepted: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SizeLawSelection:
    """Candidate fits and the model admitted by predictive/stability gates."""

    selected: SizeLawFit | None
    candidates: tuple[SizeLawFit, ...]
    holdout_errors: tuple[tuple[SizeLawModel, tuple[float, ...]], ...]
    stability: tuple[FiniteSizeStability, ...]
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExternalSizeValidation:
    """Out-of-sample ``N=11,12`` validation before a final refit."""

    prediction_errors: tuple[float, ...]
    same_direction_large_residuals: bool
    prediction_movement_50: float
    prediction_movement_reviewed_max: float
    accepted: bool


@dataclass(frozen=True)
class TimeLawValidation:
    """Fixed-segment validation of the expected ``T^(2J+1)`` law."""

    fitted_exponent: float
    expected_exponent: int
    coefficients_b_2j: tuple[float, ...]
    maximum_relative_coefficient_deviation: float
    accepted: bool


def effective_power(
    first_value: float,
    first_scale: float,
    second_value: float,
    second_scale: float,
) -> float:
    """Return ``beta`` for a local law ``value proportional to scale^-beta``."""
    values = (first_value, first_scale, second_value, second_scale)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("values and scales must be positive and finite")
    if second_scale <= first_scale:
        raise ValueError("scales must be strictly increasing")
    return float(
        math.log(first_value / second_value)
        / math.log(second_scale / first_scale)
    )


def select_asymptotic_pair(
    observations: tuple[tuple[int, float], ...],
    formal_order: int,
    *,
    relative_order_tolerance: float = 0.05,
    floating_point_floor: float = 1e-11,
) -> AsymptoticPair:
    """Select the latest consecutive segment pair in the accepted regime."""
    if formal_order < 1:
        raise ValueError("formal_order must be positive")
    if not 0 < relative_order_tolerance < 1:
        raise ValueError("relative_order_tolerance must lie in (0, 1)")
    if not np.isfinite(floating_point_floor) or floating_point_floor <= 0:
        raise ValueError("floating_point_floor must be positive and finite")
    if len(observations) < 2:
        raise ValueError("at least two observations are required")
    segments = tuple(point[0] for point in observations)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in segments
    ):
        raise ValueError("segment counts must be positive integers")
    if tuple(sorted(set(segments))) != segments:
        raise ValueError("segment counts must be strictly increasing")
    accepted: list[AsymptoticPair] = []
    for (first_segments, first_error), (second_segments, second_error) in zip(
        observations,
        observations[1:],
    ):
        if min(first_error, second_error) <= floating_point_floor:
            continue
        running = effective_power(
            first_error,
            float(first_segments),
            second_error,
            float(second_segments),
        )
        if abs(running - formal_order) <= relative_order_tolerance * formal_order:
            accepted.append(
                AsymptoticPair(
                    first_segments,
                    second_segments,
                    float(first_error),
                    float(second_error),
                    running,
                )
            )
    if not accepted:
        raise ValueError(
            "no consecutive observations satisfy the formal-order and "
            "floating-point-floor criteria"
        )
    return accepted[-1]


def observed_error_coefficient(
    error: float,
    segments: int,
    time: float,
    formal_order: int,
) -> float:
    """Remove the fixed ``r^-q T^(q+1)`` factors from one observation."""
    if segments < 1 or formal_order < 1:
        raise ValueError("segments and formal_order must be positive")
    if not np.isfinite((error, time)).all() or error <= 0 or time <= 0:
        raise ValueError("error and time must be positive and finite")
    return float(error * segments**formal_order / time ** (formal_order + 1))


def select_asymptotic_window(
    observations: tuple[tuple[int, float, bool], ...],
    time: float,
    formal_order: int,
    *,
    minimum_points: int = 4,
    relative_order_tolerance: float = 0.02,
    maximum_coefficient_deviation: float = 0.05,
    maximum_relative_mad: float = 0.02,
) -> AsymptoticWindow:
    """Select the earliest consecutive multi-point asymptotic plateau."""
    if formal_order < 1 or minimum_points < 3:
        raise ValueError("formal_order and minimum_points are invalid")
    if not math.isfinite(time) or time <= 0:
        raise ValueError("time must be positive and finite")
    if len(observations) < minimum_points:
        raise ValueError("insufficient-window: too few observations")
    segments = tuple(point[0] for point in observations)
    if tuple(sorted(set(segments))) != segments or segments[0] < 1:
        raise ValueError("segment counts must be positive and strictly increasing")
    for start in range(len(observations) - minimum_points + 1):
        points = observations[start : start + minimum_points]
        if not all(point[2] for point in points):
            continue
        pairs = tuple((point[0], point[1]) for point in points)
        running = tuple(
            effective_power(first[1], first[0], second[1], second[0])
            for first, second in zip(pairs, pairs[1:])
        )
        if any(
            abs(value - formal_order) > relative_order_tolerance * formal_order
            for value in running
        ):
            continue
        coefficients = tuple(
            observed_error_coefficient(error, count, time, formal_order)
            for count, error in pairs
        )
        median = float(np.median(coefficients))
        maximum_deviation = max(abs(value - median) for value in coefficients) / median
        relative_mad = float(np.median(np.abs(np.asarray(coefficients) - median))) / median
        if maximum_deviation > maximum_coefficient_deviation:
            continue
        if relative_mad > maximum_relative_mad:
            continue
        return AsymptoticWindow(
            pairs,
            running,
            coefficients,
            median,
            maximum_deviation,
            relative_mad,
        )
    raise ValueError(
        "insufficient-window: no consecutive observations satisfy precision, "
        "formal-order, and B_2J plateau criteria"
    )


def validate_time_law(
    observations: tuple[tuple[float, int, float, bool], ...],
    formal_order: int,
    *,
    relative_exponent_tolerance: float = 0.02,
    maximum_coefficient_deviation: float = 0.05,
) -> TimeLawValidation:
    """Validate time scaling from at least three observations at one fixed ``r``."""
    if formal_order < 1 or len(observations) < 3:
        raise ValueError("time-law validation requires a positive order and three points")
    if not all(point[3] for point in observations):
        raise ValueError("time-law validation requires precision-converged points")
    segment_counts = {point[1] for point in observations}
    if len(segment_counts) != 1:
        raise ValueError("time-law validation requires a fixed segment count")
    times = np.asarray([point[0] for point in observations], dtype=float)
    errors = np.asarray([point[2] for point in observations], dtype=float)
    if (
        not np.isfinite(times).all()
        or not np.isfinite(errors).all()
        or np.min(times) <= 0
        or np.min(errors) <= 0
        or tuple(sorted(set(times))) != tuple(times)
    ):
        raise ValueError("time-law observations must be positive, finite, and ordered")
    fitted_exponent = float(np.polyfit(np.log(times), np.log(errors), 1)[0])
    segments = next(iter(segment_counts))
    coefficients = tuple(
        observed_error_coefficient(error, segments, time, formal_order)
        for time, _, error, _ in observations
    )
    median = float(np.median(coefficients))
    maximum_deviation = max(abs(value - median) for value in coefficients) / median
    expected = formal_order + 1
    accepted = (
        abs(fitted_exponent - expected) <= relative_exponent_tolerance * expected
        and maximum_deviation <= maximum_coefficient_deviation
    )
    return TimeLawValidation(
        fitted_exponent,
        expected,
        coefficients,
        maximum_deviation,
        accepted,
    )


def _size_law_parameter_names(model: SizeLawModel) -> tuple[str, ...]:
    if model == "affine":
        return ("slope", "intercept")
    if model == "power":
        return ("amplitude", "exponent")
    if model == "power-plus-offset":
        return ("amplitude", "exponent", "offset")
    if model == "power-with-inverse-size-correction":
        return ("amplitude", "exponent", "inverse_size_correction")
    raise ValueError(f"unknown size-law model: {model}")


def evaluate_size_law(
    model: SizeLawModel,
    parameters: tuple[tuple[str, float], ...],
    system_size: float,
) -> float:
    """Evaluate one fitted coefficient law."""
    if not math.isfinite(system_size) or system_size <= 0:
        raise ValueError("system_size must be positive and finite")
    values = dict(parameters)
    if tuple(values) != _size_law_parameter_names(model):
        raise ValueError("size-law parameters do not match the model")
    if model == "affine":
        result = values["slope"] * system_size + values["intercept"]
    elif model == "power":
        result = values["amplitude"] * system_size ** values["exponent"]
    elif model == "power-plus-offset":
        result = (
            values["amplitude"] * system_size ** values["exponent"]
            + values["offset"]
        )
    else:
        result = (
            values["amplitude"]
            * system_size ** values["exponent"]
            * (1.0 + values["inverse_size_correction"] / system_size)
        )
    if not math.isfinite(result):
        raise ValueError("size-law evaluation is nonfinite")
    return float(result)


def _decoded_parameters(model: SizeLawModel, raw: np.ndarray) -> tuple[tuple[str, float], ...]:
    if model == "affine":
        value_at_four = math.exp(float(raw[0]))
        slope = math.exp(float(raw[1]))
        values = (slope, value_at_four - 4.0 * slope)
    elif model == "power":
        values = (math.exp(float(raw[0])), math.exp(float(raw[1])))
    elif model == "power-plus-offset":
        amplitude = math.exp(float(raw[0]))
        exponent = math.exp(float(raw[1]))
        value_at_four = math.exp(float(raw[2]))
        values = (
            amplitude,
            exponent,
            value_at_four - amplitude * 4.0**exponent,
        )
    else:
        values = (
            math.exp(float(raw[0])),
            math.exp(float(raw[1])),
            float(raw[2]),
        )
    return tuple(zip(_size_law_parameter_names(model), values, strict=True))


def fit_size_law(
    sizes: tuple[int, ...],
    coefficients_b_2j: tuple[float, ...],
    model: SizeLawModel,
    *,
    weights: tuple[float, ...] | None = None,
    reviewed_size_max: int = 100,
) -> SizeLawFit:
    """Fit one monotone positive candidate using weighted log residuals."""
    if len(sizes) != len(coefficients_b_2j) or len(sizes) < 3:
        raise ValueError("at least three aligned size observations are required")
    if tuple(sorted(set(sizes))) != sizes or sizes[0] < 4:
        raise ValueError("sizes must be unique, increasing, and start at N >= 4")
    if reviewed_size_max < sizes[-1]:
        raise ValueError("reviewed_size_max cannot be below observed sizes")
    response = np.asarray(coefficients_b_2j, dtype=float)
    if not np.isfinite(response).all() or np.min(response) <= 0:
        raise ValueError("B_2J observations must be positive and finite")
    weight_array = np.ones(len(sizes)) if weights is None else np.asarray(weights, dtype=float)
    if weight_array.shape != response.shape or np.min(weight_array) <= 0:
        raise ValueError("weights must be positive and aligned")
    x = np.asarray(sizes, dtype=float)
    log_response = np.log(response)
    log_slope = max(
        0.0,
        float((log_response[-1] - log_response[0]) / np.log(x[-1] / x[0])),
    )
    amplitude = float(np.exp(np.mean(log_response - log_slope * np.log(x))))
    if model == "affine":
        initial = np.log((response[0], max((response[-1] - response[0]) / (x[-1] - x[0]), response[0] * 1e-6)))
        bounds = (-np.inf, np.inf)
    elif model == "power":
        initial = np.asarray((math.log(amplitude), math.log(max(log_slope, 1e-6))))
        bounds = (-np.inf, np.inf)
    elif model == "power-plus-offset":
        initial = np.asarray(
            (math.log(amplitude), math.log(max(log_slope, 1e-6)), math.log(response[0]))
        )
        bounds = (-np.inf, np.inf)
    elif model == "power-with-inverse-size-correction":
        initial = np.asarray((math.log(amplitude), math.log(max(log_slope, 1e-6)), 0.0))
        bounds = (
            np.asarray((-np.inf, -np.inf, -3.999999)),
            np.asarray((np.inf, np.inf, np.inf)),
        )
    else:
        raise ValueError(f"unknown size-law model: {model}")

    def residual(raw: np.ndarray) -> np.ndarray:
        parameters = _decoded_parameters(model, raw)
        predictions = np.asarray(
            [evaluate_size_law(model, parameters, value) for value in x]
        )
        if np.min(predictions) <= 0:
            return np.full(len(x), 1e6)
        return np.sqrt(weight_array) * (np.log(predictions) - log_response)

    result = least_squares(residual, initial, bounds=bounds, max_nfev=20_000)
    parameters = _decoded_parameters(model, result.x)
    domain = range(4, reviewed_size_max + 1)
    domain_values = tuple(evaluate_size_law(model, parameters, value) for value in domain)
    monotone = all(second >= first for first, second in zip(domain_values, domain_values[1:]))
    predictions = tuple(evaluate_size_law(model, parameters, value) for value in sizes)
    relative_residuals = tuple(
        (prediction - observed) / observed
        for prediction, observed in zip(predictions, coefficients_b_2j, strict=True)
    )
    rss = float(np.dot(residual(result.x), residual(result.x)))
    parameter_count = len(parameters)
    observation_count = len(sizes)
    if rss == 0:
        aic = -math.inf
    else:
        aic = observation_count * math.log(rss / observation_count) + 2 * parameter_count
    if observation_count <= parameter_count + 1:
        aicc = math.inf
    else:
        aicc = aic + (
            2 * parameter_count * (parameter_count + 1)
            / (observation_count - parameter_count - 1)
        )
    return SizeLawFit(
        model,
        parameters,
        sizes,
        coefficients_b_2j,
        predictions,
        relative_residuals,
        rss,
        float(aicc),
        bool(result.success and min(domain_values) > 0 and monotone),
    )


def _has_monotone_residual_drift(fit: SizeLawFit) -> bool:
    residuals = fit.relative_residuals[-3:]
    same_sign = all(value > 0 for value in residuals) or all(value < 0 for value in residuals)
    magnitudes = tuple(abs(value) for value in residuals)
    return max(magnitudes) > 1e-4 and same_sign and all(
        second >= first for first, second in zip(magnitudes, magnitudes[1:])
    )


def shifted_window_stability(
    sizes: tuple[int, ...],
    coefficients_b_2j: tuple[float, ...],
    model: SizeLawModel,
    *,
    reviewed_size_max: int,
    strict: bool = False,
) -> FiniteSizeStability:
    """Compare one model over the shifted windows 4..8, 5..9, and 6..10."""
    lookup = dict(zip(sizes, coefficients_b_2j, strict=True))
    windows = tuple(tuple(range(start, start + 5)) for start in (4, 5, 6))
    if any(any(size not in lookup for size in window) for window in windows):
        raise ValueError("shifted-window stability requires observations for N=4..10")
    fits = tuple(
        fit_size_law(
            window,
            tuple(lookup[size] for size in window),
            model,
            reviewed_size_max=reviewed_size_max,
        )
        for window in windows
    )
    anchor_limits = (
        ((12, 0.05), (20, 0.10), (50, 0.15), (100, 0.20))
        if strict
        else ((12, 0.10), (20, 0.15), (50, 0.25), (100, 0.35))
    )
    applicable: dict[int, float] = {}
    for size, limit in anchor_limits:
        if size <= reviewed_size_max:
            applicable[size] = min(limit, applicable.get(size, limit))
    if reviewed_size_max not in applicable:
        lower = max(
            (point for point in anchor_limits if point[0] < reviewed_size_max),
            default=anchor_limits[0],
        )
        upper = min(
            (point for point in anchor_limits if point[0] > reviewed_size_max),
            default=anchor_limits[-1],
        )
        if lower[0] == upper[0]:
            reviewed_limit = lower[1]
        else:
            fraction = math.log(reviewed_size_max / lower[0]) / math.log(
                upper[0] / lower[0]
            )
            reviewed_limit = lower[1] + fraction * (upper[1] - lower[1])
        applicable[reviewed_size_max] = reviewed_limit
    spreads: list[tuple[int, float]] = []
    prediction_stable = True
    for size, limit in sorted(applicable.items()):
        predictions = tuple(fit.at(size) for fit in fits)
        median = float(np.median(predictions))
        spread = (max(predictions) - min(predictions)) / median
        spreads.append((size, spread))
        prediction_stable &= spread <= limit
    if model == "affine":
        slopes = tuple(dict(fit.parameters)["slope"] for fit in fits)
        parameter_stable = (
            (max(slopes) - min(slopes)) / float(np.median(slopes)) <= 0.20
        )
    else:
        exponents = tuple(dict(fit.parameters)["exponent"] for fit in fits)
        parameter_stable = max(exponents) - min(exponents) <= 0.15
    residual_drift_free = not any(_has_monotone_residual_drift(fit) for fit in fits)
    failure_reasons: list[str] = []
    if not all(fit.converged for fit in fits):
        failure_reasons.append("one or more shifted-window fits are inadmissible")
    if not prediction_stable:
        failure_reasons.append("extrapolated B_2J is unstable across shifted windows")
    if not parameter_stable:
        failure_reasons.append("size-law parameters are unstable across shifted windows")
    if not residual_drift_free:
        failure_reasons.append("largest-size residuals show monotone same-sign drift")
    return FiniteSizeStability(
        model,
        reviewed_size_max,
        fits,
        tuple(spreads),
        parameter_stable,
        residual_drift_free,
        not failure_reasons,
        tuple(failure_reasons),
    )


def select_size_law_model(
    sizes: tuple[int, ...],
    coefficients_b_2j: tuple[float, ...],
    *,
    reviewed_size_max: int,
    models: tuple[SizeLawModel, ...] = ("affine", "power", "power-plus-offset"),
) -> SizeLawSelection:
    """Select a size law using N=4..8 training, N=9,10 holdouts, and stability."""
    lookup = dict(zip(sizes, coefficients_b_2j, strict=True))
    if any(size not in lookup for size in range(4, 11)):
        raise ValueError("size-law selection requires observations for N=4..10")
    training_sizes = tuple(range(4, 9))
    training_values = tuple(lookup[size] for size in training_sizes)
    candidates = tuple(
        fit_size_law(
            training_sizes,
            training_values,
            model,
            reviewed_size_max=reviewed_size_max,
        )
        for model in models
    )
    holdout_errors: list[tuple[SizeLawModel, tuple[float, ...]]] = []
    stability_results: list[FiniteSizeStability] = []
    admitted: list[SizeLawFit] = []
    for fit in candidates:
        errors = tuple(
            abs(fit.at(size) - lookup[size]) / lookup[size]
            for size in (9, 10)
        )
        holdout_errors.append((fit.model, errors))
        stability = shifted_window_stability(
            sizes,
            coefficients_b_2j,
            fit.model,
            reviewed_size_max=reviewed_size_max,
        )
        stability_results.append(stability)
        if (
            fit.converged
            and float(np.median(errors)) <= 0.10
            and max(errors) <= 0.20
            and stability.accepted
        ):
            admitted.append(fit)
    simpler = [fit for fit in admitted if len(fit.parameters) == 2]
    selected: SizeLawFit | None = None
    if simpler:
        error_lookup = dict(holdout_errors)
        selected = min(
            simpler,
            key=lambda fit: (
                float(np.median(error_lookup[fit.model])),
                fit.aicc,
                0 if fit.model == "affine" else 1,
            ),
        )
    complex_fits = [fit for fit in admitted if len(fit.parameters) > 2]
    if selected is None and complex_fits:
        selected = min(complex_fits, key=lambda fit: fit.aicc)
    elif selected is not None:
        baseline_errors = dict(holdout_errors)[selected.model]
        for candidate in sorted(complex_fits, key=lambda fit: fit.aicc):
            candidate_errors = dict(holdout_errors)[candidate.model]
            baseline_median = float(np.median(baseline_errors))
            median_improvement = 1.0 - (
                float(np.median(candidate_errors))
                / max(baseline_median, 1e-300)
            )
            if (
                baseline_median > 1e-6
                and candidate.aicc <= selected.aicc - 4.0
                and median_improvement >= 0.20
                and max(candidate_errors) <= max(baseline_errors)
            ):
                selected = candidate
                baseline_errors = candidate_errors
                break
    failures = () if selected is not None else (
        "no size law passed holdout and shifted-window stability gates",
    )
    return SizeLawSelection(
        selected,
        candidates,
        tuple(holdout_errors),
        tuple(stability_results),
        failures,
    )


def validate_external_sizes(
    fit: SizeLawFit,
    all_sizes: tuple[int, ...],
    all_coefficients_b_2j: tuple[float, ...],
    *,
    reviewed_size_max: int,
) -> ExternalSizeValidation:
    """Validate N=11,12 before incorporating them into a final refit."""
    lookup = dict(zip(all_sizes, all_coefficients_b_2j, strict=True))
    if any(size not in lookup for size in range(4, 13)):
        raise ValueError("external validation requires observations for N=4..12")
    signed = tuple((fit.at(size) - lookup[size]) / lookup[size] for size in (11, 12))
    errors = tuple(abs(value) for value in signed)
    same_direction = (
        all(value > 0.10 for value in signed)
        or all(value < -0.10 for value in signed)
    )
    refit = fit_size_law(
        tuple(range(4, 13)),
        tuple(lookup[size] for size in range(4, 13)),
        fit.model,
        reviewed_size_max=reviewed_size_max,
    )

    def movement(size: int) -> float:
        return abs(refit.at(size) - fit.at(size)) / fit.at(size)

    movement_50 = movement(min(50, reviewed_size_max))
    movement_max = movement(reviewed_size_max)
    accepted = (
        max(errors) <= 0.15
        and not same_direction
        and movement_50 <= 0.15
        and movement_max <= 0.20
    )
    return ExternalSizeValidation(
        errors,
        same_direction,
        movement_50,
        movement_max,
        accepted,
    )


def fit_affine_size_coefficient(
    sizes: tuple[int, ...],
    coefficients: tuple[float, ...],
) -> AffineCalibrationFit:
    """Fit a reviewed one-dimensional bulk-plus-boundary coefficient."""
    if len(sizes) != len(coefficients) or len(sizes) < 3:
        raise ValueError("at least three aligned sizes and coefficients are required")
    if tuple(sorted(set(sizes))) != sizes or sizes[0] < 1:
        raise ValueError("sizes must be positive and strictly increasing")
    if not np.isfinite(coefficients).all() or min(coefficients) <= 0:
        raise ValueError("coefficients must be positive and finite")
    design = np.column_stack((np.asarray(sizes, dtype=float), np.ones(len(sizes))))
    response = np.asarray(coefficients, dtype=float)
    slope, intercept = np.linalg.lstsq(design, response, rcond=None)[0]
    fitted = design @ np.asarray((slope, intercept))
    residuals = response - fitted
    squared_error = float(np.dot(residuals, residuals))
    centered = response - np.mean(response)
    total = float(np.dot(centered, centered))
    r_squared = 1.0 if total == 0 else 1.0 - squared_error / total
    variance = squared_error / (len(sizes) - 2)
    covariance = variance * np.linalg.inv(design.T @ design)
    return AffineCalibrationFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        root_mean_square_error=float(np.sqrt(squared_error / len(sizes))),
        slope_standard_error=float(np.sqrt(covariance[0, 0])),
        intercept_standard_error=float(np.sqrt(covariance[1, 1])),
        residuals=tuple(float(value) for value in residuals),
    )


@dataclass(frozen=True)
class _SparsePauliAction:
    """Vectorized action of one real-weighted Pauli word."""

    coefficient: float
    permutation: np.ndarray
    phase: np.ndarray

    def apply(self, vector: np.ndarray) -> np.ndarray:
        result = np.empty_like(vector)
        result[self.permutation] = self.phase * vector
        return result


def _sparse_pauli_action(label: str, coefficient: float) -> _SparsePauliAction:
    num_qubits = len(label)
    dimension = 2**num_qubits
    x_mask = 0
    z_mask = 0
    y_count = 0
    for offset, pauli in enumerate(reversed(label)):
        if pauli in "XY":
            x_mask |= 1 << offset
        if pauli in "YZ":
            z_mask |= 1 << offset
        if pauli == "Y":
            y_count += 1
    indices = np.arange(dimension, dtype=np.uint64)
    permutation = np.asarray(indices ^ x_mask, dtype=np.intp)
    parity = np.zeros(dimension, dtype=np.uint8)
    masked = indices & z_mask
    while np.any(masked):
        parity ^= np.asarray(masked & 1, dtype=np.uint8)
        masked >>= 1
    phase = (1j**y_count) * (1 - 2 * parity.astype(np.int8))
    return _SparsePauliAction(float(coefficient), permutation, phase.astype(complex))


@lru_cache(maxsize=None)
def _dense_term_matrices(hamiltonian: PauliHamiltonian) -> tuple[np.ndarray, ...]:
    matrices = []
    for label, coefficient in hamiltonian.terms:
        term = PauliHamiltonian.from_terms(
            hamiltonian.num_qubits,
            [(label, coefficient)],
        )
        matrices.append(np.asarray(term.matrix(), dtype=complex))
    return tuple(matrices)


@lru_cache(maxsize=None)
def _dense_groups(
    hamiltonian: PauliHamiltonian,
    order: int,
    partition: TrotterPartition,
) -> tuple[np.ndarray, ...]:
    terms = _dense_term_matrices(hamiltonian)
    structure = resolve_trotter_structure(hamiltonian, order, partition)
    return tuple(
        sum(
            (terms[index] for index in group),
            start=np.zeros_like(terms[0]),
        )
        for group in structure.group_term_indices
    )


def _dense_product_step(
    groups: tuple[np.ndarray, ...],
    step_time: float,
    order: int,
) -> np.ndarray:
    spectra = tuple(np.linalg.eigh(group) for group in groups)
    return _dense_product_step_from_spectra(spectra, step_time, order)


def _dense_product_step_from_spectra(
    spectra: tuple[tuple[np.ndarray, np.ndarray], ...],
    step_time: float,
    order: int,
) -> np.ndarray:
    result = np.eye(spectra[0][1].shape[0], dtype=complex)
    for group, coefficient in suzuki_group_factors(len(spectra), order):
        eigenvalues, eigenvectors = spectra[group]
        phases = np.exp(-1j * coefficient * step_time * eigenvalues)
        unitary = (eigenvectors * phases) @ eigenvectors.conj().T
        result = unitary @ result
    return result


def dense_trotter_operator(
    hamiltonian: PauliHamiltonian,
    time: float,
    repetitions: int,
    order: int,
    *,
    partition: TrotterPartition = "auto",
) -> np.ndarray:
    groups = _dense_groups(hamiltonian, order, partition)
    step = _dense_product_step(groups, float(time) / repetitions, order)
    return np.linalg.matrix_power(step, repetitions)


def dense_mpf_operator(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    m: int,
    *,
    schedule: MPFSchedule = "new",
) -> np.ndarray:
    groups = _dense_groups(hamiltonian, 2, "individual")
    spectra = tuple(np.linalg.eigh(group) for group in groups)
    exponents = optimal_mpf_exponents(m, schedule=schedule)
    coefficients = multiproduct_coefficients(m, schedule=schedule)
    step_time = float(time) / segments
    mpf_step = np.zeros_like(groups[0])
    for coefficient, exponent in zip(coefficients, exponents, strict=True):
        base = _dense_product_step_from_spectra(spectra, step_time / exponent, 2)
        mpf_step += float(coefficient) * np.linalg.matrix_power(base, exponent)
    return np.linalg.matrix_power(mpf_step, segments)


def dense_operator_norm_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    *,
    algorithm: CalibrationAlgorithm,
    formal_order: int,
    partition: TrotterPartition = "auto",
    schedule: MPFSchedule = "new",
) -> OperatorNormEstimate:
    exact = _dense_exact_evolution(hamiltonian, float(time))
    if algorithm == "trotter":
        approximate = dense_trotter_operator(
            hamiltonian,
            time,
            segments,
            formal_order,
            partition=partition,
        )
    elif algorithm == "multiproduct":
        if formal_order % 2:
            raise ValueError("MPF formal_order must be even")
        approximate = dense_mpf_operator(
            hamiltonian,
            time,
            segments,
            formal_order // 2,
            schedule=schedule,
        )
    else:
        raise ValueError("unknown calibration algorithm")
    value = float(np.linalg.norm(approximate - exact, ord=2))
    return OperatorNormEstimate(value, "dense-svd", True, 1, 1, 0.0)


@lru_cache(maxsize=None)
def _dense_exact_evolution(
    hamiltonian: PauliHamiltonian,
    time: float,
) -> np.ndarray:
    return expm(-1j * float(time) * hamiltonian.matrix())


def _sparse_groups(
    hamiltonian: PauliHamiltonian,
    order: int,
    partition: TrotterPartition,
):
    terms = tuple(
        _sparse_pauli_action(label, coefficient)
        for label, coefficient in hamiltonian.terms
    )
    structure = resolve_trotter_structure(hamiltonian, order, partition)
    return tuple(tuple(terms[index] for index in group) for group in structure.group_term_indices)


def _apply_product_step(
    groups,
    vector: np.ndarray,
    step_time: float,
    order: int,
    *,
    adjoint: bool,
) -> np.ndarray:
    factors = suzuki_group_factors(len(groups), order)
    if adjoint:
        factors = tuple(reversed(factors))
    result = vector
    sign = 1j if adjoint else -1j
    for group, coefficient in factors:
        for term in groups[group]:
            angle = coefficient * step_time * term.coefficient
            result = (
                np.cos(angle) * result
                + sign * np.sin(angle) * term.apply(result)
            )
    return np.asarray(result)


def _apply_repeated_product(
    groups,
    vector: np.ndarray,
    time: float,
    repetitions: int,
    order: int,
    *,
    adjoint: bool,
) -> np.ndarray:
    result = vector
    for _ in range(repetitions):
        result = _apply_product_step(
            groups,
            result,
            time / repetitions,
            order,
            adjoint=adjoint,
        )
    return result


def _apply_mpf_step(
    groups,
    vector: np.ndarray,
    step_time: float,
    exponents: tuple[int, ...],
    coefficients: np.ndarray,
    *,
    adjoint: bool,
) -> np.ndarray:
    result = np.zeros_like(vector)
    for coefficient, exponent in zip(coefficients, exponents, strict=True):
        branch = vector
        for _ in range(exponent):
            branch = _apply_product_step(
                groups,
                branch,
                step_time / exponent,
                2,
                adjoint=adjoint,
            )
        result += float(coefficient) * branch
    return result


def sparse_operator_norm_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    segments: int,
    *,
    algorithm: CalibrationAlgorithm,
    formal_order: int,
    partition: TrotterPartition = "auto",
    schedule: MPFSchedule = "new",
    tolerance: float = 1e-8,
    max_iterations: int = 80,
    restarts: int = 3,
    seed: int = 2026,
) -> OperatorNormEstimate:
    """Estimate ``||U_approx-U||_2`` through power iteration on ``D^dagger D``."""
    if segments < 1 or max_iterations < 1 or restarts < 1:
        raise ValueError("segments, max_iterations, and restarts must be positive")
    if algorithm == "trotter":
        groups = _sparse_groups(hamiltonian, formal_order, partition)

        def approximate(vector: np.ndarray, adjoint: bool) -> np.ndarray:
            return _apply_repeated_product(
                groups,
                vector,
                float(time),
                segments,
                formal_order,
                adjoint=adjoint,
            )

    elif algorithm == "multiproduct":
        if formal_order % 2:
            raise ValueError("MPF formal_order must be even")
        groups = _sparse_groups(hamiltonian, 2, "individual")
        exponents = optimal_mpf_exponents(formal_order // 2, schedule=schedule)
        coefficients = multiproduct_coefficients(formal_order // 2, schedule=schedule)

        def approximate(vector: np.ndarray, adjoint: bool) -> np.ndarray:
            result = vector
            for _ in range(segments):
                result = _apply_mpf_step(
                    groups,
                    result,
                    float(time) / segments,
                    exponents,
                    coefficients,
                    adjoint=adjoint,
                )
            return result

    else:
        raise ValueError("unknown calibration algorithm")

    exact_matrix = hamiltonian.to_sparse_pauli_op().to_matrix(sparse=True).tocsr()

    def difference(vector: np.ndarray, adjoint: bool) -> np.ndarray:
        sign = 1j if adjoint else -1j
        exact = expm_multiply(sign * float(time) * exact_matrix, vector)
        return approximate(vector, adjoint) - exact

    dimension = 2**hamiltonian.num_qubits
    operator = LinearOperator(
        (dimension, dimension),
        matvec=lambda vector: difference(difference(vector, False), True),
        dtype=np.complex128,
    )
    rng = np.random.default_rng(seed)
    best = 0.0
    best_iterations = 0
    best_residual = np.inf
    any_converged = False
    for _ in range(restarts):
        vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)
        previous = 0.0
        residual = np.inf
        for iteration in range(1, max_iterations + 1):
            applied = np.asarray(operator.matvec(vector))
            norm = float(np.linalg.norm(applied))
            if norm == 0:
                eigenvalue = 0.0
                residual = 0.0
                converged = True
                break
            vector = applied / norm
            eigenvalue = max(0.0, norm)
            converged = previous > 0 and (
                abs(eigenvalue - previous) <= tolerance * max(eigenvalue, 1e-300)
            )
            previous = eigenvalue
            if converged:
                break
        reapplied = np.asarray(operator.matvec(vector))
        eigenvalue = max(0.0, float(np.vdot(vector, reapplied).real))
        residual = float(np.linalg.norm(reapplied - eigenvalue * vector)) / max(
            1.0,
            eigenvalue,
        )
        value = float(np.sqrt(eigenvalue))
        if value > best:
            best = value
            best_iterations = iteration
            best_residual = residual
        any_converged = any_converged or converged
    return OperatorNormEstimate(
        best,
        "sparse-power",
        any_converged,
        best_iterations,
        restarts,
        best_residual,
    )
