import math
import json
from pathlib import Path

import pytest

from hamiltonian_resources.calibration_study import (
    fit_size_law,
    select_asymptotic_window,
    select_size_law_model,
    shifted_window_stability,
    validate_time_law,
    validate_external_sizes,
)


def _errors(coefficient, time, formal_order, segments):
    return tuple(
        (
            count,
            coefficient * time ** (formal_order + 1) / count**formal_order,
            True,
        )
        for count in segments
    )


def test_select_asymptotic_window_requires_order_and_coefficient_plateau():
    window = select_asymptotic_window(
        _errors(3.2e-12, 4.0, 24, (20, 24, 32, 40)),
        4.0,
        24,
    )

    assert window.median_coefficient_b_2j == pytest.approx(3.2e-12)
    assert window.running_exponents == pytest.approx((24.0, 24.0, 24.0))
    assert window.maximum_relative_deviation < 1e-12


def test_select_asymptotic_window_rejects_precision_and_plateau_failures():
    observations = list(_errors(2e-7, 4.0, 18, (12, 16, 20, 24)))
    observations[1] = (observations[1][0], observations[1][1], False)
    with pytest.raises(ValueError, match="insufficient-window"):
        select_asymptotic_window(tuple(observations), 4.0, 18)

    drifting = tuple(
        (segments, error * (1 + 0.08 * index), True)
        for index, (segments, error, _) in enumerate(
            _errors(2e-7, 4.0, 18, (12, 16, 20, 24))
        )
    )
    with pytest.raises(ValueError, match="B_2J plateau"):
        select_asymptotic_window(drifting, 4.0, 18)


@pytest.mark.parametrize("system_size", [4, 5, 6])
@pytest.mark.parametrize("branch_count", [9, 12, 15])
def test_reduced_pilot_observations_pass_refined_window(system_size, branch_count):
    project_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            project_root
            / "docs"
            / "calibration_data"
            / "tfim_high_order_pilot_v1_reduced.json"
        ).read_text(encoding="utf-8")
    )
    rows = sorted(
        (
            row
            for row in payload["observations"]
            if row["system_size"] == system_size
            and row["branch_count"] == branch_count
        ),
        key=lambda row: row["segments"],
    )
    window = select_asymptotic_window(
        tuple(
            (
                row["segments"],
                float(row["error"]),
                row["precision_converged"],
            )
            for row in rows
        ),
        float(system_size),
        2 * branch_count,
    )

    assert window.maximum_relative_deviation <= 0.05
    assert window.relative_median_absolute_deviation <= 0.02


def test_time_law_uses_fixed_segments_and_formal_order_plus_one():
    formal_order = 18
    coefficient = 2e-7
    segments = 40
    observations = tuple(
        (
            time,
            segments,
            coefficient * time ** (formal_order + 1) / segments**formal_order,
            True,
        )
        for time in (3.2, 4.0, 4.8)
    )
    validation = validate_time_law(observations, formal_order)

    assert validation.accepted
    assert validation.fitted_exponent == pytest.approx(19.0)
    assert validation.maximum_relative_coefficient_deviation < 1e-12


def test_power_size_law_passes_holdout_and_shifted_window_stability():
    sizes = tuple(range(4, 13))
    values = tuple(2.5e-9 * size**1.4 for size in sizes)
    selection = select_size_law_model(
        sizes,
        values,
        reviewed_size_max=100,
    )

    assert selection.selected is not None
    assert selection.selected.model == "power"
    assert dict(selection.selected.parameters)["exponent"] == pytest.approx(1.4)
    stability = next(
        item for item in selection.stability if item.model == "power"
    )
    assert stability.accepted
    assert max(dict(stability.prediction_spreads).values()) < 1e-10


def test_shifted_window_stability_rejects_finite_size_crossover():
    sizes = tuple(range(4, 11))
    values = tuple(
        1e-8 * size ** (1.0 if size <= 7 else 2.2) / (7**1.2 if size > 7 else 1)
        for size in sizes
    )
    stability = shifted_window_stability(
        sizes,
        values,
        "power",
        reviewed_size_max=100,
    )

    assert not stability.accepted
    assert stability.failure_reasons


def test_external_size_validation_is_kept_out_of_initial_fit():
    sizes = tuple(range(4, 13))
    values = tuple(4e-10 * size**1.2 for size in sizes)
    fit = fit_size_law(
        tuple(range(4, 11)),
        values[:7],
        "power",
        reviewed_size_max=100,
    )
    validation = validate_external_sizes(
        fit,
        sizes,
        values,
        reviewed_size_max=100,
    )

    assert validation.accepted
    assert max(validation.prediction_errors) < 1e-10
    assert math.isclose(validation.prediction_movement_reviewed_max, 0.0, abs_tol=1e-10)
