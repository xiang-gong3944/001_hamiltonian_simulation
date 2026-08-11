from __future__ import annotations

from copy import deepcopy

from hamiltonian_resources.calibration_checkpoint import (
    analyze_n9_checkpoint,
    available_shifted_window_stability,
)


def test_available_shifted_windows_are_partial_and_fail_closed() -> None:
    sizes = tuple(range(4, 10))
    coefficients = tuple(2.0 * size + 1.0 for size in sizes)

    result = available_shifted_window_stability(
        sizes,
        coefficients,
        "affine",
        weights=(1.0,) * len(sizes),
        reviewed_size_max=100,
    )

    assert result["available_windows"] == [[4, 8], [5, 9]]
    assert result["missing_required_window"] == [6, 10]
    assert result["necessary_conditions_pass"] is True
    assert result["full_gate_evaluated"] is False
    assert result["full_gate_passed"] is False
    assert result["prediction_spreads"]["100"]["spread"] < 1e-8


def _checkpoint_inputs(*, sentinel_accepted: bool) -> tuple[dict, dict]:
    model = "synthetic"
    formal_order = 22
    accepted_windows = []
    tasks = []
    for size in range(4, 10):
        accepted_windows.append(
            {
                "model": model,
                "formal_order": formal_order,
                "system_size": size,
                "coefficient_b_2j": 2.0 * size + 1.0,
                "maximum_relative_deviation": 0.01,
                "relative_mad": 0.005,
                "segments": [4, 5, 6, 8],
            }
        )
        tasks.append(
            {
                "task": {
                    "kind": "primary",
                    "model": model,
                    "formal_order": formal_order,
                    "system_size": size,
                },
                "observations": [
                    {
                        "segments": segment,
                        "wall_seconds": float(size),
                        "relative_precision_change": 1e-10,
                    }
                    for segment in (4, 5, 6, 8)
                ],
            }
        )
    reduced = {
        "study_id": "synthetic-n9",
        "reviewed_size_max": 100,
        "reduced_digest": "reduced",
        "configuration": {
            "models": [model],
            "formal_orders": [formal_order],
            "time_law_checks": [
                {
                    "model": model,
                    "formal_order": 18,
                    "system_size": 8,
                }
            ],
        },
        "tasks": tasks,
    }
    assembled = {
        "assembled_digest": "assembled",
        "accepted_windows": accepted_windows,
        "time_law_checks": [
            {
                "model": model,
                "formal_order": formal_order,
                "system_size": 4,
                "accepted": True,
                "maximum_relative_coefficient_deviation": 0.01,
            },
            {
                "model": model,
                "formal_order": 18,
                "system_size": 8,
                "accepted": sentinel_accepted,
                "maximum_relative_coefficient_deviation": 0.01,
            },
        ],
    }
    return reduced, assembled


def test_failed_sentinel_keeps_numerical_row_pending() -> None:
    reduced, assembled = _checkpoint_inputs(sentinel_accepted=False)

    checkpoint = analyze_n9_checkpoint(reduced, assembled)
    row = checkpoint["rows"][0]

    assert row["numerical_N4_through_N9_passed"] is True
    assert row["time_law_gate"]["status"] == "pending-required-N8-expansion"
    assert row["all_current_numerical_and_time_law_gates_passed"] is False
    assert row["n10_scientific_need"] == "defer-until-required-N8-time-law-check"
    assert checkpoint["N10_tasks_launched"] is False
    assert checkpoint["full_finite_size_gate_evaluated"] is False


def test_passing_sentinel_allows_gate_complete_checkpoint_row() -> None:
    reduced, assembled = _checkpoint_inputs(sentinel_accepted=True)
    assembled = deepcopy(assembled)

    checkpoint = analyze_n9_checkpoint(reduced, assembled)
    row = checkpoint["rows"][0]

    assert row["time_law_gate"]["status"] == "passed"
    assert row["all_current_numerical_and_time_law_gates_passed"] is True
    assert row["n10_scientific_need"].startswith("needed-")
