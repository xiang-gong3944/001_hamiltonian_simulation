"""Behavior-preservation checks for the c88df30 structural-refactor baseline.

These tests intentionally characterize existing values and certification scopes.
They do not assert that the underlying mathematical models are permanently correct.
"""

import json
from pathlib import Path

import pytest

from hamiltonian_resources import (
    BenchmarkConfig,
    HamiltonianSpec,
    MultiproductMethod,
    QSVTMethod,
    TimeScaling,
    TrotterMethod,
    run_benchmark,
    transverse_field_ising,
)


_BASELINE = json.loads(
    (Path(__file__).parent / "data" / "c88df30_structural_refactor_baseline.json").read_text(
        encoding="utf-8"
    )
)


def _method(raw):
    if raw["family"] == "trotter":
        return TrotterMethod(raw["order"])
    if raw["family"] == "multiproduct":
        return MultiproductMethod(
            raw["term_count"],
            schedule=raw["schedule"],
            error_method=raw["error_method"],
        )
    return QSVTMethod()


@pytest.mark.parametrize("case", _BASELINE["cases"], ids=lambda case: case["expected"]["method_id"])
def test_c88df30_structural_refactor_baseline(case):
    config = BenchmarkConfig(
        hamiltonian=HamiltonianSpec(
            "tfim-characterization",
            {"coupling": 1.0, "field": 0.7, "periodic": False},
            transverse_field_ising,
        ),
        system_sizes=[case["system_qubits"]],
        target_errors=[case["target_error"]],
        time=TimeScaling("fixed", case["evolution_time"]),
        fixed_target_error=case["target_error"],
        methods=[_method(case["method"])],
    )
    row = run_benchmark(config, sweeps="system-size").iloc[0]

    assert row["status"] == "ok"
    for field, expected in case["expected"].items():
        if isinstance(expected, float):
            assert row[field] == pytest.approx(expected), field
        else:
            assert row[field] == expected, field
