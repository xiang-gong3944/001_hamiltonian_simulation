"""Notebook-equivalent empirical comparison smoke test without notebook writes."""

from hamiltonian_resources import (
    BenchmarkConfig,
    HamiltonianSpec,
    MultiproductMethod,
    TrotterMethod,
    run_benchmark,
)


def test_packaged_empirical_comparison_smoke_uses_reviewed_v2_rows():
    config = BenchmarkConfig(
        hamiltonian=HamiltonianSpec(
            "transverse_field_ising",
            {"coupling": 1.0, "field": 3.0, "periodic": False},
        ),
        system_sizes=[4, 8],
        target_errors=[1e-3],
        fixed_system_size=8,
        fixed_target_error=1e-3,
        methods=[
            TrotterMethod(2, error_policy="empirical-operator-norm"),
            MultiproductMethod(8, error_method="empirical-operator-norm"),
        ],
    )

    frame = run_benchmark(config, sweeps="system-size")

    assert set(frame["status"]) == {"ok"}
    assert set(frame["estimate_category"]) == {"empirical"}
    assert set(frame["empirical_calibration_schema_version"]) == {"2.0"}
    assert set(frame["empirical_coefficient_model"]) == {"affine"}
    assert not frame["empirical_size_extrapolated"].any()
