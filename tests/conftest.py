"""Shared fixtures for small, deterministic benchmark integration tests."""

import pytest

from hamiltonian_resources import (
    BenchmarkConfig,
    HamiltonianSpec,
    MultiproductMethod,
    QSVTMethod,
    TimeScaling,
    TrotterMethod,
)


@pytest.fixture
def small_benchmark_config() -> BenchmarkConfig:
    """A representative fixed-time job suitable for persistence and plotting."""
    return BenchmarkConfig(
        hamiltonian=HamiltonianSpec(
            parameters={"coupling": 1.0, "field": 0.7, "periodic": False}
        ),
        system_sizes=[2, 3],
        target_errors=[1e-2, 1e-3],
        time=TimeScaling("fixed", 0.2),
        fixed_system_size=3,
        fixed_target_error=1e-2,
        methods=[TrotterMethod(2), MultiproductMethod(3), QSVTMethod()],
    )
