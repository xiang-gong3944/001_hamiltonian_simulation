"""Hamiltonian-simulation circuit and resource-comparison toolkit."""

from .benchmark import (
    BenchmarkConfig,
    benchmark_scaling,
    choose_parameters,
    estimate_resources_analytically,
)
from .hamiltonians import PauliHamiltonian, heisenberg_chain, transverse_field_ising
from .multiproduct import (
    build_multiproduct_circuit,
    multiproduct_coefficients,
    optimal_mpf_exponents,
)
from .qsvt import (
    build_hamiltonian_qsvt_circuit,
    build_qsvt_circuit,
    estimate_qsvt_degree,
    synthesize_hamsim_phases,
)
from .resources import ResourceEstimate, count_circuit_resources
from .simulation import compare_with_exact
from .trotter import build_trotter_circuit

__all__ = [
    "BenchmarkConfig",
    "PauliHamiltonian",
    "ResourceEstimate",
    "benchmark_scaling",
    "build_multiproduct_circuit",
    "build_hamiltonian_qsvt_circuit",
    "build_qsvt_circuit",
    "build_trotter_circuit",
    "choose_parameters",
    "compare_with_exact",
    "count_circuit_resources",
    "estimate_qsvt_degree",
    "estimate_resources_analytically",
    "heisenberg_chain",
    "multiproduct_coefficients",
    "optimal_mpf_exponents",
    "synthesize_hamsim_phases",
    "transverse_field_ising",
]
