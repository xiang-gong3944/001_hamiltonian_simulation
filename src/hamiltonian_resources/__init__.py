"""Hamiltonian-simulation circuit and resource-comparison toolkit."""

from .benchmark import (
    BenchmarkConfig,
    benchmark_scaling,
    choose_parameters,
    estimate_resources_analytically,
)
from .benchmark_suite import (
    BENCHMARK_COLUMNS,
    METHOD_LABELS,
    MPF_TERM_COUNTS,
    SCHEMA_VERSION,
    TROTTER_ORDERS,
    ScalingBenchmarkConfig,
    generate_and_save_benchmark,
    generate_benchmark_sweep,
    load_benchmark_config,
    load_benchmark_data,
    save_benchmark_data,
    validate_benchmark_frame,
)
from .benchmark_plotting import (
    FAMILY_COLORS,
    METHOD_STYLES,
    SUMMARY_STYLES,
    create_benchmark_figure,
    plot_saved_benchmark,
)
from .hamiltonians import PauliHamiltonian, heisenberg_chain, transverse_field_ising
from .multiproduct import (
    MPFSchedule,
    build_multiproduct_circuit,
    multiproduct_coefficients,
    optimal_mpf_exponents,
)
from .qsvt import (
    HamiltonianSimulationPhases,
    build_hamiltonian_qsvt_circuit,
    estimate_qsvt_degree,
    synthesize_hamsim_phases,
)
from .resources import ResourceEstimate, count_circuit_resources
from .simulation import compare_with_exact
from .trotter import (
    SuzukiErrorEstimate,
    TrotterPartition,
    build_trotter_circuit,
    estimate_suzuki_error,
    suzuki_commutator_bounds,
)

__all__ = [
    "BenchmarkConfig",
    "BENCHMARK_COLUMNS",
    "FAMILY_COLORS",
    "HamiltonianSimulationPhases",
    "METHOD_LABELS",
    "METHOD_STYLES",
    "MPF_TERM_COUNTS",
    "MPFSchedule",
    "PauliHamiltonian",
    "ResourceEstimate",
    "SCHEMA_VERSION",
    "SUMMARY_STYLES",
    "ScalingBenchmarkConfig",
    "SuzukiErrorEstimate",
    "TrotterPartition",
    "TROTTER_ORDERS",
    "benchmark_scaling",
    "build_multiproduct_circuit",
    "build_hamiltonian_qsvt_circuit",
    "build_trotter_circuit",
    "choose_parameters",
    "compare_with_exact",
    "create_benchmark_figure",
    "count_circuit_resources",
    "estimate_qsvt_degree",
    "estimate_suzuki_error",
    "estimate_resources_analytically",
    "generate_and_save_benchmark",
    "generate_benchmark_sweep",
    "heisenberg_chain",
    "load_benchmark_config",
    "load_benchmark_data",
    "multiproduct_coefficients",
    "optimal_mpf_exponents",
    "plot_saved_benchmark",
    "suzuki_commutator_bounds",
    "save_benchmark_data",
    "synthesize_hamsim_phases",
    "transverse_field_ising",
    "validate_benchmark_frame",
]
