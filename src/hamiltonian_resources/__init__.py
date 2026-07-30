"""Hamiltonian-simulation circuit and resource-comparison toolkit."""

from .benchmark_plotting import (
    FAMILY_COLORS,
    plot_benchmark,
    save_benchmark_plots,
    select_best_by_family,
)
from .benchmark_suite import (
    BENCHMARK_COLUMNS,
    SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkJob,
    BenchmarkProgress,
    HamiltonianSpec,
    MultiproductMethod,
    QSVTMethod,
    TimeScaling,
    TrotterMethod,
    default_methods,
    load_benchmark,
    load_benchmark_job,
    run_benchmark,
    save_benchmark,
    validate_benchmark_frame,
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
    "BENCHMARK_COLUMNS",
    "BenchmarkConfig",
    "BenchmarkJob",
    "BenchmarkProgress",
    "FAMILY_COLORS",
    "HamiltonianSimulationPhases",
    "HamiltonianSpec",
    "MPFSchedule",
    "MultiproductMethod",
    "PauliHamiltonian",
    "QSVTMethod",
    "ResourceEstimate",
    "SCHEMA_VERSION",
    "SuzukiErrorEstimate",
    "TimeScaling",
    "TrotterMethod",
    "TrotterPartition",
    "build_hamiltonian_qsvt_circuit",
    "build_multiproduct_circuit",
    "build_trotter_circuit",
    "compare_with_exact",
    "count_circuit_resources",
    "default_methods",
    "estimate_qsvt_degree",
    "estimate_suzuki_error",
    "heisenberg_chain",
    "load_benchmark",
    "load_benchmark_job",
    "multiproduct_coefficients",
    "optimal_mpf_exponents",
    "plot_benchmark",
    "run_benchmark",
    "save_benchmark",
    "save_benchmark_plots",
    "select_best_by_family",
    "suzuki_commutator_bounds",
    "synthesize_hamsim_phases",
    "transverse_field_ising",
    "validate_benchmark_frame",
]
