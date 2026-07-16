"""Fixed-error parameter selection and scaling benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .hamiltonians import PauliHamiltonian
from .multiproduct import (
    build_multiproduct_circuit,
    optimal_mpf_exponents,
)
from .qsvt import build_hamiltonian_qsvt_circuit, estimate_qsvt_degree
from .resources import ResourceEstimate, count_circuit_resources, t_cost_for_z_rotation
from .trotter import build_trotter_circuit


@dataclass(frozen=True)
class BenchmarkConfig:
    time: float = 1.0
    target_error: float = 1e-3
    synthesis_error_fraction: float = 0.1
    trotter_order: int = 2
    mpf_m: int = 3
    optimization_level: int = 1

    def __post_init__(self) -> None:
        if self.time <= 0 or not 0 < self.target_error < 1:
            raise ValueError("time must be positive and target_error must lie in (0, 1)")
        if not 0 < self.synthesis_error_fraction < 1:
            raise ValueError("synthesis_error_fraction must lie in (0, 1)")
        optimal_mpf_exponents(self.mpf_m)


def choose_parameters(hamiltonian: PauliHamiltonian, config: BenchmarkConfig) -> dict[str, int]:
    """Choose comparable parameters from explicit asymptotic error proxies.

    These are conservative sizing rules, not certified instance-specific error
    bounds. Small instances should be calibrated with ``compare_with_exact``.
    Product formulas use error proxy (alpha*t)^(p+1)/r^p; an order-2m MPF uses
    (alpha*t)^(2m+1)/r^(2m). QSVT uses Jacobi-Anger query scaling.
    """
    budget = config.target_error * (1 - config.synthesis_error_fraction)
    alpha_time = hamiltonian.alpha * config.time
    p = config.trotter_order
    trotter_reps = max(1, math.ceil(((alpha_time ** (p + 1)) / budget) ** (1 / p)))
    mpf_order = 2 * config.mpf_m
    mpf_segments = max(
        1, math.ceil(((alpha_time ** (mpf_order + 1)) / budget) ** (1 / mpf_order))
    )
    return {
        "trotter_reps": trotter_reps,
        "mpf_segments": mpf_segments,
        "qsvt_degree": estimate_qsvt_degree(alpha_time, budget),
    }


def _suzuki_term_occurrences(term_count: int, reps: int, order: int) -> int:
    if order == 1:
        return term_count * reps
    return (2 * term_count - 1) * (5 ** (order // 2 - 1)) * reps


def estimate_resources_analytically(
    hamiltonian: PauliHamiltonian,
    config: BenchmarkConfig,
    algorithm: str,
) -> ResourceEstimate:
    """Estimate resources without constructing the potentially huge circuit.

    Pauli-rotation ladders are counted directly. Multi-controlled LCU gates use
    explicit upper-bound-style decomposition models, making this suitable for
    scaling comparisons rather than hardware-specific compilation claims.
    """
    params = choose_parameters(hamiltonian, config)
    mpf_exponents = optimal_mpf_exponents(config.mpf_m)
    weights = [sum(ch != "I" for ch in label) for label, _ in hamiltonian.terms]
    mean_ladder_cx = sum(2 * max(0, w - 1) for w in weights) / len(weights)
    synth_error = config.target_error * config.synthesis_error_fraction

    if algorithm == "trotter":
        rotations = _suzuki_term_occurrences(
            hamiltonian.term_count, params["trotter_reps"], config.trotter_order
        )
        cnots = math.ceil(rotations * mean_ladder_cx)
        qubits = hamiltonian.num_qubits
    elif algorithm == "multiproduct":
        branch_bits = max(1, math.ceil(math.log2(config.mpf_m)))
        base_rotations = sum(
            _suzuki_term_occurrences(
                hamiltonian.term_count, k * params["mpf_segments"], 2
            )
            for k in mpf_exponents
        )
        rotations = base_rotations * (2**branch_bits) + 2 * (2**branch_bits - 1)
        cnots = math.ceil(
            base_rotations * mean_ladder_cx * (4 * branch_bits + 1)
            + base_rotations * (2 ** (branch_bits + 1))
            + 2 * max(0, 2**branch_bits - 2)
        )
        qubits = hamiltonian.num_qubits + branch_bits
    elif algorithm == "qsvt":
        index_bits = max(1, math.ceil(math.log2(hamiltonian.term_count)))
        degree = params["qsvt_degree"]
        # Two definite-parity sequences (cosine and sine), plus an LCU qubit.
        prepare_calls = 2 * (4 * degree + 2)
        prepare_rotations = max(1, 2**index_bits - 1)
        select_cx = sum(max(1, 8 * index_bits - 6) * w for w in weights)
        projector_cx = max(0, 2 ** (index_bits + 1) - 4)
        rotations = prepare_calls * prepare_rotations + 4 * degree + 2
        cnots = (
            prepare_calls * max(0, 2**index_bits - 2)
            + 2 * degree * select_cx
            + 4 * degree * projector_cx
        )
        qubits = hamiltonian.num_qubits + index_bits + 1
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    per_rotation = synth_error / max(1, rotations)
    t_count = rotations * t_cost_for_z_rotation(0.17320508075688773, per_rotation)
    return ResourceEstimate(
        algorithm=algorithm,
        num_qubits=qubits,
        cnot_count=int(cnots),
        t_count=int(t_count),
        rotation_count=int(rotations),
        depth=-1,
        counting_mode="analytical-model",
        rotation_synthesis_error=synth_error,
    )


def benchmark_scaling(
    sizes: list[int],
    hamiltonian_factory: Callable[[int], PauliHamiltonian],
    config: BenchmarkConfig = BenchmarkConfig(),
    *,
    transpile_circuits: bool = False,
) -> pd.DataFrame:
    """Count all algorithms at each system size under one error budget.

    The default analytical model does not allocate large circuits. Its QSVT
    and MPF formulas are legacy structural estimates and do not yet include
    the quadrature-extraction or per-segment robust-OAA constants of the
    concrete circuits.
    Set ``transpile_circuits=True`` to synthesize real QSP phases and compile
    the complete circuit for small systems.
    """
    records: list[dict[str, int | float | str]] = []
    synthesis_error = config.target_error * config.synthesis_error_fraction
    for size in sizes:
        hamiltonian = hamiltonian_factory(size)
        parameters = choose_parameters(hamiltonian, config)
        circuits = None
        if transpile_circuits:
            circuits = {
                "trotter": build_trotter_circuit(
                    hamiltonian,
                    config.time,
                    parameters["trotter_reps"],
                    config.trotter_order,
                ),
                "multiproduct": build_multiproduct_circuit(
                    hamiltonian,
                    config.time,
                    config.mpf_m,
                    parameters["mpf_segments"],
                ),
                "qsvt": build_hamiltonian_qsvt_circuit(
                    hamiltonian,
                    config.time,
                    config.target_error * (1 - config.synthesis_error_fraction),
                ),
            }
        for algorithm in ("trotter", "multiproduct", "qsvt"):
            if circuits is None:
                resource = estimate_resources_analytically(hamiltonian, config, algorithm)
            else:
                resource = count_circuit_resources(
                    circuits[algorithm],
                    algorithm=algorithm,
                    total_synthesis_error=synthesis_error,
                    optimization_level=config.optimization_level,
                )
            estimate = resource.as_dict()
            if algorithm == "multiproduct":
                lcu_scale = 2.0
                nominal_success_probability = 1.0
            elif algorithm == "qsvt":
                lcu_scale = 2.0
                nominal_success_probability = 0.25
            else:
                lcu_scale = 1.0
                nominal_success_probability = 1.0
            estimate.update(
                system_size=size,
                alpha=hamiltonian.alpha,
                target_error=config.target_error,
                lcu_scale=lcu_scale,
                nominal_success_probability=nominal_success_probability,
                parameter=(
                    parameters["trotter_reps"]
                    if algorithm == "trotter"
                    else parameters["mpf_segments"]
                    if algorithm == "multiproduct"
                    else parameters["qsvt_degree"]
                ),
            )
            records.append(estimate)
    return pd.DataFrame.from_records(records)
