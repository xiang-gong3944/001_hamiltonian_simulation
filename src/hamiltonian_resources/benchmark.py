"""Fixed-error parameter selection and scaling benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hamiltonians import PauliHamiltonian
from .multiproduct import (
    MPFSchedule,
    legacy_w2_proxy_segments,
    optimal_mpf_exponents,
)
from .qsvt import estimate_qsvt_degree
from .resources import (
    T_PER_AND,
    ResourceEstimate,
    multicontrol_and_pairs,
    t_cost_for_z_rotation,
)
from .trotter import (
    TrotterPartition,
    _suzuki_term_occurrences,
    estimate_suzuki_error,
)


@dataclass(frozen=True)
class _EvaluationConfig:
    """Validated inputs for one method at one benchmark point."""

    time: float = 1.0
    target_error: float = 1e-3
    synthesis_error_fraction: float = 0.1
    trotter_order: int = 2
    trotter_partition: TrotterPartition = "auto"
    mpf_m: int = 3
    mpf_schedule: MPFSchedule = "new"
    optimization_level: int = 1

    def __post_init__(self) -> None:
        if self.time <= 0 or not 0 < self.target_error < 1:
            raise ValueError("time must be positive and target_error must lie in (0, 1)")
        if not 0 < self.synthesis_error_fraction < 1:
            raise ValueError("synthesis_error_fraction must lie in (0, 1)")
        if self.trotter_order != 1 and (
            self.trotter_order < 2 or self.trotter_order % 2
        ):
            raise ValueError("trotter_order must be 1 or a positive even integer")
        if self.trotter_partition not in ("auto", "individual", "commuting"):
            raise ValueError(
                "trotter_partition must be 'auto', 'individual', or 'commuting'"
            )
        optimal_mpf_exponents(self.mpf_m, schedule=self.mpf_schedule)


def choose_parameters(
    hamiltonian: PauliHamiltonian,
    config: _EvaluationConfig,
    algorithm: str | None = None,
) -> dict[str, int]:
    """Choose parameters from error bounds of comparable tightness.

    Orders 1 and 2 use the rigorous Childs et al. commutator bounds.  Orders 4
    and 6 use the rigorous Schubert--Mendl bound when the resolved partition is
    within the practical work cap; other even orders retain the documented
    1-norm proxy.  An order-2m MPF currently uses the explicitly legacy
    ``legacy-w2-proxy`` rule
    (alpha_eff*t)^(2m+1)/r^(2m) with
    alpha_eff = min(alpha, W2^(1/3)), which reproduces the certified order-2
    rate but extrapolates the higher-order constants; it is a documented
    heuristic, not a certified bound.  QSVT uses the rigorous Jacobi--Anger
    truncation degree.  Mixing loose 1-norm bounds for product formulas with
    the tight QSVT degree would systematically distort crossovers, which is
    why the product-formula rules are commutator-based.  Calibrate small
    instances with ``compare_with_exact``.
    """
    if algorithm not in (None, "trotter", "multiproduct", "qsvt"):
        raise ValueError(f"unknown algorithm: {algorithm}")
    budget = config.target_error * (1 - config.synthesis_error_fraction)
    time = config.time
    alpha_time = hamiltonian.alpha * time
    parameters: dict[str, int] = {}
    if algorithm in (None, "trotter"):
        p = config.trotter_order
        one_step_error = estimate_suzuki_error(
            hamiltonian,
            time,
            reps=1,
            order=p,
            partition=config.trotter_partition,
        ).error
        trotter_reps = math.ceil((one_step_error / budget) ** (1 / p))
        parameters["trotter_reps"] = max(1, trotter_reps)
    if algorithm in (None, "multiproduct"):
        parameters["mpf_segments"] = legacy_w2_proxy_segments(
            hamiltonian,
            time,
            budget,
            config.mpf_m,
        )
    if algorithm in (None, "qsvt"):
        parameters["qsvt_degree"] = estimate_qsvt_degree(alpha_time, budget)
    return parameters


#: CX cost charged per temporary-AND compute/uncompute pair.
_CX_PER_AND = 6


def estimate_resources_analytically(
    hamiltonian: PauliHamiltonian,
    config: _EvaluationConfig,
    algorithm: str,
) -> ResourceEstimate:
    """Estimate resources without constructing the potentially huge circuit.

    The models mirror the structure of the concrete circuits, including the
    per-segment robust-OAA factor of three for MPF and QSVT, the identity
    padding branches of the MPF LCU, and the cosine/sine quadrature circuits
    of QSVT.  Multi-controlled gates are compiled through temporary-AND
    ladders (``T_PER_AND`` T and ``_CX_PER_AND`` CX per ancilla pair), so
    Toffoli-type T costs are counted, unlike a rotation-only model.  Controlled
    QSVT responses assume the efficient compilation in which V and V^dagger
    share their block-encoding queries and only projector phases are selected
    on the quadrature/component qubits; the Qiskit ``.control()`` construction
    used by ``transpile_circuits=True`` is substantially more expensive.
    """
    params = choose_parameters(hamiltonian, config, algorithm)
    mpf_exponents = optimal_mpf_exponents(
        config.mpf_m,
        schedule=config.mpf_schedule,
    )
    weights = [sum(ch != "I" for ch in label) for label, _ in hamiltonian.terms]
    mean_ladder_cx = sum(2 * max(0, w - 1) for w in weights) / len(weights)
    synth_error = config.target_error * config.synthesis_error_fraction
    term_count = hamiltonian.term_count

    if algorithm == "trotter":
        rotations = _suzuki_term_occurrences(
            hamiltonian,
            params["trotter_reps"],
            config.trotter_order,
            config.trotter_partition,
        )
        and_pairs = 0
        cnots = math.ceil(rotations * mean_ladder_cx)
        qubits = hamiltonian.num_qubits
    elif algorithm == "multiproduct":
        segments = params["mpf_segments"]
        branches = len(mpf_exponents) + 2  # two cancelling identity branches
        branch_bits = max(1, math.ceil(math.log2(branches)))
        flag_pairs = multicontrol_and_pairs(branch_bits)
        # One robust-OAA round per segment: 3 SELECT, 6 PREPARE, 2 reflections.
        select_rotations = sum(
            _suzuki_term_occurrences(hamiltonian, k, 2, "individual")
            for k in mpf_exponents
        )
        prepare_rotations = 2**branch_bits - 1
        # Branch flags reduce every S2 rotation to one singly-controlled Rz
        # (two rotations); signs are one multi-controlled phase per branch.
        rotations = segments * (3 * 2 * select_rotations + 6 * prepare_rotations)
        and_pairs = segments * (
            3 * branches * flag_pairs  # branch flag per SELECT
            + 3 * branches * flag_pairs  # coefficient/padding sign phases
            + 2 * flag_pairs  # good-subspace reflections
        )
        cnots = math.ceil(
            segments
            * (
                3 * select_rotations * (mean_ladder_cx + 2)
                + 6 * max(0, 2**branch_bits - 2)
            )
            + and_pairs * _CX_PER_AND
        )
        qubits = hamiltonian.num_qubits + branch_bits + max(1, flag_pairs)
    elif algorithm == "qsvt":
        index_bits = max(1, math.ceil(math.log2(term_count)))
        sine_degree = params["qsvt_degree"]
        cosine_degree = sine_degree - 1
        # Robust OAA applies the cosine/sine LCU three times.  Within each
        # component the controlled V/V^dagger pair shares its block-encoding
        # queries, so queries = 3 * (d_cos + d_sin).
        queries = 3 * (cosine_degree + sine_degree)
        prepare_calls = 2 * queries
        prepare_rotations = max(1, 2**index_bits - 1)
        phase_slots = 3 * ((cosine_degree + 1) + (sine_degree + 1))
        # Each slot selects between phi_i and -phi_(d-i) on the quadrature
        # qubit: two controlled projector phases.
        rotations = prepare_calls * prepare_rotations + 2 * phase_slots
        select_pairs = multicontrol_and_pairs(index_bits + 1)  # + component ctrl
        phase_pairs = multicontrol_and_pairs(index_bits + 2)
        and_pairs = (
            queries * term_count * select_pairs
            + 2 * phase_slots * phase_pairs
            + 2 * phase_pairs  # OAA reflections
        )
        cnots = (
            prepare_calls * max(0, 2**index_bits - 2)
            + queries * sum(weights)  # flag-controlled Pauli applications
            + and_pairs * _CX_PER_AND
        )
        qubits = hamiltonian.num_qubits + index_bits + 2 + select_pairs
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    per_rotation = synth_error / max(1, rotations)
    t_count = rotations * t_cost_for_z_rotation(0.17320508075688773, per_rotation)
    t_count += and_pairs * T_PER_AND
    return ResourceEstimate(
        algorithm=algorithm,
        num_qubits=qubits,
        cnot_count=int(cnots),
        t_count=int(t_count),
        rotation_count=int(rotations),
        depth=-1,
        counting_mode="analytical-model",
        rotation_synthesis_error=synth_error,
        toffoli_count=int(and_pairs),
    )
