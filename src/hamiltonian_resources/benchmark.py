"""Fixed-error parameter selection and scaling benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from .evaluation import estimate_plan_resources
from .hamiltonians import PauliHamiltonian
from .method_specs import MultiproductMethod, QSVTMethod, TrotterMethod
from .multiproduct import MPFErrorMethod, MPFSchedule, optimal_mpf_exponents
from .planning import plan_simulation
from .qsvt import estimate_qsvt_degree  # noqa: F401 - compatibility monkeypatch target
from .resources import ResourceEstimate
from .trotter import TrotterPartition


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
    mpf_error_method: MPFErrorMethod = "low2019-l1-ideal-rigorous"
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
        if self.mpf_error_method not in (
            "low2019-l1-ideal-rigorous",
            "mizuta2026-commutator-ideal-rigorous",
            "low-rigorous",
            "legacy-w2-proxy",
        ):
            raise ValueError(
                "mpf_error_method must be 'low2019-l1-ideal-rigorous' "
                "(historical alias 'low-rigorous'), "
                "'mizuta2026-commutator-ideal-rigorous', or 'legacy-w2-proxy'"
            )


def choose_parameters(
    hamiltonian: PauliHamiltonian,
    config: _EvaluationConfig,
    algorithm: str | None = None,
) -> dict[str, int]:
    """Choose parameters from error bounds of comparable tightness.

    Orders 1 and 2 use the rigorous Childs et al. commutator bounds.  Orders 4
    and 6 use the rigorous Schubert--Mendl bound when the resolved partition is
    within the practical work cap; other even orders retain the documented
    1-norm proxy.  MPF defaults to the rigorous ideal-operator bound and
    sufficient segment rule of Low, Kliuchnikov, and Wiebe.  The historical
    W2-calibrated rule remains available as ``legacy-w2-proxy`` but is never
    certified.  Neither MPF estimator certifies the complete robust-OAA
    shared-ancilla circuit.  QSVT uses the rigorous Jacobi--Anger
    truncation degree.  Mixing loose 1-norm bounds for product formulas with
    the tight QSVT degree would systematically distort crossovers, which is
    why the product-formula rules are commutator-based.  Calibrate small
    instances with ``compare_with_exact``.
    """
    if algorithm not in (None, "trotter", "multiproduct", "qsvt"):
        raise ValueError(f"unknown algorithm: {algorithm}")
    families = ("trotter", "multiproduct", "qsvt") if algorithm is None else (algorithm,)
    parameters: dict[str, int] = {}
    for family in families:
        plan = _plan_for_algorithm(hamiltonian, config, family)
        if family == "trotter":
            parameters["trotter_reps"] = plan.repetitions  # type: ignore[union-attr]
        elif family == "multiproduct":
            parameters["mpf_segments"] = plan.segments  # type: ignore[union-attr]
        else:
            parameters["qsvt_degree"] = plan.degree  # type: ignore[union-attr]
    return parameters


def _plan_for_algorithm(
    hamiltonian: PauliHamiltonian,
    config: _EvaluationConfig,
    algorithm: str,
):
    if algorithm == "trotter":
        method = TrotterMethod(config.trotter_order)
    elif algorithm == "multiproduct":
        method = MultiproductMethod(
            config.mpf_m,
            schedule=config.mpf_schedule,
            error_method=config.mpf_error_method,
        )
    elif algorithm == "qsvt":
        method = QSVTMethod()
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return plan_simulation(
        hamiltonian,
        method,
        config.time,
        config.target_error,
        synthesis_error_fraction=config.synthesis_error_fraction,
        trotter_partition=config.trotter_partition,
    )


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
    plan = _plan_for_algorithm(hamiltonian, config, algorithm)
    return estimate_plan_resources(plan).resources
