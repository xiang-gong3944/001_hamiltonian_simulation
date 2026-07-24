"""Diagnostic fourth-order commutator bounds for one shared Suzuki formula."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import math
from typing import Literal, TypeAlias

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .hamiltonians import PauliHamiltonian
from .trotter import (
    TrotterPartition,
    _nested_commutator_basis,
    _pauli_l1,
    _resolve_suzuki_specification,
    _simplify,
    _suzuki_group_factors,
    _theorem_word_weights,
)


FourthOrderNormMethod: TypeAlias = Literal["pauli-l1", "spectral"]
FourthOrderBoundStatus: TypeAlias = Literal["ok", "unsupported"]

_ORDER = 4
_TIME_POWER = 5
_SUZUKI_STAGE_COUNT = 10
_FACTORIAL_FIVE = math.factorial(_TIME_POWER)


@dataclass(frozen=True)
class FourthOrderBoundProblem:
    r"""One ordered fourth-order formula shared by every bound evaluator.

    Factor tuples use Schubert--Mendl's right-to-left indexing convention:
    entry zero is :math:`A_1`, so the implemented product is
    :math:`\exp(-itA_K)\cdots\exp(-itA_1)`.
    """

    hamiltonian: PauliHamiltonian
    partition: str
    groups: tuple[SparsePauliOp, ...]
    group_sizes: tuple[int, ...]
    raw_factors: tuple[tuple[int, float], ...]
    ordered_exponentials: tuple[tuple[int, float], ...]
    merged_consecutive: bool
    order: int = _ORDER
    time_power: int = _TIME_POWER
    stage_count: int = _SUZUKI_STAGE_COUNT
    z1: float = 1 / (4 - 4 ** (1 / 3))
    z0: float = 1 - 4 / (4 - 4 ** (1 / 3))

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def exponential_count(self) -> int:
        return len(self.ordered_exponentials)

    @property
    def centered_index(self) -> int:
        return math.ceil(self.exponential_count / 2)

    @property
    def product_formula_coefficients(self) -> tuple[float, float]:
        return self.z1, self.z0


@dataclass(frozen=True)
class NestedCommutatorContribution:
    """One weighted norm in a fourth-order one-step coefficient.

    ``hamiltonian_indices`` lists the four adjoint operators followed by the
    base operator, from outermost to innermost. For an unexpanded
    Schubert--Mendl :math:`B_j`, it lists only the four adjoint operators and
    ``base_coefficients`` records the linear combination used as the base.
    Indices are zero-based Python indices.
    """

    hamiltonian_indices: tuple[int, ...]
    base_coefficients: tuple[tuple[int, float], ...]
    prefactor: float
    commutator_norm: float
    weighted_value: float
    source_term_count: int = 1


@dataclass(frozen=True)
class FourthOrderBoundResult:
    """A one-step coefficient and complete comparison diagnostics."""

    bound_family: str
    specific_result: str
    one_step_coefficient: float | None
    time_power: int
    contributions: tuple[NestedCommutatorContribution, ...]
    problem: FourthOrderBoundProblem
    center_index: int | None
    norm_method: FourthOrderNormMethod
    triangle_inequalities: tuple[str, ...]
    additional_relaxations: tuple[str, ...]
    rigorous: bool
    status: FourthOrderBoundStatus
    diagnostic_message: str

    @property
    def ordered_exponentials(self) -> tuple[tuple[int, float], ...]:
        """Return the exact factor tuple owned by the shared problem object."""
        return self.problem.ordered_exponentials

    @property
    def merged_consecutive(self) -> bool:
        return self.problem.merged_consecutive

    @property
    def centered_index(self) -> int:
        return self.problem.centered_index

    def local_error_bound(self, delta: float) -> float:
        r"""Return :math:`C_5|\delta|^5` for a supported result."""
        coefficient = _require_coefficient(self)
        if not np.isfinite(delta):
            raise ValueError("delta must be finite")
        return coefficient * abs(float(delta)) ** self.time_power

    def accumulated_error_bound(self, time: float, segments: int) -> float:
        """Return the unitary-telescoping bound :math:`C_5|t|^5/r^4`."""
        return accumulated_fourth_order_error(self, time, segments)

    def required_segments(self, time: float, target_error: float) -> int:
        """Return the least integer segment count certified by this bound."""
        return required_fourth_order_segments(self, time, target_error)


# Exact double-precision coefficients obtained from Appendix M's factorization.
# They deliberately do not use the Schubert--Mendl coefficient generator, so
# the two-term termwise equality test can catch convention and center errors.
_CHILDS_M13_COEFFICIENTS: dict[tuple[int, ...], float] = {
    (0, 0, 0, 1, 0): 0.0047013343101698826,
    (0, 0, 1, 1, 0): 0.00570381876262935,
    (0, 1, 0, 1, 0): 0.004638910081589791,
    (0, 1, 1, 1, 0): 0.00737205664595221,
    (1, 0, 0, 1, 0): 0.00968966829012217,
    (1, 0, 1, 1, 0): 0.009726162358456834,
    (1, 1, 0, 1, 0): 0.01732815305710953,
    (1, 1, 1, 1, 0): 0.0283734344054259,
}


def build_fourth_order_bound_problem(
    hamiltonian: PauliHamiltonian,
    *,
    partition: TrotterPartition = "auto",
    merge_adjacent: bool = True,
) -> FourthOrderBoundProblem:
    """Construct the fourth-order formula once for all comparison bounds."""
    specification = _resolve_suzuki_specification(hamiltonian, _ORDER, partition)
    raw_factors = _suzuki_group_factors(len(specification.groups), _ORDER)
    factors = _suzuki_group_factors(
        len(specification.groups),
        _ORDER,
        merge_adjacent=merge_adjacent,
    )
    return FourthOrderBoundProblem(
        hamiltonian=hamiltonian,
        partition=specification.partition,
        groups=specification.groups,
        group_sizes=specification.group_sizes,
        raw_factors=raw_factors,
        ordered_exponentials=factors,
        merged_consecutive=merge_adjacent,
    )


def _validate_norm_method(method: FourthOrderNormMethod) -> None:
    if method not in ("pauli-l1", "spectral"):
        raise ValueError("norm_method must be 'pauli-l1' or 'spectral'")


def _operator_norm(
    operator: SparsePauliOp,
    method: FourthOrderNormMethod,
) -> float:
    if method == "pauli-l1":
        return _pauli_l1(operator)
    return float(np.linalg.norm(operator.to_matrix(), 2))


def _norm_relaxation(method: FourthOrderNormMethod) -> tuple[str, ...]:
    if method == "pauli-l1":
        return (
            "each spectral norm is upper-bounded by the SparsePauliOp coefficient 1-norm",
        )
    return ()


def _nested_operator(
    groups: tuple[SparsePauliOp, ...],
    indices: tuple[int, ...],
) -> SparsePauliOp:
    operator = groups[indices[-1]]
    for outer_index in reversed(indices[:-1]):
        outer = groups[outer_index]
        operator = _simplify(outer @ operator - operator @ outer)
    return operator


def _canonicalize_innermost_pair(indices: tuple[int, ...]) -> tuple[int, ...]:
    """Canonicalize ``[Hi, Hj]`` by putting the larger index first.

    Swapping the innermost pair changes only a sign, which disappears under
    the norm. This is the convention used by Childs Appendix M and Table II.
    """
    inner, base = indices[-2:]
    if inner < base:
        return indices[:-2] + (base, inner)
    return indices


def _expanded_theorem_coefficients(
    problem: FourthOrderBoundProblem,
    center: int,
) -> dict[tuple[int, ...], tuple[float, int]]:
    """Expand every theorem base ``B_j`` and collect canonical coefficients."""
    aggregate: defaultdict[tuple[int, ...], list[float | int]] = defaultdict(
        lambda: [0.0, 0]
    )
    entries = _theorem_word_weights(
        problem.ordered_exponentials,
        _ORDER,
        center=center,
    )
    for prefix, weights in entries:
        for word, weight in weights.items():
            for base, base_coefficient in enumerate(prefix):
                if base_coefficient == 0 or word[0] == base:
                    continue
                indices = _canonicalize_innermost_pair(
                    tuple(reversed(word)) + (base,)
                )
                aggregate[indices][0] += (
                    weight * abs(base_coefficient) / _FACTORIAL_FIVE
                )
                aggregate[indices][1] += 1
    return {
        indices: (float(values[0]), int(values[1]))
        for indices, values in aggregate.items()
    }


def _contributions_from_coefficients(
    problem: FourthOrderBoundProblem,
    coefficients: dict[tuple[int, ...], float]
    | dict[tuple[int, ...], tuple[float, int]],
    norm_method: FourthOrderNormMethod,
) -> tuple[NestedCommutatorContribution, ...]:
    contributions: list[NestedCommutatorContribution] = []
    for indices in sorted(coefficients):
        item = coefficients[indices]
        if isinstance(item, tuple):
            prefactor, source_count = item
        else:
            prefactor, source_count = item, 1
        norm = _operator_norm(_nested_operator(problem.groups, indices), norm_method)
        contributions.append(
            NestedCommutatorContribution(
                hamiltonian_indices=indices,
                base_coefficients=((indices[-1], 1.0),),
                prefactor=float(prefactor),
                commutator_norm=norm,
                weighted_value=float(prefactor) * norm,
                source_term_count=source_count,
            )
        )
    return tuple(contributions)


def _rounded_up_sum(contributions: tuple[NestedCommutatorContribution, ...]) -> float:
    value = math.fsum(contribution.weighted_value for contribution in contributions)
    return float(np.nextafter(value, np.inf)) if value else 0.0


def childs_general_commutator_bound(
    problem: FourthOrderBoundProblem,
    *,
    norm_method: FourthOrderNormMethod = "pauli-l1",
) -> FourthOrderBoundResult:
    """Evaluate a concrete relaxation of the general Childs bound.

    The published Theorem 6 (Theorem 11 in the enhanced arXiv version) states
    only ``O(alpha_comm t**5)``. This evaluator uses the explicit
    anti-Hermitian proof relaxation in enhanced-arXiv Eqs. (189) and (191),
    yielding ``2 * Upsilon**5 * alpha_comm / 5!`` for this formula.
    """
    _validate_norm_method(norm_method)
    prefactor = 2 * problem.stage_count**_TIME_POWER / _FACTORIAL_FIVE
    basis = _nested_commutator_basis(problem.groups, _ORDER)
    contributions: list[NestedCommutatorContribution] = []
    for indices in product(range(problem.group_count), repeat=_TIME_POWER):
        word = tuple(reversed(indices[:-1]))
        base = indices[-1]
        norm = _operator_norm(basis[(word, base)], norm_method)
        contributions.append(
            NestedCommutatorContribution(
                hamiltonian_indices=indices,
                base_coefficients=((base, 1.0),),
                prefactor=prefactor,
                commutator_norm=norm,
                weighted_value=prefactor * norm,
            )
        )
    frozen = tuple(contributions)
    return FourthOrderBoundResult(
        bound_family="childs-general-commutator",
        specific_result=(
            "Childs et al. Theorem 6/11 with explicit anti-Hermitian "
            "proof relaxation, enhanced-arXiv Eqs. (189) and (191)"
        ),
        one_step_coefficient=_rounded_up_sum(frozen),
        time_power=_TIME_POWER,
        contributions=frozen,
        problem=problem,
        center_index=None,
        norm_method=norm_method,
        triangle_inequalities=(
            "sum of the norms of all fifth-degree nested commutators",
        ),
        additional_relaxations=(
            "the theorem's unspecified big-O constant is instantiated by its proof",
            "all product coefficients use |a_(v,gamma)| <= 1",
            "stage positions and term ordering are relaxed to Upsilon**4 copies of every word",
        )
        + _norm_relaxation(norm_method),
        rigorous=True,
        status="ok",
        diagnostic_message=(
            "Concrete proof-level relaxation; it is intentionally looser than "
            "the Appendix-M and Schubert--Mendl small-prefactor constructions."
        ),
    )


def childs_fourth_order_small_prefactor_bound(
    problem: FourthOrderBoundProblem,
    *,
    norm_method: FourthOrderNormMethod = "pauli-l1",
) -> FourthOrderBoundResult:
    """Evaluate Childs Appendix M for its supported two/three-term cases."""
    _validate_norm_method(norm_method)
    if problem.group_count not in (2, 3):
        return FourthOrderBoundResult(
            bound_family="childs-fourth-order-small-prefactor",
            specific_result="Childs et al. Appendix M",
            one_step_coefficient=None,
            time_power=_TIME_POWER,
            contributions=(),
            problem=problem,
            center_index=None,
            norm_method=norm_method,
            triangle_inequalities=(),
            additional_relaxations=(),
            rigorous=False,
            status="unsupported",
            diagnostic_message=(
                "Childs Appendix M provides specialized fourth-order coefficients "
                "only for decompositions with two or three summands."
            ),
        )

    if problem.group_count == 2:
        center = 6
        coefficients: dict[tuple[int, ...], float] | dict[
            tuple[int, ...], tuple[float, int]
        ] = _CHILDS_M13_COEFFICIENTS
        result_name = "Childs et al. Appendix M, Proposition M.1 and Eq. (M13)"
    else:
        center = 10
        coefficients = _expanded_theorem_coefficients(problem, center)
        result_name = "Childs et al. Appendix M, Proposition M.2 and Table II"

    contributions = _contributions_from_coefficients(
        problem,
        coefficients,
        norm_method,
    )
    return FourthOrderBoundResult(
        bound_family="childs-fourth-order-small-prefactor",
        specific_result=result_name,
        one_step_coefficient=_rounded_up_sum(contributions),
        time_power=_TIME_POWER,
        contributions=contributions,
        problem=problem,
        center_index=center,
        norm_method=norm_method,
        triangle_inequalities=(
            "Appendix M expands the accumulated base and bounds canonical commutators separately",
        ),
        additional_relaxations=_norm_relaxation(norm_method),
        rigorous=True,
        status="ok",
        diagnostic_message=(
            "Two-term coefficients are the unrounded Eq. (M13) values; "
            "three-term coefficients use Appendix M's s=10 factorization."
        ),
    )


def _combined_theorem_contributions(
    problem: FourthOrderBoundProblem,
    center: int,
    norm_method: FourthOrderNormMethod,
) -> tuple[NestedCommutatorContribution, ...]:
    basis = _nested_commutator_basis(problem.groups, _ORDER)
    aggregate: defaultdict[
        tuple[tuple[int, ...], tuple[float, ...]], list[float | int]
    ] = defaultdict(lambda: [0.0, 0])
    for prefix, weights in _theorem_word_weights(
        problem.ordered_exponentials,
        _ORDER,
        center=center,
    ):
        for word, weight in weights.items():
            aggregate[(word, prefix)][0] += weight / _FACTORIAL_FIVE
            aggregate[(word, prefix)][1] += 1

    contributions: list[NestedCommutatorContribution] = []
    for (word, prefix), values in sorted(aggregate.items()):
        operator: SparsePauliOp | None = None
        base_coefficients: list[tuple[int, float]] = []
        for base, coefficient in enumerate(prefix):
            if coefficient == 0:
                continue
            term = basis[(word, base)] * coefficient
            operator = term if operator is None else operator + term
            base_coefficients.append((base, coefficient))
        if operator is None:
            continue
        prefactor = float(values[0])
        norm = _operator_norm(_simplify(operator), norm_method)
        contributions.append(
            NestedCommutatorContribution(
                hamiltonian_indices=tuple(reversed(word)),
                base_coefficients=tuple(base_coefficients),
                prefactor=prefactor,
                commutator_norm=norm,
                weighted_value=prefactor * norm,
                source_term_count=int(values[1]),
            )
        )
    return tuple(contributions)


def schubert_mendl_small_prefactor_bound(
    problem: FourthOrderBoundProblem,
    *,
    center: int | None = None,
    norm_method: FourthOrderNormMethod = "pauli-l1",
    expand_base_triangle: bool = True,
) -> FourthOrderBoundResult:
    """Evaluate Schubert--Mendl Theorem 1 for one chosen center ``s``."""
    _validate_norm_method(norm_method)
    if center is None:
        center = problem.centered_index
    if isinstance(center, bool) or not isinstance(center, int):
        raise TypeError("center must be an integer")
    if not 1 <= center <= problem.exponential_count:
        raise ValueError("center must lie in [1, K]")

    if expand_base_triangle:
        coefficients = _expanded_theorem_coefficients(problem, center)
        contributions = _contributions_from_coefficients(
            problem,
            coefficients,
            norm_method,
        )
        triangles = (
            "B_j is expanded into Hamiltonian summands and bounded term by term",
            "innermost commutators are canonicalized up to sign before aggregation",
        )
    else:
        contributions = _combined_theorem_contributions(
            problem,
            center,
            norm_method,
        )
        triangles = (
            "triangle inequality over the two theorem sums and their weak compositions",
        )

    return FourthOrderBoundResult(
        bound_family="schubert-mendl-small-prefactor",
        specific_result="Schubert and Mendl, Theorem 1 and Eqs. (6)--(10)",
        one_step_coefficient=_rounded_up_sum(contributions),
        time_power=_TIME_POWER,
        contributions=contributions,
        problem=problem,
        center_index=center,
        norm_method=norm_method,
        triangle_inequalities=triangles,
        additional_relaxations=_norm_relaxation(norm_method),
        rigorous=True,
        status="ok",
        diagnostic_message=(
            "Centered choice" if center == problem.centered_index else "Non-centered choice"
        )
        + (
            "; B_j expanded for termwise Appendix-M comparison."
            if expand_base_triangle
            else "; B_j retained as a linear combination before norm evaluation."
        ),
    )


def all_schubert_mendl_centers(
    problem: FourthOrderBoundProblem,
    *,
    norm_method: FourthOrderNormMethod = "pauli-l1",
    expand_base_triangle: bool = True,
) -> tuple[FourthOrderBoundResult, ...]:
    """Evaluate and retain every valid Schubert--Mendl center index."""
    return tuple(
        schubert_mendl_small_prefactor_bound(
            problem,
            center=center,
            norm_method=norm_method,
            expand_base_triangle=expand_base_triangle,
        )
        for center in range(1, problem.exponential_count + 1)
    )


def minimizing_schubert_mendl_center(
    results: tuple[FourthOrderBoundResult, ...],
) -> FourthOrderBoundResult:
    """Select the minimum coefficient with stable lower-``s`` tie breaking."""
    if not results:
        raise ValueError("results must not be empty")
    problem = results[0].problem
    if any(result.problem is not problem for result in results):
        raise ValueError("all results must reference the same problem object")
    return min(
        results,
        key=lambda result: (
            _require_coefficient(result),
            result.center_index if result.center_index is not None else math.inf,
        ),
    )


def _require_coefficient(result: FourthOrderBoundResult) -> float:
    if result.status != "ok" or result.one_step_coefficient is None:
        raise ValueError(result.diagnostic_message)
    return result.one_step_coefficient


def accumulated_fourth_order_error(
    result: FourthOrderBoundResult,
    time: float,
    segments: int,
) -> float:
    """Return ``C5 * |time|**5 / segments**4`` by unitary telescoping."""
    coefficient = _require_coefficient(result)
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
        raise ValueError("segments must be a positive integer")
    return coefficient * abs(float(time)) ** _TIME_POWER / segments**_ORDER


def required_fourth_order_segments(
    result: FourthOrderBoundResult,
    time: float,
    target_error: float,
) -> int:
    """Invert the accumulated fourth-order bound against ``target_error``."""
    coefficient = _require_coefficient(result)
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if not np.isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be positive and finite")
    if coefficient == 0 or time == 0:
        return 1
    raw = (coefficient * abs(float(time)) ** _TIME_POWER / target_error) ** (
        1 / _ORDER
    )
    segments = max(1, math.ceil(raw))
    while accumulated_fourth_order_error(result, time, segments) > target_error:
        segments += 1
    while (
        segments > 1
        and accumulated_fourth_order_error(result, time, segments - 1) <= target_error
    ):
        segments -= 1
    return segments
