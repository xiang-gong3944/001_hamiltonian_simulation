"""Circuit-level Lie/Suzuki product-formula simulation and error bounds."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import math
from collections import defaultdict
from typing import Literal, TypeAlias

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from .hamiltonians import PauliHamiltonian


TrotterPartition: TypeAlias = Literal["auto", "individual", "commuting"]
_ResolvedTrotterPartition: TypeAlias = Literal["individual", "commuting"]
SuzukiErrorMethod: TypeAlias = Literal[
    "childs-commutator",
    "schubert-mendl-commutator",
    "commuting-exact",
    "alpha-proxy",
]

_MAX_HIGHER_ORDER_COMMUTATORS = 4096
_MAX_EXACT_PAULI_COMMUTATOR_TRANSITIONS = 250_000


@dataclass(frozen=True)
class SuzukiErrorEstimate:
    """A product-formula error estimate and its certification metadata.

    ``error`` is a rigorous operator-norm upper bound exactly when ``rigorous``
    is true.  Otherwise it is the documented coefficient-1-norm proxy retained
    for formulas outside the practical commutator-evaluation regime.
    """

    error: float
    prefactor: float
    time: float
    reps: int
    order: int
    partition: _ResolvedTrotterPartition
    group_count: int
    method: SuzukiErrorMethod
    rigorous: bool


@dataclass(frozen=True)
class PauliNestedCommutatorBounds:
    """Exact or rigorously upper-bounded ``alpha_com,q`` values.

    ``values[q - 2]`` bounds the sum of operator norms of all ordered
    ``q``-term nested commutators of the individual Pauli summands. Exact
    evaluation aggregates Pauli words with nonnegative norm weights, so it
    does not construct dense Hamiltonian matrices or discard cancellations.
    If the transition cap is reached, the remaining entries use Mizuta 2026
    Eq. (8), ``(q-1)! (2 k g)^(q-1) N g``.
    """

    values: tuple[float, ...]
    max_order: int
    max_exact_order: int
    state_counts: tuple[int, ...]
    used_locality_fallback: bool
    fallback_reason: str | None
    locality_k: int
    extensiveness_g: float
    decomposition: str = "ordered individual Pauli terms"

    def at(self, order: int) -> float:
        if not 2 <= order <= self.max_order:
            raise ValueError(f"order must lie between 2 and {self.max_order}")
        return self.values[order - 2]


@dataclass(frozen=True)
class _SuzukiPrefactor:
    value: float
    partition: _ResolvedTrotterPartition
    group_count: int
    method: SuzukiErrorMethod
    rigorous: bool


@dataclass(frozen=True)
class _SuzukiSpecification:
    """Resolved summands used consistently by synthesis and error analysis."""

    partition: _ResolvedTrotterPartition
    groups: tuple[SparsePauliOp, ...]

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(int(group.size) for group in self.groups)


def _simplify(operator: SparsePauliOp) -> SparsePauliOp:
    """Combine equal Paulis without discarding small nonzero coefficients."""
    return operator.simplify(atol=0.0, rtol=0.0)


def _pauli_l1(operator: SparsePauliOp) -> float:
    return float(np.sum(np.abs(_simplify(operator).coeffs)))


def _commutator(a: SparsePauliOp, b: SparsePauliOp) -> SparsePauliOp:
    return _simplify(a @ b - b @ a)


def _encoded_pauli(label: str) -> tuple[int, int]:
    """Encode a Pauli word as binary symplectic X/Z masks."""
    x_mask = 0
    z_mask = 0
    for bit, symbol in enumerate(reversed(label)):
        if symbol in "XY":
            x_mask |= 1 << bit
        if symbol in "YZ":
            z_mask |= 1 << bit
    return x_mask, z_mask


def _paulis_anticommute(left: tuple[int, int], right: tuple[int, int]) -> bool:
    left_x, left_z = left
    right_x, right_z = right
    parity_mask = (left_x & right_z) ^ (left_z & right_x)
    return bool(parity_mask.bit_count() & 1)


def _locality_parameters(
    hamiltonian: PauliHamiltonian,
) -> tuple[int, float]:
    """Return the exact Pauli support size ``k`` and extensiveness ``g``."""
    site_weights = [0.0] * hamiltonian.num_qubits
    locality_k = 0
    for label, coefficient in hamiltonian.terms:
        support = [site for site, symbol in enumerate(reversed(label)) if symbol != "I"]
        locality_k = max(locality_k, len(support))
        for site in support:
            site_weights[site] = float(
                np.nextafter(site_weights[site] + abs(coefficient), np.inf)
            )
    return locality_k, max(site_weights, default=0.0)


def _locality_commutator_bound(
    order: int,
    num_qubits: int,
    locality_k: int,
    extensiveness_g: float,
) -> float:
    """Evaluate Mizuta 2026 Eq. (8) in the log domain."""
    if locality_k == 0 or extensiveness_g == 0:
        return 0.0
    log_value = (
        math.lgamma(order)
        + (order - 1) * math.log(2 * locality_k * extensiveness_g)
        + math.log(num_qubits * extensiveness_g)
    )
    if log_value > 709:
        return math.inf
    value = math.exp(log_value)
    return float(np.nextafter(value, np.inf))


@lru_cache(maxsize=None)
def pauli_nested_commutator_bounds(
    hamiltonian: PauliHamiltonian,
    max_order: int,
    *,
    transition_cap: int = _MAX_EXACT_PAULI_COMMUTATOR_TRANSITIONS,
) -> PauliNestedCommutatorBounds:
    """Return rigorous ``alpha_com,q`` bounds through ``max_order``.

    For individual Pauli summands, every nonzero nested commutator is a
    scalar times one Pauli word. The recurrence therefore stores total norm
    weight by resultant Pauli word while retaining the sum over all ordered
    term sequences required by Aftab 2024 Eq. (10) and Mizuta 2026 Eq. (8).
    Floating operations are rounded upward with ``nextafter``.
    """
    if isinstance(max_order, bool) or not isinstance(max_order, int):
        raise TypeError("max_order must be an integer")
    if max_order < 2:
        raise ValueError("max_order must be at least 2")
    if isinstance(transition_cap, bool) or not isinstance(transition_cap, int):
        raise TypeError("transition_cap must be an integer")
    if transition_cap < 1:
        raise ValueError("transition_cap must be positive")

    encoded_terms = tuple(
        (_encoded_pauli(label), abs(float(coefficient)))
        for label, coefficient in hamiltonian.terms
        if any(symbol != "I" for symbol in label)
    )
    locality_k, extensiveness_g = _locality_parameters(hamiltonian)
    if not encoded_terms:
        return PauliNestedCommutatorBounds(
            values=(0.0,) * (max_order - 1),
            max_order=max_order,
            max_exact_order=max_order,
            state_counts=(0,) * (max_order - 1),
            used_locality_fallback=False,
            fallback_reason=None,
            locality_k=locality_k,
            extensiveness_g=extensiveness_g,
        )

    current: dict[tuple[int, int], float] = {}
    for pauli, coefficient in encoded_terms:
        current[pauli] = float(
            np.nextafter(current.get(pauli, 0.0) + coefficient, np.inf)
        )

    values: list[float] = []
    state_counts: list[int] = []
    max_exact_order = 1
    fallback_reason: str | None = None
    for order in range(2, max_order + 1):
        transition_count = len(current) * len(encoded_terms)
        if fallback_reason is not None or transition_count > transition_cap:
            if fallback_reason is None:
                fallback_reason = (
                    "exact Pauli recurrence would require "
                    f"{transition_count} transitions at order {order}, above cap "
                    f"{transition_cap}; remaining orders use Mizuta 2026 Eq. (8)"
                )
            values.append(
                _locality_commutator_bound(
                    order,
                    hamiltonian.num_qubits,
                    locality_k,
                    extensiveness_g,
                )
            )
            state_counts.append(len(current))
            continue

        following: dict[tuple[int, int], float] = {}
        for inner_pauli, inner_weight in current.items():
            for outer_pauli, outer_weight in encoded_terms:
                if not _paulis_anticommute(outer_pauli, inner_pauli):
                    continue
                result = (
                    outer_pauli[0] ^ inner_pauli[0],
                    outer_pauli[1] ^ inner_pauli[1],
                )
                contribution = float(
                    np.nextafter(2 * outer_weight * inner_weight, np.inf)
                )
                following[result] = float(
                    np.nextafter(following.get(result, 0.0) + contribution, np.inf)
                )
        current = following
        value = math.fsum(current.values())
        values.append(float(np.nextafter(value, np.inf)) if value else 0.0)
        state_counts.append(len(current))
        max_exact_order = order

        if not current:
            remaining = max_order - order
            values.extend([0.0] * remaining)
            state_counts.extend([0] * remaining)
            max_exact_order = max_order
            break

    return PauliNestedCommutatorBounds(
        values=tuple(values),
        max_order=max_order,
        max_exact_order=max_exact_order,
        state_counts=tuple(state_counts),
        used_locality_fallback=fallback_reason is not None,
        fallback_reason=fallback_reason,
        locality_k=locality_k,
        extensiveness_g=extensiveness_g,
    )


def _validate_order(order: int) -> None:
    if order != 1 and (order < 2 or order % 2):
        raise ValueError("order must be 1 or a positive even integer")


def _validate_partition(partition: TrotterPartition) -> None:
    if partition not in ("auto", "individual", "commuting"):
        raise ValueError("partition must be 'auto', 'individual', or 'commuting'")


def _individual_terms(operator: SparsePauliOp) -> tuple[SparsePauliOp, ...]:
    return tuple(
        SparsePauliOp(pauli, np.asarray([coefficient], dtype=complex))
        for pauli, coefficient in zip(operator.paulis, operator.coeffs, strict=True)
    )


def _commuting_groups(operator: SparsePauliOp) -> tuple[SparsePauliOp, ...]:
    """Greedily color the anticommutation graph while preserving term order."""
    terms = _individual_terms(operator)
    grouped_terms: list[list[SparsePauliOp]] = []
    grouped_paulis: list[list] = []
    for term in terms:
        pauli = term.paulis[0]
        for group, paulis in zip(grouped_terms, grouped_paulis, strict=True):
            if all(pauli.commutes(other) for other in paulis):
                group.append(term)
                paulis.append(pauli)
                break
        else:
            grouped_terms.append([term])
            grouped_paulis.append([pauli])
    return tuple(_simplify(sum(group[1:], start=group[0])) for group in grouped_terms)


def _commutator_prefactors(groups: tuple[SparsePauliOp, ...]) -> tuple[float, float]:
    """Evaluate the Childs et al. order-1/order-2 prefactors for given summands."""
    if len(groups) == 1:
        return 0.0, 0.0
    w1 = 0.0
    w2 = 0.0
    suffix = groups[-1]
    for gamma in range(len(groups) - 2, -1, -1):
        head = groups[gamma]
        inner = _commutator(suffix, head)
        w1 += _pauli_l1(inner) / 2
        w2 += _pauli_l1(_commutator(suffix, inner)) / 12
        w2 += _pauli_l1(_commutator(head, inner)) / 24
        suffix = _simplify(suffix + head)
    return w1, w2


def _order_commuting_groups(
    groups: tuple[SparsePauliOp, ...],
) -> tuple[SparsePauliOp, ...]:
    """Use a cheap order-2 proxy to choose among small group permutations."""
    if not 1 < len(groups) <= 3:
        return groups

    def key(ordering: tuple[int, ...]) -> tuple[float, int, tuple[int, ...]]:
        ordered = tuple(groups[index] for index in ordering)
        _, w2 = _commutator_prefactors(ordered)
        return w2, -int(ordered[-1].size), ordering

    best = min(permutations(range(len(groups))), key=key)
    return tuple(groups[index] for index in best)


@lru_cache(maxsize=None)
def _resolve_suzuki_specification(
    hamiltonian: PauliHamiltonian,
    order: int,
    partition: TrotterPartition = "auto",
) -> _SuzukiSpecification:
    _validate_order(order)
    _validate_partition(partition)
    resolved: _ResolvedTrotterPartition
    resolved = "individual" if partition == "auto" and order <= 2 else partition  # type: ignore[assignment]
    if partition == "auto" and order >= 4:
        resolved = "commuting"

    operator = hamiltonian.to_sparse_pauli_op()
    if resolved == "individual":
        groups = _individual_terms(operator)
    else:
        groups = _order_commuting_groups(_commuting_groups(operator))
    return _SuzukiSpecification(resolved, groups)


@lru_cache(maxsize=None)
def _suzuki_group_factors(
    group_count: int,
    order: int,
    *,
    merge_adjacent: bool = False,
) -> tuple[tuple[int, float], ...]:
    """Return one Qiskit-compatible Suzuki step as (group, coefficient) factors."""
    _validate_order(order)
    if group_count < 1:
        raise ValueError("group_count must be positive")
    if order == 1:
        factors = tuple((group, 1.0) for group in range(group_count))
    elif order == 2:
        halves = tuple((group, 0.5) for group in range(group_count - 1))
        factors = halves + ((group_count - 1, 1.0),) + tuple(reversed(halves))
    else:
        reduction = 1 / (4 - 4 ** (1 / (order - 1)))
        previous = _suzuki_group_factors(group_count, order - 2)
        outer = tuple((group, coefficient * reduction) for group, coefficient in previous)
        inner = tuple(
            (group, coefficient * (1 - 4 * reduction))
            for group, coefficient in previous
        )
        factors = outer + outer + inner + outer + outer

    if not merge_adjacent:
        return factors
    merged: list[tuple[int, float]] = []
    for group, coefficient in factors:
        if merged and merged[-1][0] == group:
            merged[-1] = (group, merged[-1][1] + coefficient)
        else:
            merged.append((group, coefficient))
    return tuple(merged)


def _suzuki_term_occurrences(
    hamiltonian: PauliHamiltonian,
    reps: int,
    order: int,
    partition: TrotterPartition = "auto",
) -> int:
    """Count Pauli rotations in the exact partitioned Qiskit expansion."""
    if reps < 1:
        raise ValueError("reps must be positive")
    specification = _resolve_suzuki_specification(hamiltonian, order, partition)
    sizes = specification.group_sizes
    per_step = sum(
        sizes[group] for group, _ in _suzuki_group_factors(len(sizes), order)
    )
    return reps * per_step


def _extend_word_polynomial(
    outer: dict[tuple[int, ...], float],
    group: int,
    coefficient: float,
    order: int,
    *,
    minimum_power: int,
) -> dict[tuple[int, ...], float]:
    """Prepend one factor's exponential generating series to outer words."""
    extended: defaultdict[tuple[int, ...], float] = defaultdict(float)
    magnitude = abs(coefficient)
    for outer_word, outer_weight in outer.items():
        for power in range(minimum_power, order - len(outer_word) + 1):
            word = (group,) * power + outer_word
            extended[word] += (
                outer_weight * magnitude**power / math.factorial(power)
            )
    return dict(extended)


def _theorem_word_weights(
    factors: tuple[tuple[int, float], ...],
    order: int,
) -> list[tuple[tuple[float, ...], dict[tuple[int, ...], float]]]:
    """Collapse Schubert--Mendl weak compositions into repeated group words."""
    factor_count = len(factors)
    center = math.ceil(factor_count / 2)
    group_count = 1 + max(group for group, _ in factors)

    prefixes: list[tuple[float, ...]] = []
    prefix = np.zeros(group_count, dtype=float)
    for group, coefficient in factors:
        prefixes.append(tuple(float(value) for value in prefix))
        prefix = prefix.copy()
        prefix[group] += coefficient

    entries: list[tuple[tuple[float, ...], dict[tuple[int, ...], float]]] = []
    factorial = math.factorial(order)

    # Theorem 1's first sum: j=2,...,s.  Work from the center out so the
    # q_{j+1},...,q_s polynomial is reused by the next value of j.
    outer: dict[tuple[int, ...], float] = {(): 1.0}
    for j in range(center - 1, 0, -1):
        group, coefficient = factors[j]
        positive = _extend_word_polynomial(
            outer,
            group,
            coefficient,
            order,
            minimum_power=1,
        )
        entries.append(
            (
                prefixes[j],
                {word: factorial * weight for word, weight in positive.items() if len(word) == order},
            )
        )
        outer = _extend_word_polynomial(
            outer,
            group,
            coefficient,
            order,
            minimum_power=0,
        )

    # The second sum: j=s+1,...,K.  A_j is innermost, hence each new factor
    # is prepended to the accumulated word for A_{j-1},...,A_{s+1}.
    outer = {(): 1.0}
    for j in range(center, factor_count):
        group, coefficient = factors[j]
        positive = _extend_word_polynomial(
            outer,
            group,
            coefficient,
            order,
            minimum_power=1,
        )
        entries.append(
            (
                prefixes[j],
                {word: factorial * weight for word, weight in positive.items() if len(word) == order},
            )
        )
        outer = _extend_word_polynomial(
            outer,
            group,
            coefficient,
            order,
            minimum_power=0,
        )
    return entries


def _nested_commutator_basis(
    groups: tuple[SparsePauliOp, ...],
    order: int,
) -> dict[tuple[tuple[int, ...], int], SparsePauliOp]:
    """Precompute ad-word(H_base) for all group words through ``order``."""
    basis = {((), base): group for base, group in enumerate(groups)}
    for depth in range(1, order + 1):
        previous = [
            (word, base, operator)
            for (word, base), operator in basis.items()
            if len(word) == depth - 1
        ]
        for word, base, operator in previous:
            for outer, group in enumerate(groups):
                basis[(word + (outer,), base)] = _commutator(group, operator)
    return basis


def _higher_order_commutator_prefactor(
    groups: tuple[SparsePauliOp, ...],
    order: Literal[4, 6],
) -> float:
    """Evaluate Schubert--Mendl Theorem 1 using Pauli 1-norms."""
    factors = _suzuki_group_factors(len(groups), order, merge_adjacent=True)
    if len(factors) == 1:
        return 0.0
    entries = _theorem_word_weights(factors, order)
    basis = _nested_commutator_basis(groups, order)

    contributions: defaultdict[
        tuple[int, ...], list[tuple[tuple[float, ...], float]]
    ] = defaultdict(list)
    for prefix, weights in entries:
        for word, weight in weights.items():
            contributions[word].append((prefix, weight))

    total = 0.0
    group_count = len(groups)
    for word, weighted_prefixes in contributions.items():
        coefficient_rows: dict[str, np.ndarray] = {}
        for base in range(group_count):
            for label, coefficient in basis[(word, base)].to_list():
                if coefficient == 0:
                    continue
                row = coefficient_rows.setdefault(
                    label,
                    np.zeros(group_count, dtype=complex),
                )
                row[base] += coefficient
        if not coefficient_rows:
            continue

        coefficients = np.stack(tuple(coefficient_rows.values()))
        prefixes = np.asarray(
            [prefix for prefix, _ in weighted_prefixes],
            dtype=float,
        ).T
        norms = np.sum(np.abs(coefficients @ prefixes), axis=0)
        weights = np.asarray([weight for _, weight in weighted_prefixes], dtype=float)
        total += float(norms @ weights)

    value = total / math.factorial(order + 1)
    return float(np.nextafter(value, np.inf)) if value else 0.0


@lru_cache(maxsize=None)
def _suzuki_error_prefactor(
    hamiltonian: PauliHamiltonian,
    order: int,
    partition: TrotterPartition = "auto",
) -> _SuzukiPrefactor:
    specification = _resolve_suzuki_specification(hamiltonian, order, partition)
    group_count = len(specification.groups)
    if group_count == 1:
        return _SuzukiPrefactor(
            0.0,
            specification.partition,
            group_count,
            "commuting-exact",
            True,
        )
    if order in (1, 2):
        w1, w2 = _commutator_prefactors(specification.groups)
        return _SuzukiPrefactor(
            w1 if order == 1 else w2,
            specification.partition,
            group_count,
            "childs-commutator",
            True,
        )
    if order in (4, 6) and group_count ** (order + 1) <= _MAX_HIGHER_ORDER_COMMUTATORS:
        value = _higher_order_commutator_prefactor(specification.groups, order)
        return _SuzukiPrefactor(
            value,
            specification.partition,
            group_count,
            "schubert-mendl-commutator",
            True,
        )
    return _SuzukiPrefactor(
        hamiltonian.alpha ** (order + 1),
        specification.partition,
        group_count,
        "alpha-proxy",
        False,
    )


def estimate_suzuki_error(
    hamiltonian: PauliHamiltonian,
    time: float,
    reps: int = 1,
    order: int = 2,
    *,
    partition: TrotterPartition = "auto",
) -> SuzukiErrorEstimate:
    """Estimate the operator-norm error of a partitioned Suzuki formula.

    Orders 1 and 2 use the tight Childs et al. commutator bounds.  Orders 4
    and 6 use Schubert--Mendl Theorem 1 when the number of commuting groups is
    within the practical work cap.  Other cases retain the historical
    coefficient-1-norm proxy and report ``rigorous=False``.
    """
    if reps < 1:
        raise ValueError("reps must be positive")
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    prefactor = _suzuki_error_prefactor(hamiltonian, order, partition)
    value = prefactor.value * abs(float(time)) ** (order + 1) / reps**order
    if value:
        value = float(np.nextafter(value, np.inf))
    return SuzukiErrorEstimate(
        error=value,
        prefactor=prefactor.value,
        time=float(time),
        reps=reps,
        order=order,
        partition=prefactor.partition,
        group_count=prefactor.group_count,
        method=prefactor.method,
        rigorous=prefactor.rigorous,
    )


@lru_cache(maxsize=None)
def suzuki_commutator_bounds(hamiltonian: PauliHamiltonian) -> tuple[float, float]:
    """Return prefactors (W1, W2) of the commutator Trotter error bounds.

    For the term ordering used by ``build_trotter_circuit``, Childs, Su, Tran,
    Wiebe, and Zhu (PRX 11, 011020 (2021), Prop. 9/10) give

        ||S1(d) - exp(-i d H)|| <= W1 d^2,   W1 = (1/2) sum_g ||[T_g, H_g]||
        ||S2(d) - exp(-i d H)|| <= W2 d^3,
        W2 = sum_g ( ||[T_g, [T_g, H_g]]||/12 + ||[H_g, [H_g, T_g]]||/24 )

    with T_g the sum of all terms after H_g.  Spectral norms are upper-bounded
    by Pauli coefficient 1-norms of the exactly computed nested commutators, so
    both prefactors are rigorous and scale with the commutation structure
    (O(n) for local chains) instead of the loose 1-norm power alpha^(p+1).
    """
    operator = hamiltonian.to_sparse_pauli_op()
    return _commutator_prefactors(_individual_terms(operator))


def build_trotter_circuit(
    hamiltonian: PauliHamiltonian,
    time: float,
    reps: int,
    order: int = 2,
    *,
    insert_barriers: bool = False,
    partition: TrotterPartition = "auto",
) -> QuantumCircuit:
    """Build exp(-i H time) as an actual product-formula circuit.

    This uses Pauli rotations; it deliberately does not exponentiate a dense
    matrix.  ``order=1`` selects Lie-Trotter. Higher orders must be even.
    """
    if reps < 1:
        raise ValueError("reps must be positive")
    specification = _resolve_suzuki_specification(hamiltonian, order, partition)
    if order == 1:
        synthesis = LieTrotter(
            reps=reps,
            insert_barriers=insert_barriers,
            preserve_order=True,
        )
    else:
        synthesis = SuzukiTrotter(
            order=order,
            reps=reps,
            insert_barriers=insert_barriers,
            preserve_order=True,
        )
    evolution_operator: SparsePauliOp | list[SparsePauliOp]
    if specification.partition == "individual":
        evolution_operator = hamiltonian.to_sparse_pauli_op()
    else:
        evolution_operator = list(specification.groups)
    gate = PauliEvolutionGate(
        evolution_operator,
        time=float(time),
        synthesis=synthesis,
    )
    circuit = QuantumCircuit(hamiltonian.num_qubits, name=f"Suzuki-{order}")
    circuit.append(gate, circuit.qubits)
    circuit = circuit.decompose()
    circuit.metadata = {
        **(circuit.metadata or {}),
        "trotter_order": order,
        "trotter_reps": reps,
        "trotter_partition": specification.partition,
        "trotter_group_count": len(specification.groups),
        "trotter_group_sizes": specification.group_sizes,
    }
    return circuit
