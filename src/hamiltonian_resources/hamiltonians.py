"""Validated Pauli-sum Hamiltonians and common spin-chain constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass(frozen=True)
class PauliHamiltonian:
    """A real Hermitian Pauli sum.

    Labels use Qiskit's convention: the rightmost character acts on qubit 0.
    Identity terms are allowed (they only contribute a global phase).
    """

    num_qubits: int
    terms: tuple[tuple[str, float], ...]
    name: str = "H"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terms",
            tuple((str(label), float(coefficient)) for label, coefficient in self.terms),
        )
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be positive")
        if not self.terms:
            raise ValueError("at least one Pauli term is required")
        for label, coefficient in self.terms:
            if len(label) != self.num_qubits or set(label) - set("IXYZ"):
                raise ValueError(f"invalid {self.num_qubits}-qubit Pauli label: {label!r}")
            if not np.isfinite(coefficient):
                raise ValueError("coefficients must be finite real numbers")

    @classmethod
    def from_terms(
        cls, num_qubits: int, terms: Iterable[tuple[str, float]], name: str = "H"
    ) -> "PauliHamiltonian":
        combined: dict[str, float] = {}
        for label, coefficient in terms:
            combined[label] = combined.get(label, 0.0) + float(coefficient)
        cleaned = tuple((p, c) for p, c in combined.items() if not np.isclose(c, 0.0))
        return cls(num_qubits, cleaned, name)

    @property
    def alpha(self) -> float:
        """LCU normalization alpha = sum_j |h_j|."""
        return float(sum(abs(c) for _, c in self.terms))

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def max_pauli_weight(self) -> int:
        return max(sum(ch != "I" for ch in label) for label, _ in self.terms)

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        return SparsePauliOp.from_list(list(self.terms)).simplify()

    def matrix(self) -> np.ndarray:
        """Dense matrix for validation only; circuit builders never call this."""
        return self.to_sparse_pauli_op().to_matrix()


def _two_site_label(n: int, left: int, pauli: str) -> str:
    chars = ["I"] * n
    chars[n - 1 - left] = pauli
    chars[n - 2 - left] = pauli
    return "".join(chars)


def _one_site_label(n: int, site: int, pauli: str) -> str:
    chars = ["I"] * n
    chars[n - 1 - site] = pauli
    return "".join(chars)


def transverse_field_ising(
    num_qubits: int, coupling: float = 1.0, field: float = 1.0, periodic: bool = False
) -> PauliHamiltonian:
    """H = -J sum Z_i Z_(i+1) - h sum X_i."""
    terms = [(_two_site_label(num_qubits, i, "Z"), -coupling) for i in range(num_qubits - 1)]
    if periodic and num_qubits > 2:
        chars = ["I"] * num_qubits
        chars[0] = chars[-1] = "Z"
        terms.append(("".join(chars), -coupling))
    terms.extend((_one_site_label(num_qubits, i, "X"), -field) for i in range(num_qubits))
    return PauliHamiltonian.from_terms(num_qubits, terms, f"TFIM-{num_qubits}")


def heisenberg_chain(
    num_qubits: int, coupling: float = 1.0, field_z: float = 0.0
) -> PauliHamiltonian:
    """Open XXX chain H = J sum(XX+YY+ZZ) + h sum Z."""
    terms: list[tuple[str, float]] = []
    for i in range(num_qubits - 1):
        terms.extend((_two_site_label(num_qubits, i, p), coupling) for p in "XYZ")
    terms.extend((_one_site_label(num_qubits, i, "Z"), field_z) for i in range(num_qubits))
    return PauliHamiltonian.from_terms(num_qubits, terms, f"Heisenberg-{num_qubits}")
