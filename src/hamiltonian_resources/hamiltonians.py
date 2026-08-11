"""Validated Pauli-sum Hamiltonians and common spin-chain constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias

import numpy as np
from qiskit.quantum_info import SparsePauliOp


ModelParameter: TypeAlias = bool | int | float | str


@dataclass(frozen=True)
class HamiltonianModelMetadata:
    """Stable physical-model identity used by empirical calibrations.

    The metadata is deliberately separate from the display ``name``.  It is
    populated by the built-in model constructors and omitted for arbitrary
    Pauli sums unless their caller supplies an equally explicit identity.
    """

    model: str
    parameters: tuple[tuple[str, ModelParameter], ...]
    geometry: str
    boundary_condition: str

    def __post_init__(self) -> None:
        if not self.model or not self.geometry:
            raise ValueError("model and geometry must be nonempty")
        if self.boundary_condition not in {"open", "periodic"}:
            raise ValueError("boundary_condition must be 'open' or 'periodic'")
        names = [name for name, _ in self.parameters]
        if any(not name for name in names) or names != sorted(names):
            raise ValueError("model parameters must be named and sorted")
        if len(names) != len(set(names)):
            raise ValueError("model parameter names must be unique")
        normalized: list[tuple[str, ModelParameter]] = []
        for name, value in self.parameters:
            if isinstance(value, bool) or isinstance(value, str):
                normalized.append((name, value))
            elif isinstance(value, (int, float)) and np.isfinite(value):
                normalized.append((name, float(value)))
            else:
                raise ValueError("model parameters must be finite scalars or strings")
        object.__setattr__(self, "parameters", tuple(normalized))

    @classmethod
    def from_mapping(
        cls,
        model: str,
        parameters: dict[str, ModelParameter],
        *,
        geometry: str,
        boundary_condition: str,
    ) -> "HamiltonianModelMetadata":
        return cls(
            model,
            tuple(sorted(parameters.items())),
            geometry,
            boundary_condition,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "parameters": dict(self.parameters),
            "geometry": self.geometry,
            "boundary_condition": self.boundary_condition,
        }


@dataclass(frozen=True)
class PauliHamiltonian:
    """A real Hermitian Pauli sum.

    Labels use Qiskit's convention: the rightmost character acts on qubit 0.
    Identity terms are allowed (they only contribute a global phase).
    """

    num_qubits: int
    terms: tuple[tuple[str, float], ...]
    name: str = "H"
    model_metadata: HamiltonianModelMetadata | None = None

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
        cls,
        num_qubits: int,
        terms: Iterable[tuple[str, float]],
        name: str = "H",
        model_metadata: HamiltonianModelMetadata | None = None,
    ) -> "PauliHamiltonian":
        combined: dict[str, float] = {}
        for label, coefficient in terms:
            combined[label] = combined.get(label, 0.0) + float(coefficient)
        cleaned = tuple((p, c) for p, c in combined.items() if not np.isclose(c, 0.0))
        return cls(num_qubits, cleaned, name, model_metadata)

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
    metadata = HamiltonianModelMetadata.from_mapping(
        "transverse_field_ising",
        {"coupling": coupling, "field": field, "periodic": periodic},
        geometry="1d-chain",
        boundary_condition="periodic" if periodic else "open",
    )
    return PauliHamiltonian.from_terms(
        num_qubits,
        terms,
        f"TFIM-{num_qubits}",
        metadata,
    )


def heisenberg_chain(
    num_qubits: int, coupling: float = 1.0, field_z: float = 0.0
) -> PauliHamiltonian:
    """Open XXX chain H = J sum(XX+YY+ZZ) + h sum Z."""
    terms: list[tuple[str, float]] = []
    for i in range(num_qubits - 1):
        terms.extend((_two_site_label(num_qubits, i, p), coupling) for p in "XYZ")
    terms.extend((_one_site_label(num_qubits, i, "Z"), field_z) for i in range(num_qubits))
    metadata = HamiltonianModelMetadata.from_mapping(
        "heisenberg_chain",
        {"coupling": coupling, "field_z": field_z},
        geometry="1d-chain",
        boundary_condition="open",
    )
    return PauliHamiltonian.from_terms(
        num_qubits,
        terms,
        f"Heisenberg-{num_qubits}",
        metadata,
    )
