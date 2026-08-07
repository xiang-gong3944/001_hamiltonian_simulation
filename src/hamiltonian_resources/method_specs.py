"""Backend-independent Hamiltonian-simulation method specifications."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias

from .multiproduct import MPFErrorMethod, MPFSchedule, optimal_mpf_exponents


@dataclass(frozen=True)
class TrotterMethod:
    order: int

    @property
    def family(self) -> str:
        return "trotter"

    @property
    def method_id(self) -> str:
        return f"trotter-p{self.order}"

    @property
    def label(self) -> str:
        return f"Trotter p={self.order}"

    def validate(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, Integral):
            raise ValueError("Trotter order must be 1 or a positive even integer")
        if self.order != 1 and (self.order < 2 or self.order % 2):
            raise ValueError("Trotter order must be 1 or a positive even integer")

    def as_dict(self) -> dict[str, object]:
        return {"family": self.family, "order": int(self.order)}


@dataclass(frozen=True)
class MultiproductMethod:
    term_count: int
    schedule: MPFSchedule = "new"
    error_method: MPFErrorMethod = "low2019-l1-ideal-rigorous"

    @property
    def family(self) -> str:
        return "multiproduct"

    @property
    def method_id(self) -> str:
        suffix = "" if self.schedule == "new" else f"-{self.schedule}"
        if self.error_method not in (
            "low2019-l1-ideal-rigorous",
            "low-rigorous",
        ):
            suffix += f"-{self.error_method}"
        return f"mpf-m{self.term_count}{suffix}"

    @property
    def label(self) -> str:
        suffix = "" if self.schedule == "new" else f" ({self.schedule})"
        if self.error_method == "legacy-w2-proxy":
            suffix += " [legacy W2 heuristic]"
        elif self.error_method == "mizuta2026-commutator-ideal-rigorous":
            suffix += " [Mizuta 2026 commutator]"
        elif self.error_method == "best-rigorous-ideal":
            suffix += " [best rigorous ideal bound]"
        return f"MPF J={self.term_count}, formal order={2 * self.term_count}{suffix}"

    def validate(self) -> None:
        if isinstance(self.term_count, bool) or not isinstance(self.term_count, Integral):
            raise ValueError("MPF term count must be an integer")
        optimal_mpf_exponents(int(self.term_count), schedule=self.schedule)
        if self.error_method not in (
            "low2019-l1-ideal-rigorous",
            "mizuta2026-commutator-ideal-rigorous",
            "best-rigorous-ideal",
            "low-rigorous",
            "legacy-w2-proxy",
        ):
            raise ValueError(
                "MPF error method must be 'low2019-l1-ideal-rigorous' "
                "(historical alias 'low-rigorous'), "
                "'mizuta2026-commutator-ideal-rigorous', "
                "'best-rigorous-ideal', or 'legacy-w2-proxy'"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "term_count": int(self.term_count),
            "schedule": self.schedule,
            "error_method": self.error_method,
        }


@dataclass(frozen=True)
class QSVTMethod:
    @property
    def family(self) -> str:
        return "qsvt"

    @property
    def method_id(self) -> str:
        return "qsvt"

    @property
    def label(self) -> str:
        return "QSVT"

    def validate(self) -> None:
        return None

    def as_dict(self) -> dict[str, object]:
        return {"family": self.family}


MethodSpec: TypeAlias = TrotterMethod | MultiproductMethod | QSVTMethod


def default_methods() -> list[MethodSpec]:
    return [
        *(TrotterMethod(order) for order in (1, 2, 4, 6)),
        *(MultiproductMethod(term_count) for term_count in (3, 5, 7)),
        QSVTMethod(),
    ]
