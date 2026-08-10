"""Backend-independent Hamiltonian-simulation method specifications."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias

from .multiproduct import (
    MPFBranchCountPolicy,
    MPFErrorMethod,
    MPFSchedule,
    optimal_mpf_exponents,
)


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
    term_count: int | None
    schedule: MPFSchedule = "new"
    error_method: MPFErrorMethod = "low2019-l1-ideal-rigorous"
    branch_count_policy: MPFBranchCountPolicy = "fixed"

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
        if self.branch_count_policy == "fixed":
            return f"mpf-m{self.term_count}{suffix}"
        return f"mpf-j-mizuta2026-theorem6{suffix}"

    @property
    def label(self) -> str:
        suffix = "" if self.schedule == "new" else f" ({self.schedule})"
        if self.error_method == "legacy-w2-proxy":
            suffix += " [legacy W2 heuristic]"
        elif self.error_method == "childs2021-w2-triangle-ideal-rigorous":
            suffix += " [Childs 2021 W2 triangle]"
        elif self.error_method == "mizuta2026-commutator-ideal-rigorous":
            suffix += " [Mizuta 2026 refined commutator]"
        elif self.error_method == "mizuta2026-theorem3-legacy-ideal-rigorous":
            suffix += " [Mizuta 2026 legacy Theorem 3]"
        elif self.error_method == "best-rigorous-ideal":
            suffix += " [best rigorous ideal bound]"
        if self.branch_count_policy == "fixed":
            return f"MPF J={self.term_count}, formal order={2 * self.term_count}{suffix}"
        return f"MPF J=Mizuta Theorem 6 (resolved per point){suffix}"

    def validate(self) -> None:
        if self.branch_count_policy == "fixed":
            if isinstance(self.term_count, bool) or not isinstance(self.term_count, Integral):
                raise ValueError("fixed MPF branch-count policy requires an integer term_count")
            optimal_mpf_exponents(int(self.term_count), schedule=self.schedule)
        elif self.branch_count_policy == "mizuta2026-theorem6":
            if self.term_count is not None:
                raise ValueError(
                    "mizuta2026-theorem6 branch-count policy requires term_count=None"
                )
            # Validate the schedule name before point-dependent J is available.
            optimal_mpf_exponents(2, schedule=self.schedule)
        else:
            raise ValueError(
                "branch_count_policy must be 'fixed' or 'mizuta2026-theorem6'"
            )
        if self.error_method not in (
            "low2019-l1-ideal-rigorous",
            "childs2021-w2-triangle-ideal-rigorous",
            "mizuta2026-commutator-ideal-rigorous",
            "mizuta2026-theorem3-legacy-ideal-rigorous",
            "best-rigorous-ideal",
            "low-rigorous",
            "legacy-w2-proxy",
        ):
            raise ValueError(
                "MPF error method must be 'low2019-l1-ideal-rigorous' "
                "(historical alias 'low-rigorous'), "
                "'childs2021-w2-triangle-ideal-rigorous', "
                "'mizuta2026-commutator-ideal-rigorous', "
                "'mizuta2026-theorem3-legacy-ideal-rigorous', "
                "'best-rigorous-ideal', or 'legacy-w2-proxy'"
            )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "family": self.family,
            "schedule": self.schedule,
            "error_method": self.error_method,
        }
        if self.branch_count_policy == "fixed":
            result["term_count"] = int(self.term_count)  # type: ignore[arg-type]
        else:
            result["branch_count_policy"] = self.branch_count_policy
        return result


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
