"""Tqdm rendering kept separate from structured progress callbacks."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable, TypeVar

from tqdm.auto import tqdm

from ._commutator_execution import CommutatorProgress


_T = TypeVar("_T")


def combine_callbacks(
    first: Callable[[_T], None] | None,
    second: Callable[[_T], None] | None,
) -> Callable[[_T], None] | None:
    if first is None:
        return second
    if second is None:
        return first

    def combined(event: _T) -> None:
        first(event)
        second(event)

    return combined


class TqdmProgressRenderer(AbstractContextManager["TqdmProgressRenderer"]):
    """Render one outer benchmark bar and one reusable inner stage display."""

    def __init__(self) -> None:
        self._outer = None
        self._inner = None
        self._inner_key: tuple[object, ...] | None = None
        self._inner_completed = 0

    def benchmark(self, event) -> None:
        if self._outer is None:
            self._outer = tqdm(
                total=event.total,
                desc="benchmark",
                position=0,
                leave=True,
                mininterval=0.2,
            )
        self._outer.set_postfix_str(
            f"{event.sweep} n={event.system_qubits} "
            f"eps={event.target_error:g} {event.method_id} {event.status}",
            refresh=False,
        )
        delta = event.completed - self._outer.n
        if delta > 0:
            self._outer.update(delta)

    def commutator(self, event: CommutatorProgress) -> None:
        determinate = event.total is not None
        key = (
            (event.family, event.phase, event.commutator_order, event.total, True)
            if determinate
            else (event.family, "adaptive", False)
        )
        if self._inner is None or key != self._inner_key:
            if self._inner is not None:
                self._inner.close()
            self._inner = tqdm(
                total=event.total,
                position=1,
                leave=False,
                mininterval=0.2,
                bar_format=(
                    None if determinate else "{desc} [{elapsed}, {n_fmt} updates] {postfix}"
                ),
            )
            self._inner_key = key
            self._inner_completed = 0

        if event.family == "multiproduct":
            self._inner.set_description_str(f"MPF q={event.commutator_order or '-'}")
            candidate = (
                f"r={event.segment_candidate} " if event.segment_candidate is not None else ""
            )
            target = f"eps={event.target_error:g} " if event.target_error is not None else ""
            self._inner.set_postfix_str(
                f"{candidate}n={event.system_qubits} {target}{event.phase}".strip(),
                refresh=False,
            )
        else:
            self._inner.set_description_str(
                f"Trotter p={event.formula_order} q={event.commutator_order}"
            )
            self._inner.set_postfix_str(
                f"n={event.system_qubits} {event.phase}",
                refresh=False,
            )

        target_completed = event.completed if determinate else self._inner_completed + 1
        delta = target_completed - self._inner_completed
        if delta > 0:
            self._inner.update(delta)
            self._inner_completed = target_completed

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None
        if self._outer is not None:
            self._outer.close()
            self._outer = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
