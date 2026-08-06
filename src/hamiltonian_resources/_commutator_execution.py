"""Shared process execution for expensive commutator kernels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Callable, Hashable, Iterator, Sequence, TypeVar


_PARALLEL_WORK_THRESHOLD = 50_000
_T = TypeVar("_T")
_R = TypeVar("_R")


def validate_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers < 1:
        raise ValueError("workers must be positive")
    return workers


def cost_balanced_chunks(
    items: Sequence[_T],
    costs: Sequence[int],
    workers: int,
) -> list[tuple[_T, ...]]:
    """Split ordered work without making chunk boundaries scheduler-dependent."""
    if len(items) != len(costs):
        raise ValueError("items and costs must have equal lengths")
    if not items:
        return []
    target_chunks = min(len(items), max(1, 4 * workers))
    target_cost = max(1, (sum(costs) + target_chunks - 1) // target_chunks)
    chunks: list[tuple[_T, ...]] = []
    chunk: list[_T] = []
    chunk_cost = 0
    for item, cost in zip(items, costs):
        if chunk and chunk_cost + cost > target_cost and len(chunks) + 1 < target_chunks:
            chunks.append(tuple(chunk))
            chunk = []
            chunk_cost = 0
        chunk.append(item)
        chunk_cost += cost
    if chunk:
        chunks.append(tuple(chunk))
    return chunks


class CommutatorExecution:
    """Own one lazy process pool and computation-local result cache."""

    def __init__(self, workers: int = 1) -> None:
        self.workers = validate_workers(workers)
        self._pool: ProcessPoolExecutor | None = None
        self._cache: dict[Hashable, object] = {}

    def __enter__(self) -> "CommutatorExecution":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def should_parallelize(
        self,
        work: int,
        item_count: int,
        *,
        threshold: int = _PARALLEL_WORK_THRESHOLD,
    ) -> bool:
        return self.workers > 1 and work >= threshold and item_count >= 2 * self.workers

    def cached(self, key: Hashable, factory: Callable[[], _R]) -> _R:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]  # type: ignore[return-value]

    def map_chunks(
        self,
        function: Callable[[_T], _R],
        chunks: Sequence[_T],
    ) -> list[_R]:
        if self.workers == 1 or len(chunks) < 2:
            return [function(chunk) for chunk in chunks]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        results: list[_R] = []
        window = 2 * self.workers
        for start in range(0, len(chunks), window):
            futures = [
                self._pool.submit(function, chunk) for chunk in chunks[start : start + window]
            ]
            results.extend(future.result() for future in futures)
        return results


@contextmanager
def execution_scope(
    workers: int,
    execution: CommutatorExecution | None,
) -> Iterator[CommutatorExecution]:
    validate_workers(workers)
    if execution is not None:
        if workers != execution.workers:
            raise ValueError("workers must match the shared commutator execution")
        yield execution
        return
    with CommutatorExecution(workers) as owned:
        yield owned
