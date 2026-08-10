"""Opt-in wall-time benchmark for serial and process-parallel bounds."""

from __future__ import annotations

import argparse
import statistics
import time

from hamiltonian_resources import (
    estimate_suzuki_error,
    heisenberg_chain,
    pauli_nested_commutator_bounds,
    select_mpf_segments,
    transverse_field_ising,
)


def _measure(function, repeats: int) -> float:
    samples = []
    for repetition in range(repeats):
        started = time.perf_counter()
        function(repetition)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    def trotter(workers: int, repetition: int) -> None:
        hamiltonian = heisenberg_chain(args.sites, field_z=0.3)
        hamiltonian = type(hamiltonian)(
            hamiltonian.num_qubits,
            hamiltonian.terms,
            f"{hamiltonian.name}-run-{workers}-{repetition}",
        )
        estimate_suzuki_error(
            hamiltonian,
            float(args.sites),
            order=6,
            workers=workers,
        )

    def mpf(workers: int, repetition: int) -> None:
        pauli_nested_commutator_bounds.cache_clear()
        hamiltonian = transverse_field_ising(args.sites, field=3.0)
        select_mpf_segments(
            hamiltonian,
            float(args.sites),
            1e-6,
            3,
            method="mizuta2026-commutator-ideal-rigorous",
            workers=workers,
        )

    for label, benchmark in (("trotter-p6", trotter), ("mizuta-mpf-m3", mpf)):
        serial = _measure(lambda repetition: benchmark(1, repetition), args.repeats)
        parallel = _measure(
            lambda repetition: benchmark(args.workers, repetition),
            args.repeats,
        )
        print(
            f"{label}: serial={serial:.3f}s parallel={parallel:.3f}s "
            f"speedup={serial / parallel:.2f}x"
        )


if __name__ == "__main__":
    main()
