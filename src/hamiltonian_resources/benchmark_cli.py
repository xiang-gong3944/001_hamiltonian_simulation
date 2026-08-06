"""Command-line interface for schema-2 analytical benchmark runs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .benchmark_plotting import save_benchmark_plots
from .benchmark_suite import (
    BenchmarkJob,
    BenchmarkSweep,
    load_benchmark,
    load_benchmark_job,
    run_benchmark,
    save_benchmark,
)


def _sweeps(value: str) -> tuple[BenchmarkSweep, ...]:
    if value == "all":
        return ("system-size", "target-error")
    return (value,)  # type: ignore[return-value]


def _add_sweep_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sweep",
        choices=("system-size", "target-error", "all"),
        default="all",
    )


def _add_plot_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"))
    summary = parser.add_mutually_exclusive_group()
    summary.add_argument("--summary", action="store_true", dest="summary")
    summary.add_argument("--no-summary", action="store_false", dest="summary")
    parser.set_defaults(summary=None)


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="commutator worker processes; 0 selects up to four automatically",
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="show_progress")
    progress.add_argument("--no-progress", action="store_false", dest="show_progress")
    parser.set_defaults(show_progress=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamiltonian-benchmark",
        description="Run and plot notebook-compatible analytical resource sweeps.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="run and persist benchmark data")
    generate.add_argument("--config", default="benchmark_config.json")
    generate.add_argument("--output-root", type=Path)
    _add_sweep_argument(generate)
    _add_execution_options(generate)

    plot = subparsers.add_parser("plot", help="plot a schema-2 benchmark CSV")
    plot.add_argument("--data", type=Path, required=True)
    plot.add_argument("--output-directory", type=Path)
    _add_plot_options(plot)

    run = subparsers.add_parser("run", help="run, persist, and plot a benchmark")
    run.add_argument("--config", default="benchmark_config.json")
    run.add_argument("--output-root", type=Path)
    _add_sweep_argument(run)
    _add_plot_options(run)
    _add_execution_options(run)
    return parser


def _worker_count(requested: int) -> int:
    if requested < 0:
        raise ValueError("workers must be nonnegative; use 0 for automatic selection")
    if requested:
        return requested
    return min(4, max(1, (os.cpu_count() or 1) - 1))


def _run_and_save(
    job: BenchmarkJob,
    *,
    sweeps: tuple[BenchmarkSweep, ...],
    output_root: Path | None,
    workers: int,
    show_progress: bool,
):
    frame = run_benchmark(
        job.benchmark,
        sweeps=sweeps,
        workers=workers,
        show_progress=show_progress,
    )
    root = output_root.resolve() if output_root is not None else job.output_root
    run_directory, csv_path, metadata_path = save_benchmark(frame, job.benchmark, output_root=root)
    print(f"wrote {csv_path}")
    print(f"wrote {metadata_path}")
    failures = int((frame["status"] == "error").sum())
    return frame, run_directory, failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plot":
            frame = load_benchmark(args.data)
            output_directory = args.output_directory or args.data.resolve().parent
            outputs = save_benchmark_plots(
                frame,
                output_directory=output_directory,
                output_formats=args.formats or ("png", "pdf"),
                summary=bool(args.summary),
            )
            for output in outputs:
                print(f"wrote {output}")
            return 0

        job = load_benchmark_job(args.config)
        frame, run_directory, failures = _run_and_save(
            job,
            sweeps=_sweeps(args.sweep),
            output_root=args.output_root,
            workers=_worker_count(args.workers),
            show_progress=(
                sys.stderr.isatty() if args.show_progress is None else args.show_progress
            ),
        )
        if args.command == "run":
            formats = args.formats or job.output_formats
            summary = job.generate_summary_plots if args.summary is None else args.summary
            outputs = save_benchmark_plots(
                frame,
                output_directory=run_directory,
                output_formats=formats,
                summary=summary,
            )
            for output in outputs:
                print(f"wrote {output}")
        return 1 if failures else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
