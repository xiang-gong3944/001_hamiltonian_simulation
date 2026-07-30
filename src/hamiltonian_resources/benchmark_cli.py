"""Command-line interface for schema-2 analytical benchmark runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .benchmark_plotting import save_benchmark_plots
from .benchmark_suite import (
    BenchmarkJob,
    BenchmarkProgress,
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

    plot = subparsers.add_parser("plot", help="plot a schema-2 benchmark CSV")
    plot.add_argument("--data", type=Path, required=True)
    plot.add_argument("--output-directory", type=Path)
    _add_plot_options(plot)

    run = subparsers.add_parser("run", help="run, persist, and plot a benchmark")
    run.add_argument("--config", default="benchmark_config.json")
    run.add_argument("--output-root", type=Path)
    _add_sweep_argument(run)
    _add_plot_options(run)
    return parser


def _print_progress(event: BenchmarkProgress) -> None:
    print(
        f"[{event.completed}/{event.total}] {event.sweep} n={event.system_qubits} "
        f"epsilon={event.target_error:g} {event.method_id}: {event.status}"
    )


def _run_and_save(
    job: BenchmarkJob,
    *,
    sweeps: tuple[BenchmarkSweep, ...],
    output_root: Path | None,
):
    frame = run_benchmark(job.benchmark, sweeps=sweeps, progress=_print_progress)
    root = output_root.resolve() if output_root is not None else job.output_root
    run_directory, csv_path, metadata_path = save_benchmark(
        frame, job.benchmark, output_root=root
    )
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
        )
        if args.command == "run":
            formats = args.formats or job.output_formats
            summary = (
                job.generate_summary_plots if args.summary is None else args.summary
            )
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
