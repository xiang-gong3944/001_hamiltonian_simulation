"""Command-line interface for persisted analytical benchmark sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .benchmark_plotting import plot_saved_benchmark
from .benchmark_suite import (
    SWEEP_FILENAMES,
    BenchmarkSweep,
    generate_and_save_benchmark,
    load_benchmark_config,
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
        help="benchmark sweep to process (default: all)",
    )


def _add_plot_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        help="override output formats stored in benchmark metadata",
    )
    summary = parser.add_mutually_exclusive_group()
    summary.add_argument("--summary", action="store_true", dest="summary")
    summary.add_argument("--no-summary", action="store_false", dest="summary")
    parser.set_defaults(summary=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamiltonian-benchmark",
        description="Generate and plot analytical Hamiltonian-simulation resource sweeps.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate persisted CSV data")
    generate.add_argument("--config", default="benchmark_config.json")
    _add_sweep_argument(generate)

    plot = subparsers.add_parser("plot", help="plot previously persisted CSV data")
    source = plot.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", nargs="+", type=Path, help="one or more benchmark CSVs")
    source.add_argument("--data-dir", type=Path, help="directory containing standard CSV names")
    _add_sweep_argument(plot)
    plot.add_argument("--output-directory", type=Path)
    _add_plot_overrides(plot)

    run = subparsers.add_parser("run", help="generate data, reload it, and plot it")
    run.add_argument("--config", default="benchmark_config.json")
    _add_sweep_argument(run)
    _add_plot_overrides(run)
    return parser


def _generate(config_path: str | Path, sweep_value: str) -> tuple[list[Path], int]:
    config = load_benchmark_config(config_path)
    data_paths: list[Path] = []
    failure_count = 0
    for sweep in _sweeps(sweep_value):
        frame, csv_path, metadata_path = generate_and_save_benchmark(config, sweep)
        data_paths.append(csv_path)
        failures = int((frame["status"] == "error").sum())
        skipped = int((frame["status"] == "skipped").sum())
        failure_count += failures
        print(
            f"wrote {csv_path} "
            f"({len(frame)} rows, {failures} failures, {skipped} skipped)"
        )
        print(f"wrote {metadata_path}")
    return data_paths, failure_count


def _data_paths(args: argparse.Namespace) -> list[Path]:
    if args.data:
        return [Path(path) for path in args.data]
    directory = Path(args.data_dir)
    return [directory / SWEEP_FILENAMES[sweep] for sweep in _sweeps(args.sweep)]


def _plot(paths: Sequence[Path], args: argparse.Namespace) -> None:
    formats = tuple(args.formats) if args.formats else None
    for path in paths:
        outputs = plot_saved_benchmark(
            path,
            output_directory=getattr(args, "output_directory", None),
            output_formats=formats,
            summary=args.summary,
        )
        for output in outputs:
            print(f"wrote {output}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI and return a process exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            _, failures = _generate(args.config, args.sweep)
            return 1 if failures else 0
        if args.command == "plot":
            _plot(_data_paths(args), args)
            return 0
        paths, failures = _generate(args.config, args.sweep)
        _plot(paths, args)
        return 1 if failures else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
