"""Command-line interface for resumable Cartesian resource grids."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .resource_grid import (
    load_resource_grid_config,
    resource_grid_preset,
    run_resource_grid,
)


def _worker_count(requested: int) -> int:
    if requested < 0:
        raise ValueError("workers must be nonnegative; use 0 for automatic selection")
    if requested:
        return requested
    return min(4, max(1, (os.cpu_count() or 1) - 1))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamiltonian-resource-grid",
        description="Run resumable model/N-sharded Hamiltonian resource grids.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="evaluate, checkpoint, validate, and merge a grid")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=("sanity-low", "sanity-high", "full"))
    source.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--workers", type=int, default=0)
    run.add_argument("--resume", action="store_true")
    progress = run.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="show_progress")
    progress.add_argument("--no-progress", action="store_false", dest="show_progress")
    run.set_defaults(show_progress=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.preset is not None:
            config = resource_grid_preset(arguments.preset)
            default_output = Path("outputs") / "resource_grid" / arguments.preset
        else:
            config = load_resource_grid_config(arguments.config)
            default_output = Path("outputs") / "resource_grid" / f"custom-{config.digest[:8]}"
        summary = run_resource_grid(
            config,
            arguments.output or default_output,
            resume=arguments.resume,
            workers=_worker_count(arguments.workers),
            show_progress=(
                sys.stderr.isatty()
                if arguments.show_progress is None
                else arguments.show_progress
            ),
        )
        print(f"wrote {summary.manifest_path}")
        print(f"wrote {summary.validation_path}")
        print(f"wrote {summary.merged_path}")
        print(
            f"shards={summary.completed_shards} skipped={summary.skipped_shards} "
            f"missing_empirical={summary.expected_missing_rows} "
            f"unexpected_failures={summary.failed_rows}"
        )
        return 1 if summary.failed_rows else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
