"""Command-line entry point for the standalone fourth-order bound benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .fourth_order_comparison import (
    generate_and_save_fourth_order_comparison,
    load_fourth_order_comparison_config,
    plot_fourth_order_comparison,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hamiltonian-bound-comparison",
        description="Compare fourth-order Childs and Schubert--Mendl bounds.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="generate tables and plots")
    run.add_argument("--config", required=True, type=Path)
    plot = subparsers.add_parser("plot", help="plot a saved comparison CSV")
    plot.add_argument("--data", required=True, type=Path)
    plot.add_argument("--output-dir", type=Path)
    plot.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("png", "pdf", "svg"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison command and report every written path."""
    args = _parser().parse_args(argv)
    if args.command == "run":
        config = load_fourth_order_comparison_config(args.config)
        outputs = generate_and_save_fourth_order_comparison(config)
    else:
        formats = tuple(args.formats or ("png", "pdf"))
        frame = pd.read_csv(args.data)
        outputs = plot_fourth_order_comparison(
            frame,
            output_directory=args.output_dir or args.data.parent,
            output_formats=formats,
        )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
