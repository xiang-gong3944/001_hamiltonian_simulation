"""Run or resume a configured high-precision empirical calibration matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from hamiltonian_resources.calibration_pipeline import (
    load_calibration_config,
    run_calibration_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    config = load_calibration_config(arguments.config)
    paths = run_calibration_matrix(
        config,
        arguments.run_directory / "shards",
        workers=arguments.workers,
    )
    print(f"completed {len(paths)} task shards under {arguments.run_directory}")


if __name__ == "__main__":
    main()
