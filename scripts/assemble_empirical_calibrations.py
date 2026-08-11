"""Reduce local calibration shards into deterministic review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hamiltonian_resources.calibration_pipeline import (
    assemble_calibration_artifacts,
    assemble_reproducibility_manifest,
    load_calibration_config,
    reduce_calibration_shards,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    arguments = parser.parse_args()
    config = load_calibration_config(arguments.config)
    reduced = reduce_calibration_shards(
        config,
        arguments.run_directory.joinpath("shards").glob("*.json"),
        require_complete=not arguments.allow_incomplete,
    )
    assembled = assemble_calibration_artifacts(reduced)
    provenance = assemble_reproducibility_manifest(reduced, assembled)
    _write(arguments.output_directory / "reduced_observations.json", reduced)
    _write(arguments.output_directory / "assembled_review.json", assembled)
    _write(arguments.output_directory / "provenance_manifest.json", provenance)


if __name__ == "__main__":
    main()
