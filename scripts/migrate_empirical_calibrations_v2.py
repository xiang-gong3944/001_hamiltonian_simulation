"""Create a lossless tagged-schema migration of reviewed v1 calibrations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from hamiltonian_resources.calibration_study import select_size_law_model
from hamiltonian_resources.empirical import canonical_json_digest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = PROJECT_ROOT / "docs" / "calibration_data"
PACKAGE_DATA = PROJECT_ROOT / "src" / "hamiltonian_resources" / "data"


def _write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    accepted_v1 = json.loads(
        (DOCS / "empirical_1d_v1_accepted.json").read_text(encoding="utf-8")
    )
    fits_v1 = json.loads(
        (DOCS / "empirical_1d_v1_fits.json").read_text(encoding="utf-8")
    )
    package_path_v1 = PACKAGE_DATA / "empirical_calibrations_v1.json"
    package_v1 = json.loads(package_path_v1.read_text(encoding="utf-8"))

    accepted_v2 = copy.deepcopy(accepted_v1)
    accepted_v2["schema_version"] = "2.0"
    accepted_v2["study_id"] = "empirical-operator-norm-1d-v2-lossless-migration"
    accepted_v2["migration"] = {
        "kind": "lossless-v1-observation-migration",
        "source_study_id": accepted_v1["study_id"],
        "source_canonical_digest": canonical_json_digest(accepted_v1),
    }
    accepted_path_v2 = DOCS / "empirical_1d_v2_accepted.json"
    _write(accepted_path_v2, accepted_v2)
    accepted_v2_digest = canonical_json_digest(accepted_v2)

    fits_v2 = copy.deepcopy(fits_v1)
    fits_v2["schema_version"] = "2.0"
    fits_v2["study_id"] = accepted_v2["study_id"]
    fits_v2["migration"] = "lossless-affine-parameters"
    _write(DOCS / "empirical_1d_v2_fits.json", fits_v2)

    observations_by_id = {
        row["calibration_id"]: row for row in accepted_v1["observations"]
    }
    audit_rows = []
    for calibration_id, row in sorted(observations_by_id.items()):
        sizes = tuple(int(value) for value in row["sizes"])
        if all(size in sizes for size in range(4, 11)):
            selection = select_size_law_model(
                sizes,
                tuple(float(value) for value in row["coefficients"]),
                reviewed_size_max=100,
            )
            audit_rows.append(
                {
                    "calibration_id": calibration_id,
                    "selected_model": (
                        selection.selected.model if selection.selected else None
                    ),
                    "passed_v2_size_gates": selection.selected is not None,
                    "failure_reasons": list(selection.failure_reasons),
                }
            )
        else:
            audit_rows.append(
                {
                    "calibration_id": calibration_id,
                    "selected_model": None,
                    "passed_v2_size_gates": False,
                    "failure_reasons": ["N=9,10 holdout observations are absent"],
                }
            )
    review = {
        "schema_version": "2.0",
        "study_id": accepted_v2["study_id"],
        "status": "reviewed-lossless-migration",
        "reviewed_size_policy": (
            "Each migrated record remains reviewed only through its observed v1 size "
            "maximum. The N<=100 audit is diagnostic and does not silently expand it."
        ),
        "accepted_observations_canonical_digest": accepted_v2_digest,
        "size_model_audit": audit_rows,
    }
    _write(DOCS / "empirical_1d_v2_migration_review.json", review)

    package_v2 = copy.deepcopy(package_v1)
    package_v2["schema_version"] = "2.0"
    package_v2["dataset_id"] = "empirical-operator-norm-1d-v2-lossless-migration"
    for row in package_v2["calibrations"]:
        coefficient = row["coefficient"]
        row["coefficient"] = {
            "model": "affine",
            "parameters": {
                "slope": coefficient["slope"],
                "intercept": coefficient["intercept"],
            },
        }
        row["reviewed_size_max"] = int(row["size_range"][1])
        row["stability_diagnostics"] = {"legacy_lossless_migration": 1.0}
        row["external_validation_sizes"] = []
        row["external_validation_status"] = "not-required"
        row["source"] = "docs/calibration_data/empirical_1d_v2_accepted.json"
        row["source_digest"] = accepted_v2_digest
        row["reference"] = (
            "Repository empirical operator-norm calibration study v1; "
            "lossless tagged-schema migration (2026-08-11)"
        )
    _write(PACKAGE_DATA / "empirical_calibrations_v2_migrated.json", package_v2)

    canonical_v1_digest = canonical_json_digest(accepted_v1)
    package_v1_text = package_path_v1.read_text(encoding="utf-8")
    old_digests = {row["source_digest"] for row in package_v1["calibrations"]}
    for old_digest in old_digests:
        package_v1_text = package_v1_text.replace(old_digest, canonical_v1_digest)
    package_path_v1.write_text(
        package_v1_text,
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
