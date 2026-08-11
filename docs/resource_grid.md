# Resource-grid runner

`hamiltonian-resource-grid` evaluates a Cartesian grid of Hamiltonian models,
system sizes, target errors, and resource-estimation methods. Expensive work is
checkpointed by `(model, N)`, while the merged CSV remains directly usable with
pandas and Jupyter.

## Presets

The three presets use the same execution path:

| Preset | System sizes | `log10(target_error)` |
| --- | --- | --- |
| `sanity-low` | `3, 4, 6` | `-1, -2` |
| `sanity-high` | `100, 120` | `-3, -4` |
| `full` | every integer from `3` through `120` | 31 values from `-1.0` through `-4.0` in `-0.1` steps |

All presets include open-chain TFIM `(coupling=1, field=3)`, open Heisenberg
`(coupling=1, field_z=0.3)`, evolution time `t=N`, and synthesis fraction
`0.1`. The nine series are empirical and commutator Trotter at orders 2, 4,
and 6; Jacobi--Anger QSVT; and empirical and refined-commutator MPF with the
Mizuta Theorem-6 dynamic branch-count policy.

Start with a sanity run:

```powershell
hamiltonian-resource-grid run `
  --preset sanity-low `
  --output outputs/resource_grid/sanity-low `
  --workers 4 `
  --progress
```

Run or resume the full grid with:

```powershell
hamiltonian-resource-grid run `
  --preset full `
  --output outputs/resource_grid/full `
  --workers 4 `
  --resume `
  --progress
```

Omit `--resume` for the first run. Without it, the command refuses to reuse a
nonempty output directory. `--workers 0` automatically selects at most four
outer shard processes. Inner commutator multiprocessing is disabled while
outer shards run, preventing nested oversubscription.

## Custom configuration

`--config` accepts either the resource-grid object directly or under a single
`resource_grid` key. It is mutually exclusive with `--preset`.

```json
{
  "resource_grid": {
    "models": [
      {
        "model": "transverse_field_ising",
        "parameters": {"coupling": 1.0, "field": 3.0, "periodic": false}
      }
    ],
    "system_sizes": [3, 4, 6],
    "log10_target_errors": [-1.0, -2.0],
    "methods": [
      {"family": "trotter", "order": 2},
      {"family": "trotter", "order": 2, "error_policy": "empirical-operator-norm"},
      {"family": "qsvt"}
    ],
    "time": {"mode": "proportional", "coefficient": 1.0},
    "synthesis_error_fraction": 0.1,
    "trotter_partition": "auto"
  }
}
```

The equivalent Python API is `ResourceGridConfig`, `resource_grid_preset`,
`expand_resource_grid`, `evaluate_resource_grid_shard`, and
`run_resource_grid`.

## Output and status semantics

```text
outputs/resource_grid/full/
  manifest.json
  validation.json
  transverse_field_ising/N003.csv
  ...
  heisenberg_chain/N120.csv
  resource_grid.csv
```

Each shard contains every requested target error and method for one Hamiltonian.
Rows use these statuses:

- `ok`: resources and resolved parameters were produced.
- `missing_empirical`: the existing `UnsupportedEmpiricalCalibrationError`
  reports that no reviewed calibration covers the point. The row and resolved
  dynamic MPF branch count are retained.
- `error`: an unexpected estimator failure or a non-rigorous result from a
  commutator-labelled series.

Expected empirical gaps do not make the command fail. Unexpected failures are
written for diagnosis, make `validation.json` report `valid=false`, and cause
exit status 1. Configuration, manifest, checksum, or I/O errors use exit status
2.

Successful rows preserve T/CNOT counts, `trotter_reps`, `mpf_branch_count`,
`mpf_segments`, `qsvt_degree`, rigorous-bound provenance, exact/locality
fallback metadata, and empirical extrapolation flags. MPF resource counts are
not required to be monotone across dynamic branch-count transitions.

## Resume guarantees

The manifest records the canonical configuration and digest, resource-grid
schema, line-ending-stable estimator source digest, software provenance,
expected shard inventory, checksums, row counts, and status counts. Resume:

- skips only complete shards whose checksum and full point/method key set match;
- adopts a valid atomic shard written immediately before an interrupted
  manifest update;
- retries shards containing unexpected failures;
- rejects incompatible configuration, source, environment, schema, or altered
  completed shard content.

Shard CSVs and the manifest are written through same-directory temporary files
and atomically replaced. The merged CSV is rebuilt deterministically after all
shards are available.

## Viewing and distributed execution

Open `notebooks/resource_grid_viewer.ipynb` after `resource_grid.csv` exists.
The notebook only loads and plots raw results; it never calls an estimator.
Set the `RESOURCE_GRID_DATA` environment variable to view a non-default CSV.

No GitHub Actions workflow is included. The module-level `ResourceGridShard`
and shard evaluator are picklable, so a future matrix can evaluate one
`(model,N)` per job, upload the CSVs, then validate and merge them in a final
job without changing local execution architecture.

## Troubleshooting

- Run `sanity-low` first to confirm dependencies, output permissions, empirical
  missing-row handling, and the complete method path.
- Use `sanity-high` before the full run to exercise strict-error, high-N
  fallback behavior.
- If resume reports a compatibility mismatch, use a new output directory. Do
  not copy old shards into a new manifest.
- If a completed checksum fails, preserve the directory for diagnosis and
  rerun into a fresh output path rather than silently trusting the altered CSV.
