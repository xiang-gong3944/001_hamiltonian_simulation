# Analytical resource-scaling benchmarks

The benchmark suite compares the repository's analytical T-count and CNOT-count
models for eight fixed Hamiltonian-simulation configurations:

- Trotter formulas with orders `p = 1, 2, 4, 6`;
- multiproduct formulas with `m = 3, 5, 7` terms;
- QSVT Hamiltonian simulation.

Every configuration is evaluated independently. The generator never chooses a
best Trotter order or MPF term count. Optional best-of-family curves are derived
later from the persisted raw rows.

## Configuration

[`benchmark_config.json`](../benchmark_config.json) is the default configuration.
Relative output paths are resolved relative to the JSON file, so the checked-in
configuration writes to `benchmark_outputs/` at the repository root.

| Setting | Meaning |
| --- | --- |
| `hamiltonian_model` | `transverse_field_ising` or `heisenberg_chain`. |
| `model_parameters` | Keyword arguments for the selected repository model constructor. |
| `system_qubit_values` | Values varied by the system-size sweep. |
| `target_error_values` | Values varied by the target-error sweep. |
| `evolution_time` | Fixed physical evolution time `t` in both sweeps. |
| `fixed_system_qubits_for_error_sweep` | System size held fixed while varying error. |
| `fixed_target_error_for_size_sweep` | Error held fixed while varying system size. |
| `synthesis_error_fraction` | Fraction of total error reserved for rotation synthesis. |
| `trotter_partition` | Existing `auto`, `individual`, or `commuting` partition policy. |
| `mpf_schedule` | Existing `new` or `legacy` MPF exponent table. |
| `output_directory` | Destination for CSV, metadata, and figures. |
| `output_formats` | Any nonempty combination of `png`, `pdf`, and `svg`. |
| `generate_summary_plots` | Whether plotting also writes best-of-evaluated family curves. |

The default is the open transverse-field Ising chain with `J=1`, `h=3`,
`t=1`, system sizes 2 through 12, and target errors from `1e-1` through
`1e-3`. These ranges are intended to run conveniently with analytical models;
they can be extended without constructing exact matrices.

## Commands

After installing the project in editable mode, generate the two sweeps separately:

```powershell
hamiltonian-benchmark generate --config benchmark_config.json --sweep system-size
hamiltonian-benchmark generate --config benchmark_config.json --sweep target-error
```

Generate both sweeps in one invocation:

```powershell
hamiltonian-benchmark generate --config benchmark_config.json --sweep all
```

The equivalent module form is useful before reinstalling the console entry point:

```powershell
python -m hamiltonian_resources.benchmark_cli generate --config benchmark_config.json
```

Plotting always reloads saved CSV data and does not invoke a resource estimator:

```powershell
hamiltonian-benchmark plot --data benchmark_outputs/system_size_scaling.csv
hamiltonian-benchmark plot --data benchmark_outputs/target_error_scaling.csv
hamiltonian-benchmark plot --data-dir benchmark_outputs --sweep all
```

`run` is a convenience command that generates CSVs, reloads them, and then plots:

```powershell
hamiltonian-benchmark run --config benchmark_config.json
```

Enable or disable summary figures without changing stored data by passing
`--summary` or `--no-summary` to `plot`/`run`. `--formats png pdf svg` overrides
the formats stored in the metadata sidecar.

The generator writes all requested rows before returning. Exit status `1` means
one or more method evaluations failed, but the CSV and successful results were
still saved. Exit status `2` means the configuration or command itself was invalid.

## Outputs and plot conventions

Data files are `system_size_scaling.csv` and `target_error_scaling.csv`. Each has
a same-stem `.metadata.json` file containing the resolved configuration, column
order, status counts, timestamp, software versions, and Git state.

The required figures are:

- `system_size_t_count.*` and `system_size_cnot_count.*`;
- `target_error_t_count.*` and `target_error_cnot_count.*`.

Resource axes are logarithmic. The physical error `epsilon` is retained on a
logarithmic x-axis and reversed, so precision increases from left to right.
Trotter curves share blue, MPF curves share vermillion, and QSVT uses green;
line styles and markers distinguish configurations within a family. The same
mapping is used in every figure.

Summary filenames end in `_summary`. At each x value and separately for each
resource metric, they plot the minimum successful Trotter row among
`p = 1, 2, 4, 6`, the minimum successful MPF row among `m = 3, 5, 7`, and QSVT.
They are labeled as best only among those evaluated configurations.

Failed rows are never dropped from CSV output. Plots use available positive
counts, issue a warning, and annotate the number of failures or unplottable
values. A method with no successful points remains in the legend and is named
in the annotation.

## CSV schema

Empty fields mean that a quantity does not apply to that method, or that the
evaluation failed. Successful rows always contain both resource counts.

| Column | Meaning |
| --- | --- |
| `schema_version` | Version of the ordered CSV schema. |
| `run_id` | UUID for this generated sweep. |
| `generated_at_utc` | ISO-8601 UTC generation time. |
| `config_digest` | SHA-256 digest of the resolved configuration. |
| `sweep` | `system-size` or `target-error`. |
| `hamiltonian_model` | Registered model constructor name. |
| `hamiltonian_name` | Name attached to the constructed Hamiltonian. |
| `model_parameters_json` | Deterministically serialized constructor parameters. |
| `system_qubits` | Number of physical system qubits. |
| `evolution_time` | Physical simulation time. |
| `target_error` | Total requested simulation error. |
| `hamiltonian_alpha` | Pauli-LCU coefficient 1-norm `alpha`. |
| `hamiltonian_term_count` | Number of nonzero Pauli terms. |
| `method_family` | `trotter`, `multiproduct`, or `qsvt`. |
| `method_label` | Stable legend label for the fixed configuration. |
| `trotter_order` | Product-formula order, otherwise empty. |
| `mpf_term_count` | Number of MPF terms `m`, otherwise empty. |
| `mpf_formal_order` | Formal MPF order `2m`, otherwise empty. |
| `segment_count` | Trotter repetitions or MPF time segments. |
| `query_count` | Amplified MPF controlled-`S2` or QSVT block-encoding queries. |
| `qsvt_degree` | Larger Jacobi--Anger component degree. |
| `trotter_partition` | Resolved product-formula partition. |
| `trotter_group_count` | Number of resolved Suzuki summands/groups. |
| `bound_value` | Selected bound or proxy value at the chosen parameter. |
| `bound_prefactor` | Trotter or MPF proxy prefactor when applicable. |
| `bound_method` | Bound/proxy selection method. |
| `bound_rigorous` | Whether the reported selection rule is rigorous. |
| `algorithm_error_budget` | Error left after reserving rotation-synthesis error. |
| `mpf_schedule` | MPF exponent-table name. |
| `mpf_exponents_json` | Ordered MPF exponent list. |
| `mpf_coefficients_json` | Ordered linearly combined coefficients. |
| `mpf_coefficient_l1_norm` | Coefficient 1-norm before padding. |
| `mpf_padding_weight` | Cancelling identity weight used to reach normalization two. |
| `lcu_normalization` | Pre-amplification LCU normalization. |
| `amplitude_amplification` | Amplification strategy. |
| `amplitude_amplification_rounds` | Total robust-OAA rounds in the estimate. |
| `good_subspace` | Ancilla condition defining success. |
| `nominal_success_probability` | Post-amplification nominal success probability. |
| `total_qubits` | System plus estimator-model ancilla qubits. |
| `rotation_count` | Non-Clifford rotation slots in the analytical model. |
| `toffoli_count` | Temporary-AND compute/uncompute pairs. |
| `depth` | Depth when available; analytical rows currently use `-1`. |
| `t_count` | Estimated Clifford+T synthesis cost. |
| `cnot_count` | Estimated CNOT count. |
| `counting_mode` | Resource-counting implementation, currently `analytical-model`. |
| `rotation_synthesis_error` | Total error allocated to synthesized rotations. |
| `package_version` | Installed `hamiltonian-resources` version. |
| `python_version` | Python interpreter version. |
| `qiskit_version` | Installed Qiskit version. |
| `git_commit` | Source Git commit, or `unknown`. |
| `git_dirty` | Whether tracked source changes existed during generation. |
| `status` | `ok` or `error`. |
| `error_type` | Exception class for a failed row. |
| `error_message` | Useful failure description. |

## Analytical assumptions and limitations

1. The total error is split into algorithmic and single-qubit rotation-synthesis
   portions. Generic rotations use the package's documented ancilla-free T-cost
   approximation; temporary-AND costs are also included.
2. Trotter orders 1 and 2 use rigorous Childs et al. commutator bounds. Orders
   4 and 6 use the rigorous Schubert--Mendl bound within the practical group cap;
   larger partitions fall back to `alpha-proxy` and explicitly report
   `bound_rigorous=false`.
3. MPF segment selection uses the existing
   `alpha_eff = min(alpha, W2^(1/3))` higher-order proxy. It is calibrated to the
   second-order commutator bound but is not a certified MPF error bound.
4. QSVT degree selection uses the rigorous Jacobi--Anger truncation baseline.
   The cost model assumes efficient controlled-response compilation that shares
   block-encoding queries between `V` and `V^dagger`; generic Qiskit `.control()`
   transpilation is substantially more expensive.
5. MPF and QSVT counts include one three-step robust OAA round per MPF segment or
   QSVT circuit. Their saved nominal success probability is therefore one.
6. Multi-control CNOT costs depend on architecture, ancilla availability, and
   compiler choices. The saved values use the repository's explicit comparison
   model rather than a hardware-specific compilation.
7. The scaling generator does not form dense Hamiltonian matrices. Use
   `compare_with_exact` only as a separate small-system calibration.
