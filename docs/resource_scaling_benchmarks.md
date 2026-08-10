# Analytical resource-scaling benchmarks

The schema-2 benchmark API is designed for interactive use. A mutable
`BenchmarkConfig` accepts Python sequences, NumPy arrays, or pandas indexes,
evaluates only the requested method specifications, and returns one in-memory
DataFrame containing the requested sweeps. It never writes files implicitly.

## Single-point evaluation

Use the plan/report API when a sweep and DataFrame are unnecessary:

```python
from hamiltonian_resources import (
    MultiproductMethod,
    build_simulation_circuit,
    compare_plan_with_exact,
    estimate_resources,
    transverse_field_ising,
)

hamiltonian = transverse_field_ising(2, field=0.7)
report = estimate_resources(
    hamiltonian,
    MultiproductMethod(3),
    time=0.2,
    target_error=1e-3,
)

print(report.selected_parameters)
print(report.logical_counts.as_dict())
print(report.resources.as_dict())

reference = build_simulation_circuit(report.plan)
validation = compare_plan_with_exact(report.plan)
```

`report.plan` is backend-independent and owns the selected parameters, error
metadata, and logical operation schedule. `report.resources` and
`report.resource_provenance` describe the structured analytical compilation.
Reference Qiskit circuit metadata serializes the same plan and separately names
its generic-control compilation assumptions. Temporary-AND counts, decomposed
rotations/CNOTs, and backend work qubits are never stored in the plan.

Resource planning requires positive time. The direct circuit builders retain
their existing support for finite zero and negative times.

## Python and notebook workflow

```python
import numpy as np
from hamiltonian_resources import (
    BenchmarkConfig, HamiltonianSpec, MultiproductMethod, QSVTMethod,
    TimeScaling, TrotterMethod, plot_benchmark, run_benchmark,
)

config = BenchmarkConfig(
    hamiltonian=HamiltonianSpec(
        model="transverse_field_ising",
        parameters={"coupling": 1.0, "field": 3.0, "periodic": False},
    ),
    system_sizes=np.arange(2, 14, 2),
    target_errors=np.logspace(-1, -3, 5),
    time=TimeScaling("proportional", 1.0),
    fixed_system_size=8,
    fixed_target_error=1e-3,
    methods=[
        *(TrotterMethod(p) for p in (1, 2, 4, 6)),
        *(MultiproductMethod(m) for m in (3, 5, 7)),
        QSVTMethod(),
    ],
)

data = run_benchmark(config, workers=1, show_progress=True)
figure = plot_benchmark(data, sweep="system-size", metric="t_count")
```

Python and notebook calls remain serial unless `workers` is greater than one.
`show_progress=True` displays a benchmark-point bar and reuses a transient inner
display for expensive commutator work. Trotter stages have known chunk totals;
adaptive Mizuta searches instead show completed updates and the current segment
candidate without claiming a final percentage. Structured integrations can use
the existing `progress` callback for benchmark rows and `commutator_progress`
for `CommutatorProgress` events.

`MultiproductMethod(m)` uses the Hamiltonian-1-norm bound
`error_method="low2019-l1-ideal-rigorous"` by default; the historical
`low-rigorous` spelling remains an input alias. To
use rigorous finite-size W2 information, request
`MultiproductMethod(m,
error_method="childs2021-w2-triangle-ideal-rigorous")`. This estimator applies
triangle inequalities to the proven Strang W2 bound and deliberately discards
formal-order MPF cancellation. The higher-order locality-compatible
commutator alternative is
`MultiproductMethod(m,
error_method="mizuta2026-commutator-ideal-rigorous")`. This refined method
evaluates the BCH remainder directly for each \((r,p_0)\). The former printed
Theorem-3 allocation calculation is preserved as
`mizuta2026-theorem3-legacy-ideal-rigorous`. Their theorem map and
the reason that Aftab 2024 is not exposed as a finite rigorous estimator are
documented in [MPF error bounds](mpf_error_bounds.md). To reproduce historical
W2-calibrated projections only, request
`MultiproductMethod(m, error_method="legacy-w2-proxy")`; such rows remain
available in full plots but are explicitly styled as heuristic.

`error_method="best-rigorous-ideal"` is an opt-in finite-resource policy for
one fixed MPF schedule. It evaluates the rigorous Low, W2 triangle, and Mizuta
estimators, selects the one requiring fewer segments, and records all three
candidate results alongside the selected bound provenance. It never considers
the historical W2 proxy and does not optimize across different branch counts.

MPF branch count has a separate deterministic policy:

```python
# Existing fixed-J configuration; JSON and digest are unchanged.
MultiproductMethod(term_count=3, branch_count_policy="fixed")

# Resolve J once per benchmark point from Mizuta Theorem 6, Eq. (84).
MultiproductMethod(
    term_count=None,
    branch_count_policy="mizuta2026-theorem6",
    error_method="mizuta2026-commutator-ideal-rigorous",
)
```

The dynamic form uses the row's `algorithm_error_budget` and the shared
outward-rounded Pauli extensiveness \(g\), then passes the same selected
schedule to any requested Low, legacy Mizuta, refined Mizuta, or best-bound
estimator. It never minimizes segment count over candidate orders. Fixed JSON
continues to contain `term_count` and omit `branch_count_policy`; dynamic JSON
omits `term_count` and stores
`"branch_count_policy": "mizuta2026-theorem6"`.

The registered explicit schedules cover \(2\le J\le15\). Resolution may return
a larger order. Rigorous/coefficient-dependent methods then become a
fault-isolated `status="error"` row. An empirical `new`-schedule row can
continue only with an exact reviewed calibration, in which case its analytical
resource result uses aggregate \(K\) and remains explicitly non-implementable.

`TimeScaling("proportional", tau)` means `t(n) = tau * n`; the checked-in
default has `tau=1`. This gives each one-dimensional local Hamiltonian enough
simulated time to scale with the distance over which interactions spread. It is
a benchmark convention, not a computed Lieb--Robinson velocity. Use
`TimeScaling("fixed", t)` when a fixed-time comparison is intentionally wanted.
The same rule is evaluated at `fixed_system_size` during the target-error sweep.

The default system-size and target-error x-axes are logarithmic with base ten;
the target-error axis is reversed. Both axes, their bases, the
y-scale, grouping columns, and any positive numeric resource metric can be
selected through `plot_benchmark`.

Python callers may provide a custom `HamiltonianSpec.factory` together with a
stable model name. JSON jobs accept only the registered
`transverse_field_ising` and `heisenberg_chain` models.

## JSON and CLI workflow

[`benchmark_config.json`](../benchmark_config.json) contains two objects:

- `benchmark`: the same Hamiltonian, arrays, time rule, fixed values, method
  specifications, and error settings accepted by `BenchmarkConfig`;
- `output`: CLI-only root directory, formats, and summary-plot default.

```powershell
hamiltonian-benchmark generate --config benchmark_config.json --sweep all --progress
hamiltonian-benchmark run --config benchmark_config.json --summary
hamiltonian-benchmark plot --data benchmark_outputs/<run>/benchmark.csv --summary
```

`generate` and `run` choose up to four commutator worker processes automatically.
Use `--workers 1` for serial execution or `--workers N` for an explicit count.
Progress is automatic on an interactive stderr stream and can be forced or
disabled with `--progress` or `--no-progress`. Progress never uses stdout, so it
does not contaminate machine-readable output.

`generate` and `run` create
`<UTC timestamp>_<config digest>_<run id>/benchmark.csv` and `metadata.json`
under the output root. A new run never overwrites an old one. `run` also writes
standard T-count and CNOT-count figures. `plot` accepts a standalone schema-2
CSV; metadata is not required. Exit status `1` reports one or more method rows
that failed, while status `2` reports an invalid command or configuration.

## Schema 2.0

Each row is one method at one sweep point. Important identity and scaling fields
are:

| Column | Meaning |
| --- | --- |
| `schema_version` | Exactly `2.0`; schema 1.x is intentionally incompatible. |
| `sweep` | `system-size` or `target-error`. |
| `system_qubits` | Physical system size. |
| `evolution_time` | Resolved row-specific physical time. |
| `time_scaling_mode` | `proportional` or `fixed`. |
| `time_scaling_coefficient` | `tau` in `t=tau*n`, or fixed `t`. |
| `target_error` | Total simulation-error target. |
| `method_id` | Stable method identity such as `trotter-p4` or `mpf-m5`. |
| `method_family` / `method_label` | Plot grouping and human-readable name. |
| `error_policy`, `estimate_category` | Normalized sizing policy and analytical/empirical/proxy category. |
| `mpf_term_count`, `mpf_formal_order` | Resolved row-specific \(J\) and \(2J\), including dynamic rows. |
| `mpf_branch_count_policy` | `fixed` or `mizuta2026-theorem6`; missing historical schema-2 values load as `fixed`. |
| `mpf_branch_count_policy_extensiveness_g` | Outward-rounded unweighted Pauli extensiveness used by the dynamic formula. |
| `mpf_branch_count_policy_target_error` | Algorithm-error budget supplied to branch-count resolution. |
| `t_count`, `cnot_count`, `total_qubits` | Primary resource estimates. |
| `query_count`, `rotation_count`, `toffoli_count`, `depth` | Additional metrics. |
| `bound_method`, `bound_rigorous` | Parameter-selection provenance and rigor at the declared scope. |
| `bound_reference`, `bound_theorem_or_equations` | Exact paper and theorem/equation provenance. |
| `bound_components_json`, `bound_assumptions_json` | Structured bound terms and theorem assumptions. |
| `max_nested_commutator_order`, `max_exact_nested_commutator_order` | Largest order used and largest order evaluated by exact Pauli recurrence. |
| `locality_compatible`, `commutator_cap_fallback` | Whether locality is preserved and whether the explicit work cap selected a proven fallback. |
| `mpf_r_error`, `mpf_r_time_1`, `mpf_r_time_2` | Candidate-dependent MPF error and Mizuta time-condition segment thresholds. |
| `mpf_active_constraints_json` | Every threshold active at the selected segment count, including ties. |
| `mpf_mu_upper`, `mpf_truncation_order_p0` | Selected finite Mizuta commutator inputs. |
| `mpf_auxiliary_error`, `mpf_auxiliary_allocation_fraction` | Printed-Theorem-3 legacy allocation; `None` for the refined estimator. |
| `mpf_refined_lemma9_remainder`, `mpf_refined_lemma10_remainder`, `mpf_total_branchwise_bch_remainder` | Refined order-resolved BCH components. |
| `mpf_local_step_error`, `mpf_repeated_global_error`, `mpf_local_error_dominance` | Refined local/repeated totals and commutator-vs-BCH dominance. |
| `mpf_legacy_first_time_limit`, `mpf_legacy_first_condition_passed`, `mpf_second_time_limit` | Reported printed first condition and active second time limit. |
| `mpf_schedule_weights_json`, `mpf_schedule_weighted_extensiveness` | Actual Suzuki occurrence weights and derived \(g_\alpha\). |
| `mpf_exact_commutator_cutoff`, `mpf_locality_fallback`, `mpf_refined_tail_fallback_status` | Exact-order cutoff and locality/tail fallback provenance. |
| `bound_scope`, `bound_target_satisfied` | Scope of the bound and whether a rigorous bound meets the algorithmic budget. |
| `circuit_bound_scope`, `circuit_bound_rigorous` | Implemented-circuit scope and its separate certification status. |
| `circuit_target_satisfied` | Whether the rigorous claim at `circuit_bound_scope` meets the target; inspect that scope rather than inferring full joint-unitary certification. |
| `empirical_calibration_id`, `empirical_calibration_model` | Exact reviewed row and model identity, when empirical sizing is selected. |
| `empirical_calibration_size_min`, `empirical_calibration_size_max`, `empirical_calibration_time_min`, `empirical_calibration_time_max` | Reviewed finite calibration domain. |
| `empirical_size_extrapolated`, `empirical_time_extrapolated`, `empirical_active_constraint` | Extrapolation flags and formula/domain guard provenance. |
| `mpf_exponent_sum`, `mpf_exponent_sum_source`, `mpf_explicit_schedule_available` | Aggregate \(K\), exact/extrapolated provenance, and circuit-schedule capability. |
| `status`, `error_type`, `error_message` | Per-method failure information. |

The CSV retains detailed Trotter, MPF, QSVT, synthesis, software, and Git
metadata columns. Validation requires the schema-2 core columns and valid row
states, but permits reordered columns and user-added derived columns. Loading
an early schema-2 CSV without the scoped-bound extension columns fills them
conservatively; historical MPF rows are never upgraded to circuit-rigorous.

`compare_mpf_bounds(data, metric="segment_count")` pairs Low and Mizuta rows
only when their Hamiltonian, time, target, branch-count policy, selected branch
count, schedule, and synthesis allocation agree. A fixed row cannot cross-pair
with a dynamic row that happens to select the same \(J\).
`plot_mpf_crossover` groups dynamic curves by policy and schedule, annotates
points with \(J\) and \(2J\), and shows the active Mizuta constraint and
commutator fallback explicitly.

`select_best_by_family(data, metric=..., sweep=...)` uses
`certification_policy="implemented-circuit"` by default. It therefore excludes
current MPF rows because their Low/Mizuta guarantees apply to the ideal MPF,
not the complete shared-ancilla robust-OAA circuit. Use
`certification_policy="declared-bound-scope"` to include rigorous ideal-MPF
rows with an explicit circuit-unproven label, or `"unconstrained"` to include
heuristics and empirical non-certified rows. The old `rigorous_only` argument is a deprecated compatibility
alias. Plot titles record the selected policy, and summaries retain
`selected_method_id` and `selected_method_label`.

## Analytical assumptions

1. The total error is divided into algorithmic and rotation-synthesis portions.
   Generic rotation and temporary-AND T costs are both included.
2. Trotter orders 1 and 2 use rigorous commutator bounds. Higher supported
   orders use the Schubert--Mendl bound within the practical group cap and
   report an explicit nonrigorous fallback otherwise.
3. MPF segment selection defaults to the Low--Kliuchnikov--Wiebe bound using
   Eqs. (14)--(15), with `lambda=sum_j ||h_j||=hamiltonian.alpha` for the
   individual Pauli decomposition and the registered schedule's coefficient
   1-norm; Eq. (16) supplies only its upper search bracket. The opt-in Mizuta
   refined method uses Theorem 4, Eqs. (47)--(49), with the direct Lemma-9 and
   Lemma-10 BCH remainder, exact finite-order Pauli commutators, and a proven
   locality fallback. It enumerates \(p_0\) directly. The explicit legacy
   identifier retains Theorem 3, Eqs. (33)--(35), optimizes the auxiliary
   allocation over every order allowed by the first time hypothesis, and
   keeps the fixed 50/50 audit option.
4. QSVT degree selection uses rigorous Jacobi--Anger parity-tail bounds. The
   exact scaled polynomial and ideal cubic-OAA block have separate derived
   claims; floating `pyqsp` phase residuals remain finite-grid observations,
   so the constructed QSVT circuit is not certified by those claims.
5. MPF resource counts include the implemented robust-OAA shared-ancilla
   circuit. Low or Mizuta supplies the ideal/local MPF claim; the exact OAA
   identity and Gilyén--Su--Low--Wiebe reused-ancilla product bound separately
   produce a conservative claim for `repeated-shared-ancilla-good-block`.
   This claim can be rigorous while failing the requested target. The counter
   charges controlled product formulas only to physical MPF branches, sign
   phases only to negative coefficients and the negative identity-padding
   branch, and counts three SELECT, six PREPARE/inverse-PREPARE, and two
   reflection calls per segment. Multi-control CNOT costs remain
   architecture-dependent.
6. The benchmark constructs neither dense Hamiltonian matrices nor concrete
   circuits. Use `compare_with_exact` separately for small-system calibration.
7. Empirical policies size from reviewed spectral operator-norm observations,
   create no mathematical claim, and keep synthesis error separate. Missing or
   mismatched calibrations fail closed. Aggregate-only MPF counts conditionally
   retain the existing branch-addressing/OAA architecture without implying an
   explicit schedule or constructible circuit.
