# Analytical resource-scaling benchmarks

The schema-2 benchmark API is designed for interactive use. A mutable
`BenchmarkConfig` accepts Python sequences, NumPy arrays, or pandas indexes,
evaluates only the requested method specifications, and returns one in-memory
DataFrame containing the requested sweeps. It never writes files implicitly.

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

data = run_benchmark(config)
figure = plot_benchmark(data, sweep="system-size", metric="t_count")
```

`MultiproductMethod(m)` uses
`error_method="low2019-l1-ideal-rigorous"` by default. The historical
`low-rigorous` spelling remains an input alias. To
reproduce historical W2-calibrated projections, request
`MultiproductMethod(m, error_method="legacy-w2-proxy")`. Such rows remain
available in full plots but are explicitly styled as heuristic and are not
eligible for rigorous best-of-family summaries.

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
hamiltonian-benchmark generate --config benchmark_config.json --sweep all
hamiltonian-benchmark run --config benchmark_config.json --summary
hamiltonian-benchmark plot --data benchmark_outputs/<run>/benchmark.csv --summary
```

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
| `t_count`, `cnot_count`, `total_qubits` | Primary resource estimates. |
| `query_count`, `rotation_count`, `toffoli_count`, `depth` | Additional metrics. |
| `bound_method`, `bound_rigorous` | Parameter-selection provenance and rigor at the declared scope. |
| `bound_scope`, `bound_target_satisfied` | Scope of the bound and whether a rigorous bound meets the algorithmic budget. |
| `circuit_bound_scope`, `circuit_bound_rigorous` | Implemented-circuit scope and its separate certification status. |
| `circuit_target_satisfied` | Whether the complete implemented circuit has a rigorous target-error guarantee. |
| `status`, `error_type`, `error_message` | Per-method failure information. |

The CSV retains detailed Trotter, MPF, QSVT, synthesis, software, and Git
metadata columns. Validation requires the schema-2 core columns and valid row
states, but permits reordered columns and user-added derived columns. Loading
an early schema-2 CSV without the scoped-bound extension columns fills them
conservatively; historical MPF rows are never upgraded to circuit-rigorous.

`select_best_by_family(data, metric=..., sweep=...)` returns the pointwise
minimum rigorously target-satisfying row by default and adds
`selected_method_id` and `selected_method_label`. Pass `rigorous_only=False`
only for an explicitly unconstrained comparison. Consequently, a summary
never hides which evaluated Trotter order or MPF term count was selected and
does not silently select a heuristic row.

## Analytical assumptions

1. The total error is divided into algorithmic and rotation-synthesis portions.
   Generic rotation and temporary-AND T costs are both included.
2. Trotter orders 1 and 2 use rigorous commutator bounds. Higher supported
   orders use the Schubert--Mendl bound within the practical group cap and
   report an explicit nonrigorous fallback otherwise.
3. MPF segment selection defaults to
   [Low--Kliuchnikov--Wiebe](https://arxiv.org/abs/1907.11679) Eq. (16), with
   `lambda=sum_j ||h_j||=hamiltonian.alpha` for the individual Pauli
   decomposition and the registered schedule's coefficient 1-norm. The saved
   error evaluates their Eqs. (14)--(15). This is a rigorous operator-norm
   bound for the ideal repeated MPF `M(t/r)^r`.
4. QSVT degree selection uses the Jacobi--Anger truncation baseline and assumes
   efficient controlled-response compilation.
5. MPF resource counts include the implemented robust-OAA shared-ancilla
   circuit, but the Low theorem does not by itself certify that complete
   circuit. MPF rows therefore use `bound_scope="ideal-mpf"` while
   `circuit_bound_scope="amplified-shared-ancilla"` and
   `circuit_bound_rigorous=False`. Multi-control CNOT costs remain
   architecture-dependent.
6. The benchmark constructs neither dense Hamiltonian matrices nor concrete
   circuits. Use `compare_with_exact` separately for small-system calibration.
