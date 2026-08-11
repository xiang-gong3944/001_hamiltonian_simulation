# TFIM high-order MPF pilot v1

## Purpose

This pilot validates the arbitrary-precision convention and measures the cost
of extending the empirical operator-norm study.  It is not a reviewed runtime
calibration and does not establish a size law.

The committed reduced artifact contains 36 observations for formal orders
18, 24, and 30 at system sizes 4, 5, and 6.  Every observation converged under
two or more working precisions.  Four separate mpmath/FLINT comparisons agree
to relative differences below `1e-64`.

## Refined asymptotic windows

The first coarse scan revealed visible subleading drift, so the final pilot
uses these dimensionless segment windows:

| Formal order | Branch count | r/T values |
|---:|---:|---|
| 18 | 9 | 3, 4, 5, 6 |
| 24 | 12 | 5, 6, 8, 10 |
| 30 | 15 | 10, 12, 16, 20 |

For `N=5`, formal order 30 required a further override to
`r/T = 20, 24, 32, 40` before the 2% median-absolute-deviation gate passed.

All nine `(N, J)` groups satisfy the pilot order criterion (each local
exponent lies within 2% of `2J`) and the four-point coefficient criterion
(maximum deviation from the median is below 5%).

Representative median coefficients are:

| N | B_18(N) | B_24(N) | B_30(N) |
|---:|---:|---:|---:|
| 4 | 2.0401e-7 | 8.3364e-13 | 3.7696e-18 |
| 5 | 3.5919e-7 | 3.0984e-12 | 4.1419e-18 |
| 6 | 4.7471e-7 | 6.0609e-12 | 1.3547e-17 |

The supplied `N=4` exploratory values are reproduced.  The supplied
`N=5, J=15, r=30` error is also reproduced, but the refined smaller-step
window yields a lower median `B_30(5)`, demonstrating why a multi-point
plateau is required.

## Backend decision

FLINT is selected for bulk dense work.  At the reference points it is about
18--22 times faster than mpmath, while agreeing far beyond the required
`1e-10` relative threshold.  FLINT's approximate eigensolver was needed for
the highly clustered eigenvalues of `D^dagger D`; those observations are
accepted only through cross-precision convergence and mpmath reference
checks, not as rigorous interval certificates.

## Cost and finite-size conclusion

Total FLINT wall time for the 12 observations at each size was approximately:

| N | Wall time |
|---:|---:|
| 4 | 2.7 s |
| 5 | 20.3 s |
| 6 | 256 s |

The observed growth makes an uncached dense sweep through `N=10`
impractical.  Before extending the matrix, the implementation must add task
resumption, reuse Hamiltonian/term/exact-evolution data within a task, and
apply the verified parity block reduction.  No fit over `N=4,5,6` is eligible
for a schema or reviewed-runtime decision.

The pilot supports retaining affine, pure-power, and power-plus-offset as
candidate analyses, but it does not justify choosing among them.  The public
v2 representation therefore needs a tagged model interface, while actual
record promotion remains gated on `N=9,10` holdouts, shifted windows, and the
planned high-order `N=11,12` checks when feasible.

## Reproduction

Run:

```powershell
.\.venv\Scripts\python.exe scripts/run_high_precision_pilot.py `
  --config docs/calibration_data/tfim_high_order_pilot_v1_config.json `
  --output docs/calibration_data/tfim_high_order_pilot_v1_reduced.json
```

The task configuration and reduced observation list are canonically hashed in
the output.  No per-task matrix or precision-attempt shards are committed.
