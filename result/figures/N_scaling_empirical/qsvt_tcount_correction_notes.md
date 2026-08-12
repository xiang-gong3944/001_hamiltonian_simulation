# QSVT T-count correction used in empirical plots

This package replaces the earlier QSVT T-count fit, which depended only on `(N, degree)`, with a structure-aware smooth model.

## Why the old fit was wrong

For QSVT, `rotation_count` and `toffoli_count` stay constant on Jacobi–Anger plateaus, while `t_count` continues to change with `epsilon` because the per-rotation synthesis budget tightens.
Therefore a fit of the form `log T = a + b log N + c log(d+1)` is insufficient.

## New smooth QSVT model

1. Use the previously fitted smooth degree model

   `d_smooth = c0 + c1 * alpha(N) * t + c2 * log(1/eps)/log log(1/eps)` with `t=N`.

2. Fit structural counts as smooth functions of `(N, d)`:

   `log C = a + b log N + c log(d+1)`

   for `rotation_count`, `toffoli_count`, `cnot_count`, and `query_count`.

3. Reconstruct a smooth T count using the same synthesis-cost model as the codebase, but without the outer `ceil`:

   `delta_rot = 0.1 * eps / rotation_count`

   `T_rot_smooth = 3 log2(1/delta_rot) + log2(log2(1/delta_rot))`

   `T_QSVT_smooth = rotation_count * T_rot_smooth + 4 * toffoli_count`.

The factor `0.1 * eps` comes from the synthesis budget allocation used in the benchmark model.

## Structural-fit quality

| Model | Target | R^2(log) | multiplicative RMSE |
|---|---|---:|---:|
| TFIM | rotation_count | 0.994134 | 1.20788 |
| TFIM | toffoli_count | 0.999714 | 1.04557 |
| TFIM | cnot_count | 0.999608 | 1.05325 |
| TFIM | query_count | 0.999999 | 1.00145 |
| Heisenberg | rotation_count | 0.994411 | 1.20485 |
| Heisenberg | toffoli_count | 0.999765 | 1.04168 |
| Heisenberg | cnot_count | 0.999687 | 1.048 |
| Heisenberg | query_count | 0.999997 | 1.00313 |

## Contents

- `fixed_N20_eps_sweeps/`: corrected empirical `N=20` epsilon sweeps
- `fixed_eps_n_sweeps/`: corrected empirical `epsilon=1e-3` N sweeps
- `winner_maps/`: empirical winner maps recomputed with corrected QSVT T count
- `qsvt_corrected_fit_parameters.csv`: degree and structural-count fit parameters
