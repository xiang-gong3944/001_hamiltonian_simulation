# MPF exponent-schedule cost scaling beyond the registered range

## Purpose

For empirical MPF resource estimation, the branch count \(m\) selected by the error model can exceed the currently registered exponent-schedule range \(2\le m\le15\). The error model can still determine \(m\) and the segment count \(r\), but the per-segment cost also requires the exponent-schedule sum

\[
\boxed{K(m)\equiv\sum_{j=1}^{m} k_j.}
\]

This note records the exact registered values and a simple extrapolation for \(m>15\). The repository uses `m` as the MPF branch count; the corresponding formal MPF order is \(2m\).

## Registered schedules and sums

The current `new` schedule is optimized to reduce \(\sum_j k_j\) while keeping the MPF coefficients well conditioned. The `legacy` schedule is retained for comparison.

| \(m\) | formal order \(2m\) | \(K_{\rm new}=\sum_jk_j\) | \(K_{\rm legacy}\) | \(K_{\rm new}/(m^2\ln m)\) |
|---:|---:|---:|---:|---:|
| 2 | 4 | 3 | 3 | 1.0820 |
| 3 | 6 | 7 | 9 | 0.7080 |
| 4 | 8 | 13 | 16 | 0.5861 |
| 5 | 10 | 23 | 28 | 0.5716 |
| 6 | 12 | 32 | 37 | 0.4961 |
| 7 | 14 | 46 | 58 | 0.4824 |
| 8 | 16 | 61 | 78 | 0.4584 |
| 9 | 18 | 80 | 102 | 0.4495 |
| 10 | 20 | 102 | 128 | 0.4430 |
| 11 | 22 | 126 | 158 | 0.4343 |
| 12 | 24 | 152 | 193 | 0.4248 |
| 13 | 26 | 180 | 224 | 0.4152 |
| 14 | 28 | 213 | 271 | 0.4118 |
| 15 | 30 | 248 | 316 | 0.4070 |

Representative schedules are:

- \(m=3\): `new k = (1, 2, 4)`, \(K=7\)
- \(m=5\): `new k = (1, 2, 3, 5, 12)`, \(K=23\)
- \(m=7\): `new k = (1, 2, 3, 4, 5, 9, 22)`, \(K=46\)
- \(m=10\): `new k = (1, 2, 3, 4, 5, 6, 7, 10, 18, 46)`, \(K=102\)
- \(m=15\): `new k = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 22, 40, 104)`, \(K=248\)

## Scaling law

Well-conditioned MPF schedules have the asymptotic cost scaling

\[
K(m)=O(m^2\log m).
\]

For a practical continuation beyond the registered table, fit the available values to a one-parameter model through the origin,

\[
K(m)\approx c\,m^2\ln m.
\]

Using all registered values \(m=2,\ldots,15\):

| schedule | fitted \(c\) | RMSE in \(K\) | \(R^2\) |
|---|---:|---:|---:|
| new | **0.418254** | 4.666 | 0.996394 |
| legacy | **0.529233** | 5.426 | 0.996985 |

Thus the empirical extrapolation used for the current `new` schedule is

\[
\boxed{
K_{\rm new}(m)\approx 0.418\,m^2\ln m.
}
\]

For the legacy schedule,

\[
\boxed{
K_{\rm legacy}(m)\approx 0.529\,m^2\ln m.
}
\]

The coefficient is stable against removing the smallest-\(m\) points: for the `new` schedule, the through-origin fit gives approximately \(0.418\) over \(m=3\ldots15\), \(0.417\) over \(m=8\ldots15\), and \(0.416\) over \(m=10\ldots15\). This makes \(c\simeq0.418\) a reasonable empirical continuation coefficient rather than an artifact of the smallest schedules.

## Suggested extrapolation for \(m>15\)

For resource estimation beyond the registered schedule range, use

\[
\boxed{
\widehat K(m)
=
\left\lceil 0.418\,m^2\ln m\right\rceil,
\qquad m>15.
}
\]

The ceiling is convenient because \(K\) is an integer cost quantity. This extrapolates only the aggregate schedule cost \(K=\sum_jk_j\); it does **not** construct an actual well-conditioned exponent tuple \((k_1,\ldots,k_m)\) for \(m>15\).

Example extrapolated values:

| \(m\) | formal order | \(\widehat K_{\rm new}(m)\) | \(3\widehat K\) |
|---:|---:|---:|---:|
| 16 | 32 | 297 | 891 |
| 18 | 36 | 392 | 1176 |
| 20 | 40 | 502 | 1506 |
| 25 | 50 | 842 | 2526 |
| 30 | 60 | 1281 | 3843 |
| 40 | 80 | 2469 | 7407 |
| 50 | 100 | 4091 | 12273 |

The final column is included because the current coherent MPF/OAA resource proxy has a leading per-segment query cost proportional to approximately \(3K\). If a downstream resource model uses a different amplification construction, it should consume \(K\) itself rather than hard-coding the factor 3.

## Relation to empirical error estimation

The empirical algorithmic error model and the schedule-cost extrapolation play different roles. For a fixed branch count \(m\), the empirical MPF error is modeled as

\[
\epsilon_m
\simeq
B_{2m}(\mathcal G_N,H_{\rm loc})
\frac{T^{2m+1}}{r^{2m}}.
\]

This determines

\[
r_{\rm emp}
=
\left\lceil
\left(
\frac{B_{2m}T^{2m+1}}{\epsilon_{\rm alg}}
\right)^{1/(2m)}
\right\rceil.
\]

The exponent-schedule sum \(K(m)\) does not determine this empirical error scaling. Instead, once \(m\) and \(r\) have been selected, \(K(m)\) determines the cost of implementing one MPF segment.

Schematically,

\[
\boxed{
Q_{\rm total}\propto r\,K(m),
}
\]

with an additional constant factor from the chosen LCU/amplitude-amplification construction.

## Implementation recommendation

For the empirical estimator:

1. Use the exact registered `new` schedule and exact \(K(m)\) for \(2\le m\le15\).
2. For \(m>15\), use `ceil(0.418 * m**2 * log(m))` as an aggregate-cost extrapolation.
3. Mark extrapolated rows explicitly; do not report an invented exponent tuple.
4. Keep the error calibration \(B_{2m}\) separate from the schedule-cost extrapolation.
5. If actual well-conditioned schedules are later generated for \(m>15\), replace the extrapolated \(K(m)\) by their exact sums without changing the empirical error model.

## Provenance

Exact schedules are taken from `src/hamiltonian_resources/multiproduct.py` on `refactor/error-models-and-mpf`. The fitted coefficients reproduce the previous exploratory result \(K_{\rm new}\approx0.418m^2\ln m\) and \(K_{\rm legacy}\approx0.529m^2\ln m\).