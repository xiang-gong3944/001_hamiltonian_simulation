# Winner maps (commutator-only)

These winner maps use the same smooth-fitting philosophy as the empirical-only maps.

## Plot specification

- Comparison: **commutator-only** for Trotter and MPF; QSVT uses the same structural fit as before.
- Methods: **Trotter**, **MPF**, **QSVT**.
- Trotter uses the **p=4 commutator fit**.
- MPF uses a direct smooth regression to commutator-resource data.
- QSVT uses the smooth structural fit.
- x-axis: log-scale system size $N$, from 3 to 1000.
- y-axis: $\log_{10}(\epsilon)$, inverted: top $-10$, bottom $-1$.
- Background color = winner.
- Runner-up-colored hatch where $\log_{10}(R_2/R_1)\le0.10$.
- Winner-resource contours at $10^5$ through $10^{12}$.

## Trotter fit

For commutator Trotter p=4,

$$
\log R = a + b\log N + \frac14\log(1/\epsilon) + c\log(1+\log(1/\epsilon)).
$$

## MPF smooth commutator regression

The MPF commutator resources are fitted as

$$
\log R = a + b\log N + c\log L,
$$

with

$$
L=\log\frac{Ngt}{0.9\epsilon},\qquad t=N.
$$

This is intentionally a smooth envelope fit; it removes the discrete branch staircase from the comparison.

### MPF fit parameters

| Model | Metric | a | b_logN | c_logL | R^2_log | multiplicative RMSE |
|---|---|---:|---:|---:|---:|---:|
| TFIM | cnot_count | 4.0625 | 2.63195 | 2.48391 | 0.953114 | 1.73489 |
| TFIM | t_count | 6.7918 | 2.65894 | 3.08079 | 0.954273 | 1.76278 |
| Heisenberg | cnot_count | 6.40961 | 2.3041 | 2.87862 | 0.981276 | 1.3677 |
| Heisenberg | t_count | 9.17253 | 2.31501 | 3.4523 | 0.981307 | 1.38448 |

## Important caveat

The commutator MPF smooth fit is materially less accurate than the empirical MPF fit. For TFIM its multiplicative RMSE is about 1.73–1.76; for Heisenberg it is about 1.37–1.38. Therefore fine details of the MPF winner boundary should be treated as exploratory rather than precise.

## Winner shares

| Model | Metric | Winner | Fraction |
|---|---|---|---:|
| TFIM | cnot_count | Trotter | 92.77% |
| TFIM | cnot_count | MPF | 0.00% |
| TFIM | cnot_count | QSVT | 7.23% |
| TFIM | t_count | Trotter | 46.74% |
| TFIM | t_count | MPF | 0.00% |
| TFIM | t_count | QSVT | 53.26% |
| Heisenberg | cnot_count | Trotter | 85.22% |
| Heisenberg | cnot_count | MPF | 0.00% |
| Heisenberg | cnot_count | QSVT | 14.78% |
| Heisenberg | t_count | Trotter | 35.32% |
| Heisenberg | t_count | MPF | 0.00% |
| Heisenberg | t_count | QSVT | 64.68% |
