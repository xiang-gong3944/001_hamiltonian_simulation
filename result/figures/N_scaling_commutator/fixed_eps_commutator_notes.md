# Fixed-ε N-sweep (commutator-only)

This package contains the fixed-ε N-sweep figures for the **commutator-only** comparison among **Trotter**, **MPF**, and **QSVT**.

## Plot specification

- Fixed error: $\log_{10}(\epsilon)=-3$ (that is, $\epsilon=10^{-3}$)
- Horizontal axis: system size $N$
- Range: $N\le150$
- Both axes are logarithmic.
- **Lines** denote smooth fitted curves.
- **Markers** denote discrete original data points.
- Methods:
  - **Trotter**: commutator estimator, **p=4 only**
  - **MPF**: commutator estimator, smooth regression fit
  - **QSVT**: structural smooth fit, with discrete data points also shown

## Fit models

### Trotter (commutator, p=4)

$$
\log R = a + b\log N + \frac{1}{4}\log(1/\epsilon) + c\log\bigl(1+\log(1/\epsilon)\bigr).
$$

### MPF (commutator, smooth regression)

$$
\log R = a + b\log N + c\log L,
$$

with

$$
L = \log\frac{Ngt}{0.9\epsilon},\qquad t=N.
$$

### QSVT

The degree fit is

$$
d = c_0 + c_1(\alpha t) + c_2\frac{\log(1/\epsilon)}{\log\log(1/\epsilon)},
$$

with $t=N$ and linear $\alpha(N)$, then the resource fit is

$$
\log R = a + b\log N + c\log(d+1).
$$

## MPF commutator fit quality

| Model | Metric | a | b_logN | c_logL | R^2_log | multiplicative RMSE |
|---|---|---:|---:|---:|---:|---:|
| TFIM | cnot_count | 4.0625 | 2.63195 | 2.48391 | 0.953114 | 1.73489 |
| TFIM | t_count | 6.7918 | 2.65894 | 3.08079 | 0.954273 | 1.76278 |
| Heisenberg | cnot_count | 6.40961 | 2.3041 | 2.87862 | 0.981276 | 1.3677 |
| Heisenberg | t_count | 9.17253 | 2.31501 | 3.4523 | 0.981307 | 1.38448 |

## Caveat

The commutator MPF smooth fit is visibly rougher than the empirical MPF smooth fit. Its multiplicative RMSE is about 1.73–1.76 for TFIM and about 1.37–1.38 for Heisenberg.
