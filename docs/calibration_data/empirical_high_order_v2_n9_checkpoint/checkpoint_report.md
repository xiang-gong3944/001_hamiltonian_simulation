# High-order empirical MPF calibration: N=9 checkpoint

This is a fail-closed checkpoint. No N=10 task was launched, no acceptance criterion was relaxed, and no row is eligible for `reviewed` status because the required N=6..10 shifted window and N=10 holdout are unavailable.

## Gate status

Rows passing all currently required numerical and time-law gates: TFIM 2J=22, TFIM 2J=24, TFIM 2J=28, TFIM 2J=30.

The following rows pass every N=4..9 numerical window but are not gate-complete: Heisenberg 2J=22, Heisenberg 2J=26, Heisenberg 2J=28. The Heisenberg N=8 sentinel failure requires the predeclared all-order N=8 time-law expansion before these rows can pass.

Rows removed before the fit checkpoint:

| Model | 2J | Missing or rejected primary sizes | Time-law status |
|---|---:|---|---|
| TFIM | 18 | 9 | passed |
| TFIM | 20 | 9 | passed |
| TFIM | 26 | 8, 9 | failed |
| Heisenberg | 18 | 8, 9 | failed |
| Heisenberg | 20 | 8, 9 | failed |
| Heisenberg | 24 | 7, 8, 9 | failed |
| Heisenberg | 30 | 8, 9 | failed |

## B_2J(N) observations

| Model | 2J | N=4 | N=5 | N=6 | N=7 | N=8 | N=9 | Time gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TFIM | 22 | 7.20767792e-11 | 1.95590125e-10 | 3.25988415e-10 | 4.28139741e-10 | 4.92674358e-10 | 5.30163124e-10 | passed |
| TFIM | 24 | 8.33638218e-13 | 3.09837757e-12 | 6.06093712e-12 | 8.71748041e-12 | 1.06471902e-11 | 1.18326324e-11 | passed |
| TFIM | 28 | 2.15069890e-16 | 4.87579071e-16 | 1.39755242e-15 | 2.58395141e-15 | 3.65685946e-15 | 4.48043104e-15 | passed |
| TFIM | 30 | 3.76956567e-18 | 4.14188882e-18 | 1.35473042e-17 | 2.78968282e-17 | 4.35078142e-17 | 5.65840603e-17 | passed |
| Heisenberg | 22 | 1.32702649e-16 | 1.38534265e-15 | 2.47807457e-15 | 1.43986097e-14 | 2.44487985e-14 | 2.67840506e-14 | pending-required-N8-expansion |
| Heisenberg | 26 | 3.77433660e-21 | 4.91152059e-20 | 3.65176204e-19 | 4.67664882e-19 | 1.96436109e-18 | 3.89968443e-18 | pending-required-N8-expansion |
| Heisenberg | 28 | 1.16288238e-23 | 3.32140677e-22 | 1.32468707e-21 | 4.58274544e-21 | 6.17979748e-21 | 2.56198930e-20 | pending-required-N8-expansion |

Fits use the configured weighted log-residual objective. Relative weights combine precision convergence, accepted plateau spread, and accepted time-law spread in quadrature.

## Candidate fits using N=4..8 with N=9 as the available holdout

| Model | 2J | Law | Parameters | AICc | N=9 error | B(50) | B(100) | Two-window necessary conditions |
|---|---:|---|---|---:|---:|---:|---:|---|
| TFIM | 22 | affine | slope=1.172847e-10, intercept=-3.964103e-10 | 20.381 | 24.33% | 5.46782654e-09 | 1.13320634e-08 | pass (partial only) |
| TFIM | 22 | power | amplitude=1.443134e-12, exponent=2.917398e+00 | 33.451 | 65.50% | 1.30580371e-07 | 9.86511386e-07 | fail |
| TFIM | 22 | power-plus-offset | amplitude=1.010799e-09, exponent=3.433871e-01, offset=-1.555152e-09 | 34.444 | 12.11% | 2.31804218e-09 | 3.35889248e-09 | fail |
| TFIM | 24 | affine | slope=2.506681e-12, intercept=-9.204024e-12 | 8.517 | 12.88% | 1.16130008e-10 | 2.41464041e-10 | pass (partial only) |
| TFIM | 24 | power | amplitude=8.482737e-15, exponent=3.542476e+00 | 24.516 | 72.12% | 8.85313350e-09 | 1.03154611e-07 | fail |
| TFIM | 24 | power-plus-offset | amplitude=1.441941e-12, exponent=1.215248e+00, offset=-6.945048e-12 | 27.725 | 17.30% | 1.60401672e-10 | 3.81601676e-10 | fail |
| TFIM | 28 | affine | slope=7.410334e-16, intercept=-2.787478e-15 | 41.210 | 13.36% | 3.42641914e-14 | 7.13158611e-14 | pass (partial only) |
| TFIM | 28 | power | amplitude=1.384223e-18, exponent=3.831877e+00 | 35.331 | 40.10% | 4.48177514e-12 | 6.38204689e-11 | fail |
| TFIM | 28 | power-plus-offset | amplitude=6.157274e-18, exponent=3.124641e+00, offset=-2.655600e-16 | 51.831 | 25.82% | 1.25304799e-12 | 1.09309986e-11 | fail |
| TFIM | 30 | affine | slope=7.070556e-18, intercept=-2.488017e-17 | 37.790 | 31.51% | 3.28647626e-16 | 6.82175424e-16 | fail |
| TFIM | 30 | power | amplitude=1.882943e-20, exponent=3.709064e+00 | 32.148 | 15.21% | 3.77073001e-14 | 4.93133973e-13 | fail |
| TFIM | 30 | power-plus-offset | amplitude=7.161813e-22, exponent=5.317858e+00, offset=2.415406e-18 | 49.376 | 54.53% | 7.76077312e-13 | 3.09555369e-11 | fail |
| Heisenberg | 22 | affine | slope=1.318873e-15, intercept=-5.145113e-15 | 54.688 | 74.89% | 6.07985477e-14 | 1.26742208e-13 | pass (partial only) |
| Heisenberg | 22 | power | amplitude=2.227612e-20, exponent=6.673060e+00 | 57.009 | 93.95% | 4.84360820e-09 | 4.94265567e-07 | fail |
| Heisenberg | 22 | power-plus-offset | amplitude=6.699469e-18, exponent=3.564728e+00, offset=-8.029723e-16 | 73.024 | 39.93% | 7.62711002e-12 | 9.02592700e-11 | fail |
| Heisenberg | 26 | affine | slope=1.431944e-19, intercept=-5.690334e-19 | 53.289 | 81.54% | 6.59068837e-18 | 1.37504102e-17 | fail |
| Heisenberg | 26 | power | amplitude=6.234305e-27, exponent=9.747486e+00 | 49.931 | 220.05% | 2.26712066e-10 | 1.94876895e-07 | fail |
| Heisenberg | 26 | power-plus-offset | amplitude=1.049313e-24, exponent=6.993462e+00, offset=-1.328110e-20 | 65.461 | 26.52% | 7.99073172e-13 | 1.01818875e-10 | fail |
| Heisenberg | 28 | affine | slope=7.120324e-22, intercept=-2.836601e-21 | 43.042 | 86.06% | 3.27650177e-20 | 6.83666362e-20 | fail |
| Heisenberg | 28 | power | amplitude=1.697530e-29, exponent=1.003734e+01 | 41.512 | 150.78% | 1.91850845e-12 | 2.01606865e-09 | fail |
| Heisenberg | 28 | power-plus-offset | amplitude=8.893208e-27, exponent=6.695291e+00, offset=-8.388592e-23 | 47.850 | 15.33% | 2.10939456e-15 | 2.18595591e-13 | fail |

## Sensitivity to adding the largest available points

| Model | 2J | Law | Add N=8: ΔB(50) | Add N=8: ΔB(100) | Add N=9: ΔB(50) | Add N=9: ΔB(100) | Parameter changes on adding N=9 |
|---|---:|---|---:|---:|---:|---:|---|
| TFIM | 22 | affine | -4.43% | -4.46% | -1.76% | -1.77% | slope -1.79%, intercept 2.18% |
| TFIM | 22 | power | -62.11% | -71.41% | -24.73% | -30.78% | amplitude 20.77%, exponent -4.14% |
| TFIM | 22 | power-plus-offset | -43.44% | -54.69% | -15.73% | -20.97% | amplitude 121.65%, exponent -42.96%, offset -84.40% |
| TFIM | 24 | affine | -0.84% | -0.85% | -3.01% | -3.02% | slope -3.04%, intercept 3.40% |
| TFIM | 24 | power | -73.75% | -82.42% | -64.70% | -74.34% | amplitude 113.77%, exponent -13.00% |
| TFIM | 24 | power-plus-offset | -57.16% | -69.15% | -46.46% | -58.38% | amplitude 193.49%, exponent -33.92%, offset -73.65% |
| TFIM | 28 | affine | 8.72% | 8.76% | 3.96% | 3.98% | slope 4.00%, intercept -4.46% |
| TFIM | 28 | power | -72.08% | -81.54% | -66.74% | -76.95% | amplitude 163.53%, exponent -13.81% |
| TFIM | 28 | power-plus-offset | -85.59% | -92.73% | -71.18% | -81.86% | amplitude 296.03%, exponent -21.43%, offset -101.73% |
| TFIM | 30 | affine | 22.61% | 22.77% | 11.75% | 11.82% | slope 11.89%, intercept -13.68% |
| TFIM | 30 | power | 19.70% | 26.19% | -22.69% | -28.47% | amplitude 19.90%, exponent -3.02% |
| TFIM | 30 | power-plus-offset | -97.98% | -99.46% | -89.35% | -95.17% | amplitude 815.59%, exponent -21.41%, offset -43.67% |
| Heisenberg | 22 | affine | 0.89% | 0.89% | 0.43% | 0.43% | slope 0.43%, intercept -0.44% |
| Heisenberg | 22 | power | 0.76% | 0.98% | -9.12% | -11.72% | amplitude 7.01%, exponent -0.63% |
| Heisenberg | 22 | power-plus-offset | 82.16% | 122.79% | 23.53% | 32.57% | amplitude -17.09%, exponent 2.86%, offset 5.29% |
| Heisenberg | 26 | affine | 16.98% | 16.99% | 26.15% | 26.16% | slope 26.17%, intercept -26.34% |
| Heisenberg | 26 | power | -77.95% | -85.80% | -82.71% | -89.69% | amplitude 218.96%, exponent -7.64% |
| Heisenberg | 26 | power-plus-offset | -63.08% | -73.24% | -52.99% | -63.25% | amplitude 88.91%, exponent -5.08%, offset -19.77% |
| Heisenberg | 28 | affine | 5.26% | 5.26% | 62.10% | 62.11% | slope 62.12%, intercept -62.38% |
| Heisenberg | 28 | power | -70.12% | -79.09% | -87.92% | -93.58% | amplitude 328.58%, exponent -9.09% |
| Heisenberg | 28 | power-plus-offset | -72.37% | -81.74% | 88.40% | 131.59% | amplitude -41.23%, exponent 4.45%, offset 12.75% |

## Available shifted-window diagnostics

Only N=4..8 and N=5..9 can be compared. Every `pass` below is a necessary condition only; the full gate remains unevaluated until N=6..10 exists.

| Model | 2J | Law | S(12) | S(20) | S(50) | S(100) | Parameter span / limit | Residual drift free | Partial result |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| TFIM | 22 | affine | 8.72% | 10.53% | 11.80% | 12.17% | 0.1251 / 0.2000 | True | pass |
| TFIM | 22 | power | 59.31% | 102.77% | 155.44% | 176.81% | 1.0266 / 0.1500 | True | fail |
| TFIM | 22 | power-plus-offset | 6.47% | 14.74% | 31.21% | 44.24% | 0.3434 / 0.1500 | True | fail |
| TFIM | 24 | affine | 2.26% | 2.18% | 2.13% | 2.12% | 0.0210 / 0.2000 | True | pass |
| TFIM | 24 | power | 73.89% | 122.52% | 172.17% | 188.00% | 1.2738 / 0.1500 | True | fail |
| TFIM | 24 | power-plus-offset | 26.65% | 57.07% | 110.61% | 142.99% | 1.0168 / 0.1500 | True | fail |
| TFIM | 28 | affine | 7.75% | 9.20% | 10.16% | 10.43% | 0.1068 / 0.2000 | True | pass |
| TFIM | 28 | power | 44.37% | 79.40% | 129.26% | 155.01% | 0.7615 / 0.1500 | True | fail |
| TFIM | 28 | power-plus-offset | 38.68% | 81.48% | 142.69% | 169.74% | 1.0367 / 0.1500 | True | fail |
| TFIM | 30 | affine | 16.97% | 19.83% | 21.73% | 22.26% | 0.2276 / 0.2000 | False | fail |
| TFIM | 30 | power | 15.50% | 38.14% | 76.00% | 101.10% | 0.4519 / 0.1500 | True | fail |
| TFIM | 30 | power-plus-offset | 79.81% | 145.82% | 190.42% | 197.61% | 2.0293 / 0.1500 | True | fail |
| Heisenberg | 22 | affine | 6.35% | 6.96% | 7.36% | 7.47% | 0.0758 / 0.2000 | True | pass |
| Heisenberg | 22 | power | 124.64% | 168.20% | 194.22% | 198.47% | 1.9353 / 0.1500 | True | fail |
| Heisenberg | 22 | power-plus-offset | 194.85% | 199.90% | 200.00% | 200.00% | 7.6998 / 0.1500 | True | fail |
| Heisenberg | 26 | affine | 23.44% | 23.60% | 23.70% | 23.73% | 0.2376 / 0.2000 | True | fail |
| Heisenberg | 26 | power | 143.05% | 184.14% | 198.64% | 199.79% | 2.7246 / 0.1500 | True | fail |
| Heisenberg | 26 | power-plus-offset | 27.00% | 47.75% | 82.21% | 104.99% | 0.4221 / 0.1500 | True | fail |
| Heisenberg | 28 | affine | 47.42% | 47.51% | 47.58% | 47.59% | 0.4761 / 0.2000 | False | fail |
| Heisenberg | 28 | power | 138.96% | 182.87% | 198.53% | 199.78% | 2.7262 / 0.1500 | True | fail |
| Heisenberg | 28 | power-plus-offset | 21.30% | 36.84% | 63.53% | 82.23% | 0.3115 / 0.1500 | True | fail |

No row has more than one provisionally viable model with the current data. TFIM 2J=24 and 28 each retain only the affine candidate. Thus their N=10 points would complete the holdout/stability evidence, not resolve a current affine-versus-power ambiguity.

## Wall-time scaling and N=10 projection

The row estimate repeats the measured N=8→9 task-time factor once. Task time is the sum of all precision attempts and segment points in the shard.

| Model | 2J | N=8 task min | N=9 task min | Factor | Projected N=10 h | Scientific disposition |
|---|---:|---:|---:|---:|---:|---|
| TFIM | 22 | 11.40 | 61.91 | 5.43× | 5.60 | not-informative-for-current-candidate-family |
| TFIM | 24 | 11.23 | 68.21 | 6.08× | 6.91 | needed-for-second-holdout-and-full-stability |
| TFIM | 28 | 16.02 | 85.95 | 5.37× | 7.69 | needed-for-second-holdout-and-full-stability |
| TFIM | 30 | 15.84 | 89.51 | 5.65× | 8.43 | not-informative-for-current-candidate-family |
| Heisenberg | 22 | 12.61 | 84.36 | 6.69× | 9.41 | defer-until-required-N8-time-law-check |
| Heisenberg | 26 | 15.89 | 92.73 | 5.83× | 9.02 | defer-until-required-N8-time-law-check |
| Heisenberg | 28 | 17.26 | 107.75 | 6.24× | 11.21 | defer-until-required-N8-time-law-check |

The measured two-batch N=9 makespan reconstructed from shard timings was 3.29 h. Running N=10 for all seven numerically surviving rows is projected at 58.3 task-h, or 22.7 h under ideal three-worker scheduling. The narrower total for gate-complete rows that remain scientifically informative is 14.6 task-h / 7.7 ideal three-worker hours.

No N=10 run should be started from this report. Heisenberg rows first need their required N=8 time-law expansion; TFIM N=10 candidates are identified row by row above. No high-order coefficient is promoted at this checkpoint.
