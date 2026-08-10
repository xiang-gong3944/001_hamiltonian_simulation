# Empirical 1D calibration review record

- Study: `empirical-operator-norm-1d-v1`
- Decision: `reviewed`
- Decision date: 2026-08-10
- Authority: implementation plan approved for this feature branch
- Runtime scope: only the exact model, boundary, partition/formula, schedule,
  and formal-order keys listed in the manifest and fit artifact

Review checks:

- calibration inputs are spectral operator norms, never random-state errors;
- dense and sparse kernels agree on overlapping small systems;
- accepted segment pairs stay above the declared numerical floor and their
  running exponents are within 5% of formal order;
- fixed formal powers in segment count and time are not fitted;
- the remaining coefficient is fit to `B_q(N)=a_q*N+b_q` with residual and
  uncertainty diagnostics retained;
- coupling, field, boundary, partition, and schedule mismatches fail closed;
- the pre-study's incompletely identified MPF m=3 equations are sanity checks
  only and are not runtime rows;
- changing any reviewed coefficient requires a new accepted-observation digest
  and a new explicit review decision.

The policy remains nonrigorous. Review establishes reproducible numerical
provenance; it does not convert a fitted estimate into a mathematical error
bound or certify an implemented circuit.
