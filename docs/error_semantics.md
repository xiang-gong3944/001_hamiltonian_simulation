# Structured error semantics

Parameter sizing, mathematical certification, and empirical validation are
different records in the in-memory plan/report model.

- `SizingEstimate` records the value and scope used to select repetitions,
  segments, or polynomial degree. It may be a rigorous analytical estimate,
  an explicitly selected empirical calibration, or a proxy.
- `ErrorClaim` is a mathematical statement with an explicit quantity, metric,
  object scope, and certification status.
- `MetricObservation` is an empirical value with state or calibration context.

`ErrorAnalysis` exposes separate outcomes for parameter-selection success, the
ideal algorithmic target, and the implemented logical-circuit target. A proxy
can therefore produce a complete resource estimate without certifying either
target. State error, fidelity, success probability, and finite-grid phase
residuals never satisfy a target certification.

## Empirical sizing

`error_policy="empirical-operator-norm"` is a nonrigorous sizing category,
not an empirical `ErrorClaim`. It uses a reviewed spectral-norm calibration to
select a resource shape and records exact model identity, calibration range,
extrapolation flags, and the active formula/domain constraint. Both target
assessments remain `unavailable`, even when the numerical estimate is below
the allocated error. See [the empirical guide](empirical_error_estimation.md).

## QSVT scopes

For repository degree \(d=2K+1\), the cosine and sine degrees are \(d-1\) and
\(d\); their first omitted parity terms have degrees \(d+1\) and \(d+2\).
Let the corresponding unscaled Jacobi--Anger tail bounds be \(E_c,E_s\), and
let \(s=1-\varepsilon_{\rm alg}/18\). The exact target polynomial is

\[
Q_K(x)=s(C_K(x)-iS_K(x)),
\]

with rigorous uniform/operator error

\[
\delta_{\rm poly}\le(1-s)+s(E_c+E_s).
\]

For the ideal cubic-OAA image

\[
F(Q_K)=\frac32Q_K-\frac12Q_KQ_K^\dagger Q_K,
\]

the repository records

\[
\lVert F(Q_K)-e^{-iHt}\rVert\le
\delta_{\rm poly}
+\frac12\delta_{\rm poly}(2+\delta_{\rm poly})(1+\delta_{\rm poly}).
\]

This certifies the exact scaled polynomial/OAA model. The `pyqsp` residual is
measured on 2049 Chebyshev extrema and is stored as a calibration observation;
it is not a uniform proof for the floating-point phase circuit. Consequently,
the structured QSVT report may certify its ideal polynomial target while
`implemented_circuit_target_certified` remains false.

## Persistent compatibility

Stage A keeps benchmark schema 2.0. Flat `bound_*` and `circuit_*` columns are
conservative derived views, not the canonical error model. Legacy schema-2
QSVT rows that claimed `implemented-algorithm` certification are downgraded in
memory when loaded. Persisting the complete structured records, migration
history, and withdrawn-claim provenance is deferred to schema 3.
