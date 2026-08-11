# Empirical operator-norm sizing

The empirical policy is an opt-in resource-sizing model. It is not a theorem
bound and it never creates an `ErrorClaim`. Analytical Trotter, MPF, and QSVT
defaults are unchanged.

## Reviewed scope

The packaged registry contains exact-parameter calibrations for two open
one-dimensional models:

- TFIM with `coupling=1`, `field=3`, and `periodic=False`;
- Heisenberg with `coupling=1` and `field_z=0.3`.

The reviewed v1 rows cover Trotter orders 2, 4, and 6 and MPF branch counts 2
through 8. Schema v2 preserves those rows and supports pilot-justified tagged
affine, power, and power-plus-offset size laws. High-order rows are loaded only
when their numerical, time-law, holdout, shifted-window, domain, provenance,
and human-review gates all pass. MPF calibration uses the registered `new`
schedule and the ordered-individual-term Strang base formula. There are no
enabled 2D, periodic-boundary, fermionic, rescaled-coupling, or legacy-schedule
calibrations.

The model identity stored by the built-in Hamiltonian constructors is matched
exactly. A custom `PauliHamiltonian` without `HamiltonianModelMetadata`, a
parameter mismatch, or a missing reviewed row raises
`UnsupportedEmpiricalCalibrationError`; there is no analytical fallback.

## Sizing law and domain

For an MPF with selected branch count \(J\), a reviewed row stores a positive
finite-size coefficient \(B_{2J}(N)\) and uses

\[
\epsilon_{\rm emp}=B_{2J}(N)\frac{T^{2J+1}}{r^{2J}}.
\]

The tagged coefficient model selected during review is affine \(aN+b\), power
\(AN^p\), or power-plus-offset \(AN^p+C\). These tags do not alter the fixed
formal powers. Trotter rows use the analogous law with their stored formal
order \(q\):

\[
\epsilon_{\rm emp}=B_q(N)\frac{T^{q+1}}{r^q}.
\]

The formula segment count is

\[
r_{\rm formula}=\left\lceil
\left(\frac{B_q(N)T^{q+1}}{\epsilon_{\rm alg}}\right)^{1/q}
\right\rceil.
\]

Planning takes the maximum of one, this value, and the row's maximum-step-size
guard. The plan reports which constraint was active. Sizes above the observed
maximum and times outside the checked interval are flagged as extrapolations.
Schema-v2 rows additionally carry `reviewed_size_max`, derived from the
committed downstream benchmark configuration (currently \(N=100\)). Requests
above that reviewed maximum fail with an actionable unsupported-domain error;
the code never silently extends review to an arbitrary larger size. Sizes below
the fitted minimum and nonpositive coefficient predictions also fail closed.

The algorithmic error budget remains separate from the arbitrary-rotation
synthesis budget. Empirical sizing therefore changes repetitions or segments;
it does not claim to include synthesis error.

## Selecting the policy

```python
from hamiltonian_resources import (
    MultiproductMethod,
    TrotterMethod,
    estimate_resources,
    transverse_field_ising,
)

hamiltonian = transverse_field_ising(
    8, coupling=1.0, field=3.0, periodic=False
)
trotter = estimate_resources(
    hamiltonian,
    TrotterMethod(4, error_policy="empirical-operator-norm"),
    time=8.0,
    target_error=1e-3,
)
mpf = estimate_resources(
    hamiltonian,
    MultiproductMethod(
        None,
        error_method="empirical-operator-norm",
        branch_count_policy="mizuta2026-theorem6",
    ),
    time=8.0,
    target_error=1e-3,
)
```

The resulting sizing category is `empirical`, certification is `nonrigorous`,
and both ideal and circuit target-certification outcomes are unavailable.
Benchmark comparisons must use `certification_policy="unconstrained"` when
these rows are intentionally included.

## Branch count and segment count are separate

`plan_simulation` resolves the MPF branch count before it queries the empirical
registry:

```text
resolve_mpf_branch_count(...) -> J, formal_order=2J
mpf_exponent_cost(J, ...)     -> registered schedule cost
registry.lookup(..., formal_order=2J, ...)
select_empirical_segments(...) -> r
```

For `branch_count_policy="mizuta2026-theorem6"` and nonzero \(gT\), the
repository uses

\[
J=\max\left(2,\left\lceil\frac12\log
\frac{Ng|T|}{\epsilon_{\rm alg}}\right\rceil\right).
\]

The calibration coefficient is not an input to this policy. Replacing
\(B_{2J}\) changes the selected segment count \(r\), not \(J\). A missing
calibration fails after branch selection at that exact formal order; planning
does not search another calibrated order. The fixed policy likewise uses the
explicit `term_count` unchanged.

## MPF aggregate cost versus implementation data

`mpf_exponent_cost(m, schedule="new")` returns the exact registered
\(K=\sum_jk_j\) through `m=15`. Above 15 it returns only

\[
K=\left\lceil0.418m^2\ln m\right\rceil.
\]

Such a plan has no exponent tuple, coefficients, coefficient 1-norm, padding
weight, or LCU structure. Analytical CNOT and T totals remain available as a
conditional proxy using the existing branch-addressing/OAA architecture, but
the plan is not circuit-ready. `build_simulation_circuit` raises an explicit
aggregate-cost-only error. Legacy schedules are never extrapolated.

## Calibration and review workflow

`scripts/run_empirical_calibration.py` runs or resumes the high-precision task
matrix. Raw per-task shards live below the ignored
`artifacts/calibration_runs/` directory. `scripts/assemble_empirical_calibrations.py`
reduces those shards to canonical JSON containing one observation per task
point, rejected-point counts, accepted asymptotic windows, time-law checks,
size fits, shifted-window stability, and provenance. Re-reducing identical
parsed shard contents is byte-identical and uses canonical parsed-JSON hashes,
so line endings do not affect provenance.

The reference backend constructs exact rational Richardson coefficients after
setting precision, evaluates analytic Pauli exponentials in repository term
order, forms the repeated ideal MPF operator, and computes the spectral norm
from \(D^\dagger D\). FLINT parity blocks are used for bulk execution only
after agreement with mpmath. Precision increases by 32 digits until the error
and \(B_{2J}\) agree; unsuccessful points remain explicitly
`precision-limited`.

Promotion requires four consecutive asymptotic points, the fixed time law,
\(N=9,10\) holdouts, refits over \(N=4\ldots8\), \(5\ldots9\), and
\(6\ldots10\), and stability through the actual downstream review maximum.
High formal orders also require \(N=11,12\) validation when the committed cost
and memory guards deem it feasible; otherwise the tighter documented
review-exception thresholds apply.

Changing a package calibration row requires a new source-artifact digest and
an explicit `reviewed` record. Candidate, rejected, numerically ambiguous, and
unidentified pre-study fits are not loaded at runtime. The supplied
[pre-study](empirical_error_prestudy_trotter_mpf.md) and
[MPF exponent-cost note](mpf_exponent_sum_scaling.md) provide research context
without serving as the runtime coefficient table.
