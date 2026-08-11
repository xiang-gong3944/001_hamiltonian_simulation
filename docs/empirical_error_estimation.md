# Empirical operator-norm sizing

The empirical policy is an opt-in resource-sizing model. It is not a theorem
bound and it never creates an `ErrorClaim`. Analytical Trotter, MPF, and QSVT
defaults are unchanged.

## Reviewed scope

Version 1 contains exact-parameter calibrations for two open one-dimensional
models:

- TFIM with `coupling=1`, `field=3`, and `periodic=False`;
- Heisenberg with `coupling=1` and `field_z=0.3`.

Trotter rows cover orders 2, 4, and 6 under the partition produced by the
current `auto` resolver. MPF rows cover branch counts 2 through 8 under the
registered `new` schedule and the ordered-individual-term Strang base formula.
There are no enabled 2D, periodic-boundary, fermionic, rescaled-coupling, or
legacy-schedule calibrations.

The model identity stored by the built-in Hamiltonian constructors is matched
exactly. A custom `PauliHamiltonian` without `HamiltonianModelMetadata`, a
parameter mismatch, or a missing reviewed row raises
`UnsupportedEmpiricalCalibrationError`; there is no analytical fallback.

## Sizing law and domain

Each reviewed row stores an affine finite-size coefficient

\[
B_q(N)=a_qN+b_q
\]

and uses the fixed formal-order law

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
guard. The plan reports which constraint was active. Sizes above the fitted
maximum and times outside the checked interval are allowed but flagged as
extrapolations. Sizes below the fitted minimum and nonpositive affine
predictions fail closed.

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

`scripts/calibrate_empirical_errors.py` generates a manifest, all raw
operator-norm observations, accepted consecutive asymptotic observations, an
affine fit with residual and uncertainty diagnostics, and a separate review
record. It uses explicit dense operators at small size and deterministic
power iteration on \(D^\dagger D\) with sparse `expm_multiply` exact evolution
at larger size. The approximate kernels are the ideal product formula and the
repeated ideal MPF operator; random-state errors and postselected circuit
proxies are not accepted.

Changing a package calibration row requires a new source-artifact digest and
an explicit `reviewed` record. Candidate, rejected, numerically ambiguous, and
unidentified pre-study fits are not loaded at runtime. The supplied
[pre-study](empirical_error_prestudy_trotter_mpf.md) and
[MPF exponent-cost note](mpf_exponent_sum_scaling.md) provide research context
without serving as the runtime coefficient table.
