# MPF error bounds and certification scopes

This document maps the MPF notation used by the repository to the latest
available revisions checked on 2026-07-31:

- Low, Kliuchnikov, and Wiebe (LKW),
  [arXiv:1907.11679v2](https://arxiv.org/abs/1907.11679), 2019;
- Aftab, An, and Trivisa (AAT),
  [arXiv:2403.08922v1](https://arxiv.org/abs/2403.08922), 2024;
- Mizuta, [Quantum 10, 1974 (2026)](https://quantum-journal.org/papers/q-2026-01-19-1974/),
  [arXiv:2507.06557v4](https://arxiv.org/abs/2507.06557).

The word *rigorous* below applies only to the stated operator and circuit
scope. The ideal-MPF bounds do not, by themselves, certify the repository's
repeated shared-ancilla robust-OAA circuit. A separate block-encoding product
argument for that circuit is documented below.

## Common repository construction

For a `PauliHamiltonian`, the analytical estimators use the ordered individual
Pauli decomposition

\[
H=\sum_{\gamma=1}^{\Gamma}H_\gamma,
\qquad H_\gamma=h_\gamma P_\gamma.
\]

The term order is the order stored in `hamiltonian.terms`. Identity terms are
retained by the simulator but commute exactly and do not contribute to nested
commutators. The base sequence is the ordered symmetric second-order formula

\[
S_2(\tau)=
\prod_{\gamma=\Gamma}^{1}e^{-i\tau H_\gamma/2}
\prod_{\gamma=1}^{\Gamma}e^{-i\tau H_\gamma/2}.
\]

For a \(J\)-branch repository schedule (the backward-compatible public argument
is named `m`), the ideal MPF step is

\[
M(\tau)=\sum_{j=1}^{J} a_j
\left[S_2(\tau/k_j)\right]^{k_j},
\]

where `optimal_mpf_exponents(J)` gives the positive integers \(k_j\) and
`multiproduct_coefficients(J)` gives \(a_j\). The cancellation conditions are

\[
\sum_j a_j=1,\qquad
\sum_j a_j/k_j^{2q}=0\quad(1\le q<J),
\]

so the repository formal order is \(2J\) and the leading local error order is
\(2J+1\). The coefficient norm is
\(\lVert a\rVert_1=\sum_j|a_j|\), the local time is \(\tau=t/r\), and `r` is
the positive integer segment count.

In repository notation, `term_count` is the number of MPF branches \(J\); it is
not Mizuta's formal-order parameter. For the implemented symmetric second-order
base formula,

\[
m_{\mathrm{formal}}=2J.
\]

Existing public API parameter names (including the legacy `m` spelling), method
IDs, and serialized keys are retained for backward compatibility.

## Notation map

| Concept | LKW 2019 | AAT 2024 | Mizuta 2026 | Repository |
| --- | --- | --- | --- | --- |
| Hamiltonian split | \(H=\sum_j h_j\) | \(H=\sum_\gamma H_\gamma\) | Eqs. (1), (4): \(H=\sum_Xh_X=\sum_\gamma H_\gamma\) | ordered individual \(h_\gamma P_\gamma\) |
| Base formula | symmetric \(U_2\) | Eq. (43), \(U_2\) | Eq. (6), \(T_p\), with \(p=2,c_2=2\) | ordered `S2`, `order=2`, `partition="individual"` |
| MPF | \(U_{\vec k}(\Delta)\) | Eq. (114), \(U_{\rm MP}(\Delta)\) | Eq. (9), \(M_{pmJ}(\tau)\) | \(M(\tau)=\sum_j a_jS_2(\tau/k_j)^{k_j}\) |
| Exponents | \(k_j\) | \(k_l\) | \(k_j\) | `exponents` |
| Coefficients | \(a_j\) | \(a_l\) | \(c_j\) | `coefficients` \(a_j\) |
| Branch count | implicit in the schedule | \(J\) | \(J\) | `term_count=J`; legacy APIs call this `m` |
| Formal order | \(2m\) | \(2m\) | paper parameter \(m\) | `formal_order=2*J`; this equals Mizuta's \(m\) |
| Coefficient norm | \(\lVert\vec a\rVert_1\) | \(\lVert\vec a\rVert_1\) | \(\lVert c\rVert_1\) | `coefficient_l1_norm` |
| Exponent norm | \(\lVert\vec k\rVert_1\) | \(\lVert\vec k\rVert_1\) | \(\lVert k\rVert_1\) | `sum(exponents)` |
| Local time | \(\Delta=t/r\) | \(\Delta=T/r\) | \(\tau=t/r\) | `local_step_size` |
| Commutators | qualitative in the locality discussion | Eqs. (10), (130), all orders | Eq. (8), orders through \(p_0\) in Theorem 4 | `pauli_nested_commutator_bounds` |
| Primary theorem scope | ideal MPF step and repeated ideal MPF | Theorem 8: one ideal step; Theorem 9: repeated ideal MPF/LCU complexity | Theorem 4: one ideal MPF step | repeated ideal MPF after the documented telescoping composition |
| Current circuit scope | not the repository circuit by itself | not the repository circuit | not the repository circuit by itself | `repeated-shared-ancilla-good-block`, separately bounded when the local claim is rigorous |

## Low 2019 1-norm baseline

The method identifier is `low2019-l1-ideal-rigorous`. The historical input
name `low-rigorous` is accepted as an alias and normalized to the explicit
identifier.

Set

\[
\lambda=\sum_\gamma\lVert H_\gamma\rVert
=\sum_\gamma|h_\gamma|=
\texttt{hamiltonian.alpha}.
\]

LKW Eq. (13), followed by the triangle inequality in Eq. (14), gives the
single-step bound

\[
\delta_r=
\frac{2\lVert a\rVert_1(\lambda|t|/r)^{2m+1}}
{(2m+1)!}e^{\lambda|t|/r}.
\]

Their Eq. (15) gives the repeated ideal-MPF bound

\[
E_r=r\delta_r(1+\delta_r)^{r-1}.
\]

The implementation evaluates these formulas in the log domain. LKW Eq. (16)
is used only to obtain a safe integer upper bracket. An integer binary search
then returns the smallest positive `r` for which \(E_r\) meets the algorithmic
error budget. Consequently, if `r > 1`, `r-1` fails the same implemented
bound. Overflow of the representable floating-point segment range is reported
as `OverflowError`; it is not converted to a finite estimate.

This is a worst-case operator-norm guarantee for
\(M(t/r)^r\). It is independent of commutation structure and does not certify
the robust-OAA shared-ancilla circuit.

## Finite-size comparison with the Mizuta bound

The Low 2019 bound can require fewer resources than the Mizuta 2026
commutator bound at finite \((N,t,\epsilon)\), without contradicting their
asymptotic comparison. Low controls the error through the Hamiltonian
coefficient 1-norm \(\lambda\), which generally gives poorer asymptotic
system-size scaling, but it retains favorable explicit finite-order factors
such as \((2J+1)!\) and does not impose Mizuta's finite-step time hypothesis.

Mizuta instead supplies a rigorous locality-compatible commutator scaling. Its
explicit theorem also uses sufficient time conditions, including

\[
|\tau|\le \frac{1}{8e^3c_pp_0kg},
\]

which can dominate the required segment count at finite size. Consequently,
the Mizuta bound is not necessarily numerically tighter than Low for a fixed
finite problem. Its principal advantage is the improved locality-aware
asymptotic scaling. Both methods remain rigorous within their declared scopes;
the observed finite-resource difference is therefore not a contradiction.

## Mizuta 2026 finite-order commutator bound

The method identifier is `mizuta2026-commutator-ideal-rigorous`. Mizuta's
Theorem 4 is used with base order \(p=2\), paper formal order equal to the
repository's \(2J\), and \(c_p=c_2=2\).

### Locality data and exact Pauli commutators

For Pauli terms, the theorem's locality parameters are computed directly:

\[
k=\max_\gamma|\operatorname{supp}(P_\gamma)|,
\qquad
g=\max_i\sum_{\gamma:i\in\operatorname{supp}(P_\gamma)}|h_\gamma|.
\]

The exact finite-order quantity is Mizuta Eq. (8),

\[
\alpha_{\mathrm{com},q}=
\sum_{\gamma_1,\ldots,\gamma_q}
\left\lVert
[H_{\gamma_q},\ldots,[H_{\gamma_2},H_{\gamma_1}]]
\right\rVert.
\]

Every nonzero nested commutator of individual Pauli terms is a scalar times
one Pauli word. The implementation propagates nonnegative norm weights keyed
by the resulting binary-symplectic Pauli word. This computes the sum over
ordered term sequences exactly up to upward floating-point rounding, without
forming a dense Hamiltonian matrix.

The recurrence is extended incrementally across the adaptive segment search,
so nearby truncation orders reuse their exact prefix. Large state frontiers may
be chunked across worker processes. Progress for this adaptive search reports
the current segment candidate and completed orders or chunks without presenting
an unknown final workload as a percentage.

If the explicit transition cap is reached, remaining orders use the proven
locality bound in Mizuta Eq. (8),

\[
\alpha_{\mathrm{com},q}\le
(q-1)!(2kg)^{q-1}Ng.
\]

The estimate stays rigorous under this replacement, but records the fallback
reason, the maximum exact order, and the maximum order used. It never falls
back to the W2 heuristic.

### Finite truncation and the sharp polynomial-root bound on \(\mu\)

For a chosen auxiliary \(\eta\in(0,1)\), Mizuta Eq. (33) sets

\[
p_0=\left\lceil\log(3N/\eta)\right\rceil.
\]

Theorem 4/Eq. (47) defines \(\mu_{p,m}[p_0]\) using
\(q\ge m+1\), \(n\le\lfloor(q-1)/p\rfloor\), and
\(p+1\le q_i\le p_0\), with \(\sum_iq_i=q+n-1\). Its repetition index \(n\)
is unbounded. Define

\[
A(x)=\sum_{s=p+1}^{p_0}\alpha_{\mathrm{com},s}x^s
\]

and let \(x_*>0\) solve \(A(x_*)=1\). For fixed repetition count \(n\), the
sum in Eq. (47) with total degree \(D=q+n-1\) is a coefficient of
\(A(x)^n\). Since all coefficients are nonnegative,

\[
[x^D]A(x)^n\,x_*^D\le A(x_*)^n=1.
\]

Therefore every candidate in Eq. (47) is at most \(1/x_*\), and

\[
\mu_{p,m}[p_0]\le\mu_*=1/x_*.
\]

The formal-order condition does not make this bound loose. Let
\(\pi_s=\alpha_{\mathrm{com},s}x_*^s\), so that \(\sum_s\pi_s=1\). Then

\[
[x^D]A(x)^n x_*^D=\Pr(S_n=D),
\]

where \(S_n\) is a sum of \(n\) independent variables with probabilities
\(\pi_s\). There are at most \(n(p_0-p-1)+1\) possible values of \(S_n\), so
one has probability at least the reciprocal of this quantity. Its paper index
satisfies \(q=D-n+1\ge np+1\), and therefore obeys \(q\ge m+1\) for every
sufficiently large \(n\). Since \(D\) grows linearly in \(n\), the degree root
of this coefficient converges to \(1/x_*\). Thus

\[
\mu_{p,m}[p_0]=\mu_*
\]

for nonzero supplied nonnegative commutator data, independently of every fixed
formal-order cutoff \(m\). The all-zero case has \(\mu=0\). Removing low-order
candidates is real, but the unbounded repetition index makes the same supremum
sharp at higher orders.

The implementation retains the name `mu_upper` because the exact Pauli prefix
is rounded upward and any locality-fallback entries are upper bounds on the
true \(\alpha_{\mathrm{com},q}\). The log-domain root is sharp for those
supplied bounds; remaining looseness comes from the supplied commutator bounds,
not from ignoring the MPF order condition.

At fixed \(p_0\) and fixed supplied commutator data, this root is independent
of \(J\). Across complete resource estimates, `mu_upper` can still vary with
\(J\) because the coefficient and exponent norms change the allocated
auxiliary error, which can change \(p_0\). That is a finite-truncation input
effect, not a benefit or penalty from the formal-order cutoff in Eq. (47).

### Error and time hypotheses

Mizuta Eq. (48) requires

\[
|\tau|\le\min\left\{
\frac{1}{8e^3c_pp_0kg},
\frac{1}{2c_p\mu_{\rm upper}}
\right\}.
\]

When it holds, Eq. (49), with the upper bound on \(\mu\), gives

\[
\delta_r\le
2e^{1/2}\lVert a\rVert_1
(c_p\mu_{\rm upper}|\tau|)^{2J+1}
+\lVert a\rVert_1\lVert k\rVert_1\eta.
\]

The code chooses

\[
d_r=(1+\varepsilon_{\rm target})^{1/r}-1,
\qquad
\eta=\frac{d_r}{2\lVert a\rVert_1\lVert k\rVert_1},
\]

so half of the exact local budget is reserved for the truncated-BCH remainder.
It then evaluates the displayed bound rather than assuming the other half is
sufficient.

Theorem 4 is a one-step ideal-MPF result. The repository composes it to the
repeated ideal operator explicitly: if \(U=e^{-iH\tau}\) and
\(\lVert M-U\rVert\le\delta_r\), then \(\lVert M\rVert\le1+\delta_r\) and

\[
\lVert M^r-U^r\rVert
\le\sum_{j=0}^{r-1}\lVert M\rVert^{r-j-1}\lVert M-U\rVert
\le r\delta_r(1+\delta_r)^{r-1}.
\]

The selected row is rigorous only when Eq. (48) and this repeated bound both
meet the requested budget. Mutually commuting Pauli decompositions are
recognized separately: the symmetric product formula and MPF are then exact,
so the reported error is zero.

## Why Aftab 2024 is not a selectable rigorous estimator

AAT Theorem 8/Eqs. (114)--(115) is a rigorous one-step ideal-MPF bound for the
same ordered second-order base formula. It requires an integer \(J\) such that

\[
\Delta\le\inf_{j\ge J}\alpha_{\mathrm{comm},j}^{-1/j}
\]

and its error is an infinite series involving
\(\alpha_{\mathrm{comm},j}\) at arbitrarily high orders. Theorem 9/Eqs.
(129)--(131) similarly defines its repeated-step parameter using an unbounded
supremum. A finite truncation of those quantities is not the theorem's bound.

Mizuta 2026, Section 2.2 and Theorem 1, explains why substituting the usual
fixed-order locality estimate into that arbitrary-order requirement does not
establish locality-compatible complexity: the factorial growth eventually
removes the local-chain advantage. The finite Pauli commutator engine in this
repository deliberately stops at a practical cap, so it cannot certify the
arbitrary-order AAT hypothesis by exact enumeration for general inputs.

Accordingly, there is no
`aftab2024-commutator-ideal-rigorous` selectable method. Adding a truncated AAT
curve would require an explicitly nonrigorous diagnostic name and is not done
here. The AAT result remains an important rigorous comparison and intermediate
analysis; it is not silently combined with Mizuta's finite-truncation theorem.

## Historical W2 proxy

`legacy-w2-proxy` preserves the old substitution

\[
\alpha_{\rm eff}=\min\{\alpha,W_2^{1/3}\},
\qquad
E_r=\alpha_{\rm eff}^{2J+1}|t|^{2J+1}/r^{2J}.
\]

No cited MPF theorem proves this substitution. Its rows always carry
`bound_rigorous=false`, have no commutator certification scope, and are never
selected by either rigorous policy.

## Ideal operator versus implemented circuit

The circuit builder pads \(\lVert a\rVert_1<2\) with two cancelling identity
branches so that the unamplified good block is \(B=M/2\). One robust OAA round
has good block

\[
3B-4BB^\dagger B.
\]

The same branch register is reused across segments. The final good block is
therefore not asserted to be \(M(t/r)^r\). With \(P\) the all-zero branch
projector and \(W\) one amplified segment,

\[
PW^2P=(PWP)^2+PW(I-P)WP.
\]

The second term is a real leave-and-reenter path and is retained by the
reference circuit.

Let \(A=PWP\), \(U_\tau=e^{-iH\tau}\), and suppose a rigorous ideal-MPF local
bound gives \(\lVert M-U_\tau\rVert\le\delta\). The exact cubic identity gives

\[
\lVert A-U_\tau\rVert\le
\eta=\delta+\frac12\delta(2+\delta)(1+\delta).
\]

Unitarity of \(W\) also gives

\[
A^\dagger A+C^\dagger C=I,
\qquad C=(I-P)WP,
\]

and, for \(\eta\le1\),

\[
\lVert C\rVert\le\sqrt{2\eta-\eta^2}.
\]

Gilyén--Su--Low--Wiebe Lemma 54 and Corollary 55 cover products of
scale-one unitary block encodings while reusing the same ancilla. Applying
that result to the actual sequence of \(W\) gates yields the conservative
projected-good-block bound

\[
\lVert PW^rP-e^{-iHt}\rVert\le
\begin{cases}
\eta,&r=1,\\
\min\{2,4r^2\eta\},&r>1,\ \eta\le1.
\end{cases}
\]

When \(\eta>1\), the repository emits no nontrivial repeated-use claim. The
bound is for \(PW^rP\), not full joint-unitary closeness, normalized
postselected state error, or success-overhead-adjusted resources.

The structured report therefore carries independent ideal, one-segment, and
repeated-good-block claims. Its flat compatibility view reports

```text
bound_scope = ideal-mpf
circuit_bound_scope = repeated-shared-ancilla-good-block
circuit_bound_rigorous = true only when the product claim is available
circuit_target_satisfied = (the repeated bound meets the algorithm budget)
```

Benchmark summaries use one of three explicit policies:

- `implemented-circuit` (default): require a projected-good-block guarantee
  that meets the target;
- `declared-bound-scope`: accept a rigorous target-satisfying bound at its
  declared scope; ideal MPF rows may be included even when the repeated bound
  is too loose;
- `unconstrained`: permit heuristic/noncertified rows and label them as such.

## Analytical LCU/OAA resource structure

`mpf_lcu_structure` is shared by the circuit metadata and analytical counter.
For each segment with robust OAA, the counter uses:

- \(J\) physical branches with controlled second-order product formulas;
- two cancelling identity-padding branches with no product formula;
- any remaining computational basis states as unused identity branches;
- sign phases only for negative MPF coefficients and the negative padding
  branch;
- three SELECT calls, six PREPARE/inverse-PREPARE calls, and two good-subspace
  reflections.

A reusable equality flag is assumed for each physical branch. A `b`-bit branch
equality flag uses `b-1` temporary-AND compute/uncompute pairs. A sign or
reflection phase uses `b-2` such pairs plus one terminal Clifford CZ/CX. These
are architecture-dependent efficient-compilation assumptions; the benchmark
records `counting_mode="analytical-model"`. The small-system calibration test
also records the expected gap from Qiskit's generic ancilla-free `.control()`
decomposition.
