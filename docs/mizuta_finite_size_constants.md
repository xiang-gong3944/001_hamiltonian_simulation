# Finite-size constants in Mizuta's MPF error analysis

This note reconstructs the finite-resource content of Kaoru Mizuta,
"On the commutator scaling in Hamiltonian simulation with multi-product
formulas," *Quantum* **10**, 1974 (2026).  The authoritative source used here
is [arXiv:2507.06557v4](https://arxiv.org/abs/2507.06557v4), which is the
revision linked by the [published article](https://quantum-journal.org/papers/q-2026-01-19-1974/).
All logarithms are natural.

The main finite-size conclusion is simple but severe.  The published local
bound is fully explicit:

\[
\delta(\tau)
\leq
2\sqrt e\,\lVert c\rVert _1
   \bigl(c_p\mu_{p,m}[p_0]|\tau|\bigr)^{m+1}
+\lVert c\rVert _1\lVert k\rVert _1\eta,
\tag{1}
\]

provided

\[
p_0=\left\lceil\log\frac{3N}{\eta}\right\rceil,
\qquad
|\tau|\leq
\min\left\{
\frac{1}{8e^3c_pp_0kg},
\frac{1}{2c_p\mu_{p,m}[p_0]}
\right\}.
\tag{2}
\]

For the repository's representative four-qubit example, the first condition
in (2), not the measured commutator parameter, fixes the segment count.  At
\(t=0.01\), \(J=3\), and target error \(10^{-4}\), the approximation-error
condition alone asks for only three segments and the \(\mu\)-dependent time
condition asks for one, but the truncated-BCH time condition asks for 708.

## 1. Status labels and source map

The derivation below labels its statements as follows.

- **Identity:** an exact algebraic equality.
- **Paper inequality:** a rigorous explicit inequality printed or directly
  established in the v4 proof.
- **Asymptotic step:** a big-O, big-Theta, polylogarithmic, or
  "sufficiently large" replacement that does not retain a finite constant.
- **Same-ingredients tightening:** a rigorous consequence obtained by retaining
  a term that the proof later enlarges.  It is not silently attributed to the
  theorem as printed.
- **Unrecoverable:** the paper does not specify enough implementation detail to
  determine a numerical constant.

The fixed v4 source locations used for audit are:

| Published item | Source archive location | Role |
| --- | --- | --- |
| Eqs. (4), (6), (8)–(10), (23)–(24) | `2_Preliminary.tex:36-102` | Hamiltonian split, product formula, commutators, MPF, well conditioning |
| BCH Eqs. (41)–(46) | `3_Proof.tex:23-49` | BCH coefficients and \(\lVert\Phi_q\rVert\) |
| Theorem 3, Eqs. (47)–(49) | `3_Proof.tex:55-71` | Truncation order, first time condition, truncated-BCH error |
| Proof-level Eqs. (57)–(59) | `3_Proof.tex:106-141` | Explicit BCH remainder before \(3Ne^{-p_0}\) |
| Theorem 4, Eqs. (61)–(63) | `3_Proof.tex:165-183` | Finite \(\mu\), both time conditions, local MPF error |
| Proof-level Eqs. (64)–(67) | `3_Proof.tex:185-219` | Dyson expansion, cancellation, geometric bounds, branchwise BCH error |
| Lemma 5, Eqs. (68)–(71) | `3_Proof.tex:224-253` | Explicit locality-only upper bound on \(\mu\) |
| Theorem 6, Eqs. (74)–(87) | `3_Proof.tex:277-384` | Error allocation, \(r_1,r_2\), and asymptotic query count |
| Lemma 7, Eq. (90) | `A0_basic.tex:6-60` | Local-insertion nested-commutator bound |
| Lemma 8, Eqs. (97)–(102) | `A0_basic.tex:65-111` | BCH locality and extensiveness |
| Lemma 9, Eqs. (103)–(104) | `A1_truncated_BCH.tex:10-104` | Subsystem product-formula series bound |
| Lemma 10, Eqs. (115)–(116) | `A1_truncated_BCH.tex:201-308` | Subsystem truncated-BCH series bound |

The repository metadata currently cites "Theorem 4, Eqs. (47)--(49), with
Theorem 3, Eqs. (33)--(35)."  That metadata does not match the published v4
numbering: Eqs. (47)–(49) belong to Theorem 3, while Theorem 4 is Eqs.
(61)–(63).  This is a citation defect, not a change in the implemented
formula.

## 2. Notation and repository mapping

Mizuta assumes

\[
H=\sum_{X\subseteq\Lambda:|X|\leq k}h_X
 =\sum_{\gamma=1}^{\Gamma}H_\gamma,
\qquad
\sum_{X:X\ni i}\lVert h_X\rVert\leq g.
\tag{3}
\]

The order-\(p\) product formula is

\[
T_p(\tau)=
\prod_{v=1}^{c_p\Gamma}{}^{\leftarrow}
e^{-iH_{\gamma_v}\alpha_v\tau}.
\tag{4}
\]

The integer \(c_p\) counts how many copies of the complete Hamiltonian
partition occur.  Mizuta's convention gives
\(T_2(\tau)=T_1(-\tau/2)^\dagger T_1(\tau/2)\), so **\(c_2=2\)**.
This agrees with the ordered symmetric second-order formula in the repository.

The MPF is

\[
M_{pmJ}(\tau)=\sum_{j=1}^{J}c_j
   \left[T_p(\tau/k_j)\right]^{k_j},
\qquad
M_{pmJ}(\tau)=e^{-iH\tau}+O(\tau^{m+1}).
\tag{5}
\]

For an even symmetric base formula, Richardson cancellation with \(J\)
branches gives Mizuta formal order \(m=2J\).  The repository's public argument
historically named `m` is instead the branch count \(J\).

| This note | Mizuta | Repository |
| --- | --- | --- |
| \(J\) | number of MPF terms | `term_count`, legacy argument `m` |
| \(m=2J\) | formal MPF order | `formal_order=2*term_count` |
| \(a_j\) | coefficient \(c_j\) | `multiproduct_coefficients(J)` |
| \(A=\sum_j|a_j|\) | \(\lVert c\rVert_1\) | `coefficient_l1_norm` |
| \(K=\sum_j k_j\) | \(\lVert k\rVert_1\) | sum of `exponents` |
| \(W=\sum_j|a_j|k_j\) | retained proof-level weight | not used by the estimator |
| \(p=2,c_p=2\) | base formula parameters | ordered symmetric second order |
| \(\tau=t/r\) | local simulated time | `local_step_size` |
| \(\alpha_{\mathrm{com},q}\) | Eq. (8) | exact/upward-rounded Pauli recurrence |
| \(\eta\) | Theorem 3/4 auxiliary error | `auxiliary_error` |
| \(p_0\) | finite BCH/commutator cutoff | `truncation_order_p0` |

For individual Pauli terms \(H_\gamma=h_\gamma P_\gamma\), the repository
computes

\[
k=\max_\gamma|\operatorname{supp}P_\gamma|,
\qquad
g=\max_i\sum_{\gamma:i\in\operatorname{supp}P_\gamma}|h_\gamma|.
\tag{6}
\]

## 3. Dependency graph

```text
Hamiltonian locality/extensiveness, Eqs. (1)–(5)
          |
          +--> Eq. (8): alpha_com,q <= (q-1)! (2kg)^(q-1) N g
          |        |
          |        +--> Lemma 5: explicit locality-only mu bound
          |
          +--> Lemma 7: nested commutator with one local insertion
                   |
                   +--> Lemma 8: Phi_q is qk-local and explicitly extensive
                   |        |
                   |        +--> Lemma 10: truncated-BCH subsystem series
                   |
                   +--> Lemma 9: product-formula subsystem series
                            |
                            +--> Theorem 3: finite truncated-BCH remainder
                                      |
BCH identity, Eqs. (41)–(46) --------+
                                      |
MPF Richardson identities -----------+--> Theorem 4: local MPF error
                                               |
                                               +--> Theorem 6: r1, r2,
                                                    asymptotic queries/gates
```

Theorem 3 is the only point where the factor
\(8e^3c_pp_0kg\) enters.  Theorem 4 imports that hypothesis, then introduces a
different \(\mu\)-dependent hypothesis to collapse its Dyson tail.

## 4. Nested commutators and BCH coefficients

### 4.1 Hamiltonian commutators

Mizuta defines exactly

\[
\alpha_{\mathrm{com},q}
=\sum_{\gamma_1,\ldots,\gamma_q=1}^{\Gamma}
\left\lVert
[H_{\gamma_q},\ldots,[H_{\gamma_2},H_{\gamma_1}]]
\right\rVert.
\tag{7}
\]

Locality and extensiveness give the **paper inequality**

\[
\alpha_{\mathrm{com},q}
\leq(q-1)!(2kg)^{q-1}Ng.
\tag{8}
\]

The factorial is essential at finite size and cannot be absorbed into a
constant if \(q\) grows.  This is precisely why Mizuta truncates the relevant
commutator orders.

Lemma 7 proves the related local-insertion bound

\[
\sum_{\gamma_1,\ldots,\gamma_q}
\left\lVert
[H_{\gamma_q},\ldots,[O_X,\ldots,[H_{\gamma_2},H_{\gamma_1}]]]]
\right\rVert
\leq q!(2kg)^q\lVert O_X\rVert.
\tag{9}
\]

### 4.2 BCH coefficients

Within its convergence radius the BCH expansion is the **identity**

\[
T_p(\tau)=\exp\left(-iH\tau-i\sum_{q=2}^{\infty}\Phi_q\tau^q\right).
\tag{10}
\]

The explicit right-nested-commutator expression is given in published Eqs.
(42)–(45).  It implies the **paper inequality**

\[
\lVert\Phi_q\rVert
\leq\frac{c_p^q}{q^2}\alpha_{\mathrm{com},q},
\qquad q\geq p+1,
\tag{11}
\]

and \(\Phi_q=0\) for \(2\leq q\leq p\).  The factor \(q^{-2}\) is later
dropped when a single \(\mu\) controls all orders; retaining it is a possible
finite-size improvement.

Lemma 8 also proves that \(\Phi_q\) is \(qk\)-local and

\[
g(\Phi_q)
\leq\frac{(q-1)!}{q}(2c_pkg)^{q-1}c_pg.
\tag{12}
\]

## 5. Theorem 3 with the remainder exposed

Define subsystem product formulas and truncated BCH operators as in Eqs.
(50)–(54).  The site-by-site telescoping decomposition is an **identity**.
Lemmas 9 and 10 establish

\[
\lVert T_q^i\rVert\leq(4c_pkg)^q,
\qquad
\lVert\widetilde T_q^i\rVert
\leq\frac{e^2}{2}(8e^2c_pp_0kg)^q.
\tag{13}
\]

Consequently, before selecting a time hypothesis, the proof gives the
**paper inequality**

\[
R_{\mathrm{BCH}}(p_0,\tau)
\leq N\sum_{q=p_0+1}^{\infty}
\left[(4c_pkg|\tau|)^q
+\frac{e^2}{2}(8e^2c_pp_0kg|\tau|)^q\right].
\tag{14}
\]

Whenever both ratios are smaller than one, the strongest closed form directly
recoverable at this point is

\[
R_{\mathrm{BCH}}(p_0,\tau)
\leq N\left[
\frac{u^{p_0+1}}{1-u}
+\frac{e^2}{2}\frac{v^{p_0+1}}{1-v}
\right],
\quad
u=4c_pkg|\tau|,
\quad
v=8e^2c_pp_0kg|\tau|.
\tag{15}
\]

Equation (15) is a **same-ingredients tightening**: Eq. (14) is printed, and
only the two geometric sums have been evaluated.

Theorem 3 instead imposes

\[
|\tau|\leq\frac{1}{8e^3c_pp_0kg}.
\tag{16}
\]

This makes \(v\leq e^{-1}\), and also makes the first ratio smaller than
\(e^{-1}\).  The proof then obtains

\[
R_{\mathrm{BCH}}(p_0,\tau)
\leq
N\sum_{q=p_0+1}^{\infty}
\left(e^{-q}+\tfrac12e^{-q+2}\right)
\leq3Ne^{-p_0}.
\tag{17}
\]

For an arbitrary auxiliary \(\eta\in(0,1)\), choosing

\[
p_0(N,\eta)=\left\lceil\log\frac{3N}{\eta}\right\rceil
\tag{18}
\]

gives \(3Ne^{-p_0}\leq\eta\).  This is the complete origin of Theorem 3's
truncation order and error.  Condition (16) is a convenient sufficient
condition for the selected geometric comparison.  The proof does not claim it
is necessary for a particular Hamiltonian, product formula, or truncation.

## 6. Finite \(\mu_{p,m}[p_0]\)

Theorem 4 defines

\[
\mu_{p,m}[p_0]
=\sup_{\substack{q,n\in\mathbb N:\ q\geq m+1,\\
n\leq\lfloor(q-1)/p\rfloor}}
\left[
\sum_{\substack{p+1\leq q_1,\ldots,q_n\leq p_0\\
q_1+\cdots+q_n=q+n-1}}
\prod_{i=1}^{n}\alpha_{\mathrm{com},q_i}
\right]^{1/(q+n-1)}.
\tag{19}
\]

Only \(\alpha_{\mathrm{com},q}\) through \(p_0\) is required, although the
repetition index \(n\), and therefore the outer index \(q\), is unbounded.

### 6.1 Bound in the paper

Lemma 5 combines Eq. (8), endpoint maximization of \(qN^{1/q}\), and a
composition count to prove

\[
\mu_{p,m}[p_0]
\leq4\max\left\{
(p+1)N^{1/(p+1)},e^3p_0
\right\}kg.
\tag{20}
\]

The constants arise as follows: the outer 4 comes from bounding a composition
count by a power of 2; \(e^3\) replaces the retained endpoint factor
\(N^{1/p_0}\).  Equation (20), not the following big-O statement, is the
strongest explicit locality-only inequality in the lemma.

### 6.2 Polynomial-root identity used by the repository

For supplied nonnegative finite commutator data, define

\[
A(z)=\sum_{s=p+1}^{p_0}\alpha_{\mathrm{com},s}z^s
\tag{21}
\]

and let \(z_*>0\) solve \(A(z_*)=1\).  If \(D=q+n-1\), the inner sum in
Eq. (19) is \([z^D]A(z)^n\).  Nonnegativity gives

\[
[z^D]A(z)^n z_*^D\leq A(z_*)^n=1,
\tag{22}
\]

so every candidate in (19) is at most \(1/z_*\).

For the reverse inequality, set
\(\pi_s=\alpha_{\mathrm{com},s}z_*^s\).  These values form a probability
distribution.  The sum of \(n\) independent draws has at most
\(n(p_0-p-1)+1\) possible values, so some coefficient is at least the inverse
of that number.  Its degree \(D\) grows linearly in \(n\), while its paper
index satisfies \(q=D-n+1\geq np+1\), eventually exceeding every fixed
formal-order cutoff \(m\).  Taking the \(D\)-th root makes the polynomial loss
converge to one.  Therefore

\[
\boxed{
\mu_{p,m}[p_0]=\mu_*=1/z_*\quad\Longleftrightarrow\quad
\sum_{s=p+1}^{p_0}\frac{\alpha_{\mathrm{com},s}}{\mu_*^s}=1
}
\tag{23}
\]

for nonzero supplied data; \(\mu=0\) when every supplied commutator is zero.
This is a **rigorous repository-side derivation**, not a result stated by
Mizuta.  If some supplied \(\alpha_{\mathrm{com},s}\) are upper bounds rather
than exact values, the root is an upper bound on the true \(\mu\).

An important consequence is that for fixed finite commutator data the
supremum is independent of every fixed \(m\).  Formal order still improves the
power \(m+1\) in the local error.

## 7. Theorem 4 local error, with every enlargement visible

Set

\[
x=c_p\mu_{p,m}[p_0]|\tau|.
\tag{24}
\]

The Dyson formula in published Eq. (64) is an **identity**.  Richardson
cancellation removes terms through order \(m\).  Published Eq. (66) first
gives the branchwise **paper inequality**

\[
\begin{aligned}
E_{\mathrm{MPF}}
\leq{}&
\sum_{j=1}^{J}|c_j|
\sum_{q=m+1}^{\infty}
\sum_{n=1}^{\lfloor(q-1)/p\rfloor}
\frac{|\tau|^{q+n-1}}{k_j^{q-1}n!}\\
&\times
\sum_{\substack{p+1\leq q_1,\ldots,q_n\leq p_0\\
q_1+\cdots+q_n=q+n-1}}
\prod_{i=1}^{n}\lVert\Phi_{q_i}\rVert.
\end{aligned}
\tag{25}
\]

This is the strongest displayed local MPF inequality: it retains every
\(k_j\), the finite composition constraint, and the BCH coefficient norms.
Applying Eqs. (11) and (19), dropping \(k_j^{-(q-1)}\leq1\), and extending the
\(n\)-sum gives

\[
E_{\mathrm{MPF}}
\leq A\sum_{q=m+1}^{\infty}\sum_{n=1}^{\infty}
\frac{x^{q+n-1}}{n!}.
\tag{26}
\]

For \(x<1\), this upper bound can be summed exactly:

\[
E_{\mathrm{MPF}}
\leq A\frac{x^m(e^x-1)}{1-x}.
\tag{27}
\]

Equation (27) is a **same-ingredients tightening** of the next printed line.
The paper instead uses \(e^x-1\leq xe^x\) and then \(x\leq1/2\):

\[
A\frac{x^{m+1}e^x}{1-x}
\leq2e^{1/2}Ax^{m+1}.
\tag{28}
\]

This is the sole origin of the second time hypothesis

\[
|\tau|\leq\frac{1}{2c_p\mu_{p,m}[p_0]}.
\tag{29}
\]

It is sufficient for the constant in (28), not a necessary condition for the
underlying finite-dimensional Dyson series.  Equation (27) only needs
\(x<1\); retaining the finite generating function can loosen it further.

For each branch, Theorem 3 bounds one substep at time \(\tau/k_j\) by
\(\eta\).  Repeating that substep \(k_j\) times and taking the weighted sum
first gives

\[
E_{\mathrm{trunc}}
\leq\left(\sum_j|c_j|k_j\right)\eta=W\eta.
\tag{30}
\]

The theorem enlarges this to

\[
W\eta\leq AK\eta.
\tag{31}
\]

Combining (28), (31), and the two hypotheses (16), (29) gives the published
Theorem 4 result, Eq. (1).  Thus the strongest compact inequality printed as a
theorem is (1); the strongest directly recoverable proof-level version is
(25) plus the branchwise Theorem 3 remainder.

## 8. Segment-count prescriptions

There are three different questions that should not be conflated.

### 8.1 The paper's explicit coarse count

Theorem 6 chooses

\[
\eta=\frac{\varepsilon}{4AKr}
\tag{32}
\]

and requires the printed local bound to be at most \(\varepsilon/(2r)\).
It then replaces \(\sqrt e\) by \(2^m\) and replaces \(\mu\) by Lemma 5's
locality-only bound.  The two explicit sufficient quantities are

\[
\begin{aligned}
r_{1,m}={}&
8c_p(p+1)kN^{1/(p+1)}gt
\left[
\frac{32c_p(p+1)kA N^{1/(p+1)}gt}{\varepsilon}
\right]^{1/m},\\
r_{2,m}={}&
40e^4c_pkgt(m+1)
\left(\frac{160e^3Ac_pkgt}{\varepsilon}\right)^{1/m}\\
&\times
\log^{1+1/m}\left[
\frac{(8e^3c_pkgt)(12AKN)}{\varepsilon}
\right].
\end{aligned}
\tag{33}
\]

The paper takes

\[
r_{\mathrm{coarse}}=\left\lceil\max(r_{1,m},r_{2,m})\right\rceil.
\tag{34}
\]

The derivation of \(r_{2,m}\) uses published Eq. (78), which explicitly
requires

\[
a=\frac{\varepsilon}{4A(8e^3c_pkgt)^{m+1}}
\left(\frac{\varepsilon}{12AKN}\right)^m
\in(0,1/5].
\tag{35}
\]

The source then invokes interest in "sufficiently large" \(N,t\), or
\(1/\varepsilon\) to assume (35).  Therefore Eq. (33) is not an unconditional
finite-size formula unless (35) is checked.

### 8.2 Smallest count using Theorem 4 as printed

Suppose commutator data are available through a maximum order \(P\).  For each
integer \(p_0\leq P\) with

\[
\eta_{p_0}=3Ne^{-p_0}\in(0,1),
\tag{36}
\]

compute \(\mu[p_0]\) from Eq. (19), or its rigorous root upper bound (23).
Define

\[
\delta_{p_0}(r)=
2\sqrt e\,A
\left(\frac{c_p\mu[p_0]|t|}{r}\right)^{m+1}
+AK\eta_{p_0}.
\tag{37}
\]

The smallest segment count certified by Theorem 4 and the paper's own local
condition, without the Lemma 5 and logarithmic relaxations, is

\[
\boxed{
r_{\mathrm{Mizuta}}^{\mathrm{Thm\,4}}
=\min\left\{
r\in\mathbb N:\exists p_0\leq P,
\begin{array}{l}
|t|/r\leq(8e^3c_pp_0kg)^{-1},\\
|t|/r\leq(2c_p\mu[p_0])^{-1},\\
\delta_{p_0}(r)\leq\varepsilon/(2r)
\end{array}
\right\}.
}
\tag{38}
\]

This is a finite algorithm: enumerate \(p_0\), solve the one-dimensional
integer inequality, and take the minimum.  If no candidate occurs before
\(P\), the supplied commutator data are insufficient to certify a count.

For a fixed \(p_0\) and the equal error split in Eq. (32), useful diagnostic
lower bounds are

\[
\begin{aligned}
r_{\mathrm{error}}&=
\left\lceil
\left[
\frac{8\sqrt e\,A(c_p\mu[p_0]|t|)^{m+1}}{\varepsilon}
\right]^{1/m}
\right\rceil,\\
r_{\mathrm{time},1}&=
\left\lceil8e^3c_pp_0kg|t|\right\rceil,\\
r_{\mathrm{time},2}&=
\left\lceil2c_p\mu[p_0]|t|\right\rceil.
\end{aligned}
\tag{39}
\]

There is no independent \(r_{\mathrm{trunc}}\) under Eq. (32): the choice of
\(p_0\) makes the truncated-BCH contribution consume its assigned
\(\varepsilon/(4r)\) local budget.  The truncation mechanism affects \(r\)
indirectly because increasing \(p_0\) tightens \(\eta\) but raises
\(r_{\mathrm{time},1}\) and can increase \(\mu[p_0]\).  Reporting a fourth
independent lower bound would double-count the first time hypothesis.

### 8.3 Same-ingredients tightening

For comparison, retain Eqs. (15), (27), and (30).  At \(\tau=t/r\), define

\[
\delta_{\mathrm{tight}}(r,p_0)=
A\frac{x^m(e^x-1)}{1-x}
+\sum_j|c_j|k_j
R_{\mathrm{BCH}}(p_0,\tau/k_j),
\quad x=c_p\mu[p_0]|\tau|,
\tag{40}
\]

using Eq. (15) for each branch.  This requires only the corresponding
geometric ratios to be below one.  For the repeated ideal MPF, the exact
telescoping sum implied by \(\lVert M-U\rVert\leq\delta\) and
\(\lVert M\rVert\leq1+\delta\) is

\[
\lVert M^r-U^r\rVert
\leq\delta\sum_{j=0}^{r-1}(1+\delta)^j
=(1+\delta)^r-1.
\tag{41}
\]

Thus a separately labelled tightened count is the minimum \(r,p_0\) for which
all denominators in (40) are positive and

\[
(1+\delta_{\mathrm{tight}}(r,p_0))^r-1\leq\varepsilon.
\tag{42}
\]

Equations (40)–(42) do not change the mathematical strategy, but they are not
the theorem printed in the paper.  They are used only as a diagnostic in the
audit script.

## 9. Queries and gates: what is and is not explicit

Theorem 6 says that deterministic LCU plus quantum amplitude amplification
uses

\[
O(AKr)
\tag{43}
\]

queries to controlled \(T_p\).  The numerical constant is **unrecoverable**
from Mizuta's paper: the QAA construction is cited rather than fixed, and the
well-conditioned schedules are used only through asymptotic conditions.
Likewise, the paper's gate table multiplies by an \(O(N)\) or \(O(N^k)\)
oracle cost, so it does not imply a numerical gate count.

The repository makes additional implementation choices that must be labelled
separately.  It pads \(A<2\) to normalization two and applies one three-SELECT
robust-OAA round per segment.  For that circuit structure,

\[
N_{\mathrm{SELECT}}=3r,
\qquad
N_{C[T_2]}=3Kr,
\qquad
N_{\mathrm{controlled\ exponentials}}=3Kr\,c_2\Gamma.
\tag{44}
\]

These are exact repository structural counts, not constants recovered from
Theorem 6.  They also do not by themselves certify the repository's repeated
shared-ancilla good block; that circuit scope is treated separately in
`docs/mpf_error_bounds.md`.

## 10. Numerical audit

The tables below are generated by

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_mizuta_finite_size.py --check
```

The model is the open-boundary transverse-field Ising chain

\[
H=-\sum_{i=1}^{N-1}Z_iZ_{i+1}-3\sum_{i=1}^{N}X_i.
\tag{45}
\]

For \(N=4\), its individual-Pauli decomposition has \(\Gamma=7\), \(k=2\),
and \(g=5\).  At \(J=3\),

\[
(k_1,k_2,k_3)=(1,2,4),
\qquad
(a_1,a_2,a_3)=\left(\frac1{45},-\frac49,\frac{64}{45}\right),
\tag{46}
\]

so

\[
A=\frac{17}{9},\qquad K=7,\qquad W=\frac{33}{5}.
\tag{47}
\]

### 10.1 Which constraint is active?

| \(N\) | \(J\) | \(t\) | \(\varepsilon\) | \(p_0\) | \(\mu\) | \(r_{\rm error}\) | \(r_{\rm time,1}\) | \(r_{\rm time,2}\) | repository \(r\) | active |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 4 | 2 | 0.01 | \(10^{-4}\) | 21 | 17.4126 | 6 | 675 | 1 | 675 | time 1 |
| 4 | 3 | 0.01 | \(10^{-3}\) | 20 | 17.3990 | 2 | 643 | 1 | 643 | time 1 |
| 4 | 3 | 0.01 | \(10^{-4}\) | 22 | 17.4230 | 3 | 708 | 1 | 708 | time 1 |
| 4 | 3 | 0.01 | \(10^{-6}\) | 27 | 17.4535 | 6 | 868 | 1 | 868 | time 1 |
| 4 | 4 | 0.01 | \(10^{-4}\) | 23 | 17.4320 | 2 | 740 | 1 | 740 | time 1 |
| 4 | 3 | 4 | \(10^{-4}\) | 28 | 17.4567 | 2524 | 359933 | 280 | 359933 | time 1 |

The quantity labelled \(r_{\rm error}\) uses the paper's equal local error
split.  `r_trunc` is not shown because it is absorbed into \(\eta,p_0\), as
explained after Eq. (39).

### 10.2 Coarse, theorem-driven, and tightened counts

| \(N\) | \(J\) | \(t\) | \(\varepsilon\) | Eqs. (33)–(34) | Theorem 4 minimum | same-ingredients tightening |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | 0.01 | \(10^{-4}\) | 4,684,171 | 675 | 529 |
| 4 | 3 | 0.01 | \(10^{-3}\) | 850,871 | 643 | 355 |
| 4 | 3 | 0.01 | \(10^{-4}\) | 1,450,434 | 708 | 436 |
| 4 | 3 | 0.01 | \(10^{-6}\) | 4,018,869 | 868 | 594 |
| 4 | 4 | 0.01 | \(10^{-4}\) | 868,194 | 740 | 327 |
| 4 | 3 | 4 | \(10^{-4}\) | 2,164,036,029 | 359,933 | 256,462 |

The first numerical column is intentionally enormous: it uses Lemma 5 and all
of Theorem 6's scaling-oriented relaxations.  The middle column uses the exact
finite commutators in the printed Theorem 4 bound.  The last column is the
separately labelled proof-level tightening, not the published theorem.

### 10.3 Why hundreds of segments occur with moderate \(\mu\)

For \(N=4,J=3,t=0.01,\varepsilon=10^{-4}\), the repository chooses \(p_0=22\)
and obtains \(\mu=17.4230497\).  The three diagnostic quantities are

\[
\begin{aligned}
r_{\mathrm{error}}&=3,\\
r_{\mathrm{time},2}
&=\left\lceil2\cdot2\cdot17.4230497\cdot0.01\right\rceil=1,\\
r_{\mathrm{time},1}
&=\left\lceil
8e^3\cdot2\cdot22\cdot2\cdot5\cdot0.01
\right\rceil=708.
\end{aligned}
\tag{48}
\]

The local commutator contribution at the selected \(r\) is only
\(4.36\times10^{-23}\), whereas the allocated truncated-BCH contribution is
\(7.06\times10^{-8}\).  The poor count therefore does not diagnose a large
physical commutator.  It diagnoses the universal sufficient constant in
Theorem 3.

Lemma 5 makes the source of the coincidence even clearer.  When its
\(e^3p_0\) branch dominates,

\[
\mu\leq4e^3p_0kg
\quad\Longrightarrow\quad
2c_p\mu|t|\leq8e^3c_pp_0kg|t|.
\tag{49}
\]

Thus the locality-only upper bound makes the second time constraint as large
as the first.  Exact commutators dramatically reduce the second constraint,
but they cannot reduce the first.

At the default proportional time \(t=N=4\), the same phenomenon gives
\(r_{\mathrm{time},1}=359{,}933\), while the commutator-error and
\(\mu\)-time diagnostics are respectively 2,524 and 280.

For the short-time sample, the repository-specific structure in Eq. (44)
contains 2,124 SELECT calls, 14,868 controlled \(T_2\) queries, and 208,152
controlled Pauli exponentials.  Those numbers are circuit-structure counts,
not a finite prefactor proved by Mizuta.

## 11. Constant-origin table

| Constant | First relevant source | Origin and status |
| --- | --- | --- |
| \((q-1)!\), \((2kg)^{q-1}Ng\) | Eq. (8), `2_Preliminary.tex:58-62` | Explicit locality/extensiveness commutator count |
| \(c_p^q/q^2\) | Eq. (46), `3_Proof.tex:41-45` | Explicit BCH right-nested-commutator bound |
| \(4c_pkg\) | Lemma 9, `A1_truncated_BCH.tex:10-104` | Subsystem product-formula causal-cone series |
| \(e^2(8e^2c_pp_0kg)^q/2\) | Lemma 10, `A1_truncated_BCH.tex:201-308` | Composition counts and exponential sums for truncated BCH |
| \(8e^3c_pp_0kg\) | Theorem 3 Eq. (48) | Makes both BCH-tail ratios comparable to \(e^{-1}\) |
| 3 in \(3Ne^{-p_0}\) | Theorem 3 proof Eq. (59) | Upper bound on two explicit geometric tails |
| \(2\sqrt e\) | Theorem 4 proof Eq. (66) | \(x\leq1/2\), \(e^x\leq\sqrt e\), \((1-x)^{-1}\leq2\) |
| \(AK\) | Theorem 4 proof Eq. (67) | Enlargement of the directly available \(W=\sum|c_j|k_j\) |
| 4 in Lemma 5 | Eq. (71), `3_Proof.tex:246-253` | Coarse composition-count bound |
| \(e^3p_0\) | Eqs. (70)–(71) | Replaces retained endpoint \(p_0N^{1/p_0}\) |
| \(8,32\) in \(r_1\) | Eq. (77) | Error split plus \(\sqrt e\to2^m\) and Lemma 5 |
| \(40e^4,160e^3\) in \(r_2\) | Eqs. (81)–(83) | Log-over-power inequality, \(5^{1+1/m}\), and \((m+1)^{1+1/m}\leq e(m+1)\) |
| QAA/query prefactor | Theorem 6 text | Unrecoverable; only \(O(AKr)\) is stated |
| Per-oracle gate prefactor | Table 1 caption | Unrecoverable; only \(O(N)\) or \(O(N^k)\) is stated |

## 12. Where finite-size information is suppressed

| Location | Suppression |
| --- | --- |
| Eq. (7) | Product-formula error is written as \(O(\alpha_{\rm com,p+1}\tau^{p+1})\) despite Eq. (8) being explicit |
| Eqs. (23)–(24) | Well-conditioned \(K\) and \(A\) are only required to be polynomial in \(J\) |
| Informal Theorem 2, Eqs. (36)–(39) | Both time and local error are summarized with big-O |
| Lemma 5 after Eq. (68) | Explicit max is replaced by \(O([N^{1/(p+1)}+\log(N/\eta)]g)\) |
| Eqs. (72)–(73) | The explicit Theorem 4 conditions and prefactors are replaced by big-O |
| Theorem 6 Eq. (74) | Query complexity is big-O with an unspecified polylogarithmic factor |
| Theorem 6 proof before Eq. (76) | \(\sqrt e\) is deliberately enlarged to \(2^m\) |
| Eq. (78) application | The finite condition \(a\leq1/5\) is justified only by "sufficiently large" parameters |
| Eqs. (84)–(87) | \(m\), \(A\), \(K\), \(r_1\), and \(r_2\) are converted to asymptotic scaling |
| Table 1 | \([\log(Ngt/\varepsilon)]^2\) is absorbed into `polylog`; oracle gate constants are omitted |

## 13. Audit of the current implementation

The method `mizuta2026-commutator-ideal-rigorous` is a rigorous upper bound for
its declared **ideal-MPF** scope, subject to ordinary floating-point
upper-rounding assumptions recorded by the implementation.  It correctly:

1. maps repository \(J\) to Mizuta \(m=2J\);
2. uses \(p=2,c_2=2\);
3. computes \(k\) from Pauli support and \(g\) from per-site coefficient sums;
4. computes finite \(\alpha_{\mathrm{com},q}\) exactly for Pauli terms, with
   the explicit Eq. (8) fallback when necessary;
5. chooses adaptive \(p_0\), checks both time hypotheses, and evaluates
   Theorem 4's \(2\sqrt e\) local bound;
6. uses the rigorous polynomial-root upper bound for \(\mu\);
7. searches for the smallest integer satisfying its implemented predicate.

It is not the strongest finite-size bound recoverable from the same proof:

- it uses \(AK\) instead of \(W\);
- it uses \(2\sqrt e x^{m+1}\) instead of Eq. (27) or the branchwise Eq. (25);
- it uses the single \(\eta\) remainder instead of Eq. (15) per branch;
- it fixes half of a local budget for truncation rather than jointly optimizing
  the error allocation and \(p_0\);
- its repeated-step envelope is
  \(r\delta(1+\delta)^{r-1}\), which is looser than the exact geometric sum
  \((1+\delta)^r-1\);
- its comment describing
  \((1+\varepsilon)^{1/r}-1\) as the exact local budget corresponds to the
  latter geometric sum, not to the looser envelope the code actually tests;
- it enforces \(p_0\geq3\), an extra harmless restriction for the benchmark
  regime;
- its theorem/equation metadata has the numbering reversal noted in Section 1.

The polynomial root itself is not a source of looseness for fixed supplied
commutator bounds: Eq. (23) shows it is sharp for those inputs.  In the
representative points, improving \(\mu\) does little because
\(r_{\mathrm{time},1}\) is active.

Finally, none of the ideal-MPF statements silently certifies the repository's
repeated shared-ancilla robust-OAA circuit.  That distinction in the existing
resource model remains necessary.

## 14. Tightenings that preserve the strategy

The following changes are mathematically compatible with the paper's method,
but must remain labelled as refinements rather than quotations of Theorem 4:

1. evaluate the two BCH geometric tails in Eq. (15) instead of replacing them
   by \(3Ne^{-p_0}\);
2. retain branch times \(\tau/k_j\), the factors \(k_j^{-(q-1)}\), and the
   \(q_i^{-2}\) factors as far as the available data allow;
3. use \(W=\sum|c_j|k_j\) rather than \(AK\);
4. use Eq. (27), with \(x<1\), instead of Eq. (28), with \(x\leq1/2\);
5. optimize \(p_0\) and the commutator/truncation error allocation jointly;
6. use the exact geometric telescoping sum (41) for the repeated ideal MPF;
7. retain \(p_0N^{1/p_0}\) and the exact composition count if Lemma 5 must be
   used instead of measured commutators.

For the central \(N=4,J=3,t=0.01,\varepsilon=10^{-4}\) point, the implemented
Theorem 4 count is 708 and the conservative same-ingredients calculation in
the audit script is 436.  This is a meaningful constant improvement, but it is
still hundreds of segments: the subsystem BCH coefficient bound remains the
dominant conservative ingredient.

## 15. Bottom line

- The strongest compact local theorem in the paper is Eq. (1), with the two
  explicit hypotheses in Eq. (2).
- The strongest inequalities available immediately before simplification are
  the explicit BCH tails (14), the branchwise MPF sum (25), and the weighted
  truncation term (30).
- A smallest rigorous count from finite commutator data is the finite search
  (38), not the asymptotic complexity in Theorem 6.
- The first time hypothesis commonly dominates finite examples.  It can erase
  nearly all numerical benefit of a moderate measured \(\mu\).
- Mizuta's paper does not contain enough information to recover a numerical
  QAA/query prefactor or a per-oracle gate prefactor.
- The repository implements a rigorous version of the printed Theorem 4 local
  bound for the ideal MPF, but it does not implement the strongest finite-size
  bound recoverable from the proof.
