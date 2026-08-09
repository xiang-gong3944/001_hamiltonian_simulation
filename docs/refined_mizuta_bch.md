# Refined finite-size BCH remainder for Mizuta MPF sizing

This note documents the proof implemented by
`mizuta2026-commutator-ideal-rigorous`. The separate identifier
`mizuta2026-theorem3-legacy-ideal-rigorous` reproduces Mizuta's printed
Theorems 3--4, including its auxiliary error, allocation optimization, and
first sufficient time condition.

The refinement changes only the proof-induced control of the BCH truncation
terms. It does **not** change the Hamiltonian, the symmetric product formula,
the MPF coefficients or exponents, Richardson cancellation, the segment
circuit, or the physical MPF algorithm.

## 1. From the printed theorem to a direct remainder

For one MPF step of duration \(\tau\), the printed proof first bounds every
branch remainder by one common auxiliary \(\eta\). The refined path retains
the branch time and evaluates the remainder before that replacement:

\[
\delta_{\mathrm{BCH}}(\tau,p_0)
=\sum_j |c_j|k_j
R_{p_0}\!\left(\frac{\tau}{k_j}\right),
\qquad
R_{p_0}(x)
=N\sum_{q=p_0+1}^{\infty}(A_q+B_q)|x|^q.
\tag{1}
\]

The commutator contribution is unchanged:

\[
\delta_{\mathrm{comm}}(\tau,p_0)
=2\sqrt e\,\lVert c\rVert_1
\left(c_p\mu_{p,m}[p_0]|\tau|\right)^{m+1}.
\tag{2}
\]

The local and repeated bounds are

\[
\delta_{\mathrm{local}}
=\delta_{\mathrm{comm}}+\delta_{\mathrm{BCH}},
\qquad
E_r=r\delta_{\mathrm{local}}
(1+\delta_{\mathrm{local}})^{r-1}.
\tag{3}
\]

The refined estimator retains the second Theorem-4 hypothesis

\[
|\tau|\leq\frac{1}{2c_p\mu_{p,m}[p_0]},
\tag{4}
\]

but the printed first condition is not a refined-path requirement. It is
reported for comparison and is used only if the refined tail certificate
fails and the legacy tail fallback is invoked.

## 2. Schedule-weighted extensiveness

Write the actual Suzuki formula as

\[
T_p(\tau)=\prod_v e^{-i\alpha_vH_{\gamma_v}\tau}.
\tag{5}
\]

Every triangle inequality in Lemma 9 is applied after expanding derivatives
of these factors. Therefore occurrence \(v\) contributes
\(|\alpha_v|\), not one. Regrouping schedule occurrences by Hamiltonian group
defines

\[
w_\gamma=\sum_{v:\gamma_v=\gamma}|\alpha_v|,
\qquad
g_\alpha=\max_i\sum_\gamma w_\gamma
\sum_{X\ni i}\lVert h_X^\gamma\rVert.
\tag{6}
\]

To connect this explicitly to Lemma 9, replace each occurrence in the sums of
Eqs. (109), (112), and (114) by the positive decomposition

\[
\widehat H_v=|\alpha_v|
\sum_X h_X^{\gamma_v}.
\tag{7}
\]

The nested-Hamiltonian sum in Eq. (109) and the local-insertion sum in Eq.
(112) are multilinear in the occurrence Hamiltonians after norms are taken.
Summing the occurrence labels first turns every occurrence factor into the
corresponding \(w_\gamma\). The resulting positive decomposition is still
\(k\)-local, while its per-site norm sum is exactly bounded by
\(g_\alpha\). Lemma 7's insertion argument consequently applies with
\(g_\alpha\) at each insertion. Equation (114) then sums those same positive
occurrence contributions; no additional schedule multiplicity remains.

This is the proof of the schedule-weighted replacement. It is not the
numerical substitution \(c_pg\mapsto g_\alpha\) made after the fact.

The implementation obtains \((\gamma_v,\alpha_v)\) from
`suzuki_group_factors`. For Strang splitting, every noncentral group occurs in
two half steps and the central group occurs once, so

\[
w_\gamma=\tfrac12+\tfrac12=1
\quad\text{or}\quad w_\gamma=1.
\tag{8}
\]

Thus the actual schedule recovers \(g_\alpha=g\), rather than automatically
paying the coarse count \(c_2g=2g\).

## 3. Order-resolved Lemma 9

Let \(a=2kg_\alpha\) and \(b=g_\alpha\). Keeping the number \(n\) of local
insertions before the final simplification of Lemma 9 gives

\[
A_q=\sum_{n=1}^{q}
\frac{a^{q-n}b^n}{n!}\binom{q-1}{n-1}.
\tag{9}
\]

The binomial coefficient counts compositions of total order \(q\) into
\(n\) positive insertion blocks. Summing those compositions gives

\[
\begin{aligned}
A(z)
&=1+\sum_{q\geq1}A_qz^q\\
&=\sum_{n\geq0}\frac1{n!}
\left(\frac{bz}{1-az}\right)^n
=\exp\!\left(\frac{g_\alpha z}{1-2kg_\alpha z}\right).
\end{aligned}
\tag{10}
\]

Differentiating (10) yields the positive production recurrence

\[
qA_q=b\sum_{j=0}^{q-1}(j+1)a^jA_{q-1-j},
\qquad A_0=1.
\tag{11}
\]

Tests compare (11) with the finite sum (9), including the exact Strang
schedule weights.

## 4. Order-resolved Lemma 8 data

For the truncated logarithm

\[
\widetilde H(\tau)=\sum_{s=1}^{p_0}\Phi_s\tau^{s-1},
\tag{12}
\]

the proof supplies order-specific locality and extensiveness bounds

\[
K_s=sk,
\qquad
G_s\geq g(\Phi_s).
\tag{13}
\]

The implemented pre-simplification bound is

\[
G_1=g_\alpha,
\qquad
G_s=\frac{(s-1)!}{s}
(2kg_\alpha)^{s-1}g_\alpha
\quad(3\leq s\leq p_0,\ s\text{ odd}).
\tag{14}
\]

Symmetric Strang splitting has an odd logarithm. Therefore production sets

\[
G_2=G_4=G_6=\cdots=0
\tag{15}
\]

exactly, rather than allowing floating cancellation to approximate the parity
zeros.

The subsystem-difference coefficient \(\Psi_s^i\) in Lemma 10 is controlled
by the local-insertion version of the same \(G_s\): factors supported entirely
outside the subsystem cancel, and every surviving occurrence already carries
the weight included in (6). Consequently no extra \(c_p\) or schedule count
is introduced when passing from \(\Phi_s\) to \(\Psi_s^i\).

## 5. Lemma-10 recurrences

Define \(C_q^{(\ell)}\) to majorize the norm/extensiveness sum of total
order-\(q\) interaction-picture terms containing exactly \(\ell\) nested
adjoints. With no adjoint,

\[
C_q^{(0)}=G_q.
\tag{16}
\]

Suppose the new logarithm coefficient has order \(s\). Before it is applied,
the accumulated order-\(q-s\) operator is \((q-s)k\)-local. The standard
locality/extensiveness commutator estimate

\[
g([X,Y])\leq2K_Xg(Y)
\tag{17}
\]

therefore contributes \(2k(q-s)G_s\). Summing the possible order splits gives

\[
C_q^{(\ell+1)}
=\sum_s2k(q-s)G_sC_{q-s}^{(\ell)}.
\tag{18}
\]

The exponential of the adjoint action supplies \(1/\ell!\), so

\[
D_q=\sum_{\ell\geq0}\frac{C_q^{(\ell)}}{\ell!}
\tag{19}
\]

majorizes the order-\(q\) interaction-picture generator coefficient in Lemma
10, including its subsystem difference.

The outer time-ordered/Dyson products are positive-majorized by the ordinary
scalar exponential

\[
B(z)=\exp(D(z)),
\qquad D(z)=\sum_{q\geq1}D_qz^q.
\tag{20}
\]

Indeed, differentiating \(B(z)\) gives \(B'=D'B\), hence

\[
B_0=1,
\qquad
B_q=\frac1q\sum_{s=1}^{q}sD_sB_{q-s}.
\tag{21}
\]

Low-order tests generate the nested compositions directly and compare every
\(C^{(\ell)}\), \(D_q\), and \(B_q\) coefficient with (18)--(21).

## 6. Certified infinite-tail lemma

The production tail bound is a mathematical majorant result, not a numerical
cutoff heuristic.

**Scalar-flow lemma.** Let

\[
G(z)=\sum_{s=1}^{p_0}G_sz^s,
\qquad
\mathcal Lf=2kzG(z)f'(z).
\tag{22}
\]

Then recurrence (18) is the coefficient recurrence of

\[
D(z)=e^{\mathcal L}G(z).
\tag{23}
\]

The characteristic flow satisfies

\[
\frac{dy}{du}=2kyG(y),
\qquad y(0)=z.
\tag{24}
\]

Because \(G\) has nonnegative coefficients, it is increasing on the positive
axis. If \(0<\rho<R\) and

\[
\int_\rho^R\frac{dx}{2kxG(x)}\geq1,
\tag{25}
\]

separation of variables proves that the flow starting at every
\(0\leq z\leq\rho\) exists through \(u=1\) and satisfies
\(y(1;z)\leq R\). Equations (20), (23), and monotonicity give

\[
D(\rho)\leq G(R),
\qquad
B(\rho)\leq e^{G(R)}.
\tag{26}
\]

For Lemma 9, (10) is analytic and finite whenever
\(\rho<(2kg_\alpha)^{-1}\).

**Positive Cauchy tail.** If \(F(z)=\sum_qF_qz^q\) has nonnegative
coefficients and is finite at \(\rho>0\), then
\(F_q\rho^q\leq F(\rho)\) coefficient by coefficient. Thus, for
\(0\leq x<\rho\),

\[
\sum_{q>Q}F_qx^q
\leq F(\rho)\sum_{q>Q}(x/\rho)^q
=F(\rho)\frac{(x/\rho)^{Q+1}}{1-x/\rho}.
\tag{27}
\]

Production generates \(A_q\) and \(B_q\) explicitly through

\[
Q=2p_0+32
\tag{28}
\]

and applies (27) afterward, using the exact value (10) for \(A(\rho)\) and
the certified upper bound (26) for \(B(\rho)\).

For numerical certification, set \(u=\log x\). The integral in (25) becomes

\[
\int_{\log\rho}^{\log R}
\frac{du}{2kG(e^u)}.
\tag{29}
\]

Its integrand is decreasing, so right-endpoint rectangles give a rigorous
lower bound. Interval contributions and their sum are rounded downward;
polynomial values, exponentials, coefficients, and final tails are rounded
upward with `nextafter`. Radius candidates are searched from small to large,
and the largest successful certificate is retained. If the coarse certified
grid loses too much area, the smallest radius is retried on a four-times finer
grid. Failure returns no refined certificate; it never returns an uncertified
finite number.

Tests compare the certified tail with direct coefficient generation through
at least \(4Q\), including points at 92% of the returned Cauchy radius.

## 7. Fallback and segment selection

If a refined tail cannot be certified for a candidate \((r,p_0)\), the code
uses

\[
R_{p_0}(x)\leq3Ne^{-p_0}
\tag{30}
\]

only when the printed first condition

\[
|t|/r\leq(8e^3c_pp_0kg)^{-1}
\tag{31}
\]

holds. The candidate is marked `legacy-theorem3-tail`. If neither proof is
available, the candidate is rejected.

For every segment candidate, production enumerates \(p_0\geq3\), computes
(1)--(4), and chooses the certified \(p_0\) with the smallest repeated error.
The search stops only when monotonic data prove that higher orders cannot
pass or improve the result: the second time condition has failed, the
commutator-only repeated error exceeds the target or the best bound already
found, or both refined and legacy tail certificates are unavailable. Segment
counts are bracketed by doubling and resolved by integer binary search; the
selected \(r-1\) is verified to fail.

The two implementations are therefore:

| Identifier | Selection path |
| --- | --- |
| `mizuta2026-theorem3-legacy-ideal-rigorous` | \(\eta_{\rm aux}\to p_0\to\) printed first condition; optimize allocation |
| `mizuta2026-commutator-ideal-rigorous` | \(p_0\to R_{p_0}(t/r)\to\delta_{\rm BCH}\); optimize \(p_0\) directly |

Supplying `auxiliary_allocation_fraction` to the refined identifier is a
validation error. Its `auxiliary_error` and
`auxiliary_allocation_fraction` diagnostics are `None`.

## 8. Diagnostics and benchmark results

The refined row reports the selected \(r,p_0\), `mu_upper`, separate Lemma-9
and Lemma-10 remainders, total branchwise BCH contribution, local commutator
and total errors, repeated error, both time-limit diagnostics, schedule
weights, active constraint, exact commutator cutoff/locality fallback, and
tail-fallback status. The legacy row retains its auxiliary fields.

The executed [comparison notebook](../notebooks/mpf_bound_comparison.ipynb)
uses \(J=3\), \(t=N\), and \(\epsilon=10^{-3}\). Representative rows are:

| Model | \(N\) | Low 2019 | Legacy Mizuta | Refined Mizuta | Refined \(p_0\) | Refined active/dominant | Locality fallback |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| open TFIM, \(J=1,h=3\) | 4 | 123 | 321,369 | 1,156 | 5 | error / commutator | no |
| open TFIM, \(J=1,h=3\) | 50 | 45,617 | 4,820,529 | 51,824 | 5 | error / commutator | no |
| open TFIM, \(J=1,h=3\) | 100 | 228,890 | 10,283,795 | 147,004 | 5 | error / commutator | no |
| open XXX, \(J=1,h_z=0\) | 4 | 69 | 385,643 | 978 | 8 | error / commutator | no |
| open XXX, \(J=1,h_z=0\) | 50 | 32,096 | 5,977,456 | 67,460 | 4 | error / BCH | no |
| open XXX, \(J=1,h_z=0\) | 100 | 162,427 | 12,340,554 | 374,991 | 4 | error / commutator | yes |

The notebook also contains all 66 rows of the two-model, 11-point
\(10^{-1}\)--\(10^{-6}\) target-error sweep at \(N=t=50\), with the same
constraint and fallback columns. On every representative refined row above,
the printed first condition fails; the direct remainder certificate is what
makes the candidate rigorous.
