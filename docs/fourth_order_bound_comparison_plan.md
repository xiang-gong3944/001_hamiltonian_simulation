# Fourth-order commutator-bound comparison plan

## Scope and mathematical decision

The requested comparison is mathematically well defined, with one qualification:
the general Childs--Su--Tran--Wiebe--Zhu theorem gives an asymptotic
big-$O$ statement and explicitly says that it does not evaluate the constant
prefactor. A numerical curve called the "general Childs bound" must therefore
name the concrete relaxation used to instantiate that big-$O$ result.

This implementation will use the explicit anti-Hermitian proof relaxation in
the enhanced arXiv version of Childs et al., Eqs. (189) and (191),

$$
C_{p+1}^{\mathrm{Childs,general}}
 = \frac{2\Upsilon^{p+1}}{(p+1)!}\widetilde\alpha_{\mathrm{comm}},
\qquad
\widetilde\alpha_{\mathrm{comm}}
 = \sum_{\gamma_1,\ldots,\gamma_{p+1}=1}^{\Gamma}
 \left\lVert
 [H_{\gamma_{p+1}},\ldots,[H_{\gamma_2},H_{\gamma_1}]\ldots]
 \right\rVert.
$$

Here $\Upsilon$ is the number of full stages in the unmerged representation.
For the standard fourth-order recursion built from five Strang formulas,
$\Upsilon=10$. This is a rigorous concrete relaxation for the repository's
Hermitian Hamiltonians; it is not an otherwise unspecified constant assigned
to Theorem 6 of the published paper (Theorem 11 in the enhanced arXiv version).
The implementation and output labels will preserve that distinction.

The exact small-prefactor comparison is between Childs et al. Appendix M and
Schubert--Mendl Theorem 1. It is not expected to show a constant-factor
improvement for the same two-term fourth-order formula: Schubert and Mendl
state that their theorem reproduces Childs et al. Eq. (M13), and direct
coefficient generation gives the eight coefficients

$$
\begin{array}{c|c}
(A,A,A,B,A)&0.004701334310169883\\
(A,A,B,B,A)&0.005703818762629350\\
(A,B,A,B,A)&0.004638910081589791\\
(A,B,B,B,A)&0.007372056645952210\\
(B,A,A,B,A)&0.009689668290122170\\
(B,A,B,B,A)&0.009726162358456834\\
(B,B,A,B,A)&0.017328153057109530\\
(B,B,B,B,A)&0.028373434405425900.
\end{array}
$$

They round to $0.0047,0.0057,0.0046,0.0074,0.0097,0.0097,
0.0173,0.0284$ in Eq. (M13). The tuple lists the commutator from
outermost operator to innermost base operator.

For three summands, Childs et al. also provide Proposition M.2 and Table II.
Schubert and Mendl state that programmatic evaluation of their Eq. (9) with
$s=10$ reproduces that table. The merged three-term formula has $K=21$, so the
conventional Schubert--Mendl centered choice is instead $s=11$. Any difference
between those two rows is caused by the center/factorization choice, not by a
different product formula or a contradiction between the theorems. For four
or more summands Appendix M supplies no specialized coefficient table; only
the general Childs relaxation and the general Schubert--Mendl theorem are
comparable.

Primary sources:

- Childs et al., [Theory of Trotter Error with Commutator Scaling](https://doi.org/10.1103/PhysRevX.11.011020),
  published Theorem 6, Appendix M Eqs. (M1)--(M14), Proposition M.1,
  Proposition M.2, and Table II. The enhanced
  [arXiv version](https://arxiv.org/abs/1912.08854) numbers the corresponding
  general and Appendix-M results as Theorem 11, Eqs. (189), (191), and
  Appendix J.
- Schubert and Mendl,
  [Trotter error with commutator scaling for the Fermi-Hubbard model](https://doi.org/10.1103/PhysRevB.108.195105),
  Eqs. (6)--(10), Theorem 1, and the discussion following Eq. (12).

## Repository findings

1. `src/hamiltonian_resources/trotter.py` constructs the standard recursion

   $$
   S_4(t)=S_2(z_1t)^2S_2(z_0t)S_2(z_1t)^2,
   \quad z_1=(4-4^{1/3})^{-1},\quad z_0=1-4z_1.
   $$

   `_suzuki_group_factors` agrees factor by factor with Qiskit's
   `SuzukiTrotter.expand`. With adjacent equal summands merged, the number of
   exponentials is $K=10\Gamma-9$ (11 for two summands, 21 for three).

2. The current fourth- and sixth-order implementation is Schubert--Mendl
   Theorem 1 with the fixed centered choice $s=\lceil K/2\rceil$. It combines
   the summands in $B_j$ before taking the Pauli coefficient 1-norm, so it may
   retain cancellations that are lost when $B_j$ is expanded by a triangle
   inequality.

3. The only currently implemented Childs bounds are the specialized first-
   and second-order results (published Propositions 9 and 10). There is no
   current fourth-order Childs evaluator. Thus the current order-four result
   must not be relabeled as a Childs bound.

4. The resolved Hamiltonian groups are shared by synthesis and analysis.
   `auto` gives two commuting groups for the transverse-field Ising chain and
   up to three for the Heisenberg chain. For at most three groups the repository
   may reorder groups using its second-order proxy; the comparison must record
   and reuse that resolved ordering.

## Equations and conventions to implement

Construct one immutable fourth-order problem object containing:

- the resolved ordered Hamiltonian summands $H_1,\ldots,H_\Gamma$;
- the raw recursive factor list and the identically merged list
  $A_k=a_kH_{\gamma_k}$;
- $z_1,z_0$, $p=4$, $K$, $\Upsilon=10$, partition, and group sizes.

Every evaluator receives that same object. In the Schubert--Mendl notation,

$$
S_4(t)=e^{-itA_K}\cdots e^{-itA_1},\qquad
B_j=\sum_{\ell=1}^{j-1}A_\ell,
$$

and Theorem 1 gives

$$
\begin{aligned}
C_5^{\mathrm{SM}}(s)=\frac1{5!}\Bigg(&
\sum_{j=2}^{s}\sum_{\substack{q_j+\cdots+q_s=4\\q_j\ne0}}
{4\choose q_j,\ldots,q_s}
\left\lVert\operatorname{ad}_{A_s}^{q_s}\cdots
\operatorname{ad}_{A_j}^{q_j}B_j\right\rVert\\
&+\sum_{j=s+1}^{K}\sum_{\substack{q_{s+1}+\cdots+q_j=4\\q_j\ne0}}
{4\choose q_{s+1},\ldots,q_j}
\left\lVert\operatorname{ad}_{A_{s+1}}^{q_{s+1}}\cdots
\operatorname{ad}_{A_j}^{q_j}B_j\right\rVert\Bigg).
\end{aligned}
$$

The comparison version will expand $B_j$ and apply the triangle inequality so
that its coefficients can be compared term by term with Appendix M. A
non-expanded variant will also be evaluated to isolate the effect of this
additional relaxation. Norms default to the rigorous Pauli coefficient 1-norm;
small-system validation may request the exact spectral norm.

Childs Appendix M will be represented independently by its canonical
five-index commutator coefficients: Eq. (M13) for two summands and Table II for
three summands. Exact coefficients will be generated from the Appendix-M
factorization ($s=6$ and $s=10$, respectively), not copied from the paper's
four-decimal display. Other decomposition sizes return an explicit unsupported
status.

For $r$ equal unitary segments, all concrete local bounds use unitary
telescoping:

$$
\epsilon_{\mathrm{bound}}
 \le r C_5 |t/r|^5=\frac{C_5|t|^5}{r^4},
\qquad
r_{\min}=\max\left(1,
\left\lceil(C_5|t|^5/\epsilon)^{1/4}\right\rceil\right).
$$

## APIs and data schema

Add a focused module for:

- `build_fourth_order_bound_problem`;
- `childs_general_commutator_bound`;
- `childs_fourth_order_small_prefactor_bound`;
- `schubert_mendl_small_prefactor_bound`;
- immutable result and contribution diagnostics;
- accumulated-error and required-segment helpers.

Each result will record the total $C_5$, time power, per-commutator prefactors,
indices and evaluated norms, ordered factors, coefficients, center $s$, merge
state, norm method, and every triangle/representation relaxation.

Add a separate comparison benchmark module and CSV schema. Rows will include
the requested model/decomposition/formula/bound provenance, all center choices,
$C_5$, time, requested and required segment counts, accumulated error, a named
ratio denominator, status, and diagnostics. The system-size sweep will use a
two-group transverse-field Ising case and a three-group Heisenberg case; the
target-error sweep will use a fixed small size. All valid Schubert--Mendl
centers will be retained, with roles identifying the conventional centered and
minimum rows.

Plotting remains separate from the existing T/MPF/QSVT plots and will produce:

1. system qubits versus $C_5$;
2. system qubits versus required segment count;
3. target error versus required segment count;
4. a ratio figure/table with an explicit denominator.

## Complexity and safeguards

The commutator basis contains $\Gamma^5$ fifth-degree words (four adjoint
actions and one base). The existing dynamic program collapses weak
compositions into repeated group words, avoiding enumeration exponential in
$K$. Sparse Pauli multiplication still grows with both system size and
commutator support. The comparison will retain the existing practical cap,
avoid dense matrices for scaling runs, and confine spectral norms and exact
Trotter errors to small validation cases.

## Validation plan

Tests will verify:

1. object identity/equality of the ordered sequence received by all evaluators;
2. exact agreement with Qiskit's order-four factors, including normalization,
   order, and merging;
3. $|\delta|^5$ local and $|t|^5/r^4$ accumulated scaling;
4. termwise and total equality of Schubert--Mendl $s=6$ with Childs Eq. (M13);
5. equality of Schubert--Mendl $s=10$ with Childs Table II for three terms and
   an explicit comparison against the centered $s=11$ result;
6. all-center reporting and correct minimizing-center metadata;
7. fifth-degree homogeneity and ratio invariance under common Hamiltonian
   rescaling;
8. explicit unsupported status for Appendix-M decompositions with four or
   more terms;
9. analytical bounds above the exact small-system operator-norm error;
10. coefficient/norm differences attributable separately to center choice,
    $B_j$ triangle expansion, ordering, merging, and norm evaluation.

## Files to modify

- add `src/hamiltonian_resources/fourth_order_bounds.py`;
- add `src/hamiltonian_resources/fourth_order_comparison.py`;
- minimally generalize the internal center argument in `trotter.py` and reuse
  the new centered evaluator without changing the circuit formula;
- export public APIs from `src/hamiltonian_resources/__init__.py`;
- add focused tests under `tests/`;
- extend `docs/suzuki_error_bounds.md`, `docs/resource_scaling_benchmarks.md`,
  and `README.md` with equations, commands, limitations, and interpretation;
- add a dedicated comparison configuration and generated example outputs only
  after the implementation and validation pass.
