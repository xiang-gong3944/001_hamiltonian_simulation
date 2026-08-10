# Suzuki product-formula error estimates

This package uses operator-norm error estimates that correspond to the same
term partition, order, and recursive Suzuki sequence used to build the Qiskit
circuit. This document records the implemented formulas, the classical work
required to evaluate them, and the cases that retain a heuristic fallback.

## Error bounds

For a Hamiltonian written as (H=\sum_{\gamma=1}^{G}H_\gamma), the existing
first- and second-order bounds are the small-prefactor results of Childs, Su,
Tran, Wiebe, and Zhu:

\[
\lVert S_1(\delta)-e^{-i\delta H}\rVert\le W_1|\delta|^2,
\qquad
W_1=\frac12\sum_g\lVert[T_g,H_g]\rVert,
\]

\[
\lVert S_2(\delta)-e^{-i\delta H}\rVert\le W_2|\delta|^3,
\]

\[
W_2=\sum_g\left(
\frac1{12}\lVert[T_g,[T_g,H_g]]\rVert+
\frac1{24}\lVert[H_g,[H_g,T_g]]\rVert
\right),
\]

where (T_g=\sum_{j>g}H_j). The implementation forms each commutator as a
`SparsePauliOp` and upper-bounds its spectral norm by the sum of the absolute
Pauli coefficients. The coefficient 1-norm is conservative but rigorous.

For orders (p=4) and (p=6), the implementation evaluates Schubert and
Mendl, Theorem 1. Write one merged Suzuki step as

\[
S_p(t)=e^{-itA_K}\cdots e^{-itA_1},\qquad
B_j=\sum_{\ell<j}A_\ell,
\]

and choose (s=\lceil K/2\rceil). The bound is

\[
\begin{aligned}
\lVert S_p(t)-e^{-itH}\rVert\le\frac{|t|^{p+1}}{(p+1)!}
\Bigg(&\sum_{j=2}^{s}
\sum_{\substack{q_j+\cdots+q_s=p\\q_j\ne0}}
{p\choose q_j,\ldots,q_s}
\left\lVert\operatorname{ad}_{A_s}^{q_s}\cdots
\operatorname{ad}_{A_j}^{q_j}B_j\right\rVert\\
&+\sum_{j=s+1}^{K}
\sum_{\substack{q_{s+1}+\cdots+q_j=p\\q_j\ne0}}
{p\choose q_{s+1},\ldots,q_j}
\left\lVert\operatorname{ad}_{A_{s+1}}^{q_{s+1}}\cdots
\operatorname{ad}_{A_j}^{q_j}B_j\right\rVert\Bigg).
\end{aligned}
\]

The resulting coefficient is stored as `SuzukiErrorEstimate.prefactor`. For
`reps=r`, unitary telescoping gives the implemented global bound

\[
\lVert S_p(t/r)^r-e^{-itH}\rVert\le W_p|t|^{p+1}/r^p.
\]

`choose_parameters` inverts this expression against the algorithmic portion
of the requested error budget.

## Qiskit formula correspondence

Qiskit's `SuzukiTrotter` uses

\[
S_p(t)=S_{p-2}(z_pt)^2S_{p-2}((1-4z_p)t)S_{p-2}(z_pt)^2,
\qquad z_p=(4-4^{1/(p-1)})^{-1}.
\]

The internal factor generator mirrors this recursion. Adjacent occurrences of
the same group are merged only while evaluating the theorem; the synthesized
circuit is left in Qiskit's raw recursive form. Regression tests compare every
group index and coefficient with `SuzukiTrotter.expand` for orders 2, 4, and 6.

The resolved groups are passed to `PauliEvolutionGate` as a list of commuting
`SparsePauliOp` objects with `preserve_order=True`. Since all Pauli terms inside
a group commute, Qiskit's sequential Pauli rotations exactly implement the
group exponential assumed by the theorem.

## Partition and ordering policy

`TrotterPartition` accepts three values:

- `auto`: individual Pauli terms for orders 1 and 2; full commuting groups for
  order 4 and above.
- `individual`: every Pauli term is a separate summand, reproducing the legacy
  higher-order formula.
- `commuting`: greedily color the Pauli anticommutation graph while preserving
  the Hamiltonian's input order inside each group.

For at most three commuting groups, all group permutations are compared using
the inexpensive second-order prefactor. Ties prefer the largest final group,
which reduces raw Suzuki rotation occurrences, followed by the stable original
group order. Circuit synthesis, analytical rotation counting, and the reported
bound all reuse this resolved order.

For the supported models, TFIM gives an interaction group and a field group.
The Heisenberg chain gives even/odd-style commuting interaction layers and,
when needed, a third field layer. A noncommuting local block such as a complete
Heisenberg bond plus an on-site field is deliberately not used: Qiskit's
default atomic evolution would not exponentiate that block exactly.

## Classical complexity and fallback

For (G) groups, merging adjacent factors gives

\[
K_2=2G-1,\qquad K_p=5K_{p-2}-4,
\]

so (K_4=10G-9) and (K_6=50G-49). Direct enumeration of the theorem's weak
compositions is avoided. Dynamic programming combines compositions that give
the same repeated group word, and the implementation precomputes at most
(G^{p+1}) nested group commutators. Pauli coefficient vectors are shared so
each (B_j)-dependent 1-norm can be evaluated as a small vectorized linear
combination. No dense Hamiltonian matrix is created.

The rigorous higher-order evaluator is used when (G^{p+1}\le4096). Thus
order 4 supports up to five groups and order 6 supports up to three. This
covers the repository's TFIM and Heisenberg constructors. The following cases
return the historical

\[
(\alpha |t|)^{p+1}/r^p,\qquad \alpha=\sum_j|h_j|,
\]

proxy instead:

- order 4 or 6 above the work cap;
- order 8 and higher;
- an explicit individual-term partition with too many terms.

Fallback results always have `method="alpha-proxy"` and `rigorous=False`.
Benchmark tables expose the same distinction through
`trotter_error_method` and `trotter_error_rigorous`.

The evaluator retains only the active nonzero commutator frontier. Large
frontier and theorem-reduction stages can be split across worker processes;
small stages stay serial to avoid process and serialization overhead. Python
calls default to `workers=1`, while benchmark CLI runs select a capped worker
count automatically.

## Validation

The test suite synthesizes the decomposed Qiskit circuit, forms its small dense
unitary, and compares it with `scipy.linalg.expm(-1j * t * H)` in spectral norm.
It covers TFIM, Heisenberg, fixed random Pauli sums, commuting Hamiltonians,
orders 4 and 6, and multiple Trotter repetitions. Dense matrices are confined
to these small-system validation tests and `compare_with_exact`.

## References

- A. M. Childs et al., [Theory of Trotter Error with Commutator Scaling](https://arxiv.org/abs/1912.08854), Propositions 9 and 10.
- A. Schubert and C. B. Mendl, [Trotter error with commutator scaling for the Fermi-Hubbard model](https://arxiv.org/abs/2306.10603), Theorem 1 and Eqs. 6--10.
- [Qiskit `SuzukiTrotter` API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.synthesis.SuzukiTrotter).
