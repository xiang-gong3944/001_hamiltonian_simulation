# Pre-study of empirical error models for Trotter and multiproduct formulas

## 1. Purpose

This note summarizes the numerical investigations performed before implementing an empirical error model for Hamiltonian-simulation resource estimation.

The main questions are:

1. What is the empirical dependence of the algorithmic error on the segment count \(r\)?
2. What is the physical-time dependence?
3. How does the error coefficient scale with system size?
4. How strongly do lattice geometry and boundary conditions affect the coefficient?
5. Can Trotter and multiproduct formulas (MPFs) be described by a common empirical structure?
6. Which quantities should be calibrated numerically and which should remain theory-driven?

The calculations considered:

- 1D transverse-field Ising model,
- 1D Heisenberg chain,
- 1D transverse-field XXZ chain,
- SSH tight-binding model,
- 1D Hubbard model,
- 2D square-lattice transverse-field Ising model,
- 2D square-lattice Heisenberg model.

Both dense exact-matrix calculations and sparse/state-vector methods were used.

---

# 2. Unified empirical error ansatz

The numerical results strongly suggest the following common form.

For a product formula of formal order \(q\),

\[
\boxed{
\epsilon_q(N,T,r)
\simeq
B_q(\mathcal G_N,H_{\mathrm{loc}})
\frac{T^{q+1}}{r^q}.
}
\]

Here

- \(N\) is the system size,
- \(T\) is the physical simulation time,
- \(r\) is the segment count,
- \(\mathcal G_N\) is the finite interaction graph,
- \(H_{\mathrm{loc}}\) contains local couplings and the chosen Hamiltonian decomposition,
- \(B_q\) is a finite-size geometry-dependent coefficient.

For a \(p\)-th order Suzuki--Trotter formula,

\[
q=p,
\]

so

\[
\boxed{
\epsilon_{\mathrm{Trotter}}^{(p)}
\simeq
B_p^{\mathrm{Trotter}}(\mathcal G_N,H_{\mathrm{loc}})
\frac{T^{p+1}}{r^p}.
}
\]

For an MPF with branch count \(m\) and formal order \(2m\),

\[
q=2m,
\]

so

\[
\boxed{
\epsilon_{\mathrm{MPF}}^{(m)}
\simeq
B_{2m}^{\mathrm{MPF}}(\mathcal G_N,H_{\mathrm{loc}})
\frac{T^{2m+1}}{r^{2m}}.
}
\]

Thus Trotter and MPF appear to share the same leading dependence on \(T\) and \(r\) when compared at the same formal order. The main differences are expected to enter through

\[
B_q^{\mathrm{Trotter}}
\quad\text{vs}\quad
B_q^{\mathrm{MPF}}
\]

and through the circuit cost per segment.

---

# 3. Error metric and numerical methodology

The primary empirical error measure was the spectral operator norm

\[
\boxed{
\epsilon
=
\left\|
U_{\mathrm{approx}}(T)-e^{-iHT}
\right\|_2.
}
\]

This is preferable to a state-dependent error when the goal is to calibrate a resource-estimation error model.

For small systems:

- the Hamiltonian was represented as a dense matrix,
- exact evolution was obtained by diagonalization,
- the approximate evolution operator was constructed explicitly,
- the spectral norm was computed directly.

For larger systems:

- the Hamiltonian was stored as a sparse matrix,
- exact evolution of vectors used `scipy.sparse.linalg.expm_multiply`,
- Suzuki and MPF steps were applied directly to state vectors,
- the operator norm was estimated using power iteration on
  \[
  D^\dagger D,
  \qquad
  D=U_{\mathrm{approx}}-U.
  \]

For 2D spin models, this allowed operator-norm estimates up to approximately 15 qubits in practical runs.

Random-state errors

\[
\|D|\psi_{\mathrm{rand}}\rangle\|
\]

were also used for some 16-qubit diagnostics, but **these are not interchangeable with the operator norm** and should not be used for final calibration of the extensive coefficient.

---

# 4. Segment-count scaling

## 4.1 Trotter

For 1D TFIM and Heisenberg chains, second-, fourth-, and sixth-order Suzuki formulas were tested.

At sufficiently large \(r\),

\[
\boxed{
\epsilon_p\propto r^{-p}
}
\]

was observed very clearly.

Representative large-\(r\) running exponents were:

### 1D TFIM

| order \(p\) | observed \(\beta_{\mathrm{eff}}\) |
|---:|---:|
| 2 | \(2.00\) |
| 4 | \(3.95\)--\(4.00\) |
| 6 | \(6.01\)--\(6.03\) |

### 1D Heisenberg

| order \(p\) | observed \(\beta_{\mathrm{eff}}\) |
|---:|---:|
| 2 | \(2.00\) |
| 4 | \(3.97\)--\(4.00\) |
| 6 | \(6.00\)--\(6.03\) in the asymptotic window |

The same \(r^{-p}\) behavior was also found for the fermionic models and the 2D spin models.

---

## 4.2 MPF

For MPFs, the corresponding result is

\[
\boxed{
\epsilon_m\propto r^{-2m}.
}
\]

Representative 1D running exponents were:

### TFIM

| \(m\) | expected \(2m\) | observed \(\beta_{\mathrm{eff}}\) |
|---:|---:|---:|
| 2 | 4 | 3.99 |
| 3 | 6 | 5.99 |
| 4 | 8 | 7.95 |
| 6 | 12 | 11.75 |
| 8 | 16 | 15.22 |

### Heisenberg

| \(m\) | expected \(2m\) | observed \(\beta_{\mathrm{eff}}\) |
|---:|---:|---:|
| 2 | 4 | 3.99 |
| 3 | 6 | 5.99 |
| 4 | 8 | 7.98 |
| 6 | 12 | 11.69 |
| 8 | 16 | 15.35 |

For high \(m\), the true error reaches the double-precision floor very rapidly, so \(m\gtrsim 10\) requires higher precision or analytic extraction of the leading coefficient.

---

# 5. Time scaling

The common ansatz predicts

\[
\boxed{
\epsilon_q\propto T^{q+1}.
}
\]

This is one of the most important numerical observations.

---

## 5.1 MPF: direct time sweep

For 2D TFIM and 2D Heisenberg, direct time sweeps were performed on \(3\times3\) and \(3\times4\) lattices.

For each fixed geometry and fixed \(r\), the error was fitted as

\[
\epsilon\propto T^{\gamma_T}.
\]

### Results

| model | geometry | \(m\) | fitted \(\gamma_T\) | expected \(2m+1\) |
|---|---|---:|---:|---:|
| TFIM | \(3\times3\) | 2 | 5.005 | 5 |
| TFIM | \(3\times4\) | 2 | 4.991 | 5 |
| TFIM | \(3\times3\) | 3 | 7.031 | 7 |
| TFIM | \(3\times4\) | 3 | 7.005 | 7 |
| Heisenberg | \(3\times3\) | 2 | 4.936 | 5 |
| Heisenberg | \(3\times4\) | 2 | 5.012 | 5 |
| Heisenberg | \(3\times3\) | 3 | 6.985 | 7 |
| Heisenberg | \(3\times4\) | 3 | 6.966 | 7 |

All fits had log-space \(R^2>0.9997\).

Thus the MPF data give strong direct numerical evidence for

\[
\boxed{
\epsilon_m
\propto
T^{2m+1}r^{-2m}.
}
\]

---

## 5.2 Trotter: two-time-policy comparison

For 2D TFIM and 2D Heisenberg, the same finite geometries were evaluated using

\[
T=N
\]

and

\[
T=\sqrt N.
\]

The ratio between the two error coefficients gives an effective time exponent.

Representative median values were:

| model | \(p\) | measured \(\gamma_T\) | expected \(p+1\) |
|---|---:|---:|---:|
| 2D TFIM | 2 | 2.980 | 3 |
| 2D TFIM | 4 | 4.994 | 5 |
| 2D TFIM | 6 | 6.60 | 7 |
| 2D Heisenberg | 2 | 2.992 | 3 |
| 2D Heisenberg | 4 | 5.001 | 5 |
| 2D Heisenberg | 6 | 6.77 | 7 |

The second- and fourth-order results are especially clean.

The sixth-order data retain stronger finite-\(r\) and finite-size effects, but remain consistent with \(T^7\).

A dedicated multi-point \(T\)-sweep for Trotter would still be useful as a final validation, analogous to the MPF calculation.

---

# 6. System-size scaling in one dimension

If

\[
B_q(N)\sim A_q N
\]

in the bulk, then choosing

\[
T=N
\]

gives

\[
\boxed{
\epsilon_q
\sim
A_q
N^{q+2}r^{-q}.
}
\]

This was tested extensively.

---

## 6.1 Trotter: 1D spin chains

### TFIM

At \(T=N\),

| order \(p\) | fitted \(N\)-exponent | expected \(p+2\) |
|---:|---:|---:|
| 2 | \(4.14\)--\(4.18\) | 4 |
| 4 | \(6.13\)--\(6.18\) | 6 |
| 6 | \(\sim 8.0\) | 8 |

The largest-size local exponents were approximately

\[
4.14,\qquad
6.12,\qquad
8.26.
\]

### Heisenberg chain

Finite-size corrections are stronger, but the exponents drift toward the same values:

| order \(p\) | large-size fitted behavior |
|---:|---:|
| 2 | \(N^{4.2\text{--}4.3}\) |
| 4 | approximately \(N^{6.3}\), with local exponent near 6 at the largest interval |
| 6 | \(N^{8.2\text{--}8.4}\) |

Thus the data support

\[
\boxed{
\epsilon_{\mathrm{Trotter}}^{(p)}
\sim
A_p N^{p+2}r^{-p},
\qquad T=N.
}
\]

---

## 6.2 MPF: 1D spin chains

For MPF, \(q=2m\), so the same bulk ansatz predicts

\[
\boxed{
\epsilon_m
\sim
A_mN^{2m+2}r^{-2m},
\qquad T=N.
}
\]

### TFIM

Representative large-size exponents:

| \(m\) | fitted \(\alpha\) | expected \(2m+2\) |
|---:|---:|---:|
| 2 | 6.14 | 6 |
| 3 | 8.16 | 8 |
| 4 | 10.05 | 10 |
| 6 | 14.15 | 14 |
| 7 | 16.20 | 16 |

### Heisenberg, \(m=3\)

Small sizes initially gave a very large apparent exponent. Extending the calculation to \(N=12\) showed a clear downward drift:

| fit range | fitted \(\alpha\) |
|---|---:|
| \(N=3,\dots,12\) | 9.81 |
| \(N=4,\dots,12\) | 8.72 |
| \(N=5,\dots,12\) | 8.55 |
| \(N=6,\dots,12\) | 8.40 |
| \(N=7,\dots,12\) | 8.33 |
| \(N=8,\dots,12\) | 8.22 |

This strongly indicates a finite-size correction rather than a different asymptotic exponent.

### Transverse-field XXZ, \(m=3\)

The same pattern was observed:

| fit range | fitted \(\alpha\) |
|---|---:|
| \(N=3,\dots,12\) | 9.08 |
| \(N=5,\dots,12\) | 8.48 |
| \(N=7,\dots,12\) | 8.26 |
| \(N=9,\dots,12\) | 8.06 |

Again,

\[
\boxed{
\epsilon_{m=3}\sim A_3N^8r^{-6}
}
\]

is strongly supported.

---

# 7. Fermionic systems

The same structure survives Jordan--Wigner mapping and non-spin physical models.

---

## 7.1 Trotter

### SSH tight-binding model

For \(T=N\),

| order \(p\) | global fitted exponent | expected |
|---:|---:|---:|
| 2 | 4.20 | 4 |
| 4 | 6.90 | 6 |
| 6 | 8.99 | 8 |

The fourth-order local exponent decreases with size,

\[
7.43\to6.57\to6.35,
\]

consistent with convergence toward 6.

The sixth-order fit is more sensitive to roundoff and small sizes.

### 1D Hubbard model

Using \(N=L\) physical lattice sites,

| order \(p\) | fitted exponent | expected |
|---:|---:|---:|
| 2 | 4.04 | 4 |
| 4 | 6.15 | 6 |
| 6 | 8.24 | 8 |

This is a particularly clean confirmation of

\[
\boxed{
\epsilon_p
\sim
A_pN^{p+2}r^{-p},
\qquad T=N.
}
\]

---

## 7.2 MPF

### SSH, \(m=3\)

The large-\(r\) exponent is approximately 6.

The local system-size exponents decrease as

\[
11.54\to9.03\to8.65,
\]

consistent with eventual approach to the expected exponent 8.

### Hubbard, \(m=3\)

The local exponents decrease as

\[
9.28\to8.48\to8.33,
\]

again supporting

\[
\boxed{
\epsilon_{m=3}
\sim
A_3N^8r^{-6}.
}
\]

The fermionic calculations therefore suggest that the empirical scaling is not specific to spin Hamiltonians.

---

# 8. Geometry dependence in two dimensions

The finite-size coefficient should not be treated as a universal constant.

The leading error is generated by nested commutators of local Hamiltonian terms. Their number and magnitude depend on the local overlap graph.

A useful finite-size notation is therefore

\[
\boxed{
B_q
=
B_q(\mathcal G_N,H_{\mathrm{loc}}).
}
\]

Important dependencies include:

- lattice coordination,
- spatial geometry,
- aspect ratio,
- boundary conditions,
- local coupling constants,
- anisotropy,
- commuting-group partition,
- Suzuki ordering,
- MPF schedule.

---

## 8.1 MPF geometry scan

For \(m=2,3\), 2D TFIM and Heisenberg were tested on

\[
2\times2,\;
2\times3,\;
2\times4,\;
2\times5,\;
3\times3,\;
3\times4,\;
3\times5.
\]

Using

\[
T=\sqrt N,
\]

the normalized coefficient

\[
\boxed{
\frac{B_m}{N}
=
\frac{\epsilon r^{2m}}
{N T^{2m+1}}
}
\]

was examined.

### TFIM

#### \(m=2\)

| geometry | \(B_2/N\) |
|---|---:|
| \(2\times3\) | 4.25 |
| \(2\times4\) | 4.02 |
| \(2\times5\) | 3.90 |
| \(3\times3\) | 4.03 |
| \(3\times4\) | 3.63 |
| \(3\times5\) | 3.48 |

#### \(m=3\)

| geometry | \(B_3/N\) |
|---|---:|
| \(2\times3\) | 1.42 |
| \(2\times4\) | 1.38 |
| \(2\times5\) | 1.38 |
| \(3\times3\) | 1.59 |
| \(3\times4\) | 1.52 |
| \(3\times5\) | 1.38 |

For TFIM, \(B_m/N\) becomes nearly constant rather quickly, especially for \(m=3\).

### Heisenberg

The geometry dependence is much stronger.

#### \(m=2\)

| geometry | \(B_2/N\) |
|---|---:|
| \(2\times3\) | 3.95 |
| \(2\times4\) | 4.32 |
| \(2\times5\) | 4.11 |
| \(3\times3\) | 6.09 |
| \(3\times4\) | 5.61 |
| \(3\times5\) | 5.59 |

#### \(m=3\)

| geometry | \(B_3/N\) |
|---|---:|
| \(2\times3\) | 1.28 |
| \(2\times4\) | 2.36 |
| \(2\times5\) | 2.13 |
| \(3\times3\) | 4.37 |
| \(3\times4\) | 4.91 |
| \(3\times5\) | 4.25 |

This shows that a fixed-width strip and a more genuinely two-dimensional geometry can have substantially different finite-size coefficients.

A \(2\times L\) strip is therefore **not** a reliable proxy for the true square-lattice bulk coefficient.

---

## 8.2 Trotter geometry scan

For 2D TFIM and Heisenberg, Trotter calculations also showed strong finite-geometry corrections.

For example, at \(T=N\),

### 2D TFIM

| order \(p\) | fitted \(N\)-exponent | expected bulk exponent |
|---:|---:|---:|
| 2 | 4.43 | 4 |
| 4 | 6.42 | 6 |
| 6 | 8.11 | 8 |

### 2D Heisenberg

| order \(p\) | fitted \(N\)-exponent | expected bulk exponent |
|---:|---:|---:|
| 2 | 4.19 | 4 |
| 4 | 6.84 | 6 |
| 6 | 9.46 | 8 |

The higher-order Heisenberg data retain much stronger pre-asymptotic geometry dependence.

This is qualitatively the same phenomenon seen for MPF.

---

# 9. Boundary corrections

The geometry dependence can be organized more systematically by separating bulk, edge, corner, and finite-torus contributions.

For a local error generator, the number of contributing connected local clusters should scale with the volume plus boundary corrections.

The general finite-size structure is expected to be

\[
\boxed{
B_q(\Lambda)
=
a_q^{\mathrm{bulk}}|\Lambda|
+
a_q^{\mathrm{boundary}}|\partial\Lambda|
+
\text{lower-dimensional terms}.
}
\]

---

## 9.1 One dimension

For a 1D chain,

\[
|\Lambda|=N,
\qquad
|\partial\Lambda|=O(1).
\]

Thus

\[
\boxed{
B_q^{\mathrm{OBC}}(N)
=
a_qN+b_q+o(1).
}
\]

PBC removes physical endpoints, so

\[
\boxed{
B_q^{\mathrm{PBC}}(N)
=
a_qN+b_q^{\mathrm{PBC}}+o(1).
}
\]

Hence

\[
\boxed{
B_q^{\mathrm{PBC}}-B_q^{\mathrm{OBC}}
=
O(1),
}
\]

and therefore

\[
\boxed{
\frac{
B_q^{\mathrm{PBC}}-B_q^{\mathrm{OBC}}
}{N}
=
O(N^{-1}).
}
\]

### MPF: 1D TFIM, \(m=3\)

A direct OBC/PBC fit gave

\[
B_3^{\mathrm{OBC}}
\simeq
0.9287N-0.402,
\]

\[
B_3^{\mathrm{PBC}}
\simeq
0.9301N+0.824.
\]

The bulk slopes agree almost perfectly.

For \(N=8,\dots,12\),

\[
B_3^{\mathrm{PBC}}-B_3^{\mathrm{OBC}}
=
1.41,\;
1.11,\;
1.42,\;
1.15,\;
1.24,
\]

which is consistent with an \(O(1)\) boundary contribution.

### MPF: 1D Heisenberg, \(m=3\)

Refined fits gave approximately

\[
a^{\mathrm{OBC}}\simeq0.245,
\qquad
a^{\mathrm{PBC}}\simeq0.232.
\]

The OBC/PBC difference for \(N\ge8\) remains \(O(1)\), again consistent with the same picture.

---

## 9.2 Two dimensions

For an \(L\times L\) square,

\[
N=L^2,
\qquad
|\partial\Lambda|=O(L)=O(\sqrt N).
\]

Therefore the natural OBC expansion is

\[
\boxed{
B_q^{\mathrm{OBC}}(L)
=
a_qL^2+b_qL+c_q+\cdots,
}
\]

or equivalently

\[
\boxed{
B_q^{\mathrm{OBC}}(N)
=
a_qN+b_q\sqrt N+c_q+\cdots.
}
\]

For PBC,

\[
\boxed{
B_q^{\mathrm{PBC}}(N)
=
a_qN+\delta B_q^{\mathrm{torus}}(L),
}
\]

where \(\delta B_q^{\mathrm{torus}}\) should disappear once the torus is large compared with the support of the leading error clusters.

Therefore

\[
\boxed{
\frac{B_q^{\mathrm{OBC}}}{N}
=
a_q+\frac{b_q}{\sqrt N}+O(N^{-1}),
}
\]

whereas

\[
\boxed{
\frac{B_q^{\mathrm{PBC}}}{N}
\to a_q.
}
\]

---

## 9.3 MPF 2D TFIM: direct OBC/PBC comparison

For square lattices, the difference between PBC and OBC was tested directly.

Using operator-norm estimates:

### \(m=2\)

| \(L\) | OBC \(B/N\) | PBC \(B/N\) | difference |
|---:|---:|---:|---:|
| 3 | 4.22 | 6.76 | 2.55 |
| 4 | 3.37 | 5.00 | 1.64 |

Multiplying the difference by \(L\),

\[
L\Delta_L
=
7.64,\;
6.55,
\]

which is compatible with

\[
\Delta_L
\sim
\frac{\mathrm{const}}{L}.
\]

### \(m=3\)

| \(L\) | OBC \(B/N\) | PBC \(B/N\) | difference |
|---:|---:|---:|---:|
| 3 | 1.53 | 3.87 | 2.34 |
| 4 | 1.39 | 2.60 | 1.20 |

The same qualitative decrease is observed.

These results provide direct numerical support for a surface correction of order \(L\).

---

## 9.4 Heisenberg caveat

For 2D Heisenberg, \(L=3,4\) PBC calculations show much stronger finite-torus effects.

Small periodic tori can cause local error clusters to wrap around the system and overlap with themselves. This makes them a poor approximation to the infinite 2D bulk.

Therefore the current Heisenberg OBC/PBC data should be regarded as a geometry diagnostic, not as a precise extraction of \(a_q^{\mathrm{bulk}}\).

Larger \(L\), analytic cluster counting, or linked-cluster methods are preferable.

---

# 10. Dimension-dependent boundary structure

The preceding results motivate the more general expansion

\[
\boxed{
B_q(\Lambda)
=
a_q^{(d)}N
+
b_q^{(d-1)}N^{(d-1)/d}
+
c_q^{(d-2)}N^{(d-2)/d}
+\cdots.
}
\]

For example:

### 1D

\[
B_q=a_qN+b_q+\cdots.
\]

### 2D

\[
B_q=a_qN+b_q\sqrt N+c_q+\cdots.
\]

### 3D

\[
B_q=a_qN+b_qN^{2/3}+c_qN^{1/3}+d_q+\cdots.
\]

This is the natural form if the leading error coefficient is generated by a sum over connected local commutator clusters.

---

# 11. Bulk empirical law

After finite-size corrections are separated, the asymptotic bulk form is

\[
\boxed{
B_q(\mathcal G_N,H_{\mathrm{loc}})
\sim
A_q^{\mathrm{bulk}}(\mathcal G,H_{\mathrm{loc}})N.
}
\]

Hence

\[
\boxed{
\epsilon_q
\sim
A_q^{\mathrm{bulk}}
\frac{
N T^{q+1}
}{
r^q
}.
}
\]

For Trotter,

\[
\boxed{
\epsilon_{\mathrm{Trotter}}^{(p)}
\sim
A_p^{\mathrm{Trotter}}
\frac{
N T^{p+1}
}{
r^p
}.
}
\]

For MPF,

\[
\boxed{
\epsilon_{\mathrm{MPF}}^{(m)}
\sim
A_{2m}^{\mathrm{MPF}}
\frac{
N T^{2m+1}
}{
r^{2m}
}.
}
\]

The empirical segment count is then

\[
\boxed{
r_{\mathrm{emp}}
=
\left\lceil
\left(
\frac{
B_q T^{q+1}
}{
\epsilon_{\mathrm{alg}}
}
\right)^{1/q}
\right\rceil,
}
\]

or in the bulk approximation,

\[
\boxed{
r_{\mathrm{emp}}
=
\left\lceil
\left(
\frac{
A_q^{\mathrm{bulk}}NT^{q+1}
}{
\epsilon_{\mathrm{alg}}
}
\right)^{1/q}
\right\rceil.
}
\]

---

# 12. Physical-time choices and apparent \(N\)-scaling

The observed system-size exponent depends strongly on how the physical simulation time scales with \(N\).

If

\[
T\sim N^\tau,
\]

then

\[
\epsilon_q
\sim
A_q
N^{1+\tau(q+1)}
r^{-q}.
\]

For the commonly used choice

\[
T=N,
\]

\[
\boxed{
\epsilon_q
\sim
A_q
N^{q+2}r^{-q}.
}
\]

For a \(d\)-dimensional local system with propagation time proportional to the linear size,

\[
T\sim L\sim N^{1/d},
\]

one obtains

\[
\boxed{
\epsilon_q
\sim
A_q
N^{1+(q+1)/d}
r^{-q}.
}
\]

Correspondingly,

\[
\boxed{
r
\sim
A_q^{1/q}
N^{1/q+(q+1)/(qd)}
\epsilon^{-1/q}.
}
\]

For MPF with \(q=2m\),

\[
r
\sim
A_m^{1/(2m)}
N^{
\frac{1}{2m}
+
\frac{2m+1}{2md}
}
\epsilon^{-1/(2m)}.
\]

For \(d=2,m=3\),

\[
\boxed{
r
\sim
A_3^{1/6}
N^{3/4}
\epsilon^{-1/6}.
}
\]

---

# 13. MPF order selection

The current analytic branch-count prescription is retained:

\[
\boxed{
m
=
\left\lceil
\frac12
\log\frac{NgT}{\epsilon_{\mathrm{alg}}}
\right\rceil.
}
\]

Here \(g\) is the extensiveness parameter used in the current implementation, not the interaction range.

The empirical model should therefore **not** directly fit a single global function

\[
r(N,T,\epsilon).
\]

Instead:

1. choose \(m\) analytically,
2. use the empirical fixed-\(m\) error model,
3. invert that model for \(r\),
4. feed the resulting plan into the existing resource estimator.

The resulting prescription is

\[
\boxed{
m
=
\left\lceil
\frac12
\log\frac{NgT}{\epsilon_{\mathrm{alg}}}
\right\rceil,
}
\]

followed by

\[
\boxed{
r_{\mathrm{emp}}
=
\left\lceil
\left[
\frac{
B_{2m}(\mathcal G_N,H_{\mathrm{loc}})
T^{2m+1}
}{
\epsilon_{\mathrm{alg}}
}
\right]^{1/(2m)}
\right\rceil.
}
\]

---

# 14. Interpretation of analytic bounds

The numerical results suggest that, for fixed formal order, several rigorous error models may share the same leading dependence

\[
N T^{q+1}r^{-q}
\]

while differing substantially in the coefficient.

This motivates comparing

\[
A_q^{\mathrm{exact}},
\qquad
A_q^{\mathrm{commutator}},
\qquad
A_q^{\ell_1},
\]

rather than comparing only their nominal asymptotic powers.

This may explain why a simple \(\ell_1\)-type estimate can remain resource-competitive even when a commutator-aware error bound is much tighter.

Since

\[
r\propto A_q^{1/q},
\]

a large reduction in the error prefactor is compressed by the \(q\)-th root when translated into a segment count.

---

# 15. Important caveats

## 15.1 Numerical floor

At high order, the asymptotic error rapidly approaches machine precision.

Double precision becomes unreliable once errors reach roughly

\[
10^{-13}\text{--}10^{-15}.
\]

This affects especially:

- MPF with \(m\gtrsim8\),
- sixth-order Trotter at small systems and large \(r\).

High-order calibration should use higher precision or analytic leading-error extraction.

---

## 15.2 Pre-asymptotic segment counts

A fit in \(r\) should only be trusted after the running exponent

\[
\beta_{\mathrm{eff}}
=
-
\frac{
\Delta\log\epsilon
}{
\Delta\log r
}
\]

has stabilized near the formal order \(q\).

This is particularly important for MPFs because coarse-step linear combinations can behave non-monotonically before entering the asymptotic regime.

---

## 15.3 Geometry mixing

Data from

\[
2\times L
\]

strips and

\[
L\times L
\]

square lattices should not be mixed in a single naive \(N\)-fit.

At fixed width, the boundary-to-volume ratio does not vanish, so a strip has a different asymptotic coefficient from a true 2D bulk system.

---

## 15.4 Small periodic systems

PBC does not automatically remove all finite-size effects.

If the support of the leading nested-commutator clusters is comparable with the circumference, clusters can wrap around the torus.

Thus small PBC lattices can differ more strongly from the infinite system than moderately large OBC lattices.

---

## 15.5 Hamiltonian scale

The coefficient \(B_q\) carries the energy scale required by dimensional analysis.

Under

\[
H\to\lambda H,
\]

one expects the leading error to depend on \(\lambda T\).

A more dimensionless final parametrization may therefore take the form

\[
\epsilon_q
\sim
\widetilde B_q
\frac{
(gT)^{q+1}
}{
r^q
},
\]

or an analogous expression using another local energy scale.

The precise dependence on \(g\), coupling strengths, and anisotropy has **not yet been calibrated independently** and should not be hard-coded before a coupling-scaling study.

---

# 16. Recommended empirical calibration strategy

A practical implementation should separate universal exponents from calibrated coefficients.

## Step 1: fix the algorithmic structure

For each method record:

- formal order \(q\),
- Trotter partition,
- MPF schedule,
- local Hamiltonian parameters,
- lattice family,
- boundary condition.

## Step 2: verify the \(r\)-regime

Require

\[
\beta_{\mathrm{eff}}\approx q.
\]

Discard points before this regime and points already at the floating-point floor.

## Step 3: remove the known time dependence

Define

\[
\boxed{
B_q^{\mathrm{obs}}
=
\epsilon
\frac{
r^q
}{
T^{q+1}
}.
}
\]

## Step 4: model finite-size geometry

For a \(d\)-dimensional OBC lattice use

\[
\boxed{
B_q^{\mathrm{obs}}(N)
=
a_qN
+
b_qN^{(d-1)/d}
+
c_qN^{(d-2)/d}
+\cdots.
}
\]

For 1D,

\[
B_q=a_qN+b_q.
\]

For 2D,

\[
B_q=a_qN+b_q\sqrt N+c_q.
\]

For PBC, use

\[
B_q=a_qN+\delta B_q^{\mathrm{torus}}(L)
\]

and discard sizes for which wrap-around effects are visibly large.

## Step 5: store calibrated coefficients

The empirical coefficient table should be indexed at least by

- algorithm: Trotter or MPF,
- formal order,
- model,
- model parameters,
- interaction graph / lattice family,
- boundary condition,
- decomposition / partition,
- MPF schedule where applicable.

## Step 6: invert for resources

Given an algorithmic error budget,

\[
\boxed{
r
=
\left\lceil
\left(
\frac{
B_q T^{q+1}
}{
\epsilon_{\mathrm{alg}}
}
\right)^{1/q}
\right\rceil.
}
\]

This \(r\) can then be passed to the existing resource compiler without changing the downstream gate-count model.

---

# 17. What is numerically established vs still provisional

## Strongly established numerically

The following are supported across several models:

\[
\boxed{
\epsilon_{\mathrm{Trotter}}^{(p)}
\propto
r^{-p},
}
\]

\[
\boxed{
\epsilon_{\mathrm{MPF}}^{(m)}
\propto
r^{-2m},
}
\]

and

\[
\boxed{
\epsilon_q
\propto
T^{q+1}.
}
\]

The time law is directly confirmed for MPF \(m=2,3\) in 2D TFIM and Heisenberg.

The bulk coefficient is extensive in 1D within the accessible size range:

\[
\boxed{
B_q(N)\sim a_qN+O(1).
}
\]

Boundary corrections are directly visible for MPF:

\[
\boxed{
\text{1D: } B=aN+b+\cdots,
}
\]

and the 2D TFIM OBC/PBC comparison is consistent with

\[
\boxed{
\text{2D: } B=aN+b\sqrt N+c+\cdots.
}
\]

---

## Strong empirical hypothesis

For local Hamiltonians in the bulk,

\[
\boxed{
\epsilon_q
\sim
A_q^{\mathrm{bulk}}
\frac{
NT^{q+1}
}{
r^q
}.
}
\]

This is supported by spin, fermionic, 1D, and 2D calculations, but the approach to the thermodynamic limit can be slow for models with complicated local commutator structure.

---

## Still requiring direct verification

1. Direct multi-point \(T\)-sweeps for Trotter at several orders.
2. OBC/PBC boundary scaling for Trotter, analogous to the MPF calculations.
3. Coupling-strength and extensiveness-parameter dependence.
4. True \(L\times L\), \(L\to\infty\) 2D bulk coefficients.
5. High-order calibration beyond the double-precision floor.
6. Analytic or linked-cluster computation of
   \[
   a_q^{\mathrm{bulk}},
   \quad
   b_q^{\mathrm{boundary}},
   \quad
   c_q^{\mathrm{corner}}.
   \]

---

# 18. Main conclusion

The numerical investigations support a common empirical structure for both Trotter and MPF methods:

\[
\boxed{
\epsilon_q(N,T,r)
\simeq
B_q(\mathcal G_N,H_{\mathrm{loc}})
\frac{
T^{q+1}
}{
r^q
}.
}
\]

For local lattice Hamiltonians,

\[
\boxed{
B_q(\Lambda)
=
a_q^{\mathrm{bulk}}|\Lambda|
+
a_q^{\mathrm{boundary}}|\partial\Lambda|
+\cdots.
}
\]

Consequently, the thermodynamic bulk law is

\[
\boxed{
\epsilon_q
\sim
A_q^{\mathrm{bulk}}
\frac{
NT^{q+1}
}{
r^q
}.
}
\]

The exponent structure appears universal across the models tested. The physically important model dependence is concentrated in the coefficient, which encodes:

- local couplings,
- commutator structure,
- lattice coordination,
- aspect ratio,
- boundary condition,
- product-formula decomposition,
- MPF schedule.

This suggests that an empirical resource estimator should **not learn arbitrary exponents from finite data**. Instead, it should enforce the theoretically and numerically supported powers of \(r\) and \(T\), model the finite-size geometry explicitly, and calibrate the remaining coefficients.

That provides a controlled bridge from exact small-system simulations to large-system resource estimates.
