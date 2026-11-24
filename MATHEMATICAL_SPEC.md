# Mathematical Specification — `lyapunov_shield` (v2.0.0)

This document presents the self-contained algebraic, polyhedral, and nonsmooth control-theoretic derivations implemented in `lyapunov_shield`. Every theorem below reduces a continuous-state verification claim over $\mathbb{R}^n$ to a finite set of **pure-integer identities and inequalities over $\mathbb{Z}$** that are checked in $O(m \cdot n)$ integer multiplications and additions without floating-point arithmetic, solver tolerances, or `fractions.Fraction` inside the mathematical kernel.

---

## 1. Exact Integer-Preserving Linear Algebra & Bareiss IPGE Simplex

### 1.1 Canonical Rationals and Common-Denominator Vectors
Inside the exact kernel (`lyapunov_shield/z_linalg.py`), scalars in $\mathbb{Q}$ are represented as primitive integer pairs $(p, q) \in \mathbb{Z} \times \mathbb{Z}_{>0}$ with $\gcd(|p|, q) = 1$ (and $(0, 1)$ for zero). Vectors $x \in \mathbb{Q}^n$ are stored in common-denominator form
$$x = \frac{1}{d} (x_1, \dots, x_n)^\top, \qquad d \in \mathbb{Z}_{>0}, \quad \gcd(|x_1|, \dots, |x_n|, d) = 1.$$
Linear inequalities $a^\top x \le b$ with $(a, b) \in \mathbb{Z}^n \times \mathbb{Z}$ are normalized by dividing solely by $g = \gcd(|a_1|, \dots, |a_n|, |b|) \in \mathbb{Z}_{>0}$, which strictly preserves halfspace orientation.

### 1.2 Sylvester's Determinant Identity & Fraction-Free Bareiss Elimination
Let $M^{(0)} \in \mathbb{Z}^{m \times N}$ and set $D_0 = 1$. At pivot step $k \ge 1$ with nonzero pivot $D_k = M^{(k-1)}_{r_k, c_k} \neq 0$, Bareiss Gauss-Jordan elimination (`bareiss_echelon`, `solve_lp_bareiss`) updates every non-pivot row $i \neq r_k$ and column $j$ via
$$M^{(k)}_{i, j} = \frac{D_k \, M^{(k-1)}_{i, j} - M^{(k-1)}_{i, c_k} \, M^{(k-1)}_{r_k, j}}{D_{k-1}}.$$
By **Sylvester's determinantal identity** (Bareiss, 1968), the numerator is an exact integer multiple of $D_{k-1}$ (`num % prev_piv == 0`), and every entry of $M^{(k)}$ equals a $(k \times k)$ or $((k+1) \times (k+1))$ minor of the original integer matrix $M^{(0)}$. Consequently, intermediate integer bit-lengths grow boundedly as $O(k \log \|M^{(0)}\|_\infty)$ by Hadamard's inequality rather than exponentially as $O(2^k)$ under naive rational Gaussian elimination.

### 1.3 Two-Phase Bareiss Simplex with Avis Lexicographic Anti-Cycling
To solve $\max \{c^\top x : A x \le b, \, x \in \mathbb{R}^n\}$ over $A \in \mathbb{Z}^{m \times n}$, $b \in \mathbb{Z}^m$, $c \in \mathbb{Z}^n$:
1. **Free-variable lifting**: Substitute $x = u - v_0 \mathbf{1}_n$ with $u \in \mathbb{R}_{\ge 0}^n$ and scalar $v_0 \in \mathbb{R}_{\ge 0}$ ($p = n + 1$ nonnegative columns $z = (u^\top, v_0)^\top \ge 0$), yielding $\tilde{A} z + s = b$, $s \in \mathbb{R}_{\ge 0}^m$.
2. **Symbolic lexicographic perturbation** (Avis & Fukuda, 1992): Replace the right-hand side vector $b$ by the $(m \times (m + 1))$ integer matrix
   $$\mathrm{RHS}(\varepsilon) = [\, b \;\mid\; I_m \,], \qquad 0 < \varepsilon_m \ll \varepsilon_{m-1} \ll \dots \ll \varepsilon_1 \ll 1.$$
   In Phase I (when $b_{\min} = \min_i b_i < 0$), choosing the **largest** tied row index $i^* = \max \{i : b_i = b_{\min}\}$ to pivot in scalar artificial variable $z_0 \ge 0$ guarantees that every row of $\mathrm{RHS}^{(0)}$ is strictly lexicographically positive ($\mathrm{RHS}^{(0)}_i \succ_{\mathrm{lex}} 0$).
3. **Finite termination & clean Phase-I exit**: Selecting the entering variable by Bland's smallest-index rule (`OBJ[j] > 0`) and the leaving row by the unique exact integer lexicographic minimum
   $$\frac{1}{T_{i_1, e}} \mathrm{RHS}_{i_1} \prec_{\mathrm{lex}} \frac{1}{T_{i_2, e}} \mathrm{RHS}_{i_2} \iff T_{i_2, e} \, \mathrm{RHS}_{i_1} \prec_{\mathrm{lex}} T_{i_1, e} \, \mathrm{RHS}_{i_2}$$
   maintains $\mathrm{RHS}_i \succ_{\mathrm{lex}} 0$ across every pivot. Hence the perturbed objective strictly increases at every step, basis cycling is algebraically impossible (even on highly degenerate cones such as Trap 8's 12-hyperplane 3D origin cone), and if Phase I reaches $z_0^* = 0$, $z_0$ is guaranteed to have exited the basis cleanly.

### 1.4 Standalone Zero-Trust Farkas Checkers
Given $P = \{x \in \mathbb{R}^n : A x \le b\}$:
- **Infeasibility** (`check_farkas_infeasible`): $P = \emptyset$ is certified by $y \in \mathbb{Z}_{\ge 0}^m$ satisfying
  $$A^\top y = 0 \quad \text{and} \quad b^\top y \le -1.$$
- **Affine Upper Bound** (`check_positivity_farkas`, `check_affine_nonpos_farkas`, `check_affine_neg_farkas`): On nonempty $P$, $(c_{\mathrm{num}} / c_{\mathrm{den}})^\top x \le r_{\mathrm{num}} / r_{\mathrm{den}}$ is certified by $y \in \mathbb{Z}_{\ge 0}^m$ and scale $\sigma \in \mathbb{Z}_{\ge 1}$ satisfying
  $$c_{\mathrm{den}} (A^\top y) = \sigma \, c_{\mathrm{num}}, \qquad r_{\mathrm{den}} (b^\top y) \le \sigma \, r_{\mathrm{num}} \quad (\text{or } < \sigma \, r_{\mathrm{num}} \text{ for strict negativity}).$$
  For any $x \in P$, multiplying $A x \le b$ on the left by $y^\top \ge 0$ yields $(A^\top y)^\top x \le b^\top y$, which upon scaling immediately establishes the bound.

---

## 2. Exact $\ell_\infty$ Chebyshev Full-Dimensionality & Sliver Preservation

A cell $P = \{x \in \mathbb{R}^n : A x \le b\}$ is strictly full-dimensional ($\dim(P) = n$) if and only if it contains an open $\ell_\infty$ ball $B_\infty(x_c, r) = \{x_c + \delta : \|\delta\|_\infty \le r\}$ with $r > 0$. Because
$$\max_{\|\delta\|_\infty \le r} a_i^\top (x_c + \delta) = a_i^\top x_c + r \sum_{j=1}^n |a_{i, j}| = a_i^\top x_c + \|a_i\|_1 r,$$
the exact **$\ell_\infty$ Chebyshev radius** LP (`chebyshev_ball_linf`)
$$\max_{x_c \in \mathbb{R}^n, \, r \in \mathbb{R}} \; r \quad \text{subject to} \quad a_i^\top x_c + \|a_i\|_1 r \le b_i \;\; (i = 1, \dots, m), \quad 0 \le r \le r_{\max}$$
has **100% integer coefficients** ($\|a_i\|_1 \in \mathbb{Z}_{\ge 0}$), requiring no irrational Euclidean $\ell_2$ row norms $\sqrt{\sum_j a_{i,j}^2}$.
- If $r^* \in \mathbb{Q}_{>0}$ (e.g., $r^* = \frac{1}{4 \times 10^{14}}$ on Trap 5b's $10^{-14}$-wide sliver), $P$ is retained with exact interior witness $x_c \in \mathbb{Q}^n$.
- If $r^* = 0$, $P$ lies inside an affine hyperplane ($\dim(P) \le n - 1$, such as the $1{,}016$ degenerate $0\text{D}/1\text{D}$ pullback intersections in Trap 7) and is pruned without unsound tolerance heuristics.

---

## 3. Zero-Hole Homogeneous Origin-Cone Farkas Coercivity

Delta-complete SMT verifiers (`dReal`) and grid/LP samplers must excise an $\varepsilon$-ball $B_\varepsilon(x^*)$ around the equilibrium $x^* \in \mathbb{Q}^n$, leaving a neighborhood where small limit cycles or equilibrium drift can hide. For a continuous piecewise-affine candidate $V(x)$ with $V(x^*) = 0$, `lyapunov_shield` certifies **zero-hole $\ell_1$ coercivity**
$$V(x) \ge \alpha_1 \|x - x^*\|_1 \qquad \forall x \in D$$
across all full-dimensional cells $C_i = \{x : A_i x \le b_i\}$:
1. **Coordinate Charnes-Cooper homogenization**: On each cell $C_i$ with affine piece $V_i(x) = \frac{1}{d_V}(g_i^\top x + c_i)$, homogenize $y = t(x - x^*)$ ($t \ge 0$) and solve $2n$ exact Bareiss LPs ($j = 1, \dots, n$, $\varsigma \in \{+1, -1\}$) to find the tightest rational constants $\mu_{i, j} \in \mathbb{Q}_{\ge 0}$ satisfying
   $$\varsigma (x_j - x^*_j) \le \mu_{i, j} V_i(x) \qquad \forall x \in C_i,$$
   emitting $2n$ integer Farkas certificates (`CoercivityCellCert`).
2. **Summation identity (factor $n$ / $\sum_j \mu_{i,j}$)**: Taking the maximum over $\varsigma \in \{+1, -1\}$ gives $|x_j - x^*_j| \le \mu_{i, j} V_i(x)$ for each $j = 1, \dots, n$. Summing over $j = 1, \dots, n$:
   $$\|x - x^*\|_1 = \sum_{j=1}^n |x_j - x^*_j| \le \left(\sum_{j=1}^n \mu_{i, j}\right) V_i(x) \le \left(n \max_{1 \le j \le n} \mu_{i, j}\right) V_i(x).$$
   Thus $V_i(x) \ge \alpha_{1, i} \|x - x^*\|_1$ holds down to $x = x^*$ with **zero excluded origin hole**, where $\alpha_{1, i} = \left(\sum_{j=1}^n \mu_{i, j}\right)^{-1} > 0$ and global $\alpha_1 = \min_i \alpha_{1, i} > 0$. On cells not containing $x^*$, an additional strict inhomogeneous Farkas dual certifies $-V_i(x) < 0$.

---

## 4. Discrete Pullback Shards, BFT Fake-Crease Merging & Maximal ROA

### 4.1 Two-Stage Pullback Composition & Bemporad-Fukuda-Torrisi (BFT) Merger
For discrete-time dynamics $x_{t+1} = F(x_t)$ with plant cells $\{P_a, F_a\}$ and Lyapunov cells $\{Q_b, V_b\}$, exact discrete decay $V(F(x)) - (1 - \epsilon) V(x) \le 0$ is linear on each full-dimensional **pullback shard**
$$S_{a, b} = \{x \in P_a : F_a(x) \in Q_b\} = \{x : A_{P_a} x \le b_{P_a}, \; (A_{Q_b} W_{F_a}) x \le d_{F_a} b_{Q_b} - A_{Q_b} c_{F_a}\}$$
satisfying $r^*(S_{a, b}) > 0$. Adjacent cells $C_1, C_2$ sharing an identical affine map and an opposing splitting hyperplane $a^\top x \le \beta$ vs. $-a^\top x \le -\beta$ are coalesced via **Bemporad, Fukuda & Torrisi (2001) Theorem 2**: let $P_{\mathrm{cand}}$ be formed by the inequalities of $C_1 \setminus \{a^\top x \le \beta\}$ valid on $C_2$ together with those of $C_2 \setminus \{-a^\top x \le -\beta\}$ valid on $C_1$; then $C_1 \cup C_2$ is convex and equal to $P_{\mathrm{cand}}$ if and only if $P_{\mathrm{cand}} \cap \{a^\top x \le \beta\} \subseteq C_1$ and $P_{\mathrm{cand}} \cap \{-a^\top x \le -\beta\} \subseteq C_2$, certified via exact Farkas bounds (`bft_merge_cells`).

### 4.2 Exact Rational Maximal Sublevel ROA $\Omega_{\rho^*} = \{x \in D : V(x) \le \rho^*\}$
When global decay fails near the boundary of $D$ (e.g., under torque saturation $|u| \le u_{\max}$), `verify_discrete_lyapunov(..., compute_maximal_roa=True)` and `verify_continuous_lyapunov(..., compute_maximal_roa=True)` compute:
1. $\rho_{\partial D} = \min_{x \in \partial D} V(x) \in \mathbb{Q}_{>0}$ across all boundary facets of $D$, and
2. For each non-origin shard $S_s$ where decay residual $\Delta_s(x) > 0$ is feasible, $\rho_s = \min \{V_s(x) : x \in S_s, \, \Delta_s(x) \ge 0\}$.

Setting $\rho^* = \min(\rho_{\partial D}, \min_s \rho_s) \in \mathbb{Q}_{>0}$, every shard intersected with $\{V_s(x) \le \rho^*\}$ is re-certified by an exact Farkas decay dual (`ShardDecayCert`) and every boundary facet of $\partial D$ is certified to satisfy $V(x) \ge \rho^*$ (`BoundaryLevelCert`), establishing both forward invariance and exponential attraction on $\Omega_{\rho^*}$.

---

## 5. Filippov Sliding Creases & Joint Tangent-Multiplier LP

Consider a continuous-time discontinuous or sector-bounded PWA system $\dot{x} \in \mathcal{F}(x)$ with continuous PWA Lyapunov candidate $V(x)$. Across an interior codimension-1 switching face $F_{ij} = C_i \cap C_j \subset \{x : n_{ij}^\top x = \beta_{ij}\}$ (where $n_{ij} \in \mathbb{Z}^n$ points from $C_i = \{n_{ij}^\top x \le \beta_{ij}\}$ into $C_j = \{n_{ij}^\top x \ge \beta_{ij}\}$), continuity of $V$ on $F_{ij}$ implies $\nabla V_j - \nabla V_i = \gamma_{ij} n_{ij}$ for a unique scalar kink jump $\gamma_{ij} \in \mathbb{Q}$ ($\gamma_{ij} > 0$ at convex kinks such as $|x_1|$).

### 5.1 Crease Decomposition Identity & Why Interior Decay Is Unsound
On the attractive sliding subset $F_{ij}^{\mathrm{slide}} = \{x \in F_{ij} : \nu_i(x) = n_{ij}^\top f_i(x) \ge 0, \; \nu_j(x) = n_{ij}^\top f_j(x) \le 0\}$, Filippov's convexification (Shevitz & Paden, 1994) defines the unique tangent sliding vector $f_S(x) = (1 - \alpha(x)) f_i(x) + \alpha(x) f_j(x)$ with $n_{ij}^\top f_S(x) = 0$. Since $f_S(x)$ lies in the tangent subspace $\ker(n_{ij}^\top)$, $\nabla V_i^\top f_S(x) = \nabla V_j^\top f_S(x)$, yielding the **exact crease decomposition identity**:
$$D^+_{f_S} V(x) = (1 - \alpha) \nabla V_i^\top f_i(x) + \alpha \nabla V_j^\top f_j(x) + \gamma_{ij} \big[\alpha (1 - \alpha)(\nu_i(x) - \nu_j(x))\big].$$
Whenever $\gamma_{ij} > 0$ and $\nu_i(x) > 0 > \nu_j(x)$ (inward push from both cells), the third term is **strictly positive**! In Control Trap 1 ($V = |x_1| + |x_2|$, $f_{Q_1} = (-x_1 - 2x_2, x_2)$, $f_{Q_2} = (-x_1 + 2x_2, x_2)$), $\nabla V_q^\top f_q(x) = -\|x\|_1 < 0$ holds strictly throughout all 4 open quadrants, yet at $x^* = (0, \frac{1}{2})$ with $\alpha = \frac{1}{2}$ and $\gamma_{12} = 2$, $D^+_{f_S} V(x^*) = -\frac{1}{2} + 2 \cdot \frac{1}{4} \cdot 2 = +\frac{1}{2} > 0$.

### 5.2 Exact Joint Tangent-Multiplier LP (Codimension-$k$)
On any codimension-$k$ switching ridge $F = \{x : A_F x \le b_F\} \subset \{x : N_F x = r_F\}$ ($N_F \in \mathbb{Z}^{k \times n}$) with incident cell generators $\{f_1, \dots, f_R\}$, worst-case sliding decay requires
$$\max_{x \in F} \;\max_{\alpha \in \Delta_R : \, N_F \sum_r \alpha_r f_r(x) = 0} \sum_{r=1}^R \alpha_r \big(\nabla V_{\mathrm{ref}}^\top f_r(x) + \epsilon_{\mathrm{slide}} V_{\mathrm{ref}}(x)\big) \le 0.$$
Clearing denominators to integer forms $h_r^\top x + \kappa_r$ (decay) and $Q_r x + s_r$ (normal velocity $N_F f_r(x)$), strong LP duality over $\alpha \in \Delta_R$ and $x \in F$ holds via a **shared tangent multiplier $\tau \in \mathbb{Q}^k$** and per-generator Farkas duals $y^{(r)} \in \mathbb{Q}_{\ge 0}^{m_F}$ (`_solve_joint_tangent_lp` in `filippov_crease.py`):
$$(y^{(r)})^\top A_F - \tau^\top Q_r = \sigma \, h_r^\top, \qquad (y^{(r)})^\top b_F + \tau^\top s_r \le -\sigma \, \kappa_r \qquad (r = 1, \dots, R; \; \sigma \in \mathbb{Z}_{\ge 1}).$$
Summing over $\alpha_r \ge 0$ ($\sum_r \alpha_r = 1$) for any $x \in F$ and any tangent combination $\sum_r \alpha_r (Q_r x + s_r) = 0$:
$$\sigma \sum_{r=1}^R \alpha_r (h_r^\top x + \kappa_r) = \sum_{r=1}^R \alpha_r \big((y^{(r)})^\top A_F x + \sigma \kappa_r\big) - \tau^\top \underbrace{\sum_{r=1}^R \alpha_r Q_r x}_{-\sum_r \alpha_r s_r} \le \sum_{r=1}^R \alpha_r \big((y^{(r)})^\top (A_F x - b_F)\big) \le 0.$$
When infeasible, relative-interior Chebyshev probing extracts a concrete rational sliding counterexample $(x^*, \alpha^*, f_S(x^*))$ verified by `check_filippov_witness`. Note also that when $f$ itself is continuous across $F$ (`CONTINUOUS_VECTOR_FIELD`), no sliding with opposing normal jumps can occur (cf. Yarbrough et al., *Hyperlyve*, 2026).

---

## 6. Nagumo-Bouligand NCBF Viability, Corner Deadlocks & Multi-Seed Completeness

Let the safe set be $C = \{x \in D : h(x) \ge 0\} = \bigcup_{q=1}^M V_q$, decomposed into full-dimensional convex valid sub-cells $V_q = \{x : A_{V_q} x \le b_{V_q}\}$. By **Nagumo's Theorem** (1942; Glotfelter, Cortés & Egerstedt, *IEEE L-CSS* 2017 / *IEEE TAC* 2019; Breeden & Panagou, 2023), $C$ is controlled forward invariant under $\dot{x} = f(x) + B u$, $u \in U = \{u : A_u u \le c_u\}$, only if at every boundary state $x \in \partial C$:
$$(f(x) + B U) \cap T_C(x) \neq \emptyset, \qquad T_C(x) = \bigcup_{q : \, x \in V_q} T_{V_q}(x), \quad T_{V_q}(x) = \{v \in \mathbb{R}^n : M_{\mathrm{act}, q}(x) \, v \le 0\},$$
where $M_{\mathrm{act}, q}(x)$ stacks all outward normal rows of $V_q$ active at $x$.

1. **Concave corners (conjunctive tangent cone)**: When $C = \{x : h_1(x) \ge 0 \wedge h_2(x) \ge 0\}$ lies inside a single convex valid cell $V_1$, at the corner $h_1(x^*) = h_2(x^*) = 0$ a **single** control vector $u \in U$ must satisfy **both** tangent inequalities simultaneously:
   $$M_{\mathrm{act}} (f(x^*) + B u) \le 0 \quad \text{and} \quad A_u u \le c_u.$$
   If no such $u$ exists (Control Trap 4), Farkas's Lemma yields nonnegative integer multipliers $(\lambda_{\mathrm{tan}}, \eta_{\mathrm{sat}}) \in \mathbb{Z}_{\ge 0}^{k_{\mathrm{tan}}} \times \mathbb{Z}_{\ge 0}^{m_u}$ (`CornerDeadlockSubCert`) satisfying
   $$\lambda_{\mathrm{tan}}^\top (M_{\mathrm{act}} B_{\mathrm{scaled}}) + \eta_{\mathrm{sat}}^\top (d_B A_u) = 0, \qquad -\lambda_{\mathrm{tan}}^\top (M_{\mathrm{act}} f_{\mathrm{scaled}}(x^*)) + \eta_{\mathrm{sat}}^\top c_u \le -1.$$
2. **Non-convex reflex obstacle corners (disjunctive tangent cone)**: When $C$ wraps around an obstacle exterior (e.g., $|q_x| + |q_y| - \frac{1}{2} \ge 0$), a reflex corner $x^* \in V_{q_1} \cap V_{q_2}$ belongs to multiple valid cells, so $T_C(x^*) = T_{V_{q_1}}(x^*) \cup T_{V_{q_2}}(x^*)$ is non-convex: entering *any* containing valid cell preserves safety, whereas corner deadlock requires emitting a `CornerDeadlockSubCert` on **every** containing $V_q$.
3. **Why scalar Heaviside chain rules fail (Trap 2)**: Defining formal Lie derivatives via $\mathrm{ReLU}'(0) = H \in [0, 1]$ on $h(x) = x_2 - \mathrm{ReLU}(x_1) - \mathrm{ReLU}(-x_1)$ at $x^* = (0, 0)$ along $f = (1, 0)$ produces $(-H + H) \cdot 1 = 0 \ge 0$ by algebraic cancellation, even though $(0, 0) + t(1, 0) = (t, 0)$ immediately violates $h(t, 0) = -|t| < 0$! Exact Bouligand tangent-cone enumeration catches this escape deterministically.
4. **Multi-seed valid-region completeness**: Because $C = \{h(x) \ge 0\}$ can consist of multiple **disconnected** components (e.g., $x_1 \le -\frac{1}{2}$ and $x_1 \ge +\frac{1}{2}$), `verify_barrier` enumerates all full-dimensional valid sub-cells $V_q$ across the entire domain arrangement rather than expanding from a single interior seed, guaranteeing that unsafe boundary leaks on secondary components are never missed.

---

## 7. Algebraic Period-$k$ Cycle Extraction via Cell-Transition Digraphs

Local Jacobian stability at $x^* = 0$ plus global fixed-point uniqueness $\{x \in \mathbb{R}^n : F(x) = x\} = \{0\}$ does **not** imply global asymptotic stability for discrete ReLU maps $x_{t+1} = F(x_t)$ (Control Trap 3). In `orbit_detector.py`:
1. **Exact one-step transition digraph $G = (V, E)$**: After enumerating $F$ into full-dimensional affine cells $\{C_i, F_i(x) = \frac{1}{d_i}(W_i x + c_i)\}$, directed edge $(i \to j) \in E$ is included iff $C_i \cap F_i^{-1}(C_j)$ is feasible under Bareiss Phase-I simplex.
2. **Complete fixed-point & continuum solver**: On each cell $C_i$, solving $\{x \in C_i : (d_i I - W_i) x = c_i\}$ either emits an integer `NonFixedPointCellCert` ($y \ge 0$ proving $F_i(x) \neq x$ on $C_i$), extracts a unique rational fixed point $x^* \in \mathbb{Q}^n$, or characterizes a singular 1D/kD fixed-point continuum via `bareiss_nullspace`.
3. **Unrolled Period-$k$ linear solver**: Along every primitive closed walk $\gamma = (i_0, i_1, \dots, i_{k-1})$ of length $2 \le k \le k_{\max}$ in $G$, the $k$-step composed affine map is $F_\gamma(x) = \frac{1}{d_\gamma}(M_\gamma x + c_\gamma)$. Solving the exact integer linear system
   $$(d_\gamma I_n - M_\gamma) x^{(0)} = c_\gamma$$
   via `solve_one_minus_m_q`, verifying $x^{(t)} = F_{i_{t-1}}(x^{(t-1)}) \in C_{i_t}$ for $t = 0, \dots, k - 1$, and rejecting any proper divisor $d \mid k$ with $x^{(d)} = x^{(0)}$ extracts exact rational minimal Period-$k$ limit cycles such as $\{-\frac{4}{9}, +\frac{4}{9}\}$ and $((2, 0), (0, 2), (-2, -2))$.

---

## 8. Post-Rounding Rational Equilibrium Anchor Subtraction

Suppose a floating-point stabilizing ReLU controller $\pi_\theta : \mathbb{R}^n \to \mathbb{R}^m$ satisfies $\pi_\theta(x^*) = 0$ in IEEE-754 float, and its weights/biases are quantized onto a binary lattice $\frac{1}{S}\mathbb{Z}$ ($S \in \{8, 16, 32, 64, 128\}$) to obtain $\hat{\pi}_Q$. Whenever hidden biases are active at $x^*$ with non-dyadic weights (e.g., $\frac{1}{3} \mathrm{ReLU}(x + \frac{1}{2}) - \frac{1}{6}$ at $S = 16$ in Trap 6), independent rounding produces nonzero residual actuation $\hat{\pi}_Q(x^*) = -\frac{1}{32} \neq 0$, shifting the closed-loop fixed point to $x_{\mathrm{shift}} = -\frac{1}{22} \neq 0$ and destroying origin Lyapunov stability.

Defining the **exact rational equilibrium-anchored network** (`ReLUNetwork.with_anchor_zero` in `polyhedral_complex.py` / `lattice_cegis.py`)
$$\pi_Q^{\mathrm{anc}}(x) := \hat{\pi}_Q(x) - \hat{\pi}_Q(x^*), \qquad V_Q^{\mathrm{anc}}(x) := \hat{V}_Q(x) - \hat{V}_Q(x^*)$$
evaluates $\hat{\pi}_Q(x^*) \in \mathbb{Q}^m$ and $\hat{V}_Q(x^*) \in \mathbb{Q}$ in exact arithmetic *after* weight rounding and subtracts those rational constants as exact output offsets. By construction:
1. $\pi_Q^{\mathrm{anc}}(x^*) = 0$ and $V_Q^{\mathrm{anc}}(x^*) = 0$ hold **identically over $\mathbb{Q}$**,
2. All cell-wise Jacobians $\nabla \pi_Q^{\mathrm{anc}} = \nabla \hat{\pi}_Q$ and $\nabla V_Q^{\mathrm{anc}} = \nabla \hat{V}_Q$ are preserved unchanged, and
3. Origin-cone stability and bounded-iteration Counterexample-Guided Inductive Synthesis (`synthesize_lyapunov_cegis`) recover exact zero-hole Farkas certificates across all lattice scales $S$.
