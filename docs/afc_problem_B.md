# Problem B: Bounded-Force Wrench Containment

Problem B asks: for four fixed contacts and a normal-force cap, does the grasp wrench set contain the scaled limit surface (LS)? This is independent of the Latin-square search (Problem C).

Two bounds govern the pipeline: **$\lambda_{\text{config}}(c)$** (quality of a specific placement) and **$\lambda_{\text{shape}}$** (intrinsic geometric limit of the object). Conflating them misattributes configuration failures to shape degeneracy.

**Screening (Section 11, default $n=4$):** wrench covariance $M$ yields degeneracy index $D = \sigma_1/\sigma_3$. **$D$ above threshold $\Rightarrow$ enable friction at contacts ($\mu_{\text{contact}} \approx 0.2$ often enough); $D$ modest $\Rightarrow$ normal-only at $\lambda = 2$ works for most shapes.** $D$ does not predict exact $\lambda_{\text{shape}}$ when $D$ is small (star counterexample). Do not gate on the Section 10 spectral floor $\lambda_{\text{Floor}}$.

**Three contacts (Section 14):** always use friction. At full AFC ($T=1$, $\lambda=2$), $\mu_{\text{contact}} = 0.2$ succeeds on only ~60% of standard shapes; $\mu_{\text{contact}} = 0.5$ recovered all failures. Those failures are **mostly low-$D$ / well_behaved** — $D$ does **not** predict the required $\mu$ for $n=3$.

---

## 1. Setup

### Wrench columns

Contact $i \in \{1,\ldots,4\}$: inward unit normal $n_i \in \mathbb{R}^2$, position $r_i$ relative to centroid,

$$
\tau_i := r_{i,x} n_{i,y} - r_{i,y} n_{i,x},
\qquad
g_i := \begin{pmatrix} n_{i,x} \\ n_{i,y} \\ \tau_i \end{pmatrix},
\qquad
G = [g_1\; g_2\; g_3\; g_4].
$$

Unit-normal wrench columns match `ContactPoint.calculate_contact_wrench` with $(1,0)$.

### Force cap and grasp wrench set

$\mu_s$ = `static_friction`, $\lambda$ = `force_range_scalar`,

$$
F_{\max} := \lambda\,\mu_s\, m g.
$$

Normal-only grasp wrench set (zonotope):

$$
\mathcal{W}(\lambda) = \left\{ \sum_{i=1}^4 \alpha_i g_i \;:\; 0 \le \alpha_i \le F_{\max} \right\}.
$$

### Limit surface

`calculate_limit_surface` gives $f_{\max} = T\cdot\mu_s m g$, $m_{\max}>0$ with threshold $T$. Write $\mathcal{E}(T)$ for the LS ellipsoid in $(F_x,F_y,\tau)$.

**Problem B (code):** test $\mathcal{E}(T) \subseteq \mathcal{W}(\lambda)$ via three projections $(F_x,F_y)$, $(F_y,\tau)$, $(F_x,\tau)$. Each sample point on the projected ellipse boundary is **LP feasibility**:

$$
\exists\,\alpha:\quad PG\alpha = p,\quad 0 \le \alpha_i \le F_{\max}.
$$

---

## 2. Terminology

| Term | Space | Role of $\lambda$ |
|------|--------|---------------------|
| **Force closure** | $\mathbb{R}^2$ (forces only) | None (existence) |
| **Form closure** | $\mathbb{R}^3$ (wrenches) | None in theorem; code uses $T \ll 1$ |
| **Augmented force closure (AFC)** | $(F_x,F_y)$ projection vs LS disk | **Yes** — envelope translational LS |
| **Full AFC** | all three projections, $T=1$ | **Yes** — envelope full LS |

**Force closure** and **form closure** are not the same. Force closure is strictly weaker (forces only). Form closure concerns immobilization in wrench space $\mathbb{R}^3$.

---

## 3. Form closure

**Markenscoff (planar, frictionless point contacts):** for a compact piecewise-$C^1$ body, there exists a four-contact placement that achieves **form closure** (immobilization against arbitrary external wrenches in the frictionless model).

Mathematically, form closure is a **wrench-space** condition: the grasp must generate wrenches that positively span / contain a neighborhood of the origin in $\mathbb{R}^3$. It is **not** the planar equation $\sum_i \alpha_i n_i = 0$ alone (that is force closure).

**Code proxy:** `check_wrench_space_sufficiency` with `threshold` very small (e.g. $10^{-3}$), `theory_mode=True`. This tests whether $\mathcal{W}(\lambda)$ contains a tiny LS ellipsoid — a bounded-force approximation to wrench sufficiency in $\mathbb{R}^3$.

**Empirical classification (standard shapes, theory search):** all shapes pass except the **frictionless circle** (polygonal approximation of a smooth closed curve with radial normals). The circle is the exceptional form-closure failure in this discrete, capped-force implementation.

---

## 4. Force closure

**Definition (2D, frictionless):** inward normals $\{n_i\}$ **positively span** $\mathbb{R}^2$:

$$
\exists\,\alpha_i > 0:\quad \sum_i \alpha_i n_i = 0.
$$

LP feasibility with equality in $\mathbb{R}^2$ and $\alpha_i \ge \varepsilon$ (see `check_three_edge_force_closure`).

**Nguyen (1988):** in the plane, **three** frictionless point contacts suffice for force closure on **any** object; four are not necessary for force closure alone.

**Circle:** force closure is achievable (e.g. three contacts with non-coplanar inward normals in angular order). No force cap enters the existence question.

---

## 5. Augmented force closure — $(F_x,F_y)$ projection

Force closure is existence with **unbounded** normal magnitudes. Problem B adds a **cap** $F_{\max}$ and asks whether the grasp can **envelope the translational LS disk** in $(F_x,F_y)$. This is **augmented** force closure (translational AFC): force closure plus bounded resistibility to the LS force magnitude.

Project to forces $f_i := n_i$:

$$
\mathcal{W}_{xy}(\lambda) = \left\{ \sum_i \alpha_i f_i : 0 \le \alpha_i \le F_{\max} \right\}.
$$

LS in $(F_x,F_y)$: disk $\mathcal{D}(R)$, $R = T\cdot\mu_s m g$.

### Support function

$$
h_{\mathcal{W}_{xy}}(u) = F_{\max} \sum_{i=1}^4 \max(0,\, u^\top f_i).
$$

$\mathcal{D}(R) \subseteq \mathcal{W}_{xy}(\lambda)$ iff

$$
\forall \|u\|=1:\quad h_{\mathcal{W}_{xy}}(u) \ge R.
$$

With $F_{\max} = \lambda\,\mu_s m g$, define

$$
\kappa_{xy}
:= \min_{\|u\|=1} \sum_{i=1}^4 \max(0,\, u^\top f_i).
$$

Then a **necessary and sufficient** condition for translational AFC at threshold $T$ is

$$
\boxed{\lambda \;\ge\; T / \kappa_{xy}}.
\tag{AFC-xy}
$$

### Lemma (two active contacts)

If only contacts $a,b$ are active in direction $u$, with angle $\varphi$ between $f_a,f_b$,

$$
\max_{0\le\alpha_a,\alpha_b\le F_{\max}} u^\top(\alpha_a f_a + \alpha_b f_b)
= F_{\max}\,\|f_a+f_b\|
= 2F_{\max}\cos(\varphi/2)
$$

when $u$ is the bisector of their positive span.

For four contacts, $\kappa_{xy}$ is the minimum of $\sum_i \max(0,u^\top f_i)$ over all unit directions $u$.

**Circle:** with contacts in four quadrants, $\kappa_{xy} > 0$ and (AFC-xy) holds at **finite** $\lambda$. Augmented force closure in $(F_x,F_y)$ does not require $\lambda \to \infty$ for the circle.

**Failure of $\kappa_{xy}$:** $\kappa_{xy} = 0$ iff normals do not positively span $\mathbb{R}^2$ (e.g. all in one closed half-plane). Then no finite $\lambda$ achieves translational AFC.

---

## 6. Full AFC — three projections ($T=1$)

**Full AFC** (engineering `threshold=1`): require

$$
\mathcal{E}(1) \subseteq \mathcal{W}(\lambda)
$$

in $(F_x,F_y)$, $(F_y,\tau)$, and $(F_x,\tau)$. This adds **torque** LS constraints beyond augmented force closure in $(F_x,F_y)$.

### Theorem (normal forces on smooth convex boundary)

Let the boundary be $C^1$ strictly convex with centroid at the curvature center. At contact, $r$ is centroid-to-contact and inward normal $n$ is anti-parallel to outward normal, hence $r \parallel n$. For unit normal force,

$$
\tau = r \times n = 0.
$$

**Corollary.** Every wrench in the normal-only $\mathcal{W}(\lambda)$ has $\tau=0$:

$$
\mathcal{W}(\lambda) \subseteq \mathbb{R}^2 \times \{0\}.
$$

For $\mu_s > 0$, the LS has $m_{\max} > 0$, so its $(F,\tau)$ projections contain points with $\tau \neq 0$. Hence

$$
\mathcal{E}(1) \not\subseteq \mathcal{W}(\lambda)
\quad\text{for every finite }\lambda
$$

for any four-contact normal-only grasp on such a boundary (in particular the frictionless circle).

This obstruction is **wrench-space** (form-closure / full-AFC level), not force closure in $(F_x,F_y)$.

### Polygonal bodies

On a polygon, $r \not\parallel n$ at vertices and along edges with offset centroid; $|\tau_i|>0$ at most contacts. Full AFC with normal forces only can hold at finite $\lambda_{\text{config}}$ for well-placed contacts on non-circular shapes. As a polygon refines a circle (trap A), per-contact torque leverage $\epsilon_{\max} \to 0$ while $m_{\max}$ stays $O(1)$: $\lambda_{\text{shape}}$ grows without bound even before the strict $\tau \equiv 0$ limit of Section 6.

---

## 7. Configuration bound vs. shape bound

The support-function inequality involves two different quantities that the search pipeline must not conflate.

### Configuration bound ($\lambda_{\text{config}}$)

For a **fixed** four-contact placement $c = \{p_1,\ldots,p_4\}$, define projected columns $h_i^{(p)} = P_p g_i$ for each projection $p \in \{xy,\, F_x\tau,\, F_y\tau\}$ and

$$
\kappa_p(c)
:= \min_{\|u\|=1} \sum_{i=1}^4 \max(0,\, u^\top h_i^{(p)}).
$$

The projected grasp set in projection $p$ is a zonotope with support function

$$
h_{\mathcal{W}_p}(u;\,c) = F_{\max}\sum_{i=1}^4 \max(0,\, u^\top h_i^{(p)}).
$$

The projected LS is a disk of radius $R = T\,\mu_s m g$. Containment $\mathcal{D}(R) \subseteq \mathcal{W}_p(\lambda;\,c)$ requires $h_{\mathcal{W}_p}(u) \ge R$ for all unit $u$, hence

$$
\boxed{\lambda_{\text{config}}(c) \;\ge\; \max_p \; T / \kappa_p(c).}
\tag{AFC-full}
$$

- (AFC-xy) is the $p = xy$ term alone.
- This bound is **necessary** for placement $c$: if it fails, no finite cap works for that configuration.
- It depends on **both** the object shape and the specific contact positions $(r_i, n_i)$.

### Shape-intrinsic bound ($\lambda_{\text{shape}}$)

To ask whether the **object itself** is graspable in the normal-only model (independent of search luck), take the minimum over all valid four-contact sets $\mathbb{C}$:

$$
\lambda_{\text{shape}}
:= \min_{c \in \mathbb{C}} \lambda_{\text{config}}(c).
$$

| Quantity | Answers | Depends on |
|----------|---------|------------|
| $\lambda_{\text{config}}(c)$ | Did **this** placement fail? | shape + configuration |
| $\lambda_{\text{shape}}$ | Is the **shape** fundamentally hard? | shape geometry only |

- **Strictly shape-degenerate:** $\lambda_{\text{shape}} = \infty$. No four-contact normal-only placement achieves full AFC (frictionless circle).
- **Soft shape-degenerate:** $\lambda_{\text{shape}}$ is finite but exceeds hardware capability. Normal-only search is doomed at practical `force_range_scalar`; tangent fallback is required.
- **Configuration failure only:** $\lambda_{\text{config}}(c) = \infty$ but $\lambda_{\text{shape}} < \infty$. A bad symmetric placement (mid-edge rectangle) can fail while the same shape succeeds with a better placement.

### Proposition ($\kappa_p = 0 \Rightarrow \lambda_{\text{config}} = \infty$)

If $\kappa_p(c) = 0$ for some projection $p$, then $h_{\mathcal{W}_p}(u^\ast) = 0$ for some unit $u^\ast$. The LS disk has radius $R > 0$ when $T > 0$, so placement $c$ fails full AFC for **every** finite $\lambda$ in the normal-only model.

This certifies **configuration** failure. It becomes a **shape** statement only when $\kappa_p(c) = 0$ for **every** $c \in \mathbb{C}$ (strict degeneracy).

### Worked examples — $\lambda_{\text{config}}$ for specific placements

Ideal symmetric 4-tuples (`verify_afc_problem_B_bounds.py`):

| Shape | Placement | $\kappa_{xy}$ | $\kappa_{F_x\tau}$ | $\kappa_{F_y\tau}$ | $\lambda_{\text{config}}$ |
|-------|-----------|---------------|--------------------|--------------------|---------------------------|
| Rectangle | mid-edge, 4 sides | $\approx 1.00$ | $0$ | $0$ | $\infty$ (config only) |
| Circle (64-gon) | quadrants | $\approx 1.00$ | $\approx 0$ | $\approx 0$ | $\infty$ (shape-strict) |
| Equilateral triangle | 3 edges + repeat | $\approx 0.86$ | $\approx 0.004$ | $\approx 0.009$ | $\approx 275$ (bad config) |

**Rectangle — configuration vs. shape.** Mid-edge contacts have $\tau_i = 0$, so $\kappa_{F_x\tau} = \kappa_{F_y\tau} = 0$ and $\lambda_{\text{config}} = \infty$ for that symmetric placement. This does **not** mean the rectangle is shape-degenerate: a stochastic search at $\lambda = 1.05$ finds placements with $\kappa_{xy} \approx 1.00$, $\kappa_{F_x\tau} \approx 0.78$, $\kappa_{F_y\tau} \approx 0.69$, hence $\lambda_{\text{config}} \approx 1.46$. So $\lambda_{\text{shape}}(\text{rectangle}) \lesssim 1.5$.

**Circle — shape-strict.** On a $C^1$ strictly convex boundary with centroid at the curvature center, every normal contact has $\tau = 0$ (Section 6). Then $\kappa_{F_x\tau} = \kappa_{F_y\tau} = 0$ for **every** placement, so $\lambda_{\text{shape}} = \infty$: strictly degenerate regardless of search.

**Translational layer (AFC-xy).** For axis-aligned rectangle with four side normals, $\kappa_{xy} = 1$ at mid-edge. For two contacts with angle $\varphi$, the bisector gives $\kappa_{xy} = 2\cos(\varphi/2)$; e.g. $\varphi = 90^\circ \Rightarrow \kappa_{xy} = \sqrt{2}$.

---

## 8. Shape degeneracy taxonomy and tangent forces

### Strict vs. soft shape degeneracy

| Class | Condition | Normal-only full AFC | Tangent fallback |
|-------|-----------|----------------------|------------------|
| **Well-behaved** | $\lambda_{\text{shape}}$ modest | feasible at practical $\lambda$ | optional |
| **Soft shape-degenerate** | $\lambda_{\text{shape}}$ finite but huge | fails at practical $\lambda$ | **required** |
| **Strict shape-degenerate** | $\lambda_{\text{shape}} = \infty$ | impossible at any $\lambda$ | **required** |

$\lambda_{\text{config}}(c) = \infty$ for a single bad placement is **not** shape degeneracy unless it holds for all $c \in \mathbb{C}$.

### Four geometric traps (soft shape-degeneracy)

These are **shape-intrinsic** mechanisms that inflate $\lambda_{\text{shape}}$ even when force closure and AFC-xy remain feasible at moderate $\lambda_{\text{config}}$.

| Trap | Geometry | Mechanism | Rough bound |
|------|----------|-----------|-------------|
| **A. Torque ($\epsilon$-concurrent)** | High-res polygons, near-circular ellipses | Max normal moment arm $\epsilon_{\max} \to 0$; translation easy, torque needs huge forces | $\lambda_{\text{shape}} \gtrsim T / (4\epsilon_{\max})$ |
| **B. Translational (high aspect ratio)** | Needles, narrow rectangles, extreme ovals | Surface area favors one axis; force closure in the deficient axis forces contacts onto tiny end-caps | $\lambda_{\text{shape}} \propto \rho$ (aspect ratio) |
| **C. Directional torque (chiral / ratchet)** | Saw blades, ratchet gears | Asymmetric ramps: one torque sign easy, the opposite requires collinear cliff faces | asymmetric $\kappa_{F\tau}$ by sign |
| **D. Coupled (CoM offset)** | Composite levers, weighted booms | Shifted CoM: contacts near CoM lose leverage ($r_i \approx 0$), far contacts spike torque | translation–rotation tightly coupled |

Trap A connects to Section 6: as a polygon refines a circle, per-contact $\tau_i \to 0$ while $m_{\max}$ stays $O(1)$ — torque soft-degeneracy even before the strict $\tau \equiv 0$ limit. Traps B–D explain why needle, crescent, and asymmetric shapes need `used_tangent_as_fallback=True` in practice.

### Theorem (strict degeneracy — smooth boundary)

On a $C^1$ strictly convex boundary with centroid at the curvature center, $\tau_i = 0$ for every normal contact. Then $g_i \in \mathbb{R}^2 \times \{0\}$, and for $u = (0,1)$ (pure torque),

$$
\sum_i \max(0,\, u^\top h_i^{(F_x\tau)}) = 0
\quad\Rightarrow\quad
\kappa_{F_x\tau}(c) = 0
$$

for every placement $c$. Hence $\lambda_{\text{shape}} = \infty$: normal-only full AFC is impossible at any `force_range_scalar`. This is independent of $\kappa_{xy}$.

### Corollary (when friction is necessary)

If $\lambda_{\text{shape}} = \infty$, or $\lambda_{\text{shape}}$ exceeds hardware limits, while the LS requires torque ($m_{\max} > 0$):

1. **Normal-only** (`enable_tangent_forces=False`): Problem B cannot be satisfied at practical $\lambda$.
2. **Friction cone** ($|t_i| \le \mu_\ell n_i$): unit tangent wrenches $g_i^{(t)} = (t_{i,x}, t_{i,y}, r_{i,x} t_{i,y} - r_{i,y} t_{i,x})$ generally have $\tau \neq 0$ even when normal columns have $\tau_i = 0$. The enlarged set $\mathcal{W}^{\mathrm{fric}}(\lambda)$ has $\kappa_p^{\mathrm{fric}} > 0$ in torque projections; full AFC holds at finite $\lambda$ (circle, crescent, narrow triangle with `used_tangent_as_fallback=True`).

Increasing `force_range_scalar` cannot substitute for friction when $\lambda_{\text{shape}} = \infty$.

### Friction cone (implementation)

With $|t_i| \le \mu_\ell n_i$, each contact contributes a convex force cone; LP-feasibility tests still apply (`calculate_wrench_space` with `enable_tangent_forces=True`).

---

## 9. Wrench covariance matrix — $O(1)$ shape screening

Stochastic search estimates $\lambda_{\text{shape}}$ by sampling $\mathbb{C}$. An alternative is to integrate normal-only wrench capacity over the entire boundary $\partial\mathcal{O}$ in one pass after discretization.

### Local wrench (relative to CoM)

At boundary point $x$ with outward normal $n(x)$ (inward $-n$), scale the object so max radius from CoM is $1$:

$$
g(x) = \begin{pmatrix} -n_x \\ -n_y \\ (x_x - x_{\mathrm{CoM},x})(-n_y) - (x_y - x_{\mathrm{CoM},y})(-n_x) \end{pmatrix}.
$$

(Align sign convention with Section 1: inward normal, torque about CoM.)

### Continuous integral

$$
M = \oint_{\partial\mathcal{O}} g(x)\, g(x)^\top \, ds
= \oint_{\partial\mathcal{O}} \begin{pmatrix} n_x^2 & n_x n_y & n_x \tau \\ n_x n_y & n_y^2 & n_y \tau \\ n_x \tau & n_y \tau & \tau^2 \end{pmatrix} ds.
$$

$M$ is a symmetric $3 \times 3$ matrix summarizing the shape's **global** normal-only wrench capacity (not any single 4-tuple).

### Shape degeneracy index

Let eigenvalues $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$ of $M$. Define

$$
D = \frac{\sigma_1}{\sigma_3}
\quad (\infty \text{ if } \sigma_3 = 0).
$$

| $D$ | Interpretation | $\lambda_{\text{shape}}$ proxy |
|-----|----------------|--------------------------------|
| $\approx 1$–$10$ | Well-behaved (square-like) | low |
| $\gg 100$ | Soft-degenerate (needle, weighted boom) | very large; skip normal-only search |
| $\infty$ ($\sigma_3 = 0$) | Strictly degenerate (circle) | $\lambda_{\text{shape}} = \infty$; use friction cone immediately |

**Use in pipeline:** compute $M$, $\sigma_3$, and $D$ before Latin-square search. **Gate on $D$ and $\sigma_3$**, not on $\lambda_{\text{Floor}}$ (Section 10 warning). Large $D$ or tiny $\sigma_3$ $\Rightarrow$ `used_tangent_as_fallback=True`.

*Implemented in `grasp_covariance.py`; run `scripts/test/test_calculate_grasp_covariance.py`.*

---

## 10. Spectral bounding theorem (theoretical background only)

> ### ⚠️ WARNING — Do not use $\lambda_{\text{Floor}}$ from this section for engineering decisions
>
> The spectral bounding theorem (Steps 1–4 below) proves a **weak existential lower bound**
> $\lambda_{\text{shape}} \gtrsim C\sqrt{D}$ under idealized $C^1$ assumptions and a **uniform**
> Sobolev constant $K(\kappa_{\max}, L)$ that **cannot be estimated reliably** from a polygon mesh.
>
> In code and in practice:
>
> - **$\lambda_{\text{Floor}}$ is often orders of magnitude below true $\lambda_{\text{shape}}$** (rectangle: floor $\sim 0.3$, search finds $\sim 1.5$).
> - **$\lambda_{\text{Floor}}$ can disagree with $D$** (Sobolev singularity: huge $D$, tiny floor on near-circles).
> - **$K_{\mathrm{deriv}}$ and $K_{\mathrm{tight}}$ are loose in opposite directions** on different shapes; `min(K_tight, K_deriv)` is a patch, not a theorem.
> - **The bound is not predictive:** it does not upper-bound or tightly estimate $\lambda_{\text{shape}}$; it only says “$\lambda_{\text{shape}}$ cannot be *smaller than* something” — and that something is often trivially small.
>
> **What to use instead:** treat **$D = \sigma_1/\sigma_3$** and **$\sigma_3$** as **shape screening indices** (Section 12) to decide normal-only vs friction-cone search. Use **stochastic search** or $\kappa_p(c)$ on a candidate 4-tuple for actual $\lambda_{\text{config}}$. **Do not gate the pipeline on `lambda_shape_lower_bound`.**

The degeneracy index $D$ is valuable as an **$O(1)$ correlate** of grasp difficulty, not because the spectral floor is tight.

### Setup (recap)

$$
\lambda_{\text{shape}}
= \min_{c \in \mathbb{C}} \max_{p \in \{xy,\, F_x\tau,\, F_y\tau\}} \frac{T}{\kappa_p(c)},
\qquad
D = \frac{\sigma_1}{\sigma_3}.
$$

Assume the object is scaled to unit max CoM radius (as in Section 9) and $\partial\mathcal{O}$ is $C^1$ with bounded boundary curvature.

### Theorem (spectral lower bound)

If $\sigma_3 > 0$, there exists a constant $C = C(T, \sigma_1, \kappa_{\max}) > 0$ depending on the engineering threshold $T$, the largest eigenvalue $\sigma_1$, and a curvature bound $\kappa_{\max}$ on the wrench field $g(s)$ along $\partial\mathcal{O}$, such that

$$
\boxed{\lambda_{\text{shape}} \;\ge\; C\,\sqrt{D}}.
$$

This is a **lower bound**, not an equality: large $D$ forces large $\lambda_{\text{shape}}$, but $\lambda_{\text{shape}}$ can exceed $C\sqrt{D}$.

When $\sigma_3 = 0$ ($D = \infty$), the bound is vacuous; use the strict degeneracy argument of Sections 6 and 8 instead ($\lambda_{\text{shape}} = \infty$).

### Proof sketch

**Step 1 — Weakest eigenvector.** Let $u_3$ be a unit eigenvector of $M$ for $\sigma_3$. By the Rayleigh quotient,

$$
\sigma_3 = \oint_{\partial\mathcal{O}} (u_3^\top g(s))^2\, ds.
$$

Write $f(s) = u_3^\top g(s)$, so $\sigma_3 = \oint f(s)^2\, ds$.

**Step 2 — Discrete support in direction $u_3$.** For any four-contact configuration $c = \{s_1,\ldots,s_4\}$,

$$
S(c, u_3) := \sum_{i=1}^4 \max(0,\, f(s_i)) \;\le\; 4\, f_{\max},
\qquad f_{\max} := \max_{s \in \partial\mathcal{O}} f(s).
$$

**Step 3 — Sobolev / curvature bridge.** Along a $C^1$ closed boundary of length $L$, bounded curvature of $g(s)$ (induced by bounded normal rotation and bounded moment arms) gives an $L^\infty$–$L^2$ inequality: there exists $K = K(\kappa_{\max}, L)$ such that

$$
f_{\max} \;\le\; \sqrt{K\,\sigma_3}.
$$

*(Standard 1D Sobolev/Gagliardo–Nirenberg: peak boundary wrench in the weakest direction cannot be large if its $L^2$ energy $\sigma_3$ is small unless curvature concentrates it; bounded $\kappa_{\max}$ caps that concentration.)*

Hence $S(c, u_3) \le 4\sqrt{K\,\sigma_3}$ for **every** $c \in \mathbb{C}$.

**Step 4 — Link to $\kappa_p$.** Rotate the body frame so $u_3$ lies in one of the AFC projection planes (always possible in 2D: weakest mode is force–force, $F_x$–$\tau$, or $F_y$–$\tau$). Let $p^\ast$ be that projection and $\tilde{u} = P_{p^\ast} u_3 / \|P_{p^\ast} u_3\|$. Then $u_3^\top g_i = \tilde{u}^\top h_i^{(p^\ast)}$ and

$$
\kappa_{p^\ast}(c)
= \min_{\|v\|=1} \sum_i \max(0,\, v^\top h_i^{(p^\ast)})
\;\le\; \sum_i \max(0,\, \tilde{u}^\top h_i^{(p^\ast)})
= S(c, u_3)
\;\le\; 4\sqrt{K\,\sigma_3}.
$$

Therefore, for every $c$,

$$
\max_p \frac{T}{\kappa_p(c)} \;\ge\; \frac{T}{\kappa_{p^\ast}(c)} \;\ge\; \frac{T}{4\sqrt{K\,\sigma_3}}.
$$

Taking the minimum over $c \in \mathbb{C}$,

$$
\lambda_{\text{shape}}
\;\ge\; \frac{T}{4\sqrt{K\,\sigma_3}}
\;=\; \frac{T}{4\sqrt{K\,\sigma_1}}\,\sqrt{\frac{\sigma_1}{\sigma_3}}
\;=\; \underbrace{\left(\frac{T}{4\sqrt{K\,\sigma_1}}\right)}_{C}\,\sqrt{D}.
$$

### Corollaries (theory only — not operational thresholds)

1. **Soft degeneracy (conceptual):** as $\sigma_3 \to 0^+$, $D \to \infty$ and normal-only full AFC becomes arbitrarily hard.
2. **Strict degeneracy:** $\sigma_3 = 0$ on the frictionless circle $\Rightarrow$ $\lambda_{\text{shape}} = \infty$ (Section 6) — friction is **necessary**, not optional.
3. **Do not read Corollary 1 as** “$\lambda_{\text{shape}} \approx C\sqrt{D}$” numerically; the constant $C$ is unknown and the discrete floor is unreliable (Section 11).

### What this section does not provide

- **Not a numeric estimate of $\lambda_{\text{shape}}$.**
- **Not a substitute for** `find_the_magnum_stochastic` or $\kappa_p(c)$ on a placement.
- **Not biconditional:** low $D$ does not guarantee easy grasp (e.g. star $D \approx 1.5$ can still fail normal-only search at $\lambda = 1.05$); high $D$ strongly suggests friction.

---

## 11. Why $D$ predicts grasp difficulty (and when friction is required)

Section 10’s bound is too weak to **compute** $\lambda_{\text{shape}}$. **$D$ is still useful** as a **shape screening statistic** because it measures the same object property the grasp problem cares about: **how ill-conditioned is the normal-only wrench capacity of the boundary?**

### Mechanism: $M$ as global wrench capacity

$M = \oint g(s)g(s)^\top ds$ aggregates, over the entire boundary, how much unit normal force contributes to each wrench axis. Its eigenvalues $\sigma_1 \ge \sigma_2 \ge \sigma_3$ are the **principal energies** of that capacity:

| Eigenvalue | Meaning |
|------------|---------|
| $\sigma_1$ | Strongest wrench direction the boundary supports with normal-only actuation |
| $\sigma_3$ | Weakest wrench direction the boundary supports |
| $D = \sigma_1/\sigma_3$ | **Condition number** of continuous normal-only wrench capacity |

**High $D$** means: the shape can push hard in some wrench directions (large $\sigma_1$, good forces/torques along dominant geometry) but is **nearly blind** in the weakest direction $u_3$ (tiny $\sigma_3$). Full AFC must envelope the limit surface in **all** projections, including torque. The bottleneck projection aligns with the weak mode $u_3$ (often torque-dominated on near-circular or needle-like bodies).

So $D$ does not need the Sobolev proof to be meaningful: it is the **condition number of the shape’s normal-only wrench covariance** — a direct summary of “can this boundary supply 3D wrench span without friction?”

### Link to $\lambda_{\text{shape}}$ (qualitative, not the Section 10 floor)

For any four-contact placement, $\kappa_p(c)$ is built from **four samples** of the same $g(s)$. If the **continuous** weakest capacity $\sigma_3$ is tiny, then **no sparse subset** of boundary points can produce large support in direction $u_3$ without enormous normal magnitudes:

$$
\lambda_{\text{config}}(c) = \max_p \frac{T}{\kappa_p(c)} \quad\text{grows when }\kappa_{p^\ast}(c)\text{ is starved in the weak direction.}
$$

Minimizing over placements,

$$
\lambda_{\text{shape}} = \min_c \lambda_{\text{config}}(c)
$$

rises when **every** 4-tuple struggles in the $u_3$ projection — exactly when $D$ is large and $\sigma_3$ is small. This is the **engineering** explanation for why large $D$ correlates with large $\lambda_{\text{shape}}$ without trusting $\lambda_{\text{Floor}}$:

| Regime | $D$, $\sigma_3$ | $\lambda_{\text{shape}}$ (normal-only) | Friction |
|--------|-----------------|----------------------------------------|----------|
| Well-conditioned | $D \lesssim 10$, $\sigma_3 \gtrsim 0.3$ | low ($\lesssim 2$ at $T=1$) | optional |
| Soft-degenerate | $D \gg 10^2$, $\sigma_3 \ll 0.1$ | very large or search timeout | **required** |
| Strict-degenerate | $\sigma_3 \to 0$ (circle) | $\infty$ | **required** |

**Empirical check** (`force_range_scalar = 1.05`, engineering search): rectangle $D \approx 5.5$ passes normal-only; narrow triangle $D \approx 192$, obese triangle $D \approx 326$, circle $D \approx 10^3$ require tangent fallback; crescent $D \approx 76$ passes only with friction enabled.

### Why friction is required when $D$ is large

Normal-only wrenches from the boundary span a zonotope whose **torque thickness** is controlled by $\sigma_3$. When $\sigma_3 \approx 0$:

1. **Strict case (circle):** every normal wrench has $\tau = 0$; $\mathcal{W}(\lambda) \subset \mathbb{R}^2 \times \{0\}$ while the LS needs $\tau \neq 0$ — **no finite $\lambda$** works (Section 6).
2. **Soft case (needle, obese triangle, near-circle):** $\sigma_3 > 0$ but small; normal-only AFC is **theoretically possible** at huge $\lambda_{\text{shape}}$ but **practically impossible** at `force_range_scalar` $\approx 1$–$2$.
3. **Friction cone:** tangent forces add wrench columns $g^{(t)}$ with generically $\tau \neq 0$, increasing the effective $\sigma_3$ of the enlarged actuation set. Search finds finite-$\lambda$ solutions with `enable_tangent_forces=True`.

**Decision rule (pipeline):**

```
IF σ₃ < ε_strict           → strict degenerate → friction mandatory, skip normal-only search
ELIF D ≥ D_soft (e.g. 100) → soft degenerate   → used_tangent_as_fallback = True
ELIF normal-only search fails at λ_hw           → friction fallback
ELSE                         → normal-only search OK
```

**Ignore `lambda_shape_lower_bound` for gating** unless `lambda_floor_trusted=True` and you treat it as a loose hint only.

### Sobolev singularity (why the proof and $D$ disagree numerically)

### The symptom

The computational floor (with discrete Sobolev constant) is

$$
\lambda_{\text{Floor}}
= \frac{T}{4\sqrt{K\,\sigma_3}}
= \left(\frac{T}{4\sqrt{K\,\sigma_1}}\right)\sqrt{D}.
$$

When estimating $K$ from the weakest mode via

$$
K \;\approx\; \frac{2}{L} + \frac{L}{2\sigma_3}\oint \left(\frac{df}{ds}\right)^2 ds,
$$

the second term scales as $K \propto 1/\sigma_3$ as $\sigma_3 \to 0^+$. Substituting into $\lambda_{\text{Floor}}$ **cancels** $\sqrt{\sigma_3}$:

$$
\lambda_{\text{Floor}}
\;\propto\;
\frac{\sqrt{\sigma_1/\sigma_3}}{\sqrt{(L^2/2\sigma_3)\int (f')^2 ds}}
\;\sim\; \mathcal{O}(1)
\quad\text{(finite, misleading)}.
$$

So a polygonal circle can show $D \gg 1$ while $\lambda_{\text{Floor}}$ stays small. **$D$ and $\lambda_{\text{Floor}}$ appear to disagree** — but they answer different questions at the singularity.

### Diagnosis (proof vs. implementation)

- **Theorem (Section 10):** uses a **uniform** Sobolev constant $K(\kappa_{\max}, L)$, independent of $\sigma_3$, such that $f_{\max} \le \sqrt{K\sigma_3}$. When $f_{\max} \to 0$ (strict degeneracy), the bound $0 \le \sqrt{K\sigma_3}$ is **loose** but the **direct** argument $\kappa_{p^\ast}=0 \Rightarrow \lambda_{\text{shape}}=\infty$ still applies.
- **Discrete estimator:** when $f_{\max} \approx 0$ but $\int (f')^2 \gg 0$ (high-frequency torque ripple on a near-circle), $K$ explodes while $f_{\max}$ is tiny. Inverting a loose **upper** bound on $f_{\max}$ **shrinks** $\lambda_{\text{Floor}}$ — the inequality direction is wrong for use as a tight floor at the singularity.

**The proof is valid in theory; the discrete $\lambda_{\text{Floor}}$ is not a useful output.** $D$ and $\sigma_3$ remain the operational signals.

**Discrete $\lambda_{\text{Floor}}$ (optional diagnostic):** `grasp_covariance.py` computes

$$\lambda_{\text{Floor}} = \frac{T}{4\sqrt{K\,\sigma_1}}\sqrt{D}, \quad K = \min(K_{\mathrm{tight}}, K_{\mathrm{deriv}})$$

for logging only. See warning at Section 10.

### Degeneracy gate (`grasp_covariance.py`)

Do not trust $\lambda_{\text{Floor}}$ alone. Branch on $\sigma_3$ and $D$ first:

| Step | Condition | Action |
|------|-----------|--------|
| **1. Strict gate** | $\sigma_3 < \varepsilon_{\text{strict}}$ | $\lambda_{\text{shape}}=\infty$; friction required; skip normal-only search |
| **2. Soft gate** | $D \ge D_{\text{soft}}$ (e.g. $100$) | `used_tangent_as_fallback=True` |
| **3. Search** | normal-only fails at `force_range_scalar` | enable friction cone |

**Source of truth:** $\sigma_3$ and **$D$** for branching; $\lambda_{\text{Floor}}$ is diagnostic only when `lambda_floor_trusted=True`.

### Star counterexample: low $D$ does not mean low $\lambda_{\text{shape}}$

The five-point **star** is the cleanest counterexample to using $D$ as a proxy for $\lambda_{\text{shape}}$ or search ease:

| Quantity | Star value | Interpretation |
|----------|------------|----------------|
| $\sigma_1,\sigma_2,\sigma_3$ | $\approx 1.66,\,1.66,\,1.13$ | All $O(1)$ — not torque-starved |
| $D$ | $\approx 1.5$ | **Well-conditioned** (same order as rectangle) |
| $\lambda_{\text{Floor}}$ | $\approx 0.28$ | Misleadingly small (do not gate on this) |

**Empirical search** (`find_the_magnum_stochastic`, normal-only, $T=1$, 12 s timeout):

| `force_range_scalar` | Result | Configs tried |
|----------------------|--------|---------------|
| $1.05$ | **fail** (timeout) | $\sim 2700$ |
| $1.20$ | **fail** (timeout) | $\sim 2300$ |
| $\sqrt{2} \approx 1.414$ | **success** | $\sim 125$ |
| $2.0$ | **success** | $\sim 30$ |

So $\lambda_{\text{shape}}(\text{star})$ lies **between $1.2$ and $1.414$** for this pipeline — not at $1.05$, and **not** because the shape is degenerate.

On a successful placement at $\lambda = \sqrt{2}$, per-projection support numbers (`verify_afc_problem_B_bounds.py`) are:

| Projection $p$ | $\kappa_p$ | $T/\kappa_p$ |
|----------------|------------|--------------|
| $(F_x,F_y)$ | $0.80$ | $1.25$ |
| $(F_x,\tau)$ | $0.71$ | $1.41$ |
| $(F_y,\tau)$ | **$0.62$** | **$1.62$** |

The bottleneck is **$(F_y,\tau)$**, not translational AFC-xy. The familiar $\sqrt{2}$ scale comes from **two orthogonal normals** in the force plane; the star satisfies that layer easily ($\kappa_{xy} \approx 0.80$). **Full AFC** on a non-convex five-point boundary needs a favorable mix of tip and re-entrant contacts and enough cap in the torque projections — hence $\lambda_{\text{shape}} \approx 1.6$ on a good placement, while $D \approx 1.5$ suggests “easy.”

**What failed at $\lambda = 1.05$:** neither shape degeneracy nor missing friction. The search explored thousands of Latin-square placements; none passed Problem B before timeout. Just above the threshold, a valid 4-tuple appears quickly. This is **Problem C** (search + marginal $\lambda$), not trap A/B/C from Section 8.

**Lesson:** $D$ answers “does the **continuous** boundary starve torque under normal-only actuation?” The star does not. For “will $\lambda = 1.05$ suffice?” use **$\kappa_p(c)$ on placements** or search at hardware $\lambda$ — not $D$ alone. Triangle and trapezoid failing at $1.05$ with modest $D$ follow the same pattern.

### Conclusion — what to use in the pipeline

Three statistics, three roles:

| Statistic | Role | Do **not** use it for |
|-----------|------|------------------------|
| $\sigma_3$ | Strict degeneracy ($\to \infty$ $\lambda_{\text{shape}}$) | Estimating exact $\lambda_{\text{shape}}$ on well-behaved shapes |
| $D$ | **Friction screening** — large $D$ $\Rightarrow$ normal-only torque capacity is ill-conditioned | Predicting $\lambda_{\text{shape}}$ when $D$ is small |
| $\kappa_p(c)$ / search | **$\lambda$ and placement** — whether a given cap and 4-tuple pass Problem B | Replacing $D$ for friction gating |

**Operational summary (engineering mode, $T=1$):**

1. **$D \lesssim D_{\text{soft}}$** (e.g. $100$) **and** $\sigma_3 \gtrsim \varepsilon_{\text{strict}}$: normal-only search at **`force_range_scalar = 2`** succeeds for **most** standard polygons (rectangle, plus, ellipse, star, etc.). This is the common case — no tangent forces at contacts.
2. **$D \ge D_{\text{soft}}$** or **$\sigma_3 < \varepsilon_{\text{strict}}$**: enable **`used_tangent_as_fallback=True`** (friction cone at contacts) **before** or early in search. Normal-only at $\lambda \approx 1$–$2$ is unlikely to succeed (needle, obese triangle, circle; often crescent at $D \approx 76$ even below $D_{\text{soft}}$).
3. **Low $D$ but search fails** at the target $\lambda$: do **not** jump to friction — raise $\lambda$ or improve placement search first (star at $\lambda = 1.05$; mid-edge rectangle is pure configuration failure).
4. **Ignore `lambda_shape_lower_bound` for gating** — Section 10’s spectral floor is theory-only; it can be orders of magnitude below true $\lambda_{\text{shape}}$ (star) or misleading at the Sobolev singularity (circle).

**One-line takeaway:** **$D$ above threshold $\Rightarrow$ friction at contacts; $D$ below threshold $\Rightarrow$ try normal-only at $\lambda = 2$ first** — but $D$ does not certify that $\lambda = 1.05$ is enough, and moderate $D$ (crescent) can still need friction.

### 11.5 Hardware actuator gate (before search)

Problem B at `force_range_scalar = λ` assumes each pusher can supply up to $F_{\max} = \lambda\,\mu_s m g$ normal force at its contact. **Actuators can be sanity-checked independently of wrench geometry**, but with D/σ₃ + product-µ search this gate is secondary (warn by default in the revised holonomic test).

**PyBullet product model (revised path):** object **material** µ and robot **bumper** µ multiply:

$$
\mu_{\mathrm{contact}} = \mu_{\mathrm{material}} \times \mu_{\mathrm{bumper}}.
$$

Search / GWS use $\mu_{\mathrm{contact}}$ via `GenericObject.get_contact_friction()` — **not** ground `static_friction`. API: `set_material_friction`, `apply_bumper_contact_model`, helpers in `friction_model.py`.

Let $f_{\max}$ be the translational limit-surface scale from `calculate_limit_surface` at threshold $T=1$. With holonomic omniwheel robots, estimate per-robot push capacity $F_{\mathrm{robot}}$ from wheel motor force and floor friction (`afc_hardware.estimate_robot_max_push_force`).

| Mode | Required per robot |
|------|-------------------|
| Normal-only | $F_{\mathrm{robot}} > \lambda\, f_{\max}$ |
| Tangent / soft-degenerate ($D \ge D_{\mathrm{soft}}$) | $F_{\mathrm{robot}} > \lambda\, f_{\max} / \mu_{\mathrm{contact}}$ |

The tangent inequality divides by $\mu_{\mathrm{contact}}$ because the friction cone splits the actuator budget between normal and tangential components at each contact: achieving the same wrench cap with $|t_i| \le \mu_\ell n_i$ needs larger total push force when $\mu_\ell$ is small.

If the hardware gate fails under `--strict-hw-gate`, **no AFC configuration exists at the assumed $\lambda$ on these robots** — skip search and report infeasibility. Default revised behavior: warn and continue; rely on stochastic search.

---

## 12. Code correspondence

| Concept | Implementation |
|---------|----------------|
| $g_i$, $G$ | `ContactPoint.calculate_contact_wrench` |
| $\mathcal{W}(\lambda)$ | `WrenchSpaceVisualizer.calculate_wrench_space` |
| $\lambda_{\text{config}}(c)$, $\kappa_p(c)$ | `verify_afc_problem_B_bounds.py` |
| Form-closure test | `threshold` $\ll 1$, `theory_mode=True` |
| Full AFC | `threshold = 1`, three projections in `check_wrench_space_sufficiency` |
| Force closure (2D) | `check_three_edge_force_closure` |
| Friction cone | `enable_tangent_forces=True`, `used_tangent_as_fallback=True` |
| $\lambda_{\text{shape}}$ (search estimate) | `find_the_magnum_stochastic` over $\mathbb{C}$ |
| Wrench covariance $M$, index $D$ | `grasp_covariance.calculate_grasp_covariance` |
| $D$, $\sigma_3$, screening | `grasp_covariance.calculate_grasp_covariance` — **use for friction gating** |
| Hardware actuator gate | `afc_hardware.check_robot_afc_hardware_feasible` |
| $\lambda_{\text{Floor}}$ (diagnostic only) | same; **do not use for gating** unless understood as weak hint |
| Exact LP certificate | replace hull-of-samples by feasibility (Section 1) per ellipse sample |

---

## 13. Shape classification (Problem B)

| Shape class | Force closure | Form closure ($T\ll1$) | AFC-xy | $\lambda_{\text{shape}}$ (normal-only) | Full AFC + friction |
|-------------|---------------|---------------------------|--------|------------------------------------------|---------------------|
| Typical polygon | yes | yes | yes | low ($\lesssim 2$) | yes |
| Star (5-point) | yes | yes | yes | $\approx 1.6$ ($D \approx 1.5$ — low $D$, not low $\lambda$) | yes at $\lambda \gtrsim \sqrt{2}$ |
| Frictionless circle | yes | **no** (code) | yes | **$\infty$** (strict) | yes |
| High-res near-circle | yes | varies | yes | huge (trap A) | yes |
| Needle / narrow / crescent | yes | varies | yes | huge (trap B) | yes |
| Mid-edge rectangle *placement* | yes | yes | yes | N/A — config failure, not shape | yes with better $c$ |

**Key separation:** the circle has $\lambda_{\text{shape}} = \infty$ (strict torque degeneracy, $\sigma_3 \to 0$). The rectangle has low $D$ and low $\lambda_{\text{shape}}$ but can exhibit $\lambda_{\text{config}} = \infty$ on symmetric mid-edge placements — a **configuration** issue. The star has low $D$ but $\lambda_{\text{shape}} \approx 1.6$ — a **$\kappa_p$ / $\lambda$** issue, not friction.

**Pipeline summary ($n=4$):** compute $M$, $\sigma_3$, $D$ once per shape. If $D \ge D_{\text{soft}}$ or $\sigma_3$ is strict $\Rightarrow$ friction at contacts ($\mu_{\text{contact}} \approx 0.2$–$0.3$ typically enough). If $D$ is modest $\Rightarrow$ normal-only at $\lambda = 2$ works for most shapes; use search failure (not $D$ alone) to decide whether to increase $\lambda$ or enable friction. Do not gate on $\lambda_{\text{Floor}}$ (Section 10).

For **three contacts**, see Section 14 — $D$ no longer predicts which shapes need a higher $\mu_{\text{contact}}$.

---

## 14. Three-contact vs four-contact friction requirements

`find_the_magnum_stochastic` now accepts `n_contacts` (default 4). Empirical verification (`scripts/test/test_markenscoff_form_closure.py`, July 2026) separates two regimes.

### 14.1 Form-closure proxy ($T \ll 1$)

Negligible ground-friction proxy: `threshold = 1e-3`, large $\lambda$ (e.g. 10).

| Experiment | Config | Result on 22 standard shapes |
|------------|--------|------------------------------|
| Markenscoff-4 | $n=4$, **frictionless** (normal-only) | 20 PASS; **circle** EXPECTED_FAIL ($\tau \equiv 0$); `narrow_triangle` timed out |
| Friction-3 | $n=3$, friction cone | **22/22 PASS** (including circle) |

Matches the classical picture: four frictionless contacts suffice for form closure on generic polygons; three contacts need $\mu_{\text{contact}} > 0$.

### 14.2 Full AFC ($T=1$) — $n=4$ engineering (existing Section 11)

| Shape screening | Contact mode | Typical $\mu_{\text{contact}}$ / $\lambda$ |
|-----------------|--------------|---------------------------------------------|
| $D \lesssim D_{\text{soft}}$ (well-behaved) | **Normal-only** | $\lambda = 2$ enough for most polygons |
| $D \ge D_{\text{soft}}$ or tiny $\sigma_3$ | **Friction required** | $\mu_{\text{contact}} \approx 0.2$–$0.3$ finds solutions |

$D$ gates **whether** to enable the friction cone for four contacts — not the exact $\mu$ needed.

### 14.3 Full AFC ($T=1$) — $n=3$ always with friction

Always enable tangent forces. Holding $\lambda = 2$ fixed and varying only $\mu_{\text{contact}}$:

| $\mu_{\text{contact}}$ | Pass rate | Failed shapes (then recovered at next $\mu$) |
|------------------------|-----------|-----------------------------------------------|
| $0.2$ | **13/22 (~59%)** | asym_l_shape, boot, l_shape, narrow_triangle, obese_triangle, plus, rectangle, t_shape, u_shape |
| $0.5$ | **9/9 retry = all pass** | (none remaining; $\mu = 0.8$ not needed) |

So for three contacts at engineering $T=1$, $\mu_{\text{contact}} = 0.2$ is **not** enough for $\sim 40\%$ of the standard set; $\mu_{\text{contact}} \ge 0.5$ recovered every failure in this suite.

### 14.4 Do the $n=3$ / $\mu = 0.2$ failures have high $D$?

**No.** Screening the nine failures vs the thirteen passes at $\mu = 0.2$ ($D_{\text{soft}} = 100$):

| Group | median $D$ | $D \ge D_{\text{soft}}$ | Classification |
|-------|------------|-------------------------|----------------|
| FAIL @ $\mu=0.2$ | **5.15** | **2/9** (narrow_triangle $D\!\approx\!192$, obese_triangle $D\!\approx\!326$) | **7/9 well_behaved** |
| PASS @ $\mu=0.2$ | **6.23** | 1/13 (circle $D\!\approx\!1038$) | mostly well_behaved |

Counterexamples to “high $D$ $\Rightarrow$ needs higher $\mu$ for $n=3$”:

- **circle** has the **largest** $D$ yet **passes** at $\mu = 0.2$ with three frictional contacts.
- **rectangle**, **plus**, **l_shape**, **t_shape**, etc. have **low** $D$ ($D \approx 2$–$8$) yet **fail** at $\mu = 0.2$ and succeed at $\mu = 0.5$.

**Conclusion:** $D$ (and the Section 11 gate) answers “does $n=4$ need friction vs normal-only?” It does **not** predict required $\mu_{\text{contact}}$ for **$n=3$ full AFC**. That shortfall is a **contact-count / friction-cone aperture** effect under the same $\lambda$: fewer torque generators needs a fatter cone (larger $\mu$) to envelope $\mathcal{E}(1)$.

### 14.5 Operational rule of thumb

| $n$ | Screening | Recommended contact mode for full AFC ($T=1$, $\lambda=2$) |
|-----|-----------|--------------------------------------------------------------|
| 4 | well_behaved | Normal-only first; friction at $\mu \approx 0.2$ if $D$-gate or search fails |
| 4 | soft / strict degenerate | Friction required; $\mu \approx 0.2$ typically enough |
| 3 | (always friction) | Use $\mu_{\text{contact}} \gtrsim 0.5$ as default; $\mu = 0.2$ only if you accept ~40% failures on the standard set |

Harness: `scripts/test/run_markenscoff_full_benchmark.py` writes CSVs under `/tmp/markenscoff_benchmark/`.
