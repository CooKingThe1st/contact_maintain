# Diff-Drive Flat Bumper: Feasibility Equations and I/O Spec

Companion to `test_matchingvelo_report.md` §9 and script `test_matchingvelo_segment.py`.

This document states **inputs → computation → outputs** for constant-twist segment pushing with a **fixed bumper patch** (no rim migration, \(\dot\alpha=0\)). It also lists where the current implementation matches or diverges from the math.

---

## 1. Frames and notation

| Symbol | Meaning |
|--------|---------|
| World frame | Inertial \(Oxy\) |
| Object body frame | Origin at object CoM; angle \(\theta(t)=\theta_0+\omega t\) |
| Robot body frame | Origin at diff-drive center; heading \(\zeta(t)=\zeta_0+\omega_r t\) |
| \(R(\psi)\) | 2D rotation by \(\psi\) |
| \(J\) | \(J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}\) (90° CCW) |

**Assumption (segment):** On one trajectory segment, object twist \((\mathbf{v}^b,\omega)\) is constant; robot commands \((v_r,\omega_r)\) are constant; bumper contact points on the robot body are fixed (\(\mathbf{r}_{E1}^b,\mathbf{r}_{E2}^b\)).

---

## 2. Inputs (what you must specify)

### 2.1 Object motion

| Input | Symbol | Description |
|-------|--------|-------------|
| Body translational velocity | \(\mathbf{v}^b=(v_x^b,v_y^b)\) | Constant on segment |
| Body angular rate | \(\omega\) | rad/s; \(\dot\theta=\omega\) |
| Initial object orientation | \(\theta_0\) | rad |
| Initial object CoM position | \(\mathbf{p}_o(0)\) | Often \((0,0)\) in tests |

### 2.2 Object contact geometry (two material points on one straight edge)

You need **two** object contact points in the **object body frame** (relative to CoM):

| Input | Symbol | Description |
|-------|--------|-------------|
| Contact point 1 | \(\mathbf{r}_{o1}^b\) | Material point on object boundary |
| Contact point 2 | \(\mathbf{r}_{o2}^b\) | Second point on the **same straight edge** |

On a straight edge with unit tangent \(\hat{\mathbf{t}}^b\) and outward normal \(\hat{\mathbf{n}}^b\):

\[
\mathbf{r}_{o2}^b - \mathbf{r}_{o1}^b = \ell\,\hat{\mathbf{t}}^b, \qquad
\hat{\mathbf{n}}^b \perp \hat{\mathbf{t}}^b.
\]

**Equivalent parameterization (used by the script):** Pick edge midpoint \(\mathbf{r}_c^b\), tangent \(\hat{\mathbf{t}}^b\), and half-span \(\ell/2\):

\[
\mathbf{r}_{o1}^b = \mathbf{r}_c^b - \tfrac{\ell}{2}\hat{\mathbf{t}}^b, \qquad
\mathbf{r}_{o2}^b = \mathbf{r}_c^b + \tfrac{\ell}{2}\hat{\mathbf{t}}^b.
\]

The script sets \(\ell = \|\mathbf{r}_{E2}^b-\mathbf{r}_{E1}^b\|\) (bumper chord length) and \(\mathbf{r}_c^b\) from boundary parameter \(t\) on the edge.

### 2.3 Robot design

| Input | Symbol | Description |
|-------|--------|-------------|
| Bumper endpoint 1 | \(\mathbf{r}_{E1}^b\) | Fixed in robot body frame |
| Bumper endpoint 2 | \(\mathbf{r}_{E2}^b\) | Fixed in robot body frame |
| Chord | \(\mathbf{w}^b := \mathbf{r}_{E1}^b-\mathbf{r}_{E2}^b\) | Non-zero |
| Actuator limits | \(v_{\max},\,\omega_{\max}\) | Optional |

Derived (robot-only, constant):

\[
\psi_i^b := \operatorname{atan2}(r_{Ei,y}^b,\, r_{Ei,x}^b), \qquad
\Delta\psi^b := \psi_2^b - \psi_1^b.
\]

### 2.4 Branch

| Input | Meaning |
|-------|---------|
| `forward` / `backward` | Sign of \(v_r\) (\(v_r>0\) vs \(v_r<0\)); backward uses \(\zeta_0 \leftarrow \zeta_0+\pi\). |

---

## 3. Object-side outputs at \(t=0\) (intermediate)

For each contact \(i\in\{1,2\}\):

**Contact velocity (body, constant on segment):**

\[
\mathbf{v}_{\text{cp},i}^b
= \mathbf{v}^b + \omega\, J\,\mathbf{r}_{oi}^b
= \mathbf{v}^b + \omega\begin{pmatrix}-r_{oi,y}^b\\ r_{oi,x}^b\end{pmatrix}.
\]

**Contact velocity (world, at \(t=0\)):**

\[
\boxed{
\mathbf{v}_{\text{cp},i}(0) = R(\theta_0)\,\mathbf{v}_{\text{cp},i}^b.
}
\]

**For all \(t\ge 0\)** (rigid rotation of the twist):

\[
\mathbf{v}_{\text{cp},i}(t) = R(\theta_0+\omega t)\,\mathbf{v}_{\text{cp},i}^b.
\]

---

## 4. Robot-side velocity model (diff-drive, fixed patch)

Robot center velocity: \(\mathbf{v}_{\text{base}} = v_r\, \mathbf{e}_\zeta\) with \(\mathbf{e}_\zeta=(\cos\zeta,\sin\zeta)^\top\), \(\dot\zeta=\omega_r\).

Velocity at bumper endpoint \(i\):

\[
\mathbf{v}_{r,i}
= v_r\,\mathbf{e}_\zeta + \omega_r\, J\,\mathbf{r}_{Ei}(t),
\qquad
\mathbf{r}_{Ei}(t)=R(\zeta(t))\,\mathbf{r}_{Ei}^b.
\]

**All-time matching** (same argument as disc, §5 of main report):

\[
\boxed{\omega_r = \omega.}
\]

At \(t=0\), write \(\mathbf{r}_{Ei}(0)=R(\zeta_0)\mathbf{r}_{Ei}^b\). Matching \(\mathbf{v}_{r,i}(0)=\mathbf{v}_{\text{cp},i}(0)\):

\[
v_r\cos\zeta_0 = a_i, \qquad v_r\sin\zeta_0 = b_i,
\]

with

\[
\boxed{
\begin{aligned}
a_i &:= v_{\text{cp},i,x}(0) + \omega\, r_{Ei,y}(0), \\
b_i &:= v_{\text{cp},i,y}(0) - \omega\, r_{Ei,x}(0).
\end{aligned}
}
\]

where \(r_{Ei,x}, r_{Ei,y}\) are components of \(\mathbf{r}_{Ei}(0)\).

---

## 5. Feasibility conditions (velocity-only, algebraic)

A **single** command triple \((v_r,\zeta_0,\omega_r=\omega)\) must satisfy **both** endpoints. Eliminating \(v_r,\zeta_0\):

### 5.1 Coupled scalar constraints

\[
\boxed{a_1 = a_2, \qquad b_1 = b_2.}
\]

### 5.2 Solve \(\zeta_0\) when \(\omega\neq 0\)

Let \(\mathbf{w}^b=\mathbf{r}_{E1}^b-\mathbf{r}_{E2}^b\), \(\Delta\mathbf{v}=\mathbf{v}_{\text{cp},1}(0)-\mathbf{v}_{\text{cp},2}(0)\). Then

\[
\omega\begin{pmatrix} w_x^b & w_y^b \\ -w_y^b & w_x^b \end{pmatrix}
\begin{pmatrix}\sin\zeta_0\\ \cos\zeta_0\end{pmatrix}
= \begin{pmatrix}\Delta v_x\\ \Delta v_y\end{pmatrix}.
\]

Solve the \(2\times 2\) linear system, then **normalize**: feasible only if \(\sin^2\zeta_0+\cos^2\zeta_0\approx 1\) (otherwise no real heading satisfies both endpoints).

### 5.3 Pure translation (\(\omega=0\))

Then \(\mathbf{v}_{\text{cp},1}^b=\mathbf{v}_{\text{cp},2}^b=\mathbf{v}^b\), so \(a_1=a_2\) and \(b_1=b_2\) hold automatically. Need \(\Delta\mathbf{v}=\mathbf{0}\). Heading:

\[
\zeta_0 = \operatorname{atan2}(v_{\text{cp},1,y}(0),\, v_{\text{cp},1,x}(0))
\quad\text{(or }+ \pi\text{ for backward)}.
\]

### 5.4 Commands once \(\zeta_0\) is known

\[
v_r = \pm\sqrt{a_1^2+b_1^2}, \qquad
\text{sign from forward/backward branch}.
\]

Check \(|v_r|\le v_{\max}\), \(|\omega|\le\omega_{\max}\).

---

## 6. Outputs (what the checker should return)

| Output | Symbol | Description |
|--------|--------|-------------|
| Feasible? | `yes`/`no` | All conditions satisfied |
| Initial heading | \(\zeta_0\) | rad |
| Wheel speed | \(v_r\) | m/s (constant on segment) |
| Yaw rate | \(\omega_r=\omega\) | rad/s |
| (Optional) Disc-style diagnostics | \(\alpha_{0,i}^{\text{disc}}\) | From §5 main report with \(\phi_0\) = inward normal and \(R_i=\|\mathbf{r}_{Ei}^b\|\); **not** the primary feasibility test |

**If feasible**, propagate for \(t\in[0,T]\):

- Object: \(\mathbf{p}_o(t)\), \(\theta(t)\), \(\mathbf{p}_{oi}(t)=R(\theta(t))\mathbf{r}_{oi}^b\).
- Robot: \(\mathbf{p}_r(t)\), \(\zeta(t)\), \(\mathbf{p}_{Ei}(t)=R(\zeta(t))\mathbf{r}_{Ei}^b\) (exact unicycle integration).

---

## 7. Position (geometry) — not implied by velocity alone

Velocity matching at E1 and E2 does **not** automatically keep robot endpoints on object endpoints unless **initial positions** are consistent.

**Required at \(t=0\):**

\[
\mathbf{p}_{E1}(0) = \mathbf{p}_{o1}(0), \qquad
\mathbf{p}_{E2}(0) = \mathbf{p}_{o2}(0).
\]

With one robot center \(\mathbf{p}_r(0)\) and heading \(\zeta_0\):

\[
\mathbf{p}_r(0) = \mathbf{p}_{o1}(0) - R(\zeta_0)\mathbf{r}_{E1}^b.
\]

**E2 consistency** (must hold in addition to velocity feasibility):

\[
R(\zeta_0)\,\mathbf{w}^b = R(\theta_0)\,(\mathbf{r}_{o2}^b-\mathbf{r}_{o1}^b).
\]

If bumper chord \(\mathbf{w}^b\) is parallel to object edge span in world frame and lengths match (\(\|\mathbf{w}^b\|=\ell\)), this reduces to **bumper tangent aligned with edge tangent** (plus correct placement along the edge). The script pins E1 at \(t=0\) but does **not** enforce E2 position unless geometry makes it so.

**If velocity feasible but E2 misaligned:** endpoint velocity errors stay ~0; **position error at E2 grows** (rigid slip along the edge).

---

## 8. Pipeline checklist (reference implementation)

```
INPUTS:
  v^b, omega, theta0, p_o(0)
  r_o1^b, r_o2^b  (or r_c^b, t_hat^b, ell)
  r_E1^b, r_E2^b
  branch, limits

STEP 1 — Object velocities at t=0:
  v_cp,i(0) = R(theta0) * (v^b + omega * J * r_oi^b)

STEP 2 — All-time rotation:
  omega_r = omega

STEP 3 — Heading (omega != 0):
  Solve M * [sin(zeta0); cos(zeta0)] = (v_cp,1(0) - v_cp,2(0)) / omega
  with M = [[w_x, w_y], [-w_y, w_x]], w = r_E1^b - r_E2^b
  Reject if ||[sin,cos]|| not ~ 1

STEP 3b — Heading (omega == 0):
  Require v_cp,1(0) ≈ v_cp,2(0)
  zeta0 = atan2(v_cp,1,y, v_cp,1,x)  (+ pi if backward)

STEP 4 — Verify a1=a2, b1=b2 with computed zeta0
  (redundant if step 3 exact; good numerical guard)

STEP 5 — Commands:
  v_r = +/- hypot(a1, b1)

STEP 6 — Limits and optional tangent-alignment check

STEP 7 — Position init (for simulation/plot):
  p_r(0) = p_o1(0) - R(zeta0) * r_E1^b
  Verify p_r(0) + R(zeta0)*r_E2^b ≈ p_o2(0)  [currently optional in script]

OUTPUT:
  feasible, zeta0, v_r, omega_r
```

---

## 9. Implementation audit (`test_matchingvelo_segment.py`)

| Item | Math | Implementation | Status |
|------|------|----------------|--------|
| \(\mathbf{v}_{\text{cp},i}^b\) | §3 | `v_cp_world_at` | OK |
| \(\omega_r=\omega\) | §4 | `omega_r = omega` in propagate | OK |
| \(a_i,b_i\) | §4 | `ab_from_zeta` | OK |
| \(a_1=a_2,b_1=b_2\) | §5.1 | `solve_zeta_two_endpoint` + verify | OK |
| \(\omega=0\) heading | §5.3 | `atan2(vcp1)` | OK |
| Forward/backward \(v_r\) | §5.4 | branch logic | OK |
| Object contacts from two explicit \(\mathbf{r}_{oi}^b\) | §2.2 | **Derived** from edge \(t\) + \(\ell=\|\mathbf{w}^b\|\) | **Simplification** — not general arbitrary \((\mathbf{r}_{o1},\mathbf{r}_{o2})\) |
| \(\ell\) along edge | \(\mathbf{r}_{o2}-\mathbf{r}_{o1}=\ell\hat t\) | Uses robot bumper length | OK if bumper ∥ edge; wrong if spans differ |
| Position E1 at \(t=0\) | §7 | `center0 = obj_ep1[0] - R(zeta0) r_E1^b` | OK |
| Position E2 at \(t=0\) | §7 | Not checked | **Gap** — can show 0 velocity error but nonzero position drift |
| Bumper ∥ edge | §7 | `--align-tol-deg` (default off) | Optional |
| Disc \(\alpha\) bands in scan table | diagnostic only | `disc_alpha_req` uses \(\phi_n\), \(R_i=\|\mathbf{r}_{Ei}\|\) | **Misleading label** — not bumper lever; do not use for yes/no |
| Propagation | exact unicycle | `propagate_bumper_dd_exact` | OK |
| Holonomic | always feasible (velocity) | message only | OK |

---

## 10. Suggested fixes (if tightening implementation)

1. **Accept explicit** `r_o1_b`, `r_o2_b` as inputs (not only edge-\(t\) + bumper length).
2. After solving \(\zeta_0\), **check E2 position** at \(t=0\); report `feasible_velocity` vs `feasible_position`.
3. Rename scan columns to `alpha_disc_*` so they are not confused with bumper feasibility.
4. When \(\omega\neq 0\), optionally require \(\|R(\zeta_0)\mathbf{w}^b - R(\theta_0)(\mathbf{r}_{o2}^b-\mathbf{r}_{o1}^b)\| < \varepsilon\) as part of `yes`.

---

## 11. Quick reference: vector form of feasibility (\(\omega\neq 0\))

Object difference (depends on placement along edge):

\[
\Delta\mathbf{v} = R(\theta_0)\,\omega\, J\,(\mathbf{r}_{o1}^b-\mathbf{r}_{o2}^b).
\]

Robot heading must satisfy:

\[
R(\zeta_0)\,\mathbf{w}^b = -\frac{1}{\omega}\,\Delta\mathbf{v}
\quad\text{(when the 2×2 system is consistent).}
\]

This is the “cannot rotate robot to fix two different \(\alpha_0\)” condition in vector form: \(\zeta_0\) is jointly determined by twist, both object points, and fixed \(\mathbf{w}^b\).
