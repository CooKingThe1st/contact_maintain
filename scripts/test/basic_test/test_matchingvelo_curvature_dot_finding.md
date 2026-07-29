# Finding: Velocity Matching Breaks When Curvature Changes ($\dot\kappa \neq 0$)
# + Code Audit: Continuous Fixed-$\alpha$ vs Obstructing “Relax”

**Status:** mathematical root-cause + implementation flaw note for correcting the Phase-7 / live-update diff-drive controller  
**Depends on:** `test_matchingvelo_report.md` (§5–6), `test_matchingvelo_relaxform_report.md` (§5–6), implementation in `test_multi_pusher_single_movement_diffdrive_liveupdate.py`  
**Claims:**
1. Critical contact failures follow from applying the **constant-twist, fixed-$\alpha$** closed form where $\dot\kappa\neq 0$ (or any mid-push $\alpha$ mismatch that needs re-alignment while still moving).
2. The obstructing / “passive” **relax** path in code fixes the watermelon-seed wedge by **intentionally breaking contact**, which destroys predictable SE(2) tracking.

---

## 1. What the matching theorem actually assumes

From `test_matchingvelo_report.md`:

1. Object body twist $(\mathbf{v}^b,\omega)$ is **constant** on the segment.
2. Contact material point $\mathbf{r}^b$ on the object is fixed.
3. Diff-drive uses **fixed** rim angle $\alpha$ ($\dot\alpha=0$), hence $\omega_r=\omega$.
4. There exists a **specific** $(\zeta_0,v_r)$ from Condition 2 at $t=0$ such that **constant** $(v_r,\omega_r)$ matches $\mathbf{v}_{\mathrm{cp}}$ for **all** $t$.

The all-time proof (§6) is a pure rigid-rotation identity:

$$
\mathbf{v}_{\mathrm{cp}}(t)=R(\omega t)\,\mathbf{v}_{\mathrm{cp}}(0),
\qquad
\mathbf{v}_{\mathrm{contact}}^{\mathrm{robot}}(t)=R(\omega_r t)\,\mathbf{v}_{\mathrm{contact}}^{\mathrm{robot}}(0),
$$

with $\omega_r=\omega$ and equality of the $t=0$ vectors. That factorization holds **if and only if** $\mathbf{v}_{\mathrm{cp}}^b$ is constant, which requires constant $(\mathbf{v}^b,\omega)$.

Under constant twist, the object CoM path is only of two types:

| Twist | CoM path | Path curvature $\kappa$ |
|-------|----------|-------------------------|
| $\omega=0$ | straight line | $\kappa\equiv 0$ |
| $\omega\neq 0$ | circular arc | $\kappa\equiv\mathrm{const}\neq 0$ |

So the theorem's geometric domain is exactly

$$
\boxed{\dot\kappa = 0 \quad\text{on the open segment.}}
$$

Piecewise line/arc planning (CSV `solve_constant_body_twist_from_SE2`) respects this **inside** each segment. Continuous $\kappa(t)$ changes, clothoids, pure-pursuit polylines, and **cross-track $\omega$ trims** do not.

---

## 2. Path curvature and body twist are the same degree of freedom

For a CoM path with unit tangent $\hat{\mathbf{t}}(s)$, signed curvature $\kappa(s)$, and speed $v=\dot s$:

$$
\omega_{\mathrm{path}}=\sigma\,v\,\kappa,\qquad \sigma=\pm 1.
$$

(Holds when heading is the path tangent; same identity appears in the holonomic path notes as $\omega_{\mathrm{path}}=\sigma v\kappa$.)

Differentiate at constant speed:

$$
\dot\omega_{\mathrm{path}}=\sigma\,v\,\dot\kappa.
$$

Hence:

$$
\boxed{\dot\kappa\neq 0 \;\Longleftrightarrow\; \dot\omega\neq 0
\quad\text{(constant $v$).}}
$$

Even with $v$ varying, $\dot\kappa$ injects an independent contribution into $\dot\omega$. Changing curvature is **time-varying object twist**, not a small perturbation of a constant-screw segment.

---

## 3. What happens to contact kinematics when $\dot\kappa\neq 0$

Keep $\mathbf{r}^b$ fixed. Write

$$
\mathbf{v}_{\mathrm{cp}}^b(t)=\mathbf{v}^b(t)+\omega(t)
\begin{pmatrix}-r_y^b\\ r_x^b\end{pmatrix},
\qquad
\mathbf{v}_{\mathrm{cp}}(t)=R(\theta(t))\,\mathbf{v}_{\mathrm{cp}}^b(t).
$$

Differentiate:

$$
\mathbf{a}_{\mathrm{cp}}
=
R(\theta)\Bigl(
\omega\,J\,\mathbf{v}_{\mathrm{cp}}^b
+\dot{\mathbf{v}}^b
+\dot\omega\,J\mathbf{r}^b
\Bigr),
\quad
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
$$

- If $\dot{\mathbf{v}}^b=\mathbf{0}$ and $\dot\omega=0$: only the $\omega J\mathbf{v}_{\mathrm{cp}}^b$ term remains $\Rightarrow$ $\mathbf{v}_{\mathrm{cp}}$ co-rotates at rate $\omega$ $\Rightarrow$ §6 identity.
- If $\dot\kappa\neq 0$ (hence typically $\dot\omega\neq 0$): **extra** $\dot\omega\,J\mathbf{r}^b$ (and possibly $\dot{\mathbf{v}}^b$) appear. Then

$$
\mathbf{v}_{\mathrm{cp}}(t)\;\neq\; R\Bigl(\int_0^t\omega\Bigr)\,\mathbf{v}_{\mathrm{cp}}(0)
$$

in general. The constant-$(v_r,\omega_r)$ robot side still rotates at a **single** rate $\omega_r$. No choice of fixed $\omega_r$ can match a contact velocity field whose rotation rate / body magnitude is itself changing.

**Immediate corollary:** there is **no** open interval with $\dot\kappa\neq 0$ on which the §5 table (constant $v_r$, $\omega_r=\omega$, fixed $\alpha$) is an all-time solution.

---

## 4. The fixed-$\alpha$ target itself becomes time-varying

Condition 2 (and `_init_segment_reference` in the live-update script) defines, at a frozen instant,

$$
a=v_{\mathrm{cp},x}+\omega R_r\sin\varphi,\qquad
b=v_{\mathrm{cp},y}-\omega R_r\cos\varphi,
$$

$$
\zeta^\star=\mathrm{atan2}(b,a)\quad\text{(or $+\pi$)},\qquad
\alpha^\star=\mathrm{wrap}(\varphi-\zeta^\star).
$$

### 4.1 Constant $\kappa$ (theorem regime)

On a true constant-twist segment with $\omega_r=\omega$ and fixed contact geometry:

$$
\dot\varphi=\omega,\qquad \dot\zeta^\star=\omega,\qquad \dot\alpha^\star=0.
$$

So $\alpha^\star$ is an **invariant** of the segment: re-solving later in the segment returns the **same** $\alpha^\star$ (up to branch choice). Live-resolve is then redundant for the feed-forward; it only absorbs geometric slip.

### 4.2 $\dot\kappa\neq 0$ (failure regime)

Geometry still forces the outward/inward normal to co-rotate with the object,

$$
\dot\varphi=\omega(t),
$$

but $(a,b)$ now depend on a **changing** $\omega(t)$ and $\mathbf{v}_{\mathrm{cp}}(t)$. The direction $\zeta^\star(t)=\mathrm{atan2}(b,a)$ therefore does **not** rotate at rate $\omega(t)$:

$$
\dot\zeta^\star(t)\neq\omega(t)
\quad\Rightarrow\quad
\boxed{\dot\alpha^\star(t)=\dot\varphi-\dot\zeta^\star\neq 0.}
$$

Interpretation:

- The **instantaneous** fixed-$\alpha$ solve still produces some $(\zeta^\star(t),\alpha^\star(t),v_r^\star(t))$ that would be correct **if** the current twist were held forever from that tick.
- Under $\dot\kappa\neq 0$, that “would-be forever” target **migrates in time**. The manifold of feasible fixed-$\alpha$ headings is moving.
- A physical rim contact cannot jump: $\alpha(t)$ is continuous. Chasing $\alpha^\star(t)$ requires $\dot\alpha\neq 0$, which contradicts the fixed-$\alpha$ premise $\omega_r=\omega$ of the same solve (relaxed balance: $\omega_r+\dot\alpha=\omega$).

So when curvature changes, the controller is asked to satisfy **two incompatible requirements at once**:

1. $\omega_r=\omega$ and $\dot\alpha=0$ (fixed-$\alpha$ matching theorem), and  
2. $\alpha(t)=\alpha^\star(t)$ with $\dot\alpha^\star\neq 0$ (live re-solve under $\dot\omega\neq 0$).

That is a **kinematic contradiction**, not a tuning issue.

---

## 5. How the live-update controller instantiates the contradiction

Relevant structure in `test_multi_pusher_single_movement_diffdrive_liveupdate.py`:

| Mechanism | Code / role | Effect under $\dot\kappa\neq 0$ |
|-----------|-------------|----------------------------------|
| Instantaneous CP velocity | `_compute_world_cp_velocity_ref` with current $\omega_{\mathrm{eff}}$ | Correct **snapshot** of $\mathbf{v}_{\mathrm{cp}}$, but ignores $\mathbf{a}_{\mathrm{cp}}$ |
| Fixed-$\alpha$ FF solve | `_init_segment_reference` $\Rightarrow$ `vr_ff`, `omega_ff=$\omega$`, `alpha_star` | Treats every tick as $t=0$ of a **new** constant-screw segment |
| Live-resolve every tick | PUSH loop re-solves geometry | Makes $\alpha^\star(t)$ track the drifting Condition-2 map |
| $\alpha$ / ref filters | `_smooth_live_segment_reference`, hysteresis | **Lags** a moving $\alpha^\star$; systematic lag $\Rightarrow$ sustained mismatch |
| Phase-7 yaw law | $\omega_r=\omega_{\mathrm{ff}}+K_{p,\alpha}e_\alpha+K_{d,\alpha}\dot e_\alpha+\omega_{\mathrm{tangent}}$ | To chase $\alpha^\star$, drives $\omega_r\neq\omega_{\mathrm{ff}}$, i.e. forced $\dot\alpha\neq 0$ |
| Cross-track trim | $\omega_{\mathrm{path}}=\omega_{\mathrm{ref}}+K_{\mathrm{ct}}e_d$ | **Intentionally** changes effective $\kappa$ ($v$ fixed $\Rightarrow\Delta\omega\Leftrightarrow\Delta\kappa$); injects the failure mode mid-segment |
| Segment joins | CSV line$\leftrightarrow$arc | $\kappa$ jump $\Rightarrow$ distributional $\dot\kappa$; worst-case $\alpha^\star$ jump |

Live-resolve was introduced to fight “stale reference drift” on a **constant** twist. On varying curvature it does the opposite: it **recomputes a target that is not invariant**, then feedback / filters fight that moving target by violating $\omega_r=\omega$.

Critical failures that look like “contact loss / $\alpha$ blow-up / omega thrash on arcs or transitions” are the visible symptoms of this inconsistency.

---

## 6. Straight vs curved vs transitioning (prediction table)

| Object CoM motion | $\kappa$ | Matching theorem | Live-resolve $\alpha^\star$ | Expected controller behavior |
|-------------------|----------|------------------|----------------------------|------------------------------|
| Pure translation | $0$ | Valid, $\omega_r=0$ | Constant (edge-wise) | Stable if contact force ok |
| Pure circular arc, fixed $(\mathbf{v}^b,\omega)$ | const | Valid | Constant | Stable **if** feedback does not spoil $\omega_r=\omega$ |
| Clothoid / varying-$\kappa$ path | $\dot\kappa\neq 0$ | **Invalid** | Time-varying | Structural mismatch; gains only change how failure appears |
| Line$\to$arc join (CSV) | jump | Valid only **inside** each piece | Jump at join | Failure clustered at transitions unless stop/re-align / teleport protocol |
| Cross-track $\omega$ trim on an arc | effective $\kappa$ changes | Breaks const-$\kappa$ assumption | Drifts with $e_d$ | Lateral correction **competes** with contact matching |

---

## 7. Correction directions (math $\rightarrow$ controller)

These are the implications of the finding; they define what a corrected controller must respect.

### 7.1 Do not use fixed-$\alpha$ open-loop primitive across $\dot\kappa\neq 0$

The §5 constant-$(v_r,\omega_r)$ command is a **segment primitive for constant $\kappa$ only**. Using it (even via per-tick re-solve) as a continuous law on a varying-$\kappa$ reference is invalid.

### 7.2 Keep planning in the theorem's class

Prefer object paths that are **piecewise** straight / circular (piecewise constant body twist), with an explicit **transition policy** at $\kappa$ discontinuities:

- stop-go + REALIGN (re-solve $\zeta_0,\alpha^\star$ once per piece), or  
- a short relaxed-$\alpha$ transient ($\omega_r+\dot\alpha=\omega$) to migrate rim angle between consecutive $\alpha^\star$ values, or  
- teleport / re-stage robots (already used experimentally).

### 7.3 If $\dot\kappa\neq 0$ must be executed, switch model

Use the **relaxed** balance from `test_matchingvelo_relaxform_report.md`:

$$
\omega_r+\dot\alpha=\omega(t),
$$

and plan / servo **both** $\omega_r(t)$ and $\alpha(t)$ (with singularity guard $\sin\alpha\neq 0$), together with the instantaneous $v_r$ solve

$$
v_r\,\mathbf{e}_\zeta + R_r\,\omega\,\mathbf{e}_{\varphi,\perp}=\mathbf{v}_{\mathrm{cp}}(t).
$$

Do **not** keep advertising $\alpha^\star$ as a fixed set-point from Condition 2 while $\omega(t)$ changes.

### 7.4 Cross-track / object servo must not silently rewrite $\kappa$

Any additive $\Delta\omega$ on $\omega_{\mathrm{ff}}$ changes effective curvature. Either:

- forbid mid-push $\omega$ trims when holding fixed $\alpha$, or  
- when a trim is applied, **recompute and accept** a new $\alpha$ migration (relaxed model), not a filtered chase of a new fixed-$\alpha$ $\alpha^\star$.

### 7.5 Live-resolve scope

Live re-solve of $(v_r^\star,\alpha^\star)$ is legitimate for:

- fixed contact slip / geometry update **under the same constant twist**, or  
- entering a **new** constant-$\kappa$ piece after a barrier.

It is **not** a substitute for a time-varying-$\kappa$ feed-forward.

---

## 8. One-line root cause

$$
\boxed{\text{Fixed-}\alpha\text{ matching} \;\Leftrightarrow\; \dot\kappa=0;\quad
\dot\kappa\neq 0 \;\Rightarrow\; \dot\alpha^\star\neq 0 \;\Rightarrow\;
\text{live Phase-7 fixed-}\alpha\text{ law is kinematically inconsistent.}}
$$

Controller correction must restore that equivalence (piecewise constant-$\kappa$ + transitions), or abandon fixed $\alpha$ on intervals where $\dot\kappa\neq 0$.

---

## 9. Continuous controller needs stop-and-realign (structural)

Even **inside** a valid constant-$\kappa$ segment, the fixed-$\alpha$ primitive is fragile:

- Feasibility requires a **specific** heading manifold $\zeta\approx\zeta_0$ with $\alpha=\alpha^\star$ and $\omega_r=\omega$.
- Any perturbation (slip, neighbor push, cross-track $\Delta\omega$, live-solve lag, bumper impact) produces $\mathrm{e}_\alpha\neq 0$.
- Phase 7 then runs $\omega_r=\omega+\cdots$ **while still commanding** $v_r\approx v_r^{\mathrm{ff}}$ — i.e. it tries to **self-align and velocity-match at the same time**.

Under the theorem, those are **sequenced** phases in the script (`REALIGN` with $v_r=0$, then `PUSH`). There is **no** mid-push barrier that returns to REALIGN. So a single mismatch forces the continuous law into the regime of §4.2 / relaxform $\dot\alpha\neq 0$, which the feed-forward still pretends is fixed-$\alpha$.

**Operational meaning:** a continuous fixed-$\alpha$ controller is not “wrong gains”; it is the wrong **mode machine** for recovery. Recovery while moving requires either:

- an explicit **moving realign** law (relaxed $\omega_r+\dot\alpha=\omega$ with planned $\alpha$ migration), or  
- a **stop / barrier / teleport** and re-enter REALIGN.

Live-resolve alone cannot replace that (it only refreshes $\alpha^\star$ under the old fixed-$\alpha$ assumption).

---

## 10. Watermelon seed vs obstructing “relax” (the hidden second failure)

### 10.1 Why four-contact AFC wedges (seed effect)

Magnum-four form closure can resist wrenches in-plane, but stacked **inward normal pressures** from all four discs against a polygonal object make a 3D wedge: friction + over-compression lift the object in $z$ (“watermelon seed”). That is a **contact mechanics** failure of “everyone always presses inward,” not a matching-theory failure.

Two escapes exist in the codebase:

| Escape | Intent | Mechanism in code |
|--------|--------|-------------------|
| **A. Keep all contacts, soften inward** | stay in contact, reduce squeeze | `compression_relax_*`, `z_relax_*` scale down inward $v_r^{\mathrm{ff}}$ |
| **B. Tag “obstructing / passive” and clear** | remove the side of the clamp that opposes motion | `obstructing_passive_ratio` + `obstructing_pusher_speed_scale` + inflate gap / clearance cheat |

Escape A is incomplete (local scalar on $v_r^{\mathrm{ff}}$; does not rebalance multi-contact wrenches). Escape B is what was used in practice to kill the seed — and it is where tracking dies.

### 10.2 What the code actually classifies as “passive”

At **first realign tick only** (`_seg_refs[i] is None`):

```text
move_dir  = sign(vr_ff) * (cos ζ0, sin ζ0)
ρ_normal  = n_in · move_dir          # CLI text: “cos(true_alpha)”
obstructing ⇔  ρ_normal < −|ρ_passive|   # default ρ_passive = 0.1
```

Flaws in this classifier:

1. **Wrong quantity.** Script TODO / methodology talk about **near-zero required force** for this twist (wrench redundancy). The code uses a **kinematic angle** between inward normal and the solved drive direction. That is “is the bumper advancing **outward** along the normal?”, not “is this contact force-null for the twist.”
2. **Frozen once.** `_obstructing_pushers[i]` is never recomputed in PUSH. Role does not track live force, live $\rho$, or segment changes of $(\mathbf{v}^b,\omega)$.
3. **Misses the truly idle contacts.** Contacts with $\rho\approx 0$ (tangential / non-contributing) stay role=`normal` and can keep pressing $\Rightarrow$ seed remains for the ambiguous ones; only strongly **outward** drives are tagged.
4. **Name collision.** Printed `true_alpha = arccos(ρ_normal)` is **not** the rim contact angle $\alpha=\varphi-\zeta$ used by matching. Debugging mixes two different “alpha” notions.

### 10.3 What “relax” then does to those robots (and why tracking fails)

Once tagged obstructing, PUSH applies (defaults in examples: `obstructing_pusher_speed_scale > 1`, `obstructing_inflate_gap > 0`):

1. **Inflated coupling target**  
   `couple_target = intended + inflate_gap * n_out`  
   Ring coupling (`k_couple`, and often `couple_obstructing_only=True`) servos the robot toward a **deliberate standoff**, not contact.

2. **Clearance override on $v_r$** (`use_actual_contact_clearance_cheat`, default on)  
   Forces robot normal speed to stay more outward than measured object CP normal speed by a margin. Comment in code: simulation-only cheat so the robot **clears** instead of staying wedged.

3. **Else** (cheat off): blunt $v_r \leftarrow v_r\cdot\mathrm{scale}$, which also breaks the §5 matched $v_r$ at that patch.

Net kinematic effect:

$$
\text{tagged robot} \;\not\Rightarrow\; \mathbf{v}_{\mathrm{cp}}^{\mathrm{robot}}=\mathbf{v}_{\mathrm{cp}}^{\mathrm{obj}}
\quad\text{and typically leaves the contact set.}
$$

Then:

- AFC / form closure is **no longer** the Magnum-four configuration the planner assumed.
- Object SE(2) is driven by a **varying subset** of contacts (which side cleared, when gap closes again, friction switching).
- Active robots still run the full matching FF as if the twist were realized by the original four-contact model $\Rightarrow$ **unpredictable** $(\mathbf{v},\omega)$ and tracking collapse.

So the hidden flaw is not “relax gain too high.” It is:

$$
\boxed{\text{seed fix = break contact on “passive” side} \;\Rightarrow\;
\text{lose the constraint set that makes motion deterministic.}}
$$

Escape A (compression / $z$ relax) shares a milder form of the same bug: it **detunes** matched $v_r^{\mathrm{ff}}$ without a substitute wrench plan, so it can also spoil tracking — but it at least tries to stay in contact.

### 10.4 Coupling asymmetry amplifies unpredictability

With `couple_obstructing_only=True` (default unless `--couple-all-robots`):

- Obstructing robots: coupled to inflated gap (stay away).
- Active robots: **no** ring coupling (not disturbed).

Formation is therefore one-sided. Neighbor gap information never equalizes pressure on the pushing side; yaw/sideslip of the object changes who is “ahead,” but roles stay frozen $\Rightarrow$ sustained open-loop conflict between clearance and matching.

---

## 11. Two approaches in conflict (choose one; do not mix as today)

Current pipeline **mixes** both philosophies:

| Approach | Goal | Valid when | Current code |
|----------|------|------------|--------------|
| **① Fixed-$\alpha$ continuous match (all contacts)** | Exact $\mathbf{v}_{\mathrm{cp}}$ match; predictable screw | Constant $\kappa$; all (or planned) contacts stay closed; mismatches handled by **barrier realign**, not mid-push chase | PUSH live-resolve + $\alpha$ PD while moving; no mid-push REALIGN |
| **② Role-relax / clear passives** | Avoid $z$-wedge / opposing normals | Accept **fewer** contacts and a **re-solved** twist/wrench model for the active set | Classifies by frozen $\rho_{\mathrm{normal}}$; clears contact; still commands the **full-swarm** matching FF |

Mixing them yields the worst of both: continuous law needs closed contacts and $\dot\alpha=0$, while relax **opens** contacts and injects $\omega_r\neq\omega$ / $v_r$ overrides.

### Candidate resolutions (for the next design step)

**Option A — Commit to contact maintenance (AFC-style)**  
- Keep (or gentle-press) all four contacts; kill seed with **force/impedance** limits or scheduled normal pressure, not clearance.  
- Constant-$\kappa$ segments only; on $e_\alpha$ or $\dot\kappa$ events: **stop-go / REALIGN** (or teleport) instead of chasing $\alpha$ at speed.  
- Drop obstructing clearance cheat / inflate gap for tracking runs.

**Option B — Commit to active-subset pushing**  
- Explicitly select active contacts for the twist (planner or online wrench null-space).  
- **Re-plan** $(\mathbf{v}^b,\omega)$ and matching FF for the **active set only**.  
- Passives: hold a **controlled** standoff with a dedicated mode (not Phase-7 matching), and **update roles online** (force / wrench residual — not frozen $\rho$ at realign).  
- Accept that form closure may be temporarily abandoned; tracking uses the reduced contact model on purpose.

**Do not** keep Phase-7 matching on a robot while a clearance law tells it to leave the patch — that is the flaw denoted here.

---

## 12. Code flaw checklist (point to lines / symbols)

| # | Flaw | Where | Effect |
|---|------|-------|--------|
| F1 | Fixed-$\alpha$ FF used under $\dot\kappa\neq 0$ / changing $\omega_{\mathrm{eff}}$ | live `_init_segment_reference` every tick; cross-track $\omega$ trim | $\dot\alpha^\star\neq 0$; theorem void (§1–8) |
| F2 | No mid-push REALIGN; recovery via $\omega_{\alpha},\omega_t$ while $v_r$ pushed | `_compute_phase7_command` + PUSH branch | Self-align and match cannot coexist (§9) |
| F3 | “Passive” = frozen kinematic $\rho_{\mathrm{normal}}$, not force/wrench | realign-start block setting `_obstructing_pushers` | Wrong robots cleared; roles stale (§10.2) |
| F4 | Obstructing path **commands non-contact** | `obstructing_inflate_gap`, clearance cheat / `speed_scale` | Breaks AFC ⇒ unpredictable object motion (§10.3) |
| F5 | Coupling only on obstructors toward standoff | `couple_obstructing_only` + inflated target | Asymmetric formation; tracking fights clearance (§10.4) |
| F6 | Escape-A softens matched $v_r^{\mathrm{ff}}$ with no wrench substitute | `compression_relax_*`, `z_relax_*` | Local anti-seed can still spoil velocity match |
| F7 | Philosophy mix: full matching + clear passives | PUSH applies Phase 7 **then** clearance override | Deterministic matching assumptions already false when override fires (§11) |

---

## 13. One-line summary (both roots)

$$
\boxed{
\begin{aligned}
&(1)\;\text{continuous fixed-}\alpha\text{ needs }\#\text{motion re-align after mismatch — law does not;}\\
&(2)\;\text{obstructing “relax” kills the seed by opening contacts — and with them, predictable tracking.}
\end{aligned}
}
$$

Next step: pick **Option A** or **Option B** in §11 and remove the other path from the push loop rather than tuning both.
