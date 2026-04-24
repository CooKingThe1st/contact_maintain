# Relaxed Velocity-Matching Model for Diff-Drive

## 1. Problem Statement

Given an object under all-face contact (AFC) with a circular diff-drive robot of radius $R_r$, we keep the object-side contact point fixed in the object body frame, but allow the touched point on the robot boundary to vary over time.

- **Fixed on object:** contact material point $\mathbf r_o^b$ is constant.
- **Variable on robot:** contact angle on robot body is $\alpha(t)$, not constant.

The object moves under constant body-frame twist $(\mathbf v^b,\omega)$. We seek robot trajectories/commands that satisfy contact-point velocity matching for all $t$, under this relaxed contact model.

## 2. Object Kinematics (Unchanged)

Let $\theta(t)=\theta_0+\omega t$, and $R(\theta)\in SO(2)$.

$$
\mathbf p_{cp}(t)=\mathbf p_o(t)+R(\theta(t))\,\mathbf r_o^b
$$

$$
\mathbf v_{cp}(t)=R(\theta(t))\,\mathbf v^b+\omega\,R(\theta(t))
\begin{bmatrix}
-r_{o,y}^b\\
r_{o,x}^b
\end{bmatrix}
$$

Define the constant body-frame contact velocity:

$$
\mathbf v_{cp}^b=\mathbf v^b+\omega
\begin{bmatrix}
-r_{o,y}^b\\
r_{o,x}^b
\end{bmatrix},
\qquad
\mathbf v_{cp}(t)=R(\theta(t))\,\mathbf v_{cp}^b.
$$

So the object-side contact velocity is known for all $t$.

## 3. Relaxed Contact Geometry

Let $\zeta(t)$ be robot heading. Define

$$
\phi(t)=\zeta(t)+\alpha(t),
$$

where $\phi$ is the world-frame angle from robot center to contact point, and $\alpha(t)$ is the robot-local contact angle.

Then robot center is

$$
\mathbf p_r(t)=\mathbf p_{cp}(t)-R_r
\begin{bmatrix}
\cos\phi(t)\\
\sin\phi(t)
\end{bmatrix}.
$$

This enforces geometric contact with the same world contact point.

## 4. Robot Contact-Point Velocity with Variable $\alpha(t)$

Robot base velocity:

$$
\mathbf v_{\text{base}}=
v_r
\begin{bmatrix}
\cos\zeta\\
\sin\zeta
\end{bmatrix}.
$$

Offset from robot center to contact point:

$$
\mathbf r_{rc}=R_r
\begin{bmatrix}
\cos\phi\\
\sin\phi
\end{bmatrix}.
$$

Since $\dot\phi=\dot\zeta+\dot\alpha=\omega_r+\dot\alpha$,

$$
\dot{\mathbf r}_{rc}
=
R_r(\omega_r+\dot\alpha)
\begin{bmatrix}
-\sin\phi\\
\cos\phi
\end{bmatrix}.
$$

Hence robot-side contact-point velocity:

$$
\mathbf v_{cp}^{\text{robot}}
=
v_r
\begin{bmatrix}
\cos\zeta\\
\sin\zeta
\end{bmatrix}
+
R_r(\omega_r+\dot\alpha)
\begin{bmatrix}
-\sin\phi\\
\cos\phi
\end{bmatrix}.
$$

Define effective spin

$$
u:=\omega_r+\dot\alpha.
$$

Then

$$
\mathbf v_{cp}^{\text{robot}}
=
v_r\,\mathbf e_\zeta + R_r u\,\mathbf e_{\phi,\perp},
$$

with

$$
\mathbf e_\zeta=
\begin{bmatrix}
\cos\zeta\\
\sin\zeta
\end{bmatrix},
\qquad
\mathbf e_{\phi,\perp}=
\begin{bmatrix}
-\sin\phi\\
\cos\phi
\end{bmatrix}.
$$

## 5. Velocity-Matching Constraint (Relaxed)

Require

$$
\mathbf v_{cp}^{\text{robot}}(t)=\mathbf v_{cp}(t),\qquad \forall t.
$$

So at each instant:

$$
v_r\,\mathbf e_\zeta + R_r u\,\mathbf e_{\phi,\perp}=\mathbf v_{cp}.
$$

To connect this to an all-time relation (as in the fixed-$\alpha$ report), use the contact geometry with the object normal. The contact direction $\phi$ is the inward-normal direction at the chosen object material point, and that normal co-rotates with the object at rate $\omega$. Therefore

$$
\dot\phi=\omega.
$$

But from Section 4,

$$
\dot\phi=\dot\zeta+\dot\alpha=\omega_r+\dot\alpha=u.
$$

Hence the relaxed model gives the global kinematic identity

$$
u=\omega,\qquad\text{equivalently}\qquad \omega_r+\dot\alpha=\omega.
$$

So $u$ is **not** a new actuator input; it is a derived effective spin rate of the contact radius vector, fixed by geometry once the object motion is fixed.

With this, velocity matching can be written directly as

$$
v_r\,\mathbf e_\zeta + R_r \omega\,\mathbf e_{\phi,\perp}=\mathbf v_{cp},
$$

which solves for the instantaneous feasible $v_r$ (and implied $\alpha$ evolution through $\dot\alpha=\omega-\omega_r$).

For completeness, if we keep $(v_r,u)$ as algebraic unknowns, the decomposition matrix

$$
A(t)=
\begin{bmatrix}
\mathbf e_\zeta & R_r\mathbf e_{\phi,\perp}
\end{bmatrix}
$$

is full-rank iff

$$
\det(A)=R_r\sin(\phi-\zeta)=R_r\sin\alpha\neq 0.
$$

So $\alpha\notin\{0,\pi\}$ (mod $2\pi$) avoids the local singular case.

## 6. What Changes vs. Fixed-$\alpha$ Model

The all-time comparison is:

- **Fixed-$\alpha$ model:** $\dot\alpha=0$, so from $\omega_r+\dot\alpha=\omega$ we recover

$$
\omega_r=\omega.
$$

- **Relaxed model:** $\alpha(t)$ is free to evolve, so the required all-time coupling becomes

$$
\omega_r+\dot\alpha=\omega.
$$

Therefore the strong equality moves from $\omega_r=\omega$ to a balance law between robot yaw rate and contact-angle migration rate. The diff-drive remains nonholonomic (translation still constrained to $v_r\mathbf e_\zeta$), but one internal contact-mode variable $\alpha$ absorbs part of the rotational requirement.

### 6.1 Explicit reduction when $\dot\alpha=0$ (must match fixed-$\alpha$ report)

Start from Section 5 relaxed matching:

$$
v_r\,\mathbf e_\zeta + R_r(\omega_r+\dot\alpha)\,\mathbf e_{\phi,\perp}=\mathbf v_{cp}.
$$

For fixed contact mode on the robot boundary, impose:

$$
\dot\alpha=0.
$$

Then

$$
v_r\,\mathbf e_\zeta + R_r\omega_r\,\mathbf e_{\phi,\perp}=\mathbf v_{cp}.
$$

From the geometric identity in Section 5:

$$
\omega_r+\dot\alpha=\omega
\;\Rightarrow\;
\omega_r=\omega.
$$

Substitute into the matching equation:

$$
v_r\,\mathbf e_\zeta + R_r\omega\,\mathbf e_{\phi,\perp}=\mathbf v_{cp}.
$$

At $t=0$ (using $\phi_0,\zeta_0$):

$$
\begin{bmatrix}
v_r\cos\zeta_0\\
v_r\sin\zeta_0
\end{bmatrix}
=
\mathbf v_{cp}(0)
-\omega
\begin{bmatrix}
-R_r\sin\phi_0\\
R_r\cos\phi_0
\end{bmatrix}.
$$

Define

$$
a:=v_{cp,x}(0)+\omega R_r\sin\phi_0,\qquad
b:=v_{cp,y}(0)-\omega R_r\cos\phi_0.
$$

Then

$$
v_r\cos\zeta_0=a,\qquad v_r\sin\zeta_0=b,
$$

so

$$
\zeta_0=\operatorname{atan2}(b,a)\ \text{or}\ \operatorname{atan2}(b,a)+\pi,\qquad
v_r=\pm\sqrt{a^2+b^2}.
$$

This is exactly the same fixed-$\alpha$ result reported in
`test_matchingvelo_report.md` (Section 5).

#### Why this also implies $\dot v_r=0$

From Section 2:

$$
\mathbf v_{cp}(t)=R(\theta(t))\,\mathbf v_{cp}^b,
$$

and $\mathbf v_{cp}^b$ is constant on the segment. So in world frame,
$\mathbf v_{cp}(t)$ is just a rigid rotation at rate $\omega$.

In the fixed-$\alpha$ case, we already have:

$$
\dot\alpha=0,\qquad \omega_r=\omega.
$$

Define fixed vectors at segment start:

$$
\mathbf e_{\zeta,0}:=
\begin{bmatrix}\cos\zeta_0\\ \sin\zeta_0\end{bmatrix},
\qquad
\mathbf e_{\phi,\perp,0}:=
\begin{bmatrix}-\sin\phi_0\\ \cos\phi_0\end{bmatrix}.
$$

Because $\omega_r=\omega$ and $\dot\alpha=0$, both $\zeta(t)$ and $\phi(t)$ rotate
at rate $\omega$, so

$$
\mathbf e_\zeta(t)=R(\omega t)\mathbf e_{\zeta,0},
\qquad
\mathbf e_{\phi,\perp}(t)=R(\omega t)\mathbf e_{\phi,\perp,0}.
$$

Also from Section 2:

$$
\mathbf v_{cp}(t)=R(\omega t)\mathbf v_{cp}(0).
$$

Substitute these into the matching equation:

$$
v_r(t)\,R(\omega t)\mathbf e_{\zeta,0}
+R_r\omega\,R(\omega t)\mathbf e_{\phi,\perp,0}
=R(\omega t)\mathbf v_{cp}(0).
$$

Left-multiply by $R(-\omega t)$:

$$
v_r(t)\,\mathbf e_{\zeta,0}
+R_r\omega\,\mathbf e_{\phi,\perp,0}
=\mathbf v_{cp}(0).
$$

The right-hand side and basis vectors are constant on the segment, therefore
$v_r(t)$ is constant:

$$
v_r(t)\equiv v_r,\qquad \dot v_r=0.
$$

So the fixed-$\alpha$ primitive is not only $\omega_r=\omega$ but also constant
forward speed on each constant-twist segment.

**Base form vs.\ special case.** The relaxed model ($\dot\alpha$ free) is the **general kinematic form**. The fixed-$\alpha$ analysis in `test_matchingvelo_report.md` is the **special case** $\dot\alpha=0$. In that special case, the balance $\omega_r+\dot\alpha=\omega$ collapses to $\omega_r=\omega$, and one can additionally derive a **specific initial heading** (and forward/backward $v_r$) so that **constant** $(v_r,\omega_r)$ matches velocity for the whole segment—giving very simple, open-loop-feasible commands.

**Whole horizon under relaxation.** With $\dot\alpha\neq 0$, feasibility over the horizon is no longer “one heading at $t=0$ and hold forever.” It becomes: choose $\omega_r(t)$ (and thus $\dot\alpha(t)=\omega-\omega_r(t)$) such that $\alpha(t)$ stays in a feasible range, actuator limits hold, and singularities ($\alpha\to 0,\pi$) are avoided—or handled explicitly.

**Fast self-rotation as a gate.** Large $|\omega_r|$ capability helps in **both** regimes: (i) acquiring the fixed-$\alpha$ heading manifold quickly at contact, and (ii) in the relaxed regime, keeping $|\dot\alpha|$ small when $\omega$ is large by choosing $\omega_r\approx\omega$ so $\dot\alpha$ stays a small transient. It is a natural **feasibility gate** near the end of the analytical phase: if yaw authority is too weak, the planner may require excessive $|\dot\alpha|$ (rim migration / slip) to satisfy $\omega_r+\dot\alpha=\omega$.

**Stable vs.\ changing $\dot\alpha$ in practice.** For robustness and friction, **small, smooth $\dot\alpha$** (often driving $\dot\alpha\to 0$ in steady contact) is usually preferable: less rim migration, less demand on tangential contact mechanics, simpler behavior for a diff-drive. Treating **nonzero $\dot\alpha$ as a transient** (approach mismatch, disturbance) matches “slightly wrong initial heading, then self-rotate and settle.” Aggressive $\dot\alpha$ can be feasible kinematically but is often worse for stability and contact modeling.

## 7. Fast Self-Rotation Interpretation

If the diff-drive can self-rotate quickly (bounded but large $|\omega_r|$), then:

1. Use the global identity $\omega_r(t)+\dot\alpha(t)=\omega$ (so $\dot\alpha(t)=\omega-\omega_r(t)$).
2. Choose $\omega_r(t)$ according to heading recovery, singularity avoidance, and actuator bounds.
3. Solve the instantaneous $v_r$ from $v_r\mathbf e_\zeta+R_r\omega\,\mathbf e_{\phi,\perp}=\mathbf v_{cp}$ whenever $\sin\alpha\neq 0$.

Fast yaw authority trades directly against how much contact migration $\dot\alpha$ is required for a given $\omega$.

## 8. Feasibility Constraints

For physically meaningful solutions, enforce

$$
|v_r|\le v_{\max},\qquad
|\omega_r|\le \omega_{\max},\qquad
|\dot\alpha|\le \dot\alpha_{\max},
$$

and optionally

$$
|u|\le u_{\max}.
$$

Large $|\dot\alpha|$ means rapid migration of contact point along the robot rim (rolling/sliding mode). If your contact/friction model does not permit that, $\dot\alpha$ must be tightly bounded.

## 9. Commandable Inputs vs.\ Internal Degrees of Freedom

The kinematics in Section 4 use the **actual diff-drive controls** $(v_r,\omega_r)$: translation is only along $\mathbf e_\zeta$, and yaw is $\dot\zeta=\omega_r$. The quantity $\dot\alpha$ is **not** an extra motor input; it is the rate at which the contact point **moves along the robot rim**, induced whenever $\omega_r\neq\omega$ (since $\dot\alpha=\omega-\omega_r$ under the geometric $\dot\phi=\omega$ assumption).

One motivation for the **fixed-$\alpha$** motion primitive in `test_matchingvelo_report.md` was planning simplicity: **constant** $(v_r,\omega_r)$ within limits, with a **specific initial heading** so velocity matching holds for the whole segment without internal contact state. The relaxed model keeps the same **two** actuator inputs but adds **one internal configuration** $\alpha(t)$; horizon feasibility must explicitly check that chosen $\omega_r(t)$ and implied $v_r(t)$ remain commandable and within limits.

## 10. Initial Heading: Arbitrary or Not?

At contact, the **approach direction** $\phi_0$ toward the object contact point is fixed by geometry (inward normal). The identity $\phi=\zeta+\alpha$ gives $\alpha_0=\phi_0-\zeta_0$ (mod $2\pi$). So for a given physical contact, **you cannot independently pick $\zeta_0$ and $\alpha_0$**; choosing one determines the other.

What relaxation buys is **not** “any heading with no consequence,” but **flexibility over time**: if the robot arrives with a heading error relative to the fixed-$\alpha$ primitive, you may **transiently** use $\dot\alpha\neq 0$ (and $\omega_r\neq\omega$) while still matching $\mathbf v_{cp}$, then regulate toward a desired pair $(\zeta,\alpha)$—for example $\dot\alpha\to 0$.

Whether $\zeta_0$ can be “arbitrary” depends on **feasibility at $t=0$**: you need $\sin\alpha_0\neq 0$ for the $2\times 2$ solve for $v_r$ to be well-posed, and you need $(v_r(0),\omega_r(0))$ within limits. So initial heading is constrained by **singularity avoidance** and **actuators**, not by kinematics alone.

## 11. Singularity $\alpha=0$ or $\pi$: $\det(A)=0$

When $\alpha\in\{0,\pi\}$ (mod $2\pi$), $\phi=\zeta+\alpha$ implies

$$
\mathbf e_{\phi,\perp}=\pm \mathbf e_{\zeta,\perp},
$$

so $\mathbf e_\zeta$ and $\mathbf e_{\phi,\perp}$ become linearly dependent, and

$$
\det(A)=R_r\sin\alpha=0.
$$

Using the all-time relation $u=\omega$, the matching equation

$$
v_r\mathbf e_\zeta + R_r\omega\,\mathbf e_{\phi,\perp}=\mathbf v_{cp}
$$

reduces at singularity to

$$
\mathbf v_{cp}=v_r\mathbf e_\zeta \pm R_r\omega\,\mathbf e_{\zeta,\perp}.
$$

Project onto $\mathbf e_\zeta$ and $\mathbf e_{\zeta,\perp}$:

$$
\mathbf e_\zeta^\top\mathbf v_{cp}=v_r,\qquad
\mathbf e_{\zeta,\perp}^\top\mathbf v_{cp}=\pm R_r\omega.
$$

The second equation is the **hard compatibility condition** at singularity.

- If $\mathbf e_{\zeta,\perp}^\top\mathbf v_{cp}=\pm R_r\omega$ holds, matching is compatible at that instant.
- If $\mathbf e_{\zeta,\perp}^\top\mathbf v_{cp}\neq \pm R_r\omega$, there is **no** solution that preserves contact-velocity equality.

Concrete example (the “perpendicular motion” intuition): let $\zeta=0$, so $\mathbf e_{\zeta,\perp}=[0,1]^\top$. If $R_r=0.06$ and $\omega=0.3$, then the compatibility requires

$$
\mathbf e_{\zeta,\perp}^\top\mathbf v_{cp}=v_{cp,y}=\pm(0.06)(0.3)=\pm 0.018\ \text{m/s}.
$$

If the object demands $\mathbf v_{cp}=[0,\ 0.05]^\top$, then $v_{cp,y}=0.05\neq \pm0.018$, so matching is infeasible at singularity: the required perpendicular component is outside what the diff-drive/contact geometry can realize there.

This also shows why singularity constrains object motion. Substituting

$$
\mathbf v_{cp}
=
R(\theta)\!\left(\mathbf v^b+\omega
\begin{bmatrix}
-r_{o,y}^b\\
r_{o,x}^b
\end{bmatrix}\right)
$$

into the hard condition gives

$$
\mathbf e_{\zeta,\perp}^\top
R(\theta)\!\left(\mathbf v^b+\omega
\begin{bmatrix}
-r_{o,y}^b\\
r_{o,x}^b
\end{bmatrix}\right)
=
\pm R_r\omega.
$$

So at $\alpha=0,\pi$, $(\mathbf v^b,\omega)$ and the chosen object contact point $\mathbf r_o^b$ are no longer freely selectable: they must satisfy this compatibility relation. This is a strong reason to keep nominal operation away from singularity and prefer the fixed-$\alpha$ primitive ($\dot\alpha=0$) when possible.

## 12. First Contact, “Locked” Pose, and Approach Heading

After contact is established, the robot is **geometrically** tied to $\mathbf p_{cp}$ and $R_r$; **kinematically**, a diff-drive still has only $(v_r,\omega_r)$. There is no third independent actuator for $\alpha$; only **$\dot\alpha=\omega-\omega_r$** (under $\dot\phi=\omega$).

The **non-intrusive** ideal (constant contact mode on the robot rim, no tangential migration) corresponds to **$\dot\alpha=0$**, hence $\omega_r=\omega$ and the fixed-$\alpha$ picture. If the robot approaches with a **heading error** relative to that primitive, you generally **cannot** both match $\mathbf v_{cp}$ and keep $\dot\alpha=0$ unless you **reposition** (break contact, re-approach) or accept **$\dot\alpha\neq 0$** for a transient. So the earlier statements remain true **conditional on the same contact maintenance model**: relaxation does not remove actuator limits; it adds a way to trade yaw error against rim motion.

## 13. Treating $\dot\alpha$ as a Control Objective (Special Case Strengthening)

The fixed-$\alpha$ case can be viewed as **regulating $\dot\alpha$ to zero**:

- If $\dot\alpha=0$ and the **initial heading matches** the primitive ($\alpha_0$ consistent with $\phi_0$ and desired contact point on the rim), then $\omega_r=\omega$ and the constant-command solution of `test_matchingvelo_report.md` applies: straightforward, open-loop-friendly commands.

- If $\dot\alpha=C\neq 0$ **constant** over an interval, then $\omega_r=\omega-C$ **constant** on that interval: the robot spins faster or slower than the object by exactly $C$, and the contact point **migrates steadily** along the rim at rate $C$. That can be useful as a **finite-time transition** but is often less desirable for steady pushing (more slip / wear / modeling burden).

A practical planner objective is to use **nonzero $\dot\alpha$ only transiently** (recover heading / avoid singularities), then drive **$\dot\alpha\to 0$** so that $\omega_r\to\omega$ and behavior approaches the simple primitive.

## 14. Evolution Feasibility (Not Just Instantaneous Matching)

For control implementation, matching must be checked together with command-rate limits. The relevant quantities are

$$
\dot v_r,\qquad \dot\omega_r.
$$

From the all-time coupling

$$
\omega_r+\dot\alpha=\omega,
$$

with object $\omega$ constant on a segment, we get

$$
\dot\omega_r=-\ddot\alpha.
$$

So:

- Fixed-$\alpha$ case ($\dot\alpha=0$): $\omega_r=\omega$ and $\dot\omega_r=0$.
- Constant migration case ($\dot\alpha=C$): $\omega_r=\omega-C$ and still $\dot\omega_r=0$.

Hence yaw acceleration is easy in both of those subcases. The nontrivial evolution burden is typically in $v_r(t)$.

Using

$$
v_r\mathbf e_\zeta + R_r\omega\,\mathbf e_{\phi,\perp}=\mathbf v_{cp},
$$

define

$$
\mathbf q(t):=\mathbf v_{cp}(t)-R_r\omega\,\mathbf e_{\phi,\perp}(t),
\qquad
v_r(t)=\mathbf e_\zeta(t)^\top \mathbf q(t).
$$

Then

$$
\dot v_r
=
\dot{\mathbf e}_\zeta^\top \mathbf q+\mathbf e_\zeta^\top \dot{\mathbf q}
$$

Now expand each term explicitly.

First,

$$
\mathbf e_\zeta=
\begin{bmatrix}
\cos\zeta\\
\sin\zeta
\end{bmatrix},
\qquad
\mathbf e_{\zeta,\perp}=
\begin{bmatrix}
-\sin\zeta\\
\cos\zeta
\end{bmatrix}.
$$

So by chain rule with $\dot\zeta=\omega_r$,

$$
\dot{\mathbf e}_\zeta
=
\dot\zeta
\begin{bmatrix}
-\sin\zeta\\
\cos\zeta
\end{bmatrix}
=
\omega_r\,\mathbf e_{\zeta,\perp}.
$$

Substituting gives

$$
\dot v_r
=
\omega_r\,\mathbf e_{\zeta,\perp}^\top \mathbf q+\mathbf e_\zeta^\top \dot{\mathbf q}.
$$

Second, where does $\omega$ enter? It enters through $\dot{\mathbf q}$ (not through $\dot{\mathbf e}_\zeta$):

$$
\mathbf q=\mathbf v_{cp}-R_r\omega\,\mathbf e_{\phi,\perp}.
$$

On a constant-twist segment, $\mathbf v_{cp}(t)$ rotates at rate $\omega$, and $\mathbf e_{\phi,\perp}(t)$ also rotates at rate $\dot\phi=\omega$ (from Section 5). Therefore $\mathbf q(t)$ rotates at rate $\omega$ (with constant norm on the segment).

Therefore:

- If $\dot\alpha=0$ (so $\omega_r=\omega$), $\mathbf e_\zeta$ co-rotates with $\mathbf q$, and $v_r$ is constant ($\dot v_r=0$).
- If $\dot\alpha=C\neq 0$, $\mathbf e_\zeta$ and $\mathbf q$ rotate at different rates, so $v_r(t)$ is generally time-varying (typically sinusoidal at relative rate $|C|$), implying nonzero $\dot v_r$.

So constant nonzero $\dot\alpha$ is kinematically possible, but may be dynamically hard if the required $|\dot v_r|$ exceeds actuator limits. In practice, add rate constraints

$$
|\dot v_r|\le a_{v,\max},\qquad |\dot\omega_r|\le a_{\omega,\max},
$$

and treat large $|\dot\alpha|$ as a transient tool, not a steady operating mode.

## 15. Fixed-$\alpha$ With Wrong Initial Heading: Immediate Failure

In the fixed-$\alpha$ special case, we enforce

$$
\dot\alpha=0,\qquad \omega_r=\omega.
$$

At $t=0$, the matching equation is

$$
v_r\mathbf e_{\zeta_0}+R_r\omega\,\mathbf e_{\phi_0,\perp}=\mathbf v_{cp}(0).
$$

With geometry and object motion fixed, define (same notation as `test_matchingvelo_report.md`)

$$
a:=v_{cp,x}(0)+\omega R_r\sin\phi_0,\qquad
b:=v_{cp,y}(0)-\omega R_r\cos\phi_0.
$$

Then

$$
v_r\cos\zeta_0=a,\qquad v_r\sin\zeta_0=b.
$$

This system is consistent only for

$$
\zeta_0=\operatorname{atan2}(b,a)\quad\text{or}\quad \operatorname{atan2}(b,a)+\pi,
$$

with corresponding forward/backward $v_r=\pm\sqrt{a^2+b^2}$.

Therefore, if fixed-$\alpha$ is enforced but the robot starts with a different heading, the velocity constraint fails immediately at $t=0$ (before any horizon-level argument).

## 16. Constant Nonzero $\dot\alpha$: Finite-Time Singularity Clock

If

$$
\dot\alpha=C\neq 0\quad\text{(constant)},
$$

then

$$
\alpha(t)=\alpha_0+Ct.
$$

Since singularity occurs at $\alpha\in\{k\pi\mid k\in\mathbb Z\}$, this trajectory will reach a singular set in finite time unless the horizon ends first. A useful bound is

$$
t_{\text{sing}}=
\frac{\min_{k\in\mathbb Z}|\alpha_0-k\pi|}{|C|}.
$$

So constant nonzero $\dot\alpha$ is typically a transient maneuver mode: it may satisfy kinematics for a while, but it carries an intrinsic “time-to-singularity” limit. This reinforces the practical strategy of driving $\dot\alpha\to 0$ after recovery.
