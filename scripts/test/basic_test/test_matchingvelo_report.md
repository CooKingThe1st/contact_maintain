# Velocity Matching for Robot-Object Contact Maintenance

## 1. Problem Statement

Given an object under all-face contact (AFC) with a circular robot of radius $R_r$, we seek robot commands that make the robot's contact-point velocity match the object's contact-point velocity at all times along a trajectory segment.

The object moves under **constant body-frame velocity** $(\mathbf{v}^b, \omega)$, where $\mathbf{v}^b = (v_x^b, v_y^b)$ is translational and $\omega$ is angular. Under this assumption, the object's CoM traces either a **straight line** ($\omega = 0$) or an **arc** ($\omega \neq 0$). This is the quasi-static / differential flatness property: the trajectory is fully determined by the constant flat output, and admits a direct mapping to robot commands.

## 2. Object Kinematics

### CoM propagation

With orientation $\theta(t) = \theta_0 + \omega t$ and rotation matrix $R(\theta)$:

$$\dot{\mathbf{p}} = R(\theta)\, \mathbf{v}^b, \qquad \dot{\theta} = \omega$$

### Contact-point velocity

Let $\mathbf{r}^b$ be the contact point in the object body frame (relative to CoM), obtained from the boundary parameterisation at parameter $t_{\text{param}}$. The contact-point velocity in the world frame is:

$$\mathbf{v}_{\text{cp}}(t) = R(\theta)\, \mathbf{v}^b \;+\; \omega\, R(\theta)\begin{pmatrix} -r^b_y \\ r^b_x \end{pmatrix}$$

In body-frame terms, define the **constant** body-frame contact velocity:

$$\mathbf{v}_{\text{cp}}^b = \mathbf{v}^b + \omega \begin{pmatrix} -r^b_y \\ r^b_x \end{pmatrix}$$

Then $\mathbf{v}_{\text{cp}}(t) = R(\theta_0 + \omega t)\, \mathbf{v}_{\text{cp}}^b$, which rotates in the world frame at rate $\omega$ but has **constant magnitude and body-frame direction**.

### Outward normal

The outward normal at the contact point is similarly constant in body frame: $\hat{\mathbf{n}}^b$. In world frame it rotates: $\hat{\mathbf{n}}(t) = R(\theta(t))\, \hat{\mathbf{n}}^b$.

## 3. Contact Geometry

The robot is a circle of radius $R_r$ that touches the object at the contact point from outside along the outward normal. The **approach direction** $\varphi$ is the world-frame angle from the robot center toward the contact point, equal to the inward normal direction:

$$\varphi(t) = \text{atan2}(-\hat{n}_y(t),\; -\hat{n}_x(t))$$

The robot center sits at:

$$\mathbf{p}_{\text{robot}}(t) = \mathbf{p}_{\text{cp}}(t) + R_r\, \hat{\mathbf{n}}(t)$$

The **contact angle** $\alpha$ is the angle of the contact point on the robot body relative to the robot heading $\zeta$:

$$\alpha = \varphi - \zeta$$

This is the "intended contact position on the robot body" in the methodology.

## 4. Holonomic Robot

A holonomic robot has 3 DOF: $(v_x, v_y, \omega_r)$. The velocity-matching constraint provides only 2 equations, so the system is **underdetermined** -- a solution always exists.

### Analytical solution

The robot center offset from the CoM in body frame is constant:

$$\mathbf{d}^b = \mathbf{r}^b + R_r\, \hat{\mathbf{n}}^b$$

The center position is therefore determined algebraically at every instant:

$$\mathbf{p}_{\text{robot}}(t) = \mathbf{p}_{\text{obj}}(t) + R(\theta(t))\, \mathbf{d}^b$$

The robot heading faces the contact point:

$$\zeta(t) = \theta(t) + \text{atan2}(-\hat{n}^b_y,\; -\hat{n}^b_x)$$

so $\dot{\zeta} = \omega$, meaning $\omega_r = \omega$ (heading co-rotates with the object -- constant).

The center velocity in the object body frame is constant:

$$\mathbf{v}_{\text{center}}^b = \mathbf{v}^b + \omega \begin{pmatrix} -d^b_y \\ d^b_x \end{pmatrix}$$

The translational velocity tracks the contact-point motion; the angular velocity only maintains heading alignment. Since translation is fully controllable, this is trivially achievable. The position is algebraically exact with **zero tracking error**.

## 5. Differential-Drive Robot

A differential-drive robot has 2 DOF: forward velocity $v_r$ and angular velocity $\omega_r$. The velocity-matching constraint provides 2 equations, so the system is **fully determined** -- a solution exists only for specific configurations.

### Velocity-matching equation

The robot produces contact-point velocity:

$$\mathbf{v}_{\text{contact}}^{\text{robot}} = \begin{pmatrix} v_r \cos\zeta \\ v_r \sin\zeta \end{pmatrix} + \omega_r \begin{pmatrix} -R_r \sin\varphi \\ R_r \cos\varphi \end{pmatrix}$$

where $\varphi = \zeta + \alpha$ is the world-frame contact direction.

Setting this equal to $\mathbf{v}_{\text{cp}}$ and requiring the match to hold **for all time**, we derive two conditions.

### Condition 1: angular velocity

The object-side contact velocity rotates at rate $\omega$:

$$\mathbf{v}_{\text{cp}}(t) = R(\omega t)\, \mathbf{v}_{\text{cp}}(0)$$

The robot-side contact velocity with constant $(v_r, \omega_r)$ rotates at rate $\omega_r$:

$$\mathbf{v}_{\text{contact}}^{\text{robot}}(t) = R(\omega_r t)\, \mathbf{v}_{\text{contact}}^{\text{robot}}(0)$$

For all-time matching, the rotation rates must be equal:

$$\boxed{\omega_r = \omega}$$

This is the key insight: the robot rotates at the **same rate as the object**, preserving the relative contact geometry. There is no differential rotation at the contact patch, hence no parasitic friction torque.

This is exactly the $\dot\alpha=0$ special case of the relaxed model in
`test_matchingvelo_relaxform_report.md`, where
$\omega_r+\dot\alpha=\omega \Rightarrow \omega_r=\omega$.

### Condition 2: initial velocity matching

At $t = 0$, the velocity-matching equation becomes:

$$\begin{pmatrix} v_r \cos\zeta_0 \\ v_r \sin\zeta_0 \end{pmatrix} = \mathbf{v}_{\text{cp}}(0) - \omega \begin{pmatrix} -R_r \sin\varphi_0 \\ R_r \cos\varphi_0 \end{pmatrix}$$

where $\varphi_0$ is determined by the contact geometry (inward normal direction at $t=0$). Defining:

$$a = v_{\text{cp},x}(0) + \omega\, R_r \sin\varphi_0, \qquad b = v_{\text{cp},y}(0) - \omega\, R_r \cos\varphi_0$$

we have $v_r \cos\zeta_0 = a$ and $v_r \sin\zeta_0 = b$, giving exactly **two solutions**:

**Forward** ($v_r > 0$):

$$\zeta_0 = \text{atan2}(b,\, a), \qquad v_r = +\sqrt{a^2 + b^2}$$

**Backward** ($v_r < 0$):

$$\zeta_0 = \text{atan2}(b,\, a) + \pi, \qquad v_r = -\sqrt{a^2 + b^2}$$

The contact angle on the robot body follows from the geometry:

$$\alpha = \varphi_0 - \zeta_0$$

### Summary

Given the object motion $(\mathbf{v}^b, \omega)$ and a chosen contact point on the object, there is a **specific initial heading** $\zeta_0$ (coupled to a specific contact position $\alpha$ on the robot body) such that the robot applies **constant** $(v_r, \omega_r)$ for the entire trajectory segment. The commands are:

| Quantity | Value |
|----------|-------|
| $\omega_r$ | $= \omega$ (matches object angular velocity) |
| $v_r$ | $= \pm\sqrt{a^2 + b^2}$ (forward or backward) |
| $\zeta_0$ | $= \text{atan2}(b, a)$ or $+ \pi$ |
| $\alpha$ | $= \varphi_0 - \zeta_0$ (determined, not free) |

### Straight-line special case

When $\omega = 0$: $\omega_r = 0$, $a = v_{\text{cp},x}(0)$, $b = v_{\text{cp},y}(0)$. The robot heading aligns with the contact velocity direction and the robot goes straight at constant speed $\|\mathbf{v}_{\text{cp}}\|$. No rotation of either the object or the robot.

### $\alpha_0$ along a boundary edge: bands vs pure translation

Fix a desired constant object twist $(\mathbf{v}^b,\omega)$ and consider moving the **contact assignment** along a single **straight** polygonal edge (same $\hat{\mathbf{n}}^b$, hence same $\varphi_0$ at $t=0$ for all points on that edge).

- **Pure translation ($\omega = 0$).** Then $\mathbf{v}_{\text{cp}}^b = \mathbf{v}^b$ does not depend on $\mathbf{r}^b$, so $a$ and $b$ in Condition 2 are the same at every point on the edge. Hence $\zeta_0$ is edge-wise constant and
  $$\alpha_0 = \varphi_0 - \zeta_0$$
  is **constant along the entire edge** (it may still jump when one moves to another edge because $\varphi_0$ changes with $\hat{\mathbf{n}}^b$).

- **Nonzero $\omega$.** Then $\mathbf{v}_{\text{cp}}^b = \mathbf{v}^b + \omega\,(-r_y^b,\, r_x^b)^\top$ **depends on** $\mathbf{r}^b$. Sliding the contact along the edge changes $\mathbf{r}^b$ linearly, so $a$, $b$, and therefore $\zeta_0$ vary; **$\alpha_0$ is no longer guaranteed to be a single value on that edge**---it sweeps an **interval** (a band) whose width grows with $|\omega|$ for typical $(\mathbf{v}^b,\omega)$ and edge length. In the small-$|\omega|$ limit the rotational contribution to $\mathbf{v}_{\text{cp}}^b$ is weak, so the edge-wise band of $\alpha_0$ becomes **narrow**---the constraint is **stricter** in the sense of being close to the $\omega=0$ case (almost constant $\alpha_0$ along the edge).

Empirically, the companion script `test_matchingvelo.py` supports **`--mode alpha_scan`**: it samples the global boundary parameter $t \in [0,1)$, prints the **$t$ range per edge** (fraction of perimeter), tabulates the analytical $\alpha_0$ (and related quantities) without simulation, and summarizes **min / max / span** of $\alpha_0$ on the sample per edge. That scan confirms the above: edge-wise **ranges** tied to the chosen twist, **tighter** when $|\omega|$ is small.

### Rationale for a disc-shaped differential-drive robot

The diff-drive derivation assumes the robot is a **circle of radius $R_r$**: the contact point on the robot is always at distance $R_r$ from the body center along direction $\varphi = \zeta + \alpha$, and the term $\omega_r(-R_r\sin\varphi,\, R_r\cos\varphi)^\top$ is the rigid rotation of that offset. With a **non-circular** rigid footprint, both the **normal distance to the contact** and the **map from $(v_r,\omega_r)$ to world-frame patch velocity** would depend on body-fixed contact geometry and local curvature in a way that **cannot** be reduced to a single constant $R_r$ and one body-fixed angle $\alpha$ in the same closed form. The **two scalar** velocity constraints at the contact would then couple to additional shape parameters, breaking the clean **fully determined** 2$\times$2 structure of \S5 and the design picture in which each pusher is interchangeable modulo $(\mathbf{v}^b,\omega)$ and contact assignment.

Using a **disc** model is therefore aligned with the methodology: it is the natural geometry in which **constant** $(v_r,\omega_r)$ with **fixed** $\alpha$ on the robot body matches the object's contact velocity over a segment---the analogue, within this AFC class, of a simple **omnidirectional** closure at the contact (holonomic robots remain strictly richer because they have three actuated velocities for two constraints).

## 6. Proof of All-Time Matching

To verify that the $t = 0$ matching extends to all time, factor the rotation:

**Object side:**

$$\mathbf{v}_{\text{cp}}(t) = R(\omega t)\, \mathbf{v}_{\text{cp}}(0)$$

**Robot side** (with $\omega_r = \omega$, constant $v_r$):

$$\mathbf{v}_{\text{contact}}^{\text{robot}}(t) = R(\omega t)\, \mathbf{v}_{\text{contact}}^{\text{robot}}(0)$$

Since $\mathbf{v}_{\text{contact}}^{\text{robot}}(0) = \mathbf{v}_{\text{cp}}(0)$ by construction, the equality holds for all $t$. Position matching follows by integration: if velocities match at all times and positions match at $t=0$, positions match at all times.

## 7. Numerical Verification

The test script `test_matchingvelo.py` verifies these results in **`plot`** mode (full propagation and figures). The same script also supports **`--mode alpha_scan`**, which only evaluates the analytical mapping from boundary parameter $t$ to $\alpha_0$ (per-edge $t$ intervals, tables, and per-edge min/max on the sample) as described in \S5.

| Test case | Holonomic position error | DD position error | DD velocity error |
|-----------|--------------------------|-------------------|-------------------|
| Pi shape, arc ($\omega = 0.3$, 5 s) | 0.000 mm (exact) | 0.360 mm (Euler drift) | $< 10^{-15}$ m/s |
| Rectangle, straight ($\omega = 0$, 5 s) | 0.000 mm (exact) | 0.000 mm (exact) | $< 10^{-17}$ m/s |
| Rectangle, near-arc ($\omega = 0.05$, 20 s) | 0.000 mm (exact) | 0.118 mm (Euler drift) | $< 10^{-14}$ m/s |

- **Holonomic**: position is computed algebraically (no integration), so error is exactly zero.
- **Diff-drive**: position error comes solely from Euler integration of the constant commands. The velocity match is exact (machine epsilon). The error scales as $O(\Delta t^2 \cdot T)$ and vanishes with smaller time steps.
- **Both solutions** (forward and backward) produce identical error magnitudes, differing only in the sign of $v_r$ and the heading direction.

## 8. Implications for the Planning Phase

The constant-velocity property means the planning algorithm can decompose the collision-free trajectory into arc and line segments, and for each segment:

1. Compute the object-side contact velocity from the segment's constant $(\mathbf{v}^b, \omega)$.
2. For each robot, solve the 2-equation system to obtain $(\zeta_0, \alpha, v_r, \omega_r)$.
3. Verify that $v_r$ and $\omega_r$ lie within the robot's actuator limits.
4. If not feasible, adjust the contact assignment or waypoint in the planning phase.

When $|\omega|$ is not negligible, remember that **$\alpha_0$ is not a single constant over a long edge**---it varies with where the contact sits (\S5). A planner that treats $\alpha$ as freely placeable along an edge should respect the **admissible band** implied by $(\mathbf{v}^b,\omega)$ (e.g. via `alpha_scan` sweeps or an explicit closed-form range on line segments).

This provides a direct, closed-form mapping from the object motion plan to per-robot constant-velocity commands for each segment, **provided** the robot side matches the disc diff-drive model in \S3--\S5.

## 9. Flat bumper (line contact): two-endpoint reduction

A **flat bumper** is a rigid line segment on the robot body with fixed endpoints $\mathbf{r}_{E1}^b$, $\mathbf{r}_{E2}^b$ (relative to the robot center). The object touches along a **straight polygonal edge** with constant outward normal $\hat{\mathbf{n}}^b$ and tangent $\hat{\mathbf{t}}^b$. The bumper is placed so its endpoints contact two object material points $\mathbf{r}_{o1}^b$, $\mathbf{r}_{o2}^b$ on that edge (typically $\mathbf{r}_{o2}^b - \mathbf{r}_{o1}^b = \ell\,\hat{\mathbf{t}}^b$ with $\ell$ the bumper span projected onto the edge).

### 9.1 Holonomic robot

Still **underdetermined** (3 DOF, 2 velocity constraints per point). Matching the object twist at both endpoints does **not** impose a fixed rim angle $\alpha_0$: the holonomic base can realize the same $\omega_r = \omega$ and the required translational field without the disc’s “pick $\zeta_0$ to hit one $\alpha_0$” bottleneck. Segment feasibility is therefore **much easier** than for diff-drive (see companion script `test_matchingvelo_segment.py`).

### 9.2 Diff-drive: exact constraints (fixed patch, $\dot\alpha = 0$)

As in \S5, require $\omega_r = \omega$ and **one** command pair $(v_r, \zeta_0)$ for the whole segment. At $t=0$, let $\mathbf{r}_{Ei}(0) = R(\zeta_0)\,\mathbf{r}_{Ei}^b$ and $\mathbf{v}_{\text{cp},i}(0)$ be the object contact velocity at $\mathbf{r}_{oi}^b$. Per endpoint $i \in \{1,2\}$:

$$
a_i := v_{\text{cp},i,x}(0) + \omega\, r_{Ei,y}(0), \qquad
b_i := v_{\text{cp},i,y}(0) - \omega\, r_{Ei,x}(0).
$$

A single $(v_r, \zeta_0)$ must satisfy both, hence:

$$
\boxed{a_1 = a_2, \qquad b_1 = b_2.}
$$

Then $v_r \cos\zeta_0 = a_1$, $v_r \sin\zeta_0 = b_1$, and $\omega_r = \omega$ as in \S5. For $\omega \neq 0$, eliminating $\zeta_0$ from the two equalities gives a **2-vector** constraint coupling object edge geometry and bumper chord $\mathbf{w}^b := \mathbf{r}_{E1}^b - \mathbf{r}_{E2}^b$ (see `test_matchingvelo_segment.py`).

**$\alpha_0$ picture (disc reduction).** Treat each endpoint as a separate disc contact with lever $|\mathbf{r}_{Ei}^b|$ and compute $\alpha_{0,i}^{\text{req}}$ from \S5 at the corresponding $\mathbf{r}_{oi}^b$. The robot design fixes $\psi_i^b = \operatorname{atan2}(r_{Ei,y}^b, r_{Ei,x}^b)$ and $\Delta\psi^b = \psi_2^b - \psi_1^b$. Feasibility requires

$$
\boxed{\alpha_{0,2}^{\text{req}} - \alpha_{0,1}^{\text{req}} = \Delta\psi^b}
$$

(at the **same** object placement along the edge) **and** the same $\zeta_0$ from both endpoints (equivalent to $a_1=a_2$, $b_1=b_2$). On a **disc**, $\zeta_0$ can be chosen after picking one contact, so one $\alpha_0^{\text{req}}$ is enough. On a **bumper**, $\Delta\psi^b$ is fixed; you cannot rotate the robot to reconcile two different required $\alpha_0$ values.

### 9.3 Relation to per-edge $\alpha_0$ bands (\S5)

Fix $(\mathbf{v}^b, \omega)$ and scan contact along one object edge. \S5 gives an **interval** (band) of $\alpha_0^{\text{req}}$ for a **single** disc contact. For a bumper, scan the same edge and compute bands for $\alpha_{0,1}^{\text{req}}$ and $\alpha_{0,2}^{\text{req}}$ (endpoints mapped to $\mathbf{r}_{o1}^b$, $\mathbf{r}_{o2}^b$ along the edge). A placement $t$ is feasible only if, at that $t$,

$$
\alpha_{0,2}^{\text{req}}(t) - \alpha_{0,1}^{\text{req}}(t) \approx \Delta\psi^b
$$

**and** $a_1=a_2$, $b_1=b_2$. The independent bands $[\alpha_{0,1}^{\min}, \alpha_{0,1}^{\max}]$ and $[\alpha_{0,2}^{\min}, \alpha_{0,2}^{\max}]$ are necessary but not sufficient: the difference must hold **pointwise** at the same $t$, not only as overlapping intervals. The script `test_matchingvelo_segment.py` implements **`--mode scan`** (edge-wise bands + per-$t$ yes/no) and **`--mode plot`** (pick a feasible $t$, propagate, plot).

### 9.4 Pure translation on the edge

When $\omega = 0$, $\mathbf{v}_{\text{cp},1}^b = \mathbf{v}_{\text{cp},2}^b = \mathbf{v}^b$, so $a_1=a_2$ and $b_1=b_2$ hold for every placement along the edge; only $\zeta_0$ and $v_r$ remain (as in the straight-line case \S5). The bumper–diff-drive constraint reduces to the **$\alpha$ difference** condition when $\omega \neq 0$.
