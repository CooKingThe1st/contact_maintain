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

## 6. Proof of All-Time Matching

To verify that the $t = 0$ matching extends to all time, factor the rotation:

**Object side:**

$$\mathbf{v}_{\text{cp}}(t) = R(\omega t)\, \mathbf{v}_{\text{cp}}(0)$$

**Robot side** (with $\omega_r = \omega$, constant $v_r$):

$$\mathbf{v}_{\text{contact}}^{\text{robot}}(t) = R(\omega t)\, \mathbf{v}_{\text{contact}}^{\text{robot}}(0)$$

Since $\mathbf{v}_{\text{contact}}^{\text{robot}}(0) = \mathbf{v}_{\text{cp}}(0)$ by construction, the equality holds for all $t$. Position matching follows by integration: if velocities match at all times and positions match at $t=0$, positions match at all times.

## 7. Numerical Verification

The test script `test_matchingvelo.py` verifies these results:

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

This provides a direct, closed-form mapping from the object motion plan to per-robot constant-velocity commands for each segment.
