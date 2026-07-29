# Translation vs Lever-Arm: Velocity Matching Speed Relationship

## 1. Motivation
In the existing `test_matchingvelo_report.md`, the key angular result is:

$$\boxed{\omega_r = \omega}$$

So both the object and the robot rotate at the same angular rate when velocities are matched at the contact patch.

This report derives the complementary **translation** relationship: even if the object's CoM translational speed is small, a **large lever arm** (contact point far from the CoM) can force the **contact point / robot center** translational speed to be large.

## 2. Notation
In planar motion:
- Object CoM has body-frame translational velocity $\mathbf{v}^b=(v_x^b,v_y^b)$ and angular rate $\omega$.
- The object orientation is $\theta(t)$ with $\dot{\theta}=\omega$.
- Let $\mathbf{r}^b$ be the contact point position **relative to the object CoM**, expressed in the object body frame.
- The robot is a circle of radius $R_r$. The outward normal at the contact is $\hat{\mathbf{n}}^b$ (body frame).
- Define the body-frame vector from the object CoM to the robot center:
  $$
  \mathbf{d}^b = \mathbf{r}^b + R_r\,\hat{\mathbf{n}}^b.
  $$

Let $J$ be the 90-degree rotation matrix:
$$
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},
\qquad
\omega \, J\,\mathbf{x} = \omega\,\hat{\mathbf{z}}\times \mathbf{x}\ \text{(planar cross product)}.
$$

## 3. Contact-point velocity (why lever arm matters)
The contact-point velocity in body-frame is:
$$
\mathbf{v}_{cp}^b = \mathbf{v}^b + \omega\,J\,\mathbf{r}^b.
$$
Therefore its world-frame magnitude is the same:
$$
\|\mathbf{v}_{cp}(t)\| = \|\mathbf{v}_{cp}^b\|.
$$

### Key bound (speed scaling)
Using the triangle inequality:
$$
\big|\ \|\omega\|\,\|\mathbf{r}^b\| - \|\mathbf{v}^b\|\ \big|
\le
\|\mathbf{v}_{cp}^b\|
\le
\|\mathbf{v}^b\| + \|\omega\|\,\|\mathbf{r}^b\|.
$$

Interpretation:
- If the contact point is far from the CoM (large $\|\mathbf{r}^b\|$), then even small CoM translation $\|\mathbf{v}^b\|$ can yield large $\|\mathbf{v}_{cp}\|$ due to the $\omega$ term.
- This explains the empirical observation: **object CoM translation can be low while robot/contact translation becomes high**.

## 4. Robot center translation speed under matched angular rate
From the existing derivation in `test_matchingvelo_report.md`, when angular velocity is matched ($\omega_r=\omega$) and velocity at the contact patch is matched, the robot center velocity in the object body frame can be written as:
$$
\mathbf{v}_{center}^b = \mathbf{v}^b + \omega\,J\,\mathbf{d}^b,
\qquad
\mathbf{d}^b=\mathbf{r}^b+R_r\hat{\mathbf{n}}^b.
$$
Hence:
$$
\boxed{
\|\mathbf{v}_{center}\| = \|\mathbf{v}^b + \omega J\mathbf{d}^b\|
}
$$

just a quick test

### Translation feasibility with a robot speed cap
If the robot has a translational speed magnitude limit $V_{r,\max}$ (however your controller maps it to actuator limits), then the matched-motion reference must satisfy:
$$
\|\mathbf{v}^b + \omega J\mathbf{d}^b\| \le V_{r,\max}.
$$

Let $\mathbf{w}^b := \omega J\mathbf{d}^b$. Then the constraint is:
$$
\|\mathbf{v}^b + \mathbf{w}^b\| \le V_{r,\max}.
$$

A conservative lever-arm-based upper bound on CoM speed magnitude follows from the triangle inequality:
$$
\|\mathbf{v}^b\| \le V_{r,\max} - \|\omega\|\,\|\mathbf{d}^b\|
\qquad (\text{whenever the RHS is nonnegative}).
$$

This bound makes the relationship explicit:
- Larger lever arm $\|\mathbf{d}^b\|$ reduces the allowable CoM translation magnitude.
- If you don’t limit CoM translation accordingly, the contact/center velocity can exceed what robots can do simultaneously.

## 5. Practical implication for “translation + rotation limits”
Because $\omega$ is coupled by the all-time matching requirement ($\omega_r=\omega$), any design that limits rotation implicitly sets a scale for the lever-arm-induced translational demand via $\|\omega\|\,\|\mathbf{d}^b\|$.

So a good next-feature reference shaping is:
1. enforce $\boxed{|\omega| \le \omega_{\max}}$ (already consistent with matching),
2. additionally enforce a **translation** cap that accounts for lever arm:
   $$
   \|\mathbf{v}^b\| \le V_{ref,\max}(\omega,\mathbf{d}^b)
   \approx \max\{0,\ V_{r,\max}-\|\omega\|\,\|\mathbf{d}^b\|\}.
   $$

This should reduce the mismatch case where XY finishes quickly but the orientation/ω correction drives high effective tangential motion at the contact, producing extra “self-rotation” at the end.

