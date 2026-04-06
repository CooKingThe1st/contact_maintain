# Diff-Drive XY Path + Segment Phases (Object Command)

This document matches the implementation in `contact_maintain/diffdrive_path_control.py` and `scripts/test/test_magnum_diffdrive_control.py`. For velocity-matching theory see `test_matchingvelo_report.md`; for the constant body-twist inverse see `test_motion_primitive.py` (`solve_constant_body_twist_from_SE2`).

## 1. Why diff-drive differs from holonomic

At the contact patch, a diff-drive robot has two velocity inputs \((v_r, \omega_r)\) and two matching constraints. The all-time match condition forces

\[
\omega_r = \omega_{\mathrm{obj}}
\]

and a unique forward or backward choice for \(v_r\) and initial heading \(\zeta_0\) from the geometry at \(t=0\) (see `test_matchingvelo_report.md` §5).

Therefore each **push** segment must be a valid **constant body-twist** primitive for the object (straight or arc in the quasi-static model), and **large heading changes** are not obtained by blending arbitrary \((v_x,v_y,\omega)\); they use explicit **in-place object rotation** and **robot re-heading** between retouches.

## 2. End-pose-anchored `mid_theta`

For segment \(i\), let the object start and end headings in world frame be \(\theta_s\) and \(\theta_e\) (from the theta schedule; see §4). Transform the displacement into the start frame:

\[
\begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix} = R(-\theta_s)\,(\mathbf{p}_e - \mathbf{p}_s), \qquad
\theta_{\mathrm{end,local}} = \mathrm{wrap}(\theta_e - \theta_s).
\]

Solve `solve_constant_body_twist_from_SE2` for \((\mathbf{v}^b, \omega, T)\) with \(\|\mathbf{v}^b\| = v_{\mathrm{speed}}\) (here \(v_{\mathrm{speed}} = v_{\mathrm{user,max}}\) from the test).

The **transition heading** used before the push (object rotate phase) is the start heading of the segment, which is **anchored by the end pose** as

\[
\theta_{\mathrm{mid}} = \theta_e - \omega T
\]

which equals \(\theta_s\) when the inverse is consistent (straight: \(\omega=0\); arc: \(\omega T = \theta_{\mathrm{end,local}}\)).

## 3. Six-phase segment lifecycle

Between HybridPath segment boundaries (`PathVelocityPlanner` with `look_ahead=0`), the test runs:

1. **RETOUCH_A** — same idea as holonomic hybrid retouch: hold object motion, `SwarmHost` in approaching / quick contact recovery for `hybrid_retouch_duration` (and optional timeout).
2. **ROBOT_ROTATE_A** — all robots rotate in place so their heading matches the **co-rotation** contact geometry for the upcoming object rotation: \(\zeta \approx \theta_{\mathrm{obj}} + \mathrm{atan2}(-n_y^b, -n_x^b)\) with outward normal in body frame (holonomic-style alignment from `test_matchingvelo.py` / report).
3. **OBJECT_ROTATE** — command \(\omega_{\mathrm{des}}\) with a small PID so \(\theta \to \theta_{\mathrm{mid}}\); linear velocity desired \((0,0)\).
4. **RETOUCH_B** — second retouch (mandatory after object rotation).
5. **ROBOT_ROTATE_B** — robots rotate to the **forward-branch** \(\zeta_0\) for the upcoming push primitive from the velocity-matching equations:

\[
\mathbf{v}_{\mathrm{cp}}^w = R(\theta_s)\,\mathbf{v}_{\mathrm{cp}}^b,\quad
\phi_0 = \mathrm{atan2}(-n_y^w, -n_x^w),\quad
a = v_{\mathrm{cp},x}^w + \omega\, R_r \sin\phi_0,\quad
b = v_{\mathrm{cp},y}^w - \omega\, R_r \cos\phi_0,
\]
\[
\zeta_0 = \mathrm{atan2}(b,\,a)\quad (\text{forward}).
\]

6. **PUSH** — stream desired object twist from `PathFollowingController` **only in this phase** (time-based \(s\), trapezoidal speed).

On a **segment boundary** while pushing, the controller **rewinds** one timestep (same trick as the holonomic test), marks the boundary consumed, sets `dd_next_segment_idx = seg_after`, and enters **RETOUCH_A** for the next segment.

## 4a. Fixed heading and zigzag (no net object yaw per segment)

If `--theta-mode fixed` matches the spawn heading (often `0`), then every vertex in the schedule has the same \(\theta_j \equiv \theta_{\mathrm{fix}}\). For each straight segment, \(\theta_{\mathrm{end,local}} = 0\): the motion-primitive inverse is a **pure translation** with \(\omega = 0\), so

\[
\theta_{\mathrm{mid}} = \theta_e - \omega T = \theta_e = \theta_{\mathrm{fix}}.
\]

So **no in-place object rotation** is required at segment corners; only **robot re-heading** for the new push primitive (`ROBOT_ROTATE_B`) matters. The harness therefore **skips** `ROBOT_ROTATE_A`, `OBJECT_ROTATE`, and `RETOUCH_B` when `|wrap(\theta_{\mathrm{mid}} - \theta_{\mathrm{meas}})|` is below `--obj-rotate-skip-tol` (default 0.06 rad) right after `RETOUCH_A`.

Forcing `OBJECT_ROTATE` toward \(\theta_{\mathrm{mid}}=0\) when the object is already on heading but **multi-contact / Phase7** cannot deliver a clean pure torque can saturate \(\omega\) and waste time (see log `TIMEOUT`); skipping this phase in the aligned case avoids that failure mode.

## 4. Three theta modes (vertex schedule)

Headings \(\theta_j\) are defined at each path vertex (length `num_components + 1`).

| Mode | Meaning |
|------|---------|
| `waypoint` | Zigzag: same corner tangents as `zigzag_vertex_thetas`. Other paths: same as `segment_tangent` if segment counts do not match. |
| `fixed` | \(\theta_j \equiv\) `--fixed-theta` for all vertices. |
| `segment_tangent` | \(\theta_j\) from `HybridPath` tangent at the arc length of vertex \(j\). |

These angles only **label** the segment start/end; **continuous** \(\omega = \kappa v\) “path theta” is not a separate stream for diff-drive in this harness.

## 5. Failure modes

- **Primitive inverse** fails (`solve_constant_body_twist_from_SE2` raises): degenerate segment; shorten path or change `v_speed`.
- **Velocity mismatch** at limits: reduce `v_user_max` or segment curvature.
- **Robot / object rotate timeouts**: `ROBOT_ROTATE_TIMEOUT_S` in the test forces progression to avoid deadlock (logged).

## 6. CLI (diff-drive test)

| Argument | Role |
|----------|------|
| `--xy-path` | `zigzag` \| `sine` |
| `--planner` | `hybrid` only (pure pursuit deferred; use holonomic test) |
| `--theta-mode` | `waypoint` \| `fixed` \| `segment_tangent` |
| `--fixed-theta` | radians when `fixed` |
| `--hybrid-retouch-duration`, `--hybrid-retouch-timeout` | retouch phases |

## 7. Comparison vs holonomic test

| Aspect | Holonomic (`test_magnum_holonomic_control.py`) | Diff-drive (`test_magnum_diffdrive_control.py`) |
|--------|-----------------------------------------------|-----------------------------------------------|
| Object twist source | PathFollowing / pursuit + theta PID | PathFollowing **only in PUSH** after rotate/retouch |
| Segment boundaries | Retouch + resume | Retouch → robot rotate → object rotate → retouch → robot rotate → push |
| Theta | `waypoint` / `fixed` / `path` | `waypoint` / `fixed` / `segment_tangent` |
