# Holonomic XY Path + Theta Modes (Object Command)

This document matches the implementation in `contact_maintain/holonomic_path_control.py` and `scripts/test/test_magnum_holonomic_control.py`.

## 1. Object twist

The high-level command to the object (then realized by Phase7 through contacts) is planar twist

\[
\mathbf{v}_{\mathrm{des}} = (v_x, v_y), \qquad \omega_{\mathrm{des}}.
\]

## 2. Two XY references

### 2.1 Planner `hybrid` (primary)

1. **Zigzag**: piecewise straight segments in the plane; `HybridPath` of `StraightComponentPath`.
2. **Sine**: dense samples of \(y = A\sin(\omega_x x)\), then `mod_grid.phase3_min_segments` fits **straight + circular arc** primitives (no obstacles; clearance 0). Fallback: `SplineComponentPath` on the dense polyline.

Longitudinal motion uses `PathVelocityPlanner` inside `PathFollowingController` (trapezoidal speed, lateral accel cap via curvature). `look_ahead=0` stops at internal segment joins—natural **stop-and-go** windows for contact recovery.

### 2.2 Planner `pursuit` (experimental)

Dense **polyline** along the same zigzag vertices or sine samples. `HolonomicPurePursuitPolyline` picks a lookahead point at arc length \(s + L_f\), \(L_f = k_f \|\mathbf{v}\| + L_d\), and commands speed \(\times\) unit vector toward lookahead. Speed along the path uses a scalar trapezoid on total arc length \(L\).

## 3. Three theta modes

Let \(\theta\) be object heading, \(\theta_g\) a goal heading.

### 3.1 `waypoint`

Discrete headings \(\theta_i\) at vertices (zigzag) or along the sine (12 samples with tangent \(\arctan2(A\omega_x\cos(\omega_x x), 1)\)). Arc-length milestones \(s_i\) are obtained by projecting each reference \((x,y)\) onto the hybrid path or pursuit polyline.

At runtime, with current path progress \(s\),

\[
\theta_g(s) = \theta_i \quad \text{for the largest } s_i \le s.
\]

Angular rate uses the same PD form as the rectangle test:

\[
\omega_{\mathrm{des}} = K_p \,\mathrm{wrap}(\theta_g - \theta) - K_d \,\omega.
\]

Linear \((v_x,v_y)\) comes **only** from the XY planner (PathFollowing or Pursuit), not from position PID.

### 3.2 `fixed`

\(\theta_g \equiv \theta_{\mathrm{fix}}\) (CLI `--fixed-theta`). Same \(\omega\) PD; linear velocity from XY planner.

### 3.3 `path`

- **Hybrid + PathFollowing**: use full `PathFollowingController` output, including \(\omega\) from `PathDirectionProvider` (tangent \(\times\) curvature \(\times\) speed).
- **Pursuit**: \(\theta_g\) is the heading of the path tangent at current \(s\); \(\omega\) from the same PD toward \(\theta_g\).

## 4. Phase coupling

- **Holonomic** robots: at the object level, \((v_x,v_y,\omega)\) is unconstrained. The split is **XY from path geometry / time law** and **\(\omega\)** from either coupled path geometry (`path`) or independent heading PID (`waypoint`, `fixed`).
- **Execution**: `test_magnum_holonomic_control.py` waits until all agents are in `push` mode, then resets the path controller and streams desired twist to Phase7.

## 5. Comparison

| Aspect | `hybrid` | `pursuit` |
|--------|----------|-----------|
| Geometry | Straight/arc + trapezoid | Polyline + lookahead |
| Smoothness | Fit + velocity caps | Smooth pursuit; tuning \(L_d, k_f\) |
| Segment stops | Yes (`look_ahead=0`) | No (continuous) |

## 6. CLI (holonomic test)

| Argument | Role |
|----------|------|
| `--xy-path` | `zigzag` \| `sine` |
| `--planner` | `hybrid` \| `pursuit` |
| `--theta-mode` | `waypoint` \| `fixed` \| `path` |
| `--fixed-theta` | radians when `fixed` |
| `--sine-amplitude`, `--sine-omega-x`, `--sine-x0`, `--sine-x1` | sine shape |
| `--zigzag-x0`, `--zigzag-x1`, `--zigzag-segments`, `--zigzag-y-amplitude` | zigzag shape |
