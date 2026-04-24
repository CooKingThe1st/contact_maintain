# Diff-Drive Single-Pusher: Phase 2/3 Tracking Notes

## Context

Observed multifold issues in current diff-drive single-pusher tests:

1. Large bumper geometry causes object disturbance during robot self-rotation, increasing deviation and eventually losing grip.
2. Zigzag test succeeds on first straight segment, then loses contact on the second segment and the run collapses.
3. Object self-rotation/arc behavior is weak; diff-drive struggles on curved motion compared with straight-line segments.

---

## Controller Direction (Immediate)

- Use **feedforward + feedback**, not feedforwa rd-only.
- Keep matching-velocity solution as feedforward (`v_ff`, `omega_ff`, branch choice), then add feedback projected into diff-drive command space:
  - `e_n`: normal/contact distance error (maintain clamp/contact),
  - `e_t`: tangential/slip error (prevent contact migration),
  - `e_zeta`: heading/branch-consistency error.
- Apply command limits and contact hysteresis.
- Add contact-recovery behavior when force/contact drops (slowdown + re-align + re-approach).

---

## Phase 2 Plan (Single Pusher)

Use two separate controller objectives/modes:

### Mode A - Maintain Contact (Geometry Priority)

- Primary objective: stable contact maintenance.
- Prioritize `e_n`, `e_t`, and robust force/contact hysteresis.
- Allow higher object-velocity tracking error when needed.

### Mode B - Push Desired Velocity (Tracking Priority)

- Primary objective: track desired object `(v, omega)`.
- Contact maintenance is treated as a hard constraint / fallback trigger.
- If infeasible (limits/friction/contact), degrade gracefully to stabilization behavior.

### Segment Transition / Curvature Handling

- Add transition guards between segments (do not advance unless contact + heading conditions are valid).
- In high-curvature/arc motion, reduce linear speed and tighten angular bounds.

---

## Phase 3 Plan (Diff-Drive Multi-Robot)

Do not move to full multi-robot until single-robot robustness is proven.

Recommended gate:

1. Single robot must maintain contact through multi-segment paths and arc portions.
2. Then test two robots on one side with role split:
   - contact-anchor role (stability),
   - velocity-contributor role (tracking).
3. Add per-robot feasibility weighting (de-weight robots near friction-cone saturation/contact loss).

---

## Additional Notes (New)

1. **Approach pose bound:** for diff-drive approach to object edge, heading deviation has an effective maximum of about **90 degrees**, and this can be controlled explicitly in approach logic.
2. **Adaptive contact point update:** contact point target should adapt after robot self-rotation, since slight geometric shift occurs while matching initial heading.
