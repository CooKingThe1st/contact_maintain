# `mod_grid.py` — augmented grid A* + CHOMP

Holonomic path planning in two phases (plus optional shortcut). Tunables live at the top of `mod_grid.py` (`_RISK_*`, `_HEADING_WEIGHT`, `_CHOMP_*`).

## End-to-end implementation order ↔ this file

| Step | What the plan says | Where it lives in `mod_grid.py` |
|------|-------------------|--------------------------------|
| **1. Phase 1** | Grid A* with state `(cell, m_prev)`, `c_risk` bands, `c_heading` | **`phase1_augmented_astar`**: state `(x, y, m_in)` via `_state_index` / `_AugNode`; edge cost `_u_cost` + `_risk_cost(d)` + `_HEADING_WEIGHT * _angle_between_moves`. Constants `_RISK_D1..D3`, `_RISK_P1..P3`, `_HEADING_WEIGHT`. |
| **2. Obstacle / distance field** | 2D distance-to-obstacle on the planning grid | **`_build_clearance_meters`**: EDT in meters (`scipy.ndimage.distance_transform_edt`) or BFS fallback in cell units × `reso`. Built from same `obsmap` as baseline A*. |
| **3. Phase 2 CHOMP** | Resample to \(N\) points; \(J_\text{smooth} + J_\text{obs}\); gradient descent; fixed endpoints | **`phase2_chomp`**: **`_resample_polyline`** → `_CHOMP_POINTS`; inner loop uses **`_CHOMP_LAMBDA_SMOOTH`** (velocity smoothness on interior indices) + **`_CHOMP_LAMBDA_OBS`** × **`_obs_barrier_value_grad`** on **`_sample_clearance_and_grad`**; **`qx[1:-1]`, `qy[1:-1]`** updated — endpoints fixed. Step size **`_CHOMP_STEP`**, iters **`_CHOMP_ITERS`**. |
| **4. Validation** | Collision check dense samples after **each** CHOMP iteration | **Not implemented as specified.** There is no dense sample loop inside the CHOMP `for _ in range(_CHOMP_ITERS)` body. Post-CHOMP, **`_shortcut_path`** uses **`_segment_min_distance_to_points`** against the obstacle point cloud to only keep segments with clearance **`> safe_rr`** (validation-like, but once after smoothing, not every iteration). Phase 1 still enforces hard grid collision via **`ok_cell`**. |
| **5. Arc decomposition** | Straight + arc sequence for execution / visualization | **`phase3_straight_arc`**: tangent **circular fillets** at polyline corners (radius **`_ARC_RADIUS`**, clamped by edge length); dense polyline output. **`_arc_fillet_clear`** + straight checks vs **`ox,oy`**; unsafe corners stay sharp. Tunables: **`_ARC_MIN_TURN`**, **`_ARC_POINTS_PER_RAD`**, **`_ARC_CHECK_OBSTACLES`**. Called from **`astar_planning`** after **`_shortcut_path`**. |

## Public entry point

- **`astar_planning`**: `phase1_augmented_astar` → `phase2_chomp` → `_shortcut_path` → `phase3_straight_arc`. If augmented A* never reaches the goal, Phase 1 falls back to **`base_astar.astar_planning`**.

## Conceptual design (from earlier notes)

**Phase 1 — bias without a separate “zigzag” term**

- Augment each grid cell with the **previous motion index** \(m_\text{prev}\) (which neighbor you came from).
- Costs: **`c_move`** (same as baseline), **`c_risk`** (banded penalty from clearance), **`c_heading`** (penalize turn angle vs previous move so long straight legs are cheap).

**Phase 2 — why not only shortcut**

- **Shortcut** (`_shortcut_path`): removes vertices while preserving feasibility vs the sampled obstacle list; it does not optimize smoothness or a continuous clearance cost.
- **CHOMP**: optimizes a discrete chain with a smoothness term and a soft obstacle barrier from the same clearance field.

**Phase 3 (implemented)**

- **G1 fillets**: sequence of straight segments and circular arcs; not full Dubins (no constrained heading at ends) or biarc between arbitrary tangents—corners of the smoothed polyline are rounded with a single arc each.

## Short comparison

| Piece | Role |
|-------|------|
| Phase 1 + `c_heading` | Feasible path; prefers long straight grid legs |
| Phase 1 + `c_risk` | Pushes away from obstacles before continuous optimization |
| Shortcut | Cheap cleanup; not a substitute for CHOMP |
| CHOMP | Continuous smoothness + clearance-shaped cost |
| Arc fitting | `phase3_straight_arc`: fillet corners into straight + arc polyline (executable-style sampling) |
