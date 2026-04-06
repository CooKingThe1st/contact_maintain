# HA_draw App + Planner Technical Notes

This document describes:

- `app.py` (GUI/workbench orchestration)
- `HybridAstarPlanner/mod_grid.py` (disk-based planner pipeline)
- `HybridAstarPlanner/mod_grid_SE.py` (SE(2) footprint planner pipeline)

It focuses on algorithm flow, pseudocode, equations, complexity, and the role of map resolution.

---

## 1. System Overview

The workbench has three key layers:

1. **UI + scenario management (`app.py`)**
   - Builds obstacle point cloud from drawn rectangles/polylines.
   - Selects planner and mode from radios.
   - Persists scenario (`map`, `pose`, `robot`, `draw`, `planner`, `obstacles`).
   - Visualizes path as either disk envelopes or oriented polygon footprints.

2. **Disk planner (`mod_grid.py`)**
   - Uses robot radius `rr` (for non-circle shapes, can be circumradius from selected shape).
   - Pipeline:
     - Phase 1: augmented A*
     - Phase 2: CHOMP + shortcut
     - Phase 3: shortcut + straight/arc primitive compression

3. **SE(2) planner (`mod_grid_SE.py`)**
   - Uses full polygon footprint with yaw in state.
   - Pipeline:
     - Phase 1: SE(2) augmented A*
     - Phase 3: SE(2) primitive compression with constant body-twist primitives
   - Phase 2 remains unimplemented.

---

## 2. `app.py` (Workbench Orchestration)

### 2.1 Planner selection

`app.py` exposes planner choices:

- `grid_astar` (baseline disk)
- `mod_grid` (disk safe+smooth)
- `mod_grid_SE` (full footprint SE(2))
- `hybrid_astar` (car-like planner)

For `mod_grid`:
- `stop_phase` in {1,2,3}
- optional yaw filling mode for non-circle visualization:
  - `none`
  - `linear`
  - `differential_flatness` (phase-3 primitive-based)

For `mod_grid_SE`:
- `stop_phase` in {1,3}; phase 2 disabled.

### 2.2 Radius / footprint handling

- Circle-like/disk planning radius:
  - For circle: `rr = max(width, length)/2`
  - For non-circle in disk planners: circumradius extracted from shape footprint.

- Safety margin:
  - Parsed from right-panel textbox.
  - Passed to both `mod_grid` and `mod_grid_SE`.
  - Persisted in scenario JSON under `robot.safety_margin`.

### 2.3 Obstacle sampling model

Obstacles are converted into point cloud `(ox, oy)`:

- map boundary sampled at resolution `reso`
- rectangles filled with grid points (`np.arange` on x/y)
- polylines thickened by sampling around line segments

This point-cloud representation is used by all planners for collision/clearance checks.

---

## 3. `mod_grid.py` (Disk Planner)

## 3.1 Phase 1: Augmented A*

State:

```text
s = (x_i, y_i, m_prev)
```

- `(x_i, y_i)`: grid index
- `m_prev`: incoming move index (or -1 at start)

Cost per expansion:

```text
g' = g + c_move + c_risk(d) + lambda_h * c_heading(m_prev, m_new)
```

where:
- `d`: local obstacle clearance from EDT
- `c_risk`: piecewise penalty over clearance bands

Occupancy inflation for phase 1 hard validity:

```text
r_occ = r_r + safety_margin
```

### Phase 1 pseudocode

```text
build obsmap with inflated radius r_occ
clearance_field = EDT(obsmap)
open = {start_state}
while open not empty:
  pop state with minimum f = g + h
  if at goal cell: reconstruct
  for each 8-neighbor motion:
    if occupied/outside: continue
    d = clearance at successor
    g_new = g + move_cost + risk(d) + heading_penalty
    relax successor
fallback: run baseline astar
```

## 3.2 Phase 2: CHOMP + shortcut

Path is resampled and optimized with:

```text
J = lambda_s * sum_i ||q_{i+1} - 2q_i + q_{i-1}||^2
  + lambda_o * sum_i V(d_i)
```

Obstacle potential:

```text
V(d) = exp((m - d) / sigma),  if d < m
V(d) = 0,                      if d >= m
```

Inflated clearance model:

```text
r_safe = r_r + safety_margin + max(alpha * reso, 0.15)
```

Hard projection target during CHOMP:

```text
d_target = r_r + safety_margin + max(hard_pad, 0.25 * reso)
```

## 3.3 Phase 3: straight/arc compression

Dynamic programming minimizes number of primitives:
- straight segment if min distance to obstacles > clearance
- circular arc candidates (3-point + tangent families)

With safety margin:

```text
clearance_phase3 ~= r_r + safety_margin + 0.05
```

---

## 4. `mod_grid_SE.py` (SE(2) Footprint Planner)

## 4.1 Phase 1: SE(2) augmented A*

State:

```text
s = (x_i, y_i, psi_i, m_prev)
```

- `\psi_i` is discretized yaw bin (`_YAW_BINS`). Current default is 72 bins (5°).
- Actions:
  - translation (8-connected, yaw unchanged)
  - in-place yaw change (±1 bin)

Footprint collision model (staged validity check):

Inputs:
- obstacle point cloud `(ox, oy)` from the app (boundary samples + rect fill + thickened polylines)
- optional continuous obstacle geometry (rectangles now; extensible to polygons)
- optional continuous map bounds `(xmin, ymin, xmax, ymax)` from the app

Validity at a candidate SE pose is evaluated in stages:

```text
Stage A (cheap reject):
  Sample footprint boundary points, transform to world, and compute
  d_min = min distance to obstacle point cloud.
  If d_min <= hard_pad: REJECT immediately.

Stage B (exact geometry, only if Stage A passes and Shapely is available):
  Build the full footprint polygon at that pose.
  - Map wall check: footprint polygon must be contained in the map box.
  - Obstacle polygon check: distance(footprint, obstacle) must exceed hard_pad.

Stage C (fallback robustness):
  If needed, check whether any nearby obstacle points lie inside the footprint polygon.
```

Hard occupancy prefilter radius:

```text
r_occ = r_b + safety_margin + max(0.02, 0.1 * reso)
```

where `r_b` is footprint circumradius.

Hard validity threshold:

```text
d_min > max(shape_pad, k * reso) + safety_margin
```

### Phase 1 SE pseudocode

```text
sample polygon boundary points in local frame
build coarse inflated occupancy map using bounding circle
for each candidate SE state:
  transform boundary samples by yaw + translation
  Stage A: reject if d_min(point_cloud) <= hard_pad
  Stage B: if available, run polygon-vs-(map bounds + obstacle polygons) checks
  Stage C: optional point-in-footprint fallback near obstacles
run A* over (x,y,yaw_bin,m_prev)
```

## 4.2 Phase 3: SE(2) primitive compression (constant body twist)

Given two SE nodes `(x_i,y_i,theta_i)` and `(x_j,y_j,theta_j)`:

1. Transform endpoint to node-`i` body frame.
2. Compute relative yaw with wrapped difference.
3. Build constant-body-twist primitive
   ```text
   theta_dot = omega
   p_dot = R(theta) * v_body
   ```
4. Sample the primitive, transform full footprint boundary, validate clearance.
5. DP minimizes primitive count over feasible edges.

If reconstructed yaw at final point differs from target orientation in app-side DF fill, an optional terminal in-place rotation can be appended for visualization continuity.

---

## 5. Yaw Filling in `app.py` for disk `mod_grid`

For non-circle shape display after disk planning:

- `none`: no yaw reconstruction; draw circumscribed disk.
- `linear`: interpolate yaw from start to goal.
- `differential_flatness`: for phase 3 only, request returned primitives and integrate yaw from primitive turn (anchored at start yaw), then optional final self-rotation to match goal yaw.

This is display/post-processing; core disk planner remains disk-based.

---

## 6. Complexity (high-level)

Let:
- `N_xy`: number of grid cells
- `N_yaw`: yaw bins
- `B`: boundary samples on footprint
- `M`: obstacle points
- `K`: path sample count

## 6.1 Disk planner (`mod_grid.py`)

- Phase 1 A*:
  - time: roughly `O(N_xy log N_xy)` in typical sparse cases
  - memory: `O(N_xy)`
- CHOMP:
  - each iter uses path points and local field sampling, approx `O(K)` for field ops; nearest-obstacle checks can be `O(K log M)` with KD-tree (or `O(KM)` fallback)
- Phase 3 DP compression:
  - DP span-limited (default window ~30), so practical near `O(N)` with larger constants; without span limit grows toward `O(N^2)`

## 6.2 SE planner (`mod_grid_SE.py`)

- Phase 1 A* state space:
  - worst-case nodes `~ N_xy * N_yaw * 9`
  - each state check may involve boundary sampling + nearest obstacle query:
    - KD-tree: `~ O(B log M)`
    - brute force fallback: `O(BM)`
- Phase 3 SE DP:
  - span-limited edge checks
  - each candidate edge samples primitive and tests transformed footprint:
    - approx `O(K_e * B * log M)` with KD-tree per edge

---

## 7. Resolution (`reso`) effects

`reso` is one of the most important tuning parameters:

- Smaller `reso`:
  - finer obstacle sampling / grid detail
  - potentially more accurate clearance behavior
  - larger search space and slower runtime

- Larger `reso`:
  - coarser, faster planning
  - may miss narrow features / alter clearance realism

Inflation terms with `reso`, e.g.:

```text
max(alpha * reso, c_min)
```

are used to compensate discretization artifacts and maintain conservative safety margins across different resolutions.

---

## 8. Practical guidance

- Use `mod_grid` first for speed and broad feasibility checks.
- Use `mod_grid_SE` for final shape-aware validation and SE(2)-consistent behavior.
- Keep `safety_margin` nonzero for robust deployment transfer.
- Prefer KD-tree-enabled environment (`scipy`) for performance.

