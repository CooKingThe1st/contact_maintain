# HA_draw Grid Planner Methodology (`mod_grid` + `mod_grid_SE`)

Design reference for the holonomic disk planner and the SE(2) footprint planner used in `HA_draw/app.py`. Both share the same scenario format, obstacle APIs, and a **feasible phase 1 → polish phase 3** philosophy; they differ in collision geometry and state space.

---

## 1. Shared architecture

```text
                    ┌─────────────────────────────────────────┐
                    │  Scenario: ox/oy point cloud + OBB rects │
                    └────────────────────┬────────────────────┘
                                         │
         ┌───────────────────────────────┴───────────────────────────────┐
         │                                                               │
   mod_grid (disk)                                              mod_grid_SE (footprint)
         │                                                               │
   Phase 1: grid A* on (x,y)                              Phase 1: disk bootstrap (optional fast path)
   collision: offline | online                           then 3D volume + SE A* on (x,y,θ)
         │                                                               │
   Phase 3: shortcut + S/A DP                         Phase 3: linear-yaw S/C DP
   (disk clearance vs OBBs)                                (volume-bin + exact SAT verification)
```

| Aspect | `mod_grid` (disk) | `mod_grid_SE` (footprint) |
|--------|-------------------|---------------------------|
| **Robot model** | Disk radius `rr = circumradius + safety_margin` | Convex footprint parts vs OBB SAT; column classify uses `r_cell = rr + (√2/2)·reso` |
| **Phase 1 goal** | Any feasible grid path, fast | Conservative feasible path in \((x,y,\theta)\) |
| **Phase 1 polish cost** | None (unit grid cost) | Move + rotation bin cost |
| **Phase 3** | Straight `S` + arc `A` primitives | Linear-yaw `S` / `C` DP (body-twist legacy, grayed in UI) |
| **Phase 2** | CHOMP disabled | Not implemented |
| **Collision precompute** | Optional full bitmap (**offline**) or lazy (**online**) | **Always** full disk map + column classify; **lazy SAT** for middle band |

**Design principle:** phase 1 finds a feasible route quickly; phase 3 improves geometry and smoothness with continuous clearance checks. Disk phase 1 no longer applies risk/heading costs — those belonged to an older “smooth in phase 1” design.

---

## 2. Collision checking: offline vs online

### 2.1 Disk planner (`mod_grid`) — UI: “disk collision”

Two modes (non-SE only):

| Mode | When | Work |
|------|------|------|
| **offline** | Default | Rasterize full `obsmap[ix,iy]`: point cloud + OBB disk inflation over all grid cells |
| **online** | Optional radio | No upfront raster; `grid_cell_disk_blocked(gx,gy)` on demand with per-cell cache |

**Offline** uses `base_astar.calc_parameters` + `apply_rect_disk_obstacles_to_obsmap`. Cost is \(O(N_{xy} \cdot M)\) for \(M\) obstacle points — dominates on large maps (e.g. ~660 ms on 14×14 m bottleneck @ reso 0.2).

**Online** builds grid bounds `P` only, then checks each visited cell:

```text
blocked(gx, gy) :=
  ∃ point (ox, oy) : cell_square_disk_hits_point_grid(gx, gy, ox, oy, r_eff/reso)
  ∨ ∃ rect : cell_square_disk_hits_obb(gx, gy, reso, rect, r_eff)
```

Results are cached in `(gx,gy) → bool`. Search pays only for the **reachable tube** (e.g. ~558 checks vs 4900 cells on bottleneck).

**When to use which**

- **online** — large map, narrow exploration, phase 1 only; often 3–20× faster total on bottleneck/buggy cases.
- **offline** — need full `disk_blocked` count, predictable repeat lookups, or simpler debugging.

### 2.2 SE planner (`mod_grid_SE`) — always offline disk map

SE phase 1 **always** builds the full disk map because it is reused for:

1. Disk bootstrap A* (`phase1_augmented_astar_with_meta`, offline only)
2. Column labels in the 3D volume (`disk_column_free`, FREE / TRAPPED / LAZY)
3. Disk BFS heuristic \(h_{xy}\) on the disk-free subgraph

**Lazy SAT (middle band)** applies to **footprint vs OBB** occupancy, not disk raster:

| Column label | Condition | θ occupancy |
|--------------|-----------|---------------|
| **FREE** | `disk_column_free` (never enters OBB classify), **or** disk-blocked but disk-clear of all rects in window, **or** disk-reachable **and** disk-column-free (`apply_disk_reachable_columns`) | all θ free |
| **TRAPPED** | Disk engulfed by an OBB | all θ blocked |
| **LAZY** | Disk-blocked in an OBB window, not engulfed, not clear | **unknown** until A* queries `(x,y,θ)` |

Classification iterates only **`disk_blocked ∩ OBB_influence_windows`**, not every blocked cell on the map.

During SE A*, `SE2GridVolume.is_occupied` runs SAT once per queried `(x,y,θ)` and caches the result. Upfront volume build only **classifies** columns (~10 ms); bulk θ SAT is avoided.

---

## 3. Phase 1 — disk (`mod_grid`)

**Entry:** `phase1_augmented_astar` / `phase1_augmented_astar_with_meta`  
**File:** `HybridAstarPlanner/mod_grid.py`

### 3.1 Pipeline

```text
1. Build grid para P (and obsmap if offline)
2. Grid A* from start cell to goal cell
3. Return polyline of cell centers (or empty if unreachable)
```

No fallback plain A*, no EDT, no `m_prev` augmentation.

### 3.2 State space

\[
s = (x,\ y), \quad P.\texttt{minx} < x < P.\texttt{maxx},\ P.\texttt{miny} < y < P.\texttt{maxy}
\]

\[
|V| \approx N_{xy} = (x_{\max}-x_{\min}-1)(y_{\max}-y_{\min}-1)
\]

(9× smaller than the old `(x,y,m_prev)` augmented disk search.)

### 3.3 Edges

8-connected grid moves with **diagonal gate**: for \((dx,dy)\) diagonal, both cardinals \((x+dx,y)\) and \((x,y+dy)\) must be free.

### 3.4 Edge cost

\[
\Delta g = 1 \quad \text{(cardinal and diagonal)}
\]

### 3.5 Heuristic

Chebyshev distance (admissible for unit-cost 8-way):

\[
h(s) = \max(|x - g_x|,\ |y - g_y|)
\]

### 3.6 SE bootstrap metadata

`phase1_augmented_astar_with_meta` also returns:

- `reachable_xy` — disk search **closed set** (cells settled by A*)
- `reachable_cost` — best \(g\) per cell (`m_in` slot fixed to `-1` for API compat)

SE uses this for disk-reachable column marking in the volume; open-list frontiers are excluded.

### 3.7 Timing log (app / CLI)

```text
[timing] mod_grid phase1 pipeline (disk grid A*)
  design: state=(x,y)  edges=8-way+diag gate  cost=1  collision=offline|online
  1 disk map: …ms  grid=…  disk_blk=…
     online collision: checks=…  cache_hits=…   # online only
  2 grid A*: …ms  expanded=…  goal=YES|NO
```

CLI: `python mod_grid.py scenario.json --stop_phase 1 [--disk_collision online|offline]`

---

## 4. Phase 1 — SE(2) (`mod_grid_SE`)

**Entry:** `phase1_augmented_astar_se2` / `astar_planning(..., stop_phase=1)`  
**Files:** `HybridAstarPlanner/mod_grid_SE.py`, `se2_grid_volume.py`  
**Obstacles (v1):** point cloud + OBB rects only (`obstacle_polygons` ignored in phase 1).

### 4.1 Pipeline

```text
1. Disk A* with meta (offline bitmap, same as mod_grid offline bootstrap)
   → if disk reaches goal: return disk (x,y) path + linear yaw fill (FAST PATH)
2. Volume build: classify columns; lazy SAT on demand (no bulk middle-band θ loop)
3. SE A* on (x, y, θ_bin) with 3-cell edge gate
```

### 4.2 State space

\[
s = (x,\ y,\ \theta_{\text{bin}}), \quad \theta_{\text{bin}} \in \{0,\ldots,N_\theta-1\},\ N_\theta = 36
\]

No `m_prev` (unlike legacy 2D disk design).

\[
|V| = N_{xy} \cdot N_\theta
\]

### 4.3 Edges

**12 neighbors:** 4 cardinal \((dx,dy)\) × \(d\theta \in \{-1,0,+1\}\) bins.

\[
(dx,dy) \in \{(1,0),(-1,0),(0,1),(0,-1)\}
\]

No diagonal XY; no pure rotation (every edge moves one XY cell).

### 4.4 3-cell feasibility gate

Transition \((x_1,y_1,\theta_1) \to (x_2,y_2,\theta_2)\) requires corners A, B, C free:

| Corner | Pose |
|--------|------|
| A | \((x_1, y_1, \theta_2)\) |
| B | \((x_2, y_2, \theta_2)\) |
| C | \((x_2, y_2, \theta_1)\) |

Start \((x_1,y_1,\theta_1)\) is already known free. Pose free if `disk_column_free(x,y)` **or** `not occ[x,y,θ]` (lazy SAT fills `occ` on first query).

Implementation: `se2_grid_volume.se2_edge_free_3cell`.

### 4.5 Edge cost

\[
\Delta g = c_{\text{move}} + c_{\text{rot}} \cdot |d\theta|
\]

| Symbol | Value | Meaning |
|--------|-------|---------|
| \(c_{\text{move}}\) | `1.0` | Every cardinal XY step |
| \(c_{\text{rot}}\) | `0.35` | Per non-zero yaw-bin change |

### 4.6 Heuristic (admissible)

\[
h(s) = h_{xy}(s) + w(x,y)\, h_\theta(s)
\]

**XY — disk BFS:** cardinal BFS on disk-free graph from goal; \(h_{xy} = c_{\text{move}} \cdot d_{\text{disk}}(x,y)\). Unreachable → Manhattan fallback.

**Yaw — disk-column gated:**

\[
h_\theta =
\begin{cases}
0 & \text{if goal disk column free} \\
c_{\text{rot}} \cdot \max(0,\ \Delta\theta_{\min} - m_{\text{steps}}) & \text{otherwise}
\end{cases}
\]

where \(m_{\text{steps}} = d_{\text{disk}}(x,y)\) is the same cardinal disk-BFS step count used in \(h_{xy}\) (yaw bins “affordable” while walking to goal).

\[
w(x,y) = 0 \text{ if } disk\_column\_free(x,y),\ \text{else } 1
\]

### 4.7 Goal test

\[
(x,y) = (g_x, g_y) \land \Delta\theta_{\min} \le \texttt{\_YAW\_GOAL\_TOL\_BINS}\ (= 0)
\]

### 4.8 Volume build (lazy middle band)

```text
for (ix, iy) in disk_blocked ∩ OBB_windows:
    if engulfed:     column ← TRAPPED, occ[:,:,θ] ← blocked
    elif clear:      column ← FREE
    else:            column ← LAZY, occ[:,:,θ] ← UNKNOWN

apply_disk_reachable_columns(reachable_xy ∩ disk_column_free)

# During SE A*:
is_occupied(gx, gy, t):
    if FREE column: return False
    if TRAPPED: return True
    if LAZY and occ unknown: SAT footprint at θ; cache; return
```

### 4.9 Timing log

```text
[timing] mod_grid_SE pipeline (stop phase 1)
  1 disk A*: …ms  closed=…
  2 volume: …ms  classify=…  lazy_sat=…  lazy=…  lazy_queries=…
  3 SE A*: …ms  expanded=…
```

---

## 5. Phase 3 — disk (`mod_grid`)

**Entry:** `phase3_polish` → `phase3_min_segments`  
**File:** `HybridAstarPlanner/mod_grid.py`  
**UI:** stop phase **3**; DP objective **shortest length** or **min primitive count**

### 5.1 Pipeline

```text
1. Greedy line-of-sight shortcut on phase-1 polyline
2. DP over sparse waypoints with primitives:
     - Straight segment (disk clearance along segment)
     - Circular arc (3-point circumcircle and/or tangent-discrete radius sweep)
3. Validate full primitive chain; fallback chain:
     DP invalid → shortcut polyline → phase-1 polyline
```

### 5.2 Primitives

| Type | Validation |
|------|------------|
| **Straight** `S` | `_segment_disk_clear` — hierarchical funnel vs point cloud + OBB rects |
| **Arc** `A` | Arc samples at `_ARC_POINTS_PER_RAD`; disk center must stay clear |

Arc candidates: (i) 3-point circumcircle through \((i, k, j)\) with mid \(k \in \{i{+}1, \lfloor(i{+}j)/2\rfloor, j{-}1\}\); (ii) tangent-based discrete-radius arcs (`_arc_from_start_tangent_discrete`, geometric \(r\) growth). Default nominal radius `_ARC_RADIUS = 0.35` m (clamped by edge length).

**Export:** disk phase-3 tuples use `S` / `A`; `scenario_planner_bridge.planner_primitives_to_df` maps both `A` and `C` to `kind: "arc"` in `.planned.json`.

### 5.3 DP objective

| Mode | Edge cost |
|------|-----------|
| `length` | Euclidean / arc length |
| `min_segments` | 1 per primitive |

Span window `max_span = min(30, n-1)` limits \(O(n \cdot \text{span})\) edges.

### 5.4 Clearance

\[
r_{\text{check}} = r_r + \text{safety\_margin}
\]

Uses `DiskValidationContext` from `scenario_obstacles.build_disk_validation_context`.

---

## 6. Phase 3 — SE(2) (`mod_grid_SE`)

**Entry:** `astar_planning(..., stop_phase=3)` → `phase3_interp_yaw_dp` (default) or legacy `phase3_min_segments`  
**Files:** `HybridAstarPlanner/mod_grid_SE.py`, `se2_grid_volume.py` (`pose_world_blocked`)

### 6.1 Pipeline

```text
Phase 1 SE A* → same SE2GridVolume instance (return_volume=True)
DP on spine (px, py, pyaw), span ≤ 30, no shortcut
Primitives:
  S — straight chord; θ(t) linear in parameter
  C — arc through (i, mid, j); θ linear in arc length
Verification — dense samples along primitive → volume.pose_world_blocked(...)
Fallback — unchanged phase-1 path if DP fails
```

**DP objective:** `length` (default) or `min_segments` — same modes as disk phase 3.

**Export:** SE phase-3 tuples use `S` / `C`; `.planned.json` segments are `kind: "line"` or `kind: "arc"` with `theta0` / `theta1` endpoints (`planner_primitives_to_df`).

### 6.2 Collision check mode (app radio)

| Mode | Check | Notes |
|------|-------|-------|
| `volume_bin` | floor/ceil cell centers + floor/ceil θ bins → `is_occupied` (lazy SAT on UNKNOWN); then exact SAT if bracketing bins are free but disk columns are not | Conservative certificate for continuous phase-3 samples |
| `sat_direct` | SAT at exact `(x, y, θ)` against all prepared rects | Less conservative; can accept longer compressed edges |

`volume_bin` may produce a denser displayed path (more emitted samples along accepted primitives) while `sat_direct` may fall back to the phase-1 spine when exact checks reject shortcuts.

### 6.3 Legacy body-twist DP

`phase3_min_segments` (Shapely / point clearance) remains in code but is **disabled in the UI**. It does not use the phase-1 volume and is kept only for comparison.

---

## 7. Reference benchmarks (reso 0.2, phase 1)

Representative wall times (not pinned CI numbers; re-run on your machine for citations). Paths may differ: disk often **no path** when \(r_r\) too large for gap.

| Scenario | mod_grid offline | mod_grid online | mod_grid_SE |
|----------|------------------|-----------------|-------------|
| `rectObs_scenario_bottleneck.json` | ~670 ms | ~100 ms | ~5 s |
| `rectObs_scenario_buggy_case.json` | ~80 ms | ~26 ms | ~0.8 s |

SE pays disk map twice today (disk bootstrap + volume disk map) — deduplication is a future optimization.

---

## 8. Files and tuning

| File | Role |
|------|------|
| `HybridAstarPlanner/mod_grid.py` | Disk phase 1 A*, phase 3 DP, collision modes |
| `HybridAstarPlanner/mod_grid_SE.py` | SE phase 1 A*, `phase3_interp_yaw_dp`, legacy body-twist DP |
| `se2_grid_volume.py` | Volume classify, lazy SAT, `se2_edge_free_3cell` |
| `scenario_obstacles.py` | OBB disk raster, `grid_cell_disk_blocked`, validation funnel |
| `scenario_planner_bridge.py` | Export pair (scenario + `.planned.json`), `planner_primitives_to_df` (`S`/`A` or `S`/`C` → line/arc) |
| `app.py` | Planner UI, timing logs, disk collision + SE phase-3 collision radios, export |

### SE phase 1 knobs

| Constant | Default | Effect |
|----------|---------|--------|
| `_C_MOVE` | 1.0 | Cardinal step cost |
| `_C_ROT` | 0.35 | Rotation penalty per bin |
| `_YAW_BINS` | 36 | 10° discretization |
| `_YAW_GOAL_TOL_BINS` | 0 | Exact goal yaw bin |

### Disk phase 1 knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `disk_collision_mode` | `offline` | Full bitmap vs lazy checks |
| Edge cost | 1 | Uniform grid steps |
| Heuristic | Chebyshev | Admissible for unit 8-way |

### SE phase 3 knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `se_p3_primitive` | `linear_yaw_dp` | Default DP; legacy `body_twist` in code only |
| `se_p3_collision_mode` | `volume_bin` | Bracketing bins + lazy SAT; or `sat_direct` |
| `dp_objective` | `length` | Or `min_segments` (primitive count) |

---

## 9. Related docs

- `README_APP_AND_PLANNERS.md` — app orchestration, scenario JSON, UI overview (high level)
- `HybridAstarPlanner/readme.md` — package-level notes
