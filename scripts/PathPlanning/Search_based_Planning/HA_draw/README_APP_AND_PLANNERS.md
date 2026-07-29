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
   - Uses robot radius `rr` (circumradius + safety margin for non-circle shapes).
   - Pipeline:
     - Phase 1: grid A* on `(x,y)`, unit cost, **offline** (full disk bitmap) or **online** (lazy per-cell checks)
     - Phase 3: shortcut + straight/arc primitive compression
   - Phase 2 (CHOMP) disabled in UI.

3. **SE(2) planner (`mod_grid_SE.py`)**
   - Uses full polygon footprint with yaw in state.
   - Pipeline:
     - Phase 1: disk bootstrap (optional fast path) → 3D volume (lazy middle-band SAT) → SE A* on `(x,y,θ)`
     - Phase 3: constant body-twist DP (**legacy**; not yet aligned with 3D volume)
   - Phase 2 not implemented.

**Detailed methodology (phase 1/3, offline/online, equations):** see [`SE2_PHASE1_ASTAR.md`](SE2_PHASE1_ASTAR.md).

---

## 2. `app.py` (Workbench Orchestration)

### 2.1 Planner selection

`app.py` exposes planner choices:

- `grid_astar` (baseline disk)
- `mod_grid` (disk safe+smooth)
- `mod_grid_SE` (full footprint SE(2))
- `hybrid_astar` (car-like planner)

For `mod_grid`:
- `stop_phase` in {1, 3} (phase 2 disabled)
- **disk collision:** `offline` (full bitmap) or `online` (lazy checks) — mod_grid only
- optional yaw filling mode for visualization: `linear`, `differential_flatness` (phase 3)
- phase 3 DP objective: `length` or `min_segments`

For `mod_grid_SE`:
- `stop_phase` in {1, 3}; phase 2 disabled.

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

See [`SE2_PHASE1_ASTAR.md`](SE2_PHASE1_ASTAR.md) §3 (phase 1) and §5 (phase 3) for full detail.

**Phase 1 (current):** grid A* on `(x,y)` only; edge cost 1; Chebyshev heuristic; 8-way + diagonal gate. Collision: **offline** (prebuilt `obsmap`) or **online** (`grid_cell_disk_blocked` + cache). No EDT, no `m_prev`, no fallback A*.

**Phase 3:** greedy shortcut → straight/arc DP with disk-vs-OBB validation (`phase3_polish`).

## 3.1 Legacy note

Older docs described augmented `(x,y,m_prev)` with `c_risk` + `c_heading` and CHOMP phase 2 — removed/disabled.

---

## 4. `mod_grid_SE.py` (SE(2) Footprint Planner)

See [`SE2_PHASE1_ASTAR.md`](SE2_PHASE1_ASTAR.md) §4 (phase 1) and §6 (phase 3 status) for full detail.

**Phase 1:** disk bootstrap → column classify + lazy SAT volume → SE A* on `(x,y,θ_bin)` with 12 edges and 3-cell gate.

**Phase 3:** constant body-twist DP exists but is **not yet updated** for the 3D conservative volume pipeline (next milestone).

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

- Phase 1 A*: \(O(N_{xy} \log N_{xy})\) typical; offline map build \(O(N_{xy} \cdot M)\); online avoids full raster.
- Phase 3 DP: span-limited window (~30), practical near \(O(n)\).

## 6.2 SE planner (`mod_grid_SE.py`)

- Phase 1: \(|V| \approx N_{xy} \cdot N_\theta\); lazy SAT touches only poses visited by A*.
- Phase 3 SE DP: span-limited; footprint sample checks per primitive (legacy path).

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

- Use `mod_grid` (online collision) for fast disk feasibility; `mod_grid_SE` when footprint + yaw matter.
- See [`SE2_PHASE1_ASTAR.md`](SE2_PHASE1_ASTAR.md) for methodology and benchmarks.
- Keep `safety_margin` nonzero for robust deployment transfer.

