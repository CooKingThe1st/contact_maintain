"""
Holonomic disk planner (mod_grid):

Phase 1 — Grid A* on (x, y) with unit edge cost
    State: (grid_x, grid_y) only — feasible path fast; phase 3 polishes geometry.
    Edge cost: 1 per step (cardinal and diagonal).
    Collision: offline (prebuilt disk bitmap) or online (lazy per-cell checks + cache).

Phase 3 — Path polishing (shortcut + primitive DP)
    Greedy line-of-sight shortcut on the phase-1 polyline, then DP over straight segments and
    circular arcs validated analytically against OBB obstacles (hierarchical spatial funnel).

Phase 2 (CHOMP) is deprecated: gradient-based smoothing is a poor fit for this 2D disk-on-grid
problem and is not exposed in the UI. ``stop_phase=2`` raises ``NotImplementedError``.
Legacy ``phase2_chomp`` remains in this file for reference only.
"""

from __future__ import annotations

import json
import heapq
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Ensure the sibling `HybridAstarPlanner` package from `scripts/MotionPlanning/HybridAstarPlanner`
# is importable when running this file standalone (outside `HA_draw/app.py`).
_THIS_DIR = Path(__file__).resolve().parent
_MOTION_BASE_DIR: Optional[Path] = None
for parent in [_THIS_DIR, *_THIS_DIR.parents]:
    candidate = parent / "scripts" / "MotionPlanning"
    if (candidate / "HybridAstarPlanner").exists():
        _MOTION_BASE_DIR = candidate
        break
if _MOTION_BASE_DIR is not None:
    sys.path.insert(0, str(_MOTION_BASE_DIR))

import HybridAstarPlanner.astar as base_astar

try:
    from scipy.ndimage import distance_transform_edt

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    # Optional, for fast nearest-obstacle queries in CHOMP.
    from scipy.spatial import cKDTree as _cKDTree  # type: ignore

    _HAS_KDTREE = True
except Exception:
    _HAS_KDTREE = False


# --- Tunables (world meters / cost units) ---------------------------------

# Hard inflation for occupancy (same spirit as baseline + margin)
_INFLATION_RESO_FACTOR = 0.5

# Risk: clearance d_meters to nearest obstacle cell; three bands [0,d1), [d1,d2), [d2,d3), else 0
_RISK_D1 = 0.35
_RISK_D2 = 0.85
_RISK_D3 = 1.6
_RISK_P1 = 3.0
_RISK_P2 = 12.0
_RISK_P3 = 55.0

# Heading: weight on normalized turn angle in [0, pi]
_HEADING_WEIGHT = 0.45

# CHOMP
_CHOMP_ITERS = 90
_CHOMP_STEP = 0.12
# Smoothness uses an acceleration objective (second difference); keep weights moderate.
_CHOMP_LAMBDA_SMOOTH = 1.4
_CHOMP_LAMBDA_OBS = 3.0
_CHOMP_OBS_MARGIN = 0.55  # meters; push path outward if closer
_CHOMP_OBS_SIGMA = 0.35  # meters; obstacle potential falloff (smaller = sharper)
_CHOMP_OBS_GRAD_CLIP = 8.0  # clip obstacle gradient magnitude per point (stability)
_CHOMP_HARD_CLEARANCE_PAD = 0.25  # meters; extra clearance target for hard projection
_CHOMP_POINTS = 64

# Phase 3 — arc primitives (straight + circular arc DP)
_ARC_RADIUS = 0.35  # m; clamped down if edges are short
_ARC_MIN_TURN = math.radians(4.0)  # below this |angle|, keep sharp vertex
_ARC_POINTS_PER_RAD = 10.0  # arc samples for dense polyline output

# Phase 3 DP objective modes
DP_OBJECTIVE_LENGTH = "length"
DP_OBJECTIVE_MIN_SEGMENTS = "min_segments"


M_PREV_NONE = -1
_NUM_MOTION = 8

DISK_COLLISION_OFFLINE = "offline"
DISK_COLLISION_ONLINE = "online"


def _disk_validation_ctx(
    clearance: float,
    reso: float,
    ox: List[float],
    oy: List[float],
    obstacle_rects=None,
):
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import DiskValidationContext, build_disk_validation_context

    rects = list(obstacle_rects) if obstacle_rects else []
    return build_disk_validation_context(clearance, reso, ox, oy, rects)


def _primitive_length(typ: str, params: dict) -> float:
    if typ == "S":
        return math.hypot(
            float(params["x1"]) - float(params["x0"]),
            float(params["y1"]) - float(params["y0"]),
        )
    return abs(float(params["sweep"])) * max(float(params["r"]), 1e-9)


def _get_motion() -> List[List[int]]:
    return base_astar.get_motion()


def _u_cost(m: List[int]) -> float:
    return base_astar.u_cost(m)


def _angle_between_moves(m_prev: int, m_new: int, motion: List[List[int]]) -> float:
    if m_prev < 0 or m_prev >= len(motion) or m_new < 0 or m_new >= len(motion):
        return 0.0
    dx0, dy0 = motion[m_prev][0], motion[m_prev][1]
    dx1, dy1 = motion[m_new][0], motion[m_new][1]
    # Undirected rays: prefer continuity in *direction of travel*; compare vectors as-is.
    n0 = math.hypot(dx0, dy0)
    n1 = math.hypot(dx1, dy1)
    if n0 < 1e-9 or n1 < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (dx0 * dx1 + dy0 * dy1) / (n0 * n1)))
    return math.acos(c)


def _risk_cost(d_clear: float) -> float:
    if d_clear < _RISK_D1:
        return _RISK_P3
    if d_clear < _RISK_D2:
        return _RISK_P2
    if d_clear < _RISK_D3:
        return _RISK_P1
    return 0.0


def _build_clearance_meters(obsmap: List[List[bool]], reso: float) -> np.ndarray:
    """2D array shape (xw, yw): EDT clearance in meters from cell center to nearest obstacle cell."""
    occ = np.array(obsmap, dtype=bool)
    if _HAS_SCIPY:
        dist_cells = distance_transform_edt(~occ)
        return dist_cells.astype(np.float64) * float(reso)
    # Fallback: Manhattan BFS distance in cells
    xw, yw = occ.shape
    dist = np.full((xw, yw), np.inf, dtype=np.float64)
    from collections import deque

    q = deque()
    for i in range(xw):
        for j in range(yw):
            if occ[i, j]:
                dist[i, j] = 0.0
                q.append((i, j))
    while q:
        i, j = q.popleft()
        base = dist[i, j]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < xw and 0 <= nj < yw and not occ[ni, nj]:
                nd = base + float(reso)
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    q.append((ni, nj))
    return dist


def _grid_state_index(x: int, y: int, P: base_astar.Para) -> int:
    return (y - P.miny) * P.xw + (x - P.minx)


def _state_index(x: int, y: int, m_in: int, P: base_astar.Para) -> int:
    """Legacy augmented index (x,y,m_in); retained for reference only."""
    cell = (y - P.miny) * P.xw + (x - P.minx)
    mk = m_in + 1
    return cell * 9 + mk


class _DiskNode:
    __slots__ = ("x", "y", "cost", "p_ind")

    def __init__(self, x: int, y: int, cost: float, p_ind: int):
        self.x = x
        self.y = y
        self.cost = cost
        self.p_ind = p_ind


class _AugNode:
    __slots__ = ("x", "y", "m_in", "cost", "p_ind")

    def __init__(self, x: int, y: int, m_in: int, cost: float, p_ind: int):
        self.x = x
        self.y = y
        self.m_in = m_in
        self.cost = cost
        self.p_ind = p_ind  # parent state_index


def _heuristic(x: int, y: int, gx: int, gy: int) -> float:
    """Chebyshev distance — admissible for 8-way unit-cost moves."""
    return float(max(abs(x - gx), abs(y - gy)))


def _calc_grid_para(ox_g: List[float], oy_g: List[float], reso: float) -> base_astar.Para:
    minx, miny = round(min(ox_g)), round(min(oy_g))
    maxx, maxy = round(max(ox_g)), round(max(oy_g))
    xw, yw = maxx - minx, maxy - miny
    return base_astar.Para(minx, miny, maxx, maxy, xw, yw, float(reso), _get_motion())


def _explored_xy_from_disk_nodes(
    closed: Dict[int, _DiskNode],
    open_entries: Dict[int, _DiskNode],
) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for node in closed.values():
        out.add((node.x, node.y))
    for node in open_entries.values():
        out.add((node.x, node.y))
    return out


def _explored_xy_from_nodes(
    closed: Dict[int, _AugNode],
    open_entries: Dict[int, _AugNode],
) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for node in closed.values():
        out.add((node.x, node.y))
    for node in open_entries.values():
        out.add((node.x, node.y))
    return out


def _reconstruct_phase1_path(
    goal_idx_found: int,
    closed: Dict[int, _DiskNode],
    reso: float,
) -> Tuple[List[float], List[float]]:
    path_grid: List[Tuple[int, int]] = []
    walk = goal_idx_found
    while walk != -1:
        node = closed[walk]
        path_grid.append((node.x, node.y))
        walk = node.p_ind
    path_grid.reverse()

    from scenario_obstacles import grid_cell_center_world

    pathx = [grid_cell_center_world(x, y, reso)[0] for x, y in path_grid]
    pathy = [grid_cell_center_world(x, y, reso)[1] for x, y in path_grid]
    return pathx, pathy


def _path_reaches_goal_grid(
    px: List[float],
    py: List[float],
    gx_i: int,
    gy_i: int,
    reso: float,
) -> bool:
    if not px:
        return False
    return round(px[-1] / reso) == gx_i and round(py[-1] / reso) == gy_i


def _reachable_xy_from_closed(closed: Dict[int, _DiskNode]) -> Set[Tuple[int, int]]:
    return {(node.x, node.y) for node in closed.values()}


def _best_disk_cost_by_xy(closed: Dict[int, _DiskNode]) -> Dict[Tuple[int, int], Tuple[float, int]]:
    """Minimum grid A* cost per ``(gx, gy)``; ``m_in`` slot kept at -1 for SE bootstrap compat."""
    out: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for node in closed.values():
        key = (node.x, node.y)
        prev = out.get(key)
        if prev is None or node.cost < prev[0]:
            out[key] = (float(node.cost), M_PREV_NONE)
    return out


def _ms(seconds: float) -> float:
    return 1000.0 * float(seconds)


def format_disk_phase1_report(
    timing: Dict[str, float],
    *,
    meta: Optional[Dict[str, object]] = None,
    wall_s: Optional[float] = None,
) -> List[str]:
    """Human-readable Phase 1 disk pipeline lines for the app status log."""
    meta = meta or {}
    lines: List[str] = []
    lines.append("[timing] mod_grid phase1 pipeline (disk grid A*)")
    mode = str(meta.get("collision_mode", timing.get("collision_mode_label", "offline")))
    lines.append(
        f"  design: state=(x,y)  edges=8-way+diag gate  cost=1  collision={mode}"
    )

    meta_bits = []
    if "reso" in meta:
        meta_bits.append(f"reso={meta['reso']}")
    if "map_w" in meta and "map_h" in meta:
        meta_bits.append(f"map={meta['map_w']}x{meta['map_h']}m")
    if "rr" in meta:
        meta_bits.append(f"rr={meta['rr']}")
    if "safety_margin" in meta:
        meta_bits.append(f"margin={meta['safety_margin']}")
    if "obstacle_pts" in meta:
        meta_bits.append(f"ox={meta['obstacle_pts']}")
    if meta_bits:
        lines.append("  setup: " + " ".join(meta_bits))

    map_ms = _ms(timing.get("disk_map_s", 0.0))
    astar_ms = _ms(timing.get("astar_s", 0.0))
    grid_xy = int(timing.get("grid_xy", 0.0))
    disk_blk = int(timing.get("disk_blocked", 0.0))
    lines.append(
        f"  1 disk map: {map_ms:.0f}ms grid={grid_xy} disk_blk={disk_blk}"
    )
    online_checks = int(timing.get("online_checks", 0.0))
    online_cache = int(timing.get("online_cache_hits", 0.0))
    if online_checks > 0 or online_cache > 0:
        lines.append(
            f"     online collision: checks={online_checks} cache_hits={online_cache}"
        )

    goal = "YES" if timing.get("goal_reached", 0.0) >= 0.5 else "NO"
    expanded = int(timing.get("expanded_states", 0.0))
    rate = float(timing.get("states_per_s", 0.0))
    open_max = int(timing.get("max_open", 0.0))
    stale = int(timing.get("stale_pops", 0.0))
    rate_s = f" {rate:.0f} states/s" if rate > 0.0 else ""
    lines.append(
        f"  2 grid A*: {astar_ms:.0f}ms expanded={expanded}{rate_s} "
        f"open_max={open_max} stale_pops={stale} goal={goal}"
    )

    path_pts = int(timing.get("path_pts", 0.0))
    plan_ms = _ms(timing.get("total_s", 0.0))
    wall_ms = _ms(wall_s) if wall_s is not None else plan_ms
    lines.append(f"  path {path_pts} pts  TOTAL: planner={plan_ms:.0f}ms wall={wall_ms:.0f}ms")
    return lines


def _fill_disk_phase1_timing(
    timing: Dict[str, float],
    *,
    t_all: float,
    t_map0: float,
    t_map1: float,
    t_astar0: float,
    t_astar1: float,
    goal_reached: bool,
    expanded: int,
    heap_pops: int,
    stale_pops: int,
    max_open: int,
    path_pts: int,
    grid_xy: int,
    disk_blocked: int,
    collision_mode: str,
    online_checks: int = 0,
    online_cache_hits: int = 0,
) -> None:
    astar_s = t_astar1 - t_astar0
    timing.clear()
    timing.update(
        {
            "disk_map_s": t_map1 - t_map0,
            "astar_s": astar_s,
            "goal_reached": 1.0 if goal_reached else 0.0,
            "expanded_states": float(expanded),
            "heap_pops": float(heap_pops),
            "stale_pops": float(stale_pops),
            "max_open": float(max_open),
            "states_per_s": float(expanded) / astar_s if astar_s > 1e-9 and expanded > 0 else 0.0,
            "path_pts": float(path_pts),
            "grid_xy": float(grid_xy),
            "disk_blocked": float(disk_blocked),
            "collision_mode_label": collision_mode,
            "online_checks": float(online_checks),
            "online_cache_hits": float(online_cache_hits),
            "total_s": time.perf_counter() - t_all,
        }
    )


def phase1_augmented_astar_with_meta(
    sx: float,
    sy: float,
    gx: float,
    gy: float,
    ox: List[float],
    oy: List[float],
    reso: float,
    rr: float,
    safety_margin: float = 0.0,
    obstacle_rects=None,
    timing: Optional[Dict[str, float]] = None,
    disk_collision_mode: str = DISK_COLLISION_OFFLINE,
) -> Tuple[
    List[float],
    List[float],
    bool,
    Set[Tuple[int, int]],
    Dict[Tuple[int, int], Tuple[float, int]],
]:
    """
    Phase 1 disk grid A* plus search metadata for SE(2) bootstrapping.

    Returns ``(px, py, goal_reached, reachable_xy, reachable_cost)``.

    ``disk_collision_mode`` is ``offline`` (full bitmap) or ``online`` (lazy checks).
    SE bootstrap always uses ``offline``; non-SE mod_grid may choose ``online``.
    """
    from scenario_obstacles import clamp_safety_margin, grid_cell_disk_blocked

    t_all = time.perf_counter()
    safety_margin = clamp_safety_margin(safety_margin)
    motion = _get_motion()
    occ_rr = float(rr) + float(safety_margin)
    r_eff_grid = occ_rr / float(reso)
    rects_list = list(obstacle_rects) if obstacle_rects else []

    mode = str(disk_collision_mode).lower().strip()
    if mode not in (DISK_COLLISION_OFFLINE, DISK_COLLISION_ONLINE):
        raise ValueError(f"disk_collision_mode must be 'offline' or 'online' (got {disk_collision_mode!r})")
    online = mode == DISK_COLLISION_ONLINE

    t_map0 = time.perf_counter()
    ox_g = [float(x) / reso for x in ox]
    oy_g = [float(y) / reso for y in oy]

    obsmap: Optional[List[List[bool]]] = None
    if online:
        P = _calc_grid_para(ox_g, oy_g, reso)
        disk_blocked = 0
    else:
        P, obsmap = base_astar.calc_parameters(ox_g, oy_g, occ_rr, reso)
        if rects_list:
            base_astar._apply_rect_disk_obstacles(obsmap, P, rects_list, occ_rr, reso)
        disk_blocked = sum(1 for row in obsmap for cell in row if cell)
    t_map1 = time.perf_counter()
    grid_xy = int(P.xw * P.yw)

    sx_i, sy_i = round(sx / reso), round(sy / reso)
    gx_i, gy_i = round(gx / reso), round(gy / reso)

    occ_cache: Dict[Tuple[int, int], bool] = {}
    online_checks = 0
    online_cache_hits = 0

    def _in_bounds(x: int, y: int) -> bool:
        return P.minx < x < P.maxx and P.miny < y < P.maxy

    def _cell_blocked(x: int, y: int) -> bool:
        nonlocal online_checks, online_cache_hits
        if not _in_bounds(x, y):
            return True
        if obsmap is not None:
            return bool(obsmap[x - P.minx][y - P.miny])
        key = (int(x), int(y))
        if key in occ_cache:
            online_cache_hits += 1
            return occ_cache[key]
        online_checks += 1
        blocked = grid_cell_disk_blocked(
            key[0],
            key[1],
            ox_g=ox_g,
            oy_g=oy_g,
            r_eff_grid=r_eff_grid,
            r_eff_world=occ_rr,
            reso=reso,
            rects=rects_list,
        )
        occ_cache[key] = blocked
        return blocked

    def _diagonal_clear(x: int, y: int, dx: int, dy: int) -> bool:
        if dx == 0 or dy == 0:
            return True
        for nx, ny in ((x + dx, y), (x, y + dy)):
            if not _in_bounds(nx, ny) or _cell_blocked(nx, ny):
                return False
        return True

    if not _in_bounds(sx_i, sy_i):
        if timing is not None:
            _fill_disk_phase1_timing(
                timing,
                t_all=t_all,
                t_map0=t_map0,
                t_map1=t_map1,
                t_astar0=t_map1,
                t_astar1=t_map1,
                goal_reached=False,
                expanded=0,
                heap_pops=0,
                stale_pops=0,
                max_open=0,
                path_pts=0,
                grid_xy=grid_xy,
                disk_blocked=disk_blocked,
                collision_mode=mode,
                online_checks=online_checks,
                online_cache_hits=online_cache_hits,
            )
        return [], [], False, set(), {}

    n_start = _DiskNode(sx_i, sy_i, 0.0, -1)
    start_idx = _grid_state_index(sx_i, sy_i, P)

    open_entries: Dict[int, _DiskNode] = {start_idx: n_start}
    closed: Dict[int, _DiskNode] = {}
    pq: List[Tuple[float, int]] = []
    heapq.heappush(pq, (_heuristic(sx_i, sy_i, gx_i, gy_i), start_idx))

    goal_idx_found: Optional[int] = None
    heap_pops = 0
    stale_pops = 0
    max_open = 1

    t_astar0 = time.perf_counter()
    while pq:
        heap_pops += 1
        _, idx = heapq.heappop(pq)
        if idx in closed:
            stale_pops += 1
            continue
        if idx not in open_entries:
            continue
        cur = open_entries.pop(idx)
        closed[idx] = cur
        max_open = max(max_open, len(open_entries) + 1)

        if cur.x == gx_i and cur.y == gy_i:
            goal_idx_found = idx
            break

        for mv in motion:
            dx, dy = int(mv[0]), int(mv[1])
            nx, ny = cur.x + dx, cur.y + dy
            if not _in_bounds(nx, ny) or _cell_blocked(nx, ny):
                continue
            if not _diagonal_clear(cur.x, cur.y, dx, dy):
                continue

            g_new = cur.cost + 1.0
            child_idx = _grid_state_index(nx, ny, P)
            if child_idx in closed:
                continue
            h = _heuristic(nx, ny, gx_i, gy_i)
            if child_idx in open_entries:
                if g_new < open_entries[child_idx].cost:
                    open_entries[child_idx] = _DiskNode(nx, ny, g_new, idx)
                    heapq.heappush(pq, (g_new + h, child_idx))
            else:
                open_entries[child_idx] = _DiskNode(nx, ny, g_new, idx)
                heapq.heappush(pq, (g_new + h, child_idx))
    t_astar1 = time.perf_counter()

    reachable_xy = _reachable_xy_from_closed(closed)
    reachable_cost = _best_disk_cost_by_xy(closed)

    if online and online_checks > 0:
        disk_blocked = sum(1 for v in occ_cache.values() if v)

    goal_reached = goal_idx_found is not None
    if goal_reached:
        px, py = _reconstruct_phase1_path(goal_idx_found, closed, reso)
    else:
        px, py = [], []

    if timing is not None:
        _fill_disk_phase1_timing(
            timing,
            t_all=t_all,
            t_map0=t_map0,
            t_map1=t_map1,
            t_astar0=t_astar0,
            t_astar1=t_astar1,
            goal_reached=goal_reached,
            expanded=len(closed),
            heap_pops=heap_pops,
            stale_pops=stale_pops,
            max_open=max_open,
            path_pts=len(px),
            grid_xy=grid_xy,
            disk_blocked=disk_blocked,
            collision_mode=mode,
            online_checks=online_checks,
            online_cache_hits=online_cache_hits,
        )
    return px, py, goal_reached, reachable_xy, reachable_cost


def phase1_augmented_astar(
    sx: float,
    sy: float,
    gx: float,
    gy: float,
    ox: List[float],
    oy: List[float],
    reso: float,
    rr: float,
    safety_margin: float = 0.0,
    obstacle_rects=None,
    timing: Optional[Dict[str, float]] = None,
    disk_collision_mode: str = DISK_COLLISION_OFFLINE,
) -> Tuple[List[float], List[float]]:
    px, py, _reached, _reachable, _cost = phase1_augmented_astar_with_meta(
        sx,
        sy,
        gx,
        gy,
        ox,
        oy,
        reso,
        rr,
        safety_margin=safety_margin,
        obstacle_rects=obstacle_rects,
        timing=timing,
        disk_collision_mode=disk_collision_mode,
    )
    return px, py


# --- Phase 2: CHOMP helpers ------------------------------------------------


def _resample_polyline(px: List[float], py: List[float], n: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(px) < 2:
        x = np.array(px, dtype=np.float64)
        y = np.array(py, dtype=np.float64)
        return x, y
    pts = np.column_stack([px, py]).astype(np.float64)
    d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-9:
        return np.full(n, px[0]), np.full(n, py[0])
    targets = np.linspace(0.0, total, n)
    qx = np.zeros(n, dtype=np.float64)
    qy = np.zeros(n, dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        if j >= len(s) - 1:
            qx[i], qy[i] = pts[-1]
            continue
        denom = s[j + 1] - s[j]
        if denom < 1e-12:
            qx[i], qy[i] = pts[j]
        else:
            a = (t - s[j]) / denom
            qx[i] = (1 - a) * pts[j, 0] + a * pts[j + 1, 0]
            qy[i] = (1 - a) * pts[j, 1] + a * pts[j + 1, 1]
    return qx, qy


def _world_to_grid_ix(x: float, y: float, P: base_astar.Para, reso: float) -> Tuple[float, float]:
    gx = x / reso - P.minx
    gy = y / reso - P.miny
    return gx, gy


def _sample_clearance_and_grad(
    x: float,
    y: float,
    clearance_m: np.ndarray,
    P: base_astar.Para,
    reso: float,
) -> Tuple[float, float, float]:
    """Bilinear sample of clearance + approximate gradient (d/dx, d/dy) in world meters."""
    gx, gy = _world_to_grid_ix(x, y, P, reso)
    xw, yw = clearance_m.shape

    # For CHOMP we need a usable gradient even near/beyond map bounds.
    # Clamp sampling coordinates so we can still compute the clearance gradient
    # from the distance field (which should repel from boundary obstacles).
    eps = 1e-3
    if gx <= 0 or gy <= 0 or gx >= xw - 1 or gy >= yw - 1:
        gx = min(max(gx, eps), (xw - 1) - eps)
        gy = min(max(gy, eps), (yw - 1) - eps)

    x0, y0 = int(math.floor(gx)), int(math.floor(gy))
    tx, ty = gx - x0, gy - y0
    c00 = clearance_m[x0, y0]
    c10 = clearance_m[x0 + 1, y0]
    c01 = clearance_m[x0, y0 + 1]
    c11 = clearance_m[x0 + 1, y0 + 1]
    c = (
        (1 - tx) * (1 - ty) * c00
        + tx * (1 - ty) * c10
        + (1 - tx) * ty * c01
        + tx * ty * c11
    )
    # partial w.r.t. grid coords -> chain rule /reso for world
    dc_dgx = (1 - ty) * (c10 - c00) + ty * (c11 - c01)
    dc_dgy = (1 - tx) * (c01 - c00) + tx * (c11 - c10)
    return float(c), float(dc_dgx) / reso, float(dc_dgy) / reso


def _obs_barrier_value_grad(d: float, ddx: float, ddy: float) -> Tuple[float, float, float]:
    """
    Gentler obstacle potential than a hard quadratic hinge:
      V(d) = exp((m - d)/sigma) for d < m
           = 0                 for d >= m
    where d is clearance and (ddx,ddy) is grad(d) in world coords.

    This reduces sensitivity at the margin while still pushing strongly when
    points get too close. Gradient is clipped for stability.
    """
    m = _CHOMP_OBS_MARGIN
    if d >= m:
        return 0.0, 0.0, 0.0
    sigma = max(float(_CHOMP_OBS_SIGMA), 1e-6)
    z = (m - d) / sigma
    z = min(12.0, max(0.0, z))  # clamp exponent
    val = math.exp(z)
    # dV/dd = -(1/sigma) * exp((m-d)/sigma)
    dV_dd = -(val / sigma)
    gx = dV_dd * ddx
    gy = dV_dd * ddy
    gmax = float(_CHOMP_OBS_GRAD_CLIP)
    gnorm = math.hypot(gx, gy)
    if gnorm > gmax and gnorm > 1e-12:
        s = gmax / gnorm
        gx *= s
        gy *= s
    return val, gx, gy


def phase2_chomp(
    px: List[float],
    py: List[float],
    ox: List[float],
    oy: List[float],
    reso: float,
    rr: float,
    safety_margin: float = 0.0,
    obstacle_rects=None,
) -> Tuple[List[float], List[float]]:
    if len(px) < 2:
        return px, py

    safe_rr = float(rr) + float(safety_margin) + max(_INFLATION_RESO_FACTOR * reso, 0.15)
    ox_g = [float(x) / reso for x in ox]
    oy_g = [float(y) / reso for y in oy]
    P, obsmap = base_astar.calc_parameters(ox_g, oy_g, safe_rr, reso)
    if obstacle_rects:
        base_astar._apply_rect_disk_obstacles(obsmap, P, obstacle_rects, safe_rr, reso)
    clearance_m = _build_clearance_meters(obsmap, reso)
    xw, yw = clearance_m.shape
    # Clamp CHOMP interior points inside the sampling-safe region.
    # `_sample_clearance_and_grad` requires gx in (0, xw-1) and gy in (0, yw-1).
    _chomp_eps = 1e-3
    x_min = (P.minx + _chomp_eps) * reso
    x_max = (P.minx + (xw - 1.0) - _chomp_eps) * reso
    y_min = (P.miny + _chomp_eps) * reso
    y_max = (P.miny + (yw - 1.0) - _chomp_eps) * reso

    qx, qy = _resample_polyline(px, py, _CHOMP_POINTS)
    n = len(qx)
    if n < 3:
        return px, py

    # Hard safety check against the obstacle point cloud (continuous distance),
    # to avoid CHOMP exploiting discretization gaps in the inflated occupancy grid.
    if ox and oy:
        if _HAS_KDTREE:
            _tree = _cKDTree(np.column_stack([ox, oy]).astype(np.float64))

            def _min_dist_pts(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
                d, _ = _tree.query(np.column_stack([xs, ys]), k=1)
                return d.astype(np.float64)

            def _nearest_obs_pts(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                d, idx = _tree.query(np.column_stack([xs, ys]), k=1)
                idx = np.asarray(idx, dtype=np.int64)
                obs = _tree.data[idx]
                return d.astype(np.float64), obs[:, 0].astype(np.float64), obs[:, 1].astype(np.float64)

        else:
            oarr = np.column_stack([ox, oy]).astype(np.float64)

            def _min_dist_pts(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
                out = np.empty_like(xs, dtype=np.float64)
                for ii in range(xs.shape[0]):
                    dx = oarr[:, 0] - xs[ii]
                    dy = oarr[:, 1] - ys[ii]
                    out[ii] = float(np.min(np.hypot(dx, dy)))
                return out

            def _nearest_obs_pts(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                d_out = np.empty_like(xs, dtype=np.float64)
                ox_out = np.empty_like(xs, dtype=np.float64)
                oy_out = np.empty_like(xs, dtype=np.float64)
                for ii in range(xs.shape[0]):
                    dx = oarr[:, 0] - xs[ii]
                    dy = oarr[:, 1] - ys[ii]
                    di = np.hypot(dx, dy)
                    j = int(np.argmin(di))
                    d_out[ii] = float(di[j])
                    ox_out[ii] = float(oarr[j, 0])
                    oy_out[ii] = float(oarr[j, 1])
                return d_out, ox_out, oy_out
    else:

        def _min_dist_pts(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
            return np.full_like(xs, np.inf, dtype=np.float64)

        def _nearest_obs_pts(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                np.full_like(xs, np.inf, dtype=np.float64),
                np.zeros_like(xs, dtype=np.float64),
                np.zeros_like(xs, dtype=np.float64),
            )

    target_clear = float(rr) + float(safety_margin) + max(float(_CHOMP_HARD_CLEARANCE_PAD), 0.25 * float(reso))
    rects = list(obstacle_rects) if obstacle_rects else []

    def _augment_cloud_clearance(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        dcloud, oxn, oyn = _nearest_obs_pts(xs, ys)
        if rects:
            _ha_draw = Path(__file__).resolve().parents[1]
            if str(_ha_draw) not in sys.path:
                sys.path.insert(0, str(_ha_draw))
            from scenario_obstacles import min_distance_point_to_rects

            for ii in range(xs.shape[0]):
                d_rect = min_distance_point_to_rects(float(xs[ii]), float(ys[ii]), rects)
                if d_rect < dcloud[ii]:
                    dcloud[ii] = d_rect
                    # Push away from nearest rect surface along center->sample ray.
                    if d_rect > 1e-9:
                        # Approximate outward direction using nearest cloud obstacle if available.
                        vx = xs[ii] - oxn[ii]
                        vy = ys[ii] - oyn[ii]
                    else:
                        vx, vy = 1.0, 0.0
                    vn = max(math.hypot(vx, vy), 1e-9)
                    oxn[ii] = xs[ii] - d_rect * (vx / vn)
                    oyn[ii] = ys[ii] - d_rect * (vy / vn)
        return dcloud, oxn, oyn

    for _ in range(_CHOMP_ITERS):
        gx_total = np.zeros(n, dtype=np.float64)
        gy_total = np.zeros(n, dtype=np.float64)

        qx_prev = qx.copy()
        qy_prev = qy.copy()

        # Smoothness (acceleration): sum ||q_{i+1} - 2 q_i + q_{i-1}||^2
        ax = np.zeros(n, dtype=np.float64)
        ay = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            ax[i] = qx[i + 1] - 2.0 * qx[i] + qx[i - 1]
            ay[i] = qy[i + 1] - 2.0 * qy[i] + qy[i - 1]
        for i in range(1, n - 1):
            # grad_i = 2*(a_{i-1}*1 + a_i*(-2) + a_{i+1}*1)
            gxs = -4.0 * ax[i]
            gys = -4.0 * ay[i]
            if i - 1 >= 1:
                gxs += 2.0 * ax[i - 1]
                gys += 2.0 * ay[i - 1]
            if i + 1 <= n - 2:
                gxs += 2.0 * ax[i + 1]
                gys += 2.0 * ay[i + 1]
            gx_total[i] += _CHOMP_LAMBDA_SMOOTH * gxs
            gy_total[i] += _CHOMP_LAMBDA_SMOOTH * gys

        # Obstacle barrier on interior
        for i in range(1, n - 1):
            d, ddx, ddy = _sample_clearance_and_grad(float(qx[i]), float(qy[i]), clearance_m, P, reso)
            ov, ovx, ovy = _obs_barrier_value_grad(d, ddx, ddy)
            gx_total[i] += _CHOMP_LAMBDA_OBS * ovx
            gy_total[i] += _CHOMP_LAMBDA_OBS * ovy

        qx[1:-1] -= _CHOMP_STEP * gx_total[1:-1]
        qy[1:-1] -= _CHOMP_STEP * gy_total[1:-1]

        # Numerical stability: keep CHOMP points inside the map bounds.
        # Without clamping, large smoothing steps can push points out of the
        # distance-field region, where the barrier loses its effect.
        qx[1:-1] = np.clip(qx[1:-1], x_min, x_max)
        qy[1:-1] = np.clip(qy[1:-1], y_min, y_max)

        # Safety: do not allow points to enter "occupied" regions of the
        # inflated map. In the EDT clearance, occupied cells have d=0 and a
        # near-zero gradient, so the barrier would lose its repulsion.
        # Revert any interior point whose sampled clearance drops to ~0.
        for i in range(1, n - 1):
            d_now, _, _ = _sample_clearance_and_grad(float(qx[i]), float(qy[i]), clearance_m, P, reso)
            if d_now <= 1e-9:
                qx[i] = qx_prev[i]
                qy[i] = qy_prev[i]

        # Hard projection: if a point is too close to obstacles (points + rects),
        # push it outward to the target clearance rather than just reverting.
        dcloud, oxn, oyn = _augment_cloud_clearance(qx[1:-1], qy[1:-1])
        bad = dcloud < target_clear
        if np.any(bad):
            xs = qx[1:-1].copy()
            ys = qy[1:-1].copy()
            # direction away from nearest obstacle point
            vx = xs - oxn
            vy = ys - oyn
            vn = np.hypot(vx, vy)
            vn = np.maximum(vn, 1e-9)
            ux = vx / vn
            uy = vy / vn
            xs[bad] = oxn[bad] + target_clear * ux[bad]
            ys[bad] = oyn[bad] + target_clear * uy[bad]
            # Keep projected points inside map bounds
            xs = np.clip(xs, x_min, x_max)
            ys = np.clip(ys, y_min, y_max)
            qx[1:-1] = xs
            qy[1:-1] = ys

    out_x, out_y = qx.tolist(), qy.tolist()
    eff_rr = float(rr) + float(safety_margin)
    if _path_violates_disk_clearance(
        out_x, out_y, ox, oy, eff_rr, obstacle_rects=obstacle_rects, reso=reso
    ):
        return px, py
    return out_x, out_y


def _segment_min_distance_to_points(
    x0: float, y0: float, x1: float, y1: float, ox: List[float], oy: List[float]
) -> float:
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    best = float("inf")
    for px, py in zip(ox, oy):
        if denom <= 1e-12:
            d = math.hypot(px - x0, py - y0)
        else:
            t = ((px - x0) * dx + (py - y0) * dy) / denom
            t = max(0.0, min(1.0, t))
            cx = x0 + t * dx
            cy = y0 + t * dy
            d = math.hypot(px - cx, py - cy)
        if d < best:
            best = d
    return best


def _phase3_arc_sample_count(
    sweep: float,
    arc_radius: float,
    clearance: float,
    reso: float,
    sample_mult: float = 1.0,
) -> int:
    """Number of angular samples for disk centerline clearance along an arc."""
    mult = max(1.0, float(sample_mult))
    arc_len = abs(float(sweep)) * max(float(arc_radius), 1e-6)
    along_step = max(0.06, min(0.2 * float(clearance), 0.35 * float(reso))) / mult
    n_len = int(math.ceil(arc_len / along_step)) + 1
    n_ang = int(math.ceil(abs(float(sweep)) * _ARC_POINTS_PER_RAD * mult)) + 1
    return min(240, max(8, n_len, n_ang))


def _segment_disk_clear(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ox: List[float],
    oy: List[float],
    clearance: float,
    reso: float = 0.2,
    sample_mult: float = 1.0,
    obstacle_rects=None,
    validation_ctx=None,
) -> bool:
    """True if a disk of radius ``clearance`` can traverse the straight segment."""
    del sample_mult
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import line_disk_feasible

    ctx = validation_ctx or _disk_validation_ctx(clearance, reso, ox, oy, obstacle_rects)
    return line_disk_feasible(x0, y0, x1, y1, ctx)


def _arc_clear(
    ox_c: float,
    oy_c: float,
    r: float,
    a0: float,
    sweep: float,
    ox: List[float],
    oy: List[float],
    clearance: float,
    reso: float = 0.2,
    sample_mult: float = 1.0,
    obstacle_rects=None,
    validation_ctx=None,
) -> bool:
    """Planning-only arc gate (phase 3 DP / shortcut). Viz uses ``densify_polyline``."""
    del sample_mult
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import arc_disk_feasible

    ctx = validation_ctx or _disk_validation_ctx(clearance, reso, ox, oy, obstacle_rects)
    return arc_disk_feasible(ox_c, oy_c, r, a0, sweep, ctx)


def _primitives_disk_valid(
    prims: List[Tuple[str, dict]],
    ox: List[float],
    oy: List[float],
    clearance: float,
    reso: float = 0.2,
    obstacle_rects=None,
    validation_ctx=None,
) -> bool:
    """True when every primitive passes the hierarchical disk-vs-OBB funnel."""
    if not prims:
        return True
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import primitive_disk_feasible

    ctx = validation_ctx or _disk_validation_ctx(clearance, reso, ox, oy, obstacle_rects)
    return all(primitive_disk_feasible(typ, p, ctx) for typ, p in prims)


def _primitives_min_clearance(
    prims: List[Tuple[str, dict]],
    ox: List[float],
    oy: List[float],
    clearance: float,
    reso: float = 0.2,
    sample_mult: float = 1.0,
    obstacle_rects=None,
    validation_ctx=None,
) -> float:
    """Minimum centerline clearance along primitives (returns ``clearance`` if invalid)."""
    del sample_mult
    if not prims:
        return float("inf")
    if not _primitives_disk_valid(
        prims, ox, oy, clearance, reso, obstacle_rects=obstacle_rects, validation_ctx=validation_ctx
    ):
        return float(clearance)
    return float(clearance) + 1e-3


def _shortcut_path(
    px: List[float],
    py: List[float],
    ox: List[float],
    oy: List[float],
    clearance: float,
    *,
    obstacle_rects=None,
    reso: float = 0.2,
    validation_ctx=None,
) -> Tuple[List[float], List[float]]:
    if len(px) <= 2:
        return px, py
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import line_disk_feasible

    ctx = validation_ctx or _disk_validation_ctx(clearance, reso, ox, oy, obstacle_rects)
    outx = [px[0]]
    outy = [py[0]]
    i = 0
    n = len(px)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if line_disk_feasible(px[i], py[i], px[j], py[j], ctx):
                break
            j -= 1
        outx.append(px[j])
        outy.append(py[j])
        i = j
    return outx, outy


def _path_min_segment_clearance(
    px: List[float], py: List[float], ox: List[float], oy: List[float], *, obstacle_rects=None, reso: float = 0.2
) -> float:
    if len(px) < 2:
        return float("inf")
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import segment_min_clearance

    rects = list(obstacle_rects) if obstacle_rects else []
    best = float("inf")
    for i in range(len(px) - 1):
        d = segment_min_clearance(
            px[i], py[i], px[i + 1], py[i + 1], ox, oy, obstacle_rects=rects, reso=reso
        )
        if d < best:
            best = d
    return best


def _path_violates_disk_clearance(
    px: List[float],
    py: List[float],
    ox: List[float],
    oy: List[float],
    eff_rr: float,
    *,
    obstacle_rects=None,
    reso: float = 0.2,
) -> bool:
    """True when a disk of radius ``eff_rr`` overlaps obstacle points or scenario rects."""
    if _path_min_segment_clearance(px, py, ox, oy, obstacle_rects=obstacle_rects, reso=reso) <= float(eff_rr):
        return True
    if obstacle_rects:
        _ha_draw = Path(__file__).resolve().parents[1]
        if str(_ha_draw) not in sys.path:
            sys.path.insert(0, str(_ha_draw))
        from scenario_obstacles import disk_path_min_clearance_to_rects

        if disk_path_min_clearance_to_rects(px, py, float(eff_rr), obstacle_rects, reso) < 0.0:
            return True
    return False


def _min_dist_point_to_obstacles(x: float, y: float, ox: List[float], oy: List[float]) -> float:
    if not ox:
        return float("inf")
    best = float("inf")
    for ox0, oy0 in zip(ox, oy):
        d = math.hypot(x - ox0, y - oy0)
        if d < best:
            best = d
    return best


def _arc_fillet_clear(
    ox_c: float,
    oy_c: float,
    r: float,
    a0: float,
    a1: float,
    ox: List[float],
    oy: List[float],
    clearance: float,
) -> bool:
    """Sample arc from a0 to a1 along the shorter angular sweep; True if all samples clear."""
    n = max(3, int(abs(a1 - a0) * _ARC_POINTS_PER_RAD) + 1)
    for k in range(n):
        t = k / float(n - 1) if n > 1 else 0.0
        ang = a0 + t * (a1 - a0)
        x = ox_c + r * math.cos(ang)
        y = oy_c + r * math.sin(ang)
        if _min_dist_point_to_obstacles(x, y, ox, oy) <= clearance:
            return False
    return True


def phase3_straight_arc(
    px: List[float],
    py: List[float],
    ox: List[float],
    oy: List[float],
    clearance: float,
) -> Tuple[List[float], List[float]]:
    """
    Replace interior polyline corners with tangent circular fillets (straight — arc — straight).
    Radius is clamped by adjacent edge lengths; unsafe fillets (obstacle check) keep the sharp corner.
    """
    n = len(px)
    if n < 3:
        return px, py

    out_x: List[float] = []
    out_y: List[float] = []

    def append_pt(x: float, y: float) -> None:
        if out_x and abs(out_x[-1] - x) < 1e-9 and abs(out_y[-1] - y) < 1e-9:
            return
        out_x.append(x)
        out_y.append(y)

    cur_x, cur_y = float(px[0]), float(py[0])
    append_pt(cur_x, cur_y)

    r_nom = max(_ARC_RADIUS, 1e-6)

    for i in range(1, n - 1):
        ax, ay = float(px[i - 1]), float(py[i - 1])
        cx, cy = float(px[i]), float(py[i])
        bx, by = float(px[i + 1]), float(py[i + 1])

        vinx, viny = cx - ax, cy - ay
        voutx, vouty = bx - cx, by - cy
        len_in = math.hypot(vinx, viny)
        len_out = math.hypot(voutx, vouty)
        if len_in < 1e-9 or len_out < 1e-9:
            append_pt(cx, cy)
            cur_x, cur_y = cx, cy
            continue

        uix, uiy = vinx / len_in, viny / len_in
        uox, uoy = voutx / len_out, vouty / len_out

        gamma = math.atan2(uix * uoy - uiy * uox, uix * uox + uiy * uoy)
        turn_mag = abs(gamma)
        if turn_mag < _ARC_MIN_TURN:
            append_pt(cx, cy)
            cur_x, cur_y = cx, cy
            continue

        tan_half = math.tan(0.5 * turn_mag)
        if tan_half < 1e-9:
            append_pt(cx, cy)
            cur_x, cur_y = cx, cy
            continue

        half_max = 0.48 * min(len_in, len_out)
        r_eff = min(r_nom, half_max / tan_half)
        d = r_eff * tan_half
        if d <= 1e-9 or d > 0.49 * min(len_in, len_out):
            append_pt(cx, cy)
            cur_x, cur_y = cx, cy
            continue

        p0x = cx - d * uix
        p0y = cy - d * uiy
        p1x = cx + d * uox
        p1y = cy + d * uoy

        sign = 1.0 if gamma >= 0.0 else -1.0
        perp_x, perp_y = -uiy, uix
        oc_x = p0x + sign * r_eff * perp_x
        oc_y = p0y + sign * r_eff * perp_y

        a_start = math.atan2(p0y - oc_y, p0x - oc_x)
        a_end = a_start + gamma  # signed sweep matches tangent fillet geometry

        use_fillet = True
        if ox:
            if _segment_min_distance_to_points(cur_x, cur_y, p0x, p0y, ox, oy) <= clearance:
                use_fillet = False
            elif _segment_min_distance_to_points(p1x, p1y, bx, by, ox, oy) <= clearance:
                use_fillet = False
            elif not _arc_fillet_clear(oc_x, oc_y, r_eff, a_start, a_end, ox, oy, clearance):
                use_fillet = False

        if not use_fillet:
            append_pt(cx, cy)
            cur_x, cur_y = cx, cy
            continue

        append_pt(p0x, p0y)
        n_arc = max(3, int(turn_mag * _ARC_POINTS_PER_RAD))
        for k in range(1, n_arc + 1):
            t = k / float(n_arc)
            ang = a_start + t * gamma
            append_pt(oc_x + r_eff * math.cos(ang), oc_y + r_eff * math.sin(ang))
        cur_x, cur_y = p1x, p1y

    gx, gy = float(px[-1]), float(py[-1])
    append_pt(gx, gy)

    return out_x, out_y


def _circle_from_3pts(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> Optional[Tuple[float, float, float]]:
    """Return (ox,oy,r) for the circumcircle of 3 points, or None if nearly collinear."""
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ax2ay2 = ax * ax + ay * ay
    bx2by2 = bx * bx + by * by
    cx2cy2 = cx * cx + cy * cy
    ux = (ax2ay2 * (by - cy) + bx2by2 * (cy - ay) + cx2cy2 * (ay - by)) / d
    uy = (ax2ay2 * (cx - bx) + bx2by2 * (ax - cx) + cx2cy2 * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    if not math.isfinite(r) or r < 1e-6:
        return None
    return float(ux), float(uy), float(r)


def _angle_wrap(a: float) -> float:
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def _arc_params_through_mid(
    ox_c: float,
    oy_c: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    mx: float,
    my: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Return (a0, a1, sweep) where the arc from a0 to a1 (with sweep sign)
    passes through mid-point angle. Chooses the shorter of the two arcs
    that includes the mid angle.
    """
    a0 = math.atan2(ay - oy_c, ax - ox_c)
    a1 = math.atan2(by - oy_c, bx - ox_c)
    am = math.atan2(my - oy_c, mx - ox_c)
    # Consider two sweeps from a0 to a1: ccw and cw.
    ccw = _angle_wrap(a1 - a0)
    if ccw <= 0:
        ccw += 2.0 * math.pi
    cw = ccw - 2.0 * math.pi  # negative

    def contains(a_start: float, sweep: float, a_mid: float) -> bool:
        # Normalize by rotating so start=0, check mid is within sweep interval.
        rel_mid = _angle_wrap(a_mid - a_start)
        if sweep > 0:
            if rel_mid < 0:
                rel_mid += 2.0 * math.pi
            return 0.0 <= rel_mid <= sweep + 1e-9
        # sweep < 0
        if rel_mid > 0:
            rel_mid -= 2.0 * math.pi
        return sweep - 1e-9 <= rel_mid <= 0.0

    cand: List[Tuple[float, float, float]] = []
    if contains(a0, ccw, am):
        cand.append((a0, a1, ccw))
    if contains(a0, cw, am):
        cand.append((a0, a1, cw))
    if not cand:
        return None
    # choose smaller magnitude sweep
    cand.sort(key=lambda t: abs(t[2]))
    return cand[0]


def _unit(dx: float, dy: float) -> Optional[Tuple[float, float]]:
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    return dx / n, dy / n


def _local_tangent(px: List[float], py: List[float], idx: int) -> Optional[Tuple[float, float]]:
    n = len(px)
    if n < 2:
        return None
    if idx <= 0:
        return _unit(px[1] - px[0], py[1] - py[0])
    if idx >= n - 1:
        return _unit(px[n - 1] - px[n - 2], py[n - 1] - py[n - 2])
    u0 = _unit(px[idx] - px[idx - 1], py[idx] - py[idx - 1])
    u1 = _unit(px[idx + 1] - px[idx], py[idx + 1] - py[idx])
    if u0 is None:
        return u1
    if u1 is None:
        return u0
    return _unit(u0[0] + u1[0], u0[1] + u1[1])


def _arc_from_start_tangent_discrete(
    px: List[float],
    py: List[float],
    i: int,
    j: int,
    r: float,
    turn_sign: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Construct arc candidate from start endpoint tangent + discrete radius.
    Returns (ocx, ocy, r, a0, sweep) if feasible.
    """
    if j <= i:
        return None
    ti = _local_tangent(px, py, i)
    tj = _local_tangent(px, py, j)
    if ti is None or tj is None:
        return None
    tx, ty = ti
    # left normal
    nx, ny = -ty, tx
    x0, y0 = px[i], py[i]
    x1, y1 = px[j], py[j]
    ocx = x0 + turn_sign * r * nx
    ocy = y0 + turn_sign * r * ny
    # Endpoint must lie on same circle
    if abs(math.hypot(x1 - ocx, y1 - ocy) - r) > max(0.25, 0.08 * r):
        return None
    a0 = math.atan2(y0 - ocy, x0 - ocx)
    a1 = math.atan2(y1 - ocy, x1 - ocx)
    if turn_sign > 0:
        sweep = _angle_wrap(a1 - a0)
        if sweep <= 0:
            sweep += 2.0 * math.pi
        tan_end = (-math.sin(a1), math.cos(a1))
    else:
        sweep = _angle_wrap(a1 - a0)
        if sweep >= 0:
            sweep -= 2.0 * math.pi
        tan_end = (math.sin(a1), -math.cos(a1))

    # Endpoint tangency consistency (soft gate): arc tangent should roughly
    # align with local path tangent at j.
    dot = max(-1.0, min(1.0, tan_end[0] * tj[0] + tan_end[1] * tj[1]))
    ang = math.acos(dot)
    if ang > math.radians(45.0):
        return None
    return ocx, ocy, r, a0, sweep


def _polyline_straight_primitives(
    qx: List[float], qy: List[float]
) -> List[Tuple[str, dict]]:
    """One 'S' primitive per consecutive vertex pair (for DF yaw fill fallback)."""
    return [
        (
            "S",
            {
                "x0": float(qx[ii]),
                "y0": float(qy[ii]),
                "x1": float(qx[ii + 1]),
                "y1": float(qy[ii + 1]),
            },
        )
        for ii in range(len(qx) - 1)
    ]


def phase3_min_segments(
    px: List[float],
    py: List[float],
    ox: List[float],
    oy: List[float],
    clearance: float,
    return_primitives: bool = False,
    reso: float = 0.2,
    obstacle_rects=None,
    dp_objective: str = DP_OBJECTIVE_LENGTH,
) -> Tuple[List[float], List[float]]:
    """
    Shortcut waypoint DP: straight segments and circular arcs between sparse vertices.

    ``dp_objective``:
      - ``length``: minimize total primitive arc/chord length (default).
      - ``min_segments``: minimize primitive count (legacy behaviour).

    Primitives use the hierarchical disk-vs-OBB funnel in ``scenario_obstacles``.
    """
    if dp_objective not in (DP_OBJECTIVE_LENGTH, DP_OBJECTIVE_MIN_SEGMENTS):
        raise ValueError(f"dp_objective must be 'length' or 'min_segments' (got {dp_objective!r})")

    n = len(px)
    if n < 2:
        if return_primitives:
            return px, py, []
        return px, py

    validation_ctx = _disk_validation_ctx(clearance, reso, ox, oy, obstacle_rects)
    check_clear = float(clearance)

    def _edge_cost(typ: str, params: dict) -> float:
        if dp_objective == DP_OBJECTIVE_MIN_SEGMENTS:
            return 1.0
        return _primitive_length(typ, params)

    INF = 10**9
    best_cost = [INF] * n
    best_prev: List[Optional[Tuple[int, str, dict]]] = [None] * n
    best_cost[0] = 0.0

    max_span = min(30, n - 1)

    for j in range(1, n):
        i_min = max(0, j - max_span)
        for i in range(i_min, j):
            if best_cost[i] >= INF:
                continue
            s_params = {"x0": px[i], "y0": py[i], "x1": px[j], "y1": py[j]}
            if _segment_disk_clear(
                px[i],
                py[i],
                px[j],
                py[j],
                ox,
                oy,
                check_clear,
                reso,
                obstacle_rects=obstacle_rects,
                validation_ctx=validation_ctx,
            ):
                c = best_cost[i] + _edge_cost("S", s_params)
                if c < best_cost[j]:
                    best_cost[j] = c
                    best_prev[j] = (i, "S", s_params)

            if j - i >= 2:
                for k in {i + 1, (i + j) // 2, j - 1}:
                    if not (i < k < j):
                        continue
                    circ = _circle_from_3pts(px[i], py[i], px[k], py[k], px[j], py[j])
                    if circ is None:
                        continue
                    ocx, ocy, r = circ
                    arc_par = _arc_params_through_mid(ocx, ocy, px[i], py[i], px[j], py[j], px[k], py[k])
                    if arc_par is None:
                        continue
                    a0, _a1, sweep = arc_par
                    a_params = {"ocx": ocx, "ocy": ocy, "r": r, "a0": a0, "sweep": sweep}
                    if not _arc_clear(
                        ocx,
                        ocy,
                        r,
                        a0,
                        sweep,
                        ox,
                        oy,
                        check_clear,
                        reso,
                        obstacle_rects=obstacle_rects,
                        validation_ctx=validation_ctx,
                    ):
                        continue
                    c = best_cost[i] + _edge_cost("A", a_params)
                    if c < best_cost[j]:
                        best_cost[j] = c
                        best_prev[j] = (i, "A", a_params)

                chord = math.hypot(px[j] - px[i], py[j] - py[i])
                if chord > 1e-6:
                    r_min = 0.5 * chord + 1e-4
                    # Candidate count grows with chord length, capped at 50.
                    # Keep growth geometric so we span both tight and wide arcs.
                    n_r = min(50, max(5, int(math.ceil(chord))))
                    growth = 1.08
                    r_list = [r_min * (growth**kk) for kk in range(n_r)]
                    for r_try in r_list:
                        for turn in (1.0, -1.0):
                            cand = _arc_from_start_tangent_discrete(px, py, i, j, r_try, turn)
                            if cand is None:
                                continue
                            ocx, ocy, r, a0, sweep = cand
                            a_params = {"ocx": ocx, "ocy": ocy, "r": r, "a0": a0, "sweep": sweep}
                            if not _arc_clear(
                                ocx,
                                ocy,
                                r,
                                a0,
                                sweep,
                                ox,
                                oy,
                                check_clear,
                                reso,
                                obstacle_rects=obstacle_rects,
                                validation_ctx=validation_ctx,
                            ):
                                continue
                            c = best_cost[i] + _edge_cost("A", a_params)
                            if c < best_cost[j]:
                                best_cost[j] = c
                                best_prev[j] = (i, "A", a_params)

    if best_prev[-1] is None:
        if return_primitives:
            return [float(x) for x in px], [float(y) for y in py], _polyline_straight_primitives(px, py)
        return px, py

    prims: List[Tuple[str, dict]] = []
    cur = n - 1
    while cur != 0:
        prev = best_prev[cur]
        if prev is None:
            break
        _i, typ, params = prev
        prims.append((typ, params))
        cur = _i
    prims.reverse()

    if not _primitives_disk_valid(
        prims, ox, oy, check_clear, reso, obstacle_rects=obstacle_rects, validation_ctx=validation_ctx
    ):
        if return_primitives:
            return [float(x) for x in px], [float(y) for y in py], _polyline_straight_primitives(px, py)
        return px, py

    # Emit sampled points
    outx: List[float] = [float(px[0])]
    outy: List[float] = [float(py[0])]

    def emit(x: float, y: float) -> None:
        if outx and abs(outx[-1] - x) < 1e-9 and abs(outy[-1] - y) < 1e-9:
            return
        outx.append(float(x))
        outy.append(float(y))

    for typ, p in prims:
        if typ == "S":
            emit(p["x1"], p["y1"])
        else:
            ocx = p["ocx"]
            ocy = p["ocy"]
            r = p["r"]
            a0 = p["a0"]
            sweep = p["sweep"]
            arc_len = abs(float(sweep)) * max(float(r), 1e-9)
            n_arc = max(2, int(math.ceil(arc_len / max(float(reso), 1e-9))))
            for kk in range(1, n_arc + 1):
                t = kk / float(n_arc)
                ang = a0 + t * sweep
                emit(ocx + r * math.cos(ang), ocy + r * math.sin(ang))

    if return_primitives:
        return outx, outy, prims
    return outx, outy


def _primitive_start_xy(typ: str, params: dict) -> Tuple[float, float]:
    if typ == "S":
        return float(params["x0"]), float(params["y0"])
    ocx = float(params["ocx"])
    ocy = float(params["ocy"])
    r = float(params["r"])
    a0 = float(params["a0"])
    return ocx + r * math.cos(a0), ocy + r * math.sin(a0)


def densify_polyline(
    px: Sequence[float],
    py: Sequence[float],
    reso: float,
) -> Tuple[List[float], List[float]]:
    """Resample a polyline at roughly ``reso`` spacing for footprint visualization."""
    if len(px) < 2 or len(py) != len(px):
        return list(px), list(py)
    step = max(float(reso), 1e-6)
    outx: List[float] = [float(px[0])]
    outy: List[float] = [float(py[0])]
    for i in range(len(px) - 1):
        x0, y0 = float(px[i]), float(py[i])
        x1, y1 = float(px[i + 1]), float(py[i + 1])
        seg_len = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(seg_len / step)))
        for k in range(1, n + 1):
            t = k / float(n)
            xf = x0 + t * (x1 - x0)
            yf = y0 + t * (y1 - y0)
            if abs(outx[-1] - xf) < 1e-9 and abs(outy[-1] - yf) < 1e-9:
                continue
            outx.append(xf)
            outy.append(yf)
    return outx, outy


def _point_segment_distance_and_t(
    px: float, py: float, x0: float, y0: float, x1: float, y1: float
) -> Tuple[float, float]:
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x0, py - y0), 0.0
    t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = x0 + t * dx
    cy = y0 + t * dy
    return math.hypot(px - cx, py - cy), t


def _normalize_angle_delta(a0: float, a1: float) -> float:
    return float(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0)))


def _angle_on_arc_sweep(ang: float, a0: float, sweep: float) -> Optional[float]:
    """Return fraction t in [0,1] along signed sweep if ``ang`` lies on the arc."""
    if abs(sweep) < 1e-12:
        return 0.0 if abs(_normalize_angle_delta(a0, ang)) < 1e-6 else None
    delta = _normalize_angle_delta(a0, ang)
    if sweep > 0.0:
        if delta < -1e-6 or delta > sweep + 1e-6:
            return None
        return max(0.0, min(1.0, delta / sweep))
    if delta > 1e-6 or delta < sweep - 1e-6:
        return None
    return max(0.0, min(1.0, delta / sweep))


def _point_arc_distance_and_fraction(
    px: float, py: float, ocx: float, ocy: float, r: float, a0: float, sweep: float
) -> Tuple[float, float]:
    ang = math.atan2(py - ocy, px - ocx)
    frac = _angle_on_arc_sweep(ang, a0, sweep)
    radial = abs(math.hypot(px - ocx, py - ocy) - r)
    if frac is not None:
        return radial, frac
    # Distance to arc endpoints when projection falls outside sweep.
    x_start = ocx + r * math.cos(a0)
    y_start = ocy + r * math.sin(a0)
    x_end = ocx + r * math.cos(a0 + sweep)
    y_end = ocy + r * math.sin(a0 + sweep)
    d0 = math.hypot(px - x_start, py - y_start)
    d1 = math.hypot(px - x_end, py - y_end)
    if d0 <= d1:
        return d0, 0.0
    return d1, 1.0


def _yaw_at_point_on_primitives(
    px: float,
    py: float,
    prims: Sequence[Tuple[str, dict]],
    syaw: float,
) -> float:
    cur_yaw = float(syaw)
    best_d = float("inf")
    best_yaw = cur_yaw
    for typ, p in prims:
        if typ == "S":
            d, _t = _point_segment_distance_and_t(
                px, py, float(p["x0"]), float(p["y0"]), float(p["x1"]), float(p["y1"])
            )
            if d < best_d:
                best_d = d
                best_yaw = cur_yaw
        else:
            d, frac = _point_arc_distance_and_fraction(
                px,
                py,
                float(p["ocx"]),
                float(p["ocy"]),
                float(p["r"]),
                float(p["a0"]),
                float(p["sweep"]),
            )
            if d < best_d:
                best_d = d
                best_yaw = cur_yaw + frac * float(p["sweep"])
        if typ == "A":
            cur_yaw += float(p["sweep"])
    return float(math.atan2(math.sin(best_yaw), math.cos(best_yaw)))


def yaw_on_polyline_samples(
    px: Sequence[float],
    py: Sequence[float],
    prims: Sequence[Tuple[str, dict]],
    syaw: float,
) -> List[float]:
    """Assign yaw at existing polyline samples (no geometry change)."""
    return [_yaw_at_point_on_primitives(float(x), float(y), prims, syaw) for x, y in zip(px, py)]


def yaw_fill_from_primitives_on_polyline(
    px: Sequence[float],
    py: Sequence[float],
    prims: Sequence[Tuple[str, dict]],
    syaw: float,
    reso: float,
) -> List[float]:
    """Densify ``px, py`` then assign yaw from phase-3 primitives."""
    dx, dy = densify_polyline(px, py, reso)
    return yaw_on_polyline_samples(dx, dy, prims, syaw)


def path_and_yaw_from_primitives(
    prims: List[Tuple[str, dict]],
    syaw: float,
    reso: float,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Legacy helper: densify primitives to (px, py, pyaw).

    Prefer ``yaw_fill_from_primitives_on_polyline`` when the planner path is
    already fixed and only yaw is needed.
    """
    if not prims:
        return [], [], []

    outx: List[float] = []
    outy: List[float] = []
    outyaw: List[float] = []
    cur_yaw = float(syaw)

    def emit(x: float, y: float, yaw: float) -> None:
        xf, yf, yf_yaw = float(x), float(y), float(yaw)
        if outx and abs(outx[-1] - xf) < 1e-9 and abs(outy[-1] - yf) < 1e-9:
            return
        outx.append(xf)
        outy.append(yf)
        outyaw.append(yf_yaw)

    t0, p0 = prims[0]
    sx, sy = _primitive_start_xy(t0, p0)
    emit(sx, sy, cur_yaw)

    for typ, p in prims:
        if typ == "S":
            emit(float(p["x1"]), float(p["y1"]), cur_yaw)
            continue
        ocx = float(p["ocx"])
        ocy = float(p["ocy"])
        r = max(float(p["r"]), 1e-9)
        a0 = float(p["a0"])
        sweep = float(p["sweep"])
        arc_len = abs(sweep) * r
        n_arc = max(2, int(math.ceil(arc_len / max(float(reso), 1e-9))))
        for kk in range(1, n_arc + 1):
            t = kk / float(n_arc)
            ang = a0 + t * sweep
            emit(ocx + r * math.cos(ang), ocy + r * math.sin(ang), cur_yaw + t * sweep)
        cur_yaw = outyaw[-1]

    pyaw = [float(math.atan2(math.sin(y), math.cos(y))) for y in outyaw]
    return outx, outy, pyaw


def phase3_polish(
    px: List[float],
    py: List[float],
    ox: List[float],
    oy: List[float],
    rr: float,
    *,
    safety_margin: float = 0.0,
    reso: float = 0.2,
    obstacle_rects=None,
    dp_objective: str = DP_OBJECTIVE_LENGTH,
    return_primitives: bool = False,
):
    """Greedy shortcut + primitive DP; falls back to shortcut then phase-1 polyline."""
    from scenario_obstacles import clamp_safety_margin

    safety_margin = clamp_safety_margin(safety_margin)
    eff_rr = float(rr) + float(safety_margin)
    p1x, p1y = list(px), list(py)
    validation_ctx = _disk_validation_ctx(eff_rr, reso, ox, oy, obstacle_rects)
    spx, spy = _shortcut_path(
        px, py, ox, oy, eff_rr, obstacle_rects=obstacle_rects, reso=reso, validation_ctx=validation_ctx
    )

    out_x, out_y, prims = phase3_min_segments(
        spx,
        spy,
        ox,
        oy,
        eff_rr,
        return_primitives=True,
        reso=reso,
        obstacle_rects=obstacle_rects,
        dp_objective=dp_objective,
    )
    if _primitives_disk_valid(
        prims, ox, oy, eff_rr, reso, obstacle_rects=obstacle_rects, validation_ctx=validation_ctx
    ):
        if return_primitives:
            return out_x, out_y, prims
        return out_x, out_y

    if return_primitives:
        return list(spx), list(spy), _polyline_straight_primitives(spx, spy)
    if len(spx) >= 2:
        return spx, spy
    return p1x, p1y


def astar_planning(
    sx: float,
    sy: float,
    gx: float,
    gy: float,
    ox: List[float],
    oy: List[float],
    reso: float,
    rr: float,
    stop_phase: int = 3,
    safety_margin: float = 0.0,
    return_primitives: bool = False,
    obstacle_rects=None,
    dp_objective: str = DP_OBJECTIVE_LENGTH,
    timing: Optional[Dict[str, float]] = None,
    disk_collision_mode: str = DISK_COLLISION_OFFLINE,
) -> Tuple[List[float], List[float]]:
    """
    mod_grid pipeline:
      Phase 1: augmented A*
      Phase 3: shortcut + primitive DP (``dp_objective``: ``length`` or ``min_segments``)

    Phase 2 (CHOMP) is not supported — use ``stop_phase=3``.
    """
    if stop_phase == 2:
        raise NotImplementedError(
            "mod_grid Phase 2 (CHOMP) is disabled. Use stop_phase=1 or stop_phase=3."
        )
    if stop_phase not in (1, 3):
        raise ValueError(f"stop_phase must be 1 or 3 (got {stop_phase})")
    if dp_objective not in (DP_OBJECTIVE_LENGTH, DP_OBJECTIVE_MIN_SEGMENTS):
        raise ValueError(f"dp_objective must be 'length' or 'min_segments' (got {dp_objective!r})")

    from scenario_obstacles import clamp_safety_margin

    safety_margin = clamp_safety_margin(safety_margin)

    px, py = phase1_augmented_astar(
        sx,
        sy,
        gx,
        gy,
        ox,
        oy,
        reso,
        rr,
        safety_margin=float(safety_margin),
        obstacle_rects=obstacle_rects,
        timing=timing,
        disk_collision_mode=disk_collision_mode,
    )
    if len(px) < 2:
        return px, py

    if stop_phase == 1:
        return px, py

    if return_primitives:
        return phase3_polish(
            px,
            py,
            ox,
            oy,
            rr,
            safety_margin=float(safety_margin),
            reso=reso,
            obstacle_rects=obstacle_rects,
            dp_objective=dp_objective,
            return_primitives=True,
        )
    return phase3_polish(
        px,
        py,
        ox,
        oy,
        rr,
        safety_margin=float(safety_margin),
        reso=reso,
        obstacle_rects=obstacle_rects,
        dp_objective=dp_objective,
        return_primitives=False,
    )


def _obstacle_points_from_app_scenario(
    scenario: dict,
) -> Tuple[List[float], List[float], float, float, float]:
    """Delegate to shared HA_draw scenario obstacle rasterizer."""
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import obstacle_points_for_disk_planner

    return obstacle_points_for_disk_planner(scenario)


def _disk_planner_rr_from_scenario(robot: dict, reso: float) -> float:
    """Circumradius for disk planner — matches HA_draw app._disk_planner_rr."""
    try:
        import HybridAstarPlanner.mod_grid_SE as mod_grid_SE_astar  # type: ignore
    except ModuleNotFoundError:
        import mod_grid_SE as mod_grid_SE_astar  # type: ignore

    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_planner_bridge import _resolve_robot_dict

    robot = _resolve_robot_dict(robot)
    verts = mod_grid_SE_astar._extract_robot_footprint_vertices_local(robot, reso=reso)
    return max(math.hypot(vx, vy) for vx, vy in verts)


def run_mod_grid_on_scenario(
    scenario_path: str,
    stop_phase: int = 3,
    dp_objective: str = DP_OBJECTIVE_LENGTH,
    disk_collision_mode: str = DISK_COLLISION_OFFLINE,
) -> Tuple[List[float], List[float]]:
    """
    Headless mod_grid runner for the same scenario JSON used by `HA_draw/app.py`.

    Prints a tiny diagnostic and returns (px, py).
    """
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    px0 = scenario.get("pose", {}).get("start", [0.0, 0.0, 0.0])
    gx0 = scenario.get("pose", {}).get("goal", [0.0, 0.0, 0.0])
    sx = float(px0[0])
    sy = float(px0[1])
    gx = float(gx0[0])
    gy = float(gx0[1])

    robot = scenario.get("robot", {})
    from scenario_obstacles import clamp_safety_margin

    safety_margin = clamp_safety_margin(float(robot.get("safety_margin", 0.0)))
    reso = float(scenario.get("map", {}).get("resolution", 1.0))
    rr = _disk_planner_rr_from_scenario(robot, reso=reso)

    ox, oy, reso, map_w, map_h = _obstacle_points_from_app_scenario(scenario)
    from scenario_obstacles import parse_scenario_rects

    rects = list(
        parse_scenario_rects(
            scenario.get("obstacles", {}).get("rects", {}) or {},
            map_w=map_w,
            map_h=map_h,
        ).values()
    )

    timing: Dict[str, float] = {}
    px, py = astar_planning(
        sx,
        sy,
        gx,
        gy,
        ox,
        oy,
        reso,
        rr,
        stop_phase=stop_phase,
        safety_margin=safety_margin,
        obstacle_rects=rects,
        dp_objective=dp_objective,
        timing=timing if stop_phase == 1 else None,
        disk_collision_mode=disk_collision_mode,
    )
    if stop_phase == 1:
        for line in format_disk_phase1_report(
            timing,
            meta={
                "reso": reso,
                "map_w": map_w,
                "map_h": map_h,
                "rr": rr,
                "safety_margin": safety_margin,
                "obstacle_pts": len(ox),
                "collision_mode": disk_collision_mode,
            },
            wall_s=timing.get("total_s"),
        ):
            print(line)
    if px:
        # Diagnostic: how close did the path get to the box boundary?
        dmin = min(
            min(px[i], map_w - px[i], py[i], map_h - py[i]) for i in range(len(px))
        )
        print(
            f"[mod_grid] scenario={os.path.basename(scenario_path)} stop_phase={stop_phase} "
            f"path_pts={len(px)} min_dist_to_box={dmin:.3f}m"
        )
    else:
        print(
            f"[mod_grid] scenario={os.path.basename(scenario_path)} stop_phase={stop_phase} "
            f"NO PATH"
        )
    return px, py


def _main_cli() -> None:
    """
    Minimal CLI for headless testing:
      python mod_grid.py scenario_basic_test.json --stop_phase 1
    """
    if len(sys.argv) < 2:
        print("Usage: python mod_grid.py <scenario.json> [--stop_phase 1|2|3]")
        sys.exit(2)
    scenario_path = sys.argv[1]
    stop_phase = 3
    disk_collision_mode = DISK_COLLISION_OFFLINE
    if "--stop_phase" in sys.argv:
        i = sys.argv.index("--stop_phase")
        try:
            stop_phase = int(sys.argv[i + 1])
        except Exception:
            raise SystemExit("Invalid --stop_phase value")
    if "--disk_collision" in sys.argv:
        i = sys.argv.index("--disk_collision")
        try:
            disk_collision_mode = str(sys.argv[i + 1]).lower().strip()
        except Exception:
            raise SystemExit("Invalid --disk_collision value (use offline or online)")
    run_mod_grid_on_scenario(
        scenario_path,
        stop_phase=stop_phase,
        disk_collision_mode=disk_collision_mode,
    )


if __name__ == "__main__":
    _main_cli()
