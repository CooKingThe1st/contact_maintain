"""
Two-phase holonomic planner (mod_grid):

Phase 1 — Augmented grid A*
    State: (grid_x, grid_y, m_prev) with m_prev = motion index used to enter the cell, or -1 at start.
    Edge cost: c_move + c_risk(d_min) + c_heading(m_prev, m_new)
    - c_risk: piecewise penalty from EDT clearance to obstacles (3 bands, increasing).
    - c_heading: penalty for turning away from previous move (encourages long straight legs).

Phase 2 — CHOMP-style trajectory refinement
    Resamples the polyline, then gradient descent on smoothness + soft obstacle barrier
    using the same distance field; endpoints fixed.

Phase 3 — Straight + circular arc decomposition
    Rounds polyline corners with tangent fillets (G1), clamped by edge length; optional
    clearance check vs obstacle points falls back to the sharp corner if unsafe.

Optional: line-of-sight shortcut after CHOMP (fewer vertices).
"""

from __future__ import annotations

import json
import heapq
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Phase 3 — arc fillets (straight + tangent arc)
_ARC_RADIUS = 0.35  # m; clamped down if edges are short
_ARC_MIN_TURN = math.radians(4.0)  # below this |angle|, keep sharp vertex
_ARC_POINTS_PER_RAD = 10.0  # arc samples ~ proportional to sweep angle
_ARC_CHECK_OBSTACLES = True  # if False, skip clearance validation on fillets


M_PREV_NONE = -1
_NUM_MOTION = 8


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


def _state_index(x: int, y: int, m_in: int, P: base_astar.Para) -> int:
    """Unique int for (x,y,m_in), m_in in -1..7 encoded as 0..8."""
    cell = (y - P.miny) * P.xw + (x - P.minx)
    mk = m_in + 1  # -1 -> 0, 0..7 -> 1..8
    return cell * 9 + mk


class _AugNode:
    __slots__ = ("x", "y", "m_in", "cost", "p_ind")

    def __init__(self, x: int, y: int, m_in: int, cost: float, p_ind: int):
        self.x = x
        self.y = y
        self.m_in = m_in
        self.cost = cost
        self.p_ind = p_ind  # parent state_index


def _heuristic(x: int, y: int, gx: int, gy: int) -> float:
    return math.hypot(x - gx, y - gy)


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
) -> Tuple[List[float], List[float]]:
    motion = _get_motion()
    # Phase 1 should not be more conservative than the baseline grid A*.
    # So we build the hard occupancy map using exactly `rr` (no extra margin).
    # Additional clearance preferences are handled via risk/heading costs.
    occ_rr = float(rr) + float(safety_margin)

    ox_g = [float(x) / reso for x in ox]
    oy_g = [float(y) / reso for y in oy]

    P, obsmap = base_astar.calc_parameters(ox_g, oy_g, occ_rr, reso)
    clearance_m = _build_clearance_meters(obsmap, reso)

    sx_i, sy_i = round(sx / reso), round(sy / reso)
    gx_i, gy_i = round(gx / reso), round(gy / reso)

    def ok_cell(x: int, y: int) -> bool:
        if x <= P.minx or x >= P.maxx or y <= P.miny or y >= P.maxy:
            return False
        return not obsmap[x - P.minx][y - P.miny]

    n_start = _AugNode(sx_i, sy_i, M_PREV_NONE, 0.0, -1)
    start_idx = _state_index(sx_i, sy_i, M_PREV_NONE, P)

    open_entries: Dict[int, _AugNode] = {start_idx: n_start}
    closed: Dict[int, _AugNode] = {}
    pq: List[Tuple[float, int]] = []
    heapq.heappush(pq, (0.0 + _heuristic(sx_i, sy_i, gx_i, gy_i), start_idx))

    goal_idx_found: Optional[int] = None

    while pq:
        f_pop, idx = heapq.heappop(pq)
        if idx in closed:
            continue
        if idx not in open_entries:
            continue
        cur = open_entries.pop(idx)
        closed[idx] = cur

        if cur.x == gx_i and cur.y == gy_i:
            goal_idx_found = idx
            break

        ix, iy = cur.x - P.minx, cur.y - P.miny
        d_here = float(clearance_m[ix, iy])

        for mi, mv in enumerate(motion):
            nx, ny = cur.x + mv[0], cur.y + mv[1]
            if not ok_cell(nx, ny):
                continue

            nix, niy = nx - P.minx, ny - P.miny
            d_there = float(clearance_m[nix, niy])
            # Use destination clearance for risk (what we enter)
            c = (
                _u_cost(mv)
                + _risk_cost(d_there)
                + _HEADING_WEIGHT * _angle_between_moves(cur.m_in, mi, motion)
            )
            g_new = cur.cost + c
            child_idx = _state_index(nx, ny, mi, P)
            if child_idx in closed:
                continue
            h = _heuristic(nx, ny, gx_i, gy_i)
            if child_idx in open_entries:
                if g_new < open_entries[child_idx].cost:
                    open_entries[child_idx] = _AugNode(nx, ny, mi, g_new, idx)
                    heapq.heappush(pq, (g_new + h, child_idx))
            else:
                open_entries[child_idx] = _AugNode(nx, ny, mi, g_new, idx)
                heapq.heappush(pq, (g_new + h, child_idx))

    if goal_idx_found is None:
        # No full path; try vanilla A* fallback
        try:
            return base_astar.astar_planning(sx, sy, gx, gy, ox, oy, reso, occ_rr)
        except Exception:
            # Baseline implementation can raise (e.g., KeyError) when no path exists.
            return [], []

    path_grid: List[Tuple[int, int]] = []
    walk = goal_idx_found
    while walk != -1:
        node = closed[walk]
        path_grid.append((node.x, node.y))
        walk = node.p_ind
    path_grid.reverse()

    pathx = [float(x) * reso for x, _ in path_grid]
    pathy = [float(y) * reso for _, y in path_grid]
    return pathx, pathy


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
) -> Tuple[List[float], List[float]]:
    if len(px) < 2:
        return px, py

    safe_rr = float(rr) + float(safety_margin) + max(_INFLATION_RESO_FACTOR * reso, 0.15)
    ox_g = [float(x) / reso for x in ox]
    oy_g = [float(y) / reso for y in oy]
    P, obsmap = base_astar.calc_parameters(ox_g, oy_g, safe_rr, reso)
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

        # Hard projection: if a point is too close to the obstacle point set,
        # push it outward to the target clearance rather than just reverting.
        # This fixes cases where the initial (Phase 1) path already grazes obstacles.
        target_clear = float(rr) + float(safety_margin) + max(float(_CHOMP_HARD_CLEARANCE_PAD), 0.25 * float(reso))
        dcloud, oxn, oyn = _nearest_obs_pts(qx[1:-1], qy[1:-1])
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

    return qx.tolist(), qy.tolist()


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


def _shortcut_path(
    px: List[float],
    py: List[float],
    ox: List[float],
    oy: List[float],
    clearance: float,
) -> Tuple[List[float], List[float]]:
    if len(px) <= 2:
        return px, py
    outx = [px[0]]
    outy = [py[0]]
    i = 0
    n = len(px)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if _segment_min_distance_to_points(px[i], py[i], px[j], py[j], ox, oy) > clearance:
                break
            j -= 1
        outx.append(px[j])
        outy.append(py[j])
        i = j
    return outx, outy


def _path_min_segment_clearance(
    px: List[float], py: List[float], ox: List[float], oy: List[float]
) -> float:
    if len(px) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(px) - 1):
        d = _segment_min_distance_to_points(px[i], py[i], px[i + 1], py[i + 1], ox, oy)
        if d < best:
            best = d
    return best


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
        if _ARC_CHECK_OBSTACLES and ox:
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


def _arc_clear(
    ox_c: float,
    oy_c: float,
    r: float,
    a0: float,
    sweep: float,
    ox: List[float],
    oy: List[float],
    clearance: float,
) -> bool:
    n = max(5, int(abs(sweep) * _ARC_POINTS_PER_RAD) + 1)
    for k in range(n):
        t = k / float(n - 1) if n > 1 else 0.0
        ang = a0 + t * sweep
        x = ox_c + r * math.cos(ang)
        y = oy_c + r * math.sin(ang)
        if _min_dist_point_to_obstacles(x, y, ox, oy) <= clearance:
            return False
    return True


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
) -> Tuple[List[float], List[float]]:
    """
    Alternative to fillets: fit a sequence of straight segments and circular arcs
    that minimizes the number of primitives (each straight or arc counts as 1).

    Implementation is a constrained DP over the given polyline vertices.
    - Straight candidate: connect i->j if clearance holds for the segment.
    - Arc candidate: connect i->j via circle through (i, k, j) for a limited set
      of midpoints k; accept if arc passing through k is collision-free.
    """
    n = len(px)
    if n < 2:
        if return_primitives:
            return px, py, []
        return px, py

    # DP arrays
    INF = 10**9
    best_cost = [INF] * n
    best_prev: List[Optional[Tuple[int, str, dict]]] = [None] * n
    best_cost[0] = 0

    max_span = min(30, n - 1)  # keep candidate set manageable

    for j in range(1, n):
        i_min = max(0, j - max_span)
        for i in range(i_min, j):
            if best_cost[i] >= INF:
                continue
            # Straight candidate
            if _segment_min_distance_to_points(px[i], py[i], px[j], py[j], ox, oy) > clearance:
                c = best_cost[i] + 1
                if c < best_cost[j]:
                    best_cost[j] = c
                    best_prev[j] = (i, "S", {"x0": px[i], "y0": py[i], "x1": px[j], "y1": py[j]})

            # Arc candidate: choose a few midpoints k between i and j
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
                    if not _arc_clear(ocx, ocy, r, a0, sweep, ox, oy, clearance):
                        continue
                    c = best_cost[i] + 1
                    if c < best_cost[j]:
                        best_cost[j] = c
                        best_prev[j] = (
                            i,
                            "A",
                            {"ocx": ocx, "ocy": ocy, "r": r, "a0": a0, "sweep": sweep},
                        )

                # Additional arc family: endpoint tangency + small discrete radii.
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
                            if not _arc_clear(ocx, ocy, r, a0, sweep, ox, oy, clearance):
                                continue
                            c = best_cost[i] + 1
                            if c < best_cost[j]:
                                best_cost[j] = c
                                best_prev[j] = (
                                    i,
                                    "A",
                                    {"ocx": ocx, "ocy": ocy, "r": r, "a0": a0, "sweep": sweep},
                                )

    if best_prev[-1] is None:
        # No compression found; fall back to original polyline.
        if return_primitives:
            return [float(x) for x in px], [float(y) for y in py], _polyline_straight_primitives(
                px, py
            )
        return px, py

    # Reconstruct primitives backward
    prims: List[Tuple[str, dict]] = []
    cur = n - 1
    while cur != 0:
        prev = best_prev[cur]
        if prev is None:
            break
        i, typ, params = prev
        prims.append((typ, params))
        cur = i
    prims.reverse()

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
            n_arc = max(5, int(abs(sweep) * _ARC_POINTS_PER_RAD) + 1)
            for kk in range(1, n_arc + 1):
                t = kk / float(n_arc)
                ang = a0 + t * sweep
                emit(ocx + r * math.cos(ang), ocy + r * math.sin(ang))

    if return_primitives:
        return outx, outy, prims
    return outx, outy


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
) -> Tuple[List[float], List[float]]:
    """
    mod_grid pipeline with early exit / alternatives:
    Phase 1: augmented A*
    Phase 2: Phase 1 → CHOMP → shortcut (final optimization)
    Phase 3: Phase 1 → shortcut → straight+arc fillets (alternative, skips CHOMP)
    """
    if stop_phase not in (1, 2, 3):
        raise ValueError(f"stop_phase must be 1, 2, or 3 (got {stop_phase})")

    px, py = phase1_augmented_astar(
        sx, sy, gx, gy, ox, oy, reso, rr, safety_margin=float(safety_margin)
    )
    if len(px) < 2:
        return px, py

    if stop_phase == 1:
        return px, py

    safe_rr = float(rr) + float(safety_margin) + max(_INFLATION_RESO_FACTOR * reso, 0.15)
    eff_rr = float(rr) + float(safety_margin)
    if stop_phase == 2:
        p1x, p1y = px, py
        cpx, cpy = phase2_chomp(px, py, ox, oy, reso, rr, safety_margin=float(safety_margin))
        # CHOMP may still be problematic in sparse/thickened line maps; enforce
        # segment-level clearance and fall back to phase-1 path if violated.
        if _path_min_segment_clearance(cpx, cpy, ox, oy) <= eff_rr:
            cpx, cpy = p1x, p1y
        spx, spy = _shortcut_path(cpx, cpy, ox, oy, safe_rr)
        if _path_min_segment_clearance(spx, spy, ox, oy) <= eff_rr:
            return cpx, cpy
        return spx, spy

    # stop_phase == 3: alternative corner rounding path (skip CHOMP)
    # Long chord shortcuts use a slightly inflated clearance (same as phase 2).
    safe_rr = float(rr) + float(safety_margin) + 0.05
    eff_rr = float(rr) + float(safety_margin)
    spx, spy = _shortcut_path(px, py, ox, oy, safe_rr)
    # phase3_min_segments must use eff_rr: consecutive vertices on the shortcut path are
    # only guaranteed to satisfy phase-1 / disk-rr feasibility. _shortcut_path does not
    # re-check clearance on forced single-step hops, so requiring safe_rr here can make
    # the DP fail to chain through narrow corridors while phase 1 still succeeds.
    if return_primitives:
        px, py, prims = phase3_min_segments(spx, spy, ox, oy, eff_rr, return_primitives=True)
        if _path_min_segment_clearance(px, py, ox, oy) <= eff_rr:
            return list(spx), list(spy), _polyline_straight_primitives(spx, spy)
        return px, py, prims
    px, py = phase3_min_segments(spx, spy, ox, oy, eff_rr)
    if _path_min_segment_clearance(px, py, ox, oy) <= eff_rr:
        return spx, spy
    return px, py


def _obstacle_points_from_app_scenario(
    scenario: dict,
) -> Tuple[List[float], List[float], float, float, float]:
    """
    Reconstruct obstacle point cloud exactly like `HA_draw/app.py`:
    - boundary points along the map box edges
    - rectangle obstacles: grid-sampled points within each rect
    - polyline obstacles: thickened line samples

    Returns:
        (ox, oy, reso, map_w, map_h)
    """
    m = scenario.get("map", {})
    map_w = float(m.get("width", 60.0))
    map_h = float(m.get("height", 40.0))
    reso = float(m.get("resolution", 1.0))

    draw = scenario.get("draw", {})
    line_thickness = float(draw.get("line_thickness", 1.0))

    obs = scenario.get("obstacles", {})
    rects: Dict[str, List[float]] = obs.get("rects", {})
    lines: Dict[str, List[List[float]]] = obs.get("lines", {})

    # Boundary obstacle points (same as app._boundary_points).
    ox: List[float] = []
    oy: List[float] = []
    w = int(map_w / reso)
    h = int(map_h / reso)
    r = reso
    for i in range(w + 1):
        x = i * r
        ox += [x, x]
        oy += [0.0, h * r]
    for j in range(h + 1):
        y = j * r
        ox += [0.0, w * r]
        oy += [y, y]

    # Rectangle obstacles (same as app._obstacle_points rect sampling).
    for rect in rects.values():
        if len(rect) != 4:
            continue
        x, y, w_rect, h_rect = map(float, rect)
        x0 = max(0.0, x)
        y0 = max(0.0, y)
        x1 = min(map_w, x + w_rect)
        y1 = min(map_h, y + h_rect)
        xi = np.arange(x0, x1 + 1e-6, r)
        yi = np.arange(y0, y1 + 1e-6, r)
        for xx in xi:
            for yy in yi:
                ox.append(float(xx))
                oy.append(float(yy))

    # Polyline obstacles (same as app._obstacle_points line thickening).
    thick = max(0.2, line_thickness)
    samples = max(2, int(thick / r))
    for pts in lines.values():
        if not pts or len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
            seg_len = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
            n = max(2, int(seg_len / r) * 2)
            for t in np.linspace(0.0, 1.0, n):
                cx = x0 + t * (x1 - x0)
                cy = y0 + t * (y1 - y0)
                for dx in np.linspace(-thick / 2.0, thick / 2.0, samples):
                    for dy in np.linspace(-thick / 2.0, thick / 2.0, samples):
                        ox.append(float(cx + dx))
                        oy.append(float(cy + dy))

    return ox, oy, reso, map_w, map_h


def run_mod_grid_on_scenario(
    scenario_path: str,
    stop_phase: int = 3,
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
    safety_margin = float(robot.get("safety_margin", 0.0))
    width = float(robot.get("width", 2.0))
    length = float(robot.get("length", 3.0))
    rr = max(width, length) / 2.0

    ox, oy, reso, map_w, map_h = _obstacle_points_from_app_scenario(scenario)

    px, py = astar_planning(
        sx, sy, gx, gy, ox, oy, reso, rr, stop_phase=stop_phase, safety_margin=safety_margin
    )
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
    if "--stop_phase" in sys.argv:
        i = sys.argv.index("--stop_phase")
        try:
            stop_phase = int(sys.argv[i + 1])
        except Exception:
            raise SystemExit("Invalid --stop_phase value")
    run_mod_grid_on_scenario(scenario_path, stop_phase=stop_phase)


if __name__ == "__main__":
    _main_cli()
