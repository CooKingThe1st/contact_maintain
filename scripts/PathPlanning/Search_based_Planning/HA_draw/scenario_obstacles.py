#!/usr/bin/env python3
"""
Shared HA_draw scenario obstacle parsing and geometry helpers.

Rectangle encodings (backward compatible):
  - 4 numbers [x, y, w, h]: axis-aligned, lower-left corner (legacy app.py)
  - 5 numbers [cx, cy, w, h, angle_deg]: center + size + CCW rotation in degrees
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

RectValues = Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float]]

# Minimum clearance inflation for disk / holonomic planners (meters).
MIN_SAFETY_MARGIN = 0.05

# Swarm pusher fleet robot used in Magnum controller tests (NOT the pushed object).
# Matches ROBOT_RADIUS in test_magnum_diffdrive_control.py / disc_bumper.urdf (r=0.06 m).
SWARM_PUSHER_ROBOT_RADIUS_M = 0.06
SWARM_PUSHER_ROBOT_DIAMETER_M = 2.0 * SWARM_PUSHER_ROBOT_RADIUS_M


def swarm_pusher_min_safety_margin_m(*, margin_ge_swarm_pusher_size: bool = True) -> float:
    """Minimum safety margin floor when the swarm-pusher checkbox is enabled."""
    if margin_ge_swarm_pusher_size:
        return float(SWARM_PUSHER_ROBOT_DIAMETER_M)
    return float(MIN_SAFETY_MARGIN)


def clamp_safety_margin(safety_margin: float, min_margin: float = MIN_SAFETY_MARGIN) -> float:
    """Enforce a minimum safety margin on planner and UI values."""
    return max(float(min_margin), float(safety_margin))


@dataclass(frozen=True)
class ObstacleRect:
    """Centered rectangle in world frame."""

    cx: float
    cy: float
    w: float
    h: float
    angle_deg: float = 0.0

    @property
    def angle_rad(self) -> float:
        return math.radians(float(self.angle_deg))

    @property
    def xmin(self) -> float:
        return min(p[0] for p in self.corners())

    @property
    def ymin(self) -> float:
        return min(p[1] for p in self.corners())

    @property
    def xmax(self) -> float:
        return max(p[0] for p in self.corners())

    @property
    def ymax(self) -> float:
        return max(p[1] for p in self.corners())

    def corners(self) -> List[Tuple[float, float]]:
        """Four corners CCW in world frame."""
        hw = 0.5 * float(self.w)
        hh = 0.5 * float(self.h)
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        c = math.cos(self.angle_rad)
        s = math.sin(self.angle_rad)
        out: List[Tuple[float, float]] = []
        for lx, ly in local:
            wx = self.cx + c * lx - s * ly
            wy = self.cy + s * lx + c * ly
            out.append((float(wx), float(wy)))
        return out

    def to_json_values(self) -> List[float]:
        if abs(self.angle_deg) < 1e-9:
            return [
                float(self.cx - 0.5 * self.w),
                float(self.cy - 0.5 * self.h),
                float(self.w),
                float(self.h),
            ]
        return [float(self.cx), float(self.cy), float(self.w), float(self.h), float(self.angle_deg)]

    def contains_point(self, x: float, y: float) -> bool:
        """Point-in-rotated-rect test."""
        c = math.cos(self.angle_rad)
        s = math.sin(self.angle_rad)
        dx = float(x) - self.cx
        dy = float(y) - self.cy
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        return abs(lx) <= 0.5 * self.w + 1e-9 and abs(ly) <= 0.5 * self.h + 1e-9


def _distance_point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def min_distance_point_to_rect(x: float, y: float, rect: ObstacleRect) -> float:
    """Distance from a point to a filled OBB (0 if inside). Fast local-frame clamp."""
    return _distance_point_to_obb(float(x), float(y), rect)


def _distance_point_to_obb(x: float, y: float, rect: ObstacleRect) -> float:
    dx = x - float(rect.cx)
    dy = y - float(rect.cy)
    c = math.cos(rect.angle_rad)
    s = math.sin(rect.angle_rad)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    half_w = 0.5 * float(rect.w)
    half_h = 0.5 * float(rect.h)
    closest_x = max(-half_w, min(local_x, half_w))
    closest_y = max(-half_h, min(local_y, half_h))
    return math.hypot(local_x - closest_x, local_y - closest_y)


def _distance_segment_to_segment(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> float:
    """Minimum distance between two closed line segments (O(1))."""
    ux, uy = bx - ax, by - ay
    vx, vy = dx - cx, dy - cy
    wx, wy = ax - cx, ay - cy
    a = ux * ux + uy * uy
    b = ux * vx + uy * vy
    c = vx * vx + vy * vy
    d = ux * wx + uy * wy
    e = vx * wx + vy * wy
    denom = a * c - b * b
    if denom < 1e-18:
        sc = 0.0
        tc = e / c if c > 1e-18 else 0.0
    else:
        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom
    sc = max(0.0, min(1.0, sc))
    tc = max(0.0, min(1.0, tc))
    px = ax + sc * ux
    py = ay + sc * uy
    qx = cx + tc * vx
    qy = cy + tc * vy
    return math.hypot(px - qx, py - qy)


def _distance_aabb_to_point(ax0: float, ay0: float, ax1: float, ay1: float, px: float, py: float) -> float:
    qx = min(max(px, ax0), ax1)
    qy = min(max(py, ay0), ay1)
    return math.hypot(px - qx, py - qy)


def distance_aabb_to_obb(ax0: float, ay0: float, ax1: float, ay1: float, rect: ObstacleRect) -> float:
    """
    Exact minimum Euclidean distance between a filled axis-aligned box and a filled OBB.

    O(1): four AABB-corner / OBB tests, four OBB-corner / AABB tests, and sixteen
    edge-pair segment distances cover vertex-vertex, vertex-edge, and edge-edge
    closest features for separated rectangles.
    """
    best = float("inf")
    for px, py in ((ax0, ay0), (ax1, ay0), (ax1, ay1), (ax0, ay1)):
        best = min(best, _distance_point_to_obb(px, py, rect))
    for px, py in rect.corners():
        best = min(best, _distance_aabb_to_point(ax0, ay0, ax1, ay1, px, py))

    aabb_edges = (
        (ax0, ay0, ax1, ay0),
        (ax1, ay0, ax1, ay1),
        (ax1, ay1, ax0, ay1),
        (ax0, ay1, ax0, ay0),
    )
    obb_corners = rect.corners()
    for i in range(4):
        ex0, ey0 = obb_corners[i]
        ex1, ey1 = obb_corners[(i + 1) % 4]
        for ax, ay, bx, by in aabb_edges:
            best = min(best, _distance_segment_to_segment(ax, ay, bx, by, ex0, ey0, ex1, ey1))
    return best


def grid_cell_center_world(gx: int, gy: int, reso: float) -> Tuple[float, float]:
    """World position [m] at the center of grid cell with absolute indices ``(gx, gy)``."""
    r = float(reso)
    return (float(gx) + 0.5) * r, (float(gy) + 0.5) * r


def grid_cell_aabb_world(gx: int, gy: int, reso: float) -> Tuple[float, float, float, float]:
    """Axis-aligned world bounds [m] of grid cell ``(gx, gy)``."""
    cx, cy = grid_cell_center_world(gx, gy, reso)
    half = 0.5 * float(reso)
    return cx - half, cy - half, cx + half, cy + half


def grid_cell_aabb_grid(gx: int, gy: int) -> Tuple[float, float, float, float]:
    """Axis-aligned bounds in grid-index units for absolute cell indices ``(gx, gy)``."""
    gx_f, gy_f = float(gx), float(gy)
    return gx_f, gy_f, gx_f + 1.0, gy_f + 1.0


def cell_square_disk_hits_point(
    gx: int, gy: int, reso: float, px: float, py: float, r_eff: float
) -> bool:
    """True when a disk of radius ``r_eff`` centered somewhere in the cell could cover ``(px, py)``."""
    ax0, ay0, ax1, ay1 = grid_cell_aabb_world(gx, gy, reso)
    return _distance_aabb_to_point(ax0, ay0, ax1, ay1, px, py) <= float(r_eff) + 1e-9


def cell_square_disk_hits_obb(gx: int, gy: int, reso: float, rect: ObstacleRect, r_eff: float) -> bool:
    """True when a disk of radius ``r_eff`` centered somewhere in the cell could overlap ``rect``."""
    ax0, ay0, ax1, ay1 = grid_cell_aabb_world(gx, gy, reso)
    return distance_aabb_to_obb(ax0, ay0, ax1, ay1, rect) <= float(r_eff) + 1e-9


def cell_square_disk_hits_point_grid(
    gx: int, gy: int, px: float, py: float, r_eff_grid: float
) -> bool:
    """Grid-unit variant for point-obstacle rasterization inside ``calc_obsmap``."""
    ax0, ay0, ax1, ay1 = grid_cell_aabb_grid(gx, gy)
    return _distance_aabb_to_point(ax0, ay0, ax1, ay1, px, py) <= float(r_eff_grid) + 1e-9


def grid_cell_disk_blocked(
    gx: int,
    gy: int,
    *,
    ox_g: Sequence[float],
    oy_g: Sequence[float],
    r_eff_grid: float,
    r_eff_world: float,
    reso: float,
    rects: Sequence[ObstacleRect] = (),
) -> bool:
    """On-demand disk occupancy test for one grid cell (points + OBB rects)."""
    for oxx, oyy in zip(ox_g, oy_g):
        if cell_square_disk_hits_point_grid(gx, gy, float(oxx), float(oyy), r_eff_grid):
            return True
    for rect in rects:
        if cell_square_disk_hits_obb(gx, gy, float(reso), rect, float(r_eff_world)):
            return True
    return False


def _cell_center_world(ix: int, iy: int, P, reso: float) -> Tuple[float, float]:
    return grid_cell_center_world(int(P.minx + ix), int(P.miny + iy), reso)


def _grid_influence_reach(rect: ObstacleRect, r_eff: float, reso: float) -> float:
    """Conservative world-space reach for iterating cells that could touch ``rect``."""
    half_obb = 0.5 * math.hypot(float(rect.w), float(rect.h))
    half_cell = 0.5 * math.sqrt(2.0) * float(reso)
    return half_obb + float(r_eff) + half_cell


def _obb_influence_window(rect: ObstacleRect, r_eff: float, reso: float, P) -> Tuple[int, int, int, int]:
    """Grid index bounds [ix0, ix1] x [iy0, iy1] (inclusive) to visit for one OBB."""
    reach = _grid_influence_reach(rect, r_eff, reso)
    x0_w = float(rect.cx) - reach
    x1_w = float(rect.cx) + reach
    y0_w = float(rect.cy) - reach
    y1_w = float(rect.cy) + reach
    ix0 = max(0, int(math.floor(x0_w / reso - float(P.minx) - 0.5)))
    ix1 = min(P.xw - 1, int(math.ceil(x1_w / reso - float(P.minx) - 0.5)))
    iy0 = max(0, int(math.floor(y0_w / reso - float(P.miny) - 0.5)))
    iy1 = min(P.yw - 1, int(math.ceil(y1_w / reso - float(P.miny) - 0.5)))
    return ix0, ix1, iy0, iy1


def apply_rect_disk_obstacles_to_obsmap(
    obsmap: List[List[bool]],
    P,
    rects: Sequence[ObstacleRect],
    r_eff: float,
    reso: float,
) -> None:
    """
    Conservative square-cell rasterization for OBB obstacles.

    Marks a cell occupied when a disk of effective radius ``r_eff`` (= ``rr +
    safety_margin``) centered anywhere in the cell could overlap the OBB:
    ``distance(cell_square, OBB) <= r_eff``. Pair with
    ``grid_diagonal_move_clear`` during 8-way search.
    """
    if not rects:
        return
    reso = float(reso)
    r_eff = float(r_eff)
    for rect in rects:
        ix0, ix1, iy0, iy1 = _obb_influence_window(rect, r_eff, reso, P)
        if ix0 > ix1 or iy0 > iy1:
            continue
        for ix in range(ix0, ix1 + 1):
            gx = int(P.minx + ix)
            for iy in range(iy0, iy1 + 1):
                gy = int(P.miny + iy)
                if cell_square_disk_hits_obb(gx, gy, reso, rect, r_eff):
                    obsmap[ix][iy] = True


def min_distance_point_to_rects(
    x: float, y: float, rects: Sequence[ObstacleRect]
) -> float:
    if not rects:
        return float("inf")
    return min(min_distance_point_to_rect(x, y, r) for r in rects)


@dataclass(frozen=True)
class DiskValidationContext:
    """Precomputed data for hierarchical primitive-vs-OBB disk checks."""

    rr: float
    reso: float
    ox: Tuple[float, ...]
    oy: Tuple[float, ...]
    rects: Tuple[ObstacleRect, ...]
    rect_aabbs: Tuple[Tuple[float, float, float, float], ...]


def build_disk_validation_context(
    rr: float,
    reso: float,
    ox: Sequence[float],
    oy: Sequence[float],
    rects: Sequence[ObstacleRect],
) -> DiskValidationContext:
    """Build context for phase-3 primitive validation (Filter 1–3 funnel)."""
    r = float(rr)
    inflated: List[Tuple[float, float, float, float]] = []
    for rect in rects:
        half_diag = 0.5 * math.hypot(float(rect.w), float(rect.h))
        reach = half_diag + r
        inflated.append(
            (
                float(rect.cx) - reach,
                float(rect.cy) - reach,
                float(rect.cx) + reach,
                float(rect.cy) + reach,
            )
        )
    return DiskValidationContext(
        rr=r,
        reso=float(reso),
        ox=tuple(float(x) for x in ox),
        oy=tuple(float(y) for y in oy),
        rects=tuple(rects),
        rect_aabbs=tuple(inflated),
    )


def _aabb_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _line_aabb(x0: float, y0: float, x1: float, y1: float, pad: float) -> Tuple[float, float, float, float]:
    return (
        min(x0, x1) - pad,
        min(y0, y1) - pad,
        max(x0, x1) + pad,
        max(y0, y1) + pad,
    )


def _circle_aabb(cx: float, cy: float, radius: float, pad: float) -> Tuple[float, float, float, float]:
    r = float(radius) + pad
    return (cx - r, cy - r, cx + r, cy + r)


def _adaptive_sample_count(length: float, reso: float) -> int:
    """Filter 2: one sample per grid cell along the primitive."""
    return max(2, int(math.ceil(max(float(length), 1e-9) / max(float(reso), 1e-9))))


def _point_disk_hits_obstacles(px: float, py: float, ctx: DiskValidationContext) -> bool:
    """Filter 3: exact disk center vs OBB + boundary points in range."""
    for rect in ctx.rects:
        if _distance_point_to_obb(px, py, rect) <= ctx.rr + 1e-9:
            return True
    for ox0, oy0 in zip(ctx.ox, ctx.oy):
        if math.hypot(px - ox0, py - oy0) <= ctx.rr + 1e-9:
            return True
    return False


def _filter_rect_indices(prim_aabb: Tuple[float, float, float, float], ctx: DiskValidationContext) -> List[int]:
    """Filter 1: keep only OBBs whose inflated AABB overlaps the primitive AABB."""
    out: List[int] = []
    for idx, raabb in enumerate(ctx.rect_aabbs):
        if _aabb_overlap(prim_aabb, raabb):
            out.append(idx)
    return out


def _sample_points_on_line(x0: float, y0: float, x1: float, y1: float, n: int) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for k in range(n):
        t = k / float(n - 1) if n > 1 else 0.0
        pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return pts


def _sample_points_on_arc(
    ocx: float,
    ocy: float,
    radius: float,
    a0: float,
    sweep: float,
    n: int,
) -> List[Tuple[float, float]]:
    """Centerline samples on a circular arc (planning validation only — not viz)."""
    r = max(float(radius), 1e-9)
    pts: List[Tuple[float, float]] = []
    for k in range(n):
        t = k / float(n - 1) if n > 1 else 0.0
        ang = a0 + t * sweep
        pts.append((ocx + r * math.cos(ang), ocy + r * math.sin(ang)))
    return pts


def line_disk_feasible(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ctx: DiskValidationContext,
) -> bool:
    """Hierarchical funnel: straight segment vs disk radius ``ctx.rr``."""
    length = math.hypot(x1 - x0, y1 - y0)
    prim_aabb = _line_aabb(x0, y0, x1, y1, ctx.rr)
    if ctx.rects and not _filter_rect_indices(prim_aabb, ctx):
        # No OBB overlap; still must check boundary points inside the window.
        pass
    n = _adaptive_sample_count(length, ctx.reso)
    for px, py in _sample_points_on_line(x0, y0, x1, y1, n):
        if _point_disk_hits_obstacles(px, py, ctx):
            return False
    return True


def arc_disk_feasible(
    ocx: float,
    ocy: float,
    radius: float,
    a0: float,
    sweep: float,
    ctx: DiskValidationContext,
) -> bool:
    """
    Hierarchical funnel: circular arc vs disk radius ``ctx.rr``.

    Used by phase-3 primitive DP / shortcut gates only.  Footprint visualization
    in ``app.py`` densifies the planner polyline separately and does not call this.
    """
    r = max(float(radius), 1e-9)
    arc_len = abs(float(sweep)) * r
    prim_aabb = _circle_aabb(ocx, ocy, r, ctx.rr)
    if ctx.rects and not _filter_rect_indices(prim_aabb, ctx):
        pass
    n = _adaptive_sample_count(arc_len, ctx.reso)
    for px, py in _sample_points_on_arc(ocx, ocy, r, a0, sweep, n):
        if _point_disk_hits_obstacles(px, py, ctx):
            return False
    return True


def primitive_disk_feasible(
    typ: str,
    params: dict,
    ctx: DiskValidationContext,
) -> bool:
    """Validate a straight (``S``) or arc (``A``) primitive."""
    if typ == "S":
        return line_disk_feasible(
            float(params["x0"]),
            float(params["y0"]),
            float(params["x1"]),
            float(params["y1"]),
            ctx,
        )
    if typ == "A":
        return arc_disk_feasible(
            float(params["ocx"]),
            float(params["ocy"]),
            float(params["r"]),
            float(params["a0"]),
            float(params["sweep"]),
            ctx,
        )
    return False


def grid_diagonal_move_clear(
    x: int, y: int, dx: int, dy: int, P, obsmap: List[List[bool]]
) -> bool:
    """
    Shared-neighbor gate for 8-way grid search.

    Cardinal moves always pass (caller's responsibility to check destination).
    Diagonal moves require both adjacent cardinal cells to be free.
    """
    if dx == 0 or dy == 0:
        return True
    for nx, ny in ((x + dx, y), (x, y + dy)):
        if nx <= P.minx or nx >= P.maxx or ny <= P.miny or ny >= P.maxy:
            return False
        if obsmap[nx - P.minx][ny - P.miny]:
            return False
    return True


def disk_path_min_clearance_to_rects(
    px: Sequence[float],
    py: Sequence[float],
    rr: float,
    rects: Sequence[ObstacleRect],
    reso: float = 0.2,
) -> float:
    """Minimum centerline clearance of a disk radius ``rr`` along a polyline vs rects."""
    if len(px) < 1 or not rects:
        return float("inf")
    best = float("inf")
    for i in range(len(px)):
        best = min(best, min_distance_point_to_rects(float(px[i]), float(py[i]), rects) - float(rr))
    if len(px) < 2:
        return best
    for i in range(len(px) - 1):
        best = min(
            best,
            segment_min_clearance_to_rects(
                float(px[i]),
                float(py[i]),
                float(px[i + 1]),
                float(py[i + 1]),
                rects,
                reso=reso,
            )
            - float(rr),
        )
    return best


def segment_min_clearance_to_rects(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rects: Sequence[ObstacleRect],
    *,
    reso: float = 0.2,
) -> float:
    """Minimum distance from a line segment to any rectangle (0 if intersecting)."""
    if not rects:
        return float("inf")
    best = float("inf")
    seg_len = math.hypot(x1 - x0, y1 - y0)
    n_sub = max(2, int(math.ceil(seg_len / max(0.08, 0.35 * float(reso))))) + 1
    for k in range(n_sub + 1):
        t = k / float(n_sub)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        best = min(best, min_distance_point_to_rects(x, y, rects))
    return best


def segment_min_clearance(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ox: Sequence[float],
    oy: Sequence[float],
    *,
    obstacle_rects: Sequence[ObstacleRect] = (),
    reso: float = 0.2,
) -> float:
    """Minimum distance from a segment to obstacle points and scenario rectangles."""
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
    if obstacle_rects:
        best = min(best, segment_min_clearance_to_rects(x0, y0, x1, y1, obstacle_rects, reso=reso))
    return best


def disk_edge_feasible(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rr: float,
    ox: Sequence[float],
    oy: Sequence[float],
    *,
    obstacle_rects: Sequence[ObstacleRect] = (),
    reso: float = 0.2,
) -> bool:
    """True when a disk of radius ``rr`` can move along a segment without hitting obstacles."""
    return (
        segment_min_clearance(x0, y0, x1, y1, ox, oy, obstacle_rects=obstacle_rects, reso=reso)
        > float(rr) + 1e-9
    )


def obstacle_rects_from_se_values(
    obstacle_rects: Sequence[Tuple[float, ...]],
) -> List[ObstacleRect]:
    """Convert mod_grid_SE ``(cx,cy,w,h,angle)`` tuples to ``ObstacleRect``."""
    out: List[ObstacleRect] = []
    for t in obstacle_rects:
        if len(t) == 4:
            rx, ry, rw, rh = map(float, t)
            out.append(ObstacleRect(cx=rx + 0.5 * rw, cy=ry + 0.5 * rh, w=rw, h=rh))
        elif len(t) == 5:
            cx, cy, rw, rh, ang = map(float, t)
            out.append(ObstacleRect(cx=cx, cy=cy, w=rw, h=rh, angle_deg=ang))
        else:
            out.append(parse_rect_values(t))
    return out


def rasterize_rect_boundary(rect: ObstacleRect, reso: float) -> Tuple[List[float], List[float]]:
    """Sample points along the rectangle perimeter (for denser point-cloud inflation)."""
    ox: List[float] = []
    oy: List[float] = []
    step = max(float(reso) * 0.5, 0.05)
    corners = rect.corners()
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        edge_len = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(math.ceil(edge_len / step)) + 1)
        for k in range(n):
            t = k / float(n - 1) if n > 1 else 0.0
            ox.append(float(x0 + t * (x1 - x0)))
            oy.append(float(y0 + t * (y1 - y0)))
    return ox, oy


def parse_rect_values(
    vals: Sequence[float],
    *,
    map_w: float = 1e9,
    map_h: float = 1e9,
) -> ObstacleRect:
    n = len(vals)
    if n == 5:
        cx, cy, w, h, angle_deg = (float(v) for v in vals)
        return ObstacleRect(cx=cx, cy=cy, w=max(0.0, w), h=max(0.0, h), angle_deg=angle_deg)
    if n != 4:
        raise ValueError(f"Expected 4 or 5 numbers for rect, got {n}")

    a, b, c, d = (float(v) for v in vals)
    eps = 1e-9

    # Legacy lower-left [x,y,w,h]
    r_xywh = ObstacleRect(cx=a + 0.5 * c, cy=b + 0.5 * d, w=max(0.0, c), h=max(0.0, d))
    xywh_ok = (
        r_xywh.w > eps
        and r_xywh.h > eps
        and (r_xywh.cx - 0.5 * r_xywh.w) >= -eps
        and (r_xywh.cy - 0.5 * r_xywh.h) >= -eps
        and (r_xywh.cx + 0.5 * r_xywh.w) <= map_w + 1e-3
        and (r_xywh.cy + 0.5 * r_xywh.h) <= map_h + 1e-3
    )

    # Older xmin,ymin,xmax,ymax
    xmin, xmax = (a, c) if a <= c else (c, a)
    ymin, ymax = (b, d) if b <= d else (d, b)
    r_minmax = ObstacleRect(
        cx=0.5 * (xmin + xmax),
        cy=0.5 * (ymin + ymax),
        w=max(0.0, xmax - xmin),
        h=max(0.0, ymax - ymin),
    )
    minmax_ok = (
        r_minmax.w > eps
        and r_minmax.h > eps
        and -map_w <= r_minmax.xmin <= 2 * map_w
        and -map_h <= r_minmax.ymin <= 2 * map_h
    )

    if xywh_ok and not minmax_ok:
        return r_xywh
    if minmax_ok and not xywh_ok:
        return r_minmax
    return r_xywh


def parse_scenario_rects(
    rects_raw: Dict[str, Sequence[float]],
    *,
    map_w: float,
    map_h: float,
) -> Dict[str, ObstacleRect]:
    out: Dict[str, ObstacleRect] = {}
    for name, vals in rects_raw.items():
        out[name] = parse_rect_values(vals, map_w=map_w, map_h=map_h)
    return out


def boundary_points(map_w: float, map_h: float, reso: float) -> Tuple[List[float], List[float]]:
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
    return ox, oy


def rasterize_rect(rect: ObstacleRect, reso: float, map_w: float, map_h: float) -> Tuple[List[float], List[float]]:
    """Grid-sample interior of rect (axis-aligned bbox fill for rotated rects)."""
    ox: List[float] = []
    oy: List[float] = []
    r = float(reso)
    x0 = max(0.0, rect.xmin)
    y0 = max(0.0, rect.ymin)
    x1 = min(map_w, rect.xmax)
    y1 = min(map_h, rect.ymax)
    if x1 <= x0 or y1 <= y0:
        return ox, oy
    for xx in np.arange(x0, x1 + 1e-6, r):
        for yy in np.arange(y0, y1 + 1e-6, r):
            if rect.contains_point(float(xx), float(yy)):
                ox.append(float(xx))
                oy.append(float(yy))
    return ox, oy


def rasterize_line_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    reso: float,
    line_thickness: float,
) -> Tuple[List[float], List[float]]:
    ox: List[float] = []
    oy: List[float] = []
    r = float(reso)
    thick = max(0.2, float(line_thickness))
    samples = max(2, int(thick / r))
    seg_len = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
    n = max(2, int(seg_len / r) * 2)
    for t in np.linspace(0.0, 1.0, n):
        cx = x0 + t * (x1 - x0)
        cy = y0 + t * (y1 - y0)
        for dx in np.linspace(-thick / 2.0, thick / 2.0, samples):
            for dy in np.linspace(-thick / 2.0, thick / 2.0, samples):
                ox.append(float(cx + dx))
                oy.append(float(cy + dy))
    return ox, oy


def obstacle_points_from_scenario(scenario: dict) -> Tuple[List[float], List[float], float, float, float]:
    """Reconstruct obstacle point cloud like HA_draw app.py."""
    m = scenario.get("map", {})
    map_w = float(m.get("width", 60.0))
    map_h = float(m.get("height", 40.0))
    reso = float(m.get("resolution", 1.0))

    draw = scenario.get("draw", {})
    line_thickness = float(draw.get("line_thickness", 1.0))

    obs = scenario.get("obstacles", {})
    rects_raw = obs.get("rects", {}) or {}
    lines: Dict[str, List[List[float]]] = obs.get("lines", {}) or {}

    ox, oy = boundary_points(map_w, map_h, reso)

    for vals in rects_raw.values():
        rect = parse_rect_values(vals, map_w=map_w, map_h=map_h)
        rx, ry = rasterize_rect(rect, reso, map_w, map_h)
        ox.extend(rx)
        oy.extend(ry)

    for pts in lines.values():
        if not pts or len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            x0, y0 = map(float, pts[i])
            x1, y1 = map(float, pts[i + 1])
            lx, ly = rasterize_line_segment(
                x0, y0, x1, y1, reso=reso, line_thickness=line_thickness
            )
            ox.extend(lx)
            oy.extend(ly)

    return ox, oy, reso, map_w, map_h


def obstacle_points_for_disk_planner(
    scenario: dict,
) -> Tuple[List[float], List[float], float, float, float]:
    """
    Obstacle samples for disk (grid_astar / mod_grid) planners.

    Map boundary and thick line obstacles stay as a point cloud.  Rectangles are
    omitted here because ``apply_rect_disk_obstacles_to_obsmap`` already applies
    the continuous rect⊕disk model; rasterizing rect interiors into the cloud as
    well double-blocks corridors.
    """
    m = scenario.get("map", {})
    map_w = float(m.get("width", 60.0))
    map_h = float(m.get("height", 40.0))
    reso = float(m.get("resolution", 1.0))

    draw = scenario.get("draw", {})
    line_thickness = float(draw.get("line_thickness", 1.0))

    obs = scenario.get("obstacles", {})
    lines: Dict[str, List[List[float]]] = obs.get("lines", {}) or {}

    ox, oy = boundary_points(map_w, map_h, reso)

    for pts in lines.values():
        if not pts or len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            x0, y0 = map(float, pts[i])
            x1, y1 = map(float, pts[i + 1])
            lx, ly = rasterize_line_segment(
                x0, y0, x1, y1, reso=reso, line_thickness=line_thickness
            )
            ox.extend(lx)
            oy.extend(ly)

    return ox, oy, reso, map_w, map_h


def rect_corner_polygons(rects: Dict[str, ObstacleRect]) -> List[List[Tuple[float, float]]]:
    return [r.corners() for r in rects.values()]


def rect_values_for_se(rects: Dict[str, ObstacleRect]) -> List[Tuple[float, float, float, float, float]]:
    """(cx, cy, w, h, angle_deg) for mod_grid_SE oriented-box handling."""
    return [(r.cx, r.cy, r.w, r.h, r.angle_deg) for r in rects.values()]
