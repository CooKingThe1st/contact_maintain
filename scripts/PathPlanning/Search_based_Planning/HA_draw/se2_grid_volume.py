"""
Offline SE(2) conservative occupancy volume for mod_grid_SE Phase 1.

Builds ``occ[ix, iy, yaw_idx]`` using convex footprint parts + polygon-vs-OBB SAT.
Disk-free columns from the mod_grid-style 2D map imply free at all rotations.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

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

from scenario_obstacles import (
    ObstacleRect,
    apply_rect_disk_obstacles_to_obsmap,
    clamp_safety_margin,
    grid_cell_center_world,
    min_distance_point_to_rect,
    _obb_influence_window,
)

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
    from shapely.ops import triangulate as _shapely_triangulate

    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

DEFAULT_YAW_BINS = 36
_SQRT2_OVER_2 = math.sqrt(2.0) * 0.5
_SAT_EPS = 1e-9

# ``occ[ix, iy, t]`` voxel labels (uint8).
OCC_FREE = np.uint8(0)
OCC_BLOCKED = np.uint8(1)
OCC_UNKNOWN = np.uint8(255)

# Per-column classification for lazy SAT.
COLUMN_FREE = np.uint8(0)
COLUMN_TRAPPED = np.uint8(1)
COLUMN_LAZY = np.uint8(2)

P3_COLLISION_VOLUME_BIN = "volume_bin"
P3_COLLISION_SAT_DIRECT = "sat_direct"


@dataclass(frozen=True)
class ConvexPart:
    """One convex polygon in the robot body frame (pivot at origin)."""

    vertices_local: np.ndarray  # (N, 2)


@dataclass(frozen=True)
class RobotFootprintModel:
    parts: Tuple[ConvexPart, ...]
    circumradius: float


@dataclass(frozen=True)
class PreparedObstacle:
    """Precomputed expanded OBB for fast distance / SAT / engulf tests."""

    rect: ObstacleRect
    corners: np.ndarray  # (4, 2)
    aabb: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    half_w: float
    half_h: float
    cos_a: float
    sin_a: float
    obb_axes: np.ndarray  # (2, 2) unit normals for SAT
    obb_proj_min: np.ndarray  # (2,)
    obb_proj_max: np.ndarray  # (2,)


@dataclass(frozen=True)
class VolumeBuildContext:
    model: RobotFootprintModel
    obstacles: Tuple[PreparedObstacle, ...]
    safety_margin: float
    map_bounds: Optional[Tuple[float, float, float, float]]
    r_cell: float
    rb: float


@dataclass
class SE2GridVolume:
    """3D conservative occupancy volume with optional lazy middle-band SAT."""

    occ: np.ndarray  # (xw, yw, n_yaw) uint8: OCC_FREE / OCC_BLOCKED / OCC_UNKNOWN
    P: base_astar.Para
    n_yaw: int
    yaw_step_rad: float
    circumradius: float
    disk_obsmap: np.ndarray  # (xw, yw) bool, True = blocked
    clearance_m: np.ndarray  # (xw, yw) EDT clearance in meters
    reso: float
    column_class: Optional[np.ndarray] = None  # (xw, yw) uint8 FREE / TRAPPED / LAZY
    _lazy_ctx: Optional[VolumeBuildContext] = None
    _lazy_active: Optional[Dict[Tuple[int, int], Tuple[int, ...]]] = None
    lazy_sat_queries: int = 0
    lazy_sat_s: float = 0.0
    direct_sat_queries: int = 0
    direct_sat_s: float = 0.0

    def yaw_bin_from_rad(self, theta_rad: float) -> int:
        y = float(theta_rad) % (2.0 * math.pi)
        return int(round(y / self.yaw_step_rad)) % self.n_yaw

    def yaw_bin_candidates_from_rad(self, theta_rad: float) -> Tuple[int, ...]:
        """Yaw bins bracketing a continuous yaw sample."""
        y = (float(theta_rad) % (2.0 * math.pi)) / self.yaw_step_rad
        lo = int(math.floor(y)) % self.n_yaw
        hi = int(math.ceil(y)) % self.n_yaw
        if lo == hi:
            return (lo,)
        return (lo, hi)

    def xy_cell_candidates_from_world(self, x: float, y: float) -> Tuple[Tuple[int, int], ...]:
        """
        Grid cells whose centers bracket a continuous world pose.

        This grid stores cell centers at ``(g + 0.5) * reso``.  A continuous
        phase-3 sample can lie between centers, so a conservative volume-bin
        query checks the floor/ceil center bins around the sample rather than a
        single rounded ``x / reso`` bin.
        """
        r = float(self.reso)

        def _axis_candidates(v: float) -> Tuple[int, ...]:
            q = float(v) / r - 0.5
            lo = int(math.floor(q))
            hi = int(math.ceil(q))
            if lo == hi:
                return (lo,)
            return (lo, hi)

        xs = _axis_candidates(x)
        ys = _axis_candidates(y)
        return tuple((gx, gy) for gx in xs for gy in ys)

    def pose_world_blocked(self, x: float, y: float, theta_rad: float, collision_mode: str) -> bool:
        """True when world pose ``(x, y, theta_rad)`` is in collision."""
        mode = str(collision_mode).lower().strip()
        if mode not in (P3_COLLISION_VOLUME_BIN, P3_COLLISION_SAT_DIRECT):
            raise ValueError(
                f"collision_mode must be {P3_COLLISION_VOLUME_BIN!r} or {P3_COLLISION_SAT_DIRECT!r} "
                f"(got {collision_mode!r})"
            )
        xy_cells = self.xy_cell_candidates_from_world(float(x), float(y))
        if mode == P3_COLLISION_VOLUME_BIN:
            if all(self.disk_column_free(gx, gy) for gx, gy in xy_cells):
                return False
            for gx, gy in xy_cells:
                if self.disk_column_free(gx, gy):
                    continue
                for t in self.yaw_bin_candidates_from_rad(theta_rad):
                    # ``is_occupied`` runs lazy SAT on UNKNOWN/LAZY bins only.
                    if self.is_occupied(gx, gy, t):
                        return True
            # Continuous phase-3 poses can sit between cell-center samples; once
            # the bracketing bins are free, exact SAT is the conservative cert.
            return self._pose_blocked_sat_direct_for_cells(float(x), float(y), float(theta_rad), xy_cells)

        return self._pose_blocked_sat_direct_for_cells(float(x), float(y), float(theta_rad), xy_cells)

    def _pose_blocked_sat_direct_for_cells(
        self,
        x: float,
        y: float,
        theta_rad: float,
        xy_cells: Sequence[Tuple[int, int]],
    ) -> bool:
        if all(self.disk_column_free(gx, gy) for gx, gy in xy_cells):
            return False

        blocked_without_sat_context = False
        cc = self.column_class
        for gx, gy in xy_cells:
            if self.disk_column_free(gx, gy):
                continue
            ix = gx - self.P.minx
            iy = gy - self.P.miny
            if ix < 0 or iy < 0 or ix >= self.P.xw or iy >= self.P.yw:
                return True
            if cc is None:
                blocked_without_sat_context = True
                continue
            label = int(cc[ix, iy])
            if label == int(COLUMN_FREE):
                continue
            if label == int(COLUMN_TRAPPED):
                return True
            blocked_without_sat_context = True

        # For continuous phase-3 poses the column-level active set can be too
        # narrow (the pose may sit between cell centers).  Once the disk
        # neighborhood is not wholly free, use exact SAT against every prepared
        # rect so SAT-direct cannot miss an inflated obstacle.
        ctx = self._lazy_ctx
        if ctx is not None and ctx.obstacles:
            return self._pose_blocked_sat_direct(
                float(x), float(y), float(theta_rad), tuple(range(len(ctx.obstacles)))
            )
        return blocked_without_sat_context

    def _pose_blocked_sat_direct(
        self, x: float, y: float, theta_rad: float, active: Sequence[int]
    ) -> bool:
        ctx = self._lazy_ctx
        if ctx is None:
            return True
        if not active:
            return False
        t0 = time.perf_counter()
        blocked = pose_collides_prepared(x, y, theta_rad, ctx, active)
        self.direct_sat_s += time.perf_counter() - t0
        self.direct_sat_queries += 1
        return blocked

    def is_occupied(self, gx: int, gy: int, yaw_idx: int) -> bool:
        ix = gx - self.P.minx
        iy = gy - self.P.miny
        if ix < 0 or iy < 0 or ix >= self.P.xw or iy >= self.P.yw:
            return True
        t = int(yaw_idx) % self.n_yaw
        cc = self.column_class
        if cc is not None:
            label = int(cc[ix, iy])
            if label == int(COLUMN_FREE):
                return False
            if label == int(COLUMN_TRAPPED):
                return True
            if self.occ[ix, iy, t] == OCC_UNKNOWN:
                self._lazy_fill_pose(ix, iy, t)
        return bool(self.occ[ix, iy, t] == OCC_BLOCKED)

    def _lazy_fill_pose(self, ix: int, iy: int, t: int) -> None:
        ctx = self._lazy_ctx
        active = self._lazy_active
        if ctx is None or active is None:
            self.occ[ix, iy, t] = OCC_BLOCKED
            return
        key = (int(ix), int(iy))
        obs_idx = active.get(key)
        if not obs_idx:
            self.occ[ix, iy, t] = OCC_FREE
            return
        gx = int(self.P.minx + ix)
        gy = int(self.P.miny + iy)
        cx, cy = grid_cell_center_world(gx, gy, self.reso)
        t0 = time.perf_counter()
        blocked = pose_collides_prepared(cx, cy, float(t) * self.yaw_step_rad, ctx, obs_idx)
        self.lazy_sat_s += time.perf_counter() - t0
        self.lazy_sat_queries += 1
        self.occ[ix, iy, t] = OCC_BLOCKED if blocked else OCC_FREE

    def disk_column_free(self, gx: int, gy: int) -> bool:
        ix = gx - self.P.minx
        iy = gy - self.P.miny
        if ix < 0 or iy < 0 or ix >= self.P.xw or iy >= self.P.yw:
            return False
        return not bool(self.disk_obsmap[ix, iy])


def cell_disk_radius(circumradius: float, safety_margin: float, reso: float) -> float:
    """Conservative disk covering any pose pivot in the cell (matches mod_grid spirit)."""
    return float(circumradius) + float(safety_margin) + float(reso) * _SQRT2_OVER_2


def _is_convex(vertices: Sequence[Tuple[float, float]]) -> bool:
    pts = [(float(x), float(y)) for x, y in vertices]
    n = len(pts)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        x2, y2 = pts[(i + 2) % n]
        cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def decompose_footprint_to_convex_parts(
    vertices_local: Sequence[Tuple[float, float]],
) -> RobotFootprintModel:
    pts = [(float(x), float(y)) for x, y in vertices_local]
    if len(pts) < 3:
        raise ValueError("Footprint must have >= 3 vertices")

    rb = float(max(math.hypot(x, y) for x, y in pts))
    parts: List[ConvexPart] = []

    if _is_convex(pts):
        parts.append(ConvexPart(vertices_local=np.asarray(pts, dtype=np.float64)))
    elif _HAS_SHAPELY:
        poly = _ShapelyPolygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        for tri in _shapely_triangulate(poly):
            if tri.is_empty or tri.area < 1e-12:
                continue
            if not poly.intersects(tri):
                continue
            coords = list(tri.exterior.coords)
            if len(coords) >= 2 and abs(coords[0][0] - coords[-1][0]) < 1e-9:
                coords = coords[:-1]
            if len(coords) >= 3:
                parts.append(ConvexPart(vertices_local=np.asarray(coords, dtype=np.float64)))
    else:
        arr = np.asarray(pts, dtype=np.float64)
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(arr)
            parts.append(ConvexPart(vertices_local=arr[hull.vertices]))
        except Exception:
            parts.append(ConvexPart(vertices_local=arr))

    if not parts:
        parts.append(ConvexPart(vertices_local=np.asarray(pts, dtype=np.float64)))
    return RobotFootprintModel(parts=tuple(parts), circumradius=rb)


def prepare_obstacles(rects: Sequence[ObstacleRect], safety_margin: float) -> Tuple[PreparedObstacle, ...]:
    out: List[PreparedObstacle] = []
    margin = float(safety_margin)
    for rect in rects:
        expanded = ObstacleRect(
            cx=rect.cx,
            cy=rect.cy,
            w=float(rect.w) + 2.0 * margin,
            h=float(rect.h) + 2.0 * margin,
            angle_deg=rect.angle_deg,
        )
        corners = np.asarray(expanded.corners(), dtype=np.float64)
        aabb = (
            float(np.min(corners[:, 0])),
            float(np.min(corners[:, 1])),
            float(np.max(corners[:, 0])),
            float(np.max(corners[:, 1])),
        )
        c = math.cos(expanded.angle_rad)
        s = math.sin(expanded.angle_rad)
        half_w = 0.5 * float(expanded.w)
        half_h = 0.5 * float(expanded.h)
        obb_axes = np.array([[-s, c], [-c, -s]], dtype=np.float64)
        obb_proj = corners @ obb_axes.T
        out.append(
            PreparedObstacle(
                rect=expanded,
                corners=corners,
                aabb=aabb,
                half_w=half_w,
                half_h=half_h,
                cos_a=c,
                sin_a=s,
                obb_axes=obb_axes,
                obb_proj_min=obb_proj.min(axis=0),
                obb_proj_max=obb_proj.max(axis=0),
            )
        )
    return tuple(out)


def _aabb_disjoint(
    poly: np.ndarray,
    aabb: Tuple[float, float, float, float],
) -> bool:
    xmin, ymin, xmax, ymax = aabb
    px_min = float(poly[:, 0].min())
    px_max = float(poly[:, 0].max())
    py_min = float(poly[:, 1].min())
    py_max = float(poly[:, 1].max())
    return px_max < xmin - _SAT_EPS or px_min > xmax + _SAT_EPS or py_max < ymin - _SAT_EPS or py_min > ymax + _SAT_EPS


def _intervals_separated(min_a: float, max_a: float, min_b: float, max_b: float) -> bool:
    return max_a < min_b - _SAT_EPS or max_b < min_a - _SAT_EPS


def sat_convex_vs_prepared_obb(poly_world: np.ndarray, obb: PreparedObstacle) -> bool:
    """True when convex ``poly_world`` overlaps the prepared OBB."""
    if _aabb_disjoint(poly_world, obb.aabb):
        return False

    for k in range(2):
        ax = obb.obb_axes[k]
        proj = poly_world @ ax
        if _intervals_separated(
            float(proj.min()),
            float(proj.max()),
            float(obb.obb_proj_min[k]),
            float(obb.obb_proj_max[k]),
        ):
            return False

    n = poly_world.shape[0]
    for i in range(n):
        p0 = poly_world[i]
        p1 = poly_world[(i + 1) % n]
        edge = p1 - p0
        ax = np.array([-edge[1], edge[0]], dtype=np.float64)
        norm = float(np.hypot(ax[0], ax[1]))
        if norm < 1e-12:
            continue
        ax /= norm
        proj_p = poly_world @ ax
        proj_o = obb.corners @ ax
        if _intervals_separated(float(proj_p.min()), float(proj_p.max()), float(proj_o.min()), float(proj_o.max())):
            return False
    return True


def _world_to_obb_local(px: float, py: float, obb: PreparedObstacle) -> Tuple[float, float]:
    dx = px - float(obb.rect.cx)
    dy = py - float(obb.rect.cy)
    return obb.cos_a * dx + obb.sin_a * dy, -obb.sin_a * dx + obb.cos_a * dy


def disk_engulfed_by_obb(px: float, py: float, r_disk: float, obb: PreparedObstacle) -> bool:
    """Conservative: entire cell disk lies inside the filled OBB."""
    lx, ly = _world_to_obb_local(px, py, obb)
    if abs(lx) > obb.half_w + _SAT_EPS or abs(ly) > obb.half_h + _SAT_EPS:
        return False
    return (abs(lx) + r_disk <= obb.half_w + _SAT_EPS) and (abs(ly) + r_disk <= obb.half_h + _SAT_EPS)


def disk_clear_of_obb(px: float, py: float, r_disk: float, obb: PreparedObstacle) -> bool:
    """Disk fully outside the OBB (same spirit as mod_grid disk clearance)."""
    return min_distance_point_to_rect(px, py, obb.rect) > r_disk + _SAT_EPS


def _footprint_exceeds_map(
    parts_world: Sequence[np.ndarray],
    map_bounds: Tuple[float, float, float, float],
) -> bool:
    x0, y0, x1, y1 = map_bounds
    for poly in parts_world:
        if float(poly[:, 0].max()) > x1 + _SAT_EPS or float(poly[:, 0].min()) < x0 - _SAT_EPS:
            return True
        if float(poly[:, 1].max()) > y1 + _SAT_EPS or float(poly[:, 1].min()) < y0 - _SAT_EPS:
            return True
    return False


def _transform_parts(
    model: RobotFootprintModel,
    cx: float,
    cy: float,
    theta_rad: float,
) -> Tuple[np.ndarray, ...]:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    offset = np.array([cx, cy], dtype=np.float64)
    return tuple((part.vertices_local @ rot.T) + offset for part in model.parts)


def pose_collides_prepared(
    cx: float,
    cy: float,
    theta_rad: float,
    ctx: VolumeBuildContext,
    obstacle_indices: Sequence[int],
) -> bool:
    """Footprint vs selected OBBs at one pose; early exit on first hit."""
    parts_world = _transform_parts(ctx.model, cx, cy, theta_rad)

    if ctx.map_bounds is not None and _footprint_exceeds_map(parts_world, ctx.map_bounds):
        return True

    obstacles = ctx.obstacles
    for oi in obstacle_indices:
        obb = obstacles[oi]
        for poly_w in parts_world:
            if sat_convex_vs_prepared_obb(poly_w, obb):
                return True
    return False


def _build_clearance_meters(obsmap: np.ndarray, reso: float) -> np.ndarray:
    occ = np.asarray(obsmap, dtype=bool)
    if _HAS_SCIPY:
        return _distance_transform_edt(~occ).astype(np.float64) * float(reso)
    from collections import deque

    xw, yw = occ.shape
    dist = np.full((xw, yw), np.inf, dtype=np.float64)
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


def apply_disk_reachable_columns(
    occ: np.ndarray,
    disk_obsmap: np.ndarray,
    P,
    reachable_disk_cells: Set[Tuple[int, int]],
    column_class: Optional[np.ndarray] = None,
    lazy_active: Optional[Dict[Tuple[int, int], Tuple[int, ...]]] = None,
) -> None:
    """Disk closed-set cells with free disk columns → ``occ[ix, iy, :]`` free."""
    for gx, gy in reachable_disk_cells:
        ix = int(gx) - P.minx
        iy = int(gy) - P.miny
        if ix < 0 or iy < 0 or ix >= P.xw or iy >= P.yw:
            continue
        if not disk_obsmap[ix, iy]:
            occ[ix, iy, :] = OCC_FREE
            if column_class is not None:
                column_class[ix, iy] = COLUMN_FREE
            if lazy_active is not None:
                lazy_active.pop((int(ix), int(iy)), None)


def _cells_in_obb_windows(
    disk_obsmap: np.ndarray,
    P,
    obstacles: Sequence[PreparedObstacle],
    occ_rr: float,
    reso: float,
) -> Set[Tuple[int, int]]:
    """Disk-blocked cells inside any OBB influence window only."""
    cells: Set[Tuple[int, int]] = set()
    for obb in obstacles:
        ix0, ix1, iy0, iy1 = _obb_influence_window(obb.rect, occ_rr, reso, P)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if disk_obsmap[ix, iy]:
                    cells.add((ix, iy))
    return cells


def _classify_column(
    ix: int,
    iy: int,
    *,
    cx: float,
    cy: float,
    ctx: VolumeBuildContext,
    column_class: np.ndarray,
    occ: np.ndarray,
    lazy_active: Dict[Tuple[int, int], Tuple[int, ...]],
) -> str:
    """
    Label one disk-blocked cell: TRAPPED, FREE (clear of all rects), or LAZY.

    Returns ``"trapped"``, ``"free"``, or ``"lazy"``.
    """
    active: List[int] = []
    for oi, obb in enumerate(ctx.obstacles):
        if disk_clear_of_obb(cx, cy, ctx.r_cell, obb):
            continue
        if disk_engulfed_by_obb(cx, cy, ctx.r_cell, obb):
            column_class[ix, iy] = COLUMN_TRAPPED
            occ[ix, iy, :] = OCC_BLOCKED
            return "trapped"
        active.append(oi)

    if not active:
        column_class[ix, iy] = COLUMN_FREE
        occ[ix, iy, :] = OCC_FREE
        return "free"

    column_class[ix, iy] = COLUMN_LAZY
    occ[ix, iy, :] = OCC_UNKNOWN
    lazy_active[(int(ix), int(iy))] = tuple(active)
    return "lazy"


def _fill_column_from_obstacles(
    ix: int,
    iy: int,
    occ: np.ndarray,
    *,
    cx: float,
    cy: float,
    ctx: VolumeBuildContext,
    yaw_step: float,
    n_yaw: int,
) -> None:
    active: List[int] = []
    for oi, obb in enumerate(ctx.obstacles):
        if disk_clear_of_obb(cx, cy, ctx.r_cell, obb):
            continue
        if disk_engulfed_by_obb(cx, cy, ctx.r_cell, obb):
            occ[ix, iy, :] = True
            return
        active.append(oi)

    if not active:
        return

    for it in range(n_yaw):
        if occ[ix, iy, it]:
            continue
        if pose_collides_prepared(cx, cy, float(it) * yaw_step, ctx, active):
            occ[ix, iy, it] = True


def build_se2_grid_volume(
    *,
    ox: List[float],
    oy: List[float],
    reso: float,
    robot_vertices_local: Sequence[Tuple[float, float]],
    safety_margin: float = 0.0,
    rects: Sequence[ObstacleRect] = (),
    map_bounds: Optional[Tuple[float, float, float, float]] = None,
    n_yaw_bins: int = DEFAULT_YAW_BINS,
    reachable_disk_cells: Optional[Set[Tuple[int, int]]] = None,
    timing: Optional[Dict[str, float]] = None,
) -> SE2GridVolume:
    t_all = time.perf_counter()
    safety_margin = clamp_safety_margin(safety_margin)
    reso = float(reso)
    n_yaw = int(n_yaw_bins)
    if n_yaw < 1:
        raise ValueError("n_yaw_bins must be >= 1")

    t0 = time.perf_counter()
    model = decompose_footprint_to_convex_parts(robot_vertices_local)
    t_decompose = time.perf_counter() - t0
    occ_rr = model.circumradius + float(safety_margin)
    r_cell = cell_disk_radius(model.circumradius, safety_margin, reso)

    t0 = time.perf_counter()
    ox_g = [float(x) / reso for x in ox]
    oy_g = [float(y) / reso for y in oy]
    P, disk_point_list = base_astar.calc_parameters(ox_g, oy_g, occ_rr, reso)

    disk_obsmap_list = [row[:] for row in disk_point_list]
    rects_list = list(rects)
    if rects_list:
        apply_rect_disk_obstacles_to_obsmap(disk_obsmap_list, P, rects_list, occ_rr, reso)
    t_disk_map = time.perf_counter() - t0

    disk_obsmap = np.asarray(disk_obsmap_list, dtype=bool)
    xw, yw = disk_obsmap.shape
    occ = np.zeros((xw, yw, n_yaw), dtype=np.uint8)
    column_class = np.zeros((xw, yw), dtype=np.uint8)
    yaw_step = 2.0 * math.pi / float(n_yaw)

    obstacles = prepare_obstacles(rects_list, safety_margin)
    ctx = VolumeBuildContext(
        model=model,
        obstacles=obstacles,
        safety_margin=float(safety_margin),
        map_bounds=map_bounds,
        r_cell=r_cell,
        rb=model.circumradius,
    )
    lazy_active: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    cells_to_refine = 0
    cells_lazy = 0
    cells_trapped = 0
    cells_free_class = 0
    t0 = time.perf_counter()
    if obstacles:
        cells = _cells_in_obb_windows(disk_obsmap, P, obstacles, occ_rr, reso)
        cells_to_refine = len(cells)
        for ix, iy in cells:
            gx = int(P.minx + ix)
            gy = int(P.miny + iy)
            cx, cy = grid_cell_center_world(gx, gy, reso)
            label = _classify_column(
                ix,
                iy,
                cx=cx,
                cy=cy,
                ctx=ctx,
                column_class=column_class,
                occ=occ,
                lazy_active=lazy_active,
            )
            if label == "lazy":
                cells_lazy += 1
            elif label == "trapped":
                cells_trapped += 1
            else:
                cells_free_class += 1
    t_classify = time.perf_counter() - t0

    t0 = time.perf_counter()
    if reachable_disk_cells:
        apply_disk_reachable_columns(
            occ,
            disk_obsmap,
            P,
            reachable_disk_cells,
            column_class=column_class,
            lazy_active=lazy_active,
        )
    t_disk_reuse = time.perf_counter() - t0

    t0 = time.perf_counter()
    clearance_m = _build_clearance_meters(disk_obsmap, reso)
    t_clearance = time.perf_counter() - t0

    occ_true = int(np.sum(occ == OCC_BLOCKED))

    if timing is not None:
        timing.clear()
        timing.update(
            {
                "decompose_s": t_decompose,
                "disk_map_s": t_disk_map,
                "classify_s": t_classify,
                "sat_s": 0.0,
                "disk_reuse_s": t_disk_reuse,
                "clearance_s": t_clearance,
                "total_s": time.perf_counter() - t_all,
                "cells_sat": float(cells_to_refine),
                "cells_lazy": float(cells_lazy),
                "cells_trapped": float(cells_trapped),
                "cells_free_class": float(cells_free_class),
                "occ_true": float(occ_true),
                "disk_blocked": float(int(disk_obsmap.sum())),
                "convex_parts": float(len(model.parts)),
                "grid_xy": float(xw * yw),
            }
        )
    return SE2GridVolume(
        occ=occ,
        P=P,
        n_yaw=n_yaw,
        yaw_step_rad=yaw_step,
        circumradius=model.circumradius,
        disk_obsmap=disk_obsmap,
        clearance_m=clearance_m,
        reso=reso,
        column_class=column_class,
        _lazy_ctx=ctx if lazy_active else None,
        _lazy_active=lazy_active if lazy_active else None,
    )


def se2_edge_free_3cell(
    volume: SE2GridVolume,
    x1: int,
    y1: int,
    t1: int,
    x2: int,
    y2: int,
    t2: int,
) -> bool:
    """
    Conservative cardinal edge gate (3 corners of the product-cell sweep).

    Checks ``(x1,y1,t2)``, ``(x2,y2,t2)``, ``(x2,y2,t1)``.  Start ``(x1,y1,t1)``
    is already known free when expanding from the current state.
    """
    P = volume.P
    n_yaw = volume.n_yaw
    t1w = int(t1) % n_yaw
    t2w = int(t2) % n_yaw

    def _free(gx: int, gy: int, t: int) -> bool:
        if gx <= P.minx or gx >= P.maxx or gy <= P.miny or gy >= P.maxy:
            return False
        if volume.disk_column_free(gx, gy):
            return True
        return not volume.is_occupied(gx, gy, t)

    for gx, gy, t in ((x1, y1, t2w), (x2, y2, t2w), (x2, y2, t1w)):
        if not _free(gx, gy, t):
            return False
    return True


def se2_edge_free_4corner(
    volume: SE2GridVolume,
    x1: int,
    y1: int,
    t1: int,
    x2: int,
    y2: int,
    t2: int,
) -> bool:
    """Legacy alias: 4-corner gate (includes redundant start corner)."""
    P = volume.P
    n_yaw = volume.n_yaw
    t1w = int(t1) % n_yaw
    t2w = int(t2) % n_yaw

    def _free(gx: int, gy: int, t: int) -> bool:
        if gx <= P.minx or gx >= P.maxx or gy <= P.miny or gy >= P.maxy:
            return False
        if volume.disk_column_free(gx, gy):
            return True
        return not volume.is_occupied(gx, gy, t)

    for gx, gy, t in ((x1, y1, t1w), (x2, y2, t2w), (x2, y2, t1w), (x1, y1, t2w)):
        if not _free(gx, gy, t):
            return False
    return True
