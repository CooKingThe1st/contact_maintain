"""
SE(2) holonomic planner (mod_grid_SE):

Phase 1 — SE(2) A* on a precomputed 3D conservative occupancy volume.
    State: (grid_x, grid_y, yaw_idx)  — no m_prev (unlike 2D disk mod_grid).
    - 12-connected moves: 4 cardinal XY × {dθ ∈ {-1,0,+1} bins}
    - 3-cell edge gate on the product grid (see se2_grid_volume.se2_edge_free_3cell)
    - Offline volume: convex footprint + OBB SAT; disk-free columns free at all θ
    Edge cost: c_move + c_rot·|dθ|  (rotation costs extra; no c_risk / c_heading)
    Heuristic: disk BFS h_xy + disk-column-gated h_θ (admissible)

Phase 3 — constant body-twist DP (legacy; not yet updated for the 3D volume).

``astar_planning(..., stop_phase=1)`` is the primary entrypoint.
"""

from __future__ import annotations

import json
import heapq
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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

try:
    from shapely.geometry import Point as _ShapelyPoint
    from shapely.geometry import Polygon as _ShapelyPolygon
    from shapely.geometry import box as _ShapelyBox
    from shapely.strtree import STRtree as _ShapelySTRtree

    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


def _rect_polys_from_obstacle_rects(
    obstacle_rects: Optional[List[Tuple[float, ...]]],
) -> List:
    """Rectangle obstacles as Shapely polygons (axis-aligned or oriented)."""
    if not (_HAS_SHAPELY and obstacle_rects):
        return []
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import ObstacleRect, parse_rect_values

    out: List = []
    for t in obstacle_rects:
        if len(t) == 4:
            rx, ry, rw, rh = map(float, t)
            rect = ObstacleRect(cx=rx + 0.5 * rw, cy=ry + 0.5 * rh, w=rw, h=rh)
        elif len(t) == 5:
            cx, cy, rw, rh, ang = map(float, t)
            rect = ObstacleRect(cx=cx, cy=cy, w=rw, h=rh, angle_deg=ang)
        else:
            try:
                rect = parse_rect_values(t)
            except ValueError:
                continue
        corners = rect.corners()
        out.append(_ShapelyPolygon(corners))
    return out


def _min_dist_point_to_rect_polys(px: float, py: float, rect_polys: List) -> float:
    if not rect_polys:
        return float("inf")
    pt = _ShapelyPoint(float(px), float(py))
    d = float("inf")
    for rp in rect_polys:
        d = min(d, float(pt.distance(rp)))
    return d


def _poly_obstacles_from_vertices(
    obstacle_polygons: Optional[List[List[Tuple[float, float]]]],
) -> List:
    """Arbitrary polygon obstacles as Shapely polygons (world frame, meters)."""
    if not (_HAS_SHAPELY and obstacle_polygons):
        return []
    out: List = []
    for poly in obstacle_polygons:
        if len(poly) < 3:
            continue
        p = _ShapelyPolygon([(float(x), float(y)) for (x, y) in poly])
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            out.append(p)
    return out


def _expand_bounds(bounds: Tuple[float, float, float, float], pad: float) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bounds
    p = float(max(0.0, pad))
    return (x0 - p, y0 - p, x1 + p, y1 + p)


def _bounds_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 < bx0 or bx1 < ax0:
        return False
    if ay1 < by0 or by1 < ay0:
        return False
    return True


def _geom_vertices_xy(geom: Any) -> List[Tuple[float, float]]:
    """
    Exterior vertices for polygon-like geometry.
    Used for cheap vertex-inside quick rejection.
    """ 
    if not _HAS_SHAPELY or geom is None:
        return []
    if getattr(geom, "is_empty", True):
        return []

    gtype = getattr(geom, "geom_type", "")
    out: List[Tuple[float, float]] = []

    if gtype == "Polygon":
        coords = list(geom.exterior.coords)
        if len(coords) >= 2 and abs(coords[0][0] - coords[-1][0]) < 1e-9 and abs(coords[0][1] - coords[-1][1]) < 1e-9:
            coords = coords[:-1]
        out.extend((float(x), float(y)) for (x, y) in coords)
        return out

    if gtype == "MultiPolygon":
        for p in geom.geoms:
            coords = list(p.exterior.coords)
            if len(coords) >= 2 and abs(coords[0][0] - coords[-1][0]) < 1e-9 and abs(coords[0][1] - coords[-1][1]) < 1e-9:
                coords = coords[:-1]
            out.extend((float(x), float(y)) for (x, y) in coords)
        return out

    return out


def _build_spatial_index(geoms: List) -> Tuple[Optional[Any], List[Tuple[float, float, float, float]]]:
    if not (_HAS_SHAPELY and geoms):
        return None, []
    bounds = [tuple(map(float, g.bounds)) for g in geoms]
    try:
        return _ShapelySTRtree(geoms), bounds
    except Exception:
        # Fallback to pure AABB filtering if STRtree is unavailable/incompatible.
        return None, bounds


def _query_spatial_candidates(
    query_bounds: Tuple[float, float, float, float],
    geoms: List,
    tree: Optional[Any],
    geoms_bounds: List[Tuple[float, float, float, float]],
) -> List:
    if not (_HAS_SHAPELY and geoms):
        return []

    if tree is not None:
        try:
            hits = tree.query(_ShapelyBox(*query_bounds))
            # Shapely 2: query may return integer indices.
            if len(hits) > 0 and isinstance(hits[0], (int, np.integer)):
                return [geoms[int(i)] for i in hits]
            # Shapely 1.x: query returns geometries.
            return list(hits)
        except Exception:
            pass

    out: List = []
    for g, gb in zip(geoms, geoms_bounds):
        if _bounds_intersect(query_bounds, gb):
            out.append(g)
    return out

# Optional: legacy shape loaders (OBJ->2D vertices, standard non-convex shapes).
# Used when scenario.robot provides `obj_path`/`shape_name` instead of `footprint_vertices`.
_LEGACY_SRC_DIR: Optional[Path] = None
for parent in [_THIS_DIR, *_THIS_DIR.parents]:
    cand_src = parent / "src"
    if (cand_src / "legacy" / "object_utils.py").exists():
        _LEGACY_SRC_DIR = cand_src
        break
if _LEGACY_SRC_DIR is not None:
    if str(_LEGACY_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_LEGACY_SRC_DIR))
    if str(_LEGACY_SRC_DIR / "legacy") not in sys.path:
        sys.path.insert(0, str(_LEGACY_SRC_DIR / "legacy"))


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

# SE(2) tunables
# 10° bins (36 layers) — sufficient with four-corner edge checks.
_YAW_BINS = 36
_YAW_STEP_RAD = 2.0 * math.pi / float(_YAW_BINS)
_YAW_GOAL_TOL_BINS = 0  # accept if within +/- this many yaw bins

# Rotation penalty per yaw-bin step (when dθ ≠ 0 on an edge)
_ROTATE_COST_PER_BIN = 0.35

# SE(2) Phase-1 edge costs (cardinal grid)
_C_MOVE = 1.0
_C_ROT = _ROTATE_COST_PER_BIN

# Phase 3 SE(2) primitive / collision modes
SE_P3_PRIMITIVE_LINEAR_YAW = "linear_yaw_dp"
SE_P3_PRIMITIVE_BODY_TWIST = "body_twist"
SE_P3_COLLISION_VOLUME_BIN = "volume_bin"
SE_P3_COLLISION_SAT_DIRECT = "sat_direct"

# Shape collision / clearance
_SHAPE_MAX_BOUNDARY_SAMPLES = 48
_SHAPE_SAMPLE_STEP_FACTOR = 0.5  # step ~ factor * reso
_SHAPE_HARD_CLEARANCE_PAD_FACTOR = 0.25  # hard collision uses pad ~ factor * reso
_SHAPE_MIN_HARD_CLEARANCE_PAD = 0.06  # meters


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


def _yaw_to_bin(yaw_rad: float) -> int:
    y = float(yaw_rad) % (2.0 * math.pi)
    return int(round(y / _YAW_STEP_RAD)) % _YAW_BINS


def _bin_to_yaw(yaw_idx: int) -> float:
    yaw = float(yaw_idx) * _YAW_STEP_RAD
    # Convert to [-pi, pi) for nicer downstream usage.
    if yaw >= math.pi:
        yaw -= 2.0 * math.pi
    return yaw


def _polygon_signed_area(vertices: List[Tuple[float, float]]) -> float:
    area2 = 0.0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def _polygon_centroid(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    a = _polygon_signed_area(vertices)
    if abs(a) < 1e-12:
        xs = [p[0] for p in vertices]
        ys = [p[1] for p in vertices]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    cx = 0.0
    cy = 0.0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    cx /= (6.0 * a)
    cy /= (6.0 * a)
    return float(cx), float(cy)


def _clean_vertices(vertices: List[List[float]]) -> List[Tuple[float, float]]:
    pts = [(float(p[0]), float(p[1])) for p in vertices if len(p) >= 2]
    if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
        pts = pts[:-1]
    return pts


def _vertices_centered_at_centroid(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cx, cy = _polygon_centroid(vertices)
    return [(x - cx, y - cy) for (x, y) in vertices]


def _sample_polygon_boundary(
    vertices_local: List[Tuple[float, float]],
    sample_step: float,
    max_samples: int,
) -> np.ndarray:
    n = len(vertices_local)
    if n < 3:
        raise ValueError("Robot footprint polygon must have at least 3 vertices")
    step = max(1e-6, float(sample_step))

    samples: List[Tuple[float, float]] = []
    for i in range(n):
        x0, y0 = vertices_local[i]
        x1, y1 = vertices_local[(i + 1) % n]
        edge_len = math.hypot(x1 - x0, y1 - y0)
        # Ensure at least 2 points on each edge.
        num = max(2, int(math.ceil(edge_len / step)) + 1)
        for k in range(num):
            t = k / float(num)
            # Exclude the last point of each edge (avoid duplicates at vertices).
            if k == num - 1:
                continue
            samples.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

    if len(samples) < 3:
        samples = [(x, y) for (x, y) in vertices_local]

    if len(samples) > max_samples:
        stride = max(1, int(math.ceil(len(samples) / float(max_samples))))
        samples = samples[::stride]

    return np.array(samples, dtype=np.float64)


def _extract_robot_footprint_vertices_local(scenario_robot: dict, reso: float) -> List[Tuple[float, float]]:
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_planner_bridge import _resolve_robot_dict

    scenario_robot = _resolve_robot_dict(scenario_robot)
    # Direct 2D vertices
    for k in ("footprint_vertices", "vertices", "shape_vertices"):
        if k in scenario_robot and isinstance(scenario_robot[k], list) and scenario_robot[k]:
            pts = _clean_vertices(scenario_robot[k])
            if len(pts) < 3:
                raise ValueError(f"robot.{k} must contain >= 3 points")
            return _vertices_centered_at_centroid(pts)

    # Legacy standard shapes (headless 2D)
    if "shape_name" in scenario_robot:
        shape_name = str(scenario_robot["shape_name"])
        from object_utils import create_standard_objects

        objs = create_standard_objects()
        if shape_name in objs:
            poly = objs[shape_name].geometry
            pts_raw = list(poly.exterior.coords)
            pts = [(float(x), float(y)) for (x, y) in pts_raw]
            if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
                pts = pts[:-1]
            pts = _vertices_centered_at_centroid(pts)
            return pts

    # OBJ -> 2D footprint (prefer precomputed cache from preprocess_obj_footprints.py)
    obj_path = None
    for k in ("obj_path", "mesh_obj", "obj"):
        if k in scenario_robot and scenario_robot[k]:
            obj_path = str(scenario_robot[k])
            break
    shape_name = str(scenario_robot.get("shape_name", "")).strip() or None
    if obj_path or shape_name:
        vertices = None
        try:
            import rospkg

            pkg_src = Path(rospkg.RosPack().get_path("contact_maintain")) / "src"
            if str(pkg_src) not in sys.path:
                sys.path.insert(0, str(pkg_src))
            from contact_maintain.footprint_cache import resolve_footprint_vertices

            vertices = resolve_footprint_vertices(shape_name=shape_name, obj_path=obj_path)
        except Exception:
            if obj_path:
                from object_utils import read_obj_to_vertices

                vertices = read_obj_to_vertices(obj_path)
        if vertices:
            if len(vertices) < 3:
                raise ValueError("OBJ footprint has < 3 vertices")
            pts = _vertices_centered_at_centroid(vertices)
            return pts

    if "shape_name" in scenario_robot:
        shape_name = str(scenario_robot["shape_name"])
        raise ValueError(
            f"Unknown robot.shape_name='{shape_name}' and no obj_path provided."
        )

    # Fallback: rectangle from width/length
    width = float(scenario_robot.get("width", 2.0))
    length = float(scenario_robot.get("length", 3.0))
    L2 = length / 2.0
    W2 = width / 2.0
    rect = [(-L2, -W2), (L2, -W2), (L2, W2), (-L2, W2)]
    return _vertices_centered_at_centroid(rect)


def _yaw_delta_bins(y0: int, y1: int) -> int:
    d = (y1 - y0) % _YAW_BINS
    return min(d, _YAW_BINS - d)


def _robot_circumradius(robot_vertices_local: List[Tuple[float, float]]) -> float:
    return float(max(math.hypot(x, y) for (x, y) in robot_vertices_local))


def _yaw_fill_disk_path(n: int, syaw_rad: float, gyaw_rad: float) -> List[float]:
    """Linear yaw interpolation along a disk (x,y) polyline."""
    if n <= 0:
        return []
    if n == 1:
        return [float(gyaw_rad)]
    delta = math.atan2(math.sin(gyaw_rad - syaw_rad), math.cos(gyaw_rad - syaw_rad))
    out: List[float] = []
    for i in range(n):
        t = i / float(n - 1)
        yaw = syaw_rad + t * delta
        out.append(float(math.atan2(math.sin(yaw), math.cos(yaw))))
    return out


def _normalize_obstacle_rects(
    obstacle_rects: Optional[Sequence[Union[Tuple[float, ...], Any]]],
) -> List:
    if not obstacle_rects:
        return []
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import ObstacleRect, obstacle_rects_from_se_values

    first = obstacle_rects[0]
    if isinstance(first, ObstacleRect):
        return list(obstacle_rects)
    return obstacle_rects_from_se_values(obstacle_rects)  # type: ignore[arg-type]


def _motion_index_for_delta(dx: int, dy: int, motion: List[List[int]]) -> int:
    for mi, mv in enumerate(motion):
        if mv[0] == dx and mv[1] == dy:
            return mi
    return -1


def _se2_neighbors_12() -> List[Tuple[int, int, int]]:
    """4 cardinal XY moves × {dθ ∈ {-1, 0, +1} yaw bins} = 12 edges."""
    out: List[Tuple[int, int, int]] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for dt in (-1, 0, 1):
            out.append((dx, dy, dt))
    return out


_SE2_NEIGHBORS_12 = _se2_neighbors_12()


def _manhattan_xy(x: int, y: int, gx: int, gy: int) -> int:
    return abs(int(x) - int(gx)) + abs(int(y) - int(gy))


def _build_disk_cardinal_dist_to_goal(
    gx: int,
    gy: int,
    P: base_astar.Para,
    disk_obsmap: np.ndarray,
) -> np.ndarray:
    """Cardinal BFS distance (grid steps) from each cell to goal on the disk-free subgraph."""
    from collections import deque

    xw, yw = disk_obsmap.shape
    inf = np.iinfo(np.int32).max // 4
    dist = np.full((xw, yw), inf, dtype=np.int32)
    gix = int(gx) - int(P.minx)
    giy = int(gy) - int(P.miny)
    if gix < 0 or giy < 0 or gix >= xw or giy >= yw:
        return dist
    dist[gix, giy] = 0
    q: deque = deque([(gix, giy)])
    while q:
        ix, iy = q.popleft()
        base = int(dist[ix, iy])
        for dix, diy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nix, niy = ix + dix, iy + diy
            if nix < 0 or niy < 0 or nix >= xw or niy >= yw:
                continue
            if disk_obsmap[nix, niy]:
                continue
            nd = base + 1
            if nd < dist[nix, niy]:
                dist[nix, niy] = nd
                q.append((nix, niy))
    return dist


def _state_index_se2(x: int, y: int, yaw_idx: int, P: base_astar.Para) -> int:
    cell = (y - P.miny) * P.xw + (x - P.minx)
    return cell * _YAW_BINS + (int(yaw_idx) % _YAW_BINS)


class _AugNodeSE2:
    __slots__ = ("x", "y", "yaw_idx", "cost", "p_ind")

    def __init__(self, x: int, y: int, yaw_idx: int, cost: float, p_ind: int):
        self.x = x
        self.y = y
        self.yaw_idx = yaw_idx
        self.cost = cost
        self.p_ind = p_ind


def _ms(seconds: float) -> float:
    return 1000.0 * float(seconds)


def format_se2_pipeline_report(
    timing: Dict[str, float],
    *,
    prep: Optional[Dict[str, float]] = None,
    meta: Optional[Dict[str, object]] = None,
    phase3_s: Optional[float] = None,
    wall_s: Optional[float] = None,
) -> List[str]:
    """Human-readable pipeline lines for the app status log."""
    prep = prep or {}
    meta = meta or {}
    lines: List[str] = []

    phase = int(meta.get("phase", 1))
    lines.append(f"[timing] mod_grid_SE pipeline (stop phase {phase})")

    meta_bits = []
    if "shape" in meta:
        meta_bits.append(f"shape={meta['shape']}")
    if "reso" in meta:
        meta_bits.append(f"reso={meta['reso']}")
    if "map_w" in meta and "map_h" in meta:
        meta_bits.append(f"map={meta['map_w']}x{meta['map_h']}m")
    if "obstacle_pts" in meta:
        meta_bits.append(f"ox={meta['obstacle_pts']}")
    if "footprint_verts" in meta:
        meta_bits.append(f"footprint_verts={meta['footprint_verts']}")
    if "safety_margin" in meta:
        meta_bits.append(f"margin={meta['safety_margin']}")
    if meta_bits:
        lines.append("  setup: " + " ".join(meta_bits))
    if prep.get("footprint_s", 0.0) > 0.0:
        lines.append(f"  prep: footprint={_ms(prep['footprint_s']):.0f}ms")

    disk_ms = _ms(timing.get("disk_astar_s", 0.0))
    if timing.get("disk_fast_path", 0.0) >= 0.5:
        lines.append(
            f"  1 disk A*: {disk_ms:.0f}ms -> GOAL REACHED (disk holonomic); skip volume + SE A*"
        )
    else:
        reached = "NO" if timing.get("disk_goal_reached", 0.0) < 0.5 else "YES"
        closed = int(timing.get("disk_closed_cells", 0.0))
        lines.append(
            f"  1 disk A*: {disk_ms:.0f}ms -> goal reached={reached} closed={closed}"
        )
        vol_ms = _ms(timing.get("volume_s", 0.0))
        if vol_ms > 0.0:
            cells_sat = int(timing.get("vol_cells_sat", 0.0))
            cells_lazy = int(timing.get("vol_cells_lazy", 0.0))
            occ_true = int(timing.get("vol_occ_true", 0.0))
            disk_blk = int(timing.get("vol_disk_blocked", 0.0))
            parts = int(timing.get("vol_convex_parts", 0.0))
            classify_ms = _ms(timing.get("vol_classify_s", 0.0))
            lazy_sat_ms = _ms(timing.get("vol_lazy_sat_s", 0.0))
            lazy_queries = int(timing.get("vol_lazy_sat_queries", 0.0))
            lines.append(
                f"  2 volume: {vol_ms:.0f}ms "
                f"(disk_map={_ms(timing.get('vol_disk_map_s', 0.0)):.0f}ms "
                f"classify={classify_ms:.0f}ms "
                f"lazy_sat={lazy_sat_ms:.0f}ms "
                f"clearance={_ms(timing.get('vol_clearance_s', 0.0)):.0f}ms) "
                f"cells={cells_sat} lazy={cells_lazy} lazy_queries={lazy_queries} "
                f"disk_blk={disk_blk} occ_vox={occ_true} parts={parts}"
            )
        se_ms = _ms(timing.get("se_astar_s", 0.0))
        if se_ms > 0.0:
            se_exp = int(timing.get("se_expanded_states", timing.get("se_closed_states", 0.0)))
            se_rate = float(timing.get("se_states_per_s", 0.0))
            se_max_open = int(timing.get("se_max_open", 0.0))
            se_stale = int(timing.get("se_stale_pops", 0.0))
            path_pts = int(timing.get("path_pts", 0.0))
            rate_s = f" {se_rate:.0f} states/s" if se_rate > 0.0 else ""
            lines.append(
                f"  3 SE A*: {se_ms:.0f}ms expanded={se_exp}{rate_s} "
                f"open_max={se_max_open} stale_pops={se_stale} -> path {path_pts} pts"
            )

    if phase3_s is not None:
        lines.append(f"  4 phase3 DP: {_ms(phase3_s):.0f}ms")

    path_stats = meta.get("path_stats")
    if isinstance(path_stats, dict) and path_stats:
        poly_len = float(path_stats.get("polyline_length_m", 0.0))
        prim_len = float(path_stats.get("primitive_length_m", 0.0))
        n_prims = int(path_stats.get("n_primitives", 0))
        n_s = int(path_stats.get("n_straight", 0))
        n_c = int(path_stats.get("n_arc", 0))
        out_pts = int(path_stats.get("output_pts", 0))
        p1_pts = int(path_stats.get("p1_spine_pts", 0))
        fallback = bool(path_stats.get("p3_fallback", False))
        compressed = bool(path_stats.get("p3_compressed", False))
        fb_note = " fallback=phase1" if fallback else (" compressed" if compressed else "")
        lines.append(
            f"  path: {out_pts} pts len={poly_len:.2f}m | "
            f"prims={n_prims} (S={n_s} C={n_c}) prim_len={prim_len:.2f}m | "
            f"p1={p1_pts}{fb_note}"
        )
        direct_q = int(path_stats.get("direct_sat_queries", 0))
        if direct_q > 0:
            lines.append(f"  phase3 direct SAT queries: {direct_q}")

    plan_ms = _ms(timing.get("total_s", 0.0))
    if phase3_s is not None:
        plan_ms += _ms(phase3_s)
    wall_ms = _ms(wall_s) if wall_s is not None else plan_ms
    lines.append(f"  TOTAL: planner={plan_ms:.0f}ms wall={wall_ms:.0f}ms")
    return lines


def _merge_volume_lazy_timing(vol_timing: Dict[str, float], volume: Any) -> None:
    """Fold on-demand SAT stats accumulated during SE A* into volume timing."""
    if volume is None:
        return
    lazy_s = float(getattr(volume, "lazy_sat_s", 0.0))
    lazy_q = int(getattr(volume, "lazy_sat_queries", 0))
    if lazy_s > 0.0 or lazy_q > 0:
        vol_timing["lazy_sat_s"] = lazy_s
        vol_timing["lazy_sat_queries"] = float(lazy_q)
        vol_timing["sat_s"] = lazy_s


def _fill_phase1_timing(
    timing: Dict[str, float],
    *,
    t_all: float,
    t_disk0: float,
    t_disk1: float,
    t_vol0: float,
    t_vol1: float,
    t_se0: Optional[float],
    vol_timing: Dict[str, float],
    disk_goal_reached_flag: bool,
    reachable_xy: Set[Tuple[int, int]],
    se_closed: int,
    path_pts: int,
    disk_fast_path: bool,
    se_heap_pops: int = 0,
    se_stale_pops: int = 0,
    se_max_open: int = 0,
) -> None:
    se_s = (time.perf_counter() - t_se0) if t_se0 is not None else 0.0
    timing.clear()
    timing.update(
        {
            "disk_astar_s": t_disk1 - t_disk0,
            "volume_s": t_vol1 - t_vol0,
            "se_astar_s": se_s,
            "disk_fast_path": 1.0 if disk_fast_path else 0.0,
            "disk_goal_reached": 1.0 if disk_goal_reached_flag else 0.0,
            "disk_closed_cells": float(len(reachable_xy)),
            "se_expanded_states": float(se_closed),
            "se_closed_states": float(se_closed),
            "se_heap_pops": float(se_heap_pops),
            "se_stale_pops": float(se_stale_pops),
            "se_max_open": float(se_max_open),
            "se_states_per_s": float(se_closed) / se_s if se_s > 1e-9 and se_closed > 0 else 0.0,
            "path_pts": float(path_pts),
            "total_s": time.perf_counter() - t_all,
        }
    )
    if vol_timing:
        timing.update(
            {
                "vol_decompose_s": vol_timing.get("decompose_s", 0.0),
                "vol_disk_map_s": vol_timing.get("disk_map_s", 0.0),
                "vol_classify_s": vol_timing.get("classify_s", 0.0),
                "vol_sat_s": vol_timing.get("sat_s", 0.0),
                "vol_lazy_sat_s": vol_timing.get("lazy_sat_s", 0.0),
                "vol_lazy_sat_queries": vol_timing.get("lazy_sat_queries", 0.0),
                "vol_disk_reuse_s": vol_timing.get("disk_reuse_s", 0.0),
                "vol_clearance_s": vol_timing.get("clearance_s", 0.0),
                "vol_cells_sat": vol_timing.get("cells_sat", 0.0),
                "vol_cells_lazy": vol_timing.get("cells_lazy", 0.0),
                "vol_occ_true": vol_timing.get("occ_true", 0.0),
                "vol_disk_blocked": vol_timing.get("disk_blocked", 0.0),
                "vol_convex_parts": vol_timing.get("convex_parts", 0.0),
                "vol_grid_xy": vol_timing.get("grid_xy", 0.0),
            }
        )


def _p3_phase1_result(
    px: List[float],
    py: List[float],
    pyaw: List[float],
    volume: Optional[Any],
    return_volume: bool,
):
    if return_volume:
        return px, py, pyaw, volume
    return px, py, pyaw


def phase1_augmented_astar_se2(
    sx: float,
    sy: float,
    syaw_rad: float,
    gx: float,
    gy: float,
    gyaw_rad: float,
    ox: List[float],
    oy: List[float],
    reso: float,
    robot_vertices_local: List[Tuple[float, float]],
    safety_margin: float = 0.0,
    obstacle_rects: Optional[Sequence] = None,
    obstacle_polygons: Optional[List[List[Tuple[float, float]]]] = None,
    map_bounds: Optional[Tuple[float, float, float, float]] = None,
    volume: Optional[Any] = None,
    timing: Optional[Dict[str, float]] = None,
    return_volume: bool = False,
) -> Union[Tuple[List[float], List[float], List[float]], Tuple[List[float], List[float], List[float], Any]]:
    del obstacle_polygons  # v1: OBB rects + point cloud only

    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import clamp_safety_margin
    from se2_grid_volume import build_se2_grid_volume, se2_edge_free_3cell

    import HybridAstarPlanner.mod_grid as mod_grid_disk

    safety_margin = clamp_safety_margin(safety_margin)
    rects = _normalize_obstacle_rects(obstacle_rects)
    rr = _robot_circumradius(robot_vertices_local)

    sx_i, sy_i = round(sx / reso), round(sy / reso)
    gx_i, gy_i = round(gx / reso), round(gy / reso)
    syaw_idx = _yaw_to_bin(syaw_rad)
    gyaw_idx = _yaw_to_bin(gyaw_rad)

    t_all = time.perf_counter()
    t_disk0 = time.perf_counter()
    px_disk, py_disk, disk_goal_reached, reachable_xy, _reachable_cost = (
        mod_grid_disk.phase1_augmented_astar_with_meta(
            sx,
            sy,
            gx,
            gy,
            ox,
            oy,
            reso,
            rr,
            safety_margin=float(safety_margin),
            obstacle_rects=rects if rects else None,
        )
    )
    t_disk1 = time.perf_counter()
    if disk_goal_reached and len(px_disk) >= 2:
        if timing is not None:
            timing.clear()
            timing.update(
                {
                    "disk_astar_s": t_disk1 - t_disk0,
                    "volume_s": 0.0,
                    "se_astar_s": 0.0,
                    "disk_fast_path": 1.0,
                    "disk_goal_reached": 1.0,
                    "disk_closed_cells": float(len(reachable_xy)),
                    "path_pts": float(len(px_disk)),
                    "total_s": time.perf_counter() - t_all,
                }
            )
        return _p3_phase1_result(
            px_disk, py_disk, _yaw_fill_disk_path(len(px_disk), syaw_rad, gyaw_rad),
            None, return_volume,
        )

    t_vol0 = time.perf_counter()
    if volume is None:
        vol_timing: Dict[str, float] = {}
        volume = build_se2_grid_volume(
            ox=ox,
            oy=oy,
            reso=reso,
            robot_vertices_local=robot_vertices_local,
            safety_margin=float(safety_margin),
            rects=rects,
            map_bounds=map_bounds,
            n_yaw_bins=_YAW_BINS,
            reachable_disk_cells=reachable_xy,
            timing=vol_timing,
        )
    else:
        vol_timing = {}
    t_vol1 = time.perf_counter()

    P = volume.P
    disk_obsmap = volume.disk_obsmap
    disk_dist = _build_disk_cardinal_dist_to_goal(gx_i, gy_i, P, disk_obsmap)
    disk_dist_inf = int(np.iinfo(np.int32).max // 8)
    goal_disk_free = volume.disk_column_free(gx_i, gy_i)

    def in_bounds(x: int, y: int) -> bool:
        return P.minx < x < P.maxx and P.miny < y < P.maxy

    def pose_free(x: int, y: int, yaw_idx: int) -> bool:
        if not in_bounds(x, y):
            return False
        if volume.disk_column_free(x, y):
            return True
        return not volume.is_occupied(x, y, yaw_idx)

    def _disk_steps_to_goal(x_i: int, y_i: int) -> int:
        ix = int(x_i) - int(P.minx)
        iy = int(y_i) - int(P.miny)
        if ix < 0 or iy < 0 or ix >= P.xw or iy >= P.yw:
            return _manhattan_xy(x_i, y_i, gx_i, gy_i)
        d = int(disk_dist[ix, iy])
        if d >= disk_dist_inf:
            return _manhattan_xy(x_i, y_i, gx_i, gy_i)
        return d

    def _heuristic_se2(x_i: int, y_i: int, yaw_idx: int) -> float:
        m_steps = _disk_steps_to_goal(x_i, y_i)
        h_xy = _C_MOVE * float(m_steps)
        if goal_disk_free:
            return h_xy
        d_yaw = _yaw_delta_bins(yaw_idx, gyaw_idx)
        h_theta = _C_ROT * float(max(0, d_yaw - m_steps))
        w_yaw = 0.0 if volume.disk_column_free(x_i, y_i) else 1.0
        return h_xy + w_yaw * h_theta

    def goal_reached(x_i: int, y_i: int, yaw_idx: int) -> bool:
        if x_i != gx_i or y_i != gy_i:
            return False
        return _yaw_delta_bins(yaw_idx, gyaw_idx) <= _YAW_GOAL_TOL_BINS

    if not pose_free(sx_i, sy_i, syaw_idx):
        if timing is not None:
            _fill_phase1_timing(
                timing,
                t_all=t_all,
                t_disk0=t_disk0,
                t_disk1=t_disk1,
                t_vol0=t_vol0,
                t_vol1=t_vol1,
                t_se0=None,
                vol_timing=vol_timing,
                disk_goal_reached_flag=disk_goal_reached,
                reachable_xy=reachable_xy,
                se_closed=0,
                path_pts=0,
                disk_fast_path=False,
                se_heap_pops=0,
                se_stale_pops=0,
                se_max_open=0,
            )
        return _p3_phase1_result([], [], [], volume if volume is not None else None, return_volume)

    t_se0 = time.perf_counter()

    start_idx = _state_index_se2(sx_i, sy_i, syaw_idx, P)
    n_start = _AugNodeSE2(sx_i, sy_i, syaw_idx, 0.0, -1)

    open_entries: Dict[int, _AugNodeSE2] = {start_idx: n_start}
    closed: Dict[int, _AugNodeSE2] = {}
    pq: List[Tuple[float, int]] = []
    heapq.heappush(pq, (_heuristic_se2(sx_i, sy_i, syaw_idx), start_idx))

    goal_idx_found: Optional[int] = None
    se_heap_pops = 0
    se_stale_pops = 0
    se_max_open = 1

    while pq:
        se_heap_pops += 1
        _, idx = heapq.heappop(pq)
        if idx in closed:
            se_stale_pops += 1
            continue
        if idx not in open_entries:
            continue

        cur = open_entries.pop(idx)
        closed[idx] = cur
        se_max_open = max(se_max_open, len(open_entries) + 1)

        if goal_reached(cur.x, cur.y, cur.yaw_idx):
            goal_idx_found = idx
            break

        for dx, dy, dt in _SE2_NEIGHBORS_12:
            nx, ny = cur.x + dx, cur.y + dy
            nt = (cur.yaw_idx + dt) % _YAW_BINS
            if not in_bounds(nx, ny):
                continue
            if not se2_edge_free_3cell(volume, cur.x, cur.y, cur.yaw_idx, nx, ny, nt):
                continue

            c_move = _C_MOVE
            c_rot = _C_ROT * float(abs(int(dt))) if int(dt) != 0 else 0.0
            c = c_move + c_rot

            child_idx = _state_index_se2(nx, ny, nt, P)
            g_new = cur.cost + c

            if child_idx in closed:
                continue
            h = _heuristic_se2(nx, ny, nt)
            if child_idx in open_entries:
                if g_new < open_entries[child_idx].cost:
                    open_entries[child_idx] = _AugNodeSE2(nx, ny, nt, g_new, idx)
                    heapq.heappush(pq, (g_new + h, child_idx))
            else:
                open_entries[child_idx] = _AugNodeSE2(nx, ny, nt, g_new, idx)
                heapq.heappush(pq, (g_new + h, child_idx))

    if goal_idx_found is None:
        if timing is not None:
            _merge_volume_lazy_timing(vol_timing, volume)
            _fill_phase1_timing(
                timing,
                t_all=t_all,
                t_disk0=t_disk0,
                t_disk1=t_disk1,
                t_vol0=t_vol0,
                t_vol1=t_vol1,
                t_se0=t_se0,
                vol_timing=vol_timing,
                disk_goal_reached_flag=disk_goal_reached,
                reachable_xy=reachable_xy,
                se_closed=len(closed),
                path_pts=0,
                disk_fast_path=False,
                se_heap_pops=se_heap_pops,
                se_stale_pops=se_stale_pops,
                se_max_open=se_max_open,
            )
        return _p3_phase1_result([], [], [], volume if volume is not None else None, return_volume)

    path_grid: List[Tuple[int, int, int]] = []
    walk = goal_idx_found
    while walk != -1:
        node = closed[walk]
        path_grid.append((node.x, node.y, node.yaw_idx))
        walk = node.p_ind
    path_grid.reverse()

    from scenario_obstacles import grid_cell_center_world

    pathx = [grid_cell_center_world(xi, yi, reso)[0] for (xi, yi, _yaw_i) in path_grid]
    pathy = [grid_cell_center_world(xi, yi, reso)[1] for (xi, yi, _yaw_i) in path_grid]
    pathyaw = [_bin_to_yaw(yaw_i) for (_xi, _yi, yaw_i) in path_grid]
    if timing is not None:
        _merge_volume_lazy_timing(vol_timing, volume)
        _fill_phase1_timing(
            timing,
            t_all=t_all,
            t_disk0=t_disk0,
            t_disk1=t_disk1,
            t_vol0=t_vol0,
            t_vol1=t_vol1,
            t_se0=t_se0,
            vol_timing=vol_timing,
            disk_goal_reached_flag=disk_goal_reached,
            reachable_xy=reachable_xy,
            se_closed=len(closed),
            path_pts=len(pathx),
            disk_fast_path=False,
            se_heap_pops=se_heap_pops,
            se_stale_pops=se_stale_pops,
            se_max_open=se_max_open,
        )
    return _p3_phase1_result(pathx, pathy, pathyaw, volume, return_volume)


def _p3_interp_yaw_bin_samples(theta1: float, theta2: float) -> int:
    d = _angle_wrap(float(theta2) - float(theta1))
    return max(1, int(math.ceil(abs(d) / max(1e-9, _YAW_STEP_RAD))))


def _p3_interp_sample_count(length_m: float, theta1: float, theta2: float, reso: float) -> int:
    # Half-cell / half-yaw-bin spacing keeps straight chords from stepping over
    # thin occupied bins between endpoint samples.
    n_len = max(2, int(math.ceil(float(length_m) / max(0.5 * float(reso), 1e-9))) + 1)
    n_yaw = max(2, 2 * _p3_interp_yaw_bin_samples(theta1, theta2) + 1)
    return min(480, max(4, n_len, n_yaw))


def _p3_interp_edge_length(typ: str, params: dict) -> float:
    if typ == "S":
        return math.hypot(
            float(params["x1"]) - float(params["x0"]),
            float(params["y1"]) - float(params["y0"]),
        )
    return abs(float(params["sweep"])) * max(float(params["r"]), 1e-9)


def _p3_verify_s_interp(
    volume: Any,
    x0: float,
    y0: float,
    t0: float,
    x1: float,
    y1: float,
    t1: float,
    reso: float,
    collision_mode: str,
) -> bool:
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    dtheta = _angle_wrap(float(t1) - float(t0))
    length = math.hypot(dx, dy)
    n = _p3_interp_sample_count(length, t0, t1, reso)
    for k in range(n):
        u = float(k) / float(n - 1) if n > 1 else 0.0
        x = float(x0) + u * dx
        y = float(y0) + u * dy
        th = float(t0) + u * dtheta
        if volume.pose_world_blocked(x, y, th, collision_mode):
            return False
    return True


def _p3_verify_c_interp(
    volume: Any,
    x0: float,
    y0: float,
    t0: float,
    x1: float,
    y1: float,
    t1: float,
    ocx: float,
    ocy: float,
    r: float,
    a0: float,
    sweep: float,
    reso: float,
    collision_mode: str,
) -> bool:
    arc_len = abs(float(sweep)) * max(float(r), 1e-9)
    dtheta = _angle_wrap(float(t1) - float(t0))
    n = _p3_interp_sample_count(arc_len, t0, t1, reso)
    for k in range(n):
        u = float(k) / float(n - 1) if n > 1 else 0.0
        ang = float(a0) + u * float(sweep)
        x = float(ocx) + float(r) * math.cos(ang)
        y = float(ocy) + float(r) * math.sin(ang)
        th = float(t0) + u * dtheta
        if volume.pose_world_blocked(x, y, th, collision_mode):
            return False
    return True


def _p3_output_polyline_clear(
    px: List[float],
    py: List[float],
    pyaw: List[float],
    volume: Any,
    reso: float,
    collision_mode: str,
) -> bool:
    if len(px) < 2:
        return True
    step_xy = max(float(reso) / 20.0, 1e-4)
    step_yaw = max(_YAW_STEP_RAD / 10.0, 1e-6)
    for i in range(len(px) - 1):
        x0 = float(px[i])
        y0 = float(py[i])
        t0 = float(pyaw[i])
        x1 = float(px[i + 1])
        y1 = float(py[i + 1])
        t1 = float(pyaw[i + 1])
        length = math.hypot(x1 - x0, y1 - y0)
        dtheta = _angle_wrap(t1 - t0)
        n = max(2, int(math.ceil(length / step_xy)) + 1, int(math.ceil(abs(dtheta) / step_yaw)) + 1)
        for k in range(n):
            u = float(k) / float(n - 1) if n > 1 else 0.0
            x = x0 + u * (x1 - x0)
            y = y0 + u * (y1 - y0)
            th = t0 + u * dtheta
            if volume.pose_world_blocked(x, y, th, collision_mode):
                return False
    return True


def polyline_path_length_m(px: Sequence[float], py: Sequence[float]) -> float:
    """Centerline arc length of a sampled polyline [m]."""
    if len(px) < 2 or len(py) != len(px):
        return 0.0
    total = 0.0
    for i in range(len(px) - 1):
        total += math.hypot(float(px[i + 1]) - float(px[i]), float(py[i + 1]) - float(py[i]))
    return total


def primitive_list_length_m(prims: Sequence[Tuple[str, dict]]) -> float:
    total = 0.0
    for typ, params in prims:
        total += _p3_interp_edge_length(str(typ), params)
    return total


def _fill_phase3_stats(
    stats: Optional[Dict[str, Any]],
    *,
    p1_spine_pts: int,
    output_pts: int,
    prims: Sequence[Tuple[str, dict]],
    px_out: Sequence[float],
    py_out: Sequence[float],
    fallback: bool,
) -> None:
    if stats is None:
        return
    n_prims = len(prims)
    n_s = sum(1 for typ, _ in prims if typ == "S")
    n_c = sum(1 for typ, _ in prims if typ == "C")
    prim_len = primitive_list_length_m(prims) if prims else 0.0
    stats.clear()
    stats.update(
        {
            "p1_spine_pts": int(p1_spine_pts),
            "output_pts": int(output_pts),
            "n_primitives": int(n_prims if not fallback else max(0, p1_spine_pts - 1)),
            "n_straight": int(n_s),
            "n_arc": int(n_c),
            "primitive_length_m": float(prim_len),
            "polyline_length_m": float(polyline_path_length_m(px_out, py_out)),
            "p3_fallback": bool(fallback),
            "p3_compressed": bool(not fallback and n_prims < max(0, p1_spine_pts - 1)),
        }
    )
    if prims:
        stats["primitives"] = [{"type": str(t), "params": dict(p)} for t, p in prims]


def phase3_interp_yaw_dp(
    px: List[float],
    py: List[float],
    pyaw: List[float],
    volume: Any,
    reso: float,
    collision_mode: str = SE_P3_COLLISION_VOLUME_BIN,
    dp_objective: str = "length",
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Decoupled SE(2) phase-3 DP on the phase-1 spine.

    Primitives:
      - ``S``: straight chord with yaw linear in arc-length parameter.
      - ``C``: circular arc through (i, mid, j) with yaw linear in arc length.

    Collision along each primitive is verified by dense sampling through
    ``volume.pose_world_blocked`` (volume bin or direct SAT).
    """
    from HybridAstarPlanner.mod_grid import DP_OBJECTIVE_LENGTH, DP_OBJECTIVE_MIN_SEGMENTS

    if collision_mode not in (SE_P3_COLLISION_VOLUME_BIN, SE_P3_COLLISION_SAT_DIRECT):
        raise ValueError(
            f"collision_mode must be {SE_P3_COLLISION_VOLUME_BIN!r} or {SE_P3_COLLISION_SAT_DIRECT!r} "
            f"(got {collision_mode!r})"
        )
    if dp_objective not in (DP_OBJECTIVE_LENGTH, DP_OBJECTIVE_MIN_SEGMENTS):
        raise ValueError(f"dp_objective must be 'length' or 'min_segments' (got {dp_objective!r})")

    n = len(px)
    if n < 2:
        _fill_phase3_stats(stats, p1_spine_pts=n, output_pts=n, prims=(), px_out=px, py_out=py, fallback=False)
        return px, py, pyaw
    if len(py) != n or len(pyaw) != n:
        raise ValueError("phase3_interp_yaw_dp: px/py/pyaw length mismatch")

    def _edge_cost(typ: str, params: dict) -> float:
        if dp_objective == DP_OBJECTIVE_MIN_SEGMENTS:
            return 1.0
        return _p3_interp_edge_length(typ, params)

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
            xi, yi, ti = float(px[i]), float(py[i]), float(pyaw[i])
            xj, yj, tj = float(px[j]), float(py[j]), float(pyaw[j])

            s_params = {"x0": xi, "y0": yi, "x1": xj, "y1": yj, "t0": ti, "t1": tj}
            if _p3_verify_s_interp(volume, xi, yi, ti, xj, yj, tj, reso, collision_mode):
                c = best_cost[i] + _edge_cost("S", s_params)
                if c < best_cost[j]:
                    best_cost[j] = c
                    best_prev[j] = (i, "S", s_params)

            if j - i >= 2:
                for k in {i + 1, (i + j) // 2, j - 1}:
                    if not (i < k < j):
                        continue
                    circ = _circle_from_3pts(xi, yi, float(px[k]), float(py[k]), xj, yj)
                    if circ is None:
                        continue
                    ocx, ocy, r = circ
                    arc_par = _arc_params_through_mid(
                        ocx, ocy, xi, yi, xj, yj, float(px[k]), float(py[k])
                    )
                    if arc_par is None:
                        continue
                    a0, _a1, sweep = arc_par
                    c_params = {
                        "ocx": ocx,
                        "ocy": ocy,
                        "r": r,
                        "a0": a0,
                        "sweep": sweep,
                        "t0": ti,
                        "t1": tj,
                    }
                    if not _p3_verify_c_interp(
                        volume, xi, yi, ti, xj, yj, tj, ocx, ocy, r, a0, sweep, reso, collision_mode
                    ):
                        continue
                    c = best_cost[i] + _edge_cost("C", c_params)
                    if c < best_cost[j]:
                        best_cost[j] = c
                        best_prev[j] = (i, "C", c_params)

    if best_prev[-1] is None:
        _fill_phase3_stats(stats, p1_spine_pts=n, output_pts=n, prims=(), px_out=px, py_out=py, fallback=True)
        return px, py, pyaw

    prims: List[Tuple[str, dict]] = []
    cur = n - 1
    while cur != 0:
        prev = best_prev[cur]
        if prev is None:
            _fill_phase3_stats(stats, p1_spine_pts=n, output_pts=n, prims=(), px_out=px, py_out=py, fallback=True)
            return px, py, pyaw
        _i, typ, params = prev
        prims.append((typ, params))
        cur = _i
    prims.reverse()

    outx: List[float] = [float(px[0])]
    outy: List[float] = [float(py[0])]
    outyaw: List[float] = [float(pyaw[0])]

    def emit(x: float, y: float, th: float) -> None:
        if (
            outx
            and abs(outx[-1] - x) < 1e-9
            and abs(outy[-1] - y) < 1e-9
            and abs(_angle_wrap(outyaw[-1] - th)) < 1e-9
        ):
            return
        outx.append(float(x))
        outy.append(float(y))
        outyaw.append(float(th))

    for typ, p in prims:
        t0 = float(p["t0"])
        t1 = float(p["t1"])
        dtheta = _angle_wrap(t1 - t0)
        if typ == "S":
            x0 = float(p["x0"])
            y0 = float(p["y0"])
            x1 = float(p["x1"])
            y1 = float(p["y1"])
            length = math.hypot(x1 - x0, y1 - y0)
            n_s = _p3_interp_sample_count(length, t0, t1, reso)
            for kk in range(1, n_s + 1):
                u = float(kk) / float(n_s)
                emit(x0 + u * (x1 - x0), y0 + u * (y1 - y0), t0 + u * dtheta)
        else:
            ocx = float(p["ocx"])
            ocy = float(p["ocy"])
            r = float(p["r"])
            a0 = float(p["a0"])
            sweep = float(p["sweep"])
            arc_len = abs(sweep) * max(r, 1e-9)
            n_c = _p3_interp_sample_count(arc_len, t0, t1, reso)
            for kk in range(1, n_c + 1):
                u = float(kk) / float(n_c)
                ang = a0 + u * sweep
                emit(ocx + r * math.cos(ang), ocy + r * math.sin(ang), t0 + u * dtheta)

    if not _p3_output_polyline_clear(outx, outy, outyaw, volume, reso, collision_mode):
        _fill_phase3_stats(stats, p1_spine_pts=n, output_pts=n, prims=prims, px_out=px, py_out=py, fallback=True)
        return px, py, pyaw
    _fill_phase3_stats(stats, p1_spine_pts=n, output_pts=len(outx), prims=prims, px_out=outx, py_out=outy, fallback=False)
    return outx, outy, outyaw


def phase1_augmented_astar(
    sx: float,
    sy: float,
    gx: float,
    gy: float,
    ox: List[float],
    oy: List[float],
    reso: float,
    rr: float,
) -> Tuple[List[float], List[float]]:
    motion = _get_motion()
    # Phase 1 should not be more conservative than the baseline grid A*.
    # So we build the hard occupancy map using exactly `rr` (no extra margin).
    # Additional clearance preferences are handled via risk/heading costs.
    occ_rr = rr

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

    from scenario_obstacles import grid_cell_center_world

    pathx = [grid_cell_center_world(x, y, reso)[0] for x, y in path_grid]
    pathy = [grid_cell_center_world(x, y, reso)[1] for x, y in path_grid]
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

    safe_rr = rr + float(safety_margin) + max(_INFLATION_RESO_FACTOR * reso, 0.15)
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


def _robot_bounding_radius(robot_vertices_local: List[Tuple[float, float]]) -> float:
    if not robot_vertices_local:
        return 0.5
    return float(max(math.hypot(x, y) for (x, y) in robot_vertices_local))


def _phase3_body_twist_sample_count(
    dist_end: float, theta_end_rel: float, reso: float, rb: float, sample_mult: float = 1.0
) -> int:
    """Pose samples along one constant body-twist segment (scale up for validation passes)."""
    mult = max(1.0, float(sample_mult))
    trans_step = max(0.06, min(0.3 * float(reso), 0.15 * float(rb))) / mult
    n_len = int(math.ceil(dist_end / trans_step)) + 1
    n_yaw = int(math.ceil(abs(theta_end_rel) / max(1e-6, 0.35 * _YAW_STEP_RAD))) + 1
    return min(240, max(14, n_len, n_yaw))


def phase3_min_segments(
    px: List[float],
    py: List[float],
    pyaw: List[float],
    ox: List[float],
    oy: List[float],
    robot_vertices_local: List[Tuple[float, float]],
    reso: float,
    clearance: float,
    obstacle_rects: Optional[List[Tuple[float, float, float, float]]] = None,
    obstacle_polygons: Optional[List[List[Tuple[float, float]]]] = None,
    map_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    SE(2)-aware Phase 3 DP: connect SE nodes with constant body-twist primitives.

    For each candidate edge (i -> j), we construct the *relative* SE(2) end pose in
    the body frame of i and then generate a constant body-twist trajectory that
    matches that relative endpoint:
      - theta_rel(t) = omega * t
      - p_dot_world(t) = R(theta_rel(t)) @ v_body
      - boundary points: cp_world(t) = p(t) + R(theta_world(t)) @ cp_body

    Yaw wrap is chosen using the same (-pi, pi] convention as `_angle_wrap`, which
    matches phase1's `_bin_to_yaw` usage (default bin center representation).

    We score edges by primitive count (each DP transition counts as 1) and accept
    an edge only if *all sampled poses* maintain min clearance > `clearance`.
    """

    n = len(px)
    if n < 2:
        return px, py, pyaw
    if len(pyaw) != n:
        raise ValueError("phase3_min_segments: px/py/yaw length mismatch")

    # Precompute robot boundary samples in local frame (as in phase1).
    sample_step = max(0.05, _SHAPE_SAMPLE_STEP_FACTOR * float(reso))
    robot_boundary = _sample_polygon_boundary(
        robot_vertices_local, sample_step=sample_step, max_samples=_SHAPE_MAX_BOUNDARY_SAMPLES
    )  # (N,2)
    rb = _robot_bounding_radius(robot_vertices_local)

    # Obstacles structure for fast nearest-neighbor queries.
    if ox and oy and _HAS_KDTREE:
        tree = _cKDTree(np.column_stack([ox, oy]).astype(np.float64))
        obs_arr = None
    else:
        tree = None
        obs_arr = np.column_stack([ox, oy]).astype(np.float64) if (ox and oy) else np.zeros((0, 2), dtype=np.float64)

    rect_polys = _rect_polys_from_obstacle_rects(obstacle_rects)
    poly_geoms = _poly_obstacles_from_vertices(obstacle_polygons)
    obstacle_geoms = rect_polys + poly_geoms
    obstacle_tree, obstacle_bounds = _build_spatial_index(obstacle_geoms)
    obstacle_vertices_xy = {id(g): _geom_vertices_xy(g) for g in obstacle_geoms}
    map_inner = None
    if _HAS_SHAPELY and map_bounds is not None:
        x0, y0, x1, y1 = map(float, map_bounds)
        if x1 > x0 and y1 > y0:
            world_box = _ShapelyBox(x0, y0, x1, y1)
            map_inner = world_box.buffer(1e-6)

    eps = 1e-9

    def _solve_constant_body_twist_from_SE2_no_fixed_speed(
        x_end: float, y_end: float, theta_end: float
    ) -> Tuple[Tuple[float, float], float, float]:
        """
        Solve for (v_body, omega, T) under constant body twist, allowing v_body to become 0
        (so pure in-place rotation is representable).

        We intentionally fix |omega|=1 for determinism (only changes time-scaling, not geometry).
        """
        x_end = float(x_end)
        y_end = float(y_end)
        theta_end = float(theta_end)

        if abs(theta_end) < eps:
            # Straight line: theta_rel constant ~ 0.
            disp_dist = math.hypot(x_end, y_end)
            if disp_dist < eps:
                return (0.0, 0.0), 0.0, 0.0
            ux, uy = x_end / disp_dist, y_end / disp_dist
            # Choose v_body of unit magnitude so T equals displacement length.
            return (ux, uy), 0.0, disp_dist

        # Arc case (theta_end != 0): use same algebra as the inverse solver in test_motion_primitive.py,
        # but do not enforce a fixed v_speed magnitude.
        A = float(math.sin(theta_end))
        B = float(1.0 - math.cos(theta_end))
        D = A * A + B * B
        if D < eps:
            # Numerically degenerate; treat as straight.
            disp_dist = math.hypot(x_end, y_end)
            if disp_dist < eps:
                return (0.0, 0.0), 0.0, 0.0
            ux, uy = x_end / disp_dist, y_end / disp_dist
            return (ux, uy), 0.0, disp_dist

        z1 = (A * x_end + B * y_end) / D
        z2 = (-B * x_end + A * y_end) / D
        z_norm = math.hypot(z1, z2)

        omega = math.copysign(1.0, theta_end)
        T = abs(theta_end)  # since theta_end / omega = abs(theta_end)

        if z_norm < eps:
            # Pure rotation in-place: v_body must be zero to match x_end=y_end=0.
            return (0.0, 0.0), omega, T

        v_body = (omega * z1, omega * z2)
        return v_body, omega, T

    def _propagate_body_twist_rel(
        v_body: Tuple[float, float], omega: float, T: float, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          positions_rel: (n_steps,2) in start-body frame
          theta_rel: (n_steps,) relative heading in radians
        """
        if n_steps <= 1:
            n_steps = 2
        if T <= 0.0:
            times = np.array([0.0, 0.0], dtype=np.float64)
        else:
            times = np.linspace(0.0, float(T), int(n_steps), dtype=np.float64)
        th = float(omega) * times

        vx, vy = float(v_body[0]), float(v_body[1])
        if abs(omega) < eps:
            positions_rel = times[:, None] * np.array([vx, vy], dtype=np.float64)[None, :]
        else:
            s = np.sin(th)
            c = np.cos(th)
            # Same closed-form as in test_motion_primitive.py.
            x = (vx / omega) * s - (vy / omega) * (1.0 - c)
            y = (vx / omega) * (1.0 - c) + (vy / omega) * s
            positions_rel = np.stack([x, y], axis=1).astype(np.float64)

        return positions_rel, th.astype(np.float64)

    def _rot2d_matrix(theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def _footprint_clear_at_world_pose(pxk: float, pyk: float, yawk: float) -> bool:
        ck, sk = math.cos(yawk), math.sin(yawk)

        if _HAS_SHAPELY and (map_inner is not None or obstacle_geoms):
            world_vertices = [
                (ck * vx - sk * vy + pxk, sk * vx + ck * vy + pyk) for (vx, vy) in robot_vertices_local
            ]
            poly = _ShapelyPolygon(world_vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                return False
            if map_inner is not None:
                if map_inner.is_empty or (not map_inner.covers(poly)):
                    return False
            query_bounds = _expand_bounds(tuple(map(float, poly.bounds)), float(clearance))
            obstacle_candidates = _query_spatial_candidates(
                query_bounds, obstacle_geoms, obstacle_tree, obstacle_bounds
            )
            for vx, vy in world_vertices:
                p = _ShapelyPoint(float(vx), float(vy))
                for og in obstacle_candidates:
                    if og.covers(p):
                        return False
            for og in obstacle_candidates:
                for ovx, ovy in obstacle_vertices_xy.get(id(og), []):
                    if poly.covers(_ShapelyPoint(float(ovx), float(ovy))):
                        return False
            for og in obstacle_candidates:
                if poly.distance(og) <= float(clearance) + 1e-9:
                    return False

            if tree is not None and obs_arr is None:
                query_r = float(rb + clearance + max(0.05, 0.25 * float(reso)))
                idxs = tree.query_ball_point([pxk, pyk], r=query_r)
                if idxs:
                    near_pts = np.asarray(tree.data[idxs], dtype=np.float64)
                    for ptx, pty in near_pts:
                        if poly.covers(_ShapelyPoint(float(ptx), float(pty))):
                            return False
            elif obs_arr is not None and obs_arr.shape[0] > 0:
                query_r = float(rb + clearance + max(0.05, 0.25 * float(reso)))
                dx = obs_arr[:, 0] - pxk
                dy = obs_arr[:, 1] - pyk
                mask = (dx * dx + dy * dy) <= (query_r * query_r)
                if np.any(mask):
                    for ptx, pty in obs_arr[mask]:
                        if poly.covers(_ShapelyPoint(float(ptx), float(pty))):
                            return False

        bx = robot_boundary[:, 0]
        by = robot_boundary[:, 1]
        world_pts = np.stack([ck * bx - sk * by + pxk, sk * bx + ck * by + pyk], axis=1)

        if tree is not None:
            d, _ = tree.query(world_pts, k=1)
            d_arr = np.asarray(d, dtype=np.float64).reshape(-1)
        else:
            if obs_arr is None or obs_arr.shape[0] == 0:
                d_arr = np.full(world_pts.shape[0], np.inf, dtype=np.float64)
            else:
                diff = world_pts[:, None, :] - obs_arr[None, :, :]
                dist = np.hypot(diff[..., 0], diff[..., 1])
                d_arr = np.min(dist, axis=1)

        if rect_polys:
            for ii in range(world_pts.shape[0]):
                d_rect = _min_dist_point_to_rect_polys(
                    float(world_pts[ii, 0]), float(world_pts[ii, 1]), rect_polys
                )
                d_arr[ii] = min(d_arr[ii], d_rect)

        return float(np.min(d_arr)) > float(clearance)

    def _body_twist_edge_clear(i: int, j: int, sample_mult: float = 1.0) -> bool:
        xi, yi, yawi = float(px[i]), float(py[i]), float(pyaw[i])
        xj, yj, yawj = float(px[j]), float(py[j]), float(pyaw[j])

        dx_w = xj - xi
        dy_w = yj - yi
        c, s = math.cos(yawi), math.sin(yawi)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w
        theta_end_rel = _angle_wrap(yawj - yawi)

        v_body, omega, T = _solve_constant_body_twist_from_SE2_no_fixed_speed(dx_b, dy_b, theta_end_rel)
        dist_end = math.hypot(dx_w, dy_w)
        n_steps = _phase3_body_twist_sample_count(dist_end, theta_end_rel, reso, rb, sample_mult)

        positions_rel, theta_rel = _propagate_body_twist_rel(v_body, omega, T, n_steps=n_steps)
        R_i = _rot2d_matrix(yawi)
        centers_world = (R_i @ positions_rel.T).T + np.array([xi, yi], dtype=np.float64)
        yaws_world = yawi + theta_rel

        for k in range(centers_world.shape[0]):
            if not _footprint_clear_at_world_pose(
                float(centers_world[k, 0]), float(centers_world[k, 1]), float(yaws_world[k])
            ):
                return False
            if k + 1 < centers_world.shape[0]:
                mid_x = 0.5 * (float(centers_world[k, 0]) + float(centers_world[k + 1, 0]))
                mid_y = 0.5 * (float(centers_world[k, 1]) + float(centers_world[k + 1, 1]))
                mid_yaw = float(yaws_world[k]) + 0.5 * _angle_wrap(
                    float(yaws_world[k + 1]) - float(yaws_world[k])
                )
                if not _footprint_clear_at_world_pose(mid_x, mid_y, mid_yaw):
                    return False
        return True

    def _edge_feasible(i: int, j: int) -> bool:
        return _body_twist_edge_clear(i, j, sample_mult=1.0)

    # DP over SE nodes.
    INF = 10**9
    best_cost = [INF] * n
    best_prev: List[Optional[int]] = [None] * n
    best_cost[0] = 0

    max_span = min(30, n - 1)
    for j in range(1, n):
        i_min = max(0, j - max_span)
        for i in range(i_min, j):
            if best_cost[i] >= INF:
                continue
            if _edge_feasible(i, j):
                c = best_cost[i] + 1
                if c < best_cost[j]:
                    best_cost[j] = c
                    best_prev[j] = i

    if best_prev[-1] is None:
        # No compression found; fall back.
        return px, py, pyaw

    # Reconstruct edges backward.
    edges: List[Tuple[int, int]] = []
    cur = n - 1
    while cur != 0:
        prev = best_prev[cur]
        if prev is None:
            break
        edges.append((prev, cur))
        cur = prev
    edges.reverse()

    for i, j in edges:
        if not _body_twist_edge_clear(i, j, sample_mult=2.5):
            return px, py, pyaw

    # Emit sampled SE points along each chosen primitive.
    outx: List[float] = [float(px[0])]
    outy: List[float] = [float(py[0])]
    outyaw: List[float] = [float(pyaw[0])]

    def emit(x: float, y: float, yaw: float) -> None:
        if outx and abs(outx[-1] - x) < 1e-9 and abs(outy[-1] - y) < 1e-9:
            # yaw may differ for pure-rotation; allow overwrite if same position.
            outyaw[-1] = float(yaw)
            return
        outx.append(float(x))
        outy.append(float(y))
        outyaw.append(float(yaw))

    for i, j in edges:
        xi, yi, yawi = float(px[i]), float(py[i]), float(pyaw[i])
        xj, yj, yawj = float(px[j]), float(py[j]), float(pyaw[j])

        dx_w = xj - xi
        dy_w = yj - yi
        c, s = math.cos(yawi), math.sin(yawi)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w
        theta_end_rel = _angle_wrap(yawj - yawi)

        v_body, omega, T = _solve_constant_body_twist_from_SE2_no_fixed_speed(dx_b, dy_b, theta_end_rel)

        dist_end = math.hypot(dx_w, dy_w)
        n_steps = _phase3_body_twist_sample_count(dist_end, theta_end_rel, reso, rb, sample_mult=1.0)

        positions_rel, theta_rel = _propagate_body_twist_rel(v_body, omega, T, n_steps=n_steps)
        R_i = _rot2d_matrix(yawi)
        centers_world = (R_i @ positions_rel.T).T + np.array([xi, yi], dtype=np.float64)
        yaws_world = yawi + theta_rel

        for k in range(centers_world.shape[0]):
            emit(float(centers_world[k, 0]), float(centers_world[k, 1]), float(yaws_world[k]))

    return outx, outy, outyaw


def astar_planning(
    sx: float,
    sy: float,
    syaw_rad: float,
    gx: float,
    gy: float,
    gyaw_rad: float,
    ox: List[float],
    oy: List[float],
    reso: float,
    robot_vertices_local: List[Tuple[float, float]],
    stop_phase: int = 1,
    safety_margin: float = 0.0,
    obstacle_rects: Optional[List[Tuple[float, float, float, float]]] = None,
    obstacle_polygons: Optional[List[List[Tuple[float, float]]]] = None,
    map_bounds: Optional[Tuple[float, float, float, float]] = None,
    timing: Optional[Dict[str, float]] = None,
    se_p3_primitive: str = SE_P3_PRIMITIVE_LINEAR_YAW,
    se_p3_collision_mode: str = SE_P3_COLLISION_VOLUME_BIN,
    dp_objective: str = "length",
    path_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    SE(2) mod_grid_SE pipeline.

    Currently supported:
      - stop_phase == 1: SE(2) Phase 1 augmented A* on a 3D conservative occupancy volume.
      - stop_phase == 3: primitive compression (linear-yaw DP or legacy body-twist).
    """
    if stop_phase not in (1, 3):
        raise NotImplementedError("mod_grid_SE Phase 2 is not yet implemented. Use stop_phase=1 or stop_phase=3.")

    from scenario_obstacles import clamp_safety_margin

    safety_margin = clamp_safety_margin(safety_margin)

    if stop_phase == 3:
        px0, py0, pyaw0, volume = phase1_augmented_astar_se2(
            sx=sx,
            sy=sy,
            syaw_rad=syaw_rad,
            gx=gx,
            gy=gy,
            gyaw_rad=gyaw_rad,
            ox=ox,
            oy=oy,
            reso=reso,
            robot_vertices_local=robot_vertices_local,
            safety_margin=float(safety_margin),
            obstacle_rects=obstacle_rects,
            obstacle_polygons=obstacle_polygons,
            map_bounds=map_bounds,
            timing=timing,
            return_volume=True,
        )
        if len(px0) < 2:
            return px0, py0, pyaw0
        if se_p3_primitive == SE_P3_PRIMITIVE_BODY_TWIST:
            hard_pad = (
                max(_SHAPE_MIN_HARD_CLEARANCE_PAD, _SHAPE_HARD_CLEARANCE_PAD_FACTOR * float(reso))
                + float(safety_margin)
            )
            return phase3_min_segments(
                px0,
                py0,
                pyaw0,
                ox=ox,
                oy=oy,
                robot_vertices_local=robot_vertices_local,
                reso=reso,
                clearance=hard_pad,
                obstacle_rects=obstacle_rects,
                obstacle_polygons=obstacle_polygons,
                map_bounds=map_bounds,
            )
        if volume is None:
            return px0, py0, pyaw0
        out = phase3_interp_yaw_dp(
            px0,
            py0,
            pyaw0,
            volume,
            reso=reso,
            collision_mode=se_p3_collision_mode,
            dp_objective=dp_objective,
            stats=path_stats,
        )
        if path_stats is not None:
            path_stats["direct_sat_queries"] = int(getattr(volume, "direct_sat_queries", 0))
        return out

    return phase1_augmented_astar_se2(
        sx=sx,
        sy=sy,
        syaw_rad=syaw_rad,
        gx=gx,
        gy=gy,
        gyaw_rad=gyaw_rad,
        ox=ox,
        oy=oy,
        reso=reso,
        robot_vertices_local=robot_vertices_local,
        safety_margin=float(safety_margin),
        obstacle_rects=obstacle_rects,
        obstacle_polygons=obstacle_polygons,
        map_bounds=map_bounds,
        timing=timing,
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


def run_mod_grid_on_scenario(
    scenario_path: str,
    stop_phase: int = 1,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Headless mod_grid_SE runner for the same scenario JSON used by `HA_draw/app.py`.

    Prints a tiny diagnostic and returns (px, py, pyaw).
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
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_planner_bridge import _resolve_robot_dict

    robot = _resolve_robot_dict(robot)
    syaw_rad = math.radians(float(px0[2]) if len(px0) >= 3 else 0.0)
    gyaw_rad = math.radians(float(gx0[2]) if len(gx0) >= 3 else 0.0)
    from scenario_obstacles import clamp_safety_margin

    safety_margin = clamp_safety_margin(float(robot.get("safety_margin", 0.0)))
    robot_vertices_local = _extract_robot_footprint_vertices_local(
        robot,
        reso=float(scenario.get("map", {}).get("resolution", 1.0)),
    )

    ox, oy, reso, map_w, map_h = _obstacle_points_from_app_scenario(scenario)

    obs = scenario.get("obstacles", {})
    rects_raw = obs.get("rects", {}) or {}
    _ha_draw = Path(__file__).resolve().parents[1]
    if str(_ha_draw) not in sys.path:
        sys.path.insert(0, str(_ha_draw))
    from scenario_obstacles import parse_scenario_rects

    parsed_rects = parse_scenario_rects(rects_raw, map_w=map_w, map_h=map_h)
    obstacle_rects = list(parsed_rects.values()) if parsed_rects else []
    obstacle_polygons = None

    px, py, pyaw = astar_planning(
        sx=sx,
        sy=sy,
        syaw_rad=syaw_rad,
        gx=gx,
        gy=gy,
        gyaw_rad=gyaw_rad,
        ox=ox,
        oy=oy,
        reso=reso,
        robot_vertices_local=robot_vertices_local,
        stop_phase=stop_phase,
        safety_margin=safety_margin,
        obstacle_rects=obstacle_rects if obstacle_rects else None,
        obstacle_polygons=obstacle_polygons,
        map_bounds=(0.0, 0.0, float(map_w), float(map_h)),
    )
    if px:
        # Diagnostic: how close did the path get to the box boundary?
        dmin = min(
            min(px[i], map_w - px[i], py[i], map_h - py[i]) for i in range(len(px))
        )
        print(
            f"[mod_grid_SE] scenario={os.path.basename(scenario_path)} stop_phase={stop_phase} "
            f"path_pts={len(px)} min_dist_to_box={dmin:.3f}m"
        )
    else:
        print(
            f"[mod_grid_SE] scenario={os.path.basename(scenario_path)} stop_phase={stop_phase} "
            f"NO PATH"
        )
    return px, py, pyaw


def _main_cli() -> None:
    """
    Minimal CLI for headless testing:
      python mod_grid_SE.py scenario_basic_test.json --stop_phase 1
    """
    if len(sys.argv) < 2:
        print("Usage: python mod_grid_SE.py <scenario.json> [--stop_phase 1]")
        sys.exit(2)
    scenario_path = sys.argv[1]
    stop_phase = 1
    if "--stop_phase" in sys.argv:
        i = sys.argv.index("--stop_phase")
        try:
            stop_phase = int(sys.argv[i + 1])
        except Exception:
            raise SystemExit("Invalid --stop_phase value")
    run_mod_grid_on_scenario(scenario_path, stop_phase=stop_phase)


if __name__ == "__main__":
    _main_cli()
