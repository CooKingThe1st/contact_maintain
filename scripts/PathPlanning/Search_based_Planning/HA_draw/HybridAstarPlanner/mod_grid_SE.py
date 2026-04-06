"""
SE(2) holonomic planner (mod_grid_SE):

This file is currently focused on **Phase 1**: an augmented grid A* with yaw in the state.

Phase 1 — SE(2) augmented A*
    State: (grid_x, grid_y, yaw_idx, m_prev)
    - translation edges: 8-connected moves on the x-y grid; yaw is unchanged
    - rotation edges: in-place yaw changes; x-y is unchanged
    Edge cost: c_move + c_risk(d_min) + c_heading(m_prev, m_new) + c_yaw(yaw_delta)
      - c_risk: piecewise penalty from clearance between the robot footprint boundary samples and obstacle points
      - c_heading: penalty for turning away from the previous translation direction (straight legs)

Important:
  - Phase 2/3 are still the original mod_grid implementations and are not yet SE(2)-consistent.
  - `astar_planning(..., stop_phase=1)` is the supported entrypoint for now.
"""

from __future__ import annotations

import json
import heapq
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    obstacle_rects: Optional[List[Tuple[float, float, float, float]]],
) -> List:
    """Axis-aligned rectangle obstacles as Shapely polygons (world frame, meters)."""
    if not (_HAS_SHAPELY and obstacle_rects):
        return []
    out: List = []
    for t in obstacle_rects:
        if len(t) != 4:
            continue
        rx, ry, rw, rh = map(float, t)
        x0, y0 = rx, ry
        x1, y1 = rx + rw, ry + rh
        out.append(_ShapelyPolygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
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
# 5° bins so Phase 1 can hit goal orientation reliably.
_YAW_BINS = 72
_YAW_STEP_RAD = 2.0 * math.pi / float(_YAW_BINS)
_YAW_GOAL_TOL_BINS = 0  # accept if within +/- this many yaw bins

# Rotation penalty (kept in similar "cost units" as motion costs)
_ROTATE_COST_PER_BIN = 0.35

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
        if shape_name not in objs:
            raise ValueError(f"Unknown robot.shape_name='{shape_name}'. Available keys: {sorted(objs.keys())}")
        poly = objs[shape_name].geometry
        pts_raw = list(poly.exterior.coords)
        pts = [(float(x), float(y)) for (x, y) in pts_raw]
        if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
            pts = pts[:-1]
        pts = _vertices_centered_at_centroid(pts)
        # create_standard_objects() uses fixed ~0.3–1 m shapes; scale so circumradius matches
        # max(width,length)/2 (same characteristic size as the disk holonomic robot in HA_draw).
        width = float(scenario_robot.get("width", 2.0))
        length = float(scenario_robot.get("length", 3.0))
        target_r = max(width, length) / 2.0
        r_ref = max(math.hypot(x, y) for x, y in pts)
        if r_ref > 1e-9:
            s = target_r / r_ref
            pts = [(s * x, s * y) for x, y in pts]
        return pts

    # OBJ -> 2D vertices slice (headless "real" pipeline)
    for k in ("obj_path", "mesh_obj", "obj"):
        if k in scenario_robot and scenario_robot[k]:
            from object_utils import read_obj_to_vertices

            vertices = read_obj_to_vertices(scenario_robot[k])
            if len(vertices) < 3:
                raise ValueError("OBJ->2D slice produced < 3 vertices")
            return _vertices_centered_at_centroid(vertices)

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


def _state_index_se2(x: int, y: int, yaw_idx: int, m_in: int, P: base_astar.Para) -> int:
    cell = (y - P.miny) * P.xw + (x - P.minx)
    m_id = m_in + 1  # -1 -> 0, 0..7 -> 1..8
    return (((cell * _YAW_BINS) + yaw_idx) * 9) + m_id


class _AugNodeSE2:
    __slots__ = ("x", "y", "yaw_idx", "m_in", "cost", "p_ind")

    def __init__(self, x: int, y: int, yaw_idx: int, m_in: int, cost: float, p_ind: int):
        self.x = x
        self.y = y
        self.yaw_idx = yaw_idx
        self.m_in = m_in
        self.cost = cost
        self.p_ind = p_ind


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
    obstacle_rects: Optional[List[Tuple[float, float, float, float]]] = None,
    obstacle_polygons: Optional[List[List[Tuple[float, float]]]] = None,
    map_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[List[float], List[float], List[float]]:
    motion = _get_motion()

    sx_i, sy_i = round(sx / reso), round(sy / reso)
    gx_i, gy_i = round(gx / reso), round(gy / reso)
    syaw_idx = _yaw_to_bin(syaw_rad)
    gyaw_idx = _yaw_to_bin(gyaw_rad)

    # Hard clearance threshold for footprint samples. This is the "true" collision gate.
    # We also use it to define a conservative occupancy radius for quick acceptance below.
    hard_pad = (
        max(_SHAPE_MIN_HARD_CLEARANCE_PAD, _SHAPE_HARD_CLEARANCE_PAD_FACTOR * float(reso))
        + float(safety_margin)
    )

    # Bounding circle for cheap occupancy prefiltering.
    rb = float(max(math.hypot(x, y) for (x, y) in robot_vertices_local))
    # If a cell is free for this inflated disk, then the polygon boundary samples are guaranteed
    # to have clearance > hard_pad (disk encloses polygon). This lets us skip the expensive
    # `_pose_is_valid` evaluation in the common case.
    occ_rr = rb + hard_pad

    obstacle_rect_polys = _rect_polys_from_obstacle_rects(obstacle_rects)
    obstacle_poly_geoms = _poly_obstacles_from_vertices(obstacle_polygons)
    obstacle_geoms = obstacle_rect_polys + obstacle_poly_geoms
    obstacle_tree, obstacle_bounds = _build_spatial_index(obstacle_geoms)
    obstacle_vertices_xy = {id(g): _geom_vertices_xy(g) for g in obstacle_geoms}
    map_inner = None
    if _HAS_SHAPELY and map_bounds is not None:
        x0, y0, x1, y1 = map(float, map_bounds)
        if x1 > x0 and y1 > y0:
            world_box = _ShapelyBox(x0, y0, x1, y1)
            map_inner = world_box.buffer(1e-6)

    ox_g = [float(x) / reso for x in ox]
    oy_g = [float(y) / reso for y in oy]
    P, obsmap = base_astar.calc_parameters(ox_g, oy_g, occ_rr, reso)

    def in_bounds(x: int, y: int) -> bool:
        if x <= P.minx or x >= P.maxx or y <= P.miny or y >= P.maxy:
            return False
        return True

    def ok_cell_fast(x: int, y: int) -> bool:
        """Cheap disk-occupancy check (conservative accept only)."""
        return not obsmap[x - P.minx][y - P.miny]

    # Boundary samples for clearance against obstacle point cloud.
    sample_step = max(0.05, _SHAPE_SAMPLE_STEP_FACTOR * float(reso))
    robot_boundary = _sample_polygon_boundary(
        robot_vertices_local, sample_step=sample_step, max_samples=_SHAPE_MAX_BOUNDARY_SAMPLES
    )  # (N_pts,2)

    if ox and oy and _HAS_KDTREE:
        tree = _cKDTree(np.column_stack([ox, oy]).astype(np.float64))
        obs_arr = None
    else:
        tree = None
        obs_arr = np.column_stack([ox, oy]).astype(np.float64) if (ox and oy) else np.zeros((0, 2), dtype=np.float64)

    clearance_cache: Dict[Tuple[int, int, int], float] = {}
    valid_cache: Dict[Tuple[int, int, int], bool] = {}

    def _min_clearance_at_pose(x_i: int, y_i: int, yaw_idx: int) -> float:
        key = (x_i, y_i, yaw_idx)
        if key in clearance_cache:
            return clearance_cache[key]

        xw = float(x_i) * float(reso)
        yw = float(y_i) * float(reso)
        yaw = _bin_to_yaw(yaw_idx)
        c = math.cos(yaw)
        s = math.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        world_pts = (robot_boundary @ R.T) + np.array([xw, yw], dtype=np.float64)

        if tree is not None:
            d, _ = tree.query(world_pts, k=1)
            d_min = float(np.min(d))
        else:
            if obs_arr.shape[0] == 0:
                d_min = float("inf")
            else:
                diff = world_pts[:, None, :] - obs_arr[None, :, :]
                dist = np.hypot(diff[..., 0], diff[..., 1])
                d_min = float(np.min(dist))

        clearance_cache[key] = d_min
        return d_min

    def _pose_is_valid(x_i: int, y_i: int, yaw_idx: int) -> bool:
        key = (x_i, y_i, yaw_idx)
        if key in valid_cache:
            return valid_cache[key]

        xw = float(x_i) * float(reso)
        yw = float(y_i) * float(reso)
        yaw = _bin_to_yaw(yaw_idx)
        c = math.cos(yaw)
        s = math.sin(yaw)
        world_vertices = [(c * vx - s * vy + xw, s * vx + c * vy + yw) for (vx, vy) in robot_vertices_local]

        # Stage A (cheap): if point-cloud gate fails, reject immediately.
        d_min = _min_clearance_at_pose(x_i, y_i, yaw_idx)
        if not (d_min > hard_pad):
            valid_cache[key] = False
            return False

        if not _HAS_SHAPELY:
            valid_cache[key] = True
            return True

        poly = _ShapelyPolygon(world_vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            valid_cache[key] = False
            return False

        # Stage B (exact geometry): check continuous map wall + obstacle polygons.
        if map_inner is not None:
            if map_inner.is_empty or (not map_inner.covers(poly)):
                valid_cache[key] = False
                return False

        # Stage B: narrow expensive checks to only nearby obstacle AABBs.
        query_bounds = _expand_bounds(tuple(map(float, poly.bounds)), hard_pad)
        obstacle_candidates = _query_spatial_candidates(query_bounds, obstacle_geoms, obstacle_tree, obstacle_bounds)

        # Cheap early rejects:
        # 1) any robot vertex inside an obstacle candidate
        for vx, vy in world_vertices:
            p = _ShapelyPoint(float(vx), float(vy))
            for og in obstacle_candidates:
                if og.covers(p):
                    valid_cache[key] = False
                    return False
        # 2) any obstacle vertex inside robot polygon
        for og in obstacle_candidates:
            for ovx, ovy in obstacle_vertices_xy.get(id(og), []):
                if poly.covers(_ShapelyPoint(float(ovx), float(ovy))):
                    valid_cache[key] = False
                    return False

        for og in obstacle_candidates:
            if poly.distance(og) <= hard_pad + 1e-9:
                valid_cache[key] = False
                return False

        # Stage C: sparse-point containment fallback.
        if not ox:
            valid_cache[key] = True
            return True

        query_r = float(rb + hard_pad + max(0.05, 0.25 * float(reso)))
        near_pts: np.ndarray
        if tree is not None:
            idxs = tree.query_ball_point([xw, yw], r=query_r)
            if not idxs:
                valid_cache[key] = True
                return True
            near_pts = np.asarray(tree.data[idxs], dtype=np.float64)
        else:
            if obs_arr.shape[0] == 0:
                valid_cache[key] = True
                return True
            dx = obs_arr[:, 0] - xw
            dy = obs_arr[:, 1] - yw
            mask = (dx * dx + dy * dy) <= (query_r * query_r)
            if not np.any(mask):
                valid_cache[key] = True
                return True
            near_pts = obs_arr[mask]

        for ptx, pty in near_pts:
            if poly.covers(_ShapelyPoint(float(ptx), float(pty))):
                valid_cache[key] = False
                return False

        valid_cache[key] = True
        return True

    def _heuristic_se2(x_i: int, y_i: int, yaw_idx: int) -> float:
        xy_h = math.hypot(x_i - gx_i, y_i - gy_i)
        yaw_diff_bins = _yaw_delta_bins(yaw_idx, gyaw_idx)
        return xy_h + 0.15 * float(yaw_diff_bins)

    def goal_reached(x_i: int, y_i: int, yaw_idx: int) -> bool:
        if x_i != gx_i or y_i != gy_i:
            return False
        return _yaw_delta_bins(yaw_idx, gyaw_idx) <= _YAW_GOAL_TOL_BINS

    if not in_bounds(sx_i, sy_i):
        return [], [], []
    if not ok_cell_fast(sx_i, sy_i):
        if not _pose_is_valid(sx_i, sy_i, syaw_idx):
            return [], [], []

    start_idx = _state_index_se2(sx_i, sy_i, syaw_idx, M_PREV_NONE, P)
    n_start = _AugNodeSE2(sx_i, sy_i, syaw_idx, M_PREV_NONE, 0.0, -1)

    open_entries: Dict[int, _AugNodeSE2] = {start_idx: n_start}
    closed: Dict[int, _AugNodeSE2] = {}
    pq: List[Tuple[float, int]] = []
    heapq.heappush(pq, (_heuristic_se2(sx_i, sy_i, syaw_idx), start_idx))

    goal_idx_found: Optional[int] = None

    while pq:
        _, idx = heapq.heappop(pq)
        if idx in closed:
            continue
        if idx not in open_entries:
            continue

        cur = open_entries.pop(idx)
        closed[idx] = cur

        if goal_reached(cur.x, cur.y, cur.yaw_idx):
            goal_idx_found = idx
            break

        # Translation edges (yaw unchanged)
        for mi, mv in enumerate(motion):
            nx, ny = cur.x + mv[0], cur.y + mv[1]
            if not in_bounds(nx, ny):
                continue
            nyaw_idx = cur.yaw_idx
            # If disk-occupancy says "free", accept without expensive footprint check.
            # Otherwise, fall back to the true footprint clearance test.
            if not ok_cell_fast(nx, ny):
                if not _pose_is_valid(nx, ny, nyaw_idx):
                    continue

            d_dest = _min_clearance_at_pose(nx, ny, nyaw_idx)
            c_move = _u_cost(mv)
            c_risk = _risk_cost(d_dest)
            c_heading = _HEADING_WEIGHT * _angle_between_moves(cur.m_in, mi, motion)
            c = c_move + c_risk + c_heading

            child_idx = _state_index_se2(nx, ny, nyaw_idx, mi, P)
            g_new = cur.cost + c

            if child_idx in closed:
                continue
            if child_idx in open_entries:
                if g_new < open_entries[child_idx].cost:
                    open_entries[child_idx] = _AugNodeSE2(nx, ny, nyaw_idx, mi, g_new, idx)
                    heapq.heappush(pq, (g_new + _heuristic_se2(nx, ny, nyaw_idx), child_idx))
            else:
                open_entries[child_idx] = _AugNodeSE2(nx, ny, nyaw_idx, mi, g_new, idx)
                heapq.heappush(pq, (g_new + _heuristic_se2(nx, ny, nyaw_idx), child_idx))

        # Rotation edges (in-place)
        for dyaw in (-1, 1):
            nyaw = (cur.yaw_idx + dyaw) % _YAW_BINS
            if not ok_cell_fast(cur.x, cur.y):
                if not _pose_is_valid(cur.x, cur.y, nyaw):
                    continue

            d_dest = _min_clearance_at_pose(cur.x, cur.y, nyaw)
            c_yaw = _ROTATE_COST_PER_BIN * abs(dyaw)
            c_risk = _risk_cost(d_dest)
            c = c_yaw + c_risk

            child_idx = _state_index_se2(cur.x, cur.y, nyaw, cur.m_in, P)
            g_new = cur.cost + c

            if child_idx in closed:
                continue
            if child_idx in open_entries:
                if g_new < open_entries[child_idx].cost:
                    open_entries[child_idx] = _AugNodeSE2(cur.x, cur.y, nyaw, cur.m_in, g_new, idx)
                    heapq.heappush(pq, (g_new + _heuristic_se2(cur.x, cur.y, nyaw), child_idx))
            else:
                open_entries[child_idx] = _AugNodeSE2(cur.x, cur.y, nyaw, cur.m_in, g_new, idx)
                heapq.heappush(pq, (g_new + _heuristic_se2(cur.x, cur.y, nyaw), child_idx))

    if goal_idx_found is None:
        return [], [], []

    # Reconstruct.
    path_grid: List[Tuple[int, int, int]] = []
    walk = goal_idx_found
    while walk != -1:
        node = closed[walk]
        path_grid.append((node.x, node.y, node.yaw_idx))
        walk = node.p_ind
    path_grid.reverse()

    pathx = [float(xi) * float(reso) for (xi, _yi, _yaw_i) in path_grid]
    pathy = [float(yi) * float(reso) for (_xi, yi, _yaw_i) in path_grid]
    pathyaw = [_bin_to_yaw(yaw_i) for (_xi, _yi, yaw_i) in path_grid]
    return pathx, pathy, pathyaw


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

    def _edge_feasible(i: int, j: int) -> bool:
        xi, yi, yawi = float(px[i]), float(py[i]), float(pyaw[i])
        xj, yj, yawj = float(px[j]), float(py[j]), float(pyaw[j])

        dx_w = xj - xi
        dy_w = yj - yi
        # End position expressed in i's body frame.
        c, s = math.cos(yawi), math.sin(yawi)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w

        # Default phase1 yaw wrap choice.
        theta_end_rel = _angle_wrap(yawj - yawi)

        v_body, omega, T = _solve_constant_body_twist_from_SE2_no_fixed_speed(dx_b, dy_b, theta_end_rel)

        # Sample density: enough for both translation and rotation.
        dist_end = math.hypot(dx_w, dy_w)
        n_len = int(dist_end / max(1e-6, 0.4 * float(reso))) + 1
        n_yaw = int(abs(theta_end_rel) / max(1e-6, 0.5 * _YAW_STEP_RAD)) + 1
        n_steps = max(10, n_len, n_yaw)
        n_steps = min(80, n_steps)

        positions_rel, theta_rel = _propagate_body_twist_rel(v_body, omega, T, n_steps=n_steps)

        R_i = _rot2d_matrix(yawi)
        centers_world = (R_i @ positions_rel.T).T + np.array([xi, yi], dtype=np.float64)
        yaws_world = yawi + theta_rel

        # Clearance check by sampling robot boundary points against obstacle point cloud.
        for k in range(centers_world.shape[0]):
            pxk, pyk = float(centers_world[k, 0]), float(centers_world[k, 1])
            yawk = float(yaws_world[k])
            ck, sk = math.cos(yawk), math.sin(yawk)

            if _HAS_SHAPELY and (map_inner is not None or obstacle_geoms):
                world_vertices = [(ck * vx - sk * vy + pxk, sk * vx + ck * vy + pyk) for (vx, vy) in robot_vertices_local]
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

                # Cheap early reject before expensive distance:
                # 1) robot vertices inside candidate obstacles
                for vx, vy in world_vertices:
                    p = _ShapelyPoint(float(vx), float(vy))
                    for og in obstacle_candidates:
                        if og.covers(p):
                            return False
                # 2) candidate obstacle vertices inside robot polygon
                for og in obstacle_candidates:
                    for ovx, ovy in obstacle_vertices_xy.get(id(og), []):
                        if poly.covers(_ShapelyPoint(float(ovx), float(ovy))):
                            return False

                for og in obstacle_candidates:
                    if poly.distance(og) <= float(clearance) + 1e-9:
                        return False

            # Rotate local boundary points by yawk and translate to world.
            bx = robot_boundary[:, 0]
            by = robot_boundary[:, 1]
            world_pts = np.stack([ck * bx - sk * by + pxk, sk * bx + ck * by + pyk], axis=1)

            if tree is not None:
                d, _ = tree.query(world_pts, k=1)
                d_arr = np.asarray(d, dtype=np.float64).reshape(-1)
            else:
                if obs_arr.shape[0] == 0:
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

            if float(np.min(d_arr)) <= float(clearance):
                return False
        return True

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
        n_len = int(dist_end / max(1e-6, 0.4 * float(reso))) + 1
        n_yaw = int(abs(theta_end_rel) / max(1e-6, 0.5 * _YAW_STEP_RAD)) + 1
        n_steps = max(10, n_len, n_yaw)
        n_steps = min(80, n_steps)

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
) -> Tuple[List[float], List[float], List[float]]:
    """
    SE(2) mod_grid_SE pipeline.

    Currently supported:
      - stop_phase == 1: SE(2) Phase 1 augmented A* using footprint boundary clearance.
    """
    if stop_phase != 1:
        if stop_phase != 3:
            raise NotImplementedError("mod_grid_SE Phase 2 is not yet implemented. Use stop_phase=1 or stop_phase=3.")

        # Phase 3: SE(2) primitive compression using constant body-twist trajectories.
        px0, py0, pyaw0 = phase1_augmented_astar_se2(
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
        )
        if len(px0) < 2:
            return px0, py0, pyaw0
        hard_pad = (
            max(_SHAPE_MIN_HARD_CLEARANCE_PAD, _SHAPE_HARD_CLEARANCE_PAD_FACTOR * float(reso)) + float(safety_margin)
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
    )


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
    syaw_rad = math.radians(float(px0[2]) if len(px0) >= 3 else 0.0)
    gyaw_rad = math.radians(float(gx0[2]) if len(gx0) >= 3 else 0.0)
    safety_margin = float(robot.get("safety_margin", 0.0))
    robot_vertices_local = _extract_robot_footprint_vertices_local(
        robot,
        reso=float(scenario.get("map", {}).get("resolution", 1.0)),
    )

    ox, oy, reso, map_w, map_h = _obstacle_points_from_app_scenario(scenario)

    obs = scenario.get("obstacles", {})
    rects_raw = obs.get("rects", {})
    obstacle_rects: List[Tuple[float, float, float, float]] = []
    for v in rects_raw.values():
        if len(v) == 4:
            obstacle_rects.append(tuple(map(float, v)))  # type: ignore[arg-type]

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
