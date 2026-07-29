#!/usr/bin/env python3
"""
Holonomic object-path helpers: zigzag/sine reference geometry, HybridPath construction,
holonomic Pure Pursuit (experimental), and ThetaMode handling for Phase7-style commands.

Used by scripts/test/test_magnum_holonomic_control.py (holonomic robot-object experiments).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import rospkg

rospack = rospkg.RosPack()
_pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(_pkg_path / "src" / "legacy"))

from paths_lib import (  # noqa: E402
    HybridPath,
    SplineComponentPath,
    StraightComponentPath,
    ArcComponentPath,
)


class ThetaMode(IntEnum):
    """How desired heading is specified for the holonomic object."""

    WAYPOINT = 0  # discrete headings at corners (PID on theta; XY from path follower)
    FIXED = 1  # constant goal orientation (PID on theta)
    PATH = 2  # omega from path geometry (PathFollowingController)


def _ensure_mod_grid_path() -> None:
    mod_dir = (
        _pkg_path
        / "scripts"
        / "PathPlanning"
        / "Search_based_Planning"
        / "HA_draw"
        / "HybridAstarPlanner"
    )
    if str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))


def zigzag_polyline(
    x_start: float,
    x_end: float,
    y_center: float,
    y_amplitude: float,
    num_segments: int,
) -> Tuple[List[float], List[float]]:
    px: List[float] = []
    py: List[float] = []
    segment_length = (x_end - x_start) / float(num_segments)
    for i in range(num_segments + 1):
        x = x_start + i * segment_length
        if i == 0:
            y = y_center
        elif i % 2 == 1:
            y = y_center + y_amplitude
        else:
            y = y_center - y_amplitude
        px.append(float(x))
        py.append(float(y))
    return px, py


def sine_polyline(
    x_start: float,
    x_end: float,
    amplitude: float,
    omega_x: float,
    num_samples: int,
) -> Tuple[List[float], List[float]]:
    xs = np.linspace(x_start, x_end, max(8, int(num_samples)), dtype=float)
    ys = (amplitude * np.sin(omega_x * xs)).tolist()
    return xs.tolist(), ys


def primitives_to_hybrid_path(
    prims: Sequence[Tuple[str, dict]],
) -> HybridPath:
    comps: List = []
    for typ, p in prims:
        if typ == "S":
            comps.append(
                StraightComponentPath(
                    [float(p["x0"]), float(p["y0"])],
                    [float(p["x1"]), float(p["y1"])],
                )
            )
        elif typ in ("A", "C"):
            ocx = float(p["ocx"])
            ocy = float(p["ocy"])
            r = float(p["r"])
            a0 = float(p["a0"])
            sweep = float(p["sweep"])
            a1 = a0 + sweep
            clockwise = sweep < 0.0
            comps.append(
                ArcComponentPath([ocx, ocy], r, a0, a1, clockwise=clockwise)
            )
        else:
            raise ValueError(f"Unknown primitive type {typ}")
    return HybridPath(comps)


def polyline_to_hybrid_path_straight_arc(
    px: List[float],
    py: List[float],
    clearance: float = 0.0,
) -> HybridPath:
    """Fit straight + arc via mod_grid; fallback to spline on dense polyline."""
    _ensure_mod_grid_path()
    from mod_grid import phase3_min_segments  # type: ignore

    ox: List[float] = []
    oy: List[float] = []
    outx, outy, prims = phase3_min_segments(
        [float(x) for x in px],
        [float(y) for y in py],
        ox,
        oy,
        float(clearance),
        return_primitives=True,
    )
    if not prims:
        wpts = [[outx[i], outy[i]] for i in range(len(outx))]
        return HybridPath([SplineComponentPath(wpts)])
    return primitives_to_hybrid_path(prims)


def build_hybrid_path_from_planned(
    px: Sequence[float],
    py: Sequence[float],
    primitives: Optional[Sequence[Tuple[str, dict]]] = None,
) -> HybridPath:
    """Build HybridPath from HA_draw planner output."""
    if primitives:
        return primitives_to_hybrid_path(primitives)
    px_list = [float(x) for x in px]
    py_list = [float(y) for y in py]
    if len(px_list) < 2:
        raise ValueError("Planned path must have at least 2 points")
    comps: List = []
    for i in range(len(px_list) - 1):
        comps.append(
            StraightComponentPath(
                [px_list[i], py_list[i]],
                [px_list[i + 1], py_list[i + 1]],
            )
        )
    return HybridPath(comps)


def lateral_error_to_polyline(
    pos: np.ndarray,
    px: Sequence[float],
    py: Sequence[float],
) -> float:
    """Minimum distance from pos to planned polyline (m)."""
    pos = np.asarray(pos, dtype=float).reshape(2)
    pts = np.column_stack([np.asarray(px, dtype=float), np.asarray(py, dtype=float)])
    if len(pts) < 2:
        return float(np.linalg.norm(pos - pts[0])) if len(pts) == 1 else 0.0
    best = 1e18
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-18:
            d = float(np.linalg.norm(pos - a))
        else:
            t = float(np.clip(np.dot(pos - a, ab) / denom, 0.0, 1.0))
            proj = a + t * ab
            d = float(np.linalg.norm(pos - proj))
        best = min(best, d)
    return best


def translate_polyline(
    px: List[float], py: List[float], dx: float, dy: float
) -> Tuple[List[float], List[float]]:
    return [x + dx for x in px], [y + dy for y in py]


def build_zigzag_hybrid_path_at_start(
    start_xy: Sequence[float],
    x_span: Tuple[float, float],
    y_center: float,
    y_amplitude: float,
    num_segments: int,
) -> HybridPath:
    px, py = zigzag_polyline(
        x_span[0], x_span[1], y_center, y_amplitude, num_segments
    )
    dx = float(start_xy[0]) - px[0]
    dy = float(start_xy[1]) - py[0]
    px, py = translate_polyline(px, py, dx, dy)
    comps: List = []
    for i in range(len(px) - 1):
        comps.append(
            StraightComponentPath(
                [px[i], py[i]],
                [px[i + 1], py[i + 1]],
            )
        )
    return HybridPath(comps)


def build_sine_hybrid_path_at_start(
    start_xy: Sequence[float],
    x_span: Tuple[float, float],
    amplitude: float,
    omega_x: float,
    polyline_samples: int = 256,
    clearance: float = 0.0,
    hybrid_mode: str = "polyline",
) -> HybridPath:
    """Build HybridPath for y = A*sin(wx*x) from x_span[0] to x_span[1].

    hybrid_mode:
        polyline (default): chain of StraightComponentPath along a dense sampled sine
            (smooth visually; same structure as zigzag hybrid; no mod_grid).
        arc_fit: fit few straight+arc primitives via ``phase3_min_segments`` (can look
            coarse if sampling is low; increase polyline_samples).
    """
    if hybrid_mode not in ("polyline", "arc_fit"):
        raise ValueError(f"hybrid_mode must be 'polyline' or 'arc_fit', got {hybrid_mode!r}")

    px, py = sine_polyline(
        x_span[0], x_span[1], amplitude, omega_x, polyline_samples
    )
    dx = float(start_xy[0]) - px[0]
    dy = float(start_xy[1]) - py[0]
    px, py = translate_polyline(px, py, dx, dy)

    if hybrid_mode == "polyline":
        comps: List = []
        for i in range(len(px) - 1):
            comps.append(
                StraightComponentPath(
                    [px[i], py[i]],
                    [px[i + 1], py[i + 1]],
                )
            )
        return HybridPath(comps)

    return polyline_to_hybrid_path_straight_arc(px, py, clearance=clearance)


def zigzag_vertex_thetas(
    num_segments: int,
    x_start: float = 0.0,
    x_end: float = 4.0,
    y_center: float = 0.0,
    y_amplitude: float = 0.75,
) -> List[float]:
    """Heading at each vertex (direction toward next), matching generate_waypoints zigzag."""
    segment_length = (x_end - x_start) / float(num_segments)
    thetas: List[float] = []
    waypoints: List[Tuple[float, float]] = []
    for i in range(num_segments + 1):
        x = x_start + i * segment_length
        if i == 0:
            y = y_center
        elif i % 2 == 1:
            y = y_center + y_amplitude
        else:
            y = y_center - y_amplitude
        waypoints.append((x, y))

    for i in range(num_segments + 1):
        x, y = waypoints[i]
        if i < num_segments:
            x_next = x_start + (i + 1) * segment_length
            if i == 0:
                y_next = y_center + y_amplitude
            elif (i + 1) % 2 == 1:
                y_next = y_center + y_amplitude
            else:
                y_next = y_center - y_amplitude
            dx = x_next - x
            dy = y_next - y
            thetas.append(float(math.atan2(dy, dx)))
        else:
            thetas.append(thetas[-1])
    return thetas


def cumulative_vertex_s(hp: HybridPath) -> np.ndarray:
    """Arc length at each component end (includes start 0 and path end)."""
    s_list = [0.0]
    for i in range(hp.num_components):
        s_list.append(float(hp.cumulative_lengths[i + 1]))
    return np.array(s_list, dtype=float)


def nearest_s_on_polyline(pts: np.ndarray, cum: np.ndarray, pos: np.ndarray, n_probe: int = 400) -> float:
    """Approximate arc length on polyline (pts, cum) closest to pos."""
    pts = np.asarray(pts, dtype=float)
    cum = np.asarray(cum, dtype=float)
    L = float(cum[-1])
    if L < 1e-9:
        return 0.0
    ss = np.linspace(0.0, L, n_probe)
    best_s = 0.0
    best_d = 1e18
    for s in ss:
        idx = int(np.searchsorted(cum, s, side="right") - 1)
        idx = max(0, min(idx, len(pts) - 2))
        seg_len = cum[idx + 1] - cum[idx]
        if seg_len < 1e-12:
            p = pts[idx]
        else:
            t = (s - cum[idx]) / seg_len
            p = (1.0 - t) * pts[idx] + t * pts[idx + 1]
        d = float(np.sum((p - pos) ** 2))
        if d < best_d:
            best_d = d
            best_s = float(s)
    return best_s


def nearest_s_on_hybrid_path(hp: HybridPath, pos: np.ndarray, n_probe: int = 256) -> float:
    L = hp.total_length
    if L < 1e-9:
        return 0.0
    ss = np.linspace(0.0, L, n_probe)
    best_s = 0.0
    best_d = 1e18
    for s in ss:
        p = np.asarray(hp.get_point_at_arc_length(float(s)))
        d = float(np.sum((p - pos) ** 2))
        if d < best_d:
            best_d = d
            best_s = float(s)
    return best_s


@dataclass
class ScalarTrapezoidProfile:
    """Simple symmetric trapezoid v(s) on [0,L] with v(0)=v(L)=0."""

    total_length: float
    a_max: float
    v_user_max: float

    def __post_init__(self) -> None:
        L = max(self.total_length, 0.0)
        a = max(self.a_max, 1e-9)
        if L < 1e-12:
            self.v_cruise = 0.0
            self.s_accel = 0.0
            self.s_cruise = 0.0
            self.s_decel = 0.0
            return
        v_cruise_max = math.sqrt(a * L)
        self.v_cruise = min(self.v_user_max, v_cruise_max)
        v0 = 0.0
        v1 = 0.0
        s_acc = (self.v_cruise ** 2 - v0 ** 2) / (2 * a) if self.v_cruise > v0 else 0.0
        s_dec = (self.v_cruise ** 2 - v1 ** 2) / (2 * a) if self.v_cruise > v1 else 0.0
        if s_acc + s_dec > L:
            self.v_cruise = math.sqrt(a * L / 2.0)
            s_acc = self.v_cruise ** 2 / (2 * a)
            s_dec = s_acc
        else:
            s_dec = self.v_cruise ** 2 / (2 * a)
            s_acc = self.v_cruise ** 2 / (2 * a)
        rem = L - s_acc - s_dec
        self.s_accel = s_acc
        self.s_cruise = max(rem, 0.0)
        self.s_decel = s_dec

    def get_speed_at_s(self, s: float) -> float:
        s = float(np.clip(s, 0.0, self.total_length))
        a = max(self.a_max, 1e-9)
        v_c = self.v_cruise
        sa = self.s_accel
        sc = self.s_cruise
        if s <= sa:
            return math.sqrt(max(0.0, 2 * a * s))
        elif s <= sa + sc:
            return v_c
        else:
            s_d = s - sa - sc
            sd = self.s_decel
            if sd < 1e-12:
                return 0.0
            return math.sqrt(max(0.0, v_c ** 2 - 2 * a * s_d))


class HolonomicPurePursuitPolyline:
    """
    Experimental: follow a dense polyline with holonomic pure pursuit (toward lookahead).
    Arc-length speed from ScalarTrapezoidProfile.
    """

    def __init__(
        self,
        points_xy: np.ndarray,
        *,
        a_max: float = 0.15,
        v_user_max: float = 0.1,
        Ld: float = 0.25,
        kf: float = 0.5,
    ):
        self.pts = np.asarray(points_xy, dtype=float)
        if len(self.pts) < 2:
            raise ValueError("Pure pursuit needs at least 2 points.")
        seg = np.sqrt(np.sum(np.diff(self.pts, axis=0) ** 2, axis=1))
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(self.cum[-1])
        self.profile = ScalarTrapezoidProfile(self.L, a_max, v_user_max)
        self.Ld = float(Ld)
        self.kf = float(kf)
        self.s_along = 0.0
        self._finished = False

    def reset(self) -> None:
        self.s_along = 0.0
        self._finished = False

    def _point_at_s(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.L))
        idx = int(np.searchsorted(self.cum, s, side="right") - 1)
        idx = max(0, min(idx, len(self.cum) - 2))
        seg_len = self.cum[idx + 1] - self.cum[idx]
        if seg_len < 1e-12:
            return self.pts[idx].copy()
        t = (s - self.cum[idx]) / seg_len
        return (1.0 - t) * self.pts[idx] + t * self.pts[idx + 1]

    def _tangent_at_s(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.L))
        idx = int(np.searchsorted(self.cum, s, side="right") - 1)
        idx = max(0, min(idx, len(self.pts) - 2))
        d = self.pts[idx + 1] - self.pts[idx]
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else np.array([1.0, 0.0])

    def compute_velocity(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        dt: float,
        *,
        omega_override: Optional[float] = None,
    ) -> np.ndarray:
        if self._finished or self.L < 1e-9:
            return np.array([0.0, 0.0, 0.0])

        speed = self.profile.get_speed_at_s(self.s_along)
        ds = float(np.clip(speed * dt, 0.0, self.L - self.s_along))
        self.s_along = min(self.L, self.s_along + ds)
        if self.s_along >= self.L - 1e-6:
            self._finished = True
            return np.array([0.0, 0.0, 0.0 if omega_override is None else float(omega_override)])

        s = self.s_along
        Lf = self.kf * float(np.linalg.norm(current_velocity)) + self.Ld
        s_look = min(self.L, s + Lf)
        p_look = self._point_at_s(s_look)
        direc = p_look - np.asarray(current_position, dtype=float)
        nd = float(np.linalg.norm(direc))
        if nd > 1e-9:
            direc = direc / nd
        else:
            direc = self._tangent_at_s(s)

        v_cmd = speed * direc
        om = 0.0 if omega_override is None else float(omega_override)
        return np.array([v_cmd[0], v_cmd[1], om])

    def is_completed(self) -> bool:
        return self._finished

    @property
    def s_progress(self) -> float:
        return float(self.s_along)

    def tangent_at_progress(self) -> np.ndarray:
        return self._tangent_at_s(self.s_along)

    @staticmethod
    def from_sine(
        start_xy: Sequence[float],
        x_span: Tuple[float, float],
        amplitude: float,
        omega_x: float,
        num_samples: int = 160,
        **kwargs,
    ) -> "HolonomicPurePursuitPolyline":
        px, py = sine_polyline(
            x_span[0], x_span[1], amplitude, omega_x, num_samples
        )
        dx = float(start_xy[0]) - px[0]
        dy = float(start_xy[1]) - py[0]
        px, py = translate_polyline(px, py, dx, dy)
        pts = np.column_stack([px, py])
        return HolonomicPurePursuitPolyline(pts, **kwargs)

    @staticmethod
    def from_zigzag(
        start_xy: Sequence[float],
        x_span: Tuple[float, float],
        y_center: float,
        y_amplitude: float,
        num_segments: int,
        **kwargs,
    ) -> "HolonomicPurePursuitPolyline":
        px, py = zigzag_polyline(
            x_span[0], x_span[1], y_center, y_amplitude, num_segments
        )
        dx = float(start_xy[0]) - px[0]
        dy = float(start_xy[1]) - py[0]
        px, py = translate_polyline(px, py, dx, dy)
        pts = np.column_stack([px, py])
        return HolonomicPurePursuitPolyline(pts, **kwargs)


def align_heading_to_current(current_orientation: float, heading_ref: float) -> float:
    """Shift ``heading_ref`` by integer multiples of 2π to be nearest ``current_orientation`` (for PID)."""
    two_pi = 2.0 * math.pi
    k = round((float(heading_ref) - float(current_orientation)) / two_pi)
    return float(float(heading_ref) - k * two_pi)


def orientation_pid_omega(
    current_orientation: float,
    goal_theta: float,
    current_omega: float,
    kp: float = 0.8,
    kd: float = 0.2,
    max_omega: float = 0.15,
) -> float:
    goal_theta = align_heading_to_current(current_orientation, goal_theta)
    err = goal_theta - current_orientation
    err = math.atan2(math.sin(err), math.cos(err))
    w = kp * err - kd * current_omega
    return float(np.clip(w, -max_omega, max_omega))


def theta_goal_for_waypoint_mode(
    current_s: float,
    s_milestones: Sequence[float],
    theta_milestones: Sequence[float],
) -> float:
    """Deprecated piecewise-constant fallback; prefer ``theta_goal_linear_along_segments``."""
    goal = float(theta_milestones[0])
    for s_th, th in zip(s_milestones, theta_milestones):
        if current_s + 1e-6 >= float(s_th):
            goal = float(th)
    return goal


@dataclass(frozen=True)
class SegmentThetaSpec:
    """One planner primitive span with linear theta reference."""

    s0: float
    s1: float
    theta0: float
    theta1: float


def segment_theta_specs_from_df_primitives(
    df: Sequence[dict],
) -> List[SegmentThetaSpec]:
    """Per-primitive (s0, s1, theta0, theta1) matching ``df_xy_linear_theta_v1``."""
    if not df:
        return []
    specs: List[SegmentThetaSpec] = []
    cum = 0.0
    for seg in df:
        kind = str(seg.get("kind", "line"))
        if kind == "line":
            length = math.hypot(
                float(seg["x1"]) - float(seg["x0"]),
                float(seg["y1"]) - float(seg["y0"]),
            )
        else:
            length = abs(float(seg["sweep"])) * max(float(seg["r"]), 1e-9)
        s0 = cum
        s1 = cum + float(length)
        specs.append(
            SegmentThetaSpec(
                s0=s0,
                s1=s1,
                theta0=float(seg["theta0"]),
                theta1=float(seg["theta1"]),
            )
        )
        cum = s1
    return specs


def resolve_segment_theta_specs(
    *,
    df_primitives: Optional[Sequence[dict]] = None,
    s_vertices: Optional[Sequence[float]] = None,
    theta_vertices: Optional[Sequence[float]] = None,
) -> List[SegmentThetaSpec]:
    if df_primitives:
        return segment_theta_specs_from_df_primitives(df_primitives)
    if s_vertices and theta_vertices:
        return segment_theta_specs_from_vertex_milestones(s_vertices, theta_vertices)
    return []


def segment_theta_specs_from_vertex_milestones(
    s_vertices: Sequence[float],
    theta_vertices: Sequence[float],
) -> List[SegmentThetaSpec]:
    """Build linear segment specs from headings at polyline vertices."""
    if len(s_vertices) < 2 or len(theta_vertices) < 2:
        return []
    n = min(len(s_vertices), len(theta_vertices))
    specs: List[SegmentThetaSpec] = []
    for i in range(n - 1):
        specs.append(
            SegmentThetaSpec(
                s0=float(s_vertices[i]),
                s1=float(s_vertices[i + 1]),
                theta0=float(theta_vertices[i]),
                theta1=float(theta_vertices[i + 1]),
            )
        )
    return specs


def segment_index_for_s(current_s: float, segments: Sequence[SegmentThetaSpec]) -> int:
    if not segments:
        return 0
    s = float(current_s)
    for i, seg in enumerate(segments):
        if s <= seg.s1 + 1e-6:
            return i
    return len(segments) - 1


def theta_goal_at_segment_endpoint(
    current_s: float,
    segments: Sequence[SegmentThetaSpec],
    current_orientation: Optional[float] = None,
) -> float:
    """Piecewise-constant heading: current primitive's mandated endpoint ``theta1``."""
    if not segments:
        return 0.0
    s = float(np.clip(current_s, segments[0].s0, segments[-1].s1))
    idx = segment_index_for_s(s, segments)
    goal = float(segments[idx].theta1)
    if current_orientation is not None:
        goal = align_heading_to_current(current_orientation, goal)
    return goal


def theta_goal_linear_along_segments(
    current_s: float,
    segments: Sequence[SegmentThetaSpec],
    current_orientation: Optional[float] = None,
) -> float:
    """Deprecated: use ``theta_goal_at_segment_endpoint`` (piecewise endpoint reference)."""
    return theta_goal_at_segment_endpoint(current_s, segments, current_orientation)


def orient_gate_should_hold(
    current_orientation: float,
    seg: SegmentThetaSpec,
    tolerance: float,
    max_residual: float = 0.35,
) -> bool:
    """True for a small segment-end residual that along-path PID did not clear."""
    goal = align_heading_to_current(current_orientation, float(seg.theta1))
    err = orientation_error_rad(current_orientation, goal)
    if abs(err) <= float(tolerance):
        return False
    return abs(err) <= float(max_residual)


def completed_segment_at_s_crossing(
    prev_s: float,
    current_s: float,
    segments: Sequence[SegmentThetaSpec],
) -> Optional[int]:
    """Return completed primitive index when ``current_s`` crosses its end."""
    if not segments:
        return None
    for i, seg in enumerate(segments):
        if float(prev_s) < float(seg.s1) - 1e-6 and float(current_s) >= float(seg.s1) - 1e-6:
            return i
    return None


def mandated_theta_at_segment_end(
    segments: Sequence[SegmentThetaSpec],
    completed_seg_idx: int,
) -> float:
    if not segments:
        return 0.0
    idx = int(np.clip(completed_seg_idx, 0, len(segments) - 1))
    return float(segments[idx].theta1)


@dataclass
class HolonomicSegmentOrientGate:
    """Hold XY and rotate to mandated primitive-end theta before continuing."""

    segment_specs: Sequence[SegmentThetaSpec]
    orientation_tol: float = 0.1
    orient_gate_max_residual: float = 0.35
    gate_active: bool = False
    gate_theta: float = 0.0
    active_boundary_key: Optional[Tuple[int, int]] = None
    pending_retouch_key: Optional[Tuple[int, int]] = None
    pending_retouch: bool = False
    retouch_resume_theta: float = 0.0

    def reset(self) -> None:
        self.gate_active = False
        self.gate_theta = 0.0
        self.active_boundary_key = None
        self.pending_retouch_key = None
        self.pending_retouch = False
        self.retouch_resume_theta = 0.0

    def begin_segment_end_hold(
        self,
        completed_seg_idx: int,
        *,
        boundary_key: Tuple[int, int],
        do_retouch: bool,
        retouch_already_consumed: bool,
        current_orientation: Optional[float] = None,
        current_s: Optional[float] = None,
    ) -> bool:
        """
        Arm the orient gate for primitive ``completed_seg_idx``.

        Returns True when an active XY hold + rotate is required; False when heading
        is already within tolerance or residual is too large for a gate touch-up.
        """
        self.active_boundary_key = boundary_key
        seg = self.segment_specs[int(completed_seg_idx)]
        self.gate_theta = align_heading_to_current(
            float(current_orientation or seg.theta1),
            mandated_theta_at_segment_end(self.segment_specs, completed_seg_idx),
        )
        self.retouch_resume_theta = self.gate_theta
        self.pending_retouch_key = boundary_key
        self.pending_retouch = bool(do_retouch and not retouch_already_consumed)
        if current_orientation is None:
            self.gate_active = True
            return True
        if self.orientation_satisfied(current_orientation, self.gate_theta):
            self.gate_active = False
            return False
        if not orient_gate_should_hold(
            current_orientation,
            seg,
            self.orientation_tol,
            self.orient_gate_max_residual,
        ):
            self.gate_active = False
            return False
        self.gate_active = True
        return True

    def orientation_satisfied(self, current_orientation: float, goal_theta: float) -> bool:
        goal_theta = align_heading_to_current(current_orientation, goal_theta)
        return abs(orientation_error_rad(current_orientation, goal_theta)) < float(
            self.orientation_tol
        )

    def clear_gate(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Return ``(retouch_boundary_key, orient_boundary_key)`` and disarm the gate."""
        self.gate_active = False
        retouch_key = self.pending_retouch_key if self.pending_retouch else None
        orient_key = self.active_boundary_key
        self.pending_retouch_key = None
        self.pending_retouch = False
        self.active_boundary_key = None
        return retouch_key, orient_key

    def retouch_may_resume(self, current_orientation: float) -> bool:
        return self.orientation_satisfied(current_orientation, self.retouch_resume_theta)


def apply_orientation_hold(
    *,
    current_orientation: float,
    current_angular_velocity: float,
    hold_theta: float,
    orientation_tol: float,
    kp: float = 0.8,
    kd: float = 0.2,
    max_omega: float = 0.15,
) -> Tuple[np.ndarray, float, bool]:
    """Freeze translation and PID toward ``hold_theta``; third value is satisfied."""
    hold_theta = align_heading_to_current(current_orientation, hold_theta)
    err = orientation_error_rad(current_orientation, hold_theta)
    if abs(err) < float(orientation_tol):
        return np.array([0.0, 0.0], dtype=float), 0.0, True
    omega = orientation_pid_omega(
        current_orientation,
        hold_theta,
        current_angular_velocity,
        kp=kp,
        kd=kd,
        max_omega=max_omega,
    )
    return np.array([0.0, 0.0], dtype=float), omega, False


def orientation_error_rad(current_orientation: float, goal_theta: float) -> float:
    err = float(goal_theta) - float(current_orientation)
    return math.atan2(math.sin(err), math.cos(err))


def final_theta_goal_for_mode(
    theta_mode: ThetaMode,
    *,
    s_total: float,
    theta_milestones: Sequence[float],
    segment_specs: Optional[Sequence[SegmentThetaSpec]] = None,
    fixed_theta: float = 0.0,
    path_theta_sine_amp: float = 0.0,
    path_theta_sine_k: float = 1.0,
) -> float:
    """Terminal heading used after XY path completion."""
    if theta_mode == ThetaMode.WAYPOINT:
        if segment_specs:
            return float(segment_specs[-1].theta1)
        return float(theta_milestones[-1]) if theta_milestones else 0.0
    if theta_mode == ThetaMode.FIXED:
        return float(fixed_theta)
    return float(path_theta_sine_amp) * math.sin(float(path_theta_sine_k) * float(s_total))


def holonomic_path_xy_completed(
    path_following_controller=None,
    pursuit_controller=None,
) -> bool:
    if path_following_controller is not None and path_following_controller.is_completed():
        return True
    if pursuit_controller is not None and pursuit_controller.is_completed():
        return True
    return False


def apply_path_completion_to_desired_motion(
    *,
    desired_obj_velocity: np.ndarray,
    desired_obj_omega: float,
    current_orientation: float,
    current_angular_velocity: float,
    path_xy_done: bool,
    final_theta: float,
    orientation_tol: float,
    kp: float = 0.8,
    kd: float = 0.2,
    max_omega: float = 0.15,
) -> Tuple[np.ndarray, float, bool]:
    """
    After XY path completion, hold translation and rotate toward ``final_theta``.

    Returns ``(velocity, omega, scenario_complete)`` where ``scenario_complete`` is
    True when both XY is done and ``|orientation error| < orientation_tol``.
    """
    if not path_xy_done:
        return np.asarray(desired_obj_velocity, dtype=float), float(desired_obj_omega), False

    vel = np.array([0.0, 0.0], dtype=float)
    err = orientation_error_rad(current_orientation, final_theta)
    if abs(err) < float(orientation_tol):
        return vel, 0.0, True

    omega = orientation_pid_omega(
        current_orientation,
        final_theta,
        current_angular_velocity,
        kp=kp,
        kd=kd,
        max_omega=max_omega,
    )
    return vel, omega, False
