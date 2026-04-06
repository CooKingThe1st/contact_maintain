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
        elif typ == "A":
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
    err = goal_theta - current_orientation
    err = math.atan2(math.sin(err), math.cos(err))
    w = kp * err - kd * current_omega
    return float(np.clip(w, -max_omega, max_omega))


def theta_goal_for_waypoint_mode(
    current_s: float,
    s_milestones: Sequence[float],
    theta_milestones: Sequence[float],
) -> float:
    """Piecewise-constant theta: use last milestone with s_milestone <= current_s."""
    goal = float(theta_milestones[0])
    for s_th, th in zip(s_milestones, theta_milestones):
        if current_s + 1e-6 >= float(s_th):
            goal = float(th)
    return goal
