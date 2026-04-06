#!/usr/bin/env python3
"""
Diff-drive object-path helpers: segment primitive plans (constant body twist),
end-pose-anchored mid_theta, robot heading goals for velocity matching, and
phase enum for Magnum Four + Phase7 diff-drive experiments.

See scripts/test/basic_test/test_matchingvelo_report.md and
scripts/test/basic_test/test_motion_primitive.py (solve_constant_body_twist_from_SE2).
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

from paths_lib import HybridPath  # noqa: E402


class DdThetaMode(IntEnum):
    """How desired vertex headings are set for the diff-drive object."""

    WAYPOINT = 0  # discrete headings at path vertices (e.g. zigzag corners)
    FIXED = 1  # single constant heading at all vertices
    SEGMENT_TANGENT = 2  # heading at vertex = path tangent from HybridPath


class DiffDriveSegmentPhase(IntEnum):
    RETOUCH_A = 0
    ROBOT_ROTATE_A = 1
    OBJECT_ROTATE = 2
    RETOUCH_B = 3
    ROBOT_ROTATE_B = 4
    PUSH = 5


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def solve_constant_body_twist_from_SE2(
    x_end: float,
    y_end: float,
    theta_end: float,
    *,
    v_speed: float = 1.0,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float, float]:
    """
    Same contract as scripts/test/basic_test/test_motion_primitive.py:
    start pose (0,0,0), end (x_end,y_end,theta_end) in start frame, ||v_body||=v_speed.
    """
    if v_speed <= 0:
        raise ValueError("v_speed must be > 0.")

    x_end = float(x_end)
    y_end = float(y_end)
    theta_end = float(theta_end)

    if abs(theta_end) < eps:
        disp = np.array([x_end, y_end], dtype=float)
        dist = float(np.linalg.norm(disp))
        if dist < eps:
            raise ValueError(
                "Ambiguous request: end pose has near-zero displacement and "
                "theta_end~0, so no meaningful motion duration/velocity exists."
            )
        omega = 0.0
        T = dist / v_speed
        v_body = v_speed * disp / dist
        return v_body, omega, T

    theta = theta_end
    A = float(np.sin(theta))
    B = float(1.0 - np.cos(theta))
    D = A * A + B * B
    if D < eps:
        raise ValueError("Arc inversion unstable: theta_end produced near-zero D.")

    z = (1.0 / D) * np.array(
        [A * x_end + B * y_end, -B * x_end + A * y_end],
        dtype=float,
    )
    z_norm = float(np.linalg.norm(z))
    if z_norm < eps:
        raise ValueError(
            "Degenerate request: end displacement is (near-)incompatible with a "
            "nonzero theta_end under the constant-speed body-twist model."
        )

    omega_mag = v_speed / z_norm
    omega = float(np.sign(theta) * omega_mag)
    T = float(theta / omega)

    v_body = omega * z

    v_norm = float(np.linalg.norm(v_body))
    if not np.isfinite(v_norm):
        raise ValueError("Solver produced non-finite v_body.")
    if abs(v_norm - v_speed) > 1e-6 * max(1.0, v_speed):
        raise ValueError(
            f"Numerical issue: ||v_body||={v_norm:.6g} differs from v_speed={v_speed:.6g}"
        )

    return v_body, omega, T


def compute_dd_solutions_forward_backward(
    v_cp_world0: np.ndarray,
    omega_obj: float,
    R_r: float,
    phi0: float,
) -> Tuple[dict, dict]:
    """Forward and backward constant-velocity DD solutions (see test_matchingvelo.py)."""
    a = float(v_cp_world0[0] + omega_obj * R_r * np.sin(phi0))
    b = float(v_cp_world0[1] - omega_obj * R_r * np.cos(phi0))
    speed = float(np.hypot(a, b))
    angle = float(np.arctan2(b, a))
    zeta_fwd = angle
    zeta_bwd = wrap_angle(angle + np.pi)
    return (
        dict(zeta0=zeta_fwd, v_r=speed, omega_r=omega_obj, label="forward"),
        dict(zeta0=zeta_bwd, v_r=-speed, omega_r=omega_obj, label="backward"),
    )


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def outward_normal_world(
    theta_obj: float, n_out_body: np.ndarray
) -> np.ndarray:
    R = rot2d(theta_obj)
    return R @ np.asarray(n_out_body, dtype=float).reshape(2)


def phi0_inward_from_outward_normal(n_out_world: np.ndarray) -> float:
    """phi0 = atan2(-n_y, -n_x) — inward normal / approach direction (matchingvelo)."""
    return float(np.arctan2(-n_out_world[1], -n_out_world[0]))


def contact_point_velocity_world(
    theta_obj: float,
    v_body: np.ndarray,
    omega: float,
    cp_body: np.ndarray,
) -> np.ndarray:
    R = rot2d(theta_obj)
    v_cp_body = np.asarray(v_body, dtype=float).reshape(2) + omega * np.array(
        [-cp_body[1], cp_body[0]], dtype=float
    )
    return R @ v_cp_body


def robot_heading_goal_co_rotate(
    theta_obj: float,
    n_out_body: np.ndarray,
) -> float:
    """
    Holonomic-style heading that keeps the contact normal aligned for co-rotation:
    zeta = theta_obj + atan2(-n_y^b, -n_x^b)  (body-frame inward direction in world).
    """
    psi = float(np.arctan2(-n_out_body[1], -n_out_body[0]))
    return wrap_angle(theta_obj + psi)


def robot_heading_goal_for_push_forward(
    theta_obj: float,
    v_body: np.ndarray,
    omega: float,
    cp_body: np.ndarray,
    n_out_body: np.ndarray,
    R_r: float,
) -> float:
    """Forward-branch zeta0 for constant-twist push (test_matchingvelo_report §5)."""
    v_w = contact_point_velocity_world(theta_obj, v_body, omega, cp_body)
    n_w = outward_normal_world(theta_obj, n_out_body)
    phi0 = phi0_inward_from_outward_normal(n_w)
    fwd, _bwd = compute_dd_solutions_forward_backward(v_w, omega, R_r, phi0)
    return float(fwd["zeta0"])


@dataclass
class SegmentPrimitivePlan:
    """One HybridPath segment: constant body twist from motion-primitive inverse."""

    segment_idx: int
    s_start: float
    s_end: float
    p_start: np.ndarray
    p_end: np.ndarray
    theta_start_world: float
    theta_end_world: float
    dx_local: float
    dy_local: float
    theta_end_local: float
    v_body: np.ndarray
    omega: float
    T: float
    v_speed: float
    mid_theta_world: float
    """End-pose-anchored transition heading: theta_end - omega*T (== theta_start)."""

    zeta_push_forward: float
    """Robot heading (forward branch) at the start of the push segment."""


def zeta_push_for_robot(
    plan: SegmentPrimitivePlan,
    cp_body: np.ndarray,
    n_out_body: np.ndarray,
    R_r: float,
) -> float:
    """Forward-branch zeta for this segment and robot-specific contact geometry."""
    return robot_heading_goal_for_push_forward(
        plan.theta_start_world,
        plan.v_body,
        plan.omega,
        cp_body,
        n_out_body,
        R_r,
    )


def build_vertex_thetas_for_hybrid_path(
    hp: HybridPath,
    mode: DdThetaMode,
    *,
    fixed_theta: float = 0.0,
    zigzag_num_segments: Optional[int] = None,
    zigzag_x0: float = 0.0,
    zigzag_x1: float = 4.0,
    zigzag_y_center: float = 0.0,
    zigzag_y_amplitude: float = 0.75,
) -> np.ndarray:
    """
    One heading per vertex (length num_components + 1).

    WAYPOINT: for zigzag-like paths, use holonomic zigzag_vertex_thetas when
    zigzag_num_segments matches hp.num_components; otherwise tangent at each vertex.
    """
    n = hp.num_components
    verts = n + 1
    if mode == DdThetaMode.FIXED:
        return np.full(verts, wrap_angle(fixed_theta), dtype=float)

    if mode == DdThetaMode.SEGMENT_TANGENT:
        out = np.zeros(verts, dtype=float)
        for j in range(verts):
            s = float(hp.cumulative_lengths[min(j, n)])
            t = hp.get_tangent_at_arc_length(s)
            out[j] = float(np.arctan2(t[1], t[0]))
        return out

    # WAYPOINT
    if zigzag_num_segments is not None and zigzag_num_segments == n:
        from contact_maintain.holonomic_path_control import zigzag_vertex_thetas

        th = zigzag_vertex_thetas(
            zigzag_num_segments,
            zigzag_x0,
            zigzag_x1,
            zigzag_y_center,
            zigzag_y_amplitude,
        )
        return np.asarray(th, dtype=float)

    out = np.zeros(verts, dtype=float)
    for j in range(verts):
        s = float(hp.cumulative_lengths[min(j, n)])
        t = hp.get_tangent_at_arc_length(s)
        out[j] = float(np.arctan2(t[1], t[0]))
    return out


def build_segment_primitive_plans(
    hp: HybridPath,
    vertex_thetas: Sequence[float],
    cp_body: np.ndarray,
    n_out_body: np.ndarray,
    R_r: float,
    v_speed: float,
) -> List[SegmentPrimitivePlan]:
    """
    For each HybridPath component, compute constant-twist primitive and end-anchored mid_theta.
    """
    if len(vertex_thetas) != hp.num_components + 1:
        raise ValueError(
            f"vertex_thetas length {len(vertex_thetas)} != num_components+1={hp.num_components + 1}"
        )

    plans: List[SegmentPrimitivePlan] = []
    for i in range(hp.num_components):
        s0 = float(hp.cumulative_lengths[i])
        s1 = float(hp.cumulative_lengths[i + 1])
        p0 = np.asarray(hp.get_point_at_arc_length(s0), dtype=float).reshape(2)
        p1 = np.asarray(hp.get_point_at_arc_length(s1), dtype=float).reshape(2)
        th0 = float(vertex_thetas[i])
        th1 = float(vertex_thetas[i + 1])

        Rinv = rot2d(-th0)
        delta_w = p1 - p0
        local = Rinv @ delta_w
        dx, dy = float(local[0]), float(local[1])
        th_end_local = wrap_angle(th1 - th0)

        v_body, omega, T = solve_constant_body_twist_from_SE2(
            dx, dy, th_end_local, v_speed=v_speed
        )
        mid_theta = wrap_angle(th1 - omega * T)
        zeta_push = robot_heading_goal_for_push_forward(
            th0, v_body, omega, cp_body, n_out_body, R_r
        )

        plans.append(
            SegmentPrimitivePlan(
                segment_idx=i,
                s_start=s0,
                s_end=s1,
                p_start=p0.copy(),
                p_end=p1.copy(),
                theta_start_world=th0,
                theta_end_world=th1,
                dx_local=dx,
                dy_local=dy,
                theta_end_local=th_end_local,
                v_body=np.asarray(v_body, dtype=float).copy(),
                omega=float(omega),
                T=float(T),
                v_speed=float(v_speed),
                mid_theta_world=float(mid_theta),
                zeta_push_forward=float(zeta_push),
            )
        )
    return plans


def orientation_pid_omega_simple(
    theta: float,
    theta_goal: float,
    omega_meas: float,
    kp: float = 2.0,
    w_max: float = 0.35,
) -> float:
    err = wrap_angle(theta_goal - theta)
    w = kp * err - 0.1 * omega_meas
    return float(np.clip(w, -w_max, w_max))


def robot_rotate_command_diffdrive(
    robot_heading: float,
    zeta_goal: float,
    kp: float = 2.5,
    omega_max: float = 1.2,
) -> np.ndarray:
    """Return [v_forward, omega] for in-place heading tracking."""
    err = wrap_angle(zeta_goal - robot_heading)
    omega = float(np.clip(kp * err, -omega_max, omega_max))
    return np.array([0.0, omega], dtype=float)
