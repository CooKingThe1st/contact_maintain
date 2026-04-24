#!/usr/bin/env python3
"""
Inverse motion primitive (object SE2 end pose <-> constant body twist).

Given an object boundary shape, pick a contact point parameter `t_param`
using the repo's `ContactPointParameterization`. Under the differential-
flatness / quasi-static assumption, the object's pose is fully determined
by a constant body-frame velocity and constant angular rate:

  - theta(t) = omega * t
  - p_dot_world(t) = R(theta(t)) @ v_body

This script implements the inverse problem for the special case:
  start pose fixed to (x0,y0,theta0) = (0,0,0)
  end pose desired as (x_end,y_end,theta_end)

It solves for:
  - v_body with fixed magnitude ||v_body|| = v_speed (unit direction is key)
  - constant omega
  - duration T

Then it propagates the object and computes the contact-point trajectory:
  cp_world(t) = p(t) + R(theta(t)) @ cp_body

Optionally, it can also plot the differential-drive robot contact-maintenance
trajectory via the existing constant-velocity matching logic.

Example (copy/paste):
  python3 test_motion_primitive.py \
    --shape pi --t_param 0.25 \
    --x_end 0.20 --y_end 0.10 --theta_end 0.20 \
    --v_speed 1.0 --dt 0.01 \
    --save pi_example.png --silent

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as pyb
import rospkg


rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from contact_maintain.object_bridge import obj_to_generic
from object_utils import ContactPointParameterization


OBJ_FILE_MAP = {
    "right_triangle": "right_triangle.obj",
    "bolt": "bolt.obj",
    "pi": "pi.obj",
    "root": "root.obj",
    "rect": "rect.obj",
    "hourglass": "hourglass.obj",
    "meteor": "meteor.obj",
}

SAVE_DIR = Path("/tmp/motion_primitive")


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def solve_constant_body_twist_from_SE2(
    x_end: float,
    y_end: float,
    theta_end: float,
    *,
    v_speed: float = 1.0,
    pure_spin_duration: float = 1.0,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float, float]:
    """
    Solve for (v_body, omega, T) with:
      - start pose fixed to (0,0,0)
      - theta(t) = omega*t
      - p_dot_world = R(theta) @ v_body
      - constraint ||v_body|| = v_speed

    Returns:
      v_body (2,), omega (float), T (float>0)
    """
    if v_speed < 0:
        raise ValueError("v_speed must be >= 0.")
    if pure_spin_duration <= 0:
        raise ValueError("pure_spin_duration must be > 0.")

    x_end = float(x_end)
    y_end = float(y_end)
    theta_end = float(theta_end)
    disp = np.array([x_end, y_end], dtype=float)
    dist = float(np.linalg.norm(disp))

    # Pure-rotation special case: requested translation is (near-)zero while
    # orientation change is nonzero. This is valid under the model with v_body=0.
    if abs(theta_end) >= eps and dist < eps:
        T = float(pure_spin_duration)
        omega = float(theta_end / T)
        v_body = np.zeros(2, dtype=float)
        return v_body, omega, T

    # Straight-line special case
    if abs(theta_end) < eps:
        if dist < eps:
            raise ValueError(
                "Ambiguous request: end pose has near-zero displacement and "
                "theta_end~0, so no meaningful motion duration/velocity exists."
            )
        if v_speed <= 0:
            raise ValueError(
                "Straight-line request needs v_speed > 0. "
                "Use nonzero theta_end with near-zero displacement for pure spin."
            )
        omega = 0.0
        T = dist / v_speed
        v_body = v_speed * disp / dist
        return v_body, omega, T

    # Arc case
    theta = theta_end
    A = float(np.sin(theta))
    B = float(1.0 - np.cos(theta))
    D = A * A + B * B
    if D < eps:
        # This should not happen because we already handled theta_end~0 above,
        # but keep it safe for numerical extremes.
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

    if v_speed <= 0:
        raise ValueError("Arc request needs v_speed > 0.")
    omega_mag = v_speed / z_norm  # |omega|
    omega = float(np.sign(theta) * omega_mag)
    T = float(theta / omega)  # positive by construction

    v_body = omega * z

    # Enforce/verify magnitude constraint (within float tolerance)
    v_norm = float(np.linalg.norm(v_body))
    if not np.isfinite(v_norm):
        raise ValueError("Solver produced non-finite v_body.")
    if abs(v_norm - v_speed) > 1e-6 * max(1.0, v_speed):
        raise ValueError(
            f"Numerical issue: ||v_body||={v_norm:.6g} differs from v_speed={v_speed:.6g}"
        )

    return v_body, omega, T


def propagate_object_pose_constant_body_twist(
    v_body: np.ndarray,
    omega: float,
    T: float,
    *,
    dt: float,
    cp_body: np.ndarray,
    n_out_body: np.ndarray,
    theta0: float = 0.0,
    eps: float = 1e-9,
) -> dict[str, np.ndarray]:
    """
    Drift-free analytic propagation for the object's CoM pose and contact-point
    kinematics under constant (v_body, omega).
    """
    if T < 0:
        raise ValueError("T must be >= 0.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if T == 0:
        n_steps = 1
    else:
        n_steps = int(T / dt) + 1
        n_steps = max(n_steps, 2)

    times = np.linspace(0.0, T, n_steps, dtype=float)
    theta = theta0 + omega * times

    # CoM position: start fixed to (0,0) as required by inverse solver.
    if abs(omega) < eps:
        positions = np.outer(times, v_body).astype(float)
    else:
        th = omega * times
        s = np.sin(th)
        c = np.cos(th)
        vx, vy = float(v_body[0]), float(v_body[1])
        x = (vx / omega) * s - (vy / omega) * (1.0 - c)
        y = (vx / omega) * (1.0 - c) + (vy / omega) * s
        positions = np.stack([x, y], axis=1)

    # Contact point pose: cp_world = p + R(theta) @ cp_body
    cp_x0, cp_y0 = float(cp_body[0]), float(cp_body[1])
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    cp_world_rel = np.stack(
        [cos_th * cp_x0 - sin_th * cp_y0, sin_th * cp_x0 + cos_th * cp_y0],
        axis=1,
    )
    cp_world = positions + cp_world_rel

    # Contact-point velocity: v_cp_world = R(theta) @ v_cp_body (v_cp_body constant)
    v_cp_body = v_body + omega * np.array([-cp_body[1], cp_body[0]], dtype=float)
    vcp_x0, vcp_y0 = float(v_cp_body[0]), float(v_cp_body[1])
    v_cp_world = np.stack(
        [cos_th * vcp_x0 - sin_th * vcp_y0, sin_th * vcp_x0 + cos_th * vcp_y0],
        axis=1,
    )

    # Outward normal
    n_x0, n_y0 = float(n_out_body[0]), float(n_out_body[1])
    n_out_world = np.stack(
        [cos_th * n_x0 - sin_th * n_y0, sin_th * n_x0 + cos_th * n_y0],
        axis=1,
    )

    return {
        "times": times,
        "positions": positions,
        "thetas": theta,
        "cp_world": cp_world,
        "v_cp_world": v_cp_world,
        "n_out_world": n_out_world,
    }


def load_shape(name: str):
    obj_file = OBJ_FILE_MAP.get(name)
    if obj_file is None:
        raise ValueError(f"Unknown shape '{name}'. Choose from {list(OBJ_FILE_MAP)}")

    connected_here = False
    if not pyb.isConnected():
        pyb.connect(pyb.DIRECT)
        connected_here = True

    body_uid = None
    try:
        generic, body_uid = obj_to_generic(
            obj_path=obj_file,
            shape_name=name,
            position=(0, 0, 0.2),
            orientation=0.0,
            mass=1.0,
            lateral_friction=0.8,
            blind_test=True,
        )
        cpp = ContactPointParameterization(generic)
        return generic, cpp
    finally:
        if body_uid is not None and pyb.isConnected():
            pyb.removeBody(body_uid)
        if connected_here and pyb.isConnected():
            pyb.disconnect()


def _draw_shape(ax, verts: np.ndarray, pos: np.ndarray, theta: float, centroid: np.ndarray,
                color="tab:gray", alpha_fill=0.25, lw: float = 1.0):
    # verts are polygon coords in object frame; centroid is used to rotate about the CoM.
    import matplotlib.pyplot as plt  # noqa: F401  (ensures backend init for some setups)
    from matplotlib.patches import Polygon as MplPolygon

    R = rot2d(theta)
    pts = (R @ (verts - centroid).T).T + pos
    ax.add_patch(
        MplPolygon(
            pts,
            closed=True,
            fill=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha_fill,
            linewidth=lw,
        )
    )


def _compute_dd_solutions(
    v_cp_world0: np.ndarray,
    omega_obj: float,
    R_r: float,
    phi0: float,
) -> list[dict]:
    # Copied (mathematically) from test_matching_velo_nonconvex.py
    a = v_cp_world0[0] + omega_obj * R_r * np.sin(phi0)
    b = v_cp_world0[1] - omega_obj * R_r * np.cos(phi0)
    speed = float(np.hypot(a, b))
    angle = float(np.arctan2(b, a))

    solutions = []
    for sign, label in [(+1, "forward"), (-1, "backward")]:
        v_r = float(sign * speed)
        zeta0 = angle if sign > 0 else float(_wrap_angle(angle + np.pi))
        alpha = float(_wrap_angle(phi0 - zeta0))
        solutions.append(dict(zeta0=zeta0, v_r=v_r, omega_r=omega_obj, alpha=alpha, label=label))
    return solutions


def _propagate_dd_constant(
    v_r: float,
    omega_r: float,
    zeta0: float,
    alpha: float,
    R_r: float,
    cp_world0: np.ndarray,
    T: float,
    dt: float,
) -> dict[str, np.ndarray]:
    n_steps = int(T / dt) + 1 if T > 0 else 1
    n_steps = max(n_steps, 2)
    times = np.linspace(0.0, T, n_steps, dtype=float)

    # Robot geometry: contact point lies on a disc of radius R_r at angle phi = zeta + alpha.
    phi0 = float(zeta0 + alpha)
    center0 = cp_world0 - R_r * np.array([np.cos(phi0), np.sin(phi0)], dtype=float)

    headings = zeta0 + omega_r * times  # zeta(t)
    robot_centers = np.zeros((n_steps, 2), dtype=float)
    robot_cp = np.zeros((n_steps, 2), dtype=float)

    eps = 1e-12
    if abs(omega_r) < eps:
        # Straight-line limit: zeta is constant.
        headings[:] = float(zeta0)
        robot_centers = center0 + (v_r * times)[:, None] * np.array(
            [np.cos(float(zeta0)), np.sin(float(zeta0))], dtype=float
        )[None, :]
    else:
        # Exact unicycle integration for constant (v_r, omega_r).
        # center(t) = center0 + (v_r/omega_r) * [sin(zeta(t)) - sin(zeta0), cos(zeta0) - cos(zeta(t))]
        k = v_r / omega_r
        z0 = float(zeta0)
        sin_z = np.sin(headings)
        cos_z = np.cos(headings)
        robot_centers[:, 0] = center0[0] + k * (sin_z - np.sin(z0))
        robot_centers[:, 1] = center0[1] + k * (np.cos(z0) - cos_z)

    phi = headings + float(alpha)
    robot_cp = robot_centers + R_r * np.stack([np.cos(phi), np.sin(phi)], axis=1)

    return {
        "times": times,
        "robot_centers": robot_centers,
        "robot_cp": robot_cp,
        "v_r": float(v_r),
        "omega_r": float(omega_r),
        "zeta0": float(zeta0),
        "alpha": float(alpha),
        "headings": headings.astype(float),
    }


def plot_2d(
    *,
    generic,
    verts: np.ndarray,
    centroid: np.ndarray,
    traj: dict[str, np.ndarray],
    x_end: float,
    y_end: float,
    theta_end: float,
    shape: str,
    t_param: float,
    v_body: np.ndarray,
    omega: float,
    T: float,
    dt: float,
    robot_radius: float,
    plot_dd_robot: bool,
    plot_dd_both: bool,
    plot_dd_middle: bool,
    plot_dd_n_snap: int = 8,
    plot_holo_robot: bool,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
):
    import matplotlib.pyplot as plt

    from matplotlib.patches import Circle

    times = traj["times"]
    positions = traj["positions"]
    cp_world = traj["cp_world"]
    thetas = traj["thetas"]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")

    # Trajectories
    ax.plot(positions[:, 0], positions[:, 1], "b--", lw=1.4, label="Object CoM")
    ax.plot(cp_world[:, 0], cp_world[:, 1], "g-", lw=2.0, label="Contact point")
    ax.plot([positions[0, 0], positions[-1, 0]], [positions[0, 1], positions[-1, 1]], "k:", lw=1.0)

    # Object at start and end
    _draw_shape(
        ax,
        verts,
        pos=positions[0],
        theta=float(thetas[0]),
        centroid=centroid,
        color="tab:orange",
        alpha_fill=0.25,
    )
    _draw_shape(
        ax,
        verts,
        pos=positions[-1],
        theta=float(theta_end),
        centroid=centroid,
        color="tab:gray",
        alpha_fill=0.25,
    )
    ax.scatter([positions[0, 0], positions[-1, 0]], [positions[0, 1], positions[-1, 1]], c=["blue", "blue"], s=20)
    ax.scatter([cp_world[0, 0], cp_world[-1, 0]], [cp_world[0, 1], cp_world[-1, 1]], c=["green", "green"], s=20)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    v_norm = float(np.linalg.norm(v_body))
    vx, vy = float(v_body[0]), float(v_body[1])
    ax.set_title(
        "Motion Primitive (inverse SE2->constant body twist)\n"
        f"shape={shape}, t_param={t_param:.2f}\n"
        f"end=(x={x_end:.3f}, y={y_end:.3f}, theta={theta_end:.3f})\n"
        f"v_body=[{vx:.3f}, {vy:.3f}] m/s (||v_body||={v_norm:.3f}), "
        f"omega={omega:.6f} rad/s, T={T:.4f} s"
    )

    cp_body = traj.get("cp_body")
    n_out_body = traj.get("n_out_body")

    # Optional holonomic plot (useful for debugging DD visualization)
    if plot_holo_robot:
        if cp_body is None or n_out_body is None:
            raise RuntimeError("Internal error: cp_body/n_out_body not available for robot plotting.")

        heading_body_offset = float(np.arctan2(-n_out_body[1], -n_out_body[0]))  # inward direction at theta=0
        center_offset_body = cp_body + robot_radius * n_out_body

        n_steps = len(times)
        robot_centers = np.zeros_like(positions)
        robot_cp = np.zeros_like(cp_world)
        headings_h = np.zeros(n_steps, dtype=float)

        for i in range(len(times)):
            R = rot2d(thetas[i])
            center = positions[i] + R @ center_offset_body
            zeta = thetas[i] + heading_body_offset
            robot_centers[i] = center
            robot_cp[i] = center + robot_radius * np.array([np.cos(zeta), np.sin(zeta)], dtype=float)
            headings_h[i] = zeta

        ax.plot(robot_cp[:, 0], robot_cp[:, 1], color="tab:blue", lw=2.0, alpha=0.9, label="Holonomic robot cp")
        ax.plot(robot_centers[:, 0], robot_centers[:, 1], color="tab:blue", lw=1.2, ls=":", alpha=0.7, label="Holonomic center")

        # Snapshots along the path
        n_snap = 8
        step = max(1, len(times) // n_snap)
        for idx in range(0, len(times), step):
            center = robot_centers[idx]
            heading = float(headings_h[idx])
            ax.add_patch(
                Circle(center, robot_radius, fill=False, edgecolor="tab:blue", lw=1.0, alpha=0.7)
            )
            tip = center + robot_radius * np.array([np.cos(heading), np.sin(heading)], dtype=float)
            ax.plot([center[0], tip[0]], [center[1], tip[1]], color="tab:blue", lw=1.0, alpha=0.7)
            ax.plot(robot_cp[idx, 0], robot_cp[idx, 1], "o", color="tab:blue", ms=3, alpha=0.65)
            ax.plot(cp_world[idx, 0], cp_world[idx, 1], "x", color="black", ms=4, alpha=0.7)

        # Ensure the final pose at t=T is always drawn (the step-based loop can skip it).
        last_idx = len(times) - 1
        if last_idx % step != 0:
            center = robot_centers[last_idx]
            heading = float(headings_h[last_idx])
            ax.add_patch(
                Circle(center, robot_radius, fill=False, edgecolor="tab:blue", lw=1.0, alpha=0.7)
            )
            tip = center + robot_radius * np.array([np.cos(heading), np.sin(heading)], dtype=float)
            ax.plot([center[0], tip[0]], [center[1], tip[1]], color="tab:blue", lw=1.0, alpha=0.7)
            ax.plot(robot_cp[last_idx, 0], robot_cp[last_idx, 1], "o", color="tab:blue", ms=3, alpha=0.65)
            ax.plot(cp_world[last_idx, 0], cp_world[last_idx, 1], "x", color="black", ms=4, alpha=0.7)

    # Optional diff-drive plot
    if plot_dd_robot:
        # With theta0=0, the contact velocity at t=0 is directly the world velocity of v_cp_body.
        if cp_body is None or n_out_body is None:
            # The plotting function is called with only trajectories; reconstruct needed quantities.
            # For consistency, recompute from the contact info by using the same cp_world[0].
            # This path should not happen for our script.
            raise RuntimeError("Internal error: cp_body/n_out_body not available for robot plotting.")

        # v_cp_body constant -> v_cp_world0 = v_cp_body rotated by R(theta0)=I.
        v_cp_world0 = (v_body + omega * np.array([-cp_body[1], cp_body[0]], dtype=float)).astype(float)
        phi0 = float(np.arctan2(-n_out_body[1], -n_out_body[0]))  # inward direction at theta0=0

        sols = _compute_dd_solutions(v_cp_world0, omega, robot_radius, phi0)

        to_plot = sols if plot_dd_both else [sols[0]]
        mid_idx = len(times) // 2 if plot_dd_middle else None
        if plot_dd_middle and mid_idx is not None:
            _draw_shape(
                ax,
                verts,
                pos=positions[mid_idx],
                theta=float(thetas[mid_idx]),
                centroid=centroid,
                color="tab:purple",
                alpha_fill=0.20,
            )
        for s in to_plot:
            res = _propagate_dd_constant(
                s["v_r"],
                s["omega_r"],
                s["zeta0"],
                s["alpha"],
                robot_radius,
                cp_world0=cp_world[0],
                T=float(times[-1]),
                dt=dt,
            )
            label = f"DD contact ({s['label']})"
            ax.plot(res["robot_cp"][:, 0], res["robot_cp"][:, 1], lw=1.5, ls="-" if s["label"] == "forward" else "--",
                    color="tab:red", alpha=0.95, label=label)
            ax.plot(res["robot_centers"][:, 0], res["robot_centers"][:, 1], lw=1.0, ls=":", color="tab:red", alpha=0.7,
                    label=f"DD center ({s['label']})")

            # Draw multiple robot snapshots along the path (including the start pose).
            cp_err = float(np.max(np.linalg.norm(res["robot_cp"] - cp_world, axis=1)))
            print(f"DD-{s['label']}: max |robot_cp - object_cp| = {cp_err*1e3:.3f} mm")
            n_steps = len(times)

            if plot_dd_middle:
                # With --plot_dd_middle, only show robot instances at start/mid/final.
                snap_indices = sorted(set([0, int(mid_idx), n_steps - 1]))
            else:
                # Otherwise, show a sparse set of snapshots along the path.
                n_snap = int(plot_dd_n_snap)
                step = max(1, n_steps // n_snap)
                snap_indices = list(range(0, n_steps, step))
                # Ensure the final pose at t=T is always drawn.
                last_idx = n_steps - 1
                if last_idx not in snap_indices:
                    snap_indices.append(last_idx)

            for idx in snap_indices:
                center = res["robot_centers"][idx]
                heading = float(res["headings"][idx])
                ax.add_patch(Circle(center, robot_radius, fill=False, edgecolor="tab:red", lw=1.0, alpha=0.6, zorder=5))
                tip = center + robot_radius * np.array([np.cos(heading), np.sin(heading)], dtype=float)
                ax.plot([center[0], tip[0]], [center[1], tip[1]], color="tab:red", lw=1.0, alpha=0.6, zorder=6)
                # Mark the robot contact point for spatial intuition.
                ax.plot(res["robot_cp"][idx, 0], res["robot_cp"][idx, 1], "o", color="tab:red", ms=3, alpha=0.7, zorder=6)
                ax.plot(cp_world[idx, 0], cp_world[idx, 1], "x", color="black", ms=4, alpha=0.7, zorder=7)

    ax.legend(fontsize=9, loc="best")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved figure to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inverse constant body-twist motion primitive for object SE2 end pose."
    )
    p.add_argument("--shape", default="pi", choices=list(OBJ_FILE_MAP))
    p.add_argument("--t_param", type=float, default=0.25, help="Contact point boundary parameter t in [0,1]")

    p.add_argument("--x_end", type=float, required=True)
    p.add_argument("--y_end", type=float, required=True)
    p.add_argument("--theta_end", type=float, required=True, help="End orientation (rad) in SE2.")

    p.add_argument("--v_speed", type=float, default=1.0, help="Fixed magnitude ||v_body|| in m/s.")
    p.add_argument(
        "--pure_spin_duration",
        type=float,
        default=1.0,
        help="Duration T used when x_end,y_end~0 and theta_end!=0 (pure spin case).",
    )
    p.add_argument("--dt", type=float, default=0.01, help="Time step for plotting.")
    p.add_argument("--duration_fallback_dt", type=float, default=0.01, help="(unused) compatibility knob.")

    p.add_argument("--robot_radius", type=float, default=0.06)
    p.add_argument("--plot_dd_robot", action="store_true", help="Also plot constant-command diff-drive solution.")
    p.add_argument("--plot_dd_both", action="store_true", help="If plotting DD, show both forward and backward solutions.")
    p.add_argument("--plot_dd_middle", action="store_true", help="Also plot object+DD robot at mid-time.")
    p.add_argument("--plot_dd_n_snap", type=int, default=8, help="Number of robot snapshots when plotting DD.")
    p.add_argument("--plot_holo_robot", action="store_true", help="Also plot holonomic velocity-matched solution for debugging.")

    p.add_argument("--save", type=str, default=None, help="Save filename under /tmp/motion_primitive.")
    p.add_argument("--silent", action="store_true", help="Use non-interactive backend and don't show window.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib

    if args.silent:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt  # noqa: F401

    print("=" * 80)
    print("Motion Primitive Inverse Solver (SE2 end pose -> constant body twist)")
    print("=" * 80)
    print(f"Shape={args.shape}, t_param={args.t_param}")
    print(f"Target end pose: x_end={args.x_end:.4f}, y_end={args.y_end:.4f}, theta_end={args.theta_end:.4f} rad")
    print(f"Fixed speed magnitude: v_speed={args.v_speed:.4f} m/s")

    # Load object and contact geometry (exactly as in test_matching_velo_nonconvex.py)
    generic, cpp = load_shape(args.shape)
    centroid = np.array([generic.geometry.centroid.x, generic.geometry.centroid.y], dtype=float)
    verts = np.array(generic.geometry.exterior.coords, dtype=float)

    cp_info = cpp.get_contact_info(args.t_param)
    cp_body = np.array(cp_info["point"], dtype=float) - centroid
    n_out_body = np.array(cp_info["normal_outward"], dtype=float)

    # Inverse kinematics: end pose -> (v_body, omega, T)
    v_body, omega, T = solve_constant_body_twist_from_SE2(
        args.x_end,
        args.y_end,
        args.theta_end,
        v_speed=args.v_speed,
        pure_spin_duration=args.pure_spin_duration,
    )
    v_norm = float(np.linalg.norm(v_body))

    print("\nSolved constant body twist:")
    if v_norm > 1e-12:
        v_dir = v_body / v_norm
        print(f"  v_body unit direction = [{v_dir[0]:+.6f}, {v_dir[1]:+.6f}]")
    else:
        print("  v_body unit direction = [undefined: pure spin v_body=0]")
    print(f"  omega                = {omega:+.6f} rad/s")
    print(f"  duration T          = {T:.6f} s")

    # Forward propagate (analytic) and sanity-check the end pose.
    traj = propagate_object_pose_constant_body_twist(
        v_body,
        omega,
        T,
        dt=args.dt,
        cp_body=cp_body,
        n_out_body=n_out_body,
        theta0=0.0,
    )

    pos_pred = traj["positions"][-1]
    theta_pred = float(traj["thetas"][-1])

    pos_err = float(np.linalg.norm(pos_pred - np.array([args.x_end, args.y_end], dtype=float)))
    theta_err = _wrap_angle(theta_pred - float(args.theta_end))

    print("\nSanity check (forward propagation):")
    print(f"  pos error  = {pos_err:.3e} m")
    print(f"  theta error= {theta_err:.3e} rad")
    if pos_err > 1e-5 or abs(theta_err) > 1e-6:
        raise RuntimeError("Sanity check failed: predicted end pose does not match target.")

    # Attach cp_body/n_out_body for optional DD plotting.
    traj["cp_body"] = cp_body
    traj["n_out_body"] = n_out_body

    # Save figure name
    def fmt_f(x: float) -> str:
        s = f"{x:+.3f}".replace("-", "m").replace("+", "p")
        return s.replace(".", "p")

    save_path = None
    if args.save is not None:
        save_path = SAVE_DIR / args.save
    else:
        save_path = SAVE_DIR / f"{args.shape}_t{args.t_param:.2f}_x{fmt_f(args.x_end)}_y{fmt_f(args.y_end)}_th{fmt_f(args.theta_end)}.png"

    plot_2d(
        generic=generic,
        verts=verts,
        centroid=centroid,
        traj=traj,
        x_end=args.x_end,
        y_end=args.y_end,
        theta_end=args.theta_end,
        shape=args.shape,
        t_param=args.t_param,
        v_body=v_body,
        omega=omega,
        T=T,
        dt=args.dt,
        robot_radius=args.robot_radius,
        plot_dd_robot=args.plot_dd_robot,
        plot_dd_both=args.plot_dd_both,
        plot_dd_middle=args.plot_dd_middle,
        plot_dd_n_snap=args.plot_dd_n_snap,
        plot_holo_robot=args.plot_holo_robot,
        save_path=save_path,
        show_plot=not args.silent,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()