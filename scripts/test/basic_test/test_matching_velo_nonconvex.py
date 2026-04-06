#!/usr/bin/env python3
"""
Velocity-matching test for robot-object contact maintenance.

Under constant body-frame velocity the object traces a straight line (omega=0)
or an arc (omega!=0).  By differential flatness, a diff-drive robot under
constant (v_r, omega_r) also traces a line/arc.  Therefore a *specific* initial
heading zeta_0 exists such that a single pair of constant wheel commands makes
the robot's contact-point trajectory coincide with the object's for the entire
segment.

This script:
  1. Propagates the object (pure kinematics, constant body-frame velocity).
  2. Analytically computes the two constant-velocity diff-drive solutions
     (forward and backward) from the velocity-matching equation.
  3. Propagates each robot with truly constant commands and verifies that
     the contact-point trajectories match.
  4. Shows the holonomic case for comparison (trivial: translation matches
     contact velocity, heading is reactive).
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pyb
import rospkg
from matplotlib.patches import Circle, Polygon as MplPolygon

rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from contact_maintain.object_bridge import obj_to_generic
from object_utils import ContactPointParameterization

# ---------------------------------------------------------------------------
OBJ_FILE_MAP = {
    "right_triangle": "right_triangle.obj",
    "bolt": "bolt.obj",
    "pi": "pi.obj",
    "root": "root.obj",
    "rect": "rect.obj",
    "hourglass": "hourglass.obj",
    "meteor": "meteor.obj",
}

SAVE_DIR = Path("/tmp/matching_velo")


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _wrap(a: float) -> float:
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ObjectTrajectory:
    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    thetas: np.ndarray = field(default_factory=lambda: np.empty(0))
    cp_world: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    v_cp_world: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    n_out_world: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))


@dataclass
class RobotResult:
    label: str = ""
    robot_centers: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    robot_cp: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    v_robot_cp: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    headings: np.ndarray = field(default_factory=lambda: np.empty(0))
    v_r: np.ndarray = field(default_factory=lambda: np.empty(0))
    omega_r: np.ndarray = field(default_factory=lambda: np.empty(0))
    alpha: float = 0.0
    zeta0: float = 0.0


# ---------------------------------------------------------------------------
# 1. Shape loading
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Object propagation
# ---------------------------------------------------------------------------
def propagate_object(
    v_body: np.ndarray, omega: float,
    T: float, dt: float,
    x0: float, y0: float, theta0: float,
    cp_body: np.ndarray, n_out_body: np.ndarray,
) -> ObjectTrajectory:
    n_steps = int(T / dt) + 1
    times = np.linspace(0, T, n_steps)
    positions = np.zeros((n_steps, 2))
    thetas = np.zeros(n_steps)
    cp_world = np.zeros((n_steps, 2))
    v_cp_world = np.zeros((n_steps, 2))
    n_out_world = np.zeros((n_steps, 2))

    pos = np.array([x0, y0], dtype=float)
    theta = theta0

    for i in range(n_steps):
        R = rot2d(theta)
        positions[i] = pos
        thetas[i] = theta
        cp_world[i] = pos + R @ cp_body
        v_cp_world[i] = R @ v_body + omega * R @ np.array([-cp_body[1], cp_body[0]])
        n_out_world[i] = R @ n_out_body
        if i < n_steps - 1:
            pos = pos + R @ v_body * dt
            theta = theta + omega * dt

    return ObjectTrajectory(
        times=times, positions=positions, thetas=thetas,
        cp_world=cp_world, v_cp_world=v_cp_world, n_out_world=n_out_world,
    )


# ---------------------------------------------------------------------------
# 3. Holonomic matching
# ---------------------------------------------------------------------------
def holonomic_matching(
    obj: ObjectTrajectory,
    R_r: float,
    cp_body: np.ndarray,
    n_out_body: np.ndarray,
    v_body: np.ndarray,
    omega_obj: float,
) -> RobotResult:
    """
    Holonomic robot velocity matching.

    The holonomic robot is superior to diff-drive: its translational velocity
    directly tracks the contact-point velocity, while omega_r only maintains
    the heading toward the contact point.

    Because the robot has 3 DOF for 2 velocity constraints, the solution is
    fully analytical and exact:
      - center(t) = obj_pos(t) + R(theta(t)) @ center_offset_body
      - heading(t) = theta(t) + heading_body_offset
      - omega_r = omega_obj  (constant — heading co-rotates with object)
      - v_center in body frame is constant (differential flatness)
    """
    N = len(obj.times)

    # Constant body-frame quantities
    center_offset_body = cp_body + R_r * n_out_body
    heading_body_offset = np.arctan2(-n_out_body[1], -n_out_body[0])
    v_center_body = (v_body
                     + omega_obj * np.array([-center_offset_body[1],
                                              center_offset_body[0]]))

    centers = np.zeros((N, 2))
    cp_robot = np.zeros((N, 2))
    v_robot_cp = np.zeros((N, 2))
    headings = np.zeros(N)

    for i in range(N):
        R = rot2d(obj.thetas[i])

        # Exact positions — no Euler drift
        centers[i] = obj.positions[i] + R @ center_offset_body
        zeta = obj.thetas[i] + heading_body_offset
        headings[i] = zeta

        r = R_r * np.array([np.cos(zeta), np.sin(zeta)])
        cp_robot[i] = centers[i] + r

        # Velocities
        v_center_world = R @ v_center_body
        v_rot = omega_obj * np.array([-r[1], r[0]])
        v_robot_cp[i] = v_center_world + v_rot

    return RobotResult(
        label="Holonomic", robot_centers=centers, robot_cp=cp_robot,
        v_robot_cp=v_robot_cp, headings=headings,
        v_r=np.full(N, np.linalg.norm(v_center_body)),
        omega_r=np.full(N, omega_obj),
        alpha=0.0, zeta0=headings[0],
    )


# ---------------------------------------------------------------------------
# 4. Diff-drive: analytical constant-velocity solution
# ---------------------------------------------------------------------------
def compute_dd_solutions(
    v_cp_world0: np.ndarray,
    omega_obj: float,
    R_r: float,
    phi0: float,
):
    """
    Analytical constant-velocity solutions for diff-drive velocity matching.

    From the matching equation at t=0 with omega_r = omega_obj:
        v_r cos(zeta0) = v_cp0_x + omega_obj R_r sin(phi0)   =: a
        v_r sin(zeta0) = v_cp0_y - omega_obj R_r cos(phi0)   =: b

    Two solutions exist: forward (v_r > 0) and backward (v_r < 0).
    """
    a = v_cp_world0[0] + omega_obj * R_r * np.sin(phi0)
    b = v_cp_world0[1] - omega_obj * R_r * np.cos(phi0)
    speed = np.hypot(a, b)
    angle = np.arctan2(b, a)

    solutions = []
    for sign, label in [(+1, "forward"), (-1, "backward")]:
        v_r = sign * speed
        zeta0 = angle if sign > 0 else _wrap(angle + np.pi)
        alpha = _wrap(phi0 - zeta0)
        solutions.append(dict(
            zeta0=zeta0, v_r=v_r, omega_r=omega_obj, alpha=alpha, label=label,
        ))
    return solutions


def propagate_dd_constant(
    v_r: float, omega_r: float, zeta0: float, alpha: float,
    R_r: float, cp_world0: np.ndarray, T: float, dt: float,
) -> RobotResult:
    """Propagate diff-drive robot with truly constant (v_r, omega_r)."""
    n_steps = int(T / dt) + 1
    times = np.linspace(0, T, n_steps)

    phi0 = zeta0 + alpha
    center = cp_world0 - R_r * np.array([np.cos(phi0), np.sin(phi0)])
    zeta = zeta0

    centers = np.zeros((n_steps, 2))
    cp_robot = np.zeros((n_steps, 2))
    v_robot_cp = np.zeros((n_steps, 2))
    headings = np.zeros(n_steps)

    for i in range(n_steps):
        phi = zeta + alpha
        centers[i] = center
        headings[i] = zeta
        cp_robot[i] = center + R_r * np.array([np.cos(phi), np.sin(phi)])

        r = R_r * np.array([np.cos(phi), np.sin(phi)])
        v_base = np.array([v_r * np.cos(zeta), v_r * np.sin(zeta)])
        v_rot = omega_r * np.array([-r[1], r[0]])
        v_robot_cp[i] = v_base + v_rot

        if i < n_steps - 1:
            center = center + v_base * dt
            zeta = zeta + omega_r * dt

    return RobotResult(
        label="", robot_centers=centers, robot_cp=cp_robot,
        v_robot_cp=v_robot_cp, headings=headings,
        v_r=np.full(n_steps, v_r), omega_r=np.full(n_steps, omega_r),
        alpha=alpha, zeta0=zeta0,
    )


# ---------------------------------------------------------------------------
# 5. Plotting helpers
# ---------------------------------------------------------------------------
def _draw_shape(ax, verts, pos, theta, centroid, color="k", alpha_fill=0.15, lw=1.0):
    """Draw rotated polygon.  *centroid* must be the Shapely area centroid."""
    R = rot2d(theta)
    pts = (R @ (verts - centroid).T).T + pos
    ax.add_patch(MplPolygon(pts, closed=True, fill=True,
                            facecolor=color, edgecolor=color,
                            alpha=alpha_fill, linewidth=lw))


def _draw_robot(ax, center, heading, R_r, cp_on_robot=None,
                color="tab:blue", a=0.20, emphasize=False):
    ax.add_patch(Circle(center, R_r, fill=True, facecolor=color,
                        edgecolor=color, alpha=a, linewidth=0.8))
    tip = center + R_r * np.array([np.cos(heading), np.sin(heading)])
    ax.plot([center[0], tip[0]], [center[1], tip[1]], color=color, lw=1.2)
    if cp_on_robot is not None:
        ax.plot(cp_on_robot[0], cp_on_robot[1], "o", color=color,
                ms=4, mec="k", mew=0.4, zorder=5)
    if emphasize:
        ax.add_patch(Circle(center, R_r * 1.05, fill=False, edgecolor="k",
                            alpha=0.9, linewidth=1.4, linestyle="--"))


def _spatial_panel(ax, title, obj, res, R_r, color, n_snap=8):
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.plot(obj.cp_world[:, 0], obj.cp_world[:, 1], "r--", lw=1.0,
            alpha=0.5, label="Object cp")
    ax.plot(res.robot_cp[:, 0], res.robot_cp[:, 1], "-", color=color,
            lw=1.5, label="Robot cp")
    ax.plot(res.robot_centers[:, 0], res.robot_centers[:, 1], ":",
            color=color, lw=0.8, label="Robot center")
    step = max(1, len(obj.times) // n_snap)
    for idx in range(0, len(obj.times), step):
        _draw_robot(ax, res.robot_centers[idx], res.headings[idx], R_r,
                    cp_on_robot=res.robot_cp[idx], color=color, a=0.18)
        ax.plot(obj.cp_world[idx, 0], obj.cp_world[idx, 1], "r+",
                ms=8, mew=1.5, zorder=5)
    # Highlight t=0 pose for readability.
    _draw_robot(
        ax, res.robot_centers[0], res.headings[0], R_r,
        cp_on_robot=res.robot_cp[0], color="goldenrod", a=0.45, emphasize=True
    )
    ax.legend(fontsize=6, loc="best")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _commands_panel(ax, t, res, title, color, vr_label):
    ax.set_title(title, fontsize=10)
    ax.plot(t, res.v_r, color=color, lw=1.5, label=vr_label)
    ax.axhline(0, color="gray", ls=":", lw=0.6)
    ax2 = ax.twinx()
    ax2.plot(t, res.omega_r, color=color, lw=1.0, ls=":",
             label=f"omega_r = {res.omega_r[0]:.4f} rad/s")
    ax2.set_ylabel("omega_r [rad/s]")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("v_r [m/s]")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="best")


# ---------------------------------------------------------------------------
# 6. Main plot
# ---------------------------------------------------------------------------
def plot_results(
    obj: ObjectTrajectory,
    holo: RobotResult,
    dd_fwd: RobotResult,
    dd_bwd: Optional[RobotResult],
    shape_verts: np.ndarray,
    shape_centroid: np.ndarray,
    R_r: float,
    object_title: str,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
):
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.28)

    # ---- 1. Object trajectory ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(object_title, fontsize=9)
    ax.set_aspect("equal")
    _draw_shape(ax, shape_verts, obj.positions[0], obj.thetas[0], shape_centroid,
                color="tab:gray", alpha_fill=0.30)
    _draw_shape(ax, shape_verts, obj.positions[-1], obj.thetas[-1], shape_centroid,
                color="tab:orange", alpha_fill=0.30)
    ax.plot(obj.positions[:, 0], obj.positions[:, 1], "k-", lw=1.5, label="CoM")
    ax.plot(obj.cp_world[:, 0], obj.cp_world[:, 1], "r-", lw=1.5, label="Contact pt")
    ax.plot(obj.cp_world[0, 0], obj.cp_world[0, 1], "ro", ms=6)
    ax.plot(obj.cp_world[-1, 0], obj.cp_world[-1, 1], "rs", ms=6)
    step = max(1, len(obj.times) // 15)
    s = 0.02
    ax.quiver(obj.cp_world[::step, 0], obj.cp_world[::step, 1],
              obj.n_out_world[::step, 0] * s, obj.n_out_world[::step, 1] * s,
              angles="xy", scale_units="xy", scale=1, color="green",
              width=0.003, label="Outward normal")
    ax.legend(fontsize=7)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # ---- 2. Position tracking error  (top-right) ----
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Contact-Point Position Error (should be ~0)", fontsize=10)
    err_holo = np.linalg.norm(holo.robot_cp - obj.cp_world, axis=1)
    err_fwd = np.linalg.norm(dd_fwd.robot_cp - obj.cp_world, axis=1)
    ax.plot(obj.times, err_holo * 1e3, "b-", lw=1.2, label="Holonomic")
    ax.plot(obj.times, err_fwd * 1e3, color="tab:green", lw=1.2, label="DD forward")
    if dd_bwd is not None:
        err_bwd = np.linalg.norm(dd_bwd.robot_cp - obj.cp_world, axis=1)
        ax.plot(obj.times, err_bwd * 1e3, color="tab:red", lw=1.2, ls="--", label="DD backward")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("||robot_cp - obj_cp|| [mm]")
    ax.legend(fontsize=7)

    # ---- 3. Holonomic (spatial) ----
    ax = fig.add_subplot(gs[1, 0])
    _spatial_panel(ax, "Holonomic Robot (heading reactive, translation matches v_cp)",
                   obj, holo, R_r, "tab:blue")

    # ---- 4. DD forward (spatial) ----
    ax = fig.add_subplot(gs[1, 1])
    _spatial_panel(
        ax,
        f"DD Forward: zeta0={np.degrees(dd_fwd.zeta0):.1f} deg, "
        f"alpha={np.degrees(dd_fwd.alpha):.1f} deg, v_r={dd_fwd.v_r[0]:.4f}, "
        f"omega_r={dd_fwd.omega_r[0]:.4f}",
        obj, dd_fwd, R_r, "tab:green",
    )

    # ---- 5. Bottom-left panel ----
    ax = fig.add_subplot(gs[2, 0])
    if dd_bwd is not None:
        _spatial_panel(
            ax,
            f"DD Backward: zeta0={np.degrees(dd_bwd.zeta0):.1f} deg, "
            f"alpha={np.degrees(dd_bwd.alpha):.1f} deg, v_r={dd_bwd.v_r[0]:.4f}, "
            f"omega_r={dd_bwd.omega_r[0]:.4f}",
            obj, dd_bwd, R_r, "tab:red",
        )
    else:
        _commands_panel(
            ax, obj.times, holo,
            "Holonomic Commands (constant body-frame)", "tab:blue",
            f"v_r holo = {holo.v_r[0]:.4f} m/s",
        )

    # ---- 6. Bottom-right panel ----
    ax = fig.add_subplot(gs[2, 1])
    _commands_panel(
        ax, obj.times, dd_fwd,
        "DD Forward Commands (should be flat)", "tab:green",
        f"v_r fwd = {dd_fwd.v_r[0]:.4f} m/s",
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved figure to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7. CLI & main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Velocity-matching test: constant-velocity diff-drive solution",
    )
    p.add_argument("--shape", default="pi", choices=list(OBJ_FILE_MAP))
    p.add_argument("--t_param", type=float, default=0.25)
    p.add_argument("--vx_body", type=float, default=0.05)
    p.add_argument("--vy_body", type=float, default=0.0)
    p.add_argument("--omega", type=float, default=0.3,
                   help="Object angular velocity [rad/s] (0 => straight line)")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--robot_radius", type=float, default=0.06)
    p.add_argument("--save", type=str, default=None,
                   help="Save filename (placed in /tmp/matching_velo/)")
    p.add_argument("--silent", action="store_true",
                   help="Save figure without opening an interactive plot window")
    p.add_argument("--non_mirror", action="store_true",
                   help="Use only the forward diff-drive solution (hide backward mirror)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Velocity-Matching Test  (constant-velocity solutions)")
    print("=" * 60)

    # ---- Load shape ----
    print(f"\nShape: {args.shape}")
    generic, cpp = load_shape(args.shape)
    centroid = np.array([generic.geometry.centroid.x, generic.geometry.centroid.y])
    verts = np.array(generic.geometry.exterior.coords)

    cp_info = cpp.get_contact_info(args.t_param)
    cp_body = cp_info["point"] - centroid
    n_out_body = cp_info["normal_outward"]
    print(f"  cp_body = {cp_body},  n_out_body = {n_out_body}")

    # ---- Object propagation ----
    v_body = np.array([args.vx_body, args.vy_body])
    omega = args.omega
    theta0 = 0.0
    obj = propagate_object(v_body, omega, args.duration, args.dt,
                           0.0, 0.0, theta0, cp_body, n_out_body)

    path_type = "arc" if abs(omega) > 1e-9 else "straight line"
    print(f"\nObject: v_body={v_body}, omega={omega:.3f} => {path_type}")

    # Contact velocity in body frame (constant!)
    v_cp_body = v_body + omega * np.array([-cp_body[1], cp_body[0]])
    v_cp_world0 = rot2d(theta0) @ v_cp_body
    print(f"  v_cp (body, constant) = {v_cp_body}")
    print(f"  v_cp (world, t=0)     = {v_cp_world0}")

    R_r = args.robot_radius

    # ---- Holonomic ----
    print(f"\nHolonomic (R={R_r}m):")
    holo = holonomic_matching(obj, R_r, cp_body, n_out_body, v_body, omega)
    err = np.max(np.linalg.norm(holo.v_robot_cp - obj.v_cp_world, axis=1))
    print(f"  Max velocity error: {err:.2e} m/s")
    print(f"  Max |omega_r|:      {np.max(np.abs(holo.omega_r)):.4f} rad/s")

    # ---- DD analytical solutions ----
    n_out_w0 = obj.n_out_world[0]
    phi0 = np.arctan2(-n_out_w0[1], -n_out_w0[0])  # inward normal direction
    print(f"\nDiff-drive analytical solutions (phi0={np.degrees(phi0):.1f} deg):")

    sols = compute_dd_solutions(v_cp_world0, omega, R_r, phi0)
    dd_results = []
    for s in sols:
        if args.non_mirror and s["label"] != "forward":
            continue
        print(f"\n  [{s['label'].upper()}]")
        print(f"    zeta0  = {np.degrees(s['zeta0']):+8.2f} deg")
        print(f"    alpha  = {np.degrees(s['alpha']):+8.2f} deg")
        print(f"    v_r    = {s['v_r']:+.6f} m/s   (constant)")
        print(f"    omega_r= {s['omega_r']:+.6f} rad/s (= omega_obj, constant)")

        res = propagate_dd_constant(
            s["v_r"], s["omega_r"], s["zeta0"], s["alpha"],
            R_r, obj.cp_world[0], args.duration, args.dt,
        )
        res.label = f"DD {s['label']}"

        pos_err = np.linalg.norm(res.robot_cp - obj.cp_world, axis=1)
        vel_err = np.linalg.norm(res.v_robot_cp - obj.v_cp_world, axis=1)
        print(f"    Max position error: {np.max(pos_err)*1e3:.4f} mm  (Euler drift)")
        print(f"    Max velocity error: {np.max(vel_err):.2e} m/s")
        dd_results.append(res)

    if not dd_results:
        raise RuntimeError("No diff-drive solutions available to plot.")

    dd_fwd = dd_results[0]
    dd_bwd = dd_results[1] if len(dd_results) > 1 else None

    # ---- Plot ----
    save_name = args.save or f"{args.shape}_t{args.t_param:.2f}_w{args.omega:.2f}.png"
    save_path = SAVE_DIR / save_name
    object_title = (
        f"Object ({args.shape}): CoM & Contact-Point Trajectory\n"
        f"T={args.duration:.2f}s, dt={args.dt:.3f}s, "
        f"v_body=[{v_body[0]:.3f}, {v_body[1]:.3f}] m/s, "
        f"omega={omega:.3f} rad/s, t_param={args.t_param:.3f}"
    )
    plot_results(
        obj, holo, dd_fwd, dd_bwd, verts, centroid, R_r,
        object_title=object_title,
        save_path=save_path, show_plot=not args.silent
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
