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

Mode ``--mode alpha_scan`` skips trajectory propagation: it prints boundary
parameter ``t`` in [0,1), per-edge ``t`` intervals, a table of ``alpha_0``, and
(by default) a single spatial figure with the object centered and compact robot
solid heading vectors (``zeta_0`` / ``alpha``) and dashed robot vectors (center → contact) along each edge.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pyb
import rospkg
from matplotlib.lines import Line2D
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
def edge_global_t_ranges(cpp: ContactPointParameterization) -> List[Tuple[int, float, float]]:
    """
    Global boundary parameter t in [0, 1] is arc-length / total perimeter
    (see ContactPointParameterization.parameter_to_point).  Return each
    polyline segment's corresponding interval [t_lo, t_hi].
    """
    L = float(cpp.total_length)
    if L <= 0:
        return []
    out: List[Tuple[int, float, float]] = []
    for i in range(cpp.n_segments):
        t0 = cpp.cumulative_distances[i] / L
        t1 = cpp.cumulative_distances[i + 1] / L
        out.append((i, t0, t1))
    return out


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
                color="tab:blue", a=0.20):
    ax.add_patch(Circle(center, R_r, fill=True, facecolor=color,
                        edgecolor=color, alpha=a, linewidth=0.8))
    tip = center + R_r * np.array([np.cos(heading), np.sin(heading)])
    ax.plot([center[0], tip[0]], [center[1], tip[1]], color=color, lw=1.2)
    if cp_on_robot is not None:
        ax.plot(cp_on_robot[0], cp_on_robot[1], "o", color=color,
                ms=4, mec="k", mew=0.4, zorder=5)


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
                    cp_on_robot=res.robot_cp[idx], color=color, a=0.12)
        ax.plot(obj.cp_world[idx, 0], obj.cp_world[idx, 1], "r+",
                ms=8, mew=1.5, zorder=5)
    ax.legend(fontsize=6, loc="best")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


# ---------------------------------------------------------------------------
# 6. Main plot
# ---------------------------------------------------------------------------
def plot_results(
    obj: ObjectTrajectory,
    holo: RobotResult,
    dd_fwd: RobotResult,
    dd_bwd: RobotResult,
    shape_verts: np.ndarray,
    shape_centroid: np.ndarray,
    R_r: float,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
):
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.28)

    # ---- 1. Object trajectory ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Object: CoM & Contact-Point Trajectory", fontsize=10)
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

    # ---- 2. Constant commands  (v_r and omega_r vs time — should be flat) ----
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Diff-Drive Commands (should be flat = constant velocity)", fontsize=10)
    ax.plot(obj.times, dd_fwd.v_r, color="tab:green", lw=1.5,
            label=f"v_r fwd = {dd_fwd.v_r[0]:.4f} m/s")
    ax.plot(obj.times, dd_bwd.v_r, color="tab:red", lw=1.5, ls="--",
            label=f"v_r bwd = {dd_bwd.v_r[0]:.4f} m/s")
    ax.axhline(0, color="gray", ls=":", lw=0.6)

    ax2 = ax.twinx()
    ax2.plot(obj.times, dd_fwd.omega_r, color="tab:green", lw=1.0, ls=":",
             label=f"omega_r = {dd_fwd.omega_r[0]:.4f} rad/s")
    ax2.set_ylabel("omega_r [rad/s]")
    ax.set_xlabel("t [s]"); ax.set_ylabel("v_r [m/s]")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="best")

    # ---- 3. Holonomic (spatial) ----
    ax = fig.add_subplot(gs[1, 0])
    _spatial_panel(ax, "Holonomic Robot (heading reactive, translation matches v_cp)",
                   obj, holo, R_r, "tab:blue")

    # ---- 4. DD forward (spatial) ----
    ax = fig.add_subplot(gs[1, 1])
    _spatial_panel(
        ax,
        f"DD Forward: zeta0={np.degrees(dd_fwd.zeta0):.1f} deg, "
        f"alpha={np.degrees(dd_fwd.alpha):.1f} deg, v_r={dd_fwd.v_r[0]:.4f}",
        obj, dd_fwd, R_r, "tab:green",
    )

    # ---- 5. DD backward (spatial) ----
    ax = fig.add_subplot(gs[2, 0])
    _spatial_panel(
        ax,
        f"DD Backward: zeta0={np.degrees(dd_bwd.zeta0):.1f} deg, "
        f"alpha={np.degrees(dd_bwd.alpha):.1f} deg, v_r={dd_bwd.v_r[0]:.4f}",
        obj, dd_bwd, R_r, "tab:red",
    )

    # ---- 6. Position tracking error  (Euler integration drift only) ----
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("Contact-Point Position Error (should be ~0)", fontsize=10)
    err_holo = np.linalg.norm(holo.robot_cp - obj.cp_world, axis=1)
    err_fwd = np.linalg.norm(dd_fwd.robot_cp - obj.cp_world, axis=1)
    err_bwd = np.linalg.norm(dd_bwd.robot_cp - obj.cp_world, axis=1)
    ax.plot(obj.times, err_holo * 1e3, "b-", lw=1.2, label="Holonomic")
    ax.plot(obj.times, err_fwd * 1e3, color="tab:green", lw=1.2, label="DD forward")
    ax.plot(obj.times, err_bwd * 1e3, color="tab:red", lw=1.2, ls="--", label="DD backward")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("||robot_cp - obj_cp|| [mm]")
    ax.legend(fontsize=7)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved figure to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Alpha scan: alpha_0 vs boundary parameter t (+ optional spatial figure)
# ---------------------------------------------------------------------------
def collect_alpha_scan_rows(
    cpp: ContactPointParameterization,
    centroid: np.ndarray,
    v_body: np.ndarray,
    omega: float,
    R_r: float,
    theta0: float,
    n_t: int,
    which_solutions: str = "forward",
    obj_pos: Optional[np.ndarray] = None,
) -> Tuple[List[dict], List[Tuple[int, float, float]]]:
    """Sample boundary t and compute analytical DD solutions at t=0 geometry."""
    if n_t < 2:
        raise ValueError("n_t must be at least 2")
    if obj_pos is None:
        obj_pos = np.zeros(2)
    t_samples = np.linspace(0.0, 1.0, n_t, endpoint=False)
    ranges = edge_global_t_ranges(cpp)
    R0 = rot2d(theta0)
    branch = "backward" if which_solutions == "backward" else "forward"

    rows: List[dict] = []
    for t in t_samples:
        info = cpp.get_contact_info(float(t))
        cp_body = np.asarray(info["point"], dtype=float) - centroid
        n_out_body = np.asarray(info["normal_outward"], dtype=float)
        n_out_w0 = R0 @ n_out_body
        cp_world = obj_pos + R0 @ cp_body
        phi0 = float(np.arctan2(-n_out_w0[1], -n_out_w0[0]))
        robot_center = cp_world + R_r * n_out_w0
        v_cp_body = v_body + omega * np.array([-cp_body[1], cp_body[0]])
        v_cp_world0 = R0 @ v_cp_body
        sols = compute_dd_solutions(v_cp_world0, omega, R_r, phi0)
        by_label = {s["label"]: s for s in sols}
        s = by_label[branch]

        row = {
            "t": float(t),
            "edge": int(info["segment_index"]),
            "local_t": float(info["local_parameter"]),
            "phi0": phi0,
            "phi0_deg": float(np.degrees(phi0)),
            "cp_world": cp_world.copy(),
            "robot_center": robot_center.copy(),
            "zeta0": float(s["zeta0"]),
            "alpha": float(s["alpha"]),
            "alpha_deg": float(np.degrees(s["alpha"])),
            "zeta0_deg": float(np.degrees(s["zeta0"])),
            "v_r": float(s["v_r"]),
        }
        if which_solutions in ("forward", "both"):
            sf = by_label["forward"]
            row["alpha_fwd_deg"] = float(np.degrees(sf["alpha"]))
            row["zeta0_fwd_deg"] = float(np.degrees(sf["zeta0"]))
            row["v_r_fwd"] = float(sf["v_r"])
        if which_solutions in ("backward", "both"):
            sb = by_label["backward"]
            row["alpha_bwd_deg"] = float(np.degrees(sb["alpha"]))
            row["zeta0_bwd_deg"] = float(np.degrees(sb["zeta0"]))
            row["v_r_bwd"] = float(sb["v_r"])
        rows.append(row)
    return rows, ranges


def _alpha_scan_branch_key(which_solutions: str) -> str:
    return "alpha_bwd_deg" if which_solutions == "backward" else "alpha_fwd_deg"


def print_alpha_scan_table(
    rows: List[dict],
    ranges: List[Tuple[int, float, float]],
    which_solutions: str,
    cpp: ContactPointParameterization,
    v_body: np.ndarray,
    omega: float,
    R_r: float,
    theta0: float,
) -> None:
    """Print per-edge t ranges, sample table, and per-edge alpha span."""
    print("\n" + "=" * 72)
    print("Alpha_0 scan (global boundary parameter t, no trajectory propagation)")
    print("=" * 72)
    print(f"  v_body = {v_body},  omega = {omega:.6g} rad/s,  R_r = {R_r} m")
    print(f"  theta0 = {theta0:.6g} rad (world);  centroid offset for cp_body")
    print("\nGlobal t is arc_length / perimeter (see ContactPointParameterization).")
    print("Per-edge t intervals [t_lo, t_hi]:\n")
    for i, t0, t1 in ranges:
        sl = cpp.segment_lengths[i] if i < len(cpp.segment_lengths) else 0.0
        print(f"  edge {i:3d}:  t in [{t0:.6f}, {t1:.6f}]   segment_len = {sl:.6g} m")

    print("\nSample table (each row is one contact location on the boundary):\n")
    if which_solutions == "forward":
        hdr = f"{'t':>10} {'edge':>5} {'loc_t':>8} {'phi0_deg':>10} {'alpha_fwd_deg':>14} {'zeta0_fwd_deg':>15} {'v_r_fwd':>12}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(
                f"{r['t']:10.6f} {r['edge']:5d} {r['local_t']:8.4f} "
                f"{r['phi0_deg']:10.3f} {r['alpha_fwd_deg']:14.4f} "
                f"{r['zeta0_fwd_deg']:15.4f} {r['v_r_fwd']:12.6f}"
            )
    elif which_solutions == "backward":
        hdr = (
            f"{'t':>10} {'edge':>5} {'loc_t':>8} {'phi0_deg':>10} "
            f"{'alpha_bwd_deg':>14} {'zeta0_bwd_deg':>15} {'v_r_bwd':>12}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(
                f"{r['t']:10.6f} {r['edge']:5d} {r['local_t']:8.4f} "
                f"{r['phi0_deg']:10.3f} {r['alpha_bwd_deg']:14.4f} "
                f"{r['zeta0_bwd_deg']:15.4f} {r['v_r_bwd']:12.6f}"
            )
    else:
        hdr = (
            f"{'t':>10} {'edge':>5} {'loc_t':>8} {'phi0_deg':>10} "
            f"{'a_fwd':>10} {'a_bwd':>10} {'vr_f':>10} {'vr_b':>10}"
        )
        print(hdr + "   (angles deg)")
        print("-" * len(hdr))
        for r in rows:
            print(
                f"{r['t']:10.6f} {r['edge']:5d} {r['local_t']:8.4f} "
                f"{r['phi0_deg']:10.3f} {r['alpha_fwd_deg']:10.3f} {r['alpha_bwd_deg']:10.3f} "
                f"{r['v_r_fwd']:10.5f} {r['v_r_bwd']:10.5f}"
            )

    # Per-edge statistics on sampled alpha (forward branch by default)
    key_alpha = "alpha_fwd_deg" if which_solutions != "backward" else "alpha_bwd_deg"
    print("\nPer-edge alpha range on this t-sample (useful when omega != 0):\n")
    print(f"{'edge':>5} {'n':>6}  {key_alpha + ' min':>14}  {key_alpha + ' max':>14}  {'span':>10}")
    print("-" * 58)
    for i, _, _ in ranges:
        vals = [r[key_alpha] for r in rows if r["edge"] == i]
        if not vals:
            print(f"{i:5d} {0:6d}  {'—':>14}  {'—':>14}  {'—':>10}")
            continue
        arr = np.array(vals)
        print(
            f"{i:5d} {len(vals):6d}  {arr.min():14.4f}  {arr.max():14.4f}  "
            f"{arr.max() - arr.min():10.4f}"
        )


def plot_alpha_scan_figure(
    rows: List[dict],
    ranges: List[Tuple[int, float, float]],
    verts: np.ndarray,
    centroid: np.ndarray,
    shape_name: str,
    v_body: np.ndarray,
    omega: float,
    R_r: float,
    theta0: float,
    obj_pos: np.ndarray,
    which_solutions: str,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    """
    One-shot spatial view: object at center; solid heading vectors (zeta_0) and
    dashed R_r vectors (center → contact); color by edge to show alpha_0 bands.
    """
    branch = "backward" if which_solutions == "backward" else "forward"
    key_alpha = _alpha_scan_branch_key(which_solutions)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    _draw_shape(ax, verts, obj_pos, theta0, centroid,
                color="tab:gray", alpha_fill=0.35, lw=1.5)

    # Boundary polyline (world frame at t=0)
    R0 = rot2d(theta0)
    bnd = (R0 @ (verts[:, :2] - centroid).T).T + obj_pos
    ax.plot(bnd[:, 0], bnd[:, 1], "k-", lw=1.0, alpha=0.35, zorder=1)

    cmap = plt.cm.tab10
    n_edges = max((r["edge"] for r in rows), default=0) + 1
    edge_colors = {i: cmap(i % 10) for i in range(n_edges)}

    cp_all = np.array([r["cp_world"] for r in rows])
    ax.scatter(cp_all[:, 0], cp_all[:, 1], s=14, c="crimson", zorder=4,
               label="Object contact", edgecolors="k", linewidths=0.3)

    heading_scale = 0.55 * R_r
    for r in rows:
        ec = edge_colors[r["edge"]]
        rc = r["robot_center"]
        cp = r["cp_world"]
        zeta = r["zeta0"]

        # Heading (zeta_0): solid — primary cue for alpha variation along the edge
        hx = rc[0] + heading_scale * np.cos(zeta)
        hy = rc[1] + heading_scale * np.sin(zeta)
        ax.annotate(
            "",
            xy=(hx, hy),
            xytext=(rc[0], rc[1]),
            arrowprops=dict(arrowstyle="-|>", color=ec, lw=1.6, mutation_scale=10),
            zorder=4,
        )
        # Robot center → contact (R_r): dashed — geometry reference only
        ax.plot(
            [rc[0], cp[0]], [rc[1], cp[1]],
            color=ec, ls="--", lw=1.0, alpha=0.7, zorder=2,
        )
        ax.plot(rc[0], rc[1], ".", color=ec, ms=3, zorder=3)

    # Legend: one entry per edge with alpha span on the sample
    legend_handles = []
    for i, _, _ in ranges:
        vals = [r[key_alpha] for r in rows if r["edge"] == i]
        if not vals:
            continue
        arr = np.array(vals)
        span = arr.max() - arr.min()
        legend_handles.append(
            Line2D(
                [0], [0], color=edge_colors[i], lw=2,
                label=f"edge {i}: α ∈ [{arr.min():.1f}°, {arr.max():.1f}°]  (span {span:.1f}°)",
            )
        )
    legend_handles.append(
        Line2D([0], [0], color="k", ls="-", lw=2,
               label=f"heading ζ₀ ({branch}, 0.55·R_r)")
    )
    legend_handles.append(
        Line2D([0], [0], color="gray", ls="--", lw=1.2,
               label=f"robot vector (R_r={R_r:.3f} m, dashed)")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.92, borderpad=0.6)

    omega_note = "constant α per edge" if abs(omega) < 1e-12 else "α varies along edge (band)"
    ax.set_title(
        f"Feasible DD contact geometry — {shape_name}\n"
        f"v_body=({v_body[0]:.3g}, {v_body[1]:.3g}) m/s,  ω={omega:.3g} rad/s  "
        f"[{branch}]  ({omega_note})",
        fontsize=11,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Tight limits around object + robot offsets
    pts = np.vstack([cp_all, np.array([r["robot_center"] for r in rows])])
    pad = max(R_r * 2.5, 0.08)
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"\n  Saved alpha_scan figure to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_alpha_scan(
    cpp: ContactPointParameterization,
    centroid: np.ndarray,
    verts: np.ndarray,
    shape_name: str,
    v_body: np.ndarray,
    omega: float,
    R_r: float,
    theta0: float,
    n_t: int,
    which_solutions: str = "forward",
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    do_spatial_plot: bool = True,
) -> None:
    """Print alpha_0 table and optionally render the spatial feasibility figure."""
    obj_pos = np.zeros(2)
    rows, ranges = collect_alpha_scan_rows(
        cpp, centroid, v_body, omega, R_r, theta0, n_t, which_solutions, obj_pos,
    )
    print_alpha_scan_table(
        rows, ranges, which_solutions, cpp, v_body, omega, R_r, theta0,
    )
    if do_spatial_plot:
        plot_alpha_scan_figure(
            rows, ranges, verts, centroid, shape_name,
            v_body, omega, R_r, theta0, obj_pos,
            which_solutions, save_path=save_path, show_plot=show_plot,
        )


# ---------------------------------------------------------------------------
# 8. CLI & main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Velocity-matching test: constant-velocity diff-drive solution",
    )
    p.add_argument(
        "--mode",
        choices=["plot", "alpha_scan"],
        default="plot",
        help="plot: full simulation + figure; alpha_scan: table + spatial alpha_0 figure",
    )
    p.add_argument(
        "--alpha-scan-no-plot",
        action="store_true",
        help="With --mode alpha_scan, skip the spatial feasibility figure",
    )
    p.add_argument(
        "--n-t",
        type=int,
        default=41,
        help="Number of global t samples in [0,1) for --mode alpha_scan",
    )
    p.add_argument(
        "--alpha-scan-solutions",
        choices=["forward", "backward", "both"],
        default="forward",
        help="Which DD branch(es) to list in alpha_scan table",
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

    v_body = np.array([args.vx_body, args.vy_body])
    omega = args.omega
    R_r = args.robot_radius
    theta0 = 0.0

    if args.mode == "alpha_scan":
        scan_save = SAVE_DIR / (
            args.save or f"{args.shape}_alpha_scan_w{args.omega:.2f}.png"
        )
        run_alpha_scan(
            cpp,
            centroid,
            verts,
            args.shape,
            v_body,
            omega,
            R_r,
            theta0,
            n_t=args.n_t,
            which_solutions=args.alpha_scan_solutions,
            save_path=scan_save,
            show_plot=not args.silent,
            do_spatial_plot=not args.alpha_scan_no_plot,
        )
        print("\nDone (alpha_scan).")
        return

    cp_info = cpp.get_contact_info(args.t_param)
    cp_body = cp_info["point"] - centroid
    n_out_body = cp_info["normal_outward"]
    print(f"  cp_body = {cp_body},  n_out_body = {n_out_body}")

    # ---- Object propagation ----
    obj = propagate_object(v_body, omega, args.duration, args.dt,
                           0.0, 0.0, theta0, cp_body, n_out_body)

    path_type = "arc" if abs(omega) > 1e-9 else "straight line"
    print(f"\nObject: v_body={v_body}, omega={omega:.3f} => {path_type}")

    # Contact velocity in body frame (constant!)
    v_cp_body = v_body + omega * np.array([-cp_body[1], cp_body[0]])
    v_cp_world0 = rot2d(theta0) @ v_cp_body
    print(f"  v_cp (body, constant) = {v_cp_body}")
    print(f"  v_cp (world, t=0)     = {v_cp_world0}")

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

    dd_fwd, dd_bwd = dd_results

    # ---- Plot ----
    save_name = args.save or f"{args.shape}_t{args.t_param:.2f}_w{args.omega:.2f}.png"
    save_path = SAVE_DIR / save_name
    plot_results(
        obj, holo, dd_fwd, dd_bwd, verts, centroid, R_r,
        save_path=save_path, show_plot=not args.silent
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
