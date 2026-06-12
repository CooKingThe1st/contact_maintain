#!/usr/bin/env python3
"""
Flat-bumper (two-endpoint) velocity-matching feasibility for a diff-drive robot.

Given object shape, constant body twist, robot bumper geometry (two body-fixed
endpoints), and one polygonal edge, decide whether a placement along that edge
admits constant (v_r, omega_r) with omega_r = omega that matches contact velocity
at BOTH bumper endpoints (fixed patch, dot_alpha = 0).

Modes:
  scan — sample edge parameter t (like test_matchingvelo alpha_scan); print
         per-edge alpha bands at E1/E2, Delta_psi from robot design, and
         per-sample feasible yes/no.
  plot — if feasible, pick a sample, propagate object + robot, plot motion.

See test_matchingvelo_report.md Section 9.

Note: In this workspace PyBullet has been observed to crash during interpreter
shutdown (malloc_consolidate invalid chunk). This script hard-exits after
flushing stdout/stderr to avoid that teardown path.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
# Reuse shape loading and helpers from the disc test
from test_matchingvelo import (  # noqa: E402
    OBJ_FILE_MAP,
    SAVE_DIR,
    ObjectTrajectory,
    RobotResult,
    _draw_shape,
    compute_dd_solutions,
    edge_global_t_ranges,
    load_shape,
    propagate_object,
    rot2d,
    _wrap,
)

# ---------------------------------------------------------------------------
# Robot bumper design (body frame, robot heading = +x)
# ---------------------------------------------------------------------------
@dataclass
class BumperDesign:
    """Two bumper endpoints in robot body frame + optional actuator limits."""

    r_E1_b: np.ndarray
    r_E2_b: np.ndarray
    v_max: float = 1e9
    omega_max: float = 1e9

    @property
    def delta_psi_b(self) -> float:
        return _wrap(
            np.arctan2(self.r_E2_b[1], self.r_E2_b[0])
            - np.arctan2(self.r_E1_b[1], self.r_E1_b[0])
        )

    @property
    def half_length(self) -> float:
        return 0.5 * np.linalg.norm(self.r_E2_b - self.r_E1_b)

    @property
    def center_b(self) -> np.ndarray:
        return 0.5 * (self.r_E1_b + self.r_E2_b)


def design_from_args(args) -> BumperDesign:
    """Build endpoints from CLI: center offset + half-length along body +y."""
    if args.bumper_e1 is not None and args.bumper_e2 is not None:
        e1 = np.asarray(args.bumper_e1, dtype=float)
        e2 = np.asarray(args.bumper_e2, dtype=float)
        return BumperDesign(e1, e2, v_max=args.v_max, omega_max=args.omega_max)
    cx, cy = args.bumper_center
    hl = args.bumper_half_length
    if args.bumper_along_y:
        r1 = np.array([cx, cy - hl], dtype=float)
        r2 = np.array([cx, cy + hl], dtype=float)
    else:
        r1 = np.array([cx - hl, cy], dtype=float)
        r2 = np.array([cx + hl, cy], dtype=float)
    return BumperDesign(r1, r2, v_max=args.v_max, omega_max=args.omega_max)


# ---------------------------------------------------------------------------
# Kinematics at t=0
# ---------------------------------------------------------------------------
def v_cp_world_at(
    v_body: np.ndarray,
    omega: float,
    theta0: float,
    r_body: np.ndarray,
) -> np.ndarray:
    v_cp_b = v_body + omega * np.array([-r_body[1], r_body[0]])
    return rot2d(theta0) @ v_cp_b


def solve_zeta_two_endpoint(
    vcp1: np.ndarray,
    vcp2: np.ndarray,
    omega: float,
    r1b: np.ndarray,
    r2b: np.ndarray,
    tol: float = 1e-9,
) -> Optional[float]:
    """
    Solve zeta_0 from a_1=a_2 and b_1=b_2 with lever arms r_Ei(0)=R(zeta) r_Ei^b.
    Returns None if infeasible.
    """
    w = r1b - r2b
    w2 = float(np.dot(w, w))
    if w2 < tol:
        return None

    if abs(omega) < tol:
        if np.linalg.norm(vcp1 - vcp2) > 1e-7:
            return None
        # Translation: zeta from either endpoint once vcp known
        return None  # caller uses atan2(b, a) from one endpoint

    dv = vcp2 - vcp1
    M = np.array([[w[0], w[1]], [-w[1], w[0]]], dtype=float)
    sc = np.linalg.solve(M, dv / omega)
    sn, cs = float(sc[0]), float(sc[1])
    nrm = np.hypot(sn, cs)
    if nrm < tol or abs(nrm - 1.0) > 0.02:
        return None
    return float(np.arctan2(sn / nrm, cs / nrm))


def ab_from_zeta(
    vcp: np.ndarray, omega: float, zeta: float, r_b: np.ndarray
) -> Tuple[float, float]:
    r_w = rot2d(zeta) @ r_b
    a = vcp[0] + omega * r_w[1]
    b = vcp[1] - omega * r_w[0]
    return a, b


def disc_alpha_req(
    vcp_w: np.ndarray,
    omega: float,
    phi_n: float,
    R_lever: float,
    branch: str = "forward",
) -> Tuple[float, float]:
    """Disc-style alpha_0 at one endpoint (for band visualization; uses phi_n)."""
    sols = compute_dd_solutions(vcp_w, omega, R_lever, phi_n)
    by = {s["label"]: s for s in sols}
    s = by["backward" if branch == "backward" else "forward"]
    return float(s["alpha"]), float(s["zeta0"])


@dataclass
class SegmentSample:
    t: float
    edge: int
    local_t: float
    cp_body: np.ndarray
    n_out_b: np.ndarray
    t_edge_b: np.ndarray
    r_o1_b: np.ndarray
    r_o2_b: np.ndarray
    vcp1_w: np.ndarray
    vcp2_w: np.ndarray
    phi_n: float
    feasible: bool
    zeta0: float = 0.0
    v_r: float = 0.0
    alpha1_req: float = 0.0
    alpha2_req: float = 0.0
    alpha_diff_req: float = 0.0
    zeta1_disc: float = 0.0
    zeta2_disc: float = 0.0
    reason: str = ""


def evaluate_sample(
    v_body: np.ndarray,
    omega: float,
    theta0: float,
    design: BumperDesign,
    cp_body: np.ndarray,
    n_out_b: np.ndarray,
    t_edge_b: np.ndarray,
    branch: str,
    align_tol_deg: float,
) -> SegmentSample:
    """Check dual-endpoint feasibility at one bumper placement (center on cp_body)."""
    hl = design.half_length
    t_hat = t_edge_b / (np.linalg.norm(t_edge_b) + 1e-15)
    r_o1 = cp_body - hl * t_hat
    r_o2 = cp_body + hl * t_hat

    R0 = rot2d(theta0)
    n_w = R0 @ n_out_b
    phi_n = float(np.arctan2(-n_w[1], -n_w[0]))
    vcp1 = v_cp_world_at(v_body, omega, theta0, r_o1)
    vcp2 = v_cp_world_at(v_body, omega, theta0, r_o2)

    R1 = float(np.linalg.norm(design.r_E1_b))
    R2 = float(np.linalg.norm(design.r_E2_b))
    a1d, z1 = disc_alpha_req(vcp1, omega, phi_n, R1, branch)
    a2d, z2 = disc_alpha_req(vcp2, omega, phi_n, R2, branch)

    row = SegmentSample(
        t=0.0,
        edge=0,
        local_t=0.0,
        cp_body=cp_body,
        n_out_b=n_out_b,
        t_edge_b=t_edge_b,
        r_o1_b=r_o1,
        r_o2_b=r_o2,
        vcp1_w=vcp1,
        vcp2_w=vcp2,
        phi_n=phi_n,
        feasible=False,
        alpha1_req=a1d,
        alpha2_req=a2d,
        alpha_diff_req=_wrap(a2d - a1d),
        zeta1_disc=z1,
        zeta2_disc=z2,
    )

    zeta = solve_zeta_two_endpoint(vcp1, vcp2, omega, design.r_E1_b, design.r_E2_b)
    if zeta is None and abs(omega) < 1e-12:
        zeta = float(np.arctan2(vcp1[1], vcp1[0]))
        if branch == "backward":
            zeta = _wrap(zeta + np.pi)
    elif zeta is None:
        row.reason = "no zeta solves a1=a2, b1=b2"
        return row

    a, b = ab_from_zeta(vcp1, omega, zeta, design.r_E1_b)
    a2, b2 = ab_from_zeta(vcp2, omega, zeta, design.r_E2_b)
    if np.hypot(a - a2, b - b2) > 1e-5:
        row.reason = "a,b mismatch after zeta solve"
        return row

    speed = np.hypot(a, b)
    if branch == "backward":
        v_r = -speed
        zeta_cmd = _wrap(zeta + np.pi) if speed > 1e-12 else zeta
    else:
        v_r = speed
        zeta_cmd = zeta

    if abs(v_r) > design.v_max or abs(omega) > design.omega_max:
        row.reason = "actuator limit"
        return row

    # Optional: bumper tangent ~ object edge tangent in world
    t_w = R0 @ t_hat
    bumper_t_b = design.r_E2_b - design.r_E1_b
    bumper_t_w = rot2d(zeta_cmd) @ bumper_t_b
    ang = np.degrees(
        np.arccos(
            np.clip(
                abs(np.dot(bumper_t_w, t_w))
                / (np.linalg.norm(bumper_t_w) * np.linalg.norm(t_w) + 1e-15),
                -1.0,
                1.0,
            )
        )
    )
    if align_tol_deg < 179.0 and ang > align_tol_deg:
        row.reason = f"bumper-edge misalign {ang:.1f} deg"
        return row

    dpsi = design.delta_psi_b
    psi1 = _wrap(float(np.arctan2(
        (rot2d(zeta_cmd) @ design.r_E1_b)[1],
        (rot2d(zeta_cmd) @ design.r_E1_b)[0],
    )) - zeta_cmd)
    psi2 = _wrap(float(np.arctan2(
        (rot2d(zeta_cmd) @ design.r_E2_b)[1],
        (rot2d(zeta_cmd) @ design.r_E2_b)[0],
    )) - zeta_cmd)
    alpha_geom_diff = _wrap(psi2 - psi1)
    if abs(_wrap(alpha_geom_diff - dpsi)) > 1e-4:
        row.reason = "internal: psi2-psi1 != Delta_psi"
        return row

    row.feasible = True
    row.zeta0 = zeta_cmd
    row.v_r = v_r
    row.reason = "ok"
    return row


def collect_edge_samples(
    cpp,
    centroid: np.ndarray,
    edge_idx: int,
    v_body: np.ndarray,
    omega: float,
    theta0: float,
    design: BumperDesign,
    n_t: int,
    branch: str,
    align_tol_deg: float,
) -> List[SegmentSample]:
    ranges = edge_global_t_ranges(cpp)
    match = [r for r in ranges if r[0] == edge_idx]
    if not match:
        raise ValueError(f"edge {edge_idx} not found (have {len(ranges)} edges)")
    _, t_lo, t_hi = match[0]
    ts = np.linspace(t_lo, t_hi, max(2, n_t), endpoint=False)

    out: List[SegmentSample] = []
    for t in ts:
        info = cpp.get_contact_info(float(t))
        cp_body = np.asarray(info["point"], dtype=float) - centroid
        n_out = np.asarray(info["normal_outward"], dtype=float)
        seg_i = int(info["segment_index"])
        if seg_i != edge_idx:
            continue
        # Edge tangent from segment vertices in body frame
        p0 = np.asarray(cpp.boundary_coords[seg_i], dtype=float) - centroid
        p1 = np.asarray(cpp.boundary_coords[seg_i + 1], dtype=float) - centroid
        t_edge = p1 - p0

        s = evaluate_sample(
            v_body, omega, theta0, design, cp_body, n_out, t_edge, branch, align_tol_deg,
        )
        s.t = float(t)
        s.edge = edge_idx
        s.local_t = float(info["local_parameter"])
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Propagation with two tracked endpoints
# ---------------------------------------------------------------------------
@dataclass
class SegmentTrajectories:
    obj: ObjectTrajectory
    obj_ep1: np.ndarray
    obj_ep2: np.ndarray
    robot: RobotResult
    rob_ep1: np.ndarray
    rob_ep2: np.ndarray
    v_rob_ep1: np.ndarray
    v_rob_ep2: np.ndarray


def propagate_bumper_dd_exact(
    v_r: float,
    omega_r: float,
    zeta0: float,
    center0: np.ndarray,
    design: BumperDesign,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact constant (v_r, omega_r) unicycle integration (same as test_motion_primitive).
    Returns centers, headings, ep1, ep2, (v1, v2) endpoint velocities.
    """
    n_steps = len(times)
    headings = zeta0 + omega_r * times
    centers = np.zeros((n_steps, 2), dtype=float)
    ep1 = np.zeros((n_steps, 2), dtype=float)
    ep2 = np.zeros((n_steps, 2), dtype=float)
    v1 = np.zeros((n_steps, 2), dtype=float)
    v2 = np.zeros((n_steps, 2), dtype=float)

    if abs(omega_r) < 1e-12:
        headings[:] = float(zeta0)
        direction = np.array([np.cos(float(zeta0)), np.sin(float(zeta0))], dtype=float)
        centers = center0 + (v_r * times)[:, None] * direction[None, :]
    else:
        k = v_r / omega_r
        z0 = float(zeta0)
        sin_z = np.sin(headings)
        cos_z = np.cos(headings)
        centers[:, 0] = center0[0] + k * (sin_z - np.sin(z0))
        centers[:, 1] = center0[1] + k * (np.cos(z0) - cos_z)

    for i in range(n_steps):
        Rr = rot2d(headings[i])
        ep1[i] = centers[i] + Rr @ design.r_E1_b
        ep2[i] = centers[i] + Rr @ design.r_E2_b
        v_base = np.array([v_r * np.cos(headings[i]), v_r * np.sin(headings[i])])
        for r_b, v_out in ((design.r_E1_b, v1), (design.r_E2_b, v2)):
            r_w = Rr @ r_b
            v_out[i] = v_base + omega_r * np.array([-r_w[1], r_w[0]])

    return centers, headings, ep1, ep2, v1, v2


def propagate_segment(
    v_body: np.ndarray,
    omega: float,
    T: float,
    dt: float,
    theta0: float,
    cp_body: np.ndarray,
    n_out_b: np.ndarray,
    r_o1_b: np.ndarray,
    r_o2_b: np.ndarray,
    zeta0: float,
    v_r: float,
    design: BumperDesign,
) -> SegmentTrajectories:
    obj = propagate_object(v_body, omega, T, dt, 0.0, 0.0, theta0, cp_body, n_out_b)
    n_steps = len(obj.times)
    obj_ep1 = np.zeros((n_steps, 2))
    obj_ep2 = np.zeros((n_steps, 2))
    for i in range(n_steps):
        R = rot2d(obj.thetas[i])
        obj_ep1[i] = obj.positions[i] + R @ r_o1_b
        obj_ep2[i] = obj.positions[i] + R @ r_o2_b

    center0 = obj_ep1[0] - rot2d(zeta0) @ design.r_E1_b
    centers, headings, ep1, ep2, v1, v2 = propagate_bumper_dd_exact(
        v_r, omega, zeta0, center0, design, obj.times,
    )

    rob = RobotResult(
        label="DD bumper",
        robot_centers=centers,
        robot_cp=ep1,
        v_robot_cp=v1,
        headings=headings,
        v_r=np.full(n_steps, v_r),
        omega_r=np.full(n_steps, omega),
        zeta0=zeta0,
    )

    return SegmentTrajectories(
        obj=obj, obj_ep1=obj_ep1, obj_ep2=obj_ep2, robot=rob,
        rob_ep1=ep1, rob_ep2=ep2, v_rob_ep1=v1, v_rob_ep2=v2,
    )


def vcp_obj_at_times(obj: ObjectTrajectory, r_body: np.ndarray, v_body, omega) -> np.ndarray:
    n = len(obj.times)
    v = np.zeros((n, 2))
    for i in range(n):
        R = rot2d(obj.thetas[i])
        v[i] = R @ v_body + omega * R @ np.array([-r_body[1], r_body[0]])
    return v


# ---------------------------------------------------------------------------
# Scan output
# ---------------------------------------------------------------------------
def print_scan_report(
    samples: List[SegmentSample],
    design: BumperDesign,
    v_body: np.ndarray,
    omega: float,
    edge_idx: int,
) -> bool:
    any_ok = any(s.feasible for s in samples)
    print("\n" + "=" * 72)
    print("Segment (flat bumper) feasibility scan")
    print("=" * 72)
    print(f"  edge = {edge_idx},  v_body = {v_body},  omega = {omega:.6g} rad/s")
    print(f"  r_E1^b = {design.r_E1_b},  r_E2^b = {design.r_E2_b}")
    print(f"  Delta_psi^b (robot) = {np.degrees(design.delta_psi_b):.4f} deg")

    if not samples:
        print("  No samples on this edge.")
        return False

    a1 = np.degrees([s.alpha1_req for s in samples])
    a2 = np.degrees([s.alpha2_req for s in samples])
    dif = np.degrees([s.alpha_diff_req for s in samples])
    print(f"\n  Disc-style alpha bands on this edge (endpoint vs normal, for comparison):")
    print(f"    alpha_1^req: [{a1.min():.2f}, {a1.max():.2f}] deg  span {a1.max()-a1.min():.2f}")
    print(f"    alpha_2^req: [{a2.min():.2f}, {a2.max():.2f}] deg  span {a2.max()-a2.min():.2f}")
    print(f"    alpha_2^req - alpha_1^req (disc est.): [{dif.min():.2f}, {dif.max():.2f}] deg")
    print(f"    robot Delta_psi^b = {np.degrees(design.delta_psi_b):.2f} deg")
    print("    (feasibility uses a1=a2, b1=b2 + zeta solve, not disc alpha diff alone)")

    n_ok = sum(1 for s in samples if s.feasible)
    print(f"\n  FEASIBLE: {'YES' if any_ok else 'NO'}  ({n_ok}/{len(samples)} samples)")
    print("\n  Per-sample (t on edge):")
    hdr = f"{'t':>10} {'loc_t':>8} {'a1_deg':>9} {'a2_deg':>9} {'diff':>9} {'ok':>4}  reason"
    print(hdr)
    print("-" * len(hdr))
    for s in samples:
        ok = "yes" if s.feasible else "no"
        print(
            f"{s.t:10.6f} {s.local_t:8.4f} "
            f"{np.degrees(s.alpha1_req):9.2f} {np.degrees(s.alpha2_req):9.2f} "
            f"{np.degrees(s.alpha_diff_req):9.2f} {ok:>4}  {s.reason}"
        )
    return any_ok


def plot_scan_figure(
    samples: List[SegmentSample],
    verts: np.ndarray,
    centroid: np.ndarray,
    shape_name: str,
    design: BumperDesign,
    v_body: np.ndarray,
    omega: float,
    edge_idx: int,
    save_path: Optional[Path],
    show_plot: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    obj_pos = np.zeros(2)
    theta0 = 0.0
    _draw_shape(ax, verts, obj_pos, theta0, centroid, color="tab:gray", alpha_fill=0.35)
    R0 = rot2d(theta0)
    bnd = (R0 @ (verts[:, :2] - centroid).T).T + obj_pos
    ax.plot(bnd[:, 0], bnd[:, 1], "k-", lw=1.0, alpha=0.35)

    for s in samples:
        c = "tab:green" if s.feasible else "tab:red"
        cp = obj_pos + R0 @ s.cp_body
        ax.plot(cp[0], cp[1], "o", color=c, ms=5, mec="k", mew=0.3)
        if s.feasible:
            rc = cp - rot2d(s.zeta0) @ design.center_b
            e1 = rc + rot2d(s.zeta0) @ design.r_E1_b
            e2 = rc + rot2d(s.zeta0) @ design.r_E2_b
            ax.plot([e1[0], e2[0]], [e1[1], e2[1]], "-", color=c, lw=2.0, alpha=0.85)

    ax.set_title(
        f"Segment scan — {shape_name} edge {edge_idx}\n"
        f"green = feasible, red = infeasible",
        fontsize=11,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="tab:green", lw=2, label="feasible bumper"),
            Line2D([0], [0], color="tab:red", marker="o", ls="", label="infeasible"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved scan figure to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _draw_bumper(
    ax,
    center: np.ndarray,
    heading: float,
    design: BumperDesign,
    *,
    color: str = "tab:blue",
    lw: float = 2.0,
    alpha: float = 0.85,
    draw_heading: bool = True,
    heading_scale: float = 0.08,
) -> None:
    """Draw bumper segment and optional heading arrow at one pose."""
    Rr = rot2d(heading)
    e1 = center + Rr @ design.r_E1_b
    e2 = center + Rr @ design.r_E2_b
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], "-", color=color, lw=lw, alpha=alpha, zorder=6)
    ax.plot(center[0], center[1], ".", color=color, ms=4, zorder=5)
    if draw_heading:
        tip = center + heading_scale * np.array([np.cos(heading), np.sin(heading)])
        ax.annotate(
            "",
            xy=(tip[0], tip[1]),
            xytext=(center[0], center[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=10),
            zorder=6,
        )


def _autolim_spatial(ax, pts: np.ndarray, pad_frac: float = 0.12) -> None:
    if pts.size == 0:
        return
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = max(xmax - xmin, 0.05)
    dy = max(ymax - ymin, 0.05)
    pad_x = pad_frac * dx
    pad_y = pad_frac * dy
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)


def plot_motion_figure(
    traj: SegmentTrajectories,
    verts: np.ndarray,
    centroid: np.ndarray,
    sample: SegmentSample,
    design: BumperDesign,
    shape_name: str,
    v_body: np.ndarray,
    omega: float,
    save_path: Optional[Path],
    show_plot: bool,
    n_snap: int = 8,
    plot_mid_object: bool = True,
) -> None:
    """
    Motion visualization in the style of test_motion_primitive.plot_2d:
    one large spatial panel with trajectories, object start/end, and bumper snapshots.
    """
    obj, rob = traj.obj, traj.robot
    n_steps = len(obj.times)
    mid_idx = n_steps // 2

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.32, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax.set_aspect("equal")

    # --- Object motion ---
    ax.plot(obj.positions[:, 0], obj.positions[:, 1], "b--", lw=1.4, label="Object CoM")
    ax.plot(obj.cp_world[:, 0], obj.cp_world[:, 1], color="tab:olive", lw=1.2, ls=":",
            alpha=0.8, label="Object contact (center)")
    ax.plot(traj.obj_ep1[:, 0], traj.obj_ep1[:, 1], "r-", lw=2.0, label="Object E1")
    ax.plot(traj.obj_ep2[:, 0], traj.obj_ep2[:, 1], "r--", lw=2.0, label="Object E2")

    _draw_shape(
        ax, verts, obj.positions[0], obj.thetas[0], centroid,
        color="tab:orange", alpha_fill=0.30, lw=1.2,
    )
    _draw_shape(
        ax, verts, obj.positions[-1], obj.thetas[-1], centroid,
        color="tab:gray", alpha_fill=0.25, lw=1.0,
    )
    if plot_mid_object and n_steps > 2:
        _draw_shape(
            ax, verts, obj.positions[mid_idx], obj.thetas[mid_idx], centroid,
            color="tab:purple", alpha_fill=0.18, lw=0.9,
        )

    ax.scatter(
        [obj.positions[0, 0], obj.positions[-1, 0]],
        [obj.positions[0, 1], obj.positions[-1, 1]],
        c="blue", s=24, zorder=4,
    )

    # --- Robot motion ---
    ax.plot(traj.rob_ep1[:, 0], traj.rob_ep1[:, 1], color="tab:green", lw=2.0,
            label="Robot bumper E1")
    ax.plot(traj.rob_ep2[:, 0], traj.rob_ep2[:, 1], color="tab:green", lw=2.0, ls="--",
            label="Robot bumper E2")
    ax.plot(rob.robot_centers[:, 0], rob.robot_centers[:, 1], color="tab:blue",
            lw=1.0, ls=":", alpha=0.75, label="Robot center")

    p1e = np.linalg.norm(traj.rob_ep1 - traj.obj_ep1, axis=1)
    p2e = np.linalg.norm(traj.rob_ep2 - traj.obj_ep2, axis=1)
    print(
        f"  Endpoint position error: E1 max {np.max(p1e)*1e3:.4f} mm, "
        f"E2 max {np.max(p2e)*1e3:.4f} mm"
    )

    # Snapshots along the path (motion_primitive style)
    step = max(1, n_steps // max(2, n_snap))
    snap_indices = list(range(0, n_steps, step))
    if (n_steps - 1) not in snap_indices:
        snap_indices.append(n_steps - 1)
    if 0 not in snap_indices:
        snap_indices.insert(0, 0)
    snap_indices = sorted(set(snap_indices))

    h_scale = max(0.06, 0.35 * design.half_length)
    for idx in snap_indices:
        _draw_bumper(
            ax, rob.robot_centers[idx], rob.headings[idx], design,
            color="tab:blue", lw=2.0, alpha=0.55 if idx not in (0, n_steps - 1) else 0.9,
            heading_scale=h_scale,
        )
        ax.plot(traj.rob_ep1[idx, 0], traj.rob_ep1[idx, 1], "o",
                color="tab:green", ms=4, mec="k", mew=0.3, zorder=7)
        ax.plot(traj.rob_ep2[idx, 0], traj.rob_ep2[idx, 1], "o",
                color="tab:green", ms=4, mec="k", mew=0.3, zorder=7)
        ax.plot(traj.obj_ep1[idx, 0], traj.obj_ep1[idx, 1], "x",
                color="crimson", ms=6, mew=1.2, zorder=7)
        ax.plot(traj.obj_ep2[idx, 0], traj.obj_ep2[idx, 1], "x",
                color="crimson", ms=6, mew=1.2, zorder=7)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Segment motion — {shape_name}, edge {sample.edge}, t={sample.t:.4f}\n"
        f"v_body=({v_body[0]:.3g}, {v_body[1]:.3g}) m/s, omega={omega:.3g} rad/s, "
        f"zeta0={np.degrees(sample.zeta0):.1f} deg, v_r={sample.v_r:.4f} m/s",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="best", framealpha=0.92)

    all_pts = np.vstack([
        obj.positions, obj.cp_world, traj.obj_ep1, traj.obj_ep2,
        traj.rob_ep1, traj.rob_ep2, rob.robot_centers,
    ])
    _autolim_spatial(ax, all_pts)

    # --- Diagnostics row ---
    v1o = vcp_obj_at_times(obj, sample.r_o1_b, v_body, omega)
    v2o = vcp_obj_at_times(obj, sample.r_o2_b, v_body, omega)
    e1v = np.linalg.norm(traj.v_rob_ep1 - v1o, axis=1) * 1e3
    e2v = np.linalg.norm(traj.v_rob_ep2 - v2o, axis=1) * 1e3

    ax_v = fig.add_subplot(gs[1, 0])
    ax_v.plot(obj.times, e1v, label="E1")
    ax_v.plot(obj.times, e2v, label="E2")
    ax_v.set_title("Endpoint |v| error [mm/s]")
    ax_v.set_xlabel("t [s]")
    ax_v.legend(fontsize=7)

    ax_p = fig.add_subplot(gs[1, 1])
    ax_p.plot(obj.times, p1e * 1e3, label="E1")
    ax_p.plot(obj.times, p2e * 1e3, label="E2")
    ax_p.set_title("Endpoint position error [mm]")
    ax_p.set_xlabel("t [s]")
    ax_p.legend(fontsize=7)

    ax_c = fig.add_subplot(gs[1, 2])
    ax_c.plot(obj.times, rob.v_r, color="tab:green", lw=1.5, label="v_r")
    ax_c.axhline(0, color="gray", ls=":", lw=0.6)
    ax_c.set_xlabel("t [s]")
    ax_c.set_ylabel("v_r [m/s]")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(obj.times, rob.omega_r, color="tab:orange", ls="--", lw=1.2, label="omega_r")
    ax_c2.set_ylabel("omega_r [rad/s]")
    ax_c.set_title("DD commands (flat)")
    lines1, lab1 = ax_c.get_legend_handles_labels()
    lines2, lab2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="best")

    fig.subplots_adjust(hspace=0.38, wspace=0.30)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved motion figure to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Flat-bumper two-endpoint velocity-matching feasibility",
    )
    p.add_argument("--mode", choices=["scan", "plot", "both"], default="both")
    p.add_argument("--shape", default="rect", choices=list(OBJ_FILE_MAP))
    p.add_argument("--edge", type=int, default=0, help="Polygon edge index")
    p.add_argument("--vx_body", type=float, default=0.05)
    p.add_argument("--vy_body", type=float, default=0.0)
    p.add_argument("--omega", type=float, default=0.3)
    p.add_argument("--theta0", type=float, default=0.0)
    p.add_argument("--n-t", type=int, default=41, help="Samples along edge")
    p.add_argument("--branch", choices=["forward", "backward"], default="forward")
    p.add_argument(
        "--bumper-center",
        type=float,
        nargs=2,
        default=[0.06, 0.0],
        metavar=("CX", "CY"),
        help="Bumper face center in robot body frame [m]",
    )
    p.add_argument(
        "--bumper-half-length",
        type=float,
        default=0.05,
        help="Half-length of bumper along body +x [m]",
    )
    p.add_argument(
        "--bumper-along-y",
        action="store_true",
        help="Span bumper along body +y instead of +x",
    )
    p.add_argument(
        "--bumper-e1",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Override endpoint 1 body position",
    )
    p.add_argument(
        "--bumper-e2",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Override endpoint 2 body position",
    )
    p.add_argument("--v-max", type=float, default=1e9)
    p.add_argument("--omega-max", type=float, default=1e9)
    p.add_argument(
        "--align-tol-deg",
        type=float,
        default=180.0,
        help="Max bumper vs edge tangent mismatch [deg] (180 = skip)",
    )
    p.add_argument("--t-pick", type=float, default=None,
                   help="Global t on edge for plot (default: first feasible)")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--silent", action="store_true")
    p.add_argument("--no-scan-plot", action="store_true")
    p.add_argument(
        "--n-snap",
        type=int,
        default=8,
        help="Number of robot/object snapshots along motion plot",
    )
    p.add_argument(
        "--no-mid-object",
        action="store_true",
        help="Do not draw object polygon at mid-trajectory",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("Segment velocity-matching (flat bumper)")
    print("=" * 60)

    generic, cpp = load_shape(args.shape)
    centroid = np.array([generic.geometry.centroid.x, generic.geometry.centroid.y])
    verts = np.array(generic.geometry.exterior.coords)
    v_body = np.array([args.vx_body, args.vy_body])
    omega = args.omega
    design = design_from_args(args)

    samples = collect_edge_samples(
        cpp, centroid, args.edge, v_body, omega, args.theta0,
        design, args.n_t, args.branch, args.align_tol_deg,
    )

    feasible = print_scan_report(samples, design, v_body, omega, args.edge)
    print("\n  Holonomic: YES (3 DOF vs 2 constraints per point; no fixed alpha_0 band)")

    base = args.save or f"{args.shape}_edge{args.edge}_segment"
    if args.mode in ("scan", "both") and not args.no_scan_plot:
        scan_path = SAVE_DIR / (
            base if base.endswith(".png") and "scan" in base else f"{base}_scan.png"
        )
        plot_scan_figure(
            samples, verts, centroid, args.shape, design,
            v_body, omega, args.edge, scan_path, show_plot=not args.silent,
        )

    if args.mode in ("plot", "both"):
        pick = None
        if args.t_pick is not None:
            for s in samples:
                if abs(s.t - args.t_pick) < 1e-6:
                    pick = s if s.feasible else None
                    break
        if pick is None:
            pick = next((s for s in samples if s.feasible), None)
        if pick is None:
            print("\n  PLOT: skipped (no feasible placement on this edge).")
            return 1 if not feasible else 0
        print(
            f"\n  PLOT: using t={pick.t:.6f}, zeta0={np.degrees(pick.zeta0):.2f} deg, "
            f"v_r={pick.v_r:.4f} m/s"
        )
        traj = propagate_segment(
            v_body, omega, args.duration, args.dt, args.theta0,
            pick.cp_body, pick.n_out_b, pick.r_o1_b, pick.r_o2_b,
            pick.zeta0, pick.v_r, design,
        )
        plot_path = SAVE_DIR / (
            base.replace("_scan", "_motion")
            if "_scan" in base
            else (base if str(base).endswith(".png") else f"{base}_motion.png")
        )
        plot_motion_figure(
            traj, verts, centroid, pick, design, args.shape, v_body, omega,
            plot_path, show_plot=not args.silent,
            n_snap=args.n_snap, plot_mid_object=not args.no_mid_object,
        )

    print("\nDone.")
    return 0 if feasible else 1


if __name__ == "__main__":
    # PyBullet occasionally crashes during interpreter shutdown in this workspace
    # (malloc_consolidate invalid chunk). A hard-exit avoids that teardown path.
    code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)