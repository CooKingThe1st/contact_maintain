#!/usr/bin/env python3
"""
Multi-pusher diff-drive test: N robots, single constant object twist.

Analogue of test_matchingvelo_report.md for the multi-robot case.
Each robot independently applies the Phase-7 fixed-alpha velocity-matching
controller (from test_single_pusher_diffdrive.py) at its own contact point,
all sharing the same desired constant object twist (v_ref_body, omega_ref).

Synchronization
---------------
SwarmHost (startup_mode="quick") runs per-robot rotate-then-creep approach
via RobotAgent.  The outer loop waits until EVERY robot has detected a contact
force above the threshold, then drives the following three-phase sequence:

  1. REALIGN  — every robot spins in place (v_r=0) to align its heading to the
                solved zeta0_i ("self-rotate" / stop_go step).  Each robot holds
                still once aligned and waits for the rest.
  2. PRE-PUSH HOLD — short pause (--stop-go-pre-push-hold-s) after all robots
                are aligned, letting dynamics settle.
  3. PUSH     — all robots start the Phase-7 controller simultaneously.  On
                the very first push tick each robot snaps alpha* to the actual
                push-entry contact angle (Option B from single-pusher stop_go).

    Optional ``--cross-track-integrate`` adds a bounded ω trim from lateral error
    to the nominal constant-twist screw from each push-start pose (CSV segment
    or plain ``--duration`` horizon).

    Push phase (Phase 7 — live-resolve variant)

    python3 /home/docker_user/catkin_ws/src/contact_maintain/scripts/test/test_multi_pusher_single_movement_diffdrive_liveupdate.py   --object rect --v-ref-x 0 --v-ref-y 0.2 --omega 0.2  --fixed-ref   --duration 80   --kp-p
    osition 0   --k-tangent 0.1   --k-couple 0.01   --k-force-comp 0.0   --kd-alpha 0.08   --kd-pos 0
    .2   --save-dir /tmp/multi_pusher_dd/ --record-video --obstructing-inflate-gap 0.02 --disable-act
    ual-contact-clearance-cheat --cross-track-integrate --cross-track-k -2 --cross-track-omega-max 0.2 --transition-teleport-robots --test-transition

--------------------------------------------
Each robot i independently, every control tick:
  1. Re-solves (vr_ff_i, zeta0_i, alpha*_i) from the CURRENT contact geometry
     (phi_live, v_cp_ref_live) via _init_segment_reference.  This replaces the
     single one-shot solve used in the baseline version and eliminates stale-
     reference drift without requiring any periodic stop/re-align.
  2. Runs _compute_phase7_command with the fresh seg_ref AND a PD alpha term:
       omega_r = omega_ff + kp_alpha*e_alpha + kd_alpha*(de_alpha/dt)
     The derivative term damps multi-robot coupling oscillations.
The object moves from real contact-force physics (not kinematically driven).
Robots default to planar-joint velocity cheat; use --no-planar-cheat for
wheel/bumper-limited actuation. Contact friction: --object-friction,
--bumper-contact-mu (both affect bumper-object slip).

TODO (Problem 2 — redundant contacts / non-contributing robots)
---------------------------------------------------------------
For a single fixed twist, the Magnum Four configuration guarantees FORM
CLOSURE (can push in any direction), but for THIS specific twist at least
one contact point will have near-zero required force and its robot becomes
a passive observer.  This is identical in the holonomic version.

The "ahead-of-the-object" robot drifts off contact because:
  a) Its vr_ff ≈ 0 → almost no normal force → no friction budget.
  b) The object (pushed by the other robots) physically moves away from it.

Planned fix path (in order of dependency):
  1. First solve the "contact-point adaptation" problem for actively pushing
     robots: when a robot's contact geometry drifts, it needs a mechanism to
     re-snap its reference without stopping the whole swarm.

     (change the ref alpha with accordance to new position)
  2. Then introduce a contact-force observer term: robots with F_contact < F_min
     switch to a gentle normal-press mode (small v_r bias) while the active
     robots continue with phase-7.  This uses measured contact force as a signal
     to distinguish "active pusher" from "passive/drifting" in real-time, and
     dynamically assigns roles without pre-planning which robot is redundant.

Example invocations
-------------------
 python3 test_multi_pusher_single_movement_diffdrive_liveupdate.py \
     --object rect --v-ref-x 0.05 --v-ref-y 0 --omega-ref 0.05 \
     --duration 50 --kd-alpha 0.1 --kd-pos 0.2 --save-dir /tmp/multi_pusher_dd_live

 # With ring coupling enabled (Step 2):
 python3 test_multi_pusher_single_movement_diffdrive_liveupdate.py \
     --object rect --v-ref-x 0.05 --v-ref-y 0 --omega-ref 0.05 \
     --duration 50 --kd-alpha 0.1 --kd-pos 0.2 --k-couple 0.1 \
     --save-dir /tmp/multi_pusher_dd_live
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import ContactPointParameterization, ContactPoint
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import obj_to_generic
from contact_maintain.robot_agent import RobotAgent
from contact_maintain.swarm import SwarmHost, SwarmState, RobotState
from contact_optimizer_utils import find_the_magnum_four_v3
from contact_maintain.diffdrive_path_control import solve_constant_body_twist_from_SE2


# ---------------------------------------------------------------------------
# Sim constants  (match test_single_pusher_diffdrive.py / test_diffdrive_wheel_physics.py)
# ---------------------------------------------------------------------------

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

DEFAULT_GROUND_FRICTION = 0.5
DEFAULT_WHEEL_LATERAL_FRICTION = 0.01
DEFAULT_CASTER_LATERAL_FRICTION = 0.01

DEFAULT_OBJECT_SHAPE = "rect"
DEFAULT_OBJECT_HEIGHT = 0.08
DEFAULT_OBJECT_FRICTION = 0.2

ROBOT_RADIUS = 0.06          # disc-bumper cylinder radius (diffdrive_wheel_robot_disc_bumper.urdf)
APPROACH_DISTANCE = 0.16      # spawn offset beyond contact point (metres)

# Approach contact gate: agent.contact_force must exceed this to count as "in contact".
APPROACH_CONTACT_GATE = 0.05  # N  (lower than SwarmHost default to catch first touch quickly)


# ---------------------------------------------------------------------------
# History dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RobotHistory:
    """Per-robot telemetry, recorded every control tick."""
    times: List[float] = field(default_factory=list)
    robot_positions: List[np.ndarray] = field(default_factory=list)
    robot_headings: List[float] = field(default_factory=list)
    robot_velocities: List[np.ndarray] = field(default_factory=list)
    robot_angular_velocities: List[float] = field(default_factory=list)
    intended_positions: List[np.ndarray] = field(default_factory=list)
    position_errors: List[np.ndarray] = field(default_factory=list)
    couple_target_positions: List[np.ndarray] = field(default_factory=list)
    couple_position_errors: List[np.ndarray] = field(default_factory=list)
    intended_contact_point_velocities: List[np.ndarray] = field(default_factory=list)
    contact_point_velocities: List[np.ndarray] = field(default_factory=list)        # object-side actual CP velocity
    robot_contact_point_velocities: List[np.ndarray] = field(default_factory=list)  # robot-side actual CP velocity
    intended_omegas: List[float] = field(default_factory=list)
    object_omegas: List[float] = field(default_factory=list)
    contact_forces: List[float] = field(default_factory=list)
    in_contact: List[bool] = field(default_factory=list)
    alpha_errors: List[float] = field(default_factory=list)
    v_r_history: List[float] = field(default_factory=list)
    omega_r_history: List[float] = field(default_factory=list)
    # ── v_r decomposition ─────────────────────────────────────────────────────
    v_ff_history: List[float] = field(default_factory=list)       # vr_ff (feed-forward)
    v_base_history: List[float] = field(default_factory=list)     # k_normal * normal gap
    v_speed_p_history: List[float] = field(default_factory=list)  # legacy slot: direct speed P (normally 0)
    v_pos_d_history: List[float] = field(default_factory=list)    # kd_pos * normal-gap derivative
    v_couple_history: List[float] = field(default_factory=list)   # normal-gap / force pressure sharing
    v_comp_history: List[float] = field(default_factory=list)     # force-gated contact compensation
    v_relax_history: List[float] = field(default_factory=list)    # inward vr_ff relaxation
    # ── omega_r decomposition ─────────────────────────────────────────────────
    omega_ff_history: List[float] = field(default_factory=list)       # omega_ff (feed-forward)
    omega_alpha_p_history: List[float] = field(default_factory=list)  # kp_alpha * e_alpha (alpha P)
    omega_alpha_d_history: List[float] = field(default_factory=list)  # kd_alpha * de_alpha/dt (alpha D)
    omega_tangent_history: List[float] = field(default_factory=list)  # signed k_tangent * tangential slip / R


@dataclass
class ObjectHistory:
    """Object pose / velocity, recorded every control tick."""
    times: List[float] = field(default_factory=list)
    positions: List[np.ndarray] = field(default_factory=list)
    z_positions: List[float] = field(default_factory=list)
    orientations: List[float] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    angular_velocities: List[float] = field(default_factory=list)
    desired_v_refs_body: List[np.ndarray] = field(default_factory=list)
    desired_omegas: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PyBullet / video helpers  (unchanged from test_single_pusher_diffdrive.py)
# ---------------------------------------------------------------------------

def setup_pybullet(gui: bool = True, ground_friction: float = DEFAULT_GROUND_FRICTION) -> None:
    if gui:
        pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        pyb.connect(pyb.DIRECT)
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf")
    pyb.changeDynamics(ground, -1, lateralFriction=float(ground_friction))
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=2.0, cameraYaw=0, cameraPitch=-89,
            cameraTargetPosition=[0, 0, 0],
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)


def setup_video_recording(video_path: Path, object_uid: int) -> int:
    video_path = video_path.resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path.exists():
        video_path.unlink()
    pos, _ = pyb.getBasePositionAndOrientation(object_uid)
    pyb.resetDebugVisualizerCamera(
        cameraDistance=1.75, cameraYaw=0, cameraPitch=-89,
        cameraTargetPosition=[pos[0], pos[1], 0],
    )
    log_id = pyb.startStateLogging(pyb.STATE_LOGGING_VIDEO_MP4, str(video_path))
    if log_id < 0:
        raise RuntimeError(f"Failed to start video recording (log_id={log_id})")
    print(f"[video] recording started → {video_path}")
    return log_id


def stop_video_recording(log_id: int, video_path: Path) -> None:
    if log_id < 0:
        return
    pyb.stopStateLogging(log_id)
    time.sleep(3.0)
    video_path = video_path.resolve()
    if video_path.exists():
        sz = video_path.stat().st_size / 1024 / 1024
        marker = "✓" if sz > 0 else "⚠"
        print(f"[video] {marker} {video_path} ({sz:.2f} MB)")
    else:
        print(f"[video] ✗ file not found: {video_path}")


# ---------------------------------------------------------------------------
# Phase-7 maths (module-level; ported from SinglePusherDiffdriveTemplate)
# ---------------------------------------------------------------------------

def _wrap_angle(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(x), np.cos(x)))


def _load_csv_twist_segments(
    csv_path: Path, v_speed: float
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Consecutive world poses (x,y,theta) -> constant body-twist segments at fixed ||v_body||.

    Each segment is one solve from ``solve_constant_body_twist_from_SE2`` (same as magnum CSV).
    Optional column ``t`` is ignored.

    Geometric note
    --------------
    Rows are **waypoints** in world frame. Each segment moves from pose (p_i, θ_i) to
    (p_{i+1}, θ_{i+1}) with one **constant body twist** for the solver duration ``T``.
    That motion is an SE(2) screw: path is a **straight line** in world iff ω=0; if
    ω≠0 it is generally a **circular arc**, not the straight chord from p_i to p_{i+1}.
    A polyline through the CSV xy points is therefore *not* the same as the commanded
    reference path unless every segment happens to be pure translation.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"x", "y", "theta"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"CSV must include columns {sorted(required)}; missing {sorted(missing)}"
            )
        for i, row in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "x": float(str(row["x"]).strip()),
                        "y": float(str(row["y"]).strip()),
                        "theta": float(str(row["theta"]).strip()),
                    }
                )
            except Exception as exc:
                raise ValueError(f"Invalid CSV numeric value at line {i}: {exc}") from exc
    if len(rows) < 2:
        raise ValueError("CSV needs at least 2 pose rows for segments")

    waypoints = np.array([[r["x"], r["y"], r["theta"]] for r in rows], dtype=float)

    segments: List[Dict[str, Any]] = []
    for idx in range(len(rows) - 1):
        p0 = np.array([rows[idx]["x"], rows[idx]["y"]], dtype=float)
        p1 = np.array([rows[idx + 1]["x"], rows[idx + 1]["y"]], dtype=float)
        th0 = float(rows[idx]["theta"])
        th1 = float(rows[idx + 1]["theta"])
        c, s = np.cos(th0), np.sin(th0)
        Rinv = np.array([[c, s], [-s, c]], dtype=float)
        local = Rinv @ (p1 - p0)
        dx, dy = float(local[0]), float(local[1])
        th_end_local = _wrap_angle(th1 - th0)
        v_body, omega, T = solve_constant_body_twist_from_SE2(
            dx, dy, th_end_local, v_speed=float(v_speed)
        )
        segments.append(
            {
                "segment_idx": idx,
                "v_body": np.asarray(v_body, dtype=float).reshape(2).copy(),
                "omega": float(omega),
                "T": float(T),
            }
        )
    return segments, waypoints


def _sample_constant_twist_world_com_path(
    p0: np.ndarray,
    th0: float,
    v_body: np.ndarray,
    omega: float,
    T: float,
    n: int = 96,
) -> np.ndarray:
    """World-frame CoM polyline sampling exact constant body twist over [0, T]."""
    p0 = np.asarray(p0, dtype=float).reshape(2)
    vb = np.asarray(v_body, dtype=float).reshape(2)
    w = float(omega)
    T = float(T)
    n = int(max(2, n))
    dt = T / float(n - 1)
    pts = np.zeros((n, 2), dtype=float)
    p = p0.copy()
    th = float(th0)
    pts[0] = p
    for k in range(1, n):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=float)
        p = p + (R @ vb) * dt
        th = th + w * dt
        pts[k] = p
    return pts


def _signed_cross_track_to_screw_path_m(
    q_xy: np.ndarray,
    p0_xy: np.ndarray,
    th0: float,
    v_body: np.ndarray,
    omega: float,
    T: float,
    *,
    n_samples: int = 128,
) -> float:
    """Signed lateral error (m) from q to closest point on sampled screw polyline.

    Sign: ``(t × (q - p_proj))_z`` with *t* the unit tangent along the segment at the
    closest edge (right-handed; positive means *q* lies to the left of forward *t*).
    """
    pts = _sample_constant_twist_world_com_path(
        p0_xy, th0, v_body, omega, T, n=int(max(8, n_samples))
    )
    q = np.asarray(q_xy, dtype=float).reshape(2)
    best_d = float("inf")
    best_signed = 0.0
    for k in range(int(pts.shape[0]) - 1):
        a = np.asarray(pts[k], dtype=float).reshape(2)
        b = np.asarray(pts[k + 1], dtype=float).reshape(2)
        ab = b - a
        lab2 = float(np.dot(ab, ab))
        if lab2 < 1e-18:
            continue
        s = float(np.dot(q - a, ab) / lab2)
        s = float(np.clip(s, 0.0, 1.0))
        proj = a + s * ab
        dvec = q - proj
        t = ab / max(np.sqrt(lab2), 1e-12)
        signed = float(t[0] * dvec[1] - t[1] * dvec[0])
        dist = float(np.linalg.norm(dvec))
        if dist < best_d - 1e-9:
            best_d = dist
            best_signed = signed
    return float(best_signed)


def _lowpass_angle(prev: float, new: float, alpha: float) -> float:
    """Low-pass filter an angle using wrapped shortest-path error."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return _wrap_angle(float(prev) + a * _wrap_angle(float(new) - float(prev)))


def _compute_world_cp_velocity_ref(
    object_orientation: float,
    v_ref_body: np.ndarray,
    omega_ref: float,
    r_cp_world: np.ndarray,
) -> np.ndarray:
    """World-frame contact-point velocity from a rigid-body twist + world lever arm.

    v_cp_world = R(theta) @ v_ref_b  +  omega * [-r_y, r_x]
    """
    c, s = np.cos(object_orientation), np.sin(object_orientation)
    R = np.array([[c, -s], [s, c]], dtype=float)
    v_world = R @ np.asarray(v_ref_body, dtype=float).reshape(2)
    v_rot = float(omega_ref) * np.array([-float(r_cp_world[1]), float(r_cp_world[0])], dtype=float)
    return v_world + v_rot


def _compute_body_cp_velocity(
    contact_point_body: np.ndarray,
    v_ref_body: np.ndarray,
    omega_ref: float,
) -> np.ndarray:
    """Constant body-frame CP velocity: v_cp^b = v_ref^b + omega * [-r_y, r_x]."""
    r_b = np.asarray(contact_point_body, dtype=float).reshape(2)
    return np.asarray(v_ref_body, dtype=float).reshape(2) + float(omega_ref) * np.array([-r_b[1], r_b[0]], dtype=float)


def _init_segment_reference(
    phi0: float,
    v_cp_ref_world: np.ndarray,
    omega_ref: float,
    robot_heading: float,
    branch_sign: Optional[float] = None,
) -> Dict:
    """Solve the fixed-alpha feed-forward reference for one robot.

    Equations from test_matchingvelo_report.md §5:
        a = v_cp_x + omega * R_r * sin(phi0)
        b = v_cp_y - omega * R_r * cos(phi0)
        vr_ff = ±sqrt(a² + b²),   zeta0 = atan2(b, a) [or + pi]
        alpha* = wrap(phi0 - zeta0)

    Returns
    -------
    dict with keys: vr_ff, omega_ff, zeta0, alpha_star
    """
    a = float(v_cp_ref_world[0] + omega_ref * ROBOT_RADIUS * np.sin(phi0))
    b = float(v_cp_ref_world[1] - omega_ref * ROBOT_RADIUS * np.cos(phi0))
    vr_mag = float(np.hypot(a, b))
    zeta_fwd = float(np.arctan2(b, a))
    zeta_bwd = zeta_fwd + float(np.pi)

    # Pick forward/backward branch.  In live-resolve mode this can be locked to
    # the branch chosen at realign-start; otherwise nearest-heading branch can
    # flip by pi and create discontinuous vr_ff / alpha* commands.
    if branch_sign is not None:
        use_forward = branch_sign >= 0.0
    else:
        use_forward = abs(_wrap_angle(zeta_fwd - robot_heading)) <= abs(_wrap_angle(zeta_bwd - robot_heading))

    if use_forward:
        zeta0, vr_ff = zeta_fwd, vr_mag
    else:
        zeta0, vr_ff = zeta_bwd, -vr_mag

    return {
        "vr_ff": float(vr_ff),
        "omega_ff": float(omega_ref),
        "zeta0": float(zeta0),
        "alpha_star": float(_wrap_angle(phi0 - zeta0)),
        "branch_sign": 1.0 if vr_ff >= 0.0 else -1.0,
    }


def _smooth_live_segment_reference(
    raw_ref: Dict,
    prev_ref: Optional[Dict],
    *,
    live_ref_filter_alpha: float,
    live_alpha_filter_alpha: float,
    live_alpha_hysteresis_rad: float,
) -> Dict:
    """Filter live-resolved feed-forward terms without changing branch.

    vr_ff / omega_ff are linear low-pass filtered. alpha_star is angle-filtered
    only when its wrapped delta exceeds a hysteresis band; otherwise it is held.
    This keeps live-resolve from injecting small alpha jitter every tick.
    """
    if prev_ref is None:
        return dict(raw_ref)

    ref_alpha = float(np.clip(live_ref_filter_alpha, 0.0, 1.0))
    alpha_alpha = float(np.clip(live_alpha_filter_alpha, 0.0, 1.0))
    smoothed = dict(raw_ref)
    smoothed["vr_ff"] = float(
        float(prev_ref["vr_ff"]) + ref_alpha * (float(raw_ref["vr_ff"]) - float(prev_ref["vr_ff"]))
    )
    smoothed["omega_ff"] = float(
        float(prev_ref["omega_ff"]) + ref_alpha * (float(raw_ref["omega_ff"]) - float(prev_ref["omega_ff"]))
    )
    smoothed["zeta0"] = _lowpass_angle(float(prev_ref["zeta0"]), float(raw_ref["zeta0"]), ref_alpha)

    alpha_delta = _wrap_angle(float(raw_ref["alpha_star"]) - float(prev_ref["alpha_star"]))
    if abs(alpha_delta) > float(live_alpha_hysteresis_rad):
        smoothed["alpha_star"] = _lowpass_angle(
            float(prev_ref["alpha_star"]),
            float(raw_ref["alpha_star"]),
            alpha_alpha,
        )
    else:
        smoothed["alpha_star"] = float(prev_ref["alpha_star"])

    # Preserve original branch metadata if present.  Raw live solve is already
    # branch-locked, but keeping this stable avoids accidental later flips.
    smoothed["branch_sign"] = float(prev_ref.get("branch_sign", raw_ref.get("branch_sign", 1.0)))
    return smoothed


def _plot_push_start_idx(times: List[float], t_push_start: Optional[float]) -> int:
    """First history index at push phase (skips approach/realign contact spikes)."""
    if not times or t_push_start is None:
        return 0
    return max(0, int(np.searchsorted(np.asarray(times, dtype=float), float(t_push_start))))


def _clip_series_percentile(
    data: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Clip finite samples to percentiles for clearer diagnostic plots."""
    arr = np.asarray(data, dtype=float).reshape(-1)
    out = arr.copy()
    mask = np.isfinite(out)
    if not np.any(mask):
        return out
    valid = out[mask]
    if valid.size < 2:
        return out
    lo, hi = np.percentile(valid, [lower_percentile, upper_percentile])
    out[mask] = np.clip(valid, lo, hi)
    return out


def _set_pruned_plot_ylim(
    ax,
    series_list,
    q_low: float = 2.0,
    q_high: float = 98.0,
) -> None:
    """Percentile y-limits so rare spikes do not hide steady-state behavior."""
    finite_chunks = []
    for series in series_list:
        arr = np.asarray(series, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_chunks.append(arr)
    if not finite_chunks:
        return
    vals = np.concatenate(finite_chunks)
    if vals.size < 4:
        return
    lo, hi = np.percentile(vals, [q_low, q_high])
    lo = min(float(lo), 0.0)
    hi = max(float(hi), 0.0)
    span = hi - lo
    if span <= 1e-9:
        pad = max(abs(hi), 1.0) * 0.05
        ax.set_ylim(lo - pad, hi + pad)
        return
    margin = 0.15 * span
    ax.set_ylim(lo - margin, hi + margin)


def _compute_phase7_command(
    seg_ref: Dict,
    robot_heading: float,
    position_error: np.ndarray,
    current_alpha: float,
    n_in_world: np.ndarray,
    tangent_world: np.ndarray,
    k_normal: float,
    k_tangent: float,
    kp_alpha: float,
    max_v_r: float,
    max_omega_r: float,
    kd_alpha: float = 0.0,
    e_alpha_prev: float = 0.0,
    kd_normal: float = 0.0,
    e_normal_prev: float = 0.0,
    kd_tangent: float = 0.0,
    e_tangent_prev: float = 0.0,
    tangent_authority_deadband: float = 0.25,
    tangent_error_deadband: float = 0.0,
    max_normal_speed: float = 0.08,
    max_tangent_omega: float = 0.4,
    compression_relax_gain: float = 0.0,
    max_compression_relax: float = 0.5,
    compression_relax_deadband: float = 0.002,
    z_lift: float = 0.0,
    z_relax_threshold: float = 0.003,
    z_relax_gain: float = 0.0,
    max_z_relax: float = 0.8,
    dt: float = 0.01,
) -> Tuple[float, float, Dict]:
    """Diff-drive contact controller built around the exact matching solve.

    The feed-forward terms (vr_ff, omega_ff, alpha*) come from the closed-form
    matching equations.  Feedback is split by contact-frame physics:
      * normal gap -> forward speed correction
      * tangential slip / alpha error -> angular speed correction
    This avoids treating arbitrary 2-D position error as a holonomic velocity.
    """
    drive_dir = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
    n_in = np.asarray(n_in_world, dtype=float).reshape(2)
    tangent = np.asarray(tangent_world, dtype=float).reshape(2)
    pos_err = np.asarray(position_error, dtype=float).reshape(2)

    e_normal = float(np.dot(pos_err, n_in))
    e_tangent = float(np.dot(pos_err, tangent))
    normal_drive_projection = float(np.dot(n_in, drive_dir))

    v_normal_raw = float(k_normal * e_normal * normal_drive_projection)
    if max_normal_speed > 0.0:
        v_base = float(np.clip(v_normal_raw, -max_normal_speed, max_normal_speed))
    else:
        v_base = v_normal_raw

    e_normal_dot = float((e_normal - e_normal_prev) / dt)
    v_pos_d_raw = float(kd_normal * e_normal_dot * normal_drive_projection)
    if max_normal_speed > 0.0:
        v_pos_d = float(np.clip(v_pos_d_raw, -max_normal_speed, max_normal_speed))
    else:
        v_pos_d = v_pos_d_raw

    # Direct CP velocity P is intentionally removed from v_r.  Object-speed error
    # should modify the effective object twist before the exact DD solve, not be
    # added afterward as an unrelated scalar.
    speed_err = 0.0
    v_speed_p = 0.0

    vr_ff_nominal = float(seg_ref["vr_ff"])
    compression = max(0.0, -e_normal - max(0.0, compression_relax_deadband))
    compression_relax_fraction = 0.0
    z_relax_fraction = 0.0
    vr_ff = vr_ff_nominal
    if compression > 0.0 and vr_ff_nominal * normal_drive_projection > 0.0:
        compression_relax_fraction = float(np.clip(
            compression_relax_gain * compression,
            0.0,
            max(0.0, max_compression_relax),
        ))

    # "Cheating detector" for simulation: if the object starts lifting in z,
    # reduce inward feed-forward before the contact wedge escalates.  This is
    # intentionally diagnostic/control experimental, not a real force sensor.
    z_excess = max(0.0, float(z_lift) - max(0.0, z_relax_threshold))
    if z_excess > 0.0 and vr_ff_nominal * normal_drive_projection > 0.0:
        z_relax_fraction = float(np.clip(
            z_relax_gain * z_excess,
            0.0,
            max(0.0, max_z_relax),
        ))

    relax_fraction = max(compression_relax_fraction, z_relax_fraction)
    if relax_fraction > 0.0:
        vr_ff = float(vr_ff_nominal * (1.0 - relax_fraction))
    v_relax = float(vr_ff - vr_ff_nominal)
    v_r = float(np.clip(vr_ff + v_base + v_speed_p + v_pos_d, -max_v_r, max_v_r))

    alpha_star = float(seg_ref["alpha_star"])
    e_alpha = float(np.arctan2(
        np.sin(current_alpha - alpha_star),
        np.cos(current_alpha - alpha_star),
    ))
    omega_ff = float(seg_ref["omega_ff"])
    omega_alpha_p = float(kp_alpha * e_alpha)

    # Derivative term: wrap the delta to handle angle discontinuities.
    e_alpha_dot = float(np.arctan2(
        np.sin(e_alpha - e_alpha_prev),
        np.cos(e_alpha - e_alpha_prev),
    )) / dt
    omega_alpha_d = float(kd_alpha * e_alpha_dot)
    # A fixed +/- sign on e_tangent is not valid across all contacts.  Positive
    # omega reduces alpha; the resulting tangential center-motion correction has
    # local sign sign(v_r * cos(alpha)).  Keep that sign, but make it a continuous
    # authority gate so near-90deg passive/tangential contacts do not chatter.
    tangent_authority = float(vr_ff * normal_drive_projection)
    tangent_normal_ratio = (
        float(tangent_authority / max(abs(vr_ff), 1e-9))
        if abs(vr_ff) > 1e-9
        else 0.0
    )
    authority_deadband = float(np.clip(abs(tangent_authority_deadband), 0.0, 0.95))
    abs_authority = abs(tangent_normal_ratio)
    if abs_authority <= authority_deadband:
        tangent_authority_gate = 0.0
    else:
        tangent_authority_gate = float(
            np.sign(tangent_normal_ratio)
            * (abs_authority - authority_deadband)
            / max(1.0 - authority_deadband, 1e-9)
        )

    e_tangent_dot = float((e_tangent - e_tangent_prev) / dt)
    e_tangent_cmd = 0.0 if abs(e_tangent) < abs(tangent_error_deadband) else e_tangent
    omega_tangent_raw = float(
        tangent_authority_gate
        * (
            k_tangent * e_tangent_cmd
            + kd_tangent * e_tangent_dot
        )
        / max(ROBOT_RADIUS, 1e-6)
    )
    omega_tangent = float(np.clip(
        omega_tangent_raw,
        -max_tangent_omega,
        max_tangent_omega,
    ))

    omega_r = float(np.clip(
        omega_ff + omega_alpha_p + omega_alpha_d + omega_tangent,
        -max_omega_r,
        max_omega_r,
    ))

    return v_r, omega_r, {
        # v_r decomposition
        "vr_ff": vr_ff, "v_base": v_base, "v_speed_p": v_speed_p, "v_pos_d": v_pos_d,
        # omega_r decomposition
        "omega_ff": omega_ff, "omega_alpha_p": omega_alpha_p,
        "omega_alpha_d": omega_alpha_d, "omega_tangent": omega_tangent,
        # raw errors / signals
        "e_alpha": e_alpha, "e_alpha_dot": e_alpha_dot,
        "e_pos": e_normal, "e_normal": e_normal, "e_normal_dot": e_normal_dot,
        "e_tangent": e_tangent, "e_tangent_dot": e_tangent_dot, "speed_err": speed_err,
        "vr_ff_nominal": vr_ff_nominal, "v_relax": v_relax,
        "tangent_control_sign": float(np.sign(tangent_authority_gate)) if tangent_authority_gate != 0.0 else 0.0,
        "tangent_authority_gate": tangent_authority_gate,
        "tangent_normal_ratio": tangent_normal_ratio,
        "compression": compression, "relax_fraction": relax_fraction,
        "compression_relax_fraction": compression_relax_fraction,
        "z_lift": float(z_lift), "z_relax_fraction": z_relax_fraction,
        # Filled by run() after centralized contact pressure terms.
        "v_couple": 0.0, "v_comp": 0.0,
    }


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------

class MultiPusherConstantTwistDiffdrive:
    """N diff-drive robots push an object at a single constant twist.

    Parameters
    ----------
    t_params : list of float
        Contact t_param for each robot (one robot per element).
    v_ref_body : array-like, shape (2,)
        Desired object translational velocity in the object body frame (m/s).
    omega_ref : float
        Desired object angular velocity (rad/s).
    object_name : str
        Shape key into the .obj file map.
    approach_distance : float
        How far beyond the contact point to spawn each robot (m).

    Notes
    -----
    Approach phase
        SwarmHost with startup_mode="quick" drives each robot via RobotAgent
        (rotate-then-creep, same structure as test_single_pusher_diffdrive.py's
        _compute_approach_diffdrive).  The approach loop calls
        agent.compute_velocity() — which returns a holonomic (vx, vy, omega) —
        and projects it to (v_r, omega_r) for the diff-drive robot.

    Push-phase synchronization
        The outer loop tracks per-robot approach_complete[i] (agent.contact_force
        > APPROACH_CONTACT_GATE).  As soon as ALL robots are complete, the push
        phase starts simultaneously for all robots ("centralized barrier").

    Self-rotate (stop_go) phase
        Once all robots are in contact, each robot spins in place (v_r=0) to
        align its heading to the solved zeta0_i before any forward pushing.
        A centralized barrier waits for every robot to finish rotating, then
        starts a short pre-push hold, and only then fires the push phase for
        all robots simultaneously.

    Push phase
        Each robot i independently:
          1. Calls _init_segment_reference once (at realign-start) with the
             shared desired twist evaluated at the current contact geometry.
          2. On the first push tick, snaps alpha* to the actual push-entry
             contact angle (Option B from the single-pusher stop_go logic).
          3. Calls _compute_phase7_command every control tick.
        The object moves from real contact physics — no kinematic cheat.
    """

    def __init__(
        self,
        t_params: List[float],
        v_ref_body,
        omega_ref: float,
        object_name: str = DEFAULT_OBJECT_SHAPE,
        *,
        approach_distance: float = APPROACH_DISTANCE,
        object_lateral_friction: float = DEFAULT_OBJECT_FRICTION,
        bumper_contact_mu: float = 0.01,
        wheel_lateral_friction: float = DEFAULT_WHEEL_LATERAL_FRICTION,
        caster_lateral_friction: float = DEFAULT_CASTER_LATERAL_FRICTION,
        use_planar_cheat_control: bool = True,
        kp_alpha: float = 0.5,
        kd_alpha: float = 0.0,
        kp_position: float = 1.0,
        kd_pos: float = 0.0,
        k_tangent: float = 0.0,
        kd_tangent: float = 0.0,
        tangent_authority_deadband: float = 0.25,
        tangent_error_deadband: float = 0.0,
        k_couple: float = 0.0,
        kp_obj_speed: float = 1.0,
        kp_object_omega: float = 0.0,
        max_object_v_correction: float = 0.03,
        max_object_omega_correction: float = 0.05,
        max_speed_p: float = 0.03,
        speed_p_pos_gate_m: float = 0.01,
        max_normal_speed: float = 0.08,
        max_tangent_omega: float = 0.4,
        max_couple_speed: float = 0.05,
        k_force_comp: float = 0.0,
        force_comp_threshold: float = 0.5,
        force_comp_target: float = 1.0,
        max_comp_speed: float = 0.03,
        contact_gap_deadband: float = 0.002,
        compression_relax_gain: float = 0.0,
        compression_relax_deadband: float = 0.002,
        max_compression_relax: float = 0.5,
        z_relax_threshold: float = 0.003,
        z_relax_gain: float = 0.0,
        max_z_relax: float = 0.8,
        lock_live_branch: bool = True,
        live_object_servo_scale: float = 0.0,
        live_ref_filter_alpha: float = 0.25,
        live_alpha_filter_alpha: float = 0.05,
        live_alpha_hysteresis_rad: float = np.deg2rad(1.0),
        max_forward_speed: float = 0.5,
        max_omega: float = 1.2,
        kp_realign_heading: float = 3.5,
        use_live_resolve: bool = True,
        obstructing_pusher_speed_scale: float = 1.1,
        obstructing_passive_ratio: float = 0.1,
        obstructing_inflate_gap: float = 0.02,
        couple_obstructing_only: bool = True,
        use_actual_contact_clearance_cheat: bool = True,
        csv_segments: Optional[List[Dict[str, Any]]] = None,
        csv_waypoints_world: Optional[np.ndarray] = None,
        csv_segment_v_speed: float = 0.1,
        csv_replan_each_push: bool = False,
        cross_track_integrate: bool = False,
        cross_track_k: float = 4.0,
        cross_track_omega_max: float = 0.25,
    ):
        self.n_robots = len(t_params)
        assert self.n_robots >= 1, "Need at least one t_param."
        self.t_params = [float(t) for t in t_params]
        self.v_ref_body = np.asarray(v_ref_body, dtype=float).reshape(2)
        self.omega_ref = float(omega_ref)
        self.csv_segments: Optional[List[Dict[str, Any]]] = (
            list(csv_segments) if csv_segments is not None else None
        )
        if csv_waypoints_world is None:
            self.csv_waypoints_world = None
        else:
            wa = np.asarray(csv_waypoints_world, dtype=float)
            self.csv_waypoints_world = (
                wa.reshape(-1, 3).copy() if wa.size else None
            )
        self.csv_segment_v_speed = float(csv_segment_v_speed)
        self.csv_replan_each_push = bool(csv_replan_each_push)
        self.cross_track_integrate = bool(cross_track_integrate)
        self.cross_track_k = float(cross_track_k)
        self.cross_track_omega_max = float(cross_track_omega_max)

        self.kp_alpha = float(kp_alpha)
        self.kd_alpha = float(kd_alpha)
        self.kp_position = float(kp_position)
        self.kd_pos = float(kd_pos)
        self.k_tangent = float(k_tangent)
        self.kd_tangent = float(kd_tangent)
        self.tangent_authority_deadband = float(tangent_authority_deadband)
        self.tangent_error_deadband = float(tangent_error_deadband)
        self.k_couple = float(k_couple)
        self.kp_obj_speed = float(kp_obj_speed)
        self.kp_object_omega = float(kp_object_omega)
        self.max_object_v_correction = float(max_object_v_correction)
        self.max_object_omega_correction = float(max_object_omega_correction)
        self.max_speed_p = float(max_speed_p)
        self.speed_p_pos_gate_m = float(speed_p_pos_gate_m)
        self.max_normal_speed = float(max_normal_speed)
        self.max_tangent_omega = float(max_tangent_omega)
        self.max_couple_speed = float(max_couple_speed)
        self.k_force_comp = float(k_force_comp)
        self.force_comp_threshold = float(force_comp_threshold)
        self.force_comp_target = float(force_comp_target)
        self.max_comp_speed = float(max_comp_speed)
        self.contact_gap_deadband = float(contact_gap_deadband)
        self.compression_relax_gain = float(compression_relax_gain)
        self.compression_relax_deadband = float(compression_relax_deadband)
        self.max_compression_relax = float(max_compression_relax)
        self.z_relax_threshold = float(z_relax_threshold)
        self.z_relax_gain = float(z_relax_gain)
        self.max_z_relax = float(max_z_relax)
        self.lock_live_branch = bool(lock_live_branch)
        self.live_object_servo_scale = float(live_object_servo_scale)
        self.live_ref_filter_alpha = float(live_ref_filter_alpha)
        self.live_alpha_filter_alpha = float(live_alpha_filter_alpha)
        self.live_alpha_hysteresis_rad = float(live_alpha_hysteresis_rad)
        self.max_forward_speed = float(max_forward_speed)
        self.max_omega = float(max_omega)
        self.kp_realign_heading = float(kp_realign_heading)
        self.use_live_resolve = bool(use_live_resolve)
        self.obstructing_pusher_speed_scale = float(obstructing_pusher_speed_scale)
        self.obstructing_passive_ratio = float(obstructing_passive_ratio)
        self.obstructing_inflate_gap = float(obstructing_inflate_gap)
        self.couple_obstructing_only = bool(couple_obstructing_only)
        self.use_actual_contact_clearance_cheat = bool(use_actual_contact_clearance_cheat)
        self.object_lateral_friction = float(object_lateral_friction)
        self.bumper_contact_mu = float(bumper_contact_mu)
        self.use_planar_cheat_control = bool(use_planar_cheat_control)
        self._obstructing_pushers: List[bool] = [False] * self.n_robots
        self._normal_ratio_precheck: List[float] = [0.0] * self.n_robots

        # ── Object ────────────────────────────────────────────────────────────
        _obj_file_map = {
            "right_triangle": "right_triangle.obj",
            "pi": "pi.obj",
            "root": "root.obj",
            "rect": "rect.obj",
            "hourglass": "hourglass.obj",
            "meteor": "meteor.obj",
        }
        if object_name not in _obj_file_map:
            raise ValueError(f"Unknown object '{object_name}'. Available: {sorted(_obj_file_map)}")

        self.generic_object, self.object_uid = obj_to_generic(
            obj_path=_obj_file_map[object_name],
            shape_name=object_name,
            position=(0.0, 0.0, DEFAULT_OBJECT_HEIGHT),
            orientation=0.0,
            mass=1.0,
            lateral_friction=self.object_lateral_friction,
            blind_test=True,
        )

        # ── Contact geometry (per robot, body frame, fixed for the whole run) ─
        self._parameterization = ContactPointParameterization(self.generic_object)
        self._cp_body: List[np.ndarray] = []
        self._n_out_body: List[np.ndarray] = []
        self._seg_p1_body: List[np.ndarray] = []
        self._seg_p2_body: List[np.ndarray] = []
        self._desired_cp_speed: List[float] = []

        for tp in self.t_params:
            info = self._parameterization.get_contact_info(tp)
            cp_b = np.array(info["point"], dtype=float)
            n_out = np.array(info["normal_outward"], dtype=float)
            self._cp_body.append(cp_b)
            self._n_out_body.append(n_out)

            _, seg_idx, _ = self._parameterization.parameter_to_point(tp)
            self._seg_p1_body.append(
                np.array(self._parameterization.boundary_coords[seg_idx], dtype=float)
            )
            self._seg_p2_body.append(
                np.array(self._parameterization.boundary_coords[seg_idx + 1], dtype=float)
            )
            # Body-frame CP speed is constant for the entire segment.
            v_cp_b = _compute_body_cp_velocity(cp_b, self.v_ref_body, self.omega_ref)
            self._desired_cp_speed.append(float(np.linalg.norm(v_cp_b)))

        # ── Robots + agents ───────────────────────────────────────────────────
        obj_pos_3d, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
        obj_pos2d = np.array([obj_pos_3d[0], obj_pos_3d[1]], dtype=float)
        yaw0 = float(pyb.getEulerFromQuaternion(obj_orn)[2])
        c0, s0 = np.cos(yaw0), np.sin(yaw0)
        R0 = np.array([[c0, -s0], [s0, c0]], dtype=float)

        self.robots: Dict[str, object] = {}
        self.agents: Dict[str, RobotAgent] = {}
        self.robot_histories: List[RobotHistory] = []
        self._seg_refs: List[Optional[Dict]] = [None] * self.n_robots
        self._live_seg_refs: List[Optional[Dict]] = [None] * self.n_robots

        for i in range(self.n_robots):
            name = f"R_{i + 1:02d}"
            cp_b = self._cp_body[i]
            n_out_b = self._n_out_body[i]
            n_in_b = -n_out_b

            spawn_xy = R0 @ (cp_b + float(approach_distance) * n_out_b) + obj_pos2d
            n_in_world = R0 @ n_in_b
            heading = float(np.arctan2(n_in_world[1], n_in_world[0]))

            robot = create_robot(
                kinematics="diffdrive",
                model="wheel_physics",
                position=(float(spawn_xy[0]), float(spawn_xy[1])),
                orientation=heading,
                contact_mu=self.bumper_contact_mu,
                name=name,
            )
            robot.set_wheel_friction(float(wheel_lateral_friction))
            robot.set_caster_friction(float(caster_lateral_friction))
            robot.use_planar_cheat_control = self.use_planar_cheat_control

            self.robots[name] = robot
            self.robot_histories.append(RobotHistory())

            agent = RobotAgent(
                robot=robot,
                name=name,
                object_uid=self.object_uid,
                generic_object=self.generic_object,
                navigation_type="apf",
                pushing_type="velocity",
                force_distributor=None,
            )
            self.agents[name] = agent

        # ── SwarmHost — approach coordination ─────────────────────────────────
        # position_threshold > ROBOT_RADIUS so that robots already touching the
        # object boundary are considered "at target" in the WAITING → PUSHING gate.
        self.host = SwarmHost(
            robot_agents=self.agents,
            object_uid=self.object_uid,
            generic_object=self.generic_object,
            position_threshold=0.15,
            contact_force_threshold=0.5,
            startup_mode="quick",
        )
        target_map = {f"R_{i + 1:02d}": self.t_params[i] for i in range(self.n_robots)}
        self.host.assign_targets(target_map)

        self.object_history = ObjectHistory()
        self._t_push_start: Optional[float] = None   # updated by run(); used by plot_results()
        self._csv_segment_plot_log: List[Dict[str, Any]] = []

    def _replan_csv_segment_inplace(self, seg_idx: int, obj_pos: np.ndarray, obj_theta: float) -> None:
        """Re-solve constant twist from measured world pose to CSV waypoint ``seg_idx + 1``.

        Mutates ``self.csv_segments[seg_idx]`` (``v_body``, ``omega``, ``T``).
        """
        if self.csv_waypoints_world is None or self.csv_segments is None:
            return
        if seg_idx < 0 or seg_idx >= len(self.csv_segments):
            return
        w = self.csv_waypoints_world
        p0 = np.asarray(obj_pos, dtype=float).reshape(2)
        th0 = float(obj_theta)
        p1 = w[seg_idx + 1, :2]
        th1 = float(w[seg_idx + 1, 2])
        c, s = np.cos(th0), np.sin(th0)
        Rinv = np.array([[c, s], [-s, c]], dtype=float)
        local = Rinv @ (p1 - p0)
        dx, dy = float(local[0]), float(local[1])
        th_end_local = _wrap_angle(th1 - th0)
        v_body, omega, T = solve_constant_body_twist_from_SE2(
            dx, dy, th_end_local, v_speed=float(self.csv_segment_v_speed)
        )
        seg = self.csv_segments[seg_idx]
        seg["v_body"] = np.asarray(v_body, dtype=float).reshape(2).copy()
        seg["omega"] = float(omega)
        seg["T"] = float(T)

    def _append_csv_segment_completion_log(
        self,
        seg_idx_completed: int,
        t_sim: float,
        obj_pos: np.ndarray,
        obj_theta: float,
        solver_T: float,
        log_list: List[Dict[str, Any]],
        *,
        push_start_xy: np.ndarray,
        push_start_theta: float,
        push_duration_s: float,
        stop_reason: str,
        v_body: np.ndarray,
        omega_exec: float,
        time_gate_s: Optional[float] = None,
    ) -> None:
        """Record segment end; fields support CSV trajectory overlay (measured push starts)."""
        if self.csv_waypoints_world is None:
            return
        w = self.csv_waypoints_world
        tgt = w[seg_idx_completed + 1]
        pos_err = float(np.linalg.norm(np.asarray(obj_pos, dtype=float).reshape(2) - tgt[:2]))
        yaw_err = float(abs(_wrap_angle(float(obj_theta) - float(tgt[2]))))
        vb = np.asarray(v_body, dtype=float).reshape(2)
        tg = float(time_gate_s) if time_gate_s is not None else float(solver_T)
        entry = {
            "segment_completed": int(seg_idx_completed),
            "t_s": float(t_sim),
            "solver_T_s": float(solver_T),
            "time_gate_s": tg,
            "push_start_xy_m": [float(push_start_xy[0]), float(push_start_xy[1])],
            "push_start_theta_rad": float(push_start_theta),
            "push_duration_s": float(push_duration_s),
            "stop_reason": str(stop_reason),
            "v_body": [float(vb[0]), float(vb[1])],
            "omega": float(omega_exec),
            "target_xy_m": [float(tgt[0]), float(tgt[1])],
            "target_theta_rad": float(tgt[2]),
            "actual_xy_m": [float(obj_pos[0]), float(obj_pos[1])],
            "actual_theta_rad": float(obj_theta),
            "pos_err_m": pos_err,
            "yaw_err_rad": yaw_err,
        }
        log_list.append(entry)
        print(
            f"[segment-csv] end of segment {seg_idx_completed} ({stop_reason}): "
            f"||Δpos||={pos_err:.4f} m  |Δθ|={np.degrees(yaw_err):.2f}° "
            f"(CSV waypoint row {seg_idx_completed + 2})  "
            f"push_s={push_duration_s:.3f}  solver_T={solver_T:.4f}s  T_gate={tg:.4f}s"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _robot_names(self) -> List[str]:
        return [f"R_{i + 1:02d}" for i in range(self.n_robots)]

    def _get_object_state(self):
        """Return (pos2d, z, theta, vel2d, omega) for the pushed object."""
        p3, orn = pyb.getBasePositionAndOrientation(self.object_uid)
        vl, va = pyb.getBaseVelocity(self.object_uid)
        pos2d = np.array([p3[0], p3[1]], dtype=float)
        z = float(p3[2])
        theta = float(pyb.getEulerFromQuaternion(orn)[2])
        vel2d = np.array([vl[0], vl[1]], dtype=float)
        omega = float(va[2])
        return pos2d, z, theta, vel2d, omega

    def _update_contact_geometry_from_robot_pose(
        self,
        i: int,
        obj_pos: np.ndarray,
        obj_theta: float,
        robot_pos2: np.ndarray,
        max_boundary_shift_m: float,
    ) -> float:
        """Project the landed robot pose back to the boundary and update CP geometry.

        This is intentionally bounded: the approach phase may land a few cm away
        from the nominal t_param after a transition, but a large projection jump
        would silently reassign the robot to another side/corner.
        """
        c_th, s_th = np.cos(obj_theta), np.sin(obj_theta)
        R_obj_T = np.array([[c_th, s_th], [-s_th, c_th]], dtype=float)
        robot_body = R_obj_T @ (robot_pos2 - obj_pos)
        projected = self._parameterization.point_to_parameter(robot_body)

        old_t = float(self.t_params[i])
        raw_t = float(projected["parameter"]) % 1.0
        raw_delta_t = ((raw_t - old_t + 0.5) % 1.0) - 0.5
        max_delta_t = max(0.0, float(max_boundary_shift_m)) / max(
            float(self._parameterization.total_length),
            1e-9,
        )
        delta_t = float(np.clip(raw_delta_t, -max_delta_t, max_delta_t))
        new_t = (old_t + delta_t) % 1.0

        info = self._parameterization.get_contact_info(new_t)
        cp_b = np.array(info["point"], dtype=float)
        n_out_b = np.array(info["normal_outward"], dtype=float)
        _, seg_idx, _ = self._parameterization.parameter_to_point(new_t)

        self.t_params[i] = new_t
        self._cp_body[i] = cp_b
        self._n_out_body[i] = n_out_b
        self._seg_p1_body[i] = np.array(
            self._parameterization.boundary_coords[seg_idx],
            dtype=float,
        )
        self._seg_p2_body[i] = np.array(
            self._parameterization.boundary_coords[seg_idx + 1],
            dtype=float,
        )
        self._desired_cp_speed[i] = float(np.linalg.norm(
            _compute_body_cp_velocity(cp_b, self.v_ref_body, self.omega_ref)
        ))
        return abs(delta_t) * float(self._parameterization.total_length)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        duration: float = 20.0,
        gui: bool = True,
        debug_vel: bool = False,
        debug_every: int = 50,
        save_dir: Optional[Path] = None,
        record_video: bool = False,
        align_heading_tol_rad: float = np.deg2rad(2.0),
        stop_go_sleep_after_realign_s: float = 0.5,
        test_transition: bool = False,
        transition_teleport_robots: bool = False,
        stage_position_tol: float = 0.02,
        stage_heading_tol_rad: float = np.deg2rad(5.0),
        kp_stage_position: float = 1.5,
        kp_stage_heading: float = 3.0,
        kd_stage_heading: float = 0.8,
        max_stage_omega: float = 0.4,
        update_contact_on_realign: bool = True,
        contact_update_max_distance: float = 0.06,
        csv_segment_time_only: bool = True,
        csv_segment_pos_tol_m: float = 0.045,
        csv_segment_yaw_tol_rad: float = np.deg2rad(10.0),
        csv_segment_vel_tol_m_s: float = 0.03,
        csv_segment_omega_tol_rad_s: float = 0.15,
        csv_segment_require_low_speed: bool = True,
        csv_segment_timeout_factor: float = 6.0,
        csv_segment_timeout_min_s: float = 5.0,
        csv_segment_time_scale: float = 1.0,
    ) -> Dict:
        """Run the simulation.

        Returns
        -------
        dict
            Summary metrics: push_started_at_s, push_duration_s,
            mean_position_errors_m (list, one per robot).
        """
        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        names = self._robot_names()

        # ── Phase-tracking flags ───────────────────────────────────────────────
        # STAGE → APPROACH → (all in contact) → REALIGN → HOLD → PUSH
        # Initial spawn starts at STAGE already; transitions must drive there.
        stage_complete = [False] * self.n_robots
        stage_announced = False
        all_staged = False
        approach_complete = [False] * self.n_robots
        realign_announced = False          # printed once when all approach done
        realign_complete = [False] * self.n_robots
        all_in_contact = False
        all_realigned = False
        pre_push_hold_until_t: Optional[float] = None
        push_started = False
        t_push_start: Optional[float] = None
        z_push_start: Optional[float] = None
        self._t_push_start = None   # reset each run()
        self._live_seg_refs = [None] * self.n_robots
        push_start_times: List[float] = []
        transition_done = False
        transition_time = 0.5 * float(duration)
        csv_use = bool(self.csv_segments)
        csv_seg_idx = 0
        csv_segment_push_start_t: Optional[float] = None
        csv_exit_after_final = False
        csv_segment_end_log: List[Dict[str, Any]] = []
        csv_push_start_xy = np.zeros(2, dtype=float)
        csv_push_start_theta = 0.0
        csv_push_start_valid = False
        ct_ref_xy = np.zeros(2, dtype=float)
        ct_ref_theta = 0.0
        ct_ref_valid = False

        # Per-robot state for the derivative terms and fixed-ref alpha* snap.
        e_alpha_prev_list = [0.0] * self.n_robots
        e_pos_prev_list   = [0.0] * self.n_robots  # for position D term
        e_tangent_prev_list = [0.0] * self.n_robots
        alpha_snapped = [False] * self.n_robots   # only used when use_live_resolve=False

        def _teleport_robots_to_current_approach_pose(obj_pos_now: np.ndarray, obj_theta_now: float) -> None:
            """Simulation-only transition cheat: respawn robots near current contacts."""
            c_now, s_now = np.cos(obj_theta_now), np.sin(obj_theta_now)
            R_now = np.array([[c_now, -s_now], [s_now, c_now]], dtype=float)
            for i, name in enumerate(names):
                robot = self.robots[name]
                cp_b = self._cp_body[i]
                n_out_b = self._n_out_body[i]
                n_in_b = -n_out_b
                spawn_xy = R_now @ (cp_b + APPROACH_DISTANCE * n_out_b) + obj_pos_now
                n_in_world = R_now @ n_in_b
                heading = float(np.arctan2(n_in_world[1], n_in_world[0]))
                if hasattr(robot, "reset"):
                    robot.reset(position=(float(spawn_xy[0]), float(spawn_xy[1])), orientation=heading)
                else:
                    robot.command_velocity(np.array([0.0, 0.0]))
            print("[transition] teleported robots to current contact + normal approach offset")

        def _reset_startup_barriers(reason: str) -> None:
            """Return the local scheduler to APPROACH -> REALIGN -> HOLD."""
            nonlocal stage_complete, stage_announced, all_staged
            nonlocal approach_complete, realign_announced, realign_complete
            nonlocal all_in_contact, all_realigned, pre_push_hold_until_t
            nonlocal push_started, z_push_start
            nonlocal e_alpha_prev_list, e_pos_prev_list, e_tangent_prev_list, alpha_snapped
            nonlocal ct_ref_valid

            stage_complete = [False] * self.n_robots
            stage_announced = False
            all_staged = False
            approach_complete = [False] * self.n_robots
            realign_announced = False
            realign_complete = [False] * self.n_robots
            all_in_contact = False
            all_realigned = False
            pre_push_hold_until_t = None
            push_started = False
            z_push_start = None
            self._seg_refs = [None] * self.n_robots
            self._live_seg_refs = [None] * self.n_robots
            self._obstructing_pushers = [False] * self.n_robots
            self._normal_ratio_precheck = [0.0] * self.n_robots
            e_alpha_prev_list = [0.0] * self.n_robots
            e_pos_prev_list = [0.0] * self.n_robots
            e_tangent_prev_list = [0.0] * self.n_robots
            alpha_snapped = [False] * self.n_robots
            ct_ref_valid = False
            for robot in self.robots.values():
                robot.command_velocity(np.array([0.0, 0.0]))
            self.host.assign_targets({name: self.t_params[i] for i, name in enumerate(names)})
            print(f"\n{'='*60}")
            print(f"[transition] {reason}: stopped push and reset APPROACH/REALIGN barriers")

        # Video
        video_log_id = -1
        video_path: Optional[Path] = None
        if record_video:
            if not gui:
                print("[video] --record-video ignored: requires GUI mode.")
            elif save_dir is None:
                raise ValueError("--record-video requires --save-dir.")
            else:
                save_dir.mkdir(parents=True, exist_ok=True)
                video_path = save_dir / "multi_pusher_dd_topview.mp4"
                video_log_id = setup_video_recording(video_path, self.object_uid)

        for _ in range(n_steps):
            if step_count % CTRL_STEP == 0:
                obj_pos, obj_z, obj_theta, obj_vel, obj_omega = self._get_object_state()
                obj_state_dict = {
                    "position": obj_pos,
                    "orientation": obj_theta,
                    "velocity": obj_vel,
                    "angular_velocity": obj_omega,
                }

                if (
                    test_transition
                    and (not csv_use)
                    and (not transition_done)
                    and t >= transition_time
                ):
                    old_v_ref = self.v_ref_body.copy()
                    old_omega_ref = float(self.omega_ref)
                    self.v_ref_body = np.array(
                        [-old_v_ref[1], old_v_ref[0]],
                        dtype=float,
                    )
                    self.omega_ref = -old_omega_ref
                    self._desired_cp_speed = [
                        float(np.linalg.norm(_compute_body_cp_velocity(cp_b, self.v_ref_body, self.omega_ref)))
                        for cp_b in self._cp_body
                    ]
                    transition_done = True
                    if transition_teleport_robots:
                        _teleport_robots_to_current_approach_pose(obj_pos, obj_theta)
                    _reset_startup_barriers(
                        "desired twist changed "
                        f"v=({old_v_ref[0]:+.4f}, {old_v_ref[1]:+.4f}) -> "
                        f"({self.v_ref_body[0]:+.4f}, {self.v_ref_body[1]:+.4f}) m/s, "
                        f"omega={old_omega_ref:+.4f} -> {self.omega_ref:+.4f} rad/s"
                    )

                if (
                    csv_use
                    and push_started
                    and csv_segment_push_start_t is not None
                    and self.csv_segments is not None
                    and self.csv_waypoints_world is not None
                ):
                    seg = self.csv_segments[csv_seg_idx]
                    tgt_row = self.csv_waypoints_world[csv_seg_idx + 1]
                    dt_seg = float(t - csv_segment_push_start_t)
                    T_sol = float(seg["T"])
                    T_gate = T_sol * float(csv_segment_time_scale)
                    T_cap = max(
                        float(csv_segment_timeout_factor) * T_gate,
                        float(csv_segment_timeout_min_s),
                    )
                    pos_err_gate = float(
                        np.linalg.norm(obj_pos - np.asarray(tgt_row[:2], dtype=float))
                    )
                    yaw_err_gate = float(
                        abs(_wrap_angle(float(obj_theta) - float(tgt_row[2])))
                    )
                    vel_norm = float(np.linalg.norm(obj_vel))

                    seg_done = False
                    stop_reason = ""
                    if csv_segment_time_only:
                        if dt_seg >= T_gate - 1e-9:
                            seg_done = True
                            stop_reason = "time_open_loop"
                    elif dt_seg >= T_cap:
                        seg_done = True
                        stop_reason = "timeout"
                    else:
                        pose_ok = (
                            pos_err_gate <= float(csv_segment_pos_tol_m)
                            and yaw_err_gate <= float(csv_segment_yaw_tol_rad)
                        )
                        if csv_segment_require_low_speed:
                            pose_ok = pose_ok and (
                                vel_norm <= float(csv_segment_vel_tol_m_s)
                                and abs(float(obj_omega))
                                <= float(csv_segment_omega_tol_rad_s)
                            )
                        if pose_ok:
                            seg_done = True
                            stop_reason = "pose"

                    if seg_done:
                        if not csv_push_start_valid:
                            csv_push_start_xy = np.asarray(obj_pos, dtype=float).reshape(2).copy()
                            csv_push_start_theta = float(obj_theta)
                        v_body_snap = np.asarray(seg["v_body"], dtype=float).reshape(2).copy()
                        omega_snap = float(seg["omega"])
                        self._append_csv_segment_completion_log(
                            csv_seg_idx,
                            t,
                            obj_pos,
                            obj_theta,
                            T_sol,
                            csv_segment_end_log,
                            push_start_xy=csv_push_start_xy,
                            push_start_theta=csv_push_start_theta,
                            push_duration_s=dt_seg,
                            stop_reason=stop_reason,
                            v_body=v_body_snap,
                            omega_exec=omega_snap,
                            time_gate_s=T_gate,
                        )
                        csv_push_start_valid = False
                        if csv_seg_idx + 1 < len(self.csv_segments):
                            prev_idx = csv_seg_idx
                            csv_seg_idx += 1
                            if self.csv_replan_each_push:
                                self._replan_csv_segment_inplace(
                                    csv_seg_idx, obj_pos, obj_theta
                                )
                            nxt = self.csv_segments[csv_seg_idx]
                            self.v_ref_body = np.asarray(
                                nxt["v_body"], dtype=float
                            ).reshape(2).copy()
                            self.omega_ref = float(nxt["omega"])
                            self._desired_cp_speed = [
                                float(
                                    np.linalg.norm(
                                        _compute_body_cp_velocity(
                                            cp_b, self.v_ref_body, self.omega_ref
                                        )
                                    )
                                )
                                for cp_b in self._cp_body
                            ]
                            if transition_teleport_robots:
                                _teleport_robots_to_current_approach_pose(obj_pos, obj_theta)
                            _reset_startup_barriers(
                                f"CSV segment {prev_idx} -> {csv_seg_idx} "
                                f"({stop_reason}): "
                                f"v=({self.v_ref_body[0]:+.4f}, {self.v_ref_body[1]:+.4f}) m/s, "
                                f"omega={self.omega_ref:+.4f} rad/s "
                                f"(solver T_nom={T_sol:.3f}s)"
                            )
                            csv_segment_push_start_t = None
                        else:
                            self.v_ref_body = np.array([0.0, 0.0], dtype=float)
                            self.omega_ref = 0.0
                            self._desired_cp_speed = [
                                float(
                                    np.linalg.norm(
                                        _compute_body_cp_velocity(
                                            cp_b, self.v_ref_body, self.omega_ref
                                        )
                                    )
                                )
                                for cp_b in self._cp_body
                            ]
                            csv_segment_push_start_t = None
                            csv_exit_after_final = True
                            print(
                                f"\n[segment-csv] final segment {csv_seg_idx} complete "
                                f"({stop_reason}) — zeroing reference; ending run."
                            )

                # ── Swarm update: manages approach state machine + agent goals ──
                # This calls agent.update_contact_state() for each robot, keeping
                # agent.contact_force / agent.in_contact fresh.
                self.host.update(1.0 / CTRL_FREQ, obj_state_dict)

                all_staged = all(stage_complete)
                if all_staged and not stage_announced:
                    stage_announced = True
                    print(
                        f"\n[stage] ALL {self.n_robots} ROBOTS AT APPROACH STAGING POSES "
                        f"(t={t:.2f}s) — starting contact approach"
                    )

                # Approach completion is only valid after staging.  During STAGE
                # a robot may still brush the object while clearing to its
                # approach pose; that must not open the realign barrier.
                if all_staged:
                    for i, name in enumerate(names):
                        if not approach_complete[i]:
                            if self.agents[name].contact_force > APPROACH_CONTACT_GATE:
                                approach_complete[i] = True
                                print(
                                    f"[{name}] contact detected "
                                    f"(F={self.agents[name].contact_force:.3f} N, t={t:.2f}s) — "
                                    f"waiting for remaining robots..."
                                )

                all_in_contact = all_staged and all(approach_complete)

                # ── Barrier 1: all in contact → announce realign ───────────────
                if all_in_contact and not realign_announced:
                    if update_contact_on_realign and contact_update_max_distance > 0.0:
                        shifts = []
                        for i, name in enumerate(names):
                            robot_pos3, _, _ = self.robots[name].get_state()
                            robot_pos2 = np.asarray(robot_pos3, dtype=float)[:2]
                            shift_m = self._update_contact_geometry_from_robot_pose(
                                i=i,
                                obj_pos=obj_pos,
                                obj_theta=obj_theta,
                                robot_pos2=robot_pos2,
                                max_boundary_shift_m=float(contact_update_max_distance),
                            )
                            shifts.append(shift_m)
                        self.host.assign_targets({
                            name: self.t_params[i] for i, name in enumerate(names)
                        })
                        print(
                            "[contact-update] snapped realign geometry to landed robot poses "
                            f"(bounded shifts m: {[round(s, 4) for s in shifts]})"
                        )
                    realign_announced = True
                    print(f"\n{'='*60}")
                    print(
                        f"ALL {self.n_robots} ROBOTS IN CONTACT (t={t:.2f}s)"
                        f" — STARTING HEADING REALIGN"
                    )
                    print(
                        f"  desired twist (body): "
                        f"v=({self.v_ref_body[0]:+.4f}, {self.v_ref_body[1]:+.4f}) m/s  "
                        f"omega={self.omega_ref:+.4f} rad/s"
                    )

                # ── Barrier 2: all heading-aligned → start pre-push hold ───────
                if all_in_contact and not all_realigned and all(realign_complete):
                    all_realigned = True
                    pre_push_hold_until_t = t + stop_go_sleep_after_realign_s
                    print(
                        f"\n[realign] ALL {self.n_robots} ROBOTS ALIGNED (t={t:.2f}s)"
                        f" — pre-push hold {stop_go_sleep_after_realign_s:.2f}s"
                    )

                # ── Barrier 3: hold expires → push phase ──────────────────────
                if (
                    not push_started
                    and all_realigned
                    and pre_push_hold_until_t is not None
                    and t >= pre_push_hold_until_t
                ):
                    push_started = True
                    t_push_start = t
                    z_push_start = float(obj_z)
                    push_start_times.append(float(t))
                    if self._t_push_start is None:
                        self._t_push_start = t
                    print(f"\n{'='*60}")
                    print(
                        f"ALL {self.n_robots} ROBOTS — PUSH PHASE START (t={t:.2f}s)"
                        f" [segment {len(push_start_times)}]"
                    )
                    if csv_use:
                        csv_segment_push_start_t = t
                        if (
                            self.csv_replan_each_push
                            and self.csv_segments is not None
                            and len(push_start_times) == 1
                        ):
                            self._replan_csv_segment_inplace(
                                0, obj_pos, obj_theta
                            )
                            seg_now = self.csv_segments[0]
                            self.v_ref_body = np.asarray(
                                seg_now["v_body"], dtype=float
                            ).reshape(2).copy()
                            self.omega_ref = float(seg_now["omega"])
                            self._desired_cp_speed = [
                                float(
                                    np.linalg.norm(
                                        _compute_body_cp_velocity(
                                            cp_b, self.v_ref_body, self.omega_ref
                                        )
                                    )
                                )
                                for cp_b in self._cp_body
                            ]
                            print(
                                f"[segment-csv] replanned leg 0 from measured pose before push → "
                                f"v=({self.v_ref_body[0]:+.4f}, {self.v_ref_body[1]:+.4f}) m/s  "
                                f"ω={self.omega_ref:+.4f} rad/s  T={float(seg_now['T']):.4f}s"
                            )
                        csv_push_start_xy = np.asarray(obj_pos, dtype=float).reshape(2).copy()
                        csv_push_start_theta = float(obj_theta)
                        csv_push_start_valid = True
                    if self.cross_track_integrate:
                        ct_ref_xy = np.asarray(obj_pos, dtype=float).reshape(2).copy()
                        ct_ref_theta = float(obj_theta)
                        ct_ref_valid = True

                # ── Rotation matrix for this control tick ──────────────────────
                c_th, s_th = np.cos(obj_theta), np.sin(obj_theta)
                R_obj = np.array([[c_th, -s_th], [s_th, c_th]], dtype=float)

                k_ctrl = step_count // CTRL_STEP
                z_lift = 0.0 if z_push_start is None else float(obj_z - z_push_start)

                # ── Centralized object-level twist servo ──────────────────────
                # Object velocity error modifies the desired object twist BEFORE
                # the closed-form DD matching solve.  This keeps the matching
                # equations as the backbone instead of adding a scalar speed term
                # directly to each robot's v_r.
                v_ref_world = R_obj @ self.v_ref_body
                omega_cross_trim = 0.0
                if push_started and self.cross_track_integrate:
                    if (
                        csv_use
                        and self.csv_segments is not None
                        and csv_push_start_valid
                        and csv_segment_push_start_t is not None
                    ):
                        seg_ct = self.csv_segments[csv_seg_idx]
                        vb_ct = np.asarray(seg_ct["v_body"], dtype=float).reshape(2)
                        om_ct = float(seg_ct["omega"])
                        T_ct = float(seg_ct["T"])
                        e_d_m = _signed_cross_track_to_screw_path_m(
                            obj_pos,
                            ct_ref_xy,
                            float(ct_ref_theta),
                            vb_ct,
                            om_ct,
                            T_ct,
                        )
                        omega_cross_trim = float(
                            np.clip(
                                self.cross_track_k * e_d_m,
                                -self.cross_track_omega_max,
                                self.cross_track_omega_max,
                            )
                        )
                    elif not csv_use and ct_ref_valid:
                        vb_ct = np.asarray(self.v_ref_body, dtype=float).reshape(2)
                        om_ct = float(self.omega_ref)
                        T_ct = float(max(duration, 1e-3))
                        e_d_m = _signed_cross_track_to_screw_path_m(
                            obj_pos,
                            ct_ref_xy,
                            float(ct_ref_theta),
                            vb_ct,
                            om_ct,
                            T_ct,
                        )
                        omega_cross_trim = float(
                            np.clip(
                                self.cross_track_k * e_d_m,
                                -self.cross_track_omega_max,
                                self.cross_track_omega_max,
                            )
                        )
                omega_path_ref = float(self.omega_ref) + float(omega_cross_trim)
                if push_started:
                    servo_scale = self.live_object_servo_scale if self.use_live_resolve else 1.0
                    v_obj_corr = servo_scale * self.kp_obj_speed * (v_ref_world - obj_vel)
                    corr_norm = float(np.linalg.norm(v_obj_corr))
                    if self.max_object_v_correction > 0.0 and corr_norm > self.max_object_v_correction:
                        v_obj_corr = v_obj_corr * (self.max_object_v_correction / max(corr_norm, 1e-9))
                    omega_obj_corr = float(np.clip(
                        servo_scale * self.kp_object_omega * (omega_path_ref - obj_omega),
                        -self.max_object_omega_correction,
                        self.max_object_omega_correction,
                    ))
                    v_eff_world = v_ref_world + v_obj_corr
                    omega_eff = float(omega_path_ref + omega_obj_corr)
                else:
                    v_obj_corr = np.zeros(2, dtype=float)
                    omega_obj_corr = 0.0
                    v_eff_world = v_ref_world
                    omega_eff = self.omega_ref

                # ── Per-tick geometry pre-pass ────────────────────────────────
                # Coupling is a formation term, so it must be computed from a
                # consistent snapshot of ALL robot states before any command is
                # applied.  Desired formation points are the current object-frame
                # contact offsets from cached t_params, transformed into world.
                tick_robot_data: List[Dict] = []
                for i, name in enumerate(names):
                    robot = self.robots[name]
                    cp_b = self._cp_body[i]
                    n_out_b = self._n_out_body[i]

                    cp_world = R_obj @ cp_b + obj_pos
                    n_out_w = R_obj @ n_out_b
                    n_in_w = -n_out_w
                    intended_pos = cp_world + ROBOT_RADIUS * n_out_w
                    stage_pos = cp_world + float(APPROACH_DISTANCE) * n_out_w
                    couple_target_pos = intended_pos
                    if self._obstructing_pushers[i] and self.obstructing_inflate_gap > 0.0:
                        # Coupling-only virtual geometry: for obstructing robots,
                        # maintain formation around a slightly inflated object
                        # instead of the real contact boundary.
                        couple_target_pos = intended_pos + self.obstructing_inflate_gap * n_out_w

                    robot_pos3, robot_heading, robot_vel_state = robot.get_state()
                    robot_pos2 = np.asarray(robot_pos3, dtype=float)[:2]
                    robot_vel_state = np.asarray(robot_vel_state, dtype=float).reshape(-1)
                    robot_vel2 = robot_vel_state[:2]
                    robot_omega = float(robot_vel_state[2]) if robot_vel_state.size >= 3 else 0.0
                    position_error = intended_pos - robot_pos2
                    couple_position_error = couple_target_pos - robot_pos2

                    r_cp = cp_world - obj_pos
                    v_rot = obj_omega * np.array([-r_cp[1], r_cp[0]], dtype=float)
                    cp_velocity = obj_vel + v_rot
                    v_cp_ref_w = v_eff_world + omega_eff * np.array([-r_cp[1], r_cp[0]], dtype=float)
                    robot_cp_lever = cp_world - robot_pos2
                    robot_cp_velocity = robot_vel2 + robot_omega * np.array(
                        [-robot_cp_lever[1], robot_cp_lever[0]],
                        dtype=float,
                    )

                    phi = float(np.arctan2(n_in_w[1], n_in_w[0]))
                    current_alpha = float(np.arctan2(
                        np.sin(phi - robot_heading),
                        np.cos(phi - robot_heading),
                    ))

                    tangent_w = np.array([-n_in_w[1], n_in_w[0]], dtype=float)
                    drive_dir = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
                    tick_robot_data.append({
                        "agent": self.agents[name],
                        "robot": robot,
                        "hist": self.robot_histories[i],
                        "cp_b": cp_b,
                        "cp_world": cp_world,
                        "intended_pos": intended_pos,
                        "stage_pos": stage_pos,
                        "couple_target_pos": couple_target_pos,
                        "robot_pos2": robot_pos2,
                        "robot_vel2": robot_vel2,
                        "robot_omega": robot_omega,
                        "robot_heading": robot_heading,
                        "drive_dir": drive_dir,
                        "n_in_w": n_in_w,
                        "tangent_w": tangent_w,
                        "position_error": position_error,
                        "couple_position_error": couple_position_error,
                        "r_cp": r_cp,
                        "cp_velocity": cp_velocity,
                        "v_cp_ref_w": v_cp_ref_w,
                        "robot_cp_velocity": robot_cp_velocity,
                        "phi": phi,
                        "current_alpha": current_alpha,
                        "contact_force": float(self.agents[name].contact_force),
                        "object_omega": float(obj_omega),
                        "omega_eff": omega_eff,
                    })

                push_segment_alpha_ref: List[Optional[float]] = [None] * self.n_robots

                # Diff-drive contact coupling:
                # The original paper's holonomic graph term assumes each agent
                # can directly realize a world-frame acceleration.  Here each
                # robot has only a forward/backward drive direction and must
                # preserve a very specific contact angle.  Couple only the normal
                # gap/pressure signal: e_n > 0 means the robot center is outside
                # its desired bumper-contact position and should press inward.
                # The ring term compares each normal gap against neighbour gaps,
                # then projects the inward-normal bias onto the robot drive axis.
                couple_v_drive = [0.0] * self.n_robots
                comp_v_drive = [0.0] * self.n_robots
                if self.k_couple != 0.0 and self.n_robots > 1:
                    normal_gap = [
                        float(np.dot(d["couple_position_error"], d["n_in_w"]))
                        for d in tick_robot_data
                    ]
                    for i, data in enumerate(tick_robot_data):
                        if self.couple_obstructing_only and not self._obstructing_pushers[i]:
                            continue
                        prev_i = (i - 1) % self.n_robots
                        next_i = (i + 1) % self.n_robots
                        neighbour_gap = 0.5 * (normal_gap[prev_i] + normal_gap[next_i])
                        normal_bias = self.k_couple * (
                            normal_gap[i] - neighbour_gap
                        )
                        inward_drive_projection = float(np.dot(data["n_in_w"], data["drive_dir"]))
                        couple_raw = float(normal_bias * inward_drive_projection)
                        # Geometry-only anti-overcompression: do not let coupling
                        # add inward pressure when this robot is already at/inside
                        # the desired normal contact position.  This uses normal
                        # gap, not a large analog force threshold.
                        if (
                            normal_gap[i] <= self.contact_gap_deadband
                            and couple_raw * inward_drive_projection > 0.0
                        ):
                            couple_raw = 0.0
                        couple_v_drive[i] = float(np.clip(
                            couple_raw,
                            -self.max_couple_speed,
                            self.max_couple_speed,
                        ))
                if self.k_force_comp != 0.0:
                    for i, data in enumerate(tick_robot_data):
                        force_i = float(data["contact_force"])
                        normal_gap_i = float(np.dot(data["position_error"], data["n_in_w"]))
                        if (
                            force_i < self.force_comp_threshold
                            and normal_gap_i > self.contact_gap_deadband
                        ):
                            force_deficit = max(0.0, self.force_comp_target - force_i)
                            inward_drive_projection = float(np.dot(data["n_in_w"], data["drive_dir"]))
                            comp_raw = self.k_force_comp * force_deficit * inward_drive_projection
                            comp_v_drive[i] = float(np.clip(
                                comp_raw,
                                -self.max_comp_speed,
                                self.max_comp_speed,
                            ))

                # ── Per-robot command ──────────────────────────────────────────
                for i, name in enumerate(names):
                    data = tick_robot_data[i]
                    agent = data["agent"]
                    robot = data["robot"]
                    hist = data["hist"]
                    cp_b = data["cp_b"]
                    cp_world = data["cp_world"]
                    intended_pos = data["intended_pos"]
                    stage_pos = data["stage_pos"]
                    robot_pos2 = data["robot_pos2"]
                    robot_vel2 = data["robot_vel2"]
                    robot_omega_actual = float(data["robot_omega"])
                    robot_heading = data["robot_heading"]
                    position_error = data["position_error"]
                    r_cp = data["r_cp"]
                    cp_velocity = data["cp_velocity"]
                    v_cp_ref_w = data["v_cp_ref_w"]
                    robot_cp_velocity = data["robot_cp_velocity"]
                    phi = data["phi"]
                    current_alpha = data["current_alpha"]
                    contact_force = data["contact_force"]
                    n_in_w = data["n_in_w"]
                    tangent_w = data["tangent_w"]
                    object_omega_actual = float(data["object_omega"])
                    omega_eff_i = float(data["omega_eff"])

                    # ── STAGE → APPROACH → REALIGN → HOLD → PUSH ──────────────
                    if not all_staged:
                        # ── STAGE — move to contact + outward approach offset ──
                        stage_err = stage_pos - robot_pos2
                        stage_dist = float(np.linalg.norm(stage_err))
                        e_inward_heading = _wrap_angle(phi - robot_heading)

                        if not stage_complete[i]:
                            if stage_dist > stage_position_tol:
                                # Translation subphase: face and drive to the
                                # staging point.  Do not also chase inward
                                # heading here; switching headings near the
                                # target makes the robot orbit the stage point.
                                stage_heading_target = float(np.arctan2(stage_err[1], stage_err[0]))
                                e_stage_heading = _wrap_angle(stage_heading_target - robot_heading)
                                v_r = float(np.clip(
                                    kp_stage_position * stage_dist * np.cos(e_stage_heading),
                                    -self.max_forward_speed,
                                    self.max_forward_speed,
                                ))
                                # Avoid driving hard sideways when heading is poor.
                                if abs(e_stage_heading) > np.deg2rad(75.0):
                                    v_r = 0.0
                                omega_r = float(np.clip(
                                    kp_stage_heading * e_stage_heading
                                    - kd_stage_heading * robot_omega_actual,
                                    -max_stage_omega,
                                    max_stage_omega,
                                ))
                            else:
                                # Alignment subphase: hold the staging point and
                                # rotate in place to face the contact normal.
                                v_r = 0.0
                                e_stage_heading = e_inward_heading
                                omega_r = float(np.clip(
                                    kp_stage_heading * e_inward_heading
                                    - kd_stage_heading * robot_omega_actual,
                                    -max_stage_omega,
                                    max_stage_omega,
                                ))

                            if (
                                stage_dist <= stage_position_tol
                                and abs(e_inward_heading) <= stage_heading_tol_rad
                            ):
                                stage_complete[i] = True
                                print(
                                    f"[{name}] stage complete "
                                    f"(dist={stage_dist:.3f} m, "
                                    f"|heading|={abs(e_inward_heading):.3f} rad)"
                                )
                        else:
                            v_r = 0.0
                            omega_r = 0.0
                            e_stage_heading = 0.0

                        dbg = {
                            "vr_ff": 0.0, "v_base": 0.0, "v_speed_p": 0.0,
                            "v_pos_d": 0.0, "v_couple": 0.0, "v_comp": 0.0,
                            "v_relax": 0.0,
                            "omega_ff": 0.0, "omega_alpha_p": 0.0,
                            "omega_alpha_d": 0.0, "omega_tangent": 0.0,
                            "e_alpha": e_inward_heading, "e_pos": stage_dist,
                            "e_normal": 0.0, "e_tangent": 0.0,
                        }

                        if debug_vel and k_ctrl % debug_every == 0:
                            print(
                                f"[{name} t={t:.2f}s STAGE] "
                                f"dist={stage_dist:.3f} e_head={e_stage_heading:+.3f} "
                                f"v_r={v_r:+.3f} omega_r={omega_r:+.3f}"
                            )

                    elif not all_in_contact:
                        # ── APPROACH — RobotAgent handles it ───────────────────
                        other_positions = [
                            np.asarray(self.robots[n2].get_state()[0], dtype=float)[:2]
                            for n2 in names if n2 != name
                        ]
                        cmd_holo = agent.compute_velocity(obj_state_dict, other_positions)
                        # Project holonomic (vx, vy, omega) → diffdrive (v_r, omega_r).
                        v_r = float(
                            cmd_holo[0] * np.cos(robot_heading) + cmd_holo[1] * np.sin(robot_heading)
                        )
                        omega_r = float(cmd_holo[2])
                        dbg = {
                            "vr_ff": 0.0, "v_base": 0.0, "v_speed_p": 0.0,
                            "v_pos_d": 0.0, "v_couple": 0.0, "v_comp": 0.0,
                            "v_relax": 0.0,
                            "omega_ff": 0.0, "omega_alpha_p": 0.0,
                            "omega_alpha_d": 0.0, "omega_tangent": 0.0,
                            "e_alpha": 0.0, "e_pos": 0.0,
                            "e_normal": 0.0, "e_tangent": 0.0,
                        }

                        if debug_vel and k_ctrl % debug_every == 0:
                            print(
                                f"[{name} t={t:.2f}s APPROACH] "
                                f"v_r={v_r:+.3f} omega_r={omega_r:+.3f}  "
                                f"|F|={contact_force:.2f} N  goal={agent.goal_type}"
                            )

                    elif not push_started:
                        # ── REALIGN / HOLD ─────────────────────────────────────
                        # Solve segment reference once (first realign tick).
                        if self._seg_refs[i] is None:
                            seg_ref = _init_segment_reference(
                                phi0=phi,
                                v_cp_ref_world=v_cp_ref_w,
                                omega_ref=omega_eff_i,
                                robot_heading=robot_heading,
                            )
                            self._seg_refs[i] = seg_ref
                            zeta0_i = float(seg_ref["zeta0"])
                            vr_ff_i = float(seg_ref["vr_ff"])
                            move_dir_i = np.sign(vr_ff_i) * np.array(
                                [np.cos(zeta0_i), np.sin(zeta0_i)],
                                dtype=float,
                            )
                            normal_ratio_i = (
                                float(np.dot(n_in_w, move_dir_i))
                                if abs(vr_ff_i) > 1e-12
                                else 0.0
                            )
                            self._normal_ratio_precheck[i] = normal_ratio_i
                            self._obstructing_pushers[i] = (
                                normal_ratio_i < -abs(self.obstructing_passive_ratio)
                            )
                            true_alpha_i = float(np.degrees(np.arccos(
                                np.clip(normal_ratio_i, -1.0, 1.0)
                            )))
                            v_cp_b = _compute_body_cp_velocity(cp_b, self.v_ref_body, self.omega_ref)
                            print(
                                f"[{name}] seg_ref solved at realign-start:\n"
                                f"   vr_ff={seg_ref['vr_ff']:+.4f} m/s   "
                                f"omega_ff={seg_ref['omega_ff']:+.4f} rad/s\n"
                                f"   zeta0={seg_ref['zeta0']:+.4f} rad   "
                                f"alpha*={seg_ref['alpha_star']:+.4f} rad\n"
                                f"   normal_ratio={normal_ratio_i:+.3f}   "
                                f"true_alpha={true_alpha_i:.1f} deg   "
                                f"role={'obstructing-scale' if self._obstructing_pushers[i] else 'normal'}\n"
                                f"   |v_cp_body|={float(np.linalg.norm(v_cp_b)):.4f} m/s  "
                                f"(desired_cp_speed={self._desired_cp_speed[i]:.4f})"
                            )

                        zeta_target = float(self._seg_refs[i]["zeta0"])
                        e_zeta = float(
                            np.arctan2(
                                np.sin(zeta_target - robot_heading),
                                np.cos(zeta_target - robot_heading),
                            )
                        )

                        if not realign_complete[i]:
                            # Spin in place toward zeta0.
                            v_r = 0.0
                            omega_r = float(
                                np.clip(
                                    self.kp_realign_heading * e_zeta,
                                    -self.max_omega,
                                    self.max_omega,
                                )
                            )
                            if abs(e_zeta) <= align_heading_tol_rad:
                                realign_complete[i] = True
                                print(
                                    f"[{name}] realign complete "
                                    f"(|e_zeta|={abs(e_zeta):.4f} rad <= "
                                    f"{align_heading_tol_rad:.4f} rad) — holding"
                                )
                        else:
                            # Aligned; hold still and wait for the other robots.
                            v_r = 0.0
                            omega_r = 0.0

                        dbg = {
                            "vr_ff": 0.0, "v_base": 0.0, "v_speed_p": 0.0,
                            "v_pos_d": 0.0, "v_couple": 0.0, "v_comp": 0.0,
                            "v_relax": 0.0,
                            "omega_ff": 0.0, "omega_alpha_p": 0.0,
                            "omega_alpha_d": 0.0, "omega_tangent": 0.0,
                            "e_alpha": e_zeta, "e_pos": 0.0,
                            "e_normal": 0.0, "e_tangent": 0.0,
                        }

                        if debug_vel and k_ctrl % debug_every == 0:
                            phase_tag = "HOLD" if realign_complete[i] else "REALIGN"
                            print(
                                f"[{name} t={t:.2f}s {phase_tag}] "
                                f"e_zeta={e_zeta:+.4f} rad  aligned={realign_complete[i]}"
                            )

                    else:
                        # ── PUSH ──────────────────────────────────────────────
                        refresh_seg_ref_push = (
                            self.use_live_resolve
                            or (
                                self.cross_track_integrate
                                and (
                                    (
                                        csv_use
                                        and csv_push_start_valid
                                        and csv_segment_push_start_t is not None
                                    )
                                    or (not csv_use and ct_ref_valid)
                                )
                            )
                        )
                        if refresh_seg_ref_push:
                            locked_branch = None
                            if self.lock_live_branch and self._seg_refs[i] is not None:
                                locked_branch = float(self._seg_refs[i].get("branch_sign", 1.0))
                            raw_seg_ref_push = _init_segment_reference(
                                phi0=phi,
                                v_cp_ref_world=v_cp_ref_w,
                                omega_ref=omega_eff_i,
                                robot_heading=robot_heading,
                                branch_sign=locked_branch,
                            )
                            if self.use_live_resolve:
                                prev_live_ref = self._live_seg_refs[i] or self._seg_refs[i]
                                seg_ref_push = _smooth_live_segment_reference(
                                    raw_seg_ref_push,
                                    prev_live_ref,
                                    live_ref_filter_alpha=self.live_ref_filter_alpha,
                                    live_alpha_filter_alpha=self.live_alpha_filter_alpha,
                                    live_alpha_hysteresis_rad=self.live_alpha_hysteresis_rad,
                                )
                                self._live_seg_refs[i] = seg_ref_push
                            else:
                                # Fixed-ref + cross-track-integrate: refresh vr_ff /
                                # omega_ff / alpha* from current geometry every tick (no
                                # low-pass).  One-time alpha* snap at first push tick
                                # matches legacy fixed-ref entry alignment.
                                if not alpha_snapped[i]:
                                    alpha_snapped[i] = True
                                    alpha_entry = float(np.arctan2(
                                        np.sin(phi - robot_heading),
                                        np.cos(phi - robot_heading),
                                    ))
                                    alpha_star_old = float(raw_seg_ref_push["alpha_star"])
                                    raw_seg_ref_push["alpha_star"] = float(alpha_entry)
                                    print(
                                        f"[{name}] alpha* snapped (fixed-ref): "
                                        f"{alpha_entry:.4f} rad "
                                        f"(was {alpha_star_old:.4f}, "
                                        f"delta={alpha_entry - alpha_star_old:+.4f} rad)"
                                    )
                                seg_ref_push = raw_seg_ref_push
                        else:
                            if not alpha_snapped[i]:
                                alpha_snapped[i] = True
                                alpha_entry = float(np.arctan2(
                                    np.sin(phi - robot_heading),
                                    np.cos(phi - robot_heading),
                                ))
                                alpha_star_old = float(self._seg_refs[i]["alpha_star"])
                                self._seg_refs[i]["alpha_star"] = alpha_entry
                                print(
                                    f"[{name}] alpha* snapped (fixed-ref): "
                                    f"{alpha_entry:.4f} rad "
                                    f"(was {alpha_star_old:.4f}, "
                                    f"delta={alpha_entry - alpha_star_old:+.4f} rad)"
                                )
                            seg_ref_push = self._seg_refs[i]

                        push_segment_alpha_ref[i] = float(seg_ref_push["alpha_star"])
                        v_r, omega_r, dbg = _compute_phase7_command(
                            seg_ref=seg_ref_push,
                            robot_heading=robot_heading,
                            position_error=position_error,
                            current_alpha=current_alpha,
                            n_in_world=n_in_w,
                            tangent_world=tangent_w,
                            k_normal=self.kp_position,
                            k_tangent=self.k_tangent,
                            kp_alpha=self.kp_alpha,
                            max_v_r=self.max_forward_speed,
                            max_omega_r=self.max_omega,
                            kd_alpha=self.kd_alpha,
                            e_alpha_prev=e_alpha_prev_list[i],
                            kd_normal=self.kd_pos,
                            e_normal_prev=e_pos_prev_list[i],
                            kd_tangent=self.kd_tangent,
                            e_tangent_prev=e_tangent_prev_list[i],
                            tangent_authority_deadband=self.tangent_authority_deadband,
                            tangent_error_deadband=self.tangent_error_deadband,
                            max_normal_speed=self.max_normal_speed,
                            max_tangent_omega=self.max_tangent_omega,
                            compression_relax_gain=self.compression_relax_gain,
                            max_compression_relax=self.max_compression_relax,
                            compression_relax_deadband=self.compression_relax_deadband,
                            z_lift=z_lift,
                            z_relax_threshold=self.z_relax_threshold,
                            z_relax_gain=self.z_relax_gain,
                            max_z_relax=self.max_z_relax,
                            dt=1.0 / CTRL_FREQ,
                        )
                        # Advance per-robot derivative state.
                        e_alpha_prev_list[i] = dbg["e_alpha"]
                        e_pos_prev_list[i]   = dbg["e_pos"]
                        e_tangent_prev_list[i] = dbg["e_tangent"]

                        # ── Apply diff-drive normal-gap coupling ───────────────
                        couple_v_r = couple_v_drive[i]
                        if couple_v_r != 0.0:
                            v_r = float(np.clip(
                                v_r + couple_v_r,
                                -self.max_forward_speed, self.max_forward_speed,
                            ))
                        dbg["v_couple"] = couple_v_r
                        comp_v_r = comp_v_drive[i]
                        if comp_v_r != 0.0:
                            v_r = float(np.clip(
                                v_r + comp_v_r,
                                -self.max_forward_speed, self.max_forward_speed,
                            ))
                        dbg["v_comp"] = comp_v_r

                        if debug_vel and k_ctrl % debug_every == 0:
                            print(
                                f"[{name} t={t:.2f}s PUSH] "
                                f"v_r={v_r:+.3f}(ff={dbg['vr_ff']:+.3f} "
                                f"relax={dbg['v_relax']:+.3f} "
                                f"normal={dbg['v_base']:+.3f} nD={dbg['v_pos_d']:+.3f} "
                                f"couple={dbg['v_couple']:+.3f} comp={dbg['v_comp']:+.3f})  "
                                f"e_alpha={dbg['e_alpha']:+.3f} rad  "
                                f"e_n={dbg['e_normal']:+.3f} m e_t={dbg['e_tangent']:+.3f} m  "
                                f"z_lift={dbg['z_lift']:+.4f} m  "
                                f"|F|={contact_force:.2f} N"
                            )

                    if (
                        push_started
                        and self._obstructing_pushers[i]
                        and self.obstructing_pusher_speed_scale != 1.0
                    ):
                        if self.use_actual_contact_clearance_cheat:
                            # CHEATING / SIM-ONLY FIX:
                            # Use the measured object contact-point normal
                            # velocity as a one-sided clearance constraint.  It
                            # does NOT replace the coupled command unless that
                            # command is too slow to clear the live object
                            # motion, so k_couple can still pull the robot back
                            # toward the inflated formation.  A real robot may
                            # not have this clean CP velocity; practical
                            # implementations need an observer/estimator or
                            # relative-motion proxy.
                            drive_normal_projection = float(np.dot(n_in_w, data["drive_dir"]))
                            robot_normal_speed = float(v_r * drive_normal_projection)
                            object_cp_normal_speed = float(np.dot(cp_velocity, n_in_w))
                            # For obstructing contacts, more negative normal
                            # speed means faster outward clearance.  Require a
                            # margin beyond the live object CP normal speed, but
                            # only clamp when the current coupled command is less
                            # outward than that safety speed.
                            clearance_margin = (
                                (self.obstructing_pusher_speed_scale - 1.0)
                                * max(abs(object_cp_normal_speed), abs(robot_normal_speed), 1e-6)
                            )
                            clearance_target = object_cp_normal_speed - clearance_margin
                            if (
                                abs(drive_normal_projection) > 1e-6
                                and clearance_target < robot_normal_speed
                            ):
                                v_r = float(np.clip(
                                    clearance_target / drive_normal_projection,
                                    -self.max_forward_speed,
                                    self.max_forward_speed,
                                ))
                        else:
                            # Diagnostic role-based cheat: obstructing candidates
                            # have outward fixed-ref normal motion. Increase their
                            # signed speed magnitude so they clear contact instead
                            # of staying wedged in the object path.
                            v_r = float(np.clip(
                                v_r * self.obstructing_pusher_speed_scale,
                                -self.max_forward_speed,
                                self.max_forward_speed,
                            ))

                    robot.command_velocity(np.array([v_r, omega_r]))

                    # ── History ────────────────────────────────────────────────
                    if push_started and push_segment_alpha_ref[i] is not None:
                        alpha_star_i = float(push_segment_alpha_ref[i])
                    elif self._seg_refs[i] is not None:
                        alpha_star_i = float(self._seg_refs[i]["alpha_star"])
                    else:
                        alpha_star_i = 0.0
                    e_alpha_hist = float(np.arctan2(
                        np.sin(current_alpha - alpha_star_i),
                        np.cos(current_alpha - alpha_star_i),
                    ))

                    hist.times.append(t)
                    hist.robot_positions.append(robot_pos2.copy())
                    hist.robot_headings.append(float(robot_heading))
                    hist.robot_velocities.append(robot_vel2.copy())
                    hist.robot_angular_velocities.append(float(robot_omega_actual))
                    hist.intended_positions.append(intended_pos.copy())
                    hist.position_errors.append(position_error.copy())
                    hist.couple_target_positions.append(data["couple_target_pos"].copy())
                    hist.couple_position_errors.append(data["couple_position_error"].copy())
                    hist.intended_contact_point_velocities.append(v_cp_ref_w.copy())
                    hist.contact_point_velocities.append(cp_velocity.copy())
                    hist.robot_contact_point_velocities.append(robot_cp_velocity.copy())
                    hist.intended_omegas.append(float(omega_eff_i))
                    hist.object_omegas.append(float(object_omega_actual))
                    hist.contact_forces.append(float(contact_force))
                    hist.in_contact.append(contact_force > 0.5)
                    hist.alpha_errors.append(e_alpha_hist)
                    hist.v_r_history.append(float(v_r))
                    hist.omega_r_history.append(float(omega_r))
                    hist.v_ff_history.append(float(dbg["vr_ff"]))
                    hist.v_base_history.append(float(dbg["v_base"]))
                    hist.v_speed_p_history.append(float(dbg["v_speed_p"]))
                    hist.v_pos_d_history.append(float(dbg.get("v_pos_d", 0.0)))
                    hist.v_couple_history.append(float(dbg.get("v_couple", 0.0)))
                    hist.v_comp_history.append(float(dbg.get("v_comp", 0.0)))
                    hist.v_relax_history.append(float(dbg.get("v_relax", 0.0)))
                    hist.omega_ff_history.append(float(dbg["omega_ff"]))
                    hist.omega_alpha_p_history.append(float(dbg["omega_alpha_p"]))
                    hist.omega_alpha_d_history.append(float(dbg.get("omega_alpha_d", 0.0)))
                    hist.omega_tangent_history.append(float(dbg.get("omega_tangent", 0.0)))

                # ── Object history ─────────────────────────────────────────────
                self.object_history.times.append(t)
                self.object_history.positions.append(obj_pos.copy())
                self.object_history.z_positions.append(float(obj_z))
                self.object_history.orientations.append(float(obj_theta))
                self.object_history.velocities.append(obj_vel.copy())
                self.object_history.angular_velocities.append(float(obj_omega))
                self.object_history.desired_v_refs_body.append(self.v_ref_body.copy())
                self.object_history.desired_omegas.append(float(self.omega_ref))

            pyb.stepSimulation()
            if gui:
                time.sleep(TIMESTEP * 0.3)
            t += TIMESTEP
            step_count += 1
            if csv_use and csv_exit_after_final:
                print(
                    f"[segment-csv] simulation stopped early at t={t:.2f}s "
                    f"(--duration {duration:.2f}s was upper bound)."
                )
                break

        # ── Finalize ──────────────────────────────────────────────────────────
        if video_log_id >= 0 and video_path is not None:
            stop_video_recording(video_log_id, video_path)

        push_duration = float(duration - t_push_start) if t_push_start is not None else 0.0
        mean_pos_errs = []
        for hist in self.robot_histories:
            if hist.position_errors:
                errs = [float(np.linalg.norm(e)) for e in hist.position_errors]
                mean_pos_errs.append(float(np.mean(errs)))
            else:
                mean_pos_errs.append(float("nan"))

        if save_dir is not None and csv_segment_end_log:
            log_path = Path(save_dir) / "csv_segment_end_log.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(csv_segment_end_log, f, indent=2)
            print(f"[segment-csv] wrote per-segment pose error log: {log_path}")

        self._csv_segment_plot_log = list(csv_segment_end_log) if csv_use else []

        return {
            "push_started_at_s": t_push_start,
            "push_start_times_s": push_start_times,
            "transition_enabled": bool(test_transition),
            "transition_done": bool(transition_done),
            "push_duration_s": push_duration,
            "mean_position_errors_m": mean_pos_errs,
            "csv_exit_after_final": bool(csv_exit_after_final),
            "csv_segment_end_log": list(csv_segment_end_log),
        }

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_results(self, save_dir: Optional[Path] = None) -> None:
        """Three figures: trajectory, per-robot diagnostics, object velocity."""
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        names = self._robot_names()

        # ── Figure 1: trajectory (object CoM + per-robot paths) ───────────────
        if self.object_history.times:
            obj_pos_arr = np.array(self.object_history.positions)
            obj_theta_arr = np.array(self.object_history.orientations)

            fig_traj = plt.figure(figsize=(18, 10))
            gs = GridSpec(2, 3, figure=fig_traj, hspace=0.35, wspace=0.35)
            title_lines = [
                "Multi-pusher diff-drive: trajectories",
                (
                    f"v_ref=({self.v_ref_body[0]:+.3f}, {self.v_ref_body[1]:+.3f}) m/s  "
                    f"ω_ref={self.omega_ref:+.3f} rad/s"
                ),
            ]
            if self.csv_waypoints_world is not None:
                title_lines.append(
                    "CSV: violet = waypoint polyline + θ; cyan = constant-twist integrated from "
                    "**measured push-start pose** for **actual push duration** (matches reference)"
                )
            fig_traj.suptitle("\n".join(title_lines), fontsize=12)

            # Object CoM trajectory (spans left 2 columns of top row)
            ax_obj = fig_traj.add_subplot(gs[0, :2])
            wp_arrow_len = 0.12
            wxy = self.csv_waypoints_world
            if wxy is not None and wxy.size > 0:
                wxy = np.asarray(wxy, dtype=float).reshape(-1, 3)
                segs = self.csv_segments
                if wxy.shape[0] >= 2:
                    xy_span = max(
                        float(np.ptp(wxy[:, 0])),
                        float(np.ptp(wxy[:, 1])),
                        0.2,
                    )
                    wp_arrow_len = float(np.clip(0.12 * xy_span, 0.05, 0.22))
                    plot_log = getattr(self, "_csv_segment_plot_log", None) or []
                    if plot_log:
                        nominal_parts_pl: List[np.ndarray] = []
                        for ent in plot_log:
                            p0 = np.asarray(ent["push_start_xy_m"], dtype=float).reshape(2)
                            th0 = float(ent["push_start_theta_rad"])
                            dur = float(ent["push_duration_s"])
                            npts = int(max(24, min(300, 40 + int(dur * 60))))
                            nominal_parts_pl.append(
                                _sample_constant_twist_world_com_path(
                                    p0,
                                    th0,
                                    np.asarray(ent["v_body"], dtype=float).reshape(2),
                                    float(ent["omega"]),
                                    dur,
                                    n=npts,
                                )
                            )
                        nominal_xy = np.vstack(
                            [nominal_parts_pl[0]]
                            + [part[1:] for part in nominal_parts_pl[1:]]
                        )
                        ax_obj.plot(
                            nominal_xy[:, 0],
                            nominal_xy[:, 1],
                            ":",
                            color="tab:cyan",
                            lw=2.0,
                            alpha=0.9,
                            label="Ref twist (push start + duration)",
                            zorder=2,
                        )
                    elif segs is not None and len(segs) == wxy.shape[0] - 1:
                        nominal_parts: List[np.ndarray] = []
                        for si, seg in enumerate(segs):
                            p0 = wxy[si, :2]
                            th0 = float(wxy[si, 2])
                            nominal_parts.append(
                                _sample_constant_twist_world_com_path(
                                    p0,
                                    th0,
                                    seg["v_body"],
                                    float(seg["omega"]),
                                    float(seg["T"]),
                                    n=120,
                                )
                            )
                        nominal_xy = np.vstack(
                            [nominal_parts[0]]
                            + [part[1:] for part in nominal_parts[1:]]
                        )
                        ax_obj.plot(
                            nominal_xy[:, 0],
                            nominal_xy[:, 1],
                            ":",
                            color="tab:cyan",
                            lw=2.0,
                            alpha=0.9,
                            label="Nominal SE(2) from CSV rows (no run log)",
                            zorder=2,
                        )
                    ax_obj.plot(
                        wxy[:, 0],
                        wxy[:, 1],
                        "--",
                        color="tab:purple",
                        lw=1.35,
                        alpha=0.88,
                        label="CSV polyline (chords)",
                        zorder=2,
                    )
                ax_obj.scatter(
                    wxy[:, 0],
                    wxy[:, 1],
                    s=52,
                    c="white",
                    edgecolors="darkviolet",
                    linewidths=1.15,
                    zorder=8,
                    marker="D",
                    label="CSV waypoints",
                )

            ax_obj.plot(
                obj_pos_arr[:, 0], obj_pos_arr[:, 1],
                "k-", lw=2.0, label="Object CoM", alpha=0.85,
                zorder=3,
            )
            # Overlay each robot path lightly on the same axes
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
            for i, name in enumerate(names):
                rp = np.array(self.robot_histories[i].robot_positions)
                if len(rp):
                    ax_obj.plot(
                        rp[:, 0], rp[:, 1],
                        "-", lw=1.0, alpha=0.5, color=colors[i % len(colors)],
                        label=name,
                        zorder=3,
                    )
            # Heading arrows along object trajectory
            arrow_step = max(1, len(obj_pos_arr) // 20)
            for k in range(0, len(obj_pos_arr) - 1, arrow_step):
                dx = 0.04 * np.cos(obj_theta_arr[k])
                dy = 0.04 * np.sin(obj_theta_arr[k])
                ax_obj.arrow(
                    obj_pos_arr[k, 0], obj_pos_arr[k, 1], dx, dy,
                    head_width=0.015, head_length=0.010,
                    fc="red", ec="red", alpha=0.55, zorder=5,
                )

            if wxy is not None and wxy.size > 0:
                wxy = np.asarray(wxy, dtype=float).reshape(-1, 3)
                for j in range(wxy.shape[0]):
                    ax_obj.annotate(
                        "",
                        xy=(
                            float(wxy[j, 0]) + wp_arrow_len * np.cos(float(wxy[j, 2])),
                            float(wxy[j, 1]) + wp_arrow_len * np.sin(float(wxy[j, 2])),
                        ),
                        xytext=(float(wxy[j, 0]), float(wxy[j, 1])),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="darkviolet",
                            lw=2.0,
                            alpha=0.95,
                            shrinkA=0,
                            shrinkB=0,
                        ),
                        zorder=9,
                    )

            if len(obj_pos_arr):
                ax_obj.plot(*obj_pos_arr[0], "go", ms=8, label="Start", zorder=6)
                ax_obj.plot(*obj_pos_arr[-1], "rs", ms=8, label="End", zorder=6)
            ax_obj.set_xlabel("X (m)")
            ax_obj.set_ylabel("Y (m)")
            ax_obj.set_title(
                "Object CoM trajectory (+ robot paths; CSV refs if enabled)",
                fontsize=11,
            )
            ax_obj.axis("equal")
            ax_obj.grid(True, alpha=0.3)
            ax_obj.legend(fontsize=8, loc="upper right")

            # Per-robot subplots: robot path + intended (contact-point) path
            robot_subplot_slots = [(0, 2), (1, 0), (1, 1), (1, 2)]
            for i, name in enumerate(names):
                if i >= len(robot_subplot_slots):
                    break
                row, col = robot_subplot_slots[i]
                ax_r = fig_traj.add_subplot(gs[row, col])

                rp = np.array(self.robot_histories[i].robot_positions)
                ip = np.array(self.robot_histories[i].intended_positions)
                rh = np.array(self.robot_histories[i].robot_headings)

                if len(rp):
                    ax_r.plot(rp[:, 0], rp[:, 1], "b-", lw=1.8,
                              label="Robot", alpha=0.85)
                    ax_r.plot(ip[:, 0], ip[:, 1], "g--", lw=1.0,
                              label="Intended", alpha=0.7)
                    # Heading arrows
                    hstep = max(1, len(rp) // 12)
                    for k in range(0, len(rp), hstep):
                        dx = 0.025 * np.cos(rh[k])
                        dy = 0.025 * np.sin(rh[k])
                        ax_r.arrow(
                            rp[k, 0], rp[k, 1], dx, dy,
                            head_width=0.010, head_length=0.007,
                            fc="blue", ec="blue", alpha=0.45, zorder=5,
                        )
                    ax_r.plot(*rp[0], "go", ms=7, label="Start", zorder=6)
                    ax_r.plot(*rp[-1], "rs", ms=7, label="End", zorder=6)

                tp_val = self.t_params[i]
                ax_r.set_title(f"{name}  (t={tp_val:.3f})", fontsize=10)
                ax_r.set_xlabel("X (m)", fontsize=9)
                ax_r.set_ylabel("Y (m)", fontsize=9)
                ax_r.axis("equal")
                ax_r.grid(True, alpha=0.3)
                ax_r.legend(fontsize=7, loc="upper right")

            plt.tight_layout()
            if save_dir:
                p = save_dir / "multi_pusher_dd_trajectory.png"
                plt.savefig(p, dpi=150, bbox_inches="tight")
                print(f"Saved {p}")
            else:
                plt.show()
            plt.close(fig_traj)

        # ── Figure 2: per-robot diagnostics ───────────────────────────────────
        fig, axes = plt.subplots(
            self.n_robots, 4,
            figsize=(16, 3 * self.n_robots + 1),
            squeeze=False,
        )
        fig.suptitle(
            f"Multi-pusher diff-drive  N={self.n_robots}  "
            f"v_ref^b=({self.v_ref_body[0]:+.3f}, {self.v_ref_body[1]:+.3f})  "
            f"omega_ref={self.omega_ref:+.3f} rad/s",
            fontsize=10,
        )

        push_start_idx = _plot_push_start_idx(
            self.robot_histories[0].times if self.robot_histories else [],
            self._t_push_start,
        )
        force_col_title = "Contact force (N)"
        if push_start_idx > 0:
            force_col_title += " [push phase, clipped]"
        col_titles = ["Position error (cm)", force_col_title, "Alpha error (deg)", "v_r cmd (m/s)"]
        for j, title in enumerate(col_titles):
            axes[0][j].set_title(title, fontsize=9)

        for i, (name, hist) in enumerate(zip(names, self.robot_histories)):
            if not hist.times:
                continue
            t_arr = np.array(hist.times)
            pos_err_cm = np.array([np.linalg.norm(e) for e in hist.position_errors]) * 100.0
            if hist.couple_position_errors:
                couple_err_cm = np.array([np.linalg.norm(e) for e in hist.couple_position_errors]) * 100.0
            else:
                couple_err_cm = pos_err_cm
            contact_f_raw = np.asarray(hist.contact_forces, dtype=float)[push_start_idx:]
            t_force = t_arr[push_start_idx:]
            contact_f = _clip_series_percentile(contact_f_raw)
            alpha_err_deg = np.degrees(np.array(hist.alpha_errors))
            v_r_arr = np.array(hist.v_r_history)

            axes[i][0].plot(t_arr, pos_err_cm, lw=1.0, label="contact target")
            axes[i][0].plot(t_arr, couple_err_cm, lw=1.0, ls="--", label="couple/inflated target")
            axes[i][0].set_ylabel(f"{name}\n(cm)", fontsize=8)
            axes[i][0].legend(fontsize=7, loc="upper right")
            axes[i][0].grid(True, alpha=0.3)

            axes[i][1].plot(t_force, contact_f, color="tab:red", lw=1.0)
            axes[i][1].axhline(0.5, ls="--", color="gray", lw=0.8, label="0.5 N gate")
            axes[i][1].set_ylabel("(N)", fontsize=8)
            axes[i][1].grid(True, alpha=0.3)
            _set_pruned_plot_ylim(axes[i][1], [contact_f], q_low=5.0, q_high=95.0)

            axes[i][2].plot(t_arr, alpha_err_deg, color="tab:green", lw=1.0)
            axes[i][2].axhline(0.0, ls="--", color="gray", lw=0.8)
            axes[i][2].set_ylabel("(deg)", fontsize=8)
            axes[i][2].grid(True, alpha=0.3)

            axes[i][3].plot(t_arr, v_r_arr, color="tab:purple", lw=1.0)
            axes[i][3].axhline(0.0, ls="--", color="gray", lw=0.8)
            axes[i][3].set_ylabel("(m/s)", fontsize=8)
            axes[i][3].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            p = save_dir / "multi_pusher_dd_per_robot.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved {p}")
        else:
            plt.show()
        plt.close(fig)

        # ── Figure 3: per-robot controller term decomposition ─────────────────
        # Shows separately which term dominates v_r and omega_r over time.
        # Key for diagnosing alpha drift and feedforward mismatch.
        fig3, axes3 = plt.subplots(
            self.n_robots + 1, 4,
            figsize=(22, 3 * (self.n_robots + 1) + 1),
            squeeze=False,
        )
        ref_mode = "live-resolve" if self.use_live_resolve else "fixed-ref"
        fig3.suptitle(
            f"Multi-pusher diff-drive  N={self.n_robots} — controller decomposition  "
            f"[{ref_mode}{' branch-locked' if self.use_live_resolve and self.lock_live_branch else ''}  "
            f"liveα={self.live_ref_filter_alpha:.2f}/{self.live_alpha_filter_alpha:.2f}  "
            f"liveObj={self.live_object_servo_scale:.2f}  "
            f"kp_α={self.kp_alpha:.2f}  kd_α={self.kd_alpha:.3f}  "
            f"k_n={self.kp_position:.3f}  kd_n={self.kd_pos:.3f}  "
            f"k_t={self.k_tangent:.3f}  kd_t={self.kd_tangent:.3f}  "
            f"tAuthDB={self.tangent_authority_deadband:.2f}  "
            f"k_couple={self.k_couple:.3f}  "
            f"max_couple={self.max_couple_speed:.3f}]\n"
            f"v_r = vr_ff + v_normal + v_normal_D + v_couple + v_comp    "
            f"ω_r = ω_ff + kp_α·e_α + kd_α·ė_α + gate·(k_t·e_t + kd_t·ė_t)/R",
            fontsize=9,
        )
        push_start_idx = _plot_push_start_idx(
            self.robot_histories[0].times if self.robot_histories else [],
            self._t_push_start,
        )

        for i, (name, hist) in enumerate(zip(names, self.robot_histories)):
            if not hist.times:
                continue
            t_arr = np.array(hist.times)[push_start_idx:]

            def _slice(lst):
                return list(lst)[push_start_idx:]

            # ── left: v_r decomposition ────────────────────────────────────
            ax_v = axes3[i][0]
            ax_v.plot(t_arr, _slice(hist.v_ff_history),      lw=1.0, label="vr_ff (FF)")
            ax_v.plot(t_arr, _slice(hist.v_base_history),    lw=1.0, label="v_normal", ls="--")
            ax_v.plot(t_arr, _slice(hist.v_speed_p_history), lw=1.0, label="v_speed_p (removed)", ls=":")
            ax_v.plot(t_arr, _slice(hist.v_pos_d_history),   lw=1.0, label="v_normal_D", ls="-.")
            ax_v.plot(t_arr, _slice(hist.v_couple_history),  lw=1.0, label="v_couple (normal gap)",
                      ls=(0, (3, 1, 1, 1)))
            ax_v.plot(t_arr, _slice(hist.v_comp_history),    lw=1.0, label="v_comp (force)", alpha=0.8)
            ax_v.plot(t_arr, _slice(hist.v_relax_history),   lw=1.0, label="v_relax (compression)", alpha=0.8)
            ax_v.plot(t_arr, _slice(hist.v_r_history),       lw=1.3, label="v_r total", color="k")
            ax_v.axhline(0.0, ls="-", lw=0.5, color="gray")
            ax_v.set_ylabel(f"{name}\n(m/s)", fontsize=8)
            ax_v.legend(fontsize=7, loc="upper right")
            ax_v.grid(True, alpha=0.3)
            _set_pruned_plot_ylim(ax_v, [
                _slice(hist.v_ff_history),
                _slice(hist.v_base_history),
                _slice(hist.v_speed_p_history),
                _slice(hist.v_pos_d_history),
                _slice(hist.v_couple_history),
                _slice(hist.v_comp_history),
                _slice(hist.v_relax_history),
                _slice(hist.v_r_history),
            ])
            if i == 0:
                ax_v.set_title("v_r decomposition (push phase only, pruned y)", fontsize=9)

            # ── right: omega_r decomposition ───────────────────────────────
            ax_w = axes3[i][1]
            ax_w.plot(t_arr, _slice(hist.omega_ff_history),      lw=1.0, label="ω_ff (FF)")
            ax_w.plot(t_arr, _slice(hist.omega_alpha_p_history), lw=1.0, label="kp_α·e_α (P)", ls="--")
            ax_w.plot(t_arr, _slice(hist.omega_alpha_d_history), lw=1.0, label="kd_α·ė_α (D)", ls=":")
            ax_w.plot(t_arr, _slice(hist.omega_tangent_history), lw=1.0, label="gate·(k_t·e_t+kd_t·ė_t)/R", ls="-.")
            ax_w.plot(t_arr, _slice(hist.omega_r_history),       lw=1.3, label="ω_r total", color="k")
            ax_w.axhline(0.0, ls="-", lw=0.5, color="gray")
            ax_w.set_ylabel("(rad/s)", fontsize=8)
            ax_w.legend(fontsize=7, loc="upper right")
            ax_w.grid(True, alpha=0.3)
            _set_pruned_plot_ylim(ax_w, [
                _slice(hist.omega_ff_history),
                _slice(hist.omega_alpha_p_history),
                _slice(hist.omega_alpha_d_history),
                _slice(hist.omega_tangent_history),
                _slice(hist.omega_r_history),
            ])
            if i == 0:
                ax_w.set_title("ω_r decomposition (push phase only, pruned y)", fontsize=9)

            # ── center-right: velocity matching at the contact point ─────────
            # The matching report expects robot-side CP velocity to match the
            # object-side CP velocity.  Robot center velocity is shown separately
            # because it need not equal CP velocity when omega is non-zero.
            ax_match = axes3[i][2]
            robot_v = np.array(_slice(hist.robot_velocities))
            cp_ref_v = np.array(_slice(hist.intended_contact_point_velocities))
            obj_cp_v = np.array(_slice(hist.contact_point_velocities))
            robot_cp_v = np.array(_slice(hist.robot_contact_point_velocities))
            if len(t_arr) and len(robot_v) and len(cp_ref_v) and len(obj_cp_v) and len(robot_cp_v):
                ax_match.plot(t_arr, cp_ref_v[:, 0], lw=1.0, label="ref CP vx", color="tab:blue", ls="--")
                ax_match.plot(t_arr, obj_cp_v[:, 0], lw=1.0, label="obj CP vx", color="tab:blue")
                ax_match.plot(t_arr, robot_cp_v[:, 0], lw=1.0, label="robot CP vx", color="tab:cyan", ls=":")
                ax_match.plot(t_arr, robot_v[:, 0], lw=0.9, label="robot center vx", color="tab:gray", alpha=0.8)
                ax_match.plot(t_arr, cp_ref_v[:, 1], lw=1.0, label="ref CP vy", color="tab:orange", ls="--")
                ax_match.plot(t_arr, obj_cp_v[:, 1], lw=1.0, label="obj CP vy", color="tab:orange")
                ax_match.plot(t_arr, robot_cp_v[:, 1], lw=1.0, label="robot CP vy", color="tab:red", ls=":")
                ax_match.plot(t_arr, robot_v[:, 1], lw=0.9, label="robot center vy", color="tab:pink", alpha=0.8)
            ax_match.axhline(0.0, ls="-", lw=0.5, color="gray")
            ax_match.set_ylabel("(m/s)", fontsize=8)
            ax_match.legend(fontsize=6, loc="upper right", ncol=2)
            ax_match.grid(True, alpha=0.3)
            _set_pruned_plot_ylim(ax_match, [
                robot_v[:, 0] if len(robot_v) else [],
                robot_v[:, 1] if len(robot_v) else [],
                cp_ref_v[:, 0] if len(cp_ref_v) else [],
                cp_ref_v[:, 1] if len(cp_ref_v) else [],
                obj_cp_v[:, 0] if len(obj_cp_v) else [],
                obj_cp_v[:, 1] if len(obj_cp_v) else [],
                robot_cp_v[:, 0] if len(robot_cp_v) else [],
                robot_cp_v[:, 1] if len(robot_cp_v) else [],
            ])
            if i == 0:
                ax_match.set_title("velocity match: ref/object/robot CP + robot center (pruned y)", fontsize=9)

            # ── right: angular velocity matching ─────────────────────────────
            ax_omega_match = axes3[i][3]
            intended_omega = np.array(_slice(hist.intended_omegas))
            object_omega = np.array(_slice(hist.object_omegas))
            robot_omega = np.array(_slice(hist.robot_angular_velocities))
            omega_cmd = np.array(_slice(hist.omega_r_history))
            if len(t_arr) and len(intended_omega) and len(object_omega) and len(robot_omega):
                ax_omega_match.plot(t_arr, intended_omega, lw=1.0, label="ref omega", ls="--")
                ax_omega_match.plot(t_arr, object_omega, lw=1.0, label="object omega")
                ax_omega_match.plot(t_arr, robot_omega, lw=1.0, label="robot actual omega")
                ax_omega_match.plot(t_arr, omega_cmd, lw=1.0, label="omega_r cmd", color="k", alpha=0.75)
            ax_omega_match.axhline(0.0, ls="-", lw=0.5, color="gray")
            ax_omega_match.set_ylabel("(rad/s)", fontsize=8)
            ax_omega_match.legend(fontsize=7, loc="upper right")
            ax_omega_match.grid(True, alpha=0.3)
            _set_pruned_plot_ylim(ax_omega_match, [intended_omega, object_omega, robot_omega, omega_cmd])
            if i == 0:
                ax_omega_match.set_title("omega match (pruned y)", fontsize=9)

        # ── bottom row: object height diagnostic ─────────────────────────────
        # Object z should remain flat near its spawn height.  A rising z or
        # z-z0 trace is the clearest sign that the object is climbing onto a robot.
        ax_z = axes3[self.n_robots][0]
        ax_dz = axes3[self.n_robots][1]
        if self.object_history.times and self.object_history.z_positions:
            t_obj = np.array(self.object_history.times)
            z_arr = np.array(self.object_history.z_positions)
            if self._t_push_start is not None:
                z_start_idx = max(0, int(np.searchsorted(t_obj, self._t_push_start)))
            else:
                z_start_idx = 0
            t_z = t_obj[z_start_idx:]
            z_push = z_arr[z_start_idx:]
            z0 = float(z_push[0]) if len(z_push) else float(z_arr[0])
            ax_z.plot(t_z, z_push, lw=1.2, color="tab:brown", label="object z")
            ax_z.axhline(z0, ls="--", lw=0.8, color="gray", label=f"z0={z0:.3f} m")
            ax_z.set_title("object height z (push phase only)", fontsize=9)
            ax_z.set_ylabel("z (m)", fontsize=8)
            ax_z.legend(fontsize=7, loc="upper right")
            ax_z.grid(True, alpha=0.3)

            ax_dz.plot(t_z, z_push - z0, lw=1.2, color="tab:red", label="z - z0")
            ax_dz.axhline(0.0, ls="--", lw=0.8, color="gray")
            ax_dz.set_title("object height drift", fontsize=9)
            ax_dz.set_ylabel("Δz (m)", fontsize=8)
            ax_dz.legend(fontsize=7, loc="upper right")
            ax_dz.grid(True, alpha=0.3)
        else:
            ax_z.axis("off")
            ax_dz.axis("off")
        axes3[self.n_robots][2].axis("off")
        axes3[self.n_robots][3].axis("off")

        plt.tight_layout()
        if save_dir:
            p = save_dir / "multi_pusher_dd_ctrl_decomp.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved {p}")
        else:
            plt.show()
        plt.close(fig3)

        # ── Figure 4: object velocity vs desired reference ─────────────────────
        if not self.object_history.times:
            return

        t_arr = np.array(self.object_history.times)
        v_arr = np.array(self.object_history.velocities)         # (N, 2) world-frame
        om_arr = np.array(self.object_history.angular_velocities)
        theta_arr = np.array(self.object_history.orientations)
        if self.object_history.desired_v_refs_body:
            v_ref_body_arr = np.array(self.object_history.desired_v_refs_body)
        else:
            v_ref_body_arr = np.repeat(self.v_ref_body.reshape(1, 2), len(t_arr), axis=0)
        if self.object_history.desired_omegas:
            omega_ref_arr = np.array(self.object_history.desired_omegas)
        else:
            omega_ref_arr = np.repeat(float(self.omega_ref), len(t_arr))

        # Rotate v_ref_body into world frame at each tick for the reference line.
        c_t = np.cos(theta_arr)
        s_t = np.sin(theta_arr)
        vx_ref = c_t * v_ref_body_arr[:, 0] - s_t * v_ref_body_arr[:, 1]
        vy_ref = s_t * v_ref_body_arr[:, 0] + c_t * v_ref_body_arr[:, 1]

        # Rotate actual world velocity into the object body frame.  The reference
        # is constant in this local frame, so drift is easier to see here.
        vx_body = c_t * v_arr[:, 0] + s_t * v_arr[:, 1]
        vy_body = -s_t * v_arr[:, 0] + c_t * v_arr[:, 1]
        speed_body = np.linalg.norm(np.column_stack([vx_body, vy_body]), axis=1)
        speed_ref = np.linalg.norm(v_ref_body_arr, axis=1)

        fig2, axes2 = plt.subplots(2, 3, figsize=(14, 6))
        fig2.suptitle("Object velocity: actual vs desired reference (pruned y-limits)", fontsize=10)

        axes2[0][0].plot(t_arr, v_arr[:, 0], lw=1.0, label="actual vx")
        axes2[0][0].plot(t_arr, vx_ref, "--", lw=1.0, label="ref vx")
        axes2[0][0].set_title("vx (world)")
        axes2[0][0].legend(fontsize=7)
        axes2[0][0].grid(True, alpha=0.3)
        _set_pruned_plot_ylim(axes2[0][0], [v_arr[:, 0], vx_ref], q_low=1.0, q_high=99.0)

        axes2[0][1].plot(t_arr, v_arr[:, 1], lw=1.0, label="actual vy")
        axes2[0][1].plot(t_arr, vy_ref, "--", lw=1.0, label="ref vy")
        axes2[0][1].set_title("vy (world)")
        axes2[0][1].legend(fontsize=7)
        axes2[0][1].grid(True, alpha=0.3)
        _set_pruned_plot_ylim(axes2[0][1], [v_arr[:, 1], vy_ref], q_low=1.0, q_high=99.0)

        axes2[0][2].plot(t_arr, om_arr, lw=1.0, label="actual omega")
        axes2[0][2].plot(t_arr, omega_ref_arr, "--", lw=1.0, label="ref omega")
        axes2[0][2].set_title("omega (rad/s)")
        axes2[0][2].legend(fontsize=7)
        axes2[0][2].grid(True, alpha=0.3)
        _set_pruned_plot_ylim(axes2[0][2], [om_arr, omega_ref_arr], q_low=1.0, q_high=99.0)

        axes2[1][0].plot(t_arr, vx_body, lw=1.0, label="actual vx body")
        axes2[1][0].plot(t_arr, v_ref_body_arr[:, 0], "--", lw=1.0, label="ref vx body")
        axes2[1][0].set_title("vx (object body)")
        axes2[1][0].legend(fontsize=7)
        axes2[1][0].grid(True, alpha=0.3)
        _set_pruned_plot_ylim(
            axes2[1][0],
            [vx_body, v_ref_body_arr[:, 0]],
            q_low=1.0,
            q_high=99.0,
        )

        axes2[1][1].plot(t_arr, vy_body, lw=1.0, label="actual vy body")
        axes2[1][1].plot(t_arr, v_ref_body_arr[:, 1], "--", lw=1.0, label="ref vy body")
        axes2[1][1].set_title("vy (object body)")
        axes2[1][1].legend(fontsize=7)
        axes2[1][1].grid(True, alpha=0.3)
        _set_pruned_plot_ylim(
            axes2[1][1],
            [vy_body, v_ref_body_arr[:, 1]],
            q_low=1.0,
            q_high=99.0,
        )

        axes2[1][2].plot(t_arr, speed_body, lw=1.0, label="actual |v_body|")
        axes2[1][2].plot(t_arr, speed_ref, "--", lw=1.0, label="ref |v_body|")
        axes2[1][2].set_title("speed magnitude (body)")
        axes2[1][2].legend(fontsize=7)
        axes2[1][2].grid(True, alpha=0.3)
        _set_pruned_plot_ylim(axes2[1][2], [speed_body, speed_ref], q_low=1.0, q_high=99.0)

        plt.tight_layout()
        if save_dir:
            p = save_dir / "multi_pusher_dd_object_velocity.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved {p}")
        else:
            plt.show()
        plt.close(fig2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_authority_precheck_from_live_args(args: argparse.Namespace, t_params: List[float]) -> None:
    """Update the standalone pre-push authority diagnostic for this live run."""
    script_path = Path(__file__).resolve().parent / "precheck_multi_pusher_authority.py"
    if not script_path.exists():
        print(f"[authority] skipped: script not found at {script_path}")
        return

    save_dir = Path(args.authority_save_dir)
    t_params_arg = ",".join(f"{float(t):.12g}" for t in t_params)
    cmd = [
        sys.executable,
        str(script_path),
        "--object", str(args.object),
        "--v-ref-x", str(args.v_ref_x),
        "--v-ref-y", str(args.v_ref_y),
        "--omega-ref", str(args.omega_ref),
        "--t-params", t_params_arg,
        "--passive-ratio", str(args.obstructing_passive_ratio),
        "--save-dir", str(save_dir),
    ]
    print(f"[authority] updating precheck plot in {save_dir} ...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[authority] warning: precheck failed with exit code {exc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-pusher diff-drive: 4 robots at Magnum Four contact points, "
            "single constant object twist."
        )
    )
    parser.add_argument(
        "--object", type=str, default=DEFAULT_OBJECT_SHAPE,
        help="Object shape (rect / right_triangle / pi / root / hourglass / meteor)",
    )
    parser.add_argument(
        "--v-ref-x", type=float, default=0.0,
        help="Desired object body-frame vx (m/s). Default: 0.0",
    )
    parser.add_argument(
        "--v-ref-y", type=float, default=0.05,
        help="Desired object body-frame vy (m/s). Default: 0.05",
    )
    parser.add_argument(
        "--omega-ref", type=float, default=0.0,
        help="Desired object angular velocity omega (rad/s). Default: 0.0",
    )
    parser.add_argument(
        "--twist-scale",
        type=float,
        default=1.0,
        metavar="K",
        help=(
            "Without --segment-csv: multiply --v-ref-x, --v-ref-y, and --omega-ref after parsing "
            "(same motion direction; K× twist magnitude; ideal arc time scales ~1/K). "
            "Ignored with --segment-csv—use --segment-v-speed instead."
        ),
    )
    parser.add_argument(
        "--segment-csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional CSV of world poses (columns x,y,theta; optional t ignored). "
            "Each leg commands a constant body twist toward the next waypoint. "
            "open-loop solver T_nom per leg (default). Optional --csv-segment-pose-stop "
            "enables waypoint/timeout completion (experimental). "
            "Re-solving each leg from measured pose is **on by default**; use "
            "--no-csv-replan-each-push to disable. "
            "Forces --test-transition; overrides --v-ref-x/y and --omega-ref for segment 0."
        ),
    )
    parser.add_argument(
        "--segment-v-speed",
        type=float,
        default=0.1,
        metavar="M/S",
        help=(
            "Translation speed argument to solve_constant_body_twist_from_SE2 when "
            "using --segment-csv. Default 0.1 matches test_magnum_diffdrive_control."
        ),
    )
    parser.add_argument(
        "--csv-replan-each-push",
        action="store_true",
        help=(
            "Legacy no-op with --segment-csv: replanning is **on by default** for CSV runs. "
            "Use --no-csv-replan-each-push to disable re-solving each leg from measured pose."
        ),
    )
    parser.add_argument(
        "--no-csv-replan-each-push",
        action="store_true",
        help=(
            "With --segment-csv: disable default per-leg (and leg-0 pre-push) re-solve from "
            "measured pose."
        ),
    )
    parser.add_argument(
        "--cross-track-integrate",
        action="store_true",
        dest="cross_track_integrate",
        help=(
            "Screw cross-track outer loop: add clipped k·e_d (m) to commanded ω vs the "
            "nominal constant body-twist path from each push-start pose. With --segment-csv "
            "uses per-segment solver T; plain runs use T = --duration. Under --fixed-ref, "
            "feed-forward is re-solved every tick (no VR/α low-pass)."
        ),
    )
    parser.add_argument(
        "--cross-track-k",
        type=float,
        default=4.0,
        dest="cross_track_k",
        metavar="RAD/S/M",
        help="Gain from signed cross-track error (m) to ω trim. Default: 4.0",
    )
    parser.add_argument(
        "--cross-track-omega-max",
        type=float,
        default=0.25,
        dest="cross_track_omega_max",
        metavar="RAD/S",
        help="Symmetric cap on ω trim from cross-track. Default: 0.25",
    )
    parser.add_argument(
        "--csv-segment-pose-stop",
        action="store_true",
        help=(
            "With --segment-csv: end each leg on pose proximity (+ optional vel gate) with "
            "timeout (experimental). Default is open-loop T_nom only."
        ),
    )
    parser.add_argument(
        "--csv-segment-pos-tol",
        type=float,
        default=0.045,
        metavar="M",
        help="With --segment-csv (pose mode): position error gate vs waypoint xy. Default: 0.045",
    )
    parser.add_argument(
        "--csv-segment-yaw-tol-deg",
        type=float,
        default=10.0,
        metavar="DEG",
        help="With --segment-csv (pose mode): |Δθ| gate vs waypoint. Default: 10",
    )
    parser.add_argument(
        "--csv-segment-vel-tol",
        type=float,
        default=0.03,
        metavar="M/S",
        help="With --segment-csv: max |v| when pose gate also requires low speed. Default: 0.03",
    )
    parser.add_argument(
        "--csv-segment-omega-tol",
        type=float,
        default=0.15,
        metavar="RAD/S",
        help="With --segment-csv: max |ω| when pose gate also requires low speed. Default: 0.15",
    )
    parser.add_argument(
        "--csv-segment-skip-vel-gate",
        action="store_true",
        help="With --segment-csv (pose mode): accept waypoint pose even if object still moving.",
    )
    parser.add_argument(
        "--csv-segment-timeout-factor",
        type=float,
        default=6.0,
        metavar="K",
        help="With --segment-csv (pose mode): max push time ≥ K * T_nom per leg. Default: 6",
    )
    parser.add_argument(
        "--csv-segment-timeout-min-s",
        type=float,
        default=5.0,
        metavar="S",
        help="With --segment-csv (pose mode): min cap on max push time per leg. Default: 5",
    )
    parser.add_argument(
        "--csv-segment-time-scale",
        type=float,
        default=1.0,
        metavar="K",
        help=(
            "With --segment-csv and T_nom open-loop legs: end each push after K*T_solver "
            "(default 1). Use K>1 if the object lags the ideal constant-twist distance."
        ),
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Total sim duration (s)")
    parser.add_argument(
        "--approach-distance", type=float, default=APPROACH_DISTANCE,
        help="Spawn offset beyond contact point (m)",
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--debug-vel", action="store_true",
        help="Print per-robot velocity diagnostics (throttled by --debug-vel-every)",
    )
    parser.add_argument(
        "--debug-vel-every", type=int, default=50, metavar="K",
        help="Print every K control ticks. Default: 50",
    )
    parser.add_argument("--ground-friction", type=float, default=DEFAULT_GROUND_FRICTION)
    parser.add_argument(
        "--object-friction",
        type=float,
        default=DEFAULT_OBJECT_FRICTION,
        help=(
            "Object lateral friction (PyBullet). Combined with bumper mu for "
            "bumper-object contacts. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--bumper-contact-mu",
        type=float,
        default=0.01,
        help=(
            "Robot bumper-object lateral friction. Default: %(default)s (very low; "
            "raise toward object-friction to test cone/slip sensitivity)."
        ),
    )
    parser.add_argument("--wheel-friction", type=float, default=DEFAULT_WHEEL_LATERAL_FRICTION)
    parser.add_argument("--caster-friction", type=float, default=DEFAULT_CASTER_LATERAL_FRICTION)
    parser.add_argument(
        "--no-planar-cheat",
        action="store_true",
        help=(
            "Drive diff-drive via wheel motors instead of planar-joint velocity "
            "cheat (slower; friction at wheels/bumper matters more)."
        ),
    )
    parser.add_argument("--kp-alpha", type=float, default=0.5)
    parser.add_argument(
        "--kd-alpha", type=float, default=0.0,
        help=(
            "Derivative gain on alpha error for the PD heading controller. "
            "0 = pure P (original behaviour). Try 0.05–0.2 to damp multi-robot "
            "coupling oscillations. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--fixed-ref", action="store_true",
        help=(
            "Fall back to the original one-shot reference (solve once at "
            "realign-start + alpha* snap at push-entry).  Combine with "
            "--kd-alpha to isolate the PD effect independently of live-resolve. "
            "Without this flag the live re-solve is always active (default)."
        ),
    )
    parser.add_argument(
        "--test-transition",
        action="store_true",
        help=(
            "Special transition test: push the requested object twist for half "
            "the duration, stop, switch to perpendicular body-frame translation "
            "(-vy, vx) and reversed omega, then rerun approach/realign/push. "
            "With --segment-csv this half-duration swap is disabled; CSV segments "
            "drive twist changes with the same barrier resets between segments."
        ),
    )
    parser.add_argument(
        "--transition-teleport-robots",
        action="store_true",
        help=(
            "Simulation-only transition cheat: at the twist switch, reset each "
            "robot to the current contact point plus the original normal approach "
            "offset before rerunning approach/realign."
        ),
    )
    parser.add_argument(
        "--stage-position-tol", type=float, default=0.02,
        help="Tolerance (m) for the pre-approach staging pose. Default: 0.02",
    )
    parser.add_argument(
        "--stage-heading-tol-deg", type=float, default=5.0,
        help="Heading tolerance (deg) for the pre-approach staging pose. Default: 5.0",
    )
    parser.add_argument(
        "--kp-stage-position", type=float, default=1.5,
        help="Diff-drive staging position gain before contact approach. Default: 1.5",
    )
    parser.add_argument(
        "--kp-stage-heading", type=float, default=3.0,
        help="Diff-drive staging heading gain before contact approach. Default: 3.0",
    )
    parser.add_argument(
        "--kd-stage-heading", type=float, default=0.8,
        help="Angular damping gain for staging heading control. Default: 0.8",
    )
    parser.add_argument(
        "--max-stage-omega", type=float, default=0.4,
        help="Stage-only angular speed cap (rad/s) to avoid transition spin chatter. Default: 0.4",
    )
    parser.add_argument(
        "--disable-contact-geometry-update",
        action="store_true",
        help=(
            "Disable bounded contact-point refresh at the all-contact -> realign "
            "barrier. By default, realign uses the actual landed contact point."
        ),
    )
    parser.add_argument(
        "--contact-update-max-distance", type=float, default=0.06,
        help=(
            "Maximum boundary-distance shift (m) when refreshing landed contact "
            "geometry before realign. Default: 0.06"
        ),
    )
    parser.add_argument(
        "--kd-pos", type=float, default=0.0,
        help=(
            "Derivative gain on normal contact gap (normal damper, -D·q̇ term). "
            "Turns the normal-gap P term into a spring-damper for forward speed. "
            "0 = pure P (original). Try 0.1–0.5. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--k-tangent", type=float, default=0.0,
        help=(
            "Gain that maps tangential contact slip into omega_r via "
            "a signed authority gate times k_t*e_t/R. This gives sideways slip "
            "a rotational correction instead of trying to fix it with v_r. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--kd-tangent", type=float, default=0.0,
        help=(
            "Derivative damping gain for tangential slip: authority_gate * "
            "kd_tangent * d(e_t)/dt / R. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--tangent-authority-deadband", type=float, default=0.25,
        help=(
            "Deadband on the signed tangent authority ratio sign(vr_ff)*cos(alpha). "
            "Contacts inside this band get no tangent omega correction; outside "
            "it ramps continuously to full authority. Default: 0.25"
        ),
    )
    parser.add_argument(
        "--tangent-error-deadband", type=float, default=0.0,
        help="Deadband (m) on tangential position error before k_tangent acts. Default: 0.0",
    )
    parser.add_argument(
        "--max-normal-speed", type=float, default=0.08,
        help="Hard cap (m/s) on normal-gap P/D contributions. Default: 0.08",
    )
    parser.add_argument(
        "--max-tangent-omega", type=float, default=0.4,
        help="Hard cap (rad/s) on tangential-slip omega correction. Default: 0.4",
    )
    parser.add_argument(
        "--obstructing-pusher-speed-scale", type=float, default=1.0,
        help=(
            "Diagnostic cheat applied only during PUSH: multiply the signed "
            "forward speed of precomputed obstructing pushers by this factor. "
            "Obstructing pushers are contacts with normal_ratio < "
            "-abs(--obstructing-passive-ratio). Default: 1.0"
        ),
    )
    parser.add_argument(
        "--obstructing-passive-ratio", type=float, default=0.1,
        help=(
            "Normalized normal-ratio threshold for obstructing pusher detection. "
            "normal_ratio = cos(true_alpha); robots below -abs(value) are scaled. "
            "Default: 0.1"
        ),
    )
    parser.add_argument(
        "--obstructing-inflate-gap", type=float, default=0.02,
        help=(
            "Outward virtual-object inflation gap (m) used only by coupling for "
            "obstructing pushers. This lets k_couple maintain a nearby standoff "
            "formation instead of exact contact. Default: 0.02"
        ),
    )
    parser.add_argument(
        "--couple-all-robots",
        action="store_true",
        help=(
            "Apply k_couple to all robots. By default coupling is applied only "
            "to obstructing pushers so active feed-forward pushers are not disturbed."
        ),
    )
    parser.add_argument(
        "--disable-actual-contact-clearance-cheat",
        action="store_true",
        help=(
            "Disable the simulation-only obstructing-pusher clearance rule that "
            "uses measured object contact-point velocity. Falls back to simple "
            "signed v_r scaling."
        ),
    )
    parser.add_argument(
        "--k-couple", type=float, default=0.0,
        help=(
            "Diff-drive normal-gap pressure-sharing gain. Compares each robot's "
            "normal contact gap against its two Magnum-order neighbours, then "
            "projects only the inward-normal bias onto the robot drive axis. "
            "0 = fully decoupled. Try 0.1–1.5 with --max-couple-speed limiting "
            "the actual velocity contribution. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--max-couple-speed", type=float, default=0.05,
        help=(
            "Hard cap (m/s) on the absolute v_couple normal-gap contribution. "
            "Prevents coupling from dominating vr_ff/position control. Default: 0.05"
        ),
    )
    parser.add_argument("--kp-position", type=float, default=1.0)
    parser.add_argument(
        "--kp-obj-speed", type=float, default=1.0,
        help=(
            "Object-level translational twist servo gain. It modifies the "
            "effective object velocity before each robot's exact DD matching "
            "solve; it is no longer added directly to v_r. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--kp-object-omega", type=float, default=0.0,
        help=(
            "Object-level angular twist servo gain. Adds a capped correction to "
            "omega_ref before the DD matching solve. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--max-object-v-correction", type=float, default=0.03,
        help="Cap (m/s) on object-level translational velocity correction. Default: 0.03",
    )
    parser.add_argument(
        "--max-object-omega-correction", type=float, default=0.05,
        help="Cap (rad/s) on object-level omega correction. Default: 0.05",
    )
    parser.add_argument(
        "--max-speed-p", type=float, default=0.03,
        help=(
            "Deprecated with the stronger controller; retained for CLI "
            "compatibility. Direct v_speed_p is now removed from v_r. Default: 0.03"
        ),
    )
    parser.add_argument(
        "--speed-p-pos-gate-m", type=float, default=0.01,
        help=(
            "Deprecated with the stronger controller; retained for CLI "
            "compatibility. Direct v_speed_p is now removed from v_r. Default: 0.01"
        ),
    )
    parser.add_argument(
        "--k-force-comp", type=float, default=0.0,
        help=(
            "Force-gated contact compensation gain (m/s per N). Active only when "
            "contact force is below --force-comp-threshold, adding a small "
            "inward-normal v_comp. Default: 0.0"
        ),
    )
    parser.add_argument("--force-comp-threshold", type=float, default=0.5)
    parser.add_argument("--force-comp-target", type=float, default=1.0)
    parser.add_argument("--max-comp-speed", type=float, default=0.03)
    parser.add_argument(
        "--contact-gap-deadband", type=float, default=0.002,
        help=(
            "Normal-gap deadband (m) used to prevent v_couple/v_comp from "
            "adding inward pressure when geometry says the robot is already "
            "at or inside the intended normal contact position. Default: 0.002"
        ),
    )
    parser.add_argument(
        "--compression-relax-gain", type=float, default=0.0,
        help=(
            "Geometry-only relaxation gain for inward vr_ff when normal_gap is "
            "negative (robot already compressed past intended contact). "
            "relax_fraction = gain * compression, capped by --max-compression-relax. "
            "0 disables. Try 10-30. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--compression-relax-deadband", type=float, default=0.002,
        help=(
            "Normal compression deadband (m) used only by v_relax. "
            "Separate from --contact-gap-deadband so large contact/coupling "
            "deadbands do not disable feed-forward relaxation. Default: 0.002"
        ),
    )
    parser.add_argument(
        "--max-compression-relax", type=float, default=0.5,
        help=(
            "Maximum fraction of inward vr_ff to remove under compression. "
            "0.5 means at most 50 percent of inward feed-forward is relaxed. Default: 0.5"
        ),
    )
    parser.add_argument(
        "--z-relax-threshold", type=float, default=0.003,
        help=(
            "Simulation-only creep-up detector threshold (m): if object z rises "
            "above push-start z by this amount, relax inward vr_ff. Default: 0.003"
        ),
    )
    parser.add_argument(
        "--z-relax-gain", type=float, default=0.0,
        help=(
            "Simulation-only creep-up relaxation gain. "
            "z_relax_fraction = gain * max(0, z_lift - threshold), capped by "
            "--max-z-relax. 0 disables. Try 50-150. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--max-z-relax", type=float, default=0.8,
        help="Maximum inward vr_ff relaxation fraction from object z lift. Default: 0.8",
    )
    parser.add_argument(
        "--no-branch-lock", action="store_true",
        help=(
            "Disable live-resolve branch locking. By default live-resolve keeps "
            "the forward/backward branch chosen at realign-start to avoid vr_ff "
            "sign flips and alpha* pi-jumps."
        ),
    )
    parser.add_argument(
        "--live-object-servo-scale", type=float, default=0.0,
        help=(
            "Scale applied to object-level velocity/omega servo only in "
            "live-resolve mode. 0 disables servo amplification for live-resolve; "
            "1 matches fixed-ref behavior. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--live-ref-filter-alpha", type=float, default=0.05,
        help=(
            "Low-pass coefficient for live-resolved vr_ff, omega_ff, and zeta0. "
            "1.0 = no filtering, 0.0 = hold previous live reference. Default: 0.05"
        ),
    )
    parser.add_argument(
        "--live-alpha-filter-alpha", type=float, default=0.0,
        help=(
            "Low-pass coefficient for live-resolved alpha_star after hysteresis "
            "is exceeded. 1.0 = snap to raw live alpha, 0.0 = lock alpha. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--live-alpha-hysteresis-deg", type=float, default=999.0,
        help=(
            "Hysteresis band in degrees for live alpha_star updates. Deltas "
            "inside this band are ignored to avoid alpha jitter. Default: 999.0 (effectively locked)"
        ),
    )
    parser.add_argument(
        "--magnum-verbose", action="store_true",
        help="Verbose output from the Magnum Four solver.",
    )
    parser.add_argument(
        "--magnum-visualize", action="store_true",
        help="Visualize Magnum Four solver result (requires GUI).",
    )
    parser.add_argument(
        "--align-heading-tol-deg", type=float, default=2.0, metavar="DEG",
        help=(
            "Heading tolerance (degrees) for the stop-go self-rotate step. "
            "Robot stops spinning once |e_zeta| < this value. Default: 2.0"
        ),
    )
    parser.add_argument(
        "--stop-go-pre-push-hold-s", type=float, default=0.5, metavar="S",
        help=(
            "Seconds to hold still after all robots finish realign before "
            "starting the push phase. Default: 0.5"
        ),
    )
    parser.add_argument(
        "--skip-authority-precheck",
        action="store_true",
        help="Do not auto-run precheck_multi_pusher_authority.py before the live simulation.",
    )
    parser.add_argument(
        "--authority-save-dir",
        type=str,
        default="/tmp/multi_pusher_authority",
        help="Save directory for the auto-generated authority_precheck plot/JSON/CSV.",
    )
    args = parser.parse_args()

    csv_replan_effective = bool(args.segment_csv) and (not bool(args.no_csv_replan_each_push))
    if (
        bool(args.segment_csv)
        and bool(args.no_csv_replan_each_push)
        and bool(args.csv_replan_each_push)
    ):
        print(
            "[segment-csv] --no-csv-replan-each-push overrides legacy --csv-replan-each-push."
        )

    csv_segments: Optional[List[Dict[str, Any]]] = None
    csv_waypoints_world: Optional[np.ndarray] = None
    if args.segment_csv:
        csv_path = Path(args.segment_csv).expanduser().resolve()
        csv_segments, csv_waypoints_world = _load_csv_twist_segments(
            csv_path, float(args.segment_v_speed)
        )
        args.test_transition = True
        vb0 = csv_segments[0]["v_body"]
        args.v_ref_x = float(vb0[0])
        args.v_ref_y = float(vb0[1])
        args.omega_ref = float(csv_segments[0]["omega"])
        print(
            f"\n[segment-csv] loaded {len(csv_segments)} segment(s) from {csv_path}\n"
            f"  segment-v-speed={float(args.segment_v_speed):.4f} m/s  "
            f"segment 0: v=({args.v_ref_x:+.4f}, {args.v_ref_y:+.4f}) m/s, "
            f"omega={args.omega_ref:+.4f} rad/s, T0={float(csv_segments[0]['T']):.4f}s\n"
            f"  (--test-transition forced: barrier resets between CSV segments)\n"
            "  Each leg ends after solver T_nom by default; use --csv-segment-pose-stop for "
            "experimental pose/timeout completion. See csv_segment_end_log.json.\n"
            f"  csv-replan-each-push: {csv_replan_effective}  "
            f"(use --no-csv-replan-each-push to disable)\n"
        )
        max_chord = 0.0
        max_tsol = max(float(s["T"]) for s in csv_segments)
        wp = np.asarray(csv_waypoints_world, dtype=float).reshape(-1, 3)
        for ri in range(wp.shape[0] - 1):
            max_chord = max(
                max_chord,
                float(np.linalg.norm(wp[ri + 1, :2] - wp[ri, :2])),
            )
        if max_chord > 1.5 or max_tsol > 25.0:
            print(
                "[segment-csv] Long path: open-loop T_nom assumes θ follows the constant-twist "
                "primitive. Body-frame speed can match while **world** motion misses if ω drifts "
                f"(max chord ≈ {max_chord:.2f} m, max T_nom ≈ {max_tsol:.1f} s). "
                "Try --kp-object-omega 0.5–2 --max-object-omega-correction 0.2, "
                "--max-object-v-correction 0.06–0.1, and/or --csv-segment-time-scale 1.2–1.8, "
                "or use a shorter CSV for diagnosis.\n"
            )

    if args.segment_csv and abs(float(args.twist_scale) - 1.0) > 1e-12:
        print(
            "[twist-scale] ignored with --segment-csv (change --segment-v-speed to scale CSV primitives).\n"
        )
    elif not args.segment_csv:
        k = float(args.twist_scale)
        if k <= 0.0:
            raise ValueError("--twist-scale must be > 0")
        if abs(k - 1.0) > 1e-12:
            args.v_ref_x = float(args.v_ref_x) * k
            args.v_ref_y = float(args.v_ref_y) * k
            args.omega_ref = float(args.omega_ref) * k
            print(
                f"\n[twist-scale] K={k:.6g} → "
                f"v_ref_body=({args.v_ref_x:+.6f}, {args.v_ref_y:+.6f}) m/s, "
                f"ω_ref={args.omega_ref:+.6f} rad/s\n"
                "  Ideal rigid-body: same SE(2) arc, ~K× faster (shorter --duration may suffice).\n"
            )

    selected_name = args.object

    # ── Magnum Four: solve / load optimal 4-contact-point configuration ────────
    # Cache lives next to the URDF folder so it persists across runs.
    cache_file = Path(pkg_path) / "urdf" / "magnum_four_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cached_t_params = None
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            if selected_name in cache_data:
                cached_t_params = cache_data[selected_name]
                print(
                    f"[magnum] Found cached t_params for '{selected_name}': "
                    f"{[f'{v:.4f}' for v in cached_t_params]}"
                )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[magnum] Warning: failed to load cache: {e}")

    if cached_t_params is not None:
        t_params = cached_t_params
    else:
        # Cache miss — run the solver.  Build a temporary headless sim just to
        # load the generic object for the solver, then disconnect before the
        # main sim is set up.
        _obj_file_map = {
            "right_triangle": "right_triangle.obj",
            "pi": "pi.obj",
            "root": "root.obj",
            "rect": "rect.obj",
            "hourglass": "hourglass.obj",
            "meteor": "meteor.obj",
        }
        if selected_name not in _obj_file_map:
            raise ValueError(
                f"Unknown object '{selected_name}'. Available: {sorted(_obj_file_map)}"
            )

        setup_pybullet(gui=False, ground_friction=float(args.ground_friction))
        _gen_obj, _ = obj_to_generic(
            obj_path=_obj_file_map[selected_name],
            shape_name=selected_name,
            position=(0.0, 0.0, DEFAULT_OBJECT_HEIGHT),
            orientation=0.0,
            mass=5.0,
            lateral_friction=float(args.object_friction),
            blind_test=True,
        )
        pyb.disconnect()

        print(f"\n[magnum] Computing Magnum Four contact points for '{selected_name}'...")
        magnum_result = find_the_magnum_four_v3(
            _gen_obj,
            verbose=args.magnum_verbose,
            visualize=args.magnum_visualize and (not args.no_gui),
            weighting_scheme="balanced",
            torque_method=3,
        )
        if not magnum_result or not magnum_result.get("success", False):
            raise RuntimeError("Magnum Four solver failed to produce a solution.")

        contacts = magnum_result["best_solution"]["contacts"]
        t_params = [float(c.parameter) % 1.0 for c in contacts]

        if len(t_params) != 4:
            raise RuntimeError(f"Expected 4 contacts from Magnum Four, got {len(t_params)}")

        print(f"[magnum] Solved t_params: {[f'{v:.4f}' for v in t_params]}")

        # Persist to cache.
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        cache_data[selected_name] = t_params
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"[magnum] Saved t_params to cache: {cache_file}")

    # ── Authority precheck ─────────────────────────────────────────────────────
    if not args.skip_authority_precheck:
        _run_authority_precheck_from_live_args(args, t_params)

    # ── Main simulation ────────────────────────────────────────────────────────
    setup_pybullet(gui=not args.no_gui, ground_friction=float(args.ground_friction))

    v_ref_body = np.array([args.v_ref_x, args.v_ref_y], dtype=float)

    test = MultiPusherConstantTwistDiffdrive(
        t_params=t_params,
        v_ref_body=v_ref_body,
        omega_ref=args.omega_ref,
        object_name=selected_name,
        approach_distance=args.approach_distance,
        object_lateral_friction=float(args.object_friction),
        bumper_contact_mu=float(args.bumper_contact_mu),
        wheel_lateral_friction=float(args.wheel_friction),
        caster_lateral_friction=float(args.caster_friction),
        use_planar_cheat_control=not args.no_planar_cheat,
        kp_alpha=args.kp_alpha,
        kd_alpha=args.kd_alpha,
        kp_position=args.kp_position,
        kd_pos=args.kd_pos,
        k_tangent=args.k_tangent,
        kd_tangent=args.kd_tangent,
        tangent_authority_deadband=args.tangent_authority_deadband,
        tangent_error_deadband=args.tangent_error_deadband,
        k_couple=args.k_couple,
        kp_obj_speed=args.kp_obj_speed,
        kp_object_omega=args.kp_object_omega,
        max_object_v_correction=args.max_object_v_correction,
        max_object_omega_correction=args.max_object_omega_correction,
        max_speed_p=args.max_speed_p,
        speed_p_pos_gate_m=args.speed_p_pos_gate_m,
        max_normal_speed=args.max_normal_speed,
        max_tangent_omega=args.max_tangent_omega,
        max_couple_speed=args.max_couple_speed,
        k_force_comp=args.k_force_comp,
        force_comp_threshold=args.force_comp_threshold,
        force_comp_target=args.force_comp_target,
        max_comp_speed=args.max_comp_speed,
        contact_gap_deadband=args.contact_gap_deadband,
        compression_relax_gain=args.compression_relax_gain,
        compression_relax_deadband=args.compression_relax_deadband,
        max_compression_relax=args.max_compression_relax,
        z_relax_threshold=args.z_relax_threshold,
        z_relax_gain=args.z_relax_gain,
        max_z_relax=args.max_z_relax,
        lock_live_branch=not args.no_branch_lock,
        live_object_servo_scale=args.live_object_servo_scale,
        live_ref_filter_alpha=args.live_ref_filter_alpha,
        live_alpha_filter_alpha=args.live_alpha_filter_alpha,
        live_alpha_hysteresis_rad=np.deg2rad(args.live_alpha_hysteresis_deg),
        use_live_resolve=not args.fixed_ref,
        obstructing_pusher_speed_scale=args.obstructing_pusher_speed_scale,
        obstructing_passive_ratio=args.obstructing_passive_ratio,
        obstructing_inflate_gap=args.obstructing_inflate_gap,
        couple_obstructing_only=not args.couple_all_robots,
        use_actual_contact_clearance_cheat=not args.disable_actual_contact_clearance_cheat,
        csv_segments=csv_segments,
        csv_waypoints_world=csv_waypoints_world,
        csv_segment_v_speed=float(args.segment_v_speed),
        csv_replan_each_push=csv_replan_effective,
        cross_track_integrate=bool(args.cross_track_integrate),
        cross_track_k=float(args.cross_track_k),
        cross_track_omega_max=float(args.cross_track_omega_max),
    )

    ref_mode_str = "fixed-ref (one-shot)" if args.fixed_ref else "live-resolve (every tick)"
    print(f"\n[config] N={len(t_params)} robots (Magnum Four)")
    print(f"  t_params  : {[round(t, 4) for t in t_params]}")
    print(f"  v_ref_body: ({args.v_ref_x:.4f}, {args.v_ref_y:.4f}) m/s")
    print(f"  omega_ref : {args.omega_ref:.4f} rad/s")
    if args.segment_csv:
        assert csv_segments is not None
        print(
            f"  segment-csv: {len(csv_segments)} twist segment(s)  "
            f"teleport={args.transition_teleport_robots}  "
            f"replan={csv_replan_effective}  "
            f"cross_track={bool(args.cross_track_integrate)}  "
            f"leg_end={'pose+timeout' if args.csv_segment_pose_stop else 'T_nom_open_loop'}  "
            f"T_scale={float(args.csv_segment_time_scale):.2f}"
        )
        if args.csv_segment_pose_stop:
            print(
                f"    pose tol: pos≤{args.csv_segment_pos_tol:.3f} m  "
                f"|Δθ|≤{args.csv_segment_yaw_tol_deg:.1f}°  "
                f"vel_gate={'off' if args.csv_segment_skip_vel_gate else 'on'}  "
                f"timeout≥max({args.csv_segment_timeout_factor:.1f}·T_nom, "
                f"{args.csv_segment_timeout_min_s:.1f}s)"
            )
    elif args.test_transition:
        print(
            f"  transition: enabled at t={0.5 * args.duration:.2f}s -> "
            f"v_ref_body=({-args.v_ref_y:.4f}, {args.v_ref_x:.4f}) m/s, "
            f"omega_ref={-args.omega_ref:.4f} rad/s  "
            f"teleport={args.transition_teleport_robots}"
        )
    if args.cross_track_integrate:
        _ct_horizon = (
            "per-segment T_solver"
            if args.segment_csv
            else f"T_horizon={float(args.duration):.3f}s (--duration)"
        )
        print(
            f"  cross-track-integrate: k={float(args.cross_track_k):.3f} rad/(s·m)  "
            f"cap=±{float(args.cross_track_omega_max):.3f} rad/s  ({_ct_horizon})"
        )
    print(f"  |v_cp_body| per robot: {[round(s, 4) for s in test._desired_cp_speed]}")
    print(f"  ref mode  : {ref_mode_str}")
    print(f"  branch lock: {not args.no_branch_lock}  (live-resolve only)")
    print(
        f"  live filt : ref_alpha={args.live_ref_filter_alpha:.3f}  "
        f"alpha_alpha={args.live_alpha_filter_alpha:.3f}  "
        f"alpha_hyst={args.live_alpha_hysteresis_deg:.2f} deg  "
        f"obj_servo_scale={args.live_object_servo_scale:.3f}"
    )
    print(f"  kp_alpha  : {args.kp_alpha:.3f}   kd_alpha : {args.kd_alpha:.3f}")
    print(
        f"  normal    : k_n={args.kp_position:.3f}  kd_n={args.kd_pos:.3f}  "
        f"cap={args.max_normal_speed:.3f} m/s"
    )
    print(
        f"  staging   : pos_tol={args.stage_position_tol:.3f} m  "
        f"head_tol={args.stage_heading_tol_deg:.2f} deg  "
        f"kp_pos={args.kp_stage_position:.3f}  "
        f"kp_head={args.kp_stage_heading:.3f}  "
        f"kd_head={args.kd_stage_heading:.3f}  "
        f"max_omega={args.max_stage_omega:.3f}"
    )
    print(
        f"  contact update: {not args.disable_contact_geometry_update}  "
        f"max_shift={args.contact_update_max_distance:.3f} m"
    )
    print(
        f"  tangent   : k_t={args.k_tangent:.3f}  "
        f"kd_t={args.kd_tangent:.3f}  "
        f"auth_deadband={args.tangent_authority_deadband:.3f}  "
        f"err_deadband={args.tangent_error_deadband:.4f} m  "
        f"cap={args.max_tangent_omega:.3f} rad/s"
    )
    print(
        f"  obstructing scale: speed_scale={args.obstructing_pusher_speed_scale:.3f}  "
        f"passive_ratio={args.obstructing_passive_ratio:.3f}  "
        f"actual_cp_clearance_cheat={not args.disable_actual_contact_clearance_cheat}  (push only)"
    )
    print(
        f"  obj servo : kp_v={args.kp_obj_speed:.3f}  kp_w={args.kp_object_omega:.3f}  "
        f"cap_v={args.max_object_v_correction:.3f} m/s  "
        f"cap_w={args.max_object_omega_correction:.3f} rad/s"
    )
    print(
        f"  k_couple  : {args.k_couple:.3f}  "
        f"max_couple={args.max_couple_speed:.3f} m/s "
        f"inflate_gap={args.obstructing_inflate_gap:.3f} m  "
        f"obstructing_only={not args.couple_all_robots} "
        f"(normal-gap pressure sharing, 0=decoupled)"
    )
    print(
        f"  v_comp    : k={args.k_force_comp:.3f}  "
        f"threshold={args.force_comp_threshold:.3f} N  "
        f"target={args.force_comp_target:.3f} N  cap={args.max_comp_speed:.3f} m/s  "
        f"gap_deadband={args.contact_gap_deadband:.4f} m"
    )
    print(
        f"  relax    : gain={args.compression_relax_gain:.3f}  "
        f"deadband={args.compression_relax_deadband:.4f} m  "
        f"max_fraction={args.max_compression_relax:.3f}"
    )
    print(
        f"  z relax  : gain={args.z_relax_gain:.3f}  "
        f"threshold={args.z_relax_threshold:.4f} m  "
        f"max_fraction={args.max_z_relax:.3f}  (simulation creep-up detector)"
    )
    print(
        f"  friction  : object={args.object_friction:.3f}  "
        f"bumper={args.bumper_contact_mu:.3f}  ground={args.ground_friction:.3f}  "
        f"wheel={args.wheel_friction:.3f}  caster={args.caster_friction:.3f}  "
        f"planar_cheat={not args.no_planar_cheat}"
    )
    print()

    save_dir = Path(args.save_dir) if args.save_dir else None
    run_kw: Dict[str, Any] = {
        "duration": args.duration,
        "gui": not args.no_gui,
        "debug_vel": args.debug_vel,
        "debug_every": args.debug_vel_every,
        "save_dir": save_dir,
        "record_video": args.record_video,
        "align_heading_tol_rad": np.deg2rad(args.align_heading_tol_deg),
        "stop_go_sleep_after_realign_s": float(args.stop_go_pre_push_hold_s),
        "test_transition": bool(args.test_transition),
        "transition_teleport_robots": bool(args.transition_teleport_robots),
        "stage_position_tol": float(args.stage_position_tol),
        "stage_heading_tol_rad": np.deg2rad(args.stage_heading_tol_deg),
        "kp_stage_position": float(args.kp_stage_position),
        "kp_stage_heading": float(args.kp_stage_heading),
        "kd_stage_heading": float(args.kd_stage_heading),
        "max_stage_omega": float(args.max_stage_omega),
        "update_contact_on_realign": not args.disable_contact_geometry_update,
        "contact_update_max_distance": float(args.contact_update_max_distance),
    }
    if args.segment_csv:
        run_kw.update(
            csv_segment_time_only=not bool(args.csv_segment_pose_stop),
            csv_segment_pos_tol_m=float(args.csv_segment_pos_tol),
            csv_segment_yaw_tol_rad=np.deg2rad(float(args.csv_segment_yaw_tol_deg)),
            csv_segment_vel_tol_m_s=float(args.csv_segment_vel_tol),
            csv_segment_omega_tol_rad_s=float(args.csv_segment_omega_tol),
            csv_segment_require_low_speed=not bool(args.csv_segment_skip_vel_gate),
            csv_segment_timeout_factor=float(args.csv_segment_timeout_factor),
            csv_segment_timeout_min_s=float(args.csv_segment_timeout_min_s),
            csv_segment_time_scale=float(args.csv_segment_time_scale),
        )
    results = test.run(**run_kw)
    print(f"\nResults: {results}")

    if save_dir is None:
        test.plot_results()
    else:
        test.plot_results(save_dir=save_dir)

    # Let process teardown reclaim the PyBullet connection.  In this environment
    # explicit disconnect can abort after all plots/videos have already saved.
    pass


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
