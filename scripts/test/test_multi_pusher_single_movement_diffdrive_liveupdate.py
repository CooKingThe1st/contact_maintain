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
python3 /home/docker_user/catkin_ws/src/contact_maintain/scripts/test/test_multi_pusher_single_movement_diffdrive_liveupdate.py   --object rect   --v-ref-x 0.02   --v-ref-y 0.0   --omega-ref 0.02   --duration 50   --fixed-ref --kp-position 0.1   --k-tangent 0.1   --k-couple 0.0   --k-force-comp 0.0   --kd-alpha 0.08   --kd-pos 0.2   --save-dir /tmp/multi_pusher_dd/ --record-video
    Push phase (Phase 7 — live-resolve variant)
--------------------------------------------
Each robot i independently, every control tick:
  1. Re-solves (vr_ff_i, zeta0_i, alpha*_i) from the CURRENT contact geometry
     (phi_live, v_cp_ref_live) via _init_segment_reference.  This replaces the
     single one-shot solve used in the baseline version and eliminates stale-
     reference drift without requiring any periodic stop/re-align.
  2. Runs _compute_phase7_command with the fresh seg_ref AND a PD alpha term:
       omega_r = omega_ff + kp_alpha*e_alpha + kd_alpha*(de_alpha/dt)
     The derivative term damps multi-robot coupling oscillations.
The object moves from real contact-force physics — no kinematic cheat.

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
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
DEFAULT_OBJECT_FRICTION = 0.8

ROBOT_RADIUS = 0.06          # disc-bumper cylinder radius (diffdrive_wheel_robot_disc_bumper.urdf)
APPROACH_DISTANCE = 0.12      # spawn offset beyond contact point (metres)

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
        bumper_contact_mu: float = 0.8,
        wheel_lateral_friction: float = DEFAULT_WHEEL_LATERAL_FRICTION,
        caster_lateral_friction: float = DEFAULT_CASTER_LATERAL_FRICTION,
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
        obstructing_passive_ratio: float = 0.25,
        obstructing_inflate_gap: float = 0.02,
        couple_obstructing_only: bool = True,
        use_actual_contact_clearance_cheat: bool = True,
    ):
        self.n_robots = len(t_params)
        assert self.n_robots >= 1, "Need at least one t_param."
        self.t_params = [float(t) for t in t_params]
        self.v_ref_body = np.asarray(v_ref_body, dtype=float).reshape(2)
        self.omega_ref = float(omega_ref)

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
            lateral_friction=DEFAULT_OBJECT_FRICTION,
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
                contact_mu=float(bumper_contact_mu),
                name=name,
            )
            robot.set_wheel_friction(float(wheel_lateral_friction))
            robot.set_caster_friction(float(caster_lateral_friction))
            robot.use_planar_cheat_control = True

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
        # APPROACH → (all in contact) → REALIGN → (all aligned) → HOLD → PUSH
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

        # Per-robot state for the derivative terms and fixed-ref alpha* snap.
        e_alpha_prev_list = [0.0] * self.n_robots
        e_pos_prev_list   = [0.0] * self.n_robots  # for position D term
        e_tangent_prev_list = [0.0] * self.n_robots
        alpha_snapped = [False] * self.n_robots   # only used when use_live_resolve=False

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

                # ── Swarm update: manages approach state machine + agent goals ──
                # This calls agent.update_contact_state() for each robot, keeping
                # agent.contact_force / agent.in_contact fresh.
                self.host.update(1.0 / CTRL_FREQ, obj_state_dict)

                # ── Check our own approach completion flags ─────────────────────
                for i, name in enumerate(names):
                    if not approach_complete[i]:
                        if self.agents[name].contact_force > APPROACH_CONTACT_GATE:
                            approach_complete[i] = True
                            print(
                                f"[{name}] contact detected "
                                f"(F={self.agents[name].contact_force:.3f} N, t={t:.2f}s) — "
                                f"waiting for remaining robots..."
                            )

                all_in_contact = all(approach_complete)

                # ── Barrier 1: all in contact → announce realign ───────────────
                if all_in_contact and not realign_announced:
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
                    self._t_push_start = t
                    print(f"\n{'='*60}")
                    print(
                        f"ALL {self.n_robots} ROBOTS — PUSH PHASE START (t={t:.2f}s)"
                    )

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
                if push_started:
                    servo_scale = self.live_object_servo_scale if self.use_live_resolve else 1.0
                    v_obj_corr = servo_scale * self.kp_obj_speed * (v_ref_world - obj_vel)
                    corr_norm = float(np.linalg.norm(v_obj_corr))
                    if self.max_object_v_correction > 0.0 and corr_norm > self.max_object_v_correction:
                        v_obj_corr = v_obj_corr * (self.max_object_v_correction / max(corr_norm, 1e-9))
                    omega_obj_corr = float(np.clip(
                        servo_scale * self.kp_object_omega * (self.omega_ref - obj_omega),
                        -self.max_object_omega_correction,
                        self.max_object_omega_correction,
                    ))
                    v_eff_world = v_ref_world + v_obj_corr
                    omega_eff = float(self.omega_ref + omega_obj_corr)
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

                    # ── APPROACH → REALIGN → HOLD → PUSH ──────────────────────
                    if not all_in_contact:
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
                        if self.use_live_resolve:
                            # ── Live re-solve: seg_ref rebuilt from current pose
                            # every tick.  By default, keep the forward/backward
                            # branch selected at realign-start to avoid pi-jump
                            # discontinuities in vr_ff / alpha*.
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
                            # ── Fixed reference (original baseline): one-shot
                            # solve from realign-start + single alpha* snap at
                            # push entry.  Use this to isolate the PD effect.
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
                    alpha_star_i = (
                        self._seg_refs[i]["alpha_star"] if self._seg_refs[i] is not None else 0.0
                    )
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

            pyb.stepSimulation()
            if gui:
                time.sleep(TIMESTEP * 0.3)
            t += TIMESTEP
            step_count += 1

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

        return {
            "push_started_at_s": t_push_start,
            "push_duration_s": push_duration,
            "mean_position_errors_m": mean_pos_errs,
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
            fig_traj.suptitle(
                "Multi-pusher diff-drive: trajectories\n"
                f"v_ref=({self.v_ref_body[0]:+.3f}, {self.v_ref_body[1]:+.3f}) m/s  "
                f"ω_ref={self.omega_ref:+.3f} rad/s",
                fontsize=12,
            )

            # Object CoM trajectory (spans left 2 columns of top row)
            ax_obj = fig_traj.add_subplot(gs[0, :2])
            ax_obj.plot(
                obj_pos_arr[:, 0], obj_pos_arr[:, 1],
                "k-", lw=2.0, label="Object CoM", alpha=0.85,
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
            if len(obj_pos_arr):
                ax_obj.plot(*obj_pos_arr[0], "go", ms=8, label="Start", zorder=6)
                ax_obj.plot(*obj_pos_arr[-1], "rs", ms=8, label="End", zorder=6)
            ax_obj.set_xlabel("X (m)")
            ax_obj.set_ylabel("Y (m)")
            ax_obj.set_title("Object CoM trajectory (+ robot paths)", fontsize=11)
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

        col_titles = ["Position error (cm)", "Contact force (N)", "Alpha error (deg)", "v_r cmd (m/s)"]
        for j, title in enumerate(col_titles):
            axes[0][j].set_title(title, fontsize=9)

        for i, (name, hist) in enumerate(zip(names, self.robot_histories)):
            if not hist.times:
                continue
            t_arr = np.array(hist.times)
            pos_err_cm = np.array([np.linalg.norm(e) for e in hist.position_errors]) * 100.0
            contact_f = np.array(hist.contact_forces)
            alpha_err_deg = np.degrees(np.array(hist.alpha_errors))
            v_r_arr = np.array(hist.v_r_history)

            axes[i][0].plot(t_arr, pos_err_cm, lw=1.0)
            axes[i][0].set_ylabel(f"{name}\n(cm)", fontsize=8)
            axes[i][0].grid(True, alpha=0.3)

            axes[i][1].plot(t_arr, contact_f, color="tab:red", lw=1.0)
            axes[i][1].axhline(0.5, ls="--", color="gray", lw=0.8, label="0.5 N gate")
            axes[i][1].set_ylabel("(N)", fontsize=8)
            axes[i][1].grid(True, alpha=0.3)

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
        # Only plot from push-start to avoid the approach/realign outliers.
        push_start_idx = 0
        if self.robot_histories[0].times and self._t_push_start is not None:
            t_arr_full = np.array(self.robot_histories[0].times)
            idx = np.searchsorted(t_arr_full, self._t_push_start)
            push_start_idx = max(0, int(idx))

        def _set_pruned_ylim(ax, series_list, q_low: float = 2.0, q_high: float = 98.0) -> None:
            """Use percentile y-limits so rare spikes do not hide steady behavior."""
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
            _set_pruned_ylim(ax_v, [
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
            _set_pruned_ylim(ax_w, [
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
            _set_pruned_ylim(ax_match, [
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
            _set_pruned_ylim(ax_omega_match, [intended_omega, object_omega, robot_omega, omega_cmd])
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

        # Rotate v_ref_body into world frame at each tick for the reference line.
        c_t = np.cos(theta_arr)
        s_t = np.sin(theta_arr)
        vx_ref = c_t * self.v_ref_body[0] - s_t * self.v_ref_body[1]
        vy_ref = s_t * self.v_ref_body[0] + c_t * self.v_ref_body[1]

        # Rotate actual world velocity into the object body frame.  The reference
        # is constant in this local frame, so drift is easier to see here.
        vx_body = c_t * v_arr[:, 0] + s_t * v_arr[:, 1]
        vy_body = -s_t * v_arr[:, 0] + c_t * v_arr[:, 1]
        speed_body = np.linalg.norm(np.column_stack([vx_body, vy_body]), axis=1)
        speed_ref = float(np.linalg.norm(self.v_ref_body))

        fig2, axes2 = plt.subplots(2, 3, figsize=(14, 6))
        fig2.suptitle("Object velocity: actual vs desired reference", fontsize=10)

        axes2[0][0].plot(t_arr, v_arr[:, 0], lw=1.0, label="actual vx")
        axes2[0][0].plot(t_arr, vx_ref, "--", lw=1.0, label="ref vx")
        axes2[0][0].set_title("vx (world)")
        axes2[0][0].legend(fontsize=7)
        axes2[0][0].grid(True, alpha=0.3)

        axes2[0][1].plot(t_arr, v_arr[:, 1], lw=1.0, label="actual vy")
        axes2[0][1].plot(t_arr, vy_ref, "--", lw=1.0, label="ref vy")
        axes2[0][1].set_title("vy (world)")
        axes2[0][1].legend(fontsize=7)
        axes2[0][1].grid(True, alpha=0.3)

        axes2[0][2].plot(t_arr, om_arr, lw=1.0, label="actual omega")
        axes2[0][2].axhline(self.omega_ref, ls="--", lw=1.0, label=f"ref {self.omega_ref:+.3f}")
        axes2[0][2].set_title("omega (rad/s)")
        axes2[0][2].legend(fontsize=7)
        axes2[0][2].grid(True, alpha=0.3)

        axes2[1][0].plot(t_arr, vx_body, lw=1.0, label="actual vx body")
        axes2[1][0].axhline(self.v_ref_body[0], ls="--", lw=1.0, label=f"ref {self.v_ref_body[0]:+.3f}")
        axes2[1][0].set_title("vx (object body)")
        axes2[1][0].legend(fontsize=7)
        axes2[1][0].grid(True, alpha=0.3)

        axes2[1][1].plot(t_arr, vy_body, lw=1.0, label="actual vy body")
        axes2[1][1].axhline(self.v_ref_body[1], ls="--", lw=1.0, label=f"ref {self.v_ref_body[1]:+.3f}")
        axes2[1][1].set_title("vy (object body)")
        axes2[1][1].legend(fontsize=7)
        axes2[1][1].grid(True, alpha=0.3)

        axes2[1][2].plot(t_arr, speed_body, lw=1.0, label="actual |v_body|")
        axes2[1][2].axhline(speed_ref, ls="--", lw=1.0, label=f"ref {speed_ref:.3f}")
        axes2[1][2].set_title("speed magnitude (body)")
        axes2[1][2].legend(fontsize=7)
        axes2[1][2].grid(True, alpha=0.3)

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
    parser.add_argument("--wheel-friction", type=float, default=DEFAULT_WHEEL_LATERAL_FRICTION)
    parser.add_argument("--caster-friction", type=float, default=DEFAULT_CASTER_LATERAL_FRICTION)
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
        "--obstructing-pusher-speed-scale", type=float, default=1.1,
        help=(
            "Diagnostic cheat applied only during PUSH: multiply the signed "
            "forward speed of precomputed obstructing pushers by this factor. "
            "Obstructing pushers are contacts with normal_ratio < "
            "-abs(--obstructing-passive-ratio). Default: 1.1"
        ),
    )
    parser.add_argument(
        "--obstructing-passive-ratio", type=float, default=0.25,
        help=(
            "Normalized normal-ratio threshold for obstructing pusher detection. "
            "normal_ratio = cos(true_alpha); robots below -abs(value) are scaled. "
            "Default: 0.25"
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
            lateral_friction=DEFAULT_OBJECT_FRICTION,
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
        wheel_lateral_friction=float(args.wheel_friction),
        caster_lateral_friction=float(args.caster_friction),
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
    )

    ref_mode_str = "fixed-ref (one-shot)" if args.fixed_ref else "live-resolve (every tick)"
    print(f"\n[config] N={len(t_params)} robots (Magnum Four)")
    print(f"  t_params  : {[round(t, 4) for t in t_params]}")
    print(f"  v_ref_body: ({args.v_ref_x:.4f}, {args.v_ref_y:.4f}) m/s")
    print(f"  omega_ref : {args.omega_ref:.4f} rad/s")
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
        f"  friction  : ground={args.ground_friction:.3f}  "
        f"wheel={args.wheel_friction:.3f}  caster={args.caster_friction:.3f}"
    )
    print()

    save_dir = Path(args.save_dir) if args.save_dir else None
    results = test.run(
        duration=args.duration,
        gui=not args.no_gui,
        debug_vel=args.debug_vel,
        debug_every=args.debug_vel_every,
        save_dir=save_dir,
        record_video=args.record_video,
        align_heading_tol_rad=np.deg2rad(args.align_heading_tol_deg),
        stop_go_sleep_after_realign_s=float(args.stop_go_pre_push_hold_s),
    )
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
