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

    Push phase (Phase 7)
--------------------
Each robot i independently:
  1. Solves (vr_ff_i, omega_ff_i, zeta0_i, alpha*_i) via _init_segment_reference
     using the shared (v_ref_body, omega_ref) evaluated at robot i's contact
     geometry at the moment all robots enter contact (realign-start tick).
  2. Runs _compute_phase7_command every control tick.
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
 python3 test_multi_pusher_single_movement_diffdrive.py   --object rect    --v-ref-x 0.05 --v-ref-y 0 --omega-ref 0.05   --duration 50 --save-dir /tmp/multi_pusher_dd
"""

import argparse
import json
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
    intended_positions: List[np.ndarray] = field(default_factory=list)
    position_errors: List[np.ndarray] = field(default_factory=list)
    contact_point_velocities: List[np.ndarray] = field(default_factory=list)
    contact_forces: List[float] = field(default_factory=list)
    in_contact: List[bool] = field(default_factory=list)
    alpha_errors: List[float] = field(default_factory=list)
    v_r_history: List[float] = field(default_factory=list)
    omega_r_history: List[float] = field(default_factory=list)
    # ── v_r decomposition ─────────────────────────────────────────────────────
    v_ff_history: List[float] = field(default_factory=list)       # vr_ff (feed-forward)
    v_base_history: List[float] = field(default_factory=list)     # kp_pos * e_pos (position P)
    v_speed_p_history: List[float] = field(default_factory=list)  # kp_speed * speed_err
    # ── omega_r decomposition ─────────────────────────────────────────────────
    omega_ff_history: List[float] = field(default_factory=list)       # omega_ff (feed-forward)
    omega_alpha_p_history: List[float] = field(default_factory=list)  # kp_alpha * e_alpha (alpha P)


@dataclass
class ObjectHistory:
    """Object pose / velocity, recorded every control tick."""
    times: List[float] = field(default_factory=list)
    positions: List[np.ndarray] = field(default_factory=list)
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
            cameraDistance=3.0, cameraYaw=45, cameraPitch=-45,
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
        cameraDistance=4, cameraYaw=0, cameraPitch=-89,
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

    def _wrap(x: float) -> float:
        return float(np.arctan2(np.sin(x), np.cos(x)))

    # Pick the heading branch (forward / backward) closest to current robot heading.
    if abs(_wrap(zeta_fwd - robot_heading)) <= abs(_wrap(zeta_bwd - robot_heading)):
        zeta0, vr_ff = zeta_fwd, vr_mag
    else:
        zeta0, vr_ff = zeta_bwd, -vr_mag

    return {
        "vr_ff": float(vr_ff),
        "omega_ff": float(omega_ref),
        "zeta0": float(zeta0),
        "alpha_star": float(_wrap(phi0 - zeta0)),
    }


def _compute_phase7_command(
    seg_ref: Dict,
    robot_heading: float,
    position_error: np.ndarray,
    current_alpha: float,
    desired_cp_speed: float,
    actual_object_speed: float,
    kp_position: float,
    kp_obj_speed: float,
    kp_alpha: float,
    max_v_r: float,
    max_omega_r: float,
) -> Tuple[float, float, Dict]:
    """Phase-7 fixed-alpha command for one robot.

    v_r     = vr_ff  +  kp_pos * e_pos_along_drive  +  kp_speed * e_speed
    omega_r = omega_ff  +  kp_alpha * e_alpha

    Mirrors compute_diffdrive_phase7_command in test_single_pusher_diffdrive.py.

    # ── TODO (Problem 1 — alpha tracking improvements) ────────────────────────
    # Option A — Adaptive omega_ff:
    #   Replace the fixed  omega_ff = omega_ref  with the MEASURED object angular
    #   velocity  omega_obj_measured  (low-pass filtered).  This removes the
    #   systematic feedforward mismatch when the object spins slower than omega_ref
    #   due to insufficient force.  The P-term then only fights the residual.
    #   Each robot reads the measured omega independently (still decentralised).
    #   Implementation sketch:
    #       omega_ff_adaptive = lpf(omega_obj_measured) + geometric_offset_i
    #       omega_r = clip(omega_ff_adaptive + kp_alpha * e_alpha, ...)
    #
    # Option C — PD on alpha (add derivative term):
    #   Add a  kd_alpha * d(e_alpha)/dt  term to omega_r to damp the multi-robot
    #   coupling oscillations where independent P-loops fight each other through
    #   the shared object.  Does NOT fix steady-state but reduces oscillation
    #   amplitude.  Requires storing e_alpha from the previous tick per robot.
    #       e_alpha_dot_i = (e_alpha_i - e_alpha_prev_i) / dt
    #       omega_r += kd_alpha * e_alpha_dot_i
    # ──────────────────────────────────────────────────────────────────────────
    """
    drive_dir = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
    e_pos = float(np.dot(np.asarray(position_error, dtype=float).reshape(2), drive_dir))
    v_base = kp_position * e_pos

    speed_err = desired_cp_speed - actual_object_speed
    v_speed_p = kp_obj_speed * speed_err

    vr_ff = float(seg_ref["vr_ff"])
    v_r = float(np.clip(vr_ff + v_base + v_speed_p, -max_v_r, max_v_r))

    alpha_star = float(seg_ref["alpha_star"])
    e_alpha = float(np.arctan2(
        np.sin(current_alpha - alpha_star),
        np.cos(current_alpha - alpha_star),
    ))
    omega_ff = float(seg_ref["omega_ff"])
    omega_alpha_p = float(kp_alpha * e_alpha)
    omega_r = float(np.clip(omega_ff + omega_alpha_p, -max_omega_r, max_omega_r))

    return v_r, omega_r, {
        # v_r decomposition
        "vr_ff": vr_ff, "v_base": v_base, "v_speed_p": v_speed_p,
        # omega_r decomposition
        "omega_ff": omega_ff, "omega_alpha_p": omega_alpha_p,
        # raw errors (for diagnostics)
        "e_alpha": e_alpha, "e_pos": e_pos, "speed_err": speed_err,
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
        kp_position: float = 1.0,
        kp_obj_speed: float = 1.0,
        max_forward_speed: float = 0.5,
        max_omega: float = 1.2,
        kp_realign_heading: float = 3.5,
    ):
        self.n_robots = len(t_params)
        assert self.n_robots >= 1, "Need at least one t_param."
        self.t_params = [float(t) for t in t_params]
        self.v_ref_body = np.asarray(v_ref_body, dtype=float).reshape(2)
        self.omega_ref = float(omega_ref)

        self.kp_alpha = float(kp_alpha)
        self.kp_position = float(kp_position)
        self.kp_obj_speed = float(kp_obj_speed)
        self.max_forward_speed = float(max_forward_speed)
        self.max_omega = float(max_omega)
        self.kp_realign_heading = float(kp_realign_heading)

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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _robot_names(self) -> List[str]:
        return [f"R_{i + 1:02d}" for i in range(self.n_robots)]

    def _get_object_state(self):
        """Return (pos2d, theta, vel2d, omega) for the pushed object."""
        p3, orn = pyb.getBasePositionAndOrientation(self.object_uid)
        vl, va = pyb.getBaseVelocity(self.object_uid)
        pos2d = np.array([p3[0], p3[1]], dtype=float)
        theta = float(pyb.getEulerFromQuaternion(orn)[2])
        vel2d = np.array([vl[0], vl[1]], dtype=float)
        omega = float(va[2])
        return pos2d, theta, vel2d, omega

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
        alpha_snapped = [False] * self.n_robots   # per-robot alpha* snap at push entry
        t_push_start: Optional[float] = None

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
                obj_pos, obj_theta, obj_vel, obj_omega = self._get_object_state()
                obj_speed = float(np.linalg.norm(obj_vel))
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
                    print(f"\n{'='*60}")
                    print(
                        f"ALL {self.n_robots} ROBOTS — PUSH PHASE START (t={t:.2f}s)"
                    )

                # ── Rotation matrix for this control tick ──────────────────────
                c_th, s_th = np.cos(obj_theta), np.sin(obj_theta)
                R_obj = np.array([[c_th, -s_th], [s_th, c_th]], dtype=float)

                k_ctrl = step_count // CTRL_STEP

                # ── Per-robot command ──────────────────────────────────────────
                for i, name in enumerate(names):
                    agent = self.agents[name]
                    robot = self.robots[name]
                    hist = self.robot_histories[i]

                    cp_b = self._cp_body[i]
                    n_out_b = self._n_out_body[i]

                    # Contact point and normal in world frame.
                    cp_world = R_obj @ cp_b + obj_pos
                    n_out_w = R_obj @ n_out_b
                    n_in_w = -n_out_w
                    intended_pos = cp_world + ROBOT_RADIUS * n_out_w

                    # Robot state.
                    robot_pos3, robot_heading, _ = robot.get_state()
                    robot_pos2 = np.asarray(robot_pos3, dtype=float)[:2]
                    position_error = intended_pos - robot_pos2

                    # Contact-point velocity from rigid-body kinematics.
                    r_cp = cp_world - obj_pos
                    v_rot = obj_omega * np.array([-r_cp[1], r_cp[0]], dtype=float)
                    cp_velocity = obj_vel + v_rot

                    # phi: world-frame inward-normal angle (same definition as single-pusher).
                    phi = float(np.arctan2(n_in_w[1], n_in_w[0]))
                    current_alpha = float(np.arctan2(
                        np.sin(phi - robot_heading),
                        np.cos(phi - robot_heading),
                    ))

                    contact_force = float(agent.contact_force)

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
                            "omega_ff": 0.0, "omega_alpha_p": 0.0, "e_alpha": 0.0,
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
                            v_cp_ref_w = _compute_world_cp_velocity_ref(
                                obj_theta, self.v_ref_body, self.omega_ref, r_cp
                            )
                            seg_ref = _init_segment_reference(
                                phi0=phi,
                                v_cp_ref_world=v_cp_ref_w,
                                omega_ref=self.omega_ref,
                                robot_heading=robot_heading,
                            )
                            self._seg_refs[i] = seg_ref
                            v_cp_b = _compute_body_cp_velocity(cp_b, self.v_ref_body, self.omega_ref)
                            print(
                                f"[{name}] seg_ref solved at realign-start:\n"
                                f"   vr_ff={seg_ref['vr_ff']:+.4f} m/s   "
                                f"omega_ff={seg_ref['omega_ff']:+.4f} rad/s\n"
                                f"   zeta0={seg_ref['zeta0']:+.4f} rad   "
                                f"alpha*={seg_ref['alpha_star']:+.4f} rad\n"
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
                            "omega_ff": 0.0, "omega_alpha_p": 0.0, "e_alpha": e_zeta,
                        }

                        if debug_vel and k_ctrl % debug_every == 0:
                            phase_tag = "HOLD" if realign_complete[i] else "REALIGN"
                            print(
                                f"[{name} t={t:.2f}s {phase_tag}] "
                                f"e_zeta={e_zeta:+.4f} rad  aligned={realign_complete[i]}"
                            )

                    else:
                        # ── PUSH — phase-7 controller ──────────────────────────
                        # Option B alpha* snap: on the very first push tick, override
                        # the stale solve-time alpha* with the actual push-entry angle.
                        if not alpha_snapped[i]:
                            alpha_snapped[i] = True
                            alpha_entry = float(
                                np.arctan2(
                                    np.sin(phi - robot_heading),
                                    np.cos(phi - robot_heading),
                                )
                            )
                            alpha_star_old = float(self._seg_refs[i]["alpha_star"])
                            self._seg_refs[i]["alpha_star"] = alpha_entry
                            print(
                                f"[{name}] alpha* snapped to push-entry: "
                                f"{alpha_entry:.4f} rad "
                                f"(was {alpha_star_old:.4f}, "
                                f"delta={alpha_entry - alpha_star_old:+.4f} rad)"
                            )

                        v_r, omega_r, dbg = _compute_phase7_command(
                            seg_ref=self._seg_refs[i],
                            robot_heading=robot_heading,
                            position_error=position_error,
                            current_alpha=current_alpha,
                            desired_cp_speed=self._desired_cp_speed[i],
                            actual_object_speed=obj_speed,
                            kp_position=self.kp_position,
                            kp_obj_speed=self.kp_obj_speed,
                            kp_alpha=self.kp_alpha,
                            max_v_r=self.max_forward_speed,
                            max_omega_r=self.max_omega,
                        )

                        if debug_vel and k_ctrl % debug_every == 0:
                            v_cp_star = _compute_world_cp_velocity_ref(
                                obj_theta, self.v_ref_body, self.omega_ref, r_cp
                            )
                            print(
                                f"[{name} t={t:.2f}s PUSH] "
                                f"v_cp*=({v_cp_star[0]:+.3f},{v_cp_star[1]:+.3f}) m/s  "
                                f"v_cp_act=({cp_velocity[0]:+.3f},{cp_velocity[1]:+.3f}) m/s  "
                                f"v_r={v_r:+.3f} omega_r={omega_r:+.3f}  "
                                f"e_alpha={dbg['e_alpha']:+.3f} rad  |F|={contact_force:.2f} N"
                            )

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
                    hist.intended_positions.append(intended_pos.copy())
                    hist.position_errors.append(position_error.copy())
                    hist.contact_point_velocities.append(cp_velocity.copy())
                    hist.contact_forces.append(float(contact_force))
                    hist.in_contact.append(contact_force > 0.5)
                    hist.alpha_errors.append(e_alpha_hist)
                    hist.v_r_history.append(float(v_r))
                    hist.omega_r_history.append(float(omega_r))
                    hist.v_ff_history.append(float(dbg["vr_ff"]))
                    hist.v_base_history.append(float(dbg["v_base"]))
                    hist.v_speed_p_history.append(float(dbg["v_speed_p"]))
                    hist.omega_ff_history.append(float(dbg["omega_ff"]))
                    hist.omega_alpha_p_history.append(float(dbg["omega_alpha_p"]))

                # ── Object history ─────────────────────────────────────────────
                self.object_history.times.append(t)
                self.object_history.positions.append(obj_pos.copy())
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
            self.n_robots, 2,
            figsize=(14, 3 * self.n_robots + 1),
            squeeze=False,
        )
        fig3.suptitle(
            f"Multi-pusher diff-drive  N={self.n_robots} — controller term decomposition\n"
            f"v_r = vr_ff + v_base(pos_P) + v_speed_p(speed_P)    "
            f"ω_r = ω_ff + kp_α·e_α",
            fontsize=9,
        )
        for i, (name, hist) in enumerate(zip(names, self.robot_histories)):
            if not hist.times:
                continue
            t_arr = np.array(hist.times)

            # ── left: v_r decomposition ────────────────────────────────────
            ax_v = axes3[i][0]
            ax_v.plot(t_arr, hist.v_ff_history,      lw=1.0, label="vr_ff (FF)")
            ax_v.plot(t_arr, hist.v_base_history,    lw=1.0, label="v_base (pos P)", ls="--")
            ax_v.plot(t_arr, hist.v_speed_p_history, lw=1.0, label="v_speed_p (spd P)", ls=":")
            ax_v.plot(t_arr, hist.v_r_history,       lw=1.3, label="v_r total", color="k")
            ax_v.axhline(0.0, ls="-", lw=0.5, color="gray")
            ax_v.set_ylabel(f"{name}\n(m/s)", fontsize=8)
            ax_v.legend(fontsize=7, loc="upper right")
            ax_v.grid(True, alpha=0.3)
            if i == 0:
                ax_v.set_title("v_r decomposition", fontsize=9)

            # ── right: omega_r decomposition ───────────────────────────────
            ax_w = axes3[i][1]
            ax_w.plot(t_arr, hist.omega_ff_history,      lw=1.0, label="ω_ff (FF)")
            ax_w.plot(t_arr, hist.omega_alpha_p_history, lw=1.0, label="kp_α·e_α (alpha P)", ls="--")
            ax_w.plot(t_arr, hist.omega_r_history,       lw=1.3, label="ω_r total", color="k")
            ax_w.axhline(0.0, ls="-", lw=0.5, color="gray")
            ax_w.set_ylabel("(rad/s)", fontsize=8)
            ax_w.legend(fontsize=7, loc="upper right")
            ax_w.grid(True, alpha=0.3)
            if i == 0:
                ax_w.set_title("ω_r decomposition   ", fontsize=9)

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

        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 3))
        fig2.suptitle("Object velocity: actual vs desired reference", fontsize=10)

        axes2[0].plot(t_arr, v_arr[:, 0], lw=1.0, label="actual vx")
        axes2[0].plot(t_arr, vx_ref, "--", lw=1.0, label="ref vx")
        axes2[0].set_title("vx (world)"); axes2[0].legend(fontsize=7); axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(t_arr, v_arr[:, 1], lw=1.0, label="actual vy")
        axes2[1].plot(t_arr, vy_ref, "--", lw=1.0, label="ref vy")
        axes2[1].set_title("vy (world)"); axes2[1].legend(fontsize=7); axes2[1].grid(True, alpha=0.3)

        axes2[2].plot(t_arr, om_arr, lw=1.0, label="actual omega")
        axes2[2].axhline(self.omega_ref, ls="--", lw=1.0, label=f"ref {self.omega_ref:+.3f}")
        axes2[2].set_title("omega (rad/s)"); axes2[2].legend(fontsize=7); axes2[2].grid(True, alpha=0.3)

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
    parser.add_argument("--kp-position", type=float, default=1.0)
    parser.add_argument("--kp-obj-speed", type=float, default=1.0)
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
            mass=1.0,
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
        kp_position=args.kp_position,
        kp_obj_speed=args.kp_obj_speed,
    )

    print(f"\n[config] N={len(t_params)} robots (Magnum Four)")
    print(f"  t_params  : {[round(t, 4) for t in t_params]}")
    print(f"  v_ref_body: ({args.v_ref_x:.4f}, {args.v_ref_y:.4f}) m/s")
    print(f"  omega_ref : {args.omega_ref:.4f} rad/s")
    print(f"  |v_cp_body| per robot: {[round(s, 4) for s in test._desired_cp_speed]}")
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

    pyb.disconnect()


if __name__ == "__main__":
    main()
