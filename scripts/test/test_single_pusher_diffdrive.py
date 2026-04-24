#!/usr/bin/env python3
"""
Single-pusher diff-drive test focused on Phase 7 structure.

Robot: DiffDriveWheelPhysicsRobot (disc bumper URDF), same as
scripts/test/basic_test/test_diffdrive_wheel_physics.py. Default sim knobs
(ground / wheel / caster friction) match that script; planar joints are always
used for velocity actuation (same as that test with cheat control enabled).

Implements:
- fixed-alpha matching feed-forward solve per segment
- two transition modes: continuous and stop_go
- wrench-driven segment sequence generation from contact geometry
"""

# python3 /home/docker_user/catkin_ws/src/contact_maintain/scripts/test/test_single_pusher_diffdrive.py --object-name rect --t-param 0.04 --duration 60 --control-mode stop_go --scale-up-factor 0.05 --safety-gate 0.7 --normal-force 1.0 --segment-duration 3.0 --normal-only --save-dir /tmp/single_pusher_dd/ --align-heading-tol-deg 2 --debug-vel --record-video

#  python3 /home/docker_user/catkin_ws/src/contact_maintain/scripts/test/test_single_pusher_diffdrive.py --object-name rect --t-param 0.04 --duration 20 --control-mode stop_go --scale-up-factor 0.05 --safety-gate 0.7 --normal-force 1.0 --segment-duration 3.0 --normal-only --debug-vel 

# python3 /home/docker_user/catkin_ws/src/contact_maintain/scripts/test/test_single_pusher_diffdrive.py --object-name rect --t-param 0.125 --duration 12 --control-mode continuous --scale-up-factor 1.0 --safety-gate 0.7 --normal-force 1.0 --segment-duration 3.0 --normal-only --debug-vel 
# python3 /home/docker_user/catkin_ws/src/contact_maintain/scripts/test/test_single_pusher_diffdrive.py --object-name right_triangle --t-param 0.125 --duration 12 --control-mode stop_go --align-heading-tol-deg 5 --scale-up-factor 1.0 --safety-gate 0.7 --normal-force 1.0 --segment-duration 3.0
# python3 /home/docker_user/catkin_ws/src/contact_maintain/scripts/test/test_single_pusher_diffdrive.py --object-name right_triangle --t-param 0.125 --duration 12 --control-mode continuous --no-gui --save-dir /tmp/diffdrive_phase7
# ---------------------------------------------------------------------------
# CONTROL NOTES (from design discussion; keep for future double-check)
# ---------------------------------------------------------------------------
# For each constant object-twist segment:
# 1) Given object-side contact point, solve (do not pick) the intended
#    constant-contact geometry/state:
#       - intended alpha* (robot-side contact angle)
#       - intended zeta0* (robot heading at segment start)
#       - constant feed-forward pair (vr_ff, omega_ff) with:
#           omega_ff = omega_object
#           vr_ff = constant (from matching equations, vr_dot = 0)
# 2) Use three control objectives:
#       (a) Velocity matching at contact point (feed-forward base)
#       (b) Position regulation term added to vr
#       (c) Alpha-angle regulation term added to omega_r:
#               e_alpha = wrap(alpha_current - alpha*)
#               omega_r = omega_ff - k_alpha * e_alpha
#    (angle-level regulation is preferred here over dot-level for smooth mode)
# 3) Compare two operating modes:
#       - stop-and-go: re-align to solved zeta0* per segment before pushing
#       - continuous: shift to new references at segment transitions and track
# 4) Test setup note:
#    approach toward no-torque configuration (heading toward CoM) to reveal
#    behavior differences under Dubins-like actuation constraints.
# 5) Initial approach: rotate-then-creep P control toward boundary contact
#    (same structure as RobotAgent goal_type='approach'); wrench segments start
#    only after first contact force is sensed. Use --skip-approach to disable.
# 6) How to choose the desired object-twist segments (v_k, omega_k):
#    - v_k is still a 2D vector; practical knobs are:
#        (i) direction of v_k
#        (ii) magnitude of v_k
#        (iii) omega_k
#    - Use object/contact geometry to build physically meaningful segment values:
#      (a) Given object + intended contact point, compute the wrench at CoM from
#          a unit normal contact force at that point:
#              tau_z = r_x * F_y - r_y * F_x
#              wrench = [F_x, F_y, tau_z]
#      (b) Scale wrench by SCALE_UP_FACTOR (CLI arg).
#      (c) Convert to acceleration target using mass/inertia:
#              a_x = F_x / m, a_y = F_y / m, alpha_z = tau_z / I
#      (d) For this specific test harness, set the segment "twist command value"
#          directly from that scaled acceleration tuple (simple stress-test setup).
#    - Friction-cone side segments:
#      (a) With friction coefficient mu and safety_gate s (recommended 0.7), use
#          tangential force candidates:
#              F_t = -mu * F_n * s   (left-most)
#              F_t = +mu * F_n * s   (right-most)
#      (b) Combine each with the same normal force, compute wrench at CoM, then
#          map to (v_2, omega_2) and (v_3, omega_3) via the same scaling rule.
#    - Recommended smoother schedule (4 segments):
#        seg1: normal-only
#        seg2: left-most cone edge (-mu * Fn * s)
#        seg3: normal-only (returns toward center before flipping side)
#        seg4: right-most cone edge (+mu * Fn * s)
#      This reduces abrupt jump compared with directly switching left-most <-> right-most.
#    - Mode difference remains only transition handling:
#      * stop-and-go: at segment switch, pause matching and self-rotate to new zeta0*
#                     (or alpha*) before resuming push control.
#      * continuous:  no pause; switch references immediately and keep tracking.
# ---------------------------------------------------------------------------

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import ContactPointParameterization, create_standard_objects
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import obj_to_generic, generic_to_pybullet


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)
# After stop_go heading realign completes, pause before pushing this segment.
DEFAULT_STOP_GO_SLEEP_AFTER_REALIGN_S = 0.0

# Defaults aligned with test_diffdrive_wheel_physics.main() / setup_sim().
DEFAULT_GROUND_FRICTION = 0.5
DEFAULT_WHEEL_LATERAL_FRICTION = 0.01
DEFAULT_CASTER_LATERAL_FRICTION = 0.01

DEFAULT_OBJECT_SHAPE = "right_triangle"
DEFAULT_OBJECT_HEIGHT = 0.08
DEFAULT_OBJECT_FRICTION = 0.8
# For peer-to-peer comparison with scripts/test/test_single_pusher_omni.py.
# That script uses create_standard_objects()['rectangle'] (big rectangle) and a 0.2m height.
OMNI_NATIVE_RECT_HEIGHT = 0.2
# Disc bumper cylinder radius in diffdrive_wheel_robot_disc_bumper.urdf
ROBOT_RADIUS = 0.06


@dataclass
class Phase7TemplateHistory:
    times: List[float] = field(default_factory=list)
    robot_positions: List[np.ndarray] = field(default_factory=list)
    robot_headings: List[float] = field(default_factory=list)
    robot_velocities: List[np.ndarray] = field(default_factory=list)
    robot_cmd_velocities: List[np.ndarray] = field(default_factory=list)
    intended_positions: List[np.ndarray] = field(default_factory=list)
    position_errors: List[np.ndarray] = field(default_factory=list)
    desired_headings: List[float] = field(default_factory=list)
    heading_errors: List[float] = field(default_factory=list)
    object_positions: List[np.ndarray] = field(default_factory=list)
    object_orientations: List[float] = field(default_factory=list)
    object_velocities: List[np.ndarray] = field(default_factory=list)
    object_angular_velocities: List[float] = field(default_factory=list)
    contact_point_positions: List[np.ndarray] = field(default_factory=list)
    contact_point_velocities: List[np.ndarray] = field(default_factory=list)
    contact_forces: List[float] = field(default_factory=list)
    in_contact: List[bool] = field(default_factory=list)
    v_base_history: List[float] = field(default_factory=list)
    v_constant_history: List[float] = field(default_factory=list)
    v_velo_error_pi_history: List[float] = field(default_factory=list)


def setup_pybullet(gui: bool = True, ground_friction: float = DEFAULT_GROUND_FRICTION):
    """World init matching test_diffdrive_wheel_physics.setup_sim (no extra asset search paths)."""
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
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0],
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)


def setup_video_recording(video_path: Path, object_uid: int) -> int:
    """Start PyBullet MP4 recording from a fixed top-down view.

    Parameters
    ----------
    video_path : Path
        Absolute path for the output .mp4 file.
    object_uid : int
        PyBullet body ID of the object (used only for initial camera framing).

    Returns
    -------
    int
        Video logging ID (pass to stop_video_recording).
    """
    video_path = video_path.resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    if video_path.exists():
        print(f"Removing existing video file: {video_path}")
        video_path.unlink()

    # Fix camera to top-down view before recording starts.
    pos, _ = pyb.getBasePositionAndOrientation(object_uid)
    pyb.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=0,
        cameraPitch=-89,
        cameraTargetPosition=[pos[0], pos[1], 0],
    )

    video_log_id = pyb.startStateLogging(
        pyb.STATE_LOGGING_VIDEO_MP4,
        str(video_path),
    )
    if video_log_id < 0:
        raise RuntimeError(f"Failed to start video recording (log_id={video_log_id})")

    print(f"Video recording started → {video_path} (log_id={video_log_id})")
    return video_log_id


def stop_video_recording(video_log_id: int, video_path: Path) -> None:
    """Stop PyBullet MP4 recording and report the saved file size."""
    if video_log_id < 0:
        return
    pyb.stopStateLogging(video_log_id)
    print(f"Stopped video logging (ID: {video_log_id})")

    time.sleep(3.0)  # give PyBullet time to flush the file

    video_path = video_path.resolve()
    if video_path.exists():
        size_mb = video_path.stat().st_size / 1024 / 1024
        if size_mb > 0:
            print(f"✓ Video saved: {video_path} ({size_mb:.2f} MB)")
        else:
            print(f"⚠ Video file is empty (0 bytes): {video_path}")
    else:
        print(f"✗ Video file not found: {video_path}")
        mp4s = list(video_path.parent.glob("*.mp4"))
        if mp4s:
            print(f"  .mp4 files in directory: {mp4s}")


def closest_point_on_desired_segment(
    robot_pos: np.ndarray,
    object_pos: np.ndarray,
    object_orientation: float,
    seg_p1_body: np.ndarray,
    seg_p2_body: np.ndarray,
) -> Tuple[np.ndarray, float]:
    cos_t = np.cos(-object_orientation)
    sin_t = np.sin(-object_orientation)
    r_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    robot_local = r_inv @ (robot_pos - object_pos)

    seg_vec = seg_p2_body - seg_p1_body
    seg_len2 = float(seg_vec @ seg_vec)
    if seg_len2 < 1e-12:
        u_unclamped = 0.0
        closest_body = seg_p1_body.copy()
    else:
        u_unclamped = float(((robot_local - seg_p1_body) @ seg_vec) / seg_len2)
        u_clamped = float(np.clip(u_unclamped, 0.0, 1.0))
        closest_body = seg_p1_body + u_clamped * seg_vec

    cos_t = np.cos(object_orientation)
    sin_t = np.sin(object_orientation)
    r_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    closest_world = r_mat @ closest_body + object_pos
    return closest_world, u_unclamped


class SinglePusherDiffdriveTemplate:
    def __init__(
        self,
        t_param: float,
        approach_distance: float = 0.2,
        object_name: str = DEFAULT_OBJECT_SHAPE,
        *,
        bumper_contact_mu: float = 0.8,
        wheel_lateral_friction: float = DEFAULT_WHEEL_LATERAL_FRICTION,
        caster_lateral_friction: float = DEFAULT_CASTER_LATERAL_FRICTION,
        omni_native_rectangle: bool = False,
    ):
        self.t_param = float(t_param)
        self.approach_distance = float(approach_distance)
        self.history = Phase7TemplateHistory()
        self.contact_threshold = 0.5
        self.kp_heading = 10.0
        self.kp_alpha = 0.5  # Alpha-angle regulation gain
        self.kp_position_along_drive = 1.0
        self.kp_obj_speed = 1.0  # P gain: object CoM speed error → v_r correction (like omni's kp_vel)
        self.kp_realign_heading = 3.5
        self.max_forward_speed = 0.5
        self.max_omega = 1.2
        self.segment_ref: Optional[Dict[str, float]] = None

        # Initial approach (same idea as RobotAgent goal_type='approach' in robot_agent.py)
        self.approach_kp = 0.5
        self.approach_max_speed = 0.1
        self.approach_creep_speed = 0.03
        self.approach_close_effective_distance_m = 0.06
        self.approach_heading_threshold_rad = 0.05
        self.approach_stop_force_epsilon = 0.02
        self.approach_omega_rotate_gain = 2.0
        self.approach_omega_creep_gain = 10.0

        if omni_native_rectangle:
            # Peer-to-peer: use the same big native rectangle as test_single_pusher_omni.py.
            standard_objects = create_standard_objects()
            self.generic_object = standard_objects["rectangle"]

            # Match the omni test's dynamics override.
            self.generic_object.lateral_friction = DEFAULT_OBJECT_FRICTION
            self.generic_object.mass = 1.0
            self.generic_object.moment_of_inertia = self.generic_object._calculate_moment_of_inertia()

            self.object_uid = generic_to_pybullet(
                self.generic_object,
                height=OMNI_NATIVE_RECT_HEIGHT,
                position=(0.0, 0.0, 0.0),
                orientation=0.0,
                color=(0.4, 0.7, 0.4, 1.0),
            )
            pyb.changeDynamics(
                self.object_uid,
                -1,
                lateralFriction=float(DEFAULT_OBJECT_FRICTION),
                mass=1.0,
            )
            print("[object] omni_native_rectangle enabled (create_standard_objects()['rectangle'])")
        else:
            obj_file_map = {
                "right_triangle": "right_triangle.obj",
                "pi": "pi.obj",
                "root": "root.obj",
                "rect": "rect.obj",
                "hourglass": "hourglass.obj",
                "meteor": "meteor.obj",
            }
            if object_name not in obj_file_map:
                raise ValueError(f"Unknown object '{object_name}'. Available: {list(obj_file_map.keys())}")

            self.generic_object, self.object_uid = obj_to_generic(
                obj_path=obj_file_map[object_name],
                shape_name=object_name,
                position=(0.0, 0.0, DEFAULT_OBJECT_HEIGHT),
                orientation=0.0,
                mass=1.0,
                lateral_friction=DEFAULT_OBJECT_FRICTION,
                blind_test=True,
            )

        self.contact_friction_coeff = float(
            getattr(self.generic_object, "lateral_friction", DEFAULT_OBJECT_FRICTION)
        )

        self.parameterization = ContactPointParameterization(self.generic_object)
        info = self.parameterization.get_contact_info(self.t_param)
        self.contact_point_body = np.array(info["point"], dtype=float)
        self.normal_outward = np.array(info["normal_outward"], dtype=float)
        self.normal_inward = -self.normal_outward
        self.tangent = np.array(info["tangent"], dtype=float)

        _, seg_idx, _ = self.parameterization.parameter_to_point(self.t_param)
        self.desired_seg_p1_body = np.array(self.parameterization.boundary_coords[seg_idx], dtype=float)
        self.desired_seg_p2_body = np.array(self.parameterization.boundary_coords[seg_idx + 1], dtype=float)

        obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
        yaw = float(pyb.getEulerFromQuaternion(obj_orn)[2])
        c, s = np.cos(yaw), np.sin(yaw)
        r_mat = np.array([[c, -s], [s, c]], dtype=float)
        cp = np.asarray(self.contact_point_body, dtype=float).reshape(2)
        n_out = np.asarray(self.normal_outward, dtype=float).reshape(2)
        n_in = np.asarray(self.normal_inward, dtype=float).reshape(2)
        spawn_xy = r_mat @ (cp + float(self.approach_distance) * n_out) + np.array([obj_pos[0], obj_pos[1]], dtype=float)
        n_in_w = r_mat @ n_in
        heading = float(np.arctan2(n_in_w[1], n_in_w[0]))

        # DiffDriveWheelPhysicsRobot + diffdrive_wheel_robot_disc_bumper.urdf
        # Match test_diffdrive_wheel_physics.py: bumper mu, wheel/caster friction; planar cheat always on.
        self.robot = create_robot(
            kinematics="diffdrive",
            model="wheel_physics",
            position=(float(spawn_xy[0]), float(spawn_xy[1])),
            orientation=heading,
            contact_mu=float(bumper_contact_mu),
            name="single_pusher_diffdrive_template",
        )
        self.robot.set_wheel_friction(float(wheel_lateral_friction))
        self.robot.set_caster_friction(float(caster_lateral_friction))
        self.robot.use_planar_cheat_control = True

    def _get_contact_force(self) -> float:
        """Bumper |Fxy| vs object (same convention as test_diffdrive_wheel_physics force_sensor)."""
        f = self.robot.get_contact_force([self.object_uid], max_contacts=8)
        return float(np.linalg.norm(np.asarray(f[:2], dtype=float)))

    def _get_states(self):
        robot_pos, robot_heading, robot_vel = self.robot.get_state()
        obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
        obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
        euler = pyb.getEulerFromQuaternion(obj_orn)
        object_pos = np.array([obj_pos[0], obj_pos[1]])
        object_orientation = float(euler[2])
        object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
        object_angular_velocity = float(obj_vel_ang[2])
        return robot_pos, robot_heading, robot_vel, object_pos, object_orientation, object_velocity, object_angular_velocity

    def _set_object_twist_from_segment(self, object_orientation: float, active_seg: Dict[str, np.ndarray]) -> None:
        """Drive the object directly with the segment twist (debug / matching isolation mode)."""
        v_body = np.asarray(active_seg["v_ref"], dtype=float).reshape(2)
        c = np.cos(object_orientation)
        s = np.sin(object_orientation)
        r_mat = np.array([[c, -s], [s, c]], dtype=float)
        v_world = r_mat @ v_body
        omega_world = float(active_seg["omega_ref"])
        pyb.resetBaseVelocity(
            self.object_uid,
            linearVelocity=[float(v_world[0]), float(v_world[1]), 0.0],
            angularVelocity=[0.0, 0.0, omega_world],
        )

    def _stop_object_motion(self) -> None:
        """Hold the object still while stop_go realignment is happening."""
        pyb.resetBaseVelocity(
            self.object_uid,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )

    def _set_object_kinematic(self, zero_friction: bool) -> None:
        """Toggle object lateral friction on/off."""
        mu = 0.0 if zero_friction else float(self.contact_friction_coeff)
        pyb.changeDynamics(self.object_uid, -1, lateralFriction=mu)

    def cheat_drive_object_step(
        self,
        kin_pos: List[float],
        kin_theta: float,
        active_seg: Dict[str, np.ndarray],
    ) -> Tuple[List[float], float]:
        """True kinematic object drive: Euler-integrate pose then hard-reset position+velocity.

        Equivalent to the robot's planar-joint cheat control.  By resetting the
        object's *position* (not just velocity) every sim step we bypass the
        contact solver completely — no friction or integration artifact can
        accumulate, because we overwrite state before stepSimulation each tick.

        Returns updated (kin_pos, kin_theta) for the caller to store.
        """
        omega_ref = float(active_seg["omega_ref"])
        v_ref_body = np.asarray(active_seg["v_ref"], dtype=float).reshape(2)

        # Euler integrate pose (body-frame twist, current orientation).
        c, s = np.cos(kin_theta), np.sin(kin_theta)
        v_world = np.array([c * v_ref_body[0] - s * v_ref_body[1],
                            s * v_ref_body[0] + c * v_ref_body[1]], dtype=float)
        new_pos = [kin_pos[0] + float(v_world[0]) * TIMESTEP,
                   kin_pos[1] + float(v_world[1]) * TIMESTEP,
                   kin_pos[2]]
        new_theta = kin_theta + omega_ref * TIMESTEP
        new_quat = pyb.getQuaternionFromEuler([0.0, 0.0, new_theta])

        # Hard-reset pose — physics solver sees this as the authoritative state.
        pyb.resetBasePositionAndOrientation(self.object_uid, new_pos, new_quat)

        # Also set velocity so getBaseVelocity returns the correct value.
        c2, s2 = np.cos(new_theta), np.sin(new_theta)
        v_world2 = np.array([c2 * v_ref_body[0] - s2 * v_ref_body[1],
                              s2 * v_ref_body[0] + c2 * v_ref_body[1]], dtype=float)
        pyb.resetBaseVelocity(
            self.object_uid,
            linearVelocity=[float(v_world2[0]), float(v_world2[1]), 0.0],
            angularVelocity=[0.0, 0.0, float(omega_ref)],
        )
        return new_pos, new_theta

    def _compute_approach_diffdrive(
        self,
        robot_pos: np.ndarray,
        robot_heading: float,
        contact_point_world: np.ndarray,
        contact_force: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        """Rotate toward boundary contact, then creep in with P gain; map to (v_r, omega_r)."""
        if contact_force > self.approach_stop_force_epsilon:
            return 0.0, 0.0, {"v_base": 0.0, "v_ff": 0.0, "v_pi": 0.0}

        target_pos = contact_point_world
        to_target = target_pos - robot_pos[:2]
        dist = float(np.linalg.norm(to_target))
        if dist < 1e-6:
            return 0.0, 0.0, {"v_base": 0.0, "v_ff": 0.0, "v_pi": 0.0}

        target_heading = float(np.arctan2(to_target[1], to_target[0]))
        heading_error = float(
            np.arctan2(np.sin(target_heading - robot_heading), np.cos(target_heading - robot_heading))
        )

        if abs(heading_error) > self.approach_heading_threshold_rad:
            if contact_force > self.approach_stop_force_epsilon:
                return 0.0, 0.0, {"v_base": 0.0, "v_ff": 0.0, "v_pi": 0.0}
            omega = float(
                np.clip(self.approach_omega_rotate_gain * heading_error, -self.max_omega, self.max_omega)
            )
            return 0.0, omega, {"v_base": 0.0, "v_ff": 0.0, "v_pi": omega}

        direction = to_target / dist
        effective_distance = dist - ROBOT_RADIUS
        speed = max(min(
            self.approach_kp * max(0.0, effective_distance),
            self.approach_max_speed,
        ), 0.05)
        if effective_distance < self.approach_close_effective_distance_m:
            speed = min(speed, self.approach_creep_speed)
        if contact_force > self.approach_stop_force_epsilon:
            return 0.0, 0.0, {"v_base": 0.0, "v_ff": 0.0, "v_pi": 0.0}

        vel_xy = direction * max(0.0, speed)
        forward = np.array([np.cos(robot_heading), np.sin(robot_heading)])
        v_r = float(np.dot(vel_xy, forward))
        v_r = float(np.clip(v_r, -self.max_forward_speed, self.max_forward_speed))
        omega = float(
            np.clip(self.approach_omega_creep_gain * heading_error, -self.max_omega, self.max_omega)
        )
        return v_r, omega, {"v_base": v_r, "v_ff": 0.0, "v_pi": omega}

    def _build_wrench_sequence(
        self,
        scale_up_factor: float,
        safety_gate: float,
        normal_force: float,
        segment_duration: float,
        use_four_segments: bool = True,
        normal_only: bool = False,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Build desired object-twist sequence from contact wrench heuristics.

        For this specific test, we map scaled wrench/mass-inertia directly to
        the segment's (v_ref_xy, omega_ref) command values.
        """
        m = float(self.generic_object.mass)
        I = float(self.generic_object.moment_of_inertia)
        if m <= 1e-12 or I <= 1e-12:
            raise ValueError(f"Invalid object mass/inertia: m={m}, I={I}")

        r = self.contact_point_body  # contact position in object frame
        n = self.normal_inward        # pushing normal
        t = self.tangent              # tangent direction

        def wrench_to_twist(Fn: float, Ft: float, label: str) -> Dict[str, np.ndarray]:
            F = Fn * n + Ft * t
            tau = float(r[0] * F[1] - r[1] * F[0])
            F_scaled = scale_up_factor * F
            tau_scaled = scale_up_factor * tau
            v_ref = F_scaled / m
            omega_ref = tau_scaled / I
            return {
                "name": label,
                "duration": float(segment_duration),
                "v_ref": np.array([float(v_ref[0]), float(v_ref[1])], dtype=float),
                "omega_ref": float(omega_ref),
            }

        if normal_only:
            # Single segment: inward normal only (no tangential / cone-edge jumps).
            return [wrench_to_twist(normal_force, 0.0, "normal_only")]

        ft_left = -self.contact_friction_coeff * normal_force * safety_gate
        ft_right = self.contact_friction_coeff * normal_force * safety_gate
        seg_normal = wrench_to_twist(normal_force, 0.0, "normal_only")
        seg_left = wrench_to_twist(normal_force, ft_left, "left_cone")
        seg_right = wrench_to_twist(normal_force, ft_right, "right_cone")

        if use_four_segments:
            # Smoother jump: normal -> left -> normal -> right
            return [seg_normal, seg_left, seg_normal.copy(), seg_right]
        return [seg_normal, seg_left, seg_right]

    def compute_diffdrive_phase7_command(
        self,
        robot_heading: float,
        position_error: np.ndarray,
        heading_error: float,
        current_alpha: float,
        intended_alpha: float,
        desired_contact_point_speed: float,
        actual_contact_point_velocity: np.ndarray,
        actual_object_speed: float = 0.0,
        desired_object_velocity: Optional[np.ndarray] = None,
        desired_object_angular_velocity: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Placeholder for your upcoming alpha-based Phase 7 equations.

        Returns
        -------
        v_r : float
            Diff-drive forward speed command.
        omega_r : float
            Diff-drive yaw rate command.
        debug : Dict[str, float]
            Components for plotting/debugging (stored in v_* histories).
        """
        # Fixed-alpha matching scaffold:
        #   vr = vr_ff + k_pos * e_pos_projected
        #   omega_r = omega_ff - k_alpha * wrap(alpha - alpha*)
        # where vr_ff/omega_ff/alpha* are solved once per segment.
        if self.segment_ref is None:
            raise RuntimeError("segment_ref not initialized. Call _init_segment_reference first.")

        # Position feedback: keeps robot from losing contact (same role as v_base in omni).
        drive_dir = np.array([np.cos(robot_heading), np.sin(robot_heading)])
        e_pos_along_drive = float(np.dot(position_error, drive_dir))
        v_base = self.kp_position_along_drive * e_pos_along_drive

        # Speed P: corrects friction-induced velocity deficit (same role as v_velo_error_pi in omni).
        # Tracks object CoM speed magnitude — the "relaxed" scalar target (no contact-point geometry needed).
        speed_error = desired_contact_point_speed - actual_object_speed
        v_speed_p = self.kp_obj_speed * speed_error

        vr_ff = float(self.segment_ref["vr_ff"])
        v_r = vr_ff + v_base + v_speed_p
        v_r = float(np.clip(v_r, -self.max_forward_speed, self.max_forward_speed))

        # omega: ff sets the nominal heading rate; alpha feedback corrects long-term drift.
        # alpha* is snapped at push entry (Option B) so it reflects actual contact geometry,
        # not the stale solve-time geometry.
        omega_ff = float(self.segment_ref["omega_ff"])
        e_alpha = float(np.arctan2(np.sin(current_alpha - intended_alpha), np.cos(current_alpha - intended_alpha)))
        omega_r = omega_ff + self.kp_alpha * e_alpha
        omega_r = float(np.clip(omega_r, -self.max_omega, self.max_omega))

        return v_r, omega_r, {
            "v_base": v_base,
            "v_ff": vr_ff,
            "v_speed_p": v_speed_p,
            "speed_error": speed_error,
            "omega_ff": omega_ff,
            "e_alpha": e_alpha,
            "e_pos_along_drive": e_pos_along_drive,
            "v_r_before_clip": vr_ff + v_base + v_speed_p,
        }

    def _init_segment_reference(
        self,
        phi0: float,
        v_cp_ref: np.ndarray,
        omega_ref: float,
        robot_heading_now: float,
    ) -> None:
        """Solve fixed-alpha feed-forward reference from matching equations."""
        a = float(v_cp_ref[0] + omega_ref * ROBOT_RADIUS * np.sin(phi0))
        b = float(v_cp_ref[1] - omega_ref * ROBOT_RADIUS * np.cos(phi0))
        vr_mag = float(np.hypot(a, b))

        zeta_fwd = float(np.arctan2(b, a))
        zeta_bwd = float(np.arctan2(b, a) + np.pi)

        def wrap(x: float) -> float:
            return float(np.arctan2(np.sin(x), np.cos(x)))

        # Pick the branch (forward/backward) closer to current heading.
        err_fwd = abs(wrap(zeta_fwd - robot_heading_now))
        err_bwd = abs(wrap(zeta_bwd - robot_heading_now))
        if err_fwd <= err_bwd:
            zeta0 = zeta_fwd
            vr_ff = vr_mag
        else:
            zeta0 = zeta_bwd
            vr_ff = -vr_mag

        alpha_star = wrap(phi0 - zeta0)
        self.segment_ref = {
            "omega_ff": float(omega_ref),
            "vr_ff": float(vr_ff),
            "zeta0": float(zeta0),
            "alpha_star": float(alpha_star),
        }

    def _compute_world_contact_point_velocity_ref(
        self,
        object_orientation: float,
        v_ref_body: np.ndarray,
        omega_ref: float,
        r_cp_world: np.ndarray,
    ) -> np.ndarray:
        """Return world-frame contact-point velocity from body twist + world lever arm."""
        v_ref_b = np.asarray(v_ref_body, dtype=float).reshape(2)
        c = np.cos(object_orientation)
        s = np.sin(object_orientation)
        r_mat = np.array([[c, -s], [s, c]], dtype=float)
        v_ref_world = r_mat @ v_ref_b
        v_rot_world = float(omega_ref) * np.array(
            [-float(r_cp_world[1]), float(r_cp_world[0])], dtype=float
        )
        return v_ref_world + v_rot_world

    def _compute_body_contact_point_velocity_ref(
        self,
        v_ref_body: np.ndarray,
        omega_ref: float,
    ) -> np.ndarray:
        """Constant body-frame contact-point velocity for a rigid body segment."""
        v_ref_b = np.asarray(v_ref_body, dtype=float).reshape(2)
        r_b = np.asarray(self.contact_point_body, dtype=float).reshape(2)
        return v_ref_b + float(omega_ref) * np.array([-r_b[1], r_b[0]], dtype=float)

    def _run_loop(
        self,
        duration: float,
        gui: bool,
        desired_contact_point_speed: float,
        desired_object_velocity: Optional[np.ndarray] = None,
        desired_object_angular_velocity: Optional[float] = None,
        control_mode: str = "continuous",
        align_heading_tol_rad: float = np.deg2rad(5.0),
        scale_up_factor: float = 1.0,
        safety_gate: float = 0.7,
        normal_force: float = 1.0,
        segment_duration: float = 3.0,
        use_four_segments: bool = True,
        skip_approach: bool = False,
        normal_only: bool = False,
        debug_print_velocity: bool = False,
        debug_print_velocity_every: int = 5,
        solo_object_velo: bool = False,
        stop_go_sleep_after_realign_s: float = DEFAULT_STOP_GO_SLEEP_AFTER_REALIGN_S,
    ) -> Dict:
        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        self.segment_ref = None
        active_segment_idx = -1
        in_realign_mode = False
        pre_push_hold_until_t: Optional[float] = None
        approach_complete = bool(skip_approach)
        t_contact = 0.0
        solo_push_active = False
        # Segment pointer kept for per-sim-step object kinematic cheat (solo_object_velo).
        _solo_active_seg: Optional[Dict] = None
        # Kinematic object state tracker — position and heading integrated in Python,
        # bypassing the physics solver (same principle as planar-joint cheat on the robot).
        _obj_kin_pos: Optional[List[float]] = None
        _obj_kin_theta: Optional[float] = None
        # Diagnostics: segment solve snapshot to compare against push-entry state.
        seg_diag_solve_t = None
        seg_diag_solve_obj_theta = None
        seg_diag_solve_r_cp = None
        seg_diag_solve_vcp_ref = None
        seg_diag_solve_phi = None

        if desired_object_velocity is not None and desired_object_angular_velocity is not None:
            # Use user-specified constant twist as a single segment.
            segments = [
                {
                    "name": "manual",
                    "duration": float(duration),
                    "v_ref": np.array(desired_object_velocity, dtype=float),
                    "omega_ref": float(desired_object_angular_velocity),
                }
            ]
        else:
            segments = self._build_wrench_sequence(
                scale_up_factor=scale_up_factor,
                safety_gate=safety_gate,
                normal_force=normal_force,
                segment_duration=segment_duration,
                use_four_segments=use_four_segments,
                normal_only=normal_only,
            )
        cumulative_times = np.cumsum([seg["duration"] for seg in segments])

        for _ in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Cleared each control tick; set again only in wrench branch (below).
                # On the first tick contact is detected we still run approach-only code,
                # so active_seg stays None until the next tick — debug must handle that.
                active_seg = None
                (
                    robot_pos,
                    robot_heading,
                    robot_vel,
                    object_pos,
                    object_orientation,
                    object_velocity,
                    object_angular_velocity,
                ) = self._get_states()

                c = np.cos(object_orientation)
                s = np.sin(object_orientation)
                r_mat = np.array([[c, -s], [s, c]])
                contact_point_world = r_mat @ self.contact_point_body + object_pos
                normal_outward_world = r_mat @ self.normal_outward
                normal_inward_world = -normal_outward_world

                intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
                position_error = intended_pos - robot_pos
                closest_point, _ = closest_point_on_desired_segment(
                    robot_pos,
                    object_pos,
                    object_orientation,
                    self.desired_seg_p1_body,
                    self.desired_seg_p2_body,
                )
                desired_heading = np.arctan2((closest_point - robot_pos)[1], (closest_point - robot_pos)[0])
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading), np.cos(desired_heading - robot_heading))

                r_cp = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                contact_point_velocity = object_velocity + v_rotation
                # Strict report definition: phi is inward-normal direction at contact,
                # not the current robot->contact bearing (which can include tracking error).
                phi = float(np.arctan2(normal_inward_world[1], normal_inward_world[0]))
                current_alpha = float(np.arctan2(np.sin(phi - robot_heading), np.cos(phi - robot_heading)))

                contact_force = self._get_contact_force()

                if not approach_complete:
                    v_r, omega_r, debug_terms = self._compute_approach_diffdrive(
                        robot_pos, robot_heading, contact_point_world, contact_force
                    )
                    if contact_force > self.approach_stop_force_epsilon:
                        approach_complete = True
                        t_contact = t
                        self.segment_ref = None
                        active_segment_idx = -1
                        in_realign_mode = False
                        print(f"[approach] contact at t={t:.2f}s — starting wrench segments")
                else:
                    t_seg = t - t_contact
                    new_segment_idx = int(np.searchsorted(cumulative_times, t_seg, side="right"))
                    if new_segment_idx >= len(segments):
                        new_segment_idx = len(segments) - 1
                    segment_changed = new_segment_idx != active_segment_idx
                    active_segment_idx = new_segment_idx
                    active_seg = segments[active_segment_idx]

                    if self.segment_ref is None or segment_changed:
                        omega_ref = float(active_seg["omega_ref"])
                        v_ref_b = np.asarray(active_seg["v_ref"], dtype=float).reshape(2)
                        r_b = np.asarray(self.contact_point_body, dtype=float).reshape(2)
                        # Constant rigid-body CP velocity in object body (CoM frame, z out of plane).
                        v_cp_body_const = v_ref_b + omega_ref * np.array([-r_b[1], r_b[0]], dtype=float)
                        v_cp_ref = self._compute_world_contact_point_velocity_ref(
                            object_orientation=float(object_orientation),
                            v_ref_body=v_ref_b,
                            omega_ref=omega_ref,
                            r_cp_world=r_cp,
                        )
                        mag_cp_b = float(np.linalg.norm(v_cp_body_const))
                        mag_cp_w = float(np.linalg.norm(v_cp_ref))
                        self._init_segment_reference(
                            phi0=phi,
                            v_cp_ref=v_cp_ref,
                            omega_ref=omega_ref,
                            robot_heading_now=float(robot_heading),
                        )
                        seg_diag_solve_t = float(t)
                        seg_diag_solve_obj_theta = float(object_orientation)
                        seg_diag_solve_r_cp = np.asarray(r_cp, dtype=float).copy()
                        seg_diag_solve_vcp_ref = np.asarray(v_cp_ref, dtype=float).copy()
                        seg_diag_solve_phi = float(phi)
                        print(
                            f"[segment {active_segment_idx}:{active_seg['name']}] "
                            f"duration={active_seg['duration']:.3f}s\n"
                            f"  object twist (body): v_xy=({v_ref_b[0]:+.6f},{v_ref_b[1]:+.6f}) m/s  "
                            f"omega_z={omega_ref:+.6f} rad/s\n"
                            f"  v_cp* constant (body): ({v_cp_body_const[0]:+.6f},{v_cp_body_const[1]:+.6f}) m/s  |v|={mag_cp_b:.6f}\n"
                            f"  v_cp* at seg start (world): ({v_cp_ref[0]:+.6f},{v_cp_ref[1]:+.6f}) m/s  |v|={mag_cp_w:.6f}\n"
                            f"  robot match ref: vr_ff={self.segment_ref['vr_ff']:.4f}, "
                            f"omega_ff={self.segment_ref['omega_ff']:.4f}, "
                            f"zeta0={self.segment_ref['zeta0']:.4f}, "
                            f"alpha*={self.segment_ref['alpha_star']:.4f}"
                        )
                        if control_mode == "stop_go" and segment_changed:
                            in_realign_mode = True
                            pre_push_hold_until_t = None
                            solo_push_active = False

                    intended_alpha = float(self.segment_ref["alpha_star"])

                    if control_mode == "stop_go" and in_realign_mode:
                        zeta_target = float(self.segment_ref["zeta0"])
                        e_zeta = float(
                            np.arctan2(np.sin(zeta_target - robot_heading), np.cos(zeta_target - robot_heading))
                        )
                        v_r = 0.0
                        omega_r = float(np.clip(self.kp_realign_heading * e_zeta, -self.max_omega, self.max_omega))
                        if solo_object_velo:
                            _solo_active_seg = None  # hold object still during realign
                        if abs(e_zeta) <= align_heading_tol_rad:
                            in_realign_mode = False
                            pre_push_hold_until_t = t + float(stop_go_sleep_after_realign_s)
                            print(
                                f"[stop_go] realign complete (|e_zeta|={abs(e_zeta):.4f} rad) — "
                                f"holding {float(stop_go_sleep_after_realign_s):.1f}s before push"
                            )
                        debug_terms = {"v_base": 0.0, "v_ff": 0.0, "v_pi": omega_r}
                    elif (
                        control_mode == "stop_go"
                        and pre_push_hold_until_t is not None
                        and t < pre_push_hold_until_t
                    ):
                        # Post-align, pre-push hold window. Robot stays still.
                        v_r = 0.0
                        omega_r = 0.0
                        if solo_object_velo:
                            if not solo_push_active:
                                solo_push_active = True
                                omega_ref_dbg = float(active_seg["omega_ref"])
                                v_ref_body_dbg = np.asarray(active_seg["v_ref"], dtype=float).reshape(2)
                                v_cp_body_dbg = self._compute_body_contact_point_velocity_ref(
                                    v_ref_body=v_ref_body_dbg,
                                    omega_ref=omega_ref_dbg,
                                )
                                print(
                                    f"[solo_object_velo] pre-push start seg={active_segment_idx}:{active_seg['name']} "
                                    f"for {max(0.0, pre_push_hold_until_t - t):.2f}s"
                                )
                                print(
                                    "  [solo_object_velo refs] "
                                    f"segment twist body: v_ref=({v_ref_body_dbg[0]:+.6f},{v_ref_body_dbg[1]:+.6f}) m/s, "
                                    f"omega_ref={omega_ref_dbg:+.6f} rad/s"
                                )
                                print(
                                    "  [solo_object_velo refs] "
                                    f"v_cp_ref body const=({v_cp_body_dbg[0]:+.6f},{v_cp_body_dbg[1]:+.6f}) m/s, "
                                    f"|v_cp_ref^b|={float(np.linalg.norm(v_cp_body_dbg)):.6f}"
                                )
                            if not solo_push_active:
                                self._set_object_kinematic(zero_friction=True)
                                print("[solo_object_velo] object friction zeroed — kinematic drive active")
                            _solo_active_seg = active_seg  # arm per-step refresh
                        else:
                            _solo_active_seg = None
                        debug_terms = {"v_base": 0.0, "v_ff": 0.0, "v_pi": 0.0}
                    else:
                        if pre_push_hold_until_t is not None and t >= pre_push_hold_until_t:
                            pre_push_hold_until_t = None
                            # Option B: snapshot alpha* from current geometry, not stale solve-time geometry.
                            # phi and robot_heading are current this tick, so this is the true push-entry angle.
                            alpha_star_old = float(self.segment_ref["alpha_star"])
                            alpha_entry = float(np.arctan2(np.sin(phi - robot_heading), np.cos(phi - robot_heading)))
                            self.segment_ref["alpha_star"] = alpha_entry
                            print(
                                f"[stop_go] entering push seg={active_segment_idx}:{active_seg['name']} — "
                                f"alpha* snapped to push-entry: {alpha_entry:.4f} rad "
                                f"(was {alpha_star_old:.4f} from solve, delta={alpha_entry - alpha_star_old:+.4f} rad)"
                            )
                            # Pure diagnostics: compare stale-solve assumptions vs current push-entry state.
                            if seg_diag_solve_vcp_ref is not None:
                                omega_ref_now = float(active_seg["omega_ref"])
                                v_ref_now = np.asarray(active_seg["v_ref"], dtype=float).reshape(2)
                                vcp_ref_push = self._compute_world_contact_point_velocity_ref(
                                    object_orientation=float(object_orientation),
                                    v_ref_body=v_ref_now,
                                    omega_ref=omega_ref_now,
                                    r_cp_world=r_cp,
                                )
                                dtheta = float(
                                    np.arctan2(
                                        np.sin(object_orientation - float(seg_diag_solve_obj_theta)),
                                        np.cos(object_orientation - float(seg_diag_solve_obj_theta)),
                                    )
                                )
                                dvcp = np.asarray(vcp_ref_push, dtype=float) - np.asarray(seg_diag_solve_vcp_ref, dtype=float)
                                phi_push = float(phi)
                                dphi = float(
                                    np.arctan2(
                                        np.sin(phi_push - float(seg_diag_solve_phi)),
                                        np.cos(phi_push - float(seg_diag_solve_phi)),
                                    )
                                )
                                print(
                                    f"[diag push-entry seg={active_segment_idx}:{active_seg['name']}] "
                                    f"solve_t={seg_diag_solve_t:.3f}s -> push_t={t:.3f}s, "
                                    f"dt={t-float(seg_diag_solve_t):.3f}s"
                                )
                                print(
                                    f"  obj_theta solve={float(seg_diag_solve_obj_theta):+.6f}, "
                                    f"push={float(object_orientation):+.6f}, dtheta={dtheta:+.6f} rad"
                                )
                                print(
                                    f"  r_cp solve=({float(seg_diag_solve_r_cp[0]):+.6f},{float(seg_diag_solve_r_cp[1]):+.6f}), "
                                    f"push=({float(r_cp[0]):+.6f},{float(r_cp[1]):+.6f})"
                                )
                                print(
                                    f"  v_cp_ref solve=({float(seg_diag_solve_vcp_ref[0]):+.6f},{float(seg_diag_solve_vcp_ref[1]):+.6f}) "
                                    f"|v|={float(np.linalg.norm(seg_diag_solve_vcp_ref)):.6f}"
                                )
                                print(
                                    f"  v_cp_ref push =({float(vcp_ref_push[0]):+.6f},{float(vcp_ref_push[1]):+.6f}) "
                                    f"|v|={float(np.linalg.norm(vcp_ref_push)):.6f}"
                                )
                                print(
                                    f"  dv_cp_ref=({float(dvcp[0]):+.6f},{float(dvcp[1]):+.6f}) "
                                    f"|dv|={float(np.linalg.norm(dvcp)):.6f}"
                                )
                                print(
                                    f"  phi solve={float(seg_diag_solve_phi):+.6f}, "
                                    f"push={phi_push:+.6f}, dphi={dphi:+.6f} rad"
                                )
                        if solo_object_velo and control_mode == "stop_go":
                            if not solo_push_active:
                                solo_push_active = True
                                omega_ref_dbg = float(active_seg["omega_ref"])
                                v_ref_body_dbg = np.asarray(active_seg["v_ref"], dtype=float).reshape(2)
                                v_cp_body_dbg = self._compute_body_contact_point_velocity_ref(
                                    v_ref_body=v_ref_body_dbg,
                                    omega_ref=omega_ref_dbg,
                                )
                                print(
                                    f"[solo_object_velo] activating seg={active_segment_idx}:{active_seg['name']} "
                                    "for push"
                                )
                                print(
                                    "  [solo_object_velo refs] "
                                    f"segment twist body: v_ref=({v_ref_body_dbg[0]:+.6f},{v_ref_body_dbg[1]:+.6f}) m/s, "
                                    f"omega_ref={omega_ref_dbg:+.6f} rad/s"
                                )
                                print(
                                    "  [solo_object_velo refs] "
                                    f"v_cp_ref body const=({v_cp_body_dbg[0]:+.6f},{v_cp_body_dbg[1]:+.6f}) m/s, "
                                    f"|v_cp_ref^b|={float(np.linalg.norm(v_cp_body_dbg)):.6f}"
                                )
                            if not solo_push_active:
                                self._set_object_kinematic(zero_friction=True)
                                print("[solo_object_velo] object friction zeroed — kinematic drive active (push phase)")
                            _solo_active_seg = active_seg  # arm per-step refresh
                        v_r, omega_r, debug_terms = self.compute_diffdrive_phase7_command(
                            robot_heading=robot_heading,
                            position_error=position_error,
                            heading_error=heading_error,
                            current_alpha=current_alpha,
                            intended_alpha=intended_alpha,
                            desired_contact_point_speed=desired_contact_point_speed,
                            actual_contact_point_velocity=contact_point_velocity,
                            actual_object_speed=float(np.linalg.norm(object_velocity)),
                            desired_object_velocity=desired_object_velocity,
                            desired_object_angular_velocity=desired_object_angular_velocity,
                        )

                self.robot.command_velocity(np.array([v_r, omega_r]))
                in_contact = contact_force > self.contact_threshold

                if debug_print_velocity:
                    k_ctrl = step_count // CTRL_STEP
                    every = max(1, int(debug_print_velocity_every))
                    if k_ctrl % every == 0:
                        vx_cmd = float(v_r * np.cos(robot_heading))
                        vy_cmd = float(v_r * np.sin(robot_heading))
                        vcp_act = np.asarray(contact_point_velocity, dtype=float).reshape(2)
                        mag_act = float(np.linalg.norm(vcp_act))
                        if not approach_complete:
                            print(
                                f"[vel dbg t={t:6.3f}s phase=approach] "
                                f"v_cp actual=({vcp_act[0]:+.4f},{vcp_act[1]:+.4f}) |v|={mag_act:.4f} "
                                f"cmd body: v_r={v_r:+.4f} omega_r={omega_r:+.4f}  "
                                f"cmd world: vx={vx_cmd:+.4f} vy={vy_cmd:+.4f}  "
                                f"decomp: v_ff={debug_terms['v_ff']:+.4f} v_pos_fb={debug_terms['v_base']:+.4f} "
                                f"v_speed_p={debug_terms.get('v_speed_p', 0.0):+.4f} "
                                f"|Fxy|={contact_force:.4f}"
                            )
                        elif approach_complete and active_seg is None:
                            # Same control tick as first contact: approach branch ran, wrench branch did not.
                            print(
                                f"[vel dbg t={t:6.3f}s phase=contact_acquired] "
                                f"(segment refs apply on next control step) "
                                f"v_cp actual=({vcp_act[0]:+.4f},{vcp_act[1]:+.4f}) |v|={mag_act:.4f} "
                                f"cmd body: v_r={v_r:+.4f} omega_r={omega_r:+.4f}  "
                                f"|Fxy|={contact_force:.4f}"
                            )
                        elif control_mode == "stop_go" and in_realign_mode:
                            omr = float(active_seg["omega_ref"])
                            v_cp_star_r = np.asarray(active_seg["v_ref"], dtype=float).reshape(2) + omr * np.array(
                                [-r_cp[1], r_cp[0]], dtype=float
                            )
                            mag_sr = float(np.linalg.norm(v_cp_star_r))
                            print(
                                f"[vel dbg t={t:6.3f}s phase=realign seg={active_segment_idx}:{active_seg['name']}] "
                                f"v_cp* world=({v_cp_star_r[0]:+.4f},{v_cp_star_r[1]:+.4f}) |v|={mag_sr:.4f}  "
                                f"v_cp actual=({vcp_act[0]:+.4f},{vcp_act[1]:+.4f}) |v|={mag_act:.4f}  "
                                f"cmd: v_r={v_r:+.4f} omega_r={omega_r:+.4f} (realign only)  "
                                f"|Fxy|={contact_force:.4f}"
                            )
                        else:
                            omega_ref_dbg = float(active_seg["omega_ref"])
                            v_cp_star = self._compute_world_contact_point_velocity_ref(
                                object_orientation=float(object_orientation),
                                v_ref_body=np.asarray(active_seg["v_ref"], dtype=float).reshape(2),
                                omega_ref=omega_ref_dbg,
                                r_cp_world=r_cp,
                            )
                            mag_star = float(np.linalg.norm(v_cp_star))
                            om_ff = float(debug_terms.get("omega_ff", 0.0))
                            vraw = float(debug_terms.get("v_r_before_clip", v_r))
                            v_speed_p_dbg = float(debug_terms.get("v_speed_p", 0.0))
                            speed_err_dbg = float(debug_terms.get("speed_error", 0.0))

                            # Curvature of the two trajectories we care about:
                            #   (1) Robot's intended contact point on its rim at angle phi:
                            #       p = robot_pos + R_r * [cos(phi), sin(phi)]
                            #       v = v_r * e_zeta + omega_r * R_r * e_phi_perp  (rigid-body velocity of that rim point)
                            #       kappa = omega_r / |v|
                            e_zeta_vec = np.array([np.cos(robot_heading), np.sin(robot_heading)])
                            e_phi_perp_vec = np.array([-np.sin(phi), np.cos(phi)])
                            v_cp_robot = v_r * e_zeta_vec + omega_r * ROBOT_RADIUS * e_phi_perp_vec
                            v_cp_robot_mag = float(np.linalg.norm(v_cp_robot))
                            kappa_robot_cp = (float(omega_r) / v_cp_robot_mag) if v_cp_robot_mag > 1e-6 else float("inf")
                            R_robot_cp = (1.0 / abs(kappa_robot_cp)) if abs(kappa_robot_cp) > 1e-6 else float("inf")
                            #   (2) Object contact point:
                            #       measured: v = contact_point_velocity from sim (friction-degraded)
                            #       ideal: |v_cp^b| from segment body-frame definition (what we intend to set)
                            v_cp_obj_mag = float(np.linalg.norm(vcp_act))
                            kappa_obj_cp = (float(omega_ref_dbg) / v_cp_obj_mag) if v_cp_obj_mag > 1e-6 else float("inf")
                            R_obj_cp = (1.0 / abs(kappa_obj_cp)) if abs(kappa_obj_cp) > 1e-6 else float("inf")
                            # Ideal: use the constant body-frame CP speed (unaffected by sim friction).
                            v_ref_b_dbg = np.asarray(active_seg["v_ref"], dtype=float).reshape(2)
                            v_cp_b_ideal = self._compute_body_contact_point_velocity_ref(v_ref_b_dbg, omega_ref_dbg)
                            v_cp_ideal_mag = float(np.linalg.norm(v_cp_b_ideal))
                            kappa_obj_ideal = (float(omega_ref_dbg) / v_cp_ideal_mag) if v_cp_ideal_mag > 1e-6 else float("inf")
                            R_obj_ideal = (1.0 / abs(kappa_obj_ideal)) if abs(kappa_obj_ideal) > 1e-6 else float("inf")

                            print(
                                f"[vel dbg t={t:6.3f}s seg={active_segment_idx}:{active_seg['name']}] "
                                f"v_cp* world=({v_cp_star[0]:+.4f},{v_cp_star[1]:+.4f}) |v|={mag_star:.4f}  "
                                f"v_cp actual=({vcp_act[0]:+.4f},{vcp_act[1]:+.4f}) |v|={mag_act:.4f}  "
                                f"cmd body: v_r={v_r:+.4f} (raw {vraw:+.4f}) omega_r={omega_r:+.4f}  "
                                f"cmd world: vx={vx_cmd:+.4f} vy={vy_cmd:+.4f}  "
                                f"decomp: vr_ff={debug_terms['v_ff']:+.4f} + v_pos_fb={debug_terms['v_base']:+.4f} + v_speed_p={v_speed_p_dbg:+.4f} (spd_err={speed_err_dbg:+.4f}) -> v_r; "
                                f"omega_ff={om_ff:+.4f} + alpha_fb={-self.kp_alpha * debug_terms.get('e_alpha', 0.0):+.4f} (e_alpha={debug_terms.get('e_alpha', 0.0):+.4f} rad) -> omega_r  "
                                f"|Fxy|={contact_force:.4f} "
                                f"v_des={desired_contact_point_speed:.4f} v_obj_act={float(np.linalg.norm(object_velocity)):.4f}  "
                                f"|| CURVATURE: "
                                f"robot_rim_cp kappa={kappa_robot_cp:+.4f} R={R_robot_cp:.3f}m |v|={v_cp_robot_mag:.4f}  "
                                f"obj_cp_measured kappa={kappa_obj_cp:+.4f} R={R_obj_cp:.3f}m |v|={v_cp_obj_mag:.4f}  "
                                f"obj_cp_IDEAL kappa={kappa_obj_ideal:+.4f} R={R_obj_ideal:.3f}m |v|={v_cp_ideal_mag:.4f}"
                            )

                cmd_world_like = np.array([v_r * np.cos(robot_heading), v_r * np.sin(robot_heading), omega_r])
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(float(robot_heading))
                self.history.robot_velocities.append(np.asarray(robot_vel).copy())
                self.history.robot_cmd_velocities.append(cmd_world_like.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(float(desired_heading))
                self.history.heading_errors.append(float(heading_error))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(float(object_orientation))
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(float(object_angular_velocity))
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(float(contact_force))
                self.history.in_contact.append(bool(in_contact))
                self.history.v_base_history.append(float(debug_terms["v_base"]))
                self.history.v_constant_history.append(float(debug_terms["v_ff"]))
                # v_speed_p for the main push command; v_pi for approach/realign phases
                self.history.v_velo_error_pi_history.append(
                    float(debug_terms.get("v_speed_p", debug_terms.get("v_pi", 0.0)))
                )

            # Kinematic object cheat drive every sim step (same principle as robot planar-joint cheat).
            # Integrates pose in Python and hard-resets position+velocity before stepSimulation,
            # so the physics solver never gets a chance to degrade the twist.
            if solo_object_velo:
                if _solo_active_seg is not None:
                    # Lazy-init: seed tracker from current PyBullet state the first step it arms.
                    if _obj_kin_pos is None:
                        _p, _o = pyb.getBasePositionAndOrientation(self.object_uid)
                        _obj_kin_pos = list(_p)
                        _obj_kin_theta = float(pyb.getEulerFromQuaternion(_o)[2])
                    _obj_kin_pos, _obj_kin_theta = self.cheat_drive_object_step(
                        _obj_kin_pos, _obj_kin_theta, _solo_active_seg
                    )
                else:
                    # Realign phase: hold object still and reset tracker.
                    self._stop_object_motion()
                    _obj_kin_pos = None
                    _obj_kin_theta = None
            pyb.stepSimulation()
            if gui:
                time.sleep(TIMESTEP * 0.3)
            t += TIMESTEP
            step_count += 1

        mean_pos_err = 0.0
        if self.history.position_errors:
            err = np.array([np.linalg.norm(e) for e in self.history.position_errors], dtype=float)
            mean_pos_err = float(np.mean(err))
        return {"mean_position_error_m": mean_pos_err, "samples": len(self.history.times)}

    def run_phase_7(
        self,
        desired_contact_point_speed: float,
        gui: bool = True,
        duration: float = 10.0,
        control_mode: str = "continuous",
        align_heading_tol_rad: float = np.deg2rad(1.0),
        scale_up_factor: float = 1.0,
        safety_gate: float = 0.7,
        normal_force: float = 1.0,
        segment_duration: float = 3.0,
        use_four_segments: bool = True,
        skip_approach: bool = False,
        normal_only: bool = False,
        debug_print_velocity: bool = False,
        debug_print_velocity_every: int = 5,
        solo_object_velo: bool = False,
        stop_go_sleep_after_realign_s: float = DEFAULT_STOP_GO_SLEEP_AFTER_REALIGN_S,
    ) -> Dict:
        return self._run_loop(
            duration,
            gui,
            desired_contact_point_speed=desired_contact_point_speed,
            control_mode=control_mode,
            align_heading_tol_rad=align_heading_tol_rad,
            scale_up_factor=scale_up_factor,
            safety_gate=safety_gate,
            normal_force=normal_force,
            segment_duration=segment_duration,
            use_four_segments=use_four_segments,
            skip_approach=skip_approach,
            normal_only=normal_only,
            debug_print_velocity=debug_print_velocity,
            debug_print_velocity_every=debug_print_velocity_every,
            solo_object_velo=solo_object_velo,
            stop_go_sleep_after_realign_s=stop_go_sleep_after_realign_s,
        )

    def plot_pose_results(self, save_path: Optional[Path] = None):
        if not self.history.times:
            return
        t = np.array(self.history.times)
        pos = np.array(self.history.robot_positions)
        intended = np.array(self.history.intended_positions)
        err = np.array([np.linalg.norm(e) for e in self.history.position_errors]) * 100.0

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(pos[:, 0], pos[:, 1], "b-", label="robot")
        axes[0].plot(intended[:, 0], intended[:, 1], "g--", label="intended")
        axes[0].set_title("Trajectory")
        axes[0].axis("equal")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t, err, "b-")
        axes[1].set_title("Position Error (cm)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, np.degrees(np.array(self.history.heading_errors)), "r-")
        axes[2].set_title("Heading Error (deg)")
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_phase7_velocities(self, save_path: Optional[Path] = None):
        if not self.history.times:
            return
        t = np.array(self.history.times)
        v_base = np.array(self.history.v_base_history)
        v_ff = np.array(self.history.v_constant_history)
        v_pi = np.array(self.history.v_velo_error_pi_history)
        f_contact = np.array(self.history.contact_forces)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(t, v_base, label="v_base (pos feedback)")
        axes[0].plot(t, v_ff, label="v_ff (frictionless FF)")
        axes[0].plot(t, v_pi, label="v_speed_p (speed P)")
        axes[0].set_title("Phase7 v_r Components")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t, f_contact, "r-", label="contact force")
        axes[1].set_title("Contact Force")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")
        else:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Single-pusher diff-drive Phase7 template")
    parser.add_argument("--t-param", type=float, default=0.125)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--desired-speed", type=float, default=0.10, help="Desired contact-point speed")
    parser.add_argument("--approach-distance", type=float, default=0.2)
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_SHAPE)
    parser.add_argument("--control-mode", type=str, default="continuous", choices=["continuous", "stop_go"])
    parser.add_argument("--align-heading-tol-deg", type=float, default=5.0)
    parser.add_argument("--scale-up-factor", type=float, default=1.0)
    parser.add_argument("--safety-gate", type=float, default=0.7)
    parser.add_argument("--normal-force", type=float, default=1.0)
    parser.add_argument("--segment-duration", type=float, default=3.0)
    parser.add_argument("--three-segments", action="store_true", help="Use 3 segments instead of 4-segment smoother schedule")
    parser.add_argument(
        "--omni-native-rectangle",
        action="store_true",
        help="Peer-to-peer mode: use create_standard_objects()['rectangle'] like test_single_pusher_omni.py (big native rectangle). Ignores --object-name.",
    )
    parser.add_argument(
        "--normal-only",
        action="store_true",
        help="Use a single wrench segment (normal force at contact only, no friction-cone left/right). Reduces reference jumps that can cause slip.",
    )
    parser.add_argument("--skip-approach", action="store_true", help="Skip rotate-then-creep approach (start pushing immediately)")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record PyBullet simulation as MP4 (top-down view). Requires --save-dir.",
    )
    parser.add_argument(
        "--debug-vel",
        action="store_true",
        help="Print intended vs actual contact-point velocity and robot command breakdown (throttled)",
    )
    parser.add_argument(
        "--debug-vel-every",
        type=int,
        default=50,
        metavar="K",
        help="Print every K control updates (100 Hz ctrl -> K=5 => ~20 lines/s). Default: 5",
    )
    parser.add_argument(
        "--solo-object-velo",
        action="store_true",
        help="Debug mode for stop_go: after realign, drive the object directly with the active segment twist instead of relying on contact push.",
    )
    parser.add_argument(
        "--object-mass",
        type=float,
        default=None,
        help="Override object mass (kg). Default: use value from object definition.",
    )
    parser.add_argument(
        "--stop-go-pre-push-hold-s",
        type=float,
        default=DEFAULT_STOP_GO_SLEEP_AFTER_REALIGN_S,
        help="Stop-go only: hold duration after heading realign and before push starts (seconds).",
    )
    parser.add_argument(
        "--ground-friction",
        type=float,
        default=DEFAULT_GROUND_FRICTION,
        help="Plane lateral friction (default matches test_diffdrive_wheel_physics.setup_sim)",
    )
    parser.add_argument(
        "--wheel-friction",
        type=float,
        default=DEFAULT_WHEEL_LATERAL_FRICTION,
        help="Wheel link lateral friction (default matches test_diffdrive_wheel_physics)",
    )
    parser.add_argument(
        "--caster-friction",
        type=float,
        default=DEFAULT_CASTER_LATERAL_FRICTION,
        help="Caster link lateral friction (default matches test_diffdrive_wheel_physics)",
    )
    args = parser.parse_args()

    setup_pybullet(gui=not args.no_gui, ground_friction=float(args.ground_friction))
    test = SinglePusherDiffdriveTemplate(
        t_param=args.t_param,
        approach_distance=args.approach_distance,
        object_name=args.object_name,
        wheel_lateral_friction=float(args.wheel_friction),
        caster_lateral_friction=float(args.caster_friction),
        omni_native_rectangle=bool(args.omni_native_rectangle),
    )
    if args.object_mass is not None:
        new_mass = float(args.object_mass)
        test.generic_object.mass = new_mass
        test.generic_object.moment_of_inertia = test.generic_object._calculate_moment_of_inertia()
        pyb.changeDynamics(test.object_uid, -1, mass=new_mass)
        print(f"[object] mass overridden to {new_mass:.4f} kg, new inertia={test.generic_object.moment_of_inertia:.6f}")

    print(
        f"[sim] ground_mu={args.ground_friction:.3f}, wheel_mu={args.wheel_friction:.3f}, "
        f"caster_mu={args.caster_friction:.3f}"
    )
    print(
        f"[object] name={('omni_native_rectangle' if args.omni_native_rectangle else args.object_name)}, "
        f"mass={test.generic_object.mass:.4f}, "
        f"inertia={test.generic_object.moment_of_inertia:.6f}, "
        f"lateral_friction={test.contact_friction_coeff:.3f}"
    )
    if args.normal_only:
        print("[schedule] normal_only: single segment (no cone-edge transitions)")
    if args.solo_object_velo:
        print("[mode] solo_object_velo enabled: object follows active segment twist during stop_go push")
    if args.control_mode == "stop_go":
        print(f"[mode] stop_go pre-push hold={float(args.stop_go_pre_push_hold_s):.1f}s")

    # Video recording (requires GUI and --save-dir)
    video_log_id: Optional[int] = None
    video_path: Optional[Path] = None
    if args.record_video:
        if args.no_gui:
            print("⚠ --record-video ignored: PyBullet video recording requires GUI mode (remove --no-gui).")
        elif not args.save_dir:
            raise ValueError("--record-video requires --save-dir to be specified.")
        else:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            video_path = save_dir / "phase7_topview.mp4"
            video_log_id = setup_video_recording(video_path, test.object_uid)

    results = test.run_phase_7(
        desired_contact_point_speed=args.desired_speed,
        gui=not args.no_gui,
        duration=args.duration,
        control_mode=args.control_mode,
        align_heading_tol_rad=np.deg2rad(args.align_heading_tol_deg),
        scale_up_factor=args.scale_up_factor,
        safety_gate=args.safety_gate,
        normal_force=args.normal_force,
        segment_duration=args.segment_duration,
        use_four_segments=(not args.three_segments),
        skip_approach=args.skip_approach,
        normal_only=args.normal_only,
        debug_print_velocity=args.debug_vel,
        debug_print_velocity_every=args.debug_vel_every,
        solo_object_velo=args.solo_object_velo,
        stop_go_sleep_after_realign_s=float(args.stop_go_pre_push_hold_s),
    )

    print(f"Run results: {results}")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        test.plot_pose_results(save_dir / "phase7_pose_results.png")
        test.plot_phase7_velocities(save_dir / "phase7_velocities.png")
    else:
        test.plot_pose_results()
        test.plot_phase7_velocities()

    if video_log_id is not None:
        print("Finalizing video recording...")
        stop_video_recording(video_log_id, video_path)

    pyb.disconnect()

    # Verify after disconnect (PyBullet flushes the file on disconnect)
    if video_log_id is not None and video_path is not None:
        time.sleep(1.0)
        video_path = video_path.resolve()
        if video_path.exists():
            size_mb = video_path.stat().st_size / 1024 / 1024
            if size_mb > 0:
                print(f"✓ Video confirmed: {video_path} ({size_mb:.2f} MB)")
            else:
                print(f"✗ Warning: video file is empty at {video_path}")
        else:
            print(f"✗ Warning: video file not found at {video_path}")


if __name__ == "__main__":
    main()
