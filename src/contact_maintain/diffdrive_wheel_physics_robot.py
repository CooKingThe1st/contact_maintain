"""Physics-driven differential drive robot for PyBullet simulation.

This variant intentionally avoids planar-joint velocity actuation. The base is
driven only by left/right wheel motors and ground contact/friction interaction.
"""

from pathlib import Path

import numpy as np
import pybullet as pyb
import rospkg

from contact_maintain.diffdrive_wheel_robot import (
    MAX_WHEEL_SPEED,
    WHEEL_BASE,
    WHEEL_RADIUS,
    compute_body_velocity_diffdrive,
    compute_wheel_velocities_diffdrive,
)


class DiffDriveWheelPhysicsRobot:
    """Differential-drive robot driven only by wheel-ground physics."""

    # Planar joint indices in URDF order
    X_JOINT_IDX = 0
    Y_JOINT_IDX = 1
    THETA_JOINT_IDX = 2

    def __init__(
        self,
        urdf_path=None,
        position=(0, 0),
        orientation=0.0,
        contact_mu=0.8,
        wheel_lateral_friction=1.0,
        caster_lateral_friction=0.02,
        wheel_motor_force=40.0,
        use_planar_cheat_control=False,
    ):
        if urdf_path is None:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("contact_maintain")
            urdf_path = str(Path(pkg_path) / "urdf" / "diffdrive_wheel_robot_disc_bumper.urdf")

        # Same URDF topology as DiffDriveWheelRobot. We keep planar joints but
        # do not command them; wheels are the only actuators.
        self.uid = pyb.loadURDF(
            urdf_path, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True
        )

        self.pos_init = np.array(position, dtype=float)
        self.orn_init = float(orientation)
        self.wheel_motor_force = float(wheel_motor_force)
        self.wheel_lateral_friction = float(wheel_lateral_friction)
        self.caster_lateral_friction = float(caster_lateral_friction)
        self.wheel_base = float(WHEEL_BASE)
        self.wheel_radius = float(WHEEL_RADIUS)
        self.use_planar_cheat_control = bool(use_planar_cheat_control)

        self._build_joint_info()
        self.set_contact_friction(contact_mu)
        self.set_wheel_friction(self.wheel_lateral_friction)
        self.set_caster_friction(self.caster_lateral_friction)
        self._disable_planar_joint_motors()
        self.reset()

        self.last_cmd_vel = np.array([0.0, 0.0])
        self.last_wheel_speeds = np.array([0.0, 0.0])

    def _build_joint_info(self):
        self.planar_joint_indices = [self.X_JOINT_IDX, self.Y_JOINT_IDX, self.THETA_JOINT_IDX]
        self.wheel_left_idx = None
        self.wheel_right_idx = None
        self.bumper_link_idx = None
        self.caster_link_idx = None

        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")

            if joint_name == "wheel_left_joint":
                self.wheel_left_idx = i
            elif joint_name == "wheel_right_joint":
                self.wheel_right_idx = i

            if link_name == "bumper":
                self.bumper_link_idx = i
            elif link_name == "caster_link":
                self.caster_link_idx = i

        if self.wheel_left_idx is None or self.wheel_right_idx is None:
            raise RuntimeError("Could not find wheel joints in URDF")

    def _disable_planar_joint_motors(self):
        for idx in self.planar_joint_indices:
            pyb.setJointMotorControl2(
                self.uid,
                idx,
                controlMode=pyb.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0,
            )

    def set_contact_friction(self, mu):
        if self.bumper_link_idx is not None:
            pyb.changeDynamics(self.uid, self.bumper_link_idx, lateralFriction=float(mu))

    def set_wheel_friction(self, mu):
        if self.wheel_left_idx is not None:
            pyb.changeDynamics(self.uid, self.wheel_left_idx, lateralFriction=float(mu))
        if self.wheel_right_idx is not None:
            pyb.changeDynamics(self.uid, self.wheel_right_idx, lateralFriction=float(mu))

    def set_caster_friction(self, mu):
        if self.caster_link_idx is not None:
            pyb.changeDynamics(self.uid, self.caster_link_idx, lateralFriction=float(mu))

    def get_state(self):
        states = pyb.getJointStates(self.uid, self.planar_joint_indices)
        position = np.array([states[0][0], states[1][0]])
        heading = float(states[2][0])
        velocity = np.array([states[0][1], states[1][1], states[2][1]])
        return position, heading, velocity

    def get_pose(self):
        position, heading, _ = self.get_state()
        return position, heading

    def get_wheel_velocities(self):
        left_state = pyb.getJointState(self.uid, self.wheel_left_idx)
        right_state = pyb.getJointState(self.uid, self.wheel_right_idx)
        return np.array([left_state[1], right_state[1]])

    def get_contact_position(self):
        if self.bumper_link_idx is not None:
            state = pyb.getLinkState(self.uid, self.bumper_link_idx)
            return np.array(state[0])
        pos, heading = self.get_pose()
        return np.array([pos[0] + 0.055 * np.cos(heading), pos[1] + 0.055 * np.sin(heading), 0.025])

    def get_contact_force(self, object_uids, max_contacts=1):
        from contact_maintain.pyb_simulation import get_contact_force

        total_force = np.zeros(3)
        for uid in object_uids:
            force = get_contact_force(
                self.uid,
                uid,
                linkIndexA=self.bumper_link_idx if self.bumper_link_idx is not None else -1,
                max_contacts=max_contacts,
            )
            total_force += force
        return total_force

    def command_velocity(self, velocity):
        if len(velocity) == 2:
            v, omega = float(velocity[0]), float(velocity[1])
        else:
            v, omega = float(velocity[0]), float(velocity[2])

        # Enforce body-command limits implied by max wheel speed.
        max_linear = self.wheel_radius * MAX_WHEEL_SPEED
        max_angular = 2.0 * self.wheel_radius * MAX_WHEEL_SPEED / self.wheel_base
        v = float(np.clip(v, -max_linear, max_linear))
        omega = float(np.clip(omega, -max_angular, max_angular))

        if self.use_planar_cheat_control:
            self._command_velocity_planar_cheat(v, omega)
        else:
            omega_left, omega_right = compute_wheel_velocities_diffdrive(v, omega)
            self.command_wheel_velocities(omega_left, omega_right)
            self.last_cmd_vel = np.array([v, omega])

    def _command_velocity_planar_cheat(self, v, omega):
        """Cheat mode: use planar joints as primary actuation (debug baseline)."""
        omega_left, omega_right = compute_wheel_velocities_diffdrive(v, omega)
        omega_left = float(np.clip(omega_left, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        omega_right = float(np.clip(omega_right, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        self.last_wheel_speeds = np.array([omega_left, omega_right])
        self.last_cmd_vel = np.array([v, omega])

        # Wheels spin for visualization/debug parity.
        pyb.setJointMotorControl2(
            self.uid,
            self.wheel_left_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega_left,
            force=min(self.wheel_motor_force, 10.0),
        )
        pyb.setJointMotorControl2(
            self.uid,
            self.wheel_right_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega_right,
            force=min(self.wheel_motor_force, 10.0),
        )

        # Planar-joint world-frame velocity actuation (same style as cheat robot).
        _pos, heading, _vel = self.get_state()
        vx = float(v * np.cos(heading))
        vy = float(v * np.sin(heading))
        pyb.setJointMotorControl2(
            self.uid,
            self.X_JOINT_IDX,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=vx,
            force=500.0,
        )
        pyb.setJointMotorControl2(
            self.uid,
            self.Y_JOINT_IDX,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=vy,
            force=500.0,
        )
        pyb.setJointMotorControl2(
            self.uid,
            self.THETA_JOINT_IDX,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=float(omega),
            force=200.0,
        )

    def command_wheel_velocities(self, omega_left, omega_right):
        omega_left = float(np.clip(omega_left, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        omega_right = float(np.clip(omega_right, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        self.last_wheel_speeds = np.array([omega_left, omega_right])

        pyb.setJointMotorControl2(
            self.uid,
            self.wheel_left_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega_left,
            force=self.wheel_motor_force,
        )
        pyb.setJointMotorControl2(
            self.uid,
            self.wheel_right_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega_right,
            force=self.wheel_motor_force,
        )

        v, omega = compute_body_velocity_diffdrive(omega_left, omega_right)
        self.last_cmd_vel = np.array([v, omega])

    def reset(self, position=None, orientation=None):
        if position is not None:
            self.pos_init = np.array(position, dtype=float)
        if orientation is not None:
            self.orn_init = float(orientation)

        pyb.resetJointState(self.uid, self.X_JOINT_IDX, self.pos_init[0], 0.0)
        pyb.resetJointState(self.uid, self.Y_JOINT_IDX, self.pos_init[1], 0.0)
        pyb.resetJointState(self.uid, self.THETA_JOINT_IDX, self.orn_init, 0.0)

        pyb.resetJointState(self.uid, self.wheel_left_idx, 0.0, 0.0)
        pyb.resetJointState(self.uid, self.wheel_right_idx, 0.0, 0.0)
        self._disable_planar_joint_motors()

        pyb.setJointMotorControl2(
            self.uid,
            self.wheel_left_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
        )
        pyb.setJointMotorControl2(
            self.uid,
            self.wheel_right_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
        )

        self.last_cmd_vel = np.array([0.0, 0.0])
        self.last_wheel_speeds = np.array([0.0, 0.0])

    def get_debug_info(self):
        pos, heading, vel = self.get_state()
        wheel_vels = self.get_wheel_velocities()
        return {
            "position": pos,
            "heading": heading,
            "heading_deg": np.degrees(heading),
            "velocity": vel,
            "wheel_velocities": wheel_vels,
            "last_cmd_vel": self.last_cmd_vel,
            "last_wheel_speeds": self.last_wheel_speeds,
            "wheel_motor_force": self.wheel_motor_force,
            "wheel_lateral_friction": self.wheel_lateral_friction,
            "caster_lateral_friction": self.caster_lateral_friction,
            "use_planar_cheat_control": self.use_planar_cheat_control,
        }
