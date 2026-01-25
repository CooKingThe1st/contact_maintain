#!/usr/bin/env python3
"""
Two-Robot Phase-2 Reliability Test (Hybrid Contact Control)

Goal
----
Evaluate whether the Phase-2-style position/heading controller remains stable with:
1) Two robots on the same edge (different t_params, same boundary segment)
2) Two robots on opposite edges

This is NOT a full multi-robot swarm/avoidance test. It's a focused "contact maintenance"
sanity check with 2 robots applying similar control against the same object.

Outputs
-------
- Time series of each robot's measured contact force
- Object motion (vx, vy, omega)
- Robot commanded vs measured velocities (vx, vy, omega) for holonomic

Usage
-----
python test_two_robot_hybrid_phase2.py --case same_edge --t1 0.10 --t2 0.20 --save-dir /tmp/results
python test_two_robot_hybrid_phase2.py --case opposite --t1 0.10 --t2 0.60 --no-gui
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Use non-interactive backend for headless mode
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

from object_utils import create_standard_objects, ContactPointParameterization
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import generic_to_pybullet
from contact_maintain.pyb_simulation import get_contact_force


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

DEFAULT_OBJECT_SHAPE = "rectangle"
DEFAULT_OBJECT_HEIGHT = 0.2
DEFAULT_OBJECT_FRICTION = 0.8

ROBOT_RADIUS = 0.06


def setup_pybullet(gui: bool = True):
    if gui:
        pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        pyb.connect(pyb.DIRECT)

    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)

    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0.0, 0.0, 0.0],
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)


def _robot_contact_link_index(robot) -> int:
    if hasattr(robot, "bumper_link_idx") and robot.bumper_link_idx is not None:
        return int(robot.bumper_link_idx)
    if hasattr(robot, "contact_link_idx") and robot.contact_link_idx is not None:
        return int(robot.contact_link_idx)
    return -1


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


@dataclass
class TwoRobotHistory:
    times: List[float] = field(default_factory=list)
    object_velocities: List[np.ndarray] = field(default_factory=list)
    object_angular_velocities: List[float] = field(default_factory=list)

    r1_pos: List[np.ndarray] = field(default_factory=list)
    r1_vel: List[np.ndarray] = field(default_factory=list)
    r1_cmd: List[np.ndarray] = field(default_factory=list)
    r1_force: List[float] = field(default_factory=list)

    r2_pos: List[np.ndarray] = field(default_factory=list)
    r2_vel: List[np.ndarray] = field(default_factory=list)
    r2_cmd: List[np.ndarray] = field(default_factory=list)
    r2_force: List[float] = field(default_factory=list)


class TwoRobotPhase2Test:
    def __init__(
        self,
        *,
        kinematics: str,
        t1: float,
        t2: float,
        approach_distance: float,
        kp_pos: float,
        kp_heading: float,
        max_speed: float,
        contact_threshold: float,
    ):
        self.kinematics = kinematics
        self.t1 = float(t1)
        self.t2 = float(t2)

        self.approach_distance = float(approach_distance)
        self.kp_pos = float(kp_pos)
        self.kp_heading = float(kp_heading)
        self.max_speed = float(max_speed)
        self.contact_threshold = float(contact_threshold)

        # Object + parameterization
        standard_objects = create_standard_objects()
        self.generic_object = standard_objects[DEFAULT_OBJECT_SHAPE]
        self.param = ContactPointParameterization(self.generic_object)

        self.object_uid = generic_to_pybullet(
            self.generic_object,
            height=DEFAULT_OBJECT_HEIGHT,
            position=(0.0, 0.0, 0.0),
            orientation=0.0,
            color=(0.4, 0.7, 0.4, 1.0),
        )
        pyb.changeDynamics(self.object_uid, -1, lateralFriction=DEFAULT_OBJECT_FRICTION, mass=1.0)

        # Spawn robots at intended position for each t_param
        self.robot1 = self._spawn_robot("R1", self.t1)
        self.robot2 = self._spawn_robot("R2", self.t2)

        self.r1_link = _robot_contact_link_index(self.robot1)
        self.r2_link = _robot_contact_link_index(self.robot2)

        self.history = TwoRobotHistory()

    def _spawn_robot(self, name: str, t_param: float):
        info = self.param.get_contact_info(t_param)
        contact_body = info["point"]
        normal_out = info["normal_outward"]
        normal_in = -normal_out

        # Place robot at: contact + (radius + approach_distance)*normal_out (object at origin)
        spawn_body = contact_body + (ROBOT_RADIUS + self.approach_distance) * normal_out
        heading = float(np.arctan2(normal_in[1], normal_in[0]))

        return create_robot(
            kinematics=self.kinematics,
            model="wheel",
            position=(float(spawn_body[0]), float(spawn_body[1]), 0.0),
            orientation=heading,
            name=name,
        )

    def _get_force(self, robot, link_idx: int) -> float:
        try:
            f = get_contact_force(robot.uid, self.object_uid, linkIndexA=link_idx, max_contacts=8)
            return float(np.linalg.norm(np.array(f[:2])))
        except Exception:
            return 0.0

    def _compute_cmd(self, robot_pos: np.ndarray, robot_heading: float, object_pos: np.ndarray, object_theta: float, t_param: float) -> np.ndarray:
        # Contact point (world) and outward normal (world)
        info = self.param.get_contact_info(t_param)
        p_body = info["point"]
        n_body = info["normal_outward"]
        c = np.cos(object_theta)
        s = np.sin(object_theta)
        R = np.array([[c, -s], [s, c]])

        p_world = R @ p_body + object_pos
        n_world = R @ n_body

        intended = p_world + ROBOT_RADIUS * n_world
        pos_err = intended - robot_pos

        # Heading should point from robot -> contact point (adaptively)
        to_contact = p_world - robot_pos
        desired_heading = float(np.arctan2(to_contact[1], to_contact[0]))
        h_err = wrap_angle(desired_heading - robot_heading)

        vel_xy = self.kp_pos * pos_err
        speed = float(np.linalg.norm(vel_xy))
        if speed > self.max_speed:
            vel_xy = vel_xy * (self.max_speed / speed)

        omega = float(np.clip(self.kp_heading * h_err, -3.0, 3.0))
        return np.array([vel_xy[0], vel_xy[1], omega], dtype=float)

    def run(self, *, gui: bool, duration: float) -> Dict:
        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0

        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_v_lin, obj_v_ang = pyb.getBaseVelocity(self.object_uid)
                theta = pyb.getEulerFromQuaternion(obj_orn)[2]
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_vel = np.array([obj_v_lin[0], obj_v_lin[1]])
                object_omega = float(obj_v_ang[2])

                # Robot states
                r1_pos, r1_heading, r1_vel = self.robot1.get_state()
                r2_pos, r2_heading, r2_vel = self.robot2.get_state()
                r1_pos = np.array(r1_pos[:2])
                r2_pos = np.array(r2_pos[:2])

                # Commands (Phase-2-style: position+heading only; force is just observed)
                cmd1 = self._compute_cmd(r1_pos, r1_heading, object_pos, theta, self.t1)
                cmd2 = self._compute_cmd(r2_pos, r2_heading, object_pos, theta, self.t2)

                self.robot1.command_velocity(cmd1 if self.kinematics == "holonomic" else np.array([cmd1[0], cmd1[2]]))
                self.robot2.command_velocity(cmd2 if self.kinematics == "holonomic" else np.array([cmd2[0], cmd2[2]]))

                f1 = self._get_force(self.robot1, self.r1_link)
                f2 = self._get_force(self.robot2, self.r2_link)

                # Record
                self.history.times.append(t)
                self.history.object_velocities.append(object_vel.copy())
                self.history.object_angular_velocities.append(object_omega)

                self.history.r1_pos.append(r1_pos.copy())
                self.history.r1_vel.append(np.array(r1_vel).copy())
                self.history.r1_cmd.append(cmd1.copy())
                self.history.r1_force.append(f1)

                self.history.r2_pos.append(r2_pos.copy())
                self.history.r2_vel.append(np.array(r2_vel).copy())
                self.history.r2_cmd.append(cmd2.copy())
                self.history.r2_force.append(f2)

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            if gui:
                time.sleep(TIMESTEP * 0.3)

        # Basic summary
        f1 = np.array(self.history.r1_force)
        f2 = np.array(self.history.r2_force)
        return {
            "r1_peak_force": float(np.max(f1)) if len(f1) else 0.0,
            "r2_peak_force": float(np.max(f2)) if len(f2) else 0.0,
            "r1_mean_force": float(np.mean(f1)) if len(f1) else 0.0,
            "r2_mean_force": float(np.mean(f2)) if len(f2) else 0.0,
        }

    def plot(self, save_path: Optional[Path] = None):
        times = np.array(self.history.times)
        f1 = np.array(self.history.r1_force)
        f2 = np.array(self.history.r2_force)
        obj_v = np.array(self.history.object_velocities)
        obj_w = np.array(self.history.object_angular_velocities)

        r1_cmd = np.array(self.history.r1_cmd)
        r2_cmd = np.array(self.history.r2_cmd)
        r1_vel = np.array(self.history.r1_vel)
        r2_vel = np.array(self.history.r2_vel)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"Two-Robot Phase-2 Test ({self.kinematics}), t1={self.t1:.3f}, t2={self.t2:.3f}",
                     fontsize=14, fontweight="bold")

        # Forces
        ax = axes[0, 0]
        ax.plot(times, f1, "b-", label="R1 |force|")
        ax.plot(times, f2, "g-", label="R2 |force|")
        ax.axhline(y=self.contact_threshold, color="k", linestyle="--", alpha=0.5, label="threshold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.set_title("Contact forces")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Object motion
        ax = axes[0, 1]
        ax.plot(times, obj_v[:, 0], "r-", label="obj vx")
        ax.plot(times, obj_v[:, 1], "m-", label="obj vy")
        ax.plot(times, obj_w, "k--", label="obj ω")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("m/s, rad/s")
        ax.set_title("Object velocities")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Robot vx (cmd vs actual)
        ax = axes[1, 0]
        ax.plot(times, r1_cmd[:, 0], "b--", label="R1 cmd vx")
        ax.plot(times, r1_vel[:, 0], "b-", alpha=0.7, label="R1 act vx")
        ax.plot(times, r2_cmd[:, 0], "g--", label="R2 cmd vx")
        ax.plot(times, r2_vel[:, 0], "g-", alpha=0.7, label="R2 act vx")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("vx (m/s)")
        ax.set_title("Robot vx tracking (world frame)")
        ax.legend(ncols=2, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Robot omega (cmd vs actual)
        ax = axes[1, 1]
        ax.plot(times, r1_cmd[:, 2], "b--", label="R1 cmd ω")
        ax.plot(times, r1_vel[:, 2], "b-", alpha=0.7, label="R1 act ω")
        ax.plot(times, r2_cmd[:, 2], "g--", label="R2 cmd ω")
        ax.plot(times, r2_vel[:, 2], "g-", alpha=0.7, label="R2 act ω")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ω (rad/s)")
        ax.set_title("Robot ω tracking")
        ax.legend(ncols=2, fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Two-robot Phase-2 hybrid reliability test")
    parser.add_argument("--kinematics", "-k", default="holonomic", choices=["holonomic", "diffdrive"])
    parser.add_argument("--case", choices=["same_edge", "opposite"], default="same_edge")
    parser.add_argument("--t1", type=float, default=0.10)
    parser.add_argument("--t2", type=float, default=0.20)
    parser.add_argument("--approach-distance", type=float, default=0.15)
    parser.add_argument("--kp-pos", type=float, default=2.0)
    parser.add_argument("--kp-heading", type=float, default=10.0)
    parser.add_argument("--max-speed", type=float, default=0.25)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    # Provide canonical defaults for the two cases (if user didn't override)
    t1 = float(args.t1)
    t2 = float(args.t2)
    if "--t1" not in sys.argv and "--t2" not in sys.argv:
        if args.case == "same_edge":
            t1, t2 = 0.10, 0.20
        else:
            t1, t2 = 0.10, 0.60

    setup_pybullet(gui=not args.no_gui)

    test = TwoRobotPhase2Test(
        kinematics=args.kinematics,
        t1=t1,
        t2=t2,
        approach_distance=args.approach_distance,
        kp_pos=args.kp_pos,
        kp_heading=args.kp_heading,
        max_speed=args.max_speed,
        contact_threshold=args.contact_threshold,
    )
    results = test.run(gui=not args.no_gui, duration=args.duration)

    print("\n" + "=" * 60)
    print("TWO-ROBOT PHASE-2 SUMMARY")
    print("=" * 60)
    print(f"  case: {args.case}")
    print(f"  t1={t1:.3f}, t2={t2:.3f}")
    print(f"  R1 peak force: {results['r1_peak_force']:.3f} N, mean: {results['r1_mean_force']:.3f} N")
    print(f"  R2 peak force: {results['r2_peak_force']:.3f} N, mean: {results['r2_mean_force']:.3f} N")
    print("=" * 60)

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"two_robot_phase2_{args.case}.png"
        test.plot(out)
    else:
        test.plot()

    pyb.disconnect()


if __name__ == "__main__":
    main()

