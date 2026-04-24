#!/usr/bin/env python3
"""
Smoke test: diff-drive cheat control with NEW URDF only.

This is adapted from compare_diffdrive.py but intentionally removes the dummy
holonomic robot comparison. It runs a single DiffDriveWheelRobot instance using
the new disc-bumper URDF to isolate URDF-side effects under cheat control.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg

matplotlib.use("Agg")

rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))

from contact_maintain.diffdrive_wheel_robot import DiffDriveWheelRobot


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)


def straight_trajectory(t, speed=0.3):
    return np.array([speed, 0.0])


def pure_rotation_trajectory(t, omega=1.0):
    return np.array([0.0, omega])


def arc_trajectory(t, v=0.3, omega=0.3):
    return np.array([v, omega])


def scurve_trajectory(t, v=0.3, period=4.0, omega_max=0.5):
    omega = omega_max * np.sin(2 * np.pi * t / period)
    return np.array([v, omega])


def zigzag_trajectory(t, v=0.3, period=2.0, omega_max=1.0):
    omega = omega_max * np.sign(np.sin(2 * np.pi * t / period))
    return np.array([v, omega])


TRAJECTORIES = {
    "straight": straight_trajectory,
    "rotation": pure_rotation_trajectory,
    "arc": arc_trajectory,
    "scurve": scurve_trajectory,
    "zigzag": zigzag_trajectory,
}


def setup_pybullet(gui=True):
    if gui:
        pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        pyb.connect(pyb.DIRECT)
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(ground, -1, lateralFriction=0.5)
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=0,
            cameraPitch=-60,
            cameraTargetPosition=[0, 0, 0],
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)


class DiffDriveNewUrdfSmokeTest:
    def __init__(self):
        self.robot = DiffDriveWheelRobot(
            str(Path(pkg_path) / "urdf" / "diffdrive_wheel_robot_disc_bumper.urdf"),
            position=(0.0, 0.0),
            orientation=0.0,
        )
        self.history = {
            "times": [],
            "cmd_vel": [],
            "pos": [],
            "heading": [],
            "vel": [],
            "wheel_vel": [],
            "cmd_wheel_vel": [],
        }

    def reset(self):
        self.robot.reset(position=(0.0, 0.0), orientation=0.0)
        for key in self.history:
            self.history[key] = []

    def run_test(self, trajectory_name="arc", duration=10.0, gui=True):
        self.reset()
        traj = TRAJECTORIES.get(trajectory_name, arc_trajectory)
        n_steps = int(duration / TIMESTEP)
        t = 0.0
        for step in range(n_steps):
            if step % CTRL_STEP == 0:
                cmd = traj(t)
                self.robot.command_velocity(cmd)
                pos, heading, vel = self.robot.get_state()
                wheel_vel = self.robot.get_wheel_velocities()
                self.history["times"].append(t)
                self.history["cmd_vel"].append(cmd.copy())
                self.history["pos"].append(pos.copy())
                self.history["heading"].append(heading)
                self.history["vel"].append(vel.copy())
                self.history["wheel_vel"].append(wheel_vel.copy())
                self.history["cmd_wheel_vel"].append(self.robot.last_wheel_speeds.copy())
            pyb.stepSimulation()
            if gui:
                time.sleep(TIMESTEP * 0.5)
            t += TIMESTEP
        return self.calculate_errors()

    def calculate_errors(self):
        cmd_vel = np.array(self.history["cmd_vel"])
        vel = np.array(self.history["vel"])
        heading = np.array(self.history["heading"])
        body_forward_v = np.array(
            [np.cos(h) * v[0] + np.sin(h) * v[1] for h, v in zip(heading, vel)]
        )
        v_err = np.abs(body_forward_v - cmd_vel[:, 0])
        omega_err = np.abs(vel[:, 2] - cmd_vel[:, 1])
        return {
            "v_rmse": float(np.sqrt(np.mean(v_err**2))),
            "omega_rmse": float(np.sqrt(np.mean(omega_err**2))),
        }

    def plot_results(self, save_path=None, show=False):
        times = np.array(self.history["times"])
        cmd = np.array(self.history["cmd_vel"])
        pos = np.array(self.history["pos"])
        heading = np.array(self.history["heading"])
        vel = np.array(self.history["vel"])
        wheel_vel = np.array(self.history["wheel_vel"])
        cmd_wheel_vel = np.array(self.history["cmd_wheel_vel"])

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Smoke: DiffDriveWheelRobot with NEW URDF", fontsize=14)

        rel = pos - pos[0]
        axes[0, 0].plot(rel[:, 0], rel[:, 1], "r-", lw=2)
        axes[0, 0].plot(0, 0, "go", ms=8)
        axes[0, 0].set_title("Relative Trajectory")
        axes[0, 0].axis("equal")
        axes[0, 0].grid(True, alpha=0.3)

        body_forward_v = [np.cos(h) * v[0] + np.sin(h) * v[1] for h, v in zip(heading, vel)]
        axes[0, 1].plot(times, cmd[:, 0], "k-", lw=2, label="cmd v")
        axes[0, 1].plot(times, body_forward_v, "r--", lw=1.5, label="actual v")
        axes[0, 1].set_title("Forward Velocity")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[0, 2].plot(times, cmd[:, 1], "k-", lw=2, label="cmd w")
        axes[0, 2].plot(times, vel[:, 2], "r--", lw=1.5, label="actual w")
        axes[0, 2].set_title("Angular Velocity")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        axes[1, 0].plot(times, np.degrees(heading), "m-", lw=1.5)
        axes[1, 0].set_title("Heading (deg)")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(times, wheel_vel[:, 0], "b-", lw=1.5, label="left actual")
        axes[1, 1].plot(times, wheel_vel[:, 1], "r-", lw=1.5, label="right actual")
        axes[1, 1].plot(times, cmd_wheel_vel[:, 0], "b--", lw=1, alpha=0.5, label="left cmd")
        axes[1, 1].plot(times, cmd_wheel_vel[:, 1], "r--", lw=1, alpha=0.5, label="right cmd")
        axes[1, 1].set_title("Wheel Velocities")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=8)

        v_err = np.abs(np.array(body_forward_v) - cmd[:, 0])
        w_err = np.abs(vel[:, 2] - cmd[:, 1])
        axes[1, 2].plot(times, v_err * 100, "c-", lw=1.5, label="v err (cm/s)")
        axes[1, 2].plot(times, w_err, "y-", lw=1.5, label="w err (rad/s)")
        axes[1, 2].set_title("Tracking Error")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        if show:
            try:
                plt.switch_backend("TkAgg")
                plt.show()
            except Exception as exc:
                print(f"Could not display plot interactively: {exc}")
        return fig


def main():
    parser = argparse.ArgumentParser(description="Smoke test on new diffdrive URDF (cheat control)")
    parser.add_argument(
        "--trajectory",
        "-t",
        default="arc",
        choices=list(TRAJECTORIES.keys()),
        help="Trajectory to test",
    )
    parser.add_argument("--duration", "-d", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to file")
    args = parser.parse_args()

    if args.no_gui and args.save_plot is None:
        args.save_plot = f"/tmp/smoke_diffdrive_new_urdf_{args.trajectory}.png"

    setup_pybullet(gui=not args.no_gui)
    test = DiffDriveNewUrdfSmokeTest()
    errors = test.run_test(trajectory_name=args.trajectory, duration=args.duration, gui=not args.no_gui)

    print("\n" + "=" * 56)
    print("SMOKE RESULTS (NEW URDF + CHEAT CONTROL)")
    print("=" * 56)
    print(f"  trajectory: {args.trajectory}")
    print(f"  duration:   {args.duration:.1f}s")
    print(f"  v RMSE:     {errors['v_rmse']*100:.3f} cm/s")
    print(f"  omega RMSE: {errors['omega_rmse']:.4f} rad/s")
    print("=" * 56)

    test.plot_results(save_path=args.save_plot, show=not args.no_gui)
    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()
    pyb.disconnect()


if __name__ == "__main__":
    main()
