#!/usr/bin/env python3
"""
Force vs Speed Test (single robot, straight push through object CoM)

Goal
----
1) Validate whether higher commanded speed tends to yield higher measured contact force.
2) Sanity-check whether `get_contact_force` is a usable signal for later hybrid controllers.

Important interpretation note (why force “caps” in this test)
------------------------------------------------------------
In the **single pusher** case, once the object is sliding at (roughly) steady velocity,
the contact force you measure is dominated by **Coulomb friction** (and whatever
limits/impedances exist in the robot drive + contact solver).

That means:
- Increasing commanded speed does **not** necessarily increase the *steady* measured force,
  because the object accelerates until friction + drive limits balance out.
- The force trace is most informative during **transients** (acceleration / impact / stick-slip),
  not during long steady-velocity pushing.

Also, with **two robots** on opposite sides, the “force” reported at each contact is not a clean
measurement of “this robot’s applied force”. It can include constraint/impulse effects from the
rigid-body solver as the object+robots act like a coupled system.

Design
------
- Spawn a light object in front of the robot.
- Drive straight (+X) at a constant speed for a fixed duration.
- Measure contact force magnitude at the robot's contact link.
- Sweep across speeds and report peak/steady force.

Usage
-----
python test_force_vs_speed.py --kinematics holonomic --speed-list 0.05,0.1,0.2,0.3 --save-dir /tmp/results
python test_force_vs_speed.py --kinematics diffdrive --speed-list 0.1,0.2,0.3 --no-gui

Ramp test (transient-focused)
-----------------------------
Use `--ramp` to linearly increase the target speed from 0 to `--ramp-vmax` over `--ramp-time`.
This is useful to see whether the *measured* force changes during acceleration.
"""

import argparse
import sys
import time
import subprocess
import json
import os
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

from object_utils import create_standard_objects
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import generic_to_pybullet
from contact_maintain.pyb_simulation import (
    get_contact_force_copy_from_henrik,
    ExponentialSmoother,
)


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

DEFAULT_OBJECT_SHAPE = "rectangle"
DEFAULT_OBJECT_HEIGHT = 0.2
DEFAULT_OBJECT_FRICTION = 0.8


def setup_pybullet(gui: bool = True):
    """Connect (if needed) and initialize world.

    Important: do NOT repeatedly connect/disconnect inside a sweep loop.
    Use `pyb.resetSimulation()` between runs and call this again to re-init the world.
    """
    if not pyb.isConnected():
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
            cameraTargetPosition=[0.5, 0.0, 0.0],
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)


def _robot_contact_link_index(robot) -> int:
    if hasattr(robot, "bumper_link_idx") and robot.bumper_link_idx is not None:
        return int(robot.bumper_link_idx)
    if hasattr(robot, "contact_link_idx") and robot.contact_link_idx is not None:
        return int(robot.contact_link_idx)
    return -1


@dataclass
class SweepRun:
    speed: float
    times: List[float] = field(default_factory=list)
    contact_forces: List[float] = field(default_factory=list)
    robot_velocities: List[np.ndarray] = field(default_factory=list)  # [vx, vy, omega] in world frame
    robot_positions: List[np.ndarray] = field(default_factory=list)
    object_positions: List[np.ndarray] = field(default_factory=list)
    object_velocities: List[np.ndarray] = field(default_factory=list)  # [vx, vy, omega]


def run_single_speed(
    *,
    kinematics: str,
    speed: float,
    duration: float,
    settle_time: float,
    gui: bool,
    object_mass: float,
    object_x: float,
    ramp: bool = False,
    ramp_vmax: float = 0.3,
    ramp_time: float = 2.0,
) -> SweepRun:
    standard_objects = create_standard_objects()
    generic_object = standard_objects[DEFAULT_OBJECT_SHAPE]

    # Object at (object_x, 0), robot at origin heading +X.
    obj_uid = generic_to_pybullet(
        generic_object,
        height=DEFAULT_OBJECT_HEIGHT,
        position=(object_x, 0.0, 0.0),
        orientation=0.0,
        color=(0.4, 0.7, 0.4, 1.0),
    )
    pyb.changeDynamics(obj_uid, -1, lateralFriction=DEFAULT_OBJECT_FRICTION, mass=float(object_mass))

    robot = create_robot(
        kinematics=kinematics,
        model="wheel",
        position=(0.0, 0.0, 0.0),
        orientation=0.0,
        name="force_speed_robot",
    )
    link_idx = _robot_contact_link_index(robot)

    # Henrik-style exponential smoother for contact force
    smoother = ExponentialSmoother(tau=0.05, x0=np.zeros(3))
    smooth_dt = CTRL_STEP * TIMESTEP

    n_steps = int(duration / TIMESTEP)
    settle_steps = int(settle_time / TIMESTEP)

    run = SweepRun(speed=speed)
    t = 0.0
    step_count = 0

    for step in range(n_steps):
        if step_count % CTRL_STEP == 0:
            robot_pos, _, robot_vel = robot.get_state()
            obj_pos, obj_orn = pyb.getBasePositionAndOrientation(obj_uid)
            obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(obj_uid)

            # Force signal (Henrik-style helper + exponential smoother)
            try:
                f_vec = get_contact_force_copy_from_henrik(
                    uid_slider=obj_uid,
                    uid_pusher=robot.uid,
                    linkIndex_slider=-1,
                    linkIndex_pusher=link_idx,
                    max_contacts=8,
                    smoother=smoother,
                    dt=smooth_dt,
                )
                print(f_vec, "f_vec from henrik")
                f_mag = float(np.linalg.norm(np.array(f_vec[:2])))
            except Exception:
                f_mag = 0.0

            # Command
            if step < settle_steps:
                cmd = np.array([0.0, 0.0, 0.0])
            else:
                if ramp:
                    # Linear ramp of target speed after settle_time:
                    # v(t) = ramp_vmax * clamp((t-settle_time)/ramp_time, 0..1)
                    alpha = (t - settle_time) / max(1e-6, ramp_time)
                    v_cmd = float(ramp_vmax * np.clip(alpha, 0.0, 1.0))
                else:
                    v_cmd = float(speed)

                if kinematics == "holonomic":
                    cmd = np.array([v_cmd, 0.0, 0.0])
                else:  # diffdrive
                    cmd = np.array([v_cmd, 0.0])  # (v_forward, omega)

            robot.command_velocity(cmd)

            run.times.append(t)
            run.contact_forces.append(f_mag)
            run.robot_velocities.append(np.array(robot_vel).copy())
            run.robot_positions.append(np.array(robot_pos[:2]).copy())
            run.object_positions.append(np.array([obj_pos[0], obj_pos[1]]))
            # store object velocity as [vx, vy, omega_z]
            run.object_velocities.append(
                np.array([obj_vel_lin[0], obj_vel_lin[1], obj_vel_ang[2]], dtype=float)
            )

        pyb.stepSimulation()
        t += TIMESTEP
        step_count += 1
        if gui:
            time.sleep(TIMESTEP * 0.3)

    return run


def compute_force_metrics(run: SweepRun, settle_time: float) -> Dict[str, float]:
    times = np.array(run.times)
    forces = np.array(run.contact_forces)
    if len(times) == 0:
        return {"peak_force": 0.0, "steady_force": 0.0}

    # Only consider after settle_time
    mask = times >= settle_time
    forces = forces[mask] if np.any(mask) else forces
    peak = float(np.max(forces)) if len(forces) else 0.0

    # "steady" = mean of last 25% of samples after settle
    if len(forces) >= 4:
        start = int(0.75 * len(forces))
        steady = float(np.mean(forces[start:]))
    else:
        steady = float(np.mean(forces)) if len(forces) else 0.0

    return {"peak_force": peak, "steady_force": steady}


def plot_sweep(runs: List[SweepRun], settle_time: float, save_dir: Optional[Path]):
    speeds = [r.speed for r in runs]
    peaks = [compute_force_metrics(r, settle_time)["peak_force"] for r in runs]
    steadies = [compute_force_metrics(r, settle_time)["steady_force"] for r in runs]

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle("Force vs Speed Sweep", fontsize=14, fontweight="bold")

    # Time series (overlay)
    ax = axes[0, 0]
    for r in runs:
        ax.plot(r.times, r.contact_forces, linewidth=1.2, label=f"v={r.speed:.2f}")
    ax.axvline(x=settle_time, color="k", linestyle="--", alpha=0.5, label="settle end")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|contact force| (N)")
    ax.set_title("Contact force time series")
    ax.legend(ncols=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Peak vs speed
    ax = axes[0, 1]
    ax.plot(speeds, peaks, "o-", linewidth=1.5)
    ax.set_xlabel("Command speed (m/s)")
    ax.set_ylabel("Peak |force| (N)")
    ax.set_title("Peak force vs speed")
    ax.grid(True, alpha=0.3)

    # Steady vs speed
    ax = axes[1, 0]
    ax.plot(speeds, steadies, "o-", linewidth=1.5)
    ax.set_xlabel("Command speed (m/s)")
    ax.set_ylabel("Steady |force| (N)")
    ax.set_title("Steady force vs speed")
    ax.grid(True, alpha=0.3)

    # Robot velocity tracking (vx over time)
    ax = axes[1, 1]
    for r in runs:
        vels = np.array(r.robot_velocities)
        if len(vels) == 0:
            continue
        ax.plot(r.times, vels[:, 0], linewidth=1.2, label=f"vx (v={r.speed:.2f})")
    # commanded reference lines
    for v in speeds:
        ax.axhline(y=v, color="k", linestyle="--", alpha=0.12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Robot vx (m/s)")
    ax.set_title("Robot velocity tracking (vx)")
    ax.legend(ncols=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Trajectories (last run)
    ax = axes[2, 0]
    last = runs[-1]
    rp = np.array(last.robot_positions)
    op = np.array(last.object_positions)
    ax.plot(rp[:, 0], rp[:, 1], "b-", label="robot")
    ax.plot(op[:, 0], op[:, 1], "g-", label="object")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectories (v={last.speed:.2f})")
    ax.axis("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Force vs robot vx (last run): shows whether force correlates with achieved speed
    ax = axes[2, 1]
    vels = np.array(last.robot_velocities)
    forces = np.array(last.contact_forces)
    if len(vels) and len(forces):
        ax.plot(vels[:, 0], forces, "o", markersize=2.5, alpha=0.6)
    ax.set_xlabel("Robot vx (m/s)")
    ax.set_ylabel("|contact force| (N)")
    ax.set_title(f"Force vs achieved vx (v={last.speed:.2f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "force_vs_speed.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out}")
    else:
        plt.show()
    plt.close()


def plot_ramp(run: SweepRun, *, settle_time: float, ramp_vmax: float, ramp_time: float, save_dir: Optional[Path]):
    """Plot ramp test: object & robot motion + force over time."""
    times = np.array(run.times)
    forces = np.array(run.contact_forces)
    vels = np.array(run.robot_velocities)
    vx = vels[:, 0] if len(vels) else np.zeros_like(times)
    obj_vels = np.array(run.object_velocities) if run.object_velocities else np.zeros((len(times), 3))
    obj_vx = obj_vels[:, 0]
    obj_vy = obj_vels[:, 1]
    obj_omega = obj_vels[:, 2]

    # Command profile for visualization
    v_cmd = np.zeros_like(times)
    for i, t in enumerate(times):
        if t < settle_time:
            v_cmd[i] = 0.0
        else:
            alpha = (t - settle_time) / max(1e-6, ramp_time)
            v_cmd[i] = ramp_vmax * np.clip(alpha, 0.0, 1.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Force vs Speed (Ramp Test)", fontsize=14, fontweight="bold")

    # Robot & object velocities
    ax = axes[0]
    ax.plot(times, v_cmd, "k--", linewidth=1.5, label="cmd robot vx")
    ax.plot(times, vx, "b-", linewidth=1.5, label="robot vx")
    ax.plot(times, obj_vx, "r-", linewidth=1.0, label="obj vx")
    ax.plot(times, obj_vy, "g-", linewidth=1.0, label="obj vy")
    ax2 = ax.twinx()
    ax2.plot(times, obj_omega, "m:", linewidth=1.0, label="obj omega")
    ax2.set_ylabel("obj ω (rad/s)", color="m")
    ax.axvline(x=settle_time, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("vx (m/s)")
    ax.set_title("Robot & object velocities")
    ax.grid(True, alpha=0.3)
    # Build combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # Force over time
    ax = axes[1]
    ax.plot(times, forces, "m-", linewidth=1.5)
    ax.axvline(x=settle_time, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("|force| (N)")
    ax.set_title("Measured contact force (magnitude)")
    ax.grid(True, alpha=0.3)

    # Force vs achieved vx
    ax = axes[2]
    ax.plot(vx, forces, "o", markersize=2.5, alpha=0.6)
    ax.set_xlabel("robot vx (m/s)")
    ax.set_ylabel("|force| (N)")
    ax.set_title("Force vs achieved robot vx")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "force_vs_speed_ramp.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Force vs speed sweep (single robot push)")
    parser.add_argument("--kinematics", "-k", default="holonomic", choices=["holonomic", "diffdrive"])
    parser.add_argument("--speed-list", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3",
                        help="Comma-separated speeds (m/s)")
    parser.add_argument("--ramp", action="store_true",
                        help="Ramp speed from 0 to --ramp-vmax over --ramp-time (transient test)")
    parser.add_argument("--ramp-vmax", type=float, default=0.3,
                        help="Max speed for ramp test (m/s)")
    parser.add_argument("--ramp-time", type=float, default=2.0,
                        help="Ramp duration after settle time (s)")
    parser.add_argument("--_single-speed", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--object-mass", type=float, default=1.0)
    parser.add_argument("--object-x", type=float, default=1.0, help="Object CoM x position (m)")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    speeds = [float(s.strip()) for s in args.speed_list.split(",") if s.strip()]
    if not speeds:
        raise ValueError("Empty --speed-list")

    # PyBullet can occasionally crash when repeatedly resetting the simulation in-process.
    # To make sweeps robust, run each speed in a separate subprocess unless explicitly
    # doing a single-speed run.
    if (not args.ramp) and args._single_speed is None and len(speeds) > 1:
        runs: List[SweepRun] = []
        for v in speeds:
            cmd = [
                sys.executable,
                __file__,
                "--kinematics", args.kinematics,
                "--speed-list", str(v),
                "--_single-speed", str(v),
                "--duration", str(args.duration),
                "--settle-time", str(args.settle_time),
                "--object-mass", str(args.object_mass),
                "--object-x", str(args.object_x),
            ]
            if args.no_gui:
                cmd.append("--no-gui")

            print(f"\nRunning speed={v:.3f} m/s (subprocess) ...")
            out = subprocess.check_output(cmd, text=True)
            # Expect a final JSON line: {"speed":..., "peak_force":..., "steady_force":..., "times":..., "forces":...}
            payload = json.loads(out.strip().splitlines()[-1])

            run = SweepRun(speed=float(payload["speed"]))
            run.times = list(payload["times"])
            run.contact_forces = list(payload["forces"])
            run.robot_velocities = [np.array(v) for v in payload["robot_velocities"]]
            run.robot_positions = [np.array(p) for p in payload["robot_positions"]]
            run.object_positions = [np.array(p) for p in payload["object_positions"]]
            run.object_velocities = [np.array(v) for v in payload.get("object_velocities", [])]
            metrics = compute_force_metrics(run, args.settle_time)
            print(f"  peak_force={metrics['peak_force']:.3f} N, steady_force={metrics['steady_force']:.3f} N")
            runs.append(run)

        save_dir = Path(args.save_dir) if args.save_dir else None
        plot_sweep(runs, args.settle_time, save_dir)
        # PyBullet native libs can occasionally crash during interpreter teardown
        # even though we only ran it in subprocesses. Avoid teardown paths.
        os._exit(0)

    # Single run (in-process): either single speed or ramp test.
    v = float(args._single_speed) if args._single_speed is not None else float(speeds[0])
    setup_pybullet(gui=not args.no_gui)
    run = run_single_speed(
        kinematics=args.kinematics,
        speed=v,
        duration=args.duration,
        settle_time=args.settle_time,
        gui=not args.no_gui,
        object_mass=args.object_mass,
        object_x=args.object_x,
        ramp=bool(args.ramp),
        ramp_vmax=float(args.ramp_vmax),
        ramp_time=float(args.ramp_time),
    )
    metrics = compute_force_metrics(run, args.settle_time)
    print(f"  peak_force={metrics['peak_force']:.3f} N, steady_force={metrics['steady_force']:.3f} N")
    pyb.disconnect()

    # If this is a ramp run, plot it directly here.
    if args.ramp and args._single_speed is None:
        save_dir = Path(args.save_dir) if args.save_dir else None
        plot_ramp(
            run,
            settle_time=float(args.settle_time),
            ramp_vmax=float(args.ramp_vmax),
            ramp_time=float(args.ramp_time),
            save_dir=save_dir,
        )

    # Emit machine-readable payload for the parent process.
    payload = {
        "speed": float(v),
        "peak_force": metrics["peak_force"],
        "steady_force": metrics["steady_force"],
        "times": [float(x) for x in run.times],
        "forces": [float(x) for x in run.contact_forces],
        "robot_velocities": [[float(vv[0]), float(vv[1]), float(vv[2])] for vv in run.robot_velocities],
        "robot_positions": [[float(p[0]), float(p[1])] for p in run.robot_positions],
        "object_positions": [[float(p[0]), float(p[1])] for p in run.object_positions],
        "object_velocities": [[float(vv[0]), float(vv[1]), float(vv[2])] for vv in run.object_velocities],
        "ramp": bool(args.ramp),
        "ramp_vmax": float(args.ramp_vmax),
        "ramp_time": float(args.ramp_time),
    }
    # print(json.dumps(payload))

    # PyBullet native libs can occasionally crash during interpreter teardown (heap corruption).
    # Ramp mode runs in-process; exit hard after producing outputs to keep runs stable.
    if args.ramp and args._single_speed is None:
        os._exit(0)


if __name__ == "__main__":
    main()

