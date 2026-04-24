#!/usr/bin/env python3
"""
Basic velocity-tracking tests for DiffDriveWheelPhysicsRobot.

Two focused scenarios:
1) no_load: velocity control tracking on free motion.
2) with_load: velocity control tracking while carrying external inertial load.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg

rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))

from contact_maintain.diffdrive_wheel_physics_robot import DiffDriveWheelPhysicsRobot
from contact_maintain.diffdrive_wheel_robot import (
    MAX_WHEEL_SPEED,
    WHEEL_BASE,
    WHEEL_RADIUS,
)
TIMESTEP = 1.0 / 240.0
CTRL_HZ = 100
CTRL_STEP = int((1.0 / CTRL_HZ) / TIMESTEP)


def setup_sim(gui: bool, ground_friction: float):
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
            cameraDistance=2.5,
            cameraYaw=30,
            cameraPitch=-45,
            cameraTargetPosition=[0.5, 0.0, 0.0],
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    return ground


def body_forward_speed(world_vel, heading):
    return float(np.cos(heading) * world_vel[0] + np.sin(heading) * world_vel[1])


def print_kinematic_limits():
    max_linear = WHEEL_RADIUS * MAX_WHEEL_SPEED
    max_angular = 2.0 * WHEEL_RADIUS * MAX_WHEEL_SPEED / WHEEL_BASE
    max_wheel_rpm = MAX_WHEEL_SPEED * 60.0 / (2.0 * np.pi)
    print("\n[kinematic_limits]")
    print(f"  wheel radius: {WHEEL_RADIUS:.4f} m")
    print(f"  wheel base:   {WHEEL_BASE:.4f} m")
    print(f"  max wheel speed: {MAX_WHEEL_SPEED:.3f} rad/s ({max_wheel_rpm:.1f} rpm)")
    print(f"  max linear speed:  {max_linear:.4f} m/s")
    print(f"  max angular speed: {max_angular:.4f} rad/s")


def run_no_load(robot: DiffDriveWheelPhysicsRobot, duration: float, gui: bool, v_cmd: float, w_cmd: float):
    n_steps = int(duration / TIMESTEP)
    t = 0.0
    forward_err = []
    omega_err = []

    for k in range(n_steps):
        if k % CTRL_STEP == 0:
            robot.command_velocity([v_cmd, w_cmd])
            _pos, heading, vel = robot.get_state()
            forward = body_forward_speed(vel[:2], heading)
            forward_err.append(abs(forward - v_cmd))
            omega_err.append(abs(float(vel[2]) - w_cmd))
        pyb.stepSimulation()
        if gui:
            time.sleep(TIMESTEP * 0.5)
        t += TIMESTEP

    print("\n[no_load] results")
    print(f"  duration: {duration:.2f}s")
    print(f"  cmd: v={v_cmd:.3f} m/s, omega={w_cmd:.3f} rad/s")
    print(f"  forward RMSE: {np.sqrt(np.mean(np.square(forward_err))):.4f} m/s")
    print(f"  omega   RMSE: {np.sqrt(np.mean(np.square(omega_err))):.4f} rad/s")


def create_path_boxes():
    """Spawn free boxes with varying masses along robot travel path."""
    masses = [0.5, 1.0, 2.0, 4.0]
    x_positions = [0.55, 0.95, 1.35, 1.75]
    out = []
    col = pyb.createCollisionShape(pyb.GEOM_BOX, halfExtents=[0.06, 0.06, 0.06])
    for i, (m, x) in enumerate(zip(masses, x_positions)):
        color = [0.2, 0.8 - 0.12 * i, 0.2 + 0.15 * i, 1.0]
        vis = pyb.createVisualShape(
            pyb.GEOM_BOX,
            halfExtents=[0.06, 0.06, 0.06],
            rgbaColor=color,
        )
        uid = pyb.createMultiBody(
            baseMass=float(m),
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(x), 0.0, 0.06],
            baseOrientation=[0, 0, 0, 1],
        )
        pyb.changeDynamics(uid, -1, lateralFriction=1.0)
        out.append(dict(uid=uid, mass=float(m), x0=float(x)))
    return out


def create_path_boxes_behind():
    """Spawn free boxes behind robot (negative x) for reverse-contact tests."""
    masses = [0.5, 1.0, 2.0, 4.0]
    x_positions = [-0.55, -0.95, -1.35, -1.75]
    out = []
    col = pyb.createCollisionShape(pyb.GEOM_BOX, halfExtents=[0.06, 0.06, 0.06])
    for i, (m, x) in enumerate(zip(masses, x_positions)):
        color = [0.2, 0.8 - 0.12 * i, 0.2 + 0.15 * i, 1.0]
        vis = pyb.createVisualShape(
            pyb.GEOM_BOX,
            halfExtents=[0.06, 0.06, 0.06],
            rgbaColor=color,
        )
        uid = pyb.createMultiBody(
            baseMass=float(m),
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(x), 0.0, 0.06],
            baseOrientation=[0, 0, 0, 1],
        )
        pyb.changeDynamics(uid, -1, lateralFriction=1.0)
        out.append(dict(uid=uid, mass=float(m), x0=float(x)))
    return out


def run_with_load(robot: DiffDriveWheelPhysicsRobot, duration: float, gui: bool, v_cmd: float, w_cmd: float):
    boxes = create_path_boxes()
    n_steps = int(duration / TIMESTEP)
    forward_err = []
    omega_err = []

    for k in range(n_steps):
        if k % CTRL_STEP == 0:
            robot.command_velocity([v_cmd, w_cmd])
            _pos, heading, vel = robot.get_state()
            forward = body_forward_speed(vel[:2], heading)
            forward_err.append(abs(forward - v_cmd))
            omega_err.append(abs(float(vel[2]) - w_cmd))

        pyb.stepSimulation()
        if gui:
            time.sleep(TIMESTEP * 0.5)
    box_displacements = []
    for b in boxes:
        pos_xy = np.array(pyb.getBasePositionAndOrientation(b["uid"])[0][:2], dtype=float)
        disp = float(np.linalg.norm(pos_xy - np.array([b["x0"], 0.0])))
        box_displacements.append((b["mass"], disp))

    print("\n[with_load] results")
    print(f"  duration: {duration:.2f}s")
    print(f"  cmd: v={v_cmd:.3f} m/s, omega={w_cmd:.3f} rad/s")
    print("  free obstacle boxes on path: masses=[0.5, 1.0, 2.0, 4.0] kg")
    print(f"  forward RMSE: {np.sqrt(np.mean(np.square(forward_err))):.4f} m/s")
    print(f"  omega   RMSE: {np.sqrt(np.mean(np.square(omega_err))):.4f} rad/s")
    for mass, disp in box_displacements:
        print(f"  box mass={mass:.1f} kg displacement={disp:.3f} m")


def run_force_sensor(robot: DiffDriveWheelPhysicsRobot, duration: float, gui: bool):
    """Drive through obstacle boxes and print bumper force readout."""
    boxes = create_path_boxes_behind()
    box_uids = [b["uid"] for b in boxes]
    n_steps = int(duration / TIMESTEP)
    v_cmd = -0.16
    w_cmd = 0.00
    force_hist = []
    print_every_ctrl = 10  # print every 10 control ticks to reduce spam
    ctrl_count = 0

    print("\n[force_sensor] streaming bumper contact force...")
    print("  format: t, |Fxy|, Fx, Fy")

    for k in range(n_steps):
        if k % CTRL_STEP == 0:
            robot.command_velocity([v_cmd, w_cmd])
            force_vec = robot.get_contact_force(box_uids, max_contacts=4)
            fx, fy = float(force_vec[0]), float(force_vec[1])
            fxy = float(np.linalg.norm(np.array([fx, fy], dtype=float)))
            force_hist.append(fxy)

            if ctrl_count % print_every_ctrl == 0:
                t = k * TIMESTEP
                print(f"  t={t:6.2f}s |Fxy|={fxy:7.3f} N (Fx={fx:7.3f}, Fy={fy:7.3f})")
            ctrl_count += 1

        pyb.stepSimulation()
        if gui:
            time.sleep(TIMESTEP * 0.5)

    if force_hist:
        arr = np.array(force_hist, dtype=float)
        print("\n[force_sensor] summary")
        print(f"  samples: {len(arr)}")
        print(f"  avg |Fxy|:  {np.mean(arr):.3f} N")
        print(f"  peak |Fxy|: {np.max(arr):.3f} N")
        print(f"  p95 |Fxy|:  {np.percentile(arr, 95):.3f} N")


def main():
    parser = argparse.ArgumentParser(description="DiffDriveWheelPhysicsRobot basic tests")
    parser.add_argument(
        "--mode",
        choices=["spawn_only", "no_load", "with_load", "force_sensor", "both"],
        default="both",
        help="Which test to run",
    )
    parser.add_argument("--duration", type=float, default=12.0, help="Duration per test")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    parser.add_argument(
        "--spawn-with-load",
        action="store_true",
        help="In spawn_only mode, spawn free path boxes",
    )
    parser.add_argument(
        "--no-cheat-control",
        action="store_true",
        help="Disable planar-joint cheat control and use wheel-physics drive",
    )
    parser.add_argument(
        "--ground-friction",
        type=float,
        default=None,
        help="Override plane lateral friction",
    )
    parser.add_argument(
        "--wheel-friction",
        type=float,
        default=None,
        help="Override wheel lateral friction",
    )
    parser.add_argument(
        "--caster-friction",
        type=float,
        default=None,
        help="Override caster lateral friction",
    )
    parser.add_argument(
        "--cmd-v",
        type=float,
        default=0.20,
        help="Requested forward speed command (m/s)",
    )
    parser.add_argument(
        "--cmd-omega",
        type=float,
        default=0.00,
        help="Requested angular speed command (rad/s)",
    )
    args = parser.parse_args()

    gui = not args.no_gui
    # Use the same default physics for cheat and non-cheat modes.
    ground_friction = args.ground_friction
    wheel_friction = args.wheel_friction
    caster_friction = args.caster_friction
    if ground_friction is None:
        ground_friction = 0.5
    if wheel_friction is None:
        wheel_friction = 0.01
    if caster_friction is None:
        caster_friction = 0.01

    use_cheat_control = not args.no_cheat_control
    print_kinematic_limits()
    v_req = float(args.cmd_v)
    w_req = float(args.cmd_omega)
    max_linear = WHEEL_RADIUS * MAX_WHEEL_SPEED
    max_angular = 2.0 * WHEEL_RADIUS * MAX_WHEEL_SPEED / WHEEL_BASE
    v_limited = float(np.clip(v_req, -max_linear, max_linear))
    w_limited = float(np.clip(w_req, -max_angular, max_angular))
    print(
        f"  requested cmd: v={v_req:.3f} m/s, omega={w_req:.3f} rad/s "
        f"(wheel-implied limit: v={v_limited:.3f}, omega={w_limited:.3f})"
    )

    if args.mode == "spawn_only":
        setup_sim(gui=gui, ground_friction=ground_friction)
        robot = DiffDriveWheelPhysicsRobot(
            position=(0.0, 0.0),
            orientation=0.0,
            wheel_lateral_friction=wheel_friction,
            caster_lateral_friction=caster_friction,
            use_planar_cheat_control=use_cheat_control,
        )
        if args.spawn_with_load:
            create_path_boxes()

        print("\n[spawn_only] robot spawned.")
        print("  - No control commands are sent.")
        print("  - Simulation keeps stepping for visual inspection.")
        print(f"  - cheat_control={use_cheat_control}")
        print(f"  - frictions: ground={ground_friction}, wheel={wheel_friction}, caster={caster_friction}")
        if args.spawn_with_load:
            print("  - Free obstacle boxes spawned on path.")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                pyb.stepSimulation()
                if gui:
                    time.sleep(TIMESTEP * 0.8)
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            pyb.disconnect()
        return

    # Run each selected scenario in a fresh simulation.
    if args.mode in ("no_load", "both"):
        setup_sim(gui=gui, ground_friction=ground_friction)
        robot = DiffDriveWheelPhysicsRobot(
            position=(0.0, 0.0),
            orientation=0.0,
            wheel_lateral_friction=wheel_friction,
            caster_lateral_friction=caster_friction,
            use_planar_cheat_control=use_cheat_control,
        )
        run_no_load(robot, duration=args.duration, gui=gui, v_cmd=v_req, w_cmd=w_req)
        pyb.disconnect()

    if args.mode in ("with_load", "both"):
        setup_sim(gui=gui, ground_friction=ground_friction)
        robot = DiffDriveWheelPhysicsRobot(
            position=(0.0, 0.0),
            orientation=0.0,
            wheel_lateral_friction=wheel_friction,
            caster_lateral_friction=caster_friction,
            use_planar_cheat_control=use_cheat_control,
        )
        run_with_load(robot, duration=args.duration, gui=gui, v_cmd=v_req, w_cmd=w_req)
        pyb.disconnect()

    if args.mode == "force_sensor":
        setup_sim(gui=gui, ground_friction=ground_friction)
        robot = DiffDriveWheelPhysicsRobot(
            position=(0.0, 0.0),
            orientation=0.0,
            wheel_lateral_friction=wheel_friction,
            caster_lateral_friction=caster_friction,
            use_planar_cheat_control=use_cheat_control,
        )
        run_force_sensor(robot, duration=args.duration, gui=gui)
        pyb.disconnect()


if __name__ == "__main__":
    main()
