#!/usr/bin/env python3
"""
Swarm Navigation Test (obj-to-generic + spawn + move to contact point)

Uses the distributed swarm architecture (OverworldSimulator + DistributedMonitor).
Tests only:
  1. Spawning – robots placed like test_magnum_motion_planning.py (offset along outward normal from target t_param)
  2. Initial movement to the desired contact point – navigation to target t_params
  3. No pushing – object is static, navigation_only=True so state stays NAVIGATING

No pushing phase, no reconfiguration. Optional --save-dir for plots and metrics.

Usage:
    # Basic navigation test with 3 robots
    python test_obj_to_generic_with_swarm_for_navigation.py --num-robots 3

    # With APF (default), static_single, or divide_conquer
    python test_obj_to_generic_with_swarm_for_navigation.py -n 4 --navigation-scheme apf

    # Save results and plot
    python test_obj_to_generic_with_swarm_for_navigation.py -n 3 --duration 20 --save-dir /tmp/swarm_nav/
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import ContactPointParameterization
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import obj_to_generic
from contact_maintain.overworld_sim import OverworldSimulator


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

DEFAULT_OBJECT_SHAPE = 'right_triangle'
DEFAULT_OBJECT_FRICTION = 0.3
# Match test_magnum_motion_planning.py spawn geometry for holonomic robots.
DEFAULT_ROBOT_RADIUS = 0.06
DEFAULT_APPROACH_DISTANCE = DEFAULT_ROBOT_RADIUS + 0.03
# Diff-drive wheel model needs larger stand-off in this test scene.
DIFFDRIVE_WHEEL_APPROACH_DISTANCE = DEFAULT_ROBOT_RADIUS + 0.06
OBJ_FILE_MAP = {
    'right_triangle': 'right_triangle.obj',
    'pi': 'pi.obj',
    'root': 'root.obj',
    'rect': 'rect.obj',
    'hourglass': 'hourglass.obj',
    'meteor': 'meteor.obj',
}


# ============================================================================
# SIMULATION HELPERS
# ============================================================================

def setup_pybullet(gui=True, hide_gui_panels=False):
    """Initialize PyBullet."""
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)

    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)

    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])

    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)

    urdf_dir = Path(pkg_path) / "urdf"
    if urdf_dir.exists():
        pyb.setAdditionalSearchPath(str(urdf_dir))

    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
        # On some WSL/OpenGL stacks, hiding panels can break camera interaction.
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0 if hide_gui_panels else 1)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_TINY_RENDERER, 0)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_MOUSE_PICKING, 1)

    return ground


def get_object_state(object_uid):
    """Get object state from PyBullet."""
    pos, orn = pyb.getBasePositionAndOrientation(object_uid)
    vel_lin, vel_ang = pyb.getBaseVelocity(object_uid)
    euler = pyb.getEulerFromQuaternion(orn)

    return {
        'position': np.array([pos[0], pos[1]]),
        'orientation': euler[2],
        'velocity': np.array([vel_lin[0], vel_lin[1]]),
        'angular_velocity': vel_ang[2],
    }


# ============================================================================
# TEST CLASS
# ============================================================================

class SwarmNavigationTest:
    """
    Navigation-only test: spawn, assign targets, run until duration.
    Uses OverworldSimulator with navigation_only=True (no pushing).
    Object is loaded via obj_to_generic (OBJ file -> GenericObject + PyBullet).
    """

    def __init__(
        self,
        num_robots: int = 3,
        object_shape: str = DEFAULT_OBJECT_SHAPE,
        obj_file: str = None,
        kinematics: str = 'holonomic',
        model: str = 'dummy',
        duration: float = 30.0,
        navigation_scheme: str = 'apf',
        startup_mode: str = 'quick',
        shutdown_after_s: float = 10.0,
        velocity_print_interval_s: Optional[float] = 0.5,
        approach_distance: Optional[float] = None,
    ):
        self.num_robots = num_robots
        self.object_shape = object_shape
        self.obj_file = obj_file or f"{object_shape}.obj"
        self.kinematics = kinematics
        self.model = model
        self.duration = duration
        self.navigation_scheme = navigation_scheme
        self.startup_mode = startup_mode
        self.shutdown_after_s = shutdown_after_s
        self.velocity_print_interval_s = velocity_print_interval_s
        self.approach_distance = (
            float(approach_distance)
            if approach_distance is not None
            else self._get_approach_distance()
        )

        # Create object via obj_to_generic (OBJ -> GenericObject + PyBullet)
        print(f"\nLoading OBJ file: {self.obj_file}...")
        self.generic_object, self.object_uid = obj_to_generic(
            obj_path=self.obj_file,
            shape_name=object_shape,
            position=(0.0, 0.0, 0.2),
            orientation=0.0,
            mass=1.0,
            lateral_friction=DEFAULT_OBJECT_FRICTION,
            blind_test=True,
        )
        print(f"  Mode: NAVIGATION ONLY (object static, no pushing)")

        self.contact_point_parameterization = ContactPointParameterization(self.generic_object)
        t_list = self._default_t_params_list()
        self.initial_t_params = {
            f"R_{i+1:02d}": t_list[i] for i in range(self.num_robots)
        }

        # Create robots (spawn matched to magnum motion planning test)
        self.robots: Dict[str, object] = {}
        self._create_robots(t_list)

        # Overworld simulator (distributed architecture, navigation only)
        self.overworld = OverworldSimulator(
            robots=self.robots,
            object_uid=self.object_uid,
            generic_object=self.generic_object,
            navigation_scheme=navigation_scheme,
            push_controller_type='phase7',
            navigation_only=True,
            startup_mode=startup_mode,
        )

        # History
        self.history = {
            'times': [],
            'robot_positions': {name: [] for name in self.robots},
            'robot_states': {name: [] for name in self.robots},
            'distances_to_target': {name: [] for name in self.robots},
        }

    def _default_t_params_list(self) -> List[float]:
        """Default t_params for this shape.
        
        1) Try to load Magnum Four solution from urdf/magnum_four_cache.json
           (same cache used by test_magnum_motion_planning.py).
        2) If not available, fall back to uniform layout + optional swap.
        """
        cache_file = Path(pkg_path) / "urdf" / "magnum_four_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                if self.object_shape in cache_data:
                    t_params = cache_data[self.object_shape]
                    # Ensure list length matches robot count when possible
                    if isinstance(t_params, list) and len(t_params) >= self.num_robots:
                        print(f"Using cached Magnum Four t_params for '{self.object_shape}': "
                              f"{[f'{v:.4f}' for v in t_params[:self.num_robots]]}")
                        return [float(tp) % 1.0 for tp in t_params[:self.num_robots]]
            except (json.JSONDecodeError, KeyError, OSError) as e:
                print(f"Warning: failed to load Magnum Four cache '{cache_file}': {e}")

        # Fallback: simple uniform layout with optional swap (legacy behavior)
        offset = 0.17
        initial_t_param = [(i / self.num_robots + offset) % 1.0 for i in range(self.num_robots)]
        if self.num_robots >= 4:
            initial_t_param[1], initial_t_param[3] = initial_t_param[3], initial_t_param[1]
        print(
            f"Using fallback uniform t_params for '{self.object_shape}': "
            f"{[f'{v:.4f}' for v in initial_t_param]}"
        )
        return initial_t_param

    def _create_robots(self, t_params_list: List[float]):
        """Spawn each robot outward from its target contact (same as test_magnum_motion_planning.py)."""
        if len(t_params_list) != self.num_robots:
            raise ValueError(f"Expected {self.num_robots} t_params, got {len(t_params_list)}")

        print(f"\nCreating {self.num_robots} robots ({self.kinematics}, {self.model})...")
        print(f"  Spawn: contact + {self.approach_distance:.3f} m along outward normal (magnum style)")

        for i in range(self.num_robots):
            name = f"R_{i+1:02d}"
            target_t_param = t_params_list[i]

            contact_info = self.contact_point_parameterization.get_contact_info(target_t_param)
            contact_point_body = np.array(contact_info['point'], dtype=float)
            normal_outward = np.array(contact_info['normal_outward'], dtype=float)
            normal_inward = np.array(contact_info['normal_inward'], dtype=float)

            spawn_position_body = contact_point_body + self.approach_distance * normal_outward
            robot_x = float(spawn_position_body[0])
            robot_y = float(spawn_position_body[1])
            robot_heading = float(np.arctan2(normal_inward[1], normal_inward[0]))

            robot = create_robot(
                kinematics=self.kinematics,
                model=self.model,
                position=(robot_x, robot_y),
                orientation=robot_heading,
                name=name,
            )
            self.robots[name] = robot
            print(
                f"  {name}: pos=({robot_x:.3f}, {robot_y:.3f}) heading={robot_heading:.3f} rad, "
                f"t={target_t_param:.4f}"
            )

    def _get_approach_distance(self) -> float:
        """Robot-type specific spawn stand-off from contact point."""
        if self.kinematics == 'diffdrive' and self.model == 'wheel':
            return DIFFDRIVE_WHEEL_APPROACH_DISTANCE
        return DEFAULT_APPROACH_DISTANCE

    def _print_velocity_snapshot(self, t: float, velocities: Dict[str, np.ndarray]) -> None:
        """Log commanded vs measured base velocity and contact/distance (PyBullet joint state)."""
        latched = getattr(self.overworld, "_nav_quick_all_contact_latched", False)
        print(f"[t={t:.2f}s] nav_quick_contact_latch={latched}")
        for name in sorted(self.robots.keys()):
            _, _, vel_meas = self.robots[name].get_state()
            cmd = velocities[name]
            st = self.overworld.get_status()[name]
            v_cmd = np.asarray(cmd).flatten()
            v_act = np.asarray(vel_meas).flatten()
            print(
                f"  {name}: cmd[vx,vy,w]=({v_cmd[0]:+.4f},{v_cmd[1]:+.4f},{v_cmd[2]:+.4f}) "
                f"act[vx,vy,w]=({v_act[0]:+.5f},{v_act[1]:+.5f},{v_act[2]:+.5f}) "
                f"|F={st['contact_force']:.3f}N dist={st['distance_to_target']:.4f} "
                f"contact={st['in_contact']}"
            )

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _all_robots_in_contact(self) -> bool:
        """Check if all robots are currently in contact with the object."""
        status = self.overworld.get_status()
        return all(bool(status[name]['in_contact']) for name in self.robots)

    def _all_robots_nearly_still(self, lin_eps: float = 0.01, ang_eps: float = 0.05) -> bool:
        """Check if all robots have near-zero base velocity."""
        for name in self.robots:
            _, _, vel = self.robots[name].get_state()
            vel = np.asarray(vel).flatten()
            lin_speed = float(np.linalg.norm(vel[:2]))
            ang_speed = float(abs(vel[2]))
            if lin_speed > lin_eps or ang_speed > ang_eps:
                return False
        return True

    def _run_post_contact_self_rotation(self, gui: bool) -> None:
        """For diffdrive tests: rotate each robot +90 deg after contact and rest."""
        print("\n[Post phase] Waiting for all diffdrive robots to be in contact and still...")

        # Let robots settle to nearly zero speed while preserving contact.
        settle_timeout_s = 6.0
        settle_steps = int(settle_timeout_s / TIMESTEP)
        settled = False
        for _ in range(settle_steps):
            for robot in self.robots.values():
                robot.command_velocity(np.array([0.0, 0.0]))
            pyb.stepSimulation()
            # if self._all_robots_in_contact() and self._all_robots_nearly_still():
            if self._all_robots_nearly_still():
                settled = True
                break
            if gui:
                time.sleep(TIMESTEP * 0.3)

        if not settled:
            print("[Post phase] Could not confirm contact+still for all robots; skipping self-rotation.")
            return

        print("[Post phase] All robots contact+still confirmed. Rotating each robot +90 deg...")
        start_headings = {name: self.robots[name].get_state()[1] for name in self.robots}
        target_headings = {name: start_headings[name] + (np.pi / 2.0) for name in self.robots}

        rot_timeout_s = 6.0
        rot_steps = int(rot_timeout_s / TIMESTEP)
        k_p = 2.0
        omega_max = 1.0
        done_tol = 0.03  # rad

        for _ in range(rot_steps):
            all_done = True
            for name, robot in self.robots.items():
                _, heading, _ = robot.get_state()
                err = self._wrap_to_pi(target_headings[name] - heading)
                if abs(err) > done_tol:
                    all_done = False
                omega_cmd = float(np.clip(k_p * err, -omega_max, omega_max))
                robot.command_velocity(np.array([0.0, omega_cmd]))

            pyb.stepSimulation()
            if all_done:
                break
            if gui:
                time.sleep(TIMESTEP * 0.3)

        for robot in self.robots.values():
            robot.command_velocity(np.array([0.0, 0.0]))
        for _ in range(int(0.5 / TIMESTEP)):
            pyb.stepSimulation()
            if gui:
                time.sleep(TIMESTEP * 0.3)
        print("[Post phase] Self-rotation complete.")

    def run_test(self, gui: bool = True) -> Dict:
        """Run navigation-only test: spawn + move to target t_params."""
        effective_duration_s = min(float(self.duration), float(self.shutdown_after_s))
        n_steps = int(effective_duration_s / TIMESTEP)
        step_count = 0
        t = 0.0

        initial_t_params = self.initial_t_params

        print("\n" + "=" * 60)
        print("  SWARM NAVIGATION TEST (spawn + move to contact point, no pushing)")
        print("=" * 60)
        print(f"  Robots: {self.num_robots}  |  Scheme: {self.navigation_scheme}")
        print(f"  Startup mode: {self.startup_mode}")
        print(f"  Duration: {self.duration:.2f}s, shutdown_after: {self.shutdown_after_s:.2f}s "
              f"(effective: {effective_duration_s:.2f}s)")
        print(f"  Initial t_params: {[f'{v:.4f}' for v in initial_t_params.values()]}")
        print("=" * 60 + "\n")

        self.overworld.assign_targets(initial_t_params)

        control_tick = 0
        vel_print_period = 0
        if self.velocity_print_interval_s is not None and self.velocity_print_interval_s > 0:
            vel_print_period = max(1, int(self.velocity_print_interval_s * CTRL_FREQ))

        for step in range(n_steps):
            obj_state = get_object_state(self.object_uid)

            if step_count % CTRL_STEP == 0:
                self.overworld.update(1.0 / CTRL_FREQ, obj_state)

                # Velocities come from the selected navigation scheme (apf / static_single / divide_conquer)
                # via OverworldSimulator -> DistributedMonitor -> NavigationController.
                velocities = self.overworld.compute_velocities(obj_state)

                for name, robot in self.robots.items():
                    cmd = velocities[name]
                    if self.kinematics == 'diffdrive' and len(cmd) == 3:
                        pos, heading, _ = robot.get_state()
                        v_forward = cmd[0] * np.cos(heading) + cmd[1] * np.sin(heading)
                        robot.command_velocity(np.array([v_forward, cmd[2]]))
                    else:
                        robot.command_velocity(cmd)

                control_tick += 1
                if vel_print_period and (control_tick % vel_print_period) == 0:
                    self._print_velocity_snapshot(t, velocities)

                # Record history
                self.history['times'].append(t)
                status = self.overworld.get_status()
                for name in self.robots:
                    pos, _, _ = self.robots[name].get_state()
                    self.history['robot_positions'][name].append(pos.copy())
                    self.history['robot_states'][name].append(status[name]['state'])
                    self.history['distances_to_target'][name].append(status[name]['distance_to_target'])

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            if gui:
                # Pump GUI events explicitly to keep mouse/camera controls responsive.
                pyb.getKeyboardEvents()
                pyb.getMouseEvents()
                time.sleep(TIMESTEP * 0.3)

        if self.kinematics == 'diffdrive':
            self._run_post_contact_self_rotation(gui=gui)

        return self._compute_metrics()

    def _compute_metrics(self) -> Dict:
        """Compute test metrics."""
        times = np.array(self.history['times'])
        robot_metrics = {}
        for name in self.robots:
            dists = np.array(self.history['distances_to_target'][name])
            dists_finite = np.where(np.isfinite(dists), dists, np.nan)
            robot_metrics[name] = {
                'final_distance_to_target': float(np.nanmin(dists_finite)) if np.any(np.isfinite(dists_finite)) else float('inf'),
                'min_distance_to_target': float(np.nanmin(dists_finite)) if np.any(np.isfinite(dists_finite)) else float('inf'),
            }
        return {
            'robots': robot_metrics,
            'history': self.history,
        }

    def save_results(self, save_dir: str, metrics: Dict):
        """Save metrics and plot."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        to_save = {k: v for k, v in metrics.items() if k != 'history'}
        with open(save_path / "metrics.json", 'w') as f:
            json.dump(to_save, f, indent=2, default=str)

        self._generate_plot(save_path / "swarm_navigation_test.png")
        print(f"Results saved to {save_path}")

    def _generate_plot(self, save_path: Path):
        """Plot trajectories and distance to target."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f'Swarm Navigation Test (no pushing)\n'
            f'{self.num_robots} robots | {self.navigation_scheme}',
            fontsize=12
        )
        times = np.array(self.history['times'])
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_robots))

        # Trajectories
        ax = axes[0]
        bounds = self.generic_object.geometry.bounds
        ax.fill(
            [bounds[0], bounds[2], bounds[2], bounds[0]],
            [bounds[1], bounds[1], bounds[3], bounds[3]],
            alpha=0.3, color='green', label='Object'
        )
        for (name, positions), color in zip(self.history['robot_positions'].items(), colors):
            pos_arr = np.array(positions)
            if len(pos_arr):
                ax.plot(pos_arr[:, 0], pos_arr[:, 1], '-', color=color, linewidth=1.5, label=name)
                ax.plot(pos_arr[0, 0], pos_arr[0, 1], 'o', color=color, markersize=8)
                ax.plot(pos_arr[-1, 0], pos_arr[-1, 1], 's', color=color, markersize=8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectories')
        ax.legend(fontsize=8)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        # Distance to target
        ax = axes[1]
        for (name, distances), color in zip(self.history['distances_to_target'].items(), colors):
            d = np.array(distances)
            d = np.where(np.isfinite(d), d, np.nan)
            ax.plot(times, d, '-', color=color, linewidth=1.5, label=name)
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Threshold (0.05m)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to Target (m)')
        ax.set_title('Distance to Target')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Swarm navigation test: spawn + move to contact point (no pushing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_obj_to_generic_with_swarm_for_navigation.py --num-robots 3
    python test_obj_to_generic_with_swarm_for_navigation.py -n 4 --navigation-scheme static_single
    python test_obj_to_generic_with_swarm_for_navigation.py -n 3 --duration 20 --save-dir /tmp/swarm_nav/
        """
    )
    parser.add_argument("--num-robots", "-n", type=int, default=3)
    parser.add_argument("--kinematics", "-k", default="holonomic", choices=['holonomic', 'diffdrive'])
    parser.add_argument("--model", "-m", default="dummy", choices=['dummy', 'wheel'])
    parser.add_argument("--duration", "-d", type=float, default=30.0)
    parser.add_argument(
        "--shutdown-after",
        type=float,
        default=10.0,
        help="Stop the simulation after this many seconds (default: 10).",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=DEFAULT_OBJECT_SHAPE,
        choices=list(OBJ_FILE_MAP.keys()),
        help="Object shape name",
    )
    parser.add_argument("--obj-file", type=str, default=None,
                        help="OBJ file path (default: <object>.obj)")
    parser.add_argument("--navigation-scheme", type=str, default="apf",
                        choices=['apf', 'static_single', 'divide_conquer'])
    parser.add_argument(
        "--startup-mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help=(
            "Startup behavior: 'quick' uses direct rotate-then-creep approach to contact; "
            "'full' uses navigation-scheme startup."
        ),
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument(
        "--hide-gui-panels",
        action="store_true",
        help="Hide PyBullet GUI panels (may reduce interactivity on some systems).",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--velocity-print-interval",
        type=float,
        default=0.3,
        help="Seconds between terminal logs of cmd vs actual velocity (0 disables).",
    )
    parser.add_argument(
        "--approach-distance",
        type=float,
        default=None,
        help=(
            "Spawn offset from contact point along outward normal (meters). "
            "Default is model-specific."
        ),
    )
    args = parser.parse_args()

    print("\nInitializing PyBullet...")
    ground = setup_pybullet(gui=not args.no_gui, hide_gui_panels=args.hide_gui_panels)
    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)

    if args.obj_file is not None:
        obj_file = args.obj_file
        print(f"Using custom OBJ file override: {obj_file}")
    else:
        obj_file = OBJ_FILE_MAP[args.object]
        print(f"Loading mapped OBJ file: {obj_file} for shape '{args.object}'")
    vprint = None if args.velocity_print_interval <= 0 else args.velocity_print_interval
    test = SwarmNavigationTest(
        num_robots=args.num_robots,
        object_shape=args.object,
        obj_file=obj_file,
        kinematics=args.kinematics,
        model=args.model,
        duration=args.duration,
        navigation_scheme=args.navigation_scheme,
        startup_mode=args.startup_mode,
        shutdown_after_s=args.shutdown_after,
        velocity_print_interval_s=vprint,
        approach_distance=args.approach_distance,
    )

    metrics = test.run_test(gui=not args.no_gui)

    print("\n" + "=" * 60)
    print("  NAVIGATION TEST METRICS")
    print("=" * 60)
    for name, m in metrics['robots'].items():
        dist = test.overworld.get_status()[name]['distance_to_target']
        state = test.overworld.get_status()[name]['state']
        print(f"  {name}: state={state}, dist_to_target={dist:.3f}m, min_dist={m['min_distance_to_target']:.3f}m")
    print("=" * 60)

    if args.save_dir:
        test.save_results(args.save_dir, metrics)

    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()

    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
