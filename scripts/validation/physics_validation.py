#!/usr/bin/env python3
"""
Physics Validation Script

Compares BoundaryMotionPredictor and DummyBoundaryPointController predictions
from object_utils.py against actual PyBullet physics simulation.

This validates whether the dynamics model in object_utils.py is accurate
enough for controller design.

NOTE: Validation results may show discrepancies between the object_utils.py
physics model and PyBullet. This is expected because:
1. object_utils.py uses simplified 2D quasi-static friction model
2. PyBullet uses full 3D rigid body dynamics with contact simulation
3. Friction models differ (limit surface vs Coulomb approximation)

The discrepancies observed here inform us about the limitations of the
simplified model and help guide controller design decisions.

Usage:
    # Run with default headless mode and save plot:
    python physics_validation.py --shape rectangle --save-plot /tmp/validation.png
    
    # Run with GUI:
    python physics_validation.py --shape rectangle --gui
"""
import argparse
import time
from pathlib import Path
import sys
import os

import numpy as np
import pybullet as pyb
import pybullet_data

# Use non-interactive backend for headless mode
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import (
    GenericObject, create_standard_objects, 
    DynamicObjectModel, BoundaryMotionPredictor, 
    DummyBoundaryPointController,
    ContactPointParameterization, ContactPoint
)
from contact_maintain.object_bridge import (
    generic_to_pybullet, BridgedObject, print_physics_comparison
)


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
DEFAULT_DURATION = 3.0
DEFAULT_T_PARAM = 0.25  # Track point at 25% along boundary


# ============================================================================
# SIMULATION SETUP
# ============================================================================

def setup_pybullet(gui=False):
    """Initialize PyBullet simulation.
    
    Parameters
    ----------
    gui : bool
        If True, use GUI mode. Default is False (headless/DIRECT mode).
    
    Returns
    -------
    int
        PyBullet client ID.
    int
        Ground plane UID.
    """
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    # Ground friction will be set later based on object's kinetic_friction
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-60,
            cameraTargetPosition=[0, 0, 0]
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    
    return client_id, ground


def set_ground_friction(ground_uid: int, friction: float):
    """Set ground plane friction coefficient.
    
    Parameters
    ----------
    ground_uid : int
        PyBullet ground plane UID.
    friction : float
        Friction coefficient (should match object's kinetic_friction).
    """
    pyb.changeDynamics(ground_uid, -1, lateralFriction=friction)


# ============================================================================
# VALIDATION TEST
# ============================================================================

class PhysicsValidator:
    """Compare object_utils dynamics against PyBullet using DummyBoundaryPointController."""
    
    def __init__(self, generic_obj: GenericObject, ground_uid: int, height: float = 0.1, verbose: bool = True):
        """Initialize with a GenericObject.
        
        Parameters
        ----------
        generic_obj : GenericObject
            The object to test.
        ground_uid : int
            PyBullet ground plane UID (for setting friction).
        height : float
            Extrusion height for PyBullet.
        verbose : bool
            If True, print detailed physics property comparison.
        """
        self.generic_obj = generic_obj
        self.ground_uid = ground_uid
        self.height = height
        self.verbose = verbose
        
        # Get friction coefficients
        # kinetic_friction / static_friction = ground-object friction
        # lateral_friction = object-robot contact friction
        self.ground_friction = getattr(generic_obj, 'static_friction', 0.2)
        self.contact_friction = generic_obj.lateral_friction
        
        # Print physics properties for reference
        if verbose:
            print(f"\nGenericObject '{generic_obj.name}' properties:")
            print(f"  Mass: {generic_obj.mass:.4f} kg")
            print(f"  Moment of Inertia (2D): {generic_obj.moment_of_inertia:.6f} kg⋅m²")
            print(f"  Ground Friction (kinetic): {self.ground_friction:.4f}")
            print(f"  Ground Friction (static): {getattr(generic_obj, 'static_friction', 0.4):.4f}")
            print(f"  Contact Friction (lateral): {self.contact_friction:.4f}")
        
        # Create DynamicObjectModel for prediction
        self.dynamics_model = DynamicObjectModel(
            generic_obj, 
            dt=TIMESTEP,
            position_init=np.array([0.0, 0.0]),
            orientation_init=0.0
        )
        
        # Controller will be created in run_validation with the wrench
        self.controller = None
        
        # Set ground friction from object's kinetic_friction
        # PyBullet computes effective friction as product of both body frictions
        # So we set ground friction = kinetic_friction and object friction = 1.0
        set_ground_friction(ground_uid, self.ground_friction)
        if verbose:
            print(f"  Ground plane friction set to: {self.ground_friction:.4f}")
        
        # Create PyBullet object with friction = 1.0 (so ground friction dominates)
        self.pybullet_uid = generic_to_pybullet(
            generic_obj, 
            height=height, 
            position=(0, 0, 0),
            color=(0.4, 0.7, 0.4, 1.0),
            ground_friction_mode=True  # Use kinetic_friction for object's lateralFriction
        )
        
        # Verify physics properties were transferred correctly
        if verbose:
            print_physics_comparison(generic_obj, self.pybullet_uid, height)
        
        # History for plotting
        self.history = {
            'times': [],
            'pyb_positions': [],
            'pyb_orientations': [],
            'pyb_velocities': [],
            'pyb_angular_velocities': [],
            'pred_positions': [],
            'pred_orientations': [],
            'pred_velocities': [],
            'pred_angular_velocities': [],
            'pyb_point_positions': [],
            'pred_point_positions': [],
            'pyb_point_velocities': [],
            'pred_point_velocities': [],
            'wrenches': [],
        }
    
    def reset(self, position=(0, 0), orientation=0.0):
        """Reset both models to initial state."""
        # Reset PyBullet
        pos_3d = [position[0], position[1], self.height / 2]
        orn = pyb.getQuaternionFromEuler([0, 0, orientation])
        pyb.resetBasePositionAndOrientation(self.pybullet_uid, pos_3d, orn)
        pyb.resetBaseVelocity(self.pybullet_uid, [0, 0, 0], [0, 0, 0])
        
        # Reset dynamics model
        self.dynamics_model.reset_state(
            position=np.array(position),
            orientation=orientation,
            velocity=np.zeros(2),
            angular_velocity=0.0
        )
        
        # Clear history
        for key in self.history:
            self.history[key] = []
    
    def apply_wrench_pybullet(self, wrench):
        """Apply wrench to PyBullet object in the object's local (body) frame.
        
        The wrench [Fx, Fy, tau] is defined in the object's local coordinate frame,
        consistent with how DynamicObjectModel in object_utils.py models physics.
        PyBullet's LINK_FRAME applies forces relative to the link's local frame.
        """
        force = [wrench[0], wrench[1], 0]
        torque = [0, 0, wrench[2]]
        # Apply at center of mass in local frame
        # pyb.applyExternalForce(self.pybullet_uid, -1, force, [0, 0, 0], pyb.LINK_FRAME)
        # pyb.applyExternalTorque(self.pybullet_uid, -1, torque, pyb.LINK_FRAME)
        pos, orn = pyb.getBasePositionAndOrientation(self.pybullet_uid)
        euler = pyb.getEulerFromQuaternion(orn)
        theta = euler[2]  # Yaw angle (rotation around z-axis)
        
        # Rotation matrix from body to world frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Transform force from body frame to world frame
        fx_body, fy_body = wrench[0], wrench[1]
        fx_world = cos_theta * fx_body - sin_theta * fy_body
        fy_world = sin_theta * fx_body + cos_theta * fy_body
        
        # Torque is scalar, same in both frames
        force = [fx_world, fy_world, 0]
        torque = [0, 0, wrench[2]]
        
        # Apply in WORLD_FRAME
        pyb.applyExternalForce(self.pybullet_uid, -1, force, pos, pyb.WORLD_FRAME)
        pyb.applyExternalTorque(self.pybullet_uid, -1, torque, pyb.WORLD_FRAME)

    def get_pybullet_state(self):
        """Get current state from PyBullet."""
        pos, orn = pyb.getBasePositionAndOrientation(self.pybullet_uid)
        vel_lin, vel_ang = pyb.getBaseVelocity(self.pybullet_uid)
        euler = pyb.getEulerFromQuaternion(orn)
        
        return {
            'position': np.array([pos[0], pos[1]]),
            'orientation': euler[2],
            'velocity': np.array([vel_lin[0], vel_lin[1]]),
            'angular_velocity': vel_ang[2]
        }
    
    def get_boundary_point_world(self, t_param, position, orientation):
        """Get boundary point position in world frame."""
        # Get point in body frame
        param = ContactPointParameterization(self.generic_obj)
        contact_info = param.get_contact_info(t_param)
        point_body = contact_info['point']
        
        # Transform to world frame
        cos_t = np.cos(orientation)
        sin_t = np.sin(orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        point_world = R @ point_body + position
        
        return point_world
    
    def run_validation(self, wrench, duration=DEFAULT_DURATION, t_param=DEFAULT_T_PARAM):
        """Run validation comparing predictions to PyBullet.
        
        Uses DummyBoundaryPointController for the prediction model, which
        converts wrench to contact forces and uses DynamicObjectModel.update_state().
        
        Parameters
        ----------
        wrench : array-like
            Constant wrench [Fx, Fy, tau] to apply.
        duration : float
            Simulation duration in seconds.
        t_param : float
            Boundary parameter for point tracking.
        
        Returns
        -------
        dict
            Error metrics.
        """
        wrench = np.array(wrench)
        n_steps = int(duration / TIMESTEP)
        
        self.reset()
        
        # Create DummyBoundaryPointController with the wrench
        self.controller = DummyBoundaryPointController(
            self.dynamics_model, 
            t_param=t_param, 
            fixed_wrench=wrench
        )
        self.controller.initialize()
        
        print(f"\nRunning validation for {duration}s with wrench={wrench}")
        print(f"Tracking boundary point at t_param={t_param}")
        print(f"Using DummyBoundaryPointController with {len(self.controller.dummy_contacts)} contact points")
        
        for step in range(n_steps):
            t = step * TIMESTEP
            
            # Get PyBullet state
            pyb_state = self.get_pybullet_state()
            
            # Get prediction model state  
            pred_state = {
                'position': self.dynamics_model.position.copy(),
                'orientation': self.dynamics_model.orientation,
                'velocity': self.dynamics_model.velocity_body.copy(),
                'angular_velocity': self.dynamics_model.angular_velocity
            }
            
            # Get boundary point positions
            pyb_point = self.get_boundary_point_world(
                t_param, pyb_state['position'], pyb_state['orientation']
            )
            pred_point = self.get_boundary_point_world(
                t_param, pred_state['position'], pred_state['orientation']
            )
            
            # Calculate boundary point velocities using rigid body kinematics
            pyb_point_vel = self._calculate_point_velocity(
                pyb_point, pyb_state['position'], pyb_state['velocity'],
                pyb_state['angular_velocity'], pyb_state['orientation']
            )
            pred_point_vel = self._calculate_point_velocity(
                pred_point, pred_state['position'], pred_state['velocity'],
                pred_state['angular_velocity'], pred_state['orientation']
            )
            
            # Store history
            self.history['times'].append(t)
            self.history['pyb_positions'].append(pyb_state['position'].copy())
            self.history['pyb_orientations'].append(pyb_state['orientation'])
            self.history['pyb_velocities'].append(pyb_state['velocity'].copy())
            self.history['pyb_angular_velocities'].append(pyb_state['angular_velocity'])
            self.history['pred_positions'].append(pred_state['position'].copy())
            self.history['pred_orientations'].append(pred_state['orientation'])
            self.history['pred_velocities'].append(pred_state['velocity'].copy())
            self.history['pred_angular_velocities'].append(pred_state['angular_velocity'])
            self.history['pyb_point_positions'].append(pyb_point.copy())
            self.history['pred_point_positions'].append(pred_point.copy())
            self.history['pyb_point_velocities'].append(pyb_point_vel.copy())
            self.history['pred_point_velocities'].append(pred_point_vel.copy())
            self.history['wrenches'].append(wrench.copy())
            
            # Apply wrench to PyBullet
            self.apply_wrench_pybullet(wrench)
            
            # Update prediction model using DummyBoundaryPointController
            # 1. Update controller with current state
            self.controller.update(pred_state, TIMESTEP)
            
            # 2. Get contact points and force magnitudes that produce the wrench
            contact_points, force_magnitudes = self.controller.get_control_actions()
            
            # 3. Update dynamics model with contact forces
            result = self.dynamics_model.update_state(
                contact_points, 
                force_magnitudes, 
                dt=TIMESTEP,
                friction_enabled=True
            )
            
            # 4. Post-update for controller history tracking
            self.controller.post_update(result)
            
            # Step PyBullet
            pyb.stepSimulation()
        
        # Calculate error metrics
        return self.calculate_errors()
    
    def _calculate_point_velocity(self, point_world, obj_position, obj_velocity_body, 
                                   obj_angular_velocity, obj_orientation):
        """Calculate velocity of a point on a rigid body.
        
        v_point = v_object_world + omega × r
        
        Parameters
        ----------
        point_world : np.ndarray
            Point position in world frame.
        obj_position : np.ndarray
            Object center position in world frame.
        obj_velocity_body : np.ndarray
            Object velocity in body frame.
        obj_angular_velocity : float
            Object angular velocity.
        obj_orientation : float
            Object orientation (radians).
        
        Returns
        -------
        np.ndarray
            Point velocity in world frame.
        """
        # Transform body velocity to world frame
        cos_t = np.cos(obj_orientation)
        sin_t = np.sin(obj_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        v_world = R @ obj_velocity_body
        
        # Vector from object center to point
        r = point_world - obj_position
        
        # Rotational velocity contribution: omega × r (in 2D: omega * [-r_y, r_x])
        v_rotation = obj_angular_velocity * np.array([-r[1], r[0]])
        
        return v_world + v_rotation
    
    def calculate_errors(self):
        """Calculate error metrics between prediction and PyBullet."""
        times = np.array(self.history['times'])
        
        # Position error
        pyb_pos = np.array(self.history['pyb_positions'])
        pred_pos = np.array(self.history['pred_positions'])
        pos_error = np.linalg.norm(pyb_pos - pred_pos, axis=1)
        
        # Orientation error
        pyb_orn = np.array(self.history['pyb_orientations'])
        pred_orn = np.array(self.history['pred_orientations'])
        orn_error = np.abs(np.arctan2(np.sin(pyb_orn - pred_orn), np.cos(pyb_orn - pred_orn)))
        
        # Velocity error
        pyb_vel = np.array(self.history['pyb_velocities'])
        pred_vel = np.array(self.history['pred_velocities'])
        vel_error = np.linalg.norm(pyb_vel - pred_vel, axis=1)
        
        # Boundary point position error
        pyb_point = np.array(self.history['pyb_point_positions'])
        pred_point = np.array(self.history['pred_point_positions'])
        point_pos_error = np.linalg.norm(pyb_point - pred_point, axis=1)
        
        # Boundary point velocity error
        pyb_point_vel = np.array(self.history['pyb_point_velocities'])
        pred_point_vel = np.array(self.history['pred_point_velocities'])
        point_vel_error = np.linalg.norm(pyb_point_vel - pred_point_vel, axis=1)
        
        errors = {
            'position_rmse': np.sqrt(np.mean(pos_error**2)),
            'position_max': np.max(pos_error),
            'orientation_rmse': np.sqrt(np.mean(orn_error**2)),
            'orientation_max': np.max(orn_error),
            'velocity_rmse': np.sqrt(np.mean(vel_error**2)),
            'velocity_max': np.max(vel_error),
            'point_position_rmse': np.sqrt(np.mean(point_pos_error**2)),
            'point_position_max': np.max(point_pos_error),
            'point_velocity_rmse': np.sqrt(np.mean(point_vel_error**2)),
            'point_velocity_max': np.max(point_vel_error),
        }
        
        return errors
    
    def plot_results(self, save_path=None, show=False):
        """Plot validation results.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot. If None, plot is not saved.
        show : bool
            If True, display plot interactively (requires GUI backend).
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        times = np.array(self.history['times'])
        
        if len(times) == 0:
            print("Warning: No data to plot")
            return None
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Physics Validation: {self.generic_obj.name}\n'
                    f'Mass={self.generic_obj.mass:.2f}kg, '
                    f'I={self.generic_obj.moment_of_inertia:.4f}kg⋅m², '
                    f'μ={self.generic_obj.lateral_friction:.2f}', fontsize=12)
        
        # Position X
        ax = axes[0, 0]
        pyb_pos = np.array(self.history['pyb_positions'])
        pred_pos = np.array(self.history['pred_positions'])
        ax.plot(times, pyb_pos[:, 0], 'b-', label='PyBullet', linewidth=2)
        ax.plot(times, pred_pos[:, 0], 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position X (m)')
        ax.set_title('Object Position X')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Position Y
        ax = axes[0, 1]
        ax.plot(times, pyb_pos[:, 1], 'b-', label='PyBullet', linewidth=2)
        ax.plot(times, pred_pos[:, 1], 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Y (m)')
        ax.set_title('Object Position Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Orientation
        ax = axes[0, 2]
        pyb_orn = np.array(self.history['pyb_orientations'])
        pred_orn = np.array(self.history['pred_orientations'])
        ax.plot(times, np.degrees(pyb_orn), 'b-', label='PyBullet', linewidth=2)
        ax.plot(times, np.degrees(pred_orn), 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Orientation (deg)')
        ax.set_title('Object Orientation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity X
        ax = axes[1, 0]
        pyb_vel = np.array(self.history['pyb_velocities'])
        pred_vel = np.array(self.history['pred_velocities'])
        ax.plot(times, pyb_vel[:, 0], 'b-', label='PyBullet', linewidth=2)
        ax.plot(times, pred_vel[:, 0], 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity X (m/s)')
        ax.set_title('Object Velocity X')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity Y
        ax = axes[1, 1]
        ax.plot(times, pyb_vel[:, 1], 'b-', label='PyBullet', linewidth=2)
        ax.plot(times, pred_vel[:, 1], 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Y (m/s)')
        ax.set_title('Object Velocity Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Angular Velocity
        ax = axes[1, 2]
        pyb_ang_vel = np.array(self.history['pyb_angular_velocities'])
        pred_ang_vel = np.array(self.history['pred_angular_velocities'])
        ax.plot(times, pyb_ang_vel, 'b-', label='PyBullet', linewidth=2)
        ax.plot(times, pred_ang_vel, 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Object Angular Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Boundary Point Position
        ax = axes[2, 0]
        pyb_point = np.array(self.history['pyb_point_positions'])
        pred_point = np.array(self.history['pred_point_positions'])
        ax.plot(pyb_point[:, 0], pyb_point[:, 1], 'b-', label='PyBullet', linewidth=2)
        ax.plot(pred_point[:, 0], pred_point[:, 1], 'r--', label='Predicted', linewidth=2)
        ax.plot(pyb_point[0, 0], pyb_point[0, 1], 'go', markersize=10, label='Start')
        ax.plot(pyb_point[-1, 0], pyb_point[-1, 1], 'bs', markersize=10, label='End (PyB)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Boundary Point Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Position Error
        ax = axes[2, 1]
        pos_error = np.linalg.norm(pyb_pos - pred_pos, axis=1)
        point_error = np.linalg.norm(pyb_point - pred_point, axis=1)
        ax.plot(times, pos_error * 100, 'b-', label='CoM Position', linewidth=2)
        ax.plot(times, point_error * 100, 'r-', label='Boundary Point', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (cm)')
        ax.set_title('Position Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity Error
        ax = axes[2, 2]
        vel_error = np.linalg.norm(pyb_vel - pred_vel, axis=1)
        pyb_point_vel = np.array(self.history['pyb_point_velocities'])
        pred_point_vel = np.array(self.history['pred_point_velocities'])
        point_vel_error = np.linalg.norm(pyb_point_vel - pred_point_vel, axis=1)
        ax.plot(times, vel_error * 100, 'b-', label='CoM Velocity', linewidth=2)
        ax.plot(times, point_vel_error * 100, 'r-', label='Boundary Point Vel', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (cm/s)')
        ax.set_title('Velocity Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            # Create directory if needed
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        # Show plot if requested (requires interactive backend)
        if show:
            try:
                plt.switch_backend('TkAgg')  # Switch to interactive backend
                plt.show()
            except Exception as e:
                print(f"Could not display plot interactively: {e}")
                print("Plot saved to file if save_path was provided.")
        
        return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Physics Validation - Compare object_utils.py dynamics vs PyBullet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run headless with default save path:
  python physics_validation.py --shape rectangle
  
  # Run with custom wrench and save plot:
  python physics_validation.py --shape circle --wrench 10 0 0.5 --save-plot /tmp/circle.png
  
  # Run with GUI (requires display):
  python physics_validation.py --shape triangle --gui
"""
    )
    parser.add_argument("--shape", default="rectangle", 
                       help="Shape to test (rectangle, circle, triangle, l_shape, etc.)")
    parser.add_argument("--duration", type=float, default=3.0, 
                       help="Simulation duration in seconds")
    parser.add_argument("--wrench", nargs=3, type=float, default=[5.0, 2.0, 0.5],
                       help="Wrench to apply [Fx, Fy, tau]")
    parser.add_argument("--t-param", type=float, default=0.25,
                       help="Boundary parameter for point tracking (0-1)")
    parser.add_argument("--gui", action="store_true", 
                       help="Run with GUI (default is headless mode)")
    parser.add_argument("--save-plot", type=str, default=None, 
                       help="Save plot to file. If not specified, saves to /tmp/physics_validation_<shape>.png")
    args = parser.parse_args()
    
    # Determine save path
    if args.save_plot is None:
        save_path = f"/tmp/physics_validation_{args.shape}.png"
    else:
        save_path = args.save_plot
    
    print("="*60)
    print("  PHYSICS VALIDATION")
    print("="*60)
    print(f"  Shape: {args.shape}")
    print(f"  Duration: {args.duration}s")
    print(f"  Wrench: {args.wrench}")
    print(f"  t_param: {args.t_param}")
    print(f"  Mode: {'GUI' if args.gui else 'Headless'}")
    print(f"  Save Plot: {save_path}")
    print("="*60)
    
    # Setup PyBullet (headless by default)
    print("\nInitializing PyBullet...")
    client_id, ground_uid = setup_pybullet(gui=args.gui)
    
    # Get the shape
    print(f"\nCreating {args.shape} object...")
    standard_objects = create_standard_objects()
    if args.shape not in standard_objects:
        print(f"Unknown shape: {args.shape}")
        print(f"Available: {list(standard_objects.keys())}")
        pyb.disconnect()
        return
    
    generic_obj = standard_objects[args.shape]
    
    # Create validator (will set ground friction from object's kinetic_friction)
    validator = PhysicsValidator(generic_obj, ground_uid=ground_uid, verbose=True)
    
    # Run validation
    errors = validator.run_validation(
        wrench=args.wrench,
        duration=args.duration,
        t_param=args.t_param
    )
    
    # Print errors
    print("\n" + "="*60)
    print("  VALIDATION RESULTS")
    print("="*60)
    print(f"  Position RMSE:       {errors['position_rmse']*100:.3f} cm")
    print(f"  Position Max Error:  {errors['position_max']*100:.3f} cm")
    print(f"  Orientation RMSE:    {np.degrees(errors['orientation_rmse']):.3f} deg")
    print(f"  Orientation Max:     {np.degrees(errors['orientation_max']):.3f} deg")
    print(f"  Velocity RMSE:       {errors['velocity_rmse']*100:.3f} cm/s")
    print(f"  Velocity Max Error:  {errors['velocity_max']*100:.3f} cm/s")
    print("-"*60)
    print(f"  Point Position RMSE: {errors['point_position_rmse']*100:.3f} cm")
    print(f"  Point Position Max:  {errors['point_position_max']*100:.3f} cm")
    print(f"  Point Velocity RMSE: {errors['point_velocity_rmse']*100:.3f} cm/s")
    print(f"  Point Velocity Max:  {errors['point_velocity_max']*100:.3f} cm/s")
    print("="*60)
    
    # Plot results (always save to file)
    validator.plot_results(save_path=save_path, show=args.gui)
    
    # Keep PyBullet open for inspection if GUI mode
    if args.gui:
        print("\nPress Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()

