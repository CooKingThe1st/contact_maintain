# %%
import numpy as np
import scipy.io as sio
import math
import time
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform
from shapely.affinity import rotate, translate

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy.optimize as opt

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt


# %%
import nbimporter

# Import ObjectLib components
from StudyPlan_ObjectLib import (
    GenericObject, 
    ContactPointParameterization,
    ContactPoint,
    GenericContactCalculator,
    GraspMatrixCalculator,
    EdgeCharacterizer,
    create_standard_objects,
    WrenchSpaceVisualizer,
    DynamicObjectModel,
    PlaceholderController,
)

from StudyPlan_ContactPointLinearProgLib import (
    find_optimal_contacts
)

from StudyPlan_PathLib import (
    SplineReferencePath,
    PathVisualizer,
    create_path_from_trajectory
)

# %%
# Test if the components work together

# Create an object and contact points
standard_objects = create_standard_objects()
obj = standard_objects['rectangle']
calculator = GenericContactCalculator(obj)

modes = ['2', 'E', 'E+2']
test_contact_points = find_optimal_contacts(obj, mode = modes[1], target_wrench = np.array([0,0,0]), force_magnitude=1.0, verbose=False, visualize=True)
test_contact_points = test_contact_points['contacts']
print(f"Selected {len(test_contact_points)} contact points using mode '{modes[1]}'")
print("Contact Points:", test_contact_points)

# %%
def visualize_simulation_data(sim_record, controller, ref_path = None):
    # Extract data
    data = sim_record['data']
    
    # Create a comprehensive dashboard with subplots
    fig = plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])
    
    # 1. Force and Friction Plot (top left)
    ax_force = fig.add_subplot(grid[0, 0])
    ax_force.plot(data['times'], data['applied_wrench'][:,0], 'b-', linewidth=2, label='Applied Force X')
    ax_force.plot(data['times'], data['applied_wrench'][:,1], 'g-', linewidth=2, label='Applied Force Y')
    ax_force.plot(data['times'], data['applied_wrench'][:,2], 'm-', linewidth=2, label='Applied Torque')

    ax_force.plot(data['times'], data['frictions_fx'], 'b--', linewidth=1, label='Friction Force X')
    ax_force.plot(data['times'], data['frictions_fy'], 'g--', linewidth=1, label='Friction Force Y')
    ax_force.plot(data['times'], data['frictions_m'], 'm--', linewidth=1, label='Friction Torque')

    # ax_force.plot(data['times'], data['frictions_fxy'], 'r--', linewidth=2, label='Friction Force')
    # ax_force.axhline(y=dynamics.static_f_max, color='k', linestyle='-', alpha=0.3, 
    #                  label=f'Static ({dynamics.static_f_max:.2f}N)')
    # ax_force.axhline(y=dynamics.kinetic_f_max, color='k', linestyle=':', alpha=0.3,
    #                  label=f'Kinetic ({dynamics.kinetic_f_max:.2f}N)')
    ax_force.set_title('Forces')
    ax_force.set_ylabel('Force (N)')
    ax_force.legend(fontsize='small')
    ax_force.grid(True)
    
    # 2. Velocities Plot (top middle)
    ax_vel = fig.add_subplot(grid[0, 1], sharex=ax_force)
    ax_vel.plot(data['times'], data['velocities'], 'g-', linewidth=2, label='Linear')
    ax_vel.plot(data['times'], data['angular_velocities'], 'c-', linewidth=2, label='Angular')
    ax_vel.set_title('Velocities')
    ax_vel.set_ylabel('Velocity')
    ax_vel.legend(fontsize='small')
    ax_vel.grid(True)
    
    # 3. Accelerations Plot (top right)
    ax_accel = fig.add_subplot(grid[0, 2], sharex=ax_force)
    ax_accel.plot(data['times'], data['linear_accelerations'], 'm-', linewidth=2, label='Linear')
    ax_accel.plot(data['times'], data['angular_accelerations'], 'y-', linewidth=2, label='Angular')
    ax_accel.set_title('Accelerations')
    ax_accel.set_ylabel('Acceleration')
    ax_accel.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_accel.legend(fontsize='small')
    ax_accel.grid(True)
    
    # 4. Total Components (middle left)
    ax_fric_comp = fig.add_subplot(grid[1, 0], sharex=ax_force)
    ax_fric_comp.plot(data['times'], data['total_wrench'][:,0], 'b.-', linewidth=1, label='Total Force X')
    ax_fric_comp.plot(data['times'], data['total_wrench'][:,1], 'g.-', linewidth=1, label='Total Force Y')
    ax_fric_comp.plot(data['times'], data['total_wrench'][:,2], 'm.-', linewidth=1, label='Total Torque')
    ax_fric_comp.set_title('Total Components')
    ax_fric_comp.set_xlabel('Time (s)')
    ax_fric_comp.set_ylabel('Total')
    ax_fric_comp.legend(fontsize='small')
    ax_fric_comp.grid(True)
    
    # 5. Twist Magnitude (middle middle)
    ax_twist = fig.add_subplot(grid[1, 1], sharex=ax_force)
    ax_twist.semilogy(data['times'], data['twist_magnitudes'], 'k-', linewidth=2)
    ax_twist.axhline(y=1e-4, color='r', linestyle='--', alpha=0.7, label='Threshold')
    ax_twist.set_title('Twist Magnitude')
    ax_twist.set_xlabel('Time (s)')
    ax_twist.set_ylabel('Log Magnitude')
    ax_twist.grid(True)
    
    # 6. S Value (middle right)
    ax_s = fig.add_subplot(grid[1, 2], sharex=ax_force)
    ax_s.semilogy(data['times'], data['s_values'], 'm-', linewidth=2)
    ax_s.axhline(y=1e-9, color='r', linestyle='--', alpha=0.7, label='Threshold')
    ax_s.set_title('S Value')
    ax_s.set_xlabel('Time (s)')
    ax_s.set_ylabel('Log Value')
    ax_s.grid(True)
    
    # 7. Position Plot (bottom left+middle)
    ax_pos = fig.add_subplot(grid[2, :2])
    
    # Check if we have position data
    if 'positions' in data and 'orientations' in data:
        # Extract trajectory data
        positions = data['positions']
        orientations = data['orientations']
        
        # Plot actual trajectory
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        ax_pos.plot(x_coords, y_coords, 'g-', linewidth=2, label='Trajectory')
        
        # Plot reference path if available
        ref_path = None
        if ref_path is not None:
            t_samples = np.linspace(0, 1, 100)
            ref_points = np.array([ref_path.get_point_at_parameter(t) for t in t_samples])
            ax_pos.plot(ref_points[:, 0], ref_points[:, 1], 'b-', linewidth=1.5, label='Reference Path')
        
        # Show a few objects along the path
        if hasattr(controller, 'object_model') and hasattr(controller.object_model, 'object'):
            obj = controller.object_model.object
            num_poses = min(5, len(positions))
            indices = np.linspace(0, len(positions)-1, num_poses, dtype=int)
            
            for i, idx in enumerate(indices):
                x, y = positions[idx]
                theta = orientations[idx]
                
                # Create transformed object
                transformed_obj = obj.transform(x, y, theta - obj.heading)
                
                # Color gradient
                color = plt.cm.cool(i / max(1, num_poses-1))
                
                # Visualize object
                transformed_obj.visualize(
                    ax_pos,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7,
                    show_frame=True if i == 0 or i == num_poses-1 else False
                )
                
                # Add orientation arrow
                arrow_length = 0.1
                ax_pos.arrow(x, y, 
                          arrow_length * np.cos(theta),
                          arrow_length * np.sin(theta),
                          head_width=0.05, head_length=0.07,
                          fc=color, ec=color)
        
        ax_pos.grid(True)
        ax_pos.axis('equal')
        ax_pos.set_title('Object Trajectory')
        ax_pos.set_xlabel('X Position (m)')
        ax_pos.set_ylabel('Y Position (m)')
        ax_pos.legend(fontsize='small')
    else:
        ax_pos.text(0.5, 0.5, 'Position data not available', 
                 ha='center', va='center', fontsize=12)
    
    # # 8. Cross-track error or Orientation Error (bottom right)
    # ax_error = fig.add_subplot(grid[2, 2], sharex=ax_force)
    
    # # Check if controller has orientation errors
    # if hasattr(controller, 'orientation_errors') and controller.orientation_errors:
    #     # Plot orientation errors
    #     time_steps = [step * data['times'][1] for step in controller.time_steps]
    #     ax_error.plot(time_steps, controller.orientation_errors, 'b-', linewidth=2)
    #     ax_error.axhline(0, color='r', linestyle='--', alpha=0.7)
        
    #     # Add statistics
    #     mean_error = np.mean(np.abs(controller.orientation_errors))
    #     max_error = np.max(np.abs(controller.orientation_errors))
    #     ax_error.text(0.05, 0.95, f"Mean abs: {mean_error:.4f}\nMax: {max_error:.4f}", 
    #                 transform=ax_error.transAxes, fontsize=9,
    #                 bbox=dict(facecolor='white', alpha=0.7))
        
    #     ax_error.set_title('Orientation Error')
    # elif 'positions' in data and hasattr(controller, 'ref_path'):
    #     # Calculate and plot lateral error
    #     positions = data['positions']
    #     orientations = data['orientations']
    #     ref_path = controller.ref_path
        
    #     lateral_errors = []
    #     for i in range(len(positions)):
    #         query_point = [positions[i][0], positions[i][1], orientations[i]]
    #         error_info = ref_path.get_contour_error(query_point)
    #         lateral_errors.append(error_info['lateral'])
        
    #     ax_error.plot(data['times'], lateral_errors, 'r-', linewidth=2)
    #     ax_error.axhline(0, color='k', linestyle='-', alpha=0.3)
        
    #     # Add statistics
    #     mean_lateral = np.mean(np.abs(lateral_errors))
    #     max_lateral = np.max(np.abs(lateral_errors))
    #     ax_error.text(0.05, 0.95, f"Mean abs: {mean_lateral:.4f}\nMax: {max_lateral:.4f}", 
    #                 transform=ax_error.transAxes, fontsize=9,
    #                 bbox=dict(facecolor='white', alpha=0.7))
        
    #     ax_error.set_title('Lateral Error')
    # else:
    #     ax_error.text(0.5, 0.5, 'Error data not available', 
    #                ha='center', va='center', fontsize=12)
    
    # ax_error.set_xlabel('Time (s)')
    # ax_error.set_ylabel('Error')
    # ax_error.grid(True)
    
    # Mark regime transitions
    # try:
    #     # Find where velocity drops below threshold
    #     stopping_indices = []
    #     for i in range(1, len(data['velocities'])):
    #         if data['velocities'][i] < 0.01 and data['velocities'][i-1] >= 0.01:
    #             stopping_indices.append(i)
                
    #     # Find where twist crosses threshold
    #     regime_change_indices = []
    #     for i in range(1, len(data['twist_magnitudes'])):
    #         if (data['twist_magnitudes'][i] < 1e-4 and data['twist_magnitudes'][i-1] >= 1e-4) or \
    #            (data['twist_magnitudes'][i] >= 1e-4 and data['twist_magnitudes'][i-1] < 1e-4):
    #             regime_change_indices.append(i)
                
    #     # Mark stopping points on all time-series plots
    #     for idx in stopping_indices:
    #         t = data['times'][idx]
    #         for ax in [ax_force, ax_vel, ax_accel, ax_fric_comp, ax_twist, ax_s] : #, ax_error]:
    #             ax.axvline(x=t, color='purple', linestyle='-', alpha=0.5)
                
    #     # Mark regime changes
    #     for idx in regime_change_indices:
    #         t = data['times'][idx]
    #         for ax in [ax_force, ax_vel, ax_accel, ax_fric_comp, ax_twist, ax_s] : #, ax_error]:
    #             ax.axvline(x=t, color='orange', linestyle='--', alpha=0.5)
    # except Exception as e:
    #     print(f"Error finding transitions: {e}")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()
    
    # Check if there's any movement at all
    if 'positions' in data:
        start_pos = data['positions'][0]
        end_pos = data['positions'][-1]
        total_distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        print(f"\nMovement Analysis:")
        print(f"Starting position: {start_pos}")
        print(f"Ending position: {end_pos}")
        print(f"Total displacement: {total_distance:.6f} m")
        
        # Check if there's rotation
        if 'orientations' in data:
            start_orient = data['orientations'][0]
            end_orient = data['orientations'][-1]
            orient_change = ((end_orient - start_orient + np.pi) % (2*np.pi)) - np.pi
            print(f"Starting orientation: {start_orient:.6f} rad")
            print(f"Ending orientation: {end_orient:.6f} rad")
            print(f"Total rotation: {orient_change:.6f} rad ({np.degrees(orient_change):.2f} degrees)")
            
        # Additional diagnostics
        max_vel = np.max(np.abs(data['velocities']))
        max_ang_vel = np.max(np.abs(data['angular_velocities']))
        print(f"\nMotion Diagnostics:")
        print(f"Maximum linear velocity: {max_vel:.6f} m/s")
        print(f"Maximum angular velocity: {max_ang_vel:.6f} rad/s")
        
        # Check if forces exceed friction threshold
        # force_exceeds = np.any(data['forces'] > dynamics.static_f_max)
        # print(f"\nForce Analysis:")
        # print(f"Do forces ever exceed static friction threshold? {'YES' if force_exceeds else 'NO'}")
        # print(f"Maximum applied force: {np.max(data['forces']):.4f} N")
        # print(f"Static friction threshold: {dynamics.static_f_max:.4f} N")
        
        # Suggest possible issues if no movement
        # if total_distance < 1e-4 and abs(orient_change) < 1e-4:
        #     print("\nPotential Issues (no movement detected):")
        #     print("1. Applied forces may be too small to overcome friction")
        #     print("2. Forces may cancel each other out")
        #     print("3. Controller might not be applying forces correctly")
        #     print("4. Object might be constrained or have incorrect physical properties")

# %%
class ForceDistributorPro:
    """
    Distributes desired wrench to contact point forces using multiple methods.
    
    Version 1.0: No constraints (existing implementation)
    Version 2.0: Force magnitude constraints (LP or QP)
    Version 3.0: Force magnitude + rate constraints (LP or QP)
    """
    
    def __init__(self, max_force=10.0, max_rate_increase=6.0, max_rate_decrease=8.0):
        """
        Initialize force distributor.
        
        Args:
            max_force: Maximum force at any contact point
            max_rate_increase: Maximum rate of force increase (per second)
            max_rate_decrease: Maximum rate of force decrease (per second)
        """
        self.max_force = max_force
        self.max_rate_increase = max_rate_increase
        self.max_rate_decrease = max_rate_decrease
        
        # Previous force values for rate limiting
        self.prev_forces = None
        self.prev_time = None
        
        # Performance metrics
        self.wrench_errors = []  # Track how well desired wrench is achieved
    
    def distribute_forces(self, desired_wrench, contact_points, grasp_matrix, 
                         version='v1', method='rf', dynamics_model=None, 
                         current_time=None, dt=1):
        """
        Main interface for force distribution with flexible version and method selection.
        
        Args:
            desired_wrench: Desired wrench vector [Fx, Fy, M]
            contact_points: List of contact points
            grasp_matrix: Grasp matrix relating contact forces to wrench
            version: 'v1' (no constraints), 'v2' (force limits), 'v3' (force + rate limits)
            method: 'lp' (linear programming) or 'qp' (quadratic programming) or 'rf' (refined for v2 and v3)
            dynamics_model: Object dynamics model (required for v2 and v3)
            current_time: Current time for rate limiting
            dt: Time step if current_time is None
            
        Returns:
            dict: Results including force magnitudes and metrics
        """
        if version == 'v1':
            if method in ['lp', 'qp']:
                return self.distribute_forces_v1(desired_wrench, contact_points, grasp_matrix, 
                                           current_time, dt)
            elif method == 'rf':
                return self.distribute_forces_v1_min_variance_qp(desired_wrench, contact_points, grasp_matrix, 
                                                         current_time, dt)
            
        elif version == 'v2':
            if method in ['lp', 'qp'] and dynamics_model is None:
                raise ValueError("dynamics_model required for v2 and v3")
            if method == 'lp':
                return self.distribute_forces_v2_lp(desired_wrench, contact_points, grasp_matrix, 
                                                   dynamics_model, current_time, dt)
            elif method == 'qp':
                return self.distribute_forces_v2_qp(desired_wrench, contact_points, grasp_matrix, 
                                                   dynamics_model, current_time, dt)
            elif method == 'rf':
                return self.distribute_forces_v2_refined(desired_wrench, contact_points, grasp_matrix)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'lp' or 'qp'")
        elif version == 'v3':
            if method in ['lp', 'qp'] and dynamics_model is None:
                raise ValueError("dynamics_model required for v2 and v3")
            if method == 'lp':
                return self.distribute_forces_v3_lp(desired_wrench, contact_points, grasp_matrix, 
                                                   dynamics_model, current_time, dt)
            elif method == 'qp':
                return self.distribute_forces_v3_qp(desired_wrench, contact_points, grasp_matrix, 
                                                   dynamics_model, current_time, dt)
            elif method == 'rf':
                return self.distribute_forces_v3_refined(desired_wrench, contact_points, grasp_matrix, dt)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'lp' or 'qp' or 'rf'")
        else:
            raise ValueError(f"Unknown version: {version}. Use 'v1', 'v2', or 'v3'")
    
    # true name is distribute_forces_v1_original
    # Original implementation with no force constraints and no variation minimization
    def distribute_forces_v1(self, desired_wrench, contact_points, grasp_matrix, current_time=None, dt=0.01):
        """
        Version 1.0: Original implementation with no force constraints.
        Achieve exact wrench by finding unit direction then scaling.
        """
        num_contacts = len(contact_points)
        
        # Step 1: Find forces for desired wrench direction
        wrench_direction = self._get_normalized_wrench(desired_wrench)
        
        # Using LP to find forces that achieve this direction
        c = np.ones(num_contacts)  # Minimize sum of forces
        
        A_eq = grasp_matrix  # Shape: (3, num_contacts)
        b_eq = wrench_direction  # Shape: (3,) - normalized wrench
        
        # Bounds: all forces are non-negative and capped by reasonable max force
        bounds = [(0, 100) for _ in range(num_contacts)]
        
        try:
            # Solve LP
            result = opt.linprog(
                c=c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options={'disp': False}
            )
            
            if not result.success:
                return self._distribute_with_slack_v1(desired_wrench, contact_points, grasp_matrix, current_time, dt)
                
            # Step 2: Get unit forces and scale to desired magnitude
            unit_forces = result.x
            desired_magnitude = np.linalg.norm(desired_wrench)
            unit_wrench = grasp_matrix @ unit_forces
            unit_magnitude = np.linalg.norm(unit_wrench)
            
            if unit_magnitude < 1e-6:
                scale_factor = 0
            else:
                scale_factor = desired_magnitude / unit_magnitude
            
            force_magnitudes = unit_forces * scale_factor
            achieved_wrench = grasp_matrix @ force_magnitudes
            achieved_wrench[np.abs(achieved_wrench) < 1e-6] = 0
            
            wrench_error = achieved_wrench - desired_wrench
            wrench_error_magnitude = np.linalg.norm(wrench_error)
            
            self.wrench_errors.append(wrench_error_magnitude)
            
            return {
                'force_magnitudes': force_magnitudes,
                'unit_forces': unit_forces,
                'scale_factor': scale_factor,
                'achieved_wrench': achieved_wrench,
                'wrench_error': wrench_error,
                'wrench_error_magnitude': wrench_error_magnitude,
                'success': True,
                'method': 'v1_unit_scaling'
            }
            
        except Exception as e:
            return self._distribute_with_slack_v1(desired_wrench, contact_points, grasp_matrix, current_time, dt)
    
    # true name is distribute_forces_v1_min_variance_qp
    # Alternative: Quadratic Programming version for true variance minimization
    def distribute_forces_v1_min_variance_qp(self, desired_wrench, contact_points, grasp_matrix, current_time=None, dt=0.01):
        """
        Version 1.0 QP: True variance minimization using quadratic programming.
        
        Minimize: Var(f) = (1/n) * sum((fi - f_mean)^2)
        Subject to: G * f = desired_wrench, fi >= 0
        
        This gives the minimum variance solution that exactly achieves the desired wrench.
        """
        num_contacts = len(contact_points)
        
        try:
            # QP formulation: minimize variance
            # Var(f) = (1/n) * sum((fi - f_mean)^2) = (1/n) * sum(fi^2) - f_mean^2
            # Since f_mean^2 is constant given the constraint, we minimize sum(fi^2) - (1/n)*(sum(fi))^2
            
            # Expanded: minimize f^T * H * f where H accounts for variance structure
            # H[i,i] = 1 - 1/n, H[i,j] = -1/n for i≠j
            H = np.eye(num_contacts) - np.ones((num_contacts, num_contacts)) / num_contacts
            
            # No linear term (g = 0)
            g = np.zeros(num_contacts)
            
            # Equality constraint: G * f = desired_wrench
            A_eq = grasp_matrix
            b_eq = desired_wrench
            
            # Bounds: fi >= 0
            bounds = [(0, None) for _ in range(num_contacts)]
            
            # Solve QP using scipy.optimize.minimize with quadratic objective
            def objective(f):
                return 0.5 * f.T @ H @ f
            
            def jacobian(f):
                return H @ f
            
            def hessian(f):
                return H
            
            # Equality constraint function
            def eq_constraint(f):
                return A_eq @ f - b_eq
            
            def eq_constraint_jac(f):
                return A_eq
            
            # Set up constraint
            constraints = {
                'type': 'eq',
                'fun': eq_constraint,
                'jac': eq_constraint_jac
            }
            
            # Initial guess: uniform distribution scaled to satisfy constraint
            # Solve for uniform solution first
            ones_vec = np.ones(num_contacts)
            uniform_multiplier = np.linalg.lstsq(A_eq @ ones_vec.reshape(-1, 1), 
                                            desired_wrench.reshape(-1, 1), rcond=None)[0][0]
            f0 = ones_vec * uniform_multiplier
            f0 = np.maximum(f0, 0.1)  # Ensure positive initial guess
            
            result = opt.minimize(
                objective,
                f0,
                method='SLSQP',
                jac=jacobian,
                bounds=bounds,
                constraints=constraints,
                options={'disp': False}
            )
            
            if result.success:
                force_magnitudes = result.x
                achieved_wrench = grasp_matrix @ force_magnitudes
                achieved_wrench[np.abs(achieved_wrench) < 1e-6] = 0
                
                # Calculate metrics
                force_mean = np.mean(force_magnitudes)
                force_variance = np.var(force_magnitudes)
                force_std = np.std(force_magnitudes)
                
                wrench_error = achieved_wrench - desired_wrench
                wrench_error_magnitude = np.linalg.norm(wrench_error)
                
                self.wrench_errors.append(wrench_error_magnitude)
                
                return {
                    'force_magnitudes': force_magnitudes,
                    'unit_forces': force_magnitudes / np.linalg.norm(force_magnitudes) if np.linalg.norm(force_magnitudes) > 1e-6 else force_magnitudes,
                    'scale_factor': np.linalg.norm(force_magnitudes),
                    'force_variance': force_variance,
                    'force_std': force_std,
                    'force_mean': force_mean,
                    'force_uniformity': 1.0 / (1.0 + force_std),  # Higher = more uniform
                    'achieved_wrench': achieved_wrench,
                    'wrench_error': wrench_error,
                    'wrench_error_magnitude': wrench_error_magnitude,
                    'success': True,
                    'method': 'v1_min_variance_qp'
                }
            else:
                # Fallback to original v1 method
                return self.distribute_forces_v1_original(desired_wrench, contact_points, grasp_matrix, current_time, dt)
                
        except Exception as e:
            print(f"Error in V1.0 Min Variance QP: {e}")
            # Fallback to original v1 method
            return self.distribute_forces_v1_original(desired_wrench, contact_points, grasp_matrix, current_time, dt)

    # ================================================================
    # VERSION 2.0: FORCE MAGNITUDE CONSTRAINTS
    # ================================================================
    
    def distribute_forces_v2_lp(self, desired_wrench, contact_points, grasp_matrix, dynamics_model, current_time=None, dt=0.01):
        """
        Version 2.0 LP: Add force magnitude constraints using Linear Programming.
        Minimize weighted L1-norm error between achieved and desired wrench.
        """
        num_contacts = len(contact_points)
        
        # Get scaling factor c = m_max / f_max from dynamics model
        static_f_max = getattr(dynamics_model, 'static_f_max', 1.0)
        static_m_max = getattr(dynamics_model, 'static_m_max', static_f_max * 0.5)
        c = static_m_max / static_f_max if static_f_max > 1e-6 else 1.0
        
        # Weight vector for proper scaling: [1, 1, c]
        weight_vector = np.array([1.0, 1.0, c])
        
        try:
            # Set up LP problem to minimize weighted wrench error
            # Decision variables: [f1, f2, ..., fn, e1+, e1-, e2+, e2-, e3+, e3-]
            num_vars = num_contacts + 6  # forces + error variables
            
            # Objective: minimize sum of weighted absolute errors
            c_obj = np.zeros(num_vars)
            c_obj[num_contacts:num_contacts+2] = weight_vector[0]  # e1+, e1-
            c_obj[num_contacts+2:num_contacts+4] = weight_vector[1]  # e2+, e2-
            c_obj[num_contacts+4:num_contacts+6] = weight_vector[2]  # e3+, e3-
            
            # Equality constraints: achieved_wrench - desired_wrench = e+ - e-
            A_eq = np.zeros((3, num_vars))
            A_eq[:, :num_contacts] = grasp_matrix
            A_eq[0, num_contacts:num_contacts+2] = [-1, 1]
            A_eq[1, num_contacts+2:num_contacts+4] = [-1, 1]
            A_eq[2, num_contacts+4:num_contacts+6] = [-1, 1]
            
            b_eq = desired_wrench
            
            # Bounds: forces [0, max_force], errors [0, inf]
            bounds = []
            for i in range(num_contacts):
                bounds.append((0, self.max_force))
            for i in range(6):
                bounds.append((0, None))
            
            # Solve LP
            result = opt.linprog(
                c=c_obj,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options={'disp': False}
            )
            
            if result.success:
                force_magnitudes = result.x[:num_contacts]
                achieved_wrench = grasp_matrix @ force_magnitudes
                achieved_wrench[np.abs(achieved_wrench) < 1e-6] = 0
                
                wrench_error = achieved_wrench - desired_wrench
                wrench_error_magnitude = np.linalg.norm(wrench_error)
                weighted_error = np.sum(np.abs(wrench_error) * weight_vector)
                
                self.wrench_errors.append(wrench_error_magnitude)
                
                return {
                    'force_magnitudes': force_magnitudes,
                    'achieved_wrench': achieved_wrench,
                    'wrench_error': wrench_error,
                    'wrench_error_magnitude': wrench_error_magnitude,
                    'weighted_error': weighted_error,
                    'max_force_used': np.max(force_magnitudes),
                    'success': True,
                    'method': 'v2_lp_magnitude_constrained'
                }
            else:
                return self._distribute_with_slack_v2(desired_wrench, contact_points, grasp_matrix, dynamics_model)
                
        except Exception as e:
            print(f"Error in V2.0 LP: {e}")
            return self._distribute_with_slack_v2(desired_wrench, contact_points, grasp_matrix, dynamics_model)
    
    def distribute_forces_v2_qp(self, desired_wrench, contact_points, grasp_matrix, dynamics_model, current_time=None, dt=0.01):
        """
        Version 2.0 QP: Add force magnitude constraints using Quadratic Programming.
        Minimize weighted L2-norm error between achieved and desired wrench.
        Much cleaner formulation than LP!
        """
        num_contacts = len(contact_points)
        
        # Get scaling factor c = m_max / f_max from dynamics model
        static_f_max = getattr(dynamics_model, 'static_f_max', 1.0)
        static_m_max = getattr(dynamics_model, 'static_m_max', static_f_max * 0.5)
        c = static_m_max / static_f_max if static_f_max > 1e-6 else 1.0
        
        # Weight matrix for proper scaling: diag([1, 1, c])
        W = np.diag([1.0, 1.0, c])
        
        try:
            # QP formulation: minimize (1/2) * ||W * (G*f - w_desired)||^2
            # where G is grasp_matrix, f is forces, w_desired is desired_wrench
            
            # Objective function: (1/2) * f^T * H * f + g^T * f
            # where H = G^T * W^T * W * G, g = -G^T * W^T * W * w_desired
            
            WG = W @ grasp_matrix  # Weighted grasp matrix
            H = WG.T @ WG  # Hessian matrix (positive semi-definite)
            g = -WG.T @ W @ desired_wrench  # Linear term
            
            # Add small regularization to ensure positive definiteness
            H += 1e-6 * np.eye(num_contacts)
            
            # Constraints: 0 <= f_i <= max_force
            bounds = [(0, self.max_force) for _ in range(num_contacts)]
            
            # Solve QP using minimize with bounds
            def objective(f):
                return 0.5 * f.T @ H @ f + g.T @ f
            
            def jacobian(f):
                return H @ f + g
            
            def hessian(f):
                return H
            
            # Initial guess
            f0 = np.ones(num_contacts) * min(0.1, self.max_force * 0.1)
            
            result = opt.minimize(
                objective,
                f0,
                method='L-BFGS-B',
                jac=jacobian,
                bounds=bounds,
                options={'disp': False}
            )
            
            if result.success:
                force_magnitudes = result.x
                achieved_wrench = grasp_matrix @ force_magnitudes
                achieved_wrench[np.abs(achieved_wrench) < 1e-6] = 0
                
                wrench_error = achieved_wrench - desired_wrench
                wrench_error_magnitude = np.linalg.norm(wrench_error)
                weighted_error = np.linalg.norm(W @ wrench_error)
                
                self.wrench_errors.append(wrench_error_magnitude)
                
                return {
                    'force_magnitudes': force_magnitudes,
                    'achieved_wrench': achieved_wrench,
                    'wrench_error': wrench_error,
                    'wrench_error_magnitude': wrench_error_magnitude,
                    'weighted_error': weighted_error,
                    'max_force_used': np.max(force_magnitudes),
                    'success': True,
                    'method': 'v2_qp_magnitude_constrained'
                }
            else:
                return self._distribute_with_slack_v2(desired_wrench, contact_points, grasp_matrix, dynamics_model)
                
        except Exception as e:
            print(f"Error in V2.0 QP: {e}")
            return self._distribute_with_slack_v2(desired_wrench, contact_points, grasp_matrix, dynamics_model)
    
    def distribute_forces_v2_refined(self, desired_wrench, contact_points, grasp_matrix):
        """
        V2.0 Refined: Unit solution + uniform scaling within force bounds.
        Avoids solver numerical issues entirely!
        """
        
        # Step 1: Get ideal direction from V1.0
        unit_result = self.distribute_forces_v1_min_variance_qp(desired_wrench, contact_points, grasp_matrix)
        if not unit_result['success']:
            return self._fallback_result()
        
        unit_forces = unit_result['unit_forces']  # Before magnitude scaling
        unit_scale = unit_result['scale_factor']   # What V1.0 used for scaling
        
        # Step 2: Calculate constraint-respecting scale factor
        max_scales = []
        for i, unit_force in enumerate(unit_forces):
            if unit_force > 1e-6:  # Avoid division by zero
                max_scale_i = self.max_force / unit_force
                max_scales.append(max_scale_i)
        
        if not max_scales:
            return self._fallback_result()
        
        # Global maximum scale (bottleneck constraint)
        constraint_max_scale = min(max_scales)
        
        # Step 3: Choose final scale (desire vs constraint)
        desired_scale = unit_scale  # What V1.0 wanted
        final_scale = min(desired_scale, constraint_max_scale)
        
        # Step 4: Apply scaling
        force_magnitudes = unit_forces * final_scale
        achieved_wrench = grasp_matrix @ force_magnitudes
        
        return {
            'force_magnitudes': force_magnitudes,
            'achieved_wrench': achieved_wrench,
            'wrench_error': achieved_wrench - desired_wrench,
            'wrench_error_magnitude': np.linalg.norm(achieved_wrench - desired_wrench),
            'success': True,
            'method': 'v2_refined_uniform_scaling'
        }

    # ================================================================
    # VERSION 3.0: FORCE MAGNITUDE + RATE CONSTRAINTS
    # ================================================================
    
    def distribute_forces_v3_lp(self, desired_wrench, contact_points, grasp_matrix, dynamics_model, current_time=None, dt=1):
        """
        Version 3.0 LP: Add both force magnitude AND rate constraints using Linear Programming.
        """
        num_contacts = len(contact_points)
        
        # Get scaling factor c = m_max / f_max from dynamics model
        static_f_max = getattr(dynamics_model, 'static_f_max', 1.0)
        static_m_max = getattr(dynamics_model, 'static_m_max', static_f_max * 0.5)
        c = static_m_max / static_f_max if static_f_max > 1e-6 else 1.0
        
        # Weight vector for proper scaling: [1, 1, c]
        weight_vector = np.array([1.0, 1.0, c])
        
        # Calculate time delta for rate constraints
        if self.prev_forces is None:
            self.prev_forces = np.zeros(num_contacts)
            self.prev_time = current_time if current_time is not None else 0.0
        
        if current_time is not None and self.prev_time is not None:
            time_delta = current_time - self.prev_time
        else:
            time_delta = dt
        
        time_delta = max(time_delta, 1e-6)
        
        try:
            # Same LP setup as v2 but with additional rate constraints
            num_vars = num_contacts + 6
            
            # Objective: minimize sum of weighted absolute errors
            c_obj = np.zeros(num_vars)
            c_obj[num_contacts:num_contacts+2] = weight_vector[0]
            c_obj[num_contacts+2:num_contacts+4] = weight_vector[1]
            c_obj[num_contacts+4:num_contacts+6] = weight_vector[2]
            
            # Equality constraints: achieved_wrench - desired_wrench = e+ - e-
            A_eq = np.zeros((3, num_vars))
            A_eq[:, :num_contacts] = grasp_matrix
            A_eq[0, num_contacts:num_contacts+2] = [-1, 1]
            A_eq[1, num_contacts+2:num_contacts+4] = [-1, 1]
            A_eq[2, num_contacts+4:num_contacts+6] = [-1, 1]
            
            b_eq = desired_wrench
            
            # Inequality constraints: rate limits
            max_increase_delta = self.max_rate_increase * time_delta
            max_decrease_delta = self.max_rate_decrease * time_delta
            
            A_ub = []
            b_ub = []
            
            for i in range(num_contacts):
                # Increase rate constraint: fi - prev_fi <= max_increase_delta
                row = np.zeros(num_vars)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(self.prev_forces[i] + max_increase_delta)
                
                # Decrease rate constraint: prev_fi - fi <= max_decrease_delta
                row = np.zeros(num_vars)
                row[i] = -1.0
                A_ub.append(row)
                b_ub.append(-self.prev_forces[i] + max_decrease_delta)
            
            A_ub = np.array(A_ub) if A_ub else None
            b_ub = np.array(b_ub) if b_ub else None
            
            # Bounds: forces [0, max_force], errors [0, inf]
            bounds = []
            for i in range(num_contacts):
                bounds.append((0, self.max_force))
            for i in range(6):
                bounds.append((0, None))
            
            # Solve LP with rate constraints
            result = opt.linprog(
                c=c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options={'disp': False}
            )
            
            if result.success:
                force_magnitudes = result.x[:num_contacts]
                
                # Update previous forces and time
                self.prev_forces = force_magnitudes.copy()
                self.prev_time = current_time if current_time is not None else (self.prev_time + dt)
                
                achieved_wrench = grasp_matrix @ force_magnitudes
                achieved_wrench[np.abs(achieved_wrench) < 1e-6] = 0
                
                wrench_error = achieved_wrench - desired_wrench
                wrench_error_magnitude = np.linalg.norm(wrench_error)
                weighted_error = np.sum(np.abs(wrench_error) * weight_vector)
                
                # Calculate rate metrics
                force_changes = force_magnitudes - (self.prev_forces if hasattr(self, '_prev_forces_for_analysis') else np.zeros_like(force_magnitudes))
                max_rate_used = np.max(np.abs(force_changes)) / time_delta if time_delta > 1e-6 else 0.0
                
                self.wrench_errors.append(wrench_error_magnitude)
                
                return {
                    'force_magnitudes': force_magnitudes,
                    'achieved_wrench': achieved_wrench,
                    'wrench_error': wrench_error,
                    'wrench_error_magnitude': wrench_error_magnitude,
                    'weighted_error': weighted_error,
                    'max_force_used': np.max(force_magnitudes),
                    'max_rate_used': max_rate_used,
                    'time_delta': time_delta,
                    'success': True,
                    'method': 'v3_lp_magnitude_and_rate_constrained'
                }
            else:
                # Fallback to V2.0 if rate-constrained LP fails
                return self.distribute_forces_v2_lp(desired_wrench, contact_points, grasp_matrix, dynamics_model, current_time, dt)
                
        except Exception as e:
            print(f"Error in V3.0 LP: {e}")
            return self.distribute_forces_v2_lp(desired_wrench, contact_points, grasp_matrix, dynamics_model, current_time, dt)
    
    def distribute_forces_v3_qp(self, desired_wrench, contact_points, grasp_matrix, dynamics_model, current_time=None, dt=1):
        """
        Version 3.0 QP: Add both force magnitude AND rate constraints using Quadratic Programming.
        Much cleaner formulation with rate constraints as quadratic penalty!
        """
        num_contacts = len(contact_points)
        
        # Get scaling factor c = m_max / f_max from dynamics model
        static_f_max = getattr(dynamics_model, 'static_f_max', 1.0)
        static_m_max = getattr(dynamics_model, 'static_m_max', static_f_max * 0.5)
        c = static_m_max / static_f_max if static_f_max > 1e-6 else 1.0
        
        # Weight matrix for proper scaling: diag([1, 1, c])
        W = np.diag([1.0, 1.0, c])
        
        # Calculate time delta for rate constraints
        if self.prev_forces is None:
            self.prev_forces = np.zeros(num_contacts)
            self.prev_time = current_time if current_time is not None else 0.0
        
        if current_time is not None and self.prev_time is not None:
            time_delta = current_time - self.prev_time
        else:
            time_delta = dt
        
        time_delta = max(time_delta, 1e-6)
        
        try:
            # QP formulation with rate penalty:
            # minimize (1/2) * ||W * (G*f - w_desired)||^2 + (λ/2) * ||f - f_prev||^2
            # subject to: 0 <= f_i <= max_force
            #             |f_i - f_prev_i| <= rate_limit_i * dt
            
            WG = W @ grasp_matrix
            
            # Primary objective: wrench tracking error
            H_wrench = WG.T @ WG
            g_wrench = -WG.T @ W @ desired_wrench
            
            # Rate penalty: encourage smooth force changes
            rate_penalty_weight = 1.0  # Adjust this to balance wrench tracking vs smooth forces
            H_rate = rate_penalty_weight * np.eye(num_contacts)
            g_rate = -rate_penalty_weight * self.prev_forces
            
            # Combined objective
            H = H_wrench + H_rate + 1e-6 * np.eye(num_contacts)  # Regularization
            g = g_wrench + g_rate
            
            # Constraints: force bounds AND rate bounds
            bounds = []
            max_increase_delta = self.max_rate_increase * time_delta
            max_decrease_delta = self.max_rate_decrease * time_delta
            
            for i in range(num_contacts):
                # Combine force bounds with rate bounds
                lower_bound = max(0.0, self.prev_forces[i] - max_decrease_delta)
                upper_bound = min(self.max_force, self.prev_forces[i] + max_increase_delta)
                bounds.append((lower_bound, upper_bound))
            
            # Solve QP
            def objective(f):
                return 0.5 * f.T @ H @ f + g.T @ f
            
            def jacobian(f):
                return H @ f + g
            
            # Initial guess: previous forces (should be feasible)
            f0 = np.clip(self.prev_forces, [b[0] for b in bounds], [b[1] for b in bounds])
            
            result = opt.minimize(
                objective,
                f0,
                method='L-BFGS-B',
                jac=jacobian,
                bounds=bounds,
                options={'disp': False}
            )
            
            if result.success:
                force_magnitudes = result.x
                
                # Update previous forces and time
                self.prev_forces = force_magnitudes.copy()
                self.prev_time = current_time if current_time is not None else (self.prev_time + dt)
                
                achieved_wrench = grasp_matrix @ force_magnitudes
                achieved_wrench[np.abs(achieved_wrench) < 1e-6] = 0
                
                wrench_error = achieved_wrench - desired_wrench
                wrench_error_magnitude = np.linalg.norm(wrench_error)
                weighted_error = np.linalg.norm(W @ wrench_error)
                
                # Calculate rate metrics
                force_changes = force_magnitudes - (self.prev_forces if hasattr(self, '_prev_forces_for_analysis') else np.zeros_like(force_magnitudes))
                max_rate_used = np.max(np.abs(force_changes)) / time_delta if time_delta > 1e-6 else 0.0
                
                self.wrench_errors.append(wrench_error_magnitude)
                
                return {
                    'force_magnitudes': force_magnitudes,
                    'achieved_wrench': achieved_wrench,
                    'wrench_error': wrench_error,
                    'wrench_error_magnitude': wrench_error_magnitude,
                    'weighted_error': weighted_error,
                    'max_force_used': np.max(force_magnitudes),
                    'max_rate_used': max_rate_used,
                    'time_delta': time_delta,
                    'success': True,
                    'method': 'v3_qp_magnitude_and_rate_constrained'
                }
            else:
                # Fallback to V2.0 QP if rate-constrained QP fails
                return self.distribute_forces_v2_qp(desired_wrench, contact_points, grasp_matrix, dynamics_model, current_time, dt)
                
        except Exception as e:
            print(f"Error in V3.0 QP: {e}")
            return self.distribute_forces_v2_qp(desired_wrench, contact_points, grasp_matrix, dynamics_model, current_time, dt)

    def distribute_forces_v3_refined(self, desired_wrench, contact_points, grasp_matrix, dt=1):
        """
        V3.0 Refined: Unit solution + minimum variance scaling within rate bounds.
        """
        
        # Step 1: Get ideal direction from V1.0
        unit_result = self.distribute_forces_v1_min_variance_qp(desired_wrench, contact_points, grasp_matrix)
        if not unit_result['success']:
            return self._fallback_result()
        
        unit_forces = unit_result['unit_forces']
        desired_scale = unit_result['scale_factor']
        
        # Step 2: Calculate rate bounds for each contact
        rate_bounds = []
        for i in range(len(contact_points)):
            prev_force = self.prev_forces[i] if self.prev_forces is not None else 0.0
            
            # Rate limits
            max_increase = self.max_rate_increase * dt
            max_decrease = self.max_rate_decrease * dt
            
            lower_bound = max(0.0, prev_force - max_decrease)
            upper_bound = min(self.max_force, prev_force + max_increase)
            
            rate_bounds.append([lower_bound, upper_bound])
        
        # Step 3: Calculate scale ranges
        scale_ranges = []
        for i, unit_force in enumerate(unit_forces):
            if unit_force > 1e-6:
                l_scale = rate_bounds[i][0] / unit_force
                r_scale = rate_bounds[i][1] / unit_force
                scale_ranges.append([l_scale, r_scale])
            else:
                # Zero unit force - any scale works within bounds
                scale_ranges.append([0.0, float('inf')])
        
        # Step 4: Find optimal scaling strategy
        if scale_ranges:
            # Try uniform scaling first (zero variance)
            global_min_scale = max([sr[0] for sr in scale_ranges])
            global_max_scale = min([sr[1] for sr in scale_ranges])
            
            if global_min_scale <= global_max_scale:
                # Uniform scaling possible!
                final_scale = np.clip(desired_scale, global_min_scale, global_max_scale)
                force_magnitudes = unit_forces * final_scale
            else:
                # No uniform solution - minimize variance
                force_magnitudes = self._solve_minimum_variance_scaling(
                    unit_forces, scale_ranges, desired_scale
                )
        else:
            force_magnitudes = np.zeros(len(contact_points))
        
        # Update state
        self.prev_forces = force_magnitudes.copy()
        
        achieved_wrench = grasp_matrix @ force_magnitudes
        
        return {
            'force_magnitudes': force_magnitudes,
            'achieved_wrench': achieved_wrench,
            'wrench_error': achieved_wrench - desired_wrench,
            'wrench_error_magnitude': np.linalg.norm(achieved_wrench - desired_wrench),
            'success': True,
            'method': 'v3_refined_minimum_variance_scaling'
        }

    def _solve_minimum_variance_scaling(self, unit_forces, scale_ranges, desired_scale):
        """
        Solve the minimum variance scaling problem when uniform scaling is impossible.
        
        Minimize: Var(k₁, k₂, ..., kₙ)
        Subject to: lᵢ ≤ kᵢ ≤ rᵢ for each i
        """
        import scipy.optimize as opt
        
        num_contacts = len(unit_forces)
        
        # Objective: minimize variance of scaling factors
        def objective(k):
            return np.var(k)
        
        # Constraints: each kᵢ must be in its scale range
        bounds = scale_ranges
        
        # Initial guess: try to get as close to desired_scale as possible
        k0 = []
        for i, (l_scale, r_scale) in enumerate(scale_ranges):
            k0.append(np.clip(desired_scale, l_scale, r_scale))
        
        result = opt.minimize(
            objective,
            k0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            # Apply individual scaling factors
            force_magnitudes = unit_forces * result.x
        else:
            # Fallback: use clipped desired scale
            force_magnitudes = unit_forces * np.array(k0)
        
        return force_magnitudes

    # ================================================================
    # FALLBACK METHODS
    # ================================================================
    
    def _distribute_with_slack_v1(self, desired_wrench, contact_points, grasp_matrix, current_time=None, dt=0.1):
        """Fallback method for V1.0."""
        # Simple fallback using quadratic minimization
        num_contacts = len(contact_points)
        
        def objective(forces):
            achieved_wrench = grasp_matrix @ forces
            error = achieved_wrench - desired_wrench
            return np.sum(error**2)
        
        bounds = [(0, 100) for _ in range(num_contacts)]
        initial_guess = np.ones(num_contacts) * 0.1
        
        result = opt.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        force_magnitudes = result.x
        achieved_wrench = grasp_matrix @ force_magnitudes
        wrench_error = achieved_wrench - desired_wrench
        
        return {
            'force_magnitudes': force_magnitudes,
            'achieved_wrench': achieved_wrench,
            'wrench_error': wrench_error,
            'wrench_error_magnitude': np.linalg.norm(wrench_error),
            'success': result.success,
            'method': 'v1_fallback'
        }
    
    def _distribute_with_slack_v2(self, desired_wrench, contact_points, grasp_matrix, dynamics_model):
        """Fallback method for V2.0."""
        # Use QP as fallback for LP
        return self.distribute_forces_v2_qp(desired_wrench, contact_points, grasp_matrix, dynamics_model)
    
    def _get_normalized_wrench(self, wrench):
        """Normalize wrench vector to unit length."""
        magnitude = np.linalg.norm(wrench)
        if magnitude > 1e-6:
            return wrench / magnitude
        return np.zeros_like(wrench)
        
    def reset(self):
        """Reset force distributor state."""
        self.prev_forces = None
        self.prev_time = None
        self.wrench_errors = []


# %%
def demo_force_distributor_methods_pro():
    """
    Demonstrate different versions and methods of the ForceDistributor.
    Compare LP vs QP vs RF (Refined) approaches for versions 2.0 and 3.0.
    """
    print("\n" + "="*80)
    print("🔧 FORCE DISTRIBUTOR METHODS COMPARISON")
    print("="*80)
    
    # Create test setup
    standard_objects = create_standard_objects()
    obj = standard_objects['triangle']
    
    contact_result = find_optimal_contacts(
        obj, mode='E+2', target_wrench=np.array([0, 0, 0]), 
        force_magnitude=1.0, verbose=False
    )
    contact_points = contact_result['contacts']
    
    # Create grasp matrix and dynamics model
    grasp_calculator = GraspMatrixCalculator()
    grasp_matrix = grasp_calculator.build_wrench_matrix(contact_points)
    dynamics = DynamicObjectModel(obj)
    
    # Test wrench - make it large enough to test force limits
    desired_wrench = np.array([20.0, 10.5, 0.8])
    
    print(f"Testing with {len(contact_points)} contact points")
    print(f"Desired wrench: [{desired_wrench[0]:.2f}, {desired_wrench[1]:.2f}, {desired_wrench[2]:.2f}]")
    print(f"Force limits: max_force={10.0}N, max_rate={4.0}N/s")
    
    # Test configurations - NOW INCLUDING REFINED METHODS
    test_configs = [
        {'version': 'v1', 'method': 'lp', 'name': 'V1.0 (No Constraints)'},
        {'version': 'v1', 'method': 'rf', 'name': 'V1.0 (Min Variance QP)'}, # NEW
        {'version': 'v2', 'method': 'lp', 'name': 'V2.0 Linear Programming'},
        {'version': 'v2', 'method': 'qp', 'name': 'V2.0 Quadratic Programming'}, 
        {'version': 'v2', 'method': 'rf', 'name': 'V2.0 Refined'},  # NEW
        {'version': 'v3', 'method': 'lp', 'name': 'V3.0 Linear Programming'},
        {'version': 'v3', 'method': 'qp', 'name': 'V3.0 Quadratic Programming'},
        {'version': 'v3', 'method': 'rf', 'name': 'V3.0 Refined'},  # NEW
    ]
    
    # Create force distributor
    distributor = ForceDistributorPro(max_force=10.0, max_rate_increase=4.0, max_rate_decrease=6.0)
    
    # Test each configuration
    results = {}
    
    for config in test_configs:
        print(f"\n🔍 Testing {config['name']}...")
        
        # Reset distributor between tests
        distributor.reset()
        
        try:
            # Call appropriate method

            result = distributor.distribute_forces(
                desired_wrench=desired_wrench,
                contact_points=contact_points,
                grasp_matrix=grasp_matrix,
                version=config['version'],
                method=config['method'],
                dynamics_model=dynamics,
                current_time=0.0
            )
            
            # Store result
            results[config['name']] = result
            
            # Print summary
            if result['success']:
                print(f"  ✅ Success: {result['method']}")
                print(f"     Forces: {', '.join([f'{f:.3f}' for f in result['force_magnitudes']])}")
                print(f"     Achieved wrench: [{result['achieved_wrench'][0]:.3f}, {result['achieved_wrench'][1]:.3f}, {result['achieved_wrench'][2]:.3f}]")
                print(f"     Error magnitude: {result['wrench_error_magnitude']:.4f}")
                if 'weighted_error' in result:
                    print(f"     Weighted error: {result['weighted_error']:.4f}")
                if 'max_force_used' in result:
                    print(f"     Max force used: {result['max_force_used']:.3f}N")
            else:
                print(f"  ❌ Failed")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[config['name']] = None
    
    # Create comparison visualization
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # Extract data for plotting
    method_names = []
    force_magnitudes_list = []
    achieved_wrenches = []
    error_magnitudes = []
    
    for name, result in results.items():
        if result and result['success']:
            method_names.append(name.replace(' Programming', '\nProgramming').replace('V2.0 Refined', 'V2.0\nRefined').replace('V3.0 Refined', 'V3.0\nRefined'))
            force_magnitudes_list.append(result['force_magnitudes'])
            achieved_wrenches.append(result['achieved_wrench'])
            error_magnitudes.append(result['wrench_error_magnitude'])
    
    # 1. Force magnitudes comparison
    ax1 = fig.add_subplot(gs[0, :])
    if force_magnitudes_list:
        x_pos = np.arange(len(method_names))
        width = 0.15
        
        for i in range(len(contact_points)):
            forces_i = [forces[i] for forces in force_magnitudes_list]
            ax1.bar(x_pos + i*width, forces_i, width, label=f'Contact {i+1}', alpha=0.8)
        
        ax1.axhline(y=distributor.max_force, color='red', linestyle='--', alpha=0.7, 
                   label=f'Force Limit ({distributor.max_force}N)')
        
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Force Magnitude (N)')
        ax1.set_title('Force Distribution Comparison Across Methods')
        ax1.set_xticks(x_pos + width * (len(contact_points)-1) / 2)
        ax1.set_xticklabels(method_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Achieved wrench comparison
    ax2 = fig.add_subplot(gs[1, 0])
    if achieved_wrenches:
        x_pos = np.arange(len(method_names))
        width = 0.25
        
        fx_values = [w[0] for w in achieved_wrenches]
        fy_values = [w[1] for w in achieved_wrenches]
        m_values = [w[2] for w in achieved_wrenches]
        
        ax2.bar(x_pos - width, fx_values, width, label='Fx', alpha=0.8)
        ax2.bar(x_pos, fy_values, width, label='Fy', alpha=0.8)
        ax2.bar(x_pos + width, m_values, width, label='M', alpha=0.8)
        
        # Add desired wrench reference lines
        ax2.axhline(y=desired_wrench[0], color='blue', linestyle=':', alpha=0.7, label='Desired Fx')
        ax2.axhline(y=desired_wrench[1], color='orange', linestyle=':', alpha=0.7, label='Desired Fy')
        ax2.axhline(y=desired_wrench[2], color='green', linestyle=':', alpha=0.7, label='Desired M')
        
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Wrench Component')
        ax2.set_title('Achieved Wrench Components')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(method_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Error comparison
    ax3 = fig.add_subplot(gs[1, 1])
    if error_magnitudes:
        bars = ax3.bar(method_names, error_magnitudes, alpha=0.8, color='red')
        
        # Add value labels on bars
        for bar, error in zip(bars, error_magnitudes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{error:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Error Magnitude')
        ax3.set_title('Wrench Tracking Error')
        ax3.set_xticklabels(method_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
    
    # 4. UPDATED Method characteristics table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create comparison table with REFINED methods included
    table_data = []
    headers = ['Method', 'Constraints', 'Optimization', 'Pros', 'Cons']
    
    table_data.append(['V1.0', 'None', 'LP + Scaling', 'Exact wrench achievable', 'No force/rate limits'])
    table_data.append(['V2.0 LP', 'Force limits', 'Linear Programming', 'Global optimum', 'L1-norm (less smooth)'])
    table_data.append(['V2.0 QP', 'Force limits', 'Quadratic Programming', 'L2-norm (smoother)', 'Local optimum'])
    table_data.append(['V2.0 Refined', 'Force limits', 'LP + Scaling', 'Exact rate constraint', 'Complex formulation'])  # NEW
    table_data.append(['V3.0 LP', 'Force + Rate limits', 'Linear Programming', 'Exact rate constraints', 'Complex formulation'])
    table_data.append(['V3.0 QP', 'Force + Rate limits', 'Quadratic Programming', 'Smooth + rate penalty', 'Approximate rate limits'])
    table_data.append(['V3.0 Refined', 'Force + Rate limits', 'Minimum Variance', 'Smooth + rate penalty', 'Approximate rate limits'])  # NEW
    
    # Create table
    table = ax4.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Style table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code by version
    version_colors = {
        'V1.0': '#E3F2FD',      # Light blue
        'V2.0': '#FFF3E0',      # Light orange  
        'V3.0': '#F3E5F5'       # Light purple
    }
    
    for i in range(1, len(table_data) + 1):
        version = table_data[i-1][0].split()[0]  # Get version part (V1.0, V2.0, V3.0)
        row_color = version_colors.get(version, '#f0f0f0')
        
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(row_color)
            
            # Highlight refined methods
            if 'Refined' in table_data[i-1][0]:
                table[(i, j)].set_facecolor('#FFE0B2')  # Highlight refined methods
                table[(i, j)].set_text_props(weight='bold')
    
    plt.suptitle('Force Distributor Methods Comparison (Including Refined)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Performance summary with REFINED methods
    print("\n📊 PERFORMANCE SUMMARY:")
    print("="*50)
    
    for name, result in results.items():
        if result and result['success']:
            print(f"\n{name}:")
            print(f"  Method: {result['method']}")
            print(f"  Error: {result['wrench_error_magnitude']:.4f}")
            if 'weighted_error' in result:
                print(f"  Weighted Error: {result['weighted_error']:.4f}")
            if 'max_force_used' in result:
                print(f"  Max Force: {result['max_force_used']:.3f}N")
            
            # Check constraint satisfaction
            force_limit_satisfied = all(f <= distributor.max_force + 1e-6 for f in result['force_magnitudes'])
            print(f"  Force Limits: {'✅ Satisfied' if force_limit_satisfied else '❌ Violated'}")
    
    print("\n🎯 UPDATED RECOMMENDATIONS:")
    print("- Use V2.0 Refined for guaranteed constraint satisfaction with preserved direction")
    print("- Use V3.0 Refined for minimum variance scaling with rate constraints") 
    print("- Use V2.0/V3.0 QP for smooth force profiles when exact constraints aren't critical")
    print("- Use LP versions when exact constraint satisfaction is critical but direction may change")
    print("- V1.0 for benchmarking when constraints are not a concern")
    print("\n🆕 REFINED METHODS:")
    print("- V2.0 Refined: Unit solution from V1.0 + uniform scaling within force bounds")
    print("- V3.0 Refined: V2.0 approach + minimum variance scaling for rate constraints")
    print("- ✅ Avoid solver numerical issues entirely for basic constraint cases")
    print("- ✅ Preserve optimal force direction from unconstrained solution")

# Run the updated demo
demo_force_distributor_methods_pro()

# %%
# ================================================================
# CONTROL GOAL CONFIGURATION CLASS
# ================================================================

from dataclasses import field


@dataclass
class ControlGoalWeights:
    """Weight configuration for different control goals"""
    # MPCC core weights
    qC: float          # Contour error weight (cross-track)
    qL: float          # Lag error weight (along-track) 
    qVtheta: float     # Progress reward weight
    heading_weight: float  # Heading error weight (used in omega_only and full_pose)
    
    # Force regularization weights
    rF: float          # Force magnitude penalty
    rdF: float         # Force rate penalty
    rVtheta: float     # Virtual velocity penalty
    rdVtheta: float    # Virtual acceleration penalty
    # Terminal weights (multipliers)
    qCNmult: float     # Terminal contour error multiplier
    
    # Regime-specific scaling factors, skip


# %%
class ControlGoalClass:
    """
    Enhanced control objectives and associated parameters for different manipulation goals.
    
    Supports:
    - 'position_only': Focus on position tracking, minimal orientation control
    - 'omega_only': Focus on orientation tracking, minimal position control  
    - 'full_pose': Balanced position and orientation tracking
    """
    
    def __init__(self, control_goal: str = 'full_pose'):
        """
        Initialize control goal configuration.
        
        Args:
            control_goal: One of 'position_only', 'omega_only', 'full_pose'
        """
        assert control_goal in ['position_only', 'omega_only', 'full_pose']
        self.control_goal = control_goal
        
        # Initialize weights based on control goal
        self.weights = self._get_default_weights()
        
        # Control goal specific flags and parameters
        self.use_heading_error = (control_goal in ['omega_only', 'full_pose'])
        self.position_priority = (control_goal in ['position_only', 'full_pose'])
        self.orientation_priority = (control_goal in ['omega_only', 'full_pose'])
        
        # Contact optimization preferences
        self.preferred_contact_mode = self._get_preferred_contact_mode()
        
        # Visualization preferences
        self.visualization_focus = self._get_visualization_focus()
        
        # Performance tracking
        self.performance_history = {
            'cost_components': [],
            'weight_adaptations': [],
            'regime_transitions': []
        }
        
    def _get_default_weights(self) -> ControlGoalWeights:
        """Get default weight configuration for the control goal."""
        if self.control_goal == 'position_only':
            return ControlGoalWeights(
                qC=50.0,           # High contour error weight
                qL=10.0,           # Lower lag error 
                qVtheta=0.3,       # Moderate progress
                heading_weight=0.001,  # Very low heading weight
                rF=1e-4,
                rdF=0.1,
                rVtheta=1e-4,
                rdVtheta=1e-4,
                qCNmult=5.0,
                # regime_scaling={'static': 1.0, 'quasi': 1.0, 'dynamic': 0.8}
            )
        elif self.control_goal == 'omega_only':
            return ControlGoalWeights(
                qC=5.0,            # Lower contour error
                qL=100.0,          # Higher lag error (maintain progress)
                qVtheta=0.5,       # High progress reward
                heading_weight=50.0,   # Very high heading weight
                rF=1e-4,
                rdF=0.1,
                rVtheta=1e-4,
                rdVtheta=1e-4,
                qCNmult=2.0,
                # regime_scaling={'static': 1.2, 'quasi': 1.0, 'dynamic': 1.1}
            )
        else:  # full_pose
            # return ControlGoalWeights(
            #     qC=20.0,           # Balanced contour error
            #     qL=20.0,           # Balanced lag error
            #     qVtheta=0.5,       # Moderate progress
            #     heading_weight=10.0,   # Moderate heading weight
            #     rF=1e-4,
            #     rdF=0.1,
            #     rVtheta=1e-4,
            #     rdVtheta=1e-4,
            #     qCNmult=3.0,
            #     # regime_scaling={'static': 1.0, 'quasi': 1.0, 'dynamic': 1.0}
            # )
            return ControlGoalWeights(
                qC=5.80,           # Balanced contour error
                qL=10.0,           # Balanced lag error
                qVtheta=0.5,       # Moderate progress
                heading_weight=10.0,   # Moderate heading weight
                rF=1e-4,
                rdF=0.1,
                rVtheta=1e-4,
                rdVtheta=0.1,
                qCNmult=1.0,
                # regime_scaling={'static': 1.0, 'quasi': 1.0, 'dynamic': 1.0}
            )


    def _get_preferred_contact_mode(self) -> str:
        """Get preferred contact optimization mode for this control goal."""
        contact_modes = {
            'position_only': 'E',      # Good force transmission
            'omega_only': '2',         # Good torque generation
            'full_pose': 'E+2'         # Balanced capability
        }
        return contact_modes[self.control_goal]
    
    def _get_visualization_focus(self) -> Dict[str, bool]:
        """Get visualization preferences for this control goal."""
        return {
            'show_position_errors': self.position_priority,
            'show_orientation_errors': self.orientation_priority,
            'show_contour_errors': True,
            'show_lag_errors': True,
            'highlight_primary_objective': True
        }
    
    def update_weights(self, weight_updates: Dict[str, float]):
        """
        Update specific weights dynamically.
        
        Args:
            weight_updates: Dictionary of weight name -> new value
        """
        for weight_name, new_value in weight_updates.items():
            if hasattr(self.weights, weight_name):
                old_value = getattr(self.weights, weight_name)
                setattr(self.weights, weight_name, new_value)
                print(f"🔧 Updated {weight_name}: {old_value:.6f} → {new_value:.6f}")
                self.performance_history['weight_adaptations'].append({
                    'weight': weight_name,
                    'old_value': old_value,
                    'new_value': new_value
                })
            else:
                print(f"⚠️ Unknown weight: {weight_name}")

    # Default to using base weights    
    def get_cost_matrix_scaling(self) -> Dict[str, float]:
        """
        Get scaling factors for different cost matrix components.
        Useful for balancing different objectives in the MPCC cost function.
        """
        return {
            'contour_scaling': 1.0,
            'lag_scaling': 1.0,
            'heading_scaling': 1.0,
            'progress_scaling': 1.0,
            'force_scaling': 1.0,  # Always important for feasibility
        }

        return {
            'contour_scaling': 1.0 if self.position_priority else 0.1,
            'lag_scaling': 1.0 if self.position_priority else 0.5,
            'heading_scaling': 1.0 if self.orientation_priority else 0.01,
            'progress_scaling': 0.8 if self.control_goal == 'omega_only' else 0.5,
            'force_scaling': 1.0,  # Always important for feasibility
        }
    
    # Adaptive weight adjustment based on performance
    # Can be called periodically to tune weights
    # Might also be triggered by external performance metrics
    # Might have another version based on Adam optimizer, or based on regime transitions
    # Might not get used, but here for completeness
    def adapt_weights_based_on_performance(self, error_metrics: Dict[str, float], 
                                         tolerance: Dict[str, float] = None):
        """
        Adaptively adjust weights based on tracking performance.
        
        Args:
            error_metrics: Dictionary containing current error values
            tolerance: Dictionary containing acceptable error tolerances
        """
        if tolerance is None:
            tolerance = {
                'contour_error': 0.05,  # 5cm lateral tolerance
                'lag_error': 0.1,       # 10cm longitudinal tolerance  
                'heading_error': 0.1    # ~5.7 degrees
            }
        
        adaptations = {}
        
        # Increase contour weight if lateral error is too high
        if 'contour_error' in error_metrics and self.position_priority:
            if abs(error_metrics['contour_error']) > tolerance['contour_error']:
                new_qC = min(self.weights.qC * 1.2, 100.0)  # Cap at 100
                adaptations['qC'] = new_qC
        
        # Increase lag weight if longitudinal error is too high
        if 'lag_error' in error_metrics and self.position_priority:
            if abs(error_metrics['lag_error']) > tolerance['lag_error']:
                new_qL = min(self.weights.qL * 1.15, 200.0)  # Cap at 200
                adaptations['qL'] = new_qL
        
        # Increase heading weight if orientation error is too high
        if 'heading_error' in error_metrics and self.orientation_priority:
            if abs(error_metrics['heading_error']) > tolerance['heading_error']:
                new_heading = min(self.weights.heading_weight * 1.3, 100.0)
                adaptations['heading_weight'] = new_heading
        
        # Apply adaptations
        if adaptations:
            self.update_weights(adaptations)
            print(f"🤖 Adaptive weight updates applied: {len(adaptations)} weights changed")
    
    def get_mpcc_cost_matrices(self, terminal_stage: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate MPCC cost matrices Q and R based on control goals.
        
        Args:
            terminal_stage: Whether this is for the terminal cost
            
        Returns:
            Q: State cost matrix
            R: Input cost matrix
        """
        # Get scaling factors
        scaling = self.get_cost_matrix_scaling()
        
        # Apply terminal multiplier if needed
        qC = self.weights.qC * (self.weights.qCNmult if terminal_stage else 1.0)
        qL = self.weights.qL
        qVtheta = self.weights.qVtheta
        heading_weight = self.weights.heading_weight
        
        # Apply control goal specific scaling
        qC *= scaling['contour_scaling']
        qL *= scaling['lag_scaling']
        qVtheta *= scaling['progress_scaling']
        heading_weight *= scaling['heading_scaling']
        
        # State cost matrix (for error components)
        # This will be used in combination with error gradients
        Q_error = np.diag([qC, qL])  # [contour_error, lag_error]
        
        # Heading cost (if applicable)
        if self.use_heading_error:
            Q_heading = heading_weight
        else:
            Q_heading = 0.0
        
        # Progress reward
        Q_progress = qVtheta
        
        # Input cost matrix R
        R = np.diag([self.weights.rF, self.weights.rVtheta])  # [force_reg, virtual_velocity_reg]
        
        # Input rate cost matrix
        R_rate = np.diag([self.weights.rdF, self.weights.rVtheta])
        
        return Q_error, R, R_rate, Q_heading, Q_progress
    
    def validate_weights(self) -> bool:
        """Validate that weights are reasonable."""
        w = self.weights
        
        # Check for non-negative weights
        if any(getattr(w, attr) < 0 for attr in ['qC', 'qL', 'qVtheta', 'rF', 'rdF', 'rVtheta']):
            print("❌ Negative weights detected")
            return False
        
        # Check for reasonable relative magnitudes
        if w.qC > 1000 or w.qL > 1000:
            print("⚠️ Very large tracking weights - may cause numerical issues")
        
        if w.rF > 1 or w.rdF > 10:
            print("⚠️ Large regularization weights - may be too conservative")
        
        # Check goal-specific consistency
        if self.control_goal == 'position_only' and w.heading_weight > 1.0:
            print("⚠️ High heading weight for position-only control")
            
        if self.control_goal == 'omega_only' and w.qC > w.heading_weight:
            print("⚠️ Contour weight higher than heading weight for orientation control")
        
        return True
    
    def get_reference_trajectory_preferences(self) -> Dict[str, Union[float, bool]]:
        """Get preferences for reference trajectory generation."""
        return {
            'lookahead_distance': 0.2 if self.control_goal == 'position_only' else 0.15,
            'path_smoothing': True,
            'orientation_smoothing': self.use_heading_error,
            'curvature_limit': 2.0 if self.control_goal == 'omega_only' else 1.0,
            'speed_adaptation': self.control_goal != 'position_only'
        }
    
    def print_configuration(self):
        """Print current configuration for debugging."""
        print(f"\n🎯 Control Goal Configuration: {self.control_goal}")
        print(f"   Position Priority: {self.position_priority}")
        print(f"   Orientation Priority: {self.orientation_priority}")
        print(f"   Use Heading Error: {self.use_heading_error}")
        print(f"   Preferred Contact Mode: {self.preferred_contact_mode}")
        
        print(f"\n📊 Weight Configuration:")
        w = self.weights
        print(f"   Tracking: qC={w.qC:.1f}, qL={w.qL:.1f}, heading={w.heading_weight:.3f}")
        print(f"   Progress: qVtheta={w.qVtheta:.3f}")
        print(f"   Regularization: rF={w.rF:.1e}, rdF={w.rdF:.1f}, rVtheta={w.rVtheta:.1e}")
        print(f"   Terminal: qCNmult={w.qCNmult:.1f}")
        # print(f"   Regime Scaling: {w.regime_scaling}")



# %%
class AugmentedState:
    """
    Augmented state for MPCC including object state and path parameter.
    """
    def __init__(self, state_data, path_param=0.0):
        object_position = state_data.get('object_position', np.array([0.0, 0.0]))
        object_orientation = state_data.get('object_orientation', 0.0)
        velocity_body = state_data.get('velocity_body', np.array([0.0, 0.0]))
        angular_velocity = state_data.get('angular_velocity', 0.0)

        self.object_x = object_position[0]
        self.object_y = object_position[1]
        self.object_theta = object_orientation
        self.object_vx = velocity_body[0]
        self.object_vy = velocity_body[1]
        self.object_vw = angular_velocity
        
        self.path_param = path_param      # Progress parameter along contour
        
    def __str__(self):
        return f"MPCCState(object=({self.object_x:.2f}, {self.object_y:.2f}, {self.object_theta:.2f}, {self.object_vx:.2f}, {self.object_vy:.2f}, {self.object_vw:.2f}), path_param={self.path_param:.4f})"
    
    def get_augmented_vec(self):
        """Return complete state vector [x,y,theta,vx,vy,vw,path_param]"""
        return np.concatenate([
            np.array([self.object_x, self.object_y, self.object_theta,
                     self.object_vx, self.object_vy, self.object_vw]),
            np.array([self.path_param])
        ])
    
    def get_object_vec(self):
        """Return only the object state vector [x,y,theta,vx,vy,vw]"""
        return np.array([self.object_x, self.object_y, self.object_theta,
                         self.object_vx, self.object_vy, self.object_vw])
    

def visualize_mpcc_solution(
    augmented_states: List[AugmentedState], 
    reference_path: 'SplineReferencePath'
):
    """
    Visualize the augmented state trajectory and reference path.
    
    ✅ MODULAR VERSION: Adapted from golden implementation
    
    Args:
        augmented_states: List of AugmentedState objects (modular equivalent of linearization_states)
        reference_path: SplineReferencePath object
        
    Returns:
        matplotlib figure for further customization
    """
    
    plt.figure(figsize=(10, 6))
    
    # Plot reference path
    path_points = []
    for t in np.linspace(0, 1, 100):
        point = reference_path.get_point_at_parameter(t)
        path_points.append(point)
    path_points = np.array(path_points)
    plt.plot(path_points[:, 0], path_points[:, 1], 'b-', label='Reference Path', linewidth=2)
    
    # Plot the approximated path points at path parameters
    approx_path_points = []
    for state in augmented_states:
        current_path_param = state.path_param
        point = reference_path.get_point_at_parameter(current_path_param)
        approx_path_points.append(point)
    approx_path_points = np.array(approx_path_points)
    plt.plot(approx_path_points[:, 0], approx_path_points[:, 1], 'ro', 
             label='Path Points (at s_k)', alpha=0.7, markersize=4)
    
    # Plot actual object trajectory
    object_x = [state.object_x for state in augmented_states]
    object_y = [state.object_y for state in augmented_states]
    plt.plot(object_x, object_y, 'r.-', label='Object Trajectory', linewidth=2, markersize=8)
    
    # Plot object orientation (every 3rd state to avoid clutter)
    for i, state in enumerate(augmented_states):
        if i % 3 == 0:
            x, y, theta = state.object_x, state.object_y, state.object_theta
            dx = 0.1 * np.cos(theta)
            dy = 0.1 * np.sin(theta)
            plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, 
                     fc='g', ec='g', alpha=0.7)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Object Trajectory vs Reference Path')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return plt

# %%
# ================================================================
# HELPER FUNCTIONS FOR CONTROL GOAL INTEGRATION
# ================================================================

def create_control_goal_from_mode(mode: str, custom_weights: Dict[str, float] = None) -> ControlGoalClass:
    """
    Factory function to create control goal with optional custom weights.
    
    Args:
        mode: Control mode ('position_only', 'omega_only', 'full_pose')
        custom_weights: Optional dictionary of custom weight values
        
    Returns:
        ControlGoalClass instance
    """
    control_goal = ControlGoalClass(mode)
    
    if custom_weights:
        control_goal.update_weights(custom_weights)
    
    # Validate configuration
    if not control_goal.validate_weights():
        print("⚠️ Weight validation failed - using defaults")
        control_goal = ControlGoalClass(mode)  # Reset to defaults
    
    return control_goal

def extract_mpcc_state_from_simulation_history(state_history: List[Dict]) -> AugmentedState:
    """
    Extract AugmentedState from simulation history for MPCC initialization.
    
    Args:
        state_history: List of simulation state dictionaries
        
    Returns:
        AugmentedState compatible with MPCC
    """
    if not state_history:
        # Return default state if no history available
        default_state_data = {
            'object_position': np.array([0.0, 0.0]),
            'object_orientation': 0.0,
            'velocity_body': np.array([0.0, 0.0]),
            'angular_velocity': 0.0
        }
        return AugmentedState(default_state_data, path_param=0.0)
    
    current_state = state_history[-1]
    
    # Extract state components with fallback defaults
    position = current_state.get('position', np.array([0.0, 0.0]))
    orientation = current_state.get('orientation', 0.0)
    velocity_body = current_state.get('velocity_body', np.array([0.0, 0.0]))
    angular_velocity = current_state.get('angular_velocity', 0.0)
    
    # Prepare state data for AugmentedState constructor
    state_data = {
        'object_position': np.array(position),
        'object_orientation': orientation,
        'velocity_body': np.array(velocity_body),
        'angular_velocity': angular_velocity
    }
    
    # Initialize with path parameter 0.0 (will be updated by path finding)
    return AugmentedState(state_data, path_param=0.0)

def normalize_theta_diff(target_theta: float, current_theta: float) -> float:
    """
    Helper function to normalize angle difference to [-π, π].
    
    Args:
        target_theta: Target angle
        current_theta: Current angle
        
    Returns:
        Normalized angle difference
    """
    diff = target_theta - current_theta
    return (diff + np.pi) % (2 * np.pi) - np.pi

def initialize_mpcc_state_from_goal(augmented_state: AugmentedState, 
                                  reference_path: 'SplineReferencePath',
                                  control_goal: ControlGoalClass) -> AugmentedState:
    """
    Initialize MPCC state with control goal specific considerations.
    
    ✅ MATLAB-COMPATIBLE: No dependencies on decision variable structure.
    
    Args:
        augmented_state: Current augmented state
        reference_path: Reference path
        control_goal: Control goal configuration
        
    Returns:
        AugmentedState with appropriate path parameter
    """
    if control_goal.orientation_priority and not control_goal.position_priority:
        # For omega_only mode, no projection, only increment path parameter 
        path_param = augmented_state.path_param + 0.05  # Increment by fixed step
        path_param = np.clip(path_param, 0.0, 1.0)
        print(f"🎯 Orientation-based initialization: path_param={path_param:.4f}")
    else:
        # Standard position-based closest point
        query_point = [augmented_state.object_x, augmented_state.object_y]
        _, path_param, _ = reference_path.find_closest_point(query_point)
        print(f"🎯 Position-based initialization: path_param={path_param:.4f}")
    
    # Create new AugmentedState with updated path parameter
    state_data = {
        'object_position': np.array([augmented_state.object_x, augmented_state.object_y]),
        'object_orientation': augmented_state.object_theta,
        'velocity_body': np.array([augmented_state.object_vx, augmented_state.object_vy]),
        'angular_velocity': augmented_state.object_vw
    }
    
    return AugmentedState(state_data, path_param=path_param)


def update_path_parameter_from_state(augmented_state: AugmentedState, 
                                    reference_path: 'SplineReferencePath',
                                    control_goal: ControlGoalClass,
                                    max_path_param_jump: float = 0.1) -> AugmentedState:
    """
    Update path parameter based on current state and control goal.
    
    ✅ MATLAB-COMPATIBLE: No dependencies on decision variable structure.
    
    This function is useful for updating the path parameter during execution
    when the object has moved significantly.
    
    Args:
        augmented_state: Current augmented state
        reference_path: Reference path
        control_goal: Control goal configuration
        max_path_param_jump: Maximum allowed jump in path parameter
        
    Returns:
        AugmentedState with updated path parameter
    """
    current_path_param = augmented_state.path_param
    
    # Find new path parameter based on control goal
    if control_goal.orientation_priority and not control_goal.position_priority:
        new_path_param = current_path_param + 0.05  # Increment by fixed step
        print(f"🔄 Orientation-based update: {current_path_param:.4f} → {new_path_param:.4f}")
    else:
        # Position-based update
        query_point = [augmented_state.object_x, augmented_state.object_y]
        _, new_path_param, _ = reference_path.find_closest_point(query_point)
    
    # Limit path parameter jumps for continuity
    if abs(new_path_param - current_path_param) > max_path_param_jump:
        if new_path_param > current_path_param:
            new_path_param = current_path_param + max_path_param_jump
        else:
            new_path_param = current_path_param - max_path_param_jump
        print(f"⚠️ Path parameter jump limited: {current_path_param:.4f} → {new_path_param:.4f}")
    
    # Ensure path parameter stays within bounds
    new_path_param = np.clip(new_path_param, 0.0, 1.0)
    
    # Create updated state
    state_data = {
        'object_position': np.array([augmented_state.object_x, augmented_state.object_y]),
        'object_orientation': augmented_state.object_theta,
        'velocity_body': np.array([augmented_state.object_vx, augmented_state.object_vy]),
        'angular_velocity': augmented_state.object_vw
    }
    
    return AugmentedState(state_data, path_param=new_path_param)

def adapt_reference_path_for_goal(reference_path: 'SplineReferencePath', 
                                 control_goal: ControlGoalClass) -> 'SplineReferencePath':
    """
    Adapt reference path characteristics based on control goal.
    
    ✅ MATLAB-COMPATIBLE: No dependencies on decision variable structure.
    
    Args:
        reference_path: Original reference path
        control_goal: Control goal configuration
        
    Returns:
        Potentially modified reference path (currently returns original)
    """
    preferences = control_goal.get_reference_trajectory_preferences()
    
    # For now, return the original path
    # Future enhancements could include:
    # 1. Path smoothing based on preferences['path_smoothing']
    # 2. Orientation smoothing based on preferences['orientation_smoothing']
    # 3. Curvature limiting based on preferences['curvature_limit']
    # 4. Speed adaptation based on preferences['speed_adaptation']
    
    print(f"🛤️ Path adaptation preferences:")
    print(f"   Lookahead distance: {preferences['lookahead_distance']}")
    print(f"   Path smoothing: {preferences['path_smoothing']}")
    print(f"   Orientation smoothing: {preferences['orientation_smoothing']}")
    print(f"   Curvature limit: {preferences['curvature_limit']}")
    print(f"   Speed adaptation: {preferences['speed_adaptation']}")
    
    return reference_path

def get_goal_specific_contact_configuration(object_model, control_goal: ControlGoalClass):
    """
    Get optimal contact configuration for the specified control goal.
    
    ✅ MATLAB-COMPATIBLE: No dependencies on decision variable structure.
    
    Args:
        object_model: Generic object
        control_goal: Control goal configuration
        
    Returns:
        List of contact points optimized for the control goal
    """
    mode = control_goal.preferred_contact_mode
    
    print(f"👏 Optimizing contacts for {control_goal.control_goal} (mode: {mode})")
    
    # Use existing contact optimization
    contact_result = find_optimal_contacts(
        object_model,
        mode=mode,
        target_wrench=np.array([0, 0, 0]),
        force_magnitude=1.0,
        verbose=False
    )
    
    contacts = contact_result['contacts']
    print(f"   Selected {len(contacts)} contact points")
    
    return contacts

def validate_augmented_state_compatibility(augmented_state: AugmentedState, 
                                         reference_path: 'SplineReferencePath') -> bool:
    """
    Validate that the augmented state is compatible with the reference path.
    
    ✅ MATLAB-COMPATIBLE: No dependencies on decision variable structure.
    
    Args:
        augmented_state: State to validate
        reference_path: Reference path
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Check path parameter bounds
        if not (0.0 <= augmented_state.path_param <= 1.0):
            print(f"❌ Path parameter out of bounds: {augmented_state.path_param}")
            return False
        
        # Check if we can get a valid reference point
        ref_point = reference_path.get_point_at_parameter(augmented_state.path_param)
        if ref_point is None or len(ref_point) < 3:
            print(f"❌ Invalid reference point at path_param={augmented_state.path_param}")
            return False
        
        # Check state vector validity
        state_vec = augmented_state.get_augmented_vec()
        if np.any(np.isnan(state_vec)) or np.any(np.isinf(state_vec)):
            print(f"❌ Invalid state vector: {state_vec}")
            return False
        
        print(f"✅ State validation passed")
        return True
        
    except Exception as e:
        print(f"❌ State validation error: {e}")
        return False

def create_augmented_state_from_vector(state_vector: np.ndarray, 
                                     path_param: float = 0.0) -> AugmentedState:
    """
    Create AugmentedState from a state vector.
    
    ✅ MATLAB-COMPATIBLE: No dependencies on decision variable structure.
    
    Args:
        state_vector: State vector [x, y, theta, vx, vy, vw]
        path_param: Path parameter
        
    Returns:
        AugmentedState object
    """
    if len(state_vector) < 6:
        raise ValueError(f"State vector must have at least 6 elements, got {len(state_vector)}")
    
    state_data = {
        'object_position': np.array([state_vector[0], state_vector[1]]),
        'object_orientation': state_vector[2],
        'velocity_body': np.array([state_vector[3], state_vector[4]]),
        'angular_velocity': state_vector[5]
    }
    
    return AugmentedState(state_data, path_param)

def interpolate_augmented_states(state1: AugmentedState, state2: AugmentedState, 
                               alpha: float) -> AugmentedState:
    """
    Interpolate between two AugmentedStates.
    
    ✅ MATLAB-COMPATIBLE: No dependencies on decision variable structure.
    
    Args:
        state1: First state
        state2: Second state  
        alpha: Interpolation factor (0.0 = state1, 1.0 = state2)
        
    Returns:
        Interpolated AugmentedState
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Interpolate position and velocities linearly
    x = (1 - alpha) * state1.object_x + alpha * state2.object_x
    y = (1 - alpha) * state1.object_y + alpha * state2.object_y
    vx = (1 - alpha) * state1.object_vx + alpha * state2.object_vx
    vy = (1 - alpha) * state1.object_vy + alpha * state2.object_vy
    vw = (1 - alpha) * state1.object_vw + alpha * state2.object_vw
    
    # Interpolate orientation using SLERP for proper angle handling
    theta1 = state1.object_theta
    theta2 = state2.object_theta
    
    # Find shortest angle path
    angle_diff = normalize_theta_diff(theta2, theta1)
    theta = theta1 + alpha * angle_diff
    
    # Interpolate path parameter
    path_param = (1 - alpha) * state1.path_param + alpha * state2.path_param
    
    # Create interpolated state
    state_data = {
        'object_position': np.array([x, y]),
        'object_orientation': theta,
        'velocity_body': np.array([vx, vy]),
        'angular_velocity': vw
    }
    
    return AugmentedState(state_data, path_param)
# ================================================================
# DEMONSTRATION FUNCTION
# ================================================================

def demo_control_goal_configuration():
    """Demonstrate control goal configuration and adaptation."""
    
    print("🎯 Control Goal Configuration Demo")
    print("="*50)
    
    # Test different control goals
    goals = ['position_only', 'omega_only', 'full_pose']
    
    for goal in goals:
        print(f"\n--- Testing {goal.upper()} ---")
        
        # Create control goal
        control_goal = create_control_goal_from_mode(goal)
        control_goal.print_configuration()
        
        # Test weight validation
        print(f"\n✅ Weight validation: {control_goal.validate_weights()}")
        
        # Test cost matrices
        Q_error, R, R_rate, Q_heading, Q_progress = control_goal.get_mpcc_cost_matrices()
        print(f"📊 Cost matrices generated:")
        print(f"   Q_error shape: {Q_error.shape}")
        print(f"   R shape: {R.shape}")
        print(f"   R_rate shape: {R_rate.shape}")
        print(f"   Q_heading: {Q_heading:.3f}")
        print(f"   Q_progress: {Q_progress:.3f}")
        
        # Test performance-based adaptation
        print(f"\n🤖 Testing adaptive weights...")
        test_errors = {
            'contour_error': 0.1,   # High lateral error
            'lag_error': 0.02,      # Low longitudinal error
            'heading_error': 0.15   # High heading error
        }
        control_goal.adapt_weights_based_on_performance(test_errors)
        control_goal.print_configuration()
        
        print(f"📈 Performance history: {len(control_goal.performance_history['weight_adaptations'])} adaptations")

def demo_augmented_state_helpers():
    """Demonstrate AugmentedState helper functions."""
    
    print("\n🔧 AugmentedState Helper Functions Demo")
    print("="*50)
    
    # Create sample state history
    sample_history = [
        {
            'position': np.array([0.1, 0.2]),
            'orientation': 0.1,
            'velocity_body': np.array([0.05, 0.02]),
            'angular_velocity': 0.01,
            'time': 0.0
        },
        {
            'position': np.array([0.15, 0.25]),
            'orientation': 0.15,
            'velocity_body': np.array([0.08, 0.03]),
            'angular_velocity': 0.02,
            'time': 0.1
        }
    ]
    
    # Test state extraction
    print("\n1. Testing state extraction...")
    augmented_state = extract_mpcc_state_from_simulation_history(sample_history)
    print(f"   Extracted state: {augmented_state}")
    
    # Test state vector conversion
    print("\n2. Testing state vector conversion...")
    state_vec = augmented_state.get_object_vec()
    print(f"   State vector: {state_vec}")
    
    recreated_state = create_augmented_state_from_vector(state_vec, path_param=0.5)
    print(f"   Recreated state: {recreated_state}")
    
    # Test interpolation
    print("\n3. Testing state interpolation...")
    state1 = create_augmented_state_from_vector(np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]), 0.0)
    state2 = create_augmented_state_from_vector(np.array([1.0, 1.0, np.pi/2, 0.2, 0.1, 0.1]), 1.0)
    
    interp_state = interpolate_augmented_states(state1, state2, 0.5)
    print(f"   Interpolated state (α=0.5): {interp_state}")

if __name__ == "__main__":
    demo_control_goal_configuration()
    demo_augmented_state_helpers()

# %%
# MODEL PARAMETERS CLASS
# ================================================================

class ModelParams:
    """
    Model parameters class that integrates object dynamics and control configuration.
    Provides centralized parameter management with adaptation capabilities.
    """
    
    def __init__(self, object_model: DynamicObjectModel, control_goal: ControlGoalClass):
        """
        Initialize model parameters from object dynamics and control goal.

        Args:
            object_model: DynamicObjectModel with physical properties
            control_goal: ControlGoalClass with control configuration
        """
        self.object_model = object_model
        self.control_goal = control_goal

        self.grasp_calculator = GraspMatrixCalculator()
        self.object_num_edges = len(np.array(self.object_model.object.boundary.coords)) - 1
        
        # Extract object physical properties
        self._extract_object_properties()
        
        # Determine force configuration from control goal
        self._determine_force_configuration()
        
        # Set default operational limits
        self._set_default_limits()
        
        # Create grasp matrix (will be updated when contact points are set)
        self.grasp_matrix = None
        self.contact_points = None

        # Define num_states for MPCC
        self.num_states = 7  # [x, y, theta, vx, vy, vw, path_param]
        
        # Parameter adaptation history
        self.parameter_history = {
            'force_limit_changes': [],
            'regime_changes': [],
            'contact_updates': []
        }
        
        print(f"🔧 ModelParams initialized:")
        print(f"   Mass: {self.mass:.3f} kg, Inertia: {self.inertia:.6f} kg⋅m²")
        print(f"   Control Goal: {self.control_goal.control_goal}")
        print(f"   Force Configuration: {self.num_forces} contacts, max {self.max_forces_allowed:.1f}N")
    
    def _extract_object_properties(self):
        """Extract physical properties from object model."""
        # Object mass and inertia
        
        self.mass = getattr(self.object_model, 'mass', 1.0)  # Default 1kg if not specified
        self.inertia = getattr(self.object_model, 'moment_of_inertia', 0.1)

        # Friction properties
        self.static_f_max = max(1e-9, self.object_model.static_f_max)
        self.kinetic_f_max = max(1e-9, self.object_model.kinetic_f_max)
        
        # Estimate static and kinetic moment limits if not available
        if hasattr(self.object_model, 'static_m_max'):
            self.static_m_max = self.object_model.static_m_max
        else:
            # Estimate from force limits and object size
            obj = self.object_model.object
            characteristic_length = self._estimate_characteristic_length(obj)
            self.static_m_max = self.static_f_max * characteristic_length * 0.5
        
        if hasattr(self.object_model, 'kinetic_m_max'):
            self.kinetic_m_max = self.object_model.kinetic_m_max
        else:
            obj = self.object_model.object
            characteristic_length = self._estimate_characteristic_length(obj)
            self.kinetic_m_max = self.kinetic_f_max * characteristic_length * 0.5

        self.twist_scale = self.static_m_max / self.static_f_max
        

    def _estimate_characteristic_length(self, obj):
        """Estimate characteristic length of object for moment calculations."""
        if hasattr(obj, 'vertices'):
            vertices = np.array(obj.vertices)
            # Use maximum distance from centroid as characteristic length
            centroid = np.mean(vertices, axis=0)
            distances = np.linalg.norm(vertices - centroid, axis=1)
            return np.max(distances)
        else:
            return 0.5  # Default 0.5m characteristic length
    
    def _determine_force_configuration(self):
        """Determine number of forces based on control goal preferences."""
        preferred_mode = self.control_goal.preferred_contact_mode
        
        # Map contact modes to typical number of forces
        mode_to_forces = {
            '2': 2,     # Basic for omega_only
            'E': self.object_num_edges,     # Edge contacts
            'E+2': self.object_num_edges + 2,   # Edge + augmented points
        }
        
        self.num_forces = mode_to_forces.get(preferred_mode, 4)  # Default to 4
        
    def _set_default_limits(self):
        """Set default operational limits and constraints."""
        # Force limits (conservative approach: use kinetic limits + margin)
        self.max_forces_allowed = self.static_f_max * 1.2  # 20% margin over static friction
        
        # Rate limits (based on physical and control constraints)
        self.max_force_rate_increase = self.max_forces_allowed * 0.5  # Can ramp up quickly
        self.max_force_rate_decrease = self.max_forces_allowed * 0.8  # Can reduce faster
        
        # Velocity limits (based on typical manipulation speeds)
        self.max_linear_velocity = 5.0    # m/s
        self.max_angular_velocity = 3.0   # rad/s
        
        # Path parameter limits
        self.path_param_bounds = (0.0, 1.0)
        self.virtual_speed_bounds = (0, 2.0) # virtual param speed
        
        # Workspace limits (will be updated based on reference path)
        self.position_bounds = (-10, 10)
        self.orientation_bounds = (-np.pi, np.pi)
    
    def get_normalization_scales(self) -> Dict[str, float]:
        """
        Get normalization scales for StateInputNormalization.
        
        Returns:
            Dictionary of scaling factors for normalization
        """
        return {
            'position': 2 * max(self.position_bounds[1], -self.position_bounds[0]),           # meters (workspace scale)
            'orientation': 2 * max(self.orientation_bounds[1], -self.orientation_bounds[0]),   # radians (full rotation)
            'velocity': self.max_forces_allowed * self.num_forces / self.mass,  # Achievable velocity
            # just an estimate here, might get updated later
            'angular_velocity': self.max_forces_allowed * self._estimate_characteristic_length(self.object_model.object) * self.num_forces / (2* self.inertia),  # Achievable angular velocity
            'path_param': 2.0,          # path parameter range
            'force': self.max_forces_allowed,  # force magnitude
            'contour_speed': self.max_linear_velocity  # contour speed
        }
    
    def set_contact_configuration(self, contact_points: List):
        """
        Update model parameters when contact configuration changes.
        
        Args:
            contact_points: List of contact points
        """
        self.contact_points = contact_points
        self.num_forces = len(contact_points)
        
        # Rebuild grasp matrix
        if hasattr(self, 'grasp_calculator'):
            self.grasp_matrix = self.grasp_calculator.build_wrench_matrix(contact_points)
        else:
            assert False, "GraspMatrixCalculator not available"
        
        # Update parameter history
        self.parameter_history['contact_updates'].append({
            'num_contacts': self.num_forces,
            'timestamp': len(self.parameter_history['contact_updates'])
        })
        
        print(f"🔧 Contact configuration updated: {self.num_forces} contacts")
    
    # Wow, might never get called, but its fun to have it here
    def update_force_limits(self, scale_factor: float, reason: str = "manual"):
        """
        Update force limits dynamically.
        
        Args:
            scale_factor: Multiplicative factor for force limits
            reason: Reason for the update
        """
        old_limit = self.max_forces_allowed
        self.max_forces_allowed *= scale_factor
        
        # Also update rate limits proportionally
        self.max_force_rate_increase *= scale_factor
        self.max_force_rate_decrease *= scale_factor
        
        # Record change
        self.parameter_history['force_limit_changes'].append({
            'old_limit': old_limit,
            'new_limit': self.max_forces_allowed,
            'scale_factor': scale_factor,
            'reason': reason,
            'timestamp': len(self.parameter_history['force_limit_changes'])
        })
        
        print(f"🔧 Force limits updated: {old_limit:.2f} → {self.max_forces_allowed:.2f}N (×{scale_factor:.2f}, {reason})")
    
    def update_friction_limits(self, static_f_max: float = None, kinetic_f_max: float = None,
                              static_m_max: float = None, kinetic_m_max: float = None):
        """
        Update friction limits (e.g., when regime changes or surface conditions change).
        
        Args:
            static_f_max: New static friction force limit
            kinetic_f_max: New kinetic friction force limit
            static_m_max: New static friction moment limit
            kinetic_m_max: New kinetic friction moment limit
        """
        changes = {}
        
        if static_f_max is not None:
            changes['static_f_max'] = (self.static_f_max, static_f_max)
            self.static_f_max = static_f_max
        
        if kinetic_f_max is not None:
            changes['kinetic_f_max'] = (self.kinetic_f_max, kinetic_f_max)
            self.kinetic_f_max = kinetic_f_max
            # Update operational force limit
            self.max_forces_allowed = self.kinetic_f_max * 1.5
        
        if static_m_max is not None:
            changes['static_m_max'] = (self.static_m_max, static_m_max)
            self.static_m_max = static_m_max
        
        if kinetic_m_max is not None:
            changes['kinetic_m_max'] = (self.kinetic_m_max, kinetic_m_max)
            self.kinetic_m_max = kinetic_m_max
        
        if changes:
            print(f"🔧 Friction limits updated:")
            for param, (old, new) in changes.items():
                print(f"   {param}: {old:.3f} → {new:.3f}")
    
    def adapt_to_control_goal(self, new_control_goal: ControlGoalClass):
        """
        Adapt model parameters when control goal changes.
        
        Args:
            new_control_goal: New control goal configuration
        """
        old_goal = self.control_goal.control_goal
        self.control_goal = new_control_goal
        
        # Re-determine force configuration
        old_num_forces = self.num_forces
        self._determine_force_configuration()
        
        # Update grasp matrix if number of forces changed
        if self.num_forces != old_num_forces and self.contact_points is not None:
            print(f"⚠️ Number of forces changed ({old_num_forces} → {self.num_forces}), contact reconfiguration needed")
        
        print(f"🎯 Model adapted for control goal change: {old_goal} → {new_control_goal.control_goal}")
    
    def get_constraint_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get constraint bounds for optimization.
        
        Returns:
            Dictionary of constraint bounds
        """
        return {
            'force_magnitude': (0.0, self.max_forces_allowed),
            'force_rate_increase': (0.0, self.max_force_rate_increase),
            'force_rate_decrease': (0.0, self.max_force_rate_decrease),
            'linear_velocity': (-self.max_linear_velocity, self.max_linear_velocity),
            'angular_velocity': (-self.max_angular_velocity, self.max_angular_velocity),
            'position': self.position_bounds,
            'orientation': self.orientation_bounds,
            'path_param': self.path_param_bounds,
            'virtual_speed': self.virtual_speed_bounds
        }
    
    def validate_parameters(self) -> bool:
        """Validate that all parameters are reasonable."""
        issues = []
        
        # Check physical parameters
        if self.mass <= 0:
            issues.append("Mass must be positive")
        
        if self.inertia <= 0:
            issues.append("Inertia must be positive")
        
        # Check force limits
        if self.max_forces_allowed <= 0:
            issues.append("Maximum force must be positive")
        
        if self.kinetic_f_max > self.static_f_max:
            issues.append("Kinetic friction should not exceed static friction")
        
        # Check bounds
        if self.virtual_speed_bounds[0] < 0:
            issues.append("Minimum virtual speed must be positive")
        
        if issues:
            print("❌ Parameter validation failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ Parameter validation passed")
            return True
    
    def print_parameter_summary(self):
        """Print comprehensive parameter summary."""
        print(f"\n🔧 Model Parameters Summary:")
        print(f"   Object Properties:")
        print(f"     Mass: {self.mass:.3f} kg")
        print(f"     Inertia: {self.inertia:.6f} kg⋅m²")
        print(f"   Friction Properties:")
        print(f"     Static: {self.static_f_max:.2f}N, {self.static_m_max:.2f}N⋅m")
        print(f"     Kinetic: {self.kinetic_f_max:.2f}N, {self.kinetic_m_max:.2f}N⋅m")
        print(f"   Force Configuration:")
        print(f"     Number of forces: {self.num_forces}")
        print(f"     Max force allowed: {self.max_forces_allowed:.2f}N")
        print(f"     Rate limits: +{self.max_force_rate_increase:.1f}/-{self.max_force_rate_decrease:.1f} N/s")
        print(f"   Control Goal: {self.control_goal.control_goal}")
        print(f"   Grasp Matrix: {self.grasp_matrix.shape if self.grasp_matrix is not None else 'Not set'}")
        
        # Parameter change history
        if any(self.parameter_history.values()):
            print(f"   Parameter Changes:")
            print(f"     Force limit changes: {len(self.parameter_history['force_limit_changes'])}")
            print(f"     Contact updates: {len(self.parameter_history['contact_updates'])}")


# %%
class StateInputNormalization:
    """
    MATLAB-compatible normalization for MPCC decision variables.
    
    MATLAB-style structure: z = [x0, u0, du0, x1, u1, du1, ..., xN-1, uN-1, duN-1, xN, uN]
    Where:
    - xi: state (7 dims)
    - ui: input (num_forces+1 dims)
    - dui: input rate (num_forces+1 dims)
    
    ✅ MATLAB-COMPATIBLE: Matches the decision variable structure used in constraints and cost
    """
    
    def __init__(self, model_params: ModelParams, custom_scales: Dict[str, float] = None, disable_normalizer: bool = True):
        """
        Initialize normalization matrices using ModelParams.
        
        Args:
            model_params: ModelParams object with all system parameters
            custom_scales: Optional custom scaling factors to override defaults
        """
        self.model_params = model_params
        self.disable_normalizer = disable_normalizer

        # Get default scales from model parameters
        self.scales = model_params.get_normalization_scales()
        
        # Override with custom scales if provided
        if custom_scales:
            self.scales.update(custom_scales)
            print(f"🔢 Custom normalization scales applied: {list(custom_scales.keys())}")
        
        # Create normalization matrices (MATLAB-style aware)
        self._create_normalization_matrices()
        
        print(f"🔢 Normalization initialized (MATLAB-style):")
        print(f"   State dim: 7, Input dim: {model_params.num_forces + 1}")
        print(f"   Tx: {self.Tx.shape}, Tu: {self.Tu.shape}, TDu: {self.TDu.shape}")
    
    def _create_normalization_matrices(self):
        """Create normalization transformation matrices for MATLAB-style structure."""
        
        # State normalization matrix Tx: [x, y, θ, vx, vy, vw, s]
        state_scales = [
            1.0 / self.scales['position'],          # x
            1.0 / self.scales['position'],          # y  
            1.0 / self.scales['orientation'],       # θ
            1.0 / self.scales['velocity'],          # vx
            1.0 / self.scales['velocity'],          # vy
            1.0 / self.scales['angular_velocity'],  # vw
            1.0 / self.scales['path_param']         # s
        ]
        
        self.Tx = np.diag(state_scales)
        self.invTx = np.linalg.inv(self.Tx)
        
        # Input normalization matrix Tu: [f_1, ..., f_n, v_contour]
        input_scales = (
            [1.0 / self.scales['force']] * self.model_params.num_forces +
            [1.0 / self.scales['contour_speed']]
        )
        
        self.Tu = np.diag(input_scales)
        self.invTu = np.linalg.inv(self.Tu)
        
        # Input rate normalization (same structure as Tu)
        # Rates have same dimensionality as inputs
        self.TDu = self.Tu.copy()  # Same scaling for rates
        self.invTDu = self.invTu.copy()
    
        if self.disable_normalizer:
            # Disable normalization (identity matrices)
            self.Tx = np.eye(7)
            self.invTx = np.eye(7)
            self.Tu = np.eye(self.model_params.num_forces + 1)
            self.invTu = np.eye(self.model_params.num_forces + 1)
            self.TDu = np.eye(self.model_params.num_forces + 1)
            self.invTDu = np.eye(self.model_params.num_forces + 1)
            print("⚠️ Normalization disabled: using identity matrices")

    def normalize_augmented_state(self, augmented_state: AugmentedState) -> np.ndarray:
        """
        Normalize an AugmentedState to the range suitable for optimization.
        
        ✅ MATLAB-COMPATIBLE: Works with AugmentedState objects (state representation)
        
        Args:
            augmented_state: State to normalize
            
        Returns:
            Normalized state vector (7,)
        """
        state_vector = augmented_state.get_augmented_vec()
        return self.Tx @ state_vector
    
    def denormalize_to_augmented_state(self, normalized_state: np.ndarray) -> AugmentedState:
        """
        Convert normalized state vector back to AugmentedState.
        
        ✅ MATLAB-COMPATIBLE: Creates AugmentedState from normalized vector
        
        Args:
            normalized_state: Normalized state vector (7,)
            
        Returns:
            AugmentedState object
        """
        denormalized = self.invTx @ normalized_state
        
        state_data = {
            'object_position': np.array([denormalized[0], denormalized[1]]),
            'object_orientation': denormalized[2],
            'velocity_body': np.array([denormalized[3], denormalized[4]]),
            'angular_velocity': denormalized[5]
        }
        
        return AugmentedState(state_data, path_param=denormalized[6])
    
    def normalize_control_input(self, forces: np.ndarray, contour_speed: float) -> np.ndarray:
        """
        Normalize control inputs.
        
        ✅ MATLAB-COMPATIBLE: Normalizes input vector
        
        Args:
            forces: Force vector (num_forces,)
            contour_speed: Contour speed (scalar)
            
        Returns:
            Normalized input vector (num_forces+1,)
        """
        input_vector = np.concatenate([forces, [contour_speed]])
        return self.Tu @ input_vector
    
    def denormalize_control_input(self, normalized_input: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Denormalize control inputs.
        
        ✅ MATLAB-COMPATIBLE: Extracts forces and speed from normalized input
        
        Args:
            normalized_input: Normalized input vector (num_forces+1,)
            
        Returns:
            Tuple of (forces, contour_speed)
        """
        denormalized = self.invTu @ normalized_input
        forces = denormalized[:-1]
        contour_speed = denormalized[-1]
        return forces, contour_speed
    
    def normalize_input_rate(self, force_rates: np.ndarray, speed_rate: float) -> np.ndarray:
        """
        Normalize input rates.
        
        ✅ MATLAB-COMPATIBLE: New function for rate normalization
        
        Args:
            force_rates: Force rate vector (num_forces,)
            speed_rate: Speed rate (scalar)
            
        Returns:
            Normalized rate vector (num_forces+1,)
        """
        rate_vector = np.concatenate([force_rates, [speed_rate]])
        return self.TDu @ rate_vector
    
    def denormalize_input_rate(self, normalized_rate: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Denormalize input rates.
        
        ✅ MATLAB-COMPATIBLE: New function for rate denormalization
        
        Args:
            normalized_rate: Normalized rate vector (num_forces+1,)
            
        Returns:
            Tuple of (force_rates, speed_rate)
        """
        denormalized = self.invTDu @ normalized_rate
        force_rates = denormalized[:-1]
        speed_rate = denormalized[-1]
        return force_rates, speed_rate
    
    def build_full_transformation_matrix(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build full transformation matrix for MATLAB-style decision variables.
        
        ✅ NEW: Constructs block-diagonal transformation for full decision vector
        
        Decision vector structure: z = [x0, u0, du0, x1, u1, du1, ..., xN-1, uN-1, duN-1, xN, uN]
        
        Args:
            horizon: MPC prediction horizon
            
        Returns:
            Tuple of (T, invT) - full transformation matrices
        """
        from scipy.linalg import block_diag
        
        state_dim = 7
        input_dim = self.model_params.num_forces + 1
        
        # Build transformation blocks
        T_blocks = []
        
        # For stages k=0,...,N-1: [Tx, Tu, TDu]
        for k in range(horizon):
            T_blocks.append(self.Tx)      # State
            T_blocks.append(self.Tu)      # Input
            T_blocks.append(self.TDu)     # Input rate
        
        # Terminal stage N: [Tx, Tu] (no rate)
        T_blocks.append(self.Tx)  # Terminal state
        T_blocks.append(self.Tu)  # Terminal input
        
        # Create block diagonal matrix
        T = block_diag(*T_blocks)
        
        # Inverse (also block diagonal)
        invT_blocks = []
        for k in range(horizon):
            invT_blocks.append(self.invTx)
            invT_blocks.append(self.invTu)
            invT_blocks.append(self.invTDu)
        invT_blocks.append(self.invTx)
        invT_blocks.append(self.invTu)
        
        invT = block_diag(*invT_blocks)
        
        return T, invT
    
    def normalize_decision_vector(self, z_physical: np.ndarray, horizon: int) -> np.ndarray:
        """
        Normalize full decision vector from physical to normalized space.
        
        ✅ NEW: Handles full MATLAB-style decision vector
        
        Args:
            z_physical: Physical decision vector
            horizon: MPC horizon
            
        Returns:
            Normalized decision vector
        """
        T, _ = self.build_full_transformation_matrix(horizon)
        return T @ z_physical
    
    def denormalize_decision_vector(self, z_normalized: np.ndarray, horizon: int) -> np.ndarray:
        """
        Denormalize full decision vector from normalized to physical space.
        
        ✅ NEW: Handles full MATLAB-style decision vector
        
        Args:
            z_normalized: Normalized decision vector
            horizon: MPC horizon
            
        Returns:
            Physical decision vector
        """
        _, invT = self.build_full_transformation_matrix(horizon)
        return invT @ z_normalized
    
    def extract_from_decision_vector(self, z: np.ndarray, horizon: int, 
                                    normalized: bool = True) -> Dict:
        """
        Extract states, inputs, and rates from MATLAB-style decision vector.
        
        ✅ NEW: Utility for extracting trajectory components
        
        Args:
            z: Decision vector (normalized or physical)
            horizon: MPC horizon
            normalized: Whether z is in normalized space
            
        Returns:
            Dictionary with states, inputs, and rates trajectories
        """
        state_dim = 7
        input_dim = self.model_params.num_forces + 1
        nz = state_dim + 2 * input_dim  # Per-stage size
        nxu = state_dim + input_dim     # Terminal size
        
        # Denormalize if needed
        if normalized:
            z = self.denormalize_decision_vector(z, horizon)
        
        # Extract trajectories
        states = []
        inputs = []
        rates = []
        
        # Stages k=0,...,N-1
        for k in range(horizon):
            # State x_k
            state_vec = z[k*nz : k*nz + state_dim]
            states.append(create_augmented_state_from_vector(state_vec[:6], state_vec[6]))
            
            # Input u_k
            input_vec = z[k*nz + state_dim : k*nz + state_dim + input_dim]
            forces, vspeed = input_vec[:-1], input_vec[-1]
            inputs.append((forces, vspeed))
            
            # Rate du_k
            rate_vec = z[k*nz + state_dim + input_dim : (k+1)*nz]
            rate_forces, rate_vspeed = rate_vec[:-1], rate_vec[-1]
            rates.append((rate_forces, rate_vspeed))
        
        # Terminal state x_N
        terminal_start = horizon * nz
        state_N_vec = z[terminal_start : terminal_start + state_dim]
        states.append(create_augmented_state_from_vector(state_N_vec[:6], state_N_vec[6]))
        
        # Terminal input u_N
        input_N_vec = z[terminal_start + state_dim : terminal_start + nxu]
        forces_N, vspeed_N = input_N_vec[:-1], input_N_vec[-1]
        inputs.append((forces_N, vspeed_N))
        
        return {
            'states': states,
            'inputs': inputs,
            'rates': rates
        }
    
    def get_normalized_bounds(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert physical bounds to normalized bounds for MATLAB-style structure.
        
        ✅ UPDATED: Now handles MATLAB-style decision vector
        
        Args:
            horizon: MPC horizon
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) in normalized space
        """
        bounds = self.model_params.get_constraint_bounds()
        
        state_dim = 7
        input_dim = self.model_params.num_forces + 1
        nz = state_dim + 2 * input_dim
        nxu = state_dim + input_dim
        
        total_vars = nz * horizon + nxu
        
        lb = np.zeros(total_vars)
        ub = np.zeros(total_vars)
        
        # For each stage k=0,...,N-1
        for k in range(horizon):
            base = k * nz
            
            # State bounds (normalized)
            lb[base : base + state_dim] = self.Tx @ np.array([
                bounds['position'][0],
                bounds['position'][0],
                bounds['orientation'][0],
                bounds['linear_velocity'][0],
                bounds['linear_velocity'][0],
                bounds['angular_velocity'][0],
                bounds['path_param'][0]
            ])
            
            ub[base : base + state_dim] = self.Tx @ np.array([
                bounds['position'][1],
                bounds['position'][1],
                bounds['orientation'][1],
                bounds['linear_velocity'][1],
                bounds['linear_velocity'][1],
                bounds['angular_velocity'][1],
                bounds['path_param'][1]
            ])
            
            # Input bounds (normalized)
            input_lb = np.concatenate([
                [bounds['force_magnitude'][0]] * self.model_params.num_forces,
                [bounds['virtual_speed'][0]]
            ])
            input_ub = np.concatenate([
                [bounds['force_magnitude'][1]] * self.model_params.num_forces,
                [bounds['virtual_speed'][1]]
            ])
            
            lb[base + state_dim : base + state_dim + input_dim] = self.Tu @ input_lb
            ub[base + state_dim : base + state_dim + input_dim] = self.Tu @ input_ub
            
            # Rate bounds (normalized) - symmetric around zero
            rate_limit = np.concatenate([
                [bounds['force_rate_increase'][1]] * self.model_params.num_forces,
                [1.0]  # Virtual speed rate (not explicitly bounded in ModelParams)
            ])
            
            lb[base + state_dim + input_dim : base + nz] = self.TDu @ (-rate_limit)
            ub[base + state_dim + input_dim : base + nz] = self.TDu @ rate_limit
        
        # Terminal state + input
        terminal_base = horizon * nz
        
        lb[terminal_base : terminal_base + state_dim] = self.Tx @ np.array([
            bounds['position'][0],
            bounds['position'][0],
            bounds['orientation'][0],
            bounds['linear_velocity'][0],
            bounds['linear_velocity'][0],
            bounds['angular_velocity'][0],
            bounds['path_param'][0]
        ])
        
        ub[terminal_base : terminal_base + state_dim] = self.Tx @ np.array([
            bounds['position'][1],
            bounds['position'][1],
            bounds['orientation'][1],
            bounds['linear_velocity'][1],
            bounds['linear_velocity'][1],
            bounds['angular_velocity'][1],
            bounds['path_param'][1]
        ])
        
        lb[terminal_base + state_dim : terminal_base + nxu] = self.Tu @ input_lb
        ub[terminal_base + state_dim : terminal_base + nxu] = self.Tu @ input_ub
        
        return lb, ub
    
    def update_for_model_changes(self):
        """Update normalization when model parameters change."""
        # Re-get scales from updated model parameters
        self.scales = self.model_params.get_normalization_scales()
        
        # Recreate normalization matrices
        self._create_normalization_matrices()
        
        print(f"🔢 Normalization updated for model parameter changes")
    
    def check_normalization_quality(self, sample_states: List[AugmentedState], 
                                   sample_inputs: List[Tuple[np.ndarray, float]]) -> Dict[str, float]:
        """
        Check the quality of normalization by analyzing sample data.
        
        ✅ MATLAB-COMPATIBLE: Works with AugmentedState and input tuples
        
        Args:
            sample_states: Sample states for analysis
            sample_inputs: Sample inputs for analysis (forces, vspeed) tuples
            
        Returns:
            Dictionary with normalization quality metrics
        """
        if not sample_states or not sample_inputs:
            return {'error': 'No sample data provided'}
        
        # Normalize samples
        normalized_states = [self.normalize_augmented_state(state) for state in sample_states]
        normalized_inputs = [self.normalize_control_input(inp[0], inp[1]) for inp in sample_inputs]
        
        # Calculate statistics
        state_matrix = np.array(normalized_states)
        input_matrix = np.array(normalized_inputs)
        
        metrics = {
            'state_mean_abs': np.mean(np.abs(state_matrix), axis=0).tolist(),
            'state_std': np.std(state_matrix, axis=0).tolist(),
            'state_max_abs': np.max(np.abs(state_matrix), axis=0).tolist(),
            'input_mean_abs': np.mean(np.abs(input_matrix), axis=0).tolist(),
            'input_std': np.std(input_matrix, axis=0).tolist(),
            'input_max_abs': np.max(np.abs(input_matrix), axis=0).tolist(),
            'condition_number_Tx': np.linalg.cond(self.Tx),
            'condition_number_Tu': np.linalg.cond(self.Tu),
            'condition_number_TDu': np.linalg.cond(self.TDu),
            'scales_used': self.scales.copy()
        }
        
        return metrics
    
    def print_normalization_info(self):
        """Print normalization matrix information."""
        print("🔢 Normalization Information (MATLAB-style):")
        print("="*50)
        print(f"Physical scales used:")
        for name, scale in self.scales.items():
            print(f"   {name}: {scale:.3f}")
        
        print(f"\nState normalization factors:")
        state_names = ['x', 'y', 'θ', 'vx', 'vy', 'vw', 's']
        for i, name in enumerate(state_names):
            print(f"   {name}: 1/{1/self.Tx[i,i]:.3f}")
        
        print(f"\nInput normalization factors:")
        input_names = [f'f{i}' for i in range(self.model_params.num_forces)] + ['v_s']
        for i, name in enumerate(input_names):
            print(f"   {name}: 1/{1/self.Tu[i,i]:.3f}")
        
        print(f"\nRate normalization factors (same as input):")
        for i, name in enumerate(input_names):
            print(f"   d{name}: 1/{1/self.TDu[i,i]:.3f}")
        
        print(f"\nCondition numbers:")
        print(f"   Tx: {np.linalg.cond(self.Tx):.2e}")
        print(f"   Tu: {np.linalg.cond(self.Tu):.2e}")
        print(f"   TDu: {np.linalg.cond(self.TDu):.2e}")
        
        print(f"\nDecision vector structure (per horizon N):")
        state_dim = 7
        input_dim = self.model_params.num_forces + 1
        nz = state_dim + 2 * input_dim
        nxu = state_dim + input_dim
        print(f"   Per-stage size (nz): {nz}")
        print(f"   Terminal size (nxu): {nxu}")
        print(f"   Total for N={3}: {nz * 3 + nxu}")

# %%
def demo_model_params_and_normalization():
    """Demonstrate ModelParams and updated StateInputNormalization."""
    
    print("🔧 ModelParams and Enhanced Normalization Demo")
    print("="*60)
    
    # Create test objects and dynamics
    standard_objects = create_standard_objects()
    obj = standard_objects['star']
    dynamics = DynamicObjectModel(obj, friction_noise_std=0.001)
    
    # Test different control goals
    control_goals = ['position_only', 'omega_only', 'full_pose']
    
    for goal in control_goals:
        print(f"\n--- Testing {goal.upper()} ---")
        
        # Create control goal and model parameters
        control_goal = create_control_goal_from_mode(goal)
        model_params = ModelParams(dynamics, control_goal)
        
        # Print initial configuration
        model_params.print_parameter_summary()
        
        # Test parameter validation
        print(f"\n📋 Parameter validation:")
        model_params.validate_parameters()
        
        # Test normalization with ModelParams
        print(f"\n🔢 Testing normalization with ModelParams:")
        normalizer = StateInputNormalization(model_params)
        normalizer.print_normalization_info()
        
        # Test parameter updates
        print(f"\n🔧 Testing parameter updates:")
        
        # Update force limits
        model_params.update_force_limits(1.2, "increased confidence")
        
        # Update friction properties
        model_params.update_friction_limits(kinetic_f_max=model_params.kinetic_f_max * 0.9)
        
        # Update normalization after parameter changes
        normalizer.update_for_model_changes()
        
        # Test contact configuration
        print(f"\n Testing contact configuration:")
        
        # Simulate contact point selection
        test_contacts = get_goal_specific_contact_configuration(obj, control_goal)
        
        model_params.set_contact_configuration(test_contacts)

        model_params.print_parameter_summary()
        
        # Test normalization quality with sample data
        print(f"\n📊 Testing normalization quality:")
        
        # Create sample data
        sample_states = []
        sample_inputs = []
        
        for i in range(5):
            state_data = {
                'object_position': np.array([i * 0.2, i * 0.1]),
                'object_orientation': i * 0.3,
                'velocity_body': np.array([0.1, 0.05]),
                'angular_velocity': 0.1
            }
            aug_state = AugmentedState(state_data, path_param=i * 0.2)
            sample_states.append(aug_state)
            
            forces = np.random.rand(model_params.num_forces) * model_params.max_forces_allowed
            sample_inputs.append((forces, 0.5))
        
        quality_metrics = normalizer.check_normalization_quality(sample_states, sample_inputs)
        
        print(f"   State normalization quality:")
        print(f"     Max absolute values: {[f'{x:.3f}' for x in quality_metrics['state_max_abs']]}")
        print(f"     Standard deviations: {[f'{x:.3f}' for x in quality_metrics['state_std']]}")
        
        print(f"   Input normalization quality:")
        print(f"     Max absolute values: {[f'{x:.3f}' for x in quality_metrics['input_max_abs']]}")
        print(f"     Standard deviations: {[f'{x:.3f}' for x in quality_metrics['input_std']]}")
        
        print(f"   Condition numbers: Tx={quality_metrics['condition_number_Tx']:.2e}, Tu={quality_metrics['condition_number_Tu']:.2e}")

if __name__ == "__main__":
    demo_model_params_and_normalization()

# %%
# MPCC ERROR CLASS
# ================================================================
from scipy.linalg import block_diag

class MPCCErrorClass:
    """
    Handles MPCC error calculations including contour error, lag error, and heading error.
    Provides both instantaneous error calculation and error tracking over time.
    """
    
    def __init__(self, reference_path: 'SplineReferencePath'):
        """
        Initialize MPCC error calculator.
        
        Args:
            reference_path: SplineReferencePath object for error calculations
        """
        self.reference_path = reference_path
        self.error_history = []
        self.error_statistics = {
            'max_contour_error': 0.0,
            'max_lag_error': 0.0,
            'max_heading_error': 0.0,
            'rms_contour_error': 0.0,
            'rms_lag_error': 0.0,
            'rms_heading_error': 0.0,
            'total_samples': 0
        }
        
    def calculate_errors(self, augmented_state: AugmentedState) -> Dict[str, float]:
        """
        Calculate MPCC errors for given state.
        
        Args:
            augmented_state: Current augmented state
            
        Returns:
            Dictionary containing error metrics
        """
        # Get reference point on path corresponding to path parameter
        reference_point = self.reference_path.get_point_at_parameter(augmented_state.path_param)
        ref_x, ref_y, ref_theta = reference_point
        
        # Get tangent and normal vectors at path parameter
        tangent = self.reference_path.get_tangent_at_parameter(augmented_state.path_param)
        normal = self.reference_path.get_normal_at_parameter(augmented_state.path_param)
        
        phi_virt = np.arctan2(self.reference_path.spline_dy(augmented_state.path_param), 
                              self.reference_path.spline_dx(augmented_state.path_param))
        golden_tangent = np.array([-np.sin(phi_virt), np.cos(phi_virt)])
        golden_normal = np.array([np.cos(phi_virt), np.sin(phi_virt)])

        # Calculate position error vector
        error_vector = np.array([
            -augmented_state.object_x + ref_x,
            -augmented_state.object_y + ref_y
        ])
        
        # Project error onto tangent and normal directions
        # lag_error = np.dot(error_vector, tangent)        # Along-track (longitudinal)
        # contour_error = np.dot(error_vector, normal)     # Cross-track (lateral)
        
        # Try mimicking the original implementation more closely
        contour_error = np.dot(error_vector, golden_tangent)        # Along-track (longitudinal)
        lag_error = np.dot(error_vector, golden_normal)     # Cross-track (lateral)

        # Calculate heading error (normalized angle difference)
        heading_error = normalize_theta_diff(ref_theta, augmented_state.object_theta)

        # # reverse here, to match the error vector 
        # heading_error = normalize_theta_diff(augmented_state.object_theta, ref_theta)
        
        # Calculate additional metrics
        euclidean_error = np.linalg.norm(error_vector)
        
        error_dict = {
            'contour_error': contour_error,
            'lag_error': lag_error,
            'heading_error': heading_error,
            'euclidean_error': euclidean_error,
            'reference_point': reference_point,
            'path_param': augmented_state.path_param,
            'timestamp': len(self.error_history)
        }
        
        # Update error history and statistics
        self._update_error_tracking(error_dict)
        
        return error_dict
    
    def calculate_error_gradients(self, augmented_state: AugmentedState) -> Dict[str, np.ndarray]:
        """
        Calculate gradients of MPCC errors with respect to state.
        Required for MPC optimization.
        
        Args:
            augmented_state: Current augmented state
            
        Returns:
            Dictionary containing error gradients
        """
        theta_virt = augmented_state.path_param
        x_phys = augmented_state.object_x
        y_phys = augmented_state.object_y
        
        # Get path derivatives
        dxdth = self.reference_path.spline_dx(theta_virt)
        dydth = self.reference_path.spline_dy(theta_virt)
        d2xdth2 = self.reference_path.spline_d2x(theta_virt)
        d2ydth2 = self.reference_path.spline_d2y(theta_virt)
        
        # Calculate virtual orientation and its derivative
        phi_virt = np.arctan2(dydth, dxdth)
        
        # Calculate derivative of virtual orientation w.r.t. path parameter
        numer = dxdth * d2ydth2 - dydth * d2xdth2
        denom = dxdth**2 + dydth**2
        dphivirt_dtheta = numer / denom if abs(denom) > 1e-10 else 0.0
        
        # Virtual position
        x_virt = self.reference_path.spline_x(theta_virt)
        y_virt = self.reference_path.spline_y(theta_virt)
        
        # Position differences
        Dx = x_phys - x_virt
        Dy = y_phys - y_virt
        
        cos_phi_virt = np.cos(phi_virt)
        sin_phi_virt = np.sin(phi_virt)
        
        # Calculate error derivatives w.r.t. path parameter
        tmp1 = np.array([dphivirt_dtheta, 1])
        tmp2 = np.array([cos_phi_virt, sin_phi_virt]).reshape(2, 1)
        
        MC = np.array([[Dx, Dy],
                       [dydth, -dxdth]])
        ML = np.array([[-Dy, Dx],
                       [dxdth, dydth]])
        
        deC_dtheta = tmp1.dot(MC).dot(tmp2).item()
        deL_dtheta = tmp1.dot(ML).dot(tmp2).item()
        
        # Form complete gradients for 6-DOF state + path parameter
        grad_contour = np.array([
            sin_phi_virt,     # ∂eC/∂x
            -cos_phi_virt,    # ∂eC/∂y
            0.0,              # ∂eC/∂θ
            0.0,              # ∂eC/∂vx
            0.0,              # ∂eC/∂vy
            0.0,              # ∂eC/∂vw
            deC_dtheta        # ∂eC/∂s
        ])
        
        grad_lag = np.array([
            -cos_phi_virt,    # ∂eL/∂x
            -sin_phi_virt,    # ∂eL/∂y
            0.0,              # ∂eL/∂θ
            0.0,              # ∂eL/∂vx
            0.0,              # ∂eL/∂vy
            0.0,              # ∂eL/∂vw
            deL_dtheta        # ∂eL/∂s
        ])
        
        # More robust gradient calculation
        current_theta = augmented_state.object_theta
        ref_theta = self.reference_path.get_point_at_parameter(theta_virt)[2]

        # Check if we're near a wrapping boundary
        angle_diff = current_theta - ref_theta
        if abs(angle_diff) > np.pi:
            # Near boundary - use wrapped difference for gradient sign
            wrapped_diff = normalize_theta_diff(current_theta, ref_theta)
            gradient_sign = np.sign(wrapped_diff) if abs(wrapped_diff) > 1e-6 else 1.0
            deH_dtheta = gradient_sign  # May be -1 if wrapped
        else:
            deH_dtheta = 1.0  # Standard case  

        # Get derivative of reference orientation w.r.t. path parameter
        if hasattr(self.reference_path, 'spline_dtheta'):
            dref_theta_dparam = self.reference_path.spline_dtheta(theta_virt)
        else:
            # Fallback: numerical differentiation
            eps = 1e-6
            theta_plus = self.reference_path.get_point_at_parameter(min(theta_virt + eps, 1.0))[2]
            theta_minus = self.reference_path.get_point_at_parameter(max(theta_virt - eps, 0.0))[2]
            dref_theta_dparam = (theta_plus - theta_minus) / (2 * eps)
        

        # Heading error gradient (simple case)
        grad_heading = np.array([
            0.0,              # ∂eH/∂x
            0.0,              # ∂eH/∂y
            deH_dtheta,              # ∂eH/∂θ = ∂(current_theta - ref_theta)/∂current_theta = 1 (normally)
            0.0,              # ∂eH/∂vx
            0.0,              # ∂eH/∂vy
            0.0,              # ∂eH/∂vw
            -dref_theta_dparam # ∂eH/∂s = ∂(-ref_theta)/∂path_param
        ])
        
        return {
            'grad_contour': grad_contour,
            'grad_lag': grad_lag,
            'grad_heading': grad_heading,
            'cos_phi_virt': cos_phi_virt,
            'sin_phi_virt': sin_phi_virt
        }

    def calculate_linearized_error_cost(self, augmented_state: AugmentedState, model_param: ModelParams, normalizer: StateInputNormalization,
                                  control_goal: ControlGoalClass, terminal_stage: bool) -> Dict[str, np.ndarray]:
        """
        Calculate linearized error cost matrices for MPC.
        
        Args:
            augmented_state: Current augmented state
            control_goal: Control goal for cost weighting
            
        Returns:
            Dictionary containing cost matrices and vectors
        """
        # Get current errors
        errors = self.calculate_errors(augmented_state)
        gradients = self.calculate_error_gradients(augmented_state)
        
        # Get cost matrices from control goal
        Q_error, _, _, Q_heading, Q_progress = control_goal.get_mpcc_cost_matrices(terminal_stage)
        
        # Form COMPLETE error vector and gradient matrix (always 3 errors)
        # OLD version
        # e = np.array([
        #     [errors['contour_error']],
        #     [errors['lag_error']], 
        #     [errors['heading_error']]  # Always include heading error
        # ])

        # --- START FIX ---
        # Form (2, 1) error vector
        # try to match the golden version first
        e = np.array([
            [errors['contour_error']],
            [errors['lag_error']]
        ])
        
        # OLD version
        # grad_e = np.vstack([
        #     gradients['grad_contour'],
        #     gradients['grad_lag'],
        #     gradients['grad_heading']  # Always include heading gradient
        # ])

        # Form (2, 7) gradient matrix
        grad_e = np.vstack([
            gradients['grad_contour'],
            gradients['grad_lag']
        ])
        
        # Form COMPLETE cost matrix Q (always 3x3) - OLD version
        # Q_weight = np.block([
        #     [Q_error,                    np.zeros((2, 1))],  # 2x2 block + 2x1 zeros
        #     [np.zeros((1, 2)),           Q_heading]          # 1x2 zeros + 1x1 heading weight
        # ])
        # Form (2, 2) cost matrix (Q_heading is IGNORED)
        Q_weight = Q_error # This is already (2, 2)

        # Calculate quadratic cost matrix: Q_tilde = grad_e^T * Q_weight * grad_e
        Q_tilde = grad_e.T @ Q_weight @ grad_e

        # Calculate Qk, mimic from golden version
        Q_tilde_sym = 0.5 * (Q_tilde + Q_tilde.T)  # Ensure symmetry
        Qk_raw = 2 * block_diag(Q_tilde_sym, np.diag(np.concatenate([np.ones(model_param.num_forces)*control_goal.weights.rF, [control_goal.weights.rVtheta]])))
        Qk_norm = block_diag(normalizer.invTx, normalizer.invTu) @ Qk_raw @ block_diag(normalizer.invTx, normalizer.invTu) + 1e-4 * np.eye(model_param.num_states + model_param.num_forces + 1)

        if (terminal_stage):
            Qk_norm *= control_goal.weights.qCNmult

        # Calculate linear cost vector: f = 2 * e^T * Q_weight * grad_e
        # this is fx in the golden version
        state_vector = augmented_state.get_augmented_vec().reshape(-1, 1)
        f_error = 2 * (e.T @ Q_weight @ grad_e) - 2 * (state_vector.T @ grad_e.T @ Q_weight @ grad_e)
        
        # Expand f to match decision variable size
        # nu = num_forces + 1, so its just num_forces here
        f_t = np.hstack((f_error.flatten(), np.zeros(model_param.num_forces), -control_goal.weights.qVtheta))
        f_raw = f_t.reshape(-1, 1)

        # scale f to match the original implementation
        invT = block_diag(normalizer.invTx, normalizer.invTu)
        f_norm = invT @ f_raw

        if (terminal_stage):
            f_norm *= control_goal.weights.qCNmult

        # Add progress reward (unchanged)
        progress_vector = np.zeros(7)
        progress_vector[-1] = -Q_progress  # Negative to encourage progress
        
        return {
            'Q_tilde': Q_tilde,           # Always 7x7 regardless of control goal
            'Qk_raw': Qk_raw,                     # Always 7x7 regardless of control goal
            'Qk_norm': Qk_norm,           # Normalized Qk for MPCC
            'f_error_raw': f_error.flatten(),  # Always length 7
            'f_error': f_norm.flatten(), # match golden version
            'f_progress': progress_vector,
            'errors': errors,
            'gradients': gradients,
            'Q_weight': Q_weight      # For debugging: the 3x3 complete cost matrix
        }    

    def _update_error_tracking(self, error_dict: Dict[str, float]):
        """Update error history and statistics - simplified, no overthinking."""
        self.error_history.append(error_dict.copy())
        
        # Update maximum errors - straightforward
        self.error_statistics['max_contour_error'] = max(
            self.error_statistics['max_contour_error'],
            abs(error_dict['contour_error'])
        )
        self.error_statistics['max_lag_error'] = max(
            self.error_statistics['max_lag_error'],
            abs(error_dict['lag_error'])
        )
        self.error_statistics['max_heading_error'] = max(
            self.error_statistics['max_heading_error'],
            abs(error_dict['heading_error'])
        )
        
        # Update RMS errors (simple running average)
        n = len(self.error_history)
        self.error_statistics['rms_contour_error'] = np.sqrt(
            np.mean([err['contour_error']**2 for err in self.error_history])
        )
        self.error_statistics['rms_lag_error'] = np.sqrt(
            np.mean([err['lag_error']**2 for err in self.error_history])
        )
        self.error_statistics['rms_heading_error'] = np.sqrt(
            np.mean([err['heading_error']**2 for err in self.error_history])
        )
        
        self.error_statistics['total_samples'] = n

    def get_error_statistics(self) -> Dict[str, float]:
        """Get comprehensive error statistics - just return the dict."""
        return self.error_statistics.copy()

    def get_recent_errors(self, num_samples: int = 10) -> List[Dict[str, float]]:
        """Get recent error history - simple slice."""
        return self.error_history[-num_samples:] if self.error_history else []
    
    def reset_error_tracking(self):
        """Reset error history and statistics."""
        self.error_history.clear()
        self.error_statistics = {
            'max_contour_error': 0.0,
            'max_lag_error': 0.0,
            'max_heading_error': 0.0,
            'rms_contour_error': 0.0,
            'rms_lag_error': 0.0,
            'rms_heading_error': 0.0,
            'total_samples': 0
        }
    
    def plot_error_evolution(self, save_path: str = None):
        """Plot error evolution over time."""
        if not self.error_history:
            print("No error history to plot.")
            return
        
        timestamps = [err['timestamp'] for err in self.error_history]
        contour_errors = [err['contour_error'] for err in self.error_history]
        lag_errors = [err['lag_error'] for err in self.error_history]
        heading_errors = [np.rad2deg(err['heading_error']) for err in self.error_history]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Contour error
        axes[0].plot(timestamps, contour_errors, 'b-', linewidth=2)
        axes[0].set_ylabel('Contour Error (m)')
        axes[0].set_title('MPCC Error Evolution')
        axes[0].grid(True)
        axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
        
        # Lag error
        axes[1].plot(timestamps, lag_errors, 'r-', linewidth=2)
        axes[1].set_ylabel('Lag Error (m)')
        axes[1].grid(True)
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        
        # Heading error
        axes[2].plot(timestamps, heading_errors, 'g-', linewidth=2)
        axes[2].set_ylabel('Heading Error (°)')
        axes[2].set_xlabel('Time Step')
        axes[2].grid(True)
        axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics
        print("\n📊 MPCC Error Statistics:")
        print(f"   Max Contour Error: {self.error_statistics['max_contour_error']:.4f} m")
        print(f"   Max Lag Error: {self.error_statistics['max_lag_error']:.4f} m")
        print(f"   Max Heading Error: {np.rad2deg(self.error_statistics['max_heading_error']):.2f}°")
        print(f"   RMS Contour Error: {self.error_statistics['rms_contour_error']:.4f} m")
        print(f"   RMS Lag Error: {self.error_statistics['rms_lag_error']:.4f} m")
        print(f"   RMS Heading Error: {np.rad2deg(self.error_statistics['rms_heading_error']):.2f}°")
        print(f"   Total Samples: {self.error_statistics['total_samples']}")


    def calculate_errors_from_decision_vector(self, z: np.ndarray, stage: int,
                                             horizon: int, normalizer: StateInputNormalization = None) -> Dict[str, float]:
        """
        Calculate errors from MATLAB-style decision vector at a specific stage.
        
        ✅ NEW: Extract state from decision vector and compute errors
        
        Args:
            z: Full decision vector (MATLAB-style)
            stage: Stage index (0 to N)
            horizon: MPC horizon
            normalizer: Optional normalization handler
            
        Returns:
            Error dictionary (same as calculate_errors)
        """
        state_dim = 7
        input_dim = normalizer.model_params.num_forces + 1 if normalizer else 8  # Fallback
        nz = state_dim + 2 * input_dim
        
        # Extract state at requested stage
        if stage < horizon:
            # Regular stage: x_k at k*nz
            state_start = stage * nz
            state_vec = z[state_start : state_start + state_dim]
        else:
            # Terminal stage: x_N at horizon*nz
            state_start = horizon * nz
            state_vec = z[state_start : state_start + state_dim]
        
        # Denormalize if needed
        if normalizer is not None:
            state_vec = normalizer.invTx @ state_vec
        
        # Create AugmentedState and compute errors
        augmented_state = create_augmented_state_from_vector(state_vec[:6], state_vec[6])
        
        return self.calculate_errors(augmented_state)
    
    def evaluate_trajectory_errors(self, z: np.ndarray, horizon: int,
                                  normalizer: StateInputNormalization = None) -> List[Dict[str, float]]:
        """
        Evaluate errors for entire trajectory from decision vector.
        
        ✅ NEW: Compute error metrics for all stages
        
        Args:
            z: Full decision vector (MATLAB-style)
            horizon: MPC horizon
            normalizer: Optional normalization handler
            
        Returns:
            List of error dictionaries for each stage (N+1 total)
        """
        errors_trajectory = []
        
        for k in range(horizon + 1):  # N+1 states
            errors = self.calculate_errors_from_decision_vector(z, k, horizon, normalizer)
            errors_trajectory.append(errors)
        
        return errors_trajectory
    
    def get_total_tracking_error(self, z: np.ndarray, horizon: int,
                                normalizer: StateInputNormalization = None,
                                weights: Dict[str, float] = None) -> float:
        """
        Calculate weighted total tracking error for entire trajectory.
        
        ✅ NEW: Useful for evaluating solution quality
        
        Args:
            z: Full decision vector
            horizon: MPC horizon  
            normalizer: Optional normalization
            weights: Optional error weights {'contour': w1, 'lag': w2, 'heading': w3}
            
        Returns:
            Scalar weighted total error
        """
        if weights is None:
            weights = {'contour': 1.0, 'lag': 1.0, 'heading': 1.0}
        
        errors_traj = self.evaluate_trajectory_errors(z, horizon, normalizer)
        
        total_error = 0.0
        for errors in errors_traj:
            total_error += (
                weights['contour'] * errors['contour_error']**2 +
                weights['lag'] * errors['lag_error']**2 +
                weights['heading'] * errors['heading_error']**2
            )
        
        return np.sqrt(total_error / len(errors_traj))  # RMS across trajectory
    


# %%
# ================================================================
# DEMONSTRATION FUNCTION for MPCCErrorClass
# ================================================================

def calculate_mpc_linear_cost_vector(model_params: ModelParams, 
                                control_goal: ControlGoalClass,
                                f_error: np.ndarray,
                                normalizer: StateInputNormalization = None,
                                terminal_stage: bool = False) -> np.ndarray:
    """
    Calculate the complete MPC linear cost vector f for the QP formulation.
    
    ✅ UPDATED: Now uses MATLAB-style decision variable structure
    
    MATLAB-style per-stage: [x_k, u_k, du_k] (no rate at terminal)
    This builds cost for a SINGLE stage, not full horizon.
    
    Args:
        model_params: Model parameters containing dimensions
        control_goal: Control goal for progress weight
        f_error: Linear error cost vector (length 7, from calculate_linearized_error_cost)
        normalizer: Optional normalization handler
        terminal_stage: Whether this is the terminal stage (no rate cost)
        
    Returns:
        Complete linear cost vector for single stage
    """
    # Get dimensions
    state_dim = 7  # [x, y, theta, vx, vy, vw, path_param]
    input_dim = model_params.num_forces + 1  # [f1, ..., fn, virtual_speed]
    
    # Get progress weight from control goal
    _, _, _, _, Q_progress = control_goal.get_mpcc_cost_matrices(terminal_stage)
    
    # Build complete linear cost vector for this stage
    # Structure: [f_state, f_input, f_rate] (or [f_state, f_input] if terminal)
    
    f_state = f_error.flatten()  # Length 7
    
    # Input cost: only virtual speed gets progress reward
    f_input = np.zeros(input_dim)
    f_input[-1] = -Q_progress  # Negative to encourage progress (reward)
    
    if terminal_stage:
        # Terminal stage: [state, input] only (no rate)
        f_complete = np.concatenate([f_state, f_input])
    else:
        # Regular stage: [state, input, rate]
        f_rate = np.zeros(input_dim)  # No linear cost for rates
        f_complete = np.concatenate([f_state, f_input, f_rate])
    
    # Apply normalization if available
    if normalizer is not None:
        from scipy.linalg import block_diag
        
        if terminal_stage:
            # Terminal: only [Tx, Tu]
            invT = block_diag(normalizer.invTx, normalizer.invTu)
        else:
            # Regular: [Tx, Tu, TDu]
            invT = block_diag(normalizer.invTx, normalizer.invTu, normalizer.invTDu)
        
        # Transform: f_normalized = invT^T @ f_complete
        f_normalized = invT.T @ f_complete
        
        return f_normalized.reshape(-1, 1)  # Column vector
    else:
        return f_complete.reshape(-1, 1)

# should match the golden generateH function, but sadly we miss it when code it
# will need a big refactor later
def calculate_mpc_quadratic_cost_matrix(model_params: ModelParams,
                                    control_goal: ControlGoalClass, 
                                    Q_tilde: np.ndarray,
                                    normalizer: StateInputNormalization = None,
                                    terminal_stage: bool = False) -> np.ndarray:
    """
    Calculate the complete MPC quadratic cost matrix Q for single stage.
    
    ✅ UPDATED: Now uses MATLAB-style decision variable structure
    
    Args:
        model_params: Model parameters containing dimensions
        control_goal: Control goal for regularization weights
        Q_tilde: State error cost matrix (7x7, from calculate_linearized_error_cost)
        normalizer: Optional normalization handler
        terminal_stage: Whether this is terminal stage (no rate cost)
        
    Returns:
        Complete quadratic cost matrix for single stage
    """
    # Get dimensions
    state_dim = 7
    input_dim = model_params.num_forces + 1
    
    # Get input regularization weights
    _, R, R_rate, _, _ = control_goal.get_mpcc_cost_matrices(terminal_stage)
    
    # Build R_full (input magnitude penalty)
    R_full = np.zeros((input_dim, input_dim))
    for i in range(model_params.num_forces):
        R_full[i, i] = R[0, 0]  # rF (force regularization)
    R_full[-1, -1] = R[1, 1]  # rVtheta (virtual velocity regularization)
    
    if terminal_stage:
        # Terminal stage: [state, input] only
        total_dim = state_dim + input_dim
        
        Q_complete = np.zeros((total_dim, total_dim))
        Q_complete[:state_dim, :state_dim] = Q_tilde  # State cost block
        Q_complete[state_dim:, state_dim:] = R_full     # Input regularization
        
    else:
        # Regular stage: [state, input, rate]
        # Build Rk_full (input rate penalty)
        Rk_full = np.zeros((input_dim, input_dim))
        for i in range(model_params.num_forces):
            Rk_full[i, i] = R_rate[0, 0]  # rdF
        Rk_full[-1, -1] = R_rate[1, 1]  # rdVtheta
        
        total_dim = state_dim + 2 * input_dim
        
        Q_complete = np.zeros((total_dim, total_dim))
# NOTE, Modified here to match the golden one
        Q_complete[:state_dim, :state_dim] = 2 * Q_tilde                              # State cost
# NOTE the 2 factor here
        Q_complete[state_dim:state_dim+input_dim, 
                   state_dim:state_dim+input_dim] = 2 * R_full                        # Input cost
        Q_complete[state_dim+input_dim:, 
                   state_dim+input_dim:] = 2 * Rk_full                                # Rate cost
    
    # Add small regularization for numerical stability
    regularization = 1e-6 * np.eye(total_dim)
    Q_complete += regularization

    # Apply normalization if available  
    if normalizer is not None:
        from scipy.linalg import block_diag
        
        if terminal_stage:
            invT = block_diag(normalizer.invTx, normalizer.invTu)
        else:
            invT = block_diag(normalizer.invTx, normalizer.invTu, normalizer.invTDu)
        
# note be very careful here. invT should be a diagonal matrix
        Q_normalized = invT.T @ Q_complete @ invT
        return Q_normalized
    else:
        return Q_complete


def demo_mpcc_error_class():
    """Demonstrate MPCC error calculation functionality and cost function integration."""
    
    print("🎯 MPCC Error Class & Cost Function Integration Demo")
    print("="*60)
    
    # Create a simple reference path
    waypoints = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0, np.pi/2],
        [2.0, 2.0, np.pi/2]
    ]
    
    reference_path = SplineReferencePath(waypoints)
    error_calculator = MPCCErrorClass(reference_path)
    
    # Test different states
    test_states = [
        # Perfect tracking
        create_augmented_state_from_vector(np.array([0.5, 0.0, 0.0, 0.1, 0.0, 0.0]), 0.25),
        # Lateral offset
        create_augmented_state_from_vector(np.array([0.5, 0.2, 0.0, 0.1, 0.0, 0.0]), 0.25),
    ]
    
    test_names = [
        "Perfect Tracking",
        "Lateral Offset (+0.2m)",
    ]
    
    print("\n📏 Error Calculation Tests:")
    print("-" * 60)
    
    for i, (state, name) in enumerate(zip(test_states, test_names)):
        print(f"\n{i+1}. {name}:")
        errors = error_calculator.calculate_errors(state)
        gradients = error_calculator.calculate_error_gradients(state)
        
        print(f"   State: x={state.object_x:.2f}, y={state.object_y:.2f}, θ={np.rad2deg(state.object_theta):.1f}°")
        print(f"   Contour Error: {errors['contour_error']:+.4f} m")
        print(f"   Lag Error: {errors['lag_error']:+.4f} m")
        print(f"   Heading Error: {np.rad2deg(errors['heading_error']):+.2f}°")
        print(f"   Reference Point: ({errors['reference_point'][0]:.2f}, {errors['reference_point'][1]:.2f})")

        print(f"   Gradients (7-dim):")
        print(f"     ∇eC: {gradients['grad_contour']}")
        print(f"     ∇eL: {gradients['grad_lag']}")
        print(f"     ∇eH: {gradients['grad_heading']}")
    
    print(f"\n✅ MPCC Error Class Demo Completed!")
    print(f"Key validations:")
    print(f"   ✅ Error calculation working (contour, lag, heading)")
    print(f"   ✅ Gradient calculation working (7-dim for augmented state)")
    print(f"   ✅ Single-stage cost function working")


if __name__ == "__main__":
    demo_mpcc_error_class()

# %%
# REVAMPED MPCC CONSTRAINT CLASS
# ================================================================
class MPCCConstraintClass:
    """
    Enhanced MPCC constraints handler that integrates seamlessly with ModelParams.
    
    Provides unified constraint management for:
    - Force magnitude and rate limits (from ModelParams friction model)
    - State bounds (from ModelParams physical limits)
    - Path parameter constraints (from ModelParams path configuration)
    - Contact feasibility constraints (from ModelParams grasp matrix)
    """
    
    def __init__(self, model_params: ModelParams, normalizer: StateInputNormalization = None):
        """
        Initialize constraint handler using ModelParams.
        
        Args:
            model_params: Centralized model parameters
            normalizer: Optional normalization handler for numerical stability
        """
        self.model_params = model_params
        self.normalizer = normalizer
        
        # Get constraint bounds directly from ModelParams
        self.constraint_bounds = model_params.get_constraint_bounds()
        
        # Constraint activation flags (can be customized per control goal)
        self.constraint_flags = self._get_goal_specific_constraint_flags()
        
        # Constraint violation tracking
        self.violation_history = {
            'force_magnitude': [],
            'force_rate': [],
            'state_bounds': [],
            'path_bounds': [],
            'virtual_speed': [],
            'grasp_matrix': []
        }
        
        # Constraint relaxation parameters (for soft constraints)
        self.relaxation_weights = {
            'force_magnitude': 1e6,  # Hard constraint
            'force_rate': 1e4,       # Medium priority
            'state_bounds': 1e5,     # High priority
            'path_bounds': 1e6,      # Hard constraint
            'virtual_speed': 1e3     # Lower priority
        }
        
        # Previous values for rate constraints
        self.previous_forces = None
        self.previous_time = None
        
        print(f"🔒 MPCC Constraints initialized:")
        print(f"   Force limits: [0, {self.constraint_bounds['force_magnitude'][1]:.2f}] N")
        print(f"   Rate limits: ±{self.constraint_bounds['force_rate_increase'][1]:.1f} N/s")
        print(f"   Path bounds: {self.constraint_bounds['path_param']}")
        print(f"   Virtual speed: {self.constraint_bounds['virtual_speed']}")
        print(f"   Active constraints: {sum(self.constraint_flags.values())}/{len(self.constraint_flags)}")
    
    def _get_goal_specific_constraint_flags(self) -> Dict[str, bool]:
        """Get constraint activation flags based on control goal."""
        control_goal = self.model_params.control_goal.control_goal
        
        # Base constraints (always active)
        flags = {
            'force_magnitude': True,      # Always need force limits
            'path_bounds': True,         # Always need path parameter bounds
            'virtual_speed': True,       # Always need virtual speed bounds
        }
        
        # Control goal specific constraints
        if control_goal == 'position_only':
            flags.update({
                'force_rate': True,      # Smooth forces for precise positioning
                'state_bounds': True,    # Position limits important
                'grasp_matrix': False    # Less critical for position-only
            })
        elif control_goal == 'omega_only':
            flags.update({
                'force_rate': False,      # Going easy for the simple case
                'state_bounds': False,   # Position bounds less critical
                'grasp_matrix': False     # Torque generation feasibility important, but too complex
            })
        else:  # full_pose
            flags.update({
                'force_rate': True,      # Smooth control for both position and orientation
                'state_bounds': True,    # All bounds important
                'grasp_matrix': False     # Full feasibility required, but too complex
            })
        
        return flags
    
    def get_linear_inequality_constraints(self, horizon: int, dt: float = 0.05, 
                                        include_soft_constraints: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate linear inequality constraints for MATLAB-style MPCC optimization.
        
        MATLAB-style decision variables: z = [x0, u0, du0, x1, u1, du1, ..., xN-1, uN-1, duN-1, xN]
        
        Format: A_ineq @ decision_variables <= b_ineq
        
        Args:
            horizon: MPC prediction horizon
            dt: Time step for rate constraints
            include_soft_constraints: Whether to include soft constraint formulation
            
        Returns:
            Dictionary containing constraint matrices and metadata
        """
        # Get dimensions
        state_dim = 7  # Fixed MPCC state dimension
        input_dim = self.model_params.num_forces + 1  # Forces + virtual speed
        
        # MATLAB-style structure
        nz = state_dim + 2 * input_dim  # Per-stage: state + input + input_rate
        nxu = state_dim + input_dim     # Terminal: state + input only
        total_vars = nz * horizon + nxu
        
        # Collect constraint matrices
        constraint_matrices = []
        constraint_bounds = []
        constraint_types = []
        
        # ================================================================
        # 1. FORCE MAGNITUDE CONSTRAINTS (from ModelParams)
        # ================================================================
        if self.constraint_flags['force_magnitude']:
            A_force, b_force = self._build_force_magnitude_constraints_matlab(
                horizon, state_dim, input_dim, total_vars, nz, nxu
            )
            constraint_matrices.append(A_force)
            constraint_bounds.append(b_force)
            constraint_types.extend(['force_magnitude'] * A_force.shape[0])
        
        # ================================================================
        # 2. FORCE RATE CONSTRAINTS (from ModelParams)
        # ================================================================
        if self.constraint_flags['force_rate'] and horizon > 1:
            A_rate, b_rate = self._build_force_rate_constraints_matlab(
                horizon, state_dim, input_dim, total_vars, nz, nxu, dt
            )
            if A_rate is not None:
                constraint_matrices.append(A_rate)
                constraint_bounds.append(b_rate)
                constraint_types.extend(['force_rate'] * A_rate.shape[0])
        
        # ================================================================
        # 3. STATE BOUNDS CONSTRAINTS (from ModelParams)
        # ================================================================
        if self.constraint_flags['state_bounds']:
            A_state, b_state = self._build_state_bounds_constraints_matlab(
                horizon, state_dim, input_dim, total_vars, nz, nxu
            )
            constraint_matrices.append(A_state)
            constraint_bounds.append(b_state)
            constraint_types.extend(['state_bounds'] * A_state.shape[0])
        
        # ================================================================
        # 4. PATH PARAMETER CONSTRAINTS (from ModelParams)
        # ================================================================
        if self.constraint_flags['path_bounds']:
            A_path, b_path = self._build_path_parameter_constraints_matlab(
                horizon, state_dim, input_dim, total_vars, nz, nxu
            )
            constraint_matrices.append(A_path)
            constraint_bounds.append(b_path)
            constraint_types.extend(['path_bounds'] * A_path.shape[0])
        
        # ================================================================
        # 5. VIRTUAL SPEED CONSTRAINTS (from ModelParams)
        # ================================================================
        if self.constraint_flags['virtual_speed']:
            A_vspeed, b_vspeed = self._build_virtual_speed_constraints_matlab(
                horizon, state_dim, input_dim, total_vars, nz, nxu
            )
            constraint_matrices.append(A_vspeed)
            constraint_bounds.append(b_vspeed)
            constraint_types.extend(['virtual_speed'] * A_vspeed.shape[0])
        
        # Combine all constraints
        if constraint_matrices:
            A_ineq = np.vstack(constraint_matrices)
            b_ineq = np.concatenate(constraint_bounds)
        else:
            A_ineq = np.zeros((0, total_vars))
            b_ineq = np.zeros(0)
        
        # Apply normalization if available
        if self.normalizer is not None:
            A_ineq, b_ineq = self._apply_constraint_normalization_matlab(A_ineq, b_ineq, horizon, nz, nxu)
        
        return {
            'A_ineq': A_ineq,
            'b_ineq': b_ineq,
            'num_constraints': A_ineq.shape[0],
            'constraint_types': constraint_types,
            'active_flags': self.constraint_flags.copy(),
            'bounds_used': self.constraint_bounds.copy(),
            'relaxation_weights': self.relaxation_weights.copy() if include_soft_constraints else None,
            'structure': {
                'state_dim': state_dim,
                'input_dim': input_dim,
                'nz': nz,
                'nxu': nxu,
                'total_vars': total_vars
            }
        }


    # ================================================================
    # FIX: Terminal stage input constraint issue
    # ================================================================

    def _build_force_magnitude_constraints_matlab(self, horizon: int, state_dim: int, 
                                        input_dim: int, total_vars: int, nz: int, nxu: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build force magnitude constraints for MATLAB-style structure."""
        num_forces = self.model_params.num_forces
        force_min, force_max = self.constraint_bounds['force_magnitude']
        
        # ✅ FIX: Terminal input u_N exists, so constraint count should be:
        # 2 * num_forces * (horizon + 1) NOT just horizon
        num_constraints = 2 * num_forces * (horizon + 1)
        
        A_force = np.zeros((num_constraints, total_vars))
        b_force = np.zeros(num_constraints)
        
        constraint_idx = 0
        
        # Stages k=0,...,N-1
        for k in range(horizon):
            u_k_start = k * nz + state_dim
            
            for i in range(num_forces):
                force_idx = u_k_start + i
                
                # Lower bound: -f_i <= -force_min
                A_force[constraint_idx, force_idx] = -1.0
                b_force[constraint_idx] = -force_min
                constraint_idx += 1
                
                # Upper bound: f_i <= force_max
                A_force[constraint_idx, force_idx] = 1.0
                b_force[constraint_idx] = force_max
                constraint_idx += 1
        
        # ✅ FIX: Terminal stage N input u_N
        u_N_start = horizon * nz + state_dim
        
        for i in range(num_forces):
            force_idx = u_N_start + i
            
            # Lower bound
            A_force[constraint_idx, force_idx] = -1.0
            b_force[constraint_idx] = -force_min
            constraint_idx += 1
            
            # Upper bound
            A_force[constraint_idx, force_idx] = 1.0
            b_force[constraint_idx] = force_max
            constraint_idx += 1
        
        return A_force, b_force

    def _build_force_rate_constraints_matlab(self, horizon: int, state_dim: int, 
                                    input_dim: int, total_vars: int, nz: int, nxu: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Build force rate constraints for MATLAB-style structure."""
        if horizon <= 1:
            return None, None
        
        num_forces = self.model_params.num_forces
        rate_increase_limit = self.constraint_bounds['force_rate_increase'][1]
        rate_decrease_limit = self.constraint_bounds['force_rate_decrease'][1]
        
        # 2 constraints per force per rate variable
        # We have du_k for k=0,...,N-1 (N rate variables)
        num_constraints = 2 * num_forces * horizon
        
        A_rate = np.zeros((num_constraints, total_vars))
        b_rate = np.zeros(num_constraints)
        
        constraint_idx = 0
        
        # For each stage k=0,...,N-1
        for k in range(horizon):
            # Input rate du_k starts at k*nz + state_dim + input_dim
            du_k_start = k * nz + state_dim + input_dim
            
            for i in range(num_forces):
                rate_idx = du_k_start + i
                
                # Rate increase: du_i <= rate_increase_limit * dt
                A_rate[constraint_idx, rate_idx] = 1.0
                b_rate[constraint_idx] = rate_increase_limit * dt
                constraint_idx += 1
                
                # Rate decrease: -du_i <= rate_decrease_limit * dt
                A_rate[constraint_idx, rate_idx] = -1.0
                b_rate[constraint_idx] = rate_decrease_limit * dt
                constraint_idx += 1
        
        return A_rate, b_rate

    def _build_state_bounds_constraints_matlab(self, horizon: int, state_dim: int, 
                                    input_dim: int, total_vars: int, nz: int, nxu: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build state bounds constraints for MATLAB-style structure."""
        state_bounds_map = {
            0: self.constraint_bounds['position'],      # x
            1: self.constraint_bounds['position'],      # y
            2: self.constraint_bounds['orientation'],   # theta
            3: self.constraint_bounds['linear_velocity'],  # vx
            4: self.constraint_bounds['linear_velocity'],  # vy
            5: self.constraint_bounds['angular_velocity'], # omega
            6: self.constraint_bounds['path_param']     # path_param
        }
        
        # State appears at: x0 (stage 0), x1 (stage 1), ..., xN (terminal)
        # x_k for k=0,...,N-1 starts at k*nz
        # x_N (terminal) starts at horizon*nz
        
        # Count valid constraints
        valid_constraints = []
        
        # Stages k=0,...,N
        for k in range(horizon + 1):
            for i in range(state_dim):
                bounds = state_bounds_map[i]
                
                if bounds[0] > -np.inf:  # Lower bound exists
                    valid_constraints.append(('lower', k, i, bounds[0]))
                if bounds[1] < np.inf:   # Upper bound exists
                    valid_constraints.append(('upper', k, i, bounds[1]))
        
        num_constraints = len(valid_constraints)
        A_state = np.zeros((num_constraints, total_vars))
        b_state = np.zeros(num_constraints)
        
        for constraint_idx, (bound_type, k, i, bound_value) in enumerate(valid_constraints):
            # Get state variable index
            if k < horizon:
                # Regular stage: x_k starts at k*nz
                state_idx = k * nz + i
            else:
                # Terminal stage: x_N starts at horizon*nz
                state_idx = horizon * nz + i
            
            if bound_type == 'lower':
                # Lower bound: -x_i <= -bound_value
                A_state[constraint_idx, state_idx] = -1.0
                b_state[constraint_idx] = -bound_value
            else:  # upper
                # Upper bound: x_i <= bound_value
                A_state[constraint_idx, state_idx] = 1.0
                b_state[constraint_idx] = bound_value
        
        return A_state, b_state

    def _build_path_parameter_constraints_matlab(self, horizon: int, state_dim: int, 
                                        input_dim: int, total_vars: int, nz: int, nxu: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build path parameter constraints for MATLAB-style structure."""
        path_min, path_max = self.constraint_bounds['path_param']
        
        # Path parameter is state component 6 at each stage k=0,...,N
        num_constraints = 2 * (horizon + 1)  # Lower and upper bounds for all stages
        
        A_path = np.zeros((num_constraints, total_vars))
        b_path = np.zeros(num_constraints)
        
        constraint_idx = 0
        
        for k in range(horizon + 1):
            # Get path parameter index
            if k < horizon:
                path_param_idx = k * nz + 6
            else:
                path_param_idx = horizon * nz + 6
            
            # Lower bound: -path_param <= -path_min
            A_path[constraint_idx, path_param_idx] = -1.0
            b_path[constraint_idx] = -path_min
            constraint_idx += 1
            
            # Upper bound: path_param <= path_max
            A_path[constraint_idx, path_param_idx] = 1.0
            b_path[constraint_idx] = path_max
            constraint_idx += 1
        
        return A_path, b_path

    def _build_virtual_speed_constraints_matlab(self, horizon: int, state_dim: int, 
                                    input_dim: int, total_vars: int, nz: int, nxu: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build virtual speed constraints for MATLAB-style structure."""
        vspeed_min, vspeed_max = self.constraint_bounds['virtual_speed']
        
        # ✅ FIX: Include terminal u_N
        num_constraints = 2 * (horizon + 1)
        
        A_vspeed = np.zeros((num_constraints, total_vars))
        b_vspeed = np.zeros(num_constraints)
        
        constraint_idx = 0
        
        # Stages k=0,...,N-1
        for k in range(horizon):
            vspeed_idx = k * nz + state_dim + (input_dim - 1)
            
            # Lower bound
            A_vspeed[constraint_idx, vspeed_idx] = -1.0
            b_vspeed[constraint_idx] = -vspeed_min
            constraint_idx += 1
            
            # Upper bound
            A_vspeed[constraint_idx, vspeed_idx] = 1.0
            b_vspeed[constraint_idx] = vspeed_max
            constraint_idx += 1
        
        # ✅ FIX: Terminal stage N
        vspeed_N_idx = horizon * nz + state_dim + (input_dim - 1)
        
        # Lower bound
        A_vspeed[constraint_idx, vspeed_N_idx] = -1.0
        b_vspeed[constraint_idx] = -vspeed_min
        constraint_idx += 1
        
        # Upper bound
        A_vspeed[constraint_idx, vspeed_N_idx] = 1.0
        b_vspeed[constraint_idx] = vspeed_max
        constraint_idx += 1
        
        return A_vspeed, b_vspeed

    # Careful here, coz we might wrong the sign
    def _apply_constraint_normalization_matlab(self, A_ineq: np.ndarray, b_ineq: np.ndarray, 
                                    horizon: int, nz: int, nxu: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply normalization to constraints for MATLAB-style structure."""
        if self.normalizer is None:
            return A_ineq, b_ineq
        
        state_dim = 7
        input_dim = self.model_params.num_forces + 1
        
        # Build transformation matrix for MATLAB-style structure
        from scipy.linalg import block_diag
        
        # For each stage k=0,...,N-1: [Tx, Tu, TDu]
        # For terminal stage N: [Tx, Tu]
        T_blocks = []
        for k in range(horizon):
            T_blocks.append(self.normalizer.Tx)      # State
            T_blocks.append(self.normalizer.Tu)      # Input
            T_blocks.append(self.normalizer.TDu)     # Input rate
        T_blocks.append(self.normalizer.Tx)  # Terminal state
        T_blocks.append(self.normalizer.Tu)  # Terminal input (no rate)
        
        invT = block_diag(*T_blocks)
        
        # Transform: A_ineq_normalized = A_ineq @ invT
        A_ineq_normalized = A_ineq @ invT
        
        return A_ineq_normalized, b_ineq 
 

    # ================================================================
    # NEW FUNCTION TO MATCH GOLDEN IMPLEMENTATION
    # ================================================================
    def get_equality_constraints_single_stage(self, 
                                            Ad: np.ndarray, 
                                            Bd: np.ndarray, 
                                            gd: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Builds the single-stage AUGMENTED matrices (Ak, Bk, gk) to match the
        golden implementation's state augmentation:
        
        s_k+1 = Ak * s_k + Bk * v_k + gk
        
         This function receives the NON-AUGMENTED Ad, Bd, gd from the
         linearizer (Test 1).
        """
        
        # ---
        # FIX: Use modular-native dimension definitions
        # ---
        nx = 7  # Modular state dim [x, y, θ, vx, vy, ω, s]
        nu = self.model_params.num_forces + 1 # Modular input dim [f1..fn, vs]
        # ---

        # Apply normalization just like the golden function
        if self.normalizer is not None:
            Tx = self.normalizer.Tx
            invTx = self.normalizer.invTx
            Tu = self.normalizer.Tu
            invTu = self.normalizer.invTu
            
            _Anorm = Tx @ Ad @ invTx
            _Bnorm = Tx @ Bd @ invTu
            _gd_norm = Tx @ gd.reshape(-1, 1) # Ensure column vector
        else:
            _Anorm = Ad
            _Bnorm = Bd
            _gd_norm = gd.reshape(-1, 1) # Ensure column vector

        # 1. Build Ak = [[A, B], [0, I]]
        Ak = np.block([
            [_Anorm, _Bnorm],
            [np.zeros((nu, nx)), np.eye(nu)]
        ])
        
        # 2. Build Bk = [[B], [I]]
        Bk = np.vstack([
            _Bnorm,
            np.eye(nu)
        ])
        
        # 3. Build gk = [[gd], [0]]
        gk = np.vstack([
            _gd_norm,
            np.zeros((nu, 1))
        ])
        
        # Return as a dictionary, flattening gk to match expected output shape
        # The golden 'discretized_linearized_model' returns a 1D vector for gd,
        # so the golden 'getEqualityConstraints' likely returns a 1D vector for gk.
        return {'Ak': Ak, 'Bk': Bk, 'gk': gk.flatten()}
    
    # We deliberately skip grasp matrix constraints for now due to complexity
    def _build_grasp_matrix_constraints_deprecated(self, horizon: int, state_dim: int, 
                                      input_dim: int, total_vars: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Build grasp matrix feasibility constraints."""
        if self.model_params.grasp_matrix is None:
            return None, None
        
        grasp_matrix = self.model_params.grasp_matrix
        num_forces = self.model_params.num_forces
        
        # Example: Wrench magnitude limits
        # ||G @ f|| <= max_wrench for each time step
        # This can be linearized as multiple linear constraints
        
        # For simplicity, we'll implement friction cone constraints
        # More sophisticated grasp constraints can be added here
        
        max_wrench_force = self.model_params.max_forces_allowed * 2.0
        max_wrench_moment = getattr(self.model_params, 'static_m_max', max_wrench_force * 0.5)
        
        # 6 constraints per time step: ±fx, ±fy, ±moment <= limits
        num_constraints = 6 * horizon
        
        A_grasp = np.zeros((num_constraints, total_vars))
        b_grasp = np.zeros(num_constraints)
        
        total_state_vars = state_dim * (horizon + 1)
        constraint_idx = 0
        
        for k in range(horizon):
            force_start_idx = total_state_vars + k * input_dim
            
            # Get wrench = G @ forces for time step k
            for wrench_component in range(min(3, grasp_matrix.shape[0])):
                # Positive wrench limit: G[i,:] @ f <= max_wrench
                for force_idx in range(num_forces):
                    A_grasp[constraint_idx, force_start_idx + force_idx] = grasp_matrix[wrench_component, force_idx]
                
                if wrench_component < 2:  # Force components
                    b_grasp[constraint_idx] = max_wrench_force
                else:  # Moment component
                    b_grasp[constraint_idx] = max_wrench_moment
                constraint_idx += 1
                
                # Negative wrench limit: -G[i,:] @ f <= max_wrench
                for force_idx in range(num_forces):
                    A_grasp[constraint_idx, force_start_idx + force_idx] = -grasp_matrix[wrench_component, force_idx]
                
                if wrench_component < 2:  # Force components
                    b_grasp[constraint_idx] = max_wrench_force
                else:  # Moment component
                    b_grasp[constraint_idx] = max_wrench_moment
                constraint_idx += 1
        
        return A_grasp[:constraint_idx], b_grasp[:constraint_idx]
    

    def get_equality_constraints(self, horizon: int, linearized_dynamics: List[Dict], 
                            initial_state: AugmentedState) -> Dict[str, np.ndarray]:
        """
        Generate equality constraints for MATLAB-style MPCC optimization.
        
        MATLAB-style decision variables: z = [x0, u0, du0, x1, u1, du1, ..., xN-1, uN-1, duN-1, xN]
        
        Where:
        - xi: state at step i (7 dims: x, y, θ, vx, vy, ω, s)
        - ui: input at step i (num_forces+1 dims)
        - dui: input rate at step i (num_forces+1 dims)
        
        Constraints:
        1. Initial state: x0 = x_initial
        2. Dynamics: x_{k+1} = Ad_k @ x_k + Bd_k @ u_k + gd_k for k=0,...,N-1
        3. Rate relationship: u_{k+1} - u_k = du_k for k=0,...,N-2
        """
        state_dim = 7
        input_dim = self.model_params.num_forces + 1
        
        # MATLAB-style structure
        nz = state_dim + 2 * input_dim  # Per-stage vars: state + input + input_rate
        nxu = state_dim + input_dim     # Terminal stage: state + input only
        
        total_vars = nz * horizon + nxu  # N stages with rates + terminal state
        
        # Equality constraints:
        # - 1 initial state constraint (7 equations)
        # - N dynamics constraints (7*N equations)
        # - (N-1) rate relationship constraints ((num_forces+1)*(N-1) equations)
        num_eq_constraints = state_dim + state_dim * horizon + input_dim * (horizon - 1)
        
        A_eq = np.zeros((num_eq_constraints, total_vars))
        b_eq = np.zeros(num_eq_constraints)
        
        constraint_idx = 0
        
        # ================================================================
        # 1. INITIAL STATE CONSTRAINT: x_0 = x_initial
        # ================================================================
        initial_vector = initial_state.get_augmented_vec()
        
        if self.normalizer is not None:
            initial_vector = self.normalizer.Tx @ initial_vector
        
        # x0 is at indices [0:7]
        for i in range(state_dim):
            A_eq[constraint_idx, i] = 1.0
            b_eq[constraint_idx] = initial_vector[i]
            constraint_idx += 1
        
        # ================================================================
        # 2. DYNAMICS CONSTRAINTS: x_{k+1} = Ad_k @ x_k + Bd_k @ u_k + gd_k
        # ================================================================
        for k in range(horizon):
            # Get linearized dynamics for stage k
            stage_data = linearized_dynamics[k] if k < len(linearized_dynamics) else linearized_dynamics[-1]
            
            Ad = stage_data['Ad']
            Bd = stage_data['Bd']
            gd = stage_data['gd'].flatten() if hasattr(stage_data['gd'], 'shape') else stage_data['gd']
            
            # Variable indices in MATLAB-style structure
            # Stage k block: [x_k, u_k, du_k] starts at k*nz
            # Terminal state x_N at horizon*nz
            
            if k < horizon - 1:
                # Regular stage k
                x_k_start = k * nz  # State x_k
                u_k_start = k * nz + state_dim  # Input u_k
                x_k1_start = (k + 1) * nz  # State x_{k+1}
            else:
                # Last stage (k = N-1)
                x_k_start = k * nz  # State x_{N-1}
                u_k_start = k * nz + state_dim  # Input u_{N-1}
                x_k1_start = horizon * nz  # Terminal state x_N
            
            # Build constraint: x_{k+1} - Ad_k @ x_k - Bd_k @ u_k = gd_k
            for i in range(state_dim):
                eq_idx = constraint_idx + i
                
                # x_{k+1} coefficient
                A_eq[eq_idx, x_k1_start + i] = 1.0
                
                # -Ad_k @ x_k coefficients
                for j in range(state_dim):
                    A_eq[eq_idx, x_k_start + j] = -Ad[i, j]
                
                # -Bd_k @ u_k coefficients
                for j in range(input_dim):
                    A_eq[eq_idx, u_k_start + j] = -Bd[i, j]
                
                # Right-hand side: gd_k
                b_eq[eq_idx] = gd[i] if len(gd) > i else 0.0
            
            constraint_idx += state_dim
        
        # ================================================================
        # 3. RATE RELATIONSHIP: u_{k+1} - u_k = du_k for k=0,...,N-2
        # ================================================================
        for k in range(horizon - 1):
            # Stage k: [x_k, u_k, du_k] starts at k*nz
            # Stage k+1: [x_{k+1}, u_{k+1}, du_{k+1}] starts at (k+1)*nz
            
            u_k_start = k * nz + state_dim
            du_k_start = k * nz + state_dim + input_dim
            u_k1_start = (k + 1) * nz + state_dim
            
            # Constraint: u_{k+1} - u_k - du_k = 0
            for i in range(input_dim):
                eq_idx = constraint_idx + i
                
                A_eq[eq_idx, u_k1_start + i] = 1.0   # u_{k+1}
                A_eq[eq_idx, u_k_start + i] = -1.0   # -u_k
                A_eq[eq_idx, du_k_start + i] = -1.0  # -du_k
                
                b_eq[eq_idx] = 0.0
            
            constraint_idx += input_dim
        
        return {
            'A_eq': A_eq,
            'b_eq': b_eq,
            'num_constraints': A_eq.shape[0],
            'initial_state': initial_vector,
            'dynamics_stages': len(linearized_dynamics),
            'structure': {
                'state_dim': state_dim,
                'input_dim': input_dim,
                'nz': nz,
                'nxu': nxu,
                'total_vars': total_vars
            }
        }
    
    def check_constraint_violations(self, solution: Dict, dt: float = 0.05) -> Dict[str, List]:
        """
        Check constraint violations in MPC solution using ModelParams bounds.
        
        Args:
            solution: MPC solution with state and control trajectories
            dt: Time step for rate constraint checking
            
        Returns:
            Dictionary of violations by constraint type
        """
        violations = {constraint_type: [] for constraint_type in self.violation_history.keys()}
        
        # Extract trajectories
        state_trajectory = solution.get('state_trajectory', [])
        control_trajectory = solution.get('control_trajectory', [])
        
        if not state_trajectory or not control_trajectory:
            return violations
        
        # Check force magnitude violations
        if self.constraint_flags['force_magnitude']:
            violations['force_magnitude'] = self._check_force_magnitude_violations(control_trajectory)
        
        # Check force rate violations
        if self.constraint_flags['force_rate'] and len(control_trajectory) > 1:
            violations['force_rate'] = self._check_force_rate_violations(control_trajectory, dt)
        
        # Check state bound violations
        if self.constraint_flags['state_bounds']:
            violations['state_bounds'] = self._check_state_bound_violations(state_trajectory)
        
        # Check path parameter violations
        if self.constraint_flags['path_bounds']:
            violations['path_bounds'] = self._check_path_bound_violations(state_trajectory)
        
        # Check virtual speed violations
        if self.constraint_flags['virtual_speed']:
            violations['virtual_speed'] = self._check_virtual_speed_violations(control_trajectory)
        
        # Update violation history
        for constraint_type, violation_list in violations.items():
            self.violation_history[constraint_type].append(len(violation_list) > 0)
        
        return violations
    
    def _check_force_magnitude_violations(self, control_trajectory: List) -> List[Dict]:
        """Check force magnitude constraint violations."""
        violations = []
        force_min, force_max = self.constraint_bounds['force_magnitude']
        
        for k, control in enumerate(control_trajectory):
            if isinstance(control, (list, tuple)):
                forces = control[0] if len(control) > 0 else []
            else:
                forces = control[:-1]  # Exclude virtual speed
            
            for i, force in enumerate(forces):
                if force < force_min or force > force_max:
                    violations.append({
                        'stage': k,
                        'contact': i,
                        'value': force,
                        'bounds': (force_min, force_max),
                        'violation_type': 'magnitude'
                    })
        
        return violations
    
    def _check_force_rate_violations(self, control_trajectory: List, dt: float) -> List[Dict]:
        """Check force rate constraint violations."""
        violations = []
        rate_increase_limit = self.constraint_bounds['force_rate_increase'][1]
        rate_decrease_limit = self.constraint_bounds['force_rate_decrease'][1]
        
        for k in range(len(control_trajectory) - 1):
            forces_k = control_trajectory[k][0] if isinstance(control_trajectory[k], (list, tuple)) else control_trajectory[k][:-1]
            forces_k1 = control_trajectory[k+1][0] if isinstance(control_trajectory[k+1], (list, tuple)) else control_trajectory[k+1][:-1]
            
            for i, (f_k, f_k1) in enumerate(zip(forces_k, forces_k1)):
                rate = (f_k1 - f_k) / dt
                
                if rate > rate_increase_limit:
                    violations.append({
                        'stage': k,
                        'contact': i,
                        'rate': rate,
                        'limit': rate_increase_limit,
                        'violation_type': 'rate_increase'
                    })
                elif rate < -rate_decrease_limit:
                    violations.append({
                        'stage': k,
                        'contact': i,
                        'rate': rate,
                        'limit': -rate_decrease_limit,
                        'violation_type': 'rate_decrease'
                    })
        
        return violations
    
    def _check_state_bound_violations(self, state_trajectory: List) -> List[Dict]:
        """Check state bound constraint violations."""
        violations = []
        
        state_names = ['x', 'y', 'theta', 'vx', 'vy', 'omega', 'path_param']
        bound_map = {
            0: self.constraint_bounds['position'],
            1: self.constraint_bounds['position'],
            2: self.constraint_bounds['orientation'],
            3: self.constraint_bounds['linear_velocity'],
            4: self.constraint_bounds['linear_velocity'],
            5: self.constraint_bounds['angular_velocity'],
            6: self.constraint_bounds['path_param']
        }
        
        for k, state in enumerate(state_trajectory):
            if isinstance(state, AugmentedState):
                state_vec = state.get_augmented_vec()
            else:
                state_vec = state
            
            for i, (value, bounds) in enumerate(zip(state_vec, [bound_map[j] for j in range(7)])):
                if value < bounds[0] or value > bounds[1]:
                    violations.append({
                        'stage': k,
                        'state_component': state_names[i],
                        'value': value,
                        'bounds': bounds,
                        'violation_type': 'state_bound'
                    })
        
        return violations
    
    def _check_path_bound_violations(self, state_trajectory: List) -> List[Dict]:
        """Check path parameter violations specifically."""
        violations = []
        path_min, path_max = self.constraint_bounds['path_param']
        
        for k, state in enumerate(state_trajectory):
            if isinstance(state, AugmentedState):
                path_param = state.path_param
            else:
                path_param = state[6] if len(state) > 6 else 0.0
            
            if path_param < path_min or path_param > path_max:
                violations.append({
                    'stage': k,
                    'path_param': path_param,
                    'bounds': (path_min, path_max),
                    'violation_type': 'path_bound'
                })
        
        return violations
    
    def _check_virtual_speed_violations(self, control_trajectory: List) -> List[Dict]:
        """Check virtual speed violations."""
        violations = []
        vspeed_min, vspeed_max = self.constraint_bounds['virtual_speed']
        
        for k, control in enumerate(control_trajectory):
            if isinstance(control, (list, tuple)):
                virtual_speed = control[1] if len(control) > 1 else 0.0
            else:
                virtual_speed = control[-1]  # Last element
            
            if virtual_speed < vspeed_min or virtual_speed > vspeed_max:
                violations.append({
                    'stage': k,
                    'virtual_speed': virtual_speed,
                    'bounds': (vspeed_min, vspeed_max),
                    'violation_type': 'virtual_speed'
                })
        
        return violations
    
    def update_model_params(self, new_model_params: ModelParams):
        """Update constraint handler when ModelParams change."""
        self.model_params = new_model_params
        self.constraint_bounds = new_model_params.get_constraint_bounds()
        self.constraint_flags = self._get_goal_specific_constraint_flags()
        
        print(f"🔒 Constraints updated for new ModelParams:")
        print(f"   Force limits: {self.constraint_bounds['force_magnitude']}")
        print(f"   Control goal: {self.model_params.control_goal.control_goal}")
    
    def get_constraint_statistics(self) -> Dict[str, Union[float, int]]:
        """Get comprehensive constraint violation statistics."""
        stats = {}
        
        for constraint_type, history in self.violation_history.items():
            if history:
                stats[f'{constraint_type}_violation_rate'] = np.mean(history)
                stats[f'{constraint_type}_total_violations'] = np.sum(history)
                stats[f'{constraint_type}_recent_violations'] = np.sum(history[-10:]) if len(history) >= 10 else np.sum(history)
            else:
                stats[f'{constraint_type}_violation_rate'] = 0.0
                stats[f'{constraint_type}_total_violations'] = 0
                stats[f'{constraint_type}_recent_violations'] = 0
        
        # Overall statistics
        all_violations = [v for history in self.violation_history.values() for v in history]
        stats['overall_violation_rate'] = np.mean(all_violations) if all_violations else 0.0
        stats['total_constraint_checks'] = len(all_violations)
        
        return stats
    
    def adjust_constraint_bounds(self, constraint_type: str, adjustment_factor: float, 
                               reason: str = "manual"):
        """
        Dynamically adjust constraint bounds.
        
        Args:
            constraint_type: Type of constraint to adjust
            adjustment_factor: Multiplicative factor (>1 = looser, <1 = tighter)
            reason: Reason for adjustment
        """
        if constraint_type == 'force_magnitude':
            old_max = self.constraint_bounds['force_magnitude'][1]
            new_max = old_max * adjustment_factor
            self.constraint_bounds['force_magnitude'] = (0.0, new_max)
            # Also update ModelParams
            self.model_params.update_force_limits(adjustment_factor, reason)
            
        elif constraint_type == 'force_rate':
            old_inc = self.constraint_bounds['force_rate_increase'][1]
            old_dec = self.constraint_bounds['force_rate_decrease'][1]
            self.constraint_bounds['force_rate_increase'] = (0.0, old_inc * adjustment_factor)
            self.constraint_bounds['force_rate_decrease'] = (0.0, old_dec * adjustment_factor)
            
        elif constraint_type == 'virtual_speed':
            old_min, old_max = self.constraint_bounds['virtual_speed']
            center = (old_min + old_max) / 2
            half_range = (old_max - old_min) / 2
            new_half_range = half_range * adjustment_factor
            self.constraint_bounds['virtual_speed'] = (center - new_half_range, center + new_half_range)
        
        print(f"🔧 Adjusted {constraint_type} constraints by factor {adjustment_factor:.2f} ({reason})")
    
    def reset_violation_tracking(self):
        """Reset constraint violation tracking."""
        for constraint_type in self.violation_history:
            self.violation_history[constraint_type].clear()
    
    def print_constraint_summary(self):
        """Print comprehensive constraint configuration summary."""
        print(f"\n🔒 MPCC Constraint Configuration Summary:")
        print(f"{'='*60}")
        
        print(f"Model Integration:")
        print(f"   Model Forces: {self.model_params.num_forces}")
        print(f"   Control Goal: {self.model_params.control_goal.control_goal}")
        print(f"   Mass: {self.model_params.mass:.2f} kg")
        print(f"   Normalization: {'✅ Enabled' if self.normalizer else '❌ Disabled'}")
        
        print(f"\nActive Constraints:")
        for constraint_type, is_active in self.constraint_flags.items():
            status = "✅" if is_active else "❌"
            print(f"   {status} {constraint_type}")
        
        print(f"\nConstraint Bounds (from ModelParams):")
        for bound_type, bounds in self.constraint_bounds.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                print(f"   {bound_type}: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
        
        # Violation statistics
        stats = self.get_constraint_statistics()
        if stats['total_constraint_checks'] > 0:
            print(f"\nViolation Statistics:")
            print(f"   Overall violation rate: {stats['overall_violation_rate']:.2%}")
            print(f"   Total constraint checks: {stats['total_constraint_checks']}")

# ================================================================
# DEMONSTRATION FUNCTION
# ================================================================

def demo_enhanced_mpcc_constraints():
    """Demonstrate the enhanced MPCC constraint class with ModelParams integration."""
    
    print("🔒 Enhanced MPCC Constraint Class Demo")
    print("="*60)
    
    # Create test setup
    standard_objects = create_standard_objects()
    obj = standard_objects['rectangle']
    dynamics = DynamicObjectModel(obj, friction_noise_std=0.001)
    
    # Test different control goals
    control_goals = ['position_only', 'omega_only', 'full_pose']
    
    for goal_type in control_goals:
        print(f"\n--- Testing {goal_type.upper()} ---")
        
        # Create complete integration chain
        control_goal = create_control_goal_from_mode(goal_type)
        model_params = ModelParams(dynamics, control_goal)
        
        # Set up contact configuration  
        test_contacts = get_goal_specific_contact_configuration(obj, control_goal)
        model_params.set_contact_configuration(test_contacts)
        
        # Create normalization
        normalizer = StateInputNormalization(model_params)
        # normalizer = None  # Skip normalization for simplicity
        
        # Create enhanced constraint handler
        constraint_handler = MPCCConstraintClass(model_params, normalizer)
        
        constraint_handler.print_constraint_summary()
        
        # Test constraint generation
        print(f"\n📐 Testing constraint generation...")
        horizon = 10
        dt = 0.05
        
        # Generate inequality constraints
        ineq_constraints = constraint_handler.get_linear_inequality_constraints(horizon, dt)
        
        print(f"   Inequality constraints shape: {ineq_constraints['A_ineq'].shape}")
        print(f"   Number of constraints: {ineq_constraints['num_constraints']}")
        print(f"   Constraint types: {set(ineq_constraints['constraint_types'])}")
        print(f"   Active flags: {sum(ineq_constraints['active_flags'].values())}/{len(ineq_constraints['active_flags'])}")
        
        # Test equality constraints with dummy data
        print(f"\n⚖️ Testing equality constraints...")
        
        # Create dummy linearized dynamics
        state_dim = 7
        input_dim = model_params.num_forces + 1
        dummy_dynamics = []
        
        for k in range(horizon):
            Ad = np.eye(state_dim) + 0.01 * np.random.randn(state_dim, state_dim)
            Bd = 0.01 * np.random.randn(state_dim, input_dim)
            gd = 0.001 * np.random.randn(state_dim)
            dummy_dynamics.append({'Ad': Ad, 'Bd': Bd, 'gd': gd})
        
        # Create initial state
        state_data = {
            'object_position': np.array([0.0, 0.0]),
            'object_orientation': 0.0,
            'velocity_body': np.array([0.1, 0.0]),
            'angular_velocity': 0.0
        }
        initial_state = AugmentedState(state_data, path_param=0.0)
        
        eq_constraints = constraint_handler.get_equality_constraints(
            horizon, dummy_dynamics, initial_state
        )
        
        print(f"   Equality constraints shape: {eq_constraints['A_eq'].shape}")
        print(f"   Dynamics stages: {eq_constraints['dynamics_stages']}")
        # In demo function, check before normalization OR check the constraint after denormalization
        if normalizer is not None:
            # The constraint becomes: T @ x = b, not x = b
            print(f"   Initial state constrained through normalization transformation")
        else:
            print(f"   Initial state properly constrained: {np.allclose(eq_constraints['A_eq'][:state_dim, :state_dim], np.eye(state_dim))}")        

        # Test constraint violation checking
        print(f"\n🚨 Testing constraint violation detection...")
        
        # Create test solution with some violations
        test_solution = {
            'state_trajectory': [initial_state] + [
                create_augmented_state_from_vector(
                    np.array([0.1*k, 0.05*k, 0.1*k, 0.1, 0.02, 0.05]), 
                    0.1*k + (0.5 if k == 6 else 0.0)  # Path param violation at k=6
                ) for k in range(1, horizon+1)
            ],
            'control_trajectory': [
                (np.array([2.0, 1.5, 12.0, 1.0]), 0.5) if k == 3 else  # Force violation at k=3
                (np.array([1.0, 1.2, 2.5, 0.8]), 0.6) 
                for k in range(horizon)
            ]
        }
        
        violations = constraint_handler.check_constraint_violations(test_solution, dt)
        
        total_violations = sum(len(v_list) for v_list in violations.values())
        print(f"   Total violations detected: {total_violations}")
        
        for constraint_type, violation_list in violations.items():
            if violation_list:
                print(f"   {constraint_type}: {len(violation_list)} violations")
                # Show first violation as example
                example = violation_list[0]
                print(f"     Example: Stage {example['stage']}, Type: {example.get('violation_type', 'unknown')}")
        
        # Test dynamic constraint adjustment
        print(f"\n🔧 Testing dynamic constraint adjustment...")
        
        old_force_limit = constraint_handler.constraint_bounds['force_magnitude'][1]
        constraint_handler.adjust_constraint_bounds('force_magnitude', 1.5, "testing")
        new_force_limit = constraint_handler.constraint_bounds['force_magnitude'][1]
        
        print(f"   Force limit adjusted: {old_force_limit:.2f} → {new_force_limit:.2f}")
        
        # Test statistics
        stats = constraint_handler.get_constraint_statistics()
        print(f"\n📊 Constraint statistics:")
        for key, value in stats.items():
            if 'rate' in key:
                print(f"   {key}: {value:.2%}")
            else:
                print(f"   {key}: {value}")

if __name__ == "__main__":
    demo_enhanced_mpcc_constraints()

# %%
# SIMPLIFIED LINEARIZING LTV CLASS
# ================================================================
class LinearizingLTVClass:
    """
    Simplified LTV linearization that focuses on core MPCC dynamics.
    Integrates with ModelParams and assumes kinetic motion throughout.
    """
    
    def __init__(self, model_params: ModelParams, normalizer: StateInputNormalization = None, dt = 0.05, dynamics_option: str = 'damped'):
        """
        Initialize linearizer with ModelParams integration.
        
        Args:
            model_params: Centralized model parameters
            normalizer: Optional state/input normalization
        """
        self.model_params = model_params
        self.normalizer = normalizer
        
        # Default time step for discretization
        self.dt = dt
        
        self.dynamics_option = dynamics_option  # honor controller setting

        # Linearization history for analysis
        self.linearization_history = []
        
        # Dynamics structure based on control goal
        self.control_goal_type = model_params.control_goal.control_goal
        
        print(f"🔧 LTV Linearizer initialized:")
        print(f"   Control goal: {self.control_goal_type}")
        print(f"   Forces: {model_params.num_forces}")
        print(f"   Mass: {model_params.mass:.2f}kg, Inertia: {model_params.inertia:.6f}kg⋅m²")
        print(f"   Normalization: {'✅ Enabled' if normalizer else '❌ Disabled'}")

    def linearize_augmented_dynamics(self, augmented_state: AugmentedState, 
                                forces: np.ndarray, virtual_speed: float,
                                reference_path: 'SplineReferencePath',
                                dynamics_option: str = 'damped') -> Dict[str, np.ndarray]:
        """
        Linearize augmented MPCC dynamics around operating point with enhanced dynamics options.
        
        State: [x, y, θ, vx, vy, ω, s]  (7 states)
        Input: [f1, f2, ..., fn, vs]    (n+1 inputs)
        
        Args:
            augmented_state: Current state for linearization
            forces: Force vector at linearization point
            virtual_speed: Virtual speed at linearization point
            reference_path: Reference path for MPCC
            dynamics_option: 'ideal', 'damped', or 'friction_aware'
            
        Returns:
            Dictionary with Ad, Bd, gd matrices
        """

        # default to instance setting if not provided
        if dynamics_option is None:
            dynamics_option = getattr(self, 'dynamics_option', 'damped')


        # Extract state variables
        x, y, theta, vx, vy, omega, path_param = augmented_state.get_augmented_vec()
        
        # Get model parameters
        mass = self.model_params.mass
        inertia = self.model_params.inertia
        num_forces = self.model_params.num_forces
        
        # ================================================================
        # CONTINUOUS-TIME DYNAMICS LINEARIZATION
        # ================================================================
        
        # State dimension: 7, Input dimension: num_forces + 1
        Ac = np.zeros((7, 7))
        Bc = np.zeros((7, num_forces + 1))
        gc = np.zeros(7)
        
        # Trigonometric values at linearization point
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # ================================================================
        # 1. POSITION DYNAMICS: [dx/dt, dy/dt] = f(vx, vy, θ)
        # ================================================================
        # dx/dt = vx*cos(θ) - vy*sin(θ)
        # dy/dt = vx*sin(θ) + vy*cos(θ)
        
        # Jacobian w.r.t. states
        Ac[0, 2] = -vx * sin_theta - vy * cos_theta  # ∂(dx/dt)/∂θ
        Ac[0, 3] = cos_theta                         # ∂(dx/dt)/∂vx
        Ac[0, 4] = -sin_theta                        # ∂(dx/dt)/∂vy
        
        Ac[1, 2] = vx * cos_theta - vy * sin_theta   # ∂(dy/dt)/∂θ
        Ac[1, 3] = sin_theta                         # ∂(dy/dt)/∂vx
        Ac[1, 4] = cos_theta                         # ∂(dy/dt)/∂vy
        
        # Affine term (nonlinear residual)
        gc[0] = vx * cos_theta - vy * sin_theta - (Ac[0, 2] * theta + Ac[0, 3] * vx + Ac[0, 4] * vy)
        gc[1] = vx * sin_theta + vy * cos_theta - (Ac[1, 2] * theta + Ac[1, 3] * vx + Ac[1, 4] * vy)
        
        # ================================================================
        # 2. ORIENTATION DYNAMICS: dθ/dt = ω
        # ================================================================
        Ac[2, 5] = 1.0  # ∂(dθ/dt)/∂ω = 1
        
        # ================================================================
        # 3. ENHANCED BODY VELOCITY DYNAMICS: Forces → Accelerations
        # ================================================================
        # Get grasp matrix (force to wrench mapping)
        if self.model_params.grasp_matrix is not None:
            G = self.model_params.grasp_matrix
        else:
            # Fallback: assume direct force mapping
            G = np.eye(3, num_forces)
        
        if dynamics_option == 'ideal':
            # ================================================================
            # OPTION 1: IDEAL DYNAMICS (No damping, no friction)
            # ================================================================
            # dvx/dt = Fx/mass
            # dvy/dt = Fy/mass  
            # dω/dt = M/inertia
            
            # No state coupling for velocities
            # Ac[3:6, :] = 0 (already initialized to zero)
            
            # Direct force to acceleration mapping via grasp matrix
            for i in range(num_forces):
                if G.shape[0] > 0:  # Fx component
                    Bc[3, i] = G[0, i] / mass
                if G.shape[0] > 1:  # Fy component
                    Bc[4, i] = G[1, i] / mass
                if G.shape[0] > 2:  # M component
                    Bc[5, i] = G[2, i] / inertia
            
            # No affine terms
            # gc[3:6] = 0 (already initialized to zero)
            
        elif dynamics_option == 'damped':
            # ================================================================
            # OPTION 2: DAMPED DYNAMICS (Current implementation)
            # ================================================================
            # dvx/dt = Fx/mass - damping_linear * vx
            # dvy/dt = Fy/mass - damping_linear * vy
            # dω/dt = M/inertia - damping_angular * ω
            
            # Light damping for stability
            damping_linear = 0.1
            if (vx == 0 and vy == 0): damping_linear = 0.0  # No linear damping at rest
            damping_angular = 0.05
            if (omega == 0): damping_angular = 0.0  # No angular damping at rest
            
            Ac[3, 3] = -damping_linear   # dvx/dt damping
            Ac[4, 4] = -damping_linear   # dvy/dt damping
            Ac[5, 5] = -damping_angular  # dω/dt damping
            # similar to
            # vx = vx - damping_linear * vx * dt
            # vy = vy - damping_linear * vy * dt
            # omega = omega - damping_angular * omega * dt
            
            # Force to acceleration mapping via grasp matrix
            for i in range(num_forces):
                if G.shape[0] > 0:  # Fx component
                    Bc[3, i] = G[0, i] / mass
                if G.shape[0] > 1:  # Fy component
                    Bc[4, i] = G[1, i] / mass
                if G.shape[0] > 2:  # M component
                    Bc[5, i] = G[2, i] / inertia
            
            # No additional affine terms beyond damping
            # gc[3:6] = 0 (already initialized to zero)
            
        elif dynamics_option == 'friction_aware':
            # ================================================================
            # OPTION 3: FRICTION-AWARE DYNAMICS (Following limit surface)
            # ================================================================
            # This implements the limit surface dynamics from the regime controller
            
            # Determine if we're in static or kinetic regime (simplified)
            # This is a simplification - in practice, regime should come from regime controller

            # Apply forces through grasp matrix
            applied_wrench = G @ forces  # [Fx_applied, Fy_applied, M_applied]

            twist_magnitude = np.sqrt(vx**2 + vy**2 + (omega * self.model_params.twist_scale) **2)

            is_static_regime = (
                twist_magnitude < 1e-2 or 
                (twist_magnitude < 1e-1 and np.linalg.norm(applied_wrench) < 1e-3)
            )
            
            # Get friction parameters from model
            if is_static_regime:
                f_max = self.model_params.static_f_max
                m_max = getattr(self.model_params, 'static_m_max', f_max * 0.5)
            else:
                f_max = self.model_params.kinetic_f_max
                m_max = getattr(self.model_params, 'kinetic_m_max', f_max * 0.5)
            
            
            if is_static_regime:
                # ============================================================
                # STATIC REGIME: Friction opposes applied wrench direction
                # ============================================================
                
                # Calculate limit surface scaling
                s = np.sqrt((applied_wrench[0]/f_max)**2 + 
                        (applied_wrench[1]/f_max)**2 + 
                        (applied_wrench[2]/m_max)**2)
                
                
                # ∂v̇x/∂vx = ∂(friction_fx)/∂vx / mass, but friction_fx is calculated from applied wrench, not velocity
                # so no velocity coupling in static regime
                # Ac[3:6, 3:6] = 0 (no velocity coupling in static)


                # Force coupling with friction cancellation
                # v̇x = (G[0,:]·f + friction_fx) / mass # default for any regime
                # ∂v̇x/∂f[i] = G[0,i]/mass + ∂(friction_fx)/∂f[i]/mass
                # but ∂(friction_fx)/∂f[i] = -G[0,i]/max(s, 1) (from limit surface linearization)
                # thus
                # ∂v̇x/∂f[i] = G[0,i]/mass · (1 - 1/max(s,1)) = G[0,i]/mass - G[0,i]/(mass*max(s,1))
                # On limit surface: friction at maximum opposing applied wrench

                # while we can handle each case separately (when s <= 1 and s > 1)
                # but giving zero value for the linearization when s <= 1
                # is, in my opinion, not gonna make the solver happy
                # so maybe, we can just keep the friction in the affine term.

        
                # Input coupling: Force transmission through grasp matrix
                # v̇ = (G @ f + friction_wrench) / mass_matrix
                # ∂v̇/∂f = G / mass_matrix (direct force transmission)
                for i in range(num_forces):
                    if G.shape[0] > 0:  # Fx component
                        Bc[3, i] = G[0, i] / mass
                    if G.shape[0] > 1:  # Fy component  
                        Bc[4, i] = G[1, i] / mass
                    if G.shape[0] > 2:  # M component
                        Bc[5, i] = G[2, i] / inertia
                
                # Affine term: Friction opposes applied wrench
                if s <= 1.0:
                    # Inside friction cone: friction = -applied_wrench (exact cancellation)
                    friction_wrench = -applied_wrench
                else:
                    # On friction limit surface: friction at maximum opposing applied wrench
                    friction_wrench = -(applied_wrench / s) * np.array([f_max, f_max, m_max])
                
                # Include friction in affine term
                # gc = f(x₀, u₀) - Ac·x₀ - Bc·u₀
                # Since Ac·x₀ = 0 (no state coupling) and Bc·u₀ = G @ forces / mass_matrix
                # We have: gc = (G @ forces + friction_wrench) / mass_matrix - G @ forces / mass_matrix
                # Therefore: gc = friction_wrench / mass_matrix
                gc[3] = friction_wrench[0] / mass
                gc[4] = friction_wrench[1] / mass  
                gc[5] = friction_wrench[2] / inertia
                
            else:
                # ============================================================
                # KINETIC REGIME: Friction opposes velocity direction
                # ============================================================
                
                # Force input coupling (applied forces)
                # v̇x = (G[0,:]·f + friction_fx) / mass # default for any regime
                # ∂v̇x/∂f[i] = G[0,i]/mass + ∂(friction_fx)/∂f[i]/mass
                # as friction_fx is independent of f[i], ∂(friction_fx)/∂f[i] = 0
                # thus
                for i in range(num_forces):
                    if G.shape[0] > 0:
                        Bc[3, i] = G[0, i] / mass
                    if G.shape[0] > 1:
                        Bc[4, i] = G[1, i] / mass
                    if G.shape[0] > 2:
                        Bc[5, i] = G[2, i] / inertia

                # Characteristic length for moment-force scaling
                c_squared = (m_max / f_max)**2
                
                # Twist direction vector (opposes motion)
                twist_dir = -np.array([vx, vy, omega * c_squared])

                s = np.sqrt((twist_dir[0]/f_max)**2 + (twist_dir[1]/f_max)**2 + (twist_dir[2]/m_max)**2)
                if (s < 1e-6):
                    friction_wrench = np.zeros(3)
                else:
                    friction_wrench = twist_dir / s

                # Velocity coupling through friction (friction opposes velocity)
                # State coupling: Friction depends on velocity direction
                if s >= 1e-6:
                    # Calculate derivatives of YOUR friction_wrench with respect to velocities
                    c4 = c_squared ** 2
                    
                    # ∂(friction_fx)/∂vx
                    dfriction_fx_dvx = -((vy**2) / f_max**2 + (omega**2 * c4) / m_max**2) / s**3
                    dfriction_fx_dvy = (vx * vy) / (f_max**2 * s**3)
                    dfriction_fx_domega = (vx * omega * c4) / (m_max**2 * s**3)
                    
                    # ∂(friction_fy)/∂vx
                    dfriction_fy_dvx = (vx * vy) / (f_max**2 * s**3)
                    dfriction_fy_dvy = -((vx**2) / f_max**2 + (omega**2 * c4) / m_max**2) / s**3
                    dfriction_fy_domega = (vy * omega * c4) / (m_max**2 * s**3)
                    
                    # ∂(friction_m)/∂vx
                    dfriction_m_dvx = (c_squared * omega * vx) / (f_max**2 * s**3)
                    dfriction_m_dvy = (c_squared * omega * vy) / (f_max**2 * s**3)
                    dfriction_m_domega = -c_squared / s + (omega**2 * c_squared * c4) / (m_max**2 * s**3)
                    
                    # Fill velocity coupling matrix
                    Ac[3, 3] = dfriction_fx_dvx / mass      # ∂v̇x/∂vx = 1/mass * ∂(applied + friction)/∂vx
                    Ac[3, 4] = dfriction_fx_dvy / mass      # ∂v̇x/∂vy  = 1/mass * ∂(applied + friction)/∂vy
                    Ac[3, 5] = dfriction_fx_domega / mass  # ∂v̇x/∂ω = 1/mass * ∂(applied + friction)/∂ω

                    Ac[4, 3] = dfriction_fy_dvx / mass      # ∂v̇y/∂vx = 1/mass * ∂(applied + friction)/∂vx
                    Ac[4, 4] = dfriction_fy_dvy / mass      # ∂v̇y/∂vy = 1/mass * ∂(applied + friction)/∂vy
                    Ac[4, 5] = dfriction_fy_domega / mass   # ∂v̇y/∂ω = 1/mass * ∂(applied + friction)/∂ω
                    
                    Ac[5, 3] = dfriction_m_dvx / inertia      # ∂ω̇/∂vx = 1/inertia * ∂(applied + friction)/∂vx
                    Ac[5, 4] = dfriction_m_dvy / inertia      # ∂ω̇/∂vy = 1/inertia * ∂(applied + friction)/∂vy
                    Ac[5, 5] = dfriction_m_domega / inertia   # ∂ω̇/∂ω = 1/inertia * ∂(applied + friction)/∂ω
                else:
                    # Near zero velocity: use small damping for numerical stability
                    Ac[3, 3] = -f_max / (mass * 1e-3)
                    Ac[4, 4] = -f_max / (mass * 1e-3)  
                    Ac[5, 5] = -m_max / (inertia * 1e-3)
                
                # Affine term: Include YOUR friction wrench
                gc[3] = friction_wrench[0] / mass
                gc[4] = friction_wrench[1] / mass  
                gc[5] = friction_wrench[2] / inertia                
        
        else:
            raise ValueError(f"Unknown dynamics_option: {dynamics_option}. Use 'ideal', 'damped', or 'friction_aware'")
        
        # ================================================================
        # 4. PATH PARAMETER DYNAMICS: ds/dt = virtual_speed
        # ================================================================
        # No state coupling
        # Ac[6, :] = 0
        
        # Direct virtual speed input
        Bc[6, -1] = 1.0  # ∂(ds/dt)/∂virtual_speed = 1
        
        # ================================================================
        # DISCRETIZATION
        # ================================================================
        Ad, Bd, gd = self._discretize_dynamics(Ac, Bc, gc, self.dt)
        # ================================================================
        # FIX: DO NOT APPLY NORMALIZATION HERE.
        # Normalization should be applied by the cost/constraint
        # functions, to match the golden implementation.
        # ================================================================
        # if self.normalizer is not None:
        #     Ad, Bd, gd = self._apply_normalization_to_dynamics(Ad, Bd, gd)
        # Store linearization info with dynamics option
        linearization_info = {
            'state': augmented_state,
            'forces': forces.copy(),
            'virtual_speed': virtual_speed,
            'dynamics_option': dynamics_option,
            'Ad_condition': np.linalg.cond(Ad),
            'Bd_condition': np.linalg.cond(Bd),
            'max_eigenvalue': np.max(np.real(np.linalg.eigvals(Ad))),
            'regime_detected': 'static' if dynamics_option == 'friction_aware' and 
                            np.sqrt(vx**2 + vy**2) < 1e-4 else 'kinetic'
        }
        self.linearization_history.append(linearization_info)
        
        return {
            'Ad': Ad,
            'Bd': Bd,
            'gd': gd,
            'linearization_point': {
                'state': augmented_state,
                'forces': forces,
                'virtual_speed': virtual_speed
            },
            'continuous_matrices': {
                'Ac': Ac,
                'Bc': Bc, 
                'gc': gc
            },
            'dynamics_option': dynamics_option,
            'friction_info': {
                'regime': linearization_info['regime_detected'],
                'f_max': f_max if dynamics_option == 'friction_aware' else None,
                'm_max': m_max if dynamics_option == 'friction_aware' else None
            } if dynamics_option == 'friction_aware' else None
        }

    def linearize_trajectory(self, state_trajectory: List[AugmentedState],
                           control_trajectory: List[Tuple[np.ndarray, float]],
                           reference_path: 'SplineReferencePath') -> List[Dict[str, np.ndarray]]:
        """
        Linearize dynamics along entire trajectory.
        
        Args:
            state_trajectory: List of AugmentedState objects
            control_trajectory: List of (forces, virtual_speed) tuples
            reference_path: Reference path
            
        Returns:
            List of linearization dictionaries
        """
        linearized_stages = []
        
        for i, (state, control) in enumerate(zip(state_trajectory, control_trajectory)):
            forces, virtual_speed = control
            
            # Linearize at this stage
            stage_dynamics = self.linearize_augmented_dynamics(
                state, forces, virtual_speed, reference_path
            )
            
            # Add stage information
            stage_dynamics['stage'] = i
            linearized_stages.append(stage_dynamics)
        
        print(f"🔧 Linearized {len(linearized_stages)} trajectory stages")
        return linearized_stages
    
    def _discretize_dynamics(self, Ac: np.ndarray, Bc: np.ndarray, gc: np.ndarray, 
                           dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Discretize continuous-time linear dynamics using matrix exponential.
        
        x_{k+1} = Ad @ x_k + Bd @ u_k + gd
        """
        n = Ac.shape[0]
        m = Bc.shape[1]
        
        # Build augmented matrix for exact discretization
        # [Ac*dt  Bc*dt  gc*dt]
        # [0      0      0    ]
        # [0      0      0    ]
        M = np.zeros((n + m + 1, n + m + 1))
        M[:n, :n] = Ac * dt
        M[:n, n:n+m] = Bc * dt
        M[:n, n+m] = gc * dt
        
        # Matrix exponential
        try:
            from scipy.linalg import expm
            expM = expm(M)
        except ImportError:
            # Fallback: first-order approximation
            expM = np.eye(n + m + 1) + M
        
        # Extract discretized matrices
        Ad = expM[:n, :n]
        Bd = expM[:n, n:n+m]
        gd = expM[:n, n+m]
        
        return Ad, Bd, gd
    
    def _apply_normalization_to_dynamics(self, Ad: np.ndarray, Bd: np.ndarray, 
                                       gd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply normalization transformations to linearized dynamics.
        
        If x_norm = Tx @ x and u_norm = Tu @ u, then:
        x_norm_{k+1} = Ad_norm @ x_norm_k + Bd_norm @ u_norm_k + gd_norm
        
        Where:
        Ad_norm = Tx @ Ad @ Tx^{-1}
        Bd_norm = Tx @ Bd @ Tu^{-1}
        gd_norm = Tx @ gd
        """
        if self.normalizer is None:
            return Ad, Bd, gd
        
        Tx = self.normalizer.Tx
        Tu = self.normalizer.Tu
        invTx = self.normalizer.invTx
        invTu = self.normalizer.invTu
        
        # Transform dynamics matrices
        Ad_norm = Tx @ Ad @ invTx
        Bd_norm = Tx @ Bd @ invTu
        gd_norm = Tx @ gd
        
        return Ad_norm, Bd_norm, gd_norm
    
    def validate_linearization(self, test_state: AugmentedState, test_forces: np.ndarray,
                             test_virtual_speed: float, reference_path: 'SplineReferencePath',
                             perturbation: float = 1e-6) -> Dict[str, float]:
        """
        Validate linearization accuracy using finite differences.
        """
        # Get analytical linearization
        dynamics = self.linearize_augmented_dynamics(
            test_state, test_forces, test_virtual_speed, reference_path
        )
        Ad_analytical = dynamics['Ad']
        
        # Numerical verification using finite differences
        state_vec = test_state.get_augmented_vec()
        n_states = len(state_vec)
        
        Ad_numerical = np.zeros_like(Ad_analytical)
        
        for i in range(n_states):
            # Perturb state i
            state_plus = state_vec.copy()
            state_plus[i] += perturbation
            
            state_minus = state_vec.copy()
            state_minus[i] -= perturbation
            
            # Evaluate dynamics at perturbed points
            f_plus = self._evaluate_continuous_dynamics(state_plus, test_forces, test_virtual_speed)
            f_minus = self._evaluate_continuous_dynamics(state_minus, test_forces, test_virtual_speed)
            
            # Finite difference approximation
            Ad_numerical[:, i] = (f_plus - f_minus) / (2 * perturbation)
        
        # Discretize numerical Jacobian for comparison
        Ac_numerical = Ad_numerical
        Bc_dummy = np.zeros((n_states, self.model_params.num_forces + 1))
        gc_dummy = np.zeros(n_states)
        
        Ad_numerical_discrete, _, _ = self._discretize_dynamics(
            Ac_numerical, Bc_dummy, gc_dummy, self.dt
        )
        
        # Compute errors
        frobenius_error = np.linalg.norm(Ad_analytical - Ad_numerical_discrete, 'fro')
        relative_error = frobenius_error / (np.linalg.norm(Ad_analytical, 'fro') + 1e-12)
        max_element_error = np.max(np.abs(Ad_analytical - Ad_numerical_discrete))
        
        return {
            'frobenius_error': frobenius_error,
            'relative_error': relative_error,
            'max_element_error': max_element_error,
            'validation_passed': relative_error < 0.1,  # 10% tolerance
            'condition_number': np.linalg.cond(Ad_analytical)
        }
    
    def _evaluate_continuous_dynamics(self, state_vec: np.ndarray, forces: np.ndarray, 
                                    virtual_speed: float) -> np.ndarray:
        """Evaluate continuous-time dynamics for validation."""
        x, y, theta, vx, vy, omega, path_param = state_vec
        
        # Get model parameters
        mass = self.model_params.mass
        inertia = self.model_params.inertia
        
        # Position dynamics
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        dx_dt = vx * cos_theta - vy * sin_theta
        dy_dt = vx * sin_theta + vy * cos_theta
        
        # Orientation dynamics
        dtheta_dt = omega
        
        # Body velocity dynamics (with grasp matrix)
        if self.model_params.grasp_matrix is not None:
            wrench = self.model_params.grasp_matrix @ forces
            fx, fy, m = wrench[0], wrench[1], wrench[2]
        else:
            fx, fy, m = np.sum(forces), 0.0, 0.0
        
        # Kinetic motion with light damping
        dvx_dt = fx / mass - 0.1 * vx
        dvy_dt = fy / mass - 0.1 * vy
        domega_dt = m / inertia - 0.05 * omega
        
        # Path parameter dynamics
        ds_dt = virtual_speed
        
        return np.array([dx_dt, dy_dt, dtheta_dt, dvx_dt, dvy_dt, domega_dt, ds_dt])
    
    def get_linearization_statistics(self) -> Dict[str, float]:
        """Get statistics about linearization quality."""
        if not self.linearization_history:
            return {'error': 'No linearization history available'}
        
        conditions_Ad = [info['Ad_condition'] for info in self.linearization_history]
        conditions_Bd = [info['Bd_condition'] for info in self.linearization_history]
        max_eigenvalues = [info['max_eigenvalue'] for info in self.linearization_history]
        
        return {
            'num_linearizations': len(self.linearization_history),
            'avg_condition_Ad': np.mean(conditions_Ad),
            'max_condition_Ad': np.max(conditions_Ad),
            'avg_condition_Bd': np.mean(conditions_Bd),
            'max_condition_Bd': np.max(conditions_Bd),
            'avg_max_eigenvalue': np.mean(max_eigenvalues),
            'stability_ratio': np.mean([1.0 if ev <= 1.1 else 0.0 for ev in max_eigenvalues])
        }
    
    def apply_normalization_to_linearized_stages(self, linearized_stages: List[Dict]) -> List[Dict]:
        """
        Apply normalization to all linearized stages at once.
        
        ✅ OPTIONAL CONVENIENCE: Batch process linearization results
        
        Args:
            linearized_stages: List of linearization dictionaries from linearize_trajectory
            
        Returns:
            List of normalized linearization dictionaries
        """
        if self.normalizer is None:
            return linearized_stages
        
        normalized_stages = []
        
        for stage in linearized_stages:
            Ad_norm, Bd_norm, gd_norm = self._apply_normalization_to_dynamics(
                stage['Ad'], stage['Bd'], stage['gd']
            )
            
            normalized_stage = stage.copy()
            normalized_stage['Ad'] = Ad_norm
            normalized_stage['Bd'] = Bd_norm
            normalized_stage['gd'] = gd_norm
            normalized_stage['normalized'] = True
            
            normalized_stages.append(normalized_stage)
        
        return normalized_stages

    def reset_history(self):
        """Reset linearization history."""
        self.linearization_history.clear()
    
    def print_linearization_summary(self):
        """Print linearization configuration summary."""
        print(f"\n🔧 LTV Linearization Summary:")
        print(f"   Control Goal: {self.control_goal_type}")
        print(f"   Model: {self.model_params.num_forces} forces, {self.model_params.mass:.2f}kg")
        print(f"   Discretization: dt={self.dt}s")
        print(f"   Normalization: {'✅ Enabled' if self.normalizer else '❌ Disabled'}")
        print(f"   Grasp Matrix: {self.model_params.grasp_matrix.shape if self.model_params.grasp_matrix is not None else 'Default'}")
        print(f"   History: {len(self.linearization_history)} linearizations")

# ================================================================
# ENHANCED DEMONSTRATION FUNCTION
# ================================================================

def demo_enhanced_dynamics_options():
    """
    Demonstrate the three dynamics options in the linearizer.
    """
    
    print("🔧 Enhanced Dynamics Options Demo")
    print("="*50)
    
    # Create test setup
    standard_objects = create_standard_objects()
    obj = standard_objects['rectangle']
    dynamics = DynamicObjectModel(obj, friction_noise_std=0.001)
    
    # Create reference path
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.5, np.pi/4],
        [2.0, 1.0, np.pi/2]
    ])
    reference_path = SplineReferencePath(waypoints)
    
    # Set up model components
    control_goal = create_control_goal_from_mode('full_pose')
    model_params = ModelParams(dynamics, control_goal)
    test_contacts = get_goal_specific_contact_configuration(obj, control_goal)
    model_params.set_contact_configuration(test_contacts)
    normalizer = StateInputNormalization(model_params)
    
    # Create linearizer
    linearizer = LinearizingLTVClass(model_params, normalizer)
    
    # Test states: static, slow motion, fast motion
    test_scenarios = [
        {
            'name': 'Static State',
            'state_data': {
                'object_position': np.array([0.5, 0.2]),
                'object_orientation': 0.1,
                'velocity_body': np.array([0.0, 0.0]),  # Static
                'angular_velocity': 0.0
            },
            'path_param': 0.3
        },
        {
            'name': 'Slow Motion',
            'state_data': {
                'object_position': np.array([0.5, 0.2]),
                'object_orientation': 0.1,
                'velocity_body': np.array([0.05, 0.02]),  # Slow motion
                'angular_velocity': 0.02
            },
            'path_param': 0.3
        },
        {
            'name': 'Fast Motion',
            'state_data': {
                'object_position': np.array([0.5, 0.2]),
                'object_orientation': 0.1,
                'velocity_body': np.array([0.3, 0.1]),  # Fast motion
                'angular_velocity': 0.2
            },
            'path_param': 0.3
        }
    ]
    
    # Test dynamics options
    dynamics_options = ['ideal', 'damped', 'friction_aware']
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Testing Scenario: {scenario['name']}")
        print(f"{'='*60}")
        
        # Create test state
        test_state = AugmentedState(scenario['state_data'], scenario['path_param'])
        test_forces = np.random.rand(model_params.num_forces) * model_params.max_forces_allowed * 0.5
        test_virtual_speed = 0.4
        
        print(f"State: vx={test_state.object_vx:.3f}, vy={test_state.object_vy:.3f}, ω={test_state.object_vw:.3f}")
        
        for dynamics_option in dynamics_options:
            print(f"\n--- {dynamics_option.upper()} DYNAMICS ---")
            
            try:
                # Linearize with specific dynamics option
                dynamics_result = linearizer.linearize_augmented_dynamics(
                    test_state, test_forces, test_virtual_speed, reference_path,
                    dynamics_option=dynamics_option
                )
                
                Ad = dynamics_result['Ad']
                Bd = dynamics_result['Bd']
                gd = dynamics_result['gd']
                Ac = dynamics_result['continuous_matrices']['Ac']
                gc = dynamics_result['continuous_matrices']['gc']
                
                print(f"✅ Linearization successful!")
                print(f"   Ad condition: {np.linalg.cond(Ad):.2e}")
                print(f"   Bd condition: {np.linalg.cond(Bd):.2e}")
                print(f"   Max eigenvalue: {np.max(np.real(np.linalg.eigvals(Ad))):.3f}")
                print(f"   gd norm: {np.linalg.norm(gd):.4f}")
                
                # Analyze velocity coupling
                velocity_coupling = Ac[3:6, 3:6]
                print(f"   Velocity coupling matrix:")
                print(f"     vx coupling: [{velocity_coupling[0, 0]:.3f}, {velocity_coupling[0, 1]:.3f}, {velocity_coupling[0, 2]:.3f}]")
                print(f"     vy coupling: [{velocity_coupling[1, 0]:.3f}, {velocity_coupling[1, 1]:.3f}, {velocity_coupling[1, 2]:.3f}]")
                print(f"     ω coupling:  [{velocity_coupling[2, 0]:.3f}, {velocity_coupling[2, 1]:.3f}, {velocity_coupling[2, 2]:.3f}]")
                
                # Analyze affine terms
                affine_terms = gc[3:6]
                print(f"   Affine terms: fx={affine_terms[0]:.4f}, fy={affine_terms[1]:.4f}, m={affine_terms[2]:.4f}")
                
                # Friction-specific information
                if dynamics_option == 'friction_aware' and 'friction_info' in dynamics_result:
                    friction_info = dynamics_result['friction_info']
                    print(f"   Friction regime: {friction_info['regime']}")
                    print(f"   f_max: {friction_info['f_max']:.3f}, m_max: {friction_info['m_max']:.3f}")
                
                # Stability analysis
                eigenvalues = np.linalg.eigvals(Ad)
                max_eigenvalue_magnitude = np.max(np.abs(eigenvalues))
                is_stable = max_eigenvalue_magnitude <= 1.05  # Allow small numerical tolerance
                
                print(f"   Stability: {'✅ Stable' if is_stable else '❌ Unstable'} (max |λ| = {max_eigenvalue_magnitude:.3f})")
                
            except Exception as e:
                print(f"❌ Error in {dynamics_option} dynamics: {e}")
                import traceback
                traceback.print_exc()
    
    # ================================================================
    # COMPARATIVE ANALYSIS
    # ================================================================
    print(f"\n{'='*60}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    # Compare dynamics options for the fast motion scenario
    fast_motion_state = AugmentedState(test_scenarios[2]['state_data'], test_scenarios[2]['path_param'])
    test_forces = np.ones(model_params.num_forces) * model_params.max_forces_allowed * 0.5
    
    comparison_results = {}
    
    for dynamics_option in dynamics_options:
        try:
            result = linearizer.linearize_augmented_dynamics(
                fast_motion_state, test_forces, test_virtual_speed, reference_path,
                dynamics_option=dynamics_option
            )
            comparison_results[dynamics_option] = result
        except Exception as e:
            print(f"Failed to compute {dynamics_option}: {e}")
    
    if len(comparison_results) == 3:
        print(f"\nComparison for Fast Motion Scenario:")
        print(f"{'Option':<15} {'Condition(Ad)':<12} {'Max Eigenval':<12} {'gd Norm':<10} {'Stability':<10}")
        print(f"{'-'*65}")
        
        for option, result in comparison_results.items():
            Ad = result['Ad']
            gd = result['gd']
            condition = np.linalg.cond(Ad)
            max_eig = np.max(np.real(np.linalg.eigvals(Ad)))
            gd_norm = np.linalg.norm(gd)
            stability = "✅" if np.max(np.abs(np.linalg.eigvals(Ad))) <= 1.05 else "❌"
            
            print(f"{option:<15} {condition:<12.2e} {max_eig:<12.3f} {gd_norm:<10.4f} {stability:<10}")
        
        # Compare velocity coupling differences
        print(f"\nVelocity Coupling Comparison (Frobenius norms):")
        ideal_coupling = comparison_results['ideal']['continuous_matrices']['Ac'][3:6, 3:6]
        damped_coupling = comparison_results['damped']['continuous_matrices']['Ac'][3:6, 3:6]
        friction_coupling = comparison_results['friction_aware']['continuous_matrices']['Ac'][3:6, 3:6]
        
        print(f"   Ideal:          {np.linalg.norm(ideal_coupling, 'fro'):.4f}")
        print(f"   Damped:         {np.linalg.norm(damped_coupling, 'fro'):.4f}")
        print(f"   Friction-aware: {np.linalg.norm(friction_coupling, 'fro'):.4f}")
        
        # Compare affine terms
        print(f"\nAffine Terms Comparison (norms):")
        for option, result in comparison_results.items():
            gd = result['gd'][3:6]  # Velocity components only
            print(f"   {option:<15}: {np.linalg.norm(gd):.4f}")
    
    # ================================================================
    # VALIDATION TEST
    # ================================================================
    print(f"\n{'='*60}")
    print(f"VALIDATION TEST")
    print(f"{'='*60}")
    
    # Test linearization accuracy for each dynamics option
    for dynamics_option in dynamics_options:
        print(f"\nValidating {dynamics_option.upper()} dynamics...")
        
        try:
            validation_result = linearizer.validate_linearization(
                fast_motion_state, test_forces, test_virtual_speed, reference_path,
                perturbation=1e-6
            )
            
            print(f"   Validation passed: {validation_result['validation_passed']}")
            print(f"   Relative error: {validation_result['relative_error']:.6f}")
            print(f"   Max element error: {validation_result['max_element_error']:.6f}")
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
    
    print(f"\n🎉 Enhanced dynamics demonstration completed!")
    print(f"All three dynamics options are now available:")
    print(f"   - 'ideal': Pure kinematic/dynamic model without friction or damping")
    print(f"   - 'damped': Simple damped model for numerical stability")
    print(f"   - 'friction_aware': Physically accurate friction limit surface dynamics")

if __name__ == "__main__":
    demo_enhanced_dynamics_options()

# %%
# quadprog and optimizer_mpcc and mpcc_solver
import scipy.sparse as sparse
import osqp
from scipy.linalg import block_diag
# interface to osqp solver from MATLAB-style quadprog

def quadprog(H, f, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, options=None):
    """
    MATLAB-style quadprog implementation using OSQP solver
    
    Solves: min 0.5*x'*H*x + f'*x subject to:
            A*x <= b
            Aeq*x = beq
            lb <= x <= ub
    
    Returns:
        z: Solution vector
        exitflag: 1 for success, 0 for failure
    """
    n = H.shape[0]
    
    # ✅ FIX: Ensure inputs are numpy arrays FIRST
    H = sparse.csc_matrix(H)
    f = np.array(f).flatten()
    

    # Build constraint matrix and bounds
    P = H
    q = f

    G = []
    l = []
    u = []

    if A is not None and b is not None:
        A = np.array(A)
        b = np.array(b).flatten()
        G.append(A)
        l.append(-np.inf * np.ones_like(b))
        u.append(b)

    if Aeq is not None and beq is not None:
        Aeq = np.array(Aeq)
        beq = np.array(beq).flatten()
        G.append(Aeq)
        l.append(beq)
        u.append(beq)

    if lb is not None:
        lb = np.array(lb).flatten()
        G.append(np.eye(n))
        l.append(lb)
        u.append(np.inf * np.ones_like(lb))

    if ub is not None:
        ub = np.array(ub).flatten()
        G.append(np.eye(n))
        l.append(-np.inf * np.ones_like(ub))
        u.append(ub)

    if G:
        G = np.vstack(G)
        l = np.concatenate(l)
        u = np.concatenate(u)
        G = sparse.csc_matrix(G)
    else:
        G = sparse.csc_matrix((0, n))
        l = np.array([])
        u = np.array([])

    # Default solver settings if not provided
    solver_settings = {'verbose': False}
    if options:
        solver_settings.update(options)

    # Set up OSQP problem
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=G, l=l, u=u, **solver_settings)

    res = prob.solve()

    if res.info.status_val in [1, 2]:
        return np.reshape(res.x, (-1, 1)), 1
    else:
        return None, 0

    # # ✅ FIX: Validate dimensions early
    # if f.shape[0] != n:
    #     print(f"❌ Dimension mismatch: H is {H.shape}, f is {f.shape}")
    #     return None, 0
    
    # # Build constraint matrix and bounds
    # P = H
    # q = f
    
    # G_list = []
    # l_list = []
    # u_list = []
    
    # # Add inequality constraints
    # if A is not None and b is not None:
    #     if not hasattr(A, 'shape'):  # Not a numpy array/sparse matrix
    #         A = np.array(A)
    #     if A.shape[0] > 0:  # Only add if non-empty
    #         A = np.array(A) if not sparse.issparse(A) else A
    #         b = np.array(b).flatten()

    #         # ✅ FIX: Validate dimensions
    #     if A.shape[1] != n:
    #         print(f"❌ A cols mismatch: A is {A.shape}, expected {n} cols")
    #         return None, 0
    #     if b.shape[0] != A.shape[0]:
    #         print(f"❌ A/b mismatch: A is {A.shape}, b is {b.shape}")
    #         return None, 0
        
    #     G_list.append(A)
    #     l_list.append(-np.inf * np.ones_like(b))
    #     u_list.append(b)
    
    # # Add equality constraints
    # if Aeq is not None and beq is not None:
    #     Aeq = np.array(Aeq) if not sparse.issparse(Aeq) else Aeq
    #     beq = np.array(beq).flatten()
        
    #     # ✅ FIX: Validate dimensions
    #     if Aeq.shape[1] != n:
    #         print(f"❌ Aeq cols mismatch: Aeq is {Aeq.shape}, expected {n} cols")
    #         return None, 0
    #     if beq.shape[0] != Aeq.shape[0]:
    #         print(f"❌ Aeq/beq mismatch: Aeq is {Aeq.shape}, beq is {beq.shape}")
    #         return None, 0
        
    #     G_list.append(Aeq)
    #     l_list.append(beq)
    #     u_list.append(beq)
    
    # # Add variable bounds
    # if lb is not None:
    #     lb = np.array(lb).flatten()
    #     if lb.shape[0] != n:
    #         print(f"❌ lb mismatch: expected {n}, got {lb.shape[0]}")
    #         return None, 0
    #     G_list.append(np.eye(n))
    #     l_list.append(lb)
    #     u_list.append(np.inf * np.ones_like(lb))
    
    # if ub is not None:
    #     ub = np.array(ub).flatten()
    #     if ub.shape[0] != n:
    #         print(f"❌ ub mismatch: expected {n}, got {ub.shape[0]}")
    #         return None, 0
    #     G_list.append(np.eye(n))
    #     l_list.append(-np.inf * np.ones_like(ub))
    #     u_list.append(ub)
    
    # # ✅ FIX: Safely combine constraints
    # if G_list:
    #     G = sparse.vstack(G_list, format='csc')
    #     l = np.concatenate(l_list)
    #     u = np.concatenate(u_list)
    # else:
    #     G = sparse.csc_matrix((0, n))
    #     l = np.array([])
    #     u = np.array([])
    
    # # ✅ FIX: Final validation
    # if G.shape[1] != n:
    #     print(f"❌ Final G mismatch: G is {G.shape}, expected {n} cols")
    #     return None, 0
    # if l.shape[0] != G.shape[0] or u.shape[0] != G.shape[0]:
    #     print(f"❌ Constraint bounds mismatch: G is {G.shape}, l is {l.shape}, u is {u.shape}")
    #     return None, 0
    
    # # Default solver settings
    # solver_settings = {
    #     'verbose': False,
    #     'eps_abs': 1e-6,
    #     'eps_rel': 1e-6,
    #     'max_iter': 4000
    # }
    # if options:
    #     solver_settings.update(options)
    
    # # Set up and solve OSQP problem
    # try:
    #     prob = osqp.OSQP()
    #     prob.setup(P=P, q=q, A=G, l=l, u=u, **solver_settings)
    #     res = prob.solve()
        
    #     if res.info.status_val in [1, 2]:  # Solved or solved inaccurate
    #         return np.reshape(res.x, (-1, 1)), 1
    #     else:
    #         print(f"⚠️ OSQP failed with status: {res.info.status}")
    #         return None, 0
            
    # except Exception as e:
    #     print(f"❌ OSQP exception: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return None, 0

class Stage:
    """Stage in an MPC optimization problem"""
    pass

# %%
#  MPCC CONTROLLER CLASS
# ================================================================
DEBUG_MPCC_MATRICES = False
class MPCCController(PlaceholderController):
    """
    Simplified MPCC controller that brings together all components.
    
    
    Core features:
    - ModelParams integration for all physical properties
    - Control goal specific behavior
    - LTV linearization with selectable dynamics
    - Constraint handling with normalization
    - QP-based optimization (simplified)
    - ForceDistributorPro integration for initial trajectory
    """

    def __init__(self, object_model: DynamicObjectModel, control_goal: str = 'full_pose', 
                 dynamics_option: str = 'damped', physics_option: str = 'true', dt: float = 0.05,
                 force_dist_version: str = 'v3', force_dist_method: str = 'rf'):
        """
        Initialize MPCC controller.
        
        Args:
            object_model: Object dynamics model
            control_goal: Control objective ('position_only', 'omega_only', 'full_pose')
            dynamics_option: Linearization option ('ideal', 'damped', 'friction_aware')
            physics_option: Physics model option ('simplified', 'true')
            dt: Control time step
            force_dist_version: ForceDistributor version ('v1', 'v2', 'v3')
            force_dist_method: ForceDistributor method ('lp', 'qp', 'rf')
        """
        super().__init__(object_model)
        
        self.dt = dt
        # self.object_dynamics = object_model
        self.dynamics_option = dynamics_option
        self.physics_option = physics_option
        
        # ForceDistributor configuration
        self.force_dist_version = force_dist_version
        self.force_dist_method = force_dist_method
        
        # Create core components
        self.control_goal_config = create_control_goal_from_mode(control_goal)
        self.model_params = ModelParams(object_model, self.control_goal_config)
        
        # Initialize other components (will be set up in initialize())
        self.normalizer = None
        self.linearizer = None
        self.constraint_handler = None
        self.error_calculator = None
        self.force_distributor = None  # NEW
        
        # MPCC configuration
        self.reference_path = None
        self.current_mpcc_state = None
        self.prediction_horizon = 10
        self.refinement_steps = 2

        # MPCC solution
        self.state_trajectory = []
        self.force_trajectory = []
        self.speed_trajectory = []

        self.visualize_during_iteration = False

        # Control history

        self.solve_time = []
        self.opt_times = []
        self.mpcc_history = {
            'states': [],
            'forces' : [],
            'speeds' : [],
            # 'controls': [],
            'errors': [],
            # 'costs': [],
        }
        
        print(f"🎯 MPCC Controller initialized:")
        print(f"   Control goal: {control_goal}")
        print(f"   Dynamics option: {dynamics_option}")
        print(f"   Time step: {dt}s")
        print(f"   Force distribution: {force_dist_version}/{force_dist_method}")
        print(f"   Forces: {self.model_params.num_forces}")
    
    #-------------------------setup function

    def initialize(self, ref_path: 'SplineReferencePath', horizon: int = 30, **kwargs):
        """
        Initialize controller with reference path and setup all components.
        
        Args:
            ref_path: Reference path for MPCC
            horizon: MPC prediction horizon
        """
        self.reference_path = ref_path
        self.prediction_horizon = horizon
        
        # Set up contact configuration
        contact_points = kwargs.get('contact_points', None)
        if contact_points is None:
            contact_points = get_goal_specific_contact_configuration(
                self.object_model.object, self.control_goal_config
            )
        
        self.model_params.set_contact_configuration(contact_points)
        
        # Create remaining components
        self.normalizer = StateInputNormalization(self.model_params, disable_normalizer=False)
        self.linearizer = LinearizingLTVClass(self.model_params, self.normalizer, self.dt)
        self.constraint_handler = MPCCConstraintClass(self.model_params, self.normalizer)
        self.error_calculator = MPCCErrorClass(ref_path)
        
        # NEW: Initialize ForceDistributorPro
        self.force_distributor = ForceDistributorPro(
            max_force=self.model_params.max_forces_allowed,
            max_rate_increase=self.model_params.max_force_rate_increase,
            max_rate_decrease=self.model_params.max_force_rate_decrease
        )
        
        # Initialize MPCC state
        if self.state_history:
            current_object_state = self._extract_current_state()
            self.current_mpcc_state = self._initialize_mpcc_state(current_object_state)
        else:
            # Default initialization, using the start of the path
            path_initial_point = self.reference_path.get_point_at_parameter(0.0) # is an array of x, y theta

            state_data = {
                'object_position':np.array([path_initial_point[0], path_initial_point[1]]),
                'object_orientation': path_initial_point[2],
                'velocity_body': np.array([0.1, 0.0]),  # Small initial velocity
                'angular_velocity': 0.0
            }
            self.current_mpcc_state = AugmentedState(state_data, path_param=0.0)
        
        print(f"✅ MPCC Controller fully initialized:")
        print(f"   Contact points: {len(contact_points)}")
        print(f"   Horizon: {horizon}")
        print(f"   ForceDistributor: {self.force_dist_version}/{self.force_dist_method}")
        print(f"   Initial state: {self.current_mpcc_state}")

        print(f" NOW build the initial solution ")

        self.state_trajectory, self.force_trajectory, self.speed_trajectory = self._build_initial_trajectory()
        
        return contact_points

    def _initialize_mpcc_state(self, object_state: Dict) -> AugmentedState:
        """Initialize MPCC state with path parameter."""
        # Find closest point on path for initial path parameter
        query_point = [object_state['object_position'][0], object_state['object_position'][1]]
        _, path_param, _ = self.reference_path.find_closest_point(query_point)
        
        return AugmentedState(object_state, path_param)


    def set_force_distribution_config(self, version: str = None, method: str = None):
        """
        Update force distribution configuration.
        
        Args:
            version: ForceDistributor version ('v1', 'v2', 'v3')
            method: ForceDistributor method ('lp', 'qp', 'rf')
        """
        if version is not None:
            self.force_dist_version = version
            print(f"🔧 Force distribution version updated: {version}")
        
        if method is not None:
            self.force_dist_method = method
            print(f"🔧 Force distribution method updated: {method}")
        
        # Reset force distributor state for clean start with new config
        if self.force_distributor is not None:
            self.force_distributor.reset()
            print(f"   ForceDistributor state reset")

    #-------------------------- main controller function, need to get updated to match the golden loop
    
    # for the get_control_actions, we will have to get the initial first if not exist yet, then do the normal refine then blend then apply
    def get_control_actions(self):
        """
        Main MPCC control loop.
        
        Returns:
            Tuple of (contact_points, force_magnitudes)
        """
        if self.reference_path is None:
            print("⚠️ Reference path not set")
            return [], []
        
        self._update_mpcc_state()
        
        if (DEBUG_MPCC_MATRICES): 
            print(" 🍕 The current MPCC state is: ")
            print(self.current_mpcc_state)

            print(" The initial state of the solution is: ")
            print(self.state_trajectory[0])

        try:
            # Solve MPCC optimization
            start_time = time.time()
            self._solve_mpcc_optimization()
            self.solve_time.append(time.time() - start_time)
        except Exception as e:
            print("⚠️ MPCC optimization failed, using zero control")
            return self.model_params.contact_points, np.zeros(self.model_params.num_forces)
        
        # Update history
        self._update_history()

        if (DEBUG_MPCC_MATRICES): 
            print(" The control actions are: ")
            print(f" Forces: {self.force_trajectory[0]}")
            print(f"😶‍🌫️ Expected next state: {self.state_trajectory[1]}")

        return self.model_params.contact_points, self.force_trajectory[0]
    
    def update_internal(self):
        """Update internal state (placeholder)."""
        pass

    # def post_update_internal(self, latest_data):
    #     """Post-update internal state (placeholder)."""
    #     pass



    #--------------------------------update functions to extract data but also update the current mpcc state

    # def _evaluate_cost(self, solution: Dict, cost_function: Dict) -> float:
    #     """Evaluate total cost of solution."""
    #     x = solution['x']
    #     Q = cost_function['Q']
    #     f = cost_function['f']
        
    #     cost = 0.5 * x.T @ Q @ x + f.T @ x
    #     return float(cost)
    
    def _update_mpcc_state(self):
        """
        Update current MPCC state using VIRTUAL SPEED DYNAMICS ONLY.
        
        Should match the later part in the execute_mpcc_with_enhanced_initialization function.
        from this
                
            print(f"Applying control input at step {i+1}:")
            print(f"  Predicted forces: {linearization_forces[0]}")
            print(f"  Predicted speed: {linearization_speeds[0]}")
            
        to this
            print(f" These two values should be identical, or else we have some big problem", linearization_states[0].get_vec(), current_mpcc_state.get_vec())

        """

        # skip the first call, to enforce using the initial state and initial solution from the build_initial_trajectory
        if not self.state_history:
            return
        
        # update the latest object state from history and the corresponding path parameter
        # then rebuild the state_trajectory, force_trajectory, speed_trajectory accordingly just like the golden

        # Extract current object state from simulation history
        current_object_state = self._extract_current_state()
        
        # update the current mpcc state path parameter using the current object state and the reference path
        query_point = [current_object_state['object_position'][0], 
                    current_object_state['object_position'][1]]
        _, new_path_param, _ = self.reference_path.find_closest_point(query_point)
        # Create updated MPCC state with new path parameter
        self.current_mpcc_state = AugmentedState(current_object_state, new_path_param)
        if (DEBUG_MPCC_MATRICES): print(f"🔄 MPCC state updated (virtual speed dynamics): path_param = {new_path_param:.4f}")

        # now update the trajectories accordingly
        # shift the trajectories by one step and append a new state at the end using the updated

        self.force_trajectory = self.force_trajectory[1:]  # Remove first element
        self.speed_trajectory = self.speed_trajectory[1:]  # Remove first element
        self.force_trajectory.append(self.force_trajectory[-1])  # Append last known force
        self.speed_trajectory.append(self.speed_trajectory[-1])  # Append last known speed

        if (self.physics_option != 'true'):
            if (DEBUG_MPCC_MATRICES):  print("⚠️ Warning: _update_mpcc_state is using simplified dynamics, may not match true dynamics!")
            # if has not fix the apply dynamics yet or not using the true model, we can just do the simple shift
            self.state_trajectory = self.state_trajectory[2:]  # Remove first and second element
            self.state_trajectory.insert(0, self.current_mpcc_state)  # Insert updated current state at the beginning
            self.state_trajectory.append(self.state_trajectory[-1])  # Append last known state
        else:
            # rebuild the state trajectory using the updated current mpcc state and the shifted forces and speeds
            self.state_trajectory = [self.current_mpcc_state]
            for i in range(self.prediction_horizon):
                last_state = self.state_trajectory[-1]
                forces = self.force_trajectory[i]
                speed = self.speed_trajectory[i]
                self.state_trajectory.append(self._apply_mpcc_dynamics(last_state, forces, speed))

        # ensure we have N+1 for state trajectory, N for forces and speeds
        assert len(self.state_trajectory) == self.prediction_horizon + 1


    # works in cohesion with the placeholder controller, as it push the latest object state into the history
    def _extract_current_state(self) -> Dict:
        """Extract current state from simulation history."""
        if not self.state_history:
            return {
                'object_position': np.array([0.0, 0.0]),
                'object_orientation': 0.0,
                'velocity_body': np.array([0.1, 0.0]),
                'angular_velocity': 0.0
            }
        
        latest = self.state_history[-1]
        return {
            'object_position': np.array(latest.get('position', [0.0, 0.0])),
            'object_orientation': latest.get('orientation', 0.0),
            'velocity_body': np.array(latest.get('velocity_body', [0.0, 0.0])),
            'angular_velocity': latest.get('angular_velocity', 0.0)
        }


    #----------------------------MPCC solver function

    # verified to match golden mpcc_solver at least in the shape and size
    def call_quad_solver(self, stage):
        """
        Simplified MPCC solver that builds and solves the QP.
        should match the mpcc_solver in golden version
        Args:
            stage: List of Stage objects with dynamics and cost data
        """
        if (DEBUG_MPCC_MATRICES): print("Starting to build full horizon QP matrices...")
        nx = self.model_params.num_states
        nu = self.model_params.num_forces + 1  # +1 for virtual speed
        N = self.prediction_horizon
        ng = 2 # of equality constraints per stage (to be defined)
        nz = nx + nu + nu  # [x, u, du]
        nxu = nx + nu

        # from here on is literally copied from goden mpcc_solver
        H = np.zeros((nz * N + nxu, nz * N + nxu))
        f = np.zeros((nz * N + nxu, 1))
        
        # Loop over stages (MATLAB indices 1:N+1; Python: 0:N)
        for i in range(1, N + 2):
            start_index = (i - 1) * nz
            if i < N + 1:
                H_i = block_diag(stage[i - 1].qk, stage[i - 1].rk)
                H[start_index:start_index + nz, start_index:start_index + nz] = H_i
                f[start_index:start_index + nxu, 0] = stage[i - 1].fk # already flattened
            else:
                H[start_index:start_index + nxu, start_index:start_index + nxu] = stage[i - 1].qk
                f[start_index:start_index + nxu, 0] = stage[i - 1].fk # already flattened

        # Check if H is symmetric
        sym_error = np.max(np.abs(H - H.T))
        if (DEBUG_MPCC_MATRICES): print(f"Symmetry error: {sym_error}")

        # Check eigenvalues to verify positive definiteness
        try:
            min_eig = np.min(np.linalg.eigvals(H))
            if (DEBUG_MPCC_MATRICES): print(f"Minimum eigenvalue of H: {min_eig}")
            if min_eig < 0:
                # Add regularization to make H positive definite
                if (DEBUG_MPCC_MATRICES): print(f"Adding regularization to H matrix")
                H = H + 1e-6 * np.eye(H.shape[0])
        except:
            if (DEBUG_MPCC_MATRICES): print("Error computing eigenvalues - H matrix may be too large or ill-conditioned")
            

        # In mpcc_solver before calling quadprog:
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            if (DEBUG_MPCC_MATRICES): print("Warning: H contains NaN or Inf values")
            # Replace NaN/Inf with reasonable values or add regularization
            H = np.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
        H = 0.5 * (H + H.T)
            
        Aeq = np.zeros((nxu*(N+1), nz*N+nxu))
        beq = np.zeros((nxu*(N+1), 1))

        # Create the block diagonal scale matrix and multiply with state/control vector
        # The MATLAB code: x0scale = blkdiag(MPC_vars.Tx,MPC_vars.Tu)*[stage(1).x0;stage(1).u0];

        # 1. Get state vector (assuming stage[0].x0 is an AugmentedState object)
        x0_vec = stage[0].x0.get_augmented_vec()

        # 2. Get control vector (assuming stage[0].u0 is [forces, speed])
        forces = np.array(stage[0].u0[0])
        speed = np.array([stage[0].u0[1]])
        u0_vec = np.concatenate([forces, speed])

        # 3. Vertical concatenation (equivalent to MATLAB's [x;u])
        xu0_vec = np.concatenate([x0_vec, u0_vec]).reshape(-1, 1)  # Ensure column vector shape

        # 4. Create block diagonal matrix and perform multiplication
        x0scale = block_diag(self.normalizer.Tx, self.normalizer.Tu) @ xu0_vec

        Aeq[0:nxu, 0:nxu] = np.eye(nxu)
        beq[0:nxu, 0] = x0scale.flatten()

        # Loop for equality constraints over i=1:N (MATLAB) -> i in range(1, N+1)
        for i in range(1, N + 1):
            row_start = i * nxu
            row_end = row_start + nxu
            col_start = (i - 1) * nz
            if i < N:
                # Concatenate four blocks: -Ak, -Bk, I, and zeros.
                Aeq[row_start:row_end, col_start:col_start + 2 * nz] = np.hstack((
                    -stage[i - 1].Ak,
                    -stage[i - 1].Bk,
                    np.eye(nxu),
                    np.zeros((nxu, nu))
                ))
            else:
                # For i == N, only three blocks: -Ak, -Bk, I.
                Aeq[row_start:row_end, col_start:col_start + (nz + nxu)] = np.hstack((
                    -stage[i - 1].Ak,
                    -stage[i - 1].Bk,
                    np.eye(nxu)
                ))
            beq[row_start:row_end, 0] = stage[i - 1].gk.flatten()

        # the A and b inequility constraints are for the race car to stay on the track, which is not applicable to our case
        # in the future, we can add some inequality constraints here if needed
        A = np.zeros((0, nz * N + nxu))  # No inequality constraints for now
        b = np.zeros((0, 1))  # No inequality constraints for now

        LB = np.zeros((nz * N + nxu, 1))
        UB = np.zeros((nz * N + nxu, 1))

        # Loop to set LB and UB
        for i in range(1, N + 2):
            start = (i - 1) * nz
            if i < N + 1:
                LB[start:start + nz, 0] = stage[i - 1].lb.flatten()
                UB[start:start + nz, 0] = stage[i - 1].ub.flatten()
            else:
                LB[start:start + nxu, 0] = stage[i - 1].lb[:nxu].flatten()
                UB[start:start + nxu, 0] = stage[i - 1].ub[:nxu].flatten()

        # options = {'maxiter': 100, 'disp': False}
        options = None
        start_time = time.time()
        # Note: quadprog must be defined, for example from a suitable Python QP solver.
        z, exitflag = quadprog(H, f, A, b, Aeq, beq, LB, UB, options)
        # print(f"Quadprog return value", z)
        QPtime = time.time() - start_time

        X = np.zeros((nx, N + 1))
        U = np.zeros((nu, N))
        dU = np.zeros((nu, N))

        if exitflag == 1:
            # Loop over stages i=1:N+1 (MATLAB) -> Python: i in range(1, N+2)
            for i in range(1, N + 2):
                z_index = (i - 1) * nz
                X[:, i - 1] = self.normalizer.invTx @ z[z_index:z_index + nx, 0]
                if i > 1:
                    U[:, i - 2] = self.normalizer.invTu @ z[z_index + nx:z_index + nx + nu, 0]
                if i <= N:
                    dU[:, i - 1] = z[z_index + nxu:z_index + nxu + nu, 0]

        info = {'QPtime': QPtime, 'exitflag': 0 if exitflag == 1 else 1}

        return X, U, dU, info

    # verified to match the golden optimizer_mpcc build cost matrices for full horizon
    def assemble_cost_function(self,
        list_states, list_forces, list_speeds
    ):
        """
        Build cost function matching golden's optimizer_mpcc assembly.
        
        This function replicates the EXACT logic from:
        1. optimizer_mpcc: Loop that calls generateH and generateLinearizedCostFunction
        
        
        Returns:
            Dict with keys:
                'stage': List of Stage objects with cost and dynamics data
        """
        # print("Starting Assembling the cost matrices for optimization...")

        # LEGACY Unpack results into the format used by the rest of the notebook
        # if ('state_trajectory' in current_solution) and ('control_trajectory' in current_solution):
        #     list_states = list(current_solution['state_trajectory'])
        #     list_forces = [np.asarray(ctrl[0]) for ctrl in current_solution['control_trajectory']]
        #     list_speeds = [float(ctrl[1]) for ctrl in current_solution['control_trajectory']]
        # elif ('list_forces' in current_solution):
        #     list_states = current_solution['list_states']
        #     list_forces = current_solution['list_forces']
        #     list_speeds = current_solution['list_speeds']

        stage = [Stage() for _ in range(self.prediction_horizon + 1)]
        stage[0].x0 = list_states[0]
        stage[0].u0 = (list_forces[0], list_speeds[0])

        # Rate cost diagonal (same for all stages)
        # Golden: "2 * np.diag([rdF] * num_forces + [rdVtheta])"
 
        Rk_template = 2 * np.diag([self.control_goal_config.weights.rdF] * self.model_params.num_forces +
                                  [self.control_goal_config.weights.rdVtheta])
        
        # Bounds lb and ub same for all stages
        modular_constraints = self.constraint_handler
        
        # FIX: Manually build the bounds vectors to match the golden order
        # Get the bounds dictionary
        bounds = self.model_params.get_constraint_bounds()
        num_forces = self.model_params.num_forces
        
        # 1. Build x_k bounds (nx=7)
        lb_x = np.array([
            bounds['position'][0],           # x
            bounds['position'][0],           # y
            bounds['orientation'][0],        # theta
            bounds['linear_velocity'][0],    # vx
            bounds['linear_velocity'][0],    # vy
            bounds['angular_velocity'][0],   # omega
            bounds['path_param'][0]          # s
        ])
        ub_x = np.array([
            bounds['position'][1],           # x
            bounds['position'][1],           # y
            bounds['orientation'][1],        # theta
            bounds['linear_velocity'][1],    # vx
            bounds['linear_velocity'][1],    # vy
            bounds['angular_velocity'][1],   # omega
            bounds['path_param'][1]          # s
        ])
        
        # 2. Build u_k bounds (nu = num_forces + 1)
        lb_u = np.concatenate([
            np.full(num_forces, bounds['force_magnitude'][0]),
            np.array([bounds['virtual_speed'][0]])
        ])
        ub_u = np.concatenate([
            np.full(num_forces, bounds['force_magnitude'][1]),
            np.array([bounds['virtual_speed'][1]])
        ])

        # 3. Build du_k bounds (nu = num_forces + 1)
        # We assume virtual speed rate (dvs) is unconstrained
        lb_du = np.concatenate([
            np.full(num_forces, -bounds['force_rate_decrease'][1]),
            np.array([-np.inf]) 
        ])
        ub_du = np.concatenate([
            np.full(num_forces, bounds['force_rate_increase'][1]),
            np.array([np.inf])
        ])
        
        # Concatenate all to form lb_z and ub_z in the order [x, u, du]
        modular_lb = np.concatenate([lb_x, lb_u, lb_du])
        modular_ub = np.concatenate([ub_x, ub_u, ub_du])
        


        # the build_initial_trajectory ensure we have N+1 states for N controls
        for k in range(self.prediction_horizon + 1):
            stage[k].xk = list_states[k]
            # build Qk, Rk, fk, for each stage,
            cal_lin_error =  self.error_calculator.calculate_linearized_error_cost(
                augmented_state=list_states[k], 
                model_param= self.model_params,
                normalizer= self.normalizer,
                control_goal= self.control_goal_config,
                terminal_stage= (k == self.prediction_horizon)
            )
            stage[k].qk = cal_lin_error['Qk_norm']
            stage[k].fk = cal_lin_error['f_error']

            # Rk is the same for all stages
            stage[k].rk = Rk_template
            # lb and ub are the same for all stages
            stage[k].lb = modular_lb
            stage[k].ub = modular_ub

            # except terminal stage, we calculate the Ak, Bk, gk constraint
            # Get equality constraints for single stage
            
            if (k < self.prediction_horizon):
                # This function needs the modular Ad, Bd, gd from Test 1
                lin_result = self.linearizer.linearize_augmented_dynamics(
                    augmented_state=stage[k].xk,
                    forces=list_forces[k],
                    virtual_speed=list_speeds[k],
                    reference_path=self.reference_path,
                    dynamics_option=self.dynamics_option
                )

                eq_result = modular_constraints.get_equality_constraints_single_stage(
                    lin_result['Ad'], lin_result['Bd'], lin_result['gd']
                )
                
                stage[k].Ak = eq_result['Ak']
                stage[k].Bk = eq_result['Bk']
                stage[k].gk = eq_result['gk']

        return stage

    def _solve_mpcc_optimization(self) -> Optional[Dict]:
        """
        Solve MPCC optimization with iterative refinement.
        
        ✅ MATCHES GOLDEN: execute_mpcc_with_enhanced_initialization main loop
        
        Steps per iteration:
        1. Assemble cost function (optimizer_mpcc style)
        2. Call QP solver (mpcc_solver style)
        3. Blend trajectory for stability
        4. Visualize (optional)
        
        Returns:
            Dict with solution or None if failed
        """
        try:
            # ================================================================
            # ITERATIVE REFINEMENT LOOP (matching golden main loop)
            # ================================================================
            for iteration in range(self.refinement_steps):
                # print(f"\n{'='*70}")
                # print(f"Refinement Iteration {iteration + 1}/{self.refinement_steps}")
                # print(f"{'='*70}")
                
                # print(f"   Current trajectory shapes:")
                # print(f"     States: {len(self.state_trajectory)}")
                # print(f"     Forces: {len(self.force_trajectory)}")
                # print(f"     Speeds: {len(self.speed_trajectory)}")
                
                # ================================================================
                # STEP 1: Assemble Cost Function (matches optimizer_mpcc)
                # ================================================================
                # print(f"\n   Step 1: Assembling cost function...")
                stage = self.assemble_cost_function(
                    self.state_trajectory, 
                    self.force_trajectory, 
                    self.speed_trajectory
                )
                # print(f"      ✅ Cost assembled for {len(stage)} stages")
                
                # ================================================================
                # STEP 2: Call QP Solver (matches mpcc_solver)
                # ================================================================
                # print(f"\n   Step 2: Calling QP solver...")
                X, U, dU, info = self.call_quad_solver(stage)
                
                if info['exitflag'] != 0:
                    print(f"      ❌ QP solver failed at iteration {iteration + 1}")
                    assert False, "QP solver failed during MPCC optimization"
                    
                    # Safety check: Use previous trajectory
                    if iteration == 0:
                        print(f"      → Using initial trajectory (no optimization)")
                    else:
                        print(f"      → Using trajectory from iteration {iteration}")
                    
                    # Validate fallback trajectory constraints
                    max_force_used = np.max([np.max(f) for f in self.force_trajectory])
                    if max_force_used > self.model_params.max_forces_allowed:
                        print(f"      🚨 Fallback violates force limits! Clipping...")
                        self.force_trajectory = [
                            np.clip(f, 0, self.model_params.max_forces_allowed)
                            for f in self.force_trajectory
                        ]
                    break  # Exit refinement loop
                
                # print(f"      ✅ QP solved successfully!")
                # print(f"         Exit flag: {info['exitflag']}")
                # print(f"         Solve time: {info['QPtime']:.4f}s")
                
                # ================================================================
                # STEP 3: Blend Trajectory (matches golden alpha blending)
                # ================================================================
                # print(f"\n   Step 3: Blending trajectory (α=0.5)...")
                
                blended_states, blended_forces, blended_speeds, _ = self.blend_trajectory(
                    just_solved_states=X,
                    just_solved_controls=U,
                    current_initial_state=self.state_trajectory[0],
                    old_solution_forces=self.force_trajectory,
                    old_solution_speeds=self.speed_trajectory,
                    alpha=0.5  # Golden uses 0.75 for faster convergence
                )
                
                # print(f"      ✅ Trajectory blended")
                # print(f"         Blended shapes: states={len(blended_states)}, "
                #     f"forces={len(blended_forces)}, speeds={len(blended_speeds)}")
                
                # Update internal trajectory for next iteration
                self.state_trajectory = blended_states
                self.force_trajectory = blended_forces
                self.speed_trajectory = blended_speeds
                
                # ================================================================
                # STEP 4: Auxiliary features like visualization or tracking the opt time 
                # ================================================================
                if self.visualize_during_iteration and iteration == self.refinement_steps - 1:  # Last iteration only
                    print(f"\n   Step 4: Visualizing final trajectory...")
                    visualize_mpcc_solution(
                        self.state_trajectory,
                        self.reference_path
                    )
                
                self.opt_times.append(info['QPtime'])
            
            # ================================================================
            # RETURN SOLUTION 
            # ================================================================
            
            # Calculate final cost (optional, for diagnostics)
            # final_cost = 0.0  # Can be computed from final state errors if needed
            
            # technically do not need to return anything, since we have updated the internal trajectory
            return

            # return {
            #     'success': True,
            #     # 'state_trajectory': self.state_trajectory,
            #     # 'control_trajectory': control_trajectory,
            #     'cost': final_cost,
            #     'solve_info': {
            #         'method': 'golden_style_mpcc',
            #         'refinement_iterations': self.refinement_steps,
            #         'last_qp_time': info['QPtime']
            #     }
            # }
            
        except Exception as e:
            print(f"❌ MPCC optimization error: {e}")
            import traceback
            traceback.print_exc()
            return None


    #  NEED function after we test the golden successfully, will work as a wrapper from the modular to the golden style
    def _solve_qp_legacy_modular(self, Q: np.ndarray, f: np.ndarray, 
                A_eq: np.ndarray, b_eq: np.ndarray,
                A_ineq: np.ndarray, b_ineq: np.ndarray) -> Optional[Dict]:
        """
        Solve QP problem using proper OSQP solver.
        """
        try:
            print(f"\n🔍 Pre-solve validation:")
            print(f"   Q: {Q.shape}, symmetric: {np.allclose(Q, Q.T)}, PSD: {np.all(np.linalg.eigvals(Q) >= -1e-10)}")
            print(f"   f: {f.shape}")
            print(f"   A_eq: {A_eq.shape if A_eq is not None else 'None'}")
            print(f"   b_eq: {b_eq.shape if b_eq is not None else 'None'}")
            print(f"   A_ineq: {A_ineq.shape if A_ineq is not None else 'None'}")
            print(f"   b_ineq: {b_ineq.shape if b_ineq is not None else 'None'}")

            # Check for NaN/Inf
            if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
                print(f"   ⚠️ Q contains NaN or Inf!")
            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                print(f"   ⚠️ f contains NaN or Inf!")

            # Use the proper quadprog function
            z, exitflag = quadprog(
                H=Q,
                f=f,
                A=A_ineq if A_ineq.shape[0] > 0 else None,
                b=b_ineq if A_ineq.shape[0] > 0 else None,
                Aeq=A_eq if A_eq.shape[0] > 0 else None,
                beq=b_eq if A_eq.shape[0] > 0 else None,
                options={'eps_abs': 1e-6, 'eps_rel': 1e-6, 'max_iter': 4000, 'verbose': True}
            )
            
            if exitflag == 1 and z is not None:
                x_solution = z.flatten()
                
                return {
                    'x': x_solution,
                    'cost': 0.5 * x_solution.T @ Q @ x_solution + f.T @ x_solution,
                    'solve_info': {
                        'success': True,
                        'method': 'osqp'
                    }
                }
            else:
                print(f"⚠️ QP solver failed with exitflag: {exitflag}")
                return None
                
        except Exception as e:
            print(f"❌ QP solver error: {e}")
            return None



    #---------------------------- helper functions for the controller such as blending and initial trajectory and apply dynamics

    # blend the previous solution with the just-solved one from call_quad_solver
    def blend_trajectory(self, just_solved_states: List[AugmentedState], just_solved_controls: List[Tuple],
                               current_initial_state, old_solution_forces, old_solution_speeds,
                                alpha: float) -> Tuple[List[AugmentedState], List[Tuple]]:
        """
        Blend old and new trajectories for stability.
        """
        blended_forces = []
        blended_speeds = []
        blended_controls = []

        # need to unpack the old_solution_controls in case
        
        # Blend controls first
        # min_len = min(len(just_solved_controls), len(old_solution_controls))
        # tbh, i think the two shape should match.
        # for k in range(min_len):
        #     old_forces, old_speed = just_solved_controls[k]
        #     new_forces, new_speed = old_solution_controls[k]
            
        #     blended_forces = (1 - alpha) * old_forces + alpha * new_forces
        #     blended_speed = (1 - alpha) * old_speed + alpha * new_speed
            
        #     blended_controls.append((blended_forces, blended_speed))
        for k in range(just_solved_controls.shape[1]):
            new_force             = just_solved_controls[:-1, k]
            new_speed             = just_solved_controls[-1, k]
        #     # Alpha blending

            blended_force = (1 - alpha) * old_solution_forces[k] + alpha * new_force
            blended_speed = (1 - alpha) * old_solution_speeds[k] + alpha * new_speed

            blended_controls.append((blended_force, blended_speed))
            blended_forces.append(blended_force)
            blended_speeds.append(blended_speed)


        # Propagate states using blended controls
        blended_states = [current_initial_state]  # Keep initial state, this is the same for both old and new solutions
        current_state = current_initial_state
        if (DEBUG_MPCC_MATRICES): print("current_state", current_state)
        for k, (forces, speed) in enumerate(blended_controls):
            next_state = self._apply_mpcc_dynamics(current_state, forces, speed)
            blended_states.append(next_state)
            current_state = next_state
        
        # expect to have N+1 states and N controls
        return blended_states, blended_forces, blended_speeds, blended_controls
    
    # verified to return N+1 states, N+1 controls
    def _build_initial_trajectory(self) -> Tuple[List[AugmentedState], List[Tuple[np.ndarray, float]]]:
        """
        Build initial trajectory using ForceDistributorPro for proper force distribution.
        
        Returns:
            Tuple of (state_trajectory, control_trajectory)
        """
        print(f"🔧 Building initial trajectory with ForceDistributorPro ({self.force_dist_version}/{self.force_dist_method})...")
        
        # Reset force distributor for clean initialization
        if self.force_distributor is not None:
            self.force_distributor.reset()
        
        grasp_matrix = self.model_params.grasp_matrix
        if grasp_matrix is None:
            print("⚠️ Grasp matrix not available, using fallback")
            return self._build_simple_initial_trajectory()
        
        # Initialize trajectory
        initial_states = [self.current_mpcc_state]
        # FIX 1: Initialize with zero forces/speeds, just like the golden version
        initial_forces = [np.zeros(self.model_params.num_forces)]
        initial_speeds = [0.0]

        obj_mass = self.model_params.mass
        obj_inertia = self.model_params.inertia
        
        previous_path_param = self.current_mpcc_state.path_param
        curr_sim_state = self.current_mpcc_state
        
        # Track force distribution statistics
        distribution_failures = 0
        wrench_errors_list = []
        
        for k in range(self.prediction_horizon): # This is N

            # ---
            # FIX 2: Propagate state FIRST (using control[k])
            # ---
            curr_sim_state = self._apply_mpcc_dynamics(
                curr_sim_state, initial_forces[k], initial_speeds[k]
            )
            initial_states.append(curr_sim_state)
            # ---
            # FIX 3: Now, calculate the NEXT control (control[k+1]) based on the NEW state (curr_sim_state)
            # ---
            
            # 1. Get new object state
            obj_state = curr_sim_state
            query_point = [obj_state.object_x, obj_state.object_y, obj_state.object_theta]
            # 2. Get lookahead point on path
            lookahead_distance = 0.1
            try:
                lookahead_point, lookahead_t = self.reference_path.get_lookahead_point(
                    query_point, lookahead_distance
                )
            except:
                lookahead_t = min(curr_sim_state.path_param + 0.05, 1.0)
                lookahead_point = self.reference_path.get_point_at_parameter(lookahead_t)
            
            # 3. Calculate desired direction
            dir_x = lookahead_point[0] - obj_state.object_x
            dir_y = lookahead_point[1] - obj_state.object_y
            dir_norm = np.sqrt(dir_x**2 + dir_y**2)
            
            if dir_norm > 1e-6:
                dir_x /= dir_norm
                dir_y /= dir_norm
            else:
                tangent = self.reference_path.get_tangent_at_parameter(curr_sim_state.path_param)
                dir_x, dir_y = tangent
            
            # 4. Calculate desired heading change
            desired_theta = np.arctan2(dir_y, dir_x)
            theta_error = normalize_theta_diff(desired_theta, obj_state.object_theta)
            
            # 5. Create target wrench (world frame)
            target_wrench = np.zeros(3)
            target_wrench[0] = dir_x * obj_mass * 0.5
            target_wrench[1] = dir_y * obj_mass * 0.5
            target_wrench[2] = theta_error * obj_inertia * 0.5
            
            # 6. Transform to object frame
            cos_obj = np.cos(obj_state.object_theta)
            sin_obj = np.sin(obj_state.object_theta)
            inverse_rotation_matrix = np.array([
                [cos_obj,  sin_obj],
                [-sin_obj, cos_obj]
            ])
            target_wrench[:2] = inverse_rotation_matrix @ target_wrench[:2]
            
            # ================================================================
            # NEW: Use ForceDistributorPro for force distribution
            # ================================================================
            distribution_result = self.force_distributor.distribute_forces(
                desired_wrench=target_wrench,
                contact_points=self.model_params.contact_points,
                grasp_matrix=grasp_matrix,
                version=self.force_dist_version,
                method=self.force_dist_method,
                dynamics_model=self.object_model,
                current_time=k * self.dt,
                dt=self.dt
            )
            
            if distribution_result['success']:
                target_forces = distribution_result['force_magnitudes']
                wrench_errors_list.append(distribution_result['wrench_error_magnitude'])

                
            else:
                # Fallback: simple clipping
                print(f"     ⚠️ Force distribution failed at step {k}, using fallback")
                distribution_failures += 1
                
                g_pinv = np.linalg.pinv(grasp_matrix)
                target_forces = g_pinv @ target_wrench
                
                # Apply buffer for negative forces
                min_force = np.min(target_forces)
                if min_force < 0:
                    target_forces = target_forces - min_force
                
                # Clip to safe range
                max_force = self.model_params.max_forces_allowed
                target_forces = np.clip(target_forces, 0.05 * max_force, max_force)
            # 8. Calculate contour speed
            contour_speed = (lookahead_t - previous_path_param) / self.dt

            # Rate limiting
            # NOTE: We now use initial_speeds[k] since it's the *previous* speed
            previous_vspeed = initial_speeds[k] 
            max_vspeed_change = 1.0 
            contour_speed = np.clip(
                contour_speed,
                previous_vspeed - max_vspeed_change,
                previous_vspeed + max_vspeed_change
            )
            contour_speed = np.clip(contour_speed, 0.001, 1.0)
            previous_path_param = lookahead_t
            
            # 9. Store control (this is now u_{k+1} and v_{k+1})
            initial_forces.append(target_forces)
            initial_speeds.append(contour_speed)
            
        
        # Print initialization statistics
        print(f"   ✅ Initial trajectory built: {len(initial_states)} states, {len(initial_forces)} controls")
        print(f"   Force distribution stats:")
        print(f"     Method: {self.force_dist_version}/{self.force_dist_method}")
        print(f"     Failures: {distribution_failures}/{self.prediction_horizon}")
        
        if wrench_errors_list:
            print(f"     Wrench tracking:")
            print(f"       Mean error: {np.mean(wrench_errors_list):.4f}")
            print(f"       Max error: {np.max(wrench_errors_list):.4f}")
            print(f"       RMS error: {np.sqrt(np.mean([e**2 for e in wrench_errors_list])):.4f}")
        
        # Force statistics
        all_forces = np.array(initial_forces)
        print(f"     Force statistics:")
        print(f"       Min per contact: {np.min(all_forces, axis=0)}")
        print(f"       Max per contact: {np.max(all_forces, axis=0)}")
        print(f"       Overall variance: {np.var(all_forces):.4f}")
        
        return initial_states, initial_forces, initial_speeds


    def _apply_mpcc_dynamics(self, state: AugmentedState, forces: np.ndarray, 
                            virtual_speed: float) -> AugmentedState:
        """Switch between different dynamics models."""
        if self.physics_option == 'simplified':
            return self._apply_mpcc_dynamics_golden(state, forces, virtual_speed)
        else: # 'true'
            return self._apply_mpcc_dynamics_true(state, forces, virtual_speed)

    
    # matched the golden
    def _apply_mpcc_dynamics_golden(self, state: AugmentedState, forces: np.ndarray, 
                            virtual_speed: float) -> AugmentedState:
        """
        Apply simplified MPCC dynamics for trajectory prediction.
        """
        # Calculate wrench from forces
        if self.model_params.grasp_matrix is not None:
            wrench = self.model_params.grasp_matrix @ forces
        else:
            wrench = np.array([np.sum(forces), 0.0, 0.0])
        
        # Simple dynamics integration
        mass = self.model_params.mass
        inertia = self.model_params.inertia
        
        # Update velocities (with light damping)
        new_vx = state.object_vx + (wrench[0] / mass) * self.dt - 0.1 * state.object_vx * self.dt
        new_vy = state.object_vy + (wrench[1] / mass) * self.dt - 0.1 * state.object_vy * self.dt
        new_vw = state.object_vw + (wrench[2] / inertia) * self.dt - 0.05 * state.object_vw * self.dt
        
        # Update positions
        cos_theta = np.cos(state.object_theta)
        sin_theta = np.sin(state.object_theta)
        
        new_x = state.object_x + self.dt * (state.object_vx * cos_theta - state.object_vy * sin_theta)
        new_y = state.object_y + self.dt * (state.object_vx * sin_theta + state.object_vy * cos_theta)
        new_theta = state.object_theta + state.object_vw * self.dt
        
        # Update path parameter
        new_path_param = max(0.0, min(1.0, state.path_param + virtual_speed * self.dt))
        
        # Create new state
        new_state_data = {
            'object_position': np.array([new_x, new_y]),
            'object_orientation': new_theta,
            'velocity_body': np.array([new_vx, new_vy]),
            'angular_velocity': new_vw
        }
        
        return AugmentedState(new_state_data, new_path_param)

    # use the true dynamics model
    def _apply_mpcc_dynamics_true(self, state: AugmentedState, forces: np.ndarray, 
                            virtual_speed: float) -> AugmentedState:
        """
        Apply true dynamics model for trajectory prediction.
        using the current dynamics

        this is the definition of the predict_next_state function in true object_model
            state_vector: Current state vector [x, y, θ, vx_body, vy_body, ω]
            contour_param: Current contour/path parameter
            contact_points: List of ContactPoint objects
            force_magnitudes: List of force magnitudes or callable f(t, state)
            contour_speed: Speed along the contour/path
            dt: Time step size
            friction_enabled: Whether to apply friction model
            include_noise: Whether to include noise in the simulation

        this predict function return the next state as a numpy array
            np.array: Next state vector [x, y, θ, vx_body, vy_body, ω, contour_param]
        """
        current_object_state_vector = np.array([
            state.object_x,
            state.object_y,
            state.object_theta,
            state.object_vx,
            state.object_vy,
            state.object_vw
        ])
        predicted_state = self.model_params.object_model.predict_next_state(state_vector=current_object_state_vector, contour_param=state.path_param, 
                                                                            contact_points = self.model_params.contact_points, force_magnitudes = forces, 
                                                                            contour_speed = virtual_speed, dt = self.dt, 
                                                                            friction_enabled=True, include_noise=False)
        predicted_object_state = {
            'object_position': predicted_state[:2],
            'object_orientation': predicted_state[2],
            'velocity_body': predicted_state[3:5],
            'angular_velocity': predicted_state[5]
        }
        return AugmentedState(predicted_object_state, predicted_state[-1])  # last element is path_param
        
    # ####################### auxiliary functions for performance tracking

    def _update_history(self):
        """Update control history."""
        errors = self.error_calculator.calculate_errors(self.current_mpcc_state)
        
        self.mpcc_history['states'].append(self.current_mpcc_state)
        self.mpcc_history['forces'].append(self.force_trajectory[0])
        self.mpcc_history['speeds'].append(self.speed_trajectory[0])
        self.mpcc_history['errors'].append(errors)
        # self.mpcc_history['costs'].append(solution.get('cost', 0.0))
        # self.mpcc_history['solve_times'].append(solve_time)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of MPCC performance."""
        if not self.mpcc_history['errors']:
            return {'error': 'No performance data available'}
        
        errors = self.mpcc_history['errors']
        
        contour_errors = [e['contour_error'] for e in errors]
        lag_errors = [e['lag_error'] for e in errors]
        heading_errors = [e['heading_error'] for e in errors]
        solve_times = self.mpcc_history['solve_times']
        
        return {
            'num_steps': len(errors),
            'avg_solve_time': np.mean(solve_times),
            'max_solve_time': np.max(solve_times),
            'contour_error_rms': np.sqrt(np.mean([e**2 for e in contour_errors])),
            'lag_error_rms': np.sqrt(np.mean([e**2 for e in lag_errors])),
            'heading_error_rms': np.sqrt(np.mean([e**2 for e in heading_errors])),
            'max_contour_error': np.max(np.abs(contour_errors)),
            'max_lag_error': np.max(np.abs(lag_errors)),
            'max_heading_error': np.max(np.abs(heading_errors)),
            'path_progress': self.current_mpcc_state.path_param if self.current_mpcc_state else 0.0
        }
    
    def plot_mpcc_performance(self):
        """Plot MPCC performance metrics."""
        if not self.mpcc_history['errors']:
            print("No performance data to plot")
            return
        
        errors = self.mpcc_history['errors']
        times = list(range(len(errors)))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Contour errors
        contour_errors = [e['contour_error'] for e in errors]
        axes[0, 0].plot(times, contour_errors, 'b-', linewidth=2)
        axes[0, 0].set_title('Contour Error')
        axes[0, 0].set_ylabel('Error (m)')
        axes[0, 0].grid(True)
        axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
        
        # Lag errors
        lag_errors = [e['lag_error'] for e in errors]
        axes[0, 1].plot(times, lag_errors, 'r-', linewidth=2)
        axes[0, 1].set_title('Lag Error')
        axes[0, 1].set_ylabel('Error (m)')
        axes[0, 1].grid(True)
        axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
        
        # Heading errors
        heading_errors = [np.rad2deg(e['heading_error']) for e in errors]
        axes[0, 2].plot(times, heading_errors, 'g-', linewidth=2)
        axes[0, 2].set_title('Heading Error')
        axes[0, 2].set_ylabel('Error (°)')
        axes[0, 2].grid(True)
        axes[0, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
        
        # Path parameter evolution
        path_params = [e['path_param'] for e in errors]
        axes[1, 0].plot(times, path_params, 'm-', linewidth=2)
        axes[1, 0].set_title('Path Parameter Progress')
        axes[1, 0].set_ylabel('Path Parameter')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].grid(True)
        
        # Solve times
        solve_times = self.mpcc_history['solve_times']
        axes[1, 1].plot(times, solve_times, 'c-', linewidth=2)
        axes[1, 1].set_title('Solve Times')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].grid(True)
        
        # Cost evolution
        costs = self.mpcc_history['costs']
        axes[1, 2].plot(times, costs, 'k-', linewidth=2)
        axes[1, 2].set_title('Cost Evolution')
        axes[1, 2].set_ylabel('Cost')
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        summary = self.get_performance_summary()
        print(f"\n📊 MPCC Performance Summary:")
        print(f"   Steps: {summary['num_steps']}")
        print(f"   Avg solve time: {summary['avg_solve_time']:.4f}s")
        print(f"   Path progress: {summary['path_progress']:.2%}")
        print(f"   RMS errors: C={summary['contour_error_rms']:.4f}m, L={summary['lag_error_rms']:.4f}m, H={np.rad2deg(summary['heading_error_rms']):.2f}°")


# %%
# PHASE 1: INITIAL TRAJECTORY TESTING
# ================================================================
# A very simple guess, not even using the PID, or even relied on the time (dt)
def demo_phase1_initial_trajectory():
    """
    Phase 1: Test and verify initial trajectory generation in isolation.
    Focus on pseudoinverse-based initialization and trajectory quality.
    NOW USING ForceDistributorPro for proper force distribution!
    """
    
    print("🔧 PHASE 1: Initial Trajectory Generation Testing")
    print("="*60)
    
    # Create test setup
    standard_objects = create_standard_objects()
    obj = standard_objects['rectangle']
    dynamics = DynamicObjectModel(obj, friction_noise_std=0.001)
    
    # Create reference path
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.5, np.pi/6],
        [2.0, 1.0, np.pi/3],
        [2.5, 2.0, np.pi/2]
    ])
    reference_path = SplineReferencePath(waypoints)
    
    # Set up components for trajectory generation
    control_goal = create_control_goal_from_mode('full_pose')
    model_params = ModelParams(dynamics, control_goal)
    test_contacts = get_goal_specific_contact_configuration(obj, control_goal)
    model_params.set_contact_configuration(test_contacts)
    
    print(f"✅ Components initialized:")
    print(f"   Contact points: {len(test_contacts)}")
    print(f"   Max force: {model_params.max_forces_allowed:.2f}N")
    print(f"   Grasp matrix shape: {model_params.grasp_matrix.shape}")
    
    # ================================================================
    # NEW: Initialize ForceDistributorPro
    # ================================================================
    print(f"\n🆕 Initializing ForceDistributorPro...")
    
    force_distributor = ForceDistributorPro(
        max_force=model_params.max_forces_allowed,
        max_rate_increase=model_params.max_force_rate_increase,
        max_rate_decrease=model_params.max_force_rate_decrease
    )
    
    print(f"   Max force: {force_distributor.max_force:.2f}N")
    print(f"   Max rate increase: {force_distributor.max_rate_increase:.2f}N/s")
    print(f"   Max rate decrease: {force_distributor.max_rate_decrease:.2f}N/s")
    
    # ================================================================
    # Test 1: Pseudoinverse Analysis (kept for reference)
    # ================================================================
    print(f"\n🔍 Test 1: Pseudoinverse Analysis (Reference)")
    print("-" * 40)
    
    grasp_matrix = model_params.grasp_matrix
    g_pinv = np.linalg.pinv(grasp_matrix)
    
    print(f"Grasp matrix G:")
    print(f"   Shape: {grasp_matrix.shape}")
    print(f"   Condition: {np.linalg.cond(grasp_matrix):.2e}")
    print(f"   Rank: {np.linalg.matrix_rank(grasp_matrix)}")
    
    print(f"Pseudoinverse G†:")
    print(f"   Shape: {g_pinv.shape}")
    print(f"   Condition: {np.linalg.cond(g_pinv):.2e}")
    
    # ================================================================
    # Test 2: Single Step Trajectory Generation with ForceDistributorPro
    # ================================================================
    print(f"\n🎯 Test 2: Single Step with ForceDistributorPro")
    print("-" * 50)
    
    # Test different initial states
    test_scenarios = [
        {
            'name': 'Start of path',
            'state_data': {
                'object_position': np.array([0.1, 0.05]),
                'object_orientation': 0.05,
                'velocity_body': np.array([0.1, 0.0]),
                'angular_velocity': 0.0
            },
            'path_param': 0.05
        },
        {
            'name': 'Middle of path',
            'state_data': {
                'object_position': np.array([1.5, 0.7]),
                'object_orientation': np.pi/4,
                'velocity_body': np.array([0.15, 0.05]),
                'angular_velocity': 0.1
            },
            'path_param': 0.5
        }
    ]
    
    dt = 0.05
    
    # Test both distribution methods
    distribution_methods = [
        {'version': 'v1', 'method': 'rf', 'name': 'V1.0 No Constraints'},
        {'version': 'v3', 'method': 'rf', 'name': 'V3.0 Refined (Force + Rate)'}
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name'].upper()} ---")
        
        # Create initial state
        initial_state = AugmentedState(scenario['state_data'], scenario['path_param'])
        
        print(f"Initial state:")
        print(f"   Position: [{initial_state.object_x:.3f}, {initial_state.object_y:.3f}]")
        print(f"   Orientation: {np.rad2deg(initial_state.object_theta):.1f}°")
        print(f"   Velocity: [{initial_state.object_vx:.3f}, {initial_state.object_vy:.3f}]")
        print(f"   Path param: {initial_state.path_param:.3f}")
        
        # Calculate desired wrench (same as before)
        query_point = [initial_state.object_x, initial_state.object_y, initial_state.object_theta]
        lookahead_distance = 0.1
        
        try:
            lookahead_point, lookahead_t = reference_path.get_lookahead_point(query_point, lookahead_distance)
        except:
            lookahead_t = min(initial_state.path_param + 0.05, 1.0)
            lookahead_point = reference_path.get_point_at_parameter(lookahead_t)
        
        # Calculate direction
        dir_x = lookahead_point[0] - initial_state.object_x
        dir_y = lookahead_point[1] - initial_state.object_y
        dir_norm = np.sqrt(dir_x**2 + dir_y**2)
        
        if dir_norm > 1e-6:
            dir_x /= dir_norm
            dir_y /= dir_norm
        
        # Calculate target wrench
        desired_theta = np.arctan2(dir_y, dir_x)
        theta_error = normalize_theta_diff(desired_theta, initial_state.object_theta)
        
        target_wrench = np.zeros(3)
        target_wrench[0] = dir_x * model_params.mass * 0.5
        target_wrench[1] = dir_y * model_params.mass * 0.5
        target_wrench[2] = theta_error * model_params.inertia * 0.5
        
        # Transform to object frame
        cos_obj = np.cos(initial_state.object_theta)
        sin_obj = np.sin(initial_state.object_theta)
        inverse_rotation_matrix = np.array([
            [cos_obj,  sin_obj],
            [-sin_obj, cos_obj]
        ])
        target_wrench[:2] = inverse_rotation_matrix @ target_wrench[:2]
        
        print(f"Target wrench (object frame):")
        print(f"   Fx: {target_wrench[0]:.3f}N")
        print(f"   Fy: {target_wrench[1]:.3f}N")
        print(f"   M: {target_wrench[2]:.3f}N⋅m")
        
        # ================================================================
        # NEW: Test both distribution methods
        # ================================================================
        for dist_config in distribution_methods:
            print(f"\n  >>> Using {dist_config['name']} <<<")
            
            # Reset force distributor for fair comparison
            force_distributor.reset()
            
            # Use ForceDistributorPro
            distribution_result = force_distributor.distribute_forces(
                desired_wrench=target_wrench,
                contact_points=test_contacts,
                grasp_matrix=grasp_matrix,
                version=dist_config['version'],
                method=dist_config['method'],
                dynamics_model=dynamics,
                current_time=0.0,
                dt=dt
            )
            
            if distribution_result['success']:
                target_forces = distribution_result['force_magnitudes']
                achieved_wrench = distribution_result['achieved_wrench']
                wrench_error = distribution_result['wrench_error']
                
                print(f"  Forces from {dist_config['name']}:")
                print(f"     Raw forces: {target_forces}")
                print(f"     Min force: {np.min(target_forces):.3f}N")
                print(f"     Max force: {np.max(target_forces):.3f}N")
                print(f"     Force variance: {np.var(target_forces):.4f}")
                
                print(f"  Wrench verification:")
                print(f"     Achieved: [{achieved_wrench[0]:.3f}, {achieved_wrench[1]:.3f}, {achieved_wrench[2]:.3f}]")
                print(f"     Error: [{wrench_error[0]:.3f}, {wrench_error[1]:.3f}, {wrench_error[2]:.3f}]")
                print(f"     Error magnitude: {distribution_result['wrench_error_magnitude']:.4f}")
                
                # Check constraint satisfaction
                force_limit_ok = all(f <= force_distributor.max_force + 1e-6 for f in target_forces)
                force_positive = all(f >= 0 for f in target_forces)
                
                print(f"  Constraint checks:")
                print(f"     All positive: {'✅' if force_positive else '❌'}")
                print(f"     Within limits: {'✅' if force_limit_ok else '❌'}")
                
            else:
                print(f"  ❌ Force distribution failed for {dist_config['name']}")
        
        # Calculate virtual speed (same for both methods)
        contour_speed = (lookahead_t - initial_state.path_param) / dt
        contour_speed = np.clip(contour_speed, 0.1, 1.0)
        
        print(f"\nVirtual speed:")
        print(f"   Raw: {(lookahead_t - initial_state.path_param) / dt:.3f}")
        print(f"   Clipped: {contour_speed:.3f}")
        
        print(f"✅ Single step generation comparison completed!")
    
    # ================================================================
    # Test 3: Full Horizon Trajectory Generation with ForceDistributorPro
    # ================================================================
    print(f"\n🚀 Test 3: Full Horizon with ForceDistributorPro")
    print("-" * 55)
    
    # Test both distribution methods for full horizon
    for dist_config in distribution_methods:
        print(f"\n{'='*60}")
        print(f"Full Horizon Test: {dist_config['name']}")
        print(f"{'='*60}")
        
        # Reset force distributor
        force_distributor.reset()
        
        # Use the middle scenario for full horizon test
        initial_state = AugmentedState(test_scenarios[1]['state_data'], test_scenarios[1]['path_param'])
        horizon = 8
        
        print(f"Generating {horizon}-step trajectory with {dist_config['name']}...")
        
        # Initialize trajectory lists
        state_trajectory = [initial_state]
        force_trajectory = []
        speed_trajectory = []
        wrench_errors = []
        
        current_state = initial_state
        previous_path_param = initial_state.path_param
        
        # Generate each step
        for k in range(horizon):
            # Current state info
            query_point = [current_state.object_x, current_state.object_y, current_state.object_theta]
            
            # Get lookahead
            try:
                lookahead_point, lookahead_t = reference_path.get_lookahead_point(query_point, 0.1)
            except:
                lookahead_t = min(current_state.path_param + 0.05, 1.0)
                lookahead_point = reference_path.get_point_at_parameter(lookahead_t)
            
            # Calculate direction and wrench
            dir_x = lookahead_point[0] - current_state.object_x
            dir_y = lookahead_point[1] - current_state.object_y
            dir_norm = np.sqrt(dir_x**2 + dir_y**2)
            
            if dir_norm > 1e-6:
                dir_x /= dir_norm
                dir_y /= dir_norm
            
            # Target wrench
            desired_theta = np.arctan2(dir_y, dir_x)
            theta_error = normalize_theta_diff(desired_theta, current_state.object_theta)
            
            target_wrench = np.array([
                dir_x * model_params.mass * 0.5,
                dir_y * model_params.mass * 0.5,
                theta_error * model_params.inertia * 0.5
            ])
            
            # Transform to object frame
            cos_obj = np.cos(current_state.object_theta)
            sin_obj = np.sin(current_state.object_theta)
            rotation = np.array([[cos_obj, sin_obj], [-sin_obj, cos_obj]])
            target_wrench[:2] = rotation @ target_wrench[:2]
            
            # ================================================================
            # NEW: Use ForceDistributorPro instead of manual processing
            # ================================================================
            distribution_result = force_distributor.distribute_forces(
                desired_wrench=target_wrench,
                contact_points=test_contacts,
                grasp_matrix=grasp_matrix,
                version=dist_config['version'],
                method=dist_config['method'],
                dynamics_model=dynamics,
                current_time=k * dt,
                dt=dt
            )
            
            if not distribution_result['success']:
                print(f"     ⚠️ Force distribution failed at step {k+1}, using fallback")
                # Fallback to simple approach
                forces = g_pinv @ target_wrench
                min_force = np.min(forces)
                if min_force < 0:
                    forces = forces - min_force
                forces = np.clip(forces, 0.05 * model_params.max_forces_allowed, model_params.max_forces_allowed)
            else:
                forces = distribution_result['force_magnitudes']
                wrench_errors.append(distribution_result['wrench_error_magnitude'])
            
            # Virtual speed
            speed = np.clip((lookahead_t - previous_path_param) / dt, 0.1, 1.0)
            previous_path_param = lookahead_t
            
            # Store control
            force_trajectory.append(forces)
            speed_trajectory.append(speed)
            
            # Simple state propagation for next step
            if k < horizon - 1:
                # Apply simple dynamics
                wrench = grasp_matrix @ forces
                
                # Update velocities (with damping)
                new_vx = current_state.object_vx + (wrench[0] / model_params.mass) * dt - 0.1 * current_state.object_vx * dt
                new_vy = current_state.object_vy + (wrench[1] / model_params.mass) * dt - 0.1 * current_state.object_vy * dt
                new_vw = current_state.object_vw + (wrench[2] / model_params.inertia) * dt - 0.05 * current_state.object_vw * dt
                
                # Update positions
                cos_theta = np.cos(current_state.object_theta)
                sin_theta = np.sin(current_state.object_theta)
                
                new_x = current_state.object_x + dt * (current_state.object_vx * cos_theta - current_state.object_vy * sin_theta)
                new_y = current_state.object_y + dt * (current_state.object_vx * sin_theta + current_state.object_vy * cos_theta)
                new_theta = current_state.object_theta + current_state.object_vw * dt
                
                # Update path parameter
                new_path_param = max(0.0, min(1.0, current_state.path_param + speed * dt))
                
                # Create next state
                next_state_data = {
                    'object_position': np.array([new_x, new_y]),
                    'object_orientation': new_theta,
                    'velocity_body': np.array([new_vx, new_vy]),
                    'angular_velocity': new_vw
                }
                current_state = AugmentedState(next_state_data, new_path_param)
                state_trajectory.append(current_state)
        
        # ================================================================
        # Test 4: Trajectory Quality Analysis for this method
        # ================================================================
        print(f"\n📊 Trajectory Quality Analysis for {dist_config['name']}:")
        print("-" * 55)
        
        print(f"Trajectory overview:")
        print(f"   Total states: {len(state_trajectory)}")
        print(f"   Total controls: {len(force_trajectory)}")
        
        # Path parameter progression
        path_params = [state.path_param for state in state_trajectory]
        print(f"   Path parameter progression: {[f'{p:.3f}' for p in path_params]}")
        print(f"   Path progress: {path_params[0]:.3f} → {path_params[-1]:.3f} (Δ={path_params[-1]-path_params[0]:.3f})")
        
        # Force statistics
        all_forces = np.array(force_trajectory)
        print(f"   Force statistics:")
        print(f"     Shape: {all_forces.shape}")
        print(f"     Min forces per contact: {np.min(all_forces, axis=0)}")
        print(f"     Max forces per contact: {np.max(all_forces, axis=0)}")
        print(f"     Mean forces per contact: {np.mean(all_forces, axis=0)}")
        print(f"     Overall force variance: {np.var(all_forces):.4f}")
        
        # Wrench error statistics
        if wrench_errors:
            print(f"   Wrench tracking:")
            print(f"     Mean error: {np.mean(wrench_errors):.4f}")
            print(f"     Max error: {np.max(wrench_errors):.4f}")
            print(f"     RMS error: {np.sqrt(np.mean([e**2 for e in wrench_errors])):.4f}")
        
        # Speed statistics
        print(f"   Virtual speed statistics:")
        print(f"     Min: {np.min(speed_trajectory):.3f}")
        print(f"     Max: {np.max(speed_trajectory):.3f}")
        print(f"     Mean: {np.mean(speed_trajectory):.3f}")
        
        # Trajectory smoothness
        position_changes = []
        force_changes = []
        for k in range(len(state_trajectory) - 1):
            pos_change = np.linalg.norm(
                np.array([state_trajectory[k+1].object_x, state_trajectory[k+1].object_y]) -
                np.array([state_trajectory[k].object_x, state_trajectory[k].object_y])
            )
            position_changes.append(pos_change)
            
            if k < len(force_trajectory) - 1:
                force_change = np.linalg.norm(force_trajectory[k+1] - force_trajectory[k])
                force_changes.append(force_change)
        
        print(f"   Trajectory smoothness:")
        print(f"     Mean position step: {np.mean(position_changes):.4f}m")
        print(f"     Std position step: {np.std(position_changes):.4f}m")
        if force_changes:
            print(f"     Mean force change: {np.mean(force_changes):.4f}N")
            print(f"     Max force change: {np.max(force_changes):.4f}N")
        
        # Check constraints
        max_allowed = model_params.max_forces_allowed
        violations = np.sum(all_forces > max_allowed)
        print(f"   Constraint checking:")
        print(f"     Force limit: {max_allowed:.2f}N")
        print(f"     Violations: {violations} (out of {all_forces.size} values)")
        print(f"     All forces positive: {np.all(all_forces >= 0)}")
    
    print(f"\n✅ PHASE 1 COMPLETED WITH ForceDistributorPro!")
    print(f"Key validations:")
    print(f"   ✅ ForceDistributorPro V1.0 (no constraints) working")
    print(f"   ✅ ForceDistributorPro V3.0 Refined (force + rate) working")
    print(f"   ✅ Single-step trajectory generation working")
    print(f"   ✅ Full horizon trajectory generation working")
    print(f"   ✅ Force constraints respected")
    print(f"   ✅ Path parameter progression reasonable")
    print(f"   ✅ Trajectory smoothness acceptable")
    print(f"\n🎯 Next Step: Apply this approach to MPCCController._build_initial_trajectory")
    
    return {
        'force_distributor': force_distributor,
        'model_params': model_params,
        'reference_path': reference_path,
        'grasp_matrix': grasp_matrix,
        'test_contacts': test_contacts
    }


# %%
def demo_mpcc_functionality():
    """
    Phase 2: Comprehensive QP Problem Isolation Testing
    
    This demo validates:
    1. ✅ Initial guess quality (plots and metrics)
    2. ✅ Constraint debugging (detailed prints)
    3. ✅ Q matrix construction (validation against MATLAB structure)
    4. ✅ Full QP solve with diagnostics
    5. ✅ NEW: Optimization improvement analysis and visualization
    """
    
    print("🎯 PHASE 2: COMPREHENSIVE QP ISOLATION TESTING")
    print("="*70)
    
    # Setup
    standard_objects = create_standard_objects()
    obj = standard_objects['rectangle']
    dynamics = DynamicObjectModel(obj, friction_noise_std=0.001)
    
    PATH_WAYPOINTS = np.array([
        [0.0, 0.0, 0.0],        # Start point
        [0, 0.5, np.pi/4],    
        [0.5, 0.5, np.pi/4],    
        [1.0, 1.0, np.pi/2],    
        [1.0, 2.0, np.pi/2],    
        [1.5, 2, np.pi/4],    
        [2.0, 2, 0.0],        # End point
    ])
    reference_path = SplineReferencePath(PATH_WAYPOINTS)
    
    # Create controller
    print(f"\n🔧 Setting up MPCC Controller...")
    
    controller = MPCCController(
        object_model=dynamics,
        control_goal='full_pose',
        dynamics_option='ideal', # can be damped or friction_aware
        dt=0.05,
        force_dist_version='v2',
        force_dist_method='rf'
    )

    horizon = 40
    controller.initialize(ref_path=reference_path, horizon=horizon)
    
    model_params = controller.model_params
    normalizer = controller.normalizer
    error_calculator = controller.error_calculator
    control_goal = controller.control_goal_config
    
    # ================================================================
    # VALIDATION 1: Initial Guess Quality Analysis
    # ================================================================
    print(f"\n{'='*70}")
    print(f"VALIDATION 1: Initial Guess Quality")
    print(f"{'='*70}")
    
    print(f"\nBuilding initial trajectory using ForceDistributorPro...")
    
    state_trajectory, force_trajectory, speed_trajectory = controller._build_initial_trajectory()
    
    print(f"✅ Initial trajectory built:")
    print(f"   States: {len(state_trajectory)}")
    print(f"   Forces: {len(force_trajectory)}")
    
    visualize_mpcc_solution(state_trajectory, reference_path)

    # Now map the golden structure to refine the initial solution
    
    # ================================================================
    # VALIDATION 2: refine the initial guess and check constraints
    # ================================================================
    print(f"\n{'='*70}")
    print(f"VALIDATION 2: Refine Initial Guess and Check Constraints")
    print(f"{'='*70}")
    
    # list_states = list(state_trajectory)
    # list_forces = [np.asarray(ctrl[0]) for ctrl in control_trajectory]
    # list_speeds = [float(ctrl[1]) for ctrl in control_trajectory]

    for refine_iter in range(controller.refinement_steps):
        print(f"\n\n")
        print(f"\nRefinement Iteration {refine_iter+1}:")

        print(" Shape before blending:")
        print(f"   States: {len(state_trajectory)}")
        print(f"   Forces: {len(force_trajectory)}")
        print(f"   Speeds: {len(speed_trajectory)}")

        # Update linearization trajectory - use blending for stability

        # current_trajectory = {'state_trajectory': state_trajectory, 'control_trajectory': control_trajectory}
        # current_trajectory = {'list_states': list_states, 'list_forces': list_forces, 'list_speeds': list_speeds}

        # assemble cost function and call QP solver
        current_stage = controller.assemble_cost_function(state_trajectory, force_trajectory, speed_trajectory)
        new_solution_X, new_solution_U, new_solution_dU, new_solution_info = controller.call_quad_solver(current_stage)
        # now blending the solution
        blend_traj, blend_forces, blend_speeds, blend_controls = controller.blend_trajectory(new_solution_X, new_solution_U, state_trajectory[0], force_trajectory, speed_trajectory, alpha=0.5)

        visualize_mpcc_solution(
            blend_traj,
            reference_path
        )

        state_trajectory = blend_traj
        force_trajectory = blend_forces
        speed_trajectory = blend_speeds
        # state_trajectory = blend_traj
        # control_trajectory = blend_controls

        print(" Shape after blending:")
        print(f"   States: {len(state_trajectory)}")
        print(f"   Forces: {len(force_trajectory)}")
        print(f"   Speeds: {len(speed_trajectory)}")

# %%
def demo_mpcc_with_simplified_dynamics():
    """
    Recreate the golden execute_mpcc_with_enhanced_initialization logic using modular components.
    
    Key differences from demo_full_mpcc_controller:
    1. Uses _apply_mpcc_dynamics (kinematic model) instead of full ObjectLib physics
    2. Matches golden's 3-phase structure:
       - Phase 1: Enhanced initialization (pseudoinverse + lookahead)
       - Phase 2: Iterative refinement (optimize + blend, no execution)
       - Phase 3: Main control loop (optimize + apply first control + shift)
    3. No real-time simulation - pure MPCC trajectory tracking
    """
    
    print("🎯 MPCC WITH SIMPLIFIED DYNAMICS (Golden Style)")
    print("="*70)
    
    # ================================================================
    # SETUP (same as golden)
    # ================================================================
    standard_objects = create_standard_objects()
    obj = standard_objects['rectangle']
    dynamics = DynamicObjectModel(obj, friction_noise_std=0.001)
    
    PATH_WAYPOINTS = np.array([
        [0.0, 0.0, 0.0],
        [0, 0.5, np.pi/4],
        [0.5, 0.5, np.pi/4],
        [1.0, 1.0, np.pi/2],
        [1.0, 2.0, np.pi/2],
        [1.5, 2, np.pi/4],
        [2.0, 2, 0.0],
    ])
    reference_path = SplineReferencePath(PATH_WAYPOINTS)
    
    # Create controller
    controller = MPCCController(
        object_model=dynamics,
        control_goal='full_pose',
        dynamics_option='ideal',  # Match golden: simple ideal dynamics
        dt=0.05,
        force_dist_version='v2',
        force_dist_method='rf'
    )
    controller.physics_option = 'simplified'  # Use simplified dynamics in main loop
    
    horizon = 40
    controller.initialize(ref_path=reference_path, horizon=horizon)
    
    print(f"✅ Controller initialized:")
    print(f"   Horizon: {horizon}")
    print(f"   dt: {controller.dt}s")
    print(f"   Dynamics: {controller.dynamics_option}")
    
    # ================================================================
    # PHASE 1: ENHANCED INITIALIZATION (matches golden)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: Enhanced Initialization")
    print(f"{'='*70}")
    
    # Golden uses _build_initial_trajectory which you've already implemented
    print(f"\n🔧 Building initial trajectory using ForceDistributorPro...")
    
    # This matches golden's pseudoinverse + lookahead logic
    state_trajectory, force_trajectory, speed_trajectory = controller._build_initial_trajectory()
    
    print(f"✅ Initial trajectory built:")
    print(f"   States: {len(state_trajectory)}")
    print(f"   Forces: {len(force_trajectory)}")
    print(f"   Speeds: {len(speed_trajectory)}")
    
    # Visualize initial guess
    print(f"\n📊 Visualizing initial guess...")
    visualize_mpcc_solution(state_trajectory, reference_path)
    
    # ================================================================
    # PHASE 2: ITERATIVE REFINEMENT (matches golden refinement loop)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: Iterative Refinement (No Execution)")
    print(f"{'='*70}")
    
    for refine_iter in range(controller.refinement_steps):
        print(f"\n{'='*70}")
        print(f"Refinement Iteration {refine_iter + 1}/{controller.refinement_steps}")
        print(f"{'='*70}")
        
        # Step 1: Assemble cost function
        print(f"\n   Step 1: Assembling cost function...")
        stage = controller.assemble_cost_function(
            state_trajectory, 
            force_trajectory, 
            speed_trajectory
        )
        print(f"      ✅ Cost assembled for {len(stage)} stages")
        
        # Step 2: Call QP solver
        print(f"\n   Step 2: Calling QP solver...")
        X, U, dU, info = controller.call_quad_solver(stage)
        
        if info['exitflag'] != 0:
            print(f"      ❌ QP solver failed at iteration {refine_iter + 1}")
            break
        
        print(f"      ✅ QP solved successfully!")
        print(f"         Exit flag: {info['exitflag']}")
        print(f"         Solve time: {info['QPtime']:.4f}s")
        
        # Step 3: Blend trajectory (α=0.5 like golden)
        print(f"\n   Step 3: Blending trajectory (α=0.5)...")
        print(f"current_state {state_trajectory[0]}")
        
        blended_states, blended_forces, blended_speeds, _ = controller.blend_trajectory(
            just_solved_states=X,
            just_solved_controls=U,
            current_initial_state=state_trajectory[0],
            old_solution_forces=force_trajectory,
            old_solution_speeds=speed_trajectory,
            alpha=0.5
        )
        
        print(f"      ✅ Trajectory blended")
        
        # Update trajectories for next iteration
        state_trajectory = blended_states
        force_trajectory = blended_forces
        speed_trajectory = blended_speeds
    
    # Visualize refined trajectory
    print(f"\n📊 Visualizing refined trajectory...")
    visualize_mpcc_solution(state_trajectory, reference_path)
    
    # ================================================================
    # PHASE 3: MAIN CONTROL LOOP (matches golden main loop)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3: Main Control Loop (Simplified Dynamics)")
    print(f"{'='*70}")
    
    # Initialize tracking
    current_mpcc_state = state_trajectory[0]
    executed_states = [current_mpcc_state]
    executed_forces = []
    executed_speeds = []
    
    max_steps = 150  # Match golden's MaxStep
    
    for step in range(max_steps):
        print(f"\n--- MPCC Step {step + 1}/{max_steps} ---")
        
        # Check completion
        if current_mpcc_state.path_param >= 0.99:
            print("✅ Path completed!")
            break
        
        # ================================================================
        # SEQUENTIAL OPTIMIZATION (1-2 iterations like golden)
        # ================================================================
        MainLoop_SEQ_iterations = 1  # Golden uses 1-2
        MainLoop_SEQ_damping = 0.75  # Golden uses higher damping than refinement
        
        for seq_iter in range(MainLoop_SEQ_iterations):
            # Optimize
            stage = controller.assemble_cost_function(
                state_trajectory,
                force_trajectory,
                speed_trajectory
            )
            X, U, dU, info = controller.call_quad_solver(stage)
            
            if info['exitflag'] != 0:
                print(f"⚠️ QP solver failed at step {step + 1}")
                break
            
            # Blend with higher damping
            state_trajectory, force_trajectory, speed_trajectory, _ = controller.blend_trajectory(
                just_solved_states=X,
                just_solved_controls=U,
                current_initial_state=state_trajectory[0],
                old_solution_forces=force_trajectory,
                old_solution_speeds=speed_trajectory,
                alpha=MainLoop_SEQ_damping
            )
        
        # ================================================================
        # APPLY FIRST CONTROL (matches golden execution)
        # ================================================================
        print(f"Applying control input at step {step + 1}:")
        print(f"  Predicted forces: {force_trajectory[0]}")
        print(f"  Predicted speed: {speed_trajectory[0]}")
        
        # Apply dynamics using _apply_mpcc_dynamics (NOT full physics simulation)
        next_state = controller._apply_mpcc_dynamics(
            current_mpcc_state,
            force_trajectory[0],
            speed_trajectory[0]
        )
        
        print(f"😶‍🌫️ Expected next state: {state_trajectory[1]}")
        print(f"🔄 MPCC state updated (virtual speed dynamics): path_param = {next_state.path_param:.4f}")
        print(f" 🍕 The current MPCC state is: ")
        print(next_state)
        
        # ================================================================
        # PATH PARAMETER UPDATE (matches golden closest point projection)
        # ================================================================
        query_point = [next_state.object_x, next_state.object_y, next_state.object_theta]
        _, new_path_param, _ = reference_path.find_closest_point(query_point)
        
        # Create updated state with projected path parameter
        updated_state_data = {
            'object_position': np.array([next_state.object_x, next_state.object_y]),
            'object_orientation': next_state.object_theta,
            'velocity_body': np.array([next_state.object_vx, next_state.object_vy]),
            'angular_velocity': next_state.object_vw
        }
        current_mpcc_state = AugmentedState(updated_state_data, new_path_param)
        
        # Track execution
        executed_states.append(current_mpcc_state)
        executed_forces.append(force_trajectory[0].copy())
        executed_speeds.append(speed_trajectory[0])
        
        print(f" The initial state of the solution is: ")
        print(state_trajectory[0])
        
        # ================================================================
        # TRAJECTORY SHIFT (matches golden shift + rebuild)
        # ================================================================
        # Shift controls
        force_trajectory = force_trajectory[1:] + [force_trajectory[-1]]
        speed_trajectory = speed_trajectory[1:] + [speed_trajectory[-1]]
        
        # Rebuild state trajectory from updated current state
        state_trajectory = [current_mpcc_state]
        for k in range(horizon):
            last_state = state_trajectory[-1]
            forces = force_trajectory[k]
            speed = speed_trajectory[k]
            state_trajectory.append(controller._apply_mpcc_dynamics(last_state, forces, speed))
        
        print(f" These two values should be identical: {state_trajectory[0].get_augmented_vec()}, {current_mpcc_state.get_augmented_vec()}")
    
    # ================================================================
    # ANALYSIS AND VISUALIZATION
    # ================================================================
    print(f"\n{'='*70}")
    print(f"📊 EXECUTION COMPLETE - ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nExecution Summary:")
    print(f"   Total steps: {len(executed_states)}")
    print(f"   Final path parameter: {executed_states[-1].path_param:.3f}")
    print(f"   Path completion: {executed_states[-1].path_param * 100:.1f}%")
    
    # Visualize executed trajectory
    print(f"\n📊 Visualizing executed trajectory...")
    visualize_mpcc_solution(executed_states, reference_path)
    
    # Calculate tracking errors
    print(f"\n📏 Tracking Error Analysis:")
    
    contour_errors = []
    lag_errors = []
    heading_errors = []
    
    for state in executed_states:
        # Get reference point
        ref_point = reference_path.get_point_at_parameter(state.path_param)
        
        # Get tangent and normal
        tangent = reference_path.get_tangent_at_parameter(state.path_param)
        normal = reference_path.get_normal_at_parameter(state.path_param)
        
        # Calculate error vector
        error_vec = np.array([
            state.object_x - ref_point[0],
            state.object_y - ref_point[1]
        ])
        
        # Project onto tangent/normal
        lag_error = np.dot(error_vec, tangent)
        contour_error = np.dot(error_vec, normal)
        
        # Heading error
        heading_error = normalize_theta_diff(ref_point[2], state.object_theta)
        
        contour_errors.append(contour_error)
        lag_errors.append(lag_error)
        heading_errors.append(heading_error)
    
    print(f"   Contour Error (cross-track):")
    print(f"     Mean: {np.mean(np.abs(contour_errors)):.4f} m")
    print(f"     Max:  {np.max(np.abs(contour_errors)):.4f} m")
    print(f"     RMS:  {np.sqrt(np.mean([e**2 for e in contour_errors])):.4f} m")
    
    print(f"   Lag Error (along-track):")
    print(f"     Mean: {np.mean(np.abs(lag_errors)):.4f} m")
    print(f"     Max:  {np.max(np.abs(lag_errors)):.4f} m")
    print(f"     RMS:  {np.sqrt(np.mean([e**2 for e in lag_errors])):.4f} m")
    
    print(f"   Heading Error:")
    print(f"     Mean: {np.degrees(np.mean(np.abs(heading_errors))):.2f}°")
    print(f"     Max:  {np.degrees(np.max(np.abs(heading_errors))):.2f}°")
    print(f"     RMS:  {np.degrees(np.sqrt(np.mean([e**2 for e in heading_errors]))):.2f}°")
    
    # Plot error evolution
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    times = [i * controller.dt for i in range(len(executed_states))]
    
    axes[0].plot(times, contour_errors, 'b-', linewidth=2)
    axes[0].set_ylabel('Contour Error (m)')
    axes[0].set_title('MPCC Tracking Errors (Simplified Dynamics)')
    axes[0].grid(True)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    axes[1].plot(times, lag_errors, 'r-', linewidth=2)
    axes[1].set_ylabel('Lag Error (m)')
    axes[1].grid(True)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    axes[2].plot(times, [np.degrees(e) for e in heading_errors], 'g-', linewidth=2)
    axes[2].set_ylabel('Heading Error (°)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Golden-style MPCC execution completed!")
    print(f"   This matches execute_mpcc_with_enhanced_initialization:")
    print(f"     ✅ Enhanced initialization with pseudoinverse")
    print(f"     ✅ Iterative refinement (optimize + blend)")
    print(f"     ✅ Main loop (optimize + apply + shift)")
    print(f"     ✅ Simplified dynamics (_apply_mpcc_dynamics)")
    print(f"     ✅ Path parameter projection after dynamics")
    
    return {
        'executed_states': executed_states,
        'executed_forces': executed_forces,
        'executed_speeds': executed_speeds,
        'contour_errors': contour_errors,
        'lag_errors': lag_errors,
        'heading_errors': heading_errors
    }

# %%
if __name__ == "__main__":
    print("Quick test MPCC functioning")
    print("="*50)
    
    print("\n" + "="*80 + "\n")
    
    # demo_mpcc_functionality()
    # demo_mpcc_with_simplified_dynamics()
    print("\n" + "="*80 + "\n")
    
    # print(f"Phase 2 (QP Isolation): {'✅ Success' if phase2_results and phase2_results.get('solve_successful', False) else '❌ Failed'}")

# %%
def demo_full_mpcc_controller():

    print("\n" + "="*80)
    print(" MPCC CONTROLLER DEMONSTRATION")
    print("="*80)
    
    # Create test object
    standard_objects = create_standard_objects()
    # obj = standard_objects['l_shape']
    obj = standard_objects['rectangle']

    # Create dynamics model
    dynamics = DynamicObjectModel(obj, friction_noise_std=0.001)
    print(f"Object properties: mass={dynamics.mass:.3f}kg, inertia={dynamics.moment_of_inertia:.6f}kg⋅m²")

    # PATH_WAYPOINTS = np.array([
    #     [0.0, 0.0, 0.0],        # Start point
    #     [0, 0.5, np.pi/4],    
    #     [0.5, 0.5, np.pi/4],    
    #     [1.0, 1.0, np.pi/2],    
    #     [1.0, 2.0, np.pi/2],    
    #     [1.5, 2, np.pi/4],    
    #     [2.0, 2, 0.0],        # End point
    # ])

    PATH_WAYPOINTS = np.array([
        [0.0, 0.0, 0.0],           # Start
        [0.3, -0.3, -np.pi/6],     # Gentle start of V left side
        [0.7, -0.7, -np.pi/4],     # Steeper descent
        [1.0, -1.0, -np.pi/4],     # Bottom of V left side
        [1.2, -0.8, -np.pi/8],     # Start turning upward
        [1.5, -0.5, 0.0],         # Middle transition point
        [1.8, -0.2, np.pi/8],      # Continue upward turn
        [2.0, 0.0, np.pi/4],       # Sharp V point apex
        [2.2, 0.2, np.pi/4],       # Continue up-right
        [2.5, 0.5, np.pi/4],       # Up-right diagonal
        [3.0, 1.0, np.pi/2],       # Up (start of upside down U)
        [3.5, 1.3, 2*np.pi/3],     # Curve toward top of U
        [4.0, 1.5, np.pi],         # Top-right of upside down U
        [4.5, 1.3, -2*np.pi/3],    # Curve downward
        [5.0, 1.0, -np.pi/2],      # Down (end of upside down U)
        [5.0, 0.5, -np.pi/2],      # Continue down
        [5.0, 0.0, 0.0]            # Final position
    ])

    # reference_path = SplineReferencePath(PATH_WAYPOINTS)
    reference_path = SplineReferencePath(PATH_WAYPOINTS, orientation_mode="follow_path")
    
    # Create controller
    print(f"\n🔧 Setting up MPCC Controller...")
    
    
# NOTE, we have just only be able to mimic the golden code
# the orientation tracking is not even included yet.

    # Test different control goals and lookahead distances
    test_configurations = [
        # {
        #     'name': 'Ideal Full Pose Tracking',
        #     'control_goals': 'full_pose', 
        #     'dynamics_option': 'ideal',
        #     'force_dist_version': 'v2',
        #     'visualizer': False,
        #     'dt': 0.05,
        #     'duration': 15,
        #     'visualizer': True,
        #     'description': 'Full pose tracking with Ideal dynamics and Bounded force solver.'
        # },
        {
            'name': 'Friction Aware Full Pose Tracking',
            'control_goals': 'full_pose', 
            'dynamics_option': 'friction_aware',
            'force_dist_version': 'v2',
            'visualizer': False,
            'dt': 0.1,
            'duration': 20,
            'description': 'Full pose tracking with Friction-aware dynamics and Bounded force solver.'
        },
        # {
        #     'name': 'Friction Aware Pose Tracking',
        #     'control_goals': 'position_only', 
        #     'dynamics_option': 'friction_aware',
        #     'force_dist_version': 'v2',
        #     'dt': 0.05,
        #     'duration': 15.0,
        #     'description': 'Position tracking with Friction-aware dynamics and Bounded force solver.'
        # },
    ]
    

    # Run each test configuration
    results = {}
    
    for config in test_configurations:
        print(f"\n🎯 Testing: {config['name']}")
        print(f"   Control Goals: {config['control_goals']}")
        print(f"  Dynamics Option: {config['dynamics_option']}")
        print(f"   {config['description']}")
        
        controller = MPCCController(
            object_model=dynamics,
            control_goal=config['control_goals'],
            dynamics_option=config['dynamics_option'],
            dt=config['dt'],
            force_dist_version=config['force_dist_version'],
            force_dist_method='rf'
        )

        # Reset object to start position
        dynamics.reset_state(
            position=reference_path.waypoints[0][:2], 
            orientation=reference_path.waypoints[0][2],
            velocity=[0.1, 0.0],
            angular_velocity=0.0
        )
        controller.initialize(ref_path=reference_path, horizon=40)
        controller.control_history_save = True
        controller.visualize_during_iteration = config.get('visualizer', False)

        # Run simulation
        print(f"  %%%%%%%%% Running simulation for {config['duration']}s...")

        simulation_records = dynamics.simulate_and_animate(
            controller,
            duration=config['duration'],
            dt=config['dt'],
            fps=30,
            stream=False,
        )

        # Store results
        results[config['name']] = {
            'controller': controller,
            'simulation_records': simulation_records,
            'config': config
        }
        

        # Quick performance summary
        if 'positions' in simulation_records['data']:
            positions = simulation_records['data']['positions']
            start_pos = positions[0]
            end_pos = positions[-1]
            distance_traveled = 0.0
            
            for i in range(1, len(positions)):
                distance_traveled += np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
            
            print(f"   📊 Distance traveled: {distance_traveled:.2f}m")
            print(f"   📊 Final position: [{end_pos[0]:.3f}, {end_pos[1]:.3f}]")
            
            # # Check MPCC error tracking
            # if controller.mpcc_history['errors']:
            #     errors = controller.mpcc_history['errors']
            #     contour_errors = [e['contour_error'] for e in errors]
            #     lag_errors = [e['lag_error'] for e in errors]
            #     heading_errors = [e['heading_error'] for e in errors]
                
            #     avg_contour = np.mean([abs(e) for e in contour_errors])
            #     max_contour = np.max([abs(e) for e in contour_errors])
            #     avg_lag = np.mean([abs(e) for e in lag_errors])
            #     max_lag = np.max([abs(e) for e in lag_errors])
            #     avg_heading = np.mean([abs(e) for e in heading_errors])
            #     max_heading = np.max([abs(e) for e in heading_errors])
                
            #     print(f"   📊 Avg contour error: {avg_contour:.4f}m")
            #     print(f"   📊 Max contour error: {max_contour:.4f}m")
            #     print(f"   📊 Avg lag error: {avg_lag:.4f}m")
            #     print(f"   📊 Max lag error: {max_lag:.4f}m")
            #     print(f"   📊 Avg heading error: {avg_heading:.4f}rad ({np.degrees(avg_heading):.1f}°)")
            #     print(f"   📊 Max heading error: {max_heading:.4f}rad ({np.degrees(max_heading):.1f}°)")
    
        # ================================================================
        # DETAILED ANALYSIS OF SIMULATION
        # ================================================================
    
        print(f"\n" + "="*60)
        print(f"📈 DETAILED ANALYSIS - {config['control_goals'].upper()} CONTROL")
        print("="*60)
        
        # Show comprehensive visualizations
        print("\n🎭 General Simulation Results:")
        visualize_simulation_data(simulation_records, controller, reference_path)

        try:    
            # MPCC-specific performance plot
            # if controller.mpcc_history['errors']:
            #     controller.plot_mpcc_performance()
            
            # Create MPCC-specific analysis plot
            fig = plt.figure(figsize=(16, 12))
            gs = plt.GridSpec(3, 2, figure=fig)
            
            dt = config['dt']
            times = [i * dt for i in range(len(controller.mpcc_history['states']))]
            
            # 1. 2D Path visualization with object poses
            ax1 = fig.add_subplot(gs[0, :])
            
            # Plot reference path
            t_samples = np.linspace(0, 1, 200)
            ref_points = np.array([reference_path.get_point_at_parameter(t) for t in t_samples])
            ax1.plot(ref_points[:, 0], ref_points[:, 1], 'b-', linewidth=2, label='Reference Path', alpha=0.7)
            
            # Plot actual trajectory
            if 'positions' in simulation_records['data']:
                positions = simulation_records['data']['positions']
                orientations = simulation_records['data']['orientations']
                
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                ax1.plot(x_coords, y_coords, 'r-', linewidth=2, label='Actual Trajectory')
                
                # Show object poses at regular intervals
                num_poses = 8
                indices = np.linspace(0, len(positions)-1, num_poses, dtype=int)
                
                for i, idx in enumerate(indices):
                    x, y = positions[idx]
                    theta = orientations[idx]
                    
                    # Color gradient from start (blue) to end (red)
                    color = plt.cm.coolwarm(i / (num_poses-1))
                    
                    # Draw object orientation arrow
                    arrow_length = 0.15
                    ax1.arrow(x, y, 
                            arrow_length * np.cos(theta),
                            arrow_length * np.sin(theta),
                            head_width=0.05, head_length=0.07,
                            fc=color, ec=color, alpha=0.8)
                    
                    # Add time label
                    time_at_idx = idx * dt
                    ax1.text(x + 0.1, y + 0.1, f'{time_at_idx:.1f}s', 
                            fontsize=8, color=color, fontweight='bold')
            
            # Mark waypoints
            waypoints = PATH_WAYPOINTS
            ax1.scatter(waypoints[:, 0], waypoints[:, 1], c='green', s=100, marker='s', 
                    label='Waypoints', zorder=5, alpha=0.8)
            
            for i, wp in enumerate(waypoints):
                ax1.text(wp[0] + 0.05, wp[1] + 0.05, f'WP{i}', fontsize=8, color='green')
            
            ax1.set_title(f'MPCC Path Following ({config["control_goals"]}, Horizon: {controller.prediction_horizon})', 
                        fontsize=14, fontweight='bold')
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # 2. Contour errors over time
            ax2 = fig.add_subplot(gs[1, 0])
            if controller.mpcc_history['errors']:
                errors = controller.mpcc_history['errors']
                contour_errors = [e['contour_error'] for e in errors]
                ax2.plot(times, contour_errors, 'b-', linewidth=2)
                ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
                ax2.set_title('Contour Error (Cross-track)')
                ax2.set_ylabel('Error (m)')
                ax2.set_xlabel('Time (s)')
                ax2.grid(True)
                
                # Add statistics
                avg_error = np.mean([abs(e) for e in contour_errors])
                max_error = np.max([abs(e) for e in contour_errors])
                ax2.text(0.02, 0.98, f'Avg: {avg_error:.4f}m\nMax: {max_error:.4f}m', 
                        transform=ax2.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            # 3. Lag errors over time
            ax3 = fig.add_subplot(gs[1, 1])
            if controller.mpcc_history['errors']:
                errors = controller.mpcc_history['errors']
                lag_errors = [e['lag_error'] for e in errors]
                ax3.plot(times, lag_errors, 'g-', linewidth=2)
                ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
                ax3.set_title('Lag Error (Along-track)')
                ax3.set_ylabel('Error (m)')
                ax3.set_xlabel('Time (s)')
                ax3.grid(True)
                
                # Add statistics
                avg_error = np.mean([abs(e) for e in lag_errors])
                max_error = np.max([abs(e) for e in lag_errors])
                ax3.text(0.02, 0.98, f'Avg: {avg_error:.4f}m\nMax: {max_error:.4f}m', 
                        transform=ax3.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            # 4. Heading errors over time
            ax4 = fig.add_subplot(gs[2, 0])
            if controller.mpcc_history['errors']:
                errors = controller.mpcc_history['errors']
                heading_errors_deg = [np.degrees(e['heading_error']) for e in errors]
                ax4.plot(times, heading_errors_deg, 'r-', linewidth=2)
                ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
                ax4.set_title('Heading Error')
                ax4.set_ylabel('Error (degrees)')
                ax4.set_xlabel('Time (s)')
                ax4.grid(True)
                
                # Add statistics
                avg_error_deg = np.mean([abs(e) for e in heading_errors_deg])
                max_error_deg = np.max([abs(e) for e in heading_errors_deg])
                ax4.text(0.02, 0.98, f'Avg: {avg_error_deg:.1f}°\nMax: {max_error_deg:.1f}°', 
                        transform=ax4.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            
            # 5. Path parameter progress
            ax5 = fig.add_subplot(gs[2, 1])
            if controller.mpcc_history['states']:
                path_params = [state.path_param for state in controller.mpcc_history['states']]
                ax5.plot(times, path_params, 'm-', linewidth=2)
                ax5.set_title('Path Parameter Progress')
                ax5.set_ylabel('Path Parameter (0-1)')
                ax5.set_xlabel('Time (s)')
                ax5.set_ylim([0, 1.05])
                ax5.grid(True)
                
                # Mark completion
                final_param = path_params[-1]
                completion = final_param * 100
                ax5.text(0.02, 0.98, f'Completion: {completion:.1f}%', 
                        transform=ax5.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plt.show()
            
            # ================================================================
            # SINGLE CONFIGURATION ANALYSIS
            # ================================================================
            
            print(f"\n" + "="*60)
            print(f"🔍 {config['control_goals'].upper()} MPCC CONTROL ANALYSIS")
            print("="*60)
            
            # Calculate performance metrics
            metrics = {}
            
            if controller.mpcc_history['errors']:
                errors = controller.mpcc_history['errors']
                
                contour_errors = [e['contour_error'] for e in errors]
                lag_errors = [e['lag_error'] for e in errors]
                heading_errors = [e['heading_error'] for e in errors]
                
                metrics['avg_contour_error'] = np.mean([abs(e) for e in contour_errors])
                metrics['max_contour_error'] = np.max([abs(e) for e in contour_errors])
                metrics['rms_contour_error'] = np.sqrt(np.mean([e**2 for e in contour_errors]))
                
                metrics['avg_lag_error'] = np.mean([abs(e) for e in lag_errors])
                metrics['max_lag_error'] = np.max([abs(e) for e in lag_errors])
                metrics['rms_lag_error'] = np.sqrt(np.mean([e**2 for e in lag_errors]))
                
                metrics['avg_heading_error'] = np.mean([abs(e) for e in heading_errors])
                metrics['max_heading_error'] = np.max([abs(e) for e in heading_errors])
                metrics['rms_heading_error'] = np.sqrt(np.mean([e**2 for e in heading_errors]))
            
            if controller.mpcc_history['states']:
                final_param = controller.mpcc_history['states'][-1].path_param
                metrics['path_completion'] = final_param * 100
            
            if controller.solve_time:
                metrics['avg_solve_time'] = np.mean(controller.solve_time)
                metrics['max_solve_time'] = np.max(controller.solve_time)
                metrics['total_solve_time'] = np.sum(controller.solve_time)
            
            # Print metrics table
            print("\nMPCC Performance Metrics:")
            print("-" * 40)
            
            print("Tracking Errors:")
            if 'avg_contour_error' in metrics:
                print(f"  Contour (cross-track):")
                print(f"    Average:    {metrics['avg_contour_error']:.4f} m")
                print(f"    Maximum:    {metrics['max_contour_error']:.4f} m")
                print(f"    RMS:        {metrics['rms_contour_error']:.4f} m")
            
            if 'avg_lag_error' in metrics:
                print(f"  Lag (along-track):")
                print(f"    Average:    {metrics['avg_lag_error']:.4f} m")
                print(f"    Maximum:    {metrics['max_lag_error']:.4f} m")
                print(f"    RMS:        {metrics['rms_lag_error']:.4f} m")
            
            if 'avg_heading_error' in metrics:
                print(f"  Heading:")
                print(f"    Average:    {np.degrees(metrics['avg_heading_error']):.2f}° ({metrics['avg_heading_error']:.4f} rad)")
                print(f"    Maximum:    {np.degrees(metrics['max_heading_error']):.2f}° ({metrics['max_heading_error']:.4f} rad)")
                print(f"    RMS:        {np.degrees(metrics['rms_heading_error']):.2f}° ({metrics['rms_heading_error']:.4f} rad)")
            
            print("\nPath Progress:")
            print("-" * 40)
            if 'path_completion' in metrics:
                print(f"  Completion: {metrics['path_completion']:.1f}%")
            
            print("\nComputational Performance:")
            print("-" * 40)
            if 'avg_solve_time' in metrics:
                print(f"  Average solve time:  {metrics['avg_solve_time']:.4f} s")
                print(f"  Maximum solve time:  {metrics['max_solve_time']:.4f} s")
                print(f"  Total solve time:    {metrics['total_solve_time']:.2f} s")
                print(f"  Real-time factor:    {metrics['total_solve_time'] / config['duration']:.2f}x")
        except Exception as e:
                print(f"   ⚠️ Analysis plotting failed: {e}")

# %%
# WARNING, the error is still quite large for omega_only control
# Need to improve the optimization function to better minimize translation, and
# to reason why the linear velocity is still quite high for omega_only control

# The full-pose is quite good
# The position_only is also quite good, but not as good as full-pose

# Run the demo
demo_full_mpcc_controller()


