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


# Import ObjectLib components
from object_utils import (
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

from contact_optimizer_utils import (
    find_on_demand_magnum_cps_v3
)

class ForceDistributorPro:
    """
    Distributes desired wrench to contact point forces using multiple methods.
    
    Version 1.0: No constraints (existing implementation)
    Version 2.0: Force magnitude constraints (LP or QP)
    Version 3.0: Force magnitude + rate constraints (LP or QP)
    """
    
    def __init__(self, max_force=10.0, max_rate_increase=6.0, max_rate_decrease=8.0,
                 contact_points=None, grasp_matrix=None, t_params=None):
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
        
        # Optional internal contact configuration (for shared distributor use)
        # When set, distribute_forces can be called without passing contact_points/grasp_matrix.
        self.contact_points = contact_points
        self.grasp_matrix = grasp_matrix
        self.t_params = t_params  # Optional list of boundary parameters for each contact
        
        # Previous force values for rate limiting
        self.prev_forces = None
        self.prev_time = None
        
        # Performance metrics
        self.wrench_errors = []  # Track how well desired wrench is achieved
    
    def distribute_forces(self, desired_wrench, contact_points=None, grasp_matrix=None, 
                         version='v1', method='rf', dynamics_model=None, 
                         current_time=None, dt=1):
        """
        Main interface for force distribution with flexible version and method selection.
        
        Args:
            desired_wrench: Desired wrench vector [Fx, Fy, M]
            contact_points: List of contact points. If None, uses internal contact_points set at init.
            grasp_matrix: Grasp matrix relating contact forces to wrench. If None, uses internal grasp_matrix.
            version: 'v1' (no constraints), 'v2' (force limits), 'v3' (force + rate limits)
            method: 'lp' (linear programming) or 'qp' (quadratic programming) or 'rf' (refined for v2 and v3)
            dynamics_model: Object dynamics model (required for v2 and v3)
            current_time: Current time for rate limiting
            dt: Time step if current_time is None
            
        Returns:
            dict: Results including force magnitudes and metrics
        """
        # Allow using internally stored contact configuration when not provided
        if contact_points is None:
            contact_points = self.contact_points
        if grasp_matrix is None:
            grasp_matrix = self.grasp_matrix
        
        if contact_points is None or grasp_matrix is None:
            raise ValueError("ForceDistributorPro requires contact_points and grasp_matrix "
                             "either at initialization or at call time.")
        
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
                return self.distribute_forces_v1(desired_wrench, contact_points, grasp_matrix, current_time, dt)
                
        except Exception as e:
            print(f"Error in V1.0 Min Variance QP: {e}")
            # Fallback to original v1 method
            return self.distribute_forces_v1(desired_wrench, contact_points, grasp_matrix, current_time, dt)

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
            # should not be possible, we will print out the value here
            print("  BIG ERROR, printout value for debugging ")
            print(" desired wrench ", desired_wrench)
            print("  contact points", contact_points)
            assert False
            return self._distribute_with_slack_v2(desired_wrench, contact_points, grasp_matrix)
        
        unit_forces = unit_result['unit_forces']  # Before magnitude scaling
        unit_scale = unit_result['scale_factor']   # What V1.0 used for scaling
        
        # Step 2: Calculate constraint-respecting scale factor
        max_scales = []
        for i, unit_force in enumerate(unit_forces):
            if unit_force > 1e-6:  # Avoid division by zero
                max_scale_i = self.max_force / unit_force
                max_scales.append(max_scale_i)
        
        if not max_scales:
            # should not be possible, we will print out the value here
            print("  BIG ERROR, printout value for debugging ")
            print(" desired wrench ", desired_wrench)
            print("  contact points", contact_points)
            assert False
            return self._distribute_with_slack_v2(desired_wrench, contact_points, grasp_matrix)

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
        """
        Fallback method for V1.0 - matches return format of primary V1.0 methods.
        Uses quadratic minimization when exact solution fails.
        """
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
        achieved_wrench[np.abs(achieved_wrench) < 1e-6] = 0
        
        wrench_error = achieved_wrench - desired_wrench
        wrench_error_magnitude = np.linalg.norm(wrench_error)
        
        # Calculate unit_forces and scale_factor for consistency with primary V1.0 methods
        force_magnitude_total = np.linalg.norm(force_magnitudes)
        if force_magnitude_total > 1e-6:
            unit_forces = force_magnitudes / force_magnitude_total
            scale_factor = force_magnitude_total
        else:
            unit_forces = np.zeros_like(force_magnitudes)
            scale_factor = 0.0
        
        self.wrench_errors.append(wrench_error_magnitude)
        
        return {
            'force_magnitudes': force_magnitudes,
            'unit_forces': unit_forces,
            'scale_factor': scale_factor,
            'achieved_wrench': achieved_wrench,
            'wrench_error': wrench_error,
            'wrench_error_magnitude': wrench_error_magnitude,
            'success': result.success,
            'method': 'v1_fallback_error_minimization'
        }
    
    def _distribute_with_slack_v2(self, desired_wrench, contact_points, grasp_matrix):
        """
        Fallback method for V2.0 - uses QP with proper return format.
        Ensures all expected fields are present.
        """
        # Weight matrix for proper scaling
        # as a fallback function, we only care about the force component
        W = np.diag([1.0, 1.0, 0])
        
        num_contacts = len(contact_points)
        
        # Simple QP fallback
        def objective(forces):
            achieved_wrench = grasp_matrix @ forces
            error = achieved_wrench - desired_wrench
            weighted_error = W @ error
            return np.sum(weighted_error**2)
        
        bounds = [(0, self.max_force) for _ in range(num_contacts)]
        initial_guess = np.ones(num_contacts) * min(0.1, self.max_force * 0.1)
        
        result = opt.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
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
            'success': result.success,
            'method': 'v2_fallback_weighted_error_minimization'
        } 
    
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
    # obj = standard_objects['triangle']
    obj = standard_objects['fat_triangle']
    
    contact_result = find_on_demand_magnum_cps_v3(
            obj,
            desired_goal='position_only',
            verbose=False,
            visualize=False,
            force_magnitude=1.0
        )
    contact_points = contact_result['contacts']
    
    # Create grasp matrix and dynamics model
    grasp_calculator = GraspMatrixCalculator()
    grasp_matrix = grasp_calculator.build_wrench_matrix(contact_points)
    dynamics = DynamicObjectModel(obj)
    
    # Test wrench - make it large enough to test force limits
    # desired_wrench = np.array([20.0, 10.5, 0.8])
    desired_wrench = np.array([0.0, 0.0, 0.0])
    
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
if __name__ == "__main__":
    demo_force_distributor_methods_pro()