"""Contact maintenance solver algorithms."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from contact_maintain.util import wrap_to_pi, rot2d, unit


class ContactMaintainSolverBase(ABC):
    """Base class for contact maintenance solvers.
    
    Parameters
    ----------
    approach_speed : float
        Speed when approaching object without contact.
    maintain_speed : float
        Speed for adjustments while maintaining contact.
    """
    
    def __init__(self, approach_speed=0.1, maintain_speed=0.05):
        self.approach_speed = approach_speed
        self.maintain_speed = maintain_speed
        
        # State tracking
        self.contact_established = False
        self.in_contact = False
    
    @abstractmethod
    def compute_velocity(self, robot_pos, robot_theta, object_pos, **kwargs):
        """Compute velocity command to maintain contact.
        
        Parameters
        ----------
        robot_pos : np.ndarray, shape (2,)
            Robot position.
        robot_theta : float
            Robot orientation.
        object_pos : np.ndarray, shape (2,)
            Object position.
        **kwargs : dict
            Additional solver-specific parameters.
        
        Returns
        -------
        np.ndarray, shape (3,)
            Velocity command (vx, vy, omega) for holonomic robot.
        """
        pass
    
    def reset(self):
        """Reset solver state."""
        self.contact_established = False
        self.in_contact = False


class ForceBasedContactSolver(ContactMaintainSolverBase):
    """Contact maintenance using force feedback.
    
    Uses measured contact force to maintain a target contact force.
    
    Parameters
    ----------
    target_force : float
        Target contact force magnitude to maintain.
    kp_force : float
        Proportional gain for force error.
    ki_force : float
        Integral gain for force error.
    kd_force : float
        Derivative gain for force error.
    force_threshold : float
        Minimum force to consider as contact.
    max_force : float
        Maximum allowed force (triggers backing off).
    approach_speed : float
        Speed when approaching object.
    maintain_speed : float
        Maximum adjustment speed during contact.
    """
    
    def __init__(self, target_force=5.0, kp_force=0.02, ki_force=0.0, kd_force=0.005,
                 force_threshold=0.5, max_force=20.0, approach_speed=0.15, 
                 maintain_speed=0.1):
        super().__init__(approach_speed, maintain_speed)
        
        self.target_force = target_force
        self.kp_force = kp_force
        self.ki_force = ki_force
        self.kd_force = kd_force
        self.force_threshold = force_threshold
        self.max_force = max_force
        
        # PID state
        self.force_error_integral = 0.0
        self.last_force_error = 0.0
        self.last_force = np.zeros(3)
    
    def compute_velocity(self, robot_pos, robot_theta, object_pos, 
                        contact_force=None, dt=0.01, **kwargs):
        """Compute velocity using force feedback.
        
        Parameters
        ----------
        robot_pos : np.ndarray, shape (2,)
            Robot position.
        robot_theta : float
            Robot orientation.
        object_pos : np.ndarray, shape (2,)
            Object position.
        contact_force : np.ndarray, shape (3,)
            Measured contact force.
        dt : float
            Time step for integral/derivative terms.
        
        Returns
        -------
        np.ndarray, shape (3,)
            Velocity command (vx, vy, omega).
        """
        if contact_force is None:
            contact_force = np.zeros(3)
        
        robot_pos = np.array(robot_pos)
        object_pos = np.array(object_pos)[:2] if len(object_pos) > 2 else np.array(object_pos)
        
        # Direction to object
        to_object = object_pos - robot_pos
        distance = np.linalg.norm(to_object)
        direction = unit(to_object) if distance > 0 else np.array([1, 0])
        
        # Force magnitude
        force_magnitude = np.linalg.norm(contact_force[:2])
        self.in_contact = force_magnitude > self.force_threshold
        
        if not self.in_contact:
            # Not in contact - approach object
            if self.contact_established:
                # Try to recover contact
                vel = self.approach_speed * 0.5 * direction
            else:
                vel = self.approach_speed * direction
            
            # Reset PID state when not in contact
            self.force_error_integral = 0.0
            self.last_force_error = 0.0
            
            return np.array([vel[0], vel[1], 0.0])
        
        # In contact
        self.contact_established = True
        
        # Safety: back off if force too high
        if force_magnitude > self.max_force:
            vel = -0.2 * direction
            return np.array([vel[0], vel[1], 0.0])
        
        # PID control on force magnitude
        force_error = self.target_force - force_magnitude
        
        # Proportional term
        p_term = self.kp_force * force_error
        
        # Integral term (with anti-windup)
        self.force_error_integral += force_error * dt
        self.force_error_integral = np.clip(self.force_error_integral, -10, 10)
        i_term = self.ki_force * self.force_error_integral
        
        # Derivative term
        force_error_derivative = (force_error - self.last_force_error) / dt if dt > 0 else 0
        d_term = self.kd_force * force_error_derivative
        self.last_force_error = force_error
        
        # Total velocity adjustment
        v_adjust = p_term + i_term + d_term
        v_adjust = np.clip(v_adjust, -self.maintain_speed, self.maintain_speed)
        
        # Move toward/away from object based on force error
        vel = v_adjust * direction
        
        return np.array([vel[0], vel[1], 0.0])
    
    def reset(self):
        """Reset solver state."""
        super().reset()
        self.force_error_integral = 0.0
        self.last_force_error = 0.0
        self.last_force = np.zeros(3)


class PositionBasedContactSolver(ContactMaintainSolverBase):
    """Contact maintenance without force feedback.
    
    Uses position estimates to maintain contact at a specified distance.
    
    Parameters
    ----------
    robot_radius : float
        Radius of the robot body (default 0.06m for small robot).
    object_radius : float
        Radius of the object (or half-extent for box).
    contact_offset : float
        Offset from robot center to bumper contact point.
    kp_distance : float
        Proportional gain for distance control.
    approach_speed : float
        Speed when approaching object.
    maintain_speed : float
        Speed for adjustments during contact.
    """
    
    def __init__(self, robot_radius=0.06, object_radius=0.5, contact_offset=0.055,
                 kp_distance=1.0, approach_speed=0.1, maintain_speed=0.05):
        super().__init__(approach_speed, maintain_speed)
        
        self.robot_radius = robot_radius
        self.object_radius = object_radius
        self.contact_offset = contact_offset
        self.kp_distance = kp_distance
        
        # Target distance for contact
        self.target_distance = robot_radius + object_radius + contact_offset
    
    def compute_velocity(self, robot_pos, robot_theta, object_pos, 
                        contact_force=None, **kwargs):
        """Compute velocity based on position estimate.
        
        Parameters
        ----------
        robot_pos : np.ndarray, shape (2,)
            Robot position.
        robot_theta : float
            Robot orientation.
        object_pos : np.ndarray, shape (2,)
            Object position.
        contact_force : np.ndarray, shape (3,), optional
            Contact force (only used to track actual contact state, not for control).
        
        Returns
        -------
        np.ndarray, shape (3,)
            Velocity command (vx, vy, omega).
        """
        robot_pos = np.array(robot_pos)
        object_pos = np.array(object_pos)[:2] if len(object_pos) > 2 else np.array(object_pos)
        
        # Direction to object
        to_object = object_pos - robot_pos
        distance = np.linalg.norm(to_object)
        direction = unit(to_object) if distance > 0 else np.array([1, 0])
        
        # Track actual contact state if force is provided
        if contact_force is not None:
            force_magnitude = np.linalg.norm(contact_force[:2])
            self.in_contact = force_magnitude > 0.5
            if self.in_contact:
                self.contact_established = True
        
        # Distance error (positive = too far, negative = too close)
        distance_error = distance - self.target_distance
        
        if distance_error > 0.05:
            # Too far - approach
            vel = self.approach_speed * direction
        elif distance_error < -0.03:
            # Too close - back off
            vel = -self.maintain_speed * 0.5 * direction
        else:
            # Near target distance - fine adjustments
            v_adjust = self.kp_distance * distance_error
            v_adjust = np.clip(v_adjust, -self.maintain_speed, self.maintain_speed)
            vel = v_adjust * direction
            
            # Add small forward bias to maintain contact
            vel += 0.01 * direction
        
        return np.array([vel[0], vel[1], 0.0])


class AdaptiveContactSolver(ContactMaintainSolverBase):
    """Adaptive contact maintenance with hybrid force/position control.
    
    Uses force feedback when available, falls back to position-based
    estimation when force is not reliable.
    
    Parameters
    ----------
    force_solver : ForceBasedContactSolver
        Force-based solver to use when force feedback is available.
    position_solver : PositionBasedContactSolver
        Position-based solver to use as fallback.
    force_reliability_threshold : float
        Use force control only when force exceeds this threshold.
    """
    
    def __init__(self, force_solver=None, position_solver=None,
                 force_reliability_threshold=1.0):
        super().__init__()
        
        self.force_solver = force_solver or ForceBasedContactSolver()
        self.position_solver = position_solver or PositionBasedContactSolver()
        self.force_reliability_threshold = force_reliability_threshold
        
        # Blending parameter (0 = pure position, 1 = pure force)
        self.alpha = 0.0
    
    def compute_velocity(self, robot_pos, robot_theta, object_pos,
                        contact_force=None, dt=0.01, **kwargs):
        """Compute velocity using adaptive force/position blending.
        
        Parameters
        ----------
        robot_pos : np.ndarray, shape (2,)
            Robot position.
        robot_theta : float
            Robot orientation.
        object_pos : np.ndarray, shape (2,)
            Object position.
        contact_force : np.ndarray, shape (3,), optional
            Measured contact force.
        dt : float
            Time step.
        
        Returns
        -------
        np.ndarray, shape (3,)
            Velocity command (vx, vy, omega).
        """
        # Compute velocity from both solvers
        vel_position = self.position_solver.compute_velocity(
            robot_pos, robot_theta, object_pos, contact_force=contact_force, **kwargs
        )
        
        if contact_force is not None:
            force_magnitude = np.linalg.norm(contact_force[:2])
            
            # Update blending parameter based on force reliability
            if force_magnitude > self.force_reliability_threshold:
                # Trust force feedback more
                self.alpha = min(1.0, self.alpha + 0.1)
            else:
                # Trust position more
                self.alpha = max(0.0, self.alpha - 0.05)
            
            vel_force = self.force_solver.compute_velocity(
                robot_pos, robot_theta, object_pos, contact_force=contact_force, 
                dt=dt, **kwargs
            )
            
            # Blend velocities
            vel = self.alpha * vel_force + (1 - self.alpha) * vel_position
        else:
            self.alpha = 0.0
            vel = vel_position
        
        # Update contact state
        self.in_contact = self.force_solver.in_contact or self.position_solver.in_contact
        self.contact_established = self.force_solver.contact_established or self.position_solver.contact_established
        
        return vel
    
    def reset(self):
        """Reset solver state."""
        super().reset()
        self.force_solver.reset()
        self.position_solver.reset()
        self.alpha = 0.0


class DifferentialDriveSolverMixin:
    """Mixin to convert holonomic commands to differential-drive commands."""
    
    def holonomic_to_diffdrive(self, vel_holonomic, robot_theta, 
                               object_pos, robot_pos, max_omega=2.0):
        """Convert holonomic velocity to differential-drive (v, omega).
        
        Parameters
        ----------
        vel_holonomic : np.ndarray, shape (3,)
            Holonomic velocity (vx, vy, omega).
        robot_theta : float
            Current robot orientation.
        object_pos : np.ndarray, shape (2,)
            Object position for heading reference.
        robot_pos : np.ndarray, shape (2,)
            Robot position.
        max_omega : float
            Maximum angular velocity.
        
        Returns
        -------
        tuple (v, omega)
            Differential-drive velocity command.
        """
        # Direction to object
        to_object = np.array(object_pos)[:2] - np.array(robot_pos)[:2]
        distance = np.linalg.norm(to_object)
        
        if distance > 0.01:
            desired_heading = np.arctan2(to_object[1], to_object[0])
        else:
            desired_heading = robot_theta
        
        # Heading error
        heading_error = wrap_to_pi(desired_heading - robot_theta)
        
        # Compute linear velocity (magnitude in desired direction)
        vel_linear = vel_holonomic[:2]
        v_magnitude = np.linalg.norm(vel_linear)
        
        # Only move forward if roughly aligned
        alignment = np.cos(heading_error)
        if alignment < 0:
            # Facing wrong way - just rotate
            v = 0.0
        else:
            v = v_magnitude * alignment
        
        # Angular velocity to turn toward target
        omega = 2.0 * heading_error
        omega = np.clip(omega, -max_omega, max_omega)
        
        return v, omega


class DiffDriveForceBasedSolver(ForceBasedContactSolver, DifferentialDriveSolverMixin):
    """Force-based contact maintenance for differential-drive robots.
    
    Inherits from ForceBasedContactSolver and adds differential-drive
    kinematic constraints.
    """
    
    def __init__(self, max_omega=2.0, **kwargs):
        super().__init__(**kwargs)
        self.max_omega = max_omega
    
    def compute_velocity(self, robot_pos, robot_theta, object_pos,
                        contact_force=None, dt=0.01, **kwargs):
        """Compute differential-drive velocity command.
        
        Returns
        -------
        tuple (v, omega)
            Linear and angular velocity for differential-drive robot.
        """
        # Get holonomic velocity from parent class
        vel_holo = super().compute_velocity(
            robot_pos, robot_theta, object_pos,
            contact_force=contact_force, dt=dt, **kwargs
        )
        
        # Convert to differential drive
        return self.holonomic_to_diffdrive(
            vel_holo, robot_theta, object_pos, robot_pos, self.max_omega
        )


class DiffDrivePositionBasedSolver(PositionBasedContactSolver, DifferentialDriveSolverMixin):
    """Position-based contact maintenance for differential-drive robots.
    
    Inherits from PositionBasedContactSolver and adds differential-drive
    kinematic constraints.
    """
    
    def __init__(self, max_omega=2.0, **kwargs):
        super().__init__(**kwargs)
        self.max_omega = max_omega
    
    def compute_velocity(self, robot_pos, robot_theta, object_pos,
                        contact_force=None, **kwargs):
        """Compute differential-drive velocity command.
        
        Returns
        -------
        tuple (v, omega)
            Linear and angular velocity for differential-drive robot.
        """
        # Get holonomic velocity from parent class
        vel_holo = super().compute_velocity(
            robot_pos, robot_theta, object_pos,
            contact_force=contact_force, **kwargs
        )
        
        # Convert to differential drive
        return self.holonomic_to_diffdrive(
            vel_holo, robot_theta, object_pos, robot_pos, self.max_omega
        )


class DiffDriveAdaptiveSolver(AdaptiveContactSolver, DifferentialDriveSolverMixin):
    """Adaptive contact maintenance for differential-drive robots."""
    
    def __init__(self, max_omega=2.0, force_solver=None, position_solver=None,
                 **kwargs):
        # Use diff-drive specific solvers
        if force_solver is None:
            force_solver = DiffDriveForceBasedSolver(max_omega=max_omega)
        if position_solver is None:
            position_solver = DiffDrivePositionBasedSolver(max_omega=max_omega)
        
        super().__init__(force_solver=force_solver, position_solver=position_solver, **kwargs)
        self.max_omega = max_omega
    
    def compute_velocity(self, robot_pos, robot_theta, object_pos,
                        contact_force=None, dt=0.01, **kwargs):
        """Compute differential-drive velocity command.
        
        Returns
        -------
        tuple (v, omega)
            Linear and angular velocity for differential-drive robot.
        """
        # Get velocities from both solvers
        v_pos, omega_pos = self.position_solver.compute_velocity(
            robot_pos, robot_theta, object_pos, contact_force=contact_force, **kwargs
        )
        
        if contact_force is not None:
            force_magnitude = np.linalg.norm(contact_force[:2])
            
            # Update blending parameter
            if force_magnitude > self.force_reliability_threshold:
                self.alpha = min(1.0, self.alpha + 0.1)
            else:
                self.alpha = max(0.0, self.alpha - 0.05)
            
            v_force, omega_force = self.force_solver.compute_velocity(
                robot_pos, robot_theta, object_pos, contact_force=contact_force,
                dt=dt, **kwargs
            )
            
            # Blend velocities
            v = self.alpha * v_force + (1 - self.alpha) * v_pos
            omega = self.alpha * omega_force + (1 - self.alpha) * omega_pos
        else:
            self.alpha = 0.0
            v, omega = v_pos, omega_pos
        
        # Update contact state
        self.in_contact = self.force_solver.in_contact or self.position_solver.in_contact
        self.contact_established = (self.force_solver.contact_established or 
                                   self.position_solver.contact_established)
        
        return v, omega


def create_solver(solver_type='force', robot_type='holonomic', **kwargs):
    """Factory function to create contact maintenance solvers.
    
    Parameters
    ----------
    solver_type : str
        Type of solver: 'force', 'position', 'adaptive'.
    robot_type : str
        Type of robot: 'holonomic' or 'diffdrive'.
    **kwargs : dict
        Arguments to pass to the solver constructor.
    
    Returns
    -------
    ContactMaintainSolverBase
        The created solver instance.
    """
    if robot_type == 'holonomic':
        solvers = {
            'force': ForceBasedContactSolver,
            'position': PositionBasedContactSolver,
            'adaptive': AdaptiveContactSolver,
        }
    elif robot_type == 'diffdrive':
        solvers = {
            'force': DiffDriveForceBasedSolver,
            'position': DiffDrivePositionBasedSolver,
            'adaptive': DiffDriveAdaptiveSolver,
        }
    else:
        raise ValueError(f"Unknown robot type: {robot_type}. "
                        f"Available: 'holonomic', 'diffdrive'")
    
    if solver_type not in solvers:
        raise ValueError(f"Unknown solver type: {solver_type}. "
                        f"Available: {list(solvers.keys())}")
    
    return solvers[solver_type](**kwargs)

