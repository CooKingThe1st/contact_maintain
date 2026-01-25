"""Controllers for contact maintenance robots."""
import numpy as np

from contact_maintain.util import wrap_to_pi, rot2d, unit


class HolonomicVelocityController:
    """Basic velocity controller for holonomic robots.
    
    This controller generates velocity commands to move the robot
    toward a target position or track a velocity reference.
    
    Parameters
    ----------
    kp_pos : float
        Proportional gain for position control.
    kp_theta : float
        Proportional gain for orientation control.
    max_linear_vel : float
        Maximum linear velocity (m/s).
    max_angular_vel : float
        Maximum angular velocity (rad/s).
    """
    
    def __init__(self, kp_pos=1.0, kp_theta=1.0, max_linear_vel=1.0, max_angular_vel=1.0):
        self.kp_pos = kp_pos
        self.kp_theta = kp_theta
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
    
    def position_control(self, current_pos, target_pos, current_theta=None, target_theta=None):
        """Compute velocity command to reach a target position.
        
        Parameters
        ----------
        current_pos : np.ndarray, shape (2,)
            Current (x, y) position.
        target_pos : np.ndarray, shape (2,)
            Target (x, y) position.
        current_theta : float, optional
            Current orientation. Required if target_theta is provided.
        target_theta : float, optional
            Target orientation.
        
        Returns
        -------
        np.ndarray, shape (3,)
            Velocity command (vx, vy, omega).
        """
        # Position error
        pos_error = np.array(target_pos) - np.array(current_pos)
        
        # Linear velocity command (P control)
        vel_linear = self.kp_pos * pos_error
        vel_norm = np.linalg.norm(vel_linear)
        if vel_norm > self.max_linear_vel:
            vel_linear = self.max_linear_vel * vel_linear / vel_norm
        
        # Angular velocity command
        omega = 0.0
        if target_theta is not None and current_theta is not None:
            theta_error = wrap_to_pi(target_theta - current_theta)
            omega = self.kp_theta * theta_error
            omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)
        
        return np.array([vel_linear[0], vel_linear[1], omega])
    
    def velocity_tracking(self, velocity_ref, current_theta=0.0, body_frame=False):
        """Pass through a velocity reference with saturation.
        
        Parameters
        ----------
        velocity_ref : np.ndarray, shape (2,) or (3,)
            Reference velocity. If shape (2,), interpreted as (vx, vy).
            If shape (3,), interpreted as (vx, vy, omega).
        current_theta : float, optional
            Current orientation (used if body_frame=True).
        body_frame : bool, optional
            If True, velocity_ref is in robot body frame and will be
            converted to world frame.
        
        Returns
        -------
        np.ndarray, shape (3,)
            Velocity command (vx, vy, omega) in world frame.
        """
        if len(velocity_ref) == 2:
            velocity_ref = np.array([velocity_ref[0], velocity_ref[1], 0.0])
        else:
            velocity_ref = np.array(velocity_ref)
        
        # Convert from body frame to world frame if needed
        if body_frame:
            R = rot2d(current_theta)
            vel_world = R @ velocity_ref[:2]
            velocity_ref = np.array([vel_world[0], vel_world[1], velocity_ref[2]])
        
        # Saturate linear velocity
        vel_linear = velocity_ref[:2]
        vel_norm = np.linalg.norm(vel_linear)
        if vel_norm > self.max_linear_vel:
            vel_linear = self.max_linear_vel * vel_linear / vel_norm
        
        # Saturate angular velocity
        omega = np.clip(velocity_ref[2], -self.max_angular_vel, self.max_angular_vel)
        
        return np.array([vel_linear[0], vel_linear[1], omega])


class DifferentialDriveController:
    """Basic velocity controller for differential-drive robots.
    
    Parameters
    ----------
    kp_distance : float
        Proportional gain for distance to target.
    kp_heading : float
        Proportional gain for heading correction.
    kp_theta : float
        Proportional gain for final orientation.
    max_linear_vel : float
        Maximum linear velocity (m/s).
    max_angular_vel : float
        Maximum angular velocity (rad/s).
    goal_tolerance : float
        Distance threshold to consider goal reached.
    """
    
    def __init__(self, kp_distance=1.0, kp_heading=2.0, kp_theta=1.0,
                 max_linear_vel=1.0, max_angular_vel=1.0, goal_tolerance=0.05):
        self.kp_distance = kp_distance
        self.kp_heading = kp_heading
        self.kp_theta = kp_theta
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.goal_tolerance = goal_tolerance
    
    def position_control(self, current_pos, current_theta, target_pos, target_theta=None):
        """Compute velocity command to reach a target position.
        
        Uses a move-to-pose strategy: first align heading, then drive forward.
        
        Parameters
        ----------
        current_pos : np.ndarray, shape (2,)
            Current (x, y) position.
        current_theta : float
            Current orientation.
        target_pos : np.ndarray, shape (2,)
            Target (x, y) position.
        target_theta : float, optional
            Target orientation (used for final alignment).
        
        Returns
        -------
        tuple (v, omega)
            Linear velocity and angular velocity command.
        """
        # Position error
        pos_error = np.array(target_pos) - np.array(current_pos)
        distance = np.linalg.norm(pos_error)
        
        # If close to goal, do final orientation adjustment
        if distance < self.goal_tolerance:
            if target_theta is not None:
                theta_error = wrap_to_pi(target_theta - current_theta)
                omega = self.kp_theta * theta_error
                omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)
                return 0.0, omega
            else:
                return 0.0, 0.0
        
        # Desired heading to face the target
        desired_heading = np.arctan2(pos_error[1], pos_error[0])
        heading_error = wrap_to_pi(desired_heading - current_theta)
        
        # Angular velocity to align heading
        omega = self.kp_heading * heading_error
        omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)
        
        # Linear velocity (reduce when not aligned)
        alignment = np.cos(heading_error)
        v = self.kp_distance * distance * max(0, alignment)
        v = np.clip(v, 0, self.max_linear_vel)
        
        return v, omega
    
    def velocity_tracking(self, v_ref, omega_ref):
        """Pass through velocity reference with saturation.
        
        Parameters
        ----------
        v_ref : float
            Reference linear velocity.
        omega_ref : float
            Reference angular velocity.
        
        Returns
        -------
        tuple (v, omega)
            Saturated velocity command.
        """
        v = np.clip(v_ref, -self.max_linear_vel, self.max_linear_vel)
        omega = np.clip(omega_ref, -self.max_angular_vel, self.max_angular_vel)
        return v, omega


class ContactMaintainController:
    """Controller for maintaining contact with an object.
    
    This is a placeholder for the contact maintenance algorithm.
    
    Parameters
    ----------
    approach_speed : float
        Speed when approaching the object.
    contact_force_target : float
        Target contact force to maintain.
    kp_force : float
        Proportional gain for force control.
    """
    
    def __init__(self, approach_speed=0.1, contact_force_target=5.0, kp_force=0.01):
        self.approach_speed = approach_speed
        self.contact_force_target = contact_force_target
        self.kp_force = kp_force
        
        # State
        self.in_contact = False
        self.contact_established = False
    
    def update(self, robot_pos, robot_theta, object_pos, contact_force=None):
        """Compute velocity command to approach and maintain contact.
        
        Parameters
        ----------
        robot_pos : np.ndarray, shape (2,)
            Current robot position.
        robot_theta : float
            Current robot orientation.
        object_pos : np.ndarray, shape (2,)
            Object position.
        contact_force : np.ndarray, shape (3,), optional
            Measured contact force. If None, assumes no force sensor.
        
        Returns
        -------
        np.ndarray, shape (3,)
            Velocity command (vx, vy, omega) for holonomic robot,
            or (v, omega) for differential-drive robot.
        """
        # Direction to object
        to_object = np.array(object_pos) - np.array(robot_pos)
        distance = np.linalg.norm(to_object)
        direction = unit(to_object) if distance > 0 else np.array([1, 0])
        
        if contact_force is not None:
            # Force-based control
            force_magnitude = np.linalg.norm(contact_force[:2])
            
            if force_magnitude > 0.1:  # In contact
                self.in_contact = True
                self.contact_established = True
                
                # Adjust velocity based on force error
                force_error = self.contact_force_target - force_magnitude
                v_adjust = self.kp_force * force_error
                
                # Move toward object if force too low, away if too high
                vel = v_adjust * direction
                return np.array([vel[0], vel[1], 0.0])
            else:
                self.in_contact = False
                
                # If contact was established but lost, try to recover
                if self.contact_established:
                    vel = self.approach_speed * direction
                    return np.array([vel[0], vel[1], 0.0])
        
        # No force sensor or not in contact: approach the object
        vel = self.approach_speed * direction
        return np.array([vel[0], vel[1], 0.0])
    
    def reset(self):
        """Reset the controller state."""
        self.in_contact = False
        self.contact_established = False

