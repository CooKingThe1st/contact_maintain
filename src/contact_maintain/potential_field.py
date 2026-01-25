"""Potential Field Navigation for Collision Avoidance.

This module provides potential field-based navigation for robots to reach
target positions while avoiding collisions with other robots and obstacles.

The navigation uses:
- Attractive potential: Pulls robot toward target
- Repulsive potential: Pushes robot away from obstacles

Author: Contact Maintain Team
"""
import numpy as np
from typing import List, Optional, Tuple


class PotentialFieldNavigator:
    """Potential field-based collision avoidance navigator.
    
    Uses attractive and repulsive potential fields to compute velocity
    commands that drive the robot toward a target while avoiding obstacles.
    
    Parameters
    ----------
    robot_radius : float
        Radius of the robot body (meters).
    safe_distance : float
        Distance at which repulsive forces start (meters).
        Typically 2-3x robot radius.
    k_att : float
        Attractive potential gain.
    k_rep : float
        Repulsive potential gain.
    d_goal_threshold : float
        Distance threshold for goal (reduces oscillation near goal).
    """
    
    def __init__(
        self,
        robot_radius: float = 0.06,
        safe_distance: float = 0.18,
        k_att: float = 1.0,
        k_rep: float = 0.5,
        d_goal_threshold: float = 0.02,
    ):
        self.robot_radius = robot_radius
        self.safe_distance = safe_distance
        self.k_att = k_att
        self.k_rep = k_rep
        self.d_goal_threshold = d_goal_threshold
        
        # Collision distance (absolute minimum)
        self.collision_distance = 2.0 * robot_radius
    
    def compute_velocity(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
        obstacles: List[np.ndarray],
        max_speed: float = 0.5,
        obstacle_radii: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Compute velocity command using potential field.
        
        Parameters
        ----------
        robot_pos : np.ndarray, shape (2,)
            Current robot position (x, y).
        target_pos : np.ndarray, shape (2,)
            Target position (x, y).
        obstacles : list of np.ndarray
            Positions of obstacles (other robots).
        max_speed : float
            Maximum velocity magnitude.
        obstacle_radii : list of float, optional
            Radii of obstacles. If None, uses robot_radius for all.
        
        Returns
        -------
        np.ndarray, shape (2,)
            Velocity command (vx, vy).
        """
        robot_pos = np.array(robot_pos, dtype=float)
        target_pos = np.array(target_pos, dtype=float)
        
        # Compute attractive force
        F_att = self._attractive_force(robot_pos, target_pos)
        
        # Compute repulsive forces from obstacles
        F_rep = np.zeros(2)
        if obstacle_radii is None:
            obstacle_radii = [self.robot_radius] * len(obstacles)
        
        for i, obs_pos in enumerate(obstacles):
            obs_pos = np.array(obs_pos, dtype=float)
            obs_radius = obstacle_radii[i] if i < len(obstacle_radii) else self.robot_radius
            F_rep += self._repulsive_force(robot_pos, obs_pos, obs_radius)
        
        # Total force
        F_total = F_att + F_rep
        
        # Convert to velocity (force-like output, but used as velocity)
        velocity = F_total
        
        # Limit speed
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = velocity / speed * max_speed
        
        return velocity
    
    def _attractive_force(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        """Compute attractive force toward target.
        
        Uses a conic potential (linear force) to avoid issues with
        parabolic potentials at large distances.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot position.
        target_pos : np.ndarray
            Target position.
        
        Returns
        -------
        np.ndarray
            Attractive force vector.
        """
        diff = target_pos - robot_pos
        d_goal = np.linalg.norm(diff)
        
        if d_goal < self.d_goal_threshold:
            # Very close to goal - stop
            return np.zeros(2)
        
        # Conic potential: linear force toward goal
        direction = diff / d_goal
        
        # Use parabolic near goal, linear far
        d_switch = 1.0  # meters
        if d_goal <= d_switch:
            # Parabolic (smooth approach)
            force_mag = self.k_att * d_goal
        else:
            # Conic (constant speed approach)
            force_mag = self.k_att * d_switch
        
        return force_mag * direction
    
    def _repulsive_force(
        self,
        robot_pos: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_radius: float = None,
    ) -> np.ndarray:
        """Compute repulsive force from obstacle.
        
        Uses inverse-distance potential that only activates within
        the safe distance threshold.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot position.
        obstacle_pos : np.ndarray
            Obstacle (other robot) position.
        obstacle_radius : float, optional
            Radius of obstacle. Defaults to robot_radius.
        
        Returns
        -------
        np.ndarray
            Repulsive force vector.
        """
        if obstacle_radius is None:
            obstacle_radius = self.robot_radius
        
        diff = robot_pos - obstacle_pos
        d_obs = np.linalg.norm(diff)
        
        # Distance to obstacle surface (accounting for robot and obstacle radii)
        d_surface = d_obs - self.robot_radius - obstacle_radius
        
        # Safe distance threshold (from obstacle surface)
        d_safe = self.safe_distance - self.robot_radius - obstacle_radius
        
        if d_surface >= d_safe or d_obs < 1e-6:
            # Outside influence range or coincident
            return np.zeros(2)
        
        # Direction away from obstacle
        direction = diff / d_obs
        
        # Repulsive force magnitude
        # Uses: F = k_rep * (1/d - 1/d_safe) * (1/d^2)
        # This ensures smooth transition at d_safe
        if d_surface < 0.001:
            d_surface = 0.001  # Prevent division by zero
        
        force_mag = self.k_rep * (1.0/d_surface - 1.0/d_safe) * (1.0/(d_surface**2))
        
        # Cap the force to prevent extreme values
        max_force = 5.0 * self.k_rep
        force_mag = min(force_mag, max_force)
        
        return force_mag * direction
    
    def compute_velocity_with_object(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
        other_robots: List[np.ndarray],
        object_pos: np.ndarray,
        object_boundary_points: List[np.ndarray],
        max_speed: float = 0.5,
    ) -> np.ndarray:
        """Compute velocity with object boundary avoidance.
        
        In addition to avoiding other robots, this method avoids the object
        boundary except at the target contact point.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Current robot position.
        target_pos : np.ndarray
            Target contact position on object.
        other_robots : list of np.ndarray
            Positions of other robots.
        object_pos : np.ndarray
            Object center position.
        object_boundary_points : list of np.ndarray
            Sampled points on object boundary (for avoidance).
        max_speed : float
            Maximum velocity.
        
        Returns
        -------
        np.ndarray
            Velocity command.
        """
        robot_pos = np.array(robot_pos, dtype=float)
        target_pos = np.array(target_pos, dtype=float)
        
        # Attractive force to target
        F_att = self._attractive_force(robot_pos, target_pos)
        
        # Repulsive from other robots
        F_rep_robots = np.zeros(2)
        for other_pos in other_robots:
            F_rep_robots += self._repulsive_force(robot_pos, np.array(other_pos))
        
        # Repulsive from object boundary (except near target)
        F_rep_object = np.zeros(2)
        target_boundary_dist = np.linalg.norm(target_pos - robot_pos)
        
        for boundary_point in object_boundary_points:
            boundary_point = np.array(boundary_point)
            
            # Skip points near target (we want to approach there)
            if np.linalg.norm(boundary_point - target_pos) < 0.05:
                continue
            
            # Add repulsion from boundary
            F_rep_object += self._repulsive_force(
                robot_pos, boundary_point, obstacle_radius=0.01  # Small radius for boundary
            ) * 0.3  # Weaker than robot repulsion
        
        # Total force
        F_total = F_att + F_rep_robots + F_rep_object
        
        # Limit speed
        speed = np.linalg.norm(F_total)
        if speed > max_speed:
            F_total = F_total / speed * max_speed
        
        return F_total
    
    def is_collision_imminent(
        self,
        robot_pos: np.ndarray,
        other_positions: List[np.ndarray],
    ) -> Tuple[bool, Optional[int]]:
        """Check if collision is imminent with any other robot.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot position.
        other_positions : list of np.ndarray
            Positions of other robots.
        
        Returns
        -------
        tuple
            (is_collision, index) - True if collision imminent, with index of closest.
        """
        robot_pos = np.array(robot_pos)
        
        min_dist = float('inf')
        closest_idx = None
        
        for i, other_pos in enumerate(other_positions):
            other_pos = np.array(other_pos)
            dist = np.linalg.norm(robot_pos - other_pos)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        is_collision = min_dist < self.collision_distance
        return is_collision, closest_idx if is_collision else None
    
    def get_safe_velocity_direction(
        self,
        robot_pos: np.ndarray,
        desired_velocity: np.ndarray,
        obstacles: List[np.ndarray],
    ) -> np.ndarray:
        """Adjust velocity direction to avoid obstacles.
        
        If the desired velocity would lead to collision, rotate it
        to find a safe direction.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot position.
        desired_velocity : np.ndarray
            Desired velocity vector.
        obstacles : list of np.ndarray
            Obstacle positions.
        
        Returns
        -------
        np.ndarray
            Safe velocity vector (possibly rotated from desired).
        """
        robot_pos = np.array(robot_pos)
        desired_velocity = np.array(desired_velocity)
        
        speed = np.linalg.norm(desired_velocity)
        if speed < 1e-6:
            return desired_velocity
        
        direction = desired_velocity / speed
        
        # Check if desired direction leads toward any obstacle
        for obs_pos in obstacles:
            obs_pos = np.array(obs_pos)
            to_obs = obs_pos - robot_pos
            dist_to_obs = np.linalg.norm(to_obs)
            
            if dist_to_obs < self.collision_distance:
                # Too close - back away
                escape_dir = -to_obs / dist_to_obs
                return escape_dir * speed
            
            if dist_to_obs < self.safe_distance:
                # Check if heading toward obstacle
                to_obs_norm = to_obs / dist_to_obs
                dot = np.dot(direction, to_obs_norm)
                
                if dot > 0.5:  # Heading toward obstacle
                    # Rotate velocity to go around
                    perp = np.array([-to_obs_norm[1], to_obs_norm[0]])
                    
                    # Choose rotation direction
                    cross = direction[0] * to_obs_norm[1] - direction[1] * to_obs_norm[0]
                    if cross < 0:
                        perp = -perp
                    
                    # Blend perpendicular direction
                    blend = (self.safe_distance - dist_to_obs) / self.safe_distance
                    new_dir = (1 - blend) * direction + blend * perp
                    new_dir = new_dir / np.linalg.norm(new_dir)
                    
                    return new_dir * speed
        
        return desired_velocity


class ObjectBoundaryNavigator(PotentialFieldNavigator):
    """Navigator specialized for approaching object boundary.
    
    Extends potential field navigation with specific handling for
    approaching a point on an object boundary while avoiding the
    rest of the boundary.
    """
    
    def __init__(
        self,
        robot_radius: float = 0.06,
        safe_distance: float = 0.18,
        approach_angle: float = 0.0,  # Approach perpendicular to boundary
        **kwargs
    ):
        super().__init__(robot_radius, safe_distance, **kwargs)
        self.approach_angle = approach_angle
    
    def compute_approach_velocity(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
        target_normal: np.ndarray,
        other_robots: List[np.ndarray],
        max_speed: float = 0.3,
    ) -> np.ndarray:
        """Compute velocity for approaching boundary at target point.
        
        The robot will approach from the outside of the boundary,
        following the target normal direction.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot position.
        target_pos : np.ndarray
            Target contact point on boundary.
        target_normal : np.ndarray
            Outward normal at target point.
        other_robots : list of np.ndarray
            Other robot positions.
        max_speed : float
            Maximum approach speed.
        
        Returns
        -------
        np.ndarray
            Velocity command.
        """
        robot_pos = np.array(robot_pos)
        target_pos = np.array(target_pos)
        target_normal = np.array(target_normal)
        target_normal = target_normal / (np.linalg.norm(target_normal) + 1e-6)
        
        # Compute approach waypoint (offset from boundary along normal)
        approach_offset = self.robot_radius + 0.02  # Small buffer
        approach_point = target_pos + target_normal * approach_offset
        
        # Distance to approach point
        to_approach = approach_point - robot_pos
        dist_to_approach = np.linalg.norm(to_approach)
        
        if dist_to_approach < self.d_goal_threshold:
            # At approach point - move directly to contact
            to_target = target_pos - robot_pos
            return to_target / (np.linalg.norm(to_target) + 1e-6) * 0.1  # Slow final approach
        
        # Navigate to approach point with collision avoidance
        return self.compute_velocity(
            robot_pos, approach_point, other_robots, max_speed
        )

