"""APF Navigation (Rewritten for Pushing Problem).

This is a rewritten version of APF navigation, specifically optimized
for contact maintenance scenarios. It is less general than the original
APF but simpler and more predictable.

Features:
- Simplified potential field calculations
- Optimized collision avoidance for contact maintenance
- All robots navigate simultaneously
- Specialized for multi-robot pushing scenarios

Author: Contact Maintain Team
"""
import numpy as np
from typing import List, Optional, Tuple, Any


class APFNavigatorPushing:
    """APF Navigator rewritten specifically for pushing problem.
    
    This navigator is simplified and optimized for contact maintenance
    scenarios. It trades generality for simplicity and predictability.
    
    Key simplifications:
    - Removed complex state machine (NORMAL/WAITING/YIELDING)
    - Simplified collision prediction
    - Optimized for object boundary navigation
    - Focused on contact maintenance use case
    """
    
    def __init__(
        self,
        radius: float = 0.06,
        max_speed: float = 0.3,
        k_goal: float = 1.0,
        k_obstacle: float = 0.5,
        k_wall: float = 0.3,
        repulsion_distance: float = 0.5,
    ):
        """
        Parameters
        ----------
        radius : float
            Robot radius (meters)
        max_speed : float
            Maximum speed (m/s)
        k_goal : float
            Goal attraction gain
        k_obstacle : float
            Neighbor repulsion gain
        k_wall : float
            Wall repulsion gain
        repulsion_distance : float
            Maximum distance for repulsion forces (meters)
        """
        self.radius = radius
        self.safety_radius = radius + 0.15
        self.max_speed = max_speed
        self.k_goal = k_goal
        self.k_obstacle = k_obstacle
        self.k_wall = k_wall
        self.repulsion_distance = repulsion_distance
        
        # Simplified state (no complex state machine)
        self.last_tangent_directions = {}  # robot_id -> 'increasing' or 'decreasing'
    
    def compute_velocity(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        other_robot_positions: List[np.ndarray],
        obstacles: Optional[List[List[Tuple[float, float]]]] = None,
        robot_id: int = 0,
        object_position: Optional[np.ndarray] = None,
        object_orientation: Optional[float] = None,
        target_t_param: Optional[float] = None,
    ) -> np.ndarray:
        """Compute velocity using simplified APF.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current robot position (x, y)
        target_position : np.ndarray
            Target position (x, y)
        other_robot_positions : List[np.ndarray]
            Positions of other robots
        obstacles : Optional[List[List[Tuple[float, float]]]]
            Static obstacles as list of vertex lists
        robot_id : int
            Stable robot ID for state tracking
        object_position : Optional[np.ndarray]
            Object center position (for t_param-based calculations)
        object_orientation : Optional[float]
            Object orientation (for t_param-based calculations)
        target_t_param : Optional[float]
            Target t_param (for arc-based navigation)
            
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy)
        """
        my_pos = current_position[:2]
        goal_pos = target_position[:2]
        
        # Goal attraction force
        to_goal = goal_pos - my_pos
        dist_goal = np.linalg.norm(to_goal)
        F_goal = self.k_goal * to_goal / (dist_goal + 1e-6)
        
        # Robot-robot repulsion
        F_obs = np.zeros(2)
        for other_pos in other_robot_positions:
            rel_pos = my_pos - other_pos[:2]
            dist = np.linalg.norm(rel_pos)
            if dist < self.repulsion_distance and dist > 1e-3:
                F_obs += self.k_obstacle * (1.0/dist - 1.0/self.repulsion_distance) * (rel_pos / dist)
        
        # Wall/obstacle repulsion (simplified)
        F_wall = np.zeros(2)
        if obstacles:
            F_wall = self._compute_wall_repulsion(
                my_pos, goal_pos, obstacles, robot_id,
                object_position, object_orientation, target_t_param
            )
        
        # Total force
        F_total = F_goal + F_obs + F_wall
        
        # Clamp to max speed
        vel = np.clip(F_total, -self.max_speed, self.max_speed)
        
        return vel
    
    def _compute_wall_repulsion(
        self,
        my_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[List[Tuple[float, float]]],
        robot_id: int,
        object_position: Optional[np.ndarray],
        object_orientation: Optional[float],
        target_t_param: Optional[float],
    ) -> np.ndarray:
        """Compute wall repulsion force (simplified for pushing problem)."""
        F_wall = np.zeros(2)
        
        # Try t_param-based calculation if available (more accurate for object boundary)
        use_t_param = (object_position is not None and 
                      object_orientation is not None and 
                      target_t_param is not None)
        
        if use_t_param:
            try:
                from object_utils import ContactPointParameterization
                # This would require generic_object, but we'll use a simplified version
                # In practice, this should be passed in or accessed differently
                # For now, fall back to vertex-based method
                use_t_param = False
            except ImportError:
                use_t_param = False
        
        # Fallback to vertex-based method (simplified)
        if not use_t_param and obstacles:
            for obstacle in obstacles:
                for i in range(len(obstacle)):
                    p1, p2 = np.array(obstacle[i]), np.array(obstacle[(i + 1) % len(obstacle)])
                    seg_vec = p2 - p1
                    seg_len = np.linalg.norm(seg_vec)
                    if seg_len < 1e-6:
                        continue
                    
                    proj = np.clip(np.dot(my_pos - p1, seg_vec) / (seg_len ** 2), 0, 1)
                    closest = p1 + proj * seg_vec
                    dist = np.linalg.norm(my_pos - closest)
                    
                    if dist < self.repulsion_distance:
                        n = (my_pos - closest) / (dist + 1e-6)
                        repel_gain = 2.0 * self.k_wall * (1.0/dist - 1.0/self.repulsion_distance)
                        F_wall += repel_gain * n
                        
                        # Tangential sliding if goal is blocked
                        to_goal = goal_pos - my_pos
                        F_goal_dir = to_goal / (np.linalg.norm(to_goal) + 1e-6)
                        if np.dot(F_goal_dir, n) < 0.2:  # Goal blocked by wall
                            # Use remembered direction or default
                            remembered = self.last_tangent_directions.get(robot_id, 'increasing')
                            if remembered == 'decreasing':
                                tangent = np.array([-n[1], n[0]])
                            else:
                                tangent = np.array([n[1], -n[0]])
                                if robot_id not in self.last_tangent_directions:
                                    self.last_tangent_directions[robot_id] = 'increasing'
                            
                            slide_strength = self.k_goal * 1.5
                            F_wall += slide_strength * tangent
        
        return F_wall
    
    def reset(self):
        """Reset navigator state."""
        self.last_tangent_directions.clear()
