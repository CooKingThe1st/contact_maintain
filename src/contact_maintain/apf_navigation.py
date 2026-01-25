"""APF (Artificial Potential Field) Navigation Module.

This module provides APF-based navigation with:
- Hierarchical Waiting
- Tangential Sliding (wall following)
- Local sensing simulation

Author: Contact Maintain Team
"""
import numpy as np
from typing import List, Optional, Tuple, Any
from enum import Enum


class RobotState(Enum):
    """Robot state for APF state machine"""
    NORMAL = "NORMAL"
    WAITING = "WAITING"
    YIELDING = "YIELDING"


class APFNavigator:
    """APF-based navigator for multi-robot collision avoidance.
    
    Uses artificial potential fields with hierarchical waiting and
    tangential sliding for obstacle avoidance.
    
    Parameters
    ----------
    radius : float
        Robot radius (meters).
    max_speed : float
        Maximum speed (m/s).
    sensing_radius : float
        IR sensing range (meters).
    collision_horizon : float
        Time horizon for collision prediction (seconds).
    k_goal : float
        Goal attraction gain.
    k_obstacle : float
        Neighbor repulsion gain.
    k_wall : float
        Wall repulsion gain.
    repulsion_distance : float
        Maximum distance for repulsion forces (meters).
    """
    
    def __init__(
        self,
        radius: float = 0.06,
        max_speed: float = 0.3,
        sensing_radius: float = 1.0,
        collision_horizon: float = 2.0,
        k_goal: float = 1.0,
        k_obstacle: float = 0.5,
        k_wall: float = 0.3,
        repulsion_distance: float = 0.5,
        generic_object: Optional[Any] = None,
    ):
        self.radius = radius
        self.safety_radius = radius + 0.15
        self.critical_radius = radius + 0.001  # True critical: robot touching obstacle
        self.max_speed = max_speed
        self.sensing_radius = sensing_radius
        self.collision_horizon = collision_horizon
        self.k_goal = k_goal
        self.k_obstacle = k_obstacle
        self.k_wall = k_wall
        self.repulsion_distance = repulsion_distance
        self.generic_object = generic_object
        
        # Initialize ContactPointParameterization if generic_object is provided
        self.parameterization = None
        if generic_object is not None:
            try:
                from object_utils import ContactPointParameterization
                self.parameterization = ContactPointParameterization(generic_object)
            except ImportError:
                self.parameterization = None
        
        # State machine parameters
        self.base_wait_time = 0.5
        self.scale_wait_base = 0.5
        self.gamma = 0.2
        
        # Open-space bias
        self.num_directions = 16
        self.directions = np.array([
            [np.cos(angle), np.sin(angle)]
            for angle in np.linspace(0, 2 * np.pi, self.num_directions, endpoint=False)
        ])
        
        # Per-robot state (for multi-robot scenarios)
        self.robot_states = {}  # robot_id -> RobotState
        self.wait_timers = {}  # robot_id -> float
        self.conflict_counters = {}  # robot_id -> int
        self.robot_ids = {}  # position tuple -> robot_id
        self.last_tangent_directions = {}  # robot_id -> 'ccw' or 'cw' - memory of last tangent choice
    
    def _get_robot_id(self, position: np.ndarray) -> int:
        """Get or create robot ID for state tracking."""
        pos_tuple = tuple(np.round(position, 3))
        if pos_tuple not in self.robot_ids:
            robot_id = len(self.robot_ids)
            self.robot_ids[pos_tuple] = robot_id
            self.robot_states[robot_id] = RobotState.NORMAL
            self.wait_timers[robot_id] = 0.0
            self.conflict_counters[robot_id] = 0
        return self.robot_ids[pos_tuple]
    
    def _sense_neighbors(
        self,
        my_pos: np.ndarray,
        other_robot_positions: List[np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Simulate IR sensing - returns (relative_position, relative_velocity)."""
        neighbors = []
        for other_pos in other_robot_positions:
            rel_pos = other_pos - my_pos
            dist = np.linalg.norm(rel_pos)
            if dist < self.sensing_radius:
                # Assume zero relative velocity (simplified)
                neighbors.append((rel_pos, np.zeros(2)))
        return neighbors
    
    def _predict_collision(
        self,
        my_pos: np.ndarray,
        my_vel: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray]],
        obstacles: Optional[List[List[Tuple[float, float]]]],
        intended_vel: Optional[np.ndarray] = None,
    ) -> bool:
        """Predict collision - allows parallel movement."""
        vel = intended_vel if intended_vel is not None else my_vel
        vel_mag = np.linalg.norm(vel)
        
        # Check robot-robot collisions
        for rel_pos, rel_vel_base in neighbors:
            rel_dist = np.linalg.norm(rel_pos)

            if rel_dist < 2 * self.safety_radius:
                return True
            
            # Approximate relative velocity
            rel_vel = rel_vel_base
            if intended_vel is not None:
                # Adjust by velocity difference
                vel_diff = intended_vel - my_vel
                rel_vel = rel_vel_base - vel_diff
            
            rel_vel_mag = np.linalg.norm(rel_vel)
            if rel_vel_mag > 0.01 and rel_dist < self.sensing_radius:
                rel_pos_norm = rel_pos / (rel_dist + 1e-6)
                rel_vel_dir = rel_vel / (rel_vel_mag + 1e-6)
                
                if np.dot(rel_vel_dir, -rel_pos_norm) > 0.3:
                    for t in np.linspace(0.1, self.collision_horizon, 10):
                        future_rel_pos = rel_pos + t * rel_vel
                        future_dist = np.linalg.norm(future_rel_pos)
                        if future_dist < 2 * self.safety_radius:
                            return True
        
        # Check obstacle collisions
        if obstacles:
            for obstacle in obstacles:
                for i in range(len(obstacle)):
                    p1 = np.array(obstacle[i])
                    p2 = np.array(obstacle[(i + 1) % len(obstacle)])
                    
                    seg_vec = p2 - p1
                    seg_len = np.linalg.norm(seg_vec)
                    if seg_len < 1e-6:
                        continue
                    
                    to_robot = my_pos - p1
                    proj = np.dot(to_robot, seg_vec) / (seg_len ** 2)
                    proj = np.clip(proj, 0, 1)
                    closest = p1 + proj * seg_vec
                    dist = np.linalg.norm(my_pos - closest)

                    # print(f"dist: {dist} from obstacle: {obstacle} to my_pos: {my_pos}")
                    
                    if dist < self.safety_radius:
                        # If robot is static (vel_mag ≈ 0), only consider true critical collision
                        # This allows robots close to obstacle to escape when not moving
                        if vel_mag < 0.01:
                            # Static robot: only true collision if within critical radius
                            if dist < self.critical_radius:
                                return True
                            # Close but not critical: allow movement to escape
                        else:
                            # Moving robot: use safety_radius as before
                            return True
                    
                    if vel_mag > 0.01:
                        to_obstacle = closest - my_pos
                        to_obstacle_dist = np.linalg.norm(to_obstacle)
                        
                        if to_obstacle_dist < self.repulsion_distance:
                            vel_dir = vel / vel_mag
                            to_obstacle_dir = to_obstacle / (to_obstacle_dist + 1e-6)
                            
                            if np.dot(vel_dir, to_obstacle_dir) > 0.5:
                                for t in np.linspace(0.1, min(self.collision_horizon, to_obstacle_dist / vel_mag), 10):
                                    future_pos = my_pos + t * vel
                                    to_robot_future = future_pos - p1
                                    proj_future = np.dot(to_robot_future, seg_vec) / (seg_len ** 2)
                                    proj_future = np.clip(proj_future, 0, 1)
                                    closest_future = p1 + proj_future * seg_vec
                                    dist_future = np.linalg.norm(future_pos - closest_future)
                                    
                                    if dist_future < self.safety_radius:
                                        return True
        
        return False
    
    def _get_wall_tangent(
        self,
        my_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: Optional[List[List[Tuple[float, float]]]],
    ) -> Optional[np.ndarray]:
        """Get wall tangent direction if near a wall."""
        if not obstacles:
            return None
        
        to_goal = goal_pos - my_pos
        dist_goal = np.linalg.norm(to_goal)
        F_goal = self.k_goal * to_goal / (dist_goal + 1e-6)
        
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
                    if np.dot(F_goal, n) < 0.2:
                        tangent = np.array([n[1], -n[0]])
                        return tangent
        
        return None
    
    def _open_space_bias(
        self,
        my_pos: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray]],
        obstacles: Optional[List[List[Tuple[float, float]]]],
        goal_pos: np.ndarray,
    ) -> np.ndarray:
        """Compute open-space bias direction, prioritizing wall tangent."""
        wall_tangent = self._get_wall_tangent(my_pos, goal_pos, obstacles)
        if wall_tangent is not None:
            return wall_tangent / (np.linalg.norm(wall_tangent) + 1e-6)
        
        best_dir = None
        best_score = -1.0
        
        for direction in self.directions:
            max_dist = self.sensing_radius
            min_dist = max_dist
            
            for rel_pos, _ in neighbors:
                proj = np.dot(rel_pos, direction)
                if proj > 0:
                    perp_dist = np.linalg.norm(rel_pos - proj * direction)
                    if perp_dist < 2 * self.safety_radius:
                        min_dist = min(min_dist, proj)
            
            if obstacles:
                for obstacle in obstacles:
                    for i in range(len(obstacle)):
                        p1 = np.array(obstacle[i])
                        p2 = np.array(obstacle[(i + 1) % len(obstacle)])
                        
                        seg_vec = p2 - p1
                        seg_len = np.linalg.norm(seg_vec)
                        if seg_len < 1e-6:
                            continue
                        
                        to_p1 = my_pos - p1
                        proj = np.dot(to_p1, seg_vec) / (seg_len ** 2)
                        proj = np.clip(proj, 0, 1)
                        closest_point = p1 + proj * seg_vec
                        to_closest = closest_point - my_pos
                        dist_to_segment = np.linalg.norm(to_closest)
                        proj_on_dir = np.dot(to_closest, direction)
                        
                        if proj_on_dir > 0:
                            perp_dist = np.linalg.norm(to_closest - proj_on_dir * direction)
                            if perp_dist < self.safety_radius + 0.1:
                                min_dist = min(min_dist, proj_on_dir)
            
            if min_dist > best_score:
                best_score = min_dist
                best_dir = direction
        
        if best_dir is None:
            return np.array([1.0, 0.0])
        
        return best_dir
    
    def _apf_control(
        self,
        my_pos: np.ndarray,
        goal_pos: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray]],
        obstacles: Optional[List[List[Tuple[float, float]]]],
        object_position: Optional[np.ndarray] = None,
        object_orientation: Optional[float] = None,
        robot_id: int = 0,
        target_t_param: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute APF control force."""
        to_goal = goal_pos - my_pos
        dist_goal = np.linalg.norm(to_goal)
        F_goal = self.k_goal * to_goal / (dist_goal + 1e-6)
        
        F_obs = np.zeros(2)
        for rel_pos, _ in neighbors:
            dist = np.linalg.norm(rel_pos)
            if dist < self.repulsion_distance and dist > 1e-3:
                F_obs += self.k_obstacle * (1.0/dist - 1.0/self.repulsion_distance) * (-rel_pos / dist)
        
        F_wall = np.zeros(2)
        min_obstacle_dist = float('inf')
        tangent_chosen = None
        
        # Try to use t_param-based calculation if generic_object is available
        use_t_param = (self.parameterization is not None and 
                      object_position is not None and 
                      object_orientation is not None)
        # print(f"use_t_param: {use_t_param} and object_position: {object_position} and object_orientation: {object_orientation}")
        
        if use_t_param and obstacles:
            # Use t_param-based calculation for cleaner arc distance and tangent
            try:
                # Transform to local frame for easier calculation
                c, s = np.cos(-object_orientation), np.sin(-object_orientation)
                R_inv = np.array([[c, -s], [s, c]])
                robot_local = R_inv @ (my_pos - object_position)
                goal_local = R_inv @ (goal_pos - object_position)
                
                # Use point_to_parameter to find robot's t_param
                robot_info = self.parameterization.point_to_parameter(robot_local)
                robot_t_param = robot_info['parameter']
                robot_min_dist = robot_info['distance']
                
                # Use target_t_param directly if provided, otherwise calculate from offset goal position
                if target_t_param is not None:
                    goal_t_param = target_t_param % 1.0  # Ensure in [0, 1)
                    # Still calculate distance for debugging
                    goal_info = self.parameterization.point_to_parameter(goal_local)
                    goal_min_dist = goal_info['distance']
                else:
                    # Fallback: calculate from offset goal position (less accurate)
                    goal_info = self.parameterization.point_to_parameter(goal_local)
                    goal_t_param = goal_info['parameter']
                    goal_min_dist = goal_info['distance']
                
                # Debug print: show calculated t_params
                if target_t_param is not None:
                    print(f"[robot_id={robot_id}] robot_t_param={robot_t_param:.4f} (from pos {my_pos}), goal_t_param={goal_t_param:.4f} (from target_t_param={target_t_param:.4f})")
                else:
                    print(f"[robot_id={robot_id}] robot_t_param={robot_t_param:.4f} (from pos {my_pos}), goal_t_param={goal_t_param:.4f} (from offset goal {goal_pos})")
                
                dist_to_boundary = robot_min_dist
                min_obstacle_dist = dist_to_boundary
                
                if dist_to_boundary < self.repulsion_distance:
                    # Normal vector pointing outward from boundary
                    normal_local = self.parameterization.get_normal_vector(robot_t_param, outward=True)
                    normal_world = (np.array([[np.cos(object_orientation), -np.sin(object_orientation)],
                                             [np.sin(object_orientation), np.cos(object_orientation)]]) 
                                   @ normal_local)
                    n = normal_world / (np.linalg.norm(normal_world) + 1e-6)
                    
                    # Repulsion force
                    repel_gain = 2.0 * self.k_wall * (1.0/dist_to_boundary - 1.0/self.repulsion_distance)
                    F_wall += repel_gain * n
                    
                    if np.dot(F_goal, n) < 0.2:
                        # Calculate arc distance using t_param
                        # Handle wrap-around: t_param is in [0, 1)
                        # dt_increasing: distance going forward (increasing t_param)
                        # dt_decreasing: distance going backward (decreasing t_param)
                        # They should add up to 1.0
                        raw_diff = goal_t_param - robot_t_param
                        if raw_diff >= 0:
                            dt_increasing = raw_diff
                            dt_decreasing = 1.0 - raw_diff
                        else:
                            dt_increasing = raw_diff + 1.0
                            dt_decreasing = -raw_diff
                        
                        # Verify they add to 1.0 (debug check)
                        sum_check = dt_increasing + dt_decreasing
                        if abs(sum_check - 1.0) > 1e-6:
                            print(f"WARNING: dt_increasing + dt_decreasing = {sum_check:.6f} != 1.0 (robot_t={robot_t_param:.4f}, goal_t={goal_t_param:.4f})")
                        
                        # Check if robot is very close to target to prevent oscillation
                        min_distance = min(dt_increasing, dt_decreasing)
                        close_to_target_threshold = 0.05  # ~5% of perimeter
                        
                        if min_distance < close_to_target_threshold:
                            # Robot is very close to target, disable tangent sliding to prevent oscillation
                            print(f"[robot_id={robot_id}] Very close to target (min_dist={min_distance:.4f} < {close_to_target_threshold}), disabling tangent force")
                        else:
                            # Choose direction with shorter arc length (shorter t_param difference)
                            if dt_increasing <= dt_decreasing:
                                # Increasing t_param direction is shorter
                                tangent_local = self.parameterization.get_tangent_vector(robot_t_param)
                                tangent_direction = 'increasing'
                            else:
                                # Decreasing t_param direction is shorter (negate tangent)
                                tangent_local = -self.parameterization.get_tangent_vector(robot_t_param)
                                tangent_direction = 'decreasing'
                            
                            # Transform tangent to world frame
                            tangent = (np.array([[np.cos(object_orientation), -np.sin(object_orientation)],
                                                [np.sin(object_orientation), np.cos(object_orientation)]]) 
                                      @ tangent_local)
                            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                            
                            # Store chosen direction for memory
                            self.last_tangent_directions[robot_id] = tangent_direction
                            tangent_chosen = tangent
                            
                            slide_strength = self.k_goal * 1.5
                            F_wall += slide_strength * tangent
                            
                            print(f"[robot_id={robot_id}] Chosen direction: {tangent_direction}, dt_increasing={dt_increasing:.4f}, dt_decreasing={dt_decreasing:.4f}, sum={sum_check:.4f}")
                            print(f"[robot_id={robot_id}] Distance to boundary: {dist_to_boundary:.4f}, tangent: {tangent_chosen}")
                # Memory fallback: if F_wall is zero but robot is still relatively close
                if np.linalg.norm(F_wall) < 1e-6 and min_obstacle_dist < 2.0 * self.repulsion_distance:
                    remembered_direction = self.last_tangent_directions.get(robot_id)
                    if remembered_direction is not None:
                        # Get tangent using get_tangent_vector
                        tangent_local = self.parameterization.get_tangent_vector(robot_t_param)
                        if remembered_direction == 'decreasing':
                            tangent_local = -tangent_local
                        
                        # Transform tangent to world frame
                        tangent = (np.array([[np.cos(object_orientation), -np.sin(object_orientation)],
                                            [np.sin(object_orientation), np.cos(object_orientation)]]) 
                                  @ tangent_local)
                        tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                        
                        slide_strength = self.k_goal * 0.8
                        F_wall += slide_strength * tangent
                    elif min_obstacle_dist > 2.5 * self.repulsion_distance:
                        # Clear memory if far enough
                        if robot_id in self.last_tangent_directions:
                            del self.last_tangent_directions[robot_id]
                elif min_obstacle_dist > 2.5 * self.repulsion_distance:
                    # Clear memory if far enough
                    if robot_id in self.last_tangent_directions:
                        del self.last_tangent_directions[robot_id]
                        
            except Exception as e:
                # Fallback to vertex-based method if t_param calculation fails
                use_t_param = False
                print(f"Warning: t_param calculation failed, using fallback: {e}")
        
        # Fallback to vertex-based method if t_param not available
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
                    min_obstacle_dist = min(min_obstacle_dist, dist)
                    
                    if dist < self.repulsion_distance:
                        n = (my_pos - closest) / (dist + 1e-6)
                        repel_gain = 2.0 * self.k_wall * (1.0/dist - 1.0/self.repulsion_distance)
                        F_wall += repel_gain * n
                        
                        if np.dot(F_goal, n) < 0.2:
                            # Fallback: use simple consistent tangent
                            # Use remembered direction if available
                            remembered_direction = self.last_tangent_directions.get(robot_id)
                            if remembered_direction == 'decreasing':
                                tangent = np.array([-n[1], n[0]])  # Opposite of default
                            else:
                                tangent = np.array([n[1], -n[0]])  # Default (increasing-like)
                                if remembered_direction is None:
                                    self.last_tangent_directions[robot_id] = 'increasing'
                            
                            slide_strength = self.k_goal * 1.5
                            F_wall += slide_strength * tangent
        


        F_total = F_goal + F_obs + F_wall

        # print(f"F_total: {F_total} from F_goal: {F_goal} and F_obs: {F_obs} and F_wall: {F_wall}")

        return F_total, F_goal, F_obs, F_wall
    
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
        """Compute velocity using APF.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current robot position (x, y).
        target_position : np.ndarray
            Target position (x, y).
        other_robot_positions : list
            Positions of other robots.
        obstacles : list, optional
            Static obstacles as list of vertex lists.
        robot_id : int, optional
            Stable robot ID for state tracking (default: 0).
            Should be assigned once and remain constant.
        
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy).
        """
        my_pos = current_position[:2]
        goal_pos = target_position[:2]
        
        # Initialize state for robot_id if first time
        if robot_id not in self.robot_states:
            self.robot_states[robot_id] = RobotState.NORMAL
            self.wait_timers[robot_id] = 0.0
            self.conflict_counters[robot_id] = 0
        
        state = self.robot_states[robot_id]
        
        # Sense neighbors
        neighbors = self._sense_neighbors(my_pos, other_robot_positions)
        
        # Get current velocity (assume zero for now, could be passed in future)
        my_vel = np.zeros(2)
        
        # Predict collision
        collision_predicted = self._predict_collision(my_pos, my_vel, neighbors, obstacles)
        
        # print(f"collision_predicted: {collision_predicted} and current state is: {state} with velocity: {my_vel}")
        dt = 1.0 / 60.0  # Control timestep
        
        # State machine
        if state == RobotState.NORMAL:
            if collision_predicted:
                self.robot_states[robot_id] = RobotState.WAITING
                scale_wait = self.scale_wait_base * (1 + self.gamma * self.conflict_counters[robot_id])
                # Hierarchical waiting: different robots wait different times
                # This prevents synchronized state transitions and oscillations
                self.wait_timers[robot_id] = self.base_wait_time + scale_wait * robot_id
                self.conflict_counters[robot_id] += 1
                return np.zeros(2)
            else:
                F_total, _, _, _ = self._apf_control(
                    my_pos, goal_pos, neighbors, obstacles,
                    object_position=object_position,
                    object_orientation=object_orientation,
                    robot_id=robot_id,
                    target_t_param=target_t_param
                )
                vel = np.clip(F_total, -self.max_speed, self.max_speed)
                return vel
        
        elif state == RobotState.WAITING:
            self.wait_timers[robot_id] -= dt
            # print(f"wait_timers: {self.wait_timers[robot_id]} for robot_id: {robot_id}")
            if not collision_predicted or self.wait_timers[robot_id] <= 0:
                self.robot_states[robot_id] = RobotState.YIELDING
            
            return np.zeros(2)
        
        elif state == RobotState.YIELDING:
            F_total_base, _, _, _ = self._apf_control(
                my_pos, goal_pos, neighbors, obstacles,
                object_position=object_position,
                object_orientation=object_orientation,
                robot_id=robot_id,
                target_t_param=target_t_param
            )
            bias_dir = self._open_space_bias(my_pos, neighbors, obstacles, goal_pos)
            epsilon = 0.9
            F_open_space = epsilon * self.max_speed * bias_dir
            F_total = F_total_base + F_open_space
            
            vel = np.clip(F_total, -self.max_speed, self.max_speed)


            
            collision_with_intended = self._predict_collision(my_pos, my_vel, neighbors, obstacles, intended_vel=vel)

            print(f"F_total: {F_total} for robot_id: {robot_id} in YIELDING state and velocity: {vel} from f_total_base: {F_total_base} and f_open_space: {F_open_space} and collision_with_intended: {collision_with_intended}")
            
            if not collision_with_intended:
                self.robot_states[robot_id] = RobotState.NORMAL
            else:
                if np.linalg.norm(vel) < 0.05:
                    self.wait_timers[robot_id] -= dt
                    if self.wait_timers[robot_id] <= 0:
                        self.robot_states[robot_id] = RobotState.NORMAL
                        self.wait_timers[robot_id] = self.base_wait_time
            
            return vel
        
        return np.zeros(2)
    
    def reset(self):
        """Reset navigator state."""
        self.robot_states = {}
        self.wait_timers = {}
        self.conflict_counters = {}
        self.robot_ids = {}
        self.last_tangent_directions = {}