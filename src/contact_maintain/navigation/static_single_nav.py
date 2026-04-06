"""Static Single Navigation Scheme.

This navigation scheme allows only one robot to move at a time, creating
a static environment problem instead of a dynamic one. This simplifies
collision avoidance and makes the system more predictable.

Specialized for pushing problem: Predictable, avoids dynamic collision issues.

Step structure (navigating phase):
1. Move to the ring via Bi-RRT (one robot at a time, ID-based priority)
2. (Trivial for static_single) Already at ring above target t_param
3. Move closer to object and touch it with a short P approach

Author: Contact Maintain Team
"""
import re
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from contact_maintain.navigation_controller import NavigationController
from contact_maintain.robot_message import RobotMessage, MonitorState
from contact_maintain.navigation.bi_rrt_planner import plan_to_ring_point, compute_ring_target


class StaticSingleNavigationController(NavigationController):
    """Static Single Navigation Controller.
    
    Only one robot moves at a time. Other robots remain stationary,
    creating a static environment for the moving robot.
    
    Coordination:
    - Priority system: robot ID (lowest ID moves first)
    - Message-based: robots coordinate via messages to determine who moves
    - Sequential execution: one robot at a time, ordered by ID
    """
    
    def __init__(
        self,
        radius: float = 0.06,
        max_speed: float = 0.3,
        position_threshold: float = 0.05,
    ):
        """
        Parameters
        ----------
        radius : float
            Robot radius (meters)
        max_speed : float
            Maximum speed (m/s)
        position_threshold : float
            Distance threshold to consider "at target" (meters)
        """
        self.radius = radius
        self.max_speed = max_speed
        self.position_threshold = position_threshold
        
        # Cache for robot IDs (extracted from names)
        self._robot_id_cache: Dict[str, int] = {}
        
         # Per-robot path cache for step 1 (to ring)
        self._paths: Dict[str, Optional[List[np.ndarray]]] = {}
        self._path_indices: Dict[str, int] = {}
    
    def _get_robot_id(self, robot_name: str) -> int:
        """Extract robot ID from robot name.
        
        Examples: "R_01" -> 1, "R_02" -> 2, "robot_1" -> 1
        Falls back to hash if format doesn't match.
        """
        if robot_name in self._robot_id_cache:
            return self._robot_id_cache[robot_name]
        
        # Try to extract numeric ID from name
        match = re.search(r'(\d+)', robot_name)
        if match:
            robot_id = int(match.group(1))
        else:
            # Fallback: use hash (consistent but arbitrary)
            robot_id = hash(robot_name) % 1000
        
        self._robot_id_cache[robot_name] = robot_id
        return robot_id
    
    def compute_velocity(
        self,
        robot_name: str,
        robot_position: np.ndarray,
        robot_heading: float,
        target_t_param: float,
        object_position: np.ndarray,
        object_orientation: float,
        object_state: Dict,
        other_robot_messages: List[RobotMessage],
        generic_object: Any,
    ) -> np.ndarray:
        """Compute velocity for static single navigation.
        
        Only one robot moves at a time. Robot with lowest ID moves first.
        Coordination via messages: check if any lower-ID robot is still in
        early navigation steps (step 1 or 2). If a lower-ID robot is already
        in step 3 (approach/touch, more stable), this robot is allowed to move.
        """
        # Get this robot's ID
        my_id = self._get_robot_id(robot_name)
        
        # --- Step classification: ring vs boundary ---
        # Boundary and ring positions for target t_param
        boundary_pos, ring_pos = self._compute_boundary_and_ring_positions(
            target_t_param, object_position, object_orientation, generic_object
        )
        d_to_ring = np.linalg.norm(robot_position - ring_pos)
        d_to_boundary = np.linalg.norm(robot_position - boundary_pos)
        ring_tol = 0.1
        contact_tol = 0.08
        
        # Compute distance to final ring target (above t_param)
        distance_to_ring_target = d_to_ring
        
        # Check if any robot with lower ID is currently moving (via messages)
        # A robot is "moving" if it's in NAVIGATING state and in early steps:
        # step 1 (to ring) or step 2 (along ring). If a lower-ID robot is in
        # step 3 (approach/touch), it's considered stable and we allow motion.
        lower_id_robot_moving = False
        for msg in other_robot_messages:
            other_id = self._get_robot_id(msg.robot_name)
            if other_id < my_id:
                # This robot has lower ID - check if it's moving
                if msg.state == MonitorState.NAVIGATING:
                    # navigation_step: 1=to ring, 2=along ring, 3=approach/touch
                    step = getattr(msg, "navigation_step", 0)
                    if step in (1, 2):
                        lower_id_robot_moving = True
                        break
        
        # If a robot with lower ID is moving, this robot must wait
        if lower_id_robot_moving:
            return np.zeros(3)
        
        # --- Step 1: to ring (Bi-RRT path following) ---
        if d_to_ring > ring_tol:
            return self._compute_step1_velocity(
                robot_name,
                robot_position,
                robot_heading,
                target_t_param,
                object_position,
                object_orientation,
                generic_object,
            )
        
        # --- Step 3: approach/touch (short P approach to boundary) ---
        if d_to_boundary > contact_tol:
            # We're on/near the ring but not yet close enough to boundary: P toward boundary
            return self._compute_step3_velocity(
                robot_position,
                robot_heading,
                boundary_pos,
            )
        
        # Already very close to boundary (or in contact) - hold position
            return np.zeros(3)
    
    # ------------------------------------------------------------------
    # Step 1: Bi-RRT to ring
    # ------------------------------------------------------------------
    def _compute_step1_velocity(
        self,
        robot_name: str,
        robot_position: np.ndarray,
        robot_heading: float,
        target_t_param: float,
        object_position: np.ndarray,
        object_orientation: float,
        generic_object: Any,
    ) -> np.ndarray:
        """Step 1: move to ring using Bi-RRT + P-following."""
        path = self._paths.get(robot_name)
        idx = self._path_indices.get(robot_name, 0)
        
        # Plan path if needed
        if path is None or idx >= len(path):
            new_path = plan_to_ring_point(
                start=robot_position,
                generic_object=generic_object,
                t_param=target_t_param,
                object_position=object_position,
                object_orientation=object_orientation,
                robot_radius=self.radius,
                extra_margin=0.2,
            )
            if not new_path or len(new_path) < 2:
                # Fallback: simple P controller toward ring target
                boundary_pos, ring_pos = self._compute_boundary_and_ring_positions(
                    target_t_param, object_position, object_orientation, generic_object
                )
                return self._compute_step3_velocity(
                    robot_position, robot_heading, ring_pos
                )
            self._paths[robot_name] = new_path
            self._path_indices[robot_name] = 0
            path = new_path
            idx = 0
        
        # Follow current waypoint
        waypoint = np.array(path[idx], dtype=float)
        direction = waypoint - robot_position
        dist = np.linalg.norm(direction)
        
        # Advance to next waypoint if close
        waypoint_tol = 0.05
        if dist < waypoint_tol and idx < len(path) - 1:
            idx += 1
            self._path_indices[robot_name] = idx
            waypoint = np.array(path[idx], dtype=float)
            direction = waypoint - robot_position
            dist = np.linalg.norm(direction)
        
        if dist > 1e-6:
            direction = direction / dist
            speed = min(self.max_speed, dist * 2.0)
            vel_2d = direction * speed
        else:
            vel_2d = np.zeros(2)
        
        # Heading control toward waypoint
        target_heading = np.arctan2(
            waypoint[1] - robot_position[1],
            waypoint[0] - robot_position[0]
        )
        heading_error = self._normalize_angle(target_heading - robot_heading)
        omega = 2.0 * heading_error
        
        return np.array([vel_2d[0], vel_2d[1], omega])
    
    # ------------------------------------------------------------------
    # Step 3: approach / touch
    # ------------------------------------------------------------------
    def _compute_step3_velocity(
        self,
        robot_position: np.ndarray,
        robot_heading: float,
        boundary_pos: np.ndarray,
    ) -> np.ndarray:
        """Step 3: short P approach from ring to boundary."""
        direction = boundary_pos - robot_position
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            # Slower approach speed for stability
            speed = min(0.15, distance * 2.0)
            vel_2d = direction * speed
        else:
            vel_2d = np.zeros(2)
        
        target_heading = np.arctan2(
            boundary_pos[1] - robot_position[1],
            boundary_pos[0] - robot_position[0]
        )
        heading_error = self._normalize_angle(target_heading - robot_heading)
        omega = 2.0 * heading_error
        return np.array([vel_2d[0], vel_2d[1], omega])
    
    def _compute_boundary_and_ring_positions(
        self,
        t_param: float,
        object_position: np.ndarray,
        object_orientation: float,
        generic_object: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute boundary contact point and ring point for t_param."""
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(generic_object)
        result = param.parameter_to_point(t_param)
        local_point = np.array([result[0][0], result[0][1]], dtype=float)
        
        # Transform to world frame
        c, s = np.cos(object_orientation), np.sin(object_orientation)
        R = np.array([[c, -s], [s, c]])
        boundary_pos = R @ local_point + object_position
        
        normal_local = param.get_normal_vector(t_param, outward=True)
        normal_world = R @ normal_local
        
        # Ring offset: 1.5 * radius + 0.2 (match planner and monitor)
        ring_offset = 1.5 * self.radius + 0.2
        ring_pos = boundary_pos + normal_world * ring_offset
        
        return boundary_pos, ring_pos
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def reset(self):
        """Reset controller state."""
        self._robot_id_cache.clear()
        self._paths.clear()
        self._path_indices.clear()