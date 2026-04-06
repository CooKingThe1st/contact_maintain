"""Divide-n-Conquer Navigation Scheme.

This navigation scheme assigns each robot a set of consecutive non-overlapping
edges on the object boundary. Robots only move within their assigned edges,
creating clear responsibility zones.

Specialized for pushing problem: Predictable movement patterns, clear responsibility zones.

Step structure (navigating phase):
1. Move to the ring via Bi-RRT (one robot at a time at the high level)
2. Move along the ring to the point above the assigned t_param (edge-constrained)
3. Move closer to the object and touch it with a short P approach

Author: Contact Maintain Team
"""
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from contact_maintain.navigation_controller import NavigationController
from contact_maintain.robot_message import RobotMessage
from contact_maintain.navigation.bi_rrt_planner import plan_to_ring_point


class DivideConquerNavigationController(NavigationController):
    """Divide-n-Conquer Navigation Controller.
    
    Each robot "manages" a set of consecutive non-overlapping edges.
    Robots only move within their assigned edge segments.
    
    Features:
    - Initial prep phase: Assigns edges to robots
    - Handles spawn/overlap cases
    - Distributed coordination via messages
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
        
        # Edge assignments: robot_name -> (t_start, t_end)
        # t_start and t_end are t_params defining the edge segment
        self.edge_assignments: Dict[str, Tuple[float, float]] = {}
        self.prep_phase_complete = False
        
        # Per-robot path cache for step 1 (to ring)
        self._paths: Dict[str, Optional[List[np.ndarray]]] = {}
        self._path_indices: Dict[str, int] = {}
    
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
        """Compute velocity for divide-n-conquer navigation.
        
        Robot only moves if target is within its assigned edge segment.
        """
        # Run prep phase if not complete
        if not self.prep_phase_complete:
            self._run_prep_phase(
                robot_name, robot_position, target_t_param,
                other_robot_messages, object_position, object_orientation, generic_object
            )
        
        # Check if target is within assigned edges
        if robot_name not in self.edge_assignments:
            # No assignment yet, stay still
            return np.zeros(3)
        
        t_start, t_end = self.edge_assignments[robot_name]
        
        # Handle wrap-around (t_param is periodic)
        target_in_range = False
        if t_start <= t_end:
            # Normal case: no wrap-around
            target_in_range = t_start <= target_t_param <= t_end
        else:
            # Wrap-around case: segment crosses t=0
            target_in_range = (target_t_param >= t_start) or (target_t_param <= t_end)
        
        if not target_in_range:
            # Target is outside assigned edges, stay still
            # (In a more sophisticated version, could request edge reassignment)
            return np.zeros(3)
        
        # --- Step classification: ring vs boundary ---
        boundary_pos, ring_pos = self._compute_boundary_and_ring_positions(
            target_t_param, object_position, object_orientation, generic_object
        )
        d_to_ring = np.linalg.norm(robot_position - ring_pos)
        d_to_boundary = np.linalg.norm(robot_position - boundary_pos)
        ring_tol = 0.1
        contact_tol = 0.08
        
        # --- Step 1: to ring (Bi-RRT to some ring point above assigned t_param) ---
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
        
        # --- Step 2: along ring (edge-constrained movement) ---
        # For now, we use a simple P controller toward the correct ring point.
        # Robots are already on/near the ring but may not yet be above their
        # assigned t_param along the boundary.
        if d_to_boundary > contact_tol:
            return self._compute_step2_velocity(
                robot_position,
                robot_heading,
                ring_pos,
            )
        
        # --- Step 3: approach / touch (short P toward boundary) ---
        if d_to_boundary > 0.0:
            return self._compute_step3_velocity(
                robot_position,
                robot_heading,
                boundary_pos,
            )
        
        # Already very close
        return np.zeros(3)
    
    def _run_prep_phase(
        self,
        robot_name: str,
        robot_position: np.ndarray,
        target_t_param: float,
        other_robot_messages: List[RobotMessage],
        object_position: np.ndarray,
        object_orientation: float,
        generic_object: Any,
    ):
        """Run initial prep phase to assign edges to robots.
        
        This handles cases where robots are on the same edge or need redistribution.
        """
        # Collect all robots and their target t_params
        all_robots = [(robot_name, target_t_param)]
        for msg in other_robot_messages:
            if msg.target_t_param is not None:
                all_robots.append((msg.robot_name, msg.target_t_param))
        
        if len(all_robots) == 0:
            return
        
        # Sort by t_param to assign consecutive edges
        all_robots.sort(key=lambda x: x[1])
        
        # Assign edges: divide boundary into equal segments
        num_robots = len(all_robots)
        segment_size = 1.0 / num_robots
        
        for i, (name, t_param) in enumerate(all_robots):
            t_start = i * segment_size
            t_end = ((i + 1) * segment_size) % 1.0
            
            # Ensure target is within assigned segment (with some margin)
            # If not, expand segment to include target
            if t_start <= t_end:
                if not (t_start <= t_param <= t_end):
                    # Expand segment to include target
                    if t_param < t_start:
                        t_start = t_param
                    elif t_param > t_end:
                        t_end = t_param
            else:
                # Wrap-around case
                if not ((t_param >= t_start) or (t_param <= t_end)):
                    # Expand segment
                    if t_param > t_end and t_param < t_start:
                        # Target is in the gap, expand both ends
                        t_start = t_param
                        t_end = t_param
            
            self.edge_assignments[name] = (t_start, t_end)
        
        # Mark prep phase as complete (for this robot's perspective)
        # In a fully distributed system, this would be coordinated via messages
        self.prep_phase_complete = True
    
    def _compute_target_position(
        self,
        t_param: float,
        object_position: np.ndarray,
        object_orientation: float,
        generic_object: Any,
    ) -> np.ndarray:
        """Compute target position from t_param."""
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(generic_object)
        result = param.parameter_to_point(t_param)
        local_point = np.array([result[0][0], result[0][1]], dtype=float)
        
        # Transform to world frame
        c, s = np.cos(object_orientation), np.sin(object_orientation)
        R = np.array([[c, -s], [s, c]])
        boundary_pos = R @ local_point + object_position
        
        # Get normal for offset
        normal_local = param.get_normal_vector(t_param, outward=True)
        normal_world = R @ normal_local
        offset_distance = 0.35  # 35cm offset
        target_position = boundary_pos + normal_world * offset_distance
        
        return target_position

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
                # Fallback: simple P controller toward ring position
                boundary_pos, ring_pos = self._compute_boundary_and_ring_positions(
                    target_t_param, object_position, object_orientation, generic_object
                )
                return self._compute_step2_velocity(
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
    # Step 2: along ring (toward correct ring point)
    # ------------------------------------------------------------------
    def _compute_step2_velocity(
        self,
        robot_position: np.ndarray,
        robot_heading: float,
        ring_pos: np.ndarray,
    ) -> np.ndarray:
        """Step 2: move along/around ring toward ring_pos (simple P)."""
        direction = ring_pos - robot_position
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            speed = min(self.max_speed, distance * 2.0)
            vel_2d = direction * speed
        else:
            vel_2d = np.zeros(2)
        
        target_heading = np.arctan2(
            ring_pos[1] - robot_position[1],
            ring_pos[0] - robot_position[0]
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
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def set_edge_assignment(self, robot_name: str, t_start: float, t_end: float):
        """Manually set edge assignment for a robot.
        
        This can be called during reconfiguration.
        """
        self.edge_assignments[robot_name] = (t_start, t_end)
    
    def reset(self):
        """Reset controller state."""
        self.edge_assignments.clear()
        self.prep_phase_complete = False
        self._paths.clear()
        self._path_indices.clear()