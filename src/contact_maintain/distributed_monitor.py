"""Distributed Monitor Module for Multi-Robot Swarm.

Each robot has its own DistributedMonitor that maintains local state and
coordinates between navigation and pushing controllers. The monitor updates
its state based on received messages from other robots.

State Machine (Simplified):
- NAVIGATING: Robot is moving to target position
- PUSHING: Robot is in contact and pushing (all robots must be in contact)

Author: Contact Maintain Team
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from contact_maintain.robot_message import RobotMessage, MonitorState
from contact_maintain.navigation_controller import NavigationController, create_navigation_controller
from contact_maintain.push_controller import PushController, create_push_controller


@dataclass
class RobotLocalState:
    """Local state information for a robot."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    in_contact: bool = False
    contact_force: float = 0.0
    distance_to_target: float = float('inf')
    current_t_param: float = 0.0
    navigation_step: int = 0  # 0=unknown, 1=to ring, 2=along ring, 3=approach/touch


class DistributedMonitor:
    """Per-robot distributed monitor for swarm coordination.
    
    Each robot maintains its own monitor that:
    - Tracks local state (position, contact, etc.)
    - Receives and processes messages from other robots
    - Coordinates between navigation and pushing controllers
    - Determines state transitions (NAVIGATING ↔ PUSHING)
    
    Parameters
    ----------
    robot_name : str
        Unique identifier for the robot
    robot : object
        Robot instance (from robot_factory)
    object_uid : int
        PyBullet UID of the object
    generic_object : GenericObject
        Object model for boundary parameterization
    navigation_scheme : str
        Navigation scheme: 'apf', 'static_single', or 'divide_conquer'
    push_controller_type : str
        Push controller type: 'phase7' or other
    position_threshold : float
        Distance threshold to consider "at target" (meters)
    contact_force_threshold : float
        Force threshold to consider "in contact" (Newtons)
    """
    
    def __init__(
        self,
        robot_name: str,
        robot: Any,
        object_uid: int,
        generic_object: Any,
        navigation_scheme: str = 'apf',
        push_controller_type: str = 'phase7',
        position_threshold: float = 0.05,
        contact_force_threshold: float = 0.5,
        navigation_only: bool = False,
        startup_mode: str = 'quick',
    ):
        self.robot_name = robot_name
        self.navigation_only = navigation_only
        self.robot = robot
        self.object_uid = object_uid
        self.generic_object = generic_object
        self.position_threshold = position_threshold
        self.contact_force_threshold = contact_force_threshold
        if startup_mode not in ('quick', 'full'):
            raise ValueError(
                f"Invalid startup_mode='{startup_mode}'. Expected 'quick' or 'full'."
            )
        self.startup_mode = startup_mode
        
        # State
        self.state = MonitorState.NAVIGATING
        self.target_t_param: Optional[float] = None
        self.local_state = RobotLocalState()
        self.received_messages: List[RobotMessage] = []
        self.current_time = 0.0
        
        # Controllers
        self.navigation_controller: Optional[NavigationController] = None
        self.push_controller: Optional[PushController] = None
        
        # Initialize navigation controller
        self.navigation_controller = create_navigation_controller(
            scheme=navigation_scheme,
            radius=0.06,
            max_speed=0.3,
        )
        
        # Push controller will be initialized when target_t_param is set
        self.push_controller_type = push_controller_type
        
        # Quick startup approach gains (rotate then creep to contact point)
        self.approach_kp = 0.5
        self.approach_max_speed = 0.1
        # Stop all motion as soon as sensor reports meaningful normal/lateral impulse (N).
        # Kept above PyBullet micro-noise; well below legacy 0.5 N "in_contact" gate.
        self.approach_stop_force_epsilon = 0.02
        # Within this effective distance (m), cap speed to a constant creep.
        self.approach_close_effective_distance_m = 0.06
        self.approach_creep_speed = 0.01  # m/s
    
    def set_target(self, target_t_param: float):
        """Set target t_param for navigation/pushing.
        
        Parameters
        ----------
        target_t_param : float
            Target t_param on object boundary (0-1)
        """
        self.target_t_param = target_t_param
        
        # Initialize push controller if not already done
        if self.push_controller is None and hasattr(self.robot, 'uid'):
            self.push_controller = create_push_controller(
                controller_type=self.push_controller_type,
                robot_uid=self.robot.uid,
                object_uid=self.object_uid,
                generic_object=self.generic_object,
                t_param=target_t_param,
            )
        elif self.push_controller is not None:
            # Update t_param if controller already exists
            # (This might require reinitialization depending on controller)
            pass
    
    def update(
        self,
        dt: float,
        object_state: Dict,
        received_messages: List[RobotMessage],
    ):
        """Update monitor state from received messages and local sensors.
        
        Parameters
        ----------
        dt : float
            Time step
        object_state : dict
            Object state with 'position', 'orientation', 'velocity', 'angular_velocity'
        received_messages : List[RobotMessage]
            Messages from other robots
        """
        self.current_time += dt
        self.received_messages = received_messages
        
        # Update local state from robot sensors
        self._update_local_state(object_state)
        
        # Check state transitions
        self._check_state_transitions(object_state)
    
    def _update_local_state(self, object_state: Dict):
        """Update local state from robot sensors."""
        # Get robot state
        pos, heading, _ = self.robot.get_state()
        self.local_state.position = pos[:2]
        self.local_state.heading = heading
        
        # Update contact state
        self._update_contact_state()
        
        # Compute distance to target
        if self.target_t_param is not None:
            self.local_state.distance_to_target = self._compute_distance_to_target(
                object_state['position'], object_state['orientation']
            )
            # Update current t_param (approximate from position)
            self.local_state.current_t_param = self._estimate_current_t_param(
                pos[:2], object_state['position'], object_state['orientation']
            )
            # Update navigation step (1: to ring, 2: along ring, 3: approach/touch)
            self.local_state.navigation_step = self._estimate_navigation_step(
                pos[:2], object_state['position'], object_state['orientation']
            )
        else:
            self.local_state.distance_to_target = float('inf')
            self.local_state.navigation_step = 0
    
    def _update_contact_state(self):
        """Update contact state from robot sensors."""
        # Try to get contact force from robot
        if hasattr(self.robot, 'get_contact_force'):
            try:
                force = self.robot.get_contact_force([self.object_uid], max_contacts=4)
                self.local_state.contact_force = np.linalg.norm(force[:2])
                self.local_state.in_contact = self.local_state.contact_force > self.contact_force_threshold
            except (AssertionError, Exception):
                self.local_state.contact_force = 0.0
                self.local_state.in_contact = False
        else:
            self.local_state.contact_force = 0.0
            self.local_state.in_contact = False
    
    def _compute_distance_to_target(
        self,
        object_position: np.ndarray,
        object_orientation: float,
    ) -> float:
        """Compute distance to target position."""
        if self.target_t_param is None:
            return float('inf')
        
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(self.generic_object)
        result = param.parameter_to_point(self.target_t_param)
        local_point = np.array([result[0][0], result[0][1]], dtype=float)
        
        # Transform to world frame
        c, s = np.cos(object_orientation), np.sin(object_orientation)
        R = np.array([[c, -s], [s, c]])
        target_pos = R @ local_point + object_position
        
        # Distance from robot position to target
        return np.linalg.norm(self.local_state.position - target_pos)
    
    def _estimate_current_t_param(
        self,
        robot_position: np.ndarray,
        object_position: np.ndarray,
        object_orientation: float,
    ) -> float:
        """Estimate current t_param from robot position."""
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(self.generic_object)
        
        # Transform to local frame
        c, s = np.cos(-object_orientation), np.sin(-object_orientation)
        R_inv = np.array([[c, -s], [s, c]])
        robot_local = R_inv @ (robot_position - object_position)
        
        # Find closest point on boundary
        robot_info = param.point_to_parameter(robot_local)
        return robot_info['parameter']
    
    def _estimate_navigation_step(
        self,
        robot_position: np.ndarray,
        object_position: np.ndarray,
        object_orientation: float,
    ) -> int:
        """Estimate navigation step (1: to ring, 2: along ring, 3: approach/touch).
        
        Heuristic based on distance to ring point and boundary.
        """
        if self.target_t_param is None:
            return 0
        
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(self.generic_object)
        
        # Boundary point for target t_param
        result = param.parameter_to_point(self.target_t_param)
        local_point = np.array([result[0][0], result[0][1]], dtype=float)
        
        c, s = np.cos(object_orientation), np.sin(object_orientation)
        R = np.array([[c, -s], [s, c]])
        boundary_pos = R @ local_point + object_position
        
        # Outward normal at target
        normal_local = param.get_normal_vector(self.target_t_param, outward=True)
        normal_world = R @ normal_local
        
        # Define ring offset: 1.5 * robot_radius + small margin
        robot_radius = 0.06  # default radius
        ring_offset = 1.5 * robot_radius + 0.2
        ring_pos = boundary_pos + normal_world * ring_offset
        
        d_to_ring = np.linalg.norm(robot_position - ring_pos)
        d_to_boundary = np.linalg.norm(robot_position - boundary_pos)
        
        # Simple thresholds
        ring_tol = 0.1
        contact_tol = 0.08  # distance from boundary considered "touch"
        
        if d_to_ring > ring_tol:
            # Far from ring: moving toward ring
            return 1
        elif d_to_boundary > contact_tol:
            # On/near ring but not yet close to boundary: moving along ring
            return 2
        else:
            # Close to boundary (or in contact): approach/touch
            return 3
    
    def _check_state_transitions(self, object_state: Dict):
        """Check and execute state transitions.
        
        Transition rules:
        - NAVIGATING → PUSHING: All robots at target AND all in contact
        - PUSHING → NAVIGATING: Not all robots in contact (reconfiguration or loss of contact)
        """
        # Collect state from all robots (self + received messages)
        all_robots_at_target = True
        all_robots_in_contact = True
        
        # Check self
        if self.state == MonitorState.NAVIGATING:
            at_target = self.local_state.distance_to_target < self.position_threshold
            if not at_target:
                all_robots_at_target = False
            if not self.local_state.in_contact:
                all_robots_in_contact = False
        
        # Check other robots from messages
        for msg in self.received_messages:
            if msg.state == MonitorState.NAVIGATING:
                # For navigating robots, check if they're at target
                # (We approximate this from distance - in practice, message should include this)
                # For now, assume if they're navigating, they might not be at target yet
                all_robots_at_target = False
            
            if not msg.in_contact:
                all_robots_in_contact = False
        
        # State transitions (skip PUSHING when navigation_only)
        if self.state == MonitorState.NAVIGATING and not self.navigation_only:
            # Transition to PUSHING if all robots are at target and in contact
            if all_robots_at_target and all_robots_in_contact:
                self.state = MonitorState.PUSHING
        elif self.state == MonitorState.PUSHING:
            # Transition back to NAVIGATING if not all robots in contact
            if not all_robots_in_contact:
                self.state = MonitorState.NAVIGATING

    def _compute_quick_approach_velocity(self, object_state: Dict) -> np.ndarray:
        """Compute direct approach command: rotate first, then creep to contact."""
        if self.target_t_param is None:
            return np.zeros(3)

        # Fresh read: control may run at a lower rate than physics; this minimizes
        # one-tick lag when a contact appears after the last monitor.update().
        self._update_contact_state()

        from object_utils import ContactPointParameterization

        param = ContactPointParameterization(self.generic_object)
        result = param.parameter_to_point(self.target_t_param)
        local_point = np.array([result[0][0], result[0][1]], dtype=float)

        # Transform target boundary point to world frame
        c, s = np.cos(object_state['orientation']), np.sin(object_state['orientation'])
        R = np.array([[c, -s], [s, c]])
        target_pos = R @ local_point + object_state['position']

        def _any_touch() -> bool:
            return self.local_state.contact_force > self.approach_stop_force_epsilon

        # Stop on first sensed force (do not wait for high in_contact threshold).
        if _any_touch():
            return np.zeros(3)

        # If already flagged in contact (high threshold), hold.
        if self.local_state.in_contact:
            return np.zeros(3)

        # Step 1: rotate in place to face contact point
        direction = target_pos - self.local_state.position
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return np.zeros(3)

        target_heading = np.arctan2(direction[1], direction[0])
        heading_error = self._normalize_angle(target_heading - self.local_state.heading)
        heading_threshold = 0.05
        if abs(heading_error) > heading_threshold:
            if _any_touch():
                return np.zeros(3)
            omega = np.clip(2.0 * heading_error, -1.0, 1.0)
            return np.array([0.0, 0.0, omega])

        # Step 2: creep straight toward target boundary point
        direction = direction / distance
        robot_radius = 0.06
        effective_distance = distance - robot_radius
        speed = min(self.approach_kp * max(0.0, effective_distance), self.approach_max_speed)
        if effective_distance < self.approach_close_effective_distance_m:
            speed = min(speed, self.approach_creep_speed)
        if _any_touch():
            return np.zeros(3)
        vel_2d = direction * max(0.0, speed)
        omega = 10.0 * heading_error
        return np.array([vel_2d[0], vel_2d[1], omega])
    
    def compute_velocity(
        self,
        object_state: Dict,
    ) -> np.ndarray:
        """Compute velocity command by delegating to appropriate controller.
        
        Parameters
        ----------
        object_state : dict
            Object state with 'position', 'orientation', 'velocity', 'angular_velocity'
            
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy, omega)
        """
        if self.state == MonitorState.PUSHING and self.push_controller is not None:
            # Use push controller
            return self.push_controller.compute_velocity(
                robot_position=self.local_state.position,
                robot_heading=self.local_state.heading,
                object_position=object_state['position'],
                object_orientation=object_state['orientation'],
                object_velocity=object_state['velocity'],
                object_angular_velocity=object_state['angular_velocity'],
                contact_force=self.local_state.contact_force,
                in_contact=self.local_state.in_contact,
                t=self.current_time,
                robot=self.robot,
            )
        elif self.state == MonitorState.NAVIGATING and self.navigation_controller is not None:
            # Use navigation controller
            if self.target_t_param is None:
                return np.zeros(3)

            if self.startup_mode == 'quick':
                return self._compute_quick_approach_velocity(object_state)

            return self.navigation_controller.compute_velocity(
                robot_name=self.robot_name,
                robot_position=self.local_state.position,
                robot_heading=self.local_state.heading,
                target_t_param=self.target_t_param,
                object_position=object_state['position'],
                object_orientation=object_state['orientation'],
                object_state=object_state,
                other_robot_messages=self.received_messages,
                generic_object=self.generic_object,
            )
        else:
            # No controller available or unknown state
            return np.zeros(3)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_message(self) -> RobotMessage:
        """Generate message to broadcast to other robots.
        
        Returns
        -------
        RobotMessage
            Message containing current robot state
        """
        return RobotMessage(
            robot_name=self.robot_name,
            position=self.local_state.position,
            state=self.state,
            target_t_param=self.target_t_param,
            in_contact=self.local_state.in_contact,
            contact_force=self.local_state.contact_force,
            navigation_step=self.local_state.navigation_step,
            timestamp=self.current_time,
        )
    
    def set_desired_object_motion(
        self,
        desired_velocity: np.ndarray,
        desired_angular_velocity: float,
    ):
        """Set desired object motion for push controller.
        
        Parameters
        ----------
        desired_velocity : np.ndarray
            Desired object linear velocity (vx, vy)
        desired_angular_velocity : float
            Desired object angular velocity (rad/s)
        """
        if self.push_controller is not None:
            self.push_controller.set_desired_object_motion(
                desired_velocity, desired_angular_velocity
            )
