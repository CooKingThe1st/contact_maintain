"""Robot Agent with Decentralized Navigation and Pushing.

Each robot agent has:
- Navigation module (ORCA-based)
- Pushing module (contact maintenance controller)
- Communication interface with SwarmHost

Author: Contact Maintain Team
"""
from typing import Optional, Dict, Any
import re
import numpy as np

from contact_maintain.apf_navigation import APFNavigator
from contact_maintain.contact_maintain_controller import InstantVelocityMatcher, WrenchTrackingController


class RobotAgent:
    """Decentralized robot agent with navigation and pushing capabilities.
    
    Each robot agent is responsible for:
    - Computing its own navigation velocity (using apf)
    - Computing its own pushing velocity (using contact controller)
    - Communicating with SwarmHost for goal updates
    
    Parameters
    ----------
    robot : object
        Robot instance (from robot_factory).
    name : str
        Robot name/ID.
    object_uid : int
        PyBullet UID of the object.
    generic_object : GenericObject
        Object model for boundary parameterization.
    navigation_type : str
        Navigation method ('orca' or 'potential_field').
    pushing_type : str
        Pushing controller type ('velocity' or 'wrench').
    """
    
    def __init__(
        self,
        robot: Any,
        name: str,
        object_uid: int,
        generic_object: Any,
        navigation_type: str = 'orca',
        pushing_type: str = 'velocity',
        force_distributor: Any = None,
    ):
        self.robot = robot
        self.name = name
        self.object_uid = object_uid
        self.generic_object = generic_object
        self.navigation_type = navigation_type
        self.pushing_type = pushing_type
        # Optional shared ForceDistributorPro for wrench controllers
        self.force_distributor = force_distributor
        
        # Extract robot_id from name (e.g., "R_01" -> 0, "R_02" -> 1)
        # Fallback to hash of name if format doesn't match
        try:
            # Try to extract number from name (assumes format like "R_01", "R_02", etc.)
            match = re.search(r'(\d+)', name)
            if match:
                self.robot_id = int(match.group(1)) - 1  # Convert to 0-indexed
            else:
                # Fallback: use hash of name
                self.robot_id = hash(name) % 1000
        except:
            # Ultimate fallback
            self.robot_id = hash(name) % 1000
        
        # Navigation module
        if navigation_type == 'apf':
            # repulsion_distance: distance at which obstacle repulsion is active
            # offset_distance (0.35m) should be >= repulsion_distance to avoid conflicts
            # This ensures REACHING target is outside repulsion zone
            self.navigator = APFNavigator(
                radius=0.06,
                max_speed=0.3,
                k_wall = 10,
                repulsion_distance = 0.3,  # Repulsion active within 30cm
                generic_object=generic_object  # Pass generic_object for t_param-based calculations
            )
        elif navigation_type == 'orca':
            from contact_maintain.orca_navigation import ORCANavigator
            self.navigator = ORCANavigator(
                time_step=1.0 / 60.0,
                radius=0.06,
                max_speed=0.3,
            )
        else:
            from contact_maintain.potential_field import PotentialFieldNavigator
            self.navigator = PotentialFieldNavigator(
                robot_radius=0.06,
                safe_distance=0.18,
            )
        
        # Pushing module (will be initialized when t_param is assigned)
        self.pushing_controller = None
        self.current_t_param = None
        
        # Goal from host
        self.goal_type = None  # 'navigate', 'approach', or 'push'
        self.target_t_param = None
        self.target_position = None  # Computed from t_param (actual target on boundary)
        self.apf_target_position = None  # APF target (offset from boundary, no contact)
        
        # Approach controller parameters
        self.approach_kp = 0.5  # P controller gain for approach phase
        self.approach_max_speed = 0.1  # Max speed during approach (slow)
        self.contact_target_force = 0.5  # Target contact force (N)
        
        # Optional desired object motion for drive_desired mode (InstantVelocityMatcher)
        self.desired_object_velocity = None        # np.ndarray[2], world frame
        self.desired_object_angular_velocity = 0.0 # float, rad/s
        
        # Optional desired wrench for WrenchTrackingController
        self.desired_wrench = None  # np.ndarray[3] [Fx, Fy, tau]
        
        # State
        self.in_contact = False
        self.contact_force = 0.0
    
    def set_goal(self, goal_type: str, t_param: Optional[float] = None):
        """Set goal from SwarmHost.
        
        Parameters
        ----------
        goal_type : str
            'navigate' for APF navigation (REACHING),
            'approach' for slow approach with P controller (APPROACHING),
            or 'push' to maintain contact.
        t_param : float, optional
            Target t_param for navigation, approach, or pushing.
        """
        self.goal_type = goal_type
        self.target_t_param = t_param
        
        if goal_type == 'push' and t_param is not None:
            # Initialize/update pushing controller
            if self.pushing_type == 'velocity':
                self.pushing_controller = InstantVelocityMatcher(
                    self.generic_object, t_param
                )
                # If a desired object motion is specified, switch to drive_desired mode
                if self.desired_object_velocity is not None:
                    self.pushing_controller.set_mode(
                        mode='drive_desired',
                        desired_object_velocity=self.desired_object_velocity,
                        desired_object_angular_velocity=self.desired_object_angular_velocity,
                    )
            elif self.pushing_type == 'wrench':
                # Initialize wrench controller, passing shared distributor if any
                self.pushing_controller = WrenchTrackingController(
                    self.generic_object, t_param,
                    desired_wrench=self.desired_wrench,
                    force_distributor=self.force_distributor,
                )
            self.current_t_param = t_param

    def set_desired_wrench(self, wrench: np.ndarray):
        """Set desired wrench for wrench-based pushing controller.
        
        This can be called before or after the pushing controller is created.
        """
        self.desired_wrench = np.array(wrench, dtype=float)
        # If wrench controller already exists, propagate immediately
        if (
            self.pushing_type == 'wrench'
            and self.pushing_controller is not None
            and isinstance(self.pushing_controller, WrenchTrackingController)
        ):
            self.pushing_controller.set_desired_wrench(self.desired_wrench)
    
    def update_contact_state(self):
        """Update contact state from robot sensors."""
        if hasattr(self.robot, 'get_contact_force'):
            try:
                force = self.robot.get_contact_force([self.object_uid], max_contacts=4)
                self.contact_force = np.linalg.norm(force[:2])
                self.in_contact = self.contact_force > 0.5  # Threshold
            except (AssertionError, Exception):
                self.contact_force = 0.0
                self.in_contact = False
        else:
            self.contact_force = 0.0
            self.in_contact = False
    
    def compute_target_position(
        self, 
        object_position: np.ndarray, 
        object_orientation: float,
        with_offset: bool = False,
        offset_distance: float = 0.35  # 35cm - must be >= repulsion_distance (30cm) to avoid conflicts
    ) -> Optional[np.ndarray]:
        """Compute target position from t_param.
        
        Parameters
        ----------
        object_position : np.ndarray
            Object center position (x, y).
        object_orientation : float
            Object orientation (radians).
        with_offset : bool
            If True, compute APF target with offset (no contact).
            If False, compute actual target on boundary.
        offset_distance : float
            Distance to offset from boundary for APF target (meters).
        
        Returns
        -------
        np.ndarray or None
            Target position (x, y).
        """
        if self.target_t_param is None:
            return None
        
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(self.generic_object)
        result = param.parameter_to_point(self.target_t_param)
        local_point = result[0]
        local_point = np.array([local_point[0], local_point[1]], dtype=float)
        
        # Transform to world frame
        c, s = np.cos(object_orientation), np.sin(object_orientation)
        R = np.array([[c, -s], [s, c]])
        boundary_pos = R @ local_point + object_position
        
        if with_offset:
            # Get normal vector from parameterization (perpendicular to boundary at contact point)
            normal_local = param.get_normal_vector(self.target_t_param, outward=True)
            
            # Transform normal to world frame
            normal_world = R @ normal_local
            
            # Offset outward from boundary along the true normal direction
            apf_target = boundary_pos + normal_world * offset_distance
            self.apf_target_position = apf_target
            return apf_target
        else:
            self.target_position = boundary_pos
            return boundary_pos
    
    def compute_velocity(
        self,
        object_state: Dict,
        other_robot_positions: list,
        obstacles: Optional[list] = None,
    ) -> np.ndarray:
        """Compute velocity command based on current goal.
        
        Parameters
        ----------
        object_state : dict
            Object state with 'position', 'orientation', 'velocity', 'angular_velocity'.
        other_robot_positions : list
            Positions of other robots for collision avoidance.
        obstacles : list, optional
            Static obstacles.
        
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy, omega).
        """
        pos, heading, _ = self.robot.get_state()
        object_pos = np.array(object_state.get('position', [0, 0]))
        object_orientation = object_state.get('orientation', 0.0)
        
        # print(f"goal_type: {self.goal_type} and target_t_param: {self.target_t_param}")

        if self.goal_type == 'navigate' and self.target_t_param is not None:
            # REACHING phase: APF navigation to safe distance (no contact)
            # Use APF target with offset from boundary
            apf_target = self.compute_target_position(object_pos, object_orientation, with_offset=True)
            if apf_target is None:
                return np.zeros(3)
            
            # Use APF, ORCA, or potential field navigation
            if self.navigation_type == 'apf':
                vel_2d = self.navigator.compute_velocity(
                    pos,
                    apf_target,
                    other_robot_positions,
                    obstacles,
                    robot_id=self.robot_id,  # Pass stable robot_id
                    object_position=object_pos,  # Pass object position for t_param calculations
                    object_orientation=object_orientation,  # Pass object orientation for t_param calculations
                    target_t_param=self.target_t_param,  # Pass target_t_param directly
                )
            elif self.navigation_type == 'orca':
                vel_2d = self.navigator.compute_velocity(
                    pos,
                    apf_target,
                    other_robot_positions,
                    obstacles,
                )
            else:
                vel_2d = self.navigator.compute_velocity(
                    pos,
                    apf_target,
                    other_robot_positions,
                    max_speed=0.3,
                )
            

            print(f"vel_2d: {vel_2d} and robot position: {pos} and target position: {apf_target}")
            # Add heading control
            target_heading = np.arctan2(
                apf_target[1] - pos[1],
                apf_target[0] - pos[0]
            )
            heading_error = self._normalize_angle(target_heading - heading)
            omega = 2.0 * heading_error
            
            return np.array([vel_2d[0], vel_2d[1], omega])
        
        elif self.goal_type == 'approach' and self.target_t_param is not None:
            # APPROACHING phase: Two-step approach
            # Step 1: Rotate until heading is aligned
            # Step 2: Move forward slowly until contact detected
            
            # Use actual target on boundary
            target_pos = self.compute_target_position(object_pos, object_orientation, with_offset=False)
            if target_pos is None:
                return np.zeros(3)
            
            # Check if already in contact - if so, stop moving
            self.update_contact_state()
            if self.in_contact:
                # Already have contact - stop
                return np.zeros(3)
            
            # Compute target heading
            target_heading = np.arctan2(
                target_pos[1] - pos[1],
                target_pos[0] - pos[0]
            )
            heading_error = self._normalize_angle(target_heading - heading)
            
            # Step 1: Rotate first if heading error is significant
            heading_threshold = 0.05  # radians (~6 degrees)
            if abs(heading_error) > heading_threshold:
                # Rotate in place - no translation
                omega = 2.0 * heading_error
                # Limit angular velocity
                omega = np.clip(omega, -1.0, 1.0)
                return np.array([0.0, 0.0, omega])
            
            # Step 2: Heading is aligned, move forward slowly
            direction = target_pos - pos[:2]
            distance = np.linalg.norm(direction)
            
            # Account for robot radius: we want robot edge to touch boundary, not center
            robot_radius = 0.06  # Robot radius
            effective_distance = distance - robot_radius
            # print(f"distance: {distance} from target: {target_pos} to pos: {pos} and effective distance: {effective_distance} ")

            # if effective_distance > 0.01:
            if True:
                direction = direction / distance
                # Use effective distance for P controller
                # Limit speed - slow approach
                speed = min(self.approach_kp * effective_distance, self.approach_max_speed)
                # Clamp speed to zero when very close to the target (e.g., within 1 cm)
                if effective_distance < 0.01:
                    speed = 0.02
                vel_2d = direction * speed
            else:
                vel_2d = np.zeros(2)
            
            # Small heading correction while moving
            omega = 10.0 * heading_error
            
            return np.array([vel_2d[0], vel_2d[1], omega])
        
        elif self.goal_type == 'push' and self.pushing_controller is not None:
            # Pushing mode: use contact controller
            object_velocity = np.array(object_state.get('velocity', [0, 0]))
            object_angular_velocity = object_state.get('angular_velocity', 0.0)
            
            return self.pushing_controller.compute_robot_velocity(
                pos, heading,
                object_pos, object_orientation,
                object_velocity, object_angular_velocity
            )
        
        else:
            # No goal or unknown goal type
            return np.zeros(3)
    
    def get_distance_to_target(self, object_position: np.ndarray, object_orientation: float) -> float:
        """Get distance to target position.
        
        Parameters
        ----------
        object_position : np.ndarray
            Object center position.
        object_orientation : float
            Object orientation.
        
        Returns
        -------
        float
            Distance to target (meters).
        """
        if self.target_t_param is None:
            return float('inf')
        
        target_pos = self.compute_target_position(object_position, object_orientation)
        if target_pos is None:
            return float('inf')
        
        bumper_pos = self.robot.get_contact_position()[:2] if hasattr(self.robot, 'get_contact_position') else self.robot.get_state()[0]
        return np.linalg.norm(bumper_pos - target_pos)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

