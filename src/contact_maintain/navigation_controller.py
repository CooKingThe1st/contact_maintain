"""Navigation Controller Module for Distributed Swarm.

This module provides a base class and implementations for different
navigation schemes. Each scheme is specialized for the pushing problem,
trading generality for simplicity and predictability.

Navigation Schemes:
- APF: Rewritten APF navigation optimized for contact maintenance
- Static Single: Only one robot moves at a time (static environment)
- Divide-n-Conquer: Each robot manages non-overlapping consecutive edges

Author: Contact Maintain Team
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from contact_maintain.robot_message import RobotMessage


class NavigationController(ABC):
    """Base class for navigation controllers.
    
    Each navigation scheme implements this interface to provide
    velocity commands for robots navigating to their target positions.
    """
    
    @abstractmethod
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
        """Compute velocity command for navigation.
        
        Parameters
        ----------
        robot_name : str
            Name of the robot
        robot_position : np.ndarray
            Current robot position (x, y)
        robot_heading : float
            Current robot heading (radians)
        target_t_param : float
            Target t_param on object boundary (0-1)
        object_position : np.ndarray
            Object center position (x, y)
        object_orientation : float
            Object orientation (radians)
        object_state : dict
            Object state with 'velocity', 'angular_velocity', etc.
        other_robot_messages : List[RobotMessage]
            Messages from other robots for coordination
        generic_object : GenericObject
            Object model for boundary parameterization
            
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy, omega)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset controller state."""
        pass


class APFNavigationController(NavigationController):
    """APF Navigation Controller (rewritten for pushing problem).
    
    This is a rewritten version of APF navigation, specifically optimized
    for contact maintenance scenarios. It is less general than the original
    APF but simpler and more predictable.
    
    Features:
    - All robots navigate simultaneously
    - Optimized collision avoidance for contact maintenance
    - Simplified potential field calculations
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
        self.max_speed = max_speed
        self.k_goal = k_goal
        self.k_obstacle = k_obstacle
        self.k_wall = k_wall
        self.repulsion_distance = repulsion_distance
        
        # Import here to avoid circular dependencies
        from contact_maintain.navigation.apf_nav import APFNavigatorPushing
        self.navigator = APFNavigatorPushing(
            radius=radius,
            max_speed=max_speed,
            k_goal=k_goal,
            k_obstacle=k_obstacle,
            k_wall=k_wall,
            repulsion_distance=repulsion_distance,
        )
    
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
        """Compute APF velocity command."""
        # Compute target position from t_param
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(generic_object)
        result = param.parameter_to_point(target_t_param)
        local_point = np.array([result[0][0], result[0][1]], dtype=float)
        
        # Transform to world frame
        c, s = np.cos(object_orientation), np.sin(object_orientation)
        R = np.array([[c, -s], [s, c]])
        boundary_pos = R @ local_point + object_position
        
        # Get normal for offset (APF target is offset from boundary)
        normal_local = param.get_normal_vector(target_t_param, outward=True)
        normal_world = R @ normal_local
        offset_distance = 0.35  # 35cm offset
        target_position = boundary_pos + normal_world * offset_distance
        
        # Get other robot positions
        other_positions = [msg.position for msg in other_robot_messages]
        
        # Get object as obstacle
        obstacles = self._get_object_as_obstacle(generic_object, object_position, object_orientation)
        
        # Compute velocity using navigator
        vel_2d = self.navigator.compute_velocity(
            current_position=robot_position,
            target_position=target_position,
            other_robot_positions=other_positions,
            obstacles=obstacles,
            robot_id=hash(robot_name) % 1000,  # Stable robot ID
            object_position=object_position,
            object_orientation=object_orientation,
            target_t_param=target_t_param,
        )
        
        # Add heading control
        target_heading = np.arctan2(
            target_position[1] - robot_position[1],
            target_position[0] - robot_position[0]
        )
        heading_error = self._normalize_angle(target_heading - robot_heading)
        omega = 2.0 * heading_error
        
        return np.array([vel_2d[0], vel_2d[1], omega])
    
    def _get_object_as_obstacle(self, generic_object, object_position, object_orientation):
        """Convert object boundary to obstacle format."""
        boundary_coords = list(generic_object.geometry.exterior.coords)
        c, s = np.cos(object_orientation), np.sin(object_orientation)
        R = np.array([[c, -s], [s, c]])
        
        world_vertices = []
        for local_vertex in boundary_coords:
            local_2d = np.array([local_vertex[0], local_vertex[1]])
            world_2d = R @ local_2d + object_position
            world_vertices.append((float(world_2d[0]), float(world_2d[1])))
        
        return [world_vertices]
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def reset(self):
        """Reset navigator state."""
        if hasattr(self.navigator, 'reset'):
            self.navigator.reset()


# Factory function for creating navigation controllers
def create_navigation_controller(
    scheme: str,
    **kwargs
) -> NavigationController:
    """Create a navigation controller based on scheme name.
    
    Parameters
    ----------
    scheme : str
        Navigation scheme: 'apf', 'static_single', or 'divide_conquer'
    **kwargs
        Additional arguments passed to controller constructor
        
    Returns
    -------
    NavigationController
        Navigation controller instance
    """
    if scheme == 'apf':
        return APFNavigationController(**kwargs)
    elif scheme == 'static_single':
        from contact_maintain.navigation.static_single_nav import StaticSingleNavigationController
        return StaticSingleNavigationController(**kwargs)
    elif scheme == 'divide_conquer':
        from contact_maintain.navigation.divide_conquer_nav import DivideConquerNavigationController
        return DivideConquerNavigationController(**kwargs)
    else:
        raise ValueError(f"Unknown navigation scheme: {scheme}")
