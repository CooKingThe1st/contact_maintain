"""Overworld Simulator for Distributed Swarm Coordination.

The OverworldSimulator handles communication between robots and coordinates
all DistributedMonitor instances. It simulates a fully connected network
where all robots can communicate with each other.

Author: Contact Maintain Team
"""
from typing import Dict, List, Optional, Any
import numpy as np

from contact_maintain.distributed_monitor import DistributedMonitor
from contact_maintain.robot_message import RobotMessage
from contact_maintain.reconfiguration import create_reconfiguration_planner


class OverworldSimulator:
    """Overworld Simulator for distributed swarm coordination.
    
    Responsibilities:
    - Create and manage all DistributedMonitor instances
    - Broadcast messages between robots (simulate fully connected network)
    - Handle reconfiguration triggers
    - Provide unified interface for test scripts
    
    Parameters
    ----------
    robots : Dict[str, object]
        Dictionary mapping robot names to robot instances
    object_uid : int
        PyBullet UID of the object
    generic_object : GenericObject
        Object model for boundary parameterization
    navigation_scheme : str
        Navigation scheme: 'apf', 'static_single', or 'divide_conquer'
    push_controller_type : str
        Push controller type: 'phase7' or other
    """
    
    def __init__(
        self,
        robots: Dict[str, Any],
        object_uid: int,
        generic_object: Any,
        navigation_scheme: str = 'apf',
        push_controller_type: str = 'phase7',
        navigation_only: bool = False,
        startup_mode: str = 'quick',
    ):
        self.robots = robots
        self.navigation_only = navigation_only
        self.object_uid = object_uid
        self.generic_object = generic_object
        self.navigation_scheme = navigation_scheme
        self.push_controller_type = push_controller_type
        self.startup_mode = startup_mode
        # Navigation-only + quick approach: latch all-zero commands once every robot
        # has sensed contact (avoids residual motion when force flickers below epsilon).
        self._nav_quick_all_contact_latched = False
        
        # Create monitors for each robot
        self.monitors: Dict[str, DistributedMonitor] = {}
        for name, robot in robots.items():
            self.monitors[name] = DistributedMonitor(
                robot_name=name,
                robot=robot,
                object_uid=object_uid,
                generic_object=generic_object,
                navigation_scheme=navigation_scheme,
                push_controller_type=push_controller_type,
                navigation_only=navigation_only,
                startup_mode=startup_mode,
            )
        
        # Reconfiguration planner
        self.reconfiguration_planner = create_reconfiguration_planner(navigation_scheme)
        
        # Current time
        self.current_time = 0.0
    
    def update(self, dt: float, object_state: Dict):
        """Main update loop.
        
        Parameters
        ----------
        dt : float
            Time step
        object_state : dict
            Object state with 'position', 'orientation', 'velocity', 'angular_velocity'
        """
        self.current_time += dt
        
        # Collect messages from all monitors
        messages: Dict[str, RobotMessage] = {}
        for name, monitor in self.monitors.items():
            messages[name] = monitor.get_message()
        
        # Broadcast messages to all monitors (fully connected network)
        for name, monitor in self.monitors.items():
            # Send all other robots' messages to this monitor
            other_messages = [
                msg for other_name, msg in messages.items()
                if other_name != name
            ]
            
            # Update monitor
            monitor.update(dt, object_state, other_messages)
    
    def assign_targets(self, target_t_params: Dict[str, float]):
        """Assign targets to all robots.
        
        Parameters
        ----------
        target_t_params : Dict[str, float]
            Mapping from robot name to target t_param (0-1)
        """
        for name, t_param in target_t_params.items():
            if name in self.monitors:
                self.monitors[name].set_target(t_param)
        self._nav_quick_all_contact_latched = False
    
    def reconfigure(self, new_t_params: Dict[str, float]):
        """Trigger reconfiguration to new target positions.
        
        Parameters
        ----------
        new_t_params : Dict[str, float]
            New target t_params for each robot
        """
        # Use reconfiguration planner to optimize assignment
        optimized_assignment = self.reconfiguration_planner.plan(
            new_t_params, self.monitors, self.generic_object
        )
        
        # Assign new targets
        self.assign_targets(optimized_assignment)
    
    def compute_velocities(self, object_state: Dict) -> Dict[str, np.ndarray]:
        """Compute velocity commands for all robots.
        
        Parameters
        ----------
        object_state : dict
            Object state with 'position', 'orientation', 'velocity', 'angular_velocity'
            
        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from robot name to velocity command (vx, vy, omega)
        """
        if (
            self.navigation_only
            and self.startup_mode == 'quick'
            and self.monitors
            and not self._nav_quick_all_contact_latched
        ):
            if all(
                m.local_state.contact_force > m.approach_stop_force_epsilon
                for m in self.monitors.values()
            ):
                self._nav_quick_all_contact_latched = True

        if self.navigation_only and self.startup_mode == 'quick' and self._nav_quick_all_contact_latched:
            return {name: np.zeros(3, dtype=float) for name in self.monitors}

        velocities = {}
        for name, monitor in self.monitors.items():
            velocities[name] = monitor.compute_velocity(object_state)
        return velocities
    
    def set_desired_object_motion(
        self,
        desired_velocity: np.ndarray,
        desired_angular_velocity: float,
    ):
        """Set desired object motion for all push controllers.
        
        Parameters
        ----------
        desired_velocity : np.ndarray
            Desired object linear velocity (vx, vy)
        desired_angular_velocity : float
            Desired object angular velocity (rad/s)
        """
        for monitor in self.monitors.values():
            monitor.set_desired_object_motion(desired_velocity, desired_angular_velocity)
    
    def get_status(self) -> Dict[str, Dict]:
        """Get current status for all robots.
        
        Returns
        -------
        Dict[str, Dict]
            Mapping from robot name to status dictionary
        """
        status = {}
        for name, monitor in self.monitors.items():
            status[name] = {
                'state': monitor.state.value,
                'target_t_param': monitor.target_t_param,
                'in_contact': monitor.local_state.in_contact,
                'contact_force': monitor.local_state.contact_force,
                'distance_to_target': monitor.local_state.distance_to_target,
            }
        return status
    
    def get_all_in_pushing(self) -> bool:
        """Check if all robots are in PUSHING state.
        
        Returns
        -------
        bool
            True if all robots are in PUSHING state
        """
        from contact_maintain.robot_message import MonitorState
        return all(
            monitor.state == MonitorState.PUSHING
            for monitor in self.monitors.values()
        )
    
    def get_all_at_target(self) -> bool:
        """Check if all robots are at their targets.
        
        Returns
        -------
        bool
            True if all robots are at their targets
        """
        return all(
            monitor.local_state.distance_to_target < monitor.position_threshold
            for monitor in self.monitors.values()
            if monitor.target_t_param is not None
        )
