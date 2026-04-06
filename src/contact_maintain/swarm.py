"""Swarm Coordination Module for Multi-Robot Contact Maintenance.

This module provides decentralized coordination where SwarmHost only tracks
state and coordinates goals. Each robot (RobotAgent) computes its own
navigation and pushing velocities.

State Machine:
    IDLE -> REACHING -> WAITING -> PUSHING -> CHANGING -> REACHING ...

Author: Contact Maintain Team
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class RobotState(Enum):
    """Individual robot states in the swarm."""
    IDLE = auto()       # Spawned, waiting for target assignment
    REACHING = auto()   # APF navigation to safe distance from target (no contact)
    APPROACHING = auto() # Simple P controller to slowly approach object (small contact)
    WAITING = auto()    # At target, waiting for others
    PUSHING = auto()    # In contact, coordinated pushing active
    CHANGING = auto()   # Transitioning to new configuration


class SwarmState(Enum):
    """Global swarm states."""
    IDLE = auto()       # All robots idle, no targets assigned
    REACHING = auto()   # At least one robot moving to target
    WAITING = auto()    # All at targets, waiting for contact
    PUSHING = auto()    # All in contact, pushing enabled
    CHANGING = auto()   # Reconfiguration in progress


@dataclass
class RobotStatus:
    """Status information for a single robot."""
    name: str
    state: RobotState
    t_param: float
    target_t_param: Optional[float] = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    target_position: Optional[np.ndarray] = None
    distance_to_target: float = float('inf')
    in_contact: bool = False
    contact_force: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/display."""
        return {
            'name': self.name,
            'state': self.state.name,
            't_param': self.t_param,
            'target_t_param': self.target_t_param,
            'position': self.position.tolist() if isinstance(self.position, np.ndarray) else self.position,
            'distance_to_target': self.distance_to_target,
            'in_contact': self.in_contact,
            'contact_force': self.contact_force,
        }


@dataclass
class SwarmEvent:
    """Event logged during swarm operation."""
    time: float
    event_type: str  # 'state_change', 'contact', 'collision_avoided', etc.
    robot_name: Optional[str]
    old_state: Optional[str]
    new_state: Optional[str]
    message: str
    
    def __str__(self):
        if self.robot_name:
            return f"[t={self.time:.2f}s] {self.robot_name}: {self.message}"
        else:
            return f"[t={self.time:.2f}s] SWARM: {self.message}"


# ============================================================================
# SWARM HOST
# ============================================================================

class SwarmHost:
    """Decentralized coordinator for multi-robot swarm.
    
    The SwarmHost only tracks state and coordinates goals. Each robot
    (RobotAgent) computes its own navigation and pushing velocities.
    
    Parameters
    ----------
    robot_agents : dict
        Dictionary mapping robot names to RobotAgent instances.
    object_uid : int
        PyBullet UID of the object being pushed.
    generic_object : GenericObject
        Object model for boundary parameterization.
    position_threshold : float
        Distance threshold to consider robot "at target" (meters).
    contact_force_threshold : float
        Force threshold to consider robot "in contact" (Newtons).
    """
    
    def __init__(
        self,
        robot_agents: Dict[str, Any],  # RobotAgent instances
        object_uid: int,
        generic_object,
        position_threshold: float = 0.05,  # 5cm for small robots
        contact_force_threshold: float = 0.5,  # 0.5N
        startup_mode: str = "quick",
    ):
        self.robot_agents = robot_agents
        self.object_uid = object_uid
        self.generic_object = generic_object
        self.position_threshold = position_threshold
        self.contact_force_threshold = contact_force_threshold
        if startup_mode not in ("quick", "full"):
            raise ValueError(
                f"Invalid startup_mode='{startup_mode}'. Expected 'quick' or 'full'."
            )
        self.startup_mode = startup_mode
        
        # Debug flag
        self.debug = False
        
        # State tracking
        self.swarm_state = SwarmState.IDLE
        self.robot_states: Dict[str, RobotState] = {}
        self.robot_statuses: Dict[str, RobotStatus] = {}
        self.target_t_params: Dict[str, Optional[float]] = {}
        
        # Event log
        self.events: List[SwarmEvent] = []
        self.current_time = 0.0
        
        # Initialize robot states
        for name in robot_agents.keys():
            self.robot_states[name] = RobotState.IDLE
            self.target_t_params[name] = None
            self.robot_statuses[name] = RobotStatus(
                name=name,
                state=RobotState.IDLE,
                t_param=0.0,
            )
        
        # Object state (updated each step)
        self.object_position = np.zeros(2)
        self.object_orientation = 0.0
        self.object_velocity = np.zeros(2)
        self.object_angular_velocity = 0.0
    
    def assign_targets(self, target_t_params: Dict[str, float]):
        """Assign target t_params and start startup sequence.
        
        Parameters
        ----------
        target_t_params : dict
            Mapping from robot name to target t_param (0-1).
        """
        startup_state = RobotState.APPROACHING if self.startup_mode == "quick" else RobotState.REACHING
        startup_goal = 'approach' if self.startup_mode == "quick" else 'navigate'
        startup_swarm_state = SwarmState.WAITING if self.startup_mode == "quick" else SwarmState.REACHING
        startup_state_name = "APPROACHING (quick)" if self.startup_mode == "quick" else "REACHING"

        self._log_event(
            None,
            'swarm_transition',
            self.swarm_state.name,
            startup_swarm_state.name,
            f"SWARM STATE: {self.swarm_state.name} → {startup_state_name}",
        )
        
        for name, t_param in target_t_params.items():
            if name not in self.robot_agents:
                continue
            
            old_state = self.robot_states[name]
            self.target_t_params[name] = t_param
            self.robot_states[name] = startup_state
            
            # Tell robot agent to start with selected startup strategy
            agent = self.robot_agents[name]
            agent.set_goal(startup_goal, t_param)
            
            self._log_event(name, 'state_change',
                           old_state.name, startup_state.name,
                           f"{old_state.name} → {startup_state.name} (target t={t_param:.2f})")
        
        self.swarm_state = startup_swarm_state
    
    def reconfigure(self, new_t_params: Dict[str, float]):
        """Trigger reconfiguration to new t_param assignments.
        
        All robots will transition to CHANGING state, then navigate
        to their new positions using ORCA collision avoidance.
        
        Parameters
        ----------
        new_t_params : dict
            New target t_params for each robot.
        """
        self._log_event(None, 'reconfigure',
                       self.swarm_state.name, 'CHANGING',
                       "RECONFIGURATION TRIGGERED")
        self._log_event(None, 'swarm_transition',
                       self.swarm_state.name, 'CHANGING',
                       f"SWARM STATE: {self.swarm_state.name} → CHANGING")
        
        for name, new_t in new_t_params.items():
            if name not in self.robot_agents:
                continue
            
            old_state = self.robot_states[name]
            self.target_t_params[name] = new_t
            self.robot_states[name] = RobotState.CHANGING
            
            # Tell robot agent to navigate to new t_param
            agent = self.robot_agents[name]
            agent.set_goal('navigate', new_t)
            
            self._log_event(name, 'state_change',
                           old_state.name, 'CHANGING',
                           f"{old_state.name} → CHANGING (new target t={new_t:.2f})")
        
        self.swarm_state = SwarmState.CHANGING
    
    def update(self, dt: float, object_state: Dict = None):
        """Update swarm state and coordinate goals (no velocity computation).
        
        Parameters
        ----------
        dt : float
            Time step.
        object_state : dict, optional
            Object state with keys: 'position', 'orientation', 'velocity', 'angular_velocity'.
        """
        self.current_time += dt
        
        # Update object state
        if object_state:
            self.object_position = np.array(object_state.get('position', [0, 0]))
            self.object_orientation = object_state.get('orientation', 0.0)
            self.object_velocity = np.array(object_state.get('velocity', [0, 0]))
            self.object_angular_velocity = object_state.get('angular_velocity', 0.0)
        
        # Update robot statuses (from agents)
        self._update_robot_statuses()
        
        # Check state transitions
        self._check_transitions()
        
        # Update goals for robots based on swarm state
        self._update_robot_goals()
    
    def _update_robot_statuses(self):
        """Update status for each robot from their agents."""
        for name, agent in self.robot_agents.items():
            # Get robot state from agent's robot
            robot = agent.robot
            pos, heading, vel = robot.get_state()
            
            # Update contact state in agent
            agent.update_contact_state()
            
            # Get target t_param
            target_t = self.target_t_params.get(name)
            
            # Compute distance to target
            distance_to_target = float('inf')
            if target_t is not None:
                distance_to_target = agent.get_distance_to_target(
                    self.object_position, self.object_orientation
                )
            
            # Update status
            self.robot_statuses[name] = RobotStatus(
                name=name,
                state=self.robot_states[name],
                t_param=agent.current_t_param if agent.current_t_param is not None else 0.0,
                target_t_param=target_t,
                position=pos,
                heading=heading,
                target_position=agent.target_position,
                distance_to_target=distance_to_target,
                in_contact=agent.in_contact,
                contact_force=agent.contact_force,
            )
    
    def _check_transitions(self):
        """Check and execute state transitions."""
        # Count robots in each state
        state_counts = {s: 0 for s in RobotState}
        for state in self.robot_states.values():
            state_counts[state] += 1
        
        n_robots = len(self.robot_agents)
        
        # Individual robot transitions
        for name in list(self.robot_states.keys()):
            status = self.robot_statuses[name]
            current_state = self.robot_states[name]
            
            if current_state == RobotState.REACHING:
                # Check if at APF target position (offset from boundary)
                # Must NOT be in contact during REACHING phase
                agent = self.robot_agents[name]
                apf_target = agent.compute_target_position(
                    self.object_position, 
                    self.object_orientation, 
                    with_offset=True
                )
                if apf_target is not None:
                    pos = status.position
                    dist_to_apf_target = np.linalg.norm(pos - apf_target)
                    at_apf_target = dist_to_apf_target < self.position_threshold
                    
                    # Transition to APPROACHING only if at APF target AND no contact
                    if at_apf_target and not status.in_contact:
                        self._transition_robot(name, RobotState.APPROACHING)
            
            elif current_state == RobotState.APPROACHING:
                # Check if small positive contact force is achieved
                # Target: small contact force (0.1N to 2.0N) - no pushing
                has_small_contact = (status.in_contact and 
                                   0.1 <= status.contact_force)
                
                if has_small_contact:
                    self._transition_robot(name, RobotState.WAITING)
                    self._log_event(name, 'contact',
                                   None, None,
                                   f"small contact established (force={status.contact_force:.2f}N)")
            
            elif current_state == RobotState.WAITING:
                # Only log first contact event per robot (track via _logged_contact dict)
                if not hasattr(self, '_logged_contact'):
                    self._logged_contact = {}
                if status.in_contact and name not in self._logged_contact:
                    self._log_event(name, 'contact',
                                   None, None,
                                   f"contact established (force={status.contact_force:.1f}N)")
                    self._logged_contact[name] = True
            
            elif current_state == RobotState.CHANGING:
                # CHANGING uses same logic as REACHING: APF then approach
                # Check if at APF target position (offset from boundary)
                # Must NOT be in contact during CHANGING phase
                agent = self.robot_agents[name]
                apf_target = agent.compute_target_position(
                    self.object_position, 
                    self.object_orientation, 
                    with_offset=True
                )
                if apf_target is not None:
                    pos = status.position
                    dist_to_apf_target = np.linalg.norm(pos - apf_target)
                    at_apf_target = dist_to_apf_target < self.position_threshold
                    
                    # Transition to APPROACHING only if at APF target AND no contact
                    if at_apf_target and not status.in_contact:
                        self._transition_robot(name, RobotState.APPROACHING)
                        # Reset contact logged for this robot (new position)
                        if hasattr(self, '_logged_contact') and name in self._logged_contact:
                            del self._logged_contact[name]
        
        # Update state counts after individual transitions
        state_counts = {s: 0 for s in RobotState}
        for state in self.robot_states.values():
            state_counts[state] += 1
        
        # Swarm state transitions
        if self.swarm_state == SwarmState.REACHING:
            # All robots reached their targets (through APPROACHING phase)?
            # Check if all are in WAITING (which means they completed REACHING -> APPROACHING -> WAITING)
            if state_counts[RobotState.WAITING] == n_robots:
                self._log_event(None, 'swarm_transition',
                               'REACHING', 'WAITING',
                               "SWARM STATE: REACHING → WAITING (all at targets with contact)")
                self.swarm_state = SwarmState.WAITING
        
        elif self.swarm_state == SwarmState.CHANGING:
            # All robots at new positions and waiting?
            if state_counts[RobotState.WAITING] == n_robots:
                self._log_event(None, 'swarm_transition',
                               'CHANGING', 'WAITING',
                               "SWARM STATE: CHANGING → WAITING (reconfigured)")
                self.swarm_state = SwarmState.WAITING
        
        elif self.swarm_state == SwarmState.WAITING:
            # All robots at target AND in contact (force > threshold and ~0)?
            all_at_target = all(
                self.robot_statuses[name].distance_to_target < self.position_threshold
                for name in self.robot_agents.keys()
            )
            all_in_contact = all(
                self.robot_statuses[name].in_contact 
                for name in self.robot_agents.keys()
            )
            # Check contact forces are reasonable (not too high, not zero)
            all_contact_ok = all(
                0.1 < self.robot_statuses[name].contact_force < 50.0
                for name in self.robot_agents.keys()
                if self.robot_statuses[name].in_contact
            )
            
            if all_at_target and all_in_contact and all_contact_ok:
                self._log_event(None, 'swarm_transition',
                               'WAITING', 'PUSHING',
                               "SWARM STATE: WAITING → PUSHING (all at target and in contact!)")
                self.swarm_state = SwarmState.PUSHING
                for name in self.robot_states:
                    self.robot_states[name] = RobotState.PUSHING
        
        elif self.swarm_state == SwarmState.CHANGING:
            # All robots at new positions and waiting (through APPROACHING phase)?
            if state_counts[RobotState.WAITING] == n_robots:
                self._log_event(None, 'swarm_transition',
                               'CHANGING', 'WAITING',
                               "SWARM STATE: CHANGING → WAITING (reconfigured with contact)")
                self.swarm_state = SwarmState.WAITING
    
    def _transition_robot(self, name: str, new_state: RobotState):
        """Transition a robot to a new state."""
        old_state = self.robot_states[name]
        self.robot_states[name] = new_state
        self._log_event(name, 'state_change',
                       old_state.name, new_state.name,
                       f"{old_state.name} → {new_state.name}")
    
    def _update_robot_goals(self):
        """Update goals for robot agents based on swarm state."""
        object_state = {
            'position': self.object_position,
            'orientation': self.object_orientation,
            'velocity': self.object_velocity,
            'angular_velocity': self.object_angular_velocity,
        }
        
        for name, agent in self.robot_agents.items():
            state = self.robot_states[name]
            target_t = self.target_t_params.get(name)
            
            if state == RobotState.PUSHING and target_t is not None:
                # Switch to pushing mode
                if agent.goal_type != 'push':
                    agent.set_goal('push', target_t)
            elif state == RobotState.APPROACHING and target_t is not None:
                # Switch to approach mode (P controller, slow approach)
                if agent.goal_type != 'approach':
                    agent.set_goal('approach', target_t)
            elif state in (RobotState.REACHING, RobotState.CHANGING) and target_t is not None:
                # Keep navigation goal (APF navigation to offset target)
                if agent.goal_type != 'navigate' or agent.target_t_param != target_t:
                    agent.set_goal('navigate', target_t)
            # WAITING state: keep current goal (no change needed)
    
    def _log_event(self, robot_name: Optional[str], event_type: str,
                   old_state: Optional[str], new_state: Optional[str],
                   message: str):
        """Log an event."""
        event = SwarmEvent(
            time=self.current_time,
            event_type=event_type,
            robot_name=robot_name,
            old_state=old_state,
            new_state=new_state,
            message=message,
        )
        self.events.append(event)
        print(event)  # Print to console
    
    def get_status(self) -> Dict[str, str]:
        """Get current state for all robots.
        
        Returns
        -------
        dict
            Mapping from robot name to state name string.
        """
        return {name: state.name for name, state in self.robot_states.items()}
    
    def get_detailed_status(self) -> Dict[str, dict]:
        """Get detailed status for all robots.
        
        Returns
        -------
        dict
            Mapping from robot name to status dict.
        """
        return {name: status.to_dict() for name, status in self.robot_statuses.items()}
    
    def get_metrics(self) -> Dict:
        """Get swarm metrics.
        
        Returns
        -------
        dict
            Metrics including times, contact counts, etc.
        """
        # Find key time points from events
        first_reaching_time = None
        first_waiting_time = None
        first_pushing_time = None
        
        for event in self.events:
            if event.event_type == 'swarm_transition':
                if 'REACHING' in event.new_state and first_reaching_time is None:
                    first_reaching_time = event.time
                elif 'WAITING' in event.new_state and first_waiting_time is None:
                    first_waiting_time = event.time
                elif 'PUSHING' in event.new_state and first_pushing_time is None:
                    first_pushing_time = event.time
        
        # Count events
        collision_avoidance_count = sum(
            1 for e in self.events if e.event_type == 'collision_avoided'
        )
        contact_count = sum(
            1 for e in self.events if e.event_type == 'contact'
        )
        
        # Time to push
        time_to_push = None
        if first_reaching_time is not None and first_pushing_time is not None:
            time_to_push = first_pushing_time - first_reaching_time
        
        return {
            'swarm_state': self.swarm_state.name,
            'num_robots': len(self.robot_agents),
            'time_to_first_push': time_to_push,
            'collision_avoidance_events': collision_avoidance_count,
            'contact_events': contact_count,
            'total_events': len(self.events),
            'robots_in_contact': sum(
                1 for s in self.robot_statuses.values() if s.in_contact
            ),
        }
    
    def print_events(self):
        """Print all logged events."""
        print("\n" + "=" * 60)
        print("  SWARM EVENT LOG")
        print("=" * 60)
        for event in self.events:
            print(event)
        print("=" * 60)

