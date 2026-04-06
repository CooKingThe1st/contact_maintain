"""Robot Message Protocol for Distributed Swarm Communication.

This module defines the message format used for communication between
robots in the distributed swarm architecture. Messages are broadcast
via the OverworldSimulator to simulate a fully connected network.

Author: Contact Maintain Team
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


class MonitorState(Enum):
    """Monitor state for distributed coordination."""
    NAVIGATING = "NAVIGATING"  # Robot is moving to target position
    PUSHING = "PUSHING"        # Robot is in contact and pushing


@dataclass
class RobotMessage:
    """Standardized message format for robot-to-robot communication.
    
    This message contains all information a robot needs to share with
    others for distributed coordination. Messages are broadcast via
    the OverworldSimulator.
    
    Attributes
    ----------
    robot_name : str
        Unique identifier for the robot
    position : np.ndarray
        Current robot position (x, y) in world frame
    state : MonitorState
        Current monitor state (NAVIGATING or PUSHING)
    target_t_param : Optional[float]
        Target t_param on object boundary (0-1), None if not assigned
    in_contact : bool
        Whether robot is currently in contact with object
    contact_force : float
        Current contact force magnitude (Newtons)
    navigation_step : int
        Navigation phase step (0=unknown, 1=to ring, 2=along ring, 3=approach/touch)
    timestamp : float
        Message timestamp (simulation time)
    """
    robot_name: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    state: MonitorState = MonitorState.NAVIGATING
    target_t_param: Optional[float] = None
    in_contact: bool = False
    contact_force: float = 0.0
    navigation_step: int = 0
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert message to dictionary for serialization."""
        return {
            'robot_name': self.robot_name,
            'position': self.position.tolist() if isinstance(self.position, np.ndarray) else self.position,
            'state': self.state.value,
            'target_t_param': self.target_t_param,
            'in_contact': self.in_contact,
            'contact_force': float(self.contact_force),
            'navigation_step': int(self.navigation_step),
            'timestamp': float(self.timestamp),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RobotMessage':
        """Create message from dictionary."""
        return cls(
            robot_name=data['robot_name'],
            position=np.array(data['position']),
            state=MonitorState(data['state']),
            target_t_param=data.get('target_t_param'),
            in_contact=data.get('in_contact', False),
            contact_force=float(data.get('contact_force', 0.0)),
            navigation_step=int(data.get('navigation_step', 0)),
            timestamp=float(data.get('timestamp', 0.0)),
        )
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"RobotMessage({self.robot_name}, state={self.state.value}, "
            f"step={self.navigation_step}, pos={self.position}, "
            f"target_t={self.target_t_param}, contact={self.in_contact}, "
            f"force={self.contact_force:.2f}N)"
        )
