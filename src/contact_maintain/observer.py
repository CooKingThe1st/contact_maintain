"""Contact observation and analysis utilities."""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque


@dataclass
class ContactState:
    """Represents the current contact state."""
    in_contact: bool = False
    contact_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    contact_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_magnitude: float = 0.0
    force_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    def __post_init__(self):
        if self.force_magnitude == 0.0 and np.linalg.norm(self.contact_force[:2]) > 0:
            self.force_magnitude = np.linalg.norm(self.contact_force[:2])
        if np.allclose(self.force_direction, 0) and self.force_magnitude > 0:
            self.force_direction = self.contact_force[:2] / self.force_magnitude


@dataclass
class ContactRecord:
    """A single timestep record of contact data."""
    timestamp: float
    robot_position: np.ndarray
    robot_orientation: float
    robot_velocity: np.ndarray
    contact_position: np.ndarray
    contact_force: np.ndarray
    object_position: np.ndarray
    object_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    in_contact: bool = False
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'robot_position': self.robot_position.tolist(),
            'robot_orientation': self.robot_orientation,
            'robot_velocity': self.robot_velocity.tolist(),
            'contact_position': self.contact_position.tolist(),
            'contact_force': self.contact_force.tolist(),
            'object_position': self.object_position.tolist(),
            'object_velocity': self.object_velocity.tolist(),
            'in_contact': self.in_contact,
        }


class ContactObserver:
    """Observer for tracking and analyzing contact state.
    
    Parameters
    ----------
    force_threshold : float
        Minimum force magnitude to consider as contact.
    history_size : int
        Number of timesteps to keep in history.
    smoothing_window : int
        Window size for force smoothing.
    """
    
    def __init__(self, force_threshold=0.5, history_size=1000, smoothing_window=5):
        self.force_threshold = force_threshold
        self.history_size = history_size
        self.smoothing_window = smoothing_window
        
        # State
        self.current_state = ContactState()
        self.history: Deque[ContactRecord] = deque(maxlen=history_size)
        self.force_buffer: Deque[np.ndarray] = deque(maxlen=smoothing_window)
        
        # Statistics
        self.contact_count = 0
        self.contact_lost_count = 0
        self.total_contact_time = 0.0
        self.last_timestamp = 0.0
        
    def update(self, timestamp, robot_pos, robot_theta, robot_vel, 
               contact_pos, contact_force, object_pos, object_vel=None):
        """Update the observer with new measurements.
        
        Parameters
        ----------
        timestamp : float
            Current simulation time.
        robot_pos : np.ndarray, shape (2,)
            Robot position (x, y).
        robot_theta : float
            Robot orientation.
        robot_vel : np.ndarray, shape (3,)
            Robot velocity (vx, vy, omega).
        contact_pos : np.ndarray, shape (3,)
            Contact point position.
        contact_force : np.ndarray, shape (3,)
            Contact force vector.
        object_pos : np.ndarray, shape (2,) or (3,)
            Object position.
        object_vel : np.ndarray, optional
            Object velocity.
        
        Returns
        -------
        ContactState
            Updated contact state.
        """
        # Ensure arrays
        contact_force = np.array(contact_force)
        object_pos = np.array(object_pos)
        if len(object_pos) == 2:
            object_pos = np.append(object_pos, 0)
        
        if object_vel is None:
            object_vel = np.zeros(3)
        else:
            object_vel = np.array(object_vel)
        
        # Update force buffer for smoothing
        self.force_buffer.append(contact_force)
        smoothed_force = np.mean(self.force_buffer, axis=0)
        
        # Determine contact state
        force_magnitude = np.linalg.norm(smoothed_force[:2])
        was_in_contact = self.current_state.in_contact
        is_in_contact = force_magnitude > self.force_threshold
        
        # Update statistics
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 0
        if is_in_contact:
            self.total_contact_time += dt
        
        if is_in_contact and not was_in_contact:
            self.contact_count += 1
        elif was_in_contact and not is_in_contact:
            self.contact_lost_count += 1
        
        # Update current state
        force_direction = smoothed_force[:2] / force_magnitude if force_magnitude > 0 else np.zeros(2)
        self.current_state = ContactState(
            in_contact=is_in_contact,
            contact_force=smoothed_force,
            contact_position=np.array(contact_pos),
            force_magnitude=force_magnitude,
            force_direction=force_direction,
        )
        
        # Record to history
        record = ContactRecord(
            timestamp=timestamp,
            robot_position=np.array(robot_pos),
            robot_orientation=robot_theta,
            robot_velocity=np.array(robot_vel),
            contact_position=np.array(contact_pos),
            contact_force=smoothed_force.copy(),
            object_position=object_pos.copy(),
            object_velocity=object_vel.copy(),
            in_contact=is_in_contact,
        )
        self.history.append(record)
        self.last_timestamp = timestamp
        
        return self.current_state
    
    def get_contact_ratio(self):
        """Get the ratio of time spent in contact."""
        if self.last_timestamp == 0:
            return 0.0
        return self.total_contact_time / self.last_timestamp
    
    def get_statistics(self):
        """Get summary statistics of the contact history.
        
        Returns
        -------
        dict
            Dictionary containing contact statistics.
        """
        if len(self.history) == 0:
            return {
                'contact_ratio': 0.0,
                'contact_count': 0,
                'contact_lost_count': 0,
                'mean_force': 0.0,
                'max_force': 0.0,
                'std_force': 0.0,
            }
        
        forces = [r.contact_force for r in self.history if r.in_contact]
        force_magnitudes = [np.linalg.norm(f[:2]) for f in forces] if forces else [0]
        
        return {
            'contact_ratio': self.get_contact_ratio(),
            'contact_count': self.contact_count,
            'contact_lost_count': self.contact_lost_count,
            'mean_force': np.mean(force_magnitudes),
            'max_force': np.max(force_magnitudes),
            'std_force': np.std(force_magnitudes),
            'total_contact_time': self.total_contact_time,
            'total_time': self.last_timestamp,
        }
    
    def get_history_arrays(self):
        """Get history as numpy arrays for analysis.
        
        Returns
        -------
        dict
            Dictionary of numpy arrays for each recorded quantity.
        """
        if len(self.history) == 0:
            return {}
        
        return {
            'timestamps': np.array([r.timestamp for r in self.history]),
            'robot_positions': np.array([r.robot_position for r in self.history]),
            'robot_orientations': np.array([r.robot_orientation for r in self.history]),
            'robot_velocities': np.array([r.robot_velocity for r in self.history]),
            'contact_positions': np.array([r.contact_position for r in self.history]),
            'contact_forces': np.array([r.contact_force for r in self.history]),
            'object_positions': np.array([r.object_position for r in self.history]),
            'object_velocities': np.array([r.object_velocity for r in self.history]),
            'in_contact': np.array([r.in_contact for r in self.history]),
        }
    
    def reset(self):
        """Reset the observer state."""
        self.current_state = ContactState()
        self.history.clear()
        self.force_buffer.clear()
        self.contact_count = 0
        self.contact_lost_count = 0
        self.total_contact_time = 0.0
        self.last_timestamp = 0.0


class ContactPointTracker:
    """Tracks the contact point on an object surface.
    
    Estimates where on the object the robot is making contact,
    useful for contact maintenance without direct force sensing.
    
    Parameters
    ----------
    object_radius : float
        Radius of the object (assuming circular).
    """
    
    def __init__(self, object_radius=0.15):
        self.object_radius = object_radius
        self.estimated_contact_angle = 0.0
        self.contact_point_world = np.zeros(2)
    
    def update(self, robot_contact_pos, object_pos):
        """Update estimated contact point.
        
        Parameters
        ----------
        robot_contact_pos : np.ndarray, shape (2,) or (3,)
            Position of robot's contact element.
        object_pos : np.ndarray, shape (2,) or (3,)
            Position of object center.
        
        Returns
        -------
        np.ndarray, shape (2,)
            Estimated contact point on object surface.
        """
        robot_pos_2d = np.array(robot_contact_pos)[:2]
        object_pos_2d = np.array(object_pos)[:2]
        
        # Direction from object to robot contact point
        direction = robot_pos_2d - object_pos_2d
        dist = np.linalg.norm(direction)
        
        if dist > 0:
            direction = direction / dist
            self.estimated_contact_angle = np.arctan2(direction[1], direction[0])
            
            # Contact point is on object surface facing the robot
            self.contact_point_world = object_pos_2d + self.object_radius * direction
        
        return self.contact_point_world
    
    def get_contact_angle(self):
        """Get the angle of the contact point on the object."""
        return self.estimated_contact_angle

