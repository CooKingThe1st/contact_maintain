"""ORCA (Optimal Reciprocal Collision Avoidance) Navigation Module.

This module provides ORCA-based navigation using the RVO2 library for
collision avoidance between robots and static obstacles.

Author: Contact Maintain Team
"""
import numpy as np
from typing import List, Optional, Tuple
import rvo2


class ORCANavigator:
    """ORCA-based navigator for multi-robot collision avoidance.
    
    Uses RVO2 library to compute collision-free velocities for robots
    navigating toward targets while avoiding each other and obstacles.
    
    Parameters
    ----------
    time_step : float
        Simulation time step (seconds).
    neighbor_dist : float
        Maximum distance to consider other robots as neighbors (meters).
    max_neighbors : int
        Maximum number of neighbors to consider.
    time_horizon : float
        Time horizon for collision avoidance (seconds).
    time_horizon_obst : float
        Time horizon for obstacle avoidance (seconds).
    radius : float
        Robot radius (meters).
    max_speed : float
        Maximum robot speed (m/s).
    """
    
    def __init__(
        self,
        time_step: float = 1.0 / 60.0,
        neighbor_dist: float = 2.0,
        max_neighbors: int = 10,
        time_horizon: float = 2.0,
        time_horizon_obst: float = 0.5,
        radius: float = 0.06,
        max_speed: float = 0.5,
    ):
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.radius = radius
        self.max_speed = max_speed
        
        # RVO2 simulator (will be initialized when needed)
        self.sim: Optional[rvo2.PyRVOSimulator] = None
        self.agent_ids: dict = {}  # Map robot name -> agent ID
        self.agent_names: dict = {}  # Map agent ID -> robot name
        
    def initialize(self, robot_positions: dict, obstacles: Optional[List] = None):
        """Initialize RVO2 simulator with robots and obstacles.
        
        Parameters
        ----------
        robot_positions : dict
            Mapping from robot name to (x, y) position.
        obstacles : list, optional
            List of obstacle polygons. Each obstacle is a list of (x, y) vertices.
        """
        # Create simulator
        self.sim = rvo2.PyRVOSimulator(
            self.time_step,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.radius,
            self.max_speed,
        )
        
        # Add robots as agents
        self.agent_ids = {}
        self.agent_names = {}
        for name, pos in robot_positions.items():
            agent_id = self.sim.addAgent(
                (pos[0], pos[1]),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                self.radius,
                self.max_speed,
                (0, 0)  # Initial velocity
            )
            self.agent_ids[name] = agent_id
            self.agent_names[agent_id] = name
        
        # Add obstacles if provided
        if obstacles:
            for obstacle in obstacles:
                # RVO2 expects obstacles as list of (x, y) tuples
                obstacle_vertices = [(v[0], v[1]) for v in obstacle]
                self.sim.addObstacle(obstacle_vertices)
            self.sim.processObstacles()
    
    def update_robot_position(self, name: str, position: Tuple[float, float]):
        """Update a robot's position in the simulator.
        
        Parameters
        ----------
        name : str
            Robot name.
        position : tuple
            (x, y) position.
        """
        if self.sim is None or name not in self.agent_ids:
            return
        
        agent_id = self.agent_ids[name]
        self.sim.setAgentPosition(agent_id, position)
    
    def compute_velocity(
        self,
        robot_name: str,
        target_position: np.ndarray,
        other_robot_positions: dict,
        current_velocity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute collision-free velocity for a robot.
        
        Parameters
        ----------
        robot_name : str
            Name of the robot.
        target_position : np.ndarray
            Target (x, y) position.
        other_robot_positions : dict
            Mapping from other robot names to their (x, y) positions.
        current_velocity : np.ndarray, optional
            Current velocity of the robot (for continuity).
        
        Returns
        -------
        np.ndarray
            Collision-free velocity vector (vx, vy).
        """
        if self.sim is None or robot_name not in self.agent_ids:
            # Fallback: direct velocity toward target
            if current_velocity is not None:
                return current_velocity[:2]
            direction = target_position - np.array([0, 0])
            dist = np.linalg.norm(direction)
            if dist > 0.001:
                direction = direction / dist
                return direction * min(self.max_speed, dist / self.time_step)
            return np.zeros(2)
        
        agent_id = self.agent_ids[robot_name]
        
        # Update all robot positions
        all_positions = {robot_name: target_position[:2]}
        all_positions.update(other_robot_positions)
        
        for name, pos in all_positions.items():
            if name in self.agent_ids:
                pos_tuple = (float(pos[0]), float(pos[1]))
                self.sim.setAgentPosition(self.agent_ids[name], pos_tuple)
        
        # Set preferred velocity toward target
        direction = target_position[:2] - np.array(self.sim.getAgentPosition(agent_id))
        dist = np.linalg.norm(direction)
        if dist > 0.001:
            direction = direction / dist
            pref_vel = direction * min(self.max_speed, dist / self.time_step)
        else:
            pref_vel = (0.0, 0.0)
        
        self.sim.setAgentPrefVelocity(agent_id, (pref_vel[0], pref_vel[1]))
        
        # Step simulation
        self.sim.doStep()
        
        # Get computed velocity
        vel = self.sim.getAgentVelocity(agent_id)
        return np.array([vel[0], vel[1]])
    
    def get_robot_position(self, robot_name: str) -> Optional[Tuple[float, float]]:
        """Get current position of a robot from simulator.
        
        Parameters
        ----------
        robot_name : str
            Robot name.
        
        Returns
        -------
        tuple or None
            (x, y) position if robot exists.
        """
        if self.sim is None or robot_name not in self.agent_ids:
            return None
        
        agent_id = self.agent_ids[robot_name]
        return self.sim.getAgentPosition(agent_id)
    
    def reset(self):
        """Reset the simulator."""
        self.sim = None
        self.agent_ids = {}
        self.agent_names = {}

