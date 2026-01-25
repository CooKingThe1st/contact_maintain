#!/usr/bin/env python3
"""
CBS Navigation Test (Centralized)

Tests Conflict-Based Search (CBS) with:
1. Robots in formation (2 side lines, opposite goals)
2. Static obstacles

Usage:
    python test_cbs_navigation.py --test formation
    python test_cbs_navigation.py --test obstacles
    python test_cbs_navigation.py --test all
"""
import argparse
import sys
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import heapq

import numpy as np
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))

from contact_maintain.robot_factory import create_robot


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 60
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

# Grid discretization for path planning
GRID_RESOLUTION = 0.1  # meters per grid cell
ROBOT_RADIUS = 0.06
SAFETY_MARGIN = 0.05
MAX_SPEED = 0.3


@dataclass
class Constraint:
    """Constraint: robot cannot be at (x, y) at time t"""
    robot_id: int
    x: float
    y: float
    time: int


@dataclass
class Conflict:
    """Conflict: two robots at same position at same time"""
    robot_i: int
    robot_j: int
    x: float
    y: float
    time: int


@dataclass
class CBSNode:
    """Node in CBS search tree"""
    constraints: List[Constraint]
    paths: List[List[Tuple[float, float]]]  # path for each robot
    cost: float
    
    def __lt__(self, other):
        return self.cost < other.cost


def world_to_grid(x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates to grid coordinates"""
    return (int(round(x / GRID_RESOLUTION)), int(round(y / GRID_RESOLUTION)))


def grid_to_world(gx: int, gy: int) -> Tuple[float, float]:
    """Convert grid coordinates to world coordinates"""
    return (gx * GRID_RESOLUTION, gy * GRID_RESOLUTION)


def astar_path(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: Set[Tuple[int, int]],
    constraints: List[Constraint],
    robot_id: int,
    max_time: int = 1000
) -> Optional[List[Tuple[float, float]]]:
    """
    A* pathfinding with constraints.
    Returns path as list of (x, y) positions.
    """
    start_grid = world_to_grid(start[0], start[1])
    goal_grid = world_to_grid(goal[0], goal[1])
    
    # Build constraint set for this robot
    constraint_set = {}
    for c in constraints:
        if c.robot_id == robot_id:
            c_grid = world_to_grid(c.x, c.y)
            if c.time not in constraint_set:
                constraint_set[c.time] = set()
            constraint_set[c.time].add(c_grid)
    
    def heuristic(gx, gy):
        return abs(gx - goal_grid[0]) + abs(gy - goal_grid[1])
    
    def get_neighbors(gx, gy):
        """Get 8-connected neighbors"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ngx, ngy = gx + dx, gy + dy
                neighbors.append((ngx, ngy))
        return neighbors
    
    # Check if start == goal
    if start_grid == goal_grid:
        return [start, goal]
    
    # A* search
    open_set = [(0, start_grid[0], start_grid[1], 0)]  # (f, x, y, time)
    came_from = {}
    g_score = {(start_grid[0], start_grid[1], 0): 0}
    visited = set()
    
    while open_set:
        f, gx, gy, t = heapq.heappop(open_set)
        
        if (gx, gy, t) in visited:
            continue
        visited.add((gx, gy, t))
        
        if (gx, gy) == goal_grid:
            # Reconstruct path
            path = []
            current = (gx, gy, t)
            while current in came_from:
                gx, gy, t = current
                path.append(grid_to_world(gx, gy))
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        if t >= max_time:
            continue
        
        for ngx, ngy in get_neighbors(gx, gy):
            # Check obstacle
            if (ngx, ngy) in obstacles:
                continue
            
            # Check constraint
            if t + 1 in constraint_set and (ngx, ngy) in constraint_set[t + 1]:
                continue
            
            nt = t + 1
            tentative_g = g_score.get((gx, gy, t), float('inf')) + 1
            
            if (ngx, ngy, nt) not in visited:
                if tentative_g < g_score.get((ngx, ngy, nt), float('inf')):
                    came_from[(ngx, ngy, nt)] = (gx, gy, t)
                    g_score[(ngx, ngy, nt)] = tentative_g
                    f_score = tentative_g + heuristic(ngx, ngy)
                    heapq.heappush(open_set, (f_score, ngx, ngy, nt))
    
    return None


def detect_conflict(paths: List[List[Tuple[float, float]]]) -> Optional[Conflict]:
    """Detect first conflict in paths"""
    max_len = max(len(p) for p in paths) if paths else 0
    
    for t in range(max_len):
        positions = {}
        for i, path in enumerate(paths):
            if t < len(path):
                pos = path[t]
                # Check if any other robot is at same position
                for j, other_path in enumerate(paths):
                    if i != j and t < len(other_path):
                        other_pos = other_path[t]
                        dist = np.linalg.norm(np.array(pos) - np.array(other_pos))
                        if dist < 2 * (ROBOT_RADIUS + SAFETY_MARGIN):
                            return Conflict(i, j, pos[0], pos[1], t)
    return None


def cbs_planner(
    start_positions: List[np.ndarray],
    target_positions: List[np.ndarray],
    obstacles: Set[Tuple[int, int]]
) -> Optional[List[List[Tuple[float, float]]]]:
    """
    Conflict-Based Search planner.
    Returns paths for all robots or None if no solution.
    """
    num_robots = len(start_positions)
    
    # Initial paths (without constraints)
    initial_paths = []
    for i in range(num_robots):
        start = (float(start_positions[i][0]), float(start_positions[i][1]))
        goal = (float(target_positions[i][0]), float(target_positions[i][1]))
        path = astar_path(start, goal, obstacles, [], i)
        if path is None:
            return None
        initial_paths.append(path)
    
    # CBS search
    root = CBSNode(constraints=[], paths=initial_paths, cost=sum(len(p) for p in initial_paths))
    open_set = [root]
    
    max_iterations = 1000
    iteration = 0
    
    while open_set and iteration < max_iterations:
        iteration += 1
        node = heapq.heappop(open_set)
        
        conflict = detect_conflict(node.paths)
        if conflict is None:
            return node.paths
        
        # Create child nodes for both robots
        for robot_id in [conflict.robot_i, conflict.robot_j]:
            new_constraints = node.constraints + [
                Constraint(robot_id, conflict.x, conflict.y, conflict.time)
            ]
            
            # Replan for this robot
            new_paths = node.paths.copy()
            start = (float(start_positions[robot_id][0]), float(start_positions[robot_id][1]))
            goal = (float(target_positions[robot_id][0]), float(target_positions[robot_id][1]))
            new_path = astar_path(start, goal, obstacles, new_constraints, robot_id)
            
            if new_path is not None:
                new_paths[robot_id] = new_path
                cost = sum(len(p) for p in new_paths)
                child = CBSNode(new_constraints, new_paths, cost)
                heapq.heappush(open_set, child)
    
    return None


def test_formation(gui=True):
    """Test robots in 2 side lines with opposite goals."""
    print("\n" + "=" * 60)
    print("  CBS FORMATION TEST")
    print("=" * 60)
    
    # Setup PyBullet
    if gui:
        client_id = pyb.connect(pyb.GUI)
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
    
    # Create robots in 2 side lines (same as ORCA test)
    num_robots_per_side = 3
    line_spacing = 0.5
    line_offset = 1.5
    
    robots = {}
    start_positions = []
    target_positions = []
    
    # Left side robots (facing right)
    for i in range(num_robots_per_side):
        name = f"L_{i+1:02d}"
        y = (i - num_robots_per_side/2 + 0.5) * line_spacing
        start_pos = (-line_offset, y)
        target_pos = (line_offset, y)
        
        robot = create_robot('holonomic', 'dummy', position=start_pos, orientation=0.0, name=name)
        robots[name] = robot
        start_positions.append(np.array(start_pos))
        target_positions.append(np.array(target_pos))
    
    # Right side robots (facing left)
    for i in range(num_robots_per_side):
        name = f"R_{i+1:02d}"
        y = (i - num_robots_per_side/2 + 0.5) * line_spacing
        start_pos = (line_offset, y)
        target_pos = (-line_offset, y)
        
        robot = create_robot('holonomic', 'dummy', position=start_pos, orientation=np.pi, name=name)
        robots[name] = robot
        start_positions.append(np.array(start_pos))
        target_positions.append(np.array(target_pos))
    
    print(f"  Created {len(robots)} robots")
    
    # Build obstacle grid (empty for formation test)
    obstacles = set()
    
    # Plan paths using CBS
    print("  Planning paths with CBS...")
    paths = cbs_planner(start_positions, target_positions, obstacles)
    
    if paths is None:
        print("  ERROR: CBS planner failed to find solution")
        pyb.disconnect()
        return
    
    print(f"  Planning complete. Path lengths: {[len(p) for p in paths]}")
    
    # Convert paths to time-parameterized trajectories
    robot_names = list(robots.keys())
    max_path_len = max(len(p) for p in paths)
    
    # Pad shorter paths with final position
    for i, path in enumerate(paths):
        if len(path) < max_path_len:
            paths[i] = path + [path[-1]] * (max_path_len - len(path))
    
    # Run simulation
    max_steps = int(30.0 / TIMESTEP)
    step_count = 0
    trajectories = {name: [] for name in robots}
    path_indices = {name: 0 for name in robots}
    
    print("\n  Running simulation...")
    
    for step in range(max_steps):
        # Get current positions
        current_positions = {}
        for name, robot in robots.items():
            pos, _, _ = robot.get_state()
            current_positions[name] = pos
            trajectories[name].append(pos.copy())
        
        # Control at lower frequency
        if step_count % CTRL_STEP == 0:
            ctrl_step = step_count // CTRL_STEP
            
            for i, name in enumerate(robot_names):
                robot = robots[name]
                path = paths[i]
                
                if path_indices[name] < len(path):
                    target_waypoint = path[path_indices[name]]
                    current_pos = current_positions[name]
                    
                    # Compute desired velocity
                    direction = np.array(target_waypoint) - current_pos[:2]
                    dist = np.linalg.norm(direction)
                    
                    if dist < 0.05:  # Reached waypoint
                        path_indices[name] += 1
                        if path_indices[name] < len(path):
                            target_waypoint = path[path_indices[name]]
                            direction = np.array(target_waypoint) - current_pos[:2]
                            dist = np.linalg.norm(direction)
                    
                    if dist > 0.01:
                        vel = direction / dist * MAX_SPEED
                    else:
                        vel = np.zeros(2)
                    
                    # Check if reached final goal
                    final_goal = target_positions[i]
                    if np.linalg.norm(current_pos[:2] - final_goal) < 0.1:
                        vel = np.zeros(2)
                    
                    robot.command_velocity(np.array([vel[0], vel[1], 0.0]))
        
        pyb.stepSimulation()
        step_count += 1
        
        if gui:
            time.sleep(TIMESTEP * 0.3)
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    for (name, traj), color in zip(trajectories.items(), colors):
        traj_arr = np.array(traj)
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], '-', color=color, linewidth=1.5, label=name)
        ax.plot(traj_arr[0, 0], traj_arr[0, 1], 'o', color=color, markersize=8)
        ax.plot(traj_arr[-1, 0], traj_arr[-1, 1], 's', color=color, markersize=8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('CBS Formation Test - Robot Trajectories')
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/cbs_formation_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to /tmp/cbs_formation_test.png")
    
    if gui:
        print("\n  Press Enter to exit...")
        input()
    
    pyb.disconnect()
    plt.close()


def test_obstacles(gui=True):
    """Test CBS with static obstacles."""
    print("\n" + "=" * 60)
    print("  CBS OBSTACLE TEST")
    print("=" * 60)
    
    # Setup PyBullet
    if gui:
        client_id = pyb.connect(pyb.GUI)
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=6.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
    
    # Create robots (same as ORCA test)
    num_robots = 4
    robots = {}
    start_positions = []
    target_positions = []
    
    for i in range(num_robots):
        name = f"R_{i+1:02d}"
        angle = 2 * np.pi * i / num_robots
        start_pos = (2.0 * np.cos(angle), 2.0 * np.sin(angle))
        target_pos = (2.0 * np.cos(angle + np.pi), 2.0 * np.sin(angle + np.pi))
        
        robot = create_robot('holonomic', 'dummy', position=start_pos, orientation=angle, name=name)
        robots[name] = robot
        start_positions.append(np.array(start_pos))
        target_positions.append(np.array(target_pos))
    
    # Define obstacle (square)
    obstacle = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ]
    
    # Create visual obstacle
    obstacle_height = 0.5
    obstacle_visual_shape = pyb.createVisualShape(
        shapeType=pyb.GEOM_BOX,
        halfExtents=[1.0, 1.0, obstacle_height/2],
        rgbaColor=[0.5, 0.5, 0.5, 0.8]
    )
    obstacle_collision_shape = pyb.createCollisionShape(
        shapeType=pyb.GEOM_BOX,
        halfExtents=[1.0, 1.0, obstacle_height/2]
    )
    obstacle_body = pyb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=obstacle_collision_shape,
        baseVisualShapeIndex=obstacle_visual_shape,
        basePosition=[0, 0, obstacle_height/2]
    )
    
    print(f"  Created {len(robots)} robots")
    print(f"  Obstacle: square from (-1, -1) to (1, 1)")
    
    # Build obstacle grid
    obstacles = set()
    for x in np.arange(-1.0, 1.0 + GRID_RESOLUTION, GRID_RESOLUTION):
        for y in np.arange(-1.0, 1.0 + GRID_RESOLUTION, GRID_RESOLUTION):
            gx, gy = world_to_grid(x, y)
            obstacles.add((gx, gy))
    
    # Plan paths
    print("  Planning paths with CBS...")
    paths = cbs_planner(start_positions, target_positions, obstacles)
    
    if paths is None:
        print("  ERROR: CBS planner failed to find solution")
        pyb.disconnect()
        return
    
    print(f"  Planning complete. Path lengths: {[len(p) for p in paths]}")
    
    # Convert paths to time-parameterized trajectories
    robot_names = list(robots.keys())
    max_path_len = max(len(p) for p in paths)
    
    for i, path in enumerate(paths):
        if len(path) < max_path_len:
            paths[i] = path + [path[-1]] * (max_path_len - len(path))
    
    # Run simulation
    max_steps = int(30.0 / TIMESTEP)
    step_count = 0
    trajectories = {name: [] for name in robots}
    path_indices = {name: 0 for name in robots}
    
    print("\n  Running simulation...")
    
    for step in range(max_steps):
        current_positions = {}
        for name, robot in robots.items():
            pos, _, _ = robot.get_state()
            current_positions[name] = pos
            trajectories[name].append(pos.copy())
        
        if step_count % CTRL_STEP == 0:
            ctrl_step = step_count // CTRL_STEP
            
            for i, name in enumerate(robot_names):
                robot = robots[name]
                path = paths[i]
                
                if path_indices[name] < len(path):
                    target_waypoint = path[path_indices[name]]
                    current_pos = current_positions[name]
                    
                    direction = np.array(target_waypoint) - current_pos[:2]
                    dist = np.linalg.norm(direction)
                    
                    if dist < 0.05:
                        path_indices[name] += 1
                        if path_indices[name] < len(path):
                            target_waypoint = path[path_indices[name]]
                            direction = np.array(target_waypoint) - current_pos[:2]
                            dist = np.linalg.norm(direction)
                    
                    if dist > 0.01:
                        vel = direction / dist * MAX_SPEED
                    else:
                        vel = np.zeros(2)
                    
                    final_goal = target_positions[i]
                    if np.linalg.norm(current_pos[:2] - final_goal) < 0.1:
                        vel = np.zeros(2)
                    
                    robot.command_velocity(np.array([vel[0], vel[1], 0.0]))
        
        pyb.stepSimulation()
        step_count += 1
        
        if gui:
            time.sleep(TIMESTEP * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw obstacle
    obs_x = [v[0] for v in obstacle] + [obstacle[0][0]]
    obs_y = [v[1] for v in obstacle] + [obstacle[0][1]]
    ax.fill(obs_x, obs_y, alpha=0.3, color='gray', label='Obstacle')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    for (name, traj), color in zip(trajectories.items(), colors):
        traj_arr = np.array(traj)
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], '-', color=color, linewidth=1.5, label=name)
        ax.plot(traj_arr[0, 0], traj_arr[0, 1], 'o', color=color, markersize=8)
        ax.plot(traj_arr[-1, 0], traj_arr[-1, 1], 's', color=color, markersize=8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('CBS Obstacle Test - Robot Trajectories')
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/cbs_obstacle_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to /tmp/cbs_obstacle_test.png")
    
    if gui:
        print("\n  Press Enter to exit...")
        input()
    
    pyb.disconnect()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CBS Navigation Test")
    parser.add_argument("--test", type=str, default="formation",
                       choices=['formation', 'obstacles', 'all'],
                       help="Test to run")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run without GUI")
    
    args = parser.parse_args()
    
    if args.test in ['formation', 'all']:
        test_formation(gui=not args.no_gui)
    
    if args.test in ['obstacles', 'all']:
        test_obstacles(gui=not args.no_gui)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
