#!/usr/bin/env python3
"""
ORCA Navigation Test

Tests ORCA-based collision avoidance with:
1. Robots in formation (2 side lines, opposite goals)
2. Static obstacles

Usage:
    # Basic formation test
    python test_orca_navigation.py --test formation
    
    # Test with obstacles
    python test_orca_navigation.py --test obstacles
    
    # Both tests
    python test_orca_navigation.py --test all
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))

import rvo2

from contact_maintain.robot_factory import create_robot


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 60
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)


def test_formation(gui=True):
    """Test robots in 2 side lines with opposite goals."""
    print("\n" + "=" * 60)
    print("  ORCA FORMATION TEST")
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
    
    # Create robots in 2 side lines
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
        target_pos = (line_offset, y)  # Opposite side
        
        robot = create_robot('holonomic', 'dummy', position=start_pos, orientation=0.0, name=name)
        robots[name] = robot
        start_positions.append(np.array(start_pos))
        target_positions.append(np.array(target_pos))
    
    # Right side robots (facing left)
    for i in range(num_robots_per_side):
        name = f"R_{i+1:02d}"
        y = (i - num_robots_per_side/2 + 0.5) * line_spacing
        start_pos = (line_offset, y)
        target_pos = (-line_offset, y)  # Opposite side
        
        robot = create_robot('holonomic', 'dummy', position=start_pos, orientation=np.pi, name=name)
        robots[name] = robot
        start_positions.append(np.array(start_pos))
        target_positions.append(np.array(target_pos))
    
    print(f"  Created {len(robots)} robots")
    print(f"  Left side: {num_robots_per_side} robots")
    print(f"  Right side: {num_robots_per_side} robots")
    
    # Create shared RVO2 simulator (like dummy_test_rvo2.py)
    sim = rvo2.PyRVOSimulator(
        timeStep=1.0 / CTRL_FREQ,
        neighborDist=2.0,
        maxNeighbors=10,
        timeHorizon=2.0,
        timeHorizonObst=0.5,
        radius=0.06,
        maxSpeed=0.3
    )
    
    # Add all agents to simulator
    robot_names = list(robots.keys())
    agent_ids = {}
    for i, name in enumerate(robot_names):
        agent_id = sim.addAgent((float(start_positions[i][0]), float(start_positions[i][1])))
        agent_ids[name] = agent_id
    
    # Run simulation
    max_steps = int(30.0 / TIMESTEP)  # 30 seconds
    step_count = 0
    trajectories = {name: [] for name in robots}
    
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
            # Update agent positions in simulator
            for name in robot_names:
                agent_id = agent_ids[name]
                pos = current_positions[name]
                sim.setAgentPosition(agent_id, (float(pos[0]), float(pos[1])))
            
            # Set preferred velocities for all agents (like dummy_test_rvo2.py)
            for i, name in enumerate(robot_names):
                agent_id = agent_ids[name]
                current_pos = current_positions[name]
                target_pos = target_positions[i]
                
                v_pref = target_pos - current_pos
                if np.linalg.norm(v_pref) > 1e-3:
                    v_pref = v_pref / np.linalg.norm(v_pref)
                else:
                    v_pref = np.zeros(2)
                sim.setAgentPrefVelocity(agent_id, (float(v_pref[0]), float(v_pref[1])))
            
            # Step simulator once (like dummy_test_rvo2.py)
            sim.doStep()
            
            # Get computed velocities and command robots
            for i, name in enumerate(robot_names):
                agent_id = agent_ids[name]
                target_pos = target_positions[i]
                current_pos = current_positions[name]
                
                # Get computed velocity from simulator
                vel = sim.getAgentVelocity(agent_id)
                vel_2d = np.array([vel[0], vel[1]])
                
                # Command robot
                robot = robots[name]
                robot.command_velocity(np.array([vel_2d[0], vel_2d[1], 0.0]))
                
                # Check if reached target
                dist = np.linalg.norm(current_pos - target_pos)
                if dist < 0.1:
                    # Stop
                    robot.command_velocity(np.zeros(3))
        
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
    ax.set_title('ORCA Formation Test - Robot Trajectories')
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/orca_formation_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to /tmp/orca_formation_test.png")
    
    if gui:
        print("\n  Press Enter to exit...")
        input()
    
    pyb.disconnect()
    plt.close()


def test_obstacles(gui=True):
    """Test ORCA with static obstacles."""
    print("\n" + "=" * 60)
    print("  ORCA OBSTACLE TEST")
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
    
    # Create robots
    num_robots = 4
    robots = {}
    start_positions = []
    target_positions = []
    
    # Place robots around a central obstacle
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
    
    # Create visual obstacle in PyBullet
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
        baseMass=0,  # Static
        baseCollisionShapeIndex=obstacle_collision_shape,
        baseVisualShapeIndex=obstacle_visual_shape,
        basePosition=[0, 0, obstacle_height/2]
    )
    
    print(f"  Created {len(robots)} robots")
    print(f"  Obstacle: square from (-1, -1) to (1, 1)")
    
    # Create shared RVO2 simulator with obstacles
    sim = rvo2.PyRVOSimulator(
        timeStep=1.0 / CTRL_FREQ,
        neighborDist=2.0,
        maxNeighbors=10,
        timeHorizon=2.0,
        timeHorizonObst=0.5,
        radius=0.06,
        maxSpeed=0.3
    )
    
    # Add obstacles
    obstacle_verts = [(float(v[0]), float(v[1])) for v in obstacle]
    sim.addObstacle(obstacle_verts)
    sim.processObstacles()
    
    # Add all agents to simulator
    robot_names = list(robots.keys())
    agent_ids = {}
    for i, name in enumerate(robot_names):
        agent_id = sim.addAgent((float(start_positions[i][0]), float(start_positions[i][1])))
        agent_ids[name] = agent_id
    
    # Run simulation
    max_steps = int(30.0 / TIMESTEP)
    step_count = 0
    trajectories = {name: [] for name in robots}
    
    print("\n  Running simulation...")
    
    for step in range(max_steps):
        current_positions = {}
        for name, robot in robots.items():
            pos, _, _ = robot.get_state()
            current_positions[name] = pos
            trajectories[name].append(pos.copy())
        
        if step_count % CTRL_STEP == 0:
            # Update agent positions in simulator
            for name in robot_names:
                agent_id = agent_ids[name]
                pos = current_positions[name]
                sim.setAgentPosition(agent_id, (float(pos[0]), float(pos[1])))
            
            # Set preferred velocities for all agents (like dummy_test_rvo2.py)
            for i, name in enumerate(robot_names):
                agent_id = agent_ids[name]
                current_pos = current_positions[name]
                target_pos = target_positions[i]
                
                v_pref = target_pos - current_pos
                if np.linalg.norm(v_pref) > 1e-3:
                    v_pref = v_pref / np.linalg.norm(v_pref)
                else:
                    v_pref = np.zeros(2)
                sim.setAgentPrefVelocity(agent_id, (float(v_pref[0]), float(v_pref[1])))
            
            # Step simulator once (like dummy_test_rvo2.py)
            sim.doStep()
            
            # Get computed velocities and command robots
            for i, name in enumerate(robot_names):
                agent_id = agent_ids[name]
                target_pos = target_positions[i]
                current_pos = current_positions[name]
                
                # Get computed velocity from simulator
                vel = sim.getAgentVelocity(agent_id)
                vel_2d = np.array([vel[0], vel[1]])
                
                # Command robot
                robot = robots[name]
                robot.command_velocity(np.array([vel_2d[0], vel_2d[1], 0.0]))
        
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
    ax.set_title('ORCA Obstacle Test - Robot Trajectories')
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/orca_obstacle_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to /tmp/orca_obstacle_test.png")
    
    if gui:
        print("\n  Press Enter to exit...")
        input()
    
    pyb.disconnect()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ORCA Navigation Test")
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
