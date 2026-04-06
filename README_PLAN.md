# Distributed Swarm Architecture - Implementation Plan

## Overview

This document describes the refactored distributed swarm architecture that replaces the centralized `SwarmHost` system. The new architecture follows a clean hierarchical structure: **Overworld Simulator → DistributedMonitor → Controller (Push/Navigate)**.

## Architecture

### Current Architecture Issues

1. **Centralized bottleneck**: `SwarmHost` tracks all robot states centrally
2. **Mixed responsibilities**: `RobotAgent` handles both navigation and pushing, but pushing controller is patched in test scripts
3. **Complex state machine**: 6 robot states and 5 swarm states
4. **Tight coupling**: Navigation logic embedded in `RobotAgent.compute_velocity()` making it hard to swap schemes

### Target Architecture

```
Overworld Simulator (Communication & Coordination)
    ↓
DistributedMonitor (Per Robot - State Management)
    ↓
Controller (Navigation or Push)
    ↓
Navigation Schemes (APF, Static Single, Divide-n-Conquer)
```

## Components

### 1. RobotMessage (`robot_message.py`)

Standardized message format for robot-to-robot communication.

**Fields:**
- `robot_name`: Unique identifier
- `position`: Current position (x, y)
- `state`: MonitorState (NAVIGATING or PUSHING)
- `target_t_param`: Target t_param on boundary
- `in_contact`: Contact status
- `contact_force`: Contact force magnitude
- `timestamp`: Simulation time

### 2. DistributedMonitor (`distributed_monitor.py`)

Per-robot monitor that maintains local state and coordinates between controllers.

**State Machine:**
- `NAVIGATING`: Robot is moving to target position
- `PUSHING`: Robot is in contact and pushing (all robots must be in contact)

**Key Responsibilities:**
- Update local state from sensors
- Process received messages from other robots
- Determine state transitions
- Delegate to appropriate controller (navigation or push)

### 3. NavigationController (`navigation_controller.py`)

Base class and implementations for navigation schemes.

**Schemes:**
- `APFNavigationController`: Rewritten APF navigation (specialized for pushing)
- `StaticSingleNavigationController`: Only one robot moves at a time
- `DivideConquerNavigationController`: Each robot manages non-overlapping edges 

### 4. PushController (`push_controller.py`)

Base class and implementations for pushing controllers.

**Types:**
- `Phase7PushController`: Wraps Phase7BetaVerDecouple logic

### 5. OverworldSimulator (`overworld_sim.py`)

Handles communication and coordinates all monitors.

**Responsibilities:**
- Create and manage all `DistributedMonitor` instances
- Broadcast messages between robots (simulate fully connected network)
- Handle reconfiguration triggers
- Provide unified interface for test scripts

### 6. Reconfiguration Planners (`reconfiguration.py`)

Each navigation scheme has its own reconfiguration planner:

- **APF**: All robots move simultaneously (direct assignment)
- **Static Single**: Optimal assignment using permutation check (minimize total distance)
- **Divide-n-Conquer**: Reassign edge segments, may need prep phase

## Navigation Schemes

### APF Navigation (Rewritten)

**File:** `navigation/apf_nav.py`

- **Rewritten** specifically for pushing problem (not wrapped)
- Less general than original APF, but simpler and more predictable
- All robots navigate simultaneously
- Optimized collision avoidance for contact maintenance

### Static Single Navigation

**File:** `navigation/static_single_nav.py`

- Only one robot moves at a time
- Priority system: robot closest to target moves first
- Other robots remain stationary (static environment)
- Message-based coordination
- **Specialized for pushing**: Predictable, avoids dynamic collision issues

### Divide-n-Conquer Navigation

**File:** `navigation/divide_conquer_nav.py`

- Each robot "manages" a set of consecutive non-overlapping edges
- Robots only move within their assigned edge segments
- **Initial prep phase**: Assigns edges to robots, handles spawn/overlap cases
- Distributed coordination via messages
- **Specialized for pushing**: Predictable movement patterns, clear responsibility zones

## Design Decisions

### 1. Two-State Monitor

Simplified from 6 states (IDLE, REACHING, APPROACHING, WAITING, PUSHING, CHANGING) to 2 states (NAVIGATING, PUSHING).

**Rationale:** Reduces complexity while maintaining functionality. Navigation covers all movement phases, pushing covers contact maintenance.

### 2. Distributed by Default

Each robot has its own monitor, no central coordinator.

**Rationale:** 
- No single point of failure
- Scales better
- More realistic for distributed systems
- Easier to test individual components

### 3. Modular Controllers

Navigation and pushing are separate, swappable controllers.

**Rationale:**
- Clear separation of concerns
- Easy to swap navigation schemes
- Easy to add new controllers
- Better testability

### 4. Message-Based Coordination

Robots coordinate via messages (simulated fully connected network).

**Rationale:**
- Realistic for distributed systems
- Can be extended to handle network failures
- Clear communication protocol
- Easy to debug

### 5. Scheme-Specific Reconfiguration

Each navigation scheme has optimized reconfiguration logic.

**Rationale:**
- Different schemes have different characteristics
- Optimization can be tailored to scheme
- Better performance

### 6. Specialized Navigation Schemes

All schemes are designed specifically for pushing problem.

**Rationale:**
- Trade generality for simplicity and predictability
- Better performance for specific use case
- Easier to understand and debug

### 7. APF Rewrite

APF navigation is rewritten (not wrapped) to be optimized for contact maintenance.

**Rationale:**
- Original APF is too general
- Rewritten version is simpler and more predictable
- Better suited for pushing scenarios

## Migration Guide

### Phase 1: Create New Files (No Breaking Changes)

All new files are created alongside existing ones. Old `SwarmHost` and `RobotAgent` remain unchanged.

### Phase 2: Test New Architecture

1. Use `test_distributed_swarm.py` to test each navigation scheme
2. Compare performance with old system
3. Test reconfiguration for each scheme

### Phase 3: Migration (Optional)

1. Update test scripts to use new architecture
2. Deprecate old `SwarmHost` (keep for backward compatibility)
3. Update documentation

## Usage Example

```python
from contact_maintain.overworld_sim import OverworldSimulator

# Create robots
robots = {...}  # Dict of robot instances

# Create overworld simulator
overworld = OverworldSimulator(
    robots=robots,
    object_uid=object_uid,
    generic_object=generic_object,
    navigation_scheme='apf',  # or 'static_single', 'divide_conquer'
    push_controller_type='phase7',
)

# Assign targets
target_map = {'R_01': 0.25, 'R_02': 0.5, 'R_03': 0.75, 'R_04': 0.0}
overworld.assign_targets(target_map)

# Main loop
for step in range(n_steps):
    obj_state = get_object_state(object_uid)
    
    # Update overworld
    overworld.update(dt, obj_state)
    
    # Compute velocities
    velocities = overworld.compute_velocities(obj_state)
    
    # Command robots
    for name, robot in robots.items():
        robot.command_velocity(velocities[name])
```

## Testing Strategy

### Unit Tests

Each component tested independently:
- `RobotMessage`: Serialization/deserialization
- `DistributedMonitor`: State transitions, message processing
- `NavigationController`: Velocity computation for each scheme
- `PushController`: Velocity computation
- `ReconfigurationPlanner`: Assignment optimization

### Integration Tests

1. **Navigation Tests**: Test each scheme with 4 robots
   - APF: All robots navigate simultaneously
   - Static Single: One robot moves at a time
   - Divide-n-Conquer: Robots respect edge boundaries

2. **Reconfiguration Tests**: Test reconfiguration for each scheme
   - APF: Direct assignment
   - Static Single: Optimal assignment
   - Divide-n-Conquer: Edge reassignment

3. **State Transition Tests**: Test NAVIGATING → PUSHING transitions
   - All robots at target
   - All robots in contact
   - Transition to pushing

### Comparison Tests

Run same scenarios with old and new architecture:
- Compare performance metrics
- Compare behavior
- Identify regressions

## File Structure

```
src/contact_maintain/
├── distributed_monitor.py          # Per-robot monitor
├── overworld_sim.py                # Communication & coordination
├── navigation_controller.py       # Navigation controller base & implementations
├── push_controller.py              # Push controller base & Phase7 wrapper
├── robot_message.py                # Message protocol
├── reconfiguration.py              # Reconfiguration planners
├── navigation/                     # Navigation schemes
│   ├── __init__.py
│   ├── apf_nav.py                  # APF navigation (rewritten)
│   ├── static_single_nav.py        # Static single navigation
│   └── divide_conquer_nav.py       # Divide-n-conquer navigation
├── swarm.py                        # OLD: SwarmHost (kept for compatibility)
├── robot_agent.py                  # OLD: RobotAgent (kept for compatibility)
└── apf_navigation.py               # OLD: Existing APF (kept for reference)

scripts/test/
├── test_distributed_swarm.py       # NEW: Test script for new architecture
└── test_magnum_motion_planning.py  # OLD: Test (kept for comparison)
```

## Benefits

1. **Modularity**: Easy to swap navigation schemes without changing monitor logic
2. **Distributed**: No single point of failure, scales better
3. **Testability**: Each component can be tested independently
4. **Clarity**: Clear separation of concerns (monitor → controller)
5. **Extensibility**: Easy to add new navigation schemes or controllers

## Notes

- All new files are created as new implementations to avoid breaking existing code
- Old `SwarmHost` and `RobotAgent` remain for backward compatibility
- Test scripts support both old and new architectures during transition
- Navigation schemes are specialized for pushing problem, trading generality for simplicity

## Future Work

1. **Network Simulation**: Extend message passing to simulate network failures/delays
2. **Additional Navigation Schemes**: Add more specialized schemes
3. **Performance Optimization**: Optimize reconfiguration planners
4. **Visualization**: Add visualization tools for debugging
5. **Documentation**: Expand with more examples and use cases
