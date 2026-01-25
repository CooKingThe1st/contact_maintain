"""Robot classes for contact maintenance simulation."""
import numpy as np
import pybullet as pyb
import pyb_utils

from contact_maintain.pyb_simulation import get_contact_force


class HolonomicRobot(pyb_utils.Robot):
    """Holonomic (omni-directional) robot for PyBullet simulation.
    
    Based on small mobile robot design:
    - Cylindrical body (radius ~0.06m, height ~0.08m)
    - Sphere bumper/contact sensor at front
    - 3 DOF: x, y translation and rotation around z-axis
    - Pure velocity control (no wheel physics)
    
    Parameters
    ----------
    urdf_path : str
        Path to the robot URDF file.
    position : tuple or list, optional
        Initial (x, y) position of the robot. Default is (0, 0).
    orientation : float, optional
        Initial orientation (theta) of the robot in radians. Default is 0.
    contact_mu : float, optional
        Friction coefficient for the bumper/contact element. Default is 1.0.
    """
    
    # Joint indices in the URDF
    X_JOINT_IDX = 0
    Y_JOINT_IDX = 1
    THETA_JOINT_IDX = 2
    
    # Robot dimensions (scaled to realistic small robot)
    BODY_RADIUS = 0.06
    BODY_HEIGHT = 0.08
    BUMPER_RADIUS = 0.015
    
    def __init__(self, urdf_path, position=(0, 0), orientation=0.0, contact_mu=1.0):
        # Load the URDF with fixed base (the world link is the fixed reference)
        uid = pyb.loadURDF(urdf_path, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        super().__init__(uid, tool_link_name="bumper")
        
        # Store initial configuration
        self.pos_init = np.array(position, dtype=float)
        self.orn_init = float(orientation)
        
        # Get joint info
        self._build_joint_info()
        
        # Set contact friction
        self.set_contact_friction(contact_mu)
        
        # Reset to initial configuration
        self.reset()
    
    def _build_joint_info(self):
        """Build internal joint information mappings."""
        self.joint_indices = [self.X_JOINT_IDX, self.Y_JOINT_IDX, self.THETA_JOINT_IDX]
        self.num_joints = 3
        
        # Find the bumper/contact link index
        self.contact_link_idx = None
        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            link_name = info[12].decode('utf-8')
            if link_name == 'bumper':
                self.contact_link_idx = i
                break
        
        if self.contact_link_idx is None:
            raise RuntimeError("Could not find 'bumper' link in URDF")
    
    def set_contact_friction(self, mu):
        """Set friction coefficient for the contact element.
        
        Parameters
        ----------
        mu : float
            Friction coefficient.
        """
        pyb.changeDynamics(self.uid, self.contact_link_idx, lateralFriction=mu)
    
    def get_state(self):
        """Get the current state of the robot.
        
        Returns
        -------
        position : np.ndarray, shape (2,)
            Current (x, y) position.
        orientation : float
            Current orientation (theta) in radians.
        velocity : np.ndarray, shape (3,)
            Current velocity (vx, vy, omega).
        """
        states = pyb.getJointStates(self.uid, self.joint_indices)
        position = np.array([states[0][0], states[1][0]])
        orientation = states[2][0]
        velocity = np.array([states[0][1], states[1][1], states[2][1]])
        return position, orientation, velocity
    
    def get_pose(self):
        """Get the current pose of the robot.
        
        Returns
        -------
        position : np.ndarray, shape (2,)
            Current (x, y) position.
        orientation : float
            Current orientation (theta) in radians.
        """
        position, orientation, _ = self.get_state()
        return position, orientation
    
    def get_contact_position(self):
        """Get the position of the contact element.
        
        Returns
        -------
        np.ndarray, shape (3,)
            Position of the contact element in world frame.
        """
        state = pyb.getLinkState(self.uid, self.contact_link_idx)
        return np.array(state[0])
    
    def get_contact_force(self, object_uids, max_contacts=1):
        """Get the contact force between the robot and objects.
        
        Parameters
        ----------
        object_uids : list of int
            List of PyBullet body UIDs to check contact with.
        max_contacts : int, optional
            Maximum number of contact points expected. Default is 1.
        
        Returns
        -------
        np.ndarray, shape (3,)
            Total contact force in world frame.
        """
        total_force = np.zeros(3)
        for uid in object_uids:
            force = get_contact_force(
                self.uid, uid, 
                linkIndexA=self.contact_link_idx, 
                max_contacts=max_contacts
            )
            total_force += force
        return total_force
    
    def command_velocity(self, velocity):
        """Command the velocity of the robot.
        
        Parameters
        ----------
        velocity : array-like, shape (3,)
            Velocity command (vx, vy, omega) in world frame.
        """
        vx, vy, omega = velocity
        pyb.setJointMotorControlArray(
            self.uid,
            self.joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=[vx, vy, omega],
        )
    
    def reset(self, position=None, orientation=None):
        """Reset the robot to a specified or initial configuration.
        
        Parameters
        ----------
        position : array-like, shape (2,), optional
            New (x, y) position. If None, uses initial position.
        orientation : float, optional
            New orientation (theta). If None, uses initial orientation.
        """
        if position is not None:
            self.pos_init = np.array(position, dtype=float)
        if orientation is not None:
            self.orn_init = float(orientation)
        
        # Stop the robot
        self.command_velocity([0, 0, 0])
        
        # Reset joint states
        pyb.resetJointState(self.uid, self.X_JOINT_IDX, self.pos_init[0], 0)
        pyb.resetJointState(self.uid, self.Y_JOINT_IDX, self.pos_init[1], 0)
        pyb.resetJointState(self.uid, self.THETA_JOINT_IDX, self.orn_init, 0)


class DifferentialDriveRobot(HolonomicRobot):
    """Differential-drive robot for PyBullet simulation.
    
    This robot is kinematically constrained to differential-drive motion.
    It uses the same URDF as the holonomic robot but applies velocity constraints.
    
    The robot can only move forward/backward and rotate, not sideways.
    
    Parameters
    ----------
    urdf_path : str
        Path to the robot URDF file.
    position : tuple or list, optional
        Initial (x, y) position of the robot. Default is (0, 0).
    orientation : float, optional
        Initial orientation (theta) of the robot in radians. Default is 0.
    contact_mu : float, optional
        Friction coefficient for the contact element. Default is 1.0.
    wheel_base : float, optional
        Distance between wheels (for velocity conversion). Default is 0.2.
    """
    
    def __init__(self, urdf_path, position=(0, 0), orientation=0.0, 
                 contact_mu=1.0, wheel_base=0.2):
        super().__init__(urdf_path, position, orientation, contact_mu)
        self.wheel_base = wheel_base
    
    def command_velocity(self, velocity):
        """Command the velocity of the robot with differential-drive constraints.
        
        Parameters
        ----------
        velocity : array-like
            Either (v, omega) for linear and angular velocity,
            or (v_left, v_right) for wheel velocities.
            If length is 2, interpreted as (v, omega).
        """
        if len(velocity) == 2:
            v, omega = velocity
        else:
            # If 3 values given, use first (forward) and third (angular)
            v, omega = velocity[0], velocity[2]
        
        # Get current orientation
        _, theta, _ = self.get_state()
        
        # Convert to world-frame velocities
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        
        # Apply velocities through the holonomic joints
        pyb.setJointMotorControlArray(
            self.uid,
            self.joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=[vx, vy, omega],
        )
    
    def command_wheel_velocities(self, v_left, v_right):
        """Command wheel velocities directly.
        
        Parameters
        ----------
        v_left : float
            Left wheel velocity.
        v_right : float
            Right wheel velocity.
        """
        # Convert wheel velocities to (v, omega)
        v = (v_left + v_right) / 2
        omega = (v_right - v_left) / self.wheel_base
        self.command_velocity((v, omega))

