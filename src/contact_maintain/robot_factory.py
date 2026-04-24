"""Robot Factory Module

Factory pattern for creating robots based on kinematics and model type.

Supported combinations:
- holonomic + dummy  → HolonomicRobot (direct velocity control)
- holonomic + wheel  → OmniwheelRobot (4-wheel omni control)
- diffdrive + dummy  → DifferentialDriveRobot (direct velocity control)
- diffdrive + wheel  → DiffDriveWheelRobot (2-wheel control)
- diffdrive + wheel_physics → DiffDriveWheelPhysicsRobot (wheel-contact driven)
"""
from pathlib import Path
from typing import Tuple, Optional, Union, Literal
import numpy as np

import rospkg

# Get package path
rospack = rospkg.RosPack()
PKG_PATH = Path(rospack.get_path("contact_maintain"))
URDF_DIR = PKG_PATH / "urdf"


# Type aliases
KinematicsType = Literal['holonomic', 'diffdrive']
ModelType = Literal['dummy', 'wheel', 'wheel_physics']
RobotType = Union['HolonomicRobot', 'OmniwheelRobot', 
                  'DifferentialDriveRobot', 'DiffDriveWheelRobot',
                  'DiffDriveWheelPhysicsRobot']


# Robot class imports (lazy to avoid circular imports)
_robot_classes = {}

def _get_robot_classes():
    """Lazy load robot classes."""
    global _robot_classes
    if not _robot_classes:
        from contact_maintain.robots import HolonomicRobot, DifferentialDriveRobot
        from contact_maintain.omniwheel_robot import OmniwheelRobot
        from contact_maintain.diffdrive_wheel_robot import DiffDriveWheelRobot
        from contact_maintain.diffdrive_wheel_physics_robot import DiffDriveWheelPhysicsRobot
        _robot_classes = {
            ('holonomic', 'dummy'): HolonomicRobot,
            ('holonomic', 'wheel'): OmniwheelRobot,
            ('diffdrive', 'dummy'): DifferentialDriveRobot,
            ('diffdrive', 'wheel'): DiffDriveWheelRobot,
            ('diffdrive', 'wheel_physics'): DiffDriveWheelPhysicsRobot,
        }
    return _robot_classes


# URDF paths for each robot type
URDF_PATHS = {
    ('holonomic', 'dummy'): URDF_DIR / "holonomic_robot.urdf",
    ('holonomic', 'wheel'): URDF_DIR / "omniwheel_robot.urdf",
    ('diffdrive', 'dummy'): URDF_DIR / "holonomic_robot.urdf",  # Uses same URDF, different control
    ('diffdrive', 'wheel'): URDF_DIR / "diffdrive_wheel_robot.urdf",
    ('diffdrive', 'wheel_physics'): URDF_DIR / "diffdrive_wheel_robot_disc_bumper.urdf",
}


def create_robot(
    kinematics: KinematicsType,
    model: ModelType,
    position: Tuple[float, float],
    orientation: float = 0.0,
    contact_mu: float = 0.8,
    name: Optional[str] = None,
) -> RobotType:
    """Create a robot with specified kinematics and model type.
    
    Parameters
    ----------
    kinematics : str
        'holonomic' for omni-directional, 'diffdrive' for differential drive.
    model : str
        'dummy' for direct velocity control, 'wheel' for existing wheel model,
        'wheel_physics' for wheel-contact-driven diff drive.
    position : tuple
        Initial (x, y) position.
    orientation : float
        Initial heading in radians.
    contact_mu : float
        Friction coefficient for bumper contact.
    name : str, optional
        Robot name for identification.
    
    Returns
    -------
    Robot instance
        One of HolonomicRobot, OmniwheelRobot, DifferentialDriveRobot,
        DiffDriveWheelRobot, or DiffDriveWheelPhysicsRobot.
    
    Raises
    ------
    ValueError
        If invalid kinematics or model type.
    
    Examples
    --------
    >>> robot = create_robot('holonomic', 'dummy', (0, 0), orientation=0.0)
    >>> robot = create_robot('diffdrive', 'wheel', (1, 1), orientation=np.pi/2)
    >>> robot = create_robot('diffdrive', 'wheel_physics', (1, 1), orientation=np.pi/2)
    """
    key = (kinematics, model)
    
    robot_classes = _get_robot_classes()
    
    if key not in robot_classes:
        valid_keys = list(robot_classes.keys())
        raise ValueError(
            f"Invalid robot configuration: kinematics='{kinematics}', model='{model}'. "
            f"Valid combinations: {valid_keys}"
        )
    
    robot_class = robot_classes[key]
    urdf_path = str(URDF_PATHS[key])
    
    # Create robot
    robot = robot_class(
        urdf_path=urdf_path,
        position=position,
        orientation=orientation,
        contact_mu=contact_mu,
    )
    
    # Attach metadata
    robot.kinematics_type = kinematics
    robot.model_type = model
    robot.robot_name = name
    
    return robot


def get_robot_info(kinematics: KinematicsType, model: ModelType) -> dict:
    """Get information about a robot configuration.
    
    Parameters
    ----------
    kinematics : str
        'holonomic' or 'diffdrive'.
    model : str
        'dummy', 'wheel', or 'wheel_physics'.
    
    Returns
    -------
    dict
        Robot configuration information.
    """
    key = (kinematics, model)
    robot_classes = _get_robot_classes()
    
    if key not in robot_classes:
        return None
    
    return {
        'kinematics': kinematics,
        'model': model,
        'class_name': robot_classes[key].__name__,
        'urdf_path': str(URDF_PATHS[key]),
        'has_wheel_velocities': model == 'wheel',
        'is_holonomic': kinematics == 'holonomic',
        'velocity_dof': 3 if kinematics == 'holonomic' else 2,
    }


def list_robot_configurations() -> list:
    """List all available robot configurations.
    
    Returns
    -------
    list
        List of (kinematics, model) tuples.
    """
    return [
        ('holonomic', 'dummy'),
        ('holonomic', 'wheel'),
        ('diffdrive', 'dummy'),
        ('diffdrive', 'wheel'),
        ('diffdrive', 'wheel_physics'),
    ]


def is_wheel_robot(robot) -> bool:
    """Check if robot has wheel velocity interface.
    
    Parameters
    ----------
    robot : Robot instance
        Any robot created by create_robot().
    
    Returns
    -------
    bool
        True if robot has get_wheel_velocities() method.
    """
    return hasattr(robot, 'get_wheel_velocities')


def get_wheel_velocities(robot) -> Optional[np.ndarray]:
    """Get wheel velocities if available.
    
    Parameters
    ----------
    robot : Robot instance
        Any robot.
    
    Returns
    -------
    np.ndarray or None
        Wheel velocities if robot has wheels, else None.
    """
    if hasattr(robot, 'get_wheel_velocities'):
        return robot.get_wheel_velocities()
    return None


def get_command_wheel_velocities(robot) -> Optional[np.ndarray]:
    """Get last commanded wheel velocities if available.
    
    Parameters
    ----------
    robot : Robot instance
        Any robot.
    
    Returns
    -------
    np.ndarray or None
        Last commanded wheel velocities if robot has wheels.
    """
    if hasattr(robot, 'last_wheel_speeds'):
        return robot.last_wheel_speeds.copy()
    return None

