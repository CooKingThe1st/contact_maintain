"""Contact Maintain - Contact maintenance with holonomic and differential-drive robots."""

from pathlib import Path
import rospkg

from contact_maintain.util import (
    wrap_to_pi,
    rot2d,
    skew2d,
    skew3d,
    signed_angle,
    unit,
    perp2d,
    cuboid_vertices,
)
from contact_maintain.pyb_simulation import (
    get_contact_force,
    BulletBody,
    BulletSquareSlider,
    BulletCircleSlider,
    BulletBlock,
    BulletPillar,
)
from contact_maintain.robots import (
    HolonomicRobot,
    DifferentialDriveRobot,
)
from contact_maintain.omniwheel_robot import (
    OmniwheelRobot,
    compute_wheel_velocities,
    compute_body_velocity,
)
from contact_maintain.diffdrive_wheel_robot import (
    DiffDriveWheelRobot,
    compute_wheel_velocities_diffdrive,
    compute_body_velocity_diffdrive,
)
from contact_maintain.diffdrive_wheel_physics_robot import (
    DiffDriveWheelPhysicsRobot,
)
from contact_maintain.control import (
    HolonomicVelocityController,
    DifferentialDriveController,
    ContactMaintainController,
)
from contact_maintain.observer import (
    ContactState,
    ContactRecord,
    ContactObserver,
    ContactPointTracker,
)
from contact_maintain.logging import DataLogger
from contact_maintain.visualization import (
    plot_trajectory,
    plot_contact_forces,
    plot_contact_analysis,
    plot_scene_snapshot,
)
from contact_maintain.solvers import (
    ContactMaintainSolverBase,
    ForceBasedContactSolver,
    PositionBasedContactSolver,
    AdaptiveContactSolver,
    DiffDriveForceBasedSolver,
    DiffDrivePositionBasedSolver,
    DiffDriveAdaptiveSolver,
    create_solver,
)
from contact_maintain.web_observer import (
    WebObserver,
    RobotState,
    ObjectState,
)
from contact_maintain.objects import (
    TShapeObject,
    LShapeObject,
)
from contact_maintain.object_bridge import (
    generic_to_pybullet,
    pybullet_to_generic,
    create_standard_pybullet_objects,
    BridgedObject,
    is_convex,
    decompose_to_convex_parts,
)
from contact_maintain.contact_maintain_controller import (
    InstantVelocityMatcher,
    WrenchTrackingController,
    ContactMaintenanceState,
)
from contact_maintain.robot_factory import (
    create_robot,
    get_robot_info,
    list_robot_configurations,
    is_wheel_robot,
    get_wheel_velocities,
    get_command_wheel_velocities,
)

# Package paths
rospack = rospkg.RosPack()
PACKAGE_PATH = Path(rospack.get_path("contact_maintain"))
CONFIG_DIR_PATH = PACKAGE_PATH / "config"
URDF_DIR_PATH = PACKAGE_PATH / "urdf"
