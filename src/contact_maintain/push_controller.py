"""Push Controller Module for Distributed Swarm.

This module provides a base class and implementations for pushing
controllers used when robots are in contact with the object.

Author: Contact Maintain Team
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class PushController(ABC):
    """Base class for push controllers.
    
    Push controllers are used when robots are in contact with the
    object and need to maintain contact while pushing.
    """
    
    @abstractmethod
    def compute_velocity(
        self,
        robot_position: np.ndarray,
        robot_heading: float,
        object_position: np.ndarray,
        object_orientation: float,
        object_velocity: np.ndarray,
        object_angular_velocity: float,
        contact_force: float,
        in_contact: bool,
        t: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """Compute velocity command for pushing.
        
        Parameters
        ----------
        robot_position : np.ndarray
            Current robot position (x, y)
        robot_heading : float
            Current robot heading (radians)
        object_position : np.ndarray
            Object center position (x, y)
        object_orientation : float
            Object orientation (radians)
        object_velocity : np.ndarray
            Object linear velocity (vx, vy)
        object_angular_velocity : float
            Object angular velocity (rad/s)
        contact_force : float
            Current contact force magnitude (Newtons)
        in_contact : bool
            Whether robot is in contact
        t : float
            Current simulation time
        **kwargs
            Additional arguments (e.g., robot object for wheel velocities)
            
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy, omega)
        """
        pass
    
    @abstractmethod
    def set_desired_object_motion(
        self,
        desired_velocity: np.ndarray,
        desired_angular_velocity: float,
    ):
        """Set desired object motion.
        
        Parameters
        ----------
        desired_velocity : np.ndarray
            Desired object linear velocity (vx, vy)
        desired_angular_velocity : float
            Desired object angular velocity (rad/s)
        """
        pass


class Phase7PushController(PushController):
    """Phase 7 Push Controller (wraps Phase7BetaVerDecouple).
    
    This controller wraps the existing Phase7BetaVerDecouple controller
    to provide a clean interface for the distributed architecture.
    """
    
    def __init__(
        self,
        robot_uid: int,
        object_uid: int,
        generic_object: Any,
        t_param: float,
        desired_object_velocity: Optional[np.ndarray] = None,
        desired_object_angular_velocity: float = 0.0,
    ):
        """
        Parameters
        ----------
        robot_uid : int
            PyBullet UID of the robot
        object_uid : int
            PyBullet UID of the object
        generic_object : GenericObject
            Object model for boundary parameterization
        t_param : float
            Target t_param on object boundary
        desired_object_velocity : Optional[np.ndarray]
            Initial desired object linear velocity (vx, vy)
        desired_object_angular_velocity : float
            Initial desired object angular velocity (rad/s)
        """
        self.robot_uid = robot_uid
        self.object_uid = object_uid
        self.generic_object = generic_object
        self.t_param = t_param
        
        # Import Phase7BetaVerDecouple from test script (will need to refactor later)
        # For now, we'll create it dynamically
        self._phase7_controller = None
        self._desired_object_velocity = desired_object_velocity if desired_object_velocity is not None else np.array([0.0, 0.0])
        self._desired_object_angular_velocity = desired_object_angular_velocity
        
        # Lazy initialization - will create controller when first used
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure Phase7 controller is initialized."""
        if not self._initialized:
            # Import here to avoid circular dependencies
            # Note: This assumes Phase7BetaVerDecouple is available
            # In the actual implementation, this should be imported from a proper module
            try:
                # Try to import from test script location (temporary)
                import sys
                from pathlib import Path
                import rospkg
                rospack = rospkg.RosPack()
                pkg_path = rospack.get_path("contact_maintain")
                test_path = Path(pkg_path) / "scripts" / "test"
                if str(test_path) not in sys.path:
                    sys.path.insert(0, str(test_path))
                
                # This is a workaround - in production, Phase7BetaVerDecouple should be in a proper module
                # For now, we'll create a minimal wrapper or import it
                # The actual implementation should move Phase7BetaVerDecouple to a proper module
                from test_magnum_motion_planning import Phase7BetaVerDecouple
                
                self._phase7_controller = Phase7BetaVerDecouple(
                    robot_uid=self.robot_uid,
                    object_uid=self.object_uid,
                    generic_object=self.generic_object,
                    t_param=self.t_param,
                    desired_object_velocity=self._desired_object_velocity,
                    desired_object_angular_velocity=self._desired_object_angular_velocity,
                )
                self._initialized = True
            except ImportError:
                # Fallback: create a minimal implementation
                # This should be replaced with proper module structure
                raise RuntimeError(
                    "Phase7BetaVerDecouple not available. "
                    "Please ensure it is accessible or refactor it into a proper module."
                )
    
    def compute_velocity(
        self,
        robot_position: np.ndarray,
        robot_heading: float,
        object_position: np.ndarray,
        object_orientation: float,
        object_velocity: np.ndarray,
        object_angular_velocity: float,
        contact_force: float,
        in_contact: bool,
        t: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """Compute velocity using Phase7 controller."""
        self._ensure_initialized()
        
        return self._phase7_controller.compute_velocity(
            robot_pos=robot_position,
            robot_heading=robot_heading,
            object_pos=object_position,
            object_orientation=object_orientation,
            object_velocity=object_velocity,
            object_angular_velocity=object_angular_velocity,
            contact_force=contact_force,
            in_contact=in_contact,
            t=t,
            record_history=kwargs.get('record_history', False),
            robot=kwargs.get('robot', None),
        )
    
    def set_desired_object_motion(
        self,
        desired_velocity: np.ndarray,
        desired_angular_velocity: float,
    ):
        """Update desired object motion."""
        self._desired_object_velocity = np.array(desired_velocity, dtype=float)
        self._desired_object_angular_velocity = float(desired_angular_velocity)
        
        if self._initialized:
            self._phase7_controller.desired_object_velocity = self._desired_object_velocity
            self._phase7_controller.desired_object_angular_velocity = self._desired_object_angular_velocity


# Factory function for creating push controllers
def create_push_controller(
    controller_type: str,
    **kwargs
) -> PushController:
    """Create a push controller based on type.
    
    Parameters
    ----------
    controller_type : str
        Controller type: 'phase7' or other types
    **kwargs
        Additional arguments passed to controller constructor
        
    Returns
    -------
    PushController
        Push controller instance
    """
    if controller_type == 'phase7':
        return Phase7PushController(**kwargs)
    else:
        raise ValueError(f"Unknown push controller type: {controller_type}")
