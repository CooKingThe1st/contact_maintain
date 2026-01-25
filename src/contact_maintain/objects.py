"""Object definitions for contact maintenance simulation.

Includes standard shapes and non-convex composite shapes like T-shape.
"""
import numpy as np
import pybullet as pyb


class TShapeObject:
    """T-shaped non-convex object composed of two boxes.
    
    The T-shape is created by combining:
    - A horizontal bar (top of T)
    - A vertical bar (stem of T)
    
    Parameters
    ----------
    position : array-like
        (x, y, z) position of the T-shape center.
    horizontal_size : tuple
        (half_width, half_depth, half_height) of horizontal bar.
    vertical_size : tuple
        (half_width, half_depth, half_height) of vertical bar.
    mass : float
        Total mass of the object.
    mu : float
        Friction coefficient.
    color : tuple
        RGBA color.
    """
    
    def __init__(self, position, 
                 horizontal_size=(1.05, 0.24, 0.1), # Tripled & flattened
                 vertical_size=(0.24, 0.6, 0.1),    # Tripled & flattened
                 mass=45.0, mu=0.5, color=(0.2, 0.7, 0.3, 1.0)):
        self.position = np.array(position)
        # Ensure the object is spawned exactly on the floor based on its half-height
        self.position[2] = horizontal_size[2] 
        
        self.horizontal_size = horizontal_size
        self.vertical_size = vertical_size
        self.mass = mass
        
        # Calculate offset: 
        # We want the horizontal bar to sit at the end of the vertical stem.
        # Offset = half_length_of_stem + half_width_of_top_bar
        y_offset = vertical_size[1] + horizontal_size[1]
        self.h_offset = [0, y_offset, 0] 
        
        # Create collision shapes
        h_collision = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=horizontal_size
        )
        v_collision = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=vertical_size
        )
        
        # Create visual shapes
        h_visual = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=horizontal_size,
            rgbaColor=color
        )
        v_visual = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=vertical_size,
            rgbaColor=color
        )
        
        # Create multi-body
        # Base is the vertical stem, Link 0 is the horizontal top bar
        self.uid = pyb.createMultiBody(
            baseMass=mass * 0.7,
            baseCollisionShapeIndex=v_collision,
            baseVisualShapeIndex=v_visual,
            basePosition=list(self.position),
            baseOrientation=[0, 0, 0, 1],
            linkMasses=[mass * 0.3],
            linkCollisionShapeIndices=[h_collision],
            linkVisualShapeIndices=[h_visual],
            linkPositions=[self.h_offset],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[pyb.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1]]
        )
        
        pyb.changeDynamics(self.uid, -1, lateralFriction=mu)
        pyb.changeDynamics(self.uid, 0, lateralFriction=mu)
    
    def get_pose(self):
        """Get the pose of the T-shape."""
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos), np.array(orn)
    
    def get_velocity(self):
        """Get the velocity of the T-shape."""
        vel_lin, vel_ang = pyb.getBaseVelocity(self.uid)
        return np.array(vel_lin), np.array(vel_ang)
    
    def reset(self, position=None, orientation=None):
        """Reset the object position."""
        if position is None:
            position = self.position
        if orientation is None:
            orientation = [0, 0, 0, 1]
        
        pyb.resetBaseVelocity(self.uid, [0, 0, 0], [0, 0, 0])
        pyb.resetBasePositionAndOrientation(self.uid, list(position), list(orientation))


class LShapeObject:
    """L-shaped non-convex object composed of two boxes.
    
    Parameters
    ----------
    position : array-like
        (x, y, z) position of the L-shape center.
    long_size : tuple
        (half_width, half_depth, half_height) of long arm.
    short_size : tuple
        (half_width, half_depth, half_height) of short arm.
    mass : float
        Total mass of the object.
    mu : float
        Friction coefficient.
    color : tuple
        RGBA color.
    """
    
    def __init__(self, position,
                 long_size=(0.3, 0.1, 0.1),
                 short_size=(0.1, 0.1, 0.2),
                 mass=10.0, mu=0.5, color=(0.9, 0.6, 0.2, 1.0)):
        
        self.position = np.array(position)
        self.mass = mass
        
        # Long arm along x, short arm along z at the end
        l_offset = [0, 0, 0]
        s_offset = [long_size[0] - short_size[0], 0, short_size[2]]
        
        # Create collision shapes
        l_collision = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=long_size
        )
        s_collision = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=short_size
        )
        
        # Create visual shapes
        l_visual = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=long_size,
            rgbaColor=color
        )
        s_visual = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=short_size,
            rgbaColor=color
        )
        
        # Create multi-body
        self.uid = pyb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=l_collision,
            baseVisualShapeIndex=l_visual,
            basePosition=list(position),
            baseOrientation=[0, 0, 0, 1],
            linkMasses=[mass * 0.4],
            linkCollisionShapeIndices=[s_collision],
            linkVisualShapeIndices=[s_visual],
            linkPositions=[s_offset],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[pyb.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1]]
        )
        
        # Set friction
        pyb.changeDynamics(self.uid, -1, lateralFriction=mu)
        pyb.changeDynamics(self.uid, 0, lateralFriction=mu)
    
    def get_pose(self):
        """Get the pose of the L-shape."""
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos), np.array(orn)
    
    def get_velocity(self):
        """Get the velocity of the L-shape."""
        vel_lin, vel_ang = pyb.getBaseVelocity(self.uid)
        return np.array(vel_lin), np.array(vel_ang)
    
    def reset(self, position=None, orientation=None):
        """Reset the object position."""
        if position is None:
            position = self.position
        if orientation is None:
            orientation = [0, 0, 0, 1]
        
        pyb.resetBaseVelocity(self.uid, [0, 0, 0], [0, 0, 0])
        pyb.resetBasePositionAndOrientation(self.uid, list(position), list(orientation))

