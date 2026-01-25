#!/bin/bash
# Compile xacro files to URDF

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XACRO_DIR="${SCRIPT_DIR}/xacro"
URDF_DIR="${SCRIPT_DIR}"

# Compile holonomic robot (dummy - pure velocity control)
rosrun xacro xacro "${XACRO_DIR}/holonomic_robot.urdf.xacro" > "${URDF_DIR}/holonomic_robot.urdf"
echo "Compiled holonomic_robot.urdf"

# Compile omniwheel robot (realistic - 4 wheel control)
rosrun xacro xacro "${XACRO_DIR}/omniwheel_robot.urdf.xacro" > "${URDF_DIR}/omniwheel_robot.urdf"
echo "Compiled omniwheel_robot.urdf"

# Compile differential drive robot (realistic - 2 wheel control)
rosrun xacro xacro "${XACRO_DIR}/diffdrive_wheel_robot.urdf.xacro" > "${URDF_DIR}/diffdrive_wheel_robot.urdf"
echo "Compiled diffdrive_wheel_robot.urdf"

