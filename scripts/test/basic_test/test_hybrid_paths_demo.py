#!/usr/bin/env python3
"""
Simple test script to demonstrate the hybrid paths system.

This script:
1. Loads the paths_lib module
2. Calls demo_three_hybrid_paths() to create and visualize three hybrid paths:
   - Rectangle path
   - P trajectory
   - Catenary curve with high alpha
3. Saves plots to /tmp/hybrid_path
"""

import sys
from pathlib import Path

import rospkg

# ROS package path setup
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

# Import the demo function
from paths_lib import demo_three_hybrid_paths


if __name__ == "__main__":
    print("="*60)
    print("Running Hybrid Paths Demo")
    print("="*60)
    
    # Call the demo function
    rectangle_path, p_path, catenary_path = demo_three_hybrid_paths()
    
    print("\nDemo completed successfully!")
    print("Check /tmp/hybrid_path for saved plots:")
    print("  - rectangle_analysis.png")
    print("  - p_trajectory_analysis.png")
    print("  - catenary_analysis.png")
