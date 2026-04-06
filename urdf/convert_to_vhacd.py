#!/usr/bin/env python3
"""
Convert non-convex OBJ files to VHACD-decomposed versions.

This script uses PyBullet's VHACD (Volumetric Hierarchical Approximate Convex Decomposition)
to convert non-convex OBJ files into decomposed versions that have proper collision detection.

For non-convex shapes (pi, root, bolt, hourglass), the original OBJ files need to be
converted to VHACD-decomposed versions to ensure proper collision detection in PyBullet.

Usage:
    python convert_to_vhacd.py
    
    # Convert specific shapes:
    python convert_to_vhacd.py --shapes pi root
    
    # Force re-conversion (overwrite existing):
    python convert_to_vhacd.py --force
"""

import argparse
import os
import sys
from pathlib import Path

import pybullet as p

# Non-convex shapes that require VHACD decomposition
# python basic_test/test_single_pusher.py  --kinematics holonomic --approach-distance 0.2 --duration 20 --t-param 0.3 --obj-shape root
NON_CONVEX_SHAPES = ["pi", "root", "bolt", "hourglass"]

# VHACD parameters (match PyBullet documentation)
VHACD_ALPHA = 0.05
VHACD_RESOLUTION = 400000
VHACD_CONCAVITY = 0.0001
VHACD_GAMMA = VHACD_CONCAVITY
# VHACD_MODE = 0
# VHACD_CONVEXHULL_DOWNSAMPLING=16
# VHACD_MAX_NUM_VERTICES_PER_CH=1024
# VHACD_MIN_VOLUME_PER_CH=0
# VHACD_PLANE_DOWNSAMPLING=2
# VHACD_DEPTH = 1


def backup_original(obj_path: Path) -> Path:
    """Backup original OBJ file as {name}_original.obj if it doesn't exist.
    
    Parameters
    ----------
    obj_path : Path
        Path to the original OBJ file
        
    Returns
    -------
    Path
        Path to the backup file (original or newly created)
    """
    original_path = obj_path.parent / f"{obj_path.stem}_original.obj"
    
    # Check for typo version first (hourglass_orginal.obj)
    typo_path = obj_path.parent / f"{obj_path.stem}_orginal.obj"
    
    if typo_path.exists():
        print(f"  Found existing backup with typo: {typo_path.name}")
        # Rename to correct spelling
        if not original_path.exists():
            typo_path.rename(original_path)
            print(f"  ✓ Renamed to: {original_path.name}")
        else:
            print(f"  ✓ Correct backup already exists: {original_path.name}")
    
    if not original_path.exists():
        # Create backup
        import shutil
        shutil.copy2(obj_path, original_path)
        print(f"  ✓ Backed up original to: {original_path.name}")
    else:
        print(f"  ✓ Original backup already exists: {original_path.name}")
    
    return original_path


def convert_to_vhacd(obj_path: Path, force: bool = False) -> bool:
    """Convert OBJ file to VHACD-decomposed version.
    
    IMPORTANT: Always converts from {shape_name}_original.obj, not from the current file.
    This ensures that re-conversions use the original source, not an already-converted file.
    
    Parameters
    ----------
    obj_path : Path
        Path to the OBJ file to convert (e.g., pi.obj)
    force : bool
        If True, overwrite existing VHACD file
        
    Returns
    -------
    bool
        True if conversion was successful, False otherwise
    """
    # Determine original file path
    original_path = obj_path.parent / f"{obj_path.stem}_original.obj"
    
    # Check for typo version first (hourglass_orginal.obj)
    typo_path = obj_path.parent / f"{obj_path.stem}_orginal.obj"
    
    if typo_path.exists() and not original_path.exists():
        print(f"  Found backup with typo: {typo_path.name}")
        typo_path.rename(original_path)
        print(f"  ✓ Renamed to: {original_path.name}")
    
    # Ensure original exists (backup if needed)
    if not original_path.exists():
        if obj_path.exists():
            # Backup current file as original
            backup_original(obj_path)
        else:
            print(f"  ✗ Neither {obj_path.name} nor {original_path.name} found!")
            print(f"     Please ensure the original OBJ file exists.")
            return False
    
    # Verify original file exists
    if not original_path.exists():
        print(f"  ✗ Original file not found: {original_path}")
        return False
    
    # Check if VHACD version already exists
    if obj_path.exists() and not force:
        # Check if file was already converted (VHACD files are typically larger)
        file_size = obj_path.stat().st_size
        original_size = original_path.stat().st_size
        
        # If converted file is significantly different in size, assume it's already converted
        # This is a heuristic - VHACD files are usually larger
        if abs(file_size - original_size) > 100:  # More than 100 bytes difference
            print(f"  ⚠ VHACD file may already exist (size differs from original)")
            print(f"     Original: {original_size} bytes, Current: {file_size} bytes")
            response = input(f"     Re-convert from original? [y/N]: ").strip().lower()
            if response != 'y':
                print(f"  ⊘ Skipping: {obj_path.name}")
                return True
    
    print(f"  Converting: {obj_path.name}...")
    print(f"    Source: {original_path.name} (original)")
    print(f"    Target: {obj_path.name} (VHACD-decomposed)")
    
    # Connect to PyBullet in DIRECT mode (headless)
    client_id = p.connect(p.DIRECT)
    
    try:
        # Convert using VHACD - ALWAYS from original, never from already-converted file
        name_in = str(original_path.absolute())  # Input: original file
        name_out = str(obj_path.absolute())      # Output: overwrite current file
        name_log = str(obj_path.parent / f"log_{obj_path.stem}.txt")
        
        print(f"    Input: {name_in}")
        print(f"    Output: {name_out}")
        print(f"    Log: {name_log}")
        print(f"    Parameters: alpha={VHACD_ALPHA}, resolution={VHACD_RESOLUTION}, concavity={VHACD_CONCAVITY}")
        
        # Call VHACD (may not return a meaningful value)
        p.vhacd(
            name_in,
            name_out,
            name_log,
            alpha=VHACD_ALPHA,
            resolution=VHACD_RESOLUTION,
            # concavity=VHACD_CONCAVITY,
            # gamma=VHACD_GAMMA,
            # mode=VHACD_MODE,
            # convexhullDownsampling=VHACD_CONVEXHULL_DOWNSAMPLING,
            # maxNumVerticesPerCH=VHACD_MAX_NUM_VERTICES_PER_CH,
            # minVolumePerCH=VHACD_MIN_VOLUME_PER_CH,
            # planeDownsampling=VHACD_PLANE_DOWNSAMPLING,
            # depth=VHACD_DEPTH
            # convexhullApproximation=0
        )
        
        # Check success by verifying output file exists and is valid
        # VHACD may not return a meaningful boolean, so we check the file
        output_path = Path(name_out)
        if output_path.exists():
            output_size = output_path.stat().st_size
            original_size = original_path.stat().st_size
            
            # VHACD files are typically larger than originals, but at least should exist
            if output_size > 0:
                print(f"  ✓ Successfully converted: {obj_path.name}")
                print(f"    Original size: {original_size} bytes")
                print(f"    VHACD size: {output_size} bytes")
                if Path(name_log).exists():
                    log_size = Path(name_log).stat().st_size
                    print(f"    Log file size: {log_size} bytes")
                return True
            else:
                print(f"  ✗ VHACD conversion failed: output file is empty")
                return False
        else:
            print(f"  ✗ VHACD conversion failed: output file was not created")
            print(f"     Expected: {name_out}")
            # Check log file for error details
            if Path(name_log).exists():
                log_size = Path(name_log).stat().st_size
                print(f"     Log file exists ({log_size} bytes) - check for error details")
                if log_size > 0:
                    try:
                        with open(name_log, 'r') as f:
                            log_content = f.read()
                            if log_content:
                                print(f"     Last 500 chars of log:")
                                print(f"     {log_content[-500:]}")
                    except:
                        pass
            return False
            
    except Exception as e:
        print(f"  ✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        p.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Convert non-convex OBJ files to VHACD-decomposed versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all non-convex shapes:
    python convert_to_vhacd.py
    
    # Convert specific shapes:
    python convert_to_vhacd.py --shapes pi root
    
    # Force re-conversion (overwrite existing):
    python convert_to_vhacd.py --force
    
    # Convert specific shapes with force:
    python convert_to_vhacd.py --shapes bolt hourglass --force
        """
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=NON_CONVEX_SHAPES,
        choices=NON_CONVEX_SHAPES,
        help=f"Shapes to convert (default: all: {', '.join(NON_CONVEX_SHAPES)})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if VHACD file already exists"
    )
    args = parser.parse_args()
    
    # Get script directory (should be urdf directory)
    script_dir = Path(__file__).parent
    print("="*60)
    print("VHACD Conversion for Non-Convex OBJ Files")
    print("="*60)
    print(f"Working directory: {script_dir}")
    print(f"Shapes to convert: {', '.join(args.shapes)}")
    print(f"Force mode: {args.force}")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for shape_name in args.shapes:
        obj_file = script_dir / f"{shape_name}.obj"
        print(f"\n[{shape_name}]")
        
        if convert_to_vhacd(obj_file, force=args.force):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print("="*60)
    
    if fail_count > 0:
        print("\n⚠ Some conversions failed. Check the log files for details.")
        sys.exit(1)
    else:
        print("\n✓ All conversions completed successfully!")
        print("\nNext step: Test the converted OBJ files with:")
        print("  python test_single_pusher.py --obj-shape <shape_name> --no-gui")


if __name__ == "__main__":
    main()
