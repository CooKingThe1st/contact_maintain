#!/usr/bin/env python3
"""
Test script to compare vertices extracted from OBJ files vs DXF files.

This script:
1. Loads OBJ and DXF files for each shape (right_triangle, bolt, pi, root)
2. Extracts 2D vertices using read_obj_to_vertices (OBJ) and dxf_to_generic (DXF)
3. Compares the vertices to detect orientation/position mismatches
4. Reports any differences found
"""

import sys
from pathlib import Path
import numpy as np

import rospkg

# ROS package path setup
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import read_obj_to_vertices, dxf_to_generic
from shapely.geometry import Polygon


def normalize_vertices(vertices):
    """
    Normalize vertices by:
    1. Centering at origin (subtract centroid)
    2. Optionally sorting by angle to handle different starting points
    """
    vertices = np.array(vertices)
    
    # Center at origin
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    
    return centered.tolist()


def compare_vertex_lists(vertices1, vertices2, name1="Method 1", name2="Method 2", tolerance=1e-3):
    """
    Compare two vertex lists and report differences.
    
    Parameters
    ----------
    vertices1 : list
        First list of (x, y) vertices
    vertices2 : list
        Second list of (x, y) vertices
    name1 : str
        Name for first method
    name2 : str
        Name for second method
    tolerance : float
        Tolerance for vertex position comparison
    
    Returns
    -------
    dict
        Comparison results with differences detected
    """
    v1 = np.array(vertices1)
    v2 = np.array(vertices2)
    
    results = {
        "num_vertices_match": len(v1) == len(v2),
        "num_vertices_1": len(v1),
        "num_vertices_2": len(v2),
        "area_match": False,
        "centroid_match": False,
        "orientation_match": False,
        "vertex_positions_match": False,
        "differences": []
    }
    
    # Check number of vertices
    if not results["num_vertices_match"]:
        results["differences"].append(
            f"Vertex count mismatch: {name1} has {len(v1)} vertices, "
            f"{name2} has {len(v2)} vertices"
        )
        return results
    
    # Normalize both sets (center at origin)
    v1_norm = normalize_vertices(v1)
    v2_norm = normalize_vertices(v2)
    
    # Create polygons for area/orientation comparison
    try:
        poly1 = Polygon(v1_norm)
        poly2 = Polygon(v2_norm)
        
        area1 = abs(poly1.area)
        area2 = abs(poly2.area)
        area_diff = abs(area1 - area2)
        area_relative_diff = area_diff / max(area1, area2) if max(area1, area2) > 0 else 0
        
        results["area_match"] = area_relative_diff < tolerance
        if not results["area_match"]:
            results["differences"].append(
                f"Area mismatch: {name1} area={area1:.6f}, {name2} area={area2:.6f}, "
                f"relative_diff={area_relative_diff:.6f}"
            )
        
        # Check orientation (clockwise vs counter-clockwise)
        # Positive signed area = CCW, negative = CW
        coords1 = np.array(poly1.exterior.coords[:-1])  # Exclude duplicate last point
        coords2 = np.array(poly2.exterior.coords[:-1])
        
        # Calculate signed area using shoelace formula
        def signed_area(coords):
            n = len(coords)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += (coords[j][0] - coords[i][0]) * (coords[j][1] + coords[i][1])
            return area
        
        signed_area1 = signed_area(coords1)
        signed_area2 = signed_area(coords2)
        
        # Same sign = same orientation
        results["orientation_match"] = (signed_area1 * signed_area2) >= 0
        if not results["orientation_match"]:
            results["differences"].append(
                f"Orientation mismatch: {name1} is {'CW' if signed_area1 > 0 else 'CCW'}, "
                f"{name2} is {'CW' if signed_area2 > 0 else 'CCW'}"
            )
        
    except Exception as e:
        results["differences"].append(f"Failed to create polygons for comparison: {e}")
        return results
    
    # Try to match vertices by finding the best rotation/starting point
    # Try all rotations of v2_norm and find the best match
    best_match_error = float('inf')
    best_rotation = 0
    best_match_found = False
    
    for rotation in range(len(v2_norm)):
        # Rotate v2_norm
        v2_rotated = np.roll(v2_norm, rotation, axis=0)
        
        # Try both forward and reversed order
        for reverse in [False, True]:
            if reverse:
                v2_test = np.flipud(v2_rotated)
            else:
                v2_test = v2_rotated
            
            # Calculate mean squared error
            mse = np.mean(np.sum((v1_norm - v2_test)**2, axis=1))
            
            if mse < best_match_error:
                best_match_error = mse
                best_rotation = rotation
                best_match_found = (mse < tolerance)
    
    results["vertex_positions_match"] = best_match_found
    if not results["vertex_positions_match"]:
        results["differences"].append(
            f"Vertex positions mismatch: Best match MSE={best_match_error:.6f} "
            f"(tolerance={tolerance}), best rotation={best_rotation}"
        )
    
    # Check centroid match (should be close to origin after normalization)
    centroid1 = np.mean(v1_norm, axis=0)
    centroid2 = np.mean(v2_norm, axis=0)
    centroid_diff = np.linalg.norm(centroid1 - centroid2)
    results["centroid_match"] = centroid_diff < tolerance
    
    if not results["centroid_match"]:
        results["differences"].append(
            f"Centroid mismatch after normalization: diff={centroid_diff:.6f}"
        )
    
    return results


def main():
    """Main test function."""
    print("="*80)
    print("  OBJ vs DXF Vertex Comparison Test")
    print("="*80)
    print()
    
    # Define shapes to test
    shapes = ["right_triangle", "hourglass", "pi", "root", "rect"]
    
    # Path to urdf directory
    urdf_dir = Path(pkg_path) / "urdf"
    
    if not urdf_dir.exists():
        print(f"✗ Error: URDF directory not found: {urdf_dir}")
        return
    
    print(f"URDF directory: {urdf_dir}")
    print()
    
    all_results = {}
    
    for shape_name in shapes:
        print(f"{'='*80}")
        print(f"  Testing shape: {shape_name}")
        print(f"{'='*80}")
        
        obj_file = urdf_dir / f"{shape_name}.obj"
        dxf_file = urdf_dir / f"{shape_name}.dxf"
        
        # Check if files exist
        if not obj_file.exists():
            print(f"  ✗ OBJ file not found: {obj_file}")
            all_results[shape_name] = {"error": "OBJ file not found"}
            continue
        
        if not dxf_file.exists():
            print(f"  ✗ DXF file not found: {dxf_file}")
            all_results[shape_name] = {"error": "DXF file not found"}
            continue
        
        print(f"  OBJ file: {obj_file}")
        print(f"  DXF file: {dxf_file}")
        print()
        
        # Extract vertices from OBJ
        try:
            print(f"  Extracting vertices from OBJ file...")
            obj_vertices = read_obj_to_vertices(obj_file)
            print(f"    ✓ Extracted {len(obj_vertices)} vertices from OBJ")
            print(f"    The vertices: {obj_vertices}")
            
            # Validate polygon from OBJ vertices
            obj_geometry = Polygon(obj_vertices)
            if not obj_geometry.is_valid or obj_geometry.area <= 0:
                print(f"    ✗ Invalid polygon from OBJ vertices:")
                print(f"      Vertices: {obj_vertices}")
                print(f"      Geometry: {obj_geometry}")
                print(f"      Area: {obj_geometry.area}")
                print(f"      Is valid: {obj_geometry.is_valid}")
                all_results[shape_name] = {"error": f"Invalid polygon from OBJ vertices: area={obj_geometry.area}, is_valid={obj_geometry.is_valid}"}
                
            print(f"    ✓ OBJ polygon is valid (area={obj_geometry.area:.6f})")
        except Exception as e:
            print(f"    ✗ Failed to extract vertices from OBJ: {e}")
            import traceback
            traceback.print_exc()
            all_results[shape_name] = {"error": f"OBJ extraction failed: {e}"}
        
        # Extract vertices from DXF
        try:
            print(f"  Extracting vertices from DXF file...")
            generic_obj = dxf_to_generic(dxf_file, name=shape_name, reverse_sign=True)
            dxf_vertices = list(generic_obj.geometry.exterior.coords[:-1])  # Exclude duplicate last point
            print(f"    ✓ Extracted {len(dxf_vertices)} vertices from DXF")
            print(f"    The vertices: {dxf_vertices}")
            
            # Validate polygon from DXF vertices
            dxf_geometry = Polygon(dxf_vertices)
            if not dxf_geometry.is_valid or dxf_geometry.area <= 0:
                print(f"    ✗ Invalid polygon from DXF vertices:")
                print(f"      Vertices: {dxf_vertices}")
                print(f"      Geometry: {dxf_geometry}")
                print(f"      Area: {dxf_geometry.area}")
                print(f"      Is valid: {dxf_geometry.is_valid}")
                all_results[shape_name] = {"error": f"Invalid polygon from DXF vertices: area={dxf_geometry.area}, is_valid={dxf_geometry.is_valid}"}
                
            print(f"    ✓ DXF polygon is valid (area={dxf_geometry.area:.6f})")
        except Exception as e:
            print(f"    ✗ Failed to extract vertices from DXF: {e}")
            import traceback
            traceback.print_exc()
            all_results[shape_name] = {"error": f"DXF extraction failed: {e}"}
            
        
        print()
        
        # Compare vertices
        print(f"  Comparing vertices...")
        comparison = compare_vertex_lists(
            obj_vertices,
            dxf_vertices,
            name1="OBJ (read_obj_to_vertices)",
            name2="DXF (dxf_to_generic)",
            tolerance=1e-3
        )
        
        # Print results
        print(f"    Number of vertices match: {comparison['num_vertices_match']}")
        print(f"    Area match: {comparison['area_match']}")
        print(f"    Orientation match: {comparison['orientation_match']}")
        print(f"    Vertex positions match: {comparison['vertex_positions_match']}")
        print(f"    Centroid match: {comparison['centroid_match']}")
        
        if comparison['differences']:
            print()
            print(f"    ⚠ Differences detected:")
            for diff in comparison['differences']:
                print(f"      - {diff}")
        else:
            print()
            print(f"    ✓ No differences detected! OBJ and DXF vertices match.")
        
        all_results[shape_name] = comparison
        print()
    
    # Summary
    print()
    print("="*80)
    print("  SUMMARY")
    print("="*80)
    
    for shape_name, result in all_results.items():
        if "error" in result:
            print(f"  {shape_name}: ✗ {result['error']}")
        elif result['differences']:
            print(f"  {shape_name}: ⚠ {len(result['differences'])} difference(s) detected")
        else:
            print(f"  {shape_name}: ✓ No differences")
    
    print()
    print("="*80)
    print("  Test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
