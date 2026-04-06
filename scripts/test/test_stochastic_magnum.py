#!/usr/bin/env python3
"""
Comprehensive test script for find_the_magnum_stochastic.

Tests the stochastic Latin square-based search on all standard objects
and records success rate, elapsed time, and other statistics.
"""

import sys
import time
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np

import rospkg

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))
sys.path.insert(0, str(pkg_path / "src" / "legacy"))

from contact_optimizer_utils_test_ver import find_the_magnum_stochastic  # noqa: E402
from object_utils import create_standard_objects, WrenchSpaceVisualizer  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402


def visualize_stochastic_solution(obj, contacts, shape_name, save_path, threshold=1.0, force_range_scalar=2.0,
                                   enable_tangent_forces=False):
    """
    Visualize the object shape with contact points and 2D wrench space projections.
    
    Args:
        obj: GenericObject instance
        contacts: List of ContactPoint objects
        shape_name: Name of the shape (for title)
        save_path: Path to save the figure
        threshold: LS coverage threshold (default 1.0)
        force_range_scalar: Force range scalar (default 2.0)
        enable_tangent_forces: If True, build GWS with tangent forces (match solution found via tangent fallback).
    """
    # Create 2x2 subplot layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # =====================================================================
    # TOP LEFT: Object with contact points
    # =====================================================================
    ax_obj = fig.add_subplot(gs[0, 0])
    
    # Draw object
    obj.visualize(ax=ax_obj, alpha=0.3, facecolor='lightcyan', show_frame=True)
    
    # Color map for contact points
    contact_colors = ['red', 'blue', 'green', 'purple']
    
    # Draw contact points and normal vectors
    for i, contact in enumerate(contacts):
        pos = contact.position
        normal = contact.normal_inward
        
        # Choose color
        color = contact_colors[i % len(contact_colors)]
        
        # Draw contact point (numbered marker)
        ax_obj.plot(pos[0], pos[1], 'o', 
                color=color, 
                markersize=14,
                markeredgecolor='black',
                markeredgewidth=2,
                alpha=0.9,
                label=f'Contact {i+1}' if i < 4 else '')
        
        # Add number label
        ax_obj.text(pos[0], pos[1], str(i+1),
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                color='white')
        
        # Draw normal force vector (inward)
        force_scale = 0.1
        force_vec = normal * force_scale
        ax_obj.arrow(pos[0], pos[1], 
                force_vec[0], force_vec[1],
                head_width=0.02, 
                head_length=0.02,
                linewidth=2.5,
                fc=color, 
                ec='black',
                alpha=0.8)
    
    # Add centroid
    centroid = obj.get_centroid()
    ax_obj.plot(centroid.x, centroid.y,
            'x',
            color='black',
            markersize=12,
            markeredgewidth=2,
            label='Centroid')
    
    ax_obj.set_xlabel('X (m)', fontsize=11)
    ax_obj.set_ylabel('Y (m)', fontsize=11)
    ax_obj.set_title(f'Object: {shape_name.upper()}', fontsize=12, fontweight='bold')
    ax_obj.legend(loc='upper right', fontsize=9)
    ax_obj.grid(True, alpha=0.3)
    ax_obj.axis('equal')
    
    # =====================================================================
    # Calculate wrench space and limit surface
    # =====================================================================
    wrench_visualizer = WrenchSpaceVisualizer()
    
    # Calculate maximum force based on object's static friction
    normal_force = obj.mass * 9.81
    static_f_max = obj.static_friction * normal_force
    max_force = force_range_scalar * static_f_max
    
    # Calculate wrench space (use same tangent-forces setting as the sufficiency check that accepted this solution)
    wrench_data = wrench_visualizer.calculate_wrench_space(
        contacts,
        force_ranges=[(0.0, max_force)] * len(contacts),
        sampling_density=3,
        enable_tangent_forces=enable_tangent_forces,
    )
    
    wrenches = wrench_data['wrenches']
    feasible_mask = wrench_data['feasible_mask']
    feasible_wrenches = wrenches[feasible_mask] if len(wrenches) > 0 else np.array([])
    
    # Calculate limit surface
    ls_data = wrench_visualizer.calculate_limit_surface(
        obj,
        resolution=50,
        scaling_factor=threshold,
        grid_size=30,
    )
    
    f_max = ls_data['f_max']
    m_max = ls_data['m_max']
    
    # =====================================================================
    # TOP RIGHT: Fx vs Fy projection
    # =====================================================================
    ax_xy = fig.add_subplot(gs[0, 1])
    
    if len(feasible_wrenches) > 0:
        # Plot GWS points
        ax_xy.scatter(feasible_wrenches[:, 0], feasible_wrenches[:, 1],
                     c='green', s=10, alpha=0.5, label='GWS Points')
        
        # Plot GWS convex hull
        if len(feasible_wrenches) > 4:
            try:
                hull_xy = ConvexHull(feasible_wrenches[:, :2])
                hull_points_xy = feasible_wrenches[hull_xy.vertices, :2]
                hull_points_xy = np.vstack([hull_points_xy, hull_points_xy[0]])  # Close loop
                ax_xy.plot(hull_points_xy[:, 0], hull_points_xy[:, 1],
                          'g-', linewidth=2, label='GWS Hull')
            except:
                pass
    
    # Plot Limit Surface (circle for Fx-Fy)
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = f_max * np.cos(theta)
    y_circle = f_max * np.sin(theta)
    ax_xy.plot(x_circle, y_circle, 'b--', linewidth=2, label=f'LS (f_max={f_max:.2f}N)')
    
    # Origin
    ax_xy.plot(0, 0, 'ko', markersize=6, label='Origin')
    
    ax_xy.set_xlabel('Fx (N)', fontsize=11)
    ax_xy.set_ylabel('Fy (N)', fontsize=11)
    ax_xy.set_title('Fx vs Fy Projection', fontsize=12, fontweight='bold')
    ax_xy.legend(fontsize=9)
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis('equal')
    
    # =====================================================================
    # BOTTOM LEFT: Fx vs Torque projection
    # =====================================================================
    ax_xt = fig.add_subplot(gs[1, 0])
    
    if len(feasible_wrenches) > 0:
        # Plot GWS points
        ax_xt.scatter(feasible_wrenches[:, 0], feasible_wrenches[:, 2],
                     c='green', s=10, alpha=0.5, label='GWS Points')
        
        # Plot GWS convex hull
        if len(feasible_wrenches) > 4:
            try:
                hull_xt = ConvexHull(feasible_wrenches[:, [0, 2]])
                hull_points_xt = feasible_wrenches[hull_xt.vertices, :][:, [0, 2]]
                hull_points_xt = np.vstack([hull_points_xt, hull_points_xt[0]])  # Close loop
                ax_xt.plot(hull_points_xt[:, 0], hull_points_xt[:, 1],
                          'g-', linewidth=2, label='GWS Hull')
            except:
                pass
    
    # Plot Limit Surface (ellipse for Fx-Torque)
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    x_ellipse = f_max * np.cos(theta)
    z_ellipse = m_max * np.sin(theta)
    ax_xt.plot(x_ellipse, z_ellipse, 'b--', linewidth=2, 
              label=f'LS (f_max={f_max:.2f}N, m_max={m_max:.2f}Nm)')
    
    # Origin
    ax_xt.plot(0, 0, 'ko', markersize=6, label='Origin')
    
    ax_xt.set_xlabel('Fx (N)', fontsize=11)
    ax_xt.set_ylabel('Torque (N⋅m)', fontsize=11)
    ax_xt.set_title('Fx vs Torque Projection', fontsize=12, fontweight='bold')
    ax_xt.legend(fontsize=9)
    ax_xt.grid(True, alpha=0.3)
    
    # =====================================================================
    # BOTTOM RIGHT: Fy vs Torque projection
    # =====================================================================
    ax_yt = fig.add_subplot(gs[1, 1])
    
    if len(feasible_wrenches) > 0:
        # Plot GWS points
        ax_yt.scatter(feasible_wrenches[:, 1], feasible_wrenches[:, 2],
                     c='green', s=10, alpha=0.5, label='GWS Points')
        
        # Plot GWS convex hull
        if len(feasible_wrenches) > 4:
            try:
                hull_yt = ConvexHull(feasible_wrenches[:, [1, 2]])
                hull_points_yt = feasible_wrenches[hull_yt.vertices, :][:, [1, 2]]
                hull_points_yt = np.vstack([hull_points_yt, hull_points_yt[0]])  # Close loop
                ax_yt.plot(hull_points_yt[:, 0], hull_points_yt[:, 1],
                          'g-', linewidth=2, label='GWS Hull')
            except:
                pass
    
    # Plot Limit Surface (ellipse for Fy-Torque, same shape as Fx-Torque)
    ax_yt.plot(x_ellipse, z_ellipse, 'b--', linewidth=2,
              label=f'LS (f_max={f_max:.2f}N, m_max={m_max:.2f}Nm)')
    
    # Origin
    ax_yt.plot(0, 0, 'ko', markersize=6, label='Origin')
    
    ax_yt.set_xlabel('Fy (N)', fontsize=11)
    ax_yt.set_ylabel('Torque (N⋅m)', fontsize=11)
    ax_yt.set_title('Fy vs Torque Projection', fontsize=12, fontweight='bold')
    ax_yt.legend(fontsize=9)
    ax_yt.grid(True, alpha=0.3)
    
    # Overall title (indicate if solution used tangent-force fallback)
    title = f'Stochastic Magnum Solution: {shape_name.upper()} (Threshold={threshold:.2f})'
    if enable_tangent_forces:
        title += ' [tangent forces]'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 COMPREHENSIVE STOCHASTIC MAGNUM SEARCH TEST")
    print("=" * 80)
    
    # Get all standard objects
    standard_objects = create_standard_objects()
    shape_names = sorted(standard_objects.keys())
    
    print(f"\n📋 Testing {len(shape_names)} shapes: {', '.join(shape_names)}")
    print("\n" + "-" * 80)
    
    # Create output directory for visualizations
    output_dir = Path("/tmp/basic_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving visualizations to: {output_dir}")
    
    # Statistics collection
    results = []
    total_start_time = time.time()
    
    # Test each shape
    for idx, shape_name in enumerate(shape_names, 1):
        obj = standard_objects[shape_name]
        
        print(f"\n[{idx}/{len(shape_names)}] Testing: {shape_name.upper()}")
        print("-" * 80)
        
        # Run stochastic search
        test_start_time = time.time()
        find_magnum_result = find_the_magnum_stochastic(
            obj, 
            verbose=False,  # Set to False to reduce per-shape output
            threshold=1,
            max_batches=20,
            timeout=10.0,
            force_range_scalar=2,
            robot_radius=0.06,
            used_tangent_as_fallback=True
        )
        test_elapsed_time = time.time() - test_start_time
        
        # Extract statistics
        success = find_magnum_result.get('success', False)
        configs_tested = find_magnum_result.get('configs_tested', 0)
        batches_tested = find_magnum_result.get('batches_tested', 0)
        pruned_count = find_magnum_result.get('pruned_count', {})
        found_by = find_magnum_result.get('found_by', None)
        
        # Calculate total pruned
        total_pruned = sum(pruned_count.values()) if pruned_count else 0
        
        # Visualize if solution found
        viz_path = None
        if success:
            contacts = find_magnum_result.get('contacts', [])
            if contacts:
                viz_filename = f"stochastic_{shape_name}.jpg"
                viz_path = output_dir / viz_filename
                try:
                    # Get threshold and force_range_scalar from the search result
                    threshold = find_magnum_result.get('threshold', 1.0)
                    used_tangent_fallback = find_magnum_result.get('used_tangent_fallback', False)
                    # Note: force_range_scalar is not in result, use default 2.0
                    visualize_stochastic_solution(
                        obj, contacts, shape_name, str(viz_path),
                        threshold=threshold, force_range_scalar=2.0,
                        enable_tangent_forces=used_tangent_fallback,
                    )
                    print(f"   💾 Saved visualization: {viz_filename}")
                except Exception as e:
                    print(f"   ⚠️  Visualization failed: {e}")
                    viz_path = None
        
        # Store results
        result = {
            'shape_name': shape_name,
            'success': success,
            'elapsed_time': test_elapsed_time,
            'configs_tested': configs_tested,
            'batches_tested': batches_tested,
            'total_pruned': total_pruned,
            'pruned_count': pruned_count.copy(),
            'found_by': found_by,
            'viz_path': str(viz_path) if viz_path else None,
        }
        results.append(result)
        
        # Print per-shape summary
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status} | Time: {test_elapsed_time:.3f}s | Configs: {configs_tested} | Batches: {batches_tested}")
        if total_pruned > 0:
            print(f"   Pruned: {total_pruned} ({pruned_count})")
    
    total_elapsed_time = time.time() - total_start_time
    
    # =====================================================================
    # SUMMARY STATISTICS
    # =====================================================================
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)
    
    # Calculate aggregate statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    success_rate = len(successful) / len(results) * 100 if results else 0
    
    # Time statistics
    all_times = [r['elapsed_time'] for r in results]
    successful_times = [r['elapsed_time'] for r in successful]
    failed_times = [r['elapsed_time'] for r in failed]
    
    # Config statistics
    all_configs = [r['configs_tested'] for r in results]
    successful_configs = [r['configs_tested'] for r in successful]
    
    # Pruning statistics
    all_pruned = [r['total_pruned'] for r in results]
    
    print(f"\n📈 Overall Results:")
    print(f"   Total shapes tested    : {len(results)}")
    print(f"   Successful            : {len(successful)} ({success_rate:.1f}%)")
    print(f"   Failed                 : {len(failed)} ({100-success_rate:.1f}%)")
    print(f"   Total elapsed time    : {total_elapsed_time:.3f} s")
    print(f"   Average time per shape: {sum(all_times)/len(all_times):.3f} s" if all_times else "   N/A")
    
    if successful:
        print(f"\n✅ Successful Searches:")
        print(f"   Average time          : {sum(successful_times)/len(successful_times):.3f} s")
        print(f"   Min time               : {min(successful_times):.3f} s")
        print(f"   Max time               : {max(successful_times):.3f} s")
        print(f"   Average configs tested: {sum(successful_configs)/len(successful_configs):.1f}")
        print(f"   Min configs            : {min(successful_configs)}")
        print(f"   Max configs            : {max(successful_configs)}")
    
    if failed:
        print(f"\n❌ Failed Searches:")
        print(f"   Average time           : {sum(failed_times)/len(failed_times):.3f} s")
        print(f"   Shapes: {', '.join([r['shape_name'] for r in failed])}")
    
    if all_pruned:
        print(f"\n🔍 Pruning Statistics:")
        print(f"   Average pruned per test: {sum(all_pruned)/len(all_pruned):.1f}")
        print(f"   Max pruned             : {max(all_pruned)}")
    
    # =====================================================================
    # DETAILED RESULTS TABLE
    # =====================================================================
    print("\n" + "=" * 80)
    print("📋 DETAILED RESULTS")
    print("=" * 80)
    print(f"\n{'Shape':<20} {'Status':<10} {'Time (s)':<10} {'Configs':<10} {'Batches':<10} {'Pruned':<10}")
    print("-" * 80)
    
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        print(f"{r['shape_name']:<20} {status:<10} {r['elapsed_time']:<10.3f} "
              f"{r['configs_tested']:<10} {r['batches_tested']:<10} {r['total_pruned']:<10}")
    
    print("\n" + "=" * 80)
    print(f"✅ Test suite completed in {total_elapsed_time:.3f} seconds")
    print("=" * 80)
