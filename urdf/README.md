# Creating Pushing Objects for Contact Maintain

This directory contains OBJ mesh files and DXF sketch files for objects used in pushing simulations. This document describes the workflow for creating new objects.

## Copy files from host to docker
docker cp C:\Users\minhd\Downloads\right_triangle.dxf ros-noetic-stable:/home/docker_user/catkin_ws/src/contact_maintain/urdf/right_triangle.dxf

## Workflow for Creating a New Pushing Object

Follow these steps to create a new object that can be used with the `obj_to_generic` conversion system:
Remember, everything is export in (meter) unit

### 1. Draw the Sketch in Fusion 360
- Create a 2D sketch in the **(X, Y) plane** in Fusion 360
- Design your object shape (e.g., bolt, pi symbol, root shape, etc.)
- Ensure the sketch forms a closed loop (polygon)

### 2. Extrude to Z-Dimension
- Extrude the sketch along the **Z-axis** to create a 3D solid
- Typical thickness: ~0.05m (5cm) for pushing objects
- This creates the 3D geometry that will be used in PyBullet

### 3. Find the Center of Mass (CoM)
- Use Fusion 360's **Inspect** tool to find the Center of Mass
- Note the CoM coordinates (X, Y, Z)

### 4. Move Object to Origin
- **Move** the object such that the **CoM is now at the origin (0, 0, 0)**
- This ensures the object's physics center aligns with its geometric center
- Important: The CoM should be at the origin in the final exported files

### 5. Create Top-Face Sketch
- Create a **new sketch** by selecting the **top face** of the extruded object
- This sketch represents the 2D boundary that will be used for `GenericObject` geometry
- Ensure this sketch matches the original (X, Y) plane sketch

### 6. Export Files
Export the OBJ file:

- **OBJ file** (`*.obj`):
  - Export the **3D mesh** (the extruded solid)
  - This will be loaded into PyBullet for physics simulation
  - The 2D boundary vertices will be automatically extracted from the OBJ file by slicing at the bottom (z_min)
  - File: `{shape_name}.obj`

**Note**: DXF files are no longer needed. The system now extracts 2D vertices directly from the OBJ file using `read_obj_to_vertices`, which gives correct vertex coordinates.

### 7. Convert Non-Convex OBJ Files to VHACD (CRITICAL for Non-Convex Shapes)

**⚠️ IMPORTANT**: For **non-convex shapes** (pi, root, bolt, hourglass), you **MUST** convert the OBJ file to a VHACD-decomposed version for proper collision detection in PyBullet.

**What is VHACD?**
- VHACD (Volumetric Hierarchical Approximate Convex Decomposition) decomposes non-convex meshes into convex parts
- PyBullet requires this for accurate collision detection with non-convex objects
- Without VHACD conversion, collision detection will be incorrect or fail

**Conversion Process:**

1. **Backup the original file first** (the script does this automatically):
   ```bash
   cp {shape_name}.obj {shape_name}_original.obj
   ```

2. **Run the conversion script**:
   ```bash
   cd /home/docker_user/catkin_ws/src/contact_maintain/urdf
   python convert_to_vhacd.py --shapes {shape_name}
   ```
   
   Or convert all non-convex shapes at once:
   ```bash
   python convert_to_vhacd.py
   ```

3. **The script will**:
   - Automatically backup the original as `{shape_name}_original.obj` (if not already exists)
   - Convert the OBJ file using PyBullet's VHACD function
   - Overwrite `{shape_name}.obj` with the VHACD-decomposed version
   - Create a log file `log_{shape_name}.txt` with conversion details

**VHACD Parameters:**
- `alpha=0.04`: Controls the accuracy of the decomposition
- `resolution=50000`: Controls the resolution of the decomposition

**Which shapes need VHACD?**
- ✅ **Non-convex shapes** (REQUIRED): `pi`, `root`, `bolt`, `hourglass`
- ❌ **Convex shapes** (NOT needed): `right_triangle`, `rect`, etc.

**Note**: The original OBJ file is preserved as `{shape_name}_original.obj` so you can always revert if needed.

### 8. Test OBJ File Loading (REQUIRED)

**⚠️ CRITICAL**: After exporting and converting (for non-convex shapes), you **MUST** test that the OBJ file loads properly before using it in simulations.

**Test with single pusher:**
```bash
cd /home/docker_user/catkin_ws/src/contact_maintain/scripts/test/basic_test
python test_single_pusher.py --obj-shape {shape_name} --no-gui --duration 5
```

**What to verify:**
- The object loads without errors
- The object appears correctly in the simulation
- Collision detection works (robot can contact the object)
- For non-convex shapes: verify that VHACD conversion worked (collision should be accurate)

**If the object doesn't load or collision is incorrect:**
- For non-convex shapes: Re-run VHACD conversion with `--force` flag
- Check that the OBJ file is valid (can be opened in a 3D viewer)
- Verify that the CoM is at the origin
- Check the log file from VHACD conversion for errors

**Additional alignment test (optional):**
You can also test that the 2D geometry (extracted from OBJ) aligns correctly with the 3D object:

```bash
python3 test_obj_to_generic.py --obj-shape {shape_name} --duration 10
```

**What to verify in alignment test:**
- The red sphere (contact point marker) should stay **exactly on the boundary** of the 3D object
- The green/yellow arrows (normal vectors) should point correctly relative to the object
- The blue arrow (tangent vector) should follow the boundary direction
- As you traverse `t_param` (parameter along the boundary), markers should not drift off the object

**If markers drift:**
- Verify that the CoM is at the origin in the OBJ file
- Check that the OBJ mesh has a clean bottom face (the slicing plane extracts vertices from z_min)
- Ensure the mesh is properly exported from Fusion 360

## File Naming Convention

- Use lowercase with underscores: `right_triangle`, `bolt`, `pi`, `root`
- Only the OBJ file is needed:
  - `{shape_name}.obj` - 3D mesh for PyBullet (2D boundary is extracted automatically)

## Current Objects

- `right_triangle.obj` - Right triangle shape (convex, no VHACD needed)
- `rect.obj` - Rectangle shape (convex, no VHACD needed)
- `bolt.obj` - Bolt/nut shape (non-convex, **VHACD required**)
- `pi.obj` - Pi symbol shape (non-convex, **VHACD required**)
- `root.obj` - Root symbol shape (non-convex, **VHACD required**)
- `hourglass.obj` - Hourglass shape (non-convex, **VHACD required**)

**Note**: 
- DXF files may still exist in this directory but are no longer used by the system.
- Non-convex shapes have `{shape_name}_original.obj` backups of the pre-VHACD versions.

## Usage in Code

Once files are placed in this directory, use them with:

```python
from contact_maintain.object_bridge import obj_to_generic

generic_object, object_uid = obj_to_generic(
    obj_path="bolt.obj",
    shape_name="bolt",
    position=(0, 0, 0.5),
    orientation=0.0,
    mass=1.0,
    lateral_friction=0.8
)
```

The `obj_to_generic` function will:
1. Load the OBJ file into PyBullet for physics simulation
2. Automatically extract 2D vertices from the OBJ file by slicing at the bottom (z_min)
3. Create the `GenericObject` geometry from the extracted vertices
4. Return both the `GenericObject` (for contact point parameterization) and the PyBullet body UID

## Notes

- **CoM at origin is critical**: If the CoM is not at the origin, the 2D geometry and 3D physics will be misaligned, causing markers to drift off the boundary during simulation.
- **OBJ file format**: Standard OBJ format with vertices and faces. PyBullet will handle mesh loading automatically.
- **Material files (.mtl)**: MTL files are optional. Colors are assigned programmatically in `obj_to_generic` for visual distinction.
- **DXF files deprecated**: DXF files are no longer needed. The system extracts 2D vertices directly from OBJ files using `read_obj_to_vertices`, which gives correct coordinates.
- **VHACD for non-convex shapes**: Non-convex shapes (pi, root, bolt, hourglass) **MUST** be converted to VHACD-decomposed versions using `convert_to_vhacd.py` for proper collision detection. The original files are automatically backed up as `{shape_name}_original.obj`.
- **Testing requirement**: Always test OBJ files with `test_single_pusher.py` after conversion to ensure they load properly and collision detection works correctly.

## Known Issues and Workarounds

### ⚠️ DXF File Bug (DEPRECATED - DXF files no longer needed)

**Issue**: The `dxf_to_generic` function has a confirmed bug where it extracts vertices with incorrect signs: all x and y coordinates are negated (i.e., vertices are in (-x, -y) format instead of (x, y)). This was verified by comparing with Fusion 360's actual geometry.

**Status**: **DXF files are no longer required**. The system now uses `read_obj_to_vertices` which extracts 2D vertices directly from the OBJ file by slicing at the bottom (z_min). This method gives the correct vertex coordinates and eliminates the need for DXF files.

**Migration**: 
- The `obj_to_generic` function now uses `read_obj_to_vertices` instead of `dxf_to_generic`
- You only need to export the OBJ file from Fusion 360
- DXF files are no longer used in the pipeline

**Note**: The `dxf_to_generic` function is still present in the codebase but is flagged as deprecated/buggy. It should not be used for new objects.
