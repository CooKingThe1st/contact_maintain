Final Implementation Plan
1) Product Goal and Scope
Build an interactive planning workbench with:
Two planners:
Holonomic -> A\*
Car-like -> Hybrid A\*
Obstacle editing:
Simple mode: rectangle drag
Complex mode: freehand line drawing
Textbox inputs for start/goal pose
Manual Replan (full re-solve every time)
Save/load/export scenarios
Zoom/pan support
Fixed canvas size, but map dimensions configurable via textbox
No incremental replanning (Anytime D* logic is reference-only).
2) High-Level Modules
Create a small app package (e.g. scripts/test/planning_gui/):
app.py - app bootstrap, event wiring
map_model.py - world/map state and transforms
obstacle_model.py - rect/line obstacle entities + export IDs
planner_adapter.py - A\*/Hybrid A\* wrapper
gui_view.py - canvas + controls + render loop
io_service.py - save/load/export JSON
3) Core Data Model
Use explicit models:
MapState
map_width, map_height (world units or cells)
resolution
occupancy representation
ViewState
fixed canvas pixel size (e.g. 1000x700)
zoom
pan_x, pan_y
RobotState
robot_type (holonomic/car)
footprint size (width, length)
PoseState
start (x,y,yaw_deg)
goal (x,y,yaw_deg)
ObstacleState
list of rect and line objects with unique IDs:
RECT_###
LINE_###
4) Fixed Canvas + Variable Map (your key requirement)
Implement a world-to-view transform layer:
Canvas dimensions stay constant
Map size can change from textboxes (map_width, map_height, maybe resolution)
Drawing always occurs in world coordinates through inverse transform from mouse position
Zoom/pan changes view transform only, not map data
This allows very large maps while keeping UI stable.
5) Zoom / Pan Feature
Implement in the canvas controller:
Mouse wheel: zoom in/out around cursor focus point
Middle-drag (or Shift+drag): pan
Reset View button to fit map to canvas
Optional min/max zoom clamp
matplotlib is fine for MVP and fast to build; custom canvas not required initially.
6) Editing Modes
A) Simple rectangle mode
Mouse drag defines axis-aligned rectangle
Create obstacle with RECT_###
Optional snapping to grid/cell resolution
B) Complex line mode
Mouse-drag freehand polyline (or click-segment mode)
Rasterize to occupancy with configurable thickness
Create obstacle with LINE_###
C) Erase support
Erase by selection or brush
Removing an obstacle removes its object entry and occupancy contribution
7) Planner Integration
A) Holonomic mode
Grid A\* with occupancy + robot footprint inflation/collision
Use as primary planner for your actual problem
B) Car mode
Hybrid A\* adapter using MotionPlanning/HybridAstarPlanner/hybrid_astar.py
C) Replan pipeline
On Replan click:
Validate map and pose inputs
Ensure start/goal not in collision
Build planner input from current map/robot type
Full solve from scratch
Render path and status metrics
8) Textbox Controls
Right panel textboxes:
Start pose: sx, sy, syaw
Goal pose: gx, gy, gyaw
Map size: map_width, map_height (+ optional resolution)
Robot size: robot_width, robot_length
Draw thickness / rectangle settings
Buttons:
Apply Pose
Apply Map Size
Apply Robot Size
Replan
Save
Load
Export Obstacles
Clear Obstacles
Reset View
Visual markers:
Start: green dot + arrow
Goal: red dot + arrow
9) Save/Load/Export
Scenario Save/Load JSON
Include:
map config
view-independent obstacle data (rect/line geometry)
start/goal poses
robot mode + size
metadata/version
Obstacle Export
Preserve prefixes:
RECT_... entries for simple obstacles
LINE_... entries for complex obstacles
10) Reference Usage Plan (explicit)
Use these repos as implementation references:
Hybrid A* integration reference:
MotionPlanning/HybridAstarPlanner/hybrid_astar.py
A/search patterns + map modeling ideas:
PathPlanning/Search_based_Planning/... A* modules
Obstacle update/visual refresh patterns only (not algorithm reuse):
PathPlanning/Search_based_Planning/Search_2D/Anytime_D_star.py
11) Milestone Plan
M1: fixed canvas + variable map transforms + zoom/pan
M2: rectangle/line obstacle editing with IDs and erase
M3: pose/map/robot textbox controls + validation
M4: A\* planner integration + path rendering
M5: Hybrid A\* mode integration
M6: save/load/export + metrics + UX polish
12) Acceptance Criteria (ready-to-code checklist)
Map size can be changed without resizing canvas
User can zoom/pan and still draw accurately
Both obstacle modes work and export with RECT_ / LINE_
Start/goal via textbox display as dot+arrow
Replan works for both A\* and Hybrid A\*
Save/load round-trip reproduces same map and path behavior
This plan is now implementation-ready.
