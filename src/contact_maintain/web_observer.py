"""
Web-based observer for contact maintenance simulation.

Provides real-time monitoring via Flask + SocketIO web dashboard.
Based on the Webots observer pattern.
"""
import io
import base64
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

import numpy as np

# Matplotlib with non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from flask import Flask, jsonify, request
from flask_socketio import SocketIO

# ============================================================================
# CONFIGURATION
# ============================================================================

PLOT_HISTORY_LENGTH = 500  # Number of points to keep in plots
UPDATE_RATE = 10  # Hz for web updates
WEB_PORT = 5000

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RobotState:
    """Current state of a robot."""
    name: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))  # x, y
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # vx, vy, omega
    heading: float = 0.0  # theta
    bumper_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Contact state
    in_contact: bool = False
    contact_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    contact_force_magnitude: float = 0.0
    contact_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Pushing direction (direction robot is pushing toward object)
    pushing_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pushing_angle: float = 0.0  # angle of pushing direction
    
    # Velocity command from controller
    cmd_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # cmd vx, vy, omega
    
    def to_dict(self):
        return {
            'name': self.name,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'heading': float(self.heading),
            'heading_deg': float(np.degrees(self.heading)),
            'bumper_position': self.bumper_position.tolist(),
            'in_contact': self.in_contact,
            'contact_force': self.contact_force.tolist(),
            'contact_force_magnitude': float(self.contact_force_magnitude),
            'contact_direction': self.contact_direction.tolist(),
            'pushing_direction': self.pushing_direction.tolist(),
            'pushing_angle': float(self.pushing_angle),
            'pushing_angle_deg': float(np.degrees(self.pushing_angle)),
            'cmd_velocity': self.cmd_velocity.tolist(),
        }


@dataclass
class ObjectState:
    """Current state of an object."""
    name: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # x, y, z
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # vx, vy, vz
    orientation: float = 0.0  # theta (z-rotation)
    angular_velocity: float = 0.0  # omega_z
    
    def to_dict(self):
        return {
            'name': self.name,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'orientation': float(self.orientation),
            'orientation_deg': float(np.degrees(self.orientation)),
            'angular_velocity': float(self.angular_velocity),
        }


@dataclass
class RobotHistory:
    """Time-series data for a robot."""
    name: str
    timestamps: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    positions: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    velocities: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    headings: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    in_contacts: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    contact_forces: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    pushing_angles: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    cmd_velocities: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))  # (vx, vy, omega)
    
    def add_sample(self, t, state: RobotState):
        self.timestamps.append(t)
        self.positions.append(state.position.copy())
        self.velocities.append(np.linalg.norm(state.velocity[:2]))
        self.headings.append(state.heading)
        self.in_contacts.append(state.in_contact)
        self.contact_forces.append(state.contact_force_magnitude)
        self.pushing_angles.append(state.pushing_angle)
        self.cmd_velocities.append(state.cmd_velocity.copy())


@dataclass
class ObjectHistory:
    """Time-series data for an object."""
    name: str
    timestamps: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    positions: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    velocities: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))  # magnitude
    velocities_xy: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))  # (vx, vy)
    angular_velocities: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))  # omega
    orientations: deque = field(default_factory=lambda: deque(maxlen=PLOT_HISTORY_LENGTH))
    
    def add_sample(self, t, state: ObjectState):
        self.timestamps.append(t)
        self.positions.append(state.position[:2].copy())
        self.velocities.append(np.linalg.norm(state.velocity[:2]))
        self.velocities_xy.append(state.velocity[:2].copy())
        self.angular_velocities.append(state.angular_velocity)
        self.orientations.append(state.orientation)


# ============================================================================
# WEB OBSERVER CLASS
# ============================================================================

class WebObserver:
    """Observer that streams data to web interface."""
    
    def __init__(self, port=WEB_PORT):
        self.port = port
        
        # Current state
        self.robots: Dict[str, RobotState] = {}
        self.objects: Dict[str, ObjectState] = {}
        
        # History for plotting
        self.robot_history: Dict[str, RobotHistory] = {}
        self.object_history: Dict[str, ObjectHistory] = {}
        
        # Metrics
        self.contact_maintained_time = 0.0
        self.contact_loss_count = 0
        self.last_contact_state: Dict[str, bool] = {}
        self.simulation_time = 0.0
        self.last_update_time = 0.0
        
        # Selected entities for focused view
        self.selected_robot = None
        self.selected_object = None
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'contact_maintain_observer'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", 
                                  async_mode='threading', logger=False, 
                                  engineio_logger=False)
        
        self._setup_routes()
        
        # Threading
        self.running = False
        self.web_thread = None
        self.update_thread = None
        self._lock = threading.Lock()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return self._get_dashboard_html()
        
        @self.app.route('/api/state')
        def get_state():
            with self._lock:
                return jsonify({
                    'robots': {k: v.to_dict() for k, v in self.robots.items()},
                    'objects': {k: v.to_dict() for k, v in self.objects.items()},
                    'metrics': self._get_metrics(),
                    'time': self.simulation_time,
                })
        
        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(self._get_metrics())
        
        @self.app.route('/api/set_selection', methods=['POST'])
        def set_selection():
            data = request.json
            if 'robot' in data:
                self.selected_robot = data['robot']
            if 'object' in data:
                self.selected_object = data['object']
            return jsonify({'status': 'ok'})
        
        @self.socketio.on('connect')
        def handle_connect():
            print('[Web] Client connected')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('[Web] Client disconnected')
    
    def register_robot(self, name: str):
        """Register a robot to track."""
        with self._lock:
            self.robots[name] = RobotState(name=name)
            self.robot_history[name] = RobotHistory(name=name)
            self.last_contact_state[name] = False
            if self.selected_robot is None:
                self.selected_robot = name
            print(f"[Observer] Registered robot: {name}")
    
    def register_object(self, name: str):
        """Register an object to track."""
        with self._lock:
            self.objects[name] = ObjectState(name=name)
            self.object_history[name] = ObjectHistory(name=name)
            if self.selected_object is None:
                self.selected_object = name
            print(f"[Observer] Registered object: {name}")
    
    def update_robot(self, name: str, position, velocity, heading, bumper_pos,
                    in_contact, contact_force, timestamp, object_position=None,
                    cmd_velocity=None):
        """Update robot state.
        
        Parameters
        ----------
        object_position : array-like, optional
            Position of the object being pushed (for computing pushing direction).
        cmd_velocity : array-like, optional
            Velocity command from controller (vx, vy, omega).
        """
        with self._lock:
            if name not in self.robots:
                self.register_robot(name)
            
            state = self.robots[name]
            state.position = np.array(position)
            state.velocity = np.array(velocity)
            state.heading = float(heading)
            state.bumper_position = np.array(bumper_pos)
            state.in_contact = bool(in_contact)
            state.contact_force = np.array(contact_force)
            state.contact_force_magnitude = float(np.linalg.norm(contact_force[:2]))
            if state.contact_force_magnitude > 0:
                state.contact_direction = contact_force[:2] / state.contact_force_magnitude
            else:
                state.contact_direction = np.zeros(2)
            
            # Compute pushing direction (from robot toward object)
            if object_position is not None:
                obj_pos = np.array(object_position)[:2]
                to_object = obj_pos - state.position
                dist = np.linalg.norm(to_object)
                if dist > 0.01:
                    state.pushing_direction = to_object / dist
                    state.pushing_angle = np.arctan2(to_object[1], to_object[0])
                else:
                    state.pushing_direction = np.array([np.cos(heading), np.sin(heading)])
                    state.pushing_angle = heading
            else:
                # Default to robot heading direction
                state.pushing_direction = np.array([np.cos(heading), np.sin(heading)])
                state.pushing_angle = heading
            
            # Store velocity command
            if cmd_velocity is not None:
                state.cmd_velocity = np.array(cmd_velocity)
            else:
                state.cmd_velocity = np.zeros(3)
            
            # Update history
            self.robot_history[name].add_sample(timestamp, state)
            
            # Track contact metrics
            dt = timestamp - self.last_update_time if self.last_update_time > 0 else 0
            was_in_contact = self.last_contact_state.get(name, False)
            
            if in_contact:
                self.contact_maintained_time += dt
            
            if was_in_contact and not in_contact:
                self.contact_loss_count += 1
            
            self.last_contact_state[name] = in_contact
            self.simulation_time = timestamp
            self.last_update_time = timestamp
    
    def update_object(self, name: str, position, velocity, orientation, 
                     angular_velocity, timestamp):
        """Update object state."""
        with self._lock:
            if name not in self.objects:
                self.register_object(name)
            
            state = self.objects[name]
            state.position = np.array(position)
            state.velocity = np.array(velocity)
            state.orientation = float(orientation)
            state.angular_velocity = float(angular_velocity)
            
            # Update history
            self.object_history[name].add_sample(timestamp, state)
    
    def _get_metrics(self):
        """Get evaluation metrics."""
        metrics = {
            'simulation_time': self.simulation_time,
            'contact_maintained_time': self.contact_maintained_time,
            'contact_loss_count': self.contact_loss_count,
            'active_robots': len(self.robots),
            'active_objects': len(self.objects),
            'selected_robot': self.selected_robot,
            'selected_object': self.selected_object,
        }
        
        # Add per-robot contact status and pushing direction
        for name, state in self.robots.items():
            metrics[f'{name}_contact'] = state.in_contact
            metrics[f'{name}_force'] = state.contact_force_magnitude
            metrics[f'{name}_push_angle'] = np.degrees(state.pushing_angle)
        
        return metrics
    
    def _create_plots(self) -> Optional[str]:
        """Create matplotlib plots and return as base64 image."""
        try:
            with self._lock:
                robot_name = self.selected_robot
                obj_name = self.selected_object
                
                if robot_name not in self.robot_history:
                    return None
                if obj_name not in self.object_history:
                    return None
                
                robot_hist = self.robot_history[robot_name]
                obj_hist = self.object_history[obj_name]
                
                if len(robot_hist.timestamps) < 2:
                    return None
                
                # Copy data for thread safety
                times = list(robot_hist.timestamps)
                robot_positions = list(robot_hist.positions)
                robot_velocities = list(robot_hist.velocities)
                robot_headings = list(robot_hist.headings)
                robot_contacts = list(robot_hist.in_contacts)
                robot_forces = list(robot_hist.contact_forces)
                robot_pushing_angles = list(robot_hist.pushing_angles)
                robot_cmd_velocities = list(robot_hist.cmd_velocities)
                
                obj_positions = list(obj_hist.positions)
                obj_velocities = list(obj_hist.velocities)
                obj_velocities_xy = list(obj_hist.velocities_xy)
                obj_angular_velocities = list(obj_hist.angular_velocities)
                obj_orientations = list(obj_hist.orientations)
                
                robot_state = self.robots[robot_name]
                obj_state = self.objects[obj_name]
            
            fig = Figure(figsize=(18, 12))
            
            # 1. Trajectory plot with pushing direction (top-left)
            ax1 = fig.add_subplot(2, 4, 1)
            if len(robot_positions) > 0:
                robot_x = [p[0] for p in robot_positions]
                robot_y = [p[1] for p in robot_positions]
                ax1.plot(robot_x, robot_y, 'b-', linewidth=2, label=robot_name)
                ax1.plot(robot_x[-1], robot_y[-1], 'bo', markersize=10)
                
                # Draw current pushing direction arrow
                if len(robot_positions) > 0:
                    arrow_scale = 0.3
                    ax1.arrow(robot_x[-1], robot_y[-1],
                             robot_state.pushing_direction[0] * arrow_scale,
                             robot_state.pushing_direction[1] * arrow_scale,
                             head_width=0.05, head_length=0.03, fc='green', ec='green',
                             linewidth=2, label='Push Dir')
            
            if len(obj_positions) > 0:
                obj_x = [p[0] for p in obj_positions]
                obj_y = [p[1] for p in obj_positions]
                ax1.plot(obj_x, obj_y, 'r-', linewidth=2, label=obj_name)
                ax1.plot(obj_x[-1], obj_y[-1], 'rs', markersize=10)
            
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.set_title('Trajectory + Push Direction')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # 2. Robot Velocity: Actual vs Command (top-middle-left)
            ax2 = fig.add_subplot(2, 4, 2)
            ax2.plot(times, robot_velocities, 'b-', linewidth=2, label='Actual |v|')
            if len(robot_cmd_velocities) > 0:
                cmd_speed = [np.linalg.norm(v[:2]) for v in robot_cmd_velocities]
                ax2.plot(times, cmd_speed, 'g--', linewidth=2, label='Cmd |v|')
            ax2.plot(times[:len(obj_velocities)], obj_velocities, 'r-', linewidth=1.5, alpha=0.7, label='Object |v|')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Speed (m/s)')
            ax2.set_title('Linear Speed (Actual vs Cmd)')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Object Velocity Components (top-middle-right)
            ax3 = fig.add_subplot(2, 4, 3)
            if len(obj_velocities_xy) > 0:
                obj_vx = [v[0] for v in obj_velocities_xy]
                obj_vy = [v[1] for v in obj_velocities_xy]
                obj_times = times[:len(obj_vx)]
                ax3.plot(obj_times, obj_vx, 'r-', linewidth=2, label='vx')
                ax3.plot(obj_times, obj_vy, 'g-', linewidth=2, label='vy')
                ax3.plot(obj_times[:len(obj_angular_velocities)], obj_angular_velocities, 
                        'b-', linewidth=2, label='ω (rad/s)')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Velocity')
            ax3.set_title('Object Velocity (vx, vy, ω)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Contact force (top-right)
            ax4 = fig.add_subplot(2, 4, 4)
            contact_fill = [1.0 if c else 0.0 for c in robot_contacts]
            max_force = max(robot_forces) if robot_forces else 1.0
            ax4.fill_between(times, 0, [max(max_force, 1) * f for f in contact_fill], 
                           alpha=0.2, color='green', label='In Contact')
            ax4.plot(times, robot_forces, 'r-', linewidth=2, label='Force')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Force (N)')
            ax4.set_title('Contact Force')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Heading and Pushing Direction (bottom-left)
            ax5 = fig.add_subplot(2, 4, 5)
            robot_heading_deg = [np.degrees(h) for h in robot_headings]
            pushing_angle_deg = [np.degrees(a) for a in robot_pushing_angles]
            ax5.plot(times, robot_heading_deg, 'b-', linewidth=2, label='Robot Heading')
            ax5.plot(times, pushing_angle_deg, 'g--', linewidth=2, label='Push Direction')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Angle (degrees)')
            ax5.set_title('Heading vs Push Direction')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. Velocity Command Components (bottom-middle-left)
            ax6 = fig.add_subplot(2, 4, 6)
            if len(robot_cmd_velocities) > 0:
                cmd_vx = [v[0] for v in robot_cmd_velocities]
                cmd_vy = [v[1] for v in robot_cmd_velocities]
                cmd_omega = [v[2] for v in robot_cmd_velocities]
                ax6.plot(times, cmd_vx, 'r-', linewidth=2, label='cmd vx')
                ax6.plot(times, cmd_vy, 'g-', linewidth=2, label='cmd vy')
                ax6.plot(times, cmd_omega, 'b-', linewidth=2, label='cmd ω')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Velocity')
            ax6.set_title('Velocity Command (vx, vy, ω)')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            # 7. Object Orientation (bottom-middle-right)
            ax7 = fig.add_subplot(2, 4, 7)
            obj_orient_deg = [np.degrees(o) for o in obj_orientations]
            ax7.plot(times[:len(obj_orient_deg)], obj_orient_deg, 'r-', linewidth=2)
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Angle (degrees)')
            ax7.set_title('Object Orientation')
            ax7.grid(True, alpha=0.3)
            
            # 8. Metrics summary (bottom-right)
            ax8 = fig.add_subplot(2, 4, 8)
            ax8.axis('off')
            
            cmd_v = robot_state.cmd_velocity
            metrics_text = f"""
EVALUATION METRICS

Simulation Time: {self.simulation_time:.2f} s
Contact Maintained: {self.contact_maintained_time:.2f} s
Contact Losses: {self.contact_loss_count}

Robot ({robot_name}):
  Position: ({robot_state.position[0]:.3f}, {robot_state.position[1]:.3f})
  Heading: {np.degrees(robot_state.heading):.1f}°
  Push Dir: {np.degrees(robot_state.pushing_angle):.1f}°
  
Velocity Command:
  vx: {cmd_v[0]:.3f}  vy: {cmd_v[1]:.3f}
  ω: {cmd_v[2]:.3f} rad/s
  |v|: {np.linalg.norm(cmd_v[:2]):.3f} m/s
  
Contact State:
  In Contact: {'YES' if robot_state.in_contact else 'NO'}
  Force: {robot_state.contact_force_magnitude:.2f} N

Object ({obj_name}):
  Position: ({obj_state.position[0]:.3f}, {obj_state.position[1]:.3f})
  Velocity: ({obj_state.velocity[0]:.3f}, {obj_state.velocity[1]:.3f})
  Angular: {obj_state.angular_velocity:.3f} rad/s
            """
            
            ax8.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
                    verticalalignment='center')
            
            fig.tight_layout()
            
            # Convert to base64
            canvas = FigureCanvasAgg(fig)
            buf = io.BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            print(f"[Observer] Error creating plots: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _web_update_loop(self):
        """Thread loop to push updates to web clients."""
        while self.running:
            try:
                time.sleep(1.0 / UPDATE_RATE)
                
                # Generate and emit plot
                img_base64 = self._create_plots()
                if img_base64:
                    self.socketio.emit('update_plot', {'image': img_base64})
                
                # Emit metrics
                metrics = self._get_metrics()
                self.socketio.emit('update_metrics', metrics)
                
            except Exception as e:
                print(f"[Observer] Update error: {e}")
                time.sleep(0.5)
    
    def start(self):
        """Start the web server and update thread."""
        self.running = True
        
        # Start Flask in background thread
        self.web_thread = threading.Thread(
            target=lambda: self.socketio.run(
                self.app, host='0.0.0.0', port=self.port,
                debug=False, use_reloader=False, log_output=False,
                allow_unsafe_werkzeug=True
            )
        )
        self.web_thread.daemon = True
        self.web_thread.start()
        
        # Give server time to start
        time.sleep(1.0)
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._web_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print(f"[Observer] Web server running at http://localhost:{self.port}")
    
    def stop(self):
        """Stop the web server."""
        self.running = False
    
    def _get_dashboard_html(self):
        """Return the HTML for the dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Maintenance Observer</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee; min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 1900px; margin: 0 auto; background: #1a1a2e;
            border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); overflow: hidden;
            border: 1px solid #333;
        }
        .header {
            background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
            color: white; padding: 20px; text-align: center;
        }
        .header h1 { font-size: 1.8em; margin-bottom: 5px; }
        .header p { font-size: 0.95em; opacity: 0.9; }
        .controls-bar {
            padding: 15px 30px; background: #0f3460;
            display: flex; gap: 30px; flex-wrap: wrap; align-items: center;
            border-bottom: 1px solid #333;
        }
        .control-group { display: flex; align-items: center; gap: 10px; }
        .control-group label { font-size: 0.9em; color: #aaa; }
        .control-group select {
            padding: 8px 12px; border: 1px solid #444; border-radius: 5px;
            background: #1a1a2e; color: #fff; font-size: 0.9em; cursor: pointer;
        }
        .control-group select:hover { border-color: #e94560; }
        .metrics-bar {
            padding: 15px 30px; background: #16213e;
            display: flex; gap: 20px; flex-wrap: wrap; justify-content: space-around;
            border-bottom: 1px solid #333;
        }
        .metric { text-align: center; padding: 8px 15px; }
        .metric-value { font-size: 1.6em; font-weight: bold; color: #e94560; }
        .metric-label { font-size: 0.8em; color: #888; margin-top: 3px; }
        .status-indicator {
            display: inline-block; width: 12px; height: 12px;
            border-radius: 50%; margin-left: 8px; animation: pulse 1.5s infinite;
        }
        .status-contact { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
        .status-no-contact { background: #ff4444; box-shadow: 0 0 10px #ff4444; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
        .robot-status-bar {
            padding: 10px 30px; background: #12192c;
            display: flex; gap: 15px; flex-wrap: wrap;
            border-bottom: 1px solid #333;
        }
        .robot-badge {
            padding: 8px 15px; border-radius: 20px; font-size: 0.85em;
            background: #1a1a2e; border: 1px solid #333;
            display: flex; align-items: center; gap: 8px;
        }
        .robot-badge.in-contact { border-color: #00ff88; background: rgba(0,255,136,0.1); }
        .robot-badge .dot {
            width: 8px; height: 8px; border-radius: 50%; background: #ff4444;
        }
        .robot-badge.in-contact .dot { background: #00ff88; }
        .plot-container { padding: 15px; background: #1a1a2e; }
        .plot-container img {
            width: 100%; height: auto; border-radius: 10px;
            border: 1px solid #333;
        }
        .loading { text-align: center; padding: 50px; font-size: 1.2em; color: #666; }
        .connection-status {
            position: fixed; top: 20px; right: 20px; padding: 8px 15px;
            border-radius: 20px; background: #16213e; box-shadow: 0 5px 15px rgba(0,0,0,0.4);
            display: flex; align-items: center; gap: 8px; z-index: 1000;
            border: 1px solid #333; font-size: 0.85em;
        }
        .connection-dot { width: 8px; height: 8px; border-radius: 50%; background: #00ff88; }
        .connection-dot.disconnected { background: #ff4444; }
    </style>
</head>
<body>
    <div class="connection-status">
        <div class="connection-dot" id="connectionDot"></div>
        <span id="connectionText">Connected</span>
    </div>
    <div class="container">
        <div class="header">
            <h1>🤖 Multi-Robot Contact Maintenance Observer</h1>
            <p>PyBullet Simulation - Real-time Monitoring</p>
        </div>
        <div class="controls-bar">
            <div class="control-group">
                <label>Track Robot:</label>
                <select id="robotSelect"></select>
            </div>
            <div class="control-group">
                <label>Track Object:</label>
                <select id="objectSelect"></select>
            </div>
            <div class="control-group">
                <span style="color:#888; font-size:0.85em;" id="entityCount">Robots: 0 | Objects: 0</span>
            </div>
        </div>
        <div class="robot-status-bar" id="robotStatusBar">
            <!-- Robot badges will be inserted here -->
        </div>
        <div class="metrics-bar">
            <div class="metric">
                <div class="metric-value" id="simTime">0.00s</div>
                <div class="metric-label">Sim Time</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="contactTime">0.00s</div>
                <div class="metric-label">Contact Time</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="contactLosses">0</div>
                <div class="metric-label">Losses</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="robotsInContact">0/0</div>
                <div class="metric-label">In Contact</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="contactForce">0.00N</div>
                <div class="metric-label">Selected Force</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="pushDirection">0.0°</div>
                <div class="metric-label">Push Dir</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="contactStatus">
                    <span id="contactStatusText">NO</span>
                    <span class="status-indicator status-no-contact" id="contactIndicator"></span>
                </div>
                <div class="metric-label">Selected Contact</div>
            </div>
        </div>
        <div class="plot-container">
            <div class="loading" id="loadingText">Waiting for simulation data...</div>
            <img id="plotImage" style="display:none;" alt="Real-time plots">
        </div>
    </div>
    <script>
        const socket = io();
        let selectedRobot = null;
        let selectedObject = null;
        let knownRobots = new Set();
        let knownObjects = new Set();
        
        socket.on('connect', () => {
            document.getElementById('connectionDot').classList.remove('disconnected');
            document.getElementById('connectionText').textContent = 'Connected';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('connectionDot').classList.add('disconnected');
            document.getElementById('connectionText').textContent = 'Disconnected';
        });
        
        socket.on('update_plot', (data) => {
            if (data.image) {
                document.getElementById('plotImage').src = 'data:image/png;base64,' + data.image;
                document.getElementById('plotImage').style.display = 'block';
                document.getElementById('loadingText').style.display = 'none';
            }
        });
        
        socket.on('update_metrics', (metrics) => {
            document.getElementById('simTime').textContent = metrics.simulation_time.toFixed(2) + 's';
            document.getElementById('contactTime').textContent = metrics.contact_maintained_time.toFixed(2) + 's';
            document.getElementById('contactLosses').textContent = metrics.contact_loss_count;
            document.getElementById('entityCount').textContent = 
                `Robots: ${metrics.active_robots} | Objects: ${metrics.active_objects}`;
            
            // Update robot/object dropdowns
            updateDropdowns(metrics);
            
            // Update robot status badges
            updateRobotBadges(metrics);
            
            // Count robots in contact
            let contactCount = 0;
            let totalRobots = 0;
            for (let key in metrics) {
                if (key.endsWith('_contact')) {
                    totalRobots++;
                    if (metrics[key]) contactCount++;
                }
            }
            document.getElementById('robotsInContact').textContent = `${contactCount}/${totalRobots}`;
            
            // Get selected robot's data
            if (selectedRobot) {
                const force = metrics[selectedRobot + '_force'] || 0;
                const pushAngle = metrics[selectedRobot + '_push_angle'] || 0;
                const inContact = metrics[selectedRobot + '_contact'] || false;
                
                document.getElementById('contactForce').textContent = force.toFixed(2) + 'N';
                document.getElementById('pushDirection').textContent = pushAngle.toFixed(1) + '°';
                document.getElementById('contactStatusText').textContent = inContact ? 'YES' : 'NO';
                
                const indicator = document.getElementById('contactIndicator');
                if (inContact) {
                    indicator.classList.remove('status-no-contact');
                    indicator.classList.add('status-contact');
                } else {
                    indicator.classList.remove('status-contact');
                    indicator.classList.add('status-no-contact');
                }
            }
        });
        
        function updateDropdowns(metrics) {
            // Extract robot and object names
            let robots = [];
            let objects = [];
            
            for (let key in metrics) {
                if (key.endsWith('_contact')) {
                    robots.push(key.replace('_contact', ''));
                }
            }
            
            if (metrics.selected_object) {
                objects.push(metrics.selected_object);
            }
            
            // Update robot dropdown if new robots found
            const robotSelect = document.getElementById('robotSelect');
            robots.forEach(r => {
                if (!knownRobots.has(r)) {
                    knownRobots.add(r);
                    const opt = document.createElement('option');
                    opt.value = r;
                    opt.textContent = r;
                    robotSelect.appendChild(opt);
                }
            });
            
            // Set default selection
            if (!selectedRobot && robots.length > 0) {
                selectedRobot = robots[0];
                robotSelect.value = selectedRobot;
            }
            
            // Update object dropdown
            const objectSelect = document.getElementById('objectSelect');
            objects.forEach(o => {
                if (!knownObjects.has(o)) {
                    knownObjects.add(o);
                    const opt = document.createElement('option');
                    opt.value = o;
                    opt.textContent = o;
                    objectSelect.appendChild(opt);
                }
            });
            
            if (!selectedObject && objects.length > 0) {
                selectedObject = objects[0];
                objectSelect.value = selectedObject;
            }
        }
        
        function updateRobotBadges(metrics) {
            const bar = document.getElementById('robotStatusBar');
            let robots = [];
            
            for (let key in metrics) {
                if (key.endsWith('_contact')) {
                    const name = key.replace('_contact', '');
                    robots.push({
                        name: name,
                        inContact: metrics[key],
                        force: metrics[name + '_force'] || 0
                    });
                }
            }
            
            // Sort robots by name
            robots.sort((a, b) => a.name.localeCompare(b.name));
            
            // Build badges HTML
            let html = '';
            robots.forEach(r => {
                const contactClass = r.inContact ? 'in-contact' : '';
                html += `<div class="robot-badge ${contactClass}">
                    <span class="dot"></span>
                    <span>${r.name}</span>
                    <span style="color:#888">${r.force.toFixed(1)}N</span>
                </div>`;
            });
            bar.innerHTML = html;
        }
        
        document.getElementById('robotSelect').addEventListener('change', (e) => {
            selectedRobot = e.target.value;
            fetch('/api/set_selection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({robot: selectedRobot})
            });
        });
        
        document.getElementById('objectSelect').addEventListener('change', (e) => {
            selectedObject = e.target.value;
            fetch('/api/set_selection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({object: selectedObject})
            });
        });
    </script>
</body>
</html>
"""

