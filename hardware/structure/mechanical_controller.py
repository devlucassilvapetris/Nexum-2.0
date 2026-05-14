"""
Nexum v2.0 Mechanical Controller
Advanced mechanical structure control and physics simulation
"""

import numpy as np
import math
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..kinematics.motor_controller import MotorController, MotorConfig, MotorType


class MechanicalState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    BALANCING = "balancing"
    FALLING = "falling"
    ERROR = "error"


class PhysicsEngine:
    """Physics simulation for mechanical structure"""
    
    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity
        self.air_resistance = 0.1
        self.friction_coefficient = 0.3
    
    def calculate_forces(self, mass: float, acceleration: np.ndarray, 
                        velocity: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate forces acting on body"""
        # Gravitational force
        gravity_force = np.array([0, -mass * self.gravity, 0])
        
        # Air resistance
        air_force = -self.air_resistance * velocity * np.linalg.norm(velocity)
        
        # Net force
        net_force = mass * acceleration + gravity_force + air_force
        
        # Torque (simplified)
        torque = np.linalg.norm(np.cross([0, 1, 0], net_force))
        
        return net_force, torque
    
    def check_stability(self, center_of_mass: np.ndarray, 
                       support_points: List[np.ndarray]) -> bool:
        """Check if body is stable"""
        # Calculate support polygon
        if len(support_points) < 3:
            return False
        
        # Project center of mass onto support plane
        com_projection = center_of_mass.copy()
        com_projection[1] = 0  # Project to ground plane
        
        # Check if COM projection is within support polygon
        return self._point_in_polygon(com_projection[:2], 
                                    [p[:2] for p in support_points])
    
    def _point_in_polygon(self, point: np.ndarray, polygon: List[np.ndarray]) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside


class BalanceController:
    """Balance and stability control system"""
    
    def __init__(self, physics_engine: PhysicsEngine):
        self.physics = physics_engine
        self.target_com = np.array([0, 0.9, 0])  # Target center of mass height
        self.balance_threshold = 0.05  # meters
        self.max_correction_angle = 15  # degrees
        
    def calculate_balance_correction(self, current_com: np.ndarray, 
                                   current_orientation: np.ndarray) -> np.ndarray:
        """Calculate balance correction forces"""
        # Calculate COM error
        com_error = self.target_com - current_com
        
        # Calculate orientation error
        orientation_error = np.array([0, 0, 0])  # Roll, pitch, yaw
        orientation_error[0] = -current_orientation[0]  # Roll correction
        orientation_error[1] = -current_orientation[1]  # Pitch correction
        
        # Calculate correction forces
        correction_force = np.zeros(3)
        correction_torque = np.zeros(3)
        
        # COM correction
        if np.linalg.norm(com_error) > self.balance_threshold:
            correction_force = com_error * 50.0  # Proportional control
        
        # Orientation correction
        for i in range(2):  # Roll and pitch only
            if abs(orientation_error[i]) > math.radians(5):
                correction_torque[i] = orientation_error[i] * 10.0
        
        return correction_force, correction_torque
    
    def is_balanced(self, com: np.ndarray, orientation: np.ndarray) -> bool:
        """Check if body is balanced"""
        com_stable = np.linalg.norm(com - self.target_com) < self.balance_threshold
        orientation_stable = (abs(orientation[0]) < math.radians(5) and 
                            abs(orientation[1]) < math.radians(5))
        
        return com_stable and orientation_stable


@dataclass
class MechanicalLink:
    """Mechanical link between body segments"""
    id: str
    parent_segment: str
    child_segment: str
    joint_type: str  # "revolute", "prismatic", "fixed"
    joint_axis: np.ndarray
    joint_limits: Tuple[float, float]  # min, max angle or position
    joint_position: float
    joint_velocity: float
    joint_torque: float


@dataclass
class MechanicalSegment:
    """Mechanical body segment"""
    id: str
    mass: float
    center_of_mass: np.ndarray
    inertia_tensor: np.ndarray
    dimensions: Tuple[float, float, float]  # width, height, depth
    position: np.ndarray
    orientation: np.ndarray  # Euler angles
    velocity: np.ndarray
    angular_velocity: np.ndarray
    links: List[MechanicalLink]


class MechanicalController:
    """
    Advanced mechanical structure controller for Nexum v2.0
    Handles physics simulation, balance control, and structural integrity
    """
    
    def __init__(self):
        self.segments: Dict[str, MechanicalSegment] = {}
        self.links: Dict[str, MechanicalLink] = {}
        self.physics_engine = PhysicsEngine()
        self.balance_controller = BalanceController(self.physics_engine)
        
        # Control state
        self.state = MechanicalState.IDLE
        self.running = False
        self.control_thread = None
        self.update_rate = 100  # Hz
        
        # Support points for stability
        self.support_points: List[np.ndarray] = []
        
        # Callbacks
        self.state_callbacks: List[Callable[[MechanicalState], None]] = []
        self.balance_callbacks: List[Callable[[bool], None]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize mechanical controller"""
        try:
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            
            self.logger.info("Mechanical controller initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize mechanical controller: {e}")
            return False
    
    def add_segment(self, segment_id: str, mass: float, dimensions: Tuple[float, float, float],
                   position: np.ndarray = None, orientation: np.ndarray = None) -> bool:
        """Add mechanical segment"""
        try:
            if position is None:
                position = np.array([0.0, 0.0, 0.0])
            
            if orientation is None:
                orientation = np.array([0.0, 0.0, 0.0])
            
            # Calculate inertia tensor (simplified as rectangular box)
            w, h, d = dimensions
            inertia = np.array([
                [mass * (h**2 + d**2) / 12, 0, 0],
                [0, mass * (w**2 + d**2) / 12, 0],
                [0, 0, mass * (w**2 + h**2) / 12]
            ])
            
            # Center of mass at geometric center
            com = np.array([0.0, h/2, 0.0])
            
            segment = MechanicalSegment(
                id=segment_id,
                mass=mass,
                center_of_mass=com,
                inertia_tensor=inertia,
                dimensions=dimensions,
                position=position,
                orientation=orientation,
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                links=[]
            )
            
            self.segments[segment_id] = segment
            
            # Update support points if this is a foot segment
            if "foot" in segment_id.lower() or "leg" in segment_id.lower():
                foot_position = position + np.array([0, 0, -d/2])
                self.support_points.append(foot_position)
            
            self.logger.info(f"Added mechanical segment: {segment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add segment {segment_id}: {e}")
            return False
    
    def add_link(self, link_id: str, parent_segment: str, child_segment: str,
                 joint_type: str, joint_axis: np.ndarray, 
                 joint_limits: Tuple[float, float]) -> bool:
        """Add mechanical link between segments"""
        try:
            if parent_segment not in self.segments or child_segment not in self.segments:
                return False
            
            link = MechanicalLink(
                id=link_id,
                parent_segment=parent_segment,
                child_segment=child_segment,
                joint_type=joint_type,
                joint_axis=joint_axis / np.linalg.norm(joint_axis),
                joint_limits=joint_limits,
                joint_position=0.0,
                joint_velocity=0.0,
                joint_torque=0.0
            )
            
            self.links[link_id] = link
            
            # Add link to segments
            self.segments[parent_segment].links.append(link)
            self.segments[child_segment].links.append(link)
            
            self.logger.info(f"Added mechanical link: {link_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add link {link_id}: {e}")
            return False
    
    def _control_loop(self):
        """Main control loop"""
        last_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # Update physics simulation
                self._update_physics(dt)
                
                # Check balance
                self._check_balance()
                
                # Update structural integrity
                self._check_structural_integrity()
                
                time.sleep(1.0 / self.update_rate)
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                time.sleep(0.01)
    
    def _update_physics(self, dt: float):
        """Update physics simulation"""
        for segment_id, segment in self.segments.items():
            # Calculate forces
            net_force, torque = self.physics_engine.calculate_forces(
                segment.mass, 
                segment.velocity / dt if dt > 0 else np.zeros(3),
                segment.velocity
            )
            
            # Update velocity and position
            acceleration = net_force / segment.mass
            segment.velocity += acceleration * dt
            segment.position += segment.velocity * dt
            
            # Update angular velocity and orientation
            angular_acceleration = np.linalg.inv(segment.inertia_tensor) @ torque
            segment.angular_velocity += angular_acceleration * dt
            segment.orientation += segment.angular_velocity * dt
            
            # Apply joint constraints
            for link in segment.links:
                if link.child_segment == segment_id:
                    # Apply joint limits
                    if link.joint_position < link.joint_limits[0]:
                        link.joint_position = link.joint_limits[0]
                        link.joint_velocity = 0
                    elif link.joint_position > link.joint_limits[1]:
                        link.joint_position = link.joint_limits[1]
                        link.joint_velocity = 0
    
    def _check_balance(self):
        """Check system balance"""
        # Calculate overall center of mass
        total_mass = 0
        weighted_com = np.zeros(3)
        
        for segment in self.segments.values():
            total_mass += segment.mass
            weighted_com += segment.mass * (segment.position + segment.center_of_mass)
        
        overall_com = weighted_com / total_mass if total_mass > 0 else np.zeros(3)
        
        # Check stability
        is_stable = self.physics_engine.check_stability(overall_com, self.support_points)
        
        # Calculate balance correction if needed
        if not is_stable:
            correction_force, correction_torque = self.balance_controller.calculate_balance_correction(
                overall_com, self.segments.get("torso", MechanicalSegment("", 0, np.zeros(3), np.zeros(3), 
                np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), [])).orientation
            )
            
            # Apply correction to base segment
            if "torso" in self.segments:
                torso = self.segments["torso"]
                torso.velocity += correction_force / torso.mass * 0.01
        
        # Trigger callbacks
        for callback in self.balance_callbacks:
            try:
                callback(is_stable)
            except Exception as e:
                self.logger.error(f"Balance callback error: {e}")
        
        # Update state
        if not is_stable:
            self.state = MechanicalState.BALANCING
        else:
            self.state = MechanicalState.IDLE
    
    def _check_structural_integrity(self):
        """Check structural integrity and stress"""
        for link_id, link in self.links.items():
            # Check joint stress
            max_torque = 100.0  # Nm (example value)
            
            if abs(link.joint_torque) > max_torque:
                self.logger.warning(f"Joint {link_id} exceeding torque limit: {link.joint_torque:.2f} Nm")
                self.state = MechanicalState.ERROR
                
                # Trigger state callbacks
                for callback in self.state_callbacks:
                    try:
                        callback(self.state)
                    except Exception as e:
                        self.logger.error(f"State callback error: {e}")
    
    def set_joint_position(self, link_id: str, position: float, velocity: float = 0.0) -> bool:
        """Set joint position and velocity"""
        if link_id not in self.links:
            return False
        
        link = self.links[link_id]
        
        # Check limits
        position = max(link.joint_limits[0], min(link.joint_limits[1], position))
        
        link.joint_position = position
        link.joint_velocity = velocity
        
        return True
    
    def get_joint_position(self, link_id: str) -> Optional[float]:
        """Get joint position"""
        if link_id not in self.links:
            return None
        
        return self.links[link_id].joint_position
    
    def get_segment_position(self, segment_id: str) -> Optional[np.ndarray]:
        """Get segment position"""
        if segment_id not in self.segments:
            return None
        
        return self.segments[segment_id].position.copy()
    
    def get_center_of_mass(self) -> np.ndarray:
        """Get overall center of mass"""
        total_mass = 0
        weighted_com = np.zeros(3)
        
        for segment in self.segments.values():
            total_mass += segment.mass
            weighted_com += segment.mass * (segment.position + segment.center_of_mass)
        
        return weighted_com / total_mass if total_mass > 0 else np.zeros(3)
    
    def is_stable(self) -> bool:
        """Check if system is stable"""
        com = self.get_center_of_mass()
        return self.physics_engine.check_stability(com, self.support_points)
    
    def register_state_callback(self, callback: Callable[[MechanicalState], None]):
        """Register state change callback"""
        self.state_callbacks.append(callback)
    
    def register_balance_callback(self, callback: Callable[[bool], None]):
        """Register balance status callback"""
        self.balance_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            "state": self.state.value,
            "segments": {
                seg_id: {
                    "position": segment.position.tolist(),
                    "orientation": segment.orientation.tolist(),
                    "velocity": segment.velocity.tolist(),
                    "mass": segment.mass
                }
                for seg_id, segment in self.segments.items()
            },
            "links": {
                link_id: {
                    "position": link.joint_position,
                    "velocity": link.joint_velocity,
                    "torque": link.joint_torque,
                    "limits": link.joint_limits
                }
                for link_id, link in self.links.items()
            },
            "center_of_mass": self.get_center_of_mass().tolist(),
            "is_stable": self.is_stable(),
            "support_points": len(self.support_points)
        }
    
    def emergency_stop(self):
        """Emergency stop all motion"""
        for segment in self.segments.values():
            segment.velocity = np.zeros(3)
            segment.angular_velocity = np.zeros(3)
        
        for link in self.links.values():
            link.joint_velocity = 0.0
            link.joint_torque = 0.0
        
        self.state = MechanicalState.IDLE
        self.logger.warning("Emergency stop activated")
    
    def shutdown(self):
        """Shutdown mechanical controller"""
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2)
        
        self.emergency_stop()
        
        self.logger.info("Mechanical controller shutdown complete")


def main():
    """Example usage of mechanical controller"""
    controller = MechanicalController()
    
    if controller.initialize():
        # Add segments (simplified humanoid)
        controller.add_segment("torso", 30.0, (0.4, 0.6, 0.2), np.array([0, 0.9, 0]))
        controller.add_segment("head", 5.0, (0.2, 0.25, 0.2), np.array([0, 1.5, 0]))
        controller.add_segment("left_foot", 2.0, (0.15, 0.05, 0.3), np.array([-0.1, 0, 0]))
        controller.add_segment("right_foot", 2.0, (0.15, 0.05, 0.3), np.array([0.1, 0, 0]))
        
        # Add links
        controller.add_link("neck", "torso", "head", "revolute", np.array([0, 0, 1]), (-45, 45))
        controller.add_link("left_ankle", "left_foot", "torso", "revolute", np.array([1, 0, 0]), (-30, 30))
        controller.add_link("right_ankle", "right_foot", "torso", "revolute", np.array([1, 0, 0]), (-30, 30))
        
        # Simulate for a few seconds
        time.sleep(2)
        
        # Get status
        status = controller.get_system_status()
        print("=== Mechanical System Status ===")
        print(f"State: {status['state']}")
        print(f"Stable: {status['is_stable']}")
        print(f"Center of Mass: {status['center_of_mass']}")
        
        controller.shutdown()
    else:
        print("Failed to initialize mechanical controller")


if __name__ == "__main__":
    main()
