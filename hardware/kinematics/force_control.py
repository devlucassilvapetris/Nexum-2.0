"""
Nexum v2.0 Force Control and Sensor Feedback
Advanced force control and sensory feedback integration
"""

import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import deque


class SensorType(Enum):
    FORCE_TORQUE = "force_torque"
    TACTILE = "tactile"
    PROXIMITY = "proximity"
    CURRENT = "current"
    POSITION = "position"
    TEMPERATURE = "temperature"


@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    value: float
    unit: str
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class ForceControlConfig:
    max_force: float  # Newtons
    max_torque: float  # N⋅m
    force_gain: float = 1.0
    torque_gain: float = 1.0
    damping_ratio: float = 0.7
    stiffness: float = 1000.0  # N/m
    compliance: float = 0.001  # m/N


@dataclass
class ImpedanceControlParams:
    mass: float = 1.0  # kg
    damping: float = 10.0  # N⋅s/m
    stiffness: float = 100.0  # N/m


class ForceSensor:
    """Force/torque sensor interface"""
    
    def __init__(self, sensor_id: str, max_force: float = 100.0, max_torque: float = 10.0):
        self.sensor_id = sensor_id
        self.max_force = max_force
        self.max_torque = max_torque
        self.logger = logging.getLogger(__name__)
        
        # Sensor state
        self.current_force = np.zeros(3)  # Fx, Fy, Fz
        self.current_torque = np.zeros(3)  # Tx, Ty, Tz
        self.last_reading_time = 0
        
        # Calibration
        self.force_offset = np.zeros(3)
        self.torque_offset = np.zeros(3)
        self.calibrated = False
    
    def calibrate(self, samples: int = 100) -> bool:
        """Calibrate sensor by averaging zero-force readings"""
        self.logger.info(f"Calibrating force sensor {self.sensor_id}")
        
        force_readings = []
        torque_readings = []
        
        for _ in range(samples):
            # Simulated reading (in real implementation, read from hardware)
            force_noise = np.random.normal(0, 0.1, 3)
            torque_noise = np.random.normal(0, 0.01, 3)
            
            force_readings.append(force_noise)
            torque_readings.append(torque_noise)
            time.sleep(0.01)
        
        self.force_offset = np.mean(force_readings, axis=0)
        self.torque_offset = np.mean(torque_readings, axis=0)
        self.calibrated = True
        
        self.logger.info(f"Sensor {self.sensor_id} calibrated")
        return True
    
    def read(self) -> SensorReading:
        """Read current force/torque values"""
        # Simulated reading (in real implementation, read from hardware)
        force_noise = np.random.normal(0, 0.1, 3)
        torque_noise = np.random.normal(0, 0.01, 3)
        
        raw_force = force_noise
        raw_torque = torque_noise
        
        # Apply calibration offset
        if self.calibrated:
            self.current_force = raw_force - self.force_offset
            self.current_torque = raw_torque - self.torque_offset
        else:
            self.current_force = raw_force
            self.current_torque = raw_torque
        
        self.last_reading_time = time.time()
        
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.FORCE_TORQUE,
            timestamp=self.last_reading_time,
            value=np.linalg.norm(self.current_force),
            unit="N",
            raw_data={
                "force": self.current_force.tolist(),
                "torque": self.current_torque.tolist()
            }
        )
    
    def get_force_vector(self) -> np.ndarray:
        """Get force vector (Fx, Fy, Fz)"""
        return self.current_force.copy()
    
    def get_torque_vector(self) -> np.ndarray:
        """Get torque vector (Tx, Ty, Tz)"""
        return self.current_torque.copy()
    
    def get_force_magnitude(self) -> float:
        """Get magnitude of force vector"""
        return np.linalg.norm(self.current_force)
    
    def get_torque_magnitude(self) -> float:
        """Get magnitude of torque vector"""
        return np.linalg.norm(self.current_torque)


class TactileSensor:
    """Tactile sensor array for surface contact detection"""
    
    def __init__(self, sensor_id: str, num_elements: int = 16):
        self.sensor_id = sensor_id
        self.num_elements = num_elements
        self.logger = logging.getLogger(__name__)
        
        # Sensor state
        self.pressure_values = np.zeros(num_elements)
        self.contact_detected = False
        self.contact_center = np.zeros(2)
        self.total_force = 0.0
    
    def read(self) -> SensorReading:
        """Read tactile sensor array"""
        # Simulated reading (in real implementation, read from hardware)
        noise = np.random.normal(0, 0.01, self.num_elements)
        self.pressure_values = np.maximum(noise, 0)
        
        # Detect contact
        threshold = 0.1
        self.contact_detected = np.any(self.pressure_values > threshold)
        
        if self.contact_detected:
            # Compute contact center
            active_elements = self.pressure_values > threshold
            if np.any(active_elements):
                indices = np.where(active_elements)[0]
                self.contact_center = np.array([
                    np.mean(indices % 4),
                    np.mean(indices // 4)
                ])
                self.total_force = np.sum(self.pressure_values)
        else:
            self.contact_center = np.zeros(2)
            self.total_force = 0.0
        
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.TACTILE,
            timestamp=time.time(),
            value=self.total_force,
            unit="N",
            raw_data={
                "pressure_values": self.pressure_values.tolist(),
                "contact_detected": self.contact_detected,
                "contact_center": self.contact_center.tolist()
            }
        )
    
    def get_contact_detected(self) -> bool:
        """Check if contact is detected"""
        return self.contact_detected
    
    def get_contact_center(self) -> np.ndarray:
        """Get center of contact"""
        return self.contact_center.copy()
    
    def get_pressure_map(self) -> np.ndarray:
        """Get pressure values as 2D map"""
        return self.pressure_values.reshape(4, 4)


class ImpedanceController:
    """Impedance control for compliant interaction"""
    
    def __init__(self, params: ImpedanceControlParams):
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # State
        self.position_error = np.zeros(3)
        self.velocity_error = np.zeros(3)
        self.force_error = np.zeros(3)
        
        # Desired trajectory
        self.desired_position = np.zeros(3)
        self.desired_velocity = np.zeros(3)
        self.desired_force = np.zeros(3)
        
        # Current state
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_force = np.zeros(3)
    
    def set_desired_trajectory(self, position: np.ndarray, velocity: np.ndarray = None):
        """Set desired position and velocity"""
        self.desired_position = np.array(position)
        if velocity is not None:
            self.desired_velocity = np.array(velocity)
        else:
            self.desired_velocity = np.zeros(3)
    
    def set_desired_force(self, force: np.ndarray):
        """Set desired force"""
        self.desired_force = np.array(force)
    
    def update(self, 
              current_position: np.ndarray,
              current_velocity: np.ndarray,
              current_force: np.ndarray,
              dt: float) -> np.ndarray:
        """
        Update impedance controller and compute force command
        Returns: force command to apply
        """
        self.current_position = np.array(current_position)
        self.current_velocity = np.array(current_velocity)
        self.current_force = np.array(current_force)
        
        # Compute errors
        self.position_error = self.desired_position - self.current_position
        self.velocity_error = self.desired_velocity - self.current_velocity
        self.force_error = self.desired_force - self.current_force
        
        # Impedance control law: F = M*a + D*v + K*x
        # Rearranged for force command: F_cmd = K*(x_d - x) + D*(v_d - v) + F_d
        stiffness_force = self.params.stiffness * self.position_error
        damping_force = self.params.damping * self.velocity_error
        desired_force_term = self.desired_force
        
        # Total force command
        force_command = stiffness_force + damping_force + desired_force_term
        
        return force_command
    
    def get_errors(self) -> Dict[str, np.ndarray]:
        """Get current error values"""
        return {
            "position_error": self.position_error.copy(),
            "velocity_error": self.velocity_error.copy(),
            "force_error": self.force_error.copy()
        }


class AdmittanceController:
    """Admittance control for force-controlled motion"""
    
    def __init__(self, params: ImpedanceControlParams):
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # State
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.measured_force = np.zeros(3)
        
        # Integration state
        self.integrated_velocity = np.zeros(3)
    
    def update(self, 
              measured_force: np.ndarray,
              desired_position: np.ndarray,
              dt: float) -> np.ndarray:
        """
        Update admittance controller and compute position correction
        Returns: position correction to apply
        """
        self.measured_force = np.array(measured_force)
        
        # Admittance control: a = (F_ext - K*(x - x_d) - D*v) / M
        # Simplified: position correction based on force
        force_error = self.measured_force
        
        # Compute acceleration from force
        acceleration = force_error / self.params.mass
        
        # Integrate to get velocity
        self.integrated_velocity += acceleration * dt
        
        # Apply damping
        self.integrated_velocity *= (1.0 - self.params.damping * dt)
        
        # Integrate to get position correction
        position_correction = self.integrated_velocity * dt
        
        # Update current position
        self.current_position = np.array(desired_position) + position_correction
        
        return position_correction
    
    def get_current_position(self) -> np.ndarray:
        """Get current position after admittance correction"""
        return self.current_position.copy()


class ForceController:
    """
    Main force controller integrating sensors and control strategies
    """
    
    def __init__(self, config: ForceControlConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sensors
        self.force_sensors: Dict[str, ForceSensor] = {}
        self.tactile_sensors: Dict[str, TactileSensor] = {}
        
        # Controllers
        self.impedance_controller: Optional[ImpedanceController] = None
        self.admittance_controller: Optional[AdmittanceController] = None
        
        # Control mode
        self.control_mode = "position"  # position, force, impedance, admittance
        
        # Monitoring
        self.force_history = deque(maxlen=100)
        self.contact_history = deque(maxlen=50)
        self.running = False
        self.monitor_thread = None
    
    def add_force_sensor(self, sensor_id: str, max_force: float = 100.0, max_torque: float = 10.0):
        """Add force/torque sensor"""
        sensor = ForceSensor(sensor_id, max_force, max_torque)
        self.force_sensors[sensor_id] = sensor
        self.logger.info(f"Added force sensor: {sensor_id}")
    
    def add_tactile_sensor(self, sensor_id: str, num_elements: int = 16):
        """Add tactile sensor"""
        sensor = TactileSensor(sensor_id, num_elements)
        self.tactile_sensors[sensor_id] = sensor
        self.logger.info(f"Added tactile sensor: {sensor_id}")
    
    def set_impedance_control(self, params: ImpedanceControlParams):
        """Enable impedance control with specified parameters"""
        self.impedance_controller = ImpedanceController(params)
        self.control_mode = "impedance"
        self.logger.info("Impedance control enabled")
    
    def set_admittance_control(self, params: ImpedanceControlParams):
        """Enable admittance control with specified parameters"""
        self.admittance_controller = AdmittanceController(params)
        self.control_mode = "admittance"
        self.logger.info("Admittance control enabled")
    
    def set_position_control(self):
        """Set position control mode"""
        self.control_mode = "position"
        self.logger.info("Position control mode")
    
    def set_force_control(self):
        """Set force control mode"""
        self.control_mode = "force"
        self.logger.info("Force control mode")
    
    def calibrate_sensors(self) -> bool:
        """Calibrate all force sensors"""
        success = True
        for sensor_id, sensor in self.force_sensors.items():
            if not sensor.calibrate():
                self.logger.error(f"Failed to calibrate sensor {sensor_id}")
                success = False
        
        return success
    
    def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read all sensors"""
        readings = {}
        
        for sensor_id, sensor in self.force_sensors.items():
            readings[sensor_id] = sensor.read()
        
        for sensor_id, sensor in self.tactile_sensors.items():
            readings[sensor_id] = sensor.read()
        
        return readings
    
    def get_net_force(self) -> np.ndarray:
        """Get net force from all force sensors"""
        net_force = np.zeros(3)
        
        for sensor in self.force_sensors.values():
            net_force += sensor.get_force_vector()
        
        return net_force
    
    def get_net_torque(self) -> np.ndarray:
        """Get net torque from all force sensors"""
        net_torque = np.zeros(3)
        
        for sensor in self.force_sensors.values():
            net_torque += sensor.get_torque_vector()
        
        return net_torque
    
    def check_contact(self) -> bool:
        """Check if any tactile sensor detects contact"""
        for sensor in self.tactile_sensors.values():
            if sensor.get_contact_detected():
                return True
        return False
    
    def compute_force_command(self,
                             desired_position: np.ndarray,
                             desired_velocity: np.ndarray = None,
                             desired_force: np.ndarray = None,
                             current_position: np.ndarray = None,
                             current_velocity: np.ndarray = None) -> np.ndarray:
        """
        Compute force command based on control mode
        """
        if self.control_mode == "impedance" and self.impedance_controller:
            if current_position is None or current_velocity is None:
                return np.zeros(3)
            
            current_force = self.get_net_force()
            force_command = self.impedance_controller.update(
                current_position,
                current_velocity,
                current_force,
                dt=0.01
            )
            
            return force_command
        
        elif self.control_mode == "admittance" and self.admittance_controller:
            if desired_position is None:
                return np.zeros(3)
            
            measured_force = self.get_net_force()
            position_correction = self.admittance_controller.update(
                measured_force,
                desired_position,
                dt=0.01
            )
            
            # Convert position correction to equivalent force
            force_command = position_correction * self.config.stiffness
            return force_command
        
        elif self.control_mode == "force" and desired_force is not None:
            # Direct force control
            force_command = np.array(desired_force)
            
            # Apply force limits
            force_magnitude = np.linalg.norm(force_command)
            if force_magnitude > self.config.max_force:
                force_command = force_command / force_magnitude * self.config.max_force
            
            return force_command
        
        else:
            # Position control - no force command
            return np.zeros(3)
    
    def start_monitoring(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Start background sensor monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(callback,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Force monitoring started")
    
    def stop_monitoring(self):
        """Stop sensor monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("Force monitoring stopped")
    
    def _monitor_loop(self, callback: Optional[Callable[[Dict[str, Any]], None]]):
        """Background monitoring loop"""
        while self.running:
            try:
                # Read all sensors
                readings = self.read_all_sensors()
                
                # Update history
                net_force = self.get_net_force()
                self.force_history.append(np.linalg.norm(net_force))
                self.contact_history.append(self.check_contact())
                
                # Check force limits
                force_magnitude = np.linalg.norm(net_force)
                if force_magnitude > self.config.max_force:
                    self.logger.warning(f"Force limit exceeded: {force_magnitude:.2f} N")
                
                # Callback if provided
                if callback:
                    callback({
                        "readings": readings,
                        "net_force": net_force.tolist(),
                        "net_torque": self.get_net_torque().tolist(),
                        "contact_detected": self.check_contact(),
                        "force_magnitude": force_magnitude
                    })
                
                time.sleep(0.01)  # 100 Hz
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(0.1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get force control statistics"""
        return {
            "control_mode": self.control_mode,
            "net_force_magnitude": np.linalg.norm(self.get_net_force()),
            "net_torque_magnitude": np.linalg.norm(self.get_net_torque()),
            "contact_detected": self.check_contact(),
            "force_history_length": len(self.force_history),
            "average_force": np.mean(self.force_history) if self.force_history else 0.0,
            "max_force": np.max(self.force_history) if self.force_history else 0.0,
            "contact_ratio": sum(self.contact_history) / len(self.contact_history) if self.contact_history else 0.0
        }
    
    def emergency_stop(self):
        """Emergency stop - zero all forces"""
        self.logger.warning("Emergency stop triggered")
        
        # Reset controllers
        if self.impedance_controller:
            self.impedance_controller.desired_force = np.zeros(3)
        
        # In real implementation, send stop command to hardware
