"""
Nexum v2.0 Motor Controller
Handles motor control for kinematics and movement
"""

import math
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np


class MotorState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    HOLDING = "holding"
    ERROR = "error"


class MotorType(Enum):
    STEPPER = "stepper"
    SERVO = "servo"
    DC = "dc"
    BRUSHLESS = "brushless"


@dataclass
class MotorConfig:
    id: str
    name: str
    type: MotorType
    max_speed: float  # rad/s
    max_acceleration: float  # rad/s²
    max_torque: float  # N⋅m
    gear_ratio: float
    encoder_resolution: int
    min_position: float  # rad
    max_position: float  # rad
    home_position: float  # rad


@dataclass
class MotorStatus:
    position: float  # rad
    velocity: float  # rad/s
    acceleration: float  # rad/s²
    torque: float  # N⋅m
    temperature: float  # °C
    state: MotorState
    error_code: Optional[int]


class TrajectoryPlanner:
    """Plans smooth trajectories for motor movements"""
    
    def __init__(self, max_velocity: float, max_acceleration: float):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
    
    def plan_trajectory(self, start_pos: float, end_pos: float, duration: float) -> List[Tuple[float, float, float]]:
        """
        Plan a trajectory from start to end position
        Returns list of (position, velocity, acceleration) tuples
        """
        distance = end_pos - start_pos
        
        # Simple trapezoidal velocity profile
        if abs(distance) < 0.001:
            return [(start_pos, 0, 0)]
        
        # Calculate acceleration and deceleration phases
        accel_time = self.max_velocity / self.max_acceleration
        accel_distance = 0.5 * self.max_acceleration * accel_time ** 2
        
        if 2 * accel_distance > abs(distance):
            # Triangular profile (no constant velocity phase)
            peak_velocity = math.sqrt(abs(distance) * self.max_acceleration)
            accel_time = peak_velocity / self.max_acceleration
            total_time = 2 * accel_time
        else:
            # Trapezoidal profile
            peak_velocity = self.max_velocity
            total_time = 2 * accel_time + (abs(distance) - 2 * accel_distance) / peak_velocity
        
        # Generate trajectory points
        trajectory = []
        num_points = int(total_time * 100)  # 100 Hz update rate
        dt = total_time / num_points
        
        for i in range(num_points + 1):
            t = i * dt
            
            if t <= accel_time:
                # Acceleration phase
                velocity = (peak_velocity / accel_time) * t
                position = start_pos + 0.5 * (peak_velocity / accel_time) * t ** 2
                acceleration = peak_velocity / accel_time
            elif t <= total_time - accel_time:
                # Constant velocity phase
                velocity = peak_velocity
                position = start_pos + accel_distance + peak_velocity * (t - accel_time)
                acceleration = 0
            else:
                # Deceleration phase
                t_decel = t - (total_time - accel_time)
                velocity = peak_velocity - (peak_velocity / accel_time) * t_decel
                position = end_pos - 0.5 * (peak_velocity / accel_time) * (accel_time - t_decel) ** 2
                acceleration = -peak_velocity / accel_time
            
            # Adjust direction based on movement direction
            if distance < 0:
                velocity = -velocity
                position = start_pos - (position - start_pos)
                acceleration = -acceleration
            
            trajectory.append((position, velocity, acceleration))
        
        return trajectory


class PIDController:
    """PID controller for motor position/velocity control"""
    
    def __init__(self, kp: float, ki: float, kd: float, output_min: float, output_max: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        
        self.integral = 0
        self.previous_error = 0
        self.last_time = time.time()
    
    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Calculate PID output"""
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        self.integral = max(min(self.integral, self.output_max / self.ki), self.output_min / self.ki)
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        output = max(min(output, self.output_max), self.output_min)
        
        self.previous_error = error
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0
        self.previous_error = 0


class Motor:
    """Individual motor instance"""
    
    def __init__(self, config: MotorConfig):
        self.config = config
        self.status = MotorStatus(
            position=0,
            velocity=0,
            acceleration=0,
            torque=0,
            temperature=0,
            state=MotorState.IDLE,
            error_code=None
        )
        
        self.trajectory_planner = TrajectoryPlanner(
            config.max_speed,
            config.max_acceleration
        )
        
        # PID controllers for position and velocity
        self.position_pid = PIDController(
            kp=10.0, ki=0.1, kd=1.0,
            output_min=-config.max_torque,
            output_max=config.max_torque
        )
        
        self.velocity_pid = PIDController(
            kp=5.0, ki=0.05, kd=0.5,
            output_min=-config.max_torque,
            output_max=config.max_torque
        )
        
        self.current_trajectory = []
        self.trajectory_index = 0
        self.target_position = config.home_position
        self.target_velocity = 0
    
    def move_to(self, position: float, duration: float = None) -> bool:
        """Move motor to specified position"""
        # Clamp position to limits
        position = max(min(position, self.config.max_position), self.config.min_position)
        
        if duration is None:
            # Calculate minimum duration based on constraints
            distance = abs(position - self.status.position)
            if distance < 0.001:
                return True
            
            # Use trapezoidal profile calculation
            accel_time = self.config.max_speed / self.config.max_acceleration
            accel_distance = 0.5 * self.config.max_acceleration * accel_time ** 2
            
            if 2 * accel_distance > distance:
                peak_velocity = math.sqrt(distance * self.config.max_acceleration)
                accel_time = peak_velocity / self.config.max_acceleration
                duration = 2 * accel_time
            else:
                duration = 2 * accel_time + (distance - 2 * accel_distance) / self.config.max_speed
        
        # Plan trajectory
        self.current_trajectory = self.trajectory_planner.plan_trajectory(
            self.status.position, position, duration
        )
        self.trajectory_index = 0
        self.target_position = position
        
        if self.current_trajectory:
            self.status.state = MotorState.MOVING
            return True
        
        return False
    
    def move_with_velocity(self, velocity: float) -> bool:
        """Move with specified velocity"""
        # Clamp velocity
        velocity = max(min(velocity, self.config.max_speed), -self.config.max_speed)
        
        self.target_velocity = velocity
        self.current_trajectory = []
        self.status.state = MotorState.MOVING
        return True
    
    def update(self, dt: float) -> bool:
        """Update motor state and control"""
        if self.status.state == MotorState.ERROR:
            return False
        
        # Follow trajectory if available
        if self.current_trajectory and self.trajectory_index < len(self.current_trajectory):
            target_pos, target_vel, target_acc = self.current_trajectory[self.trajectory_index]
            
            # Update position PID
            torque_command = self.position_pid.update(target_pos, self.status.position, dt)
            
            # Update motor status
            self.status.position += self.status.velocity * dt
            self.status.velocity += (torque_command / self.config.gear_ratio) * dt
            self.status.acceleration = target_acc
            self.status.torque = torque_command
            
            self.trajectory_index += 1
            
            # Check if trajectory complete
            if self.trajectory_index >= len(self.current_trajectory):
                self.current_trajectory = []
                self.status.state = MotorState.HOLDING
                self.status.velocity = 0
                self.status.acceleration = 0
        
        elif self.status.state == MotorState.MOVING and self.target_velocity != 0:
            # Velocity control mode
            torque_command = self.velocity_pid.update(self.target_velocity, self.status.velocity, dt)
            
            self.status.position += self.status.velocity * dt
            self.status.velocity += (torque_command / self.config.gear_ratio) * dt
            self.status.torque = torque_command
        
        elif self.status.state == MotorState.HOLDING:
            # Position holding mode
            torque_command = self.position_pid.update(self.target_position, self.status.position, dt)
            self.status.torque = torque_command
        
        # Simulate temperature (simple model)
        if abs(self.status.torque) > 0.1 * self.config.max_torque:
            self.status.temperature += 0.1 * dt
        else:
            self.status.temperature = max(0, self.status.temperature - 0.05 * dt)
        
        # Check for overheating
        if self.status.temperature > 80:
            self.status.state = MotorState.ERROR
            self.status.error_code = 1  # Overheating error
            return False
        
        return True
    
    def stop(self):
        """Emergency stop"""
        self.current_trajectory = []
        self.target_velocity = 0
        self.status.state = MotorState.IDLE
        self.status.velocity = 0
        self.status.acceleration = 0
        self.position_pid.reset()
        self.velocity_pid.reset()
    
    def home(self) -> bool:
        """Home the motor"""
        return self.move_to(self.config.home_position, duration=2.0)
    
    def get_status(self) -> MotorStatus:
        """Get current motor status"""
        return self.status


class MotorController:
    """
    Central motor controller for Nexum v2.0
    Manages multiple motors and coordinated movements
    """
    
    def __init__(self):
        self.motors: Dict[str, Motor] = {}
        self.running = False
        self.control_thread = None
        self.update_rate = 100  # Hz
        self.logger = logging.getLogger(__name__)
    
    def add_motor(self, config: MotorConfig) -> bool:
        """Add a motor to the controller"""
        try:
            motor = Motor(config)
            self.motors[config.id] = motor
            self.logger.info(f"Added motor: {config.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add motor {config.id}: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize motor controller and start control loop"""
        try:
            if not self.motors:
                self.logger.warning("No motors configured")
                return False
            
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            
            # Home all motors
            self.home_all()
            
            self.logger.info("Motor controller initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize motor controller: {e}")
            return False
    
    def _control_loop(self):
        """Main control loop"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update all motors
            for motor_id, motor in self.motors.items():
                try:
                    motor.update(dt)
                except Exception as e:
                    self.logger.error(f"Error updating motor {motor_id}: {e}")
                    motor.status.state = MotorState.ERROR
            
            # Sleep to maintain update rate
            time.sleep(1.0 / self.update_rate)
    
    def move_motor(self, motor_id: str, position: float, duration: float = None) -> bool:
        """Move a specific motor"""
        if motor_id not in self.motors:
            return False
        
        return self.motors[motor_id].move_to(position, duration)
    
    def move_motors(self, movements: Dict[str, Tuple[float, float]]) -> bool:
        """Move multiple motors simultaneously"""
        success = True
        for motor_id, (position, duration) in movements.items():
            if not self.move_motor(motor_id, position, duration):
                success = False
        
        return success
    
    def set_motor_velocity(self, motor_id: str, velocity: float) -> bool:
        """Set motor velocity"""
        if motor_id not in self.motors:
            return False
        
        return self.motors[motor_id].move_with_velocity(velocity)
    
    def stop_motor(self, motor_id: str) -> bool:
        """Stop a specific motor"""
        if motor_id not in self.motors:
            return False
        
        self.motors[motor_id].stop()
        return True
    
    def stop_all(self):
        """Emergency stop all motors"""
        for motor in self.motors.values():
            motor.stop()
    
    def home_motor(self, motor_id: str) -> bool:
        """Home a specific motor"""
        if motor_id not in self.motors:
            return False
        
        return self.motors[motor_id].home()
    
    def home_all(self) -> bool:
        """Home all motors"""
        success = True
        for motor_id in self.motors:
            if not self.home_motor(motor_id):
                success = False
        
        return success
    
    def get_motor_status(self, motor_id: str) -> Optional[MotorStatus]:
        """Get status of a specific motor"""
        if motor_id not in self.motors:
            return None
        
        return self.motors[motor_id].get_status()
    
    def get_all_status(self) -> Dict[str, MotorStatus]:
        """Get status of all motors"""
        return {motor_id: motor.get_status() for motor_id, motor in self.motors.items()}
    
    def is_movement_complete(self, motor_id: str = None) -> bool:
        """Check if movement is complete"""
        if motor_id:
            if motor_id not in self.motors:
                return True
            motor = self.motors[motor_id]
            return motor.status.state in [MotorState.IDLE, MotorState.HOLDING]
        else:
            return all(
                motor.status.state in [MotorState.IDLE, MotorState.HOLDING]
                for motor in self.motors.values()
            )
    
    def wait_for_movement(self, motor_id: str = None, timeout: float = 30.0) -> bool:
        """Wait for movement to complete"""
        start_time = time.time()
        
        while not self.is_movement_complete(motor_id):
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.1)
        
        return True
    
    def shutdown(self):
        """Shutdown motor controller"""
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2)
        
        # Stop all motors
        self.stop_all()
        
        self.logger.info("Motor controller shutdown complete")
