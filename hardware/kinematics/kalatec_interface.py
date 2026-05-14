"""
Nexum v2.0 Kalatec Interface
Specific interface for Kalatec motors and controllers
"""

import serial
import time
import struct
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

from .motor_controller import MotorConfig, MotorType, MotorState


class KalatecCommand(Enum):
    GET_POSITION = 0x01
    SET_POSITION = 0x02
    GET_VELOCITY = 0x03
    SET_VELOCITY = 0x04
    GET_TORQUE = 0x05
    SET_TORQUE = 0x06
    GET_STATUS = 0x07
    SET_PID = 0x08
    HOME = 0x09
    STOP = 0x0A
    RESET = 0x0B


@dataclass
class KalatecStatus:
    position: float
    velocity: float
    torque: float
    temperature: float
    voltage: float
    current: float
    error_flags: int
    mode: int


class KalatecProtocol:
    """Kalatec communication protocol"""
    
    def __init__(self, serial_port: str, baudrate: int = 115200):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
        # Protocol constants
        self.HEADER = 0xAA
        self.FOOTER = 0x55
        self.TIMEOUT = 1.0
    
    def connect(self) -> bool:
        """Connect to Kalatec controller"""
        try:
            self.connection = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=self.TIMEOUT,
                write_timeout=self.TIMEOUT
            )
            
            # Wait for connection to stabilize
            time.sleep(0.1)
            
            # Test connection
            if self._test_connection():
                self.logger.info(f"Connected to Kalatec on {self.serial_port}")
                return True
            else:
                self.connection.close()
                self.connection = None
                return False
                
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to Kalatec: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Kalatec controller"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def _test_connection(self) -> bool:
        """Test connection to Kalatec controller"""
        try:
            status = self.get_status()
            return status is not None
        except:
            return False
    
    def _send_command(self, command: KalatecCommand, data: bytes = b'') -> Optional[bytes]:
        """Send command to Kalatec controller"""
        if not self.connection:
            return None
        
        try:
            # Build packet
            packet = bytearray()
            packet.append(self.HEADER)
            packet.append(command.value)
            packet.append(len(data))
            packet.extend(data)
            
            # Calculate checksum
            checksum = sum(packet[1:]) & 0xFF
            packet.append(checksum)
            packet.append(self.FOOTER)
            
            # Send packet
            self.connection.write(packet)
            self.connection.flush()
            
            # Read response
            response = self._read_response()
            return response
            
        except serial.SerialException as e:
            self.logger.error(f"Serial error sending command: {e}")
            return None
    
    def _read_response(self) -> Optional[bytes]:
        """Read response from Kalatec controller"""
        if not self.connection:
            return None
        
        try:
            # Wait for header
            while self.connection.in_waiting > 0:
                byte = self.connection.read(1)
                if byte[0] == self.HEADER:
                    break
            
            # Read packet
            packet = bytearray()
            packet.append(self.HEADER)
            
            # Read command, length, data, checksum, footer
            command = self.connection.read(1)
            if not command:
                return None
            packet.extend(command)
            
            length = self.connection.read(1)
            if not length:
                return None
            packet.extend(length)
            
            data = self.connection.read(length[0])
            packet.extend(data)
            
            checksum = self.connection.read(1)
            if not checksum:
                return None
            packet.extend(checksum)
            
            footer = self.connection.read(1)
            if not footer:
                return None
            packet.extend(footer)
            
            # Verify checksum and footer
            calculated_checksum = sum(packet[1:-2]) & 0xFF
            if checksum[0] != calculated_checksum or footer[0] != self.FOOTER:
                self.logger.warning("Invalid packet checksum or footer")
                return None
            
            return bytes(data)
            
        except serial.SerialException as e:
            self.logger.error(f"Serial error reading response: {e}")
            return None
    
    def get_status(self) -> Optional[KalatecStatus]:
        """Get motor status"""
        response = self._send_command(KalatecCommand.GET_STATUS)
        if not response or len(response) < 28:
            return None
        
        try:
            # Unpack response data
            # Format: position(4), velocity(4), torque(4), temp(2), voltage(2), current(2), error(4), mode(1)
            values = struct.unpack('<ffffHHHB', response[:23])
            
            return KalatecStatus(
                position=values[0],
                velocity=values[1],
                torque=values[2],
                temperature=values[3],
                voltage=values[4] / 10.0,  # Convert to volts
                current=values[5] / 100.0,  # Convert to amps
                error_flags=values[6],
                mode=values[7]
            )
            
        except struct.error as e:
            self.logger.error(f"Error unpacking status data: {e}")
            return None
    
    def set_position(self, position: float) -> bool:
        """Set target position"""
        data = struct.pack('<f', position)
        response = self._send_command(KalatecCommand.SET_POSITION, data)
        return response is not None
    
    def set_velocity(self, velocity: float) -> bool:
        """Set target velocity"""
        data = struct.pack('<f', velocity)
        response = self._send_command(KalatecCommand.SET_VELOCITY, data)
        return response is not None
    
    def set_torque(self, torque: float) -> bool:
        """Set target torque"""
        data = struct.pack('<f', torque)
        response = self._send_command(KalatecCommand.SET_TORQUE, data)
        return response is not None
    
    def home(self) -> bool:
        """Execute homing sequence"""
        response = self._send_command(KalatecCommand.HOME)
        return response is not None
    
    def stop(self) -> bool:
        """Emergency stop"""
        response = self._send_command(KalatecCommand.STOP)
        return response is not None
    
    def reset(self) -> bool:
        """Reset controller"""
        response = self._send_command(KalatecCommand.RESET)
        return response is not None
    
    def set_pid(self, kp: float, ki: float, kd: float) -> bool:
        """Set PID parameters"""
        data = struct.pack('<fff', kp, ki, kd)
        response = self._send_command(KalatecCommand.SET_PID, data)
        return response is not None


class KalatecMotor:
    """Kalatec motor wrapper"""
    
    def __init__(self, motor_id: str, config: MotorConfig, protocol: KalatecProtocol):
        self.motor_id = motor_id
        self.config = config
        self.protocol = protocol
        self.status = None
        self.last_update = 0
        self.update_interval = 0.01  # 100 Hz
        
    def update_status(self) -> bool:
        """Update motor status from controller"""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return True
        
        try:
            status = self.protocol.get_status()
            if status:
                self.status = status
                self.last_update = current_time
                return True
        except Exception as e:
            logging.error(f"Error updating {self.motor_id} status: {e}")
        
        return False
    
    def move_to(self, position: float) -> bool:
        """Move motor to position"""
        # Clamp to limits
        position = max(min(position, self.config.max_position), self.config.min_position)
        return self.protocol.set_position(position)
    
    def set_velocity(self, velocity: float) -> bool:
        """Set motor velocity"""
        # Clamp to limits
        velocity = max(min(velocity, self.config.max_speed), -self.config.max_speed)
        return self.protocol.set_velocity(velocity)
    
    def stop(self) -> bool:
        """Stop motor"""
        return self.protocol.stop()
    
    def home(self) -> bool:
        """Home motor"""
        return self.protocol.home()
    
    def get_position(self) -> Optional[float]:
        """Get current position"""
        if self.update_status():
            return self.status.position if self.status else None
        return None
    
    def get_velocity(self) -> Optional[float]:
        """Get current velocity"""
        if self.update_status():
            return self.status.velocity if self.status else None
        return None
    
    def get_torque(self) -> Optional[float]:
        """Get current torque"""
        if self.update_status():
            return self.status.torque if self.status else None
        return None
    
    def get_temperature(self) -> Optional[float]:
        """Get motor temperature"""
        if self.update_status():
            return self.status.temperature if self.status else None
        return None
    
    def is_moving(self) -> bool:
        """Check if motor is moving"""
        if self.status:
            return abs(self.status.velocity) > 0.01
        return False
    
    def has_error(self) -> bool:
        """Check if motor has error"""
        if self.status:
            return self.status.error_flags != 0
        return False


class KalatecInterface:
    """
    Interface for managing multiple Kalatec motors
    """
    
    def __init__(self):
        self.motors: Dict[str, KalatecMotor] = {}
        self.protocols: Dict[str, KalatecProtocol] = {}
        self.running = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
    
    def add_motor(self, motor_id: str, config: MotorConfig, serial_port: str) -> bool:
        """Add a Kalatec motor"""
        try:
            # Create protocol if not exists for this port
            if serial_port not in self.protocols:
                protocol = KalatecProtocol(serial_port)
                if not protocol.connect():
                    self.logger.error(f"Failed to connect to {serial_port}")
                    return False
                self.protocols[serial_port] = protocol
            
            # Create motor
            motor = KalatecMotor(motor_id, config, self.protocols[serial_port])
            self.motors[motor_id] = motor
            
            self.logger.info(f"Added Kalatec motor: {motor_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add motor {motor_id}: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize Kalatec interface"""
        try:
            if not self.motors:
                self.logger.warning("No motors configured")
                return False
            
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            # Test all motors
            for motor_id, motor in self.motors.items():
                if not motor.update_status():
                    self.logger.warning(f"Failed to get status from {motor_id}")
            
            self.logger.info("Kalatec interface initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kalatec interface: {e}")
            return False
    
    def _monitor_loop(self):
        """Monitor motor status"""
        while self.running:
            try:
                for motor_id, motor in self.motors.items():
                    motor.update_status()
                
                time.sleep(0.01)  # 100 Hz update rate
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(0.1)
    
    def move_motor(self, motor_id: str, position: float) -> bool:
        """Move specific motor"""
        if motor_id not in self.motors:
            return False
        
        return self.motors[motor_id].move_to(position)
    
    def move_motors(self, movements: Dict[str, float]) -> bool:
        """Move multiple motors"""
        success = True
        for motor_id, position in movements.items():
            if not self.move_motor(motor_id, position):
                success = False
        
        return success
    
    def set_motor_velocity(self, motor_id: str, velocity: float) -> bool:
        """Set motor velocity"""
        if motor_id not in self.motors:
            return False
        
        return self.motors[motor_id].set_velocity(velocity)
    
    def stop_motor(self, motor_id: str) -> bool:
        """Stop specific motor"""
        if motor_id not in self.motors:
            return False
        
        return self.motors[motor_id].stop()
    
    def stop_all(self) -> bool:
        """Stop all motors"""
        success = True
        for motor_id in self.motors:
            if not self.stop_motor(motor_id):
                success = False
        
        return success
    
    def home_motor(self, motor_id: str) -> bool:
        """Home specific motor"""
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
    
    def get_motor_status(self, motor_id: str) -> Optional[KalatecStatus]:
        """Get motor status"""
        if motor_id not in self.motors:
            return None
        
        motor = self.motors[motor_id]
        motor.update_status()
        return motor.status
    
    def get_all_status(self) -> Dict[str, KalatecStatus]:
        """Get all motor status"""
        status = {}
        for motor_id, motor in self.motors.items():
            motor.update_status()
            status[motor_id] = motor.status
        
        return status
    
    def is_movement_complete(self, motor_id: str = None) -> bool:
        """Check if movement is complete"""
        if motor_id:
            if motor_id not in self.motors:
                return True
            motor = self.motors[motor_id]
            return not motor.is_moving()
        else:
            return all(not motor.is_moving() for motor in self.motors.values())
    
    def wait_for_movement(self, motor_id: str = None, timeout: float = 30.0) -> bool:
        """Wait for movement to complete"""
        start_time = time.time()
        
        while not self.is_movement_complete(motor_id):
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.1)
        
        return True
    
    def check_errors(self) -> Dict[str, int]:
        """Check for motor errors"""
        errors = {}
        for motor_id, motor in self.motors.items():
            if motor.has_error():
                errors[motor_id] = motor.status.error_flags
        
        return errors
    
    def clear_errors(self) -> bool:
        """Clear all motor errors"""
        success = True
        for protocol in self.protocols.values():
            if not protocol.reset():
                success = False
        
        return success
    
    def shutdown(self):
        """Shutdown Kalatec interface"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # Stop all motors
        self.stop_all()
        
        # Close all connections
        for protocol in self.protocols.values():
            protocol.disconnect()
        
        self.logger.info("Kalatec interface shutdown complete")
