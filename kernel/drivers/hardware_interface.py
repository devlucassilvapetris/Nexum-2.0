"""
Nexum v2.0 Hardware Interface
Direct hardware communication and control
"""

import serial
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available. Running in simulation mode.")

import usb.core
import usb.util


class HardwareType(Enum):
    MOTOR = "motor"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    AUDIO = "audio"
    CAMERA = "camera"
    GPIO = "gpio"


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class HardwareDevice:
    id: str
    name: str
    type: HardwareType
    connection_type: str  # serial, usb, gpio, i2c, spi
    address: str
    state: ConnectionState
    last_heartbeat: float
    data_buffer: List[Dict[str, Any]]


class HardwareInterface:
    """
    Central hardware interface for Nexum v2.0
    Manages all hardware connections and data flow
    """
    
    def __init__(self):
        self.devices: Dict[str, HardwareDevice] = {}
        self.serial_connections: Dict[str, serial.Serial] = {}
        self.usb_devices: Dict[str, usb.core.Device] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        self.monitoring_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPIO if available
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
    
    def initialize(self) -> bool:
        """Initialize hardware interface and start monitoring"""
        try:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitor_devices, daemon=True)
            self.monitoring_thread.start()
            
            # Auto-detect available hardware
            self._detect_hardware()
            
            self.logger.info("Hardware interface initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hardware interface: {e}")
            return False
    
    def _detect_hardware(self):
        """Auto-detect available hardware devices"""
        # Detect serial devices
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self._register_serial_device(port.device, port.description)
        except Exception as e:
            self.logger.error(f"Error detecting serial devices: {e}")
        
        # Detect USB devices
        try:
            devices = usb.core.find(find_all=True)
            for device in devices:
                if device.idVendor and device.idProduct:
                    self._register_usb_device(device)
        except Exception as e:
            self.logger.error(f"Error detecting USB devices: {e}")
    
    def _register_serial_device(self, port: str, description: str):
        """Register a serial device"""
        device_id = f"serial_{port.replace('/', '_').replace('\\', '_')}"
        device = HardwareDevice(
            id=device_id,
            name=description or f"Serial Device on {port}",
            type=HardwareType.SENSOR,
            connection_type="serial",
            address=port,
            state=ConnectionState.DISCONNECTED,
            last_heartbeat=0,
            data_buffer=[]
        )
        self.devices[device_id] = device
        self.logger.info(f"Registered serial device: {device.name}")
    
    def _register_usb_device(self, device: usb.core.Device):
        """Register a USB device"""
        device_id = f"usb_{device.idVendor:04x}_{device.idProduct:04x}"
        device_name = f"USB Device {device.idVendor:04x}:{device.idProduct:04x}"
        
        hw_device = HardwareDevice(
            id=device_id,
            name=device_name,
            type=HardwareType.SENSOR,
            connection_type="usb",
            address=f"{device.idVendor:04x}:{device.idProduct:04x}",
            state=ConnectionState.DISCONNECTED,
            last_heartbeat=0,
            data_buffer=[]
        )
        self.devices[device_id] = hw_device
        self.usb_devices[device_id] = device
        self.logger.info(f"Registered USB device: {device_name}")
    
    def connect_device(self, device_id: str, **kwargs) -> bool:
        """Connect to a specific hardware device"""
        if device_id not in self.devices:
            self.logger.error(f"Device {device_id} not found")
            return False
        
        device = self.devices[device_id]
        
        try:
            if device.connection_type == "serial":
                return self._connect_serial_device(device_id, **kwargs)
            elif device.connection_type == "usb":
                return self._connect_usb_device(device_id, **kwargs)
            elif device.connection_type == "gpio":
                return self._connect_gpio_device(device_id, **kwargs)
            else:
                self.logger.error(f"Unsupported connection type: {device.connection_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect device {device_id}: {e}")
            device.state = ConnectionState.ERROR
            return False
    
    def _connect_serial_device(self, device_id: str, baudrate: int = 9600, timeout: float = 1.0) -> bool:
        """Connect to a serial device"""
        device = self.devices[device_id]
        
        try:
            ser = serial.Serial(
                port=device.address,
                baudrate=baudrate,
                timeout=timeout
            )
            
            self.serial_connections[device_id] = ser
            device.state = ConnectionState.CONNECTED
            device.last_heartbeat = time.time()
            
            self.logger.info(f"Connected to serial device: {device.name}")
            return True
            
        except serial.SerialException as e:
            self.logger.error(f"Serial connection failed: {e}")
            device.state = ConnectionState.ERROR
            return False
    
    def _connect_usb_device(self, device_id: str) -> bool:
        """Connect to a USB device"""
        device = self.devices[device_id]
        
        try:
            usb_device = self.usb_devices[device_id]
            
            # Try to claim the device
            for cfg in usb_device:
                usb_device.set_configuration()
                break
            
            device.state = ConnectionState.CONNECTED
            device.last_heartbeat = time.time()
            
            self.logger.info(f"Connected to USB device: {device.name}")
            return True
            
        except usb.core.USBError as e:
            self.logger.error(f"USB connection failed: {e}")
            device.state = ConnectionState.ERROR
            return False
    
    def _connect_gpio_device(self, device_id: str, pin: int, mode: str = "input") -> bool:
        """Connect to a GPIO device"""
        if not GPIO_AVAILABLE:
            self.logger.error("GPIO not available")
            return False
        
        device = self.devices[device_id]
        
        try:
            if mode.lower() == "input":
                GPIO.setup(pin, GPIO.IN)
            elif mode.lower() == "output":
                GPIO.setup(pin, GPIO.OUT)
            else:
                raise ValueError(f"Invalid GPIO mode: {mode}")
            
            device.state = ConnectionState.CONNECTED
            device.last_heartbeat = time.time()
            
            self.logger.info(f"Connected GPIO device on pin {pin}")
            return True
            
        except Exception as e:
            self.logger.error(f"GPIO connection failed: {e}")
            device.state = ConnectionState.ERROR
            return False
    
    def send_command(self, device_id: str, command: str, parameters: Dict[str, Any] = None) -> bool:
        """Send command to hardware device"""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        
        if device.state != ConnectionState.CONNECTED:
            self.logger.error(f"Device {device_id} not connected")
            return False
        
        try:
            message = {
                "command": command,
                "timestamp": time.time(),
                "parameters": parameters or {}
            }
            
            if device.connection_type == "serial":
                return self._send_serial_command(device_id, message)
            elif device.connection_type == "usb":
                return self._send_usb_command(device_id, message)
            elif device.connection_type == "gpio":
                return self._send_gpio_command(device_id, command, parameters)
            
        except Exception as e:
            self.logger.error(f"Failed to send command to {device_id}: {e}")
            return False
    
    def _send_serial_command(self, device_id: str, message: Dict[str, Any]) -> bool:
        """Send command via serial"""
        if device_id not in self.serial_connections:
            return False
        
        ser = self.serial_connections[device_id]
        try:
            command_str = json.dumps(message) + "\n"
            ser.write(command_str.encode())
            return True
        except serial.SerialException as e:
            self.logger.error(f"Serial write error: {e}")
            return False
    
    def _send_usb_command(self, device_id: str, message: Dict[str, Any]) -> bool:
        """Send command via USB"""
        # USB command implementation would depend on specific device protocol
        self.logger.info(f"USB command sent: {message}")
        return True
    
    def _send_gpio_command(self, device_id: str, command: str, parameters: Dict[str, Any]) -> bool:
        """Send command via GPIO"""
        if not GPIO_AVAILABLE:
            return False
        
        try:
            pin = parameters.get("pin")
            if pin is None:
                return False
            
            if command.lower() == "set_high":
                GPIO.output(pin, GPIO.HIGH)
            elif command.lower() == "set_low":
                GPIO.output(pin, GPIO.LOW)
            elif command.lower() == "toggle":
                current = GPIO.input(pin)
                GPIO.output(pin, not current)
            
            return True
            
        except Exception as e:
            self.logger.error(f"GPIO command error: {e}")
            return False
    
    def read_data(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Read data from hardware device"""
        if device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        
        if device.state != ConnectionState.CONNECTED:
            return None
        
        try:
            if device.connection_type == "serial":
                return self._read_serial_data(device_id)
            elif device.connection_type == "usb":
                return self._read_usb_data(device_id)
            elif device.connection_type == "gpio":
                return self._read_gpio_data(device_id)
                
        except Exception as e:
            self.logger.error(f"Failed to read data from {device_id}: {e}")
            return None
    
    def _read_serial_data(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Read data from serial device"""
        if device_id not in self.serial_connections:
            return None
        
        ser = self.serial_connections[device_id]
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode().strip()
                if line:
                    try:
                        data = json.loads(line)
                        return data
                    except json.JSONDecodeError:
                        return {"raw_data": line}
        except serial.SerialException as e:
            self.logger.error(f"Serial read error: {e}")
        
        return None
    
    def _read_usb_data(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Read data from USB device"""
        # USB data reading would depend on specific device protocol
        return {"timestamp": time.time(), "device_id": device_id}
    
    def _read_gpio_data(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Read data from GPIO device"""
        if not GPIO_AVAILABLE:
            return None
        
        try:
            # This is a simplified example - actual implementation would depend on specific GPIO setup
            return {"timestamp": time.time(), "device_id": device_id, "value": 0}
        except Exception as e:
            self.logger.error(f"GPIO read error: {e}")
            return None
    
    def _monitor_devices(self):
        """Monitor device connections and health"""
        while self.running:
            try:
                current_time = time.time()
                
                for device_id, device in self.devices.items():
                    # Check device heartbeat
                    if device.state == ConnectionState.CONNECTED:
                        if current_time - device.last_heartbeat > 30:  # 30 second timeout
                            self.logger.warning(f"Device {device.name} timeout, marking as disconnected")
                            device.state = ConnectionState.DISCONNECTED
                    
                    # Read data from connected devices
                    if device.state == ConnectionState.CONNECTED:
                        data = self.read_data(device_id)
                        if data:
                            device.data_buffer.append(data)
                            # Keep buffer size manageable
                            if len(device.data_buffer) > 100:
                                device.data_buffer.pop(0)
                            
                            # Trigger callbacks
                            self._trigger_callbacks(device_id, data)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Device monitoring error: {e}")
                time.sleep(5)
    
    def _trigger_callbacks(self, device_id: str, data: Dict[str, Any]):
        """Trigger registered callbacks for device data"""
        if device_id in self.callbacks:
            for callback in self.callbacks[device_id]:
                try:
                    callback(device_id, data)
                except Exception as e:
                    self.logger.error(f"Callback error for device {device_id}: {e}")
    
    def register_callback(self, device_id: str, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for device data"""
        if device_id not in self.callbacks:
            self.callbacks[device_id] = []
        self.callbacks[device_id].append(callback)
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific device"""
        if device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        return {
            "id": device.id,
            "name": device.name,
            "type": device.type.value,
            "connection_type": device.connection_type,
            "address": device.address,
            "state": device.state.value,
            "last_heartbeat": device.last_heartbeat,
            "buffer_size": len(device.data_buffer)
        }
    
    def get_all_devices(self) -> List[Dict[str, Any]]:
        """Get status of all devices"""
        return [self.get_device_status(device_id) for device_id in self.devices]
    
    def disconnect_device(self, device_id: str) -> bool:
        """Disconnect from a hardware device"""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        
        try:
            if device.connection_type == "serial" and device_id in self.serial_connections:
                self.serial_connections[device_id].close()
                del self.serial_connections[device_id]
            
            device.state = ConnectionState.DISCONNECTED
            self.logger.info(f"Disconnected device: {device.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to disconnect device {device_id}: {e}")
            return False
    
    def shutdown(self):
        """Shutdown hardware interface"""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Close all connections
        for device_id in list(self.devices.keys()):
            self.disconnect_device(device_id)
        
        # Cleanup GPIO
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        
        self.logger.info("Hardware interface shutdown complete")
