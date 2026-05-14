"""
Nexum v2.0 Core System
Main system controller and integration point
"""

import asyncio
import logging
import signal
import sys
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from kernel import SystemManager, CommunicationProtocol, HardwareInterface
from hardware import MotorController, BiomimeticEars, MechanicalController
from vision import ImageProcessor
from ai import LocalModelManager
from hardware.structure import BodyDesigner


class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SystemConfig:
    # Core settings
    log_level: str = "INFO"
    max_memory_mb: int = 8192
    enable_gpu: bool = True
    
    # Hardware settings
    enable_motors: bool = True
    enable_audio: bool = True
    enable_vision: bool = True
    enable_ai: bool = True
    
    # Performance settings
    update_rate_hz: int = 100
    buffer_size: int = 1024
    
    # Safety settings
    emergency_stop_enabled: bool = True
    temperature_limit_c: float = 80.0
    power_limit_w: float = 500.0


class NexumSystem:
    """
    Main Nexum v2.0 system controller
    Integrates all modules and provides unified interface
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.state = SystemState.INITIALIZING
        self.logger = self._setup_logging()
        
        # Core components
        self.system_manager = None
        self.communication = None
        self.hardware_interface = None
        
        # Hardware modules
        self.motor_controller = None
        self.biomimetic_ears = None
        self.mechanical_controller = None
        
        # Processing modules
        self.image_processor = None
        self.ai_manager = None
        
        # Design module
        self.body_designer = None
        
        # Runtime state
        self.running = False
        self.startup_time = 0
        self.shutdown_requested = False
        
        # Statistics
        self.stats = {
            "uptime": 0,
            "total_operations": 0,
            "errors": 0,
            "warnings": 0
        }
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nexum.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown")
        self.shutdown_requested = True
        self.stop()
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing Nexum v2.0 System...")
            self.state = SystemState.INITIALIZING
            self.startup_time = time.time()
            
            # Initialize core components
            if not await self._initialize_core():
                return False
            
            # Initialize hardware modules
            if not await self._initialize_hardware():
                return False
            
            # Initialize processing modules
            if not await self._initialize_processing():
                return False
            
            # Initialize design module
            if not await self._initialize_design():
                return False
            
            # Start system monitoring
            asyncio.create_task(self._system_monitor())
            
            self.state = SystemState.RUNNING
            self.running = True
            self.logger.info("Nexum v2.0 System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.state = SystemState.ERROR
            return False
    
    async def _initialize_core(self) -> bool:
        """Initialize core system components"""
        try:
            # System Manager
            self.system_manager = SystemManager()
            if not self.system_manager.initialize():
                return False
            
            # Communication Protocol
            self.communication = CommunicationProtocol("nexum_core")
            if not await self.communication.initialize():
                return False
            
            # Hardware Interface
            self.hardware_interface = HardwareInterface()
            if not self.hardware_interface.initialize():
                return False
            
            self.logger.info("Core components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Core initialization failed: {e}")
            return False
    
    async def _initialize_hardware(self) -> bool:
        """Initialize hardware modules"""
        try:
            if self.config.enable_motors:
                # Motor Controller
                self.motor_controller = MotorController()
                if not self.motor_controller.initialize():
                    return False
                
                # Mechanical Controller
                self.mechanical_controller = MechanicalController()
                if not self.mechanical_controller.initialize():
                    return False
            
            if self.config.enable_audio:
                # Biomimetic Ears
                from hardware.acoustic import AudioConfig
                audio_config = AudioConfig(
                    sample_rate=44100,
                    buffer_size=self.config.buffer_size,
                    channels=2
                )
                self.biomimetic_ears = BiomimeticEars(audio_config)
                if not self.biomimetic_ears.initialize():
                    return False
                
                if not self.biomimetic_ears.start_listening():
                    return False
            
            self.logger.info("Hardware modules initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            return False
    
    async def _initialize_processing(self) -> bool:
        """Initialize processing modules"""
        try:
            if self.config.enable_vision:
                # Image Processor
                from vision.processing import ImageConfig
                vision_config = ImageConfig(
                    max_resolution=(1920, 1080),
                    processing_threads=4,
                    enable_gpu=self.config.enable_gpu
                )
                self.image_processor = ImageProcessor(vision_config)
                if not self.image_processor.initialize():
                    return False
            
            if self.config.enable_ai:
                # AI Manager
                self.ai_manager = LocalModelManager(
                    cache_size_mb=1024,
                    memory_limit_mb=self.config.max_memory_mb
                )
                if not self.ai_manager.initialize():
                    return False
            
            self.logger.info("Processing modules initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Processing initialization failed: {e}")
            return False
    
    async def _initialize_design(self) -> bool:
        """Initialize design module"""
        try:
            self.body_designer = BodyDesigner()
            
            # Design default humanoid body
            design = self.body_designer.design_humanoid_body(
                height=1.75,
                weight_target=75
            )
            
            self.logger.info("Design module initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Design initialization failed: {e}")
            return False
    
    async def run(self) -> bool:
        """Run the main system loop"""
        if not self.running:
            if not await self.initialize():
                return False
        
        try:
            self.logger.info("Starting Nexum v2.0 main loop...")
            
            while self.running and not self.shutdown_requested:
                # Update system statistics
                await self._update_stats()
                
                # Process system events
                await self._process_events()
                
                # Check system health
                await self._check_health()
                
                # Sleep to maintain update rate
                await asyncio.sleep(1.0 / self.config.update_rate_hz)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.state = SystemState.ERROR
            return False
    
    async def _system_monitor(self):
        """Background system monitoring"""
        while self.running:
            try:
                # Monitor system resources
                if self.system_manager:
                    status = self.system_manager.get_system_status()
                    resources = status.get('resources', {})
                    
                    # Check memory usage
                    memory_usage = resources.get('memory_usage', 0)
                    if memory_usage > 90:
                        self.logger.warning(f"High memory usage: {memory_usage}%")
                    
                    # Check temperature
                    temperature = resources.get('temperature_celsius', 0)
                    if temperature > self.config.temperature_limit_c:
                        self.logger.warning(f"High temperature: {temperature}°C")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"System monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _update_stats(self):
        """Update system statistics"""
        self.stats["uptime"] = time.time() - self.startup_time
        self.stats["total_operations"] += 1
    
    async def _process_events(self):
        """Process system events"""
        # This would handle incoming commands, data processing, etc.
        pass
    
    async def _check_health(self):
        """Check system health"""
        try:
            # Check all components
            components = [
                ("System Manager", self.system_manager),
                ("Motor Controller", self.motor_controller),
                ("Image Processor", self.image_processor),
                ("AI Manager", self.ai_manager)
            ]
            
            for name, component in components:
                if component and hasattr(component, 'get_status'):
                    try:
                        status = component.get_status()
                        if status.get('state') == 'error':
                            self.logger.error(f"Component {name} in error state")
                            self.stats["errors"] += 1
                    except Exception as e:
                        self.logger.error(f"Error checking {name}: {e}")
                        self.stats["errors"] += 1
        
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
    
    async def stop(self):
        """Stop the system gracefully"""
        if not self.running:
            return
        
        self.logger.info("Stopping Nexum v2.0 System...")
        self.state = SystemState.STOPPING
        self.running = False
        
        # Stop all components
        await self._shutdown_components()
        
        self.state = SystemState.STOPPED
        self.logger.info("Nexum v2.0 System stopped")
    
    async def _shutdown_components(self):
        """Shutdown all system components"""
        shutdown_tasks = []
        
        # Shutdown hardware
        if self.motor_controller:
            shutdown_tasks.append(self.motor_controller.shutdown())
        
        if self.biomimetic_ears:
            shutdown_tasks.append(self.biomimetic_ears.shutdown())
        
        if self.mechanical_controller:
            shutdown_tasks.append(self.mechanical_controller.shutdown())
        
        if self.hardware_interface:
            shutdown_tasks.append(self.hardware_interface.shutdown())
        
        # Shutdown processing
        if self.image_processor:
            shutdown_tasks.append(self.image_processor.shutdown())
        
        if self.ai_manager:
            shutdown_tasks.append(self.ai_manager.shutdown())
        
        # Shutdown core
        if self.communication:
            shutdown_tasks.append(self.communication.shutdown())
        
        if self.system_manager:
            shutdown_tasks.append(self.system_manager.shutdown())
        
        # Wait for all shutdowns to complete
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            "state": self.state.value,
            "uptime": self.stats["uptime"],
            "total_operations": self.stats["total_operations"],
            "errors": self.stats["errors"],
            "warnings": self.stats["warnings"],
            "components": {
                "system_manager": self.system_manager.get_system_status() if self.system_manager else None,
                "motor_controller": self.motor_controller.get_all_status() if self.motor_controller else None,
                "image_processor": self.image_processor.get_statistics() if self.image_processor else None,
                "ai_manager": self.ai_manager.get_all_status() if self.ai_manager else None
            }
        }
    
    async def execute_command(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute system command"""
        try:
            self.stats["total_operations"] += 1
            
            if command == "status":
                return {"success": True, "data": self.get_status()}
            
            elif command == "move_motor":
                if not self.motor_controller:
                    return {"success": False, "error": "Motor controller not available"}
                
                motor_id = parameters.get("motor_id")
                position = parameters.get("position")
                success = self.motor_controller.move_motor(motor_id, position)
                return {"success": success}
            
            elif command == "process_image":
                if not self.image_processor:
                    return {"success": False, "error": "Image processor not available"}
                
                # This would need actual image data
                return {"success": False, "error": "Image data required"}
            
            elif command == "ai_inference":
                if not self.ai_manager:
                    return {"success": False, "error": "AI manager not available"}
                
                model_id = parameters.get("model_id")
                inputs = parameters.get("inputs")
                result = self.ai_manager.infer(model_id, inputs)
                return {"success": result is not None, "data": result}
            
            elif command == "design_body":
                if not self.body_designer:
                    return {"success": False, "error": "Body designer not available"}
                
                height = parameters.get("height", 1.75)
                weight_target = parameters.get("weight_target", 75)
                design = self.body_designer.design_humanoid_body(height, weight_target)
                return {"success": True, "data": design}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
        
        except Exception as e:
            self.logger.error(f"Command execution error: {e}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}


async def main():
    """Main entry point"""
    # Create system with default configuration
    config = SystemConfig(
        log_level="INFO",
        enable_gpu=True,
        enable_motors=True,
        enable_audio=True,
        enable_vision=True,
        enable_ai=True
    )
    
    # Create and run system
    nexum = NexumSystem(config)
    
    try:
        success = await nexum.run()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        await nexum.stop()


if __name__ == "__main__":
    asyncio.run(main())
