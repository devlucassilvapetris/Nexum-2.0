"""
Nexum v2.0 System Manager
Core system management and resource allocation
"""

import os
import sys
import threading
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import psutil
import GPUtil


class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    SUSPENDED = "suspended"
    SHUTDOWN = "shutdown"


@dataclass
class SystemResources:
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    available_memory: int
    gpu_memory: int
    temperature: float


class SystemManager:
    """
    Core system manager for Nexum v2.0
    Handles resource monitoring, process management, and system state
    """
    
    def __init__(self):
        self.state = SystemState.INITIALIZING
        self.processes: Dict[str, Any] = {}
        self.resources = SystemResources(0, 0, 0, 0, 0, 0)
        self.monitoring_thread = None
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize the system manager and start monitoring"""
        try:
            print("Initializing Nexum v2.0 Kernel...")
            
            # Check system requirements
            if not self._check_system_requirements():
                return False
                
            # Start resource monitoring
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitoring_thread.start()
            
            self.state = SystemState.RUNNING
            print("System manager initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize system manager: {e}")
            return False
    
    def _check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements"""
        # Check Python version
        if sys.version_info < (3, 8):
            print("Python 3.8 or higher required")
            return False
            
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024 * 1024 * 1024:  # 4GB minimum
            print("Minimum 4GB RAM required")
            return False
            
        # Check GPU availability (optional but recommended)
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                print("Warning: No GPU detected. CPU-only mode")
        except:
            print("Warning: GPU monitoring not available")
            
        return True
    
    def _monitor_resources(self):
        """Monitor system resources in background thread"""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                available_memory = memory.available
                
                # GPU usage
                gpu_usage = 0
                gpu_memory = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_usage = gpu.load * 100
                        gpu_memory = gpu.memoryFree
                except:
                    pass
                
                # Temperature (if available)
                temperature = 0
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if entries:
                                temperature = entries[0].current
                                break
                except:
                    pass
                
                self.resources = SystemResources(
                    cpu_usage=cpu_percent,
                    memory_usage=memory_percent,
                    gpu_usage=gpu_usage,
                    available_memory=available_memory,
                    gpu_memory=gpu_memory,
                    temperature=temperature
                )
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                time.sleep(5)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "state": self.state.value,
            "resources": {
                "cpu_usage": self.resources.cpu_usage,
                "memory_usage": self.resources.memory_usage,
                "gpu_usage": self.resources.gpu_usage,
                "available_memory_gb": self.resources.available_memory / (1024**3),
                "gpu_memory_mb": self.resources.gpu_memory,
                "temperature_celsius": self.resources.temperature
            },
            "processes": list(self.processes.keys())
        }
    
    def register_process(self, name: str, process: Any) -> bool:
        """Register a new process with the system manager"""
        try:
            self.processes[name] = {
                "process": process,
                "pid": getattr(process, "pid", None),
                "start_time": time.time(),
                "status": "running"
            }
            print(f"Process '{name}' registered successfully")
            return True
        except Exception as e:
            print(f"Failed to register process '{name}': {e}")
            return False
    
    def shutdown(self) -> bool:
        """Gracefully shutdown the system"""
        print("Shutting down Nexum v2.0 Kernel...")
        
        self.state = SystemState.SHUTDOWN
        self.running = False
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Shutdown all registered processes
        for name, proc_info in self.processes.items():
            try:
                process = proc_info["process"]
                if hasattr(process, "terminate"):
                    process.terminate()
                elif hasattr(process, "stop"):
                    process.stop()
                print(f"Process '{name}' terminated")
            except Exception as e:
                print(f"Error terminating process '{name}': {e}")
        
        print("System shutdown complete")
        return True


def main():
    """Main entry point for the Nexum kernel"""
    system_manager = SystemManager()
    
    if system_manager.initialize():
        try:
            while system_manager.state == SystemState.RUNNING:
                status = system_manager.get_system_status()
                print(f"System Status: {status['state']} | "
                      f"CPU: {status['resources']['cpu_usage']:.1f}% | "
                      f"Memory: {status['resources']['memory_usage']:.1f}%")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        finally:
            system_manager.shutdown()
    else:
        print("Failed to initialize system")
        sys.exit(1)


if __name__ == "__main__":
    main()
