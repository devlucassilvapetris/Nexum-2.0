"""
Nexum v2.0 Kernel Module
Hybrid Operating System Core
"""

__version__ = "2.0.0"
__author__ = "Nexum Development Team"

from .core.system_manager import SystemManager
from .communication.protocol import CommunicationProtocol
from .drivers.hardware_interface import HardwareInterface

__all__ = [
    "SystemManager",
    "CommunicationProtocol", 
    "HardwareInterface"
]
