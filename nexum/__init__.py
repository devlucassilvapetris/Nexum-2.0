"""
Nexum v2.0 - Hybrid Operating System
Main entry point for the Nexum system
"""

__version__ = "2.0.0"
__author__ = "Nexum Development Team"

from .core import NexumSystem
from .cli import NexumCLI

__all__ = [
    "NexumSystem",
    "NexumCLI"
]
