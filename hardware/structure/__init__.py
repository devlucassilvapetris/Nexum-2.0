"""
Nexum v2.0 Structure Module
Mechanical structure and body design systems
"""

__version__ = "2.0.0"

from .body_design import BodyDesigner, MaterialType, BodySection
from .mechanical_controller import MechanicalController

__all__ = [
    "BodyDesigner",
    "MaterialType", 
    "BodySection",
    "MechanicalController"
]
