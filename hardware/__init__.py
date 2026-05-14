"""
Nexum v2.0 Hardware Module
Hardware control and kinematics systems
"""

__version__ = "2.0.0"

from .kinematics.motor_controller import MotorController
from .kinematics.kalatec_interface import KalatecInterface
from .acoustic.biomimetic_ears import BiomimeticEars
from .acoustic.audio_processor import AudioProcessor
from .structure.mechanical_controller import MechanicalController

__all__ = [
    "MotorController",
    "KalatecInterface", 
    "BiomimeticEars",
    "AudioProcessor",
    "MechanicalController"
]
