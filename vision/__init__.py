"""
Nexum v2.0 Vision Module
Computer vision and image processing systems
"""

__version__ = "2.0.0"

from .processing.image_processor import ImageProcessor
from .processing.video_stream import VideoStream
from .recognition.object_detector import ObjectDetector
from .recognition.face_recognizer import FaceRecognizer
from .recognition.scene_analyzer import SceneAnalyzer

__all__ = [
    "ImageProcessor",
    "VideoStream",
    "ObjectDetector",
    "FaceRecognizer", 
    "SceneAnalyzer"
]
