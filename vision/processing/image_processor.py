"""
Nexum v2.0 Image Processor
Advanced image processing and computer vision pipeline
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from PIL import Image
import scipy.ndimage as ndimage


class ImageFormat(Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"
    LAB = "lab"
    YUV = "yuv"


class ProcessingMode(Enum):
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"


class FilterType(Enum):
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"
    MORPHOLOGICAL = "morphological"


@dataclass
class ImageConfig:
    max_resolution: Tuple[int, int] = (1920, 1080)
    processing_threads: int = 4
    color_space: ImageFormat = ImageFormat.BGR
    bit_depth: int = 8
    compression_quality: int = 95
    enable_gpu: bool = True


@dataclass
class ProcessedImage:
    timestamp: float
    data: np.ndarray
    original_shape: Tuple[int, int, int]
    processing_time: float
    metadata: Dict[str, Any]


class ImageEnhancer:
    """Image enhancement and restoration"""
    
    def __init__(self):
        self.clipping_limit = 2.0
        self.tile_grid_size = (8, 8)
    
    def enhance_contrast(self, image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """Enhance image contrast"""
        if method == "clahe":
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=self.clipping_limit, tileGridSize=self.tile_grid_size)
            l = clahe.apply(l)
            
            # Merge channels back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        
        elif method == "histogram_equalization":
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            y = cv2.equalizeHist(y)
            enhanced = cv2.merge([y, u, v])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YUV2BGR)
            
            return enhanced
        
        return image
    
    def reduce_noise(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """Reduce image noise"""
        if method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        elif method == "nlm":
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image
    
    def sharpen_image(self, image: np.ndarray, method: str = "unsharp_mask") -> np.ndarray:
        """Sharpen image"""
        if method == "unsharp_mask":
            # Create unsharp mask
            blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            return sharpened
        elif method == "laplacian":
            # Laplacian sharpening
            laplacian = cv2.Laplacian(image, cv2.CV_32F)
            sharpened = cv2.convertScaleAbs(image - 0.3 * laplacian)
            return sharpened
        
        return image


class FeatureExtractor:
    """Extract visual features from images"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        self.akaze = cv2.AKAZE_create()
    
    def extract_keypoints(self, image: np.ndarray, method: str = "sift") -> Tuple[List, np.ndarray]:
        """Extract keypoints and descriptors"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == "sift":
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        elif method == "orb":
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        elif method == "akaze":
            keypoints, descriptors = self.akaze.detectAndCompute(gray, None)
        else:
            keypoints, descriptors = [], None
        
        return keypoints, descriptors
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Resize for HOG
        resized = cv2.resize(gray, (64, 128))
        
        # Compute HOG
        win_size = (64, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(resized)
        
        return features.flatten()
    
    def extract_color_histogram(self, image: np.ndarray, bins: int = 256) -> np.ndarray:
        """Extract color histogram features"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate histograms
        hist_bgr = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [bins//2, bins//2, bins//2], [0, 180, 0, 256, 0, 256])
        hist_lab = cv2.calcHist([lab], [0, 1, 2], None, [bins//2, bins//2, bins//2], [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        hist_bgr = cv2.normalize(hist_bgr, hist_bgr).flatten()
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
        
        # Concatenate features
        features = np.concatenate([hist_bgr, hist_hsv, hist_lab])
        
        return features


class ImageTransformer:
    """Geometric transformations and image warping"""
    
    def __init__(self):
        pass
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    interpolation: str = "linear") -> np.ndarray:
        """Resize image with specified interpolation"""
        interp_methods = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
            "lanczos": cv2.INTER_LANCZOS4
        }
        
        interp = interp_methods.get(interpolation, cv2.INTER_LINEAR)
        return cv2.resize(image, target_size, interpolation=interp)
    
    def rotate_image(self, image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Rotate image around center or specified point"""
        h, w = image.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return rotated
    
    def affine_transform(self, image: np.ndarray, src_points: np.ndarray, 
                        dst_points: np.ndarray) -> np.ndarray:
        """Apply affine transformation"""
        # Calculate transformation matrix
        transform_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        # Apply transformation
        h, w = image.shape[:2]
        transformed = cv2.warpAffine(image, transform_matrix, (w, h))
        
        return transformed
    
    def perspective_transform(self, image: np.ndarray, src_points: np.ndarray, 
                              dst_points: np.ndarray) -> np.ndarray:
        """Apply perspective transformation"""
        # Calculate transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        h, w = image.shape[:2]
        transformed = cv2.warpPerspective(image, transform_matrix, (w, h))
        
        return transformed


class ImageProcessor:
    """
    Advanced image processing pipeline for Nexum v2.0
    Handles real-time image enhancement, feature extraction, and transformations
    """
    
    def __init__(self, config: ImageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing components
        self.enhancer = ImageEnhancer()
        self.feature_extractor = FeatureExtractor()
        self.transformer = ImageTransformer()
        
        # Processing state
        self.processing_mode = ProcessingMode.REALTIME
        self.running = False
        self.processing_threads = []
        
        # Buffers
        self.input_buffer = []
        self.output_buffer = []
        
        # Statistics
        self.frames_processed = 0
        self.total_processing_time = 0.0
        self.average_fps = 0.0
        
        # GPU acceleration
        self.gpu_available = False
        if config.enable_gpu:
            try:
                cv2.cuda.setDevice(0)
                self.gpu_available = True
                self.logger.info("GPU acceleration enabled")
            except:
                self.logger.warning("GPU not available, using CPU")
    
    def initialize(self) -> bool:
        """Initialize image processor"""
        try:
            self.running = True
            
            # Start processing threads
            for i in range(self.config.processing_threads):
                thread = threading.Thread(target=self._processing_loop, daemon=True)
                thread.start()
                self.processing_threads.append(thread)
            
            self.logger.info(f"Image processor initialized with {self.config.processing_threads} threads")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize image processor: {e}")
            return False
    
    def process_image(self, image: np.ndarray, operations: List[str] = None) -> ProcessedImage:
        """Process single image with specified operations"""
        start_time = time.time()
        
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid image input")
            
            # Store original shape
            original_shape = image.shape
            
            # Convert color space if needed
            processed_image = self._convert_color_space(image)
            
            # Resize if necessary
            if processed_image.shape[:2] != self.config.max_resolution:
                processed_image = self.transformer.resize_image(
                    processed_image, self.config.max_resolution
                )
            
            # Apply processing operations
            if operations:
                for operation in operations:
                    processed_image = self._apply_operation(processed_image, operation)
            
            # Create processed image object
            processing_time = time.time() - start_time
            result = ProcessedImage(
                timestamp=time.time(),
                data=processed_image,
                original_shape=original_shape,
                processing_time=processing_time,
                metadata={
                    "operations": operations or [],
                    "final_shape": processed_image.shape,
                    "gpu_used": self.gpu_available
                }
            )
            
            # Update statistics
            self.frames_processed += 1
            self.total_processing_time += processing_time
            self.average_fps = 1.0 / (self.total_processing_time / self.frames_processed)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image processing error: {e}")
            raise
    
    def _convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """Convert image to configured color space"""
        if self.config.color_space == ImageFormat.RGB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.config.color_space == ImageFormat.GRAY:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.config.color_space == ImageFormat.HSV:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.config.color_space == ImageFormat.LAB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.config.color_space == ImageFormat.YUV:
            return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        return image  # Default BGR
    
    def _apply_operation(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Apply specific processing operation"""
        try:
            if operation.startswith("enhance_contrast"):
                method = operation.split(":")[1] if ":" in operation else "clahe"
                return self.enhancer.enhance_contrast(image, method)
            
            elif operation.startswith("reduce_noise"):
                method = operation.split(":")[1] if ":" in operation else "bilateral"
                return self.enhancer.reduce_noise(image, method)
            
            elif operation.startswith("sharpen"):
                method = operation.split(":")[1] if ":" in operation else "unsharp_mask"
                return self.enhancer.sharpen_image(image, method)
            
            elif operation.startswith("extract_keypoints"):
                method = operation.split(":")[1] if ":" in operation else "sift"
                keypoints, descriptors = self.feature_extractor.extract_keypoints(image, method)
                # Draw keypoints on image
                result = image.copy()
                return cv2.drawKeypoints(result, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            elif operation.startswith("rotate"):
                angle = float(operation.split(":")[1]) if ":" in operation else 0
                return self.transformer.rotate_image(image, angle)
            
            elif operation.startswith("resize"):
                size_str = operation.split(":")[1] if ":" in operation else "640,480"
                width, height = map(int, size_str.split(","))
                return self.transformer.resize_image(image, (width, height))
            
            else:
                self.logger.warning(f"Unknown operation: {operation}")
                return image
                
        except Exception as e:
            self.logger.error(f"Error applying operation {operation}: {e}")
            return image
    
    def _processing_loop(self):
        """Background processing loop"""
        while self.running:
            try:
                if self.input_buffer:
                    image_data, operations = self.input_buffer.pop(0)
                    processed = self.process_image(image_data, operations)
                    self.output_buffer.append(processed)
                
                time.sleep(0.001)  # 1ms processing interval
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                time.sleep(0.01)
    
    def add_image(self, image: np.ndarray, operations: List[str] = None):
        """Add image to processing queue"""
        self.input_buffer.append((image, operations))
        
        # Keep buffer size manageable
        if len(self.input_buffer) > 100:
            self.input_buffer.pop(0)
    
    def get_processed_image(self) -> Optional[ProcessedImage]:
        """Get next processed image"""
        if self.output_buffer:
            return self.output_buffer.pop(0)
        return None
    
    def extract_features(self, image: np.ndarray, feature_types: List[str] = None) -> Dict[str, np.ndarray]:
        """Extract multiple types of features from image"""
        features = {}
        
        if feature_types is None:
            feature_types = ["keypoints", "hog", "histogram"]
        
        for feature_type in feature_types:
            try:
                if feature_type == "keypoints":
                    keypoints, descriptors = self.feature_extractor.extract_keypoints(image)
                    features["keypoints"] = keypoints
                    features["descriptors"] = descriptors
                
                elif feature_type == "hog":
                    hog_features = self.feature_extractor.extract_hog_features(image)
                    features["hog"] = hog_features
                
                elif feature_type == "histogram":
                    hist_features = self.feature_extractor.extract_color_histogram(image)
                    features["histogram"] = hist_features
                
            except Exception as e:
                self.logger.error(f"Error extracting {feature_type} features: {e}")
        
        return features
    
    def batch_process(self, images: List[np.ndarray], operations: List[str] = None) -> List[ProcessedImage]:
        """Process multiple images in batch"""
        results = []
        
        for image in images:
            try:
                processed = self.process_image(image, operations)
                results.append(processed)
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "frames_processed": self.frames_processed,
            "total_processing_time": self.total_processing_time,
            "average_fps": self.average_fps,
            "average_processing_time": self.total_processing_time / max(self.frames_processed, 1),
            "input_buffer_size": len(self.input_buffer),
            "output_buffer_size": len(self.output_buffer),
            "gpu_available": self.gpu_available,
            "processing_threads": len(self.processing_threads)
        }
    
    def set_processing_mode(self, mode: ProcessingMode):
        """Set processing mode"""
        self.processing_mode = mode
    
    def clear_buffers(self):
        """Clear all buffers"""
        self.input_buffer.clear()
        self.output_buffer.clear()
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.frames_processed = 0
        self.total_processing_time = 0.0
        self.average_fps = 0.0
    
    def shutdown(self):
        """Shutdown image processor"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=2)
        
        self.clear_buffers()
        
        self.logger.info("Image processor shutdown complete")
