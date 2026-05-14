"""
Nexum v2.0 Biomimetic Ears
High-fidelity acoustic capture system mimicking human perception
"""

import numpy as np
import sounddevice as sd
import librosa
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum

import scipy.signal
from scipy.fft import fft, fftfreq


class EarState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    ERROR = "error"


class AudioDirection(Enum):
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    REAR = "rear"
    OMNI = "omni"


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    buffer_size: int = 1024
    channels: int = 2  # Stereo
    bit_depth: int = 16
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    latency: str = "low"
    gain_left: float = 1.0
    gain_right: float = 1.0
    noise_threshold: float = 0.01


@dataclass
class AudioFrame:
    timestamp: float
    data: np.ndarray
    sample_rate: int
    channels: int
    direction: AudioDirection


@dataclass
class AudioFeatures:
    energy: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    mfcc: np.ndarray
    direction_estimation: float
    voice_activity: bool


class Beamforming:
    """Beamforming for directional audio processing"""
    
    def __init__(self, sample_rate: int, mic_array: np.ndarray):
        self.sample_rate = sample_rate
        self.mic_array = mic_array  # Microphone positions
        self.speed_of_sound = 343.0  # m/s
        self.steering_vectors = {}
        self._compute_steering_vectors()
    
    def _compute_steering_vectors(self):
        """Compute steering vectors for different directions"""
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)  # 10-degree steps
        
        for angle in angles:
            # Compute delays for each microphone
            delays = []
            for mic_pos in self.mic_array:
                # Calculate time delay based on angle
                x, y = mic_pos
                delay = (x * np.cos(angle) + y * np.sin(angle)) / self.speed_of_sound
                delays.append(delay)
            
            # Create steering vector
            steering_vector = np.exp(-1j * 2 * np.pi * delays * self.sample_rate)
            self.steering_vectors[angle] = steering_vector
    
    def beamform(self, audio_data: np.ndarray, angle: float) -> np.ndarray:
        """Apply beamforming for specific direction"""
        # Find closest steering vector
        angles = np.array(list(self.steering_vectors.keys()))
        idx = np.argmin(np.abs(angles - angle))
        steering_vector = self.steering_vectors[angles[idx]]
        
        # Apply beamforming
        beamformed = np.zeros(audio_data.shape[0], dtype=complex)
        for i in range(audio_data.shape[1]):
            beamformed += steering_vector[i] * audio_data[:, i]
        
        return beamformed
    
    def estimate_direction(self, audio_data: np.ndarray) -> float:
        """Estimate sound source direction"""
        max_power = 0
        best_angle = 0
        
        for angle, steering_vector in self.steering_vectors.items():
            # Apply steering and compute power
            beamformed = np.zeros(audio_data.shape[0], dtype=complex)
            for i in range(audio_data.shape[1]):
                beamformed += steering_vector[i] * audio_data[:, i]
            
            power = np.mean(np.abs(beamformed) ** 2)
            
            if power > max_power:
                max_power = power
                best_angle = angle
        
        return best_angle


class VoiceActivityDetector:
    """Voice activity detection"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.energy_threshold = 0.01
        self.zcr_threshold = 0.1
        self.spectral_threshold = 3000  # Hz
    
    def detect(self, audio_data: np.ndarray) -> bool:
        """Detect voice activity in audio frame"""
        # Compute energy
        energy = np.mean(audio_data ** 2)
        
        # Compute zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
        
        # Compute spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sample_rate)[0])
        
        # Simple decision logic
        voice_activity = (
            energy > self.energy_threshold and
            zcr < self.zcr_threshold and
            spectral_centroid < self.spectral_threshold
        )
        
        return voice_activity


class BiomimeticEars:
    """
    Biomimetic ear system for high-fidelity acoustic capture
    """
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.state = EarState.IDLE
        self.logger = logging.getLogger(__name__)
        
        # Audio streams
        self.input_stream = None
        self.audio_buffer = []
        self.processing_thread = None
        self.running = False
        
        # Microphone array configuration (simulated binaural setup)
        self.mic_array = np.array([
            [-0.1, 0],   # Left ear
            [0.1, 0]     # Right ear
        ])
        
        # Processing components
        self.beamforming = Beamforming(config.sample_rate, self.mic_array)
        self.vad = VoiceActivityDetector(config.sample_rate)
        
        # Callbacks
        self.audio_callbacks: List[Callable[[AudioFrame], None]] = []
        self.feature_callbacks: List[Callable[[AudioFeatures], None]] = []
        
        # Statistics
        self.frame_count = 0
        self.voice_frame_count = 0
        self.last_direction = 0.0
    
    def initialize(self) -> bool:
        """Initialize biomimetic ear system"""
        try:
            # Test audio device
            if not self._test_audio_device():
                return False
            
            # Start audio processing
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.state = EarState.LISTENING
            self.logger.info("Biomimetic ears initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize biomimetic ears: {e}")
            self.state = EarState.ERROR
            return False
    
    def _test_audio_device(self) -> bool:
        """Test audio input device"""
        try:
            # List available devices
            devices = sd.query_devices()
            
            if self.config.input_device is None:
                # Find default input device
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        self.config.input_device = i
                        break
            
            if self.config.input_device is None:
                self.logger.error("No input device found")
                return False
            
            # Test recording
            test_data = sd.rec(
                frames=1024,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                input_device_index=self.config.input_device
            )
            sd.wait()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio device test failed: {e}")
            return False
    
    def start_listening(self) -> bool:
        """Start audio capture"""
        if self.state == EarState.LISTENING:
            return True
        
        try:
            self.input_stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                blocksize=self.config.buffer_size,
                device=self.config.input_device,
                latency=self.config.latency,
                callback=self._audio_callback
            )
            
            self.input_stream.start()
            self.state = EarState.LISTENING
            
            self.logger.info("Started audio capture")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            self.state = EarState.ERROR
            return False
    
    def stop_listening(self):
        """Stop audio capture"""
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        
        self.state = EarState.IDLE
        self.logger.info("Stopped audio capture")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Apply gain
        processed_data = indata.copy()
        processed_data[:, 0] *= self.config.gain_left
        processed_data[:, 1] *= self.config.gain_right
        
        # Create audio frame
        frame = AudioFrame(
            timestamp=time.time(),
            data=processed_data,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            direction=AudioDirection.OMNI
        )
        
        # Add to buffer
        self.audio_buffer.append(frame)
        
        # Keep buffer size manageable
        if len(self.audio_buffer) > 100:
            self.audio_buffer.pop(0)
    
    def _processing_loop(self):
        """Audio processing loop"""
        while self.running:
            try:
                if self.audio_buffer:
                    frame = self.audio_buffer.pop(0)
                    
                    # Process audio frame
                    self._process_frame(frame)
                    
                    # Trigger callbacks
                    for callback in self.audio_callbacks:
                        try:
                            callback(frame)
                        except Exception as e:
                            self.logger.error(f"Audio callback error: {e}")
                
                time.sleep(0.001)  # 1ms processing interval
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                time.sleep(0.01)
    
    def _process_frame(self, frame: AudioFrame):
        """Process individual audio frame"""
        try:
            self.state = EarState.PROCESSING
            
            # Extract features
            features = self._extract_features(frame)
            
            # Update direction estimation
            self.last_direction = features.direction_estimation
            
            # Update statistics
            self.frame_count += 1
            if features.voice_activity:
                self.voice_frame_count += 1
            
            # Trigger feature callbacks
            for callback in self.feature_callbacks:
                try:
                    callback(features)
                except Exception as e:
                    self.logger.error(f"Feature callback error: {e}")
            
            self.state = EarState.LISTENING
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            self.state = EarState.ERROR
    
    def _extract_features(self, frame: AudioFrame) -> AudioFeatures:
        """Extract audio features from frame"""
        data = frame.data
        sample_rate = frame.sample_rate
        
        # Convert to mono for feature extraction
        mono_data = np.mean(data, axis=1)
        
        # Energy
        energy = np.mean(mono_data ** 2)
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(mono_data)[0])
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(
            y=mono_data, sr=sample_rate)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
            y=mono_data, sr=sample_rate)[0])
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
            y=mono_data, sr=sample_rate)[0])
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=mono_data, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Direction estimation
        direction = self.beamforming.estimate_direction(data)
        
        # Voice activity detection
        voice_activity = self.vad.detect(mono_data)
        
        return AudioFeatures(
            energy=energy,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            mfcc=mfcc_mean,
            direction_estimation=direction,
            voice_activity=voice_activity
        )
    
    def register_audio_callback(self, callback: Callable[[AudioFrame], None]):
        """Register audio frame callback"""
        self.audio_callbacks.append(callback)
    
    def register_feature_callback(self, callback: Callable[[AudioFeatures], None]):
        """Register feature callback"""
        self.feature_callbacks.append(callback)
    
    def get_direction(self) -> float:
        """Get current sound direction estimation"""
        return self.last_direction
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audio processing statistics"""
        voice_ratio = self.voice_frame_count / max(self.frame_count, 1)
        
        return {
            "frames_processed": self.frame_count,
            "voice_frames": self.voice_frame_count,
            "voice_activity_ratio": voice_ratio,
            "current_direction": self.last_direction,
            "state": self.state.value
        }
    
    def set_gain(self, left_gain: float, right_gain: float):
        """Set gain for left and right channels"""
        self.config.gain_left = max(0, min(10, left_gain))
        self.config.gain_right = max(0, min(10, right_gain))
    
    def set_noise_threshold(self, threshold: float):
        """Set noise detection threshold"""
        self.config.noise_threshold = max(0, min(1, threshold))
        self.vad.energy_threshold = threshold
    
    def focus_direction(self, angle: float) -> np.ndarray:
        """Focus audio processing on specific direction"""
        if self.audio_buffer:
            frame = self.audio_buffer[-1]
            return self.beamforming.beamform(frame.data, angle)
        return np.array([])
    
    def is_listening(self) -> bool:
        """Check if currently listening"""
        return self.state == EarState.LISTENING and self.input_stream is not None
    
    def shutdown(self):
        """Shutdown biomimetic ear system"""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        self.stop_listening()
        
        self.logger.info("Biomimetic ears shutdown complete")
