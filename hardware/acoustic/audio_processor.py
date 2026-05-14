"""
Nexum v2.0 Audio Processor
Advanced audio processing and analysis for biomimetic ears
"""

import numpy as np
import librosa
import scipy.signal
from scipy.fft import fft, fftfreq
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

import sounddevice as sd
from .biomimetic_ears import AudioFrame, AudioFeatures, AudioDirection


class ProcessingMode(Enum):
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"


class AudioEffect(Enum):
    NOISE_REDUCTION = "noise_reduction"
    ECHO_CANCELLATION = "echo_cancellation"
    EQUALIZATION = "equalization"
    COMPRESSION = "compression"
    PITCH_SHIFT = "pitch_shift"


@dataclass
class ProcessingConfig:
    window_size: int = 1024
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    n_mfcc: int = 13
    f_min: float = 20.0
    f_max: float = 8000.0
    noise_reduction_strength: float = 0.5
    echo_cancellation_delay: float = 0.1
    compression_ratio: float = 4.0
    compression_threshold: float = -20.0


class NoiseReducer:
    """Noise reduction using spectral subtraction"""
    
    def __init__(self, strength: float = 0.5):
        self.strength = strength
        self.noise_spectrum = None
        self.noise_frames = 0
    
    def estimate_noise(self, audio_frames: List[np.ndarray]):
        """Estimate noise spectrum from silent frames"""
        if not audio_frames:
            return
        
        # Average spectrum of noise frames
        spectra = []
        for frame in audio_frames:
            spectrum = np.abs(fft(frame))
            spectra.append(spectrum)
        
        self.noise_spectrum = np.mean(spectra, axis=0)
        self.noise_frames = len(audio_frames)
    
    def reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        if self.noise_spectrum is None:
            return audio_data
        
        # Compute spectrum
        spectrum = fft(audio_data)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Spectral subtraction
        noise_factor = self.strength * self.noise_spectrum[:len(magnitude)]
        magnitude = np.maximum(magnitude - noise_factor, 0.1 * magnitude)
        
        # Reconstruct signal
        clean_spectrum = magnitude * np.exp(1j * phase)
        clean_audio = np.real(np.fft.ifft(clean_spectrum))
        
        return clean_audio.astype(audio_data.dtype)


class EchoCanceller:
    """Echo cancellation using adaptive filtering"""
    
    def __init__(self, delay: float = 0.1, sample_rate: int = 44100):
        self.delay_samples = int(delay * sample_rate)
        self.filter_length = 512
        self.filter_taps = np.zeros(self.filter_length)
        self.step_size = 0.01
        self.buffer = np.zeros(self.filter_length)
    
    def cancel_echo(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray:
        """Cancel echo from near-end signal"""
        # Simple adaptive filter implementation
        if len(far_end) < self.delay_samples:
            return near_end
        
        # Delay far-end signal
        delayed_far = np.zeros_like(near_end)
        if len(far_end) >= self.delay_samples:
            delayed_far = far_end[-self.delay_samples:][:len(near_end)]
        
        # Apply adaptive filter
        echo_estimate = np.zeros_like(near_end)
        for i in range(len(near_end)):
            # Update buffer
            self.buffer = np.roll(self.buffer, 1)
            self.buffer[0] = delayed_far[i] if i < len(delayed_far) else 0
            
            # Filter output
            echo_estimate[i] = np.dot(self.filter_taps, self.buffer)
            
            # Update filter coefficients (LMS algorithm)
            if i < len(near_end):
                error = near_end[i] - echo_estimate[i]
                self.filter_taps += self.step_size * error * self.buffer
        
        # Remove echo
        clean_signal = near_end - echo_estimate
        
        return clean_signal


class Equalizer:
    """Multi-band equalizer"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000),
            'ultra_high': (8000, 20000)
        }
        self.gains = {band: 1.0 for band in self.bands}
        self.filters = {}
        self._design_filters()
    
    def _design_filters(self):
        """Design band-pass filters for each band"""
        for band, (f_low, f_high) in self.bands.items():
            # Design band-pass filter
            nyquist = self.sample_rate / 2
            low = f_low / nyquist
            high = f_high / nyquist
            
            if low > 0 and high < 1:
                b, a = scipy.signal.butter(4, [low, high], btype='band')
                self.filters[band] = (b, a)
    
    def set_gain(self, band: str, gain_db: float):
        """Set gain for specific band in dB"""
        if band in self.bands:
            self.gains[band] = 10 ** (gain_db / 20)
    
    def apply(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply equalization"""
        result = np.zeros_like(audio_data)
        
        for band, (b, a) in self.filters.items():
            try:
                # Apply band-pass filter
                band_signal = scipy.signal.filtfilt(b, a, audio_data)
                
                # Apply gain
                band_signal *= self.gains[band]
                
                # Add to result
                result += band_signal
                
            except Exception as e:
                logging.warning(f"Equalizer band {band} error: {e}")
        
        return result


class Compressor:
    """Dynamic range compressor"""
    
    def __init__(self, ratio: float = 4.0, threshold: float = -20.0, 
                 attack: float = 0.003, release: float = 0.1):
        self.ratio = ratio
        self.threshold = threshold
        self.attack = attack
        self.release = release
        self.envelope = 0.0
        self.sample_rate = 44100
    
    def apply(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply compression"""
        # Convert threshold to linear scale
        threshold_linear = 10 ** (threshold / 20)
        
        # Compute envelope
        level = np.abs(audio_data)
        target = np.maximum(level, threshold_linear)
        
        # Smooth envelope
        attack_coeff = np.exp(-1.0 / (self.attack * self.sample_rate))
        release_coeff = np.exp(-1.0 / (self.release * self.sample_rate))
        
        for i in range(1, len(level)):
            if target[i] > self.envelope:
                self.envelope = attack_coeff * self.envelope + (1 - attack_coeff) * target[i]
            else:
                self.envelope = release_coeff * self.envelope + (1 - release_coeff) * target[i]
        
        # Compute gain reduction
        gain_reduction = np.ones_like(audio_data)
        over_threshold = self.envelope > threshold_linear
        
        if np.any(over_threshold):
            # Apply compression formula
            over_amount = self.envelope[over_threshold] / threshold_linear
            compressed = threshold_linear * (over_amount ** (1.0 / self.ratio))
            reduction = compressed / self.envelope[over_threshold]
            gain_reduction[over_threshold] = reduction
        
        # Apply gain reduction
        compressed_audio = audio_data * gain_reduction
        
        return compressed_audio


class AudioProcessor:
    """
    Advanced audio processing for Nexum v2.0
    Handles real-time audio effects and analysis
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing components
        self.noise_reducer = NoiseReducer(config.noise_reduction_strength)
        self.echo_canceller = EchoCanceller(config.echo_cancellation_delay)
        self.equalizer = Equalizer()
        self.compressor = Compressor(config.compression_ratio, config.compression_threshold)
        
        # Processing state
        self.processing_mode = ProcessingMode.REALTIME
        self.running = False
        self.processing_thread = None
        
        # Buffers
        self.input_buffer = []
        self.output_buffer = []
        self.noise_buffer = []
        
        # Statistics
        self.frames_processed = 0
        self.processing_time = 0.0
    
    def initialize(self) -> bool:
        """Initialize audio processor"""
        try:
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.logger.info("Audio processor initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio processor: {e}")
            return False
    
    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        """Process single audio frame"""
        start_time = time.time()
        
        try:
            audio_data = frame.data.copy()
            
            # Apply processing chain
            audio_data = self._apply_processing_chain(audio_data)
            
            # Create processed frame
            processed_frame = AudioFrame(
                timestamp=frame.timestamp,
                data=audio_data,
                sample_rate=frame.sample_rate,
                channels=frame.channels,
                direction=frame.direction
            )
            
            # Update statistics
            self.frames_processed += 1
            self.processing_time += time.time() - start_time
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame
    
    def _apply_processing_chain(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply full processing chain to audio data"""
        processed = audio_data.copy()
        
        # Noise reduction
        if self.noise_reducer.noise_spectrum is not None:
            processed = self.noise_reducer.reduce_noise(processed)
        
        # Echo cancellation (requires reference signal)
        # This would be implemented with actual far-end signal
        
        # Equalization
        processed = self.equalizer.apply(processed)
        
        # Compression
        processed = self.compressor.apply(processed)
        
        # Normalize
        max_val = np.max(np.abs(processed))
        if max_val > 0:
            processed = processed / max_val * 0.95
        
        return processed
    
    def _processing_loop(self):
        """Background processing loop"""
        while self.running:
            try:
                if self.input_buffer:
                    frame = self.input_buffer.pop(0)
                    processed_frame = self.process_frame(frame)
                    self.output_buffer.append(processed_frame)
                
                time.sleep(0.001)  # 1ms processing interval
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                time.sleep(0.01)
    
    def add_frame(self, frame: AudioFrame):
        """Add frame to input buffer"""
        self.input_buffer.append(frame)
        
        # Keep buffer size manageable
        if len(self.input_buffer) > 50:
            self.input_buffer.pop(0)
    
    def get_processed_frame(self) -> Optional[AudioFrame]:
        """Get next processed frame"""
        if self.output_buffer:
            return self.output_buffer.pop(0)
        return None
    
    def estimate_noise_profile(self, duration: float = 2.0):
        """Estimate noise profile from silence"""
        # This would capture audio during silence and estimate noise
        # For now, use existing noise buffer
        if self.noise_buffer:
            self.noise_reducer.estimate_noise(self.noise_buffer)
            self.logger.info("Noise profile estimated")
    
    def add_noise_sample(self, audio_data: np.ndarray):
        """Add noise sample for profile estimation"""
        self.noise_buffer.append(audio_data)
        
        # Keep limited number of samples
        if len(self.noise_buffer) > 20:
            self.noise_buffer.pop(0)
    
    def set_effect_parameter(self, effect: AudioEffect, parameter: str, value: float):
        """Set effect parameter"""
        try:
            if effect == AudioEffect.NOISE_REDUCTION:
                if parameter == "strength":
                    self.noise_reducer.strength = value
            elif effect == AudioEffect.ECHO_CANCELLATION:
                if parameter == "delay":
                    self.echo_canceller.delay_samples = int(value * self.echo_canceller.sample_rate)
            elif effect == AudioEffect.EQUALIZATION:
                if parameter.startswith("gain_"):
                    band = parameter.replace("gain_", "")
                    self.equalizer.set_gain(band, value)
            elif effect == AudioEffect.COMPRESSION:
                if parameter == "ratio":
                    self.compressor.ratio = value
                elif parameter == "threshold":
                    self.compressor.threshold = value
        except Exception as e:
            self.logger.error(f"Error setting effect parameter: {e}")
    
    def analyze_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze audio data and return features"""
        try:
            # Convert to mono if needed
            if audio_data.ndim > 1:
                mono_data = np.mean(audio_data, axis=1)
            else:
                mono_data = audio_data
            
            # Basic features
            rms = np.sqrt(np.mean(mono_data ** 2))
            peak = np.max(np.abs(mono_data))
            zcr = np.mean(librosa.feature.zero_crossing_rate(mono_data)[0])
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=mono_data, sr=self.config.f_min * 2)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                y=mono_data, sr=self.config.f_min * 2)[0])
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                y=mono_data, sr=self.config.f_min * 2)[0])
            
            # MFCC
            mfcc = librosa.feature.mfcc(
                y=mono_data, 
                sr=int(self.config.f_min * 2), 
                n_mfcc=self.config.n_mfcc
            )
            mfcc_mean = np.mean(mfcc, axis=1)
            
            return {
                "rms": rms,
                "peak": peak,
                "zero_crossing_rate": zcr,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_rolloff": spectral_rolloff,
                "mfcc": mfcc_mean.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Audio analysis error: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_processing_time = self.processing_time / max(self.frames_processed, 1)
        
        return {
            "frames_processed": self.frames_processed,
            "average_processing_time": avg_processing_time,
            "processing_rate": 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            "input_buffer_size": len(self.input_buffer),
            "output_buffer_size": len(self.output_buffer),
            "noise_samples": len(self.noise_buffer)
        }
    
    def set_processing_mode(self, mode: ProcessingMode):
        """Set processing mode"""
        self.processing_mode = mode
    
    def clear_buffers(self):
        """Clear all buffers"""
        self.input_buffer.clear()
        self.output_buffer.clear()
        self.noise_buffer.clear()
    
    def reset_effects(self):
        """Reset all effects to default"""
        self.noise_reducer = NoiseReducer(self.config.noise_reduction_strength)
        self.echo_canceller = EchoCanceller(self.config.echo_cancellation_delay)
        self.equalizer = Equalizer()
        self.compressor = Compressor(self.config.compression_ratio, self.config.compression_threshold)
    
    def shutdown(self):
        """Shutdown audio processor"""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        self.clear_buffers()
        
        self.logger.info("Audio processor shutdown complete")
