"""
Nexum v2.0 Local Speech Recognition
Offline speech recognition using local models
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import deque

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class RecognitionEngine(Enum):
    SPHINX = "sphinx"
    GOOGLE = "google"
    VOSK = "vosk"
    WHISPER = "whisper"
    KEYWORD = "keyword"


@dataclass
class RecognitionResult:
    text: str
    confidence: float
    language: str
    processing_time: float
    engine: RecognitionEngine
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class KeywordSpottingResult:
    keyword: str
    confidence: float
    start_time: float
    end_time: float
    audio_segment: Optional[np.ndarray] = None


class SpeechRecognitionConfig:
    """Configuration for speech recognition"""
    
    def __init__(self,
                 engine: RecognitionEngine = RecognitionEngine.SPHINX,
                 language: str = "en-US",
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 timeout: float = 5.0,
                 phrase_timeout: float = 3.0):
        self.engine = engine
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.phrase_timeout = phrase_timeout


class KeywordSpotter:
    """Local keyword spotting using MFCC features"""
    
    def __init__(self, keywords: List[str], sample_rate: int = 16000):
        self.keywords = keywords
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Keyword templates (simplified - in real implementation, use trained models)
        self.keyword_templates = self._create_keyword_templates()
        
        # Detection parameters
        self.mfcc_threshold = 0.7
        self.dtw_threshold = 0.8
    
    def _create_keyword_templates(self) -> Dict[str, np.ndarray]:
        """Create simplified keyword templates"""
        # In real implementation, these would be trained from data
        templates = {}
        
        for keyword in self.keywords:
            # Generate pseudo-random template based on keyword
            np.random.seed(hash(keyword) % 2**32)
            template_length = len(keyword) * 10
            template = np.random.randn(template_length, 13)  # 13 MFCC coefficients
            templates[keyword.lower()] = template
        
        return templates
    
    def extract_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        try:
            import librosa
            # Convert to mono if needed
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=512,
                hop_length=256
            )
            
            return mfcc.T  # Transpose to (time, features)
            
        except ImportError:
            self.logger.warning("librosa not available, using dummy MFCC")
            # Generate dummy MFCC
            num_frames = len(audio_data) // 256
            return np.random.randn(num_frames, 13)
    
    def dynamic_time_warping(self, template: np.ndarray, test: np.ndarray) -> float:
        """Compute DTW distance between template and test sequence"""
        n, m = len(template), len(test)
        
        # Initialize distance matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Compute DTW
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(template[i-1] - test[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        # Normalize by path length
        distance = dtw_matrix[n, m]
        path_length = n + m
        normalized_distance = distance / path_length
        
        # Convert to similarity score
        similarity = 1.0 / (1.0 + normalized_distance)
        
        return similarity
    
    def spot_keywords(self, audio_data: np.ndarray) -> List[KeywordSpottingResult]:
        """Spot keywords in audio data"""
        results = []
        
        # Extract MFCC features
        mfcc_features = self.extract_mfcc(audio_data)
        
        # Slide window through audio
        window_size = 50  # frames
        step_size = 10    # frames
        
        for i in range(0, len(mfcc_features) - window_size, step_size):
            window = mfcc_features[i:i + window_size]
            
            # Check each keyword
            for keyword, template in self.keyword_templates.items():
                # Compute similarity using DTW
                similarity = self.dynamic_time_warping(template, window)
                
                if similarity > self.dtw_threshold:
                    start_time = i * 256 / self.sample_rate
                    end_time = (i + window_size) * 256 / self.sample_rate
                    
                    result = KeywordSpottingResult(
                        keyword=keyword,
                        confidence=similarity,
                        start_time=start_time,
                        end_time=end_time,
                        audio_segment=audio_data[int(start_time * self.sample_rate):int(end_time * self.sample_rate)]
                    )
                    results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def add_keyword(self, keyword: str, template: Optional[np.ndarray] = None):
        """Add new keyword to spotter"""
        if template is None:
            # Create template from keyword name
            np.random.seed(hash(keyword) % 2**32)
            template_length = len(keyword) * 10
            template = np.random.randn(template_length, 13)
        
        self.keyword_templates[keyword.lower()] = template
        self.keywords.append(keyword)


class LocalSpeechRecognizer:
    """
    Local speech recognition using various engines
    Supports offline recognition with multiple backends
    """
    
    def __init__(self, config: SpeechRecognitionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Recognition engine
        self.recognizer = None
        self.vosk_model = None
        self.whisper_model = None
        
        # Keyword spotter
        self.keyword_spotter: Optional[KeywordSpotter] = None
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=1000)
        self.is_listening = False
        self.listening_thread = None
        
        # Callbacks
        self.result_callbacks: List[Callable[[RecognitionResult], None]] = []
        self.keyword_callbacks: List[Callable[[KeywordSpottingResult], None]] = []
        
        # Statistics
        self.recognition_count = 0
        self.total_processing_time = 0.0
        
        # Initialize engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize recognition engine"""
        if self.config.engine == RecognitionEngine.SPHINX and SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.logger.info("Initialized Sphinx speech recognition")
            
        elif self.config.engine == RecognitionEngine.GOOGLE and SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.logger.info("Initialized Google speech recognition")
            
        elif self.config.engine == RecognitionEngine.VOSK and VOSK_AVAILABLE:
            try:
                model_path = "vosk-model-small-en-us-0.15"  # Default model path
                self.vosk_model = vosk.Model(model_path)
                self.logger.info("Initialized Vosk speech recognition")
            except Exception as e:
                self.logger.error(f"Failed to initialize Vosk: {e}")
                self.vosk_model = None
                
        elif self.config.engine == RecognitionEngine.WHISPER and WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                self.logger.info("Initialized Whisper speech recognition")
            except Exception as e:
                self.logger.error(f"Failed to initialize Whisper: {e}")
                self.whisper_model = None
        
        else:
            self.logger.warning(f"Engine {self.config.engine} not available, using keyword spotting")
            self.config.engine = RecognitionEngine.KEYWORD
    
    def recognize_audio(self, audio_data: np.ndarray) -> RecognitionResult:
        """
        Recognize speech from audio data
        """
        start_time = time.time()
        
        try:
            if self.config.engine == RecognitionEngine.SPHINX and self.recognizer:
                result = self._recognize_with_sphinx(audio_data)
            elif self.config.engine == RecognitionEngine.GOOGLE and self.recognizer:
                result = self._recognize_with_google(audio_data)
            elif self.config.engine == RecognitionEngine.VOSK and self.vosk_model:
                result = self._recognize_with_vosk(audio_data)
            elif self.config.engine == RecognitionEngine.WHISPER and self.whisper_model:
                result = self._recognize_with_whisper(audio_data)
            else:
                result = self._recognize_with_keywords(audio_data)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Update statistics
            self.recognition_count += 1
            self.total_processing_time += processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Recognition error: {e}")
            return RecognitionResult(
                text="",
                confidence=0.0,
                language=self.config.language,
                processing_time=time.time() - start_time,
                engine=self.config.engine,
                raw_data={"error": str(e)}
            )
    
    def _recognize_with_sphinx(self, audio_data: np.ndarray) -> RecognitionResult:
        """Recognize using Sphinx (offline)"""
        try:
            # Convert audio data to AudioData
            audio_sample = sr.AudioData(
                audio_data.tobytes(),
                sample_rate=self.config.sample_rate,
                sample_width=audio_data.dtype.itemsize
            )
            
            # Recognize
            text = self.recognizer.recognize_sphinx(audio_sample, language=self.config.language)
            
            return RecognitionResult(
                text=text,
                confidence=0.8,  # Sphinx doesn't provide confidence
                language=self.config.language,
                processing_time=0.0,
                engine=RecognitionEngine.SPHINX
            )
            
        except Exception as e:
            self.logger.error(f"Sphinx recognition error: {e}")
            raise
    
    def _recognize_with_google(self, audio_data: np.ndarray) -> RecognitionResult:
        """Recognize using Google (requires internet)"""
        try:
            audio_sample = sr.AudioData(
                audio_data.tobytes(),
                sample_rate=self.config.sample_rate,
                sample_width=audio_data.dtype.itemsize
            )
            
            text = self.recognizer.recognize_google(audio_sample, language=self.config.language)
            
            return RecognitionResult(
                text=text,
                confidence=0.9,
                language=self.config.language,
                processing_time=0.0,
                engine=RecognitionEngine.GOOGLE
            )
            
        except Exception as e:
            self.logger.error(f"Google recognition error: {e}")
            raise
    
    def _recognize_with_vosk(self, audio_data: np.ndarray) -> RecognitionResult:
        """Recognize using Vosk (offline)"""
        try:
            # Convert to 16-bit PCM
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create rec
            rec = vosk.KaldiRec(self.vosk_model, self.config.sample_rate)
            
            # Accept data
            rec.AcceptWaveform(audio_data.tobytes())
            
            # Get result
            result = rec.FinalResult()
            import json
            result_dict = json.loads(result)
            
            text = result_dict.get("text", "")
            
            return RecognitionResult(
                text=text,
                confidence=0.85,
                language=self.config.language,
                processing_time=0.0,
                engine=RecognitionEngine.VOSK,
                raw_data=result_dict
            )
            
        except Exception as e:
            self.logger.error(f"Vosk recognition error: {e}")
            raise
    
    def _recognize_with_whisper(self, audio_data: np.ndarray) -> RecognitionResult:
        """Recognize using Whisper (offline)"""
        try:
            # Resample if necessary
            if self.config.sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=self.config.sample_rate, target_sr=16000)
            
            # Transcribe
            result = self.whisper_model.transcribe(audio_data, language=self.config.language[:2])
            text = result["text"].strip()
            
            return RecognitionResult(
                text=text,
                confidence=0.95,
                language=self.config.language,
                processing_time=0.0,
                engine=RecognitionEngine.WHISPER,
                raw_data=result
            )
            
        except Exception as e:
            self.logger.error(f"Whisper recognition error: {e}")
            raise
    
    def _recognize_with_keywords(self, audio_data: np.ndarray) -> RecognitionResult:
        """Recognize using keyword spotting"""
        if self.keyword_spotter is None:
            # Initialize with default keywords
            default_keywords = ["hello", "stop", "start", "yes", "no"]
            self.keyword_spotter = KeywordSpotter(default_keywords, self.config.sample_rate)
        
        # Spot keywords
        keyword_results = self.keyword_spotter.spot_keywords(audio_data)
        
        if keyword_results:
            # Use best match
            best_result = keyword_results[0]
            text = best_result.keyword
            confidence = best_result.confidence
        else:
            text = ""
            confidence = 0.0
        
        return RecognitionResult(
            text=text,
            confidence=confidence,
            language=self.config.language,
            processing_time=0.0,
            engine=RecognitionEngine.KEYWORD,
            raw_data={"keyword_results": len(keyword_results)}
        )
    
    def enable_keyword_spotting(self, keywords: List[str]):
        """Enable keyword spotting with specific keywords"""
        self.keyword_spotter = KeywordSpotter(keywords, self.config.sample_rate)
        self.logger.info(f"Keyword spotting enabled for: {keywords}")
    
    def spot_keywords(self, audio_data: np.ndarray) -> List[KeywordSpottingResult]:
        """Spot keywords in audio data"""
        if self.keyword_spotter is None:
            self.logger.warning("Keyword spotter not initialized")
            return []
        
        return self.keyword_spotter.spot_keywords(audio_data)
    
    def start_continuous_listening(self, audio_source):
        """Start continuous listening from audio source"""
        self.is_listening = True
        self.listening_thread = threading.Thread(
            target=self._listening_loop,
            args=(audio_source,),
            daemon=True
        )
        self.listening_thread.start()
        self.logger.info("Started continuous listening")
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        self.logger.info("Stopped continuous listening")
    
    def _listening_loop(self, audio_source):
        """Background listening loop"""
        while self.is_listening:
            try:
                # Read audio chunk
                audio_data = audio_source.read(self.config.chunk_size)
                
                if audio_data is not None:
                    # Add to buffer
                    self.audio_buffer.append(audio_data)
                    
                    # Process if buffer has enough data
                    if len(self.audio_buffer) >= self.config.sample_rate // self.config.chunk_size * 2:
                        # Combine buffer
                        full_audio = np.concatenate(list(self.audio_buffer))
                        
                        # Recognize
                        result = self.recognize_audio(full_audio)
                        
                        # Trigger callbacks
                        for callback in self.result_callbacks:
                            try:
                                callback(result)
                            except Exception as e:
                                self.logger.error(f"Result callback error: {e}")
                        
                        # Clear buffer
                        self.audio_buffer.clear()
                
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Listening loop error: {e}")
                time.sleep(0.1)
    
    def register_result_callback(self, callback: Callable[[RecognitionResult], None]):
        """Register callback for recognition results"""
        self.result_callbacks.append(callback)
    
    def register_keyword_callback(self, callback: Callable[[KeywordSpottingResult], None]):
        """Register callback for keyword spotting results"""
        self.keyword_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recognition statistics"""
        avg_processing_time = self.total_processing_time / max(self.recognition_count, 1)
        
        return {
            "engine": self.config.engine.value,
            "recognition_count": self.recognition_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "is_listening": self.is_listening,
            "buffer_size": len(self.audio_buffer)
        }
    
    def set_engine(self, engine: RecognitionEngine):
        """Change recognition engine"""
        self.config.engine = engine
        self._initialize_engine()
    
    def reset_statistics(self):
        """Reset recognition statistics"""
        self.recognition_count = 0
        self.total_processing_time = 0.0


class VoiceCommandProcessor:
    """Process voice commands and map to actions"""
    
    def __init__(self, speech_recognizer: LocalSpeechRecognizer):
        self.speech_recognizer = speech_recognizer
        self.logger = logging.getLogger(__name__)
        
        # Command mappings
        self.command_mappings: Dict[str, Callable] = {}
        
        # Register callback for results
        self.speech_recognizer.register_result_callback(self._process_result)
    
    def add_command(self, phrase: str, action: Callable):
        """Add voice command mapping"""
        self.command_mappings[phrase.lower()] = action
        self.logger.info(f"Added command: '{phrase}'")
    
    def remove_command(self, phrase: str):
        """Remove voice command mapping"""
        if phrase.lower() in self.command_mappings:
            del self.command_mappings[phrase.lower()]
            self.logger.info(f"Removed command: '{phrase}'")
    
    def _process_result(self, result: RecognitionResult):
        """Process recognition result and execute commands"""
        text = result.text.lower().strip()
        
        if not text:
            return
        
        # Check for exact matches
        if text in self.command_mappings:
            try:
                self.command_mappings[text]()
                self.logger.info(f"Executed command: '{text}'")
            except Exception as e:
                self.logger.error(f"Command execution error: {e}")
            return
        
        # Check for partial matches
        for phrase, action in self.command_mappings.items():
            if phrase in text or text in phrase:
                try:
                    action()
                    self.logger.info(f"Executed partial match command: '{phrase}'")
                except Exception as e:
                    self.logger.error(f"Command execution error: {e}")
                return
        
        self.logger.debug(f"No command matched for: '{text}'")
    
    def get_available_commands(self) -> List[str]:
        """Get list of available commands"""
        return list(self.command_mappings.keys())
