"""
Nexum v2.0 Local Models Manager
Manages local AI models without external API dependencies
"""

import os
import json
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
import h5py


class ModelType(Enum):
    TEXT_GENERATION = "text_generation"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SPEECH_RECOGNITION = "speech_recognition"
    TRANSLATION = "translation"
    EMBEDDING = "embedding"


class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class ModelConfig:
    id: str
    name: str
    type: ModelType
    model_path: str
    config_path: str
    tokenizer_path: Optional[str] = None
    max_sequence_length: int = 512
    batch_size: int = 1
    device: str = "auto"
    precision: str = "fp16"
    cache_size: int = 1024  # MB
    memory_limit: int = 4096  # MB


@dataclass
class ModelInfo:
    config: ModelConfig
    status: ModelStatus
    load_time: float
    memory_usage: float
    inference_count: int
    average_inference_time: float
    last_used: float


class ModelCache:
    """Model memory cache management"""
    
    def __init__(self, max_size_mb: int = 1024):
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.cache: Dict[str, torch.nn.Module] = {}
        self.sizes: Dict[str, int] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def get_model_size(self, model: torch.nn.Module) -> int:
        """Estimate model size in bytes"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def store(self, key: str, model: torch.nn.Module) -> bool:
        """Store model in cache"""
        with self.lock:
            size = self.get_model_size(model)
            
            # Check if we need to evict
            while (len(self.cache) > 0 and 
                   self._total_size() + size > self.max_size):
                if not self._evict_least_recent():
                    return False  # Can't evict enough space
            
            self.cache[key] = model
            self.sizes[key] = size
            self.access_times[key] = time.time()
            
            return True
    
    def get(self, key: str) -> Optional[torch.nn.Module]:
        """Get model from cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def remove(self, key: str) -> bool:
        """Remove model from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.sizes[key]
                del self.access_times[key]
                return True
            return False
    
    def _total_size(self) -> int:
        """Get total cache size"""
        return sum(self.sizes.values())
    
    def _evict_least_recent(self) -> bool:
        """Evict least recently used model"""
        if not self.cache:
            return False
        
        # Find least recently used
        lru_key = min(self.access_times, key=self.access_times.get)
        
        # Remove from cache
        del self.cache[lru_key]
        del self.sizes[lru_key]
        del self.access_times[lru_key]
        
        return True
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.sizes.clear()
            self.access_times.clear()


class LocalModel:
    """Wrapper for local AI model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_used = 0.0
    
    def _get_device(self) -> torch.device:
        """Determine best device for model"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def load(self) -> bool:
        """Load model and tokenizer"""
        try:
            self.logger.info(f"Loading model {self.config.name}...")
            
            # Load configuration
            if os.path.exists(self.config.config_path):
                model_config = AutoConfig.from_pretrained(self.config.config_path)
            else:
                model_config = AutoConfig.from_pretrained(self.config.model_path)
            
            # Load model
            if self.config.type == ModelType.TEXT_GENERATION:
                # Use AutoModelForCausalLM for text generation models (properly exposes .generate())
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    config=model_config,
                    torch_dtype=torch.float16 if self.config.precision == "fp16" else torch.float32
                )
                
                # Load tokenizer
                if self.config.tokenizer_path:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            elif self.config.type == ModelType.TRANSLATION:
                # Use AutoModelForSeq2SeqLM for translation models
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_path,
                    config=model_config,
                    torch_dtype=torch.float16 if self.config.precision == "fp16" else torch.float32
                )
                
                # Load tokenizer
                if self.config.tokenizer_path:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            elif self.config.type == ModelType.EMBEDDING:
                self.model = AutoModel.from_pretrained(
                    self.config.model_path,
                    config=model_config,
                    torch_dtype=torch.float16 if self.config.precision == "fp16" else torch.float32
                )
                
                if self.config.tokenizer_path:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            else:
                # For other model types, load with AutoModel
                self.model = AutoModel.from_pretrained(
                    self.config.model_path,
                    config=model_config,
                    torch_dtype=torch.float16 if self.config.precision == "fp16" else torch.float32
                )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Set padding token if not present
            if self.tokenizer and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Model {self.config.name} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.config.name}: {e}")
            return False
    
    def unload(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def infer(self, inputs: Union[str, List[str], torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Run inference with the model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Prepare inputs based on model type
            if self.config.type in [ModelType.TEXT_GENERATION, ModelType.EMBEDDING, ModelType.TRANSLATION]:
                result = self._text_inference(inputs, **kwargs)
            else:
                result = self._general_inference(inputs, **kwargs)
            
            # Update statistics
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            self.last_used = time.time()
            
            result["inference_time"] = inference_time
            result["model_id"] = self.config.id
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inference error for model {self.config.name}: {e}")
            raise
    
    def _text_inference(self, inputs: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Text-based model inference"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # Tokenize inputs
        if isinstance(inputs, str):
            inputs = [inputs]
        
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        # Process outputs based on model type
        if self.config.type == ModelType.TEXT_GENERATION:
            # For generation models, we might need to use generate method
            if hasattr(self.model, 'generate'):
                max_length = kwargs.get('max_length', 50)
                generated = self.model.generate(
                    **encoded,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=kwargs.get('temperature', 1.0),
                    do_sample=kwargs.get('do_sample', True)
                )
                
                # Decode generated text
                generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                
                return {
                    "generated_text": generated_text,
                    "logits": outputs.logits.cpu().numpy() if hasattr(outputs, 'logits') else None
                }
        
        elif self.config.type == ModelType.EMBEDDING:
            # Return embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            return {
                "embeddings": embeddings,
                "last_hidden_state": outputs.last_hidden_state.cpu().numpy()
            }
        
        else:
            # General text processing
            return {
                "last_hidden_state": outputs.last_hidden_state.cpu().numpy(),
                "pooler_output": outputs.pooler_output.cpu().numpy() if hasattr(outputs, 'pooler_output') else None,
                "logits": outputs.logits.cpu().numpy() if hasattr(outputs, 'logits') else None
            }
    
    def _general_inference(self, inputs: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """General model inference for non-text models"""
        # Move inputs to device
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        
        inputs = inputs.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Return outputs
        result = {}
        if hasattr(outputs, 'last_hidden_state'):
            result["last_hidden_state"] = outputs.last_hidden_state.cpu().numpy()
        if hasattr(outputs, 'logits'):
            result["logits"] = outputs.logits.cpu().numpy()
        if hasattr(outputs, 'pooler_output'):
            result["pooler_output"] = outputs.pooler_output.cpu().numpy()
        
        return result
    
    def get_memory_usage(self) -> float:
        """Get model memory usage in MB"""
        if self.model is None:
            return 0.0
        
        # Estimate memory usage
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        
        return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics"""
        avg_inference_time = (self.total_inference_time / max(self.inference_count, 1))
        
        return {
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": avg_inference_time,
            "memory_usage_mb": self.get_memory_usage(),
            "last_used": self.last_used,
            "device": str(self.device)
        }


class LocalModelManager:
    """
    Central manager for local AI models
    Handles loading, caching, and inference coordination
    """
    
    def __init__(self, cache_size_mb: int = 1024, memory_limit_mb: int = 4096):
        self.models: Dict[str, LocalModel] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.cache = ModelCache(cache_size_mb)
        self.memory_limit = memory_limit_mb
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.model_registry: Dict[str, ModelConfig] = {}
        self._load_model_registry()
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.running = False
    
    def _load_model_registry(self):
        """Load model registry from configuration"""
        # Default model configurations
        default_models = {
            "text_generator": ModelConfig(
                id="text_generator",
                name="Local Text Generator",
                type=ModelType.TEXT_GENERATION,
                model_path="models/text_generator",
                config_path="models/text_generator/config.json",
                tokenizer_path="models/text_generator/tokenizer",
                max_sequence_length=512,
                device="auto"
            ),
            "embedding_model": ModelConfig(
                id="embedding_model",
                name="Local Embedding Model",
                type=ModelType.EMBEDDING,
                model_path="models/embedding_model",
                config_path="models/embedding_model/config.json",
                tokenizer_path="models/embedding_model/tokenizer",
                max_sequence_length=512,
                device="auto"
            )
        }
        
        self.model_registry.update(default_models)
        
        # Try to load from file if exists
        registry_path = "config/model_registry.json"
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    file_registry = json.load(f)
                
                for model_data in file_registry:
                    config = ModelConfig(**model_data)
                    self.model_registry[config.id] = config
                    
            except Exception as e:
                self.logger.warning(f"Failed to load model registry: {e}")
    
    def initialize(self) -> bool:
        """Initialize model manager"""
        try:
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            
            self.logger.info("Local model manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model manager: {e}")
            return False
    
    def register_model(self, config: ModelConfig) -> bool:
        """Register a new model configuration"""
        try:
            self.model_registry[config.id] = config
            
            # Save to registry file
            self._save_model_registry()
            
            self.logger.info(f"Registered model: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            return False
    
    def _save_model_registry(self):
        """Save model registry to file"""
        try:
            os.makedirs("config", exist_ok=True)
            
            registry_data = []
            for config in self.model_registry.values():
                registry_data.append({
                    "id": config.id,
                    "name": config.name,
                    "type": config.type.value,
                    "model_path": config.model_path,
                    "config_path": config.config_path,
                    "tokenizer_path": config.tokenizer_path,
                    "max_sequence_length": config.max_sequence_length,
                    "batch_size": config.batch_size,
                    "device": config.device,
                    "precision": config.precision,
                    "cache_size": config.cache_size,
                    "memory_limit": config.memory_limit
                })
            
            with open("config/model_registry.json", 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
    
    def load_model(self, model_id: str) -> bool:
        """Load a model into memory"""
        if model_id not in self.model_registry:
            self.logger.error(f"Model {model_id} not found in registry")
            return False
        
        if model_id in self.models:
            self.logger.info(f"Model {model_id} already loaded")
            return True
        
        config = self.model_registry[model_id]
        
        # Check memory limits
        current_memory = sum(model.get_memory_usage() for model in self.models.values())
        if current_memory > self.memory_limit * 0.8:
            self.logger.warning("Approaching memory limit, consider unloading unused models")
        
        try:
            # Create and load model
            model = LocalModel(config)
            
            if model.load():
                self.models[model_id] = model
                
                # Create model info
                self.model_info[model_id] = ModelInfo(
                    config=config,
                    status=ModelStatus.READY,
                    load_time=time.time(),
                    memory_usage=model.get_memory_usage(),
                    inference_count=0,
                    average_inference_time=0.0,
                    last_used=time.time()
                )
                
                self.logger.info(f"Model {model_id} loaded successfully")
                return True
            else:
                self.logger.error(f"Failed to load model {model_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id not in self.models:
            return True
        
        try:
            model = self.models[model_id]
            model.unload()
            
            del self.models[model_id]
            if model_id in self.model_info:
                self.model_info[model_id].status = ModelStatus.UNLOADED
            
            self.logger.info(f"Model {model_id} unloaded")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unloading model {model_id}: {e}")
            return False
    
    def infer(self, model_id: str, inputs: Union[str, List[str], torch.Tensor], **kwargs) -> Optional[Dict[str, Any]]:
        """Run inference with a specific model"""
        if model_id not in self.models:
            # Try to load the model
            if not self.load_model(model_id):
                return None
        
        model = self.models[model_id]
        
        try:
            # Update status
            if model_id in self.model_info:
                self.model_info[model_id].status = ModelStatus.PROCESSING
            
            # Run inference
            result = model.infer(inputs, **kwargs)
            
            # Update statistics
            if model_id in self.model_info:
                info = self.model_info[model_id]
                info.inference_count += 1
                info.average_inference_time = (
                    (info.average_inference_time * (info.inference_count - 1) + 
                     result["inference_time"]) / info.inference_count
                )
                info.last_used = time.time()
                info.status = ModelStatus.READY
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inference error for model {model_id}: {e}")
            
            if model_id in self.model_info:
                self.model_info[model_id].status = ModelStatus.ERROR
            
            return None
    
    def get_model_status(self, model_id: str) -> Optional[ModelInfo]:
        """Get status of a specific model"""
        return self.model_info.get(model_id)
    
    def get_all_status(self) -> Dict[str, ModelInfo]:
        """Get status of all models"""
        return self.model_info.copy()
    
    def get_available_models(self) -> List[str]:
        """Get list of available model IDs"""
        return list(self.model_registry.keys())
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model IDs"""
        return list(self.models.keys())
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                # Check for unused models
                current_time = time.time()
                unused_threshold = 300  # 5 minutes
                
                for model_id, info in list(self.model_info.items()):
                    if (info.status == ModelStatus.READY and 
                        current_time - info.last_used > unused_threshold):
                        
                        self.logger.info(f"Unloading unused model: {model_id}")
                        self.unload_model(model_id)
                
                # Check memory usage
                total_memory = sum(model.get_memory_usage() for model in self.models.values())
                if total_memory > self.memory_limit:
                    # Unload least recently used models
                    sorted_models = sorted(
                        self.model_info.items(),
                        key=lambda x: x[1].last_used
                    )
                    
                    for model_id, info in sorted_models:
                        if info.status == ModelStatus.READY:
                            self.unload_model(model_id)
                            total_memory = sum(model.get_memory_usage() for model in self.models.values())
                            if total_memory <= self.memory_limit * 0.8:
                                break
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(60)
    
    def shutdown(self):
        """Shutdown model manager"""
        self.running = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Unload all models
        for model_id in list(self.models.keys()):
            self.unload_model(model_id)
        
        # Clear cache
        self.cache.clear()
        
        self.logger.info("Local model manager shutdown complete")
