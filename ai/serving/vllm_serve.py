"""
vLLM Serving
Serve models with vLLM for high-throughput inference
"""

import json
from typing import Dict, Any, List, Optional
from vllm import LLM, SamplingParams


class VLLMServe:
    """Serve models using vLLM"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.llm = None
        self.sampling_params = None
        
        self._setup_llm()
        self._setup_sampling_params()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_llm(self):
        """Setup vLLM engine"""
        vllm_config = self.config["vllm"]
        
        self.llm = LLM(
            model=vllm_config["model"],
            tensor_parallel_size=vllm_config["tensor_parallel_size"],
            gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
            max_model_len=vllm_config["max_model_len"],
            dtype=vllm_config["dtype"],
            trust_remote_code=vllm_config["trust_remote_code"]
        )
    
    def _setup_sampling_params(self):
        """Setup default sampling parameters"""
        gen_config = self.config["generation"]
        
        self.sampling_params = SamplingParams(
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            top_k=gen_config["top_k"],
            max_tokens=gen_config["max_tokens"],
            presence_penalty=gen_config["presence_penalty"],
            frequency_penalty=gen_config["frequency_penalty"]
        )
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """Generate text from prompts"""
        params = sampling_params or self.sampling_params
        
        outputs = self.llm.generate(prompts, params)
        
        return [output.outputs[0].text for output in outputs]
    
    def generate_single(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """Generate text from a single prompt"""
        return self.generate([prompt], sampling_params)[0]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """Generate response from chat messages"""
        # Format messages into a prompt
        prompt = self._format_messages(messages)
        return self.generate_single(prompt, sampling_params)
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt"""
        formatted = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted.append(f"### {role}:\n{content}")
        return "\n\n".join(formatted)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": self.config["vllm"]["model"],
            "max_model_len": self.llm.llm_engine.model_config.max_model_len,
            "tensor_parallel_size": self.config["vllm"]["tensor_parallel_size"],
            "dtype": self.config["vllm"]["dtype"]
        }
