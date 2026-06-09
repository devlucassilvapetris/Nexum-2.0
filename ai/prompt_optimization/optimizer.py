"""
DSPy Optimizer
Prompt optimization using DSPy framework
"""

import json
from typing import Dict, Any, List, Optional
import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch


class DSPOptimizer:
    """Optimize prompts using DSPy framework"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_lm()
        self._setup_rm()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_lm(self):
        """Setup language model"""
        lm_config = self.config["lm"]
        
        if lm_config["provider"] == "vllm":
            self.lm = dspy.HFClientVLLM(
                model=lm_config["model"],
                base=lm_config["api_base"],
                api_key=lm_config["api_key"],
                temperature=lm_config["temperature"],
                max_tokens=lm_config["max_tokens"]
            )
        else:
            self.lm = dspy.HFClientVLLM(
                model=lm_config["model"],
                **lm_config
            )
        
        dspy.settings.configure(lm=self.lm)
    
    def _setup_rm(self):
        """Setup retrieval model"""
        rm_config = self.config.get("rm", {})
        
        if rm_config.get("provider") == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.rm = SentenceTransformer(rm_config["model"])
            dspy.settings.configure(rm=self.rm)
    
    def optimize_module(
        self,
        module: dspy.Module,
        trainset: List[dspy.Example],
        metric: Optional callable = None,
        num_trials: Optional[int] = None
    ) -> dspy.Module:
        """Optimize a DSPy module"""
        opt_config = self.config["optimization"]
        
        num_trials = num_trials or opt_config["num_trials"]
        
        # Create teleprompter
        teleprompter = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=opt_config["max_bootstrapped_demos"],
            max_labeled_demos=opt_config["max_labeled_demos"]
        )
        
        # Optimize
        optimized_module = teleprompter.compile(
            module,
            trainset=trainset
        )
        
        return optimized_module
    
    def optimize_with_random_search(
        self,
        module: dspy.Module,
        trainset: List[dspy.Example],
        valset: List[dspy.Example],
        metric: callable,
        num_trials: Optional[int] = None
    ) -> dspy.Module:
        """Optimize with random search"""
        opt_config = self.config["optimization"]
        
        num_trials = num_trials or opt_config["num_trials"]
        
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=opt_config["max_bootstrapped_demos"],
            max_labeled_demos=opt_config["max_labeled_demos"],
            num_candidate_programs=10,
            max_rounds=opt_config.get("max_rounds", 1)
        )
        
        optimized_module = teleprompter.compile(
            module,
            trainset=trainset,
            valset=valset
        )
        
        return optimized_module
    
    def evaluate(
        self,
        module: dspy.Module,
        testset: List[dspy.Example],
        metric: callable
    ) -> Dict[str, Any]:
        """Evaluate optimized module"""
        from dspy.evaluate import Evaluate
        
        evaluator = Evaluate(
            devset=testset,
            metric=metric,
            num_threads=4,
            display_progress=True,
            display_table=5
        )
        
        results = evaluator(module)
        
        return {
            "accuracy": results,
            "num_examples": len(testset)
        }
    
    def save_module(self, module: dspy.Module, path: str):
        """Save optimized module"""
        module.save(path)
    
    def load_module(self, path: str) -> dspy.Module:
        """Load optimized module"""
        return dspy.Module.load(path)
