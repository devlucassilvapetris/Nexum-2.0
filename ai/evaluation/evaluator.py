"""
Main Evaluator
Combine Ragas and Weights & Biases for comprehensive evaluation
"""

import os
from typing import Dict, Any, List, Optional
from datasets import Dataset

from .ragas_evaluator import RagasEvaluator
from .wandb_tracker import WandBTracker


class Evaluator:
    """Main evaluator combining Ragas and Weights & Biases"""
    
    def __init__(self, config_path: str, run_name: Optional[str] = None):
        self.ragas_evaluator = RagasEvaluator(config_path)
        self.wandb_tracker = WandBTracker(config_path, run_name)
        self.config = self.ragas_evaluator.config
    
    def evaluate(
        self,
        dataset: Dataset,
        metrics: Optional[List[str]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run complete evaluation"""
        # Run Ragas evaluation
        results = self.ragas_evaluator.evaluate(dataset, metrics)
        
        # Log to Weights & Biases
        self.wandb_tracker.log_evaluation_results(results)
        
        # Save results if requested
        if save_results:
            output_path = os.path.join(
                self.config["output"]["results_dir"],
                "evaluation_results.json"
            )
            self.ragas_evaluator.save_results(results, output_path)
        
        return results
    
    def evaluate_from_files(
        self,
        test_file: str,
        ground_truth_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate from data files"""
        # Load dataset
        dataset = self.ragas_evaluator.load_dataset(test_file)
        
        # Load ground truth if provided
        if ground_truth_file:
            ground_truth = self.ragas_evaluator.load_ground_truth(ground_truth_file)
            # Merge datasets
            dataset = self._merge_datasets(dataset, ground_truth)
        
        # Run evaluation
        return self.evaluate(dataset)
    
    def _merge_datasets(self, dataset: Dataset, ground_truth: Dataset) -> Dataset:
        """Merge dataset with ground truth"""
        merged_data = []
        for i in range(len(dataset)):
            item = dataset[i]
            if i < len(ground_truth):
                item["ground_truth"] = ground_truth[i]["ground_truth"]
            merged_data.append(item)
        
        return Dataset.from_list(merged_data)
    
    def log_training_progress(self, metrics: Dict[str, float], epoch: int):
        """Log training progress"""
        self.wandb_tracker.log_training_metrics(metrics, epoch)
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model artifact"""
        self.wandb_tracker.log_model(model_path, model_name)
    
    def finish(self):
        """Finish evaluation"""
        self.wandb_tracker.finish()
