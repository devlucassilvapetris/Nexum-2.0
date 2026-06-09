"""
Weights & Biases Tracker
Track experiments and metrics with Weights & Biases
"""

import json
import os
from typing import Dict, Any, List, Optional
import wandb


class WandBTracker:
    """Track experiments with Weights & Biases"""
    
    def __init__(self, config_path: str, run_name: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.run = None
        self.run_name = run_name
        
        if self.config["wandb"]["enabled"]:
            self._init_run()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _init_run(self):
        """Initialize Weights & Biases run"""
        wandb_config = self.config["wandb"]
        
        self.run = wandb.init(
            project=wandb_config["project"],
            entity=wandb_config.get("entity"),
            name=self.run_name or wandb_config.get("run_name"),
            config=self.config
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.run:
            wandb.log(metrics, step=step)
    
    def log_evaluation_results(self, results: Dict[str, Any]):
        """Log Ragas evaluation results"""
        if self.run:
            if hasattr(results, 'to_pandas'):
                df = results.to_pandas()
                
                # Log overall metrics
                for column in df.columns:
                    if df[column].dtype in ['float64', 'int64']:
                        wandb.log({
                            f"eval/{column}_mean": df[column].mean(),
                            f"eval/{column}_std": df[column].std(),
                            f"eval/{column}_min": df[column].min(),
                            f"eval/{column}_max": df[column].max()
                        })
                
                # Log as table
                wandb.log({"evaluation_results": wandb.Table(dataframe=df)})
            else:
                wandb.log(results)
    
    def log_training_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log training metrics"""
        if self.run:
            wandb.log(metrics, step=epoch)
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model artifact"""
        if self.run:
            artifact = wandb.Artifact(model_name, type="model")
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)
    
    def log_dataset(self, dataset_path: str, dataset_name: str = "dataset"):
        """Log dataset artifact"""
        if self.run:
            artifact = wandb.Artifact(dataset_name, type="dataset")
            artifact.add_dir(dataset_path)
            wandb.log_artifact(artifact)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        if self.run:
            wandb.config.update(config)
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save checkpoint"""
        if self.run:
            wandb.save(checkpoint_path)
    
    def finish(self):
        """Finish the run"""
        if self.run:
            wandb.finish()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish()
