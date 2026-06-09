"""
Trainer for Fine-tuning
Main training loop with Weights & Biases integration
"""

import os
import json
import wandb
from typing import Dict, Any
from transformers import Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

from .finetuner import UnslothFinetuner
from .data_processor import DataProcessor


class Trainer:
    """Main trainer class for fine-tuning"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.finetuner = None
        self.data_processor = None
        self.trainer = None
        
        # Initialize Weights & Biases
        if self.config["wandb"]["enabled"]:
            wandb.init(
                project=self.config["wandb"]["project"],
                entity=self.config["wandb"].get("entity"),
                name=self.config["wandb"].get("run_name"),
                config=self.config
            )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup(self):
        """Setup model, tokenizer, and data processor"""
        # Initialize finetuner
        self.finetuner = UnslothFinetuner(self.config)
        
        # Load model
        self.finetuner.load_model()
        
        # Apply LoRA
        self.finetuner.apply_lora()
        
        # Setup data processor
        self.data_processor = DataProcessor(
            tokenizer=self.finetuner.tokenizer,
            max_length=self.config["model"]["max_seq_length"]
        )
    
    def prepare_data(self):
        """Prepare training and validation datasets"""
        data_config = self.config["data"]
        
        # Load training data
        train_data = self.data_processor.load_from_jsonl(
            data_config["train_file"],
            max_samples=data_config.get("max_samples")
        )
        train_dataset = self.data_processor.create_dataset(train_data)
        train_dataset = self.data_processor.prepare_dataset(train_dataset)
        
        # Load validation data
        if data_config.get("validation_file"):
            val_data = self.data_processor.load_from_jsonl(
                data_config["validation_file"],
                max_samples=data_config.get("max_samples")
            )
            val_dataset = self.data_processor.create_dataset(val_data)
            val_dataset = self.data_processor.prepare_dataset(val_dataset)
        else:
            val_dataset = None
        
        return train_dataset, val_dataset
    
    def train(self):
        """Run fine-tuning"""
        # Setup
        self.setup()
        
        # Prepare data
        train_dataset, val_dataset = self.prepare_data()
        
        # Get training arguments
        training_args = self.finetuner.get_training_arguments()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.finetuner.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.finetuner.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        self.trainer.train()
        
        # Save model
        output_dir = os.path.join(
            self.config["training"]["output_dir"],
            "final"
        )
        self.finetuner.save_model(output_dir)
        
        # Finish wandb run
        if self.config["wandb"]["enabled"]:
            wandb.finish()
        
        return output_dir
    
    def resume_training(self, checkpoint_dir: str):
        """Resume training from checkpoint"""
        self.setup()
        train_dataset, val_dataset = self.prepare_data()
        
        training_args = self.finetuner.get_training_arguments()
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.finetuner.tokenizer,
            mlm=False
        )
        
        self.trainer = Trainer(
            model=self.finetuner.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        self.trainer.train(resume_from_checkpoint=checkpoint_dir)
        
        output_dir = os.path.join(
            self.config["training"]["output_dir"],
            "final"
        )
        self.finetuner.save_model(output_dir)
        
        return output_dir
