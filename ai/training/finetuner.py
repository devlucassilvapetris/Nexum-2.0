"""
Unsloth Fine-tuner
Efficient fine-tuning with Unsloth and PEFT
"""

import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class UnslothFinetuner:
    """Fine-tune models using Unsloth and PEFT"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def load_model(self):
        """Load base model with Unsloth optimizations"""
        model_config = self.config["model"]
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["base_model"],
            max_seq_length=model_config["max_seq_length"],
            dtype=model_config["dtype"],
            load_in_4bit=model_config["load_in_4bit"],
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        return self.model, self.tokenizer
    
    def apply_lora(self):
        """Apply LoRA adapters for parameter-efficient fine-tuning"""
        lora_config = self.config["lora"]
        
        self.peft_model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config["r"],
            target_modules=lora_config["target_modules"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
            random_state=lora_config["random_state"],
        )
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def get_training_arguments(self):
        """Get training arguments from config"""
        training_config = self.config["training"]
        
        from transformers import TrainingArguments
        
        return TrainingArguments(
            output_dir=training_config["output_dir"],
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            learning_rate=training_config["learning_rate"],
            warmup_steps=training_config["warmup_steps"],
            logging_steps=training_config["logging_steps"],
            save_steps=training_config["save_steps"],
            eval_steps=training_config["eval_steps"],
            optim=training_config["optim"],
            weight_decay=training_config["weight_decay"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            gradient_checkpointing=training_config["gradient_checkpointing"],
            fp16=training_config["fp16"],
            report_to="wandb" if self.config["wandb"]["enabled"] else None,
            run_name=self.config["wandb"].get("run_name"),
            save_total_limit=2,
            load_best_model_at_end=True,
        )
    
    def save_model(self, output_dir: str):
        """Save fine-tuned model"""
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save for vLLM inference
        self.peft_model.save_pretrained_merged(
            output_dir,
            self.tokenizer,
            save_method="merged_16bit"
        )
