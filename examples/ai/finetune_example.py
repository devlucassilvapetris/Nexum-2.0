"""
Example: Fine-tuning with Unsloth + PEFT
"""

import json
from nexum.ai.training import Trainer


def main():
    # Initialize trainer with config
    trainer = Trainer("config/ai/training_config.json")
    
    # Run fine-tuning
    output_dir = trainer.train()
    
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
