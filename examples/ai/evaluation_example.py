"""
Example: Evaluation with Ragas and Weights & Biases
"""

from datasets import Dataset
from nexum.ai.evaluation import Evaluator


def main():
    # Initialize evaluator
    evaluator = Evaluator("config/ai/evaluation_config.json", run_name="rag-evaluation")
    
    # Create sample evaluation dataset
    dataset = Dataset.from_dict({
        "question": [
            "What is Nexum v2.0?",
            "What technologies are used for fine-tuning?"
        ],
        "answer": [
            "Nexum v2.0 is a hybrid operating system for advanced human-machine interface.",
            "Unsloth and PEFT are used for efficient fine-tuning."
        ],
        "contexts": [
            ["Nexum v2.0 is a sophisticated hybrid operating system."],
            ["Fine-tuning uses Unsloth for efficient training with PEFT."]
        ],
        "ground_truth": [
            "Nexum v2.0 is a hybrid operating system.",
            "Unsloth and PEFT are used for fine-tuning."
        ]
    })
    
    # Run evaluation
    results = evaluator.evaluate(dataset)
    
    # Generate report
    report = evaluator.ragas_evaluator.generate_report(results)
    print("Evaluation Report:")
    print(report)
    
    # Finish evaluation
    evaluator.finish()


if __name__ == "__main__":
    main()
