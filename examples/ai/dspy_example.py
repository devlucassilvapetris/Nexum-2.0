"""
Example: DSPy Prompt Optimization
"""

import dspy
from nexum.ai.prompt_optimization import DSPOptimizer, RAGModule


def main():
    # Initialize optimizer
    optimizer = DSPOptimizer("config/ai/dspy_config.json")
    
    # Create a RAG module
    rag_module = RAGModule(num_passages=3)
    
    # Create training examples
    trainset = [
        dspy.Example(
            question="What is Nexum v2.0?",
            answer="Nexum v2.0 is a hybrid operating system for advanced human-machine interface."
        ).with_inputs("question"),
        dspy.Example(
            question="What technologies are used for fine-tuning?",
            answer="Unsloth and PEFT are used for efficient fine-tuning."
        ).with_inputs("question"),
    ]
    
    # Define a simple metric
    def exact_match_metric(gold, pred, trace=None):
        return gold.answer.lower() == pred.answer.lower()
    
    # Optimize the module
    optimized_module = optimizer.optimize_module(
        rag_module,
        trainset,
        metric=exact_match_metric
    )
    
    # Test the optimized module
    test_question = "What is the purpose of Nexum?"
    result = optimized_module(question=test_question)
    
    print(f"Question: {test_question}")
    print(f"Answer: {result.answer}")


if __name__ == "__main__":
    main()
