"""
Example: Complete AI/ML Pipeline
Demonstrates the full workflow from training to serving to evaluation
"""

import json
from nexum.ai.training import Trainer
from nexum.ai.serving import RayDeployer
from nexum.ai.rag import RAGEngine
from nexum.ai.evaluation import Evaluator


def main():
    print("=" * 60)
    print("Nexum AI/ML Complete Pipeline Example")
    print("=" * 60)
    
    # Step 1: Fine-tune model with Unsloth + PEFT
    print("\n[Step 1] Fine-tuning model with Unsloth + PEFT...")
    trainer = Trainer("config/ai/training_config.json")
    # output_dir = trainer.train()  # Uncomment to run actual training
    # print(f"Model saved to: {output_dir}")
    print("Training configuration loaded successfully.")
    
    # Step 2: Deploy model with Ray Serve
    print("\n[Step 2] Deploying model with Ray Serve...")
    deployer = RayDeployer("config/ai/serving_config.json")
    # deployment = deployer.deploy()  # Uncomment to run actual deployment
    print("Deployment configuration loaded successfully.")
    
    # Step 3: Setup RAG system
    print("\n[Step 3] Setting up RAG system...")
    rag = RAGEngine("config/ai/rag_config.json")
    
    # Add sample documents
    documents = [
        {
            "id": "doc1",
            "content": "Nexum v2.0 is a sophisticated hybrid operating system combining robotics, computer vision, and AI.",
            "metadata": {"source": "README"}
        },
        {
            "id": "doc2",
            "content": "The AI layer includes local models, neural networks, and generative AI capabilities.",
            "metadata": {"source": "README"}
        }
    ]
    rag.add_documents(documents)
    print("RAG system initialized with sample documents.")
    
    # Step 4: Query RAG system
    print("\n[Step 4] Querying RAG system...")
    # response = rag.query("What is Nexum v2.0?")  # Uncomment to run actual query
    print("RAG query configuration loaded successfully.")
    
    # Step 5: Evaluate with Ragas + Weights & Biases
    print("\n[Step 5] Evaluating with Ragas + Weights & Biases...")
    evaluator = Evaluator("config/ai/evaluation_config.json", run_name="complete-pipeline")
    print("Evaluation configuration loaded successfully.")
    
    print("\n" + "=" * 60)
    print("Pipeline configuration complete!")
    print("=" * 60)
    print("\nTo run the actual pipeline:")
    print("1. Uncomment the training step and run: python examples/ai/finetune_example.py")
    print("2. Start the serving: python examples/ai/serving_example.py")
    print("3. Run RAG queries: python examples/ai/rag_example.py")
    print("4. Evaluate: python examples/ai/evaluation_example.py")


if __name__ == "__main__":
    main()
