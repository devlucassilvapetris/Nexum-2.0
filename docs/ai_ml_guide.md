# Nexum AI/ML Guide

This guide covers the comprehensive AI/ML implementation in Nexum v2.0, including fine-tuning, serving, RAG, prompt optimization, and evaluation.

## Overview

Nexum v2.0 includes a complete AI/ML stack with the following components:

- **Training/Fine-tuning**: PyTorch + Transformers + Unsloth + PEFT
- **Serving**: vLLM + Ray Serve
- **RAG**: LangChain + vLLM + Qdrant + Instructor
- **Prompt Optimization**: DSPy
- **Vector Databases**: Qdrant + Milvus
- **Evaluation**: Ragas + Weights & Biases

## Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All AI/ML configurations are stored in `config/ai/`:

- `training_config.json` - Fine-tuning configuration
- `serving_config.json` - Model serving configuration
- `rag_config.json` - RAG system configuration
- `evaluation_config.json` - Evaluation configuration
- `dspy_config.json` - DSPy prompt optimization configuration

## Fine-tuning with Unsloth + PEFT

### Overview

Unsloth provides efficient fine-tuning with memory optimizations, while PEFT (Parameter-Efficient Fine-Tuning) enables training with minimal parameters using LoRA adapters.

### Configuration

Edit `config/ai/training_config.json`:

```json
{
  "model": {
    "base_model": "unsloth/llama-3-8b-bnb-4bit",
    "max_seq_length": 2048,
    "load_in_4bit": true
  },
  "training": {
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 2
  },
  "lora": {
    "r": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", ...]
  }
}
```

### Usage

```python
from nexum.ai.training import Trainer

# Initialize trainer
trainer = Trainer("config/ai/training_config.json")

# Run fine-tuning
output_dir = trainer.train()

print(f"Model saved to: {output_dir}")
```

### Data Format

Training data should be in JSONL format:

```json
{"instruction": "What is Nexum?", "input": "", "output": "Nexum is a hybrid OS..."}
{"instruction": "Explain the AI layer", "input": "", "output": "The AI layer includes..."}
```

## Model Serving with vLLM + Ray Serve

### Overview

vLLM provides high-throughput LLM serving with PagedAttention, while Ray Serve enables scalable deployment with autoscaling.

### Configuration

Edit `config/ai/serving_config.json`:

```json
{
  "vllm": {
    "model": "./checkpoints/final",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9
  },
  "ray_serve": {
    "enabled": true,
    "num_replicas": 1,
    "max_concurrent_queries": 100
  }
}
```

### Usage

```python
from nexum.ai.serving import VLLMServe, RayDeployer

# Option 1: Direct vLLM serving
vllm_serve = VLLMServe("config/ai/serving_config.json")
outputs = vllm_serve.generate(["What is Nexum?"])

# Option 2: Deploy with Ray Serve
deployer = RayDeployer("config/ai/serving_config.json")
deployment = deployer.deploy()
```

### API Access

Once deployed, access the API at `http://localhost:8000`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is Nexum?"}]}'
```

## RAG with LangChain + vLLM + Qdrant + Instructor

### Overview

Retrieval-Augmented Generation combines document retrieval with LLM generation for accurate, context-aware responses.

### Configuration

Edit `config/ai/rag_config.json`:

```json
{
  "vector_db": {
    "type": "qdrant",
    "host": "localhost",
    "port": 6333,
    "collection_name": "nexum_documents"
  },
  "embedding": {
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "retrieval": {
    "top_k": 5,
    "score_threshold": 0.7
  }
}
```

### Usage

```python
from nexum.ai.rag import RAGEngine

# Initialize RAG engine
rag = RAGEngine("config/ai/rag_config.json")

# Add documents
documents = [
    {
        "id": "doc1",
        "content": "Nexum v2.0 is a hybrid operating system...",
        "metadata": {"source": "README"}
    }
]
rag.add_documents(documents)

# Query
response = rag.query("What is Nexum v2.0?")
print(f"Answer: {response.answer.answer}")
print(f"Confidence: {response.answer.confidence}")
```

### Structured Outputs with Instructor

Instructor provides structured outputs using Pydantic schemas:

```python
from nexum.ai.rag.schemas import Answer

# The RAG engine automatically uses Instructor for structured outputs
# Returns Answer object with: answer, sources, confidence, reasoning
```

## Vector Databases

### Milvus (Production)

Milvus is the recommended vector database for production deployments, designed for enterprise-scale applications handling billions of vectors:

```python
# Start Milvus
docker run -p 19530:19530 milvusdb/milvus:latest

# Configure in rag_config.json
"vector_db": {
  "type": "milvus",
  "host": "localhost",
  "port": 19530
}
```

**Why Milvus for Production:**
- Scales to billions of vectors with sub-millisecond latency
- Supports multiple index types (IVF, HNSW, DiskANN)
- Cloud-native architecture with Kubernetes support
- Advanced filtering and hybrid search capabilities
- Enterprise-grade security and monitoring

### Qdrant (Development/PoC)

Qdrant is ideal for rapid prototyping and local development:

```python
# Start Milvus
docker run -p 19530:19530 milvusdb/milvus:latest

# Configure in rag_config.json
"milvus": {
  "enabled": true,
  "host": "localhost",
  "port": 19530
}
```

## DSPy Prompt Optimization

### Overview

DSPy enables programmatic prompting with automatic prompt optimization.

### Configuration

Edit `config/ai/dspy_config.json`:

```json
{
  "lm": {
    "provider": "vllm",
    "model": "./checkpoints/final",
    "api_base": "http://localhost:8000/v1"
  },
  "optimization": {
    "num_trials": 10,
    "max_bootstrapped_demos": 5
  }
}
```

### Usage

```python
import dspy
from nexum.ai.prompt_optimization import DSPOptimizer, RAGModule

# Initialize optimizer
optimizer = DSPOptimizer("config/ai/dspy_config.json")

# Create module
rag_module = RAGModule(num_passages=3)

# Create training examples
trainset = [
    dspy.Example(
        question="What is Nexum?",
        answer="Nexum is a hybrid OS..."
    ).with_inputs("question")
]

# Optimize
optimized_module = optimizer.optimize_module(
    rag_module,
    trainset,
    metric=exact_match_metric
)
```

### Pre-built Modules

- `RAGModule` - Retrieval-augmented generation
- `QAModule` - Question answering
- `ClassificationModule` - Text classification
- `SummarizationModule` - Text summarization
- `TranslationModule` - Translation
- `CodeGenerationModule` - Code generation

## Evaluation with Ragas + Weights & Biases

### Overview

Ragas provides comprehensive RAG evaluation metrics, while Weights & Biases tracks experiments.

### Configuration

Edit `config/ai/evaluation_config.json`:

```json
{
  "ragas": {
    "metrics": [
      "faithfulness",
      "answer_relevancy",
      "context_precision",
      "context_recall"
    ]
  },
  "wandb": {
    "enabled": true,
    "project": "nexum-evaluation"
  }
}
```

### Usage

```python
from datasets import Dataset
from nexum.ai.evaluation import Evaluator

# Initialize evaluator
evaluator = Evaluator("config/ai/evaluation_config.json")

# Create dataset
dataset = Dataset.from_dict({
    "question": ["What is Nexum?"],
    "answer": ["Nexum is a hybrid OS..."],
    "contexts": [["Nexum v2.0 is..."]],
    "ground_truth": ["Nexum is a hybrid OS..."]
})

# Evaluate
results = evaluator.evaluate(dataset)

# Generate report
report = evaluator.ragas_evaluator.generate_report(results)
print(report)
```

### Ragas Metrics

- **Faithfulness** - Factual consistency of answer with context
- **Answer Relevancy** - Relevance of answer to question
- **Context Precision** - Precision of retrieved context
- **Context Recall** - Recall of retrieved context
- **Context Entity Recall** - Entity-level recall

### Weights & Biases Integration

All evaluation metrics are automatically logged to Weights & Biases:

```python
# Training metrics
evaluator.log_training_progress({"loss": 0.5, "accuracy": 0.9}, epoch=1)

# Model artifacts
evaluator.log_model("./checkpoints/final", "nexum-model")
```

## Complete Pipeline Example

Run the complete pipeline from training to evaluation:

```bash
# 1. Fine-tune model
python examples/ai/finetune_example.py

# 2. Start serving
python examples/ai/serving_example.py

# 3. Run RAG queries
python examples/ai/rag_example.py

# 4. Evaluate
python examples/ai/evaluation_example.py
```

Or use the complete pipeline script:

```bash
python examples/ai/complete_pipeline.py
```

## Architecture

```
ai/
├── training/           # Fine-tuning with Unsloth + PEFT
│   ├── finetuner.py
│   ├── data_processor.py
│   └── trainer.py
├── serving/            # Model serving with vLLM + Ray Serve
│   ├── vllm_serve.py
│   └── ray_deployer.py
├── rag/                # RAG with LangChain + vLLM + Qdrant + Instructor
│   ├── rag_engine.py
│   ├── vector_store.py
│   └── schemas.py
├── prompt_optimization/  # DSPy prompt optimization
│   ├── optimizer.py
│   └── modules.py
├── evaluation/         # Ragas + Weights & Biases
│   ├── ragas_evaluator.py
│   ├── wandb_tracker.py
│   └── evaluator.py
└── vector_db/          # Vector database utilities
```

## Best Practices

### Fine-tuning

1. Start with 4-bit quantization to save memory
2. Use LoRA with r=16 for good performance
3. Monitor training with Weights & Biases
4. Save checkpoints regularly

### Serving

1. Use Ray Serve for production deployments
2. Enable autoscaling for variable load
3. Set appropriate GPU memory utilization
4. Monitor query latency and throughput

### RAG

1. Use appropriate chunk sizes for documents
2. Tune top_k and score_threshold
3. Use Instructor for structured outputs
4. Evaluate with Ragas metrics

### Evaluation

1. Use diverse evaluation datasets
2. Track all metrics in Weights & Biases
3. Compare different configurations
4. Iterate based on evaluation results

## Troubleshooting

### CUDA Out of Memory

- Reduce `per_device_train_batch_size`
- Enable gradient checkpointing
- Use 4-bit quantization
- Reduce `max_seq_length`

### Slow Inference

- Increase `tensor_parallel_size`
- Optimize batch size
- Use vLLM's PagedAttention
- Enable Ray Serve autoscaling

### Poor RAG Performance

- Improve document quality
- Tune retrieval parameters
- Use better embedding models
- Evaluate with Ragas to identify issues

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/)
- [LangChain Documentation](https://python.langchain.com/)
- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [Ragas Documentation](https://docs.ragas.ai/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
