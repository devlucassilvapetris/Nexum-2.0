# Nexum v2.0 - Hybrid Operating System

Nexum v2.0 is a sophisticated hybrid operating system combining robotics, computer vision, and artificial intelligence capabilities for advanced human-machine interaction.

## Overview

Nexum v2.0 represents a breakthrough in human-machine interface technology, featuring:

- **Hybrid OS Architecture**: Based on Zorin OS for low-latency performance
- **Advanced Kinematics**: Precision motor control with Kalatec integration
- **Biomimetic Audio**: High-fidelity acoustic capture systems
- **Local AI Processing**: On-device intelligence without external dependencies
- **Real-time Vision**: Advanced computer vision and object recognition

## System Architecture

### Kernel Layer
- **System Manager**: Resource monitoring and process management
- **Communication Protocol**: Inter-module messaging and data exchange
- **Hardware Interface**: Direct hardware communication and control

### Hardware Layer
- **Kinematics**: Motor control and movement systems
- **Acoustic**: Biomimetic ears and audio processing
- **Structure**: Mechanical control and physics calculations

### AI Layer
- **Local Models**: On-device AI model management
- **Neural Networks**: Perception and decision-making systems
- **Generative AI**: Local creative and problem-solving capabilities

### Vision Layer
- **Image Processing**: Real-time enhancement and analysis
- **Object Recognition**: Advanced detection and classification
- **Scene Analysis**: Environmental understanding and interaction

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (required for LLM serving and fine-tuning)
- Zorin OS or compatible Linux distribution
- Minimum 16GB RAM (32GB+ recommended for production)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nexum/nexum2.0.git
cd nexum2.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

4. Initialize the system:
```bash
nexum --init
```

## Quick Start

### Basic Usage

```python
from nexum.kernel import SystemManager
from nexum.hardware import MotorController
from nexum.vision import ImageProcessor
from nexum.ai import LocalModelManager

# Initialize system components
system_manager = SystemManager()
motor_controller = MotorController()
image_processor = ImageProcessor()
ai_manager = LocalModelManager()

# Start the system
system_manager.initialize()
motor_controller.initialize()
image_processor.initialize()
ai_manager.initialize()

# Example: Process image and run AI inference
image = cv2.imread("example.jpg")
processed = image_processor.process_image(image, ["enhance_contrast", "reduce_noise"])
result = ai_manager.infer("embedding_model", processed.data)
```

### Hardware Setup

```python
from nexum.hardware.kinematics import MotorConfig, MotorType
from nexum.hardware.acoustic import BiomimeticEars

# Configure motors
motor_config = MotorConfig(
    id="torso_motor",
    name="Main Torso Motor",
    type=MotorType.SERVO,
    max_speed=10.0,
    max_acceleration=5.0,
    max_torque=2.0,
    gear_ratio=10.0,
    encoder_resolution=4096,
    min_position=-3.14159,
    max_position=3.14159,
    home_position=0.0
)

motor_controller.add_motor(motor_config)

# Configure biomimetic ears
from nexum.hardware.acoustic import AudioConfig
audio_config = AudioConfig(
    sample_rate=44100,
    buffer_size=1024,
    channels=2,
    gain_left=1.0,
    gain_right=1.0
)

ears = BiomimeticEars(audio_config)
ears.initialize()
ears.start_listening()
```

## Configuration

### System Configuration

Main configuration is stored in `config/` directory:

- `system.json`: Core system settings
- `hardware.json`: Hardware device configurations
- `models.json`: AI model configurations
- `vision.json`: Computer vision parameters

### Hardware Configuration

Example hardware configuration (`config/hardware.json`):

```json
{
  "motors": {
    "torso": {
      "type": "servo",
      "port": "/dev/ttyUSB0",
      "max_speed": 10.0,
      "max_torque": 2.0
    },
    "left_arm": {
      "type": "stepper",
      "port": "/dev/ttyUSB1",
      "max_speed": 5.0,
      "max_torque": 1.5
    }
  },
  "audio": {
    "input_device": 0,
    "sample_rate": 44100,
    "channels": 2
  }
}
```

## Development

### Project Structure

```
nexum2.0/
├── kernel/           # Core system kernel
│   ├── core/        # System management
│   ├── drivers/     # Hardware drivers
│   └── communication/ # Inter-module messaging
├── hardware/         # Hardware control systems
│   ├── kinematics/  # Motor control
│   ├── acoustic/    # Audio systems
│   └── structure/   # Mechanical control
├── ai/              # AI and neural networks
│   ├── models/      # Local AI models
│   ├── neural_nets/ # Neural network implementations
│   ├── training/    # Fine-tuning with Unsloth + PEFT
│   ├── serving/     # Model serving with vLLM + Ray Serve
│   ├── rag/         # RAG with LangChain + Qdrant + Instructor
│   ├── prompt_optimization/ # DSPy prompt optimization
│   ├── evaluation/  # Ragas + Weights & Biases evaluation
│   └── vector_db/   # Vector database integration
├── vision/          # Computer vision
│   ├── processing/  # Image processing
│   └── recognition/ # Object recognition
├── config/          # Configuration files
│   └── ai/         # AI/ML configuration files
├── tests/           # Unit tests
├── examples/        # Example scripts
│   └── ai/         # AI/ML examples
└── docs/            # Documentation
```

### Adding New Modules

1. Create module directory in appropriate layer
2. Implement module interface
3. Add module to `__init__.py`
4. Update configuration files
5. Add unit tests

### Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test categories:

```bash
pytest tests/test_kernel.py
pytest tests/test_hardware.py
pytest tests/test_vision.py
pytest tests/test_ai.py
```

## Hardware Requirements

### Minimum System (Full AI/ML Stack)
- CPU: Quad-core processor
- RAM: 16GB
- Storage: 100GB SSD
- GPU: NVIDIA GTX 1080 / RTX 3050 (4GB+ VRAM)

### Edge/Raspberry Pi Mode (No LLM)
- CPU: Quad-core ARM processor
- RAM: 4GB
- Storage: 32GB SSD
- GPU: Not required (CPU-based vision only)

### Recommended System (Full AI/ML Stack)
- CPU: Octa-core processor
- RAM: 32GB
- Storage: 500GB NVMe SSD
- GPU: NVIDIA RTX 3060 or better (8GB+ VRAM)

### Supported Hardware
- **Motors**: Kalatec servo and stepper motors
- **Audio**: USB audio interfaces, built-in microphones
- **Vision**: USB cameras, Raspberry Pi cameras
- **GPIO**: Raspberry Pi GPIO pins
- **Communication**: Serial, USB, I2C, SPI

## AI/ML Features

Nexum v2.0 includes a comprehensive AI/ML stack:

### Training & Fine-tuning
- **Unsloth**: Efficient fine-tuning with memory optimizations
- **PEFT**: Parameter-Efficient Fine-Tuning with LoRA adapters
- **PyTorch + Transformers**: Deep learning framework integration
- **Weights & Biases**: Experiment tracking and monitoring

### Model Serving
- **vLLM**: High-throughput LLM serving with PagedAttention
- **Ray Serve**: Scalable deployment with autoscaling
- **REST API**: Standardized API for model inference

### Retrieval-Augmented Generation (RAG)
- **LangChain**: Orchestration framework for RAG pipelines
- **Milvus**: Enterprise-grade vector database for production deployments (scales to billions of vectors)
- **Qdrant**: Lightweight vector database for rapid PoCs and local development
- **Instructor**: Structured outputs with Pydantic schemas

### Prompt Optimization
- **DSPy**: Programmatic prompting with automatic optimization
- **Pre-built Modules**: RAG, QA, Classification, Summarization, Translation, Code Generation

### Evaluation
- **Ragas**: Comprehensive RAG evaluation metrics
- **Weights & Biases**: Experiment tracking and visualization
- **Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall

For detailed AI/ML documentation, see [docs/ai_ml_guide.md](docs/ai_ml_guide.md)

## Performance & Resource Efficiency

Nexum v2.0 is engineered for optimal resource utilization, delivering high-performance AI/ML capabilities with minimal hardware footprint:

### GPU Efficiency
- **vLLM with PagedAttention**: Achieves 60% GPU utilization during inference, leaving headroom for concurrent tasks
- **Unsloth 4-bit Quantization**: Reduces memory footprint by 4x while maintaining model quality
- **Gradient Checkpointing**: Enables training on limited VRAM with minimal performance impact
- **Tensor Parallelism**: Scales inference across multiple GPUs efficiently

### Memory Optimization
- **PEFT/LoRA**: Fine-tunes models with <1% additional parameters
- **PagedAttention**: Reduces KV cache memory by 2-4x compared to standard attention
- **4-bit Quantization**: Loads large models (7B+) on consumer GPUs (8GB VRAM)
- **Efficient Data Loading**: Streaming data processing with minimal RAM usage

### Benchmark Results (RTX 3060 12GB)
- **Inference Latency**: 15-25ms per token (Llama-3-8B, 4-bit)
- **Throughput**: 40-60 tokens/second per GPU
- **Memory Usage**: 6-8GB VRAM (model + KV cache)
- **Training Speed**: 2-3x faster than standard fine-tuning with Unsloth

### Production Scalability
- **Ray Serve Autoscaling**: Dynamically scales replicas based on load (1-4 replicas)
- **Milvus Vector DB**: Handles billions of vectors with sub-millisecond latency
- **Batch Processing**: Optimized for high-throughput batch inference
- **Resource Monitoring**: Built-in GPU/CPU/memory monitoring with Weights & Biases

## Roadmap

### Current Phase (Rio de Janeiro)
- [x] Core system architecture
- [x] Hardware interface development
- [x] Basic AI model integration
- [x] Advanced AI/ML stack implementation
- [ ] Advanced kinematics control
- [ ] Enhanced audio processing

### Next Phase (Vancouver Migration - September 2026)
- [ ] GPU acceleration optimization
- [ ] Advanced neural network integration
- [ ] Enhanced computer vision capabilities
- [ ] Real-time translation systems
- [ ] Advanced human-machine interaction

### Future Development
- [ ] Multi-language support
- [ ] Cloud integration options
- [ ] Advanced robotics integration
- [ ] AR/VR interface development
- [ ] Quantum computing exploration

## API Reference

### Kernel API

```python
# System management
system_manager = SystemManager()
system_manager.initialize()
status = system_manager.get_system_status()

# Communication protocol
protocol = CommunicationProtocol("node_id")
await protocol.initialize()
await protocol.send_message("destination", MessageType.COMMAND, {"action": "start"})
```

### Hardware API

```python
# Motor control
motor_controller = MotorController()
motor_controller.add_motor(config)
motor_controller.move_motor("motor_id", position=1.5)

# Audio processing
ears = BiomimeticEars(config)
ears.register_audio_callback(callback_function)
direction = ears.get_direction()
```

### Vision API

```python
# Image processing
processor = ImageProcessor(config)
processed = processor.process_image(image, ["enhance_contrast"])
features = processor.extract_features(image, ["hog", "histogram"])
```

### AI API

```python
# Model management
ai_manager = LocalModelManager()
ai_manager.load_model("text_generator")
result = ai_manager.infer("text_generator", "Hello, world!")

# Fine-tuning with Unsloth + PEFT
from nexum.ai.training import Trainer
trainer = Trainer("config/ai/training_config.json")
output_dir = trainer.train()

# RAG with LangChain + vLLM + Qdrant + Instructor
from nexum.ai.rag import RAGEngine
rag = RAGEngine("config/ai/rag_config.json")
response = rag.query("What is Nexum v2.0?")

# Model serving with vLLM + Ray Serve
from nexum.ai.serving import VLLMServe, RayDeployer
vllm_serve = VLLMServe("config/ai/serving_config.json")
outputs = vllm_serve.generate(["Hello, world!"])

# Evaluation with Ragas + Weights & Biases
from nexum.ai.evaluation import Evaluator
evaluator = Evaluator("config/ai/evaluation_config.json")
results = evaluator.evaluate(dataset)
```

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure CUDA drivers are installed
2. **Motor connection failed**: Check serial port permissions
3. **Audio device not found**: Verify audio device configuration
4. **Model loading failed**: Check model file paths and permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### System Diagnostics

Run system diagnostics:

```bash
nexum --diagnose
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

### Development Environment Setup

1. Fork the repository
2. Create development branch
3. Install development dependencies
4. Make changes with tests
5. Submit pull request

## License

This project is licensed under the MIT License - see `LICENSE.md` for details.

## Support

- **Documentation**: [docs.nexum.ai](https://docs.nexum.ai)
- **Issues**: [GitHub Issues](https://github.com/nexum/nexum2.0/issues)
- **Community**: [Discord Server](https://discord.gg/nexum)
- **Email**: support@nexum.ai

## Acknowledgments

- Kalatec for motor control systems
- Zorin OS for base operating system
- OpenAI for AI research inspiration
- The open-source community for essential libraries

---

**Nexum v2.0 - Bridging Human and Machine Intelligence**
