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
- CUDA-capable GPU (recommended for optimal performance)
- Zorin OS or compatible Linux distribution
- Minimum 4GB RAM (8GB+ recommended)

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
│   └── neural_nets/ # Neural network implementations
├── vision/          # Computer vision
│   ├── processing/  # Image processing
│   └── recognition/ # Object recognition
├── config/          # Configuration files
├── tests/           # Unit tests
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

### Minimum System
- CPU: Quad-core processor
- RAM: 4GB
- Storage: 50GB SSD
- GPU: Integrated graphics (CPU mode)

### Recommended System
- CPU: Octa-core processor
- RAM: 16GB
- Storage: 500GB NVMe SSD
- GPU: NVIDIA RTX 3060 or better (6GB+ VRAM)

### Supported Hardware
- **Motors**: Kalatec servo and stepper motors
- **Audio**: USB audio interfaces, built-in microphones
- **Vision**: USB cameras, Raspberry Pi cameras
- **GPIO**: Raspberry Pi GPIO pins
- **Communication**: Serial, USB, I2C, SPI

## Roadmap

### Current Phase (Rio de Janeiro)
- [x] Core system architecture
- [x] Hardware interface development
- [x] Basic AI model integration
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
