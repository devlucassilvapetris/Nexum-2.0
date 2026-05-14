"""
Nexum v2.0 AI Module
Artificial intelligence and neural network systems
"""

__version__ = "2.0.0"

from .models.local_models import LocalModelManager
from .models.transformer_models import TransformerModel
from .neural_nets.perception_network import PerceptionNetwork
from .neural_nets.decision_network import DecisionNetwork
from .neural_nets.generative_ai import GenerativeAI

__all__ = [
    "LocalModelManager",
    "TransformerModel",
    "PerceptionNetwork",
    "DecisionNetwork",
    "GenerativeAI"
]
