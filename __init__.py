"""
LoomOS Main Package

The Iron Suit for AI Models - Enterprise Distributed AI Runtime Platform
"""

# Core LoomOS components
from .core import LoomCore, LoomAgent, Marketplace
from .nexus import MasterNode, Worker, FailoverCoordinator
from .rl import PPOTrainer, LoomRLGym, create_integrated_rl_system
from .blocks import ModelRegistry, create_adapter

__version__ = "1.0.0"
__author__ = "Loom Labs"

__all__ = [
    "LoomCore", "LoomAgent", "Marketplace",
    "MasterNode", "Worker", "FailoverCoordinator", 
    "PPOTrainer", "LoomRLGym", "create_integrated_rl_system",
    "ModelRegistry", "create_adapter"
]