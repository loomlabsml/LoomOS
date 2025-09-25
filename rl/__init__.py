"""
LoomOS RL Module - Reinforcement Learning Components

This module provides comprehensive RL training capabilities including:
- Advanced PPO training with GAE
- Multi-environment RL Gym
- Distributed training integration
- WEAVE algorithm for hierarchical RL with adaptive exploration
- Complete environment suite (Math, Game, ToolCall, Code, Language, Multimodal)
"""

# Core RL Training
from .ppo_trainer import PPOTrainer, PPOConfig, PPOPolicy
from .algos.ppo import PPO
from .algos.dpo import DPO  
from .algos.grpo import GRPO

# WEAVE Algorithm
from .algos.weave import (
    WeaveAlgorithm,
    WeaveConfig,
    ExplorationStrategy,
    RewardThread,
    create_weave_algorithm
)
from .algos.weave_gym_integration import (
    WeaveGymTrainer,
    create_weave_gym_trainer
)
from .algos.apollo_weave_config import (
    ApolloWeaveTrainer,
    create_apollo_weave_trainer
)

# RL Gym System
from .loom_gym import (
    LoomRLGym,
    LoomEnvironment,
    EnvironmentType,
    EnvironmentFactory,
    TrajectoryAPI,
    Trajectory,
    MathEnvironment,
    GameEnvironment,
    ToolCallEnvironment
)

# Integration Layer
from .gym_integration import (
    GymPPOIntegration,
    GymTrainingConfig,
    create_integrated_rl_system
)

# Environment utilities
from .envs import create_custom_environment

__all__ = [
    # Core training
    "PPOTrainer", "PPOConfig", "PPOPolicy",
    "PPO", "DPO", "GRPO",
    
    # WEAVE Algorithm
    "WeaveAlgorithm", "WeaveConfig", "ExplorationStrategy", "RewardThread",
    "create_weave_algorithm", "WeaveGymTrainer", "create_weave_gym_trainer",
    "ApolloWeaveTrainer", "create_apollo_weave_trainer",
    
    # Gym system  
    "LoomRLGym", "LoomEnvironment", "EnvironmentType", "EnvironmentFactory",
    "TrajectoryAPI", "Trajectory",
    "MathEnvironment", "GameEnvironment", "ToolCallEnvironment",
    
    # Integration
    "GymPPOIntegration", "GymTrainingConfig", "create_integrated_rl_system",
    
    # Utilities
    "create_custom_environment"
]

# Version info
__version__ = "1.0.0"
__author__ = "Loom Labs"