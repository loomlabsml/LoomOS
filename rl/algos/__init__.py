"""
WEAVE Algorithm Module - __init__.py

Weighted Exploration, Adaptive Value Estimation (WEAVE) algorithm for LoomOS.

WEAVE integrates:
- Hierarchical reinforcement learning
- Adaptive weighting of exploration strategies  
- Distributed value estimation across nodes
- Hierarchical reward shaping

Key Components:
- WeaveAlgorithm: Core algorithm implementation
- WeaveGymTrainer: Integration with LoomOS RL Gym
- ApolloWeaveTrainer: Specialized for multi-agent chatbot training
"""

from .weave import (
    # Core algorithm
    WeaveAlgorithm,
    WeaveConfig,
    
    # Algorithm components
    ExplorationStrategy,
    RewardThread,
    ExplorationDistribution,
    RewardThreadProcessor,
    DistributedValueEstimator,
    WeaveActor,
    
    # Factory function
    create_weave_algorithm,
    
    # Example
    example_weave_training
)

from .weave_gym_integration import (
    # Gym integration
    WeaveGymTrainer,
    WeaveGymConfig,
    
    # Factory function
    create_weave_gym_trainer,
    
    # Example
    example_weave_gym_integration
)

from .apollo_weave_config import (
    # Apollo-specific training
    ApolloWeaveTrainer,
    ApolloTrainingConfig,
    
    # Factory function
    create_apollo_weave_trainer,
    
    # Example
    example_apollo_weave_training
)

__all__ = [
    # Core WEAVE
    "WeaveAlgorithm",
    "WeaveConfig", 
    "ExplorationStrategy",
    "RewardThread",
    "create_weave_algorithm",
    
    # Gym Integration
    "WeaveGymTrainer",
    "WeaveGymConfig",
    "create_weave_gym_trainer",
    
    # Apollo Training
    "ApolloWeaveTrainer", 
    "ApolloTrainingConfig",
    "create_apollo_weave_trainer",
    
    # Examples
    "example_weave_training",
    "example_weave_gym_integration", 
    "example_apollo_weave_training"
]

# Algorithm metadata
WEAVE_VERSION = "1.0.0"
WEAVE_DESCRIPTION = """
WEAVE (Weighted Exploration, Adaptive Value Estimation) integrates hierarchical 
reinforcement learning, adaptive weighting of exploration strategies, and distributed 
value estimation across nodes. Inspired by the metaphor of a loom weaving threads 
into fabric, WEAVE combines multiple "threads" of learning (policies, exploration 
modes, reward signals) into a unified model that scales across distributed clusters.
"""

def get_weave_info():
    """Get WEAVE algorithm information"""
    return {
        "name": "WEAVE",
        "version": WEAVE_VERSION,
        "description": WEAVE_DESCRIPTION,
        "features": [
            "Hierarchical reinforcement learning",
            "Adaptive exploration strategy weighting", 
            "Distributed value estimation",
            "Hierarchical reward shaping",
            "Multi-objective optimization",
            "Node specialization support"
        ],
        "use_cases": [
            "Multi-agent chatbot training",
            "Distributed RL across clusters",
            "Multi-objective policy learning",
            "Adaptive exploration in complex environments"
        ]
    }