"""
WEAVE Integration with LoomOS RL Gym

This module integrates the WEAVE algorithm with the LoomOS RL Gym system,
providing seamless distributed training with hierarchical reinforcement learning,
adaptive exploration, and distributed value estimation.

The integration enables:
- Multi-environment WEAVE training across specialized nodes
- Dynamic reward thread balancing for different objectives
- Adaptive exploration strategy weighting based on environment type
- Distributed value estimation with node specialization
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .weave import (
    WeaveAlgorithm, WeaveConfig, ExplorationStrategy, RewardThread,
    create_weave_algorithm
)

# Import LoomOS components
try:
    from ..loom_gym import LoomRLGym, LoomEnvironment
    from ..gym_integration import GymPPOIntegration
    LOOM_AVAILABLE = True
except ImportError:
    LOOM_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class WeaveGymConfig:
    """Configuration for WEAVE-LoomOS integration"""
    
    # Environment specialization
    environment_specializations: Dict[str, List[RewardThread]] = None
    exploration_by_env_type: Dict[str, List[ExplorationStrategy]] = None
    
    # Node allocation strategy
    nodes_per_environment: int = 2
    load_balancing_strategy: str = "round_robin"  # round_robin, performance_based, random
    
    # Training parameters
    episodes_per_update: int = 8
    experience_buffer_size: int = 10000
    curriculum_learning: bool = True
    
    # WEAVE-specific parameters
    reward_thread_adaptation_rate: float = 0.01
    exploration_weight_decay: float = 0.995
    node_trust_update_frequency: int = 100

class WeaveGymTrainer:
    """WEAVE trainer integrated with LoomOS RL Gym"""
    
    def __init__(self, 
                 gym: Optional['LoomRLGym'] = None,
                 config: Optional[WeaveGymConfig] = None):
        self.gym = gym
        self.config = config or WeaveGymConfig()
        
        # Initialize default specializations if not provided
        if self.config.environment_specializations is None:
            self.config.environment_specializations = self._default_env_specializations()
        
        if self.config.exploration_by_env_type is None:
            self.config.exploration_by_env_type = self._default_exploration_strategies()
        
        # WEAVE algorithms per environment type
        self.weave_algorithms: Dict[str, WeaveAlgorithm] = {}
        
        # Experience buffers
        self.experience_buffers: Dict[str, List] = {}
        
        # Training statistics
        self.training_stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "environment_performances": {},
            "reward_thread_evolution": {},
            "exploration_adaptation": {},
            "node_specialization_metrics": {}
        }
        
        # Node allocation tracking
        self.node_assignments: Dict[str, List[int]] = {}
        
        logger.info("WeaveGymTrainer initialized")
    
    def _default_env_specializations(self) -> Dict[str, List[RewardThread]]:
        """Default reward thread specializations by environment type"""
        return {
            "math": [RewardThread.TASK_COMPLETION, RewardThread.EFFICIENCY],
            "game": [RewardThread.TASK_COMPLETION, RewardThread.CREATIVITY, RewardThread.ENGAGEMENT],
            "tool_call": [RewardThread.TASK_COMPLETION, RewardThread.SAFETY, RewardThread.EFFICIENCY],
            "code": [RewardThread.TASK_COMPLETION, RewardThread.SAFETY, RewardThread.COHERENCE],
            "language": [RewardThread.COHERENCE, RewardThread.CREATIVITY, RewardThread.ENGAGEMENT],
            "multimodal": [RewardThread.TASK_COMPLETION, RewardThread.CREATIVITY, RewardThread.COHERENCE]
        }
    
    def _default_exploration_strategies(self) -> Dict[str, List[ExplorationStrategy]]:
        """Default exploration strategies by environment type"""
        return {
            "math": [ExplorationStrategy.EPSILON_GREEDY, ExplorationStrategy.UCB_EXPLORATION],
            "game": [ExplorationStrategy.CURIOSITY_DRIVEN, ExplorationStrategy.ENTROPY_BASED],
            "tool_call": [ExplorationStrategy.EPSILON_GREEDY, ExplorationStrategy.THOMPSON_SAMPLING],
            "code": [ExplorationStrategy.UCB_EXPLORATION, ExplorationStrategy.CURIOSITY_DRIVEN],
            "language": [ExplorationStrategy.ENTROPY_BASED, ExplorationStrategy.BOLTZMANN],
            "multimodal": [ExplorationStrategy.CURIOSITY_DRIVEN, ExplorationStrategy.ENTROPY_BASED, ExplorationStrategy.UCB_EXPLORATION]
        }
    
    def initialize_weave_algorithms(self):
        """Initialize WEAVE algorithms for each environment type"""
        
        if not self.gym:
            logger.warning("No gym provided, using mock environments")
            env_types = list(self.config.environment_specializations.keys())
        else:
            env_types = self.gym.get_available_environment_types()
        
        for env_type in env_types:
            # Get specialization for this environment
            reward_threads = self.config.environment_specializations.get(env_type, [
                RewardThread.TASK_COMPLETION,
                RewardThread.SAFETY
            ])
            
            exploration_strategies = self.config.exploration_by_env_type.get(env_type, [
                ExplorationStrategy.ENTROPY_BASED,
                ExplorationStrategy.CURIOSITY_DRIVEN
            ])
            
            # Create WEAVE algorithm
            weave = create_weave_algorithm(
                exploration_strategies=exploration_strategies,
                reward_threads=reward_threads,
                num_nodes=self.config.nodes_per_environment
            )
            
            self.weave_algorithms[env_type] = weave
            self.experience_buffers[env_type] = []
            
            # Assign nodes to this environment
            self.node_assignments[env_type] = list(range(
                len(self.node_assignments) * self.config.nodes_per_environment,
                (len(self.node_assignments) + 1) * self.config.nodes_per_environment
            ))
            
            logger.info(f"Initialized WEAVE for {env_type} with {len(reward_threads)} reward threads")
    
    async def train_episode(self, environment_type: str, episode_config: Dict = None) -> Dict:
        """Train a single episode using WEAVE algorithm"""
        
        if environment_type not in self.weave_algorithms:
            raise ValueError(f"No WEAVE algorithm for environment type: {environment_type}")
        
        weave = self.weave_algorithms[environment_type]
        
        # Get or create environment
        if self.gym:
            env = await self.gym.create_environment(environment_type, episode_config or {})
        else:
            # Mock environment for demo
            env = MockEnvironment(environment_type)
        
        # Episode data collection
        episode_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
            "info": []
        }
        
        # Reset environment
        if hasattr(env, 'reset'):
            state = await env.reset() if asyncio.iscoroutinefunction(env.reset) else env.reset()
        else:
            state = np.random.randn(weave.config.state_dim)
        
        total_reward = 0.0
        step_count = 0
        max_steps = 200
        
        # Episode loop
        while step_count < max_steps:
            # Get action from WEAVE policy
            action, log_prob, value = weave.get_action(state)
            
            # Take step in environment
            if hasattr(env, 'step'):
                if asyncio.iscoroutinefunction(env.step):
                    next_state, base_reward, done, info = await env.step(action)
                else:
                    next_state, base_reward, done, info = env.step(action)
            else:
                # Mock step
                next_state = np.random.randn(weave.config.state_dim)
                base_reward = np.random.randn() * 0.1
                done = step_count >= max_steps - 1
                info = {"mock": True}
            
            # Compute hierarchical reward using WEAVE
            hierarchical_reward, thread_rewards = weave.compute_hierarchical_reward(
                state, action, base_reward, info
            )
            
            # Store experience
            episode_data["states"].append(state)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(hierarchical_reward)
            episode_data["log_probs"].append(log_prob)
            episode_data["values"].append(value)
            episode_data["dones"].append(done)
            episode_data["info"].append({**info, "thread_rewards": thread_rewards})
            
            total_reward += hierarchical_reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        # Add episode to experience buffer
        self.experience_buffers[environment_type].append(episode_data)
        
        # Trim buffer if too large
        if len(self.experience_buffers[environment_type]) > self.config.experience_buffer_size:
            self.experience_buffers[environment_type].pop(0)
        
        # Update training statistics
        self.training_stats["total_episodes"] += 1
        self.training_stats["total_steps"] += step_count
        
        if environment_type not in self.training_stats["environment_performances"]:
            self.training_stats["environment_performances"][environment_type] = []
        
        self.training_stats["environment_performances"][environment_type].append(total_reward)
        
        return {
            "environment_type": environment_type,
            "total_reward": total_reward,
            "step_count": step_count,
            "thread_rewards": thread_rewards if 'thread_rewards' in locals() else {},
            "weave_stats": weave.get_weave_stats()
        }
    
    async def update_weave_algorithms(self) -> Dict:
        """Update all WEAVE algorithms with collected experience"""
        
        update_metrics = {}
        
        for env_type, weave in self.weave_algorithms.items():
            if len(self.experience_buffers[env_type]) < self.config.episodes_per_update:
                continue
            
            # Prepare batch data from multiple episodes
            batch_data = self._prepare_batch_data(env_type)
            
            # Update WEAVE algorithm
            metrics = weave.train_step(batch_data)
            update_metrics[env_type] = metrics
            
            # Adaptive reward thread weight adjustment
            if self.config.curriculum_learning:
                self._adapt_reward_threads(env_type, weave)
            
            # Clear processed episodes
            self.experience_buffers[env_type] = []
        
        return update_metrics
    
    def _prepare_batch_data(self, environment_type: str) -> Dict:
        """Prepare batch data from collected episodes"""
        
        episodes = self.experience_buffers[environment_type][-self.config.episodes_per_update:]
        
        all_states, all_actions, all_rewards = [], [], []
        all_log_probs, all_dones = [], []
        
        for episode in episodes:
            all_states.extend(episode["states"])
            all_actions.extend(episode["actions"])
            all_rewards.extend(episode["rewards"])
            all_log_probs.extend(episode["log_probs"])
            all_dones.extend(episode["dones"])
        
        return {
            "states": np.array(all_states),
            "actions": np.array(all_actions),
            "rewards": np.array(all_rewards),
            "log_probs": np.array(all_log_probs),
            "dones": np.array(all_dones)
        }
    
    def _adapt_reward_threads(self, environment_type: str, weave: WeaveAlgorithm):
        """Adapt reward thread weights based on performance"""
        
        # Get recent performance
        recent_rewards = self.training_stats["environment_performances"][environment_type][-10:]
        
        if len(recent_rewards) < 2:
            return
        
        # Compute performance trend
        recent_avg = np.mean(recent_rewards[-5:]) if len(recent_rewards) >= 5 else np.mean(recent_rewards)
        older_avg = np.mean(recent_rewards[:-5]) if len(recent_rewards) >= 10 else recent_rewards[0]
        
        performance_trend = recent_avg - older_avg
        
        # Adjust reward thread weights based on performance
        if performance_trend > 0:
            # Performance improving - slight exploration in reward weights
            noise = np.random.normal(0, 0.01, len(weave.reward_thread_weights))
            if hasattr(weave.reward_thread_weights, 'data'):
                weave.reward_thread_weights.data += noise
                weave.reward_thread_weights.data = torch.clamp(weave.reward_thread_weights.data, 0.01, 1.0)
                weave.reward_thread_weights.data /= weave.reward_thread_weights.data.sum()
        
        # Track evolution
        if environment_type not in self.training_stats["reward_thread_evolution"]:
            self.training_stats["reward_thread_evolution"][environment_type] = []
        
        current_weights = weave.reward_thread_weights.tolist() if hasattr(weave.reward_thread_weights, 'tolist') else [1.0]
        self.training_stats["reward_thread_evolution"][environment_type].append(current_weights)
    
    async def run_multi_environment_training(self, 
                                           num_episodes_per_env: int = 100,
                                           environments: List[str] = None) -> Dict:
        """Run training across multiple environments simultaneously"""
        
        if not self.weave_algorithms:
            self.initialize_weave_algorithms()
        
        env_types = environments or list(self.weave_algorithms.keys())
        
        print(f"🧵 Starting WEAVE multi-environment training")
        print(f"   Environments: {env_types}")
        print(f"   Episodes per environment: {num_episodes_per_env}")
        
        # Track training progress
        completed_episodes = {env_type: 0 for env_type in env_types}
        training_results = {env_type: [] for env_type in env_types}
        
        # Training loop
        while any(completed < num_episodes_per_env for completed in completed_episodes.values()):
            
            # Schedule episodes across environments
            for env_type in env_types:
                if completed_episodes[env_type] >= num_episodes_per_env:
                    continue
                
                # Train episode
                episode_result = await self.train_episode(env_type)
                training_results[env_type].append(episode_result)
                completed_episodes[env_type] += 1
                
                # Progress update
                if completed_episodes[env_type] % 10 == 0:
                    avg_reward = np.mean([r["total_reward"] for r in training_results[env_type][-10:]])
                    print(f"   {env_type}: {completed_episodes[env_type]}/{num_episodes_per_env} episodes, avg reward: {avg_reward:.3f}")
            
            # Update WEAVE algorithms periodically
            if sum(completed_episodes.values()) % (self.config.episodes_per_update * len(env_types)) == 0:
                update_results = await self.update_weave_algorithms()
                print(f"   Updated WEAVE algorithms: {list(update_results.keys())}")
        
        # Final algorithm updates
        final_updates = await self.update_weave_algorithms()
        
        # Compile results
        final_results = {
            "training_completed": True,
            "total_episodes": sum(completed_episodes.values()),
            "environment_results": training_results,
            "final_performances": {
                env_type: np.mean([r["total_reward"] for r in results])
                for env_type, results in training_results.items()
            },
            "weave_final_stats": {
                env_type: weave.get_weave_stats()
                for env_type, weave in self.weave_algorithms.items()
            },
            "training_stats": self.training_stats
        }
        
        print(f"✅ WEAVE multi-environment training completed!")
        print(f"   Total episodes: {final_results['total_episodes']}")
        for env_type, performance in final_results["final_performances"].items():
            print(f"   {env_type}: {performance:.3f} average reward")
        
        return final_results
    
    def get_best_algorithm(self) -> Tuple[str, WeaveAlgorithm]:
        """Get the best performing WEAVE algorithm"""
        
        best_env_type = None
        best_performance = float('-inf')
        
        for env_type in self.weave_algorithms:
            if env_type in self.training_stats["environment_performances"]:
                recent_performance = np.mean(
                    self.training_stats["environment_performances"][env_type][-10:]
                )
                if recent_performance > best_performance:
                    best_performance = recent_performance
                    best_env_type = env_type
        
        if best_env_type:
            return best_env_type, self.weave_algorithms[best_env_type]
        else:
            # Return first algorithm if no performance data
            env_type = list(self.weave_algorithms.keys())[0]
            return env_type, self.weave_algorithms[env_type]
    
    def export_weave_models(self, save_path: str = "weave_models"):
        """Export trained WEAVE models"""
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for env_type, weave in self.weave_algorithms.items():
            model_path = os.path.join(save_path, f"weave_{env_type}.json")
            
            # Export model configuration and weights (simplified)
            model_data = {
                "environment_type": env_type,
                "config": {
                    "exploration_strategies": [s.value for s in weave.config.exploration_strategies],
                    "reward_threads": [t.value for t in weave.config.reward_threads],
                    "num_nodes": weave.config.num_nodes
                },
                "stats": weave.get_weave_stats(),
                "training_history": self.training_stats
            }
            
            import json
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            print(f"Exported WEAVE model for {env_type} to {model_path}")

# Mock environment for testing without LoomOS
class MockEnvironment:
    """Mock environment for testing WEAVE integration"""
    
    def __init__(self, env_type: str):
        self.env_type = env_type
        self.state_dim = 512
        self.action_dim = 64
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.state_dim)
    
    def step(self, action):
        self.step_count += 1
        
        next_state = np.random.randn(self.state_dim)
        reward = np.random.randn() * 0.1 + (0.1 if self.env_type == "math" else 0.0)
        done = self.step_count >= 100
        info = {
            "env_type": self.env_type,
            "step": self.step_count,
            "safety_score": np.random.uniform(0.5, 1.0),
            "efficiency": np.random.uniform(0.8, 1.2)
        }
        
        return next_state, reward, done, info

# Factory function for easy creation
def create_weave_gym_trainer(
    gym: Optional['LoomRLGym'] = None,
    environment_specializations: Dict[str, List[RewardThread]] = None,
    **kwargs
) -> WeaveGymTrainer:
    """Create WEAVE gym trainer with specified configuration"""
    
    config = WeaveGymConfig(
        environment_specializations=environment_specializations,
        **kwargs
    )
    
    return WeaveGymTrainer(gym=gym, config=config)

# Example usage
async def example_weave_gym_integration():
    """Example of WEAVE-LoomOS integration"""
    
    print("🧵 WEAVE-LoomOS RL Gym Integration Demo")
    print("=" * 60)
    
    # Create trainer
    trainer = create_weave_gym_trainer()
    
    # Initialize algorithms
    trainer.initialize_weave_algorithms()
    
    print(f"✅ Initialized WEAVE for {len(trainer.weave_algorithms)} environment types")
    for env_type in trainer.weave_algorithms:
        specializations = trainer.config.environment_specializations.get(env_type, [])
        print(f"   {env_type}: {[s.value for s in specializations]}")
    
    # Run multi-environment training
    results = await trainer.run_multi_environment_training(
        num_episodes_per_env=50,
        environments=["math", "game", "code"]
    )
    
    # Show results
    print(f"\n📊 Training Results:")
    for env_type, performance in results["final_performances"].items():
        print(f"   {env_type}: {performance:.3f} average reward")
    
    # Get best algorithm
    best_env, best_algo = trainer.get_best_algorithm()
    print(f"\n🏆 Best performing algorithm: {best_env}")
    
    # Export models
    trainer.export_weave_models()
    
    print(f"\n✅ WEAVE-LoomOS integration demo completed!")

if __name__ == "__main__":
    asyncio.run(example_weave_gym_integration())