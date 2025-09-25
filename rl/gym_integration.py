"""
LoomOS RL Gym Integration - Connecting Environments with Training

This module provides the integration layer between:
- LoomOS RL Gym (multiple environments)
- PPO Trainer (policy optimization)
- Inference Engine (model serving)
- Nexus (distributed training)

Architecture follows the Atropos pattern with LoomOS enhancements.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone

# Import LoomOS components
from .loom_gym import LoomRLGym, EnvironmentType, Trajectory, TrajectoryAPI
from .ppo_trainer import PPOTrainer, PPOConfig, PPOPolicy

logger = logging.getLogger(__name__)

@dataclass
class GymTrainingConfig:
    """Configuration for gym-integrated training"""
    
    # Environment configuration
    environments_per_type: int = 2
    max_parallel_envs: int = 8
    episode_batch_size: int = 32
    
    # Training configuration
    collect_episodes_per_update: int = 50
    training_updates_per_collection: int = 5
    target_success_rate: float = 0.8
    
    # Environment types to include
    enabled_env_types: List[EnvironmentType] = None
    
    # Advanced settings
    adaptive_difficulty: bool = True
    curriculum_learning: bool = True
    multi_task_training: bool = True
    
    def __post_init__(self):
        if self.enabled_env_types is None:
            self.enabled_env_types = [
                EnvironmentType.MATH,
                EnvironmentType.GAME, 
                EnvironmentType.TOOLCALL
            ]

class GymPPOIntegration:
    """Integration layer between LoomOS RL Gym and PPO Trainer"""
    
    def __init__(self, 
                 gym_config: GymTrainingConfig,
                 ppo_config: PPOConfig):
        self.gym_config = gym_config
        self.ppo_config = ppo_config
        
        # Initialize components
        self.gym = LoomRLGym()
        self.ppo_trainer = None
        self.current_policy = None
        
        # Training state
        self.training_iteration = 0
        self.total_episodes_collected = 0
        self.integration_metrics = {
            "training_iterations": 0,
            "episodes_per_iteration": [],
            "average_reward_per_env": {},
            "success_rate_per_env": {},
            "curriculum_progress": {}
        }
        
        logger.info("Gym-PPO integration initialized")
    
    async def initialize(self):
        """Initialize the integrated training system"""
        
        # Initialize gym environments
        env_configs = self._create_environment_configs()
        self.gym.initialize_environments(env_configs)
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(self.ppo_config)
        await self.ppo_trainer.initialize()
        
        # Get initial policy
        self.current_policy = self.ppo_trainer.policy
        
        # Connect components
        self.gym.connect_rl_trainer(self.ppo_trainer)
        
        logger.info("Gym-PPO integration fully initialized")
    
    def _create_environment_configs(self) -> List[Dict[str, Any]]:
        """Create environment configurations based on gym config"""
        configs = []
        
        for env_type in self.gym_config.enabled_env_types:
            for i in range(self.gym_config.environments_per_type):
                
                if env_type == EnvironmentType.MATH:
                    # Create math environments with varying difficulty
                    difficulties = ["easy", "medium", "hard"]
                    problem_types = ["algebra", "calculus", "geometry"]
                    
                    config = {
                        "type": env_type.value,
                        "id": f"math_env_{i}",
                        "config": {
                            "difficulty_levels": [difficulties[i % len(difficulties)]],
                            "problem_types": [problem_types[i % len(problem_types)]],
                            "max_steps": 20 + i * 10  # Curriculum: easier envs have fewer steps
                        }
                    }
                
                elif env_type == EnvironmentType.GAME:
                    config = {
                        "type": env_type.value,
                        "id": f"game_env_{i}",
                        "config": {
                            "game_type": "tic_tac_toe",
                            "max_moves": 9,
                            "opponent_strength": 0.3 + i * 0.2  # Curriculum: stronger opponents
                        }
                    }
                
                elif env_type == EnvironmentType.TOOLCALL:
                    tool_sets = [
                        ["calculator"],
                        ["calculator", "web_search"],
                        ["calculator", "web_search", "file_reader", "code_executor"]
                    ]
                    
                    config = {
                        "type": env_type.value,
                        "id": f"tool_env_{i}",
                        "config": {
                            "tools": tool_sets[i % len(tool_sets)],
                            "max_tool_calls": 3 + i * 2  # Curriculum: more complex tasks
                        }
                    }
                
                configs.append(config)
        
        return configs
    
    async def run_training_loop(self, max_iterations: int = 100):
        """Run the complete gym-integrated training loop"""
        
        logger.info(f"Starting gym-integrated training for {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            self.training_iteration = iteration
            
            # Phase 1: Collect episodes from all environments
            logger.info(f"Iteration {iteration}: Collecting episodes...")
            trajectories = await self._collect_training_episodes()
            
            # Phase 2: Train PPO on collected data
            logger.info(f"Iteration {iteration}: Training PPO...")
            training_metrics = await self._train_ppo_on_trajectories(trajectories)
            
            # Phase 3: Update curriculum and difficulty
            if self.gym_config.curriculum_learning:
                await self._update_curriculum()
            
            # Phase 4: Update integration metrics
            self._update_integration_metrics(trajectories, training_metrics)
            
            # Phase 5: Log progress
            await self._log_training_progress(iteration, trajectories, training_metrics)
            
            # Check for early stopping
            if self._should_stop_training():
                logger.info(f"Early stopping at iteration {iteration}")
                break
        
        logger.info("Gym-integrated training completed")
        return self.integration_metrics
    
    async def _collect_training_episodes(self) -> List[Trajectory]:
        """Collect episodes from gym environments using current policy"""
        
        trajectories = []
        episodes_needed = self.gym_config.collect_episodes_per_update
        
        # Distribute episodes across environments
        env_ids = list(self.gym.environments.keys())
        episodes_per_env = max(1, episodes_needed // len(env_ids))
        
        # Collect episodes in parallel
        collection_tasks = []
        for env_id in env_ids[:self.gym_config.max_parallel_envs]:
            for _ in range(episodes_per_env):
                task = asyncio.create_task(
                    self._run_single_episode(env_id)
                )
                collection_tasks.append(task)
        
        # Wait for all episodes to complete
        completed_trajectories = await asyncio.gather(*collection_tasks)
        trajectories.extend(completed_trajectories)
        
        self.total_episodes_collected += len(trajectories)
        
        logger.info(f"Collected {len(trajectories)} episodes across {len(env_ids)} environments")
        return trajectories
    
    async def _run_single_episode(self, env_id: str) -> Trajectory:
        """Run a single episode with current policy"""
        
        environment = self.gym.environments[env_id]
        trajectory_id = self.gym.trajectory_api.start_trajectory(env_id)
        
        # Reset environment
        state = environment.reset()
        done = False
        
        while not done:
            # Get action from current policy
            if self.current_policy:
                action, log_prob, value = await self._get_policy_action(state, environment)
            else:
                # Random action if no policy yet
                action = environment.get_action_space().sample()
                log_prob = 0.0
                value = 0.0
            
            # Execute action
            next_state, reward, done, info = environment.step(action)
            
            # Add experience to trajectory
            self.gym.trajectory_api.add_experience(
                trajectory_id, state, action, reward, log_prob, value, info
            )
            
            state = next_state
        
        # Complete trajectory
        success = self._determine_episode_success(info, reward)
        trajectory = self.gym.trajectory_api.complete_trajectory(trajectory_id, success)
        
        return trajectory
    
    async def _get_policy_action(self, state, environment):
        """Get action from the PPO policy"""
        
        # Convert state to policy input format
        if isinstance(state, str):
            # Text-based environments
            policy_input = self.ppo_trainer.tokenizer.encode(
                state, return_tensors="pt", truncation=True, max_length=512
            )
        elif isinstance(state, np.ndarray):
            # Numeric environments (like game boards)
            policy_input = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        else:
            # Default: convert to string
            policy_input = self.ppo_trainer.tokenizer.encode(
                str(state), return_tensors="pt", truncation=True, max_length=512
            )
        
        # Get action from policy
        with torch.no_grad():
            action_logits, value = self.current_policy(policy_input)
            
            # Sample action based on environment type
            if hasattr(environment.get_action_space(), 'n'):
                # Discrete action space
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                action = action.item()
            else:
                # Continuous or text action space - use argmax for simplicity
                action = action_logits.argmax(dim=-1).item()
                log_prob = 0.0
        
        return action, log_prob.item() if hasattr(log_prob, 'item') else log_prob, value.item()
    
    def _determine_episode_success(self, info: Dict[str, Any], final_reward: float) -> bool:
        """Determine if an episode was successful"""
        
        # Check for explicit success indicators
        if "success" in info:
            return info["success"]
        
        if "winner" in info:
            return info["winner"] == 1
        
        if "task_complete" in info:
            return info["task_complete"]
        
        # Fallback: positive reward indicates success
        return final_reward > 0
    
    async def _train_ppo_on_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Train PPO using collected trajectories"""
        
        # Convert trajectories to PPO training format
        training_data = self._convert_trajectories_to_training_data(trajectories)
        
        # Run PPO training updates
        training_metrics = {}
        for update in range(self.gym_config.training_updates_per_collection):
            
            # Run single PPO update
            update_metrics = await self.ppo_trainer.train_step(training_data)
            
            # Accumulate metrics
            for key, value in update_metrics.items():
                if key not in training_metrics:
                    training_metrics[key] = []
                training_metrics[key].append(value)
        
        # Average metrics across updates
        averaged_metrics = {
            key: np.mean(values) for key, values in training_metrics.items()
        }
        
        # Update current policy
        self.current_policy = self.ppo_trainer.policy
        
        return averaged_metrics
    
    def _convert_trajectories_to_training_data(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Convert gym trajectories to PPO training format"""
        
        all_states = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_values = []
        all_advantages = []
        
        for trajectory in trajectories:
            # Calculate advantages using GAE
            advantages = self._calculate_advantages(trajectory.rewards, trajectory.values)
            
            # Add trajectory data
            all_states.extend(trajectory.states)
            all_actions.extend(trajectory.actions)
            all_rewards.extend(trajectory.rewards)
            all_log_probs.extend(trajectory.log_probs)
            all_values.extend(trajectory.values)
            all_advantages.extend(advantages)
        
        return {
            "states": all_states,
            "actions": all_actions,
            "rewards": all_rewards,
            "log_probs": all_log_probs,
            "values": all_values,
            "advantages": all_advantages
        }
    
    def _calculate_advantages(self, rewards: List[float], values: List[float], gamma: float = 0.99, lambda_: float = 0.95) -> List[float]:
        """Calculate GAE advantages"""
        
        if len(rewards) == 0:
            return []
        
        # Add bootstrap value (assume 0 for terminal states)
        values_with_bootstrap = values + [0.0]
        
        # Calculate TD residuals
        deltas = []
        for t in range(len(rewards)):
            delta = rewards[t] + gamma * values_with_bootstrap[t + 1] - values_with_bootstrap[t]
            deltas.append(delta)
        
        # Calculate GAE advantages
        advantages = []
        advantage = 0.0
        
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + gamma * lambda_ * advantage
            advantages.insert(0, advantage)
        
        return advantages
    
    async def _update_curriculum(self):
        """Update curriculum difficulty based on performance"""
        
        if not self.gym_config.curriculum_learning:
            return
        
        # Get recent performance metrics
        env_stats = self.gym.trajectory_api.get_environment_stats()
        
        for env_id, env in self.gym.environments.items():
            metrics = env_stats["environment_metrics"].get(env_id, {})
            success_rate = metrics.get("success_rate", 0.0)
            
            # Increase difficulty if success rate is high
            if success_rate > self.gym_config.target_success_rate:
                await self._increase_environment_difficulty(env_id)
            
            # Decrease difficulty if success rate is very low
            elif success_rate < 0.2:
                await self._decrease_environment_difficulty(env_id)
    
    async def _increase_environment_difficulty(self, env_id: str):
        """Increase difficulty for an environment"""
        
        environment = self.gym.environments[env_id]
        
        if environment.env_type == EnvironmentType.MATH:
            # Increase math problem complexity
            current_steps = environment.config.get("max_steps", 20)
            environment.config["max_steps"] = min(50, current_steps + 5)
            
        elif environment.env_type == EnvironmentType.TOOLCALL:
            # Add more tools or increase max calls
            current_calls = environment.config.get("max_tool_calls", 5)
            environment.config["max_tool_calls"] = min(15, current_calls + 2)
        
        logger.info(f"Increased difficulty for environment {env_id}")
    
    async def _decrease_environment_difficulty(self, env_id: str):
        """Decrease difficulty for an environment"""
        
        environment = self.gym.environments[env_id]
        
        if environment.env_type == EnvironmentType.MATH:
            # Decrease math problem complexity
            current_steps = environment.config.get("max_steps", 20)
            environment.config["max_steps"] = max(10, current_steps - 3)
            
        elif environment.env_type == EnvironmentType.TOOLCALL:
            # Reduce max tool calls
            current_calls = environment.config.get("max_tool_calls", 5)
            environment.config["max_tool_calls"] = max(3, current_calls - 1)
        
        logger.info(f"Decreased difficulty for environment {env_id}")
    
    def _update_integration_metrics(self, trajectories: List[Trajectory], training_metrics: Dict[str, Any]):
        """Update integration-specific metrics"""
        
        self.integration_metrics["training_iterations"] += 1
        self.integration_metrics["episodes_per_iteration"].append(len(trajectories))
        
        # Calculate metrics per environment type
        env_rewards = {}
        env_successes = {}
        
        for trajectory in trajectories:
            env_name = trajectory.environment_name
            
            if env_name not in env_rewards:
                env_rewards[env_name] = []
                env_successes[env_name] = []
            
            env_rewards[env_name].append(trajectory.total_reward)
            env_successes[env_name].append(trajectory.success)
        
        # Update average rewards and success rates
        for env_name in env_rewards:
            self.integration_metrics["average_reward_per_env"][env_name] = np.mean(env_rewards[env_name])
            self.integration_metrics["success_rate_per_env"][env_name] = np.mean(env_successes[env_name])
    
    async def _log_training_progress(self, iteration: int, trajectories: List[Trajectory], training_metrics: Dict[str, Any]):
        """Log comprehensive training progress"""
        
        total_reward = sum(t.total_reward for t in trajectories)
        avg_reward = total_reward / len(trajectories) if trajectories else 0
        success_rate = sum(t.success for t in trajectories) / len(trajectories) if trajectories else 0
        
        logger.info(f"Training Iteration {iteration}:")
        logger.info(f"  Episodes: {len(trajectories)}")
        logger.info(f"  Average Reward: {avg_reward:.3f}")
        logger.info(f"  Success Rate: {success_rate:.3f}")
        logger.info(f"  Policy Loss: {training_metrics.get('policy_loss', 0):.6f}")
        logger.info(f"  Value Loss: {training_metrics.get('value_loss', 0):.6f}")
        
        # Log per-environment metrics
        for env_name in self.integration_metrics["success_rate_per_env"]:
            env_success = self.integration_metrics["success_rate_per_env"][env_name]
            env_reward = self.integration_metrics["average_reward_per_env"][env_name]
            logger.info(f"  {env_name}: Success={env_success:.3f}, Reward={env_reward:.3f}")
    
    def _should_stop_training(self) -> bool:
        """Determine if training should stop early"""
        
        # Check if all environments have reached target success rate
        success_rates = list(self.integration_metrics["success_rate_per_env"].values())
        
        if len(success_rates) == 0:
            return False
        
        avg_success_rate = np.mean(success_rates)
        return avg_success_rate >= self.gym_config.target_success_rate
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        
        gym_stats = self.gym.get_gym_stats()
        
        return {
            "integration_metrics": self.integration_metrics,
            "gym_stats": gym_stats,
            "ppo_metrics": self.ppo_trainer.metrics if self.ppo_trainer else {},
            "total_episodes_collected": self.total_episodes_collected,
            "training_iterations": self.training_iteration
        }

# Factory function for easy setup
async def create_integrated_rl_system(
    gym_config: GymTrainingConfig = None,
    ppo_config: PPOConfig = None
) -> GymPPOIntegration:
    """Create and initialize the complete integrated RL system"""
    
    if gym_config is None:
        gym_config = GymTrainingConfig()
    
    if ppo_config is None:
        from .ppo_trainer import PPOConfig
        ppo_config = PPOConfig(
            total_episodes=1000,
            batch_size=32,
            learning_rate=3e-4,
            model_name="microsoft/DialoGPT-small"
        )
    
    # Create integration
    integration = GymPPOIntegration(gym_config, ppo_config)
    await integration.initialize()
    
    return integration

# Example usage
async def example_integrated_training():
    """Example of running the complete integrated RL system"""
    
    print("🎮 LoomOS Gym-PPO Integration Demo")
    print("=" * 60)
    
    # Create integrated system
    print("\n🚀 Initializing integrated RL system...")
    integration = await create_integrated_rl_system()
    
    # Run training
    print("\n🎓 Starting integrated training...")
    final_metrics = await integration.run_training_loop(max_iterations=5)
    
    # Show results
    print("\n📊 Final Training Results:")
    stats = integration.get_integration_stats()
    
    print(f"  Training Iterations: {stats['training_iterations']}")
    print(f"  Total Episodes: {stats['total_episodes_collected']}")
    print(f"  Average Success Rate: {np.mean(list(stats['integration_metrics']['success_rate_per_env'].values())):.3f}")
    
    print("\n✅ Integrated training demo complete!")

if __name__ == "__main__":
    asyncio.run(example_integrated_training())