"""
WEAVE Algorithm - Weighted Exploration, Adaptive Value Estimation

WEAVE integrates hierarchical reinforcement learning, adaptive weighting of exploration 
strategies, and distributed value estimation across nodes. Inspired by the metaphor 
of a loom weaving threads into fabric, WEAVE combines multiple "threads" of learning 
(policies, exploration modes, reward signals) into a unified model that scales 
across distributed clusters.

Mathematical Foundation:
- Weighted Exploration: π_WEAVE(a|s) = Σ w_i(s) * π_i(a|s)
- Adaptive Value Estimation: V_WEAVE(s) = (1/Z) * Σ α_j(s) * V_j(s) 
- Hierarchical Reward Shaping: R(s,a) = Σ β_h * R_h(s,a)

Author: Loom Labs
Integration: LoomOS Distributed RL System
"""

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

# Mock torch for demo - replace with real torch in production
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock implementation
    class torch:
        class Tensor:
            def __init__(self, data): self.data = data
            def to(self, device): return self
            def squeeze(self): return self
            def unsqueeze(self, dim): return self
            def mean(self): return 0.5
            def sum(self): return 1.0
            def item(self): return 0.5
            def backward(self): pass
            def detach(self): return self
        
        @staticmethod
        def zeros(*args, **kwargs): return torch.Tensor([])
        @staticmethod
        def ones(*args, **kwargs): return torch.Tensor([])
        @staticmethod
        def randn(*args, **kwargs): return torch.Tensor([])
        @staticmethod
        def tensor(data, **kwargs): return torch.Tensor(data)
        @staticmethod
        def softmax(x, dim): return torch.Tensor([0.5, 0.5])
        @staticmethod
        def stack(tensors): return torch.Tensor([])
        @staticmethod
        def cat(tensors, dim=0): return torch.Tensor([])
        
        class nn:
            class Module:
                def __init__(self): pass
                def forward(self, x): return torch.Tensor([])
                def parameters(self): return []
                def train(self): pass
                def eval(self): pass
            
            class Linear(Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.weight = torch.randn(out_features, in_features)
                    self.bias = torch.randn(out_features)
            
            class Sequential(Module):
                def __init__(self, *layers): super().__init__()
        
        class optim:
            class Adam:
                def __init__(self, params, lr=1e-3): pass
                def step(self): pass
                def zero_grad(self): pass

logger = logging.getLogger(__name__)

class ExplorationStrategy(Enum):
    """Types of exploration strategies in WEAVE"""
    ENTROPY_BASED = "entropy_based"
    CURIOSITY_DRIVEN = "curiosity_driven"
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB_EXPLORATION = "ucb_exploration"
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"

class RewardThread(Enum):
    """Types of reward threads in hierarchical shaping"""
    TASK_COMPLETION = "task_completion"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    HUMAN_FEEDBACK = "human_feedback"
    CREATIVITY = "creativity"
    COHERENCE = "coherence"
    ENGAGEMENT = "engagement"

@dataclass
class WeaveConfig:
    """Configuration for WEAVE algorithm"""
    
    # Exploration strategy configuration
    exploration_strategies: List[ExplorationStrategy] = field(default_factory=lambda: [
        ExplorationStrategy.ENTROPY_BASED,
        ExplorationStrategy.CURIOSITY_DRIVEN,
        ExplorationStrategy.UCB_EXPLORATION
    ])
    
    # Reward thread configuration
    reward_threads: List[RewardThread] = field(default_factory=lambda: [
        RewardThread.TASK_COMPLETION,
        RewardThread.SAFETY,
        RewardThread.EFFICIENCY
    ])
    
    # Distributed nodes configuration
    num_nodes: int = 8
    node_specializations: Dict[str, List[RewardThread]] = field(default_factory=dict)
    
    # Learning parameters
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    meta_lr: float = 1e-4
    
    # WEAVE-specific parameters
    exploration_weight_entropy: float = 0.01
    value_ensemble_temp: float = 1.0
    reward_thread_momentum: float = 0.9
    
    # PPO parameters
    ppo_epochs: int = 4
    ppo_clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Architecture parameters
    hidden_size: int = 256
    state_dim: int = 512
    action_dim: int = 64

class ExplorationDistribution:
    """Individual exploration distribution π_i(a|s)"""
    
    def __init__(self, strategy: ExplorationStrategy, config: WeaveConfig):
        self.strategy = strategy
        self.config = config
        
        # Strategy-specific parameters
        self.epsilon = 0.1  # for epsilon-greedy
        self.temperature = 1.0  # for Boltzmann
        self.ucb_c = 1.0  # for UCB
        
        # Internal state for adaptive strategies
        self.visit_counts = {}
        self.curiosity_network = self._build_curiosity_network()
    
    def _build_curiosity_network(self):
        """Build curiosity-driven exploration network"""
        if not TORCH_AVAILABLE:
            return None
        
        return torch.nn.Sequential(
            torch.nn.Linear(self.config.state_dim, self.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, self.config.action_dim)
        )
    
    def get_action_distribution(self, state, base_policy_logits):
        """Get exploration-modified action distribution"""
        
        if self.strategy == ExplorationStrategy.ENTROPY_BASED:
            return self._entropy_exploration(base_policy_logits)
        
        elif self.strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            return self._curiosity_exploration(state, base_policy_logits)
        
        elif self.strategy == ExplorationStrategy.UCB_EXPLORATION:
            return self._ucb_exploration(state, base_policy_logits)
        
        elif self.strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_exploration(base_policy_logits)
        
        elif self.strategy == ExplorationStrategy.BOLTZMANN:
            return self._boltzmann_exploration(base_policy_logits)
        
        else:
            # Default: return base policy
            return torch.softmax(base_policy_logits, dim=-1)
    
    def _entropy_exploration(self, logits):
        """Entropy-regularized exploration"""
        # Add entropy bonus to encourage exploration
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        entropy_bonus = self.config.exploration_weight_entropy * entropy
        
        # Reshape entropy bonus to match logits
        if len(logits.shape) > 1:
            entropy_bonus = entropy_bonus.expand_as(logits)
        
        modified_logits = logits + entropy_bonus
        return torch.softmax(modified_logits, dim=-1)
    
    def _curiosity_exploration(self, state, logits):
        """Curiosity-driven exploration using intrinsic motivation"""
        if self.curiosity_network is None:
            return torch.softmax(logits, dim=-1)
        
        # Compute curiosity bonus (simplified)
        curiosity_logits = self.curiosity_network(state)
        curiosity_bonus = 0.1 * curiosity_logits
        
        modified_logits = logits + curiosity_bonus
        return torch.softmax(modified_logits, dim=-1)
    
    def _ucb_exploration(self, state, logits):
        """Upper Confidence Bound exploration"""
        # Simplified UCB using visit counts
        state_key = str(state.detach().numpy() if TORCH_AVAILABLE else state)
        
        if state_key not in self.visit_counts:
            self.visit_counts[state_key] = torch.ones_like(logits)
        
        visit_counts = self.visit_counts[state_key]
        total_visits = torch.sum(visit_counts)
        
        # UCB bonus: c * sqrt(log(total_visits) / visit_counts)
        ucb_bonus = self.ucb_c * torch.sqrt(
            torch.log(total_visits + 1) / (visit_counts + 1)
        )
        
        modified_logits = logits + ucb_bonus
        return torch.softmax(modified_logits, dim=-1)
    
    def _epsilon_greedy_exploration(self, logits):
        """Epsilon-greedy exploration"""
        probs = torch.softmax(logits, dim=-1)
        
        # With probability epsilon, uniform random
        # With probability 1-epsilon, greedy
        uniform_probs = torch.ones_like(probs) / probs.shape[-1]
        
        # Mix probabilities
        mixed_probs = (1 - self.epsilon) * probs + self.epsilon * uniform_probs
        return mixed_probs
    
    def _boltzmann_exploration(self, logits):
        """Boltzmann (temperature-based) exploration"""
        scaled_logits = logits / self.temperature
        return torch.softmax(scaled_logits, dim=-1)

class RewardThreadProcessor:
    """Processes individual reward threads R_h(s,a)"""
    
    def __init__(self, thread_type: RewardThread, config: WeaveConfig):
        self.thread_type = thread_type
        self.config = config
        
        # Thread-specific processors
        self.safety_threshold = 0.8
        self.efficiency_baseline = 1.0
        self.creativity_network = self._build_creativity_network()
    
    def _build_creativity_network(self):
        """Build network for creativity assessment"""
        if not TORCH_AVAILABLE:
            return None
        
        return torch.nn.Sequential(
            torch.nn.Linear(self.config.state_dim + self.config.action_dim, self.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, 1)
        )
    
    def compute_reward(self, state, action, base_reward, info=None):
        """Compute thread-specific reward component"""
        
        if self.thread_type == RewardThread.TASK_COMPLETION:
            return self._task_completion_reward(base_reward, info)
        
        elif self.thread_type == RewardThread.SAFETY:
            return self._safety_reward(state, action, info)
        
        elif self.thread_type == RewardThread.EFFICIENCY:
            return self._efficiency_reward(state, action, info)
        
        elif self.thread_type == RewardThread.HUMAN_FEEDBACK:
            return self._human_feedback_reward(info)
        
        elif self.thread_type == RewardThread.CREATIVITY:
            return self._creativity_reward(state, action)
        
        elif self.thread_type == RewardThread.COHERENCE:
            return self._coherence_reward(state, action, info)
        
        elif self.thread_type == RewardThread.ENGAGEMENT:
            return self._engagement_reward(state, action, info)
        
        else:
            return 0.0
    
    def _task_completion_reward(self, base_reward, info):
        """Basic task completion reward"""
        completion_bonus = 0.0
        if info and info.get("task_complete", False):
            completion_bonus = 10.0
        return base_reward + completion_bonus
    
    def _safety_reward(self, state, action, info):
        """Safety-based reward component"""
        # Simplified safety scoring
        safety_score = info.get("safety_score", 0.5) if info else 0.5
        
        if safety_score >= self.safety_threshold:
            return 1.0  # Safety bonus
        else:
            return -5.0  # Safety penalty
    
    def _efficiency_reward(self, state, action, info):
        """Efficiency-based reward component"""
        # Reward based on action efficiency
        action_cost = info.get("action_cost", 1.0) if info else 1.0
        efficiency = self.efficiency_baseline / max(action_cost, 0.1)
        return 0.1 * efficiency
    
    def _human_feedback_reward(self, info):
        """Human feedback reward component"""
        if info and "human_rating" in info:
            # Scale human rating (1-5) to reward signal
            return (info["human_rating"] - 3.0) * 2.0  # -4 to +4 range
        return 0.0
    
    def _creativity_reward(self, state, action):
        """Creativity-based reward component"""
        if self.creativity_network is None:
            return np.random.normal(0, 0.1)  # Random creativity for demo
        
        # Concatenate state and action for creativity assessment
        if TORCH_AVAILABLE:
            state_action = torch.cat([state, action], dim=-1)
            creativity_score = self.creativity_network(state_action)
            return creativity_score.item()
        
        return 0.0
    
    def _coherence_reward(self, state, action, info):
        """Coherence-based reward component"""
        # Simplified coherence based on action consistency
        coherence_score = info.get("coherence_score", 0.0) if info else 0.0
        return 0.5 * coherence_score
    
    def _engagement_reward(self, state, action, info):
        """Engagement-based reward component"""
        # Reward based on user engagement metrics
        engagement_metrics = info.get("engagement", {}) if info else {}
        
        engagement_score = 0.0
        engagement_score += engagement_metrics.get("user_interest", 0.0) * 0.3
        engagement_score += engagement_metrics.get("interaction_quality", 0.0) * 0.4
        engagement_score += engagement_metrics.get("response_relevance", 0.0) * 0.3
        
        return engagement_score

class DistributedValueEstimator:
    """Handles distributed value estimation V_WEAVE(s) = (1/Z) * Σ α_j(s) * V_j(s)"""
    
    def __init__(self, config: WeaveConfig):
        self.config = config
        self.num_nodes = config.num_nodes
        
        # Local value networks for each node
        self.local_critics = self._build_local_critics()
        
        # Trust coefficients α_j(s) for each node
        self.trust_network = self._build_trust_network()
        
        # Node reliability tracking
        self.node_reliability = torch.ones(self.num_nodes)
        self.node_variance_history = [[] for _ in range(self.num_nodes)]
    
    def _build_local_critics(self):
        """Build local value networks for each node"""
        if not TORCH_AVAILABLE:
            return [None] * self.num_nodes
        
        critics = []
        for i in range(self.num_nodes):
            critic = torch.nn.Sequential(
                torch.nn.Linear(self.config.state_dim, self.config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config.hidden_size, 1)
            )
            critics.append(critic)
        
        return critics
    
    def _build_trust_network(self):
        """Build network to compute trust coefficients α_j(s)"""
        if not TORCH_AVAILABLE:
            return None
        
        return torch.nn.Sequential(
            torch.nn.Linear(self.config.state_dim + self.num_nodes, self.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, self.num_nodes),
            torch.nn.Softmax(dim=-1)
        )
    
    def compute_local_values(self, state):
        """Compute local value estimates V_j(s) for all nodes"""
        if not TORCH_AVAILABLE:
            return torch.ones(self.num_nodes)
        
        local_values = []
        for i, critic in enumerate(self.local_critics):
            if critic is not None:
                value = critic(state)
                local_values.append(value)
            else:
                local_values.append(torch.tensor(0.0))
        
        return torch.stack(local_values)
    
    def compute_trust_coefficients(self, state):
        """Compute trust coefficients α_j(s) based on node reliability"""
        if self.trust_network is None:
            # Uniform trust for demo
            return torch.ones(self.num_nodes) / self.num_nodes
        
        # Combine state with node reliability features
        reliability_features = self.node_reliability
        combined_input = torch.cat([state, reliability_features])
        
        trust_coeffs = self.trust_network(combined_input)
        return trust_coeffs
    
    def compute_ensemble_value(self, state):
        """Compute global value function V_WEAVE(s)"""
        # Get local value estimates
        local_values = self.compute_local_values(state)
        
        # Get trust coefficients
        trust_coeffs = self.compute_trust_coefficients(state)
        
        # Compute weighted ensemble
        ensemble_value = torch.sum(trust_coeffs * local_values.squeeze())
        
        return ensemble_value
    
    def update_node_reliability(self, node_id, prediction_error):
        """Update node reliability based on prediction accuracy"""
        # Exponential moving average of reliability
        decay = 0.99
        accuracy = max(0.1, 1.0 - abs(prediction_error))
        
        self.node_reliability[node_id] = (
            decay * self.node_reliability[node_id] + 
            (1 - decay) * accuracy
        )
        
        # Track variance for trust computation
        self.node_variance_history[node_id].append(prediction_error ** 2)
        if len(self.node_variance_history[node_id]) > 100:
            self.node_variance_history[node_id].pop(0)

class WeaveActor:
    """WEAVE Actor Network with weighted exploration strategies"""
    
    def __init__(self, config: WeaveConfig):
        self.config = config
        
        # Base policy network
        self.base_policy = self._build_base_policy()
        
        # Exploration distributions
        self.exploration_dists = [
            ExplorationDistribution(strategy, config) 
            for strategy in config.exploration_strategies
        ]
        
        # Meta-weight network for exploration strategies w_i(s)
        self.exploration_weight_network = self._build_weight_network(
            len(config.exploration_strategies)
        )
        
        # Optimizers
        if TORCH_AVAILABLE:
            self.optimizer = torch.optim.Adam(
                list(self.base_policy.parameters()) + 
                list(self.exploration_weight_network.parameters()),
                lr=config.actor_lr
            )
    
    def _build_base_policy(self):
        """Build base policy network"""
        if not TORCH_AVAILABLE:
            return None
        
        return torch.nn.Sequential(
            torch.nn.Linear(self.config.state_dim, self.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, self.config.action_dim)
        )
    
    def _build_weight_network(self, num_strategies):
        """Build network to compute exploration weights w_i(s)"""
        if not TORCH_AVAILABLE:
            return None
        
        return torch.nn.Sequential(
            torch.nn.Linear(self.config.state_dim, self.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, num_strategies),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """Forward pass: π_WEAVE(a|s) = Σ w_i(s) * π_i(a|s)"""
        if not TORCH_AVAILABLE:
            # Mock output for demo
            action_probs = np.random.softmax(np.random.randn(self.config.action_dim))
            return torch.tensor(action_probs)
        
        # Get base policy logits
        base_logits = self.base_policy(state)
        
        # Get exploration weights w_i(s)
        exploration_weights = self.exploration_weight_network(state)
        
        # Compute weighted exploration mixture
        weighted_probs = torch.zeros_like(torch.softmax(base_logits, dim=-1))
        
        for i, (weight, exploration_dist) in enumerate(zip(
            exploration_weights, self.exploration_dists
        )):
            # Get exploration-modified distribution
            exploration_probs = exploration_dist.get_action_distribution(state, base_logits)
            
            # Add weighted contribution
            weighted_probs += weight * exploration_probs
        
        return weighted_probs
    
    def get_action_and_log_prob(self, state):
        """Sample action and compute log probability"""
        action_probs = self.forward(state)
        
        if TORCH_AVAILABLE:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
        else:
            # Mock for demo
            action = np.random.choice(self.config.action_dim, p=action_probs.data)
            log_prob = np.log(action_probs.data[action] + 1e-8)
            return action, log_prob

class WeaveAlgorithm:
    """Main WEAVE Algorithm Implementation"""
    
    def __init__(self, config: WeaveConfig = None):
        self.config = config or WeaveConfig()
        
        # Core components
        self.actor = WeaveActor(self.config)
        self.value_estimator = DistributedValueEstimator(self.config)
        
        # Reward thread processors
        self.reward_processors = {
            thread: RewardThreadProcessor(thread, self.config)
            for thread in self.config.reward_threads
        }
        
        # Reward thread weights β_h
        self.reward_thread_weights = torch.ones(len(self.config.reward_threads))
        self.reward_thread_weights /= self.reward_thread_weights.sum()
        
        # Meta-optimizer for reward thread weights
        if TORCH_AVAILABLE:
            self.meta_optimizer = torch.optim.Adam([self.reward_thread_weights], lr=self.config.meta_lr)
        
        # Training state
        self.training_step = 0
        self.performance_history = []
        
        # Metrics
        self.metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "meta_loss": [],
            "exploration_weights": [],
            "reward_thread_weights": [],
            "node_trust_coefficients": [],
            "average_return": []
        }
        
        logger.info("WEAVE algorithm initialized")
    
    def compute_hierarchical_reward(self, state, action, base_reward, info=None):
        """Compute hierarchical reward: R(s,a) = Σ β_h * R_h(s,a)"""
        
        total_reward = 0.0
        thread_rewards = {}
        
        for i, (thread_type, processor) in enumerate(self.reward_processors.items()):
            # Compute thread-specific reward
            thread_reward = processor.compute_reward(state, action, base_reward, info)
            thread_rewards[thread_type] = thread_reward
            
            # Add weighted contribution
            weight = self.reward_thread_weights[i] if TORCH_AVAILABLE else 1.0 / len(self.reward_processors)
            total_reward += weight * thread_reward
        
        return total_reward, thread_rewards
    
    def update_actor(self, states, actions, advantages, old_log_probs):
        """Update actor using PPO-style clipped objective"""
        if not TORCH_AVAILABLE:
            return {"actor_loss": 0.0}
        
        # Compute current log probabilities
        action_probs = self.actor.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip_epsilon, 1 + self.config.ppo_clip_epsilon)
        
        actor_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Add entropy bonus
        entropy = dist.entropy().mean()
        actor_loss -= self.config.entropy_coef * entropy
        
        # Backward pass
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.item(),
            "ratio_mean": ratio.mean().item()
        }
    
    def update_critic(self, states, returns):
        """Update distributed value estimator"""
        if not TORCH_AVAILABLE:
            return {"critic_loss": 0.0}
        
        # Compute ensemble value estimates
        value_estimates = self.value_estimator.compute_ensemble_value(states)
        
        # Value loss
        value_loss = F.mse_loss(value_estimates, returns)
        
        # Update local critics (simplified - in practice, distribute this)
        for critic in self.value_estimator.local_critics:
            if critic is not None:
                optimizer = torch.optim.Adam(critic.parameters(), lr=self.config.critic_lr)
                optimizer.zero_grad()
                
                local_values = critic(states)
                local_loss = F.mse_loss(local_values.squeeze(), returns)
                local_loss.backward()
                optimizer.step()
        
        return {"critic_loss": value_loss.item()}
    
    def update_meta_weights(self, performance_improvement):
        """Update reward thread weights β_h using meta-gradients"""
        if not TORCH_AVAILABLE:
            return {"meta_loss": 0.0}
        
        # Meta-objective: maximize performance improvement
        meta_loss = -performance_improvement
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Renormalize weights
        self.reward_thread_weights /= self.reward_thread_weights.sum()
        
        return {"meta_loss": meta_loss.item()}
    
    def train_step(self, batch_data):
        """Single WEAVE training step"""
        
        states = batch_data["states"]
        actions = batch_data["actions"]
        rewards = batch_data["rewards"]
        old_log_probs = batch_data["log_probs"]
        dones = batch_data["dones"]
        
        # Convert to tensors if needed
        if TORCH_AVAILABLE:
            if not isinstance(states, torch.Tensor):
                states = torch.tensor(states, dtype=torch.float32)
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, dtype=torch.long)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, dtype=torch.float32)
            if not isinstance(old_log_probs, torch.Tensor):
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        
        # Compute value estimates and advantages
        with torch.no_grad() if TORCH_AVAILABLE else nullcontext():
            values = self.value_estimator.compute_ensemble_value(states)
            
            # Compute returns and advantages (simplified GAE)
            returns = rewards  # Simplified - should use proper return computation
            advantages = returns - values
            
            # Normalize advantages
            if TORCH_AVAILABLE:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update actor
        actor_metrics = self.update_actor(states, actions, advantages, old_log_probs)
        
        # Update critic
        critic_metrics = self.update_critic(states, returns)
        
        # Update meta-weights based on performance
        current_return = rewards.mean() if TORCH_AVAILABLE else np.mean(rewards)
        self.performance_history.append(current_return)
        
        if len(self.performance_history) > 1:
            performance_improvement = (
                self.performance_history[-1] - self.performance_history[-2]
            )
            meta_metrics = self.update_meta_weights(performance_improvement)
        else:
            meta_metrics = {"meta_loss": 0.0}
        
        # Update training step
        self.training_step += 1
        
        # Collect metrics
        step_metrics = {
            **actor_metrics,
            **critic_metrics,
            **meta_metrics,
            "training_step": self.training_step,
            "average_return": current_return
        }
        
        # Update stored metrics
        for key, value in step_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        return step_metrics
    
    def get_action(self, state):
        """Get action from WEAVE policy"""
        if TORCH_AVAILABLE and not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        with torch.no_grad() if TORCH_AVAILABLE else nullcontext():
            action, log_prob = self.actor.get_action_and_log_prob(state)
            value = self.value_estimator.compute_ensemble_value(state)
        
        return action, log_prob, value
    
    def get_weave_stats(self):
        """Get comprehensive WEAVE algorithm statistics"""
        
        stats = {
            "algorithm": "WEAVE",
            "training_step": self.training_step,
            "config": {
                "exploration_strategies": [s.value for s in self.config.exploration_strategies],
                "reward_threads": [t.value for t in self.config.reward_threads],
                "num_nodes": self.config.num_nodes
            },
            "current_weights": {
                "reward_threads": self.reward_thread_weights.tolist() if TORCH_AVAILABLE else [1.0] * len(self.config.reward_threads),
                "node_trust": self.value_estimator.node_reliability.tolist() if TORCH_AVAILABLE else [1.0] * self.config.num_nodes
            },
            "performance": {
                "recent_returns": self.performance_history[-10:] if self.performance_history else [],
                "average_return": np.mean(self.performance_history) if self.performance_history else 0.0
            },
            "metrics": self.metrics
        }
        
        return stats

# Helper context manager for non-torch environments
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Factory function for easy creation
def create_weave_algorithm(
    exploration_strategies=None,
    reward_threads=None,
    num_nodes=8,
    **kwargs
) -> WeaveAlgorithm:
    """Create WEAVE algorithm with specified configuration"""
    
    config = WeaveConfig(
        exploration_strategies=exploration_strategies or [
            ExplorationStrategy.ENTROPY_BASED,
            ExplorationStrategy.CURIOSITY_DRIVEN,
            ExplorationStrategy.UCB_EXPLORATION
        ],
        reward_threads=reward_threads or [
            RewardThread.TASK_COMPLETION,
            RewardThread.SAFETY,
            RewardThread.EFFICIENCY
        ],
        num_nodes=num_nodes,
        **kwargs
    )
    
    return WeaveAlgorithm(config)

# Example usage for LoomOS integration
async def example_weave_training():
    """Example of WEAVE algorithm in action"""
    
    print("🧵 WEAVE Algorithm Demo")
    print("=" * 50)
    
    # Create WEAVE algorithm
    weave = create_weave_algorithm(
        exploration_strategies=[
            ExplorationStrategy.ENTROPY_BASED,
            ExplorationStrategy.CURIOSITY_DRIVEN,
            ExplorationStrategy.UCB_EXPLORATION
        ],
        reward_threads=[
            RewardThread.TASK_COMPLETION,
            RewardThread.SAFETY,
            RewardThread.CREATIVITY,
            RewardThread.ENGAGEMENT
        ],
        num_nodes=8
    )
    
    print(f"✅ Created WEAVE with {len(weave.config.exploration_strategies)} exploration strategies")
    print(f"✅ Configured {len(weave.config.reward_threads)} reward threads")
    print(f"✅ Distributed across {weave.config.num_nodes} nodes")
    
    # Simulate training data
    batch_size = 32
    state_dim = weave.config.state_dim
    action_dim = weave.config.action_dim
    
    # Mock training batch
    batch_data = {
        "states": np.random.randn(batch_size, state_dim),
        "actions": np.random.randint(0, action_dim, batch_size),
        "rewards": np.random.randn(batch_size) * 2.0,
        "log_probs": np.random.randn(batch_size) * 0.1,
        "dones": np.random.choice([True, False], batch_size)
    }
    
    # Run training steps
    print("\n🎓 Running WEAVE training steps...")
    for step in range(5):
        metrics = weave.train_step(batch_data)
        
        print(f"  Step {step + 1}:")
        print(f"    Actor Loss: {metrics['actor_loss']:.6f}")
        print(f"    Critic Loss: {metrics['critic_loss']:.6f}")
        print(f"    Average Return: {metrics['average_return']:.3f}")
    
    # Get final statistics
    stats = weave.get_weave_stats()
    print(f"\n📊 WEAVE Statistics:")
    print(f"  Training Steps: {stats['training_step']}")
    print(f"  Exploration Strategies: {stats['config']['exploration_strategies']}")
    print(f"  Reward Threads: {stats['config']['reward_threads']}")
    print(f"  Average Performance: {stats['performance']['average_return']:.3f}")
    
    # Demonstrate action selection
    print(f"\n🎯 Action Selection Demo:")
    test_state = np.random.randn(state_dim)
    action, log_prob, value = weave.get_action(test_state)
    print(f"  Selected Action: {action}")
    print(f"  Log Probability: {log_prob:.6f}")
    print(f"  State Value: {value:.3f}")
    
    print("\n✅ WEAVE Demo Complete!")
    print("🧵 Successfully wove exploration, rewards, and values into unified policy!")

if __name__ == "__main__":
    asyncio.run(example_weave_training())