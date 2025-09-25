"""
LoomOS RL Gym - Comprehensive Reinforcement Learning Environment System

Architecture inspired by Atropos with LoomOS-specific enhancements:
- Multiple specialized RL environments (Math, Game, ToolCall, Code, etc.)
- Central Trajectory API for experience collection
- Integration with RL Trainer for policy updates
- Connection to Inference Engine for rollouts
- Distributed environment scaling
- Real-time performance monitoring
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from collections import deque, namedtuple

logger = logging.getLogger(__name__)

# Experience and trajectory data structures
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'log_prob', 'value', 'advantage', 'timestamp', 'env_id'
])

@dataclass
class Trajectory:
    """A complete trajectory from an environment episode"""
    trajectory_id: str
    environment_name: str
    episode_id: str
    
    # Experience data
    states: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    
    # Episode metadata
    total_reward: float = 0.0
    episode_length: int = 0
    success: bool = False
    completion_time: float = 0.0
    
    # Environment-specific data
    environment_info: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    
    def add_step(self, state, action, reward, log_prob, value, info=None):
        """Add a single step to the trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        self.total_reward += reward
        self.episode_length += 1
        
        if info:
            self.environment_info.update(info)
    
    def finalize(self, success: bool = False):
        """Mark trajectory as complete"""
        self.end_time = datetime.now(timezone.utc)
        self.success = success
        if self.start_time and self.end_time:
            self.completion_time = (self.end_time - self.start_time).total_seconds()

class EnvironmentType(Enum):
    """Types of RL environments in LoomOS Gym"""
    MATH = "math"
    GAME = "game"
    TOOLCALL = "toolcall"
    CODE = "code"
    LANGUAGE = "language"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

class EnvironmentStatus(Enum):
    """Status of environment instances"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"

# Base Environment Interface
class LoomEnvironment(ABC):
    """Base class for all LoomOS RL environments"""
    
    def __init__(self, env_id: str, env_type: EnvironmentType, config: Dict[str, Any] = None):
        self.env_id = env_id
        self.env_type = env_type
        self.config = config or {}
        self.status = EnvironmentStatus.INITIALIZING
        
        # Environment state
        self.current_episode = 0
        self.total_steps = 0
        self.metrics = {
            "total_episodes": 0,
            "total_reward": 0.0,
            "average_reward": 0.0,
            "success_rate": 0.0,
            "average_episode_length": 0.0
        }
        
        logger.info(f"Initializing {env_type.value} environment: {env_id}")
    
    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment and return initial state"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Execute action and return (next_state, reward, done, info)"""
        pass
    
    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """Return the action space"""
        pass
    
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """Return the observation space"""
        pass
    
    def get_status(self) -> EnvironmentStatus:
        """Get current environment status"""
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get environment performance metrics"""
        return self.metrics.copy()

# Specialized Environment Implementations

class MathEnvironment(LoomEnvironment):
    """Mathematical reasoning and problem solving environment"""
    
    def __init__(self, env_id: str, config: Dict[str, Any] = None):
        super().__init__(env_id, EnvironmentType.MATH, config)
        
        # Math-specific configuration
        self.difficulty_levels = config.get("difficulty_levels", ["easy", "medium", "hard"])
        self.problem_types = config.get("problem_types", ["algebra", "calculus", "geometry", "statistics"])
        self.max_steps = config.get("max_steps", 50)
        
        # Current problem state
        self.current_problem = None
        self.current_solution_steps = []
        self.current_step = 0
        
        # Action and observation spaces
        self.action_space = spaces.Text(max_length=200)  # Mathematical expressions/steps
        self.observation_space = spaces.Text(max_length=1000)  # Problem statement + context
        
        self.status = EnvironmentStatus.READY
    
    def reset(self) -> str:
        """Generate a new math problem"""
        self.current_episode += 1
        self.current_step = 0
        self.current_solution_steps = []
        
        # Generate problem based on configuration
        difficulty = np.random.choice(self.difficulty_levels)
        problem_type = np.random.choice(self.problem_types)
        
        self.current_problem = self._generate_math_problem(difficulty, problem_type)
        
        observation = f"Problem: {self.current_problem['statement']}\n"
        observation += f"Type: {problem_type}, Difficulty: {difficulty}\n"
        observation += "Solve step by step. Current step: 1"
        
        return observation
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Process a solution step"""
        self.current_step += 1
        self.total_steps += 1
        self.current_solution_steps.append(action)
        
        # Evaluate the step
        step_score = self._evaluate_step(action)
        
        # Check if solution is complete
        is_correct, is_complete = self._check_solution_completion()
        
        # Calculate reward
        reward = self._calculate_reward(step_score, is_correct, is_complete)
        
        # Determine if episode is done
        done = is_complete or self.current_step >= self.max_steps
        
        # Prepare next observation
        if done:
            observation = f"Solution complete. Correct: {is_correct}"
            self._update_metrics(reward, is_correct)
        else:
            observation = f"Problem: {self.current_problem['statement']}\n"
            observation += f"Previous steps: {' -> '.join(self.current_solution_steps)}\n"
            observation += f"Current step: {self.current_step + 1}"
        
        info = {
            "step_score": step_score,
            "is_correct": is_correct,
            "is_complete": is_complete,
            "problem_type": self.current_problem.get("type", "unknown"),
            "difficulty": self.current_problem.get("difficulty", "unknown")
        }
        
        return observation, reward, done, info
    
    def get_action_space(self) -> spaces.Space:
        return self.action_space
    
    def get_observation_space(self) -> spaces.Space:
        return self.observation_space
    
    def _generate_math_problem(self, difficulty: str, problem_type: str) -> Dict[str, Any]:
        """Generate a math problem based on type and difficulty"""
        problems = {
            ("algebra", "easy"): {
                "statement": "Solve for x: 2x + 5 = 13",
                "solution": "x = 4",
                "steps": ["2x + 5 = 13", "2x = 13 - 5", "2x = 8", "x = 4"]
            },
            ("calculus", "medium"): {
                "statement": "Find the derivative of f(x) = x² + 3x + 2",
                "solution": "f'(x) = 2x + 3",
                "steps": ["f(x) = x² + 3x + 2", "f'(x) = d/dx(x²) + d/dx(3x) + d/dx(2)", "f'(x) = 2x + 3 + 0", "f'(x) = 2x + 3"]
            }
        }
        
        key = (problem_type, difficulty)
        if key in problems:
            problem = problems[key].copy()
            problem["type"] = problem_type
            problem["difficulty"] = difficulty
            return problem
        
        # Default problem
        return {
            "statement": f"Solve this {problem_type} problem (difficulty: {difficulty})",
            "solution": "Solution varies",
            "steps": ["Step 1", "Step 2", "Final answer"],
            "type": problem_type,
            "difficulty": difficulty
        }
    
    def _evaluate_step(self, action: str) -> float:
        """Evaluate the quality of a solution step"""
        # Simplified evaluation - in practice, use more sophisticated math parsing
        if not action.strip():
            return 0.0
        
        # Check for mathematical symbols, proper formatting, etc.
        math_indicators = ['=', '+', '-', '*', '/', 'x', 'y', 'dx', 'dy']
        score = sum(1 for indicator in math_indicators if indicator in action) / len(math_indicators)
        
        return min(score, 1.0)
    
    def _check_solution_completion(self) -> Tuple[bool, bool]:
        """Check if the solution is complete and correct"""
        if not self.current_solution_steps:
            return False, False
        
        # Simplified check - look for final answer format
        last_step = self.current_solution_steps[-1].lower()
        
        is_complete = any(phrase in last_step for phrase in ['answer:', 'solution:', 'x =', 'y =', 'result:'])
        
        # For demo purposes, mark as correct if solution follows logical pattern
        is_correct = is_complete and len(self.current_solution_steps) >= 2
        
        return is_correct, is_complete
    
    def _calculate_reward(self, step_score: float, is_correct: bool, is_complete: bool) -> float:
        """Calculate reward for the current step"""
        reward = step_score * 0.1  # Small reward for each valid step
        
        if is_complete:
            if is_correct:
                reward += 10.0  # Large reward for correct solution
            else:
                reward += 1.0   # Small reward for completing attempt
        
        return reward
    
    def _update_metrics(self, reward: float, is_correct: bool):
        """Update environment metrics"""
        self.metrics["total_episodes"] += 1
        self.metrics["total_reward"] += reward
        self.metrics["average_reward"] = self.metrics["total_reward"] / self.metrics["total_episodes"]
        
        if is_correct:
            self.metrics["success_rate"] = (self.metrics["success_rate"] * (self.metrics["total_episodes"] - 1) + 1) / self.metrics["total_episodes"]
        else:
            self.metrics["success_rate"] = (self.metrics["success_rate"] * (self.metrics["total_episodes"] - 1)) / self.metrics["total_episodes"]
        
        self.metrics["average_episode_length"] = (self.metrics["average_episode_length"] * (self.metrics["total_episodes"] - 1) + self.current_step) / self.metrics["total_episodes"]

class GameEnvironment(LoomEnvironment):
    """Gaming environment for strategic decision making"""
    
    def __init__(self, env_id: str, config: Dict[str, Any] = None):
        super().__init__(env_id, EnvironmentType.GAME, config)
        
        self.game_type = config.get("game_type", "tic_tac_toe")
        self.max_moves = config.get("max_moves", 9)
        
        # Game state
        self.board = None
        self.current_player = 1
        self.move_count = 0
        
        # Spaces depend on game type
        if self.game_type == "tic_tac_toe":
            self.action_space = spaces.Discrete(9)  # 9 positions
            self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        
        self.status = EnvironmentStatus.READY
    
    def reset(self) -> np.ndarray:
        """Reset game to initial state"""
        self.current_episode += 1
        self.move_count = 0
        self.current_player = 1
        
        if self.game_type == "tic_tac_toe":
            self.board = np.zeros((3, 3), dtype=np.int8)
        
        return self.board.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Make a move in the game"""
        self.move_count += 1
        self.total_steps += 1
        
        # Execute move
        reward, done, info = self._execute_move(action)
        
        # Switch players
        self.current_player *= -1
        
        if done:
            self._update_metrics(reward, info.get("winner") == 1)
        
        return self.board.copy(), reward, done, info
    
    def _execute_move(self, action: int) -> Tuple[float, bool, Dict[str, Any]]:
        """Execute a tic-tac-toe move"""
        row, col = action // 3, action % 3
        
        # Check if move is valid
        if self.board[row, col] != 0:
            return -10.0, True, {"winner": -1, "reason": "invalid_move"}
        
        # Make move
        self.board[row, col] = self.current_player
        
        # Check for win
        winner = self._check_winner()
        if winner != 0:
            reward = 10.0 if winner == 1 else -10.0
            return reward, True, {"winner": winner, "reason": "game_won"}
        
        # Check for draw
        if self.move_count >= self.max_moves:
            return 0.0, True, {"winner": 0, "reason": "draw"}
        
        # Game continues
        return 0.1, False, {"winner": 0, "reason": "ongoing"}
    
    def _check_winner(self) -> int:
        """Check for tic-tac-toe winner"""
        # Check rows
        for row in self.board:
            if abs(sum(row)) == 3:
                return row[0]
        
        # Check columns
        for col in range(3):
            if abs(sum(self.board[:, col])) == 3:
                return self.board[0, col]
        
        # Check diagonals
        if abs(sum(self.board.diagonal())) == 3:
            return self.board[0, 0]
        
        if abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return self.board[0, 2]
        
        return 0
    
    def get_action_space(self) -> spaces.Space:
        return self.action_space
    
    def get_observation_space(self) -> spaces.Space:
        return self.observation_space
    
    def _update_metrics(self, reward: float, won: bool):
        """Update game metrics"""
        self.metrics["total_episodes"] += 1
        self.metrics["total_reward"] += reward
        self.metrics["average_reward"] = self.metrics["total_reward"] / self.metrics["total_episodes"]
        
        if won:
            self.metrics["success_rate"] = (self.metrics["success_rate"] * (self.metrics["total_episodes"] - 1) + 1) / self.metrics["total_episodes"]
        else:
            self.metrics["success_rate"] = (self.metrics["success_rate"] * (self.metrics["total_episodes"] - 1)) / self.metrics["total_episodes"]

class ToolCallEnvironment(LoomEnvironment):
    """Environment for learning to use tools and APIs effectively"""
    
    def __init__(self, env_id: str, config: Dict[str, Any] = None):
        super().__init__(env_id, EnvironmentType.TOOLCALL, config)
        
        # Available tools
        self.available_tools = config.get("tools", [
            "calculator", "web_search", "file_reader", "code_executor", "database_query"
        ])
        
        self.max_tool_calls = config.get("max_tool_calls", 10)
        
        # Current task state
        self.current_task = None
        self.tool_calls_made = []
        self.task_progress = 0.0
        
        # Action space: tool selection + parameters
        self.action_space = spaces.Dict({
            "tool": spaces.Discrete(len(self.available_tools)),
            "parameters": spaces.Text(max_length=500)
        })
        
        self.observation_space = spaces.Text(max_length=1000)
        self.status = EnvironmentStatus.READY
    
    def reset(self) -> str:
        """Start a new tool-use task"""
        self.current_episode += 1
        self.tool_calls_made = []
        self.task_progress = 0.0
        
        # Generate a task requiring tool use
        self.current_task = self._generate_task()
        
        observation = f"Task: {self.current_task['description']}\n"
        observation += f"Available tools: {', '.join(self.available_tools)}\n"
        observation += "What tool would you like to use first?"
        
        return observation
    
    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute a tool call"""
        self.total_steps += 1
        
        tool_name = self.available_tools[action["tool"]]
        parameters = action["parameters"]
        
        # Execute tool call
        result = self._execute_tool_call(tool_name, parameters)
        self.tool_calls_made.append({
            "tool": tool_name,
            "parameters": parameters,
            "result": result
        })
        
        # Evaluate progress
        reward, task_complete = self._evaluate_progress(tool_name, parameters, result)
        
        # Check if done
        done = task_complete or len(self.tool_calls_made) >= self.max_tool_calls
        
        # Prepare observation
        if done:
            observation = f"Task complete! Success: {task_complete}\n"
            observation += f"Tools used: {[call['tool'] for call in self.tool_calls_made]}"
            self._update_metrics(reward, task_complete)
        else:
            observation = f"Tool result: {result}\n"
            observation += f"Task progress: {self.task_progress:.2f}\n"
            observation += "What's your next action?"
        
        info = {
            "tool_used": tool_name,
            "result": result,
            "progress": self.task_progress,
            "task_complete": task_complete
        }
        
        return observation, reward, done, info
    
    def _generate_task(self) -> Dict[str, Any]:
        """Generate a task requiring tool use"""
        tasks = [
            {
                "description": "Calculate the compound interest on $1000 at 5% annual rate for 3 years",
                "required_tools": ["calculator"],
                "success_criteria": "final_amount_calculated"
            },
            {
                "description": "Find the current weather in New York and calculate if it's suitable for outdoor activities",
                "required_tools": ["web_search", "calculator"],
                "success_criteria": "weather_decision_made"
            },
            {
                "description": "Read a data file and calculate summary statistics",
                "required_tools": ["file_reader", "calculator"],
                "success_criteria": "statistics_calculated"
            }
        ]
        
        return np.random.choice(tasks)
    
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """Simulate tool execution"""
        tool_results = {
            "calculator": f"Calculation result: {np.random.randint(100, 1000)}",
            "web_search": f"Search results for '{parameters}': Found 5 relevant results",
            "file_reader": f"File content: Sample data with {np.random.randint(10, 100)} rows",
            "code_executor": f"Code executed successfully. Output: {np.random.randint(1, 10)}",
            "database_query": f"Query returned {np.random.randint(1, 50)} records"
        }
        
        return tool_results.get(tool_name, "Tool not available")
    
    def _evaluate_progress(self, tool_name: str, parameters: str, result: str) -> Tuple[float, bool]:
        """Evaluate task progress and calculate reward"""
        # Reward for using appropriate tools
        reward = 0.0
        
        if tool_name in self.current_task["required_tools"]:
            reward += 2.0  # Reward for using correct tool
            self.task_progress += 0.3
        else:
            reward += 0.5  # Small reward for any tool use
            self.task_progress += 0.1
        
        # Check task completion
        task_complete = self.task_progress >= 1.0 or all(
            tool in [call["tool"] for call in self.tool_calls_made] 
            for tool in self.current_task["required_tools"]
        )
        
        if task_complete:
            reward += 5.0  # Completion bonus
        
        return reward, task_complete
    
    def get_action_space(self) -> spaces.Space:
        return self.action_space
    
    def get_observation_space(self) -> spaces.Space:
        return self.observation_space
    
    def _update_metrics(self, reward: float, success: bool):
        """Update tool-use metrics"""
        self.metrics["total_episodes"] += 1
        self.metrics["total_reward"] += reward
        self.metrics["average_reward"] = self.metrics["total_reward"] / self.metrics["total_episodes"]
        
        if success:
            self.metrics["success_rate"] = (self.metrics["success_rate"] * (self.metrics["total_episodes"] - 1) + 1) / self.metrics["total_episodes"]
        else:
            self.metrics["success_rate"] = (self.metrics["success_rate"] * (self.metrics["total_episodes"] - 1)) / self.metrics["total_episodes"]

# Trajectory API - Central coordination system
class TrajectoryAPI:
    """Central API for managing trajectories across all environments"""
    
    def __init__(self):
        self.active_trajectories: Dict[str, Trajectory] = {}
        self.completed_trajectories: List[Trajectory] = []
        self.environment_pool: Dict[str, LoomEnvironment] = {}
        
        # Performance tracking
        self.metrics = {
            "total_trajectories": 0,
            "active_environments": 0,
            "average_episode_length": 0.0,
            "success_rate_by_env": {},
            "total_steps": 0
        }
        
        logger.info("Trajectory API initialized")
    
    def register_environment(self, environment: LoomEnvironment):
        """Register an environment with the trajectory API"""
        self.environment_pool[environment.env_id] = environment
        self.metrics["active_environments"] = len(self.environment_pool)
        logger.info(f"Registered environment: {environment.env_id} ({environment.env_type.value})")
    
    def start_trajectory(self, env_id: str) -> str:
        """Start a new trajectory in the specified environment"""
        if env_id not in self.environment_pool:
            raise ValueError(f"Environment {env_id} not registered")
        
        environment = self.environment_pool[env_id]
        trajectory_id = f"traj_{env_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        episode_id = f"ep_{environment.current_episode + 1}"
        
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            environment_name=env_id,
            episode_id=episode_id
        )
        
        self.active_trajectories[trajectory_id] = trajectory
        self.metrics["total_trajectories"] += 1
        
        logger.info(f"Started trajectory {trajectory_id} in environment {env_id}")
        return trajectory_id
    
    def add_experience(self, trajectory_id: str, state, action, reward, log_prob, value, info=None):
        """Add an experience step to a trajectory"""
        if trajectory_id not in self.active_trajectories:
            raise ValueError(f"Trajectory {trajectory_id} not found")
        
        trajectory = self.active_trajectories[trajectory_id]
        trajectory.add_step(state, action, reward, log_prob, value, info)
        self.metrics["total_steps"] += 1
    
    def complete_trajectory(self, trajectory_id: str, success: bool = False) -> Trajectory:
        """Mark a trajectory as complete and move to completed list"""
        if trajectory_id not in self.active_trajectories:
            raise ValueError(f"Trajectory {trajectory_id} not found")
        
        trajectory = self.active_trajectories[trajectory_id]
        trajectory.finalize(success)
        
        # Move to completed trajectories
        self.completed_trajectories.append(trajectory)
        del self.active_trajectories[trajectory_id]
        
        # Update metrics
        self._update_trajectory_metrics(trajectory)
        
        logger.info(f"Completed trajectory {trajectory_id}: success={success}, reward={trajectory.total_reward:.2f}")
        return trajectory
    
    def get_trajectories_for_training(self, min_trajectories: int = 10) -> List[Trajectory]:
        """Get trajectories for RL training"""
        if len(self.completed_trajectories) < min_trajectories:
            return []
        
        # Return recent trajectories
        return self.completed_trajectories[-min_trajectories:]
    
    def query_trajectories(self, 
                          env_type: Optional[EnvironmentType] = None,
                          min_reward: Optional[float] = None,
                          success_only: bool = False,
                          limit: int = 100) -> List[Trajectory]:
        """Query trajectories with filters"""
        filtered = self.completed_trajectories
        
        if env_type:
            env_names = [env_id for env_id, env in self.environment_pool.items() 
                        if env.env_type == env_type]
            filtered = [t for t in filtered if t.environment_name in env_names]
        
        if min_reward is not None:
            filtered = [t for t in filtered if t.total_reward >= min_reward]
        
        if success_only:
            filtered = [t for t in filtered if t.success]
        
        return filtered[-limit:]
    
    def get_environment_stats(self) -> Dict[str, Any]:
        """Get comprehensive environment statistics"""
        stats = {
            "total_environments": len(self.environment_pool),
            "environment_types": {},
            "environment_metrics": {}
        }
        
        # Aggregate by environment type
        for env_id, env in self.environment_pool.items():
            env_type = env.env_type.value
            if env_type not in stats["environment_types"]:
                stats["environment_types"][env_type] = 0
            stats["environment_types"][env_type] += 1
            
            stats["environment_metrics"][env_id] = env.get_metrics()
        
        # Overall trajectory stats
        stats["trajectory_stats"] = self.metrics.copy()
        
        return stats
    
    def _update_trajectory_metrics(self, trajectory: Trajectory):
        """Update overall trajectory metrics"""
        total_trajs = len(self.completed_trajectories)
        
        # Update average episode length
        self.metrics["average_episode_length"] = (
            (self.metrics["average_episode_length"] * (total_trajs - 1) + trajectory.episode_length) 
            / total_trajs
        )
        
        # Update success rate by environment
        env_name = trajectory.environment_name
        if env_name not in self.metrics["success_rate_by_env"]:
            self.metrics["success_rate_by_env"][env_name] = {"successes": 0, "total": 0}
        
        self.metrics["success_rate_by_env"][env_name]["total"] += 1
        if trajectory.success:
            self.metrics["success_rate_by_env"][env_name]["successes"] += 1

# Environment Factory
class EnvironmentFactory:
    """Factory for creating and managing RL environments"""
    
    @staticmethod
    def create_environment(env_type: EnvironmentType, env_id: str, config: Dict[str, Any] = None) -> LoomEnvironment:
        """Create an environment of the specified type"""
        
        if env_type == EnvironmentType.MATH:
            return MathEnvironment(env_id, config)
        elif env_type == EnvironmentType.GAME:
            return GameEnvironment(env_id, config)
        elif env_type == EnvironmentType.TOOLCALL:
            return ToolCallEnvironment(env_id, config)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
    
    @staticmethod
    def create_environment_suite(config: Dict[str, Any] = None) -> List[LoomEnvironment]:
        """Create a complete suite of environments"""
        environments = []
        
        # Math environments
        math_configs = [
            {"difficulty_levels": ["easy"], "problem_types": ["algebra"]},
            {"difficulty_levels": ["medium"], "problem_types": ["calculus"]},
            {"difficulty_levels": ["hard"], "problem_types": ["geometry"]}
        ]
        
        for i, math_config in enumerate(math_configs):
            env = EnvironmentFactory.create_environment(
                EnvironmentType.MATH, 
                f"math_env_{i}", 
                math_config
            )
            environments.append(env)
        
        # Game environments
        game_configs = [
            {"game_type": "tic_tac_toe", "max_moves": 9}
        ]
        
        for i, game_config in enumerate(game_configs):
            env = EnvironmentFactory.create_environment(
                EnvironmentType.GAME,
                f"game_env_{i}",
                game_config
            )
            environments.append(env)
        
        # ToolCall environments
        tool_configs = [
            {"tools": ["calculator", "web_search"], "max_tool_calls": 5},
            {"tools": ["file_reader", "code_executor"], "max_tool_calls": 8}
        ]
        
        for i, tool_config in enumerate(tool_configs):
            env = EnvironmentFactory.create_environment(
                EnvironmentType.TOOLCALL,
                f"tool_env_{i}",
                tool_config
            )
            environments.append(env)
        
        return environments

# Main LoomOS RL Gym class
class LoomRLGym:
    """Main LoomOS RL Gym system - coordinates all environments and training"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.trajectory_api = TrajectoryAPI()
        self.environments = {}
        
        # Integration points
        self.rl_trainer = None  # Will be connected to rl.ppo_trainer
        self.inference_engine = None  # Will be connected to inference system
        
        # Performance monitoring
        self.gym_metrics = {
            "total_episodes": 0,
            "total_steps": 0,
            "training_sessions": 0,
            "average_performance": 0.0
        }
        
        logger.info("LoomOS RL Gym initialized")
    
    def initialize_environments(self, env_configs: List[Dict[str, Any]] = None):
        """Initialize all environments"""
        if env_configs is None:
            # Create default environment suite
            environments = EnvironmentFactory.create_environment_suite()
        else:
            environments = []
            for config in env_configs:
                env = EnvironmentFactory.create_environment(
                    EnvironmentType(config["type"]),
                    config["id"], 
                    config.get("config", {})
                )
                environments.append(env)
        
        # Register all environments
        for env in environments:
            self.environments[env.env_id] = env
            self.trajectory_api.register_environment(env)
        
        logger.info(f"Initialized {len(environments)} environments")
    
    def connect_rl_trainer(self, rl_trainer):
        """Connect to the LoomOS RL trainer"""
        self.rl_trainer = rl_trainer
        logger.info("Connected RL trainer to gym")
    
    def connect_inference_engine(self, inference_engine):
        """Connect to the LoomOS inference engine"""
        self.inference_engine = inference_engine
        logger.info("Connected inference engine to gym")
    
    async def run_episode(self, env_id: str, policy=None) -> Trajectory:
        """Run a complete episode in an environment"""
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        environment = self.environments[env_id]
        trajectory_id = self.trajectory_api.start_trajectory(env_id)
        
        # Reset environment
        state = environment.reset()
        done = False
        
        while not done:
            # Get action from policy (or random if no policy)
            if policy:
                action, log_prob, value = policy.get_action(state)
            else:
                # Random action for testing
                action = environment.get_action_space().sample()
                log_prob = 0.0
                value = 0.0
            
            # Execute action
            next_state, reward, done, info = environment.step(action)
            
            # Add to trajectory
            self.trajectory_api.add_experience(
                trajectory_id, state, action, reward, log_prob, value, info
            )
            
            state = next_state
        
        # Complete trajectory
        success = info.get("winner") == 1 if "winner" in info else reward > 0
        trajectory = self.trajectory_api.complete_trajectory(trajectory_id, success)
        
        self.gym_metrics["total_episodes"] += 1
        self.gym_metrics["total_steps"] += trajectory.episode_length
        
        return trajectory
    
    async def collect_training_data(self, num_episodes: int = 100) -> List[Trajectory]:
        """Collect training data from all environments"""
        trajectories = []
        
        for _ in range(num_episodes):
            # Distribute episodes across environments
            env_id = np.random.choice(list(self.environments.keys()))
            trajectory = await self.run_episode(env_id)
            trajectories.append(trajectory)
        
        logger.info(f"Collected {len(trajectories)} trajectories for training")
        return trajectories
    
    def update_inference_weights(self, model_weights):
        """Update inference engine with new model weights"""
        if self.inference_engine:
            self.inference_engine.update_weights(model_weights)
            logger.info("Updated inference engine weights")
    
    def query_rollouts(self, env_type: EnvironmentType = None, limit: int = 1000) -> List[Trajectory]:
        """Query rollouts for the inference engine"""
        return self.trajectory_api.query_trajectories(
            env_type=env_type,
            limit=limit
        )
    
    def get_gym_stats(self) -> Dict[str, Any]:
        """Get comprehensive gym statistics"""
        stats = {
            "gym_metrics": self.gym_metrics.copy(),
            "environment_stats": self.trajectory_api.get_environment_stats(),
            "active_environments": len(self.environments),
            "total_trajectories": len(self.trajectory_api.completed_trajectories)
        }
        
        return stats

# Example usage and integration
async def example_gym_usage():
    """Example of how to use the LoomOS RL Gym"""
    
    # Initialize gym
    gym = LoomRLGym()
    gym.initialize_environments()
    
    print("🎮 LoomOS RL Gym Demo")
    print("=" * 50)
    
    # Run some episodes
    print("\n🚀 Running sample episodes...")
    for env_id in list(gym.environments.keys())[:3]:  # Run 3 different environments
        trajectory = await gym.run_episode(env_id)
        print(f"  {env_id}: {trajectory.episode_length} steps, reward: {trajectory.total_reward:.2f}")
    
    # Collect training data
    print("\n📊 Collecting training data...")
    training_trajectories = await gym.collect_training_data(20)
    print(f"  Collected {len(training_trajectories)} trajectories")
    
    # Get statistics
    print("\n📈 Gym Statistics:")
    stats = gym.get_gym_stats()
    print(f"  Total Episodes: {stats['gym_metrics']['total_episodes']}")
    print(f"  Total Steps: {stats['gym_metrics']['total_steps']}")
    print(f"  Active Environments: {stats['active_environments']}")
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    asyncio.run(example_gym_usage())